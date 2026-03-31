"""
Training loop module (optimized).

Includes epoch-level training and validation functions for HPA-UDE.

Optimizations:
1. Remove per-batch R² computation to reduce CPU-GPU sync.
2. Support bfloat16 (often more stable than float16).
3. Support gradient accumulation.
4. Use return_flux=False in validation for speed.
"""
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from losses import (
    calc_flux_boundary_loss,
    calc_mass_conservation_loss,
    calc_monotonicity_loss,
    calc_weighted_huber_loss,
)
from utils import calc_r2, call_model, get_pred_from_output, unpack_batch_simple


def _get_amp_dtype(device: torch.device, prefer_bf16: bool = True) -> torch.dtype:
    """Choose AMP dtype: prefer bfloat16, then float16."""
    if device.type != "cuda":
        return torch.float32
    if prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def pretrain_one_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler = None,
    use_amp: bool = False,
) -> float:
    """Pretraining: masked modeling reconstruction of dynamic drivers."""
    model.train()
    criterion = nn.MSELoss()
    total_loss, total_count = 0.0, 0
    loop = tqdm(loader, desc="Pretrain", leave=False)
    amp_dtype = _get_amp_dtype(device) if use_amp else torch.float32

    for batch in loop:
        x_stat, x_dyn, _ = unpack_batch_simple(batch)
        x_stat, x_dyn = x_stat.to(device), x_dyn.to(device)
        # Check NaN/Inf before forward to avoid wasted compute.
        if not torch.isfinite(x_dyn).all() or not torch.isfinite(x_stat).all():
            raise ValueError("Input contains NaN/Inf. Please check data cleaning or normalization.")
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            recon = call_model(model, x_stat, x_dyn, mode="pretrain", adjoint=False)
            recon = get_pred_from_output(recon)
            # Pretrain head reconstructs meteorological variables only.
            meteo_dim = recon.size(-1)
            loss = criterion(recon, x_dyn[..., :meteo_dim])

        if not torch.isfinite(loss):
            raise ValueError("Pretraining loss is NaN/Inf. Lower LR or inspect input data.")

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * x_stat.size(0)
        total_count += x_stat.size(0)
        loop.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(total_count, 1)


def finetune_one_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    spinup_days: int,
    adjoint: bool,
    scaler: GradScaler = None,
    use_amp: bool = False,
    lambda_mass: float = 0.001,
    lambda_flux: float = 0.001,
    accum_steps: int = 1,
    lambda_mono: float = 0.0,
) -> Tuple[float, float]:
    """
    Physics-aware finetuning: ODE prediction + mass conservation + flux bounds.

    Optimizations:
    1. bfloat16 support for better numerical stability.
    2. Remove per-batch R² to reduce sync overhead.
    3. Support gradient accumulation (accum_steps > 1).
    4. Use model-returned flux terms for physical losses.
    """
    model.train()
    total_loss, total_count, step_i = 0.0, 0, 0
    all_pred, all_obs = [], []
    loop = tqdm(loader, desc="Finetune", leave=False)
    amp_dtype = _get_amp_dtype(device) if use_amp else torch.float32
    max_grad_norm = 1.0

    # Gradient accumulation: clear grads before first step.
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loop):
        x_stat, x_dyn, y = unpack_batch_simple(batch)
        x_stat, x_dyn, y = x_stat.to(device), x_dyn.to(device), y.to(device)
        # Check NaN/Inf before forward.
        if not (torch.isfinite(x_dyn).all() and torch.isfinite(x_stat).all() and torch.isfinite(y).all()):
            raise ValueError("Input contains NaN/Inf. Please check data cleaning or normalization.")

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            # Compute flux only when physical losses are enabled.
            need_flux = lambda_mass > 0 or lambda_flux > 0
            output = model(x_stat, x_dyn, mode="finetune", adjoint=adjoint, return_flux=need_flux)

            flux_terms = None
            if isinstance(output, tuple) and len(output) >= 4:
                pred, pred_phy, theta_phy, flux_terms = output[:4]
            elif isinstance(output, tuple) and len(output) >= 3:
                pred, pred_phy, theta_phy = output[:3]
            else:
                pred = get_pred_from_output(output)
                pred_phy = pred

            pred_eval = pred[:, spinup_days:] if pred.size(1) > spinup_days else pred
            y_eval = y[:, spinup_days:] if y.size(1) > spinup_days else y

            loss_mse = calc_weighted_huber_loss(pred_eval, y_eval)
            loss_mass = loss_flux = torch.tensor(0.0, device=device)

            # Physical losses from model-computed true fluxes.
            if (lambda_mass > 0 or lambda_flux > 0) and flux_terms is not None:
                e_act_real = flux_terms[..., 0]  # (B, T)
                d_term_real = flux_terms[..., 1]  # (B, T)
                # De-normalize precipitation/PET to physical units.
                p_lag_norm, pet_lag_norm = x_dyn[..., 4], x_dyn[..., 5]
                if hasattr(model, "dyn_mean") and hasattr(model, "dyn_scale"):
                    p = p_lag_norm * model.dyn_scale[4] + model.dyn_mean[4]
                    pet = pet_lag_norm * model.dyn_scale[5] + model.dyn_mean[5]
                else:
                    p, pet = p_lag_norm, pet_lag_norm
                p, pet = F.relu(p), F.relu(pet)
                loss_mass = calc_mass_conservation_loss(pred_phy.squeeze(-1), p, e_act_real, d_term_real)
                loss_flux = calc_flux_boundary_loss(e_act_real, pet)

            # KAN monotonicity loss:
            #   ∂S_max/∂BD <= 0 and ∂K_sat/∂Sand >= 0
            loss_mono = torch.tensor(0.0, device=device)
            if lambda_mono > 0 and hasattr(model, "hypernet"):
                loss_mono = calc_monotonicity_loss(model.hypernet, x_stat)

            loss = loss_mse + lambda_mass * loss_mass + lambda_flux * loss_flux + lambda_mono * loss_mono
            loss = loss / accum_steps  # Gradient accumulation scaling

        # NaN/Inf guard
        if not torch.isfinite(loss):
            loop.write("  ⚠️ Skipping batch: loss is NaN/Inf")
            continue

        # Backward
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update every accum_steps batches.
        if (batch_idx + 1) % accum_steps == 0:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                if torch.isfinite(grad_norm):
                    optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step_i += 1
            if step_i % 50 == 0:
                loop.write(f"  Step {step_i}, Loss: {loss.item() * accum_steps:.4f}")

        total_loss += loss.item() * accum_steps * x_stat.size(0)
        total_count += x_stat.size(0)

        # Keep predictions on GPU; move once at epoch end.
        all_pred.append(pred_eval.detach())
        all_obs.append(y_eval.detach())

        # Show only loss, skip per-batch R².
        loop.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

    # Handle remaining batches when len(loader) % accum_steps != 0.
    if (batch_idx + 1) % accum_steps != 0:
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if len(all_pred) == 0:
        return 0.0, -999.0

    # Compute R² once per epoch to reduce synchronization overhead.
    all_pred_np = torch.cat(all_pred, dim=0).float().cpu().numpy()
    all_obs_np = torch.cat(all_obs, dim=0).float().cpu().numpy()
    r2 = calc_r2(all_pred_np, all_obs_np)

    return total_loss / max(total_count, 1), r2


def validate_one_epoch(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    spinup_days: int,
    adjoint: bool,
) -> Tuple[float, float]:
    """Validation epoch: compute loss and R²."""
    model.eval()
    criterion = nn.MSELoss()
    total_loss, total_count = 0.0, 0
    all_pred, all_obs = [], []
    loop = tqdm(loader, desc="Val", leave=False)

    with torch.no_grad():
        for batch in loop:
            x_stat, x_dyn, y = unpack_batch_simple(batch)
            x_stat, x_dyn, y = x_stat.to(device), x_dyn.to(device), y.to(device)

            # Validation does not need flux_terms.
            pred = call_model(model, x_stat, x_dyn, mode="finetune", adjoint=adjoint, return_flux=False)
            pred = get_pred_from_output(pred)

            pred_eval = pred[:, spinup_days:] if pred.size(1) > spinup_days else pred
            y_eval = y[:, spinup_days:] if y.size(1) > spinup_days else y
            loss = criterion(pred_eval, y_eval)

            total_loss += loss.item() * x_stat.size(0)
            total_count += x_stat.size(0)

            # Keep tensors on GPU during loop.
            all_pred.append(pred_eval.detach())
            all_obs.append(y_eval.detach())
            loop.set_postfix(loss=f"{loss.item():.4f}")

    # Move once to CPU for final R².
    all_pred_np = torch.cat(all_pred, dim=0).float().cpu().numpy()
    all_obs_np = torch.cat(all_obs, dim=0).float().cpu().numpy()
    r2 = calc_r2(all_pred_np, all_obs_np)

    return total_loss / max(total_count, 1), r2