"""Utility functions: helpers, metrics, and model-call wrappers."""
import inspect
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn


# ==================== File and Directory Utilities ====================
def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


# ==================== Model Calling Helpers ====================
# Cache inspect.signature results to avoid repeated reflection in high-frequency calls.
_kwarg_cache: Dict[Tuple[int, str], bool] = {}


def _supports_kwarg(fn, name: str) -> bool:
    """Check whether a function supports a keyword argument (with caching)."""
    key = (id(fn), name)
    if key not in _kwarg_cache:
        try:
            sig = inspect.signature(fn)
            _kwarg_cache[key] = name in sig.parameters
        except (TypeError, ValueError):
            _kwarg_cache[key] = False
    return _kwarg_cache[key]


def call_model(
    model: nn.Module,
    x_stat: torch.Tensor,
    x_dyn: torch.Tensor,
    mode: str,
    adjoint: bool,
    return_flux: bool = True,
) -> Any:
    """Call model forward compatibly across different forward signatures."""
    if _supports_kwarg(model.forward, "adjoint"):
        if _supports_kwarg(model.forward, "return_flux"):
            return model(x_stat, x_dyn, mode=mode, adjoint=adjoint, return_flux=return_flux)
        return model(x_stat, x_dyn, mode=mode, adjoint=adjoint)
    return model(x_stat, x_dyn, mode=mode)


def get_pred_from_output(output: Any) -> torch.Tensor:
    """Extract prediction tensor from tuple/list model outputs."""
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Parse DataLoader output to a unified format.

    Supports both dict and tuple/list batches.
    Returns: (x_stat, x_dyn, y, meta_dict)
    """
    meta: Dict[str, Any] = {}
    if isinstance(batch, dict):
        meta = {k: batch.get(k) for k in ["grid_id", "lon", "lat", "index"] if k in batch}
        return batch["x_stat"], batch["x_dyn"], batch.get("y"), meta
    if isinstance(batch, (tuple, list)):
        if len(batch) == 3:
            return batch[0], batch[1], batch[2], meta
        if len(batch) >= 4:
            meta = {"index": batch[3]}
            return batch[0], batch[1], batch[2], meta
    raise ValueError("Unsupported batch format from dataset.")


def unpack_batch_simple(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lightweight batch parser when metadata is not needed."""
    x_stat, x_dyn, y, _ = unpack_batch(batch)
    return x_stat, x_dyn, y


# ==================== Hydrological Metrics ====================
def calc_r2(pred: np.ndarray, obs: np.ndarray) -> float:
    """R² (coefficient of determination)."""
    pred = pred.reshape(-1)
    obs = obs.reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[mask], obs[mask]
    if pred.size == 0:
        return np.nan
    ss_res = np.sum((pred - obs) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def calc_nse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency (equivalent to R²)."""
    return calc_r2(pred, obs)


def calc_rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Root Mean Squared Error."""
    pred = pred.reshape(-1)
    obs = obs.reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[mask], obs[mask]
    if pred.size == 0:
        return np.nan
    return np.sqrt(np.mean((pred - obs) ** 2))


def calc_mae(pred: np.ndarray, obs: np.ndarray) -> float:
    """Mean Absolute Error."""
    pred = pred.reshape(-1)
    obs = obs.reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[mask], obs[mask]
    if pred.size == 0:
        return np.nan
    return np.mean(np.abs(pred - obs))


def calc_bias(pred: np.ndarray, obs: np.ndarray) -> float:
    """Mean bias."""
    pred = pred.reshape(-1)
    obs = obs.reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[mask], obs[mask]
    if pred.size == 0:
        return np.nan
    return np.mean(pred - obs)


def calc_ubrmse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Unbiased RMSE."""
    rmse = calc_rmse(pred, obs)
    bias = calc_bias(pred, obs)
    if np.isnan(rmse) or np.isnan(bias):
        return np.nan
    return np.sqrt(max(rmse ** 2 - bias ** 2, 0.0))


def calc_kge(pred: np.ndarray, obs: np.ndarray) -> float:
    """Kling-Gupta Efficiency."""
    pred = pred.reshape(-1)
    obs = obs.reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[mask], obs[mask]
    if pred.size == 0 or np.std(obs) == 0:
        return np.nan
    r = np.corrcoef(pred, obs)[0, 1]
    alpha = np.std(pred) / np.std(obs)
    beta = np.mean(pred) / np.mean(obs)
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def calc_hydro_metrics(pred: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
    """Compute all hydrological metrics in one call."""
    return {
        "R2": calc_r2(pred, obs),
        "NSE": calc_nse(pred, obs),
        "RMSE": calc_rmse(pred, obs),
        "MAE": calc_mae(pred, obs),
        "Bias": calc_bias(pred, obs),
        "ubRMSE": calc_ubrmse(pred, obs),
        "KGE": calc_kge(pred, obs),
    }


# ==================== Training Helpers ====================
def check_finite(tensors: Dict[str, torch.Tensor], stage: str = "") -> None:
    """Check whether tensors contain NaN/Inf values for debugging."""
    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"[{stage}] {name} contains NaN/Inf!")