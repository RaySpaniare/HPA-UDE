"""
HPA-UDE training entrypoint (physics-prior direct finetuning).

Strategy: directly use K-means-based physics priors in GeoHyperNet and skip
separate self-supervised pretraining.

Core optimizations:
1. Remove pretraining stage.
2. Use stratified hold-out split by Cluster ID.
3. Optional GPU preloading for Windows acceleration.
4. Optional modular torch.compile.
5. Gradient accumulation support.
"""
import argparse
import glob
import os
import platform
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler  # pyright: ignore[reportPrivateImportUsage]
from torch.utils.data import DataLoader

from dataset import SpatioTemporalDataset, get_dataloaders
from dataset_config import STATIC_COLS
from model import HPA_UDE_Model
from preload import get_dataloaders_preloaded
from sklearn.preprocessing import StandardScaler
from trainers import finetune_one_epoch, validate_one_epoch
from utils import ensure_dir


def _default_data_dir() -> str:
    """Find a default data directory by searching annual parquet files."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    matches = sorted(
        glob.glob(os.path.join(repo_root, "**", "Soil_Moisture_Data_2015.parquet"), recursive=True)
    )
    if matches:
        return os.path.dirname(matches[0])
    return os.path.join(repo_root, "data")


def save_loss_curves(loss_df: pd.DataFrame, results_dir: str) -> None:
    """Save loss curves once at the end (JPEG + PDF)."""
    if loss_df.empty:
        print("⚠️ Loss history is empty; skipping curve export.")
        return

    # PDF Type42 keeps text editable (no path conversion to outlines).
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.sans-serif"] = "Arial"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.plot(loss_df["epoch"], loss_df["train_loss"], label="Train Loss", linewidth=2.2)
    ax.plot(loss_df["epoch"], loss_df["val_loss"], label="Val Loss", linewidth=2.2)
    ax.set_xlabel("Epoch", fontsize=13, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=13, fontweight="bold")
    ax.set_title("Training & Validation Loss Curves", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=False, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    jpeg_path = os.path.join(results_dir, "loss_curves.jpeg")
    pdf_path = os.path.join(results_dir, "loss_curves.pdf")
    fig.savefig(jpeg_path, dpi=800, bbox_inches="tight", format="jpeg")
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
    plt.close(fig)

    print(f"✅ Loss curve saved: {jpeg_path}")
    print(f"✅ Loss curve saved: {pdf_path}")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def adjust_num_workers(num_workers: int) -> int:
    return 0 if platform.system().lower().startswith("win") else num_workers


def main() -> None:
    """
    HPA-UDE training entrypoint (physics-prior direct finetuning).

    Strategy summary:
    - Skip pretraining and directly finetune with K-means-based physics priors.
    - GeoHyperNet already includes CLUSTER_PHYSICS_BIAS.
    - Use stratified hold-out split (60%/20%/20%).
    """
    parser = argparse.ArgumentParser(description="Train HPA-UDE Model (Physics Prior Direct Finetuning)")
    parser.add_argument("--data_dir", type=str, default=_default_data_dir())
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--spinup_days", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lambda_mass", type=float, default=0.001)
    parser.add_argument("--lambda_flux", type=float, default=0.001)
    parser.add_argument(
        "--lambda_mono",
        type=float,
        default=0.0,
        help="Weight for KAN monotonicity loss (0 disables, recommended: 0.01)",
    )
    parser.add_argument("--physics_warmup", type=int, default=5, help="Epochs of pure MSE before physics losses")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision (enabled by default)")
    parser.add_argument("--use_compile", action="store_true", help="Enable torch.compile (requires PyTorch 2.0+)")
    parser.add_argument("--use_preload", action="store_true", help="Preload data to GPU (recommend >= 8GB VRAM)")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")
    args = parser.parse_args()

    set_seed(args.seed)

    # Initialize model (physics priors already embedded in GeoHyperNet)
    # static_dim=13: [Soil(5), Terrain(2), LonLat(2), Cluster(4)]
    # dynamic_dim=9: [Pre, PET, LST, LAI, Pre_lag1, PET_lag1, DOY_sin, DOY_cos, Year_norm]
    model = HPA_UDE_Model(static_dim=13, dynamic_dim=9, hidden_dim=64)
    device = torch.device(args.device)
    model.to(device)

    # Print run configuration
    print("=" * 60)
    print("🚀 HPA-UDE Training (Physics Prior Direct Finetuning)")
    print("=" * 60)
    print("📋 Configuration:")
    print(f"   Device: {device}")
    print(f"   Physics: λ_mass={args.lambda_mass}, λ_flux={args.lambda_flux}, λ_mono={args.lambda_mono}")
    print(f"   Batch: {args.batch_size} | Accum: {args.accum_steps} | Effective: {args.batch_size * args.accum_steps}")
    print(f"   Physics warmup: {args.physics_warmup} epochs (MSE only)")
    print("\n📌 Physics Prior Bias (from K-means clustering, logit-space):")
    print("   C1 (transition):     K_sat=0.0,  S_max=0.0,  c_exp=0.0")
    print("   C2 (dry-hot/clayey): K_sat=-1.0, S_max=+1.5, c_exp=+0.5")
    print("   C3 (transition):     K_sat=0.0,  S_max=0.0,  c_exp=0.0")
    print("   C4 (humid/sandy):    K_sat=+1.0, S_max=-1.0, c_exp=-0.5")
    print("   (K_sat/c_exp use softplus activation, S_max uses sigmoid)")

    # Modular torch.compile to avoid dynamic-control-flow blockers.
    if args.use_compile and hasattr(torch, "compile"):
        print("\n🚀 Enabling torch.compile (modular)...")
        try:
            model.hypernet = torch.compile(model.hypernet, mode="reduce-overhead", fullgraph=False)  # pyright: ignore[reportAttributeAccessIssue]
            model.backbone = torch.compile(model.backbone, mode="reduce-overhead", fullgraph=False)  # pyright: ignore[reportAttributeAccessIssue]
            print("   ✅ hypernet & backbone compiled")
        except Exception as e:
            print(f"   ⚠️ torch.compile failed: {e}")

    # Mixed precision configuration (enabled by default on CUDA).
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    if use_amp:
        amp_type = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        print(f"🚀 AMP enabled ({amp_type})!")

    checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    ensure_dir(checkpoints_dir)
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    ensure_dir(results_dir)
    best_path = os.path.join(checkpoints_dir, "best_model.pth")

    # ========== Data Loading (Stratified Hold-Out) ==========
    print("\n" + "=" * 60)
    print("📦 Loading Data (Stratified Hold-Out)")
    print("=" * 60)

    if args.use_preload and device.type == "cuda":
        print("🚀 Using GPU preloaded dataloaders (recommend >= 8GB VRAM)...")
        train_loader, val_loader, _ = get_dataloaders_preloaded(
            args.data_dir, args.batch_size, device=device
        )
    else:
        train_loader, val_loader, _ = get_dataloaders(
            args.data_dir, args.batch_size, adjust_num_workers(args.num_workers)
        )

    # Set dynamic stats for physical-unit de-normalization.
    model.set_dyn_stats(
        train_loader.dataset.dynamic_scaler.mean_,  # pyright: ignore[reportAttributeAccessIssue]
        train_loader.dataset.dynamic_scaler.scale_,  # pyright: ignore[reportAttributeAccessIssue]
    )
    print(f"📏 SM p95 (reference) = {train_loader.dataset.sm_scale:.4f} mm")  # pyright: ignore[reportAttributeAccessIssue]

    # ========== Training ==========
    print("\n" + "=" * 60)
    print(f"🎯 Training: {args.epochs} epochs, LR={args.lr}, Spinup={args.spinup_days} days")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-6
    )

    best_r2, no_improve = -float("inf"), 0
    loss_history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Progressive physics: first N epochs are pure data-driven MSE.
        if epoch <= args.physics_warmup:
            cur_lm, cur_lf, cur_lmono = 0.0, 0.0, 0.0
            warmup_tag = " [MSE only]"
        else:
            cur_lm, cur_lf, cur_lmono = args.lambda_mass, args.lambda_flux, args.lambda_mono
            warmup_tag = ""

        train_loss, train_r2 = finetune_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.spinup_days,
            False,
            scaler,
            use_amp,
            cur_lm,
            cur_lf,
            args.accum_steps,
            cur_lmono,  # pyright: ignore[reportArgumentType]
        )
        val_loss, val_r2 = validate_one_epoch(model, val_loader, device, args.spinup_days, False)

        elapsed = int(time.time() - t0)
        print(
            f"[Epoch {epoch:3d}/{args.epochs}] Train: {train_loss:.4f} (R²:{train_r2:.4f}) | "
            f"Val: {val_loss:.4f} (R²:{val_r2:.4f}) | {elapsed}s{warmup_tag}"
        )

        loss_history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_r2": float(train_r2),
                "val_r2": float(val_r2),
                "lambda_mass": float(cur_lm),
                "lambda_flux": float(cur_lf),
                "lambda_mono": float(cur_lmono),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_sec": int(elapsed),
            }
        )

        scheduler.step(val_r2)
        lr = optimizer.param_groups[0]["lr"]

        if val_r2 > best_r2:
            best_r2, no_improve = val_r2, 0
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ New best R²={best_r2:.4f} (LR={lr:.2e})")
        else:
            no_improve += 1
            if no_improve % 10 == 0:
                print(f"  ⚠️ No improvement for {no_improve} epochs (LR={lr:.2e})")

        if lr <= 1e-6 and no_improve >= 15:
            print(f"\n🛑 Early stopping at epoch {epoch}!")
            break

    # Save training history (minimal overhead)
    loss_df = pd.DataFrame(loss_history)
    loss_csv_path = os.path.join(results_dir, "loss_history.csv")
    loss_df.to_csv(loss_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Loss history saved: {loss_csv_path}")
    save_loss_curves(loss_df, results_dir)

    # ========== Evaluation ==========
    print("\n" + "=" * 60)
    print("📊 Evaluation on All Splits (Train / Val / Test)")
    print("=" * 60)

    from evaluate import IndexedDataset, evaluate_and_save

    train_loader_eval, val_loader_eval, test_loader_eval = get_dataloaders(
        args.data_dir, args.batch_size, adjust_num_workers(args.num_workers)
    )

    def _wrap_loader(loader):
        return DataLoader(
            IndexedDataset(loader.dataset),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=adjust_num_workers(args.num_workers),
            pin_memory=torch.cuda.is_available(),
        )

    model.load_state_dict(torch.load(best_path, map_location=device))
    evaluate_and_save(
        model,
        loaders={
            "Train": _wrap_loader(train_loader_eval),
            "Val": _wrap_loader(val_loader_eval),
            "Test": _wrap_loader(test_loader_eval),
        },
        device=device,
        spinup_days=args.spinup_days,
        results_dir=results_dir,
        adjoint=False,
    )

    print("\n" + "=" * 60)
    print("🎉 Training Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()