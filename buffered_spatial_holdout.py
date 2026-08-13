# -*- coding: utf-8 -*-
r'''
@File    :   buffered_spatial_holdout.py
@Time    :   2026-06-27
@Desc    :   Run five-fold spatial-block holdout validation for HPA-UDE without
             changing the model architecture, input variables, loss functions,
             optimizer, learning-rate schedule, spin-up period, or metrics. Grid
             cells are ordered along the dominant spatial extent and divided into
             contiguous blocks. In each fold, one block is held out for testing,
             the adjacent block is used for validation, and the remaining blocks
             are used for training. Static and dynamic scalers are fitted only on
             training grid cells. The script reuses the repository's dataset,
             model, trainer, and evaluation modules and writes fold-specific
             checkpoints, predictions, metrics, and summary tables.
@Notice  :   Paths default to locations relative to this script. Use --data_dir
             and --output_root to override them. Use --prepare_only to inspect the
             spatial partition without training the model.
'''
import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
ORIGINAL_PROJECT_DIR = SCRIPT_DIR
OUTPUT_ROOT = SCRIPT_DIR
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
N_SPATIAL_BLOCKS = 5
ORIGINAL_TRAIN_SPINUP_DAYS = 60


class CleanTextStream:
    """Remove decorative symbols emitted by legacy modules from plain-text logs."""

    _SYMBOL_PATTERN = re.compile(
        "["
        "\U0001F300-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U00002600-\U000026FF"
        "\U0001F1E6-\U0001F1FF"
        "\uFE0F"
        "]+"
    )

    def __init__(self, wrapped) -> None:
        self.wrapped = wrapped

    def write(self, text: str) -> int:
        return self.wrapped.write(self._SYMBOL_PATTERN.sub("", str(text)))

    def flush(self) -> None:
        self.wrapped.flush()

    def reconfigure(self, *args, **kwargs) -> None:
        if hasattr(self.wrapped, "reconfigure"):
            self.wrapped.reconfigure(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.wrapped, name)


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
sys.stdout = CleanTextStream(sys.stdout)
sys.stderr = CleanTextStream(sys.stderr)

if str(ORIGINAL_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_PROJECT_DIR))

from dataset import SpatioTemporalDataset  # noqa: E402  # pyright: ignore[reportMissingImports]
from dataset_config import DYNAMIC_COLS, STATIC_COLS  # noqa: E402  # pyright: ignore[reportMissingImports]
from evaluate import IndexedDataset, evaluate_and_save  # noqa: E402  # pyright: ignore[reportMissingImports]
from model import HPA_UDE_Model  # noqa: E402  # pyright: ignore[reportMissingImports]
from train import save_loss_curves, set_seed  # noqa: E402  # pyright: ignore[reportMissingImports]
from trainers import finetune_one_epoch, validate_one_epoch  # noqa: E402  # pyright: ignore[reportMissingImports]
from utils import ensure_dir  # noqa: E402  # pyright: ignore[reportMissingImports]


def fit_dynamic_scaler_from_store(
    store,
    grid_ids: Sequence[int],
    chunk_rows: int = 1_000_000,
) -> StandardScaler:
    """Fit dynamic-feature scaling incrementally using training grid cells only."""
    dynamic_scaler = StandardScaler()
    chunks: List[np.ndarray] = []
    buffered_rows = 0
    fitted = False

    for grid_id in grid_ids:
        values = store.dyn_dict.get(int(grid_id))
        if values is None or values.size == 0:
            continue
        chunks.append(values)
        buffered_rows += values.shape[0]
        if buffered_rows >= chunk_rows:
            dynamic_scaler.partial_fit(np.concatenate(chunks, axis=0))
            chunks.clear()
            buffered_rows = 0
            fitted = True

    if chunks:
        dynamic_scaler.partial_fit(np.concatenate(chunks, axis=0))
        fitted = True
    if not fitted:
        raise ValueError("No dynamic feature rows are available for scaler fitting.")

    dynamic_scaler.scale_[dynamic_scaler.scale_ == 0] = 1.0
    return dynamic_scaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spatial block holdout validation for HPA-UDE.")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output_root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--spinup_days",
        type=int,
        default=ORIGINAL_TRAIN_SPINUP_DAYS,
        help="Spin-up days aligned with the current default in train.py.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lambda_mass", type=float, default=0.001)
    parser.add_argument("--lambda_flux", type=float, default=0.001)
    parser.add_argument("--lambda_mono", type=float, default=0.0)
    parser.add_argument("--physics_warmup", type=int, default=5)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument(
        "--batch_benchmark_sizes",
        type=int,
        nargs="*",
        default=[64, 128, 256, 512, 1024, 2048, 4096],
    )
    parser.add_argument("--batch_benchmark_warmup_steps", type=int, default=1)
    parser.add_argument("--batch_benchmark_timed_steps", type=int, default=3)
    parser.add_argument("--skip_batch_benchmark", action="store_true")
    parser.add_argument("--force_batch_benchmark", action="store_true")
    parser.add_argument("--benchmark_only", action="store_true")
    parser.add_argument(
        "--fixed_batch_size",
        action="store_true",
        help="Record benchmark only; do not replace formal training batch_size.",
    )
    parser.add_argument("--year_start", type=int, default=2015)
    parser.add_argument("--year_end", type=int, default=2019)
    parser.add_argument("--n_blocks", type=int, default=N_SPATIAL_BLOCKS)
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--force_rebuild_splits", action="store_true")
    parser.add_argument("--skip_timeseries", action="store_true")
    return parser.parse_args()


def adjust_num_workers(num_workers: int) -> int:
    return 0 if platform.system().lower().startswith("win") else num_workers


def build_grid_metadata(base_dataset: SpatioTemporalDataset) -> pd.DataFrame:
    meta = base_dataset.static_df.loc[:, ["Lon", "Lat"]].copy()
    meta.insert(0, "Grid_ID", meta.index.astype(np.int64))
    meta["Cluster_ID"] = [
        int(base_dataset.store.cluster_dict.get(int(gid), -1)) for gid in meta["Grid_ID"]
    ]
    return meta.reset_index(drop=True)


def assign_spatial_blocks(meta_df: pd.DataFrame, n_blocks: int) -> pd.DataFrame:
    """Order grid cells along the dominant spatial extent and form contiguous blocks."""
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least 2.")
    meta = meta_df.copy()
    lon = meta["Lon"].to_numpy(dtype=np.float64)
    lat = meta["Lat"].to_numpy(dtype=np.float64)
    x = (lon - np.nanmean(lon)) * 111.32 * np.cos(np.deg2rad(float(np.nanmean(lat))))
    y = (lat - np.nanmean(lat)) * 110.57
    score = x if np.nanmax(x) - np.nanmin(x) >= np.nanmax(y) - np.nanmin(y) else y
    order = np.argsort(score, kind="mergesort")
    blocks = np.empty(len(meta), dtype=np.int16)
    for block_id, idx in enumerate(np.array_split(order, n_blocks)):
        blocks[idx] = block_id
    meta["Spatial_Block"] = blocks
    meta["Spatial_Score"] = score
    return meta


def create_or_load_splits(
    base_dataset: SpatioTemporalDataset,
    output_root: Path,
    n_blocks: int,
    force_rebuild: bool,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, np.ndarray]]]:
    splits_dir = output_root / "splits"
    ensure_dir(str(splits_dir))
    block_path = splits_dir / "spatial_blocks.csv"
    split_path = splits_dir / "spatial_block_holdout_splits.csv"

    if block_path.exists() and split_path.exists() and not force_rebuild:
        meta = pd.read_csv(block_path)
        split_df = pd.read_csv(split_path)
    else:
        meta = assign_spatial_blocks(build_grid_metadata(base_dataset), n_blocks)
        rows: List[Dict[str, object]] = []
        for fold in range(n_blocks):
            test_block = fold
            val_block = (fold + 1) % n_blocks
            for _, row in meta.iterrows():
                block = int(row["Spatial_Block"])
                if block == test_block:
                    split = "Test"
                elif block == val_block:
                    split = "Val"
                else:
                    split = "Train"
                rows.append(
                    {
                        "Fold": fold,
                        "Grid_ID": int(row["Grid_ID"]),
                        "Split": split,
                        "Spatial_Block": block,
                        "Lon": float(row["Lon"]),
                        "Lat": float(row["Lat"]),
                        "Cluster_ID": int(row["Cluster_ID"]),
                    }
                )
        split_df = pd.DataFrame(rows)
        meta.to_csv(block_path, index=False, encoding="utf-8-sig")
        split_df.to_csv(split_path, index=False, encoding="utf-8-sig")

    summary = (
        split_df.groupby(["Fold", "Split"])
        .size()
        .rename("N_Grids")
        .reset_index()
        .sort_values(["Fold", "Split"])
    )
    summary.to_csv(splits_dir / "spatial_block_holdout_summary.csv", index=False, encoding="utf-8-sig")

    fold_splits: Dict[int, Dict[str, np.ndarray]] = {}
    for fold in sorted(split_df["Fold"].unique()):
        fold_df = split_df.loc[split_df["Fold"] == fold]
        fold_splits[int(fold)] = {
            split: fold_df.loc[fold_df["Split"] == split, "Grid_ID"].to_numpy(dtype=np.int64)
            for split in ["Train", "Val", "Test"]
        }
    return meta, fold_splits


def build_spatial_dataloaders(
    data_dir: str,
    year_range: Tuple[int, int],
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    test_ids: Sequence[int],
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    base_dataset = SpatioTemporalDataset(data_dir=data_dir, year_range=year_range, mode="train")
    train_ids = np.asarray(train_ids, dtype=np.int64)
    val_ids = np.asarray(val_ids, dtype=np.int64)
    test_ids = np.asarray(test_ids, dtype=np.int64)

    static_scaler = StandardScaler()
    static_vals = base_dataset.static_df.loc[train_ids, STATIC_COLS].to_numpy()
    static_scaler.fit(static_vals)
    static_scaler.scale_[static_scaler.scale_ == 0] = 1.0
    dynamic_scaler = fit_dynamic_scaler_from_store(base_dataset.store, train_ids)

    train_ds = SpatioTemporalDataset(
        data_dir=data_dir,
        year_range=year_range,
        mode="train",
        grid_ids=train_ids,
        static_scaler=static_scaler,
        dynamic_scaler=dynamic_scaler,
    )
    val_ds = SpatioTemporalDataset(
        data_dir=data_dir,
        year_range=year_range,
        mode="val",
        grid_ids=val_ids,
        static_scaler=static_scaler,
        dynamic_scaler=dynamic_scaler,
    )
    test_ds = SpatioTemporalDataset(
        data_dir=data_dir,
        year_range=year_range,
        mode="test",
        grid_ids=test_ids,
        static_scaler=static_scaler,
        dynamic_scaler=dynamic_scaler,
    )

    actual_workers = adjust_num_workers(num_workers)
    loader_kw: Dict[str, object] = {"pin_memory": torch.cuda.is_available()}
    if actual_workers > 0:
        loader_kw["prefetch_factor"] = 2
        loader_kw["persistent_workers"] = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=actual_workers,
        drop_last=True,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=actual_workers,
        **loader_kw,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=actual_workers,
        **loader_kw,
    )
    return train_loader, val_loader, test_loader


def build_model(args: argparse.Namespace, device: torch.device) -> HPA_UDE_Model:
    model = HPA_UDE_Model(static_dim=13, dynamic_dim=9, hidden_dim=64)
    model.to(device)
    if args.use_compile and hasattr(torch, "compile"):
        try:
            model.hypernet = torch.compile(model.hypernet, mode="reduce-overhead", fullgraph=False)  # pyright: ignore[reportAttributeAccessIssue]
            model.backbone = torch.compile(model.backbone, mode="reduce-overhead", fullgraph=False)  # pyright: ignore[reportAttributeAccessIssue]
            print("INFO torch.compile enabled for hypernet and backbone.")
        except Exception as exc:
            print(f"WARN torch.compile skipped: {exc}")
    return model


def wrap_eval_loader(loader: DataLoader, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        IndexedDataset(loader.dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=adjust_num_workers(num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def amp_dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def run_batch_benchmark(
    args: argparse.Namespace,
    split_ids: Dict[str, np.ndarray],
    year_range: Tuple[int, int],
) -> int:
    """Benchmark batch throughput using the actual split, model, and backpropagation."""
    output_root = Path(args.output_root)
    csv_path = output_root / "batch_benchmark.csv"
    json_path = output_root / "batch_benchmark.json"
    if csv_path.exists() and json_path.exists() and not args.force_batch_benchmark:
        old_df = pd.read_csv(csv_path)
        ok_df = old_df.loc[old_df["status"] == "ok"].copy()
        if len(ok_df) > 0:
            best_row = ok_df.loc[ok_df["samples_per_sec"].astype(float).idxmax()]
            best_batch = int(best_row["batch_size"])
            print(f"INFO Reusing existing batch benchmark. Best batch_size={best_batch}.")
            return best_batch

    device = torch.device(args.device)
    use_amp = (not args.no_amp) and device.type == "cuda"
    criterion = torch.nn.MSELoss()
    rows: List[Dict[str, object]] = []

    print("INFO Running real batch size benchmark.")
    for batch_size in args.batch_benchmark_sizes:
        row: Dict[str, object] = {
            "batch_size": int(batch_size),
            "status": "ok",
            "samples_per_sec": np.nan,
            "avg_step_sec": np.nan,
            "timed_steps": 0,
            "processed_samples": 0,
            "peak_memory_mb": 0.0,
            "error": "",
        }
        try:
            train_loader, _, _ = build_spatial_dataloaders(
                data_dir=args.data_dir,
                year_range=year_range,
                train_ids=split_ids["Train"],
                val_ids=split_ids["Val"],
                test_ids=split_ids["Test"],
                batch_size=int(batch_size),
                num_workers=args.num_workers,
            )
            model = build_model(args, device)
            model.set_dyn_stats(
                train_loader.dataset.dynamic_scaler.mean_,  # pyright: ignore[reportAttributeAccessIssue]
                train_loader.dataset.dynamic_scaler.scale_,  # pyright: ignore[reportAttributeAccessIssue]
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            iterator = iter(train_loader)
            processed_samples = 0

            def one_step(batch) -> int:
                optimizer.zero_grad(set_to_none=True)
                x_stat = batch["x_stat"].to(device)
                x_dyn = batch["x_dyn"].to(device)
                y = batch["y"].to(device)
                with autocast(device_type="cuda", dtype=amp_dtype_for_device(device), enabled=use_amp):
                    output = model(
                        x_stat,
                        x_dyn,
                        mode="finetune",
                        adjoint=False,
                        return_flux=False,
                    )
                    pred = output[0] if isinstance(output, (tuple, list)) else output
                    pred_eval = pred[:, args.spinup_days:] if pred.size(1) > args.spinup_days else pred
                    y_eval = y[:, args.spinup_days:] if y.size(1) > args.spinup_days else y
                    loss = criterion(pred_eval, y_eval)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                return int(x_stat.size(0))

            for _ in range(max(0, int(args.batch_benchmark_warmup_steps))):
                one_step(next(iterator))
            if device.type == "cuda":
                torch.cuda.synchronize()

            timed_steps = max(1, int(args.batch_benchmark_timed_steps))
            start = time.perf_counter()
            for _ in range(timed_steps):
                processed_samples += one_step(next(iterator))
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = max(time.perf_counter() - start, 1e-12)

            row["timed_steps"] = timed_steps
            row["processed_samples"] = processed_samples
            row["avg_step_sec"] = elapsed / timed_steps
            row["samples_per_sec"] = processed_samples / elapsed
            if device.type == "cuda":
                row["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(
                f"INFO Benchmark batch_size={batch_size}: "
                f"{float(row['samples_per_sec']):.2f} samples/sec"
            )
        except RuntimeError as exc:
            message = str(exc)
            row["status"] = "oom" if "out of memory" in message.lower() else "error"
            row["error"] = message[:500]
            print(f"WARN Benchmark batch_size={batch_size} failed: {row['status']}")
        except StopIteration:
            row["status"] = "error"
            row["error"] = "not enough batches for benchmark steps"
            print(f"WARN Benchmark batch_size={batch_size} failed: not enough batches")
        finally:
            rows.append(row)
            try:
                del model, optimizer, train_loader
            except UnboundLocalError:
                pass
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    ok_df = df.loc[df["status"] == "ok"].copy()
    if len(ok_df) == 0:
        raise RuntimeError("All benchmark batch sizes failed.")
    best_row = ok_df.loc[ok_df["samples_per_sec"].astype(float).idxmax()]
    best_batch = int(best_row["batch_size"])

    payload = {
        "environment": {
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "cpu": platform.processor(),
        },
        "benchmark": rows,
        "best_batch_size": best_batch,
        "formal_training_policy": "use best non-OOM samples_per_sec unless --fixed_batch_size is set",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"INFO Batch benchmark saved: {csv_path}")
    print(f"INFO Best batch_size={best_batch}.")
    return best_batch


def train_one_fold(
    fold: int,
    args: argparse.Namespace,
    split_ids: Dict[str, np.ndarray],
    year_range: Tuple[int, int],
) -> None:
    fold_dir = Path(args.output_root) / f"fold_{fold}"
    checkpoints_dir = fold_dir / "checkpoints"
    results_dir = fold_dir / "results"
    ensure_dir(str(checkpoints_dir))
    ensure_dir(str(results_dir))

    set_seed(args.seed + fold)
    device = torch.device(args.device)
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    train_loader, val_loader, test_loader = build_spatial_dataloaders(
        data_dir=args.data_dir,
        year_range=year_range,
        train_ids=split_ids["Train"],
        val_ids=split_ids["Val"],
        test_ids=split_ids["Test"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args, device)
    model.set_dyn_stats(
        train_loader.dataset.dynamic_scaler.mean_,  # pyright: ignore[reportAttributeAccessIssue]
        train_loader.dataset.dynamic_scaler.scale_,  # pyright: ignore[reportAttributeAccessIssue]
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-6
    )
    best_path = checkpoints_dir / "best_model.pth"
    best_r2 = -float("inf")
    no_improve = 0
    loss_history: List[Dict[str, float]] = []

    print("=" * 60)
    print(f"INFO Fold {fold} spatial block holdout")
    print(
        f"INFO Train={len(split_ids['Train'])}, "
        f"Val={len(split_ids['Val'])}, Test={len(split_ids['Test'])}"
    )
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
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
            cur_lmono,
        )
        val_loss, val_r2 = validate_one_epoch(model, val_loader, device, args.spinup_days, False)
        elapsed = int(time.time() - t0)
        print(
            f"[Fold {fold} Epoch {epoch:3d}/{args.epochs}] "
            f"Train: {train_loss:.4f} (R2:{train_r2:.4f}) | "
            f"Val: {val_loss:.4f} (R2:{val_r2:.4f}) | {elapsed}s{warmup_tag}"
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
                "elapsed_sec": float(elapsed),
            }
        )

        scheduler.step(val_r2)
        lr = float(optimizer.param_groups[0]["lr"])
        if val_r2 > best_r2:
            best_r2 = float(val_r2)
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"INFO New best fold {fold} R2={best_r2:.4f}, LR={lr:.8f}")
        else:
            no_improve += 1

        if lr <= 1e-6 and no_improve >= 15:
            print(f"INFO Early stopping fold {fold} at epoch {epoch}.")
            break

    loss_df = pd.DataFrame(loss_history)
    loss_df.to_csv(results_dir / "loss_history.csv", index=False, encoding="utf-8-sig")
    save_loss_curves(loss_df, str(results_dir))

    model.load_state_dict(torch.load(best_path, map_location=device))
    evaluate_and_save(
        model,
        loaders={
            "Train": wrap_eval_loader(train_loader, args.batch_size, args.num_workers),
            "Val": wrap_eval_loader(val_loader, args.batch_size, args.num_workers),
            "Test": wrap_eval_loader(test_loader, args.batch_size, args.num_workers),
        },
        device=device,
        spinup_days=args.spinup_days,
        results_dir=str(results_dir),
        adjoint=False,
        year_range=year_range,
        save_timeseries=not args.skip_timeseries,
    )


def save_run_config(args: argparse.Namespace, meta: pd.DataFrame) -> None:
    output_root = Path(args.output_root)
    payload = {
        "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "split_method": "spatial_block_holdout_by_grid_id",
        "n_blocks": int(args.n_blocks),
        "data_dir": args.data_dir,
        "output_root": args.output_root,
        "n_grid": int(len(meta)),
        "block_counts": {
            str(k): int(v) for k, v in meta["Spatial_Block"].value_counts().sort_index().to_dict().items()
        },
        "args": vars(args),
    }
    (output_root / "spatial_block_holdout_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def combine_fold_metrics(output_root: Path, folds: Sequence[int]) -> None:
    frames = []
    for fold in folds:
        path = output_root / f"fold_{fold}" / "results" / "metrics_summary.csv"
        if path.exists():
            df = pd.read_csv(path)
            df.insert(0, "Fold", int(fold))
            frames.append(df)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_root / "spatial_block_holdout_metrics_all_folds.csv", index=False, encoding="utf-8-sig")
    summary = (
        combined.groupby("Split")[["R2", "NSE", "RMSE", "MAE", "Bias", "ubRMSE", "KGE"]]
        .agg(["mean", "std", "min", "max"])
    )
    summary.to_csv(output_root / "spatial_block_holdout_metrics_summary_by_split.csv", encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    ensure_dir(str(output_root))
    ensure_dir(str(output_root / "splits"))

    year_range = (args.year_start, args.year_end)
    base_dataset = SpatioTemporalDataset(data_dir=args.data_dir, year_range=year_range, mode="train")
    meta, fold_splits = create_or_load_splits(
        base_dataset,
        output_root,
        args.n_blocks,
        args.force_rebuild_splits,
    )

    folds = args.folds if args.folds is not None else list(range(args.n_blocks))
    print(f"INFO Prepared {args.n_blocks} spatial blocks for {len(meta)} grids.")
    print(f"INFO Folds to run: {folds}")
    if args.prepare_only:
        print("INFO prepare_only is set; stop before model training.")
        return

    if not args.skip_batch_benchmark:
        benchmark_fold = int(folds[0])
        best_batch_size = run_batch_benchmark(args, fold_splits[benchmark_fold], year_range)
        if not args.fixed_batch_size:
            args.batch_size = best_batch_size
            print(f"INFO Formal training batch_size set to benchmark best: {args.batch_size}.")
        else:
            print(f"INFO Formal training keeps fixed batch_size={args.batch_size}.")
    if args.benchmark_only:
        save_run_config(args, meta)
        print("INFO benchmark_only is set; stop before model training.")
        return

    save_run_config(args, meta)

    for fold in folds:
        train_one_fold(int(fold), args, fold_splits[int(fold)], year_range)
    combine_fold_metrics(output_root, folds)
    print("DONE Spatial block holdout complete.")


if __name__ == "__main__":
    main()
