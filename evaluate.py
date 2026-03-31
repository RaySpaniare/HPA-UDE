import argparse
import glob
import os
import platform
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import SpatioTemporalDataset, get_dataloaders
from model import HPA_UDE_Model
from utils import (
    calc_hydro_metrics,
    call_model,
    ensure_dir,
    unpack_batch,
)


def _default_data_dir() -> str:
    """Find a reasonable default data directory by scanning parquet files."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    matches = sorted(
        glob.glob(os.path.join(repo_root, "**", "Soil_Moisture_Data_2015.parquet"), recursive=True)
    )
    if matches:
        return os.path.dirname(matches[0])
    return os.path.join(repo_root, "data")


class IndexedDataset(Dataset):
    # Wrap dataset and attach sample index for Grid metadata backtracking.
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        if isinstance(sample, dict):
            sample = dict(sample)
            sample["index"] = idx
            return sample
        if isinstance(sample, (tuple, list)):
            return (*sample, idx)
        raise ValueError("Unsupported sample format from dataset.")


def build_grid_meta(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Read Grid_ID / Lon / Lat from dataset metadata.
    if hasattr(dataset, "get_metadata"):
        meta = dataset.get_metadata()
        return (
            meta["Grid_ID"].to_numpy(),
            meta["Lon"].to_numpy(),
            meta["Lat"].to_numpy(),
        )
    raise RuntimeError(
        "Dataset must implement get_metadata() returning Grid_ID, Lon, Lat."
    )


def evaluate_split(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    spinup_days: int,
    adjoint: bool,
    split_name: str = "test",
) -> Dict[str, Any]:
    """
    Evaluate one split and return structured outputs.

    Returns:
        dict: pred, obs, phy_params, flux_terms, grid_ids, lons, lats, cluster_ids
    """
    model.eval()

    all_pred: List[np.ndarray] = []
    all_obs: List[np.ndarray] = []
    all_phy: List[np.ndarray] = []
    all_flux: List[np.ndarray] = []
    all_grid_ids: List[Any] = []
    all_lons: List[float] = []
    all_lats: List[float] = []

    loop = tqdm(loader, desc=f"Eval-{split_name}", leave=False)
    with torch.no_grad():
        for batch in loop:
            x_stat, x_dyn, y, meta = unpack_batch(batch)
            x_stat = x_stat.to(device)
            x_dyn = x_dyn.to(device)
            y = y.to(device)

            output = call_model(model, x_stat, x_dyn, mode="finetune", adjoint=adjoint)
            if isinstance(output, (tuple, list)) and len(output) == 4:
                pred, _pred_phy, phy_params, flux_terms = output
            elif isinstance(output, (tuple, list)) and len(output) == 3:
                pred, _pred_phy, phy_params = output
                flux_terms = torch.zeros(pred.size(0), pred.size(1), 2, device=device)
            else:
                raise RuntimeError("Model forward must return 3 or 4-element tuple.")

            all_pred.append(pred.cpu().numpy())
            all_obs.append(y.cpu().numpy())
            all_phy.append(phy_params.cpu().numpy())
            all_flux.append(flux_terms.cpu().numpy())

            if "grid_id" in meta and "lon" in meta and "lat" in meta:
                all_grid_ids.extend(meta["grid_id"])
                all_lons.extend(meta["lon"])
                all_lats.extend(meta["lat"])
            elif "index" in meta:
                all_grid_ids.extend(meta["index"])

    pred_arr = np.concatenate(all_pred, axis=0)
    obs_arr = np.concatenate(all_obs, axis=0)
    phy_arr = np.concatenate(all_phy, axis=0)
    flux_arr = np.concatenate(all_flux, axis=0)
    # pred_arr / obs_arr are already in mm physical units; no inverse normalization needed.

    # Resolve grid metadata
    if len(all_lons) == 0 or len(all_lats) == 0:
        dataset = loader.dataset
        if isinstance(dataset, IndexedDataset):
            dataset = dataset.base
        grid_ids, lons, lats = build_grid_meta(dataset)
        if len(all_grid_ids) == len(pred_arr):
            idx_array = np.asarray(all_grid_ids, dtype=int)
            grid_ids = grid_ids[idx_array]
            lons = lons[idx_array]
            lats = lats[idx_array]
    else:
        grid_ids = np.asarray(all_grid_ids)
        lons = np.asarray(all_lons)
        lats = np.asarray(all_lats)

    # Collect cluster IDs
    base_ds = loader.dataset.base if isinstance(loader.dataset, IndexedDataset) else loader.dataset
    cluster_ids = np.array(
        [base_ds.store.cluster_dict.get(int(gid), -1) for gid in grid_ids],
        dtype=np.int32,
    )

    return {
        "pred": pred_arr,
        "obs": obs_arr,
        "phy_params": phy_arr,
        "flux_terms": flux_arr,
        "grid_ids": grid_ids,
        "lons": lons,
        "lats": lats,
        "cluster_ids": cluster_ids,
        "split": split_name,
    }


def _build_metrics_rows(
    result: Dict[str, Any],
    spinup_days: int,
) -> List[Dict[str, Any]]:
    """Build per-grid metric rows from one split result."""
    pred = result["pred"]
    obs = result["obs"]
    phy = result["phy_params"]
    grid_ids = result["grid_ids"]
    lons = result["lons"]
    lats = result["lats"]
    cluster_ids = result["cluster_ids"]
    split = result["split"]

    # Spin-up truncation
    if pred.shape[1] > spinup_days:
        pred_m = pred[:, spinup_days:]
        obs_m = obs[:, spinup_days:]
    else:
        pred_m = pred
        obs_m = obs

    rows = []
    for i in range(pred_m.shape[0]):
        metrics = calc_hydro_metrics(pred_m[i], obs_m[i])
        # pred/obs are in mm in evaluate_split.
        # Convert error metrics to volumetric water content (cm3/cm3) via 0-10 cm scaling.
        for err_key in ["RMSE", "MAE", "Bias", "ubRMSE"]:
            if err_key in metrics and pd.notna(metrics[err_key]):
                metrics[err_key] = float(metrics[err_key]) / 100.0

        rows.append({
            "Grid_ID": int(grid_ids[i]),
            "Lon": float(lons[i]),
            "Lat": float(lats[i]),
            "Cluster_ID": int(cluster_ids[i]),
            "Split": split,
            "K_sat": float(phy[i, 0]),
            "S_max": float(phy[i, 1]),
            "c_exp": float(phy[i, 2]),
            **metrics,
        })
    return rows


def _build_timeseries_df(
    result: Dict[str, Any],
    spinup_days: int,
    year_range: Tuple[int, int] = (2015, 2019),
    base_dataset: Optional[SpatioTemporalDataset] = None,
) -> pd.DataFrame:
    """Build test-set timeseries DataFrame in a memory-friendly vectorized way."""
    pred = np.asarray(result["pred"]).squeeze(-1)  # (N, T)
    obs = np.asarray(result["obs"]).squeeze(-1)
    flux = np.asarray(result["flux_terms"])  # (N, T, 2)
    grid_ids = np.asarray(result["grid_ids"])

    n_grids, t_full = pred.shape

    # Build date index
    dates = None
    if base_dataset is not None and hasattr(base_dataset, "store"):
        # Try to get real dates from the first grid
        sample_gid = int(grid_ids[0])
        ddf = base_dataset.store.dynamic_df
        sample_dates = ddf.loc[ddf["Grid_ID"] == sample_gid, "Date"].values
        if len(sample_dates) == t_full:
            dates = pd.to_datetime(sample_dates)

    if dates is None:
        # Fallback: generate dates from year_range
        start = pd.Timestamp(f"{year_range[0]}-01-01")
        dates = pd.date_range(start, periods=t_full, freq="D")
    dates = pd.to_datetime(dates)

    # Spin-up truncation
    if t_full > spinup_days:
        pred = pred[:, spinup_days:]
        obs = obs[:, spinup_days:]
        flux = flux[:, spinup_days:]
        dates = dates[spinup_days:]

    t = pred.shape[1]
    dates_np = dates.to_numpy(dtype="datetime64[ns]")

    return pd.DataFrame(
        {
            "Grid_ID": np.repeat(grid_ids.astype(np.int64, copy=False), t),
            "Date": np.tile(dates_np, n_grids),
            "Pred": pred.reshape(-1).astype(np.float32, copy=False),
            "Obs": obs.reshape(-1).astype(np.float32, copy=False),
            "E_act": flux[:, :, 0].reshape(-1).astype(np.float32, copy=False),
            "D_flux": flux[:, :, 1].reshape(-1).astype(np.float32, copy=False),
        }
    )


def _generate_report(
    all_metrics: pd.DataFrame,
    results_dir: str,
) -> str:
    """Generate evaluation report text."""
    lines = [
        "=" * 60,
        "HPA-UDE Model Evaluation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
    ]

    for split in ["Train", "Val", "Test"]:
        df_split = all_metrics[all_metrics["Split"] == split]
        if len(df_split) == 0:
            continue
        lines.append(f"{'─' * 50}")
        lines.append(f"  {split} Set  (N = {len(df_split)} grids)")
        lines.append(f"{'─' * 50}")
        for col in ["R2", "RMSE", "MAE", "Bias", "ubRMSE", "KGE", "NSE"]:
            if col in df_split.columns:
                vals = df_split[col].dropna()
                unit = " (cm3/cm3)" if col in {"RMSE", "MAE", "Bias", "ubRMSE"} else ""
                lines.append(
                    f"  {col:>8s}: {vals.mean():.4f} ± {vals.std():.4f}  "
                    f"[{vals.min():.4f}, {vals.max():.4f}]{unit}"
                )
        # Cluster-wise summary
        for cid in sorted(df_split["Cluster_ID"].unique()):
            df_c = df_split[df_split["Cluster_ID"] == cid]
            lines.append(
                f"    Cluster {cid}: R²={df_c['R2'].mean():.4f}, "
                f"RMSE={df_c['RMSE'].mean():.4f} cm3/cm3 (N={len(df_c)})"
            )
        # Physical parameter summary
        lines.append("  Physical params:")
        for p in ["K_sat", "S_max", "c_exp"]:
            if p in df_split.columns:
                vals = df_split[p].dropna()
                lines.append(f"    {p:>6s}: {vals.mean():.4f} ± {vals.std():.4f}")
        lines.append("")

    return "\n".join(lines)


def evaluate_and_save(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    spinup_days: int,
    results_dir: str,
    adjoint: bool = False,
    year_range: Tuple[int, int] = (2015, 2019),
    save_timeseries: bool = True,
) -> None:
    """
    Evaluate all splits and save outputs.

    Output files:
    1. metrics_summary.parquet: per-grid metrics + physical parameters + cluster info
    2. timeseries_predictions.parquet: test-set prediction time series with fluxes
    3. evaluation_report.txt: summary statistics over train/val/test
    """
    ensure_dir(results_dir)
    all_metrics_rows = []

    # Evaluate each split
    split_results = {}
    for split_name, loader in loaders.items():
        if loader is None:
            continue
        result = evaluate_split(model, loader, device, spinup_days, adjoint, split_name)
        split_results[split_name] = result
        rows = _build_metrics_rows(result, spinup_days)
        all_metrics_rows.extend(rows)
        print(f"  ✓ {split_name}: {len(rows)} grids evaluated")

    # 1) Save metrics_summary.parquet
    metrics_df = pd.DataFrame(all_metrics_rows)
    metrics_path = os.path.join(results_dir, "metrics_summary.parquet")
    metrics_df.to_parquet(metrics_path, index=False, engine="pyarrow")
    print(f"📊 Saved: {metrics_path}")

    # Compatibility: save CSV as well for quick inspection
    csv_path = os.path.join(results_dir, "metrics_summary.csv")
    metrics_df.to_csv(csv_path, index=False)

    # 2) Save test timeseries predictions
    if save_timeseries and "Test" in split_results:
        test_result = split_results["Test"]
        base_ds = None
        test_loader = loaders["Test"]
        ds = test_loader.dataset
        if isinstance(ds, IndexedDataset):
            ds = ds.base
        if isinstance(ds, SpatioTemporalDataset):
            base_ds = ds

        try:
            ts_df = _build_timeseries_df(test_result, spinup_days, year_range, base_ds)
            ts_path = os.path.join(results_dir, "timeseries_predictions.parquet")
            ts_df.to_parquet(ts_path, index=False, engine="pyarrow")
            print(f"📈 Saved: {ts_path} ({len(ts_df)} rows)")
        except MemoryError:
            print("⚠️ Skipped timeseries export due to insufficient memory; metrics/report are saved.")
        except Exception as e:
            print(f"⚠️ Skipped timeseries export due to error: {e}")

    # 3) Save evaluation report
    report = _generate_report(metrics_df, results_dir)
    report_path = os.path.join(results_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"📝 Saved: {report_path}")

    # Print summary report
    print("\n" + report)


# ==================== Backward-Compatible API ====================
def evaluate_all(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    spinup_days: int,
    results_dir: str,
    adjoint: bool,
) -> None:
    """Backward-compatible API: evaluate a single loader as Test split."""
    evaluate_and_save(
        model=model,
        loaders={"Test": loader},
        device=device,
        spinup_days=spinup_days,
        results_dir=results_dir,
        adjoint=adjoint,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HPA-UDE Model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=_default_data_dir(),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--spinup_days", type=int, default=365)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth"),
    )
    parser.add_argument("--skip_timeseries", action="store_true", help="Skip timeseries_predictions export")
    args = parser.parse_args()
    num_workers = 0 if platform.system().lower().startswith("win") else args.num_workers

    # Build train/val/test dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )

    # Wrap as IndexedDataset
    def _wrap(loader):
        return DataLoader(
            IndexedDataset(loader.dataset),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    loaders = {
        "Train": _wrap(train_loader),
        "Val": _wrap(val_loader),
        "Test": _wrap(test_loader),
    }

    # Load model
    model = HPA_UDE_Model(
        static_dim=13,
        dynamic_dim=9,
        hidden_dim=64,
    )
    model.set_dyn_stats(
        test_loader.dataset.dynamic_scaler.mean_,
        test_loader.dataset.dynamic_scaler.scale_,
    )
    device = torch.device(args.device)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    evaluate_and_save(
        model,
        loaders,
        device,
        args.spinup_days,
        results_dir,
        adjoint=False,
        save_timeseries=not args.skip_timeseries,
    )


if __name__ == "__main__":
    main()