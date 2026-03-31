"""Dataset module: loading, caching, normalization, and stratified hold-out split."""
import gc
import hashlib
import os
import pickle
import platform
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from dataset_config import (
    CLUSTER_CSV_PATH,
    COLUMN_ALIASES,
    DYNAMIC_COLS,
    DYNAMIC_COLS_RAW,
    DYNAMIC_DIM,
    METEO_DIM,
    N_CLUSTERS,
    STATIC_COLS,
    TARGET_COL,
    DataStore,
)

# Global cache to avoid repeatedly reading the same annual parquet files.
_DATA_CACHE: Dict[str, DataStore] = {}
SM_TO_MM_SCALE = 100.0


def _load_parquet_files(data_dir: str, year_range: Tuple[int, int]) -> pd.DataFrame:
    """Read and concatenate annual parquet files."""
    start_year, end_year = year_range
    frames = []
    for year in range(start_year, end_year + 1):
        file_name = f"Soil_Moisture_Data_{year}.parquet"
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing parquet file: {file_path}")
        df_year = pd.read_parquet(file_path)
        df_year = df_year.fillna(0).replace([np.inf, -np.inf], 0)
        num_cols = df_year.select_dtypes(include=[np.number]).columns
        df_year[num_cols] = df_year[num_cols].astype(np.float32, copy=False)
        frames.append(df_year)

    df = pd.concat(frames, axis=0, ignore_index=True)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def _build_store(data_dir: str, year_range: Tuple[int, int]) -> DataStore:
    """Build DataStore with in-memory cache plus on-disk pickle cache."""
    cache_version = "sm_mm_v1"
    cache_key = f"{data_dir}|{year_range[0]}-{year_range[1]}|{cache_version}"
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    # Disk cache
    cache_hash = hashlib.md5(
        f"{data_dir}_{year_range}_{cache_version}".encode()
    ).hexdigest()[:8]
    cache_path = os.path.join(data_dir, f"_cache_store_{cache_hash}.pkl")

    if os.path.exists(cache_path):
        try:
            print(f"📦 Loading cached DataStore from {cache_path}...")
            with open(cache_path, "rb") as f:
                store = pickle.load(f)
            # Validate cache schema (temporal_enc_dict replaces old doy_enc_dict).
            if not hasattr(store, "temporal_enc_dict") or store.temporal_enc_dict is None:
                print("⚠️ Cache outdated (missing temporal_enc_dict), rebuilding...")
            else:
                _DATA_CACHE[cache_key] = store
                return store
        except Exception as e:
            print(f"⚠️ Cache load failed: {e}, rebuilding...")

    df = _load_parquet_files(data_dir, year_range)

    # Normalize column aliases.
    new_cols = [COLUMN_ALIASES.get(c, c) for c in df.columns]
    if new_cols != list(df.columns):
        df.columns = new_cols

    # Validate required columns.
    required_cols = set(
        ["Grid_ID", "Date"] + STATIC_COLS + DYNAMIC_COLS_RAW + [TARGET_COL]
    )
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df[DYNAMIC_COLS_RAW + [TARGET_COL]] = df[DYNAMIC_COLS_RAW + [TARGET_COL]].fillna(
        0.0
    )

    # Unit unification: convert volumetric SM (~0-1) to water depth in mm (0-10 cm layer).
    # Example: 0.2 -> 20 mm (multiply by 100 mm).
    df[TARGET_COL] = (df[TARGET_COL].astype(np.float32) * SM_TO_MM_SCALE).astype(
        np.float32
    )

    # ---- Generate lagged variables ----
    print("🔧 Generating lag variables (Pre_lag1, PET_lag1)...")
    df = df.sort_values(["Grid_ID", "Date"])
    df["Pre_lag1"] = (
        df.groupby("Grid_ID")["Pre"].shift(1).fillna(0.0).astype(np.float32)
    )
    df["PET_lag1"] = (
        df.groupby("Grid_ID")["PET"].shift(1).fillna(0.0).astype(np.float32)
    )

    static_df = df.groupby("Grid_ID")[STATIC_COLS].first().copy()
    dynamic_df = df[["Grid_ID", "Date"] + DYNAMIC_COLS + [TARGET_COL]].copy()
    grid_ids = static_df.index.to_numpy()

    # ---- Vectorized construction of dyn_dict / y_dict / temporal_enc_dict ----
    dyn_dict: Dict[int, np.ndarray] = {}
    y_dict: Dict[int, np.ndarray] = {}
    temporal_enc_dict: Dict[int, np.ndarray] = {}

    dynamic_df_sorted = dynamic_df.sort_values(["Grid_ID", "Date"])
    dates_series = pd.to_datetime(dynamic_df_sorted["Date"])
    doy_values = dates_series.dt.dayofyear.values
    doy_phase = (2.0 * np.pi * doy_values / 365.0).astype(np.float32)
    dynamic_df_sorted["__doy_sin"] = np.sin(doy_phase)
    dynamic_df_sorted["__doy_cos"] = np.cos(doy_phase)

    # Year normalization in [0, 1] as input to the learnable temporal KAN encoder.
    years = dates_series.dt.year.values
    year_min, year_max = year_range
    if year_max == year_min:
        year_norm = np.zeros(len(years), dtype=np.float32)
    else:
        year_norm = ((years - year_min) / (year_max - year_min)).astype(np.float32)
    dynamic_df_sorted["__year_norm"] = year_norm

    grp_idx = dynamic_df_sorted.groupby("Grid_ID", sort=False)
    dyn_vals_all = dynamic_df_sorted[DYNAMIC_COLS].to_numpy(dtype=np.float32)
    y_vals_all = dynamic_df_sorted[[TARGET_COL]].to_numpy(dtype=np.float32)
    temporal_enc_all = dynamic_df_sorted[
        ["__doy_sin", "__doy_cos", "__year_norm"]
    ].to_numpy(dtype=np.float32)

    for gid, idx_arr in grp_idx.indices.items():
        dyn_dict[int(gid)] = dyn_vals_all[idx_arr]
        y_dict[int(gid)] = y_vals_all[idx_arr]
        temporal_enc_dict[int(gid)] = temporal_enc_all[idx_arr]

    del dynamic_df_sorted, dyn_vals_all, y_vals_all, temporal_enc_all

    # SM 95th percentile (used for physical parameter initialization).
    sm_values = df[TARGET_COL].values
    sm_values = sm_values[sm_values > 0]
    sm_p95 = float(np.percentile(sm_values, 95)) if len(sm_values) > 0 else 0.5
    print(f"📊 Data stats: SM 95th percentile = {sm_p95:.4f}")

    # ---- Load cluster labels ----
    cluster_dict: Dict[int, int] = {}
    if os.path.exists(CLUSTER_CSV_PATH):
        print(f"🔗 Loading cluster labels from {CLUSTER_CSV_PATH}...")
        cluster_df = pd.read_csv(CLUSTER_CSV_PATH)
        gids = cluster_df["Grid_ID"].values.astype(int)
        clusters = (cluster_df["Cluster"].values - 1).astype(int)
        cluster_dict = dict(zip(gids, clusters))
        print(f"   ✓ Loaded {len(cluster_dict)} cluster labels (K={N_CLUSTERS})")
    else:
        print(f"⚠️ Cluster CSV not found: {CLUSTER_CSV_PATH}, using zeros.")
        cluster_dict = {int(gid): 0 for gid in grid_ids}

    store = DataStore(
        df=df,
        static_df=static_df,
        dynamic_df=dynamic_df,
        grid_ids=grid_ids,
        dyn_dict=dyn_dict,
        temporal_enc_dict=temporal_enc_dict,
        y_dict=y_dict,
        sm_p95=sm_p95,
        cluster_dict=cluster_dict,
    )

    del df
    gc.collect()

    # Persist disk cache
    try:
        print(f"💾 Saving DataStore cache to {cache_path}...")
        with open(cache_path, "wb") as f:
            pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"⚠️ Cache save failed: {e}")

    _DATA_CACHE[cache_key] = store
    return store


# ====================== Dataset ======================


class SpatioTemporalDataset(Dataset):
    """Spatio-temporal dataset: each sample is the full series of one grid."""

    def __init__(
        self,
        data_dir: str,
        year_range: Tuple[int, int] = (2015, 2019),
        mode: str = "train",
        grid_ids: Optional[Sequence[int]] = None,
        static_scaler: Optional[StandardScaler] = None,
        dynamic_scaler: Optional[StandardScaler] = None,
    ) -> None:
        self.data_dir = data_dir
        self.year_range = year_range
        self.mode = mode

        self.store = _build_store(data_dir, year_range)
        self.static_df = self.store.static_df

        if grid_ids is None:
            self.grid_ids = self.store.grid_ids.tolist()
        else:
            self.grid_ids = list(grid_ids)

        # ---- Fit scalers on training-only data when not provided ----
        if static_scaler is None:
            static_scaler = StandardScaler()
            static_vals = self.static_df.loc[self.grid_ids, STATIC_COLS].to_numpy()
            static_scaler.fit(static_vals)
            static_scaler.scale_[static_scaler.scale_ == 0] = 1.0
        if dynamic_scaler is None:
            dynamic_scaler = StandardScaler()
            sample_ids = self.grid_ids[: min(len(self.grid_ids), 2000)]
            dyn_chunks = [
                self.store.dyn_dict[int(gid)]
                for gid in sample_ids
                if int(gid) in self.store.dyn_dict
            ]
            if dyn_chunks:
                dyn_vals = np.concatenate(dyn_chunks, axis=0)
                dynamic_scaler.fit(dyn_vals)
                del dyn_chunks, dyn_vals
                gc.collect()
            dynamic_scaler.scale_[dynamic_scaler.scale_ == 0] = 1.0

        self.static_scaler = static_scaler
        self.dynamic_scaler = dynamic_scaler

        # Precompute static features + cluster one-hot -> (N, 13)
        static_vals = self.static_df.loc[self.grid_ids, STATIC_COLS].to_numpy(
            dtype=np.float32
        )
        self.static_scaled = self.static_scaler.transform(static_vals).astype(
            np.float32
        )
        cluster_ids = np.array(
            [self.store.cluster_dict.get(int(gid), 0) for gid in self.grid_ids],
            dtype=np.int32,
        )
        self.cluster_onehot = np.eye(N_CLUSTERS, dtype=np.float32)[cluster_ids]
        self.static_with_cluster = np.concatenate(
            [self.static_scaled, self.cluster_onehot], axis=1
        )

        # Cache dynamic normalization statistics.
        self._dyn_mean = self.dynamic_scaler.mean_.astype(np.float32)
        self._dyn_scale = self.dynamic_scaler.scale_.astype(np.float32)

        # sm_scale is exposed for model/evaluation reference only; y is not normalized.
        # y always remains in mm physical units, consistent with ODE output.
        _raw_p95 = float(self.store.sm_p95)
        self.sm_scale: float = _raw_p95 if _raw_p95 > 0 else 1.0

    def __len__(self) -> int:
        return len(self.grid_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gid = int(self.grid_ids[idx])
        x_stat = self.static_with_cluster[idx]

        # Dynamic feature normalization (6 meteorological dimensions).
        raw_dyn = self.store.dyn_dict[gid]
        x_dyn = ((raw_dyn - self._dyn_mean) / self._dyn_scale).astype(np.float32)

        # Temporal encoding [DOY_sin, DOY_cos, Year_norm], not normalized.
        # Concatenate to tail -> (T, 9)
        temporal_enc = self.store.temporal_enc_dict[gid]
        x_dyn = np.concatenate([x_dyn, temporal_enc], axis=-1)

        # Keep y in mm units without normalization (same unit as ODE s_seq output).
        y = self.store.y_dict[gid]  # (T, 1), unit: mm
        return {
            "x_stat": torch.from_numpy(x_stat),
            "x_dyn": torch.from_numpy(x_dyn),
            "y": torch.from_numpy(y),
            "grid_id": gid,
            "lon": float(self.static_df.loc[gid, "Lon"]),
            "lat": float(self.static_df.loc[gid, "Lat"]),
        }

    def get_metadata(self) -> pd.DataFrame:
        meta = self.static_df.loc[self.grid_ids, ["Lon", "Lat"]].copy()
        meta.insert(0, "Grid_ID", meta.index)
        return meta.reset_index(drop=True)


# ====================== DataLoader Factory ======================


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build stratified hold-out dataloaders (Train 60% / Val 20% / Test 20%).

    Stratification uses Cluster ID so each subset retains all cluster types
    with matched proportions.
    """
    base_dataset = SpatioTemporalDataset(data_dir=data_dir, mode="train")
    grid_ids = np.asarray(base_dataset.grid_ids)

    cluster_ids = np.array(
        [base_dataset.store.cluster_dict.get(int(gid), 0) for gid in grid_ids],
        dtype=np.int32,
    )

    # First split: separate test set.
    remain_ids, test_ids = train_test_split(
        grid_ids, test_size=test_ratio, random_state=seed, stratify=cluster_ids
    )
    remain_clusters = np.array(
        [base_dataset.store.cluster_dict.get(int(gid), 0) for gid in remain_ids],
        dtype=np.int32,
    )

    # Second split: split remaining into train and validation.
    adj_val = val_ratio / (1.0 - test_ratio)
    train_ids, val_ids = train_test_split(
        remain_ids, test_size=adj_val, random_state=seed, stratify=remain_clusters
    )

    print(
        f"📦 Stratified Hold-Out: "
        f"Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}"
    )

    # Fit scalers on training subset only to avoid data leakage.
    static_scaler = StandardScaler()
    static_vals = base_dataset.static_df.loc[train_ids, STATIC_COLS].to_numpy()
    static_scaler.fit(static_vals)
    static_scaler.scale_[static_scaler.scale_ == 0] = 1.0

    dynamic_scaler = StandardScaler()
    mask = base_dataset.store.dynamic_df["Grid_ID"].isin(train_ids)
    dyn_vals = base_dataset.store.dynamic_df.loc[mask, DYNAMIC_COLS].to_numpy()
    dynamic_scaler.fit(dyn_vals)
    dynamic_scaler.scale_[dynamic_scaler.scale_ == 0] = 1.0

    # Build three subsets.
    train_ds = SpatioTemporalDataset(
        data_dir=data_dir,
        mode="train",
        grid_ids=train_ids,
        static_scaler=static_scaler,
        dynamic_scaler=dynamic_scaler,
    )
    val_ds = SpatioTemporalDataset(
        data_dir=data_dir,
        mode="val",
        grid_ids=val_ids,
        static_scaler=static_scaler,
        dynamic_scaler=dynamic_scaler,
    )
    test_ds = SpatioTemporalDataset(
        data_dir=data_dir,
        mode="test",
        grid_ids=test_ids,
        static_scaler=static_scaler,
        dynamic_scaler=dynamic_scaler,
    )

    # DataLoader runtime configuration.
    is_windows = platform.system().lower().startswith("win")
    actual_workers = 0 if is_windows else num_workers
    loader_kw: Dict = {"pin_memory": torch.cuda.is_available()}
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