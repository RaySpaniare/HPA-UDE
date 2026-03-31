"""Dataset configuration: field definitions, constants, and data structures."""
import glob
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


# -------------------------
# Data field groups (static / dynamic / target)
# -------------------------
STATIC_COLS = [
    "Clay",
    "Sand",
    "BD",
    "OC",
    "Porosity",
    "Dem",
    "Slope",
    "Lon",
    "Lat",
]


def _resolve_cluster_csv_path() -> str:
    """Resolve cluster CSV path from env override or workspace search."""
    env_path = os.environ.get("HPA_UDE_CLUSTER_CSV", "").strip()
    if env_path:
        return env_path

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    matches = sorted(
        glob.glob(os.path.join(repo_root, "**", "Clustering_Results.csv"), recursive=True)
    )
    return matches[0] if matches else "Clustering_Results.csv"


# Cluster results CSV path (K=4)
CLUSTER_CSV_PATH = _resolve_cluster_csv_path()
N_CLUSTERS = 4  # Number of clusters

# Dynamic variables: 4 original + 2 lagged = 6 normalized columns
# [Pre, PET, LST, LAI, Pre_lag1, PET_lag1]
DYNAMIC_COLS_RAW = ["Pre", "PET", "LST", "LAI"]  # Original variables
DYNAMIC_COLS = ["Pre", "PET", "LST", "LAI", "Pre_lag1", "PET_lag1"]  # Includes lagged features
TARGET_COL = "SM"

# ---- Dimension split (meteorological vs temporal) ----
METEO_DIM = 6  # Number of normalized meteorological columns = len(DYNAMIC_COLS)
TEMPORAL_RAW_DIM = 3  # Raw temporal features [DOY_sin, DOY_cos, Year_norm]
TEMPORAL_EMBED_DIM = 4  # Learnable KAN temporal embedding output dimension

# ---- Learnable KAN Temporal Encoder ----
# Replace hardcoded sin/cos with KAN-learned multi-scale temporal representation.
# dataset.py provides [DOY_sin, DOY_cos, Year_norm], then the model encodes them via KAN.
# Final dynamic_dim = 6 (normalized columns) + 3 (raw temporal) = 9
DYNAMIC_DIM = 9  # Actual dynamic input dimension consumed by the model

# Column aliases for handling spelling/casing differences
COLUMN_ALIASES = {
    "porpsity": "Porosity",
    "porosity": "Porosity",
}


@dataclass
class DataStore:
    """Data container that stores raw tables and efficient indices."""

    # Raw merged table (multi-year parquet)
    df: pd.DataFrame
    # Static attributes table (deduplicated by Grid_ID)
    static_df: pd.DataFrame
    # Dynamic sequence table (sorted by Date)
    dynamic_df: pd.DataFrame
    # Grid index array
    grid_ids: np.ndarray
    # Dynamic driver dict: gid -> (T, 6) [raw 6 columns before normalization]
    dyn_dict: Dict[int, np.ndarray]
    # Temporal encoding dict: gid -> (T, 3) [DOY_sin, DOY_cos, Year_norm]
    # Raw inputs for the learnable KAN temporal encoder
    temporal_enc_dict: Dict[int, np.ndarray]
    # Target sequence dict: gid -> (T, 1)
    y_dict: Dict[int, np.ndarray]
    # 95th percentile of SM (used for s_max initialization)
    sm_p95: float
    # Cluster label dict: gid -> cluster_id (0-3)
    cluster_dict: Optional[Dict[int, int]] = None