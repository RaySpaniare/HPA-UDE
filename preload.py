"""
GPU preloaded dataset (Windows-oriented acceleration).

Preloads all samples into GPU memory to remove CPU-GPU transfer bottlenecks.
Best suited for Windows (num_workers=0) when dataset size fits VRAM.
"""
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import SpatioTemporalDataset, get_dataloaders


class PreloadedDataset(Dataset):
    """GPU-preloaded dataset: move all samples to VRAM in one pass."""

    def __init__(
        self,
        base_dataset: SpatioTemporalDataset,
        device: Union[str, torch.device] = "cuda",
        verbose: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.base_dataset = base_dataset
        n = len(base_dataset)

        if verbose:
            print(f"🚀 Preloading {n} samples to {self.device}...")

        x_stat_list, x_dyn_list, y_list = [], [], []
        grid_ids, lons, lats = [], [], []

        for i in range(n):
            sample = base_dataset[i]
            x_stat_list.append(sample["x_stat"])
            x_dyn_list.append(sample["x_dyn"])
            y_list.append(sample["y"])
            grid_ids.append(sample["grid_id"])
            lons.append(sample["lon"])
            lats.append(sample["lat"])

        self.x_stat = torch.stack(x_stat_list).to(self.device)
        self.x_dyn = torch.stack(x_dyn_list).to(self.device)
        self.y = torch.stack(y_list).to(self.device)
        self.grid_ids = grid_ids
        self.lons = lons
        self.lats = lats

        # Inherit scalers for downstream access.
        self.static_scaler = base_dataset.static_scaler
        self.dynamic_scaler = base_dataset.dynamic_scaler

        if verbose:
            mem_mb = (
                (self.x_stat.nbytes + self.x_dyn.nbytes + self.y.nbytes) / 1024 / 1024
            )
            print(f"✅ Preloaded! GPU memory used: ~{mem_mb:.1f} MB")

    def __len__(self) -> int:
        return self.x_stat.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x_stat": self.x_stat[idx],
            "x_dyn": self.x_dyn[idx],
            "y": self.y[idx],
            "grid_id": self.grid_ids[idx],
            "lon": self.lons[idx],
            "lat": self.lats[idx],
        }

    def get_metadata(self) -> pd.DataFrame:
        return self.base_dataset.get_metadata()


def get_dataloaders_preloaded(
    data_dir: str,
    batch_size: int = 64,
    device: Union[str, torch.device] = "cuda",
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build GPU-preloaded dataloaders.

    Data is moved to GPU up-front, and loading uses single-process mode.
    """
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,
        seed=seed,
    )

    device = torch.device(device)
    train_pre = PreloadedDataset(train_loader.dataset, device=device)
    val_pre = PreloadedDataset(val_loader.dataset, device=device, verbose=False)
    test_pre = PreloadedDataset(test_loader.dataset, device=device, verbose=False)

    kw = {"num_workers": 0}
    train_loader = DataLoader(
        train_pre, batch_size=batch_size, shuffle=True, drop_last=True, **kw
    )
    val_loader = DataLoader(val_pre, batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test_pre, batch_size=batch_size, shuffle=False, **kw)

    return train_loader, val_loader, test_loader