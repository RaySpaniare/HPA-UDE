# HPA-UDE (Hydrology Physics-Aware Universal Differential Equation)

This folder contains a physics-aware deep learning workflow for soil moisture modeling.
The model combines:

- A static hyper-network (`GeoHyperNet`) to generate physically constrained parameters.
- A dynamic backbone (`FiLM + Mamba`) for temporal representation learning.
- An ODE-based water balance core for physically consistent prediction.
- Optional symbolic regression for interpretability.

## Project Structure

- `train.py`: Main training entrypoint (direct finetuning with physics priors).
- `evaluate.py`: Evaluation pipeline for Train/Val/Test splits and report export.
- `dataset.py`: Data loading, caching, normalization, and stratified hold-out split.
- `dataset_config.py`: Dataset field definitions and constants.
- `model.py`: Main model definitions (`GeoHyperNet`, `_ODEFunc`, `HPA_UDE_Model`).
- `model_components.py`: Core neural/physics components.
- `losses.py`: Physical and robust objective functions.
- `trainers.py`: Epoch-level training/validation loops.
- `preload.py`: Optional GPU preloading dataloader for Windows acceleration.
- `drought_indices_optimized.py`: Optimized drought index preprocessing (DOY, SPEI, SMDI).
- `symbolic_regression.py`: Native KAN symbolic regression and interpretability tools.
- `utils.py`: Shared utility helpers and hydrological metrics.

## Data Expectations

The training/evaluation dataset is expected as annual parquet files:

- `Soil_Moisture_Data_2015.parquet`
- `Soil_Moisture_Data_2016.parquet`
- `...`

Required columns include:

- Static: `Clay`, `Sand`, `BD`, `OC`, `Porosity`, `Dem`, `Slope`, `Lon`, `Lat`
- Dynamic: `Pre`, `PET`, `LST`, `LAI`
- Keys/target: `Grid_ID`, `Date`, `SM`

Cluster labels are loaded from `Clustering_Results.csv`.
You can override the cluster file path via environment variable:

- `HPA_UDE_CLUSTER_CSV`

## Installation

Create and activate a Python environment, then install dependencies (example):

```bash
pip install torch torchvision torchaudio
pip install numpy pandas pyarrow scikit-learn matplotlib tqdm scipy
pip install torchdiffeq mamba-ssm numba pandarallel sympy
```

Notes:

- `torchdiffeq`, `mamba-ssm`, `numba`, and `pandarallel` are optional but recommended.
- The code has built-in fallbacks when some optional packages are unavailable.

## Quick Start

### 1. Train

```bash
python train.py --data_dir /path/to/parquet_folder --device cuda
```

Useful options:

- `--use_preload`: preload dataset to GPU memory (if VRAM is sufficient).
- `--use_compile`: enable modular `torch.compile` acceleration.
- `--lambda_mass`, `--lambda_flux`, `--lambda_mono`: physics-loss weights.
- `--physics_warmup`: pure MSE warmup epochs before enabling physics losses.

### 2. Evaluate

```bash
python evaluate.py --data_dir /path/to/parquet_folder --checkpoint checkpoints/best_model.pth
```

Outputs are saved to `results/`:

- `metrics_summary.parquet`
- `metrics_summary.csv`
- `timeseries_predictions.parquet` (unless `--skip_timeseries`)
- `evaluation_report.txt`

### 3. Drought Index Preprocessing (Optional)

```bash
python drought_indices_optimized.py --data_dir /path/to/parquet_folder
```

### 4. Symbolic Regression / Interpretability (Optional)

```bash
python symbolic_regression.py --data_dir ./drought_analysis_gpu --model_path checkpoints/best_model.pth
```

## Model Outputs and Units

Core physical unit convention:

- Soil moisture state (`pred_sm`, `pred_phy`): `mm`
- Flux terms (`E_act`, `D_flux`, precipitation, PET): `mm/day`

Metric exports in `evaluate.py` convert error metrics (`RMSE`, `MAE`, `Bias`, `ubRMSE`) to volumetric water content scale (`cm3/cm3`) by dividing by 100 for reporting.

## Reproducibility

- Seed control is configured in `train.py` (`--seed`).
- Train/Val/Test split is stratified by cluster ID for balanced distribution.

## Troubleshooting

- If cluster CSV is not found, the pipeline falls back to a zero-cluster assignment.
- On Windows, dataloader workers are forced to `0` by default for stability.
- If mixed precision becomes unstable, disable AMP via `--no_amp`.
