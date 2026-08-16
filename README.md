# HPA-UDE (Hydrology Physics-Aware Universal Differential Equation)

This repository contains the source code used for soil-moisture modeling with the
Hybrid Physics-AI Universal Differential Equation (HPA-UDE) framework. The model
combines a static hypernetwork, a FiLM-conditioned Mamba temporal encoder, an
ODE water-balance core, and optional symbolic regression.

## Data availability

The five annual pre-processed spatiotemporal datasets used for model training and
evaluation (2015-2019) are openly available on Zenodo:

- https://doi.org/10.5281/zenodo.19343205

The required cluster-label file (`Clustering_Results.csv`) is openly available
in a separate Zenodo record:

- https://doi.org/10.5281/zenodo.21918634

Both records and their files can be accessed without logging in.

## Repository structure

- `train.py`: main training entry point.
- `evaluate.py`: Train/Validation/Test evaluation and report export.
- `buffered_spatial_holdout.py`: five-fold spatial-block holdout experiment.
- `plot_spatial_holdout_boxplots.py`: Figure S10 spatial-holdout plots.
- `dataset.py`: data loading, caching, normalization, and stratified splitting.
- `dataset_config.py`: dataset fields and constants.
- `model.py`: `GeoHyperNet`, `_ODEFunc`, and `HPA_UDE_Model`.
- `model_components.py`: neural-network and physics components.
- `losses.py`: physical and robust objective functions.
- `trainers.py`: epoch-level training and validation.
- `drought_indices_optimized.py`: drought-index preprocessing.
- `symbolic_regression.py`: KAN symbolic regression and formula extraction.
- `requirements.txt` and `environment.yml`: pinned environments.

## Data layout

Place the following files in one directory:

```text
/path/to/data/
  Soil_Moisture_Data_2015.parquet
  Soil_Moisture_Data_2016.parquet
  Soil_Moisture_Data_2017.parquet
  Soil_Moisture_Data_2018.parquet
  Soil_Moisture_Data_2019.parquet
  Clustering_Results.csv
```

Required columns are:

- Static: `Clay`, `Sand`, `BD`, `OC`, `Porosity`, `Dem`, `Slope`,
  `Lon`, and `Lat`.
- Dynamic: `Pre`, `PET`, `LST`, and `LAI`.
- Keys and target: `Grid_ID`, `Date`, and `SM`.

If the cluster CSV is stored elsewhere, set `HPA_UDE_CLUSTER_CSV` to its
absolute path.

## Installation

Python 3.11.15 and a CUDA-enabled PyTorch build were used for the deposited
environment. Linux or WSL2 with an NVIDIA CUDA toolchain is recommended because
`mamba-ssm` may require platform-specific compiled extensions.

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate spatial-holdout-hpa-ude
```

### Option 2: pip

Create and activate a clean Python 3.11 environment, then run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm that the paper-configuration dependencies are available:

```bash
python -c "import torch, mamba_ssm, torchdiffeq, numba, pandarallel, sympy; print('dependency check: OK')"
```

The code contains limited fallbacks for diagnostic use, but the reported
configuration requires `mamba-ssm` and `torchdiffeq`. Do not use the fallback
Mamba encoder or fallback ODE integration when reproducing the reported results.

## Standard train/evaluate workflow

Run all commands from the repository root.

### 1. Train and create the checkpoint

```bash
python train.py --data_dir /path/to/data --device cuda
```

Training creates the checkpoint expected by `evaluate.py` at:

```text
checkpoints/best_model.pth
```

The checkpoint is a generated training artifact and is not stored in Git. To
obtain it, run the training command above. To evaluate a checkpoint stored at a
different location, pass its path explicitly with `--checkpoint`.

### 2. Evaluate

```bash
python evaluate.py \
  --data_dir /path/to/data \
  --checkpoint checkpoints/best_model.pth \
  --device cuda
```

Evaluation outputs are written to `results/`, including
`metrics_summary.csv`, `metrics_summary.parquet`, and, unless disabled,
`timeseries_predictions.parquet`.

## Five-fold spatial-block holdout

The spatial experiment orders grid cells along the dominant spatial extent and
divides them into five contiguous blocks. In each fold, one block is held out
for testing, the adjacent block is used for validation, and the other three
blocks are used for training. Static and dynamic scalers are fitted only on the
training cells in each fold.

Inspect and cache the spatial partition without training:

```bash
python buffered_spatial_holdout.py \
  --data_dir /path/to/data \
  --output_root . \
  --prepare_only
```

Run the complete five-fold experiment:

```bash
python buffered_spatial_holdout.py \
  --data_dir /path/to/data \
  --output_root . \
  --device cuda
```

To run selected folds, add, for example, `--folds 0 1`. Each fold writes its
checkpoint, predictions, loss history, and metrics under
`fold_<n>/checkpoints/` and `fold_<n>/results/`.

After all five folds have completed, generate Figure S10 and its plotting data:

```bash
python plot_spatial_holdout_boxplots.py
```

The plotting script reads `fold_0/results/metrics_summary.csv` through
`fold_4/results/metrics_summary.csv` and writes JPG, PDF, and CSV outputs to
`figures/`.

## Optional preprocessing and symbolic regression

```bash
python drought_indices_optimized.py --data_dir /path/to/data
python symbolic_regression.py \
  --data_dir /path/to/data \
  --model_path checkpoints/best_model.pth
```

## Model outputs and units

- Soil-moisture states (`pred_sm` and `pred_phy`): mm.
- Flux terms (`E_act`, `D_flux`, precipitation, and PET): mm/day.
- `evaluate.py` reports RMSE, MAE, Bias, and ubRMSE in volumetric water
  content scale (cm3/cm3) by dividing the internal millimetre-scale errors by
  100.

## Reproducibility notes

- The standard Train/Validation/Test workflow uses cluster-stratified splitting.
- The spatial-transfer experiment uses the five-fold contiguous spatial-block
  design described above.
- Random seeds are controlled through `--seed` (default: 42).
- Batch-size benchmarking is enabled in the spatial-holdout workflow; use
  `--fixed_batch_size` when the reported fixed batch size must be retained.
- Run `python -m compileall .` after downloading the repository to verify that
  all Python modules are syntactically valid.
- On Windows, data-loader workers are reduced for stability. Linux is
  recommended for the exact Mamba/CUDA environment.
