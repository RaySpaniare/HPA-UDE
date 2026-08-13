# -*- coding: utf-8 -*-
r'''
@File    :   plot_spatial_holdout_boxplots.py
@Time    :   2026-06-27
@Desc    :   Summarize test-set metrics from the five HPA-UDE spatial-block
             holdout folds and create a 2 x 2 boxplot figure for R2, KGE, RMSE,
             and Bias. The script reads fold_*/results/metrics_summary.csv and
             strictly retains rows with Split == "Test" so that training and
             validation grid cells are excluded from spatial-transfer estimates.
             Each panel shows fold-specific distributions using translucent boxes,
             jittered grid-cell points, mean markers, medians, and one-sided KDE
             outlines. Axis limits are derived from the observed distributions.
             RMSE and Bias are used directly in the volumetric soil-moisture units
             exported by evaluate.py. Outputs include a 300-DPI JPG, a Type42 PDF,
             and CSV files containing the plotted data and summary statistics.
@Notice  :   This script reads existing results only; it does not modify training
             outputs, recompute predictions, or change model parameters. Input and
             output paths are resolved relative to this script.
'''

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


ROOT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = ROOT_DIR / "figures"
METRICS = ["R2", "KGE", "RMSE", "Bias"]
FOLD_ORDER = [0, 1, 2, 3, 4]
FOLD_LABELS = [f"Fold {fold}" for fold in FOLD_ORDER]
COLORS = ["#4E79A7", "#59A14F", "#F28E2B", "#B07AA1", "#76B7B2"]
SCATTER_COLORS = ["#9EB9D4", "#A8D49F", "#F6C28B", "#D8B8D0", "#B8DDD9"]
OUTPUT_STEM = "spatial_holdout_test_metrics_boxplot_2x2"


def disable_sci_axis(ax) -> None:
    """Disable scientific notation and offset text for stable readable axes."""
    try:
        ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    except Exception:
        pass
    try:
        ax.xaxis.get_offset_text().set_visible(False)
        ax.yaxis.get_offset_text().set_visible(False)
    except Exception:
        pass


def load_test_metrics(root_dir: Path) -> pd.DataFrame:
    """Load per-grid Test metrics from the five fold-specific summary files."""
    frames: List[pd.DataFrame] = []
    for fold in FOLD_ORDER:
        metrics_path = root_dir / f"fold_{fold}" / "results" / "metrics_summary.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
        df = pd.read_csv(metrics_path)
        missing_cols = {"Split", *METRICS} - set(df.columns)
        if missing_cols:
            raise ValueError(f"{metrics_path} missing columns: {sorted(missing_cols)}")
        test_df = df.loc[df["Split"] == "Test", ["Grid_ID", "Split", *METRICS]].copy()
        test_df.insert(0, "Fold", fold)
        frames.append(test_df)
    plot_df = pd.concat(frames, ignore_index=True)
    for metric in METRICS:
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    return plot_df


def build_summary(plot_df: pd.DataFrame) -> pd.DataFrame:
    """Compute fold-level summary statistics for auditing and reporting."""
    rows: List[Dict[str, float]] = []
    for metric in METRICS:
        for fold in FOLD_ORDER:
            values = plot_df.loc[plot_df["Fold"] == fold, metric].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "Metric": metric,
                    "Fold": fold,
                    "N": int(values.size),
                    "Mean": float(np.mean(values)) if values.size else np.nan,
                    "Median": float(np.median(values)) if values.size else np.nan,
                    "Std": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
                    "Q1": float(np.percentile(values, 25)) if values.size else np.nan,
                    "Q3": float(np.percentile(values, 75)) if values.size else np.nan,
                    "Min": float(np.min(values)) if values.size else np.nan,
                    "Max": float(np.max(values)) if values.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def values_by_fold(plot_df: pd.DataFrame, metric: str) -> List[np.ndarray]:
    """Return finite metric values in fold order."""
    grouped = []
    for fold in FOLD_ORDER:
        values = plot_df.loc[plot_df["Fold"] == fold, metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        grouped.append(values)
    return grouped


def compute_ylim(values_list: List[np.ndarray], metric: str) -> Tuple[float, float]:
    """Set axis limits from the observed distribution and metric semantics."""
    if metric == "R2":
        return -1.0, 1.0
    values = np.concatenate([v for v in values_list if v.size > 0])
    if values.size == 0:
        return 0.0, 1.0
    lower = float(np.nanpercentile(values, 1))
    upper = float(np.nanpercentile(values, 99))
    data_min = float(np.nanmin(values))
    data_max = float(np.nanmax(values))
    lower = min(lower, data_min)
    upper = max(upper, data_max)
    if metric in {"R2", "KGE"}:
        lower = min(lower, 0.0)
        upper = min(max(upper, 1.0), 1.1)
    if metric == "Bias":
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
    span = upper - lower
    if span <= 0:
        span = max(abs(upper), 1.0) * 0.1
    pad = span * 0.12
    return lower - pad, upper + pad


def draw_kde(ax, x_index: int, values: np.ndarray) -> None:
    """Draw a one-sided KDE outline and skip degenerate samples."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size <= 2 or np.nanstd(finite_values) == 0:
        return
    try:
        kde = gaussian_kde(finite_values)
        y_values = np.linspace(float(finite_values.min()), float(finite_values.max()), 180)
        density = kde(y_values)
        if np.nanmax(density) <= 0:
            return
        density_scaled = density / np.nanmax(density) * 0.22
        ax.plot(
            x_index + density_scaled,
            y_values,
            color="#222222",
            linestyle="--",
            linewidth=0.9,
            alpha=0.85,
            zorder=4,
        )
    except Exception:
        return


def draw_metric_panel(ax, plot_df: pd.DataFrame, metric: str, panel_label: str) -> None:
    """Draw one metric panel with five folds, points, means, and KDE outlines."""
    rng = np.random.default_rng(20260627)
    positions = np.arange(len(FOLD_ORDER), dtype=float)
    grouped_values = values_by_fold(plot_df, metric)
    box = ax.boxplot(
        grouped_values,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "red",
            "markersize": 4.8,
            "markeredgecolor": "black",
            "markeredgewidth": 0.5,
            "zorder": 9,
        },
        medianprops={"color": "black", "linewidth": 1.5, "linestyle": "--"},
        whiskerprops={"color": "#333333", "linewidth": 1.0},
        capprops={"color": "#333333", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markersize": 2.2,
            "markerfacecolor": "#555555",
            "markeredgecolor": "none",
            "alpha": 0.25,
        },
    )
    for patch, color in zip(box["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_edgecolor("#333333")
        patch.set_alpha(0.72)
        patch.set_linewidth(1.0)

    for i, values in enumerate(grouped_values):
        draw_kde(ax, i, values)
        if values.size == 0:
            continue
        scatter_values = values
        if values.size > 1200:
            idx = rng.choice(values.size, size=1200, replace=False)
            scatter_values = values[idx]
        x = rng.normal(i, 0.045, size=scatter_values.size)
        ax.scatter(
            x,
            scatter_values,
            s=5,
            alpha=1.0,
            color=SCATTER_COLORS[i],
            edgecolors="none",
            rasterized=False,
            clip_on=True,
            zorder=1,
        )

    if metric == "Bias":
        ax.axhline(0.0, color="#A00000", linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(FOLD_LABELS, fontsize=9)
    ylabel = metric if metric in {"R2", "KGE"} else f"{metric} (cm3/cm3)"
    ax.set_ylabel(ylabel, fontsize=10.5, fontweight="bold")
    ax.set_title(f"{panel_label} {metric}", loc="left", fontsize=11.5, fontweight="bold")
    ax.set_ylim(*compute_ylim(grouped_values, metric))
    ax.grid(axis="y", alpha=0.22, linestyle="-", linewidth=0.45)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.9)
    ax.tick_params(axis="y", labelsize=9)
    disable_sci_axis(ax)


def save_figure(fig, output_stem: Path) -> None:
    """Export matching JPG and PDF versions of the figure."""
    fig.savefig(
        output_stem.with_suffix(".jpg"),
        dpi=300,
        format="jpeg",
        bbox_inches="tight",
        pil_kwargs={"quality": 95},
    )
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def create_boxplot_figure(plot_df: pd.DataFrame, output_dir: Path) -> None:
    """Create the 2 x 2 spatial-holdout test-metric figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=False)
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, metric, panel_label in zip(axes.ravel(), METRICS, panel_labels):
        draw_metric_panel(ax, plot_df, metric, panel_label)

    mean_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="white",
        markerfacecolor="red",
        markeredgecolor="black",
        markersize=5,
        label="Mean",
    )
    median_handle = Line2D([0], [0], color="black", linewidth=1.5, linestyle="--", label="Median")
    kde_handle = Line2D([0], [0], color="#222222", linewidth=0.9, linestyle="--", label="KDE")
    fig.legend(
        handles=[mean_handle, median_handle, kde_handle],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), w_pad=2.0, h_pad=2.0)
    save_figure(fig, output_dir / OUTPUT_STEM)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = load_test_metrics(ROOT_DIR)
    summary_df = build_summary(plot_df)
    plot_df.to_csv(FIGURE_DIR / "spatial_holdout_test_metrics_long.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(FIGURE_DIR / "spatial_holdout_test_metrics_summary.csv", index=False, encoding="utf-8-sig")
    create_boxplot_figure(plot_df, FIGURE_DIR)
    print(f"DONE figure saved to: {FIGURE_DIR / (OUTPUT_STEM + '.jpg')}")
    print(f"DONE figure saved to: {FIGURE_DIR / (OUTPUT_STEM + '.pdf')}")
    print(f"DONE summary saved to: {FIGURE_DIR / 'spatial_holdout_test_metrics_summary.csv'}")


if __name__ == "__main__":
    main()
