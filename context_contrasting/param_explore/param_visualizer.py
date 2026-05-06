from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

TARGET_ORDER = [
    "un_un",
    "un_FB",
    "FF_un",
    "FF_FB_broad",
    "FF_FB_narrow_familiar",
    "FF_FB_narrow_novel",
    "FB_FB",
]
TARGET_COLORS = {
    "un_un": "#7f7f7f",
    "un_FB": "#4c78a8",
    "FF_un": "#f58518",
    "FF_FB_broad": "#54a24b",
    "FF_FB_narrow_familiar": "#e45756",
    "FF_FB_narrow_novel": "#72b7b2",
    "FB_FB": "#b279a2",
}


def _style_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        fontsize=9,
        frameon=False,
        loc="best",
    )


def _scatter_2d(ax, df: pd.DataFrame, x: str, y: str, *, alpha: float = 0.55, title: str | None = None) -> None:
    for target in TARGET_ORDER:
        subset = df.loc[df["assigned_target"].eq(target)]
        if subset.empty:
            continue
        ax.scatter(
            subset[x],
            subset[y],
            s=12,
            alpha=alpha,
            c=TARGET_COLORS[target],
            label=target,
            edgecolors="none",
        )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title is not None:
        ax.set_title(title)
    _style_legend(ax)


def _scatter_3d(ax, df: pd.DataFrame, x: str, y: str, z: str, *, alpha: float = 0.4, title: str | None = None) -> None:
    for target in TARGET_ORDER:
        subset = df.loc[df["assigned_target"].eq(target)]
        if subset.empty:
            continue
        ax.scatter(
            subset[x],
            subset[y],
            subset[z],
            s=12,
            alpha=alpha,
            c=TARGET_COLORS[target],
            label=target,
            depthshade=False,
        )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    if title is not None:
        ax.set_title(title)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_space_2d(df: pd.DataFrame, output_dir: Path, suffix: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    pairs = [
        ("w_ff_0", "w_fb_0", "Parameter Space: familiar-biased excitatory axis"),
        ("w_ff_1", "w_fb_1", "Parameter Space: novel-biased excitatory axis"),
        ("w_lat_0", "w_pv_lat_0", "Parameter Space: familiar inhibitory axis"),
        ("w_lat_1", "w_pv_lat_1", "Parameter Space: novel inhibitory axis"),
    ]
    for ax, (x, y, title) in zip(axes.flat, pairs, strict=True):
        _scatter_2d(ax, df, x, y, title=title)
    _save(fig, output_dir / f"parameter_space_2d_{suffix}.png")


def plot_parameter_space_3d(df: pd.DataFrame, output_dir: Path, suffix: str) -> None:
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    _scatter_3d(
        ax1,
        df,
        "w_ff_0",
        "w_fb_0",
        "w_lat_0",
        title="3D parameter space: familiar axis",
    )
    _scatter_3d(
        ax2,
        df,
        "w_ff_1",
        "w_fb_1",
        "w_lat_1",
        title="3D parameter space: novel axis",
    )
    _style_legend(ax2)
    _save(fig, output_dir / f"parameter_space_3d_{suffix}.png")


def plot_naive_expert_scalar_space(df: pd.DataFrame, output_dir: Path, suffix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _scatter_2d(
        axes[0],
        df,
        "familiar_naive_state",
        "familiar_expert_state",
        title="Familiar scalar space: naive vs expert",
    )
    _scatter_2d(
        axes[1],
        df,
        "novel_naive_state",
        "novel_expert_state",
        title="Novel scalar space: naive vs expert",
    )
    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    _save(fig, output_dir / f"scalar_space_naive_expert_{suffix}.png")


def plot_ff_fb_scalar_space(df: pd.DataFrame, output_dir: Path, suffix: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    configs = [
        ("familiar_naive_ff_scalar", "familiar_naive_fb_scalar", "Familiar naive FF vs FB scalar"),
        ("familiar_expert_ff_scalar", "familiar_expert_fb_scalar", "Familiar expert FF vs FB scalar"),
        ("novel_naive_ff_scalar", "novel_naive_fb_scalar", "Novel naive FF vs FB scalar"),
        ("novel_expert_ff_scalar", "novel_expert_fb_scalar", "Novel expert FF vs FB scalar"),
    ]
    for ax, (x, y, title) in zip(axes.flat, configs, strict=True):
        _scatter_2d(ax, df, x, y, title=title)
    _save(fig, output_dir / f"scalar_space_ff_fb_{suffix}.png")


def plot_scalar_space_3d(df: pd.DataFrame, output_dir: Path, suffix: str) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _scatter_3d(
        ax,
        df,
        "familiar_naive_state",
        "familiar_expert_state",
        "novel_expert_state",
        title="3D scalar space: familiar naive, familiar expert, novel expert",
    )
    _style_legend(ax)
    _save(fig, output_dir / f"scalar_space_3d_{suffix}.png")


def _load_input_frame(results_dir: Path, use_region_members: bool) -> tuple[pd.DataFrame, str]:
    if use_region_members and (results_dir / "stable_region_members.csv").exists():
        return pd.read_csv(results_dir / "stable_region_members.csv"), "stable_regions"
    return pd.read_csv(results_dir / "combined_candidates.csv"), "all_candidates"


def build_plots(results_dir: Path, *, use_region_members: bool = False) -> None:
    df, suffix = _load_input_frame(results_dir, use_region_members=use_region_members)
    output_dir = results_dir / "plots"
    plot_parameter_space_2d(df, output_dir, suffix)
    plot_parameter_space_3d(df, output_dir, suffix)
    plot_naive_expert_scalar_space(df, output_dir, suffix)
    plot_ff_fb_scalar_space(df, output_dir, suffix)
    plot_scalar_space_3d(df, output_dir, suffix)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize sampled parameter regions and scalar transition spaces from "
            "the parameter exploration outputs."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to a parameter exploration results directory.",
    )
    parser.add_argument(
        "--use-region-members",
        action="store_true",
        help="Plot only the stable region members instead of all sampled candidates.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_plots(
        Path(args.results_dir),
        use_region_members=args.use_region_members,
    )


if __name__ == "__main__":
    main()
