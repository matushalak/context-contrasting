from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

import context_contrasting.data_analysis.transitions_helpers as th


DEFAULT_RESPONSE_THRESHOLD = 0.3

COUNT_COLORS = (
    "#9E9E9E",
    "#0072B2",
    "#56B4E9",
    "#009E73",
    "#E69F00",
    "#D55E00",
    "#CC79A7",
    "#332288",
    "#88CCEE",
)

DISPLAY_LABELS = {
    "task": "Task",
    "expert_familiar": "Expert familiar",
    "expert_novel": "Expert novel",
}

TARGET_LABELS = {
    "task": "Task",
    "expert_familiar": "Expert",
    "expert_novel": "Expert",
}


def _threshold_tag(threshold: float) -> str:
    return f"gt_{threshold:g}".replace(".", "_").replace("-", "minus_")


def _count_cmap(max_count: int) -> tuple[ListedColormap, BoundaryNorm]:
    if max_count < len(COUNT_COLORS):
        colors = COUNT_COLORS[: max_count + 1]
    else:
        colors = plt.cm.tab20(np.linspace(0.0, 1.0, max_count + 1))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, max_count + 1.5, 1.0), cmap.N)
    return cmap, norm


def build_naive_responsiveness_counts(
    transition_table: pd.DataFrame,
    *,
    response_threshold: float = DEFAULT_RESPONSE_THRESHOLD,
) -> pd.DataFrame:
    """Count images with a naive non-occluded response above threshold per neuron."""
    naive = transition_table.loc[transition_table["stage"].astype(str) == "Pre"].copy()
    naive = naive[
        [
            "neuron_idx",
            "image_group",
            "image_idx_original",
            "image_idx_within_group",
            "NO",
        ]
    ].drop_duplicates()
    naive["naive_responsive"] = naive["NO"] > response_threshold

    total_counts = (
        naive.groupby("neuron_idx", as_index=False)
        .agg(
            naive_responsive_image_count=("naive_responsive", "sum"),
            naive_total_image_count=("naive_responsive", "size"),
            naive_mean_NO=("NO", "mean"),
            naive_max_NO=("NO", "max"),
        )
    )

    group_counts = (
        naive.groupby(["neuron_idx", "image_group"])["naive_responsive"]
        .sum()
        .unstack(fill_value=0)
        .astype(int)
    )
    group_counts.columns = [f"naive_responsive_{group}_count" for group in group_counts.columns]

    group_totals = (
        naive.groupby(["neuron_idx", "image_group"])
        .size()
        .unstack(fill_value=0)
        .astype(int)
    )
    group_totals.columns = [f"naive_total_{group}_count" for group in group_totals.columns]

    counts = total_counts.merge(group_counts.reset_index(), on="neuron_idx", how="left")
    counts = counts.merge(group_totals.reset_index(), on="neuron_idx", how="left")

    count_cols = [col for col in counts.columns if col.endswith("_count")]
    counts[count_cols] = counts[count_cols].fillna(0).astype(int)
    counts["response_threshold"] = response_threshold
    return counts


def build_colored_summary(
    transition_table: pd.DataFrame,
    *,
    image_group: str,
    target_stage: str,
    response_threshold: float,
    sector_threshold: float,
) -> pd.DataFrame:
    summary = th.build_mean_summary(
        transition_table,
        image_group=image_group,
        pre_stage="Pre",
        target_stage=target_stage,
        threshold=sector_threshold,
    )
    counts = build_naive_responsiveness_counts(
        transition_table,
        response_threshold=response_threshold,
    )
    colored = summary.merge(counts, on="neuron_idx", how="left", validate="one_to_one")
    if colored["naive_responsive_image_count"].isna().any():
        missing = colored.loc[colored["naive_responsive_image_count"].isna(), "neuron_idx"].tolist()
        raise ValueError(f"Missing naive responsiveness counts for neurons: {missing[:10]}")
    colored["summary_image_group"] = image_group
    colored["target_stage"] = target_stage
    return colored


def _draw_response_guides(ax: plt.Axes, response_lims: list[float]) -> None:
    ax.plot(response_lims, response_lims, "--", color="0.75", linewidth=1.0, zorder=0)
    ax.axhline(0.0, color="0.86", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.86", linewidth=0.8, zorder=0)


def _count_rgba(
    frame: pd.DataFrame,
    *,
    cmap: ListedColormap,
    norm: BoundaryNorm,
    alpha: float,
) -> np.ndarray:
    colors = cmap(norm(frame["naive_responsive_image_count"].to_numpy(dtype=int)))
    colors[:, 3] = alpha
    return colors


def _draw_displacement_vectors(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    cmap: ListedColormap,
    norm: BoundaryNorm,
    alpha: float = 0.28,
) -> None:
    ax.quiver(
        frame["NO_Pre"],
        frame["O_Pre"],
        frame["dNO"],
        frame["dO"],
        color=_count_rgba(frame, cmap=cmap, norm=norm, alpha=alpha),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0032,
        headwidth=3.4,
        headlength=4.2,
        headaxislength=3.7,
        zorder=1,
    )


def _scatter_response_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    response_lims: list[float],
    cmap: ListedColormap,
    norm: BoundaryNorm,
    point_size: float,
) -> None:
    _draw_displacement_vectors(ax, frame, cmap=cmap, norm=norm)
    ax.scatter(
        frame[x_col],
        frame[y_col],
        c=frame["naive_responsive_image_count"],
        cmap=cmap,
        norm=norm,
        s=point_size,
        alpha=0.86,
        edgecolors="white",
        linewidths=0.25,
        zorder=2,
    )
    _draw_response_guides(ax, response_lims)
    ax.set_xlim(response_lims)
    ax.set_ylim(response_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("NO")
    ax.set_ylabel("O")


def plot_combined_responsiveness_scatter(
    summaries: dict[str, pd.DataFrame],
    *,
    response_lims: list[float],
    response_threshold: float,
    point_size: float = 30.0,
) -> plt.Figure:
    max_count = int(max(frame["naive_responsive_image_count"].max() for frame in summaries.values()))
    cmap, norm = _count_cmap(max_count)

    fig, axes = plt.subplots(
        len(summaries),
        2,
        figsize=(8.6, 4.05 * len(summaries)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(summaries) == 1:
        axes = np.asarray([axes])

    for row_idx, (label, frame) in enumerate(summaries.items()):
        display_label = DISPLAY_LABELS.get(label, label)
        target_label = TARGET_LABELS.get(label, str(frame["target_stage"].iloc[0]))
        _scatter_response_panel(
            axes[row_idx, 0],
            frame,
            x_col="NO_Pre",
            y_col="O_Pre",
            title=f"{display_label}: naive",
            response_lims=response_lims,
            cmap=cmap,
            norm=norm,
            point_size=point_size,
        )
        _scatter_response_panel(
            axes[row_idx, 1],
            frame,
            x_col="NO_Target",
            y_col="O_Target",
            title=f"{display_label}: {target_label}",
            response_lims=response_lims,
            cmap=cmap,
            norm=norm,
            point_size=point_size,
        )

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        ticks=np.arange(0, max_count + 1),
        shrink=0.78,
        pad=0.02,
    )
    cbar.set_label(f"Naive responsive images (Pre NO > {response_threshold:g})")

    fig.suptitle("Chronically matched neurons by naive image responsiveness", fontsize=14, fontweight="bold")
    return fig


def plot_single_responsiveness_scatter(
    frame: pd.DataFrame,
    *,
    label: str,
    response_lims: list[float],
    response_threshold: float,
    point_size: float = 30.0,
) -> plt.Figure:
    max_count = int(frame["naive_responsive_image_count"].max())
    cmap, norm = _count_cmap(max_count)
    display_label = DISPLAY_LABELS.get(label, label)
    target_label = TARGET_LABELS.get(label, str(frame["target_stage"].iloc[0]))

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), sharex=True, sharey=True, constrained_layout=True)
    _scatter_response_panel(
        axes[0],
        frame,
        x_col="NO_Pre",
        y_col="O_Pre",
        title=f"{display_label}: naive",
        response_lims=response_lims,
        cmap=cmap,
        norm=norm,
        point_size=point_size,
    )
    _scatter_response_panel(
        axes[1],
        frame,
        x_col="NO_Target",
        y_col="O_Target",
        title=f"{display_label}: {target_label}",
        response_lims=response_lims,
        cmap=cmap,
        norm=norm,
        point_size=point_size,
    )

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        ticks=np.arange(0, max_count + 1),
        shrink=0.82,
        pad=0.02,
    )
    cbar.set_label(f"Naive responsive images (Pre NO > {response_threshold:g})")

    fig.suptitle(f"{display_label}: chronic scatter by naive responsiveness", fontsize=13, fontweight="bold")
    return fig


def build_mean_displacement_by_responsiveness(merged: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        merged.groupby(["summary_name", "naive_responsive_image_count"], as_index=False)
        .agg(
            neuron_count=("neuron_idx", "size"),
            mean_NO_Pre=("NO_Pre", "mean"),
            mean_O_Pre=("O_Pre", "mean"),
            mean_NO_Target=("NO_Target", "mean"),
            mean_O_Target=("O_Target", "mean"),
            mean_dNO=("dNO", "mean"),
            mean_dO=("dO", "mean"),
            sem_dNO=("dNO", "sem"),
            sem_dO=("dO", "sem"),
            median_dNO=("dNO", "median"),
            median_dO=("dO", "median"),
            mean_dNorm=("dNorm", "mean"),
            median_dNorm=("dNorm", "median"),
        )
        .sort_values(["summary_name", "naive_responsive_image_count"])
    )
    totals = merged.groupby("summary_name").size()
    grouped["fraction_of_summary"] = grouped.apply(
        lambda row: row["neuron_count"] / totals.loc[row["summary_name"]],
        axis=1,
    )
    grouped["mean_vector_norm"] = np.hypot(grouped["mean_dNO"], grouped["mean_dO"])
    grouped["mean_vector_angle_deg"] = np.degrees(np.arctan2(grouped["mean_dO"], grouped["mean_dNO"]))
    return grouped


def _draw_shift_guides(ax: plt.Axes, shift_lims: list[float]) -> None:
    ax.axhline(0.0, color="0.82", linewidth=1.0, zorder=0)
    ax.axvline(0.0, color="0.82", linewidth=1.0, zorder=0)
    extent = float(max(abs(shift_lims[0]), abs(shift_lims[1])))
    guide = np.array([-extent, extent], dtype=float)
    ax.plot(guide, guide, "--", color="0.86", linewidth=0.8, zorder=0)
    ax.plot(guide, -guide, "--", color="0.86", linewidth=0.8, zorder=0)


def plot_mean_displacement_by_responsiveness(
    mean_displacements: pd.DataFrame,
    *,
    response_threshold: float,
) -> plt.Figure:
    summary_names = [name for name in DISPLAY_LABELS if name in set(mean_displacements["summary_name"])]
    summary_names.extend(
        name for name in mean_displacements["summary_name"].drop_duplicates().tolist() if name not in summary_names
    )
    max_count = int(mean_displacements["naive_responsive_image_count"].max())
    cmap, norm = _count_cmap(max_count)

    max_abs = float(
        np.nanmax(
            np.abs(mean_displacements[["mean_dNO", "mean_dO"]].to_numpy(dtype=float))
        )
    )
    extent = max(0.25, max_abs * 1.35)
    shift_lims = [-extent, extent]

    fig, axes = plt.subplots(
        1,
        len(summary_names),
        figsize=(4.2 * len(summary_names), 4.1),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)

    for ax, summary_name in zip(axes, summary_names):
        rows = mean_displacements.loc[mean_displacements["summary_name"] == summary_name]
        _draw_shift_guides(ax, shift_lims)
        for row in rows.itertuples(index=False):
            count = int(row.naive_responsive_image_count)
            color = cmap(norm([count]))[0]
            arrow_width = max(0.006, extent * 0.018)
            head_width = max(0.024, extent * 0.070)
            head_length = max(0.030, extent * 0.090)
            ax.arrow(
                0.0,
                0.0,
                float(row.mean_dNO),
                float(row.mean_dO),
                color=color,
                alpha=0.92,
                linewidth=2.1,
                width=arrow_width,
                head_width=head_width,
                head_length=head_length,
                length_includes_head=True,
                zorder=2,
            )
            ax.scatter(
                [row.mean_dNO],
                [row.mean_dO],
                s=24.0 + 2.0 * float(row.neuron_count),
                color=[color],
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )
            text_dx = 0.035 * extent if row.mean_dNO >= 0 else -0.035 * extent
            text_dy = 0.035 * extent if row.mean_dO >= 0 else -0.035 * extent
            ax.text(
                float(row.mean_dNO) + text_dx,
                float(row.mean_dO) + text_dy,
                f"{count}",
                color=color,
                fontsize=9,
                fontweight="bold",
                ha="left" if row.mean_dNO >= 0 else "right",
                va="bottom" if row.mean_dO >= 0 else "top",
                zorder=4,
            )

        ax.set_title(DISPLAY_LABELS.get(summary_name, summary_name))
        ax.set_xlim(shift_lims)
        ax.set_ylim(shift_lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("mean dNO")
        ax.set_ylabel("mean dO")

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        ticks=np.arange(0, max_count + 1),
        shrink=0.82,
        pad=0.02,
    )
    cbar.set_label(f"Naive responsive images (Pre NO > {response_threshold:g})")
    fig.suptitle("Mean displacement vector by naive responsiveness count", fontsize=13, fontweight="bold")
    return fig


def export_naive_responsiveness_plots(
    *,
    data_dir: Path,
    output_dir: Path,
    response_threshold: float,
    sector_threshold: float,
    point_size: float,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    act_table = th.load_transition_table(data_dir / "transitions_act.csv")
    post_table = th.load_transition_table(data_dir / "transitions_post.csv")

    summaries = {
        "task": build_colored_summary(
            act_table,
            image_group="all",
            target_stage="Task",
            response_threshold=response_threshold,
            sector_threshold=sector_threshold,
        ),
        "expert_familiar": build_colored_summary(
            post_table,
            image_group="familiar",
            target_stage="Post",
            response_threshold=response_threshold,
            sector_threshold=sector_threshold,
        ),
        "expert_novel": build_colored_summary(
            post_table,
            image_group="novel",
            target_stage="Post",
            response_threshold=response_threshold,
            sector_threshold=sector_threshold,
        ),
    }

    for label, frame in summaries.items():
        frame.insert(0, "summary_name", label)

    response_lims = th.compute_response_limits(*summaries.values())
    threshold_tag = _threshold_tag(response_threshold)
    saved_paths: list[Path] = []

    combined = plot_combined_responsiveness_scatter(
        summaries,
        response_lims=response_lims,
        response_threshold=response_threshold,
        point_size=point_size,
    )
    for suffix in ("png", "svg"):
        path = output_dir / f"chronic_scatter_by_naive_responsive_count_{threshold_tag}.{suffix}"
        combined.savefig(path, dpi=300, bbox_inches="tight")
        saved_paths.append(path)
    plt.close(combined)

    for label, frame in summaries.items():
        fig = plot_single_responsiveness_scatter(
            frame,
            label=label,
            response_lims=response_lims,
            response_threshold=response_threshold,
            point_size=point_size,
        )
        for suffix in ("png", "svg"):
            path = output_dir / f"{label}_scatter_by_naive_responsive_count_{threshold_tag}.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            saved_paths.append(path)
        plt.close(fig)

    merged = pd.concat(summaries.values(), ignore_index=True)
    summary_csv = output_dir / f"chronic_scatter_by_naive_responsive_count_{threshold_tag}.csv"
    merged.to_csv(summary_csv, index=False)
    saved_paths.append(summary_csv)

    count_distribution = (
        merged.groupby(["summary_name", "naive_responsive_image_count"], as_index=False)
        .agg(neuron_count=("neuron_idx", "size"))
        .sort_values(["summary_name", "naive_responsive_image_count"])
    )
    distribution_csv = output_dir / f"naive_responsive_count_distribution_{threshold_tag}.csv"
    count_distribution.to_csv(distribution_csv, index=False)
    saved_paths.append(distribution_csv)

    mean_displacements = build_mean_displacement_by_responsiveness(merged)
    mean_displacement_csv = output_dir / f"mean_displacement_by_naive_responsive_count_{threshold_tag}.csv"
    mean_displacements.to_csv(mean_displacement_csv, index=False)
    saved_paths.append(mean_displacement_csv)

    mean_fig = plot_mean_displacement_by_responsiveness(
        mean_displacements,
        response_threshold=response_threshold,
    )
    for suffix in ("png", "svg"):
        path = output_dir / f"mean_displacement_by_naive_responsive_count_{threshold_tag}.{suffix}"
        mean_fig.savefig(path, dpi=300, bbox_inches="tight")
        saved_paths.append(path)
    plt.close(mean_fig)

    return saved_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot chronic NO/O scatter panels colored by the number of images "
            "that were non-occluded responsive in the naive/pre state."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "naive_responsiveness_count_scatter",
    )
    parser.add_argument(
        "--response-threshold",
        type=float,
        default=DEFAULT_RESPONSE_THRESHOLD,
        help="Pre/NO response threshold used to classify a neuron-image pair as responsive.",
    )
    parser.add_argument(
        "--sector-threshold",
        type=float,
        default=0.3,
        help="Small-delta threshold passed to the existing transition sector summary.",
    )
    parser.add_argument("--point-size", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    saved_paths = export_naive_responsiveness_plots(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        response_threshold=args.response_threshold,
        sector_threshold=args.sector_threshold,
        point_size=args.point_size,
    )
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
