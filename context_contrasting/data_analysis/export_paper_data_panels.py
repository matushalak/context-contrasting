from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import context_contrasting.data_analysis.transitions_helpers as th


PLOT_STYLE = th.DEFAULT_PLOT_STYLE | {
    "pre_point_alpha": 1.0,
    "target_point_alpha": 1.0,
    "shift_point_alpha": 1.0,
    "individual_vector_width": 0.005,
    "mean_arrow_width": 3.1,
    "mean_arrow_mutation_scale": 16.5,
}
RESPONSE_X_LABEL = "Non-occluded response z-scored $\\Delta$F/F"
RESPONSE_Y_LABEL = "Occluded response z-scored $\\Delta$F/F"
SHIFT_X_LABEL = "$dNO$"
SHIFT_Y_LABEL = "$dO$"


def _robust_response_limits(summaries: list[pd.DataFrame], *, hi_percentile: float, pad: float = 0.4) -> list[float]:
    cols = ["NO_Pre", "O_Pre", "NO_Target", "O_Target"]
    values = np.concatenate([summary[cols].to_numpy(dtype=float).reshape(-1) for summary in summaries])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [-1.0, 1.0]
    lo = float(np.nanpercentile(values, max(0.0, 100.0 - hi_percentile)))
    hi = float(np.nanpercentile(values, hi_percentile))
    return [lo - pad, hi + pad]


def _robust_shift_limits(
    summaries: list[pd.DataFrame],
    *,
    hi_percentile: float,
    pad_ratio: float = 0.12,
    fallback: float = 0.5,
) -> list[float]:
    values = np.concatenate([summary[["dNO", "dO"]].to_numpy(dtype=float).reshape(-1) for summary in summaries])
    values = np.abs(values[np.isfinite(values)])
    if values.size == 0:
        return [-fallback, fallback]
    extent = float(np.nanpercentile(values, hi_percentile))
    if not np.isfinite(extent) or extent == 0:
        extent = fallback
    else:
        extent *= 1.0 + pad_ratio
    return [-extent, extent]


def _export_sector_response_panels(
    summary: pd.DataFrame,
    output_dir: Path,
    basename: str,
    *,
    formats: tuple[str, ...],
    target_label: str = "Expert",
    dpi: int = 300,
) -> list[Path]:
    def shared_limits(low_margin: float = 0.25, high_margin: float = 0.75) -> list[float]:
        # Asymmetric padding: tight on the low end (no wasted whitespace below the
        # zero-response cloud) and generous on the high end (room for the legend
        # without it overlapping the cloud).
        cols = ["NO_Pre", "O_Pre", "NO_Target", "O_Target"]
        values = summary[cols].to_numpy(dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return [-1.0, 1.0]
        return [float(values.min()) - low_margin, float(values.max()) + high_margin]

    def draw_panel(ax: plt.Axes, *, x_col: str, y_col: str) -> None:
        log_norms = (
            summary["log_dNorm"].to_numpy(dtype=float)
            if "log_dNorm" in summary.columns
            else np.log(summary["dNorm"].to_numpy(dtype=float) + th.LOG_NORM_EPS)
        )
        alphas = th._map_norms_to_alphas(log_norms, min_alpha=PLOT_STYLE["alpha_min"], max_alpha=PLOT_STYLE["alpha_max"])
        sectors = summary["RotatedSector"].to_numpy()
        for sector in th._sector_plot_order(small_delta_first=True):
            sector_rows = summary.loc[summary["RotatedSector"] == sector]
            if sector_rows.empty:
                continue
            pos_idx = np.flatnonzero(sectors == sector)
            rgb = np.array(th.mcolors.to_rgb(th.ROTATED_SECTOR_PALETTE[sector])).reshape(1, 3)
            rgba = np.repeat(rgb, len(sector_rows), axis=0)
            rgba = np.concatenate([rgba, alphas[pos_idx].reshape(-1, 1)], axis=1)
            ax.scatter(
                sector_rows[x_col],
                sector_rows[y_col],
                s=PLOT_STYLE["point_size"],
                c=rgba,
                edgecolors="none",
                zorder=th._sector_scatter_zorder(sector),
            )

    def style_axis(ax: plt.Axes, *, title: str) -> None:
        th._draw_diagonal(ax, lims)
        ax.axhline(0.0, color="0.85", lw=1.0, zorder=0)
        ax.axvline(0.0, color="0.85", lw=1.0, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_title(title, fontsize=28, pad=14)
        ax.tick_params(axis="both", labelsize=22, width=1.4, length=5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)

    output_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(dict.fromkeys((*formats, "eps")))
    lims = shared_limits()
    ticks = np.arange(np.ceil(lims[0]), np.floor(lims[1]) + 1.0, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.22, top=0.82, wspace=0.18)

    draw_panel(axes[0], x_col="NO_Pre", y_col="O_Pre")
    style_axis(axes[0], title="Naive")
    draw_panel(axes[1], x_col="NO_Target", y_col="O_Target")
    style_axis(axes[1], title=target_label)
    fig.supxlabel(RESPONSE_X_LABEL, fontsize=24, y=0.055)
    fig.supylabel(RESPONSE_Y_LABEL, fontsize=24, x=0.04)

    saved: list[Path] = []
    for fmt in formats:
        path = output_dir / f"{basename}_naive_expert_sector_scatter.{fmt}"
        fig.savefig(path, dpi=dpi)
        saved.append(path)
    plt.close(fig)

    legend_paths = th.save_rotated_sector_unit_legend(
        summary,
        output_dir / f"{basename}_sector_legend.{formats[0]}",
        title=None,
        formats=formats,
    )
    saved.extend(legend_paths)
    return saved


def _build_image_summaries(
    transition_table: pd.DataFrame,
    *,
    image_group: str,
    pre_stage: str,
    target_stage: str,
    threshold: float,
) -> list[pd.DataFrame]:
    frame = transition_table.loc[transition_table["image_group"] == image_group].copy()
    if frame.empty:
        raise ValueError(f"No rows found for image group {image_group!r}.")

    image_keys = (
        frame[["image_idx_original", "image_idx_within_group"]]
        .drop_duplicates()
        .sort_values(["image_idx_within_group", "image_idx_original"])
    )

    summaries: list[pd.DataFrame] = []
    for image in image_keys.itertuples(index=False):
        image_frame = frame.loc[
            (frame["image_idx_original"] == image.image_idx_original)
            & (frame["image_idx_within_group"] == image.image_idx_within_group)
        ].copy()
        summary = th.build_mean_summary(
            image_frame,
            image_group=image_group,
            pre_stage=pre_stage,
            target_stage=target_stage,
            threshold=threshold,
        )
        summary["image_idx_original"] = image.image_idx_original
        summary["image_idx_within_group"] = image.image_idx_within_group
        summary["summary_image_group"] = image_group
        summary.attrs["image_idx_original"] = image.image_idx_original
        summary.attrs["image_idx_within_group"] = image.image_idx_within_group
        summaries.append(summary)
    return summaries


def _image_row_label(summary: pd.DataFrame) -> str:
    original = summary.attrs.get("image_idx_original", summary["image_idx_original"].iloc[0])
    return f"Image {int(original)}"


def _sector_rgba(summary: pd.DataFrame, sector: str, alphas: np.ndarray) -> np.ndarray:
    sector_mask = summary["RotatedSector"].to_numpy() == sector
    rgb = np.array(th.mcolors.to_rgb(th.ROTATED_SECTOR_PALETTE[sector])).reshape(1, 3)
    rgba = np.repeat(rgb, int(sector_mask.sum()), axis=0)
    return np.concatenate([rgba, alphas[sector_mask].reshape(-1, 1)], axis=1)


def _norm_alphas(summary: pd.DataFrame) -> np.ndarray:
    log_norms = (
        summary["log_dNorm"].to_numpy(dtype=float)
        if "log_dNorm" in summary.columns
        else np.log(summary["dNorm"].to_numpy(dtype=float) + th.LOG_NORM_EPS)
    )
    return th._map_norms_to_alphas(
        log_norms,
        min_alpha=PLOT_STYLE["alpha_min"],
        max_alpha=PLOT_STYLE["alpha_max"],
    )


def _draw_sector_scatter(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    point_size: float,
) -> None:
    alphas = _norm_alphas(summary)
    for sector in th._sector_plot_order(small_delta_first=True):
        sector_rows = summary.loc[summary["RotatedSector"] == sector]
        if sector_rows.empty:
            continue
        ax.scatter(
            sector_rows[x_col],
            sector_rows[y_col],
            s=point_size,
            c=_sector_rgba(summary, sector, alphas),
            edgecolors="none",
            zorder=th._sector_scatter_zorder(sector),
        )


def _style_response_axis(
    ax: plt.Axes,
    *,
    response_lims: list[float],
    ticks: np.ndarray,
) -> None:
    th._draw_diagonal(ax, response_lims)
    ax.axhline(0.0, color="0.85", lw=1.0, zorder=0)
    ax.axvline(0.0, color="0.85", lw=1.0, zorder=0)
    ax.set_xlim(response_lims)
    ax.set_ylim(response_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


def _style_shift_axis(
    ax: plt.Axes,
    *,
    shift_lims: list[float],
    ticks: np.ndarray,
) -> None:
    th._draw_origin_guides(ax)
    th._draw_rotated_guides(ax, shift_lims)
    ax.set_xlim(shift_lims)
    ax.set_ylim(shift_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


def _draw_sector_mean_arrows(ax: plt.Axes, summary: pd.DataFrame) -> None:
    sector_means = th.sector_mean_table(summary)
    sector_arrow_alphas = th._sector_percentage_alphas(summary)
    for sector in th.ROTATED_SECTOR_ORDER:
        if sector == "small ∆":
            continue
        mean_rows = sector_means.loc[sector_means["RotatedSector"] == sector]
        if mean_rows.empty:
            continue
        mean_row = mean_rows.iloc[0]
        th._draw_arrow(
            ax,
            (0.0, 0.0),
            (float(mean_row["dNO"]), float(mean_row["dO"])),
            color=th._darken_color(th.ROTATED_SECTOR_PALETTE[sector]),
            linewidth=max(3.0, PLOT_STYLE["mean_arrow_width"] * 0.9),
            mutation_scale=PLOT_STYLE["mean_arrow_mutation_scale"],
            alpha=sector_arrow_alphas[sector],
            zorder=4,
        )


def _export_image_row_sector_panels(
    summaries: list[pd.DataFrame],
    output_dir: Path,
    basename: str,
    *,
    formats: tuple[str, ...],
    target_label: str,
    figure_title: str,
    response_lims: list[float],
    shift_lims: list[float],
    dpi: int = 300,
) -> list[Path]:
    if not summaries:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(dict.fromkeys((*formats, "eps")))
    n_rows = len(summaries)
    response_ticks = np.arange(np.ceil(response_lims[0]), np.floor(response_lims[1]) + 1.0, 1.0)
    shift_ticks = np.arange(np.ceil(shift_lims[0]), np.floor(shift_lims[1]) + 1.0, 1.0)

    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(12.6, 3.75 * n_rows),
        sharex="col",
        sharey="col",
        squeeze=False,
    )
    fig.subplots_adjust(left=0.14, right=0.975, bottom=0.16, top=0.92, wspace=0.24, hspace=0.28)
    fig.suptitle(figure_title, fontsize=19, fontweight="bold", y=0.985)

    for row_idx, summary in enumerate(summaries):
        row_label = _image_row_label(summary)

        _draw_sector_scatter(
            axes[row_idx, 0],
            summary,
            x_col="NO_Pre",
            y_col="O_Pre",
            point_size=PLOT_STYLE["point_size"],
        )
        _style_response_axis(axes[row_idx, 0], response_lims=response_lims, ticks=response_ticks)
        axes[row_idx, 0].text(
            -0.28,
            0.5,
            row_label,
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

        _draw_sector_scatter(
            axes[row_idx, 1],
            summary,
            x_col="NO_Target",
            y_col="O_Target",
            point_size=PLOT_STYLE["point_size"],
        )
        _style_response_axis(axes[row_idx, 1], response_lims=response_lims, ticks=response_ticks)

        _draw_sector_scatter(
            axes[row_idx, 2],
            summary,
            x_col="dNO",
            y_col="dO",
            point_size=PLOT_STYLE["point_size"],
        )
        _draw_sector_mean_arrows(axes[row_idx, 2], summary)
        _style_shift_axis(axes[row_idx, 2], shift_lims=shift_lims, ticks=shift_ticks)

    axes[0, 0].set_title("Naive", fontsize=14, pad=8)
    axes[0, 1].set_title(target_label, fontsize=14, pad=8)
    axes[0, 2].set_title(f"{target_label} - naive", fontsize=14, pad=8)
    for ax in axes[:-1, :].ravel():
        ax.tick_params(labelbottom=False)
    for ax in axes[:, 1].ravel():
        ax.tick_params(labelleft=False)
    fig.text(0.43, 0.095, RESPONSE_X_LABEL, ha="center", va="center", fontsize=11)
    fig.text(0.82, 0.095, SHIFT_X_LABEL, ha="center", va="center", fontsize=11)
    fig.text(0.035, 0.51, RESPONSE_Y_LABEL, ha="center", va="center", rotation=90, fontsize=11)
    axes[n_rows // 2, 2].set_ylabel(SHIFT_Y_LABEL, fontsize=11)

    handles = th._legend_handles({sector: sector for sector in th.ROTATED_SECTOR_ORDER}, linewidth=2.5)
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.54, 0.01))

    saved: list[Path] = []
    for fmt in formats:
        path = output_dir / f"{basename}.{fmt}"
        fig.savefig(path, dpi=dpi)
        saved.append(path)
    plt.close(fig)

    summary_csv = output_dir / f"{basename}.csv"
    pd.concat(summaries, ignore_index=True).to_csv(summary_csv, index=False)
    saved.append(summary_csv)
    return saved


def export_paper_data_panels(
    *,
    data_dir: Path,
    output_dir: Path,
    threshold: float,
    axis_clip_percentile: float,
    formats: tuple[str, ...],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    act_table = th.load_transition_table(data_dir / "transitions_act.csv")
    post_table = th.load_transition_table(data_dir / "transitions_post.csv")

    summaries = {
        "task": (
            th.build_mean_summary(
                act_table,
                image_group="all",
                pre_stage="Pre",
                target_stage="Task",
                threshold=threshold,
            ),
            "Task",
        ),
        "novel": (
            th.build_mean_summary(
                post_table,
                image_group="novel",
                pre_stage="Pre",
                target_stage="Post",
                threshold=threshold,
            ),
            "Expert",
        ),
    }
    saved_paths: list[Path] = []
    for name, (summary, target_label) in summaries.items():
        saved_paths.extend(
            _export_sector_response_panels(
                summary,
                output_dir,
                f"ground_truth_{name}_summary",
                formats=formats,
                target_label=target_label,
            )
        )

    by_image_specs = {
        "task": (
            _build_image_summaries(
                act_table,
                image_group="all",
                pre_stage="Pre",
                target_stage="Task",
                threshold=threshold,
            ),
            "Task",
            "Naive to task by familiar image",
        ),
        "expert_familiar": (
            _build_image_summaries(
                post_table,
                image_group="familiar",
                pre_stage="Pre",
                target_stage="Post",
                threshold=threshold,
            ),
            "Expert",
            "Naive to expert by familiar image",
        ),
        "novel": (
            _build_image_summaries(
                post_table,
                image_group="novel",
                pre_stage="Pre",
                target_stage="Post",
                threshold=threshold,
            ),
            "Novel expert",
            "Naive to novel expert by novel image",
        ),
    }
    all_image_summaries = [summary for summaries_for_group, _, _ in by_image_specs.values() for summary in summaries_for_group]
    response_lims = _robust_response_limits(all_image_summaries, hi_percentile=axis_clip_percentile)
    shift_lims = _robust_shift_limits(all_image_summaries, hi_percentile=axis_clip_percentile)
    for name, (image_summaries, target_label, figure_title) in by_image_specs.items():
        saved_paths.extend(
            _export_image_row_sector_panels(
                image_summaries,
                output_dir,
                f"ground_truth_{name}_by_image_sector_panels",
                formats=formats,
                target_label=target_label,
                figure_title=figure_title,
                response_lims=response_lims,
                shift_lims=shift_lims,
            )
        )
    return saved_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper comparison panels from real transition data.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "paper_data_exports",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--axis-clip-percentile", type=float, default=99.0)
    parser.add_argument("--formats", nargs="+", default=["png", "svg", "eps"], choices=("png", "svg", "eps"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    export_paper_data_panels(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        axis_clip_percentile=args.axis_clip_percentile,
        formats=tuple(args.formats),
    )


if __name__ == "__main__":
    main()
