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
    dpi: int = 300,
) -> list[Path]:
    def shared_limits(margin: float = 0.5) -> list[float]:
        cols = ["NO_Pre", "O_Pre", "NO_Target", "O_Target"]
        values = summary[cols].to_numpy(dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return [-1.0, 1.0]
        return [float(values.min()) - margin, float(values.max()) + margin]

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

    output_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(dict.fromkeys((*formats, "eps")))
    lims = shared_limits()
    ticks = np.arange(np.ceil(lims[0]), np.floor(lims[1]) + 1.0, 1.0)
    saved: list[Path] = []
    for suffix, x_col, y_col in (
        ("naive_sector_scatter", "NO_Pre", "O_Pre"),
        ("expert_sector_scatter", "NO_Target", "O_Target"),
    ):
        panel_fig, ax = plt.subplots(figsize=(4.0, 4.0))
        panel_fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.96)
        draw_panel(ax, x_col=x_col, y_col=y_col)
        th._draw_diagonal(ax, lims)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.tick_params(axis="both", labelsize=24, width=1.4, length=5)
        for fmt in formats:
            path = output_dir / f"{basename}_{suffix}.{fmt}"
            panel_fig.savefig(path, dpi=dpi)
            saved.append(path)
        plt.close(panel_fig)
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
        "task": th.build_mean_summary(
            act_table,
            image_group="all",
            pre_stage="Pre",
            target_stage="Task",
            threshold=threshold,
        ),
        "novel": th.build_mean_summary(
            post_table,
            image_group="novel",
            pre_stage="Pre",
            target_stage="Post",
            threshold=threshold,
        ),
    }
    saved_paths: list[Path] = []
    for name, summary in summaries.items():
        saved_paths.extend(
            _export_sector_response_panels(
                summary,
                output_dir,
                f"ground_truth_{name}_summary",
                formats=formats,
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
