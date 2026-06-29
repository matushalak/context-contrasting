from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox

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
    fig: plt.Figure,
    output_dir: Path,
    basename: str,
    *,
    formats: tuple[str, ...],
    dpi: int = 300,
    pad_inches: float = 0.035,
) -> list[Path]:
    def axis_points(ax: plt.Axes) -> np.ndarray:
        offsets = []
        for collection in ax.collections:
            if not hasattr(collection, "get_offsets"):
                continue
            arr = np.asarray(collection.get_offsets(), dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                offsets.append(arr[:, :2])
        if not offsets:
            return np.empty((0, 2), dtype=float)
        points = np.concatenate(offsets, axis=0)
        return points[np.isfinite(points).all(axis=1)]

    def apply_shared_square_limits(axes: list[plt.Axes], margin: float = 0.5) -> None:
        points = [axis_points(ax) for ax in axes]
        points = [arr for arr in points if arr.size]
        if not points:
            return
        combined = np.concatenate(points, axis=0)
        low = min(
            *(ax.get_xlim()[0] for ax in axes),
            *(ax.get_ylim()[0] for ax in axes),
            float(combined.min()) - margin,
        )
        high = max(
            *(ax.get_xlim()[1] for ax in axes),
            *(ax.get_ylim()[1] for ax in axes),
            float(combined.max()) + margin,
        )
        for ax in axes:
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_aspect("equal", adjustable="box")

    def square_bbox(bbox: Bbox) -> Bbox:
        width = bbox.x1 - bbox.x0
        height = bbox.y1 - bbox.y0
        side = max(width, height)
        cx = 0.5 * (bbox.x0 + bbox.x1)
        cy = 0.5 * (bbox.y0 + bbox.y1)
        return Bbox.from_extents(cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(dict.fromkeys((*formats, "eps")))
    saved: list[Path] = []
    axes_to_export = [fig.axes[idx] for idx in (4, 5)]
    apply_shared_square_limits(axes_to_export)
    for suffix, ax_idx in (("naive_sector_scatter", 4), ("expert_sector_scatter", 5)):
        ax = fig.axes[ax_idx]
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_xticks(np.arange(np.ceil(x0), np.floor(x1) + 1.0, 1.0))
        ax.set_yticks(np.arange(np.ceil(y0), np.floor(y1) + 1.0, 1.0))
        ax.tick_params(axis="both", labelsize=24, width=1.4, length=5)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = ax.get_tightbbox(renderer)
        if bbox is None:
            bbox = ax.get_window_extent(renderer)
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        bbox = Bbox.from_extents(
            bbox.x0 - pad_inches,
            bbox.y0 - pad_inches,
            bbox.x1 + pad_inches,
            bbox.y1 + pad_inches,
        )
        bbox = square_bbox(bbox)
        for fmt in formats:
            path = output_dir / f"{basename}_{suffix}.{fmt}"
            fig.savefig(path, bbox_inches=bbox, dpi=dpi)
            saved.append(path)
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
    response_lims = _robust_response_limits(list(summaries.values()), hi_percentile=axis_clip_percentile)
    shift_lims = _robust_shift_limits(list(summaries.values()), hi_percentile=axis_clip_percentile)

    saved_paths: list[Path] = []
    for name, summary in summaries.items():
        fig = th.plot_mean_transition_summary(
            summary,
            title=f"Ground-truth {name} transition summary",
            start_label="Naive",
            end_label="Expert",
            response_lims=response_lims,
            shift_lims=shift_lims,
            style=PLOT_STYLE,
        )
        saved_paths.extend(
            _export_sector_response_panels(
                fig,
                output_dir,
                f"ground_truth_{name}_summary",
                formats=formats,
            )
        )
        plt.close(fig)
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
