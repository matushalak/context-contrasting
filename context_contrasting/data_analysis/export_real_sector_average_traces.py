from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import context_contrasting.data_analysis.transitions_helpers as th


SECTOR_ORDER = ("+NO axis", "+O axis", "-NO axis", "-O axis")
SECTOR_LABELS = {sector: sector.replace(" axis", "") for sector in SECTOR_ORDER}
SECTOR_MODES = ("sector-average", "sector-per-image")
DIAGONAL_HALF_WIDTH_RAD = np.pi / 8.0
DIAGONAL_SPECS = {
    "task": {"key": "minus_no_plus_o", "label": "-NO/+O", "angle": 3.0 * np.pi / 4.0},
    "expert_familiar": {"key": "minus_no_plus_o", "label": "-NO/+O", "angle": 3.0 * np.pi / 4.0},
    "novel": {"key": "plus_no_plus_o", "label": "+NO/+O", "angle": np.pi / 4.0},
}
IMAGE_TYPE_SPECS = (
    ("Occl", "O", "red"),
    ("Full", "NO", "black"),
)
SCALE_BAR_UNITS = 1.0
MIN_ROW_Y_SPAN = 1.35


@dataclass(frozen=True)
class TraceExportSpec:
    key: str
    transition_csv: str
    trace_csv: str
    image_group: str
    pre_stage: str
    target_stage: str
    target_label: str
    folder: str
    basename: str


DEFAULT_SPECS = (
    TraceExportSpec(
        key="task",
        transition_csv="transitions_act.csv",
        trace_csv="transitions_act_traces.csv",
        image_group="all",
        pre_stage="Pre",
        target_stage="Task",
        target_label="Task",
        folder="task",
        basename="sector_average_task_examples_sem",
    ),
    TraceExportSpec(
        key="expert_familiar",
        transition_csv="transitions_post.csv",
        trace_csv="transitions_post_traces.csv",
        image_group="familiar",
        pre_stage="Pre",
        target_stage="Post",
        target_label="Expert",
        folder="familiar",
        basename="sector_average_familiar_examples_sem",
    ),
    TraceExportSpec(
        key="novel",
        transition_csv="transitions_post.csv",
        trace_csv="transitions_post_traces.csv",
        image_group="novel",
        pre_stage="Pre",
        target_stage="Post",
        target_label="Novel expert",
        folder="novel",
        basename="sector_average_novel_examples_sem",
    ),
)


def _load_trace_table(path: Path, *, image_group: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["image_group"].eq(image_group)].copy()
    if frame.empty:
        raise ValueError(f"No trace rows found for image_group={image_group!r} in {path}.")
    frame["neuron_idx"] = frame["neuron_idx"].astype(int)
    return frame


def _attach_sector_labels(
    trace_table: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    key_cols: list[str],
    require_all: bool = True,
) -> pd.DataFrame:
    sectors = summary[key_cols + ["RotatedSector"]].copy()
    sectors["neuron_idx"] = sectors["neuron_idx"].astype(int)
    labeled = trace_table.merge(
        sectors,
        on=key_cols,
        how="left" if require_all else "inner",
        validate="many_to_one",
    )
    if require_all and labeled["RotatedSector"].isna().any():
        missing = (
            labeled.loc[labeled["RotatedSector"].isna(), key_cols]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(f"Trace rows missing sector labels for keys: {missing}")
    return labeled


def _angle_distance(angle: pd.Series | np.ndarray, target: float) -> np.ndarray:
    values = np.asarray(angle, dtype=float)
    return np.abs(np.arctan2(np.sin(values - target), np.cos(values - target)))


def _mean_sector_summary(
    transition_table: pd.DataFrame,
    *,
    spec: TraceExportSpec,
    threshold: float,
) -> pd.DataFrame:
    return th.build_mean_summary(
        transition_table,
        image_group=spec.image_group,
        pre_stage=spec.pre_stage,
        target_stage=spec.target_stage,
        threshold=threshold,
    )


def _per_image_sector_summary(
    transition_table: pd.DataFrame,
    *,
    spec: TraceExportSpec,
    threshold: float,
) -> pd.DataFrame:
    frame = transition_table.loc[transition_table["image_group"].eq(spec.image_group)].copy()
    if frame.empty:
        raise ValueError(f"No transition rows found for image_group={spec.image_group!r}.")

    image_keys = (
        frame[["image_idx_original", "image_idx_within_group"]]
        .drop_duplicates()
        .sort_values(["image_idx_within_group", "image_idx_original"])
    )
    summaries: list[pd.DataFrame] = []
    for image in image_keys.itertuples(index=False):
        image_frame = frame.loc[
            frame["image_idx_original"].eq(image.image_idx_original)
            & frame["image_idx_within_group"].eq(image.image_idx_within_group)
        ].copy()
        summary = th.build_mean_summary(
            image_frame,
            image_group=spec.image_group,
            pre_stage=spec.pre_stage,
            target_stage=spec.target_stage,
            threshold=threshold,
        )
        summary["image_idx_original"] = int(image.image_idx_original)
        summary["image_idx_within_group"] = int(image.image_idx_within_group)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def _diagonal_assignments(
    summary: pd.DataFrame,
    *,
    spec: TraceExportSpec,
    key_cols: list[str],
    threshold: float,
) -> pd.DataFrame:
    diagonal = DIAGONAL_SPECS[spec.key]
    mask = (
        _angle_distance(summary["Angle"], float(diagonal["angle"])) <= DIAGONAL_HALF_WIDTH_RAD
    ) & (summary["dNorm"].astype(float) > threshold)
    selected = summary.loc[mask, key_cols].copy()
    selected["RotatedSector"] = str(diagonal["key"])
    return selected


def _summarize_traces(
    labeled: pd.DataFrame,
    *,
    spec: TraceExportSpec,
    trace_order: tuple[str, ...] = SECTOR_ORDER,
    trace_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    if trace_labels is None:
        trace_labels = SECTOR_LABELS
    index_cols = [
        "RotatedSector",
        "image_group",
        "image_idx_original",
        "image_idx_within_group",
        "stage",
        "image_type",
        "time",
    ]
    grouped = (
        labeled.loc[labeled["RotatedSector"].isin(trace_order)]
        .groupby(index_cols, observed=True, as_index=False)
        .agg(
            mean_response=("response", "mean"),
            sd_response=("response", "std"),
            n_cells=("neuron_idx", "nunique"),
        )
        .sort_values(index_cols)
    )
    grouped["sem"] = grouped["sd_response"].fillna(0.0) / np.sqrt(np.maximum(grouped["n_cells"], 1))
    grouped.insert(0, "trace_group", spec.key)
    grouped["sector_label"] = grouped["RotatedSector"].map(trace_labels)
    grouped["stage_label"] = grouped["stage"].replace({spec.pre_stage: "Naive", spec.target_stage: spec.target_label})
    grouped["response_type"] = grouped["image_type"].map({"Full": "NO", "Occl": "O"})
    return grouped


def _summarize_pooled_traces(
    labeled: pd.DataFrame,
    *,
    spec: TraceExportSpec,
    trace_order: tuple[str, ...] = SECTOR_ORDER,
    trace_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    if trace_labels is None:
        trace_labels = SECTOR_LABELS
    index_cols = [
        "RotatedSector",
        "image_group",
        "stage",
        "image_type",
        "time",
    ]
    grouped = (
        labeled.loc[labeled["RotatedSector"].isin(trace_order)]
        .groupby(index_cols, observed=True, as_index=False)
        .agg(
            mean_response=("response", "mean"),
            sd_response=("response", "std"),
            n_responses=("response", "size"),
            n_cells=("neuron_idx", "nunique"),
            n_images=("image_idx_original", "nunique"),
        )
        .sort_values(index_cols)
    )
    grouped["sem"] = grouped["sd_response"].fillna(0.0) / np.sqrt(np.maximum(grouped["n_responses"], 1))
    grouped.insert(0, "trace_group", spec.key)
    grouped["sector_label"] = grouped["RotatedSector"].map(trace_labels)
    grouped["stage_label"] = grouped["stage"].replace({spec.pre_stage: "Naive", spec.target_stage: spec.target_label})
    grouped["response_type"] = grouped["image_type"].map({"Full": "NO", "Occl": "O"})
    return grouped


def _column_pairs(summary: pd.DataFrame, *, pre_stage: str, target_stage: str) -> list[tuple[int, str]]:
    image_keys = (
        summary[["image_idx_within_group", "image_idx_original"]]
        .drop_duplicates()
        .sort_values(["image_idx_within_group", "image_idx_original"])
    )
    stages = [pre_stage, target_stage]
    pairs: list[tuple[int, str]] = []
    for image in image_keys.itertuples(index=False):
        for stage in stages:
            pairs.append((int(image.image_idx_original), stage))
    return pairs


def _row_y_limits(row_bounds: list[float]) -> tuple[float, float] | None:
    if not row_bounds:
        return None
    lo = float(np.nanmin([*row_bounds, 0.0, SCALE_BAR_UNITS]))
    hi = float(np.nanmax([*row_bounds, 0.0, SCALE_BAR_UNITS]))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    center = 0.5 * (lo + hi)
    span = max((hi - lo) * 1.2, MIN_ROW_Y_SPAN)
    return center - 0.5 * span, center + 0.5 * span


def _add_scale_bar(ax: plt.Axes, *, length: float = SCALE_BAR_UNITS) -> None:
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    x_span = x_hi - x_lo
    y_span = y_hi - y_lo
    if x_span <= 0 or y_span <= 0:
        return

    x = x_lo + 0.08 * x_span
    cap = 0.018 * x_span
    y0 = 0.0
    y1 = length

    ax.plot([x, x], [y0, y1], color="0.15", lw=1.3, solid_capstyle="butt", zorder=6)
    ax.plot([x - cap, x + cap], [y0, y0], color="0.15", lw=1.3, solid_capstyle="butt", zorder=6)
    ax.plot([x - cap, x + cap], [y1, y1], color="0.15", lw=1.3, solid_capstyle="butt", zorder=6)
    ax.text(
        x - 1.8 * cap,
        0.5 * (y0 + y1),
        "1 z",
        ha="right",
        va="center",
        fontsize=7,
        color="0.15",
        rotation=90,
    )


def _plot_sector_average_panel(
    summary: pd.DataFrame,
    *,
    spec: TraceExportSpec,
    output_dir: Path,
    basename: str,
    formats: tuple[str, ...],
    dpi: int,
    trace_order: tuple[str, ...] = SECTOR_ORDER,
    trace_labels: dict[str, str] | None = None,
    title: str = "Real-data sector-average traces +/- SEM",
) -> list[Path]:
    if trace_labels is None:
        trace_labels = SECTOR_LABELS
    column_pairs = _column_pairs(summary, pre_stage=spec.pre_stage, target_stage=spec.target_stage)
    if not column_pairs:
        return []

    fig, axes = plt.subplots(
        len(trace_order),
        len(column_pairs),
        figsize=(max(7.0, 1.65 * len(column_pairs)), 1.8 * len(trace_order) + 1.0),
        squeeze=False,
        sharex=True,
        sharey=False,
    )
    fig.subplots_adjust(left=0.12, right=0.995, top=0.82, bottom=0.06, wspace=0.14, hspace=0.22)

    for col_idx, (image_idx, stage) in enumerate(column_pairs):
        stage_label = "Naive" if stage == spec.pre_stage else spec.target_label
        axes[0, col_idx].set_title(f"image {image_idx}\n{stage_label}", fontsize=8)

    for row_idx, sector in enumerate(trace_order):
        sector_rows = summary.loc[summary["RotatedSector"].eq(sector)].copy()
        row_bounds: list[float] = []
        for col_idx, (image_idx, stage) in enumerate(column_pairs):
            ax = axes[row_idx, col_idx]
            ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
            ax.axvspan(0.0, 1.0, color="0.92", zorder=-1)
            for image_type, _response_type, color in IMAGE_TYPE_SPECS:
                trace_df = sector_rows.loc[
                    sector_rows["image_idx_original"].eq(image_idx)
                    & sector_rows["stage"].eq(stage)
                    & sector_rows["image_type"].eq(image_type)
                ].sort_values("time")
                if trace_df.empty:
                    continue
                x = trace_df["time"].to_numpy(dtype=float)
                y = trace_df["mean_response"].to_numpy(dtype=float)
                sem = trace_df["sem"].fillna(0.0).to_numpy(dtype=float)
                ax.plot(x, y, color=color, lw=1.2)
                ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.18, linewidth=0)
                row_bounds.extend((float(np.nanmin(y - sem)), float(np.nanmax(y + sem))))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[row_idx, 0].text(
            -0.08,
            0.5,
            trace_labels[sector],
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9,
        )
        y_limits = _row_y_limits(row_bounds)
        if y_limits is not None:
            for ax in axes[row_idx, :]:
                ax.set_ylim(y_limits)
            _add_scale_bar(axes[row_idx, 0])

    fig.suptitle(title, y=0.98, fontsize=11)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        path = output_dir / f"{basename}.{fmt}"
        fig.savefig(path, dpi=dpi)
        saved.append(path)
    plt.close(fig)
    return saved


def _plot_pooled_sector_panel(
    pooled_summary: pd.DataFrame,
    *,
    spec: TraceExportSpec,
    output_dir: Path,
    basename: str,
    formats: tuple[str, ...],
    dpi: int,
    trace_order: tuple[str, ...] = SECTOR_ORDER,
    trace_labels: dict[str, str] | None = None,
    title: str = "Real-data pooled sector-average traces +/- SEM",
) -> list[Path]:
    if trace_labels is None:
        trace_labels = SECTOR_LABELS
    stage_pairs = [(spec.pre_stage, "Naive"), (spec.target_stage, spec.target_label)]
    fig, axes = plt.subplots(
        len(trace_order),
        len(stage_pairs),
        figsize=(7.0, 1.8 * len(trace_order) + 1.0),
        squeeze=False,
        sharex=True,
        sharey=False,
    )
    fig.subplots_adjust(left=0.14, right=0.995, top=0.82, bottom=0.06, wspace=0.16, hspace=0.22)

    for col_idx, (_stage, stage_label) in enumerate(stage_pairs):
        axes[0, col_idx].set_title(f"pooled\n{stage_label}", fontsize=8)

    for row_idx, sector in enumerate(trace_order):
        sector_rows = pooled_summary.loc[pooled_summary["RotatedSector"].eq(sector)].copy()
        row_bounds: list[float] = []
        for col_idx, (stage, _stage_label) in enumerate(stage_pairs):
            ax = axes[row_idx, col_idx]
            ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
            ax.axvspan(0.0, 1.0, color="0.92", zorder=-1)
            for image_type, _response_type, color in IMAGE_TYPE_SPECS:
                trace_df = sector_rows.loc[
                    sector_rows["stage"].eq(stage)
                    & sector_rows["image_type"].eq(image_type)
                ].sort_values("time")
                if trace_df.empty:
                    continue
                x = trace_df["time"].to_numpy(dtype=float)
                y = trace_df["mean_response"].to_numpy(dtype=float)
                sem = trace_df["sem"].fillna(0.0).to_numpy(dtype=float)
                ax.plot(x, y, color=color, lw=1.2)
                ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.18, linewidth=0)
                row_bounds.extend((float(np.nanmin(y - sem)), float(np.nanmax(y + sem))))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[row_idx, 0].text(
            -0.08,
            0.5,
            trace_labels[sector],
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9,
        )
        y_limits = _row_y_limits(row_bounds)
        if y_limits is not None:
            for ax in axes[row_idx, :]:
                ax.set_ylim(y_limits)
            _add_scale_bar(axes[row_idx, 0])

    fig.suptitle(title, y=0.98, fontsize=11)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        path = output_dir / f"{basename}.{fmt}"
        fig.savefig(path, dpi=dpi)
        saved.append(path)
    plt.close(fig)
    return saved


def _mode_basename(spec: TraceExportSpec, *, sector_mode: str, pooled: bool = False) -> str:
    basename = spec.basename
    if sector_mode == "sector-per-image":
        basename = basename.replace("sector_average", "sector_per_image")
    if pooled:
        basename = basename.replace("_examples_sem", "_pooled_examples_sem")
    return basename


def _diagonal_basename(spec: TraceExportSpec, *, sector_mode: str, pooled: bool = False) -> str:
    diagonal_key = str(DIAGONAL_SPECS[spec.key]["key"])
    mode_prefix = "diagonal_per_image" if sector_mode == "sector-per-image" else "diagonal_average"
    basename = f"{mode_prefix}_{diagonal_key}_{spec.folder}_examples_sem"
    if pooled:
        basename = basename.replace("_examples_sem", "_pooled_examples_sem")
    return basename


def export_real_sector_average_traces(
    *,
    data_dir: Path,
    output_dir: Path,
    threshold: float,
    formats: tuple[str, ...],
    dpi: int,
    sector_modes: tuple[str, ...],
    specs: tuple[TraceExportSpec, ...] = DEFAULT_SPECS,
) -> list[Path]:
    saved: list[Path] = []
    transition_cache: dict[Path, pd.DataFrame] = {}

    invalid_modes = set(sector_modes) - set(SECTOR_MODES)
    if invalid_modes:
        raise ValueError(f"Unknown sector modes: {sorted(invalid_modes)}")

    for spec in specs:
        transition_path = data_dir / spec.transition_csv
        trace_path = data_dir / spec.trace_csv
        if transition_path not in transition_cache:
            transition_cache[transition_path] = th.load_transition_table(transition_path)

        transition_table = transition_cache[transition_path]
        trace_table = _load_trace_table(trace_path, image_group=spec.image_group)

        for sector_mode in sector_modes:
            if sector_mode == "sector-average":
                sector_summary = _mean_sector_summary(
                    transition_table,
                    spec=spec,
                    threshold=threshold,
                )
                key_cols = ["neuron_idx"]
            else:
                sector_summary = _per_image_sector_summary(
                    transition_table,
                    spec=spec,
                    threshold=threshold,
                )
                key_cols = ["neuron_idx", "image_idx_original", "image_idx_within_group"]

            labeled = _attach_sector_labels(trace_table, sector_summary, key_cols=key_cols)
            trace_summary = _summarize_traces(labeled, spec=spec)
            pooled_summary = _summarize_pooled_traces(labeled, spec=spec)

            spec_dir = output_dir / sector_mode / spec.folder
            spec_dir.mkdir(parents=True, exist_ok=True)

            basename = _mode_basename(spec, sector_mode=sector_mode)
            csv_path = spec_dir / f"{basename}.csv"
            trace_summary.to_csv(csv_path, index=False)
            saved.append(csv_path)
            saved.extend(
                _plot_sector_average_panel(
                    trace_summary,
                    spec=spec,
                    output_dir=spec_dir,
                    basename=basename,
                    formats=formats,
                    dpi=dpi,
                )
            )

            pooled_basename = _mode_basename(spec, sector_mode=sector_mode, pooled=True)
            pooled_csv_path = spec_dir / f"{pooled_basename}.csv"
            pooled_summary.to_csv(pooled_csv_path, index=False)
            saved.append(pooled_csv_path)
            saved.extend(
                _plot_pooled_sector_panel(
                    pooled_summary,
                    spec=spec,
                    output_dir=spec_dir,
                    basename=pooled_basename,
                    formats=formats,
                    dpi=dpi,
                )
            )

            diagonal = DIAGONAL_SPECS[spec.key]
            diagonal_order = (str(diagonal["key"]),)
            diagonal_labels = {str(diagonal["key"]): str(diagonal["label"])}
            diagonal_assignments = _diagonal_assignments(
                sector_summary,
                spec=spec,
                key_cols=key_cols,
                threshold=threshold,
            )
            if diagonal_assignments.empty:
                continue
            diagonal_labeled = _attach_sector_labels(
                trace_table,
                diagonal_assignments,
                key_cols=key_cols,
                require_all=False,
            )
            diagonal_summary = _summarize_traces(
                diagonal_labeled,
                spec=spec,
                trace_order=diagonal_order,
                trace_labels=diagonal_labels,
            )
            diagonal_pooled_summary = _summarize_pooled_traces(
                diagonal_labeled,
                spec=spec,
                trace_order=diagonal_order,
                trace_labels=diagonal_labels,
            )
            diagonal_dir = output_dir / "diagonal-average" / sector_mode / spec.folder
            diagonal_dir.mkdir(parents=True, exist_ok=True)

            diagonal_basename = _diagonal_basename(spec, sector_mode=sector_mode)
            diagonal_csv_path = diagonal_dir / f"{diagonal_basename}.csv"
            diagonal_summary.to_csv(diagonal_csv_path, index=False)
            saved.append(diagonal_csv_path)
            saved.extend(
                _plot_sector_average_panel(
                    diagonal_summary,
                    spec=spec,
                    output_dir=diagonal_dir,
                    basename=diagonal_basename,
                    formats=formats,
                    dpi=dpi,
                    trace_order=diagonal_order,
                    trace_labels=diagonal_labels,
                    title=f"Real-data {diagonal['label']} diagonal traces +/- SEM",
                )
            )

            diagonal_pooled_basename = _diagonal_basename(spec, sector_mode=sector_mode, pooled=True)
            diagonal_pooled_csv_path = diagonal_dir / f"{diagonal_pooled_basename}.csv"
            diagonal_pooled_summary.to_csv(diagonal_pooled_csv_path, index=False)
            saved.append(diagonal_pooled_csv_path)
            saved.extend(
                _plot_pooled_sector_panel(
                    diagonal_pooled_summary,
                    spec=spec,
                    output_dir=diagonal_dir,
                    basename=diagonal_pooled_basename,
                    formats=formats,
                    dpi=dpi,
                    trace_order=diagonal_order,
                    trace_labels=diagonal_labels,
                    title=f"Real-data pooled {diagonal['label']} diagonal traces +/- SEM",
                )
            )

    return saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export real-data sector-average trace panels.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "real_sector_average_examples",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument(
        "--sector-mode",
        choices=(*SECTOR_MODES, "both"),
        default="both",
        help="Use group-level sectors, per-image sectors, or export both.",
    )
    parser.add_argument("--formats", nargs="+", default=["svg", "png"], choices=("png", "svg", "eps"))
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sector_modes = SECTOR_MODES if args.sector_mode == "both" else (args.sector_mode,)
    saved = export_real_sector_average_traces(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        formats=tuple(args.formats),
        dpi=args.dpi,
        sector_modes=sector_modes,
    )
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
