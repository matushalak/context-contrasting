"""Export familiar-sector transfer plots for mean-parameter and source-trace cells.

This post-hoc analysis uses familiar transition sectors to define three source
groups (+NO, +O, -NO). It then:

1. averages parameters within each familiar sector, runs one mean-parameter cell
   per sector, and plots those cells on novel images with expert ablations;
2. averages the already simulated traces of the same source neurons, producing
   by-image and familiar/novel-pooled trace figures without parameter averaging.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from context_contrasting.paper import transitions_helpers as th
from context_contrasting.paper.export_mean_field_sector_cells import (
    CONDITION_LABELS,
    CONDITION_ORDER,
    MEAN_FIELD_SECTORS,
    RESPONSE_COLORS,
    SHORT_COLUMN_LABELS,
    _collapse_familiar_conditions,
    _displacement_axis_limits,
    _load_model_inputs,
    _plot_displacement_scatter,
    _plot_trace_axis,
    _run_mean_field_traces,
    _trace_y_limits,
    build_mean_field_configs,
)
from context_contrasting.paper.export_expert_silencing_response_scatter import (
    _simulate_or_load_trace_scalars,
)
from context_contrasting.paper.visualize_s import TRANSITION_RESPONSE_COLUMN_SPECS


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = PACKAGE_DIR / "done-amen"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "figures" / "organized_sector_results" / "familiar_sector_novel_transfer"


def _familiar_colored_summaries(familiar: pd.DataFrame, novel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    familiar_labels = familiar[["neuron_idx", "RotatedSector"]].rename(columns={"RotatedSector": "familiar_sector"})
    familiar_colored = familiar.rename(columns={"RotatedSector": "familiar_sector"}).copy()
    novel_colored = (
        novel.drop(columns=["RotatedSector"], errors="ignore")
        .merge(familiar_labels, on="neuron_idx", how="left", validate="one_to_one")
    )
    if novel_colored["familiar_sector"].isna().any():
        missing = novel_colored.loc[novel_colored["familiar_sector"].isna(), "neuron_idx"].tolist()
        raise ValueError(f"Missing familiar sector labels for novel rows: {missing[:10]}")
    return familiar_colored, novel_colored


def _filter_conditions(trace_df: pd.DataFrame, conditions: tuple[str, ...]) -> pd.DataFrame:
    return trace_df.loc[trace_df["condition"].isin(conditions)].copy()


def render_transfer_figure(
    trace_df: pd.DataFrame,
    familiar_colored: pd.DataFrame,
    novel_colored_by_familiar: pd.DataFrame,
    *,
    output_path: Path,
    conditions: tuple[str, ...],
    title: str,
) -> None:
    column_specs = tuple(TRANSITION_RESPONSE_COLUMN_SPECS)
    n_trace_cols = len(conditions) * len(column_specs)
    sectors = [sector for sector in MEAN_FIELD_SECTORS if sector in set(trace_df["familiar_sector"])]
    n_rows = 1 + len(sectors)
    fig_width = max(10.0, 1.22 * n_trace_cols)
    fig_height = 3.0 + 1.55 * len(sectors)
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    grid = GridSpec(
        n_rows,
        n_trace_cols,
        figure=fig,
        height_ratios=[2.15] + [1.0] * len(sectors),
        hspace=0.82,
        wspace=0.42,
    )

    displacement_limits = _displacement_axis_limits(familiar_colored, novel_colored_by_familiar)
    split = max(1, n_trace_cols // 2)
    ax_fam = fig.add_subplot(grid[0, :split])
    ax_nov = fig.add_subplot(grid[0, split:])
    _plot_displacement_scatter(
        ax_fam,
        familiar_colored,
        title="Familiar transition displacement, colored by familiar sector",
        limits=displacement_limits,
    )
    _plot_displacement_scatter(
        ax_nov,
        novel_colored_by_familiar,
        title="Novel transition displacement, colored by familiar sector",
        limits=displacement_limits,
    )

    for row_idx, sector in enumerate(sectors, start=1):
        for condition_idx, condition in enumerate(conditions):
            segment_subset = trace_df[
                (trace_df["familiar_sector"] == sector)
                & (trace_df["condition"] == condition)
            ]
            segment_ylim = _trace_y_limits(segment_subset) if not segment_subset.empty else (-1.0, 1.0)
            for col_idx, column_spec in enumerate(column_specs):
                col = condition_idx * len(column_specs) + col_idx
                ax = fig.add_subplot(grid[row_idx, col])
                subset = trace_df[
                    (trace_df["familiar_sector"] == sector)
                    & (trace_df["condition"] == condition)
                    & (trace_df["column_key"] == column_spec["key"])
                ]
                _plot_trace_axis(ax, subset, sector=sector, ylim=segment_ylim)
                if row_idx == 1:
                    label = SHORT_COLUMN_LABELS.get(str(column_spec["label"]), str(column_spec["label"]))
                    if col_idx == 0:
                        label = f"{CONDITION_LABELS.get(condition, condition)}\n{label}"
                    ax.set_title(label, fontsize=6.5, pad=2)
                else:
                    ax.set_title("")
                if row_idx != len(sectors):
                    ax.set_xticklabels([])
                if col != 0:
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel(sector, color=th.ROTATED_SECTOR_PALETTE[sector], fontsize=9)

    fig.text(0.022, 0.37, "Z-scored response", rotation=90, va="center", fontsize=9)
    sector_handles = [
        Line2D([0], [0], color=th.ROTATED_SECTOR_PALETTE[sector], lw=2, marker="o", markersize=4, label=sector)
        for sector in th.ROTATED_SECTOR_ORDER
        if sector in set(familiar_colored["familiar_sector"].astype(str))
    ]
    response_handles = [
        Line2D([0], [0], color=RESPONSE_COLORS["NO"], lw=1.5, label="NO"),
        Line2D([0], [0], color=RESPONSE_COLORS["O"], lw=1.5, label="O"),
    ]
    fig.legend(
        handles=sector_handles + response_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=min(8, len(sector_handles) + len(response_handles)),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(title, y=0.995, fontsize=12)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.82, bottom=0.075)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _source_neuron_trace_averages(
    trace_df: pd.DataFrame,
    familiar_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = familiar_summary.loc[
        familiar_summary["RotatedSector"].astype(str).isin(MEAN_FIELD_SECTORS),
        ["neuron_idx", "RotatedSector"],
    ].rename(columns={"RotatedSector": "familiar_sector"})
    source = trace_df.merge(membership, on="neuron_idx", how="inner", validate="many_to_one")
    group_cols = [
        "familiar_sector",
        "condition",
        "column_key",
        "column_label",
        "experiment_phase",
        "response_type",
        "image_type",
        "x_seconds",
    ]
    averaged = (
        source.groupby(group_cols, as_index=False)
        .agg(
            y=("y", "mean"),
            stim_start_seconds=("stim_start_seconds", "first"),
            stim_end_seconds=("stim_end_seconds", "first"),
            n_trials=("n_trials", "sum"),
            n_source_cells=("neuron_idx", "nunique"),
        )
        .sort_values(["familiar_sector", "condition", "column_key", "response_type", "x_seconds"])
    )
    sector_cell_ids = {sector: idx + 1 for idx, sector in enumerate(MEAN_FIELD_SECTORS)}
    averaged["cell_id"] = averaged["familiar_sector"].map(sector_cell_ids).astype(int)
    averaged["sector_source"] = "familiar_source_trace_average"
    counts = (
        membership.groupby("familiar_sector", as_index=False)
        .agg(n_source_cells=("neuron_idx", "nunique"))
        .sort_values("familiar_sector")
    )
    return averaged, counts


def _write_run_summary(
    output_dir: Path,
    *,
    run_dir: Path,
    mean_params: pd.DataFrame,
    source_counts: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    summary: dict[str, Any] = {
        "source_run_dir": str(run_dir),
        "analysis": "familiar_sector_novel_transfer",
        "mean_parameter_cells": mean_params[
            ["sector_source", "sector", "source_sectors", "mean_field_cell_id", "n_source_cells"]
        ].to_dict(orient="records"),
        "source_trace_average_cells": source_counts.to_dict(orient="records"),
        "uses_ff_activity_accumulator": bool(
            metadata["pruned_mini_variant"]["ff_activity_accumulator"]["use_ff_activity_accumulator"]
        ),
        "ff_activity_accumulator": metadata["pruned_mini_variant"]["ff_activity_accumulator"],
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))


def export_familiar_sector_novel_transfer(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configs, familiar, novel, metadata = _load_model_inputs(run_dir, n_jobs=args.n_jobs)
    familiar_colored, novel_colored = _familiar_colored_summaries(familiar, novel)
    sector_threshold = float(metadata.get("sector_threshold", 0.3))

    familiar_configs, mean_params = build_mean_field_configs(
        configs,
        familiar,
        sector_source="familiar",
        threshold=sector_threshold,
        start_cell_id=1,
    )
    mean_trace_all = _run_mean_field_traces(familiar_configs, metadata=metadata)
    mean_trace_all.to_csv(output_dir / "familiar_sector_mean_parameter_traces_all_conditions.csv", index=False)
    mean_params.to_csv(output_dir / "familiar_sector_mean_parameter_cells.csv", index=False)

    mean_trace_novel = _filter_conditions(mean_trace_all, ("novel",))
    mean_trace_novel.to_csv(output_dir / "familiar_sector_mean_parameter_traces_novel.csv", index=False)
    for suffix in ("png", "svg"):
        render_transfer_figure(
            mean_trace_novel,
            familiar_colored,
            novel_colored,
            output_path=output_dir / f"familiar_sector_mean_parameter_novel_transfer.{suffix}",
            conditions=("novel",),
            title="Familiar-sector mean-parameter cells evaluated on novel images",
        )

    trace_cache_dir = run_dir / "figures" / "expert_silencing_response_scatter"
    trace_cache = trace_cache_dir / "expert_silencing_response_traces.csv"
    if not trace_cache.exists() or args.force_trace_cache:
        trace_cache_dir.mkdir(parents=True, exist_ok=True)
        _simulate_or_load_trace_scalars(run_dir=run_dir, output_dir=trace_cache_dir, n_jobs=args.n_jobs, force=True)
    all_cell_traces = pd.read_csv(trace_cache)
    source_trace_avg, source_counts = _source_neuron_trace_averages(all_cell_traces, familiar)
    source_trace_avg.to_csv(output_dir / "familiar_sector_source_neuron_trace_averages_by_image.csv", index=False)
    source_counts.to_csv(output_dir / "familiar_sector_source_neuron_counts.csv", index=False)

    source_trace_novel = _filter_conditions(source_trace_avg, ("novel",))
    source_trace_novel.to_csv(output_dir / "familiar_sector_source_neuron_trace_averages_novel.csv", index=False)
    source_trace_pooled = _collapse_familiar_conditions(source_trace_avg)
    source_trace_pooled.to_csv(output_dir / "familiar_sector_source_neuron_trace_averages_pooled_familiar_novel.csv", index=False)
    for suffix in ("png", "svg"):
        render_transfer_figure(
            source_trace_novel,
            familiar_colored,
            novel_colored,
            output_path=output_dir / f"familiar_sector_source_trace_average_novel_transfer.{suffix}",
            conditions=("novel",),
            title="Average source-neuron traces from familiar sectors evaluated on novel images",
        )
        render_transfer_figure(
            source_trace_avg,
            familiar_colored,
            novel_colored,
            output_path=output_dir / f"familiar_sector_source_trace_average_by_image.{suffix}",
            conditions=CONDITION_ORDER,
            title="Average traces of source neurons grouped by familiar transition sector",
        )
        render_transfer_figure(
            source_trace_pooled,
            familiar_colored,
            novel_colored,
            output_path=output_dir / f"familiar_sector_source_trace_average_pooled_familiar_novel.{suffix}",
            conditions=("familiar", "novel"),
            title="Average source-neuron traces pooled by image familiarity",
        )

    _write_run_summary(
        output_dir,
        run_dir=run_dir,
        mean_params=mean_params,
        source_counts=source_counts,
        metadata=metadata,
    )
    print(f"[familiar-sector-transfer] wrote outputs to {output_dir}")
    print("[familiar-sector-transfer] mean-parameter source cells:")
    print(mean_params[["sector", "source_sectors", "mean_field_cell_id", "n_source_cells"]].to_string(index=False))
    print("[familiar-sector-transfer] source trace-average cells:")
    print(source_counts.to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Saved model-scatter run directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--n-jobs", type=int, default=8, help="Parallel jobs for any required trace-cache simulation.")
    parser.add_argument("--force-trace-cache", action="store_true", help="Regenerate the all-cell trace cache first.")
    return parser


def main() -> None:
    export_familiar_sector_novel_transfer(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
