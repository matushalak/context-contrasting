"""Run mean-field model cells defined by image-familiarity-specific sectors.

This is an additive post-hoc analysis for the canonical model-scatter output. It
uses rotated sectors to group the saved model population, averages each group's
cell parameters, then runs one synthetic "mean-field" cell per non-empty large
sector with the standard expert ablations. Familiar panels use cells averaged
from familiar-sector groups; novel panels use cells averaged from novel-sector
groups.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from context_contrasting.paper import transitions_helpers as th
from context_contrasting.paper.example_selection import (
    DIAGONAL_ANGLES,
    NEAR_DIAGONAL_MAX_DISTANCE,
    _angle_distance,
)
from context_contrasting.paper.model_scatter import (
    _build_model_scatter_test_stimuli,
    _build_model_scatter_training_stimuli,
    _run_sample,
    _run_sector_average_panel_config,
    _transition_table,
    _wide_table,
)
from context_contrasting.paper.visualize_s import TRANSITION_RESPONSE_COLUMN_SPECS


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = PACKAGE_DIR / "done-amen-final"
DEFAULT_OUTPUT_NAME = "mean_field_sector_cells"
MEAN_FIELD_SECTORS = ("+NO axis", "+O axis", "-NO axis")
CONDITION_ORDER = ("familiar_1", "familiar_2", "novel")
CONDITION_LABELS = {
    "familiar_1": "Familiar 1",
    "familiar_2": "Familiar 2",
    "familiar": "Familiar",
    "novel": "Novel",
}
SHORT_COLUMN_LABELS = {
    "Naive": "Naive",
    "Expert": "Expert",
    "FB silencing": "FB off",
    "PV silencing": "PV off",
    "FB & PV silencing": "FB+PV off",
}
RESPONSE_COLORS = {"NO": "black", "O": "red"}
SECTOR_SOURCE_LABELS = {"familiar": "Familiar", "novel": "Novel"}
SECTOR_DIAGONAL_MEMBER_OVERRIDES = {
    ("novel", "+O axis"): {
        "token": "+NO/+O",
        "axis_tokens": ("+NO", "+O"),
        "rotated_sectors": ("+NO axis", "+O axis"),
    },
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _average_values(values: list[Any], *, bool_threshold: float = 0.5) -> Any:
    first = values[0]
    if isinstance(first, bool):
        return bool(float(np.mean(values)) >= bool_threshold)
    if _is_number(first):
        return float(np.mean(values))
    if isinstance(first, str):
        unique = {str(value) for value in values}
        if len(unique) != 1:
            raise ValueError(f"Cannot average non-constant string values: {sorted(unique)}")
        return first
    if isinstance(first, list):
        if not all(isinstance(value, list) and len(value) == len(first) for value in values):
            raise ValueError("Cannot average lists with mismatched lengths.")
        return [_average_values([value[i] for value in values], bool_threshold=bool_threshold) for i in range(len(first))]
    if isinstance(first, tuple):
        lists = [list(value) for value in values]
        return tuple(_average_values([value[i] for value in lists], bool_threshold=bool_threshold) for i in range(len(first)))
    if isinstance(first, dict):
        keys = set(first)
        if not all(isinstance(value, dict) and set(value) == keys for value in values):
            raise ValueError("Cannot average dicts with mismatched keys.")
        return {key: _average_values([value[key] for value in values], bool_threshold=bool_threshold) for key in first}
    raise TypeError(f"Unsupported parameter type for averaging: {type(first)!r}")


def _coerce_constructor_config(config: dict[str, Any], *, sector: str, sector_source: str, cell_id: int) -> dict[str, Any]:
    coerced = copy.deepcopy(config)
    for key in ("n_features", "n_pv", "n_context", "seed"):
        if key in coerced:
            coerced[key] = int(round(float(coerced[key])))
    coerced.pop("activation", None)
    coerced["_canonical_transition"] = f"mean_field_{_slugify(sector_source)}_{_slugify(sector)}"
    coerced["_sample_idx"] = 1
    coerced["_sample_global_idx"] = cell_id
    return coerced


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), child, out)
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            _flatten(f"{prefix}_{i}", child, out)
    else:
        out[prefix] = value


def _slugify(value: str) -> str:
    return (
        value.lower()
        .replace("+", "plus_")
        .replace("-", "minus_")
        .replace("∆", "delta")
        .replace(" ", "_")
        .replace("__", "_")
        .strip("_")
    )


def _load_model_inputs(
    run_dir: Path,
    *,
    n_jobs: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metadata = json.loads((run_dir / "metadata.json").read_text())
    configs = json.loads((run_dir / "sampled_configs.json").read_text())
    familiar_path = run_dir / "summaries" / "aggregate_familiar_summary.csv"
    novel_path = run_dir / "summaries" / "aggregate_novel_summary.csv"
    if familiar_path.exists() and novel_path.exists():
        familiar = pd.read_csv(familiar_path)
        novel = pd.read_csv(novel_path)
    else:
        familiar, novel = _reconstruct_summaries_from_configs(configs, metadata, run_dir=run_dir, n_jobs=n_jobs)
    if len(configs) != len(familiar):
        raise ValueError(f"Config count ({len(configs)}) does not match familiar summary rows ({len(familiar)}).")
    return configs, familiar, novel, metadata


def _reconstruct_summaries_from_configs(
    configs: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    run_dir: Path,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[mean-field] reconstructing missing aggregate summaries from saved configs", flush=True)
    test_stimuli = _build_model_scatter_test_stimuli(
        n_steps_per_phase=int(metadata["n_steps_per_phase"]),
        n_trials=int(metadata["test_trials"]),
    )
    training_stimuli = _build_model_scatter_training_stimuli(
        n_steps_per_phase=int(metadata["n_steps_per_phase"]),
        n_trials=int(metadata["training_trials"]),
        order=str(metadata["training_stimulus_order"]),
        seed=int(metadata["seed"]),
    )
    clean_configs = [_strip_saved_only_config_fields(config) for config in configs]
    response_frames = Parallel(n_jobs=n_jobs, verbose=10 if n_jobs != 1 else 0)(
        delayed(_run_sample)(
            config,
            n_steps_per_phase=int(metadata["n_steps_per_phase"]),
            response_tail_fraction=float(metadata.get("response_tail_fraction", 1.0)),
            test_stimuli=test_stimuli,
            training_stimuli=training_stimuli,
            zscore_std_floor=float(metadata.get("zscore_std_floor", 0.04)),
        )
        for config in clean_configs
    )
    transition_table = _transition_table(pd.concat(response_frames, ignore_index=True))
    wide = _wide_table(transition_table)
    threshold = float(metadata.get("sector_threshold", 0.3))
    familiar = th.build_mean_summary(
        wide,
        image_group="familiar",
        pre_stage="Naive",
        target_stage="Expert",
        threshold=threshold,
    )
    novel = th.build_mean_summary(
        wide,
        image_group="novel",
        pre_stage="Naive",
        target_stage="Expert",
        threshold=threshold,
    )
    config_meta = pd.DataFrame(
        {
            "neuron_idx": [int(config["_sample_global_idx"]) for config in configs],
            "transition": [config["_canonical_transition"] for config in configs],
            "sample_order": [int(config["_sample_global_idx"]) for config in configs],
        }
    )
    familiar = familiar.merge(config_meta, on="neuron_idx", how="left", validate="one_to_one")
    novel = novel.merge(config_meta, on="neuron_idx", how="left", validate="one_to_one")
    summaries_dir = run_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    familiar.to_csv(summaries_dir / "aggregate_familiar_summary.csv", index=False)
    novel.to_csv(summaries_dir / "aggregate_novel_summary.csv", index=False)
    transition_table.to_csv(run_dir / "transition_table_reconstructed.csv", index=False)
    return familiar, novel


def _strip_saved_only_config_fields(config: dict[str, Any]) -> dict[str, Any]:
    clean = copy.deepcopy(config)
    clean.pop("activation", None)
    return clean


def _member_rows_for_mean_field_cell(
    summary: pd.DataFrame,
    *,
    sector_source: str,
    sector: str,
    threshold: float,
) -> tuple[pd.DataFrame, str]:
    diagonal_spec = SECTOR_DIAGONAL_MEMBER_OVERRIDES.get((sector_source, sector))
    if diagonal_spec is None:
        return summary.loc[summary["RotatedSector"].astype(str).eq(sector)].copy(), sector

    diagonal_angle = DIAGONAL_ANGLES[frozenset(diagonal_spec["axis_tokens"])]
    rows = summary.copy()
    rows["dNorm_float"] = rows["dNorm"].astype(float)
    rows["diagonal_distance"] = _angle_distance(rows["Angle"], float(diagonal_angle))
    member_rows = rows.loc[
        rows["RotatedSector"].astype(str).isin(diagonal_spec["rotated_sectors"])
        & (rows["dNorm_float"] > threshold)
        & (rows["diagonal_distance"] <= NEAR_DIAGONAL_MAX_DISTANCE)
    ].copy()
    return member_rows, f"diagonal:{diagonal_spec['token']}"


def build_mean_field_configs(
    configs: list[dict[str, Any]],
    summary: pd.DataFrame,
    *,
    sector_source: str,
    threshold: float,
    sectors: tuple[str, ...] = MEAN_FIELD_SECTORS,
    start_cell_id: int = 1,
) -> tuple[dict[tuple[str, str], dict[str, Any]], pd.DataFrame]:
    config_by_id = {int(config["_sample_global_idx"]): config for config in configs}
    records: list[dict[str, Any]] = []
    mean_configs: dict[tuple[str, str], dict[str, Any]] = {}
    excluded_keys = {"activation"}

    for offset, sector in enumerate(sectors):
        cell_id = start_cell_id + offset
        member_rows, source_sectors = _member_rows_for_mean_field_cell(
            summary,
            sector_source=sector_source,
            sector=sector,
            threshold=threshold,
        )
        member_ids = member_rows["neuron_idx"].astype(int).tolist()
        if not member_ids:
            continue
        member_configs = [config_by_id[member_id] for member_id in member_ids]
        keys = sorted(set(member_configs[0]) - excluded_keys)
        public_keys = [key for key in keys if not key.startswith("_")]
        averaged = {
            key: _average_values([config[key] for config in member_configs])
            for key in public_keys
            if all(key in config for config in member_configs)
        }
        coerced = _coerce_constructor_config(averaged, sector=sector, sector_source=sector_source, cell_id=cell_id)
        mean_configs[(sector_source, sector)] = coerced

        flat: dict[str, Any] = {
            "sector_source": sector_source,
            "sector": sector,
            "source_sectors": source_sectors,
            "mean_field_cell_id": cell_id,
            "n_source_cells": len(member_ids),
        }
        if "diagonal_distance" in member_rows:
            flat["mean_diagonal_distance"] = float(member_rows["diagonal_distance"].mean())
            flat["max_diagonal_distance"] = float(member_rows["diagonal_distance"].max())
        for key, value in averaged.items():
            _flatten(key, value, flat)
        flat["applied_seed"] = coerced.get("seed")
        flat["applied_receives_context"] = json.dumps(coerced.get("receives_context"))
        records.append(flat)

    return mean_configs, pd.DataFrame(records)


def _run_mean_field_traces(
    configs_by_sector: dict[tuple[str, str], dict[str, Any]],
    *,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for (sector_source, sector), config in configs_by_sector.items():
        _, trace_df, _ = _run_sector_average_panel_config(
            f"{sector_source}:{sector}",
            config,
            n_steps_per_phase=int(metadata["n_steps_per_phase"]),
            test_trials=int(metadata["test_trials"]),
            training_trials=int(metadata["training_trials"]),
            training_stimulus_order=str(metadata["training_stimulus_order"]),
            seed=int(metadata["seed"]),
            zscore_std_floor=float(metadata.get("zscore_std_floor", 0.04)),
        )
        trace_df = trace_df.copy()
        trace_df["sector_source"] = sector_source
        trace_df["familiar_sector"] = sector
        frames.append(trace_df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _collapse_familiar_conditions(trace_df: pd.DataFrame) -> pd.DataFrame:
    familiar_df = trace_df[trace_df["condition"].isin(["familiar_1", "familiar_2"])].copy()
    novel_df = trace_df[trace_df["condition"] == "novel"].copy()
    group_cols = [
        "cell_id",
        "sector_source",
        "familiar_sector",
        "column_key",
        "column_label",
        "experiment_phase",
        "response_type",
        "image_type",
        "x_seconds",
    ]
    collapsed = (
        familiar_df.groupby(group_cols, as_index=False)
        .agg(
            y=("y", "mean"),
            stim_start_seconds=("stim_start_seconds", "first"),
            stim_end_seconds=("stim_end_seconds", "first"),
            n_trials=("n_trials", "sum"),
        )
        .assign(condition="familiar")
    )
    novel_df = novel_df.assign(condition="novel")
    return pd.concat([collapsed, novel_df], ignore_index=True, sort=False)


def _select_matching_sector_source(trace_df: pd.DataFrame) -> pd.DataFrame:
    familiar_conditions = trace_df["condition"].isin(["familiar_1", "familiar_2", "familiar"])
    novel_conditions = trace_df["condition"].eq("novel")
    return trace_df.loc[
        (familiar_conditions & trace_df["sector_source"].eq("familiar"))
        | (novel_conditions & trace_df["sector_source"].eq("novel"))
    ].copy()


def _displacement_axis_limits(*frames: pd.DataFrame) -> tuple[float, float]:
    x_values: list[float] = []
    y_values: list[float] = []
    for frame in frames:
        x_values.extend(frame["dNO"].to_numpy(dtype=float))
        y_values.extend(frame["dO"].to_numpy(dtype=float))
    extent = float(np.nanmax(np.abs(np.asarray(x_values + y_values, dtype=float))))
    if not np.isfinite(extent) or extent == 0.0:
        extent = 0.5
    extent *= 1.12
    return -extent, extent


def _plot_displacement_scatter(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    title: str,
    limits: tuple[float, float],
) -> None:
    guide = np.asarray(limits, dtype=float)
    ax.plot(guide, guide, "--", color="0.82", linewidth=0.8, zorder=0)
    ax.plot(guide, -guide, "--", color="0.82", linewidth=0.8, zorder=0)
    ax.axhline(0.0, color="0.72", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.72", linewidth=0.8, zorder=0)
    for sector in th.ROTATED_SECTOR_ORDER:
        sector_df = frame[frame["familiar_sector"] == sector]
        if sector_df.empty:
            continue
        color = th.ROTATED_SECTOR_PALETTE[sector]
        ax.quiver(
            np.zeros(len(sector_df)),
            np.zeros(len(sector_df)),
            sector_df["dNO"].to_numpy(dtype=float),
            sector_df["dO"].to_numpy(dtype=float),
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            alpha=0.22,
            width=0.003,
            zorder=1,
        )
        ax.scatter(
            sector_df["dNO"],
            sector_df["dO"],
            s=15,
            color=color,
            alpha=0.68,
            linewidths=0,
            zorder=2,
        )
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("dNO")
    ax.set_ylabel("dO")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _trace_y_limits(trace_df: pd.DataFrame) -> tuple[float, float]:
    y_min = float(trace_df["y"].min())
    y_max = float(trace_df["y"].max())
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return (-1.0, 1.0)
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    pad = 0.08 * (y_max - y_min)
    return y_min - pad, y_max + pad


def _plot_trace_axis(ax: plt.Axes, subset: pd.DataFrame, *, sector: str, ylim: tuple[float, float]) -> None:
    if subset.empty:
        ax.set_axis_off()
        return
    stim_start = float(subset["stim_start_seconds"].iloc[0])
    stim_end = float(subset["stim_end_seconds"].iloc[0])
    ax.axvspan(stim_start, stim_end, color="0.92", linewidth=0, zorder=0)
    ax.axhline(0.0, color="0.78", linewidth=0.6, zorder=1)
    for response_type in ("NO", "O"):
        trace = subset[subset["response_type"] == response_type].sort_values("x_seconds")
        if trace.empty:
            continue
        ax.plot(
            trace["x_seconds"],
            trace["y"],
            color=RESPONSE_COLORS[response_type],
            linestyle="-",
            linewidth=1.25,
            alpha=0.95,
            zorder=2,
        )
    ax.set_xlim(float(subset["x_seconds"].min()), float(subset["x_seconds"].max()))
    ax.set_ylim(*ylim)
    ax.tick_params(axis="both", labelsize=6, length=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_combined_figure(
    trace_df: pd.DataFrame,
    familiar: pd.DataFrame,
    novel: pd.DataFrame,
    *,
    output_path: Path,
    conditions: tuple[str, ...],
    title: str,
) -> None:
    column_specs = tuple(TRANSITION_RESPONSE_COLUMN_SPECS)
    n_trace_cols = len(conditions) * len(column_specs)
    n_rows = 1 + trace_df["familiar_sector"].nunique()
    fig_width = max(14.0, 1.18 * n_trace_cols)
    fig_height = 3.1 + 1.55 * (n_rows - 1)
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    grid = GridSpec(
        n_rows,
        n_trace_cols,
        figure=fig,
        height_ratios=[2.15] + [1.0] * (n_rows - 1),
        hspace=0.82,
        wspace=0.42,
    )

    displacement_limits = _displacement_axis_limits(familiar, novel)
    split = n_trace_cols // 2
    ax_fam = fig.add_subplot(grid[0, :split])
    ax_nov = fig.add_subplot(grid[0, split:])
    _plot_displacement_scatter(
        ax_fam,
        familiar,
        title="Familiar transition displacement, colored by familiar sector",
        limits=displacement_limits,
    )
    _plot_displacement_scatter(
        ax_nov,
        novel,
        title="Novel transition displacement, colored by novel sector",
        limits=displacement_limits,
    )

    sectors = [sector for sector in MEAN_FIELD_SECTORS if sector in set(trace_df["familiar_sector"])]
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

    familiar_sector_values = set(familiar["familiar_sector"].astype(str))
    sector_handles = [
        Line2D([0], [0], color=th.ROTATED_SECTOR_PALETTE[sector], lw=2, marker="o", markersize=4, label=sector)
        for sector in th.ROTATED_SECTOR_ORDER
        if sector in familiar_sector_values
    ]
    response_handles = [
        Line2D([0], [0], color="black", lw=1.5, linestyle="-", label="NO"),
        Line2D([0], [0], color="red", lw=1.5, linestyle="-", label="O"),
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


def export_mean_field_sector_cells(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / DEFAULT_OUTPUT_NAME).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configs, familiar, novel, metadata = _load_model_inputs(run_dir, n_jobs=args.n_jobs)
    sector_threshold = float(metadata.get("sector_threshold", 0.3))
    familiar_configs, familiar_params = build_mean_field_configs(
        configs,
        familiar,
        sector_source="familiar",
        threshold=sector_threshold,
        start_cell_id=1,
    )
    novel_configs, novel_params = build_mean_field_configs(
        configs,
        novel,
        sector_source="novel",
        threshold=sector_threshold,
        start_cell_id=1 + len(MEAN_FIELD_SECTORS),
    )
    mean_configs = {**familiar_configs, **novel_configs}
    mean_params = pd.concat([familiar_params, novel_params], ignore_index=True)
    if not mean_configs:
        raise RuntimeError("No non-empty familiar or novel sectors were available for mean-field cells.")

    all_trace_df = _run_mean_field_traces(mean_configs, metadata=metadata)
    trace_df = _select_matching_sector_source(all_trace_df)
    familiar_colored = familiar.rename(columns={"RotatedSector": "familiar_sector"}).copy()
    novel_colored = novel.rename(columns={"RotatedSector": "familiar_sector"}).copy()
    pooled_df = _collapse_familiar_conditions(trace_df)

    mean_params.to_csv(output_dir / "mean_field_sector_parameters.csv", index=False)
    all_trace_df.to_csv(output_dir / "mean_field_sector_traces_by_image_all_parameter_sets.csv", index=False)
    trace_df.to_csv(output_dir / "mean_field_sector_traces_by_image.csv", index=False)
    pooled_df.to_csv(output_dir / "mean_field_sector_traces_pooled_familiar_novel.csv", index=False)

    render_combined_figure(
        trace_df,
        familiar_colored,
        novel_colored,
        output_path=output_dir / "mean_field_sector_cells_by_image.png",
        conditions=CONDITION_ORDER,
        title="Modeled population sectors and native mean-field sector-cell responses",
    )
    render_combined_figure(
        pooled_df,
        familiar_colored,
        novel_colored,
        output_path=output_dir / "mean_field_sector_cells_pooled_familiar_novel.png",
        conditions=("familiar", "novel"),
        title="Modeled population sectors and native mean-field responses pooled by image familiarity",
    )
    for png in (
        output_dir / "mean_field_sector_cells_by_image.png",
        output_dir / "mean_field_sector_cells_pooled_familiar_novel.png",
    ):
        render_combined_figure(
            trace_df if "by_image" in png.name else pooled_df,
            familiar_colored,
            novel_colored,
            output_path=png.with_suffix(".svg"),
            conditions=CONDITION_ORDER if "by_image" in png.name else ("familiar", "novel"),
            title=(
                "Modeled population sectors and native mean-field sector-cell responses"
                if "by_image" in png.name
                else "Modeled population sectors and native mean-field responses pooled by image familiarity"
            ),
        )

    print(f"[mean-field] wrote outputs to {output_dir}")
    print("[mean-field] source cells per sector source and sector:")
    print(mean_params[["sector_source", "sector", "mean_field_cell_id", "n_source_cells"]].to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Saved model-scatter run directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory; defaults under --run-dir.")
    parser.add_argument("--n-jobs", type=int, default=8, help="Parallel jobs for reconstructing missing population summaries.")
    return parser


def main() -> None:
    export_mean_field_sector_cells(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
