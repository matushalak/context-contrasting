"""Export NO-vs-O response scatters for naive, expert, and expert silencing.

Rows are image familiarity groups (familiar, novel). Columns are the standard
transition-response panel states: naive, expert, expert feedback off, expert PV
off, and expert feedback+PV off. Points are colored by the native rotated sector
for that image familiarity group.
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
from matplotlib.lines import Line2D

from context_contrasting.paper import transitions_helpers as th
from context_contrasting.paper.model_scatter import (
    PLOT_STYLE,
    RESPONSE_X_LABEL,
    RESPONSE_Y_LABEL,
    _run_sector_average_panel_config,
)
from context_contrasting.paper.visualize_s import TRANSITION_RESPONSE_COLUMN_SPECS


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = PACKAGE_DIR / "done-amen-final"
DEFAULT_OUTPUT_NAME = "expert_silencing_response_scatter"
IMAGE_GROUP_ORDER = ("familiar", "novel")
IMAGE_GROUP_LABELS = {"familiar": "Familiar", "novel": "Novel"}
COLUMN_KEYS = ("naive", "expert", "expert_no_fb", "expert_no_lat", "expert_no_fb_no_lat")
COLUMN_LABELS = {
    "naive": "Naive",
    "expert": "Expert",
    "expert_no_fb": "Expert FB off",
    "expert_no_lat": "Expert PV off",
    "expert_no_fb_no_lat": "Expert FB+PV off",
}
SECTOR_DRAW_ORDER = tuple(sector for sector in th._sector_plot_order(small_delta_first=True) if sector != "+NO axis") + (
    "+NO axis",
)


def _strip_saved_only_config_fields(config: dict[str, Any]) -> dict[str, Any]:
    clean = copy.deepcopy(config)
    clean.pop("activation", None)
    return clean


def _run_one_config(
    config: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    _transition, traces, _stimuli = _run_sector_average_panel_config(
        f"cell_{int(config['_sample_global_idx'])}",
        _strip_saved_only_config_fields(config),
        n_steps_per_phase=int(metadata["n_steps_per_phase"]),
        test_trials=int(metadata["test_trials"]),
        training_trials=int(metadata["training_trials"]),
        training_stimulus_order=str(metadata["training_stimulus_order"]),
        seed=int(metadata["seed"]),
        zscore_std_floor=float(metadata.get("zscore_std_floor", 0.04)),
    )
    traces = traces.copy()
    traces["neuron_idx"] = int(config["_sample_global_idx"])
    traces["transition"] = str(config["_canonical_transition"])
    return traces


def _simulate_or_load_trace_scalars(
    *,
    run_dir: Path,
    output_dir: Path,
    n_jobs: int,
    force: bool,
) -> pd.DataFrame:
    scalar_path = output_dir / "expert_silencing_response_scatter_values.csv"
    if scalar_path.exists() and not force:
        return pd.read_csv(scalar_path)

    metadata = json.loads((run_dir / "metadata.json").read_text())
    configs = json.loads((run_dir / "sampled_configs.json").read_text())
    print(f"[silencing-scatter] simulating {len(configs)} saved configs", flush=True)
    trace_frames = Parallel(n_jobs=n_jobs, verbose=10 if n_jobs != 1 else 0)(
        delayed(_run_one_config)(config, metadata=metadata) for config in configs
    )
    traces = pd.concat(trace_frames, ignore_index=True)
    traces.to_csv(output_dir / "expert_silencing_response_traces.csv", index=False)

    response_tail_fraction = float(metadata.get("response_tail_fraction", 1.0))
    stim_start = traces["stim_start_seconds"].astype(float)
    stim_end = traces["stim_end_seconds"].astype(float)
    tail_start = stim_start + (1.0 - response_tail_fraction) * (stim_end - stim_start)
    stimulus_rows = traces.loc[
        traces["column_key"].isin(COLUMN_KEYS)
        & (traces["x_seconds"].astype(float) >= tail_start)
        & (traces["x_seconds"].astype(float) <= stim_end)
    ].copy()
    stimulus_rows["image_group"] = np.where(stimulus_rows["condition"].isin(["familiar_1", "familiar_2"]), "familiar", "novel")

    scalar = (
        stimulus_rows.groupby(
            ["neuron_idx", "transition", "image_group", "column_key", "column_label", "response_type"],
            as_index=False,
        )
        .agg(response=("y", "mean"))
        .pivot_table(
            index=["neuron_idx", "transition", "image_group", "column_key", "column_label"],
            columns="response_type",
            values="response",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    scalar = scalar.rename(columns={"NO": "NO_response", "O": "O_response"})
    scalar.to_csv(scalar_path, index=False)
    return scalar


def _attach_native_sector_labels(scalar: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    labeled_frames: list[pd.DataFrame] = []
    for image_group in IMAGE_GROUP_ORDER:
        summary = pd.read_csv(run_dir / "summaries" / f"aggregate_{image_group}_summary.csv")
        label_cols = ["neuron_idx", "RotatedSector", "dNorm", "log_dNorm"]
        group_scalar = scalar.loc[scalar["image_group"].eq(image_group)].copy()
        group_scalar = group_scalar.merge(summary[label_cols], on="neuron_idx", how="left", validate="many_to_one")
        if group_scalar["RotatedSector"].isna().any():
            missing = group_scalar.loc[group_scalar["RotatedSector"].isna(), "neuron_idx"].drop_duplicates().tolist()
            raise ValueError(f"Missing {image_group} sector labels for neurons: {missing[:10]}")
        labeled_frames.append(group_scalar)
    return pd.concat(labeled_frames, ignore_index=True)


def _response_axis_limits(frame: pd.DataFrame, *, pad_fraction: float = 0.08) -> tuple[float, float]:
    values = frame[["NO_response", "O_response"]].to_numpy(dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    pad = pad_fraction * max(hi - lo, 1.0)
    return lo - pad, hi + pad


def _draw_response_scatter_axis(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    limits: tuple[float, float],
) -> None:
    if rows.empty:
        ax.set_axis_off()
        return
    log_norms = rows["log_dNorm"].to_numpy(dtype=float)
    alphas = th._map_norms_to_alphas(
        log_norms,
        min_alpha=PLOT_STYLE["alpha_min"],
        max_alpha=PLOT_STYLE["alpha_max"],
    )
    sectors = rows["RotatedSector"].astype(str).to_numpy()
    for sector in SECTOR_DRAW_ORDER:
        sector_mask = sectors == sector
        if not np.any(sector_mask):
            continue
        sector_rows = rows.loc[sector_mask]
        rgb = np.array(th.mcolors.to_rgb(th.ROTATED_SECTOR_PALETTE[sector])).reshape(1, 3)
        rgba = np.repeat(rgb, len(sector_rows), axis=0)
        rgba = np.concatenate([rgba, alphas[sector_mask].reshape(-1, 1)], axis=1)
        ax.scatter(
            sector_rows["NO_response"],
            sector_rows["O_response"],
            s=PLOT_STYLE["point_size"],
            c=rgba,
            edgecolors="none",
            zorder=th._sector_scatter_zorder(sector),
        )
    th._draw_diagonal(ax, list(limits))
    ax.axhline(0.0, color="0.85", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.85", linewidth=0.8, zorder=0)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8, length=2)


def _plot_response_scatter_grid(frame: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(
        len(IMAGE_GROUP_ORDER),
        len(COLUMN_KEYS),
        figsize=(14.0, 6.2),
        sharex=False,
        sharey=False,
        constrained_layout=False,
    )

    for row_idx, image_group in enumerate(IMAGE_GROUP_ORDER):
        for col_idx, column_key in enumerate(COLUMN_KEYS):
            ax = axes[row_idx, col_idx]
            rows = frame.loc[frame["image_group"].eq(image_group) & frame["column_key"].eq(column_key)]
            limits = _response_axis_limits(rows)
            _draw_response_scatter_axis(ax, rows, limits=limits)
            if row_idx == 0:
                ax.set_title(COLUMN_LABELS[column_key], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{IMAGE_GROUP_LABELS[image_group]}\nO", fontsize=10)
            else:
                ax.set_ylabel("")
            if row_idx == len(IMAGE_GROUP_ORDER) - 1:
                ax.set_xlabel("NO", fontsize=10)
            else:
                ax.set_xlabel("")

    sector_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=th.ROTATED_SECTOR_PALETTE[sector], markersize=5, label=sector)
        for sector in th.ROTATED_SECTOR_ORDER
    ]
    fig.legend(handles=sector_handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=5, frameon=False, fontsize=8)
    fig.suptitle("Modeled population NO/O response scatter under expert silencing", y=1.03, fontsize=13)
    fig.supxlabel(RESPONSE_X_LABEL, fontsize=11, y=0.02)
    fig.supylabel(RESPONSE_Y_LABEL, fontsize=11, x=0.015)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.11, top=0.82, wspace=0.18, hspace=0.2)
    for fmt in ("png", "svg"):
        fig.savefig(output_dir / f"expert_silencing_response_scatter.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_expert_silencing_response_scatter(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "figures" / DEFAULT_OUTPUT_NAME).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scalar = _simulate_or_load_trace_scalars(
        run_dir=run_dir,
        output_dir=output_dir,
        n_jobs=args.n_jobs,
        force=args.force,
    )
    labeled = _attach_native_sector_labels(scalar, run_dir)
    labeled.to_csv(output_dir / "expert_silencing_response_scatter_values_labeled.csv", index=False)
    _plot_response_scatter_grid(labeled, output_dir)
    print(f"[silencing-scatter] wrote outputs to {output_dir}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Rerun simulations even if cached scalar values exist.")
    return parser


def main() -> None:
    export_expert_silencing_response_scatter(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
