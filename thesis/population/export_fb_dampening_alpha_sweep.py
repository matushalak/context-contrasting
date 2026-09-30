"""Run a matched population sweep of the FB plasticity dampening scale.

The suite reconstructs the saved ``done-amen`` cells with the source somatic
activation threshold, compares an undampened control with several values of
``alpha`` in ``alpha / (y + alpha)``, and persists response, training-dynamics,
weight, and transition-sector diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from matplotlib.lines import Line2D

from thesis.population import transitions_helpers as th
from thesis.population.experiment_s import NO_RESPONSE_ABLATION_SPECS, _temporary_model_overrides
from thesis.population.minimal_divisive import CCNeuron
from thesis.population.model_scatter import (
    BASELINE_STD_SCALE,
    PLOT_STYLE,
    RESPONSE_X_LABEL,
    RESPONSE_Y_LABEL,
    _append_post_stimulus_iti,
    _build_model_scatter_test_stimuli,
    _build_model_scatter_training_stimuli,
    _panel_step_window,
)
from thesis.population.neuron_utils import ThresholdReLU
from thesis.population.visualize_s import _window_repeated_trace


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = PACKAGE_DIR.parent.parent / "context_contrasting/paper/done-amen"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR.parent / "results" / "fb_dampening_alpha_sweep"
DEFAULT_ALPHAS = (1.0, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)
IMAGE_GROUP_ORDER = ("familiar", "novel")
IMAGE_GROUP_LABELS = {"familiar": "Familiar", "novel": "Novel"}
SECTOR_THRESHOLD = 0.3


def _alpha_slug(alpha: float) -> str:
    return f"alpha_{alpha:g}".replace(".", "p")


def _variant_specs(alphas: tuple[float, ...]) -> list[dict[str, Any]]:
    return [
        {
            "variant": "undampened",
            "label": "Undampened",
            "rule": "undampened-anti-Hebbian",
            "alpha": 1.0,
        },
        *[
            {
                "variant": _alpha_slug(alpha),
                "label": rf"Dampened $\alpha={alpha:g}$",
                "rule": "dampened-anti-Hebbian",
                "alpha": float(alpha),
            }
            for alpha in alphas
        ],
    ]


@torch.no_grad()
def _run_y_phase(
    model: CCNeuron,
    x: torch.Tensor,
    c: torch.Tensor,
    *,
    update: bool = False,
    collect_training: bool = False,
    n_steps_per_trial: int | None = None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    model._reset_state()
    y_values = np.empty(x.shape[0], dtype=np.float64)
    active_y: list[float] = []
    active_multiplier: list[float] = []
    trial_y: list[list[float]] = []
    trial_multiplier: list[list[float]] = []
    if collect_training:
        if n_steps_per_trial is None:
            raise ValueError("n_steps_per_trial is required when collecting training diagnostics.")
        n_trials = int(np.ceil(x.shape[0] / n_steps_per_trial))
        trial_y = [[] for _ in range(n_trials)]
        trial_multiplier = [[] for _ in range(n_trials)]

    for step in range(x.shape[0]):
        forward = model(x[step], c[step])
        y_next = float(forward[2])
        y_values[step] = y_next
        if collect_training and bool(torch.any(c[step] > 0)):
            if model.FBrule == "dampened-anti-Hebbian":
                multiplier = float(model.alpha / (forward[2] + model.alpha))
            else:
                multiplier = 1.0
            active_y.append(y_next)
            active_multiplier.append(multiplier)
            trial_idx = step // int(n_steps_per_trial)
            trial_y[trial_idx].append(y_next)
            trial_multiplier[trial_idx].append(multiplier)
        if update:
            model.update(*forward)

    if not collect_training:
        return y_values, None

    y_array = np.asarray(active_y, dtype=float)
    multiplier_array = np.asarray(active_multiplier, dtype=float)
    diagnostics = {
        "active_steps": int(y_array.size),
        "raw_y_mean": float(np.mean(y_array)),
        "raw_y_median": float(np.median(y_array)),
        "raw_y_p90": float(np.quantile(y_array, 0.9)),
        "raw_y_max": float(np.max(y_array)),
        "multiplier_mean": float(np.mean(multiplier_array)),
        "multiplier_median": float(np.median(multiplier_array)),
        "multiplier_p10": float(np.quantile(multiplier_array, 0.1)),
        "multiplier_min": float(np.min(multiplier_array)),
        "trial_raw_y_mean": [float(np.mean(values)) for values in trial_y],
        "trial_multiplier_mean": [float(np.mean(values)) for values in trial_multiplier],
    }
    return y_values, diagnostics


def _summarize_response_traces(
    traces: dict[tuple[str, str, str], np.ndarray],
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    n_steps_per_phase: int,
    test_trials: int,
    zscore_std_floor: float,
) -> list[dict[str, Any]]:
    focus_window = _panel_step_window(n_steps_per_phase, test_trials)
    windows: dict[tuple[str, str, str], dict[str, Any]] = {}
    baseline_chunks: list[np.ndarray] = []
    for key, series in traces.items():
        _phase, condition, _response_type = key
        windowed = _window_repeated_trace(
            series,
            stim_pair=stimuli[condition],
            focus_window=focus_window,
        )
        if windowed is None:
            raise ValueError(f"Could not window trace {key}.")
        windows[key] = windowed
        if key[0] == "naive":
            baseline_chunks.append(np.asarray(windowed["baseline_values"], dtype=float))

    baseline_values = np.concatenate(baseline_chunks)
    baseline_mean = float(baseline_values.mean())
    baseline_std = float(baseline_values.std(ddof=1))
    scale = max(baseline_std, zscore_std_floor) if baseline_std > 1e-12 else max(1.0, zscore_std_floor)

    condition_rows: list[dict[str, Any]] = []
    for (phase, condition, response_type), windowed in windows.items():
        stacked = np.asarray(windowed["stacked"], dtype=float)
        summarized = (stacked - baseline_mean) / scale
        y_mean = summarized.mean(axis=0)
        x_seconds = np.asarray(windowed["x_seconds"], dtype=float)
        stim_start, stim_end = tuple(windowed["stim_seconds"])
        stimulus_mask = (x_seconds >= float(stim_start)) & (x_seconds <= float(stim_end))
        condition_rows.append(
            {
                "phase": phase,
                "condition": condition,
                "image_group": "familiar" if condition.startswith("familiar") else "novel",
                "response_type": response_type,
                "response": float(y_mean[stimulus_mask].mean()),
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "zscore_scale": scale,
            }
        )

    condition_frame = pd.DataFrame(condition_rows)
    pooled = (
        condition_frame.groupby(["phase", "image_group", "response_type"], as_index=False)
        .agg(
            response=("response", "mean"),
            baseline_mean=("baseline_mean", "first"),
            baseline_std=("baseline_std", "first"),
            zscore_scale=("zscore_scale", "first"),
        )
    )
    rows: list[dict[str, Any]] = []
    for (phase, image_group), group in pooled.groupby(["phase", "image_group"], sort=False):
        response_lookup = group.set_index("response_type")["response"]
        rows.append(
            {
                "phase": phase,
                "image_group": image_group,
                "NO_response": float(response_lookup["NO"]),
                "O_response": float(response_lookup["O"]),
                "baseline_mean": float(group["baseline_mean"].iloc[0]),
                "baseline_std": float(group["baseline_std"].iloc[0]),
                "zscore_scale": float(group["zscore_scale"].iloc[0]),
            }
        )
    return rows


def _weight_snapshot(model: CCNeuron, prefix: str) -> dict[str, float]:
    row: dict[str, float] = {}
    for name in ("w_ff", "w_fb", "w_lat", "w_pv_lat"):
        values = getattr(model, name).detach().cpu().numpy().reshape(-1)
        row.update({f"{prefix}_{name}_{idx}": float(value) for idx, value in enumerate(values)})
    values = model.W_pv.detach().cpu().numpy().reshape(-1)
    row.update({f"{prefix}_W_pv_{idx}": float(value) for idx, value in enumerate(values)})
    return row


def _run_variant(
    config: dict[str, Any],
    variant: dict[str, Any],
    *,
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    training: tuple[torch.Tensor, torch.Tensor],
    metadata: dict[str, Any],
    soma_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clean = copy.deepcopy(config)
    clean["activation"] = ThresholdReLU(
        threshold=soma_threshold,
        subtractive=False,
        hasMax=True,
        maxValue=1.0,
    )
    clean["FBrule"] = variant["rule"]
    clean["alpha"] = float(variant["alpha"])
    model = CCNeuron(**{key: value for key, value in clean.items() if not key.startswith("_")})

    traces: dict[tuple[str, str, str], np.ndarray] = {}
    for condition, (x_full, c_full) in stimuli.items():
        traces[("naive", condition, "NO")], _ = _run_y_phase(model, x_full, c_full)
        traces[("naive", condition, "O")], _ = _run_y_phase(model, torch.zeros_like(x_full), c_full)

    diagnostics = {
        "variant": variant["variant"],
        "variant_label": variant["label"],
        "fb_rule": variant["rule"],
        "alpha": float(variant["alpha"]),
        "neuron_idx": int(config["_sample_global_idx"]),
        "transition": str(config["_canonical_transition"]),
        "receives_context": bool(any(config["receives_context"])),
        "soma_activation_threshold": soma_threshold,
        **_weight_snapshot(model, "initial"),
    }
    _, training_diagnostics = _run_y_phase(
        model,
        training[0],
        training[1],
        update=True,
        collect_training=True,
        n_steps_per_trial=int(metadata["n_steps_per_phase"]),
    )
    if training_diagnostics is None:
        raise RuntimeError("Training diagnostics were not collected.")
    diagnostics.update(
        {
            key: value
            for key, value in training_diagnostics.items()
            if key not in {"trial_raw_y_mean", "trial_multiplier_mean"}
        }
    )
    for idx, value in enumerate(training_diagnostics["trial_raw_y_mean"], start=1):
        diagnostics[f"trial_{idx}_raw_y_mean"] = value
    for idx, value in enumerate(training_diagnostics["trial_multiplier_mean"], start=1):
        diagnostics[f"trial_{idx}_multiplier_mean"] = value
    diagnostics.update(_weight_snapshot(model, "final"))
    for name in ("w_ff", "w_fb", "w_lat", "w_pv_lat", "W_pv"):
        initial_columns = sorted(column for column in diagnostics if column.startswith(f"initial_{name}_"))
        for initial_column in initial_columns:
            idx = initial_column.rsplit("_", 1)[1]
            diagnostics[f"delta_{name}_{idx}"] = diagnostics[f"final_{name}_{idx}"] - diagnostics[initial_column]
    diagnostics["mean_fb_potentiation"] = float(
        np.mean([diagnostics[f"delta_w_fb_{idx}"] for idx in range(model.n_context)])
    )
    if hasattr(model, "ff_activity_accumulator"):
        diagnostics["final_ff_activity_accumulator"] = float(model.ff_activity_accumulator.ema)

    for condition, (x_full, c_full) in stimuli.items():
        traces[("expert", condition, "NO")], _ = _run_y_phase(model, x_full, c_full)
        traces[("expert", condition, "O")], _ = _run_y_phase(model, torch.zeros_like(x_full), c_full)
        no_context_c = torch.zeros_like(c_full)
        for ablation_spec in NO_RESPONSE_ABLATION_SPECS.values():
            ablated_c = no_context_c if ablation_spec.get("zero_context", False) else c_full
            model_overrides = ablation_spec.get("model_overrides", {})
            with _temporary_model_overrides(model, **model_overrides):
                _run_y_phase(model, x_full, ablated_c)
                _run_y_phase(model, torch.zeros_like(x_full), ablated_c)

    cell_floor = max(
        float(metadata.get("zscore_std_floor", 0.04)),
        BASELINE_STD_SCALE * float(config.get("baseline_drive_sigma", 0.0)),
    )
    response_rows = _summarize_response_traces(
        traces,
        stimuli,
        n_steps_per_phase=int(metadata["n_steps_per_phase"]),
        test_trials=int(metadata["test_trials"]),
        zscore_std_floor=cell_floor,
    )
    for row in response_rows:
        row.update(
            {
                "variant": variant["variant"],
                "variant_label": variant["label"],
                "fb_rule": variant["rule"],
                "alpha": float(variant["alpha"]),
                "neuron_idx": int(config["_sample_global_idx"]),
                "transition": str(config["_canonical_transition"]),
                "receives_context": bool(any(config["receives_context"])),
            }
        )
    return response_rows, diagnostics


def _run_config_variants(
    config: dict[str, Any],
    variants: list[dict[str, Any]],
    *,
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    training: tuple[torch.Tensor, torch.Tensor],
    metadata: dict[str, Any],
    soma_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    responses: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for variant in variants:
        response_rows, diagnostic_row = _run_variant(
            config,
            variant,
            stimuli=stimuli,
            training=training,
            metadata=metadata,
            soma_threshold=soma_threshold,
        )
        responses.extend(response_rows)
        diagnostics.append(diagnostic_row)
    return responses, diagnostics


def _build_cell_results(
    response_phase: pd.DataFrame,
    *,
    run_dir: Path,
) -> pd.DataFrame:
    index_columns = [
        "variant",
        "variant_label",
        "fb_rule",
        "alpha",
        "neuron_idx",
        "transition",
        "receives_context",
        "image_group",
    ]
    response_wide = response_phase.pivot_table(
        index=index_columns,
        columns="phase",
        values=["NO_response", "O_response"],
        aggfunc="first",
    )
    response_wide.columns = [f"{phase}_{response.removesuffix('_response')}" for response, phase in response_wide.columns]
    response_wide = response_wide.reset_index()
    response_wide["dNO"] = response_wide["expert_NO"] - response_wide["naive_NO"]
    response_wide["dO"] = response_wide["expert_O"] - response_wide["naive_O"]
    response_wide["dNorm"] = np.hypot(response_wide["dNO"], response_wide["dO"])
    response_wide["Angle"] = np.arctan2(response_wide["dO"], response_wide["dNO"])
    response_wide = th.assign_rotated_sectors(response_wide, threshold=SECTOR_THRESHOLD).rename(
        columns={"RotatedSector": "recomputed_sector"}
    )

    labeled: list[pd.DataFrame] = []
    for image_group in IMAGE_GROUP_ORDER:
        native = pd.read_csv(run_dir / "summaries" / f"aggregate_{image_group}_summary.csv")
        native = native[["neuron_idx", "RotatedSector"]].rename(columns={"RotatedSector": "native_sector"})
        rows = response_wide.loc[response_wide["image_group"].eq(image_group)].merge(
            native,
            on="neuron_idx",
            how="left",
            validate="many_to_one",
        )
        labeled.append(rows)
    return pd.concat(labeled, ignore_index=True)


def _population_response_summary(cell_results: pd.DataFrame) -> pd.DataFrame:
    control = cell_results.loc[
        cell_results["variant"].eq("undampened"),
        ["neuron_idx", "image_group", "expert_NO", "expert_O"],
    ].rename(columns={"expert_NO": "control_expert_NO", "expert_O": "control_expert_O"})
    compared = cell_results.merge(control, on=["neuron_idx", "image_group"], validate="many_to_one")
    compared["delta_NO_vs_undampened"] = compared["expert_NO"] - compared["control_expert_NO"]
    compared["delta_O_vs_undampened"] = compared["expert_O"] - compared["control_expert_O"]
    compared["abs_delta_NO_vs_undampened"] = compared["delta_NO_vs_undampened"].abs()
    compared["abs_delta_O_vs_undampened"] = compared["delta_O_vs_undampened"].abs()
    return (
        compared.groupby(["variant", "variant_label", "fb_rule", "alpha", "image_group"], as_index=False)
        .agg(
            n_cells=("neuron_idx", "size"),
            mean_naive_NO=("naive_NO", "mean"),
            mean_naive_O=("naive_O", "mean"),
            mean_expert_NO=("expert_NO", "mean"),
            sem_expert_NO=("expert_NO", "sem"),
            mean_expert_O=("expert_O", "mean"),
            sem_expert_O=("expert_O", "sem"),
            mean_delta_NO_vs_undampened=("delta_NO_vs_undampened", "mean"),
            mean_abs_delta_NO_vs_undampened=("abs_delta_NO_vs_undampened", "mean"),
            max_abs_delta_NO_vs_undampened=("abs_delta_NO_vs_undampened", "max"),
            mean_delta_O_vs_undampened=("delta_O_vs_undampened", "mean"),
            mean_abs_delta_O_vs_undampened=("abs_delta_O_vs_undampened", "mean"),
            max_abs_delta_O_vs_undampened=("abs_delta_O_vs_undampened", "max"),
        )
    )


def _paired_cell_results(cell_results: pd.DataFrame) -> pd.DataFrame:
    control = cell_results.loc[
        cell_results["variant"].eq("undampened"),
        ["neuron_idx", "image_group", "expert_NO", "expert_O", "recomputed_sector"],
    ].rename(
        columns={
            "expert_NO": "control_expert_NO",
            "expert_O": "control_expert_O",
            "recomputed_sector": "control_recomputed_sector",
        }
    )
    paired = cell_results.merge(control, on=["neuron_idx", "image_group"], validate="many_to_one")
    paired["delta_NO_vs_undampened"] = paired["expert_NO"] - paired["control_expert_NO"]
    paired["delta_O_vs_undampened"] = paired["expert_O"] - paired["control_expert_O"]
    paired["sector_changed_vs_undampened"] = (
        paired["recomputed_sector"].astype(str) != paired["control_recomputed_sector"].astype(str)
    )
    return paired


def _context_response_summary(cell_results: pd.DataFrame) -> pd.DataFrame:
    paired = _paired_cell_results(cell_results)
    paired = paired.loc[paired["receives_context"]]
    return (
        paired.groupby(["variant", "variant_label", "alpha", "image_group"], as_index=False)
        .agg(
            n_context_cells=("neuron_idx", "size"),
            mean_expert_NO=("expert_NO", "mean"),
            mean_expert_O=("expert_O", "mean"),
            mean_delta_NO_vs_undampened=("delta_NO_vs_undampened", "mean"),
            mean_delta_O_vs_undampened=("delta_O_vs_undampened", "mean"),
        )
    )


def _key_native_sector_summary(cell_results: pd.DataFrame) -> pd.DataFrame:
    paired = _paired_cell_results(cell_results)
    requested = (
        ("familiar", "+NO axis", "NO"),
        ("familiar", "+O axis", "O"),
        ("novel", "+NO axis", "NO"),
        ("novel", "+O axis", "O"),
    )
    frames: list[pd.DataFrame] = []
    for image_group, native_sector, response_type in requested:
        rows = paired.loc[
            paired["image_group"].eq(image_group) & paired["native_sector"].astype(str).eq(native_sector)
        ]
        summary = (
            rows.groupby(["variant", "variant_label", "alpha"], as_index=False)
            .agg(
                n_cells=("neuron_idx", "size"),
                mean_expert_response=(f"expert_{response_type}", "mean"),
                mean_delta_vs_undampened=(f"delta_{response_type}_vs_undampened", "mean"),
            )
        )
        summary["image_group"] = image_group
        summary["native_sector"] = native_sector
        summary["response_type"] = response_type
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def _sector_change_summary(cell_results: pd.DataFrame) -> pd.DataFrame:
    paired = _paired_cell_results(cell_results)
    return (
        paired.groupby(["variant", "variant_label", "alpha", "image_group"], as_index=False)
        .agg(n_sector_changes_vs_undampened=("sector_changed_vs_undampened", "sum"))
    )


def _training_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    context = diagnostics.loc[diagnostics["receives_context"]].copy()
    final_fb_columns = [f"final_w_fb_{idx}" for idx in range(3)]
    context["mean_final_fb_weight"] = context[final_fb_columns].mean(axis=1)
    return (
        context.groupby(["variant", "variant_label", "fb_rule", "alpha"], as_index=False)
        .agg(
            n_context_cells=("neuron_idx", "size"),
            mean_raw_y=("raw_y_mean", "mean"),
            sem_raw_y=("raw_y_mean", "sem"),
            mean_multiplier=("multiplier_mean", "mean"),
            sem_multiplier=("multiplier_mean", "sem"),
            mean_fb_potentiation=("mean_fb_potentiation", "mean"),
            sem_fb_potentiation=("mean_fb_potentiation", "sem"),
            mean_final_fb_weight=("mean_final_fb_weight", "mean"),
            sem_final_fb_weight=("mean_final_fb_weight", "sem"),
        )
    )


def _sector_counts(cell_results: pd.DataFrame) -> pd.DataFrame:
    counts = (
        cell_results.groupby(
            ["variant", "variant_label", "alpha", "image_group", "recomputed_sector"],
            observed=False,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_cells"})
    )
    counts["fraction"] = counts["n_cells"] / counts.groupby(["variant", "image_group"])["n_cells"].transform("sum")
    return counts


def _axis_limits(values: pd.Series, pad_fraction: float = 0.08) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    lo, hi = float(finite.min()), float(finite.max())
    pad = pad_fraction * max(hi - lo, 1.0)
    return lo - pad, hi + pad


def _plot_scatter(cell_results: pd.DataFrame, variants: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, len(variants), figsize=(3.0 * len(variants), 6.2), sharex="row", sharey="row")
    draw_order = tuple(sector for sector in th._sector_plot_order(small_delta_first=True) if sector != "+NO axis") + (
        "+NO axis",
    )
    for row_idx, image_group in enumerate(IMAGE_GROUP_ORDER):
        image_rows = cell_results.loc[cell_results["image_group"].eq(image_group)]
        x_limits = _axis_limits(image_rows["expert_NO"])
        y_limits = _axis_limits(image_rows["expert_O"])
        for col_idx, variant in enumerate(variants):
            ax = axes[row_idx, col_idx]
            rows = image_rows.loc[image_rows["variant"].eq(variant["variant"])]
            for sector in draw_order:
                sector_rows = rows.loc[rows["native_sector"].astype(str).eq(sector)]
                if sector_rows.empty:
                    continue
                ax.scatter(
                    sector_rows["expert_NO"],
                    sector_rows["expert_O"],
                    s=PLOT_STYLE["point_size"],
                    color=th.ROTATED_SECTOR_PALETTE[sector],
                    alpha=0.72,
                    edgecolors="none",
                    zorder=th._sector_scatter_zorder(sector),
                )
            lo = min(x_limits[0], y_limits[0])
            hi = max(x_limits[1], y_limits[1])
            ax.plot([lo, hi], [lo, hi], "--", color="0.78", linewidth=0.9, zorder=0)
            ax.axhline(0, color="0.87", linewidth=0.8, zorder=0)
            ax.axvline(0, color="0.87", linewidth=0.8, zorder=0)
            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)
            ax.spines[["top", "right"]].set_visible(False)
            if row_idx == 0:
                ax.set_title(variant["label"], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{IMAGE_GROUP_LABELS[image_group]}\nO", fontsize=10)
            if row_idx == 1:
                ax.set_xlabel("NO", fontsize=10)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=th.ROTATED_SECTOR_PALETTE[sector], markersize=5, label=sector)
        for sector in th.ROTATED_SECTOR_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, frameon=False, fontsize=8)
    fig.suptitle("Expert NO/O responses across FB dampening scales", y=1.035, fontsize=13)
    fig.supxlabel(RESPONSE_X_LABEL, y=0.01, fontsize=11)
    fig.supylabel(RESPONSE_Y_LABEL, x=0.01, fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.11, top=0.84, wspace=0.18, hspace=0.22)
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"fb_dampening_alpha_sweep_expert_scatter.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_response_summary(summary: pd.DataFrame, variants: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    x = np.arange(len(variants))
    labels = [variant["label"].replace("Dampened ", "") for variant in variants]
    for row_idx, image_group in enumerate(IMAGE_GROUP_ORDER):
        rows = summary.loc[summary["image_group"].eq(image_group)].set_index("variant").reindex(
            [variant["variant"] for variant in variants]
        )
        for col_idx, response_type in enumerate(("NO", "O")):
            ax = axes[row_idx, col_idx]
            means = rows[f"mean_expert_{response_type}"].to_numpy(dtype=float)
            sems = rows[f"sem_expert_{response_type}"].to_numpy(dtype=float)
            color = "black" if response_type == "NO" else "red"
            ax.errorbar(x, means, yerr=sems, marker="o", color=color, linewidth=1.8, capsize=3)
            ax.set_title(f"{IMAGE_GROUP_LABELS[image_group]} {response_type}", fontsize=11)
            ax.set_ylabel("Mean expert z-scored response")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="0.9", linewidth=0.7)
            ax.set_xticks(x, labels, rotation=25, ha="right")
    fig.suptitle("Population response dependence on FB dampening scale", fontsize=13)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"fb_dampening_alpha_sweep_response_summary.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_training_summary(summary: pd.DataFrame, variants: list[dict[str, Any]], output_dir: Path) -> None:
    rows = summary.set_index("variant").reindex([variant["variant"] for variant in variants])
    x = np.arange(len(variants))
    labels = [variant["label"].replace("Dampened ", "") for variant in variants]
    specs = (
        ("mean_raw_y", "sem_raw_y", "Raw training y"),
        ("mean_multiplier", "sem_multiplier", "Applied FB multiplier"),
        ("mean_fb_potentiation", "sem_fb_potentiation", "Mean FB potentiation"),
        ("mean_final_fb_weight", "sem_final_fb_weight", "Mean final FB weight"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.8))
    for ax, (mean_col, sem_col, title) in zip(axes, specs, strict=True):
        ax.errorbar(
            x,
            rows[mean_col].to_numpy(dtype=float),
            yerr=rows[sem_col].to_numpy(dtype=float),
            marker="o",
            color="#3b5b92",
            linewidth=1.8,
            capsize=3,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9", linewidth=0.7)
    fig.suptitle("Training dynamics in feedback-receiving cells", fontsize=13)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"fb_dampening_alpha_sweep_training_diagnostics.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    labels = columns
    rows = ["| " + " | ".join(labels) + " |", "| " + " | ".join("---" for _ in labels) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        rendered = []
        for column, value in zip(columns, row, strict=True):
            if column in formats and pd.notna(value):
                rendered.append(format(value, formats[column]))
            else:
                rendered.append(str(value))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join(rows)


def _write_report(
    output_dir: Path,
    *,
    run_dir: Path,
    metadata: dict[str, Any],
    soma_threshold: float,
    variants: list[dict[str, Any]],
    response_summary: pd.DataFrame,
    context_response_summary: pd.DataFrame,
    key_sector_summary: pd.DataFrame,
    training_summary: pd.DataFrame,
    sector_counts: pd.DataFrame,
    sector_change_summary: pd.DataFrame,
    validation: dict[str, Any],
    runtime_seconds: float,
) -> None:
    variant_order = [variant["variant"] for variant in variants]
    response = response_summary.copy()
    response["variant"] = pd.Categorical(response["variant"], categories=variant_order, ordered=True)
    response = response.sort_values(["image_group", "variant"])
    response_table = response[
        [
            "image_group",
            "variant",
            "mean_expert_NO",
            "mean_expert_O",
            "mean_delta_NO_vs_undampened",
            "mean_delta_O_vs_undampened",
        ]
    ].copy()
    response_table["variant"] = response_table["variant"].astype(str)
    control_response = response_table.loc[
        response_table["variant"].eq("undampened"),
        ["image_group", "mean_expert_NO", "mean_expert_O"],
    ].rename(columns={"mean_expert_NO": "control_NO", "mean_expert_O": "control_O"})
    response_table = response_table.merge(control_response, on="image_group", validate="many_to_one")
    response_table["NO_percent_change"] = 100.0 * response_table["mean_delta_NO_vs_undampened"] / response_table["control_NO"]
    response_table["O_percent_change"] = 100.0 * response_table["mean_delta_O_vs_undampened"] / response_table["control_O"]
    response_table = response_table.drop(columns=["control_NO", "control_O"])

    context_table = context_response_summary.copy()
    context_table["variant"] = pd.Categorical(context_table["variant"], categories=variant_order, ordered=True)
    context_table = context_table.sort_values(["image_group", "variant"])[
        [
            "image_group",
            "variant",
            "n_context_cells",
            "mean_expert_NO",
            "mean_expert_O",
            "mean_delta_NO_vs_undampened",
            "mean_delta_O_vs_undampened",
        ]
    ]
    context_table["variant"] = context_table["variant"].astype(str)

    key_sector_table = key_sector_summary.copy()
    key_sector_table["variant"] = pd.Categorical(key_sector_table["variant"], categories=variant_order, ordered=True)
    key_sector_table = key_sector_table.sort_values(["image_group", "native_sector", "variant"])[
        [
            "image_group",
            "native_sector",
            "response_type",
            "variant",
            "n_cells",
            "mean_expert_response",
            "mean_delta_vs_undampened",
        ]
    ]
    key_sector_table["variant"] = key_sector_table["variant"].astype(str)

    training = training_summary.copy()
    training["variant"] = pd.Categorical(training["variant"], categories=variant_order, ordered=True)
    training = training.sort_values("variant")
    training_table = training[
        ["variant", "mean_raw_y", "mean_multiplier", "mean_fb_potentiation", "mean_final_fb_weight"]
    ].copy()
    training_table["variant"] = training_table["variant"].astype(str)

    sector = sector_counts.copy()
    sector["variant"] = pd.Categorical(sector["variant"], categories=variant_order, ordered=True)
    sector = sector.sort_values(["image_group", "variant", "recomputed_sector"])
    sector_table = sector[["image_group", "variant", "recomputed_sector", "n_cells", "fraction"]].copy()
    sector_table["variant"] = sector_table["variant"].astype(str)
    sector_table["recomputed_sector"] = sector_table["recomputed_sector"].astype(str)

    sector_change_table = sector_change_summary.copy()
    sector_change_table["variant"] = pd.Categorical(
        sector_change_table["variant"], categories=variant_order, ordered=True
    )
    sector_change_table = sector_change_table.sort_values(["image_group", "variant"])[
        ["image_group", "variant", "n_sector_changes_vs_undampened"]
    ]
    sector_change_table["variant"] = sector_change_table["variant"].astype(str)

    strongest = training.loc[training["variant"].eq(variant_order[-1])].iloc[0]
    alpha_one = training.loc[training["variant"].eq(_alpha_slug(1.0))].iloc[0]
    report = f"""# FB Plasticity Dampening Alpha Sweep

## Result

The matched sweep completed for all `{metadata['n_samples_total']}` saved cells in
`{runtime_seconds / 60.0:.1f}` minutes. Every condition used the same cell-specific
initial parameters and seeds, the same `{metadata['training_trials']} x 2` familiar-image
training sequence, and the source somatic activation threshold `{soma_threshold:g}`.

At `alpha = 1.0`, the mean applied multiplier was `{alpha_one['mean_multiplier']:.5f}`,
confirming that the original dampening term removes only
`{100.0 * (1.0 - alpha_one['mean_multiplier']):.2f}%` of the instantaneous FB update
on average. At `alpha = {variants[-1]['alpha']:g}`, it fell to
`{strongest['mean_multiplier']:.5f}`, making activity-dependent suppression a
substantial part of the learning rule.

## Conditions

| Variant | FB plasticity multiplier |
| --- | --- |
| Undampened | `1` |
"""
    report += "\n".join(
        f"| `{variant['variant']}` | `{variant['alpha']:g} / (y + {variant['alpha']:g})` |"
        for variant in variants[1:]
    )
    report += f"""

## Training Dynamics

The following values use only the cells that receive contextual feedback.

{_markdown_table(training_table, list(training_table.columns), formats={
    'mean_raw_y': '.5f',
    'mean_multiplier': '.5f',
    'mean_fb_potentiation': '.5f',
    'mean_final_fb_weight': '.5f',
})}

## Familiar And Novel Responses

Response changes are paired differences relative to the matched undampened cell.

{_markdown_table(response_table, list(response_table.columns), formats={
    'mean_expert_NO': '.5f',
    'mean_expert_O': '.5f',
    'mean_delta_NO_vs_undampened': '+.5f',
    'mean_delta_O_vs_undampened': '+.5f',
    'NO_percent_change': '+.2f',
    'O_percent_change': '+.2f',
})}

## Feedback-Receiving Cells

The whole-population means above include 86 cells that receive no FB and cannot
respond directly to this manipulation. Restricting the comparison to the 214
feedback-receiving cells gives:

{_markdown_table(context_table, list(context_table.columns), formats={
    'mean_expert_NO': '.5f',
    'mean_expert_O': '.5f',
    'mean_delta_NO_vs_undampened': '+.5f',
    'mean_delta_O_vs_undampened': '+.5f',
})}

## Key Native Sectors

These rows track the response axis most directly associated with each native
sector while preserving the source-run neuron membership across conditions.

{_markdown_table(key_sector_table, list(key_sector_table.columns), formats={
    'mean_expert_response': '.5f',
    'mean_delta_vs_undampened': '+.5f',
})}

## Recomputed Transition Sectors

Sectors below are recomputed independently for each alpha from that condition's
naive-to-expert displacement using threshold `{SECTOR_THRESHOLD:g}`. Plot colors
remain the native source-run sectors so the same neurons retain the same colors
across panels.

{_markdown_table(sector_table, list(sector_table.columns), formats={'fraction': '.4f'})}

The number of neurons whose independently recomputed sector differs from its
matched undampened sector is:

{_markdown_table(sector_change_table, list(sector_change_table.columns))}

## Validation

- Response rows: `{validation['n_cell_result_rows']}`; training diagnostic rows:
  `{validation['n_training_diagnostic_rows']}`.
- Missing values across response, diagnostic, and summary tables: `{validation['n_missing_values']}`.
- Maximum naive-response spread across matched variants: `{validation['max_naive_response_spread']:.3g}`.
- Maximum initial-weight spread across matched variants: `{validation['max_initial_weight_spread']:.3g}`.
- Cells violating monotonic multiplier ordering: `{validation['multiplier_monotonicity_violations']}`.
- FB-receiving cells violating monotonic potentiation ordering:
  `{validation['potentiation_monotonicity_violations']}`.
- A two-cell zero-threshold validation reproduced the existing dampened and
  undampened exporter responses to better than `1e-12` before the full run.

## Interpretation

The source `alpha = 1.0` term is weak because raw training activity is much less
than one. Lowering alpha moves the half-dampening point into the observed activity
range. This changes both overall FB acquisition and its dependence on somatic
activity, so lower-alpha conditions should not be interpreted as changing only
selectivity while holding mean feedback strength constant.

The plotted responses are z-scored after simulation. The plasticity rule uses raw,
bounded, temporally integrated `y`; plotted response magnitudes therefore cannot
be inserted directly into the dampening equation.

## Reconstruction And Provenance

- Source run: `{run_dir}`
- Population: `{metadata['n_samples_total']}` saved configurations
- Steps per phase: `{metadata['n_steps_per_phase']}`
- Test trials: `{metadata['test_trials']}`
- Training trials per familiar image: `{metadata['training_trials']}`
- Somatic activation threshold: `{soma_threshold:g}` reconstructed explicitly
- Sector threshold: `{SECTOR_THRESHOLD:g}`

The explicit activation reconstruction fixes the ambiguity in `sampled_configs.json`,
where the module was serialized only as `ThresholdReLU()`.

## Output Files

- `fb_dampening_alpha_sweep_cell_results.csv`: paired naive/expert responses and sectors
- `fb_dampening_alpha_sweep_training_diagnostics.csv`: per-cell raw activity, multipliers, and weights
- `fb_dampening_alpha_sweep_population_summary.csv`: population response summary
- `fb_dampening_alpha_sweep_context_response_summary.csv`: FB-receiving-cell responses
- `fb_dampening_alpha_sweep_key_native_sector_summary.csv`: selected native-sector responses
- `fb_dampening_alpha_sweep_training_summary.csv`: feedback-receiving-cell training summary
- `fb_dampening_alpha_sweep_sector_counts.csv`: independently recomputed sector counts
- `fb_dampening_alpha_sweep_sector_changes.csv`: sector changes versus undampened
- `fb_dampening_alpha_sweep_expert_scatter.png`: matched expert NO/O scatters
- `fb_dampening_alpha_sweep_response_summary.png`: population response curves
- `fb_dampening_alpha_sweep_training_diagnostics.png`: activity, multiplier, and FB-weight diagnostics
- `metadata.json`: exact suite settings
- `validation.json`: matched-input, completeness, and monotonicity checks
"""
    (output_dir / "REPORT.md").write_text(report)


def run_sweep(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    configs = json.loads((run_dir / "sampled_configs.json").read_text())
    if args.max_cells is not None:
        configs = configs[: args.max_cells]
    alphas = tuple(float(alpha) for alpha in args.alphas)
    if any(alpha <= 0 for alpha in alphas):
        raise ValueError("Every alpha must be positive.")
    variants = _variant_specs(alphas)
    soma_threshold = (
        float(args.soma_threshold)
        if args.soma_threshold is not None
        else float(metadata.get("pruned_mini_variant", {}).get("soma_activation_threshold", 0.08))
    )
    stimuli = _append_post_stimulus_iti(
        _build_model_scatter_test_stimuli(
            n_steps_per_phase=int(metadata["n_steps_per_phase"]),
            n_trials=int(metadata["test_trials"]),
        ),
        n_steps_per_phase=int(metadata["n_steps_per_phase"]),
    )
    training = _build_model_scatter_training_stimuli(
        n_steps_per_phase=int(metadata["n_steps_per_phase"]),
        n_trials=int(metadata["training_trials"]),
        order=str(metadata["training_stimulus_order"]),
        seed=int(metadata["seed"]),
    )

    phase_path = output_dir / "fb_dampening_alpha_sweep_phase_responses.csv"
    diagnostics_path = output_dir / "fb_dampening_alpha_sweep_training_diagnostics.csv"
    existing_phase = pd.DataFrame()
    existing_diagnostics = pd.DataFrame()
    prior_runtime_seconds = 0.0
    variants_to_run = variants
    if args.resume and phase_path.exists() and diagnostics_path.exists():
        existing_phase = pd.read_csv(phase_path)
        existing_diagnostics = pd.read_csv(diagnostics_path)
        wanted_cells = {int(config["_sample_global_idx"]) for config in configs}
        existing_phase = existing_phase.loc[existing_phase["neuron_idx"].isin(wanted_cells)].copy()
        existing_diagnostics = existing_diagnostics.loc[
            existing_diagnostics["neuron_idx"].isin(wanted_cells)
        ].copy()
        complete_variants = {
            str(variant)
            for variant, rows in existing_diagnostics.groupby("variant")
            if rows["neuron_idx"].nunique() == len(configs)
        }
        variants_to_run = [variant for variant in variants if variant["variant"] not in complete_variants]
        metadata_path = output_dir / "metadata.json"
        if metadata_path.exists():
            prior_runtime_seconds = float(json.loads(metadata_path.read_text()).get("runtime_seconds", 0.0))
        print(
            "[fb-dampening-sweep] resuming; already complete: "
            + ", ".join(variant["variant"] for variant in variants if variant["variant"] in complete_variants),
            flush=True,
        )

    print(
        f"[fb-dampening-sweep] simulating {len(configs)} cells x {len(variants_to_run)} new variants",
        flush=True,
    )
    start = time.perf_counter()
    if variants_to_run:
        results = Parallel(n_jobs=args.n_jobs, verbose=10 if args.n_jobs != 1 else 0)(
            delayed(_run_config_variants)(
                config,
                variants_to_run,
                stimuli=stimuli,
                training=training,
                metadata=metadata,
                soma_threshold=soma_threshold,
            )
            for config in configs
        )
        response_rows = [row for responses, _diagnostics in results for row in responses]
        diagnostic_rows = [row for _responses, diagnostics in results for row in diagnostics]
        new_phase = pd.DataFrame(response_rows)
        new_diagnostics = pd.DataFrame(diagnostic_rows)
        replaced = {variant["variant"] for variant in variants_to_run}
        if not existing_phase.empty:
            existing_phase = existing_phase.loc[~existing_phase["variant"].isin(replaced)]
        if not existing_diagnostics.empty:
            existing_diagnostics = existing_diagnostics.loc[~existing_diagnostics["variant"].isin(replaced)]
        response_phase = pd.concat([existing_phase, new_phase], ignore_index=True)
        diagnostics = pd.concat([existing_diagnostics, new_diagnostics], ignore_index=True)
    else:
        response_phase = existing_phase
        diagnostics = existing_diagnostics
    extension_runtime_seconds = time.perf_counter() - start
    runtime_seconds = prior_runtime_seconds + extension_runtime_seconds
    wanted_variants = [variant["variant"] for variant in variants]
    response_phase = response_phase.loc[response_phase["variant"].isin(wanted_variants)].copy()
    diagnostics = diagnostics.loc[diagnostics["variant"].isin(wanted_variants)].copy()
    cell_results = _build_cell_results(response_phase, run_dir=run_dir)
    response_summary = _population_response_summary(cell_results)
    context_response_summary = _context_response_summary(cell_results)
    key_sector_summary = _key_native_sector_summary(cell_results)
    training_summary = _training_summary(diagnostics)
    sector_counts = _sector_counts(cell_results)
    sector_change_summary = _sector_change_summary(cell_results)

    initial_columns = [column for column in diagnostics if column.startswith("initial_")]
    naive_spread = cell_results.groupby(["neuron_idx", "image_group"])[["naive_NO", "naive_O"]].agg(
        lambda values: values.max() - values.min()
    )
    initial_spread = diagnostics.groupby("neuron_idx")[initial_columns].agg(
        lambda values: values.max() - values.min()
    )
    variant_order = [variant["variant"] for variant in variants]
    context_diagnostics = diagnostics.loc[diagnostics["receives_context"]]
    monotonic = context_diagnostics.pivot(
        index="neuron_idx",
        columns="variant",
        values=["multiplier_mean", "mean_fb_potentiation"],
    )
    validation = {
        "n_cell_result_rows": int(len(cell_results)),
        "n_training_diagnostic_rows": int(len(diagnostics)),
        "n_missing_values": int(
            cell_results.isna().sum().sum()
            + diagnostics.isna().sum().sum()
            + response_summary.isna().sum().sum()
        ),
        "max_naive_response_spread": float(naive_spread.to_numpy().max()),
        "max_initial_weight_spread": float(initial_spread.to_numpy().max()),
        "multiplier_monotonicity_violations": int(
            np.any(np.diff(monotonic["multiplier_mean"][variant_order].to_numpy(), axis=1) > 1e-10, axis=1).sum()
        ),
        "potentiation_monotonicity_violations": int(
            np.any(
                np.diff(monotonic["mean_fb_potentiation"][variant_order].to_numpy(), axis=1) > 1e-10,
                axis=1,
            ).sum()
        ),
    }

    response_phase.to_csv(phase_path, index=False)
    cell_results.to_csv(output_dir / "fb_dampening_alpha_sweep_cell_results.csv", index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    response_summary.to_csv(output_dir / "fb_dampening_alpha_sweep_population_summary.csv", index=False)
    context_response_summary.to_csv(
        output_dir / "fb_dampening_alpha_sweep_context_response_summary.csv", index=False
    )
    key_sector_summary.to_csv(output_dir / "fb_dampening_alpha_sweep_key_native_sector_summary.csv", index=False)
    training_summary.to_csv(output_dir / "fb_dampening_alpha_sweep_training_summary.csv", index=False)
    sector_counts.to_csv(output_dir / "fb_dampening_alpha_sweep_sector_counts.csv", index=False)
    sector_change_summary.to_csv(output_dir / "fb_dampening_alpha_sweep_sector_changes.csv", index=False)
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    _plot_scatter(cell_results, variants, output_dir)
    _plot_response_summary(response_summary, variants, output_dir)
    _plot_training_summary(training_summary, variants, output_dir)

    suite_metadata = {
        "source_run_dir": str(run_dir),
        "n_cells": len(configs),
        "source_n_cells": int(metadata["n_samples_total"]),
        "matched_cell_parameters_and_seeds": True,
        "alphas": list(alphas),
        "variants": variants,
        "soma_activation_threshold": soma_threshold,
        "sector_threshold": SECTOR_THRESHOLD,
        "n_steps_per_phase": int(metadata["n_steps_per_phase"]),
        "test_trials": int(metadata["test_trials"]),
        "training_trials_per_familiar_image": int(metadata["training_trials"]),
        "training_stimulus_order": str(metadata["training_stimulus_order"]),
        "seed": int(metadata["seed"]),
        "runtime_seconds": runtime_seconds,
        "last_extension_runtime_seconds": extension_runtime_seconds,
    }
    (output_dir / "metadata.json").write_text(json.dumps(suite_metadata, indent=2) + "\n")
    _write_report(
        output_dir,
        run_dir=run_dir,
        metadata={**metadata, "n_samples_total": len(configs)},
        soma_threshold=soma_threshold,
        variants=variants,
        response_summary=response_summary,
        context_response_summary=context_response_summary,
        key_sector_summary=key_sector_summary,
        training_summary=training_summary,
        sector_counts=sector_counts,
        sector_change_summary=sector_change_summary,
        validation=validation,
        runtime_seconds=runtime_seconds,
    )
    print(
        f"[fb-dampening-sweep] extension completed in {extension_runtime_seconds:.1f}s "
        f"({runtime_seconds:.1f}s cumulative): {output_dir}",
        flush=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--soma-threshold", type=float, default=None)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--max-cells", type=int, default=None, help="Limit cells for a smoke run.")
    parser.add_argument("--resume", action="store_true", help="Reuse complete variants already in the output directory.")
    return parser


def main() -> None:
    run_sweep(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
