"""Export NO-vs-O response scatters for expert silencing and plasticity ablations.

Rows are image familiarity groups (familiar, novel). Columns are the standard
transition-response panel states: naive, expert, expert feedback off, expert PV
off, expert feedback+PV off, plus training-plasticity perturbations. FB
strengthening is undampened by default in every simulation. Points are colored
by the native rotated sector for that image familiarity group.
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
import torch
from joblib import Parallel, delayed
from matplotlib.lines import Line2D

from thesis.population import transitions_helpers as th
from thesis.population.model_scatter import (
    BASELINE_STD_SCALE,
    PLOT_STYLE,
    RESPONSE_X_LABEL,
    RESPONSE_Y_LABEL,
    _append_post_stimulus_iti,
    _build_model_scatter_test_stimuli,
    _build_model_scatter_training_stimuli,
    _compact_trace_frame,
    _run_sector_average_panel_config,
    _summarize_cell_panel_traces,
)
from thesis.population.experiment_s import run_experimental_phase
from thesis.population.minimal_divisive import CCNeuron
from thesis.population.neuron_utils import ThresholdReLU
from thesis.population.visualize_s import TRANSITION_RESPONSE_COLUMN_SPECS


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = PACKAGE_DIR.parent.parent / "context_contrasting/paper/done-amen"
DEFAULT_OUTPUT_NAME = "expert_silencing_response_scatter"
BASELINE_FB_RULE = "undampened-anti-Hebbian"
ABLATION_DEFINITION_VERSION = "base_fb_lr_5x_10x_sdiag2_offdiag1p2_context2"
FB_RULE_CHOICES = ("dampened-anti-Hebbian", "undampened-anti-Hebbian")
IMAGE_GROUP_ORDER = ("familiar", "novel")
IMAGE_GROUP_LABELS = {"familiar": "Familiar", "novel": "Novel"}
COLUMN_KEYS = (
    "naive",
    "expert",
    "expert_no_fb",
    "expert_no_lat",
    "expert_no_fb_no_lat",
    "expert_no_ff_adapt",
    "expert_fb_specificity_full_diag",
    "expert_fb_signal_general",
    "expert_fb_lr_5x",
    "expert_fb_lr_10x",
)
COLUMN_LABELS = {
    "naive": "Naive",
    "expert": "Expert",
    "expert_no_fb": "Expert FB off",
    "expert_no_lat": "Expert PV off",
    "expert_no_fb_no_lat": "Expert FB+PV off",
    "expert_no_ff_adapt": "FF adapt OFF",
    "expert_fb_specificity_full_diag": "FB S full diag",
    "expert_fb_signal_general": "FB signal gen",
    "expert_fb_lr_5x": "FB LR=5x",
    "expert_fb_lr_10x": "FB LR=10x",
}
TRAINING_PLASTICITY_PERTURBATIONS = (
    {
        "key": "expert_no_ff_adapt",
        "label": COLUMN_LABELS["expert_no_ff_adapt"],
        "training_overrides": {"FF_plasticity": False},
    },
    {
        "key": "expert_fb_specificity_full_diag",
        "label": COLUMN_LABELS["expert_fb_specificity_full_diag"],
        "training_overrides": {"fb_specificity": "diag_2_offdiag_1.2"},
    },
    {
        "key": "expert_fb_signal_general",
        "label": COLUMN_LABELS["expert_fb_signal_general"],
        "training_overrides": {"training_context_signal": "stimulus_2_others_1.2"},
    },
    {
        "key": "expert_fb_lr_5x",
        "label": COLUMN_LABELS["expert_fb_lr_5x"],
        "training_overrides": {"lr_fb": "scale:lr_fb:5"},
    },
    {
        "key": "expert_fb_lr_10x",
        "label": COLUMN_LABELS["expert_fb_lr_10x"],
        "training_overrides": {"lr_fb": "scale:lr_fb:10"},
    },
)
PERTURBATION_BY_KEY = {str(perturbation["key"]): perturbation for perturbation in TRAINING_PLASTICITY_PERTURBATIONS}
SECTOR_DRAW_ORDER = tuple(sector for sector in th._sector_plot_order(small_delta_first=True) if sector != "+NO axis") + (
    "+NO axis",
)
SYMLOG_LINTHRESH = 0.05
PATENT_FIGURE_PARTS = (
    (
        "part_a_acute_silencing",
        "Part A: acute silencing",
        ("naive", "expert", "expert_no_fb", "expert_no_lat", "expert_no_fb_no_lat"),
    ),
    (
        "part_b_plasticity_ablations",
        "Part B: training-plasticity perturbations",
        (
            "expert",
            "expert_no_ff_adapt",
            "expert_fb_specificity_full_diag",
            "expert_fb_signal_general",
            "expert_fb_lr_5x",
            "expert_fb_lr_10x",
        ),
    ),
)
GRAYSCALE_SECTOR_STYLES = {
    "+NO axis": {"marker": "o", "gray": "0.05"},
    "+O axis": {"marker": "^", "gray": "0.25"},
    "-NO axis": {"marker": "s", "gray": "0.50"},
    "-O axis": {"marker": "D", "gray": "0.72"},
    "small ∆": {"marker": "x", "gray": "0.35"},
}


def _strip_saved_only_config_fields(
    config: dict[str, Any],
    *,
    fb_rule: str,
    fb_alpha: float,
    soma_threshold: float | None,
) -> dict[str, Any]:
    clean = copy.deepcopy(config)
    if soma_threshold is None:
        clean.pop("activation", None)
    else:
        clean["activation"] = ThresholdReLU(
            threshold=float(soma_threshold),
            subtractive=False,
            hasMax=True,
            maxValue=1.0,
        )
    clean["FBrule"] = fb_rule
    clean["alpha"] = float(fb_alpha)
    return clean


def _cache_definition_version(fb_rule: str, fb_alpha: float, soma_threshold: float | None) -> str:
    rule_prefix = fb_rule.removesuffix("-anti-Hebbian").replace("-", "_")
    alpha_slug = f"{fb_alpha:g}".replace(".", "p")
    threshold_slug = "legacy" if soma_threshold is None else f"{soma_threshold:g}".replace(".", "p")
    return f"{rule_prefix}_alpha_{alpha_slug}_threshold_{threshold_slug}_{ABLATION_DEFINITION_VERSION}"


def _apply_training_overrides(model: CCNeuron, overrides: dict[str, Any]) -> dict[str, Any]:
    originals: dict[str, Any] = {}
    for attr, value in overrides.items():
        if attr == "training_context_signal":
            continue
        if attr == "fb_specificity":
            originals[attr] = model.fb_specificity.detach().clone()
            if value == "all_ones":
                model.fb_specificity = torch.ones_like(model.fb_specificity)
            elif value == "full_diag_value":
                diag_value = float(torch.diagonal(model.fb_specificity).mean())
                model.fb_specificity = torch.full_like(model.fb_specificity, diag_value)
            elif value == "full_diag_value_2x":
                diag_value = 2.0 * float(torch.diagonal(model.fb_specificity).mean())
                model.fb_specificity = torch.full_like(model.fb_specificity, diag_value)
            elif value == "diag_2_offdiag_1.2":
                off_diag = torch.full_like(model.fb_specificity, 1.2)
                model.fb_specificity = off_diag + torch.eye(
                    model.fb_specificity.shape[0],
                    dtype=model.fb_specificity.dtype,
                    device=model.fb_specificity.device,
                ) * 0.8
            else:
                model.fb_specificity = torch.as_tensor(value, dtype=model.fb_specificity.dtype).reshape_as(model.fb_specificity)
            continue
        originals[attr] = getattr(model, attr)
        if attr == "lr_fb" and isinstance(value, str):
            if value.startswith("mean:"):
                source_attrs = [name.strip() for name in value.removeprefix("mean:").split(",")]
                setattr(model, attr, float(np.mean([float(getattr(model, source_attr)) for source_attr in source_attrs])))
            elif value.startswith("scale:"):
                _prefix, source_attr, factor = value.split(":")
                setattr(model, attr, float(getattr(model, source_attr)) * float(factor))
            else:
                setattr(model, attr, float(getattr(model, value)))
        else:
            setattr(model, attr, value)
    return originals


def _restore_training_overrides(model: CCNeuron, originals: dict[str, Any]) -> None:
    for attr, value in originals.items():
        setattr(model, attr, value)


def _transform_training_context_signal(c: torch.Tensor, mode: str | None) -> torch.Tensor:
    if mode is None:
        return c
    if mode == "stimulus_1_others_0.6":
        stimulated_value = 1.0
        unstimulated_value = 0.6
    elif mode == "stimulus_2_others_1.2":
        stimulated_value = 2.0
        unstimulated_value = 1.2
    else:
        raise ValueError(f"Unsupported training_context_signal override: {mode}")
    active = c.sum(dim=1, keepdim=True) > 0
    generalized = torch.where(c > 0, torch.full_like(c, stimulated_value), torch.full_like(c, unstimulated_value))
    return torch.where(active, generalized, torch.zeros_like(c))


def _run_one_config(
    config: dict[str, Any],
    *,
    metadata: dict[str, Any],
    fb_rule: str,
    fb_alpha: float,
    soma_threshold: float | None,
    perturbation_keys: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    clean_config = _strip_saved_only_config_fields(
        config,
        fb_rule=fb_rule,
        fb_alpha=fb_alpha,
        soma_threshold=soma_threshold,
    )
    trace_frames: list[pd.DataFrame] = []
    if perturbation_keys is None:
        _transition, traces, _stimuli = _run_sector_average_panel_config(
            f"cell_{int(config['_sample_global_idx'])}",
            clean_config,
            n_steps_per_phase=int(metadata["n_steps_per_phase"]),
            test_trials=int(metadata["test_trials"]),
            training_trials=int(metadata["training_trials"]),
            training_stimulus_order=str(metadata["training_stimulus_order"]),
            seed=int(metadata["seed"]),
            zscore_std_floor=float(metadata.get("zscore_std_floor", 0.04)),
        )
        trace_frames.append(traces)
        selected_perturbations = TRAINING_PLASTICITY_PERTURBATIONS
    else:
        selected_perturbations = tuple(PERTURBATION_BY_KEY[key] for key in perturbation_keys)
    prepared_baseline = _prepare_training_perturbation_baseline(clean_config, metadata=metadata)
    for perturbation in selected_perturbations:
        trace_frames.append(
            _run_training_plasticity_perturbation_config(
                clean_config,
                metadata=metadata,
                column_key=str(perturbation["key"]),
                column_label=str(perturbation["label"]),
                training_overrides=dict(perturbation["training_overrides"]),
                prepared_baseline=prepared_baseline,
            )
        )
    traces = pd.concat(trace_frames, ignore_index=True)
    traces["neuron_idx"] = int(config["_sample_global_idx"])
    traces["transition"] = str(config["_canonical_transition"])
    return traces


def _prepare_training_perturbation_baseline(
    config: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    n_steps_per_phase = int(metadata["n_steps_per_phase"])
    test_trials = int(metadata["test_trials"])
    zscore_std_floor = float(metadata.get("zscore_std_floor", 0.04))
    cell_floor = max(zscore_std_floor, BASELINE_STD_SCALE * float(config.get("baseline_drive_sigma", 0.0)))
    model = CCNeuron(**{key: value for key, value in config.items() if not key.startswith("_")})
    stimuli = _append_post_stimulus_iti(
        _build_model_scatter_test_stimuli(n_steps_per_phase=n_steps_per_phase, n_trials=test_trials),
        n_steps_per_phase=n_steps_per_phase,
    )
    training = _build_model_scatter_training_stimuli(
        n_steps_per_phase=n_steps_per_phase,
        n_trials=int(metadata["training_trials"]),
        order=str(metadata["training_stimulus_order"]),
        seed=int(metadata["seed"]),
    )
    frames: list[pd.DataFrame] = []
    for condition, (x_full, c_full) in stimuli.items():
        occluded_x = torch.zeros_like(x_full)
        frames.append(
            _compact_trace_frame(
                run_experimental_phase(model, x_full, c_full, f"full_{condition}_naive", update=False),
                condition=condition,
                image_type="full",
                phase="naive",
                zscore_std_floor=cell_floor,
            )
        )
        frames.append(
            _compact_trace_frame(
                run_experimental_phase(model, occluded_x, c_full, f"occlusion_{condition}_naive", update=False),
                condition=condition,
                image_type="occlusion",
                phase="naive",
                zscore_std_floor=cell_floor,
            )
        )
    return {
        "model": model,
        "stimuli": stimuli,
        "training": training,
        "frames": frames,
        "rng_state": torch.random.get_rng_state().clone(),
        "cell_floor": cell_floor,
    }


def _run_training_plasticity_perturbation_config(
    config: dict[str, Any],
    *,
    metadata: dict[str, Any],
    column_key: str,
    column_label: str,
    training_overrides: dict[str, Any],
    prepared_baseline: dict[str, Any] | None = None,
) -> pd.DataFrame:
    n_steps_per_phase = int(metadata["n_steps_per_phase"])
    test_trials = int(metadata["test_trials"])
    training_trials = int(metadata["training_trials"])
    zscore_std_floor = float(metadata.get("zscore_std_floor", 0.04))
    cell_floor = max(zscore_std_floor, BASELINE_STD_SCALE * float(config.get("baseline_drive_sigma", 0.0)))

    if prepared_baseline is None:
        prepared_baseline = _prepare_training_perturbation_baseline(config, metadata=metadata)
    model = copy.deepcopy(prepared_baseline["model"])
    stimuli = prepared_baseline["stimuli"]
    training = prepared_baseline["training"]
    frames = [frame.copy() for frame in prepared_baseline["frames"]]
    torch.random.set_rng_state(prepared_baseline["rng_state"].clone())

    originals = _apply_training_overrides(model, training_overrides)
    training_c = _transform_training_context_signal(training[1], training_overrides.get("training_context_signal"))
    run_experimental_phase(model, training[0], training_c, "full_familiar_training", update=True)
    _restore_training_overrides(model, originals)

    for condition, (x_full, c_full) in stimuli.items():
        occluded_x = torch.zeros_like(x_full)
        frames.append(
            _compact_trace_frame(
                run_experimental_phase(model, x_full, c_full, f"full_{condition}_{column_key}", update=False),
                condition=condition,
                image_type="full",
                phase="expert",
                zscore_std_floor=cell_floor,
            )
        )
        frames.append(
            _compact_trace_frame(
                run_experimental_phase(model, occluded_x, c_full, f"occlusion_{condition}_{column_key}", update=False),
                condition=condition,
                image_type="occlusion",
                phase="expert",
                zscore_std_floor=cell_floor,
            )
        )

    traces = _summarize_cell_panel_traces(
        frames,
        stimuli,
        n_steps_per_phase=n_steps_per_phase,
        test_trials=test_trials,
        zscore_std_floor=cell_floor,
        cell_id=int(config["_sample_global_idx"]),
    )
    traces = traces.loc[traces["column_key"].eq("expert")].copy()
    traces["column_key"] = column_key
    traces["column_label"] = column_label
    traces["experiment_phase"] = "expert_plasticity_perturbation"
    return traces


def _simulate_or_load_trace_scalars(
    *,
    run_dir: Path,
    output_dir: Path,
    n_jobs: int,
    force: bool,
    fb_rule: str,
    fb_alpha: float,
    soma_threshold: float | None,
    max_cells: int | None,
    save_traces: bool,
) -> pd.DataFrame:
    scalar_path = output_dir / "expert_silencing_response_scatter_values.csv"
    cache_metadata_path = output_dir / "expert_silencing_response_scatter_cache_metadata.json"
    cache_definition_version = _cache_definition_version(fb_rule, fb_alpha, soma_threshold)
    source_n_cells = int(json.loads((run_dir / "metadata.json").read_text())["n_samples_total"])
    expected_n_cells = source_n_cells if max_cells is None else min(int(max_cells), source_n_cells)
    if scalar_path.exists() and not force:
        scalar = pd.read_csv(scalar_path)
        if cache_metadata_path.exists():
            cache_metadata = json.loads(cache_metadata_path.read_text())
        else:
            cache_metadata = {}
        cache_version_matches = (
            cache_metadata.get("ablation_definition_version") == cache_definition_version
            and int(cache_metadata.get("n_cells", -1)) == expected_n_cells
        )
        if not cache_version_matches:
            print("[silencing-scatter] cache definition changed; recomputing all columns", flush=True)
        else:
            present_columns = set(scalar.get("column_key", pd.Series(dtype=str)).astype(str))
            missing_columns = tuple(
                column_key
                for column_key in COLUMN_KEYS
                if column_key not in present_columns and column_key in PERTURBATION_BY_KEY
            )
            recompute_columns = ()
            non_incremental_missing = [
                column_key for column_key in COLUMN_KEYS if column_key not in present_columns and column_key not in PERTURBATION_BY_KEY
            ]
            if not missing_columns and not non_incremental_missing:
                return scalar.loc[scalar["column_key"].isin(COLUMN_KEYS)].copy()
            if missing_columns and not non_incremental_missing:
                print(f"[silencing-scatter] simulating missing/refreshed columns: {', '.join(missing_columns)}", flush=True)
                metadata = json.loads((run_dir / "metadata.json").read_text())
                configs = json.loads((run_dir / "sampled_configs.json").read_text())
                if max_cells is not None:
                    configs = configs[:max_cells]
                trace_frames = Parallel(n_jobs=n_jobs, verbose=10 if n_jobs != 1 else 0)(
                    delayed(_run_one_config)(
                        config,
                        metadata=metadata,
                        fb_rule=fb_rule,
                        fb_alpha=fb_alpha,
                        soma_threshold=soma_threshold,
                        perturbation_keys=missing_columns,
                    )
                    for config in configs
                )
                new_traces = pd.concat(trace_frames, ignore_index=True)
                trace_path = output_dir / "expert_silencing_response_traces.csv"
                if save_traces:
                    if trace_path.exists():
                        new_traces.to_csv(trace_path, mode="a", header=False, index=False)
                    else:
                        new_traces.to_csv(trace_path, index=False)
                new_scalar = _scalarize_trace_responses(new_traces, metadata)
                scalar = pd.concat([scalar.loc[~scalar["column_key"].isin(missing_columns)], new_scalar], ignore_index=True)
                scalar = scalar.loc[scalar["column_key"].isin(COLUMN_KEYS)].copy()
                scalar.to_csv(scalar_path, index=False)
                _write_cache_metadata(
                    cache_metadata_path,
                    fb_rule=fb_rule,
                    fb_alpha=fb_alpha,
                    soma_threshold=soma_threshold,
                    n_cells=len(configs),
                    traces_saved=save_traces,
                )
                return scalar
            print("[silencing-scatter] cached scalar file is missing non-incremental columns; recomputing", flush=True)

    metadata = json.loads((run_dir / "metadata.json").read_text())
    configs = json.loads((run_dir / "sampled_configs.json").read_text())
    if max_cells is not None:
        configs = configs[:max_cells]
    print(f"[silencing-scatter] simulating {len(configs)} saved configs", flush=True)
    trace_frames = Parallel(n_jobs=n_jobs, verbose=10 if n_jobs != 1 else 0)(
        delayed(_run_one_config)(
            config,
            metadata=metadata,
            fb_rule=fb_rule,
            fb_alpha=fb_alpha,
            soma_threshold=soma_threshold,
        )
        for config in configs
    )
    traces = pd.concat(trace_frames, ignore_index=True)
    if save_traces:
        traces.to_csv(output_dir / "expert_silencing_response_traces.csv", index=False)

    scalar = _scalarize_trace_responses(traces, metadata)
    scalar.to_csv(scalar_path, index=False)
    _write_cache_metadata(
        cache_metadata_path,
        fb_rule=fb_rule,
        fb_alpha=fb_alpha,
        soma_threshold=soma_threshold,
        n_cells=len(configs),
        traces_saved=save_traces,
    )
    return scalar


def _write_cache_metadata(
    path: Path,
    *,
    fb_rule: str,
    fb_alpha: float,
    soma_threshold: float | None,
    n_cells: int,
    traces_saved: bool,
) -> None:
    metadata = {
        "ablation_definition_version": _cache_definition_version(fb_rule, fb_alpha, soma_threshold),
        "fb_rule": fb_rule,
        "fb_alpha": fb_alpha,
        "soma_activation_threshold": soma_threshold,
        "n_cells": n_cells,
        "traces_saved": traces_saved,
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def _scalarize_trace_responses(traces: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
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


def _single_response_axis_limits(values: pd.Series | np.ndarray, *, pad_fraction: float = 0.08) -> tuple[float, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    pad = pad_fraction * max(hi - lo, 1.0)
    return lo - pad, hi + pad


def _response_xy_axis_limits(frame: pd.DataFrame, *, pad_fraction: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        _single_response_axis_limits(frame["NO_response"], pad_fraction=pad_fraction),
        _single_response_axis_limits(frame["O_response"], pad_fraction=pad_fraction),
    )


def _draw_identity_line(ax: plt.Axes, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> None:
    lo = min(x_limits[0], y_limits[0])
    hi = max(x_limits[1], y_limits[1])
    ax.plot([lo, hi], [lo, hi], "--", color="0.75", linewidth=1.0, zorder=0)


def _draw_response_scatter_axis(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    axis_scale: str,
    grayscale: bool = False,
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
        if grayscale:
            sector_style = GRAYSCALE_SECTOR_STYLES[sector]
            rgb = np.array(th.mcolors.to_rgb(sector_style["gray"])).reshape(1, 3)
            marker = str(sector_style["marker"])
        else:
            rgb = np.array(th.mcolors.to_rgb(th.ROTATED_SECTOR_PALETTE[sector])).reshape(1, 3)
            marker = "o"
        rgba = np.repeat(rgb, len(sector_rows), axis=0)
        rgba = np.concatenate([rgba, alphas[sector_mask].reshape(-1, 1)], axis=1)
        scatter_kwargs: dict[str, Any] = {
            "s": PLOT_STYLE["point_size"] * (1.25 if grayscale else 1.0),
            "c": rgba,
            "marker": marker,
            "zorder": th._sector_scatter_zorder(sector),
        }
        if grayscale and marker != "x":
            scatter_kwargs.update(edgecolors="black", linewidths=0.25)
        elif grayscale:
            scatter_kwargs.update(linewidths=0.65)
        else:
            scatter_kwargs.update(edgecolors="none", linewidths=0.0)
        ax.scatter(sector_rows["NO_response"], sector_rows["O_response"], **scatter_kwargs)
    _draw_identity_line(ax, x_limits, y_limits)
    ax.axhline(0.0, color="0.85", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.85", linewidth=0.8, zorder=0)
    if axis_scale == "symlog":
        ax.set_xscale("symlog", linthresh=SYMLOG_LINTHRESH, linscale=0.5)
        ax.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH, linscale=0.5)
    elif axis_scale != "linear":
        raise ValueError(f"Unsupported axis_scale: {axis_scale}")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("auto")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8, length=2)


def _plot_response_scatter_grid(frame: pd.DataFrame, output_dir: Path, *, shared_axes: bool, axis_scale: str = "linear") -> None:
    if shared_axes:
        row_limits = {
            image_group: _response_xy_axis_limits(frame.loc[frame["image_group"].eq(image_group)])
            for image_group in IMAGE_GROUP_ORDER
        }
        title_suffix = "row-shared axes"
    else:
        row_limits = {}
        title_suffix = "individual axes"
    if axis_scale == "linear":
        output_stem = "expert_silencing_response_scatter_shared_axes" if shared_axes else "expert_silencing_response_scatter"
        scale_suffix = ""
    elif axis_scale == "symlog":
        output_stem = (
            "expert_silencing_response_scatter_symlog_shared_axes"
            if shared_axes
            else "expert_silencing_response_scatter_symlog"
        )
        scale_suffix = f", symlog linthresh={SYMLOG_LINTHRESH:g}"
    else:
        raise ValueError(f"Unsupported axis_scale: {axis_scale}")

    fig, axes = plt.subplots(
        len(IMAGE_GROUP_ORDER),
        len(COLUMN_KEYS),
        figsize=(max(14.0, 2.75 * len(COLUMN_KEYS)), 6.2),
        sharex="row" if shared_axes else False,
        sharey="row" if shared_axes else False,
        constrained_layout=False,
    )

    for row_idx, image_group in enumerate(IMAGE_GROUP_ORDER):
        for col_idx, column_key in enumerate(COLUMN_KEYS):
            ax = axes[row_idx, col_idx]
            rows = frame.loc[frame["image_group"].eq(image_group) & frame["column_key"].eq(column_key)]
            x_limits, y_limits = row_limits.get(image_group, _response_xy_axis_limits(rows))
            _draw_response_scatter_axis(ax, rows, x_limits=x_limits, y_limits=y_limits, axis_scale=axis_scale)
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
    fig.suptitle(
        f"Modeled population NO/O response scatter under expert silencing and plasticity ablations ({title_suffix}{scale_suffix})",
        y=1.03,
        fontsize=13,
    )
    fig.supxlabel(RESPONSE_X_LABEL, fontsize=11, y=0.02)
    fig.supylabel(RESPONSE_Y_LABEL, fontsize=11, x=0.015)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.11, top=0.82, wspace=0.18, hspace=0.2)
    for fmt in ("png", "svg"):
        fig.savefig(output_dir / f"{output_stem}.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _patent_sector_handles(*, grayscale: bool) -> list[Line2D]:
    handles: list[Line2D] = []
    for sector in th.ROTATED_SECTOR_ORDER:
        if grayscale:
            style = GRAYSCALE_SECTOR_STYLES[sector]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=str(style["marker"]),
                    linestyle="",
                    markerfacecolor=str(style["gray"]),
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                    color=str(style["gray"]),
                    markersize=6,
                    label=sector,
                )
            )
        else:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color=th.ROTATED_SECTOR_PALETTE[sector],
                    markersize=6,
                    label=sector,
                )
            )
    return handles


def _plot_patent_split_response_scatter_grids(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    shared_axes: bool,
    grayscale: bool,
) -> None:
    style_label = "grayscale" if grayscale else "color"
    axes_label = "row-shared axes" if shared_axes else "individual axes"
    for part_stem, part_title, column_keys in PATENT_FIGURE_PARTS:
        part_frame = frame.loc[frame["column_key"].isin(column_keys)]
        row_limits = {
            image_group: _response_xy_axis_limits(
                part_frame.loc[part_frame["image_group"].eq(image_group)]
            )
            for image_group in IMAGE_GROUP_ORDER
        }
        fig, axes = plt.subplots(
            len(IMAGE_GROUP_ORDER),
            len(column_keys),
            figsize=(11.69, 8.27),
            sharex="row" if shared_axes else False,
            sharey="row" if shared_axes else False,
            constrained_layout=False,
        )
        for row_idx, image_group in enumerate(IMAGE_GROUP_ORDER):
            for col_idx, column_key in enumerate(column_keys):
                ax = axes[row_idx, col_idx]
                rows = frame.loc[
                    frame["image_group"].eq(image_group) & frame["column_key"].eq(column_key)
                ]
                x_limits, y_limits = (
                    row_limits[image_group] if shared_axes else _response_xy_axis_limits(rows)
                )
                _draw_response_scatter_axis(
                    ax,
                    rows,
                    x_limits=x_limits,
                    y_limits=y_limits,
                    axis_scale="linear",
                    grayscale=grayscale,
                )
                if row_idx == 0:
                    ax.set_title(COLUMN_LABELS[column_key], fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(IMAGE_GROUP_LABELS[image_group], fontsize=10)
                else:
                    ax.set_ylabel("")
                if row_idx == len(IMAGE_GROUP_ORDER) - 1:
                    ax.set_xlabel("NO", fontsize=10)
                else:
                    ax.set_xlabel("")

        fig.legend(
            handles=_patent_sector_handles(grayscale=grayscale),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=5,
            frameon=False,
            fontsize=9,
        )
        fig.suptitle(
            f"Modeled population NO/O responses, {part_title} ({style_label}, {axes_label})",
            y=0.965,
            fontsize=14,
        )
        fig.supxlabel(RESPONSE_X_LABEL, fontsize=11, y=0.025)
        fig.supylabel(RESPONSE_Y_LABEL, fontsize=11, x=0.02)
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.11, top=0.78, wspace=0.28, hspace=0.25)
        shared_suffix = "_shared_axes" if shared_axes else ""
        output_stem = f"expert_silencing_response_scatter_{style_label}_{part_stem}{shared_suffix}"
        for fmt in ("png", "svg"):
            fig.savefig(output_dir / f"{output_stem}.{fmt}", dpi=300)
        plt.close(fig)


def _write_patent_split_readme(output_dir: Path) -> None:
    (output_dir / "PATENT_GRAYSCALE_SPLITS.md").write_text(
        """# Patent grayscale and A4 split exports

The original color figures are unchanged. The additional `grayscale` figures
encode the native transition sectors redundantly by marker and gray level:

| Sector | Marker | Gray level |
| --- | --- | --- |
| +NO axis | circle | black |
| +O axis | upward triangle | dark gray |
| -NO axis | square | medium gray |
| -O axis | diamond | light gray |
| small delta | x | medium-dark gray |

Part A contains Naive, Expert, Expert FB off, Expert PV off, and Expert FB+PV
off. Part B repeats Expert as a visual reference and contains FF adaptation off,
FB specificity full diagonal, generalized FB signal, FB LR=5x, and FB LR=10x.

Every split is A4-landscape proportioned and exported as PNG and vector SVG.
`shared_axes` files retain common NO and O limits within each familiarity row of
that figure part. Files without that suffix crop each panel to its own observed
data. Sector membership remains the native unablated naive-to-expert label for
the corresponding familiar or novel row; ablations move points but do not
reassign their symbols.
"""
    )


def _save_o_response_delta_summary(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    expert = frame.loc[
        frame["column_key"].eq("expert"),
        ["neuron_idx", "image_group", "O_response", "NO_response"],
    ].rename(columns={"O_response": "expert_O_response", "NO_response": "expert_NO_response"})
    deltas = frame.merge(expert, on=["neuron_idx", "image_group"], how="left", validate="many_to_one")
    deltas["delta_O_vs_expert"] = deltas["O_response"] - deltas["expert_O_response"]
    deltas["delta_NO_vs_expert"] = deltas["NO_response"] - deltas["expert_NO_response"]

    summaries: list[pd.DataFrame] = []
    all_summary = (
        deltas.groupby(["image_group", "column_key", "column_label"], as_index=False)
        .agg(
            n_cells=("neuron_idx", "nunique"),
            mean_O_response=("O_response", "mean"),
            mean_delta_O_vs_expert=("delta_O_vs_expert", "mean"),
            median_delta_O_vs_expert=("delta_O_vs_expert", "median"),
            mean_NO_response=("NO_response", "mean"),
            mean_delta_NO_vs_expert=("delta_NO_vs_expert", "mean"),
        )
        .assign(sector_group="all")
    )
    summaries.append(all_summary)
    sector_summary = (
        deltas.groupby(["image_group", "RotatedSector", "column_key", "column_label"], as_index=False)
        .agg(
            n_cells=("neuron_idx", "nunique"),
            mean_O_response=("O_response", "mean"),
            mean_delta_O_vs_expert=("delta_O_vs_expert", "mean"),
            median_delta_O_vs_expert=("delta_O_vs_expert", "median"),
            mean_NO_response=("NO_response", "mean"),
            mean_delta_NO_vs_expert=("delta_NO_vs_expert", "mean"),
        )
        .rename(columns={"RotatedSector": "sector_group"})
    )
    summaries.append(sector_summary)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(output_dir / "expert_silencing_o_response_delta_summary.csv", index=False)
    return summary


def export_expert_silencing_response_scatter(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or PACKAGE_DIR.parent / "results" / DEFAULT_OUTPUT_NAME).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.fb_alpha <= 0:
        raise ValueError("--fb-alpha must be positive.")

    scalar = _simulate_or_load_trace_scalars(
        run_dir=run_dir,
        output_dir=output_dir,
        n_jobs=args.n_jobs,
        force=args.force,
        fb_rule=args.fb_rule,
        fb_alpha=args.fb_alpha,
        soma_threshold=args.soma_threshold,
        max_cells=args.max_cells,
        save_traces=not args.no_save_traces,
    )
    labeled = _attach_native_sector_labels(scalar, run_dir)
    labeled.to_csv(output_dir / "expert_silencing_response_scatter_values_labeled.csv", index=False)
    _save_o_response_delta_summary(labeled, output_dir)
    _plot_response_scatter_grid(labeled, output_dir, shared_axes=False)
    _plot_response_scatter_grid(labeled, output_dir, shared_axes=True)
    _plot_response_scatter_grid(labeled, output_dir, shared_axes=False, axis_scale="symlog")
    _plot_response_scatter_grid(labeled, output_dir, shared_axes=True, axis_scale="symlog")
    if args.patent_splits:
        for grayscale in (False, True):
            for shared_axes in (False, True):
                _plot_patent_split_response_scatter_grids(
                    labeled,
                    output_dir,
                    shared_axes=shared_axes,
                    grayscale=grayscale,
                )
        _write_patent_split_readme(output_dir)
    print(f"[silencing-scatter] wrote outputs to {output_dir}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Rerun simulations even if cached scalar values exist.")
    parser.add_argument("--fb-alpha", type=float, default=1.0, help="Alpha in alpha / (y + alpha).")
    parser.add_argument(
        "--soma-threshold",
        type=float,
        default=None,
        help="Explicit ThresholdReLU threshold; omit to preserve the legacy exporter behavior.",
    )
    parser.add_argument("--max-cells", type=int, default=None, help="Limit saved cells for a smoke run.")
    parser.add_argument(
        "--no-save-traces",
        action="store_true",
        help="Save scalar responses and figures without the large per-timepoint trace table.",
    )
    parser.add_argument(
        "--patent-splits",
        action="store_true",
        help="Also export A4-landscape color and grayscale figures split by acute versus training ablations.",
    )
    parser.add_argument(
        "--fb-rule",
        choices=FB_RULE_CHOICES,
        default=BASELINE_FB_RULE,
        help="FB plasticity rule to apply during baseline expert training and every retrained plasticity ablation.",
    )
    return parser


def main() -> None:
    export_expert_silencing_response_scatter(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
