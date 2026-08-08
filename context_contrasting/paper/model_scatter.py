"""Generate the model rotated-sector scatterplots that approximate the real
chronic-imaging transitions (Seignette et al.) with the minimal two-compartment
PyC + PV circuit (``minimal2``).

Pipeline
--------
1. Draw a population of model cells from the transition templates registered by
   ``transition_templates``. Each cell's initial synaptic weights and sampled
   scalar parameters are jittered by the registered perturbation hook.
2. Run every cell through the experimental protocol (``_run_sample``): a naive
   probe of all stimuli, a block of familiar-image training (the only phase with
   plasticity on), then an expert probe.
3. Read out, for each stimulus, the full ("NO", non-occluded) and occluded ("O")
   responses, z-scored to the cell's naive-probe baseline, and take the
   naive->expert shift (dNO, dO).
4. Classify each cell into a rotated sector (+NO / +O / -NO / -O / small) from the
   angle of its shift, and plot/aggregate exactly like the real-data notebook
   (``data_analysis/transitions_helpers``) at the same threshold (0.3).

The two output panels are the model analogues of the real **familiar** (post,
-NO dominant) and **novel** (post, +NO dominant) populations.
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

from . import example_selection as exs
from . import transitions_helpers as th
from .experiment_s import (
    NO_RESPONSE_ABLATION_SPECS,
    PRIMARY_EXPERIMENT_SERIES,
    STIMULUS_SPECS,
    _temporary_model_overrides,
    _run_test_phase_variants,
    run_experimental_phase,
)
from .minimal_divisive import CCNeuron
from .visualize_s import (
    TRANSITION_RESPONSE_COLUMN_SPECS,
    _build_trace_series_lookup,
    _collect_naive_row_baseline_stats,
    _infer_row_zscore_std_floor,
    _summarize_windowed_repeated_trace,
    format_transition_label,
    save_grouped_transition_panels,
    visualize_transition_response_matrix,
    wide_to_long,
)


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs"
SECTOR_TRACE_SCALE_BAR_UNITS = 1.0
SECTOR_TRACE_MIN_ROW_Y_SPAN = 1.35

STAGES = {"naive": "Naive", "expert": "Expert"}
TRACE_TYPES = {"full": "Full", "occlusion": "Occl"}
IMAGE_INFO = {
    "familiar_1": ("familiar", 1, 1),
    "familiar_2": ("familiar", 2, 2),
    "novel": ("novel", 3, 1),
}


def _design_model_scatter_phase(
    *,
    input_mean: torch.Tensor | list[float],
    context_mean: torch.Tensor | list[float],
    n_steps: int,
    n_trials: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic phase: exact-zero ITI and exact stimulus/context vectors."""
    pre_steps = 3 * n_steps // 4
    stim_steps = n_steps - pre_steps
    x_stim = torch.as_tensor(input_mean, dtype=torch.float32).reshape(1, -1).repeat(stim_steps, 1)
    c_stim = torch.as_tensor(context_mean, dtype=torch.float32).reshape(1, -1).repeat(stim_steps, 1)
    x = torch.cat((torch.zeros(pre_steps, x_stim.shape[1], dtype=x_stim.dtype), x_stim), dim=0)
    c = torch.cat((torch.zeros(pre_steps, c_stim.shape[1], dtype=c_stim.dtype), c_stim), dim=0)

    if n_trials is not None:
        x = x.repeat((n_trials, 1))
        c = c.repeat((n_trials, 1))
    return x, c


def _training_trial_order(
    familiar_names: list[str],
    *,
    n_trials: int,
    order: str,
    seed: int,
) -> list[str]:
    trial_order = [name for _ in range(n_trials) for name in familiar_names]
    if order == "fixed":
        return trial_order
    if order != "randomized":
        raise ValueError("training_stimulus_order must be 'fixed' or 'randomized'.")
    rng = np.random.default_rng(seed)
    return rng.permutation(trial_order).tolist()


def _template_number_map(transition_order: list[str]) -> dict[int, str]:
    return {idx: name for idx, name in enumerate(transition_order, start=1)}


def _build_model_scatter_test_stimuli(
    *,
    n_steps_per_phase: int,
    n_trials: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {
        name: _design_model_scatter_phase(
            input_mean=input_mean,
            context_mean=context_mean,
            n_steps=n_steps_per_phase,
            n_trials=n_trials,
        )
        for name, (input_mean, context_mean) in STIMULUS_SPECS.items()
    }


def _build_model_scatter_training_stimuli(
    *,
    n_steps_per_phase: int,
    n_trials: int,
    order: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    familiar_phases = {
        name: _design_model_scatter_phase(
            input_mean=input_mean,
            context_mean=context_mean,
            n_steps=n_steps_per_phase,
            n_trials=None,
        )
        for name, (input_mean, context_mean) in STIMULUS_SPECS.items()
        if name.startswith("familiar")
    }
    phases = [familiar_phases[name] for name in _training_trial_order(list(familiar_phases), n_trials=n_trials, order=order, seed=seed)]
    x = torch.cat([phase[0] for phase in phases], dim=0)
    c = torch.cat([phase[1] for phase in phases], dim=0)
    return x, c


INIT_KEYS = ("w_ff_init", "w_fb_init", "w_lat_init", "w_pv_lat_init", "W_pv_init")
INIT_ALIASES = {
    "ff": "w_ff_init",
    "fb": "w_fb_init",
    "lat": "w_lat_init",
    "pvlat": "w_pv_lat_init",
    "pv": "W_pv_init",
}
minimal_configs3: dict[str, dict[str, Any]] = {}
TRANSITIONS: dict[str, dict[str, Any]] = {}
FIXED_SCALARS = ("lr_ff", "lr_fb", "lr_lat", "lr_pv", "pyc_decay", "pv_decay")
SHARED_LEARNING_RATES: dict[str, float] = {}
SCALAR_NOISE: dict[str, tuple] = {}
GLOBAL_SCALAR_CLIP: dict[str, tuple[Any, Any]] = {}
BASELINE_STD_SCALE = 0.27

PLOT_STYLE = th.DEFAULT_PLOT_STYLE | {
    "pre_point_alpha": 1.0,
    "target_point_alpha": 1.0,
    "shift_point_alpha": 1.0,
    "individual_vector_width": 0.005,
    "mean_arrow_width": 3.1,
    "mean_arrow_mutation_scale": 16.5,
}
HIGHLIGHT_EXAMPLE_FIGSIZE_INCHES = (7.0 / 2.54, 7.0 / 2.54)
RESPONSE_X_LABEL = "Non-occluded response z-scored $\\Delta$F/F"
RESPONSE_Y_LABEL = "Occluded response z-scored $\\Delta$F/F"


def weight_init(
    center: list[float],
    rel_noise: float | None = None,
    noise_floor: float | None = None,
    lo: Any = 0.0,
    hi: Any = 1.0,
) -> dict[str, Any]:
    """Initial-weight spec for one synaptic-weight vector.

    Center-only specs are fixed template weights. Specs with both ``rel_noise``
    and ``noise_floor`` draw per-cell Gaussian samples centered on ``center``,
    with per-element sd ``max(|center| * rel_noise, noise_floor)``, clipped to
    ``[lo, hi]`` (``lo``/``hi`` may be scalars or per-element lists).
    """
    return {"center": center, "rel_noise": rel_noise, "noise_floor": noise_floor, "lo": lo, "hi": hi}


def transition(
    sampling_weight: float,
    *,
    fixed: dict | None = None,
    clip: dict | None = None,
    **weight_inits: dict[str, Any],
) -> dict:
    """Spec for one transition template (a cell type in the population).

    ``sampling_weight``  -- this type's share of the sampled population.
    ``fixed``            -- scalar model parameters pinned to a constant value.
    ``clip``             -- (lo, hi) bounds applied to scalar parameters after
                            their random perturbation.
    ``weight_inits``     -- per-synapse init specs, keyed by the short aliases in
                            ``INIT_ALIASES`` (ff, fb, lat, pvlat, pv).
    """
    return {
        "weight": sampling_weight,
        "init": {INIT_ALIASES[key]: spec for key, spec in weight_inits.items()},
        "fix": fixed or {},
        "clip": clip or {},
    }


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _clip(value: float, lo: float | None, hi: float | None) -> float:
    if lo is not None:
        value = max(value, lo)
    if hi is not None:
        value = min(value, hi)
    return float(value)


def _bound(bound: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    if bound is None:
        return None
    arr = np.asarray(bound if isinstance(bound, (list, tuple, np.ndarray)) else [bound], dtype=float)
    return np.full(shape, float(arr.item())) if arr.size == 1 else np.broadcast_to(arr, shape).astype(float)


def _clip_array(values: np.ndarray, lo: Any, hi: Any) -> np.ndarray:
    lo_arr, hi_arr = _bound(lo, values.shape), _bound(hi, values.shape)
    if lo_arr is not None:
        values = np.maximum(values, lo_arr)
    if hi_arr is not None:
        values = np.minimum(values, hi_arr)
    return values


def _set_init(config: dict[str, Any], key: str, values: np.ndarray, sigma: Any = 0.0) -> None:
    config[key] = {"mu": [float(value) for value in values.reshape(-1)], "sigma": sigma}


def _init_center(init_spec: dict[str, Any]) -> np.ndarray:
    return np.asarray(init_spec["center"], dtype=float)


def _init_bounds(init_spec: dict[str, Any]) -> tuple[Any, Any]:
    return init_spec.get("lo", 0.0), init_spec.get("hi", 1.0)


def _center_init_values(init_spec: dict[str, Any]) -> np.ndarray:
    center = _init_center(init_spec)
    if init_spec["rel_noise"] is None and init_spec["noise_floor"] is None:
        return center
    return _clip_array(center, *_init_bounds(init_spec))


def _draw_init(init_spec: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """Sample one initial-weight vector from a `weight_init` spec (Gaussian about
    `center`, sd `max(|center|*rel_noise, noise_floor)`, clipped to [lo, hi])."""
    if init_spec["rel_noise"] is None or init_spec["noise_floor"] is None:
        raise ValueError("Cannot sample from a center-only weight_init spec.")
    center = _init_center(init_spec)
    scale = np.maximum(np.abs(center) * init_spec["rel_noise"], init_spec["noise_floor"])
    return _clip_array(center + rng.normal(0.0, scale, size=center.shape), *_init_bounds(init_spec))


def _draw_scalar(value: float, spec: tuple, rng: np.random.Generator, multiplier: float) -> float:
    """Sample one scalar parameter from a `SCALAR_NOISE` spec (see that table)."""
    mode, scale, lo, hi, floor = spec
    if mode == "log" and value > 0.0:
        sampled = value * float(np.exp(rng.normal(0.0, scale * multiplier)))
    else:
        sampled = value + float(rng.normal(0.0, max(abs(value) * scale, floor) * multiplier))
    return _clip(sampled, lo, hi)


def _apply_shared_learning_rates(config: dict[str, Any]) -> None:
    config.update(SHARED_LEARNING_RATES)


def _perturb_config(*args, **kwargs) -> dict[str, Any]:
    raise RuntimeError("model_scatter must be configured by transition_templates before running.")


def _center_config(transition: str) -> dict[str, Any]:
    raise RuntimeError("model_scatter must be configured by transition_templates before rendering center panels.")


def _draw_transition_names(
    transition_order: list[str],
    *,
    n_samples: int,
    transition_sampling: str,
    rng: np.random.Generator,
) -> list[str]:
    """Pick which transition template each of the `n_samples` cells belongs to.

    "data-like" (default) draws each cell at random in proportion to the
    `sampling_weight`s, so the population matches the measured class proportions;
    "equal" gives every template the same count (useful for inspecting a type).
    """
    if transition_sampling == "equal":
        repeats = int(np.ceil(n_samples / len(transition_order)))
        return [name for name in transition_order for _ in range(repeats)][:n_samples]
    if transition_sampling != "data-like":
        raise ValueError("transition_sampling must be 'data-like' or 'equal'.")
    weights = np.asarray([TRANSITIONS[name]["weight"] for name in transition_order], dtype=float)
    return rng.choice(transition_order, size=n_samples, p=weights / weights.sum()).tolist()


def _sample_configs(args: argparse.Namespace, transition_order: list[str]) -> list[dict[str, Any]]:
    """Build the full list of per-cell configs to simulate.

    `--canonical-only` runs the registered center configs once each (a debug mode);
    otherwise draw `n_samples` cells from the weighted mixture and perturb each.
    Per-cell child RNGs (spawned from one seed) keep the population reproducible.
    """
    if args.canonical_only:
        samples = []
        for global_idx, (name, config) in enumerate(minimal_configs3.items(), start=1):
            sample = copy.deepcopy(config)
            _apply_shared_learning_rates(sample)
            sample.update(_canonical_transition=name, _sample_idx=1, _sample_global_idx=global_idx)
            samples.append(sample)
        return samples

    draws = _draw_transition_names(
        transition_order,
        n_samples=args.n_samples,
        transition_sampling=args.transition_sampling,
        rng=np.random.default_rng(args.seed),
    )
    child_rngs = np.random.SeedSequence(args.seed).spawn(len(draws))
    seen = dict.fromkeys(transition_order, 0)
    samples = []
    for global_idx, transition in enumerate(draws, start=1):
        seen[transition] += 1
        samples.append(
            _perturb_config(
                transition,
                minimal_configs3[transition],
                sample_idx=seen[transition],
                global_idx=global_idx,
                seed=args.seed + global_idx,
                rng=np.random.default_rng(child_rngs[global_idx - 1]),
                scalar_noise_multiplier=args.scalar_noise_multiplier,
            )
        )
    return samples


def _response_from_frame(
    frame: pd.DataFrame,
    *,
    n_steps_per_phase: int,
    response_tail_fraction: float,
    baseline: tuple[float, float],
    zscore_std_floor: float,
) -> float:
    """Mean z-scored response over the stimulus window of a probe trace.

    Averages the firing rate `y` over the last `response_tail_fraction` of each
    trial's stimulus window (the stimulus occupies the final quarter of a trial),
    then z-scores by the `(mean, std)` baseline using `max(std, zscore_std_floor)`.
    """
    stim_start = 3 * n_steps_per_phase // 4
    stim_len = n_steps_per_phase - stim_start
    tail_start = stim_start + int(round((1.0 - response_tail_fraction) * stim_len))
    trial_step = frame["step"].to_numpy(dtype=int) % n_steps_per_phase
    mask = trial_step >= tail_start
    values = frame.loc[mask, "y"].to_numpy(dtype=float)
    mean, std = baseline
    scale = max(std, zscore_std_floor) if np.isfinite(std) and std > 1e-12 else max(1.0, zscore_std_floor)
    return float(np.nanmean((values - mean) / scale))


def _baseline(frames: list[pd.DataFrame], n_steps_per_phase: int) -> tuple[float, float]:
    """Spontaneous `(mean, std)` of `y` over the inter-trial windows (the first
    three quarters of every trial, when no stimulus is present)."""
    chunks = []
    stim_start = 3 * n_steps_per_phase // 4
    for frame in frames:
        trial_step = frame["step"].to_numpy(dtype=int) % n_steps_per_phase
        chunks.append(frame.loc[trial_step < stim_start, "y"].to_numpy(dtype=float))
    values = np.concatenate([chunk for chunk in chunks if chunk.size])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    std = float(np.nanstd(values, ddof=1)) if values.size > 1 else 1.0
    return float(np.nanmean(values)), std if std > 1e-12 else 1.0


def _probe_rows(
    model: CCNeuron,
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    phase: str,
    n_steps_per_phase: int,
    response_tail_fraction: float,
    baseline: tuple[float, float] | None,
    zscore_std_floor: float,
) -> tuple[list[dict[str, Any]], tuple[float, float], list[tuple[str, str, pd.DataFrame]]]:
    """Probe every stimulus full (non-occluded, "NO") and occluded ("O") without
    plasticity, returning one response row per (stimulus, trace) plus this phase's
    own spontaneous baseline. Responses are z-scored to `baseline` if given, else
    to this phase's baseline (used so the naive probe normalises itself)."""
    traces = []
    for condition, (x_full, c_full) in stimuli.items():
        for trace, x_phase in (("full", x_full), ("occlusion", torch.zeros_like(x_full))):
            frame = run_experimental_phase(model, x_phase, c_full, f"{trace}_{condition}_{phase}", update=False)
            traces.append((condition, trace, frame))

    local_baseline = _baseline([frame for _, _, frame in traces], n_steps_per_phase)
    ref_baseline = baseline or local_baseline
    rows = [
        dict(
            condition=condition,
            phase=phase,
            stage=STAGES[phase],
            trace=trace,
            image_type=TRACE_TYPES[trace],
            response=_response_from_frame(
                frame,
                n_steps_per_phase=n_steps_per_phase,
                response_tail_fraction=response_tail_fraction,
                baseline=ref_baseline,
                zscore_std_floor=zscore_std_floor,
            ),
        )
        for condition, trace, frame in traces
    ]
    return rows, local_baseline, traces


def _run_sample(
    config: dict[str, Any],
    *,
    n_steps_per_phase: int,
    response_tail_fraction: float,
    test_stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    training_stimuli: tuple[torch.Tensor, torch.Tensor],
    zscore_std_floor: float,
) -> pd.DataFrame:
    """Run one cell through naive probe -> familiar training -> expert probe and
    return its response rows. Both probes are z-scored to the *naive* baseline, so
    the naive->expert shift reflects only the change in evoked response (the
    baseline cancels) rather than any drift in spontaneous activity."""
    model = CCNeuron(**{key: value for key, value in config.items() if not key.startswith("_")})
    # Per-cell z-score floor scaled by the cell's spontaneous (baseline) drive.
    cell_floor = max(zscore_std_floor, BASELINE_STD_SCALE * float(config.get("baseline_drive_sigma", 0.0)))
    rows, naive_baseline, _ = _probe_rows(
        model,
        test_stimuli,
        phase="naive",
        n_steps_per_phase=n_steps_per_phase,
        response_tail_fraction=response_tail_fraction,
        baseline=None,
        zscore_std_floor=cell_floor,
    )
    run_experimental_phase(model, training_stimuli[0], training_stimuli[1], "full_familiar_training", update=True)
    expert_rows, _, _ = _probe_rows(
        model,
        test_stimuli,
        phase="expert",
        n_steps_per_phase=n_steps_per_phase,
        response_tail_fraction=response_tail_fraction,
        baseline=naive_baseline,
        zscore_std_floor=cell_floor,
    )
    rows.extend(expert_rows)
    return pd.DataFrame(rows).assign(
        transition=config["_canonical_transition"],
        sample_idx=config["_sample_idx"],
        sample_global_idx=config["_sample_global_idx"],
        seed=config["seed"],
        experiment_series=PRIMARY_EXPERIMENT_SERIES,
    )


def _run_sample_with_sector_panel_trace(
    config: dict[str, Any],
    *,
    n_steps_per_phase: int,
    test_trials: int,
    response_tail_fraction: float,
    test_stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    training_stimuli: tuple[torch.Tensor, torch.Tensor],
    n_steps_per_phase_display: int,
    zscore_std_floor: float,
) -> tuple[pd.DataFrame, str, pd.DataFrame, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    model = CCNeuron(**{key: value for key, value in config.items() if not key.startswith("_")})
    display_stimuli = _append_post_stimulus_iti(test_stimuli, n_steps_per_phase=n_steps_per_phase_display)
    cell_floor = max(zscore_std_floor, BASELINE_STD_SCALE * float(config.get("baseline_drive_sigma", 0.0)))
    rows, naive_baseline, _ = _probe_rows(
        model,
        test_stimuli,
        phase="naive",
        n_steps_per_phase=n_steps_per_phase,
        response_tail_fraction=response_tail_fraction,
        baseline=None,
        zscore_std_floor=cell_floor,
    )
    panel_frames: list[pd.DataFrame] = []
    for condition, (x_full, c_full) in display_stimuli.items():
        occluded_x = torch.zeros_like(x_full)
        panel_frames.append(
            _compact_trace_frame(
                run_experimental_phase(model, x_full, c_full, f"full_{condition}_naive", update=False),
                condition=condition,
                image_type="full",
                phase="naive",
                zscore_std_floor=cell_floor,
            )
        )
        panel_frames.append(
            _compact_trace_frame(
                run_experimental_phase(model, occluded_x, c_full, f"occlusion_{condition}_naive", update=False),
                condition=condition,
                image_type="occlusion",
                phase="naive",
                zscore_std_floor=cell_floor,
            )
        )
    run_experimental_phase(model, training_stimuli[0], training_stimuli[1], "full_familiar_training", update=True)
    expert_rows, _, _ = _probe_rows(
        model,
        test_stimuli,
        phase="expert",
        n_steps_per_phase=n_steps_per_phase,
        response_tail_fraction=response_tail_fraction,
        baseline=naive_baseline,
        zscore_std_floor=cell_floor,
    )
    rows.extend(expert_rows)
    key = f"cell_{int(config['_sample_global_idx'])}"
    for condition, (x_full, c_full) in display_stimuli.items():
        occluded_x = torch.zeros_like(x_full)
        panel_frames.append(
            _compact_trace_frame(
                run_experimental_phase(model, x_full, c_full, f"full_{condition}_expert", update=False),
                condition=condition,
                image_type="full",
                phase="expert",
                zscore_std_floor=cell_floor,
            )
        )
        panel_frames.append(
            _compact_trace_frame(
                run_experimental_phase(model, occluded_x, c_full, f"occlusion_{condition}_expert", update=False),
                condition=condition,
                image_type="occlusion",
                phase="expert",
                zscore_std_floor=cell_floor,
            )
        )
    panel_frames.extend(
        _run_expert_ablation_compact_traces(
            model,
            display_stimuli,
            zscore_std_floor=cell_floor,
        )
    )
    response_df = pd.DataFrame(rows).assign(
        transition=config["_canonical_transition"],
        sample_idx=config["_sample_idx"],
        sample_global_idx=config["_sample_global_idx"],
        seed=config["seed"],
        experiment_series=PRIMARY_EXPERIMENT_SERIES,
    )
    panel_df = _summarize_cell_panel_traces(
        panel_frames,
        display_stimuli,
        n_steps_per_phase=n_steps_per_phase,
        test_trials=test_trials,
        zscore_std_floor=cell_floor,
        cell_id=int(config["_sample_global_idx"]),
    )
    return response_df, key, panel_df, display_stimuli


def _transition_table(response_df: pd.DataFrame) -> pd.DataFrame:
    """Reshape the raw response rows into the long table schema the real-data
    plotting helpers expect (one row per cell x stimulus x trace x stage, tagged
    with image group/index), mapping each probe condition via `IMAGE_INFO`."""
    rows = []
    for row in response_df.itertuples(index=False):
        image_group, image_idx_original, image_idx_within_group = IMAGE_INFO[row.condition]
        rows.append(
            dict(
                transition=row.transition,
                image_group=image_group,
                image_idx_original=image_idx_original,
                image_idx_within_group=image_idx_within_group,
                neuron_idx=int(row.sample_global_idx),
                image_type=row.image_type,
                stage=row.stage,
                response=float(row.response),
            )
        )
    return pd.DataFrame(rows)


def _wide_table(transition_table: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long table so the full and occluded responses sit side by side as
    `NO` and `O` columns (one row per cell x stimulus x stage)."""
    stage_order = transition_table["stage"].drop_duplicates().tolist()
    wide = transition_table.pivot_table(
        index=["transition", "image_group", "image_idx_original", "image_idx_within_group", "neuron_idx", "stage"],
        columns="image_type",
        values="response",
        aggfunc="mean",
    ).reset_index().rename(columns={"Full": "NO", "Occl": "O"})
    wide["stage"] = pd.Categorical(wide["stage"], categories=stage_order, ordered=True)
    return wide.sort_values(["transition", "image_group", "image_idx_original", "neuron_idx", "stage"]).reset_index(drop=True)


def _flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten one sampled config into a single CSV row: weight vectors become
    `key.mu_i` columns, boolean tuples become `key_i`, scalars pass through (for
    the per-cell `sampled_config_parameters.csv` reproducibility table)."""
    flat = {
        "transition": config["_canonical_transition"],
        "sample_idx": config["_sample_idx"],
        "sample_global_idx": config["_sample_global_idx"],
        "seed": config["seed"],
    }
    for key, value in config.items():
        if key.startswith("_") or key == "seed":
            continue
        if key in INIT_KEYS and isinstance(value, dict):
            for idx, mu in enumerate(np.asarray(value.get("mu", []), dtype=float).reshape(-1)):
                flat[f"{key}.mu_{idx}"] = float(mu)
            if _is_num(value.get("sigma")):
                flat[f"{key}.sigma"] = float(value["sigma"])
        elif _is_num(value) or isinstance(value, str):
            flat[key] = value
        elif isinstance(value, tuple) and all(isinstance(item, bool) for item in value):
            flat.update({f"{key}_{idx}": bool(item) for idx, item in enumerate(value)})
    return flat


def _panel_step_window(n_steps_per_phase: int, test_trials: int) -> tuple[int, int]:
    """Focus transition panels on a real stimulus with pre/post ITI context.

    The displayed window is one trial long: 1/4 pre-stimulus ITI, 1/4 stimulus,
    and 1/2 post-stimulus ITI. In the continuous protocol the stimulus occupies
    the final quarter of each generated trial, so the displayed window spans into
    the appended following ITI.
    """
    trial_idx = max(0, test_trials - 1)
    stim_start = trial_idx * n_steps_per_phase + 3 * n_steps_per_phase // 4
    stim_end = (trial_idx + 1) * n_steps_per_phase
    stimulus_len = stim_end - stim_start
    return stim_start - stimulus_len, stim_end + 2 * stimulus_len


def _append_post_stimulus_iti(
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    n_steps_per_phase: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Append display-only ITI samples so the final requested trial has a post window."""
    post_steps = n_steps_per_phase // 2
    if post_steps <= 0:
        return stimuli

    extended: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for condition, (x_full, c_full) in stimuli.items():
        tail_len = min(post_steps, x_full.shape[0], c_full.shape[0])
        x_tail = x_full[:tail_len].clone()
        c_tail = c_full[:tail_len].clone()
        extended[condition] = (
            torch.cat((x_full, x_tail), dim=0),
            torch.cat((c_full, c_tail), dim=0),
        )
    return extended


def _run_scatter_order_phase_traces(
    model: CCNeuron,
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    phase_label: str,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    ablation_frames: list[pd.DataFrame] = []
    ablation_inputs: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for condition_name, (x_full, c_full) in stimuli.items():
        occluded_x = torch.zeros_like(x_full)
        frames.append(run_experimental_phase(model, x_full, c_full, condition_name=f"full_{condition_name}_{phase_label}", update=False))
        frames.append(run_experimental_phase(model, occluded_x, c_full, condition_name=f"occlusion_{condition_name}_{phase_label}", update=False))
        ablation_inputs.append((condition_name, x_full, occluded_x, c_full))

    for condition_name, x_full, occluded_x, c_full in ablation_inputs:
        no_context_c = torch.zeros_like(c_full)
        for ablation_label, ablation_spec in NO_RESPONSE_ABLATION_SPECS.items():
            ablated_c = no_context_c if ablation_spec.get("zero_context", False) else c_full
            model_overrides = ablation_spec.get("model_overrides", {})
            condition_prefix = ablation_spec.get("condition_prefix", ablation_label)
            with _temporary_model_overrides(model, **model_overrides):
                ablation_frames.append(
                    run_experimental_phase(
                        model,
                        x_full,
                        ablated_c,
                        condition_name=f"{condition_prefix}_{condition_name}_{phase_label}",
                        update=False,
                    )
                )
                ablation_frames.append(
                    run_experimental_phase(
                        model,
                        occluded_x,
                        ablated_c,
                        condition_name=f"occlusion_{condition_name}_{condition_prefix}_{phase_label}",
                        update=False,
                    )
                )
    return frames + ablation_frames


def _run_scatter_order_panel_config(
    transition: str,
    config: dict[str, Any],
    *,
    n_steps_per_phase: int,
    test_trials: int,
    training_trials: int,
    training_stimulus_order: str,
    seed: int,
    zscore_std_floor: float,
) -> tuple[str, pd.DataFrame, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    model = CCNeuron(**{key: value for key, value in config.items() if not key.startswith("_")})
    stimuli = _append_post_stimulus_iti(
        _build_model_scatter_test_stimuli(n_steps_per_phase=n_steps_per_phase, n_trials=test_trials),
        n_steps_per_phase=n_steps_per_phase,
    )
    training = _build_model_scatter_training_stimuli(
        n_steps_per_phase=n_steps_per_phase,
        n_trials=training_trials,
        order=training_stimulus_order,
        seed=seed,
    )

    frames = _run_scatter_order_phase_traces(model, stimuli, phase_label="naive")
    run_experimental_phase(model, training[0], training[1], "full_familiar_training", update=True)
    frames.extend(_run_scatter_order_phase_traces(model, stimuli, phase_label="expert"))

    df = pd.concat([frame.assign(experiment_series=PRIMARY_EXPERIMENT_SERIES) for frame in frames], ignore_index=True)
    df["seed"] = config.get("seed", 42)
    long_df = wide_to_long(df)
    long_df = long_df.loc[long_df["experiment_phase"].isin(["naive", "expert"])].copy()
    long_df["_zscore_std_floor"] = max(zscore_std_floor, BASELINE_STD_SCALE * float(config.get("baseline_drive_sigma", 0.0)))
    return transition, long_df, stimuli


def _run_panel_config(
    transition: str,
    config: dict[str, Any],
    *,
    n_steps_per_phase: int,
    test_trials: int,
    training_trials: int,
    training_stimulus_order: str,
    seed: int,
    zscore_std_floor: float,
) -> tuple[str, pd.DataFrame, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Lightweight naive->train->expert run for ONE config, including the same
    no-feedback and no-LAT variants used by the grouped minimal2 panels."""
    model = CCNeuron(**{key: value for key, value in config.items() if not key.startswith("_")})
    stimuli = _append_post_stimulus_iti(
        _build_model_scatter_test_stimuli(n_steps_per_phase=n_steps_per_phase, n_trials=test_trials),
        n_steps_per_phase=n_steps_per_phase,
    )
    training = _build_model_scatter_training_stimuli(
        n_steps_per_phase=n_steps_per_phase,
        n_trials=training_trials,
        order=training_stimulus_order,
        seed=seed,
    )

    frames = _run_test_phase_variants(model, stimuli, phase_label="naive")
    run_experimental_phase(model, training[0], training[1], "full_familiar_training", update=True)
    frames.extend(_run_test_phase_variants(model, stimuli, phase_label="expert"))

    df = pd.concat([frame.assign(experiment_series=PRIMARY_EXPERIMENT_SERIES) for frame in frames], ignore_index=True)
    df["seed"] = config.get("seed", 42)
    long_df = wide_to_long(df)
    long_df = long_df.loc[long_df["experiment_phase"].isin(["naive", "expert"])].copy()
    long_df["_zscore_std_floor"] = max(zscore_std_floor, BASELINE_STD_SCALE * float(config.get("baseline_drive_sigma", 0.0)))
    return transition, long_df, stimuli


def _run_panel_configs(
    configs_by_transition: dict[str, dict[str, Any]],
    *,
    n_steps_per_phase: int,
    test_trials: int,
    training_trials: int,
    training_stimulus_order: str,
    seed: int,
    n_jobs: int,
    zscore_std_floor: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[torch.Tensor, torch.Tensor]] | None]:
    order = list(configs_by_transition)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_panel_config)(
            t,
            configs_by_transition[t],
            n_steps_per_phase=n_steps_per_phase,
            test_trials=test_trials,
            training_trials=training_trials,
            training_stimulus_order=training_stimulus_order,
            seed=seed,
            zscore_std_floor=zscore_std_floor,
        )
        for t in order
    )
    long_dfs = {t: long_df for t, long_df, _ in results}
    stimuli = results[0][2] if results else None
    return long_dfs, stimuli


def _run_scatter_order_panel_configs(
    configs_by_transition: dict[str, dict[str, Any]],
    *,
    n_steps_per_phase: int,
    test_trials: int,
    training_trials: int,
    training_stimulus_order: str,
    seed: int,
    n_jobs: int,
    zscore_std_floor: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[torch.Tensor, torch.Tensor]] | None]:
    order = list(configs_by_transition)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_scatter_order_panel_config)(
            t,
            configs_by_transition[t],
            n_steps_per_phase=n_steps_per_phase,
            test_trials=test_trials,
            training_trials=training_trials,
            training_stimulus_order=training_stimulus_order,
            seed=seed,
            zscore_std_floor=zscore_std_floor,
        )
        for t in order
    )
    long_dfs = {t: long_df for t, long_df, _ in results}
    stimuli = results[0][2] if results else None
    return long_dfs, stimuli


def _compact_trace_frame(
    frame: pd.DataFrame,
    *,
    condition: str,
    image_type: str,
    phase: str,
    zscore_std_floor: float,
) -> pd.DataFrame:
    compact = frame[["step", "y"]].copy()
    compact["condition"] = condition
    compact["image_type"] = image_type
    compact["experiment_phase"] = phase
    compact["_zscore_std_floor"] = zscore_std_floor
    return compact


def _run_expert_ablation_compact_traces(
    model: CCNeuron,
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    zscore_std_floor: float,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for condition, (x_full, c_full) in stimuli.items():
        occluded_x = torch.zeros_like(x_full)
        no_context_c = torch.zeros_like(c_full)
        for ablation_label, ablation_spec in NO_RESPONSE_ABLATION_SPECS.items():
            ablated_c = no_context_c if ablation_spec.get("zero_context", False) else c_full
            model_overrides = ablation_spec.get("model_overrides", {})
            with _temporary_model_overrides(model, **model_overrides):
                frames.append(
                    _compact_trace_frame(
                        run_experimental_phase(model, x_full, ablated_c, f"{ablation_label}_{condition}_expert", update=False),
                        condition=condition,
                        image_type=ablation_label,
                        phase="expert",
                        zscore_std_floor=zscore_std_floor,
                    )
                )
                frames.append(
                    _compact_trace_frame(
                        run_experimental_phase(
                            model,
                            occluded_x,
                            ablated_c,
                            f"occlusion_{ablation_label}_{condition}_expert",
                            update=False,
                        ),
                        condition=condition,
                        image_type=f"occlusion_{ablation_label}",
                        phase="expert",
                        zscore_std_floor=zscore_std_floor,
                    )
                )
    return frames


def _summarize_cell_panel_traces(
    frames: list[pd.DataFrame],
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    n_steps_per_phase: int,
    test_trials: int,
    zscore_std_floor: float,
    cell_id: int,
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    long_df = pd.concat(frames, ignore_index=True)
    step_window = _panel_step_window(n_steps_per_phase, test_trials)
    series_lookup = _build_trace_series_lookup(long_df)
    response_trace_types = tuple(
        dict.fromkeys(
            trace_type
            for column_spec in TRANSITION_RESPONSE_COLUMN_SPECS
            for trace_type in (column_spec["o_trace"], column_spec["no_trace"])
        )
    )
    baseline_stats = _collect_naive_row_baseline_stats(
        long_df,
        selected_conditions=list(stimuli),
        trace_types=response_trace_types,
        stimuli=stimuli,
        focus_window=step_window,
        series_lookup=series_lookup,
        zscore_std_floor=_infer_row_zscore_std_floor(long_df, zscore_std_floor),
    )
    if baseline_stats is None:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for column_spec in TRANSITION_RESPONSE_COLUMN_SPECS:
        for condition, stim_pair in stimuli.items():
            for response_type, trace_type in (
                ("O", column_spec["o_trace"]),
                ("NO", column_spec["no_trace"]),
            ):
                summary = _summarize_windowed_repeated_trace(
                    long_df,
                    condition=condition,
                    phase=column_spec["phase"],
                    image_type=trace_type,
                    stim_pair=stim_pair,
                    focus_window=step_window,
                    zscore=True,
                    baseline_stats=baseline_stats,
                    series_lookup=series_lookup,
                )
                if summary is None:
                    continue
                stim_start, stim_end = tuple(summary["stim_seconds"])
                rows.extend(
                    {
                        "cell_id": cell_id,
                        "condition": condition,
                        "column_key": column_spec["key"],
                        "column_label": column_spec["label"],
                        "experiment_phase": column_spec["phase"],
                        "response_type": response_type,
                        "image_type": trace_type,
                        "x_seconds": float(time),
                        "y": float(value),
                        "stim_start_seconds": float(stim_start),
                        "stim_end_seconds": float(stim_end),
                        "n_trials": int(summary["n_trials"]),
                    }
                    for time, value in zip(summary["x_seconds"], summary["y_mean"], strict=False)
                )
    return pd.DataFrame(rows)


def _run_sector_average_panel_config(
    transition: str,
    config: dict[str, Any],
    *,
    n_steps_per_phase: int,
    test_trials: int,
    training_trials: int,
    training_stimulus_order: str,
    seed: int,
    zscore_std_floor: float,
) -> tuple[str, pd.DataFrame, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    model = CCNeuron(**{key: value for key, value in config.items() if not key.startswith("_")})
    stimuli = _append_post_stimulus_iti(
        _build_model_scatter_test_stimuli(n_steps_per_phase=n_steps_per_phase, n_trials=test_trials),
        n_steps_per_phase=n_steps_per_phase,
    )
    training = _build_model_scatter_training_stimuli(
        n_steps_per_phase=n_steps_per_phase,
        n_trials=training_trials,
        order=training_stimulus_order,
        seed=seed,
    )
    cell_floor = max(zscore_std_floor, BASELINE_STD_SCALE * float(config.get("baseline_drive_sigma", 0.0)))

    frames: list[pd.DataFrame] = []
    for phase in ("naive", "expert"):
        if phase == "expert":
            run_experimental_phase(model, training[0], training[1], "full_familiar_training", update=True)
        for condition, (x_full, c_full) in stimuli.items():
            occluded_x = torch.zeros_like(x_full)
            frames.append(
                _compact_trace_frame(
                    run_experimental_phase(model, x_full, c_full, f"full_{condition}_{phase}", update=False),
                    condition=condition,
                    image_type="full",
                    phase=phase,
                    zscore_std_floor=cell_floor,
                )
            )
            frames.append(
                _compact_trace_frame(
                    run_experimental_phase(model, occluded_x, c_full, f"occlusion_{condition}_{phase}", update=False),
                    condition=condition,
                    image_type="occlusion",
                    phase=phase,
                    zscore_std_floor=cell_floor,
                )
            )
            if phase != "expert":
                continue
            no_context_c = torch.zeros_like(c_full)
            for ablation_label, ablation_spec in NO_RESPONSE_ABLATION_SPECS.items():
                ablated_c = no_context_c if ablation_spec.get("zero_context", False) else c_full
                model_overrides = ablation_spec.get("model_overrides", {})
                with _temporary_model_overrides(model, **model_overrides):
                    frames.append(
                        _compact_trace_frame(
                            run_experimental_phase(model, x_full, ablated_c, f"{ablation_label}_{condition}_{phase}", update=False),
                            condition=condition,
                            image_type=ablation_label,
                            phase=phase,
                            zscore_std_floor=cell_floor,
                        )
                    )
                    frames.append(
                        _compact_trace_frame(
                            run_experimental_phase(
                                model,
                                occluded_x,
                                ablated_c,
                                f"occlusion_{ablation_label}_{condition}_{phase}",
                                update=False,
                            ),
                            condition=condition,
                            image_type=f"occlusion_{ablation_label}",
                            phase=phase,
                            zscore_std_floor=cell_floor,
                        )
                    )

    return (
        transition,
        _summarize_cell_panel_traces(
            frames,
            stimuli,
            n_steps_per_phase=n_steps_per_phase,
            test_trials=test_trials,
            zscore_std_floor=cell_floor,
            cell_id=int(config["_sample_global_idx"]),
        ),
        stimuli,
    )


def _run_sector_average_panel_configs(
    configs_by_transition: dict[str, dict[str, Any]],
    *,
    n_steps_per_phase: int,
    test_trials: int,
    training_trials: int,
    training_stimulus_order: str,
    seed: int,
    n_jobs: int,
    zscore_std_floor: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[torch.Tensor, torch.Tensor]] | None]:
    order = list(configs_by_transition)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_sector_average_panel_config)(
            t,
            configs_by_transition[t],
            n_steps_per_phase=n_steps_per_phase,
            test_trials=test_trials,
            training_trials=training_trials,
            training_stimulus_order=training_stimulus_order,
            seed=seed,
            zscore_std_floor=zscore_std_floor,
        )
        for t in order
    )
    long_dfs = {t: long_df for t, long_df, _ in results}
    stimuli = results[0][2] if results else None
    return long_dfs, stimuli


def _render_panels(
    long_dfs: dict[str, pd.DataFrame],
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    *,
    out_dir: Path,
    n_steps_per_phase: int,
    test_trials: int,
    image_format: str,
    image_mode: str | None = None,
    transition_labels: dict[str, str] | None = None,
    name: str = "transition_panel",
    figure_size_inches: tuple[float, float] | None = None,
    zscore_std_floor: float | None = None,
    step_window: tuple[int, int] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if stimuli is None:
        return
    order = list(long_dfs)
    labels = transition_labels or {t: format_transition_label(t) for t in order}
    resolved_step_window = step_window or _panel_step_window(n_steps_per_phase, test_trials)
    if image_mode is None:
        save_grouped_transition_panels(
            long_dfs,
            stimuli=stimuli,
            save_path=str(out_dir),
            transition_order=order,
            transition_labels=labels,
            step_window=resolved_step_window,
            save_in_transition_subdir=False,
            zscore_activity=True,
            image_format=image_format,
            zscore_std_floor=zscore_std_floor,
        )
    else:
        visualize_transition_response_matrix(
            long_dfs,
            STIMULI=stimuli,
            save_path=str(out_dir),
            name=name,
            image_mode=image_mode,
            transition_order=order,
            transition_labels=labels,
            step_window=resolved_step_window,
            save_in_transition_subdir=False,
            save_csv=False,
            zscore_activity=True,
            image_format=image_format,
            figure_size_inches=figure_size_inches,
            zscore_std_floor=zscore_std_floor,
        )


def _save_panels(
    configs_by_transition: dict[str, dict[str, Any]],
    *,
    out_dir: Path,
    n_steps_per_phase: int,
    test_trials: int,
    training_trials: int,
    training_stimulus_order: str,
    seed: int,
    image_format: str,
    n_jobs: int,
    image_mode: str | None = None,
    transition_labels: dict[str, str] | None = None,
    name: str = "transition_panel",
    figure_size_inches: tuple[float, float] | None = None,
    zscore_std_floor: float = 0.04,
) -> None:
    """Render a transition panel for the given configs, parallelised across configs."""
    long_dfs, stimuli = _run_panel_configs(
        configs_by_transition,
        n_steps_per_phase=n_steps_per_phase,
        test_trials=test_trials,
        training_trials=training_trials,
        training_stimulus_order=training_stimulus_order,
        seed=seed,
        n_jobs=n_jobs,
        zscore_std_floor=zscore_std_floor,
    )
    _render_panels(
        long_dfs,
        stimuli,
        out_dir=out_dir,
        n_steps_per_phase=n_steps_per_phase,
        test_trials=test_trials,
        image_format=image_format,
        image_mode=image_mode,
        transition_labels=transition_labels,
        name=name,
        figure_size_inches=figure_size_inches,
        zscore_std_floor=zscore_std_floor,
    )


def _center_config_with_tuning(transition: str, tuned_indices: tuple[int, ...] | None = None) -> dict[str, Any]:
    if tuned_indices is None:
        return _center_config(transition)
    try:
        return _center_config(transition, tuned_indices=tuned_indices)
    except TypeError:
        return _center_config(transition)


def _center_panel_configs(
    transition_order: list[str],
    *,
    narrow_mode: str,
) -> dict[str, dict[str, Any]]:
    narrow_cycle = [(0,), (1,), (2,)]
    narrow_seen = 0
    configs: dict[str, dict[str, Any]] = {}
    for transition in transition_order:
        tuned_indices = None
        if "narrow" in transition:
            if narrow_mode == "familiar":
                tuned_indices = (0,)
            elif narrow_mode == "novel":
                tuned_indices = (2,)
            elif narrow_mode == "cycle":
                tuned_indices = narrow_cycle[narrow_seen % len(narrow_cycle)]
                narrow_seen += 1
            else:
                raise ValueError("narrow_mode must be 'familiar', 'novel', or 'cycle'.")
        configs[transition] = _center_config_with_tuning(transition, tuned_indices)
    return configs


def _save_center_panels(
    transition_order: list[str],
    *,
    output_dir: Path,
    n_steps_per_phase: int,
    test_trials: int,
    training_trials: int,
    training_stimulus_order: str,
    seed: int,
    image_format: str = "png",
    n_jobs: int = -1,
    zscore_std_floor: float = 0.04,
) -> None:
    """Sanity check: transition panels for the exact noise-free sampler centers."""
    center_dir = output_dir / "center_panels"
    _save_panels(
        _center_panel_configs(transition_order, narrow_mode="cycle"),
        out_dir=center_dir,
        n_steps_per_phase=n_steps_per_phase,
        test_trials=test_trials,
        training_trials=training_trials,
        training_stimulus_order=training_stimulus_order,
        seed=seed,
        image_format=image_format,
        n_jobs=n_jobs,
        zscore_std_floor=zscore_std_floor,
    )
    labels = {transition: format_transition_label(transition) for transition in transition_order}
    _save_panels(
        _center_panel_configs(transition_order, narrow_mode="familiar"),
        out_dir=center_dir,
        n_steps_per_phase=n_steps_per_phase,
        test_trials=test_trials,
        training_trials=training_trials,
        training_stimulus_order=training_stimulus_order,
        seed=seed,
        image_format=image_format,
        n_jobs=n_jobs,
        image_mode="familiar",
        transition_labels=labels,
        name="transitions_FAM",
        zscore_std_floor=zscore_std_floor,
    )
    _save_panels(
        _center_panel_configs(transition_order, narrow_mode="novel"),
        out_dir=center_dir,
        n_steps_per_phase=n_steps_per_phase,
        test_trials=test_trials,
        training_trials=training_trials,
        training_stimulus_order=training_stimulus_order,
        seed=seed,
        image_format=image_format,
        n_jobs=n_jobs,
        image_mode="novel",
        transition_labels=labels,
        name="transitions_NOV",
        zscore_std_floor=zscore_std_floor,
    )


def _save_highlight_example_panels(
    selected_examples: dict[str, list[dict[str, Any]]],
    *,
    output_dir: Path,
    n_steps_per_phase: int,
    test_trials: int,
    training_trials: int,
    training_stimulus_order: str,
    seed: int,
    image_format: str,
    n_jobs: int,
    zscore_std_floor: float,
    precomputed_long_dfs: dict[str, pd.DataFrame] | None = None,
    precomputed_stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> None:
    for group, examples in selected_examples.items():
        if not examples:
            continue
        configs_by_label: dict[str, dict[str, Any]] = {}
        transition_labels: dict[str, str] = {}
        for example in examples:
            key = example.get("example_key")
            if key is None:
                key = f"example_{example['number']:02d}"
            key = str(key)
            configs_by_label[key] = copy.deepcopy(example["sample"])
            transition_labels[key] = str(example.get("display_number", example["number"]))
        if precomputed_long_dfs is None:
            long_dfs, stimuli = _run_scatter_order_panel_configs(
                configs_by_label,
                n_steps_per_phase=n_steps_per_phase,
                test_trials=test_trials,
                training_trials=training_trials,
                training_stimulus_order=training_stimulus_order,
                seed=seed,
                n_jobs=n_jobs,
                zscore_std_floor=zscore_std_floor,
            )
            _render_panels(
                long_dfs,
                stimuli,
                out_dir=output_dir / "highlight_examples" / group,
                n_steps_per_phase=n_steps_per_phase,
                test_trials=test_trials,
                image_format=image_format,
                image_mode=group,
                transition_labels=transition_labels,
                name=f"highlighted_{group}_examples",
                figure_size_inches=HIGHLIGHT_EXAMPLE_FIGSIZE_INCHES,
                zscore_std_floor=zscore_std_floor,
            )
        else:
            long_dfs = {key: precomputed_long_dfs[key] for key in configs_by_label if key in precomputed_long_dfs}
            _render_panels(
                long_dfs,
                precomputed_stimuli,
                out_dir=output_dir / "highlight_examples" / group,
                n_steps_per_phase=n_steps_per_phase,
                test_trials=test_trials,
                image_format=image_format,
                image_mode=group,
                transition_labels=transition_labels,
                name=f"highlighted_{group}_examples",
                figure_size_inches=HIGHLIGHT_EXAMPLE_FIGSIZE_INCHES,
                zscore_std_floor=zscore_std_floor,
            )


def _save_sector_average_highlight_panels(
    transition_table: pd.DataFrame,
    samples: list[dict[str, Any]],
    *,
    output_dir: Path,
    n_steps_per_phase: int,
    test_trials: int,
    training_trials: int,
    training_stimulus_order: str,
    seed: int,
    image_format: str,
    n_jobs: int,
    zscore_std_floor: float,
    threshold: float,
    precomputed_long_dfs: dict[str, pd.DataFrame] | None = None,
    precomputed_stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> None:
    sample_by_idx = {int(sample["_sample_global_idx"]): sample for sample in samples}
    wide = _wide_table(transition_table)
    sector_order = ("+NO axis", "+O axis", "-NO axis", "-O axis")
    sector_labels = {sector: sector.replace(" axis", "") for sector in sector_order}
    sector_modes = ("sector-average", "sector-per-image")
    condition_by_image = {
        (image_group, image_idx_original, image_idx_within_group): condition
        for condition, (image_group, image_idx_original, image_idx_within_group) in IMAGE_INFO.items()
    }
    for legacy_group in ("familiar", "novel"):
        legacy_dir = output_dir / "sector_average_examples" / legacy_group
        if legacy_dir.exists():
            for legacy_file in legacy_dir.glob("sector_average_*_examples_sem.*"):
                if legacy_file.is_file():
                    legacy_file.unlink()

    def row_y_limits(row_bounds: list[float]) -> tuple[float, float] | None:
        if not row_bounds:
            return None
        lo = float(np.nanmin([*row_bounds, 0.0, SECTOR_TRACE_SCALE_BAR_UNITS]))
        hi = float(np.nanmax([*row_bounds, 0.0, SECTOR_TRACE_SCALE_BAR_UNITS]))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return None
        center = 0.5 * (lo + hi)
        span = max((hi - lo) * 1.2, SECTOR_TRACE_MIN_ROW_Y_SPAN)
        return center - 0.5 * span, center + 0.5 * span

    def add_scale_bar(ax: plt.Axes, *, length: float = SECTOR_TRACE_SCALE_BAR_UNITS) -> None:
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

    def build_sector_assignments(*, group: str, sector_mode: str) -> pd.DataFrame:
        if sector_mode == "sector-average":
            summary = th.build_mean_summary(
                wide,
                image_group=group,
                pre_stage="Naive",
                target_stage="Expert",
                threshold=threshold,
            )
            return summary[["neuron_idx", "RotatedSector"]].copy()

        group_wide = wide.loc[wide["image_group"].eq(group)].copy()
        if group_wide.empty:
            return pd.DataFrame(columns=["neuron_idx", "condition", "RotatedSector"])
        summaries: list[pd.DataFrame] = []
        image_keys = (
            group_wide[["image_idx_original", "image_idx_within_group"]]
            .drop_duplicates()
            .sort_values(["image_idx_within_group", "image_idx_original"])
        )
        for image in image_keys.itertuples(index=False):
            image_wide = group_wide.loc[
                group_wide["image_idx_original"].eq(image.image_idx_original)
                & group_wide["image_idx_within_group"].eq(image.image_idx_within_group)
            ].copy()
            summary = th.build_mean_summary(
                image_wide,
                image_group=group,
                pre_stage="Naive",
                target_stage="Expert",
                threshold=threshold,
            )
            condition = condition_by_image.get((group, int(image.image_idx_original), int(image.image_idx_within_group)))
            if condition is None:
                continue
            summary["condition"] = condition
            summaries.append(summary[["neuron_idx", "condition", "RotatedSector"]])
        if not summaries:
            return pd.DataFrame(columns=["neuron_idx", "condition", "RotatedSector"])
        return pd.concat(summaries, ignore_index=True)

    def trace_df_for_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
        neuron_ids = sorted(assignments["neuron_idx"].astype(int).unique().tolist())
        configs = {
            f"cell_{neuron_idx}": copy.deepcopy(sample_by_idx[neuron_idx])
            for neuron_idx in neuron_ids
            if neuron_idx in sample_by_idx
        }
        if precomputed_long_dfs is not None:
            member_trace_dfs = {key: precomputed_long_dfs[key] for key in configs if key in precomputed_long_dfs}
        elif configs:
            member_trace_dfs, _ = _run_sector_average_panel_configs(
                configs,
                n_steps_per_phase=n_steps_per_phase,
                test_trials=test_trials,
                training_trials=training_trials,
                training_stimulus_order=training_stimulus_order,
                seed=seed,
                n_jobs=n_jobs,
                zscore_std_floor=zscore_std_floor,
            )
        else:
            member_trace_dfs = {}
        if not member_trace_dfs:
            return pd.DataFrame()
        return pd.concat(member_trace_dfs.values(), ignore_index=True)

    def attach_sectors(
        trace_df: pd.DataFrame,
        assignments: pd.DataFrame,
        *,
        sector_mode: str,
        selected_conditions: list[str],
    ) -> pd.DataFrame:
        if trace_df.empty or assignments.empty:
            return pd.DataFrame()
        traces = trace_df.loc[trace_df["condition"].isin(selected_conditions)].copy()
        assignments = assignments.copy()
        assignments["cell_id"] = assignments["neuron_idx"].astype(int)
        if sector_mode == "sector-average":
            sector_lookup = assignments[["cell_id", "RotatedSector"]].drop_duplicates()
            labeled = traces.merge(sector_lookup, on="cell_id", how="inner", validate="many_to_one")
        else:
            sector_lookup = assignments[["cell_id", "condition", "RotatedSector"]].drop_duplicates()
            labeled = traces.merge(sector_lookup, on=["cell_id", "condition"], how="inner", validate="many_to_one")
        return labeled.loc[labeled["RotatedSector"].astype(str).isin(sector_order)].copy()

    def summarize_trace_rows(labeled: pd.DataFrame, *, pooled: bool, group: str, sector_mode: str) -> pd.DataFrame:
        if labeled.empty:
            return pd.DataFrame()
        if pooled:
            index_cols = [
                "RotatedSector",
                "column_key",
                "column_label",
                "experiment_phase",
                "response_type",
                "image_type",
                "x_seconds",
                "stim_start_seconds",
                "stim_end_seconds",
            ]
        else:
            index_cols = [
                "RotatedSector",
                "condition",
                "column_key",
                "column_label",
                "experiment_phase",
                "response_type",
                "image_type",
                "x_seconds",
                "stim_start_seconds",
                "stim_end_seconds",
            ]
        summary = (
            labeled.groupby(index_cols, observed=True, as_index=False)
            .agg(
                mean_y=("y", "mean"),
                sd_y=("y", "std"),
                n_cells=("cell_id", "nunique"),
                n_responses=("y", "size"),
                n_conditions=("condition", "nunique"),
            )
            .sort_values(index_cols)
        )
        denom = summary["n_responses"] if pooled else summary["n_cells"]
        summary["sem"] = summary["sd_y"].fillna(0.0) / np.sqrt(np.maximum(denom.to_numpy(dtype=float), 1.0))
        summary.insert(0, "sector_mode", sector_mode)
        summary.insert(1, "image_group", group)
        summary["sector_label"] = summary["RotatedSector"].map(sector_labels)
        return summary

    def save_trace_panel(
        *,
        group: str,
        trace_summary: pd.DataFrame,
        sector_mode: str,
        pooled: bool,
    ) -> None:
        selected_conditions = ["familiar_1", "familiar_2"] if group == "familiar" else ["novel"]
        if pooled:
            column_pairs = [(column_spec, "pooled") for column_spec in TRANSITION_RESPONSE_COLUMN_SPECS]
        else:
            column_pairs = [
                (column_spec, condition)
                for column_spec in TRANSITION_RESPONSE_COLUMN_SPECS
                for condition in selected_conditions
                if not trace_summary.loc[
                    trace_summary.get("condition", pd.Series(dtype=object)).astype(str).eq(condition)
                    & trace_summary["column_key"].eq(column_spec["key"])
                ].empty
            ]
        if not column_pairs:
            return

        out_dir = output_dir / "sector_average_examples" / sector_mode / group
        out_dir.mkdir(parents=True, exist_ok=True)
        name_prefix = "sector_per_image" if sector_mode == "sector-per-image" else "sector_average"
        pooled_suffix = "_pooled" if pooled else ""
        base = out_dir / f"{name_prefix}_{group}{pooled_suffix}_examples_sem"
        trace_summary.to_csv(f"{base}.csv", index=False)
        fig, axes = plt.subplots(
            len(sector_order),
            len(column_pairs),
            figsize=(max(7.0, 1.65 * len(column_pairs)), 1.8 * len(sector_order) + 1.0),
            squeeze=False,
            sharex=True,
            sharey=False,
        )
        fig.subplots_adjust(left=0.12, right=0.995, top=0.82, bottom=0.06, wspace=0.14, hspace=0.22)

        for col_idx, (column_spec, condition) in enumerate(column_pairs):
            condition_label = "pooled" if pooled else condition.replace("_", " ")
            title = f"{condition_label}\n{column_spec['label']}"
            axes[0, col_idx].set_title(title, fontsize=8)

        for row_idx, sector in enumerate(sector_order):
            sector_df = trace_summary.loc[trace_summary["RotatedSector"].astype(str).eq(sector)].copy()
            row_bounds: list[float] = []
            for col_idx, (column_spec, condition) in enumerate(column_pairs):
                ax = axes[row_idx, col_idx]
                ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
                for response_type, trace_type, color in (
                    ("O", column_spec["o_trace"], "red"),
                    ("NO", column_spec["no_trace"], "black"),
                ):
                    if sector_df.empty:
                        continue
                    mask = (
                        sector_df["column_key"].eq(column_spec["key"])
                        & sector_df["experiment_phase"].eq(column_spec["phase"])
                        & sector_df["image_type"].eq(trace_type)
                        & sector_df["response_type"].eq(response_type)
                    )
                    if not pooled:
                        mask &= sector_df["condition"].eq(condition)
                    trace_df = sector_df.loc[mask].copy()
                    if trace_df.empty:
                        continue
                    trace_df = trace_df.sort_values("x_seconds")
                    x_seconds = trace_df["x_seconds"].to_numpy(dtype=float)
                    y_mean = trace_df["mean_y"].to_numpy(dtype=float)
                    y_spread = trace_df["sem"].fillna(0.0).to_numpy(dtype=float)
                    ax.plot(x_seconds, y_mean, color=color, lw=1.2)
                    ax.fill_between(
                        x_seconds,
                        y_mean - y_spread,
                        y_mean + y_spread,
                        color=color,
                        alpha=0.18,
                        linewidth=0,
                    )
                    row_bounds.extend((float(np.nanmin(y_mean - y_spread)), float(np.nanmax(y_mean + y_spread))))
                    if not trace_df.empty:
                        ax.axvspan(
                            float(trace_df["stim_start_seconds"].iloc[0]),
                            float(trace_df["stim_end_seconds"].iloc[0]),
                            color="0.92",
                            zorder=-1,
                        )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            for ax in axes[row_idx, :]:
                ax.text(
                    -0.08,
                    0.5,
                    sector_labels[sector],
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=9,
                )
                break
            y_limits = row_y_limits(row_bounds)
            if y_limits is not None:
                for ax in axes[row_idx, :]:
                    ax.set_ylim(y_limits)
                add_scale_bar(axes[row_idx, 0])

        title = "Sector-average traces +/- SEM"
        fig.suptitle(title, y=0.98, fontsize=11)
        fig.savefig(f"{base}.{image_format}", dpi=300)
        plt.close(fig)

    for group in ("familiar", "novel"):
        selected_conditions = ["familiar_1", "familiar_2"] if group == "familiar" else ["novel"]
        for sector_mode in sector_modes:
            assignments = build_sector_assignments(group=group, sector_mode=sector_mode)
            trace_df = trace_df_for_assignments(assignments)
            labeled = attach_sectors(
                trace_df,
                assignments,
                sector_mode=sector_mode,
                selected_conditions=selected_conditions,
            )
            if labeled.empty:
                continue
            for pooled in (False, True):
                trace_summary = summarize_trace_rows(
                    labeled,
                    pooled=pooled,
                    group=group,
                    sector_mode=sector_mode,
                )
                save_trace_panel(
                    group=group,
                    trace_summary=trace_summary,
                    sector_mode=sector_mode,
                    pooled=pooled,
                )


def _single_y_frame(long_df: pd.DataFrame, *, condition: str, phase: str, image_type: str) -> pd.DataFrame:
    frame = long_df.loc[
        long_df["condition"].eq(condition)
        & long_df["experiment_phase"].eq(phase)
        & long_df["image_type"].eq(image_type),
        ["step", "y"],
    ].copy()
    return frame.drop_duplicates("step").sort_values("step").reset_index(drop=True)


def _response_df_from_panel_long_dfs(
    long_dfs: dict[str, pd.DataFrame],
    samples_by_key: dict[str, dict[str, Any]],
    *,
    n_steps_per_phase: int,
    response_tail_fraction: float,
    zscore_std_floor: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, long_df in long_dfs.items():
        sample = samples_by_key[key]
        cell_floor = max(zscore_std_floor, BASELINE_STD_SCALE * float(sample.get("baseline_drive_sigma", 0.0)))
        naive_frames = [
            _single_y_frame(long_df, condition=condition, phase="naive", image_type=image_type)
            for condition in IMAGE_INFO
            for image_type in TRACE_TYPES
        ]
        naive_baseline = _baseline(naive_frames, n_steps_per_phase)
        for phase in ("naive", "expert"):
            for condition in IMAGE_INFO:
                for trace, image_type in TRACE_TYPES.items():
                    frame = _single_y_frame(long_df, condition=condition, phase=phase, image_type=trace)
                    rows.append(
                        dict(
                            condition=condition,
                            phase=phase,
                            stage=STAGES[phase],
                            trace=trace,
                            image_type=image_type,
                            response=_response_from_frame(
                                frame,
                                n_steps_per_phase=n_steps_per_phase,
                                response_tail_fraction=response_tail_fraction,
                                baseline=naive_baseline,
                                zscore_std_floor=cell_floor,
                            ),
                            transition=sample["_canonical_transition"],
                            sample_idx=sample["_sample_idx"],
                            sample_global_idx=sample["_sample_global_idx"],
                            seed=sample["seed"],
                            experiment_series=PRIMARY_EXPERIMENT_SERIES,
                        )
                    )
    return pd.DataFrame(rows)


def _robust_response_limits(summaries: list[pd.DataFrame], *, hi_percentile: float, pad: float = 0.4) -> list[float]:
    """Response axis limits that ignore a few extreme outliers (point 6): scale to
    the bulk so a handful of very strong responders fall outside rather than
    blowing up the whole panel. hi_percentile=100 reproduces the real-data min/max."""
    cols = ["NO_Pre", "O_Pre", "NO_Target", "O_Target"]
    values = np.concatenate([s[cols].to_numpy(dtype=float).reshape(-1) for s in summaries])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [-1.0, 1.0]
    lo = float(np.nanpercentile(values, max(0.0, 100.0 - hi_percentile)))
    hi = float(np.nanpercentile(values, hi_percentile))
    return [lo - pad, hi + pad]


def _robust_shift_limits(summaries: list[pd.DataFrame], *, hi_percentile: float, pad_ratio: float = 0.12, fallback: float = 0.5) -> list[float]:
    values = np.concatenate([s[["dNO", "dO"]].to_numpy(dtype=float).reshape(-1) for s in summaries])
    values = np.abs(values[np.isfinite(values)])
    if values.size == 0:
        return [-fallback, fallback]
    extent = float(np.nanpercentile(values, hi_percentile))
    if not np.isfinite(extent) or extent == 0:
        extent = fallback
    else:
        extent *= 1.0 + pad_ratio
    return [-extent, extent]


def _annotate_examples_on_axis(ax: plt.Axes, examples: list[dict[str, Any]], *, x_key: str, y_key: str) -> None:
    # Match the regular scatter style so the highlighted center is just another
    # point of its sector colour and size -- only the leader-line + numeric label
    # tells the reader it's a highlighted example.
    point_size = th.DEFAULT_PLOT_STYLE.get("point_size", 28)
    for order_idx, example in enumerate(examples):
        offset_x = -24 if order_idx % 2 else 24
        offset_y = 18 + 5 * (order_idx % 3)
        ax.scatter(
            [example[x_key]],
            [example[y_key]],
            s=point_size,
            facecolors=example["color"],
            edgecolors="none",
            linewidths=0,
            zorder=20,
        )
        # Per-group 1..n label (the position in --fam-examples / --nov-examples);
        # falls back to the global template number if `display_number` isn't set.
        label = str(example.get("display_number", example["number"]))
        ax.annotate(
            label,
            xy=(example[x_key], example[y_key]),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=13,
            color="0.15",
            arrowprops=dict(arrowstyle="-", color="0.25", lw=1.0, shrinkA=1, shrinkB=4),
            zorder=21,
        )


def _annotate_highlights(fig: plt.Figure, examples: list[dict[str, Any]]) -> None:
    if not examples:
        return
    axis_specs = (
        (0, "dNO", "dO"),
        (1, "NO_Pre", "O_Pre"),
        (2, "NO_Target", "O_Target"),
        (3, "dNO", "dO"),
        (4, "NO_Pre", "O_Pre"),
        (5, "NO_Target", "O_Target"),
        (6, "dNO", "dO"),
        (7, "NO_Pre", "O_Pre"),
        (8, "NO_Target", "O_Target"),
    )
    for ax_idx, x_key, y_key in axis_specs:
        if ax_idx >= len(fig.axes):
            continue
        _annotate_examples_on_axis(fig.axes[ax_idx], examples, x_key=x_key, y_key=y_key)


def _draw_rotated_sector_shift_axis(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    title: str,
    shift_lims: list[float],
    highlights: list[dict[str, Any]] | None = None,
    show_legend: bool = True,
) -> None:
    sector_means = th.sector_mean_table(summary)
    sector_labels = th.sector_labels_with_counts(summary)
    sector_arrow_alphas = th._sector_percentage_alphas(summary)
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
            sector_rows["dNO"],
            sector_rows["dO"],
            s=PLOT_STYLE["point_size"],
            c=rgba,
            edgecolors="none",
            zorder=th._sector_scatter_zorder(sector),
        )
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

    th._draw_origin_guides(ax)
    th._draw_rotated_guides(ax, shift_lims)
    ax.set_xlim(shift_lims)
    ax.set_ylim(shift_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("dNO")
    ax.set_ylabel("dO")
    if show_legend:
        ax.legend(
            handles=th._legend_handles(sector_labels, linewidth=PLOT_STYLE["mean_arrow_width"]),
            frameon=False,
            loc="best",
            fontsize=9,
        )
    _annotate_examples_on_axis(ax, highlights or [], x_key="dNO", y_key="dO")


def _save_rotated_sector_shift_summary_panel(
    aggregate: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
    shift_lims: list[float],
    image_format: str,
    highlights: dict[str, list[dict[str, Any]]],
) -> None:
    out_dir = output_dir / "figures" / "summary_panels"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.2), sharex=True, sharey=True)
    for ax, group in zip(axes, ("familiar", "novel"), strict=False):
        _draw_rotated_sector_shift_axis(
            ax,
            aggregate[group],
            title=group.capitalize(),
            shift_lims=shift_lims,
            highlights=highlights.get(group, []),
            show_legend=True,
        )
    fig.suptitle("Expert - Naive by rotated sector", fontsize=17)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(out_dir / f"aggregate_rotated_sector_shift_summary_panel.{image_format}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for group, summary in aggregate.items():
        fig, ax = plt.subplots(figsize=(5.4, 5.2))
        _draw_rotated_sector_shift_axis(
            ax,
            summary,
            title=f"{group.capitalize()}: Expert - Naive by rotated sector",
            shift_lims=shift_lims,
            highlights=highlights.get(group, []),
            show_legend=True,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"aggregate_{group}_rotated_sector_shift.{image_format}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _export_sector_response_panels(
    summary: pd.DataFrame,
    output_dir: Path,
    basename: str,
    *,
    response_lims: list[float],
    image_format: str,
    highlights: list[dict[str, Any]] | None = None,
    dpi: int = 300,
) -> list[Path]:
    def draw_panel(ax: plt.Axes, *, x_col: str, y_col: str, x_key: str, y_key: str) -> None:
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
        if highlights:
            for order_idx, example in enumerate(highlights):
                offset_x = -24 if order_idx % 2 else 24
                offset_y = 18 + 5 * (order_idx % 3)
                ax.scatter(
                    [example[x_key]],
                    [example[y_key]],
                    s=42,
                    facecolors=example["color"],
                    edgecolors="black",
                    linewidths=0.4,
                    zorder=20,
                )
                ax.annotate(
                    str(example.get("display_number", example["number"])),
                    xy=(example[x_key], example[y_key]),
                    xytext=(offset_x, offset_y),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="0.15",
                    arrowprops=dict(arrowstyle="-", color="0.25", lw=1.0, shrinkA=1, shrinkB=4),
                    zorder=21,
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
    formats = (image_format,)
    lims = response_lims
    ticks = np.arange(np.ceil(lims[0]), np.floor(lims[1]) + 1.0, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.22, top=0.82, wspace=0.18)

    draw_panel(axes[0], x_col="NO_Pre", y_col="O_Pre", x_key="NO_Pre", y_key="O_Pre")
    style_axis(axes[0], title="Naive")
    draw_panel(axes[1], x_col="NO_Target", y_col="O_Target", x_key="NO_Target", y_key="O_Target")
    style_axis(axes[1], title="Expert")
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


def _save_summary(
    summary: pd.DataFrame,
    path: Path,
    title: str,
    response_lims: list[float],
    shift_lims: list[float],
    export_panels: bool,
    image_format: str,
    highlights: list[dict[str, Any]] | None = None,
    export_sector_response_panels: bool = False,
) -> None:
    """Render and save one rotated-sector scatter figure (plus its sector legend)
    via the shared real-data plotting helper, using a fixed Naive->Expert frame."""
    fig = th.plot_mean_transition_summary(
        summary,
        title=title,
        start_label="Naive",
        end_label="Expert",
        response_lims=response_lims,
        shift_lims=shift_lims,
        style=PLOT_STYLE,
    )
    _annotate_highlights(fig, highlights or [])
    path = path.with_suffix(f".{image_format}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if export_panels:
        if export_sector_response_panels:
            _export_sector_response_panels(
                summary,
                path.parent / f"{path.stem}_panels",
                path.stem,
                response_lims=response_lims,
                image_format=image_format,
                highlights=highlights,
            )
        else:
            th.export_figure_panels(fig, path.parent / f"{path.stem}_panels", path.stem, formats=(image_format,))
    plt.close(fig)
    th.save_rotated_sector_unit_legend(summary, path.with_name(f"{path.stem}_sector_legend.{image_format}"), title=None, formats=(image_format,))


def _save_plots(
    transition_table: pd.DataFrame,
    *,
    output_dir: Path,
    transition_order: list[str],
    samples: list[dict[str, Any]],
    highlights: dict[str, list[dict[str, Any]]],
    template_numbers: dict[int, str],
    example_transition_table: pd.DataFrame | None = None,
    example_samples: list[dict[str, Any]] | None = None,
    example_source: str = "sample",
    threshold: float,
    plot_by_transition: bool,
    export_panels: bool,
    image_format: str,
    axis_clip_percentile: float,
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate the per-cell responses into familiar/novel rotated-sector
    summaries, save their figures and sector-fraction tables (and optionally a
    per-transition breakdown)."""
    figures_dir = output_dir / "figures"
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    wide = _wide_table(transition_table)
    aggregate = {
        group: exs.summary_with_transition(
            th.build_mean_summary(wide, image_group=group, pre_stage="Naive", target_stage="Expert", threshold=threshold),
            samples,
        )
        for group in ("familiar", "novel")
    }
    example_wide = _wide_table(example_transition_table) if example_transition_table is not None else wide
    example_sample_list = example_samples if example_samples is not None else samples
    example_summaries = {
        group: exs.summary_with_transition(
            th.build_mean_summary(example_wide, image_group=group, pre_stage="Naive", target_stage="Expert", threshold=threshold),
            example_sample_list,
        )
        for group in ("familiar", "novel")
    }
    selected_examples = exs.select_highlight_examples(
        example_summaries,
        samples=example_sample_list,
        aggregate_summaries=aggregate,
        aggregate_samples=samples,
        highlights=highlights,
        template_numbers=template_numbers,
        threshold=threshold,
    )
    # A single shared response/shift frame across the familiar and novel panels
    # (like the transitions>threshold notebook), but scaled to the bulk so a few
    # extreme outliers fall outside the panel instead of compressing it
    # (axis_clip_percentile=100 reproduces the notebook's exact min/max framing).
    summaries = list(aggregate.values())
    if example_transition_table is not None:
        summaries.extend(example_summaries.values())
    response_lims = _robust_response_limits(summaries, hi_percentile=axis_clip_percentile)
    shift_lims = _robust_shift_limits(summaries, hi_percentile=axis_clip_percentile)

    for group, summary in aggregate.items():
        _save_summary(
            summary,
            figures_dir / f"aggregate_{group}_summary.png",
            f"Model scatter - all transitions - {group}",
            response_lims,
            shift_lims,
            export_panels,
            image_format,
            highlights=selected_examples[group],
            export_sector_response_panels=True,
        )

    if export_panels:
        _save_rotated_sector_shift_summary_panel(
            aggregate,
            output_dir=output_dir,
            shift_lims=shift_lims,
            image_format=image_format,
            highlights=selected_examples,
        )

    if plot_by_transition:
        for transition in transition_order:
            subset = wide.loc[wide["transition"] == transition].copy()
            if subset.empty:
                continue
            for group in ("familiar", "novel"):
                summary = th.build_mean_summary(subset, image_group=group, pre_stage="Naive", target_stage="Expert", threshold=threshold)
                _save_summary(
                    summary,
                    figures_dir / "by_transition" / f"{transition}_{group}_summary.png",
                    f"Model scatter - {transition} - {group}",
                    response_lims,
                    shift_lims,
                    export_panels,
                    image_format,
                )

    rows = [
        {
            "image_group": group,
            "template_number": example["number"],
            "display_number": example["display_number"],
            "transition": example["transition"],
            "neuron_idx": example["neuron_idx"],
            "sector": example["sector"],
            "source": example_source,
            "selection_rule": example["selection_rule"],
            "requested_sector": example["requested_sector"],
            "requested_diagonal": example["requested_diagonal"],
            "magnitude_band": example["magnitude_band"],
            "NO_Pre": example["NO_Pre"],
            "O_Pre": example["O_Pre"],
            "NO_Target": example["NO_Target"],
            "O_Target": example["O_Target"],
            "dNO": example["dNO"],
            "dO": example["dO"],
            "dNorm": float(np.hypot(example["dNO"], example["dO"])),
            "diagonal_distance": example["diagonal_distance"],
        }
        for group, examples in selected_examples.items()
        for example in examples
    ]
    del rows
    return selected_examples


def _result_summary(transition_table: pd.DataFrame, samples: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    wide = _wide_table(transition_table)
    result: dict[str, Any] = {}
    for group in ("familiar", "novel"):
        summary = exs.summary_with_transition(
            th.build_mean_summary(wide, image_group=group, pre_stage="Naive", target_stage="Expert", threshold=threshold),
            samples,
        )
        sector_counts = summary["RotatedSector"].astype(str).value_counts().to_dict()
        n = int(len(summary))
        result[group] = {
            "n": n,
            "sector_counts": {sector: int(sector_counts.get(sector, 0)) for sector in th.ROTATED_SECTOR_ORDER},
            "sector_fractions": {
                sector: float(sector_counts.get(sector, 0) / n) if n else 0.0
                for sector in th.ROTATED_SECTOR_ORDER
            },
            "mean_dNO": float(summary["dNO"].mean()) if n else float("nan"),
            "mean_dO": float(summary["dO"].mean()) if n else float("nan"),
            "mean_dNorm": float(summary["dNorm"].mean()) if n else float("nan"),
            "median_dNorm": float(summary["dNorm"].median()) if n else float("nan"),
        }
    return result


def run_model_scatter(args: argparse.Namespace) -> None:
    """End-to-end driver: sample the population, simulate every cell in parallel,
    write the response tables/metadata, and render the scatter figures (and, by
    default, the center/canonical sanity panels)."""
    if args.n_samples < 1:
        raise ValueError("n_samples must be >= 1.")
    if args.n_steps_per_phase < 4:
        raise ValueError("n_steps_per_phase must be >= 4.")
    if args.test_trials < 1 or args.training_trials < 1:
        raise ValueError("test_trials and training_trials must be >= 1.")
    if not 0.0 < args.response_tail_fraction <= 1.0:
        raise ValueError("response_tail_fraction must be in (0, 1].")
    if args.training_stimulus_order not in {"fixed", "randomized"}:
        raise ValueError("training_stimulus_order must be 'fixed' or 'randomized'.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in args.output_dir.rglob("*"):
        if stale_path.is_file() and stale_path.suffix.lower() in {".csv", ".eps"}:
            stale_path.unlink()
    transition_order = list(minimal_configs3)
    template_numbers = _template_number_map(transition_order)
    highlights = exs.parse_highlight_numbers(args)
    exs.validate_highlight_numbers(highlights, transition_order)
    samples = _sample_configs(args, transition_order)

    torch.manual_seed(args.seed)
    test_stimuli = _build_model_scatter_test_stimuli(n_steps_per_phase=args.n_steps_per_phase, n_trials=args.test_trials)
    training_stimuli = _build_model_scatter_training_stimuli(
        n_steps_per_phase=args.n_steps_per_phase,
        n_trials=args.training_trials,
        order=args.training_stimulus_order,
        seed=args.seed,
    )
    run_start = time.perf_counter()
    print("[model_scatter] simulating sampled cells", flush=True)
    sample_outputs = Parallel(n_jobs=args.n_jobs, verbose=10 if args.n_jobs != 1 else 0)(
        delayed(_run_sample_with_sector_panel_trace)(
            sample,
            n_steps_per_phase=args.n_steps_per_phase,
            test_trials=args.test_trials,
            response_tail_fraction=args.response_tail_fraction,
            test_stimuli=test_stimuli,
            training_stimuli=training_stimuli,
            n_steps_per_phase_display=args.n_steps_per_phase,
            zscore_std_floor=args.zscore_std_floor,
        )
        for sample in samples
    )
    print(f"[model_scatter] sampled-cell simulation finished in {time.perf_counter() - run_start:.1f}s", flush=True)
    response_frames = [response_frame for response_frame, _, _, _ in sample_outputs]
    sector_panel_long_dfs = {key: panel_df for _, key, panel_df, _ in sample_outputs}
    sector_panel_stimuli = sample_outputs[0][3] if sample_outputs else None
    response_df = pd.concat(response_frames, ignore_index=True)
    transition_table = _transition_table(response_df)
    invalid = transition_table.loc[~np.isfinite(transition_table["response"])].copy()
    highlight_example_source = "uniform_sample_first_above_or_closest"
    highlight_example_samples: list[dict[str, Any]] = []
    highlight_example_transition_table: pd.DataFrame | None = None
    if exs.highlight_requested(highlights):
        highlight_example_samples, highlight_example_transition_table = exs.sample_uniform_highlight_pool(
            args,
            transition_order=transition_order,
            highlights=highlights,
            template_numbers=template_numbers,
            minimal_configs=minimal_configs3,
            perturb_config=_perturb_config,
            run_sample=_run_sample,
            transition_table=_transition_table,
            wide_table=_wide_table,
            test_stimuli=test_stimuli,
            training_stimuli=training_stimuli,
        )
        if highlight_example_transition_table is not None and highlight_example_transition_table.empty:
            highlight_example_transition_table = None

    (args.output_dir / "sampled_configs.json").write_text(json.dumps(samples, indent=2, default=repr))

    counts = {name: sum(sample["_canonical_transition"] == name for sample in samples) for name in transition_order}
    metadata = {
        "requested_n_samples": args.n_samples,
        "n_samples_total": len(samples),
        "transition_sampling": "canonical" if args.canonical_only else args.transition_sampling,
        "transition_sample_counts": counts,
        "transition_weights": {name: TRANSITIONS[name]["weight"] for name in transition_order},
        "template_numbers": template_numbers,
        "highlight_requested_examples": highlights,
        "highlight_example_source": highlight_example_source,
        "highlight_candidate_n_samples": len(highlight_example_samples),
        "highlight_candidate_transition_sample_counts": {
            name: sum(sample["_canonical_transition"] == name for sample in highlight_example_samples)
            for name in transition_order
        },
        "seed": args.seed,
        "n_steps_per_phase": args.n_steps_per_phase,
        "test_trials": args.test_trials,
        "test_condition_order": list(STIMULUS_SPECS),
        "training_trials": args.training_trials,
        "training_stimulus_order": args.training_stimulus_order,
        "training_trial_order": _training_trial_order(
            [name for name in STIMULUS_SPECS if name.startswith("familiar")],
            n_trials=args.training_trials,
            order=args.training_stimulus_order,
            seed=args.seed,
        ),
        "fixed_scalars": list(FIXED_SCALARS),
        "sampled_init_keys": ["w_ff_init", "w_fb_init"],
        "fixed_template_init_keys": ["w_lat_init", "w_pv_lat_init", "W_pv_init"],
        "scalar_noise_keys": list(SCALAR_NOISE),
        "zscore_std_floor": args.zscore_std_floor,
        "response_tail_fraction": args.response_tail_fraction,
        "sector_threshold": args.threshold,
        "stimulus_specs": STIMULUS_SPECS,
        "n_invalid_response_rows": int(len(invalid)),
        "result_summary": _result_summary(transition_table, samples, threshold=args.threshold),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=repr))

    plot_start = time.perf_counter()
    print("[model_scatter] rendering aggregate scatter plots", flush=True)
    selected_examples = _save_plots(
        transition_table,
        output_dir=args.output_dir,
        transition_order=transition_order,
        samples=samples,
        highlights=highlights,
        template_numbers=template_numbers,
        example_transition_table=highlight_example_transition_table,
        example_samples=highlight_example_samples or None,
        example_source=highlight_example_source,
        threshold=args.threshold,
        plot_by_transition=args.plot_by_transition,
        export_panels=args.export_panels,
        image_format=args.image_format,
        axis_clip_percentile=args.axis_clip_percentile,
    )
    print(f"[model_scatter] aggregate scatter plots finished in {time.perf_counter() - plot_start:.1f}s", flush=True)
    panel_start = time.perf_counter()
    print("[model_scatter] rendering highlighted example panels", flush=True)
    _save_highlight_example_panels(
        selected_examples,
        output_dir=args.output_dir,
        n_steps_per_phase=args.n_steps_per_phase,
        test_trials=args.test_trials,
        training_trials=args.training_trials,
        training_stimulus_order=args.training_stimulus_order,
        seed=args.seed,
        image_format=args.image_format,
        n_jobs=args.n_jobs,
        zscore_std_floor=args.zscore_std_floor,
        precomputed_long_dfs=None,
        precomputed_stimuli=None,
    )
    print(f"[model_scatter] highlighted example panels finished in {time.perf_counter() - panel_start:.1f}s", flush=True)
    sector_start = time.perf_counter()
    print("[model_scatter] rendering sector-average trace panels", flush=True)
    _save_sector_average_highlight_panels(
        transition_table,
        samples,
        output_dir=args.output_dir,
        n_steps_per_phase=args.n_steps_per_phase,
        test_trials=args.test_trials,
        training_trials=args.training_trials,
        training_stimulus_order=args.training_stimulus_order,
        seed=args.seed,
        image_format=args.image_format,
        n_jobs=args.n_jobs,
        zscore_std_floor=args.zscore_std_floor,
        threshold=args.threshold,
        precomputed_long_dfs=sector_panel_long_dfs,
        precomputed_stimuli=sector_panel_stimuli,
    )
    print(f"[model_scatter] sector-average trace panels finished in {time.perf_counter() - sector_start:.1f}s", flush=True)

    if not args.skip_center_panels and not args.canonical_only:
        center_start = time.perf_counter()
        print("[model_scatter] rendering center panels", flush=True)
        _save_center_panels(
            transition_order,
            output_dir=args.output_dir,
            n_steps_per_phase=args.n_steps_per_phase,
            test_trials=args.test_trials,
            training_trials=args.training_trials,
            training_stimulus_order=args.training_stimulus_order,
            seed=args.seed,
            image_format=args.image_format,
            n_jobs=args.n_jobs,
            zscore_std_floor=args.zscore_std_floor,
        )
        print(f"[model_scatter] center panels finished in {time.perf_counter() - center_start:.1f}s", flush=True)
    print(f"[model_scatter] run finished in {time.perf_counter() - run_start:.1f}s", flush=True)
