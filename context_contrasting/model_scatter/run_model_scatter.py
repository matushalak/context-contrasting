from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from pandas import DataFrame

import context_contrasting.data_analysis.transitions_helpers as th
from context_contrasting.minimal2.config_s import minimal_configs3
from context_contrasting.minimal2.experiment_s import (
    PRIMARY_EXPERIMENT_SERIES,
    STIMULUS_SPECS,
    _build_test_stimuli,
    _build_training_stimuli,
    run_experimental_phase,
)
from context_contrasting.minimal2.minimal_s import CCNeuron


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs"

STAGE_LABELS = {"naive": "Naive", "expert": "Expert"}
TRACE_TO_IMAGE_TYPE = {"full": "Full", "occlusion": "Occl"}
CONDITION_TO_IMAGE_INFO = {
    "familiar_1": ("familiar", 1, 1),
    "familiar_2": ("familiar", 2, 2),
    "novel": ("novel", 3, 1),
}

PLOT_STYLE = th.DEFAULT_PLOT_STYLE | {
    "pre_point_alpha": 1.0,
    "target_point_alpha": 1.0,
    "shift_point_alpha": 1.0,
    "pre_vector_alpha": 0.5,
    "target_vector_alpha": 0.5,
    "individual_vector_width": 0.005,
    "mean_arrow_width": 3.1,
    "mean_arrow_mutation_scale": 16.5,
}


@dataclass(frozen=True)
class ScalarNoiseSpec:
    mode: str
    scale: float
    lower: float | None = None
    upper: float | None = None
    zero_floor: float = 0.0


POSITIVE_SCALAR_SPECS = {
    "lr_ff": ScalarNoiseSpec("log", 0.25, 1e-5, 0.2),
    "lr_fb": ScalarNoiseSpec("log", 0.25, 1e-5, 0.2),
    "lr_lat": ScalarNoiseSpec("log", 0.25, 1e-5, 0.2),
    "lr_pv": ScalarNoiseSpec("log", 0.25, 1e-5, 0.2),
    "pyc_decay": ScalarNoiseSpec("log", 0.15, 1e-4, 0.95),
    "pv_decay": ScalarNoiseSpec("log", 0.15, 1e-4, 0.95),
    "apical_gain_strength": ScalarNoiseSpec("log", 0.18, 0.1, 50.0),
    "apical_gain_k": ScalarNoiseSpec("log", 0.18, 0.1, 30.0),
    "baseline_drive_sigma": ScalarNoiseSpec("log", 0.20, 0.0, 1.0),
    "pv_noise_sigma": ScalarNoiseSpec("log", 0.20, 0.0, 0.5),
    "alpha": ScalarNoiseSpec("log", 0.12, 0.05, 10.0),
}

ADDITIVE_SCALAR_SPECS = {
    "apical_drive_threshold": ScalarNoiseSpec("add", 0.12, 0.0, 3.0, zero_floor=0.05),
    "apical_gain_threshold": ScalarNoiseSpec("add", 0.08, -1.0, 1.0, zero_floor=0.04),
}

INIT_KEYS = ("w_ff_init", "w_fb_init", "w_lat_init", "w_pv_lat_init", "W_pv_init")
LEARNING_RATE_KEYS = frozenset(("lr_ff", "lr_fb", "lr_lat", "lr_pv"))

DATA_LIKE_TRANSITION_WEIGHTS = {
    "un_un": 0.30,
    "un_FB": 0.18,
    "un_novel_FF": 0.07,
    "FF_un": 0.14,
    "FF_FB_broad": 0.06,
    "FF_FB_broad_novel": 0.04,
    "FF_FB_narrow_familiar": 0.07,
    "FF_FB_narrow_familiar_novel": 0.05,
    "FF_FB_narrow_novel": 0.04,
    "FB_FB": 0.05,
}


def _json_default(value: Any) -> str:
    return repr(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _clip(value: float, lower: float | None, upper: float | None) -> float:
    if lower is not None:
        value = max(lower, value)
    if upper is not None:
        value = min(upper, value)
    return float(value)


def _sample_scalar(
    value: float,
    spec: ScalarNoiseSpec,
    rng: np.random.Generator,
    *,
    scalar_noise_multiplier: float,
) -> float:
    value = float(value)
    if spec.mode == "log" and value > 0:
        sampled = value * float(np.exp(rng.normal(0.0, spec.scale * scalar_noise_multiplier)))
    else:
        sd = max(abs(value) * spec.scale, spec.zero_floor) * scalar_noise_multiplier
        sampled = value + float(rng.normal(0.0, sd))
    return _clip(sampled, spec.lower, spec.upper)


def _sample_init_dict(
    init_dict: dict[str, Any],
    *,
    rng: np.random.Generator,
    weight_noise_rel: float,
    weight_noise_floor: float,
    keep_init_sigma: bool,
) -> dict[str, Any]:
    sampled = copy.deepcopy(init_dict)
    mu = np.asarray(init_dict["mu"], dtype=float)
    mu_scale = np.maximum(np.abs(mu) * weight_noise_rel, weight_noise_floor)
    sampled_mu = np.clip(mu + rng.normal(0.0, mu_scale, size=mu.shape), 0.0, None)
    sampled["mu"] = sampled_mu.tolist()

    if not keep_init_sigma:
        sampled["sigma"] = 0.0

    return sampled


def sample_config_around_canonical(
    canonical_name: str,
    canonical_config: dict[str, Any],
    *,
    sample_idx: int,
    global_idx: int,
    seed: int,
    rng: np.random.Generator,
    weight_noise_rel: float,
    weight_noise_floor: float,
    scalar_noise_multiplier: float,
    freeze_learning_rates: bool,
    initial_weights_only: bool,
    keep_init_sigma: bool,
) -> dict[str, Any]:
    sampled = copy.deepcopy(canonical_config)

    for key in INIT_KEYS:
        if key in sampled:
            sampled[key] = _sample_init_dict(
                sampled[key],
                rng=rng,
                weight_noise_rel=weight_noise_rel,
                weight_noise_floor=weight_noise_floor,
                keep_init_sigma=keep_init_sigma,
            )

    for key, spec in POSITIVE_SCALAR_SPECS.items():
        if initial_weights_only:
            continue
        if freeze_learning_rates and key in LEARNING_RATE_KEYS:
            continue
        if key in sampled and _is_number(sampled[key]):
            sampled[key] = _sample_scalar(
                float(sampled[key]),
                spec,
                rng,
                scalar_noise_multiplier=scalar_noise_multiplier,
            )

    for key, spec in ADDITIVE_SCALAR_SPECS.items():
        if initial_weights_only:
            continue
        if key in sampled and _is_number(sampled[key]):
            sampled[key] = _sample_scalar(
                float(sampled[key]),
                spec,
                rng,
                scalar_noise_multiplier=scalar_noise_multiplier,
            )

    sampled["seed"] = int(seed)
    sampled["_canonical_transition"] = canonical_name
    sampled["_sample_idx"] = int(sample_idx)
    sampled["_sample_global_idx"] = int(global_idx)
    return sampled


def sample_configs(
    canonical_configs: dict[str, dict[str, Any]],
    *,
    samples_per_transition: int,
    seed: int,
    weight_noise_rel: float,
    weight_noise_floor: float,
    scalar_noise_multiplier: float,
    freeze_learning_rates: bool,
    initial_weights_only: bool,
    keep_init_sigma: bool,
    transition_sampling: str,
) -> list[dict[str, Any]]:
    transition_counts = _transition_sample_counts(
        canonical_configs,
        samples_per_transition=samples_per_transition,
        transition_sampling=transition_sampling,
    )
    child_rngs = np.random.SeedSequence(seed).spawn(sum(transition_counts.values()))
    sampled_configs: list[dict[str, Any]] = []
    global_idx = 1

    for canonical_name, canonical_config in canonical_configs.items():
        for sample_idx in range(1, transition_counts[canonical_name] + 1):
            rng = np.random.default_rng(child_rngs[global_idx - 1])
            sampled_configs.append(
                sample_config_around_canonical(
                    canonical_name,
                    canonical_config,
                    sample_idx=sample_idx,
                    global_idx=global_idx,
                    seed=seed + global_idx,
                    rng=rng,
                    weight_noise_rel=weight_noise_rel,
                    weight_noise_floor=weight_noise_floor,
                    scalar_noise_multiplier=scalar_noise_multiplier,
                    freeze_learning_rates=freeze_learning_rates,
                    initial_weights_only=initial_weights_only,
                    keep_init_sigma=keep_init_sigma,
                )
            )
            global_idx += 1

    return sampled_configs


def _transition_sample_counts(
    canonical_configs: dict[str, dict[str, Any]],
    *,
    samples_per_transition: int,
    transition_sampling: str,
) -> dict[str, int]:
    if transition_sampling == "equal":
        return {name: samples_per_transition for name in canonical_configs}
    if transition_sampling != "data-like":
        raise ValueError("transition_sampling must be 'data-like' or 'equal'.")

    total = len(canonical_configs) * samples_per_transition
    weights = np.asarray(
        [DATA_LIKE_TRANSITION_WEIGHTS.get(name, 0.0) for name in canonical_configs],
        dtype=float,
    )
    if np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError("DATA_LIKE_TRANSITION_WEIGHTS must contain nonnegative values with a positive sum.")
    weights = weights / weights.sum()
    raw_counts = weights * total
    counts = np.floor(raw_counts).astype(int)
    remainder = total - int(counts.sum())
    if remainder > 0:
        order = np.argsort(raw_counts - counts)[::-1]
        counts[order[:remainder]] += 1
    return {name: int(count) for name, count in zip(canonical_configs, counts)}


def canonical_configs_as_samples(
    canonical_configs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    sampled_configs: list[dict[str, Any]] = []
    for global_idx, (canonical_name, canonical_config) in enumerate(canonical_configs.items(), start=1):
        sampled = copy.deepcopy(canonical_config)
        sampled["_canonical_transition"] = canonical_name
        sampled["_sample_idx"] = 1
        sampled["_sample_global_idx"] = global_idx
        sampled_configs.append(sampled)
    return sampled_configs


def _sampling_mode_name(
    *,
    canonical_only: bool,
    initial_weights_only: bool,
    freeze_learning_rates: bool,
) -> str:
    if canonical_only:
        return "canonical"
    if initial_weights_only:
        return "initial_weights_only"
    if freeze_learning_rates:
        return "fixed_lrs"
    return "vary_lrs"


def _model_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _response_from_frame(
    frame: DataFrame,
    *,
    n_steps_per_phase: int,
    response_tail_fraction: float,
    baseline_mean: float | None = None,
    baseline_std: float | None = None,
) -> float:
    stim_start = 3 * n_steps_per_phase // 4
    stim_len = n_steps_per_phase - stim_start
    tail_start = stim_start + int(round((1.0 - response_tail_fraction) * stim_len))
    trial_step = frame["step"].to_numpy(dtype=int) % n_steps_per_phase
    mask = trial_step >= tail_start
    if not np.any(mask):
        raise ValueError("Response window selected no samples.")
    y = frame.loc[mask, "y"].to_numpy(dtype=float)
    if baseline_mean is None or baseline_std is None:
        return float(np.nanmean(y))
    scale = baseline_std if np.isfinite(baseline_std) and baseline_std > 1e-12 else 1.0
    return float(np.nanmean((y - baseline_mean) / scale))


def _baseline_stats_from_frames(
    frames: list[DataFrame],
    *,
    n_steps_per_phase: int,
) -> dict[str, float | int]:
    stim_start = 3 * n_steps_per_phase // 4
    baseline_chunks: list[np.ndarray] = []
    for frame in frames:
        trial_step = frame["step"].to_numpy(dtype=int) % n_steps_per_phase
        baseline = frame.loc[trial_step < stim_start, "y"].to_numpy(dtype=float)
        if baseline.size:
            baseline_chunks.append(baseline)

    if not baseline_chunks:
        return {"baseline_mean": 0.0, "baseline_std": 1.0, "baseline_n": 0}

    values = np.concatenate(baseline_chunks)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"baseline_mean": np.nan, "baseline_std": np.nan, "baseline_n": 0}

    std = float(np.nanstd(finite, ddof=1)) if finite.size > 1 else 1.0
    return {
        "baseline_mean": float(np.nanmean(finite)),
        "baseline_std": std if std > 1e-12 else 1.0,
        "baseline_n": int(finite.size),
    }


def _run_full_and_occlusion_phase(
    model: CCNeuron,
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    phase: str,
    n_steps_per_phase: int,
    response_tail_fraction: float,
    zscore_responses: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trace_frames: list[tuple[str, str, DataFrame]] = []

    for condition, (x_full, c_full) in stimuli.items():
        for trace_name, x_phase in (("full", x_full), ("occlusion", torch.zeros_like(x_full))):
            frame = run_experimental_phase(
                model,
                x_phase,
                c_full,
                condition_name=f"{trace_name}_{condition}_{phase}",
                update=False,
            )
            trace_frames.append((condition, trace_name, frame))

    baseline_stats = _baseline_stats_from_frames(
        [frame for _, _, frame in trace_frames],
        n_steps_per_phase=n_steps_per_phase,
    )
    baseline_mean = float(baseline_stats["baseline_mean"])
    baseline_std = float(baseline_stats["baseline_std"])

    for condition, trace_name, frame in trace_frames:
        raw_response = _response_from_frame(
            frame,
            n_steps_per_phase=n_steps_per_phase,
            response_tail_fraction=response_tail_fraction,
        )
        if zscore_responses:
            response = _response_from_frame(
                frame,
                n_steps_per_phase=n_steps_per_phase,
                response_tail_fraction=response_tail_fraction,
                baseline_mean=baseline_mean,
                baseline_std=baseline_std,
            )
        else:
            response = raw_response
        rows.append(
            {
                "condition": condition,
                "phase": phase,
                "stage": STAGE_LABELS[phase],
                "trace": trace_name,
                "image_type": TRACE_TO_IMAGE_TYPE[trace_name],
                "response": response,
                "raw_response": raw_response,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "baseline_n": int(baseline_stats["baseline_n"]),
            }
        )

    return rows


def run_sampled_config(
    config: dict[str, Any],
    *,
    n_steps_per_phase: int,
    response_tail_fraction: float,
    zscore_responses: bool,
    test_stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    training_stimuli: tuple[torch.Tensor, torch.Tensor],
) -> DataFrame:
    model = CCNeuron(**_model_kwargs(config))
    rows = _run_full_and_occlusion_phase(
        model,
        test_stimuli,
        phase="naive",
        n_steps_per_phase=n_steps_per_phase,
        response_tail_fraction=response_tail_fraction,
        zscore_responses=zscore_responses,
    )

    run_experimental_phase(
        model,
        training_stimuli[0],
        training_stimuli[1],
        condition_name="full_familiar_training",
        update=True,
    )

    rows.extend(
        _run_full_and_occlusion_phase(
            model,
            test_stimuli,
            phase="expert",
            n_steps_per_phase=n_steps_per_phase,
            response_tail_fraction=response_tail_fraction,
            zscore_responses=zscore_responses,
        )
    )

    return pd.DataFrame(rows).assign(
        transition=config["_canonical_transition"],
        sample_idx=config["_sample_idx"],
        sample_global_idx=config["_sample_global_idx"],
        seed=config["seed"],
        experiment_series=PRIMARY_EXPERIMENT_SERIES,
    )


def _to_transition_table(response_df: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []

    for row in response_df.itertuples(index=False):
        image_group, image_idx_original, image_idx_within_group = CONDITION_TO_IMAGE_INFO[row.condition]
        rows.append(
            {
                "transition": row.transition,
                "image_group": image_group,
                "image_idx_original": image_idx_original,
                "image_idx_within_group": image_idx_within_group,
                "neuron_idx": int(row.sample_global_idx),
                "image_type": row.image_type,
                "stage": row.stage,
                "response": float(row.response),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "transition",
            "image_group",
            "image_idx_original",
            "image_idx_within_group",
            "neuron_idx",
            "image_type",
            "stage",
            "response",
        ],
    )


def _to_wide_transition_table(transition_table: DataFrame) -> DataFrame:
    stage_order = transition_table["stage"].drop_duplicates().tolist()
    wide = (
        transition_table.pivot_table(
            index=[
                "transition",
                "image_group",
                "image_idx_original",
                "image_idx_within_group",
                "neuron_idx",
                "stage",
            ],
            columns="image_type",
            values="response",
            aggfunc="mean",
        )
        .reset_index()
        .rename(columns={"Full": "NO", "Occl": "O"})
    )
    wide["stage"] = pd.Categorical(wide["stage"], categories=stage_order, ordered=True)
    return wide.sort_values(
        ["transition", "image_group", "image_idx_original", "neuron_idx", "stage"]
    ).reset_index(drop=True)


def _flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {
        "transition": config["_canonical_transition"],
        "sample_idx": config["_sample_idx"],
        "sample_global_idx": config["_sample_global_idx"],
        "seed": config["seed"],
    }

    for key, value in config.items():
        if key.startswith("_") or key == "seed":
            continue
        if key in INIT_KEYS and isinstance(value, dict):
            mu = np.asarray(value.get("mu", []), dtype=float).reshape(-1)
            for idx, mu_value in enumerate(mu):
                flat[f"{key}.mu_{idx}"] = float(mu_value)
            sigma = value.get("sigma")
            if _is_number(sigma):
                flat[f"{key}.sigma"] = float(sigma)
            continue
        if _is_number(value):
            flat[key] = float(value)
        elif isinstance(value, tuple) and all(isinstance(item, bool) for item in value):
            for idx, item in enumerate(value):
                flat[f"{key}_{idx}"] = bool(item)
        elif isinstance(value, str):
            flat[key] = value

    return flat


def _build_summary(
    transition_table: DataFrame,
    *,
    image_group: str,
    threshold: float,
) -> DataFrame:
    return th.build_mean_summary(
        transition_table,
        image_group=image_group,
        pre_stage="Naive",
        target_stage="Expert",
        threshold=threshold,
    )


def _robust_response_limits(
    *frames: DataFrame,
    percentile: float,
    pad: float = 0.4,
) -> list[float]:
    if percentile >= 100.0:
        return th.compute_response_limits(*frames, pad=pad)

    values = np.concatenate(
        [
            frame[["NO_Pre", "O_Pre", "NO_Target", "O_Target"]].to_numpy(dtype=float).reshape(-1)
            for frame in frames
        ]
    )
    lower = float(np.nanpercentile(values, max(0.0, 100.0 - percentile)))
    upper = float(np.nanpercentile(values, min(100.0, percentile)))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return th.compute_response_limits(*frames, pad=pad)
    return [lower - pad, upper + pad]


def _robust_shift_limits(
    *frames: DataFrame,
    percentile: float,
    pad_ratio: float = 0.12,
    fallback_extent: float = 0.5,
) -> list[float]:
    if percentile >= 100.0:
        return th.compute_shift_limits(*frames, pad_ratio=pad_ratio, fallback_extent=fallback_extent)

    values = np.concatenate(
        [
            frame[["dNO", "dO"]].to_numpy(dtype=float).reshape(-1)
            for frame in frames
        ]
    )
    extent = float(np.nanpercentile(np.abs(values), min(100.0, percentile)))
    if not np.isfinite(extent) or extent <= 0.0:
        extent = fallback_extent
    else:
        extent *= 1.0 + pad_ratio
    return [-extent, extent]


def _summaries_outside_limits(
    summaries: dict[str, DataFrame],
    *,
    response_lims: list[float],
    shift_lims: list[float],
) -> DataFrame:
    frames: list[DataFrame] = []
    for image_group, summary in summaries.items():
        response_outside = (
            (summary[["NO_Pre", "O_Pre", "NO_Target", "O_Target"]] < response_lims[0])
            | (summary[["NO_Pre", "O_Pre", "NO_Target", "O_Target"]] > response_lims[1])
        ).any(axis=1)
        shift_outside = (
            (summary[["dNO", "dO"]] < shift_lims[0])
            | (summary[["dNO", "dO"]] > shift_lims[1])
        ).any(axis=1)
        outside = summary.loc[response_outside | shift_outside].copy()
        if outside.empty:
            continue
        outside["image_group"] = image_group
        outside["outside_response_lims"] = response_outside.loc[outside.index].to_numpy(dtype=bool)
        outside["outside_shift_lims"] = shift_outside.loc[outside.index].to_numpy(dtype=bool)
        frames.append(outside)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _save_summary_figure(
    summary: DataFrame,
    *,
    title: str,
    out_path: Path,
    response_lims: list[float] | None,
    shift_lims: list[float] | None,
    export_panels: bool,
) -> None:
    fig = th.plot_mean_transition_summary(
        summary,
        title=title,
        start_label="Naive",
        end_label="Expert",
        response_lims=response_lims,
        shift_lims=shift_lims,
        style=PLOT_STYLE,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    if export_panels:
        th.export_figure_panels(fig, out_path.parent / f"{out_path.stem}_panels", out_path.stem)
    plt.close(fig)
    th.save_rotated_sector_unit_legend(
        summary,
        out_path.with_name(f"{out_path.stem}_sector_legend.png"),
        title=None,
    )


def _save_sector_fraction_plot(fraction_df: DataFrame, out_path: Path) -> None:
    scopes = fraction_df["scope"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(scopes), figsize=(5.0 * len(scopes), 4.2), sharey=True, squeeze=False)
    max_fraction = max(0.35, float(fraction_df["Fraction"].max()) + 0.05)

    for ax, scope in zip(axes.reshape(-1), scopes):
        table = fraction_df.loc[fraction_df["scope"] == scope].copy()
        colors = [th.ROTATED_SECTOR_PALETTE[sector] for sector in table["RotatedSector"]]
        ax.bar(table["RotatedSector"], table["Fraction"], color=colors, alpha=0.9)
        ax.axhline(0.25, color="0.3", linestyle="--", linewidth=1)
        ax.set_title(scope)
        ax.set_ylim(0.0, max_fraction)
        ax.tick_params(axis="x", rotation=25)
        ax.set_xlabel("Rotated sector")

    axes[0, 0].set_ylabel("Fraction of transitions")
    fig.suptitle("Model-scatter transition fractions by rotated sector")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def save_plots(
    transition_table: DataFrame,
    *,
    output_dir: Path,
    transition_order: list[str],
    threshold: float,
    limit_percentile: float,
    plot_by_transition: bool,
    export_panels: bool,
) -> None:
    figures_dir = output_dir / "figures"
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    wide_table = _to_wide_transition_table(transition_table)
    aggregate_summaries = {
        image_group: _build_summary(wide_table, image_group=image_group, threshold=threshold)
        for image_group in ("familiar", "novel")
    }
    response_lims = _robust_response_limits(*aggregate_summaries.values(), percentile=limit_percentile)
    shift_lims = _robust_shift_limits(*aggregate_summaries.values(), percentile=limit_percentile)
    outliers = _summaries_outside_limits(
        aggregate_summaries,
        response_lims=response_lims,
        shift_lims=shift_lims,
    )
    if not outliers.empty:
        outliers.to_csv(summaries_dir / "aggregate_points_outside_plot_limits.csv", index=False)

    fraction_frames: list[DataFrame] = []
    for image_group, summary in aggregate_summaries.items():
        summary.to_csv(summaries_dir / f"aggregate_{image_group}_summary.csv", index=False)
        fraction_frames.append(
            th.sector_fraction_table(summary).assign(scope=f"aggregate {image_group}", transition="all", image_group=image_group)
        )
        _save_summary_figure(
            summary,
            title=f"Model scatter - all transitions - {image_group}",
            out_path=figures_dir / f"aggregate_{image_group}_summary.png",
            response_lims=response_lims,
            shift_lims=shift_lims,
            export_panels=export_panels,
        )

    if plot_by_transition:
        for transition in transition_order:
            subset = wide_table.loc[wide_table["transition"] == transition].copy()
            if subset.empty:
                continue
            for image_group in ("familiar", "novel"):
                summary = _build_summary(subset, image_group=image_group, threshold=threshold)
                summary.to_csv(summaries_dir / f"{transition}_{image_group}_summary.csv", index=False)
                fraction_frames.append(
                    th.sector_fraction_table(summary).assign(
                        scope=f"{transition} {image_group}",
                        transition=transition,
                        image_group=image_group,
                    )
                )
                _save_summary_figure(
                    summary,
                    title=f"Model scatter - {transition} - {image_group}",
                    out_path=figures_dir / "by_transition" / f"{transition}_{image_group}_summary.png",
                    response_lims=response_lims,
                    shift_lims=shift_lims,
                    export_panels=export_panels,
                )

    fraction_df = pd.concat(fraction_frames, ignore_index=True)
    fraction_df.to_csv(summaries_dir / "sector_fractions.csv", index=False)
    _save_sector_fraction_plot(
        fraction_df.loc[fraction_df["transition"] == "all"].copy(),
        figures_dir / "aggregate_sector_fractions.png",
    )


def run_model_scatter(
    *,
    output_dir: Path,
    samples_per_transition: int,
    n_steps_per_phase: int,
    seed: int,
    n_jobs: int,
    weight_noise_rel: float,
    weight_noise_floor: float,
    scalar_noise_multiplier: float,
    keep_init_sigma: bool,
    freeze_learning_rates: bool,
    initial_weights_only: bool,
    canonical_only: bool,
    transition_sampling: str,
    zscore_responses: bool,
    response_tail_fraction: float,
    threshold: float,
    limit_percentile: float,
    plot_by_transition: bool,
    export_panels: bool,
) -> None:
    if not 0.0 < response_tail_fraction <= 1.0:
        raise ValueError("response_tail_fraction must be in (0, 1].")

    output_dir.mkdir(parents=True, exist_ok=True)
    transition_order = list(minimal_configs3)
    if canonical_only:
        sampled_configs = canonical_configs_as_samples(minimal_configs3)
        transition_counts = {name: 1 for name in transition_order}
    else:
        sampled_configs = sample_configs(
            minimal_configs3,
            samples_per_transition=samples_per_transition,
            seed=seed,
            weight_noise_rel=weight_noise_rel,
            weight_noise_floor=weight_noise_floor,
            scalar_noise_multiplier=scalar_noise_multiplier,
            freeze_learning_rates=freeze_learning_rates,
            initial_weights_only=initial_weights_only,
            keep_init_sigma=keep_init_sigma,
            transition_sampling=transition_sampling,
        )
        transition_counts = _transition_sample_counts(
            minimal_configs3,
            samples_per_transition=samples_per_transition,
            transition_sampling=transition_sampling,
        )

    torch.manual_seed(seed)
    test_stimuli = _build_test_stimuli(n_steps_per_phase=n_steps_per_phase, n_trials=10)
    training_stimuli = _build_training_stimuli(n_steps_per_phase=n_steps_per_phase, n_trials=5)

    response_frames = Parallel(n_jobs=n_jobs, verbose=10 if n_jobs != 1 else 0)(
        delayed(run_sampled_config)(
            config,
            n_steps_per_phase=n_steps_per_phase,
            response_tail_fraction=response_tail_fraction,
            zscore_responses=zscore_responses,
            test_stimuli=test_stimuli,
            training_stimuli=training_stimuli,
        )
        for config in sampled_configs
    )

    response_df = pd.concat(response_frames, ignore_index=True)
    transition_table = _to_transition_table(response_df)
    config_df = pd.DataFrame(_flatten_config(config) for config in sampled_configs)
    invalid_responses = transition_table.loc[~np.isfinite(transition_table["response"])].copy()

    response_df.to_csv(output_dir / "sample_responses.csv", index=False)
    transition_table.to_csv(output_dir / "transition_table.csv", index=False)
    config_df.to_csv(output_dir / "sampled_config_parameters.csv", index=False)
    if not invalid_responses.empty:
        invalid_responses.to_csv(output_dir / "invalid_responses.csv", index=False)
    (output_dir / "sampled_configs.json").write_text(json.dumps(sampled_configs, indent=2, default=_json_default))

    metadata = {
        "samples_per_transition": samples_per_transition,
        "n_transitions": len(transition_order),
        "n_samples_total": len(sampled_configs),
        "n_invalid_response_rows": int(len(invalid_responses)),
        "n_invalid_sample_ids": int(invalid_responses["neuron_idx"].nunique()) if not invalid_responses.empty else 0,
        "n_steps_per_phase": n_steps_per_phase,
        "seed": seed,
        "weight_noise_rel": weight_noise_rel,
        "weight_noise_floor": weight_noise_floor,
        "scalar_noise_multiplier": scalar_noise_multiplier,
        "freeze_learning_rates": freeze_learning_rates,
        "initial_weights_only": initial_weights_only,
        "sampling_mode": _sampling_mode_name(
            canonical_only=canonical_only,
            initial_weights_only=initial_weights_only,
            freeze_learning_rates=freeze_learning_rates,
        ),
        "varied_learning_rates": not freeze_learning_rates and not initial_weights_only and not canonical_only,
        "varied_scalar_hyperparameters": not initial_weights_only and not canonical_only,
        "keep_init_sigma": keep_init_sigma,
        "canonical_only": canonical_only,
        "transition_sampling": transition_sampling,
        "transition_sample_counts": transition_counts,
        "response_units": "zscore" if zscore_responses else "raw",
        "zscore_responses": zscore_responses,
        "response_tail_fraction": response_tail_fraction,
        "sector_threshold": threshold,
        "limit_percentile": limit_percentile,
        "transitions": transition_order,
        "stimulus_specs": STIMULUS_SPECS,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default))

    save_plots(
        transition_table,
        output_dir=output_dir,
        transition_order=transition_order,
        threshold=threshold,
        limit_percentile=limit_percentile,
        plot_by_transition=plot_by_transition,
        export_panels=export_panels,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample noisy minimal2 configs around each canonical transition, run the model, "
            "and make data-analysis-style NO/O scatter plots."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples-per-transition", type=int, default=96)
    parser.add_argument("--n-steps-per-phase", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--weight-noise-rel", type=float, default=0.55)
    parser.add_argument("--weight-noise-floor", type=float, default=0.07)
    parser.add_argument("--scalar-noise-multiplier", type=float, default=1.75)
    parser.add_argument(
        "--freeze-learning-rates",
        action="store_true",
        help="Keep lr_ff, lr_fb, lr_lat, and lr_pv fixed at canonical values while sampling other parameters.",
    )
    parser.add_argument(
        "--initial-weights-only",
        action="store_true",
        help="Only sample initial weight means; keep all scalar hyperparameters fixed at canonical values.",
    )
    parser.add_argument("--keep-init-sigma", action="store_true")
    parser.add_argument(
        "--canonical-only",
        action="store_true",
        help="Run exactly the canonical minimal2 configs without sampled perturbations.",
    )
    parser.add_argument(
        "--transition-sampling",
        choices=("data-like", "equal"),
        default="data-like",
        help="How to allocate noisy samples across canonical transition configs.",
    )
    parser.add_argument(
        "--raw-responses",
        action="store_true",
        help="Use raw model activation responses instead of baseline z-scored responses.",
    )
    parser.add_argument("--response-tail-fraction", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument(
        "--limit-percentile",
        type=float,
        default=99.0,
        help="Percentile used for plot axis limits; set to 100 for full data extents.",
    )
    parser.add_argument("--skip-by-transition", action="store_true")
    parser.add_argument("--export-panels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_model_scatter(
        output_dir=args.output_dir,
        samples_per_transition=args.samples_per_transition,
        n_steps_per_phase=args.n_steps_per_phase,
        seed=args.seed,
        n_jobs=args.n_jobs,
        weight_noise_rel=args.weight_noise_rel,
        weight_noise_floor=args.weight_noise_floor,
        scalar_noise_multiplier=args.scalar_noise_multiplier,
        freeze_learning_rates=args.freeze_learning_rates,
        initial_weights_only=args.initial_weights_only,
        keep_init_sigma=args.keep_init_sigma,
        canonical_only=args.canonical_only,
        transition_sampling=args.transition_sampling,
        zscore_responses=not args.raw_responses,
        response_tail_fraction=args.response_tail_fraction,
        threshold=args.threshold,
        limit_percentile=args.limit_percentile,
        plot_by_transition=not args.skip_by_transition,
        export_panels=args.export_panels,
    )


if __name__ == "__main__":
    main()
