from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm.auto import tqdm

from context_contrasting.minimal.config import broad, minimal_configs
from context_contrasting.minimal.experiment import design_experimental_phase, run_experiment
from context_contrasting.minimal.minimal import CCNeuron
from context_contrasting.minimal.transition_types import (
    num_state_to_category,
    single_state_scalar,
    transitions,
)

PRIMARY_PHASES = ("naive", "expert")
ALL_PHASES = ("naive", "expert", "expert2")
SUMMARY_KEYS = (
    "full_familiar_naive",
    "occlusion_familiar_naive",
    "full_novel_naive",
    "occlusion_novel_naive",
    "full_familiar_expert",
    "occlusion_familiar_expert",
    "full_novel_expert",
    "occlusion_novel_expert",
    "full_familiar_expert2",
    "occlusion_familiar_expert2",
    "full_novel_expert2",
    "occlusion_novel_expert2",
)
WEIGHT_PARAM_ORDER = (
    "w_ff_0",
    "w_ff_1",
    "w_fb_0",
    "w_fb_1",
    "w_lat_0",
    "w_lat_1",
    "w_pv_lat_0",
    "w_pv_lat_1",
)
LR_PARAM_ORDER = ("lr_ff", "lr_fb", "lr_lat", "lr_pv")
PARAM_ORDER = WEIGHT_PARAM_ORDER + LR_PARAM_ORDER
STANDARD_TARGETS = (
    "un_FB",
    "FF_FB_broad",
    "FF_FB_narrow_familiar",
    "FF_FB_narrow_novel",
    "FB_FB",
)
SPECIAL_TARGETS = ("FF_un",)
SEARCH_TARGETS = STANDARD_TARGETS + SPECIAL_TARGETS
REFERENCE_ONLY_TARGETS = ("un_un",)
TARGET_ORDER = REFERENCE_ONLY_TARGETS + SEARCH_TARGETS
TARGET_COLORS = {
    "un_un": "#7f7f7f",
    "un_FB": "#4c78a8",
    "FF_un": "#f58518",
    "FF_FB_broad": "#54a24b",
    "FF_FB_narrow_familiar": "#e45756",
    "FF_FB_narrow_novel": "#72b7b2",
    "FB_FB": "#b279a2",
}
DEFAULT_ACTIVITY_THRESHOLD = 0.025
SIGNATURE_KEYS = (
    *SUMMARY_KEYS,
    "familiar_naive_state",
    "familiar_expert_state",
    "familiar_expert2_state",
    "novel_naive_state",
    "novel_expert_state",
    "novel_expert2_state",
    "familiar_state_delta",
    "familiar_continuation_delta",
    "novel_state_delta",
    "novel_continuation_delta",
    "delta_full_familiar_expert",
    "delta_occlusion_familiar_expert",
    "delta_full_novel_expert",
    "delta_occlusion_novel_expert",
    "delta_full_familiar_expert2",
    "delta_occlusion_familiar_expert2",
    "delta_full_novel_expert2",
    "delta_occlusion_novel_expert2",
)
PHASE_CATEGORY_KEYS = (
    "familiar_naive_category",
    "familiar_expert_category",
    "familiar_expert2_category",
    "novel_naive_category",
    "novel_expert_category",
    "novel_expert2_category",
)
DEFAULT_PARAM_RANGES = {
    "w_ff_0": (1e-4, 2.0),
    "w_ff_1": (1e-4, 2.0),
    "w_fb_0": (1e-4, 2.0),
    "w_fb_1": (1e-4, 2.0),
    "w_lat_0": (1e-4, 2.0),
    "w_lat_1": (1e-4, 2.0),
    "w_pv_lat_0": (1e-4, 2.0),
    "w_pv_lat_1": (1e-4, 2.0),
    "lr_ff": (1e-4, 8e-2),
    "lr_fb": (5e-5, 2e-2),
    "lr_lat": (5e-5, 5e-2),
    "lr_pv": (1e-4, 2e-2),
}


@dataclass(frozen=True)
class LandscapeSettings:
    n_steps_per_phase: int = 80
    n_trials: int = 6
    continuation_trials_multiplier: int = 2
    tail_window: int = 60
    intertrial_sigma: float = 0.05
    global_samples_standard: int = 1800
    global_samples_ff_un: int = 1400
    local_samples_per_target: int = 160
    local_sigma: float = 0.24
    activity_threshold: float = DEFAULT_ACTIVITY_THRESHOLD
    fit_scale: float = DEFAULT_ACTIVITY_THRESHOLD
    prototype_mismatch_penalty: float = 0.35
    stable_margin_threshold: float = 0.08
    stable_min_points: int = 40
    n_jobs: int = -1
    random_seed: int = 17
    backtest_full_steps: int = 100
    backtest_tail_window: int = 80
    backtest_seeds: tuple[int, ...] = (11, 23, 37)
    quantile_core_candidates: tuple[tuple[float, float], ...] = (
        (0.05, 0.95),
        (0.10, 0.90),
        (0.15, 0.85),
        (0.20, 0.80),
        (0.25, 0.75),
    )
    boundary_pairs_per_class_pair: int = 2
    boundary_interp_points: int = 7
    tsne_perplexity: float = 35.0


def _base_search_config(receives_context: tuple[bool, bool]) -> dict[str, Any]:
    config = deepcopy(broad)
    config["receives_context"] = receives_context
    config.setdefault(
        "w_pv_lat_init",
        {
            "mu": list(config["w_lat_init"]["mu"]),
            "sigma": config["w_lat_init"]["sigma"],
        },
    )
    return config


def _target_receives_context(target_name: str) -> tuple[bool, bool]:
    return (False, False) if target_name == "FF_un" else (True, True)


def _reference_config_for_target(target_name: str) -> dict[str, Any]:
    config = deepcopy(minimal_configs[target_name])
    config["receives_context"] = _target_receives_context(target_name)
    return config


def _param_dict_from_config(config: dict[str, Any]) -> dict[str, float]:
    return {
        "w_ff_0": float(config["w_ff_init"]["mu"][0]),
        "w_ff_1": float(config["w_ff_init"]["mu"][1]),
        "w_fb_0": float(config["w_fb_init"]["mu"][0]),
        "w_fb_1": float(config["w_fb_init"]["mu"][1]),
        "w_lat_0": float(config["w_lat_init"]["mu"][0]),
        "w_lat_1": float(config["w_lat_init"]["mu"][1]),
        "w_pv_lat_0": float(config["w_pv_lat_init"]["mu"][0]),
        "w_pv_lat_1": float(config["w_pv_lat_init"]["mu"][1]),
        "lr_ff": float(config["lr_ff"]),
        "lr_fb": float(config["lr_fb"]),
        "lr_lat": float(config["lr_lat"]),
        "lr_pv": float(config["lr_pv"]),
    }


def _config_from_params(
    params: dict[str, float],
    *,
    base_config: dict[str, Any],
    seed: int | None = None,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    config["w_ff_init"] = {"mu": [params["w_ff_0"], params["w_ff_1"]], "sigma": 0.0}
    config["w_fb_init"] = {"mu": [params["w_fb_0"], params["w_fb_1"]], "sigma": 0.0}
    config["w_lat_init"] = {"mu": [params["w_lat_0"], params["w_lat_1"]], "sigma": 0.0}
    config["w_pv_lat_init"] = {
        "mu": [params["w_pv_lat_0"], params["w_pv_lat_1"]],
        "sigma": 0.0,
    }
    config["lr_ff"] = float(params["lr_ff"])
    config["lr_fb"] = float(params["lr_fb"])
    config["lr_lat"] = float(params["lr_lat"])
    config["lr_pv"] = float(params["lr_pv"])
    if seed is not None:
        config["seed"] = seed
    return config


def _run_phase_mean_response(
    model: CCNeuron,
    X: torch.Tensor,
    C: torch.Tensor,
    *,
    update: bool,
    reset_rates: bool,
    tail_window: int,
) -> float:
    if reset_rates:
        model._reset_state()

    responses: list[float] = []
    for step in range(X.shape[0]):
        x, y_t, y_next, p, c = model(X[step], C[step])
        if update:
            model.update(x, y_t, y_next, p, c)
        responses.append(float(y_next))

    tail = min(tail_window, len(responses))
    return float(sum(responses[-tail:]) / tail)


def simulate_extended_summary(
    model_config: dict[str, Any],
    *,
    n_steps_per_phase: int,
    n_trials: int,
    continuation_trials_multiplier: int,
    tail_window: int,
    intertrial_sigma: float,
) -> dict[str, float]:
    model = CCNeuron(**{key: value for key, value in model_config.items() if not key.startswith("_")})

    X1, C1 = design_experimental_phase(
        input_mean=[1, 0],
        input_var=0.05,
        context_mean=[1, 0],
        context_var=0.05,
        n_steps=n_steps_per_phase,
        n_trials=n_trials,
        intertrial_sigma=intertrial_sigma,
    )
    X2, C2 = design_experimental_phase(
        input_mean=[0, 1],
        input_var=0.05,
        context_mean=[0, 1],
        context_var=0.05,
        n_steps=n_steps_per_phase,
        n_trials=n_trials,
        intertrial_sigma=intertrial_sigma,
    )
    X1_long, C1_long = design_experimental_phase(
        input_mean=[1, 0],
        input_var=0.05,
        context_mean=[1, 0],
        context_var=0.05,
        n_steps=2 * n_steps_per_phase,
        n_trials=max(n_trials * continuation_trials_multiplier, n_trials),
        intertrial_sigma=intertrial_sigma,
    )
    O = torch.zeros_like(X1)
    O_long = torch.zeros_like(X1_long)

    summary = {
        "full_familiar_naive": _run_phase_mean_response(
            model, X1, C1, update=False, reset_rates=True, tail_window=tail_window
        ),
        "occlusion_familiar_naive": _run_phase_mean_response(
            model, O, C1, update=False, reset_rates=True, tail_window=tail_window
        ),
        "full_novel_naive": _run_phase_mean_response(
            model, X2, C2, update=False, reset_rates=True, tail_window=tail_window
        ),
        "occlusion_novel_naive": _run_phase_mean_response(
            model, O, C2, update=False, reset_rates=True, tail_window=tail_window
        ),
    }

    _run_phase_mean_response(
        model, X1, C1, update=True, reset_rates=True, tail_window=tail_window
    )
    summary.update(
        {
            "full_familiar_expert": _run_phase_mean_response(
                model, X1, C1, update=False, reset_rates=True, tail_window=tail_window
            ),
            "occlusion_familiar_expert": _run_phase_mean_response(
                model, O, C1, update=False, reset_rates=True, tail_window=tail_window
            ),
            "full_novel_expert": _run_phase_mean_response(
                model, X2, C2, update=False, reset_rates=True, tail_window=tail_window
            ),
            "occlusion_novel_expert": _run_phase_mean_response(
                model, O, C2, update=False, reset_rates=True, tail_window=tail_window
            ),
        }
    )

    _run_phase_mean_response(
        model, O_long, C1_long, update=True, reset_rates=True, tail_window=tail_window
    )
    summary.update(
        {
            "full_familiar_expert2": _run_phase_mean_response(
                model, X1, C1, update=False, reset_rates=True, tail_window=tail_window
            ),
            "occlusion_familiar_expert2": _run_phase_mean_response(
                model, O, C1, update=False, reset_rates=True, tail_window=tail_window
            ),
            "full_novel_expert2": _run_phase_mean_response(
                model, X2, C2, update=False, reset_rates=True, tail_window=tail_window
            ),
            "occlusion_novel_expert2": _run_phase_mean_response(
                model, O, C2, update=False, reset_rates=True, tail_window=tail_window
            ),
        }
    )
    return summary


def summarize_experiment_df_extended(
    df: pd.DataFrame,
    *,
    tail_window: int,
    experiment_series: str = "training_familiar",
) -> dict[str, float]:
    if experiment_series == "training_familiar":
        subset = df.loc[df["experiment_series"].eq("training_familiar")]
        return {
            "full_familiar_naive": float(subset.loc[subset["condition"].eq("full_familiar_naive"), "y"].tail(tail_window).mean()),
            "occlusion_familiar_naive": float(subset.loc[subset["condition"].eq("occlusion_familiar_naive"), "y"].tail(tail_window).mean()),
            "full_novel_naive": float(subset.loc[subset["condition"].eq("full_novel_naive"), "y"].tail(tail_window).mean()),
            "occlusion_novel_naive": float(subset.loc[subset["condition"].eq("occlusion_novel_naive"), "y"].tail(tail_window).mean()),
            "full_familiar_expert": float(subset.loc[subset["condition"].eq("full_familiar_expert"), "y"].tail(tail_window).mean()),
            "occlusion_familiar_expert": float(subset.loc[subset["condition"].eq("occlusion_familiar_expert"), "y"].tail(tail_window).mean()),
            "full_novel_expert": float(subset.loc[subset["condition"].eq("full_novel_expert"), "y"].tail(tail_window).mean()),
            "occlusion_novel_expert": float(subset.loc[subset["condition"].eq("occlusion_novel_expert"), "y"].tail(tail_window).mean()),
            "full_familiar_expert2": float("nan"),
            "occlusion_familiar_expert2": float("nan"),
            "full_novel_expert2": float("nan"),
            "occlusion_novel_expert2": float("nan"),
        }

    primary = df.loc[df["experiment_series"].eq("training_familiar")]
    continuation = df.loc[df["experiment_series"].eq("training_occluded_only")]
    return {
        "full_familiar_naive": float(primary.loc[primary["condition"].eq("full_familiar_naive"), "y"].tail(tail_window).mean()),
        "occlusion_familiar_naive": float(primary.loc[primary["condition"].eq("occlusion_familiar_naive"), "y"].tail(tail_window).mean()),
        "full_novel_naive": float(primary.loc[primary["condition"].eq("full_novel_naive"), "y"].tail(tail_window).mean()),
        "occlusion_novel_naive": float(primary.loc[primary["condition"].eq("occlusion_novel_naive"), "y"].tail(tail_window).mean()),
        "full_familiar_expert": float(primary.loc[primary["condition"].eq("full_familiar_expert"), "y"].tail(tail_window).mean()),
        "occlusion_familiar_expert": float(primary.loc[primary["condition"].eq("occlusion_familiar_expert"), "y"].tail(tail_window).mean()),
        "full_novel_expert": float(primary.loc[primary["condition"].eq("full_novel_expert"), "y"].tail(tail_window).mean()),
        "occlusion_novel_expert": float(primary.loc[primary["condition"].eq("occlusion_novel_expert"), "y"].tail(tail_window).mean()),
        "full_familiar_expert2": float(continuation.loc[continuation["condition"].eq("full_familiar_expert2"), "y"].tail(tail_window).mean()),
        "occlusion_familiar_expert2": float(continuation.loc[continuation["condition"].eq("occlusion_familiar_expert2"), "y"].tail(tail_window).mean()),
        "full_novel_expert2": float(continuation.loc[continuation["condition"].eq("full_novel_expert2"), "y"].tail(tail_window).mean()),
        "occlusion_novel_expert2": float(continuation.loc[continuation["condition"].eq("occlusion_novel_expert2"), "y"].tail(tail_window).mean()),
    }


def _phase_state_fields(summary: dict[str, float], image: str, phase: str, threshold: float) -> dict[str, Any]:
    full = float(summary[f"full_{image}_{phase}"])
    occluded = float(summary[f"occlusion_{image}_{phase}"])
    state_scalar, responsive = single_state_scalar(
        full,
        occluded,
        activity_threshold=threshold,
    )
    return {
        f"{image}_{phase}_state": float(state_scalar),
        f"{image}_{phase}_responsive": bool(responsive),
        f"{image}_{phase}_ff_scalar": float(max(state_scalar, 0.0)),
        f"{image}_{phase}_fb_scalar": float(max(-state_scalar, 0.0)),
        f"{image}_{phase}_category": num_state_to_category(state_scalar, responsive),
    }


def signature_from_summary(summary: dict[str, float], *, activity_threshold: float) -> dict[str, Any]:
    features: dict[str, Any] = dict(summary)
    for image in ("familiar", "novel"):
        for phase in ALL_PHASES:
            features.update(_phase_state_fields(summary, image, phase, activity_threshold))

    features["familiar_primary_transition"] = transitions(
        summary["full_familiar_naive"],
        summary["occlusion_familiar_naive"],
        summary["full_familiar_expert"],
        summary["occlusion_familiar_expert"],
        activity_threshold=activity_threshold,
    )
    features["novel_primary_transition"] = transitions(
        summary["full_novel_naive"],
        summary["occlusion_novel_naive"],
        summary["full_novel_expert"],
        summary["occlusion_novel_expert"],
        activity_threshold=activity_threshold,
    )
    features["familiar_state_delta"] = features["familiar_expert_state"] - features["familiar_naive_state"]
    features["familiar_continuation_delta"] = features["familiar_expert2_state"] - features["familiar_expert_state"]
    features["novel_state_delta"] = features["novel_expert_state"] - features["novel_naive_state"]
    features["novel_continuation_delta"] = features["novel_expert2_state"] - features["novel_expert_state"]

    for image in ("familiar", "novel"):
        features[f"delta_full_{image}_expert"] = (
            summary[f"full_{image}_expert"] - summary[f"full_{image}_naive"]
        )
        features[f"delta_occlusion_{image}_expert"] = (
            summary[f"occlusion_{image}_expert"] - summary[f"occlusion_{image}_naive"]
        )
        features[f"delta_full_{image}_expert2"] = (
            summary[f"full_{image}_expert2"] - summary[f"full_{image}_expert"]
        )
        features[f"delta_occlusion_{image}_expert2"] = (
            summary[f"occlusion_{image}_expert2"] - summary[f"occlusion_{image}_expert"]
        )
    return features


def _signature_vector(signature: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(signature[key]) for key in SIGNATURE_KEYS], dtype=float)


def _prototype_phase_mismatch(candidate_signature: dict[str, Any], prototype_signature: dict[str, Any]) -> int:
    return sum(
        candidate_signature[key] != prototype_signature[key]
        for key in PHASE_CATEGORY_KEYS
    )


def _serializable_settings(settings: LandscapeSettings) -> dict[str, Any]:
    return {
        field_name: getattr(settings, field_name)
        for field_name in settings.__dataclass_fields__
    }


def _reference_records(
    settings: LandscapeSettings,
    target_names: tuple[str, ...] = TARGET_ORDER,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for target_name in target_names:
        target_config = _reference_config_for_target(target_name)
        summary = simulate_extended_summary(
            target_config,
            n_steps_per_phase=settings.n_steps_per_phase,
            n_trials=settings.n_trials,
            continuation_trials_multiplier=settings.continuation_trials_multiplier,
            tail_window=settings.tail_window,
            intertrial_sigma=settings.intertrial_sigma,
        )
        signature = signature_from_summary(
            summary,
            activity_threshold=settings.activity_threshold,
        )
        records[target_name] = {
            "params": _param_dict_from_config(target_config),
            "summary": summary,
            "signature": signature,
        }
    return records


def _prototype_scale(reference_records: dict[str, dict[str, Any]]) -> np.ndarray:
    matrix = np.vstack([
        _signature_vector(record["signature"])
        for record in reference_records.values()
    ])
    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return scale


def _classify_signature(
    signature: dict[str, Any],
    *,
    prototype_records: dict[str, dict[str, Any]],
    prototype_scale: np.ndarray,
    settings: LandscapeSettings,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    vector = _signature_vector(signature)
    scored: list[dict[str, Any]] = []
    for target_name, record in prototype_records.items():
        proto_vector = _signature_vector(record["signature"])
        rmse = float(np.sqrt(np.mean(((vector - proto_vector) / prototype_scale) ** 2)))
        mismatch_count = _prototype_phase_mismatch(signature, record["signature"])
        familiar_mismatch = int(
            signature["familiar_primary_transition"] != record["signature"]["familiar_primary_transition"]
        )
        novel_mismatch = int(
            signature["novel_primary_transition"] != record["signature"]["novel_primary_transition"]
        )
        objective = (
            rmse
            + settings.prototype_mismatch_penalty * mismatch_count
            + 0.2 * (familiar_mismatch + novel_mismatch)
        )
        scored.append(
            {
                "target": target_name,
                "objective": objective,
                "prototype_rmse": rmse,
                "phase_category_mismatches": mismatch_count,
                "primary_transition_mismatches": familiar_mismatch + novel_mismatch,
            }
        )
    scored.sort(key=lambda item: (item["objective"], item["prototype_rmse"]))
    best = scored[0]
    second = scored[1] if len(scored) > 1 else None
    classification = {
        "assigned_target": best["target"],
        "assigned_objective": best["objective"],
        "assigned_prototype_rmse": best["prototype_rmse"],
        "assigned_phase_category_mismatches": best["phase_category_mismatches"],
        "second_target": second["target"] if second is not None else None,
        "second_objective": second["objective"] if second is not None else float("inf"),
        "objective_margin": (
            (second["objective"] - best["objective"])
            if second is not None
            else float("inf")
        ),
    }
    return classification, scored


def _sobol_samples(
    n_samples: int,
    *,
    param_ranges: dict[str, tuple[float, float]],
    seed: int,
) -> list[dict[str, float]]:
    engine = torch.quasirandom.SobolEngine(len(PARAM_ORDER), scramble=True, seed=seed)
    unit = engine.draw(n_samples).numpy()
    samples: list[dict[str, float]] = []
    for row in unit:
        params: dict[str, float] = {}
        for value, name in zip(row, PARAM_ORDER, strict=True):
            low, high = param_ranges[name]
            log_low, log_high = math.log10(low), math.log10(high)
            params[name] = float(10.0 ** (log_low + value * (log_high - log_low)))
        samples.append(params)
    return samples


def _local_samples(
    parents: list[dict[str, float]],
    *,
    n_samples: int,
    param_ranges: dict[str, tuple[float, float]],
    sigma: float,
    seed: int,
) -> list[dict[str, float]]:
    if not parents or n_samples <= 0:
        return []
    generator = torch.Generator().manual_seed(seed)
    parent_logs = torch.tensor(
        [
            [math.log10(max(parent[name], param_ranges[name][0])) for name in PARAM_ORDER]
            for parent in parents
        ],
        dtype=torch.float32,
    )
    parent_indices = torch.randint(
        low=0,
        high=len(parents),
        size=(n_samples,),
        generator=generator,
    )
    noise = torch.randn((n_samples, len(PARAM_ORDER)), generator=generator) * sigma
    proposal_logs = parent_logs[parent_indices] + noise
    for idx, name in enumerate(PARAM_ORDER):
        low, high = param_ranges[name]
        proposal_logs[:, idx].clamp_(min=math.log10(low), max=math.log10(high))
    proposal = 10.0 ** proposal_logs
    return [
        {name: float(proposal[row_idx, col_idx]) for col_idx, name in enumerate(PARAM_ORDER)}
        for row_idx in range(n_samples)
    ]


def _candidate_record(
    params: dict[str, float],
    *,
    base_config: dict[str, Any],
    settings: LandscapeSettings,
    prototype_records: dict[str, dict[str, Any]],
    prototype_scale: np.ndarray,
    search_group: str,
) -> dict[str, Any]:
    config = _config_from_params(
        params,
        base_config=base_config,
        seed=base_config.get("seed", broad["seed"]),
    )
    summary = simulate_extended_summary(
        config,
        n_steps_per_phase=settings.n_steps_per_phase,
        n_trials=settings.n_trials,
        continuation_trials_multiplier=settings.continuation_trials_multiplier,
        tail_window=settings.tail_window,
        intertrial_sigma=settings.intertrial_sigma,
    )
    signature = signature_from_summary(
        summary,
        activity_threshold=settings.activity_threshold,
    )
    classification, all_scores = _classify_signature(
        signature,
        prototype_records=prototype_records,
        prototype_scale=prototype_scale,
        settings=settings,
    )
    return {
        "search_group": search_group,
        "params": params,
        "summary": summary,
        "signature": signature,
        "classification": classification,
        "scores": all_scores,
    }


def _evaluate_candidates(
    param_sets: list[dict[str, float]],
    *,
    base_config: dict[str, Any],
    settings: LandscapeSettings,
    prototype_records: dict[str, dict[str, Any]],
    prototype_scale: np.ndarray,
    search_group: str,
) -> list[dict[str, Any]]:
    return Parallel(n_jobs=settings.n_jobs, prefer="processes")(
        delayed(_candidate_record)(
            params,
            base_config=base_config,
            settings=settings,
            prototype_records=prototype_records,
            prototype_scale=prototype_scale,
            search_group=search_group,
        )
        for params in tqdm(param_sets, desc=f"Simulating {search_group} candidates")
    )


def _candidate_frame_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        row = {
            "candidate_idx": idx,
            "candidate_uid": f"{record['search_group']}::{idx}",
            "search_group": record["search_group"],
            **record["params"],
            **record["summary"],
            **record["classification"],
        }
        signature = record["signature"]
        for key in SIGNATURE_KEYS:
            row[key] = float(signature[key])
        for key in (
            "familiar_primary_transition",
            "novel_primary_transition",
            *PHASE_CATEGORY_KEYS,
        ):
            row[key] = signature[key]
        for score in record["scores"]:
            suffix = score["target"]
            row[f"objective__{suffix}"] = score["objective"]
            row[f"rmse__{suffix}"] = score["prototype_rmse"]
        rows.append(row)
    return pd.DataFrame(rows)


def _sample_param_sets_for_group(
    *,
    base_config: dict[str, Any],
    targets: tuple[str, ...],
    prototype_records: dict[str, dict[str, Any]],
    settings: LandscapeSettings,
    global_samples: int,
    seed_offset: int,
) -> list[dict[str, float]]:
    params = [_param_dict_from_config(base_config)]
    params.extend(dict(prototype_records[target]["params"]) for target in targets)
    params.extend(
        _sobol_samples(
            global_samples,
            param_ranges=DEFAULT_PARAM_RANGES,
            seed=settings.random_seed + seed_offset,
        )
    )
    for offset, target in enumerate(targets):
        params.extend(
            _local_samples(
                [prototype_records[target]["params"]],
                n_samples=settings.local_samples_per_target,
                param_ranges=DEFAULT_PARAM_RANGES,
                sigma=settings.local_sigma,
                seed=settings.random_seed + seed_offset + 100 + offset,
            )
        )
    return params


def _region_members_for_class(
    frame: pd.DataFrame,
    *,
    target: str,
    settings: LandscapeSettings,
) -> pd.DataFrame:
    region = frame.loc[
        frame["assigned_target"].eq(target)
        & frame["objective_margin"].ge(settings.stable_margin_threshold)
    ].copy()
    if len(region) < settings.stable_min_points:
        region = frame.loc[frame["assigned_target"].eq(target)].copy()
    return region.sort_values(["assigned_objective", "assigned_prototype_rmse"]).reset_index(drop=True)


def _select_quantile_boundary_points(
    region: pd.DataFrame,
    *,
    q_low: float,
    q_high: float,
) -> pd.DataFrame:
    selected: set[int] = set()
    if region.empty:
        return region.copy()
    selected.add(int(region.nsmallest(1, "assigned_objective").index[0]))
    for param in PARAM_ORDER:
        low = float(region[param].quantile(q_low))
        high = float(region[param].quantile(q_high))
        selected.add(int((region[param] - low).abs().idxmin()))
        selected.add(int((region[param] - high).abs().idxmin()))
    return region.loc[sorted(selected)].copy()


def _backtest_point(
    row: pd.Series,
    *,
    base_config: dict[str, Any],
    seeds: tuple[int, ...],
    settings: LandscapeSettings,
    prototype_records: dict[str, dict[str, Any]],
    prototype_scale: np.ndarray,
) -> list[dict[str, Any]]:
    params = {name: float(row[name]) for name in PARAM_ORDER}
    details: list[dict[str, Any]] = []
    for seed in seeds:
        config = _config_from_params(params, base_config=base_config, seed=seed)
        df, _ = run_experiment(config, n_steps_per_phase=settings.backtest_full_steps)
        summary = summarize_experiment_df_extended(
            df,
            tail_window=settings.backtest_tail_window,
            experiment_series="training_occluded_only",
        )
        signature = signature_from_summary(
            summary,
            activity_threshold=settings.activity_threshold,
        )
        classification, _ = _classify_signature(
            signature,
            prototype_records=prototype_records,
            prototype_scale=prototype_scale,
            settings=settings,
        )
        details.append(
            {
                "candidate_uid": row["candidate_uid"],
                "candidate_idx": int(row["candidate_idx"]),
                "search_group": row["search_group"],
                "target": row["assigned_target"],
                "seed": int(seed),
                "predicted_target": classification["assigned_target"],
                "matched_target": classification["assigned_target"] == row["assigned_target"],
                "predicted_objective": classification["assigned_objective"],
                **params,
                **summary,
            }
        )
    return details


def _validate_core_quantiles(
    region_members: dict[str, pd.DataFrame],
    *,
    settings: LandscapeSettings,
    prototype_records: dict[str, dict[str, Any]],
    prototype_scale: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    backtest_rows: list[dict[str, Any]] = []
    for target, region in region_members.items():
        base_config = _base_search_config(_target_receives_context(target))
        accepted_quantiles: tuple[float, float] | None = None
        accepted_details: list[dict[str, Any]] = []
        for q_low, q_high in settings.quantile_core_candidates:
            representative_points = _select_quantile_boundary_points(region, q_low=q_low, q_high=q_high)
            candidate_details = Parallel(n_jobs=settings.n_jobs, prefer="processes")(
                delayed(_backtest_point)(
                    row,
                    base_config=base_config,
                    seeds=settings.backtest_seeds,
                    settings=settings,
                    prototype_records=prototype_records,
                    prototype_scale=prototype_scale,
                )
                for _, row in representative_points.iterrows()
            )
            flat = [item for sublist in candidate_details for item in sublist]
            if flat and all(item["matched_target"] for item in flat):
                accepted_quantiles = (q_low, q_high)
                accepted_details = flat
                break
            if accepted_quantiles is None:
                accepted_details = flat
        if accepted_quantiles is None:
            accepted_quantiles = settings.quantile_core_candidates[-1]

        row = {
            "target": target,
            "n_region_points": int(len(region)),
            "validated_quantile_low": accepted_quantiles[0],
            "validated_quantile_high": accepted_quantiles[1],
            "familiar_primary_transition": region["familiar_primary_transition"].iloc[0],
            "novel_primary_transition": region["novel_primary_transition"].iloc[0],
            "core_boundary_backtest_success": float(np.mean([item["matched_target"] for item in accepted_details])) if accepted_details else float("nan"),
        }
        for param in PARAM_ORDER:
            row[f"{param}_low"] = float(region[param].quantile(accepted_quantiles[0]))
            row[f"{param}_high"] = float(region[param].quantile(accepted_quantiles[1]))
        rows.append(row)
        backtest_rows.extend(accepted_details)
    return pd.DataFrame(rows), pd.DataFrame(backtest_rows)


def _compute_embeddings(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    param_matrix = frame.loc[:, PARAM_ORDER].to_numpy(dtype=float)
    scaler = StandardScaler()
    X = scaler.fit_transform(param_matrix)

    pca3 = PCA(n_components=3, random_state=0)
    pca_coords = pca3.fit_transform(X)

    embedded = frame.copy()
    embedded["pca1"] = pca_coords[:, 0]
    embedded["pca2"] = pca_coords[:, 1]
    embedded["pca3"] = pca_coords[:, 2]
    metadata = {
        "pca_explained_variance_ratio": pca3.explained_variance_ratio_.tolist(),
        "pca_components": pca3.components_.tolist(),
        "param_scaler_mean": scaler.mean_.tolist(),
        "param_scaler_scale": scaler.scale_.tolist(),
    }
    return embedded, metadata, X


def _fit_classifier_diagnostics(frame: pd.DataFrame, X_scaled: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y = frame["assigned_target"].to_numpy()
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_enc,
        test_size=0.2,
        random_state=0,
        stratify=y_enc,
    )

    logistic = LogisticRegression(
        multi_class="multinomial",
        max_iter=4000,
        class_weight="balanced",
        random_state=0,
    )
    forest = RandomForestClassifier(
        n_estimators=500,
        random_state=0,
        class_weight="balanced_subsample",
        n_jobs=-1,
        min_samples_leaf=3,
    )
    logistic.fit(X_train, y_train)
    forest.fit(X_train, y_train)

    logistic_acc = cross_val_score(logistic, X_scaled, y_enc, cv=5, n_jobs=1).tolist()
    forest_acc = cross_val_score(forest, X_scaled, y_enc, cv=5, n_jobs=1).tolist()

    logistic_report = classification_report(
        y_test,
        logistic.predict(X_test),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    forest_report = classification_report(
        y_test,
        forest.predict(X_test),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    feature_importance = pd.DataFrame(
        {
            "parameter": PARAM_ORDER,
            "random_forest_importance": forest.feature_importances_,
        }
    ).sort_values("random_forest_importance", ascending=False)

    coef_rows: list[dict[str, Any]] = []
    for class_name, coef in zip(label_encoder.classes_, logistic.coef_, strict=True):
        for param, value in zip(PARAM_ORDER, coef, strict=True):
            coef_rows.append({"target": class_name, "parameter": param, "coefficient": float(value)})
    logistic_coefs = pd.DataFrame(coef_rows)

    diagnostics = {
        "logistic_cv_accuracy_mean": float(np.mean(logistic_acc)),
        "logistic_cv_accuracy_std": float(np.std(logistic_acc)),
        "forest_cv_accuracy_mean": float(np.mean(forest_acc)),
        "forest_cv_accuracy_std": float(np.std(forest_acc)),
        "logistic_report": logistic_report,
        "forest_report": forest_report,
    }
    return diagnostics, feature_importance, logistic_coefs


def _boundary_pairs(
    stable_frame: pd.DataFrame,
) -> pd.DataFrame:
    if stable_frame.empty:
        return pd.DataFrame()
    scaler = StandardScaler()
    X = scaler.fit_transform(stable_frame.loc[:, PARAM_ORDER].to_numpy(dtype=float))
    class_to_indices = {
        target: stable_frame.index[stable_frame["assigned_target"].eq(target)].to_numpy()
        for target in stable_frame["assigned_target"].unique()
    }
    rows: list[dict[str, Any]] = []
    for class_a, class_b in combinations(sorted(class_to_indices), 2):
        idx_a = class_to_indices[class_a]
        idx_b = class_to_indices[class_b]
        if len(idx_a) == 0 or len(idx_b) == 0:
            continue
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X[np.searchsorted(stable_frame.index.to_numpy(), idx_b)])
        distances, neighbors = nn.kneighbors(X[np.searchsorted(stable_frame.index.to_numpy(), idx_a)])
        pair_rows: list[dict[str, Any]] = []
        for dist, neighbor_idx, source_idx in zip(distances[:, 0], neighbors[:, 0], idx_a, strict=True):
            target_idx = idx_b[neighbor_idx]
            row_a = stable_frame.loc[source_idx]
            row_b = stable_frame.loc[target_idx]
            log_diffs = {
                param: abs(math.log10(float(row_b[param])) - math.log10(float(row_a[param])))
                for param in PARAM_ORDER
            }
            dominant = sorted(log_diffs.items(), key=lambda item: item[1], reverse=True)[:3]
            pair_rows.append(
                {
                    "class_a": class_a,
                    "class_b": class_b,
                    "candidate_idx_a": int(row_a["candidate_idx"]),
                    "candidate_idx_b": int(row_b["candidate_idx"]),
                    "distance": float(dist),
                    "dominant_param_1": dominant[0][0],
                    "dominant_logdiff_1": float(dominant[0][1]),
                    "dominant_param_2": dominant[1][0],
                    "dominant_logdiff_2": float(dominant[1][1]),
                    "dominant_param_3": dominant[2][0],
                    "dominant_logdiff_3": float(dominant[2][1]),
                    **{f"{param}_a": float(row_a[param]) for param in PARAM_ORDER},
                    **{f"{param}_b": float(row_b[param]) for param in PARAM_ORDER},
                }
            )
        pair_rows.sort(key=lambda item: item["distance"])
        rows.extend(pair_rows[:2])
    return pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)


def _geometric_interpolation(params_a: dict[str, float], params_b: dict[str, float], alpha: float) -> dict[str, float]:
    interpolated: dict[str, float] = {}
    for param in PARAM_ORDER:
        log_a = math.log10(params_a[param])
        log_b = math.log10(params_b[param])
        interpolated[param] = float(10.0 ** ((1.0 - alpha) * log_a + alpha * log_b))
    return interpolated


def _boundary_interpolation_analysis(
    boundary_pairs: pd.DataFrame,
    *,
    settings: LandscapeSettings,
    prototype_records: dict[str, dict[str, Any]],
    prototype_scale: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if boundary_pairs.empty:
        return pd.DataFrame()
    alphas = np.linspace(0.0, 1.0, settings.boundary_interp_points)
    for _, pair in boundary_pairs.iterrows():
        class_a = pair["class_a"]
        class_b = pair["class_b"]
        base_config = _base_search_config((True, True))
        params_a = {param: float(pair[f"{param}_a"]) for param in PARAM_ORDER}
        params_b = {param: float(pair[f"{param}_b"]) for param in PARAM_ORDER}
        for alpha in alphas:
            params = _geometric_interpolation(params_a, params_b, float(alpha))
            config = _config_from_params(params, base_config=base_config, seed=settings.random_seed)
            summary = simulate_extended_summary(
                config,
                n_steps_per_phase=settings.n_steps_per_phase,
                n_trials=settings.n_trials,
                continuation_trials_multiplier=settings.continuation_trials_multiplier,
                tail_window=settings.tail_window,
                intertrial_sigma=settings.intertrial_sigma,
            )
            signature = signature_from_summary(
                summary,
                activity_threshold=settings.activity_threshold,
            )
            classification, _ = _classify_signature(
                signature,
                prototype_records=prototype_records,
                prototype_scale=prototype_scale,
                settings=settings,
            )
            rows.append(
                {
                    "class_a": class_a,
                    "class_b": class_b,
                    "candidate_idx_a": int(pair["candidate_idx_a"]),
                    "candidate_idx_b": int(pair["candidate_idx_b"]),
                    "alpha": float(alpha),
                    "assigned_target": classification["assigned_target"],
                    "assigned_objective": classification["assigned_objective"],
                    "objective_margin": classification["objective_margin"],
                    "familiar_primary_transition": signature["familiar_primary_transition"],
                    "novel_primary_transition": signature["novel_primary_transition"],
                    "familiar_expert2_state": float(signature["familiar_expert2_state"]),
                    "novel_expert2_state": float(signature["novel_expert2_state"]),
                    **params,
                }
            )
    return pd.DataFrame(rows)


def _plot_scatter(
    ax,
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    alpha: float = 0.45,
    size: float = 10.0,
    title: str | None = None,
) -> None:
    for target in SEARCH_TARGETS:
        subset = df.loc[df["assigned_target"].eq(target)]
        if subset.empty:
            continue
        ax.scatter(
            subset[x],
            subset[y],
            s=size,
            alpha=alpha,
            c=TARGET_COLORS[target],
            label=target,
            edgecolors="none",
        )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title is not None:
        ax.set_title(title)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_embeddings(frame: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    _plot_scatter(ax, frame, "pca1", "pca2", title="PCA (2D)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=9)
    _save_fig(fig, output_dir / "embedding_overview_2d.png")

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    for target in SEARCH_TARGETS:
        subset = frame.loc[frame["assigned_target"].eq(target)]
        if subset.empty:
            continue
        ax.scatter(
            subset["pca1"],
            subset["pca2"],
            subset["pca3"],
            s=8,
            alpha=0.35,
            c=TARGET_COLORS[target],
            label=target,
            depthshade=False,
        )
    ax.set_xlabel("pca1")
    ax.set_ylabel("pca2")
    ax.set_zlabel("pca3")
    ax.legend(frameon=False, fontsize=8, loc="best")
    _save_fig(fig, output_dir / "embedding_overview_3d.png")


def _plot_pca_boundaries(frame: pd.DataFrame, output_dir: Path) -> None:
    X = frame[["pca1", "pca2"]].to_numpy(dtype=float)
    y = frame["assigned_target"].to_numpy()
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    clf = KNeighborsClassifier(n_neighbors=25, weights="distance")
    clf.fit(X, y_enc)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 220),
        np.linspace(y_min, y_max, 220),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    pred = clf.predict(grid).reshape(xx.shape)
    proba = clf.predict_proba(grid)
    entropy = -(proba * np.log(np.clip(proba, 1e-9, 1.0))).sum(axis=1).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    cmap = plt.matplotlib.colors.ListedColormap([TARGET_COLORS[name] for name in encoder.classes_])
    axes[0].contourf(xx, yy, pred, levels=np.arange(len(encoder.classes_) + 1) - 0.5, cmap=cmap, alpha=0.18)
    _plot_scatter(axes[0], frame, "pca1", "pca2", alpha=0.55, size=10.0, title="PCA kNN decision regions")
    heat = axes[1].contourf(xx, yy, entropy, levels=25, cmap="magma")
    _plot_scatter(axes[1], frame, "pca1", "pca2", alpha=0.35, size=9.0, title="PCA decision-boundary uncertainty")
    fig.colorbar(heat, ax=axes[1], shrink=0.8, label="prediction entropy")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        axes[0].legend(unique.values(), unique.keys(), frameon=False, fontsize=8)
    _save_fig(fig, output_dir / "pca_decision_boundaries.png")


def _plot_parameter_panels(frame: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    pairs = [
        ("w_ff_0", "lr_ff", "familiar FF weight vs lr_ff"),
        ("w_ff_1", "lr_ff", "novel FF weight vs lr_ff"),
        ("w_fb_0", "lr_fb", "familiar FB weight vs lr_fb"),
        ("w_fb_1", "lr_fb", "novel FB weight vs lr_fb"),
        ("w_lat_0", "lr_lat", "familiar LAT weight vs lr_lat"),
        ("w_lat_1", "lr_lat", "novel LAT weight vs lr_lat"),
    ]
    for ax, (x, y, title) in zip(axes.flat, pairs, strict=True):
        _plot_scatter(ax, frame, x, y, title=title)
        ax.set_xscale("log")
        ax.set_yscale("log")
    _save_fig(fig, output_dir / "parameter_lr_panels.png")


def _plot_scalar_panels(frame: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    configs = [
        ("familiar_naive_state", "familiar_expert_state", "Familiar naive vs expert state"),
        ("familiar_expert_state", "familiar_expert2_state", "Familiar expert vs expert2 state"),
        ("novel_naive_state", "novel_expert_state", "Novel naive vs expert state"),
        ("novel_expert_state", "novel_expert2_state", "Novel expert vs expert2 state"),
    ]
    for ax, (x, y, title) in zip(axes.flat, configs, strict=True):
        _plot_scatter(ax, frame, x, y, title=title)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.25)
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.25)
    _save_fig(fig, output_dir / "state_scalar_panels.png")


def _plot_feature_importance(feature_importance: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    top = feature_importance.head(12).iloc[::-1]
    ax.barh(top["parameter"], top["random_forest_importance"], color="#4c78a8")
    ax.set_xlabel("Random forest importance")
    ax.set_title("Most informative parameters for class separation")
    _save_fig(fig, output_dir / "feature_importance.png")


def _plot_boundary_paths(boundary_paths: pd.DataFrame, output_dir: Path) -> None:
    if boundary_paths.empty:
        return
    for (class_a, class_b, idx_a, idx_b), subset in boundary_paths.groupby(
        ["class_a", "class_b", "candidate_idx_a", "candidate_idx_b"],
        sort=False,
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].plot(subset["alpha"], subset["familiar_expert2_state"], marker="o", color="#e45756", label="familiar expert2")
        axes[0].plot(subset["alpha"], subset["novel_expert2_state"], marker="o", color="#4c78a8", label="novel expert2")
        axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.25)
        axes[0].set_xlabel("interpolation alpha")
        axes[0].set_ylabel("expert2 state scalar")
        axes[0].set_title(f"{class_a} to {class_b}: state path")
        axes[0].legend(frameon=False, fontsize=8)

        encoded = subset["assigned_target"].astype("category")
        axes[1].scatter(subset["alpha"], np.arange(len(subset)), c=[TARGET_COLORS[val] for val in subset["assigned_target"]], s=40)
        axes[1].set_yticks(np.arange(len(subset)))
        axes[1].set_yticklabels(subset["assigned_target"].tolist(), fontsize=8)
        axes[1].set_xlabel("interpolation alpha")
        axes[1].set_title("Assigned class along path")
        _save_fig(fig, output_dir / f"boundary_path_{class_a}_to_{class_b}_{idx_a}_{idx_b}.png")


def _boundary_switch_summary(boundary_paths: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if boundary_paths.empty:
        return pd.DataFrame()
    grouped = boundary_paths.groupby(
        ["class_a", "class_b", "candidate_idx_a", "candidate_idx_b"],
        sort=False,
    )
    for (class_a, class_b, idx_a, idx_b), subset in grouped:
        subset = subset.sort_values("alpha")
        switch_points = subset.loc[subset["assigned_target"].ne(class_a)]
        first_switch_alpha = float(switch_points["alpha"].iloc[0]) if not switch_points.empty else float("nan")
        class_b_points = subset.loc[subset["assigned_target"].eq(class_b)]
        first_class_b_alpha = float(class_b_points["alpha"].iloc[0]) if not class_b_points.empty else float("nan")
        rows.append(
            {
                "class_a": class_a,
                "class_b": class_b,
                "candidate_idx_a": int(idx_a),
                "candidate_idx_b": int(idx_b),
                "first_switch_alpha": first_switch_alpha,
                "first_class_b_alpha": first_class_b_alpha,
                "path_sequence": " | ".join(subset["assigned_target"].tolist()),
            }
        )
    return pd.DataFrame(rows)


def _save_report(
    *,
    save_dir: Path,
    settings: LandscapeSettings,
    reference_records: dict[str, dict[str, Any]],
    diagnostics: dict[str, Any],
    region_summary: pd.DataFrame,
    validated_core: pd.DataFrame,
    boundary_pairs: pd.DataFrame,
    embedding_metadata: dict[str, Any],
) -> None:
    lines = [
        "# Learning-Rate + Initial-Weight Landscape",
        "",
        "This analysis expands the searched space from initial weights alone to `8` initial-weight parameters plus `4` learning rates.",
        "The standard search uses `receives_context=(True, True)`.",
        "The `FF_un` search remains separate with `receives_context=(False, False)`.",
        "Because of that categorical difference, `FF_un` is treated as a separate manifold when discussing class-switch boundaries.",
        "",
        "## Search Space",
        "",
        f"- compact simulator: `{settings.n_steps_per_phase}` steps/phase, `{settings.n_trials}` trials, tail window `{settings.tail_window}`",
        f"- standard samples: `{settings.global_samples_standard}` global + `{settings.local_samples_per_target}` local per target",
        f"- FF_un samples: `{settings.global_samples_ff_un}` global + `{settings.local_samples_per_target}` local",
        f"- backtest: `{settings.backtest_full_steps}` steps/phase across seeds `{settings.backtest_seeds}`",
        "",
        "## Prototype Notes",
        "",
        "- `un_un` is kept only as a diagnostic reference. Under `receives_context=(True, True)`, it is not separately identifiable from `un_FB` with this minimal model.",
        "",
        "## Embedding Overview",
        "",
        f"- PCA explained variance: {', '.join(f'{value:.3f}' for value in embedding_metadata['pca_explained_variance_ratio'])}",
        "",
        "## Classifier Diagnostics",
        "",
        f"- logistic regression CV accuracy: `{diagnostics['logistic_cv_accuracy_mean']:.3f} ± {diagnostics['logistic_cv_accuracy_std']:.3f}`",
        f"- random forest CV accuracy: `{diagnostics['forest_cv_accuracy_mean']:.3f} ± {diagnostics['forest_cv_accuracy_std']:.3f}`",
        "",
        "## Stable Regions",
        "",
    ]
    for row in region_summary.to_dict(orient="records"):
        lines.append(
            (
                f"- `{row['target']}`: n=`{int(row['n_region_points'])}`, "
                f"primary familiar=`{row['familiar_primary_transition']}`, "
                f"primary novel=`{row['novel_primary_transition']}`"
            )
        )
    lines.extend(["", "## Validated Core Ranges", ""])
    for row in validated_core.to_dict(orient="records"):
        lines.append(
            (
                f"- `{row['target']}`: q{int(100 * row['validated_quantile_low'])}-q{int(100 * row['validated_quantile_high'])}, "
                f"boundary success=`{row['core_boundary_backtest_success']:.2f}`"
            )
        )
    lines.extend(["", "## Boundary Pairs", ""])
    if boundary_pairs.empty:
        lines.append("- No cross-class boundary pairs were identified.")
    else:
        for row in boundary_pairs.head(10).to_dict(orient="records"):
            lines.append(
                (
                    f"- `{row['class_a']}` vs `{row['class_b']}`: normalized distance `{row['distance']:.3f}`, "
                    f"dominant changes `{row['dominant_param_1']}`, `{row['dominant_param_2']}`, `{row['dominant_param_3']}`"
                )
            )

    (save_dir / "overview_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_landscape_analysis(
    *,
    settings: LandscapeSettings = LandscapeSettings(),
    save_dir: str | Path | None = None,
) -> dict[str, Any]:
    reference_records = _reference_records(settings)
    standard_prototypes = {target: reference_records[target] for target in STANDARD_TARGETS}
    search_prototypes = {target: reference_records[target] for target in SEARCH_TARGETS}
    standard_scale = _prototype_scale(standard_prototypes)
    search_scale = _prototype_scale(search_prototypes)

    standard_base = _base_search_config((True, True))
    ff_un_base = _base_search_config((False, False))

    standard_params = _sample_param_sets_for_group(
        base_config=standard_base,
        targets=STANDARD_TARGETS,
        prototype_records=search_prototypes,
        settings=settings,
        global_samples=settings.global_samples_standard,
        seed_offset=0,
    )
    ff_un_params = _sample_param_sets_for_group(
        base_config=ff_un_base,
        targets=SPECIAL_TARGETS,
        prototype_records=search_prototypes,
        settings=settings,
        global_samples=settings.global_samples_ff_un,
        seed_offset=10000,
    )

    standard_records = _evaluate_candidates(
        standard_params,
        base_config=standard_base,
        settings=settings,
        prototype_records=standard_prototypes,
        prototype_scale=standard_scale,
        search_group="standard",
    )
    ff_un_records = _evaluate_candidates(
        ff_un_params,
        base_config=ff_un_base,
        settings=settings,
        prototype_records=search_prototypes,
        prototype_scale=search_scale,
        search_group="ff_un",
    )

    combined = pd.concat(
        [
            _candidate_frame_from_records(standard_records),
            _candidate_frame_from_records(ff_un_records),
        ],
        ignore_index=True,
    )
    region_members = {
        target: _region_members_for_class(combined, target=target, settings=settings)
        for target in SEARCH_TARGETS
    }
    stable_members = pd.concat(
        [region.assign(target=target) for target, region in region_members.items() if not region.empty],
        ignore_index=True,
    )
    region_summary = pd.DataFrame(
        [
            {
                "target": target,
                "n_region_points": int(len(region)),
                "familiar_primary_transition": region["familiar_primary_transition"].iloc[0] if not region.empty else None,
                "novel_primary_transition": region["novel_primary_transition"].iloc[0] if not region.empty else None,
            }
            for target, region in region_members.items()
        ]
    )

    standard_regions = {target: region_members[target] for target in STANDARD_TARGETS}
    ff_un_regions = {target: region_members[target] for target in SPECIAL_TARGETS}
    validated_core_standard, core_backtests_standard = _validate_core_quantiles(
        standard_regions,
        settings=settings,
        prototype_records=standard_prototypes,
        prototype_scale=standard_scale,
    )
    validated_core_ff_un, core_backtests_ff_un = _validate_core_quantiles(
        ff_un_regions,
        settings=settings,
        prototype_records=search_prototypes,
        prototype_scale=search_scale,
    )
    validated_core = pd.concat([validated_core_standard, validated_core_ff_un], ignore_index=True)
    core_backtests = pd.concat([core_backtests_standard, core_backtests_ff_un], ignore_index=True)
    embedded, embedding_metadata, X_scaled = _compute_embeddings(combined)
    diagnostics, feature_importance, logistic_coefs = _fit_classifier_diagnostics(embedded, X_scaled)
    stable_embedded = embedded.loc[embedded["candidate_uid"].isin(stable_members["candidate_uid"])].copy()
    boundary_pairs = _boundary_pairs(
        stable_embedded.loc[stable_embedded["search_group"].eq("standard")].copy()
    )
    boundary_paths = _boundary_interpolation_analysis(
        boundary_pairs,
        settings=settings,
        prototype_records=standard_prototypes,
        prototype_scale=standard_scale,
    )
    boundary_switches = _boundary_switch_summary(boundary_paths)

    result = {
        "settings": settings,
        "reference_records": reference_records,
        "combined": combined,
        "stable_members": stable_members,
        "region_summary": region_summary,
        "validated_core": validated_core,
        "core_backtests": core_backtests,
        "embedded": embedded,
        "embedding_metadata": embedding_metadata,
        "diagnostics": diagnostics,
        "feature_importance": feature_importance,
        "logistic_coefs": logistic_coefs,
        "boundary_pairs": boundary_pairs,
        "boundary_paths": boundary_paths,
        "boundary_switches": boundary_switches,
    }

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        combined.to_csv(save_path / "combined_candidates.csv", index=False)
        stable_members.to_csv(save_path / "stable_region_members.csv", index=False)
        region_summary.to_csv(save_path / "region_summary.csv", index=False)
        validated_core.to_csv(save_path / "validated_core_summary.csv", index=False)
        core_backtests.to_csv(save_path / "validated_core_backtests.csv", index=False)
        embedded.to_csv(save_path / "embedded_candidates.csv", index=False)
        feature_importance.to_csv(save_path / "feature_importance.csv", index=False)
        logistic_coefs.to_csv(save_path / "logistic_coefficients.csv", index=False)
        boundary_pairs.to_csv(save_path / "boundary_pairs.csv", index=False)
        boundary_paths.to_csv(save_path / "boundary_paths.csv", index=False)
        boundary_switches.to_csv(save_path / "boundary_switches.csv", index=False)
        with (save_path / "reference_signatures.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "settings": _serializable_settings(settings),
                    "references": {
                        target: {
                            "params": {key: float(value) for key, value in record["params"].items()},
                            "summary": {key: float(value) for key, value in record["summary"].items()},
                            "signature": {
                                key: (float(value) if isinstance(value, (int, float, np.floating)) else value)
                                for key, value in record["signature"].items()
                            },
                        }
                        for target, record in reference_records.items()
                    },
                    "embedding_metadata": embedding_metadata,
                    "diagnostics": diagnostics,
                },
                handle,
                indent=2,
            )

        plot_dir = save_path / "plots"
        _plot_embeddings(embedded, plot_dir)
        _plot_pca_boundaries(embedded, plot_dir)
        _plot_parameter_panels(stable_members, plot_dir)
        _plot_scalar_panels(stable_members, plot_dir)
        _plot_feature_importance(feature_importance, plot_dir)
        _plot_boundary_paths(boundary_paths, plot_dir)
        _save_report(
            save_dir=save_path,
            settings=settings,
            reference_records=reference_records,
            diagnostics=diagnostics,
            region_summary=region_summary,
            validated_core=validated_core,
            boundary_pairs=boundary_pairs,
            embedding_metadata=embedding_metadata,
        )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explore the joint landscape of initial weights and learning rates for "
            "the minimal context-contrasting model, then analyze embeddings and class boundaries."
        )
    )
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--n-trials", type=int, default=6)
    parser.add_argument("--tail-window", type=int, default=60)
    parser.add_argument("--global-samples-standard", type=int, default=1800)
    parser.add_argument("--global-samples-ff-un", type=int, default=1400)
    parser.add_argument("--local-samples-per-target", type=int, default=160)
    parser.add_argument("--local-sigma", type=float, default=0.24)
    parser.add_argument("--stable-margin-threshold", type=float, default=0.08)
    parser.add_argument("--stable-min-points", type=int, default=40)
    parser.add_argument("--backtest-full-steps", type=int, default=100)
    parser.add_argument("--backtest-tail-window", type=int, default=80)
    parser.add_argument("--backtest-seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="context_contrasting/sbi/results/latest_lr_landscape",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = LandscapeSettings(
        n_steps_per_phase=args.n_steps,
        n_trials=args.n_trials,
        tail_window=args.tail_window,
        global_samples_standard=args.global_samples_standard,
        global_samples_ff_un=args.global_samples_ff_un,
        local_samples_per_target=args.local_samples_per_target,
        local_sigma=args.local_sigma,
        stable_margin_threshold=args.stable_margin_threshold,
        stable_min_points=args.stable_min_points,
        backtest_full_steps=args.backtest_full_steps,
        backtest_tail_window=args.backtest_tail_window,
        backtest_seeds=tuple(args.backtest_seeds),
        n_jobs=args.n_jobs,
        random_seed=args.seed,
    )
    result = run_landscape_analysis(
        settings=settings,
        save_dir=args.save_dir,
    )
    print(result["region_summary"].to_string(index=False))
    print(result["validated_core"][["target", "validated_quantile_low", "validated_quantile_high", "core_boundary_backtest_success"]].to_string(index=False))


if __name__ == "__main__":
    main()
