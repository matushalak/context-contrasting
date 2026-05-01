from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from context_contrasting.minimal.config import broad, minimal_configs
from context_contrasting.minimal.experiment import design_experimental_phase, run_experiment
from context_contrasting.minimal.minimal import CCNeuron
from context_contrasting.minimal.transition_types import (
    scalar_state_profile_from_summary,
    split_transition_label,
    transition_match_score,
    transition_profile_from_summary,
)

SUMMARY_KEYS = (
    "full_familiar_naive",
    "occlusion_familiar_naive",
    "full_familiar_expert",
    "occlusion_familiar_expert",
    "full_novel_naive",
    "occlusion_novel_naive",
    "full_novel_expert",
    "occlusion_novel_expert",
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
REFERENCE_ONLY_TARGETS = ("un_un",)
STANDARD_REGION_TARGETS = (
    "un_FB",
    "FF_FB_broad",
    "FF_FB_narrow_familiar",
    "FF_FB_narrow_novel",
    "FB_FB",
)
SPECIAL_REGION_TARGETS = ("FF_un",)
SEARCH_REGION_TARGETS = STANDARD_REGION_TARGETS + SPECIAL_REGION_TARGETS
ALL_REGION_TARGETS = REFERENCE_ONLY_TARGETS + SEARCH_REGION_TARGETS
DEFAULT_ACTIVITY_THRESHOLD = 0.025


@dataclass(frozen=True)
class SweepSettings:
    n_steps_per_phase: int = 80
    n_trials: int = 6
    tail_window: int = 60
    intertrial_sigma: float = 0.05
    global_samples: int = 2048
    local_samples_per_target: int = 192
    local_sigma: float = 0.22
    n_jobs: int = -1
    random_seed: int = 7
    activity_threshold: float = DEFAULT_ACTIVITY_THRESHOLD
    weight_floor: float = 1e-4
    weight_ceiling: float = 2.0
    fit_scale: float = DEFAULT_ACTIVITY_THRESHOLD
    region_gap_threshold: float = 0.02
    region_min_points: int = 12
    backtest_full_steps: int = 100
    backtest_tail_window: int = 80
    backtest_seeds: tuple[int, ...] = (11, 23, 37)
    backtest_random_points: int = 4


def _summary_tensor(summary: dict[str, float]) -> torch.Tensor:
    return torch.tensor([summary[key] for key in SUMMARY_KEYS], dtype=torch.float32)


def _round_dict(data: dict[str, float], ndigits: int = 6) -> dict[str, float]:
    return {key: round(float(value), ndigits) for key, value in data.items()}


def _serializable_settings(settings: SweepSettings) -> dict[str, Any]:
    return {
        field_name: getattr(settings, field_name)
        for field_name in settings.__dataclass_fields__
    }


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
    if target_name == "FF_un":
        return (False, False)
    return (True, True)


def _reference_config_for_target(target_name: str) -> dict[str, Any]:
    config = deepcopy(minimal_configs[target_name])
    config["receives_context"] = _target_receives_context(target_name)
    return config


def _weights_from_config(config: dict[str, Any]) -> dict[str, float]:
    return {
        "w_ff_0": float(config["w_ff_init"]["mu"][0]),
        "w_ff_1": float(config["w_ff_init"]["mu"][1]),
        "w_fb_0": float(config["w_fb_init"]["mu"][0]),
        "w_fb_1": float(config["w_fb_init"]["mu"][1]),
        "w_lat_0": float(config["w_lat_init"]["mu"][0]),
        "w_lat_1": float(config["w_lat_init"]["mu"][1]),
        "w_pv_lat_0": float(config["w_pv_lat_init"]["mu"][0]),
        "w_pv_lat_1": float(config["w_pv_lat_init"]["mu"][1]),
    }


def _config_from_weights(
    weights: dict[str, float],
    *,
    base_config: dict[str, Any],
    seed: int | None = None,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    config["w_ff_init"] = {"mu": [weights["w_ff_0"], weights["w_ff_1"]], "sigma": 0.0}
    config["w_fb_init"] = {"mu": [weights["w_fb_0"], weights["w_fb_1"]], "sigma": 0.0}
    config["w_lat_init"] = {"mu": [weights["w_lat_0"], weights["w_lat_1"]], "sigma": 0.0}
    config["w_pv_lat_init"] = {
        "mu": [weights["w_pv_lat_0"], weights["w_pv_lat_1"]],
        "sigma": 0.0,
    }
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


def simulate_response_summary(
    model_config: dict[str, Any],
    *,
    n_steps_per_phase: int,
    n_trials: int,
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
    O = torch.zeros_like(X1)

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
    return summary


def summarize_experiment_df(
    df: pd.DataFrame,
    *,
    tail_window: int,
    experiment_series: str = "training_familiar",
) -> dict[str, float]:
    subset = df.loc[df["experiment_series"].eq(experiment_series)]
    return {
        key: float(subset.loc[subset["condition"].eq(key), "y"].tail(tail_window).mean())
        for key in SUMMARY_KEYS
    }


def _target_reference_records_compact(
    settings: SweepSettings,
    target_names: tuple[str, ...] = ALL_REGION_TARGETS,
) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for target_name in target_names:
        target_config = _reference_config_for_target(target_name)
        summary = simulate_response_summary(
            target_config,
            n_steps_per_phase=settings.n_steps_per_phase,
            n_trials=settings.n_trials,
            tail_window=settings.tail_window,
            intertrial_sigma=settings.intertrial_sigma,
        )
        references[target_name] = {
            "summary": summary,
            "summary_tensor": _summary_tensor(summary),
            "transitions": transition_profile_from_summary(
                summary,
                activity_threshold=settings.activity_threshold,
            ),
            "scalars": scalar_state_profile_from_summary(
                summary,
                activity_threshold=settings.activity_threshold,
            ),
            "reference_weights": _weights_from_config(target_config),
        }
    return references


def _target_reference_records_full(
    settings: SweepSettings,
    target_names: tuple[str, ...] = ALL_REGION_TARGETS,
) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for target_name in target_names:
        target_config = _reference_config_for_target(target_name)
        df, _ = run_experiment(target_config, n_steps_per_phase=settings.backtest_full_steps)
        summary = summarize_experiment_df(df, tail_window=settings.backtest_tail_window)
        references[target_name] = {
            "summary": summary,
            "summary_tensor": _summary_tensor(summary),
            "transitions": transition_profile_from_summary(
                summary,
                activity_threshold=settings.activity_threshold,
            ),
            "scalars": scalar_state_profile_from_summary(
                summary,
                activity_threshold=settings.activity_threshold,
            ),
            "reference_weights": _weights_from_config(target_config),
        }
    return references


def _candidate_record(
    weights: dict[str, float],
    *,
    base_config: dict[str, Any],
    settings: SweepSettings,
) -> dict[str, Any]:
    candidate_config = _config_from_weights(
        weights,
        base_config=base_config,
        seed=base_config.get("seed", broad["seed"]),
    )
    summary = simulate_response_summary(
        candidate_config,
        n_steps_per_phase=settings.n_steps_per_phase,
        n_trials=settings.n_trials,
        tail_window=settings.tail_window,
        intertrial_sigma=settings.intertrial_sigma,
    )
    scalars = scalar_state_profile_from_summary(
        summary,
        activity_threshold=settings.activity_threshold,
    )
    return {
        "weights": weights,
        "summary": summary,
        "scalars": scalars,
        "transitions": {
            "familiar": scalars["familiar_transition"],
            "novel": scalars["novel_transition"],
        },
    }


def _score_summary_for_target(
    summary: dict[str, float],
    transitions: dict[str, str],
    *,
    target_name: str,
    target_reference: dict[str, Any],
    fit_scale: float,
    activity_threshold: float,
) -> dict[str, Any]:
    summary_tensor = _summary_tensor(summary)
    target_tensor = target_reference["summary_tensor"]
    normalized_rmse = float(torch.sqrt(torch.mean(((summary_tensor - target_tensor) / fit_scale) ** 2)))

    familiar_target = target_reference["transitions"]["familiar"]
    novel_target = target_reference["transitions"]["novel"]
    familiar_score = transition_match_score(
        familiar_target,
        summary["full_familiar_naive"],
        summary["occlusion_familiar_naive"],
        summary["full_familiar_expert"],
        summary["occlusion_familiar_expert"],
        activity_threshold=activity_threshold,
    )
    novel_score = transition_match_score(
        novel_target,
        summary["full_novel_naive"],
        summary["occlusion_novel_naive"],
        summary["full_novel_expert"],
        summary["occlusion_novel_expert"],
        activity_threshold=activity_threshold,
    )

    target_familiar_states = split_transition_label(familiar_target)
    target_novel_states = split_transition_label(novel_target)
    candidate_familiar_states = split_transition_label(transitions["familiar"])
    candidate_novel_states = split_transition_label(transitions["novel"])
    mismatch_penalty = sum(
        target_state != candidate_state
        for target_state, candidate_state in (
            (target_familiar_states[0], candidate_familiar_states[0]),
            (target_familiar_states[1], candidate_familiar_states[1]),
            (target_novel_states[0], candidate_novel_states[0]),
            (target_novel_states[1], candidate_novel_states[1]),
        )
    )
    transition_bonus = (familiar_score + novel_score) / fit_scale
    objective = normalized_rmse + 0.5 * mismatch_penalty - 0.05 * transition_bonus

    return {
        "target": target_name,
        "objective": float(objective),
        "normalized_rmse": normalized_rmse,
        "mismatch_penalty": int(mismatch_penalty),
        "transition_score_familiar": float(familiar_score),
        "transition_score_novel": float(novel_score),
        "transition_bonus": float(transition_bonus),
    }


def _classify_summary(
    summary: dict[str, float],
    *,
    target_references: dict[str, dict[str, Any]],
    settings: SweepSettings,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scalars = scalar_state_profile_from_summary(
        summary,
        activity_threshold=settings.activity_threshold,
    )
    transitions = {
        "familiar": scalars["familiar_transition"],
        "novel": scalars["novel_transition"],
    }
    scores = [
        _score_summary_for_target(
            summary,
            transitions,
            target_name=target_name,
            target_reference=target_reference,
            fit_scale=settings.fit_scale,
            activity_threshold=settings.activity_threshold,
        )
        for target_name, target_reference in target_references.items()
    ]
    scores.sort(key=lambda record: (record["objective"], record["normalized_rmse"]))
    best = scores[0]
    second = scores[1] if len(scores) > 1 else None

    classification = {
        "assigned_target": best["target"],
        "assigned_objective": best["objective"],
        "assigned_rmse": best["normalized_rmse"],
        "assigned_mismatch_penalty": best["mismatch_penalty"],
        "assigned_transition_bonus": best["transition_bonus"],
        "second_target": second["target"] if second is not None else None,
        "second_objective": second["objective"] if second is not None else None,
        "objective_gap": (
            (second["objective"] - best["objective"])
            if second is not None
            else float("inf")
        ),
        **transitions,
        **scalars,
    }
    return classification, scores


def _sobol_weight_samples(
    n_samples: int,
    *,
    settings: SweepSettings,
    seed: int,
) -> list[dict[str, float]]:
    dim = len(WEIGHT_PARAM_ORDER)
    engine = torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed)
    unit = engine.draw(n_samples)
    log_low = math.log10(settings.weight_floor)
    log_high = math.log10(settings.weight_ceiling)
    log_weights = log_low + unit * (log_high - log_low)
    weights = 10.0 ** log_weights
    return [
        {
            name: float(weights[row_idx, col_idx])
            for col_idx, name in enumerate(WEIGHT_PARAM_ORDER)
        }
        for row_idx in range(n_samples)
    ]


def _local_weight_samples(
    parents: list[dict[str, float]],
    *,
    n_samples: int,
    settings: SweepSettings,
    seed: int,
) -> list[dict[str, float]]:
    if not parents or n_samples <= 0:
        return []

    generator = torch.Generator().manual_seed(seed)
    log_low = math.log10(settings.weight_floor)
    log_high = math.log10(settings.weight_ceiling)
    parent_logs = torch.tensor(
        [
            [math.log10(max(parent[name], settings.weight_floor)) for name in WEIGHT_PARAM_ORDER]
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
    noise = torch.randn((n_samples, len(WEIGHT_PARAM_ORDER)), generator=generator) * settings.local_sigma
    proposal_logs = parent_logs[parent_indices] + noise
    proposal_logs = proposal_logs.clamp_(min=log_low, max=log_high)
    proposals = 10.0 ** proposal_logs
    return [
        {
            name: float(proposals[row_idx, col_idx])
            for col_idx, name in enumerate(WEIGHT_PARAM_ORDER)
        }
        for row_idx in range(n_samples)
    ]


def _evaluate_candidates(
    weight_sets: list[dict[str, float]],
    *,
    base_config: dict[str, Any],
    settings: SweepSettings,
) -> list[dict[str, Any]]:
    if not weight_sets:
        return []

    return Parallel(n_jobs=settings.n_jobs, prefer="processes")(
        delayed(_candidate_record)(
            weights,
            base_config=base_config,
            settings=settings,
        )
        for weights in tqdm(weight_sets, desc="Simulating candidates")
    )


def _build_candidate_frame(
    candidates: list[dict[str, Any]],
    *,
    target_references: dict[str, dict[str, Any]],
    settings: SweepSettings,
    search_group: str,
    receives_context: tuple[bool, bool],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_idx, candidate in enumerate(candidates):
        classification, scores = _classify_summary(
            candidate["summary"],
            target_references=target_references,
            settings=settings,
        )
        row = {
            "candidate_idx": candidate_idx,
            "search_group": search_group,
            "receives_context_familiar": bool(receives_context[0]),
            "receives_context_novel": bool(receives_context[1]),
            **candidate["weights"],
            **candidate["summary"],
            **classification,
        }
        for score in scores:
            suffix = score["target"]
            row[f"objective__{suffix}"] = score["objective"]
            row[f"rmse__{suffix}"] = score["normalized_rmse"]
            row[f"mismatch__{suffix}"] = score["mismatch_penalty"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["assigned_target", "assigned_objective", "assigned_rmse"]
    ).reset_index(drop=True)


def _sample_weight_sets(
    *,
    allowed_targets: tuple[str, ...],
    target_references: dict[str, dict[str, Any]],
    settings: SweepSettings,
    seed: int,
    include_base_weights: dict[str, float],
) -> list[dict[str, float]]:
    weight_sets = [include_base_weights]
    for target_name in allowed_targets:
        weight_sets.append(dict(target_references[target_name]["reference_weights"]))

    weight_sets.extend(
        _sobol_weight_samples(
            settings.global_samples,
            settings=settings,
            seed=seed,
        )
    )
    for offset, target_name in enumerate(allowed_targets):
        weight_sets.extend(
            _local_weight_samples(
                [target_references[target_name]["reference_weights"]],
                n_samples=settings.local_samples_per_target,
                settings=settings,
                seed=seed + 1000 + offset,
            )
        )
    return weight_sets


def _extract_region_members(
    candidate_frame: pd.DataFrame,
    *,
    allowed_targets: tuple[str, ...],
    target_references: dict[str, dict[str, Any]],
    settings: SweepSettings,
) -> dict[str, pd.DataFrame]:
    regions: dict[str, pd.DataFrame] = {}
    for target_name in allowed_targets:
        reference = target_references[target_name]
        region = candidate_frame.loc[
            candidate_frame["assigned_target"].eq(target_name)
            & candidate_frame["familiar_transition"].eq(reference["transitions"]["familiar"])
            & candidate_frame["novel_transition"].eq(reference["transitions"]["novel"])
            & candidate_frame["objective_gap"].ge(settings.region_gap_threshold)
        ].copy()
        if len(region) < settings.region_min_points:
            region = candidate_frame.loc[
                candidate_frame["assigned_target"].eq(target_name)
                & candidate_frame["familiar_transition"].eq(reference["transitions"]["familiar"])
                & candidate_frame["novel_transition"].eq(reference["transitions"]["novel"])
            ].copy()
        regions[target_name] = region.sort_values(["assigned_objective", "assigned_rmse"]).reset_index(drop=True)
    return regions


def _summarize_regions(region_members: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stat_columns = list(WEIGHT_PARAM_ORDER) + [
        "assigned_objective",
        "assigned_rmse",
        "objective_gap",
        "familiar_naive_state",
        "familiar_expert_state",
        "novel_naive_state",
        "novel_expert_state",
    ]
    for target_name, region in region_members.items():
        if region.empty:
            rows.append({"target": target_name, "n_region_points": 0})
            continue

        row = {
            "target": target_name,
            "n_region_points": int(len(region)),
            "familiar_transition": region["familiar_transition"].iloc[0],
            "novel_transition": region["novel_transition"].iloc[0],
        }
        for column in stat_columns:
            values = region[column]
            row[f"{column}_min"] = float(values.min())
            row[f"{column}_q05"] = float(values.quantile(0.05))
            row[f"{column}_median"] = float(values.median())
            row[f"{column}_q95"] = float(values.quantile(0.95))
            row[f"{column}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def _select_region_representatives(
    region: pd.DataFrame,
    *,
    settings: SweepSettings,
) -> pd.DataFrame:
    if region.empty:
        return region.copy()

    selected_indices: set[int] = set()
    selected_indices.add(int(region.nsmallest(1, "assigned_objective").index[0]))
    for column in WEIGHT_PARAM_ORDER:
        selected_indices.add(int(region[column].idxmin()))
        selected_indices.add(int(region[column].idxmax()))

    if settings.backtest_random_points > 0 and len(region) > len(selected_indices):
        sampled = region.sample(
            n=min(settings.backtest_random_points, len(region)),
            random_state=settings.random_seed,
        )
        selected_indices.update(int(idx) for idx in sampled.index.tolist())

    return region.loc[sorted(selected_indices)].copy()


def _backtest_representatives(
    representatives: pd.DataFrame,
    *,
    base_config: dict[str, Any],
    backtest_references: dict[str, dict[str, Any]],
    settings: SweepSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if representatives.empty:
        return pd.DataFrame(), pd.DataFrame()

    representative_records = representatives.to_dict(orient="records")

    def _run_single_backtest(rep: dict[str, Any], seed: int) -> dict[str, Any]:
        weights = {name: float(rep[name]) for name in WEIGHT_PARAM_ORDER}
        config = _config_from_weights(weights, base_config=base_config, seed=seed)
        df, _ = run_experiment(config, n_steps_per_phase=settings.backtest_full_steps)
        summary = summarize_experiment_df(df, tail_window=settings.backtest_tail_window)
        classification, _ = _classify_summary(
            summary,
            target_references=backtest_references,
            settings=settings,
        )
        matched = classification["assigned_target"] == rep["assigned_target"]
        return {
            "search_group": rep["search_group"],
            "candidate_idx": int(rep["candidate_idx"]),
            "target": rep["assigned_target"],
            "seed": int(seed),
            "matched_target": bool(matched),
            "predicted_target": classification["assigned_target"],
            "predicted_objective": classification["assigned_objective"],
            "predicted_rmse": classification["assigned_rmse"],
            "predicted_gap": classification["objective_gap"],
            "familiar_transition": classification["familiar_transition"],
            "novel_transition": classification["novel_transition"],
            **weights,
            **summary,
        }

    detail_rows = Parallel(n_jobs=settings.n_jobs, prefer="processes")(
        delayed(_run_single_backtest)(rep, seed)
        for rep in tqdm(representative_records, desc="Backtesting region boundaries")
        for seed in settings.backtest_seeds
    )
    detail_frame = pd.DataFrame(detail_rows)
    summary_rows: list[dict[str, Any]] = []
    for rep in representative_records:
        rep_rows = detail_frame.loc[detail_frame["candidate_idx"].eq(int(rep["candidate_idx"]))]
        summary_rows.append(
            {
                "search_group": rep["search_group"],
                "candidate_idx": int(rep["candidate_idx"]),
                "target": rep["assigned_target"],
                "backtest_success_rate": float(rep_rows["matched_target"].mean()),
                **{name: float(rep[name]) for name in WEIGHT_PARAM_ORDER},
            }
        )

    return pd.DataFrame(summary_rows), detail_frame


def run_region_search_group(
    *,
    search_group: str,
    allowed_targets: tuple[str, ...],
    receives_context: tuple[bool, bool],
    settings: SweepSettings,
    compact_references: dict[str, dict[str, Any]],
    backtest_references: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base_config = _base_search_config(receives_context=receives_context)
    weight_sets = _sample_weight_sets(
        allowed_targets=allowed_targets,
        target_references=compact_references,
        settings=settings,
        seed=settings.random_seed + (0 if search_group == "standard" else 10000),
        include_base_weights=_weights_from_config(base_config),
    )
    candidates = _evaluate_candidates(
        weight_sets,
        base_config=base_config,
        settings=settings,
    )
    candidate_frame = _build_candidate_frame(
        candidates,
        target_references=compact_references,
        settings=settings,
        search_group=search_group,
        receives_context=receives_context,
    )
    region_members = _extract_region_members(
        candidate_frame,
        allowed_targets=allowed_targets,
        target_references=compact_references,
        settings=settings,
    )
    region_summary = _summarize_regions(region_members)

    representatives = pd.concat(
        [
            _select_region_representatives(region, settings=settings)
            for region in region_members.values()
            if not region.empty
        ],
        ignore_index=True,
    ) if any(not region.empty for region in region_members.values()) else pd.DataFrame()
    backtest_summary, backtest_details = _backtest_representatives(
        representatives,
        base_config=base_config,
        backtest_references=backtest_references,
        settings=settings,
    )
    return {
        "search_group": search_group,
        "candidate_frame": candidate_frame,
        "region_members": region_members,
        "region_summary": region_summary,
        "representatives": representatives,
        "backtest_summary": backtest_summary,
        "backtest_details": backtest_details,
    }


def _save_region_results(
    result: dict[str, Any],
    *,
    compact_references: dict[str, dict[str, Any]],
    backtest_references: dict[str, dict[str, Any]],
    settings: SweepSettings,
    save_dir: str | Path,
) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    candidate_frames = [group["candidate_frame"] for group in result["groups"].values()]
    region_frames = [
        region.assign(target=target_name)
        for group in result["groups"].values()
        for target_name, region in group["region_members"].items()
        if not region.empty
    ]
    summary_frames = [group["region_summary"] for group in result["groups"].values()]
    representative_frames = [group["representatives"] for group in result["groups"].values() if not group["representatives"].empty]
    backtest_summaries = [group["backtest_summary"] for group in result["groups"].values() if not group["backtest_summary"].empty]
    backtest_details = [group["backtest_details"] for group in result["groups"].values() if not group["backtest_details"].empty]

    combined_candidates = pd.concat(candidate_frames, ignore_index=True)
    combined_candidates.to_csv(save_path / "combined_candidates.csv", index=False)

    if region_frames:
        pd.concat(region_frames, ignore_index=True).to_csv(save_path / "stable_region_members.csv", index=False)
    if summary_frames:
        region_summary = pd.concat(summary_frames, ignore_index=True)
        if backtest_summaries:
            backtest_summary = pd.concat(backtest_summaries, ignore_index=True)
            target_success = (
                backtest_summary.groupby("target", as_index=False)["backtest_success_rate"]
                .agg(["min", "mean"])
                .reset_index()
                .rename(columns={"min": "boundary_backtest_success_min", "mean": "boundary_backtest_success_mean"})
            )
            region_summary = region_summary.merge(target_success, on="target", how="left")
        region_summary.to_csv(save_path / "region_summary.csv", index=False)
    if representative_frames:
        pd.concat(representative_frames, ignore_index=True).to_csv(save_path / "region_representatives.csv", index=False)
    if backtest_summaries:
        pd.concat(backtest_summaries, ignore_index=True).to_csv(save_path / "backtest_summary.csv", index=False)
    if backtest_details:
        pd.concat(backtest_details, ignore_index=True).to_csv(save_path / "backtest_details.csv", index=False)

    with (save_path / "reference_summaries.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "settings": _serializable_settings(settings),
                "compact_references": {
                    target: {
                        "summary": _round_dict(reference["summary"]),
                        "transitions": reference["transitions"],
                        "scalars": _round_dict({
                            key: value
                            for key, value in reference["scalars"].items()
                            if isinstance(value, (int, float))
                        }),
                        "weights": _round_dict(reference["reference_weights"]),
                    }
                    for target, reference in compact_references.items()
                },
                "backtest_references": {
                    target: {
                        "summary": _round_dict(reference["summary"]),
                        "transitions": reference["transitions"],
                    }
                    for target, reference in backtest_references.items()
                },
            },
            handle,
            indent=2,
        )

    summary_lines = [
        "# Parameter Regions",
        "",
        "The standard sweep fixes `receives_context=(True, True)` for all behaviors except `FF_un`.",
        "The `FF_un` sweep uses `receives_context=(False, False)` as requested.",
        "Under this constraint, `un_un` is not separately identifiable from `un_FB`; it is kept only as a diagnostic reference, not as a labeled region.",
        "",
        "## Region Counts",
        "",
    ]
    region_summary = pd.read_csv(save_path / "region_summary.csv") if (save_path / "region_summary.csv").exists() else pd.DataFrame()
    for row in region_summary.to_dict(orient="records"):
        summary_lines.append(
            (
                f"- `{row['target']}`: n={int(row['n_region_points'])}, "
                f"familiar=`{row.get('familiar_transition', 'NA')}`, "
                f"novel=`{row.get('novel_transition', 'NA')}`, "
                f"boundary success min={row.get('boundary_backtest_success_min', float('nan'))}"
            )
        )
    (save_path / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def run_all_region_searches(
    *,
    settings: SweepSettings = SweepSettings(),
    save_dir: str | Path | None = None,
) -> dict[str, Any]:
    compact_references = _target_reference_records_compact(settings, target_names=ALL_REGION_TARGETS)
    backtest_references = _target_reference_records_full(settings, target_names=ALL_REGION_TARGETS)
    search_compact_references = {
        target: compact_references[target]
        for target in SEARCH_REGION_TARGETS
    }
    search_backtest_references = {
        target: backtest_references[target]
        for target in SEARCH_REGION_TARGETS
    }

    standard_result = run_region_search_group(
        search_group="standard",
        allowed_targets=STANDARD_REGION_TARGETS,
        receives_context=(True, True),
        settings=settings,
        compact_references=search_compact_references,
        backtest_references=search_backtest_references,
    )
    ff_un_result = run_region_search_group(
        search_group="ff_un",
        allowed_targets=SPECIAL_REGION_TARGETS,
        receives_context=(False, False),
        settings=settings,
        compact_references=search_compact_references,
        backtest_references=search_backtest_references,
    )
    result = {
        "settings": settings,
        "compact_references": compact_references,
        "backtest_references": backtest_references,
        "groups": {
            "standard": standard_result,
            "ff_un": ff_un_result,
        },
    }
    if save_dir is not None:
        _save_region_results(
            result,
            compact_references=compact_references,
            backtest_references=backtest_references,
            settings=settings,
            save_dir=save_dir,
        )
    return result


def _print_console_summary(result: dict[str, Any]) -> None:
    for group_name, group in result["groups"].items():
        print(group_name)
        summary = group["region_summary"]
        for row in summary.to_dict(orient="records"):
            print(
                (
                    f"  {row['target']}: n={int(row['n_region_points'])}, "
                    f"familiar={row.get('familiar_transition')}, "
                    f"novel={row.get('novel_transition')}"
                )
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explore parameter regions for the minimal context-contrasting model and "
            "recover sets of initial weights that lead to the same named behavior."
        )
    )
    parser.add_argument("--n-steps", type=int, default=80, help="Compact simulator steps per phase.")
    parser.add_argument("--n-trials", type=int, default=6, help="Compact simulator trials per phase.")
    parser.add_argument("--tail-window", type=int, default=60, help="Compact simulator response tail length.")
    parser.add_argument("--global-samples", type=int, default=2048, help="Sobol samples per sweep.")
    parser.add_argument("--local-samples-per-target", type=int, default=192, help="Local perturbation samples around each reference target.")
    parser.add_argument("--local-sigma", type=float, default=0.22, help="Log10-space local perturbation standard deviation.")
    parser.add_argument("--region-gap-threshold", type=float, default=0.02, help="Minimum best-vs-second-best objective gap for the stable region filter.")
    parser.add_argument("--region-min-points", type=int, default=12, help="Minimum retained region points before relaxing the gap filter.")
    parser.add_argument("--backtest-full-steps", type=int, default=100, help="Full experiment steps per phase for boundary backtesting.")
    parser.add_argument("--backtest-tail-window", type=int, default=80, help="Tail window used when summarizing full backtests.")
    parser.add_argument("--backtest-seeds", nargs="+", type=int, default=[11, 23, 37], help="Seeds used for boundary backtesting.")
    parser.add_argument("--backtest-random-points", type=int, default=4, help="Extra random region points to include in boundary backtesting.")
    parser.add_argument("--weight-floor", type=float, default=1e-4, help="Minimum explored weight value.")
    parser.add_argument("--weight-ceiling", type=float, default=2.0, help="Maximum explored weight value.")
    parser.add_argument("--fit-scale", type=float, default=DEFAULT_ACTIVITY_THRESHOLD, help="Scale used when normalizing response RMSE.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for Sobol and local sampling.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel worker count.")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="context_contrasting/sbi/results/latest_regions",
        help="Directory where CSV, JSON, and Markdown outputs are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = SweepSettings(
        n_steps_per_phase=args.n_steps,
        n_trials=args.n_trials,
        tail_window=args.tail_window,
        global_samples=args.global_samples,
        local_samples_per_target=args.local_samples_per_target,
        local_sigma=args.local_sigma,
        n_jobs=args.n_jobs,
        random_seed=args.seed,
        weight_floor=args.weight_floor,
        weight_ceiling=args.weight_ceiling,
        fit_scale=args.fit_scale,
        region_gap_threshold=args.region_gap_threshold,
        region_min_points=args.region_min_points,
        backtest_full_steps=args.backtest_full_steps,
        backtest_tail_window=args.backtest_tail_window,
        backtest_seeds=tuple(args.backtest_seeds),
        backtest_random_points=args.backtest_random_points,
    )
    result = run_all_region_searches(
        settings=settings,
        save_dir=args.save_dir,
    )
    _print_console_summary(result)


if __name__ == "__main__":
    main()
