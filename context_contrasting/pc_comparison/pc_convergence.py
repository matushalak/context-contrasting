from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from context_contrasting.paper.model_scatter import _training_trial_order
from context_contrasting.pc_comparison.pc_templates import (
    LEARNING_RATE_REFERENCE_STEPS,
    N_FEATURES,
    scaled_learning_rate,
)


FAMILIAR_FEATURES = (0, 1)


@dataclass(frozen=True)
class ConvergenceSummary:
    reference_learning_rate: float
    n_steps_per_phase: int
    effective_learning_rate: float
    max_abs_prediction_error: float
    mean_abs_prediction_error: float
    converged_fraction: float
    tolerance: float

    @property
    def converged(self) -> bool:
        return self.max_abs_prediction_error <= self.tolerance


def _matrix(configs: pd.DataFrame, prefix: str) -> np.ndarray:
    return configs[[f"{prefix}.mu_{idx}" for idx in range(N_FEATURES)]].to_numpy(dtype=np.float32)


def simulate_training(
    configs: pd.DataFrame,
    *,
    reference_learning_rate: float,
    n_steps_per_phase: int,
    training_trials: int,
    training_stimulus_order: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized form of the canonical PC update for matched familiar inputs."""
    if configs.empty:
        raise ValueError("configs must not be empty.")
    if training_trials <= 0 or n_steps_per_phase <= 0:
        raise ValueError("training_trials and n_steps_per_phase must be positive.")

    pyc_weights = _matrix(configs, "pyc_excitatory_init").copy()
    pv_weights = _matrix(configs, "pv_excitatory_init")
    w_lat = configs["w_lat_init.mu_0"].to_numpy(dtype=np.float32)
    pv_decay = configs["pv_decay"].to_numpy(dtype=np.float32)
    pv_activity = np.zeros(len(configs), dtype=np.float32)
    effective_rate = np.float32(scaled_learning_rate(reference_learning_rate, n_steps_per_phase))
    active_steps = n_steps_per_phase - 3 * n_steps_per_phase // 4
    iti_steps = n_steps_per_phase - active_steps
    trial_order = _training_trial_order(
        [f"familiar_{idx + 1}" for idx in FAMILIAR_FEATURES],
        n_trials=training_trials,
        order=training_stimulus_order,
        seed=seed,
    )

    for trial_name in trial_order:
        feature_idx = int(trial_name.rsplit("_", 1)[1]) - 1
        pv_activity *= np.power(1.0 - pv_decay, iti_steps)
        for _ in range(active_steps):
            pv_drive = np.clip(pv_weights[:, feature_idx], 0.0, 1.0)
            pv_activity = (1.0 - pv_decay) * pv_activity + pv_decay * pv_drive
            prediction_error = pyc_weights[:, feature_idx] - w_lat * pv_activity
            raw_delta = -effective_rate * prediction_error
            current = pyc_weights[:, feature_idx]
            bounded_delta = np.where(raw_delta >= 0.0, raw_delta * (1.0 - current), raw_delta * current)
            pyc_weights[:, feature_idx] = np.maximum(current + bounded_delta, 0.0)

    steady_pv = np.clip(pv_weights[:, FAMILIAR_FEATURES], 0.0, 1.0)
    errors = pyc_weights[:, FAMILIAR_FEATURES] - w_lat[:, None] * steady_pv
    return pyc_weights, errors


def convergence_summary(
    configs: pd.DataFrame,
    *,
    reference_learning_rate: float,
    n_steps_per_phase: int = LEARNING_RATE_REFERENCE_STEPS,
    training_trials: int = 7,
    training_stimulus_order: str = "randomized",
    seed: int = 7151,
    tolerance: float = 1e-3,
) -> ConvergenceSummary:
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    _, errors = simulate_training(
        configs,
        reference_learning_rate=reference_learning_rate,
        n_steps_per_phase=n_steps_per_phase,
        training_trials=training_trials,
        training_stimulus_order=training_stimulus_order,
        seed=seed,
    )
    per_cell = np.max(np.abs(errors), axis=1)
    return ConvergenceSummary(
        reference_learning_rate=float(reference_learning_rate),
        n_steps_per_phase=int(n_steps_per_phase),
        effective_learning_rate=scaled_learning_rate(reference_learning_rate, n_steps_per_phase),
        max_abs_prediction_error=float(per_cell.max()),
        mean_abs_prediction_error=float(np.mean(np.abs(errors))),
        converged_fraction=float(np.mean(per_cell <= tolerance)),
        tolerance=float(tolerance),
    )


def find_minimum_learning_rate(
    configs: pd.DataFrame,
    *,
    n_steps_per_phase: int = LEARNING_RATE_REFERENCE_STEPS,
    training_trials: int = 7,
    training_stimulus_order: str = "randomized",
    seed: int = 7151,
    tolerance: float = 1e-3,
    lower: float = 0.0,
    upper: float = 1.0,
    rate_tolerance: float = 1e-8,
) -> ConvergenceSummary:
    """Find the smallest reference rate that passes a finite convergence tolerance."""
    if lower < 0.0 or upper <= lower:
        raise ValueError("learning-rate bounds must satisfy 0 <= lower < upper.")
    if rate_tolerance <= 0.0:
        raise ValueError("rate_tolerance must be positive.")

    summary_kwargs = {
        "configs": configs,
        "n_steps_per_phase": n_steps_per_phase,
        "training_trials": training_trials,
        "training_stimulus_order": training_stimulus_order,
        "seed": seed,
        "tolerance": tolerance,
    }
    lower_summary = convergence_summary(reference_learning_rate=lower, **summary_kwargs)
    if lower_summary.converged:
        return lower_summary

    scan_start = max(lower, 1e-10)
    scan_rates = np.geomspace(scan_start, upper, num=257)
    bracket: tuple[float, ConvergenceSummary, float, ConvergenceSummary] | None = None
    previous_rate = lower
    previous_summary = lower_summary
    for candidate_rate in scan_rates:
        candidate_summary = convergence_summary(
            reference_learning_rate=float(candidate_rate),
            **summary_kwargs,
        )
        if candidate_summary.converged:
            bracket = (previous_rate, previous_summary, float(candidate_rate), candidate_summary)
            break
        previous_rate = float(candidate_rate)
        previous_summary = candidate_summary
    if bracket is None:
        raise ValueError(f"no convergent learning rate found in [{lower:g}, {upper:g}].")

    lower, lower_summary, upper, upper_summary = bracket

    while upper - lower > rate_tolerance:
        midpoint = (lower + upper) / 2.0
        midpoint_summary = convergence_summary(reference_learning_rate=midpoint, **summary_kwargs)
        if midpoint_summary.converged:
            upper = midpoint
            upper_summary = midpoint_summary
        else:
            lower = midpoint
            lower_summary = midpoint_summary
    return upper_summary


def convergence_rows(
    configs: pd.DataFrame,
    *,
    reference_learning_rate: float,
    n_steps_per_phase: int,
    training_trials: int,
    training_stimulus_order: str,
    seed: int,
    tolerance: float,
) -> pd.DataFrame:
    final_weights, errors = simulate_training(
        configs,
        reference_learning_rate=reference_learning_rate,
        n_steps_per_phase=n_steps_per_phase,
        training_trials=training_trials,
        training_stimulus_order=training_stimulus_order,
        seed=seed,
    )
    rows = configs[["sample_global_idx", "transition"]].copy()
    for feature_idx in range(N_FEATURES):
        rows[f"final_pyc_excitatory_{feature_idx}"] = final_weights[:, feature_idx]
    for local_idx, feature_idx in enumerate(FAMILIAR_FEATURES):
        rows[f"familiar_{feature_idx + 1}_prediction_error"] = errors[:, local_idx]
    rows["max_abs_prediction_error"] = np.max(np.abs(errors), axis=1)
    rows["converged"] = rows["max_abs_prediction_error"] <= tolerance
    return rows
