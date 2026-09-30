from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


N_FEATURES = 3
LEARNING_RATE_REFERENCE_STEPS = 400
DEFAULT_CONVERGENCE_TOLERANCE = 0.005
DEFAULT_BASELINE_DRIVE = 0.2
DEFAULT_BASELINE_DRIVE_SIGMA = 0.30
# Calibrated for DEFAULT_PARAMETER_SPACE with seed 7151, 300 samples,
# 400 steps/phase and seven trials per familiar image at |PE| <= 0.005.
FULL_PROTOCOL_CALIBRATED_LEARNING_RATE = 0.2820285747289002


@dataclass(frozen=True)
class PCParameterSpace:
    """Circuit-independent search space for matched PPE/NPE populations."""

    pyc_weight_min: float = 0.05
    pyc_weight_max: float = 0.95
    pv_weight_min: float = 0.05
    pv_weight_max: float = 0.95
    w_lat_min: float = 0.05
    w_lat_max: float = 0.95
    pyc_tuning_widths: tuple[int, ...] = (1, 3)
    pv_tuning_widths: tuple[int, ...] = (1, 3)
    untuned_weight: float = 0.0

    def validate(self) -> None:
        for name in (
            "pyc_weight_min",
            "pyc_weight_max",
            "pv_weight_min",
            "pv_weight_max",
            "w_lat_min",
            "w_lat_max",
            "untuned_weight",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        for lo_name, hi_name in (
            ("pyc_weight_min", "pyc_weight_max"),
            ("pv_weight_min", "pv_weight_max"),
            ("w_lat_min", "w_lat_max"),
        ):
            if float(getattr(self, lo_name)) > float(getattr(self, hi_name)):
                raise ValueError(f"{lo_name} must not exceed {hi_name}.")
        for widths_name in ("pyc_tuning_widths", "pv_tuning_widths"):
            widths = tuple(int(width) for width in getattr(self, widths_name))
            if not widths or any(width < 1 or width > N_FEATURES for width in widths):
                raise ValueError(f"{widths_name} must contain widths from 1 to {N_FEATURES}.")


DEFAULT_PARAMETER_SPACE = PCParameterSpace()


def scaled_learning_rate(
    learning_rate: float,
    n_steps_per_phase: int,
    *,
    reference_steps: int = LEARNING_RATE_REFERENCE_STEPS,
) -> float:
    """Scale a full-protocol rate to preserve updates per stimulus horizon."""
    if learning_rate < 0.0:
        raise ValueError("learning_rate must be nonnegative.")
    if n_steps_per_phase <= 0 or reference_steps <= 0:
        raise ValueError("step counts must be positive.")
    return float(learning_rate) * float(reference_steps) / float(n_steps_per_phase)


def _draw_indices(width: int, rng: np.random.Generator) -> tuple[int, ...]:
    return tuple(sorted(int(idx) for idx in rng.choice(N_FEATURES, size=width, replace=False)))


def _tuned_vector(
    tuned_weight: float,
    tuned_indices: tuple[int, ...],
    *,
    untuned_weight: float,
) -> np.ndarray:
    values = np.full(N_FEATURES, float(untuned_weight), dtype=float)
    values[list(tuned_indices)] = float(tuned_weight)
    return values


def _write_init_columns(row: dict[str, Any], key: str, values: np.ndarray) -> None:
    for idx, value in enumerate(np.asarray(values, dtype=float).reshape(-1)):
        row[f"{key}.mu_{idx}"] = float(value)
    row[f"{key}.sigma"] = 0.0


def _stratified_uniform(
    rng: np.random.Generator,
    n_samples: int,
    low: float,
    high: float,
) -> np.ndarray:
    if n_samples == 1:
        return np.asarray([(low + high) / 2.0], dtype=float)
    bins = (np.arange(n_samples, dtype=float) + rng.random(n_samples)) / n_samples
    rng.shuffle(bins)
    return low + bins * (high - low)


def sample_shared_pc_configs(
    *,
    n_samples: int,
    seed: int,
    n_steps_per_phase: int,
    learning_rate: float = 0.0,
    parameter_space: PCParameterSpace = DEFAULT_PARAMETER_SPACE,
) -> pd.DataFrame:
    """Sample one parameter pool that is instantiated as both PPE and NPE.

    Sampling is stratified independently along the three continuous dimensions.
    The resulting rows are circuit-free: the circuit mapping happens only when
    a model is instantiated.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    parameter_space.validate()
    rng = np.random.default_rng(seed)
    pyc_weights = _stratified_uniform(
        rng,
        n_samples,
        parameter_space.pyc_weight_min,
        parameter_space.pyc_weight_max,
    )
    pv_weights = _stratified_uniform(
        rng,
        n_samples,
        parameter_space.pv_weight_min,
        parameter_space.pv_weight_max,
    )
    w_lats = _stratified_uniform(
        rng,
        n_samples,
        parameter_space.w_lat_min,
        parameter_space.w_lat_max,
    )
    effective_rate = scaled_learning_rate(learning_rate, n_steps_per_phase)

    rows: list[dict[str, Any]] = []
    for sample_idx in range(n_samples):
        pyc_width = int(rng.choice(parameter_space.pyc_tuning_widths))
        pv_width = int(rng.choice(parameter_space.pv_tuning_widths))
        pyc_indices = _draw_indices(pyc_width, rng)
        pv_indices = _draw_indices(pv_width, rng)
        pyc_vector = _tuned_vector(
            float(pyc_weights[sample_idx]),
            pyc_indices,
            untuned_weight=parameter_space.untuned_weight,
        )
        pv_vector = _tuned_vector(
            float(pv_weights[sample_idx]),
            pv_indices,
            untuned_weight=parameter_space.untuned_weight,
        )
        tuning_label = f"pyc_{pyc_width}of{N_FEATURES}_pv_{pv_width}of{N_FEATURES}"
        row: dict[str, Any] = {
            "transition": tuning_label,
            "sample_idx": sample_idx + 1,
            "sample_global_idx": sample_idx + 1,
            "seed": seed + sample_idx + 1,
            "n_features": N_FEATURES,
            "n_pv": 1,
            "n_context": N_FEATURES,
            "learning_rate": effective_rate,
            "reference_learning_rate": float(learning_rate),
            "learning_rate_reference_steps": LEARNING_RATE_REFERENCE_STEPS,
            "pyc_decay": 0.05,
            "pv_decay": 0.5,
            "baseline_drive_mu": DEFAULT_BASELINE_DRIVE,
            "baseline_drive_sigma": DEFAULT_BASELINE_DRIVE_SIGMA,
            "pv_noise_sigma": 0.0,
            "pc_pyc_tuning_width": pyc_width,
            "pc_pv_tuning_width": pv_width,
            "pyc_tuned_weight": float(pyc_weights[sample_idx]),
            "pv_tuned_weight": float(pv_weights[sample_idx]),
            "w_lat_scalar": float(w_lats[sample_idx]),
        }
        _write_init_columns(row, "pyc_excitatory_init", pyc_vector)
        _write_init_columns(row, "pv_excitatory_init", pv_vector)
        _write_init_columns(row, "w_lat_init", np.asarray([w_lats[sample_idx]], dtype=float))
        for idx in range(N_FEATURES):
            row[f"pyc_tuned_index_{idx}"] = int(idx in pyc_indices)
            row[f"pv_tuned_index_{idx}"] = int(idx in pv_indices)
        rows.append(row)
    return pd.DataFrame(rows)


def parameter_space_metadata(parameter_space: PCParameterSpace = DEFAULT_PARAMETER_SPACE) -> dict[str, Any]:
    metadata = asdict(parameter_space)
    metadata["pyc_tuning_widths"] = list(parameter_space.pyc_tuning_widths)
    metadata["pv_tuning_widths"] = list(parameter_space.pv_tuning_widths)
    return metadata
