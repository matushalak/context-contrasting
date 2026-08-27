"""Fit real pooled sector traces with `sbi` NPE.

This fits one model cell each for +NO, +O, and -NO familiar-sector targets from
the real pooled familiar/novel traces. It trains an NPE posterior with the
installed `sbi` package, samples candidate parameters at the observed real trace,
then reruns posterior samples through the simulator and keeps the best
stimulus-window match for ablation evaluation.
"""

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
from sbi.inference import NPE_C
from sbi.utils import BoxUniform

from context_contrasting.paper import transitions_helpers as th
from context_contrasting.paper.model_scatter import _run_sector_average_panel_config


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_RUN_DIR = PACKAGE_DIR / "done-amen-final"
DEFAULT_TARGET_CSV = (
    REPO_ROOT
    / "context_contrasting"
    / "data_analysis"
    / "familiar_sector_novel_transfer"
    / "familiar_sector_novel_transfer_pooled_trace_summary.csv"
)
DEFAULT_TARGET_EXAMPLE_TRACE_CSV = REPO_ROOT / "context_contrasting" / "data_analysis" / "transitions_post_traces.csv"
DEFAULT_TARGET_EXAMPLE_SUMMARY_CSV = (
    REPO_ROOT
    / "context_contrasting"
    / "data_analysis"
    / "familiar_sector_novel_transfer"
    / "familiar_sector_novel_transfer_neuron_summary.csv"
)
DEFAULT_OUTPUT_NAME = "sbi_npe_real_sector_trace_fit"
DEFAULT_HOT_START_SUMMARY_CSV = DEFAULT_RUN_DIR / "summaries" / "aggregate_familiar_summary.csv"
DEFAULT_FIT_WINDOW = (0.0, 1.0)
FIT_SECTORS = ("+NO axis", "+O axis", "-NO axis")
FORCE_FEEDBACK_SECTORS = ("+NO axis", "+O axis")
FREE_APICAL_GAIN_STRENGTH_SECTORS = ("+NO axis",)
FINAL_COLUMN_KEYS = ("naive", "expert", "expert_no_fb", "expert_no_lat", "expert_no_fb_no_lat")
COLUMN_LABELS = {
    "naive": "Naive",
    "expert": "Expert",
    "expert_no_fb": "FB off",
    "expert_no_lat": "PV off",
    "expert_no_fb_no_lat": "FB+PV off",
}
GROUP_LABELS = {"familiar": "Familiar", "novel": "Novel"}
RESPONSE_COLORS = {"NO": "black", "O": "red"}
REAL_RESPONSE_LABEL = {"Full": "NO", "Occl": "O"}
MODEL_TO_REAL_STAGE = {"naive": "Pre", "expert": "Post"}
FIT_PANEL_KEYS_ALL = (
    ("familiar", "Pre", "NO"),
    ("familiar", "Pre", "O"),
    ("familiar", "Post", "NO"),
    ("familiar", "Post", "O"),
    ("novel", "Pre", "NO"),
    ("novel", "Pre", "O"),
    ("novel", "Post", "NO"),
    ("novel", "Post", "O"),
)
FIT_PANEL_KEYS_FAMILIAR = FIT_PANEL_KEYS_ALL[:4]
FIT_PANEL_KEY_SETS = {
    "all": FIT_PANEL_KEYS_ALL,
    "familiar": FIT_PANEL_KEYS_FAMILIAR,
}
THETA_SPECS = (
    ("w_ff_0", 0.0, 0.7),
    ("w_ff_1", 0.0, 0.7),
    ("w_ff_2", 0.0, 0.7),
    ("w_fb_0", 0.0, 0.9),
    ("w_fb_1", 0.0, 0.9),
    ("w_fb_2", 0.0, 0.9),
    ("w_lat", 0.0, 1.0),
    ("w_pv_lat", 0.0, 0.5),
    ("W_pv_0", 0.0, 1.0),
    ("W_pv_1", 0.0, 1.0),
    ("W_pv_2", 0.0, 1.0),
    ("apical_drive_threshold", 0.0, 1.6),
    ("apical_gain_strength", 1.0, 12.0),
)

THETA_LOW = torch.tensor([spec[1] for spec in THETA_SPECS], dtype=torch.float32)
THETA_HIGH = torch.tensor([spec[2] for spec in THETA_SPECS], dtype=torch.float32)


@dataclass(frozen=True)
class FitSettings:
    n_steps_per_phase: int
    test_trials: int
    training_trials: int
    training_stimulus_order: str
    seed: int
    zscore_std_floor: float


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


def _load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], FitSettings]:
    metadata = json.loads((run_dir / "metadata.json").read_text())
    configs = json.loads((run_dir / "sampled_configs.json").read_text())
    base_config = copy.deepcopy(configs[0])
    base_config.pop("activation", None)
    settings = FitSettings(
        n_steps_per_phase=int(metadata["n_steps_per_phase"]),
        test_trials=int(metadata["test_trials"]),
        training_trials=int(metadata["training_trials"]),
        training_stimulus_order=str(metadata["training_stimulus_order"]),
        seed=int(metadata["seed"]),
        zscore_std_floor=float(metadata.get("zscore_std_floor", 0.04)),
    )
    return base_config, metadata, settings


def _theta_prior() -> BoxUniform:
    return BoxUniform(low=THETA_LOW, high=THETA_HIGH)


def _project_theta_to_prior(theta: torch.Tensor) -> torch.Tensor:
    low = THETA_LOW.to(dtype=theta.dtype, device=theta.device)
    high = THETA_HIGH.to(dtype=theta.dtype, device=theta.device)
    eps = torch.finfo(theta.dtype).eps * 1024
    return torch.minimum(torch.maximum(theta, low + eps), high - eps)


def _clip_scalar_to_theta_range(name: str, value: float) -> float:
    for spec_name, low, high in THETA_SPECS:
        if spec_name == name:
            eps = np.finfo(np.float32).eps * 1024
            return float(np.clip(value, low + eps, high - eps))
    raise KeyError(name)


def _theta_to_dict(theta: torch.Tensor | np.ndarray) -> dict[str, float]:
    values = np.asarray(theta.detach().cpu().numpy() if isinstance(theta, torch.Tensor) else theta, dtype=float).reshape(-1)
    return {name: float(values[idx]) for idx, (name, _lo, _hi) in enumerate(THETA_SPECS)}


def _theta_from_config(config: dict[str, Any], *, sector: str) -> torch.Tensor:
    receives_context = [True, True, True] if sector in FORCE_FEEDBACK_SECTORS else list(config.get("receives_context", [True, True, True]))
    values = {
        "log10_lr_ff": np.log10(float(config["lr_ff"])),
        "log10_lr_fb": np.log10(float(config["lr_fb"])),
        "log10_lr_lat": np.log10(float(config["lr_lat"])),
        "log10_ff_accumulator_scale": np.log10(float(config["ff_accumulator_scale"])),
        "w_lat": float(np.asarray(config["w_lat_init"]["mu"], dtype=float).reshape(-1)[0]),
        "w_pv_lat": float(np.asarray(config["w_pv_lat_init"]["mu"], dtype=float).reshape(-1)[0]),
        "apical_drive_threshold": float(config["apical_drive_threshold"]),
        "apical_gain_strength": float(config["apical_gain_strength"]),
        "apical_gain_k": float(config["apical_gain_k"]),
        "baseline_drive_sigma": float(config["baseline_drive_sigma"]),
        "divisive_gain": float(config["divisive_gain"]),
    }
    for prefix, init_key in (("w_ff", "w_ff_init"), ("w_fb", "w_fb_init"), ("W_pv", "W_pv_init")):
        for idx, value in enumerate(np.asarray(config[init_key]["mu"], dtype=float).reshape(-1)[:3]):
            values[f"{prefix}_{idx}"] = float(value)
    for idx, enabled in enumerate(receives_context[:3]):
        values[f"context_{idx}"] = 1.0 if bool(enabled) else 0.0
    return torch.tensor(
        [_clip_scalar_to_theta_range(name, values[name]) for name, _low, _high in THETA_SPECS],
        dtype=torch.float32,
    )


def _load_sector_hot_start_thetas(
    *,
    run_dir: Path,
    summary_csv: Path,
    sector: str,
    n_requested: int,
    seed: int,
) -> tuple[torch.Tensor | None, list[int]]:
    if n_requested <= 0:
        return None, []
    summary = pd.read_csv(summary_csv)
    rows = summary.loc[summary["RotatedSector"].eq(sector)].copy()
    if rows.empty:
        return None, []
    rng = np.random.default_rng(seed)
    if len(rows) > n_requested:
        chosen_positions = rng.choice(len(rows), size=n_requested, replace=False)
        rows = rows.iloc[np.sort(chosen_positions)].copy()
    configs = json.loads((run_dir / "sampled_configs.json").read_text())
    configs_by_global_idx = {int(config["_sample_global_idx"]): config for config in configs}
    source_indices = [int(value) for value in rows["neuron_idx"].to_list()]
    theta_rows = [
        _theta_from_config(configs_by_global_idx[global_idx], sector=sector)
        for global_idx in source_indices
        if global_idx in configs_by_global_idx
    ]
    if not theta_rows:
        return None, []
    return torch.stack(theta_rows), source_indices[: len(theta_rows)]


def _receives_context(values: dict[str, float], *, sector: str) -> list[bool]:
    if sector in FORCE_FEEDBACK_SECTORS:
        return [True, True, True]
    if not all(f"context_{idx}" in values for idx in range(3)):
        return [False, False, False]
    return [values[f"context_{idx}"] >= 0.5 for idx in range(3)]


def _config_from_theta(
    theta: torch.Tensor | np.ndarray,
    *,
    base_config: dict[str, Any],
    sector: str,
    cell_id: int,
    seed: int,
) -> dict[str, Any]:
    values = _theta_to_dict(theta)
    config = copy.deepcopy(base_config)
    config.update(
        {
            "w_ff_init": {"mu": [values[f"w_ff_{idx}"] for idx in range(3)], "sigma": 0.0},
            "w_fb_init": {"mu": [values[f"w_fb_{idx}"] for idx in range(3)], "sigma": 0.0},
            "w_lat_init": {"mu": [values["w_lat"]], "sigma": 0.0},
            "w_pv_lat_init": {"mu": [values["w_pv_lat"]], "sigma": 0.0},
            "W_pv_init": {"mu": [values[f"W_pv_{idx}"] for idx in range(3)], "sigma": 0.0},
            "apical_drive_threshold": values["apical_drive_threshold"],
            "receives_context": _receives_context(values, sector=sector),
            "seed": int(seed),
            "_canonical_transition": f"sbi_npe_{_slugify(sector)}",
            "_sample_idx": int(cell_id),
            "_sample_global_idx": int(cell_id),
        }
    )
    if sector in FREE_APICAL_GAIN_STRENGTH_SECTORS:
        config["apical_gain_strength"] = values["apical_gain_strength"]
    config.pop("activation", None)
    return config


def _load_targets(target_csv: Path, *, sectors: tuple[str, ...]) -> pd.DataFrame:
    target = pd.read_csv(target_csv)
    target = target.loc[target["RotatedSector"].isin(sectors)].copy()
    target = target.loc[target["image_group"].isin(["familiar", "novel"])].copy()
    target = target.loc[target["stage"].isin(["Pre", "Post"])].copy()
    target = target.loc[target["image_type"].isin(["Full", "Occl"])].copy()
    target["response_type"] = target["image_type"].map(REAL_RESPONSE_LABEL)
    target["sem"] = target["sem"].fillna(0.0)
    return target


def _load_target_examples(
    trace_csv: Path,
    summary_csv: Path,
    *,
    sectors: tuple[str, ...],
) -> pd.DataFrame:
    trace = pd.read_csv(trace_csv)
    summary = pd.read_csv(summary_csv)
    sector_lookup = summary[["neuron_idx", "RotatedSector_familiar"]].rename(
        columns={"RotatedSector_familiar": "RotatedSector"}
    )
    examples = trace.merge(sector_lookup, on="neuron_idx", how="inner", validate="many_to_one")
    examples = examples.loc[examples["RotatedSector"].isin(sectors)].copy()
    examples = examples.loc[examples["image_group"].isin(["familiar", "novel"])].copy()
    examples = examples.loc[examples["stage"].isin(["Pre", "Post"])].copy()
    examples = examples.loc[examples["image_type"].isin(["Full", "Occl"])].copy()
    examples["response_type"] = examples["image_type"].map(REAL_RESPONSE_LABEL)
    index_cols = ["RotatedSector", "neuron_idx", "image_group", "stage", "image_type", "response_type", "time"]
    return (
        examples.groupby(index_cols, as_index=False, observed=True)
        .agg(mean_response=("response", "mean"))
        .sort_values(index_cols)
        .reset_index(drop=True)
    )


def _window_mask(values: np.ndarray, fit_window: tuple[float, float] | None) -> np.ndarray:
    if fit_window is None:
        return np.ones(values.shape, dtype=bool)
    start, end = fit_window
    return (values >= start) & (values <= end)


def _target_time_grid(
    target: pd.DataFrame,
    fit_timepoints: int | None,
    *,
    fit_window: tuple[float, float] | None,
) -> np.ndarray:
    full_grid = np.sort(target["time"].drop_duplicates().to_numpy(dtype=float))
    full_grid = full_grid[_window_mask(full_grid, fit_window)]
    if len(full_grid) == 0:
        raise ValueError(f"No target timepoints found in fit window {fit_window}.")
    if fit_timepoints is None or fit_timepoints <= 0 or fit_timepoints >= len(full_grid):
        return full_grid
    idx = np.unique(np.round(np.linspace(0, len(full_grid) - 1, fit_timepoints)).astype(int))
    return full_grid[idx]


def _stimulus_summary_features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.asarray([100.0, 100.0, 100.0, 100.0], dtype=float)
    tail_start = max(0, int(np.floor(0.75 * len(values))))
    return np.asarray(
        [
            float(np.mean(values)),
            float(np.max(values)),
            float(np.min(values)),
            float(np.mean(values[tail_start:])),
        ],
        dtype=float,
    )


def _target_vector(
    target: pd.DataFrame,
    *,
    sector: str,
    time_grid: np.ndarray,
    panel_keys: tuple[tuple[str, str, str], ...],
    fit_representation: str,
) -> torch.Tensor:
    sector_target = target.loc[target["RotatedSector"].eq(sector)]
    values: list[np.ndarray] = []
    for image_group, stage, response_type in panel_keys:
        rows = sector_target.loc[
            sector_target["image_group"].eq(image_group)
            & sector_target["stage"].eq(stage)
            & sector_target["response_type"].eq(response_type)
        ].sort_values("time")
        panel_values = np.interp(time_grid, rows["time"].to_numpy(dtype=float), rows["mean_response"].to_numpy(dtype=float))
        if fit_representation == "summary":
            panel_values = _stimulus_summary_features(panel_values)
        values.append(panel_values)
    return torch.as_tensor(np.concatenate(values), dtype=torch.float32)


def _target_example_vectors(
    target_examples: pd.DataFrame,
    *,
    sector: str,
    time_grid: np.ndarray,
    panel_keys: tuple[tuple[str, str, str], ...],
    fit_representation: str,
) -> tuple[torch.Tensor, np.ndarray]:
    sector_examples = target_examples.loc[target_examples["RotatedSector"].eq(sector)]
    vectors: list[np.ndarray] = []
    neuron_ids: list[int] = []
    for neuron_idx, neuron_rows in sector_examples.groupby("neuron_idx", sort=True):
        panel_values: list[np.ndarray] = []
        complete = True
        for image_group, stage, response_type in panel_keys:
            rows = neuron_rows.loc[
                neuron_rows["image_group"].eq(image_group)
                & neuron_rows["stage"].eq(stage)
                & neuron_rows["response_type"].eq(response_type)
            ].sort_values("time")
            if rows.empty:
                complete = False
                break
            values = np.interp(time_grid, rows["time"].to_numpy(dtype=float), rows["mean_response"].to_numpy(dtype=float))
            if fit_representation == "summary":
                values = _stimulus_summary_features(values)
            panel_values.append(values)
        if complete:
            vectors.append(np.concatenate(panel_values))
            neuron_ids.append(int(neuron_idx))
    if not vectors:
        raise ValueError(f"No complete target examples found for {sector}.")
    return torch.as_tensor(np.vstack(vectors), dtype=torch.float32), np.asarray(neuron_ids, dtype=int)


def _score_vectors_against_examples(
    vectors: torch.Tensor,
    target_vectors: torch.Tensor,
    *,
    n_panels: int,
    example_score: str,
) -> torch.Tensor:
    pred = vectors.to(dtype=torch.float32)
    target = target_vectors.to(dtype=torch.float32)
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    if target.ndim == 1:
        target = target.reshape(1, -1)
    if pred.shape[1] != target.shape[1]:
        raise ValueError(f"Vector dimension mismatch: {pred.shape[1]} vs {target.shape[1]}")
    if pred.shape[1] % n_panels != 0:
        raise ValueError(f"Vector dimension {pred.shape[1]} is not divisible by {n_panels} panels.")
    width = pred.shape[1] // n_panels
    diff = pred.reshape(pred.shape[0], 1, n_panels, width) - target.reshape(1, target.shape[0], n_panels, width)
    per_example = torch.sqrt(torch.mean(diff**2, dim=(2, 3)))
    if example_score == "mean":
        return torch.mean(per_example, dim=1)
    if example_score == "median":
        return torch.median(per_example, dim=1).values
    if example_score == "min":
        return torch.min(per_example, dim=1).values
    raise ValueError(f"Unknown example score: {example_score}")


def _run_config_trace(config: dict[str, Any], settings: FitSettings) -> pd.DataFrame:
    _, trace_df, _ = _run_sector_average_panel_config(
        str(config["_canonical_transition"]),
        config,
        n_steps_per_phase=settings.n_steps_per_phase,
        test_trials=settings.test_trials,
        training_trials=settings.training_trials,
        training_stimulus_order=settings.training_stimulus_order,
        seed=settings.seed,
        zscore_std_floor=settings.zscore_std_floor,
    )
    return trace_df


def _pool_model_trace(trace_df: pd.DataFrame) -> pd.DataFrame:
    pooled = trace_df.loc[trace_df["column_key"].isin(FINAL_COLUMN_KEYS)].copy()
    pooled["image_group"] = np.where(pooled["condition"].isin(["familiar_1", "familiar_2"]), "familiar", "novel")
    return (
        pooled.groupby(["image_group", "column_key", "response_type", "x_seconds"], as_index=False)
        .agg(y=("y", "mean"))
        .sort_values(["image_group", "column_key", "response_type", "x_seconds"])
        .reset_index(drop=True)
    )


def _model_vector(
    pooled_model: pd.DataFrame,
    *,
    time_grid: np.ndarray,
    panel_keys: tuple[tuple[str, str, str], ...],
    fit_representation: str,
) -> torch.Tensor:
    values: list[np.ndarray] = []
    for image_group, stage, response_type in panel_keys:
        column_key = "naive" if stage == "Pre" else "expert"
        rows = pooled_model.loc[
            pooled_model["image_group"].eq(image_group)
            & pooled_model["column_key"].eq(column_key)
            & pooled_model["response_type"].eq(response_type)
        ].sort_values("x_seconds")
        if rows.empty:
            width = 4 if fit_representation == "summary" else len(time_grid)
            return torch.full((width * len(panel_keys),), 100.0, dtype=torch.float32)
        panel_values = np.interp(time_grid, rows["x_seconds"].to_numpy(dtype=float), rows["y"].to_numpy(dtype=float))
        if fit_representation == "summary":
            panel_values = _stimulus_summary_features(panel_values)
        values.append(panel_values)
    vector = np.concatenate(values)
    if not np.all(np.isfinite(vector)):
        vector = np.full_like(vector, 100.0)
    return torch.as_tensor(vector, dtype=torch.float32)


def _simulate_theta(
    theta: torch.Tensor,
    *,
    base_config: dict[str, Any],
    sector: str,
    cell_id: int,
    seed: int,
    settings: FitSettings,
    time_grid: np.ndarray,
    panel_keys: tuple[tuple[str, str, str], ...],
    fit_representation: str,
) -> torch.Tensor:
    try:
        config = _config_from_theta(theta, base_config=base_config, sector=sector, cell_id=cell_id, seed=seed)
        trace = _run_config_trace(config, settings)
        return _model_vector(
            _pool_model_trace(trace),
            time_grid=time_grid,
            panel_keys=panel_keys,
            fit_representation=fit_representation,
        )
    except Exception:
        width = 4 if fit_representation == "summary" else len(time_grid)
        return torch.full((width * len(panel_keys),), 100.0, dtype=torch.float32)


def _trace_rmse(
    pooled_model: pd.DataFrame,
    target: pd.DataFrame,
    *,
    sector: str,
    sem_floor: float,
    panel_keys: tuple[tuple[str, str, str], ...],
    fit_window: tuple[float, float] | None,
    weighting: str,
) -> float:
    sector_target = target.loc[target["RotatedSector"].eq(sector)]
    sq: list[np.ndarray] = []
    wt: list[np.ndarray] = []
    panel_mse: list[float] = []
    for image_group, stage, response_type in panel_keys:
        column_key = "naive" if stage == "Pre" else "expert"
        model_rows = pooled_model.loc[
            pooled_model["image_group"].eq(image_group)
            & pooled_model["column_key"].eq(column_key)
            & pooled_model["response_type"].eq(response_type)
        ].sort_values("x_seconds")
        target_rows = sector_target.loc[
            sector_target["image_group"].eq(image_group)
            & sector_target["stage"].eq(stage)
            & sector_target["response_type"].eq(response_type)
        ].sort_values("time")
        if fit_window is not None:
            times = target_rows["time"].to_numpy(dtype=float)
            target_rows = target_rows.loc[_window_mask(times, fit_window)]
        if target_rows.empty or model_rows.empty:
            return 100.0
        pred = np.interp(
            target_rows["time"].to_numpy(dtype=float),
            model_rows["x_seconds"].to_numpy(dtype=float),
            model_rows["y"].to_numpy(dtype=float),
        )
        obs = target_rows["mean_response"].to_numpy(dtype=float)
        sem = np.maximum(target_rows["sem"].to_numpy(dtype=float), sem_floor)
        panel_sq = (pred - obs) ** 2
        panel_mse.append(float(np.mean(panel_sq)))
        sq.append(panel_sq)
        wt.append(1.0 / (sem**2))
    if weighting == "sem":
        return float(np.sqrt(np.average(np.concatenate(sq), weights=np.concatenate(wt))))
    if weighting == "panel-balanced":
        return float(np.sqrt(np.mean(panel_mse)))
    if weighting == "unweighted":
        return float(np.sqrt(np.mean(np.concatenate(sq))))
    raise ValueError(f"Unknown weighting: {weighting}")


def _trace_diagnostics(
    pooled_model: pd.DataFrame,
    target: pd.DataFrame,
    *,
    sector: str,
    sem_floor: float,
    panel_keys: tuple[tuple[str, str, str], ...],
    stimulus_window: tuple[float, float],
) -> dict[str, float]:
    return {
        "stimulus_rmse_panel_balanced": _trace_rmse(
            pooled_model,
            target,
            sector=sector,
            sem_floor=sem_floor,
            panel_keys=panel_keys,
            fit_window=stimulus_window,
            weighting="panel-balanced",
        ),
        "stimulus_rmse_unweighted": _trace_rmse(
            pooled_model,
            target,
            sector=sector,
            sem_floor=sem_floor,
            panel_keys=panel_keys,
            fit_window=stimulus_window,
            weighting="unweighted",
        ),
        "full_trace_rmse_unweighted": _trace_rmse(
            pooled_model,
            target,
            sector=sector,
            sem_floor=sem_floor,
            panel_keys=panel_keys,
            fit_window=None,
            weighting="unweighted",
        ),
        "full_trace_rmse_sem_weighted": _trace_rmse(
            pooled_model,
            target,
            sector=sector,
            sem_floor=sem_floor,
            panel_keys=panel_keys,
            fit_window=None,
            weighting="sem",
        ),
    }


def _theta_row(theta: torch.Tensor | np.ndarray) -> dict[str, float]:
    values = _theta_to_dict(theta)
    return dict(values)


def _theta_row_with_context(theta: torch.Tensor | np.ndarray, *, sector: str) -> dict[str, Any]:
    row = _theta_row(theta)
    for idx, enabled in enumerate(_receives_context(row, sector=sector)):
        row[f"receives_context_{idx}"] = bool(enabled)
    return row


def _train_sector_posterior(
    sector: str,
    *,
    prior: BoxUniform,
    base_config: dict[str, Any],
    settings: FitSettings,
    time_grid: np.ndarray,
    panel_keys: tuple[tuple[str, str, str], ...],
    fit_representation: str,
    x_obs: torch.Tensor,
    n_simulations: int,
    num_rounds: int,
    hot_start_thetas: torch.Tensor | None,
    hot_start_source_indices: list[int],
    n_jobs: int,
    seed: int,
    max_epochs: int,
    stop_after_epochs: int,
    density_estimator: str,
) -> tuple[Any, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    cell_id = FIT_SECTORS.index(sector) + 1
    inference = NPE_C(prior=prior, density_estimator=density_estimator, show_progress_bars=True)
    posterior: Any | None = None
    theta_batches: list[torch.Tensor] = []
    x_batches: list[torch.Tensor] = []
    round_batches: list[np.ndarray] = []
    source_batches: list[np.ndarray] = []

    for round_idx in range(num_rounds):
        if posterior is None:
            if hot_start_thetas is not None and len(hot_start_thetas):
                n_hot = min(len(hot_start_thetas), n_simulations)
                theta_parts = [hot_start_thetas[:n_hot]]
                if n_hot < n_simulations:
                    theta_parts.append(prior.sample((n_simulations - n_hot,)))
                theta = torch.cat(theta_parts, dim=0)
                source_ids = np.full(len(theta), -1, dtype=int)
                source_ids[:n_hot] = np.asarray(hot_start_source_indices[:n_hot], dtype=int)
            else:
                theta = prior.sample((n_simulations,))
                source_ids = np.full(len(theta), -1, dtype=int)
            proposal = None
        else:
            theta = posterior.sample(
                (n_simulations,),
                x=x_obs,
                show_progress_bars=False,
                reject_outside_prior=False,
            ).detach().cpu()
            theta = _project_theta_to_prior(theta)
            source_ids = np.full(len(theta), -1, dtype=int)
            proposal = posterior
        print(f"[sbi-npe] {sector}: round {round_idx + 1}/{num_rounds} simulating {n_simulations} draws", flush=True)
        x_rows = Parallel(n_jobs=n_jobs, verbose=10 if n_jobs != 1 else 0)(
            delayed(_simulate_theta)(
                theta[idx],
                base_config=base_config,
                sector=sector,
                cell_id=cell_id,
                seed=seed + round_idx * n_simulations + idx + 1,
                settings=settings,
                time_grid=time_grid,
                panel_keys=panel_keys,
                fit_representation=fit_representation,
            )
            for idx in range(n_simulations)
        )
        x = torch.stack(x_rows).to(dtype=torch.float32)
        valid = torch.isfinite(x).all(dim=1) & (torch.amax(torch.abs(x), dim=1) < 50.0)
        invalid_count = int((~valid).sum().item())
        if invalid_count:
            print(f"[sbi-npe] {sector}: dropping {invalid_count}/{n_simulations} invalid simulator draws", flush=True)
        theta = theta[valid]
        x = x[valid]
        source_ids = source_ids[valid.detach().cpu().numpy()]
        if len(theta) < max(4, min(12, n_simulations // 2)):
            raise RuntimeError(f"{sector}: only {len(theta)} valid simulator draws remain after filtering")
        theta_batches.append(theta)
        x_batches.append(x)
        round_batches.append(np.full(len(theta), round_idx + 1, dtype=int))
        source_batches.append(source_ids)
        estimator = inference.append_simulations(theta, x, proposal=proposal).train(
            training_batch_size=min(128, sum(len(batch) for batch in theta_batches)),
            max_num_epochs=max_epochs,
            stop_after_epochs=stop_after_epochs,
        )
        posterior = inference.build_posterior(estimator).set_default_x(x_obs)
    if posterior is None:
        raise RuntimeError(f"{sector}: no posterior was trained")
    return posterior, torch.cat(theta_batches), torch.cat(x_batches), np.concatenate(round_batches), np.concatenate(source_batches)


def _perturb_thetas(
    theta: torch.Tensor,
    *,
    n_samples: int,
    scale: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    theta_np = theta.detach().cpu().numpy().astype(float).reshape(1, -1)
    low = THETA_LOW.numpy().reshape(1, -1)
    high = THETA_HIGH.numpy().reshape(1, -1)
    proposal = theta_np + rng.normal(loc=0.0, scale=scale, size=(n_samples, theta_np.shape[1])) * (high - low)
    return _project_theta_to_prior(torch.as_tensor(proposal, dtype=torch.float32))


def _evaluate_posterior_samples(
    sector: str,
    *,
    posterior: Any,
    x_obs: torch.Tensor,
    conditioning_vectors: torch.Tensor,
    conditioning_neuron_ids: np.ndarray,
    training_thetas: torch.Tensor,
    training_candidate_indices: np.ndarray,
    base_config: dict[str, Any],
    settings: FitSettings,
    target: pd.DataFrame,
    target_example_score_vectors: torch.Tensor | None,
    target_example_neuron_ids: np.ndarray,
    score_time_grid: np.ndarray,
    target_mode: str,
    example_score: str,
    panel_keys: tuple[tuple[str, str, str], ...],
    posterior_samples: int,
    fit_window: tuple[float, float],
    score_weighting: str,
    local_search_rounds: int,
    local_search_starts: int,
    local_search_samples: int,
    local_search_scale: float,
    n_jobs: int,
    seed: int,
    sem_floor: float,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    torch.manual_seed(seed)
    if conditioning_vectors.ndim == 1:
        conditioning_vectors = conditioning_vectors.reshape(1, -1)
    if len(conditioning_neuron_ids) != len(conditioning_vectors):
        conditioning_neuron_ids = np.full(len(conditioning_vectors), -1, dtype=int)
    n_per_conditioning = max(1, int(np.ceil(max(1, posterior_samples) / len(conditioning_vectors))))
    candidate_records: list[dict[str, Any]] = []
    posterior_idx = 0
    for obs_idx, obs_vector in enumerate(conditioning_vectors):
        samples = posterior.sample(
            (n_per_conditioning,),
            x=obs_vector,
            show_progress_bars=False,
            reject_outside_prior=False,
        ).detach().cpu()
        samples = _project_theta_to_prior(samples)
        for sample in samples:
            candidate_records.append(
                {
                    "theta": sample,
                    "candidate_source": "posterior",
                    "posterior_sample_idx": posterior_idx,
                    "conditioning_example_idx": obs_idx,
                    "conditioning_neuron_idx": int(conditioning_neuron_ids[obs_idx]),
                    "training_simulation_idx": -1,
                    "local_parent_idx": -1,
                    "local_round": 0,
                }
            )
            posterior_idx += 1
    for rank, train_idx in enumerate(training_candidate_indices):
        candidate_records.append(
            {
                "theta": training_thetas[int(train_idx)].detach().cpu(),
                "candidate_source": "training",
                "posterior_sample_idx": -1,
                "conditioning_example_idx": -1,
                "conditioning_neuron_idx": -1,
                "training_simulation_idx": int(train_idx),
                "local_parent_idx": -1,
                "local_round": 0,
            }
        )
    cell_id = FIT_SECTORS.index(sector) + 1
    next_candidate_idx = 0
    theta_by_candidate_idx: dict[int, torch.Tensor] = {}
    all_results: list[tuple[dict[str, Any], pd.DataFrame]] = []

    def evaluate(record: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
        idx = int(record["candidate_idx"])
        theta = record["theta"]
        config = _config_from_theta(theta, base_config=base_config, sector=sector, cell_id=cell_id, seed=seed + 10_000 + idx)
        trace = _run_config_trace(config, settings)
        pooled = _pool_model_trace(trace)
        if target_mode == "examples":
            if target_example_score_vectors is None:
                raise ValueError("target_example_score_vectors is required when target_mode='examples'.")
            model_score_vector = _model_vector(
                pooled,
                time_grid=score_time_grid,
                panel_keys=panel_keys,
                fit_representation="trace",
            )
            distance = float(
                _score_vectors_against_examples(
                    model_score_vector,
                    target_example_score_vectors,
                    n_panels=len(panel_keys),
                    example_score=example_score,
                )[0].item()
            )
        else:
            distance = _trace_rmse(
                pooled,
                target,
                sector=sector,
                sem_floor=sem_floor,
                panel_keys=panel_keys,
                fit_window=fit_window,
                weighting=score_weighting,
            )
        row = {
            "sector": sector,
            "candidate_source": record["candidate_source"],
            "candidate_idx": idx,
            "posterior_sample_idx": int(record["posterior_sample_idx"]),
            "conditioning_example_idx": int(record["conditioning_example_idx"]),
            "conditioning_neuron_idx": int(record["conditioning_neuron_idx"]),
            "training_simulation_idx": int(record["training_simulation_idx"]),
            "local_parent_idx": int(record["local_parent_idx"]),
            "local_round": int(record["local_round"]),
            "distance": distance,
            "n_target_examples": 0 if target_example_score_vectors is None else int(len(target_example_score_vectors)),
        }
        row.update(
            _trace_diagnostics(
                pooled,
                target,
                sector=sector,
                sem_floor=sem_floor,
                panel_keys=panel_keys,
                stimulus_window=fit_window,
            )
        )
        row.update(_theta_row_with_context(theta, sector=sector))
        return row, pooled.assign(RotatedSector=sector, posterior_sample_idx=idx)

    def score_records(records: list[dict[str, Any]]) -> list[tuple[dict[str, Any], pd.DataFrame]]:
        nonlocal next_candidate_idx
        for record in records:
            record["candidate_idx"] = next_candidate_idx
            theta_by_candidate_idx[next_candidate_idx] = record["theta"].detach().cpu()
            next_candidate_idx += 1
        return Parallel(n_jobs=n_jobs, verbose=10 if n_jobs != 1 else 0)(delayed(evaluate)(record) for record in records)

    all_results.extend(score_records(candidate_records))
    rng = np.random.default_rng(seed)
    for round_idx in range(1, local_search_rounds + 1):
        rows_so_far = pd.DataFrame([row for row, _pooled in all_results]).sort_values("distance", kind="mergesort")
        parent_rows = rows_so_far.head(max(0, local_search_starts))
        if parent_rows.empty or local_search_samples <= 0:
            break
        local_records = []
        round_scale = float(local_search_scale) / np.sqrt(round_idx)
        for parent in parent_rows.itertuples(index=False):
            parent_idx = int(parent.candidate_idx)
            for theta in _perturb_thetas(
                theta_by_candidate_idx[parent_idx],
                n_samples=local_search_samples,
                scale=round_scale,
                rng=rng,
            ):
                local_records.append(
                    {
                        "theta": theta,
                        "candidate_source": "local_search",
                        "posterior_sample_idx": -1,
                        "conditioning_example_idx": -1,
                        "conditioning_neuron_idx": -1,
                        "training_simulation_idx": -1,
                        "local_parent_idx": parent_idx,
                        "local_round": round_idx,
                    }
                )
        print(
            f"[sbi-npe] {sector}: local search round {round_idx}/{local_search_rounds} "
            f"simulating {len(local_records)} perturbations",
            flush=True,
        )
        all_results.extend(score_records(local_records))

    rows = pd.DataFrame([row for row, _pooled in all_results]).sort_values("distance", kind="mergesort").reset_index(drop=True)
    best_idx = int(rows.iloc[0]["candidate_idx"])
    best_theta = theta_by_candidate_idx[best_idx]
    best_config = _config_from_theta(
        best_theta,
        base_config=base_config,
        sector=sector,
        cell_id=cell_id,
        seed=seed + 10_000 + best_idx,
    )
    best_pooled = next(pooled for row, pooled in all_results if int(row["candidate_idx"]) == best_idx)
    return rows, best_config, best_pooled


def _final_traces(best_configs: dict[str, dict[str, Any]], settings: FitSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    traces = []
    for sector, config in best_configs.items():
        trace = _run_config_trace(config, settings)
        trace["RotatedSector"] = sector
        trace["image_group"] = np.where(trace["condition"].isin(["familiar_1", "familiar_2"]), "familiar", "novel")
        traces.append(trace)
    by_image = pd.concat(traces, ignore_index=True)
    pooled = (
        by_image.groupby(["RotatedSector", "image_group", "column_key", "column_label", "response_type", "x_seconds"], as_index=False)
        .agg(y=("y", "mean"))
        .sort_values(["RotatedSector", "image_group", "column_key", "response_type", "x_seconds"])
        .reset_index(drop=True)
    )
    return by_image, pooled


def _plot_fit(
    pooled_model: pd.DataFrame,
    target: pd.DataFrame,
    best: pd.DataFrame,
    *,
    output_path: Path,
    distance_label: str,
    fit_window: tuple[float, float],
) -> None:
    columns = [(group, key) for group in ("familiar", "novel") for key in FINAL_COLUMN_KEYS]
    fig, axes = plt.subplots(
        len(FIT_SECTORS),
        len(columns),
        figsize=(1.55 * len(columns) + 2.4, 5.6),
        sharex=True,
        sharey=False,
    )
    for row_idx, sector in enumerate(FIT_SECTORS):
        sector_model = pooled_model.loc[pooled_model["RotatedSector"].eq(sector)]
        sector_target = target.loc[target["RotatedSector"].eq(sector)]
        y_values = np.concatenate(
            [
                sector_model["y"].to_numpy(dtype=float),
                sector_target["mean_response"].to_numpy(dtype=float),
            ]
        )
        lo = min(0.0, float(np.nanmin(y_values)))
        hi = max(1.0, float(np.nanmax(y_values)))
        pad = 0.12 * max(hi - lo, 1.0)
        for col_idx, (image_group, column_key) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            ax.axhline(0.0, color="0.82", linewidth=0.6)
            ax.axvspan(fit_window[0], fit_window[1], color="0.92", linewidth=0, zorder=0)
            if column_key in MODEL_TO_REAL_STAGE:
                real_stage = MODEL_TO_REAL_STAGE[column_key]
                real_rows = sector_target.loc[
                    sector_target["image_group"].eq(image_group)
                    & sector_target["stage"].eq(real_stage)
                ]
                for image_type, response_type in REAL_RESPONSE_LABEL.items():
                    real_trace = real_rows.loc[real_rows["image_type"].eq(image_type)].sort_values("time")
                    x = real_trace["time"].to_numpy(dtype=float)
                    y = real_trace["mean_response"].to_numpy(dtype=float)
                    sem = real_trace["sem"].fillna(0.0).to_numpy(dtype=float)
                    ax.plot(x, y, color=RESPONSE_COLORS[response_type], linewidth=2.0, alpha=0.35)
                    ax.fill_between(x, y - sem, y + sem, color=RESPONSE_COLORS[response_type], alpha=0.10, linewidth=0)
            model_rows = sector_model.loc[
                sector_model["image_group"].eq(image_group)
                & sector_model["column_key"].eq(column_key)
            ]
            for response_type in ("NO", "O"):
                model_trace = model_rows.loc[model_rows["response_type"].eq(response_type)].sort_values("x_seconds")
                ax.plot(
                    model_trace["x_seconds"],
                    model_trace["y"],
                    color=RESPONSE_COLORS[response_type],
                    linewidth=1.15,
                )
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                title = f"{GROUP_LABELS[image_group]}\n{COLUMN_LABELS[column_key]}" if col_idx % len(FINAL_COLUMN_KEYS) == 0 else COLUMN_LABELS[column_key]
                ax.set_title(title, fontsize=8)
            if col_idx == 0:
                row = best.loc[best["sector"].eq(sector)].iloc[0]
                ax.text(
                    -0.18,
                    0.5,
                    f"{sector}\n{distance_label} {float(row['distance']):.3f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8,
                    color=th.ROTATED_SECTOR_PALETTE[sector],
                    fontweight="bold",
                )
    handles = [
        plt.Line2D([0], [0], color="black", lw=1.5, label="NO model"),
        plt.Line2D([0], [0], color="red", lw=1.5, label="O model"),
        plt.Line2D([0], [0], color="black", lw=2.5, alpha=0.35, label="NO real target"),
        plt.Line2D([0], [0], color="red", lw=2.5, alpha=0.35, label="O real target"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4, frameon=False, fontsize=8)
    fig.suptitle("sbi NPE fit to real pooled familiar-sector stimulus traces, with ablation evaluation", y=1.0, fontsize=12)
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.05, top=0.78, wspace=0.12, hspace=0.2)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_sbi_fit(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / DEFAULT_OUTPUT_NAME).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config, metadata, settings = _load_run(run_dir)
    target = _load_targets(args.target_csv.resolve(), sectors=FIT_SECTORS)
    target_examples = None
    if args.target_mode == "examples":
        target_examples = _load_target_examples(
            args.target_example_trace_csv.resolve(),
            args.target_example_summary_csv.resolve(),
            sectors=FIT_SECTORS,
        )
    panel_keys = FIT_PANEL_KEY_SETS[args.fit_panels]
    fit_window = (float(args.fit_window_start), float(args.fit_window_end))
    prior = _theta_prior()
    all_sim_rows: list[pd.DataFrame] = []
    all_posterior_rows: list[pd.DataFrame] = []
    best_configs: dict[str, dict[str, Any]] = {}

    for sector_idx, sector in enumerate(FIT_SECTORS):
        time_grid = _target_time_grid(
            target.loc[target["RotatedSector"].eq(sector)],
            args.fit_timepoints,
            fit_window=fit_window,
        )
        target_example_fit_vectors: torch.Tensor | None = None
        target_example_score_vectors: torch.Tensor | None = None
        target_example_neuron_ids = np.asarray([], dtype=int)
        if args.target_mode == "examples":
            if target_examples is None:
                raise ValueError("target_examples were not loaded.")
            target_example_fit_vectors, target_example_neuron_ids = _target_example_vectors(
                target_examples,
                sector=sector,
                time_grid=time_grid,
                panel_keys=panel_keys,
                fit_representation=args.fit_representation,
            )
            target_example_score_vectors, score_neuron_ids = _target_example_vectors(
                target_examples,
                sector=sector,
                time_grid=time_grid,
                panel_keys=panel_keys,
                fit_representation="trace",
            )
            if not np.array_equal(target_example_neuron_ids, score_neuron_ids):
                raise ValueError(f"Example neuron id mismatch for {sector}.")
            x_obs = target_example_fit_vectors.mean(dim=0)
            conditioning_vectors = target_example_fit_vectors
            conditioning_neuron_ids = target_example_neuron_ids
        else:
            x_obs = _target_vector(
                target,
                sector=sector,
                time_grid=time_grid,
                panel_keys=panel_keys,
                fit_representation=args.fit_representation,
            )
            conditioning_vectors = x_obs.reshape(1, -1)
            conditioning_neuron_ids = np.asarray([-1], dtype=int)
        sector_seed = int(args.seed + 1009 * sector_idx)
        hot_start_thetas: torch.Tensor | None = None
        hot_start_source_indices: list[int] = []
        if args.hot_start_sector_samples:
            hot_start_thetas, hot_start_source_indices = _load_sector_hot_start_thetas(
                run_dir=run_dir,
                summary_csv=args.hot_start_summary_csv.resolve(),
                sector=sector,
                n_requested=args.num_simulations,
                seed=sector_seed,
            )
            n_hot = 0 if hot_start_thetas is None else len(hot_start_thetas)
            print(
                f"[sbi-npe] {sector}: hot-starting round 1 with {n_hot} observed "
                f"familiar-sector parameter sets",
                flush=True,
            )
        print(
            f"[sbi-npe] fitting {sector}: simulations={args.num_simulations}, "
            f"rounds={args.num_rounds}, x_dim={int(x_obs.numel())}, "
            f"posterior_samples={args.posterior_samples}, fit_panels={args.fit_panels}, "
            f"fit_window={fit_window}, fit_representation={args.fit_representation}, "
            f"score_weighting={args.score_weighting}, target_mode={args.target_mode}, "
            f"n_examples={0 if target_example_fit_vectors is None else len(target_example_fit_vectors)}",
            flush=True,
        )
        posterior, theta, x, round_ids, hot_start_source_ids = _train_sector_posterior(
            sector,
            prior=prior,
            base_config=base_config,
            settings=settings,
            time_grid=time_grid,
            panel_keys=panel_keys,
            fit_representation=args.fit_representation,
            x_obs=x_obs,
            n_simulations=args.num_simulations,
            num_rounds=args.num_rounds,
            hot_start_thetas=hot_start_thetas,
            hot_start_source_indices=hot_start_source_indices,
            n_jobs=args.n_jobs,
            seed=sector_seed,
            max_epochs=args.max_epochs,
            stop_after_epochs=args.stop_after_epochs,
            density_estimator=args.density_estimator,
        )
        sim_rows = pd.DataFrame([_theta_row_with_context(row, sector=sector) for row in theta])
        sim_rows.insert(0, "sector", sector)
        sim_rows.insert(1, "simulation_idx", np.arange(len(sim_rows)))
        sim_rows.insert(2, "round", round_ids)
        sim_rows.insert(3, "hot_start_source_neuron_idx", hot_start_source_ids)
        all_sim_rows.append(sim_rows)
        if args.target_mode == "examples":
            if target_example_fit_vectors is None:
                raise ValueError("target_example_fit_vectors are required for example-mode training distances.")
            training_distance = _score_vectors_against_examples(
                x,
                target_example_fit_vectors,
                n_panels=len(panel_keys),
                example_score=args.example_score,
            )
        else:
            training_distance = torch.sqrt(torch.mean((x - x_obs.reshape(1, -1)) ** 2, dim=1))
        n_training_candidates = min(int(args.training_candidates), len(training_distance))
        training_candidate_indices = torch.argsort(training_distance)[:n_training_candidates].detach().cpu().numpy()
        posterior_rows, best_config, _best_pooled = _evaluate_posterior_samples(
            sector,
            posterior=posterior,
            x_obs=x_obs,
            conditioning_vectors=conditioning_vectors,
            conditioning_neuron_ids=conditioning_neuron_ids,
            training_thetas=theta,
            training_candidate_indices=training_candidate_indices,
            base_config=base_config,
            settings=settings,
            target=target,
            target_example_score_vectors=target_example_score_vectors,
            target_example_neuron_ids=target_example_neuron_ids,
            score_time_grid=time_grid,
            target_mode=args.target_mode,
            example_score=args.example_score,
            panel_keys=panel_keys,
            posterior_samples=args.posterior_samples,
            fit_window=fit_window,
            score_weighting=args.score_weighting,
            local_search_rounds=args.local_search_rounds,
            local_search_starts=args.local_search_starts,
            local_search_samples=args.local_search_samples,
            local_search_scale=args.local_search_scale,
            n_jobs=args.n_jobs,
            seed=sector_seed + 50_000,
            sem_floor=args.sem_floor,
        )
        all_posterior_rows.append(posterior_rows)
        best_configs[sector] = best_config
        print(
            f"[sbi-npe] {sector} best candidate stimulus RMSE="
            f"{float(posterior_rows.iloc[0]['distance']):.4f} "
            f"({posterior_rows.iloc[0]['candidate_source']})",
            flush=True,
        )

    simulations = pd.concat(all_sim_rows, ignore_index=True)
    posterior_samples = pd.concat(all_posterior_rows, ignore_index=True)
    best = (
        posterior_samples.sort_values(["sector", "distance"], kind="mergesort")
        .groupby("sector", as_index=False)
        .head(1)
        .sort_values("sector")
        .reset_index(drop=True)
    )
    simulations.to_csv(output_dir / "npe_training_thetas.csv", index=False)
    posterior_samples.to_csv(output_dir / "posterior_sample_scores.csv", index=False)
    best.to_csv(output_dir / "best_fit_summary.csv", index=False)
    target.to_csv(output_dir / "real_trace_targets_used.csv", index=False)
    if target_examples is not None:
        target_examples.to_csv(output_dir / "real_trace_target_examples_used.csv", index=False)
    with (output_dir / "best_configs.json").open("w") as handle:
        json.dump(best_configs, handle, indent=2, default=repr)

    final_by_image, final_pooled = _final_traces(best_configs, settings)
    final_by_image.to_csv(output_dir / "fitted_sector_traces_by_image_with_ablations.csv", index=False)
    final_pooled.to_csv(output_dir / "fitted_sector_traces_pooled_with_ablations.csv", index=False)
    for fmt in ("png", "svg"):
        distance_label = "example RMSE" if args.target_mode == "examples" else "stim RMSE"
        _plot_fit(
            final_pooled,
            target,
            best,
            output_path=output_dir / f"fitted_vs_real_traces_with_ablations.{fmt}",
            distance_label=distance_label,
            fit_window=fit_window,
        )

    metadata_out = {
        "method": "sbi NPE_C posterior fit, posterior/training/local-search candidates reranked by stimulus-window trace RMSE",
        "sbi_version": __import__("sbi").__version__,
        "run_dir": str(run_dir),
        "target_csv": str(args.target_csv.resolve()),
        "sectors": list(FIT_SECTORS),
        "force_feedback_sectors": list(FORCE_FEEDBACK_SECTORS),
        "free_apical_gain_strength_sectors": list(FREE_APICAL_GAIN_STRENGTH_SECTORS),
        "fit_panels": args.fit_panels,
        "fit_panel_keys": [list(item) for item in panel_keys],
        "fit_window_seconds": list(fit_window),
        "fit_representation": args.fit_representation,
        "score_weighting": args.score_weighting,
        "target_mode": args.target_mode,
        "example_score": args.example_score,
        "target_example_trace_csv": str(args.target_example_trace_csv.resolve()) if args.target_mode == "examples" else None,
        "target_example_summary_csv": str(args.target_example_summary_csv.resolve()) if args.target_mode == "examples" else None,
        "hot_start_sector_samples": args.hot_start_sector_samples,
        "hot_start_summary_csv": str(args.hot_start_summary_csv.resolve()) if args.hot_start_sector_samples else None,
        "theta_specs": [{"name": name, "low": lo, "high": hi} for name, lo, hi in THETA_SPECS],
        "num_simulations": args.num_simulations,
        "num_rounds": args.num_rounds,
        "posterior_samples": args.posterior_samples,
        "training_candidates": args.training_candidates,
        "local_search_rounds": args.local_search_rounds,
        "local_search_starts": args.local_search_starts,
        "local_search_samples": args.local_search_samples,
        "local_search_scale": args.local_search_scale,
        "fit_timepoints": args.fit_timepoints,
        "density_estimator": args.density_estimator,
        "max_epochs": args.max_epochs,
        "stop_after_epochs": args.stop_after_epochs,
        "sem_floor": args.sem_floor,
        "fit_settings": settings.__dict__,
        "source_metadata": metadata,
    }
    (output_dir / "fit_metadata.json").write_text(json.dumps(metadata_out, indent=2, default=repr))
    print(f"[sbi-npe] wrote outputs to {output_dir}")
    print(
        best[
            [
                "sector",
                "distance",
                "candidate_source",
                "candidate_idx",
                "posterior_sample_idx",
                "conditioning_neuron_idx",
                "training_simulation_idx",
                "local_parent_idx",
                "local_round",
                "n_target_examples",
                "full_trace_rmse_sem_weighted",
            ]
        ].to_string(index=False)
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET_CSV)
    parser.add_argument("--target-mode", type=str, default="mean", choices=["mean", "examples"])
    parser.add_argument("--target-example-trace-csv", type=Path, default=DEFAULT_TARGET_EXAMPLE_TRACE_CSV)
    parser.add_argument("--target-example-summary-csv", type=Path, default=DEFAULT_TARGET_EXAMPLE_SUMMARY_CSV)
    parser.add_argument("--example-score", type=str, default="mean", choices=["mean", "median", "min"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-simulations", type=int, default=64)
    parser.add_argument("--num-rounds", type=int, default=1)
    parser.add_argument("--posterior-samples", type=int, default=32)
    parser.add_argument("--training-candidates", type=int, default=8)
    parser.add_argument("--fit-panels", type=str, default="all", choices=sorted(FIT_PANEL_KEY_SETS))
    parser.add_argument("--hot-start-sector-samples", action="store_true")
    parser.add_argument("--hot-start-summary-csv", type=Path, default=DEFAULT_HOT_START_SUMMARY_CSV)
    parser.add_argument("--fit-timepoints", type=int, default=0, help="0 uses all real target timepoints.")
    parser.add_argument("--fit-window-start", type=float, default=DEFAULT_FIT_WINDOW[0])
    parser.add_argument("--fit-window-end", type=float, default=DEFAULT_FIT_WINDOW[1])
    parser.add_argument("--fit-representation", type=str, default="trace", choices=["trace", "summary"])
    parser.add_argument("--score-weighting", type=str, default="panel-balanced", choices=["panel-balanced", "unweighted", "sem"])
    parser.add_argument("--local-search-rounds", type=int, default=2)
    parser.add_argument("--local-search-starts", type=int, default=6)
    parser.add_argument("--local-search-samples", type=int, default=8)
    parser.add_argument("--local-search-scale", type=float, default=0.08)
    parser.add_argument("--density-estimator", type=str, default="maf", choices=["maf", "nsf", "mdn", "made"])
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--stop-after-epochs", type=int, default=25)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--sem-floor", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=26082026)
    return parser


def main() -> None:
    run_sbi_fit(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
