from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from context_contrasting.minimal.config import broad, minimal_configs
from context_contrasting.minimal.experiment import design_experimental_phase
from context_contrasting.minimal.minimal import CCNeuron
from context_contrasting.minimal.transition_types import (
    scalar_state_profile_from_summary,
    transition_points_from_summary,
)

PARAMETER_ORDER = [
    "w_ff_0",
    "w_ff_1",
    "w_fb_0",
    "w_fb_1",
    "w_lat_0",
    "w_lat_1",
    "w_pv_lat_0",
    "w_pv_lat_1",
    "W_pv_00",
    "W_pv_01",
    "W_pv_10",
    "W_pv_11",
]

PARAMETER_PLOT_GROUPS = [
    ("w_ff_0", "w_ff_1", "FF initial weights"),
    ("w_fb_0", "w_fb_1", "FB initial weights"),
    ("w_lat_0", "w_lat_1", "LAT initial weights"),
    ("w_pv_lat_0", "w_pv_lat_1", "PV->LAT initial weights"),
    ("W_pv_00", "W_pv_01", "PV cell 0 input weights"),
    ("W_pv_10", "W_pv_11", "PV cell 1 input weights"),
]

SUMMARY_ORDER = [
    "full_familiar_naive",
    "occlusion_familiar_naive",
    "full_familiar_expert",
    "occlusion_familiar_expert",
    "full_novel_naive",
    "occlusion_novel_naive",
    "full_novel_expert",
    "occlusion_novel_expert",
]

TRANSITION_LABEL_ORDER = [
    "FF -> FF",
    "FF -> FB",
    "FF -> unresponsive",
    "FB -> FF",
    "FB -> FB",
    "FB -> unresponsive",
    "unresponsive -> FF",
    "unresponsive -> FB",
    "unresponsive -> unresponsive",
]

TRANSITION_COLORS = {
    "FF -> FF": "#1f77b4",
    "FF -> FB": "#d62728",
    "FF -> unresponsive": "#ff7f0e",
    "FB -> FF": "#9467bd",
    "FB -> FB": "#8c564b",
    "FB -> unresponsive": "#e377c2",
    "unresponsive -> FF": "#17becf",
    "unresponsive -> FB": "#2ca02c",
    "unresponsive -> unresponsive": "#7f7f7f",
}

CONTEXT_MODES = [
    (True, True),
    (True, False),
    (False, True),
    (False, False),
]

CONTEXT_MARKERS = {
    (True, True): "o",
    (True, False): "^",
    (False, True): "s",
    (False, False): "*",
}

CONTEXT_LABELS = {
    (True, True): "ctx=(T,T)",
    (True, False): "ctx=(T,F)",
    (False, True): "ctx=(F,T)",
    (False, False): "ctx=(F,F)",
}


@dataclass(frozen=True)
class ExplorationSettings:
    n_steps_per_phase: int = 100
    n_trials: int = 10
    tail_window: int = 25
    intertrial_sigma: float = 0.05
    weight_floor: float = 1e-4
    weight_ceiling: float = 2.0
    grid_levels: int = 2
    sobol_samples: int = 4096
    random_seed: int = 7
    activity_threshold: float = 0.025
    n_jobs: int = -1
    max_grid_points: int = 50000


def context_mode_key(mode: tuple[bool, bool]) -> str:
    return f"{int(mode[0])}{int(mode[1])}"


def _base_config(receives_context: tuple[bool, bool]) -> dict[str, Any]:
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


def params_from_config(config: dict[str, Any]) -> dict[str, float]:
    return {
        "w_ff_0": float(config["w_ff_init"]["mu"][0]),
        "w_ff_1": float(config["w_ff_init"]["mu"][1]),
        "w_fb_0": float(config["w_fb_init"]["mu"][0]),
        "w_fb_1": float(config["w_fb_init"]["mu"][1]),
        "w_lat_0": float(config["w_lat_init"]["mu"][0]),
        "w_lat_1": float(config["w_lat_init"]["mu"][1]),
        "w_pv_lat_0": float(config["w_pv_lat_init"]["mu"][0]),
        "w_pv_lat_1": float(config["w_pv_lat_init"]["mu"][1]),
        "W_pv_00": float(config["W_pv_init"]["mu"][0][0]),
        "W_pv_01": float(config["W_pv_init"]["mu"][0][1]),
        "W_pv_10": float(config["W_pv_init"]["mu"][1][0]),
        "W_pv_11": float(config["W_pv_init"]["mu"][1][1]),
    }


def config_from_params(
    params: dict[str, float],
    *,
    receives_context: tuple[bool, bool],
    seed: int,
) -> dict[str, Any]:
    config = _base_config(receives_context)
    config["seed"] = seed
    config["w_ff_init"] = {"mu": [params["w_ff_0"], params["w_ff_1"]], "sigma": 0.0}
    config["w_fb_init"] = {"mu": [params["w_fb_0"], params["w_fb_1"]], "sigma": 0.0}
    config["w_lat_init"] = {"mu": [params["w_lat_0"], params["w_lat_1"]], "sigma": 0.0}
    config["w_pv_lat_init"] = {"mu": [params["w_pv_lat_0"], params["w_pv_lat_1"]], "sigma": 0.0}
    config["W_pv_init"] = {
        "mu": (
            [params["W_pv_00"], params["W_pv_01"]],
            [params["W_pv_10"], params["W_pv_11"]],
        ),
        "sigma": [0.0, 0.0],
    }
    return config


def _run_phase_mean_response(
    model: CCNeuron,
    X: torch.Tensor,
    C: torch.Tensor,
    *,
    update: bool,
    reset_rates: bool,
    tail_window: int,
    n_steps_per_phase: int,
    n_trials: int,
) -> float:
    if reset_rates:
        model._reset_state()

    responses: list[float] = []
    for step in range(X.shape[0]):
        x, y_t, y_next, p, c = model(X[step], C[step])
        if update:
            model.update(x, y_t, y_next, p, c)
        responses.append(float(y_next))

    active_steps = n_steps_per_phase - (3 * n_steps_per_phase // 4)
    window = min(tail_window, active_steps)
    if window <= 0:
        raise ValueError("Active response window must be positive.")

    values: list[float] = []
    for trial_idx in range(n_trials):
        trial_end = (trial_idx + 1) * n_steps_per_phase
        trial_start = trial_end - window
        values.extend(responses[trial_start:trial_end])
    return float(sum(values) / len(values))


def simulate_summary(
    params: dict[str, float],
    *,
    receives_context: tuple[bool, bool],
    settings: ExplorationSettings,
    seed: int,
) -> dict[str, float]:
    config = config_from_params(params, receives_context=receives_context, seed=seed)
    model = CCNeuron(**{key: value for key, value in config.items() if not key.startswith("_")})

    X1, C1 = design_experimental_phase(
        input_mean=[1, 0],
        input_var=0.05,
        context_mean=[1, 0],
        context_var=0.05,
        n_steps=settings.n_steps_per_phase,
        n_trials=settings.n_trials,
        intertrial_sigma=settings.intertrial_sigma,
    )
    X2, C2 = design_experimental_phase(
        input_mean=[0, 1],
        input_var=0.05,
        context_mean=[0, 1],
        context_var=0.05,
        n_steps=settings.n_steps_per_phase,
        n_trials=settings.n_trials,
        intertrial_sigma=settings.intertrial_sigma,
    )
    O = torch.zeros_like(X1)

    summary = {
        "full_familiar_naive": _run_phase_mean_response(
            model,
            X1,
            C1,
            update=False,
            reset_rates=True,
            tail_window=settings.tail_window,
            n_steps_per_phase=settings.n_steps_per_phase,
            n_trials=settings.n_trials,
        ),
        "occlusion_familiar_naive": _run_phase_mean_response(
            model,
            O,
            C1,
            update=False,
            reset_rates=True,
            tail_window=settings.tail_window,
            n_steps_per_phase=settings.n_steps_per_phase,
            n_trials=settings.n_trials,
        ),
        "full_novel_naive": _run_phase_mean_response(
            model,
            X2,
            C2,
            update=False,
            reset_rates=True,
            tail_window=settings.tail_window,
            n_steps_per_phase=settings.n_steps_per_phase,
            n_trials=settings.n_trials,
        ),
        "occlusion_novel_naive": _run_phase_mean_response(
            model,
            O,
            C2,
            update=False,
            reset_rates=True,
            tail_window=settings.tail_window,
            n_steps_per_phase=settings.n_steps_per_phase,
            n_trials=settings.n_trials,
        ),
    }

    _run_phase_mean_response(
        model,
        X1,
        C1,
        update=True,
        reset_rates=True,
        tail_window=settings.tail_window,
        n_steps_per_phase=settings.n_steps_per_phase,
        n_trials=settings.n_trials,
    )

    summary.update(
        {
            "full_familiar_expert": _run_phase_mean_response(
                model,
                X1,
                C1,
                update=False,
                reset_rates=True,
                tail_window=settings.tail_window,
                n_steps_per_phase=settings.n_steps_per_phase,
                n_trials=settings.n_trials,
            ),
            "occlusion_familiar_expert": _run_phase_mean_response(
                model,
                O,
                C1,
                update=False,
                reset_rates=True,
                tail_window=settings.tail_window,
                n_steps_per_phase=settings.n_steps_per_phase,
                n_trials=settings.n_trials,
            ),
            "full_novel_expert": _run_phase_mean_response(
                model,
                X2,
                C2,
                update=False,
                reset_rates=True,
                tail_window=settings.tail_window,
                n_steps_per_phase=settings.n_steps_per_phase,
                n_trials=settings.n_trials,
            ),
            "occlusion_novel_expert": _run_phase_mean_response(
                model,
                O,
                C2,
                update=False,
                reset_rates=True,
                tail_window=settings.tail_window,
                n_steps_per_phase=settings.n_steps_per_phase,
                n_trials=settings.n_trials,
            ),
        }
    )
    return summary


def summarize_candidate(
    params: dict[str, float],
    *,
    receives_context: tuple[bool, bool],
    settings: ExplorationSettings,
    seed: int,
) -> dict[str, Any]:
    summary = simulate_summary(
        params,
        receives_context=receives_context,
        settings=settings,
        seed=seed,
    )
    state_profile = scalar_state_profile_from_summary(
        summary,
        activity_threshold=settings.activity_threshold,
    )
    transition_points = transition_points_from_summary(
        summary,
        activity_threshold=settings.activity_threshold,
    )

    row: dict[str, Any] = {
        **params,
        **summary,
        **state_profile,
        "familiar_transition_label": state_profile["familiar_transition"],
        "novel_transition_label": state_profile["novel_transition"],
        "familiar_transition_point_x": float(transition_points["familiar"][0]),
        "familiar_transition_point_y": float(transition_points["familiar"][1]),
        "novel_transition_point_x": float(transition_points["novel"][0]),
        "novel_transition_point_y": float(transition_points["novel"][1]),
        "receives_context_familiar": bool(receives_context[0]),
        "receives_context_novel": bool(receives_context[1]),
        "context_mode": context_mode_key(receives_context),
        "context_marker": CONTEXT_MARKERS[receives_context],
        "context_label": CONTEXT_LABELS[receives_context],
    }
    for name in PARAMETER_ORDER:
        row[f"log10_{name}"] = math.log10(float(params[name]))
    return row


def evaluate_parameter_sets(
    parameter_sets: list[dict[str, float]],
    *,
    settings: ExplorationSettings,
    method: str,
) -> pd.DataFrame:
    tasks = [
        (idx, params, receives_context, settings.random_seed + idx)
        for receives_context in CONTEXT_MODES
        for idx, params in enumerate(parameter_sets)
    ]
    def _run_task(weight_point_idx: int, params: dict[str, float], receives_context: tuple[bool, bool], seed: int) -> dict[str, Any]:
        row = summarize_candidate(
            params,
            receives_context=receives_context,
            settings=settings,
            seed=seed,
        )
        row["weight_point_idx"] = int(weight_point_idx)
        return row

    rows = Parallel(n_jobs=settings.n_jobs, prefer="processes")(
        delayed(_run_task)(weight_point_idx, params, receives_context, seed)
        for weight_point_idx, params, receives_context, seed in tqdm(tasks, desc=f"Simulating {method} exploration")
    )
    frame = pd.DataFrame(rows)
    frame.insert(0, "method", method)
    frame.insert(1, "sample_idx", range(len(frame)))
    return frame


def generate_grid_parameter_sets(settings: ExplorationSettings) -> list[dict[str, float]]:
    n_points = settings.grid_levels ** len(PARAMETER_ORDER)
    if n_points > settings.max_grid_points:
        raise ValueError(
            f"Grid would contain {n_points} points, above max_grid_points={settings.max_grid_points}. "
            "Use fewer grid levels or switch to Sobol sampling."
        )
    values = torch.logspace(
        math.log10(settings.weight_floor),
        math.log10(settings.weight_ceiling),
        settings.grid_levels,
    ).tolist()
    return [
        {
            name: float(value)
            for name, value in zip(PARAMETER_ORDER, combo, strict=True)
        }
        for combo in product(values, repeat=len(PARAMETER_ORDER))
    ]


def generate_sobol_parameter_sets(
    n_samples: int,
    *,
    settings: ExplorationSettings,
) -> list[dict[str, float]]:
    engine = torch.quasirandom.SobolEngine(len(PARAMETER_ORDER), scramble=True, seed=settings.random_seed)
    unit = engine.draw(n_samples)
    log_low = math.log10(settings.weight_floor)
    log_high = math.log10(settings.weight_ceiling)
    log_values = log_low + unit * (log_high - log_low)
    weights = 10.0 ** log_values
    return [
        {
            name: float(weights[row_idx, col_idx])
            for col_idx, name in enumerate(PARAMETER_ORDER)
        }
        for row_idx in range(n_samples)
    ]


def reference_transition_table(settings: ExplorationSettings) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, config in minimal_configs.items():
        params = params_from_config(config)
        summary = simulate_summary(
            params,
            receives_context=tuple(config["receives_context"]),
            settings=settings,
            seed=int(config.get("seed", settings.random_seed)),
        )
        profile = scalar_state_profile_from_summary(
            summary,
            activity_threshold=settings.activity_threshold,
        )
        points = transition_points_from_summary(
            summary,
            activity_threshold=settings.activity_threshold,
        )
        rows.append(
            {
                "reference_name": name,
                "receives_context_familiar": bool(config["receives_context"][0]),
                "receives_context_novel": bool(config["receives_context"][1]),
                **params,
                **summary,
                **profile,
                "familiar_transition_label": profile["familiar_transition"],
                "novel_transition_label": profile["novel_transition"],
                "familiar_transition_point_x": float(points["familiar"][0]),
                "familiar_transition_point_y": float(points["familiar"][1]),
                "novel_transition_point_x": float(points["novel"][0]),
                "novel_transition_point_y": float(points["novel"][1]),
            }
        )
    return pd.DataFrame(rows)


def observation_tensor(
    row: dict[str, Any] | pd.Series,
    *,
    target_mode: str,
) -> torch.Tensor:
    if target_mode == "familiar":
        values = [row["familiar_transition_point_x"], row["familiar_transition_point_y"]]
    elif target_mode == "novel":
        values = [row["novel_transition_point_x"], row["novel_transition_point_y"]]
    elif target_mode == "joint":
        values = [
            row["familiar_transition_point_x"],
            row["familiar_transition_point_y"],
            row["novel_transition_point_x"],
            row["novel_transition_point_y"],
        ]
    else:
        raise ValueError(f"Unknown target_mode {target_mode!r}.")
    return torch.tensor(values, dtype=torch.float32)


def parameter_tensor(parameter_sets: Iterable[dict[str, float]]) -> torch.Tensor:
    rows = [
        [math.log10(float(params[name])) for name in PARAMETER_ORDER]
        for params in parameter_sets
    ]
    return torch.tensor(rows, dtype=torch.float32)
