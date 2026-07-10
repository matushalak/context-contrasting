from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from context_contrasting.paper import model_scatter as paper_scatter
from context_contrasting.paper import transition_templates as paper_templates


N_FEATURES = 3
STIMULUS_ORDER = ("familiar_1", "familiar_2", "novel")
IMAGE_INFO = {
    "familiar_1": ("familiar", 1, 1),
    "familiar_2": ("familiar", 2, 2),
    "novel": ("novel", 3, 1),
}
BROAD_TUNING_WIDTH = paper_templates.BROAD_TUNING_WIDTH
NARROW_TUNING_WIDTH = 1
PV_TUNING_WIDTH = paper_templates.PV_TUNING_WIDTH
PC_LEARNING_RATE_SCALE = {"PPE": 3.0, "NPE": 4.0}


@dataclass(frozen=True)
class PCWeightTemplate:
    name: str
    weight: float
    pyc_width: str
    pyc_ff: str
    pyc_fb: str
    pv_width: str
    pv_ff: str
    pv_fb: str
    w_lat: float
    baseline_drive_sigma: float


PPE_TEMPLATES: tuple[PCWeightTemplate, ...] = (
    PCWeightTemplate("ppe_strengthen_narrow_mid_ff_strong_pvfb", 0.14, "narrow", "mid", "none", "broad", "weak", "strong", 0.01, 0.10),
    PCWeightTemplate("ppe_strengthen_broad_strong_ff_strong_pvfb", 0.24, "broad", "strong", "none", "broad", "mid", "strong", 0.02, 0.14),
    PCWeightTemplate("ppe_strengthen_broad_strong_ff_verystrong_pvfb", 0.14, "broad", "strong", "none", "broad", "mid", "very_strong", 0.04, 0.16),
    PCWeightTemplate("ppe_release_narrow_veryweak_ff_verystrong_pvfb", 0.20, "narrow", "very_weak", "none", "broad", "weak", "very_strong", 0.95, 0.10),
    PCWeightTemplate("ppe_release_broad_weak_ff_verystrong_pvfb", 0.20, "broad", "weak", "none", "broad", "mid", "very_strong", 0.95, 0.14),
    PCWeightTemplate("ppe_release_broad_mid_ff_verystrong_pvfb", 0.08, "broad", "mid", "none", "broad", "mid", "very_strong", 0.95, 0.16),
)

NPE_TEMPLATES: tuple[PCWeightTemplate, ...] = (
    PCWeightTemplate("npe_over_fb_narrow_mid_low_pvff", 0.14, "narrow", "silent", "mid", "broad", "weak", "none", 0.02, 0.10),
    PCWeightTemplate("npe_over_fb_broad_mid_low_pvff", 0.18, "broad", "silent", "mid", "broad", "weak", "none", 0.03, 0.14),
    PCWeightTemplate("npe_over_fb_broad_strong_mid_pvff", 0.18, "broad", "silent", "strong", "broad", "mid", "none", 0.06, 0.16),
    PCWeightTemplate("npe_under_fb_narrow_weak_strong_pvff", 0.16, "narrow", "silent", "weak", "broad", "strong", "none", 0.95, 0.10),
    PCWeightTemplate("npe_under_fb_broad_mid_strong_pvff", 0.22, "broad", "silent", "mid", "broad", "strong", "none", 0.95, 0.14),
    PCWeightTemplate("npe_under_fb_broad_strong_verystrong_pvff", 0.12, "broad", "silent", "strong", "broad", "very_strong", "none", 0.95, 0.16),
)


def _templates_for(circuit: str) -> tuple[PCWeightTemplate, ...]:
    if circuit == "PPE":
        return PPE_TEMPLATES
    if circuit == "NPE":
        return NPE_TEMPLATES
    raise ValueError(f"unknown circuit: {circuit}")


def _draw_template(rng: np.random.Generator, templates: tuple[PCWeightTemplate, ...]) -> PCWeightTemplate:
    weights = np.asarray([template.weight for template in templates], dtype=float)
    return templates[int(rng.choice(len(templates), p=weights / weights.sum()))]


def _draw_indices(width: str, rng: np.random.Generator) -> tuple[int, ...]:
    if width == "narrow":
        size = NARROW_TUNING_WIDTH
    elif width == "broad":
        size = BROAD_TUNING_WIDTH
    else:
        raise ValueError(f"unknown tuning width: {width}")
    return tuple(sorted(int(idx) for idx in rng.choice(N_FEATURES, size=size, replace=False)))


def _bool_vector(indices: tuple[int, ...]) -> tuple[bool, ...]:
    index_set = set(indices)
    return tuple(idx in index_set for idx in range(N_FEATURES))


def _level_vector(indices: tuple[int, ...], *, tuned: float, silent: float) -> list[float]:
    index_set = set(indices)
    return [float(tuned if idx in index_set else silent) for idx in range(N_FEATURES)]


def _init_from_ff_level(level: str, indices: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    spec = paper_templates.FF_STRENGTHS[level]
    init = paper_scatter.weight_init(
        _level_vector(indices, tuned=spec["tuned"], silent=spec["silent"]),
        spec["rel"],
        spec["floor"],
        _level_vector(indices, tuned=spec["lo"], silent=spec["lo"]),
        _level_vector(indices, tuned=spec["hi"], silent=spec["hi"]),
    )
    return paper_scatter._draw_init(init, rng)


def _init_from_pv_level(level: str, indices: tuple[int, ...]) -> np.ndarray:
    spec = paper_templates.PV_STRENGTHS[level]
    return np.asarray(_level_vector(indices, tuned=spec["tuned"], silent=spec["silent"]), dtype=float)


def _init_from_fb_level(level: str, indices: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    spec = paper_templates.FB_LEVELS[level]
    none = paper_templates.FB_LEVELS["none"]
    tuned = float(spec["center"])
    silent = float(none["center"])
    init = paper_scatter.weight_init(
        _level_vector(indices, tuned=tuned, silent=silent),
        spec["rel"],
        spec["floor"],
        _level_vector(indices, tuned=spec["lo"], silent=none["lo"]),
        _level_vector(indices, tuned=spec["hi"], silent=none["hi"]),
    )
    return paper_scatter._draw_init(init, rng)


def _write_init_columns(row: dict[str, Any], key: str, values: np.ndarray) -> None:
    for idx, value in enumerate(np.asarray(values, dtype=float).reshape(-1)):
        row[f"{key}.mu_{idx}"] = float(value)
    row[f"{key}.sigma"] = 0.0


def _pc_learning_rates(circuit: str, n_steps_per_phase: int) -> dict[str, float]:
    scale = PC_LEARNING_RATE_SCALE[circuit]
    rates = paper_templates._effective_learning_rates(n_steps_per_phase)
    return {name: float(value) * scale for name, value in rates.items()}


def sample_pc_template_configs(
    *,
    circuit: str,
    n_samples: int,
    seed: int,
    n_steps_per_phase: int,
) -> pd.DataFrame:
    templates = _templates_for(circuit)
    rates = _pc_learning_rates(circuit, n_steps_per_phase)
    draw_rng = np.random.default_rng(seed + (10_000 if circuit == "PPE" else 20_000))
    child_rngs = np.random.SeedSequence(seed + (10_000 if circuit == "PPE" else 20_000)).spawn(n_samples)
    seen: dict[str, int] = {template.name: 0 for template in templates}
    rows: list[dict[str, Any]] = []

    for sample_global_idx in range(1, n_samples + 1):
        template = _draw_template(draw_rng, templates)
        seen[template.name] += 1
        rng = np.random.default_rng(child_rngs[sample_global_idx - 1])
        pyc_indices = _draw_indices(template.pyc_width, rng)
        pv_indices = _draw_indices(template.pv_width, rng)

        w_ff = _init_from_ff_level(template.pyc_ff, pyc_indices, rng)
        w_fb = _init_from_fb_level(template.pyc_fb, pyc_indices, rng)
        w_pv_ff = _init_from_pv_level(template.pv_ff, pv_indices)
        w_pv_fb = _init_from_fb_level(template.pv_fb, pv_indices, rng)
        w_pv = w_pv_fb if circuit == "PPE" else w_pv_ff

        row: dict[str, Any] = {
            "transition": template.name,
            "sample_idx": seen[template.name],
            "sample_global_idx": sample_global_idx,
            "seed": seed + sample_global_idx,
            "n_features": N_FEATURES,
            "n_pv": 1,
            "n_context": N_FEATURES,
            "lr_ff": rates["lr_ff"],
            "lr_fb": rates["lr_fb"],
            "lr_lat": rates["lr_lat"],
            "lr_pv": rates["lr_pv"],
            "pyc_decay": paper_templates.BASE_CONFIG["pyc_decay"],
            "pv_decay": paper_templates.BASE_CONFIG["pv_decay"],
            "baseline_drive_sigma": template.baseline_drive_sigma,
            "pv_noise_sigma": paper_templates.PV_NOISE_SIGMA,
            "pc_learning_rate_scale": PC_LEARNING_RATE_SCALE[circuit],
            "circuit": circuit,
            "pc_pyc_width": template.pyc_width,
            "pc_pv_width": template.pv_width,
            "ff_strength": template.pyc_ff,
            "fb_level": template.pyc_fb,
            "pv_ff_strength": template.pv_ff,
            "pv_fb_level": template.pv_fb,
            "ff_tuning_width": len(pyc_indices),
            "pv_tuning_width": len(pv_indices),
            "receives_context_0": True,
            "receives_context_1": True,
            "receives_context_2": True,
        }
        _write_init_columns(row, "w_ff_init", w_ff)
        _write_init_columns(row, "w_fb_init", w_fb)
        _write_init_columns(row, "w_lat_init", np.asarray([template.w_lat], dtype=float))
        _write_init_columns(row, "W_pv_init", w_pv)
        _write_init_columns(row, "w_pv_context_init", w_pv_fb)
        _write_init_columns(row, "w_pv_ff_init", w_pv_ff)
        for idx in range(N_FEATURES):
            row[f"tuned_index_{idx}"] = int(idx in pyc_indices)
            row[f"pv_tuned_index_{idx}"] = int(idx in pv_indices)
        rows.append(row)

    return pd.DataFrame(rows)


def template_trace_series(
    *,
    response: float,
    n_steps_per_phase: int,
    n_trials: int = 1,
    post_steps: int = 0,
    seed: int = 0,
) -> np.ndarray:
    pre_steps = 3 * n_steps_per_phase // 4
    stim_steps = n_steps_per_phase - pre_steps
    total_steps = n_steps_per_phase * n_trials + post_steps
    target = np.zeros(total_steps, dtype=float)
    for trial_idx in range(n_trials):
        start = trial_idx * n_steps_per_phase + pre_steps
        target[start : start + stim_steps] = float(response)

    rng = np.random.default_rng(seed)
    trace = np.zeros(total_steps, dtype=float)
    noise_state = 0.0
    for step, target_value in enumerate(target):
        previous = trace[step - 1] if step else 0.0
        alpha = 0.055 if target_value > previous else 0.020
        noise_state = 0.94 * noise_state + rng.normal(0.0, 0.006)
        trace[step] = max(0.0, previous + alpha * (target_value - previous) + noise_state)
    return trace
