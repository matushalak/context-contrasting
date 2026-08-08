from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from context_contrasting.paper import model_scatter as paper_scatter
from context_contrasting.paper import transition_templates as paper_templates


N_FEATURES = 3
BROAD_TUNING_WIDTH = paper_templates.BROAD_TUNING_WIDTH
NARROW_TUNING_WIDTH = 1
PV_TUNING_WIDTH = paper_templates.PV_TUNING_WIDTH
PC_LEARNING_RATE_SCALE = {"PPE": 3.0, "NPE": 12.0}


@dataclass(frozen=True)
class PCWeightTemplate:
    name: str
    weight: float
    mismatch: str
    pyc_widths: tuple[str, ...]
    pyc_ffs: tuple[str, ...]
    pyc_fbs: tuple[str, ...]
    pv_ffs: tuple[str, ...]
    pv_fbs: tuple[str, ...]
    w_lats: tuple[float, ...]
    baseline_drive_sigmas: tuple[float, ...]


# PC templates are balance-mismatch templates, not response templates.
# PPE compares PyC FF drive against context-driven PV suppression:
#   under_inhibited -> low lateral inhibition; learning strengthens w_LAT.
#   over_inhibited  -> high lateral inhibition; signed voltage weakens w_LAT.
PPE_TEMPLATES: tuple[PCWeightTemplate, ...] = (
    PCWeightTemplate(
        name="ppe_under_inhibited",
        weight=0.52,
        mismatch="under_inhibited",
        pyc_widths=("narrow", "broad", "broad"),
        pyc_ffs=("mid", "strong", "strong"),
        pyc_fbs=("none",),
        pv_ffs=("weak", "mid", "mid"),
        pv_fbs=("strong", "strong", "very_strong"),
        w_lats=(0.01, 0.02, 0.04),
        baseline_drive_sigmas=(0.10, 0.14, 0.16),
    ),
    PCWeightTemplate(
        name="ppe_over_inhibited",
        weight=0.48,
        mismatch="over_inhibited",
        pyc_widths=("narrow", "broad", "broad"),
        pyc_ffs=("very_weak", "weak", "mid"),
        pyc_fbs=("none",),
        pv_ffs=("weak", "mid", "mid"),
        pv_fbs=("very_strong",),
        w_lats=(0.95,),
        baseline_drive_sigmas=(0.10, 0.14, 0.16),
    ),
)

# NPE compares PyC FB drive against feedforward PV suppression:
#   over_predicted  -> FB starts too strong; learning weakens w_FB toward PV match.
#   under_predicted -> FB starts too weak; learning strengthens w_FB toward PV match.
NPE_TEMPLATES: tuple[PCWeightTemplate, ...] = (
    PCWeightTemplate(
        name="npe_over_predicted",
        weight=0.50,
        mismatch="over_predicted",
        pyc_widths=("narrow", "broad", "broad"),
        pyc_ffs=("silent",),
        pyc_fbs=("very_strong",),
        pv_ffs=("strong",),
        pv_fbs=("none",),
        w_lats=(0.25, 0.45),
        baseline_drive_sigmas=(0.10, 0.12, 0.14),
    ),
    PCWeightTemplate(
        name="npe_under_predicted",
        weight=0.50,
        mismatch="under_predicted",
        pyc_widths=("narrow", "broad", "broad"),
        pyc_ffs=("silent",),
        pyc_fbs=("very_weak", "weak"),
        pv_ffs=("mid", "strong", "very_strong"),
        pv_fbs=("none",),
        w_lats=(0.12, 0.25, 0.45, 0.70),
        baseline_drive_sigmas=(0.10, 0.12, 0.14, 0.16),
    ),
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


def _draw_option(options: tuple[Any, ...], rng: np.random.Generator) -> Any:
    return options[int(rng.integers(0, len(options)))]


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
        pyc_width = str(_draw_option(template.pyc_widths, rng))
        pyc_ff = str(_draw_option(template.pyc_ffs, rng))
        pyc_fb = str(_draw_option(template.pyc_fbs, rng))
        pv_ff = str(_draw_option(template.pv_ffs, rng))
        pv_fb = str(_draw_option(template.pv_fbs, rng))
        w_lat = float(_draw_option(template.w_lats, rng))
        baseline_drive_sigma = float(_draw_option(template.baseline_drive_sigmas, rng))
        pyc_indices = _draw_indices(pyc_width, rng)
        pv_indices = _draw_indices("broad", rng)

        w_ff = _init_from_ff_level(pyc_ff, pyc_indices, rng)
        w_fb = _init_from_fb_level(pyc_fb, pyc_indices, rng)
        w_pv_ff = _init_from_pv_level(pv_ff, pv_indices)
        w_pv_fb = _init_from_fb_level(pv_fb, pv_indices, rng)
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
            "baseline_drive_sigma": baseline_drive_sigma,
            "pv_noise_sigma": paper_templates.PV_NOISE_SIGMA,
            "pc_learning_rate_scale": PC_LEARNING_RATE_SCALE[circuit],
            "circuit": circuit,
            "pc_mismatch": template.mismatch,
            "pc_pyc_width": pyc_width,
            "pc_pv_width": "broad",
            "ff_strength": pyc_ff,
            "fb_level": pyc_fb,
            "pv_ff_strength": pv_ff,
            "pv_fb_level": pv_fb,
            "ff_tuning_width": len(pyc_indices),
            "pv_tuning_width": len(pv_indices),
            "receives_context_0": True,
            "receives_context_1": True,
            "receives_context_2": True,
        }
        _write_init_columns(row, "w_ff_init", w_ff)
        _write_init_columns(row, "w_fb_init", w_fb)
        _write_init_columns(row, "w_lat_init", np.asarray([w_lat], dtype=float))
        _write_init_columns(row, "W_pv_init", w_pv)
        _write_init_columns(row, "w_pv_context_init", w_pv_fb)
        _write_init_columns(row, "w_pv_ff_init", w_pv_ff)
        for idx in range(N_FEATURES):
            row[f"tuned_index_{idx}"] = int(idx in pyc_indices)
            row[f"pv_tuned_index_{idx}"] = int(idx in pv_indices)
        rows.append(row)

    return pd.DataFrame(rows)
