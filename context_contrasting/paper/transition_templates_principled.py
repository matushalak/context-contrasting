"""Principled publication template sampler.

This variant keeps the same mini-model surface as ``transition_templates.py`` but
samples cell types as a shared distribution over FF tuning width and FB access:
broad FF, narrow FF with one random preferred image, no FB, broad FB, matched FB,
or partial FB.  There are no explicitly novel-only templates; novel behavior has
to emerge because novel FF is unadapted while familiar training strengthens FB
and LAT in cells that receive at least one familiar context channel.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import model_scatter as base
from .neuron_utils import ThresholdReLU


DEFAULT_OUTPUT_DIR = base.PACKAGE_DIR / "outputs_principled"
N_FEATURES = 3
NOVEL_INDEX = 2

LEARNING_RATE_REFERENCE_STEPS = 200
SHARED_LEARNING_RATES = {"lr_ff": 0.0135, "lr_fb": 0.00220, "lr_lat": 0.0220, "lr_pv": 0.0}
SCALAR_NOISE_KEYS = ("apical_gain_strength", "apical_drive_threshold")
SOMA_ACTIVATION_THRESHOLD = 0.08
APICAL_DRIVE_SUBTRACTIVE = True
PV_NOISE_SIGMA = 0.075
DIVISIVE_GAIN = 10.0
BASELINE_STD_SCALE = 0.27
SCALAR_NOISE = {
    "apical_gain_strength": ("log", 0.20, 0.1, 50.0, 0.0),
    "apical_drive_threshold": ("add", 0.12, 0.0, 3.0, 0.05),
}
UNIFORM_FF_NOISE = dict(rel=0.40, floor=0.016, lo=0.0, hi=0.16)
UNIFORM_FB_NOISE = dict(rel=0.50, floor=0.016, lo=0.0, hi=0.45)
UNIFORM_GAIN_CLIP = (1.5, 8.0)
NARROW_GAIN_CLIP = (1.5, 6.8)
UNIFORM_DRIVE_CLIP = (0.0, 1.5)

BASE_CONFIG = {
    "n_features": 3,
    "n_pv": 1,
    "n_context": 3,
    "activation": ThresholdReLU(threshold=0.1, subtractive=False, hasMax=True, maxValue=1.0),
    "lr_ff": 0.032,
    "lr_fb": 0.0035,
    "lr_lat": 0.002,
    "lr_pv": 0.005,
    "w_ff_init": {"mu": [0.5, 0.5, 0.01], "sigma": 0},
    "w_fb_init": {"mu": [0.03, 0.03, 0.03], "sigma": 0},
    "w_lat_init": {"mu": [0.1], "sigma": 0},
    "W_pv_init": {"mu": [0.1, 0.1, 0.05], "sigma": 0},
    "pyc_decay": 0.05,
    "pv_decay": 0.5,
    "apical_drive_threshold": 0.3,
    "apical_drive_subtractive": False,
    "apical_gain_strength": 8.0,
    "apical_gain_k": 5.0,
    "apical_gain_threshold": 0.0,
    "baseline_drive_sigma": 0.08,
    "pv_noise_sigma": PV_NOISE_SIGMA,
    "alpha": 1.0,
    "weight_decay": 0.0,
    "seed": 42,
    "receives_context": (True, True, True),
    "FBrule": "dampened-anti-Hebbian",
    "w_pv_lat_init": {"mu": [0.4], "sigma": 0},
}


def _effective_learning_rates(n_steps_per_phase: int) -> dict[str, float]:
    scale = LEARNING_RATE_REFERENCE_STEPS / float(n_steps_per_phase)
    return {
        name: (float(rate) * scale if name != "lr_pv" else float(rate))
        for name, rate in SHARED_LEARNING_RATES.items()
    }


WIDTH_CLASSES: dict[str, dict[str, Any]] = {
    "broad": dict(
        ff_plasticity_scale=8.0,
        gain=4.0,
        gain_clip=UNIFORM_GAIN_CLIP,
        gain_threshold=0.0,
        drive=0.15,
        drive_clip=UNIFORM_DRIVE_CLIP,
        baseline=0.16,
    ),
    "narrow": dict(
        ff_plasticity_scale=2.0,
        gain=6.4,
        gain_clip=NARROW_GAIN_CLIP,
        gain_threshold=0.0,
        gain_k=7.0,
        drive=1.25,
        drive_clip=UNIFORM_DRIVE_CLIP,
        baseline=0.07,
    ),
}

FF_STRENGTHS: dict[str, dict[str, Any]] = {
    "silent": dict(tuned=0.010, silent=0.010, **UNIFORM_FF_NOISE),
    "diag_weak": dict(tuned=0.042, silent=0.004, **UNIFORM_FF_NOISE),
    "very_weak": dict(tuned=0.065, silent=0.005, **UNIFORM_FF_NOISE),
    "weak": dict(tuned=0.085, silent=0.006, **UNIFORM_FF_NOISE),
    "mid": dict(tuned=0.120, silent=0.006, **UNIFORM_FF_NOISE),
    "strong": dict(tuned=0.155, silent=0.008, **UNIFORM_FF_NOISE),
}

FB_LEVELS: dict[str, dict[str, Any]] = {
    "none": dict(receives=False, center=0.004, **UNIFORM_FB_NOISE),
    "weak": dict(receives=True, center=0.040, **UNIFORM_FB_NOISE),
    "mid": dict(receives=True, center=0.075, **UNIFORM_FB_NOISE),
    "strong": dict(receives=True, center=0.110, **UNIFORM_FB_NOISE),
    "very_strong": dict(receives=True, center=0.300, **UNIFORM_FB_NOISE),
}

CONTEXT_MODES = ("none", "all", "matched", "random2", "familiar")

TEMPLATES: dict[str, dict[str, Any]] = {
    # Silent / weak broad cells preserve the familiar +O and unresponsive clouds.
    "silent_broad_FFonly": dict(width="broad", ff="silent", fb="none", tuning="all", context="none", weight=0.015),
    "silent_broad_FB_weak": dict(width="broad", ff="silent", fb="weak", tuning="all", context="all", weight=0.005, drive=0.035, gain=3.0, baseline=0.18),
    "silent_broad_FB_mid": dict(width="broad", ff="silent", fb="mid", tuning="all", context="all", weight=0.005, drive=0.035, gain=3.0, baseline=0.18),
    "silent_broad_FB_partial2": dict(width="broad", ff="silent", fb="mid", tuning="all", context="random2", weight=0.025, drive=0.035, gain=3.0, baseline=0.18),
    "silent_broad_FB_strong": dict(width="broad", ff="silent", fb="very_strong", tuning="all", context="all", weight=0.040, drive=0.18, gain=2.2, baseline=0.26),

    # Broad FF-only cells adapt familiar FF and learn LAT; novel FF remains intact
    # but can be reduced by learned LAT.
    "weak_broad_FFonly": dict(width="broad", ff="weak", fb="none", tuning="all", context="none", weight=0.130, baseline=0.22),
    "mid_broad_FFonly": dict(width="broad", ff="mid", fb="none", tuning="all", context="none", weight=0.100, baseline=0.22),
    "strong_broad_FFonly": dict(width="broad", ff="strong", fb="none", tuning="all", context="none", weight=0.055, baseline=0.22),

    # FF-bearing broad FB cells are the main novel +NO / +NO+O source because
    # familiar training strengthens broad FB while novel FF is not adapted.
    "weak_broad_FB_broad": dict(width="broad", ff="weak", fb="mid", tuning="all", context="all", weight=0.180, drive=0.430, gain=8.0, baseline=0.20),
    "mid_broad_FB_broad": dict(width="broad", ff="weak", fb="mid", tuning="all", context="all", weight=0.080, drive=0.430, gain=8.0, baseline=0.20),
    "strong_broad_FB_broad": dict(width="broad", ff="weak", fb="strong", tuning="all", context="all", weight=0.040, drive=0.410, gain=7.5, baseline=0.20),
    "weak_broad_FB_partial2": dict(width="broad", ff="weak", fb="mid", tuning="all", context="random2", weight=0.100, drive=0.400, gain=8.0, baseline=0.18),
    "weak_broad_FB_familiar": dict(width="broad", ff="weak", fb="mid", tuning="all", context="familiar", weight=0.030, drive=0.430, gain=7.5, baseline=0.18),

    # Narrow classes draw preferred image at random, so fam1/fam2/novel proportions
    # are matched by construction. Most FB-receiving narrow cells receive broad FB;
    # a smaller matched class exists and only trains if tuned to a familiar image.
    "narrow_FFonly": dict(width="narrow", ff="very_weak", fb="none", tuning="permuted1", context="none", weight=0.080, gain=6.4),
    "narrow_FB_broad_very_weak": dict(width="narrow", ff="diag_weak", fb="weak", tuning="permuted1", context="all", weight=0.250, gain=6.4),
    "narrow_FB_broad_weak": dict(width="narrow", ff="very_weak", fb="weak", tuning="permuted1", context="all", weight=0.180, gain=6.4),
    "narrow_FB_matched": dict(width="narrow", ff="very_weak", fb="weak", tuning="permuted1", context="matched", weight=0.060, gain=6.4),
    "narrow_FB_partial2": dict(width="narrow", ff="very_weak", fb="weak", tuning="permuted1", context="random2", weight=0.110, gain=6.4),
}

SURROUND_SETTINGS: dict[str, dict[str, float]] = {
    "silent_broad_FFonly": dict(lat=0.04, pvlat=0.08, pv_tuned=0.18, pv_silent=0.18),
    "silent_broad_FB_weak": dict(lat=0.85, pvlat=0.05, pv_tuned=0.60, pv_silent=0.60),
    "silent_broad_FB_mid": dict(lat=0.85, pvlat=0.05, pv_tuned=0.60, pv_silent=0.60),
    "silent_broad_FB_partial2": dict(lat=0.85, pvlat=0.05, pv_tuned=0.60, pv_silent=0.60),
    "silent_broad_FB_strong": dict(lat=0.85, pvlat=0.05, pv_tuned=0.60, pv_silent=0.60),
    "weak_broad_FFonly": dict(lat=0.035, pvlat=0.10, pv_tuned=0.24, pv_silent=0.24),
    "mid_broad_FFonly": dict(lat=0.050, pvlat=0.10, pv_tuned=0.28, pv_silent=0.28),
    "strong_broad_FFonly": dict(lat=0.050, pvlat=0.10, pv_tuned=0.28, pv_silent=0.28),
    "weak_broad_FB_broad": dict(lat=0.24, pvlat=0.05, pv_tuned=0.52, pv_silent=0.52),
    "mid_broad_FB_broad": dict(lat=0.24, pvlat=0.05, pv_tuned=0.55, pv_silent=0.55),
    "strong_broad_FB_broad": dict(lat=0.24, pvlat=0.05, pv_tuned=0.55, pv_silent=0.55),
    "weak_broad_FB_partial2": dict(lat=0.24, pvlat=0.05, pv_tuned=0.52, pv_silent=0.52),
    "weak_broad_FB_familiar": dict(lat=0.24, pvlat=0.05, pv_tuned=0.52, pv_silent=0.52),
    "narrow_FFonly": dict(lat=0.03, pvlat=0.03, pv_tuned=0.045, pv_silent=0.03),
    "narrow_FB_broad_very_weak": dict(lat=0.03, pvlat=0.03, pv_tuned=0.045, pv_silent=0.03),
    "narrow_FB_broad_weak": dict(lat=0.03, pvlat=0.03, pv_tuned=0.045, pv_silent=0.03),
    "narrow_FB_matched": dict(lat=0.03, pvlat=0.03, pv_tuned=0.045, pv_silent=0.03),
    "narrow_FB_partial2": dict(lat=0.03, pvlat=0.03, pv_tuned=0.045, pv_silent=0.03),
}

_SCALAR_OVERRIDES = {
    "gain": "apical_gain_strength",
    "drive": "apical_drive_threshold",
    "ff_plasticity_scale": "ff_plasticity_scale",
    "gain_threshold": "apical_gain_threshold",
    "gain_k": "apical_gain_k",
}


def _draw_tuned_indices(tuning: str, rng: np.random.Generator) -> tuple[int, ...]:
    if tuning == "all":
        return tuple(range(N_FEATURES))
    if tuning == "permuted1":
        return (int(rng.integers(0, N_FEATURES)),)
    raise ValueError(f"unknown tuning mode: {tuning}")


def _draw_context_indices(mode: str, tuned_indices: tuple[int, ...], rng: np.random.Generator) -> tuple[int, ...]:
    if mode == "none":
        return ()
    if mode == "all":
        return tuple(range(N_FEATURES))
    if mode == "matched":
        return tuple(tuned_indices)
    if mode == "familiar":
        return (0, 1)
    if mode == "random2":
        return tuple(sorted(int(idx) for idx in rng.choice(N_FEATURES, size=2, replace=False)))
    raise ValueError(f"unknown context mode: {mode}")


def _canonical_context_indices(mode: str, tuned_indices: tuple[int, ...]) -> tuple[int, ...]:
    if mode == "none":
        return ()
    if mode == "all":
        return tuple(range(N_FEATURES))
    if mode == "matched":
        return tuple(tuned_indices)
    if mode == "familiar":
        return (0, 1)
    if mode == "random2":
        return (0, NOVEL_INDEX)
    raise ValueError(f"unknown context mode: {mode}")


def _vector(tuned_indices: tuple[int, ...], tuned: float, silent: float) -> list[float]:
    tuned_set = set(tuned_indices)
    return [float(tuned if idx in tuned_set else silent) for idx in range(N_FEATURES)]


def _general_vector(value: float) -> list[float]:
    return [float(value)] * N_FEATURES


def _bool_vector(indices: tuple[int, ...]) -> tuple[bool, ...]:
    index_set = set(indices)
    return tuple(idx in index_set for idx in range(N_FEATURES))


def _build_transition_specs() -> dict[str, dict[str, Any]]:
    return {name: base.transition(template["weight"]) for name, template in TEMPLATES.items()}


def _build_config(name: str, tuned_indices: tuple[int, ...], context_indices: tuple[int, ...]) -> dict[str, Any]:
    template = TEMPLATES[name]
    width_class = WIDTH_CLASSES[template["width"]]
    ff = FF_STRENGTHS[template["ff"]]
    fb = FB_LEVELS[template["fb"]]
    surround = SURROUND_SETTINGS[name]
    pv_vec = _vector(tuned_indices, surround["pv_tuned"], surround["pv_silent"])
    config: dict[str, Any] = {
        "w_ff_init": base.weight_init(_vector(tuned_indices, ff["tuned"], ff["silent"]), ff["rel"], ff["floor"], _general_vector(ff["lo"]), _general_vector(ff["hi"])),
        "w_fb_init": base.weight_init(_general_vector(fb["center"]), fb["rel"], fb["floor"], _general_vector(fb["lo"]), _general_vector(fb["hi"])),
        "w_lat_init": base.weight_init([surround["lat"]]),
        "w_pv_lat_init": base.weight_init([surround["pvlat"]]),
        "W_pv_init": base.weight_init(pv_vec),
    }
    fixed = {
        "receives_context": _bool_vector(context_indices) if fb["receives"] else (False,) * N_FEATURES,
        "baseline_drive_sigma": template.get("baseline", width_class["baseline"]),
        "pv_noise_sigma": PV_NOISE_SIGMA,
        "pv_plasticity": False,
        "pv_lat_plasticity": False,
    }
    for key, model_key in _SCALAR_OVERRIDES.items():
        value = template.get(key, width_class.get(key))
        if value is not None:
            fixed[model_key] = value
    clip = {
        "apical_gain_strength": template.get("gain_clip", width_class["gain_clip"]),
        "apical_drive_threshold": template.get("drive_clip", width_class["drive_clip"]),
    }
    return {"init": config, "fix": fixed, "clip": clip, "width": len(tuned_indices), "context_width": len(context_indices)}


def _perturb_config_factory():
    def _perturb_config(transition: str, base_config: dict[str, Any], *, sample_idx: int, global_idx: int, seed: int, rng: np.random.Generator, scalar_noise_multiplier: float) -> dict[str, Any]:
        template = TEMPLATES[transition]
        tuned_indices = _draw_tuned_indices(template["tuning"], rng)
        context_indices = _draw_context_indices(template.get("context", "all"), tuned_indices, rng)
        spec = _build_config(transition, tuned_indices, context_indices)
        config = copy.deepcopy(base_config)
        for key, init_spec in spec["init"].items():
            values = base._draw_init(init_spec, rng) if key in {"w_ff_init", "w_fb_init"} else base._center_init_values(init_spec)
            base._set_init(config, key, values)
        config.update(spec["fix"])
        for key, scalar_spec in base.SCALAR_NOISE.items():
            if key in config and base._is_num(config[key]):
                config[key] = base._draw_scalar(float(config[key]), scalar_spec, rng, scalar_noise_multiplier)
        for key, (lo, hi) in (base.GLOBAL_SCALAR_CLIP | spec["clip"]).items():
            if key in config and base._is_num(config[key]):
                config[key] = base._clip(float(config[key]), lo, hi)
        base._apply_shared_learning_rates(config)
        config.update(
            seed=int(seed),
            _canonical_transition=transition,
            _sample_idx=int(sample_idx),
            _sample_global_idx=int(global_idx),
            _ff_tuning_width=int(spec["width"]),
            _ff_strength=template["ff"],
            _fb_level=template["fb"],
            _fb_context_mode=template.get("context", "all"),
            _tuned_indices=list(tuned_indices),
            _context_indices=list(context_indices),
        )
        return config

    return _perturb_config


def _canonical_tuned_indices(tuning: str) -> tuple[int, ...]:
    if tuning == "all":
        return tuple(range(N_FEATURES))
    if tuning == "permuted1":
        return (0,)
    raise ValueError(f"unknown tuning mode: {tuning}")


def _center_config(name: str) -> dict[str, Any]:
    template = TEMPLATES[name]
    tuned_indices = _canonical_tuned_indices(template["tuning"])
    context_indices = _canonical_context_indices(template.get("context", "all"), tuned_indices)
    spec = _build_config(name, tuned_indices, context_indices)
    config = copy.deepcopy(base.minimal_configs3[name])
    for key, init_spec in spec["init"].items():
        base._set_init(config, key, base._center_init_values(init_spec))
    config.update(spec["fix"])
    for key, (lo, hi) in (base.GLOBAL_SCALAR_CLIP | spec["clip"]).items():
        if key in config and base._is_num(config[key]):
            config[key] = base._clip(float(config[key]), lo, hi)
    base._apply_shared_learning_rates(config)
    config.update(_canonical_transition=name, _sample_idx=0, _sample_global_idx=0, _ff_tuning_width=int(spec["width"]), _ff_strength=template["ff"], _fb_level=template["fb"], _fb_context_mode=template.get("context", "all"), _tuned_indices=list(tuned_indices), _context_indices=list(context_indices))
    return config


def _flatten_config_factory(original_flatten):
    def _flatten_config(config: dict[str, Any]) -> dict[str, Any]:
        flat = original_flatten(config)
        flat["ff_tuning_width"] = config.get("_ff_tuning_width")
        flat["ff_strength"] = config.get("_ff_strength")
        flat["fb_level"] = config.get("_fb_level")
        flat["fb_context_mode"] = config.get("_fb_context_mode")
        tuned = config.get("_tuned_indices", [])
        context = config.get("_context_indices", [])
        for idx in range(N_FEATURES):
            flat[f"tuned_index_{idx}"] = int(idx in tuned)
            flat[f"receives_context_{idx}"] = int(idx in context)
        return flat

    return _flatten_config


def configure_model_scatter(n_steps_per_phase: int = LEARNING_RATE_REFERENCE_STEPS) -> None:
    if set(SURROUND_SETTINGS) != set(TEMPLATES):
        missing = sorted(set(TEMPLATES) - set(SURROUND_SETTINGS))
        extra = sorted(set(SURROUND_SETTINGS) - set(TEMPLATES))
        raise ValueError(f"SURROUND_SETTINGS must match TEMPLATES exactly; missing={missing}, extra={extra}")
    transitions = _build_transition_specs()
    generic_config = copy.deepcopy(BASE_CONFIG)
    generic_config["activation"] = ThresholdReLU(threshold=SOMA_ACTIVATION_THRESHOLD, subtractive=False, hasMax=True, maxValue=1.0)
    generic_config["apical_drive_subtractive"] = APICAL_DRIVE_SUBTRACTIVE
    generic_config["pv_noise_sigma"] = PV_NOISE_SIGMA
    generic_config["divisive_gain"] = DIVISIVE_GAIN
    base.TRANSITIONS = transitions
    base.minimal_configs3 = {name: copy.deepcopy(generic_config) for name in transitions}
    base.SHARED_LEARNING_RATES = _effective_learning_rates(n_steps_per_phase)
    base.SCALAR_NOISE = {key: SCALAR_NOISE[key] for key in SCALAR_NOISE_KEYS}
    base.GLOBAL_SCALAR_CLIP = {}
    base.BASELINE_STD_SCALE = BASELINE_STD_SCALE
    base._perturb_config = _perturb_config_factory()
    base._flatten_config = _flatten_config_factory(base._flatten_config)
    base._center_config = _center_config


def write_metadata(args) -> None:
    path = args.output_dir / "metadata.json"
    metadata = json.loads(path.read_text()) if path.exists() else {}
    metadata["principled_template_variant"] = {
        "description": "Shared broad/narrow FF x none/broad/matched/partial FB population; no explicit novel-only templates.",
        "disabled_plasticity": ["W_pv", "w_pv_lat"],
        "enabled_plasticity": ["w_ff", "w_fb", "w_lat"],
        "shared_learning_rates": _effective_learning_rates(args.n_steps_per_phase),
        "base_shared_learning_rates": SHARED_LEARNING_RATES,
        "uniform_ff_noise": UNIFORM_FF_NOISE,
        "uniform_fb_noise": UNIFORM_FB_NOISE,
        "uniform_gain_clip": UNIFORM_GAIN_CLIP,
        "narrow_gain_clip": NARROW_GAIN_CLIP,
        "width_classes": WIDTH_CLASSES,
        "ff_strengths": FF_STRENGTHS,
        "fb_levels": FB_LEVELS,
        "context_modes": CONTEXT_MODES,
        "surround_settings": SURROUND_SETTINGS,
        "templates": {name: {k: v for k, v in t.items()} for name, t in TEMPLATES.items()},
    }
    path.write_text(json.dumps(metadata, indent=2, default=repr))
