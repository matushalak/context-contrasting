"""Systematic mini model-scatter variant.

This entrypoint keeps the fixed-PV mini assumptions, but replaces the historical
transition-template table with a mechanistic grid:

    FF tuning width x FF initial strength x FB level x surround level

The width-0 templates have no FF-strength axis. Width 1 and 2 templates draw a
uniform FF tuning permutation per sampled cell, then apply FF noise around the
permuted center. FB is general: weak and strong FB are equal across all three
context channels. The surround level couples initial w_LAT, W_pv, and w_pvLAT;
only w_LAT is plastic during training.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from context_contrasting.model_scatter import run_model_scatter as base


DEFAULT_OUTPUT_DIR = base.PACKAGE_DIR / "outputs_systematic_mini"
GENERIC_BASE_CONFIG = "FF_FB_broad"
REFERENCE_COMMIT = "30fa7296cadba542955a8f05d33918a186ba50ca"
N_FEATURES = 3


@dataclass(frozen=True)
class WidthClass:
    width: int
    sampling_weight: float
    ff_plasticity_scale: float
    apical_gain_center: float
    apical_drive_center: float
    baseline_center: float
    apical_gain_clip: tuple[float, float]
    apical_drive_clip: tuple[float, float | None]
    baseline_clip: tuple[float, float]


@dataclass(frozen=True)
class FFStrength:
    name: str
    sampling_weight: float
    tuned_center: float
    silent_center: float
    rel_noise: float
    noise_floor: float
    tuned_lo: float
    silent_lo: float
    tuned_hi: float
    silent_hi: float


@dataclass(frozen=True)
class FBLevel:
    name: str
    sampling_weight: float
    receives_context: bool
    center: float
    rel_noise: float
    noise_floor: float
    lo: float
    hi: float


@dataclass(frozen=True)
class InhibitionSpec:
    lat: float
    pvlat: float
    pv_tuned: float
    pv_silent: float


@dataclass(frozen=True)
class SurroundLevel:
    name: str
    sampling_weight: float
    lat_multiplier: float
    pvlat_multiplier: float
    pv_multiplier: float


WIDTH_CLASSES: OrderedDict[str, WidthClass] = OrderedDict(
    [
        (
            "ff0",
            WidthClass(
                width=0,
                sampling_weight=0.18,
                ff_plasticity_scale=0.01,
                apical_gain_center=5.6,
                apical_drive_center=0.16,
                baseline_center=0.34,
                apical_gain_clip=(3.5, 8.0),
                apical_drive_clip=(0.08, 0.36),
                baseline_clip=(0.24, 0.52),
            ),
        ),
        (
            "ff1",
            WidthClass(
                width=1,
                sampling_weight=0.33,
                ff_plasticity_scale=0.05,
                apical_gain_center=11.0,
                apical_drive_center=1.35,
                baseline_center=0.12,
                apical_gain_clip=(6.0, 14.0),
                apical_drive_clip=(1.05, None),
                baseline_clip=(0.085, 0.18),
            ),
        ),
        (
            "ff2",
            WidthClass(
                width=2,
                sampling_weight=0.31,
                ff_plasticity_scale=1.4,
                apical_gain_center=7.5,
                apical_drive_center=0.85,
                baseline_center=0.18,
                apical_gain_clip=(4.5, 9.0),
                apical_drive_clip=(0.45, None),
                baseline_clip=(0.12, 0.28),
            ),
        ),
        (
            "ff3",
            WidthClass(
                width=3,
                sampling_weight=0.21,
                ff_plasticity_scale=5.2,
                apical_gain_center=4.5,
                apical_drive_center=0.24,
                baseline_center=0.28,
                apical_gain_clip=(3.0, 6.0),
                apical_drive_clip=(0.12, 0.50),
                baseline_clip=(0.18, 0.40),
            ),
        ),
    ]
)

FF0_STRENGTH = FFStrength(
    name="unresponsive",
    sampling_weight=1.0,
    tuned_center=0.010,
    silent_center=0.010,
    rel_noise=0.65,
    noise_floor=0.012,
    tuned_lo=0.0,
    silent_lo=0.0,
    tuned_hi=0.050,
    silent_hi=0.050,
)

FF_STRENGTHS: OrderedDict[str, FFStrength] = OrderedDict(
    [
        (
            "weak",
            FFStrength(
                name="weak",
                sampling_weight=0.25,
                tuned_center=0.0,
                silent_center=0.008,
                rel_noise=0.42,
                noise_floor=0.010,
                tuned_lo=0.0,
                silent_lo=0.0,
                tuned_hi=0.0,
                silent_hi=0.030,
            ),
        ),
        (
            "strong",
            FFStrength(
                name="strong",
                sampling_weight=0.75,
                tuned_center=0.0,
                silent_center=0.008,
                rel_noise=0.34,
                noise_floor=0.012,
                tuned_lo=0.0,
                silent_lo=0.0,
                tuned_hi=0.0,
                silent_hi=0.030,
            ),
        ),
    ]
)

FF_STRENGTH_BY_WIDTH: dict[int, dict[str, tuple[float, float, float]]] = {
    1: {
        "weak": (0.085, 0.035, 0.165),
        "strong": (0.165, 0.075, 0.300),
    },
    2: {
        "weak": (0.085, 0.040, 0.165),
        "strong": (0.180, 0.080, 0.320),
    },
    3: {
        "weak": (0.085, 0.040, 0.170),
        "strong": (0.190, 0.080, 0.340),
    },
}

FB_LEVELS: OrderedDict[str, FBLevel] = OrderedDict(
    [
        (
            "fb_none",
            FBLevel(
                name="none",
                sampling_weight=0.14,
                receives_context=False,
                center=0.004,
                rel_noise=0.65,
                noise_floor=0.005,
                lo=0.0,
                hi=0.030,
            ),
        ),
        (
            "fb_weak",
            FBLevel(
                name="weak",
                sampling_weight=0.52,
                receives_context=True,
                center=0.045,
                rel_noise=0.42,
                noise_floor=0.010,
                lo=0.006,
                hi=0.120,
            ),
        ),
        (
            "fb_strong",
            FBLevel(
                name="strong",
                sampling_weight=0.34,
                receives_context=True,
                center=0.360,
                rel_noise=0.30,
                noise_floor=0.024,
                lo=0.160,
                hi=0.700,
            ),
        ),
    ]
)

SURROUND_LEVELS: OrderedDict[str, SurroundLevel] = OrderedDict(
    [
        (
            "surround_weak",
            SurroundLevel(
                name="weak",
                sampling_weight=0.55,
                lat_multiplier=0.72,
                pvlat_multiplier=0.72,
                pv_multiplier=0.80,
            ),
        ),
        (
            "surround_strong",
            SurroundLevel(
                name="strong",
                sampling_weight=0.45,
                lat_multiplier=1.05,
                pvlat_multiplier=0.88,
                pv_multiplier=1.55,
            ),
        ),
    ]
)

LEGACY_CONFIG_MAPPING = [
    ("un_un", "ff0_fb_none_surround_*", "Untuned FF and no received FB."),
    ("un_FB", "ff0_fb_weak_surround_strong", "Untuned FF plus weak generalized FB; initially silent/high-surround part of the distribution."),
    ("fb_fb_weak", "ff0_fb_weak_surround_weak", "Untuned FF plus weak generalized FB; weak naive O-responsive/lower-surround part of the distribution."),
    ("FB_FB", "ff0_fb_strong_surround_*", "Untuned FF plus strong generalized driving FB."),
    ("weak_FF", "ff1_weak_fb_weak_surround_*", "Weak narrow FF; FB increases gain on NO with little FF adaptation."),
    ("un_novel_FF", "ff1_strong_fb_weak_surround_strong", "Strong narrow FF; permutation tuned to novel with strong initial surround."),
    ("FF_FB_narrow_familiar", "ff1_strong_fb_weak_surround_*", "Strong narrow FF; permutation tuned to one familiar input."),
    ("FF_FB_narrow_familiar_2", "ff1_strong_fb_weak_surround_*", "Strong narrow FF; permutation tuned to one familiar input."),
    ("FF_FB_narrow_familiar_novel", "ff1_strong_fb_weak_surround_*", "Strong narrow FF; permutation may select familiar or novel input."),
    ("FF_FB_narrow_novel", "ff1_strong_fb_weak_surround_*", "Strong narrow FF; permutation tuned to novel."),
    ("weak_FB", "ff2_weak_fb_weak_surround_* / ff3_weak_fb_weak_surround_*", "Weak broad FF with weak generalized FB."),
    ("FF_un", "ff2_*_fb_none_surround_* / ff3_*_fb_none_surround_*", "Broad FF with no received FB."),
    ("FF_FB_broad", "ff2_strong_fb_weak/strong_surround_* / ff3_strong_fb_weak/strong_surround_*", "Strong broad FF with generalized FB."),
    ("FF_FB_broad_weak", "ff2_weak_fb_weak/strong_surround_* / ff3_weak_fb_weak/strong_surround_*", "Weak broad FF with generalized FB."),
    ("FF_FB_broad_novel", "ff2_strong_*_surround_* / ff3_strong_*_surround_*", "Strong broad FF whose sampled tuned set includes novel."),
    ("FF_FB_narrow_familiar_2_novel", "ff2_strong_*_surround_*", "Subsumed by width-2 strong FF tuned to one familiar input and novel."),
    ("O_un", "ff2_strong_fb_strong_surround_* / ff3_strong_fb_strong_surround_*", "Broad strong FF plus strong FB with intermediate surround."),
]


def _strength_options(width_key: str) -> OrderedDict[str, FFStrength]:
    width = WIDTH_CLASSES[width_key].width
    if width == 0:
        return OrderedDict([("", FF0_STRENGTH)])
    options: OrderedDict[str, FFStrength] = OrderedDict()
    for strength_key, strength in FF_STRENGTHS.items():
        center, lo, hi = FF_STRENGTH_BY_WIDTH[width][strength_key]
        options[strength_key] = FFStrength(
            name=strength.name,
            sampling_weight=strength.sampling_weight,
            tuned_center=center,
            silent_center=strength.silent_center,
            rel_noise=strength.rel_noise,
            noise_floor=strength.noise_floor,
            tuned_lo=lo,
            silent_lo=strength.silent_lo,
            tuned_hi=hi,
            silent_hi=strength.silent_hi,
        )
    return options


def _vector(width: int, tuned_indices: tuple[int, ...], tuned: float, silent: float) -> list[float]:
    if width == 0:
        return [float(silent)] * N_FEATURES
    tuned_set = set(tuned_indices)
    return [float(tuned if idx in tuned_set else silent) for idx in range(N_FEATURES)]


def _general_vector(value: float) -> list[float]:
    return [float(value)] * N_FEATURES


def _tuned_index_options(width: int) -> list[tuple[int, ...]]:
    if width == 0:
        return [()]
    return list(itertools.combinations(range(N_FEATURES), width))


def _draw_tuned_indices(width: int, rng: np.random.Generator) -> tuple[int, ...]:
    options = _tuned_index_options(width)
    return options[int(rng.integers(0, len(options)))]


def _canonical_tuned_indices(width: int) -> tuple[int, ...]:
    return tuple(range(width))


def _transition_name(width_key: str, strength_key: str, fb_key: str, surround_key: str) -> str:
    surround = SURROUND_LEVELS[surround_key].name
    if WIDTH_CLASSES[width_key].width == 0:
        return f"{width_key}_{fb_key}_surround_{surround}"
    return f"{width_key}_{strength_key}_{fb_key}_surround_{surround}"


def _surround_sampling_weight(width: int, strength_name: str, fb_name: str, surround_name: str) -> float:
    del strength_name
    if fb_name == "strong":
        weights = {"weak": 0.25, "strong": 0.75}
    elif fb_name == "weak" and width >= 2:
        weights = {"weak": 0.65, "strong": 0.35}
    else:
        weights = {"weak": 0.55, "strong": 0.45}
    return weights[surround_name]


def _inhibition_spec(width: int, strength_name: str, fb_name: str, surround_name: str) -> InhibitionSpec:
    strength_factor = {"unresponsive": 0.0, "weak": 0.45, "strong": 1.0}[strength_name]
    fb_factor = {"none": 0.0, "weak": 0.55, "strong": 1.0}[fb_name]
    surround = next(level for level in SURROUND_LEVELS.values() if level.name == surround_name)
    width_factor = width / 3.0

    lat = 0.030 + 0.045 * width_factor + 0.025 * strength_factor + 0.032 * fb_factor
    pvlat = 0.045 + 0.060 * width_factor + 0.045 * strength_factor + 0.055 * fb_factor
    pv_tuned = 0.075 + 0.105 * width_factor + 0.060 * strength_factor + 0.075 * fb_factor
    pv_silent = 0.060 + 0.045 * width_factor + 0.030 * strength_factor + 0.055 * fb_factor

    if width == 0 and fb_name == "weak":
        # Intermediate surround: scalar noise separates initially silent un_FB-like
        # cells from weak naive O-responsive fb_fb_weak-like cells.
        lat += 0.020
        pvlat += 0.025
        pv_tuned += 0.030
        pv_silent += 0.030
    if width == 0 and fb_name == "strong":
        lat += 0.015
        pv_tuned += 0.160
        pv_silent += 0.160
    if width in {1, 2} and fb_name == "weak":
        lat -= 0.015
        pvlat -= 0.020
        pv_tuned -= 0.030
        pv_silent -= 0.030
    if fb_name == "strong" and width >= 2:
        # O_un-like strong broad responders need enough surround to remove NO
        # while preserving/increasing O after training.
        lat += 0.025
        pvlat += 0.030

    return InhibitionSpec(
        lat=round(max(lat * surround.lat_multiplier, 0.005), 6),
        pvlat=round(max(pvlat * surround.pvlat_multiplier, 0.005), 6),
        pv_tuned=round(max(pv_tuned * surround.pv_multiplier, 0.005), 6),
        pv_silent=round(max(pv_silent * surround.pv_multiplier, 0.005), 6),
    )


def _build_template_spec(
    width_key: str,
    strength_key: str,
    fb_key: str,
    surround_key: str,
    *,
    tuned_indices: tuple[int, ...],
    pv_init_scale: float,
    pvlat_init_scale: float,
) -> dict[str, Any]:
    width_class = WIDTH_CLASSES[width_key]
    strength = _strength_options(width_key)[strength_key]
    fb = FB_LEVELS[fb_key]
    surround = SURROUND_LEVELS[surround_key]
    inhibition = _inhibition_spec(width_class.width, strength.name, fb.name, surround.name)
    surround_weight = _surround_sampling_weight(width_class.width, strength.name, fb.name, surround.name)
    sampling_weight = width_class.sampling_weight * strength.sampling_weight * fb.sampling_weight * surround_weight
    receives_context = (fb.receives_context,) * N_FEATURES

    return base.transition(
        sampling_weight,
        fixed={
            "receives_context": receives_context,
            "ff_plasticity_scale": width_class.ff_plasticity_scale,
            "apical_gain_strength": width_class.apical_gain_center,
            "apical_drive_threshold": width_class.apical_drive_center,
            "baseline_drive_sigma": width_class.baseline_center,
            "pv_plasticity": False,
            "pv_lat_plasticity": False,
        },
        clip={
            "apical_gain_strength": width_class.apical_gain_clip,
            "apical_drive_threshold": width_class.apical_drive_clip,
            "baseline_drive_sigma": width_class.baseline_clip,
        },
        ff=base.weight_init(
            _vector(width_class.width, tuned_indices, strength.tuned_center, strength.silent_center),
            strength.rel_noise,
            strength.noise_floor,
            _vector(width_class.width, tuned_indices, strength.tuned_lo, strength.silent_lo),
            _vector(width_class.width, tuned_indices, strength.tuned_hi, strength.silent_hi),
        ),
        fb=base.weight_init(
            _general_vector(fb.center),
            fb.rel_noise,
            fb.noise_floor,
            _general_vector(fb.lo),
            _general_vector(fb.hi),
        ),
        lat=base.weight_init([inhibition.lat]),
        pvlat=base.weight_init([inhibition.pvlat * pvlat_init_scale]),
        pv=base.weight_init(
            [
                value * pv_init_scale
                for value in _vector(width_class.width, tuned_indices, inhibition.pv_tuned, inhibition.pv_silent)
            ]
        ),
    )


def _template_lookup() -> dict[str, tuple[str, str, str, str]]:
    return {
        _transition_name(width_key, strength_key, fb_key, surround_key): (width_key, strength_key, fb_key, surround_key)
        for width_key in WIDTH_CLASSES
        for strength_key in _strength_options(width_key)
        for fb_key in FB_LEVELS
        for surround_key in SURROUND_LEVELS
    }


def _build_systematic_transitions(args: argparse.Namespace) -> OrderedDict[str, dict[str, Any]]:
    transitions: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for width_key, width_class in WIDTH_CLASSES.items():
        tuned_indices = _canonical_tuned_indices(width_class.width)
        for strength_key in _strength_options(width_key):
            for fb_key in FB_LEVELS:
                for surround_key in SURROUND_LEVELS:
                    transitions[_transition_name(width_key, strength_key, fb_key, surround_key)] = _build_template_spec(
                        width_key,
                        strength_key,
                        fb_key,
                        surround_key,
                        tuned_indices=tuned_indices,
                        pv_init_scale=args.pv_init_scale,
                        pvlat_init_scale=args.pvlat_init_scale,
                    )
    return transitions


def _apply_shared_learning_rates(args: argparse.Namespace) -> dict[str, float]:
    shared_lrs = dict(base.SHARED_LEARNING_RATES)
    shared_lrs["lr_ff"] *= args.ff_lr_scale
    shared_lrs["lr_fb"] *= args.fb_lr_scale
    shared_lrs["lr_lat"] *= args.lat_lr_scale
    shared_lrs["lr_pv"] = 0.0
    return shared_lrs


def _systematic_perturb_config_factory(args: argparse.Namespace):
    lookup = _template_lookup()

    def _systematic_perturb_config(
        transition: str,
        base_config: dict[str, Any],
        *,
        sample_idx: int,
        global_idx: int,
        seed: int,
        rng: np.random.Generator,
        scalar_noise_multiplier: float,
    ) -> dict[str, Any]:
        width_key, strength_key, fb_key, surround_key = lookup[transition]
        width_class = WIDTH_CLASSES[width_key]
        strength = _strength_options(width_key)[strength_key]
        fb = FB_LEVELS[fb_key]
        surround = SURROUND_LEVELS[surround_key]
        tuned_indices = _draw_tuned_indices(width_class.width, rng)
        spec = _build_template_spec(
            width_key,
            strength_key,
            fb_key,
            surround_key,
            tuned_indices=tuned_indices,
            pv_init_scale=args.pv_init_scale,
            pvlat_init_scale=args.pvlat_init_scale,
        )

        config = copy.deepcopy(base_config)
        for key, init_spec in spec["init"].items():
            if key in {"w_ff_init", "w_fb_init"}:
                values = base._draw_init(init_spec, rng)
            else:
                values = base._center_init_values(init_spec)
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
            _ff_tuning_width=int(width_class.width),
            _ff_strength=strength.name,
            _fb_level=fb.name,
            _surround_level=surround.name,
            _tuned_indices=list(tuned_indices),
        )
        return config

    return _systematic_perturb_config


def _flatten_config_factory(original_flatten):
    def _flatten_config(config: dict[str, Any]) -> dict[str, Any]:
        flat = original_flatten(config)
        flat["ff_tuning_width"] = config.get("_ff_tuning_width")
        flat["ff_strength"] = config.get("_ff_strength")
        flat["fb_level"] = config.get("_fb_level")
        flat["surround_level"] = config.get("_surround_level")
        tuned = config.get("_tuned_indices", [])
        for idx in range(N_FEATURES):
            flat[f"tuned_index_{idx}"] = int(idx in tuned)
        return flat

    return _flatten_config


def _configure_systematic_variant(args: argparse.Namespace) -> None:
    transitions = _build_systematic_transitions(args)
    generic_config = copy.deepcopy(base.minimal_configs3[GENERIC_BASE_CONFIG])

    base.TRANSITIONS = transitions
    base.minimal_configs3 = OrderedDict((name, copy.deepcopy(generic_config)) for name in transitions)
    base.SHARED_LEARNING_RATES = _apply_shared_learning_rates(args)
    base._perturb_config = _systematic_perturb_config_factory(args)
    base._flatten_config = _flatten_config_factory(base._flatten_config)


def _serializable_width_classes() -> dict[str, Any]:
    return {key: value.__dict__ for key, value in WIDTH_CLASSES.items()}


def _serializable_ff_strengths() -> dict[str, Any]:
    return {
        width_key: {strength_key or "unresponsive": strength.__dict__ for strength_key, strength in _strength_options(width_key).items()}
        for width_key in WIDTH_CLASSES
    }


def _serializable_fb_levels() -> dict[str, Any]:
    return {key: value.__dict__ for key, value in FB_LEVELS.items()}


def _serializable_surround_levels() -> dict[str, Any]:
    return {key: value.__dict__ for key, value in SURROUND_LEVELS.items()}


def _serializable_inhibition_by_template() -> dict[str, Any]:
    specs: dict[str, Any] = {}
    for transition, (width_key, strength_key, fb_key, surround_key) in _template_lookup().items():
        width_class = WIDTH_CLASSES[width_key]
        strength = _strength_options(width_key)[strength_key]
        fb = FB_LEVELS[fb_key]
        surround = SURROUND_LEVELS[surround_key]
        specs[transition] = _inhibition_spec(width_class.width, strength.name, fb.name, surround.name).__dict__
    return specs


def _serializable_surround_sampling_by_template() -> dict[str, float]:
    weights: dict[str, float] = {}
    for transition, (width_key, strength_key, fb_key, surround_key) in _template_lookup().items():
        width_class = WIDTH_CLASSES[width_key]
        strength = _strength_options(width_key)[strength_key]
        fb = FB_LEVELS[fb_key]
        surround = SURROUND_LEVELS[surround_key]
        weights[transition] = _surround_sampling_weight(width_class.width, strength.name, fb.name, surround.name)
    return weights


def _write_legacy_mapping(output_dir: Path) -> None:
    path = output_dir / "legacy_config_mapping.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["legacy_config", "systematic_template", "notes"])
        writer.writerows(LEGACY_CONFIG_MAPPING)


def _write_systematic_metadata(args: argparse.Namespace) -> None:
    metadata_path = args.output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["systematic_mini_variant"] = {
        "description": "Fixed-PV mini variant with FF width x FF strength x generalized FB level x coupled surround-level templates.",
        "reference_commit": args.reference_commit,
        "n_templates": len(_template_lookup()),
        "template_axes": {
            "ff_tuning_widths": list(WIDTH_CLASSES),
            "ff_strengths": ["weak", "strong"],
            "fb_levels": list(FB_LEVELS),
            "surround_levels": list(SURROUND_LEVELS),
            "width0_has_ff_strength_axis": False,
        },
        "width_classes": _serializable_width_classes(),
        "ff_strengths_by_width": _serializable_ff_strengths(),
        "fb_levels": _serializable_fb_levels(),
        "surround_levels": _serializable_surround_levels(),
        "surround_sampling_by_template": _serializable_surround_sampling_by_template(),
        "inhibition_by_template": _serializable_inhibition_by_template(),
        "legacy_config_mapping": [
            {"legacy_config": legacy, "systematic_template": template, "notes": notes}
            for legacy, template, notes in LEGACY_CONFIG_MAPPING
        ],
        "permutation_sampling": "For each sampled cell, FF tuned stimulus indices are drawn uniformly among combinations of the template width before FF noise is applied. FB is generalized and equal across all context channels. Surround weak/strong couples initial w_LAT, W_pv, and w_pv_lat.",
        "disabled_plasticity": ["W_pv", "w_pv_lat"],
        "enabled_plasticity": ["w_ff", "w_fb", "w_lat"],
        "pv_init_scale": args.pv_init_scale,
        "pvlat_init_scale": args.pvlat_init_scale,
        "ff_lr_scale": args.ff_lr_scale,
        "fb_lr_scale": args.fb_lr_scale,
        "lat_lr_scale": args.lat_lr_scale,
        "shared_learning_rates": base.SHARED_LEARNING_RATES,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=repr))
    _write_legacy_mapping(args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the systematic mini model-scatter variant with 42 mechanistic transition templates."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-samples", type=int, default=1200, help="Number of model cells to draw from the transition mixture.")
    parser.add_argument("--n-steps-per-phase", type=int, default=200, help="Time steps per stimulus trial.")
    parser.add_argument("--test-trials", type=int, default=2, help="Repeats of each probe stimulus at naive and expert.")
    parser.add_argument("--training-trials", type=int, default=5, help="Repeats of the familiar-image training block.")
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers (joblib); -1 uses all cores.")
    parser.add_argument("--scalar-noise-multiplier", type=float, default=1.75)
    parser.add_argument("--canonical-only", action="store_true")
    parser.add_argument("--transition-sampling", choices=("data-like", "equal"), default="data-like")
    parser.add_argument("--zscore-std-floor", type=float, default=0.04)
    parser.add_argument("--response-tail-fraction", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--image-format", choices=("png", "svg", "eps"), default="png")
    parser.add_argument("--axis-clip-percentile", type=float, default=99.0)
    parser.add_argument(
        "--with-center-panels",
        dest="skip_center_panels",
        action="store_false",
        default=True,
        help="Also render center/canonical transition panels. Off by default because permutation sampling has no single canonical orientation.",
    )
    parser.add_argument("--plot-by-transition", action="store_true")
    parser.add_argument("--export-panels", action="store_true")

    parser.add_argument("--pv-init-scale", type=float, default=1.5, help="Multiplier for fixed W_pv template centers.")
    parser.add_argument("--pvlat-init-scale", type=float, default=1.0, help="Multiplier for fixed w_pv_lat template centers.")
    parser.add_argument("--ff-lr-scale", type=float, default=1.0, help="Multiplier for the shared FF learning rate.")
    parser.add_argument("--fb-lr-scale", type=float, default=2.5, help="Multiplier for the shared FB learning rate.")
    parser.add_argument("--lat-lr-scale", type=float, default=1.7, help="Multiplier for the shared LAT learning rate.")
    parser.add_argument("--reference-commit", default=REFERENCE_COMMIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.canonical_only:
        raise ValueError("--canonical-only is not supported by the systematic mini sampler; use --with-center-panels for deterministic center checks.")
    _configure_systematic_variant(args)
    base.run_model_scatter(args)
    _write_systematic_metadata(args)


if __name__ == "__main__":
    main()
