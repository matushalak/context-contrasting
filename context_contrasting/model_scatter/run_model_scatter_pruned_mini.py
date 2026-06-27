"""Pruned, principled mini model-scatter variant.

Keeps the mini model assumptions -- only the PyC synapses ``w_ff``, ``w_fb`` and
``w_lat`` learn (PV tuning ``W_pv`` / ``w_pv_lat`` is fixed), and per-cell noise is
injected only into the FF/FB initial weights, ``apical_drive_threshold`` and
``apical_gain_strength`` -- but replaces the 17
hand-tuned, partly ad-hoc transition templates with a small set keyed entirely by
**tuning width**:

  * ``silent``: near-silent FF drive. With no feedback it stays near the origin
    (small dNO/dO); with strong generalized feedback it is a feedback-driven
    occluded ("O") responder.
  * ``narrow`` (width 1): weak FF (anti-Hebbian) plasticity + a high apical drive
    threshold, so feedback can only *gain-modulate*. The non-adapted FF drive is
    amplified into a ``+NO`` response. Each narrow cell's single preferred
    stimulus is drawn at random per cell, so the familiar-vs-novel ``+NO``
    asymmetry **emerges** from the protocol (only the familiar images are shown
    during the plastic phase, so their FF adapts while the novel image's does not)
    rather than being hand-assigned to separate familiar/novel templates.
  * ``broad`` (width 3): strong FF plasticity + a low apical drive threshold, so
    familiar FF adapts away (``-NO``) and the strengthened, generalized feedback
    drives the occluded response (``+O``).

Two principled choices distinguish this from the legacy templates: feedback is
always **generalized over the context channels a cell receives** (no hand-painted
per-channel weight asymmetry), and narrow/broad tuning is the *only* thing that
sets the FF-plasticity scale and the feedback drive-vs-gain regime. Surround
weights (``w_lat``, ``W_pv``, ``w_pv_lat``) are coupled per template; only
``w_lat`` is plastic.

The width-class scalars and the mixture weights at the top of this file are the
tuning levers; everything downstream reuses ``run_model_scatter.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from context_contrasting.model_scatter import run_model_scatter as base
from context_contrasting.utils import ThresholdReLU


DEFAULT_OUTPUT_DIR = base.PACKAGE_DIR / "outputs_pruned_mini"
GENERIC_BASE_CONFIG = "FF_FB_broad"
N_FEATURES = 3

# Only PyC synapses learn; PV feedforward learning is off (lr_pv = 0). Feedback
# strengthening and lateral surround learning stay shared across templates. These
# rates are calibrated at 200 steps/phase; longer phase durations scale them down
# so the accumulated plasticity stays close to the clearer 200-step scatter.
LEARNING_RATE_REFERENCE_STEPS = 200
SHARED_LEARNING_RATES = {"lr_ff": 0.0135, "lr_fb": 0.00062, "lr_lat": 0.0052, "lr_pv": 0.0}
SCALAR_NOISE_KEYS = ("apical_gain_strength", "apical_drive_threshold")
SOMA_ACTIVATION_THRESHOLD = 0.08


def _effective_learning_rates(n_steps_per_phase: int) -> dict[str, float]:
    if n_steps_per_phase <= 0:
        raise ValueError("n_steps_per_phase must be positive.")
    scale = LEARNING_RATE_REFERENCE_STEPS / float(n_steps_per_phase)
    return {
        name: (float(rate) * scale if name != "lr_pv" else float(rate))
        for name, rate in SHARED_LEARNING_RATES.items()
    }

# Two tuning-width classes only. BROAD: strong FF (anti-Hebbian) plasticity + a low
# apical drive threshold, so feedback DRIVES the soma. NARROW: weak FF plasticity +
# a high drive threshold, so feedback only GAIN-modulates. "Untuned" cells are
# modeled as weak / already-adapted BROAD cells (a broad cell whose FF was depressed
# by past adaptation), so they share the broad width class and merely carry a
# near-silent feedforward weight -- there is no separate untuned regime. Each entry
# gives the fixed center and post-noise clip for the jittered scalars
# (apical_gain_strength, apical_drive_threshold, baseline_drive_sigma).
WIDTH_CLASSES: dict[str, dict[str, Any]] = {
    "broad": dict(
        ff_plasticity_scale=8.0,
        # Broad cells need enough gain that intact novel FF drive can be amplified
        # alongside strengthened generalized FB; otherwise novel expert NO gets
        # suppressed even for mixed FF+FB responders.
        gain=3.8, gain_clip=(2.8, 5.8),
        drive=0.16, drive_clip=(0.10, 0.45),
        baseline=0.16, baseline_clip=(0.12, 0.26),
    ),
    "narrow": dict(
        # Weak (not zero) FF plasticity: a narrow cell tuned to a *familiar* image
        # partially adapts its FF during training and is pulled back toward
        # small/-NO, while one tuned to the *novel* image keeps its FF intact and is
        # amplified to +NO -- so the familiar<novel +NO asymmetry emerges from the
        # protocol rather than being hand-assigned.
        ff_plasticity_scale=0.55,
        gain=5.8, gain_clip=(4.0, 8.5),
        # Shift + steepen the gain sigmoid so the small feedback growth during
        # training moves the gain enough to turn the non-adapted FF drive into a
        # clear +NO shift at expert (suppressed naive -> amplified expert).
        gain_threshold=0.03, gain_k=8.0,
        drive=1.25, drive_clip=(1.05, None),
        # Low baseline -> low adaptation current, so the occluded basal stays near 0
        # and the rising gain barely drags the occluded response negative: the +NO
        # cells stay centred at O~0 (their NO shift is FF-driven, baseline-independent).
        baseline=0.07, baseline_clip=(0.050, 0.10),
    ),
}

# Shared feedforward initial-weight levels. These are independent of tuning width:
# the tuned value is assigned to preferred stimulus channels and the silent value
# to non-preferred channels. Broad cells simply tune all channels, while narrow
# cells tune the drawn preferred channel only.
FF_STRENGTHS: dict[str, dict[str, Any]] = {
    "silent": dict(tuned=0.010, silent=0.010, rel=0.60, floor=0.012, lo=0.0, hi=0.050),
    "weak": dict(tuned=0.090, silent=0.006, rel=0.50, floor=0.010, lo=0.0, hi=0.220),
    "mid": dict(tuned=0.120, silent=0.006, rel=0.46, floor=0.010, lo=0.0, hi=0.250),
    "strong": dict(tuned=0.155, silent=0.008, rel=0.34, floor=0.014, lo=0.0, hi=0.300),
}

# Generalized feedback levels (equal across all three context channels):
# (receives_context, center, rel_noise, noise_floor, lo, hi).
FB_LEVELS: dict[str, dict[str, Any]] = {
    "none": dict(receives=False, center=0.004, rel=0.60, floor=0.005, lo=0.0, hi=0.030),
    "weak": dict(receives=True, center=0.040, rel=0.42, floor=0.010, lo=0.006, hi=0.120),
    "mid": dict(receives=True, center=0.075, rel=0.40, floor=0.012, lo=0.012, hi=0.170),
    "strong": dict(receives=True, center=0.110, rel=0.28, floor=0.016, lo=0.050, hi=0.300),
}

# Which feedforward channels each cell prefers (drawn per cell):
#   "all"       -> tuned to every image (broad)
#   "permuted1" -> one random preferred image (narrow; emergent fam/novel asymmetry)
#   "novel"     -> the novel image, index 2 (the narrow_novel special case)
NOVEL_INDEX = 2

# Which context channels each feedback-receiving cell receives. The FB weights are
# still drawn from the shared generalized FB levels; this only masks whether a
# channel exists for a subclass of cells.
CONTEXT_MODES = ("none", "all", "random1", "random2", "familiar", "novel")

# The pruned population mixture. Each template = (width, FF strength, FB level,
# tuning, context-receive mode) plus its coupled surround (w_lat, w_pv_lat, W_pv on
# tuned/silent channels), optional baseline override, and optional per-template
# scalar overrides (gain etc.). `weight` is its share of the population.
TEMPLATES: dict[str, dict[str, Any]] = {
    # Silent / already-adapted broad cell, no feedback -> unresponsive (small dNO/dO).
    "silent_broad_FFonly": dict(
        width="broad", ff="silent", fb="none", tuning="all", context="none", weight=0.12,
        lat=0.04, pvlat=0.08, pv_tuned=0.18, pv_silent=0.18,
    ),
    # Silent broad + weak generalized FB: starts close to the unresponsive cloud,
    # then shared FB/LAT plasticity moves it vertically into the O cloud while the
    # learned full-image surround keeps expert NO near 0.
    "silent_broad_FB_weak": dict(
        width="broad", ff="silent", fb="weak", tuning="all", context="all", weight=0.040,
        drive=0.035, drive_clip=(0.0, 0.10),
        gain=3.0, gain_clip=(1.4, 3.4),
        lat=0.08, pvlat=0.030, pv_tuned=0.55, pv_silent=0.55,
        baseline=0.18, baseline_clip=(0.13, 0.24),
    ),
    # FB->FB-like: little FF and moderate generalized FB. This supplies the missing
    # naive O-responder cloud. It still uses the same shared FB/LAT plasticity as the
    # other broad feedback templates; the stronger full-image surround keeps the
    # expert endpoint vertical rather than drifting to +NO.
    "silent_broad_FB_mid": dict(
        width="broad", ff="silent", fb="mid", tuning="all", context="all", weight=0.025,
        drive=0.035, drive_clip=(0.0, 0.10),
        gain=3.0, gain_clip=(1.4, 3.4),
        lat=0.10, pvlat=0.035, pv_tuned=0.60, pv_silent=0.60,
        baseline=0.18, baseline_clip=(0.13, 0.24),
    ),
    # Partial-context silent broad cells: same generalized FB on received channels,
    # but only one or two context inputs exist. These fill the weak/intermediate O
    # band without forcing every stimulus from the cell into the strong O cloud.
    "silent_broad_FB_partial2": dict(
        width="broad", ff="silent", fb="mid", tuning="all", context="random2", weight=0.090,
        drive=0.035, drive_clip=(0.0, 0.10),
        gain=3.0, gain_clip=(1.4, 3.4),
        lat=0.09, pvlat=0.035, pv_tuned=0.56, pv_silent=0.56,
        baseline=0.18, baseline_clip=(0.13, 0.24),
    ),
    # Mid broad FF, no feedback -> familiar FF adapts away -> pure -NO. Kept small:
    # on novel these are static NO cells (no FB -> no movement), which the data does
    # not show as the dominant broad population.
    "mid_broad_FFonly": dict(
        width="broad", ff="mid", fb="none", tuning="all", context="none", weight=0.160,
        lat=0.11, pvlat=0.12, pv_tuned=0.20, pv_silent=0.20,
    ),

    # Broad FF + generalized FB: the key medium NO/O -> expert O class. Familiar
    # FF adapts down while generalized FB grows; novel keeps its FF and should move
    # +NO and/or +O rather than being suppressed.
    "mid_broad_FB_weak": dict(
        width="broad", ff="mid", fb="weak", tuning="all", context="all", weight=0.095,
        drive=0.085, drive_clip=(0.060, 0.22),
        gain=4.2, lat=0.05, pvlat=0.035, pv_tuned=0.50, pv_silent=0.50,
        baseline=0.18, baseline_clip=(0.13, 0.26),
    ),
    "mid_broad_FB_partial2": dict(
        width="broad", ff="mid", fb="mid", tuning="all", context="random2", weight=0.075,
        drive=0.080, drive_clip=(0.055, 0.20),
        gain=4.0, lat=0.045, pvlat=0.035, pv_tuned=0.48, pv_silent=0.48,
        baseline=0.18, baseline_clip=(0.13, 0.26),
    ),
    "strong_broad_FB_strong": dict(
        width="broad", ff="strong", fb="strong", tuning="all", context="all", weight=0.065,
        drive=0.105, drive_clip=(0.070, 0.24),
        gain=4.6, lat=0.06, pvlat=0.04, pv_tuned=0.55, pv_silent=0.55,
        baseline=0.24, baseline_clip=(0.18, 0.34),
    ),
    # Narrow, one random preferred image -> +NO via gain; the emergent familiar/novel
    # +NO asymmetry. Weak and mid FF variants give a range of +NO magnitudes.
    "narrow_weak": dict(
        width="narrow", ff="weak", fb="weak", tuning="permuted1", context="all", weight=0.13,
        lat=0.03, pvlat=0.03, pv_tuned=0.045, pv_silent=0.03,
    ),
    "narrow_mid": dict(
        width="narrow", ff="mid", fb="weak", tuning="permuted1", context="all", weight=0.105,
        lat=0.03, pvlat=0.03, pv_tuned=0.045, pv_silent=0.03,
    ),
    # Special case: a narrowly NOVEL-tuned cell. A weak novel FF seed plus strong
    # generalized FB keeps naive responses near silent while giving FB growth enough
    # leverage to produce a qualitative silent -> novel-NO transition.
    "narrow_novel": dict(
        width="narrow", ff="weak", fb="weak", tuning="novel", context="all", weight=0.095,
        gain=6.5, gain_threshold=0.06,
        lat=0.03, pvlat=0.03, pv_tuned=0.045, pv_silent=0.03,
    ),
}

# Per-template scalar overrides of the width-class defaults.
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
    if tuning == "novel":
        return (NOVEL_INDEX,)
    if tuning == "permuted1":
        return (int(rng.integers(0, N_FEATURES)),)
    raise ValueError(f"unknown tuning mode: {tuning}")


def _draw_context_indices(mode: str, rng: np.random.Generator) -> tuple[int, ...]:
    if mode == "none":
        return ()
    if mode == "all":
        return tuple(range(N_FEATURES))
    if mode == "familiar":
        return (0, 1)
    if mode == "novel":
        return (NOVEL_INDEX,)
    if mode == "random1":
        return (int(rng.integers(0, N_FEATURES)),)
    if mode == "random2":
        return tuple(sorted(int(idx) for idx in rng.choice(N_FEATURES, size=2, replace=False)))
    raise ValueError(f"unknown context mode: {mode}")


def _canonical_context_indices(mode: str) -> tuple[int, ...]:
    if mode == "none":
        return ()
    if mode == "all":
        return tuple(range(N_FEATURES))
    if mode == "familiar":
        return (0, 1)
    if mode == "novel":
        return (NOVEL_INDEX,)
    if mode == "random1":
        return (0,)
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
    """Translate the TEMPLATES table into base ``transition`` specs (only their
    ``weight`` is read downstream; the real config is built in the perturb hook)."""
    specs: dict[str, dict[str, Any]] = {}
    for name, template in TEMPLATES.items():
        specs[name] = base.transition(template["weight"])
    return specs


def _build_config(
    name: str,
    tuned_indices: tuple[int, ...],
    context_indices: tuple[int, ...],
) -> dict[str, Any]:
    """Assemble the (noise-free) config for one cell of a template, given the drawn
    preferred-stimulus set ``tuned_indices``."""
    template = TEMPLATES[name]
    width_class = WIDTH_CLASSES[template["width"]]
    ff = FF_STRENGTHS[template["ff"]]
    fb = FB_LEVELS[template["fb"]]

    pv_vec = _vector(tuned_indices, template["pv_tuned"], template["pv_silent"])
    config: dict[str, Any] = {
        "w_ff_init": base.weight_init(
            _vector(tuned_indices, ff["tuned"], ff["silent"]),
            ff["rel"], ff["floor"],
            _vector(tuned_indices, ff["lo"], ff["lo"]),
            _vector(tuned_indices, ff["hi"], ff["hi"]),
        ),
        "w_fb_init": base.weight_init(
            _general_vector(fb["center"]), fb["rel"], fb["floor"],
            _general_vector(fb["lo"]), _general_vector(fb["hi"]),
        ),
        "w_lat_init": base.weight_init([template["lat"]]),
        "w_pv_lat_init": base.weight_init([template["pvlat"]]),
        "W_pv_init": base.weight_init(pv_vec),
    }
    fixed = {
        "receives_context": _bool_vector(context_indices) if fb["receives"] else (False,) * N_FEATURES,
        "baseline_drive_sigma": template.get("baseline", width_class["baseline"]),
        "pv_plasticity": False,
        "pv_lat_plasticity": False,
    }
    # Width-class scalars, with optional per-template overrides.
    for key, model_key in _SCALAR_OVERRIDES.items():
        value = template.get(key, width_class.get(key))
        if value is not None:
            fixed[model_key] = value
    clip = {
        "apical_gain_strength": template.get("gain_clip", width_class["gain_clip"]),
        "apical_drive_threshold": template.get("drive_clip", width_class["drive_clip"]),
        "baseline_drive_sigma": template.get("baseline_clip", width_class["baseline_clip"]),
    }
    return {"init": config, "fix": fixed, "clip": clip, "width": len(tuned_indices), "context_width": len(context_indices)}


def _perturb_config_factory():
    """Replacement for ``base._perturb_config`` that draws each cell's preferred
    stimulus (FF tuning) at random, then applies the usual FF/FB weight noise and
    scalar jitter and clipping."""

    def _perturb_config(
        transition: str,
        base_config: dict[str, Any],
        *,
        sample_idx: int,
        global_idx: int,
        seed: int,
        rng: np.random.Generator,
        scalar_noise_multiplier: float,
    ) -> dict[str, Any]:
        template = TEMPLATES[transition]
        tuned_indices = _draw_tuned_indices(template["tuning"], rng)
        context_indices = _draw_context_indices(template.get("context", "all"), rng)
        spec = _build_config(transition, tuned_indices, context_indices)
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
            _ff_tuning_width=int(spec["width"]),
            _ff_strength=template["ff"],
            _fb_level=template["fb"],
            _tuned_indices=list(tuned_indices),
            _context_indices=list(context_indices),
        )
        return config

    return _perturb_config


def _canonical_tuned_indices(tuning: str) -> tuple[int, ...]:
    """A fixed, representative preferred-stimulus set for a tuning mode, used for the
    noise-free center-panel traces (permuted cells are shown tuned to familiar_1)."""
    if tuning == "all":
        return tuple(range(N_FEATURES))
    if tuning == "novel":
        return (NOVEL_INDEX,)
    if tuning == "permuted1":
        return (0,)
    raise ValueError(f"unknown tuning mode: {tuning}")


def _center_config(name: str) -> dict[str, Any]:
    """Noise-free config at the template centers (no weight/scalar jitter), for the
    center-panel trace plots. Replaces ``base._center_config`` so each pruned
    template renders correctly."""
    template = TEMPLATES[name]
    tuned_indices = _canonical_tuned_indices(template["tuning"])
    context_indices = _canonical_context_indices(template.get("context", "all"))
    spec = _build_config(name, tuned_indices, context_indices)
    config = copy.deepcopy(base.minimal_configs3[name])
    for key, init_spec in spec["init"].items():
        base._set_init(config, key, base._center_init_values(init_spec))
    config.update(spec["fix"])
    for key, (lo, hi) in (base.GLOBAL_SCALAR_CLIP | spec["clip"]).items():
        if key in config and base._is_num(config[key]):
            config[key] = base._clip(float(config[key]), lo, hi)
    base._apply_shared_learning_rates(config)
    config.update(
        _canonical_transition=name,
        _sample_idx=0,
        _sample_global_idx=0,
        _ff_tuning_width=int(spec["width"]),
        _ff_strength=template["ff"],
        _fb_level=template["fb"],
        _tuned_indices=list(tuned_indices),
        _context_indices=list(context_indices),
    )
    return config


def _flatten_config_factory(original_flatten):
    def _flatten_config(config: dict[str, Any]) -> dict[str, Any]:
        flat = original_flatten(config)
        flat["ff_tuning_width"] = config.get("_ff_tuning_width")
        flat["ff_strength"] = config.get("_ff_strength")
        flat["fb_level"] = config.get("_fb_level")
        tuned = config.get("_tuned_indices", [])
        context = config.get("_context_indices", [])
        for idx in range(N_FEATURES):
            flat[f"tuned_index_{idx}"] = int(idx in tuned)
            flat[f"receives_context_{idx}"] = int(idx in context)
        return flat

    return _flatten_config


def _configure_pruned_variant(n_steps_per_phase: int = LEARNING_RATE_REFERENCE_STEPS) -> None:
    transitions = _build_transition_specs()
    generic_config = copy.deepcopy(base.minimal_configs3[GENERIC_BASE_CONFIG])
    generic_config["activation"] = ThresholdReLU(threshold=SOMA_ACTIVATION_THRESHOLD, hard=False)
    base.TRANSITIONS = transitions
    base.minimal_configs3 = {name: copy.deepcopy(generic_config) for name in transitions}
    base.SHARED_LEARNING_RATES = _effective_learning_rates(n_steps_per_phase)
    base.SCALAR_NOISE = {key: base.SCALAR_NOISE[key] for key in SCALAR_NOISE_KEYS}
    base._perturb_config = _perturb_config_factory()
    base._flatten_config = _flatten_config_factory(base._flatten_config)
    # Center-panel traces (one panel per template) use the noise-free template
    # centers; there is no separate config_s canonical example, so both panel sets
    # show the same template centers.
    base._center_config = _center_config
    # base._canonical_config = _center_config


def _write_metadata(args: argparse.Namespace) -> None:
    path = args.output_dir / "metadata.json"
    metadata = json.loads(path.read_text()) if path.exists() else {}
    metadata["pruned_mini_variant"] = {
        "description": "Mini model (only PyC plasticity) with a small width-keyed principled template set, generalized feedback, and emergent FF tuning via random per-cell permutation.",
        "disabled_plasticity": ["W_pv", "w_pv_lat"],
        "enabled_plasticity": ["w_ff", "w_fb", "w_lat"],
        "shared_learning_rates": _effective_learning_rates(args.n_steps_per_phase),
        "base_shared_learning_rates": SHARED_LEARNING_RATES,
        "learning_rate_reference_steps": LEARNING_RATE_REFERENCE_STEPS,
        "soma_activation_threshold": SOMA_ACTIVATION_THRESHOLD,
        "scalar_noise_keys": list(SCALAR_NOISE_KEYS),
        "width_classes": WIDTH_CLASSES,
        "ff_strengths": FF_STRENGTHS,
        "fb_levels": FB_LEVELS,
        "context_modes": CONTEXT_MODES,
        "templates": {name: {k: v for k, v in t.items()} for name, t in TEMPLATES.items()},
    }
    path.write_text(json.dumps(metadata, indent=2, default=repr))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pruned principled mini model-scatter variant.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-samples", type=int, default=250)
    parser.add_argument("--n-steps-per-phase", type=int, default=300)
    parser.add_argument("--test-trials", type=int, default=4)
    parser.add_argument("--training-trials", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--scalar-noise-multiplier", type=float, default=1.75)
    parser.add_argument("--canonical-only", action="store_true", help="Not supported (permutation sampling has no single canonical orientation).")
    parser.add_argument("--transition-sampling", choices=("data-like", "equal"), default="data-like")
    parser.add_argument("--zscore-std-floor", type=float, default=0.04)
    parser.add_argument("--response-tail-fraction", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--image-format", choices=("png", "svg", "eps"), default="png")
    parser.add_argument("--axis-clip-percentile", type=float, default=99.0)
    parser.add_argument("--plot-center-panels", action="store_true", help="Also render one naive->expert trace panel per transition template (off by default).")
    parser.add_argument("--plot-by-transition", action="store_true")
    parser.add_argument("--export-panels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.canonical_only:
        raise ValueError("--canonical-only is not supported by the pruned mini sampler (permutation sampling has no canonical orientation).")
    # Base runner skips the per-template trace panels when skip_center_panels is set;
    # expose the inverse opt-in flag here.
    args.skip_center_panels = not args.plot_center_panels
    _configure_pruned_variant(args.n_steps_per_phase)
    base.run_model_scatter(args)
    _write_metadata(args)


if __name__ == "__main__":
    main()
