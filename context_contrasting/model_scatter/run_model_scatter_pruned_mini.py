"""Pruned, principled mini model-scatter variant.

Keeps the mini model assumptions -- only the PyC synapses ``w_ff``, ``w_fb`` and
``w_lat`` learn (PV tuning ``W_pv`` / ``w_pv_lat`` is fixed), and per-cell noise is
injected only into the FF/FB initial weights, ``apical_drive_threshold``,
``apical_gain_strength`` and ``baseline_drive_sigma`` -- but replaces the 17
hand-tuned, partly ad-hoc transition templates with a small set keyed entirely by
**tuning width**:

  * ``untuned`` (width 0): no FF tuning. With no feedback it stays near the origin
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
always **generalized** (equal across all three context channels, as in the
biology -- no hand-painted per-channel asymmetry), and narrow/broad tuning is the
*only* thing that sets the FF-plasticity scale and the feedback drive-vs-gain
regime. Surround weights (``w_lat``, ``W_pv``, ``w_pv_lat``) are coupled per
template; only ``w_lat`` is plastic.

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


DEFAULT_OUTPUT_DIR = base.PACKAGE_DIR / "outputs_pruned_mini"
GENERIC_BASE_CONFIG = "FF_FB_broad"
N_FEATURES = 3

# Only PyC synapses learn; PV feedforward learning is off (lr_pv = 0). Lateral
# (surround) learning is doubled relative to the base, and feedback learning is
# boosted: feedback strengthening is the engine of every learning-induced change
# (narrow +NO gain amplification and broad +O drive), so the tiny base lr_fb
# leaves the population stuck near naive (everything reads as small dNO/dO).
SHARED_LEARNING_RATES = {"lr_ff": 0.015, "lr_fb": 0.0012, "lr_lat": 0.005, "lr_pv": 0.0}

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
        # Capped gain keeps the strong naive NO responders within ~0-2 SD (they then
        # strengthen toward 3 via plasticity / amplification) without compressing the
        # -NO adaptation shift (which a higher baseline would).
        gain=4.0, gain_clip=(3.0, 5.0),
        drive=0.16, drive_clip=(0.10, 0.45),
        baseline=0.21, baseline_clip=(0.15, 0.34),
    ),
    "narrow": dict(
        # Weak (not zero) FF plasticity: a narrow cell tuned to a *familiar* image
        # partially adapts its FF during training and is pulled back toward
        # small/-NO, while one tuned to the *novel* image keeps its FF intact and is
        # amplified to +NO -- so the familiar<novel +NO asymmetry emerges from the
        # protocol rather than being hand-assigned.
        ff_plasticity_scale=0.55,
        gain=8.0, gain_clip=(5.0, 11.0),
        # Shift + steepen the gain sigmoid so the small feedback growth during
        # training moves the gain enough to turn the non-adapted FF drive into a
        # clear +NO shift at expert (suppressed naive -> amplified expert).
        gain_threshold=0.03, gain_k=8.0,
        drive=1.25, drive_clip=(1.05, None),
        # Low baseline -> low adaptation current, so the occluded basal stays near 0
        # and the rising gain barely drags the occluded response negative: the +NO
        # cells stay centred at O~0 (their NO shift is FF-driven, baseline-independent).
        baseline=0.08, baseline_clip=(0.055, 0.12),
    ),
}

# Feedforward initial-weight strength, picked per template (not per width): a weak/
# already-adapted broad cell ("silent") and a strongly tuned broad cell share the
# broad width class but differ only in FF drive. (tuned, silent center; rel_noise;
# noise_floor; lo; hi). `broad_weak` is the weakly-NO-tuned broad cell that becomes
# an O responder (naive NO ~0.25-0.75); `broad` is the strongly NO-tuned -NO cell.
FF_STRENGTHS: dict[str, dict[str, Any]] = {
    "silent": dict(tuned=0.010, silent=0.010, rel=0.60, floor=0.012, lo=0.0, hi=0.05),
    "broad_weak": dict(tuned=0.080, silent=0.060, rel=0.40, floor=0.012, lo=0.0, hi=0.17),
    "broad": dict(tuned=0.115, silent=0.090, rel=0.40, floor=0.014, lo=0.0, hi=0.22),
    # Moderate FF with TIGHT noise: stays under ~2 SD at naive, but its intact novel
    # FF reaches NO~1 at expert -> the mixed +NO+O novel cloud.
    "broad_mid": dict(tuned=0.155, silent=0.120, rel=0.26, floor=0.014, lo=0.0, hi=0.24),
    "narrow_weak": dict(tuned=0.090, silent=0.006, rel=0.50, floor=0.010, lo=0.0, hi=0.22),
    "narrow": dict(tuned=0.120, silent=0.006, rel=0.48, floor=0.010, lo=0.0, hi=0.24),
    "narrow_strong": dict(tuned=0.140, silent=0.008, rel=0.46, floor=0.012, lo=0.0, hi=0.28),
}

# Generalized feedback levels (equal across all three context channels):
# (receives_context, center, rel_noise, noise_floor, lo, hi).
FB_LEVELS: dict[str, dict[str, Any]] = {
    "none": dict(receives=False, center=0.004, rel=0.60, floor=0.005, lo=0.0, hi=0.030),
    "weak": dict(receives=True, center=0.040, rel=0.42, floor=0.010, lo=0.006, hi=0.120),
    "weak_plus": dict(receives=True, center=0.070, rel=0.40, floor=0.012, lo=0.012, hi=0.150),
    "strong": dict(receives=True, center=0.250, rel=0.28, floor=0.024, lo=0.140, hi=0.480),
    # Near-saturated feedback: a naive O responder whose feedback is already high and
    # grows little during training -> its O barely shifts -> it sits in the naive
    # O-responder cloud as a small-delta / slight -O cell rather than rising to +O.
    "strong_sat": dict(receives=True, center=0.400, rel=0.26, floor=0.024, lo=0.260, hi=0.560),
}

# Which feedforward channels each cell prefers (drawn per cell):
#   "all"       -> tuned to every image (broad)
#   "permuted1" -> one random preferred image (narrow; emergent fam/novel asymmetry)
#   "novel"     -> the novel image, index 2 (the narrow_novel special case)
NOVEL_INDEX = 2

# The pruned population mixture. Each template = (width, FF strength, FB level,
# tuning) plus its coupled surround (w_lat, w_pv_lat, W_pv on tuned/silent channels),
# optional baseline override, and optional per-template scalar overrides (gain etc.).
# `weight` is its share of the population.
TEMPLATES: dict[str, dict[str, Any]] = {
    # Weak / already-adapted broad cell, no feedback -> unresponsive (small dNO/dO).
    "weak_broad": dict(
        width="broad", ff="silent", fb="none", tuning="all", weight=0.10,
        lat=0.04, pvlat=0.08, pv_tuned=0.18, pv_silent=0.18,
    ),
    # Weak broad + strong generalized feedback (FB->FB-like: little FF, strong FB)
    # -> naive feedback-driven O responder (NO~0, O>0) for BOTH familiar and novel:
    # the distinct naive O-responder cloud.
    "weak_broad_FB": dict(
        width="broad", ff="silent", fb="strong_sat", tuning="all", weight=0.14,
        lat=0.18, pvlat=0.18, pv_tuned=0.51, pv_silent=0.51,
        baseline=0.35, baseline_clip=(0.28, 0.46),
    ),
    # Strong broad FF, no feedback -> familiar FF adapts away -> pure -NO. Kept small:
    # on novel these are static extreme-NO cells (no FB -> no movement), which the
    # data does not show, so most broad cells instead carry feedback (below).
    "broad_FFonly": dict(
        width="broad", ff="broad", fb="none", tuning="all", weight=0.13,
        lat=0.11, pvlat=0.12, pv_tuned=0.20, pv_silent=0.20,
    ),
    # Broad FF + feedback: the key naive-NO -> expert-O transition. Naive weak O&NO
    # responder (NO~0.5-1.2, O~0.3-0.8). FAMILIAR: FF adapts + growing FF->PV surround
    # cancels the drive on the full image -> moves UP and LEFT into the expert O cloud
    # at NO~0 (a range of O levels). NOVEL: FF stays intact + FB grows -> moves UP and
    # RIGHT into the mixed +NO+O cloud. Weak/strong FB span weak->strong O responders.
    "broad_FB_weak": dict(
        width="broad", ff="broad", fb="weak", tuning="all", weight=0.15,
        lat=0.13, pvlat=0.18, pv_tuned=0.50, pv_silent=0.50,
        baseline=0.30, baseline_clip=(0.22, 0.42),
    ),
    "broad_FB_strong": dict(
        width="broad", ff="broad_mid", fb="strong", tuning="all", weight=0.15,
        lat=0.13, pvlat=0.18, pv_tuned=0.50, pv_silent=0.50,
        baseline=0.38, baseline_clip=(0.30, 0.50),
    ),
    # Narrow, one random preferred image -> +NO via gain; the emergent familiar/novel
    # +NO asymmetry. Weak and strong FF variants give a range of +NO magnitudes.
    "narrow_weak": dict(
        width="narrow", ff="narrow_weak", fb="weak", tuning="permuted1", weight=0.13,
        lat=0.03, pvlat=0.03, pv_tuned=0.045, pv_silent=0.03,
    ),
    "narrow_strong": dict(
        width="narrow", ff="narrow_strong", fb="weak", tuning="permuted1", weight=0.13,
        lat=0.03, pvlat=0.03, pv_tuned=0.045, pv_silent=0.03,
    ),
    # Special case: a narrowly NOVEL-tuned cell -- suppressed at naive (gain < 1 so it
    # starts ~0, "unresponsive"), then strong gain + feedback growth make it a clear
    # +NO responder at expert. The dedicated unresponsive->novel +NO population
    # (cf. mini's FF_FB_narrow_novel).
    "narrow_novel": dict(
        width="narrow", ff="narrow", fb="weak_plus", tuning="novel", weight=0.16,
        gain=8.5, lat=0.03, pvlat=0.03, pv_tuned=0.045, pv_silent=0.03,
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


def _vector(tuned_indices: tuple[int, ...], tuned: float, silent: float) -> list[float]:
    tuned_set = set(tuned_indices)
    return [float(tuned if idx in tuned_set else silent) for idx in range(N_FEATURES)]


def _build_transition_specs() -> dict[str, dict[str, Any]]:
    """Translate the TEMPLATES table into base ``transition`` specs (only their
    ``weight`` is read downstream; the real config is built in the perturb hook)."""
    specs: dict[str, dict[str, Any]] = {}
    for name, template in TEMPLATES.items():
        specs[name] = base.transition(template["weight"])
    return specs


def _build_config(name: str, tuned_indices: tuple[int, ...]) -> dict[str, Any]:
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
            [fb["center"]] * N_FEATURES, fb["rel"], fb["floor"],
            [fb["lo"]] * N_FEATURES, [fb["hi"]] * N_FEATURES,
        ),
        "w_lat_init": base.weight_init([template["lat"]]),
        "w_pv_lat_init": base.weight_init([template["pvlat"]]),
        "W_pv_init": base.weight_init(pv_vec),
    }
    fixed = {
        "receives_context": (fb["receives"],) * N_FEATURES,
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
        "apical_gain_strength": width_class["gain_clip"],
        "apical_drive_threshold": width_class["drive_clip"],
        "baseline_drive_sigma": template.get("baseline_clip", width_class["baseline_clip"]),
    }
    return {"init": config, "fix": fixed, "clip": clip, "width": len(tuned_indices)}


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
        spec = _build_config(transition, _draw_tuned_indices(TEMPLATES[transition]["tuning"], rng))
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
        )
        return config

    return _perturb_config


def _configure_pruned_variant() -> None:
    transitions = _build_transition_specs()
    generic_config = copy.deepcopy(base.minimal_configs3[GENERIC_BASE_CONFIG])
    base.TRANSITIONS = transitions
    base.minimal_configs3 = {name: copy.deepcopy(generic_config) for name in transitions}
    base.SHARED_LEARNING_RATES = dict(SHARED_LEARNING_RATES)
    base._perturb_config = _perturb_config_factory()


def _write_metadata(args: argparse.Namespace) -> None:
    path = args.output_dir / "metadata.json"
    metadata = json.loads(path.read_text()) if path.exists() else {}
    metadata["pruned_mini_variant"] = {
        "description": "Mini model (only PyC plasticity) with a small width-keyed principled template set, generalized feedback, and emergent FF tuning via random per-cell permutation.",
        "disabled_plasticity": ["W_pv", "w_pv_lat"],
        "enabled_plasticity": ["w_ff", "w_fb", "w_lat"],
        "shared_learning_rates": SHARED_LEARNING_RATES,
        "width_classes": WIDTH_CLASSES,
        "templates": {name: {k: v for k, v in t.items()} for name, t in TEMPLATES.items()},
    }
    path.write_text(json.dumps(metadata, indent=2, default=repr))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pruned principled mini model-scatter variant.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-samples", type=int, default=250)
    parser.add_argument("--n-steps-per-phase", type=int, default=200)
    parser.add_argument("--test-trials", type=int, default=3)
    parser.add_argument("--training-trials", type=int, default=5)
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
    parser.add_argument("--skip-center-panels", action="store_true", default=True)
    parser.add_argument("--plot-by-transition", action="store_true")
    parser.add_argument("--export-panels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.canonical_only:
        raise ValueError("--canonical-only is not supported by the pruned mini sampler (permutation sampling has no canonical orientation).")
    _configure_pruned_variant()
    base.run_model_scatter(args)
    _write_metadata(args)


if __name__ == "__main__":
    main()
