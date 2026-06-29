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

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import model_scatter as base
from .neuron_utils import ThresholdReLU


DEFAULT_OUTPUT_DIR = base.PACKAGE_DIR / "outputs_pruned_mini"
N_FEATURES = 3

# Only PyC synapses learn; PV feedforward learning is off (lr_pv = 0). Feedback
# strengthening and lateral surround learning stay shared across templates. These
# rates are calibrated at 200 steps/phase; longer phase durations scale them down
# so the accumulated plasticity stays close to the clearer 200-step scatter.
LEARNING_RATE_REFERENCE_STEPS = 200
SHARED_LEARNING_RATES = {"lr_ff": 0.0135, "lr_fb": 0.00050, "lr_lat": 0.0240, "lr_pv": 0.0}
SCALAR_NOISE_KEYS = ("apical_gain_strength", "apical_drive_threshold")
SOMA_ACTIVATION_THRESHOLD = 0.08
APICAL_DRIVE_SUBTRACTIVE = True
PV_NOISE_SIGMA = 0.075
DIVISIVE_GAIN = 10.0
BASELINE_STD_SCALE = 0.27
SCALAR_NOISE = {
    "apical_gain_strength": ("log", 0.18, 0.1, 50.0, 0.0),
    "apical_drive_threshold": ("add", 0.12, 0.0, 3.0, 0.05),
}
UNIFORM_FF_NOISE = dict(rel=0.50, floor=0.016, lo=0.0, hi=0.40)
UNIFORM_FB_NOISE = dict(rel=0.50, floor=0.016, lo=0.0, hi=0.40)
UNIFORM_GAIN_CLIP = (1.5, 8.0)
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
# gives fixed centers and post-noise clips for the two jittered scalar parameters
# (apical_gain_strength, apical_drive_threshold). Baseline sigma is a fixed center.
WIDTH_CLASSES: dict[str, dict[str, Any]] = {
    "broad": dict(
        ff_plasticity_scale=8.0,
        # Broad cells need enough gain that intact novel FF drive can be amplified
        # alongside strengthened generalized FB; otherwise novel expert NO gets
        # suppressed even for mixed FF+FB responders.
        gain=3.8, gain_clip=UNIFORM_GAIN_CLIP,
        drive=0.16, drive_clip=UNIFORM_DRIVE_CLIP,
        baseline=0.16,
    ),
    "narrow": dict(
        # Weak (not zero) FF plasticity: a narrow cell tuned to a *familiar* image
        # partially adapts its FF during training and is pulled back toward
        # small/-NO, while one tuned to the *novel* image keeps its FF intact and is
        # amplified to +NO -- so the familiar<novel +NO asymmetry emerges from the
        # protocol rather than being hand-assigned.
        ff_plasticity_scale=0.8,
        gain=5.6, gain_clip=UNIFORM_GAIN_CLIP,
        # Shift + steepen the gain sigmoid so the small feedback growth during
        # training moves the gain enough to turn the non-adapted FF drive into a
        # clear +NO shift at expert (suppressed naive -> amplified expert).
        gain_threshold=0.03, gain_k=7.0,
        drive=1.25, drive_clip=UNIFORM_DRIVE_CLIP,
        # Low baseline -> low adaptation current, so the occluded basal stays near 0
        # and the rising gain barely drags the occluded response negative: the +NO
        # cells stay centred at O~0 (their NO shift is FF-driven, baseline-independent).
        baseline=0.07,
    ),
}

# Shared feedforward initial-weight levels. These are independent of tuning width:
# the tuned value is assigned to preferred stimulus channels and the silent value
# to non-preferred channels. Broad cells simply tune all channels, while narrow
# cells tune the drawn preferred channel only.
FF_STRENGTHS: dict[str, dict[str, Any]] = {
    "silent": dict(tuned=0.010, silent=0.010, **UNIFORM_FF_NOISE),
    "very_weak": dict(tuned=0.065, silent=0.005, **UNIFORM_FF_NOISE),
    "diag_weak": dict(tuned=0.042, silent=0.004, **UNIFORM_FF_NOISE),
    "weak": dict(tuned=0.090, silent=0.006, **UNIFORM_FF_NOISE),
    "mid": dict(tuned=0.120, silent=0.006, **UNIFORM_FF_NOISE),
    "strong": dict(tuned=0.155, silent=0.008, **UNIFORM_FF_NOISE),
    # Narrow-only FF levels keep the +NO mechanism graded while preventing rare
    # high FF draws from producing >3 SD novel NO outliers.
    "narrow_weak": dict(tuned=0.080, silent=0.006, **UNIFORM_FF_NOISE),
    "narrow_mid": dict(tuned=0.105, silent=0.006, **UNIFORM_FF_NOISE),
}

# Generalized feedback levels (equal across all three context channels):
# (receives_context, center, rel_noise, noise_floor, lo, hi).
FB_LEVELS: dict[str, dict[str, Any]] = {
    "none": dict(receives=False, center=0.004, **UNIFORM_FB_NOISE),
    "weak": dict(receives=True, center=0.040, **UNIFORM_FB_NOISE),
    "mid": dict(receives=True, center=0.075, **UNIFORM_FB_NOISE),
    "strong": dict(receives=True, center=0.110, **UNIFORM_FB_NOISE),
    # High initial FB with limited plastic headroom: creates naive O responders
    # without making their post-training O shift dominate the novel scatter. Tight
    # rel-noise + raised lo keep the O spread (hence the NO spread = O - gain*Dy_lat)
    # narrow so the expert O cloud sits squarely at NO=0 rather than smearing into a
    # negative tail / the diagonal.
    "strong_sat": dict(receives=True, center=0.300, **UNIFORM_FB_NOISE),
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
# tuning, context-receive mode), optional baseline override, and optional
# per-template scalar overrides (gain etc.). `weight` is its share of the
# population. Coupled surround parameters live in SURROUND_SETTINGS below.
TEMPLATES: dict[str, dict[str, Any]] = {
    # Silent / already-adapted broad cell, no feedback -> unresponsive (small dNO/dO).
    "silent_broad_FFonly": dict(
        width="broad", ff="silent", fb="none", tuning="all", context="none", weight=0.035,
    ),
    # Silent broad + weak generalized FB: starts close to the unresponsive cloud,
    # then shared FB/LAT plasticity moves it vertically into the O cloud while the
    # learned full-image surround keeps expert NO near 0.
    "silent_broad_FB_weak": dict(
        width="broad", ff="silent", fb="weak", tuning="all", context="all", weight=0.010,
        drive=0.035,
        gain=3.0,
        baseline=0.18,
    ),
    # FB->FB-like: little FF and moderate generalized FB. This fills weaker O
    # responders; the stronger naive O plume comes from silent_broad_FB_strong.
    "silent_broad_FB_mid": dict(
        width="broad", ff="silent", fb="mid", tuning="all", context="all", weight=0.005,
        drive=0.035,
        gain=3.0,
        baseline=0.18,
    ),
    # Partial-context silent broad cells: same generalized FB on received channels,
    # but only one or two context inputs exist. These fill the weak/intermediate O
    # band without forcing every stimulus from the cell into the strong O cloud.
    "silent_broad_FB_partial2": dict(
        width="broad", ff="silent", fb="mid", tuning="all", context="random2", weight=0.045,
        drive=0.035,
        gain=3.0,
        baseline=0.18,
    ),
    # Mid broad FF, no feedback -> familiar FF adapts away -> pure -NO. Kept small:
    # on novel these are static NO cells (no FB -> no movement), which the data does
    # not show as the dominant broad population.
    "mid_broad_FFonly": dict(
        width="broad", ff="mid", fb="none", tuning="all", context="none", weight=0.155,
    ),
    # Weak broad FF + partial generalized FB: fills the missing weak O&NO bridge.
    # These start below the mid-broad rows, then move up-and-left into the lower
    # familiar +O band instead of jumping straight into a detached high-O plume.
    # The designated NOVEL +NO+O transition cell. Uses a HIGH initial w_lat (little
    # growth headroom) so on FAMILIAR the surround cancels the feedback drive on the
    # full image -> familiar O sits at NO=0 (not diagonal), while the surround barely
    # GROWS during training so it does NOT drag the novel response down. Real (weak)
    # intact novel FF + HIGH gain then lets novel NO rise NOTICEABLY at expert; the
    # kept apical drive threshold lets generalized FB lift O by a similar amount ->
    # the mixed (+NO,+O) mover. Familiar FF adapts via the shared broad
    # ff_plasticity_scale (active-channel gated; novel channel untouched).
    "very_weak_broad_FB_partial2": dict(
        width="broad", ff="weak", fb="mid", tuning="all", context="random2", weight=0.040,
        drive=0.035,
        gain=5.0,
        baseline=0.15,
    ),
    # Small weak-both bridge: modest broad FF plus generalized FB. It fills the
    # familiar naive cloud around (NO~0.5, O~0.5), then uses the mover surround
    # timing in the divisive variant to travel up-and-left into the expert O axis.
    "weak_broad_FB_mixed_bridge": dict(
        width="broad", ff="very_weak", fb="mid", tuning="all", context="all", weight=0.020,
        drive=0.005,
        gain=3.3,
        baseline=0.16,
    ),

    # Broad FF + generalized FB: the key medium NO/O -> expert O class. Familiar
    # FF adapts down while generalized FB grows; novel keeps its FF and should move
    # +NO and/or +O rather than being suppressed.
    # FF-bearing broad FB cells: primarily the FAMILIAR naive-NO/O -> expert-O movers.
    # On familiar the FF adapts and the FF->PV surround (W_pv -> Delta y_lat) must be
    # just strong enough to cancel the feedback drive on the FULL image so the expert
    # O response sits directly above NO=0 (NOT on the zO=zNO diagonal, NOT below 0).
    # W_pv kept moderate (~0.40) -> novel NO stays roughly flat (these do NOT feed the
    # novel -NO population, which should come only from mid_broad_FFonly).
    "mid_broad_FB_weak": dict(
        width="broad", ff="mid", fb="mid", tuning="all", context="all", weight=0.050,
        drive=0.085,
        gain=4.4,
        baseline=0.18,
    ),
    "mid_broad_FB_partial2": dict(
        width="broad", ff="mid", fb="mid", tuning="all", context="random2", weight=0.040,
        drive=0.080,
        gain=4.2,
        baseline=0.18,
    ),
    "strong_broad_FB_strong": dict(
        width="broad", ff="strong", fb="strong", tuning="all", context="all", weight=0.025,
        drive=0.105,
        gain=4.8,
        baseline=0.25,
    ),
    # Narrow, one random preferred image -> +NO via gain; the emergent familiar/novel
    # +NO asymmetry. Weak and mid FF variants give a range of +NO magnitudes.
    "narrow_weak": dict(
        width="narrow", ff="narrow_weak", fb="weak", tuning="permuted1", context="all", weight=0.13,
    ),
    "narrow_mid": dict(
        width="narrow", ff="narrow_mid", fb="weak", tuning="permuted1", context="all", weight=0.120,
    ),
    # Special case: a narrowly NOVEL-tuned cell. A weak novel FF seed plus strong
    # generalized FB keeps naive responses near silent while giving FB growth enough
    # leverage to produce a qualitative silent -> novel-NO transition.
    "narrow_novel": dict(
        width="narrow", ff="narrow_weak", fb="weak", tuning="novel", context="all", weight=0.225,
        gain=6.8, gain_threshold=0.045,
    ),
    # Novel diagonal bridge: novel-tuned weak FF plus generalized FB. These cells
    # start near the silent/weak novel NO&O cloud and, after familiar training grows
    # generalized FB, move into the modest novel expert NO&O interior with a
    # +NO-dominant diagonal shift.
    "novel_weak_FB_diagonal": dict(
        width="broad", ff="diag_weak", fb="weak", tuning="novel", context="novel", weight=0.065,
        drive=0.035,
        gain=3.0,
        gain_threshold=0.035, gain_k=7.0,
        baseline=0.14,
    ),
    # Weak broad FF-only bridge: same -NO mechanism as mid_broad_FFonly, but with
    # weaker initial FF so the -NO population also fills the 0.3-0.5 naive NO band.
    "weak_broad_FFonly": dict(
        width="broad", ff="weak", fb="none", tuning="all", context="none", weight=0.105,
        baseline=0.14,
    ),
    # Shared naive O responders. High initial generalized FB puts these cells in
    # the O cloud before training; limited FB headroom and strong fixed PV keep the
    # subtype from turning novel +O into the dominant transition.
    "silent_broad_FB_strong": dict(
        width="broad", ff="silent", fb="strong_sat", tuning="all", context="all", weight=0.055,
        drive=0.180,
        gain=2.2,
        baseline=0.26,
    ),
}

SURROUND_SETTINGS: dict[str, dict[str, float]] = {
    "silent_broad_FFonly": dict(lat=0.04, pvlat=0.08, pv_tuned=0.18, pv_silent=0.18),
    "silent_broad_FB_strong": dict(lat=0.85, pvlat=0.05, pv_tuned=0.60, pv_silent=0.60),
    "silent_broad_FB_weak": dict(lat=0.85, pvlat=0.05, pv_tuned=0.60, pv_silent=0.60),
    "silent_broad_FB_mid": dict(lat=0.85, pvlat=0.05, pv_tuned=0.60, pv_silent=0.60),
    "silent_broad_FB_partial2": dict(lat=0.85, pvlat=0.05, pv_tuned=0.60, pv_silent=0.60),
    "mid_broad_FFonly": dict(lat=0.050, pvlat=0.1, pv_tuned=0.28, pv_silent=0.28),
    "mid_broad_FB_weak": dict(lat=0.14, pvlat=0.05, pv_tuned=0.55, pv_silent=0.55),
    "mid_broad_FB_partial2": dict(lat=0.14, pvlat=0.05, pv_tuned=0.55, pv_silent=0.55),
    "strong_broad_FB_strong": dict(lat=0.14, pvlat=0.05, pv_tuned=0.55, pv_silent=0.55),
    "very_weak_broad_FB_partial2": dict(lat=0.14, pvlat=0.05, pv_tuned=0.55, pv_silent=0.55),
    "weak_broad_FB_mixed_bridge": dict(lat=0.20, pvlat=0.05, pv_tuned=0.62, pv_silent=0.62),
    "narrow_weak": dict(lat=0.03, pvlat=0.03, pv_tuned=0.045, pv_silent=0.03),
    "narrow_mid": dict(lat=0.03, pvlat=0.03, pv_tuned=0.045, pv_silent=0.03),
    "narrow_novel": dict(lat=0.03, pvlat=0.03, pv_tuned=0.045, pv_silent=0.03),
    "novel_weak_FB_diagonal": dict(lat=0.08, pvlat=0.025, pv_tuned=0.20, pv_silent=0.06),
    "weak_broad_FFonly": dict(lat=0.035, pvlat=0.10, pv_tuned=0.24, pv_silent=0.24),
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
    surround = SURROUND_SETTINGS[name]

    pv_vec = _vector(tuned_indices, surround["pv_tuned"], surround["pv_silent"])
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
    # Width-class scalars, with optional per-template overrides.
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
    # Center-panel traces use the noise-free template centers.
    base._center_config = _center_config
    # base._canonical_config = _center_config


def write_metadata(args) -> None:
    path = args.output_dir / "metadata.json"
    metadata = json.loads(path.read_text()) if path.exists() else {}
    metadata["pruned_mini_variant"] = {
        "description": "Mini model (only PyC plasticity) with a small width-keyed principled template set, generalized feedback, and emergent FF tuning via random per-cell permutation.",
        "disabled_plasticity": ["W_pv", "w_pv_lat"],
        "enabled_plasticity": ["w_ff", "w_fb", "w_lat"],
        "model": "minimal_divisive.CCNeuron",
        "divisive_gain": DIVISIVE_GAIN,
        "shared_learning_rates": _effective_learning_rates(args.n_steps_per_phase),
        "base_shared_learning_rates": SHARED_LEARNING_RATES,
        "learning_rate_reference_steps": LEARNING_RATE_REFERENCE_STEPS,
        "soma_activation_threshold": SOMA_ACTIVATION_THRESHOLD,
        "apical_drive_subtractive": APICAL_DRIVE_SUBTRACTIVE,
        "pv_noise_sigma": PV_NOISE_SIGMA,
        "uniform_ff_noise": UNIFORM_FF_NOISE,
        "uniform_fb_noise": UNIFORM_FB_NOISE,
        "uniform_gain_clip": UNIFORM_GAIN_CLIP,
        "uniform_drive_clip": UNIFORM_DRIVE_CLIP,
        "scalar_noise_keys": list(SCALAR_NOISE_KEYS),
        "width_classes": WIDTH_CLASSES,
        "ff_strengths": FF_STRENGTHS,
        "fb_levels": FB_LEVELS,
        "context_modes": CONTEXT_MODES,
        "surround_settings": SURROUND_SETTINGS,
        "templates": {name: {k: v for k, v in t.items()} for name, t in TEMPLATES.items()},
    }
    path.write_text(json.dumps(metadata, indent=2, default=repr))
