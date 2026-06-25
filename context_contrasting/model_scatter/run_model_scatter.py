from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

import context_contrasting.data_analysis.transitions_helpers as th
from context_contrasting.minimal2.config_s import minimal_configs3
from context_contrasting.minimal2.experiment_s import (
    PRIMARY_EXPERIMENT_SERIES,
    STIMULUS_SPECS,
    _build_test_stimuli,
    _build_training_stimuli,
    run_experiment,
    run_experimental_phase,
)
from context_contrasting.minimal2.minimal_s import CCNeuron
from context_contrasting.minimal2.visualize_s import (
    format_transition_label,
    visualize_transition_panel,
    wide_to_long,
)


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs"

STAGES = {"naive": "Naive", "expert": "Expert"}
TRACE_TYPES = {"full": "Full", "occlusion": "Occl"}
IMAGE_INFO = {
    "familiar_1": ("familiar", 1, 1),
    "familiar_2": ("familiar", 2, 2),
    "novel": ("novel", 3, 1),
}
INIT_KEYS = ("w_ff_init", "w_fb_init", "w_lat_init", "w_pv_lat_init", "W_pv_init")
INIT_ALIASES = {
    "ff": "w_ff_init",
    "fb": "w_fb_init",
    "lat": "w_lat_init",
    "pvlat": "w_pv_lat_init",
    "pv": "W_pv_init",
}
FIXED_SCALARS = ("lr_ff", "lr_fb", "lr_lat", "lr_pv", "pyc_decay", "pv_decay")
SHARED_LEARNING_RATES = {
    "lr_ff": 0.015,
    "lr_fb": 0.0005,
    "lr_lat": 0.0015,
    "lr_pv": 0.003,
}

# Feedforward plasticity (anti-Hebbian adaptation) scale is a property of tuning
# width, not of the individual transition: broadly tuned cells adapt their FF
# drive strongly (and become FB-driven), narrowly tuned cells adapt FF only
# weakly (and instead develop enhanced FF responses via increased FB gain). The
# narrow value is deliberately small but never 0, so that the pure-FF response
# (noLAT & noFB ablation) of narrowly tuned cells still adapts a little.
FF_PLASTICITY_BROAD = 8.0
FF_PLASTICITY_NARROW = 0.05
NARROW_TRANSITIONS = frozenset({
    "weak_FF",
    "un_novel_FF",
    "FF_FB_narrow_familiar",
    "FF_FB_narrow_familiar_2",
    "FF_FB_narrow_familiar_novel",
    "FF_FB_narrow_familiar_2_novel",
    "FF_FB_narrow_novel",
})


def _ff_plasticity_scale(transition: str) -> float:
    return FF_PLASTICITY_NARROW if transition in NARROW_TRANSITIONS else FF_PLASTICITY_BROAD

PLOT_STYLE = th.DEFAULT_PLOT_STYLE | {
    "pre_point_alpha": 1.0,
    "target_point_alpha": 1.0,
    "shift_point_alpha": 1.0,
    "individual_vector_width": 0.005,
    "mean_arrow_width": 3.1,
    "mean_arrow_mutation_scale": 16.5,
}


def I(center: list[float], rel: float, floor: float, lo: Any = 0.0, hi: Any = 1.0) -> tuple:
    return (center, rel, floor, lo, hi)


def S(weight: float, *, fix: dict | None = None, clip: dict | None = None, **inits: tuple) -> dict:
    return {
        "weight": weight,
        "init": {INIT_ALIASES[key]: spec for key, spec in inits.items()},
        "fix": fix or {},
        "clip": clip or {},
    }


BASELINE_MIN = {"baseline_drive_sigma": (0.08, None)}
NARROW_GAIN_CLIP = {
    "apical_drive_threshold": (1.05, None),
    "apical_gain_strength": (5.5, 9.0),
    "baseline_drive_sigma": (0.12, 0.28),
}
GLOBAL_SCALAR_CLIP = {
    "baseline_drive_sigma": (0.18, 0.55),
    "pv_noise_sigma": (0.04, 0.16),
}

# Higher baseline (spontaneous) activity for the feedback-driven O responders.
# Because the post responses are z-scored to the naive baseline, a larger baseline
# raises the subtracted mean and pulls the absolute O/NO positions down out of the
# extreme (z~3) band into a realistic 0.5-1.5 range -- without changing the
# expert-naive shift (the baseline cancels in the difference), so sector fractions
# are preserved while the clouds move into the right place on the scatter.
O_RESPONDER_BASELINE = {"baseline_drive_sigma": (0.26, 0.42)}

# Per-cell z-score denominator. The pyramidal EMA over-smooths the *measured*
# baseline std (~0.05 for every cell), so a signal-poor response (e.g. an O
# responder's NO ~ 0) gets divided by the tiny floor and its z-score cloud blows
# up far past the real data (familiar O-responder NO std is ~0.26 in the data but
# ~0.50 in the model). Scale the floor with each cell's baseline-drive sigma -- a
# proxy for the true spontaneous variability the EMA flattens. Then for a noisy
# (high-baseline) O responder both its NO noise *and* its denominator scale with
# sigma, so the NO z-noise is bounded; meanwhile a strong NO/-NO responder has low
# baseline sigma -> small denominator -> it keeps its large z-score and stays
# sectored. This positions/tightens the O cloud without touching the other cells.
BASELINE_STD_SCALE = 0.27

# Main sampling knobs: transition proportions and allowed parameter variation.
TRANSITIONS = {
    "weak_FB": S(
        0.035,
        fix={
            "apical_drive_threshold": 0.24,
        },
        clip={"apical_gain_strength": (5.0, 8.5), "baseline_drive_sigma": (0.10, 0.25)},
        ff=I([0.032, 0.032, 0.014], 0.45, 0.010, [0.006, 0.006, 0.0], [0.075, 0.075, 0.040]),
        fb=I([0.050, 0.050, 0.040], 0.42, 0.008, [0.012, 0.012, 0.008], [0.105, 0.105, 0.085]),
        lat=I([0.22], 0.32, 0.030, 0.08, 0.45),
        pvlat=I([0.12], 0.40, 0.020, 0.03, 0.28),
        pv=I([0.26, 0.26, 0.22], 0.32, 0.035, [0.10, 0.10, 0.06], [0.52, 0.52, 0.44]),
    ),
    "weak_FF": S(
        0.045,
        clip={
            "apical_drive_threshold": (1.05, None),
            "apical_gain_strength": (4.5, 8.0),
            "baseline_drive_sigma": (0.085, 0.17),
        },
        ff=I([0.080, 0.006, 0.006], 0.42, 0.008, [0.035, 0.0, 0.0], [0.145, 0.020, 0.020]),
        fb=I([0.045, 0.045, 0.035], 0.40, 0.006, hi=0.100),
        lat=I([0.02], 0.45, 0.008, hi=0.08),
        pvlat=I([0.02], 0.45, 0.008, hi=0.08),
        pv=I([0.025, 0.025, 0.025], 0.45, 0.012, hi=0.10),
    ),
    "un_un": S(
        0.155,
        # A weakly tuned cell that simply does not receive feedback (context off
        # via the canonical config); plasticity stays on, it just stays subthreshold.
        fix={"receives_context": (False, False, False)},
        clip={"apical_gain_strength": (3.0, 8.0), "apical_drive_threshold": (0.15, 0.50), "baseline_drive_sigma": (0.18, 0.40)},
        ff=I([0.01, 0.01, 0.01], 0.65, 0.012, hi=0.05),
        fb=I([0.004, 0.004, 0.004], 0.65, 0.005, hi=0.035),
        lat=I([0.04], 0.60, 0.020, hi=0.15),
        pvlat=I([0.08], 0.50, 0.025, hi=0.20),
        pv=I([0.12, 0.12, 0.12], 0.45, 0.040, hi=0.35),
    ),
    "un_FB": S(
        0.040,
        # Drive threshold must be low enough that the strengthened (and generalized)
        # feedback actually crosses it, so these cells genuinely become FB-driven O
        # responders. Strong-but-not-crushing inhibition keeps NO ~0 and lands O in
        # the 0.5-1.0 contiguous band rising out of the nonresponders.
        fix={"apical_drive_threshold": 0.13},
        clip={"apical_gain_strength": (2.5, 4.5), "baseline_drive_sigma": (0.35, 0.52)},
        ff=I([0.010, 0.010, 0.010], 0.60, 0.008, hi=0.040),
        fb=I([0.075, 0.075, 0.070], 0.35, 0.012, [0.030, 0.030, 0.028], [0.17, 0.17, 0.16]),
        lat=I([0.12], 0.30, 0.025, 0.04, 0.32),
        pvlat=I([0.10], 0.30, 0.025, 0.03, 0.30),
        pv=I([0.20, 0.20, 0.08], 0.26, 0.025, [0.07, 0.07, 0.0], [0.38, 0.38, 0.20]),
    ),
    "un_novel_FF": S(
        0.055,
        clip={"apical_drive_threshold": (1.10, None), "apical_gain_strength": (6.0, 11.0), "baseline_drive_sigma": (0.10, 0.21)},
        ff=I([0.003, 0.003, 0.09], 0.42, 0.005, [0.0, 0.0, 0.03], [0.012, 0.012, 0.20]),
        fb=I([0.012, 0.012, 0.02], 0.45, 0.003, hi=[0.026, 0.026, 0.045]),
        lat=I([0.02], 0.45, 0.010, hi=0.10),
        pvlat=I([0.02], 0.45, 0.010, hi=0.10),
        pv=I([0.03, 0.03, 0.015], 0.45, 0.010, hi=0.10),
    ),
    "FF_un": S(
        0.125,
        clip={"apical_drive_threshold": (0.85, None), "apical_gain_strength": (3.5, 8.0), "baseline_drive_sigma": (0.14, 0.30)},
        ff=I([0.115, 0.115, 0.115], 0.36, 0.020, [0.040, 0.040, 0.040], [0.22, 0.22, 0.22]),
        fb=I([0.001, 0.001, 0.001], 0.45, 0.002, hi=0.020),
        lat=I([0.075], 0.40, 0.018, 0.020, 0.22),
        pvlat=I([0.08], 0.45, 0.025, hi=0.25),
        pv=I([0.025, 0.025, 0.012], 0.35, 0.008, [0.006, 0.006, 0.0], [0.08, 0.08, 0.06]),
    ),
    "FF_FB_broad": S(
        0.080,
        # Strong broadly-tuned naive NO responder whose FF fully adapts away while
        # the (now strengthened) feedback takes over -> a clear expert O responder
        # (NO ~ 0, O in the 1.0-1.5 band) -- the canonical "FF replaced by FB"
        # transition the data shows. Strong FF -> high naive NO; strong FB -> high
        # expert O; the FF->PV surround cancels the full-image feedback at expert
        # so the full (NO) response stays adapted while the occluded (O) rises.
        fix={"apical_drive_threshold": 0.18},
        clip={"apical_gain_strength": (3.2, 5.5), "baseline_drive_sigma": (0.20, 0.34)},
        ff=I([0.22, 0.22, 0.150], 0.24, 0.014, [0.080, 0.080, 0.040], [0.34, 0.34, 0.26]),
        fb=I([0.145, 0.145, 0.025], 0.28, 0.010, [0.055, 0.055, 0.0], [0.30, 0.30, 0.065]),
        lat=I([0.14], 0.28, 0.022, 0.05, 0.40),
        pvlat=I([0.08], 0.35, 0.018, 0.02, 0.26),
        pv=I([0.26, 0.26, 0.10], 0.25, 0.026, [0.10, 0.10, 0.0], [0.50, 0.50, 0.24]),
    ),
    "FF_FB_broad_weak": S(
        0.080,
        # Broadly tuned, FF adapts away, only a moderate feedback O survives ->
        # weak expert O responders (NO~0, O~0.5) that fill the contiguous band
        # between nonresponders and the strong +O cloud. The FF->PV drive
        # surround-suppresses the full image at expert.
        fix={"apical_drive_threshold": 0.12},
        clip={"apical_gain_strength": (3.0, 5.0), "baseline_drive_sigma": (0.30, 0.46)},
        ff=I([0.095, 0.095, 0.075], 0.26, 0.014, [0.035, 0.035, 0.025], [0.20, 0.20, 0.16]),
        fb=I([0.190, 0.190, 0.028], 0.30, 0.012, [0.080, 0.080, 0.0], [0.32, 0.32, 0.065]),
        lat=I([0.13], 0.30, 0.022, 0.05, 0.30),
        pvlat=I([0.18], 0.30, 0.030, 0.06, 0.42),
        pv=I([0.28, 0.28, 0.09], 0.26, 0.026, [0.12, 0.12, 0.0], [0.52, 0.52, 0.22]),
    ),
    "FF_FB_broad_novel": S(
        0.050,
        # Strong novel feedforward (gain-amplified -> strong novel NO) AND strong
        # generalized novel feedback crossing a low drive threshold (-> strong novel
        # O): the model's prediction of novel expert neurons with BOTH strong novel
        # O and NO. The low novel FF->PV surround leaves the full image unsuppressed
        # so the novel NO survives alongside the occluded O.
        fix={"apical_drive_threshold": 0.14},
        clip={"apical_gain_strength": (3.5, 6.0), "baseline_drive_sigma": (0.24, 0.42)},
        ff=I([0.048, 0.048, 0.050], 0.28, 0.010, [0.015, 0.015, 0.022], [0.14, 0.14, 0.11]),
        fb=I([0.118, 0.118, 0.130], 0.24, 0.012, [0.038, 0.038, 0.055], [0.25, 0.25, 0.27]),
        lat=I([0.12], 0.28, 0.024, 0.04, 0.40),
        pvlat=I([0.09], 0.35, 0.018, 0.02, 0.28),
        pv=I([0.22, 0.22, 0.18], 0.25, 0.026, [0.08, 0.08, 0.06], [0.46, 0.46, 0.34]),
    ),
    "FF_FB_narrow_familiar": S(
        0.017,
        clip={**NARROW_GAIN_CLIP, "apical_gain_strength": (4.8, 8.0), "baseline_drive_sigma": (0.085, 0.17)},
        ff=I([0.112, 0.010, 0.010], 0.42, 0.010, [0.060, 0.0, 0.0], [0.205, 0.022, 0.018]),
        fb=I([0.025, 0.025, 0.020], 0.50, 0.004, hi=0.065),
        lat=I([0.035], 0.45, 0.012, hi=0.14),
        pvlat=I([0.03], 0.45, 0.012, hi=0.12),
        pv=I([0.03, 0.03, 0.03], 0.45, 0.012, hi=0.12),
    ),
    "FF_FB_narrow_familiar_2": S(
        0.015,
        clip={**NARROW_GAIN_CLIP, "apical_gain_strength": (4.8, 8.0), "baseline_drive_sigma": (0.085, 0.17)},
        ff=I([0.010, 0.112, 0.010], 0.32, 0.010, [0.0, 0.060, 0.0], [0.022, 0.205, 0.018]),
        fb=I([0.025, 0.025, 0.020], 0.50, 0.004, hi=0.065),
        lat=I([0.035], 0.45, 0.012, hi=0.14),
        pvlat=I([0.03], 0.45, 0.012, hi=0.12),
        pv=I([0.03, 0.03, 0.03], 0.45, 0.012, hi=0.12),
    ),
    "FF_FB_narrow_familiar_novel": S(
        0.024,
        fix={"apical_gain_threshold": 0.03},
        clip={**NARROW_GAIN_CLIP, "apical_gain_strength": (4.8, 8.0), "baseline_drive_sigma": (0.085, 0.17)},
        ff=I([0.112, 0.010, 0.112], 0.32, 0.010, [0.060, 0.0, 0.060], [0.205, 0.022, 0.205]),
        fb=I([0.025, 0.025, 0.020], 0.50, 0.004, hi=0.065),
        lat=I([0.035], 0.45, 0.012, hi=0.16),
        pvlat=I([0.03], 0.45, 0.012, hi=0.12),
        pv=I([0.03, 0.03, 0.03], 0.45, 0.012, hi=0.16),
    ),
    "FF_FB_narrow_familiar_2_novel": S(
        0.022,
        fix={"apical_gain_threshold": 0.03},
        clip={**NARROW_GAIN_CLIP, "apical_gain_strength": (4.8, 8.0), "baseline_drive_sigma": (0.085, 0.17)},
        ff=I([0.010, 0.112, 0.112], 0.32, 0.010, [0.0, 0.060, 0.060], [0.022, 0.205, 0.205]),
        fb=I([0.025, 0.025, 0.020], 0.50, 0.004, hi=0.065),
        lat=I([0.055], 0.45, 0.012, hi=0.16),
        pvlat=I([0.03], 0.45, 0.012, hi=0.12),
        pv=I([0.03, 0.08, 0.005], 0.45, 0.012, hi=0.16),
    ),
    "FF_FB_narrow_novel": S(
        0.165,
        fix={"apical_gain_threshold": 0.03},
        clip={"apical_drive_threshold": (1.2, None), "apical_gain_strength": (5.0, 9.5), "baseline_drive_sigma": (0.085, 0.17)},
        ff=I([0.003, 0.003, 0.080], 0.42, 0.006, [0.0, 0.0, 0.03], [0.014, 0.014, 0.160]),
        fb=I([0.012, 0.012, 0.02], 0.45, 0.003, hi=[0.026, 0.026, 0.045]),
        lat=I([0.03], 0.45, 0.012, hi=0.12),
        pvlat=I([0.03], 0.45, 0.012, hi=0.12),
        pv=I([0.03, 0.03, 0.03], 0.45, 0.012, hi=0.12),
    ),
    "FB_FB": S(
        0.045,
        # Strong, broadly generalizing feedback with little FF -> O responders with
        # small NO. The symmetric (familiar+novel) feedback also drives the occluded
        # novel response, so this is the main source of novel O responders. Surround
        # kept moderate so the O response is not fully suppressed.
        fix={"apical_drive_threshold": 0.13},
        clip={"apical_gain_strength": (4.0, 7.0), "baseline_drive_sigma": (0.24, 0.40)},
        ff=I([0.001, 0.001, 0.001], 0.60, 0.003, hi=0.010),
        fb=I([0.32, 0.32, 0.32], 0.26, 0.025, [0.16, 0.16, 0.16], [0.58, 0.58, 0.58]),
        lat=I([0.14], 0.30, 0.026, 0.05, 0.34),
        pvlat=I([0.12], 0.32, 0.022, 0.04, 0.32),
        pv=I([0.28, 0.28, 0.28], 0.28, 0.034, [0.09, 0.09, 0.09], [0.52, 0.52, 0.52]),
    ),
    "fb_fb_weak": S(
        0.024,
        # Moderate naive feedback-driven O responder. With FF adaptation and a
        # ramping surround it tends to shed full-image (NO) drive faster than the
        # occluded (O) response, so it lands mostly in -NO / -O rather than gaining
        # O. Kept modest so naive O stays in a realistic 0.5-1.5 band.
        fix={"apical_drive_threshold": 0.13},
        clip={"apical_gain_strength": (2.8, 5.2), "baseline_drive_sigma": (0.16, 0.30)},
        ff=I([0.010, 0.010, 0.010], 0.60, 0.008, hi=0.04),
        fb=I([0.165, 0.165, 0.165], 0.28, 0.012, [0.070, 0.070, 0.070], [0.32, 0.32, 0.32]),
        lat=I([0.12], 0.32, 0.025, 0.03, 0.26),
        pvlat=I([0.45], 0.30, 0.040, 0.18, 0.85),
        pv=I([0.55, 0.55, 0.42], 0.30, 0.040, [0.18, 0.18, 0.12], [0.90, 0.90, 0.75]),
    ),
    "O_un": S(
        0.042,
        # Moderate naive occluded (FB-driven) responder with little FF -> a low-NO,
        # moderate-O naive cloud (the elevated baseline keeps O ~1.0 rather than
        # blowing the z-score up to ~3). At expert the surround ramps and the
        # response weakens toward the origin. (A clean -O is not reachable here -
        # the occluded response lacks the FF->PV surround that suppresses the full
        # image, so feedback-driven changes read out as -NO rather than -O.)
        fix={"apical_drive_threshold": 0.13},
        clip={"apical_gain_strength": (4.0, 6.5), **O_RESPONDER_BASELINE},
        ff=I([0.004, 0.004, 0.004], 0.40, 0.004, hi=0.018),
        fb=I([0.32, 0.32, 0.32], 0.22, 0.025, [0.17, 0.17, 0.17], [0.56, 0.56, 0.56]),
        lat=I([0.18], 0.28, 0.026, 0.07, 0.38),
        pvlat=I([0.32], 0.28, 0.040, 0.12, 0.62),
        pv=I([0.38, 0.38, 0.34], 0.24, 0.034, [0.16, 0.16, 0.12], [0.68, 0.68, 0.62]),
    ),
}

# Feedforward plasticity scale is governed solely by tuning width, consistently
# across every transition (broad vs narrow), never per-transition and never 0.
for _transition, _spec in TRANSITIONS.items():
    _spec["fix"]["ff_plasticity_scale"] = _ff_plasticity_scale(_transition)

SCALAR_NOISE = {
    "apical_gain_strength": ("log", 0.18, 0.1, 50.0, 0.0),
    "apical_gain_k": ("log", 0.18, 0.1, 30.0, 0.0),
    "baseline_drive_sigma": ("log", 0.20, 0.0, 1.0, 0.0),
    "pv_noise_sigma": ("log", 0.20, 0.0, 0.5, 0.0),
    "alpha": ("log", 0.12, 0.05, 10.0, 0.0),
    "apical_drive_threshold": ("add", 0.12, 0.0, 3.0, 0.05),
    "apical_gain_threshold": ("add", 0.08, -1.0, 1.0, 0.04),
}


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _clip(value: float, lo: float | None, hi: float | None) -> float:
    if lo is not None:
        value = max(value, lo)
    if hi is not None:
        value = min(value, hi)
    return float(value)


def _bound(bound: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    if bound is None:
        return None
    arr = np.asarray(bound if isinstance(bound, (list, tuple, np.ndarray)) else [bound], dtype=float)
    return np.full(shape, float(arr.item())) if arr.size == 1 else np.broadcast_to(arr, shape).astype(float)


def _clip_array(values: np.ndarray, lo: Any, hi: Any) -> np.ndarray:
    lo_arr, hi_arr = _bound(lo, values.shape), _bound(hi, values.shape)
    if lo_arr is not None:
        values = np.maximum(values, lo_arr)
    if hi_arr is not None:
        values = np.minimum(values, hi_arr)
    return values


def _set_init(config: dict[str, Any], key: str, values: np.ndarray, sigma: Any = 0.0) -> None:
    config[key] = {"mu": [float(value) for value in values.reshape(-1)], "sigma": sigma}


def _draw_init(init_spec: tuple, rng: np.random.Generator) -> np.ndarray:
    center, rel, floor, lo, hi = init_spec
    center = np.asarray(center, dtype=float)
    scale = np.maximum(np.abs(center) * rel, floor)
    return _clip_array(center + rng.normal(0.0, scale, size=center.shape), lo, hi)


def _draw_scalar(value: float, spec: tuple, rng: np.random.Generator, multiplier: float) -> float:
    mode, scale, lo, hi, floor = spec
    if mode == "log" and value > 0.0:
        sampled = value * float(np.exp(rng.normal(0.0, scale * multiplier)))
    else:
        sampled = value + float(rng.normal(0.0, max(abs(value) * scale, floor) * multiplier))
    return _clip(sampled, lo, hi)


def _apply_shared_learning_rates(config: dict[str, Any]) -> None:
    config.update(SHARED_LEARNING_RATES)


def _perturb_config(
    transition: str,
    base_config: dict[str, Any],
    *,
    sample_idx: int,
    global_idx: int,
    seed: int,
    rng: np.random.Generator,
    initial_condition_mode: str,
    weight_noise_rel: float,
    weight_noise_floor: float,
    scalar_noise_multiplier: float,
    keep_init_sigma: bool,
    initial_weights_only: bool,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    spec = TRANSITIONS[transition]

    if initial_condition_mode == "canonical-neighborhood":
        for key in INIT_KEYS:
            if key not in config:
                continue
            center = np.asarray(config[key].get("mu", []), dtype=float)
            scale = np.maximum(np.abs(center) * weight_noise_rel, weight_noise_floor)
            sigma = config[key].get("sigma", 0.0) if keep_init_sigma else 0.0
            _set_init(config, key, np.clip(center + rng.normal(0.0, scale, size=center.shape), 0.0, None), sigma)
        for key, init_spec in spec["init"].items():
            if key in config:
                values = np.asarray(config[key]["mu"], dtype=float)
                _set_init(config, key, _clip_array(values, init_spec[3], init_spec[4]), config[key].get("sigma", 0.0))
    else:
        for key, init_spec in spec["init"].items():
            _set_init(config, key, _draw_init(init_spec, rng))

    if not initial_weights_only:
        for key, scalar_spec in SCALAR_NOISE.items():
            if key in config and _is_num(config[key]):
                config[key] = _draw_scalar(float(config[key]), scalar_spec, rng, scalar_noise_multiplier)

    config.update(spec["fix"])
    for key, (lo, hi) in (GLOBAL_SCALAR_CLIP | spec["clip"]).items():
        if key in config and _is_num(config[key]):
            config[key] = _clip(float(config[key]), lo, hi)
    _apply_shared_learning_rates(config)

    config.update(
        seed=int(seed),
        _canonical_transition=transition,
        _sample_idx=int(sample_idx),
        _sample_global_idx=int(global_idx),
    )
    return config


def _center_config(transition: str) -> dict[str, Any]:
    """Noise-free config at exactly the centers the sampler draws around.

    Same fix/clip/shared-learning-rate pipeline as `_perturb_config`, but the
    initial weights are the bare `TRANSITIONS` centers (clipped to their bounds)
    and no scalar perturbation is applied. Used for the center-panel sanity check.
    """
    config = copy.deepcopy(minimal_configs3[transition])
    spec = TRANSITIONS[transition]
    for key, init_spec in spec["init"].items():
        center = np.asarray(init_spec[0], dtype=float)
        _set_init(config, key, _clip_array(center, init_spec[3], init_spec[4]))
    config.update(spec["fix"])
    for key, (lo, hi) in (GLOBAL_SCALAR_CLIP | spec["clip"]).items():
        if key in config and _is_num(config[key]):
            config[key] = _clip(float(config[key]), lo, hi)
    _apply_shared_learning_rates(config)
    config.update(_canonical_transition=transition, _sample_idx=0, _sample_global_idx=0)
    return config


def _canonical_config(transition: str) -> dict[str, Any]:
    """The raw config_s canonical example, processed like the sampled cells (the
    per-transition `fix`/`clip`/shared-learning-rate pipeline) but KEEPING the
    config_s initial weights -- so the canonical examples are comparable to (and
    can be highlighted on) the sampler scatter, and can be contrasted with the
    drifted `_center_config` centers."""
    config = copy.deepcopy(minimal_configs3[transition])
    spec = TRANSITIONS[transition]
    config.update(spec["fix"])
    for key, (lo, hi) in (GLOBAL_SCALAR_CLIP | spec["clip"]).items():
        if key in config and _is_num(config[key]):
            config[key] = _clip(float(config[key]), lo, hi)
    _apply_shared_learning_rates(config)
    config.update(_canonical_transition=transition, _sample_idx=0, _sample_global_idx=0)
    return config


def _draw_transition_names(
    transition_order: list[str],
    *,
    n_samples: int,
    transition_sampling: str,
    rng: np.random.Generator,
) -> list[str]:
    if transition_sampling == "equal":
        repeats = int(np.ceil(n_samples / len(transition_order)))
        return [name for name in transition_order for _ in range(repeats)][:n_samples]
    if transition_sampling != "data-like":
        raise ValueError("transition_sampling must be 'data-like' or 'equal'.")
    weights = np.asarray([TRANSITIONS[name]["weight"] for name in transition_order], dtype=float)
    return rng.choice(
        transition_order,
        size=n_samples,
        p=weights / weights.sum(),
    ).tolist()


def _sample_configs(args: argparse.Namespace, transition_order: list[str]) -> list[dict[str, Any]]:
    if args.canonical_only:
        samples = []
        for global_idx, (name, config) in enumerate(minimal_configs3.items(), start=1):
            sample = copy.deepcopy(config)
            _apply_shared_learning_rates(sample)
            sample.update(_canonical_transition=name, _sample_idx=1, _sample_global_idx=global_idx)
            samples.append(sample)
        return samples

    draws = _draw_transition_names(
        transition_order,
        n_samples=args.n_samples,
        transition_sampling=args.transition_sampling,
        rng=np.random.default_rng(args.seed),
    )
    child_rngs = np.random.SeedSequence(args.seed).spawn(len(draws))
    seen = dict.fromkeys(transition_order, 0)
    samples = []
    for global_idx, transition in enumerate(draws, start=1):
        seen[transition] += 1
        samples.append(
            _perturb_config(
                transition,
                minimal_configs3[transition],
                sample_idx=seen[transition],
                global_idx=global_idx,
                seed=args.seed + global_idx,
                rng=np.random.default_rng(child_rngs[global_idx - 1]),
                initial_condition_mode=args.initial_condition_mode,
                weight_noise_rel=args.weight_noise_rel,
                weight_noise_floor=args.weight_noise_floor,
                scalar_noise_multiplier=args.scalar_noise_multiplier,
                keep_init_sigma=args.keep_init_sigma,
                initial_weights_only=args.initial_weights_only,
            )
        )
    return samples


def _response_from_frame(
    frame: pd.DataFrame,
    *,
    n_steps_per_phase: int,
    response_tail_fraction: float,
    baseline: tuple[float, float] | None = None,
    zscore_std_floor: float = 0.0,
) -> float:
    stim_start = 3 * n_steps_per_phase // 4
    stim_len = n_steps_per_phase - stim_start
    tail_start = stim_start + int(round((1.0 - response_tail_fraction) * stim_len))
    mask = frame["step"].to_numpy(dtype=int) % n_steps_per_phase >= tail_start
    values = frame.loc[mask, "y"].to_numpy(dtype=float)
    if baseline is None:
        return float(np.nanmean(values))
    mean, std = baseline
    scale = max(std, zscore_std_floor) if np.isfinite(std) and std > 1e-12 else max(1.0, zscore_std_floor)
    return float(np.nanmean((values - mean) / scale))


def _baseline(frames: list[pd.DataFrame], n_steps_per_phase: int) -> tuple[float, float, int]:
    stim_start = 3 * n_steps_per_phase // 4
    chunks = []
    for frame in frames:
        trial_step = frame["step"].to_numpy(dtype=int) % n_steps_per_phase
        chunks.append(frame.loc[trial_step < stim_start, "y"].to_numpy(dtype=float))
    values = np.concatenate([chunk for chunk in chunks if chunk.size])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, 0
    std = float(np.nanstd(values, ddof=1)) if values.size > 1 else 1.0
    return float(np.nanmean(values)), std if std > 1e-12 else 1.0, int(values.size)


def _probe_rows(
    model: CCNeuron,
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    phase: str,
    n_steps_per_phase: int,
    response_tail_fraction: float,
    zscore_responses: bool,
    baseline: tuple[float, float, int] | None = None,
    zscore_std_floor: float = 0.0,
) -> list[dict[str, Any]]:
    traces = []
    for condition, (x_full, c_full) in stimuli.items():
        for trace, x_phase in (("full", x_full), ("occlusion", torch.zeros_like(x_full))):
            frame = run_experimental_phase(model, x_phase, c_full, f"{trace}_{condition}_{phase}", update=False)
            traces.append((condition, trace, frame))

    local_baseline = _baseline([frame for _, _, frame in traces], n_steps_per_phase)
    baseline_mean, baseline_std, baseline_n = baseline or local_baseline
    rows = []
    for condition, trace, frame in traces:
        raw = _response_from_frame(
            frame,
            n_steps_per_phase=n_steps_per_phase,
            response_tail_fraction=response_tail_fraction,
        )
        response = raw
        if zscore_responses:
            response = _response_from_frame(
                frame,
                n_steps_per_phase=n_steps_per_phase,
                response_tail_fraction=response_tail_fraction,
                baseline=(baseline_mean, baseline_std),
                zscore_std_floor=zscore_std_floor,
            )
        rows.append(
            dict(
                condition=condition,
                phase=phase,
                stage=STAGES[phase],
                trace=trace,
                image_type=TRACE_TYPES[trace],
                response=response,
                raw_response=raw,
                baseline_mean=baseline_mean,
                baseline_std=baseline_std,
                baseline_n=baseline_n,
                local_baseline_mean=local_baseline[0],
                local_baseline_std=local_baseline[1],
                local_baseline_n=local_baseline[2],
            )
        )
    return rows, local_baseline


def _run_sample(
    config: dict[str, Any],
    *,
    n_steps_per_phase: int,
    response_tail_fraction: float,
    zscore_responses: bool,
    test_stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    training_stimuli: tuple[torch.Tensor, torch.Tensor],
    response_normalization: str,
    zscore_std_floor: float,
) -> pd.DataFrame:
    model = CCNeuron(**{key: value for key, value in config.items() if not key.startswith("_")})
    # Per-cell z-score floor scaled by the cell's spontaneous (baseline) drive.
    cell_floor = max(zscore_std_floor, BASELINE_STD_SCALE * float(config.get("baseline_drive_sigma", 0.0)))
    rows, naive_baseline = _probe_rows(
        model,
        test_stimuli,
        phase="naive",
        n_steps_per_phase=n_steps_per_phase,
        response_tail_fraction=response_tail_fraction,
        zscore_responses=zscore_responses,
        zscore_std_floor=cell_floor,
    )
    run_experimental_phase(model, training_stimuli[0], training_stimuli[1], "full_familiar_training", update=True)
    expert_baseline = naive_baseline if response_normalization == "naive" else None
    expert_rows, _ = _probe_rows(
        model,
        test_stimuli,
        phase="expert",
        n_steps_per_phase=n_steps_per_phase,
        response_tail_fraction=response_tail_fraction,
        zscore_responses=zscore_responses,
        baseline=expert_baseline,
        zscore_std_floor=cell_floor,
    )
    rows.extend(expert_rows)
    return pd.DataFrame(rows).assign(
        transition=config["_canonical_transition"],
        sample_idx=config["_sample_idx"],
        sample_global_idx=config["_sample_global_idx"],
        seed=config["seed"],
        experiment_series=PRIMARY_EXPERIMENT_SERIES,
    )


def _transition_table(response_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in response_df.itertuples(index=False):
        image_group, image_idx_original, image_idx_within_group = IMAGE_INFO[row.condition]
        rows.append(
            dict(
                transition=row.transition,
                image_group=image_group,
                image_idx_original=image_idx_original,
                image_idx_within_group=image_idx_within_group,
                neuron_idx=int(row.sample_global_idx),
                image_type=row.image_type,
                stage=row.stage,
                response=float(row.response),
            )
        )
    return pd.DataFrame(rows)


def _wide_table(transition_table: pd.DataFrame) -> pd.DataFrame:
    stage_order = transition_table["stage"].drop_duplicates().tolist()
    wide = transition_table.pivot_table(
        index=["transition", "image_group", "image_idx_original", "image_idx_within_group", "neuron_idx", "stage"],
        columns="image_type",
        values="response",
        aggfunc="mean",
    ).reset_index().rename(columns={"Full": "NO", "Occl": "O"})
    wide["stage"] = pd.Categorical(wide["stage"], categories=stage_order, ordered=True)
    return wide.sort_values(["transition", "image_group", "image_idx_original", "neuron_idx", "stage"]).reset_index(drop=True)


def _flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    flat = {key: config[key] for key in ("_canonical_transition", "_sample_idx", "_sample_global_idx", "seed")}
    flat = {
        "transition": flat["_canonical_transition"],
        "sample_idx": flat["_sample_idx"],
        "sample_global_idx": flat["_sample_global_idx"],
        "seed": flat["seed"],
    }
    for key, value in config.items():
        if key.startswith("_") or key == "seed":
            continue
        if key in INIT_KEYS and isinstance(value, dict):
            for idx, mu in enumerate(np.asarray(value.get("mu", []), dtype=float).reshape(-1)):
                flat[f"{key}.mu_{idx}"] = float(mu)
            if _is_num(value.get("sigma")):
                flat[f"{key}.sigma"] = float(value["sigma"])
        elif _is_num(value) or isinstance(value, str):
            flat[key] = value
        elif isinstance(value, tuple) and all(isinstance(item, bool) for item in value):
            flat.update({f"{key}_{idx}": bool(item) for idx, item in enumerate(value)})
    return flat


# Center/canonical panels are rendered with the canonical experiment_s protocol
# (long stimulus window) so the per-step EMA dynamics develop and the traces look
# like minimal2/plotsexperiment_s/transition_panels.
CENTER_PANEL_N_STEPS = 400
CENTER_PANEL_TEST_TRIALS = 4


def _run_panel_config(
    transition: str,
    config: dict[str, Any],
    *,
    training_trials: int,
) -> tuple[str, pd.DataFrame, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Lightweight naive->train->expert run for ONE config, full + occlusion traces
    only (no ablation variants), for the transition-panel plot. Module-level so it
    can be parallelised."""
    model = CCNeuron(**{key: value for key, value in config.items() if not key.startswith("_")})
    stimuli = _build_test_stimuli(n_steps_per_phase=CENTER_PANEL_N_STEPS, n_trials=CENTER_PANEL_TEST_TRIALS)
    training = _build_training_stimuli(n_steps_per_phase=CENTER_PANEL_N_STEPS, n_trials=training_trials)

    frames: list[pd.DataFrame] = []

    def probe(phase: str) -> None:
        for condition, (x_full, c_full) in stimuli.items():
            frames.append(run_experimental_phase(model, x_full, c_full, f"full_{condition}_{phase}", update=False))
            frames.append(run_experimental_phase(model, torch.zeros_like(x_full), c_full, f"occlusion_{condition}_{phase}", update=False))

    probe("naive")
    run_experimental_phase(model, training[0], training[1], "full_familiar_training", update=True)
    probe("expert")

    df = pd.concat([frame.assign(experiment_series=PRIMARY_EXPERIMENT_SERIES) for frame in frames], ignore_index=True)
    df["seed"] = config.get("seed", 42)
    long_df = wide_to_long(df)
    long_df = long_df.loc[long_df["experiment_phase"].isin(["naive", "expert"])].copy()
    return transition, long_df, stimuli


def _save_panels(
    configs_by_transition: dict[str, dict[str, Any]],
    *,
    out_dir: Path,
    name: str,
    training_trials: int,
    image_format: str,
    n_jobs: int,
) -> None:
    """Render a transition panel (full+occlusion traces) for the given configs,
    parallelised across configs. No per-config CSV (it is huge and not needed for
    the figure)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    order = list(configs_by_transition)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_panel_config)(t, configs_by_transition[t], training_trials=training_trials) for t in order
    )
    long_dfs = {t: long_df for t, long_df, _ in results}
    stimuli = results[0][2] if results else None
    if stimuli is None:
        return
    visualize_transition_panel(
        long_dfs,
        STIMULI=stimuli,
        save_path=str(out_dir),
        name=name,
        image_mode="both",
        transition_order=order,
        transition_labels={t: format_transition_label(t) for t in order},
        trace_types=("full", "occlusion"),
        save_in_transition_subdir=False,
        save_csv=False,
        image_format=image_format,
    )


def _save_center_panels(
    transition_order: list[str],
    *,
    output_dir: Path,
    training_trials: int,
    image_format: str = "png",
    n_jobs: int = -1,
) -> None:
    """Sanity check: transition panels for the exact (noise-free) sampler centers
    AND the raw config_s canonical examples, so both can be checked against the
    model mechanism (the centers can drift from what their names imply)."""
    _save_panels(
        {t: _center_config(t) for t in transition_order},
        out_dir=output_dir / "center_panels",
        name="center_transition_panel_naive_expert",
        training_trials=training_trials,
        image_format=image_format,
        n_jobs=n_jobs,
    )
    _save_panels(
        {t: _canonical_config(t) for t in transition_order},
        out_dir=output_dir / "canonical_panels",
        name="canonical_transition_panel_naive_expert",
        training_trials=training_trials,
        image_format=image_format,
        n_jobs=n_jobs,
    )


def _robust_response_limits(summaries: list[pd.DataFrame], *, hi_percentile: float, pad: float = 0.4) -> list[float]:
    """Response axis limits that ignore a few extreme outliers (point 6): scale to
    the bulk so a handful of very strong responders fall outside rather than
    blowing up the whole panel. hi_percentile=100 reproduces the real-data min/max."""
    cols = ["NO_Pre", "O_Pre", "NO_Target", "O_Target"]
    values = np.concatenate([s[cols].to_numpy(dtype=float).reshape(-1) for s in summaries])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [-1.0, 1.0]
    lo = float(np.nanpercentile(values, max(0.0, 100.0 - hi_percentile)))
    hi = float(np.nanpercentile(values, hi_percentile))
    return [lo - pad, hi + pad]


def _robust_shift_limits(summaries: list[pd.DataFrame], *, hi_percentile: float, pad_ratio: float = 0.12, fallback: float = 0.5) -> list[float]:
    values = np.concatenate([s[["dNO", "dO"]].to_numpy(dtype=float).reshape(-1) for s in summaries])
    values = np.abs(values[np.isfinite(values)])
    if values.size == 0:
        return [-fallback, fallback]
    extent = float(np.nanpercentile(values, hi_percentile))
    if not np.isfinite(extent) or extent == 0:
        extent = fallback
    else:
        extent *= 1.0 + pad_ratio
    return [-extent, extent]


def _panel_point_summaries(
    configs_by_transition: dict[str, dict[str, Any]],
    *,
    args: argparse.Namespace,
    test_stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    training_stimuli: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, pd.DataFrame]:
    """One (NO, O) naive/expert position per transition, run through exactly the
    scatter pipeline (per-cell denominator and all), for overlaying the canonical
    examples / sampler centers on the aggregate scatter."""
    frames = []
    for idx, (name, config) in enumerate(configs_by_transition.items(), start=1):
        cfg = copy.deepcopy(config)
        cfg.update(_canonical_transition=name, _sample_idx=1, _sample_global_idx=idx)
        cfg.setdefault("seed", 42)
        frames.append(
            _run_sample(
                cfg,
                n_steps_per_phase=args.n_steps_per_phase,
                response_tail_fraction=args.response_tail_fraction,
                zscore_responses=not args.raw_responses,
                test_stimuli=test_stimuli,
                training_stimuli=training_stimuli,
                response_normalization=args.response_normalization,
                zscore_std_floor=args.zscore_std_floor,
            )
        )
    wide = _wide_table(_transition_table(pd.concat(frames, ignore_index=True)))
    out: dict[str, pd.DataFrame] = {}
    for group in ("familiar", "novel"):
        piv = wide.loc[wide["image_group"] == group].pivot_table(
            index="transition", columns="stage", values=["NO", "O"], aggfunc="mean"
        )
        df = pd.DataFrame({
            "transition": piv.index.to_numpy(),
            "NO_Pre": piv[("NO", "Naive")].to_numpy(),
            "O_Pre": piv[("O", "Naive")].to_numpy(),
            "NO_Target": piv[("NO", "Expert")].to_numpy(),
            "O_Target": piv[("O", "Expert")].to_numpy(),
        })
        df["dNO"] = df["NO_Target"] - df["NO_Pre"]
        df["dO"] = df["O_Target"] - df["O_Pre"]
        out[group] = df
    return out


def _overlay_examples(fig: plt.Figure, points: pd.DataFrame, *, marker: str, color: str, label: str, annotate: bool) -> None:
    """Overlay highlighted example points (one per transition) on the three
    'by rotated sector' panels (naive / expert / shift)."""
    # 3x3 grid -> fig.axes = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),...]; the sector
    # panels are (1,0)=shift, (1,1)=naive, (1,2)=expert.
    ax_shift, ax_naive, ax_expert = fig.axes[3], fig.axes[4], fig.axes[5]
    for ax, (xc, yc) in ((ax_naive, ("NO_Pre", "O_Pre")), (ax_expert, ("NO_Target", "O_Target")), (ax_shift, ("dNO", "dO"))):
        ax.scatter(points[xc], points[yc], s=95, marker=marker, facecolors="none", edgecolors=color, linewidths=1.7, zorder=6, label=label)
    if annotate:
        for _, row in points.iterrows():
            ax_expert.annotate(
                format_transition_label(str(row["transition"])),
                (row["NO_Target"], row["O_Target"]),
                fontsize=5.5, color=color, zorder=7, xytext=(3, 2), textcoords="offset points",
            )
    handles, labels = ax_expert.get_legend_handles_labels()
    seen: dict[str, Any] = {}
    for h, lb in zip(handles, labels):
        seen.setdefault(lb, h)
    ax_expert.legend(seen.values(), seen.keys(), loc="upper right", fontsize=7, frameon=True)


def _save_summary(
    summary: pd.DataFrame,
    path: Path,
    title: str,
    response_lims: list[float],
    shift_lims: list[float],
    export_panels: bool,
    image_format: str,
    overlays: list[tuple[pd.DataFrame, dict]] | None = None,
) -> None:
    fig = th.plot_mean_transition_summary(
        summary,
        title=title,
        start_label="Naive",
        end_label="Expert",
        response_lims=response_lims,
        shift_lims=shift_lims,
        style=PLOT_STYLE,
    )
    for points, style in overlays or []:
        _overlay_examples(fig, points, **style)
    path = path.with_suffix(f".{image_format}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if export_panels:
        th.export_figure_panels(fig, path.parent / f"{path.stem}_panels", path.stem, formats=(image_format,))
    plt.close(fig)
    th.save_rotated_sector_unit_legend(summary, path.with_name(f"{path.stem}_sector_legend.{image_format}"), title=None, formats=(image_format,))


def _save_plots(
    transition_table: pd.DataFrame,
    *,
    output_dir: Path,
    transition_order: list[str],
    threshold: float,
    plot_by_transition: bool,
    export_panels: bool,
    image_format: str,
    axis_clip_percentile: float,
    overlays_by_group: dict[str, list[tuple[pd.DataFrame, dict]]] | None = None,
) -> None:
    figures_dir = output_dir / "figures"
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    overlays_by_group = overlays_by_group or {}

    wide = _wide_table(transition_table)
    aggregate = {
        group: th.build_mean_summary(wide, image_group=group, pre_stage="Naive", target_stage="Expert", threshold=threshold)
        for group in ("familiar", "novel")
    }
    # A single shared response/shift frame across the familiar and novel panels
    # (like the transitions>threshold notebook), but scaled to the bulk so a few
    # extreme outliers fall outside the panel instead of compressing it
    # (axis_clip_percentile=100 reproduces the notebook's exact min/max framing).
    summaries = list(aggregate.values())
    response_lims = _robust_response_limits(summaries, hi_percentile=axis_clip_percentile)
    shift_lims = _robust_shift_limits(summaries, hi_percentile=axis_clip_percentile)

    fraction_frames = []
    for group, summary in aggregate.items():
        summary.to_csv(summaries_dir / f"aggregate_{group}_summary.csv", index=False)
        fraction_frames.append(th.sector_fraction_table(summary).assign(scope=f"aggregate {group}", transition="all", image_group=group))
        _save_summary(
            summary,
            figures_dir / f"aggregate_{group}_summary.png",
            f"Model scatter - all transitions - {group}",
            response_lims,
            shift_lims,
            export_panels,
            image_format,
            overlays=overlays_by_group.get(group),
        )

    if plot_by_transition:
        for transition in transition_order:
            subset = wide.loc[wide["transition"] == transition].copy()
            if subset.empty:
                continue
            for group in ("familiar", "novel"):
                summary = th.build_mean_summary(subset, image_group=group, pre_stage="Naive", target_stage="Expert", threshold=threshold)
                summary.to_csv(summaries_dir / f"{transition}_{group}_summary.csv", index=False)
                fraction_frames.append(th.sector_fraction_table(summary).assign(scope=f"{transition} {group}", transition=transition, image_group=group))
                _save_summary(
                    summary,
                    figures_dir / "by_transition" / f"{transition}_{group}_summary.png",
                    f"Model scatter - {transition} - {group}",
                    response_lims,
                    shift_lims,
                    export_panels,
                    image_format,
                )

    pd.concat(fraction_frames, ignore_index=True).to_csv(summaries_dir / "sector_fractions.csv", index=False)


def run_model_scatter(args: argparse.Namespace) -> None:
    if args.n_samples < 1:
        raise ValueError("n_samples must be >= 1.")
    if args.n_steps_per_phase < 4:
        raise ValueError("n_steps_per_phase must be >= 4.")
    if args.test_trials < 1 or args.training_trials < 1:
        raise ValueError("test_trials and training_trials must be >= 1.")
    if not 0.0 < args.response_tail_fraction <= 1.0:
        raise ValueError("response_tail_fraction must be in (0, 1].")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transition_order = list(minimal_configs3)
    samples = _sample_configs(args, transition_order)

    torch.manual_seed(args.seed)
    test_stimuli = _build_test_stimuli(n_steps_per_phase=args.n_steps_per_phase, n_trials=args.test_trials)
    training_stimuli = _build_training_stimuli(n_steps_per_phase=args.n_steps_per_phase, n_trials=args.training_trials)
    response_frames = Parallel(n_jobs=args.n_jobs, verbose=10 if args.n_jobs != 1 else 0)(
        delayed(_run_sample)(
            sample,
            n_steps_per_phase=args.n_steps_per_phase,
            response_tail_fraction=args.response_tail_fraction,
            zscore_responses=not args.raw_responses,
            test_stimuli=test_stimuli,
            training_stimuli=training_stimuli,
            response_normalization=args.response_normalization,
            zscore_std_floor=args.zscore_std_floor,
        )
        for sample in samples
    )

    response_df = pd.concat(response_frames, ignore_index=True)
    transition_table = _transition_table(response_df)
    invalid = transition_table.loc[~np.isfinite(transition_table["response"])].copy()

    response_df.to_csv(args.output_dir / "sample_responses.csv", index=False)
    transition_table.to_csv(args.output_dir / "transition_table.csv", index=False)
    pd.DataFrame(_flatten_config(sample) for sample in samples).to_csv(args.output_dir / "sampled_config_parameters.csv", index=False)
    (args.output_dir / "sampled_configs.json").write_text(json.dumps(samples, indent=2, default=repr))
    if not invalid.empty:
        invalid.to_csv(args.output_dir / "invalid_responses.csv", index=False)

    counts = {name: sum(sample["_canonical_transition"] == name for sample in samples) for name in transition_order}
    metadata = {
        "requested_n_samples": args.n_samples,
        "n_samples_total": len(samples),
        "transition_sampling": "canonical" if args.canonical_only else args.transition_sampling,
        "transition_sample_counts": counts,
        "transition_weights": {name: TRANSITIONS[name]["weight"] for name in transition_order},
        "initial_condition_mode": args.initial_condition_mode,
        "seed": args.seed,
        "n_steps_per_phase": args.n_steps_per_phase,
        "test_trials": args.test_trials,
        "training_trials": args.training_trials,
        "fixed_scalars": list(FIXED_SCALARS),
        "response_units": "raw" if args.raw_responses else "zscore",
        "response_normalization": "none" if args.raw_responses else args.response_normalization,
        "zscore_std_floor": None if args.raw_responses else args.zscore_std_floor,
        "response_tail_fraction": args.response_tail_fraction,
        "sector_threshold": args.threshold,
        "ff_plasticity_broad": FF_PLASTICITY_BROAD,
        "ff_plasticity_narrow": FF_PLASTICITY_NARROW,
        "stimulus_specs": STIMULUS_SPECS,
        "n_invalid_response_rows": int(len(invalid)),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=repr))

    overlays_by_group: dict[str, list[tuple[pd.DataFrame, dict]]] = {}
    example_points_path = args.output_dir / "example_points.csv"
    if args.overlay_examples and not args.canonical_only:
        canonical = _panel_point_summaries(
            {t: _canonical_config(t) for t in transition_order},
            args=args, test_stimuli=test_stimuli, training_stimuli=training_stimuli,
        )
        centers = _panel_point_summaries(
            {t: _center_config(t) for t in transition_order},
            args=args, test_stimuli=test_stimuli, training_stimuli=training_stimuli,
        )
        for group in ("familiar", "novel"):
            overlays_by_group[group] = [
                (centers[group], {"marker": "D", "color": "0.35", "label": "sampler center", "annotate": False}),
                (canonical[group], {"marker": "*", "color": "black", "label": "config_s canonical", "annotate": True}),
            ]
        canonical_csv = pd.concat([df.assign(image_group=g, kind="canonical") for g, df in canonical.items()], ignore_index=True)
        center_csv = pd.concat([df.assign(image_group=g, kind="center") for g, df in centers.items()], ignore_index=True)
        pd.concat([canonical_csv, center_csv], ignore_index=True).to_csv(example_points_path, index=False)
    elif example_points_path.exists():
        example_points_path.unlink()

    _save_plots(
        transition_table,
        output_dir=args.output_dir,
        transition_order=transition_order,
        threshold=args.threshold,
        plot_by_transition=args.plot_by_transition,
        export_panels=args.export_panels,
        image_format=args.image_format,
        axis_clip_percentile=args.axis_clip_percentile,
        overlays_by_group=overlays_by_group,
    )

    if not args.skip_center_panels and not args.canonical_only:
        _save_center_panels(
            transition_order,
            output_dir=args.output_dir,
            training_trials=args.training_trials,
            image_format=args.image_format,
            n_jobs=args.n_jobs,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample noisy minimal2 configs and plot model-scatter transitions.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-samples", type=int, default=1200)
    parser.add_argument("--n-steps-per-phase", type=int, default=400)
    parser.add_argument("--test-trials", type=int, default=2)
    parser.add_argument("--training-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--weight-noise-rel", type=float, default=0.55)
    parser.add_argument("--weight-noise-floor", type=float, default=0.07)
    parser.add_argument("--scalar-noise-multiplier", type=float, default=1.75)
    parser.add_argument("--keep-init-sigma", action="store_true")
    parser.add_argument("--initial-weights-only", action="store_true")
    parser.add_argument("--canonical-only", action="store_true")
    parser.add_argument("--transition-sampling", choices=("data-like", "equal"), default="data-like")
    parser.add_argument("--initial-condition-mode", choices=("spec", "canonical-neighborhood"), default="spec")
    parser.add_argument("--raw-responses", action="store_true")
    parser.add_argument("--response-normalization", choices=("naive", "phase"), default="naive")
    parser.add_argument("--zscore-std-floor", type=float, default=0.04)
    parser.add_argument("--response-tail-fraction", type=float, default=1.0)
    # Default matches the transitions>threshold notebook so the model scatter is
    # sectorized exactly like the real data.
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--image-format", choices=("png", "svg", "eps"), default="png", help="Figure output format (default png).")
    parser.add_argument("--axis-clip-percentile", type=float, default=99.0, help="Scale figure axes to this percentile of responses so a few extreme outliers fall outside the panel (100 = exact min/max like the real-data notebook).")
    parser.add_argument("--skip-center-panels", action="store_true", help="Skip the experiment_s transition panel for the exact sampler centers.")
    parser.add_argument("--overlay-examples", action="store_true", help="Overlay the config_s canonical examples and sampler centers on the aggregate scatter.")
    parser.add_argument("--plot-by-transition", action="store_true")
    parser.add_argument("--export-panels", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_model_scatter(parse_args())


if __name__ == "__main__":
    main()
