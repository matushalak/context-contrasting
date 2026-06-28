"""EXPERIMENT (throwaway): the FINAL collapse. ONE shared weight-noise setting for
BOTH w_ff and w_fb, AND no scalar (gain/drive) noise. So the entire sampler's noise
model is a single (rel, floor, lo, hi) applied to every plastic weight vector; every
other degree of freedom is the per-template/per-level CENTER plus the fixed tunings.

Equalized weight noise (FF == FB):
    rel = 0.40, floor = 0.011, lo = 0.0, hi = 0.30
Scalar noise (apical_gain_strength, apical_drive_threshold): OFF (centers only).

Everything else kept: all centers (ff tuned/silent, fb center, gain, drive), lat,
pvlat, W_pv, baseline, gain_threshold/k, learning rates, divisive overrides + k.

NOTE: hi=0.30 equals the strong_sat FB center (0.300), so the upper half of those
draws is clipped -- watch the naive O cloud top.

  python -m context_contrasting.model_scatter.run_model_scatter_pruned_mini_divisive_onenoise \
      --n-jobs 10 --n-steps-per-phase 300 --training-trials 7 --test-trials 4 \
      --n-samples 250 --plot-center-panels --output-dir outputs_onenoise_test
"""

from __future__ import annotations

import copy
import os

os.environ.setdefault("CC_MODEL", "divisive")

from context_contrasting.model_scatter import run_model_scatter_pruned_mini as pm  # noqa: E402
from context_contrasting.model_scatter import run_model_scatter_pruned_mini_divisive as div  # noqa: E402

ONE_NOISE: dict[str, float] = dict(rel=0.40, floor=0.011, lo=0.0, hi=0.30)


def _equalize_ff_fb() -> None:
    ff = copy.deepcopy(pm.FF_STRENGTHS)
    for level in ff.values():
        level.update(ONE_NOISE)  # keeps tuned/silent centers
    pm.FF_STRENGTHS = ff
    fb = copy.deepcopy(pm.FB_LEVELS)
    for level in fb.values():
        level.update(ONE_NOISE)  # keeps receives/center
    pm.FB_LEVELS = fb


_orig_apply = div._apply_divisive
_orig_configure = pm._configure_pruned_variant


def _patched_apply() -> None:
    _orig_apply()
    _equalize_ff_fb()


def _patched_configure(n_steps_per_phase: int) -> None:
    _orig_configure(n_steps_per_phase)
    pm.base.SCALAR_NOISE = {}  # no gain/drive jitter


div._apply_divisive = _patched_apply
pm._configure_pruned_variant = _patched_configure


if __name__ == "__main__":
    div.main()
