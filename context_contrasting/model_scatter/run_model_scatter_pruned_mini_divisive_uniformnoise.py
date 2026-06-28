"""EXPERIMENT (throwaway): the divisive pruned-mini run, but with ONE noise
characterization per sampled parameter instead of a per-template/per-level one.

Question being tested: the center weights already differ per template/level -- how
much does the scatter actually change if we stop hand-tuning (rel, floor, lo, hi)
per FF strength / FB level and per-template (gain_clip, drive_clip), and instead use
a single, cautiously small-medium spread for each of the four sampled parameters
(FF, FB, apical_gain_strength, apical_drive_threshold)?

What is kept untouched:
  * every template's / level's CENTER (ff tuned/silent, fb center, gain, drive),
  * all NON-sampled tunings (lat, pvlat, W_pv, baseline center+clip, gain_threshold,
    gain_k, ff_plasticity_scale, learning rates, the divisive overrides + k).

What is unified:
  * FF_STRENGTHS[*]   -> (rel, floor, lo, hi) = UNIFORM_FF
  * FB_LEVELS[*]      -> (rel, floor, lo, hi) = UNIFORM_FB
  * every gain_clip   -> UNIFORM_GAIN_CLIP   (the only per-template knob for gain;
                          the gain SD itself is already global in base.SCALAR_NOISE)
  * every drive_clip  -> UNIFORM_DRIVE_CLIP  (likewise)

Clips are chosen wide enough to contain ALL kept centers so none get pinned
(gain centers span 2.2..6.8; drive centers span 0.005..1.25 because narrow cells
sit at a deliberately different drive scale -- so a single drive clip necessarily
loosens the broad-cell drive a lot; that is part of what this test reveals).

Run like the normal divisive script, just a different output dir:

  python -m context_contrasting.model_scatter.run_model_scatter_pruned_mini_divisive_uniformnoise \
      --n-jobs 10 --n-steps-per-phase 300 --training-trials 7 --test-trials 4 \
      --n-samples 250 --plot-center-panels --output-dir outputs_uniform_noise_test
"""

from __future__ import annotations

import copy
import os

os.environ.setdefault("CC_MODEL", "divisive")

from context_contrasting.model_scatter import run_model_scatter_pruned_mini as pm  # noqa: E402
from context_contrasting.model_scatter import run_model_scatter_pruned_mini_divisive as div  # noqa: E402

# Cautiously small-medium spread (one per parameter). rel ~0.40 sits in the middle
# of the per-level options (FF 0.30-0.62, FB 0.18-0.60); floor near the small end.
UNIFORM_FF: dict[str, float] = dict(rel=0.40, floor=0.010, lo=0.0, hi=0.20)
UNIFORM_FB: dict[str, float] = dict(rel=0.40, floor=0.012, lo=0.0, hi=0.40)
# Clips must contain every kept center: gain 2.2..6.8, drive 0.005..1.25.
UNIFORM_GAIN_CLIP: tuple[float, float] = (1.5, 8.0)
UNIFORM_DRIVE_CLIP: tuple[float, float] = (0.0, 1.5)


def _uniformize() -> None:
    """Overwrite only the noise/clip fields, leaving every center + other tuning."""
    ff = copy.deepcopy(pm.FF_STRENGTHS)
    for level in ff.values():
        level.update(UNIFORM_FF)  # keeps tuned/silent centers
    pm.FF_STRENGTHS = ff

    fb = copy.deepcopy(pm.FB_LEVELS)
    for level in fb.values():
        level.update(UNIFORM_FB)  # keeps receives/center
    pm.FB_LEVELS = fb

    templates = copy.deepcopy(pm.TEMPLATES)
    for tpl in templates.values():
        if "gain_clip" in tpl:
            tpl["gain_clip"] = UNIFORM_GAIN_CLIP
        if "drive_clip" in tpl:
            tpl["drive_clip"] = UNIFORM_DRIVE_CLIP
    pm.TEMPLATES = templates

    width_classes = copy.deepcopy(pm.WIDTH_CLASSES)
    for cls in width_classes.values():
        cls["gain_clip"] = UNIFORM_GAIN_CLIP
        cls["drive_clip"] = UNIFORM_DRIVE_CLIP
    pm.WIDTH_CLASSES = width_classes


# Run div._apply_divisive first (it deepcopies + reassigns pm.TEMPLATES /
# pm.WIDTH_CLASSES), THEN uniformize on top of the divisive-tuned tables.
_orig_apply = div._apply_divisive


def _patched_apply() -> None:
    _orig_apply()
    _uniformize()


div._apply_divisive = _patched_apply


if __name__ == "__main__":
    div.main()
