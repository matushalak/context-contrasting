"""EXPERIMENT (throwaway): the divisive pruned-mini run, but with NO per-cell noise
on the scalar parameters apical_gain_strength and apical_drive_threshold -- they are
pinned to their per-template centers. Only w_ff and w_fb still get Gaussian draws.

Question: do gain / drive_threshold need cell-to-cell jitter at all, or is the
per-template center enough once FF/FB weights carry the heterogeneity?

Mechanism: in the pruned variant ``_configure_pruned_variant`` restricts
``base.SCALAR_NOISE`` to exactly {apical_gain_strength, apical_drive_threshold}
(baseline is not jittered there). Emptying it removes all scalar jitter; the
centers, clips, and every other tuning are untouched.

Run like the normal divisive script, different output dir:

  python -m context_contrasting.model_scatter.run_model_scatter_pruned_mini_divisive_noscalarnoise \
      --n-jobs 10 --n-steps-per-phase 300 --training-trials 7 --test-trials 4 \
      --n-samples 250 --plot-center-panels --output-dir outputs_noscalarnoise_test
"""

from __future__ import annotations

import os

os.environ.setdefault("CC_MODEL", "divisive")

from context_contrasting.model_scatter import run_model_scatter_pruned_mini as pm  # noqa: E402
from context_contrasting.model_scatter import run_model_scatter_pruned_mini_divisive as div  # noqa: E402

# Patch _configure_pruned_variant (called by div.main AFTER _apply_divisive) so we
# empty SCALAR_NOISE *after* it has set it -> no gain/drive jitter.
_orig_configure = pm._configure_pruned_variant


def _patched_configure(n_steps_per_phase: int) -> None:
    _orig_configure(n_steps_per_phase)
    pm.base.SCALAR_NOISE = {}


pm._configure_pruned_variant = _patched_configure


if __name__ == "__main__":
    div.main()
