"""EXPERIMENT: the principled pruned-mini setup run on the DIVISIVE-inhibition
model (``minimal_divisive.CCNeuron``), without touching the reference
``run_model_scatter_pruned_mini.py`` (which uses subtractive ``minimal_s``).

Motivation. In the subtractive model the surround enters as
``y = act(apical_gain*(y_ff - y_lat) + apical_drive + baseline)`` -- the FF->PV
surround CANCELS the feedback drive on the full image by SUBTRACTION, which
overshoots the (positive, noise-set) spontaneous baseline -> O-responders fall
into NO<0 instead of landing in the data's neutral zone at NO~0. The real data
shows a clear floor at baseline (divisive / shunting PV inhibition: a cancelled
drive is divided toward the resting rate, never pushed below it).

``minimal_divisive`` changes ``forward()`` so the surround NORMALIZES the drive
instead of subtracting it:

    y = act( (apical_gain * y_ff + apical_drive) / (1 + y_lat) + baseline_drive - a )

Consequences / how to tune in this regime:
  * The full-image response now FLOORS at baseline (NO -> 0+ as the surround
    strengthens, never below) -> the NO<0 knife-edge is gone, so the surround can
    be pushed hard without overshooting negative.
  * To pull an O-responder's NO down to ~0 the division must be strong, i.e.
    ``1 + y_lat`` must be order >1 -> w_lat (``lat``) needs to be MUCH larger than
    in the subtractive templates (there it was ~0.1-0.3; here order ~1+).
  * A genuine -NO population now comes from the SUBTRACTIVE adaptation term ``a``
    (kept outside the division), NOT from surround cancellation -- so -NO is tuned
    via FF adaptation on the no-FB cells, independently of the O-cloud centring.

The model is swapped via the CC_MODEL=divisive env var (read at import by
run_model_scatter, so loky workers inherit it too). Per-template retuning for the
divisive regime goes in ``DIVISIVE_OVERRIDES``.

Run exactly like the reference:

  python -m context_contrasting.model_scatter.run_model_scatter_pruned_mini_divisive \
      --n-jobs 10 --n-steps-per-phase 300 --training-trials 7 --test-trials 4 \
      --n-samples 250 --plot-center-panels --output-dir outputs_pruned_divisive
"""

from __future__ import annotations

import copy
import os

# Must be set before run_model_scatter (and its workers) import CCNeuron.
os.environ["CC_MODEL"] = "divisive"

from context_contrasting.model_scatter import run_model_scatter_pruned_mini as pm  # noqa: E402


# Divisive-surround gain k (response drive normalized by 1 + k*y_lat). Overridable
# via the CC_DIVISIVE_K env var so a sweep can set it per run without editing here.
DIVISIVE_K: float = float(os.environ.get("CC_DIVISIVE_K", "10.0"))

# Per-template retuning for the divisive regime. Start empty (reference templates
# verbatim); fill in as we retune.
DIVISIVE_OVERRIDES: dict[str, dict] = {
    # O-responders (silent FF + FB): strong surround so the divisive term (1+k*y_lat)
    # pulls the full-image NO down onto the baseline floor (NO~0) while occluded
    # (no FF -> low y_lat) keeps its feedback-driven O.
    "silent_broad_FB_strong": dict(lat=0.85, pvlat=0.05, pv_tuned=0.60, pv_silent=0.60),
    "silent_broad_FB_weak": dict(lat=0.85, pvlat=0.05, pv_tuned=0.60, pv_silent=0.60),
    "silent_broad_FB_mid": dict(lat=0.85, pvlat=0.05, pv_tuned=0.60, pv_silent=0.60),
    "silent_broad_FB_partial2": dict(lat=0.85, pvlat=0.05, pv_tuned=0.60, pv_silent=0.60),
    # broad + FB movers: moderate surround so the intact NOVEL FF partly survives the
    # division (-> weak novel +NO+O) while familiar (adapted FF) centres at NO~0.
    "mid_broad_FB_weak": dict(lat=0.55, pvlat=0.05, pv_tuned=0.50, pv_silent=0.50),
    "mid_broad_FB_partial2": dict(lat=0.55, pvlat=0.05, pv_tuned=0.50, pv_silent=0.50),
    "strong_broad_FB_strong": dict(lat=0.55, pvlat=0.05, pv_tuned=0.50, pv_silent=0.50),
    "very_weak_broad_FB_partial2": dict(lat=0.55, pvlat=0.05, pv_tuned=0.50, pv_silent=0.50),
    # FF-only (-NO) and narrow (+NO) cells keep LOW surround: their NO must survive
    # (then drop via adaptation -> -NO, or be gain-amplified -> +NO).
}


def _apply_divisive() -> None:
    if DIVISIVE_OVERRIDES:
        templates = copy.deepcopy(pm.TEMPLATES)
        for name, override in DIVISIVE_OVERRIDES.items():
            templates[name].update(override)
        pm.TEMPLATES = templates


def main() -> None:
    args = pm.parse_args()
    if args.canonical_only:
        raise ValueError("--canonical-only is not supported by the pruned mini sampler.")
    args.skip_center_panels = not args.plot_center_panels
    _apply_divisive()
    pm._configure_pruned_variant(args.n_steps_per_phase)
    # Inject k into every base config so it rides along (pickled) to the workers.
    for cfg in pm.base.minimal_configs3.values():
        cfg["divisive_gain"] = DIVISIVE_K
    pm.base.run_model_scatter(args)
    pm._write_metadata(args)


if __name__ == "__main__":
    main()
