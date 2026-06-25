"""Mini model-scatter variant with fixed PV tuning.

This entrypoint reuses ``run_model_scatter.py`` but disables plasticity of
``W_pv`` and ``w_pv_lat`` in every transition template. The only learned weights
during familiar training are therefore ``w_ff``, ``w_fb``, and ``w_lat``. Fixed
PV tuning is pre-strengthened by default from a quick ablation grid; the scaling
controls remain exposed for follow-up sweeps.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from context_contrasting.model_scatter import run_model_scatter as base


DEFAULT_OUTPUT_DIR = base.PACKAGE_DIR / "outputs_mini"


def _scale_center(init_spec: dict[str, Any], scale: float) -> None:
    if scale == 1.0:
        return
    init_spec["center"] = [float(value) * scale for value in init_spec["center"]]


def _configure_mini_variant(args: argparse.Namespace) -> None:
    """Patch the imported base runner to use fixed PV tuning for this process."""
    transitions = copy.deepcopy(base.TRANSITIONS)
    for spec in transitions.values():
        spec["fix"].update(
            {
                "pv_plasticity": False,
                "pv_lat_plasticity": False,
            }
        )
        _scale_center(spec["init"]["W_pv_init"], args.pv_init_scale)
        _scale_center(spec["init"]["w_pv_lat_init"], args.pvlat_init_scale)

    shared_lrs = dict(base.SHARED_LEARNING_RATES)
    shared_lrs["lr_ff"] *= args.ff_lr_scale
    shared_lrs["lr_fb"] *= args.fb_lr_scale
    shared_lrs["lr_lat"] *= args.lat_lr_scale
    shared_lrs["lr_pv"] = 0.0

    base.TRANSITIONS = transitions
    base.SHARED_LEARNING_RATES = shared_lrs


def _write_mini_metadata(args: argparse.Namespace) -> None:
    metadata_path = args.output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["mini_variant"] = {
        "description": "PV tuning fixed; plasticity only in w_ff, w_fb, and w_lat.",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the model-scatter variant with fixed PV tuning and only PyC-weight plasticity."
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
    parser.add_argument("--skip-center-panels", action="store_true")
    parser.add_argument("--plot-by-transition", action="store_true")
    parser.add_argument("--export-panels", action="store_true")

    parser.add_argument("--pv-init-scale", type=float, default=1.5, help="Multiplier for fixed W_pv template centers.")
    parser.add_argument("--pvlat-init-scale", type=float, default=1.0, help="Multiplier for fixed w_pv_lat template centers.")
    parser.add_argument("--ff-lr-scale", type=float, default=1.0, help="Multiplier for the shared FF learning rate.")
    parser.add_argument("--fb-lr-scale", type=float, default=1.0, help="Multiplier for the shared FB learning rate.")
    parser.add_argument("--lat-lr-scale", type=float, default=2.0, help="Multiplier for the shared LAT learning rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_mini_variant(args)
    base.run_model_scatter(args)
    _write_mini_metadata(args)


if __name__ == "__main__":
    main()
