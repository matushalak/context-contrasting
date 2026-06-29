from __future__ import annotations

import argparse
from pathlib import Path

from . import model_scatter, transition_templates_principled as transition_templates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the principled publication model-scatter simulation.")
    parser.add_argument("--output-dir", type=Path, default=transition_templates.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-samples", type=int, default=250)
    parser.add_argument("--n-steps-per-phase", type=int, default=300)
    parser.add_argument("--test-trials", type=int, default=4)
    parser.add_argument("--training-trials", type=int, default=7)
    parser.add_argument("--training-stimulus-order", choices=("randomized", "fixed"), default="randomized")
    parser.add_argument("--seed", type=int, default=7151)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--scalar-noise-multiplier", type=float, default=1.75)
    parser.add_argument("--canonical-only", action="store_true", help="Not supported; template tuning is sampled by cell.")
    parser.add_argument("--transition-sampling", choices=("data-like", "equal"), default="data-like")
    parser.add_argument("--zscore-std-floor", type=float, default=0.04)
    parser.add_argument("--response-tail-fraction", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--image-format", choices=("png", "svg", "eps"), default="png")
    parser.add_argument("--axis-clip-percentile", type=float, default=99.0)
    parser.add_argument("--plot-center-panels", action="store_true")
    parser.add_argument("--plot-by-transition", action="store_true")
    parser.add_argument("--export-panels", action="store_true")
    parser.add_argument("--fam-examples", type=int, nargs="*", default=[], metavar="TEMPLATE_NUM")
    parser.add_argument("--nov-examples", type=int, nargs="*", default=[], metavar="TEMPLATE_NUM")
    parser.add_argument("--use-center-examples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.canonical_only:
        raise ValueError("--canonical-only is not supported by the publication sampler.")
    args.skip_center_panels = not args.plot_center_panels
    transition_templates.configure_model_scatter(args.n_steps_per_phase)
    model_scatter.run_model_scatter(args)
    transition_templates.write_metadata(args)


if __name__ == "__main__":
    main()
