from __future__ import annotations

import argparse
from pathlib import Path

from . import model_scatter, transition_templates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the publication model-scatter simulation.")
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
    
    """
    1  silent_broad_FFonly
    2  silent_broad_FB_weak
    3  silent_broad_FB_mid
    4  silent_broad_FB_partial2
    5  mid_broad_FFonly
    6  very_weak_broad_FB_partial2
    7  weak_broad_FB_mixed_bridge
    8  mid_broad_FB_weak
    9  mid_broad_FB_partial2
    10 strong_broad_FB_strong
    11 narrow_weak
    12 narrow_mid
    13 narrow_novel
    14 novel_weak_FB_diagonal
    15 weak_broad_FFonly
    16 silent_broad_FB_strong
    """
    
    parser.add_argument(
        "--fam-examples",
        type=int,
        nargs="*",
        default=[],
        metavar="TEMPLATE_NUM",
        help="Template numbers to highlight in the aggregate familiar scatter and example trace panel.",
    )
    parser.add_argument(
        "--nov-examples",
        type=int,
        nargs="*",
        default=[],
        metavar="TEMPLATE_NUM",
        help="Template numbers to highlight in the aggregate novel scatter and example trace panel.",
    )
    parser.add_argument(
        "--use-center-examples",
        action="store_true",
        help="Highlight and plot noise-free template centers instead of sampled cells for the requested examples.",
    )
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
