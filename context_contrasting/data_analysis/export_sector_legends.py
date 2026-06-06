from __future__ import annotations

import argparse
from pathlib import Path

import context_contrasting.data_analysis.transitions_helpers as th


def export_real_data_sector_legends(
    *,
    data_dir: Path,
    output_dir: Path,
    threshold: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    act_table = th.load_transition_table(data_dir / "transitions_act.csv")
    post_table = th.load_transition_table(data_dir / "transitions_post.csv")

    summaries = {
        "act_mean": th.build_mean_summary(
            act_table,
            image_group="all",
            pre_stage="Pre",
            target_stage="Task",
            threshold=threshold,
        ),
        "post_familiar_mean": th.build_mean_summary(
            post_table,
            image_group="familiar",
            pre_stage="Pre",
            target_stage="Post",
            threshold=threshold,
        ),
        "post_novel_mean": th.build_mean_summary(
            post_table,
            image_group="novel",
            pre_stage="Pre",
            target_stage="Post",
            threshold=threshold,
        ),
    }

    for name, summary in summaries.items():
        th.save_rotated_sector_unit_legend(
            summary,
            output_dir / f"{name}_sector_legend.png",
            title=None,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export standalone sector legends for real transition data.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "transition_sector_legends",
    )
    parser.add_argument("--threshold", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    export_real_data_sector_legends(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
