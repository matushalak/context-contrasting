from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from context_contrasting.param_explore.common import (
    CONTEXT_LABELS,
    ExplorationSettings,
    generate_grid_parameter_sets,
    generate_sobol_parameter_sets,
    reference_transition_table,
    evaluate_parameter_sets,
)
from context_contrasting.param_explore.plotting import (
    plot_parameter_panels,
    plot_transition_space,
)


def _label_counts(frame: pd.DataFrame, *, image_type: str) -> pd.DataFrame:
    label_col = f"{image_type}_transition_label"
    grouped = (
        frame.groupby(["context_label", label_col], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["context_label", "count"], ascending=[True, False])
    )
    return grouped


def _save_outputs(frame: pd.DataFrame, *, settings: ExplorationSettings, method: str, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(save_dir / "samples.csv", index=False)

    reference_table = reference_transition_table(settings)
    reference_table.to_csv(save_dir / "reference_transition_points.csv", index=False)

    familiar_counts = _label_counts(frame, image_type="familiar")
    novel_counts = _label_counts(frame, image_type="novel")
    familiar_counts.to_csv(save_dir / "familiar_transition_counts.csv", index=False)
    novel_counts.to_csv(save_dir / "novel_transition_counts.csv", index=False)

    plot_dir = save_dir / "plots"
    plot_transition_space(frame, image_type="familiar", output_path=plot_dir / "familiar_transition_plane.png")
    plot_transition_space(frame, image_type="novel", output_path=plot_dir / "novel_transition_plane.png")
    plot_parameter_panels(frame, image_type="familiar", output_path=plot_dir / "familiar_parameter_panels.png")
    plot_parameter_panels(frame, image_type="novel", output_path=plot_dir / "novel_parameter_panels.png")

    summary = {
        "method": method,
        "n_samples_total": int(len(frame)),
        "n_weight_points": int(len(frame) // len(CONTEXT_LABELS)),
        "settings": settings.__dict__,
    }
    with (save_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def run_sampling_exploration(
    *,
    method: str,
    settings: ExplorationSettings,
    save_dir: str | Path,
) -> pd.DataFrame:
    if method == "grid":
        parameter_sets = generate_grid_parameter_sets(settings)
    elif method == "sobol":
        parameter_sets = generate_sobol_parameter_sets(settings.sobol_samples, settings=settings)
    else:
        raise ValueError(f"Unknown method {method!r}. Expected 'grid' or 'sobol'.")

    frame = evaluate_parameter_sets(
        parameter_sets,
        settings=settings,
        method=method,
    )
    _save_outputs(frame, settings=settings, method=method, save_dir=Path(save_dir))
    return frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore the transition plane over initial weights using grid or Sobol sampling."
    )
    parser.add_argument("--method", choices=["grid", "sobol"], required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--tail-window", type=int, default=25)
    parser.add_argument("--grid-levels", type=int, default=2)
    parser.add_argument("--sobol-samples", type=int, default=4096)
    parser.add_argument("--weight-floor", type=float, default=1e-4)
    parser.add_argument("--weight-ceiling", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-grid-points", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = ExplorationSettings(
        n_steps_per_phase=args.n_steps,
        n_trials=args.n_trials,
        tail_window=args.tail_window,
        grid_levels=args.grid_levels,
        sobol_samples=args.sobol_samples,
        weight_floor=args.weight_floor,
        weight_ceiling=args.weight_ceiling,
        random_seed=args.seed,
        n_jobs=args.n_jobs,
        max_grid_points=args.max_grid_points,
    )
    frame = run_sampling_exploration(
        method=args.method,
        settings=settings,
        save_dir=args.save_dir,
    )
    print(frame[[
        "method",
        "context_label",
        "familiar_transition_label",
        "novel_transition_label",
    ]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
