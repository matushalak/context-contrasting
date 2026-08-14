from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from context_contrasting.pc_comparison.pc_convergence import (
    convergence_rows,
    convergence_summary,
    find_minimum_learning_rate,
)
from context_contrasting.pc_comparison.pc_templates import (
    DEFAULT_CONVERGENCE_TOLERANCE,
    LEARNING_RATE_REFERENCE_STEPS,
    parameter_space_metadata,
    sample_shared_pc_configs,
)


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs" / "pc_learning_rate_calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate one matched PPE/NPE learning rate and verify the full paper horizon."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7151)
    parser.add_argument("--training-trials", type=int, default=7)
    parser.add_argument("--training-stimulus-order", choices=("fixed", "randomized"), default="randomized")
    parser.add_argument("--quick-steps", type=int, default=100)
    parser.add_argument("--full-steps", type=int, default=LEARNING_RATE_REFERENCE_STEPS)
    parser.add_argument(
        "--prediction-error-tolerance",
        type=float,
        default=DEFAULT_CONVERGENCE_TOLERANCE,
    )
    parser.add_argument("--rate-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-learning-rate", type=float, default=1.0)
    return parser.parse_args()


def _plot_rate_scan(
    configs,
    *,
    args: argparse.Namespace,
    quick,
    full,
) -> None:
    rates = np.geomspace(1e-4, float(args.max_learning_rate), 100)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for steps, label, color in (
        (int(args.quick_steps), f"Quick ({args.quick_steps} steps)", "#2878b5"),
        (int(args.full_steps), f"Full ({args.full_steps} steps)", "#d1495b"),
    ):
        errors = [
            convergence_summary(
                configs,
                reference_learning_rate=float(rate),
                n_steps_per_phase=steps,
                training_trials=int(args.training_trials),
                training_stimulus_order=str(args.training_stimulus_order),
                seed=int(args.seed),
                tolerance=float(args.prediction_error_tolerance),
            ).max_abs_prediction_error
            for rate in rates
        ]
        ax.plot(rates, errors, color=color, lw=1.8, label=label)
    ax.axhline(
        float(args.prediction_error_tolerance),
        color="0.2",
        ls="--",
        lw=1.0,
        label="Convergence tolerance",
    )
    ax.scatter(
        [quick.reference_learning_rate, full.reference_learning_rate],
        [quick.max_abs_prediction_error, full.max_abs_prediction_error],
        c=["#2878b5", "#d1495b"],
        edgecolors="white",
        linewidths=0.7,
        zorder=4,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Learning rate at 400-step reference")
    ax.set_ylabel("Maximum absolute prediction error")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(args.output_dir / f"learning_rate_convergence.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_full_population(rows, *, args: argparse.Namespace) -> None:
    values = np.sort(rows["max_abs_prediction_error"].to_numpy(dtype=float))[::-1]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.scatter(np.arange(1, len(values) + 1), values, s=12, color="#d1495b", alpha=0.8)
    ax.axhline(
        float(args.prediction_error_tolerance),
        color="0.2",
        ls="--",
        lw=1.0,
        label="Convergence tolerance",
    )
    ax.set_xlabel("Sampled parameter row, sorted")
    ax.set_ylabel("Final maximum absolute prediction error")
    ax.set_ylim(bottom=0.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(args.output_dir / f"full_protocol_error_distribution.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configs = sample_shared_pc_configs(
        n_samples=int(args.n_samples),
        seed=int(args.seed),
        n_steps_per_phase=int(args.full_steps),
        learning_rate=0.0,
    )
    common = {
        "configs": configs,
        "training_trials": int(args.training_trials),
        "training_stimulus_order": str(args.training_stimulus_order),
        "seed": int(args.seed),
        "tolerance": float(args.prediction_error_tolerance),
        "upper": float(args.max_learning_rate),
        "rate_tolerance": float(args.rate_tolerance),
    }
    quick = find_minimum_learning_rate(n_steps_per_phase=int(args.quick_steps), **common)
    full = find_minimum_learning_rate(n_steps_per_phase=int(args.full_steps), **common)
    quick_rate_at_full = convergence_summary(
        configs,
        reference_learning_rate=quick.reference_learning_rate,
        n_steps_per_phase=int(args.full_steps),
        training_trials=int(args.training_trials),
        training_stimulus_order=str(args.training_stimulus_order),
        seed=int(args.seed),
        tolerance=float(args.prediction_error_tolerance),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = convergence_rows(
        configs,
        reference_learning_rate=full.reference_learning_rate,
        n_steps_per_phase=int(args.full_steps),
        training_trials=int(args.training_trials),
        training_stimulus_order=str(args.training_stimulus_order),
        seed=int(args.seed),
        tolerance=float(args.prediction_error_tolerance),
    )
    rows.to_csv(args.output_dir / "full_protocol_convergence.csv", index=False)
    _plot_rate_scan(configs, args=args, quick=quick, full=full)
    _plot_full_population(rows, args=args)
    metadata = {
        "definition": (
            "Convergence means every sampled cell has absolute steady-state signed prediction error "
            "at or below prediction_error_tolerance on both familiar features after training."
        ),
        "symmetry": "The same result applies to PPE and NPE because matched familiar x and c swap roles exactly.",
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "training_trials_per_familiar_image": int(args.training_trials),
        "training_stimulus_order": str(args.training_stimulus_order),
        "prediction_error_tolerance": float(args.prediction_error_tolerance),
        "rate_search_tolerance": float(args.rate_tolerance),
        "learning_rate_reference_steps": LEARNING_RATE_REFERENCE_STEPS,
        "parameter_space": parameter_space_metadata(),
        "quick_search": asdict(quick) | {"converged": quick.converged},
        "quick_rate_verified_at_full_horizon": asdict(quick_rate_at_full)
        | {"converged": quick_rate_at_full.converged},
        "full_search": asdict(full) | {"converged": full.converged},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
