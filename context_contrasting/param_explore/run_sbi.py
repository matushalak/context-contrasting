from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sbi.inference import SNPE
from sbi.utils import BoxUniform

from context_contrasting.param_explore.common import (
    CONTEXT_MODES,
    CONTEXT_LABELS,
    ExplorationSettings,
    PARAMETER_ORDER,
    evaluate_parameter_sets,
    observation_tensor,
    reference_transition_table,
)
from context_contrasting.param_explore.plotting import (
    plot_parameter_panels,
    plot_transition_space,
)


def _label_counts(frame: pd.DataFrame, *, image_type: str) -> pd.DataFrame:
    label_col = f"{image_type}_transition_label"
    return (
        frame.groupby(["context_label", label_col], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["context_label", "count"], ascending=[True, False])
    )


def _params_from_theta(theta_row: torch.Tensor) -> dict[str, float]:
    return {
        name: float(10.0 ** theta_row[idx].item())
        for idx, name in enumerate(PARAMETER_ORDER)
    }


def _target_observation(
    *,
    settings: ExplorationSettings,
    target_mode: str,
    target_from_reference: str | None,
    target_values: list[float] | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if target_from_reference is not None:
        refs = reference_transition_table(settings)
        row = refs.loc[refs["reference_name"].eq(target_from_reference)]
        if row.empty:
            raise ValueError(f"Unknown reference target {target_from_reference!r}.")
        record = row.iloc[0]
        return observation_tensor(record, target_mode=target_mode), {
            "target_from_reference": target_from_reference,
            "target_mode": target_mode,
        }

    if not target_values:
        raise ValueError("Provide either --target-from-reference or --target-values.")

    expected_dims = {"familiar": 2, "novel": 2, "joint": 4}[target_mode]
    if len(target_values) != expected_dims:
        raise ValueError(
            f"target_mode={target_mode!r} expects {expected_dims} target values, got {len(target_values)}."
        )
    return torch.tensor(target_values, dtype=torch.float32), {
        "target_from_reference": None,
        "target_mode": target_mode,
        "target_values": target_values,
    }


def _simulate_training_data(
    *,
    settings: ExplorationSettings,
    num_simulations: int,
) -> tuple[torch.Tensor, pd.DataFrame]:
    low = math.log10(settings.weight_floor)
    high = math.log10(settings.weight_ceiling)
    theta = torch.rand((num_simulations, len(PARAMETER_ORDER)), dtype=torch.float32)
    theta = low + theta * (high - low)
    parameter_sets = [_params_from_theta(theta_row) for theta_row in theta]
    frame = evaluate_parameter_sets(parameter_sets, settings=settings, method="sbi_train")
    return theta, frame


def _train_single_posterior(
    *,
    theta: torch.Tensor,
    context_frame: pd.DataFrame,
    target_mode: str,
    settings: ExplorationSettings,
) -> Any:
    x = torch.stack(
        [
            observation_tensor(row, target_mode=target_mode)
            for _, row in context_frame.sort_values("weight_point_idx").iterrows()
        ],
        dim=0,
    )
    low = torch.full((len(PARAMETER_ORDER),), math.log10(settings.weight_floor), dtype=torch.float32)
    high = torch.full((len(PARAMETER_ORDER),), math.log10(settings.weight_ceiling), dtype=torch.float32)
    prior = BoxUniform(low=low, high=high)
    inference = SNPE(prior=prior)
    density_estimator = inference.append_simulations(theta, x).train()
    return inference.build_posterior(density_estimator)


def run_sbi_exploration(
    *,
    settings: ExplorationSettings,
    save_dir: str | Path,
    target_mode: str,
    target_from_reference: str | None,
    target_values: list[float] | None,
    num_simulations: int,
    posterior_samples: int,
) -> pd.DataFrame:
    theta, training_frame = _simulate_training_data(settings=settings, num_simulations=num_simulations)
    target_x, target_meta = _target_observation(
        settings=settings,
        target_mode=target_mode,
        target_from_reference=target_from_reference,
        target_values=target_values,
    )

    posterior_rows: list[pd.DataFrame] = []
    posterior_metadata: list[dict[str, Any]] = []

    for mode in CONTEXT_MODES:
        context_frame = training_frame.loc[
            training_frame["receives_context_familiar"].eq(mode[0])
            & training_frame["receives_context_novel"].eq(mode[1])
        ].copy()
        posterior = _train_single_posterior(
            theta=theta,
            context_frame=context_frame,
            target_mode=target_mode,
            settings=settings,
        )
        theta_samples = posterior.sample((posterior_samples,), x=target_x)
        parameter_sets = [_params_from_theta(theta_row) for theta_row in theta_samples]
        frame = evaluate_parameter_sets(parameter_sets, settings=settings, method="sbi_posterior")
        frame = frame.loc[
            frame["receives_context_familiar"].eq(mode[0])
            & frame["receives_context_novel"].eq(mode[1])
        ].copy()
        frame["posterior_context_label"] = CONTEXT_LABELS[mode]
        posterior_rows.append(frame)
        posterior_metadata.append(
            {
                "context_mode": CONTEXT_LABELS[mode],
                "n_training_samples": int(len(context_frame)),
                "n_posterior_samples": int(len(frame)),
            }
        )

    posterior_frame = pd.concat(posterior_rows, ignore_index=True)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    reference_transition_table(settings).to_csv(save_path / "reference_transition_points.csv", index=False)
    training_frame.to_csv(save_path / "training_samples.csv", index=False)
    posterior_frame.to_csv(save_path / "posterior_samples.csv", index=False)
    _label_counts(posterior_frame, image_type="familiar").to_csv(
        save_path / "familiar_transition_counts.csv",
        index=False,
    )
    _label_counts(posterior_frame, image_type="novel").to_csv(
        save_path / "novel_transition_counts.csv",
        index=False,
    )

    with (save_path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "settings": settings.__dict__,
                "target": target_meta,
                "num_simulations": num_simulations,
                "posterior_samples_per_context": posterior_samples,
                "contexts": posterior_metadata,
            },
            handle,
            indent=2,
        )

    plot_dir = save_path / "plots"
    plot_transition_space(posterior_frame, image_type="familiar", output_path=plot_dir / "familiar_transition_plane.png")
    plot_transition_space(posterior_frame, image_type="novel", output_path=plot_dir / "novel_transition_plane.png")
    plot_parameter_panels(posterior_frame, image_type="familiar", output_path=plot_dir / "familiar_parameter_panels.png")
    plot_parameter_panels(posterior_frame, image_type="novel", output_path=plot_dir / "novel_parameter_panels.png")
    return posterior_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore inverse parameter regions with SBI in the transition plane."
    )
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--target-mode", choices=["familiar", "novel", "joint"], default="joint")
    parser.add_argument("--target-from-reference", type=str, default=None)
    parser.add_argument("--target-values", nargs="*", type=float, default=None)
    parser.add_argument("--num-simulations", type=int, default=1024)
    parser.add_argument("--posterior-samples", type=int, default=400)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--tail-window", type=int, default=25)
    parser.add_argument("--weight-floor", type=float, default=1e-4)
    parser.add_argument("--weight-ceiling", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = ExplorationSettings(
        n_steps_per_phase=args.n_steps,
        n_trials=args.n_trials,
        tail_window=args.tail_window,
        weight_floor=args.weight_floor,
        weight_ceiling=args.weight_ceiling,
        random_seed=args.seed,
        n_jobs=args.n_jobs,
    )
    posterior_frame = run_sbi_exploration(
        settings=settings,
        save_dir=args.save_dir,
        target_mode=args.target_mode,
        target_from_reference=args.target_from_reference,
        target_values=args.target_values,
        num_simulations=args.num_simulations,
        posterior_samples=args.posterior_samples,
    )
    print(
        posterior_frame[[
            "context_label",
            "familiar_transition_label",
            "novel_transition_label",
        ]].head(12).to_string(index=False)
    )


if __name__ == "__main__":
    main()
