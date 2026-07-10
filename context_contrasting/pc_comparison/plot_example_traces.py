from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from context_contrasting.paper.model_scatter import (
    _append_post_stimulus_iti,
    _build_model_scatter_test_stimuli,
    _panel_step_window,
)
from context_contrasting.paper.visualize_s import visualize_transition_panel, wide_to_long
from context_contrasting.pc_comparison.pc_templates import template_trace_series
from context_contrasting.pc_comparison.run_pc_comparison import DEFAULT_OUTPUT_DIR, PAPER_DONE_FINAL_FIX


TRACE_CONDITIONS = ["familiar_1", "familiar_2", "novel"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PC-template traces with paper.visualize_s.")
    parser.add_argument("--paper-output-dir", type=Path, default=PAPER_DONE_FINAL_FIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "trace_examples")
    parser.add_argument("--pc-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-format", choices=("png", "svg", "eps"), default="png")
    parser.add_argument("--extra-format", choices=("png", "svg", "eps", "none"), default="svg")
    return parser.parse_args()


def _formats(args: argparse.Namespace) -> tuple[str, ...]:
    formats = [args.image_format]
    if args.extra_format != "none" and args.extra_format not in formats:
        formats.append(args.extra_format)
    return tuple(formats)


def _select_examples(circuit: str, summaries_dir: Path) -> list[tuple[str, int]]:
    familiar = pd.read_csv(summaries_dir / f"{circuit.lower()}_familiar_summary.csv")
    novel = pd.read_csv(summaries_dir / f"{circuit.lower()}_novel_summary.csv")
    if circuit == "PPE":
        candidates = [
            familiar.sort_values(["dNO", "sample_order"], ascending=[True, True]),
            novel.sort_values(["dNO", "sample_order"], ascending=[False, True]),
        ]
    else:
        mixed = familiar.loc[(familiar["dNO"] > 0.25) & (familiar["dO"] > 0.25)]
        candidates = [
            familiar.sort_values(["dO", "sample_order"], ascending=[False, True]),
            familiar.sort_values(["dO", "sample_order"], ascending=[True, True]),
            mixed.sort_values(["dNorm", "sample_order"], ascending=[False, True]) if not mixed.empty else familiar,
        ]

    examples = []
    used: set[int] = set()
    for frame in candidates:
        if frame.empty:
            continue
        for _, row in frame.iterrows():
            neuron_idx = int(row["neuron_idx"])
            if neuron_idx in used:
                continue
            used.add(neuron_idx)
            label = (
                f"{circuit} #{neuron_idx}\n"
                f"{row['transition']}\n"
                f"{row['RotatedSector']} dNO={float(row['dNO']):.2f} dO={float(row['dO']):.2f}"
            )
            examples.append((label, neuron_idx))
            break
    return examples[:3]


def _series_for_response(response: float, *, n_steps_per_phase: int, test_trials: int) -> np.ndarray:
    post = np.zeros(n_steps_per_phase // 2, dtype=float)
    return template_trace_series(
        response=response,
        n_steps_per_phase=n_steps_per_phase,
        n_trials=test_trials,
        post_steps=len(post),
        seed=int(abs(response) * 10_000) + 17,
    )


def _frame_for_trace(
    *,
    response: float,
    condition_name: str,
    phase: str,
    trace: str,
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    n_steps_per_phase: int,
    test_trials: int,
) -> pd.DataFrame:
    x_full, c_full = stimuli[condition_name]
    x_values = torch.zeros_like(x_full) if trace == "occlusion" else x_full
    y = _series_for_response(response, n_steps_per_phase=n_steps_per_phase, test_trials=test_trials)
    n = min(len(y), x_values.shape[0], c_full.shape[0])
    rows = []
    for step in range(n):
        row = {"step": step, "y": float(y[step])}
        for idx in range(x_values.shape[1]):
            row[f"x_{idx}"] = float(x_values[step, idx])
            row[f"w_ff_{idx}"] = 0.0
        for idx in range(c_full.shape[1]):
            row[f"c_{idx}"] = float(c_full[step, idx])
            row[f"w_fb_{idx}"] = 0.0
        row["p_0"] = 0.0
        row["w_lat_0"] = 0.0
        row["w_pv_lat_0"] = 0.0
        for idx in range(x_values.shape[1]):
            row[f"W_pv_0_{idx}"] = 0.0
        rows.append(row)
    df = pd.DataFrame(rows)
    df["condition"] = f"{trace}_{condition_name}_{phase}"
    df["seed"] = 7151
    df["experiment_series"] = "pc_templates"
    return df


def _long_df_for_example(
    responses: pd.DataFrame,
    *,
    neuron_idx: int,
    n_steps_per_phase: int,
    test_trials: int,
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> pd.DataFrame:
    frames = []
    subset = responses.loc[responses["sample_global_idx"].astype(int).eq(neuron_idx)]
    for row in subset.itertuples(index=False):
        frames.append(
            _frame_for_trace(
                response=float(row.response),
                condition_name=str(row.condition),
                phase=str(row.phase),
                trace=str(row.trace),
                stimuli=stimuli,
                n_steps_per_phase=n_steps_per_phase,
                test_trials=test_trials,
            )
        )
    long_df = wide_to_long(pd.concat(frames, ignore_index=True))
    long_df = long_df.loc[long_df["experiment_phase"].isin(["naive", "expert"])].copy()
    long_df["_zscore_std_floor"] = 1.0
    return long_df


def _plot_circuit(
    circuit: str,
    *,
    metadata: dict,
    output_dir: Path,
    pc_output_dir: Path,
    formats: tuple[str, ...],
) -> None:
    n_steps_per_phase = int(metadata.get("n_steps_per_phase", 400))
    test_trials = int(metadata.get("test_trials", 5))
    responses = pd.read_csv(pc_output_dir / circuit.lower() / f"{circuit.lower()}_sample_responses.csv")
    stimuli = _append_post_stimulus_iti(
        _build_model_scatter_test_stimuli(n_steps_per_phase=n_steps_per_phase, n_trials=test_trials),
        n_steps_per_phase=n_steps_per_phase,
    )
    long_dfs = {
        label: _long_df_for_example(
            responses,
            neuron_idx=neuron_idx,
            n_steps_per_phase=n_steps_per_phase,
            test_trials=test_trials,
            stimuli=stimuli,
        )
        for label, neuron_idx in _select_examples(circuit, pc_output_dir / "summaries")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        visualize_transition_panel(
            long_dfs,
            stimuli,
            save_path=str(output_dir),
            name=f"{circuit.lower()}_representative_traces",
            image_mode="both",
            include_novel_image=True,
            transition_order=list(long_dfs),
            transition_labels={label: label for label in long_dfs},
            trace_types=("full", "occlusion"),
            step_window=_panel_step_window(n_steps_per_phase, test_trials),
            save_in_transition_subdir=False,
            save_csv=True,
            zscore_activity=False,
            image_format=fmt,
        )


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.paper_output_dir / "metadata.json").read_text())
    for circuit in ("PPE", "NPE"):
        _plot_circuit(
            circuit,
            metadata=metadata,
            output_dir=args.output_dir,
            pc_output_dir=args.pc_output_dir,
            formats=_formats(args),
        )


if __name__ == "__main__":
    main()
