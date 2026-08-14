from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from context_contrasting.paper import model_scatter as paper_scatter
from context_contrasting.paper import transition_templates
from context_contrasting.paper import transitions_helpers as th
from context_contrasting.paper.visualize_s import visualize_transition_panel
from context_contrasting.pc_comparison.pc_neuron import CorrectPCneuron
from context_contrasting.pc_comparison.run_pc_comparison import (
    DEFAULT_OUTPUT_DIR,
    PAPER_DONE_FINAL_FIX,
    _model_params_from_row,
    _summary_with_transition,
    _wide_table,
)


SECTOR_TRACE_ORDER = ("+NO axis", "+O axis", "-NO axis", "-O axis", "small ∆")
SECTOR_TRACE_LABELS = {
    "+NO axis": "+NO",
    "+O axis": "+O",
    "-NO axis": "-NO",
    "-O axis": "-O",
    "small ∆": "small",
}
SECTOR_MODES = ("sector-average", "sector-per-image")
DIAGONAL_HALF_WIDTH_RAD = np.pi / 8.0
DIAGONAL_BY_GROUP = {
    "familiar": ("minus_no_plus_o", "-NO/+O", 3.0 * np.pi / 4.0),
    "novel": ("plus_no_plus_o", "+NO/+O", np.pi / 4.0),
}
GROUP_CONDITIONS = {
    "familiar": ["familiar_1", "familiar_2"],
    "novel": ["novel"],
}
CONDITION_BY_IMAGE = {
    "familiar": {
        (1, 1): "familiar_1",
        (2, 2): "familiar_2",
    },
    "novel": {
        (3, 1): "novel",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot real PC-template traces with paper.visualize_s.")
    parser.add_argument("--paper-output-dir", type=Path, default=PAPER_DONE_FINAL_FIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "trace_examples")
    parser.add_argument("--pc-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-format", choices=("png", "svg", "eps"), default="png")
    parser.add_argument("--extra-format", choices=("png", "svg", "eps", "none"), default="svg")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--circuits", nargs="+", choices=("PPE", "NPE"), default=("PPE", "NPE"))
    parser.add_argument("--max-sector-traces-per-sector", type=int, default=8)
    parser.add_argument("--skip-representatives", action="store_true")
    parser.add_argument("--skip-sector-averages", action="store_true")
    return parser.parse_args()


def _formats(args: argparse.Namespace) -> tuple[str, ...]:
    formats = [args.image_format]
    if args.extra_format != "none" and args.extra_format not in formats:
        formats.append(args.extra_format)
    return tuple(formats)


def _worker_count(n_jobs: int, n_tasks: int) -> int:
    if n_tasks <= 1:
        return 1
    if n_jobs < 0:
        n_jobs = max((os.cpu_count() or 1) + 1 + n_jobs, 1)
    return max(1, min(int(n_jobs), n_tasks))


def _select_examples(circuit: str, summaries_dir: Path) -> list[tuple[str, int]]:
    familiar = pd.read_csv(summaries_dir / f"{circuit.lower()}_familiar_summary.csv")
    novel = pd.read_csv(summaries_dir / f"{circuit.lower()}_novel_summary.csv")
    if circuit == "PPE":
        candidates = [
            familiar.sort_values(["dNO", "sample_order"], ascending=[True, True]),
            familiar.sort_values(["dNO", "sample_order"], ascending=[False, True]),
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


def _stimuli(metadata: dict[str, Any]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    n_steps_per_phase = int(metadata.get("n_steps_per_phase", 400))
    test_trials = int(metadata.get("test_trials", 5))
    return paper_scatter._append_post_stimulus_iti(
        paper_scatter._build_model_scatter_test_stimuli(
            n_steps_per_phase=n_steps_per_phase,
            n_trials=test_trials,
        ),
        n_steps_per_phase=n_steps_per_phase,
    )


def _training_stimuli(metadata: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    return paper_scatter._build_model_scatter_training_stimuli(
        n_steps_per_phase=int(metadata.get("n_steps_per_phase", 400)),
        n_trials=int(metadata.get("training_trials", 7)),
        order=str(metadata.get("training_stimulus_order", "randomized")),
        seed=int(metadata.get("seed", 7151)),
    )


def _cell_floor(row: pd.Series, metadata: dict[str, Any]) -> float:
    return max(
        float(metadata.get("zscore_std_floor", 0.04)),
        transition_templates.BASELINE_STD_SCALE * float(row.get("baseline_drive_sigma", 0.0)),
    )


def _run_compact_phase(
    model: CorrectPCneuron,
    x_phase: torch.Tensor,
    c_phase: torch.Tensor,
    *,
    update: bool,
    reset_rates: bool = True,
) -> pd.DataFrame:
    if reset_rates:
        model._reset_state()
    rows = []
    with torch.no_grad():
        for step in range(x_phase.shape[0]):
            x, y_t, y_next, p, c = model(x_phase[step], c_phase[step])
            if update:
                model.update(x, y_t, y_next, p, c)
            rows.append({"step": step, "y": float(y_next.item())})
    return pd.DataFrame(rows)


def _trace_frames(
    model: CorrectPCneuron,
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    phase: str,
    cell_floor: float,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for condition, (x_full, c_full) in stimuli.items():
        occluded_x = torch.zeros_like(x_full)
        full = _run_compact_phase(model, x_full, c_full, update=False)
        full["condition"] = condition
        full["image_type"] = "full"
        full["experiment_phase"] = phase
        full["_zscore_std_floor"] = cell_floor
        frames.append(full)

        occluded = _run_compact_phase(model, occluded_x, c_full, update=False)
        occluded["condition"] = condition
        occluded["image_type"] = "occlusion"
        occluded["experiment_phase"] = phase
        occluded["_zscore_std_floor"] = cell_floor
        frames.append(occluded)
    return frames


def _long_df_for_config(
    row: pd.Series,
    *,
    circuit: str,
    metadata: dict[str, Any],
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    training_stimuli: tuple[torch.Tensor, torch.Tensor],
) -> pd.DataFrame:
    model = CorrectPCneuron(_model_params_from_row(row, circuit=circuit))
    cell_floor = _cell_floor(row, metadata)
    probe_noise_state = model.get_noise_state()
    frames = _trace_frames(model, stimuli, phase="naive", cell_floor=cell_floor)
    _run_compact_phase(model, training_stimuli[0], training_stimuli[1], update=True)
    model.set_noise_state(probe_noise_state)
    frames.extend(_trace_frames(model, stimuli, phase="expert", cell_floor=cell_floor))
    return pd.concat(frames, ignore_index=True)


def _long_df_for_config_dict(
    row_dict: dict[str, Any],
    *,
    circuit: str,
    metadata: dict[str, Any],
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    training_stimuli: tuple[torch.Tensor, torch.Tensor],
) -> pd.DataFrame:
    return _long_df_for_config(
        pd.Series(row_dict),
        circuit=circuit,
        metadata=metadata,
        stimuli=stimuli,
        training_stimuli=training_stimuli,
    )


def _config_by_neuron(configs: pd.DataFrame) -> dict[int, pd.Series]:
    return {int(row["sample_global_idx"]): row for _, row in configs.iterrows()}


def _render_neuron_traces(
    neuron_ids: list[int],
    *,
    configs_by_neuron: dict[int, pd.Series],
    circuit: str,
    metadata: dict[str, Any],
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    training_stimuli: tuple[torch.Tensor, torch.Tensor],
    n_jobs: int,
    progress_label: str,
) -> dict[int, pd.DataFrame]:
    workers = _worker_count(n_jobs, len(neuron_ids))
    if workers <= 1:
        rendered = {}
        for pos, neuron_idx in enumerate(neuron_ids, start=1):
            rendered[neuron_idx] = _long_df_for_config(
                configs_by_neuron[neuron_idx],
                circuit=circuit,
                metadata=metadata,
                stimuli=stimuli,
                training_stimuli=training_stimuli,
            )
            if pos % 5 == 0 or pos == len(neuron_ids):
                print(f"{progress_label}: rendered {pos}/{len(neuron_ids)} traces", flush=True)
        return rendered

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    rendered: dict[int, pd.DataFrame] = {}
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _long_df_for_config_dict,
                    configs_by_neuron[neuron_idx].to_dict(),
                    circuit=circuit,
                    metadata=metadata,
                    stimuli=stimuli,
                    training_stimuli=training_stimuli,
                ): neuron_idx
                for neuron_idx in neuron_ids
            }
            for pos, future in enumerate(as_completed(futures), start=1):
                neuron_idx = futures[future]
                rendered[neuron_idx] = future.result()
                if pos % 5 == 0 or pos == len(neuron_ids):
                    print(f"{progress_label}: rendered {pos}/{len(neuron_ids)} traces", flush=True)
    finally:
        torch.set_num_threads(previous_threads)
    return rendered


def _plot_circuit(
    circuit: str,
    *,
    metadata: dict[str, Any],
    output_dir: Path,
    pc_output_dir: Path,
    formats: tuple[str, ...],
    n_jobs: int,
) -> None:
    configs = pd.read_csv(pc_output_dir / circuit.lower() / f"{circuit.lower()}_final_parameters.csv")
    configs_by_neuron = _config_by_neuron(configs)
    stimuli = _stimuli(metadata)
    training = _training_stimuli(metadata)
    examples = [(label, neuron_idx) for label, neuron_idx in _select_examples(circuit, pc_output_dir / "summaries") if neuron_idx in configs_by_neuron]
    rendered = _render_neuron_traces(
        [neuron_idx for _, neuron_idx in examples],
        configs_by_neuron=configs_by_neuron,
        circuit=circuit,
        metadata=metadata,
        stimuli=stimuli,
        training_stimuli=training,
        n_jobs=n_jobs,
        progress_label=f"{circuit} representatives",
    )
    long_dfs = {label: rendered[neuron_idx] for label, neuron_idx in examples}
    if not long_dfs:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    n_steps_per_phase = int(metadata.get("n_steps_per_phase", 400))
    test_trials = int(metadata.get("test_trials", 5))
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
            step_window=paper_scatter._panel_step_window(n_steps_per_phase, test_trials),
            save_in_transition_subdir=False,
            save_csv=True,
            zscore_activity=True,
            image_format=fmt,
            condition_title_size=22,
            phase_title_size=26,
            panel_top=0.80,
            phase_title_y=0.94,
        )


def _sector_members(summary: pd.DataFrame, *, max_per_sector: int) -> dict[str, list[int]]:
    observed = set(summary["RotatedSector"].dropna().astype(str))
    members: dict[str, list[int]] = {}
    for sector in SECTOR_TRACE_ORDER:
        if sector not in observed:
            continue
        rows = summary.loc[summary["RotatedSector"].astype(str).eq(sector)].copy()
        if sector != "small ∆":
            rows = rows.sort_values(["dNorm", "sample_order"], ascending=[False, True])
        else:
            rows = rows.sort_values(["dNorm", "sample_order"], ascending=[True, True])
        members[sector] = [int(neuron_idx) for neuron_idx in rows["neuron_idx"].head(max_per_sector)]
    return members


def _angle_distance(angle: pd.Series, target: float) -> pd.Series:
    values = angle.to_numpy(dtype=float)
    return pd.Series(
        np.abs(np.arctan2(np.sin(values - target), np.cos(values - target))),
        index=angle.index,
    )


def _diagonal_members(summary: pd.DataFrame, *, group: str, threshold: float) -> dict[str, list[int]]:
    diagonal_key, _diagonal_label, diagonal_angle = DIAGONAL_BY_GROUP[group]
    mask = _angle_distance(summary["Angle"], float(diagonal_angle)).le(DIAGONAL_HALF_WIDTH_RAD)
    if "dNorm" in summary:
        mask &= summary["dNorm"].astype(float).gt(threshold)
    rows = summary.loc[mask].sort_values(["dNorm", "sample_order"], ascending=[False, True])
    return {diagonal_key: [int(neuron_idx) for neuron_idx in rows["neuron_idx"]]}


def _average_member_traces(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby(["condition", "image_type", "experiment_phase", "step"], observed=True, as_index=False)
        .agg(y=("y", "mean"), _zscore_std_floor=("_zscore_std_floor", "max"))
        .sort_values(["condition", "image_type", "experiment_phase", "step"])
        .reset_index(drop=True)
    )


def _per_image_summaries(
    transition_table: pd.DataFrame,
    configs: pd.DataFrame,
    *,
    group: str,
    threshold: float,
) -> pd.DataFrame:
    wide = _wide_table(transition_table)
    group_wide = wide.loc[wide["image_group"].eq(group)].copy()
    summaries: list[pd.DataFrame] = []
    for image in (
        group_wide[["image_idx_original", "image_idx_within_group"]]
        .drop_duplicates()
        .sort_values(["image_idx_within_group", "image_idx_original"])
        .itertuples(index=False)
    ):
        image_wide = group_wide.loc[
            group_wide["image_idx_original"].eq(image.image_idx_original)
            & group_wide["image_idx_within_group"].eq(image.image_idx_within_group)
        ].copy()
        summary = th.build_mean_summary(
            image_wide,
            image_group=group,
            pre_stage="Naive",
            target_stage="Expert",
            threshold=threshold,
        )
        summary = _summary_with_transition(summary, configs)
        condition = CONDITION_BY_IMAGE[group].get((int(image.image_idx_original), int(image.image_idx_within_group)))
        if condition is None:
            continue
        summary["condition"] = condition
        summaries.append(summary)
    if not summaries:
        return pd.DataFrame()
    return pd.concat(summaries, ignore_index=True)


def _members_by_condition(
    summary: pd.DataFrame,
    *,
    selected_conditions: list[str],
    sector_mode: str,
    max_per_sector: int,
) -> dict[str, dict[str, list[int]]]:
    if sector_mode == "sector-average":
        members = _sector_members(summary, max_per_sector=max_per_sector)
        return {condition: members for condition in selected_conditions}
    return {
        condition: _sector_members(
            summary.loc[summary["condition"].eq(condition)].copy(),
            max_per_sector=max_per_sector,
        )
        for condition in selected_conditions
    }


def _diagonal_members_by_condition(
    summary: pd.DataFrame,
    *,
    group: str,
    selected_conditions: list[str],
    sector_mode: str,
    threshold: float,
) -> dict[str, dict[str, list[int]]]:
    if sector_mode == "sector-average":
        members = _diagonal_members(summary, group=group, threshold=threshold)
        return {condition: members for condition in selected_conditions}
    return {
        condition: _diagonal_members(
            summary.loc[summary["condition"].eq(condition)].copy(),
            group=group,
            threshold=threshold,
        )
        for condition in selected_conditions
    }


def _needed_neurons(members_by_condition: dict[str, dict[str, list[int]]]) -> set[int]:
    return {
        neuron_idx
        for members_by_sector in members_by_condition.values()
        for neuron_ids in members_by_sector.values()
        for neuron_idx in neuron_ids
    }


def _sector_average_long_dfs(
    long_df_cache: dict[int, pd.DataFrame],
    members_by_condition: dict[str, dict[str, list[int]]],
    *,
    selected_conditions: list[str],
    pooled: bool,
    trace_order: tuple[str, ...] = SECTOR_TRACE_ORDER,
) -> dict[str, pd.DataFrame]:
    long_dfs: dict[str, pd.DataFrame] = {}
    for sector in trace_order:
        frames: list[pd.DataFrame] = []
        for condition in selected_conditions:
            neuron_ids = members_by_condition.get(condition, {}).get(sector, [])
            for neuron_idx in neuron_ids:
                if neuron_idx not in long_df_cache:
                    continue
                frame = long_df_cache[neuron_idx].loc[long_df_cache[neuron_idx]["condition"].eq(condition)].copy()
                if frame.empty:
                    continue
                if pooled:
                    frame["condition"] = "pooled"
                frames.append(frame)
        if frames:
            long_dfs[sector] = _average_member_traces(frames)
    return long_dfs


def _plot_sector_average_circuit(
    circuit: str,
    *,
    metadata: dict[str, Any],
    output_dir: Path,
    pc_output_dir: Path,
    formats: tuple[str, ...],
    n_jobs: int,
    max_sector_traces_per_sector: int,
) -> None:
    configs = pd.read_csv(pc_output_dir / circuit.lower() / f"{circuit.lower()}_final_parameters.csv")
    configs_by_neuron = _config_by_neuron(configs)
    stimuli = _stimuli(metadata)
    training = _training_stimuli(metadata)
    n_steps_per_phase = int(metadata.get("n_steps_per_phase", 400))
    test_trials = int(metadata.get("test_trials", 5))
    threshold = float(metadata.get("sector_threshold", 0.3))
    summaries_dir = pc_output_dir / "summaries"
    transition_table = pd.read_csv(pc_output_dir / circuit.lower() / f"{circuit.lower()}_transition_table.csv")
    summaries_by_mode_group: dict[str, dict[str, pd.DataFrame]] = {
        "sector-average": {
            group: pd.read_csv(summaries_dir / f"{circuit.lower()}_{group}_summary.csv")
            for group in ("familiar", "novel")
        },
        "sector-per-image": {
            group: _per_image_summaries(
                transition_table,
                configs,
                group=group,
                threshold=threshold,
            )
            for group in ("familiar", "novel")
        },
    }
    members_lookup: dict[tuple[str, str], dict[str, dict[str, list[int]]]] = {}
    diagonal_members_lookup: dict[tuple[str, str], dict[str, dict[str, list[int]]]] = {}
    needed: set[int] = set()
    for sector_mode in SECTOR_MODES:
        for group in ("familiar", "novel"):
            selected_conditions = GROUP_CONDITIONS[group]
            members = _members_by_condition(
                summaries_by_mode_group[sector_mode][group],
                selected_conditions=selected_conditions,
                sector_mode=sector_mode,
                max_per_sector=max_sector_traces_per_sector,
            )
            members_lookup[(sector_mode, group)] = members
            needed.update(_needed_neurons(members))
            diagonal_members = _diagonal_members_by_condition(
                summaries_by_mode_group[sector_mode][group],
                group=group,
                selected_conditions=selected_conditions,
                sector_mode=sector_mode,
                threshold=threshold,
            )
            diagonal_members_lookup[(sector_mode, group)] = diagonal_members
            needed.update(_needed_neurons(diagonal_members))
    needed_neurons = sorted(neuron_idx for neuron_idx in needed if neuron_idx in configs_by_neuron)
    long_df_cache = _render_neuron_traces(
        needed_neurons,
        configs_by_neuron=configs_by_neuron,
        circuit=circuit,
        metadata=metadata,
        stimuli=stimuli,
        training_stimuli=training,
        n_jobs=n_jobs,
        progress_label=f"{circuit} sector averages",
    )
    for group in ("familiar", "novel"):
        legacy_dir = output_dir / circuit.lower() / group
        if legacy_dir.exists():
            for legacy_file in legacy_dir.glob(f"{circuit.lower()}_sector_average_*"):
                if legacy_file.is_file():
                    legacy_file.unlink()

    for sector_mode in SECTOR_MODES:
        name_prefix = "sector_per_image" if sector_mode == "sector-per-image" else "sector_average"
        for group in ("familiar", "novel"):
            selected_conditions = GROUP_CONDITIONS[group]
            members_by_condition = members_lookup[(sector_mode, group)]
            long_dfs = _sector_average_long_dfs(
                long_df_cache,
                members_by_condition,
                selected_conditions=selected_conditions,
                pooled=False,
            )
            pooled_long_dfs = _sector_average_long_dfs(
                long_df_cache,
                members_by_condition,
                selected_conditions=selected_conditions,
                pooled=True,
            )
            group_output_dir = output_dir / sector_mode / circuit.lower() / group
            group_output_dir.mkdir(parents=True, exist_ok=True)
            for fmt in formats:
                if long_dfs:
                    visualize_transition_panel(
                        long_dfs,
                        stimuli,
                        save_path=str(group_output_dir),
                        name=f"{circuit.lower()}_{name_prefix}_{group}_examples",
                        image_mode=group,
                        include_novel_image=(group == "novel"),
                        transition_order=list(long_dfs),
                        transition_labels={sector: SECTOR_TRACE_LABELS.get(sector, sector) for sector in long_dfs},
                        trace_types=("full", "occlusion"),
                        step_window=paper_scatter._panel_step_window(n_steps_per_phase, test_trials),
                        save_in_transition_subdir=False,
                        save_csv=True,
                        zscore_activity=True,
                        image_format=fmt,
                        condition_title_size=22,
                        phase_title_size=26,
                        panel_top=0.80,
                        phase_title_y=0.94,
                        match_sector_export_style=True,
                    )

            diagonal_key, diagonal_label, _diagonal_angle = DIAGONAL_BY_GROUP[group]
            diagonal_order = (diagonal_key,)
            diagonal_members = diagonal_members_lookup[(sector_mode, group)]
            diagonal_long_dfs = _sector_average_long_dfs(
                long_df_cache,
                diagonal_members,
                selected_conditions=selected_conditions,
                pooled=False,
                trace_order=diagonal_order,
            )
            diagonal_pooled_long_dfs = _sector_average_long_dfs(
                long_df_cache,
                diagonal_members,
                selected_conditions=selected_conditions,
                pooled=True,
                trace_order=diagonal_order,
            )
            diagonal_output_dir = output_dir / "diagonal-average" / sector_mode / circuit.lower() / group
            diagonal_output_dir.mkdir(parents=True, exist_ok=True)
            diagonal_name_prefix = (
                f"diagonal_per_image_{diagonal_key}"
                if sector_mode == "sector-per-image"
                else f"diagonal_average_{diagonal_key}"
            )
            for fmt in formats:
                if diagonal_long_dfs:
                    visualize_transition_panel(
                        diagonal_long_dfs,
                        stimuli,
                        save_path=str(diagonal_output_dir),
                        name=f"{circuit.lower()}_{diagonal_name_prefix}_{group}_examples",
                        image_mode=group,
                        include_novel_image=(group == "novel"),
                        transition_order=list(diagonal_long_dfs),
                        transition_labels={diagonal_key: diagonal_label},
                        trace_types=("full", "occlusion"),
                        step_window=paper_scatter._panel_step_window(n_steps_per_phase, test_trials),
                        save_in_transition_subdir=False,
                        save_csv=True,
                        zscore_activity=True,
                        image_format=fmt,
                        condition_title_size=22,
                        phase_title_size=26,
                        panel_top=0.80,
                        phase_title_y=0.94,
                        match_sector_export_style=True,
                    )
                if diagonal_pooled_long_dfs:
                    pooled_stimuli = {"pooled": stimuli[selected_conditions[0]]}
                    visualize_transition_panel(
                        diagonal_pooled_long_dfs,
                        pooled_stimuli,
                        save_path=str(diagonal_output_dir),
                        name=f"{circuit.lower()}_{diagonal_name_prefix}_{group}_pooled_examples",
                        image_mode=None,
                        include_novel_image=False,
                        transition_order=list(diagonal_pooled_long_dfs),
                        transition_labels={diagonal_key: diagonal_label},
                        trace_types=("full", "occlusion"),
                        step_window=paper_scatter._panel_step_window(n_steps_per_phase, test_trials),
                        save_in_transition_subdir=False,
                        save_csv=True,
                        zscore_activity=True,
                        image_format=fmt,
                        condition_title_size=22,
                        phase_title_size=26,
                        panel_top=0.80,
                        phase_title_y=0.94,
                        match_sector_export_style=True,
                    )
                if pooled_long_dfs:
                    pooled_stimuli = {"pooled": stimuli[selected_conditions[0]]}
                    visualize_transition_panel(
                        pooled_long_dfs,
                        pooled_stimuli,
                        save_path=str(group_output_dir),
                        name=f"{circuit.lower()}_{name_prefix}_{group}_pooled_examples",
                        image_mode=None,
                        include_novel_image=False,
                        transition_order=list(pooled_long_dfs),
                        transition_labels={sector: SECTOR_TRACE_LABELS.get(sector, sector) for sector in pooled_long_dfs},
                        trace_types=("full", "occlusion"),
                        step_window=paper_scatter._panel_step_window(n_steps_per_phase, test_trials),
                        save_in_transition_subdir=False,
                        save_csv=True,
                        zscore_activity=True,
                        image_format=fmt,
                        condition_title_size=22,
                        phase_title_size=26,
                        panel_top=0.80,
                        phase_title_y=0.94,
                        match_sector_export_style=True,
                    )


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.paper_output_dir / "metadata.json").read_text())
    formats = _formats(args)
    for circuit in args.circuits:
        if not args.skip_representatives:
            _plot_circuit(
                circuit,
                metadata=metadata,
                output_dir=args.output_dir,
                pc_output_dir=args.pc_output_dir,
                formats=formats,
                n_jobs=args.n_jobs,
            )
        if not args.skip_sector_averages:
            _plot_sector_average_circuit(
                circuit,
                metadata=metadata,
                output_dir=args.pc_output_dir / "sector_average_examples",
                pc_output_dir=args.pc_output_dir,
                formats=formats,
                n_jobs=args.n_jobs,
                max_sector_traces_per_sector=int(args.max_sector_traces_per_sector),
            )


if __name__ == "__main__":
    main()
