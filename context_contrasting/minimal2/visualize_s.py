import os
import re
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from pandas import DataFrame

from context_contrasting.figures import FigureBuilder
from context_contrasting.minimal2 import (
    PLOTSDIR,
    PLOT_ALL_PANELS_DIR,
    PLOT_PANEL_A_DIR,
    PLOT_TRANSITION_PANELS_DIR,
)

PLOT_CONDITION_LABELS = {
    "full": "Nonoccluded",
    "occlusion": "Occluded",
    "no_context": "No feedback",
    "nolat": "No LAT",
    "no_context_nolat": "No fb/no LAT",
}
PLOT_CONDITION_ORDER = ["full", "occlusion", "no_context", "nolat", "no_context_nolat"]
PLOT_COLORS = {
    "Nonoccluded": "black",
    "Occluded": "red",
    "No feedback": "blue",
    "No LAT": "green",
    "No fb/no LAT": "darkorange",
}
TRANSITION_ORDER = [
    "un_un",
    "un_FB",
    "un_novel_FF",
    "FF_un",
    "FF_FB_broad",
    "FF_FB_broad_novel",
    "FF_FB_narrow_familiar",
    "FF_FB_narrow_familiar_novel",
    "FF_FB_narrow_novel",
    "FB_FB",
]
TRANSITION_LABELS = {
    "un_un": "un -> un",
    "un_FB": "un -> FB",
    "un_novel_FF": "un -> novel NO",
    "FF_un": "FF -> un",
    "FF_FB_broad": "FF -> FB\n(broad)",
    "FF_FB_broad_novel": "FF_FB_broad_novel",
    "FF_FB_narrow_familiar": "FF -> FB\n(narrow fam)",
    "FF_FB_narrow_familiar_novel": "FF_FB_narrow_familiar_novel",
    "FF_FB_narrow_novel": "FF -> FB\n(narrow nov)",
    "FB_FB": "FB -> FB",
}
TRACE_COLORS = {
    "full": "black",
    "occlusion": "red",
    "no_context": "blue",
    "nolat": "green",
    "no_context_nolat": "darkorange",
}
TRACE_LINESTYLES = {
    "full": "-",
    "occlusion": "-",
    "no_context": ":",
    "nolat": "--",
    "no_context_nolat": "--",
}
TRACE_LABELS = {
    "full": "Nonoccluded",
    "occlusion": "Occluded",
    "no_context": "No feedback",
    "nolat": "No LAT",
    "no_context_nolat": "No fb/no LAT",
}
IMAGE_LABELS = {"familiar": "Familiar Image", "novel": "Novel Image"}
AXIS_LABEL_FONTSIZE = 32
AXIS_TICK_FONTSIZE = 32
TIME_STEPS_PER_SECOND = 100.0
PHASE_DISPLAY_LABELS = {
    "naive": "Naive",
    "expert": "Expert",
}
PHASE_ORDER = ["naive", "expert"]


def _resolve_phase_sequence(long_df: DataFrame) -> list[str]:
    if "experiment_phase" not in long_df.columns:
        return []
    present = long_df["experiment_phase"].dropna().astype(str).unique().tolist()
    return [phase for phase in PHASE_ORDER if phase in present]


def _resolve_plot_dirs(base_dir: str) -> dict[str, str]:
    if os.path.abspath(base_dir) == os.path.abspath(PLOTSDIR):
        plot_dirs = {
            "all_panels": PLOT_ALL_PANELS_DIR,
            "panel_a": PLOT_PANEL_A_DIR,
            "transition_panels": PLOT_TRANSITION_PANELS_DIR,
        }
    else:
        plot_dirs = {
            "all_panels": os.path.join(base_dir, "all_panels"),
            "panel_a": os.path.join(base_dir, "panel_A"),
            "transition_panels": os.path.join(base_dir, "transition_panels"),
        }

    for path in plot_dirs.values():
        os.makedirs(path, exist_ok=True)
    return plot_dirs


def _natural_sort_key(text: str) -> list[str | int]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def _is_familiar_condition(condition: str) -> bool:
    return str(condition).startswith("familiar")


def _is_novel_condition(condition: str) -> bool:
    return str(condition).startswith("novel")


def _condition_sort_key(condition: str) -> tuple[int, int, list[str | int]]:
    text = str(condition)
    if _is_familiar_condition(text):
        return (0, 0 if text == "familiar" else 1, _natural_sort_key(text))
    if _is_novel_condition(text):
        return (1, 0 if text == "novel" else 1, _natural_sort_key(text))
    return (2, 0, _natural_sort_key(text))


def _resolve_condition_sequence(
    conditions: list[str],
    preferred: list[str] | None = None,
) -> list[str]:
    present = [str(condition) for condition in conditions if pd.notna(condition)]
    if not present:
        return []

    ordered: list[str] = []
    seen: set[str] = set()

    for condition in preferred or []:
        text = str(condition)
        if text in present and text not in seen:
            ordered.append(text)
            seen.add(text)

    for condition in sorted(set(present), key=_condition_sort_key):
        if condition not in seen:
            ordered.append(condition)
            seen.add(condition)

    return ordered


def _display_condition_label(condition: str) -> str:
    text = str(condition)
    if text in IMAGE_LABELS:
        return IMAGE_LABELS[text]
    return text.replace("_", " ").title()


def _add_plot_condition_labels(df: DataFrame) -> DataFrame:
    styled = df.copy()
    if "image_type" in styled.columns:
        styled["plot_condition"] = styled["image_type"].map(PLOT_CONDITION_LABELS).fillna(styled["image_type"])
    return styled


def _plot_condition_order(image_types: list[str]) -> list[str]:
    return [PLOT_CONDITION_LABELS[k] for k in PLOT_CONDITION_ORDER if k in image_types]


def _resolve_xlim(xlim: tuple[float, float] | None) -> tuple[float, float]:
    if xlim is None:
        return (1000.0, 1350.0)
    start, end = xlim
    if start >= end:
        raise ValueError("xlim must be an increasing (start, end) tuple.")
    return float(start), float(end)


def _style_axis_fonts(ax, *, tick_labelsize: int = AXIS_TICK_FONTSIZE) -> None:
    ax.xaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=tick_labelsize)


def _to_np_2d(ts: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(ts, torch.Tensor):
        arr = ts.detach().cpu().numpy()
    else:
        arr = np.asarray(ts)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _condition_token_to_image_type(image_type: str, condition_token: str) -> tuple[str, str]:
    if image_type == "nocontext":
        return "no_context", condition_token
    if image_type == "nocontextnolat":
        return "no_context_nolat", condition_token
    if condition_token.endswith("_nocontext"):
        base_condition = condition_token.removesuffix("_nocontext")
        return "no_context", base_condition
    return image_type, condition_token


def _resolve_trace_types(
    image_types: list[str] | set[str],
    *,
    include_no_response_ablations: bool,
    base_trace_types: tuple[str, ...] = ("full", "occlusion"),
) -> tuple[str, ...]:
    available = {str(image_type) for image_type in image_types if pd.notna(image_type)}
    trace_types = [trace_type for trace_type in base_trace_types if trace_type in available]

    if include_no_response_ablations:
        for trace_type in PLOT_CONDITION_ORDER:
            if trace_type in base_trace_types:
                continue
            if trace_type in available:
                trace_types.append(trace_type)

    return tuple(trace_types)


def _indexed_palette(indices: list[int], palette_name: str) -> dict[int, tuple[float, float, float]]:
    unique_indices = sorted({int(idx) for idx in indices})
    colors = sns.color_palette(palette_name, n_colors=max(1, len(unique_indices)))
    return {idx: colors[pos] for pos, idx in enumerate(unique_indices)}


def _resolve_image_mode(
    available_conditions: list[str],
    image_mode: Literal["familiar", "novel", "both"] | None,
    include_novel_image: bool | None,
) -> list[str]:
    ordered = _resolve_condition_sequence(available_conditions)
    familiar_conditions = [condition for condition in ordered if _is_familiar_condition(condition)]
    novel_conditions = [condition for condition in ordered if _is_novel_condition(condition)]
    other_conditions = [condition for condition in ordered if condition not in familiar_conditions and condition not in novel_conditions]

    if image_mode is None:
        image_mode = "both" if include_novel_image else "familiar"
    if image_mode not in {"familiar", "novel", "both"}:
        raise ValueError("image_mode must be one of 'familiar', 'novel', or 'both'.")

    if image_mode == "familiar":
        selected = familiar_conditions or other_conditions
    elif image_mode == "novel":
        selected = novel_conditions or other_conditions
    else:
        selected = familiar_conditions + novel_conditions + other_conditions

    return selected or ordered


def _extract_repeated_y(
    long_df: DataFrame,
    condition: str,
    phase: str,
    image_type: str,
) -> np.ndarray:
    cell = long_df.loc[
        long_df["condition"].eq(condition)
        & long_df["experiment_phase"].eq(phase)
        & long_df["image_type"].eq(image_type),
        ["step", "y"],
    ].drop_duplicates()
    if cell.empty:
        return np.asarray([], dtype=float)
    return cell.sort_values("step")["y"].to_numpy(dtype=float)


def _extract_stimulus_intervals(
    stim_pair: tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray],
) -> list[tuple[int, int]]:
    x = _to_np_2d(stim_pair[0])
    c = _to_np_2d(stim_pair[1])
    stimulus_strength = np.maximum(np.abs(x).max(axis=1), np.abs(c).max(axis=1))
    if stimulus_strength.size == 0:
        return []

    peak = float(np.nanmax(stimulus_strength))
    if not np.isfinite(peak) or peak <= 0:
        return []

    active = stimulus_strength > max(0.2 * peak, 0.15)
    onsets = np.flatnonzero(np.diff(np.r_[0, active.astype(int)]) == 1)
    offsets = np.flatnonzero(np.diff(np.r_[active.astype(int), 0]) == -1) + 1
    return [(int(onset), int(offset)) for onset, offset in zip(onsets, offsets, strict=False)]


def _expand_window_to_event_bounds(
    stim_pair: tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray],
    focus_window: tuple[int, int],
) -> tuple[int, int]:
    x = _to_np_2d(stim_pair[0])
    c = _to_np_2d(stim_pair[1])
    stimulus_strength = np.maximum(np.abs(x).max(axis=1), np.abs(c).max(axis=1))
    if stimulus_strength.size == 0:
        return focus_window

    peak = float(np.nanmax(stimulus_strength))
    if not np.isfinite(peak) or peak <= 0:
        return focus_window

    active = stimulus_strength > max(0.2 * peak, 0.15)
    onsets = np.flatnonzero(np.diff(np.r_[0, active.astype(int)]) == 1)
    offsets = np.flatnonzero(np.diff(np.r_[active.astype(int), 0]) == -1) + 1
    if onsets.size == 0 or offsets.size == 0:
        return focus_window

    start, end = focus_window
    best_idx = None
    best_overlap = -1
    for idx, (onset, offset) in enumerate(zip(onsets, offsets, strict=False)):
        overlap = min(end, offset) - max(start, onset)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx

    if best_idx is None or best_overlap <= 0:
        return focus_window

    current_onset = int(onsets[best_idx])
    current_offset = int(offsets[best_idx])

    if best_idx > 0:
        prev_offset = int(offsets[best_idx - 1])
        expanded_start = int(round(0.5 * (prev_offset + current_onset)))
    else:
        expanded_start = 0

    if best_idx < len(onsets) - 1:
        next_onset = int(onsets[best_idx + 1])
        expanded_end = int(round(0.5 * (current_offset + next_onset)))
    else:
        expanded_end = int(stimulus_strength.size)

    return expanded_start, expanded_end


def _relative_seconds_ticks(xlim_seconds: tuple[float, float]) -> tuple[np.ndarray, list[str]]:
    tick_start = float(np.ceil(xlim_seconds[0]))
    tick_end = float(np.floor(xlim_seconds[1]))
    if tick_end < tick_start:
        ticks = np.asarray([xlim_seconds[0], xlim_seconds[1]], dtype=float)
    else:
        ticks = np.arange(tick_start, tick_end + 1.0, 1.0, dtype=float)

    labels = [str(int(round(tick))) if np.isclose(tick, round(tick)) else f"{tick:g}" for tick in ticks]
    return ticks, labels


def _window_repeated_trace(
    series: np.ndarray,
    *,
    stim_pair: tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray],
    focus_window: tuple[float, float],
) -> dict[str, np.ndarray | float | int] | None:
    if series.size == 0:
        return None

    intervals = _extract_stimulus_intervals(stim_pair)
    if not intervals:
        return None

    window_start, window_end = focus_window
    focus_interval = None
    for onset, offset in intervals:
        if onset < window_end and offset > window_start:
            focus_interval = (onset, offset)
            break
    if focus_interval is None:
        focus_interval = intervals[0]

    rel_start = int(round(window_start - focus_interval[0]))
    rel_end = int(round(window_end - focus_interval[0]))
    if rel_end <= rel_start:
        return None

    windows: list[np.ndarray] = []
    for onset, _ in intervals:
        start = onset + rel_start
        end = onset + rel_end
        if start < 0 or end > series.size:
            continue
        windows.append(series[start:end])

    if not windows:
        return None

    stacked = np.vstack(windows).astype(float)
    baseline_stop = max(0, min(-rel_start, stacked.shape[1]))
    if baseline_stop > 0:
        baseline_values = stacked[:, :baseline_stop].reshape(-1)
    else:
        baseline_values = stacked.reshape(-1)

    relative_step = np.arange(rel_start, rel_end, dtype=float)
    return {
        "stacked": stacked,
        "baseline_values": baseline_values,
        "stim_seconds": (
            0.0,
            float(focus_interval[1] - focus_interval[0]) / TIME_STEPS_PER_SECOND,
        ),
        "xlim_seconds": (
            float(rel_start) / TIME_STEPS_PER_SECOND,
            float(rel_end) / TIME_STEPS_PER_SECOND,
        ),
        "x_seconds": relative_step / TIME_STEPS_PER_SECOND,
    }


def _compute_baseline_stats(baseline_values: np.ndarray) -> dict[str, float | int]:
    baseline_values = np.asarray(baseline_values, dtype=float).reshape(-1)
    baseline_n = int(baseline_values.size)
    baseline_mean = float(baseline_values.mean()) if baseline_n else 0.0
    baseline_std = float(baseline_values.std(ddof=0)) if baseline_n else 0.0
    return {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "baseline_n": baseline_n,
    }


def _collect_shared_baseline_stats(
    long_df: DataFrame,
    *,
    trace_specs: list[tuple[str, str, str]],
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    focus_window: tuple[float, float],
) -> dict[str, float | int] | None:
    baseline_chunks: list[np.ndarray] = []
    for condition, phase, image_type in trace_specs:
        stim_pair = stimuli.get(condition)
        if stim_pair is None:
            continue
        series = _extract_repeated_y(long_df, condition=condition, phase=phase, image_type=image_type)
        if series.size == 0:
            continue
        windowed = _window_repeated_trace(series, stim_pair=stim_pair, focus_window=focus_window)
        if windowed is None:
            continue
        baseline_values = np.asarray(windowed["baseline_values"], dtype=float).reshape(-1)
        if baseline_values.size:
            baseline_chunks.append(baseline_values)

    if not baseline_chunks:
        return None
    return _compute_baseline_stats(np.concatenate(baseline_chunks))


def _summarize_windowed_repeated_trace(
    long_df: DataFrame,
    *,
    condition: str,
    phase: str,
    image_type: str,
    stim_pair: tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray],
    focus_window: tuple[float, float],
    zscore: bool = True,
    baseline_stats: dict[str, float | int] | None = None,
) -> dict[str, np.ndarray | float | int] | None:
    series = _extract_repeated_y(long_df, condition=condition, phase=phase, image_type=image_type)
    windowed = _window_repeated_trace(series, stim_pair=stim_pair, focus_window=focus_window)
    if windowed is None:
        return None

    stacked = np.asarray(windowed["stacked"], dtype=float)
    if baseline_stats is None:
        baseline_stats = _compute_baseline_stats(np.asarray(windowed["baseline_values"], dtype=float))

    baseline_mean = float(baseline_stats["baseline_mean"])
    baseline_std = float(baseline_stats["baseline_std"])
    baseline_n = int(baseline_stats["baseline_n"])
    if zscore:
        scale = baseline_std if np.isfinite(baseline_std) and baseline_std > 1e-12 else 1.0
        summarized = (stacked - baseline_mean) / scale
    else:
        summarized = stacked

    y_mean = summarized.mean(axis=0)
    if summarized.shape[0] > 1:
        y_sem = summarized.std(axis=0, ddof=1) / np.sqrt(summarized.shape[0])
    else:
        y_sem = np.zeros_like(y_mean)

    return {
        "x_seconds": np.asarray(windowed["x_seconds"], dtype=float),
        "y_mean": y_mean,
        "y_sem": y_sem,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "baseline_n": baseline_n,
        "n_trials": int(stacked.shape[0]),
        "stim_seconds": tuple(windowed["stim_seconds"]),
        "xlim_seconds": tuple(windowed["xlim_seconds"]),
    }


def _build_windowed_transition_export(
    long_dfs_by_transition: dict[str, DataFrame],
    *,
    ordered_transitions: list[str],
    labels: dict[str, str],
    phases: list[str],
    selected_conditions: list[str],
    trace_types: tuple[str, ...],
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    plot_window: tuple[float, float],
    zscore_activity: bool,
) -> DataFrame:
    export_frames: list[DataFrame] = []

    for transition_row, transition_name in enumerate(ordered_transitions):
        long_df = long_dfs_by_transition[transition_name]
        for phase_index, phase in enumerate(phases):
            baseline_stats = _collect_shared_baseline_stats(
                long_df,
                trace_specs=[
                    (condition, phase, trace_type)
                    for condition in selected_conditions
                    for trace_type in trace_types
                ],
                stimuli=stimuli,
                focus_window=plot_window,
            )
            for condition_index, condition in enumerate(selected_conditions):
                stim_pair = stimuli.get(condition)
                if stim_pair is None:
                    continue
                for trace_index, trace_type in enumerate(trace_types):
                    summary = _summarize_windowed_repeated_trace(
                        long_df,
                        condition=condition,
                        phase=phase,
                        image_type=trace_type,
                        stim_pair=stim_pair,
                        focus_window=plot_window,
                        zscore=zscore_activity,
                        baseline_stats=baseline_stats,
                    )
                    if summary is None:
                        continue
                    y_mean = np.asarray(summary["y_mean"], dtype=float)
                    y_sem = np.asarray(summary["y_sem"], dtype=float)
                    x_seconds = np.asarray(summary["x_seconds"], dtype=float)
                    stim_start, stim_end = summary["stim_seconds"]

                    export_frames.append(
                        pd.DataFrame(
                            {
                                "time_seconds": x_seconds,
                                "y": y_mean,
                                "y_sem": y_sem,
                            }
                        ).assign(
                            transition=transition_name,
                            transition_label=labels.get(transition_name, transition_name),
                            transition_row=transition_row,
                            experiment_phase=phase,
                            phase_label=PHASE_DISPLAY_LABELS.get(phase, phase.title()),
                            phase_index=phase_index,
                            condition=condition,
                            condition_label=_display_condition_label(condition),
                            condition_index=condition_index,
                            image_type=trace_type,
                            trace_label=TRACE_LABELS.get(trace_type, trace_type),
                            trace_index=trace_index,
                            baseline_mean=float(summary["baseline_mean"]),
                            baseline_std=float(summary["baseline_std"]),
                            baseline_n=int(summary["baseline_n"]),
                            n_trials=int(summary["n_trials"]),
                            stim_start_seconds=float(stim_start),
                            stim_end_seconds=float(stim_end),
                        )
                    )

    if not export_frames:
        return pd.DataFrame(
            columns=[
                "transition",
                "transition_label",
                "transition_row",
                "experiment_phase",
                "phase_label",
                "phase_index",
                "condition",
                "condition_label",
                "condition_index",
                "image_type",
                "trace_label",
                "trace_index",
                "time_seconds",
                "y",
                "y_sem",
                "baseline_mean",
                "baseline_std",
                "baseline_n",
                "n_trials",
                "experiment_series",
                "seed",
                "stim_start_seconds",
                "stim_end_seconds",
            ]
        )

    export_df = pd.concat(export_frames, ignore_index=True)
    ordered_columns = [
        "transition",
        "transition_label",
        "transition_row",
        "experiment_phase",
        "phase_label",
        "phase_index",
        "condition",
        "condition_label",
        "condition_index",
        "image_type",
        "trace_label",
        "trace_index",
        "time_seconds",
        "y",
        "y_sem",
        "baseline_mean",
        "baseline_std",
        "baseline_n",
        "n_trials",
        "stim_start_seconds",
        "stim_end_seconds",
    ]
    optional_columns = [column for column in ("experiment_series", "seed") if column in export_df.columns]
    return export_df.loc[:, ordered_columns + optional_columns]


def _plot_panel_a_activity(
    ax_grid: np.ndarray,
    y_df: DataFrame,
    activity_layout: list[tuple[str, str]],
    STIMULI: dict[str, tuple[torch.Tensor, torch.Tensor]],
    xlim: tuple[float, float],
    *,
    zscore_activity: bool = True,
    include_novel_no_context: bool = False,
    image_types: list[str] | None = None,
) -> None:
    flat_axes = np.asarray(ax_grid).reshape(-1)
    if flat_axes.size == 0 or not activity_layout:
        return

    for ax in flat_axes[len(activity_layout):]:
        ax.set_visible(False)

    ref_ax = flat_axes[0]
    for ax in flat_axes[1:len(activity_layout)]:
        ax.sharex(ref_ax)
        ax.sharey(ref_ax)

    available_image_types = set(image_types or y_df["image_type"].dropna().unique().tolist())
    trace_types = _resolve_trace_types(
        available_image_types,
        include_no_response_ablations=include_novel_no_context,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TRACE_COLORS[trace_type],
            linestyle=TRACE_LINESTYLES.get(trace_type, "-"),
            lw=5.0,
            label=TRACE_LABELS[trace_type],
        )
        for trace_type in trace_types
        if trace_type in TRACE_COLORS
    ]

    phase_baseline_stats: dict[str, dict[str, float | int]] = {}
    for phase in list(dict.fromkeys(phase_name for _, phase_name in activity_layout)):
        trace_specs: list[tuple[str, str, str]] = []
        for condition, condition_phase in activity_layout:
            if condition_phase != phase or condition not in STIMULI:
                continue
            trace_specs.extend((condition, phase, trace_type) for trace_type in trace_types)
        baseline_stats = _collect_shared_baseline_stats(
            y_df,
            trace_specs=trace_specs,
            stimuli=STIMULI,
            focus_window=xlim,
        )
        if baseline_stats is not None:
            phase_baseline_stats[phase] = baseline_stats

    xlims_seconds: list[tuple[float, float]] = []
    stim_windows_seconds: dict[str, tuple[float, float] | None] = {}
    for condition, _ in activity_layout:
        if condition not in STIMULI:
            continue
        summary = _summarize_windowed_repeated_trace(
            y_df,
            condition=condition,
            phase=activity_layout[0][1],
            image_type="full",
            stim_pair=STIMULI[condition],
            focus_window=xlim,
            zscore=zscore_activity,
            baseline_stats=phase_baseline_stats.get(activity_layout[0][1]),
        )
        if summary is None:
            continue
        xlims_seconds.append(summary["xlim_seconds"])
        stim_windows_seconds[condition] = summary["stim_seconds"]

    if xlims_seconds:
        xlim_seconds = (
            min(bounds[0] for bounds in xlims_seconds),
            max(bounds[1] for bounds in xlims_seconds),
        )
    else:
        xlim_seconds = (
            float(xlim[0]) / TIME_STEPS_PER_SECOND,
            float(xlim[1]) / TIME_STEPS_PER_SECOND,
        )
    xticks, xticklabels = _relative_seconds_ticks(xlim_seconds)

    global_y_bounds: list[tuple[float, float]] = []
    for idx, (condition, phase) in enumerate(activity_layout):
        ax = flat_axes[idx]
        trace_specs = [
            (trace_type, TRACE_COLORS.get(trace_type, "black"))
            for trace_type in trace_types
        ]

        has_trace = False
        for image_type, color in trace_specs:
            if condition not in STIMULI:
                continue
            summary = _summarize_windowed_repeated_trace(
                y_df,
                condition=condition,
                phase=phase,
                image_type=image_type,
                stim_pair=STIMULI[condition],
                focus_window=xlim,
                zscore=zscore_activity,
                baseline_stats=phase_baseline_stats.get(phase),
            )
            if summary is None:
                continue
            has_trace = True
            ax.plot(
                np.asarray(summary["x_seconds"], dtype=float),
                np.asarray(summary["y_mean"], dtype=float),
                color=color,
                linestyle=TRACE_LINESTYLES.get(image_type, "-"),
                lw=5.0,
            )
            global_y_bounds.append(
                (
                    float(np.min(np.asarray(summary["y_mean"], dtype=float))),
                    float(np.max(np.asarray(summary["y_mean"], dtype=float))),
                )
            )

        if not has_trace:
            ax.set_visible(False)
            continue

        stim_interval = stim_windows_seconds.get(condition)
        if stim_interval is not None:
            ax.axvspan(
                stim_interval[0],
                stim_interval[1],
                ymin=0.02,
                ymax=0.055,
                color="#8c5a2b",
                clip_on=False,
                zorder=3,
            )

        ax.set_xlim(xlim_seconds)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_title(
            f"{_display_condition_label(condition)} | {PHASE_DISPLAY_LABELS.get(phase, phase.title())}",
            fontsize=19,
            pad=10,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(5.0)
        ax.spines["bottom"].set_linewidth(5.0)
        ax.tick_params(axis="both", width=1.6, length=5, labelsize=AXIS_TICK_FONTSIZE)
        ax.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE)
        if idx == 0:
            ax.set_ylabel("Z-scored activity" if zscore_activity else "Activation", fontsize=AXIS_LABEL_FONTSIZE)
            ax.legend(
                handles=legend_handles,
                loc="upper left",
                frameon=False,
                handlelength=2.0,
                borderaxespad=0.2,
                fontsize=15,
            )
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        _style_axis_fonts(ax)

    if global_y_bounds:
        y_min = min(bound[0] for bound in global_y_bounds)
        y_max = max(bound[1] for bound in global_y_bounds)
        span = y_max - y_min
        if span <= 0:
            span = max(abs(y_min), abs(y_max), 0.1)
        pad = 0.08 * span
        for ax in flat_axes[:len(activity_layout)]:
            if ax.get_visible():
                ax.set_ylim(y_min - pad, y_max + pad)


def visualize_transition_panel(
    long_dfs_by_transition: dict[str, DataFrame],
    STIMULI: dict[str, tuple[torch.Tensor, torch.Tensor]],
    save_path: str = PLOTSDIR,
    name: str = "transition_panel",
    image_mode: Literal["familiar", "novel", "both"] | None = None,
    include_novel_image: bool | None = None,
    transition_order: list[str] | None = None,
    transition_labels: dict[str, str] | None = None,
    trace_types: tuple[str, ...] | None = None,
    step_window: tuple[int, int] = (1000, 1350),
    save_in_transition_subdir: bool = True,
    save_csv: bool = True,
    zscore_activity: bool = False,
) -> str:
    if not long_dfs_by_transition:
        raise ValueError("long_dfs_by_transition must contain at least one transition result.")

    ordered_transitions = transition_order or TRANSITION_ORDER
    ordered_transitions = [transition for transition in ordered_transitions if transition in long_dfs_by_transition]
    if not ordered_transitions:
        ordered_transitions = list(long_dfs_by_transition)

    labels = TRANSITION_LABELS.copy()
    if transition_labels is not None:
        labels.update(transition_labels)

    sample_df = long_dfs_by_transition[ordered_transitions[0]]
    phases = _resolve_phase_sequence(sample_df)
    if not phases:
        raise ValueError("Transition panel requires experiment_phase values.")

    phase_filtered_df = sample_df.loc[sample_df["experiment_phase"].isin(phases)].copy()
    available_conditions = _resolve_condition_sequence(
        phase_filtered_df["condition"].dropna().astype(str).unique().tolist() if "condition" in phase_filtered_df.columns else [],
        preferred=list(STIMULI),
    )
    available_image_types = phase_filtered_df["image_type"].dropna().astype(str).unique().tolist() if "image_type" in phase_filtered_df.columns else []
    resolved_trace_types = trace_types or _resolve_trace_types(
        available_image_types,
        include_no_response_ablations=True,
    )
    selected_conditions = _resolve_image_mode(
        available_conditions=available_conditions or list(STIMULI),
        image_mode=image_mode,
        include_novel_image=include_novel_image,
    )

    display_windows = [
        _expand_window_to_event_bounds(STIMULI[condition], focus_window=step_window)
        for condition in selected_conditions
        if condition in STIMULI
    ]
    if display_windows:
        plot_window = (
            min(window[0] for window in display_windows),
            max(window[1] for window in display_windows),
        )
    else:
        plot_window = step_window

    condition_summaries: dict[str, dict[str, np.ndarray | float | int]] = {}
    for condition in selected_conditions:
        stim_pair = STIMULI.get(condition)
        if stim_pair is None:
            continue
        summary = _summarize_windowed_repeated_trace(
            sample_df,
            condition=condition,
            phase=phases[0],
            image_type=resolved_trace_types[0],
            stim_pair=stim_pair,
            focus_window=plot_window,
            zscore=zscore_activity,
        )
        if summary is not None:
            condition_summaries[condition] = summary

    if not condition_summaries:
        raise ValueError("STIMULI must contain at least one of the requested conditions.")

    stim_windows = {
        condition: tuple(summary["stim_seconds"])
        for condition, summary in condition_summaries.items()
    }
    xlim_transition = (
        min(float(summary["xlim_seconds"][0]) for summary in condition_summaries.values()),
        max(float(summary["xlim_seconds"][1]) for summary in condition_summaries.values()),
    )

    column_specs = [(phase, condition) for phase in phases for condition in selected_conditions]
    n_rows = len(ordered_transitions)
    n_cols = len(column_specs)
    transition_export_df = None
    if save_csv:
        transition_export_df = _build_windowed_transition_export(
            long_dfs_by_transition,
            ordered_transitions=ordered_transitions,
            labels=labels,
            phases=phases,
            selected_conditions=selected_conditions,
            trace_types=resolved_trace_types,
            stimuli=STIMULI,
            plot_window=plot_window,
            zscore_activity=zscore_activity,
        )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols + 1.8, 3 * n_rows + 1.9),
        squeeze=False,
        sharex=True,
        sharey=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.18, right=0.99, top=0.89, bottom=0.05, wspace=0.12, hspace=0.18)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TRACE_COLORS[trace_type],
            linestyle=TRACE_LINESTYLES.get(trace_type, "-"),
            lw=1.6,
            label=TRACE_LABELS.get(trace_type, trace_type),
        )
        for trace_type in resolved_trace_types
        if trace_type in TRACE_COLORS
    ]
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.985),
            frameon=False,
            ncol=len(legend_handles),
            handlelength=2.0,
            columnspacing=1.2,
        )

    for col_idx, (phase, condition) in enumerate(column_specs):
        axes[0, col_idx].set_title(_display_condition_label(condition), fontsize=32, pad=12)

    for phase_idx, phase in enumerate(phases):
        start_col = phase_idx * len(selected_conditions)
        end_col = start_col + len(selected_conditions) - 1
        x_center = 0.5 * (axes[0, start_col].get_position().x0 + axes[0, end_col].get_position().x1)
        fig.text(
            x_center,
            0.945,
            PHASE_DISPLAY_LABELS.get(phase, phase.title()),
            ha="center",
            va="center",
            fontsize=32,
        )

    for row_idx, transition_name in enumerate(ordered_transitions):
        long_df = long_dfs_by_transition[transition_name]
        row_bounds: list[tuple[float, float]] = []
        phase_baseline_stats = {
            phase: _collect_shared_baseline_stats(
                long_df,
                trace_specs=[
                    (condition, phase, trace_type)
                    for condition in selected_conditions
                    for trace_type in resolved_trace_types
                ],
                stimuli=STIMULI,
                focus_window=plot_window,
            )
            for phase in phases
        }

        for col_idx, (phase, condition) in enumerate(column_specs):
            ax = axes[row_idx, col_idx]
            stim_interval = stim_windows.get(condition)
            if condition not in STIMULI:
                ax.set_visible(False)
                continue

            if stim_interval is not None:
                ax.axvspan(stim_interval[0], stim_interval[1], color="0.9", zorder=0)
            ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)

            for trace_type in resolved_trace_types:
                summary = _summarize_windowed_repeated_trace(
                    long_df,
                    condition=condition,
                    phase=phase,
                    image_type=trace_type,
                    stim_pair=STIMULI[condition],
                    focus_window=plot_window,
                    zscore=zscore_activity,
                    baseline_stats=phase_baseline_stats.get(phase),
                )
                if summary is None:
                    continue
                ax.plot(
                    np.asarray(summary["x_seconds"], dtype=float),
                    np.asarray(summary["y_mean"], dtype=float),
                    color=TRACE_COLORS.get(trace_type, "black"),
                    linestyle=TRACE_LINESTYLES.get(trace_type, "-"),
                    lw=5,
                )
                row_bounds.append(
                    (
                        float(np.min(np.asarray(summary["y_mean"], dtype=float))),
                        float(np.max(np.asarray(summary["y_mean"], dtype=float))),
                    )
                )

            ax.set_xlim(*xlim_transition)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        label_ax = axes[row_idx, 0]
        label_ax.text(
            -0.12,
            0.5,
            labels.get(transition_name, transition_name),
            transform=label_ax.transAxes,
            ha="right",
            va="center",
            fontsize=32,
        )
        if row_bounds:
            row_min = min(bound[0] for bound in row_bounds)
            row_max = max(bound[1] for bound in row_bounds)
            span = row_max - row_min
            if span <= 0:
                span = max(abs(row_min), abs(row_max), 0.1)
            pad = 0.12 * span
            for ax in axes[row_idx, :]:
                if ax.get_visible():
                    ax.set_ylim(row_min - pad, row_max + pad)

    if save_in_transition_subdir:
        plot_dirs = _resolve_plot_dirs(save_path)
        out_path = os.path.join(plot_dirs["transition_panels"], f"{name}_{'_'.join(selected_conditions)}.svg")
    else:
        os.makedirs(save_path, exist_ok=True)
        out_path = os.path.join(save_path, f"{name}_{'_'.join(selected_conditions)}.svg")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    if transition_export_df is not None:
        transition_export_df.to_csv(os.path.splitext(out_path)[0] + ".csv", index=False)
    return out_path


def save_grouped_transition_panels(
    long_dfs_by_transition: dict[str, DataFrame],
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    save_path: str,
    transition_order: list[str],
    save_in_transition_subdir: bool = True,
) -> None:
    sample_df = next(iter(long_dfs_by_transition.values()), None)
    if sample_df is None:
        return

    combined_transitions: dict[str, DataFrame] = {}
    for transition_name in transition_order:
        long_df = long_dfs_by_transition.get(transition_name)
        if long_df is None:
            continue

        combined = long_df.loc[long_df["experiment_phase"].isin(["naive", "expert"])].copy()
        if combined.empty:
            continue

        combined_transitions[transition_name] = combined

    if combined_transitions:
        visualize_transition_panel(
            combined_transitions,
            STIMULI=stimuli,
            save_path=save_path,
            name="transition_panel_naive_expert",
            image_mode="both",
            transition_order=[name for name in transition_order if name in combined_transitions],
            transition_labels={name: TRANSITION_LABELS.get(name, name) for name in combined_transitions},
            save_in_transition_subdir=save_in_transition_subdir,
            save_csv=True,
        )


def visualize_naive_expert_results(
    long_df: DataFrame,
    STIMULI: dict[str, tuple[torch.Tensor, torch.Tensor]],
    save_path: str = PLOTSDIR,
    name: str | None = None,
    full_plots: bool = True,
    include_novel_no_context: bool = False,
    xlim: tuple[float, float] | None = None,
) -> None:
    xlim = _resolve_xlim(xlim)
    phases = _resolve_phase_sequence(long_df)
    pre_post_df = long_df.loc[long_df["experiment_phase"].isin(phases)].copy()
    image_types = sorted(pre_post_df["image_type"].dropna().unique().tolist()) if "image_type" in pre_post_df.columns else []
    conditions = _resolve_condition_sequence(
        pre_post_df["condition"].dropna().unique().tolist() if "condition" in pre_post_df.columns else [],
        preferred=list(STIMULI),
    )
    hue_order = _plot_condition_order(image_types)
    y_df = _add_plot_condition_labels(
        pre_post_df[["step", "y", "condition", "experiment_phase", "image_type"]].drop_duplicates()
    )
    pv_df = _add_plot_condition_labels(
        pre_post_df[["step", "pv_value", "pv_index", "condition", "experiment_phase", "image_type"]].drop_duplicates()
    )
    training_rows = long_df.loc[long_df["experiment_phase"].eq("training")].copy()
    if training_rows.empty:
        training_rows = pre_post_df.copy()

    if {"image_type", "condition", "experiment_phase"}.issubset(long_df.columns):
        familiar_training_mask = long_df["condition"].astype(str).map(_is_familiar_condition).fillna(False)
        weight_rows = long_df.loc[
            long_df["image_type"].eq("full")
            & familiar_training_mask
            & long_df["experiment_phase"].eq("training")
        ].copy()
    else:
        weight_rows = pd.DataFrame()
    if weight_rows.empty:
        weight_rows = training_rows.copy()

    n_activity_panels = max(1, len(conditions) * len(phases))
    panel_width = max(24, 4.8 * n_activity_panels)

    if full_plots:
        builder = FigureBuilder.from_matrix(
            [["A", "A", "A", "A"],
             ["B", "B", "B", "B"],
             ["C", "C", "D", "D"]],
            figsize=(panel_width, 18),
            height_ratios=[1.0, 1.0, 1.4],
            constrained_layout=False,
            grid_wspace=0.25,
            grid_hspace=0.15,
            subfigure_wspace=0.15,
            subfigure_hspace=0.2,
        )
        builder.update_panel("A", subgrid=(1, n_activity_panels), title=None, label="A", wspace=0.18)
        builder.update_panel("B", subgrid=(1, n_activity_panels), title="PV Activity", label="B")
        builder.update_panel("C", subgrid=(2, 1), title="Y and PV activity over training", label="C")
        builder.update_panel("D", subgrid=(4, 1), title="Weight evolution over training", label="D")
    else:
        builder = FigureBuilder.from_matrix(
            [["A", "A", "A", "A"]],
            figsize=(panel_width, 5.5),
            height_ratios=[1.0],
            constrained_layout=False,
            grid_wspace=0.25,
            grid_hspace=0.15,
            subfigure_wspace=0.15,
            subfigure_hspace=0.2,
        )
        builder.update_panel("A", subgrid=(1, n_activity_panels), title=None, label="A")

    x_indices = sorted(long_df["x_index"].dropna().astype(int).unique().tolist()) if "x_index" in long_df.columns else []
    c_indices = sorted(long_df["c_index"].dropna().astype(int).unique().tolist()) if "c_index" in long_df.columns else []
    pv_indices = sorted(long_df["pv_index"].dropna().astype(int).unique().tolist()) if "pv_index" in long_df.columns else []
    x_colors = _indexed_palette(x_indices, "crest")
    c_colors = _indexed_palette(c_indices, "rocket")
    pv_colors = _indexed_palette(pv_indices, "flare")

    def _get_stim_pair(name: str) -> tuple[np.ndarray, np.ndarray]:
        default = (np.zeros((1, 1), dtype=float), np.zeros((1, 1), dtype=float))
        pair = STIMULI.get(name, default)
        return _to_np_2d(pair[0]), _to_np_2d(pair[1])

    def _extract_training_signal(value_col: str, index_col: str) -> tuple[np.ndarray, np.ndarray]:
        if training_rows.empty or value_col not in training_rows.columns or index_col not in training_rows.columns:
            return np.asarray([], dtype=float), np.zeros((0, 0), dtype=float)

        pivot = (
            training_rows[["step", index_col, value_col]]
            .drop_duplicates()
            .pivot(index="step", columns=index_col, values=value_col)
            .sort_index()
        )
        if pivot.empty:
            return np.asarray([], dtype=float), np.zeros((0, 0), dtype=float)

        ordered_indices = sorted(pivot.columns.tolist())
        pivot = pivot.reindex(columns=ordered_indices, fill_value=0.0)
        return pivot.index.to_numpy(dtype=float), pivot.to_numpy(dtype=float)

    training_steps, training_X = _extract_training_signal("x_value", "x_index")
    _, training_C = _extract_training_signal("c_value", "c_index")
    if training_steps.size == 0:
        fallback_condition = next((condition for condition in conditions if _is_familiar_condition(condition)), next(iter(STIMULI), None))
        if fallback_condition is not None:
            fallback_X, fallback_C = _get_stim_pair(fallback_condition)
            training_X = fallback_X
            training_C = fallback_C
            training_steps = np.arange(training_X.shape[0], dtype=float)

    training_title = "Training input/context"
    activity_layout = [(condition, phase) for condition in conditions for phase in phases]
    def plot_y(ax_grid, _):
        flat_axes = np.asarray(ax_grid).reshape(-1)
        if flat_axes.size == 0:
            return
        for ax in flat_axes[1:len(activity_layout)]:
            ax.sharey(flat_axes[0])
        for idx, (condition, phase) in enumerate(activity_layout):
            ax = flat_axes[idx]
            cell = y_df[(y_df["experiment_phase"] == phase) & (y_df["condition"] == condition)]
            cell = cell.loc[(cell.step > xlim[0]) & (cell.step < xlim[1])]
            if cell.empty:
                ax.set_visible(False)
                continue
            sns.lineplot(
                data=cell,
                x="step",
                y="y",
                hue="plot_condition",
                hue_order=hue_order,
                style="plot_condition",
                palette=PLOT_COLORS,
                errorbar=None,
                ax=ax,
                legend=(idx == 0),
            )
            ax.set_title(f"{_display_condition_label(condition)} | {PHASE_DISPLAY_LABELS.get(phase, phase.title())}")
            ax.set_xlabel("Time steps")
            ax.set_ylabel("PV activity")
            _style_axis_fonts(ax)
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title(None)
            if idx > 0:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

    def plot_pv(ax_grid, _):
        flat_axes = np.asarray(ax_grid).reshape(-1)
        if flat_axes.size == 0:
            return
        for ax in flat_axes[1:len(activity_layout)]:
            ax.sharey(flat_axes[0])
        for idx, (condition, phase) in enumerate(activity_layout):
            ax = flat_axes[idx]
            cell = pv_df[(pv_df["experiment_phase"] == phase) & (pv_df["condition"] == condition)]
            cell = cell.loc[(cell.step > xlim[0]) & (cell.step < xlim[1])]
            if cell.empty:
                ax.set_visible(False)
                continue
            sns.lineplot(
                data=cell,
                x="step",
                y="pv_value",
                hue="plot_condition",
                hue_order=hue_order,
                palette=PLOT_COLORS,
                style="pv_index",
                errorbar=None,
                ax=ax,
                legend=(idx == 0),
            )
            ax.set_title(f"{_display_condition_label(condition)} | {PHASE_DISPLAY_LABELS.get(phase, phase.title())}")
            ax.set_xlabel("Time steps")
            ax.set_ylabel("PV activity")
            _style_axis_fonts(ax)
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title(None)
            if idx > 0:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

    def plot_panel_a(ax_grid, _):
        _plot_panel_a_activity(
            ax_grid,
            y_df=y_df,
            activity_layout=activity_layout,
            STIMULI=STIMULI,
            xlim=xlim,
            zscore_activity=False,
            include_novel_no_context=include_novel_no_context,
            image_types=image_types,
        )

    def plot_training_activity(ax_grid, _):
        if training_X.size:
            for column_idx in range(training_X.shape[1]):
                ax_grid[0, 0].plot(
                    training_steps,
                    training_X[:, column_idx],
                    color=x_colors.get(column_idx, None),
                    lw=1.5,
                    label=f"x_{column_idx}",
                )
        if training_C.size:
            for column_idx in range(training_C.shape[1]):
                ax_grid[0, 0].plot(
                    training_steps,
                    training_C[:, column_idx],
                    color=c_colors.get(column_idx, None),
                    linestyle="--",
                    lw=1.5,
                    label=f"c_{column_idx}",
                )
        ax_grid[0, 0].set_title(training_title)
        ax_grid[0, 0].set_xlabel("")
        _style_axis_fonts(ax_grid[0, 0])
        ax_grid[0, 0].tick_params(labelbottom=False)

        y_train = training_rows[["step", "y"]].drop_duplicates().groupby("step", as_index=False)["y"].mean()
        pv_train = (
            training_rows[["step", "pv_index", "pv_value"]]
            .drop_duplicates()
            .groupby(["step", "pv_index"], as_index=False)["pv_value"]
            .mean()
        )
        ax_grid[1, 0].plot(y_train["step"], y_train["y"], color="black", lw=1.6, label="y")
        for pv_idx, cell in pv_train.groupby("pv_index", sort=True):
            ax_grid[1, 0].plot(
                cell["step"],
                cell["pv_value"],
                color=pv_colors.get(int(pv_idx), None),
                lw=1.4,
                label=f"pv_{pv_idx}",
            )
        ax_grid[1, 0].set_title("Training Y and PV activity")
        ax_grid[1, 0].set_xlabel("Time steps")
        _style_axis_fonts(ax_grid[1, 0])

    def plot_weight_evolution(ax_grid, _):
        wff = (
            weight_rows[["step", "x_index", "w_ff"]]
            .drop_duplicates()
            .dropna(subset=["w_ff"])
            .sort_values(["x_index", "step"])
        )
        wfb = (
            weight_rows[["step", "c_index", "w_fb"]]
            .drop_duplicates()
            .dropna(subset=["w_fb"])
            .sort_values(["c_index", "step"])
        )
        wlat = (
            weight_rows[["step", "pv_index", "w_lat"]]
            .drop_duplicates()
            .dropna(subset=["w_lat"])
            .sort_values(["pv_index", "step"])
        )
        wpv = (
            weight_rows[["step", "pv_index", "x_index", "W_pv"]]
            .drop_duplicates()
            .dropna(subset=["W_pv"])
            .sort_values(["pv_index", "x_index", "step"])
        )
        wpv["pair"] = "pv" + wpv["pv_index"].astype(str) + "-x" + wpv["x_index"].astype(str)

        sns.lineplot(
            data=wff,
            x="step",
            y="w_ff",
            hue="x_index",
            hue_order=sorted(wff["x_index"].dropna().astype(int).unique().tolist()),
            palette=x_colors,
            errorbar=None,
            ax=ax_grid[0, 0],
        )
        ax_grid[0, 0].set_title("Training w_ff evolution")
        sns.lineplot(
            data=wfb,
            x="step",
            y="w_fb",
            hue="c_index",
            hue_order=sorted(wfb["c_index"].dropna().astype(int).unique().tolist()),
            palette=c_colors,
            errorbar=None,
            ax=ax_grid[1, 0],
        )
        ax_grid[1, 0].set_title("Training w_fb evolution")
        sns.lineplot(
            data=wlat,
            x="step",
            y="w_lat",
            hue="pv_index",
            hue_order=sorted(wlat["pv_index"].dropna().astype(int).unique().tolist()),
            palette=pv_colors,
            errorbar=None,
            ax=ax_grid[2, 0],
        )
        ax_grid[2, 0].set_title("Training w_lat evolution")
        sns.lineplot(data=wpv, x="step", y="W_pv", hue="pair", errorbar=None, ax=ax_grid[3, 0])
        ax_grid[3, 0].set_title("Training W_pv evolution")

        for i in range(ax_grid.shape[0]):
            _style_axis_fonts(ax_grid[i, 0])
            if i < ax_grid.shape[0] - 1:
                ax_grid[i, 0].set_xlabel("")
                ax_grid[i, 0].tick_params(labelbottom=False)

    builder.set_plotter("A", plot_panel_a)
    if full_plots:
        builder.set_plotter("B", plot_pv)
        builder.set_plotter("C", plot_training_activity)
        builder.set_plotter("D", plot_weight_evolution)

    os.makedirs(save_path, exist_ok=True)
    fig, _ = builder.render(save_path=os.path.join(save_path, f"experiment_results_{name}.png"), show=False)
    plt.close(fig)


def visualize_experiment_results(
    DF: DataFrame,
    STIMULI: dict[str, tuple[torch.Tensor, torch.Tensor]],
    save_path: str = PLOTSDIR,
    name: str | None = None,
    include_novel_no_context: bool = False,
    xlim: tuple[float, float] | None = None,
) -> DataFrame:
    long_df = wide_to_long(DF)
    plot_dirs = _resolve_plot_dirs(save_path)

    if "experiment_series" in long_df.columns:
        series_names = long_df["experiment_series"].dropna().unique().tolist()
    else:
        series_names = []

    if not series_names:
        series_names = [None]

    for idx, series_name in enumerate(series_names):
        series_df = long_df if series_name is None else long_df.loc[long_df["experiment_series"].eq(series_name)].copy()
        if series_df.empty:
            continue

        name_suffix = "" if idx == 0 else f"_{series_name}"
        if name is None:
            series_plot_name = name_suffix.removeprefix("_") or None
        else:
            series_plot_name = f"{name}{name_suffix}"
        panel_a_name = f"{series_plot_name}panel_A" if series_plot_name is not None else "panel_A"

        visualize_naive_expert_results(
            series_df,
            STIMULI=STIMULI,
            save_path=plot_dirs["all_panels"],
            name=series_plot_name,
            include_novel_no_context=include_novel_no_context,
            xlim=xlim,
        )
        visualize_naive_expert_results(
            series_df,
            STIMULI=STIMULI,
            save_path=plot_dirs["panel_a"],
            name=panel_a_name,
            full_plots=False,
            include_novel_no_context=include_novel_no_context,
            xlim=xlim,
        )

    return long_df


def wide_to_long(DF: DataFrame) -> DataFrame:
    if "step" not in DF.columns:
        raise ValueError("Input DataFrame must contain a 'step' column.")
    n = len(DF)

    x_idx = sorted(
        int(match.group(1))
        for column in DF.columns
        for match in [re.match(r"^x_(\d+)$", column)]
        if match
    )
    c_idx = sorted(
        int(match.group(1))
        for column in DF.columns
        for match in [re.match(r"^c_(\d+)$", column)]
        if match
    )
    pv_idx = sorted(
        int(match.group(1))
        for column in DF.columns
        for match in [re.match(r"^p_(\d+)$", column)]
        if match
    )
    if not x_idx or not c_idx or not pv_idx:
        return pd.DataFrame(columns=[
            "step", "y", "x_index", "x_value", "w_ff",
            "c_index", "c_value", "w_fb", "pv_index", "pv_value",
            "w_lat", "w_pv_lat", "W_pv", "image_type", "condition", "experiment_phase", "experiment_series", "seed",
        ])

    nx = len(x_idx)
    nc = len(c_idx)
    npv = len(pv_idx)
    rep = nx * nc * npv

    x_vals = DF[[f"x_{idx}" for idx in x_idx]].to_numpy(dtype=float)
    c_vals = DF[[f"c_{idx}" for idx in c_idx]].to_numpy(dtype=float)
    wff_vals = DF[[f"w_ff_{idx}" for idx in x_idx]].to_numpy(dtype=float)
    wfb_vals = DF[[f"w_fb_{idx}" for idx in c_idx]].to_numpy(dtype=float)
    p_vals = DF[[f"p_{idx}" for idx in pv_idx]].to_numpy(dtype=float)
    wlat_vals = DF[[f"w_lat_{idx}" for idx in pv_idx]].to_numpy(dtype=float)
    wpvlat_vals = (
        DF[[f"w_pv_lat_{idx}" for idx in pv_idx]].to_numpy(dtype=float)
        if all(f"w_pv_lat_{idx}" in DF.columns for idx in pv_idx)
        else np.full((n, npv), np.nan, dtype=float)
    )

    wpv_vals = np.full((n, npv, nx), np.nan, dtype=float)
    for pv_position, pv_value in enumerate(pv_idx):
        for x_position, x_value in enumerate(x_idx):
            column_name = f"W_pv_{pv_value}_{x_value}"
            if column_name in DF.columns:
                wpv_vals[:, pv_position, x_position] = DF[column_name].to_numpy(dtype=float)

    step = np.repeat(DF["step"].to_numpy(dtype=float), rep)
    y = np.repeat(DF["y"].to_numpy(dtype=float), rep)

    x_index_grid, c_index_grid, pv_index_grid = np.meshgrid(
        np.array(x_idx, dtype=int),
        np.array(c_idx, dtype=int),
        np.array(pv_idx, dtype=int),
        indexing="ij",
    )
    combo_x_index = x_index_grid.reshape(-1)
    combo_c_index = c_index_grid.reshape(-1)
    combo_pv_index = pv_index_grid.reshape(-1)

    x_value = np.broadcast_to(x_vals[:, :, None, None], (n, nx, nc, npv)).reshape(-1)
    w_ff = np.broadcast_to(wff_vals[:, :, None, None], (n, nx, nc, npv)).reshape(-1)
    c_value = np.broadcast_to(c_vals[:, None, :, None], (n, nx, nc, npv)).reshape(-1)
    w_fb = np.broadcast_to(wfb_vals[:, None, :, None], (n, nx, nc, npv)).reshape(-1)
    pv_value = np.broadcast_to(p_vals[:, None, None, :], (n, nx, nc, npv)).reshape(-1)
    w_lat = np.broadcast_to(wlat_vals[:, None, None, :], (n, nx, nc, npv)).reshape(-1)
    w_pv_lat = np.broadcast_to(wpvlat_vals[:, None, None, :], (n, nx, nc, npv)).reshape(-1)
    W_pv = np.broadcast_to(np.transpose(wpv_vals, (0, 2, 1))[:, :, None, :], (n, nx, nc, npv)).reshape(-1)

    long_df = pd.DataFrame({
        "step": step,
        "y": y,
        "x_index": np.tile(combo_x_index, n),
        "x_value": x_value,
        "w_ff": w_ff,
        "c_index": np.tile(combo_c_index, n),
        "c_value": c_value,
        "w_fb": w_fb,
        "pv_index": np.tile(combo_pv_index, n),
        "pv_value": pv_value,
        "w_lat": w_lat,
        "w_pv_lat": w_pv_lat,
        "W_pv": W_pv,
    })

    if "seed" in DF.columns:
        long_df["seed"] = np.repeat(DF["seed"].to_numpy(), rep)
    if "experiment_series" in DF.columns:
        long_df["experiment_series"] = np.repeat(DF["experiment_series"].astype(str).to_numpy(), rep)

    if "condition" in DF.columns:
        cond = DF["condition"].astype(str).to_numpy()
        cond_rep = np.repeat(cond, rep)
        parts = pd.Series(cond_rep).str.rsplit("_", n=1, expand=True)
        if parts.shape[1] == 2:
            prefix = parts[0].astype(str)
            image_type_token = prefix.str.split("_", n=1).str[0]
            condition_token = prefix.str.split("_", n=1).str[1]
            converted = [
                _condition_token_to_image_type(image_type=image_type, condition_token=condition)
                for image_type, condition in zip(image_type_token, condition_token, strict=False)
            ]
            long_df["experiment_phase"] = parts[1].to_numpy()
            long_df["image_type"] = [image_type for image_type, _ in converted]
            long_df["condition"] = [condition for _, condition in converted]
        else:
            long_df["condition"] = cond_rep

    result_cols = [
        "step", "y", "x_index", "x_value", "w_ff",
        "c_index", "c_value", "w_fb", "pv_index", "pv_value",
        "w_lat", "w_pv_lat", "W_pv", "image_type", "condition", "experiment_phase", "experiment_series", "seed",
    ]
    return long_df[[column for column in result_cols if column in long_df.columns]]
