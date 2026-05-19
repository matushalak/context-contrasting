import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas import DataFrame

from context_contrasting.minimal2 import PLOT_ABLATIONS_DIR
from context_contrasting.minimal2.config_s import minimal_configs3 as minimal_configs
from context_contrasting.minimal2.experiment_s import PRIMARY_EXPERIMENT_SERIES, run_experiment
from context_contrasting.minimal2.visualize_s import (
    PHASE_DISPLAY_LABELS,
    TRANSITION_LABELS,
    _collect_shared_baseline_stats,
    _expand_window_to_event_bounds,
    _is_familiar_condition,
    _summarize_windowed_repeated_zscored_trace,
    save_grouped_transition_panels as _save_grouped_transition_panels,
    visualize_naive_expert_results,
    wide_to_long,
)

SIMPLE_PLOT_ABLATIONS_DIR = PLOT_ABLATIONS_DIR.replace('plots', 'plotsexperiment_s')

ABLATION_COMPONENTS = (
    "use_FF_connection",
    "FF_plasticity",
    "use_FB_connection",
    "FB_plasticity",
    "use_lat_connection",
    "lat_plasticity",
    "use_pv_lat_connection",
    "pv_lat_plasticity",
    "use_pv_connection",
    "pv_plasticity",
)

ABLATION_COMPONENT_ALIASES = {
    "use_pv_lat": "use_pv_lat_connection",
}

ADAPTATION_ABLATION_COMPONENTS = {
    "all_adaptation_plasticity": (
        "FF_plasticity",
        "lat_plasticity",
        "pv_lat_plasticity",
        "pv_plasticity",
    ),
    "all_lat_plasticity": (
        "lat_plasticity",
        "pv_lat_plasticity",
    ),
    "non_pv_adaptation_plasticity": (
        "FF_plasticity",
        "lat_plasticity",
        "pv_lat_plasticity",
    ),
    "all_non_pv": (
        "FF_plasticity",
        "lat_plasticity",
        "use_pv_lat_connection",
        "pv_lat_plasticity",
    ),
}

SUMMARY_RUN_GROUP = "familiar_occlusion_minus_full_imshow"
SUMMARY_PHASES = ("naive", "expert")
SUMMARY_TRACE_TYPES = ("full", "occlusion")
SUMMARY_STEP_WINDOW = (1000, 1400)
FULL_MODEL_LABEL = "full_model"
SUMMARY_ABLATION_LABELS = {
    FULL_MODEL_LABEL: "Full model",
    "use_FF_connection": "No FF conn.",
    "FF_plasticity": "No FF plast.",
    "use_FB_connection": "No FB conn.",
    "FB_plasticity": "No FB plast.",
    "use_lat_connection": "No LAT conn.",
    "lat_plasticity": "No LAT plast.",
    "use_pv_lat_connection": "No PY->PV conn.",
    "pv_lat_plasticity": "No PY->PV plast.",
    "use_pv_connection": "No FF->PV conn.",
    "pv_plasticity": "No FF->PV plast.",
    "all_adaptation_plasticity": "No all adapt. plast.",
    "all_lat_plasticity": "No LAT adapt. plast.",
    "non_pv_adaptation_plasticity": "No non-PV adapt. plast.",
    "all_non_pv": "No all non-PV",
}
SUMMARY_TRANSITION_LABELS = {
    **TRANSITION_LABELS,
    "FF_FB_broad_novel": "FF -> FB\n(broad nov)",
    "FF_FB_narrow_familiar_novel": "FF -> FB\n(narrow fam+nov)",
}


def _copy_init_dict(init_dict: dict) -> dict:
    return {
        "mu": init_dict["mu"],
        "sigma": init_dict["sigma"],
    }


def _normalize_ablation_config(config: dict) -> dict:
    normalized = config.copy()
    normalized.setdefault("w_pv_lat_init", _copy_init_dict(normalized["w_lat_init"]))
    return normalized


def _normalize_component_names(components: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized = tuple(ABLATION_COMPONENT_ALIASES.get(component, component) for component in components)
    unknown = [component for component in normalized if component not in ABLATION_COMPONENTS]
    if unknown:
        raise ValueError(f"Unknown ablation component(s): {', '.join(unknown)}")
    return normalized


def _ablation_root_dir(label: str, run_group: str | None = None) -> str:
    if run_group is None:
        return os.path.join(SIMPLE_PLOT_ABLATIONS_DIR, label)
    return os.path.join(SIMPLE_PLOT_ABLATIONS_DIR, run_group, label)


def _plot_dirs_for_root(root_dir: str) -> dict[str, str]:
    plot_dirs = {
        "root": root_dir,
        "all_panels": os.path.join(root_dir, "all_panels"),
        "transition_panels": os.path.join(root_dir, "transition_panels"),
    }
    for path in plot_dirs.values():
        os.makedirs(path, exist_ok=True)
    return plot_dirs


def _summary_root_dir(run_group: str = SUMMARY_RUN_GROUP) -> str:
    root_dir = os.path.join(SIMPLE_PLOT_ABLATIONS_DIR, run_group)
    os.makedirs(root_dir, exist_ok=True)
    return root_dir


def _summary_ablation_specs() -> list[tuple[str, tuple[str, ...]]]:
    return (
        [(FULL_MODEL_LABEL, ())]
        + [(component, (component,)) for component in ABLATION_COMPONENTS]
        + list(ADAPTATION_ABLATION_COMPONENTS.items())
    )


def _summary_ablation_label(label: str) -> str:
    return SUMMARY_ABLATION_LABELS.get(label, label.replace("_", " "))


def _summary_transition_label(name: str) -> str:
    return SUMMARY_TRANSITION_LABELS.get(name, name.replace("_", " "))


def _primary_series_df(long_df: DataFrame) -> DataFrame:
    if "experiment_series" not in long_df.columns:
        return long_df.copy()
    primary_df = long_df.loc[long_df["experiment_series"].eq(PRIMARY_EXPERIMENT_SERIES)].copy()
    return primary_df if not primary_df.empty else long_df.copy()


def _summary_familiar_conditions(stimuli: dict[str, tuple]) -> list[str]:
    return [name for name in stimuli if _is_familiar_condition(name)]


def _summary_plot_window(
    stimuli: dict[str, tuple],
    familiar_conditions: list[str],
    step_window: tuple[int, int],
) -> tuple[float, float]:
    display_windows = [
        _expand_window_to_event_bounds(stimuli[condition], focus_window=step_window)
        for condition in familiar_conditions
        if condition in stimuli
    ]
    if not display_windows:
        return step_window
    return (
        min(window[0] for window in display_windows),
        max(window[1] for window in display_windows),
    )


def _stimulus_epoch_mean(summary: dict[str, np.ndarray | float | int]) -> float:
    x_seconds = np.asarray(summary["x_seconds"], dtype=float)
    y_mean = np.asarray(summary["y_mean"], dtype=float)
    stim_start, stim_end = summary["stim_seconds"]
    mask = (x_seconds >= float(stim_start)) & (x_seconds < float(stim_end))
    if not np.any(mask):
        return float("nan")
    return float(np.nanmean(y_mean[mask]))


def _build_component_summary_tidy(
    *,
    ablation_label: str,
    long_dfs_by_transition: dict[str, DataFrame],
    stimuli: dict[str, tuple],
    transition_order: list[str],
    step_window: tuple[int, int],
) -> DataFrame:
    familiar_conditions = _summary_familiar_conditions(stimuli)
    if not familiar_conditions:
        raise ValueError("Expected at least one familiar condition in stimuli.")

    plot_window = _summary_plot_window(
        stimuli=stimuli,
        familiar_conditions=familiar_conditions,
        step_window=step_window,
    )
    records: list[dict[str, object]] = []

    for transition_name in transition_order:
        long_df = long_dfs_by_transition.get(transition_name)
        if long_df is None:
            continue
        primary_df = _primary_series_df(long_df)

        for phase in SUMMARY_PHASES:
            baseline_stats = _collect_shared_baseline_stats(
                primary_df,
                trace_specs=[
                    (condition, phase, trace_type)
                    for condition in familiar_conditions
                    for trace_type in SUMMARY_TRACE_TYPES
                ],
                stimuli=stimuli,
                focus_window=plot_window,
            )
            condition_values: list[float] = []

            for condition in familiar_conditions:
                full_summary = _summarize_windowed_repeated_zscored_trace(
                    primary_df,
                    condition=condition,
                    phase=phase,
                    image_type="full",
                    stim_pair=stimuli[condition],
                    focus_window=plot_window,
                    baseline_stats=baseline_stats,
                )
                occlusion_summary = _summarize_windowed_repeated_zscored_trace(
                    primary_df,
                    condition=condition,
                    phase=phase,
                    image_type="occlusion",
                    stim_pair=stimuli[condition],
                    focus_window=plot_window,
                    baseline_stats=baseline_stats,
                )
                if full_summary is None or occlusion_summary is None:
                    continue

                condition_values.append(
                    _stimulus_epoch_mean(occlusion_summary) - _stimulus_epoch_mean(full_summary)
                )

            records.append(
                {
                    "ablation": ablation_label,
                    "ablation_label": _summary_ablation_label(ablation_label),
                    "transition": transition_name,
                    "transition_label": _summary_transition_label(transition_name),
                    "phase": phase,
                    "phase_label": PHASE_DISPLAY_LABELS.get(phase, phase.title()),
                    "value": float(np.mean(condition_values)) if condition_values else np.nan,
                    "n_familiar_conditions": len(condition_values),
                }
            )

    return pd.DataFrame.from_records(records)


def _plot_component_summary_imshow(
    summary_df: DataFrame,
    *,
    row_order: list[str],
    transition_order: list[str],
    out_path: str,
) -> None:
    column_specs = [(transition, phase) for transition in transition_order for phase in SUMMARY_PHASES]
    matrix = np.full((len(row_order), len(column_specs)), np.nan, dtype=float)

    for row_idx, ablation_label in enumerate(row_order):
        for col_idx, (transition_name, phase) in enumerate(column_specs):
            match = summary_df.loc[
                summary_df["ablation"].eq(ablation_label)
                & summary_df["transition"].eq(transition_name)
                & summary_df["phase"].eq(phase),
                "value",
            ]
            if not match.empty:
                matrix[row_idx, col_idx] = float(match.iloc[0])

    finite_values = matrix[np.isfinite(matrix)]
    vmax = float(np.max(np.abs(finite_values))) if finite_values.size else 1.0
    if vmax <= 0:
        vmax = 1.0

    fig_width = max(18.0, 1.05 * len(column_specs))
    fig_height = max(6.0, 0.7 * len(row_order) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)

    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels([_summary_ablation_label(label) for label in row_order], fontsize=12)
    ax.set_xticks(np.arange(len(column_specs)))
    ax.set_xticklabels(
        [PHASE_DISPLAY_LABELS.get(phase, phase.title()) for _, phase in column_specs],
        rotation=90,
        fontsize=10,
    )
    ax.set_ylabel("Ablation")
    ax.set_xlabel("Phase")
    ax.set_title("Familiar mean stimulus response: occluded - nonoccluded")

    for separator_idx in range(2, len(column_specs), 2):
        ax.axvline(separator_idx - 0.5, color="white", lw=1.4, alpha=0.9)

    for separator_idx in range(1, len(row_order)):
        ax.axhline(separator_idx - 0.5, color="white", lw=0.8, alpha=0.45)

    top_ax = ax.secondary_xaxis("top")
    top_ax.set_xticks([idx * 2 + 0.5 for idx in range(len(transition_order))])
    top_ax.set_xticklabels(
        [_summary_transition_label(name) for name in transition_order],
        fontsize=10,
    )
    top_ax.tick_params(length=0, pad=8)

    colorbar = fig.colorbar(im, ax=ax, shrink=0.94, pad=0.02)
    colorbar.set_label("Mean z-scored response (O - NO)")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _run_summary_study(
    label: str,
    disabled_components: tuple[str, ...] | list[str],
    *,
    run_group: str,
    base_configs: dict[str, dict] | None,
    n_steps_per_phase: int,
) -> tuple[dict[str, DataFrame], dict]:
    plots_root = _ablation_root_dir(label, run_group=run_group)
    configs = build_ablation_configs(
        minimal_configs if base_configs is None else base_configs,
        disabled_components=disabled_components,
        plots_dir=plots_root,
    )
    results = Parallel(n_jobs=-1)(
        delayed(_run_single_ablation_config)(cfg_name, cfg, label, n_steps_per_phase)
        for cfg_name, cfg in configs.items()
    )

    long_dfs_by_transition = {
        cfg_name: wide_to_long(df)
        for cfg_name, df, _ in results
    }
    shared_stimuli = results[0][2] if results else {}
    return long_dfs_by_transition, shared_stimuli


def build_ablation_configs(
    base_configs: dict[str, dict],
    disabled_components: tuple[str, ...] | list[str],
    plots_dir: str,
) -> dict[str, dict]:
    normalized_components = _normalize_component_names(disabled_components)
    configs: dict[str, dict] = {}

    for config_name, config in base_configs.items():
        ablated_config = {
            **_normalize_ablation_config(config),
            "_plots_dir": plots_dir,
        }
        for component in normalized_components:
            ablated_config[component] = False
        configs[config_name] = ablated_config

    return configs


def _run_single_ablation_config(
    cfg_name: str,
    cfg: dict,
    label: str,
    n_steps_per_phase: int,
) -> tuple[str, DataFrame, dict]:
    print(f"Running ablation set {label} for config: {cfg_name}")
    df, stimuli = run_experiment(cfg, n_steps_per_phase=n_steps_per_phase)
    return cfg_name, df, stimuli


def _plot_ablation_results(
    results: list[tuple[str, DataFrame, dict]],
    plots_root: str,
    include_novel_no_context: bool,
    xlim: tuple[float, float],
) -> dict[str, DataFrame]:
    plot_dirs = _plot_dirs_for_root(plots_root)
    long_dfs_by_transition: dict[str, DataFrame] = {}
    shared_stimuli = results[0][2] if results else None

    for cfg_name, df, stimuli in results:
        long_df = wide_to_long(df)
        long_dfs_by_transition[cfg_name] = long_df

        if "experiment_series" not in long_df.columns:
            continue

        for series_name in long_df["experiment_series"].dropna().unique().tolist():
            series_df = long_df.loc[long_df["experiment_series"].eq(series_name)].copy()
            if series_df.empty:
                continue
            visualize_naive_expert_results(
                series_df,
                STIMULI=stimuli,
                save_path=plot_dirs["all_panels"],
                name=f"{cfg_name}_{series_name}",
                full_plots=True,
                include_novel_no_context=include_novel_no_context,
                xlim=xlim,
            )

    if shared_stimuli is not None and long_dfs_by_transition:
        _save_grouped_transition_panels(
            long_dfs_by_transition,
            stimuli=shared_stimuli,
            save_path=plot_dirs["transition_panels"],
            transition_order=list(minimal_configs),
            save_in_transition_subdir=False,
        )

    return long_dfs_by_transition


def run_ablation_study(
    label: str,
    disabled_components: tuple[str, ...] | list[str],
    *,
    run_group: str | None = None,
    base_configs: dict[str, dict] | None = None,
    n_steps_per_phase: int = 400,
    include_novel_no_context: bool = True,
    xlim: tuple[float, float] = (1000, 1400),
) -> dict[str, DataFrame]:
    plots_root = _ablation_root_dir(label, run_group=run_group)
    configs = build_ablation_configs(
        minimal_configs if base_configs is None else base_configs,
        disabled_components=disabled_components,
        plots_dir=plots_root,
    )

    results = Parallel(n_jobs=-1)(
        delayed(_run_single_ablation_config)(cfg_name, cfg, label, n_steps_per_phase)
        for cfg_name, cfg in configs.items()
    )

    return _plot_ablation_results(
        results,
        plots_root=plots_root,
        include_novel_no_context=include_novel_no_context,
        xlim=xlim,
    )


def run_component_ablation_studies(
    base_configs: dict[str, dict] | None = None,
    n_steps_per_phase: int = 400,
    include_novel_no_context: bool = True,
    xlim: tuple[float, float] = (1000, 1400),
) -> dict[str, dict[str, DataFrame]]:
    all_results: dict[str, dict[str, DataFrame]] = {}
    for component in ABLATION_COMPONENTS:
        all_results[component] = run_ablation_study(
            component,
            disabled_components=(component,),
            base_configs=base_configs,
            n_steps_per_phase=n_steps_per_phase,
            include_novel_no_context=include_novel_no_context,
            xlim=xlim,
        )
    return all_results


def run_adaptation_ablation_studies(
    base_configs: dict[str, dict] | None = None,
    n_steps_per_phase: int = 400,
    include_novel_no_context: bool = True,
    xlim: tuple[float, float] = (1000, 1400),
) -> dict[str, dict[str, DataFrame]]:
    all_results: dict[str, dict[str, DataFrame]] = {}
    for label, components in ADAPTATION_ABLATION_COMPONENTS.items():
        all_results[label] = run_ablation_study(
            label,
            disabled_components=components,
            run_group="adaptation_ablations",
            base_configs=base_configs,
            n_steps_per_phase=n_steps_per_phase,
            include_novel_no_context=include_novel_no_context,
            xlim=xlim,
        )
    return all_results


def run_all_ablation_studies(
    base_configs: dict[str, dict] | None = None,
    n_steps_per_phase: int = 400,
    include_novel_no_context: bool = True,
    xlim: tuple[float, float] = (1000, 1400),
) -> dict[str, dict[str, dict[str, DataFrame]]]:
    return {
        "component_ablations": run_component_ablation_studies(
            base_configs=base_configs,
            n_steps_per_phase=n_steps_per_phase,
            include_novel_no_context=include_novel_no_context,
            xlim=xlim,
        ),
        "adaptation_ablations": run_adaptation_ablation_studies(
            base_configs=base_configs,
            n_steps_per_phase=n_steps_per_phase,
            include_novel_no_context=include_novel_no_context,
            xlim=xlim,
        ),
    }


def run_component_ablation_occlusion_minus_full_imshow(
    *,
    base_configs: dict[str, dict] | None = None,
    n_steps_per_phase: int = 400,
    step_window: tuple[int, int] = SUMMARY_STEP_WINDOW,
    run_group: str = SUMMARY_RUN_GROUP,
) -> DataFrame:
    summary_root = _summary_root_dir(run_group=run_group)
    configs = minimal_configs if base_configs is None else base_configs
    transition_order = list(configs)
    row_order = [label for label, _ in _summary_ablation_specs()]

    summary_frames: list[DataFrame] = []
    for label, disabled_components in _summary_ablation_specs():
        long_dfs_by_transition, stimuli = _run_summary_study(
            label,
            disabled_components=disabled_components,
            run_group=run_group,
            base_configs=configs,
            n_steps_per_phase=n_steps_per_phase,
        )
        summary_frames.append(
            _build_component_summary_tidy(
                ablation_label=label,
                long_dfs_by_transition=long_dfs_by_transition,
                stimuli=stimuli,
                transition_order=transition_order,
                step_window=step_window,
            )
        )

    summary_df = pd.concat(summary_frames, ignore_index=True)
    tidy_csv_path = os.path.join(summary_root, "familiar_occlusion_minus_full_summary.csv")
    summary_df.to_csv(tidy_csv_path, index=False)

    matrix_df = (
        summary_df.assign(column_key=summary_df["transition"] + "__" + summary_df["phase"])
        .pivot(index="ablation", columns="column_key", values="value")
        .reindex(index=row_order, columns=[f"{transition}__{phase}" for transition in transition_order for phase in SUMMARY_PHASES])
    )
    matrix_df.index.name = "ablation"
    matrix_df.to_csv(os.path.join(summary_root, "familiar_occlusion_minus_full_summary_matrix.csv"))

    _plot_component_summary_imshow(
        summary_df,
        row_order=row_order,
        transition_order=transition_order,
        out_path=os.path.join(summary_root, "familiar_occlusion_minus_full_summary.png"),
    )
    return summary_df


if __name__ == "__main__":
    run_all_ablation_studies()
