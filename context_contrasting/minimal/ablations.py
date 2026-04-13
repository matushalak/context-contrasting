import os

from joblib import Parallel, delayed
from pandas import DataFrame

from context_contrasting.minimal import PLOT_ABLATIONS_DIR
from context_contrasting.minimal.config import minimal_configs
from context_contrasting.minimal.experiment import run_experiment
from context_contrasting.minimal.utils import _save_grouped_transition_panels
from context_contrasting.minimal.visualize import visualize_naive_expert_results, wide_to_long

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
        return os.path.join(PLOT_ABLATIONS_DIR, label)
    return os.path.join(PLOT_ABLATIONS_DIR, run_group, label)


def _plot_dirs_for_root(root_dir: str) -> dict[str, str]:
    plot_dirs = {
        "root": root_dir,
        "all_panels": os.path.join(root_dir, "all_panels"),
        "transition_panels": os.path.join(root_dir, "transition_panels"),
    }
    for path in plot_dirs.values():
        os.makedirs(path, exist_ok=True)
    return plot_dirs


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
        # "component_ablations": run_component_ablation_studies(
        #     base_configs=base_configs,
        #     n_steps_per_phase=n_steps_per_phase,
        #     include_novel_no_context=include_novel_no_context,
        #     xlim=xlim,
        # ),
        "adaptation_ablations": run_adaptation_ablation_studies(
            base_configs=base_configs,
            n_steps_per_phase=n_steps_per_phase,
            include_novel_no_context=include_novel_no_context,
            xlim=xlim,
        ),
    }


if __name__ == "__main__":
    run_all_ablation_studies()
