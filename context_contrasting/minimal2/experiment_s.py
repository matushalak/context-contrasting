# author: Matúš Halák (@matushalak)
from contextlib import contextmanager

import torch
from joblib import Parallel, delayed
from pandas import DataFrame, concat as pd_concat

from context_contrasting.minimal2 import PLOTSDIR
from context_contrasting.minimal2.config_s import minimal_configs3 as minimal_configs_s
from context_contrasting.minimal2.minimal_s import CCNeuron
from context_contrasting.minimal2.utils import (
    build_res,
    collect_outputs,
    prepare_collect,
    _resolve_plots_dir,
)
from context_contrasting.minimal2.visualize_s import (
    save_grouped_transition_panels,
    visualize_experiment_results,
)
from context_contrasting.utils import randn_reparam

PRIMARY_EXPERIMENT_SERIES = "training_familiar"

PLOTSDIR = PLOTSDIR+"experiment_s"

STIMULUS_SPECS = {
    "familiar_1": ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
    "familiar_2": ([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
    "novel": ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
}

NO_RESPONSE_ABLATION_SPECS = {
    "no_context": {"condition_prefix": "nocontext", "zero_context": True},
    "nolat": {"condition_prefix": "nolat", "model_overrides": {"use_lat_connection": False}},
    "no_context_nolat": {
        "condition_prefix": "nocontextnolat",
        "zero_context": True,
        "model_overrides": {"use_lat_connection": False},
    },
}


def _run_single_config(
    cfg_name: str,
    cfg: dict,
    n_steps_per_phase: int,
) -> tuple[str, DataFrame, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    print(f"Running experiment for config: {cfg_name}")
    df, stimuli = run_experiment(cfg, n_steps_per_phase=n_steps_per_phase)
    return cfg_name, df, stimuli


def design_experimental_phase(
    input_mean: torch.Tensor | list[float],
    input_var: float | list[float],
    context_mean: torch.Tensor | list[float],
    context_var: float | list[float],
    n_steps: int = 100,
    n_trials: int | None = 10,
    intertrial_sigma: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one repeated stimulus/context phase."""
    nzeros = 3 * n_steps // 4
    X = randn_reparam(size=(n_steps - nzeros,), mu=input_mean, sigma=input_var)
    C = randn_reparam(size=(n_steps - nzeros,), mu=context_mean, sigma=context_var)
    intertrial = randn_reparam(size=(nzeros, *X.shape[1:]), mu=0.0, sigma=intertrial_sigma)

    X = torch.cat((intertrial, X), dim=0)
    C = torch.cat((intertrial, C), dim=0)

    if n_trials is not None:
        X = X.repeat((n_trials, 1))
        C = C.repeat((n_trials, 1))

    return X, C


def _combine_experimental_phases(
    phases: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    n_trials: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.cat([phase[0] for phase in phases], dim=0)
    C = torch.cat([phase[1] for phase in phases], dim=0)
    if n_trials is not None:
        X = X.repeat((n_trials, 1))
        C = C.repeat((n_trials, 1))
    return X, C


@contextmanager
def _temporary_model_overrides(model: CCNeuron, **overrides: bool):
    originals = {name: getattr(model, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(model, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(model, name, value)


def run_experimental_phase(
    model: CCNeuron,
    X: torch.Tensor,
    C: torch.Tensor,
    condition_name: str = "default",
    update: bool = False,
    reset_rates: bool = True,
) -> DataFrame:
    data_collection = prepare_collect()

    if reset_rates:
        model._reset_state()

    for step in range(X.shape[0]):
        x, y_t, y_next, p, c = model(X[step], C[step])
        if update:
            model.update(x, y_t, y_next, p, c)
        data_collection = collect_outputs(step, x, y_next, p, c, model, data_collection)

    df = build_res(data_collection, model)
    df["condition"] = condition_name
    return df


def _run_test_phase_variants(
    model: CCNeuron,
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    phase_label: str,
) -> list[DataFrame]:
    frames: list[DataFrame] = []

    for condition_name, (X, C) in stimuli.items():
        occluded_X = torch.zeros_like(X)
        no_context_C = torch.zeros_like(C)

        frames.append(
            run_experimental_phase(model, X, C, condition_name=f"full_{condition_name}_{phase_label}", update=False)
        )
        frames.append(
            run_experimental_phase(
                model,
                occluded_X,
                C,
                condition_name=f"occlusion_{condition_name}_{phase_label}",
                update=False,
            )
        )

        for ablation_label, ablation_spec in NO_RESPONSE_ABLATION_SPECS.items():
            ablated_C = no_context_C if ablation_spec.get("zero_context", False) else C
            model_overrides = ablation_spec.get("model_overrides", {})
            condition_prefix = ablation_spec.get("condition_prefix", ablation_label)
            with _temporary_model_overrides(model, **model_overrides):
                frames.append(
                    run_experimental_phase(
                        model,
                        X,
                        ablated_C,
                        condition_name=f"{condition_prefix}_{condition_name}_{phase_label}",
                        update=False,
                    )
                )

    return frames


def _build_test_stimuli(
    *,
    n_steps_per_phase: int,
    n_trials: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {
        name: design_experimental_phase(
            input_mean=input_mean,
            input_var=0.05,
            context_mean=context_mean,
            context_var=0.05,
            n_steps=n_steps_per_phase,
            n_trials=n_trials,
        )
        for name, (input_mean, context_mean) in STIMULUS_SPECS.items()
    }


def _build_training_stimuli(
    *,
    n_steps_per_phase: int,
    n_trials: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    single_trial_familiars = [
        design_experimental_phase(
            input_mean=input_mean,
            input_var=0.05,
            context_mean=context_mean,
            context_var=0.05,
            n_steps=n_steps_per_phase,
            n_trials=None,
        )
        for name, (input_mean, context_mean) in STIMULUS_SPECS.items()
        if name.startswith("familiar")
    ]
    return _combine_experimental_phases(single_trial_familiars, n_trials=n_trials)


def run_experiment(
    model_config: dict,
    n_steps_per_phase: int = 100,
) -> tuple[DataFrame, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    model = CCNeuron(**{key: value for key, value in model_config.items() if not key.startswith("_")})

    stimuli = _build_test_stimuli(n_steps_per_phase=n_steps_per_phase, n_trials=10)
    training_X, training_C = _build_training_stimuli(n_steps_per_phase=n_steps_per_phase, n_trials=5)

    naive_frames = _run_test_phase_variants(model, stimuli, phase_label="naive")

    training_frame = run_experimental_phase(
        model,
        training_X,
        training_C,
        condition_name="full_familiar_training",
        update=True,
    )

    expert_frames = _run_test_phase_variants(model, stimuli, phase_label="expert")

    df = pd_concat(
        [
            *(frame.assign(experiment_series=PRIMARY_EXPERIMENT_SERIES) for frame in naive_frames),
            training_frame.assign(experiment_series=PRIMARY_EXPERIMENT_SERIES),
            *(frame.assign(experiment_series=PRIMARY_EXPERIMENT_SERIES) for frame in expert_frames),
        ],
        ignore_index=True,
    )
    df["seed"] = model_config["seed"]

    return df, stimuli


if __name__ == "__main__":
    results = Parallel(n_jobs=-1)(
        delayed(_run_single_config)(
            cfg_name,
            cfg,
            400,
        )
        for cfg_name, cfg in minimal_configs_s.items()
    )
    long_dfs_by_transition: dict[str, DataFrame] = {}
    shared_stimuli = results[0][2] if results else None
    shared_plots_dir = _resolve_plots_dir(
        next(iter(minimal_configs_s.values())),
        PLOTSDIR=PLOTSDIR,
    ) if minimal_configs_s else PLOTSDIR

    for cfg_name, df, stimuli in results:
        cfg = minimal_configs_s[cfg_name]
        long_df = visualize_experiment_results(
            df,
            STIMULI=stimuli,
            save_path=_resolve_plots_dir(cfg, PLOTSDIR=PLOTSDIR),
            name=cfg_name,
            include_novel_no_context=True,
            xlim=(1000, 1400),
        )
        long_dfs_by_transition[cfg_name] = long_df

    if shared_stimuli is not None and long_dfs_by_transition:
        save_grouped_transition_panels(
            long_dfs_by_transition,
            stimuli=shared_stimuli,
            save_path=shared_plots_dir,
            transition_order=list(minimal_configs_s),
        )
