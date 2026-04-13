# author: Matúš Halák (@matushalak)
import torch
from pandas import DataFrame, concat as pd_concat

from context_contrasting.utils import randn_reparam
from context_contrasting.minimal import PLOTSDIR
from context_contrasting.minimal.config import minimal_configs
from context_contrasting.minimal.minimal import CCNeuron
from context_contrasting.minimal.utils import (build_res, collect_outputs, prepare_collect, 
                                               _rename_phase, _resolve_plots_dir, 
                                               _save_grouped_transition_panels)
from context_contrasting.minimal.visualize import visualize_experiment_results

from joblib import Parallel, delayed
from tqdm import tqdm

PRIMARY_EXPERIMENT_SERIES = "training_familiar"
OCCLUDED_ONLY_EXPERIMENT_SERIES = "training_occluded_only"


def _run_single_config(
    cfg_name: str,
    cfg: dict,
    n_steps_per_phase: int,
)-> tuple[str, DataFrame, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    print(f"Running experiment for config: {cfg_name}")
    df, stimuli = run_experiment(cfg, n_steps_per_phase=n_steps_per_phase)
    return cfg_name, df, stimuli


def design_experimental_phase(input_mean:torch.Tensor, input_var:torch.Tensor,
                              context_mean:torch.Tensor, context_var:torch.Tensor,
                              n_steps:int = 100, n_trials:int | None = 10,
                              intertrial_sigma:float = 0.05
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Example experiment stimulation for using the minimal CCNeuron model.
        Generates random input and context sequences.
    """
    nzeros = 3 * n_steps // 4
    # Generate random input and context sequences according to provided distributions
    X = randn_reparam(size = (n_steps-nzeros,), mu = input_mean, sigma = input_var)
    C = randn_reparam(size = (n_steps-nzeros,), mu = context_mean, sigma = context_var)
    intertrial = randn_reparam(size=(nzeros, *X.shape[1:]), mu=0.0, sigma=intertrial_sigma)
    
    # append a few 0's to indicate initial state
    X = torch.cat((intertrial, X), dim=0)
    C = torch.cat((intertrial, C), dim=0)
    
    if n_trials is not None:
        X = X.repeat((n_trials, 1))
        C = C.repeat((n_trials, 1))
    
    return X, C


def run_experimental_phase(model:CCNeuron, X:torch.Tensor, C:torch.Tensor,
                           condition_name:str = 'default', 
                           update:bool = False, reset_rates:bool = True)->DataFrame:
    """
    Run the model over an experimental sequence.
    """
    # Prepare collections for output data
    data_collection = prepare_collect()
    
    if reset_rates: # reset pyc and pv rates to zero before starting the phase
        model._reset_state()

    # Run the model over the sequence and collect outputs
    for step in range(X.shape[0]):
        x, y_t, y_next, p, c = model(X[step], C[step])
        if update:
            model.update(x, y_t, y_next, p, c)
        
        # Collect the raw tensors
        data_collection = collect_outputs(step, x, y_next, p, c, model, data_collection)
    
    # Make data frame from collected data
    DF:DataFrame = build_res(data_collection, model)
    # broadcast condition name to new column and all rows of dataframe
    DF['condition'] = condition_name
    return DF


def run_experiment(
    model_config:dict,
    n_steps_per_phase:int = 100,
) -> tuple[DataFrame, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    model = CCNeuron(**{key: value for key, value in model_config.items() if not key.startswith("_")})

    # Image 1 ("familiar", trained on)
    X1, C1 = design_experimental_phase(input_mean=[1,0], input_var = 0.05,
                                       context_mean=[1,0], context_var=0.05,
                                       n_steps = n_steps_per_phase)
    
    X1_long, C1_long = design_experimental_phase(input_mean=[1,0], input_var = 0.05,
                                                context_mean=[1,0], context_var=0.05,
                                                n_steps = 2*n_steps_per_phase, n_trials=20)
    
    # Image 2 ("novel", not trained on)
    X2, C2 = design_experimental_phase(input_mean=[0,1], input_var=0.05,
                                       context_mean=[0,1], context_var=0.05,
                                       n_steps = n_steps_per_phase)
    O = torch.zeros_like(X1) # occlusion (no input)
    O_long = torch.zeros_like(X1_long)
    
    STIMULI = {'familiar': (X1, C1), 'novel': (X2, C2)}

    # Initial test on all images without updates
    DF1 = run_experimental_phase(model, X1, C1, condition_name='full_familiar_naive', update=False)
    DF2 = run_experimental_phase(model, X2, C2, condition_name='full_novel_naive', update=False)
    DFO1 = run_experimental_phase(model, O, C1, condition_name='occlusion_familiar_naive', update=False)
    DFO2 = run_experimental_phase(model, O, C2, condition_name='occlusion_novel_naive', update=False)
    DFNn = run_experimental_phase(model, X2, O, condition_name='full_novel_nocontext_naive', update=False)

    # Now run the same sequences again with updates, to see how the model learns
    DF_training_familiar = run_experimental_phase(model, X1, C1, condition_name='full_familiar_training', update=True)
    
    # Now test everything again without changing weights
    DF_familiar = run_experimental_phase(model, X1, C1, condition_name='full_familiar_expert', update=False)
    DF_novel = run_experimental_phase(model, X2, C2, condition_name='full_novel_expert', update=False)
    DFO_familiar = run_experimental_phase(model, O, C1, condition_name='occlusion_familiar_expert', update=False)
    DFO_novel = run_experimental_phase(model, O, C2, condition_name='occlusion_novel_expert', update=False)
    DFNe = run_experimental_phase(model, X2, O, condition_name='full_novel_nocontext_expert', update=False)

    # Continue from the already-trained state, but train only on the occluded familiar stimulus.
    DF_training_familiar_occluded_only = run_experimental_phase(
        model,
        O_long,
        C1_long,
        condition_name='occlusion_familiar_training',
        update=True,
    )

    # Now test everything again without changing weights
    DF_familiar_occluded_only = run_experimental_phase(model, X1, C1, condition_name='full_familiar_expert', update=False)
    DF_novel_occluded_only = run_experimental_phase(model, X2, C2, condition_name='full_novel_expert', update=False)
    DFO_familiar_occluded_only = run_experimental_phase(model, O, C1, condition_name='occlusion_familiar_expert', update=False)
    DFO_novel_occluded_only = run_experimental_phase(model, O, C2, condition_name='occlusion_novel_expert', update=False)
    DFNe_occluded_only = run_experimental_phase(model, X2, O, condition_name='full_novel_nocontext_expert', update=False)

    df = pd_concat(
        [
            DF1.assign(experiment_series=PRIMARY_EXPERIMENT_SERIES),
            DF2.assign(experiment_series=PRIMARY_EXPERIMENT_SERIES),
            DFO1.assign(experiment_series=PRIMARY_EXPERIMENT_SERIES),
            DFO2.assign(experiment_series=PRIMARY_EXPERIMENT_SERIES),
            DFNn.assign(experiment_series=PRIMARY_EXPERIMENT_SERIES),
            DF_training_familiar.assign(experiment_series=PRIMARY_EXPERIMENT_SERIES),
            DF_familiar.assign(experiment_series=PRIMARY_EXPERIMENT_SERIES),
            DF_novel.assign(experiment_series=PRIMARY_EXPERIMENT_SERIES),
            DFO_familiar.assign(experiment_series=PRIMARY_EXPERIMENT_SERIES),
            DFO_novel.assign(experiment_series=PRIMARY_EXPERIMENT_SERIES),
            DFNe.assign(experiment_series=PRIMARY_EXPERIMENT_SERIES),
            DF_familiar.assign(experiment_series=OCCLUDED_ONLY_EXPERIMENT_SERIES),
            DF_novel.assign(experiment_series=OCCLUDED_ONLY_EXPERIMENT_SERIES),
            DFO_familiar.assign(experiment_series=OCCLUDED_ONLY_EXPERIMENT_SERIES),
            DFO_novel.assign(experiment_series=OCCLUDED_ONLY_EXPERIMENT_SERIES),
            DFNe.assign(experiment_series=OCCLUDED_ONLY_EXPERIMENT_SERIES),
            DF_training_familiar_occluded_only.assign(experiment_series=OCCLUDED_ONLY_EXPERIMENT_SERIES),
            _rename_phase(DF_familiar_occluded_only, "expert", "expert2").assign(experiment_series=OCCLUDED_ONLY_EXPERIMENT_SERIES),
            _rename_phase(DF_novel_occluded_only, "expert", "expert2").assign(experiment_series=OCCLUDED_ONLY_EXPERIMENT_SERIES),
            _rename_phase(DFO_familiar_occluded_only, "expert", "expert2").assign(experiment_series=OCCLUDED_ONLY_EXPERIMENT_SERIES),
            _rename_phase(DFO_novel_occluded_only, "expert", "expert2").assign(experiment_series=OCCLUDED_ONLY_EXPERIMENT_SERIES),
            _rename_phase(DFNe_occluded_only, "expert", "expert2").assign(experiment_series=OCCLUDED_ONLY_EXPERIMENT_SERIES),
        ],
        ignore_index=True,
    )
    df['seed'] = model_config['seed']

    return df, STIMULI


if __name__ == "__main__":
    results = Parallel(n_jobs=-1)(
        delayed(_run_single_config)(
            cfg_name,
            cfg,
            400,
        )
        for cfg_name, cfg in minimal_configs.items()
    )
    long_dfs_by_transition: dict[str, DataFrame] = {}
    shared_stimuli = results[0][2] if results else None
    shared_plots_dir = _resolve_plots_dir(next(iter(minimal_configs.values())), 
                                          PLOTSDIR=PLOTSDIR) if minimal_configs else PLOTSDIR

    for cfg_name, df, stimuli in results:
        cfg = minimal_configs[cfg_name]
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
        _save_grouped_transition_panels(
            long_dfs_by_transition,
            stimuli=shared_stimuli,
            save_path=shared_plots_dir,
            transition_order=list(minimal_configs),
        )
