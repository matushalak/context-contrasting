import numpy as np
from pandas import DataFrame, concat as pd_concat
import torch
from context_contrasting.minimal2.visualize_s import (
    TRANSITION_LABELS,
    visualize_transition_response_matrix,
    visualize_transition_panel,
)


def prepare_collect():
    # Collect raw outputs during the loop
    steps = []
    ys = []
    xs = []
    ps = []
    cs = []
    w_ffs = []
    w_fbs = []
    w_lats = []
    w_pv_lats = []
    W_pvs = []
    return steps, ys, xs, ps, cs, w_ffs, w_fbs, w_lats, w_pv_lats, W_pvs

def collect_outputs(step, x, y, p, c, model, collections):
    steps, ys, xs, ps, cs, w_ffs, w_fbs, w_lats, w_pv_lats, W_pvs = collections
    steps.append(step)
    ys.append(y.item())
    # Snapshot immutable copies; avoid NumPy views to tensors that mutate in-place.
    xs.append(x.detach().cpu().numpy().copy())
    ps.append(p.detach().cpu().numpy().copy())
    cs.append(c.detach().cpu().numpy().copy())
    w_ffs.append(model.w_ff.detach().cpu().numpy().copy())
    w_fbs.append(model.w_fb.detach().cpu().numpy().copy())
    w_lats.append(model.w_lat.detach().cpu().numpy().copy())
    w_pv_lats.append(model.w_pv_lat.detach().cpu().numpy().copy())
    W_pvs.append(model.W_pv.detach().cpu().numpy().copy())
    return steps, ys, xs, ps, cs, w_ffs, w_fbs, w_lats, w_pv_lats, W_pvs

def build_res(collections, model, debug=False):
    steps, ys, xs, ps, cs, w_ffs, w_fbs, w_lats, w_pv_lats, W_pvs = collections
    # Build results DataFrame once at the end
    res = {'step': steps, 'y': ys}
    
    # Stack collected arrays
    xs = np.array(xs)
    ps = np.array(ps)
    cs = np.array(cs)
    w_ffs = np.array(w_ffs)
    w_fbs = np.array(w_fbs)
    w_lats = np.array(w_lats)
    w_pv_lats = np.array(w_pv_lats)
    W_pvs = np.array(W_pvs)
    
    if debug: breakpoint()
    
    # Add all columns to results dict
    for i_in in range(model.n_features):
        res[f'x_{i_in}'] = xs[:, i_in]
        res[f'w_ff_{i_in}'] = w_ffs[:, i_in]
    for i_ctxt in range(model.n_context):
        res[f'c_{i_ctxt}'] = cs[:, i_ctxt]
        res[f'w_fb_{i_ctxt}'] = w_fbs[:, i_ctxt]
    for i_pv in range(model.n_pv):
        res[f'p_{i_pv}'] = ps[:, i_pv]
        res[f'w_lat_{i_pv}'] = w_lats[:, i_pv]
        res[f'w_pv_lat_{i_pv}'] = w_pv_lats[:, i_pv]
        for i_pv_in in range(model.n_features):
            res[f'W_pv_{i_pv}_{i_pv_in}'] = W_pvs[:, i_pv, i_pv_in]
    
    return DataFrame(res)


def _resolve_plots_dir(model_config: dict, PLOTSDIR: str) -> str:
    return model_config.get("_plots_dir", PLOTSDIR)


def _rename_phase(df: DataFrame, source_phase: str, target_phase: str) -> DataFrame:
    renamed = df.copy()
    renamed["condition"] = renamed["condition"].astype(str).str.replace(
        f"_{source_phase}",
        f"_{target_phase}",
        regex=False,
    )
    return renamed

def _save_grouped_transition_panels(
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
        ordered_combined = [name for name in transition_order if name in combined_transitions]
        combined_labels = {name: TRANSITION_LABELS.get(name, name) for name in combined_transitions}
        visualize_transition_panel(
            combined_transitions,
            STIMULI=stimuli,
            save_path=save_path,
            name="transition_panel_naive_expert",
            image_mode="both",
            transition_order=ordered_combined,
            transition_labels=combined_labels,
            trace_types=("full", "occlusion"),
            save_in_transition_subdir=save_in_transition_subdir,
        )
        visualize_transition_response_matrix(
            combined_transitions,
            STIMULI=stimuli,
            save_path=save_path,
            name="transitions_FAM",
            image_mode="familiar",
            transition_order=ordered_combined,
            transition_labels=combined_labels,
            save_in_transition_subdir=save_in_transition_subdir,
            save_csv=True,
        )
        visualize_transition_response_matrix(
            combined_transitions,
            STIMULI=stimuli,
            save_path=save_path,
            name="transitions_NOV",
            image_mode="novel",
            transition_order=ordered_combined,
            transition_labels=combined_labels,
            save_in_transition_subdir=save_in_transition_subdir,
            save_csv=True,
        )
