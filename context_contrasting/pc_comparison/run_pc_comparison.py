from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from context_contrasting.paper import model_scatter as paper_scatter
from context_contrasting.paper import transition_templates
from context_contrasting.paper import transitions_helpers as th
from context_contrasting.paper.experiment_s import run_experimental_phase
from context_contrasting.paper.neuron_utils import ThresholdReLU
from context_contrasting.pc_comparison.pc_templates import sample_pc_template_configs
from context_contrasting.pc_comparison.pc_neuron import CorrectPCneuron


PACKAGE_DIR = Path(__file__).resolve().parent
PAPER_DONE_FINAL_FIX = PACKAGE_DIR.parent / "paper" / "done-final-fix"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs"
N_FEATURES = 3

STIMULUS_SPECS = {
    "familiar_1": ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
    "familiar_2": ([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
    "novel": ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
}
IMAGE_INFO = {
    "familiar_1": ("familiar", 1, 1),
    "familiar_2": ("familiar", 2, 2),
    "novel": ("novel", 3, 1),
}
STAGES = {"naive": "Naive", "expert": "Expert"}
TRACE_TYPES = {"full": "Full", "occlusion": "Occl"}

PLOT_STYLE = th.DEFAULT_PLOT_STYLE | {
    "point_size": 13,
    "mean_arrow_width": 1.8,
    "mean_arrow_mutation_scale": 10.5,
}
CLASS_EDGE_COLORS = {"PPE": "#111111", "NPE": "#00a6d6"}
PC_SECTOR_DRAW_ORDER = ("small ∆", "+O axis", "-O axis", "-NO axis", "+NO axis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PPE/NPE comparison panels using only the submitted paper template weights."
    )
    parser.add_argument("--paper-output-dir", type=Path, default=PAPER_DONE_FINAL_FIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-format", choices=("png", "svg", "eps"), default="svg")
    parser.add_argument("--extra-format", choices=("png", "svg", "eps", "none"), default="eps")
    parser.add_argument("--axis-clip-percentile", type=float, default=99.0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--pc-plasticity-mode", choices=("lat", "ppe_ff_npe_fb"), default="ppe_ff_npe_fb")
    parser.add_argument("--copy-reference", action="store_true", default=True)
    parser.add_argument("--no-copy-reference", dest="copy_reference", action="store_false")
    return parser.parse_args()


def _formats(args: argparse.Namespace) -> tuple[str, ...]:
    formats = [args.image_format]
    if args.extra_format != "none" and args.extra_format not in formats:
        formats.append(args.extra_format)
    return tuple(formats)


def _vec(row: pd.Series, prefix: str) -> np.ndarray:
    return np.asarray([float(row[f"{prefix}.mu_{idx}"]) for idx in range(N_FEATURES)], dtype=float)


def _index_vector(row: pd.Series, prefix: str) -> np.ndarray:
    return np.asarray([int(row.get(f"{prefix}_{idx}", 0)) for idx in range(N_FEATURES)], dtype=bool)


def _specific_fb_vector(row: pd.Series) -> np.ndarray:
    sampled = _vec(row, "w_fb_init")
    receives = _index_vector(row, "receives_context")
    if not np.allclose(sampled, sampled[0]):
        return np.where(receives, sampled, 0.0)

    pyc_tuned = _index_vector(row, "tuned_index")
    if not pyc_tuned.any():
        pyc_tuned = receives.copy()
    if not pyc_tuned.any():
        return np.zeros(N_FEATURES, dtype=float)

    fb_level = str(row.get("fb_level", "none"))
    fb_levels = transition_templates.FB_LEVELS
    level = fb_levels.get(fb_level, {})
    none_level = fb_levels.get("none", {})
    tuned = float(level.get("tuned", level.get("center", sampled[0])))
    silent = float(level.get("silent", none_level.get("center", 0.0)))
    return np.where(receives & pyc_tuned, tuned, np.where(receives, silent, 0.0))


def _init_spec(values: np.ndarray | list[float] | tuple[float, ...] | float) -> dict[str, Any]:
    return {"mu": [float(value) for value in np.asarray(values, dtype=float).reshape(-1)], "sigma": 0.0}


def _model_params_from_row(row: pd.Series, *, circuit: str = "PPE", pc_plasticity_mode: str | None = None) -> dict[str, Any]:
    return {
        "w_ff_init": _init_spec(_vec(row, "w_ff_init")),
        "w_fb_init": _init_spec(_specific_fb_vector(row)),
        "W_pv_init": _init_spec(_vec(row, "W_pv_init")),
        "w_lat_init": _init_spec([float(row["w_lat_init.mu_0"])]),
        "w_pv_lat_init": _init_spec([0.0]),
        "receives_context": tuple(bool(row.get(f"receives_context_{idx}", True)) for idx in range(N_FEATURES)),
        "lr_ff": float(row["lr_ff"]),
        "lr_fb": float(row["lr_fb"]),
        "lr_lat": float(row["lr_lat"]),
        "pyc_decay": float(row["pyc_decay"]),
        "pv_decay": float(row["pv_decay"]),
        "baseline_drive_sigma": float(row.get("baseline_drive_sigma", 0.0)),
        "pv_noise_sigma": float(row.get("pv_noise_sigma", 0.0)),
        "activation": ThresholdReLU(
            threshold=transition_templates.SOMA_ACTIVATION_THRESHOLD,
            subtractive=False,
            hasMax=True,
            maxValue=1.0,
        ),
        "seed": int(row.get("seed", 0)),
        "circuit": circuit,
        "pc_plasticity_mode": pc_plasticity_mode or str(row.get("pc_plasticity_mode", "lat")),
    }


def simulate_circuit(
    configs: pd.DataFrame,
    *,
    circuit: str,
    metadata: dict[str, Any],
    n_jobs: int = 1,
    pc_plasticity_mode: str = "ppe_ff_npe_fb",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_samples = int(metadata.get("n_samples_total", metadata.get("requested_n_samples", len(configs))))
    template_configs = sample_pc_template_configs(
        circuit=circuit,
        n_samples=n_samples,
        seed=int(metadata.get("seed", 7151)),
        n_steps_per_phase=int(metadata.get("n_steps_per_phase", 400)),
    )
    template_configs["pc_plasticity_mode"] = pc_plasticity_mode
    response_df, final_weights = simulate_template_weight_circuit(
        template_configs,
        circuit=circuit,
        metadata=metadata,
        n_jobs=n_jobs,
        pc_plasticity_mode=pc_plasticity_mode,
    )
    template_configs = template_configs.merge(
        final_weights,
        on=["circuit", "sample_global_idx", "transition"],
        how="left",
    )
    return response_df, template_configs


def _simulate_template_weight_cell(
    row_dict: dict[str, Any],
    *,
    circuit: str,
    metadata: dict[str, Any],
    test_stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    training_stimuli: tuple[torch.Tensor, torch.Tensor],
    response_tail_fraction: float,
    n_steps_per_phase: int,
    zscore_std_floor: float,
    pc_plasticity_mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    torch.set_num_threads(1)
    row = pd.Series(row_dict)
    model = CorrectPCneuron(_model_params_from_row(row, circuit=circuit, pc_plasticity_mode=pc_plasticity_mode))
    cell_floor = max(
        zscore_std_floor,
        transition_templates.BASELINE_STD_SCALE * float(row.get("baseline_drive_sigma", 0.0)),
    )
    naive_rows, naive_baseline, _ = paper_scatter._probe_rows(
        model,
        test_stimuli,
        phase="naive",
        n_steps_per_phase=n_steps_per_phase,
        response_tail_fraction=response_tail_fraction,
        baseline=None,
        zscore_std_floor=cell_floor,
    )
    run_experimental_phase(model, training_stimuli[0], training_stimuli[1], "full_familiar_training", update=True)
    expert_rows, _, _ = paper_scatter._probe_rows(
        model,
        test_stimuli,
        phase="expert",
        n_steps_per_phase=n_steps_per_phase,
        response_tail_fraction=response_tail_fraction,
        baseline=naive_baseline,
        zscore_std_floor=cell_floor,
    )
    rows: list[dict[str, Any]] = []
    for response_row in naive_rows + expert_rows:
        response_row.update(
            {
                "transition": row["transition"],
                "sample_idx": int(row["sample_idx"]),
                "sample_global_idx": int(row["sample_global_idx"]),
                "seed": int(row["seed"]),
                "circuit": circuit,
                "response_scale": cell_floor,
            }
        )
        rows.append(response_row)
    final_row = {
        "circuit": circuit,
        "sample_global_idx": int(row["sample_global_idx"]),
        "transition": row["transition"],
        "final_w_lat": float(model.w_lat.detach().cpu().reshape(-1)[0]),
        "final_w_ff_0": float(model.w_ff.detach().cpu().reshape(-1)[0]),
        "final_w_ff_1": float(model.w_ff.detach().cpu().reshape(-1)[1]),
        "final_w_ff_2": float(model.w_ff.detach().cpu().reshape(-1)[2]),
        "final_w_fb_0": float(model.w_fb.detach().cpu().reshape(-1)[0]),
        "final_w_fb_1": float(model.w_fb.detach().cpu().reshape(-1)[1]),
        "final_w_fb_2": float(model.w_fb.detach().cpu().reshape(-1)[2]),
    }
    return pd.DataFrame(rows), final_row


def simulate_template_weight_circuit(
    configs: pd.DataFrame,
    *,
    circuit: str,
    metadata: dict[str, Any],
    n_jobs: int = 1,
    pc_plasticity_mode: str = "ppe_ff_npe_fb",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    response_tail_fraction = float(metadata.get("response_tail_fraction", 1.0))
    n_steps_per_phase = int(metadata.get("n_steps_per_phase", 400))
    test_trials = int(metadata.get("test_trials", 5))
    test_stimuli = paper_scatter._build_model_scatter_test_stimuli(
        n_steps_per_phase=n_steps_per_phase,
        n_trials=test_trials,
    )
    training_stimuli = paper_scatter._build_model_scatter_training_stimuli(
        n_steps_per_phase=n_steps_per_phase,
        n_trials=int(metadata.get("training_trials", 7)),
        order=str(metadata.get("training_stimulus_order", "randomized")),
        seed=int(metadata.get("seed", 7151)),
    )
    zscore_std_floor = float(metadata.get("zscore_std_floor", 0.04))

    records = configs.to_dict("records")
    if n_jobs == 1:
        results = [
            _simulate_template_weight_cell(
                row,
                circuit=circuit,
                metadata=metadata,
                test_stimuli=test_stimuli,
                training_stimuli=training_stimuli,
                response_tail_fraction=response_tail_fraction,
                n_steps_per_phase=n_steps_per_phase,
                zscore_std_floor=zscore_std_floor,
                pc_plasticity_mode=pc_plasticity_mode,
            )
            for row in records
        ]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_simulate_template_weight_cell)(
                row,
                circuit=circuit,
                metadata=metadata,
                test_stimuli=test_stimuli,
                training_stimuli=training_stimuli,
                response_tail_fraction=response_tail_fraction,
                n_steps_per_phase=n_steps_per_phase,
                zscore_std_floor=zscore_std_floor,
                pc_plasticity_mode=pc_plasticity_mode,
            )
            for row in records
        )

    response_frames, final_rows = zip(*results, strict=True)
    return pd.concat(response_frames, ignore_index=True), pd.DataFrame(final_rows)


def _transition_table(response_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in response_df.itertuples(index=False):
        image_group, image_idx_original, image_idx_within_group = IMAGE_INFO[row.condition]
        rows.append(
            {
                "transition": row.transition,
                "image_group": image_group,
                "image_idx_original": image_idx_original,
                "image_idx_within_group": image_idx_within_group,
                "neuron_idx": int(row.sample_global_idx),
                "image_type": row.image_type,
                "stage": row.stage,
                "response": float(row.response),
            }
        )
    return pd.DataFrame(rows)


def _wide_table(transition_table: pd.DataFrame) -> pd.DataFrame:
    stage_order = transition_table["stage"].drop_duplicates().tolist()
    wide = (
        transition_table.pivot_table(
            index=[
                "transition",
                "image_group",
                "image_idx_original",
                "image_idx_within_group",
                "neuron_idx",
                "stage",
            ],
            columns="image_type",
            values="response",
            aggfunc="mean",
        )
        .reset_index()
        .rename(columns={"Full": "NO", "Occl": "O"})
    )
    wide["stage"] = pd.Categorical(wide["stage"], categories=stage_order, ordered=True)
    return wide.sort_values(["transition", "image_group", "image_idx_original", "neuron_idx", "stage"]).reset_index(drop=True)


def _summary_with_transition(summary: pd.DataFrame, configs: pd.DataFrame) -> pd.DataFrame:
    transition_by_neuron = configs.set_index("sample_global_idx")["transition"].to_dict()
    summary = summary.copy()
    summary["transition"] = summary["neuron_idx"].map(transition_by_neuron)
    summary["sample_order"] = summary["neuron_idx"].astype(int)
    return summary


def _build_summaries(
    transition_table: pd.DataFrame,
    configs: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, pd.DataFrame]:
    wide = _wide_table(transition_table)
    return {
        group: _summary_with_transition(
            th.build_mean_summary(wide, image_group=group, pre_stage="Naive", target_stage="Expert", threshold=threshold),
            configs,
        )
        for group in ("familiar", "novel")
    }


def _robust_shift_limits(summaries: Iterable[pd.DataFrame], *, hi_percentile: float, pad_ratio: float = 0.12) -> list[float]:
    values = np.concatenate([s[["dNO", "dO"]].to_numpy(dtype=float).reshape(-1) for s in summaries])
    values = np.abs(values[np.isfinite(values)])
    if values.size == 0:
        return [-0.5, 0.5]
    extent = float(np.nanpercentile(values, hi_percentile))
    if not np.isfinite(extent) or extent <= 0.0:
        extent = 0.5
    return [-extent * (1.0 + pad_ratio), extent * (1.0 + pad_ratio)]


def _robust_response_limits(summaries: Iterable[pd.DataFrame], *, hi_percentile: float = 99.0, pad_ratio: float = 0.10) -> list[float]:
    values = np.concatenate(
        [
            s[["NO_Pre", "O_Pre", "NO_Target", "O_Target"]].to_numpy(dtype=float).reshape(-1)
            for s in summaries
            if not s.empty
        ]
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [0.0, 1.0]
    lo = min(0.0, float(np.nanpercentile(values, 1.0)))
    hi = float(np.nanpercentile(values, hi_percentile))
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    pad = pad_ratio * max(hi - lo, 1e-6)
    return [lo - pad, hi + pad]


def _class_edgecolors(summary: pd.DataFrame, *, use_class_outlines: bool) -> str | list[str]:
    if not use_class_outlines or "pc_class" not in summary.columns:
        return "none"
    return [CLASS_EDGE_COLORS.get(str(value), "0.15") for value in summary["pc_class"]]


def _class_linewidths(summary: pd.DataFrame, *, use_class_outlines: bool) -> float | np.ndarray:
    if not use_class_outlines or "pc_class" not in summary.columns:
        return 0.0
    return np.full(len(summary), 0.55, dtype=float)


def _sector_rgba(summary: pd.DataFrame, *, alphas: np.ndarray) -> np.ndarray:
    colors = []
    for sector, alpha in zip(summary["RotatedSector"].astype(str), alphas, strict=False):
        rgb = np.asarray(th.mcolors.to_rgb(th.ROTATED_SECTOR_PALETTE.get(sector, "0.5")))
        colors.append(np.concatenate([rgb, [float(alpha)]]))
    return np.asarray(colors, dtype=float)


def _sector_labels(summary_df: pd.DataFrame) -> dict[str, str]:
    counts = summary_df["RotatedSector"].value_counts().reindex(th.ROTATED_SECTOR_ORDER, fill_value=0)
    total = max(int(counts.sum()), 1)
    return {
        sector: f"{sector} (n={int(count)}, {100.0 * int(count) / total:.1f}%)"
        for sector, count in counts.items()
    }


def _sector_draw_order() -> tuple[str, ...]:
    return PC_SECTOR_DRAW_ORDER


def _draw_vector_axis(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    title: str,
    shift_lims: list[float],
    highlights: pd.DataFrame | None = None,
    show_legend: bool = True,
    use_class_outlines: bool = False,
    legend_outside: bool = False,
) -> None:
    sector_means = th.sector_mean_table(summary)
    sector_arrow_alphas = th._sector_percentage_alphas(summary)
    log_norms = summary["log_dNorm"].to_numpy(dtype=float)
    alphas = th._map_norms_to_alphas(log_norms, min_alpha=PLOT_STYLE["alpha_min"], max_alpha=PLOT_STYLE["alpha_max"])
    sectors = summary["RotatedSector"].astype(str).to_numpy()

    for sector in _sector_draw_order():
        sector_rows = summary.loc[summary["RotatedSector"].astype(str).eq(sector)]
        if sector_rows.empty:
            continue
        pos_idx = np.flatnonzero(sectors == sector)
        rgb = np.asarray(th.mcolors.to_rgb(th.ROTATED_SECTOR_PALETTE[sector])).reshape(1, 3)
        rgba = np.repeat(rgb, len(sector_rows), axis=0)
        rgba = np.concatenate([rgba, alphas[pos_idx].reshape(-1, 1)], axis=1)
        ax.scatter(
            sector_rows["dNO"],
            sector_rows["dO"],
            s=PLOT_STYLE["point_size"],
            c=rgba,
            edgecolors=_class_edgecolors(sector_rows, use_class_outlines=use_class_outlines),
            linewidths=_class_linewidths(sector_rows, use_class_outlines=use_class_outlines),
            zorder=th._sector_scatter_zorder(sector),
        )
        if sector == "small ∆":
            continue
        mean_row = sector_means.loc[sector_means["RotatedSector"].astype(str).eq(sector)]
        if mean_row.empty:
            continue
        mean_row = mean_row.iloc[0]
        th._draw_arrow(
            ax,
            (0.0, 0.0),
            (float(mean_row["dNO"]), float(mean_row["dO"])),
            color=th._darken_color(th.ROTATED_SECTOR_PALETTE[sector]),
            linewidth=max(3.0, PLOT_STYLE["mean_arrow_width"] * 0.9),
            mutation_scale=PLOT_STYLE["mean_arrow_mutation_scale"],
            alpha=sector_arrow_alphas.get(sector, 1.0),
            zorder=4,
        )

    if highlights is not None and not highlights.empty:
        for example in highlights.itertuples(index=False):
            x = float(example.dNO)
            y = float(example.dO)
            x_span = max(abs(float(shift_lims[0])), abs(float(shift_lims[1])))
            y_span = x_span
            offset_rank = max(int(example.display_number) - 1, 0)
            x_sign = -1 if x > 0.65 * x_span else 1
            y_sign = -1 if y > 0.65 * y_span else 1
            x_offset = x_sign * (18 + 4 * offset_rank)
            y_offset = y_sign * (18 + 5 * offset_rank)
            ax.scatter(
                [x],
                [y],
                s=28,
                facecolors=th.ROTATED_SECTOR_PALETTE.get(str(example.sector), "0.35"),
                edgecolors="black",
                linewidths=0.6,
                zorder=20,
            )
            ax.annotate(
                str(int(example.display_number)),
                xy=(x, y),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=6,
                color="0.15",
                arrowprops={"arrowstyle": "-", "color": "0.25", "lw": 0.9, "shrinkA": 1, "shrinkB": 4},
                zorder=21,
            )

    th._draw_origin_guides(ax)
    th._draw_rotated_guides(ax, shift_lims)
    ax.set_xlim(shift_lims)
    ax.set_ylim(shift_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xlabel("dNO", fontsize=7)
    ax.set_ylabel("dO", fontsize=7)
    ax.tick_params(axis="both", labelsize=6, width=0.8, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    if show_legend:
        handles = th._legend_handles(_sector_labels(summary), linewidth=PLOT_STYLE["mean_arrow_width"])
        if use_class_outlines:
            handles.extend(
                [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="none",
                        markerfacecolor="white",
                        markeredgecolor=color,
                        markeredgewidth=1.0,
                        markersize=5,
                        label=pc_class,
                    )
                    for pc_class, color in CLASS_EDGE_COLORS.items()
                ]
            )
        legend_kwargs = (
            {"loc": "upper left", "bbox_to_anchor": (1.02, 1.0)}
            if legend_outside
            else {"loc": "best"}
        )
        ax.legend(
            handles=handles,
            frameon=False,
            fontsize=5,
            handlelength=1.4,
            borderpad=0.2,
            **legend_kwargs,
        )


def _draw_response_axis(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    stage: str,
    title: str,
    response_lims: list[float],
    use_class_outlines: bool,
) -> None:
    no_col = "NO_Pre" if stage == "pre" else "NO_Target"
    o_col = "O_Pre" if stage == "pre" else "O_Target"
    log_norms = summary["log_dNorm"].to_numpy(dtype=float)
    alphas = th._map_norms_to_alphas(log_norms, min_alpha=PLOT_STYLE["alpha_min"], max_alpha=PLOT_STYLE["alpha_max"])
    for sector in _sector_draw_order():
        sector_rows = summary.loc[summary["RotatedSector"].astype(str).eq(sector)]
        if sector_rows.empty:
            continue
        pos_idx = np.flatnonzero(summary["RotatedSector"].astype(str).to_numpy() == sector)
        ax.scatter(
            sector_rows[no_col],
            sector_rows[o_col],
            s=PLOT_STYLE["point_size"],
            c=_sector_rgba(sector_rows, alphas=alphas[pos_idx]),
            edgecolors=_class_edgecolors(sector_rows, use_class_outlines=use_class_outlines),
            linewidths=_class_linewidths(sector_rows, use_class_outlines=use_class_outlines),
            zorder=th._sector_scatter_zorder(sector),
        )
    th._draw_diagonal(ax, response_lims)
    ax.set_xlim(response_lims)
    ax.set_ylim(response_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xlabel("NO", fontsize=7)
    ax.set_ylabel("O", fontsize=7)
    ax.tick_params(axis="both", labelsize=6, width=0.8, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save_naive_expert_vector_plots(
    summaries: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
    basename: str,
    title: str,
    response_lims: list[float],
    shift_lims: list[float],
    formats: tuple[str, ...],
    use_class_outlines: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for group, summary in summaries.items():
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
        _draw_response_axis(
            axes[0],
            summary,
            stage="pre",
            title=f"{group.capitalize()} naive",
            response_lims=response_lims,
            use_class_outlines=use_class_outlines,
        )
        _draw_response_axis(
            axes[1],
            summary,
            stage="target",
            title=f"{group.capitalize()} expert",
            response_lims=response_lims,
            use_class_outlines=use_class_outlines,
        )
        _draw_vector_axis(
            axes[2],
            summary,
            title=f"{group.capitalize()} vector",
            shift_lims=shift_lims,
            show_legend=True,
            use_class_outlines=use_class_outlines,
            legend_outside=True,
        )
        fig.suptitle(title, fontsize=9)
        fig.tight_layout()
        _save_figure(fig, output_dir / f"{basename}_{group}_naive_expert_vector", formats=formats)


def save_familiar_novel_naive_expert_grid(
    summaries: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
    basename: str,
    title: str,
    response_lims: list[float],
    shift_lims: list[float],
    formats: tuple[str, ...],
    use_class_outlines: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(7.35, 4.85))
    for row_idx, group in enumerate(("familiar", "novel")):
        summary = summaries[group].sort_values("neuron_idx").reset_index(drop=True)
        _draw_response_axis(
            axes[row_idx, 0],
            summary,
            stage="pre",
            title=f"{group.capitalize()} naive",
            response_lims=response_lims,
            use_class_outlines=use_class_outlines,
        )
        _draw_response_axis(
            axes[row_idx, 1],
            summary,
            stage="target",
            title=f"{group.capitalize()} expert",
            response_lims=response_lims,
            use_class_outlines=use_class_outlines,
        )
        _draw_vector_axis(
            axes[row_idx, 2],
            summary,
            title=f"{group.capitalize()} vector",
            shift_lims=shift_lims,
            show_legend=(row_idx == 0),
            use_class_outlines=use_class_outlines,
            legend_outside=True,
        )
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    _save_figure(fig, output_dir / f"{basename}_familiar_novel_naive_expert_vector", formats=formats)


def _save_figure(fig: plt.Figure, path: Path, *, formats: tuple[str, ...], dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_vector_plots(
    summaries: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
    basename: str,
    title: str,
    shift_lims: list[float],
    formats: tuple[str, ...],
    highlights: dict[str, pd.DataFrame] | None = None,
    use_class_outlines: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for group, summary in summaries.items():
        fig, ax = plt.subplots(figsize=(2.35, 2.35))
        _draw_vector_axis(
            ax,
            summary,
            title=f"{title} {group}",
            shift_lims=shift_lims,
            highlights=(highlights or {}).get(group),
            show_legend=True,
            use_class_outlines=use_class_outlines,
        )
        _save_figure(fig, output_dir / f"{basename}_{group}_vectors", formats=formats)

    fig, axes = plt.subplots(2, 1, figsize=(2.45, 4.85), sharex=True, sharey=True)
    for ax, group in zip(axes, ("familiar", "novel"), strict=True):
        _draw_vector_axis(
            ax,
            summaries[group],
            title=group.capitalize(),
            shift_lims=shift_lims,
            highlights=(highlights or {}).get(group),
            show_legend=(group == "familiar"),
            use_class_outlines=use_class_outlines,
        )
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    _save_figure(fig, output_dir / f"{basename}_vectors", formats=formats)


def _choose_highlights(
    summaries: dict[str, pd.DataFrame],
    *,
    circuit: str,
    threshold: float,
    max_per_group: int = 3,
) -> dict[str, pd.DataFrame]:
    preferred = {
        "PPE": {
            "familiar": ["-NO axis", "small ∆", "+NO axis", "+O axis"],
            "novel": ["-NO axis", "small ∆", "+NO axis", "+O axis"],
        },
        "NPE": {
            "familiar": ["+O axis", "small ∆", "+NO axis", "-NO axis"],
            "novel": ["+O axis", "small ∆", "+NO axis", "-NO axis"],
        },
    }[circuit]
    selected: dict[str, pd.DataFrame] = {}

    def is_distinct(row: pd.Series, rows: list[pd.Series]) -> bool:
        if not rows:
            return True
        point = np.asarray([float(row["dNO"]), float(row["dO"])])
        previous = np.asarray([[float(prev["dNO"]), float(prev["dO"])] for prev in rows])
        return bool(np.min(np.hypot(*(previous - point).T)) > 0.04)

    for group, summary in summaries.items():
        rows: list[pd.Series] = []
        used: set[int] = set()
        for sector in preferred[group]:
            candidates = summary.loc[
                summary["RotatedSector"].astype(str).eq(sector)
                & ~summary["neuron_idx"].astype(int).isin(used)
            ].copy()
            if sector != "small ∆":
                candidates = candidates.loc[candidates["dNorm"].astype(float) > threshold]
            if candidates.empty:
                continue
            for _, row in candidates.sort_values(["dNorm", "sample_order"], ascending=[False, True]).iterrows():
                if not is_distinct(row, rows):
                    continue
                rows.append(row)
                used.add(int(row["neuron_idx"]))
                break
            if len(rows) >= max_per_group:
                break
        if len(rows) < max_per_group:
            fill = summary.loc[~summary["neuron_idx"].astype(int).isin(used)].sort_values(
                ["dNorm", "sample_order"], ascending=[False, True]
            )
            for _, row in fill.iterrows():
                if float(row["dNorm"]) <= 0.0 or not is_distinct(row, rows):
                    continue
                rows.append(row)
                if len(rows) >= max_per_group:
                    break
        frame = pd.DataFrame(rows).reset_index(drop=True)
        if not frame.empty:
            frame.insert(0, "display_number", np.arange(1, len(frame) + 1))
            frame.insert(0, "image_group", group)
            frame["sector"] = frame["RotatedSector"].astype(str)
            frame["circuit"] = circuit
        selected[group] = frame
    return selected


def _write_summaries(
    summaries: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
    prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fractions = []
    for group, summary in summaries.items():
        summary.to_csv(output_dir / f"{prefix}_{group}_summary.csv", index=False)
        fractions.append(th.sector_fraction_table(summary).assign(circuit=prefix, image_group=group))
    pd.concat(fractions, ignore_index=True).to_csv(output_dir / f"{prefix}_sector_fractions.csv", index=False)


def _with_pc_class(summaries: dict[str, pd.DataFrame], pc_class: str) -> dict[str, pd.DataFrame]:
    return {group: summary.assign(pc_class=pc_class) for group, summary in summaries.items()}


def _combined_pc_summaries(
    ppe_summaries: dict[str, pd.DataFrame],
    npe_summaries: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    combined = {}
    for group in ("familiar", "novel"):
        frame = pd.concat(
            [
                ppe_summaries[group].assign(pc_class="PPE"),
                npe_summaries[group].assign(pc_class="NPE"),
            ],
            ignore_index=True,
        )
        frame["sample_order"] = np.arange(1, len(frame) + 1)
        combined[group] = frame
    return combined


def _comparison_metrics(
    summaries_by_model: dict[str, dict[str, pd.DataFrame]],
    *,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, summaries in summaries_by_model.items():
        for group, summary in summaries.items():
            rows.append(
                {
                    "model": model,
                    "image_group": group,
                    "n": len(summary),
                    "mean_dNO": float(summary["dNO"].mean()),
                    "mean_dO": float(summary["dO"].mean()),
                    "fraction_dNO_positive": float((summary["dNO"] > threshold).mean()),
                    "fraction_dO_positive": float((summary["dO"] > threshold).mean()),
                }
            )
    return pd.DataFrame(rows)


def _copy_reference_outputs(paper_output_dir: Path, output_dir: Path) -> None:
    reference_dir = output_dir / "context_contrasting_reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        paper_output_dir / "figures" / "aggregate_familiar_summary.svg",
        paper_output_dir / "figures" / "aggregate_novel_summary.svg",
        paper_output_dir / "figures" / "aggregate_familiar_summary_panels" / "aggregate_familiar_summary_naive_expert_sector_scatter.svg",
        paper_output_dir / "figures" / "aggregate_novel_summary_panels" / "aggregate_novel_summary_naive_expert_sector_scatter.svg",
        paper_output_dir / "highlight_examples" / "familiar" / "highlighted_familiar_examples.svg",
        paper_output_dir / "highlight_examples" / "novel" / "highlighted_novel_examples.svg",
    ]
    for source in candidates:
        if source.exists():
            shutil.copy2(source, reference_dir / source.name)


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.paper_output_dir / "metadata.json").read_text())
    threshold = float(args.threshold if args.threshold is not None else metadata.get("sector_threshold", 0.3))
    configs = pd.read_csv(args.paper_output_dir / "sampled_config_parameters.csv")
    formats = _formats(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_reference:
        _copy_reference_outputs(args.paper_output_dir, args.output_dir)

    cc_summaries = {
        "familiar": pd.read_csv(args.paper_output_dir / "summaries" / "aggregate_familiar_summary.csv"),
        "novel": pd.read_csv(args.paper_output_dir / "summaries" / "aggregate_novel_summary.csv"),
    }

    response_frames: list[pd.DataFrame] = []
    final_frames: list[pd.DataFrame] = []
    summaries_by_model: dict[str, dict[str, pd.DataFrame]] = {"CC": cc_summaries}
    for circuit in ("PPE", "NPE"):
        response_df, circuit_configs = simulate_circuit(
            configs,
            circuit=circuit,
            metadata=metadata,
            n_jobs=args.n_jobs,
            pc_plasticity_mode=args.pc_plasticity_mode,
        )
        transition = _transition_table(response_df)
        summaries = _build_summaries(transition, circuit_configs, threshold=threshold)
        summaries_by_model[circuit] = summaries

        circuit_dir = args.output_dir / circuit.lower()
        circuit_dir.mkdir(parents=True, exist_ok=True)
        response_df.to_csv(circuit_dir / f"{circuit.lower()}_sample_responses.csv", index=False)
        transition.to_csv(circuit_dir / f"{circuit.lower()}_transition_table.csv", index=False)
        circuit_configs.to_csv(circuit_dir / f"{circuit.lower()}_final_parameters.csv", index=False)
        _write_summaries(summaries, output_dir=args.output_dir / "summaries", prefix=circuit.lower())
        response_frames.append(response_df)
        final_frames.append(circuit_configs)

    shift_lims = _robust_shift_limits(
        [summary for summaries in summaries_by_model.values() for summary in summaries.values()],
        hi_percentile=args.axis_clip_percentile,
    )
    response_lims = _robust_response_limits(
        [
            summary
            for model_name, summaries in summaries_by_model.items()
            if model_name in {"PPE", "NPE"}
            for summary in summaries.values()
        ],
        hi_percentile=args.axis_clip_percentile,
    )
    figures_dir = args.output_dir / "figures"
    save_vector_plots(
        cc_summaries,
        output_dir=figures_dir,
        basename="panel_D_context_contrasting",
        title="Context-contrasting",
        shift_lims=shift_lims,
        formats=formats,
    )

    highlight_tables = []
    for circuit, panel_name, panel_title, example_panel in (
        ("PPE", "panel_E_ppe", "PPE", "panel_H_ppe_examples"),
        ("NPE", "panel_F_npe", "NPE", "panel_I_npe_examples"),
    ):
        highlights = _choose_highlights(summaries_by_model[circuit], circuit=circuit, threshold=threshold)
        save_vector_plots(
            summaries_by_model[circuit],
            output_dir=figures_dir,
            basename=panel_name,
            title=panel_title,
            shift_lims=shift_lims,
            formats=formats,
        )
        save_vector_plots(
            summaries_by_model[circuit],
            output_dir=args.output_dir / "highlight_examples" / circuit.lower(),
            basename=example_panel,
            title=f"{panel_title} examples",
            shift_lims=shift_lims,
            formats=formats,
            highlights=highlights,
        )
        for group, table in highlights.items():
            if table is not None and not table.empty:
                table.to_csv(
                    args.output_dir / "highlight_examples" / circuit.lower() / f"{circuit.lower()}_{group}_highlighted_examples.csv",
                    index=False,
                )
                highlight_tables.append(table)

    ppe_plots = _with_pc_class(summaries_by_model["PPE"], "PPE")
    npe_plots = _with_pc_class(summaries_by_model["NPE"], "NPE")
    combined_pc_plots = _combined_pc_summaries(summaries_by_model["PPE"], summaries_by_model["NPE"])
    save_naive_expert_vector_plots(
        ppe_plots,
        output_dir=figures_dir,
        basename="panel_E_ppe",
        title="PPE",
        response_lims=response_lims,
        shift_lims=shift_lims,
        formats=formats,
    )
    save_familiar_novel_naive_expert_grid(
        ppe_plots,
        output_dir=figures_dir,
        basename="panel_E_ppe",
        title="PPE",
        response_lims=response_lims,
        shift_lims=shift_lims,
        formats=formats,
    )
    save_naive_expert_vector_plots(
        npe_plots,
        output_dir=figures_dir,
        basename="panel_F_npe",
        title="NPE",
        response_lims=response_lims,
        shift_lims=shift_lims,
        formats=formats,
    )
    save_familiar_novel_naive_expert_grid(
        npe_plots,
        output_dir=figures_dir,
        basename="panel_F_npe",
        title="NPE",
        response_lims=response_lims,
        shift_lims=shift_lims,
        formats=formats,
    )
    save_vector_plots(
        combined_pc_plots,
        output_dir=figures_dir,
        basename="panel_EF_pc_combined",
        title="PPE + NPE",
        shift_lims=shift_lims,
        formats=formats,
        use_class_outlines=True,
    )
    save_naive_expert_vector_plots(
        combined_pc_plots,
        output_dir=figures_dir,
        basename="panel_EF_pc_combined",
        title="PPE + NPE",
        response_lims=response_lims,
        shift_lims=shift_lims,
        formats=formats,
        use_class_outlines=True,
    )
    save_familiar_novel_naive_expert_grid(
        combined_pc_plots,
        output_dir=figures_dir,
        basename="panel_EF_pc_combined",
        title="PPE + NPE",
        response_lims=response_lims,
        shift_lims=shift_lims,
        formats=formats,
        use_class_outlines=True,
    )

    if highlight_tables:
        pd.concat(highlight_tables, ignore_index=True).to_csv(
            args.output_dir / "summaries" / "highlighted_examples.csv",
            index=False,
        )

    pd.concat(response_frames, ignore_index=True).to_csv(args.output_dir / "sample_responses.csv", index=False)
    pd.concat(final_frames, ignore_index=True).to_csv(args.output_dir / "final_parameters.csv", index=False)
    _comparison_metrics(summaries_by_model, threshold=threshold).to_csv(
        args.output_dir / "summaries" / "comparison_metrics.csv",
        index=False,
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source_paper_output_dir": str(args.paper_output_dir),
                "source_metadata": {
                    key: metadata.get(key)
                    for key in (
                        "requested_n_samples",
                        "n_samples_total",
                        "seed",
                        "n_steps_per_phase",
                        "test_trials",
                        "training_trials",
                        "training_stimulus_order",
                        "zscore_std_floor",
                        "response_tail_fraction",
                        "sector_threshold",
                    )
                },
                "pc_plasticity_mode": args.pc_plasticity_mode,
                "pc_model": {
                    "source": "context_contrasting.pc_comparison.pc_templates",
                    "templates": {
                        "PPE": "paper-style weight templates: PyC FF tuning is narrow/broad; PV context weights are broad and sampled from paper FB levels.",
                        "NPE": "paper-style weight templates: PyC FB tuning is narrow/broad; PV feedforward weights are broad and sampled from paper PV levels.",
                    },
                    "notes": "PC templates specify initial weights, baseline-drive sigma, and tuning width only; naive/expert shifts are generated by CorrectPCneuron.",
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
