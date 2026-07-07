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

from context_contrasting.paper import transitions_helpers as th
from context_contrasting.pc_comparison.pc_neuron import PCNeuron


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
    "pre_point_alpha": 1.0,
    "target_point_alpha": 1.0,
    "shift_point_alpha": 1.0,
    "sector_point_alpha": 0.30,
    "point_size": 13,
    "individual_vector_width": 0.0035,
    "mean_arrow_width": 1.8,
    "mean_arrow_mutation_scale": 10.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PPE/NPE comparison panels from the submitted context-contrasting template population."
    )
    parser.add_argument("--paper-output-dir", type=Path, default=PAPER_DONE_FINAL_FIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-format", choices=("png", "svg", "eps"), default="svg")
    parser.add_argument("--extra-format", choices=("png", "svg", "eps", "none"), default="eps")
    parser.add_argument("--axis-clip-percentile", type=float, default=95.0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--weak-o-threshold", type=float, default=0.3)
    parser.add_argument("--response-scale-multiplier", type=float, default=1.75)
    parser.add_argument("--baseline-std-scale", type=float, default=0.27)
    parser.add_argument("--zscore-std-floor", type=float, default=None)
    parser.add_argument("--heterogeneity-seed", type=int, default=9021)
    parser.add_argument("--pc-gain-jitter", type=float, default=0.55)
    parser.add_argument("--pc-lr-jitter", type=float, default=0.55)
    parser.add_argument("--pc-bias-sd", type=float, default=0.030)
    parser.add_argument("--pc-bias-max", type=float, default=0.090)

    parser.add_argument("--ppe-lr-lat", type=float, default=1.90)
    parser.add_argument("--ppe-ff-gain", type=float, default=1.0)
    parser.add_argument("--ppe-pv-gain", type=float, default=4.6)
    parser.add_argument("--ppe-lat-gain", type=float, default=2.0)
    parser.add_argument("--ppe-bias", type=float, default=0.0)
    parser.add_argument("--ppe-occlusion-ff-leak", type=float, default=0.32)
    parser.add_argument("--ppe-response-max", type=float, default=0.34)

    parser.add_argument("--npe-lr-fb", type=float, default=0.08)
    parser.add_argument("--npe-fb-gain", type=float, default=1.0)
    parser.add_argument("--npe-pv-gain", type=float, default=1.0)
    parser.add_argument("--npe-lat-gain", type=float, default=1.65)
    parser.add_argument("--npe-bias", type=float, default=0.0)
    parser.add_argument("--npe-pv-source", choices=("w_ff", "W_pv"), default="w_ff")
    parser.add_argument("--npe-occlusion-ff-leak", type=float, default=0.10)
    parser.add_argument("--npe-response-max", type=float, default=0.34)
    parser.add_argument("--occlusion-leak-jitter", type=float, default=0.70)

    parser.add_argument(
        "--context-generalization",
        type=float,
        default=None,
        help="Override both circuit-specific inactive context-channel plasticity fractions.",
    )
    parser.add_argument("--ppe-context-generalization", type=float, default=0.0)
    parser.add_argument("--npe-context-generalization", type=float, default=0.20)
    parser.add_argument("--max-weight", type=float, default=5.0)
    parser.add_argument("--copy-reference", action="store_true", default=True)
    parser.add_argument("--no-copy-reference", dest="copy_reference", action="store_false")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _formats(args: argparse.Namespace) -> tuple[str, ...]:
    formats = [args.image_format]
    if args.extra_format != "none" and args.extra_format not in formats:
        formats.append(args.extra_format)
    return tuple(formats)


def _vec(row: pd.Series, prefix: str) -> np.ndarray:
    return np.asarray([float(row[f"{prefix}.mu_{idx}"]) for idx in range(N_FEATURES)], dtype=float)


def _sample_map(configs: pd.DataFrame) -> dict[int, dict[str, Any]]:
    mapping: dict[int, dict[str, Any]] = {}
    for row in configs.to_dict(orient="records"):
        idx = int(row["sample_global_idx"])
        mapping[idx] = {
            "_sample_global_idx": idx,
            "_sample_idx": int(row["sample_idx"]),
            "_canonical_transition": str(row["transition"]),
            "_highlight_candidate_order": idx,
        }
    return mapping


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


def _response_scale(row: pd.Series, *, zscore_std_floor: float, baseline_std_scale: float, multiplier: float) -> float:
    baseline_sigma = float(row.get("baseline_drive_sigma", 0.0))
    scale = max(float(zscore_std_floor), baseline_std_scale * baseline_sigma)
    return max(scale * multiplier, 1e-9)


def _row_rng(row: pd.Series, circuit: str, args: argparse.Namespace) -> np.random.Generator:
    circuit_offset = 10_000 if circuit == "PPE" else 20_000
    seed = int(args.heterogeneity_seed) + circuit_offset + int(row["sample_global_idx"]) * 17
    return np.random.default_rng(seed)


def _lognormal_factor(rng: np.random.Generator, sigma: float) -> float:
    if sigma <= 0.0:
        return 1.0
    return float(np.exp(rng.normal(0.0, sigma)))


def _cell_bias(center: float, rng: np.random.Generator, args: argparse.Namespace) -> float:
    if args.pc_bias_sd <= 0.0:
        return float(np.clip(center, 0.0, args.pc_bias_max))
    return float(np.clip(center + rng.normal(0.0, args.pc_bias_sd), 0.0, args.pc_bias_max))


def _cell_occlusion_leak(base: float, rng: np.random.Generator, args: argparse.Namespace) -> float:
    value = base * _lognormal_factor(rng, args.occlusion_leak_jitter)
    return float(np.clip(value, 0.0, 0.55))


def _pc_parameters(row: pd.Series, circuit: str, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, float]]:
    w_ff = _vec(row, "w_ff_init")
    w_fb = _vec(row, "w_fb_init")
    w_lat = float(row["w_lat_init.mu_0"])
    rng = _row_rng(row, circuit, args)
    gain = {
        "ff": _lognormal_factor(rng, args.pc_gain_jitter),
        "fb": _lognormal_factor(rng, args.pc_gain_jitter),
        "pv": _lognormal_factor(rng, args.pc_gain_jitter),
        "lat": _lognormal_factor(rng, args.pc_gain_jitter),
        "lr": _lognormal_factor(rng, args.pc_lr_jitter),
    }
    if circuit == "PPE":
        params = {
            "w_ff": w_ff,
            "w_fb": w_fb,
            "w_pv_fb": w_fb,
            "w_lat": w_lat,
            "lr_lat": args.ppe_lr_lat * gain["lr"],
            "ff_gain": args.ppe_ff_gain * gain["ff"],
            "pv_gain": args.ppe_pv_gain * gain["pv"],
            "lat_gain": args.ppe_lat_gain * gain["lat"],
            "bias": _cell_bias(args.ppe_bias, rng, args),
            "max_weight": args.max_weight,
            "response_max": args.ppe_response_max,
        }
        factors = gain | {"bias": params["bias"], "occlusion_ff_leak": _cell_occlusion_leak(args.ppe_occlusion_ff_leak, rng, args)}
        return params, factors

    w_pv_ff = w_ff if args.npe_pv_source == "w_ff" else _vec(row, "W_pv_init")
    params = {
        "w_ff": w_ff,
        "w_fb": w_fb,
        "w_pv_ff": w_pv_ff,
        "w_lat": w_lat,
        "lr_fb": args.npe_lr_fb * gain["lr"],
        "fb_gain": args.npe_fb_gain * gain["fb"],
        "pv_gain": args.npe_pv_gain * gain["pv"],
        "lat_gain": args.npe_lat_gain * gain["lat"],
        "bias": _cell_bias(args.npe_bias, rng, args),
        "max_weight": args.max_weight,
        "response_max": args.npe_response_max,
    }
    factors = gain | {"bias": params["bias"], "occlusion_ff_leak": _cell_occlusion_leak(args.npe_occlusion_ff_leak, rng, args)}
    return params, factors


def _plasticity_context(context: np.ndarray, generalization: float) -> np.ndarray:
    if generalization <= 0.0:
        return context
    if generalization > 1.0:
        raise ValueError("--context-generalization must be <= 1.")
    learned = np.full_like(context, float(generalization))
    learned[context > 0.0] = 1.0
    return learned


def _training_order(metadata: dict[str, Any]) -> list[str]:
    order = metadata.get("training_trial_order")
    if order:
        return [str(name) for name in order]
    training_trials = int(metadata.get("training_trials", 7))
    familiar = [name for name in STIMULUS_SPECS if name.startswith("familiar")]
    return [name for _ in range(training_trials) for name in familiar]


def _circuit_context_generalization(circuit: str, args: argparse.Namespace) -> float:
    if args.context_generalization is not None:
        return float(args.context_generalization)
    return float(args.ppe_context_generalization if circuit == "PPE" else args.npe_context_generalization)


def simulate_circuit(
    configs: pd.DataFrame,
    *,
    circuit: str,
    args: argparse.Namespace,
    metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    zscore_std_floor = float(args.zscore_std_floor if args.zscore_std_floor is not None else metadata.get("zscore_std_floor", 0.04))
    training_order = _training_order(metadata)
    context_generalization = _circuit_context_generalization(circuit, args)

    stimulus_vectors = {
        name: (np.asarray(ff, dtype=float), np.asarray(context, dtype=float))
        for name, (ff, context) in STIMULUS_SPECS.items()
    }

    for _, row in configs.iterrows():
        params, factors = _pc_parameters(row, circuit, args)
        neuron = PCNeuron(int(row["sample_global_idx"]), circuit, params)
        scale = _response_scale(
            row,
            zscore_std_floor=zscore_std_floor,
            baseline_std_scale=args.baseline_std_scale,
            multiplier=args.response_scale_multiplier,
        )

        def probe(phase: str) -> None:
            for condition, (ff_full, context) in stimulus_vectors.items():
                ff_occluded = factors["occlusion_ff_leak"] * context
                for trace, ff_input in (("full", ff_full), ("occlusion", ff_occluded)):
                    response = neuron.activate(ff_input, context) / scale
                    rows.append(
                        {
                            "condition": condition,
                            "phase": phase,
                            "stage": STAGES[phase],
                            "trace": trace,
                            "image_type": TRACE_TYPES[trace],
                            "response": response,
                            "raw_response": response * scale,
                            "response_scale": scale,
                            "transition": row["transition"],
                            "sample_idx": int(row["sample_idx"]),
                            "sample_global_idx": int(row["sample_global_idx"]),
                            "seed": int(row["seed"]),
                            "circuit": circuit,
                        }
                    )

        probe("naive")
        for condition in training_order:
            ff_full, context = stimulus_vectors[condition]
            neuron.train(
                ff_full,
                context,
                plasticity_context=_plasticity_context(context, context_generalization),
                repetitions=1,
            )
        probe("expert")

        info = neuron.get_info()
        final_rows.append(
            {
                "circuit": circuit,
                "sample_global_idx": int(row["sample_global_idx"]),
                "transition": row["transition"],
                "final_w_lat": info["w_lat"],
                "cell_bias": factors["bias"],
                "cell_ff_gain_factor": factors["ff"],
                "cell_fb_gain_factor": factors["fb"],
                "cell_pv_gain_factor": factors["pv"],
                "cell_lat_gain_factor": factors["lat"],
                "cell_lr_factor": factors["lr"],
                "cell_occlusion_ff_leak": factors["occlusion_ff_leak"],
                **{f"final_w_fb_{idx}": value for idx, value in enumerate(info["w_fb"])},
            }
        )

    response_df = pd.DataFrame(rows)
    final_df = pd.DataFrame(final_rows)
    return response_df, final_df


def _robust_shift_limits(summaries: Iterable[pd.DataFrame], *, hi_percentile: float, pad_ratio: float = 0.12) -> list[float]:
    values = np.concatenate([s[["dNO", "dO"]].to_numpy(dtype=float).reshape(-1) for s in summaries])
    values = np.abs(values[np.isfinite(values)])
    if values.size == 0:
        return [-0.5, 0.5]
    extent = float(np.nanpercentile(values, hi_percentile))
    if not np.isfinite(extent) or extent <= 0.0:
        extent = 0.5
    extent *= 1.0 + pad_ratio
    return [-extent, extent]


def _sector_labels(summary_df: pd.DataFrame) -> dict[str, str]:
    counts = summary_df["RotatedSector"].value_counts().reindex(th.ROTATED_SECTOR_ORDER, fill_value=0)
    total = max(int(counts.sum()), 1)
    return {
        sector: f"{sector} (n={int(count)}, {100.0 * int(count) / total:.1f}%)"
        for sector, count in counts.items()
    }


def _draw_vector_axis(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    title: str,
    shift_lims: list[float],
    highlights: pd.DataFrame | None = None,
    show_legend: bool = True,
) -> None:
    sector_means = th.sector_mean_table(summary)
    sector_arrow_alphas = th._sector_percentage_alphas(summary)
    log_norms = summary["log_dNorm"].to_numpy(dtype=float)
    alphas = th._map_norms_to_alphas(log_norms, min_alpha=PLOT_STYLE["alpha_min"], max_alpha=PLOT_STYLE["alpha_max"])
    sectors = summary["RotatedSector"].astype(str).to_numpy()

    for sector in th._sector_plot_order(small_delta_first=True):
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
            edgecolors="none",
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
            ax.scatter(
                [float(example.dNO)],
                [float(example.dO)],
                s=28,
                facecolors=th.ROTATED_SECTOR_PALETTE.get(str(example.sector), "0.35"),
                edgecolors="black",
                linewidths=0.6,
                zorder=20,
            )
            ax.annotate(
                str(int(example.display_number)),
                xy=(float(example.dNO), float(example.dO)),
                xytext=(18 if int(example.display_number) % 2 else -18, 18),
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
        ax.legend(handles=handles, frameon=False, loc="best", fontsize=5, handlelength=1.4, borderpad=0.2)


def _save_figure(fig: plt.Figure, path: Path, *, formats: tuple[str, ...], dpi: int = 300) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        saved.append(out)
    plt.close(fig)
    return saved


def save_vector_plots(
    summaries: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
    basename: str,
    title: str,
    shift_lims: list[float],
    formats: tuple[str, ...],
    highlights: dict[str, pd.DataFrame] | None = None,
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
        )
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    _save_figure(fig, output_dir / f"{basename}_vectors", formats=formats)


def _choose_highlights(
    summaries: dict[str, pd.DataFrame],
    *,
    circuit: str,
    threshold: float,
    max_per_group: int = 5,
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
            row = candidates.sort_values(["dNorm", "sample_order"], ascending=[False, True]).iloc[0]
            rows.append(row)
            used.add(int(row["neuron_idx"]))
            if len(rows) >= max_per_group:
                break
        if len(rows) < max_per_group:
            fill = summary.loc[~summary["neuron_idx"].astype(int).isin(used)].sort_values(
                ["dNorm", "sample_order"], ascending=[False, True]
            )
            for _, row in fill.head(max_per_group - len(rows)).iterrows():
                rows.append(row)
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
        fractions.append(
            th.sector_fraction_table(summary).assign(
                circuit=prefix,
                image_group=group,
            )
        )
    pd.concat(fractions, ignore_index=True).to_csv(output_dir / f"{prefix}_sector_fractions.csv", index=False)


def _comparison_metrics(
    summaries_by_model: dict[str, dict[str, pd.DataFrame]],
    *,
    threshold: float,
    weak_o_threshold: float,
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
                    "fraction_NO_amplified_weak_O": float(
                        ((summary["dNO"] > threshold) & (summary["O_Target"] < weak_o_threshold)).mean()
                    ),
                    "fraction_target_NO_dominant_mixed": float(
                        (
                            (summary["NO_Target"] > weak_o_threshold)
                            & (summary["O_Target"] > weak_o_threshold)
                            & (summary["NO_Target"] > summary["O_Target"])
                        ).mean()
                    ),
                    "fraction_target_O_dominant_mixed": float(
                        (
                            (summary["NO_Target"] > weak_o_threshold)
                            & (summary["O_Target"] > weak_o_threshold)
                            & (summary["O_Target"] >= summary["NO_Target"])
                        ).mean()
                    ),
                    "fraction_target_O_positive": float((summary["O_Target"] > weak_o_threshold).mean()),
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _load_json(args.paper_output_dir / "metadata.json")
    threshold = float(args.threshold if args.threshold is not None else metadata.get("sector_threshold", 0.3))
    configs = pd.read_csv(args.paper_output_dir / "sampled_config_parameters.csv")
    formats = _formats(args)

    if args.copy_reference:
        _copy_reference_outputs(args.paper_output_dir, args.output_dir)

    cc_summaries = {
        "familiar": pd.read_csv(args.paper_output_dir / "summaries" / "aggregate_familiar_summary.csv"),
        "novel": pd.read_csv(args.paper_output_dir / "summaries" / "aggregate_novel_summary.csv"),
    }

    response_frames = []
    final_frames = []
    summaries_by_model: dict[str, dict[str, pd.DataFrame]] = {"CC": cc_summaries}
    for circuit in ("PPE", "NPE"):
        response_df, final_df = simulate_circuit(configs, circuit=circuit, args=args, metadata=metadata)
        transition = _transition_table(response_df)
        summaries = _build_summaries(transition, configs, threshold=threshold)
        summaries_by_model[circuit] = summaries

        circuit_dir = args.output_dir / circuit.lower()
        circuit_dir.mkdir(parents=True, exist_ok=True)
        response_df.to_csv(circuit_dir / f"{circuit.lower()}_sample_responses.csv", index=False)
        transition.to_csv(circuit_dir / f"{circuit.lower()}_transition_table.csv", index=False)
        final_df.to_csv(circuit_dir / f"{circuit.lower()}_final_parameters.csv", index=False)
        _write_summaries(summaries, output_dir=args.output_dir / "summaries", prefix=circuit.lower())
        response_frames.append(response_df)
        final_frames.append(final_df)

    shift_lims = _robust_shift_limits(
        [summary for summaries in summaries_by_model.values() for summary in summaries.values()],
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

    if highlight_tables:
        pd.concat(highlight_tables, ignore_index=True).to_csv(
            args.output_dir / "summaries" / "highlighted_examples.csv",
            index=False,
        )

    metrics = _comparison_metrics(
        summaries_by_model,
        threshold=threshold,
        weak_o_threshold=args.weak_o_threshold,
    )
    metrics.to_csv(args.output_dir / "summaries" / "comparison_metrics.csv", index=False)
    pd.concat(response_frames, ignore_index=True).to_csv(args.output_dir / "sample_responses.csv", index=False)
    pd.concat(final_frames, ignore_index=True).to_csv(args.output_dir / "final_parameters.csv", index=False)

    run_metadata = {
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
                "sector_threshold",
            )
        },
        "pc_parameters": {
            "threshold": threshold,
            "weak_o_threshold": args.weak_o_threshold,
            "response_scale_multiplier": args.response_scale_multiplier,
            "baseline_std_scale": args.baseline_std_scale,
            "zscore_std_floor": args.zscore_std_floor if args.zscore_std_floor is not None else metadata.get("zscore_std_floor"),
            "heterogeneity_seed": args.heterogeneity_seed,
            "pc_gain_jitter": args.pc_gain_jitter,
            "pc_lr_jitter": args.pc_lr_jitter,
            "pc_bias_sd": args.pc_bias_sd,
            "pc_bias_max": args.pc_bias_max,
            "ppe_lr_lat": args.ppe_lr_lat,
            "ppe_ff_gain": args.ppe_ff_gain,
            "ppe_pv_gain": args.ppe_pv_gain,
            "ppe_lat_gain": args.ppe_lat_gain,
            "ppe_bias": args.ppe_bias,
            "ppe_occlusion_ff_leak": args.ppe_occlusion_ff_leak,
            "ppe_response_max": args.ppe_response_max,
            "npe_lr_fb": args.npe_lr_fb,
            "npe_fb_gain": args.npe_fb_gain,
            "npe_pv_gain": args.npe_pv_gain,
            "npe_lat_gain": args.npe_lat_gain,
            "npe_bias": args.npe_bias,
            "npe_pv_source": args.npe_pv_source,
            "npe_occlusion_ff_leak": args.npe_occlusion_ff_leak,
            "npe_response_max": args.npe_response_max,
            "occlusion_leak_jitter": args.occlusion_leak_jitter,
            "context_generalization_override": args.context_generalization,
            "ppe_context_generalization": args.ppe_context_generalization,
            "npe_context_generalization": args.npe_context_generalization,
            "max_weight": args.max_weight,
        },
        "outputs": {
            "figures_dir": str(figures_dir),
            "metrics": str(args.output_dir / "summaries" / "comparison_metrics.csv"),
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(run_metadata, indent=2))


if __name__ == "__main__":
    main()
