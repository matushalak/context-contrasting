from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from context_contrasting.paper import model_scatter as paper_scatter
from context_contrasting.paper import transition_templates
from context_contrasting.paper.experiment_s import run_experimental_phase
from context_contrasting.paper.minimal_divisive import CCNeuron
from context_contrasting.paper.neuron_utils import ThresholdReLU
from context_contrasting.pc_comparison import pc_templates
from context_contrasting.pc_comparison.pc_neuron import CorrectPCneuron


PACKAGE_DIR = Path(__file__).resolve().parent
PAPER_DIR = PACKAGE_DIR.parent / "paper"
DEFAULT_PAPER_OUTPUT_DIR = PAPER_DIR / "done-final-fix"
DEFAULT_OUTPUT_DIR = PAPER_DIR / "paper-timed-weight-sweeps"
N_FEATURES = 3
FAMILIAR_FEATURES = (0, 1)
NOVEL_FEATURE = 2
SweepName = Literal["ff_fb", "ff_lat", "fb_lat"]


@dataclass(frozen=True)
class TuningProfile:
    name: str
    label: str
    mask: tuple[float, float, float]
    tuned_feature: int | None


@dataclass(frozen=True)
class CCSweepSpec:
    name: SweepName
    x_weight: str
    y_weight: str
    fixed_weight: str
    output_stem: str
    title: str


PROFILES: tuple[TuningProfile, ...] = (
    TuningProfile("broad_3of3", "broad PyC tuning (3/3)", (1.0, 1.0, 1.0), None),
    TuningProfile("narrow_familiar_1of3", "narrow PyC tuning: familiar 1 (1/3)", (1.0, 0.0, 0.0), 0),
    TuningProfile("narrow_novel_1of3", "narrow PyC tuning: novel (1/3)", (0.0, 0.0, 1.0), NOVEL_FEATURE),
)

CC_SWEEPS: tuple[CCSweepSpec, ...] = (
    CCSweepSpec("ff_fb", "w_ff", "w_fb", "w_lat", "cc_ff_fb", "CC: vary w_FF and w_FB"),
    CCSweepSpec("ff_lat", "w_ff", "w_lat", "w_fb", "cc_ff_lat", "CC: vary w_FF and w_LAT"),
    CCSweepSpec("fb_lat", "w_fb", "w_lat", "w_ff", "cc_fb_lat", "CC: vary w_FB and w_LAT"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-timed CC/PPE/NPE initial-weight grid sweeps.")
    parser.add_argument("--paper-output-dir", type=Path, default=DEFAULT_PAPER_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-min", type=float, default=0.05)
    parser.add_argument("--grid-max", type=float, default=0.95)
    parser.add_argument("--grid-size", type=int, default=11)
    parser.add_argument("--fixed-weight", type=float, default=0.5)
    parser.add_argument("--pv-input-weight", type=float, default=0.5)
    parser.add_argument("--untuned-weight", type=float, default=0.0)
    parser.add_argument("--baseline-drive-sigma", type=float, default=0.12)
    parser.add_argument("--pv-noise-sigma", type=float, default=0.0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--image-format", choices=("png", "svg", "eps"), default="png")
    parser.add_argument("--run", choices=("all", "cc", "pc"), default="all")
    return parser.parse_args()


def _metadata(args: argparse.Namespace) -> dict[str, Any]:
    metadata = json.loads((args.paper_output_dir / "metadata.json").read_text())
    return {
        "seed": int(metadata.get("seed", 7151)),
        "n_steps_per_phase": int(metadata.get("n_steps_per_phase", 400)),
        "test_trials": int(metadata.get("test_trials", 5)),
        "training_trials": int(metadata.get("training_trials", 7)),
        "training_stimulus_order": str(metadata.get("training_stimulus_order", "randomized")),
        "response_tail_fraction": float(metadata.get("response_tail_fraction", 1.0)),
        "zscore_std_floor": float(metadata.get("zscore_std_floor", 0.04)),
        "sector_threshold": float(metadata.get("sector_threshold", 0.3)),
    }


def _init_spec(values: np.ndarray | list[float] | tuple[float, ...] | float) -> dict[str, Any]:
    return {"mu": [float(v) for v in np.asarray(values, dtype=float).reshape(-1)], "sigma": 0.0}


def _profile_values(value: float, profile: TuningProfile, untuned: float) -> np.ndarray:
    mask = np.asarray(profile.mask, dtype=float)
    return mask * float(value) + (1.0 - mask) * float(untuned)


def _paper_stimuli(meta: dict[str, Any]) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]:
    test = paper_scatter._build_model_scatter_test_stimuli(
        n_steps_per_phase=int(meta["n_steps_per_phase"]),
        n_trials=int(meta["test_trials"]),
    )
    train = paper_scatter._build_model_scatter_training_stimuli(
        n_steps_per_phase=int(meta["n_steps_per_phase"]),
        n_trials=int(meta["training_trials"]),
        order=str(meta["training_stimulus_order"]),
        seed=int(meta["seed"]),
    )
    return test, train


def _probe_model(model: Any, meta: dict[str, Any], test_stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> tuple[dict[str, dict[str, float]], tuple[float, float]]:
    rows, baseline, _ = paper_scatter._probe_rows(
        model,
        test_stimuli,
        phase="naive",
        n_steps_per_phase=int(meta["n_steps_per_phase"]),
        response_tail_fraction=float(meta["response_tail_fraction"]),
        baseline=None,
        zscore_std_floor=float(meta["zscore_std_floor"]),
    )
    return _rows_to_responses(rows), baseline


def _probe_model_with_baseline(
    model: Any,
    meta: dict[str, Any],
    test_stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    baseline: tuple[float, float],
) -> dict[str, dict[str, float]]:
    rows, _, _ = paper_scatter._probe_rows(
        model,
        test_stimuli,
        phase="expert",
        n_steps_per_phase=int(meta["n_steps_per_phase"]),
        response_tail_fraction=float(meta["response_tail_fraction"]),
        baseline=baseline,
        zscore_std_floor=float(meta["zscore_std_floor"]),
    )
    return _rows_to_responses(rows)


def _rows_to_responses(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    frame = pd.DataFrame(rows)
    pivot = frame.pivot_table(index="condition", columns="image_type", values="response", aggfunc="mean")
    familiar = pivot.loc[["familiar_1", "familiar_2"]].mean(axis=0)
    novel = pivot.loc["novel"]
    return {
        "familiar": {"NO": float(familiar["Full"]), "O": float(familiar["Occl"])},
        "novel": {"NO": float(novel["Full"]), "O": float(novel["Occl"])},
    }


def _transition_flags(row: dict[str, Any], threshold: float) -> None:
    row["target_familiar_less_NO_more_O"] = bool(row["familiar_dNO"] < -threshold and row["familiar_dO"] > threshold)
    row["bonus_with_increased_novel_NO"] = bool(row["target_familiar_less_NO_more_O"] and row["novel_dNO"] > threshold)


def _fill_response_columns(row: dict[str, Any], pre: dict[str, dict[str, float]], post: dict[str, dict[str, float]]) -> None:
    for group in ("familiar", "novel"):
        for response_type in ("NO", "O"):
            row[f"{group}_pre_{response_type}"] = pre[group][response_type]
            row[f"{group}_post_{response_type}"] = post[group][response_type]
    row["familiar_dNO"] = post["familiar"]["NO"] - pre["familiar"]["NO"]
    row["familiar_dO"] = post["familiar"]["O"] - pre["familiar"]["O"]
    row["novel_dNO"] = post["novel"]["NO"] - pre["novel"]["NO"]
    row["novel_dO"] = post["novel"]["O"] - pre["novel"]["O"]


def _cc_values(spec: CCSweepSpec, x_value: float, y_value: float, fixed_value: float) -> dict[str, float]:
    values = {"w_ff": fixed_value, "w_fb": fixed_value, "w_lat": fixed_value}
    values[spec.x_weight] = x_value
    values[spec.y_weight] = y_value
    return values


def _make_cc_model(spec: CCSweepSpec, profile: TuningProfile, x_value: float, y_value: float, args: argparse.Namespace, meta: dict[str, Any]) -> CCNeuron:
    values = _cc_values(spec, x_value, y_value, float(args.fixed_weight))
    rates = transition_templates._effective_learning_rates(int(meta["n_steps_per_phase"]))
    return CCNeuron(
        n_features=N_FEATURES,
        n_pv=1,
        n_context=N_FEATURES,
        activation=ThresholdReLU(threshold=transition_templates.SOMA_ACTIVATION_THRESHOLD, subtractive=False, hasMax=True, maxValue=1.0),
        lr_ff=float(rates["lr_ff"]),
        lr_fb=float(rates["lr_fb"]),
        lr_lat=float(rates["lr_lat"]),
        lr_pv=0.0,
        w_ff_init=_init_spec(_profile_values(values["w_ff"], profile, float(args.untuned_weight))),
        w_fb_init=_init_spec([values["w_fb"]] * N_FEATURES),
        w_lat_init=_init_spec([values["w_lat"]]),
        w_pv_lat_init=_init_spec([float(args.pv_input_weight)]),
        W_pv_init=_init_spec([float(args.pv_input_weight)] * N_FEATURES),
        receives_context=(True, True, True),
        FFrule="anti-Hebbian",
        FBrule="dampened-anti-Hebbian",
        pyc_decay=transition_templates.BASE_CONFIG["pyc_decay"],
        pv_decay=transition_templates.BASE_CONFIG["pv_decay"],
        apical_drive_threshold=transition_templates.WIDTH_CLASSES["broad"]["drive"],
        apical_drive_subtractive=transition_templates.APICAL_DRIVE_SUBTRACTIVE,
        apical_gain_strength=transition_templates.WIDTH_CLASSES["broad"]["gain"],
        apical_gain_k=transition_templates.BASE_CONFIG["apical_gain_k"],
        apical_gain_threshold=transition_templates.SHARED_GAIN_THRESHOLD,
        baseline_drive_mu=0.0,
        baseline_drive_sigma=float(args.baseline_drive_sigma),
        divisive_gain=transition_templates.DIVISIVE_GAIN,
        pv_noise_sigma=float(args.pv_noise_sigma),
        alpha=1.0,
        weight_decay=0.0,
        seed=int(meta["seed"]),
        use_FF_connection=True,
        FF_plasticity=True,
        ff_plasticity_scale=1.0,
        use_ff_activity_accumulator=False,
        use_FB_connection=True,
        FB_plasticity=True,
        use_lat_connection=True,
        lat_plasticity=True,
        use_pv_connection=True,
        pv_plasticity=False,
        use_pv_lat_connection=True,
        pv_lat_plasticity=False,
    )


def _simulate_cc_point(spec: CCSweepSpec, profile: TuningProfile, x_value: float, y_value: float, args: argparse.Namespace, meta: dict[str, Any]) -> dict[str, Any]:
    test_stimuli, training_stimuli = _paper_stimuli(meta)
    model = _make_cc_model(spec, profile, x_value, y_value, args, meta)
    pre, baseline = _probe_model(model, meta, test_stimuli)
    run_experimental_phase(model, training_stimuli[0], training_stimuli[1], "full_familiar_training", update=True)
    post = _probe_model_with_baseline(model, meta, test_stimuli, baseline)
    row: dict[str, Any] = {
        "model": "CC",
        "profile": profile.name,
        "profile_label": profile.label,
        "sweep": spec.name,
        "x_weight": spec.x_weight,
        "y_weight": spec.y_weight,
        "fixed_weight": spec.fixed_weight,
        "x_init": float(x_value),
        "y_init": float(y_value),
        "fixed_init": float(args.fixed_weight),
    }
    _fill_response_columns(row, pre, post)
    _transition_flags(row, float(meta["sector_threshold"]))
    return row


def _make_pc_model(
    circuit: Literal["PPE", "NPE"],
    profile: TuningProfile,
    *,
    plastic_value: float,
    lat_value: float,
    args: argparse.Namespace,
    meta: dict[str, Any],
) -> CorrectPCneuron:
    rates = pc_templates._pc_learning_rates(circuit, int(meta["n_steps_per_phase"]))
    plastic = _profile_values(plastic_value, profile, float(args.untuned_weight))
    if circuit == "PPE":
        w_ff = plastic
        w_fb = np.zeros(N_FEATURES, dtype=float)
    else:
        w_ff = np.zeros(N_FEATURES, dtype=float)
        w_fb = plastic
    return CorrectPCneuron(
        n_features=N_FEATURES,
        n_pv=1,
        n_context=N_FEATURES,
        circuit=circuit,
        pc_plasticity_mode="ppe_ff_npe_fb",
        activation=ThresholdReLU(threshold=transition_templates.SOMA_ACTIVATION_THRESHOLD, subtractive=False, hasMax=True, maxValue=1.0),
        lr_ff=float(rates["lr_ff"]),
        lr_fb=float(rates["lr_fb"]),
        lr_lat=float(rates["lr_lat"]),
        w_ff_init=_init_spec(w_ff),
        w_fb_init=_init_spec(w_fb),
        W_pv_init=_init_spec([float(args.pv_input_weight)] * N_FEATURES),
        w_lat_init=_init_spec([lat_value]),
        w_pv_lat_init=_init_spec([0.0]),
        receives_context=(True, True, True),
        pyc_decay=transition_templates.BASE_CONFIG["pyc_decay"],
        pv_decay=transition_templates.BASE_CONFIG["pv_decay"],
        baseline_drive_mu=0.0,
        baseline_drive_sigma=float(args.baseline_drive_sigma),
        pv_noise_sigma=float(args.pv_noise_sigma),
        seed=int(meta["seed"]),
    )


def _simulate_single_pc(
    circuit: Literal["PPE", "NPE"],
    profile: TuningProfile,
    *,
    plastic_value: float,
    lat_value: float,
    args: argparse.Namespace,
    meta: dict[str, Any],
    grid_name: str,
    x_init: float,
    y_init: float,
) -> dict[str, Any]:
    test_stimuli, training_stimuli = _paper_stimuli(meta)
    model = _make_pc_model(circuit, profile, plastic_value=plastic_value, lat_value=lat_value, args=args, meta=meta)
    pre, baseline = _probe_model(model, meta, test_stimuli)
    run_experimental_phase(model, training_stimuli[0], training_stimuli[1], "full_familiar_training", update=True)
    post = _probe_model_with_baseline(model, meta, test_stimuli, baseline)
    row: dict[str, Any] = {
        "model": circuit,
        "profile": profile.name,
        "profile_label": profile.label,
        "sweep": grid_name,
        "x_init": float(x_init),
        "y_init": float(y_init),
        "fixed_init": float(args.fixed_weight),
        "pv_input_weight": float(args.pv_input_weight),
    }
    _fill_response_columns(row, pre, post)
    _transition_flags(row, float(meta["sector_threshold"]))
    return row


def _sum_response_dicts(a: dict[str, dict[str, float]], b: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        group: {key: float(a[group][key] + b[group][key]) for key in ("NO", "O")}
        for group in ("familiar", "novel")
    }


def _simulate_combined_pc(profile: TuningProfile, ppe_ff: float, npe_fb: float, args: argparse.Namespace, meta: dict[str, Any]) -> dict[str, Any]:
    test_stimuli, training_stimuli = _paper_stimuli(meta)
    ppe = _make_pc_model("PPE", profile, plastic_value=ppe_ff, lat_value=float(args.fixed_weight), args=args, meta=meta)
    npe = _make_pc_model("NPE", profile, plastic_value=npe_fb, lat_value=float(args.fixed_weight), args=args, meta=meta)
    ppe_pre, ppe_baseline = _probe_model(ppe, meta, test_stimuli)
    npe_pre, npe_baseline = _probe_model(npe, meta, test_stimuli)
    run_experimental_phase(ppe, training_stimuli[0], training_stimuli[1], "full_familiar_training", update=True)
    run_experimental_phase(npe, training_stimuli[0], training_stimuli[1], "full_familiar_training", update=True)
    ppe_post = _probe_model_with_baseline(ppe, meta, test_stimuli, ppe_baseline)
    npe_post = _probe_model_with_baseline(npe, meta, test_stimuli, npe_baseline)
    pre = _sum_response_dicts(ppe_pre, npe_pre)
    post = _sum_response_dicts(ppe_post, npe_post)
    row: dict[str, Any] = {
        "model": "PPE+NPE",
        "profile": profile.name,
        "profile_label": profile.label,
        "sweep": "ppe_ff_npe_fb",
        "x_init": float(ppe_ff),
        "y_init": float(npe_fb),
        "fixed_init": float(args.fixed_weight),
        "pv_input_weight": float(args.pv_input_weight),
    }
    _fill_response_columns(row, pre, post)
    _transition_flags(row, float(meta["sector_threshold"]))
    return row


def _run_cc(args: argparse.Namespace, meta: dict[str, Any]) -> pd.DataFrame:
    grid = np.linspace(float(args.grid_min), float(args.grid_max), int(args.grid_size))
    tasks = [(spec, profile, float(x), float(y)) for profile in PROFILES for spec in CC_SWEEPS for x in grid for y in grid]
    return pd.DataFrame(
        Parallel(n_jobs=int(args.n_jobs))(
            delayed(_simulate_cc_point)(spec, profile, x, y, args, meta)
            for spec, profile, x, y in tasks
        )
    )


def _run_pc(args: argparse.Namespace, meta: dict[str, Any]) -> pd.DataFrame:
    grid = np.linspace(float(args.grid_min), float(args.grid_max), int(args.grid_size))
    tasks: list[tuple[str, TuningProfile, float, float]] = []
    for profile in PROFILES:
        for ff in grid:
            for lat in grid:
                tasks.append(("PPE", profile, float(ff), float(lat)))
        for fb in grid:
            for lat in grid:
                tasks.append(("NPE", profile, float(fb), float(lat)))
        for ppe_ff in grid:
            for npe_fb in grid:
                tasks.append(("PPE+NPE", profile, float(ppe_ff), float(npe_fb)))

    def run_task(kind: str, profile: TuningProfile, x: float, y: float) -> dict[str, Any]:
        if kind == "PPE":
            return _simulate_single_pc("PPE", profile, plastic_value=x, lat_value=y, args=args, meta=meta, grid_name="ppe_ff_lat", x_init=x, y_init=y)
        if kind == "NPE":
            return _simulate_single_pc("NPE", profile, plastic_value=x, lat_value=y, args=args, meta=meta, grid_name="npe_fb_lat", x_init=x, y_init=y)
        return _simulate_combined_pc(profile, x, y, args, meta)

    return pd.DataFrame(
        Parallel(n_jobs=int(args.n_jobs))(
            delayed(run_task)(kind, profile, x, y)
            for kind, profile, x, y in tasks
        )
    )


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "profile", "sweep"], observed=True)
        .agg(
            n=("model", "size"),
            target_hits=("target_familiar_less_NO_more_O", "sum"),
            bonus_hits=("bonus_with_increased_novel_NO", "sum"),
            min_familiar_dNO=("familiar_dNO", "min"),
            max_familiar_dNO=("familiar_dNO", "max"),
            min_familiar_dO=("familiar_dO", "min"),
            max_familiar_dO=("familiar_dO", "max"),
            min_novel_dNO=("novel_dNO", "min"),
            max_novel_dNO=("novel_dNO", "max"),
        )
        .reset_index()
    )


def _plot_heatmap(rows: pd.DataFrame, *, title: str, x_label: str, y_label: str, output_path: Path) -> None:
    panels = [
        ("familiar_dNO", "fam dNO"),
        ("familiar_dO", "fam dO"),
        ("novel_dNO", "novel dNO"),
        ("novel_dO", "novel dO"),
    ]
    pre = [("familiar_pre_NO", "fam NO"), ("familiar_pre_O", "fam O"), ("novel_pre_NO", "novel NO"), ("novel_pre_O", "novel O")]
    post = [("familiar_post_NO", "fam NO"), ("familiar_post_O", "fam O"), ("novel_post_NO", "novel NO"), ("novel_post_O", "novel O")]
    delta_extent = float(np.nanmax(np.abs(rows[[c for c, _ in panels]].to_numpy(dtype=float))))
    if not np.isfinite(delta_extent) or delta_extent <= 0.0:
        delta_extent = 1.0
    response_extent = float(np.nanmax(rows[[c for c, _ in pre + post]].to_numpy(dtype=float)))
    if not np.isfinite(response_extent) or response_extent <= 0.0:
        response_extent = 1.0
    fig, axes = plt.subplots(3, 4, figsize=(13.4, 8.9), sharex=True, sharey=True)
    for ax, (column, label) in zip(axes[0], panels, strict=True):
        values = rows.pivot_table(values=column, index="y_init", columns="x_init", aggfunc="mean")
        im = ax.imshow(values.to_numpy(float), origin="lower", aspect="auto", cmap="coolwarm", vmin=-delta_extent, vmax=delta_extent, extent=[values.columns.min(), values.columns.max(), values.index.min(), values.index.max()])
        ax.set_title(label, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for row_idx, panel_set in ((1, pre), (2, post)):
        for ax, (column, label) in zip(axes[row_idx], panel_set, strict=True):
            values = rows.pivot_table(values=column, index="y_init", columns="x_init", aggfunc="mean")
            im = ax.imshow(values.to_numpy(float), origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=response_extent, extent=[values.columns.min(), values.columns.max(), values.index.min(), values.index.max()])
            ax.set_title(label, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[-1]:
        ax.set_xlabel(x_label)
    for ax, label in zip(axes[:, 0], ("delta", "pre", "post"), strict=True):
        ax.set_ylabel(f"{label}\n{y_label}")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_outputs(df: pd.DataFrame, args: argparse.Namespace, meta: dict[str, Any], label: str) -> None:
    out_dir = args.output_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "sweep_results.csv", index=False)
    summary = _summary(df)
    summary.to_csv(out_dir / "sweep_summary.csv", index=False)
    for (model, profile, sweep), rows in df.groupby(["model", "profile", "sweep"], observed=True):
        if model == "CC":
            spec = next(spec for spec in CC_SWEEPS if spec.name == sweep)
            x_label = f"initial {spec.x_weight}"
            y_label = f"initial {spec.y_weight}"
            stem = f"{profile}_{spec.output_stem}"
        elif model == "PPE":
            x_label = "initial PPE w_FF"
            y_label = "initial PPE w_LAT"
            stem = f"{profile}_ppe_ff_lat"
        elif model == "NPE":
            x_label = "initial NPE w_FB"
            y_label = "initial NPE w_LAT"
            stem = f"{profile}_npe_fb_lat"
        else:
            x_label = "initial PPE w_FF"
            y_label = "initial NPE w_FB"
            stem = f"{profile}_ppe_npe_combined"
        _plot_heatmap(
            rows,
            title=f"{model}: {rows['profile_label'].iloc[0]} / {sweep}",
            x_label=x_label,
            y_label=y_label,
            output_path=out_dir / f"{stem}.{args.image_format}",
        )
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "paper_timing_source": str(args.paper_output_dir),
                "paper_timing": meta,
                "grid_min": args.grid_min,
                "grid_max": args.grid_max,
                "grid_size": args.grid_size,
                "fixed_weight": args.fixed_weight,
                "pv_input_weight": args.pv_input_weight,
                "untuned_weight": args.untuned_weight,
                "baseline_drive_sigma": args.baseline_drive_sigma,
                "pv_noise_sigma": args.pv_noise_sigma,
            },
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string(index=False))
    print(f"Wrote {label} outputs to {out_dir}")


def main() -> None:
    args = parse_args()
    meta = _metadata(args)
    transition_templates.configure_model_scatter(int(meta["n_steps_per_phase"]))
    if args.run in {"all", "cc"}:
        _write_outputs(_run_cc(args, meta), args, meta, "cc_grid")
    if args.run in {"all", "pc"}:
        _write_outputs(_run_pc(args, meta), args, meta, "pc_grid")


if __name__ == "__main__":
    main()
