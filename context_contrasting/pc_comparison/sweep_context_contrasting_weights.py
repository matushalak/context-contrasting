from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from context_contrasting.paper.minimal_divisive import CCNeuron
from context_contrasting.paper.neuron_utils import ThresholdReLU


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs" / "context_contrasting_weight_sweeps"
N_FEATURES = 3
FAMILIAR_FEATURES = (0, 1)
NOVEL_FEATURE = 2
SweepName = Literal["ff_fb", "ff_lat", "fb_lat"]
ProfileName = Literal["broad_3of3", "narrow_familiar_1of3", "narrow_novel_1of3"]


@dataclass(frozen=True)
class SweepSpec:
    name: SweepName
    x_weight: str
    y_weight: str
    fixed_weight: str
    output_stem: str
    title: str


@dataclass(frozen=True)
class TuningProfile:
    name: ProfileName
    label: str
    ff_mask: tuple[float, float, float]
    tuned_feature: int | None


SWEEP_SPECS: tuple[SweepSpec, ...] = (
    SweepSpec("ff_fb", "w_ff", "w_fb", "w_lat", "cc_ff_fb", "CC: vary w_FF and w_FB"),
    SweepSpec("ff_lat", "w_ff", "w_lat", "w_fb", "cc_ff_lat", "CC: vary w_FF and w_LAT"),
    SweepSpec("fb_lat", "w_fb", "w_lat", "w_ff", "cc_fb_lat", "CC: vary w_FB and w_LAT"),
)

TUNING_PROFILES: tuple[TuningProfile, ...] = (
    TuningProfile("broad_3of3", "broad PyC FF tuning (3/3)", (1.0, 1.0, 1.0), None),
    TuningProfile("narrow_familiar_1of3", "narrow PyC FF tuning: familiar 1 (1/3)", (1.0, 0.0, 0.0), 0),
    TuningProfile("narrow_novel_1of3", "narrow PyC FF tuning: novel (1/3)", (0.0, 0.0, 1.0), NOVEL_FEATURE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pairwise initial-weight sweeps for the context-contrasting CCNeuron."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-min", type=float, default=0.05)
    parser.add_argument("--grid-max", type=float, default=0.95)
    parser.add_argument("--grid-size", type=int, default=11)
    parser.add_argument("--fixed-weight", type=float, default=1.0)
    parser.add_argument("--untuned-ff-weight", type=float, default=0.0)
    parser.add_argument("--pv-input-weight", type=float, default=1.0)
    parser.add_argument("--n-steps-per-stimulus", type=int, default=50)
    parser.add_argument("--training-trials", type=int, default=7)
    parser.add_argument("--response-tail-fraction", type=float, default=0.5)
    parser.add_argument("--zscore-std-floor", type=float, default=0.04)
    parser.add_argument("--lr-ff", type=float, default=0.0155)
    parser.add_argument("--lr-fb", type=float, default=0.00065)
    parser.add_argument("--lr-lat", type=float, default=0.0300)
    parser.add_argument("--divisive-gain", type=float, default=10.0)
    parser.add_argument("--apical-drive-threshold", type=float, default=0.16)
    parser.add_argument("--apical-drive-subtractive", action="store_true", default=True)
    parser.add_argument("--apical-drive-hard-threshold", dest="apical_drive_subtractive", action="store_false")
    parser.add_argument("--apical-gain-strength", type=float, default=3.8)
    parser.add_argument("--apical-gain-k", type=float, default=5.0)
    parser.add_argument("--pyc-decay", type=float, default=0.05)
    parser.add_argument("--pv-decay", type=float, default=0.5)
    parser.add_argument("--image-format", choices=("png", "svg", "eps"), default="png")
    return parser.parse_args()


def _one_hot(idx: int) -> torch.Tensor:
    values = torch.zeros(N_FEATURES, dtype=torch.float32)
    values[idx] = 1.0
    return values


def _scalar_init(value: float, size: int) -> dict[str, list[float] | float]:
    return {"mu": [float(value)] * size, "sigma": 0.0}


def _vector_init(values: np.ndarray | list[float] | tuple[float, ...]) -> dict[str, list[float] | float]:
    return {"mu": [float(value) for value in np.asarray(values, dtype=float).reshape(-1)], "sigma": 0.0}


def _initial_values(spec: SweepSpec, x_value: float, y_value: float, fixed_value: float) -> dict[str, float]:
    values = {"w_ff": fixed_value, "w_fb": fixed_value, "w_lat": fixed_value}
    values[spec.x_weight] = x_value
    values[spec.y_weight] = y_value
    return values


def _ff_init_values(value: float, profile: TuningProfile, args: argparse.Namespace) -> np.ndarray:
    mask = np.asarray(profile.ff_mask, dtype=float)
    return mask * float(value) + (1.0 - mask) * float(args.untuned_ff_weight)


def _make_model(spec: SweepSpec, profile: TuningProfile, x_value: float, y_value: float, args: argparse.Namespace) -> CCNeuron:
    values = _initial_values(spec, x_value, y_value, float(args.fixed_weight))
    w_ff_values = _ff_init_values(values["w_ff"], profile, args)
    return CCNeuron(
        n_features=N_FEATURES,
        n_pv=1,
        n_context=N_FEATURES,
        activation=ThresholdReLU(threshold=0.1, subtractive=False, hasMax=True, maxValue=1.0),
        lr_ff=float(args.lr_ff),
        lr_fb=float(args.lr_fb),
        lr_lat=float(args.lr_lat),
        lr_pv=0.0,
        w_ff_init=_vector_init(w_ff_values),
        w_fb_init=_scalar_init(values["w_fb"], N_FEATURES),
        w_lat_init=_scalar_init(values["w_lat"], 1),
        w_pv_lat_init=_scalar_init(float(args.pv_input_weight), 1),
        W_pv_init=_scalar_init(float(args.pv_input_weight), N_FEATURES),
        receives_context=(True, True, True),
        FFrule="anti-Hebbian",
        FBrule="dampened-anti-Hebbian",
        pyc_decay=float(args.pyc_decay),
        pv_decay=float(args.pv_decay),
        apical_drive_threshold=float(args.apical_drive_threshold),
        apical_drive_subtractive=bool(args.apical_drive_subtractive),
        apical_gain_strength=float(args.apical_gain_strength),
        apical_gain_k=float(args.apical_gain_k),
        apical_gain_threshold=0.05,
        baseline_drive_mu=0.0,
        baseline_drive_sigma=0.0,
        divisive_gain=float(args.divisive_gain),
        pv_noise_sigma=0.0,
        alpha=1.0,
        weight_decay=0.0,
        seed=42,
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


def _run_constant_phase(
    model: CCNeuron,
    x: torch.Tensor,
    c: torch.Tensor,
    *,
    n_steps: int,
    update: bool,
    reset_state: bool,
) -> np.ndarray:
    if reset_state:
        model._reset_state()
    responses = []
    with torch.no_grad():
        for _ in range(n_steps):
            x_t, y_t, y_next, p_t, c_t = model(x, c)
            if update:
                model.update(x_t, y_t, y_next, p_t, c_t)
            responses.append(float(y_next.item()))
    return np.asarray(responses, dtype=float)


def _tail_mean(values: np.ndarray, tail_fraction: float) -> float:
    start = int(round((1.0 - tail_fraction) * len(values)))
    return float(np.mean(values[start:]))


def _response_pair(model: CCNeuron, feature_idx: int, args: argparse.Namespace) -> dict[str, float]:
    x = _one_hot(feature_idx)
    c = _one_hot(feature_idx)
    no_trace = _run_constant_phase(
        model,
        x,
        c,
        n_steps=int(args.n_steps_per_stimulus),
        update=False,
        reset_state=True,
    )
    o_trace = _run_constant_phase(
        model,
        torch.zeros_like(x),
        c,
        n_steps=int(args.n_steps_per_stimulus),
        update=False,
        reset_state=True,
    )
    return {
        "NO": _tail_mean(no_trace, float(args.response_tail_fraction)),
        "O": _tail_mean(o_trace, float(args.response_tail_fraction)),
    }


def _measure_model(model: CCNeuron, args: argparse.Namespace) -> dict[str, dict[str, float]]:
    familiar = [_response_pair(model, feature_idx, args) for feature_idx in FAMILIAR_FEATURES]
    familiar_mean = {
        key: float(np.mean([response[key] for response in familiar]))
        for key in familiar[0]
    }
    return {
        "familiar": familiar_mean,
        "novel": _response_pair(model, NOVEL_FEATURE, args),
    }


def _train_model(model: CCNeuron, args: argparse.Namespace) -> None:
    for _ in range(int(args.training_trials)):
        for feature_idx in FAMILIAR_FEATURES:
            x = _one_hot(feature_idx)
            c = _one_hot(feature_idx)
            _run_constant_phase(
                model,
                x,
                c,
                n_steps=int(args.n_steps_per_stimulus),
                update=True,
                reset_state=True,
            )


def _prefixed(values: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in values.items()}


def _final_weight_columns(model: CCNeuron) -> dict[str, float]:
    w_ff = model.w_ff.detach().cpu().numpy().reshape(-1)
    w_fb = model.w_fb.detach().cpu().numpy().reshape(-1)
    w_lat = model.w_lat.detach().cpu().numpy().reshape(-1)
    rows: dict[str, float] = {}
    for idx, value in enumerate(w_ff):
        rows[f"final_w_ff_{idx}"] = float(value)
    for idx, value in enumerate(w_fb):
        rows[f"final_w_fb_{idx}"] = float(value)
    rows["final_w_lat"] = float(w_lat[0])
    return rows


def _simulate_grid_point(
    spec: SweepSpec,
    profile: TuningProfile,
    x_value: float,
    y_value: float,
    args: argparse.Namespace,
) -> dict[str, float | str | bool]:
    model = _make_model(spec, profile, x_value, y_value, args)
    pre = _measure_model(model, args)
    _train_model(model, args)
    post = _measure_model(model, args)

    familiar_dno = post["familiar"]["NO"] - pre["familiar"]["NO"]
    familiar_do = post["familiar"]["O"] - pre["familiar"]["O"]
    novel_dno = post["novel"]["NO"] - pre["novel"]["NO"]
    novel_do = post["novel"]["O"] - pre["novel"]["O"]
    row: dict[str, float | str | bool] = {
        "profile": profile.name,
        "profile_label": profile.label,
        "tuned_feature": -1 if profile.tuned_feature is None else int(profile.tuned_feature),
        "ff_tuning_width": int(sum(1 for value in profile.ff_mask if value > 0.0)),
        "sweep": spec.name,
        "x_weight": spec.x_weight,
        "y_weight": spec.y_weight,
        "fixed_weight": spec.fixed_weight,
        "x_init": float(x_value),
        "y_init": float(y_value),
        "fixed_init": float(args.fixed_weight),
        "untuned_ff_weight": float(args.untuned_ff_weight),
        "pv_input_weight": float(args.pv_input_weight),
        "familiar_dNO": float(familiar_dno),
        "familiar_dO": float(familiar_do),
        "novel_dNO": float(novel_dno),
        "novel_dO": float(novel_do),
        "plus_NO": bool(familiar_dno > 1e-9),
        "minus_NO": bool(familiar_dno < -1e-9),
        "plus_O": bool(familiar_do > 1e-9),
        "minus_O": bool(familiar_do < -1e-9),
        "target_familiar_less_NO_more_O": bool(familiar_dno < -1e-9 and familiar_do > 1e-9),
        "bonus_with_increased_novel_NO": bool(familiar_dno < -1e-9 and familiar_do > 1e-9 and novel_dno > 1e-9),
    }
    for group in ("familiar", "novel"):
        row.update(_prefixed(pre[group], f"{group}_pre"))
        row.update(_prefixed(post[group], f"{group}_post"))
    row.update(_final_weight_columns(model))
    return row


def run_sweeps(args: argparse.Namespace) -> pd.DataFrame:
    grid = np.linspace(float(args.grid_min), float(args.grid_max), int(args.grid_size))
    rows = []
    for profile in TUNING_PROFILES:
        for spec in SWEEP_SPECS:
            for x_value in grid:
                for y_value in grid:
                    rows.append(_simulate_grid_point(spec, profile, float(x_value), float(y_value), args))
    return pd.DataFrame(rows)


def _add_zscore_columns(df: pd.DataFrame, *, zscore_std_floor: float) -> pd.DataFrame:
    if zscore_std_floor <= 0.0:
        raise ValueError("zscore_std_floor must be positive.")
    df = df.copy()
    response_columns = [
        column
        for column in df.columns
        if (
            (
                column.startswith("familiar_pre_")
                or column.startswith("familiar_post_")
                or column.startswith("novel_pre_")
                or column.startswith("novel_post_")
            )
            and not column.endswith("_z")
        )
    ]
    for column in response_columns:
        df[f"{column}_z"] = df[column].astype(float) / zscore_std_floor
    for column in ("familiar_dNO", "familiar_dO", "novel_dNO", "novel_dO"):
        df[f"{column}_z"] = df[column].astype(float) / zscore_std_floor
    df["response_zscore_floor"] = float(zscore_std_floor)
    return df


def summarize_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["profile", "sweep"], observed=True)
        .agg(
            n=("sweep", "size"),
            plus_NO=("plus_NO", "sum"),
            minus_NO=("minus_NO", "sum"),
            plus_O=("plus_O", "sum"),
            minus_O=("minus_O", "sum"),
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


def _plot_sweep_heatmap(
    rows: pd.DataFrame,
    spec: SweepSpec,
    profile: TuningProfile,
    output_dir: Path,
    image_format: str,
    *,
    fixed_value: float,
) -> None:
    pivot_kwargs = {"index": "y_init", "columns": "x_init"}
    delta_heatmaps = [
        ("familiar_dNO_z", "fam dNO"),
        ("familiar_dO_z", "fam dO"),
        ("novel_dNO_z", "novel dNO"),
        ("novel_dO_z", "novel dO"),
    ]
    pre_response_heatmaps = [
        ("familiar_pre_NO_z", "fam NO"),
        ("familiar_pre_O_z", "fam O"),
        ("novel_pre_NO_z", "novel NO"),
        ("novel_pre_O_z", "novel O"),
    ]
    post_response_heatmaps = [
        ("familiar_post_NO_z", "fam NO"),
        ("familiar_post_O_z", "fam O"),
        ("novel_post_NO_z", "novel NO"),
        ("novel_post_O_z", "novel O"),
    ]
    delta_extent = float(np.nanmax(np.abs(rows[[column for column, _ in delta_heatmaps]].to_numpy(dtype=float))))
    if not np.isfinite(delta_extent) or delta_extent == 0.0:
        delta_extent = 1.0
    response_columns = [column for column, _ in pre_response_heatmaps + post_response_heatmaps]
    response_vmax = float(np.nanmax(rows[response_columns].to_numpy(dtype=float)))
    if not np.isfinite(response_vmax) or response_vmax == 0.0:
        response_vmax = 1.0

    fig, axes = plt.subplots(3, 4, figsize=(13.4, 8.9), sharex=True, sharey=True)
    for ax, (column, panel_title) in zip(axes[0], delta_heatmaps, strict=True):
        values = rows.pivot_table(values=column, aggfunc="mean", **pivot_kwargs)
        image = ax.imshow(
            values.to_numpy(dtype=float),
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=-delta_extent,
            vmax=delta_extent,
            extent=[
                float(values.columns.min()),
                float(values.columns.max()),
                float(values.index.min()),
                float(values.index.max()),
            ],
        )
        ax.set_title(f"{panel_title} (z)", fontsize=10)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for row_idx, response_heatmaps in ((1, pre_response_heatmaps), (2, post_response_heatmaps)):
        for ax, (column, panel_title) in zip(axes[row_idx], response_heatmaps, strict=True):
            values = rows.pivot_table(values=column, aggfunc="mean", **pivot_kwargs)
            image = ax.imshow(
                values.to_numpy(dtype=float),
                origin="lower",
                aspect="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=response_vmax,
                extent=[
                    float(values.columns.min()),
                    float(values.columns.max()),
                    float(values.index.min()),
                    float(values.index.max()),
                ],
            )
            ax.set_title(f"{panel_title} response (z)", fontsize=10)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[-1]:
        ax.set_xlabel(f"initial {spec.x_weight}")
    for ax, label in zip(axes[:, 0], ("delta", "pre", "post"), strict=True):
        ax.set_ylabel(f"{label}\ninitial {spec.y_weight}")
    fig.suptitle(f"{profile.label}; {spec.title}, {spec.fixed_weight}={fixed_value:g}", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), w_pad=1.2, h_pad=1.4)
    fig.savefig(output_dir / f"{profile.name}_{spec.output_stem}.{image_format}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_outputs(df: pd.DataFrame, args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = _add_zscore_columns(df, zscore_std_floor=float(args.zscore_std_floor))
    summary = summarize_sweeps(df)
    df.to_csv(args.output_dir / "sweep_results.csv", index=False)
    summary.to_csv(args.output_dir / "sweep_summary.csv", index=False)
    for profile in TUNING_PROFILES:
        for spec in SWEEP_SPECS:
            rows = df.loc[df["profile"].eq(profile.name) & df["sweep"].eq(spec.name)].copy()
            _plot_sweep_heatmap(
                rows,
                spec,
                profile,
                args.output_dir,
                args.image_format,
                fixed_value=float(args.fixed_weight),
            )
    metadata = {
        "grid_min": args.grid_min,
        "grid_max": args.grid_max,
        "grid_size": args.grid_size,
        "fixed_weight": args.fixed_weight,
        "untuned_ff_weight": args.untuned_ff_weight,
        "pv_input_weight": args.pv_input_weight,
        "tuning_profiles": [
            {
                "name": profile.name,
                "label": profile.label,
                "ff_mask": list(profile.ff_mask),
                "tuned_feature": profile.tuned_feature,
            }
            for profile in TUNING_PROFILES
        ],
        "n_steps_per_stimulus": args.n_steps_per_stimulus,
        "training_trials": args.training_trials,
        "response_tail_fraction": args.response_tail_fraction,
        "zscore_std_floor": args.zscore_std_floor,
        "lr_ff": args.lr_ff,
        "lr_fb": args.lr_fb,
        "lr_lat": args.lr_lat,
        "divisive_gain": args.divisive_gain,
        "apical_drive_threshold": args.apical_drive_threshold,
        "apical_drive_subtractive": args.apical_drive_subtractive,
        "apical_gain_strength": args.apical_gain_strength,
        "apical_gain_k": args.apical_gain_k,
        "note": (
            "W_pv and w_pv_lat are initialized to pv_input_weight and kept nonplastic. "
            "Tuning profiles mask PyC feedforward weights only; feedback remains generalized over all context channels. "
            "For bounded Hebbian weights, an initial value of 1.0 is effectively saturated unless weight decay is enabled."
        ),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"\nWrote outputs to {args.output_dir}")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.response_tail_fraction <= 1.0:
        raise ValueError("response_tail_fraction must be in (0, 1].")
    df = run_sweeps(args)
    write_outputs(df, args)


if __name__ == "__main__":
    main()
