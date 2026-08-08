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


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs" / "ppe_ff_plasticity_sweep"
N_FEATURES = 3
FAMILIAR_FEATURES = (0, 1)
NOVEL_FEATURE = 2
TeachingSignal = Literal["signed", "rectified"]
PlasticityMode = Literal["ppe_ff_npe_fb", "lat"]


@dataclass
class SweepCell:
    circuit: Literal["PPE", "NPE"]
    plasticity_mode: PlasticityMode
    plastic_weight: np.ndarray
    fixed_drive: np.ndarray
    pv_drive: np.ndarray
    learning_rate: float
    teaching_signal: TeachingSignal

    def signed_error(self, x: np.ndarray, c: np.ndarray) -> float:
        if self.plasticity_mode == "lat":
            if self.circuit == "PPE":
                direct_drive = np.dot(self.fixed_drive, x)
                pv_drive = np.dot(self.pv_drive, c)
            else:
                direct_drive = np.dot(self.fixed_drive, c)
                pv_drive = np.dot(self.pv_drive, x)
            return float(direct_drive - self.plastic_weight.item() * pv_drive)

        if self.circuit == "PPE":
            return float(np.dot(self.plastic_weight, x) - np.dot(self.fixed_drive, c))
        return float(np.dot(self.plastic_weight, c) - np.dot(self.fixed_drive, x))

    def response(self, x: np.ndarray, c: np.ndarray) -> float:
        return max(self.signed_error(x, c), 0.0)

    def update(self, x: np.ndarray, c: np.ndarray) -> None:
        error = self.signed_error(x, c)
        if self.teaching_signal == "rectified":
            error = max(error, 0.0)
        if self.plasticity_mode == "lat":
            if self.circuit == "PPE":
                presynaptic = np.asarray([np.dot(self.pv_drive, c)], dtype=float)
            else:
                presynaptic = np.asarray([np.dot(self.pv_drive, x)], dtype=float)
            raw_delta = self.learning_rate * error * presynaptic
        else:
            presynaptic = x if self.circuit == "PPE" else c
            raw_delta = -self.learning_rate * error * presynaptic
        self.plastic_weight = _bounded_signed_update(self.plastic_weight, raw_delta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the modified predictive-coding case where PPE neurons update "
            "feedforward synapses with the same anti-Hebbian rule used for NPE "
            "feedback plasticity, while all other synapses are frozen."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-min", type=float, default=0.05)
    parser.add_argument("--grid-max", type=float, default=0.95)
    parser.add_argument("--grid-size", type=int, default=19)
    parser.add_argument(
        "--balance-weight",
        type=float,
        default=0.5,
        help="Frozen opponent weight. At this value, unit sensory and context inputs cancel exactly.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument(
        "--plasticity-mode",
        choices=("ppe_ff_npe_fb", "lat"),
        default="ppe_ff_npe_fb",
        help="Which predictive-coding synapse is plastic: PPE w_FF/NPE w_FB anti-Hebbian, or scalar w_LAT Hebbian.",
    )
    parser.add_argument(
        "--pv-drive",
        type=float,
        default=1.0,
        help="Frozen PV drive used only for --plasticity-mode lat; w_LAT=balance_weight cancels direct drive when pv_drive=1.",
    )
    parser.add_argument("--training-trials", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument(
        "--zscore-std-floor",
        type=float,
        default=0.04,
        help="Response scale used to convert deterministic baseline-zero responses to z-score units.",
    )
    parser.add_argument("--teaching-signal", choices=("signed", "rectified"), default="signed")
    parser.add_argument("--image-format", choices=("png", "svg", "eps"), default="png")
    return parser.parse_args()


def _bounded_signed_update(weights: np.ndarray, raw_delta: np.ndarray) -> np.ndarray:
    scaled_delta = np.where(raw_delta >= 0.0, raw_delta * (1.0 - weights), raw_delta * weights)
    return np.clip(weights + scaled_delta, 0.0, 1.0)


def _one_hot(idx: int) -> np.ndarray:
    values = np.zeros(N_FEATURES, dtype=float)
    values[idx] = 1.0
    return values


def _make_cell(
    circuit: Literal["PPE", "NPE"],
    *,
    plasticity_mode: PlasticityMode,
    plastic_init: float,
    balance_weight: float,
    pv_drive: float,
    learning_rate: float,
    teaching_signal: TeachingSignal,
) -> SweepCell:
    plastic_shape = (1,) if plasticity_mode == "lat" else (N_FEATURES,)
    return SweepCell(
        circuit=circuit,
        plasticity_mode=plasticity_mode,
        plastic_weight=np.full(plastic_shape, float(plastic_init), dtype=float),
        fixed_drive=np.full(N_FEATURES, float(balance_weight), dtype=float),
        pv_drive=np.full(N_FEATURES, float(pv_drive), dtype=float),
        learning_rate=float(learning_rate),
        teaching_signal=teaching_signal,
    )


def _response_pair(cell: SweepCell, feature_idx: int) -> dict[str, float]:
    x = _one_hot(feature_idx)
    c = _one_hot(feature_idx)
    return {
        "NO": cell.response(x, c),
        "O": cell.response(np.zeros(N_FEATURES, dtype=float), c),
        "FF_only": cell.response(x, np.zeros(N_FEATURES, dtype=float)),
        "FB_only": cell.response(np.zeros(N_FEATURES, dtype=float), c),
    }


def _train(cell: SweepCell, training_trials: int) -> None:
    for _ in range(training_trials):
        for feature_idx in FAMILIAR_FEATURES:
            x = _one_hot(feature_idx)
            c = _one_hot(feature_idx)
            cell.update(x, c)


def _prefixed(values: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in values.items()}


def _summed_metrics(ppe: dict[str, float], npe: dict[str, float]) -> dict[str, float]:
    keys = set(ppe) | set(npe)
    return {key: float(ppe.get(key, 0.0) + npe.get(key, 0.0)) for key in keys}


def _measure_cell(cell: SweepCell) -> dict[str, dict[str, float]]:
    familiar = [_response_pair(cell, feature_idx) for feature_idx in FAMILIAR_FEATURES]
    familiar_mean = {
        key: float(np.mean([response[key] for response in familiar]))
        for key in familiar[0]
    }
    return {
        "familiar": familiar_mean,
        "novel": _response_pair(cell, NOVEL_FEATURE),
    }


def _final_weights(cell: SweepCell, prefix: str) -> dict[str, float]:
    key = "w_lat" if cell.plasticity_mode == "lat" else "w"
    return {f"{prefix}_final_{key}_{idx}": float(value) for idx, value in enumerate(cell.plastic_weight)}


def _simulate_pair(
    ppe_ff_init: float,
    npe_fb_init: float,
    *,
    plasticity_mode: PlasticityMode,
    balance_weight: float,
    pv_drive: float,
    learning_rate: float,
    training_trials: int,
    threshold: float,
    teaching_signal: TeachingSignal,
) -> list[dict[str, float | str | bool]]:
    ppe = _make_cell(
        "PPE",
        plasticity_mode=plasticity_mode,
        plastic_init=ppe_ff_init,
        balance_weight=balance_weight,
        pv_drive=pv_drive,
        learning_rate=learning_rate,
        teaching_signal=teaching_signal,
    )
    npe = _make_cell(
        "NPE",
        plasticity_mode=plasticity_mode,
        plastic_init=npe_fb_init,
        balance_weight=balance_weight,
        pv_drive=pv_drive,
        learning_rate=learning_rate,
        teaching_signal=teaching_signal,
    )

    ppe_pre = _measure_cell(ppe)
    npe_pre = _measure_cell(npe)
    _train(ppe, training_trials)
    _train(npe, training_trials)
    ppe_post = _measure_cell(ppe)
    npe_post = _measure_cell(npe)

    class_metrics = {
        "PPE": (ppe_pre, ppe_post),
        "NPE": (npe_pre, npe_post),
        "PPE+NPE": (
            {
                group: _summed_metrics(ppe_pre[group], npe_pre[group])
                for group in ("familiar", "novel")
            },
            {
                group: _summed_metrics(ppe_post[group], npe_post[group])
                for group in ("familiar", "novel")
            },
        ),
    }

    rows: list[dict[str, float | str | bool]] = []
    for class_name, (pre, post) in class_metrics.items():
        familiar_dno = post["familiar"]["NO"] - pre["familiar"]["NO"]
        familiar_do = post["familiar"]["O"] - pre["familiar"]["O"]
        novel_dno = post["novel"]["NO"] - pre["novel"]["NO"]
        novel_do = post["novel"]["O"] - pre["novel"]["O"]
        row: dict[str, float | str | bool] = {
            "class": class_name,
            "ppe_ff_init": float(ppe_ff_init),
            "npe_fb_init": float(npe_fb_init),
            "ppe_plastic_init": float(ppe_ff_init),
            "npe_plastic_init": float(npe_fb_init),
            "plasticity_mode": plasticity_mode,
            "balance_weight": float(balance_weight),
            "pv_drive": float(pv_drive),
            "learning_rate": float(learning_rate),
            "training_trials": int(training_trials),
            "teaching_signal": teaching_signal,
            "familiar_dNO": float(familiar_dno),
            "familiar_dO": float(familiar_do),
            "novel_dNO": float(novel_dno),
            "novel_dO": float(novel_do),
            "target_familiar_less_NO_more_O": bool(familiar_dno < -threshold and familiar_do > threshold),
            "bonus_with_increased_novel_NO": bool(
                familiar_dno < -threshold and familiar_do > threshold and novel_dno > threshold
            ),
        }
        for group in ("familiar", "novel"):
            row.update(_prefixed(pre[group], f"{group}_pre"))
            row.update(_prefixed(post[group], f"{group}_post"))
        row.update(_final_weights(ppe, "ppe"))
        row.update(_final_weights(npe, "npe"))
        rows.append(row)
    return rows


def run_sweep(args: argparse.Namespace) -> pd.DataFrame:
    grid = np.linspace(args.grid_min, args.grid_max, args.grid_size)
    rows: list[dict[str, float | str | bool]] = []
    for ppe_ff_init in grid:
        for npe_fb_init in grid:
            rows.extend(
                _simulate_pair(
                    float(ppe_ff_init),
                    float(npe_fb_init),
                    plasticity_mode=args.plasticity_mode,
                    balance_weight=float(args.balance_weight),
                    pv_drive=float(args.pv_drive),
                    learning_rate=float(args.learning_rate),
                    training_trials=int(args.training_trials),
                    threshold=float(args.threshold),
                    teaching_signal=args.teaching_signal,
                )
            )
    return pd.DataFrame(rows)


def summarize_hits(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("class", observed=True)
        .agg(
            n=("class", "size"),
            familiar_target_hits=("target_familiar_less_NO_more_O", "sum"),
            bonus_hits=("bonus_with_increased_novel_NO", "sum"),
            min_familiar_dNO=("familiar_dNO", "min"),
            max_familiar_dO=("familiar_dO", "max"),
            max_novel_dNO=("novel_dNO", "max"),
            max_novel_dO=("novel_dO", "max"),
        )
        .reset_index()
    )


def _add_zscore_columns(df: pd.DataFrame, *, zscore_std_floor: float) -> pd.DataFrame:
    if zscore_std_floor <= 0.0:
        raise ValueError("zscore_std_floor must be positive.")
    df = df.copy()
    response_columns = [
        column
        for column in df.columns
        if (
            (column.startswith("familiar_pre_") or column.startswith("familiar_post_") or column.startswith("novel_pre_") or column.startswith("novel_post_"))
            and not column.endswith("_z")
        )
    ]
    for column in response_columns:
        df[f"{column}_z"] = df[column].astype(float) / zscore_std_floor
    for column in ("familiar_dNO", "familiar_dO", "novel_dNO", "novel_dO"):
        df[f"{column}_z"] = df[column].astype(float) / zscore_std_floor
    df["response_zscore_floor"] = float(zscore_std_floor)
    return df


def _plot_response_scatter(df: pd.DataFrame, output_dir: Path, image_format: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharex=True, sharey=True)
    for ax, class_name in zip(axes, ("PPE", "NPE", "PPE+NPE"), strict=True):
        rows = df.loc[df["class"].eq(class_name)]
        colors = np.where(rows["target_familiar_less_NO_more_O"], "#c0362c", "#5d6872")
        ax.scatter(rows["familiar_dNO_z"], rows["familiar_dO_z"], s=14, c=colors, alpha=0.72, linewidths=0)
        ax.axhline(0.0, color="0.2", lw=0.8)
        ax.axvline(0.0, color="0.2", lw=0.8)
        ax.set_title(class_name)
        ax.set_xlabel("familiar dNO (z)")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("familiar dO (z)")
    fig.tight_layout()
    fig.savefig(output_dir / f"familiar_response_shift_scatter.{image_format}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _axis_labels(df: pd.DataFrame) -> tuple[str, str]:
    mode = str(df["plasticity_mode"].iloc[0]) if "plasticity_mode" in df.columns and not df.empty else "ppe_ff_npe_fb"
    if mode == "lat":
        return "PPE initial w_LAT", "NPE initial w_LAT"
    return "PPE initial w_FF", "NPE initial w_FB"


def _plot_class_heatmaps(
    df: pd.DataFrame,
    output_dir: Path,
    image_format: str,
    *,
    class_name: str,
    output_stem: str,
    title: str,
) -> None:
    rows = df.loc[df["class"].eq(class_name)].copy()
    if rows.empty:
        return
    x_label, y_label = _axis_labels(rows)
    pivot_kwargs = {"index": "npe_plastic_init", "columns": "ppe_plastic_init"}
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
    response_values = rows[response_columns].to_numpy(dtype=float)
    response_vmax = float(np.nanmax(response_values))
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
        ax.set_xlabel(x_label)
    for ax, label in zip(axes[:, 0], ("delta", "pre", "post"), strict=True):
        ax.set_ylabel(f"{label}\n{y_label}")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), w_pad=1.2, h_pad=1.4)
    fig.savefig(output_dir / f"{output_stem}.{image_format}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_pair_heatmaps(df: pd.DataFrame, output_dir: Path, image_format: str) -> None:
    plot_specs = (
        ("PPE", "ppe_heatmaps", "PPE responses"),
        ("NPE", "npe_heatmaps", "NPE responses"),
        ("PPE+NPE", "ppe_npe_pair_heatmaps", "PPE+NPE summed responses"),
    )
    for class_name, output_stem, title in plot_specs:
        _plot_class_heatmaps(
            df,
            output_dir,
            image_format,
            class_name=class_name,
            output_stem=output_stem,
            title=title,
        )


def _plot_individual_vs_sum_heatmaps(df: pd.DataFrame, output_dir: Path, image_format: str) -> None:
    x_label, y_label = _axis_labels(df)
    pivot_kwargs = {"index": "npe_plastic_init", "columns": "ppe_plastic_init"}
    panels = [
        ("PPE", "familiar_post_NO_z", "PPE familiar NO"),
        ("NPE", "familiar_post_O_z", "NPE familiar O"),
        ("PPE+NPE", "familiar_post_NO_z", "sum familiar NO"),
        ("PPE+NPE", "familiar_post_O_z", "sum familiar O"),
    ]
    values_for_scale = []
    for class_name, column, _ in panels:
        values_for_scale.append(df.loc[df["class"].eq(class_name), column].to_numpy(dtype=float))
    vmax = float(np.nanmax(np.concatenate(values_for_scale)))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.0), sharex=True, sharey=True)
    for ax, (class_name, column, title) in zip(axes, panels, strict=True):
        rows = df.loc[df["class"].eq(class_name)]
        values = rows.pivot_table(values=column, aggfunc="mean", **pivot_kwargs)
        image = ax.imshow(
            values.to_numpy(dtype=float),
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            extent=[
                float(values.columns.min()),
                float(values.columns.max()),
                float(values.index.min()),
                float(values.index.max()),
            ],
        )
        ax.set_title(f"{title} response (z)")
        ax.set_xlabel(x_label)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel(y_label)
    fig.tight_layout()
    fig.savefig(output_dir / f"ppe_npe_individual_vs_sum_responses.{image_format}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_outputs(df: pd.DataFrame, args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = _add_zscore_columns(df, zscore_std_floor=float(args.zscore_std_floor))
    summary = summarize_hits(df)
    df.to_csv(args.output_dir / "sweep_results.csv", index=False)
    summary.to_csv(args.output_dir / "hit_summary.csv", index=False)
    metadata = {
        "grid_min": args.grid_min,
        "grid_max": args.grid_max,
        "grid_size": args.grid_size,
        "balance_weight": args.balance_weight,
        "plasticity_mode": args.plasticity_mode,
        "pv_drive": args.pv_drive,
        "learning_rate": args.learning_rate,
        "training_trials": args.training_trials,
        "threshold": args.threshold,
        "zscore_std_floor": args.zscore_std_floor,
        "teaching_signal": args.teaching_signal,
        "n_rows": int(len(df)),
        "interpretation": (
            "Rows labeled PPE and NPE are individual dedicated PC neurons. Rows labeled PPE+NPE "
            "are a summed observable made from one PPE and one NPE unit with independently swept "
            "initial plastic weights."
        ),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _plot_response_scatter(df, args.output_dir, args.image_format)
    _plot_pair_heatmaps(df, args.output_dir, args.image_format)
    _plot_individual_vs_sum_heatmaps(df, args.output_dir, args.image_format)


def main() -> None:
    args = parse_args()
    df = run_sweep(args)
    write_outputs(df, args)
    print(summarize_hits(df).to_string(index=False))
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
