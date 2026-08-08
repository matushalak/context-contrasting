from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs" / "signed_pe_convergence"
Circuit = Literal["PPE", "NPE"]
PlasticityMode = Literal["lat_hebbian", "ff_fb_anti_hebbian"]


@dataclass
class TracePoint:
    plasticity_mode: PlasticityMode
    circuit: Circuit
    initial_signed_pe: float
    plastic_weight_init: float
    trial: int
    trial_average_signed_pe: float
    plastic_weight_mean: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot trial-average signed PPE/NPE convergence for symmetric initial "
            "prediction errors produced by varying only the plastic weight."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--training-trials", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--fixed-weight", type=float, default=0.5)
    parser.add_argument(
        "--pv-activity",
        type=float,
        default=1.0,
        help=(
            "Unit PV presynaptic activity for the lateral Hebbian update. With "
            "--fixed-weight 0.5, pv_activity=1 makes w_LAT=0.5 the zero-PE point."
        ),
    )
    parser.add_argument("--max-initial-abs-pe", type=float, default=0.45)
    parser.add_argument("--n-initializations", type=int, default=9)
    parser.add_argument("--image-format", choices=("png", "svg", "eps"), default="png")
    return parser.parse_args()


def _bounded_signed_update(weight: np.ndarray, raw_delta: np.ndarray) -> np.ndarray:
    scaled_delta = np.where(raw_delta >= 0.0, raw_delta * (1.0 - weight), raw_delta * weight)
    return np.clip(weight + scaled_delta, 0.0, 1.0)


def _initial_signed_pes(max_abs_pe: float, n_initializations: int) -> np.ndarray:
    if max_abs_pe <= 0.0:
        raise ValueError("max_initial_abs_pe must be positive.")
    if n_initializations < 3:
        raise ValueError("n_initializations must be at least 3.")
    return np.linspace(-float(max_abs_pe), float(max_abs_pe), int(n_initializations))


def _plastic_init_from_error(
    initial_signed_pe: float,
    *,
    plasticity_mode: PlasticityMode,
    fixed_weight: float,
    pv_activity: float,
) -> float:
    if plasticity_mode == "lat_hebbian":
        if pv_activity <= 0.0:
            raise ValueError("pv_activity must be positive for lateral plasticity.")
        return (fixed_weight - initial_signed_pe) / pv_activity
    return fixed_weight + initial_signed_pe


def _signed_pe(
    plastic_weight: np.ndarray,
    *,
    plasticity_mode: PlasticityMode,
    fixed_weight: float,
    pv_activity: float,
) -> float:
    plastic_drive = float(np.mean(plastic_weight))
    if plasticity_mode == "lat_hebbian":
        return fixed_weight - plastic_drive * pv_activity
    return plastic_drive - fixed_weight


def _update_weight(
    plastic_weight: np.ndarray,
    signed_pe: float,
    *,
    plasticity_mode: PlasticityMode,
    learning_rate: float,
    pv_activity: float,
) -> np.ndarray:
    if plasticity_mode == "lat_hebbian":
        raw_delta = np.full_like(plastic_weight, learning_rate * signed_pe * pv_activity)
    else:
        raw_delta = np.full_like(plastic_weight, -learning_rate * signed_pe)
    return _bounded_signed_update(plastic_weight, raw_delta)


def simulate_traces(args: argparse.Namespace) -> list[TracePoint]:
    initial_pes = _initial_signed_pes(args.max_initial_abs_pe, args.n_initializations)
    traces: list[TracePoint] = []
    for plasticity_mode in ("lat_hebbian", "ff_fb_anti_hebbian"):
        for circuit in ("PPE", "NPE"):
            for initial_pe in initial_pes:
                plastic_init = _plastic_init_from_error(
                    float(initial_pe),
                    plasticity_mode=plasticity_mode,
                    fixed_weight=float(args.fixed_weight),
                    pv_activity=float(args.pv_activity),
                )
                if not 0.0 <= plastic_init <= 1.0:
                    raise ValueError(
                        f"initial PE {initial_pe:.3g} maps to plastic weight "
                        f"{plastic_init:.3g}, outside [0, 1]."
                    )

                plastic_weight = np.full(2, plastic_init, dtype=float)
                for trial in range(int(args.training_trials) + 1):
                    signed_pe = _signed_pe(
                        plastic_weight,
                        plasticity_mode=plasticity_mode,
                        fixed_weight=float(args.fixed_weight),
                        pv_activity=float(args.pv_activity),
                    )
                    traces.append(
                        TracePoint(
                            plasticity_mode=plasticity_mode,
                            circuit=circuit,
                            initial_signed_pe=float(initial_pe),
                            plastic_weight_init=float(plastic_init),
                            trial=trial,
                            trial_average_signed_pe=float(signed_pe),
                            plastic_weight_mean=float(np.mean(plastic_weight)),
                        )
                    )
                    if trial < int(args.training_trials):
                        plastic_weight = _update_weight(
                            plastic_weight,
                            signed_pe,
                            plasticity_mode=plasticity_mode,
                            learning_rate=float(args.learning_rate),
                            pv_activity=float(args.pv_activity),
                        )
    return traces


def write_trace_csv(traces: list[TracePoint], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "signed_pe_convergence_traces.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TracePoint.__dataclass_fields__))
        writer.writeheader()
        for trace in traces:
            writer.writerow(trace.__dict__)


def write_metadata(args: argparse.Namespace, output_dir: Path) -> None:
    metadata = {
        "training_trials": int(args.training_trials),
        "learning_rate": float(args.learning_rate),
        "fixed_weight": float(args.fixed_weight),
        "pv_activity": float(args.pv_activity),
        "max_initial_abs_pe": float(args.max_initial_abs_pe),
        "n_initializations": int(args.n_initializations),
        "plasticity_modes": {
            "lat_hebbian": {
                "plastic_weight": "w_LAT in both PPE and NPE",
                "signed_pe": "fixed_weight - w_LAT * pv_activity",
                "update": "delta w_LAT = learning_rate * signed_PE * PV",
            },
            "ff_fb_anti_hebbian": {
                "plastic_weight": "w_FF in PPE; w_FB in NPE",
                "signed_pe": "plastic_weight - fixed_weight",
                "update": "delta w = -learning_rate * signed_PE * presynaptic_input",
            },
        },
        "note": (
            "The plastic weight is the only varied weight. Non-plastic opponent "
            "weights are fixed at fixed_weight. Presynaptic activity is one on "
            "the trained channel; two trained channels are averaged per trial."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def plot_traces(traces: list[TracePoint], output_dir: Path, image_format: str) -> Path:
    subplot_titles = {
        ("lat_hebbian", "PPE"): r"PPE: $\frac{dw_{\mathrm{LAT}}(t)}{dt}=e(t)\cdot p(t)$",
        ("lat_hebbian", "NPE"): r"NPE: $\frac{dw_{\mathrm{LAT}}(t)}{dt}=e(t)\cdot p(t)$",
        ("ff_fb_anti_hebbian", "PPE"): r"PPE: $\frac{d\mathbf{w}_{\mathrm{FF}}(t)}{dt}=-e(t)\cdot \mathbf{x}(t)$",
        ("ff_fb_anti_hebbian", "NPE"): r"NPE: $\frac{d\mathbf{w}_{\mathrm{FB}}(t)}{dt}=-e(t)\cdot \mathbf{c}(t)$",
    }
    modes: tuple[PlasticityMode, ...] = ("lat_hebbian", "ff_fb_anti_hebbian")
    circuits: tuple[Circuit, ...] = ("PPE", "NPE")
    initial_pes = sorted({trace.initial_signed_pe for trace in traces})
    max_abs_pe = max(abs(value) for value in initial_pes)
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-max_abs_pe, vmax=max_abs_pe)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.86, bottom=0.10, top=0.88, wspace=0.20, hspace=0.58)
    fig.text(0.475, 0.965, "Hebbian plasticity of inhibitory synapses", ha="center", va="top", fontsize=11)
    fig.text(0.475, 0.505, "Anti-Hebbian plasticity of excitatory synapses", ha="center", va="top", fontsize=11)
    for row_idx, plasticity_mode in enumerate(modes):
        for col_idx, circuit in enumerate(circuits):
            ax = axes[row_idx, col_idx]
            for initial_pe in initial_pes:
                rows = [
                    trace
                    for trace in traces
                    if trace.plasticity_mode == plasticity_mode
                    and trace.circuit == circuit
                    and trace.initial_signed_pe == initial_pe
                ]
                rows.sort(key=lambda trace: trace.trial)
                ax.plot(
                    [trace.trial for trace in rows],
                    [trace.trial_average_signed_pe for trace in rows],
                    color=cmap(norm(initial_pe)),
                    lw=1.8,
                )
            ax.axhline(0.0, color="0.20", lw=0.8)
            ax.set_title(subplot_titles[(plasticity_mode, circuit)], fontsize=9)
            if col_idx == 0:
                ax.set_ylabel("Signed prediction error (e(t))")
            if row_idx == len(modes) - 1:
                ax.set_xlabel("Training trial")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=axes, fraction=0.035, pad=0.025)
    colorbar.set_label("Initial signed prediction error (e(t))")
    fig.savefig(output_dir / f"signed_pe_convergence_2x2.{image_format}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_dir / f"signed_pe_convergence_2x2.{image_format}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces = simulate_traces(args)
    write_trace_csv(traces, args.output_dir)
    write_metadata(args, args.output_dir)
    plot_path = plot_traces(traces, args.output_dir, args.image_format)
    print(f"Wrote {plot_path}")
    print(f"Wrote {args.output_dir / 'signed_pe_convergence_traces.csv'}")
    print(f"Wrote {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
