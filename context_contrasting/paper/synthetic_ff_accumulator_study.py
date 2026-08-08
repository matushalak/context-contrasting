"""Synthetic FF-accumulator diagnostics.

This file isolates the proposed feedforward anti-Hebbian update from the full
PyC/PV circuit. It is intentionally small and deterministic: three FF weights,
two familiar stimuli during training, no novel stimulus exposure, and an
always-on activity accumulator that continues through ITI.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent / "outputs_ff_accumulator_synthetic_study"


@dataclass(frozen=True)
class SyntheticConfig:
    lr_ff: float = 0.0155 * 0.5  # effective 400-step run value after 200/400 scaling
    alpha: float = 0.05 * 0.05  # pyc_decay * ff_accumulator_alpha_factor = 0.0025
    power: int = 2
    accumulator_scale: float = 2000.0
    stim_steps: int = 100
    iti_steps: int = 300
    cycles: int = 7
    baseline_y: float = 0.035
    response_gain: float = 1.0
    w_init: tuple[float, float, float] = (0.19, 0.19, 0.008)
    stimuli: tuple[int, ...] = (0, 1)
    use_accumulator: bool = True
    fixed_scale: float = 1.0


def decay_residual(alpha: float, steps: int) -> float:
    return float((1.0 - alpha) ** steps)


def run_synthetic(config: SyntheticConfig) -> pd.DataFrame:
    w = np.asarray(config.w_init, dtype=float).copy()
    accumulator = 0.0
    rows: list[dict] = []
    step = 0
    trial = 0

    def record(phase: str, stimulus: int | None, y: float, plasticity_scale: float) -> None:
        rows.append(
            {
                "step": step,
                "trial": trial,
                "phase": phase,
                "stimulus": -1 if stimulus is None else stimulus,
                "y": y,
                "accumulator": accumulator,
                "effective_scale": plasticity_scale,
                "effective_lr": config.lr_ff * plasticity_scale,
                "w0": w[0],
                "w1": w[1],
                "w2": w[2],
            }
        )

    for cycle in range(config.cycles):
        for stimulus in config.stimuli:
            trial += 1
            x = np.zeros(3, dtype=float)
            x[stimulus] = 1.0
            for _ in range(config.stim_steps):
                y = config.baseline_y + config.response_gain * float(np.dot(w, x))
                y = float(np.clip(y, 0.0, 1.0))
                accumulator = (1.0 - config.alpha) * accumulator + config.alpha * y
                plasticity_scale = config.accumulator_scale * (accumulator**config.power) if config.use_accumulator else config.fixed_scale
                dw = -config.lr_ff * plasticity_scale * y * x * w
                w += dw
                w = np.clip(w, 0.0, None)
                record("stimulus", stimulus, y, plasticity_scale)
                step += 1

            for _ in range(config.iti_steps):
                y = config.baseline_y
                accumulator = (1.0 - config.alpha) * accumulator + config.alpha * y
                plasticity_scale = config.accumulator_scale * (accumulator**config.power) if config.use_accumulator else config.fixed_scale
                record("iti", None, y, plasticity_scale)
                step += 1

    return pd.DataFrame(rows)


def summarize_final(config: SyntheticConfig, label: str) -> dict:
    df = run_synthetic(config)
    final = df.iloc[-1]
    stim = df.loc[df["phase"].eq("stimulus")]
    return {
        "label": label,
        "lr_ff": config.lr_ff,
        "alpha": config.alpha,
        "tau_steps_approx": 1.0 / config.alpha,
        "residual_300": decay_residual(config.alpha, 300),
        "residual_400": decay_residual(config.alpha, 400),
        "power": config.power,
        "accumulator_scale": config.accumulator_scale,
        "stim_steps": config.stim_steps,
        "iti_steps": config.iti_steps,
        "w0_final": final["w0"],
        "w1_final": final["w1"],
        "w2_final": final["w2"],
        "acc_final": final["accumulator"],
        "mean_stim_acc": stim["accumulator"].mean(),
        "mean_stim_effective_scale": stim["effective_scale"].mean(),
        "mean_stim_effective_lr": stim["effective_lr"].mean(),
    }


def plot_weight_traces(configs: list[tuple[str, SyntheticConfig]], path: Path, title: str) -> None:
    fig, axes = plt.subplots(len(configs), 1, figsize=(8.0, 2.5 * len(configs)), sharex=True)
    if len(configs) == 1:
        axes = [axes]
    for ax, (label, config) in zip(axes, configs):
        df = run_synthetic(config)
        ax.plot(df["step"], df["w0"], label="w0 familiar 1", lw=1.8)
        ax.plot(df["step"], df["w1"], label="w1 familiar 2", lw=1.8)
        ax.plot(df["step"], df["w2"], label="w2 novel", lw=1.8)
        ax2 = ax.twinx()
        ax2.plot(df["step"], df["accumulator"], color="0.35", alpha=0.45, lw=1.2, label="accumulator")
        ax.set_ylabel("FF weight")
        ax2.set_ylabel("acc")
        ax.set_title(label)
        ax.grid(alpha=0.25)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", ncol=4, fontsize=8)
    axes[-1].set_xlabel("training step")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_broad_narrow_examples(path: Path) -> pd.DataFrame:
    old_lr = 0.0155 * 0.5
    acc_lr = 0.0155 * 0.5
    examples = [
        (
            "broad old fixed scale=8",
            SyntheticConfig(
                lr_ff=old_lr,
                use_accumulator=False,
                fixed_scale=8.0,
                w_init=(0.19, 0.19, 0.008),
                response_gain=1.0,
            ),
        ),
        (
            "broad accumulator p=2",
            SyntheticConfig(
                lr_ff=acc_lr,
                use_accumulator=True,
                w_init=(0.19, 0.19, 0.008),
                response_gain=1.0,
            ),
        ),
        (
            "narrow old fixed scale=2",
            SyntheticConfig(
                lr_ff=old_lr,
                use_accumulator=False,
                fixed_scale=2.0,
                w_init=(0.105, 0.006, 0.006),
                response_gain=1.0,
            ),
        ),
        (
            "narrow accumulator p=2",
            SyntheticConfig(
                lr_ff=acc_lr,
                use_accumulator=True,
                w_init=(0.105, 0.006, 0.006),
                response_gain=1.0,
            ),
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.5), sharex=True, sharey=True)
    summary_rows: list[dict] = []
    for ax, (label, config) in zip(axes.reshape(-1), examples):
        df = run_synthetic(config)
        final = df.iloc[-1]
        stim = df.loc[df["phase"].eq("stimulus")]
        summary_rows.append(
            {
                "label": label,
                "lr_ff": config.lr_ff,
                "use_accumulator": config.use_accumulator,
                "fixed_scale": config.fixed_scale,
                "accumulator_scale": config.accumulator_scale,
                "mean_stim_acc": stim["accumulator"].mean(),
                "mean_stim_effective_scale": stim["effective_scale"].mean(),
                "mean_stim_effective_lr": stim["effective_lr"].mean(),
                "w0_initial": config.w_init[0],
                "w1_initial": config.w_init[1],
                "w2_initial": config.w_init[2],
                "w0_final": final["w0"],
                "w1_final": final["w1"],
                "w2_final": final["w2"],
            }
        )
        ax.plot(df["step"], df["w0"], label="w0 familiar 1", lw=2.0)
        ax.plot(df["step"], df["w1"], label="w1 familiar 2", lw=2.0)
        ax.plot(df["step"], df["w2"], label="w2 novel", lw=2.0)
        ax2 = ax.twinx()
        ax2.plot(df["step"], df["accumulator"], color="0.35", alpha=0.45, lw=1.2, label="accumulator")
        ax.set_title(label)
        ax.set_ylabel("FF weight")
        ax2.set_ylabel("acc")
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("training step")
    axes[-1, 1].set_xlabel("training step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Broad vs narrow synthetic examples: old fixed scale vs activity accumulator", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return pd.DataFrame(summary_rows)


def plot_final_grid(rows: list[dict], x: str, path: Path, title: str, logx: bool = False) -> None:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for weight in ("w0_final", "w1_final", "w2_final"):
        axes[0].plot(df[x], df[weight], marker="o", label=weight.replace("_final", ""))
    axes[0].set_ylabel("final FF weight")
    axes[0].set_xlabel(x)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].plot(df[x], df["mean_stim_acc"], marker="o", label="mean stim accumulator")
    axes[1].plot(df[x], df["mean_stim_effective_scale"], marker="o", label="mean acc^p")
    axes[1].plot(df[x], df["mean_stim_effective_lr"], marker="o", label="mean lr*acc^p")
    axes[1].set_ylabel("scale")
    axes[1].set_xlabel(x)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    if logx:
        axes[0].set_xscale("log")
        axes[1].set_xscale("log")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = SyntheticConfig()

    # Adapt-equivalent alpha: pyc_decay * 0.2 = 0.01, compared to tau~400 alpha=0.0025.
    trace_configs = [
        ("adapt-like alpha=0.01, lr=21.31", replace(base, alpha=0.01, lr_ff=21.3125)),
        ("tau~400 alpha=0.0025, lr=23.85", base),
        ("slower alpha=0.00173 (half after 400), lr=25.16", replace(base, alpha=0.00173137, lr_ff=25.1565)),
    ]
    plot_weight_traces(trace_configs, OUT_DIR / "weight_traces_timescale.png", "Timescale controls accumulator carry-over through ITI")
    broad_narrow_summary = plot_broad_narrow_examples(OUT_DIR / "broad_narrow_weight_examples.png")
    broad_narrow_summary.to_csv(OUT_DIR / "broad_narrow_weight_examples.csv", index=False)

    lr_rows = [
        summarize_final(replace(base, lr_ff=lr), f"lr={lr:g}")
        for lr in [0.002, 0.004, 0.00775, 0.012, 0.016, 0.020]
    ]
    plot_final_grid(lr_rows, "lr_ff", OUT_DIR / "sweep_lr_ff.png", "Shared lr_ff sweep at tau~400, power=2")

    alpha_values = [0.0005, 0.001, 0.00173137, 0.0025, 0.005, 0.01, 0.015]
    alpha_rows = [
        summarize_final(replace(base, alpha=alpha), f"alpha={alpha:g}")
        for alpha in alpha_values
    ]
    plot_final_grid(alpha_rows, "alpha", OUT_DIR / "sweep_accumulator_alpha.png", "Accumulator alpha sweep at fixed lr_ff and power=2", logx=True)

    power_rows = [
        summarize_final(replace(base, power=power, lr_ff=lr), f"power={power}, lr={lr:g}")
        for power, lr in [(1, 0.0006), (2, 0.00775), (3, 0.195)]
    ]
    plot_final_grid(power_rows, "power", OUT_DIR / "sweep_accumulator_power.png", "Accumulator power changes the lr scale needed")

    window_rows = []
    for stim_steps, iti_steps in [(50, 150), (50, 350), (100, 100), (100, 300), (100, 600), (200, 300)]:
        label = f"stim={stim_steps}, iti={iti_steps}"
        window_rows.append(summarize_final(replace(base, stim_steps=stim_steps, iti_steps=iti_steps), label))
    pd.DataFrame(window_rows).to_csv(OUT_DIR / "window_sweep_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    x = np.arange(len(window_rows))
    width = 0.25
    window_df = pd.DataFrame(window_rows)
    ax.bar(x - width, window_df["w0_final"], width, label="w0 familiar 1")
    ax.bar(x, window_df["w1_final"], width, label="w1 familiar 2")
    ax.bar(x + width, window_df["w2_final"], width, label="w2 novel")
    ax.set_xticks(x)
    ax.set_xticklabels(window_df["label"], rotation=30, ha="right")
    ax.set_ylabel("final FF weight")
    ax.set_title("Stimulus and ITI duration change accumulated plasticity")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sweep_stim_iti_windows.png", dpi=180)
    plt.close(fig)

    summary = pd.concat(
        [
            pd.DataFrame(lr_rows).assign(sweep="lr_ff"),
            pd.DataFrame(alpha_rows).assign(sweep="alpha"),
            pd.DataFrame(power_rows).assign(sweep="power"),
            pd.DataFrame(window_rows).assign(sweep="windows"),
        ],
        ignore_index=True,
    )
    summary.to_csv(OUT_DIR / "synthetic_summary.csv", index=False)

    notes = {
        "update": "w_i <- w_i - lr_ff * accumulator_scale * accumulator**power * y * x_i * w_i",
        "accumulator": "acc <- (1-alpha)*acc + alpha*y, every step including ITI",
        "tau400_alpha": 1.0 / 400.0,
        "tau400_residual_400": decay_residual(1.0 / 400.0, 400),
        "adapt_like_alpha": 0.01,
        "adapt_like_residual_400": decay_residual(0.01, 400),
        "baseline_y_during_iti": base.baseline_y,
    }
    pd.Series(notes).to_json(OUT_DIR / "notes.json", indent=2)


if __name__ == "__main__":
    main()
