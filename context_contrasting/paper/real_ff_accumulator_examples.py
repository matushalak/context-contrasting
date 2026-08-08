"""Real-model FF weight examples for the activity accumulator.

This diagnostic uses the actual paper `transition_templates` configs and
`minimal_divisive.CCNeuron`, then records FF weights during the familiar training
phase. It is meant to complement `synthetic_ff_accumulator_study.py`, which only
isolates the algebra of the FF update.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import model_scatter as ms
from . import transition_templates as tt
from .minimal_divisive import CCNeuron


OUT_DIR = Path(__file__).resolve().parent / "outputs_real_ff_accumulator_examples"
N_STEPS_PER_PHASE = 400
TRAINING_TRIALS = 7
SEED = 7151
OLD_EFFECTIVE_LR_FF = 0.0155 * (200 / N_STEPS_PER_PHASE)
ACCUMULATOR_EFFECTIVE_LR_FF = OLD_EFFECTIVE_LR_FF
ACCUMULATOR_SCALE = 2000.0


def model_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def old_fixed_config(config: dict[str, Any], *, width: int) -> dict[str, Any]:
    cfg = dict(config)
    cfg["use_ff_activity_accumulator"] = False
    cfg["lr_ff"] = OLD_EFFECTIVE_LR_FF
    cfg["ff_plasticity_scale"] = 8.0 if width == 2 else 2.0
    return cfg


def accumulator_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    cfg["use_ff_activity_accumulator"] = True
    cfg["ff_accumulator_alpha_factor"] = 0.05
    cfg["ff_accumulator_power"] = 2.0
    cfg["ff_accumulator_scale"] = ACCUMULATOR_SCALE
    cfg["lr_ff"] = ACCUMULATOR_EFFECTIVE_LR_FF
    cfg["ff_plasticity_scale"] = 1.0
    return cfg


def run_training_trace(config: dict[str, Any], label: str) -> pd.DataFrame:
    model = CCNeuron(**model_kwargs(config))
    x_train, c_train = ms._build_model_scatter_training_stimuli(
        n_steps_per_phase=N_STEPS_PER_PHASE,
        n_trials=TRAINING_TRIALS,
        order="randomized",
        seed=SEED,
    )
    model._reset_state()
    rows: list[dict[str, Any]] = []
    for step in range(x_train.shape[0]):
        x, y_t, y_next, p, c = model(x_train[step], c_train[step])
        model.update(x, y_t, y_next, p, c)
        acc = float(model.ff_activity_accumulator.ema) if getattr(model, "use_ff_activity_accumulator", False) else np.nan
        if np.isfinite(acc):
            scale = float(model.ff_accumulator_scale) * (acc ** float(model.ff_accumulator_power))
        else:
            scale = float(model.ff_plasticity_scale)
        rows.append(
            {
                "label": label,
                "step": step,
                "trial_step": step % N_STEPS_PER_PHASE,
                "phase": "stimulus" if bool(np.any(x.detach().cpu().numpy() > 0)) else "iti",
                "stimulus": int(np.argmax(x.detach().cpu().numpy())) if bool(np.any(x.detach().cpu().numpy() > 0)) else -1,
                "y": float(y_next),
                "accumulator": acc,
                "effective_scale": scale,
                "effective_lr": float(model.lr_ff) * scale,
                "w0": float(model.w_ff[0]),
                "w1": float(model.w_ff[1]),
                "w2": float(model.w_ff[2]),
            }
        )
    return pd.DataFrame(rows)


def plot_traces(trace_sets: list[tuple[str, pd.DataFrame]], path: Path) -> None:
    fig, axes = plt.subplots(len(trace_sets), 1, figsize=(10.5, 2.7 * len(trace_sets)), sharex=True)
    if len(trace_sets) == 1:
        axes = [axes]
    for ax, (label, df) in zip(axes, trace_sets):
        ax.plot(df["step"], df["w0"], label="w0", lw=1.8)
        ax.plot(df["step"], df["w1"], label="w1", lw=1.8)
        ax.plot(df["step"], df["w2"], label="w2", lw=1.8)
        ax.set_ylabel("FF weight")
        ax.set_title(label)
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(df["step"], df["y"], color="0.25", alpha=0.35, lw=1.0, label="y")
        if df["accumulator"].notna().any():
            ax2.plot(df["step"], df["accumulator"], color="tab:purple", alpha=0.75, lw=1.1, label="acc")
        ax2.set_ylabel("y / acc")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", ncol=5, fontsize=8)
    axes[-1].set_xlabel("training step")
    fig.suptitle("Actual minimal_divisive training traces from transition_templates", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tt.configure_model_scatter(N_STEPS_PER_PHASE)

    examples: list[tuple[str, dict[str, Any], int]] = []
    broad_fam_pair = ms._center_config("strong_broad_FFonly", tuned_indices=(0, 1))
    broad_canonical = ms._center_config("strong_broad_FFonly", tuned_indices=(0, 2))
    narrow_familiar = ms._center_config("narrow_very_weak_FB_all", tuned_indices=(0,))
    examples.extend(
        [
            ("broad strong FFonly tuned f1+f2", broad_fam_pair, 2),
            ("broad strong FFonly tuned f1+novel", broad_canonical, 2),
            ("narrow weak FB tuned f1", narrow_familiar, 1),
        ]
    )

    traces: list[pd.DataFrame] = []
    plot_sets: list[tuple[str, pd.DataFrame]] = []
    summary_rows: list[dict[str, Any]] = []
    for base_label, base_config, width in examples:
        variants = [
            ("old fixed", old_fixed_config(base_config, width=width)),
            ("accumulator", accumulator_config(base_config)),
        ]
        for variant_label, cfg in variants:
            label = f"{base_label} - {variant_label}"
            df = run_training_trace(cfg, label)
            traces.append(df)
            plot_sets.append((label, df))
            stim = df.loc[df["phase"].eq("stimulus")]
            final = df.iloc[-1]
            summary_rows.append(
                {
                    "label": label,
                    "width": width,
                    "lr_ff": cfg["lr_ff"],
                    "use_ff_activity_accumulator": cfg.get("use_ff_activity_accumulator", False),
                    "ff_plasticity_scale": cfg.get("ff_plasticity_scale"),
                    "mean_stim_y": stim["y"].mean(),
                    "mean_stim_accumulator": stim["accumulator"].mean(),
                    "mean_stim_effective_scale": stim["effective_scale"].mean(),
                    "mean_stim_effective_lr": stim["effective_lr"].mean(),
                    "w0_initial": df.iloc[0]["w0"],
                    "w1_initial": df.iloc[0]["w1"],
                    "w2_initial": df.iloc[0]["w2"],
                    "w0_final": final["w0"],
                    "w1_final": final["w1"],
                    "w2_final": final["w2"],
                }
            )

    all_traces = pd.concat(traces, ignore_index=True)
    all_traces.to_csv(OUT_DIR / "real_ff_training_traces.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "real_ff_training_summary.csv", index=False)
    plot_traces(plot_sets, OUT_DIR / "real_ff_weight_examples.png")


if __name__ == "__main__":
    main()
