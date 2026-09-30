"""Plot traces and six transition vectors using only ground_truth.csv and model.csv."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SECTORS = ("+NO axis", "+O axis", "-NO axis")
COLORS = ("#3159a7", "#ef202f", "#e68600")
SOURCES = ("familiar", "novel")
LABELS = {"naive": "Naive", "expert": "Expert", "expert_no_fb": "FB silencing",
          "expert_no_lat": "PV silencing", "expert_no_fb_no_lat": "FB & PV silencing"}


def save(fig, path):
    for extension in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{extension}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_traces(table, path):
    traces = table.loc[table.record_type.eq("trace")]
    keys = ["source", "sector", "condition_key", "response_type", "time_seconds"]
    # Average images within each neuron first, then give neurons equal weight.
    neurons = traces.groupby(keys + ["observation_id"])["response"].mean()
    means = neurons.groupby(keys).mean().reset_index()
    conditions = [c for c in LABELS if c in means.condition_key.unique()]
    fig, axes = plt.subplots(3, 2 * len(conditions), figsize=(3.1 * len(conditions), 6),
                             sharex=True, sharey="row", squeeze=False, layout="constrained")
    for row, sector in enumerate(SECTORS):
        for column, (condition, source) in enumerate((c, s) for c in conditions for s in SOURCES):
            ax = axes[row, column]
            frame = means.loc[means.sector.eq(sector) & means.source.eq(source) & means.condition_key.eq(condition)]
            ax.axvspan(0, 1, color="0.92")
            ax.axhline(0, color="0.8", lw=0.6)
            for response, color in (("NO", "black"), ("O", "red")):
                line = frame.loc[frame.response_type.eq(response)]
                ax.plot(line.time_seconds, line.response, color=color, lw=1.2, label=response)
            ax.set_xlim(-1, 3)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(f"{LABELS[condition]}\n{source.capitalize()}", fontsize=9)
            if column == 0:
                ax.set_ylabel(sector.replace(" axis", "") + "\nResponse (baseline SD)", color=COLORS[row])
            if row == 2:
                ax.set_xlabel("Time (s)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, path)


def plot_vectors(data, model, path, uncertainty):
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), layout="constrained")
    # TODO: get percentage of transitions in each sector
    for row, (table, label) in enumerate(((data, "Data"), (model, "Model"))):
        for column, source in enumerate(SOURCES):
            ax = axes[row, column]
            vectors = table.loc[table.record_type.eq("transition") & table.source.eq(source)]
            extent = 0.5
            for sector, color in zip(SECTORS, COLORS):
                points = vectors.loc[vectors.sector.eq(sector), ["delta_NO", "delta_O"]].to_numpy()
                center = points.mean(axis=0)
                radius = np.zeros(2)
                if len(points) > 1:
                    covariance = np.cov(points, rowvar=False)
                    if uncertainty == "sem":
                        covariance /= len(points)
                    values, directions = np.linalg.eigh(covariance)
                    angle = np.degrees(np.arctan2(directions[1, -1], directions[0, -1]))
                    width, height = 2 * np.sqrt(np.maximum(values[::-1], 0))
                    ax.add_patch(Ellipse(center, width, height, angle=angle, color=color, alpha=0.18))
                    radius = np.sqrt(np.diag(covariance))
                extent = max(extent, np.max(np.abs(center) + radius) * 1.2)
                ax.annotate("", xy=center, xytext=(0, 0),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5, mutation_scale=14))
                ax.plot([], [], color=color, label=sector.replace(" axis", ""))
            ax.axhline(0, color="0.7", lw=0.7)
            ax.axvline(0, color="0.7", lw=0.7)
            ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal",
                   title=f"{label}: {source}", xlabel="Delta NO", ylabel="Delta O")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "ground_truth.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "model.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    data, model = pd.read_csv(args.data), pd.read_csv(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    plot_traces(data, args.output / "ground_truth_traces")
    plot_traces(model, args.output / "model_traces")
    for uncertainty in ("sd", "sem"):
        plot_vectors(data, model, args.output / f"transition_vectors_{uncertainty}", uncertainty)


if __name__ == "__main__":
    main()
