"""Plot traces and transition vectors using only ground_truth.csv and model.csv."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
SECTORS = ("+NO axis", "+O axis", "-NO axis")
DATA_SECTORS = (*SECTORS, "-O axis")
COLORS = {"+NO axis": "#3159a7", "+O axis": "#ef202f",
          "-NO axis": "#e68600", "-O axis": "#16843e"}
SOURCES = ("familiar", "novel")
LABELS = {"naive": "Naive", "expert": "Expert", "expert_no_fb": "FB silencing",
          "expert_no_lat": "PV silencing", "expert_no_fb_no_lat": "FB & PV silencing"}


def save(fig, path):
    for extension in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{extension}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def observation_columns(table):
    return ["observation_id"] + (["seed"] if "seed" in table and table.seed.notna().any() else [])


def summarize_traces(samples, keys):
    summary = samples.groupby(keys)["response"].agg(response="mean", sd="std", n="count").reset_index()
    summary["sem"] = summary.sd / np.sqrt(summary.n)
    return summary


def trace_observations(table):
    traces = table.loc[table.record_type.eq("trace")]
    keys = ["source", "sector", "condition_key", "response_type", "time_seconds"]
    # Images are repeated measures, not independent neurons or seed replicates.
    return traces.groupby(keys + observation_columns(traces))["response"].mean().reset_index()


def plot_traces(table, path, uncertainty="sd"):
    neurons = trace_observations(table)
    keys = ["source", "sector", "condition_key", "response_type", "time_seconds"]
    means = summarize_traces(neurons, keys)
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
                ax.fill_between(line.time_seconds, line.response - line[uncertainty],
                                line.response + line[uncertainty], color=color, alpha=0.18, linewidth=0)
            ax.set_xlim(-1, 3)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(f"{LABELS[condition]}\n{source.capitalize()}", fontsize=9)
            if column == 0:
                ax.set_ylabel(sector.replace(" axis", "") + "\nResponse (baseline SD)", color=COLORS[sector])
            if row == 2:
                ax.set_xlabel("Time (s)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, path)


def empirical_sector_weights(data):
    vectors = data.loc[data.record_type.eq("transition") & data.sector.isin(SECTORS)]
    counts = vectors.groupby(["source", "sector"])["observation_id"].nunique()
    return counts / counts.groupby(level="source").transform("sum")


def average_traces(table, weights=None):
    observations = trace_observations(table)
    keys = ["source", "sector", "condition_key", "response_type", "time_seconds"]
    output_keys = [key for key in keys if key != "sector"]
    if weights is None:
        return summarize_traces(observations, output_keys)
    # Preserve covariance across templates sharing a replicate seed: weight first,
    # then summarize complete seed-level population traces, never pool templates.
    if "seed" not in observations:
        observations["seed"] = 0
    sectors = observations.groupby(keys + ["seed"])["response"].mean().reset_index()
    sectors["weight"] = [weights.loc[(source, sector)] for source, sector in
                         zip(sectors.source, sectors.sector)]
    sectors["response"] *= sectors["weight"]
    counts = sectors.groupby(output_keys + ["seed"])["sector"].nunique()
    expected = counts.index.get_level_values("source").map(weights.groupby(level="source").size())
    if not np.array_equal(counts.to_numpy(), expected.to_numpy()):
        raise ValueError("Weighted model traces require every sector for each seed/time/condition")
    samples = sectors.groupby(output_keys + ["seed"])["response"].sum().reset_index()
    return summarize_traces(samples, output_keys)


def plot_average_traces(data, model, path, uncertainty="sd"):
    means = ((average_traces(data), "Data"),
             (average_traces(model, empirical_sector_weights(data)), "Weighted model"))
    columns = [(condition, source) for condition in ("naive", "expert") for source in SOURCES]
    fig, axes = plt.subplots(2, len(columns), figsize=(8.2, 4.2), sharex=True, sharey="row",
                             squeeze=False, layout="constrained")
    for row, (table, label) in enumerate(means):
        for column, (condition, source) in enumerate(columns):
            ax = axes[row, column]
            frame = table.loc[table.source.eq(source) & table.condition_key.eq(condition)]
            ax.axvspan(0, 1, color="0.92")
            ax.axhline(0, color="0.8", lw=0.6)
            for response, color in (("NO", "black"), ("O", "red")):
                line = frame.loc[frame.response_type.eq(response)]
                ax.plot(line.time_seconds, line.response, color=color, lw=1.2, label=response)
                ax.fill_between(line.time_seconds, line.response - line[uncertainty],
                                line.response + line[uncertainty], color=color, alpha=0.18, linewidth=0)
            ax.set_xlim(-1, 3)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(f"{LABELS[condition]}\n{source.capitalize()}", fontsize=9)
            if column == 0:
                ax.set_ylabel(f"{label}\nResponse (baseline SD)")
            if row == 1:
                ax.set_xlabel("Time (s)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, path)


def model_transition_vectors(model, condition):
    traces = model.loc[model.record_type.eq("trace") & model.time_seconds.between(0, 1)]
    keys = ["source", "sector"] + observation_columns(traces) + ["condition_key", "response_type"]
    responses = traces.groupby(keys)["response"].mean().unstack(["condition_key", "response_type"])
    vectors = responses.index.to_frame(index=False)
    for response in ("NO", "O"):
        vectors[f"delta_{response}"] = (responses[(condition, response)] -
                                         responses[("naive", response)]).to_numpy()
    return vectors


def overall_transition(vectors, source, weights=None):
    frame = vectors.loc[vectors.source.eq(source)]
    if weights is None:
        return frame[["delta_NO", "delta_O"]].mean().to_numpy()
    means = frame.groupby("sector")[["delta_NO", "delta_O"]].mean()
    return sum(weights.loc[(source, sector)] * means.loc[sector].to_numpy() for sector in SECTORS)


def plot_vectors(data, model, path, uncertainty="sd", condition="expert"):
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), layout="constrained",
                             sharex='all', sharey = 'all')
    weights = empirical_sector_weights(data)
    data_vectors = data.loc[data.record_type.eq("transition")]
    model_vectors = model_transition_vectors(model, condition)
    for column, (vectors, label) in enumerate(((data_vectors, "Data"),
                                                (model_vectors, f"Model ({LABELS[condition]})"))):
        for row, source in enumerate(SOURCES):
            ax = axes[row, column]
            source_vectors = vectors.loc[vectors.source.eq(source)]
            extent = 0.5
            for sector in DATA_SECTORS if label == "Data" else SECTORS:
                color = COLORS[sector]
                points = source_vectors.loc[source_vectors.sector.eq(sector),
                                             ["delta_NO", "delta_O"]].to_numpy()
                if not len(points):
                    continue
                center = points.mean(axis=0)
                radius = np.zeros(2)
                # Neuron-to-neuron (data) or seed-to-seed (model) covariance.
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
            overall = overall_transition(vectors, source, None if label == "Data" else weights)
            extent = max(extent, np.max(np.abs(overall)) * 1.2)
            ax.annotate("", xy=overall, xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=3.2, mutation_scale=16))
            ax.plot([], [], color="black", lw=3.2, label="Overall transition")
            ax.axhline(0, color="0.7", lw=0.7)
            ax.axvline(0, color="0.7", lw=0.7)
            ax.set(xlim=(-2, 2), ylim=(-2, 2), aspect="equal",
                   title=f"{label}: {source}", xlabel="Delta NO", ylabel="Delta O")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, path)


def transition_cosine_sim(transitions_A, transitions_B):
    dot = np.sum(transitions_A * transitions_B, axis=-1)
    norm = np.linalg.norm(transitions_A, axis=-1) * np.linalg.norm(transitions_B, axis=-1)
    return np.divide(dot, norm, out=np.full_like(dot, np.nan, dtype=float), where=norm != 0)


def cosine_similarities(data, model):
    data_vectors = data.loc[data.record_type.eq("transition")]
    weights = empirical_sector_weights(data)
    rows = []
    for condition in tuple(LABELS)[1:]:
        model_vectors = model_transition_vectors(model, condition)
        for source in SOURCES:
            for sector in SECTORS:
                observed = data_vectors.loc[data_vectors.source.eq(source) &
                                             data_vectors.sector.eq(sector),
                                             ["delta_NO", "delta_O"]].to_numpy()
                predicted = model_vectors.loc[model_vectors.source.eq(source) &
                                               model_vectors.sector.eq(sector),
                                               ["delta_NO", "delta_O"]].mean().to_numpy()
                for similarity in transition_cosine_sim(observed, predicted):
                    rows.append((source, LABELS[condition], sector.replace(" axis", ""),
                                 float(similarity)))
            observed = overall_transition(data_vectors, source)
            predicted = overall_transition(model_vectors, source, weights)
            rows.append((source, LABELS[condition], "Overall",
                         float(transition_cosine_sim(observed, predicted))))

    return pd.DataFrame(rows, columns=["source", "condition", "sector", "similarity"])


def plot_cosine_sim(data, model, path, errorbars=False):
    similarities = cosine_similarities(data, model)
    order = list(LABELS.values())[1:]
    hues = [sector.replace(" axis", "") for sector in SECTORS] + ["Overall"]
    palette = {sector.replace(" axis", ""): color for sector, color in COLORS.items()}
    palette["Overall"] = "black"
    fig, axes = plt.subplots(2, 1, figsize=(4, 6), sharey=True, layout="constrained")
    for ax, source in zip(axes, SOURCES):
        sns.pointplot(data=similarities.loc[similarities.source.eq(source)], x="condition",
                      y="similarity", hue="sector", order=order, hue_order=hues,
                      palette=palette, dodge=0.45, errorbar="sd" if errorbars else None, ax=ax,
                      linestyle = 'none')
        ax.axhline(0, color="0.7", lw=0.7)
        ax.set(title=source.capitalize(), xlabel=None, ylabel="Cosine similarity", ylim=(-1.05, 1.05))
        ax.tick_params(axis="x", rotation=25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].get_legend().remove()
    axes[0].legend(frameon=False, fontsize=8, title=None)
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "ground_truth.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "model.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    parser.add_argument("--include-minus-o", action="store_true",
                        help="Include empirical -O cells in overall vectors and traces")
    parser.add_argument("--cosine-errorbars", action="store_true",
                        help="Plot SD across neuron-wise cosine similarities")
    parser.add_argument("--uncertainty", choices=("sd", "sem", "both"), default="both",
                        help="Trace bands and vector covariance ellipses (default: both)")
    args = parser.parse_args()
    data, model = pd.read_csv(args.data), pd.read_csv(args.model)
    plotted_data = data if args.include_minus_o else data.loc[~data.sector.eq("-O axis")]
    args.output.mkdir(parents=True, exist_ok=True)
    vectors_output = args.output / "transition_vectors"
    vectors_output.mkdir(exist_ok=True)
    for uncertainty in ("sd", "sem") if args.uncertainty == "both" else (args.uncertainty,):
        suffix = "" if uncertainty == "sd" else "_sem"
        plot_traces(data.loc[~data.sector.eq("-O axis")], args.output / f"ground_truth_traces{suffix}", uncertainty)
        plot_traces(model, args.output / f"model_traces{suffix}", uncertainty)
        plot_average_traces(plotted_data, model, args.output / f"average_traces{suffix}", uncertainty)
        for condition in tuple(LABELS)[1:]:
            prefix = "" if condition == "expert" else f"_{condition}"
            plot_vectors(plotted_data, model, vectors_output / f"{prefix}_transition_vectors_{uncertainty}",
                         uncertainty=uncertainty, condition=condition)
    plot_cosine_sim(plotted_data, model, args.output / "cosine_similarity", args.cosine_errorbars)


if __name__ == "__main__":
    main()
