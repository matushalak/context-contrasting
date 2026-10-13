"""Plot traces and transition vectors using only ground_truth.csv and model.csv."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
SECTORS = ("-NO axis", "+O axis", "+NO axis")
TRACE_ROWS = ("Overall", *SECTORS)
DATA_SECTORS = (*SECTORS, "-O axis")
COLORS = {"+NO axis": "#3159a7", "+O axis": "#ef202f",
          "-NO axis": "#e68600", "-O axis": "#16843e"}
SOURCES = ("familiar", "novel")
LABELS = {"naive": "Naive", "expert": "Expert", "expert_no_fb": "FB silencing",
          "expert_no_lat": "PV silencing", "expert_no_fb_no_lat": "FB & PV silencing"}
MODEL_VERSIONS = {"core": ("naive", "expert"),
                  "ablations": tuple(LABELS)[2:], "all": tuple(LABELS)}


def use_myriad_pro():
    adobe_fonts = Path("/Library/Application Support/Adobe/Acrobat/DC/WebResources/Resource0/OWP/default/fonts")
    for name in ("MyriadPro-Regular.otf", "MyriadPro-Bold.otf", "MyriadPro-Light.otf"):
        path = adobe_fonts / name
        if path.exists():
            font_manager.fontManager.addfont(path)
    font_manager.findfont("Myriad Pro", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Myriad Pro", "font.size": 12,
                         "axes.titlesize": 18, "axes.labelsize": 18,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "legend.fontsize": 12, "legend.title_fontsize": 12,
                         "svg.fonttype": "none"})


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


def trace_summary(table, overall_table=None, weights=None):
    observations = trace_observations(table.loc[table.sector.isin(SECTORS)])
    keys = ["source", "sector", "condition_key", "response_type", "time_seconds"]
    sectors = summarize_traces(observations, keys)
    overall = average_traces(table if overall_table is None else overall_table, weights)
    overall["sector"] = "Overall"
    return pd.concat([overall, sectors], ignore_index=True)


def trace_limits(summary):
    limits = {}
    for sector in TRACE_ROWS:
        frame = summary.loc[summary.sector.eq(sector)]
        lower = (frame.response - frame.sd.fillna(0)).min()
        upper = (frame.response + frame.sd.fillna(0)).max()
        margin = max((upper - lower) * 0.08, 0.1)
        limits[sector] = (lower - margin, max(upper + margin, lower - margin + 1.5))
    return limits


def plot_traces(summary, source, path, uncertainty="sd", conditions=None, limits=None):
    conditions = tuple(LABELS) if conditions is None else conditions
    limits = trace_limits(summary) if limits is None else limits
    trace_width = 1.2
    fig, axes = plt.subplots(len(TRACE_ROWS), len(conditions),
                             figsize=(2.2 * len(conditions), 6.3),
                             sharex=True, sharey="row", squeeze=False, layout="constrained")
    for row, sector in enumerate(TRACE_ROWS):
        for column, condition in enumerate(conditions):
            ax = axes[row, column]
            ax.axvspan(0, 1, color="0.92")
            frame = summary.loc[summary.sector.eq(sector) & summary.source.eq(source) &
                                summary.condition_key.eq(condition)]
            for response, color in (("NO", "black"), ("O", "red")):
                line = frame.loc[frame.response_type.eq(response)]
                ax.plot(line.time_seconds, line.response, color=color, lw=trace_width, label=response)
                ax.fill_between(line.time_seconds, line.response - line[uncertainty],
                                line.response + line[uncertainty], color=color, alpha=0.18, linewidth=0)
            ax.set(xlim=(-1, 3), ylim=limits[sector])
            if row == 0:
                ax.set_title(LABELS[condition])
            ax.set_axis_off()
            if column == 0:
                ax.text(-0.05, 0.5, sector.replace(" axis", ""), transform=ax.transAxes,
                        ha="right", va="center", color=COLORS.get(sector, "black"))
                lower, upper = limits[sector]
                x, y = -0.8, lower + 0.02 * (upper - lower)
                ax.plot([x, x + 1], [y, y], color="black", lw=trace_width, clip_on=False)
                ax.plot([x, x], [y, y + 1], color="black", lw=trace_width, clip_on=False)
                ax.text(x + 0.5, y, "1 s", ha="center", va="top", fontsize=8)
                ax.text(x + 0.06, y + 1, "1 S.D.", ha="left", va="bottom", fontsize=8)
    axes[0, -1].legend(frameon=False)
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


def plot_vectors(data, model, source, path, uncertainty="sd", condition="expert"):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), #layout="constrained",
                             sharex=True, sharey=True, squeeze=False)
    weights = empirical_sector_weights(data)
    data_vectors = data.loc[data.record_type.eq("transition")]
    model_vectors = model_transition_vectors(model, condition)
    for column, (vectors, label) in enumerate(((data_vectors, "Data"),
                                                (model_vectors, f"Model ({LABELS[condition]})"))):
        ax = axes[0, column]
        source_vectors = vectors.loc[vectors.source.eq(source)]
        for sector in DATA_SECTORS if label == "Data" else SECTORS:
            color = COLORS[sector]
            points = source_vectors.loc[source_vectors.sector.eq(sector),
                                         ["delta_NO", "delta_O"]].to_numpy()
            if not len(points):
                continue
            center = points.mean(axis=0)
            # Neuron-to-neuron (data) or seed-to-seed (model) covariance.
            if len(points) > 1:
                covariance = np.cov(points, rowvar=False)
                if uncertainty == "sem":
                    covariance /= len(points)
                values, directions = np.linalg.eigh(covariance)
                angle = np.degrees(np.arctan2(directions[1, -1], directions[0, -1]))
                width, height = 2 * np.sqrt(np.maximum(values[::-1], 0))
                ax.add_patch(Ellipse(center, width, height, angle=angle, color=color, alpha=0.18))
            ax.annotate("", xy=center, xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5, mutation_scale=14))
            ax.plot([], [], color=color, label=sector.replace(" axis", ""))
        overall = overall_transition(vectors, source, None if label == "Data" else weights)
        ax.annotate("", xy=overall, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=3.2, mutation_scale=16))
        ax.plot([], [], color="black", lw=3.2, label="Overall transition")
        ax.axhline(0, color="0.7", lw=0.7)
        ax.axvline(0, color="0.7", lw=0.7)
        ax.set(xlim=(-1.4, 1.4), ylim=(-.3, 1.0), #aspect=1.0,
               title=label, xlabel="Delta NO", ylabel="Delta O")
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


def plot_cosine_sim(data, model, source, path, errorbars=False):
    similarities = cosine_similarities(data, model)
    order = list(LABELS.values())[1:]
    hues = ["Overall"] + [sector.replace(" axis", "") for sector in SECTORS]
    palette = {sector.replace(" axis", ""): color for sector, color in COLORS.items()}
    palette["Overall"] = "black"
    fig, ax = plt.subplots(figsize=(4, 3.3), layout="constrained")
    for value in (-1, 0, 1):
        ax.axhline(value, color="0.7", lw=0.7, linestyle="--")
    sns.pointplot(data=similarities.loc[similarities.source.eq(source)], x="condition",
                  y="similarity", hue="sector", order=order, hue_order=hues,
                  palette=palette, dodge=0.45, errorbar="sd" if errorbars else None, ax=ax,
                  linestyle="none")
    ax.set(xlabel=None, ylabel="Cosine similarity", ylim=(-1.2, 1.2))
    ax.tick_params(axis="x", rotation=25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, title=None)
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
    use_myriad_pro()
    data, model = pd.read_csv(args.data), pd.read_csv(args.model)
    plotted_data = data if args.include_minus_o else data.loc[~data.sector.eq("-O axis")]
    args.output.mkdir(parents=True, exist_ok=True)
    weights = empirical_sector_weights(data)
    data_summary = trace_summary(data, plotted_data)
    model_summary = trace_summary(model, weights=weights)
    core_conditions = MODEL_VERSIONS["core"]
    core_limits = trace_limits(pd.concat([
        data_summary, model_summary.loc[model_summary.condition_key.isin(core_conditions)]
    ]))
    model_limits = {version: trace_limits(model_summary.loc[
        model_summary.condition_key.isin(conditions)]) for version, conditions in MODEL_VERSIONS.items()}
    model_limits["core"] = core_limits
    for uncertainty in ("sd", "sem") if args.uncertainty == "both" else (args.uncertainty,):
        suffix = "" if uncertainty == "sd" else "_sem"
        for source in SOURCES:
            output = args.output / source
            vectors_output = output / "transition_vectors"
            vectors_output.mkdir(parents=True, exist_ok=True)
            plot_traces(data_summary, source, output / f"ground_truth_traces{suffix}",
                        uncertainty, conditions=core_conditions, limits=core_limits)
            for version, conditions in MODEL_VERSIONS.items():
                plot_traces(model_summary, source, output / f"model_traces_{version}{suffix}",
                            uncertainty, conditions=conditions, limits=model_limits[version])
            for condition in tuple(LABELS)[1:]:
                prefix = "" if condition == "expert" else f"_{condition}"
                plot_vectors(plotted_data, model, source,
                             vectors_output / f"{prefix}_transition_vectors_{uncertainty}",
                             uncertainty=uncertainty, condition=condition)
    for source in SOURCES:
        plot_cosine_sim(plotted_data, model, source, args.output / source / "cosine_similarity",
                        args.cosine_errorbars)


if __name__ == "__main__":
    main()
