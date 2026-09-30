"""Plot native real-data sector means in the simplified mean-field figure layout.

Expert uses Pre -> Post and the four familiar or two novel images.  The act
reference uses Pre -> Task and its four recorded task images.  Sectors are
assigned once from each group's image-averaged transition, then the same cells
are used for every image, state, response type, and plotted trace in that group.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis.data_analysis import transitions_helpers as th


SECTORS = ("+NO axis", "+O axis", "-NO axis")
COLORS = {"Full": "black", "Occl": "red"}
DEFAULT_DATA_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "real_mean_field_style_expert_act"


@dataclass(frozen=True)
class Source:
    key: str
    transition_file: str
    trace_file: str
    group: str
    target: str
    image_label: str


SOURCES = (
    Source("expert_familiar", "transitions_post.csv", "transitions_post_traces.csv", "familiar", "Post", "Familiar"),
    Source("expert_novel", "transitions_post.csv", "transitions_post_traces.csv", "novel", "Post", "Novel"),
    Source("act_familiar", "transitions_act.csv", "transitions_act_traces.csv", "all", "Task", "Act familiar"),
)


def _source_frames(source: Source, data_dir: Path, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transitions = th.load_transition_table(data_dir / source.transition_file)
    transitions = transitions.loc[transitions["image_group"].eq(source.group)].copy()
    summary = th.build_mean_summary(
        transitions,
        image_group=source.group,
        pre_stage="Pre",
        target_stage=source.target,
        threshold=threshold,
    )
    summary.insert(0, "source", source.key)
    summary = summary.loc[summary["RotatedSector"].isin(SECTORS)].copy()
    if summary.empty:
        raise ValueError(f"No cells in plotted sectors for {source.key}")
    traces = pd.read_csv(data_dir / source.trace_file)
    traces = traces.loc[traces["image_group"].eq(source.group)].copy()
    membership = summary[["neuron_idx", "RotatedSector"]].rename(columns={"RotatedSector": "sector"})
    traces = traces.merge(membership, on="neuron_idx", how="inner", validate="many_to_one")
    traces["source"] = source.key
    if source.target == "Task":
        # Pre ends at 3 s; show the same first stimulus epoch for both stages.
        # The transition response itself was measured in 0.2 < t < 1 s.
        traces = traces.loc[traces["time"].between(-1.0, 3.05)].copy()
    images = sorted(int(value) for value in transitions["image_idx_original"].unique())
    observed = sorted(int(value) for value in traces["image_idx_original"].unique())
    if observed != images:
        raise ValueError(f"Trace image identities do not match response data for {source.key}: {observed} != {images}")
    return transitions, summary, traces, membership.assign(source=source.key)


def _mean_traces(traces: pd.DataFrame, *, pooled: bool) -> pd.DataFrame:
    index = ["source", "sector", "stage", "image_type", "time"]
    if not pooled:
        index.insert(2, "image_idx_original")
        per_cell = traces.groupby(index + ["neuron_idx"], observed=True, as_index=False)["response"].mean()
    else:
        # Each neuron contributes one image-averaged trace; all images have equal weight.
        per_cell = traces.groupby(index + ["neuron_idx"], observed=True, as_index=False)["response"].mean()
    output = per_cell.groupby(index, observed=True, as_index=False).agg(
        mean_response=("response", "mean"),
        sd_response=("response", "std"),
        n_cells=("neuron_idx", "nunique"),
    )
    output["sem_response"] = output["sd_response"].fillna(0) / np.sqrt(output["n_cells"])
    return output


def _draw_trace(ax: plt.Axes, data: pd.DataFrame, ylim: tuple[float, float]) -> None:
    ax.axvspan(0, 1, color="0.92", linewidth=0, zorder=0)
    ax.axhline(0, color="0.78", linewidth=0.6, zorder=1)
    for image_type, color in COLORS.items():
        line = data.loc[data["image_type"].eq(image_type)].sort_values("time")
        if not line.empty:
            ax.plot(line["time"], line["mean_response"], color=color, lw=1.25, alpha=0.95)
    ax.set_xlim(-1.0, 3.05)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="both", labelsize=6, length=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _row_limits(data: pd.DataFrame) -> tuple[float, float]:
    values = data["mean_response"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return (-1, 1)
    lo, hi = min(0.0, float(values.min())), max(0.0, float(values.max()))
    pad = max((hi - lo) * 0.08, 0.08)
    return lo - pad, hi + pad


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_pooled(expert_familiar: pd.DataFrame, expert_novel: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(9.4, 5.1), sharex=True, squeeze=False)
    columns = ((expert_familiar, "Pre", "Familiar\nNaive"), (expert_familiar, "Post", "Expert"),
               (expert_novel, "Pre", "Novel\nNaive"), (expert_novel, "Post", "Expert"))
    for row, sector in enumerate(SECTORS):
        row_data = pd.concat([expert_familiar, expert_novel], ignore_index=True)
        ylim = _row_limits(row_data.loc[row_data["sector"].eq(sector)])
        for col, (data, stage, title) in enumerate(columns):
            subset = data.loc[data["sector"].eq(sector) & data["stage"].eq(stage)]
            _draw_trace(axes[row, col], subset, ylim)
            if row == 0:
                axes[row, col].set_title(title, fontsize=9)
            if col == 0:
                axes[row, col].set_ylabel(sector, color=th.ROTATED_SECTOR_PALETTE[sector], fontsize=10)
            else:
                axes[row, col].tick_params(labelleft=False)
            if row != len(SECTORS)-1:
                axes[row, col].tick_params(labelbottom=False)
    fig.text(0.015, 0.5, "Baseline-subtracted response", va="center", rotation=90, fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.88, bottom=0.12, wspace=0.36, hspace=0.55)
    _save(fig, output)


def _plot_one_pooled(data: pd.DataFrame, target: str, output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(5.0, 5.1), sharex=True, squeeze=False)
    for row, sector in enumerate(SECTORS):
        ylim = _row_limits(data.loc[data["sector"].eq(sector)])
        for col, stage in enumerate(("Pre", target)):
            subset = data.loc[data["sector"].eq(sector) & data["stage"].eq(stage)]
            _draw_trace(axes[row, col], subset, ylim)
            if row == 0:
                axes[row, col].set_title("Naive" if col == 0 else ("Task / act" if target == "Task" else "Expert"), fontsize=9)
            if col == 0:
                axes[row, col].set_ylabel(sector, color=th.ROTATED_SECTOR_PALETTE[sector], fontsize=10)
            else:
                axes[row, col].tick_params(labelleft=False)
            if row != len(SECTORS)-1:
                axes[row, col].tick_params(labelbottom=False)
    fig.text(0.01, 0.5, "Baseline-subtracted response", va="center", rotation=90, fontsize=9)
    fig.subplots_adjust(left=0.18, right=0.99, top=0.89, bottom=0.12, wspace=0.34, hspace=0.55)
    _save(fig, output)


def _plot_images(data: pd.DataFrame, source: Source, output: Path) -> None:
    images = sorted(int(value) for value in data["image_idx_original"].unique())
    ncols = 2 * len(images)
    fig, axes = plt.subplots(3, ncols, figsize=(max(8, 1.25 * ncols), 5.1), sharex=True, squeeze=False)
    for row, sector in enumerate(SECTORS):
        ylim = _row_limits(data.loc[data["sector"].eq(sector)])
        for state_idx, stage in enumerate(("Pre", source.target)):
            for image_idx, image in enumerate(images):
                col = state_idx * len(images) + image_idx
                subset = data.loc[data["sector"].eq(sector) & data["stage"].eq(stage)
                                  & data["image_idx_original"].eq(image)]
                _draw_trace(axes[row, col], subset, ylim)
                if row == 0:
                    title = f"{'Naive' if state_idx == 0 else ('Task / act' if source.target == 'Task' else 'Expert')}\nImage {image}"
                    axes[row, col].set_title(title, fontsize=8)
                if col == 0:
                    axes[row, col].set_ylabel(sector, color=th.ROTATED_SECTOR_PALETTE[sector], fontsize=10)
                else:
                    axes[row, col].tick_params(labelleft=False)
                if row != len(SECTORS)-1:
                    axes[row, col].tick_params(labelbottom=False)
    fig.text(0.015, 0.5, "Baseline-subtracted response", va="center", rotation=90, fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.85, bottom=0.12, wspace=0.30, hspace=0.55)
    _save(fig, output)


def _vector_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.groupby(["source", "RotatedSector"], observed=True, as_index=False).agg(
        dNO=("dNO", "mean"), dO=("dO", "mean"), n_cells=("neuron_idx", "nunique"),
    ).rename(columns={"RotatedSector": "sector"})


def _draw_vectors(ax: plt.Axes, vectors: pd.DataFrame, title: str, extent: float) -> None:
    ax.axhline(0, color="0.8", lw=0.7)
    ax.axvline(0, color="0.8", lw=0.7)
    ax.plot([-extent, extent], [-extent, extent], "--", color="0.88", lw=0.7)
    ax.plot([-extent, extent], [extent, -extent], "--", color="0.88", lw=0.7)
    for sector in SECTORS:
        row = vectors.loc[vectors["sector"].eq(sector)]
        if row.empty:
            continue
        item = row.iloc[0]
        dx, dy = float(item["dNO"]), float(item["dO"])
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops={"arrowstyle": "-|>", "color": th.ROTATED_SECTOR_PALETTE[sector], "lw": 2.2, "mutation_scale": 14})
        ax.text(dx, dy, f" {sector.replace(' axis', '')} (n={int(item['n_cells'])})", fontsize=8, va="center")
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), xlabel=r"$\Delta R_{NO}$", ylabel=r"$\Delta R_O$", title=title)
    ax.set_aspect("equal", adjustable="box")


def _plot_expert_vectors(familiar: pd.DataFrame, novel: pd.DataFrame, output: Path) -> None:
    extent = max(0.5, 1.4 * float(np.nanmax(np.abs(pd.concat([familiar[["dNO", "dO"]], novel[["dNO", "dO"]]]).to_numpy()))))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), constrained_layout=True)
    _draw_vectors(axes[0], familiar, "Familiar: Pre → Post", extent)
    _draw_vectors(axes[1], novel, "Novel: Pre → Post", extent)
    _save(fig, output)


def _plot_act_vectors(vectors: pd.DataFrame, output: Path) -> None:
    extent = max(0.5, 1.4 * float(np.nanmax(np.abs(vectors[["dNO", "dO"]].to_numpy()))))
    fig, ax = plt.subplots(figsize=(4.5, 4.2), constrained_layout=True)
    _draw_vectors(ax, vectors, "Familiar task: Pre → Task", extent)
    _save(fig, output)


def export(data_dir: Path, output_dir: Path, threshold: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[pd.DataFrame] = []
    member_rows: list[pd.DataFrame] = []
    pooled: dict[str, pd.DataFrame] = {}
    vectors: dict[str, pd.DataFrame] = {}
    for source in SOURCES:
        _, summary, traces, membership = _source_frames(source, data_dir, threshold)
        source_dir = output_dir / source.key
        source_dir.mkdir(parents=True, exist_ok=True)
        by_image = _mean_traces(traces, pooled=False)
        pooled[source.key] = _mean_traces(traces, pooled=True)
        by_image.to_csv(source_dir / "traces_by_image.csv", index=False)
        pooled[source.key].to_csv(source_dir / "traces_pooled_images.csv", index=False)
        _plot_images(by_image, source, source_dir / "traces_images_side_by_side")
        summary_rows.append(summary)
        member_rows.append(membership)
        vectors[source.key] = _vector_rows(summary)
    _plot_pooled(pooled["expert_familiar"], pooled["expert_novel"], output_dir / "expert_familiar_novel_pooled_traces")
    _plot_one_pooled(pooled["act_familiar"], "Task", output_dir / "act_familiar_pooled_traces")
    _plot_expert_vectors(vectors["expert_familiar"], vectors["expert_novel"], output_dir / "expert_familiar_novel_transition_vectors")
    _plot_act_vectors(vectors["act_familiar"], output_dir / "act_familiar_transition_vectors")
    pd.concat(summary_rows, ignore_index=True).to_csv(output_dir / "source_cell_transitions.csv", index=False)
    pd.concat(member_rows, ignore_index=True).to_csv(output_dir / "sector_membership.csv", index=False)
    pd.concat(vectors.values(), ignore_index=True).to_csv(output_dir / "sector_mean_transition_vectors.csv", index=False)
    (output_dir / "README.md").write_text(
        "# Real-data plots matched to the mean-field layout\n\n"
        "Expert uses the native Pre to Post transition. Familiar images are 1, 2, 4, 5; "
        "novel images are 3, 6. The act reference uses the four images in the Pre to Task "
        "dataset. Its image numbering is native to that dataset and has not been assumed "
        "to match the Pre to Post image identities.\n\n"
        f"Sectors are assigned from each neuron's image-averaged transition with a {threshold:g} "
        "minimum displacement, independently for expert familiar, expert novel, and act. "
        "Only +NO, +O and -NO rows are plotted. These fixed memberships are then reused "
        "for every trace, image, and state within a dataset. Pooled traces first average "
        "the images for each neuron, then average the neurons. The plotted traces are "
        "baseline-subtracted source responses, with black for full (NO) and red for "
        "occluded (O); the stimulus occupies 0 to 1 s and y limits are shared within "
        "each row. Task traces are displayed from -1 to 3.05 s to match the Pre display.\n\n"
        "Transition vectors are means of the same member neurons' image-averaged dNO "
        "and dO from the response CSVs. Those responses use 0.2 < t < 1 s relative "
        "to each stage's own t < 0 baseline. The vector CSV records exact means and "
        "cell counts. Expert and act use different recorded populations and are not "
        "paired by neuron identity.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()
    export(args.data_dir, args.output_dir, args.threshold)


if __name__ == "__main__":
    main()
