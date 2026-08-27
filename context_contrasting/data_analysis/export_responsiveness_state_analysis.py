from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import context_contrasting.data_analysis.transitions_helpers as th


STATE_ORDER = (
    "non-responsive",
    "O responsive",
    "NO responsive",
    "NO & O responsive",
)
STATE_PALETTE = {
    "non-responsive": "#8c8c8c",
    "O responsive": "#d62728",
    "NO responsive": "#1f77b4",
    "NO & O responsive": "#6a3d9a",
}
CONDITION_ORDER = (
    "familiar_naive",
    "familiar_expert",
    "novel_naive",
    "novel_expert",
)
CONDITION_LABELS = {
    "familiar_naive": "Familiar naive",
    "familiar_expert": "Familiar expert",
    "novel_naive": "Novel naive",
    "novel_expert": "Novel expert",
}
HEATMAP_SPECS = (
    ("familiar_naive", "novel_naive", "A) familiar naive -> novel naive"),
    ("familiar_naive", "familiar_expert", "B) familiar naive -> familiar expert"),
    ("familiar_naive", "novel_expert", "C) familiar naive -> novel expert"),
    ("novel_naive", "novel_expert", "D) novel naive -> novel expert"),
    ("familiar_expert", "novel_expert", "E) familiar expert -> novel expert"),
)


@dataclass(frozen=True)
class StateCondition:
    key: str
    image_group: str
    stage: str


CONDITIONS = (
    StateCondition("familiar_naive", "familiar", "Pre"),
    StateCondition("familiar_expert", "familiar", "Post"),
    StateCondition("novel_naive", "novel", "Pre"),
    StateCondition("novel_expert", "novel", "Post"),
)


def _classify_state(no_response: pd.Series, o_response: pd.Series, *, threshold: float) -> pd.Series:
    no_active = no_response.to_numpy(dtype=float) > threshold
    o_active = o_response.to_numpy(dtype=float) > threshold
    labels = np.full(len(no_response), "non-responsive", dtype=object)
    labels[o_active & ~no_active] = "O responsive"
    labels[no_active & ~o_active] = "NO responsive"
    labels[no_active & o_active] = "NO & O responsive"
    return pd.Series(
        pd.Categorical(labels, categories=STATE_ORDER, ordered=True),
        index=no_response.index,
        name="state",
    )


def _build_condition_states(
    transition_table: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    summaries: list[pd.DataFrame] = []
    for condition in CONDITIONS:
        frame = transition_table.loc[
            transition_table["image_group"].eq(condition.image_group)
            & transition_table["stage"].astype(str).eq(condition.stage)
        ].copy()
        if frame.empty:
            raise ValueError(f"No rows found for {condition.image_group=} {condition.stage=}.")

        means = (
            frame.groupby("neuron_idx", as_index=False)[["NO", "O"]]
            .mean()
            .rename(columns={"NO": "NO_response", "O": "O_response"})
        )
        means.insert(1, "condition", condition.key)
        means.insert(2, "image_group", condition.image_group)
        means.insert(3, "stage", condition.stage)
        means["state"] = _classify_state(
            means["NO_response"],
            means["O_response"],
            threshold=threshold,
        )
        summaries.append(means)

    states = pd.concat(summaries, ignore_index=True)
    states["condition"] = pd.Categorical(states["condition"], categories=CONDITION_ORDER, ordered=True)
    return states


def _wide_state_table(condition_states: pd.DataFrame) -> pd.DataFrame:
    wide = condition_states.pivot(index="neuron_idx", columns="condition", values="state")
    missing = [condition for condition in CONDITION_ORDER if condition not in wide.columns]
    if missing:
        raise ValueError(f"Missing state columns: {missing}")
    wide = wide.loc[:, list(CONDITION_ORDER)].reset_index()
    for condition in CONDITION_ORDER:
        wide[condition] = pd.Categorical(wide[condition], categories=STATE_ORDER, ordered=True)
    return wide


def _state_distribution(condition_states: pd.DataFrame) -> pd.DataFrame:
    counts = (
        condition_states.groupby(["condition", "state"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby("condition", observed=False)["count"].transform("sum")
    counts["condition_total"] = totals.astype(int)
    counts["percent"] = np.divide(
        counts["count"].to_numpy(dtype=float) * 100.0,
        np.maximum(totals.to_numpy(dtype=float), 1.0),
    )
    return counts


def _plot_state_distribution_bars(
    distribution: pd.DataFrame,
    *,
    output_dir: Path,
    basename: str,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.8), sharey=True)
    stage_pairs = {
        "Familiar images": ("familiar_naive", "familiar_expert"),
        "Novel images": ("novel_naive", "novel_expert"),
    }
    for ax, (title, conditions) in zip(axes, stage_pairs.items(), strict=True):
        bottoms = np.zeros(len(conditions), dtype=float)
        x = np.arange(len(conditions))
        for state in STATE_ORDER:
            rows = (
                distribution.loc[
                    distribution["condition"].isin(conditions)
                    & distribution["state"].eq(state)
                ]
                .set_index("condition")
                .reindex(conditions)
            )
            values = rows["percent"].fillna(0.0).to_numpy(dtype=float)
            ax.bar(
                x,
                values,
                bottom=bottoms,
                color=STATE_PALETTE[state],
                edgecolor="white",
                linewidth=0.8,
                label=state,
            )
            for idx, value in enumerate(values):
                if value < 6.0:
                    continue
                ax.text(
                    idx,
                    bottoms[idx] + 0.5 * value,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if state in {"NO responsive", "NO & O responsive"} else "black",
                )
            bottoms += values

        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[condition].split()[-1].title() for condition in conditions])
        ax.set_ylim(0.0, 100.0)
        ax.set_ylabel("Percent of neurons")
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Responsiveness-state percentages", fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        path = output_dir / f"{basename}.{fmt}"
        fig.savefig(path, dpi=dpi)
        saved.append(path)
    plt.close(fig)
    return saved


def _transition_matrix(wide_states: pd.DataFrame, *, source: str, target: str) -> pd.DataFrame:
    paired = wide_states[["neuron_idx", source, target]].copy()
    paired = paired.rename(columns={source: "source_state", target: "target_state"})
    paired["source_state"] = pd.Categorical(
        paired["source_state"],
        categories=STATE_ORDER,
        ordered=True,
    )
    paired["target_state"] = pd.Categorical(
        paired["target_state"],
        categories=STATE_ORDER,
        ordered=True,
    )
    counts = (
        paired.groupby(["target_state", "source_state"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    source_totals = counts.groupby("source_state", observed=False)["count"].transform("sum")
    counts["source_total"] = source_totals.astype(int)
    counts["column_percent"] = np.divide(
        counts["count"].to_numpy(dtype=float) * 100.0,
        np.maximum(source_totals.to_numpy(dtype=float), 1.0),
    )
    counts.insert(0, "source_condition", source)
    counts.insert(1, "target_condition", target)
    return counts


def _matrix_arrays(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    percent_matrix = (
        matrix.pivot(index="target_state", columns="source_state", values="column_percent")
        .reindex(index=STATE_ORDER, columns=STATE_ORDER)
        .fillna(0.0)
    )
    count_matrix = (
        matrix.pivot(index="target_state", columns="source_state", values="count")
        .reindex(index=STATE_ORDER, columns=STATE_ORDER)
        .fillna(0)
        .astype(int)
    )
    return percent_matrix, count_matrix, count_matrix.sum(axis=0)


def _draw_transition_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    title: str,
    vmax: float,
) -> None:
    percent_matrix, count_matrix, column_totals = _matrix_arrays(matrix)
    image = ax.imshow(percent_matrix.to_numpy(dtype=float), cmap="viridis", vmin=0.0, vmax=vmax)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_xticks(np.arange(len(STATE_ORDER)))
    ax.set_yticks(np.arange(len(STATE_ORDER)))
    ax.set_xticklabels(
        [f"{state}\nn={int(column_totals.loc[state])}" for state in STATE_ORDER],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax.set_yticklabels(STATE_ORDER, fontsize=8)
    ax.set_xlabel("Source state", fontsize=9)
    ax.set_ylabel("Target state", fontsize=9)

    for row_idx, target_state in enumerate(STATE_ORDER):
        for col_idx, source_state in enumerate(STATE_ORDER):
            pct = float(percent_matrix.loc[target_state, source_state])
            count = int(count_matrix.loc[target_state, source_state])
            text_color = "white" if pct <= 0.22 * vmax or pct >= 0.78 * vmax else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{pct:.1f}%\n{count}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
            )
    return image


def _plot_transition_heatmaps(
    matrices: dict[tuple[str, str], pd.DataFrame],
    *,
    output_dir: Path,
    basename: str,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    max_observed_percent = max(float(matrix["column_percent"].max()) for matrix in matrices.values())
    if not np.isfinite(max_observed_percent) or max_observed_percent <= 0.0:
        max_observed_percent = 1.0

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 9.2), squeeze=False)
    fig.subplots_adjust(left=0.12, right=0.86, bottom=0.13, top=0.87, wspace=0.62, hspace=0.78)
    axes_flat = axes.ravel()
    last_image = None
    for panel_idx, (ax, (source, target, title)) in enumerate(
        zip(axes_flat, HEATMAP_SPECS, strict=False)
    ):
        matrix = matrices[(source, target)]
        last_image = _draw_transition_heatmap(
            ax,
            matrix,
            title=title,
            vmax=max_observed_percent,
        )
        if panel_idx % 3 != 0:
            ax.set_ylabel("")
    axes_flat[-1].axis("off")
    if last_image is not None:
        cbar_ax = fig.add_axes((0.9, 0.22, 0.018, 0.58))
        cbar = fig.colorbar(last_image, cax=cbar_ax)
        cbar.set_label("Column-normalized percentage")
    fig.suptitle("Responsiveness-state transition matrices", fontweight="bold", y=0.98)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        path = output_dir / f"{basename}.{fmt}"
        fig.savefig(path, dpi=dpi)
        saved.append(path)
    plt.close(fig)
    return saved


def export_responsiveness_state_analysis(
    *,
    data_dir: Path,
    output_dir: Path,
    threshold: float,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    transition_table = th.load_transition_table(data_dir / "transitions_post.csv")
    condition_states = _build_condition_states(transition_table, threshold=threshold)
    wide_states = _wide_state_table(condition_states)
    distribution = _state_distribution(condition_states)
    matrices = {
        (source, target): _transition_matrix(wide_states, source=source, target=target)
        for source, target, _title in HEATMAP_SPECS
    }
    all_matrices = pd.concat(matrices.values(), ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    state_csv = output_dir / "responsiveness_states_by_condition.csv"
    wide_csv = output_dir / "responsiveness_states_wide.csv"
    distribution_csv = output_dir / "responsiveness_state_percentages.csv"
    transition_csv = output_dir / "responsiveness_state_transition_matrices.csv"

    condition_states.to_csv(state_csv, index=False)
    wide_states.to_csv(wide_csv, index=False)
    distribution.to_csv(distribution_csv, index=False)
    all_matrices.to_csv(transition_csv, index=False)

    saved = [state_csv, wide_csv, distribution_csv, transition_csv]
    saved.extend(
        _plot_state_distribution_bars(
            distribution,
            output_dir=output_dir,
            basename="responsiveness_state_percentages",
            formats=formats,
            dpi=dpi,
        )
    )
    saved.extend(
        _plot_transition_heatmaps(
            matrices,
            output_dir=output_dir,
            basename="responsiveness_state_transition_heatmaps",
            formats=formats,
            dpi=dpi,
        )
    )
    return saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify neurons by NO/O responsiveness and export state percentages/transitions."
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "responsiveness_state_analysis",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--formats", nargs="+", default=["png", "svg", "eps"], choices=("png", "svg", "eps"))
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    saved = export_responsiveness_state_analysis(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        formats=tuple(args.formats),
        dpi=args.dpi,
    )
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
