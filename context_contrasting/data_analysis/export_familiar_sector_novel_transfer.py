from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import context_contrasting.data_analysis.transitions_helpers as th


IMAGE_TYPE_SPECS = (
    ("Occl", "O", "red"),
    ("Full", "NO", "black"),
)
PRE_STAGE = "Pre"
POST_STAGE = "Post"


def _observed_shift_limits(summaries: list[pd.DataFrame]) -> list[float]:
    values = np.concatenate(
        [summary[["dNO", "dO"]].to_numpy(dtype=float).reshape(-1) for summary in summaries]
    )
    values = np.abs(values[np.isfinite(values)])
    if values.size == 0:
        return [-0.5, 0.5]
    extent = float(np.nanmax(values))
    if not np.isfinite(extent) or extent == 0.0:
        extent = 0.5
    else:
        extent *= 1.12
    return [-extent, extent]


def _sector_counts(summary: pd.DataFrame) -> dict[str, int]:
    counts = (
        summary["RotatedSector"]
        .value_counts(sort=False)
        .reindex(th.ROTATED_SECTOR_ORDER)
        .fillna(0)
        .astype(int)
    )
    return {sector: int(counts.loc[sector]) for sector in th.ROTATED_SECTOR_ORDER}


def _sector_rgba(
    summary: pd.DataFrame,
    *,
    sector: str,
    alpha_source: pd.Series,
    min_alpha: float = 0.55,
    max_alpha: float = 1.0,
) -> np.ndarray:
    sector_mask = summary["RotatedSector"].to_numpy() == sector
    alphas = th._map_norms_to_alphas(
        np.log(alpha_source.to_numpy(dtype=float) + th.LOG_NORM_EPS),
        min_alpha=min_alpha,
        max_alpha=max_alpha,
    )
    rgb = np.array(th.mcolors.to_rgb(th.ROTATED_SECTOR_PALETTE[sector])).reshape(1, 3)
    rgba = np.repeat(rgb, int(sector_mask.sum()), axis=0)
    return np.concatenate([rgba, alphas[sector_mask].reshape(-1, 1)], axis=1)


def _draw_shift_scatter(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    title: str,
    shift_limits: list[float],
    alpha_source: pd.Series,
) -> None:
    ticks = np.arange(np.ceil(shift_limits[0]), np.floor(shift_limits[1]) + 1.0, 1.0)
    th._draw_origin_guides(ax)
    th._draw_rotated_guides(ax, shift_limits)

    for sector in th._sector_plot_order(small_delta_first=True):
        sector_rows = summary.loc[summary["RotatedSector"] == sector]
        if sector_rows.empty:
            continue
        ax.scatter(
            sector_rows["dNO"],
            sector_rows["dO"],
            s=34,
            c=_sector_rgba(summary, sector=sector, alpha_source=alpha_source),
            edgecolors="none",
            zorder=th._sector_scatter_zorder(sector),
        )

    sector_means = th.sector_mean_table(summary)
    for sector in th.ROTATED_SECTOR_ORDER:
        if sector == "small \u2206":
            continue
        mean_row = sector_means.loc[sector_means["RotatedSector"] == sector]
        if mean_row.empty:
            continue
        mean_row = mean_row.iloc[0]
        th._draw_arrow(
            ax,
            (0.0, 0.0),
            (float(mean_row["dNO"]), float(mean_row["dO"])),
            color=th._darken_color(th.ROTATED_SECTOR_PALETTE[sector]),
            linewidth=2.3,
            mutation_scale=13.0,
            alpha=0.95,
            zorder=4,
        )

    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlim(shift_limits)
    ax.set_ylim(shift_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$dNO$")
    ax.set_ylabel("$dO$")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


def _attach_familiar_sector_labels(
    frame: pd.DataFrame,
    familiar_summary: pd.DataFrame,
) -> pd.DataFrame:
    sectors = familiar_summary[["neuron_idx", "RotatedSector", "dNorm"]].copy()
    sectors = sectors.rename(columns={"dNorm": "familiar_dNorm"})
    labeled = frame.merge(sectors, on="neuron_idx", how="left", validate="many_to_one")
    if labeled["RotatedSector"].isna().any():
        missing = (
            labeled.loc[labeled["RotatedSector"].isna(), "neuron_idx"]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(f"Rows missing familiar-sector labels for neuron_idx values: {missing}")
    labeled["RotatedSector"] = pd.Categorical(
        labeled["RotatedSector"],
        categories=th.ROTATED_SECTOR_ORDER,
        ordered=True,
    )
    return labeled


def _build_transfer_summaries(
    transition_table: pd.DataFrame,
    *,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    familiar = th.build_mean_summary(
        transition_table,
        image_group="familiar",
        pre_stage="Pre",
        target_stage=POST_STAGE,
        threshold=threshold,
    )
    novel_native = th.build_mean_summary(
        transition_table,
        image_group="novel",
        pre_stage="Pre",
        target_stage=POST_STAGE,
        threshold=threshold,
    )
    novel = novel_native.drop(columns=["RotatedSector"]).pipe(_attach_familiar_sector_labels, familiar)
    novel = novel.merge(
        novel_native[["neuron_idx", "RotatedSector"]].rename(
            columns={"RotatedSector": "NovelRotatedSector"}
        ),
        on="neuron_idx",
        how="left",
        validate="one_to_one",
    )
    merged = familiar.merge(
        novel,
        on="neuron_idx",
        suffixes=("_familiar", "_novel"),
        validate="one_to_one",
    )
    return familiar, novel, merged, novel_native


def _sector_transition_table(
    familiar: pd.DataFrame,
    novel_native: pd.DataFrame,
) -> pd.DataFrame:
    paired = familiar[["neuron_idx", "RotatedSector"]].rename(
        columns={"RotatedSector": "FamiliarSector"}
    )
    paired = paired.merge(
        novel_native[["neuron_idx", "RotatedSector"]].rename(
            columns={"RotatedSector": "NovelSector"}
        ),
        on="neuron_idx",
        how="inner",
        validate="one_to_one",
    )
    paired["FamiliarSector"] = pd.Categorical(
        paired["FamiliarSector"],
        categories=th.ROTATED_SECTOR_ORDER,
        ordered=True,
    )
    paired["NovelSector"] = pd.Categorical(
        paired["NovelSector"],
        categories=th.ROTATED_SECTOR_ORDER,
        ordered=True,
    )

    counts = (
        paired.groupby(["NovelSector", "FamiliarSector"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    column_totals = counts.groupby("FamiliarSector", observed=False)["count"].transform("sum")
    counts["familiar_sector_total"] = column_totals.astype(int)
    counts["column_percent"] = np.divide(
        counts["count"].to_numpy(dtype=float) * 100.0,
        np.maximum(column_totals.to_numpy(dtype=float), 1.0),
    )
    return counts


def _plot_sector_transition_heatmap(
    sector_transitions: pd.DataFrame,
    *,
    output_dir: Path,
    basename: str,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    percent_matrix = (
        sector_transitions.pivot(
            index="NovelSector",
            columns="FamiliarSector",
            values="column_percent",
        )
        .reindex(index=th.ROTATED_SECTOR_ORDER, columns=th.ROTATED_SECTOR_ORDER)
        .fillna(0.0)
    )
    count_matrix = (
        sector_transitions.pivot(
            index="NovelSector",
            columns="FamiliarSector",
            values="count",
        )
        .reindex(index=th.ROTATED_SECTOR_ORDER, columns=th.ROTATED_SECTOR_ORDER)
        .fillna(0)
        .astype(int)
    )
    column_totals = count_matrix.sum(axis=0)
    max_observed_percent = float(np.nanmax(percent_matrix.to_numpy(dtype=float)))
    if not np.isfinite(max_observed_percent) or max_observed_percent <= 0.0:
        max_observed_percent = 1.0

    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    image = ax.imshow(
        percent_matrix.to_numpy(dtype=float),
        cmap="viridis",
        vmin=0.0,
        vmax=max_observed_percent,
    )
    ax.set_xticks(np.arange(len(th.ROTATED_SECTOR_ORDER)))
    ax.set_yticks(np.arange(len(th.ROTATED_SECTOR_ORDER)))
    ax.set_xticklabels(
        [f"{sector}\nn={int(column_totals.loc[sector])}" for sector in th.ROTATED_SECTOR_ORDER],
        rotation=35,
        ha="right",
    )
    ax.set_yticklabels(th.ROTATED_SECTOR_ORDER)
    ax.set_xlabel("Familiar expert transition sector")
    ax.set_ylabel("Novel expert transition sector")
    ax.set_title("Novel sector reached from each familiar sector", pad=14, fontweight="bold")

    for row_idx, novel_sector in enumerate(th.ROTATED_SECTOR_ORDER):
        for col_idx, familiar_sector in enumerate(th.ROTATED_SECTOR_ORDER):
            pct = float(percent_matrix.loc[novel_sector, familiar_sector])
            count = int(count_matrix.loc[novel_sector, familiar_sector])
            text_color = "white" if pct <= 12.0 or pct >= 42.0 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{pct:.1f}%\n{count}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Column-normalized percentage")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        path = output_dir / f"{basename}.{fmt}"
        fig.savefig(path, dpi=dpi)
        saved.append(path)
    plt.close(fig)
    return saved


def _stage_label(stage: str) -> str:
    return "Naive" if stage == PRE_STAGE else "Expert"


def _summarize_traces_by_image(
    trace_table: pd.DataFrame,
    familiar_summary: pd.DataFrame,
) -> pd.DataFrame:
    labeled = _attach_familiar_sector_labels(trace_table, familiar_summary)
    index_cols = [
        "RotatedSector",
        "image_group",
        "image_idx_original",
        "image_idx_within_group",
        "stage",
        "image_type",
        "time",
    ]
    summary = (
        labeled.groupby(index_cols, observed=True, as_index=False)
        .agg(
            mean_response=("response", "mean"),
            sd_response=("response", "std"),
            n_responses=("response", "size"),
            n_cells=("neuron_idx", "nunique"),
        )
        .sort_values(index_cols)
    )
    summary["sem"] = summary["sd_response"].fillna(0.0) / np.sqrt(
        np.maximum(summary["n_responses"].to_numpy(dtype=float), 1.0)
    )
    return summary


def _summarize_traces_pooled_by_group(
    trace_table: pd.DataFrame,
    familiar_summary: pd.DataFrame,
) -> pd.DataFrame:
    labeled = _attach_familiar_sector_labels(trace_table, familiar_summary)
    index_cols = [
        "RotatedSector",
        "image_group",
        "stage",
        "image_type",
        "time",
    ]
    summary = (
        labeled.groupby(index_cols, observed=True, as_index=False)
        .agg(
            mean_response=("response", "mean"),
            sd_response=("response", "std"),
            n_responses=("response", "size"),
            n_cells=("neuron_idx", "nunique"),
            n_images=("image_idx_original", "nunique"),
        )
        .sort_values(index_cols)
    )
    summary["sem"] = summary["sd_response"].fillna(0.0) / np.sqrt(
        np.maximum(summary["n_responses"].to_numpy(dtype=float), 1.0)
    )
    return summary


def _trace_column_order(trace_summary: pd.DataFrame, *, by_image: bool) -> pd.DataFrame:
    stage_order = {PRE_STAGE: 0, POST_STAGE: 1}
    if by_image:
        columns = (
            trace_summary[
                ["image_group", "image_idx_within_group", "image_idx_original", "stage"]
            ]
            .drop_duplicates()
            .assign(
                group_order=lambda frame: frame["image_group"].map({"familiar": 0, "novel": 1}),
                stage_order=lambda frame: frame["stage"].map(stage_order),
                group_label=lambda frame: frame["image_group"].map(
                    {
                        "familiar": "Familiar images",
                        "novel": "Novel images",
                    }
                ),
                column_label=lambda frame: frame.apply(
                    lambda row: (
                        f"{'Fam' if row.image_group == 'familiar' else 'Novel'} "
                        f"image {int(row.image_idx_original)}\n{_stage_label(str(row.stage))}"
                    ),
                    axis=1,
                ),
            )
            .sort_values(
                ["group_order", "image_idx_within_group", "image_idx_original", "stage_order"]
            )
            .reset_index(drop=True)
        )
    else:
        columns = (
            trace_summary[["image_group", "stage"]]
            .drop_duplicates()
            .assign(
                image_idx_within_group=-1,
                image_idx_original=-1,
                group_order=lambda frame: frame["image_group"].map({"familiar": 0, "novel": 1}),
                stage_order=lambda frame: frame["stage"].map(stage_order),
                group_label=lambda frame: frame["image_group"].map(
                    {
                        "familiar": "Familiar images pooled",
                        "novel": "Novel images pooled",
                    }
                ),
                column_label=lambda frame: frame.apply(
                    lambda row: (
                        f"{'Familiar' if row.image_group == 'familiar' else 'Novel'}\n"
                        f"{_stage_label(str(row.stage))}"
                    ),
                    axis=1,
                ),
            )
            .sort_values(["group_order", "stage_order"])
            .reset_index(drop=True)
        )
    return columns


def _row_y_limits(trace_summary: pd.DataFrame, *, sector: str) -> tuple[float, float]:
    rows = trace_summary.loc[trace_summary["RotatedSector"].eq(sector)]
    if rows.empty:
        return (-1.0, 1.0)
    lo = float(np.nanmin(rows["mean_response"] - rows["sem"].fillna(0.0)))
    hi = float(np.nanmax(rows["mean_response"] + rows["sem"].fillna(0.0)))
    lo = min(lo, 0.0)
    hi = max(hi, 0.0, 1.0)
    center = 0.5 * (lo + hi)
    span = max(1.35, 1.18 * (hi - lo))
    return center - 0.5 * span, center + 0.5 * span


def _add_scale_bar(ax: plt.Axes, *, length: float = 1.0) -> None:
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    x_span = x_hi - x_lo
    cap = 0.018 * x_span
    x = x_lo + 0.08 * x_span
    ax.plot([x, x], [0.0, length], color="0.15", lw=1.0, solid_capstyle="butt", zorder=6)
    ax.plot([x - cap, x + cap], [0.0, 0.0], color="0.15", lw=1.0, solid_capstyle="butt", zorder=6)
    ax.plot([x - cap, x + cap], [length, length], color="0.15", lw=1.0, solid_capstyle="butt", zorder=6)
    ax.text(
        x - 1.8 * cap,
        0.5 * length,
        "1 z",
        ha="right",
        va="center",
        fontsize=6.5,
        color="0.15",
        rotation=90,
    )


def _draw_trace_grid(
    axes: np.ndarray,
    trace_summary: pd.DataFrame,
    familiar_counts: dict[str, int],
    columns: pd.DataFrame,
    *,
    by_image: bool,
) -> None:
    for col_idx, col in enumerate(columns.itertuples(index=False)):
        axes[0, col_idx].set_title(str(col.column_label), fontsize=8.5, pad=6)

    for row_idx, sector in enumerate(th.ROTATED_SECTOR_ORDER):
        y_limits = _row_y_limits(trace_summary, sector=sector)
        sector_color = th.ROTATED_SECTOR_PALETTE[sector]
        for col_idx, col in enumerate(columns.itertuples(index=False)):
            ax = axes[row_idx, col_idx]
            ax.axhline(0.0, color="0.84", lw=0.6, zorder=0)
            ax.axvspan(0.0, 1.0, color="0.92", zorder=-1)
            rows = trace_summary.loc[
                trace_summary["RotatedSector"].eq(sector)
                & trace_summary["image_group"].eq(col.image_group)
                & trace_summary["stage"].eq(col.stage)
            ]
            if by_image:
                rows = rows.loc[rows["image_idx_original"].eq(col.image_idx_original)]
            for image_type, _response_type, color in IMAGE_TYPE_SPECS:
                trace_df = rows.loc[rows["image_type"].eq(image_type)].sort_values("time")
                if trace_df.empty:
                    continue
                x = trace_df["time"].to_numpy(dtype=float)
                y = trace_df["mean_response"].to_numpy(dtype=float)
                sem = trace_df["sem"].fillna(0.0).to_numpy(dtype=float)
                ax.plot(x, y, color=color, lw=1.1)
                ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.16, linewidth=0)
            ax.set_ylim(y_limits)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col_idx == 0:
                _add_scale_bar(ax)

        axes[row_idx, 0].text(
            -0.11,
            0.5,
            f"{sector}\nn={familiar_counts[sector]}",
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8.5,
            color=sector_color,
            fontweight="bold",
        )


def _add_group_headers(fig: plt.Figure, axes: np.ndarray, columns: pd.DataFrame) -> None:
    y = axes[0, 0].get_position().y1 + 0.023
    for group_label, group_columns in columns.groupby("group_label", sort=False):
        left_idx = int(group_columns.index.min())
        right_idx = int(group_columns.index.max())
        left = axes[0, left_idx].get_position().x0
        right = axes[0, right_idx].get_position().x1
        fig.text(
            0.5 * (left + right),
            y,
            str(group_label),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


def _draw_legend(ax: plt.Axes, familiar_counts: dict[str, int]) -> None:
    ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=th.ROTATED_SECTOR_PALETTE[sector],
            markeredgecolor="none",
            label=f"{sector} (n={familiar_counts[sector]})",
        )
        for sector in th.ROTATED_SECTOR_ORDER
    ]
    trace_handles = [
        Line2D([0], [0], color="black", lw=1.5, label="NO trace"),
        Line2D([0], [0], color="red", lw=1.5, label="O trace"),
    ]
    ax.legend(
        handles=handles + trace_handles,
        frameon=False,
        loc="center left",
        fontsize=9,
        handlelength=1.8,
        borderaxespad=0.0,
    )
    ax.set_title("Familiar sector labels", fontsize=11, loc="left", pad=4)


def _save_combined_figure(
    *,
    familiar: pd.DataFrame,
    novel: pd.DataFrame,
    trace_summary: pd.DataFrame,
    by_image: bool,
    output_dir: Path,
    basename: str,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    familiar_counts = _sector_counts(familiar)
    shift_limits = _observed_shift_limits([familiar, novel])
    trace_columns = _trace_column_order(trace_summary, by_image=by_image)
    figure_width = max(12.0, 1.55 * len(trace_columns) + 3.2)

    fig = plt.figure(figsize=(figure_width, 14.0))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 2.7],
        left=0.075,
        right=0.99,
        bottom=0.04,
        top=0.95,
        hspace=0.28,
    )
    top = outer[0].subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.64], wspace=0.25)
    ax_familiar = fig.add_subplot(top[0, 0])
    ax_novel = fig.add_subplot(top[0, 1])
    ax_legend = fig.add_subplot(top[0, 2])

    _draw_shift_scatter(
        ax_familiar,
        familiar,
        title="Familiar expert shift, colored by familiar sector",
        shift_limits=shift_limits,
        alpha_source=familiar["dNorm"],
    )
    _draw_shift_scatter(
        ax_novel,
        novel,
        title="Novel expert shift, colored by familiar sector",
        shift_limits=shift_limits,
        alpha_source=novel["familiar_dNorm"],
    )
    _draw_legend(ax_legend, familiar_counts)

    bottom = outer[1].subgridspec(
        len(th.ROTATED_SECTOR_ORDER),
        len(trace_columns),
        wspace=0.16,
        hspace=0.2,
    )
    trace_axes = np.empty((len(th.ROTATED_SECTOR_ORDER), len(trace_columns)), dtype=object)
    for row_idx in range(len(th.ROTATED_SECTOR_ORDER)):
        for col_idx in range(len(trace_columns)):
            trace_axes[row_idx, col_idx] = fig.add_subplot(bottom[row_idx, col_idx])

    _draw_trace_grid(
        trace_axes,
        trace_summary,
        familiar_counts,
        trace_columns,
        by_image=by_image,
    )
    _add_group_headers(fig, trace_axes, trace_columns)

    fig.suptitle(
        "Familiar-expert rotated sectors transferred to novel-expert responses",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )

    saved: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{basename}.{fmt}"
        fig.savefig(path, dpi=dpi)
        saved.append(path)
    plt.close(fig)
    return saved


def export_familiar_sector_novel_transfer(
    *,
    data_dir: Path,
    output_dir: Path,
    threshold: float,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    transition_table = th.load_transition_table(data_dir / "transitions_post.csv")
    trace_table = pd.read_csv(data_dir / "transitions_post_traces.csv")
    familiar, novel, merged, novel_native = _build_transfer_summaries(
        transition_table,
        threshold=threshold,
    )
    trace_by_image = _summarize_traces_by_image(trace_table, familiar)
    trace_pooled = _summarize_traces_pooled_by_group(trace_table, familiar)
    sector_transitions = _sector_transition_table(familiar, novel_native)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "familiar_sector_novel_transfer_neuron_summary.csv"
    trace_by_image_csv = output_dir / "familiar_sector_novel_transfer_by_image_trace_summary.csv"
    trace_pooled_csv = output_dir / "familiar_sector_novel_transfer_pooled_trace_summary.csv"
    sector_transition_csv = output_dir / "familiar_to_novel_sector_transition_matrix.csv"
    merged.to_csv(summary_csv, index=False)
    trace_by_image.to_csv(trace_by_image_csv, index=False)
    trace_pooled.to_csv(trace_pooled_csv, index=False)
    sector_transitions.to_csv(sector_transition_csv, index=False)

    saved = [summary_csv, trace_by_image_csv, trace_pooled_csv, sector_transition_csv]
    saved.extend(
        _save_combined_figure(
            familiar=familiar,
            novel=novel,
            trace_summary=trace_by_image,
            by_image=True,
            output_dir=output_dir,
            basename="familiar_sector_novel_transfer_by_image",
            formats=formats,
            dpi=dpi,
        )
    )
    saved.extend(
        _save_combined_figure(
            familiar=familiar,
            novel=novel,
            trace_summary=trace_pooled,
            by_image=False,
            output_dir=output_dir,
            basename="familiar_sector_novel_transfer_pooled",
            formats=formats,
            dpi=dpi,
        )
    )
    saved.extend(
        _plot_sector_transition_heatmap(
            sector_transitions,
            output_dir=output_dir,
            basename="familiar_to_novel_sector_transition_heatmap",
            formats=formats,
            dpi=dpi,
        )
    )
    return saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot familiar-expert rotated sectors and reuse those same neuron labels "
            "for novel-expert scatter and naive/expert trace averages."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "familiar_sector_novel_transfer",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--formats", nargs="+", default=["png", "svg", "eps"], choices=("png", "svg", "eps"))
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    saved = export_familiar_sector_novel_transfer(
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
