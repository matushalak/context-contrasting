from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

import context_contrasting.data_analysis.transitions_helpers as th


DEFAULT_RESPONSE_THRESHOLD = 0.3
PAPER_STIMULUS_SELECTIVITY_THRESHOLD = 0.3
PAPER_NOVEL_SELECTIVITY_THRESHOLD = 0.5

COUNT_COLORS = (
    "#9E9E9E",
    "#0072B2",
    "#56B4E9",
    "#009E73",
    "#E69F00",
    "#D55E00",
    "#CC79A7",
    "#332288",
    "#88CCEE",
)

DISPLAY_LABELS = {
    "task_familiar": "Task familiar",
    "expert_familiar": "Expert familiar",
    "expert_novel": "Expert novel",
}

TARGET_LABELS = {
    "task_familiar": "Task",
    "expert_familiar": "Expert",
    "expert_novel": "Expert",
}


def _threshold_tag(threshold: float) -> str:
    return f"gt_{threshold:g}".replace(".", "_").replace("-", "minus_")


def _threshold_folder(threshold: float) -> str:
    return f"{threshold:g}"


def _count_cmap(max_count: int) -> tuple[ListedColormap, BoundaryNorm]:
    if max_count < len(COUNT_COLORS):
        colors = COUNT_COLORS[: max_count + 1]
    else:
        colors = plt.cm.tab20(np.linspace(0.0, 1.0, max_count + 1))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, max_count + 1.5, 1.0), cmap.N)
    return cmap, norm


def build_naive_responsiveness_counts(
    transition_table: pd.DataFrame,
    *,
    response_threshold: float = DEFAULT_RESPONSE_THRESHOLD,
) -> pd.DataFrame:
    """Count images with a naive non-occluded response above threshold per neuron."""
    naive = transition_table.loc[transition_table["stage"].astype(str) == "Pre"].copy()
    naive = naive[
        [
            "neuron_idx",
            "image_group",
            "image_idx_original",
            "image_idx_within_group",
            "NO",
        ]
    ].drop_duplicates()
    naive["naive_responsive"] = naive["NO"] > response_threshold

    total_counts = (
        naive.groupby("neuron_idx", as_index=False)
        .agg(
            naive_responsive_image_count=("naive_responsive", "sum"),
            naive_total_image_count=("naive_responsive", "size"),
            naive_mean_NO=("NO", "mean"),
            naive_max_NO=("NO", "max"),
        )
    )

    group_counts = (
        naive.groupby(["neuron_idx", "image_group"])["naive_responsive"]
        .sum()
        .unstack(fill_value=0)
        .astype(int)
    )
    group_counts.columns = [f"naive_responsive_{group}_count" for group in group_counts.columns]

    group_totals = (
        naive.groupby(["neuron_idx", "image_group"])
        .size()
        .unstack(fill_value=0)
        .astype(int)
    )
    group_totals.columns = [f"naive_total_{group}_count" for group in group_totals.columns]

    counts = total_counts.merge(group_counts.reset_index(), on="neuron_idx", how="left")
    counts = counts.merge(group_totals.reset_index(), on="neuron_idx", how="left")

    count_cols = [col for col in counts.columns if col.endswith("_count")]
    counts[count_cols] = counts[count_cols].fillna(0).astype(int)
    counts["response_threshold"] = response_threshold
    return counts


def _lifetime_sparseness(responses: pd.DataFrame) -> pd.Series:
    values = responses.to_numpy(dtype=float)
    n_stimuli = values.shape[1]
    sum_responses = np.nansum(values, axis=1)
    sum_squared_responses = np.nansum(values**2, axis=1)
    denominator = n_stimuli * sum_squared_responses

    sparseness = np.full(values.shape[0], np.nan, dtype=float)
    valid = denominator > 0.0
    sparseness[valid] = 1.0 - (sum_responses[valid] ** 2) / denominator[valid]
    return pd.Series(sparseness, index=responses.index)


def build_lifetime_sparseness(
    transition_table: pd.DataFrame,
    *,
    stage: str,
    image_groups: str | tuple[str, ...],
    response_col: str = "NO",
    response_threshold: float | None = PAPER_STIMULUS_SELECTIVITY_THRESHOLD,
    expected_image_count: int | None = None,
    output_col: str,
) -> pd.DataFrame:
    """Compute the paper lifetime-sparseness selectivity formula per neuron."""
    groups = (image_groups,) if isinstance(image_groups, str) else tuple(image_groups)
    rows = transition_table.loc[
        transition_table["stage"].astype(str).eq(stage)
        & transition_table["image_group"].astype(str).isin(groups)
    ].copy()
    if rows.empty:
        raise ValueError(f"No rows found for stage={stage!r}, image_groups={groups!r}.")

    responses = rows.pivot_table(
        index="neuron_idx",
        columns="image_idx_original",
        values=response_col,
        aggfunc="mean",
    )
    if expected_image_count is not None:
        responses = responses.loc[responses.notna().sum(axis=1) == expected_image_count]

    result = pd.DataFrame({"neuron_idx": responses.index})
    result[output_col] = _lifetime_sparseness(responses).to_numpy(dtype=float)
    result[f"{output_col}_mean_response"] = responses.mean(axis=1).to_numpy(dtype=float)
    result[f"{output_col}_image_count"] = responses.notna().sum(axis=1).to_numpy(dtype=int)

    if response_threshold is not None:
        responsive = result[f"{output_col}_mean_response"] > response_threshold
        result.loc[~responsive, output_col] = np.nan
    return result.reset_index(drop=True)


def build_paper_novel_selectivity(
    transition_table: pd.DataFrame,
    *,
    stage: str,
    response_col: str = "NO",
    novel_response_threshold: float = PAPER_NOVEL_SELECTIVITY_THRESHOLD,
    output_col: str = "paper_novel_selectivity",
) -> pd.DataFrame:
    """Average pairwise (R_novel - R_familiar) / (R_novel + R_familiar) per neuron."""
    rows = transition_table.loc[
        transition_table["stage"].astype(str).eq(stage)
        & transition_table["image_group"].astype(str).isin(["familiar", "novel"])
    ].copy()
    if rows.empty:
        raise ValueError(f"No familiar/novel rows found for stage={stage!r}.")

    responses = rows.pivot_table(
        index="neuron_idx",
        columns=["image_group", "image_idx_original"],
        values=response_col,
        aggfunc="mean",
    )
    familiar_cols = [col for col in responses.columns if col[0] == "familiar"]
    novel_cols = [col for col in responses.columns if col[0] == "novel"]

    result_rows: list[dict[str, float | int]] = []
    for neuron_idx, row in responses.iterrows():
        familiar = row[familiar_cols].dropna().to_numpy(dtype=float)
        novel = row[novel_cols].dropna().to_numpy(dtype=float)
        pair_indices: list[float] = []
        if len(novel) and float(np.nanmean(novel)) > novel_response_threshold:
            for novel_response in novel:
                for familiar_response in familiar:
                    if novel_response > 0.0 and familiar_response > 0.0:
                        denominator = novel_response + familiar_response
                        if denominator > 0.0:
                            pair_indices.append((novel_response - familiar_response) / denominator)
        result_rows.append(
            {
                "neuron_idx": int(neuron_idx),
                output_col: float(np.nanmean(pair_indices)) if pair_indices else np.nan,
                f"{output_col}_pair_count": len(pair_indices),
                f"{output_col}_mean_novel_response": float(np.nanmean(novel)) if len(novel) else np.nan,
                f"{output_col}_mean_familiar_response": (
                    float(np.nanmean(familiar)) if len(familiar) else np.nan
                ),
            }
        )

    return pd.DataFrame(result_rows)


def build_colored_summary(
    transition_table: pd.DataFrame,
    *,
    image_group: str,
    target_stage: str,
    response_threshold: float,
    sector_threshold: float,
) -> pd.DataFrame:
    summary = th.build_mean_summary(
        transition_table,
        image_group=image_group,
        pre_stage="Pre",
        target_stage=target_stage,
        threshold=sector_threshold,
    )
    counts = build_naive_responsiveness_counts(
        transition_table,
        response_threshold=response_threshold,
    )
    colored = summary.merge(counts, on="neuron_idx", how="left", validate="one_to_one")
    if colored["naive_responsive_image_count"].isna().any():
        missing = colored.loc[colored["naive_responsive_image_count"].isna(), "neuron_idx"].tolist()
        raise ValueError(f"Missing naive responsiveness counts for neurons: {missing[:10]}")
    colored["summary_image_group"] = image_group
    colored["target_stage"] = target_stage
    return colored


def _merge_neuron_metric(summary: pd.DataFrame, metric: pd.DataFrame) -> pd.DataFrame:
    return summary.merge(metric, on="neuron_idx", how="left", validate="one_to_one")


def _draw_response_guides(ax: plt.Axes, response_lims: list[float]) -> None:
    ax.plot(response_lims, response_lims, "--", color="0.75", linewidth=1.0, zorder=0)
    ax.axhline(0.0, color="0.86", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.86", linewidth=0.8, zorder=0)


def _count_rgba(
    frame: pd.DataFrame,
    *,
    cmap: ListedColormap,
    norm: BoundaryNorm,
    alpha: float,
) -> np.ndarray:
    colors = cmap(norm(frame["naive_responsive_image_count"].to_numpy(dtype=int)))
    colors[:, 3] = alpha
    return colors


def _draw_displacement_vectors(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    cmap: ListedColormap,
    norm: BoundaryNorm,
    alpha: float = 0.28,
) -> None:
    ax.quiver(
        frame["NO_Pre"],
        frame["O_Pre"],
        frame["dNO"],
        frame["dO"],
        color=_count_rgba(frame, cmap=cmap, norm=norm, alpha=alpha),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0032,
        headwidth=3.4,
        headlength=4.2,
        headaxislength=3.7,
        zorder=1,
    )


def _scatter_response_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    response_lims: list[float],
    cmap: ListedColormap,
    norm: BoundaryNorm,
    point_size: float,
) -> None:
    ax.scatter(
        frame[x_col],
        frame[y_col],
        c=frame["naive_responsive_image_count"],
        cmap=cmap,
        norm=norm,
        s=point_size,
        alpha=0.86,
        edgecolors="white",
        linewidths=0.25,
        zorder=2,
    )
    _draw_response_guides(ax, response_lims)
    ax.set_xlim(response_lims)
    ax.set_ylim(response_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("NO")
    ax.set_ylabel("O")


def plot_combined_responsiveness_scatter(
    summaries: dict[str, pd.DataFrame],
    *,
    response_lims: list[float],
    response_threshold: float,
    point_size: float = 30.0,
) -> plt.Figure:
    max_count = int(max(frame["naive_responsive_image_count"].max() for frame in summaries.values()))
    cmap, norm = _count_cmap(max_count)

    fig, axes = plt.subplots(
        len(summaries),
        2,
        figsize=(8.6, 4.05 * len(summaries)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(summaries) == 1:
        axes = np.asarray([axes])

    for row_idx, (label, frame) in enumerate(summaries.items()):
        display_label = DISPLAY_LABELS.get(label, label)
        target_label = TARGET_LABELS.get(label, str(frame["target_stage"].iloc[0]))
        _scatter_response_panel(
            axes[row_idx, 0],
            frame,
            x_col="NO_Pre",
            y_col="O_Pre",
            title=f"{display_label}: naive",
            response_lims=response_lims,
            cmap=cmap,
            norm=norm,
            point_size=point_size,
        )
        _scatter_response_panel(
            axes[row_idx, 1],
            frame,
            x_col="NO_Target",
            y_col="O_Target",
            title=f"{display_label}: {target_label}",
            response_lims=response_lims,
            cmap=cmap,
            norm=norm,
            point_size=point_size,
        )

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        ticks=np.arange(0, max_count + 1),
        shrink=0.78,
        pad=0.02,
    )
    cbar.set_label(f"Naive responsive images (Pre NO > {response_threshold:g})")

    fig.suptitle("Chronically matched neurons by naive image responsiveness", fontsize=14, fontweight="bold")
    return fig


def plot_single_responsiveness_scatter(
    frame: pd.DataFrame,
    *,
    label: str,
    response_lims: list[float],
    response_threshold: float,
    point_size: float = 30.0,
) -> plt.Figure:
    max_count = int(frame["naive_responsive_image_count"].max())
    cmap, norm = _count_cmap(max_count)
    display_label = DISPLAY_LABELS.get(label, label)
    target_label = TARGET_LABELS.get(label, str(frame["target_stage"].iloc[0]))

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), sharex=True, sharey=True, constrained_layout=True)
    _scatter_response_panel(
        axes[0],
        frame,
        x_col="NO_Pre",
        y_col="O_Pre",
        title=f"{display_label}: naive",
        response_lims=response_lims,
        cmap=cmap,
        norm=norm,
        point_size=point_size,
    )
    _scatter_response_panel(
        axes[1],
        frame,
        x_col="NO_Target",
        y_col="O_Target",
        title=f"{display_label}: {target_label}",
        response_lims=response_lims,
        cmap=cmap,
        norm=norm,
        point_size=point_size,
    )

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        ticks=np.arange(0, max_count + 1),
        shrink=0.82,
        pad=0.02,
    )
    cbar.set_label(f"Naive responsive images (Pre NO > {response_threshold:g})")

    fig.suptitle(f"{display_label}: chronic scatter by naive responsiveness", fontsize=13, fontweight="bold")
    return fig


def build_mean_displacement_by_responsiveness(merged: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        merged.groupby(["summary_name", "naive_responsive_image_count"], as_index=False)
        .agg(
            neuron_count=("neuron_idx", "size"),
            mean_NO_Pre=("NO_Pre", "mean"),
            mean_O_Pre=("O_Pre", "mean"),
            mean_NO_Target=("NO_Target", "mean"),
            mean_O_Target=("O_Target", "mean"),
            mean_dNO=("dNO", "mean"),
            mean_dO=("dO", "mean"),
            sem_dNO=("dNO", "sem"),
            sem_dO=("dO", "sem"),
            median_dNO=("dNO", "median"),
            median_dO=("dO", "median"),
            mean_dNorm=("dNorm", "mean"),
            median_dNorm=("dNorm", "median"),
        )
        .sort_values(["summary_name", "naive_responsive_image_count"])
    )
    totals = merged.groupby("summary_name").size()
    grouped["fraction_of_summary"] = grouped.apply(
        lambda row: row["neuron_count"] / totals.loc[row["summary_name"]],
        axis=1,
    )
    grouped["mean_vector_norm"] = np.hypot(grouped["mean_dNO"], grouped["mean_dO"])
    grouped["mean_vector_angle_deg"] = np.degrees(np.arctan2(grouped["mean_dO"], grouped["mean_dNO"]))
    return grouped


def _draw_shift_guides(ax: plt.Axes, shift_lims: list[float]) -> None:
    ax.axhline(0.0, color="0.82", linewidth=1.0, zorder=0)
    ax.axvline(0.0, color="0.82", linewidth=1.0, zorder=0)
    extent = float(max(abs(shift_lims[0]), abs(shift_lims[1])))
    guide = np.array([-extent, extent], dtype=float)
    ax.plot(guide, guide, "--", color="0.86", linewidth=0.8, zorder=0)
    ax.plot(guide, -guide, "--", color="0.86", linewidth=0.8, zorder=0)


def plot_mean_displacement_by_responsiveness(
    mean_displacements: pd.DataFrame,
    *,
    response_threshold: float,
) -> plt.Figure:
    summary_names = [name for name in DISPLAY_LABELS if name in set(mean_displacements["summary_name"])]
    summary_names.extend(
        name for name in mean_displacements["summary_name"].drop_duplicates().tolist() if name not in summary_names
    )
    max_count = int(mean_displacements["naive_responsive_image_count"].max())
    cmap, norm = _count_cmap(max_count)

    max_abs = float(
        np.nanmax(
            np.abs(mean_displacements[["mean_dNO", "mean_dO"]].to_numpy(dtype=float))
        )
    )
    extent = max(0.25, max_abs * 1.35)
    shift_lims = [-extent, extent]

    fig, axes = plt.subplots(
        1,
        len(summary_names),
        figsize=(4.2 * len(summary_names), 4.1),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)

    for ax, summary_name in zip(axes, summary_names):
        rows = mean_displacements.loc[mean_displacements["summary_name"] == summary_name]
        _draw_shift_guides(ax, shift_lims)
        for row in rows.itertuples(index=False):
            count = int(row.naive_responsive_image_count)
            color = cmap(norm([count]))[0]
            arrow_width = max(0.006, extent * 0.018)
            head_width = max(0.024, extent * 0.070)
            head_length = max(0.030, extent * 0.090)
            ax.arrow(
                0.0,
                0.0,
                float(row.mean_dNO),
                float(row.mean_dO),
                color=color,
                alpha=0.92,
                linewidth=2.1,
                width=arrow_width,
                head_width=head_width,
                head_length=head_length,
                length_includes_head=True,
                zorder=2,
            )
            ax.scatter(
                [row.mean_dNO],
                [row.mean_dO],
                s=24.0 + 2.0 * float(row.neuron_count),
                color=[color],
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )
            text_dx = 0.035 * extent if row.mean_dNO >= 0 else -0.035 * extent
            text_dy = 0.035 * extent if row.mean_dO >= 0 else -0.035 * extent
            ax.text(
                float(row.mean_dNO) + text_dx,
                float(row.mean_dO) + text_dy,
                f"{count}",
                color=color,
                fontsize=9,
                fontweight="bold",
                ha="left" if row.mean_dNO >= 0 else "right",
                va="bottom" if row.mean_dO >= 0 else "top",
                zorder=4,
            )

        ax.set_title(DISPLAY_LABELS.get(summary_name, summary_name))
        ax.set_xlim(shift_lims)
        ax.set_ylim(shift_lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("mean dNO")
        ax.set_ylabel("mean dO")

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        ticks=np.arange(0, max_count + 1),
        shrink=0.82,
        pad=0.02,
    )
    cbar.set_label(f"Naive responsive images (Pre NO > {response_threshold:g})")
    fig.suptitle("Mean displacement vector by naive responsiveness count", fontsize=13, fontweight="bold")
    return fig


def _delta_lims(frame: pd.DataFrame) -> list[float]:
    max_abs = float(np.nanmax(np.abs(frame[["dNO", "dO"]].to_numpy(dtype=float))))
    extent = max(0.25, max_abs * 1.08)
    return [-extent, extent]


def plot_delta_by_responsive_count(
    frame: pd.DataFrame,
    *,
    label: str,
    response_threshold: float,
    point_size: float = 34.0,
) -> plt.Figure:
    ordered = frame.sort_values("neuron_idx").reset_index(drop=True)
    max_count = int(ordered["naive_responsive_image_count"].max())
    cmap, norm = _count_cmap(max_count)
    display_label = DISPLAY_LABELS.get(label, label)
    target_label = TARGET_LABELS.get(label, str(ordered["target_stage"].iloc[0]))
    delta_lims = _delta_lims(ordered)

    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    _draw_shift_guides(ax, delta_lims)
    ax.scatter(
        ordered["dNO"],
        ordered["dO"],
        c=ordered["naive_responsive_image_count"],
        cmap=cmap,
        norm=norm,
        s=point_size,
        alpha=0.86,
        edgecolors="white",
        linewidths=0.25,
        zorder=2,
    )
    ax.set_xlim(delta_lims)
    ax.set_ylim(delta_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("dNO")
    ax.set_ylabel("dO")
    ax.set_title(f"{display_label}: {target_label} - Pre")

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        ticks=np.arange(0, max_count + 1),
        shrink=0.82,
        pad=0.02,
    )
    cbar.set_label(f"Naive responsive images (Pre NO > {response_threshold:g})")
    fig.suptitle(
        "Transition plane colored by naive responsiveness count",
        fontsize=13,
        fontweight="bold",
    )
    return fig


def _x_values(frame: pd.DataFrame, *, x_col: str, jitter: float = 0.0) -> np.ndarray:
    values = frame[x_col].to_numpy(dtype=float)
    if jitter <= 0.0:
        return values
    rng = np.random.default_rng(1729)
    return values + rng.uniform(-jitter, jitter, len(frame))


def _directional_sectors() -> tuple[str, ...]:
    return tuple(sector for sector in th.ROTATED_SECTOR_ORDER if sector != "small ∆")


def build_sector_count_summary(frame: pd.DataFrame) -> pd.DataFrame:
    count_col = "naive_responsive_image_count"
    grouped = (
        frame.groupby(["RotatedSector", count_col], observed=True, as_index=False)
        .agg(
            neuron_count=("neuron_idx", "size"),
            mean_dNO=("dNO", "mean"),
            mean_dO=("dO", "mean"),
            sem_dNO=("dNO", "sem"),
            sem_dO=("dO", "sem"),
        )
        .sort_values(["RotatedSector", count_col])
    )
    totals = frame.groupby(count_col, observed=True).size().rename("count_total").reset_index()
    grouped = grouped.merge(totals, on=count_col, how="left", validate="many_to_one")
    grouped["percent_within_count"] = grouped["neuron_count"] / grouped["count_total"] * 100.0
    return grouped


def _selectivity_specs_for_label(label: str) -> list[tuple[str, str, str, str]]:
    specs: list[tuple[str, str, str, str]] = []
    if label in {"task_familiar", "expert_familiar"}:
        specs.append(
            (
                "paper_stimulus_selectivity",
                "paper stimulus selectivity",
                "paper stimulus selectivity",
                "paper_stimulus_selectivity_resp_gt_0_3",
            )
        )
    if label == "expert_novel":
        specs.append(
            (
                "paper_novel_selectivity",
                "paper novel selectivity",
                "paper novel selectivity",
                "paper_novel_selectivity_novel_gt_0_5",
            )
        )
    if label in {"expert_familiar", "expert_novel"}:
        specs.append(
            (
                "all6_lifetime_sparseness",
                "lifetime sparseness over all 6 images",
                "six-image lifetime sparseness",
                "all6_lifetime_sparseness_resp_gt_0_3",
            )
        )
    return specs


def _fit_line(x: np.ndarray, y: np.ndarray) -> dict[str, float | int] | None:
    finite = np.isfinite(x) & np.isfinite(y)
    x_valid = x[finite]
    y_valid = y[finite]
    if len(x_valid) < 2 or len(np.unique(x_valid)) < 2:
        return None

    slope, intercept = np.polyfit(x_valid, y_valid, deg=1)
    if len(x_valid) > 1 and np.nanstd(x_valid) > 0.0 and np.nanstd(y_valid) > 0.0:
        r_value = float(np.corrcoef(x_valid, y_valid)[0, 1])
    else:
        r_value = np.nan
    return {
        "n": int(len(x_valid)),
        "slope": float(slope),
        "intercept": float(intercept),
        "r": r_value,
        "x_min": float(np.nanmin(x_valid)),
        "x_max": float(np.nanmax(x_valid)),
    }


def build_sector_x_fit_summary(
    frame: pd.DataFrame,
    *,
    x_col: str,
    x_measure: str,
    label: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for delta_col in ("dNO", "dO"):
        for sector in th.ROTATED_SECTOR_ORDER:
            sector_rows = frame.loc[frame["RotatedSector"].astype(str).eq(sector)]
            fit = _fit_line(
                sector_rows[x_col].to_numpy(dtype=float),
                sector_rows[delta_col].to_numpy(dtype=float),
            )
            if fit is None:
                finite_count = int(
                    (
                        np.isfinite(sector_rows[x_col].to_numpy(dtype=float))
                        & np.isfinite(sector_rows[delta_col].to_numpy(dtype=float))
                    ).sum()
                )
                rows.append(
                    {
                        "summary_name": label,
                        "x_measure": x_measure,
                        "delta_component": delta_col,
                        "RotatedSector": sector,
                        "n": finite_count,
                        "slope": np.nan,
                        "intercept": np.nan,
                        "r": np.nan,
                        "x_min": np.nan,
                        "x_max": np.nan,
                    }
                )
            else:
                rows.append(
                    {
                        "summary_name": label,
                        "x_measure": x_measure,
                        "delta_component": delta_col,
                        "RotatedSector": sector,
                        **fit,
                    }
                )
    return pd.DataFrame(rows)


def _x_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return (0.0, 1.0)
    x_min = float(np.nanmin(finite))
    x_max = float(np.nanmax(finite))
    if np.isclose(x_min, x_max):
        pad = max(0.05, abs(x_min) * 0.05)
    else:
        pad = max(0.025, (x_max - x_min) * 0.06)
    return (x_min - pad, x_max + pad)


def plot_delta_by_x_sectors(
    frame: pd.DataFrame,
    *,
    label: str,
    x_col: str,
    x_label: str,
    x_measure_label: str,
    jitter: float = 0.08,
    point_size: float = 34.0,
    integer_ticks: bool = False,
) -> plt.Figure:
    ordered = (
        frame.loc[np.isfinite(frame[x_col].to_numpy(dtype=float))]
        .sort_values("neuron_idx")
        .reset_index(drop=True)
    )
    x = _x_values(ordered, x_col=x_col, jitter=jitter)
    display_label = DISPLAY_LABELS.get(label, label)
    x_raw = ordered[x_col].to_numpy(dtype=float)
    x_min, x_max = _x_limits(x_raw)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7), sharex=True, constrained_layout=True)
    for col_idx, delta_col in enumerate(("dNO", "dO")):
        ax = axes[col_idx]
        for sector in th._sector_plot_order(small_delta_first=True):
            sector_mask = ordered["RotatedSector"].astype(str).eq(sector).to_numpy()
            if not sector_mask.any():
                continue
            ax.scatter(
                x[sector_mask],
                ordered.loc[sector_mask, delta_col],
                color=th.ROTATED_SECTOR_PALETTE[sector],
                s=point_size,
                alpha=0.86,
                edgecolors="white",
                linewidths=0.25,
                label=sector,
                zorder=th._sector_scatter_zorder(sector),
            )
        for sector in th.ROTATED_SECTOR_ORDER:
            sector_rows = ordered.loc[ordered["RotatedSector"].astype(str).eq(sector)]
            fit = _fit_line(
                sector_rows[x_col].to_numpy(dtype=float),
                sector_rows[delta_col].to_numpy(dtype=float),
            )
            if fit is None:
                continue
            x_line = np.linspace(float(fit["x_min"]), float(fit["x_max"]), 100)
            y_line = float(fit["slope"]) * x_line + float(fit["intercept"])
            ax.plot(
                x_line,
                y_line,
                color=th.ROTATED_SECTOR_PALETTE[sector],
                linewidth=2.0,
                alpha=0.95,
                zorder=6,
            )
        ax.set_title(f"{delta_col} colored by rotated sector")
        ax.axhline(0.0, color="0.75", lw=0.9, zorder=0)
        ax.set_xlim(x_min, x_max)
        if integer_ticks:
            tick_min = int(np.floor(np.nanmin(x_raw)))
            tick_max = int(np.ceil(np.nanmax(x_raw)))
            ax.set_xticks(np.arange(tick_min, tick_max + 1, 1))
        ax.set_xlabel(x_label)
        ax.set_ylabel(delta_col)

    handles, legend_labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            legend_labels,
            frameon=False,
            loc="center right",
            bbox_to_anchor=(1.08, 0.50),
        )
    fig.suptitle(
        f"{display_label}: response components by {x_measure_label}",
        fontsize=13,
        fontweight="bold",
    )
    return fig


def plot_delta_by_responsive_count_sectors(
    frame: pd.DataFrame,
    *,
    label: str,
    jitter: float = 0.08,
    point_size: float = 34.0,
) -> plt.Figure:
    return plot_delta_by_x_sectors(
        frame,
        label=label,
        x_col="naive_responsive_image_count",
        x_label="naive responsive image count",
        x_measure_label="naive responsiveness count",
        jitter=jitter,
        point_size=point_size,
        integer_ticks=True,
    )


def plot_sector_percentage_by_responsive_count(
    frame: pd.DataFrame,
    *,
    label: str,
) -> plt.Figure:
    ordered = frame.sort_values("neuron_idx").reset_index(drop=True)
    max_count = int(ordered["naive_responsive_image_count"].max())
    display_label = DISPLAY_LABELS.get(label, label)
    sector_count_summary = build_sector_count_summary(ordered)
    count_index = pd.Index(range(max_count + 1), name="naive_responsive_image_count")

    fig, ax = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
    for sector in _directional_sectors():
        rows = sector_count_summary.loc[
            sector_count_summary["RotatedSector"].astype(str).eq(sector)
        ].set_index("naive_responsive_image_count").reindex(count_index)
        ax.plot(
            count_index.to_numpy(dtype=int),
            rows["percent_within_count"].fillna(0.0),
            color=th.ROTATED_SECTOR_PALETTE[sector],
            marker="o",
            linewidth=2.2,
            label=sector,
        )

    ax.set_xlim(-0.45, max_count + 0.45)
    ax.set_xticks(np.arange(0, max_count + 1, 1))
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("naive responsive image count")
    ax.set_ylabel("neurons in sector (%)")
    ax.set_title(f"{display_label}: rotated-sector composition by count")
    ax.grid(axis="y", color="0.90", linewidth=0.8)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    return fig


def export_naive_responsiveness_plots(
    *,
    data_dir: Path,
    output_dir: Path,
    response_threshold: float,
    sector_threshold: float,
    point_size: float,
) -> list[Path]:
    output_dir = output_dir / _threshold_folder(response_threshold)
    output_dir.mkdir(parents=True, exist_ok=True)

    act_table = th.load_transition_table(data_dir / "transitions_act.csv")
    post_table = th.load_transition_table(data_dir / "transitions_post.csv")

    summaries = {
        "task_familiar": build_colored_summary(
            act_table,
            image_group="all",
            target_stage="Task",
            response_threshold=response_threshold,
            sector_threshold=sector_threshold,
        ),
        "expert_familiar": build_colored_summary(
            post_table,
            image_group="familiar",
            target_stage="Post",
            response_threshold=response_threshold,
            sector_threshold=sector_threshold,
        ),
        "expert_novel": build_colored_summary(
            post_table,
            image_group="novel",
            target_stage="Post",
            response_threshold=response_threshold,
            sector_threshold=sector_threshold,
        ),
    }

    task_stimulus_selectivity = build_lifetime_sparseness(
        act_table,
        stage="Pre",
        image_groups="all",
        response_threshold=PAPER_STIMULUS_SELECTIVITY_THRESHOLD,
        expected_image_count=4,
        output_col="paper_stimulus_selectivity",
    )
    expert_familiar_stimulus_selectivity = build_lifetime_sparseness(
        post_table,
        stage="Pre",
        image_groups="familiar",
        response_threshold=PAPER_STIMULUS_SELECTIVITY_THRESHOLD,
        expected_image_count=4,
        output_col="paper_stimulus_selectivity",
    )
    expert_novel_selectivity = build_paper_novel_selectivity(
        post_table,
        stage="Pre",
        novel_response_threshold=PAPER_NOVEL_SELECTIVITY_THRESHOLD,
        output_col="paper_novel_selectivity",
    )
    expert_all6_sparseness = build_lifetime_sparseness(
        post_table,
        stage="Pre",
        image_groups=("familiar", "novel"),
        response_threshold=PAPER_STIMULUS_SELECTIVITY_THRESHOLD,
        expected_image_count=6,
        output_col="all6_lifetime_sparseness",
    )

    summaries["task_familiar"] = _merge_neuron_metric(
        summaries["task_familiar"],
        task_stimulus_selectivity,
    )
    summaries["expert_familiar"] = _merge_neuron_metric(
        summaries["expert_familiar"],
        expert_familiar_stimulus_selectivity,
    )
    summaries["expert_familiar"] = _merge_neuron_metric(
        summaries["expert_familiar"],
        expert_all6_sparseness,
    )
    summaries["expert_novel"] = _merge_neuron_metric(
        summaries["expert_novel"],
        expert_novel_selectivity,
    )
    summaries["expert_novel"] = _merge_neuron_metric(
        summaries["expert_novel"],
        expert_all6_sparseness,
    )

    for label, frame in summaries.items():
        frame.insert(0, "summary_name", label)

    response_lims = th.compute_response_limits(*summaries.values())
    threshold_tag = _threshold_tag(response_threshold)
    saved_paths: list[Path] = []

    combined = plot_combined_responsiveness_scatter(
        summaries,
        response_lims=response_lims,
        response_threshold=response_threshold,
        point_size=point_size,
    )
    for suffix in ("png", "svg"):
        path = output_dir / f"chronic_scatter_by_naive_responsive_count_{threshold_tag}.{suffix}"
        combined.savefig(path, dpi=300, bbox_inches="tight")
        saved_paths.append(path)
    plt.close(combined)

    for label, frame in summaries.items():
        fig = plot_single_responsiveness_scatter(
            frame,
            label=label,
            response_lims=response_lims,
            response_threshold=response_threshold,
            point_size=point_size,
        )
        for suffix in ("png", "svg"):
            path = output_dir / f"{label}_scatter_by_naive_responsive_count_{threshold_tag}.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            saved_paths.append(path)
        plt.close(fig)

        delta_fig = plot_delta_by_responsive_count(
            frame,
            label=label,
            response_threshold=response_threshold,
            point_size=point_size,
        )
        for suffix in ("png", "svg"):
            path = output_dir / f"{label}_delta_by_naive_responsive_count_{threshold_tag}.{suffix}"
            delta_fig.savefig(path, dpi=300, bbox_inches="tight")
            saved_paths.append(path)
        plt.close(delta_fig)

        sector_delta_fig = plot_delta_by_responsive_count_sectors(
            frame,
            label=label,
            point_size=point_size,
        )
        for suffix in ("png", "svg"):
            path = output_dir / f"{label}_delta_by_rotated_sector_count_{threshold_tag}.{suffix}"
            sector_delta_fig.savefig(path, dpi=300, bbox_inches="tight")
            saved_paths.append(path)
        plt.close(sector_delta_fig)

        for x_col, x_label, x_measure_label, x_slug in _selectivity_specs_for_label(label):
            selectivity_delta_fig = plot_delta_by_x_sectors(
                frame,
                label=label,
                x_col=x_col,
                x_label=x_label,
                x_measure_label=x_measure_label,
                jitter=0.0,
                point_size=point_size,
            )
            for suffix in ("png", "svg"):
                path = output_dir / f"{label}_delta_by_rotated_sector_{x_slug}_{threshold_tag}.{suffix}"
                selectivity_delta_fig.savefig(path, dpi=300, bbox_inches="tight")
                saved_paths.append(path)
            plt.close(selectivity_delta_fig)

        sector_percent_fig = plot_sector_percentage_by_responsive_count(
            frame,
            label=label,
        )
        for suffix in ("png", "svg"):
            path = output_dir / f"{label}_sector_percentage_by_naive_responsive_count_{threshold_tag}.{suffix}"
            sector_percent_fig.savefig(path, dpi=300, bbox_inches="tight")
            saved_paths.append(path)
        plt.close(sector_percent_fig)

    merged = pd.concat(summaries.values(), ignore_index=True)
    summary_csv = output_dir / f"chronic_scatter_by_naive_responsive_count_{threshold_tag}.csv"
    merged.to_csv(summary_csv, index=False)
    saved_paths.append(summary_csv)

    count_distribution = (
        merged.groupby(["summary_name", "naive_responsive_image_count"], as_index=False)
        .agg(neuron_count=("neuron_idx", "size"))
        .sort_values(["summary_name", "naive_responsive_image_count"])
    )
    distribution_csv = output_dir / f"naive_responsive_count_distribution_{threshold_tag}.csv"
    count_distribution.to_csv(distribution_csv, index=False)
    saved_paths.append(distribution_csv)

    sector_count_summary = pd.concat(
        [
            build_sector_count_summary(frame).assign(summary_name=label)
            for label, frame in summaries.items()
        ],
        ignore_index=True,
    )
    sector_count_csv = output_dir / f"sector_count_summary_{threshold_tag}.csv"
    sector_count_summary.to_csv(sector_count_csv, index=False)
    saved_paths.append(sector_count_csv)

    sector_fit_summary = pd.concat(
        [
            build_sector_x_fit_summary(
                frame,
                x_col="naive_responsive_image_count",
                x_measure="naive_responsive_image_count",
                label=label,
            )
            for label, frame in summaries.items()
        ]
        + [
            build_sector_x_fit_summary(
                frame,
                x_col=x_col,
                x_measure=x_slug,
                label=label,
            )
            for label, frame in summaries.items()
            for x_col, _, _, x_slug in _selectivity_specs_for_label(label)
        ],
        ignore_index=True,
    )
    sector_fit_csv = output_dir / f"sector_x_fit_summary_{threshold_tag}.csv"
    sector_fit_summary.to_csv(sector_fit_csv, index=False)
    saved_paths.append(sector_fit_csv)

    mean_displacements = build_mean_displacement_by_responsiveness(merged)
    mean_displacement_csv = output_dir / f"mean_displacement_by_naive_responsive_count_{threshold_tag}.csv"
    mean_displacements.to_csv(mean_displacement_csv, index=False)
    saved_paths.append(mean_displacement_csv)

    return saved_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot chronic NO/O scatter panels colored by the number of images "
            "that were non-occluded responsive in the naive/pre state."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "naive_responsiveness_count_scatter",
    )
    parser.add_argument(
        "--response-threshold",
        type=float,
        default=DEFAULT_RESPONSE_THRESHOLD,
        help="Pre/NO response threshold used to classify a neuron-image pair as responsive.",
    )
    parser.add_argument(
        "--sector-threshold",
        type=float,
        default=0.3,
        help="Small-delta threshold passed to the existing transition sector summary.",
    )
    parser.add_argument("--point-size", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    saved_paths = export_naive_responsiveness_plots(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        response_threshold=args.response_threshold,
        sector_threshold=args.sector_threshold,
        point_size=args.point_size,
    )
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
