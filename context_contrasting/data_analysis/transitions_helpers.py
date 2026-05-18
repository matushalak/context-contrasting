from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


MATLAB_COLORS = (
    "#0072BD",
    "#D95319",
    "#EDB120",
    "#7E2F8E",
    "#77AC30",
    "#4DBEEE",
    "#A2142F",
)

ROTATED_SECTOR_ORDER = ("+NO axis", "+O axis", "-NO axis", "-O axis")
ROTATED_SECTOR_PALETTE = {
    "+NO axis": MATLAB_COLORS[0],
    "+O axis": MATLAB_COLORS[1],
    "-NO axis": MATLAB_COLORS[2],
    "-O axis": MATLAB_COLORS[3],
}

DEFAULT_PLOT_STYLE = {
    "pre_point_alpha": 0.38,
    "target_point_alpha": 0.22,
    "shift_point_alpha": 0.22,
    "sector_point_alpha": 0.24,
    "point_size": 28,
    "pre_vector_alpha": 0.42,
    "target_vector_alpha": 0.62,
    "individual_vector_width": 0.0046,
    "mean_arrow_width": 2.9,
    "mean_arrow_mutation_scale": 16.0,
}


def resolve_plot_style(style: dict | None = None) -> dict:
    merged = DEFAULT_PLOT_STYLE.copy()
    if style is not None:
        merged.update(style)
    return merged


def build_transition_cmap(size: int = 256) -> np.ndarray:
    reds = np.linspace(0.0, 1.0, size)
    blues = np.linspace(0.0, 1.0, size)
    xx, yy = np.meshgrid(reds, blues)

    cmap = np.zeros((size, size, 3))
    cmap[:, :, 0] = xx
    cmap[:, :, 2] = yy
    return cmap


def _rgb_from_values(
    no_values: pd.Series | np.ndarray,
    o_values: pd.Series | np.ndarray,
    *,
    no_min: float,
    no_max: float,
    o_min: float,
    o_max: float,
    cmap: np.ndarray,
) -> np.ndarray:
    no_values = np.asarray(no_values, dtype=float)
    o_values = np.asarray(o_values, dtype=float)

    no_span = no_max - no_min if no_max > no_min else 1.0
    o_span = o_max - o_min if o_max > o_min else 1.0

    no_idx = np.clip(np.rint((no_values - no_min) / no_span * 255).astype(int), 0, 255)
    o_idx = np.clip(np.rint((o_values - o_min) / o_span * 255).astype(int), 0, 255)
    return cmap[o_idx, no_idx]


def add_direction_columns(
    frame: pd.DataFrame,
    *,
    dx_col: str = "dNO",
    dy_col: str = "dO",
) -> pd.DataFrame:
    frame = frame.copy()
    frame["Angle"] = np.arctan2(frame[dy_col], frame[dx_col])
    frame["CosAngle"] = np.cos(frame["Angle"])
    frame["SinAngle"] = np.sin(frame["Angle"])
    return frame


def load_transition_table(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    raw = pd.read_csv(csv_path)
    stage_order = raw["stage"].drop_duplicates().tolist()

    wide = (
        raw.pivot(
            index=[
                "transition",
                "image_group",
                "image_idx_original",
                "image_idx_within_group",
                "neuron_idx",
                "stage",
            ],
            columns="image_type",
            values="response",
        )
        .reset_index()
        .rename(columns={"Full": "NO", "Occl": "O"})
    )

    wide["stage"] = pd.Categorical(wide["stage"], categories=stage_order, ordered=True)
    wide = wide.sort_values(
        ["transition", "image_group", "image_idx_original", "neuron_idx", "stage"]
    ).reset_index(drop=True)
    return wide


def _resolve_target_stage(
    frame: pd.DataFrame,
    *,
    pre_stage: str = "Pre",
    target_stage: str | None = None,
) -> str:
    if target_stage is not None:
        return target_stage

    candidates = [stage for stage in frame["stage"].astype(str).drop_duplicates().tolist() if stage != pre_stage]
    if not candidates:
        raise ValueError(f"Could not infer a target stage after {pre_stage}.")
    return candidates[0]


def assign_rotated_sectors(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    angle = frame["Angle"]

    sector = np.full(len(frame), "+NO axis", dtype=object)
    sector[(angle >= np.pi / 4.0) & (angle < 3.0 * np.pi / 4.0)] = "+O axis"
    sector[(angle >= 3.0 * np.pi / 4.0) | (angle < -3.0 * np.pi / 4.0)] = "-NO axis"
    sector[(angle >= -3.0 * np.pi / 4.0) & (angle < -np.pi / 4.0)] = "-O axis"

    frame["RotatedSector"] = pd.Categorical(
        sector,
        categories=ROTATED_SECTOR_ORDER,
        ordered=True,
    )
    return frame


def build_mean_summary(
    transition_table: pd.DataFrame,
    *,
    image_group: str | None = None,
    pre_stage: str = "Pre",
    target_stage: str | None = None,
) -> pd.DataFrame:
    frame = transition_table.copy()
    if image_group is not None:
        frame = frame.loc[frame["image_group"] == image_group].copy()

    if frame.empty:
        raise ValueError("No rows left after filtering transition data.")

    target_stage = _resolve_target_stage(frame, pre_stage=pre_stage, target_stage=target_stage)

    stage_means = (
        frame.groupby(["neuron_idx", "stage"], as_index=False)[["NO", "O"]]
        .mean()
    )

    summary = stage_means.pivot(index="neuron_idx", columns="stage", values=["NO", "O"])
    summary.columns = [f"{metric}_{stage}" for metric, stage in summary.columns]
    summary = summary.reset_index()
    summary = summary.dropna(
        subset=[f"NO_{pre_stage}", f"O_{pre_stage}", f"NO_{target_stage}", f"O_{target_stage}"]
    )

    summary = summary.rename(
        columns={
            f"NO_{pre_stage}": "NO_Pre",
            f"O_{pre_stage}": "O_Pre",
            f"NO_{target_stage}": "NO_Target",
            f"O_{target_stage}": "O_Target",
        }
    )
    summary["dNO"] = summary["NO_Target"] - summary["NO_Pre"]
    summary["dO"] = summary["O_Target"] - summary["O_Pre"]
    summary = add_direction_columns(summary)
    summary = assign_rotated_sectors(summary)

    summary.attrs["pre_stage"] = pre_stage
    summary.attrs["target_stage"] = target_stage
    summary.attrs["image_group"] = image_group or frame["image_group"].mode().iloc[0]
    summary.attrs["transition"] = frame["transition"].mode().iloc[0]
    return summary


def attach_pre_colors(summary_df: pd.DataFrame, *, cmap: np.ndarray | None = None) -> pd.DataFrame:
    cmap = build_transition_cmap() if cmap is None else cmap
    summary_df = summary_df.copy()
    summary_df["PreColor"] = list(
        _rgb_from_values(
            summary_df["NO_Pre"],
            summary_df["O_Pre"],
            no_min=summary_df["NO_Pre"].min(),
            no_max=summary_df["NO_Pre"].max(),
            o_min=summary_df["O_Pre"].min(),
            o_max=summary_df["O_Pre"].max(),
            cmap=cmap,
        )
    )
    return summary_df


def compute_response_limits(*frames: pd.DataFrame, pad: float = 0.4) -> list[float]:
    values: list[np.ndarray] = []
    for frame in frames:
        values.extend(
            [
                frame["NO_Pre"].to_numpy(dtype=float),
                frame["O_Pre"].to_numpy(dtype=float),
                frame["NO_Target"].to_numpy(dtype=float),
                frame["O_Target"].to_numpy(dtype=float),
            ]
        )

    all_values = np.concatenate(values)
    return [float(np.nanmin(all_values) - pad), float(np.nanmax(all_values) + pad)]


def compute_shift_limits(
    *frames: pd.DataFrame,
    pad_ratio: float = 0.12,
    fallback_extent: float = 0.5,
) -> list[float]:
    values: list[np.ndarray] = []
    for frame in frames:
        values.extend([frame["dNO"].to_numpy(dtype=float), frame["dO"].to_numpy(dtype=float)])

    extent = np.nanmax(np.abs(np.concatenate(values)))
    if not np.isfinite(extent) or extent == 0:
        extent = fallback_extent
    else:
        extent *= 1.0 + pad_ratio
    return [-float(extent), float(extent)]


def sector_labels_with_counts(summary_df: pd.DataFrame) -> dict[str, str]:
    total = max(len(summary_df), 1)
    counts = (
        summary_df["RotatedSector"]
        .value_counts(sort=False)
        .reindex(ROTATED_SECTOR_ORDER)
        .fillna(0)
        .astype(int)
    )
    return {
        sector: f"{sector} (n={count}, {100.0 * count / total:.1f}%)"
        for sector, count in counts.items()
    }


def sector_mean_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    sector_means = (
        summary_df.groupby("RotatedSector", observed=True, as_index=False)
        .agg(
            NO_Pre=("NO_Pre", "mean"),
            O_Pre=("O_Pre", "mean"),
            NO_Target=("NO_Target", "mean"),
            O_Target=("O_Target", "mean"),
            dNO=("dNO", "mean"),
            dO=("dO", "mean"),
            Count=("neuron_idx", "size"),
        )
    )
    sector_means["MeanAngle"] = np.arctan2(sector_means["dO"], sector_means["dNO"])
    return sector_means


def _stack_pre_colors(summary_df: pd.DataFrame) -> np.ndarray:
    return np.vstack(summary_df["PreColor"].to_numpy())


def _draw_individual_vectors(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    *,
    alpha: float,
    width: float,
) -> None:
    ax.quiver(
        summary_df["NO_Pre"],
        summary_df["O_Pre"],
        summary_df["dNO"],
        summary_df["dO"],
        color=_stack_pre_colors(summary_df),
        angles="xy",
        scale_units="xy",
        scale=1,
        width=width,
        alpha=alpha,
        zorder=2,
    )


def _draw_arrow(
    ax: plt.Axes,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    *,
    color: str,
    linewidth: float,
    mutation_scale: float,
    zorder: int = 4,
) -> None:
    arrow = FancyArrowPatch(
        start_xy,
        end_xy,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        alpha=0.98,
        shrinkA=0.0,
        shrinkB=0.0,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def _draw_diagonal(ax: plt.Axes, lims: list[float]) -> None:
    ax.plot(lims, lims, "--", color="0.75", linewidth=1.0, zorder=0)


def _draw_origin_guides(ax: plt.Axes) -> None:
    ax.axhline(0.0, color="0.82", linewidth=1.0, zorder=0)
    ax.axvline(0.0, color="0.82", linewidth=1.0, zorder=0)


def _draw_rotated_guides(ax: plt.Axes, lims: list[float]) -> None:
    extent = float(max(abs(lims[0]), abs(lims[1])))
    guide = np.array([-extent, extent], dtype=float)
    ax.plot(guide, guide, "--", color="0.82", linewidth=1.0, zorder=0)
    ax.plot(guide, -guide, "--", color="0.82", linewidth=1.0, zorder=0)


def _legend_handles(labels: dict[str, str], *, linewidth: float) -> list[Line2D]:
    handles = []
    for sector in ROTATED_SECTOR_ORDER:
        handles.append(
            Line2D(
                [0],
                [0],
                color=ROTATED_SECTOR_PALETTE[sector],
                linewidth=linewidth,
                marker="o",
                markersize=7,
                label=labels[sector],
            )
        )
    return handles


def plot_mean_transition_summary(
    summary_df: pd.DataFrame,
    *,
    title: str,
    start_label: str | None = None,
    end_label: str | None = None,
    response_lims: list[float] | None = None,
    shift_lims: list[float] | None = None,
    style: dict | None = None,
) -> plt.Figure:
    summary_df = attach_pre_colors(summary_df)
    style = resolve_plot_style(style)
    start_label = start_label or summary_df.attrs.get("pre_stage", "Pre")
    end_label = end_label or summary_df.attrs.get("target_stage", "Target")
    response_lims = compute_response_limits(summary_df) if response_lims is None else response_lims
    shift_lims = compute_shift_limits(summary_df) if shift_lims is None else shift_lims

    sector_means = sector_mean_table(summary_df)
    sector_labels = sector_labels_with_counts(summary_df)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.2), sharex=False, sharey=False)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.scatter(
        summary_df["NO_Pre"],
        summary_df["O_Pre"],
        s=style["point_size"],
        c=_stack_pre_colors(summary_df),
        alpha=style["pre_point_alpha"],
        edgecolors="none",
        zorder=1,
    )
    _draw_individual_vectors(
        ax,
        summary_df,
        alpha=style["pre_vector_alpha"],
        width=style["individual_vector_width"],
    )
    _draw_diagonal(ax, response_lims)
    ax.set_xlim(response_lims)
    ax.set_ylim(response_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{start_label} by pre color")
    ax.set_xlabel("NO")
    ax.set_ylabel("O")

    ax = axes[0, 1]
    ax.scatter(
        summary_df["NO_Target"],
        summary_df["O_Target"],
        s=style["point_size"],
        c=_stack_pre_colors(summary_df),
        alpha=style["target_point_alpha"],
        edgecolors="none",
        zorder=1,
    )
    _draw_individual_vectors(
        ax,
        summary_df,
        alpha=style["target_vector_alpha"],
        width=style["individual_vector_width"],
    )
    _draw_diagonal(ax, response_lims)
    ax.set_xlim(response_lims)
    ax.set_ylim(response_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{end_label} by pre color")
    ax.set_xlabel("NO")
    ax.set_ylabel("O")

    ax = axes[0, 2]
    ax.scatter(
        summary_df["dNO"],
        summary_df["dO"],
        s=style["point_size"],
        c=_stack_pre_colors(summary_df),
        alpha=style["shift_point_alpha"],
        edgecolors="none",
        zorder=1,
    )
    _draw_origin_guides(ax)
    _draw_rotated_guides(ax, shift_lims)
    ax.set_xlim(shift_lims)
    ax.set_ylim(shift_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{end_label} - {start_label} by pre color")
    ax.set_xlabel("dNO")
    ax.set_ylabel("dO")

    ax = axes[1, 0]
    for sector in ROTATED_SECTOR_ORDER:
        sector_rows = summary_df.loc[summary_df["RotatedSector"] == sector]
        if sector_rows.empty:
            continue
        color = ROTATED_SECTOR_PALETTE[sector]
        ax.scatter(
            sector_rows["NO_Pre"],
            sector_rows["O_Pre"],
            s=style["point_size"],
            color=color,
            alpha=style["sector_point_alpha"],
            edgecolors="none",
        )
        mean_row = sector_means.loc[sector_means["RotatedSector"] == sector].iloc[0]
        _draw_arrow(
            ax,
            (float(mean_row["NO_Pre"]), float(mean_row["O_Pre"])),
            (float(mean_row["NO_Target"]), float(mean_row["O_Target"])),
            color=color,
            linewidth=style["mean_arrow_width"],
            mutation_scale=style["mean_arrow_mutation_scale"],
        )
    _draw_diagonal(ax, response_lims)
    ax.set_xlim(response_lims)
    ax.set_ylim(response_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{start_label} by rotated sector")
    ax.set_xlabel("NO")
    ax.set_ylabel("O")

    ax = axes[1, 1]
    for sector in ROTATED_SECTOR_ORDER:
        sector_rows = summary_df.loc[summary_df["RotatedSector"] == sector]
        if sector_rows.empty:
            continue
        color = ROTATED_SECTOR_PALETTE[sector]
        ax.scatter(
            sector_rows["NO_Target"],
            sector_rows["O_Target"],
            s=style["point_size"],
            color=color,
            alpha=style["sector_point_alpha"],
            edgecolors="none",
        )
        mean_row = sector_means.loc[sector_means["RotatedSector"] == sector].iloc[0]
        _draw_arrow(
            ax,
            (float(mean_row["NO_Pre"]), float(mean_row["O_Pre"])),
            (float(mean_row["NO_Target"]), float(mean_row["O_Target"])),
            color=color,
            linewidth=style["mean_arrow_width"],
            mutation_scale=style["mean_arrow_mutation_scale"],
        )
    _draw_diagonal(ax, response_lims)
    ax.set_xlim(response_lims)
    ax.set_ylim(response_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{end_label} by rotated sector")
    ax.set_xlabel("NO")
    ax.set_ylabel("O")

    ax = axes[1, 2]
    for sector in ROTATED_SECTOR_ORDER:
        sector_rows = summary_df.loc[summary_df["RotatedSector"] == sector]
        if sector_rows.empty:
            continue
        color = ROTATED_SECTOR_PALETTE[sector]
        ax.scatter(
            sector_rows["dNO"],
            sector_rows["dO"],
            s=style["point_size"],
            color=color,
            alpha=style["sector_point_alpha"],
            edgecolors="none",
        )
        mean_row = sector_means.loc[sector_means["RotatedSector"] == sector].iloc[0]
        _draw_arrow(
            ax,
            (0.0, 0.0),
            (float(mean_row["dNO"]), float(mean_row["dO"])),
            color=color,
            linewidth=style["mean_arrow_width"],
            mutation_scale=style["mean_arrow_mutation_scale"],
        )
    _draw_origin_guides(ax)
    _draw_rotated_guides(ax, shift_lims)
    ax.set_xlim(shift_lims)
    ax.set_ylim(shift_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{end_label} - {start_label} by rotated sector")
    ax.set_xlabel("dNO")
    ax.set_ylabel("dO")
    ax.legend(
        handles=_legend_handles(sector_labels, linewidth=style["mean_arrow_width"]),
        frameon=False,
        loc="best",
    )

    fig.text(
        0.5,
        0.03,
        "Rotated sectors are defined by dO = +/- dNO.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="0.35",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig
