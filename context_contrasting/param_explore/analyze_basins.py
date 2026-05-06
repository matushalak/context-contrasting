from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from context_contrasting.param_explore.common import (
    CONTEXT_LABELS,
    CONTEXT_MARKERS,
    ExplorationSettings,
    PARAMETER_ORDER,
    TRANSITION_COLORS,
    TRANSITION_LABEL_ORDER,
    reference_transition_table,
)
from context_contrasting.param_explore.plotting import _plot_kde

IMAGE_TYPES = ("familiar", "novel")
FF_FB_SUBCLASS_ORDER = [
    "FF_FB_broad",
    "FF_FB_narrow_familiar",
    "FF_FB_narrow_novel",
    "other",
]
FF_FB_SUBCLASS_COLORS = {
    "FF_FB_broad": "#ff4d4d",
    "FF_FB_narrow_familiar": "#ffb000",
    "FF_FB_narrow_novel": "#00c2d1",
    "other": "#b7b7b7",
}


@dataclass(frozen=True)
class BasinSettings:
    knn_k: int = 24
    purity_threshold: float = 0.65
    min_component_size: int = 20


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _observed_labels(frame: pd.DataFrame, label_col: str) -> list[str]:
    observed = set(frame[label_col].dropna().unique().tolist())
    return [label for label in TRANSITION_LABEL_ORDER if label in observed]


def _sample_group(subset: pd.DataFrame, limit: int = 220) -> pd.DataFrame:
    if len(subset) <= limit:
        return subset
    return subset.sample(n=limit, random_state=0)


def _transition_legend_handles(labels: list[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=TRANSITION_COLORS[label],
            markeredgecolor="none",
            label=label,
        )
        for label in labels
    ]


def _context_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markersize=7,
            markerfacecolor="black",
            markeredgecolor="black",
            label=CONTEXT_LABELS[mode],
        )
        for mode, marker in CONTEXT_MARKERS.items()
    ]


def _log_parameter_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.log10(frame[PARAMETER_ORDER].to_numpy(dtype=float))


def _transition_columns(image_type: str) -> tuple[str, str, str]:
    return (
        f"{image_type}_transition_label",
        f"{image_type}_transition_point_x",
        f"{image_type}_transition_point_y",
    )


def _state_category(state: float, responsive: bool) -> str:
    if not responsive:
        return "unresponsive"
    if state > 0:
        return "FF"
    if state < 0:
        return "FB"
    return "unresponsive"


def _joint_ff_fb_subclass(row: pd.Series) -> str:
    familiar = row["familiar_transition_label"]
    novel = row["novel_transition_label"]
    if familiar == "FF -> FB" and novel == "FF -> FF":
        return "FF_FB_broad"
    if familiar == "FF -> FB" and novel == "unresponsive -> FB":
        return "FF_FB_narrow_familiar"
    if familiar == "unresponsive -> FB" and novel == "FF -> FF":
        return "FF_FB_narrow_novel"
    return "other"


def _initial_state_ff_fb_subclass(row: pd.Series) -> str:
    familiar_naive = _state_category(
        float(row["familiar_naive_state"]),
        bool(row["familiar_naive_responsive"]),
    )
    novel_naive = _state_category(
        float(row["novel_naive_state"]),
        bool(row["novel_naive_responsive"]),
    )
    if familiar_naive == "FF" and novel_naive == "FF":
        return "FF_FB_broad"
    if familiar_naive == "FF" and novel_naive == "unresponsive":
        return "FF_FB_narrow_familiar"
    if familiar_naive == "unresponsive" and novel_naive == "FF":
        return "FF_FB_narrow_novel"
    return "other"


def _knn_structure(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_points = len(matrix)
    if n_points == 0:
        empty_i = np.empty((0, 0), dtype=int)
        empty_f = np.empty((0, 0), dtype=float)
        empty_g = coo_matrix((0, 0)).tocsr()
        return empty_i, empty_f, empty_g

    n_neighbors = min(k + 1, n_points)
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    model.fit(matrix)
    distances, indices = model.kneighbors(matrix)
    neighbor_indices = indices[:, 1:]
    neighbor_distances = distances[:, 1:]

    rows = np.repeat(np.arange(n_points), neighbor_indices.shape[1])
    cols = neighbor_indices.reshape(-1)
    data = np.ones(len(rows), dtype=float)
    graph = coo_matrix((data, (rows, cols)), shape=(n_points, n_points)).tocsr()
    graph = graph.maximum(graph.transpose())
    return neighbor_indices, neighbor_distances, graph


def _local_purity(labels: np.ndarray, neighbor_indices: np.ndarray) -> np.ndarray:
    if len(labels) == 0:
        return np.empty((0,), dtype=float)
    if neighbor_indices.shape[1] == 0:
        return np.ones((len(labels),), dtype=float)
    return (labels[neighbor_indices] == labels[:, None]).mean(axis=1)


def _component_ids(graph, mask: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return np.full((len(mask),), -1, dtype=int), {}
    subgraph = graph[indices][:, indices]
    n_components, labels = connected_components(subgraph, directed=False, return_labels=True)
    component_ids = np.full((len(mask),), -1, dtype=int)
    component_ids[indices] = labels
    sizes = {
        component_idx: int((labels == component_idx).sum())
        for component_idx in range(n_components)
    }
    return component_ids, sizes


def _stable_basin_assignments(
    frame: pd.DataFrame,
    *,
    image_type: str,
    settings: BasinSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_col, x_col, y_col = _transition_columns(image_type)
    basin_frame = frame.copy().reset_index(drop=True)
    labels = basin_frame[label_col].to_numpy(dtype=object)
    log_matrix = _log_parameter_matrix(basin_frame)
    neighbor_indices, _, graph = _knn_structure(log_matrix, settings.knn_k)
    purity = _local_purity(labels, neighbor_indices)
    basin_frame[f"{image_type}_local_purity"] = purity
    basin_frame[f"{image_type}_stable_core"] = False
    basin_frame[f"{image_type}_raw_component_id"] = -1
    basin_frame[f"{image_type}_raw_component_size"] = 0
    basin_frame[f"{image_type}_basin_id"] = ""
    basin_frame[f"{image_type}_basin_size"] = 0

    raw_component_sizes_by_row = np.zeros(len(basin_frame), dtype=int)
    basin_rows: list[dict[str, Any]] = []

    for transition_label in _observed_labels(basin_frame, label_col):
        label_mask = basin_frame[label_col].eq(transition_label).to_numpy()
        raw_component_ids, raw_sizes = _component_ids(graph, label_mask)
        basin_frame.loc[label_mask, f"{image_type}_raw_component_id"] = raw_component_ids[label_mask]
        for component_idx, size in raw_sizes.items():
            raw_component_sizes_by_row[(raw_component_ids == component_idx) & label_mask] = size
        basin_frame.loc[label_mask, f"{image_type}_raw_component_size"] = raw_component_sizes_by_row[label_mask]

        stable_mask = label_mask & (purity >= settings.purity_threshold)
        stable_component_ids, stable_sizes = _component_ids(graph, stable_mask)
        if not stable_sizes:
            continue

        rank = 0
        for component_idx, size in sorted(stable_sizes.items(), key=lambda item: item[1], reverse=True):
            if size < settings.min_component_size:
                continue
            rank += 1
            member_mask = stable_component_ids == component_idx
            basin_id = f"{image_type}|{basin_frame['context_mode'].iloc[0]}|{transition_label}|{rank:02d}"
            basin_frame.loc[member_mask, f"{image_type}_stable_core"] = True
            basin_frame.loc[member_mask, f"{image_type}_basin_id"] = basin_id
            basin_frame.loc[member_mask, f"{image_type}_basin_size"] = int(size)

            member_rows = basin_frame.loc[member_mask].copy()
            row: dict[str, Any] = {
                "image_type": image_type,
                "context_mode": str(basin_frame["context_mode"].iloc[0]),
                "context_label": str(basin_frame["context_label"].iloc[0]),
                "transition_label": transition_label,
                "basin_id": basin_id,
                "n_core_points": int(size),
                "raw_component_id": int(member_rows[f"{image_type}_raw_component_id"].mode().iloc[0]),
                "raw_component_size": int(member_rows[f"{image_type}_raw_component_size"].mode().iloc[0]),
                "purity_min": float(member_rows[f"{image_type}_local_purity"].min()),
                "purity_median": float(member_rows[f"{image_type}_local_purity"].median()),
                "naive_state_median": float(member_rows[x_col].median()),
                "expert_state_median": float(member_rows[y_col].median()),
                "naive_state_q10": float(member_rows[x_col].quantile(0.10)),
                "naive_state_q90": float(member_rows[x_col].quantile(0.90)),
                "expert_state_q10": float(member_rows[y_col].quantile(0.10)),
                "expert_state_q90": float(member_rows[y_col].quantile(0.90)),
            }
            for param_name in PARAMETER_ORDER:
                values = member_rows[param_name]
                row[f"{param_name}_q10"] = float(values.quantile(0.10))
                row[f"{param_name}_median"] = float(values.median())
                row[f"{param_name}_q90"] = float(values.quantile(0.90))
            basin_rows.append(row)

    edge_rows: list[dict[str, Any]] = []
    if len(basin_frame) > 1:
        rows, cols = graph.nonzero()
        unique_mask = rows < cols
        rows = rows[unique_mask]
        cols = cols[unique_mask]
        for left_idx, right_idx in zip(rows.tolist(), cols.tolist(), strict=True):
            left_label = str(labels[left_idx])
            right_label = str(labels[right_idx])
            if left_label == right_label:
                continue
            left_sorted, right_sorted = sorted((left_label, right_label))
            distance = float(np.linalg.norm(log_matrix[left_idx] - log_matrix[right_idx]))
            edge_rows.append(
                {
                    "image_type": image_type,
                    "context_mode": str(basin_frame["context_mode"].iloc[0]),
                    "context_label": str(basin_frame["context_label"].iloc[0]),
                    "transition_label_a": left_sorted,
                    "transition_label_b": right_sorted,
                    "log10_param_distance": distance,
                    "mid_naive_state": float(basin_frame.iloc[[left_idx, right_idx]][x_col].mean()),
                    "mid_expert_state": float(basin_frame.iloc[[left_idx, right_idx]][y_col].mean()),
                    "left_purity": float(purity[left_idx]),
                    "right_purity": float(purity[right_idx]),
                }
            )

    boundary_summary_rows: list[dict[str, Any]] = []
    edge_frame = pd.DataFrame(edge_rows)
    if not edge_frame.empty:
        for (label_a, label_b), group in edge_frame.groupby(["transition_label_a", "transition_label_b"]):
            boundary_summary_rows.append(
                {
                    "image_type": image_type,
                    "context_mode": str(basin_frame["context_mode"].iloc[0]),
                    "context_label": str(basin_frame["context_label"].iloc[0]),
                    "transition_label_a": label_a,
                    "transition_label_b": label_b,
                    "edge_count": int(len(group)),
                    "median_log10_param_distance": float(group["log10_param_distance"].median()),
                    "median_mid_naive_state": float(group["mid_naive_state"].median()),
                    "median_mid_expert_state": float(group["mid_expert_state"].median()),
                    "median_left_purity": float(group["left_purity"].median()),
                    "median_right_purity": float(group["right_purity"].median()),
                }
            )
    boundary_summary = pd.DataFrame(boundary_summary_rows)
    if not boundary_summary.empty:
        boundary_summary = boundary_summary.sort_values(
            ["edge_count", "median_log10_param_distance"],
            ascending=[False, True],
        ).reset_index(drop=True)

    basin_summary = pd.DataFrame(basin_rows)
    if not basin_summary.empty:
        basin_summary = basin_summary.sort_values(
            ["n_core_points", "purity_median"],
            ascending=[False, False],
        ).reset_index(drop=True)
    return basin_frame, basin_summary, boundary_summary


def _plot_stable_transition_space(
    frame: pd.DataFrame,
    *,
    image_type: str,
    output_path: Path,
) -> None:
    label_col, x_col, y_col = _transition_columns(image_type)
    stable_col = f"{image_type}_stable_core"
    subset = frame.loc[frame[stable_col]].copy()
    if subset.empty:
        return
    labels = _observed_labels(subset, label_col)

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    for mode, marker in CONTEXT_MARKERS.items():
        mode_subset = subset.loc[
            subset["receives_context_familiar"].eq(mode[0])
            & subset["receives_context_novel"].eq(mode[1])
        ]
        if mode_subset.empty:
            continue
        mode_subset = pd.concat(
            [_sample_group(group) for _, group in mode_subset.groupby(label_col)],
            ignore_index=True,
        )
        ax.scatter(
            mode_subset[x_col],
            mode_subset[y_col],
            s=18,
            alpha=0.6,
            c=mode_subset[label_col].map(TRANSITION_COLORS),
            marker=marker,
            edgecolors="none",
        )

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.25)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.25)
    ax.set_xlabel("naive_state")
    ax.set_ylabel("expert_state")
    ax.set_title(f"{image_type.capitalize()} stable basins")
    transition_legend = ax.legend(
        handles=_transition_legend_handles(labels),
        title="Transition tile",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )
    ax.add_artist(transition_legend)
    ax.legend(
        handles=_context_legend_handles(),
        title="receives_context",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
    )
    _save_fig(fig, output_path)


def _plot_stable_parameter_panels(
    frame: pd.DataFrame,
    *,
    image_type: str,
    output_path: Path,
) -> None:
    label_col, _, _ = _transition_columns(image_type)
    stable_col = f"{image_type}_stable_core"
    subset = frame.loc[frame[stable_col]].copy()
    if subset.empty:
        return
    labels = _observed_labels(subset, label_col)

    from context_contrasting.param_explore.common import PARAMETER_PLOT_GROUPS

    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.2))
    for ax, (x_col, y_col, title) in zip(axes.flat, PARAMETER_PLOT_GROUPS, strict=True):
        for mode, marker in CONTEXT_MARKERS.items():
            mode_subset = subset.loc[
                subset["receives_context_familiar"].eq(mode[0])
                & subset["receives_context_novel"].eq(mode[1])
            ]
            if mode_subset.empty:
                continue
            mode_subset = pd.concat(
                [_sample_group(group) for _, group in mode_subset.groupby(label_col)],
                ignore_index=True,
            )
            ax.scatter(
                mode_subset[x_col],
                mode_subset[y_col],
                s=10,
                alpha=0.28,
                c=mode_subset[label_col].map(TRANSITION_COLORS),
                marker=marker,
                edgecolors="none",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)

    transition_legend = axes[0, 2].legend(
        handles=_transition_legend_handles(labels),
        title=f"{image_type} tile",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )
    axes[0, 2].add_artist(transition_legend)
    axes[1, 2].legend(
        handles=_context_legend_handles(),
        title="receives_context",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
    )
    fig.suptitle(f"{image_type.capitalize()} stable basins over initial weights", y=1.02)
    _save_fig(fig, output_path)


def _plot_basin_pca(frame: pd.DataFrame, *, image_type: str, output_path: Path) -> None:
    stable_col = f"{image_type}_stable_core"
    label_col, _, _ = _transition_columns(image_type)
    subset = frame.loc[frame[stable_col]].copy()
    if subset.empty:
        return

    matrix = _log_parameter_matrix(subset)
    coords = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(matrix))
    subset["pca1"] = coords[:, 0]
    subset["pca2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    for transition_label in _observed_labels(subset, label_col):
        group = subset.loc[subset[label_col].eq(transition_label)]
        if group.empty:
            continue
        ax.scatter(
            group["pca1"],
            group["pca2"],
            s=18,
            alpha=0.55,
            color=TRANSITION_COLORS[transition_label],
            label=transition_label,
            edgecolors="none",
        )
    ax.set_xlabel("pca1")
    ax.set_ylabel("pca2")
    ax.set_title(f"{image_type.capitalize()} stable-basin PCA")
    ax.legend(frameon=False, fontsize=8, loc="best")
    _save_fig(fig, output_path)


def _group_legend_handles(label_order: list[str], color_map: dict[str, str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=color_map[label],
            markeredgecolor="none",
            label=label,
        )
        for label in label_order
    ]


def _plot_stable_group_transition_space(
    frame: pd.DataFrame,
    *,
    image_type: str,
    group_col: str,
    group_order: list[str],
    color_map: dict[str, str],
    output_path: Path,
    title: str,
    legend_title: str,
) -> None:
    label_col, x_col, y_col = _transition_columns(image_type)
    stable_col = f"{image_type}_stable_core"
    subset = frame.loc[frame[stable_col]].copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    labels = [label for label in group_order if label in set(subset[group_col].unique())]
    for label in labels:
        group = subset.loc[subset[group_col].eq(label)]
        if group.empty:
            continue
        _plot_kde(
            ax,
            group[x_col].to_numpy(dtype=float),
            group[y_col].to_numpy(dtype=float),
            color=color_map[label],
            log_axes=False,
        )

    for mode, marker in CONTEXT_MARKERS.items():
        mode_subset = subset.loc[
            subset["receives_context_familiar"].eq(mode[0])
            & subset["receives_context_novel"].eq(mode[1])
        ]
        if mode_subset.empty:
            continue
        mode_subset = pd.concat(
            [_sample_group(group) for _, group in mode_subset.groupby(group_col)],
            ignore_index=True,
        )
        ax.scatter(
            mode_subset[x_col],
            mode_subset[y_col],
            s=10,
            alpha=0.12,
            c=mode_subset[group_col].map(color_map),
            marker=marker,
            edgecolors="none",
        )

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.25)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.25)
    ax.set_xlabel("naive_state")
    ax.set_ylabel("expert_state")
    ax.set_title(title)
    group_legend = ax.legend(
        handles=_group_legend_handles(labels, color_map),
        title=legend_title,
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )
    ax.add_artist(group_legend)
    ax.legend(
        handles=_context_legend_handles(),
        title="receives_context",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
    )
    _save_fig(fig, output_path)


def _plot_stable_group_parameter_panels(
    frame: pd.DataFrame,
    *,
    image_type: str,
    group_col: str,
    group_order: list[str],
    color_map: dict[str, str],
    output_path: Path,
    title: str,
    legend_title: str,
) -> None:
    stable_col = f"{image_type}_stable_core"
    subset = frame.loc[frame[stable_col]].copy()
    if subset.empty:
        return

    from context_contrasting.param_explore.common import PARAMETER_PLOT_GROUPS

    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.2))
    labels = [label for label in group_order if label in set(subset[group_col].unique())]
    for ax, (x_col, y_col, panel_title) in zip(axes.flat, PARAMETER_PLOT_GROUPS, strict=True):
        for label in labels:
            group = subset.loc[subset[group_col].eq(label)]
            if group.empty:
                continue
            _plot_kde(
                ax,
                group[x_col].to_numpy(dtype=float),
                group[y_col].to_numpy(dtype=float),
                color=color_map[label],
                log_axes=True,
            )

        for mode, marker in CONTEXT_MARKERS.items():
            mode_subset = subset.loc[
                subset["receives_context_familiar"].eq(mode[0])
                & subset["receives_context_novel"].eq(mode[1])
            ]
            if mode_subset.empty:
                continue
            mode_subset = pd.concat(
                [_sample_group(group) for _, group in mode_subset.groupby(group_col)],
                ignore_index=True,
            )
            ax.scatter(
                mode_subset[x_col],
                mode_subset[y_col],
                s=6,
                alpha=0.08,
                c=mode_subset[group_col].map(color_map),
                marker=marker,
                edgecolors="none",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(panel_title)

    group_legend = axes[0, 2].legend(
        handles=_group_legend_handles(labels, color_map),
        title=legend_title,
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )
    axes[0, 2].add_artist(group_legend)
    axes[1, 2].legend(
        handles=_context_legend_handles(),
        title="receives_context",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
    )
    fig.suptitle(title, y=1.02)
    _save_fig(fig, output_path)


def _reference_tiles(settings: ExplorationSettings) -> pd.DataFrame:
    frame = reference_transition_table(settings)
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        for image_type in IMAGE_TYPES:
            label_col, x_col, y_col = _transition_columns(image_type)
            rows.append(
                {
                    "reference_name": row["reference_name"],
                    "image_type": image_type,
                    "context_mode": f"{int(row['receives_context_familiar'])}{int(row['receives_context_novel'])}",
                    "context_label": CONTEXT_LABELS[(bool(row["receives_context_familiar"]), bool(row["receives_context_novel"]))],
                    "transition_label": row[label_col],
                    "naive_state": float(row[x_col]),
                    "expert_state": float(row[y_col]),
                }
            )
    return pd.DataFrame(rows)


def _write_report(
    *,
    output_path: Path,
    source_dir: Path,
    basin_settings: BasinSettings,
    exploration_settings: dict[str, Any],
    basin_summary: pd.DataFrame,
    boundary_summary: pd.DataFrame,
) -> None:
    lines = [
        "Transition-plane basin analysis",
        "",
        f"Source samples: {source_dir}",
        "",
        "Definition",
        f"- Separate analysis per image type (`familiar`, `novel`) and per `receives_context` mode.",
        f"- Build a symmetric kNN graph in 12D `log10` weight space with `k={basin_settings.knn_k}`.",
        f"- Compute local purity = fraction of kNN neighbors with the same transition-plane tile.",
        f"- Stable core threshold: purity >= {basin_settings.purity_threshold:.2f}.",
        f"- Stable basin = connected component of stable-core points within one tile, size >= {basin_settings.min_component_size}.",
        "",
        "Sampling settings",
        f"- n_steps_per_phase = {exploration_settings['n_steps_per_phase']}",
        f"- n_trials = {exploration_settings['n_trials']}",
        f"- tail_window = {exploration_settings['tail_window']}",
        "",
        "Largest basins",
    ]

    if basin_summary.empty:
        lines.append("- No stable basins found.")
    else:
        top = basin_summary.head(16)
        for _, row in top.iterrows():
            lines.append(
                f"- {row['basin_id']}: {row['context_label']} | {row['image_type']} | "
                f"{row['transition_label']} | core={int(row['n_core_points'])} | "
                f"purity_median={row['purity_median']:.3f}"
            )

    lines.extend(["", "Strongest boundaries"])
    if boundary_summary.empty:
        lines.append("- No cross-tile neighborhood edges found.")
    else:
        top = boundary_summary.head(16)
        for _, row in top.iterrows():
            lines.append(
                f"- {row['context_label']} | {row['image_type']} | {row['transition_label_a']} vs "
                f"{row['transition_label_b']} | edges={int(row['edge_count'])} | "
                f"median_log10_distance={row['median_log10_param_distance']:.3f}"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _subclass_summary(
    frame: pd.DataFrame,
    *,
    group_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for image_type in IMAGE_TYPES:
        stable_col = f"{image_type}_stable_core"
        subset = frame.loc[frame[stable_col]].copy()
        if subset.empty:
            continue
        for (context_mode, group_label), group in subset.groupby(["context_mode", group_col], dropna=False):
            if group.empty:
                continue
            label_col, x_col, y_col = _transition_columns(image_type)
            rows.append(
                {
                    "image_type": image_type,
                    "context_mode": str(context_mode),
                    "context_label": str(group["context_label"].iloc[0]),
                    "subclass_label": str(group_label),
                    "n_stable_points": int(len(group)),
                    "median_naive_state": float(group[x_col].median()),
                    "median_expert_state": float(group[y_col].median()),
                    "dominant_transition_label": str(group[label_col].mode().iloc[0]),
                }
            )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["image_type", "context_mode", "n_stable_points"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
    return summary


def analyze_basins(
    *,
    results_dir: str | Path,
    basin_settings: BasinSettings,
) -> dict[str, pd.DataFrame]:
    source_dir = Path(results_dir)
    samples_path = source_dir / "samples.csv"
    if not samples_path.exists():
        raise FileNotFoundError(f"Expected samples at {samples_path}.")

    frame = pd.read_csv(samples_path)
    frame["ff_fb_transition_subclass"] = frame.apply(_joint_ff_fb_subclass, axis=1)
    frame["ff_fb_initial_subclass"] = frame.apply(_initial_state_ff_fb_subclass, axis=1)
    summary_path = source_dir / "summary.json"
    exploration_settings = json.loads(summary_path.read_text(encoding="utf-8"))["settings"]
    settings = ExplorationSettings(**exploration_settings)
    output_dir = source_dir / "basins"
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_frames: list[pd.DataFrame] = []
    basin_summaries: list[pd.DataFrame] = []
    boundary_summaries: list[pd.DataFrame] = []

    for context_mode in CONTEXT_LABELS:
        subset = frame.loc[
            frame["receives_context_familiar"].eq(context_mode[0])
            & frame["receives_context_novel"].eq(context_mode[1])
        ].copy()
        if subset.empty:
            continue
        annotated = subset.reset_index(drop=True)
        for image_type in IMAGE_TYPES:
            annotated, basin_summary, boundary_summary = _stable_basin_assignments(
                annotated,
                image_type=image_type,
                settings=basin_settings,
            )
            if not basin_summary.empty:
                basin_summaries.append(basin_summary)
            if not boundary_summary.empty:
                boundary_summaries.append(boundary_summary)
        annotated_frames.append(annotated)

    annotated_frame = pd.concat(annotated_frames, ignore_index=True)
    basin_summary = pd.concat(basin_summaries, ignore_index=True) if basin_summaries else pd.DataFrame()
    boundary_summary = pd.concat(boundary_summaries, ignore_index=True) if boundary_summaries else pd.DataFrame()

    reference_tiles = _reference_tiles(settings)
    annotated_frame.to_csv(output_dir / "samples_with_basins.csv", index=False)
    basin_summary.to_csv(output_dir / "basin_summary.csv", index=False)
    boundary_summary.to_csv(output_dir / "boundary_summary.csv", index=False)
    reference_tiles.to_csv(output_dir / "reference_tiles.csv", index=False)
    _subclass_summary(
        annotated_frame,
        group_col="ff_fb_initial_subclass",
    ).to_csv(output_dir / "ff_fb_initial_subclass_summary.csv", index=False)
    _subclass_summary(
        annotated_frame,
        group_col="ff_fb_transition_subclass",
    ).to_csv(output_dir / "ff_fb_transition_subclass_summary.csv", index=False)

    plots_dir = output_dir / "plots"
    for image_type in IMAGE_TYPES:
        _plot_stable_transition_space(
            annotated_frame,
            image_type=image_type,
            output_path=plots_dir / f"{image_type}_stable_transition_plane.png",
        )
        _plot_stable_parameter_panels(
            annotated_frame,
            image_type=image_type,
            output_path=plots_dir / f"{image_type}_stable_parameter_panels.png",
        )
        _plot_basin_pca(
            annotated_frame,
            image_type=image_type,
            output_path=plots_dir / f"{image_type}_stable_basin_pca.png",
        )
        _plot_stable_group_transition_space(
            annotated_frame,
            image_type=image_type,
            group_col="ff_fb_initial_subclass",
            group_order=FF_FB_SUBCLASS_ORDER,
            color_map=FF_FB_SUBCLASS_COLORS,
            output_path=plots_dir / f"{image_type}_stable_transition_plane_ff_fb_initial_subclasses.png",
            title=f"{image_type.capitalize()} stable basins by FF_FB initial subclass",
            legend_title="FF_FB initial subclass",
        )
        _plot_stable_group_parameter_panels(
            annotated_frame,
            image_type=image_type,
            group_col="ff_fb_initial_subclass",
            group_order=FF_FB_SUBCLASS_ORDER,
            color_map=FF_FB_SUBCLASS_COLORS,
            output_path=plots_dir / f"{image_type}_stable_parameter_panels_ff_fb_initial_subclasses.png",
            title=f"{image_type.capitalize()} stable basins over initial weights (FF_FB initial subclasses)",
            legend_title="FF_FB initial subclass",
        )
        _plot_stable_group_transition_space(
            annotated_frame,
            image_type=image_type,
            group_col="ff_fb_transition_subclass",
            group_order=FF_FB_SUBCLASS_ORDER,
            color_map=FF_FB_SUBCLASS_COLORS,
            output_path=plots_dir / f"{image_type}_stable_transition_plane_ff_fb_transition_subclasses.png",
            title=f"{image_type.capitalize()} stable basins by FF_FB transition subclass",
            legend_title="FF_FB transition subclass",
        )
        _plot_stable_group_parameter_panels(
            annotated_frame,
            image_type=image_type,
            group_col="ff_fb_transition_subclass",
            group_order=FF_FB_SUBCLASS_ORDER,
            color_map=FF_FB_SUBCLASS_COLORS,
            output_path=plots_dir / f"{image_type}_stable_parameter_panels_ff_fb_transition_subclasses.png",
            title=f"{image_type.capitalize()} stable basins over initial weights (FF_FB transition subclasses)",
            legend_title="FF_FB transition subclass",
        )

    _write_report(
        output_path=output_dir / "basin_report.md",
        source_dir=source_dir,
        basin_settings=basin_settings,
        exploration_settings=exploration_settings,
        basin_summary=basin_summary,
        boundary_summary=boundary_summary,
    )

    metadata = {
        "source_dir": str(source_dir),
        "basin_settings": basin_settings.__dict__,
        "exploration_settings": exploration_settings,
        "n_samples": int(len(frame)),
        "n_stable_points_familiar": int(annotated_frame["familiar_stable_core"].sum()),
        "n_stable_points_novel": int(annotated_frame["novel_stable_core"].sum()),
        "n_basins": int(len(basin_summary)),
    }
    (output_dir / "basin_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "annotated_frame": annotated_frame,
        "basin_summary": basin_summary,
        "boundary_summary": boundary_summary,
        "reference_tiles": reference_tiles,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find stable basins directly in transition-plane tiles using a kNN graph in log-weight space."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--knn-k", type=int, default=24)
    parser.add_argument("--purity-threshold", type=float, default=0.65)
    parser.add_argument("--min-component-size", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    basin_settings = BasinSettings(
        knn_k=args.knn_k,
        purity_threshold=args.purity_threshold,
        min_component_size=args.min_component_size,
    )
    outputs = analyze_basins(
        results_dir=args.results_dir,
        basin_settings=basin_settings,
    )
    print(
        outputs["basin_summary"][
            ["image_type", "context_label", "transition_label", "basin_id", "n_core_points"]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
