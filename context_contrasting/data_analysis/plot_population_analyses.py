from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_ORDER = [
    ("Pre", "Full"),
    ("Pre", "Occl"),
    ("Post", "Full"),
    ("Post", "Occl"),
]

CONDITION_LABELS = {
    ("Pre", "Full"): "Pre Full",
    ("Pre", "Occl"): "Pre Occl",
    ("Post", "Full"): "Post Full",
    ("Post", "Occl"): "Post Occl",
}

CONDITION_COLORS = {
    ("Pre", "Full"): "#202020",
    ("Pre", "Occl"): "#C33C2D",
    ("Post", "Full"): "#1F77B4",
    ("Post", "Occl"): "#7B3294",
}

SECTOR_ORDER = ["-NO axis", "+O axis", "+NO axis"]
SECTOR_SLUG = {
    "-NO axis": "minus_no_axis",
    "+O axis": "plus_o_axis",
    "+NO axis": "plus_no_axis",
}

ANALYSIS_DISPLAY = {
    "full_population": "full population",
    "subpopulation_minus_no_axis": "-NO axis subpopulation",
    "subpopulation_plus_o_axis": "+O axis subpopulation",
    "subpopulation_plus_no_axis": "+NO axis subpopulation",
}

CONDITION_ABBREV = {
    ("Pre", "Full"): "PreF",
    ("Pre", "Occl"): "PreO",
    ("Post", "Full"): "PostF",
    ("Post", "Occl"): "PostO",
}


def assign_rotated_sector(frame: pd.DataFrame, threshold: float = 0.0) -> pd.Series:
    angle = np.arctan2(frame["dO"].to_numpy(dtype=float), frame["dNO"].to_numpy(dtype=float))
    norm = np.hypot(frame["dNO"].to_numpy(dtype=float), frame["dO"].to_numpy(dtype=float))

    sector = np.full(len(frame), "+NO axis", dtype=object)
    sector[(angle >= np.pi / 4.0) & (angle < 3.0 * np.pi / 4.0)] = "+O axis"
    sector[(angle >= 3.0 * np.pi / 4.0) | (angle < -3.0 * np.pi / 4.0)] = "-NO axis"
    sector[(angle >= -3.0 * np.pi / 4.0) & (angle < -np.pi / 4.0)] = "-O axis"
    sector[norm <= threshold] = "small delta"
    return pd.Series(sector, index=frame.index, name="rotated_sector")


def build_sector_assignments(scalar_csv: Path, threshold: float) -> pd.DataFrame:
    raw = pd.read_csv(scalar_csv)
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
    stage_means = wide.groupby(["neuron_idx", "stage"], as_index=False)[["NO", "O"]].mean()
    summary = stage_means.pivot(index="neuron_idx", columns="stage", values=["NO", "O"])
    summary.columns = [f"{metric}_{stage}" for metric, stage in summary.columns]
    summary = summary.reset_index()
    summary["dNO"] = summary["NO_Post"] - summary["NO_Pre"]
    summary["dO"] = summary["O_Post"] - summary["O_Pre"]
    summary["dNorm"] = np.hypot(summary["dNO"], summary["dO"])
    summary["angle_rad"] = np.arctan2(summary["dO"], summary["dNO"])
    summary["rotated_sector"] = assign_rotated_sector(summary, threshold=threshold)
    return summary


def zscore_from_pre_full_baseline(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = frame.loc[
        frame["stage"].eq("Pre") & frame["image_type"].eq("Full") & (frame["time"] < 0)
    ]
    stats = (
        baseline.groupby("neuron_idx")["response"]
        .agg(baseline_mean="mean", baseline_std="std", baseline_samples="size")
        .reset_index()
    )
    bad = stats["baseline_std"].isna() | (stats["baseline_std"] <= 0)
    if bad.any():
        bad_neurons = stats.loc[bad, "neuron_idx"].tolist()
        raise ValueError(f"Cannot z-score neurons with zero/NaN baseline std: {bad_neurons}")

    z = frame.merge(stats, on="neuron_idx", how="left", validate="many_to_one")
    if z["baseline_std"].isna().any():
        raise ValueError("Some trace rows did not receive baseline statistics.")
    z["z_response"] = (z["response"] - z["baseline_mean"]) / z["baseline_std"]
    return z, stats


def unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Cannot normalize a zero or non-finite vector.")
    return vector / norm


def correlation_distance(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    denom = np.linalg.norm(a0) * np.linalg.norm(b0)
    if denom <= 0:
        return np.nan
    return float(1.0 - np.dot(a0, b0) / denom)


def correlation_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(centered, axis=1, keepdims=True)
    denom[denom == 0] = np.nan
    normalized = centered / denom
    corr = normalized @ normalized.T
    return 1.0 - corr


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - np.nanmean(a)
    b0 = b - np.nanmean(b)
    denom = np.linalg.norm(a0) * np.linalg.norm(b0)
    if denom <= 0:
        return np.nan
    return float(np.dot(a0, b0) / denom)


def build_window_vectors(
    z_trace: pd.DataFrame,
    neuron_ids: list[int],
    *,
    window_start: float,
    window_end: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    frame = z_trace.loc[
        z_trace["neuron_idx"].isin(neuron_ids)
        & (z_trace["time"] > window_start)
        & (z_trace["time"] < window_end)
    ].copy()
    index_cols = [
        "image_group",
        "image_idx_original",
        "image_idx_within_group",
        "stage",
        "image_type",
    ]
    meaned = frame.groupby(index_cols + ["neuron_idx"], as_index=False)["z_response"].mean()
    wide = meaned.pivot(index=index_cols, columns="neuron_idx", values="z_response")
    wide = wide.loc[:, neuron_ids]
    if wide.isna().any().any():
        raise ValueError("Stimulus-window population matrix contains missing values.")
    meta = wide.index.to_frame(index=False)
    order_key = {
        (stage, image_type): i
        for i, (stage, image_type) in enumerate(CONDITION_ORDER)
    }
    meta["_condition_order"] = [order_key[(r.stage, r.image_type)] for r in meta.itertuples()]
    meta["_row"] = np.arange(len(meta))
    meta = meta.sort_values(["_condition_order", "image_idx_original"]).reset_index(drop=True)
    row_order = meta.pop("_row").to_numpy(dtype=int)
    meta = meta.drop(columns=["_condition_order"])
    return meta, wide.to_numpy(dtype=float)[row_order]


def build_time_vectors(z_trace: pd.DataFrame, neuron_ids: list[int]) -> tuple[pd.DataFrame, np.ndarray]:
    frame = z_trace.loc[z_trace["neuron_idx"].isin(neuron_ids)].copy()
    index_cols = [
        "image_group",
        "image_idx_original",
        "image_idx_within_group",
        "stage",
        "image_type",
        "time",
    ]
    wide = frame.pivot(index=index_cols, columns="neuron_idx", values="z_response")
    wide = wide.loc[:, neuron_ids]
    if wide.isna().any().any():
        raise ValueError("Time-resolved population matrix contains missing values.")
    meta = wide.index.to_frame(index=False)
    return meta, wide.to_numpy(dtype=float)


def analysis_neuron_sets(
    z_trace: pd.DataFrame,
    sectors: pd.DataFrame,
) -> dict[str, list[int]]:
    sets = {"full_population": sorted(z_trace["neuron_idx"].astype(int).unique().tolist())}
    for sector in SECTOR_ORDER:
        sets[f"subpopulation_{SECTOR_SLUG[sector]}"] = sorted(
            sectors.loc[sectors["rotated_sector"].eq(sector), "neuron_idx"].astype(int).tolist()
        )
    return sets


def compute_rsa(
    analysis_name: str,
    meta: pd.DataFrame,
    matrix: np.ndarray,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dist = correlation_distance_matrix(matrix)
    labels = [
        f"{CONDITION_ABBREV[(row.stage, row.image_type)]}{int(row.image_idx_original)}"
        for row in meta.itertuples()
    ]
    wide = pd.DataFrame(dist, index=labels, columns=labels)
    wide.insert(0, "label", labels)
    wide.to_csv(output_dir / f"{analysis_name}_rsa_distance_matrix.csv", index=False)

    rows = []
    for i, j in combinations(range(len(meta)), 2):
        left = meta.iloc[i]
        right = meta.iloc[j]
        rows.append(
            {
                "analysis": analysis_name,
                "left_stage": left.stage,
                "left_image_type": left.image_type,
                "left_image_idx": int(left.image_idx_original),
                "right_stage": right.stage,
                "right_image_type": right.image_type,
                "right_image_idx": int(right.image_idx_original),
                "correlation_distance": dist[i, j],
            }
        )
    return wide, pd.DataFrame(rows)


def compute_same_different(
    analysis_name: str,
    meta: pd.DataFrame,
    matrix: np.ndarray,
) -> pd.DataFrame:
    rows = []
    vector_lookup = {
        (row.stage, row.image_type, int(row.image_idx_original)): matrix[i]
        for i, row in enumerate(meta.itertuples())
    }
    group_lookup = {
        int(row.image_idx_original): row.image_group
        for row in meta.itertuples()
    }
    groups = {
        "all": sorted(group_lookup),
        "familiar": sorted([idx for idx, group in group_lookup.items() if group == "familiar"]),
        "novel": sorted([idx for idx, group in group_lookup.items() if group == "novel"]),
    }
    for stage in ["Pre", "Post"]:
        for group_name, image_indices in groups.items():
            if len(image_indices) < 2:
                continue
            for image_idx in image_indices:
                full_vec = vector_lookup[(stage, "Full", image_idx)]
                same = correlation_distance(full_vec, vector_lookup[(stage, "Occl", image_idx)])
                different = [
                    correlation_distance(full_vec, vector_lookup[(stage, "Occl", other_idx)])
                    for other_idx in image_indices
                    if other_idx != image_idx
                ]
                rows.append(
                    {
                        "analysis": analysis_name,
                        "stage": stage,
                        "image_group": group_name,
                        "image_idx_original": image_idx,
                        "same_distance": same,
                        "different_mean_distance": float(np.nanmean(different)),
                        "different_median_distance": float(np.nanmedian(different)),
                        "different_minus_same": float(np.nanmean(different) - same),
                        "n_different": len(different),
                    }
                )
    return pd.DataFrame(rows)


def compute_rsa_followups(
    analysis_name: str,
    meta: pd.DataFrame,
    matrix: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vector_lookup = {
        (row.stage, row.image_type, int(row.image_idx_original)): matrix[i]
        for i, row in enumerate(meta.itertuples())
    }
    group_lookup = {
        int(row.image_idx_original): row.image_group
        for row in meta.itertuples()
    }
    image_indices = sorted(group_lookup)

    distance_rows = []
    for image_idx in image_indices:
        entries = [
            (
                "occlusion_gap_pre",
                "Pre Full_i vs Pre Occl_i",
                vector_lookup[("Pre", "Full", image_idx)],
                vector_lookup[("Pre", "Occl", image_idx)],
            ),
            (
                "occlusion_gap_post",
                "Post Full_i vs Post Occl_i",
                vector_lookup[("Post", "Full", image_idx)],
                vector_lookup[("Post", "Occl", image_idx)],
            ),
            (
                "learning_shift_full",
                "Pre Full_i vs Post Full_i",
                vector_lookup[("Pre", "Full", image_idx)],
                vector_lookup[("Post", "Full", image_idx)],
            ),
            (
                "learning_shift_occl",
                "Pre Occl_i vs Post Occl_i",
                vector_lookup[("Pre", "Occl", image_idx)],
                vector_lookup[("Post", "Occl", image_idx)],
            ),
        ]
        for metric, label, left, right in entries:
            distance_rows.append(
                {
                    "analysis": analysis_name,
                    "metric": metric,
                    "label": label,
                    "image_idx_original": image_idx,
                    "image_group": group_lookup[image_idx],
                    "correlation_distance": correlation_distance(left, right),
                }
            )

    identity_rows = []
    groups = {
        "all": image_indices,
        "familiar": [idx for idx in image_indices if group_lookup[idx] == "familiar"],
        "novel": [idx for idx in image_indices if group_lookup[idx] == "novel"],
    }
    for stage in ["Pre", "Post"]:
        for group_name, indices in groups.items():
            if len(indices) < 2:
                continue
            for image_idx in indices:
                matched = correlation_distance(
                    vector_lookup[(stage, "Full", image_idx)],
                    vector_lookup[(stage, "Occl", image_idx)],
                )
                unmatched = [
                    correlation_distance(
                        vector_lookup[(stage, "Full", image_idx)],
                        vector_lookup[(stage, "Occl", other_idx)],
                    )
                    for other_idx in indices
                    if other_idx != image_idx
                ]
                identity_rows.append(
                    {
                        "analysis": analysis_name,
                        "stage": stage,
                        "image_group": group_name,
                        "image_idx_original": image_idx,
                        "matched_full_occl_distance": matched,
                        "unmatched_full_occl_mean_distance": float(np.nanmean(unmatched)),
                        "unmatched_minus_matched": float(np.nanmean(unmatched) - matched),
                    }
                )

    geometry_rows = []
    conditions = [
        ("Pre", "Full"),
        ("Pre", "Occl"),
        ("Post", "Full"),
        ("Post", "Occl"),
    ]
    reference_condition = ("Pre", "Full")
    for group_name, indices in groups.items():
        if len(indices) < 3:
            continue

        def rdm_vector(stage: str, image_type: str) -> np.ndarray:
            vectors = np.stack([vector_lookup[(stage, image_type, idx)] for idx in indices])
            dist = correlation_distance_matrix(vectors)
            tri = np.triu_indices(len(indices), k=1)
            return dist[tri]

        reference = rdm_vector(*reference_condition)
        for stage, image_type in conditions:
            candidate = rdm_vector(stage, image_type)
            geometry_rows.append(
                {
                    "analysis": analysis_name,
                    "image_group": group_name,
                    "reference_condition": CONDITION_LABELS[reference_condition],
                    "comparison_condition": CONDITION_LABELS[(stage, image_type)],
                    "rdm_pearson_r": pearson_correlation(reference, candidate),
                    "n_pairwise_image_distances": len(reference),
                }
            )

    return (
        pd.DataFrame(distance_rows),
        pd.DataFrame(identity_rows),
        pd.DataFrame(geometry_rows),
    )


def compute_supervised_axes(
    analysis_name: str,
    window_meta: pd.DataFrame,
    window_matrix: np.ndarray,
    time_meta: pd.DataFrame,
    time_matrix: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def condition_mean(stage: str, image_type: str, image_idx: int | None = None) -> np.ndarray:
        mask = window_meta["stage"].eq(stage) & window_meta["image_type"].eq(image_type)
        if image_idx is not None:
            mask &= window_meta["image_idx_original"].eq(image_idx)
        return window_matrix[mask.to_numpy()].mean(axis=0)

    pre_full_mean = condition_mean("Pre", "Full")
    pre_occl_mean = condition_mean("Pre", "Occl")
    post_full_mean = condition_mean("Post", "Full")
    fullness_axis = unit_vector(pre_full_mean - pre_occl_mean)
    learning_axis = unit_vector(post_full_mean - pre_full_mean)
    origin = pre_full_mean

    projections = time_meta.copy()
    projections.insert(0, "analysis", analysis_name)
    centered_time = time_matrix - origin
    projections["fullness_axis_score"] = centered_time @ fullness_axis
    projections["learning_axis_score"] = centered_time @ learning_axis

    image_rows = []
    image_indices = sorted(window_meta["image_idx_original"].unique().tolist())
    for image_idx in image_indices:
        image_mask = window_meta["image_idx_original"].eq(image_idx)
        other_pre_full = window_matrix[
            window_meta["stage"].eq("Pre").to_numpy()
            & window_meta["image_type"].eq("Full").to_numpy()
            & (~image_mask.to_numpy())
        ].mean(axis=0)
        image_axis = unit_vector(condition_mean("Pre", "Full", image_idx) - other_pre_full)
        for row_idx, row in window_meta.iterrows():
            if int(row.image_idx_original) != int(image_idx):
                continue
            score = float((window_matrix[row_idx] - other_pre_full) @ image_axis)
            image_rows.append(
                {
                    "analysis": analysis_name,
                    "image_idx_original": int(row.image_idx_original),
                    "image_group": row.image_group,
                    "stage": row.stage,
                    "image_type": row.image_type,
                    "condition": CONDITION_LABELS[(row.stage, row.image_type)],
                    "image_axis_score": score,
                }
            )

    axis_rows = pd.DataFrame(
        [
            {
                "analysis": analysis_name,
                "axis": "fullness",
                "definition": "mean Pre Full minus mean Pre Occl, stimulus window",
                "axis_norm_before_normalization": float(np.linalg.norm(pre_full_mean - pre_occl_mean)),
                "dot_with_other_axis": float(np.dot(fullness_axis, learning_axis)),
            },
            {
                "analysis": analysis_name,
                "axis": "learning",
                "definition": "mean Post Full minus mean Pre Full, stimulus window",
                "axis_norm_before_normalization": float(np.linalg.norm(post_full_mean - pre_full_mean)),
                "dot_with_other_axis": float(np.dot(learning_axis, fullness_axis)),
            },
        ]
    )
    return projections, pd.DataFrame(image_rows), axis_rows


def plot_rsa_heatmaps(
    rsa_wide_by_analysis: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    vmax = max(
        frame.drop(columns=["label"]).to_numpy(dtype=float).max()
        for frame in rsa_wide_by_analysis.values()
    )
    for ax, (analysis_name, frame) in zip(axes.ravel(), rsa_wide_by_analysis.items()):
        labels = frame["label"].tolist()
        matrix = frame.drop(columns=["label"]).to_numpy(dtype=float)
        im = ax.imshow(matrix, cmap="magma_r", vmin=0, vmax=vmax)
        ax.set_title(ANALYSIS_DISPLAY.get(analysis_name, analysis_name.replace("_", " ")), fontsize=11)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        for boundary in [6, 12, 18]:
            ax.axhline(boundary - 0.5, color="white", lw=1.0)
            ax.axvline(boundary - 0.5, color="white", lw=1.0)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="correlation distance")
    fig.suptitle("RSA distance matrices: stimulus-window population vectors", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_separability_by_image(separability: pd.DataFrame, output_path: Path) -> None:
    analyses = separability["analysis"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(analyses), 1, figsize=(10.5, 3.0 * len(analyses)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, analysis in zip(axes, analyses):
        sub = separability.loc[
            separability["analysis"].eq(analysis) & separability["image_group"].eq("all")
        ]
        for stage, color in [("Pre", "#696969"), ("Post", "#1F77B4")]:
            part = sub.loc[sub["stage"].eq(stage)].sort_values("image_idx_original")
            ax.plot(
                part["image_idx_original"],
                part["different_minus_same"],
                marker="o",
                lw=2,
                color=color,
                label=stage,
            )
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_ylabel("different - same")
        ax.set_title(ANALYSIS_DISPLAY.get(analysis, analysis.replace("_", " ")))
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("image")
    fig.suptitle("Full/Occl same-image separability by image", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_separability_summary(separability: pd.DataFrame, output_path: Path) -> None:
    analyses = separability["analysis"].drop_duplicates().tolist()
    groups = ["all", "familiar", "novel"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, constrained_layout=True)
    for ax, analysis in zip(axes.ravel(), analyses):
        sub = (
            separability.loc[separability["analysis"].eq(analysis)]
            .groupby(["stage", "image_group"], as_index=False)["different_minus_same"]
            .mean()
        )
        x = np.arange(len(groups))
        width = 0.34
        for offset, stage, color in [(-width / 2, "Pre", "#696969"), (width / 2, "Post", "#1F77B4")]:
            vals = [
                sub.loc[sub["stage"].eq(stage) & sub["image_group"].eq(group), "different_minus_same"].mean()
                for group in groups
            ]
            ax.bar(x + offset, vals, width=width, label=stage, color=color)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_title(ANALYSIS_DISPLAY.get(analysis, analysis.replace("_", " ")))
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("mean different - same")
    axes[1, 0].set_ylabel("mean different - same")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Same-image Full/Occl separability", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_rsa_distance_followups(distances: pd.DataFrame, output_path: Path) -> None:
    analyses = distances["analysis"].drop_duplicates().tolist()
    metric_order = [
        ("occlusion_gap_pre", "Pre F/O"),
        ("occlusion_gap_post", "Post F/O"),
        ("learning_shift_full", "Full learn"),
        ("learning_shift_occl", "Occl learn"),
    ]
    metric_colors = {
        "occlusion_gap_pre": "#696969",
        "occlusion_gap_post": "#1F77B4",
        "learning_shift_full": "#2CA02C",
        "learning_shift_occl": "#9467BD",
    }
    fig, axes = plt.subplots(len(analyses), 1, figsize=(10.8, 3.0 * len(analyses)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, analysis in zip(axes, analyses):
        sub = distances.loc[distances["analysis"].eq(analysis)]
        for metric, label in metric_order:
            part = sub.loc[sub["metric"].eq(metric)].sort_values("image_idx_original")
            ax.plot(
                part["image_idx_original"],
                part["correlation_distance"],
                marker="o",
                lw=2,
                color=metric_colors[metric],
                label=label,
            )
        ax.set_ylabel("corr. distance")
        ax.set_title(ANALYSIS_DISPLAY.get(analysis, analysis.replace("_", " ")))
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("image")
    fig.suptitle("Targeted RSA distances by image", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_rsa_identity_alignment(identity: pd.DataFrame, output_path: Path) -> None:
    analyses = identity["analysis"].drop_duplicates().tolist()
    groups = ["all", "familiar", "novel"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, constrained_layout=True)
    for ax, analysis in zip(axes.ravel(), analyses):
        sub = (
            identity.loc[identity["analysis"].eq(analysis)]
            .groupby(["stage", "image_group"], as_index=False)[
                ["matched_full_occl_distance", "unmatched_full_occl_mean_distance"]
            ]
            .mean()
        )
        x = np.arange(len(groups))
        width = 0.18
        bars = [
            ("Pre", "matched_full_occl_distance", "Pre matched", "#8C8C8C", -1.5 * width),
            ("Pre", "unmatched_full_occl_mean_distance", "Pre unmatched", "#C7C7C7", -0.5 * width),
            ("Post", "matched_full_occl_distance", "Post matched", "#1F77B4", 0.5 * width),
            ("Post", "unmatched_full_occl_mean_distance", "Post unmatched", "#9ECAE1", 1.5 * width),
        ]
        for stage, col, label, color, offset in bars:
            vals = [
                sub.loc[sub["stage"].eq(stage) & sub["image_group"].eq(group), col].mean()
                for group in groups
            ]
            ax.bar(x + offset, vals, width=width, label=label, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_title(ANALYSIS_DISPLAY.get(analysis, analysis.replace("_", " ")))
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("mean corr. distance")
    axes[1, 0].set_ylabel("mean corr. distance")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("RSA Full/Occl identity alignment", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_rsa_geometry_preservation(geometry: pd.DataFrame, output_path: Path) -> None:
    analyses = geometry["analysis"].drop_duplicates().tolist()
    comparisons = [
        "Pre Full",
        "Pre Occl",
        "Post Full",
        "Post Occl",
    ]
    groups = ["all", "familiar"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, constrained_layout=True)
    for ax, analysis in zip(axes.ravel(), analyses):
        sub = geometry.loc[geometry["analysis"].eq(analysis)]
        x = np.arange(len(comparisons))
        width = 0.34
        for offset, group_name, color in [(-width / 2, "all", "#4C78A8"), (width / 2, "familiar", "#F58518")]:
            vals = [
                sub.loc[
                    sub["image_group"].eq(group_name)
                    & sub["comparison_condition"].eq(comparison),
                    "rdm_pearson_r",
                ].mean()
                for comparison in comparisons
            ]
            ax.bar(x + offset, vals, width=width, label=group_name, color=color)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_ylim(-1, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(comparisons, rotation=20, ha="right")
        ax.set_title(ANALYSIS_DISPLAY.get(analysis, analysis.replace("_", " ")))
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("RDM correlation vs Pre Full")
    axes[1, 0].set_ylabel("RDM correlation vs Pre Full")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("RSA image-geometry preservation", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_supervised_axis_trajectories(projections: pd.DataFrame, output_path: Path) -> None:
    analyses = projections["analysis"].drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, analysis in zip(axes.ravel(), analyses):
        sub = projections.loc[projections["analysis"].eq(analysis)]
        for stage, image_type in CONDITION_ORDER:
            part = (
                sub.loc[sub["stage"].eq(stage) & sub["image_type"].eq(image_type)]
                .groupby("time", as_index=False)[["fullness_axis_score", "learning_axis_score"]]
                .mean()
                .sort_values("time")
            )
            ax.plot(
                part["fullness_axis_score"],
                part["learning_axis_score"],
                color=CONDITION_COLORS[(stage, image_type)],
                lw=2.0,
                label=CONDITION_LABELS[(stage, image_type)],
            )
            stim = part.iloc[(part["time"].abs()).argmin()]
            ax.scatter(
                stim["fullness_axis_score"],
                stim["learning_axis_score"],
                color=CONDITION_COLORS[(stage, image_type)],
                marker="|",
                s=80,
            )
        ax.axhline(0, color="0.82", lw=0.8)
        ax.axvline(0, color="0.82", lw=0.8)
        ax.set_title(ANALYSIS_DISPLAY.get(analysis, analysis.replace("_", " ")))
        ax.set_xlabel("fullness axis")
        ax.set_ylabel("learning axis")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Supervised-axis mean trajectories", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_supervised_axis_timecourses(projections: pd.DataFrame, output_path: Path) -> None:
    analyses = projections["analysis"].drop_duplicates().tolist()
    axis_cols = [
        ("fullness_axis_score", "fullness axis"),
        ("learning_axis_score", "learning axis"),
    ]
    fig, axes = plt.subplots(
        len(analyses),
        len(axis_cols),
        figsize=(11.5, 3.0 * len(analyses)),
        sharex=True,
        constrained_layout=True,
    )
    for row, analysis in enumerate(analyses):
        sub = projections.loc[projections["analysis"].eq(analysis)]
        for col, (axis_col, axis_label) in enumerate(axis_cols):
            ax = axes[row, col]
            for stage, image_type in CONDITION_ORDER:
                part = (
                    sub.loc[sub["stage"].eq(stage) & sub["image_type"].eq(image_type)]
                    .groupby("time", as_index=False)[axis_col]
                    .mean()
                    .sort_values("time")
                )
                ax.plot(
                    part["time"],
                    part[axis_col],
                    color=CONDITION_COLORS[(stage, image_type)],
                    lw=2.0,
                    label=CONDITION_LABELS[(stage, image_type)],
                )
            ax.axvline(0, color="0.7", lw=0.9)
            ax.axvspan(0.2, 1.0, color="0.92", zorder=-1)
            ax.axhline(0, color="0.82", lw=0.8)
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(axis_label)
            if col == 0:
                ax.set_ylabel(ANALYSIS_DISPLAY.get(analysis, analysis.replace("_", " ")))
            if row == len(analyses) - 1:
                ax.set_xlabel("time")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Supervised-axis mean time courses", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_supervised_axis_image_trajectories(projections: pd.DataFrame, output_dir: Path) -> None:
    analyses = projections["analysis"].drop_duplicates().tolist()
    for analysis in analyses:
        sub = projections.loc[projections["analysis"].eq(analysis)].copy()
        images = sorted(sub["image_idx_original"].unique().tolist())
        ncols = 3
        nrows = int(np.ceil(len(images) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.8 * ncols, 4.0 * nrows),
            squeeze=False,
        )

        x = sub["fullness_axis_score"].to_numpy(dtype=float)
        y = sub["learning_axis_score"].to_numpy(dtype=float)
        x_pad = max((np.nanmax(x) - np.nanmin(x)) * 0.08, 1.0)
        y_pad = max((np.nanmax(y) - np.nanmin(y)) * 0.08, 1.0)
        xlim = (float(np.nanmin(x) - x_pad), float(np.nanmax(x) + x_pad))
        ylim = (float(np.nanmin(y) - y_pad), float(np.nanmax(y) + y_pad))

        for ax, image_idx in zip(axes.ravel(), images):
            img = sub.loc[sub["image_idx_original"].eq(image_idx)]
            image_group = img["image_group"].mode().iat[0]
            for stage, image_type in CONDITION_ORDER:
                part = img.loc[
                    img["stage"].eq(stage) & img["image_type"].eq(image_type)
                ].sort_values("time")
                color = CONDITION_COLORS[(stage, image_type)]
                label = CONDITION_LABELS[(stage, image_type)]
                ax.plot(
                    part["fullness_axis_score"],
                    part["learning_axis_score"],
                    color=color,
                    lw=1.9,
                    label=label,
                )
                start = part.iloc[0]
                stim = part.iloc[(part["time"].abs()).argmin()]
                end = part.iloc[-1]
                ax.scatter(
                    start["fullness_axis_score"],
                    start["learning_axis_score"],
                    color=color,
                    s=18,
                    marker="o",
                    alpha=0.75,
                )
                ax.scatter(
                    stim["fullness_axis_score"],
                    stim["learning_axis_score"],
                    color=color,
                    s=70,
                    marker="|",
                    linewidth=1.8,
                )
                ax.scatter(
                    end["fullness_axis_score"],
                    end["learning_axis_score"],
                    color=color,
                    s=28,
                    marker=">",
                    alpha=0.9,
                )
            ax.axhline(0, color="0.82", lw=0.8)
            ax.axvline(0, color="0.82", lw=0.8)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_title(f"Image {int(image_idx)} ({image_group})", fontsize=10)
            ax.set_xlabel("fullness axis")
            ax.set_ylabel("learning axis")
            ax.grid(alpha=0.22)

        for ax in axes.ravel()[len(images) :]:
            ax.axis("off")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.suptitle(
            f"{ANALYSIS_DISPLAY.get(analysis, analysis.replace('_', ' '))}: image trajectories in fullness/learning plane",
            y=0.99,
            fontsize=14,
            fontweight="bold",
        )
        fig.legend(
            handles[:4],
            labels[:4],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=4,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(output_dir / f"{analysis}_supervised_axis_by_image.png", dpi=200)
        plt.close(fig)


def plot_image_axis_scores(image_scores: pd.DataFrame, output_path: Path) -> None:
    analyses = image_scores["analysis"].drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    for ax, analysis in zip(axes.ravel(), analyses):
        sub = image_scores.loc[image_scores["analysis"].eq(analysis)]
        for stage, image_type in CONDITION_ORDER:
            part = sub.loc[sub["stage"].eq(stage) & sub["image_type"].eq(image_type)].sort_values("image_idx_original")
            ax.plot(
                part["image_idx_original"],
                part["image_axis_score"],
                marker="o",
                lw=2,
                color=CONDITION_COLORS[(stage, image_type)],
                label=CONDITION_LABELS[(stage, image_type)],
            )
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_title(ANALYSIS_DISPLAY.get(analysis, analysis.replace("_", " ")))
        ax.set_xlabel("image")
        ax.set_ylabel("image-axis score")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Image-specific supervised-axis scores", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Population-level RSA, separability, and supervised-axis analyses.")
    parser.add_argument(
        "--trace-csv",
        type=Path,
        default=Path("context_contrasting/data_analysis/transitions_post_traces.csv"),
    )
    parser.add_argument(
        "--scalar-csv",
        type=Path,
        default=Path("context_contrasting/data_analysis/transitions_post.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("context_contrasting/data_analysis/population_analyses"),
    )
    parser.add_argument("--window-start", type=float, default=0.2)
    parser.add_argument("--window-end", type=float, default=1.0)
    parser.add_argument("--sector-threshold", type=float, default=0.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace = pd.read_csv(args.trace_csv)
    z_trace, baseline_stats = zscore_from_pre_full_baseline(trace)
    sectors = build_sector_assignments(args.scalar_csv, args.sector_threshold)
    neuron_sets = analysis_neuron_sets(z_trace, sectors)

    baseline_stats.to_csv(args.output_dir / "pre_full_baseline_zscore_stats.csv", index=False)
    sectors.to_csv(args.output_dir / "post_transition_sector_assignments.csv", index=False)

    rsa_matrices: dict[str, pd.DataFrame] = {}
    rsa_long_frames = []
    separability_frames = []
    rsa_target_distance_frames = []
    rsa_identity_frames = []
    rsa_geometry_frames = []
    projection_frames = []
    image_score_frames = []
    axis_summary_frames = []

    for analysis_name, neuron_ids in neuron_sets.items():
        if len(neuron_ids) < 2:
            continue
        window_meta, window_matrix = build_window_vectors(
            z_trace,
            neuron_ids,
            window_start=args.window_start,
            window_end=args.window_end,
        )
        time_meta, time_matrix = build_time_vectors(z_trace, neuron_ids)

        rsa_wide, rsa_long = compute_rsa(analysis_name, window_meta, window_matrix, args.output_dir)
        rsa_matrices[analysis_name] = rsa_wide
        rsa_long_frames.append(rsa_long)
        separability_frames.append(compute_same_different(analysis_name, window_meta, window_matrix))
        target_distances, identity_alignment, geometry = compute_rsa_followups(
            analysis_name,
            window_meta,
            window_matrix,
        )
        rsa_target_distance_frames.append(target_distances)
        rsa_identity_frames.append(identity_alignment)
        rsa_geometry_frames.append(geometry)
        projections, image_scores, axis_summary = compute_supervised_axes(
            analysis_name,
            window_meta,
            window_matrix,
            time_meta,
            time_matrix,
        )
        projection_frames.append(projections)
        image_score_frames.append(image_scores)
        axis_summary_frames.append(axis_summary)

    rsa_long_all = pd.concat(rsa_long_frames, ignore_index=True)
    separability_all = pd.concat(separability_frames, ignore_index=True)
    rsa_target_distances_all = pd.concat(rsa_target_distance_frames, ignore_index=True)
    rsa_identity_all = pd.concat(rsa_identity_frames, ignore_index=True)
    rsa_geometry_all = pd.concat(rsa_geometry_frames, ignore_index=True)
    projections_all = pd.concat(projection_frames, ignore_index=True)
    image_scores_all = pd.concat(image_score_frames, ignore_index=True)
    axis_summary_all = pd.concat(axis_summary_frames, ignore_index=True)

    rsa_long_all.to_csv(args.output_dir / "rsa_pairwise_distances.csv", index=False)
    separability_all.to_csv(args.output_dir / "same_different_separability.csv", index=False)
    rsa_target_distances_all.to_csv(args.output_dir / "rsa_targeted_distances.csv", index=False)
    rsa_identity_all.to_csv(args.output_dir / "rsa_identity_alignment.csv", index=False)
    rsa_geometry_all.to_csv(args.output_dir / "rsa_geometry_preservation.csv", index=False)
    projections_all.to_csv(args.output_dir / "supervised_axis_trajectories.csv", index=False)
    image_scores_all.to_csv(args.output_dir / "image_axis_scores.csv", index=False)
    axis_summary_all.to_csv(args.output_dir / "supervised_axis_summary.csv", index=False)

    plot_rsa_heatmaps(rsa_matrices, args.output_dir / "rsa_distance_heatmaps.png")
    plot_rsa_distance_followups(rsa_target_distances_all, args.output_dir / "rsa_targeted_distances_by_image.png")
    plot_rsa_identity_alignment(rsa_identity_all, args.output_dir / "rsa_identity_alignment.png")
    plot_rsa_geometry_preservation(rsa_geometry_all, args.output_dir / "rsa_geometry_preservation.png")
    plot_separability_by_image(separability_all, args.output_dir / "same_different_separability_by_image.png")
    plot_separability_summary(separability_all, args.output_dir / "same_different_separability_summary.png")
    plot_supervised_axis_trajectories(projections_all, args.output_dir / "supervised_axis_trajectories.png")
    plot_supervised_axis_timecourses(projections_all, args.output_dir / "supervised_axis_timecourses.png")
    plot_supervised_axis_image_trajectories(projections_all, args.output_dir)
    plot_image_axis_scores(image_scores_all, args.output_dir / "image_axis_scores.png")

    metadata = {
        "trace_csv": str(args.trace_csv),
        "scalar_csv": str(args.scalar_csv),
        "output_dir": str(args.output_dir),
        "zscore_reference": "stage == Pre, image_type == Full, time < 0",
        "stimulus_window": [args.window_start, args.window_end],
        "distance_metric": "correlation distance on stimulus-window mean z-scored population vectors",
        "supervised_axes": {
            "fullness": "mean Pre Full minus mean Pre Occl, stimulus window",
            "learning": "mean Post Full minus mean Pre Full, stimulus window",
            "image_axis": "Pre Full image i minus mean Pre Full other images, stimulus window",
        },
        "n_neurons": {name: len(ids) for name, ids in neuron_sets.items()},
        "sector_threshold": args.sector_threshold,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
