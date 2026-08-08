from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


CONDITION_ORDER = [
    ("Pre", "Full"),
    ("Pre", "Occl"),
    ("Post", "Full"),
    ("Post", "Occl"),
]

CONDITION_STYLE = {
    ("Pre", "Full"): {"label": "Pre Full", "color": "#202020", "linestyle": "-"},
    ("Pre", "Occl"): {"label": "Pre Occl", "color": "#C33C2D", "linestyle": "-"},
    ("Post", "Full"): {"label": "Post Full", "color": "#1F77B4", "linestyle": "-"},
    ("Post", "Occl"): {"label": "Post Occl", "color": "#7B3294", "linestyle": "-"},
}

SECTOR_ORDER = ["-NO axis", "+O axis", "+NO axis"]
SECTOR_SLUG = {
    "-NO axis": "minus_no_axis",
    "+O axis": "plus_o_axis",
    "+NO axis": "plus_no_axis",
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


def condition_matrix(
    frame: pd.DataFrame,
    neuron_ids: list[int],
) -> tuple[pd.DataFrame, np.ndarray]:
    index_cols = [
        "image_group",
        "image_idx_original",
        "image_idx_within_group",
        "image_type",
        "stage",
        "time",
    ]
    wide = frame.pivot(index=index_cols, columns="neuron_idx", values="z_response")
    wide = wide.loc[:, neuron_ids]
    if wide.isna().any().any():
        raise ValueError("PCA matrix contains missing z-scored responses.")
    meta = wide.index.to_frame(index=False)
    return meta, wide.to_numpy(dtype=float)


def fit_project_pca(
    z_frame: pd.DataFrame,
    neuron_ids: list[int],
    analysis_name: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = z_frame.loc[z_frame["neuron_idx"].isin(neuron_ids)].copy()
    meta, matrix = condition_matrix(sub, neuron_ids)
    fit_mask = meta["stage"].eq("Pre") & meta["image_type"].eq("Full")
    n_components = min(3, int(fit_mask.sum()), len(neuron_ids))
    if n_components < 2:
        raise ValueError(f"Need at least two PCA components for {analysis_name}.")

    pca = PCA(n_components=n_components)
    pca.fit(matrix[fit_mask.to_numpy()])
    scores = pca.transform(matrix)

    coords = meta.copy()
    coords.insert(0, "analysis", analysis_name)
    coords["condition"] = coords["stage"] + " " + coords["image_type"]
    for i in range(n_components):
        coords[f"PC{i + 1}"] = scores[:, i]
    for i in range(n_components, 3):
        coords[f"PC{i + 1}"] = np.nan

    explained = pd.DataFrame(
        {
            "analysis": analysis_name,
            "component": [f"PC{i + 1}" for i in range(n_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "n_neurons": len(neuron_ids),
            "n_fit_observations": int(fit_mask.sum()),
        }
    )

    coords.to_csv(output_dir / f"{analysis_name}_coordinates.csv", index=False)
    explained.to_csv(output_dir / f"{analysis_name}_explained_variance.csv", index=False)
    return coords, explained


def _axis_limits(coord_frames: list[pd.DataFrame], columns: tuple[str, str]) -> tuple[tuple[float, float], tuple[float, float]]:
    x = np.concatenate([f[columns[0]].dropna().to_numpy(dtype=float) for f in coord_frames])
    y = np.concatenate([f[columns[1]].dropna().to_numpy(dtype=float) for f in coord_frames])
    x_pad = max((x.max() - x.min()) * 0.08, 0.5)
    y_pad = max((y.max() - y.min()) * 0.08, 0.5)
    return (float(x.min() - x_pad), float(x.max() + x_pad)), (float(y.min() - y_pad), float(y.max() + y_pad))


def plot_pc12_trajectories(
    coords: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
) -> None:
    images = sorted(coords["image_idx_original"].unique())
    ncols = 3
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.8 * nrows), squeeze=False)
    xlim, ylim = _axis_limits([coords], ("PC1", "PC2"))

    for ax, image_idx in zip(axes.ravel(), images):
        img = coords.loc[coords["image_idx_original"].eq(image_idx)].copy()
        group = img["image_group"].mode().iat[0]
        for stage, image_type in CONDITION_ORDER:
            part = img.loc[img["stage"].eq(stage) & img["image_type"].eq(image_type)].sort_values("time")
            style = CONDITION_STYLE[(stage, image_type)]
            ax.plot(part["PC1"], part["PC2"], lw=1.8, color=style["color"], linestyle=style["linestyle"], label=style["label"])
            start = part.iloc[0]
            end = part.iloc[-1]
            stim = part.iloc[(part["time"].abs()).argmin()]
            ax.scatter(start["PC1"], start["PC2"], s=18, color=style["color"], marker="o", alpha=0.75)
            ax.scatter(stim["PC1"], stim["PC2"], s=28, color=style["color"], marker="|", linewidth=1.8)
            ax.scatter(end["PC1"], end["PC2"], s=30, color=style["color"], marker=">", alpha=0.9)
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"Image {image_idx} ({group})", fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)

    for ax in axes.ravel()[len(images) :]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(title, y=0.99, fontsize=13, fontweight="bold")
    fig.legend(
        handles[:4],
        labels[:4],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pc_timecourses(coords: pd.DataFrame, *, title: str, output_path: Path) -> None:
    pcs = ["PC1", "PC2", "PC3"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharex=True)
    for ax, pc in zip(axes, pcs):
        for stage, image_type in CONDITION_ORDER:
            part = (
                coords.loc[coords["stage"].eq(stage) & coords["image_type"].eq(image_type)]
                .groupby("time", as_index=False)[pc]
                .mean()
                .sort_values("time")
            )
            style = CONDITION_STYLE[(stage, image_type)]
            ax.plot(part["time"], part[pc], lw=2.0, color=style["color"], label=style["label"])
        ax.axvline(0, color="0.7", lw=1.0)
        ax.axvspan(0.2, 1.0, color="0.92", zorder=-1)
        ax.axhline(0, color="0.85", lw=0.8)
        ax.set_title(pc)
        ax.set_xlabel("time")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("mean score across images")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title, y=0.99, fontsize=13, fontweight="bold")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_explained_variance(explained: pd.DataFrame, *, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    analyses = explained["analysis"].drop_duplicates().tolist()
    x = np.arange(len(analyses))
    width = 0.24
    for i, pc in enumerate(["PC1", "PC2", "PC3"]):
        vals = [
            explained.loc[
                explained["analysis"].eq(analysis) & explained["component"].eq(pc),
                "explained_variance_ratio",
            ].sum()
            for analysis in analyses
        ]
        ax.bar(x + (i - 1) * width, vals, width=width, label=pc)
    ax.set_xticks(x)
    ax.set_xticklabels(analyses, rotation=20, ha="right")
    ax.set_ylabel("explained variance ratio")
    ax.set_ylim(0, max(0.05, explained["explained_variance_ratio"].max() * 1.25))
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit Pre Full baseline-z-scored PCA trajectories and project occluded/post conditions."
    )
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
        default=Path("context_contrasting/data_analysis/pca_trajectories"),
    )
    parser.add_argument("--sector-threshold", type=float, default=0.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace = pd.read_csv(args.trace_csv)
    z_trace, baseline_stats = zscore_from_pre_full_baseline(trace)
    baseline_stats.to_csv(args.output_dir / "pre_full_baseline_zscore_stats.csv", index=False)

    sector_assignments = build_sector_assignments(args.scalar_csv, threshold=args.sector_threshold)
    sector_assignments.to_csv(args.output_dir / "post_transition_sector_assignments.csv", index=False)

    all_neurons = sorted(z_trace["neuron_idx"].unique().tolist())
    all_coords, all_explained = fit_project_pca(z_trace, all_neurons, "full_population", args.output_dir)
    plot_pc12_trajectories(
        all_coords,
        title="Full population: PCA fit on Pre Full, all conditions projected",
        output_path=args.output_dir / "full_population_pc12_trajectories.png",
    )
    plot_pc_timecourses(
        all_coords,
        title="Full population: mean projected PC time courses",
        output_path=args.output_dir / "full_population_pc_timecourses.png",
    )

    all_explained_frames = [all_explained]
    sector_counts: dict[str, int] = {}
    for sector in SECTOR_ORDER:
        neuron_ids = sorted(
            sector_assignments.loc[sector_assignments["rotated_sector"].eq(sector), "neuron_idx"]
            .astype(int)
            .tolist()
        )
        sector_counts[sector] = len(neuron_ids)
        if len(neuron_ids) < 2:
            continue
        analysis_name = f"subpopulation_{SECTOR_SLUG[sector]}"
        coords, explained = fit_project_pca(z_trace, neuron_ids, analysis_name, args.output_dir)
        all_explained_frames.append(explained)
        plot_pc12_trajectories(
            coords,
            title=f"{sector} subpopulation: PCA fit on Pre Full, all conditions projected",
            output_path=args.output_dir / f"{analysis_name}_pc12_trajectories.png",
        )
        plot_pc_timecourses(
            coords,
            title=f"{sector} subpopulation: mean projected PC time courses",
            output_path=args.output_dir / f"{analysis_name}_pc_timecourses.png",
        )

    combined_explained = pd.concat(all_explained_frames, ignore_index=True)
    combined_explained.to_csv(args.output_dir / "all_explained_variance.csv", index=False)
    plot_explained_variance(
        combined_explained,
        output_path=args.output_dir / "explained_variance_summary.png",
    )

    metadata = {
        "trace_csv": str(args.trace_csv),
        "scalar_csv": str(args.scalar_csv),
        "output_dir": str(args.output_dir),
        "zscore_reference": "stage == Pre, image_type == Full, time < 0",
        "pca_fit_reference": "stage == Pre, image_type == Full after baseline z-scoring",
        "condition_order": [f"{stage} {image_type}" for stage, image_type in CONDITION_ORDER],
        "n_neurons_full_population": len(all_neurons),
        "sector_threshold": args.sector_threshold,
        "sector_counts": sector_counts,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
