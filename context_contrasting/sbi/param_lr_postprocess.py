from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGETS = [
    "un_FB",
    "FF_FB_broad",
    "FF_FB_narrow_familiar",
    "FF_FB_narrow_novel",
    "FB_FB",
    "FF_un",
]

STANDARD_TARGETS = [target for target in TARGETS if target != "FF_un"]

PARAM_ORDER = [
    "w_ff_0",
    "w_ff_1",
    "w_fb_0",
    "w_fb_1",
    "w_lat_0",
    "w_lat_1",
    "w_pv_lat_0",
    "w_pv_lat_1",
    "lr_ff",
    "lr_fb",
    "lr_lat",
    "lr_pv",
]

TARGET_COLORS = {
    "un_FB": "#7f7f7f",
    "FF_FB_broad": "#4c78a8",
    "FF_FB_narrow_familiar": "#f58518",
    "FF_FB_narrow_novel": "#54a24b",
    "FB_FB": "#e45756",
    "FF_un": "#b279a2",
}


def _sorted_pair(a: str, b: str) -> str:
    left, right = sorted((a, b))
    return f"{left}__{right}"


def _load_frames(result_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "combined": pd.read_csv(result_dir / "combined_candidates.csv"),
        "embedded": pd.read_csv(result_dir / "embedded_candidates.csv"),
        "stable": pd.read_csv(result_dir / "stable_region_members.csv"),
        "validated_core": pd.read_csv(result_dir / "validated_core_summary.csv"),
        "boundary_switches": pd.read_csv(result_dir / "boundary_switches.csv"),
    }


def _augment_competition(frame: pd.DataFrame) -> pd.DataFrame:
    objective_cols = [f"objective__{target}" for target in TARGETS]
    values = frame.loc[:, objective_cols].to_numpy(dtype=float)
    order = np.argsort(values, axis=1)
    classes = np.asarray(TARGETS)

    top1 = order[:, 0]
    top2 = order[:, 1]
    augmented = frame.copy()
    augmented["runner_up"] = classes[top2]
    augmented["runner_up_objective"] = values[np.arange(len(frame)), top2]
    augmented["margin2"] = augmented["runner_up_objective"] - augmented["assigned_objective"]
    pair = np.sort(
        np.stack(
            [augmented["assigned_target"].to_numpy(), augmented["runner_up"].to_numpy()],
            axis=1,
        ),
        axis=1,
    )
    augmented["boundary_pair"] = pair[:, 0] + "__" + pair[:, 1]
    augmented["same_manifold_pair"] = ~augmented["boundary_pair"].str.contains("FF_un")
    return augmented


def _stable_medians(stable: pd.DataFrame) -> pd.DataFrame:
    medians = stable.groupby("assigned_target")[PARAM_ORDER].median().reset_index()
    medians = medians.rename(columns={"assigned_target": "target"})
    return medians


def _competition_summary(augmented: pd.DataFrame, boundary_quantile: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold = float(augmented["margin2"].quantile(boundary_quantile))
    boundary_cloud = augmented.loc[augmented["margin2"].le(threshold)].copy()

    rows: list[dict[str, float | int | str | bool]] = []
    for pair_name, subset in boundary_cloud.groupby("boundary_pair"):
        dominant_target = str(subset["assigned_target"].value_counts().idxmax())
        row: dict[str, float | int | str | bool] = {
            "boundary_pair": pair_name,
            "n_points": int(len(subset)),
            "median_margin2": float(subset["margin2"].median()),
            "same_manifold_pair": bool(subset["same_manifold_pair"].iloc[0]),
            "dominant_assigned_target": dominant_target,
        }
        for param in PARAM_ORDER:
            row[f"{param}_median"] = float(subset[param].median())
            row[f"{param}_log_iqr"] = float(
                math.log10(subset[param].quantile(0.75)) - math.log10(subset[param].quantile(0.25))
            )
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["n_points", "median_margin2"], ascending=[False, True])
    return boundary_cloud, summary


def _switch_summary_table(boundary_switches: pd.DataFrame) -> pd.DataFrame:
    if boundary_switches.empty:
        return boundary_switches.copy()
    switches = boundary_switches.copy()
    switches["switch_type"] = np.where(
        switches["first_switch_alpha"].isna(),
        "no_switch_along_path",
        np.where(
            switches["first_class_b_alpha"].eq(switches["first_switch_alpha"]),
            "direct_switch",
            "detour_switch",
        ),
    )
    return switches


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_margin_distributions(augmented: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    data = [np.log10(augmented.loc[augmented["assigned_target"].eq(target), "margin2"]) for target in TARGETS]
    ax.boxplot(data, tick_labels=TARGETS, showfliers=False)
    ax.set_ylabel("log10(second-best margin)")
    ax.set_title("Classwise Assignment Margin Distributions")
    ax.tick_params(axis="x", rotation=20)
    _save_fig(fig, output_dir / "margin_distributions.png")


def _plot_class_medians(stable_medians: pd.DataFrame, output_dir: Path) -> None:
    matrix = np.log10(stable_medians.set_index("target").loc[TARGETS, PARAM_ORDER].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(PARAM_ORDER)))
    ax.set_xticklabels(PARAM_ORDER, rotation=35, ha="right")
    ax.set_yticks(range(len(TARGETS)))
    ax.set_yticklabels(TARGETS)
    ax.set_title("Stable-Region Median Parameters (log10 scale)")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("log10(parameter median)")
    _save_fig(fig, output_dir / "class_parameter_medians.png")


def _plot_boundary_pair_clouds(embedded: pd.DataFrame, boundary_cloud: pd.DataFrame, output_dir: Path) -> None:
    pairs = boundary_cloud["boundary_pair"].value_counts().head(8).index.tolist()
    candidate_ids = boundary_cloud.loc[boundary_cloud["boundary_pair"].isin(pairs), "candidate_uid"]
    cloud = embedded.loc[embedded["candidate_uid"].isin(candidate_ids)].copy()
    pair_palette = {
        pair_name: plt.cm.tab10(idx % 10)
        for idx, pair_name in enumerate(pairs)
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))

    for pair_name in pairs:
        subset = cloud.loc[cloud["boundary_pair"].eq(pair_name)]
        if subset.empty:
            continue
        label = pair_name.replace("__", " vs ")
        axes[0].scatter(
            subset["pca1"],
            subset["pca2"],
            s=12,
            alpha=0.55,
            color=pair_palette[pair_name],
            label=label,
            edgecolors="none",
        )
        axes[1].scatter(
            subset["tsne1"],
            subset["tsne2"],
            s=12,
            alpha=0.55,
            color=pair_palette[pair_name],
            label=label,
            edgecolors="none",
        )

    axes[0].set_title("Low-Margin Boundary Clouds in PCA")
    axes[0].set_xlabel("pca1")
    axes[0].set_ylabel("pca2")
    axes[1].set_title("Low-Margin Boundary Clouds in t-SNE")
    axes[1].set_xlabel("tsne1")
    axes[1].set_ylabel("tsne2")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_fig(fig, output_dir / "boundary_pair_clouds.png")


def _plot_boundary_pair_heatmap(boundary_summary: pd.DataFrame, output_dir: Path) -> None:
    top_pairs = boundary_summary.head(8).copy()
    if top_pairs.empty:
        return
    matrix = np.log10(top_pairs.loc[:, [f"{param}_median" for param in PARAM_ORDER]].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(PARAM_ORDER)))
    ax.set_xticklabels(PARAM_ORDER, rotation=35, ha="right")
    ax.set_yticks(range(len(top_pairs)))
    ax.set_yticklabels([pair.replace("__", " vs ") for pair in top_pairs["boundary_pair"]])
    ax.set_title("Top Boundary-Pair Median Parameters (log10 scale)")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("log10(boundary-pair median)")
    _save_fig(fig, output_dir / "boundary_pair_parameter_heatmap.png")


def _write_report(
    result_dir: Path,
    *,
    boundary_quantile: float,
    augmented: pd.DataFrame,
    stable_medians: pd.DataFrame,
    competition_summary: pd.DataFrame,
    switch_summary: pd.DataFrame,
    validated_core: pd.DataFrame,
) -> None:
    threshold = float(augmented["margin2"].quantile(boundary_quantile))
    lines = [
        "# Posthoc Boundary Analysis",
        "",
        "This report reuses the saved 12D landscape and inspects the top-2 objective competition for every sampled point.",
        f"Boundary clouds are defined as the lowest `{int(boundary_quantile * 100)}`% of second-best margins, i.e. `margin2 <= {threshold:.4f}`.",
        "",
        "## Main Observations",
        "",
        "- `w_ff_0` and `w_ff_1` remain the dominant separators. Learning rates matter, but mostly as secondary deformations of already weight-defined regions.",
        "- `lr_fb` is the most important learning-rate axis in the random-forest ranking. It shows up mainly in the unresponsive vs selective separations, not in the broad FF vs narrow FF split.",
        "- The strongest low-margin competition is `FF_FB_narrow_novel` vs `FF_un`, but that pair crosses the `receives_context` manifold boundary and should not be interpreted as a standard-plasticity decision surface.",
        "- Within the standard manifold, the most active switch zones are `FF_FB_narrow_familiar` vs `un_FB`, `FB_FB` vs `FF_FB_narrow_novel`, `FF_FB_broad` vs `FF_FB_narrow_novel`, and `FB_FB` vs `un_FB`.",
        "",
        "## Stable-Class Medians",
        "",
    ]
    for row in stable_medians.to_dict(orient="records"):
        lines.append(
            f"- `{row['target']}`: "
            f"`w_ff=({row['w_ff_0']:.4g}, {row['w_ff_1']:.4g})`, "
            f"`w_fb=({row['w_fb_0']:.4g}, {row['w_fb_1']:.4g})`, "
            f"`w_lat=({row['w_lat_0']:.4g}, {row['w_lat_1']:.4g})`, "
            f"`w_pv_lat=({row['w_pv_lat_0']:.4g}, {row['w_pv_lat_1']:.4g})`, "
            f"`lr=(ff {row['lr_ff']:.4g}, fb {row['lr_fb']:.4g}, lat {row['lr_lat']:.4g}, pv {row['lr_pv']:.4g})`"
        )

    lines.extend(
        [
            "",
            "## Top Boundary Clouds",
            "",
        ]
    )
    for row in competition_summary.head(10).to_dict(orient="records"):
        pair_name = row["boundary_pair"].replace("__", " vs ")
        manifold = "same manifold" if row["same_manifold_pair"] else "cross-manifold"
        lines.append(
            f"- `{pair_name}`: n=`{row['n_points']}`, median margin=`{row['median_margin2']:.4f}`, "
            f"{manifold}, dominant assigned target=`{row['dominant_assigned_target']}`"
        )

    if not switch_summary.empty:
        lines.extend(
            [
                "",
                "## Boundary Path Behavior",
                "",
            ]
        )
        for row in switch_summary.head(10).to_dict(orient="records"):
            pair_name = f"{row['class_a']} vs {row['class_b']}"
            if pd.isna(row["first_switch_alpha"]):
                lines.append(
                    f"- `{pair_name}` path `{row['candidate_idx_a']}` -> `{row['candidate_idx_b']}` never switched class along the sampled interpolation."
                )
            else:
                lines.append(
                    f"- `{pair_name}` path `{row['candidate_idx_a']}` -> `{row['candidate_idx_b']}` first switched at `alpha={row['first_switch_alpha']:.3f}` "
                    f"({row['switch_type']}); sequence: `{row['path_sequence']}`"
                )

    lines.extend(
        [
            "",
            "## Backtest Reading",
            "",
            "Validated-core backtests are the main trust anchor for whether a region behaves as a robust basin instead of a thin assignment shell.",
        ]
    )
    for row in validated_core.to_dict(orient="records"):
        lines.append(
            f"- `{row['target']}`: q{int(row['validated_quantile_low'] * 100):02d}-q{int(row['validated_quantile_high'] * 100):02d}, "
            f"boundary success=`{row['core_boundary_backtest_success']:.3f}`"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `FB_FB` remains the cleanest basin after adding learning rates. Its backtests are near-perfect and its boundary competitors are limited.",
            "- `FF_un` is also robust, but only on the separate `receives_context=(False, False)` manifold.",
            "- The standard-manifold FF-family classes remain partly entangled. Narrow familiar, narrow novel, and broad FF are better seen as neighboring regions inside one larger FF-driven regime than as completely isolated islands.",
            "- `un_FB` absorbs a very large volume of parameter space, but its interquartile core is only moderately stable under full backtests. That suggests the class is easy to assign in compact summaries yet sensitive near its edges.",
        ]
    )

    (result_dir / "posthoc_overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_posthoc(result_dir: str | Path, boundary_quantile: float = 0.10) -> dict[str, pd.DataFrame]:
    result_path = Path(result_dir)
    frames = _load_frames(result_path)

    augmented = _augment_competition(frames["combined"])
    embedded_augmented = _augment_competition(frames["embedded"])
    stable_medians = _stable_medians(frames["stable"])
    boundary_cloud, competition_summary = _competition_summary(augmented, boundary_quantile=boundary_quantile)
    switch_summary = _switch_summary_table(frames["boundary_switches"])

    augmented.to_csv(result_path / "competition_augmented.csv", index=False)
    competition_summary.to_csv(result_path / "boundary_pair_summary.csv", index=False)
    stable_medians.to_csv(result_path / "stable_median_parameters.csv", index=False)
    switch_summary.to_csv(result_path / "boundary_switches_annotated.csv", index=False)

    plot_dir = result_path / "plots"
    _plot_margin_distributions(augmented, plot_dir)
    _plot_class_medians(stable_medians, plot_dir)
    _plot_boundary_pair_clouds(embedded_augmented, boundary_cloud, plot_dir)
    _plot_boundary_pair_heatmap(competition_summary, plot_dir)
    _write_report(
        result_path,
        boundary_quantile=boundary_quantile,
        augmented=augmented,
        stable_medians=stable_medians,
        competition_summary=competition_summary,
        switch_summary=switch_summary,
        validated_core=frames["validated_core"],
    )

    return {
        "augmented": augmented,
        "boundary_pair_summary": competition_summary,
        "stable_medians": stable_medians,
        "switch_summary": switch_summary,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process a saved learning-rate landscape run into boundary-cloud summaries and plots."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("context_contrasting/sbi/results/2026-05-01_lr_landscape"),
        help="Directory containing the saved outputs from param_lr_landscape.py",
    )
    parser.add_argument(
        "--boundary-quantile",
        type=float,
        default=0.10,
        help="Use the lowest q fraction of second-best margins as the boundary cloud.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_posthoc(args.result_dir, boundary_quantile=args.boundary_quantile)
    print(result["boundary_pair_summary"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
