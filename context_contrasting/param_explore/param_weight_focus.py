from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from context_contrasting.param_explore.param_explorer import (
    SweepSettings,
    _reference_config_for_target,
    simulate_response_summary,
)

WEIGHT_COLS = [
    "w_ff_0",
    "w_ff_1",
    "w_fb_0",
    "w_fb_1",
    "w_lat_0",
    "w_lat_1",
    "w_pv_lat_0",
    "w_pv_lat_1",
]

SUMMARY_COLS = [
    "full_familiar_naive",
    "occlusion_familiar_naive",
    "full_familiar_expert",
    "occlusion_familiar_expert",
    "full_novel_naive",
    "occlusion_novel_naive",
    "full_novel_expert",
    "occlusion_novel_expert",
]

FOCUS_TARGETS = [
    "un_FB",
    "FF_FB_broad",
    "FF_FB_narrow_familiar",
    "FF_FB_narrow_novel",
    "FF_un",
]

REFERENCE_ONLY_TARGETS = ["un_un"]

TARGET_COLORS = {
    "un_un": "#7f7f7f",
    "un_FB": "#4c78a8",
    "FF_FB_broad": "#54a24b",
    "FF_FB_narrow_familiar": "#e45756",
    "FF_FB_narrow_novel": "#72b7b2",
    "FF_un": "#f58518",
}


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _style_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=9, loc="best")


def _load_frames(results_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "combined": pd.read_csv(results_dir / "combined_candidates.csv"),
        "stable": pd.read_csv(results_dir / "stable_region_members.csv"),
        "core": pd.read_csv(results_dir / "validated_core_summary.csv"),
    }


def _reference_table(settings: SweepSettings) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in [*REFERENCE_ONLY_TARGETS, *FOCUS_TARGETS]:
        config = _reference_config_for_target(target)
        summary = simulate_response_summary(
            config,
            n_steps_per_phase=settings.n_steps_per_phase,
            n_trials=settings.n_trials,
            tail_window=settings.tail_window,
            intertrial_sigma=settings.intertrial_sigma,
        )
        row = {
            "target": target,
            "receives_context_familiar": bool(config["receives_context"][0]),
            "receives_context_novel": bool(config["receives_context"][1]),
        }
        row.update({key: float(summary[key]) for key in SUMMARY_COLS})
        rows.append(row)
    return pd.DataFrame(rows)


def _focused_core_ranges(core: pd.DataFrame) -> pd.DataFrame:
    cols = ["target", "n_region_points", "validated_quantile_low", "validated_quantile_high", "familiar_transition", "novel_transition"]
    for weight in WEIGHT_COLS:
        cols.extend([f"{weight}_low", f"{weight}_high"])
    return core.loc[core["target"].isin(FOCUS_TARGETS), cols].copy()


def _stable_medians(stable: pd.DataFrame) -> pd.DataFrame:
    subset = stable.loc[stable["assigned_target"].isin(FOCUS_TARGETS)].copy()
    med = subset.groupby("assigned_target", as_index=False)[WEIGHT_COLS + SUMMARY_COLS].median()
    return med.rename(columns={"assigned_target": "target"})


def _pairwise_weight_differences(stable: pd.DataFrame) -> pd.DataFrame:
    subset = stable.loc[stable["assigned_target"].isin(FOCUS_TARGETS)].copy()
    rows: list[dict[str, Any]] = []
    for target_a, target_b in combinations(FOCUS_TARGETS, 2):
        class_a = subset.loc[subset["assigned_target"].eq(target_a)]
        class_b = subset.loc[subset["assigned_target"].eq(target_b)]
        if class_a.empty or class_b.empty:
            continue
        median_a = np.log10(class_a[WEIGHT_COLS].median())
        median_b = np.log10(class_b[WEIGHT_COLS].median())
        diff = (median_a - median_b).abs().sort_values(ascending=False)
        row: dict[str, Any] = {
            "target_a": target_a,
            "target_b": target_b,
            "top_param_1": diff.index[0],
            "top_abs_log10_median_diff_1": float(diff.iloc[0]),
            "top_param_2": diff.index[1],
            "top_abs_log10_median_diff_2": float(diff.iloc[1]),
            "top_param_3": diff.index[2],
            "top_abs_log10_median_diff_3": float(diff.iloc[2]),
        }
        for weight in WEIGHT_COLS:
            row[f"{weight}_median_{target_a}"] = float(class_a[weight].median())
            row[f"{weight}_median_{target_b}"] = float(class_b[weight].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["target_a", "target_b"]
    ).reset_index(drop=True)


def _competition_summary(combined: pd.DataFrame) -> pd.DataFrame:
    subset = combined.loc[combined["assigned_target"].isin(FOCUS_TARGETS)].copy()
    subset = subset.loc[subset["second_target"].isin(FOCUS_TARGETS)].copy()
    if subset.empty:
        return pd.DataFrame()
    pair_names = []
    for a, b in zip(subset["assigned_target"], subset["second_target"], strict=True):
        left, right = sorted((a, b))
        pair_names.append(f"{left}__{right}")
    subset["pair"] = pair_names
    rows: list[dict[str, Any]] = []
    for pair_name, group in subset.groupby("pair"):
        rows.append(
            {
                "pair": pair_name,
                "n_points": int(len(group)),
                "median_objective_gap": float(group["objective_gap"].median()),
                "q10_objective_gap": float(group["objective_gap"].quantile(0.10)),
                "dominant_assigned_target": str(group["assigned_target"].value_counts().idxmax()),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_points", "median_objective_gap"], ascending=[False, True])


def _compute_pca(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrix = np.log10(df[WEIGHT_COLS].to_numpy(dtype=float))
    scaled = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(scaled)
    out = df.copy()
    out["pca1"] = coords[:, 0]
    out["pca2"] = coords[:, 1]
    return out, {"explained_variance_ratio": pca.explained_variance_ratio_.tolist(), "components": pca.components_.tolist()}


def _plot_weight_pca(stable: pd.DataFrame, output_dir: Path) -> None:
    subset = stable.loc[stable["assigned_target"].isin(FOCUS_TARGETS)].copy()
    embedded, _ = _compute_pca(subset)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for target in FOCUS_TARGETS:
        group = embedded.loc[embedded["assigned_target"].eq(target)]
        if group.empty:
            continue
        ax.scatter(group["pca1"], group["pca2"], s=16, alpha=0.6, color=TARGET_COLORS[target], label=target, edgecolors="none")
    ax.set_xlabel("pca1")
    ax.set_ylabel("pca2")
    ax.set_title("Weight-only PCA of stable regions")
    _style_legend(ax)
    _save_fig(fig, output_dir / "focused_weight_pca.png")


def _plot_weight_axes(stable: pd.DataFrame, output_dir: Path) -> None:
    subset = stable.loc[stable["assigned_target"].isin(FOCUS_TARGETS)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.2))
    configs = [
        ("w_ff_0", "w_ff_1", "FF initial weights"),
        ("w_fb_0", "w_fb_1", "FB initial weights"),
        ("w_lat_0", "w_lat_1", "LAT initial weights"),
        ("w_pv_lat_0", "w_pv_lat_1", "PV->LAT initial weights"),
    ]
    for ax, (x, y, title) in zip(axes.flat, configs, strict=True):
        for target in FOCUS_TARGETS:
            group = subset.loc[subset["assigned_target"].eq(target)]
            if group.empty:
                continue
            ax.scatter(group[x], group[y], s=14, alpha=0.5, color=TARGET_COLORS[target], label=target, edgecolors="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
    _style_legend(axes[0, 0])
    _save_fig(fig, output_dir / "focused_weight_axes.png")


def _plot_response_axes(stable: pd.DataFrame, output_dir: Path) -> None:
    subset = stable.loc[stable["assigned_target"].isin(FOCUS_TARGETS)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.0))
    configs = [
        ("full_familiar_naive", "occlusion_familiar_naive", "Familiar naive: full vs occluded"),
        ("full_familiar_expert", "occlusion_familiar_expert", "Familiar expert: full vs occluded"),
        ("full_novel_naive", "occlusion_novel_naive", "Novel naive: full vs occluded"),
        ("full_novel_expert", "occlusion_novel_expert", "Novel expert: full vs occluded"),
    ]
    for ax, (x, y, title) in zip(axes.flat, configs, strict=True):
        for target in FOCUS_TARGETS:
            group = subset.loc[subset["assigned_target"].eq(target)]
            if group.empty:
                continue
            ax.scatter(group[x], group[y], s=14, alpha=0.5, color=TARGET_COLORS[target], label=target, edgecolors="none")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        ax.axhline(0.025, color="black", linewidth=0.8, alpha=0.25)
        ax.axvline(0.025, color="black", linewidth=0.8, alpha=0.25)
    _style_legend(axes[0, 0])
    _save_fig(fig, output_dir / "focused_response_axes.png")


def _plot_transition_axes(stable: pd.DataFrame, output_dir: Path) -> None:
    subset = stable.loc[stable["assigned_target"].isin(FOCUS_TARGETS)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))
    configs = [
        ("familiar_naive_state", "familiar_expert_state", "Familiar transition: naive vs expert"),
        ("novel_naive_state", "novel_expert_state", "Novel transition: naive vs expert"),
    ]
    for ax, (x, y, title) in zip(axes.flat, configs, strict=True):
        for target in FOCUS_TARGETS:
            group = subset.loc[subset["assigned_target"].eq(target)]
            if group.empty:
                continue
            ax.scatter(group[x], group[y], s=16, alpha=0.55, color=TARGET_COLORS[target], label=target, edgecolors="none")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.25)
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.25)
    _style_legend(axes[0])
    _save_fig(fig, output_dir / "focused_transition_axes.png")


def _write_report(
    out_dir: Path,
    *,
    reference_table: pd.DataFrame,
    core_ranges: pd.DataFrame,
    stable_medians: pd.DataFrame,
    pairwise_diffs: pd.DataFrame,
    competitions: pd.DataFrame,
    pca_meta: dict[str, Any],
) -> None:
    un_un = reference_table.loc[reference_table["target"].eq("un_un"), SUMMARY_COLS].iloc[0]
    un_fb = reference_table.loc[reference_table["target"].eq("un_FB"), SUMMARY_COLS].iloc[0]
    identical = bool(np.allclose(un_un.to_numpy(dtype=float), un_fb.to_numpy(dtype=float), atol=1e-10, rtol=0.0))

    lines = [
        "# Focused Weight Analysis",
        "",
        "This report ignores expert2 and focuses on the weight-only exploration, using only the naive and expert1 summaries for familiar/novel and full/occluded responses.",
        "",
        "## What The Old `FB_FB vs FF_FB...` Plots Were",
        "",
        "Those plots were not parameter-range plots.",
        "They came from the later learning-rate script and showed a straight interpolation path between one sampled point from class A and one sampled point from class B.",
        "The left panel tracked how a summary statistic changed along that interpolation, and the right panel showed which class the interpolated point got assigned to.",
        "So `FB_FB vs FF_FB_broad` meant: take one stable `FB_FB` point, take one stable `FF_FB_broad` point, interpolate between them in parameter space, and see where the class label flips.",
        "That can be useful for probing boundaries, but it is not the main visualization if the goal is simply to understand class-defining parameter ranges.",
        "",
        "## un_un vs un_FB",
        "",
    ]
    if identical:
        lines.append("`un_un` and `un_FB` are not separable in the current naive/expert1, full/occluded summary. Their eight reference summary values are exactly identical in the current setup.")
    else:
        lines.append("`un_un` and `un_FB` differ in the current summary, but only weakly.")
    lines.extend(
        [
            "",
            "Reference summary values are saved in `focused_reference_summaries.csv`.",
            "",
            "## PCA",
            "",
            f"PCA explained variance ratio on log10 weights: `{pca_meta['explained_variance_ratio'][0]:.3f}`, `{pca_meta['explained_variance_ratio'][1]:.3f}`.",
            "",
            "## Validated core ranges",
            "",
        ]
    )
    for row in core_ranges.to_dict(orient="records"):
        lines.append(
            f"- `{row['target']}`: familiar=`{row['familiar_transition']}`, novel=`{row['novel_transition']}`, "
            f"`w_ff=({row['w_ff_0_low']:.4g}-{row['w_ff_0_high']:.4g}, {row['w_ff_1_low']:.4g}-{row['w_ff_1_high']:.4g})`, "
            f"`w_fb=({row['w_fb_0_low']:.4g}-{row['w_fb_0_high']:.4g}, {row['w_fb_1_low']:.4g}-{row['w_fb_1_high']:.4g})`"
        )
    lines.extend(["", "## Main differentiators", ""])
    for row in pairwise_diffs.to_dict(orient="records"):
        lines.append(
            f"- `{row['target_a']}` vs `{row['target_b']}`: largest median log10 differences are "
            f"`{row['top_param_1']}` ({row['top_abs_log10_median_diff_1']:.2f}), "
            f"`{row['top_param_2']}` ({row['top_abs_log10_median_diff_2']:.2f}), "
            f"`{row['top_param_3']}` ({row['top_abs_log10_median_diff_3']:.2f})."
        )
    lines.extend(["", "## Most active pairwise competitions", ""])
    for row in competitions.head(8).to_dict(orient="records"):
        lines.append(
            f"- `{row['pair'].replace('__', ' vs ')}`: n=`{row['n_points']}`, median objective gap=`{row['median_objective_gap']:.4f}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `FF_FB_broad` differs from the narrow classes mainly by keeping both FF weights elevated rather than collapsing one axis.",
            "- `FF_FB_narrow_familiar` and `FF_FB_narrow_novel` are mainly distinguished by which FF axis stays large: `w_ff_0` for familiar-narrow, `w_ff_1` for novel-narrow.",
            "- `FF_un` differs from the FF->FB classes mostly by pushing both FB weights toward the floor while keeping both FF weights high.",
            "- `un_FB` sits near the low-FF, low-FB regime. In the current summary, there is no separate `un_un` region next to it.",
        ]
    )
    (out_dir / "focused_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_focus_analysis(
    results_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    settings: SweepSettings = SweepSettings(),
) -> dict[str, Any]:
    results_path = Path(results_dir)
    out_dir = Path(output_dir) if output_dir is not None else results_path / "focused_weights"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = _load_frames(results_path)
    reference_table = _reference_table(settings)
    core_ranges = _focused_core_ranges(frames["core"])
    medians = _stable_medians(frames["stable"])
    pairwise_diffs = _pairwise_weight_differences(frames["stable"])
    competitions = _competition_summary(frames["combined"])
    pca_source = frames["stable"].loc[frames["stable"]["assigned_target"].isin(FOCUS_TARGETS)].copy()
    _, pca_meta = _compute_pca(pca_source)

    reference_table.to_csv(out_dir / "focused_reference_summaries.csv", index=False)
    core_ranges.to_csv(out_dir / "focused_core_ranges.csv", index=False)
    medians.to_csv(out_dir / "focused_stable_medians.csv", index=False)
    pairwise_diffs.to_csv(out_dir / "pairwise_weight_differences.csv", index=False)
    competitions.to_csv(out_dir / "pairwise_competition_summary.csv", index=False)
    with (out_dir / "focused_pca_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(pca_meta, handle, indent=2)

    plot_dir = out_dir / "plots"
    _plot_weight_pca(frames["stable"], plot_dir)
    _plot_weight_axes(frames["stable"], plot_dir)
    _plot_response_axes(frames["stable"], plot_dir)
    _plot_transition_axes(frames["stable"], plot_dir)
    _write_report(
        out_dir,
        reference_table=reference_table,
        core_ranges=core_ranges,
        stable_medians=medians,
        pairwise_diffs=pairwise_diffs,
        competitions=competitions,
        pca_meta=pca_meta,
    )

    return {
        "reference_table": reference_table,
        "core_ranges": core_ranges,
        "medians": medians,
        "pairwise_diffs": pairwise_diffs,
        "competitions": competitions,
        "output_dir": out_dir,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce a focused weight-only report and PCA plots from a parameter-region search."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("context_contrasting/sbi/results/2026-04-30_parameter_regions_v2"),
        help="Existing weight-only region-search results directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the focused outputs. Defaults to <results-dir>/focused_weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_focus_analysis(args.results_dir, output_dir=args.output_dir)
    print(result["pairwise_diffs"].to_string(index=False))


if __name__ == "__main__":
    main()
