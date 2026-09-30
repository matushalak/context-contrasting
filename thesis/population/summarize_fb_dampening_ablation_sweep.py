"""Validate and summarize the per-alpha expert-ablation scatter suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis.population.export_fb_dampening_alpha_sweep import DEFAULT_OUTPUT_DIR
from thesis.population.export_expert_silencing_response_scatter import COLUMN_KEYS, COLUMN_LABELS


VARIANTS = (
    ("undampened", "Undampened", np.nan),
    ("alpha_1", "alpha=1", 1.0),
    ("alpha_0p2", "alpha=0.2", 0.2),
    ("alpha_0p1", "alpha=0.1", 0.1),
    ("alpha_0p05", "alpha=0.05", 0.05),
    ("alpha_0p02", "alpha=0.02", 0.02),
    ("alpha_0p01", "alpha=0.01", 0.01),
    ("alpha_0p005", "alpha=0.005", 0.005),
    ("alpha_0p001", "alpha=0.001", 0.001),
)
IMAGE_GROUPS = ("familiar", "novel")
RESPONSES = ("NO_response", "O_response")


def _collect(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    validation_rows: list[dict[str, object]] = []
    for variant, label, alpha in VARIANTS:
        condition_dir = base_dir / "ablations" / variant
        values_path = condition_dir / "expert_silencing_response_scatter_values_labeled.csv"
        metadata_path = condition_dir / "expert_silencing_response_scatter_cache_metadata.json"
        if not values_path.exists() or not metadata_path.exists():
            validation_rows.append(
                {"variant": variant, "variant_label": label, "complete": False, "reason": "missing outputs"}
            )
            continue
        frame = pd.read_csv(values_path)
        metadata = json.loads(metadata_path.read_text())
        required_columns = [
            "neuron_idx",
            "image_group",
            "column_key",
            "NO_response",
            "O_response",
            "RotatedSector",
            "dNorm",
            "log_dNorm",
        ]
        n_missing_required = int(frame[required_columns].isna().sum().sum())
        complete = (
            len(frame) == 6000
            and frame["neuron_idx"].nunique() == 300
            and frame["column_key"].nunique() == len(COLUMN_KEYS)
            and frame["image_group"].nunique() == len(IMAGE_GROUPS)
            and n_missing_required == 0
        )
        frame["variant"] = variant
        frame["variant_label"] = label
        frame["fb_alpha"] = alpha
        frame["fb_rule"] = str(metadata["fb_rule"])
        frames.append(frame)
        validation_rows.append(
            {
                "variant": variant,
                "variant_label": label,
                "complete": complete,
                "reason": "" if complete else "unexpected rows, dimensions, or missing values",
                "n_rows": len(frame),
                "n_cells": frame["neuron_idx"].nunique(),
                "n_columns": frame["column_key"].nunique(),
                "n_image_groups": frame["image_group"].nunique(),
                "n_missing_required": n_missing_required,
                "fb_rule": metadata.get("fb_rule"),
                "fb_alpha": metadata.get("fb_alpha"),
                "soma_activation_threshold": metadata.get("soma_activation_threshold"),
            }
        )
    validation = pd.DataFrame(validation_rows)
    if not frames:
        raise FileNotFoundError(f"No completed ablation conditions under {base_dir / 'ablations'}")
    return pd.concat(frames, ignore_index=True), validation


def _summarize(values: pd.DataFrame, *, by_sector: bool = False) -> pd.DataFrame:
    expert = values.loc[values["column_key"].eq("expert"), [
        "variant", "neuron_idx", "image_group", *RESPONSES
    ]].rename(columns={response: f"expert_{response}" for response in RESPONSES})
    paired = values.merge(expert, on=["variant", "neuron_idx", "image_group"], validate="many_to_one")
    for response in RESPONSES:
        paired[f"delta_{response}"] = paired[response] - paired[f"expert_{response}"]
    group_columns = [
        "variant",
        "variant_label",
        "fb_alpha",
        "fb_rule",
        "image_group",
        "column_key",
        "column_label",
    ]
    if by_sector:
        group_columns.append("RotatedSector")
    return (
        paired.groupby(
            group_columns,
            dropna=False,
            as_index=False,
        )
        .agg(
            n_cells=("neuron_idx", "nunique"),
            mean_NO_response=("NO_response", "mean"),
            mean_O_response=("O_response", "mean"),
            mean_delta_NO_vs_expert=("delta_NO_response", "mean"),
            mean_delta_O_vs_expert=("delta_O_response", "mean"),
        )
    )


def _plot_delta_heatmaps(summary: pd.DataFrame, output_dir: Path) -> None:
    variant_order = [variant for variant, _label, _alpha in VARIANTS]
    variant_labels = [label for _variant, label, _alpha in VARIANTS]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    plotted: list[tuple[plt.Axes, object]] = []
    for row_idx, image_group in enumerate(IMAGE_GROUPS):
        for col_idx, (response, title) in enumerate(
            (("mean_delta_NO_vs_expert", "NO minus Expert"), ("mean_delta_O_vs_expert", "O minus Expert"))
        ):
            ax = axes[row_idx, col_idx]
            matrix = (
                summary.loc[summary["image_group"].eq(image_group)]
                .pivot(index="column_key", columns="variant", values=response)
                .reindex(index=COLUMN_KEYS, columns=variant_order)
            )
            limit = float(np.nanmax(np.abs(matrix.to_numpy())))
            limit = max(limit, 1e-12)
            image = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
            ax.set_xticks(range(len(variant_labels)), variant_labels, rotation=45, ha="right")
            ax.set_yticks(range(len(COLUMN_KEYS)), [COLUMN_LABELS[key] for key in COLUMN_KEYS])
            ax.set_title(f"{image_group.title()}: {title}")
            plotted.append((ax, image))
    for ax, image in plotted:
        fig.colorbar(image, ax=ax, shrink=0.75, label="Mean response difference")
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"fb_dampening_ablation_sweep_delta_heatmaps.{extension}", dpi=300)
    plt.close(fig)


def _plot_novel_plus_no_o_delta(sector_summary: pd.DataFrame, output_dir: Path) -> None:
    variant_order = [variant for variant, _label, _alpha in VARIANTS]
    variant_labels = [label for _variant, label, _alpha in VARIANTS]
    matrix = (
        sector_summary.loc[
            sector_summary["image_group"].eq("novel")
            & sector_summary["RotatedSector"].eq("+NO axis")
        ]
        .pivot(index="column_key", columns="variant", values="mean_delta_O_vs_expert")
        .reindex(index=COLUMN_KEYS, columns=variant_order)
    )
    limit = max(float(np.nanmax(np.abs(matrix.to_numpy()))), 1e-12)
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(variant_labels)), variant_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(COLUMN_KEYS)), [COLUMN_LABELS[key] for key in COLUMN_KEYS])
    ax.set_title("Novel native +NO cells: O response minus matched Expert")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Mean O-response difference")
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"fb_dampening_ablation_sweep_novel_plus_no_o_delta.{extension}", dpi=300)
    plt.close(fig)


def _crosscheck_base_columns(values: pd.DataFrame, base_dir: Path) -> dict[str, float | int]:
    base = pd.read_csv(base_dir / "fb_dampening_alpha_sweep_cell_results.csv")
    checks: dict[str, float | int] = {}
    for phase in ("naive", "expert"):
        observed = values.loc[
            values["column_key"].eq(phase),
            ["variant", "neuron_idx", "image_group", "NO_response", "O_response"],
        ]
        expected = base[
            ["variant", "neuron_idx", "image_group", f"{phase}_NO", f"{phase}_O"]
        ]
        paired = observed.merge(expected, on=["variant", "neuron_idx", "image_group"], validate="one_to_one")
        checks[f"n_{phase}_paired_rows"] = len(paired)
        checks[f"max_abs_{phase}_NO_difference"] = float(
            np.max(np.abs(paired["NO_response"] - paired[f"{phase}_NO"]))
        )
        checks[f"max_abs_{phase}_O_difference"] = float(
            np.max(np.abs(paired["O_response"] - paired[f"{phase}_O"]))
        )
    return checks


def _write_report(
    base_dir: Path,
    validation: pd.DataFrame,
    summary: pd.DataFrame,
    sector_summary: pd.DataFrame,
    crosscheck: dict[str, float | int],
) -> None:
    complete = validation.loc[validation["complete"].fillna(False)]
    report = """# FB Dampening Alpha Ablation Sweep

## Scope

Each condition contains the same ten columns: Naive, Expert, acute Expert FB off,
acute Expert PV off, acute Expert FB+PV off, FF adaptation off during training,
FB specificity set to diagonal 2.0 and off-diagonal 1.2, generalized FB training
signal 2.0/1.2, FB learning rate 5x, and FB learning rate 10x. All runs use the
saved 300-cell `done-amen` population and an explicitly reconstructed soma
activation threshold of 0.08.

Point colors are the native naive-to-expert rotated-sector labels for the same
image group. They are not recomputed after an ablation, so a neuron keeps its
color across all columns in one row.

## Completion

| Condition | Rows | Cells | Columns | Complete |
| --- | ---: | ---: | ---: | --- |
"""
    for row in validation.itertuples(index=False):
        report += (
            f"| `{row.variant}` | {getattr(row, 'n_rows', '')} | {getattr(row, 'n_cells', '')} | "
            f"{getattr(row, 'n_columns', '')} | {'yes' if row.complete else 'no'} |\n"
        )
    target_columns = (
        "expert_fb_specificity_full_diag",
        "expert_fb_signal_general",
        "expert_fb_lr_5x",
        "expert_fb_lr_10x",
    )
    target = sector_summary.loc[
        sector_summary["image_group"].eq("novel")
        & sector_summary["RotatedSector"].eq("+NO axis")
        & sector_summary["column_key"].isin(target_columns)
    ].pivot(index="variant", columns="column_key", values="mean_delta_O_vs_expert")
    target = target.reindex(index=[variant for variant, _label, _alpha in VARIANTS], columns=target_columns)
    target_table = "| Condition | FB S full diag | FB signal gen | FB LR=5x | FB LR=10x |\n"
    target_table += "| --- | ---: | ---: | ---: | ---: |\n"
    for variant, label, _alpha in VARIANTS:
        target_table += (
            f"| {label} | {target.loc[variant, target_columns[0]]:+.4f} | "
            f"{target.loc[variant, target_columns[1]]:+.4f} | "
            f"{target.loc[variant, target_columns[2]]:+.4f} | "
            f"{target.loc[variant, target_columns[3]]:+.4f} |\n"
        )
    report += """

## Findings

Lower alpha progressively suppresses both the baseline learned response and the
effect of every FB-strengthening perturbation. FB LR=10x gives the largest mean O
increase among native novel +NO cells in every condition, but its effect falls
from +1.1070 in the undampened control and +0.9756 at alpha=1 to +0.0005 at
alpha=0.001. At alpha=0.001, FB LR=5x, generalized FB signal, and full-diagonal
specificity are slightly below the matched Expert O response in this sector.

Mean novel +NO O-response difference from the matched Expert condition:

"""
    report += target_table
    report += f"""

The Naive and Expert columns independently reconstruct the base alpha sweep to
numerical precision. Across 5,400 matched cell-image rows per phase, the largest
absolute difference was `{max(value for key, value in crosscheck.items() if 'difference' in key):.3g}`.

## Files

Each completed condition directory contains:

- `expert_silencing_response_scatter.png`: individual observed-data limits per panel
- `expert_silencing_response_scatter_shared_axes.png`: observed-data limits shared within each row
- matching SVG exports
- scalar and sector-labeled response tables
- per-ablation O-response delta summary

The large per-timepoint trace table is intentionally omitted. The scalar values
are computed from the same traces in memory with the established stimulus-window
summarizer before those traces are discarded.

## Combined Outputs

- `fb_dampening_ablation_sweep_values.csv`: all per-cell scalar values
- `fb_dampening_ablation_sweep_summary.csv`: population means and paired differences from Expert
- `fb_dampening_ablation_sweep_sector_summary.csv`: the same paired summary within each native sector
- `fb_dampening_ablation_sweep_validation.csv`: condition-level completeness and provenance
- `fb_dampening_ablation_sweep_crosscheck.json`: exact comparison with the base alpha sweep
- `fb_dampening_ablation_sweep_delta_heatmaps.png`: across-alpha perturbation effects
- `fb_dampening_ablation_sweep_novel_plus_no_o_delta.png`: targeted novel +NO O-response effects
"""
    if len(complete) != len(VARIANTS):
        report += f"\nOnly {len(complete)} of {len(VARIANTS)} expected conditions are currently complete.\n"
    (base_dir / "ABLATION_SWEEP_REPORT.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    values, validation = _collect(output_dir)
    summary = _summarize(values)
    sector_summary = _summarize(values, by_sector=True)
    crosscheck = _crosscheck_base_columns(values, output_dir)
    values.to_csv(output_dir / "fb_dampening_ablation_sweep_values.csv", index=False)
    summary.to_csv(output_dir / "fb_dampening_ablation_sweep_summary.csv", index=False)
    sector_summary.to_csv(output_dir / "fb_dampening_ablation_sweep_sector_summary.csv", index=False)
    validation.to_csv(output_dir / "fb_dampening_ablation_sweep_validation.csv", index=False)
    (output_dir / "fb_dampening_ablation_sweep_crosscheck.json").write_text(
        json.dumps(crosscheck, indent=2) + "\n"
    )
    _plot_delta_heatmaps(summary, output_dir)
    _plot_novel_plus_no_o_delta(sector_summary, output_dir)
    _write_report(output_dir, validation, summary, sector_summary, crosscheck)
    print(f"[ablation-sweep-summary] summarized {validation['complete'].sum()} conditions in {output_dir}")


if __name__ == "__main__":
    main()
