from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from context_contrasting.pc_comparison.run_pc_comparison import (
    DEFAULT_OUTPUT_DIR,
    _draw_response_axis,
    _draw_vector_axis,
    _robust_response_limits,
    _robust_shift_limits,
    _save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export separate PPE/NPE thesis panels.")
    parser.add_argument("--pc-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "thesis_exports")
    return parser.parse_args()


def _load_summaries(output_dir: Path) -> dict[str, dict[str, pd.DataFrame]]:
    summaries = {
        circuit: {
            group: pd.read_csv(output_dir / "summaries" / f"{circuit.lower()}_{group}_summary.csv")
            for group in ("familiar", "novel")
        }
        for circuit in ("PPE", "NPE")
    }
    return summaries


def _export_scatter_panel(
    summary: pd.DataFrame,
    *,
    circuit: str,
    output_dir: Path,
    response_lims: list[float],
    formats: tuple[str, ...],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.35))
    _draw_response_axis(
        axes[0],
        summary,
        stage="pre",
        title="Familiar naive",
        response_lims=response_lims,
        use_class_outlines=False,
    )
    _draw_response_axis(
        axes[1],
        summary,
        stage="target",
        title="Familiar expert",
        response_lims=response_lims,
        use_class_outlines=False,
    )
    fig.suptitle(circuit, fontsize=9)
    fig.tight_layout()
    _save_figure(
        fig,
        output_dir / f"{circuit.lower()}_familiar_scatter_panel",
        formats=formats,
    )


def _export_vector_panel(
    summary: pd.DataFrame,
    *,
    circuit: str,
    image_group: str,
    output_dir: Path,
    shift_lims: list[float],
    formats: tuple[str, ...],
) -> None:
    fig, ax = plt.subplots(figsize=(2.35, 2.35))
    _draw_vector_axis(
        ax,
        summary,
        title=f"{circuit} {image_group}",
        shift_lims=shift_lims,
        show_legend=True,
    )
    _save_figure(
        fig,
        output_dir / f"{circuit.lower()}_{image_group}_vector_panel",
        formats=formats,
    )


def main() -> None:
    args = parse_args()
    summaries = _load_summaries(args.pc_output_dir)
    all_summaries = [summary for circuit in summaries.values() for summary in circuit.values()]
    response_lims = _robust_response_limits(all_summaries, hi_percentile=99.5)
    shift_lims = _robust_shift_limits(all_summaries, hi_percentile=99.5)
    formats = ("png", "svg", "eps")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for circuit in ("PPE", "NPE"):
        familiar = summaries[circuit]["familiar"]
        _export_scatter_panel(
            familiar,
            circuit=circuit,
            output_dir=args.output_dir,
            response_lims=response_lims,
            formats=formats,
        )
        _export_vector_panel(
            familiar,
            circuit=circuit,
            image_group="familiar",
            output_dir=args.output_dir,
            shift_lims=shift_lims,
            formats=formats,
        )
        _export_vector_panel(
            summaries[circuit]["novel"],
            circuit=circuit,
            image_group="novel",
            output_dir=args.output_dir,
            shift_lims=shift_lims,
            formats=formats,
        )

    metadata = json.loads((args.pc_output_dir / "metadata.json").read_text())
    (args.output_dir / "panel_metadata.json").write_text(
        json.dumps(
            {
                "source": str(args.pc_output_dir),
                "circuits": ["PPE", "NPE"],
                "scatter_image_groups": ["familiar"],
                "vector_image_groups": ["familiar", "novel"],
                "formats": list(formats),
                "response_limits": response_lims,
                "shift_limits": shift_lims,
                "pc_model": metadata["pc_model"],
                "convergence_calibration": metadata["convergence_calibration"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
