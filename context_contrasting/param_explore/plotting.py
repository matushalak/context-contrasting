from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from scipy.stats import gaussian_kde

from context_contrasting.param_explore.common import (
    CONTEXT_LABELS,
    CONTEXT_MARKERS,
    PARAMETER_PLOT_GROUPS,
    TRANSITION_COLORS,
    TRANSITION_LABEL_ORDER,
)

MAX_SCATTER_POINTS_PER_GROUP = 180


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _transition_legend_handles(labels: list[str]) -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="none", markersize=7, markerfacecolor=color, markeredgecolor="none", label=label)
        for label in labels
        for color in [TRANSITION_COLORS[label]]
    ]


def _context_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker=marker, linestyle="none", markersize=7, markerfacecolor="black", markeredgecolor="black", label=label)
        for mode, marker in CONTEXT_MARKERS.items()
        for label in [CONTEXT_LABELS[mode]]
    ]


def _plot_kde(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    log_axes: bool,
) -> None:
    if len(x) < 20:
        return
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return

    try:
        fill_alphas = (0.18, 0.30, 0.45, 0.62, 0.80)
        level_quantiles = np.array([0.55, 0.72, 0.84, 0.93, 0.98, 0.995], dtype=float)
        rgba = to_rgba(color)
        if log_axes:
            values = np.vstack([np.log10(x), np.log10(y)])
            kde = gaussian_kde(values)
            grid_x = np.linspace(values[0].min(), values[0].max(), 70)
            grid_y = np.linspace(values[1].min(), values[1].max(), 70)
            xx, yy = np.meshgrid(grid_x, grid_y)
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            positive = zz[zz > 0]
            if len(positive) == 0:
                return
            levels = np.unique(np.quantile(positive, level_quantiles))
            if len(levels) < 2:
                return
            fill_colors = [(rgba[0], rgba[1], rgba[2], alpha) for alpha in fill_alphas[: len(levels) - 1]]
            ax.contourf(
                10.0 ** xx,
                10.0 ** yy,
                zz,
                levels=levels,
                colors=fill_colors,
                antialiased=True,
            )
            ax.contour(
                10.0 ** xx,
                10.0 ** yy,
                zz,
                levels=levels[1:],
                colors=[color],
                linewidths=0.9,
                alpha=0.95,
            )
        else:
            values = np.vstack([x, y])
            kde = gaussian_kde(values)
            grid_x = np.linspace(x.min(), x.max(), 90)
            grid_y = np.linspace(y.min(), y.max(), 90)
            xx, yy = np.meshgrid(grid_x, grid_y)
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            positive = zz[zz > 0]
            if len(positive) == 0:
                return
            levels = np.unique(np.quantile(positive, level_quantiles))
            if len(levels) < 2:
                return
            fill_colors = [(rgba[0], rgba[1], rgba[2], alpha) for alpha in fill_alphas[: len(levels) - 1]]
            ax.contourf(
                xx,
                yy,
                zz,
                levels=levels,
                colors=fill_colors,
                antialiased=True,
            )
            ax.contour(
                xx,
                yy,
                zz,
                levels=levels[1:],
                colors=[color],
                linewidths=0.9,
                alpha=0.95,
            )
    except Exception:
        return


def _observed_labels(frame: pd.DataFrame, label_col: str) -> list[str]:
    observed = set(frame[label_col].dropna().unique().tolist())
    return [label for label in TRANSITION_LABEL_ORDER if label in observed]


def _sample_group(subset: pd.DataFrame) -> pd.DataFrame:
    if len(subset) <= MAX_SCATTER_POINTS_PER_GROUP:
        return subset
    return subset.sample(n=MAX_SCATTER_POINTS_PER_GROUP, random_state=0)


def plot_transition_space(
    frame: pd.DataFrame,
    *,
    image_type: str,
    output_path: Path,
) -> None:
    label_col = f"{image_type}_transition_label"
    x_col = f"{image_type}_transition_point_x"
    y_col = f"{image_type}_transition_point_y"
    labels = _observed_labels(frame, label_col)

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    for label in labels:
        subset = frame.loc[frame[label_col].eq(label)]
        if subset.empty:
            continue
        _plot_kde(
            ax,
            subset[x_col].to_numpy(dtype=float),
            subset[y_col].to_numpy(dtype=float),
            color=TRANSITION_COLORS[label],
            log_axes=False,
        )

    for mode, marker in CONTEXT_MARKERS.items():
        subset = frame.loc[
            frame["receives_context_familiar"].eq(mode[0])
            & frame["receives_context_novel"].eq(mode[1])
        ]
        if subset.empty:
            continue
        subset = pd.concat(
            [_sample_group(group) for _, group in subset.groupby(label_col)],
            ignore_index=True,
        )
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=10,
            alpha=0.14,
            c=subset[label_col].map(TRANSITION_COLORS),
            marker=marker,
            edgecolors="none",
        )

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.25)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.25)
    ax.set_xlabel("naive_state")
    ax.set_ylabel("expert_state")
    ax.set_title(f"{image_type.capitalize()} transition plane")

    transition_legend = ax.legend(
        handles=_transition_legend_handles(labels),
        title="Transition region",
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


def plot_parameter_panels(
    frame: pd.DataFrame,
    *,
    image_type: str,
    output_path: Path,
) -> None:
    label_col = f"{image_type}_transition_label"
    labels = _observed_labels(frame, label_col)
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.2))

    for ax, (x_col, y_col, title) in zip(axes.flat, PARAMETER_PLOT_GROUPS, strict=True):
        for label in labels:
            subset = frame.loc[frame[label_col].eq(label)]
            if subset.empty:
                continue
            _plot_kde(
                ax,
                subset[x_col].to_numpy(dtype=float),
                subset[y_col].to_numpy(dtype=float),
                color=TRANSITION_COLORS[label],
                log_axes=True,
            )

        for mode, marker in CONTEXT_MARKERS.items():
            subset = frame.loc[
                frame["receives_context_familiar"].eq(mode[0])
                & frame["receives_context_novel"].eq(mode[1])
            ]
            if subset.empty:
                continue
            subset = pd.concat(
                [_sample_group(group) for _, group in subset.groupby(label_col)],
                ignore_index=True,
            )
            ax.scatter(
                subset[x_col],
                subset[y_col],
                s=7,
                alpha=0.10,
                c=subset[label_col].map(TRANSITION_COLORS),
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
        title=f"{image_type} transition",
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
    fig.suptitle(f"{image_type.capitalize()} transition regions over initial weights", y=1.02)
    _save_fig(fig, output_path)
