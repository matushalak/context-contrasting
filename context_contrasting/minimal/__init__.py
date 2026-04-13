# author: Matúš Halák (@matushalak)
"""Minimal single-neuron contextual contrasting model."""
import os

PLOTSDIR = os.path.join(os.path.dirname(__file__), 'plots')
if not os.path.exists(PLOTSDIR):
    os.makedirs(PLOTSDIR)

PLOT_ALL_PANELS_DIR = os.path.join(PLOTSDIR, "all_panels")
PLOT_PANEL_A_DIR = os.path.join(PLOTSDIR, "panel_A")
PLOT_NOVEL_ONLY_DIR = os.path.join(PLOTSDIR, "novel_only")
PLOT_TRANSITION_PANELS_DIR = os.path.join(PLOTSDIR, "transition_panels")
PLOT_ABLATIONS_DIR = os.path.join(PLOTSDIR, "ablations")

for _path in (
    PLOT_ALL_PANELS_DIR,
    PLOT_PANEL_A_DIR,
    PLOT_NOVEL_ONLY_DIR,
    PLOT_TRANSITION_PANELS_DIR,
    PLOT_ABLATIONS_DIR,
):
    os.makedirs(_path, exist_ok=True)
