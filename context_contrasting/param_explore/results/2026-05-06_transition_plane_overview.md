This refactor replaces the old class-boundary workflow with a transition-plane workflow centered on the continuous 2-tuple `(naive_state, expert_state)`.

Code:

- [transition_types.py](/Users/matushalak/Documents/context-contrasting/context_contrasting/minimal/transition_types.py:111) defines the new continuous transition-plane helpers.
- [common.py](/Users/matushalak/Documents/context-contrasting/context_contrasting/param_explore/common.py:1) runs the fixed-learning-rate simulations and evaluates all 12 initial weights plus the 4 `receives_context` modes.
- [run_sampling.py](/Users/matushalak/Documents/context-contrasting/context_contrasting/param_explore/run_sampling.py:1) is the forward explorer for `grid` and `sobol`.
- [run_sbi.py](/Users/matushalak/Documents/context-contrasting/context_contrasting/param_explore/run_sbi.py:1) is the inverse explorer for SBI.
- [plotting.py](/Users/matushalak/Documents/context-contrasting/context_contrasting/param_explore/plotting.py:1) makes the transition-plane plots and the 6 weight-pair panels.

What changed:

- Only the initial weights are explored:
  `w_ff`, `w_fb`, `w_lat`, `w_pv_lat`, and `W_pv`, for a total of 12 parameters.
- Learning rates are fixed to the `broad` config.
- Only naive and first expert timepoints are simulated.
- `receives_context` varies over all 4 boolean combinations and is shown with markers:
  `o=(T,T)`, `^=(T,F)`, `s=(F,T)`, `*=(F,F)`.
- Point color is the observed transition label.
- KDE contours are drawn before the point cloud.

Reference-plane consequence:

- In this naive/expert-only projection, several old named configs collapse to the same point.
- With `n_steps=60`, `n_trials=2`, `tail_window=30`, the reference table shows:
  - `un_un`, `un_FF`, and `un_FB` all map to `(0, 0)` for familiar and novel.
  - `FF_FB_broad` maps to `(0.0425, 0.0000)` for familiar and `(0.0436, 0.0486)` for novel.
  - `FB_FB` maps to negative familiar and novel coordinates, as expected for the FB quadrant.

Runs performed:

- Grid baseline:
  [2026-05-06_transition_plane_grid](/Users/matushalak/Documents/context-contrasting/context_contrasting/param_explore/results/2026-05-06_transition_plane_grid)
  - 2-level full grid over all 12 weights
  - 4096 weight points, 16384 simulations after the 4 context modes
  - useful as a coarse corner baseline only
- Sobol interior coverage:
  [2026-05-06_transition_plane_sobol](/Users/matushalak/Documents/context-contrasting/context_contrasting/param_explore/results/2026-05-06_transition_plane_sobol)
  - 2048 joint 12D Sobol samples
  - 8192 simulations after the 4 context modes
  - this is the most useful forward coverage of the new space
- SBI inverse example:
  [2026-05-06_transition_plane_sbi_ff_fb_broad](/Users/matushalak/Documents/context-contrasting/context_contrasting/param_explore/results/2026-05-06_transition_plane_sbi_ff_fb_broad)
  - target: `FF_FB_broad`
  - target mode: `joint`
  - 512 prior simulations per context-conditioned posterior
  - 128 posterior samples per context

Top-level observations from the new plane:

- The Sobol familiar plane is dominated by three macroscopic regions:
  `unresponsive -> unresponsive`, `FF -> FF`, and a smaller `FF -> FB` / `FB -> FB` sector.
- The familiar `FF -> FF` region grows strongly with larger `w_ff` values, especially along the `w_ff_0` axis.
- Negative-plane occupancy is much rarer than positive-plane occupancy under the fixed broad learning rates.
- The new projection is continuous and useful, but it also discards distinctions that depended on later expert states. That is why several old named classes now coincide.

Most useful files:

- Grid familiar plane:
  [familiar_transition_plane.png](/Users/matushalak/Documents/context-contrasting/context_contrasting/param_explore/results/2026-05-06_transition_plane_grid/plots/familiar_transition_plane.png)
- Sobol familiar plane:
  [familiar_transition_plane.png](/Users/matushalak/Documents/context-contrasting/context_contrasting/param_explore/results/2026-05-06_transition_plane_sobol/plots/familiar_transition_plane.png)
- Sobol familiar weight panels:
  [familiar_parameter_panels.png](/Users/matushalak/Documents/context-contrasting/context_contrasting/param_explore/results/2026-05-06_transition_plane_sobol/plots/familiar_parameter_panels.png)
- Sobol sample table:
  [samples.csv](/Users/matushalak/Documents/context-contrasting/context_contrasting/param_explore/results/2026-05-06_transition_plane_sobol/samples.csv)
- Reference transition points:
  [reference_transition_points.csv](/Users/matushalak/Documents/context-contrasting/context_contrasting/param_explore/results/2026-05-06_transition_plane_sobol/reference_transition_points.csv)
