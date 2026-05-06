This package explores the minimal-model transition landscape in the 2D plane
`(naive_state, expert_state)`.

The exploration is weight-only:

- The 12 swept parameters are the initial weights
  `w_ff`, `w_fb`, `w_lat`, `w_pv_lat`, and `W_pv`.
- The learning rates stay fixed to the `broad` configuration.
- `receives_context` is varied over all four boolean combinations.
- Only the naive and first expert timepoints are simulated.
- Response summaries are computed over the stimulus-active window within each
  trial, not over intertrial baseline periods.

Entry points:

- `python -m context_contrasting.param_explore.run_sampling --method grid ...`
- `python -m context_contrasting.param_explore.run_sampling --method sobol ...`
- `python -m context_contrasting.param_explore.run_sbi ...`
- `python -m context_contrasting.param_explore.analyze_basins --results-dir ...`

Outputs:

- `samples.csv` or `training_samples.csv` / `posterior_samples.csv`
- separate familiar and novel transition-plane plots
- separate familiar and novel 6-panel weight plots
- optional `basins/` subdirectory with stable-core basin assignments, summaries,
  and boundary reports

Plot conventions:

- point color = observed transition label
- point marker = `receives_context` mode
- KDE contours are drawn before points for each observed transition label
