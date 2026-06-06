# Model Scatter

This folder is a separate analysis for treating noisy samples around the canonical
`minimal2/config_s.py` transition configs as a model analogue of the real-neuron
scatter plots in `context_contrasting/data_analysis`.

Run it from the repository root in the project Conda environment:

```bash
conda activate context-contrasting
python -m context_contrasting.model_scatter.run_model_scatter
```

By default this samples 640 noisy configs total, allocated across canonical
transitions with a data-like bias toward unresponsive initial conditions. It runs
the minimal2 model with shared test/training stimuli, writes a
`transition_table.csv` with the same columns as the real transition CSVs, and
saves figures under `context_contrasting/model_scatter/outputs/figures`.

Canonical-only run:

```bash
python -m context_contrasting.model_scatter.run_model_scatter \
  --canonical-only \
  --skip-by-transition \
  --output-dir context_contrasting/model_scatter/outputs_canonical
```

No-LR-variation comparison:

```bash
python -m context_contrasting.model_scatter.run_model_scatter \
  --freeze-learning-rates \
  --output-dir context_contrasting/model_scatter/outputs_zscore_data_like_fixed_lrs
```

Useful smaller smoke run:

```bash
python -m context_contrasting.model_scatter.run_model_scatter \
  --samples-per-transition 3 \
  --n-steps-per-phase 40 \
  --n-jobs 1 \
  --output-dir context_contrasting/model_scatter/outputs_smoke
```

Outputs:

- `transition_table.csv`: real-data-compatible table with `Naive` and `Expert`
  stages plus `Full` and `Occl` image types.
- `sample_responses.csv`: one scalar response per sampled config, phase,
  condition, and trace type.
- `sampled_config_parameters.csv`: flattened sampled parameters.
- `sampled_configs.json`: full sampled config dictionaries for reproducibility.
- `summaries/*.csv`: mean transition summaries and rotated-sector fractions.
- `figures/*.png` and `figures/*.svg`: the same 3x3 plotting scheme used by
  `data_analysis/transitions_helpers.py`.

Sampling notes:

- Weight/init means are perturbed additively and clipped nonnegative.
- Learning rates and positive scale parameters are perturbed in log space.
- Each plotted noisy dot gets an independent RNG stream spawned from the run
  seed; dots do not share a perturbation draw.
- Model scatter responses are baseline z-scored by default to match the real
  transition plots. Use `--raw-responses` to recover raw activations.
- Learning rates are sampled by default. Use `--freeze-learning-rates` to keep
  `lr_ff`, `lr_fb`, `lr_lat`, and `lr_pv` fixed at their canonical values while
  still sampling weights and other scalar hyperparameters.
- Noisy runs use `--transition-sampling data-like` by default so the aggregate
  naive cloud is weighted toward unresponsive starts, with fewer O responders
  and more NO than O responders. Use `--transition-sampling equal` for equal
  counts around every canonical transition.
- Boolean and categorical settings, including `receives_context` and learning
  rule names, are kept fixed.
- Init `sigma` is set to zero by default so the external config sampling is the
  source of variability. Use `--keep-init-sigma` to preserve canonical init
  sigmas.
- Plot axes use the 99th percentile by default because a small number of noisy
  sampled configs can produce runaway model responses. Use
  `--limit-percentile 100` for full-extents plots. Points outside the aggregate
  plot limits are written to
  `summaries/aggregate_points_outside_plot_limits.csv`.
- Each summary plot also exports a standalone unit-circle sector legend ending
  in `_sector_legend.png` and `_sector_legend.svg`.

Saved output folders:

- `outputs_zscore_data_like_vary_lrs`: z-scored data-like noisy run with learning
  rates varied.
- `outputs_zscore_data_like_fixed_lrs`: matched z-scored data-like noisy run with
  learning rates fixed.
- `outputs_canonical_zscore`: exact canonical configs, z-scored.
