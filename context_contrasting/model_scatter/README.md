# Model Scatter

This folder is a separate analysis for treating noisy samples around the canonical
`minimal2/config_s.py` transition configs as a model analogue of the real-neuron
scatter plots in `context_contrasting/data_analysis`.

Run it from the repository root in the project Conda environment:

```bash
conda activate context-contrasting
python -m context_contrasting.model_scatter.run_model_scatter
```

By default this samples 96 noisy configs per canonical transition equivalent,
allocated across canonical transitions with a data-like bias toward weak and
unresponsive initial conditions. Probe phases use the same `--n-steps-per-phase`
duration as training so response windows and trace exports stay aligned; only
the number of probe repeats can be lowered with `--test-trials`. The runner writes a
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

Initial-weights-only comparison:

```bash
python -m context_contrasting.model_scatter.run_model_scatter \
  --initial-weights-only \
  --output-dir context_contrasting/model_scatter/outputs_zscore_data_like_wide_initial_weights_only
```

Useful smaller smoke run:

```bash
python -m context_contrasting.model_scatter.run_model_scatter \
  --samples-per-transition 1 \
  --test-trials 1 \
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
- `summaries/*.csv`: mean transition summaries, rotated-sector fractions, and
  paper-style NO/O separation-index diagnostics.
- `figures/*.png` and `figures/*.svg`: the same 3x3 plotting scheme used by
  `data_analysis/transitions_helpers.py`.

Sampling notes:

- Weight/init means are perturbed additively and clipped nonnegative.
- Initial weights use the low-response `naive-cloud` sampler by default so most
  samples begin near the origin, with a smaller high-NO tail for FF-origin
  cases. Use `--initial-condition-mode canonical-neighborhood` for diagnostic
  runs around the raw canonical prototypes.
- Positive scale parameters are perturbed in log space.
- Each plotted noisy dot gets an independent RNG stream spawned from the run
  seed; dots do not share a perturbation draw.
- Model scatter responses are baseline z-scored by default to match the real
  transition plots. Use `--raw-responses` to recover raw activations.
- Learning rates and neuronal decay time constants (`pyc_decay`, `pv_decay`) are
  fixed across samples. Other scalar hyperparameters can still be sampled unless
  `--initial-weights-only` is used.
- Narrow FF classes keep `FF_plasticity` enabled but use
  `ff_plasticity_scale` to reduce the effective FF anti-Hebbian update. Novel-only
  FF classes use a zero scale because training never presents the novel image;
  broad familiar FF classes use full-scale FF adaptation.
- Use `--initial-weights-only` to keep every scalar hyperparameter fixed at its
  canonical prototype value and sample only initial weight means.
- Noisy runs use `--transition-sampling data-like` by default so the aggregate
  naive cloud is weighted toward weak/unresponsive starts, familiar FF->FB
  switching, novel gain-only FF->FB, and a small `FF_un` tail for stronger naive
  NO responders. Use `--transition-sampling equal` for equal counts around every
  canonical transition.
- Model-scatter sector plots use `--threshold 0.3` by default, matching the
  current real-data transition analysis; raising this to 0.4 keeps more weak
  shifts in the central gray cloud without changing responses.
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
- `outputs_zscore_data_like_wide_vary_lrs`: wider, higher-sample z-scored
  data-like noisy run with learning rates varied.
- `outputs_zscore_data_like_wide_fixed_lrs`: matched wider run with learning
  rates fixed but other scalar hyperparameters varied.
- `outputs_zscore_data_like_wide_initial_weights_only`: matched wider run with
  only initial weight means varied.
- `outputs_canonical_zscore`: exact canonical configs, z-scored.
- `outputs_fixed_lrs_original_like_weak_ff_gain_v14_novel_gain`: intermediate
  fixed-dynamics run with the former separate weak FF gain class, FF->FB classes, naive-cloud initial
  conditions, and the tuned novel +NO branch.
- `outputs_fixed_lrs_no_extra_flags_fffb_v17`: current chronic-matching
  fixed-dynamics run using the original apical drive/gain equations, stronger
  FF->FB representation, and no `apical_drive_subtract_threshold` or
  `apical_gain_positive_only` model options.
- `outputs_fixed_lrs_no_extra_flags_fffb_v18`: current chronic-matching
  fixed-dynamics run with tuning-dependent FF plasticity scaling, explicit
  familiar-1/familiar-2/novel narrow FF classes, and reweighted data-like
  sampling. Aggregate familiar sectors are
  roughly `small delta` 33%, `-NO` 24%, `+O` 21%, `+NO` 12%; aggregate novel
  sectors are roughly `+NO` 29%, `small delta` 33%, `-O` 17%, `-NO` 12%,
  `+O` 8%.
- `outputs_fixed_lrs_no_extra_flags_fffb_v19_improved`: current improved
  fixed-dynamics run using the refined canonical anchors, `--threshold 0.3`,
  preserved training duration, and lower probe repeats (`--test-trials 3`).
  Aggregate familiar sectors are roughly `small delta` 36%, `-NO` 28%, `+O`
  21%, `+NO` 10%, `-O` 5%; aggregate novel sectors are roughly `small delta`
  44%, `+NO` 32%, `-NO` 10%, `-O` 7%, `+O` 7%.
