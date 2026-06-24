# Model Scatter

`run_model_scatter.py` samples noisy variants of the canonical
`minimal2/config_s.py` transition configs, runs the minimal circuit, exports a
real-data-compatible transition table, and passes that table to
`data_analysis/transitions_helpers.py` for the scatter plots.

Run from the repository root in the project environment:

```bash
python -m context_contrasting.model_scatter.run_model_scatter
```

Useful smoke run:

```bash
python -m context_contrasting.model_scatter.run_model_scatter \
  --n-samples 15 \
  --test-trials 1 \
  --training-trials 1 \
  --n-jobs 1 \
  --skip-by-transition \
  --output-dir context_contrasting/model_scatter/outputs_smoke
```

Canonical-only run:

```bash
python -m context_contrasting.model_scatter.run_model_scatter \
  --canonical-only \
  --skip-by-transition \
  --output-dir context_contrasting/model_scatter/outputs_canonical
```

The main tuning levers are at the top of `run_model_scatter.py`:

- `TRANSITIONS`: transition sampling proportions plus per-transition initial
  weight centers, relative noise, noise floors, and clipping bounds.
- `SCALAR_NOISE`: global scalar hyperparameter perturbation ranges.
- Learning rates are fixed globally across all transition types via
  `SHARED_LEARNING_RATES`; transition-specific learning-rate values are
  intentionally ignored by the sampler.
- `FF_PLASTICITY_BROAD` / `FF_PLASTICITY_NARROW` and `NARROW_TRANSITIONS`: the
  feedforward (anti-Hebbian) plasticity scale is a property of tuning width, not
  of the individual transition. Every broadly tuned transition shares one value
  (strong FF adaptation) and every narrowly tuned transition shares another
  (weak, but never 0 -- the pure-FF noLAT/noFB ablation of narrow cells still
  adapts a little). This is injected into each transition's `fix` after the
  `TRANSITIONS` table is defined, overriding any per-transition value.
- `--n-samples`: total cells to draw from the weighted transition mixture
  (`1200` by default, so the model distribution is smoother than the empirical
  sample).
- `--training-trials`: familiar-training repeats (`6` by default for the
  expert/task scatter regime).
- `--samples-per-transition`: compatibility override for old commands; draws
  `len(transitions) * value` samples.
- `--transition-sampling data-like|equal`: random weighted transition draws or
  equal samples per canonical transition.
- `--response-normalization naive|phase`: z-score post responses using the
  naive probe baseline by default, so adapted cells are not inflated by a tiny
  post-training baseline variance.
- `--zscore-std-floor`: denominator floor for model z-scores (`0.04` by
  default). Protects sectorization from tiny post-training baseline variances.
- `--threshold`: rotated-sector small-shift threshold (`0.3` by default, matching
  the `transitions>threshold` notebook so the model is sectorized exactly like
  the real data).
- `--plot-by-transition`: also save separate familiar/novel plots for each
  canonical transition; aggregate-only plots are the default.
- `--image-format png|svg|eps`: figure output format (`png` by default).
- `--axis-clip-percentile`: scale figure axes to this percentile of the response
  distribution (`99` by default) so a few extreme outliers fall outside the panel
  instead of compressing it; `100` reproduces the real-data notebook's exact
  min/max framing.
- `--skip-center-panels`: by default every run also writes `center_panels/` and
  `canonical_panels/`: the naive->expert transition panel (full + occlusion
  traces) for configs built from the noise-free sampler centers (`center_panels`)
  and for the raw `config_s` canonical examples (`canonical_panels`), so both can
  be sanity-checked against the model mechanism (the sampler centers can drift
  from what their names imply). Rendered with the canonical `n_steps_per_phase`
  (400) so the traces match `minimal2/plotsexperiment_s/transition_panels`. These
  runs are parallelised across transitions (use `--n-jobs`).
- `--no-overlay-examples`: by default the aggregate scatter highlights one point
  per transition for the `config_s` canonical examples (black stars, labelled)
  and the sampler centers (gray diamonds), each run through the full scatter
  pipeline; their positions are also written to `example_points.csv`.
- `--initial-condition-mode spec|canonical-neighborhood`: sample from the
  compact transition range table, or sample around the raw canonical config
  while still clipping to the transition bounds.
- `--initial-weights-only`: skip scalar hyperparameter perturbation.
- Aggregate figure axis limits match the real-data figures: a single shared
  response/shift frame across the familiar and novel panels via
  `transitions_helpers.compute_response_limits`/`compute_shift_limits`. The old
  `--response-limit-percentile` / `--shift-limit-percentile` / `--limit-percentile`
  flags are accepted but ignored.

Outputs:

- `transition_table.csv`: table compatible with the real transition plotting
  helpers.
- `sample_responses.csv`: response rows before transition-table projection.
- `sampled_config_parameters.csv` and `sampled_configs.json`: sampled parameter
  values for reproducibility.
- `metadata.json`: run settings and realized transition counts.
- `summaries/*.csv`: aggregate and optional by-transition sector summaries.
- `figures/*`: aggregate and optional by-transition scatter plots generated by
  `transitions_helpers.py`, in `--image-format` (PNG by default).
- `center_panels/`: the naive->expert transition panel, FAM/NOV response
  matrices, and their CSVs for the exact noise-free sampler centers, unless
  `--skip-center-panels` is passed.
