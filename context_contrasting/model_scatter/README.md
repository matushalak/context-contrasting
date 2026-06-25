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
  --skip-center-panels \
  --output-dir context_contrasting/model_scatter/outputs_smoke
```

Fast tuning run:

```bash
python -m context_contrasting.model_scatter.run_model_scatter \
  --n-samples 250 \
  --n-jobs 10 \
  --skip-center-panels \
  --output-dir context_contrasting/model_scatter/outputs_final200 \
  --n-steps-per-phase 200 \
  --test-trials 2 \
  --training-trials 5
```

Fixed-PV mini variant:

```bash
python -m context_contrasting.model_scatter.run_model_scatter_mini \
  --n-samples 250 \
  --n-jobs 10 \
  --skip-center-panels \
  --output-dir context_contrasting/model_scatter/outputs_mini_final200 \
  --n-steps-per-phase 200 \
  --test-trials 2 \
  --training-trials 5
```

`run_model_scatter_mini.py` disables plasticity of `W_pv` and `w_pv_lat`, so
only `w_ff`, `w_fb`, and `w_lat` learn during familiar training. Its tuned
defaults pre-strengthen fixed PV feedforward tuning (`--pv-init-scale 1.5`) and
increase lateral learning (`--lat-lr-scale 2.0`); both remain command-line
controls for follow-up sweeps.

Canonical-only run:

```bash
python -m context_contrasting.model_scatter.run_model_scatter \
  --canonical-only \
  --output-dir context_contrasting/model_scatter/outputs_canonical
```

The main tuning levers are at the top of `run_model_scatter.py`:

- `TRANSITIONS`: transition sampling proportions plus per-transition initial
  weight centers. Center-only `weight_init(...)` specs are fixed exactly at that
  center; specs that also provide relative noise, a noise floor, and bounds are
  sampled.
- Only FF and FB initial weights are sampled. LAT, PV-lateral, and PV feedforward
  initial weights are fixed at their transition-template centers. FF/FB samples
  use independent Gaussian noise per weight element, with the transition's
  relative noise, floor, and bounds.
- `SCALAR_NOISE`: perturbation ranges for `apical_drive_threshold`,
  `apical_gain_strength`, and `baseline_drive_sigma`; other scalar parameters are
  fixed per transition template unless explicitly set by the template.
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
- `--n-steps-per-phase`: simulation steps per test trial (`200` by default; the
  generated stimulus occupies the final quarter).
- `--test-trials`: continuous probe repeats per stimulus (`2` by default), using
  the same repeated-trial protocol as before.
- `--training-trials`: familiar-training repeats (`5` by default, matching the
  canonical trace experiment).
- `--transition-sampling data-like|equal`: random weighted transition draws or
  equal samples per canonical transition.
- Responses are always z-scored to the naive probe baseline, so adapted cells are
  not inflated by a tiny post-training baseline variance and the naive->expert
  shift reflects only the change in evoked response.
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
  `canonical_panels/`: grouped `transitions_FAM` / `transitions_NOV` response
  matrices plus the naive->expert transition panel for configs built from the
  noise-free sampler centers (`center_panels`) and for the raw `config_s`
  canonical examples (`canonical_panels`). These panels use the same
  `--n-steps-per-phase`, `--test-trials`, and `--training-trials` settings as
  the main run, include the no-feedback and no-LAT traces, and z-score each
  displayed response to the dynamically selected pre-stimulus panel window. The
  displayed panel window is 1/4 baseline ITI, 1/4 stimulus, and 1/2 post-stimulus
  ITI, even though the generated test protocol remains continuous repeated
  trials. These runs are parallelised across transitions (use `--n-jobs`).
- `--export-panels`: also export each scatter's individual panels as separate
  images.
- `--scalar-noise-multiplier`: global multiplier on the `SCALAR_NOISE` jitter
  widths (`1.75` by default).
- Aggregate figure axis limits match the real-data figures: a single shared
  response/shift frame across the familiar and novel panels via
  `transitions_helpers.compute_response_limits`/`compute_shift_limits`.

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
- `canonical_panels/`: the same panel set for the raw `config_s` canonical
  examples, unless `--skip-center-panels` is passed.
