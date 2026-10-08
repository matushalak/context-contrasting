# Six mean-field transitions

Six independent models: familiar and novel, each with +NO, +O, and -NO transitions. Initial conditions are ordinary YAML parameters; simulation does not load a population, calculate sectors, fit parameters, or import the old paper code.

## Files to edit

| File | Purpose |
|---|---|
| `configs/familiar_*.yaml`, `configs/novel_*.yaml` | One complete parameter set per model |
| `configs/protocol.yaml` | Trial timing, counts, contextual breadth, and training-order seed |
| `model.py` | One PyC/PV pair: dynamics and three plastic synapses |
| `simulate.py` | Naive probes, familiar training, expert probes and acute silencing |
| `export_ground_truth.py` | Rebuild the empirical plotting CSV from the saved Pre/Post tables |
| `plot.py` | Traces and transition vectors, reading only the two CSVs |
| `robustness.py` | One-at-a-time parameter and initial-weight sweeps |

```bash
uv run python -m paper_v2.export_ground_truth
uv run python -m paper_v2.simulate
uv run python -m paper_v2.plot
uv run python -m paper_v2.robustness
```

Plots exclude the empirical -O sector by default. To include it in the overall empirical transition arrows and average traces, use:

```bash
uv run python -m paper_v2.plot --include-minus-o
```

The transition-vector figures are written under `figures/transition_vectors/` and include the expert and all three acute-silencing conditions, each calculated from naive to that condition. Cosines compare the seed-averaged model vector with each individual empirical neuron vector, then average those similarities. `--cosine-errorbars` adds SD across neurons without changing that comparison. The overall point compares the two overall average vectors.

Simulation defaults to **20 measurement trials and 10 seeds (0 through 9) per config**. `--seeds` supplies an explicit, reproducible seed set, overriding the seeds inside individual YAML configs. The same seed set is used for every config, so familiar/novel comparisons do not rely on different lucky seeds. The output YAML preserves the original cell configs and records the overriding `seeds` list; every CSV row carries the actual `seed` used.

Each config/seed pair is an independent process task. The default `--jobs -1` uses all available CPU cores; use, for example, `--jobs 8` to leave resources free. PyTorch runs one thread per task, avoiding nested thread oversubscription. Each task sets its own RNG seed, so changing worker count does not change results. Parallelism uses processes, not threads, because PyTorch's RNG state is process-global.

```bash
uv run python -m paper_v2.simulate --test-trials 20 --seeds 0 1 2 3 4 5 6 7 8 9
uv run python -m paper_v2.plot --uncertainty both --cosine-errorbars
```

Plotting produces **both SD and SEM** versions by default (`--uncertainty sd` or `sem` selects one). Unsuffixed trace files show SD; `_sem` trace files show SEM. Vector filenames end in `_sd` or `_sem`. Images are averaged within a seed/neuron before calculating uncertainty. Model uncertainty is across seeds, empirical uncertainty across neurons, not across images or individual time samples. Weighted model traces first combine the sector templates within each seed, then calculate SD/SEM across those complete weighted traces. This retains between-sector covariance. A legacy single-run CSV still plots, but has no estimable model SD/SEM band.

Ellipses use the covariance of seed/neuron transition vectors; SEM divides covariance by the number of observations. These are one-SD/one-SEM ellipses, not 95% confidence regions. Trace bands likewise use sample SD (ddof=1) or SD/sqrt(n). Seed-to-seed spread includes training-noise and finite-test-trial variation; using 10 seeds measures variability, rather than making every individual seed robust.

Run from the repository root. Simulation writes `model.csv` and an exact input snapshot, `model.yaml`. Plotting writes PNG/SVG figures to `figures/`. Robustness writes `robustness/results.csv`, its input snapshot `config.yaml`, and PNG/SVG figures: retention heatmaps for scalars, weight coordinates, and weight families; the thirty most sensitive model/parameter combinations; the baseline transition vectors; and one ΔNO/ΔO sweep curve per parameter under `curves/`. `--plot-only` redraws the figures from a saved run. To try a small robustness sweep:

```bash
uv run python -m paper_v2.robustness --parameter lr_fb --factors 0.8 1.2 \
  --test-trials 2 --output /tmp/mean-field-robustness-smoke
```

## Model and protocol

The PV rate follows its sensory drive and fixed PyC-to-PV input. Somatic input is

```text
(apical_gain * FF_drive + apical_drive) / (1 + divisive_gain * lateral_drive)
    + baseline_noise - adaptation
```

Rates are leaky integrators with rectification and a ceiling of one. The three learning rules are FF anti-Hebbian adaptation scaled by the slow activity accumulator, undampened FB strengthening, and Hebbian PV-to-PyC strengthening. PV input weights and PyC-to-PV weights remain fixed. The baseline noise stays outside divisive normalization.

The default protocol has 400 steps per trial, 100 steps per second, seven training presentations of each familiar image, and 20 measurement trials per seed. A trial has three seconds of baseline and one second of stimulus. Generalized context is a unit-sum permutation of `[1, 0.6, 0.6] / 2.2`, used for both learning and testing. Training uses a fixed shuffled familiar-image sequence; the ensemble seeds change the model's training and measurement noise, not this presentation order. The original five-trial probe is replayed before training to keep learning noise independent of the measurement trial count.

Displayed traces span -1 to +3 seconds and average trials. All conditions of a model use its measured naive baseline mean and sample SD, pooled across its six naive image/response probes over the one second before stimulus onset. There is no normalization floor. Familiar results average images 1 and 2; novel results use image 3. As in the saved final checkpoint, transition response means include samples from stimulus onset through its offset (0 to 1 second, inclusive).

## The two CSV inputs

- `ground_truth.csv`: fixed empirical sector memberships, individual-neuron traces, and individual-neuron transition vectors for the familiar and novel expert groups. It retains 197 neuron/group transition rows and 297,600 trace rows across the +NO, +O, -NO, and -O sectors. Images are averaged within neuron before averaging neurons.
- `model.csv`: 36,000 trial-averaged trace samples and six transition rows **per seed**, including naive, expert, FB silencing, PV silencing, and combined silencing. The default 10-seed run contains 360,000 trace rows and 60 transition rows.

Both use `record_type` (`trace` or `transition`), `source`, `sector`, and `observation_id`. Trace rows have `condition_key`, `image_id`, `response_type`, `time_seconds`, and `response`. Transition rows have `naive_NO`, `target_NO`, `naive_O`, `target_O`, `delta_NO`, and `delta_O`. Unused fields are empty. The model also records `seed`, trial counts and baseline statistics. Its `observation_id` identifies the configuration; `(observation_id, seed)` identifies a replicate. Biological neuron IDs are preserved; `sector` is a fixed group label, not recalculated by plotting.

`ground_truth.csv` is generated by `export_ground_truth.py` from `thesis/data_analysis/transitions_post.csv` and `transitions_post_traces.csv`. For each source separately, responses are averaged across images within a neuron, the Pre-to-Post vector is calculated, cells with transition magnitude at most 0.3 are excluded, and the remaining cells are assigned to a directional sector. That fixed `(source, neuron_idx)` membership is then joined back to every saved per-image trace. `Full`/`Occl` become `NO`/`O`, and `Pre`/`Post` become `naive`/`expert`. Running the export command above reproduces the checked-in CSV.

The checked-in CSV retains -O so this choice does not require rebuilding data. By default, plotting removes those rows once after loading, before computing the empirical overall transition and average traces. `--include-minus-o` keeps them for those two summaries. The sector-specific trace figure always filters out -O immediately before calling `plot_traces`. Vector plots show empirical sector means, their empirical SD covariance ellipses, and the direct neuron-average "overall transition". The model overall arrow and traces average its three available sector templates using the corresponding empirical cell fractions, normalized across +NO, +O, and -NO separately for familiar and novel data.

## Robustness

Every continuous scalar and complete weight family is tested at `0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 5` times its YAML value, leaving everything else fixed. Scaled family weights are clipped to the allowed `[0, 1]`, and the PyC and PV leak rates, which are fractions per step, are capped at 1. Each individual weight coordinate is instead set to the absolute values `0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1` and to its YAML value. Seeds and binary context masks are not perturbed. A point is retained if its directional sector is unchanged and its transition magnitude is at least 25% of its baseline magnitude; degenerate zero-SD probes are invalid.

The robustness protocol preserves the previous robustness method: ten measurement trials by default, separate fixed training and measurement seeds, normalization over the full three-second native-image naive baseline, and response means over the stimulus samples only. Thus robustness baselines differ slightly from the main figure baselines. This distinction is explicit in the two short simulation routines rather than hidden in exporter dependencies.

## Provenance and validation

These initial conditions descend from sector-averaged parameters, followed by the existing tuning, rounding, FF scaling, and noise adjustments. They are now explicit starting parameters, not recomputed averages. The source is the latest final-figure checkpoint:

context_contrasting/paper/done-amen/figures/organized_sector_results/mean_field_fb_generalization_direct_signal/single_familiar_plus_no_image1_tuned_novel_tuned_rounded_2sf_wff_family_0p75_noise_familiar_plusno_0p075_novel_plusno_0p2_other_0p2_generalized_test_all_cells_50_trials_measured_baseline_std

The empirical inputs and sector calculation are shared with `thesis/data_analysis/export_real_mean_field_style.py`; the separate task/act analysis is outside this study. No data extraction is needed to plot the saved CSVs.

Validation of this migration: a complete 50-trial simulation matched all 36,000 native trace samples and six transition vectors from the checkpoint to below 3e-15 absolute error. The model also matched the original implementation exactly over 400 learning steps for every configuration. The robustness design contains 326–329 points per model (1,971 total, including the six baselines). In the saved full run, 1,804 of 1,965 perturbed points (91.8%) are retained: 92.9% of scalar, 90.1% of weight-coordinate, and 91.8% of weight-family points.

All six two-trial robustness baselines were also compared directly with the previous robustness routine, matching to below 1e-15 absolute error. The six ten-trial baseline vectors and all 816 perturbed points shared with the earlier `robustness_oat_extensive` screen reproduce it to below 6e-15; retention fractions differ only because the grids differ.
