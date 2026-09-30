# Six mean-field transitions

Six independent models: familiar and novel, each with +NO, +O, and -NO transitions. Initial conditions are ordinary YAML parameters; simulation does not load a population, calculate sectors, fit parameters, or import the old paper code.

## Files to edit

| File | Purpose |
|---|---|
| `configs/familiar_*.yaml`, `configs/novel_*.yaml` | One complete parameter set per model |
| `configs/protocol.yaml` | Trial timing, counts, contextual breadth, and training-order seed |
| `model.py` | One PyC/PV pair: dynamics and three plastic synapses |
| `simulate.py` | Naive probes, familiar training, expert probes and acute silencing |
| `plot.py` | Traces and transition vectors, reading only the two CSVs |
| `robustness.py` | One-at-a-time parameter and initial-weight sweeps |

```bash
uv run python -m paper_v2.simulate
uv run python -m paper_v2.plot
uv run python -m paper_v2.robustness
```

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

The default protocol has 400 steps per trial, 100 steps per second, seven training presentations of each familiar image, and 50 measurement trials. A trial has three seconds of baseline and one second of stimulus. Generalized context is a unit-sum permutation of `[1, 0.6, 0.6] / 2.2`, used for both learning and testing. Training uses a shuffled familiar-image sequence. The original five-trial probe is replayed before training to keep learning noise independent of the measurement trial count.

Displayed traces span -1 to +3 seconds and average trials. All conditions of a model use its measured naive baseline mean and sample SD, pooled across its six naive image/response probes over the one second before stimulus onset. There is no normalization floor. Familiar results average images 1 and 2; novel results use image 3. As in the saved final checkpoint, transition response means include samples from stimulus onset through its offset (0 to 1 second, inclusive).

## The two CSV inputs

- `ground_truth.csv`: fixed empirical sector memberships, individual-neuron traces, and individual-neuron transition vectors for the six expert groups. It retains 173 neuron/group transition rows and 261,888 trace rows. Images are averaged within neuron before averaging neurons.
- `model.csv`: 36,000 trial-averaged trace samples and six transition rows from the simulation, including naive, expert, FB silencing, PV silencing, and combined silencing.

Both use `record_type` (`trace` or `transition`), `source`, `sector`, and `observation_id`. Trace rows have `condition_key`, `image_id`, `response_type`, `time_seconds`, and `response`. Transition rows have `naive_NO`, `target_NO`, `naive_O`, `target_O`, `delta_NO`, and `delta_O`. Unused fields are empty. The model also records trial counts and baseline statistics. Biological neuron IDs are preserved; `sector` is a fixed group label, not recalculated by plotting.

Vector plots show means and empirical SD/SEM covariance ellipses. SEM divides the covariance by the number of neurons; it does not account for clustering within animals. This retains the previous figure convention.

## Robustness

Every continuous scalar and complete weight family is tested at `0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 5` times its YAML value, leaving everything else fixed. Scaled family weights are clipped to the allowed `[0, 1]`, and the PyC and PV leak rates, which are fractions per step, are capped at 1. Each individual weight coordinate is instead set to the absolute values `0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1` and to its YAML value. Seeds and binary context masks are not perturbed. A point is retained if its directional sector is unchanged and its transition magnitude is at least 25% of its baseline magnitude; degenerate zero-SD probes are invalid.

The robustness protocol preserves the previous robustness method: ten measurement trials by default, separate fixed training and measurement seeds, normalization over the full three-second native-image naive baseline, and response means over the stimulus samples only. Thus robustness baselines differ slightly from the main figure baselines. This distinction is explicit in the two short simulation routines rather than hidden in exporter dependencies.

## Provenance and validation

These initial conditions descend from sector-averaged parameters, followed by the existing tuning, rounding, FF scaling, and noise adjustments. They are now explicit starting parameters, not recomputed averages. The source is the latest final-figure checkpoint:

context_contrasting/paper/done-amen/figures/organized_sector_results/mean_field_fb_generalization_direct_signal/single_familiar_plus_no_image1_tuned_novel_tuned_rounded_2sf_wff_family_0p75_noise_familiar_plusno_0p075_novel_plusno_0p2_other_0p2_generalized_test_all_cells_50_trials_measured_baseline_std

`ground_truth.csv` preserves the expert rows from `output/publication_plot_data_v1/data_figure_data.csv`, using the saved memberships from the original empirical export; the separate task/act analysis is outside this six-transition study. That exporter remains in `thesis/data_analysis/export_real_mean_field_style.py`, with the raw empirical tables beside it. No data extraction is needed to plot the saved study.

Validation of this migration: a complete 50-trial simulation matched all 36,000 native trace samples and six transition vectors from the checkpoint to below 3e-15 absolute error. The model also matched the original implementation exactly over 400 learning steps for every configuration. The robustness design contains 326–329 points per model (1,971 total, including the six baselines). In the saved full run, 1,804 of 1,965 perturbed points (91.8%) are retained: 92.9% of scalar, 90.1% of weight-coordinate, and 91.8% of weight-family points.

All six two-trial robustness baselines were also compared directly with the previous robustness routine, matching to below 1e-15 absolute error. The six ten-trial baseline vectors and all 816 perturbed points shared with the earlier `robustness_oat_extensive` screen reproduce it to below 6e-15; retention fractions differ only because the grids differ.
