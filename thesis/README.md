# Thesis

The original study, separated from the six-model paper study.

- `population/`: the committed population model, sampling, scatterplots, highlighted traces, and visualization helpers.
- `data_analysis/`: the original empirical analyses and notebooks.
- `pc_comparison/`: matched predictive-coding models and comparisons.
- `circuit/`, `mini_network/`, `simple/`: earlier original model implementations.

The population core is taken from commit `668fa10`. Its dampened-FB default and sampling parameters are preserved. The existing later ablation exporter and alpha-sweep scripts are included; the model additionally accepts the undampened rule those comparisons require. No mean-field fitting, sector-parameter extraction, or mean-field sweep exporters were copied here.

## Population study

From the repository root:

```bash
uv run python -m thesis.population.run_model_scatter \
  --n-samples 250 --n-steps-per-phase 300 \
  --training-trials 7 --test-trials 5 --n-jobs 10 \
  --plot-center-panels --plot-by-transition --export-panels
```

Outputs go to `thesis/results/population/`. The original command-line options and plotting variants remain available via `--help`.

## Ablations and alpha sweep

```bash
uv run python -m thesis.population.export_expert_silencing_response_scatter \
  --run-dir thesis/results/population --patent-splits

uv run python -m thesis.population.export_fb_dampening_alpha_sweep \
  --run-dir thesis/results/population --alphas 1 0.2 0.1 0.05 0.02 0.01 0.005 0.001
```

Omitting `--run-dir` reads the saved original population in `context_contrasting/paper/done-amen/`. New outputs go to `thesis/results/`; the reference run is not rewritten. The ablation exporter includes acute FB/PV silencing, FF-adaptation removal, feedback-generalization changes, feedback learning-rate changes, and color/grayscale views.

The alpha command measures matched populations across dampening strengths. To render all ablations at a particular alpha, use the ablation exporter with `--fb-rule dampened-anti-Hebbian --fb-alpha 0.2 --output-dir thesis/results/ablations_alpha_0p2`. Use a distinct output directory for each alpha.

For the original predictive-coding comparison:

```bash
uv run python -m thesis.pc_comparison.run_pc_comparison
```

Migration checks passed with 12 sampled cells (300 steps, 7 training and 5 test trials), including the optional center panels, per-transition plots, and panel exports. The ablation exporter passed a one-cell run including color/grayscale splits. The alpha sweep passed three cells with undampened, alpha=1, and alpha=0.2 conditions. The retained predictive-coding tests also pass. These are execution checks, not new population-level scientific results.
