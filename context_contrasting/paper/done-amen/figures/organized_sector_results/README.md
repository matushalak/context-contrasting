# Organized Sector Results

Generated from the internally consistent `done-amen` modeled population run.

## Provenance

- Source run directory: `context_contrasting/paper/done-amen`
- Population configs: `sampled_configs.json`
- Population summaries: `summaries/aggregate_familiar_summary.csv`, `summaries/aggregate_novel_summary.csv`
- Run metadata: `metadata.json`
- Organized on: 2026-08-26

`done-amen-final` was not used as the source for this package because its saved summaries were regenerated later and no longer match the sector counts stored in its metadata.

## Contents

- `no_o_response_scatter/`
  - `expert_silencing_response_scatter.png`
  - `expert_silencing_response_scatter.svg`
  - `expert_silencing_response_scatter_values.csv`
  - `expert_silencing_response_scatter_values_labeled.csv`
- `mean_field_sector_cells/`
  - `mean_field_sector_cells_by_image.png`
  - `mean_field_sector_cells_by_image.svg`
  - `mean_field_sector_cells_pooled_familiar_novel.png`
  - `mean_field_sector_cells_pooled_familiar_novel.svg`
  - `mean_field_sector_cells_pooled_familiar_trace_only.png`
  - `mean_field_sector_cells_pooled_familiar_trace_only.svg`
  - `mean_field_sector_cells_pooled_novel_trace_only.png`
  - `mean_field_sector_cells_pooled_novel_trace_only.svg`
  - `mean_field_sector_parameters.csv`
  - `mean_field_sector_traces_by_image.csv`
  - `mean_field_sector_traces_by_image_all_parameter_sets.csv`
  - `mean_field_sector_traces_pooled_familiar_novel.csv`
- `familiar_sector_novel_transfer/`
  - `familiar_sector_mean_parameter_novel_transfer.png`
  - `familiar_sector_mean_parameter_novel_transfer.svg`
  - `familiar_sector_mean_parameter_cells.csv`
  - `familiar_sector_mean_parameter_traces_all_conditions.csv`
  - `familiar_sector_mean_parameter_traces_novel.csv`
  - `familiar_sector_source_trace_average_by_image.png`
  - `familiar_sector_source_trace_average_by_image.svg`
  - `familiar_sector_source_trace_average_novel_transfer.png`
  - `familiar_sector_source_trace_average_novel_transfer.svg`
  - `familiar_sector_source_trace_average_pooled_familiar_novel.png`
  - `familiar_sector_source_trace_average_pooled_familiar_novel.svg`
  - `familiar_sector_source_neuron_trace_averages_by_image.csv`
  - `familiar_sector_source_neuron_trace_averages_novel.csv`
  - `familiar_sector_source_neuron_trace_averages_pooled_familiar_novel.csv`
  - `familiar_sector_source_neuron_counts.csv`
  - `analysis_summary.json`
- `real_data_sector_per_image_trace_only/`
  - `sector_per_image_familiar_pooled_examples_sem.csv`
  - `sector_per_image_familiar_pooled_examples_sem_trace_only.png`
  - `sector_per_image_familiar_pooled_examples_sem_trace_only.svg`
  - `sector_per_image_novel_pooled_examples_sem.csv`
  - `sector_per_image_novel_pooled_examples_sem_trace_only.png`
  - `sector_per_image_novel_pooled_examples_sem_trace_only.svg`
- `selection_manifests/`
  - `real_data_sector_per_image_trace_membership.csv`
  - `real_data_sector_per_image_trace_membership_summary.csv`
  - `real_data_sector_per_image_trace_membership_by_image.csv`
  - `model_mean_field_parameter_membership.csv`
  - `model_mean_field_parameter_membership_summary.csv`
  - `selection_manifest_index.json`

## Source Cell Counts

Mean-field sector cells:

| sector source | sector | source sector definition | n source cells |
| --- | --- | --- | ---: |
| familiar | +NO axis | +NO axis | 16 |
| familiar | +O axis | +O axis | 48 |
| familiar | -NO axis | -NO axis | 100 |
| novel | +NO axis | +NO axis | 58 |
| novel | +O axis | diagonal +NO/+O | 8 |
| novel | -NO axis | -NO axis | 40 |

Familiar-sector transfer source groups:

| familiar sector | n source cells |
| --- | ---: |
| +NO axis | 16 |
| +O axis | 48 |
| -NO axis | 100 |

Real-data sector-per-image pooled source counts:

| image group | sector | n cells | n images |
| --- | --- | ---: | ---: |
| familiar | +NO axis | 61 | 4 |
| familiar | +O axis | 63 | 4 |
| familiar | -NO axis | 107 | 4 |
| familiar | -O axis | 33 | 4 |
| novel | +NO axis | 67 | 2 |
| novel | +O axis | 26 | 2 |
| novel | -NO axis | 44 | 2 |
| novel | -O axis | 24 | 2 |

The exact neuron IDs used for these averages are recorded in `selection_manifests/`. For real-data plots, membership is per image: sectors are assigned separately for each neuron-image transition, then pooled. For model mean-field cells, membership is per saved modeled neuron: parameters are averaged across the listed `sampled_configs.json` rows.

Population width composition from `metadata.json`:

| width class | n cells | fraction |
| --- | ---: | ---: |
| broad | 190 | 0.6333 |
| narrow | 110 | 0.3667 |

## Learning Rates And Sampling

This run used fixed shared learning-rate scalars:

| parameter | value |
| --- | ---: |
| `lr_ff` | 0.00775 |
| `lr_fb` | 0.000325 |
| `lr_lat` | 0.015 |
| `lr_pv` | 0.0 |

The metadata lists these as `fixed_scalars`, and all saved sampled configs share the same values. The `base_shared_learning_rates` are exactly twice those values, with `learning_rate_reference_steps = 200`; the saved model population itself uses the fixed shared values above.

This run did use the slow FF activity accumulation variable. In the saved configs, `use_ff_activity_accumulator = true` for every cell, with `ff_accumulator_alpha_factor = 0.05`, `ff_accumulator_power = 2.0`, and `ff_accumulator_scale = 2000.0`. In the model this creates `ff_activity_accumulator = EMA(alpha = pyc_decay * ff_accumulator_alpha_factor)`, so with `pyc_decay = 0.05` the accumulator alpha is `0.0025`. During FF anti-Hebbian updates, the fixed FF plasticity scale is replaced by `ff_accumulator_scale * accumulator ** ff_accumulator_power`.

The broad/narrow composition was sampled from fixed `data-like` template weights and then realized as fixed transition counts in the saved 300-cell population. It was not adaptively adjusted by the resulting sector counts.
