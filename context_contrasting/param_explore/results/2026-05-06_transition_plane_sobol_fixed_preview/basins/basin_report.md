Transition-plane basin analysis

Source samples: /Users/matushalak/Documents/context-contrasting/context_contrasting/param_explore/results/2026-05-06_transition_plane_sobol_fixed_preview

Definition
- Separate analysis per image type (`familiar`, `novel`) and per `receives_context` mode.
- Build a symmetric kNN graph in 12D `log10` weight space with `k=32`.
- Compute local purity = fraction of kNN neighbors with the same transition-plane tile.
- Stable core threshold: purity >= 0.80.
- Stable basin = connected component of stable-core points within one tile, size >= 12.

Sampling settings
- n_steps_per_phase = 100
- n_trials = 10
- tail_window = 25

Largest basins
- familiar|11|unresponsive -> FB|01: ctx=(T,T) | familiar | unresponsive -> FB | core=21 | purity_median=0.844
- familiar|10|unresponsive -> FB|01: ctx=(T,F) | familiar | unresponsive -> FB | core=21 | purity_median=0.875
- novel|10|unresponsive -> unresponsive|01: ctx=(T,F) | novel | unresponsive -> unresponsive | core=201 | purity_median=0.906
- familiar|1|unresponsive -> unresponsive|01: ctx=(F,T) | familiar | unresponsive -> unresponsive | core=199 | purity_median=0.875
- novel|1|FF -> FF|01: ctx=(F,T) | novel | FF -> FF | core=18 | purity_median=0.844
- familiar|0|unresponsive -> unresponsive|01: ctx=(F,F) | familiar | unresponsive -> unresponsive | core=208 | purity_median=0.906
- novel|0|unresponsive -> unresponsive|01: ctx=(F,F) | novel | unresponsive -> unresponsive | core=206 | purity_median=0.906

Strongest boundaries
- ctx=(T,T) | familiar | FF -> FB vs unresponsive -> FB | edges=2474 | median_log10_distance=4.209
- ctx=(T,T) | familiar | FB -> FB vs FF -> FB | edges=667 | median_log10_distance=4.235
- ctx=(T,T) | familiar | FF -> FB vs FF -> FF | edges=662 | median_log10_distance=4.208
- ctx=(T,T) | familiar | FB -> FB vs unresponsive -> FB | edges=495 | median_log10_distance=4.181
- ctx=(T,T) | familiar | FF -> FF vs unresponsive -> FB | edges=452 | median_log10_distance=4.254
- ctx=(T,T) | familiar | unresponsive -> FB vs unresponsive -> FF | edges=387 | median_log10_distance=4.120
- ctx=(T,T) | familiar | FF -> FB vs unresponsive -> FF | edges=204 | median_log10_distance=4.165
- ctx=(T,T) | familiar | FF -> FF vs unresponsive -> FF | edges=57 | median_log10_distance=4.189
- ctx=(T,T) | familiar | FB -> FB vs FF -> FF | edges=24 | median_log10_distance=4.150
- ctx=(T,T) | familiar | FB -> FB vs unresponsive -> FF | edges=23 | median_log10_distance=4.213
- ctx=(T,T) | novel | FF -> FF vs unresponsive -> FB | edges=1637 | median_log10_distance=4.222
- ctx=(T,T) | novel | unresponsive -> FB vs unresponsive -> FF | edges=990 | median_log10_distance=4.122
- ctx=(T,T) | novel | FF -> FF vs unresponsive -> FF | edges=896 | median_log10_distance=4.145
- ctx=(T,T) | novel | FB -> FB vs unresponsive -> FB | edges=702 | median_log10_distance=4.200
- ctx=(T,T) | novel | FB -> FB vs FF -> FF | edges=675 | median_log10_distance=4.214
- ctx=(T,T) | novel | FF -> FB vs FF -> FF | edges=346 | median_log10_distance=4.173
