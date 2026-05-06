Transition-plane basin analysis

Source samples: /Users/matushalak/Documents/context-contrasting/context_contrasting/param_explore/results/2026-05-06_transition_plane_sobol_fixed

Definition
- Separate analysis per image type (`familiar`, `novel`) and per `receives_context` mode.
- Build a symmetric kNN graph in 12D `log10` weight space with `k=24`.
- Compute local purity = fraction of kNN neighbors with the same transition-plane tile.
- Stable core threshold: purity >= 0.65.
- Stable basin = connected component of stable-core points within one tile, size >= 20.

Sampling settings
- n_steps_per_phase = 100
- n_trials = 10
- tail_window = 25

Largest basins
- familiar|11|unresponsive -> FB|01: ctx=(T,T) | familiar | unresponsive -> FB | core=194 | purity_median=0.750
- familiar|11|FF -> FB|01: ctx=(T,T) | familiar | FF -> FB | core=168 | purity_median=0.750
- novel|11|FF -> FF|01: ctx=(T,T) | novel | FF -> FF | core=165 | purity_median=0.708
- novel|11|unresponsive -> FB|01: ctx=(T,T) | novel | unresponsive -> FB | core=60 | purity_median=0.708
- familiar|10|unresponsive -> FB|01: ctx=(T,F) | familiar | unresponsive -> FB | core=207 | purity_median=0.750
- familiar|10|FF -> FB|01: ctx=(T,F) | familiar | FF -> FB | core=159 | purity_median=0.750
- novel|10|unresponsive -> unresponsive|01: ctx=(T,F) | novel | unresponsive -> unresponsive | core=601 | purity_median=0.875
- novel|10|FF -> FF|01: ctx=(T,F) | novel | FF -> FF | core=36 | purity_median=0.708
- familiar|1|unresponsive -> unresponsive|01: ctx=(F,T) | familiar | unresponsive -> unresponsive | core=589 | purity_median=0.875
- novel|1|FF -> FF|01: ctx=(F,T) | novel | FF -> FF | core=226 | purity_median=0.750
- novel|1|unresponsive -> FF|01: ctx=(F,T) | novel | unresponsive -> FF | core=35 | purity_median=0.708
- familiar|0|unresponsive -> unresponsive|01: ctx=(F,F) | familiar | unresponsive -> unresponsive | core=614 | purity_median=0.917
- novel|0|unresponsive -> unresponsive|01: ctx=(F,F) | novel | unresponsive -> unresponsive | core=616 | purity_median=0.875
- novel|0|FF -> FF|01: ctx=(F,F) | novel | FF -> FF | core=47 | purity_median=0.708

Strongest boundaries
- ctx=(T,T) | familiar | FF -> FB vs unresponsive -> FB | edges=3414 | median_log10_distance=3.759
- ctx=(T,T) | familiar | FF -> FB vs FF -> FF | edges=1059 | median_log10_distance=3.783
- ctx=(T,T) | familiar | FB -> FB vs FF -> FB | edges=948 | median_log10_distance=3.805
- ctx=(T,T) | familiar | FB -> FB vs unresponsive -> FB | edges=706 | median_log10_distance=3.817
- ctx=(T,T) | familiar | unresponsive -> FB vs unresponsive -> FF | edges=628 | median_log10_distance=3.743
- ctx=(T,T) | familiar | FF -> FF vs unresponsive -> FB | edges=514 | median_log10_distance=3.799
- ctx=(T,T) | familiar | FF -> FB vs unresponsive -> FF | edges=254 | median_log10_distance=3.751
- ctx=(T,T) | familiar | FF -> FF vs unresponsive -> FF | edges=76 | median_log10_distance=3.828
- ctx=(T,T) | familiar | FB -> FB vs FF -> FF | edges=28 | median_log10_distance=3.784
- ctx=(T,T) | familiar | FB -> FB vs unresponsive -> FF | edges=22 | median_log10_distance=3.892
- ctx=(T,T) | novel | FF -> FF vs unresponsive -> FB | edges=2032 | median_log10_distance=3.785
- ctx=(T,T) | novel | unresponsive -> FB vs unresponsive -> FF | edges=1612 | median_log10_distance=3.688
- ctx=(T,T) | novel | FF -> FF vs unresponsive -> FF | edges=1585 | median_log10_distance=3.706
- ctx=(T,T) | novel | FB -> FB vs FF -> FF | edges=854 | median_log10_distance=3.820
- ctx=(T,T) | novel | FB -> FB vs unresponsive -> FB | edges=829 | median_log10_distance=3.786
- ctx=(T,T) | novel | FF -> FB vs FF -> FF | edges=616 | median_log10_distance=3.755
