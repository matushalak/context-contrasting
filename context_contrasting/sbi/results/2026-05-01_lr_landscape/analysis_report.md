# Learning-Rate Landscape Analysis

## Scope

This run extends the earlier initial-weight exploration to a joint 12D parameter space:

- 8 initial weights: `w_ff_0`, `w_ff_1`, `w_fb_0`, `w_fb_1`, `w_lat_0`, `w_lat_1`, `w_pv_lat_0`, `w_pv_lat_1`
- 4 learning rates: `lr_ff`, `lr_fb`, `lr_lat`, `lr_pv`

The standard plasticity classes were searched on the usual manifold with `receives_context=(True, True)`.
`FF_un` was searched separately on the requested `receives_context=(False, False)` manifold.

`un_un` was kept only as a diagnostic reference and not treated as a searchable class, because under `receives_context=(True, True)` it remains effectively indistinguishable from `un_FB` in this minimal model.

## Code

The main search code is in:

- `context_contrasting/sbi/param_lr_landscape.py`
- `context_contrasting/sbi/param_lr_postprocess.py`
- `context_contrasting/minimal/transition_types.py`

The search script builds richer transition signatures than the earlier weight-only sweep. It uses:

- naive, expert, and expert2 response means
- scalarized state summaries for familiar and novel conditions
- phase-to-phase state deltas and response deltas
- primary transition labels from the scalarized states
- prototype-distance objectives against reference transition families

The post-processing script then reuses the saved objective scores to find low-margin class competitions and summarize likely decision boundaries.

## Search and Validation Setup

- compact search simulator: `80` steps per phase, `6` trials, tail window `60`
- full backtests: `100` steps per phase, tail window `80`, seeds `(11, 23, 37)`
- standard search samples: `1800` global Sobol + `160` local samples per class
- `FF_un` search samples: `1400` global Sobol + `160` local samples

The full result tables are:

- `combined_candidates.csv`
- `stable_region_members.csv`
- `validated_core_summary.csv`
- `validated_core_backtests.csv`
- `embedded_candidates.csv`
- `boundary_pairs.csv`
- `boundary_paths.csv`
- `boundary_switches.csv`
- `boundary_pair_summary.csv`
- `competition_augmented.csv`

## Dimensionality Reduction

UMAP was not available in the environment, so the analysis uses:

- PCA for global linear structure
- t-SNE for nonlinear local neighborhood structure

PCA explained variance ratio:

- `0.202`, `0.105`, `0.087`

This is a useful sanity check: there is no low-dimensional plane that cleanly captures the entire class structure. The landscape is genuinely high-dimensional and only partially separable.

## What Separates the Classes

The random-forest feature ranking shows that the dominant separators are still the initial FF weights:

1. `w_ff_1`
2. `w_ff_0`
3. `w_fb_0`
4. `w_fb_1`
5. `w_lat_1`
6. `w_lat_0`

The learning rates do matter, but they enter later:

- strongest learning-rate axis: `lr_fb`
- then `lr_lat`
- then `lr_ff`
- then `lr_pv`

The practical interpretation is:

- transition-family identity is still defined mainly by the initial weight geometry
- learning rates deform, widen, or destabilize those regions rather than replacing the role of the weights

## Stable Region Medians

Median parameter values inside the stable regions are in `stable_median_parameters.csv`.

The clearest medians are:

- `un_FB`: both FF and FB weights near zero, moderate lateral weights, lower `lr_ff` and especially lower `lr_fb`
- `FF_FB_broad`: both FF weights high and symmetric, FB weights modest, lateral weights low
- `FF_FB_narrow_familiar`: high `w_ff_0`, very low `w_ff_1`, low FB weights, asymmetric lateral support
- `FF_FB_narrow_novel`: mirror image of narrow familiar, with high `w_ff_1`
- `FB_FB`: both FF weights very low, both FB weights high, strongest lateral weights, cleanest robust basin
- `FF_un`: both FF weights high, both FB weights near zero, on the separate no-context manifold

## Validated Core Regions

The full quantile ranges are in `validated_core_summary.csv`.

Core backtest success rates:

- `FB_FB`: `0.986`
- `FF_un`: `1.000`
- `FF_FB_broad`: `0.875`
- `FF_FB_narrow_familiar`: `0.840`
- `FF_FB_narrow_novel`: `0.764`
- `un_FB`: `0.625`

This is the key trust ranking for the 12D search:

- `FB_FB` is the most reliable standard-manifold basin
- `FF_un` is fully reliable, but only on its separate manifold
- the FF-family standard-manifold classes are real, but partially entangled
- `un_FB` occupies a large volume of assigned space, but its inner core is only moderately stable under full backtests

## Boundary and Switch Structure

Boundary clouds were defined as the lowest 10% of top-2 margins:

- threshold: `margin2 <= 0.1457`

Most active low-margin competitions:

- `FF_FB_narrow_novel` vs `FF_un`: `109` points, but cross-manifold and therefore not a standard decision surface
- `FF_FB_narrow_familiar` vs `un_FB`: `76` points
- `FB_FB` vs `FF_FB_narrow_novel`: `59` points
- `FF_FB_broad` vs `FF_FB_narrow_novel`: `52` points
- `FB_FB` vs `un_FB`: `51` points
- `FF_FB_narrow_novel` vs `un_FB`: `22` points

The annotated boundary-path summary is in `boundary_switches_annotated.csv`.

Across the sampled nearest-neighbor paths:

- `16` paths were direct switches
- `1` path was a detour switch
- `3` paths never switched class along straight log-space interpolation

The single detour path is informative:

- `FF_FB_broad -> un_FB` can pass through `FF_FB_narrow_familiar`

That means some class boundaries are not pairwise in a simple sense. A straight interpolation can cross a third regime before reaching the intended target, which is evidence that the class regions are curved and interleaved rather than arranged as clean convex cells.

The no-switch paths are also informative:

- some nearest cross-class neighbors in Euclideanized parameter space do not cross the expected class boundary under straight interpolation

This means "nearest neighbor from another class" does not always correspond to a direct topological border. The geometry is warped.

## Classification Difficulty

A classifier trained only on the 12 raw parameters reaches:

- multinomial logistic regression CV accuracy: `0.688 +- 0.047`
- random forest CV accuracy: `0.641 +- 0.084`

This is high enough to confirm non-random structure, but far from perfect.

The implication is important:

- the class regions are structured
- but they are not cleanly linearly or even simply nonlinearly separable in raw parameter space
- much of the apparent region volume is shared near fuzzy boundaries

## Main Interpretation

The best way to read the 12D landscape is:

1. `FB_FB` is a robust and relatively isolated attractor family.
2. `FF_un` is also robust, but only if the neuron never receives context.
3. The standard-manifold FF-family classes (`FF_FB_broad`, `FF_FB_narrow_familiar`, `FF_FB_narrow_novel`) form neighboring subregions inside a larger FF-dominated regime.
4. `un_FB` behaves more like a large catchment with soft edges than a sharply isolated basin.
5. Learning rates refine these regions and modulate stability, but they do not overturn the dominant role of the initial FF and FB weight geometry.

## Most Useful Plots

For quick inspection, the most informative figures are:

- `plots/class_parameter_medians.png`
- `plots/boundary_pair_clouds.png`
- `plots/pca_decision_boundaries.png`
- `plots/parameter_lr_panels.png`
- `plots/state_scalar_panels.png`

## Limitations

- This is still a prototype-based classification of a minimal model, not an exact analytical phase diagram.
- `un_un` remains non-identifiable from `un_FB` under the standard context regime.
- Some region assignments are easier to obtain than they are to validate under longer, multi-seed backtests.
- Straight log-space interpolation is only one probe of boundary shape; it detects boundary crossings but does not prove convexity or smoothness.

## Bottom Line

Adding learning rates broadens and warps the parameter regions, but it does not fundamentally change the identity of what separates the classes.

The strongest standard-manifold result is a clean `FB_FB` basin.
The strongest overall result is `FF_un`, but only with `receives_context=(False, False)`.
The main ambiguity is not between all classes equally; it is concentrated inside the FF-family and at the `un_FB` interface.
