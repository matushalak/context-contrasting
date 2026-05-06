# Focused Weight Analysis

This report ignores expert2 and focuses on the weight-only exploration, using only the naive and expert1 summaries for familiar/novel and full/occluded responses.

## What The Old `FB_FB vs FF_FB...` Plots Were

Those plots were not parameter-range plots.
They came from the later learning-rate script and showed a straight interpolation path between one sampled point from class A and one sampled point from class B.
The left panel tracked how a summary statistic changed along that interpolation, and the right panel showed which class the interpolated point got assigned to.
So `FB_FB vs FF_FB_broad` meant: take one stable `FB_FB` point, take one stable `FF_FB_broad` point, interpolate between them in parameter space, and see where the class label flips.
That can be useful for probing boundaries, but it is not the main visualization if the goal is simply to understand class-defining parameter ranges.

## un_un vs un_FB

`un_un` and `un_FB` are not separable in the current naive/expert1, full/occluded summary. Their eight reference summary values are exactly identical in the current setup.

Reference summary values are saved in `focused_reference_summaries.csv`.

## PCA

PCA explained variance ratio on log10 weights: `0.313`, `0.161`.

## Validated core ranges

- `un_FB`: familiar=`unresponsive -> unresponsive`, novel=`unresponsive -> unresponsive`, `w_ff=(0.0001647-0.3909, 0.0001538-0.2016)`, `w_fb=(0.0001629-0.05249, 0.0001588-0.06547)`
- `FF_FB_broad`: familiar=`FF -> unresponsive`, novel=`FF -> FF`, `w_ff=(0.2308-0.7134, 0.08657-0.7728)`, `w_fb=(0.0003057-0.06443, 0.003756-0.2357)`
- `FF_FB_narrow_familiar`: familiar=`FF -> unresponsive`, novel=`unresponsive -> unresponsive`, `w_ff=(0.2921-1.32, 0.0008105-0.2896)`, `w_fb=(0.0006031-0.04361, 0.0005287-0.0582)`
- `FF_FB_narrow_novel`: familiar=`unresponsive -> unresponsive`, novel=`FF -> FF`, `w_ff=(0.0001855-0.1655, 0.05885-1.924)`, `w_fb=(0.0001732-0.04989, 0.0004119-1.054)`
- `FF_un`: familiar=`FF -> unresponsive`, novel=`FF -> FF`, `w_ff=(0.2707-0.955, 0.2765-1.174)`, `w_fb=(0.0001-0.2198, 0.0001-0.2928)`

## Main differentiators

- `FF_FB_broad` vs `FF_FB_narrow_familiar`: largest median log10 differences are `w_ff_1` (1.66), `w_fb_1` (0.68), `w_fb_0` (0.54).
- `FF_FB_broad` vs `FF_FB_narrow_novel`: largest median log10 differences are `w_ff_0` (1.75), `w_fb_0` (0.71), `w_fb_1` (0.45).
- `FF_FB_broad` vs `FF_un`: largest median log10 differences are `w_fb_1` (2.61), `w_fb_0` (2.53), `w_lat_0` (0.47).
- `FF_FB_narrow_familiar` vs `FF_FB_narrow_novel`: largest median log10 differences are `w_ff_1` (1.94), `w_ff_0` (1.91), `w_fb_1` (1.14).
- `FF_FB_narrow_familiar` vs `FF_un`: largest median log10 differences are `w_fb_0` (1.99), `w_fb_1` (1.92), `w_ff_1` (1.75).
- `FF_FB_narrow_novel` vs `FF_un`: largest median log10 differences are `w_fb_1` (3.06), `w_ff_0` (1.83), `w_fb_0` (1.82).
- `un_FB` vs `FF_FB_broad`: largest median log10 differences are `w_ff_1` (1.77), `w_ff_0` (1.77), `w_fb_1` (0.83).
- `un_FB` vs `FF_FB_narrow_familiar`: largest median log10 differences are `w_ff_0` (1.93), `w_pv_lat_1` (0.72), `w_lat_1` (0.53).
- `un_FB` vs `FF_FB_narrow_novel`: largest median log10 differences are `w_ff_1` (2.05), `w_fb_1` (1.29), `w_lat_1` (0.20).
- `un_FB` vs `FF_un`: largest median log10 differences are `w_ff_1` (1.86), `w_ff_0` (1.84), `w_fb_1` (1.77).

## Most active pairwise competitions

- `FF_FB_narrow_familiar vs un_FB`: n=`1999`, median objective gap=`1.1738`.
- `FF_FB_narrow_novel vs FF_un`: n=`457`, median objective gap=`1.7535`.
- `FF_FB_broad vs FF_un`: n=`433`, median objective gap=`0.3293`.
- `FF_FB_narrow_familiar vs FF_un`: n=`197`, median objective gap=`1.1657`.
- `FF_FB_broad vs FF_FB_narrow_novel`: n=`155`, median objective gap=`0.3043`.
- `FF_FB_broad vs un_FB`: n=`64`, median objective gap=`0.5361`.
- `FF_FB_broad vs FF_FB_narrow_familiar`: n=`58`, median objective gap=`1.1970`.

## Interpretation

- `FF_FB_broad` differs from the narrow classes mainly by keeping both FF weights elevated rather than collapsing one axis.
- `FF_FB_narrow_familiar` and `FF_FB_narrow_novel` are mainly distinguished by which FF axis stays large: `w_ff_0` for familiar-narrow, `w_ff_1` for novel-narrow.
- `FF_un` differs from the FF->FB classes mostly by pushing both FB weights toward the floor while keeping both FF weights high.
- `un_FB` sits near the low-FF, low-FB regime. In the current summary, there is no separate `un_un` region next to it.
