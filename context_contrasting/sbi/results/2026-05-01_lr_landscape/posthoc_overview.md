# Posthoc Boundary Analysis

This report reuses the saved 12D landscape and inspects the top-2 objective competition for every sampled point.
Boundary clouds are defined as the lowest `10`% of second-best margins, i.e. `margin2 <= 0.1457`.

## Main Observations

- `w_ff_0` and `w_ff_1` remain the dominant separators. Learning rates matter, but mostly as secondary deformations of already weight-defined regions.
- `lr_fb` is the most important learning-rate axis in the random-forest ranking. It shows up mainly in the unresponsive vs selective separations, not in the broad FF vs narrow FF split.
- The strongest low-margin competition is `FF_FB_narrow_novel` vs `FF_un`, but that pair crosses the `receives_context` manifold boundary and should not be interpreted as a standard-plasticity decision surface.
- Within the standard manifold, the most active switch zones are `FF_FB_narrow_familiar` vs `un_FB`, `FB_FB` vs `FF_FB_narrow_novel`, `FF_FB_broad` vs `FF_FB_narrow_novel`, and `FB_FB` vs `un_FB`.

## Stable-Class Medians

- `FB_FB`: `w_ff=(0.001508, 0.0009672)`, `w_fb=(0.3641, 0.3885)`, `w_lat=(0.2324, 0.238)`, `w_pv_lat=(0.09737, 0.1088)`, `lr=(ff 0.01029, fb 0.002571, lat 0.00658, pv 0.002029)`
- `FF_FB_broad`: `w_ff=(0.4354, 0.4337)`, `w_fb=(0.07724, 0.04372)`, `w_lat=(0.01234, 0.01252)`, `w_pv_lat=(0.06357, 0.04517)`, `lr=(ff 0.008774, fb 0.002316, lat 0.002454, pv 0.001769)`
- `FF_FB_narrow_familiar`: `w_ff=(0.758, 0.008863)`, `w_fb=(0.01278, 0.01059)`, `w_lat=(0.03064, 0.1153)`, `w_pv_lat=(0.09419, 0.09249)`, `lr=(ff 0.009216, fb 0.00176, lat 0.004494, pv 0.001887)`
- `FF_FB_narrow_novel`: `w_ff=(0.009251, 0.7828)`, `w_fb=(0.007887, 0.09065)`, `w_lat=(0.1259, 0.03417)`, `w_pv_lat=(0.1322, 0.1151)`, `lr=(ff 0.01247, fb 0.00248, lat 0.004514, pv 0.001989)`
- `FF_un`: `w_ff=(0.3768, 0.4416)`, `w_fb=(0.0003214, 0.0002821)`, `w_lat=(0.06839, 0.05724)`, `w_pv_lat=(0.07135, 0.06811)`, `lr=(ff 0.01716, fb 0.002345, lat 0.005942, pv 0.002057)`
- `un_FB`: `w_ff=(0.007007, 0.006688)`, `w_fb=(0.009926, 0.009347)`, `w_lat=(0.04968, 0.05263)`, `w_pv_lat=(0.03096, 0.03382)`, `lr=(ff 0.005079, fb 0.001046, lat 0.002849, pv 0.001756)`

## Top Boundary Clouds

- `FF_FB_narrow_novel vs FF_un`: n=`109`, median margin=`0.0850`, cross-manifold, dominant assigned target=`FF_un`
- `FF_FB_narrow_familiar vs un_FB`: n=`76`, median margin=`0.0801`, same manifold, dominant assigned target=`un_FB`
- `FB_FB vs FF_FB_narrow_novel`: n=`59`, median margin=`0.0385`, same manifold, dominant assigned target=`FB_FB`
- `FF_FB_broad vs FF_FB_narrow_novel`: n=`52`, median margin=`0.0833`, same manifold, dominant assigned target=`FF_FB_broad`
- `FB_FB vs un_FB`: n=`51`, median margin=`0.0714`, same manifold, dominant assigned target=`FB_FB`
- `FF_FB_narrow_novel vs un_FB`: n=`22`, median margin=`0.0600`, same manifold, dominant assigned target=`FF_FB_narrow_novel`
- `FB_FB vs FF_FB_broad`: n=`14`, median margin=`0.0867`, same manifold, dominant assigned target=`FB_FB`
- `FF_FB_broad vs FF_FB_narrow_familiar`: n=`13`, median margin=`0.1000`, same manifold, dominant assigned target=`FF_FB_broad`
- `FB_FB vs FF_FB_narrow_familiar`: n=`12`, median margin=`0.0649`, same manifold, dominant assigned target=`FB_FB`
- `FF_FB_narrow_familiar vs FF_un`: n=`6`, median margin=`0.0490`, cross-manifold, dominant assigned target=`FF_un`

## Boundary Path Behavior

- `FF_FB_narrow_familiar vs un_FB` path `1546` -> `1065` first switched at `alpha=0.833` (direct_switch); sequence: `FF_FB_narrow_familiar | FF_FB_narrow_familiar | FF_FB_narrow_familiar | FF_FB_narrow_familiar | FF_FB_narrow_familiar | un_FB | FF_FB_narrow_familiar`
- `FF_FB_narrow_familiar vs un_FB` path `609` -> `1139` first switched at `alpha=0.167` (direct_switch); sequence: `FF_FB_narrow_familiar | un_FB | un_FB | un_FB | un_FB | un_FB | un_FB`
- `FF_FB_narrow_novel vs un_FB` path `393` -> `127` first switched at `alpha=0.167` (direct_switch); sequence: `FF_FB_narrow_novel | un_FB | un_FB | un_FB | un_FB | un_FB | un_FB`
- `FF_FB_narrow_novel vs un_FB` path `916` -> `816` first switched at `alpha=0.667` (direct_switch); sequence: `FF_FB_narrow_novel | FF_FB_narrow_novel | FF_FB_narrow_novel | FF_FB_narrow_novel | un_FB | un_FB | un_FB`
- `FB_FB vs FF_FB_broad` path `539` -> `1396` first switched at `alpha=1.000` (direct_switch); sequence: `FB_FB | FB_FB | FB_FB | FB_FB | FB_FB | FB_FB | FF_FB_broad`
- `FF_FB_broad vs FF_FB_narrow_novel` path `276` -> `1416` first switched at `alpha=0.500` (direct_switch); sequence: `FF_FB_broad | FF_FB_broad | FF_FB_broad | FF_FB_narrow_novel | FF_FB_narrow_novel | FF_FB_narrow_novel | FF_FB_narrow_novel`
- `FB_FB vs FF_FB_narrow_novel` path `991` -> `920` never switched class along the sampled interpolation.
- `FB_FB vs un_FB` path `535` -> `495` never switched class along the sampled interpolation.
- `FB_FB vs un_FB` path `967` -> `495` never switched class along the sampled interpolation.
- `FB_FB vs FF_FB_narrow_novel` path `474` -> `268` first switched at `alpha=0.167` (direct_switch); sequence: `FB_FB | FF_FB_narrow_novel | FF_FB_narrow_novel | FF_FB_narrow_novel | FF_FB_narrow_novel | FF_FB_narrow_novel | FF_FB_narrow_novel`

## Backtest Reading

Validated-core backtests are the main trust anchor for whether a region behaves as a robust basin instead of a thin assignment shell.
- `un_FB`: q25-q75, boundary success=`0.625`
- `FF_FB_broad`: q25-q75, boundary success=`0.875`
- `FF_FB_narrow_familiar`: q25-q75, boundary success=`0.840`
- `FF_FB_narrow_novel`: q25-q75, boundary success=`0.764`
- `FB_FB`: q25-q75, boundary success=`0.986`
- `FF_un`: q10-q90, boundary success=`1.000`

## Interpretation

- `FB_FB` remains the cleanest basin after adding learning rates. Its backtests are near-perfect and its boundary competitors are limited.
- `FF_un` is also robust, but only on the separate `receives_context=(False, False)` manifold.
- The standard-manifold FF-family classes remain partly entangled. Narrow familiar, narrow novel, and broad FF are better seen as neighboring regions inside one larger FF-driven regime than as completely isolated islands.
- `un_FB` absorbs a very large volume of parameter space, but its interquartile core is only moderately stable under full backtests. That suggests the class is easy to assign in compact summaries yet sensitive near its edges.
