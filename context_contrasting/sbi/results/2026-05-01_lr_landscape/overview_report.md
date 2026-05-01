# Learning-Rate + Initial-Weight Landscape

This analysis expands the searched space from initial weights alone to `8` initial-weight parameters plus `4` learning rates.
The standard search uses `receives_context=(True, True)`.
The `FF_un` search remains separate with `receives_context=(False, False)`.
Because of that categorical difference, `FF_un` is treated as a separate manifold when discussing class-switch boundaries.

## Search Space

- compact simulator: `80` steps/phase, `6` trials, tail window `60`
- standard samples: `1800` global + `160` local per target
- FF_un samples: `1400` global + `160` local
- backtest: `100` steps/phase across seeds `(11, 23, 37)`

## Prototype Notes

- `un_un` is kept only as a diagnostic reference. Under `receives_context=(True, True)`, it is not separately identifiable from `un_FB` with this minimal model.

## Embedding Overview

- PCA explained variance: 0.202, 0.105, 0.087

## Classifier Diagnostics

- logistic regression CV accuracy: `0.688 ± 0.047`
- random forest CV accuracy: `0.641 ± 0.084`

## Stable Regions

- `un_FB`: n=`1644`, primary familiar=`unresponsive -> unresponsive`, primary novel=`unresponsive -> unresponsive`
- `FF_FB_broad`: n=`276`, primary familiar=`FF -> unresponsive`, primary novel=`FF -> FF`
- `FF_FB_narrow_familiar`: n=`606`, primary familiar=`FF -> unresponsive`, primary novel=`unresponsive -> unresponsive`
- `FF_FB_narrow_novel`: n=`557`, primary familiar=`unresponsive -> unresponsive`, primary novel=`FF -> FF`
- `FB_FB`: n=`558`, primary familiar=`FB -> FB`, primary novel=`FB -> FB`
- `FF_un`: n=`298`, primary familiar=`FF -> unresponsive`, primary novel=`FF -> FF`

## Validated Core Ranges

- `un_FB`: q25-q75, boundary success=`0.62`
- `FF_FB_broad`: q25-q75, boundary success=`0.88`
- `FF_FB_narrow_familiar`: q25-q75, boundary success=`0.84`
- `FF_FB_narrow_novel`: q25-q75, boundary success=`0.76`
- `FB_FB`: q25-q75, boundary success=`0.99`
- `FF_un`: q10-q90, boundary success=`1.00`

## Boundary Pairs

- `FF_FB_narrow_familiar` vs `un_FB`: normalized distance `0.306`, dominant changes `w_pv_lat_1`, `w_pv_lat_0`, `w_fb_1`
- `FF_FB_narrow_familiar` vs `un_FB`: normalized distance `0.344`, dominant changes `w_fb_0`, `w_ff_0`, `w_pv_lat_0`
- `FF_FB_narrow_novel` vs `un_FB`: normalized distance `0.344`, dominant changes `w_fb_1`, `w_ff_1`, `w_pv_lat_0`
- `FF_FB_narrow_novel` vs `un_FB`: normalized distance `0.360`, dominant changes `w_fb_1`, `w_lat_0`, `w_fb_0`
- `FB_FB` vs `FF_FB_broad`: normalized distance `0.375`, dominant changes `w_fb_1`, `w_pv_lat_1`, `w_ff_0`
- `FF_FB_broad` vs `FF_FB_narrow_novel`: normalized distance `0.392`, dominant changes `w_pv_lat_0`, `lr_ff`, `w_ff_0`
- `FB_FB` vs `FF_FB_narrow_novel`: normalized distance `0.396`, dominant changes `w_ff_1`, `w_pv_lat_1`, `w_fb_0`
- `FB_FB` vs `un_FB`: normalized distance `0.407`, dominant changes `w_pv_lat_0`, `w_pv_lat_1`, `w_ff_1`
- `FB_FB` vs `un_FB`: normalized distance `0.428`, dominant changes `w_fb_0`, `w_lat_0`, `w_ff_1`
- `FB_FB` vs `FF_FB_narrow_novel`: normalized distance `0.443`, dominant changes `w_lat_0`, `w_lat_1`, `w_ff_0`
