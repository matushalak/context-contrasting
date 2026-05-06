# Learning-Rate + Initial-Weight Landscape

This analysis expands the searched space from initial weights alone to `8` initial-weight parameters plus `4` learning rates.
The standard search uses `receives_context=(True, True)`.
The `FF_un` search remains separate with `receives_context=(False, False)`.
Because of that categorical difference, `FF_un` is treated as a separate manifold when discussing class-switch boundaries.

## Search Space

- compact simulator: `30` steps/phase, `3` trials, tail window `20`
- standard samples: `60` global + `8` local per target
- FF_un samples: `40` global + `8` local
- backtest: `30` steps/phase across seeds `(11, 23, 37)`

## Prototype Notes

- `un_un` is kept only as a diagnostic reference. Under `receives_context=(True, True)`, it is not separately identifiable from `un_FB` with this minimal model.

## Embedding Overview

- PCA explained variance: 0.239, 0.119, 0.101

## Classifier Diagnostics

- logistic regression CV accuracy: `0.641 ± 0.090`
- random forest CV accuracy: `0.693 ± 0.146`

## Stable Regions

- `un_FB`: n=`58`, primary familiar=`unresponsive -> unresponsive`, primary novel=`unresponsive -> unresponsive`
- `FF_FB_broad`: n=`13`, primary familiar=`FF -> FF`, primary novel=`FF -> FF`
- `FF_FB_narrow_familiar`: n=`28`, primary familiar=`FF -> FF`, primary novel=`unresponsive -> unresponsive`
- `FF_FB_narrow_novel`: n=`30`, primary familiar=`unresponsive -> unresponsive`, primary novel=`FF -> FF`
- `FB_FB`: n=`15`, primary familiar=`FB -> FB`, primary novel=`FB -> FB`
- `FF_un`: n=`11`, primary familiar=`FF -> FF`, primary novel=`FF -> FF`

## Validated Core Ranges

- `un_FB`: q15-q85, boundary success=`0.68`
- `FF_FB_broad`: q5-q95, boundary success=`1.00`
- `FF_FB_narrow_familiar`: q15-q85, boundary success=`0.73`
- `FF_FB_narrow_novel`: q15-q85, boundary success=`0.94`
- `FB_FB`: q15-q85, boundary success=`0.81`
- `FF_un`: q5-q95, boundary success=`1.00`

## Boundary Pairs

- `FF_FB_narrow_novel` vs `un_FB`: normalized distance `0.777`
- `FF_FB_narrow_familiar` vs `un_FB`: normalized distance `0.783`
- `FF_FB_narrow_familiar` vs `un_FB`: normalized distance `0.882`
- `FF_FB_broad` vs `FF_FB_narrow_novel`: normalized distance `1.030`
- `FF_FB_broad` vs `FF_FB_narrow_novel`: normalized distance `1.030`
- `FF_FB_broad` vs `FF_FB_narrow_familiar`: normalized distance `1.325`
- `FF_FB_broad` vs `FF_FB_narrow_familiar`: normalized distance `1.325`
- `FF_FB_narrow_familiar` vs `FF_FB_narrow_novel`: normalized distance `1.387`
- `FF_FB_broad` vs `un_FB`: normalized distance `1.444`
- `FF_FB_narrow_familiar` vs `FF_FB_narrow_novel`: normalized distance `1.525`
