# Parameter Regions

The standard sweep fixes `receives_context=(True, True)` for all behaviors except `FF_un`.
The `FF_un` sweep uses `receives_context=(False, False)` as requested.
Under this constraint, `un_un` is not separately identifiable from `un_FB`; it is kept only as a diagnostic reference, not as a labeled region.

## Region Counts

- `un_FB`: n=590, familiar=`unresponsive -> unresponsive`, novel=`unresponsive -> unresponsive`, boundary success min=1.0
- `FF_FB_broad`: n=31, familiar=`FF -> unresponsive`, novel=`FF -> FF`, boundary success min=1.0
- `FF_FB_narrow_familiar`: n=176, familiar=`FF -> unresponsive`, novel=`unresponsive -> unresponsive`, boundary success min=0.0
- `FF_FB_narrow_novel`: n=259, familiar=`unresponsive -> unresponsive`, novel=`FF -> FF`, boundary success min=0.0
- `FB_FB`: n=115, familiar=`FB -> FB`, novel=`FB -> FB`, boundary success min=1.0
- `FF_un`: n=99, familiar=`FF -> unresponsive`, novel=`FF -> FF`, boundary success min=1.0
