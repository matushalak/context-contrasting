# Parameter Regions

The standard sweep fixes `receives_context=(True, True)` for all behaviors except `FF_un`.
The `FF_un` sweep uses `receives_context=(False, False)` as requested.

## Region Counts

- `un_un`: n=0, familiar=`nan`, novel=`nan`, boundary success min=nan
- `un_FB`: n=720, familiar=`unresponsive -> unresponsive`, novel=`unresponsive -> unresponsive`, boundary success min=1.0
- `FF_FB_broad`: n=32, familiar=`FF -> unresponsive`, novel=`FF -> FF`, boundary success min=1.0
- `FF_FB_narrow_familiar`: n=162, familiar=`FF -> unresponsive`, novel=`unresponsive -> unresponsive`, boundary success min=0.0
- `FF_FB_narrow_novel`: n=262, familiar=`unresponsive -> unresponsive`, novel=`FF -> FF`, boundary success min=0.0
- `FB_FB`: n=123, familiar=`FB -> FB`, novel=`FB -> FB`, boundary success min=1.0
- `FF_un`: n=99, familiar=`FF -> unresponsive`, novel=`FF -> FF`, boundary success min=1.0
