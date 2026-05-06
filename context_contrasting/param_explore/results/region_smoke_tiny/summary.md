# Parameter Regions

The standard sweep fixes `receives_context=(True, True)` for all behaviors except `FF_un`.
The `FF_un` sweep uses `receives_context=(False, False)` as requested.

## Region Counts

- `un_un`: n=1, familiar=`unresponsive -> unresponsive`, novel=`unresponsive -> unresponsive`, boundary success min=0.3333333333333333
- `un_FB`: n=5, familiar=`unresponsive -> unresponsive`, novel=`unresponsive -> unresponsive`, boundary success min=1.0
- `FF_FB_broad`: n=3, familiar=`FF -> FF`, novel=`FF -> FF`, boundary success min=1.0
- `FF_FB_narrow_familiar`: n=5, familiar=`FF -> FF`, novel=`unresponsive -> unresponsive`, boundary success min=1.0
- `FF_FB_narrow_novel`: n=4, familiar=`unresponsive -> unresponsive`, novel=`FF -> FF`, boundary success min=1.0
- `FB_FB`: n=2, familiar=`FB -> FB`, novel=`FB -> FB`, boundary success min=1.0
- `FF_un`: n=3, familiar=`FF -> FF`, novel=`FF -> FF`, boundary success min=1.0
