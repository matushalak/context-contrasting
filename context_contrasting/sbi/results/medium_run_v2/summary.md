# Parameter Exploration

This search keeps the broad learning-rule configuration fixed and explores only initial weights.

## Best Fits

- `FB_FB`: objective=-0.1420, rmse=0.0100, mismatches=0, familiar=`FB -> FB`, novel=`FB -> FB`
- `FF_FB_broad`: objective=-0.1314, rmse=0.0000, mismatches=0, familiar=`FF -> unresponsive`, novel=`FF -> FF`
- `FF_FB_narrow_familiar`: objective=-0.1518, rmse=0.0000, mismatches=0, familiar=`FF -> unresponsive`, novel=`unresponsive -> unresponsive`
- `FF_FB_narrow_novel`: objective=-0.4655, rmse=0.0000, mismatches=0, familiar=`unresponsive -> unresponsive`, novel=`FF -> FF`
- `FF_un`: objective=-0.0840, rmse=0.1818, mismatches=0, familiar=`FF -> unresponsive`, novel=`FF -> FF`
- `un_FB`: objective=-0.1861, rmse=0.0000, mismatches=0, familiar=`unresponsive -> unresponsive`, novel=`unresponsive -> unresponsive`
- `un_un`: objective=-0.1214, rmse=0.0671, mismatches=0, familiar=`unresponsive -> unresponsive`, novel=`unresponsive -> unresponsive`
