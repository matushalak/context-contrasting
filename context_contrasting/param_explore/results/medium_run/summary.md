# Parameter Exploration

This search keeps the broad learning-rule configuration fixed and explores only initial weights.

## Best Fits

- `FB_FB`: objective=-0.9733, rmse=0.5691, mismatches=0, familiar=`FB -> FB`, novel=`FB -> FB`
- `FF_FB_broad`: objective=-0.6760, rmse=2.2521, mismatches=0, familiar=`FF -> unresponsive`, novel=`FF -> FF`
- `FF_FB_narrow_familiar`: objective=-0.7591, rmse=0.0000, mismatches=0, familiar=`FF -> unresponsive`, novel=`unresponsive -> unresponsive`
- `FF_FB_narrow_novel`: objective=-2.3273, rmse=0.0000, mismatches=0, familiar=`unresponsive -> unresponsive`, novel=`FF -> FF`
- `FF_un`: objective=-1.1472, rmse=0.1818, mismatches=0, familiar=`FF -> unresponsive`, novel=`FF -> FF`
- `un_FB`: objective=-0.9309, rmse=0.0051, mismatches=0, familiar=`unresponsive -> unresponsive`, novel=`unresponsive -> unresponsive`
- `un_un`: objective=-0.8772, rmse=0.0678, mismatches=0, familiar=`unresponsive -> unresponsive`, novel=`unresponsive -> unresponsive`
