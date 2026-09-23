# Subject 6 primary checkpoint

This is an interim, single-subject checkpoint. The final decision must use the
complete Sub4/Sub5/Sub6 pilot and the strict group reporter.

Primary comparison: identical diagonal-LDA decoder and leave-one-run-out folds.
The legacy input is the existing 8-mm-smoothed LS-A beta set; the new input is
fMRIPrep PEPOLAR-SDC plus GLMsingle Type-D single-trial betas using the 20-value
fractional-ridge grid. New betas were decoded with either 0 or 3 mm additional
mask-normalized smoothing.

| Contrast | Legacy | New 0 mm | Delta | New 3 mm | Delta |
|---|---:|---:|---:|---:|---:|
| Within natural pleasant | 0.545 | 0.570 | +0.025 | 0.555 | +0.010 |
| Within AI pleasant | 0.570 | 0.520 | -0.050 | 0.495 | -0.075 |
| Within natural unpleasant | 0.550 | 0.595 | +0.045 | 0.615 | +0.065 |
| Within AI unpleasant | 0.505 | 0.520 | +0.015 | 0.510 | +0.005 |
| Natural to AI pleasant | 0.465 | 0.480 | +0.015 | 0.520 | +0.055 |
| AI to natural pleasant | 0.445 | 0.500 | +0.055 | 0.470 | +0.025 |
| Natural to AI unpleasant | 0.495 | 0.540 | +0.045 | 0.555 | +0.060 |
| AI to natural unpleasant | 0.495 | 0.515 | +0.020 | 0.520 | +0.025 |
| Mean of four within-source contrasts | 0.5425 | 0.5513 | +0.0088 | 0.5438 | +0.0013 |
| Mean of four cross-source contrasts | 0.4750 | 0.5088 | +0.0338 | 0.5163 | +0.0413 |

All 32 new accuracy maps and 240 fold rows passed validation. Maps are float32,
finite and within `[0, 1]` at all 247,810 evaluated centers, and NaN outside the
analysis mask.

The 20-value fit selected the minimum fraction (0.05) in 245,410/247,810 voxels
(99.03%), so fractional-ridge selection remains boundary-limited. However, the
decoded result was stable against the five-value coarse fit: across all 32
map-summary rows, mean absolute whole-mask accuracy difference was 0.00422
(maximum 0.01), and mean absolute spatial-mean difference was 0.000162.

Canonical outputs:

- GLMsingle: `pilot_raw_pipeline_sub4_6/glmsingle_sdc_defaultgrid/Sub6`
- New decoding/maps: `pilot_raw_pipeline_sub4_6/decoding_new_sdc/Sub6`
- Matched legacy decoding/maps: `pilot_raw_pipeline_sub4_6/decoding_corrected_legacy/Sub6`
