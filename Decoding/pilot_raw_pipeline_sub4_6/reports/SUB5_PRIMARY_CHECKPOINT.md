# Sub5 primary raw-pipeline checkpoint

Status: passed merged distortion-corrected fMRIPrep QC, quantitative tSNR QC,
mixed-run-length GLMsingle validation, matched decoding, run-08 sensitivity,
and voxelwise NIfTI integrity validation.

- fMRIPrep: February runs 01/03/04/05 use fieldmap-less SyN SDC; March runs
  02/06/07/08/09/10 use PEPOLAR SDC. All decoding inputs are 2 mm
  `MNI152NLin6Asym` outputs derived from the native 1.8 mm acquisition.
- Quantitative QC: equal-run mean tSNR 22.4126. Runs 01 (20.4205) and 08
  (21.1972) have the lowest mean tSNR and the highest motion: respectively
  27/223 and 29/217 valid frames have FD > 0.5 mm.
- GLMsingle: 245,000 mask voxels, 600 finite float32 Type-D trial betas, exact
  mixed run lengths `[224,218,224,224,224,218,218,218,218,218]`, 20-value
  fractional-ridge grid, and 10 candidate GLMdenoise PCs (0 selected).
- Ridge limitation: 0.05, the minimum requested fraction, was selected in
  244,828/245,000 voxels (99.9298%); 0.10 was selected in the other 172.
- Decoder: diagonal LDA, variance shrinkage 0.1, 5 mm searchlights, 0/3 mm
  additional mask-normalized smoothing, leave-one-run-out primary CV, and
  identity-GroupKFold sensitivity.
- Primary output gate: 32 summary rows, 240 fold rows, and 32 float32 `.nii.gz`
  maps. Every map passed mask/grid/range/NaN validation.

## Primary leave-one-run-out result

`Legacy` is matched leakage-safe decoding of the existing 8-mm-smoothed LS-A
betas. `New 0/3 mm` uses true single-trial GLMsingle Type-D betas with that
amount of additional smoothing.

| Contrast | Legacy | New 0 mm | Delta | New 3 mm | Delta |
|---|---:|---:|---:|---:|---:|
| Natural pleasant (within) | 0.520 | 0.480 | -0.040 | 0.540 | +0.020 |
| AI pleasant (within) | 0.615 | 0.525 | -0.090 | 0.540 | -0.075 |
| Natural unpleasant (within) | 0.465 | 0.440 | -0.025 | 0.490 | +0.025 |
| AI unpleasant (within) | 0.540 | 0.510 | -0.030 | 0.465 | -0.075 |
| Natural to AI pleasant | 0.565 | 0.570 | +0.005 | 0.560 | -0.005 |
| AI to natural pleasant | 0.505 | 0.540 | +0.035 | 0.535 | +0.030 |
| Natural to AI unpleasant | 0.545 | 0.525 | -0.020 | 0.510 | -0.035 |
| AI to natural unpleasant | 0.545 | 0.505 | -0.040 | 0.535 | -0.010 |

Mean pooled whole-mask accuracy:

- Within-source: legacy 0.53500; new 0 mm 0.48875 (-0.04625); new 3 mm
  0.50875 (-0.02625).
- Cross-source: legacy 0.54000; new 0 mm 0.53500 (-0.00500); new 3 mm
  0.53500 (-0.00500).
- All eight contrasts: legacy 0.53750; new 0 mm 0.51188 (-0.02562); new 3 mm
  0.52188 (-0.01562).

## Run-08 exclusion sensitivity

The diagnostic excludes all 60 run-08 trials from every training and test
fold. It passed its exact gate: 16 maps, 144 folds, nine held-out runs, 540
included trials, and no run-08 fold.

| Additional smoothing | Primary within | Exclude-08 within | Primary cross | Exclude-08 cross | Primary all | Exclude-08 all |
|---:|---:|---:|---:|---:|---:|---:|
| 0 mm | 0.48875 | 0.50000 | 0.53500 | 0.55556 | 0.51188 | 0.52778 |
| 3 mm | 0.50875 | 0.52083 | 0.53500 | 0.54861 | 0.52188 | 0.53472 |

Removing run 08 modestly improves the aggregate result and nearly matches the
legacy overall mean at 3 mm, but it is not the primary analysis. Run 01 also
has 27 high-motion frames, so this single-run sensitivity does not isolate all
motion effects.

These are single-subject descriptive accuracies, not inference. The 0.05 ridge
boundary and modest tSNR indicate that regularization/SNR remain limitations.

Canonical outputs:

- GLMsingle: `pilot_raw_pipeline_sub4_6/glmsingle_sdc_defaultgrid/Sub5`
- New decoding/maps: `pilot_raw_pipeline_sub4_6/decoding_new_sdc/Sub5`
- Matched legacy decoding/maps: `pilot_raw_pipeline_sub4_6/decoding_corrected_legacy/Sub5`
- Run-08 diagnostic: `pilot_raw_pipeline_sub4_6/smoke/diagnostic_sub05_run08_excluded_v1`

