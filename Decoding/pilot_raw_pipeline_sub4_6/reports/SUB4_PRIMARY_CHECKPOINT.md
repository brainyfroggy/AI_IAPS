# Sub4 primary raw-pipeline checkpoint

Status: passed fMRIPrep QC, standard-grid GLMsingle validation, matched decoding,
and voxelwise NIfTI integrity validation.

- fMRIPrep: PEPOLAR SDC for all 10 runs; 2 mm `MNI152NLin6Asym` output.
- GLMsingle: 244,217 mask voxels, 600 finite float32 Type-D trial betas,
  20-value fractional-ridge grid, 10 candidate noise PCs (3 selected).
- Ridge limitation: the minimum requested fraction, 0.05, was selected in
  241,278/244,217 voxels (98.7966%).
- Decoder: diagonal LDA, shrinkage 0.1, 5 mm searchlights, 0/3 mm additional
  smoothing, leave-one-run-out primary CV and identity-GroupKFold sensitivity.
- Output gate: 32 summary rows, 240 fold rows, and 32 float32 `.nii.gz` maps;
  every map passed mask/grid/range/NaN validation.

## Primary leave-one-run-out result

`Legacy` is matched leakage-safe decoding of the existing LS-A betas, whose
input images were already smoothed 8 mm. `New 0/3 mm` uses true single-trial
GLMsingle Type-D betas with that amount of additional mask-normalized smoothing.

| Contrast | Legacy whole | New 0 mm | Delta | New 3 mm | Delta | Legacy spatial | New 0 spatial | New 3 spatial |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Nat pleasant (within) | 0.650 | 0.675 | +0.025 | 0.700 | +0.050 | 0.5181 | 0.5126 | 0.5132 |
| AI pleasant (within) | 0.615 | 0.600 | -0.015 | 0.630 | +0.015 | 0.5076 | 0.5048 | 0.5067 |
| Nat unpleasant (within) | 0.620 | 0.590 | -0.030 | 0.615 | -0.005 | 0.5086 | 0.5068 | 0.5084 |
| AI unpleasant (within) | 0.490 | 0.620 | +0.130 | 0.610 | +0.120 | 0.5021 | 0.5042 | 0.5047 |
| Nat to AI pleasant | 0.530 | 0.580 | +0.050 | 0.600 | +0.070 | 0.5044 | 0.5047 | 0.5057 |
| AI to Nat pleasant | 0.630 | 0.590 | -0.040 | 0.600 | -0.030 | 0.5044 | 0.5050 | 0.5060 |
| Nat to AI unpleasant | 0.510 | 0.565 | +0.055 | 0.550 | +0.040 | 0.5021 | 0.5058 | 0.5061 |
| AI to Nat unpleasant | 0.520 | 0.615 | +0.095 | 0.615 | +0.095 | 0.5022 | 0.5060 | 0.5070 |

Mean pooled whole-mask accuracy:

- Within-source: legacy 0.59375; new 0 mm 0.62125 (+0.02750); new 3 mm
  0.63875 (+0.04500).
- Cross-source: legacy 0.54750; new 0 mm 0.58750 (+0.04000); new 3 mm
  0.59125 (+0.04375).
- All eight contrasts: legacy 0.57063; new 0 mm 0.60437 (+0.03375); new 3 mm
  0.61500 (+0.04437).

These are single-subject descriptive accuracies, not group inference. Spatial
mean searchlight accuracy and pooled whole-mask accuracy are different metrics.
