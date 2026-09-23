# Subject 4 spatial/estimator sensitivity analysis

This project tests how spatial representation and smoothing affect **single-trial,
within-subject Kastner/Wang ROI decoding** for Subject 4. It does not use LSS and
does not average trials into ERP-style pseudo-trials.

The original frozen core contains 32 decoding setups. The user-approved
`DESIGN_AMENDMENT_40.json` adds one processing-order workflow, producing 40
setups in the active production analysis:

- 2 estimator families: GLMsingle Type-D and historical-style SPM LS-A;
- 5 spatial workflows: acquired-grid BOLD, subject T1w, MNI152NLin6Asym
  res-native, MNI152NLin6Asym res-2, and native beta estimation followed by
  beta-map normalization to MNI res-2;
- 4 smoothing levels: 0, 3, 5, and 8 mm FWHM.

Only one fMRIPrep execution is used to generate all four spatial branches. The
independent GLM fits and decoding jobs may then run in parallel with bounded
concurrency.

The added workflow is not a duplicate of the MNI branches. It estimates the
single-trial betas on the acquired grid first, transforms those beta maps to
the 2-mm MNI target, and only then performs MNI-space ROI decoding. The original
MNI branches transform BOLD before GLM estimation.

## Important interpretation boundary

Smoothing follows each pipeline's established location:

- GLMsingle Type-D: independent mask-normalized smoothing of each estimated
  single-trial beta (post-GLM), matching the completed pilot comparison.
- SPM LS-A: spatial smoothing of each BOLD volume before model estimation,
  matching the historical SPM workflow.

Consequently, a GLMsingle-versus-SPM comparison is a comparison of two model
families, including their nuisance models and smoothing placement. Within an
estimator family, the space and smoothing factors are controlled.

The original Subject 4 SPM artifact is retained as an external historical
benchmark. Its `SPM.mat` proves 224 scans/run and pre-GLM 8 mm smoothing. It is
never overwritten.

## Persistent-storage rule

All inputs, outputs, logs, and analysis intermediates live under this directory
on `N:`. Installed programs, MATLAB/SPM, Docker Desktop, and Docker's transient
named work volume may reside on `C:`.
