# Sub-05 February SyN preflight correction

The first isolated February fMRIPrep launch was stopped deliberately during
workflow startup. It was not accepted as an analysis result.

- fMRIPrep: 25.1.3
- Input: the four-run Sub-05 February BIDS subset
- Observed startup message: fieldmap-less SyN was requested, but no estimator
  was created; all four runs would have skipped susceptibility-distortion
  correction.
- The BOLD sidecars were verified to contain `PhaseEncodingDirection: j-`,
  `EffectiveEchoSpacing`, and `TotalReadoutTime`.
- Root cause: sdcflows 2.13.0 restricted its T1w query to `ses-01`, excluding
  the valid sessionless `sub-05/anat/sub-05_T1w` before examining the BOLD
  phase-encoding metadata.
- Corrective action: keep the canonical BIDS files unchanged and pass a
  fieldmap BIDS filter whose session selector is `Query.OPTIONAL`. The isolated
  production run must also use SyN error mode so that a missing estimator
  fails closed.

The stopped output and its Docker work volume are disposable partial products.
They must not be resumed or merged.
