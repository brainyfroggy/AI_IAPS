# Sub5 February SyN estimator-discovery fix

Date: 2026-07-28  
Pinned runtime: fMRIPrep 25.1.3, sdcflows 2.13.1, PyBIDS 0.19.0

## Diagnosis

The February BOLD files are correctly labelled `ses-01` and have
`PhaseEncodingDirection: j-`. The T1w is valid subject-level/sessionless data
at `sub-05/anat/sub-05_T1w.nii.gz`.

In this runtime, `sdcflows.utils.wrangler.find_estimators()` obtains
`sessions=['01']` from the layout and reuses that list in its anatomical
query. The query therefore asks for a T1w with `session=['01']` and excludes
the sessionless T1w. Estimator discovery returns an empty list before the BOLD
phase-encoding metadata can produce an anatomical (SyN) estimator.

## Fix

The fMRIPrep BIDS filter sets only `fmap.session` to serialized PyBIDS
`Query.OPTIONAL`:

```json
{
  "fmap": {
    "session": "<Query.OPTIONAL: 3>"
  }
}
```

This makes sdcflows discovery include sessionless anatomy and session-labelled
BOLD. It does not rename either acquisition, put the March T1w under ses-01,
or change ordinary BOLD/anatomical input selection.

The runner now supports `-BidsFilterFile`, validated `-SynSdcMode warn|error`,
and `-RequireSynAnatEstimator`. Its exact-image preflight runs before it creates
an output directory or work volume and requires every selected BOLD input to
be covered by an `ANAT` estimator. The isolated branch uses
`-SynSdcMode error` as a second fail-closed gate.

## Real-data preflight result

Input was mounted read-only from
`C:\Users\yujunchen\.cache\ai_iaps_fmriprep\bids_sub05_feb_syn`.

```text
SyN ANAT estimator preflight passed: 4 estimators cover all 4 BOLD inputs
Preflight-only gate passed; fMRIPrep was not launched.
```

The four targets are runs 01, 03, 04, and 05, each with
`PhaseEncodingDirection: j-`. The C-local filter used for the gate is
`C:\Users\yujunchen\.cache\ai_iaps_fmriprep\bids_sub05_feb_syn_fmriprep_bids_filter.json`;
SHA-256 is
`ACA68315DE44714D07BC182CCCAE64A8A619485FF2EA2734B588CC6667751195`.

The full filesystem/unit suite passed 33/33 tests after the change; the config
JSON and PowerShell runner also passed syntax parsing.

## Provenance caveat

Sub5's sessionless T1w was acquired in March and is intentionally reused as
the anatomical reference for the February BOLD runs. The filter preserves that
fact by leaving the source BIDS entities unchanged. Retain the filter, this
report, and both fMRIPrep branch logs with the merged derivative. The preflight
proves estimator assignment, but the completed February reportlets must still
show SyN and pass visual SDC/coregistration QC before merging.
