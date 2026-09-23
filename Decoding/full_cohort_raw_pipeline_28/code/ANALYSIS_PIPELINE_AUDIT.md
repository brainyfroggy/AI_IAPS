# Full-cohort GLMsingle and ERP analysis code audit

Status on 2026-07-29: code implemented and unit-tested; no production
GLMsingle or decoding job was launched by this work.

## Source of truth

Every entry point reads `../ANALYSIS_FREEZE.json`. Subjects outside its 28
included subjects fail before analysis. The code also verifies the fixed three
pipelines, eight contrasts, and 17 plotted Kastner/Wang ROIs.

The numerical GLMsingle and decoding routines are reused from the immutable
Sub4--6 pilot. The full-cohort files do not modify the pilot files.

## GLMsingle entry point

`run_glmsingle.py` accepts any integer subject and then requires membership in
the frozen cohort. It derives the ten run-order session indicators from the
fMRIPrep BIDS `ses-` entities and cross-checks them against
`../config/cohort.json`. A path/config disagreement is fatal. A subject without
`ses-` entities can use the config mapping; if neither contains multiple
sessions, the only allowed fallback is ten indicators for one session.

Before a production fit it checks:

- GLMsingle commit `1ab54a65edd3ea41a6133d4b4ecb78a9c7296684`;
- the two ordered patch SHA-256 values in the freeze;
- the exact 20-value fractional-ridge grid from 1.00 through 0.05;
- 10 candidate noise PCs, TR 1.8 s, duration 3.0 s, and seed 20260728;
- ten fMRIPrep BOLD/mask/confounds/JSON sets and run volume invariants;
- a non-existing production output destination.

Production rejects fixed/coarse ridge overrides, voxel limits, overwrite, and
paths containing `failed`, `smoke`, `coarse`, or `test`. The
`--nonproduction` switch exists only for explicit isolated smoke tests and its
use is recorded in provenance.

Safe preflight form:

```powershell
python code/run_glmsingle.py `
  --subject 11 `
  --fmriprep-root <verified-fmriprep-root> `
  --bids-root <verified-bids-root> `
  --validate-only `
  --preflight-json audit/Sub11_glmsingle_preflight.json
```

### Runtime dependency gate

Resolved on 2026-07-29 without network access. The exact `fracridge==2.0`
files were restored from the verified WSL pilot installation into
`../vendor/python`. `run_glmsingle.py` now checks the pinned RECORD SHA-256,
all 12 RECORD-backed payload hashes and sizes, version, module origin, and
distribution-metadata origin before loading GLMsingle. That validation is
included in preflight and subject provenance. Recovery evidence and the full
file manifest are in `../audit/FRACRIDGE_RUNTIME_REPAIR_20260729.md` and
`../vendor/python/FRACRIDGE_VENDOR_PROVENANCE.json`.

## Per-subject ERP entry point

`decode_erp_subject.py` processes one subject and all of these branches:

1. legacy SPM LS-A, already smoothed 8 mm;
2. GLMsingle Type-D, no added smoothing;
3. the same Type-D trials after independent mask-normalized 3 mm smoothing.

For every branch it runs the exact historical Avg(random) decoder and the
run-wise LORO condition-mean decoder. It does not produce group statistics or
plots. It writes into a sibling staging directory and publishes by atomic
rename only after all validation passes. A failed staging directory is
preserved with a `.failed-<timestamp>-...` name. `--resume` skips a completed
subject only when its request signature exactly matches the current freeze,
code, atlas, manifests, legacy beta set, and GLMsingle inputs.

Required rows per completed subject:

| File content | Rows |
|---|---:|
| Historical subject accuracies | 408 |
| Historical fold/repeat rows | 20,400 |
| Run-wise subject accuracies | 408 |
| Run-wise held-run rows | 4,080 |
| ROI voxel counts | 51 |

The validator checks exact keys, not row counts alone. Accuracy must be finite
and in `[0,1]`; each ROI must retain at least 10 analysis voxels.

Example after a subject has a validated Type-D result:

```powershell
python code/decode_erp_subject.py `
  --subject 11 `
  --glmsingle-subject-root <typed-output-for-Sub11> `
  --output <subject-result-root>/Sub11 `
  --resume
```

## Deterministic aggregation

`aggregate_erp_results.py` reads only complete per-subject directories. By
default it requires exact equality with the 28 frozen subjects and rejects
duplicates, missing subjects, unexpected subjects, mismatched freeze hashes,
and incomplete row/fold keys. `--allow-partial` must be supplied explicitly for
an interim progress review.

Means and SEMs are computed from subject-level accuracies, never fold rows. The
historical legacy branch is compared cell-by-cell with the attached historical
result and must pass the pilot's fixed numerical tolerance. Plot labels derive
`n` from the actual subject rows; no sample size is hard-coded. Both plots show
only the 17 frozen Kastner/Wang regions.

Optional `--export-roi-maps` creates atlas-filled ROI mean-accuracy `.nii.gz`
files plus JSON sidecars. These outputs are explicitly named and described as
ROI group-summary maps, **not voxelwise whole-brain decoding maps**. Voxels
outside the selected ROIs contain zero.

## Tests run

The command below passed 15 tests in the `neuro_161` environment:

```powershell
python -m unittest discover -s tests -p test_full_cohort_analysis.py -v
```

Covered gates include the 28-subject freeze, arbitrary-subject CLI parsing,
session config/BIDS agreement, pinned GLMsingle constants, exact result and fold
keys, duplicate/missing-key failure, dynamic `n` labels, exact ROI-count keys,
exact RECORD-pinned `fracridge==2.0` import, and a real NIfTI/JSON atlas-filled
ROI-map export. A one-subject partial
aggregation integration test also verified atomic publication, attached-result
reproduction, both ROI-only plots, and dynamic sample size. The immutable pilot
ERP self-test also passed independently.
