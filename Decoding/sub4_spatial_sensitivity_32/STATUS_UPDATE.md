# Subject 4 spatial/estimator sensitivity — status update

Last updated: 2026-08-01 14:49 UTC (10:49 EDT)

## What was broken

The previous run (`production_v2_40`, PID 44184) waited for fMRIPrep attempt
05, which completed successfully (exit 0) at 2026-07-31 20:14 EDT. The very
next job, `validate_fmriprep_multispace`, failed immediately (exit 1).

Root cause: `run_glmsingle_space.py`'s `BRANCH_GLOBS` (shared by the
validator, the atlas-transform script, and the native-beta-to-MNI transform
script) assumed fMRIPrep 25.1.3 would emit literal `space-func` and
`res-native` filename tokens. It doesn't:

- The acquired/native grid has **no `space-` entity at all**:
  `sub-04_ses-01_task-iaps_run-01_desc-preproc_bold.nii.gz`
- Native-resolution MNI has **no `res-` entity** (only `mni_res_2` gets one):
  `sub-04_ses-01_task-iaps_run-01_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz`

This same wrong assumption (a literal `_space-func` substring) also broke the
coreg-transform filename lookup in `prepare_kastner_atlases.py` and
`transform_native_betas_to_mni.py`.

## What was fixed

Three files, all verified directly against the real
`preprocessing/fmriprep_multispace_attempt_05` derivatives before relaunch:

- `code/run_glmsingle_space.py` — corrected `BRANCH_GLOBS`, added an explicit
  exclusion for `_space-` in the acquired-grid glob, and fixed the confounds
  filename derivation (it was splitting the full filename on `_space-` instead
  of the already `_desc-preproc_bold`-stripped prefix, so acquired-grid
  confounds paths were never being stripped correctly).
- `code/prepare_kastner_atlases.py` — fixed the run-01 coreg-transform
  filename lookup for the acquired-grid atlas.
- `code/transform_native_betas_to_mni.py` — same fix for the native-beta-to-
  MNI workflow (the 5th spatial branch).

Full details are in `AUDIT_LOG.md` ("Production v2_40 attempt 01 failure and
correction"). Per this project's fail-closed convention, `production_v2_40`
and its ledger are retained untouched as the failed attempt's evidence — the
corrected run uses a new namespace, `production_v2_40_attempt02`, launched via
`code/run_production_after_fmriprep_attempt02.py` (identical logic, only the
output namespace changed).

## Preflight confirmed before relaunch

- WSL `Ubuntu-22.04` GLMsingle venv: present
- PowerShell 7 (`pwsh.exe`): present
- SPM12 r7771 (`C:\spm12_complete_r7771`): present
- MATLAB R2024a: present
- C: 225 GiB free, N: 17.3 TiB free (both well above the 50 GiB hard floor)

## COMPLETE (2026-08-03 13:32 UTC)

All 40 cells finished successfully. `production_v2_40_attempt02/FULL_40_COMPLETE.json`
confirms 5,440 subject-level rows and 54,400 fold-level rows across the 5
spatial workflows x 4 smoothing levels x 2 estimators.

Aggregated comparison: `production_v2_40_attempt02/summary/` contains
`all_subject_results.csv`, `all_fold_results.csv`, `setup_comparison.json`
(40-row ranked summary), `within_roi_detail.csv`, and `roi_drilldown.json`.

Interactive comparison artifact: https://claude.ai/code/artifact/f3c0813c-cc6d-4540-b629-da2ec0365fba

**Headline finding:** estimator choice dominates and flips direction between
tasks. Within-source decoding: GLMsingle Type-D 53.7% vs SPM LS-A
(historical) 48.5% -- GLMsingle wins every one of its 20 settings over every
one of SPM's 20 (no overlap). Cross-source generalization (Natural<->AI):
reverses to SPM LS-A 53.9% vs GLMsingle 52.0%. Within either estimator, the
spatial workflow / smoothing spread is only 1-2.4 points -- far smaller than
the ~5-point estimator gap. GLMsingle beats SPM LS-A in 17/17 Kastner ROIs
at a matched spatial setting (Subject T1w, 8mm).

## Incident + resolution (2026-08-02 23:21-23:35 UTC / 19:21-19:35 EDT)

Both protected jobs (`mni_res_native` 3mm, 5mm) finished **naturally and
successfully** (600 trial betas each, SPM.mat + provenance.json + trial
index all present and validated) -- zero compute lost, exactly as designed.

The handoff script then hit a real bug: killing the original orchestrator's
python.exe does not kill its grandchild MATLAB processes (pwsh.exe spawns
matlab.exe spawns MATLAB.exe -- three separate processes). My orphan-killer
only matched on `run_spm_lsa_job.ps1`, which is only in the pwsh wrapper's
command line, not MATLAB's. Two orphaned MATLAB processes were left running
unsupervised holding a lock on a partial `mni_res_native` 8mm smoothing file,
which crashed the cleanup step (`PermissionError`) and the whole finish
script exited (exit code 1).

Fixed: broadened the orphan match to also catch `run_spm_lsa(` (present on
every process in the pwsh -> matlab.exe -> MATLAB.exe chain, confirmed
directly from their real command lines), added a retry-with-backoff around
the directory cleanup, and re-scan-before-delete as a second safety net.
Manually verified before relaunch: both orphaned MATLAB processes killed,
file lock released, corrupted partial `smoothing_8mm` directory removed, and
the two protected cells' outputs independently re-validated as complete and
correct. No data from any completed cell was affected.

Relaunched immediately after the fix. This time it worked as intended: all 5
remaining SPM cells (`mni_res_native` 8mm + all 4 `mni_res_2` smoothing
levels) are now running **simultaneously** (confirmed via live process list:
5 pwsh wrappers + 5 matlab.exe + 5 MATLAB.exe, all correctly mapped to
distinct cells), instead of 2 at a time.

## Second incident + resolution (2026-08-03 11:29-11:46 UTC)

Good news buried in this one: **the entire 16-cell SPM LSA stage finished
successfully** (all 5 remaining cells ran genuinely in parallel and passed).

The next stage -- 5 native-beta-to-MNI transform jobs, run at
`TRANSFORM_WORKERS=5` -- failed across the board, but for two different
reasons:

1. **OOM**: 3 of the 5 (`glmsingle_typed`, `smoothing_0mm`, `smoothing_8mm`)
   died with Docker exit code 137 (SIGKILL/OOM) inside `antsApplyTransforms`.
   WSL2 here is capped at 32 GiB via `.wslconfig` (not the host's 128 GiB),
   and 5 concurrent ANTs transforms of 600-volume 4D images exceeded that.
   Fixed by dropping `TRANSFORM_WORKERS` from 5 to 2.
2. **Pre-existing, unrelated bug**: the other 2 (`smoothing_3mm`,
   `smoothing_5mm`) actually completed the real ANTs computation
   successfully, then crashed writing `provenance.json`:
   `reference.header.get_zooms()` returns numpy `float32` values, and
   `list(...)` doesn't convert them to native Python floats, so
   `json.dumps` rejected them. This bug was already in
   `transform_native_betas_to_mni.py` before any of my changes -- fixed with
   a one-line cast (`[float(value) for value in ...]`), matching the pattern
   already used correctly elsewhere in this codebase.

Since my own dynamic "is this cell done" check was keyed on
`transformed_betas.nii.gz` rather than `provenance.json`, and that file
already existed for the 2 cells that got that far, I also changed the marker
to `provenance.json` (matching the SPM stage's completion-detection logic)
and had the script fully clear + rerun all 5 cells rather than try to resume
mid-pipeline. Added corresponding cleanup for stale output directories and
stale job-log directories so reruns don't trip the fail-closed
already-exists guards. Confirmed via ledger: all 5 stale directories and 5
stale job logs cleared correctly, SPM stage correctly recognized as already
done and skipped, and the transform stage is now running at a safe 2-way
concurrency.

## Current status: RUNNING (as of 2026-08-02 18:31 UTC / 14:31 EDT)

Update: `subject_t1w` branch finished all 4 smoothing levels (0/3/5/8mm).
`mni_res_native` now started: 0mm done (61 min), 3mm and 5mm running
(~1.5h and ~1.4h in so far). `mni_res_2` (4 jobs) not yet started.
**9 of 16 SPM jobs done, 2 running, 5 remaining.** No failures. Pattern
holds: 0mm ~60-130 min, smoothed 3/5/8mm ~5-8h each.

## Earlier snapshot (2026-08-02 15:20 UTC / 11:20 EDT)

Progress so far, all exit code 0 (no failures):

1. `validate_fmriprep_multispace` — done (254.7 s)
2. `prepare_kastner_atlases` — done (32.2 s)
3. `prepare_spm_inputs` (4 branches, 2 workers) — done (~12.8 min wall time)
4. `glmsingle` (4 branches, 2 workers) — done, ~1.4-2.2 hours per branch.
   Stage wall time: 15:07-18:48 UTC Aug 1 (~3h42m).
5. `spm_lsa` (4 branches × 4 smoothing = 16 MATLAB jobs, 2 workers) —
   **in progress since 18:48 UTC Aug 1**. Confirmed pattern: 0mm jobs finish
   in ~70-131 min; smoothed (3/5/8mm) jobs take ~5.7-8 hours each. This is
   consistent across both branches finished so far, so it is the real cost
   of this stage, not a stall.
   - `bold_acquired_grid`: all 4 smoothing levels (0/3/5/8mm) — **done**
     (131, 467, 478, 461 min respectively)
   - `subject_t1w`: 0mm done (70 min), 3mm done (342 min); **5mm running**
     since 10:16 UTC (~5h so far), **8mm running** since 11:49 UTC (~3.5h so
     far)
   - `mni_res_native`: not started
   - `mni_res_2`: not started
   - 6 of 16 SPM jobs done, 2 running, 8 not yet started
6. `normalize_native_betas_to_mni_res_2` — not started
7. `single_trial_roi_decoding` (40 jobs) — not started
8. Aggregation / `FULL_40_COMPLETE.json` — not started

**Revised time expectation:** smoothed SPM jobs are consistently taking
5.7-8 hours each regardless of branch, vs ~70-130 min for unsmoothed (0mm).
With 2 concurrent workers and 6 more smoothed jobs plus 2 more unsmoothed
jobs remaining across `mni_res_native`/`mni_res_2`, stage 5 alone likely
needs another ~1-1.5 days before moving to native-beta-to-MNI transform and
decoding (both fast, minutes each based on earlier stage timing). No
failures; this is simply how long AR(1) whole-brain LS-A estimation with
pre-GLM smoothing takes on this data/hardware.

## How to check progress

- Ledger (append-only, one line per event):
  `logs/production_v2_40_attempt02/ledger.jsonl`
- Per-job stdout/stderr:
  `logs/production_v2_40_attempt02/jobs/<job_name>/{stdout,stderr}.log`
- Driver process log: `logs/production_v2_40_attempt02_driver.log`
- Success marker (only appears when all 40 cells finish and pass the
  row-count sanity checks): `production_v2_40_attempt02/FULL_40_COMPLETE.json`

## Next step

I'm monitoring this run in the background. Once `FULL_40_COMPLETE.json`
appears, I'll build a comparison view across all 5 spatial branches × 4
smoothing levels × 2 estimators in the 17 Kastner/Wang ROIs so you can see
which setting decodes best, and update this file with the result. If any
stage fails, I'll investigate the root cause before retrying rather than
blindly relaunching.
