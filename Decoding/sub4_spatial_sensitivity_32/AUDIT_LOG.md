# Audit log

All timestamps are America/New_York unless the source log states otherwise.
Failed attempts are retained; none is silently overwritten or reused.

## User-approved 40-cell amendment

- Before production modeling began, the user added a fifth spatial workflow:
  estimate native/acquired-grid single-trial betas, transform the beta maps to
  MNI152NLin6Asym res-2, and decode there.
- This adds 8 cells (2 estimators x 4 smoothing levels), for 40 total.
- `DESIGN_FREEZE.json` is retained unchanged as the original 32-cell freeze.
  `DESIGN_AMENDMENT_40.json` records the added workflow and interpretation.
- The waiting `production_v1` 32-cell launcher is superseded before fMRIPrep
  completion and before it created models or results. Its append-only ledger is
  retained. The active run uses isolated namespace `production_v2_40` and
  ledger `logs/production_v2_40/ledger.jsonl`.
- Beta images use linear interpolation; the support mask uses GenericLabel.
  The functional-to-anatomical transform is the fixed run-01 transform already
  frozen for the common acquired-grid branch, followed by T1w-to-MNI nonlinear
  normalization.
- The 40-cell runner launched at 2026-07-31 19:49 EDT as PID 44184. Its first
  ledger records are `RUN_START`, `WAIT_FMRIPREP_START`, and a healthy wait
  heartbeat. At launch, C: had 236.80 GiB free and N: had 17,366.55 GiB free.
  Post-fMRIPrep work is capped at two concurrent independent jobs.

## Frozen analysis

- `DESIGN_FREEZE.json` defines 32 cells: 2 estimator families × 4 spatial
  branches × 4 smoothing levels.
- Subject: 4 only.
- Trial unit: individual trial beta; no LSS and no ERP/pseudo-trial averaging.
- Decoder: linear SVC, C=1, ten leave-one-run-out folds, per-voxel scaling fit
  on outer-training trials only.
- Atlas: 17 frozen Kastner/Wang ROIs; IPS combines IPS0 through IPS5.

## Historical SPM ground truth

- The authoritative artifact is `GLM_singletrial/betas/Sub4/SPM.mat`.
- Verified design: 2240 scans × 670 columns (600 trials, 60 motion, 10 run
  constants), 224 scans/run, canonical HRF, 3 s duration, microtime 64/32,
  global scaling, 128 s high-pass, and AR(1).
- Input header and the saved preprocessing batch prove 8 mm isotropic
  smoothing before GLM estimation.
- `SPM.mat` records `spm_spm.m` revision 7738. Official SPM12 package r7771
  contains that exact estimator file revision.

## fMRIPrep multi-space launch record

1. Attempt 01 stopped during command-line parsing because fMRIPrep 25.1.3
   rejects `bold` as a nonstandard space token. No preprocessing ran.
2. Attempt 02 used the valid `func` token but the Windows Docker CLI exposed
   the mapped N: bind as an empty directory. No preprocessing ran.
3. Attempt 03 verified that Ubuntu-22.04 could read the BIDS data, but a
   cross-shell Go-template quoting error stopped the wrapper at image-digest
   validation. No container was created.
4. Attempt 04 reached fMRIPrep argument validation, then stopped because a
   custom path converter retained Windows backslashes. No preprocessing ran.
5. Attempt 05 preflighted the BIDS root, output directory, and FreeSurfer
   license from inside the container. It launched successfully through
   Ubuntu-22.04 and selected the measured AP/PA PEPOLAR estimator. Output
   spaces are `func`, `T1w`, `MNI152NLin6Asym:res-native`, and
   `MNI152NLin6Asym:res-2`.

Scientific settings did not change across attempts other than correcting the
invalid functional-space token. Each attempt has isolated logs and, where
created, an isolated output/work namespace.

## SPM runtime recovery

- The pre-existing `C:\spm12` reports package r7771 but lacks
  `matlabbatch/cfg_getfile`, so it cannot initialize a complete SPM session.
- It was left untouched.
- Official `spm/spm12` tag r7771 was installed separately at
  `C:\spm12_complete_r7771`, Git commit
  `3085dac00ac804adb190a7e82c6ef11866c8af02`.
- MATLAB/SPM initialization passed with this installation. Its `spm_spm.m` and
  `spm_fmri_spm_ui.m` files are both revision 7738.
- The first MNI res-2 LS-A job launch failed before modeling because the
  background call split MATLAB's batch expression.
- The next launch reached the incomplete SPM installation and stopped before
  model specification.
- Attempt 03 uses the complete official r7771 installation and reached SPM
  design creation and global calculation.

## Completed cells

- GLMsingle Type-D, MNI res-2, 0 mm post-beta smoothing: complete. The result
  contains 136 subject-level values (17 ROIs × 8 contrasts) and 1360 fold-level
  values (× 10 held-out runs). This uses the already validated pilot Type-D
  artifact; final inclusion is conditional on equivalence checks against the
  new multi-space run's MNI res-2 derivative.

## Production v2_40 attempt 01 failure and correction

- The 40-cell runner launched 2026-07-31 19:49 EDT (PID 44184) waited for
  fMRIPrep attempt 05, which completed successfully at 20:14 EDT (exit 0).
- `validate_fmriprep_multispace` then failed immediately (exit 1, 2026-08-01
  00:14:40 UTC). Root cause: `BRANCH_GLOBS` in `run_glmsingle_space.py`
  assumed fMRIPrep would emit literal `space-func` and `res-native` filename
  tokens. fMRIPrep 25.1.3 instead omits the space entity entirely for the
  acquired/native grid (e.g. `sub-04_ses-01_task-iaps_run-01_desc-preproc_bold.nii.gz`)
  and omits the resolution entity entirely for native-resolution MNI
  (`..._space-MNI152NLin6Asym_desc-preproc_bold.nii.gz`, no `res-` token).
  Confirmed directly against `preprocessing/fmriprep_multispace_attempt_05`.
- Fix applied to `run_glmsingle_space.py` (shared by the validator and the
  GLMsingle branch runner): `bold_acquired_grid`'s glob now matches
  `*_desc-preproc_bold.nii*` and explicitly excludes any filename containing
  `_space-`; `mni_res_native`'s glob dropped the nonexistent `res-native`
  token. A second, related bug in the same function derived the confounds
  filename by splitting the full filename on `_space-`, which is a no-op for
  the acquired-grid branch (no `_space-` token present) and left
  `_desc-preproc_bold` stuck in the confounds path. Fixed to split the
  already-`desc-preproc_bold`-stripped prefix instead. All four branches
  verified to discover exactly 10 runs with correct bold/mask/confounds/json
  companions against the real attempt_05 derivatives after the fix.
- Per the fail-closed/no-silent-resume policy, `production_v2_40` and its
  ledger are retained unchanged as the failed attempt's evidence. Production
  resumes under new namespace `production_v2_40_attempt02` via
  `code/run_production_after_fmriprep_attempt02.py` (identical logic, only the
  namespace paths changed).
- The same false `_space-func` assumption was also present in
  `prepare_kastner_atlases.py` (deriving the run-01 boldref-to-T1w coreg
  transform filename for the acquired-grid atlas) and
  `transform_native_betas_to_mni.py` (deriving the same transform filename for
  the native-beta-to-MNI workflow). Both split on the nonexistent
  `_space-func` token, which is a no-op and produces a filename glob that
  matches nothing. Both fixed to split on `_desc-preproc_bold` instead, and
  both verified directly against attempt_05: the coreg transform, the
  MNI-to-T1w transform, and the T1w-to-MNI transform all resolve to exactly
  one file each.
- Preflight before relaunch confirmed: WSL `Ubuntu-22.04` GLMsingle venv at
  `/home/yujun/.cache/ai_iaps_pilot_venv` present; PowerShell 7 at
  `C:\Program Files\PowerShell\7\pwsh.exe` present; SPM12 r7771 at
  `C:\spm12_complete_r7771` present; MATLAB R2024a present; C: 225.18 GiB free,
  N: 17,352.55 GiB free (both well above the 50 GiB hard floor).

## Storage policy

- Persistent data, code, outputs, logs, and intermediates are on N:.
- MATLAB, SPM, Docker Desktop, and Docker's transient named work volume are on
  C: as programs/runtime state.
- The frozen C: hard floor is 50 GiB. It was above 250 GiB at launch.
