# AI-IAPS remaining-cohort production v4 operations record

Frozen and launched: 2026-07-30, America/New_York

This is the operational source of truth for reproducing, monitoring, auditing,
and safely resuming the 25-subject raw-data-to-ERP-decoding production run. The
incident-specific explanation is in
`audit/PRODUCTION_V4_PATH_RECOVERY_20260730.md`.

## Cohort and analysis scope

- Intended final cohort: 28 subjects.
- Completed pilot subjects: 4, 5, 6.
- Production subjects, in frozen order: 1, 2, 7, 9, 11-31 (25 subjects).
- Prespecified exclusions: 3 (Emily), 8 (earrings), 10 (not usable).
- Each production subject must have 10 runs, 60 trials per run, and 600 trials.
- No subject may be silently skipped. Any new QC exclusion must be made without
  inspecting decoding accuracy and recorded as a versioned amendment.

The three compared beta pipelines are the original SPM LS-A 8-mm pipeline,
GLMsingle Type-D without added beta smoothing, and GLMsingle Type-D with 3-mm
mask-normalized smoothing. ERP-style decoding is run only in the frozen
Kastner/Wang ROI set and includes both Historical Avg(random) reproduction and
the primary run-wise leave-one-run-out ERP means.

## Source locations

- Raw MRI: `N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\rawfMRI\IAPS-DEV`
  (WSL: `/mnt/n/Experimental_Data/yujunchen/projects/LAB_IAPS_AI/rawfMRI/IAPS-DEV`).
- Onset/data-recording logs:
  `N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\DataRecording`.
- Frozen stimulus order:
  `N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\stimuli_600trials.csv`.
- Exact approved source inventory: `manifests/source_selection.tsv` (394 rows;
  250 BOLD runs; every selected source has a frozen SHA-256).
- Cohort/acquisition rules: `config/cohort.json`.

All persistent inputs, outputs, audit evidence, and logs are stored on N:. C:
is used only by installed programs, Docker Desktop/WSL storage, the active
subject-scoped Docker work volume, and the FreeSurfer license.

## Host and runtime snapshot

- Windows 11 Enterprise 10.0.22631; PowerShell 7.6.4.
- Control Python: 3.11.13 at
  `C:\Users\yujunchen\AppData\Local\miniconda3\envs\neuro_161\python.exe`.
- WSL 2.7.11.0, kernel 6.18.33.2-2; execution distribution Ubuntu-22.04.
- WSL analysis Python: 3.11.14 at
  `/home/yujun/miniconda3/envs/pycortex/bin/python3.11`.
- Docker client/server 28.3.0, invoked only as
  `C:\Windows\System32\wsl.exe -d Ubuntu-22.04 -- /usr/bin/docker`.
- BIDS validator 3.0.1 at
  `/home/yujun/.cache/ai_iaps_bids_validator_venv/bin/bids-validator-deno`.
- fMRIPrep image:
  `nipreps/fmriprep@sha256:4e5cfd99f6d80a9ef10a87929f8e74e4caf9dc108b49551eb23c775e61cd16f7`.

Direct Windows Docker binds of the mapped N: drive are not approved on this
host. Every Docker N: bind is routed through Ubuntu-22.04 `/mnt/n`. The path
mapper accepts drive-letter N: paths and the exact resolved backing UNC share;
it rejects arbitrary UNC shares.

## Pilot-equivalent fMRIPrep settings

- fMRIPrep 25.1.3 at the exact digest above.
- 6 total threads; 2 OMP threads; 26,000 MB memory.
- Low-memory mode enabled; stop on first crash.
- FreeSurfer surface reconstruction disabled (`--fs-no-reconall`).
- Output space `MNI152NLin6Asym:res-2`.
- Same-session distortion correction only: measured PEPOLAR or GRE when
  present, otherwise an isolated fieldmap-less SyN branch. Fieldmaps are never
  borrowed across acquisition sessions.
- Mixed-session subjects are processed in isolated functional-session branches
  containing all subject T1w inputs and only the branch session's BOLD and
  same-session fieldmaps. Merge requires matching anatomical references and
  session-specific SDC evidence.
- One subject is processed at a time. Each subject uses a named Docker work
  volume, which is retired only after required derivatives, QC, checksums, and
  the archive receipt verify.

These are the compute settings that worked for Subjects 4-6. V4 changes the
host path boundary and storage admission policy, not these compute settings.

## Storage policy

- Hard C: launch minimum: 50 GiB (53,687,091,200 bytes).
- Preferred C: free space: 60 GiB.
- C: is measured immediately before every stage and recorded in that attempt.
- At v4 launch C: had approximately 59.22 GiB free; N: had 17.83 TiB free.
- The user authorized clearing only the Recycle Bin. It was cleared and
  recovered 0 MiB; no other user data were deleted or archived.
- Falling below the hard floor prevents the next stage from launching and is
  recorded as a failed gate. It does not delete or overwrite prior outputs.

## Immutable control identity

- Analysis ID: `ai_iaps_full28_sdc_glmsingle_erp_v4`.
- Freeze: `ANALYSIS_FREEZE_v4.json`, SHA-256
  `a56e49372707c9125fd32d6a26eed6be90cf472aaa1dc9b635c4f7534e4b9fdb`.
- Stage contract: `config/stage_contracts_v4.json`, SHA-256
  `6459f27b49811df607bedaf0abb82bc8dc38dde22afe0d6cdb83ff1468037a16`.
- Production approval: `runtime/production_approval_v4.json`, SHA-256
  `16766a027c771d24ca4cf187f1faa45c99f35c19621494c7c3e4ec7bea4c5cf2`.
- fMRIPrep/SDC configuration: `config/fmriprep_sdc_config.json`, SHA-256
  `8281b639601850bcf525f1a34283e694c68f651c2702caaaa7a769734c012ad0`.
- Hash-chained ledger: `logs/stage_attempts_v4/ledger.jsonl`.
- Batch stdout/stderr: `logs/batch_v4/runner.stdout.log` and
  `logs/batch_v4/runner.stderr.log`.

The approval pins the hashes and sizes of the freeze, contract, manifest,
configuration, scripts, tools, atlas, labels, and external inputs. Editing a
pinned file invalidates launch or resume. A required change must create a new
versioned control set; prior evidence is preserved.

## Output namespaces

- Canonical BIDS inputs: `runtime/production_v3/bids/SubXX`. This namespace is
  intentionally retained because Sub01 was built in v3 before the path-layer
  stop. V4 adopts an existing subject only after complete byte verification;
  later subjects are built by the same verified route.
- V4 fMRIPrep derivatives: `runtime/production_v4/fmriprep/SubXX`.
- V4 isolated SDC branch inputs and outputs:
  `runtime/production_v4/bids_branches/SubXX` and
  `runtime/production_v4/fmriprep_branch_outputs/SubXX`.
- V4 GLMsingle: `derivatives/production_v4/glmsingle/SubXX`.
- V4 ERP subject results: `results/production_v4/erp_subjects/SubXX`.
- V4 audit/QC/archives: `audit/production_v4/...`.
- V4 immutable attempts: `logs/stage_attempts_v4/subjects/SubXX/stages/...`.

Sub01's adopted BIDS tree contains 40 data/metadata files totaling
2,422,566,465 bytes. Every source image, generated file, static metadata file,
provenance identity, and the exact file set passed verification. The independent
official-validator UNC-path smoke report is
`audit/smoke_v4_bids_validation/Sub01.json` and contains zero errors.

## Stage order and gates

Each subject proceeds in this exact order:

1. `source_audit`: re-read exact selected sources, onsets, metadata, and hashes.
2. `bids_build`: build atomically through WSL, or fully verify and adopt an
   existing immutable BIDS tree.
3. `bids_validate`: official validator; any error stops the batch.
4. `sdc_preflight`: construct/audit isolated SDC branches and run external
   estimator/Docker-mount gates.
5. `fmriprep`: execute the frozen pilot-equivalent container configuration.
6. `quantitative_qc`: verify all ten expected res-2 runs and quantitative QC.
7. `visual_qc`: render visual evidence and require a hash-bound human decision.
8. `archive`: checksum required products and only then retire the exact Docker
   work volume.
9. `glmsingle`: generate Type-D betas (0-mm and independently smoothed 3-mm).
10. `erp`: Kastner ROI Historical Avg(random) and run-wise LORO ERP decoding.
11. `subject_gate`: independently verify the complete subject lifecycle.

A stage begins only after all earlier stages for that subject passed under the
same control identity. Existing successful stages cannot be executed twice.
Failures stop the entire sequential runner; there is no automatic skipping.

Visual QC is intentionally not automatic. A PENDING or REJECT decision blocks
archive, GLMsingle, ERP decoding, and the subject gate. The reviewer must inspect
the generated images and create the exact hash-bound ACCEPT attestation before
an audited resume.

## Launch command

Run from this project root with the control Python:

```powershell
$PY = 'C:\Users\yujunchen\AppData\Local\miniconda3\envs\neuro_161\python.exe'
& $PY .\code\run_remaining_cohort.py `
  --freeze .\ANALYSIS_FREEZE_v4.json `
  --contract .\config\stage_contracts_v4.json `
  --approval .\runtime\production_approval_v4.json `
  --log-root .\logs\stage_attempts_v4 `
  --execute `
  --ack-analysis-id ai_iaps_full28_sdc_glmsingle_erp_v4
```

The current run was launched as a hidden background process with stdout and
stderr redirected to the two N:-resident batch logs above.

## Monitoring and completion

Safe live checks do not change analysis state:

```powershell
Get-Content .\logs\batch_v4\runner.stdout.log -Tail 30
Get-Content .\logs\batch_v4\runner.stderr.log -Tail 30
Get-Content .\logs\stage_attempts_v4\ledger.jsonl -Tail 10

$PY = 'C:\Users\yujunchen\AppData\Local\miniconda3\envs\neuro_161\python.exe'
& $PY .\code\status_ledger.py `
  --freeze .\ANALYSIS_FREEZE_v4.json `
  --contract .\config\stage_contracts_v4.json `
  --approval .\runtime\production_approval_v4.json `
  --log-root .\logs\stage_attempts_v4
```

The full run is complete only when all of the following agree:

- batch stdout contains `FULL_BATCH_COMPLETE`;
- the verified ledger reports every one of the 275 stages passed;
- all 25 subjects have `subject_gate=passed` and `overall_status=complete`;
- the final aggregation independently verifies the exact 28-subject result set
  (3 pilot plus 25 production subjects) before group summaries are reported.

## Failure and resume policy

1. Do not delete a failed attempt, lock, staging directory, BIDS tree, Docker
   volume, or partial derivative.
2. Read batch stderr, the last ledger event, and the attempt's `stdout.log`,
   `stderr.log`, `attempt.json`, and `result.json`/completion evidence.
3. Confirm the runner is no longer active and identify the exact failed stage.
4. Correct only the documented cause. If any pinned file must change, create a
   new frozen control version; never mutate the v4 approval.
5. If no pinned identity changed and the existing failed attempt is audited,
   resume with the same launch command plus `--resume-failed`.
6. The runner re-verifies the entire ledger, event mirrors, attempt evidence,
   control identity, prior-stage completion, and current storage before resume.

No destructive cleanup is part of automatic recovery. Docker work-volume
retirement is limited to the exact volume named in the verified launch receipt
and occurs only after archive verification.

## Prelaunch evidence

- 36 production-control/SDC tests passed.
- 34 analysis/postprocessing tests passed.
- The official BIDS validator UNC-path smoke test passed with zero errors.
- Representative Sub01 PEPOLAR, Sub17 mixed GRE/SyN, and Sub19 GRE SDC
  preflights passed before production.
- The pinned container performed an N: read/write round trip through `/mnt/n`.
- No Docker/fMRIPrep process was launched by the superseded v3 run.

