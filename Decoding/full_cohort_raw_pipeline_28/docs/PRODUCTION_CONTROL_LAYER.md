# Production control layer

## Purpose

This layer controls the 25-subject production batch for analysis
`ai_iaps_full28_sdc_glmsingle_erp_v1`. It contains no decoding or GLM numerical
code. Its job is to prevent an unreviewed command, changed input, low-storage
launch, out-of-order stage, duplicate stage, or silently omitted subject from
entering the production record.

The approved production order is fixed by `ANALYSIS_FREEZE.json`:

`1, 2, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31`.

Pilot subjects 4, 5, and 6 are not rerun by this batch. Subjects 3, 8, and 10
cannot be inserted because they are absent from both the frozen and approved
production-subject list.

## Files and responsibilities

- `config/stage_contracts.json`: exact argv templates, lifecycle order,
  required pinned files, N:-drive output destinations, and the C: >=60 GiB
  gate. A stage marked `blocked` cannot be launched.
- `code/freeze_production_controls.py`: creates the one-time immutable
  `runtime/production_approval.json`. It never overwrites an approval. Both the
  freeze-declared canonical cohort file and the detailed acquisition config are
  independently pinned.
- `code/audit_stage_runner.py`: renders one exact command and records one
  attempt. It accepts no free-form command.
- `code/run_remaining_cohort.py`: visits all 25 subjects and all stages in the
  frozen order. There is no subject-skip or stage-skip option.
- `code/status_ledger.py`: verifies the entire ledger/hash mirror and reports
  one row for each of the 25 subjects.
- `code/production_control.py`: shared validation and hash-chain functions.
- `code/production_stage_helpers.py`: atomic BIDS-validator adapter and an
  explicit fail-closed command for unapproved stages.
- `code/production_postproc_gates.py`: all-session, hash-bound visual review;
  N:-resident checksum verification and exact Docker work-volume retirement;
  and the final GLMsingle/ERP/ledger subject gate. Operational details are in
  `docs/POST_FMRIPREP_GATES.md`.

All attempt files, stdout/stderr, event mirrors, BIDS datasets, fMRIPrep
outputs, QC, GLMsingle outputs, and ERP outputs are directed to the N: project
tree. C: is used only for installed executables, the FreeSurfer license, WSL,
and Docker Desktop storage that those programs require.

## Attempt invariants

Immediately before a command starts, `audit_stage_runner.py` requires and
records all of the following:

1. The freeze status is exactly `frozen_for_production` and contains exactly 28
   included subjects and the exact 25 remaining subjects.
2. `runtime/production_approval.json` is present, says
   `approved_for_production`, and has the same analysis ID and production
   subject order.
3. SHA-256 and byte size of the freeze, freeze-declared canonical cohort file,
   detailed acquisition config, source manifest, stage contract, executable,
   scripts, atlas/manifests, and every stage-required control file still match
   the approval. The canonical cohort hash must also equal the hash written
   inside the freeze.
4. The requested stage is `executable`, and its argv is rendered exactly from
   the approved template. The runner has no positional/free-form command
   interface.
5. The immediately preceding stage for the same subject passed under the same
   control identity.
6. No uncompleted attempt, subject lifecycle lock, or passed duplicate exists.
7. C: free space is measured afresh and is at least 64,424,509,440 bytes
   (60 GiB). This measurement is embedded in `attempt.json` and the start event.
8. A failed or storage-blocked attempt is retried only with explicit `--resume`
   and an identical command/config/control signature.

The global `ledger.jsonl` is append-only and SHA-256 chained. Each line has a
separately created immutable mirror in `status_history/events`. Missing,
changed, reordered, truncated, or extra mirror history stops all new work.
Per-subject `.lifecycle.lock` files prevent overlapping stages. Stale locks and
started attempts without a completion event require manual audit; they are
never cleared automatically.

## Prelaunch state on 2026-07-29

Production is intentionally unable to start. This is expected and correct.

- `ANALYSIS_FREEZE.json` is still
  `prelaunch_pending_source_manifest_and_bids_validation`.
- The reviewed production source manifest has not yet been installed as the
  immutable approved manifest.
- C: must pass the current 60 GiB free-space gate.
- Two stage contracts remain fail-closed:
  - `sdc_preflight`: needs a subject/session-aware validator that proves
    measured PEPOLAR/GRE assignment and SyN only where the same session lacks a
    measured fieldmap.
  - `fmriprep`: depends on that SDC policy and exact per-subject filter/launcher
    proof.

The `visual_qc`, `archive`, and `subject_gate` commands are now executable and
tested. `visual_qc` deliberately exits non-zero after its first rendering pass
until a human accepts every run in the manifest-hash-bound attestation; see
`docs/POST_FMRIPREP_GATES.md`.

The production approval generator also requires all 11 stages to be executable,
so changing only the freeze status cannot bypass these blockers.

## Safe audit commands

Use the explicit Python executable because the bare `python` command on this
Windows host resolves to the Microsoft Store alias.

```powershell
$Py = 'C:\Users\yujunchen\AppData\Local\miniconda3\envs\neuro_161\python.exe'
$Root = 'N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\full_cohort_raw_pipeline_28'

# Expected to fail until every launch gate above is resolved.
& $Py "$Root\code\freeze_production_controls.py" --dry-run

# Once an approval exists, print the exact 25 x 11 plan without launching.
& $Py "$Root\code\run_remaining_cohort.py"

# Verify and display every subject's ledger-derived status.
& $Py "$Root\code\status_ledger.py"
```

## One-time production approval

After the reviewed manifest is installed, every blocked stage has an approved
implementation, the freeze is formally set to `frozen_for_production`, and C:
passes the storage gate:

```powershell
& $Py "$Root\code\freeze_production_controls.py" --dry-run
& $Py "$Root\code\freeze_production_controls.py"
```

Review `runtime/production_approval.json` and preserve it. Do not edit it. If a
frozen file must change before launch, remove no history; create a new analysis
version, new approval destination, and new log root.

## Launch and resume

The launch requires an explicit analysis-ID acknowledgement:

```powershell
& $Py "$Root\code\run_remaining_cohort.py" `
  --execute `
  --ack-analysis-id ai_iaps_full28_sdc_glmsingle_erp_v1
```

The batch stops on the first failed or blocked attempt. After inspecting that
attempt's immutable `attempt.json`, `result.json`, `stdout.log`, and
`stderr.log`, retry the exact signature with:

```powershell
& $Py "$Root\code\run_remaining_cohort.py" `
  --execute `
  --resume-failed `
  --ack-analysis-id ai_iaps_full28_sdc_glmsingle_erp_v1
```

Already passed stages are printed as `VERIFIED-COMPLETE`; they are not silently
skipped. A changed command or config cannot resume under the prior approval.

## Test evidence

Run all control and analysis tests with:

```powershell
& $Py -m unittest discover -s "$Root\code\tests" -p 'test_*.py' -v
```

The control tests cover arbitrary-command rejection, exact subject membership,
freeze and config hash mismatch, current storage failure, explicit resume,
prior-stage dependency, duplicate execution, lifecycle locks, ledger tampering,
blocked contracts, exact 25-subject manifest validation, and BIDS-validator
report parsing.
