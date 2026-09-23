# AI-IAPS production v6 operations

V6 is the active 25-subject production control set. It supersedes v5 only to
recover Subject 1's successful fMRIPrep output and correct Windows publication
path handling. The imaging and decoding analysis plan is unchanged.

## Active run

- Analysis ID: `ai_iaps_full28_sdc_glmsingle_erp_v6`
- Subjects: 1, 2, 7, 9, 11-31
- Exclusions: 3, 8, 10
- Pilot subjects retained: 4, 5, 6
- Runner PID at launch: 43180
- Batch stdout: `logs/batch_v6/runner.stdout.log`
- Batch stderr: `logs/batch_v6/runner.stderr.log`
- Hash-chained ledger: `logs/stage_attempts_v6/ledger.jsonl`
- Monitor: `monitor-ai-iaps-production-v4` (name and prompt updated to v6)

## Frozen controls

- `ANALYSIS_FREEZE_v6.json`
  SHA-256 `6dabcadb8aa299117f89c0d0440987cef5588c9d4310820732276d46aa2f0208`
- `config/stage_contracts_v6.json`
  SHA-256 `418f461371e32f1c90f3b4666ac60445148beb23de0c0ea5ec1daa5da9efb53b`
- `runtime/production_approval_v6.json`
  SHA-256 `886647e01573cec07005e2056c61848d6e2fa5da370f8be6fe1b140392140ede`
- `code/fmriprep_sdc_workflow_v6.py`
  SHA-256 `82bb5e56ebb09cc14b3fbb9c800968f3731c614ac107c9814e11f32d80154a17`
- Recovery attestation
  SHA-256 `66a4eaa9f29ca4035ef40fba3cd8f2f4aaec901866cce591081d9a1307446094`

Do not edit these files. A required change creates another version and retains
all v6 evidence.

## Compute configuration

Every fresh fMRIPrep branch retains the successful pilot profile: fMRIPrep
25.1.3 at the pinned digest, 6 threads, 2 OMP threads, 26,000 MB, low-memory
mode, no recon-all, stop on first crash, and MNI152NLin6Asym res-2 output.
Measured same-session SDC is preferred; isolated SyN branches are used only
where measured same-session fieldmaps are unavailable.

## Subject 1 recovery

V6 repeats only quick source/BIDS/SDC audit stages for Subject 1. Its fMRIPrep
stage verifies and publishes the completed v5 derivative without recompute.
The v5 source tree and failed-attempt evidence remain intact. The exact recovery
details are in `audit/PRODUCTION_V5_SUB01_PUBLICATION_FAILURE_20260730.md`.

## Monitoring and stop conditions

Read-only status checks:

```powershell
Get-Content .\logs\batch_v6\runner.stdout.log -Tail 30
Get-Content .\logs\batch_v6\runner.stderr.log -Tail 30
Get-Content .\logs\stage_attempts_v6\ledger.jsonl -Tail 10
```

Stop and report rather than automatically resuming if a stage fails, the
runner disappears unexpectedly, C: approaches the 50-GiB floor, ledger/evidence
verification disagrees, or visual QC requests human ACCEPT. Full completion
requires `FULL_BATCH_COMPLETE` plus 275 verified passing stage completions.

## Known later-stage dependency gate

The pilot suite has 34 passing tests, but `test_run_glmsingle` currently cannot
import all vendor dependencies: Windows `neuro_161` lacks `fracridge`, and WSL
`pycortex` lacks `tqdm`. Do not let the batch enter GLMsingle until the intended
pilot-equivalent runtime is identified or recreated and then pinned in a new
control version. This issue does not affect active source audit, BIDS, SDC, or
fMRIPrep processing.
