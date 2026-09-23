# fMRIPrep/SDC production design for the remaining 25 subjects

Status: implemented and dry-tested; production has not been launched.

## Frozen runtime and storage policy

- fMRIPrep: `nipreps/fmriprep:25.1.3`, executed by the immutable digest
  `sha256:4e5cfd99f6d80a9ef10a87929f8e74e4caf9dc108b49551eb23c775e61cd16f7`.
- Output: `MNI152NLin6Asym:res-2`.
- FreeSurfer surface reconstruction: disabled (`--fs-no-reconall`).
- Threads/memory: 6 total threads, 2 OpenMP threads, 26,000 MB.
- All BIDS inputs, branch inputs, outputs, reports, QC images, logs, and manifests
  are on `N:`. Docker Desktop and its named work volume remain on `C:` because
  Docker requires them there. The work volume is temporary and cannot be
  retired until the final checksum and human visual-QC gates pass.
- Container launch is **only** through
  `C:\Windows\System32\wsl.exe -d Ubuntu-22.04 -- /usr/bin/docker`.
  Every persistent bind source is translated from `N:\...` on the Windows
  control side to `/mnt/n/...` before it reaches Docker. A direct Windows
  `N:` or UNC bind is forbidden: the Docker daemon cannot resolve that mapped
  network drive. The read-only FreeSurfer license may be bound from `/mnt/c`.
- Bind mounts use Docker's `--mount type=bind` form, not `-v`, so a missing
  source fails instead of being silently created. Image execution uses
  `--pull never` after the exact digest gate.
- Every launch rechecks at least 60 GiB free on `C:`. The preferred headroom is
  80 GiB.

## Subject/session SDC plan

The source inventory fixes these four patterns:

| Pattern | Subjects | Production strategy |
|---|---|---|
| One-session PEPOLAR | 1, 2 | One direct subject run |
| Two-session PEPOLAR | 7, 9, 11-16 | One all-session subject run; session-specific `B0FieldIdentifier` values |
| Two-session GRE | 19, 22, 23, 25, 27 | One all-session subject run; session-specific phasediff associations |
| Mixed GRE/SyN | 17, 18, 20, 21, 24, 26, 28-31 | Two isolated functional-session branches followed by a strict merge |

For a mixed subject, each branch contains all of that subject's T1w inputs so
both branches construct the same anatomical reference. It contains BOLD data
from only one session and fieldmaps from only that same session. The SyN branch
contains no fieldmap and uses both an exact-image ANAT-estimator preflight and
`--use-syn-sdc error`. This prevents a GRE fieldmap from the other session from
suppressing SyN or being borrowed across sessions.

This extends the proven Sub5 solution. Sub5 could reuse one identical
sessionless T1w in both branches. The remaining mixed subjects have
session-labelled T1w inputs, so retaining the complete T1w set in both branches
is required. The merge refuses to proceed unless the subject-level anatomical
derivative trees are byte-identical.

## Fail-closed gates

1. The canonical subject BIDS tree must already have passed the frozen source
   audit and official BIDS validator.
2. The production stage order must insert `docker_mount_gate` immediately
   before `sdc_preflight`. It creates a subject-specific, immutable receipt
   after a pinned-image `/mnt/n` read/write round trip. A failed attempt keeps
   its UUID-named probe directory for diagnosis; an audited resume uses a new
   UUID and never overwrites evidence.
3. `preflight` audits every BOLD, T1w, fieldmap role, phase-encoding pair,
   `B0FieldSource`, `B0FieldIdentifier`, and `IntendedFor` association. It rejects
   any cross-session functional or fieldmap file in an isolated branch.
4. Each canonical or isolated branch is independently run through BIDS
   Validator 3.0.1 and `sdcflows.find_estimators()` inside the exact pinned
   fMRIPrep image. Both the validator and Docker CLI are entered through the
   pinned Ubuntu-22.04 distribution. The estimator probe has networking
   disabled. The receipt includes a SHA-256 manifest of every branch input.
5. `run-subject` recomputes that complete input identity, verifies the Docker
   digest and `C:` storage gate, then creates a new named work volume. Stale
   output directories or volumes are rejected; there is no implicit resume.
6. A completed branch must have all target-space BOLDs, masks, confounds, and
   fMRIPrep summary reports. The summary's SDC line must exclusively match
   PEPOLAR, GRE fieldmap-based, or fieldmap-less SyN as planned.
7. Mixed branches are merged by session only. Complete original branch reports
   are preserved under `logs/branch_reports/`. The final top-level HTML is an
   index to both reports, not a claim that one fMRIPrep report covered both.
8. Derivative validation requires exactly 10 runs, a common MNI grid, matching
   BOLD/mask grids, one confound row per BOLD volume, target-space metadata,
   expected SDC marker, and the SDC/coregistration/carpet reportlets.
9. Quantitative QC computes run and aggregate tSNR with the already-tested pilot
   implementation and records framewise-displacement summaries. tSNR is a
   descriptive QC measure, not a universal pass threshold.
10. Visual-QC contact sheets are generated for every acquisition session. Their
   generation leaves every run `PENDING`; it never implies acceptance. A human
   reviewer must mark SDC, coregistration, carpet/motion, and overall decision as
   `ACCEPT`, with name and UTC time. This workstation has no Windows Inkscape;
   the active production stage must use the separately pinned WSL renderer. The
   Python `qc-bundle` command remains a tested parameterized fallback, not the
   active rendering contract.
11. The final fMRIPrep derivative and QC bundle receive a complete SHA-256
    manifest on `N:`. Docker work-volume retirement is allowed only after that
    manifest re-verifies and the visual attestation fully passes.

## Exact production commands

The active stage runner should consume
`config/stage_contracts_fmriprep_proposal.json` only after its final code/config
hashes are copied into the coordinated production freeze. The active contract
must remain blocked until it replaces the pilot Windows-Docker wrapper and
direct bind command. The proposal's first two commands are, in logical form:

```powershell
& $Python $Workflow --cohort-config $Cohort --sdc-config $SdcConfig preflight `
  --subject $Subject --canonical-bids-root $SubjectBids `
  --branch-root $BranchBids --output $PreflightReceipt `
  --link-mode auto --run-external-gates

& $Python $Workflow --cohort-config $Cohort --sdc-config $SdcConfig run-subject `
  --subject $Subject --preflight-root $PreflightReceipt `
  --branch-output-root $BranchOutputs --final-root $FinalDerivative `
  --license-file $FreeSurferLicense --link-mode auto --execute
```

The Windows Python workflow invoked above constructs every container command
with this exact prefix:

```text
C:\Windows\System32\wsl.exe -d Ubuntu-22.04 -- /usr/bin/docker
```

For example, the BIDS and derivative sources reach Docker as
`type=bind,src=/mnt/n/...,...`; neither `N:\...` nor a UNC path may appear in a
container mount argument. The `--execute` token is intentionally absent from
all planning and tests. A
failed fMRIPrep branch preserves its output and named work volume for diagnosis;
it must not be silently resumed or merged.

After Docker Desktop is restarted, prove that its Linux container can read and
write the mapped network drive through Ubuntu-22.04 before building any subject:

```powershell
& "$Root\code\smoke_docker_n_drive_mount.ps1" `
  -ProbeDirectory "$Root\audit\docker_n_mount_smoke_20260729" `
  -OutputReceipt "$Root\audit\docker_n_mount_smoke_20260729.json"
```

This wrapper enters `/usr/bin/docker` through Ubuntu-22.04, converts the probe
directory to `/mnt/n/...`, uses the pinned image with `--pull never`, disables
container networking, writes only two tiny test files on `N:`, and preserves a
hash receipt including the exact argv and Docker CLI SHA-256. A failure blocks
production; it is not evidence that fMRIPrep itself failed. The successful
2026-07-29 evidence is recorded in
`audit/DOCKER_N_DRIVE_WSL_MOUNT_SMOKE_20260729.json`; it does not authorize a
later run after Docker restart without a fresh smoke receipt.

## Human-review blocker

Automated checks can establish file identity, estimator discovery, planned SDC
method, numerical grid integrity, motion, and tSNR. They cannot decide whether
susceptibility correction improved anatomy alignment without introducing local
warping. Decoding remains blocked for a subject until its generated contact
sheets have been reviewed and the attestation contains 10 explicit acceptances.
