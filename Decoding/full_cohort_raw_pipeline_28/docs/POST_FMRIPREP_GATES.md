# Post-fMRIPrep production gates

## Scope

`code/production_postproc_gates.py` implements the three production stages
between quantitative fMRIPrep QC and final subject completion:

1. `visual_qc`: render every acquisition session and require an explicit,
   hash-bound human decision for all ten runs;
2. `archive`: create and re-read a complete SHA-256 manifest on `N:` before
   retiring any Docker work volume;
3. `subject_gate`: revalidate visual review, archive bytes, GLMsingle Type-D,
   ERP tables, and the immutable stage ledger as one subject identity.

The commands are exact templates in `config/stage_contracts.json`. They were
made executable without changing the still-blocked `sdc_preflight` or
`fmriprep` contracts. No production subject was launched while implementing or
testing these gates.

## Visual-QC lifecycle

The renderer uses `/usr/bin/inkscape` inside `Ubuntu-22.04` because no Windows
Inkscape installation is present. Its required SHA-256 is
`ad5b4926bc77472d0c730f968b1aaab2028d784f57d419f6efdffd5d0f9859b9`;
the observed version was `Inkscape 1.1.2 (0a00cf5339, 2022-02-04)`. Both the
WSL executable and Linux renderer identity are checked before rendering. All
temporary SVGs, PNGs, manifests, and review files are written on `N:`.

For each session, the stage requires the exact run set from `config/cohort.json`
and renders:

- SDC background and foreground layers;
- BOLD-to-T1w coregistration background and foreground layers;
- ROI reportlets;
- carpet/motion reportlets;
- one deterministic run-ordered contact sheet for each of those six views.

Each source SVG is SHA-256 checked before and after rendering. The published
`visual_artifact_manifest.json` contains a sorted, exact file set with size and
SHA-256 for every session output. The attestation template has exactly ten rows
and binds each row to that manifest hash.

The first `visual_qc` attempt is expected to exit non-zero after generating the
bundle because every decision starts as `PENDING`. This is a deliberate review
pause, not an automatic exclusion. A reviewer must inspect the contact sheets
without seeing decoding accuracy, then edit
`visual_review_attestation.tsv` so that every run has:

- `ACCEPT` in `sdc_review`, `coreg_review`, `carpet_motion_review`, and
  `overall_decision`;
- a nonempty reviewer name;
- a timezone-aware ISO-8601 `reviewed_at_utc`;
- the unchanged `visual_manifest_sha256`.

Any pending, rejected, missing, duplicated, wrong-session, wrong-SDC-mode, or
wrong-hash row fails closed. After review, audit the failed attempt and resume
the same signature with the production runner's `--resume-failed` option. A
successful resume writes `visual_gate_receipt.json`; rendering by itself can
never create a passing receipt.

## N:-resident checksum archive and Docker retirement

The `archive` stage independently runs the generalized derivative validator,
then hashes every file in these exact roots:

- `runtime/fmriprep/SubXX`;
- `audit/quantitative_qc/SubXX`;
- `audit/visual_qc/SubXX`.

`audit/archives/SubXX/required_products_sha256.json` records root label,
absolute root, relative path, byte size, and full SHA-256 for every file. The
manifest is immediately reread byte-for-byte. A manifest on any drive other
than `N:` is rejected.

Only after the archive, visual manifest, and accepted attestation all reverify
does the command read the immutable fMRIPrep launch receipt. The work-volume
set must equal the subject/session plan exactly. Each exact Docker volume is
inspected, removed without a shell or wildcard, inspected again, and assigned
an immutable per-volume retirement receipt. If a volume is already absent but
has no matching receipt, the stage fails rather than guessing. Mixed-session
subjects can therefore resume safely after a partial multi-volume retirement.

Docker work-volume retirement is irreversible, but it removes only fMRIPrep
scratch state. The fMRIPrep derivative, QC artifacts, full checksum manifest,
and retirement evidence remain on `N:`. The checksum tree is verified again
after every retirement and again by the final subject gate.

## Final cross-artifact subject gate

The last lifecycle stage requires all of the following in one invocation:

- accepted ten-run visual attestation bound to unchanged rendered artifacts;
- unchanged archive roots, sizes, SHA-256 values, launch receipt, and exact
  Docker retirement receipts;
- GLMsingle provenance matching the current freeze and cohort config, pinned
  GLMsingle/fracridge identities, 10 session indicators, and the frozen ridge
  grid/PC count;
- an independent blockwise scan of Type-D `betasmd` with shape
  `(n_voxels, 1, 1, 600)`, with all 600 betas finite at every analysis voxel;
- finite HRF index, ridge fraction, and R2 datasets, an exact mask/index voxel
  count, and an exact 600-row/60-trials-per-run chronology;
- ERP validation and provenance for exactly three pipelines, two ERP methods,
  eight contrasts, and the 17 frozen Kastner/Wang ROIs, including complete
  subject/fold keys, finite `[0,1]` accuracy, and at least ten voxels per ROI;
- a valid hash-chained ledger and event mirrors in which every prior stage for
  the subject passed under the same production control identity and the only
  incomplete `subject_gate` attempt is the currently executing one.

Success writes `audit/subject_gates/SubXX.json` atomically. Existing evidence
is never overwritten.

## Verification evidence

Focused tests are in `tests/test_production_postproc_gates.py`. They cover the
pending-attestation stop, manifest-hash binding, archive-before-removal order,
missing-volume failure, independent finite-Type-D scan, exact ERP dimensions,
and ledger control identity. On 2026-07-29:

- all 34 analysis/workspace tests passed;
- all 35 production-control/SDC tests passed;
- the actual WSL Inkscape binary/hash/version gate passed;
- an actual N:/UNC-to-WSL render smoke test produced a valid 1108 × 586 PNG;
- all three active command templates rendered successfully for Sub01 without
  launching them.

These tests do not authorize production. Source approval, stable C: headroom,
the production freeze, and the generalized SDC/fMRIPrep launch contracts remain
separate prerequisites.
