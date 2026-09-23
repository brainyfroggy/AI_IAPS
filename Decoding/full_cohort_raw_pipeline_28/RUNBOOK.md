# AI-IAPS full 28-subject SDC + GLMsingle ERP decoding runbook

This directory is the versioned production workspace for extending the
completed Sub4/5/6 pilot to the predetermined 28-subject cohort. The remaining
batch is Subjects 1, 2, 7, 9, and 11–31. Subjects 3, 8, and 10 are excluded for
the reasons frozen in `ANALYSIS_FREEZE.json`.

The current state is **prelaunch**. Raw, onset, and legacy inputs passed the
read-only audit for every remaining subject. The 60-GiB local-storage minimum
passed on 2026-07-29 after cleanup, but that does not authorize launch:
production must still wait for the exact approved source manifest, corrected
BIDS construction and validation, subject/session-aware SDC preflight, and the
remaining stage-contract gates.

## Why this is a new workspace

The pilot code and reports intentionally fail closed on Subjects 4/5/6. They
must remain immutable evidence of the pipeline screen. Full-cohort scripts and
outputs live here and must never overwrite pilot results.

## Fixed cohort and analysis

The machine-readable definition is `ANALYSIS_FREEZE.json`. In summary:

- Legacy comparison: original SPM LS-A betas from 8-mm-smoothed data.
- New branches: SDC-aware fMRIPrep 25.1.3, GLMsingle Type-D, and 0/3-mm added
  beta smoothing.
- ROI set: the 17 plotted Kastner/Wang regions only.
- Decoding: exact historical Avg(random) for figure comparability and run-wise
  LORO condition means as the primary ERP-style run-generalization analysis.
- Group summaries use subject-level values and paired within-subject deltas.

The concise canonical cohort file has SHA-256
`4e70a48896df734e42f1f10429469c9f0e570c4df2bf7fb1d1673a5ae1f88bbe`.
The schema-v2 acquisition/source configuration is `config/cohort.json`,
SHA-256 `7d4d2901944e0fce500eff43fbd04d8c455bcf909c70e9486c28962dd04b793e`.
Both identities are frozen; a changed file requires a reviewed amendment.

## Input audit result

All 25 remaining subjects passed these gates:

1. Ten selected full-size original task NIfTIs and JSON sidecars.
2. Anatomy for every true acquisition session.
3. Ten raw MATLAB logs, each with 60 stimuli in exact manifest order and finite,
   strictly increasing onsets.
4. Complete legacy SPM outputs and 600-trial extracted beta arrays.
5. No existing GLMsingle Type-D output, so fresh processing is required.

Existing derivatives under `MyIAPS_BIDS` are not production inputs. Their HTML
reports state `Susceptibility distortion correction: None`, several subjects
are absent or incomplete, and their raw BIDS tree lacks the needed events and
fieldmaps.

## Acquisition and SDC rules

- The selected native BOLD NIfTI spacing is **1.7966 × 1.7966 × 2.25 mm**.
  “1.8 mm” is a nominal in-plane label, not an isotropic voxel size.
- Raw volume-count review found two prospective QC flags. Retain the reviewed
  non-MoCo sources, but carry the flags through BIDS, fMRIPrep, and quantitative
  QC: Sub29 session 1 run 6 has 216 volumes (two below the 218-volume mode), and
  Sub30 session 1 run 6 has 210 (eight below mode). Their final event ends at
  369.142 s and 372.775 s, within acquisitions of 388.8 s and 378.0 s,
  respectively. These facts do not by themselves exclude either subject.

- One-session duplicate subsets: use the complete direct tree for Sub1 and
  Sub2; do not treat `_2` as a second visit.
- Explicit scanner-label offsets must be encoded, never inferred:
  - Sub11 session 2 logical runs 7–10 use scanner labels 8–11.
  - Sub12 session 1 logical run 1 uses label 1 and logical 2–6 use 3–7.
  - Sub16 session 2 logical 7/8 use 7/8 and logical 9/10 use 10/11.
  - Sub17 session 1 logical run 1 uses label 1 and logical 2–6 use 3–7.
- Duplicate resolution must be explicit for Sub1, Sub13 session 2, and Sub15
  session 1. Scanner online `MoCoSeries` outputs are rejected.
- PEPOLAR is available for Sub1, 2, 7, 9, and 11–16 in every relevant session.
- GRE is available for both sessions of Sub19, 22, 23, 25, and 27.
- Mixed measured/SyN subjects:
  - Sub17, 18, 20, and 21: GRE session 1; isolated SyN session 2.
  - Sub24, 26, 28, 29, 30, and 31: isolated SyN session 1; GRE session 2.
- A fieldmap is never borrowed across sessions. Every measured and isolated SyN
  branch must be estimator-preflighted and its derivative source verified before
  merge.

## Per-subject production lifecycle

Process strictly one subject at a time:

1. Freeze exact source rows and full SHA-256 values for BOLD, JSON, T1,
   fieldmap files, and onset logs.
2. Build a subject-specific corrected BIDS tree from the frozen rows.
3. Require 10 runs, 600 events, zero BIDS errors, and explicit SDC coverage for
   every run.
4. Run fMRIPrep with the pinned image and unique output/container/work names.
5. Require 10 BOLD/mask/JSON/confounds sets, equal BOLD/confound lengths,
   completed reports, correct SDC markers, visual alignment review, and tSNR /
   motion QC.
6. Archive to `N:` with a full SHA-256 manifest and verify every archived file.
7. Run GLMsingle Type-D with the pinned commit, patches, nuisance model, 10 PCs,
   and 20-value ridge grid. A failed partial fit is preserved and restarted in a
   new versioned destination; production overwrite is forbidden. The preflight
   must also pass the cohort-local `fracridge==2.0` RECORD/file/origin gate
   documented in `audit/FRACRIDGE_RUNTIME_REPAIR_20260729.md`.
8. Require 600 finite Type-D betas, exact manifest alignment, 60 trials/run, and
   approved ridge/PC values.
9. Run the 17-ROI historical and run-wise ERP analyses independently for that
   subject. Validate row keys, bounds, fold completeness, and the prospectively
   fixed minimum of 10 retained analysis voxels in every ROI.
10. Hash outputs and mark the subject complete in `status/subject_status.tsv`.
    Only then retire that subject's temporary local stages.

Every command must capture start/end timestamps, full command line, exit code,
software/config hashes, and stdout/stderr under `logs/`. Resume is allowed only
when the stored configuration hash exactly matches the requested command.

## Subject gates

- Raw/BIDS: 10 runs, 60 events/run, 600 unique trial rows, exact stimulus order.
- fMRIPrep: 10 preprocessed BOLDs, masks, JSON sidecars, and confounds tables;
  confound rows equal BOLD volumes; expected measured or SyN report markers.
- GLMsingle: HDF5 Type-D shape `(n_voxels, 1, 1, 600)`, float32, all finite;
  600 unique beta indices and nonempty analysis mask.
- ERP: all 17 ROIs retain at least 10 analysis voxels; all eight contrasts and
  three pipelines are present; accuracies are finite and in `[0,1]`.
- QC decisions are made without inspecting decoding accuracy. tSNR is reported
  descriptively unless a prospective exclusion threshold is added by a
  versioned freeze amendment.

## Completeness gates

For the remaining 25 subjects:

| Output | Required rows |
|---|---:|
| ERP subject results, each method | 10,200 |
| Historical Avg(random) fold results | 510,000 |
| Run-wise LORO fold results | 102,000 |
| ROI voxel-count rows | 1,275 |

For all 28 subjects:

| Output | Required rows |
|---|---:|
| ERP subject results, each method | 11,424 |
| Historical Avg(random) fold results | 571,200 |
| Run-wise LORO fold results | 114,240 |
| ROI voxel-count rows | 1,428 |
| Fixed attached historical reference | 3,808 |

Counts alone are insufficient. Aggregation must require exact subject-set and
unique-key equality across branches, `n_subjects=28` in every group cell, and
no input path containing failed, smoke, or coarse-grid outputs.

## Frozen external identities

`ANALYSIS_FREEZE.json` pins path, size, and full SHA-256 for the Kastner atlas
and labels, the 600-trial stimulus manifest, legacy beta grouping, historical
decoding reference, and the immutable pilot ERP/GLMsingle scripts. It also pins
the verified `fracridge==2.0` RECORD hash
`a9ec96dc55b4d215d4dd02caacf2d710e235c650d58c5165af22ed83551b359c`
and combined payload hash
`1334b2224ed818142e05a6f2ec8337b490fea208cdb99be67079c5a51424b26b`.
Production must fail closed if any identity changes.

## Resource plan and current launch status

Observed fMRIPrep time is about 5.5–9 hours per standard subject; total strict
sequential time is expected to be roughly 8–12 continuous days. Permanent `N:`
storage is estimated at 280–285 GiB.

The 2026-07-29 cleanup first moved the reconstructable pilot fMRIPrep cache
(27,857,496,192 bytes; 25.944 GiB) and then nine additional verified
cache/working directories (20,096,583,723 bytes; 18.716 GiB) from `C:` to
recoverable audit archives on `N:`: 47,954,079,915 bytes (44.661 GiB) total.
It also permanently emptied 38,332 current-user Recycle Bin entries (about
35.66 GB of regular files) and removed the completed pilot Docker volumes.
Only protected zero-size `$RHRQS75` and `desktop.ini` remain in that Recycle Bin.
Docker then compacted its data VHDX from 45.481 to 11.539 GiB automatically.
Docker Engine 28.3.0 is healthy and the fMRIPrep image resolves to the frozen
digest; the problem was local capacity, not a failure of the Docker pipeline
that worked for Subjects 4/5/6.

At `2026-07-29T22:49:34Z`, the storage gate passed with 61.631 GiB free on `C:`
and 18,459.199 GiB free on `N:`. This meets the hard 60-GiB floor but not the
preferred 80 GiB. Dropbox was stopped because it was writing about 85 MB/s and
refilling `C:`; keep it paused for the batch. All BIDS trees, derivatives,
archives, logs, QC, GLMsingle products, and decoding results belong on `N:`.
Only installed programs and Docker/WSL storage that must remain local may use
`C:`. A fresh `code/check_storage_gate.ps1` result still controls every launch.

A second check at `2026-07-29T22:58:06Z` still passed but found only 60.942 GiB
free on `C:`—0.942 GiB above the hard floor and 0.689 GiB lower than roughly
8.5 minutes earlier. The Docker VHD remained 11.539 GiB and Dropbox remained
stopped; the likely source was allocation by the Ubuntu-22.04 `ext4.vhdx`
during WSL tests (63,348,670,464 bytes; 58.997 GiB). Free space then held near
60.949 GiB over a five-second sample, but the margin is still not sufficient to
start fMRIPrep. Recover additional headroom before launch even though the
Boolean minimum check currently says `passed`.

Current fail-closed blockers are the header-only, unapproved production source
manifest; unstable local-storage headroom; unbuilt/unvalidated corrected BIDS
trees; and unapproved mixed-session measured-fieldmap/SyN launch logic. The
all-session visual-QC, checksum/archive, and final cross-artifact subject gates
are implemented and tested as documented in
`docs/POST_FMRIPREP_GATES.md`. Do not change
`freeze_status` to `frozen_for_production` until all are resolved.
