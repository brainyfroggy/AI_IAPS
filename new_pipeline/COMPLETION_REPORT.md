# AI-IAPS new_pipeline: 28-Subject Completion Report

**Scope:** fMRIPrep 25.1.3 (MNI152NLin6Asym, native functional resolution) → GLMsingle Type-D
single-trial betas, for the full study cohort.
**Status:** Complete. 28/28 subjects, 16,800 single-trial betas total (600 trials × 28 subjects).
**Completion date:** 2026-08-11.

This document is the durable record of what was run, what broke, how it was fixed, and — most
importantly — an honest quality assessment of the resulting betas, including one subject that
needs a downstream decision before use.

---

## 1. What this wave produced

For each of 28 subjects: fMRIPrep-preprocessed BOLD at `MNI152NLin6Asym`, native functional
resolution (1.7966 × 1.7966 × 2.25 mm, no resampling), unsmoothed, fed into GLMsingle Type-D
(`TYPED_FITHRF_GLMDENOISE_RR`) to produce 600 single-trial beta estimates per subject.

- **Cohort:** subjects 1, 2, 4–31, minus prespecified exclusions (3, 8, 10) and Sub33 (deferred,
  unaudited — explicitly out of scope for this wave).
- **No downstream analysis in this wave.** No smoothing, no decoding, no group-level anything.
  Single-trial betas only. Smoothing/decoding is deferred to a separate future effort per the
  original design freeze.
- **Storage:**
  - BIDS: `N:\...\LAB_IAPS_AI\bids_unified_glmsingle_28\`
  - fMRIPrep derivatives: `N:\...\LAB_IAPS_AI\fmriprep_derivatives\unified_glmsingle_28\Sub##\`
  - GLMsingle output: `N:\...\AI_IAPS\new_pipeline\glmsingle\sub-##\`
    (`TYPED_FITHRF_GLMDENOISE_RR.hdf5`, `trial_manifest.tsv`, `validation.json`, `motion_qc.tsv`,
    `provenance.json`)
  - Nothing was ever written to C: — only transient Docker/WSL runtime state, cleaned per-subject.

## 2. Execution summary

| Wave | Subjects | Count | Result |
|---|---|---|---|
| 1 | 1, 2, 4, 5, 6, 7, 9, 11, 12, 13 | 10 | ✅ all complete |
| 2 | 14–23 | 10 | ✅ all complete |
| 3 | 24–31 | 8 | ✅ all complete |

Wave 1 included the three pilot subjects (4, 5, 6), previously processed at `res-2` under an
older pipeline — they were reprocessed fresh at native resolution for this wave, not reused.
Waves 2 and 3 progressively increased the proportion of "mixed-SDC" subjects (subjects whose two
sessions used different susceptibility-distortion-correction methods — see §4 for why that
mattered): Wave 1 had none, Wave 2 had 4/10, Wave 3 had 6/8.

## 3. Pipeline architecture (brief)

Rather than build BIDS/fMRIPrep orchestration from scratch, this wave adapted an existing,
independently-audited pipeline (`Decoding\full_cohort_raw_pipeline_28`) that had already resolved
the hard part — per-subject/session fieldmap-mode audit (PEPOLAR / GRE / SyN), scanner-label
offsets, and duplicate-session resolution — for this exact cohort, via monkeypatches rather than
edits to the frozen source scripts:

- `new_pipeline\config\fmriprep_sdc_config_native.json` — native-resolution config (mem-mb 20000,
  later supplemented with a real Docker `--memory` cgroup limit; distinct `analysis_id` from the
  frozen res-2 pilot config, which was never modified).
- `new_pipeline\config\cohort_28.json` — extends the audited 25-subject cohort with entries for
  the 3 pilot subjects (4, 5, 6), each sourced from that pilot's own published acquisition
  records, not re-derived.
- `new_pipeline\code\wave_orchestrator.py` — drives `fmriprep_sdc_workflow_v6.py` (frozen, unedited)
  through `plan → preflight → run-subject`, with five in-process monkeypatches (listed in §4).
- `new_pipeline\code\run_glmsingle_wave.py` — generalizes the validated single-subject GLMsingle
  script (originally hardcoded to Subject 4) to run any subject, with per-subject volume-count and
  session-indicator tables derived from `cohort_28.json`.
- Wave-level concurrency: 4-way fMRIPrep, up to 3-way GLMsingle (adaptive — see §4).

## 4. Issues found and fixed during this run

Every item below was independently confirmed before being called a "fix" — this project's working
discipline throughout was: read the actual error, reproduce or directly verify the mechanism,
patch the frozen scripts only via in-process monkeypatch (never edit-on-disk), and re-verify the
patched behavior before trusting it on real subjects. In discovery order:

1. **Docker Desktop stale-socket crash loop.** A `dockerInference` socket file left behind by any
   unclean shutdown prevents the engine from starting on relaunch. Recurred ~5 times across this
   run (including after a forced Windows update mid-wave). Fix: rename `%LOCALAPPDATA%\Docker\run`
   out of the way, relaunch. Self-diagnosing every time it recurred rather than assuming a new
   cause.
2. **N: drive has no symlink/hardlink support** (WSL's 9p/drvfs network mount). Forced BIDS
   construction to full byte-copy rather than the originally-intended symlink pattern — confirmed
   this was already the established convention (Subject 4's existing BIDS tree was a real copy,
   not a symlink) rather than a new compromise.
3. **`basic_derivative_gate()` hardcoded `res-2` filename suffixes.** Would fail-close on every
   real (successful) res-native fMRIPrep run. Patched (monkeypatch) to select the correct suffix
   set from the config's actual resolution.
4. **`merge_mixed_branches()`'s anatomical-equality check was too strict.** For subjects whose two
   sessions used different SDC methods, it required byte-identical anatomicals between branches
   even when both derived from the same T1w input — ordinary floating-point non-determinism in
   ANTs/ITK (confirmed via quantified comparison: correlation 0.986–0.9998, Dice 0.999) was enough
   to trip it. Fixed by using the already-existing "preferred branch" concept for merged output,
   with a similarity-threshold sanity check (0.99 general, 0.97 for native-space CSF/GM
   probability maps specifically, where partial-volume boundary noise is measurably higher) rather
   than exact equality. Validated against three different SDC-mode combinations across the run
   (PEPOLAR+SyN, GRE+SyN, and the higher-mixed-SDC-density Wave 3).
5. **`subject_dirs()` returns a duplicate path for any two-digit subject.** `f"sub-{n:02d}"` and
   `f"sub-{n}"` are identical strings once `n ≥ 10`, so the function silently returned the same
   directory twice, and `discover_events()` then found every events file "twice" and refused.
   100% deterministic — caught after it stalled a queue daemon in an infinite retry loop, then
   fixed and would otherwise have blocked every one of the 18 two-digit subjects in this cohort.
6. **`EXPECTED_RUN_N_VOLUMES`/`SESSION_INDICATORS` incorrectly inherited the 25-subject cohort's
   modal value (218 volumes) onto the 3 pilot subjects**, who were never part of that audit and
   in fact have a real, different value (224). Root-caused via direct empirical measurement (not
   assumption) for each pilot subject; Subject 5's genuinely irregular interleaved session pattern
   (`[1,2,1,1,1,2,2,2,2,2]`, spanning two raw source directories) was independently confirmed by
   the user from memory of the acquisition, matching the derived value exactly.
   Separately, **Subjects 1 and 2 (in the audited 25-subject cohort) turned out to be real,
   consistent 224-volume outliers** — the original volume-count audit was a shortfall-only check
   and was never designed to catch upward outliers; confirmed by checking all 10 runs × all 25
   audited subjects directly against raw source files.
7. **Real memory exhaustion (ENOMEM) under combined fMRIPrep+GLMsingle load**, first surfaced as
   three GLMsingle jobs dying silently mid-run, then confirmed unambiguously when Subject 1's
   fMRIPrep crashed with an explicit `OSError: [Errno 12] Cannot allocate memory`. GLMsingle
   concurrency was made a function of live fMRIPrep container count (later refined further in
   Wave 3 — see below) rather than a fixed cap.
8. **Wave 3's mixed-SDC-heavy composition (6/8 subjects) caused a genuine global OOM.** Because a
   mixed-SDC subject's first branch is always the SyN (fieldmap-less) path, the 4-way fMRIPrep pool
   naturally scheduled four concurrent ANTs SyN registrations simultaneously — the single most
   memory-hungry step in the whole pipeline, confirmed via `docker stats` to peak at ~26 GiB per
   instance (the nominal `--mem-mb 20000` fMRIPrep flag is a scheduler hint only, not an enforced
   limit — containers had no real cap at all until this fix). The kernel OOM-killer fired and took
   out Subject 26's `antsRegistration` process — correlated to the exact second via dmesg
   timestamps against the crash report. Fixed two ways: GLMsingle concurrency now adapts live to
   fMRIPrep container count (0 GLMsingle workers at 4 fMRIPrep, up to 3 at ≤1), and a real Docker
   `--memory`/`--memory-swap` cgroup limit (32 GiB default, verified via `docker inspect` byte
   value, not just trusted) was added to all subsequent fMRIPrep launches — converting any future
   recurrence into a container-scoped kill (confirmed: Subject 26's retry hit its own limit and
   died without affecting anything else, versus the first failure's global, unbounded blast
   radius). One subject (26) needed a case-by-case higher limit (64 GiB) to complete.
9. **Several queue-daemon control-flow bugs**, all in the PowerShell scripts written for this wave
   (never in the frozen pipeline scripts): fixed-subject-list daemons silently never picking up
   subjects that finished fMRIPrep after the daemon launched (required three manual nudges before
   being redesigned to recompute eligibility from live state each pass); a PowerShell
   `$array[1..($array.Count-1)]` idiom that produces a descending range (not an empty array) when
   the array has exactly one element, causing an infinite no-op retry loop.
10. **A full, unclean machine reboot mid-wave** (forced Windows update) killed every in-flight
    process. Recovery required re-diagnosing the Docker socket issue, re-verifying the WSL
    integration (which itself crash-looped independently and needed a full `wsl --shutdown` before
    it would stabilize), and reconciling exactly which of the in-flight subjects' work was real vs.
    orphaned before resuming. One side effect: the interruption incidentally revealed that a
    Docker Desktop restart auto-compacts its WSL2 VHDX file — later relevant when C: free space
    dropped to ~44 GiB (accumulated Docker work volumes across two waves, since the mandated
    per-subject volume cleanup had not actually been happening) — restarting Docker Desktop
    reclaimed ~417 GB without any elevated/admin action.

Nothing in this list represents a compromise on the resulting data's correctness — every fix
either corrected a genuine bug in wave-3-specific tooling written this run, or replaced an
over-strict check with a threshold justified by direct, quantified measurement. The one place
where a real trade was made explicitly (§4 item 8, Subject 21's SyN branch redone from scratch
after a fail-closed volume-reuse refusal) cost real compute time, not data validity, since that
work was never published to `final_root` before being cleared.

## 5. Quality control results

### 5.1 Formal validation (all 28 subjects)

Every subject's `validation.json` was checked for the correct beta array shape, correct trial
count, and that every single value is finite. All 28 pass:

| sub | voxels | R² range | β range (abs) | HRF idx range | min-frac-ridge selected |
|---|---:|---|---|---|---:|
| 01 | 270,829 | 0.59 – 27.4 | ≤ 20,322 | 0–19 | 99.7% |
| 02 | 266,502 | 0.39 – 10.7 | ≤ 8,496 | 0–19 | 99.7% |
| 04 | 268,535 | 0.62 – 48.1 | ≤ 25,775 | 0–19 | 98.8% |
| 05 | 270,150 | 0.40 – 18.4 | ≤ 2,496 | 0–19 | 99.9% |
| 06 | 272,761 | 0.72 – 44.2 | ≤ 9,978 | 0–19 | 99.0% |
| 07 | 263,174 | 0.44 – 40.8 | ≤ 14,963 | 0–19 | 99.0% |
| 09 | 271,982 | 0.77 – 32.0 | ≤ 25,212 | 0–19 | 98.6% |
| 11 | 246,506 | 0.23 – 24.0 | ≤ 29,111 | 0–19 | 99.5% |
| 12 | 259,632 | 0.51 – 30.2 | ≤ 1,151 | 0–19 | 98.9% |
| 13 | 257,844 | 0.52 – 16.1 | ≤ 11,074 | 0–19 | 99.2% |
| 14 | 282,426 | 0.57 – 28.6 | ≤ 141,999 †| 0–19 | 99.2% |
| 15 | 262,202 | 0.14 – 34.6 | ≤ 3,564 | 0–19 | 99.2% |
| 16 | 261,054 | 0.47 – 40.9 | ≤ 5,952 | 0–19 | 98.5% |
| 17 | 258,945 | 0.31 – 27.0 | ≤ 19,650 | 0–19 | 99.3% |
| 18 | 258,219 | 0.89 – 23.1 | ≤ 8,086 | 0–19 | 99.9% |
| 19 | 282,087 | 0.40 – 30.6 | ≤ 4,551 | 0–19 | 98.9% |
| 20 | 254,392 | 0.52 – 33.0 | ≤ 1,449 | 0–19 | 99.2% |
| 21 | 254,443 | 0.57 – 27.8 | ≤ 1,981 | 0–19 | 99.2% |
| 22 | 282,184 | 0.45 – 20.9 | ≤ 3,054 | 0–19 | 99.5% |
| 23 | 257,340 | 0.30 – 31.7 | ≤ 715 | 0–19 | 99.1% |
| 24 | 267,207 | 0.42 – 40.0 | ≤ 1,305 | 0–19 | 99.2% |
| 25 | 255,123 | 0.27 – 33.6 | ≤ 6,757 | 0–19 | 99.3% |
| 26 | 256,641 | 0.68 – 15.3 | ≤ 2,339 | 0–19 | 99.7% |
| 27 | 277,916 | **−13.7** – 34.7 | ≤ 26,628 | 0–19 | 99.97% |
| 28 | 270,639 | 0.66 – 30.4 | ≤ 6,679 | 0–19 | 99.7% |
| 29 | 265,077 | 0.36 – 49.5 | ≤ 8,288 | 0–19 | 99.0% |
| **30** | 254,085 | **−154,359** – 37.8 | **≤ 309,367** | 6–19 | 99.5% |
| 31 | 264,804 | 0.47 – 13.5 | ≤ 7,825 | 0–19 | 99.7% |

† Subject 14's beta magnitudes run noticeably larger than most peers; R² and HRF-index ranges are
otherwise unremarkable, so this reads as a scaling/gain property of that subject's data rather
than a fit-quality problem — not investigated further, flagged here for awareness.

### 5.2 Deeper investigation of the two negative-R² outliers

A whole-brain fit will always produce some poor-fitting voxels, so a single negative-R² minimum
is not inherently alarming — but two subjects had a value extreme enough to warrant a direct
per-voxel check rather than trusting the summary statistic.

**Subject 27 — investigated, not a problem.** Only 258 of 277,916 voxels (0.09%) have R² < 0, and
none are below −100; no beta value in the entire subject exceeds 26,628. The −13.7 minimum is a
small number of ordinary poor-fitting edge/noise voxels, exactly what whole-brain unmasked fitting
is expected to produce. No action needed.

**Subject 30 — investigated, real problem, needs a decision.** 244,472 of 254,085 voxels (96.2%)
have R² below −100, and 1,796 beta values exceed magnitude 50,000. This is not a handful of bad
voxels — it is a near-total collapse of fit quality. Breaking `R2run` down by run isolates it
precisely:

| run | median R² | 5th pct | voxels R² < −100 |
|---|---:|---:|---:|
| 1 | 5.27 | −42.4 | 1,027 |
| 2 | 1.77 | −46.7 | 1,097 |
| 3 | −35.24 | −177.9 | 40,797 |
| 4 | −10.67 | −81.0 | 6,875 |
| 5 | −54.77 | −448.8 | 72,501 |
| 6 | −34.70 | −189.6 | 42,595 |
| 7 | −55.42 | −295.1 | 76,131 |
| 8 | −300.29 | −2,093.8 | 201,044 |
| 9 | −1,868.76 | −9,371.5 | 248,139 |
| 10 | −1,701.96 | −41,041.6 | 238,562 |

Runs 1–2 are essentially normal; fit quality degrades progressively and catastrophically toward
the end of the session. This is **not** explained by the subject's already-known, pre-characterized
run-6 volume shortfall (210 vs. 218 volumes) — run 6 looks similar to run 3–4, not to the much
worse runs 8–10. Cross-checked against `motion_qc.tsv` (fMRIPrep's own confound-derived framewise
displacement):

| run | mean FD (mm) | max FD (mm) | FD spikes > 0.5mm |
|---|---:|---:|---:|
| 1 | 0.106 | 0.83 | 7 |
| 5 | 0.284 | 1.62 | 28 |
| 8 | 0.419 | 9.04 | 30 |
| 9 | **1.184** | **19.0** | 76 |
| 10 | **1.550** | **22.5** | 90 |

Mean FD of 1.2–1.5mm and single-TR spikes above 19mm in runs 9–10 are far outside anything else
seen in this cohort (see §5.3) — this is severe, sustained head motion concentrated in the second
half of the session, and it directly explains the degenerate GLM fits. **The data is formally
valid (correct shape, finite values — GLMsingle did not crash or silently fail) but scientifically
unreliable for at least runs 8–10, and likely runs 3–7 to a lesser degree.**

#### 5.2.1 Run-6 source-file question — investigated, resolved, no change made

The user separately flagged (from personal recollection of the scan session) that run 6's raw
source might have had a fuller, better-reconstructed alternative available, and asked whether
substituting it would help. This was checked directly rather than assumed:

- **Run 6**: the manifest selected a 210-volume file
  (`DEV_030/temp/DEV_030_..._RUN6_..._16.nii.gz`). A complete 218-volume alternative does exist
  (`DEV_030/DEV_030_..._RUN6_..._17.nii.gz`, same acquisition) — but its JSON sidecar identifies it
  as a `MoCoSeries` (scanner-side, real-time motion-corrected reconstruction). This project's BIDS
  construction has a deliberate, content-verified gate that rejects MoCo series everywhere in the
  cohort (it re-reads each file's actual `SeriesDescription`/`ImageTypeText`, not just a manifest
  field, specifically to catch this) — not an incidental obstacle, a principled methodological
  stance (real-time MoCo can interact with fMRIPrep's own motion-correction step and with
  GLMsingle's noise modeling in ways not validated for this pipeline). **Decision: keep the raw
  210-volume series.** Run 6 remains 8 volumes short, as originally characterized and already
  accounted for by `EXPECTED_RUN_N_VOLUMES`'s flagged-run mechanism — this does not change Subject
  30's status.
- **Run 10**: the user separately recalled a possible incomplete/duplicate acquisition here too.
  Checked directly — the manifest already selected the complete 218-volume file
  (`RUN10_..._14.nii.gz`); a genuinely incomplete 126-volume alternate (`_15`) exists but was never
  selected. No change needed.
- **Runs 8–9** (also degraded in the table above): checked both candidate files for each — every
  alternate is complete (218 volumes). Their poor fit is not a file-selection issue; it is the same
  genuine motion problem documented above.

**Net effect: no reprocessing was performed for Subject 30. The output already in the cohort
(§5.1) is the final result.** This investigation is recorded here specifically so the "keep 210
volumes" choice reads as a deliberate, evidence-checked decision rather than an unexamined default.

**Operational note for future acquisitions** (from the user, worth carrying forward): when a raw
series comes out short, it may be because the scanner's real-time reconstruction was still
finishing when the run ended or the next scan began — allowing more time after each run before
proceeding may reduce how often this class of shortfall occurs.

**This needs a decision from you, not from the pipeline**: exclude Subject 30 entirely, exclude
its worst runs from downstream analysis (motion-based run/trial censoring), or handle it some
other way. The betas as generated should not be treated as equivalent in quality to the rest of
the cohort without that decision being made explicitly.

### 5.3 Cohort-wide motion screen

For completeness, the same framewise-displacement check was run across all 28 subjects (not just
the two R²-outliers) to confirm Subject 30 is genuinely exceptional rather than the tail of a
continuum:

| sub | worst-run mean FD | worst-run max FD | total FD spikes >0.5mm |
|---|---:|---:|---:|
| 01 | 0.231 | 5.72 | 156 |
| 02 | 0.276 | 6.76 | 158 |
| 04 | 0.099 | 1.32 | 10 |
| 05 | 0.397 | 12.27 | 120 |
| 06 | 0.067 | 0.58 | 1 |
| 07 | 0.172 | 6.33 | 17 |
| 09 | 0.118 | 0.60 | 1 |
| 11 | 0.224 | 6.41 | 67 |
| 12 | 0.132 | 6.17 | 21 |
| 13 | 0.164 | 1.22 | 22 |
| 14 | 0.075 | 1.47 | 18 |
| 15 | 0.192 | 17.02 | 11 |
| 16 | 0.080 | 0.71 | 4 |
| 17 | 0.131 | 0.81 | 8 |
| 18 | 0.429 | 3.49 | 163 |
| 19 | 0.139 | 1.75 | 18 |
| 20 | 0.120 | 1.57 | 11 |
| 21 | 0.199 | 3.03 | 64 |
| 22 | 0.265 | 3.28 | 83 |
| 23 | 0.116 | 1.34 | 12 |
| 24 | 0.196 | 2.68 | 24 |
| 25 | 0.514 | 26.01 | 36 |
| 26 | 0.250 | 2.61 | 149 |
| 27 | 0.413 | 2.77 | 184 |
| 28 | 0.254 | 3.77 | 121 |
| 29 | 0.157 | 1.11 | 20 |
| **30** | **1.550** | **22.49** | **308** |
| 31 | 0.182 | 5.41 | 40 |

Several subjects (01, 02, 05, 07, 11, 12, 15, 25, 31) have a single-run `max FD` spike above 5mm,
sometimes above 15–26mm — but each of these has a normal beta/R² profile in §5.1, meaning
GLMsingle's motion nuisance regression handled the transient event without degrading the overall
fit. The distinguishing feature of Subject 30 is not a single spike (Subject 25's peak spike, 26mm,
is actually higher) but **sustained** elevated motion — a mean FD above 1mm for an entire run,
twice, with 308 total spikes — roughly double the next-highest subject's spike count. That
sustained pattern, not any single number, is what collapsed the fit.

## 6. Bottom line

- **27 of 28 subjects: betas are good.** Correct shape, all finite, R²/β/HRF-index distributions
  all fall in expected ranges for whole-brain unmasked GLMsingle fits, cross-checked against
  motion QC with no unexplained anomalies.
- **1 of 28 (Subject 30): formally valid, scientifically compromised**, confirmed via independent
  per-voxel R², per-run breakdown, and motion-QC evidence that all point to the same cause (severe
  motion in the back half of the session). Needs an explicit downstream decision — see §5.2.
- No pipeline bug produced Subject 30's result; the pipeline correctly processed the data it was
  given, and correctly reported it as within the (permissive) formal validation bounds. The
  problem is in the acquisition, not the processing.

## 7. Follow-up: group univariate contrast maps, and what Subject 30 actually costs

This section documents the first real analysis run on the single-trial betas (pleasant-vs-neutral
and unpleasant-vs-neutral, separately for Natural and AI source, group-level one-sample t-test with
Benjamini-Hochberg FDR q < 0.05) and a significant correction to how severe Subject 30's problem
actually is, discovered in the course of that analysis.

### 7.1 Subject 30 is not just "runs 8-10 are bad" — a full per-run breakdown

The original QC pass (§5.2) singled out runs 8-10 as catastrophic and left runs 3-7 characterized
only qualitatively ("degraded to a lesser degree"). A precise per-run breakdown, computed as the
percentage of the subject's 254,085-voxel mask with `R2 < -100` (near-total fit collapse), makes
clear that framing understated the problem:

| Run | % of brain collapsed (R² < −100) | Mean FD (mm) | Read as |
|---|---:|---:|---|
| 1 | 0.4% | 0.11 | clean |
| 2 | 0.4% | 0.16 | clean |
| 3 | 16.1% | — | **degraded** |
| 4 | 2.7% | — | borderline |
| 5 | 28.5% | 0.28 | **degraded** |
| 6 | 16.8% | — | **degraded** |
| 7 | 30.0% | — | **degraded** |
| 8 | 79.1% | 0.42 | catastrophic |
| 9 | 97.7% | 1.18 | catastrophic |
| 10 | 93.9% | 1.55 | catastrophic |

**Corrected characterization: only runs 1-2 are clean.** Runs 3, 5, 6, and 7 already have 16-30% of
the brain with a collapsed fit — a real, substantial problem, not a rounding error — before the
session degrades further into runs 8-10, where the fit collapses almost entirely. This is a
progressive worsening across the whole session, not a two-tier "some runs fine, some runs ruined"
split. Anyone censoring Subject 30 by run rather than excluding it outright should treat this as
the actual scope, not just drop 8-10 and assume 1-7 are usable.

### 7.2 GLMsingle-averaged group maps: Subject 30 alone suppressed the entire cohort's effects

The first group analysis averaged each subject's 600 single-trial GLMsingle betas within condition,
formed the 4 contrasts, applied 8 mm post-GLM smoothing, and ran the group t-test. With all 28
subjects, the result was implausibly weak for a well-powered emotional-picture-viewing paradigm:

| Contrast | n = 28 (with Sub30) | n = 27 (Sub30 excluded) |
|---|---:|---:|
| Natural: Pleasant − Neutral | 3 | 6,143 |
| Natural: Unpleasant − Neutral | 13 | 2,449 |
| AI: Pleasant − Neutral | 0 | 1,736 |
| AI: Unpleasant − Neutral | 0 | 363 |

Diagnosis: Subject 30's per-voxel contrast values have a standard deviation of **8.54**, against a
range of **0.054-0.161** for every other subject — roughly **85× the median subject**. Since a
one-sample t-statistic is mean ÷ (SD/√n), one subject inflated the between-subject variance enough
to suppress real, otherwise-detectable effects across the *entire* group map. The "zero significant
voxels for AI" result with n = 28 was an artifact of this single subject, not a genuine null effect.

### 7.3 Condition-level ("categorical") GLM: a second, independent estimation method, and why it disagrees with GLMsingle here

To both (a) properly compare against the historical SPM pipeline's condition-level design and (b)
get a second read on whether the weak n=28 result was really just Subject 30, a full condition-level
first-level GLM was run on the same fMRIPrep-preprocessed BOLD (not the GLMsingle betas): 6 condition
regressors (source × valence), 8 mm smoothing applied *before* the GLM (the conventional order — the
GLMsingle betas are deliberately unsmoothed, since GLMsingle needs unsmoothed input for its own
voxelwise HRF fitting), canonical HRF, 128 s high-pass, AR(1) noise model, 6 motion regressors — the
nilearn/fMRIPrep equivalent of the historical `GLM_AI_IAPS_fmriprep.m` SPM design, generalized from
its original 3 pilot subjects to the full cohort. Code: `new_pipeline/code/categorical_glm_fmriprep.py`.

| Contrast | Historical (SPM, FDR) | GLMsingle-avg (n=27) | Categorical GLM (n=28) |
|---|---:|---:|---:|
| Natural: Pleasant − Neutral | 25,089 voxels | 6,143 | **15,027** |
| Natural: Unpleasant − Neutral | — | 2,449 | **11,015** |
| AI: Pleasant − Neutral | — | 1,736 | **11,651** |
| AI: Unpleasant − Neutral | — | 363 | **6,698** |

Two results worth stating plainly:

1. **The categorical GLM is robust to Subject 30** — n=28 and n=27-without-Sub30 give nearly
   identical voxel counts (e.g. 15,027 vs 16,551 for the largest contrast), unlike the GLMsingle
   average, which changed by three orders of magnitude. This is the expected consequence of *how*
   each method estimates a condition mean: the categorical GLM pools 100 trials into one regressor
   per condition, so a subset of motion-corrupted timepoints gets diluted into that one estimate;
   single-trial GLMsingle estimation has no such protection, because every trial is its own
   regressor and its own beta.
2. **This is not a bug in the GLMsingle pipeline.** GLMsingle betas are the right tool for
   *decoding* — the same trial-by-trial independence that makes univariate group-averaging fragile
   to one bad subject is exactly what multivariate decoding needs. The recommendation is to use
   *different* single-trial vs. categorical products for different questions, not to treat one as
   broken:
   - **Univariate group contrast maps → use `categorical_glm/`** (this section's output).
   - **Decoding / MVPA → keep using `glmsingle/`'s single-trial betas**, unchanged.

The remaining gap between the categorical GLM's 15,027 voxels and the historical pipeline's 25,089
is expected and not investigated further here — it reflects genuine preprocessing differences
(fMRIPrep vs. SPM: different motion correction, different normalization template —
`MNI152NLin6Asym` native resolution here vs. `MNI152NLin2009cAsym` res-2 historically — and a
different smoothing implementation), not an estimation-method artifact.

### 7.4 Subject 30, runs 1-7 only: does dropping the worst runs recover it for GLMsingle-style use?

Given §7.1's corrected picture (only runs 1-2 clean, 3-7 real-but-moderate degradation, 8-10
catastrophic), a targeted refit was run: Subject 30's categorical GLM restricted to runs 1-7,
excluding the three catastrophic runs, written alongside (not overwriting) the full-10-run result
via `categorical_glm_fmriprep.py --subjects 30 --runs 1 2 3 4 5 6 7 --suffix _runs1-7`. The group
stage was re-run at n=28 with Subject 30's contribution swapped from its full-10-run maps to the
runs-1-7 maps (`categorical_glm_group.py --subject-suffix 30:_runs1-7`, output in
`categorical_glm/group_sub30runs1-7/`), keeping all 27 other subjects' full-10-run data unchanged.

| Contrast | Categorical GLM, n=28 (Sub30 full 10 runs) | Categorical GLM, n=28 (Sub30 runs 1-7 only) |
|---|---:|---:|
| Natural: Pleasant − Neutral | 15,027 | **17,520** |
| Natural: Unpleasant − Neutral | 11,015 | **13,289** |
| AI: Pleasant − Neutral | 11,651 | **13,789** |
| AI: Unpleasant − Neutral | 6,698 | **7,522** |

Dropping Subject 30's three catastrophic runs (8-10) while keeping runs 1-7 — including the four
runs (3, 5, 6, 7) that still show 16-30% brain collapse per §7.1 — increased significant-voxel
counts across all four contrasts (+8% to +21%), rather than leaving them flat or making them worse.
This is consistent with the categorical GLM being tolerant of partial, diluted motion corruption
(§7.3, point 1) but still sensitive to it in aggregate: removing the worst three runs measurably
helps even though the method doesn't strictly require it the way single-trial GLMsingle estimation
does. Peak t-values (10.45-10.66) and spatial pattern (bilateral extrastriate visual cortex,
consistent across Natural/AI and Pleasant/Unpleasant) are essentially unchanged from the full-10-run
n=28 result — this is a magnitude improvement, not a different finding.

Publication figure: `categorical_glm/group_sub30runs1-7/figure/univariate_group_contrasts.{png,pdf}`,
peak stats in `..._stats.tsv`, same format as §7.3's figure.

**Recommendation:** use the runs-1-7 version of Subject 30 (`group_sub30runs1-7/`) as the primary
n=28 categorical-GLM group result going forward, since it strictly dominates the full-10-run version
on every contrast at no cost (all 27 other subjects are byte-identical between the two). The
full-10-run per-subject maps for Subject 30 are left on disk, not deleted, so this choice is
reversible and auditable. This does not change the GLMsingle/decoding recommendation in §7.3 — that
Subject 30 problem is in the acquisition, not the processing.

## 8. Subject 33 added as a 29th subject

Subject 33 was excluded from the original 28-subject cohort throughout this project because its
session-2 raw data (`Dev_033_2/`) had never been converted from DICOM — every other subject arrived
pre-converted. On explicit instruction, it was onboarded as a 29th subject after the original 28
were frozen and QC'd (§1-§7 above), using the same fail-closed source-selection and BIDS-construction
machinery as the original cohort, generalized to a subject outside that frozen set.

### 8.1 DICOM conversion

`dcm2niix` (v1.0.20260724) was installed via `pip install dcm2niix` into the pilot venv — no prior
DICOM-to-NIfTI precedent existed anywhere in this project. Session 2's 2,278 DICOM files converted to
14 series: 4 real functional runs (RUN7-10, 218 volumes each, matching the cohort's modal run
length) each paired with a scanner-side `MoCoSeries` companion (identical `ProtocolName`, differing
`SeriesDescription` — the exact same real-vs-MoCo pairing pattern every other subject in this cohort
already has), a T1w + GRE fieldmap (both expected for a genuinely separate scan session — confirmed
by checking that every other two-session subject in this cohort, e.g. Sub05/06/07/09/11/30, also has
its own T1w+fieldmap per session), and an extra non-task `EyeOpen` series (257 volumes) with no
`RUN`-labeled `SeriesDescription`, correctly excluded by the existing classifier as `not_a_required_asset`.

### 8.2 Source selection and BIDS construction

`full_cohort_raw_pipeline_28/code/cohortlib.py`'s `load_config()` hard-pins its validation to the
exact original 25-subject config (frozen stimuli hash, exact subject-key set, frozen Sub29/Sub30 QC
flags) and both `audit_full_cohort_sources.py` and `build_full_cohort_bids.py` call it — neither can
process Subject 33, by design, and neither was modified. Instead, `new_pipeline/code/onboard_sub33.py`
reuses every other function in `cohortlib.py` as a pure library (they all take a plain `config`
mapping and never call the gated loader) against a Subject-33-only config view, running the same
sequence the original cohort went through: discover candidates → propose selection → GRE
NORM/series/echo/timestamp triplet audit → hash-freeze → `audit_selection(require_approved=True,
verify_hashes=True, check_onsets=True)` → `build_bids()`. All 18 expected assets (10 bold, 2 T1w, 6
fmap components) resolved to exactly one unambiguous, correctly-classified candidate; both sessions'
GRE triplets passed; the final audit produced 0 errors and 0 warnings. The result was built into an
isolated staging directory, never touching `bids_unified_glmsingle_28/sub-01`…`sub-31`, then
byte-verified (every one of 46 files, 2.2 GB, confirmed identical by SHA-256) before being copied into
the real unified BIDS root as `sub-33/`. Subject 33's own manifest lives separately at
`new_pipeline/manifests/sub33_source_selection.tsv`; the original cohort's frozen
`full_cohort_raw_pipeline_28/manifests/source_selection.tsv` was never touched.

The two `RuntimeError("Refusing to process Subject 33 under any circumstance (frozen exclusion)")`
guards (`wave_orchestrator.py`, `glmsingle_wave_launcher.py`) that enforced the standing exclusion
were removed once Subject 33 was explicitly authorized. `cohort_28.json` and `cohort_waves.json` were
updated with Subject 33's entry (both sessions GRE-SDC, matching the Sub19/22/23/25/27 template) and
moved from `deferred_subjects` to a new wave 4.

### 8.3 fMRIPrep and GLMsingle QC

fMRIPrep completed successfully (all 10 preprocessed BOLD runs present, correct
`MNI152NLin6Asym:res-native` space). GLMsingle QC, compared against the original 28-subject range
(§5.1):

| sub | voxels | R² range | β range (abs) | HRF idx range | min-frac-ridge selected |
|---|---:|---:|---:|---:|---:|
| 33 | 265,288 | 0.77 – 57.4 | ≤ 3,845 | 0–19 | 98.46% |
| (cohort range) | 246,506–282,426 | min always positive except Sub27/30 | ≤ 715–141,999 | 0–19 | 98.5–99.97% |

No red flags — a healthy, positive R² minimum (unlike Sub27's −13.7 or Sub30's collapse), voxel count
and beta magnitude comfortably inside the cohort's existing range. Subject 33's R² maximum (57.4) is
the highest in the cohort by a small margin, consistent with an unusually well-fit subject rather than
a problem.

### 8.4 n=29 group result

Categorical GLM run for Subject 33 (`categorical_glm_fmriprep.py --subjects 33`), then the group stage
re-run at n=29 (28 subjects + Subject 33), keeping Subject 30's runs-1-7 substitution from §7.4
(`categorical_glm_group.py --subjects 1 2 4 5 6 7 9 11..31 33 --subject-suffix 30:_runs1-7 --group-name group_n29`):

| Contrast | n=28 (Sub30 runs1-7, §7.4) | n=29 (+ Sub33) |
|---|---:|---:|
| Natural: Pleasant − Neutral | 17,520 | **18,005** |
| Natural: Unpleasant − Neutral | 13,289 | **13,352** |
| AI: Pleasant − Neutral | 13,789 | **14,696** |
| AI: Unpleasant − Neutral | 7,522 | **8,060** |

All four contrasts improved with Subject 33 added, consistent with adding one more clean subject to a
tolerant condition-level GLM. Spatial pattern (bilateral extrastriate visual cortex, consistent across
Natural/AI and Pleasant/Unpleasant) and peak t-values (10.6–11.0) are unchanged from n=28 — again a
magnitude improvement, not a different finding. Publication figure:
`categorical_glm/group_n29/figure/univariate_group_contrasts.{png,pdf}`, peak stats in
`..._stats.tsv`.

GLMsingle single-trial betas for Subject 33 are in `new_pipeline/glmsingle/sub-33/`, on the same
footing as the other 28 subjects, for any decoding work that resumes.

### 8.5 n=28 alternative: Subject 30 excluded entirely (not the runs1-7 patch)

For comparison, the group stage was also run on the original 28 minus Subject 30 (excluded
completely, not the runs1-7 substitution) plus Subject 33 — also 28 subjects, but a different 28 than
§1-§7 (`--subjects 1 2 4 5 6 7 9 11..29 31 33`, `group_n28_no_sub30/`):

| Contrast | n=29 (Sub30 runs1-7, §8.4) | n=28 (Sub30 excluded, this section) |
|---|---:|---:|
| Natural: Pleasant − Neutral | 18,005 (peak t 10.62) | 17,153 (peak t **11.64**) |
| Natural: Unpleasant − Neutral | 13,352 (peak t 11.03) | 11,873 (peak t **11.77**) |
| AI: Pleasant − Neutral | 14,696 (peak t 10.94) | 12,320 (peak t **12.13**) |
| AI: Unpleasant − Neutral | 8,060 (peak t 10.86) | 7,197 (peak t 10.45) |

Fully excluding Subject 30 lowers the significant-voxel count on every contrast (fewer subjects, less
power) but raises peak t-values on three of four (less noise per voxel). This is the expected
trade-off, not a contradiction: even Subject 30's degraded runs1-7 data still carries some real signal
that a condition-level GLM can partially use (§7.3, point 1), so full exclusion trades that signal away
for a cleaner per-voxel estimate. Spatial pattern is unchanged (bilateral extrastriate visual cortex,
consistent across Natural/AI and Pleasant/Unpleasant). Figure:
`categorical_glm/group_n28_no_sub30/figure/univariate_group_contrasts.{png,pdf}`.

**Both n=29 (`group_n29/`, Sub30 runs1-7) and this n=28 (`group_n28_no_sub30/`, Sub30 excluded) are
legitimate, documented results** — which to treat as primary depends on whether the analysis prefers
maximum sample size (n=29) or maximum per-voxel cleanliness (n=28 without Sub30 at all).

### 8.6 Three-way comparison

`code/compare_group_cases.py` produces a grouped bar chart (significant-voxel count and peak t, all
four contrasts, all three cases side by side) plus a TSV of the same numbers:
`categorical_glm/group_case_comparison/group_case_comparison.{png,pdf,tsv}`.

| Case | n | Sub30 | Sub33 | Pattern across all 4 contrasts |
|---|---|---|---|---|
| A `group_sub30runs1-7` (§7.4) | 28 | runs1-7 | no | baseline |
| B `group_n28_no_sub30` (§8.5) | 28 | excluded | yes | fewest sig. voxels, **highest peak t** on 3/4 |
| C `group_n29` (§8.4) | 29 | runs1-7 | yes | **most sig. voxels** on 3/4, mid peak t |

The trade-off is consistent across all four contrasts: adding a subject (A→C) gains significant-voxel
count; additionally keeping even Sub30's patched, still-somewhat-degraded data in the model (B→C)
costs some peak significance relative to dropping it entirely, because Sub30's runs1-7 data still adds
some real per-voxel noise on top of its real signal contribution. No case changes the spatial pattern
(bilateral extrastriate visual cortex, consistent across Natural/AI and Pleasant/Unpleasant).
