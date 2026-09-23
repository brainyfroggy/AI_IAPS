# Unified GLMsingle analysis — design freeze & handoff

**Status:** `frozen_for_wave_1_prep` — **scope narrowed to single-trial beta
generation only.** Parameters below are fixed. Read this whole document
before touching anything; it is written to be self-contained for a fresh
agent session (Codex or a new Claude Code chat) with no prior context.

**Last updated:** 2026-08-05

---

## 1. Why this exists

The project has an existing group-level **univariate** contrast-map pipeline
(SPM, categorical GLM, historically old-normalize+8mm; see
`AI_IAPS/GLM/Step2_1st_2nd_lvl_analysis.ipynb`) and an existing **multivariate**
single-subject sensitivity study (`AI_IAPS/Decoding/sub4_spatial_sensitivity_32/`,
40 settings on Subject 4 only) that established GLMsingle Type-D outperforms the
historical SPM LS-A estimator for within-source decoding by a wide margin, and
that spatial-workflow/smoothing choices matter far less than the estimator
choice.

This project (`new_pipeline`) exists to eventually run **one unified
analysis** — same space, same resolution, same estimator — across the full
cohort, feeding both a univariate view and multivariate decoding from a
single GLM fit per subject, so no reviewer can characterize the two analyses
as using inconsistent methods.

**Current scope (2026-08-05 revision): this wave generates single-trial
betas only.** Downstream analyses (univariate, ROI decoding, categorical GLM
comparison, searchlight) are deliberately deferred — see section 6. This is
a resequencing, not a scope cut: the expensive, foundational, hard-to-reverse
work (fMRIPrep + GLMsingle estimation) happens now; the cheap, easily-revised
decisions (smoothing level, which downstream analyses, in what order) are
made later once actually needed.

## 2. Frozen scientific parameters — this wave

| Parameter | Value |
|---|---|
| Space | MNI152NLin6Asym |
| Resolution | `res-native` — acquired functional resolution, 1.7966 × 1.7966 × 2.25 mm |
| Primary estimator | GLMsingle Type-D (voxelwise HRF + GLMdenoise + fractional ridge) |
| Smoothing | **Not decided — not needed for this wave, see below** |
| TR | 1.8 s |
| Stimulus duration | 3.0 s |
| Runs / subject | 10 |
| Trials / run | 60 (600 / subject) |
| Conditions | Pleasant, Neutral, Unpleasant × {Natural, AI} |
| fMRIPrep version | 25.1.3 (pinned digest — see `sub4_spatial_sensitivity_32/DESIGN_FREEZE.json` for the exact digest string; reuse the same pin) |

### Why smoothing is a non-decision for this wave, not a deferred risk

This isn't a gap — it genuinely doesn't need an answer yet. GLMsingle's
voxelwise HRF fitting and GLMdenoise noise-pool selection require **unsmoothed**
input regardless of what happens downstream (confirmed in the Subject 4
study and in `run_glmsingle_space.py`'s own convention). Smoothing has
**always** been applied post-GLM, per downstream analysis, in this project's
established convention — never baked into the GLM estimation step itself.
So generating single-trial betas now, with the smoothing question still
open, produces exactly the same betas regardless of what gets decided later.
The betas are smoothing-agnostic by construction.

When downstream work resumes, the applicable prior finding still holds and
does not need to be re-litigated: the Subject 4 40-cell study found
GLMsingle within-source decoding accuracy at 8mm was equal to or better than
every lighter level in every spatial branch (`subject_t1w` 55.1% at 8mm vs
52.7–53.7% at 0–5mm; same pattern for `bold_acquired_grid` and
`mni_res_native`) — so 8mm remains a well-supported default when that
decision is actually needed, just not committed to here.

### Why `res-native` rather than `res-2` (unchanged reasoning)

The acquisition is ~1.8mm in-plane; resampling to a 2mm isotropic grid
discards spatial specificity deliberately acquired, which matters most in
the small, densely-labelled regions the existing univariate contrast maps
already implicate (amygdala, OFC, insula). `res-native` reaches a
group-comparable template space while preserving the acquired resolution.

## 3. Cohort — **28 subjects, Subject 33 explicitly excluded from this wave**

```
Included (28): 1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                22, 23, 24, 25, 26, 27, 28, 29, 30, 31
Excluded (prespecified): 3 (Emily), 8 (earrings), 10 (not usable)
Deferred: 33 — new subject, raw data present
  (N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\rawfMRI\IAPS-DEV\Dev_033
   and Dev_033_2; onsets in
   N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\DataRecording\Sub33),
  but session 2 is unconverted DICOM (dcm2niix not yet run) and it has no
  audit history. Do not include Sub33 in this wave under any circumstance
  unless explicitly told otherwise. It will be handled in a later wave.
```

### Wave breakdown (10 / 10 / 8)

Sequential by subject ID — no other ordering logic implied. Reorder only on
explicit instruction.

```
Wave 1 (10): 1, 2, 4, 5, 6, 7, 9, 11, 12, 13
Wave 2 (10): 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
Wave 3 (8):  24, 25, 26, 27, 28, 29, 30, 31
```

Do not launch Wave 2 until Wave 1 fully completes and passes its gates
(fMRIPrep QC, GLMsingle output validation — see section 9). Same for Wave 3
after Wave 2.

### Known per-subject acquisition notes (carry forward from the prior audit)

- Sub1, Sub2: use the complete direct raw tree; ignore `_2` duplicate subsets.
- Sub11 ses-02 logical runs 7–10 → scanner labels 8–11.
- Sub12 ses-01 logical 2–6 → scanner labels 3–7.
- Sub13 ses-02, Sub15 ses-01: explicit duplicate resolution required.
- Sub16 ses-02 logical 9–10 → scanner labels 10–11.
- Sub17, Sub20, Sub21: GRE ses-01, SyN ses-02 (Sub17 also has logical 2–6 →
  scanner 3–7).
- Sub18: GRE ses-01, SyN ses-02.
- Sub19, Sub22, Sub23, Sub25, Sub27: GRE both sessions.
- Sub24, Sub26, Sub28: SyN ses-01, GRE ses-02.
- Sub29: ses-01 run-06 has 216/218 volumes (2 short); final event still within
  acquisition — retain with QC flag, do not silently drop.
- Sub30: ses-01 run-06 has 210/218 volumes (8 short); final event still within
  acquisition — retain with QC flag, do not silently drop.
- Sub31: SyN ses-01, GRE ses-02.

Full detail in `AI_IAPS/Decoding/full_cohort_raw_pipeline_28/RUNBOOK.md` and
`config/cohort.json` — reuse those input-audit gates verbatim rather than
re-deriving them; this cohort was already audited once for a different (ERP
Avg-decoding) analysis, and the raw-data facts above don't change with the
new analysis design.

## 4. Storage plan — nothing persists on C:

### `LAB_IAPS_AI` — raw + preprocessed fMRI data

```
N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\
  rawfMRI\IAPS-DEV\DEV_0XX\              <- existing, unchanged, read-only input
  DataRecording\SubXX\                    <- existing, unchanged, read-only input (onsets)
  bids_unified_glmsingle_28\              <- NEW: BIDS root for this pipeline
    sub-XX\...
  fmriprep_derivatives\unified_glmsingle_28\   <- NEW: fMRIPrep outputs
    sub-XX\...
```

Do **not** reuse `preprocessed_fMRIdata/dev-sess/` — that is the old SPM
realign+normalize pipeline's convention (different format, different
purpose).

`bids_unified_glmsingle_28` should **symlink** BOLD/anat files back to the
raw acquisition rather than copying bytes (confirmed pattern: see
`~/ai_iaps_pilot_bids` inside WSL, all entries are symlinks to
`/mnt/n/...rawfMRI/IAPS-DEV/...`). Keeps the BIDS tree tiny.

### `AI_IAPS/new_pipeline` — this wave populates only `glmsingle/`

```
N:\Experimental_Data\yujunchen\projects\AI_IAPS\new_pipeline\
  DESIGN_FREEZE.md          <- this file
  config\                    <- cohort/wave config, frozen parameters as JSON
  code\                      <- pipeline scripts (see section 7)
  logs\                      <- ledgers, per-job stdout/stderr
  status\                    <- subject_status.tsv-style tracking
  docs\                      <- RESTORE_DROPBOX.txt and other operational notes
  glmsingle\sub-XX\                        <- THIS WAVE'S OUTPUT: raw single-trial betas
                                               (native res, unsmoothed, mni_res_native space)
  --- everything below is a placeholder only, not populated this wave ---
  smoothed_betas\
  univariate_condition_averaged\
  categorical_glm_comparison\
  roi_decoding\
  group_level\
  searchlight\
  summary\
```

One real example already exists and can be used as a format reference:
`AI_IAPS/Decoding/sub4_spatial_sensitivity_32/production_v2_40_attempt02/glmsingle/mni_res_native/`
contains Subject 4's GLMsingle output for this exact space (`analysis_mask.nii.gz`,
`flat_mask_indices.npy`, `trial_manifest.tsv`, `glmsingle/TYPED_FITHRF_GLMDENOISE_RR.hdf5`
with a `betasmd` dataset of shape `(n_voxels, 1, 1, 600)`). This wave's output
for each of the 28 subjects should match this structure. (Whether to literally
reuse Subject 4's existing file or regenerate it under the new run is still
open — see section 10.)

### Hard rule: C: drive

- **No project data, results, or intermediates are written to C: at any
  point.** All of the above lives on N:.
- The **only** things allowed on C: are transient tool/runtime state that
  cannot be relocated: Docker Desktop's own VHDX (holds the pinned fMRIPrep
  image + per-run scratch volumes) and the WSL2 Linux filesystem (holds the
  GLMsingle Python venv). Both are software/runtime, not data — see section 8.
- fMRIPrep's Docker working volume for each subject must be a **named,
  subject-specific volume** (e.g. `ai_iaps_unified28_fmriprep_work_sub-XX`)
  and must be **removed (`docker volume rm`) immediately after that
  subject's fMRIPrep run succeeds and its outputs are verified on N:.** Do
  not let scratch volumes accumulate across 28 subjects — this is exactly
  what filled C: to 2 GiB free earlier this week.
- Before launching any wave, check `C:` free space against the 50 GiB hard
  floor. If below floor, stop and report — do not proceed.

## 5. Concurrency / resource plan

WSL2 is configured at **96 GB memory / 24 processors** (raised from a prior
32 GB/12 cap this week specifically to support this pipeline; confirmed via
`free -h` inside WSL: ~94 GiB actually available after overhead).

| Stage | Concurrency | Per-instance resource | Notes |
|---|---|---|---|
| fMRIPrep | **4** | **`memory_mb: 20000`** (not 26000), `nthreads: 6`, `omp_nthreads: 2` | See flag below. **In scope this wave.** |
| GLMsingle | **5** | — | Empirically validated safe at 5-way on this machine. **In scope this wave.** |
| Downstream analysis | 5 (reserved) | — | **Not used this wave** — no downstream analysis is being run. Documented here so the number carries forward correctly whenever that work resumes; don't rederive it. |
| Searchlight | out of scope | — | Separate future scoping pass, unrelated to this wave |

**⚠️ Resource flag, read before configuring fMRIPrep:** the previously-frozen
`sub4_spatial_sensitivity_32` fMRIPrep config used `memory_mb: 26000`. At
**4-way concurrency that is 104 GB against ~94 GiB actually available** — it
will not fit and risks an OOM kill mid-subject. **This document changes
`memory_mb` to `20000`** (4 × 20 GB = 80 GB, ~14 GB headroom). Do not
silently revert to 26000 — that value was tuned for a *different*
concurrency (1–2 way).

`C:` hard floor: 50 GiB, enforced before every stage launch (see section 4).

## 6. Scope for this wave: **single-trial beta generation only**

```
fMRIPrep (preprocessing, MNI152NLin6Asym res-native)
   -> GLMsingle Type-D (single-trial betas, unsmoothed, per subject)
   -> STOP
```

That is the entire pipeline for this wave. Nothing downstream of GLMsingle
is in scope:

- **No smoothing step.** Smoothing is post-GLM, per-analysis, and no
  analysis is being run yet (section 2).
- **No condition-averaged univariate contrasts.**
- **No ROI decoding.**
- **No categorical GLM comparison arm.**
- **No searchlight.**

All of the above remain real, planned future work — nothing here is
cancelled, only resequenced. When downstream work resumes, revisit section
10's open items (smoothing level, categorical-GLM contrast set, group-level
statistical approach) before building against this wave's output.

## 7. Code to reuse vs. build

### Reuse directly (already fixed and validated this week)

From `AI_IAPS/Decoding/sub4_spatial_sensitivity_32/code/`:
- `run_glmsingle_space.py` — GLMsingle runner + `discover_branch()` fMRIPrep
  derivative discovery. **Contains fixes for 3 real bugs found this week**:
  fMRIPrep 25.1.3 filename discovery (no literal `space-func`/`res-native`
  tokens exist), a confounds-path derivation bug, and these are load-bearing
  — do not revert to an earlier version or re-derive from scratch.

That is the only script this wave's scope actually needs from the prior
codebase. (`decode_roi_singletrial.py`, `transform_native_betas_to_mni.py`,
`prepare_kastner_atlases.py`, and the SPM LS-A scripts are all downstream-analysis
tooling — keep them in mind for later, but nothing to wire up now.)

### Build new for this pipeline (this wave)

- BIDS construction script targeting
  `LAB_IAPS_AI\bids_unified_glmsingle_28\` (symlink pattern, per section 4).
- fMRIPrep orchestrator: multi-subject, wave-aware, `memory_mb: 20000`,
  4-way concurrent, named per-subject work volumes with post-success cleanup,
  writing to `LAB_IAPS_AI\fmriprep_derivatives\unified_glmsingle_28\`.
- GLMsingle orchestrator: multi-subject, wave-aware, 5-way concurrent,
  wraps `run_glmsingle_space.py` (branch = `mni_res_native` only — this
  pipeline doesn't run the other 4 spatial branches from the sensitivity
  study, just the one frozen target space), writing to
  `AI_IAPS\new_pipeline\glmsingle\sub-XX\`.
- Wave-level orchestrator with the same fail-closed conventions as
  `sub4_spatial_sensitivity_32` / `full_cohort_raw_pipeline_28`: refuse to
  overwrite existing output, refuse ambiguous resume, append-only ledger,
  new namespace per retry rather than silent overwrite.

**Not needed this wave** (deferred along with the analyses that need them):
post-GLM smoothing step, condition-averaging script, categorical GLM MATLAB
script, ROI decoding invocation, searchlight anything.

**Before writing any new script, read `sub4_spatial_sensitivity_32/code/`
end to end.** Nearly every failure mode this week (filename discovery,
orphaned child processes surviving a parent kill, JSON serialization of numpy
types, OOM from over-parallelized Docker jobs) has already been hit and
fixed there.

## 8. C: drive incident — for context, already resolved

Earlier this week C: filled to 2 GiB free from two causes, both diagnosed and
fixed: (1) Dropbox's desktop client restarted and re-hydrated ~443 GiB of
cloud files onto C: — its autostart is now disabled (per-user
`StartupApproved` flag). (2) Five Docker named volumes holding fMRIPrep
working directories from the Subject-4 study had accumulated on C: unremoved
— deleted, and section 4 above codifies "remove per-subject work volume
immediately after success" to prevent recurrence at 28-subject scale.
Verified via direct inspection that no actual project data/results were ever
duplicated onto C: — BIDS files referenced from WSL are symlinks to N:, not
copies.

## 9. Execution checklist for whoever picks this up

1. Check `C:` free space ≥ 50 GiB. If not, stop.
2. Confirm WSL2 is at 96 GB/24 processors (`wsl -d Ubuntu-22.04 -- free -h`).
3. Read `sub4_spatial_sensitivity_32/code/` and `RUNBOOK.md` /
   `AUDIT_LOG.md` in full before writing new code.
4. Build the BIDS construction + fMRIPrep orchestrator + GLMsingle
   orchestrator per section 7.
5. Recommend a 1–2 subject smoke test (fMRIPrep → GLMsingle, stop there)
   before launching Wave 1 at full 10-subject scale — this configuration has
   never been run end-to-end, even at this narrower scope. Cheaper to
   validate now than with the old full-pipeline scope, so there's less
   reason to skip this step.
6. Launch Wave 1 (10 subjects). Gate = fMRIPrep QC passed + GLMsingle output
   validated (correct `betasmd` shape, all-finite, matches
   `sub4_spatial_sensitivity_32`'s reference structure). Do not start Wave 2
   until Wave 1 passes.
7. Launch Wave 2, then Wave 3, same gate discipline.
8. **Stop.** Do not proceed to smoothing, condition-averaging, ROI decoding,
   categorical GLM, group-level aggregation, or searchlight without a new,
   explicit scoping conversation — all of that is deferred by this revision,
   not implied to follow automatically once betas exist.

## 10. Open items — none block this wave, all block resuming downstream work

- **Smoothing level** for post-GLM analysis — not needed now (section 2);
  8mm is the well-supported default from the Subject 4 study whenever this
  is revisited, but not committed to here.
- Exact categorical-GLM contrast set (assumed same 6-condition design as
  `GLM_AI_IAPS.m`) — confirm before building that arm, whenever it resumes.
- Group-level statistical approach for the univariate view (one-sample vs.
  paired tests, multiple-comparison correction) — not yet specified.
- Whether Subject 4's existing `mni_res_native` GLMsingle output in
  `sub4_spatial_sensitivity_32/production_v2_40_attempt02/glmsingle/mni_res_native/`
  can be reused for this pipeline's Sub4 rather than recomputed — same space
  and estimator, but built under a different fMRIPrep run/derivative tree.
  Partial evidence this is compatible: a condition-averaging script was
  successfully run against this exact file on 2026-08-05 (see
  `new_pipeline/univariate_condition_averaged/sub-04/`, built ahead of this
  wave's own scope as a format check) and produced valid, correctly-shaped
  output. That's encouraging but is not the same as verifying it against
  whatever this wave's own GLMsingle orchestrator produces — **do not skip
  Sub4 in Wave 1 based on this alone; verify direct compatibility (identical
  mask, identical trial manifest, identical fMRIPrep run parameters) before
  substituting the old file for a freshly-generated one.**
- Sub33 handling is fully deferred — no decisions needed now.

## 11. Change log

- **2026-08-05 (this revision)**: Narrowed scope to single-trial beta
  generation only. Removed smoothing from frozen parameters (deferred, not
  decided — see section 2). Removed all three downstream views and the
  searchlight placeholder from this wave's active scope (section 6) —
  retained as explicitly deferred future work, not cancelled. Concurrency
  plan simplified: fMRIPrep=4 and GLMsingle=5 remain in scope; the
  downstream-analysis=5 setting is retained in section 5 as a reserved value
  for later, unused this wave. Section 7's "build new" list cut to BIDS +
  fMRIPrep + GLMsingle orchestrators only. Execution checklist (section 9)
  now stops after Wave 3's GLMsingle output is validated.
- **2026-08-05 (earlier same day)**: Removed Subject 33 from this wave
  (deferred). Set concurrency to fMRIPrep=4 / GLMsingle=5 /
  downstream-analysis=5, and adjusted `memory_mb` from 26000→20000 to make
  4-way fMRIPrep concurrency fit in the 94 GiB WSL2 budget. Added explicit
  10/10/8 wave breakdown for the 28-subject cohort. Removed searchlight from
  scope entirely (was previously bundled as a downstream view); made it a
  separate future effort. Storage plan finalized: fMRIPrep derivatives under
  `LAB_IAPS_AI`, everything else under `AI_IAPS/new_pipeline`. Workspace
  relocated from `AI_IAPS/Decoding/unified_glmsingle_29/` to
  `AI_IAPS/new_pipeline/`.
- **2026-08-03**: Initial design freeze (29 subjects incl. Sub33, searchlight
  bundled as a downstream view, workspace at `Decoding/unified_glmsingle_29/`).

## 12. Relationship to existing workspaces

- `sub4_spatial_sensitivity_32/` — the completed 40-cell methodological study
  that motivated these choices. Retained, not modified. Also holds a
  condition-averaging format check for Sub4 (section 10) run ahead of this
  wave's own scope, under `new_pipeline/univariate_condition_averaged/sub-04/`.
- `full_cohort_raw_pipeline_28/` — an earlier 28-subject workspace frozen
  around a *different* analysis (ERP-style Avg(random) decoding, its own
  space config). Retained unmodified; this is a new, separate workspace
  rather than an edit to it, consistent with this project's versioning
  convention.
