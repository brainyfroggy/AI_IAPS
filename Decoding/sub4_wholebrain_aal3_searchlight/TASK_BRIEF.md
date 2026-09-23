# Sub4 whole-brain decoding: AAL3 ROI + searchlight radius sweep

**Purpose of this file:** self-contained brief for a fresh chat session (Codex
or a new Claude Code conversation) with no prior context. The user asked to
start this as a new chat rather than continue in the session that wrote this
brief, so treat this file as the entire handoff — don't assume anything not
stated here or in the paths it points to.

**Created:** 2026-08-05

---

## 1. The ask, verbatim

Whole-brain decoding on Subject 4, two approaches:
1. **ROI-based decoding using the AAL3 atlas** (not the Kastner/Wang visual
   atlas used in prior work — AAL3 covers the whole brain, ~166 regions).
2. **Searchlight decoding**, sweeping across several **radius** values, to
   produce **data that justifies a specific radius choice** — not just
   picking a conventional default. This means: run the searchlight at
   multiple radii, compare results systematically, and report why one radius
   is the right call for this dataset.

**Revision (2026-08-05): scope narrowed to cross-source decoding only.**
Both arms (AAL3 ROI and searchlight) run **only the 4 cross-source
contrasts** — train on Natural, test on AI, and vice versa, for
pleasant-vs-neutral and unpleasant-vs-neutral. **Do not run the 4
within-source contrasts for this task.** This was an open scope question in
the original brief (section 8); it's now resolved — cross-source only, not a
starting subset to be expanded later.

## 2. What already exists — verified, reuse this, do not recompute

Subject 4 already has GLMsingle Type-D single-trial betas estimated in
**MNI152NLin6Asym at native functional resolution** (1.7966×1.7966×2.25mm),
from the prior 40-cell spatial/estimator sensitivity study. Confirmed present
at:

```
N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\sub4_spatial_sensitivity_32\
  production_v2_40_attempt02\glmsingle\mni_res_native\
    glmsingle\TYPED_FITHRF_GLMDENOISE_RR.hdf5   <- the Type-D betas (use this one)
    glmsingle\TYPEB_FITHRF.hdf5                  <- a different, lesser model; ignore
    analysis_mask.nii.gz                          <- brain mask, 268,697 voxels
    flat_mask_indices.npy                          <- maps flat voxel index -> mask
    trial_manifest.tsv                             <- 600 trials: run, condition, source, onset
    provenance.json                                <- confirms n_voxels=268697, 10 runs x 224 vols
```

This is the **unsmoothed** native-space beta volume (600 single-trial betas ×
268,697 voxels). Confirm the HDF5 internal structure before writing code
against it (open it, check dataset names/shapes — don't assume from this
description alone).

**Do not re-run fMRIPrep or GLMsingle for Subject 4.** The only new work here
is decoding on top of these existing betas: AAL3 ROI extraction, and a new
searchlight implementation (searchlight does not exist anywhere in this
project yet — this would be the first).

### Reuse for structure/convention, adapt the atlas/method

`sub4_spatial_sensitivity_32/code/decode_roi_singletrial.py` — the existing
ROI decoding script (Kastner/Wang atlas). Its classifier/CV/contrast
structure is the established convention for this project:
- Linear SVC, C=1
- 10-fold leave-one-run-out
- Per-voxel mean/std normalization fit on training folds only
- 8 contrasts: 4 within-source + 4 cross-source (see list below)

For the AAL3 arm, adapt this script's ROI-extraction/classification logic to
the AAL3 atlas instead of Kastner/Wang. For the searchlight arm, this
classifier/CV/contrast scaffolding is still the right one to reuse — only the
feature-extraction step (searchlight sphere instead of ROI mask) is new.

```
Contrasts — CROSS-SOURCE ONLY, per the 2026-08-05 revision (section 1):
train_natural_pleasant_vs_neutral_test_ai
train_ai_pleasant_vs_neutral_test_natural
train_natural_unpleasant_vs_neutral_test_ai
train_ai_unpleasant_vs_neutral_test_natural

Do NOT run these (excluded by the revision, listed only so nobody
accidentally reintroduces them from habit — every prior script in this
project's history defaults to all 8):
within_natural_pleasant_vs_neutral
within_ai_pleasant_vs_neutral
within_natural_unpleasant_vs_neutral
within_ai_unpleasant_vs_neutral
```

This halves the contrast count relative to the original brief (4 instead of
8), which also roughly halves the searchlight cost estimate in section 5.

## 3. AAL3 atlas — verified present

```
N:\Experimental_Data\yujunchen\projects\data\masks\AAL3\
  AAL3v1.nii.gz          <- 2mm-space atlas, use this as the primary
  AAL3v1_1mm.nii.gz      <- 1mm-space variant, also available
  AAL3v1.nii.txt         <- region labels
  roi_labels.csv         <- also region labels, check which is authoritative
```

The atlas is at 2mm or 1mm — Subject 4's betas are at native ~1.8×1.8×2.25mm
in MNI152NLin6Asym. **The atlas needs to be resampled into the betas' grid**
(nearest-neighbor/label interpolation), not the other way around — matching
the established convention from `prepare_kastner_atlases.py` in
`sub4_spatial_sensitivity_32/code/` (read that script for the pattern: it
warps an atlas into each subject-space branch using the same transforms
fMRIPrep produced, rather than resampling functional data into atlas space).

There is also prior AAL3 usage in this project from an earlier (different,
ERP-averaged) decoding pipeline:
```
N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\aal3_dict.npy            (43 MB)
N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\aal3_dict_65rois.npy      (154 MB)
```
These are from a **different space/pipeline** (the old ERP Avg-decoding
convention, not this GLMsingle/MNI152NLin6Asym-native-res pipeline). Check
their format for reference on how ROI dictionaries were structured previously
in this codebase, but do not assume they're directly usable here — the
underlying voxel grid is almost certainly different. Verify before reusing,
same discipline as everywhere else in this project this week.

## 4. Searchlight — build from scratch, no existing implementation to reuse

This project has never run a searchlight analysis. Build it new, but follow
established conventions:
- Same classifier (linear SVC, C=1), same CV (10-fold LORO), same
  per-voxel normalization-within-fold discipline as `decode_roi_singletrial.py`.
- Sphere-based neighborhoods centered on each in-mask voxel (268,697 mask
  voxels for this subject/space — see section 2).
- **Radius sweep**: pick a small set of candidate radii (e.g., something like
  2, 3, 4, 5 voxels — voxel size here is ~1.8×1.8×2.25mm, so state radii in
  both voxels and approximate mm for anyone reading the results later) and
  run the full searchlight at each.
- For each radius, produce a whole-brain accuracy map, and **summary
  statistics that support a radius choice**: e.g., peak accuracy, spatial
  extent/number of above-chance voxels (with a defined threshold), how much
  overlap/stability there is between radii, and computational cost per
  radius. The deliverable the user asked for is a **comparison with data**,
  not just "here's a map at radius=3" — mirror the spirit of the
  `sub4_spatial_sensitivity_32` 40-cell sensitivity study (which produced an
  explicit ranked comparison + interactive artifact) rather than a single
  qualitative pick.
- Fail-closed conventions from this project apply: don't silently overwrite
  a radius's output if rerunning; use a new namespace per attempt if
  something breaks mid-run; append-only ledger of what ran and its result.

## 5. Cost warning — scope the sweep before launching

A prior design pass in this project estimated whole-brain searchlight at
roughly **~5 hours per contrast per subject** at full mask size (~200-270k
voxels × 10 CV folds × an SVM fit per sphere), for a single radius. **With
the cross-source-only revision (section 1), this task runs 4 contrasts, not
8** — roughly ~20 hours per radius at that per-contrast estimate, before any
sweep. **A radius sweep multiplies this by the number of radii tested.** For
one subject this is still tractable (unlike the earlier 29-subject
full-cohort searchlight estimate that ran into the 1,000+ hour range), but:
- Confirm actual per-radius runtime with a small pilot (a handful of
  contrasts, or a masked/coarse sub-region) before committing to a full
  whole-brain × all-radii × all-4-contrasts run.
- Larger radii mean more voxels per sphere → slower per-sphere fits; smaller
  radii mean less spatial pooling → potentially noisier estimates. Both ends
  of the sweep have a real cost/benefit tradeoff worth stating explicitly in
  whatever comparison is produced.

## 6. Machine state, as of 2026-08-05 (relevant if using WSL/Docker)

- WSL2 raised this week to 96 GB memory / 24 processors (from a prior 32
  GB/12 cap) — confirm current state with `wsl -d Ubuntu-22.04 -- free -h`
  before assuming it's still at this level.
- **C: drive discipline is strict in this project as of this week**: nothing
  project-related gets written to C:. C: filled to 2 GiB free earlier this
  week from (a) Dropbox re-hydrating ~443 GiB of cloud files, now
  autostart-disabled, and (b) unremoved Docker fMRIPrep working volumes, now
  cleaned up. If this analysis needs any Docker/containerized step (unlikely
  — GLMsingle betas already exist, this is pure Python/analysis work), keep
  all outputs on N: and check `C:` free space ≥ 50 GiB before launching
  anything.
- GLMsingle-adjacent Python environment already exists in WSL:
  `/home/yujun/.cache/ai_iaps_pilot_venv` — check what's installed there
  (nibabel, sklearn, nilearn?) before setting up a new environment; nilearn
  in particular has built-in searchlight support
  (`nilearn.decoding.SearchLight`) that may be worth using directly rather
  than hand-rolling sphere logic, if its CV/scoring interface can be made to
  match this project's exact fold/normalization discipline. Evaluate, don't
  assume it's a drop-in fit.

## 7. Output location

New workspace, this file's own directory:
```
N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\sub4_wholebrain_aal3_searchlight\
  TASK_BRIEF.md          <- this file
  code\                   <- new scripts (AAL3 ROI decode, searchlight)
  results\
    roi_aal3\              <- AAL3 ROI decoding results, cross-source contrasts only
    searchlight\
      radius_2\ radius_3\ radius_4\ radius_5\  <- (adjust to actual sweep), cross-source only
    radius_comparison\    <- the comparison artifact justifying final radius choice
```
All results above are cross-source decoding only (section 1 revision) — no
within-source outputs should appear anywhere in this workspace.
Nothing on C:, per section 6.

## 8. Open questions to confirm with the user before or during the work

- Exact radius values to sweep (this brief suggests 2-5 voxels as a
  starting point, not a firm decision).
- ~~Whether the searchlight radius sweep should cover all 8 contrasts or
  start with a subset~~ — resolved 2026-08-05: cross-source only, 4
  contrasts (section 1).
- Whether AAL3's 2mm or 1mm variant is preferred once resampled into the
  native-res beta grid (2mm is the more common convention; 1mm may
  over-resolve relative to the ~1.8-2.25mm acquisition).
- Whether smoothing should be applied to the betas before AAL3/searchlight
  decoding, and if so at what level — the parent project's frozen convention
  (`AI_IAPS/new_pipeline/DESIGN_FREEZE.md`) uses 8mm post-GLM smoothing for
  its own reasons; this task didn't specify smoothing, so ask rather than
  assume it should inherit that convention unchanged.
