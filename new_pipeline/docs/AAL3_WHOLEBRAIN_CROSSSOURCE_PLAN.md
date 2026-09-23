# Whole-brain AAL3 cross-source decoding — execution plan (n=30)

**Created:** 2026-08-21
**Scope:** GLMsingle Type-D single-trial betas → AAL3 whole-brain ROI decoding,
**cross-source contrasts only**, all 30 subjects.
**Status:** PLAN ONLY — nothing executed yet. All parameter decisions resolved (§7).

---

## 1. What this is

Take the same 30-subject GLMsingle Type-D betas already used for the Kastner/Wang
visual-ROI work, and decode valence across the **whole brain** using AAL3 parcels
instead of 17 visual ROIs. Cross-source only — 4 contrasts:

```
train_natural_pleasant_vs_neutral_test_ai
train_ai_pleasant_vs_neutral_test_natural
train_natural_unpleasant_vs_neutral_test_ai
train_ai_unpleasant_vs_neutral_test_natural
```

Cross-source uses the **corrected no-CV design** established 2026-08-21: train one
classifier on all 200 source-A trials, test once on all 200 source-B trials. No
folds, no repeats. The prior Sub4 AAL3 script used 10-fold LORO for cross-source —
that is the exact design flaw fixed this session and **must not be carried forward**.

---

## 2. Verified facts (probed, not assumed)

| Fact | Value |
|---|---|
| Beta grid | (83, 105, 71) @ 1.797 × 1.797 × 2.25 mm, MNI152NLin6Asym native-res |
| Grid/affine across 30 subjects | **Identical** — one atlas resample serves all 30 |
| Betas | `glmsingle/sub-XX/glmsingle/TYPED_FITHRF_GLMDENOISE_RR.hdf5`, shape (n_vox, 1, 1, 600), ~700 MB each |
| Analysis mask | per-subject, 246,506 – 282,426 voxels (median 263,989) |
| AAL3 source | `projects/data/masks/AAL3/AAL3v1.nii.gz` (2 mm) + `roi_labels.csv` |
| AAL3 regions used | **170 raw, L/R separate** (not bilateral-merged) |
| Labeled voxels on target grid | 204,677 (matches prior Sub4 run exactly) |

### ROI attrition — the frozen set is 154 regions

Applying a 10-voxel floor that must hold in **all 30** subjects:

- **KEEP: 154 regions.** Sizes (sub-01): min 18, median 834, p75 2,024, max 5,837;
  196,640 voxels retained.
- **DROP: 16 regions**, all brainstem/placeholder structures:
  `Cingulate_Ant_L/R` and `Thalamus_L/R` (empty AAL3 placeholder ids — the atlas
  replaced these with named subdivisions that are present and decode normally),
  `VTA_L/R`, `SN_pc_L/R`, `Red_N_L/R`, `Raphe_D`, `Raphe_M`, `Thal_Re_L/R`, `LC_L/R`.

Attrition is subject-dependent, so the set must be frozen from the 30-subject
intersection up front — group statistics assume a balanced n=30 in every cell and
break if the ROI list varies per subject. (Sub4's single-subject run dropped fewer
regions for exactly this reason.)

---

## 3. Phases

Everything runs **twice — once at 8 mm, once at 0 mm** (§7 Q1), into parallel output
trees. Phases 0 and the ROI freeze are shared.

### Phase 0 — Freeze ROI set + prepare atlas *(one-time, ~2 min)*
- Resample `AAL3v1.nii.gz` → native grid, nearest-neighbor (`order=0`). Adapt
  `prepare_aal3_atlas.py` from the Sub4 work (validated; reproduces the 204,677-voxel
  result), switching from `group_id` grouping to raw `id`/`roi_name`.
- Intersect with all 30 analysis masks; emit `roi_manifest.csv` with the frozen
  154-region set, per-subject voxel counts, and an exclusion reason per dropped region.
- **Gate:** manifest lists exactly 154 kept / 16 dropped, every kept region ≥10 voxels
  in all 30 subjects.

### Phase 1 — Cache per-subject AAL3 matrices *(~30–50 min total, 4–6 workers)*
- Per subject per smoothing level: load Type-D betas → smooth (8 mm / none) → extract
  the 154 regions' voxels → save `cache/sub-XX.npy` **ordered by region** so each is a
  contiguous column slice, plus `sub-XX_index.npz` of region→(start, stop).
- Plain `.npy`, **not** compressed `.npz`, so Phase 3 can later `mmap_mode="r"` and
  page in only the region it needs.
- Size: 600 × ~197k × float32 ≈ 472 MB/subject ⇒ **~14 GB per smoothing level, ~28 GB
  total** on N:.
- **Gate:** pilot 3 subjects; verify shapes/dtypes and that a memmap region slice
  round-trips identically to a direct load. Cast any label/string arrays to
  fixed-width unicode — the `allow_pickle` bug that bit the Kastner cache.

### Phase 2 — Observed cross-source decode *(~15–30 min per smoothing level)*
- 30 subjects × 154 regions × 4 contrasts = **18,480 fits** per smoothing level
  (36,960 total).
- Per fit: per-voxel z-score fit on train-source only → `SVC(kernel="linear", C=1)` →
  score once on all 200 test-source trials.
- Parallelize over (region, contrast) with `threadpool_limits(1)` inside workers +
  `joblib.Parallel(backend="loky")` outside — the pattern proven in
  `decode_roi_random10fold.py`.
- **Gate:** pilot Sub01; 616 rows, accuracies in a sane band, deterministic on rerun.

### Phase 3 — Group statistics, t-test track *(~2 min)*
- One-sample t-test vs 50% per cell, BH-FDR across the **154 regions within each
  contrast** (matches the Kastner convention). Also emit an across-all-616-cells
  variant.
- Emit `group_aal3_stats.csv` per smoothing level, plus an 8 mm vs 0 mm comparison
  table (per-region accuracy delta, significance agreement/disagreement).

### Phase 4 — Outputs
- **Ranked horizontal bar chart** per contrast — 154 regions won't fit the Kastner
  grouped layout; paginate or show top-N with the full table alongside. Same
  blue/orange palette and starring convention as the current charts.
- **Accuracy-map NIfTI** per contrast — paint each region's group-mean accuracy back
  onto the native grid for brain rendering (prior precedent:
  `aal3_3d_brain_accuracy_maps_*`).
- Method note stating the no-CV cross-source design and the FDR scope.

### Phase 5 — Stelzer permutation *(DEFERRED — §7 Q3)*
Not part of this pass. When authorized: per (subject, region) 100 label shuffles × 4
contrasts, single fit/score per shuffle → 30 × 154 × 4 × 100 = **1,848,000 fits per
smoothing level** (~18 h serial, **~1–1.5 h at 20 workers**; ~2–3 h for both levels).
Dispatch (subject, region) = 4,620 units, restartable with skip-if-exists. Then
`stelzer_group_resample.py` with the 154-region list, 100,000 group draws per cell.
Until this lands, **significance is t-test-based and provisional.**

---

## 4. Cost estimate

Anchor: prior Sub4 AAL3 run did 2,840 fits in ~100 s ⇒ **~0.035 s/fit** unsmoothed at
this voxel scale. 8 mm raises per-fit cost (the Kastner 8 mm pilot ran ~0.12 s/fit on
much smaller ROIs), so the 8 mm tree will be the slower of the two.

| Phase | Work | Estimate |
|---|---|---|
| 0 | atlas + 154-region manifest | ~2 min |
| 1 | cache 30 subjects × 2 smoothing | 30–50 min |
| 2 | 36,960 fits (both levels) | 30–60 min |
| 3 | t-test + FDR + 8 vs 0 comparison | ~2 min |
| 4 | charts + accuracy maps | ~10 min |
| **Total to first result** | | **~1.5–2 h** |
| 5 | permutation (deferred) | +2–3 h at 20 workers |

---

## 5. Methodological caveats to state in the writeup

1. **ROI size confound.** Kept regions span 18 to 5,837 voxels. Larger parcels give
   the classifier more features; accuracy differences across regions partly reflect
   size, not only information content. Do not rank regions naively without noting
   this. (Optional mitigation: fixed-k voxel selection per region.)
2. **Group-mean inference.** Both the t-test and (later) Stelzer resampling answer
   "is the population mean above chance", weaker than a prevalence claim
   (Allefeld/Görgen/Haynes 2016). Same caveat as the Kastner results.
3. **Dropped nuclei.** The 16 excluded regions are brainstem/small subcortical
   structures; exclusion is a resolution limit, not a null result.
4. **Provisional significance** until Phase 5 runs — the t-test is the weaker test
   this project already moved away from for the Kastner work.
5. **Smoothing comparison is not free of circularity** if the 8 mm vs 0 mm choice is
   made post hoc based on which gives more hits. Report both; pre-commit to 8 mm as
   the headline (cohort convention) with 0 mm as the robustness check.

---

## 6. Conventions honored

- All outputs on **N:** under `new_pipeline/aal3_crosssource/{smooth08,smooth00}/`.
  Nothing on C:.
- **No SPM anywhere** — GLMsingle betas only. The AAL3 NIfTI ships inside an SPM
  toolbox folder but is used purely as atlas geometry; no SPM estimation is involved.
  Flagged explicitly rather than passed over silently.
- **Fail-closed** — scripts refuse to overwrite an existing output dir; per-unit
  skip-if-exists for restartability.
- Deterministic seeding via `zlib.crc32` (never Python's randomized `hash()`).
- Pilot-before-launch at every phase gate.
- All WSL invocations through the PowerShell tool, distro `Ubuntu-22.04`.

---

## 7. Decisions — resolved 2026-08-21

**Q1. Smoothing → BOTH.** Run the full pipeline at 8 mm and 0 mm into parallel trees
and compare. 8 mm is the cohort convention and keeps these results comparable to the
Kastner charts; 0 mm respects parcel boundaries. Doubles Phases 1–2 (cheap) and would
double Phase 5 (deferred).

**Q2. Granularity → 170 raw regions, L/R separate.** Preserves lateralization, which
bilateral merging discards by construction. Frozen usable set: **154 regions**.
FDR family is 154 per contrast.

**Q3. Permutation → deferred.** Land the observed whole-brain map plus t-test+FDR
first, then decide whether to spend the permutation compute.

---

## 8. Unrelated items still open (carried forward, not part of this task)

- **Sub30 motion** — exclude / censor runs 8–10 / keep with caveat. Never decided.
- **Whitening leakage** — leak-free fix validated in `test_whitening_leakage.py`,
  never wired into production.
