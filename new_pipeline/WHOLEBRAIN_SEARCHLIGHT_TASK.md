# Whole-Brain Searchlight Decoding — Task Specification

**Audience:** a student (working with an AI coding assistant) picking this up fresh, with
no other context from the project's chat history. Everything needed to implement, pilot,
and run this analysis — code, data locations, and exact commands — is in this file or in
the files it points to.

**Goal:** whole-brain searchlight decoding using the exact ERP-style decoding methodology
this project already finalized (see `FINDINGS_AND_REPRODUCTION.md` in this same folder for
the full history/justification — read that first). This document covers what's *new* for
searchlight: sphere neighborhoods instead of atlas ROIs, parameters specific to this
analysis (`n_repeats=30`, `n_perms=10,000` — different from the ROI-level project's
`n_repeats=100`/`n_perms=100`, a deliberate choice made for this task), and a significance
design chosen to support applying different multiple-comparisons corrections **after** the
expensive part runs once, without rerunning anything.

**Where things go:**
- This file: `new_pipeline/WHOLEBRAIN_SEARCHLIGHT_TASK.md` (already here).
- All code (already written, see §6): `new_pipeline/searchlight_decoding/sphere_*.py`.
- All results: `new_pipeline/searchlight_fixedhrf_full/` (created fresh by the scripts).

---

## 1. The decoding recipe (unchanged from the rest of this project)

1. **Input data:** GLMsingle single-trial betas from `glmsingle_fixedhrf_full_pilot/`
   (the finalized config: `wantlibrary=0` fixed HRF, `wantglmdenoise=1`, `wantfracridge=1`
   — file `sub-XX/glmsingle/TYPED_FITHRF_GLMDENOISE_RR.hdf5`, 600 trials/subject).
2. **Cohort (n=30):**
   `1 2 4 5 6 7 9 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 33 34`
3. **Smoothing:** 8mm FWHM, mask-normalized Gaussian (same as every other analysis here).
4. **Voxel-source z-score** (`decode_roi_erpstyle.compute_voxel_source_zscore`): per-trial
   pattern z-score across voxels, then per-voxel z-score across that source's pooled 300
   trials. Never touches valence labels — computed once per (subject, sphere), reused for
   the observed decode and every permutation shuffle.
5. **Decoding** (`decode_roi_erpstyle.decode_within_avg` / `decode_cross_avg`):
   `SVC(kernel="linear", C=1.0)`. Within-source: `KFold(4, shuffle=True)`, train chunk-
   averaged into 3 patterns/condition, held-out into 1. Cross-source: no fold (trains on
   all of source A, tests on all of source B), both sides chunk-averaged into 4
   patterns/condition.
6. **`n_repeats = 30`** for this analysis specifically (see §4 for why this differs from
   the ROI work's 100 — it's a compute-budget choice for the searchlight scale, not a
   methodological downgrade; if compute allows, revisit upward).
7. **Contrast scope: confirm before running.** This document assumes all 8 contrasts
   (4 within-source + 4 cross-source), matching the Kastner default. If you only need
   cross-source (as the AAL3 whole-brain analysis in this project used), pass
   `--contrast-kind cross` to `erp_wave_launcher.py` in §7 — it cuts total compute
   roughly 5x (see the AAL3 cost breakdown in `FINDINGS_AND_REPRODUCTION.md`-adjacent
   work), which may matter a lot given §4 below.

Reuse `compute_voxel_source_zscore`, `condition_label`, `decode_within_avg`,
`decode_cross_avg` from `code/decode_roi_erpstyle.py` — don't reimplement them.

---

## 2. Sphere neighborhoods instead of atlas ROIs

All subjects in this project share an **identical grid and affine**: shape `(83, 105, 71)`,
voxel size `1.797 x 1.797 x 2.25mm`, confirmed identical across sub-01/02/34. This means
one shared sphere-center grid, built once from any single subject's `analysis_mask.nii.gz`,
is valid for every subject — you do not need to rebuild it per subject. (If you ever run
this on a different cohort, re-verify this assumption first — see the snippet at the top
of `searchlight_decoding/sphere_common.py`'s docstring for how it was checked here.)

`searchlight_decoding/sphere_common.py` (already written) provides:
- `build_sphere_centers(reference_mask_path, radius_mm, step, min_voxels)` — places a
  candidate center on a regular `step`-voxel grid across the mask, keeps every voxel
  within `radius_mm` (true mm distance, not voxel count, since the grid is anisotropic)
  that also lies in the mask, and drops any center whose sphere has fewer than
  `min_voxels` analysis voxels. Returns the union of all retained voxels, a
  `{center_name: indices}` map, and a manifest DataFrame with each center's `(i, j, k)`
  voxel coordinates (needed later to place values back into a volume).
- `load_sphere_glmsingle(root, union, smoothing)` — identical HDF5-load +
  mask-normalized-smoothing logic as `aal3_common.load_aal3_glmsingle`, just fed a
  precomputed voxel union instead of an atlas-derived one.

**Important reuse trick:** the manifest this writes uses the exact same schema
(`id, roi_name, kept`) as the AAL3 kept-regions CSV, which is what
`erp_wave_launcher.py --roi-manifest`, `erp_aggregate_observed.py --roi-manifest`, and
`stelzer_group_resample.py --roi-manifest` already accept (added when this project ran the
AAL3 whole-brain analysis — see those flags in the scripts). **None of those three scripts
need any changes for searchlight** — point `--roi-manifest` at the sphere manifest instead
of AAL3's, same as you'd point it at a different atlas.

---

## 3. Cache build

```bash
python3 searchlight_decoding/sphere_cache_subject_matrices.py \
  --glmsingle-root glmsingle_fixedhrf_full_pilot \
  --reference-subject 1 \
  --radius-mm <RADIUS_MM> \
  --step <STEP> \
  --min-voxels 10 \
  --out-dir searchlight_fixedhrf_full/cache \
  --workers 4
```
Writes `searchlight_fixedhrf_full/cache/roi_manifest.csv` (for `--roi-manifest` below),
`centers_ijk.csv` (center name -> `(i,j,k)`, used by the map-reconstruction step in §8),
and one `sub-XX.npz` per subject in the same schema `stelzer_cache_subject_matrices.py`/
`aal3_stelzer_cache_subject_matrices.py` already use. **Pick `--radius-mm`/`--step` only
after reading §4** — they directly set how many centers exist and therefore how long
everything downstream takes.

---

## 4. Compute cost — read this before choosing a grid density

### 4a. Why the significance-test design changed from an earlier draft of this document

An earlier version of this document recommended a max-statistic (Nichols & Holmes 2002)
whole-brain permutation instead of the per-ROI independent design this project uses
everywhere else, purely because it's cheaper. **That recommendation is retracted for this
version.** The requirement now is: run the expensive part once, then be free to apply
*whatever* significance test/correction seems right afterward — including plain BH-FDR —
without rerunning anything. A max-statistic null is a single distribution shared across
the whole brain; it doesn't give you an independent p-value per voxel, so you can't
retroactively run FDR on top of it in the standard way. The per-ROI **two-tier Stelzer
design already used for Kastner and AAL3 in this project does exactly what's needed**: it
produces one empirical p-value **per sphere, per contrast** (`p_value_raw` in
`stelzer_group_resample.py`'s output), and BH-FDR is a cheap, instant, re-runnable
post-processing step on top of that raw p-value column — recompute it at any alpha, swap
in a different correction method (Bonferroni, Holm, cluster-based, anything), all without
touching the permutation results. **So: reuse `erp_permute_worker.py` /
`erp_wave_launcher.py` / `stelzer_group_resample.py` completely unchanged**, exactly as
was done for AAL3 — only the ROI manifest (§2) is different.

### 4b. The honest cost math

Measured on this project's machine (24 cores), at the ROI-level project's settings
(`n_perms=100`, `n_repeats=100`, cross-only/4 contrasts):

| Sphere size | Measured time/unit |
|---|---|
| ~854 voxels | 269s |
| ~5,842 voxels | 1,229s |
| Fitted | `time ≈ 105s + 0.19s x n_voxels` |

**This analysis uses different settings** — `n_repeats=30` (0.3x) and `n_perms=10,000`
(100x) — a net **~30x more shuffle-and-decode work per unit** than the numbers above
reflect. A rough (deliberately conservative/over-) estimate, scaling the *entire* measured
time by 30x rather than trying to separate the truly-fixed overhead (cache load, one-time
z-score) from the part that actually scales with shuffle count — i.e. this likely
overestimates, which is the safer direction to be wrong in:

```text
time(voxels) ~ 30 x (105 + 0.19 x voxels) seconds  [rough upper bound]
```

A ~125-voxel sphere (radius ~6mm) -> **~64 minutes/unit**. At a step-3 grid
(~10,000 centers, ~1/27 of the ~270,829-voxel mask) x 30 subjects:
**10,000 x 30 x ~3,860s ~ 1.16 billion seconds ~ 36.7 years serial ~ 1.8 years at 20
workers.** Even a much coarser grid doesn't rescue this on a single machine.

**This means the parameters as literally stated (`n_repeats=30`, `n_perms=10,000` as the
first-level within-subject shuffle count) are not tractable on commodity hardware at
whole-brain scale, even with a sparse grid.** Before running anything at scale:

1. **Run your own timing pilot first** (§5) — the estimate above is extrapolated from a
   different parameter regime and could be off in either direction; don't trust it over a
   real measurement on your own hardware.
2. **Consider spending "10,000" on the *group-resample* draws instead of the first-level
   shuffle count.** This project's own ROI-level finalization hit the identical ambiguity
   ("permutation = 10,000 times" could mean either tier) and resolved it by keeping
   first-level shuffles modest (100) and using the extra resolution on the *group* stage
   (`stelzer_group_resample.py --n-draws`, cheap regardless of value — it's pure resampling
   over already-computed null pools, not new decoding). If that's what was actually meant
   here too, use `--n-perms 100` (or a modest number the pilot shows is affordable) in
   `erp_wave_launcher.py` and `--n-draws 10000` (or higher — 100,000 is already the
   existing default and is nearly free) in `stelzer_group_resample.py` instead — this
   removes the 100x multiplier above entirely while still supporting full post-hoc FDR
   flexibility.
3. **A coarser grid or a restricted mask** (e.g. only cortex, or only a specific network,
   rather than truly whole-brain) is the other lever — every doubling of `--step` cuts
   center count by roughly 8x (3 dimensions).
4. **Confirm the actual intent with whoever asked for `n=10,000`** before committing serious
   compute — the gap between the two readings above is a ~100x difference in runtime, and
   it's worth 30 seconds to check.

---

## 5. Pilot-before-scale (mandatory)

1. **Correctness pilot:** 1 subject, a handful of centers placed inside a known Kastner
   ROI (e.g. V1v). Confirm the sphere-based accuracy is in the same ballpark as that ROI's
   already-known result (`erp_permutation_fixedhrf_full/group/group_roi_stats.csv`) —
   catches indexing/affine bugs cheaply.
2. **Timing pilot at the ACTUAL settings you intend to use** (`n_repeats=30` and whatever
   `--n-perms` you land on per §4b) — one small sphere, one large sphere, single worker,
   wrapped in real timestamps (not `$(date +%s)` inside nested shell quoting — write a
   standalone `.sh` script and run it directly; nested PowerShell/WSL/bash quoting silently
   swallowed a timing wrapper earlier in this project's own history). Use the result to
   redo the §4b math with real numbers before picking a final grid.
3. **Null calibration check** — after any permutation run, confirm the null distribution
   centers near 50% (`stelzer_group_resample.py` prints this automatically as
   "null_mean sanity"). Do not trust a permutation result that fails this check.

---

## 6. Code inventory — everything needed, all already written

| File | Status | Role |
|---|---|---|
| `code/decode_roi_erpstyle.py` | existing, unchanged | z-score + decode functions (§1) |
| `code/decode_roi_singletrial.py` | existing, unchanged | `CONTRASTS`, `validate_manifest`, `load_hdf5_matrix` |
| `searchlight_decoding/sphere_common.py` | **new, written** | sphere-center grid builder + sphere-aware GLMsingle loader |
| `searchlight_decoding/sphere_cache_subject_matrices.py` | **new, written** | Phase 1: builds the manifest + per-subject cache |
| `code/erp_permute_worker.py` | existing, unchanged | Phase B: observed + 100-shuffle-equivalent null per (subject, sphere); supports `--contrast-kind` |
| `code/erp_wave_launcher.py` | existing, unchanged | `ProcessPoolExecutor` dispatcher; `--roi-manifest` accepts the sphere manifest directly |
| `code/erp_aggregate_observed.py` | existing, unchanged | aggregates observed CSVs, t-test+FDR (diagnostic only — the permutation result in §4a is primary), `--roi-manifest`/`--contrast-kind` |
| `code/stelzer_group_resample.py` | existing, unchanged | group-resample, raw p-values + BH-FDR q-values, `--roi-manifest`/`--contrast-kind` |
| `searchlight_decoding/sphere_reconstruct_maps.py` | **new, written** | generic CSV -> NIfTI: places any per-center value column into a volume using `centers_ijk.csv` |

No further code needs to be written to run this end to end — only the parameters (§4)
need to be settled.

---

## 7. Running it (once §4's parameters are settled)

```bash
# Phase 1: cache (see §3)
python3 searchlight_decoding/sphere_cache_subject_matrices.py \
  --glmsingle-root glmsingle_fixedhrf_full_pilot --reference-subject 1 \
  --radius-mm <RADIUS_MM> --step <STEP> --min-voxels 10 \
  --out-dir searchlight_fixedhrf_full/cache --workers 4

# Phase 2: permutation wave (n_perms/n_repeats per SECTION 4 -- these are examples, not
# a recommendation to run as-is)
python3 code/erp_wave_launcher.py \
  --roi-manifest searchlight_fixedhrf_full/cache/roi_manifest.csv \
  --n-perms <N_PERMS> --n-repeats 30 --workers 20 \
  --cache-dir searchlight_fixedhrf_full/cache \
  --out-dir searchlight_fixedhrf_full/permutation

# Phase 3: aggregate observed (t-test, diagnostic) + group-resample (primary significance)
python3 code/erp_aggregate_observed.py \
  --root searchlight_fixedhrf_full/permutation \
  --roi-manifest searchlight_fixedhrf_full/cache/roi_manifest.csv

python3 code/stelzer_group_resample.py \
  --null-pools-dir searchlight_fixedhrf_full/permutation/null_pools \
  --observed-csv searchlight_fixedhrf_full/permutation/group/group_roi_stats.csv \
  --out-dir searchlight_fixedhrf_full/permutation/group_permutation \
  --roi-manifest searchlight_fixedhrf_full/cache/roi_manifest.csv \
  --n-draws 100000
```

**Everything the significance test could ever need is preserved by this pipeline as-is —
no extra work required for the "reapply FDR later without rerunning" requirement.**
`group_permutation_results.csv` already contains `p_value_raw` (the raw, uncorrected
empirical p-value per sphere per contrast) alongside `q_value_fdr`/`sig_fdr_q05` — the FDR
correction already applied. To try a different alpha or a different correction method
entirely, just recompute from the `p_value_raw` column in that CSV; the expensive part
(building the null pools) never needs to run again. `null_pools/` itself (the raw
per-subject, per-shuffle null values) is also kept on disk, so even the group-resample
step (§7 Phase 3's second command) can be redone with different settings — different
`--n-draws`, different `--seed` — without rebuilding the cache or rerunning the
permutation wave.

---

## 8. Building the requested output maps (.nii.gz)

Three deliverables, all reconstructed with `searchlight_decoding/sphere_reconstruct_maps.py` from the CSVs
Phase 2/3 already produced — no new computation, just placing values into a volume.

**(a) Subject-wise decoding accuracy maps** (one map per subject per contrast), from
`searchlight_fixedhrf_full/permutation/group/all_subject_results.csv`:
```bash
for SUB in Sub01 Sub02 ... ; do
  for CONTRAST in within_natural_pleasant_vs_neutral ... ; do
    python3 searchlight_decoding/sphere_reconstruct_maps.py \
      --input-csv searchlight_fixedhrf_full/permutation/group/all_subject_results.csv \
      --centers-ijk-csv searchlight_fixedhrf_full/cache/centers_ijk.csv \
      --reference-nifti glmsingle_fixedhrf_full_pilot/sub-01/analysis_mask.nii.gz \
      --center-col roi --value-col accuracy \
      --filter subject "$SUB" --filter contrast "$CONTRAST" \
      --out-nifti searchlight_fixedhrf_full/maps/subject/"${SUB}_${CONTRAST}_accuracy.nii.gz"
  done
done
```

**(b) Group-averaged decoding accuracy map** (one per contrast), from
`searchlight_fixedhrf_full/permutation/group/group_roi_stats.csv`'s `mean_accuracy`:
```bash
python3 searchlight_decoding/sphere_reconstruct_maps.py \
  --input-csv searchlight_fixedhrf_full/permutation/group/group_roi_stats.csv \
  --centers-ijk-csv searchlight_fixedhrf_full/cache/centers_ijk.csv \
  --reference-nifti glmsingle_fixedhrf_full_pilot/sub-01/analysis_mask.nii.gz \
  --center-col roi --value-col mean_accuracy \
  --filter contrast "$CONTRAST" \
  --out-nifti searchlight_fixedhrf_full/maps/group/"${CONTRAST}_mean_accuracy.nii.gz"
```

**(c) Decoding accuracy at group-significant centers** (one per contrast), from
`group_permutation_results.csv` — accuracy value kept only where `sig_fdr_q05` is True,
everywhere else left at background (default NaN):
```bash
python3 searchlight_decoding/sphere_reconstruct_maps.py \
  --input-csv searchlight_fixedhrf_full/permutation/group_permutation/group_permutation_results.csv \
  --centers-ijk-csv searchlight_fixedhrf_full/cache/centers_ijk.csv \
  --reference-nifti glmsingle_fixedhrf_full_pilot/sub-01/analysis_mask.nii.gz \
  --center-col roi --value-col observed_accuracy \
  --filter contrast "$CONTRAST" --filter sig_fdr_q05 True \
  --out-nifti searchlight_fixedhrf_full/maps/significant/"${CONTRAST}_sig_q05_accuracy.nii.gz"
```
This is a *convenience default* (q<0.05) — because `p_value_raw` is preserved (§7), the
exact same command with a different pre-filtered CSV (e.g. thresholded at a different
alpha, or a different correction method entirely) produces the equivalent map at that
threshold instead, without rerunning §7.

**Middle products already saved, nothing extra to do:**
- `searchlight_fixedhrf_full/cache/*.npz` — per-subject sphere-extracted betas (reusable
  for any future re-decode without touching the raw GLMsingle data again).
- `searchlight_fixedhrf_full/permutation/observed/*.csv` and `null_pools/*.csv` — every
  individual shuffle's null accuracy, per subject per sphere per contrast.
- `searchlight_fixedhrf_full/permutation/group_permutation/subject_null_pools.npz` — the
  aggregated null-pool array (`stelzer_group_resample.py` writes this as its first step).
- `searchlight_fixedhrf_full/permutation/group_permutation/group_permutation_results.csv`
  — raw p-values, null mean/SD, and every FDR-threshold variant, per sphere per contrast.

Add a `p_value_raw`-as-NIfTI map to §8 the same way as (c) if a continuous (unthresholded)
significance map is useful for visualization — same command, `--value-col p_value_raw`
(or `q_value_fdr`), no `sig_fdr_q05` filter.

---

## 9. If anything here is ambiguous

Read `FINDINGS_AND_REPRODUCTION.md` first. If something genuinely isn't answered by either
document, flag it rather than guessing — especially the §4 parameter question, which was
an explicit, unresolved judgment call at the time this document was written.
