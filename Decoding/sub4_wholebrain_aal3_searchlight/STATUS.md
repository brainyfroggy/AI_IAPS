# Status as of 2026-08-06 (work stopped by explicit user request)

**This file documents current state for handoff.** See `TASK_BRIEF.md` for
the original ask/scope. Nothing is currently running — all searchlight
processes were confirmed killed (checked via `ps aux` in WSL, no
`decode_searchlight`/`loky` processes remain) before writing this file.

## Summary

| Arm | Status |
|---|---|
| AAL3 ROI decoding | **Complete and validated** |
| Searchlight | Script built and validated; timing pilot done; radius sweep **partially run, then stopped** |
| Radius comparison deliverable | **Not started** (blocked on full sweep) |

## 1. AAL3 ROI decoding — done

- `code/prepare_aal3_atlas.py`: resamples `AAL3v1.nii.gz` (2mm) onto the
  native beta grid via nearest-neighbor (`nibabel.processing.resample_from_to`,
  order=0) — no ANTs/Docker needed, same MNI-template family as the Kastner
  atlas's `mni_res_native` handling. Output:
  `results/atlas/aal3_native_labels.nii.gz` + `provenance.json`.
  75 groups, 204,677 labeled voxels (76% of the 268,697-voxel mask). 2
  groups have zero atlas voxels in the source image (`Thalamus` generic
  placeholder ids 81/82, `CingulateAnt` generic placeholder ids 35/36) —
  both are known AAL3 quirks where the atlas replaced a generic region with
  named subdivisions that are present and decoded normally.
- `code/decode_roi_aal3_singletrial.py`: forked from
  `sub4_spatial_sensitivity_32/code/decode_roi_singletrial.py`. Same
  SVC(kernel="linear", C=1.0), 10-fold leave-one-run-out, per-fold
  train-only mean/std normalization. ROI groups loaded directly from
  `roi_labels.csv`'s `group_id`/`group_roi_name` columns (**not** name-keyed
  — an early version collided two distinct `group_id`s that share the
  display name "Cerebellum", losing one group; fixed by disambiguating with
  `group_id` suffix when names collide). 4 cross-source contrasts only, no
  smoothing (`smoothing_fwhm_mm=0`).
- **Run complete**: `results/roi_aal3/` — 71 of 75 groups decoded (4
  dropped for <10 analysis voxels: `CingulateAnt`, `Thalamus`, `ThalRe`,
  `LC` — all small/placeholder structures, not a bug). 284 subject-level
  rows (71×4), 2840 fold rows (71×4×10), validated against expected counts.
  Runtime: ~1m40s.
  - Mean accuracy overall: 0.521 (range 0.415–0.680).
  - Per-contrast mean: natural→ai pleasant 0.521, ai→natural pleasant
    0.522, natural→ai unpleasant 0.522, ai→natural unpleasant 0.519 — all
    close to chance on average, as expected for a whole-brain ROI average.
  - Top region: `Fusiform` at 0.680 (train_ai→test_natural, pleasant) and
    0.655 (train_natural→test_ai, pleasant).
  - Files: `roi_union_features.npy`, `subject_results.csv`,
    `fold_results.csv`, `roi_counts.csv`, `provenance.json`.

## 2. Searchlight — built, validated, sweep interrupted mid-run

- `code/decode_searchlight_singletrial.py`: uses
  `nilearn.decoding.SearchLight` with a custom
  `Pipeline(StandardScaler, SVC(kernel="linear", C=1.0))` estimator so
  nilearn's internal `cross_val_score` per sphere reproduces the ROI
  script's per-fold train-only normalization exactly. Custom cross-source
  10-fold splits (same logic as the ROI script) passed as `cv=<list of
  (train_idx,test_idx)>`. `--radius-voxels` is a nominal in-plane voxel
  count converted to mm (nilearn's `radius` param is in mm, world-space,
  correctly handles the anisotropic ~1.7966×1.7966×2.25mm native voxels).
- **Sanity check passed**: hand-rolled reference decode (manual
  train-fold-only normalization + SVC) at 6 random in-mask voxels, radius=3
  voxels (~5.39mm, ~89-voxel spheres), matched nilearn's
  `Pipeline`-based output to floating-point exactness (0/6 mismatches).
  Confirms the nilearn+Pipeline design is not a silent approximation.
- **Timing pilot** (radius=3 voxels, 1 contrast
  `train_natural_pleasant_vs_neutral_test_ai`, full 268,697-voxel
  whole-brain mask, `n_jobs=24`): **67.7 minutes**. Mean accuracy 0.502,
  max 0.675 — the peak lines up well with the ROI arm's Fusiform result for
  the same contrast family, a good cross-arm consistency signal.
  Output: `results/searchlight/radius_3_pilot/` (single contrast only,
  informational — superseded by the full radius_3 run below, safe to
  delete or ignore).
- **Radius sweep decision (confirmed with user 2026-08-06)**: sweep radii
  are **3, 4, 5, 6 voxels** — replacing the brief's original 2–5 suggestion
  (user explicitly chose to drop 2 and add 6). Extrapolated full-sweep cost
  at the time: ~18–30 hours of unattended compute (4 radii × 4 contrasts,
  ~70–110 min each based on subset timing).

### Sweep progress at time of stop

| Radius | Contrast | Status |
|---|---|---|
| 3 | train_natural_pleasant_vs_neutral_test_ai | **Done** (68.3 min, mean 0.5019, max 0.675) |
| 3 | train_ai_pleasant_vs_neutral_test_natural | **Done** (66.2 min, mean 0.5024, max 0.680) |
| 3 | train_natural_unpleasant_vs_neutral_test_ai | **Not run** — was in progress when stopped |
| 3 | train_ai_unpleasant_vs_neutral_test_natural | Not started |
| 4 | all 4 | Not started |
| 5 | all 4 | Not started |
| 6 | all 4 | Not started |

- `results/searchlight/radius_3/` contains only the 2 completed
  `*_accuracy.nii.gz` files above and **no `provenance.json`** (the script
  only writes it after all requested contrasts finish) — this directory is
  **incomplete**, not a finished radius_3 result. `logs/ledger.jsonl` has a
  matching `INTERRUPTED` event recording exactly this.
- Per project convention (fail-closed, namespace-per-attempt): **do not
  resume by writing into `results/searchlight/radius_3/`** — the script
  will refuse (output-exists check) and even if forced, the two existing
  maps + a differently-parameterized rerun could get confused. To resume,
  either (a) run the 2 missing contrasts into a fresh dir (e.g.
  `radius_3_remainder/`) and treat the 4 maps across both dirs as the
  complete radius_3 result, or (b) delete the partial `radius_3/` dir and
  rerun all 4 contrasts fresh into it. Given each contrast is a ~67min
  independent run, (a) avoids re-paying for the 2 already-done contrasts.

## 3. Not started

- Radius 4, 5, 6 sweep runs.
- `code/compare_searchlight_radii.py` (task 7 in the plan) — the
  radius-comparison deliverable (peak accuracy, above-chance voxel extent,
  spatial overlap between radii, cost from the ledger). Needs all 4 radii
  complete first.

## Resuming this work

1. Decide whether to resume the radius_3 remainder now or restart it
   fresh — see note above.
2. Launch remaining sweep runs one radius at a time (each uses all 24 WSL
   cores; do not run radii concurrently — confirmed via the completed
   pilot that per-contrast cost is real, not overhead-dominated).
3. Command pattern (adjust `--contrasts` for a partial/remainder run):
   ```
   python3 decode_searchlight_singletrial.py \
     --input-root /mnt/n/.../sub4_spatial_sensitivity_32/production_v2_40_attempt02/glmsingle/mni_res_native \
     --radius-voxels <3|4|5|6> \
     --n-jobs 24 \
     --ledger /mnt/n/.../sub4_wholebrain_aal3_searchlight/logs/ledger.jsonl \
     --output /mnt/n/.../sub4_wholebrain_aal3_searchlight/results/searchlight/radius_<N>
   ```
4. After all 4 radii are complete, build and run
   `compare_searchlight_radii.py` (task 7).
5. `logs/ledger.jsonl` has the full append-only history of every job
   started/ended/interrupted this session — read it for exact timings
   before re-estimating anything.
