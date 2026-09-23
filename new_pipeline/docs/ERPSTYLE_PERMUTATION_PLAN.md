# ERP-style (trial-averaged) decoding + Stelzer/Bo permutation test — execution plan

**Created:** 2026-08-24
**Scope:** ERP-style averaged-trial decoding on GLMsingle Type-D betas, with a faithful
Stelzer et al. 2013 / Bo et al. 2021 permutation test for significance.
**Status:** ✅ **COMPLETE (2026-08-24)** — all phases executed, 510/510 units, 0 failures.
Results in `erp_permutation/`. See §8 for outcomes.

---

## 1. Important: the observed decode already exists

`code/decode_roi_erpstyle.py` already applies ERP-style averaging to the **GLMsingle**
betas (run 2026-08-18 → `roi_decoding_erpstyle/`). It is a verbatim reimplementation of
the lab's original `decoding_multisub_parallel_avg_voxel_trial_source_zscore_AAL3_all.ipynb`
scheme, with only the beta source swapped to GLMsingle + 8 mm smoothing.

So this task is **not** "build ERP-style decoding" — it is:
1. Close the n=29 → n=30 gap (Sub34 was onboarded 08-19, after that run).
2. Add the permutation test, which the ERP arm has never had (it currently reports a
   plain one-sample t-test + FDR).

### What "ERP-style" means here, precisely

| Step | Detail |
|---|---|
| Voxel-source z-score | Once per subject/ROI/source: pattern z-score each trial across voxels, then z-score each voxel across that source's pooled **300** trials (pleasant+neutral+unpleasant). Uses **no valence labels**. |
| Within-source | `KFold(4, shuffle=True)` × **20 repeats**. Inside each fold: pattern z-score again, per-voxel z-score fit on train only. Train trials → **3** random-chunk averaged patterns per condition; held-out test trials → **1** averaged pattern per condition. SVC fits 6 patterns, scores **2**. |
| Cross-source | No held-out fold (the other source *is* the test set). Per repeat, both sources' 100+100 trials → **4** random chunks each, averaged. SVC fits 8 train-source patterns, scores **8** test-source patterns. |
| Classifier | `SVC(kernel="linear", C=1.0)` — same as everywhere else in this project. |

Note the cross-source arm is **already** free of the fold-leakage flaw fixed on 08-21:
it trains on all 200 source-A trials and tests on all 200 source-B trials; the chunking
is only for averaging, not cross-validation. No correction needed there.

### Why it matters — ERP averaging buys a lot of accuracy

| Contrast | ERP-style | Single-trial | Gain |
|---|---|---|---|
| within_ai_pleasant | 63.4% | 54.4% | **+9.0 pts** |
| within_natural_pleasant | 61.2% | 53.9% | +7.3 |
| train_natural_pleasant→ai | 60.5% | 53.0% | +7.5 |
| train_ai_pleasant→natural | 59.8% | 52.8% | +7.0 |
| within_ai_unpleasant | 56.9% | 52.1% | +4.8 |
| within_natural_unpleasant | 55.9% | 52.1% | +3.8 |
| train_natural_unpleasant→ai | 54.3% | 50.8% | +3.5 |
| train_ai_unpleasant→natural | 54.4% | 50.7% | +3.7 |

The pleasant > unpleasant asymmetry seen in every other analysis holds here too.

---

## 2. Verified facts (probed, not assumed)

- Existing ERP run: **n=29** (missing Sub34), Kastner 17 ROIs, all 8 contrasts, 8 mm,
  78/136 cells significant by t-test+FDR.
- **The Stelzer cache is directly reusable**: `stelzer_permutation/cache/` holds all 30
  subjects at Kastner/8 mm — `data (600, ~8k)`, `source` (300 natural / 300 ai),
  `valence` (200 each), and 17 ROI position arrays. Exactly what the ERP permutation
  needs. **No re-caching, no re-smoothing.**
- Timing anchor: the 29-subject observed wave took ~8 min wall at 3 workers
  (~6,800 SVM fits/subject; the fits are tiny — 6–8 training patterns each).
- **Per-subject variance is 3.7× larger than single-trial**: median across-subject SD
  0.148 (ERP) vs 0.040 (single-trial). This is the direct consequence of scoring only
  2 (within) or 8 (cross) averaged patterns instead of 200 trials — and it drives the
  key decision in §6 Q2.

---

## 3. Permutation design — the two things that must be right

### 3a. Shuffle at the single-trial level, *then* re-average

The label shuffle must happen on the **600 single trials**, with chunking and averaging
redone from the shuffled labels. Shuffling the already-averaged patterns instead would
leave the averaging step computed from *true* labels and produce a null that is not a
null at all. For cross-source, the train-source and test-source pools are shuffled
independently (as in `stelzer_permute_worker.py`'s corrected `permute_cross`).

### 3b. The voxel-source z-score is invariant under the shuffle — exploit it

`compute_voxel_source_zscore` pools all 300 trials of a source and never touches valence
labels. Permuting valence therefore cannot change it. Two consequences:

- **Compute it once per (subject, ROI) and reuse across all 100 shuffles** — a large
  saving, since it is the only step that touches the full 300-trial matrix.
- It also resolves a standing methodological worry: that z-score pools across the
  train/test split (transductive). Because observed and null share the *identical*
  preprocessing, the permutation test remains valid — the null absorbs whatever
  advantage that pooling confers. Worth stating explicitly in the writeup rather than
  leaving it as an unaddressed caveat.

The per-fold `preprocess_voxel_trial_inside_fold` (pattern + trial z-score fit on train
only) *does* depend on which trials carry which label, so it is recomputed inside every
shuffle, as it must be.

### 3c. Group stage — unchanged from the Kastner work

Draw one null value from each of the 30 subjects' 100-value pools, average across
subjects, repeat **100,000** times. Report raw empirical p (+1 corrected), BH-FDR across
the 17 ROIs within each contrast, and store boolean flags at 0.05 / 0.01 / 0.001 for both
raw and FDR so the threshold stays selectable afterwards. Reuses
`stelzer_group_resample.py` essentially unchanged.

---

## 4. Phases

| Phase | Work | Estimate |
|---|---|---|
| 0 | Re-run **all 30** subjects through `decode_roi_erpstyle.py` at `--n-repeats 30` into a fresh `roi_decoding_erpstyle_30rep/` tree (closes the Sub34 gap and matches the null's repeat count in one pass); aggregate group stats | ~12 min |
| 1 | New `erp_permute_worker.py`: per (subject, ROI), 100 shuffles × 8 contrasts × 30 repeats, reading the existing Stelzer cache. Pilot 1 subject × 2 ROIs, check null mean ≈ 50% | ~15 min |
| 2 | Full permutation wave, 510 (subject, ROI) units, 20 workers, skip-if-exists | **~3–4 h at 20 workers** |
| 3 | Group resample (100k draws) + FDR via `stelzer_group_resample.py` | ~3 min |
| 4 | Bar charts with permutation-based stars (reuse `plot_bo_style_charts.py`), plus a t-test-vs-permutation comparison table | ~10 min |

Phase 2 sizing: 30 repeats × 100 shuffles per cell →
within `68 cells × 100 × 30 × 4 folds = 816k fits`, cross `68 × 100 × 30 = 204k fits`
≈ **1.02M fits/subject**, ~30.6M across the cohort. At the measured ~0.0074 s/fit that is
~63 h serial → **~3–4 h at 20 workers**. Phase 1's pilot exists to confirm this before
committing.

---

## 5. Methodological caveats to state

1. **Coarse accuracy granularity.** Within-source scores only 2 patterns per fold, so a
   single fold's accuracy is 0, 0.5, or 1.0. Even averaged over folds/repeats this is a
   much lumpier statistic than the single-trial analysis, and the null will be visibly
   discrete. Not a bug — an inherent property of the design — but it should be shown
   (plot a null histogram for one representative cell).
2. **Averaging inflates accuracy, not evidence.** The +3.5 to +9.0 pt gain over
   single-trial reflects SNR gain from averaging, not more information; the effective
   N per subject drops from 200 trials to 8 patterns. The permutation test is what makes
   the two arms' significance claims comparable — the raw accuracies are not.
3. **Group-mean inference**, same Allefeld/Görgen/Haynes 2016 caveat as everywhere else.
4. n=30 including Sub30, whose motion exclusion decision is still open (§7).

---

## 6. Decisions — resolved 2026-08-24

**Q1. ROI set → Kastner 17 only.** Matches the existing ERP run and gives a clean
ERP-vs-single-trial comparison on identical ROIs. The Stelzer cache is directly reusable.
AAL3 whole-brain ERP is not part of this pass.

**Q2. Repeat count → 30, for BOTH observed and null.** The observed decode is re-run at
30 repeats so the two match. This matters: the permutation p-value is
`P(null_draw ≥ observed)`, so if the null averaged *more* repeats than the observed
statistic, its draws would be tighter, its upper tail too thin, and p-values biased
**small** (anti-conservative, inflated false positives). Matching the repeat count keeps
the null and the observed statistic on the same variance footing, which is what makes the
empirical p-value interpretable at face value. Erring the other way (a 1-pass null) is
merely conservative; erring this way is not, so the match is not optional bookkeeping.

**Q3. Contrast scope → all 8** (4 within-source + 4 cross-source), matching the existing
ERP run. The within-source arm is where ERP's accuracy gains are largest (+9 pts at
`within_ai_pleasant`).

The 20-repeat results in `roi_decoding_erpstyle/` are left untouched on disk; the
30-repeat work lands in a new tree.

---

## 8. Outcomes (2026-08-24)

Ran as one fused pass: `erp_permute_worker.py` computes the observed statistic **and** the
100-shuffle null per (subject, ROI), reusing the single shuffle-invariant voxel-source
z-score for all 101 evaluations. 510 units, 20 workers, ~75 min, 0 failures.

| Metric | Value |
|---|---|
| Observed, within-source grand mean | **59.49%** |
| Observed, cross-source grand mean | **57.11%** |
| Significant, permutation + BH-FDR q<0.05 | **93/136** |
| Significant, t-test + BH-FDR q<0.05 | 77/136 |
| Null calibration | max \|null_mean − 0.5\| = **0.66 pp** |
| 30-rep vs existing 20-rep observed (Sub01/V1v) | agrees within ≤0.019 |

**The permutation test is more powerful than the t-test here — 93 vs 77 significant cells,
and all 16 disagreements run the same direction** (permutation significant, t-test not:
V2d/V2v/LO2/V1v/V3a/V3b/IPS/VO2/PHC1 across four contrasts, mostly the unpleasant
cross-source ones). This is the expected consequence of §5 caveat 1: ERP accuracies are
coarse-grained and far from normal across subjects (SD ≈ 0.15), so the t-test's normality
assumption is poorly met and it loses power. The nonparametric test makes no such
assumption. This is a substantive argument for reporting the permutation result as primary
for the ERP arm.

Outputs:
- `erp_permutation/observed/`, `erp_permutation/null_pools/` — 510 unit files each
- `erp_permutation/group/group_roi_stats.csv` — observed + t-test/FDR
- `erp_permutation/group_permutation/group_permutation_results.csv` — permutation p/q,
  thresholds at .05/.01/.001 for raw and FDR
- `erp_permutation/group/charts/{within,cross}_erpstyle_kastner_n30_30rep_perm.png`

The 20-repeat `roi_decoding_erpstyle/` tree (n=29) is untouched.

---

## 7. Still open, unrelated

- **Sub30 motion** — exclude / censor runs 8–10 / keep with caveat. Never decided.
- **Whitening leakage** — leak-free fix validated, never wired into production.
- **AAL3 Phase 5** — Stelzer permutation for the whole-brain map, deferred by choice.
