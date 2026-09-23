# AI-IAPS GLMsingle HRF/Denoise/Ridge Investigation — Findings & Reproduction

**Project:** AI-IAPS — fMRI decoding of emotional valence (Pleasant/Neutral/Unpleasant) from
Natural vs. AI-generated images, 30-subject cohort, Kastner/Wang visual ROI atlas (17 ROIs).
**Scope of this document:** why GLMsingle initially underperformed SPM in decoding, the full
factorial investigation that explains and fixes the gap, and the finalized production
configuration + its permutation-test-based significance results. Written so another
researcher can reproduce every step on their own machine.

---

## 1. TL;DR — the answer

GLMsingle was never fundamentally worse than SPM. The default production configuration
(`wantlibrary=1`, i.e. per-voxel HRF fitting) plus GLMdenoise was suppressing accuracy;
switching to a **fixed (assumed) canonical HRF** (`wantlibrary=0`) and keeping
**GLMdenoise + fracridge (ridge regression)** on recovers — and exceeds — SPM's performance.

**Adopted production configuration:**

```text
wantlibrary   = 0   (fixed/assumed HRF, no per-voxel HRF library search)
wantglmdenoise= 1   (GLMdenoise on)
wantfracridge = 1   (ridge/fracridge on)
```

Output directory: `glmsingle_fixedhrf_full_pilot/` (n=30). Decoding is **ERP-style
(trial-averaged) by default going forward**, `n_repeats=100`, significance via the
**Stelzer et al. (2013) permutation test + Benjamini-Hochberg FDR (q<0.05)**, not the
t-test that was used earlier in the investigation.

**Final significance result (n=30, ERP-style, 100 repeats, permutation+FDR):**

| Metric | Value |
|---|---|
| Significant cells | **122/136** (FDR q<0.05, permutation) |
| Significant cells (t-test+FDR, for comparison) | 117/136 |
| Within-source grand mean accuracy | 64.98% |
| Cross-source grand mean accuracy | 61.30% |
| SPM reference (t-test+FDR, ERP-style, n=29) | 113/136, within=63.19%, cross=59.11% |

---

## 2. Why this investigation happened

SPM (canonical HRF, LS-A GLM) was outperforming GLMsingle (per-voxel-fitted HRF,
`wantlibrary=1`, GLMdenoise + ridge = production "Type-D") in ERP-style (trial-averaged)
decoding, especially in one cell (`within_natural_pleasant_vs_neutral`, ~11pt gap). Several
hypotheses were tested and ruled out before finding the real cause:

| Hypothesis | Test | Result |
|---|---|---|
| Ridge regularization strength/choice | swept fracridge fractions | ruled out — not the driver |
| Serial (AR1) correlation in the GLM residuals | Prais-Winsten whitening, custom OLS reimplementation | ruled out — rho≈0, +0.07pt negligible |
| Post-hoc smoothing kernel width | swept 8/12/16/20mm | ruled out — flat/non-monotonic |
| Amplitude compression (mass-univariate) | Cohen's d per condition | ruled out — wrong tool for a multivariate pattern question (r=0.008 with actual decoding gaps) |
| **Fixed vs. per-voxel-fitted HRF** | reran GLMsingle with `wantlibrary=0` | **confirmed as the answer** |

Rerunning with `wantlibrary=0` (fixed HRF) instead of `wantlibrary=1` (per-voxel HRF
library search) recovered SPM-level (and better) performance. This led to a full factorial
sweep to characterize *why*, and to isolate GLMdenoise's and ridge's individual
contributions.

---

## 3. The full factorial: 2 HRF settings × 4 regularization configs × 2 decoding styles

All cells: n=30 subjects, 17 Kastner/Wang ROIs, 8 contrasts (4 within-source + 4
cross-source) = 136 cells/config. Significance = one-sample t-test vs. chance (50%) +
Benjamini-Hochberg FDR within each contrast across its 17 ROIs, q<0.05. (This table
predates the permutation-test finalization in §5 — see that section for the final
significance numbers on the adopted config.)

### Fixed HRF (`wantlibrary=0`)

| Config | ERP overall/within/cross % | ERP sig/136 | Non-ERP overall/within/cross % | Non-ERP sig/136 |
|---|---|---|---|---|
| No denoise, no ridge | 59.60 / 63.20 / 56.00 | 102 | 51.77 / 51.22 / 52.32 | 74 |
| Denoise only | 59.70 / 63.19 / 56.21 | 99 | 51.78 / 51.20 / 52.35 | 76 |
| Ridge only | 63.04 / 65.01 / 61.06 | 113 | 53.96 / 54.63 / 53.28 | 127 |
| **Both (adopted: `fixedhrf_full`)** | **63.15 / 65.04 / 61.25** | **116** | 53.99 / 54.66 / 53.31 | 123 |

### Per-voxel HRF (`wantlibrary=1`, old production default)

| Config | ERP overall/within/cross % | ERP sig/136 | Non-ERP overall/within/cross % | Non-ERP sig/136 |
|---|---|---|---|---|
| No denoise, no ridge (TypeB) | 58.03 / 58.83 / 57.23 | 76 | 51.83 / 51.56 / 52.09 | 69 |
| Denoise only (TypeC) | 57.94 / 59.18 / 56.69 | 83 | 51.78 / 51.55 / 52.00 | 62 |
| Ridge only | 58.12 / 59.02 / 57.21 | 81 | 52.48 / 53.16 / 51.80 | 87 |
| Both (TypeD, old production) | 58.30 / 59.49 / 57.11 | 77 | 52.48 / 53.13 / 51.82 | 87 |

### SPM reference

| | overall/within/cross % | sig/136 |
|---|---|---|
| SPM, ERP-style (n=29) | 61.15 / 63.19 / 59.11 | 113 |
| SPM, single-trial, Kebo-CV (n=28, **different CV scheme — caveat**) | 52.09 / 51.06 / 53.11 | 103 |

### Interpretation

- **GLMdenoise is inert everywhere** — never a reliable gain in any of the 8
  HRF×style combinations, and mildly counterproductive in one (per-voxel-HRF, non-ERP).
- **Ridge is the dominant lever** in 3 of 4 HRF×style combinations. The one exception is
  per-voxel-HRF + ERP-style, where all four regularization configs cluster together
  (76–83/136) — per-voxel HRF fitting and ridge are both cross-validated, voxel-adaptive
  noise-reduction strategies that overlap; ERP's trial-averaging already suppresses most
  noise before ridge gets a chance to add anything, whereas single-trial decoding never
  gets that averaging boost, so ridge keeps helping there even under per-voxel HRF.
- Both fixed-HRF+ridge configs (ridge-only, both) beat SPM's ERP number (113) and clearly
  beat SPM's single-trial number (103, CV-scheme caveat noted above).

---

## 4. Two GLMsingle vendor bugs found and fixed

Vendor path: `AI_IAPS/.codex_work/pilot_pipeline/vendor/GLMsingle/glmsingle/glmsingle.py`

1. **`HRFindex = np.ones(xyz)` crash** (undefined `xyz`). Triggered when
   `wantfileoutputs[1]==0 AND wantmemoryoutputs[1]==0 AND hrflibrary.shape[1]==1`
   (library size 1 happens when `wantlibrary=0`). **No vendor edit needed** — worked
   around by always requesting the TYPEB output slot (`wantfileoutputs[1]=1`).

2. **`pcregressors[run_i]` IndexError** when `wantglmdenoise=0` but `wantfracridge=1`
   (ridge-only, no denoise). Root cause: `pcregressors = []` when denoise is off, but the
   ridge-fitting path unconditionally indexes it per run. **Fixed with a direct vendor
   edit**, `glmsingle.py:1267`:
   ```python
   pcregressors = [None] * numruns   # was: pcregressors = []
   ```
   Safe because `pcnum=0` in this branch means `_merge_extra_regressors`'s `n_pc<=0` path
   never reads the placeholder values — only their indexability is required. A benign
   side-effect warning ("Could not save pcregressors: Object dtype... has no native HDF5
   equivalent") appears in ridge-only runs afterward; it does not affect `betasmd`
   correctness.

**Anyone reproducing this on a fresh GLMsingle checkout must apply patch #2 manually**
before running `wantlibrary=0, wantglmdenoise=0, wantfracridge=1` (ridge-only, no
denoise) configurations. The other 3 fixed-HRF configs (no/no, denoise-only, both) do not
need it.

---

## 5. Finalization: repeat count, ERP-as-default, and the permutation test

Three decisions were made to finalize the pipeline:

1. **CV repeats = 100** (was 30) for ERP-style decoding, for both the observed statistic
   and every permutation-null shuffle (must match — see below).
2. **ERP-style decoding is the default going forward**, unless explicitly told otherwise.
   Single-trial/random-10-fold decoding is no longer run by default.
3. **Significance = Stelzer et al. (2013) permutation test + BH-FDR**, not the t-test,
   because ERP-style accuracies are coarse-grained (2 or 8 averaged patterns per fold) and
   far from normal across subjects (SD≈0.15) — the t-test's normality assumption is poorly
   met and loses power. The permutation test makes no such assumption and found 122/136 vs.
   the t-test's 117/136 on the same data.

### 5a. Why repeat-count matching is mandatory

The permutation p-value is `P(null_draw >= observed)`. If the null used *more* averaging
(repeats) than the observed statistic, its draws would be tighter and its upper tail too
thin, biasing p-values **small** (anti-conservative — false positives). Matching repeat
counts exactly keeps the null and the observed statistic on the same variance footing. A
null with *fewer* repeats than observed is merely conservative (safe, not required).

### 5b. The permutation test, precisely (Stelzer et al. 2013 two-level design)

**Level 1 — per-subject null pools** (`erp_permute_worker.py`, dispatched in parallel by
`erp_wave_launcher.py`):

1. Load that subject's cached, 8mm-smoothed, ROI-extracted single-trial betas.
2. Compute the voxel-source z-score **once**: pattern-z-score each trial across voxels,
   then voxel-z-score across the source's pooled 300 trials. This step never reads valence
   labels, so it is *invariant* under label permutation — computed once and reused for the
   observed decode and all 100 shuffles, guaranteeing bit-identical preprocessing between
   observed and null (what makes the p-value interpretable at face value).
3. For each of the 8 contrasts: compute the **observed** accuracy at `n_repeats=100`
   (true labels), then run **100 shuffles**, each re-deriving the full averaging pipeline
   from a freshly relabeled trial pool (not from the already-averaged patterns) at the
   same `n_repeats=100`:
   - Within-source: pool the condition's 200 trials (100 positive + 100 neutral), randomly
     repartition into two new 100-trial groups.
   - Cross-source: shuffle the train-source and test-source 200-trial pools
     **independently** of each other.

**Level 2 — group resample** (`stelzer_group_resample.py`): for each (ROI, contrast)
cell, draw one value at random from each of the 30 subjects' 100-value null pools,
average across subjects, repeat **100,000 times** (`--n-draws`, default), to build the
group-level null distribution. Empirical p-value:
`p = (1 + count(null_draws >= observed_group_mean)) / (n_draws + 1)`.
BH-FDR is then applied within each contrast, across its 17 ROIs, same as the t-test
pipeline; significant at q<0.05.

**Null calibration check** (always verify this after a permutation run): max
`|null_mean − 0.5|` across all 136 cells should be small. Achieved: **0.673 percentage
points** — consistent with the earlier pilot project's 0.66pp, confirming correct
implementation.

### 5c. Compute cost note

A literal reading of "10,000 permutations" at the first level (with matched 100 repeats)
would cost **~44 days at 20 workers** — impractical. The adopted setting keeps the
first-level shuffle count at **100** (unchanged from the original pilot project) and
spends compute on `n_repeats=100` instead; the group-resample draw count stayed at the
default **100,000** (not bumped to 1,000,000 — it's cheap either way, but 100,000 is
already ample resolution and was the explicit final call). Measured full-wave runtime:
**~9.8 hours at 20 workers**, 510/510 (subject×ROI) units, 0 failures.

---

## 6. Reproduction — step by step

### 6.0 Environment

- WSL Ubuntu-22.04 (all steps below run there; from Windows PowerShell:
  `wsl -d Ubuntu-22.04 -- bash -c "<command>"`).
- Python venv used throughout this project:
  `/home/yujun/.cache/ai_iaps_pilot_venv/bin/python3` — needs numpy, pandas, scipy,
  scikit-learn, nibabel, nilearn, h5py, threadpoolctl, and a **vendored, patched**
  GLMsingle (see §4) importable from the path the scripts add via `sys.path.insert`.
- Repo root: `new_pipeline/` (all commands below assume this as `cwd`; paths shown are
  the WSL mount `/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline`).
- Atlas: Kastner/Wang 17-ROI atlas at
  `AI_IAPS/Decoding/full_cohort_raw_pipeline_28/resources/kastner/{kastner.nii.gz,kastner.nii.txt}`.
- fMRIPrep derivatives root and per-subject BIDS root are **site-specific** — pass your
  own paths to `--fmriprep-derivatives-root` / `--per-subject-bids-root` below.
- Subject cohort (n=30, GLMsingle list — note: excludes Sub3/Sub8/Sub10, which SPM's
  cohort includes):
  ```text
  1 2 4 5 6 7 9 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 33 34
  ```

### 6.1 Rerun GLMsingle at the adopted configuration (if not already done)

```bash
python3 code/run_glmsingle_fixedhrf_wave.py \
  --subjects 1 2 4 5 6 7 9 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 33 34 \
  --fmriprep-derivatives-root <YOUR_FMRIPREP_DERIVATIVES_ROOT> \
  --per-subject-bids-root <YOUR_PER_SUBJECT_BIDS_ROOT> \
  --branch mni_res_native \
  --output-root glmsingle_fixedhrf_full_pilot \
  --want-glmdenoise 1 --want-fracridge 1 --want-library 0 \
  --workers 5
```
Produces `glmsingle_fixedhrf_full_pilot/sub-XX/glmsingle/TYPED_FITHRF_GLMDENOISE_RR.hdf5`
per subject (script skips subjects whose output already exists).

### 6.2 Observed ERP-style decode at 100 repeats

```bash
python3 code/decode_erpstyle_glmsingle_variant.py \
  --hdf5-root glmsingle_fixedhrf_full_pilot \
  --hdf5-name TYPED_FITHRF_GLMDENOISE_RR.hdf5 \
  --output erp_fixedhrf_full_100rep_n30 \
  --n-repeats 100
```
Writes per-subject CSVs plus `erp_fixedhrf_full_100rep_n30/group/group_roi_stats.csv`
(t-test+FDR, for comparison against the permutation result).

### 6.3 Rebuild the Stelzer permutation cache from this data

```bash
python3 code/stelzer_cache_subject_matrices.py \
  --glmsingle-root glmsingle_fixedhrf_full_pilot \
  --out-dir stelzer_permutation_fixedhrf_full/cache \
  --workers 4
```
(Memory-bound — keep `--workers` modest, ~4. Rebuild this whenever the underlying
GLMsingle config changes; a cache built from a different config is silently stale.)

### 6.4 Run the full permutation wave

```bash
python3 code/erp_wave_launcher.py \
  --workers 20 --n-perms 100 --n-repeats 100 \
  --cache-dir stelzer_permutation_fixedhrf_full/cache \
  --out-dir erp_permutation_fixedhrf_full
```
510 (subject × ROI) units, skip-if-exists. Budget **~4-10 hours** depending on hardware
(measured 9.8h at 20 workers on the original machine; a single-unit timing pilot — 1
subject × 1 ROI, `--workers 1` — is recommended before committing to the full wave on a
new machine, since per-ROI voxel count varies runtime substantially).

### 6.5 Aggregate observed stats + group-resample + FDR

```bash
python3 code/erp_aggregate_observed.py --root erp_permutation_fixedhrf_full

python3 code/stelzer_group_resample.py \
  --null-pools-dir erp_permutation_fixedhrf_full/null_pools \
  --observed-csv erp_permutation_fixedhrf_full/group/group_roi_stats.csv \
  --out-dir erp_permutation_fixedhrf_full/group_permutation
```
Final numbers land in
`erp_permutation_fixedhrf_full/group_permutation/group_permutation_results.csv`
(columns include `sig_fdr_q05`) — this is the file that reproduced 122/136 above. Check
the printed "null_mean sanity" line: it should be a small fraction of a percentage point.

### 6.6 Charts (Bo et al.-style within/cross bar charts)

```bash
python3 code/plot_bo_style_charts.py \
  --group-csv erp_permutation_fixedhrf_full/group/group_roi_stats.csv \
  --sig-csv erp_permutation_fixedhrf_full/group_permutation/group_permutation_results.csv \
  --sig-col sig_fdr_q05 \
  --out-dir erp_permutation_fixedhrf_full/group \
  --out-prefix erp_fixedhrf_full_permutation_n30 \
  --title "GLMsingle Fixed HRF, GLMdenoise+Ridge, ERP-style, n=30, 100 repeats" \
  --y-low 0.40 --y-high 0.90 --step 0.05 \
  --method-note "Significance: Stelzer et al. 2013 permutation test (100 within-subject shuffles x 100,000 group-level draws) + Benjamini-Hochberg FDR across 17 ROIs per contrast, q<0.05. Not a t-test."
```
Writes `within_erp_fixedhrf_full_permutation_n30.png` and
`cross_erp_fixedhrf_full_permutation_n30.png`.

---

## 7. Output directory naming convention (GLMsingle reruns)

| Directory | `wantlibrary` | `wantglmdenoise` | `wantfracridge` |
|---|---|---|---|
| `glmsingle_fixedhrf_pilot/` | 0 | 0 | 0 |
| `glmsingle_denoiseonly_pilot/` | 0 | 1 | 0 |
| `glmsingle_ridgeonly_pilot/` | 0 | 0 | 1 |
| **`glmsingle_fixedhrf_full_pilot/` (adopted)** | **0** | **1** | **1** |
| `glmsingle_denoiseonly_lib1_pilot/` | 1 | 1 | 0 |
| `glmsingle_ridgeonly_lib1_pilot/` | 1 | 0 | 1 |
| `glmsingle/` (old production, TypeB/TypeD) | 1 | 0/1 | 0/1 |

All at n=30 subjects, 8mm smoothing, `mni_res_native` branch, Kastner/Wang 17-ROI atlas.

---

## 8. Known open items (not resolved by this investigation)

- **Sub30 motion exclusion** — never decided whether to exclude Sub30 or censor
  runs 8–10 for motion; all reported numbers include Sub30 at n=30 with this caveat open.
- **Whitening leakage fix** — a leak-free preprocessing fix was validated during the
  hypothesis-testing phase but never wired into the production pipeline.
- **AAL3 whole-brain permutation** — deferred by choice; the same
  `stelzer_cache_subject_matrices.py` → `erp_permute_worker.py`/`erp_wave_launcher.py` →
  `stelzer_group_resample.py` machinery is reusable for a 154-region whole-brain map by
  swapping the ROI list, no new infrastructure needed.
- **SPM single-trial comparison uses a different CV scheme** ("Kebo's way", not the exact
  Bo et al. random-10-fold scheme used for GLMsingle) — the §3/§5 SPM non-ERP numbers are
  directionally informative but not a strictly matched comparison.
- **Non-ERP (single-trial) results were never re-run** at the finalized `n_repeats`/
  permutation-test settings, since ERP-style is now the default; the §3 non-ERP numbers
  reflect the earlier t-test+FDR methodology only.
