# AI-IAPS Decoding Project Knowledge

Generated from the current Codex project work on 2026-07-10.

This file summarizes the current working state of the AI-IAPS fMRI decoding analyses: data locations, notebooks, saved results, analysis logic, key findings, and the next planned feature-selection/single-trial decoding direction.

## Scope

This summary is based on the AI-IAPS decoding work available in this project workspace and the analysis decisions/results developed in recent Codex sessions. It is intended as the handoff document for continuing work in a new chat/thread.

Main project root:

```text
N:\Experimental_Data\yujunchen\projects\AI_IAPS
```

Main decoding folder:

```text
N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding
```

Main notebook:

```text
N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\decoding_multisub_parallel_avg_voxel_trial_source_zscore_AAL3_all.ipynb
```

Notebook timestamp:

```text
Last modified: 2026-07-07 01:39:29
Size: 3,242,304 bytes
```

## Data Inputs

Single-trial beta directory:

```text
N:\Experimental_Data\yujunchen\projects\AI_IAPS\GLM_singletrial\betas
```

Beta group/label file:

```text
N:\Experimental_Data\yujunchen\projects\AI_IAPS\GLM_singletrial\beta_groups.csv
```

Onset/log directory:

```text
N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\DataRecording
```

Brain mask:

```text
N:\Experimental_Data\yujunchen\projects\data\masks\MNI152_T1_2mm_brain_mask.nii.gz
```

Kastner/Wang atlas:

```text
C:\MRIcroGL\Resources\atlas\kastner.nii.gz
C:\MRIcroGL\Resources\atlas\kastner.nii.txt
```

AAL3 atlas:

```text
N:\Experimental_Data\yujunchen\projects\data\masks\AAL3\AAL3v1.nii.gz
C:\MRIcroGL\Resources\atlas\AAL3v1.nii.txt
```

Subject list in the notebook:

```text
Sub1, Sub2, Sub3, Sub4, Sub5, Sub6, Sub7, Sub8, Sub9,
Sub11, Sub12, Sub13, Sub14, Sub15, Sub16, Sub17, Sub18,
Sub19, Sub20, Sub21, Sub22, Sub23, Sub24, Sub25, Sub26,
Sub27, Sub28, Sub29, Sub30, Sub31
```

Unless an analysis explicitly states otherwise, the default cohort is 28 subjects.
Exclude `Sub3` (Emily), `Sub8` (earrings), and `Sub10` (not usable). `Sub10`
was already absent from the 30-subject beta dataset, so excluding `Sub3` and
`Sub8` from that dataset leaves 28 subjects. The machine-readable source of
truth is `Decoding/cohort_config.json`.

Runs:

```text
Run01 through Run10
```

Categories:

```text
Natural: pleasant, neutral, unpleasant
AI: pleasantAI, neutralAI, unpleasantAI
```

## Main Decoding Code

Primary notebook:

```text
Decoding\decoding_multisub_parallel_avg_voxel_trial_source_zscore_AAL3_all.ipynb
```

Core mode:

```text
AVG_MODE = 'voxel_trial_source_zscore'
```

Primary decoding parameters:

```text
N_REPEATS = 20
N_FOLDS = 4
N_AVG_GROUPS = 3
N_JOBS = 10
```

The main analysis uses Avg/random ERP-style decoding:

1. Load 600 single-trial beta images per subject.
2. Extract ROI voxel patterns.
3. Remove invalid voxels with NaNs.
4. Apply voxel/source z-scoring.
5. In each cross-validation fold, apply trial/voxel preprocessing inside the fold.
6. Average same-category training trials into `N_AVG_GROUPS` chunks.
7. Average each test condition into one test pattern.
8. Train a linear SVM.
9. Repeat over folds and repeats.

Important point: this is not raw single-trial decoding. It is averaged-trial decoding with repeated random/fold splits.

## Saved Main Decoding Results

Main saved decoding pickle:

```text
N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\results\decoding_multisub_avg_voxel_trial_source_zscore_aal3_all.pkl
```

Timestamp:

```text
Last modified: 2026-06-23 03:15:53
Size: 953,083 bytes
```

This file stores the original within-source and cross-source decoding accuracies:

```text
results_within_avg
results_cross_avg
subs
roi_names
kastner_roi_names
aal3_roi_names
AVG_MODE metadata
```

Main weight-map cache:

```text
N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\results\weight_map_cache_avg_voxel_trial_source_zscore_superset_rois.pkl
```

Timestamp:

```text
Last modified: 2026-06-29 00:30:48
Size: 141,986,928 bytes
```

This cache stores subject-level SVM weights and Haufe-transformed patterns for the superset ROI list, generated during the main decoding loop so the data do not need to be reloaded/redecoded for weight-map analyses.

## ROI Sets

The notebook combines:

1. Original Kastner/Wang decoding ROIs.
2. All AAL3 ROIs.
3. Superset ROIs from the ROI tables.

Important Kastner/Wang plotting set:

```text
V1v, V1d, V2v, V2d, V3v, V3d, hV4, V3a, V3b,
LO1, LO2, VO1, VO2, IPS, PHC1, PHC2, MST, hMT, SPL1, FEF
```

For some summary plots, `FEF` and `MST` were removed from plotting, depending on the specific figure request.

For the superset ROI cache, AAL3 left/right ROIs were collapsed into bilateral base ROIs where needed, e.g.:

```text
Amygdala_L + Amygdala_R -> Amygdala
Fusiform_L + Fusiform_R -> Fusiform
```

## Main Plotting Sections

Important notebook sections:

```text
## Plot Results - Kastner/Wang and AAL3 Separately
## AAL3 3D Brain Heatmaps
## ROI Linear SVM Weight Maps
## Amygdala Haufe Weight-Map Matrix
## Subject vs Group Haufe Correlation Check
## Superset ROI Mean Subject-Level Haufe Correlations
## Fusiform Haufe Seed And Cache Diagnostic
## Fusiform Original Decoding Performance
## Fusiform Cross-Decoding vs Haufe Similarity Discrepancy Audit
## Evidence For Shared Subspace But Partly Different Fusiform Maps
## Fusiform Top-50 Percent Haufe Weight-Map Correlation
```

## Haufe Weight-Map Logic

Raw linear SVM weights are useful for prediction but are not directly interpretable as activation patterns because they are affected by feature covariance.

Haufe transformation converts the classifier model into an activation-pattern-like vector:

```text
Haufe pattern = cov(X_train, decision_function(X_train)) / var(decision_function(X_train))
```

In the notebook code:

```python
Xc = X_train.astype(np.float64) - np.mean(X_train, axis=0, keepdims=True)
dc = decision.astype(np.float64) - np.mean(decision)
haufe = Xc.T @ dc / np.dot(dc, dc)
```

For matrix plots:

1. The one-dimensional voxel vector is reshaped into a near-square 2D matrix.
2. Each cell is one voxel.
3. The cell color is the z-scored Haufe or SVM weight.
4. The matrix is only a display form. It is not a voxel-by-voxel multiplication matrix.

## Weight-Map and Haufe Output Folders

General ROI weight-map output root:

```text
N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\results\roi_weight_maps_voxel_trial_source_zscore
```

Amygdala subject-level Haufe matrix plots:

```text
N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\results\roi_weight_maps_voxel_trial_source_zscore\Amygdala\haufe_weight_map_matrix\subject_level_matrix_plots
```

Fusiform group-level Haufe matrix folder:

```text
N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\results\roi_weight_maps_voxel_trial_source_zscore\Fusiform\haufe_weight_map_matrix
```

Important Fusiform group files:

```text
Fusiform_cached_haufe_pattern_maps_matrix_form.png
Fusiform_cached_haufe_pattern_weight_map_matrix_results.xlsx
group_Fusiform_natural_pleasant_vs_neutral_haufe_pattern_zscore.nii.gz
group_Fusiform_ai_pleasant_vs_neutral_haufe_pattern_zscore.nii.gz
group_Fusiform_natural_unpleasant_vs_neutral_haufe_pattern_zscore.nii.gz
group_Fusiform_ai_unpleasant_vs_neutral_haufe_pattern_zscore.nii.gz
```

Fusiform subject-level Haufe matrix plots were later generated for the feature-selected top-50 analyses, not the original full-Fusiform group folder.

## Fusiform Cross-Decoding Results

Results are for bilateral Fusiform, excluding `Sub3` and `Sub8`.

Cross-decoding accuracy:

```text
Pleasant vs Neutral:
  trained on Natural, tested on AI: 83.1%
  trained on AI, tested on Natural: 84.6%
  mean bidirectional cross-decoding accuracy: 83.9%

Unpleasant vs Neutral:
  trained on Natural, tested on AI: 64.7%
  trained on AI, tested on Natural: 64.7%
  mean bidirectional cross-decoding accuracy: 64.7%
```

Interpretation:

```text
Fusiform cross decoding is strong for Pleasant vs Neutral and above chance for Unpleasant vs Neutral. Pleasant transfers much more strongly between Natural and AI.
```

## Fusiform Haufe Similarity Results

Subject-first Natural-vs-AI Haufe correlations:

```text
Pleasant vs Neutral:
  full Fusiform ROI r = 0.496
  Fisher-z mean r = 0.508

Unpleasant vs Neutral:
  full Fusiform ROI r = 0.235
  Fisher-z mean r = 0.239
```

These correlations are lower than cross-decoding accuracy. This is not contradictory:

```text
Cross decoding asks whether a transferable decision boundary exists.
Haufe-map correlation asks whether the full voxelwise pattern is spatially similar.
```

Current best interpretation:

```text
Fusiform shows transferable Natural/AI emotional decoding, especially for pleasant-vs-neutral, but Natural and AI do not evoke identical voxelwise patterns. The shared information is real, but spatially partial.
```

## Seed and Cache Diagnostic

Diagnostic folder:

```text
N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\results\haufe_seed_cache_diagnostic
```

Fusiform unpleasant-vs-neutral diagnostic results:

```text
cache_natural_vs_ai_pearson_r mean: 0.235316
rerun_primary_natural_vs_ai_pearson_r mean: 0.235316
altseed_natural_vs_ai_pearson_r mean: 0.235313

cache_vs_rerun_primary_natural_pearson_r: 1.000000
cache_vs_rerun_primary_ai_pearson_r: 1.000000
same_seed_rerun_natural_pearson_r: 1.000000
same_seed_rerun_ai_pearson_r: 1.000000
primary_vs_secondary_seed_natural_pearson_r: 0.999996
primary_vs_secondary_seed_ai_pearson_r: 0.999995
```

Conclusion:

```text
The low Natural-vs-AI Haufe correlation is not caused by random seed, rerun instability, or cache/storage problems.
```

## Subject-First vs Group-Map Correlations

Preferred result:

```text
correlate Natural vs AI maps within each subject first, then average correlations across subjects.
```

This avoids overestimating similarity due to group averaging.

Group-map correlation:

```text
average subject maps first, then correlate group-average Natural and AI maps.
```

Group-map correlations can be higher because averaging reduces subject-specific noise. They are useful for QC and visualization, but not the main inferential value for subject-level similarity.

## Fusiform Discrepancy Audit

Audit output folder:

```text
N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\results\fusiform_cross_vs_haufe_discrepancy_audit
```

Important audit files:

```text
fusiform_subject_cross_vs_haufe_audit.csv
fusiform_summary_cross_vs_haufe_audit.csv
fusiform_cross_vs_haufe_discrepancy_audit.xlsx
fusiform_accuracy_vs_haufe_summary.png
fusiform_subject_cross_accuracy_vs_haufe_r.png
```

Main finding:

```text
High cross decoding and modest full-ROI Haufe similarity coexist because the classifier can rely on a smaller shared discriminative subspace while the full ROI still contains source-specific or weak/noisy voxel patterns.
```

Evidence:

```text
Across subjects, mean cross accuracy correlates strongly with subject-first Haufe similarity:
  Pleasant: r about 0.84
  Unpleasant: r about 0.88
```

So the measures are related, but not equivalent.

## Top-k Haufe Feature Selection

Top-k analysis selects voxels based on absolute Natural Haufe weight:

```text
For each subject and contrast:
1. Take the Natural Haufe vector.
2. Keep voxels valid in both Natural and AI.
3. Rank by abs(Natural Haufe weight).
4. Select top k percent.
5. Compute Natural-vs-AI correlation on the selected voxels.
```

This is subject-specific and contrast-specific.

Top 5% Fusiform voxels by Natural Haufe weight:

```text
Pleasant Natural-vs-AI similarity: r about 0.786
Unpleasant Natural-vs-AI similarity: r about 0.514
```

Top 50% Fusiform voxels by Natural Haufe weight:

```text
Pleasant Natural-vs-AI similarity: r about 0.615
Unpleasant Natural-vs-AI similarity: r about 0.317
```

Whole Fusiform ROI:

```text
Pleasant Natural-vs-AI similarity: r about 0.496
Unpleasant Natural-vs-AI similarity: r about 0.235
```

Interpretation:

```text
The shared Natural/AI pattern is strongest among the most informative Fusiform voxels, remains elevated when using the top 50%, and becomes weaker when the entire ROI is included.
```

Top-50 Haufe output folder:

```text
N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\results\fusiform_cross_vs_haufe_discrepancy_audit\fusiform_top50_weight_map_correlation
```

Important files:

```text
Fusiform_top50_subject_level_haufe_correlations.csv
Fusiform_top50_summary_haufe_correlations.csv
Fusiform_top50_haufe_correlations.xlsx
Fusiform_top50_summary_haufe_correlations.png
Fusiform_top50_subject_level_matrix_plot_index.csv
```

Subject-level top-50 Haufe scatter plots:

```text
...\fusiform_top50_weight_map_correlation\subject_level_plots
```

Subject-level top-50 Haufe matrix plots:

```text
...\fusiform_top50_weight_map_correlation\subject_level_matrix_plots
```

Matrix plot count:

```text
28 subject plots
```

Matrix value CSVs:

```text
...\fusiform_top50_weight_map_correlation\subject_level_matrix_values
```

The top-50 Haufe matrix plots were regenerated with a clearer GridSpec layout: Natural and AI matrices, separate statistics column, separate colorbar, Pearson/Spearman r and p-values.

## Top-50 SVM Feature-Selection Control

Reason for control:

```text
Haufe weights are better for interpretable activation-pattern similarity.
SVM weights are better for classifier mechanics.
```

The SVM control tests whether selecting classifier-relevant voxels gives the same qualitative conclusion.

Selection rule:

```text
For each subject and contrast:
select top 50% voxels by absolute Natural SVM weight.
```

Top-50 SVM output folder:

```text
N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\results\fusiform_cross_vs_haufe_discrepancy_audit\fusiform_top50_svm_weight_correlation
```

Timestamp for latest SVM top-50 outputs:

```text
2026-07-10 23:46
```

Important files:

```text
Fusiform_top50_svm_subject_level_weight_correlations.csv
Fusiform_top50_svm_summary_weight_correlations.csv
Fusiform_top50_svm_weight_correlations.xlsx
Fusiform_top50_svm_summary_weight_correlations.png
Fusiform_top50_svm_subject_level_matrix_plot_index.csv
```

Subject-level top-50 SVM scatter plots:

```text
...\fusiform_top50_svm_weight_correlation\subject_level_plots
```

Subject-level top-50 SVM matrix plots:

```text
...\fusiform_top50_svm_weight_correlation\subject_level_matrix_plots
```

Matrix plot count:

```text
28 subject plots
```

Matrix value CSVs:

```text
...\fusiform_top50_svm_weight_correlation\subject_level_matrix_values
```

Top-50 SVM summary:

```text
Pleasant vs Neutral:
  Full SVM r: 0.451
  Top 50% SVM r: 0.563
  Haufe r in same SVM-selected top 50%: 0.600

Unpleasant vs Neutral:
  Full SVM r: 0.213
  Top 50% SVM r: 0.285
  Haufe r in same SVM-selected top 50%: 0.311
```

Interpretation:

```text
The SVM-based control tells the same story: selecting more classifier-relevant Fusiform voxels increases Natural-vs-AI similarity, but the full pattern is still only partially shared.
```

## Current Scientific Interpretation

Current wording:

```text
Fusiform shows transferable Natural/AI emotional decoding, especially for pleasant-vs-neutral, but Natural and AI do not evoke identical voxelwise patterns. The shared information is real, but spatially partial.
```

Expanded interpretation:

```text
ROI decoding uses all voxels, but the classifier does not average them equally. A smaller subset of informative voxels can dominate the decision boundary through large weights. Therefore, strong cross decoding can coexist with modest full-ROI Haufe-map correlation.
```

The top-k results support this:

```text
Top 5% and top 50% informative voxels show stronger Natural-vs-AI similarity than the full ROI.
```

## Important Cautions

1. Do not interpret high cross decoding as proof that Natural and AI evoke identical voxelwise responses.
2. Cross decoding means a decision boundary transfers.
3. Haufe similarity means the full voxelwise response/importance pattern is spatially similar.
4. Subject-first correlations are preferred for inference.
5. Group-map correlations are useful for visualization/QC, but can overestimate similarity because group averaging reduces noise.
6. Feature selection must be performed inside training data for any new decoding analysis that tests predictive performance. Otherwise, feature selection may leak test information.

## Next Planned Analysis

Goal:

```text
Use feature-selected voxels, such as top 50% from Natural Haufe or SVM weights, to run balanced-data single-trial decoding instead of the previous ERP-style random averaged-trial decoding.
```

Key design requirement:

```text
Feature selection must be nested inside the training fold for any within-source or cross-source decoding accuracy estimate.
```

Why:

```text
The existing top-50 analyses selected voxels from already computed subject-level maps for descriptive similarity analysis. That is fine for post hoc interpretation, but for a new decoding performance analysis, voxel selection must avoid using test data.
```

Recommended new analysis structure:

1. Start from single-trial beta patterns.
2. For each subject and ROI, use balanced trial counts per condition.
3. For each fold:
   - Split data into train/test.
   - Compute feature-selection weights using training data only.
   - Select top 50% voxels from training-derived weights.
   - Train a linear SVM on selected voxels using single trials, not averaged chunks.
   - Test on held-out single trials.
4. For cross decoding:
   - Train on one source, test on the other source.
   - Decide whether feature selection is based on training source only, or on a training-only nested combined procedure.
   - Preferred first version: select features from the training source only, then apply those voxels to the test source.
5. Save subject-level accuracies, selected voxel indices, selected voxel masks, and model weights.

Recommended comparisons:

```text
No feature selection
Top 50% by training-fold SVM weight
Top 50% by training-fold Haufe pattern
Optional top 5%, top 10%, top 20%, top 50%
```

Recommended first ROI:

```text
Fusiform
```

Then expand to:

```text
Amygdala
Superset ROI list
```

Recommended outputs:

```text
subject-level single-trial decoding accuracy CSV
group summary CSV/XLSX
selected voxel mask NIfTI files
selected voxel frequency maps
subject-level selected voxel matrix plots
comparison plots against ERP-style decoding
```

## New Chat Handoff Prompt

Use this file as context for the new thread:

```text
N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\PROJECT_KNOWLEDGE_DECODING.md
```

New thread task:

```text
Implement feature-selected balanced single-trial decoding for Fusiform first. Use top 50% voxels selected within the training fold from SVM weights and/or Haufe patterns. Avoid leakage. Compare against no-feature-selection single-trial decoding and the existing ERP-style averaged decoding results.
```
