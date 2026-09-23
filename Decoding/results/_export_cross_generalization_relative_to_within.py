import os
import pickle
import numpy as np
import pandas as pd

# Cross-decoding interpreted relative to within-source decodability.
default_results_path = r'N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\results\decoding_multisub_avg_voxel_trial_source_zscore_aal3_all.pkl'
with open(default_results_path, 'rb') as f:
    all_results = pickle.load(f)

results_within_avg = all_results['within_avg']
results_cross_avg = all_results['cross_avg']
subs = all_results['subs']
roi_names = all_results['roi_names']
AVG_MODE = all_results.get('avg_z_mode', 'voxel_trial_source_zscore')

subs_analysis = [s for s in subs if s not in ['Sub3', 'Sub8']]
CHANCE = 0.50
MIN_WITHIN_MARGIN = 0.03  # conservative descriptive flag: each within-source accuracy >= 53%
EPS = 1e-8

contrast_specs = [
    {
        'contrast': 'pleasant_vs_neutral',
        'short_label': 'pleasant',
        'within_natural_comp': 'pleasant_vs_neutral',
        'within_ai_comp': 'pleasantAI_vs_neutralAI',
        'cross_natural_to_ai_comp': 'train_pleasantvneutral_test_pleasantAIvneutralAI',
        'cross_ai_to_natural_comp': 'train_pleasantAIvneutralAI_test_pleasantvneutral',
    },
    {
        'contrast': 'unpleasant_vs_neutral',
        'short_label': 'unpleasant',
        'within_natural_comp': 'unpleasant_vs_neutral',
        'within_ai_comp': 'unpleasantAI_vs_neutralAI',
        'cross_natural_to_ai_comp': 'train_unpleasantvneutral_test_unpleasantAIvneutralAI',
        'cross_ai_to_natural_comp': 'train_unpleasantAIvneutralAI_test_unpleasantvneutral',
    },
]

kastner_rois = set(all_results.get('kastner_roi_names', []))
aal3_rois = set(all_results.get('aal3_roi_names', []))
aal3_rois |= {f'AAL3_{r}' for r in all_results.get('aal3_roi_names', [])}


def roi_set_name(roi):
    if roi in kastner_rois:
        return 'kastner_wang'
    if roi in aal3_rois:
        return 'aal3'
    return 'other'


def get_within(sub, comp, roi):
    return results_within_avg.get(sub, {}).get(comp, {}).get(AVG_MODE, {}).get(roi, np.nan)


def get_cross(sub, comp, roi):
    return results_cross_avg.get(sub, {}).get(comp, {}).get(AVG_MODE, {}).get(roi, np.nan)


def finite_mean(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else np.nan


def finite_sem(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return np.nan
    return float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))


def normalized_ratio(numerator, denominator):
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= EPS:
        return np.nan
    return float(numerator / denominator)


def compute_metrics(w_nat, w_ai, c_nat_to_ai, c_ai_to_nat):
    w_mean = finite_mean([w_nat, w_ai])
    c_mean = finite_mean([c_nat_to_ai, c_ai_to_nat])
    w_min = np.nanmin([w_nat, w_ai]) if np.any(np.isfinite([w_nat, w_ai])) else np.nan

    within_mean_signal = w_mean - CHANCE if np.isfinite(w_mean) else np.nan
    within_min_signal = w_min - CHANCE if np.isfinite(w_min) else np.nan
    cross_signal = c_mean - CHANCE if np.isfinite(c_mean) else np.nan

    return {
        'within_mean': w_mean,
        'within_min': float(w_min) if np.isfinite(w_min) else np.nan,
        'cross_mean': c_mean,
        'within_mean_signal': within_mean_signal,
        'within_min_signal': within_min_signal,
        'cross_signal': cross_signal,
        'cross_generalization_index_mean_within': normalized_ratio(cross_signal, within_mean_signal),
        'shared_index_min_within': normalized_ratio(cross_signal, within_min_signal),
        'cross_asymmetry_natural_to_ai_minus_ai_to_natural': (
            c_nat_to_ai - c_ai_to_nat if np.isfinite(c_nat_to_ai) and np.isfinite(c_ai_to_nat) else np.nan
        ),
        'within_asymmetry_natural_minus_ai': (
            w_nat - w_ai if np.isfinite(w_nat) and np.isfinite(w_ai) else np.nan
        ),
        'within_both_above_chance': bool(
            np.isfinite(w_nat) and np.isfinite(w_ai) and w_nat > CHANCE and w_ai > CHANCE
        ),
        'cross_mean_above_chance': bool(np.isfinite(c_mean) and c_mean > CHANCE),
        'source_generalization_candidate': bool(
            np.isfinite(w_nat) and np.isfinite(w_ai) and np.isfinite(c_mean)
            and w_nat > CHANCE and w_ai > CHANCE and c_mean > CHANCE
        ),
        'source_generalization_candidate_margin_53': bool(
            np.isfinite(w_nat) and np.isfinite(w_ai) and np.isfinite(c_mean)
            and w_nat >= CHANCE + MIN_WITHIN_MARGIN
            and w_ai >= CHANCE + MIN_WITHIN_MARGIN
            and c_mean > CHANCE
        ),
    }


subject_rows = []
for sub in subs_analysis:
    for roi in roi_names:
        for spec in contrast_specs:
            w_nat = get_within(sub, spec['within_natural_comp'], roi)
            w_ai = get_within(sub, spec['within_ai_comp'], roi)
            c_nat_to_ai = get_cross(sub, spec['cross_natural_to_ai_comp'], roi)
            c_ai_to_nat = get_cross(sub, spec['cross_ai_to_natural_comp'], roi)
            metrics = compute_metrics(w_nat, w_ai, c_nat_to_ai, c_ai_to_nat)
            subject_rows.append({
                'subject': sub,
                'roi': roi,
                'roi_set': roi_set_name(roi),
                'contrast': spec['contrast'],
                'short_label': spec['short_label'],
                'within_natural_comp': spec['within_natural_comp'],
                'within_ai_comp': spec['within_ai_comp'],
                'cross_natural_to_ai_comp': spec['cross_natural_to_ai_comp'],
                'cross_ai_to_natural_comp': spec['cross_ai_to_natural_comp'],
                'within_natural': w_nat,
                'within_ai': w_ai,
                'cross_natural_to_ai': c_nat_to_ai,
                'cross_ai_to_natural': c_ai_to_nat,
                **metrics,
            })

subject_df = pd.DataFrame(subject_rows)

# Group summary: raw accuracies are averaged first; normalized indices are then computed from group means.
group_rows = []
raw_metric_cols = ['within_natural', 'within_ai', 'cross_natural_to_ai', 'cross_ai_to_natural']
subject_index_cols = ['cross_generalization_index_mean_within', 'shared_index_min_within']

for roi in roi_names:
    for spec in contrast_specs:
        d = subject_df[(subject_df['roi'] == roi) & (subject_df['contrast'] == spec['contrast'])]
        raw_means = {col: finite_mean(d[col].values) for col in raw_metric_cols}
        raw_sems = {f'{col}_sem': finite_sem(d[col].values) for col in raw_metric_cols}
        group_metrics = compute_metrics(
            raw_means['within_natural'],
            raw_means['within_ai'],
            raw_means['cross_natural_to_ai'],
            raw_means['cross_ai_to_natural'],
        )
        group_rows.append({
            'roi': roi,
            'roi_set': roi_set_name(roi),
            'contrast': spec['contrast'],
            'short_label': spec['short_label'],
            'n_subjects': int(d['subject'].nunique()),
            **raw_means,
            **raw_sems,
            **{f'{k}_from_group_mean': v for k, v in group_metrics.items()},
            'mean_subject_cross_generalization_index_mean_within': finite_mean(d['cross_generalization_index_mean_within'].values),
            'sem_subject_cross_generalization_index_mean_within': finite_sem(d['cross_generalization_index_mean_within'].values),
            'mean_subject_shared_index_min_within': finite_mean(d['shared_index_min_within'].values),
            'sem_subject_shared_index_min_within': finite_sem(d['shared_index_min_within'].values),
        })

group_df = pd.DataFrame(group_rows)

ranked_df = group_df.sort_values(
    by=['source_generalization_candidate_margin_53_from_group_mean', 'shared_index_min_within_from_group_mean', 'cross_mean_from_group_mean'],
    ascending=[False, False, False],
    na_position='last',
).reset_index(drop=True)
ranked_df.insert(0, 'rank', np.arange(1, len(ranked_df) + 1))

out_dir = os.path.join(
    os.path.dirname(default_results_path),
    'cross_generalization_relative_to_within_voxel_trial_source_zscore'
)
os.makedirs(out_dir, exist_ok=True)

subject_csv = os.path.join(out_dir, 'cross_generalization_subject_level.csv')
group_csv = os.path.join(out_dir, 'cross_generalization_group_summary.csv')
ranked_csv = os.path.join(out_dir, 'cross_generalization_ranked_rois.csv')
pkl_path = os.path.join(out_dir, 'cross_generalization_relative_to_within_results.pkl')

subject_df.to_csv(subject_csv, index=False)
group_df.to_csv(group_csv, index=False)
ranked_df.to_csv(ranked_csv, index=False)

export = {
    'subject_level': subject_df,
    'group_summary': group_df,
    'ranked_rois': ranked_df,
    'contrast_specs': contrast_specs,
    'subs_analysis': subs_analysis,
    'roi_names': roi_names,
    'avg_mode': AVG_MODE,
    'chance': CHANCE,
    'min_within_margin': MIN_WITHIN_MARGIN,
    'notes': {
        'within_mean': '(within_natural + within_ai) / 2',
        'cross_mean': '(cross_natural_to_ai + cross_ai_to_natural) / 2',
        'cross_generalization_index_mean_within': '(cross_mean - chance) / (within_mean - chance)',
        'shared_index_min_within': '(cross_mean - chance) / (min(within_natural, within_ai) - chance)',
        'recommended_interpretation': 'Interpret shared_index only when both within-source accuracies are above chance; the margin_53 flag is a conservative descriptive screen, not a permutation test.',
    },
}
with open(pkl_path, 'wb') as f:
    pickle.dump(export, f)

print(f"Subjects used: {len(subs_analysis)}")
print(f"ROIs analyzed: {len(roi_names)}")
print(f"Mode: {AVG_MODE}")
print(f"Saved subject-level table: {subject_csv}")
print(f"Saved group summary table: {group_csv}")
print(f"Saved ranked ROI table: {ranked_csv}")
print(f"Saved pickle: {pkl_path}")

print("\nTop source-generalization candidates by group-level shared index:")
cols = [
    'rank', 'roi', 'roi_set', 'contrast',
    'within_natural', 'within_ai', 'cross_mean_from_group_mean',
    'shared_index_min_within_from_group_mean',
    'source_generalization_candidate_margin_53_from_group_mean',
]
print(ranked_df[cols].head(20).to_string(index=False))
