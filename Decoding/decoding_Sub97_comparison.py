"""
Single-Trial Beta Decoding: Sub97_new vs Sub97_old Comparison
=============================================================
This script performs MVPA decoding on single-trial betas from two preprocessing
pipelines (Sub97_new and Sub97_old) to compare decoding accuracy.

Analyses:
1. Within-source decoding (single trial betas, with and without z-scoring)
2. Within-source decoding (averaged betas: 100 trials -> 5 averaged trials)
3. Cross-source decoding (train on one source, test on another)

Comparisons:
- pleasant_AI vs neutral_AI
- pleasant_Natural vs neutral_Natural
- unpleasant_AI vs neutral_AI
- unpleasant_Natural vs neutral_Natural

Beta indexing: 10 runs x 60 trials/run = 600 trial betas
  Run1: beta_0001-0060, Run2: beta_0067-0126, ..., Run10: beta_0595-0654
  (betas 61-66, 127-132, etc. are motion regressors, skipped)
"""

import os
import re
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import masking
from nilearn.image import resample_img
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.stats import zscore
from tqdm import tqdm
import warnings
import pickle

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIGURATION
# ============================================================
beta_dir = r'N:\Experimental_Data\yujunchen\projects\AI_IAPS\GLM_singletrial\betas'
label_file = r'N:\Experimental_Data\yujunchen\projects\AI_IAPS\GLM_singletrial\beta_groups.csv'
mask_file = r'N:\Experimental_Data\yujunchen\projects\data\masks\MNI152_T1_2mm_brain_mask.nii.gz'
roi_pkl = r'N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\kastner_dict.pkl'  # adjust path if needed

subs = ['Sub97_new', 'Sub97_old']

# Decoding parameters
N_REPEATS = 50
N_FOLDS = 4
N_AVG_GROUPS = 5  # 100 trials / 20 stimuli per category, 5 repetitions -> average into 5

# Within-source comparisons
within_comparisons = [
    ('pleasantAI', 'neutralAI'),
    ('pleasant', 'neutral'),
    ('unpleasantAI', 'neutralAI'),
    ('unpleasant', 'neutral'),
]

# Cross-source comparisons: (train_cond1, train_cond2, test_cond1, test_cond2)
cross_comparisons = [
    ('pleasantAI', 'neutralAI', 'pleasant', 'neutral'),
    ('pleasant', 'neutral', 'pleasantAI', 'neutralAI'),
    ('unpleasantAI', 'neutralAI', 'unpleasant', 'neutral'),
    ('unpleasant', 'neutral', 'unpleasantAI', 'neutralAI'),
]

# ============================================================
# 1. LOAD BETA LABELS
# ============================================================
print("Loading beta labels...")
beta_labels = pd.read_csv(label_file, header=None, names=['beta_file', 'category'])
print(f"  Total betas: {len(beta_labels)}")
print(f"  Categories: {beta_labels['category'].value_counts().to_dict()}")

# Extract beta numbers
beta_labels['beta_num'] = beta_labels['beta_file'].apply(
    lambda x: int(re.findall(r'\d+', x)[0])
)

# ============================================================
# 2. LOAD ROI MASKS
# ============================================================
print("\nLoading ROI masks...")
with open(roi_pkl, 'rb') as f:
    roi_masks = pickle.load(f)
print(f"  Loaded {len(roi_masks)} ROIs: {list(roi_masks.keys())}")

# ============================================================
# 3. LOAD BETA DATA FOR EACH SUBJECT
# ============================================================
def load_betas(sub_name, beta_labels_df, mask_img=None):
    """Load all trial beta images for a subject, apply brain mask."""
    sub_dir = os.path.join(beta_dir, sub_name)
    beta_data = []

    # Get reference image for resampling mask
    first_beta = os.path.join(sub_dir, beta_labels_df['beta_file'].iloc[0])
    ref_img = nib.load(first_beta)

    if mask_img is not None:
        resampled_mask = resample_img(mask_img, target_affine=ref_img.affine,
                                       target_shape=ref_img.shape)
        mask_data = resampled_mask.get_fdata() > 0.5
    else:
        mask_data = None

    for _, row in tqdm(beta_labels_df.iterrows(), total=len(beta_labels_df),
                        desc=f'Loading {sub_name}'):
        fpath = os.path.join(sub_dir, row['beta_file'])
        img = nib.load(fpath)
        data = img.get_fdata().flatten()
        beta_data.append(data)

    beta_data = np.array(beta_data)  # (600, n_voxels)
    return beta_data, ref_img

print("\nLoading beta data...")
mask_img = nib.load(mask_file)
sub_data = {}
for sub in subs:
    print(f"\n  Loading {sub}...")
    sub_data[sub], ref_img = load_betas(sub, beta_labels, mask_img)
    print(f"  Shape: {sub_data[sub].shape}")

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================
def get_roi_voxels(roi_mask, ref_img):
    """Get voxel indices for an ROI, resampled to match beta images."""
    resampled = resample_img(roi_mask, target_affine=ref_img.affine,
                              target_shape=ref_img.shape)
    flat = resampled.get_fdata().flatten()
    return np.where(flat > 0)[0]

def extract_roi_data(data, labels, conditions, voxel_idx):
    """Extract data for given conditions within an ROI, remove NaN voxels.

    Args:
        data: (n_betas, n_voxels) full brain data
        labels: DataFrame with 'category' column
        conditions: list of condition names
        voxel_idx: array of voxel indices for this ROI

    Returns:
        dict mapping condition -> (n_trials, n_valid_voxels) array
    """
    cond_data = {}
    for cond in conditions:
        idx = labels[labels['category'] == cond].index.values
        cond_data[cond] = data[idx][:, voxel_idx]

    # Find voxels that are valid (non-NaN) across ALL conditions
    valid = np.ones(len(voxel_idx), dtype=bool)
    for cond in conditions:
        valid &= ~np.any(np.isnan(cond_data[cond]), axis=0)

    for cond in conditions:
        cond_data[cond] = cond_data[cond][:, valid]

    return cond_data, np.sum(valid)

def average_into_groups(data, n_groups=5):
    """Average trials into n_groups (e.g., 100 trials -> 5 groups of 20).
    Each group averages across repetitions of the same stimuli set."""
    n_trials = data.shape[0]
    group_size = n_trials // n_groups
    averaged = np.zeros((n_groups, data.shape[1]))
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        averaged[g] = np.mean(data[start:end], axis=0)
    return averaged

def decode_within_source(d1, d2, n_repeats=N_REPEATS, n_folds=N_FOLDS):
    """Within-source decoding using repeated stratified k-fold CV."""
    X = np.vstack([d1, d2])
    y = np.array([1] * len(d1) + [0] * len(d2))

    accuracies = []
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats,
                                     random_state=42)

    for train_idx, test_idx in rskf.split(X, y):
        clf = SVC(kernel='linear', C=1.0)
        clf.fit(X[train_idx], y[train_idx])
        acc = clf.score(X[test_idx], y[test_idx])
        accuracies.append(acc)

    return np.mean(accuracies)

def decode_cross_source(train_d1, train_d2, test_d1, test_d2):
    """Cross-source decoding: train on one source, test on another."""
    X_train = np.vstack([train_d1, train_d2])
    y_train = np.array([1] * len(train_d1) + [0] * len(train_d2))

    X_test = np.vstack([test_d1, test_d2])
    y_test = np.array([1] * len(test_d1) + [0] * len(test_d2))

    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

# ============================================================
# 5. WITHIN-SOURCE DECODING — SINGLE TRIAL BETAS
# ============================================================
print("\n" + "=" * 70)
print("WITHIN-SOURCE DECODING — SINGLE TRIAL BETAS")
print("=" * 70)

results_within_single = {}

for sub in subs:
    results_within_single[sub] = {}
    data = sub_data[sub]

    for cond1, cond2 in within_comparisons:
        comp_name = f"{cond1}_vs_{cond2}"
        print(f"\n  {sub} | {comp_name}")

        roi_accs_no_z = {}
        roi_accs_z = {}

        for roi_name, roi_mask in roi_masks.items():
            voxel_idx = get_roi_voxels(roi_mask, ref_img)
            cond_data, n_valid = extract_roi_data(data, beta_labels,
                                                   [cond1, cond2], voxel_idx)

            if n_valid < 10:
                print(f"    {roi_name}: too few voxels ({n_valid}), skipping")
                continue

            d1_roi = cond_data[cond1]
            d2_roi = cond_data[cond2]

            # WITHOUT z-scoring
            acc_no_z = decode_within_source(d1_roi, d2_roi)
            roi_accs_no_z[roi_name] = acc_no_z

            # WITH z-scoring
            # Step 1: z-score across voxels per trial (pattern normalization)
            d1_z = zscore(d1_roi, axis=1)
            d2_z = zscore(d2_roi, axis=1)
            # Step 2: z-score across trials per voxel (mean-center each voxel)
            combined_z = np.vstack([d1_z, d2_z])
            combined_z = zscore(combined_z, axis=0)
            d1_z = combined_z[:len(d1_roi)]
            d2_z = combined_z[len(d1_roi):]

            acc_z = decode_within_source(d1_z, d2_z)
            roi_accs_z[roi_name] = acc_z

            print(f"    {roi_name}: no_z={acc_no_z:.3f}, z={acc_z:.3f}")

        results_within_single[sub][comp_name] = {
            'no_zscore': roi_accs_no_z,
            'zscore': roi_accs_z
        }

# ============================================================
# 6. WITHIN-SOURCE DECODING — AVERAGED BETAS
# ============================================================
print("\n" + "=" * 70)
print("WITHIN-SOURCE DECODING — AVERAGED BETAS (100 -> 5 groups)")
print("=" * 70)

results_within_avg = {}

for sub in subs:
    results_within_avg[sub] = {}
    data = sub_data[sub]

    for cond1, cond2 in within_comparisons:
        comp_name = f"{cond1}_vs_{cond2}"
        print(f"\n  {sub} | {comp_name}")

        roi_accs = {}

        for roi_name, roi_mask in roi_masks.items():
            voxel_idx = get_roi_voxels(roi_mask, ref_img)
            cond_data, n_valid = extract_roi_data(data, beta_labels,
                                                   [cond1, cond2], voxel_idx)

            if n_valid < 10:
                continue

            # Average 100 trials into 5 groups (each group = 20 trials averaged)
            d1_avg = average_into_groups(cond_data[cond1], n_groups=N_AVG_GROUPS)
            d2_avg = average_into_groups(cond_data[cond2], n_groups=N_AVG_GROUPS)

            # With only 5 samples per class, use leave-one-out style
            acc = decode_within_source(d1_avg, d2_avg, n_repeats=50, n_folds=N_AVG_GROUPS)
            roi_accs[roi_name] = acc

            print(f"    {roi_name}: acc={acc:.3f}")

        results_within_avg[sub][comp_name] = roi_accs

# ============================================================
# 7. CROSS-SOURCE DECODING — SINGLE TRIAL BETAS
# ============================================================
print("\n" + "=" * 70)
print("CROSS-SOURCE DECODING — SINGLE TRIAL BETAS")
print("=" * 70)

results_cross_single = {}

for sub in subs:
    results_cross_single[sub] = {}
    data = sub_data[sub]

    for train_c1, train_c2, test_c1, test_c2 in cross_comparisons:
        comp_name = f"train_{train_c1}v{train_c2}_test_{test_c1}v{test_c2}"
        print(f"\n  {sub} | {comp_name}")

        roi_accs_no_z = {}
        roi_accs_z = {}

        for roi_name, roi_mask in roi_masks.items():
            voxel_idx = get_roi_voxels(roi_mask, ref_img)
            all_conds = [train_c1, train_c2, test_c1, test_c2]
            cond_data, n_valid = extract_roi_data(data, beta_labels,
                                                   all_conds, voxel_idx)

            if n_valid < 10:
                continue

            train_d1 = cond_data[train_c1]
            train_d2 = cond_data[train_c2]
            test_d1 = cond_data[test_c1]
            test_d2 = cond_data[test_c2]

            # WITHOUT z-scoring
            acc_no_z = decode_cross_source(train_d1, train_d2, test_d1, test_d2)
            roi_accs_no_z[roi_name] = acc_no_z

            # WITH z-scoring (z-score train and test SEPARATELY to avoid data leakage)
            train_all = np.vstack([train_d1, train_d2])
            train_all_z = zscore(zscore(train_all, axis=1), axis=0)
            train_d1_z = train_all_z[:len(train_d1)]
            train_d2_z = train_all_z[len(train_d1):]

            test_all = np.vstack([test_d1, test_d2])
            test_all_z = zscore(zscore(test_all, axis=1), axis=0)
            test_d1_z = test_all_z[:len(test_d1)]
            test_d2_z = test_all_z[len(test_d1):]

            acc_z = decode_cross_source(train_d1_z, train_d2_z, test_d1_z, test_d2_z)
            roi_accs_z[roi_name] = acc_z

            print(f"    {roi_name}: no_z={acc_no_z:.3f}, z={acc_z:.3f}")

        results_cross_single[sub][comp_name] = {
            'no_zscore': roi_accs_no_z,
            'zscore': roi_accs_z
        }

# ============================================================
# 8. CROSS-SOURCE DECODING — AVERAGED BETAS
# ============================================================
print("\n" + "=" * 70)
print("CROSS-SOURCE DECODING — AVERAGED BETAS (100 -> 5 groups)")
print("=" * 70)

results_cross_avg = {}

for sub in subs:
    results_cross_avg[sub] = {}
    data = sub_data[sub]

    for train_c1, train_c2, test_c1, test_c2 in cross_comparisons:
        comp_name = f"train_{train_c1}v{train_c2}_test_{test_c1}v{test_c2}"
        print(f"\n  {sub} | {comp_name}")

        roi_accs = {}

        for roi_name, roi_mask in roi_masks.items():
            voxel_idx = get_roi_voxels(roi_mask, ref_img)
            all_conds = [train_c1, train_c2, test_c1, test_c2]
            cond_data, n_valid = extract_roi_data(data, beta_labels,
                                                   all_conds, voxel_idx)

            if n_valid < 10:
                continue

            train_d1 = average_into_groups(cond_data[train_c1], N_AVG_GROUPS)
            train_d2 = average_into_groups(cond_data[train_c2], N_AVG_GROUPS)
            test_d1 = average_into_groups(cond_data[test_c1], N_AVG_GROUPS)
            test_d2 = average_into_groups(cond_data[test_c2], N_AVG_GROUPS)

            acc = decode_cross_source(train_d1, train_d2, test_d1, test_d2)
            roi_accs[roi_name] = acc

            print(f"    {roi_name}: acc={acc:.3f}")

        results_cross_avg[sub][comp_name] = roi_accs

# ============================================================
# 9. SUMMARY TABLE
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY — MEAN ACCURACY ACROSS ROIs")
print("=" * 70)

def summarize_results(results_dict, title):
    """Print summary table of mean accuracy across ROIs."""
    print(f"\n--- {title} ---")
    for sub in subs:
        print(f"\n  {sub}:")
        for comp, data in results_dict[sub].items():
            if isinstance(data, dict) and 'no_zscore' in data:
                accs_no_z = list(data['no_zscore'].values())
                accs_z = list(data['zscore'].values())
                if accs_no_z:
                    print(f"    {comp}:")
                    print(f"      no_z: {np.mean(accs_no_z):.3f} +/- {np.std(accs_no_z):.3f}")
                    print(f"      z:    {np.mean(accs_z):.3f} +/- {np.std(accs_z):.3f}")
            else:
                accs = list(data.values())
                if accs:
                    print(f"    {comp}: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")

summarize_results(results_within_single, "Within-Source (Single Trial)")
summarize_results(results_within_avg, "Within-Source (Averaged)")
summarize_results(results_cross_single, "Cross-Source (Single Trial)")
summarize_results(results_cross_avg, "Cross-Source (Averaged)")

# ============================================================
# 10. SAVE ALL RESULTS
# ============================================================
output_dir = r'N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\results'
os.makedirs(output_dir, exist_ok=True)

all_results = {
    'within_single': results_within_single,
    'within_averaged': results_within_avg,
    'cross_single': results_cross_single,
    'cross_averaged': results_cross_avg,
    'subs': subs,
    'within_comparisons': within_comparisons,
    'cross_comparisons': cross_comparisons,
}

save_path = os.path.join(output_dir, 'decoding_Sub97_comparison.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(all_results, f)
print(f"\nResults saved to: {save_path}")

print("\nDone!")
