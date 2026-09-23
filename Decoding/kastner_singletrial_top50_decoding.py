"""Kastner/Wang balanced single-trial decoding with nested top-50 voxels.

This is the ROI-wide version of the new single-trial analysis. It uses the same
Kastner/Wang ROIs and figure style as the ERP-style decoding plots, but trains
and tests on balanced single trials rather than averaged trial chunks.

Feature selection is nested inside each training fold by default:

- none: no feature selection
- svm_top50: top 50% voxels by absolute training-fold linear SVM weight
- haufe_top50: top 50% voxels by absolute training-fold Haufe pattern

The fixed feature-scope mode can instead reuse one selected voxel set across
folds. With --feature-basis erp_avg, that fixed voxel set is ranked from
ERP-style averaged trial chunks, then evaluated with single-trial decoding.

Outputs are saved under Decoding/results/kastner_singletrial_top50_feature_selection.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io
from joblib import Parallel, delayed
from nilearn.image import resample_img
from scipy import stats
from scipy.stats import zscore
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC, SVC
from tqdm import tqdm


PROJECT_ROOT = Path(r"N:\Experimental_Data\yujunchen\projects\AI_IAPS")
DECODING_ROOT = PROJECT_ROOT / "Decoding"


@dataclass
class Config:
    beta_dir: str = str(PROJECT_ROOT / "GLM_singletrial" / "betas")
    label_file: str = str(PROJECT_ROOT / "GLM_singletrial" / "beta_groups.csv")
    onset_base_dir: str = r"N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\DataRecording"
    kastner_atlas_file: str = r"C:\MRIcroGL\Resources\atlas\kastner.nii.gz"
    kastner_label_file: str = r"C:\MRIcroGL\Resources\atlas\kastner.nii.txt"
    output_dir: str = str(DECODING_ROOT / "results" / "kastner_singletrial_top50_feature_selection")
    existing_erp_results: str = str(DECODING_ROOT / "results" / "decoding_multisub_avg_voxel_trial_source_zscore_aal3_all.pkl")
    subjects: Tuple[str, ...] = (
        "Sub1", "Sub2", "Sub3", "Sub4", "Sub5", "Sub6", "Sub7", "Sub8", "Sub9",
        "Sub11", "Sub12", "Sub13", "Sub14", "Sub15", "Sub16", "Sub17", "Sub18",
        "Sub19", "Sub20", "Sub21", "Sub22", "Sub23", "Sub24", "Sub25", "Sub26",
        "Sub27", "Sub28", "Sub29", "Sub30", "Sub31",
    )
    group_exclude_subjects: Tuple[str, ...] = ("Sub3", "Sub8")
    runs: Tuple[str, ...] = (
        "Run01", "Run02", "Run03", "Run04", "Run05",
        "Run06", "Run07", "Run08", "Run09", "Run10",
    )
    selectors: Tuple[str, ...] = ("none", "svm_top50", "haufe_top50")
    n_repeats: int = 20
    n_folds: int = 4
    seed_base: int = 42
    top_fraction: float = 0.50
    min_voxels: int = 10
    clf_c: float = 1.0
    svm_backend: str = "svc"
    linear_svc_max_iter: int = 10000
    feature_scope: str = "nested"
    feature_basis: str = "single_trial"
    n_avg_groups: int = 3
    save_selection_frequency_maps: bool = True


NATURAL_CATS = ("pleasant", "neutral", "unpleasant")
AI_CATS = ("pleasantAI", "neutralAI", "unpleasantAI")
ALL_CATEGORIES = NATURAL_CATS + AI_CATS

KASTNER_ROI_GROUPS = {
    "V1v": ["V1v"], "V1d": ["V1d"],
    "V2v": ["V2v"], "V2d": ["V2d"],
    "V3v": ["V3v"], "V3d": ["V3d"],
    "hV4": ["hV4"],
    "V3a": ["V3a"], "V3b": ["V3b"],
    "IPS": ["IPS0", "IPS1", "IPS2", "IPS3", "IPS4", "IPS5"],
    "SPL1": ["SPL1"], "FEF": ["FEF"],
    "LO1": ["LO1"], "LO2": ["LO2"],
    "MST": ["MST"], "hMT": ["hMT"],
    "VO1": ["VO1"], "VO2": ["VO2"],
    "PHC1": ["PHC1"], "PHC2": ["PHC2"],
}

# Match the current ERP-style Kastner figure: all grouped Kastner ROIs except FEF and MST.
KASTNER_PLOT_EXCLUDE = {"FEF", "MST"}
KASTNER_PLOT_ROIS = tuple(r for r in KASTNER_ROI_GROUPS if r not in KASTNER_PLOT_EXCLUDE)

WITHIN_ORDER = (
    "pleasant_vs_neutral",
    "pleasantAI_vs_neutralAI",
    "unpleasant_vs_neutral",
    "unpleasantAI_vs_neutralAI",
)
CROSS_ORDER = (
    "train_pleasantvneutral_test_pleasantAIvneutralAI",
    "train_pleasantAIvneutralAI_test_pleasantvneutral",
    "train_unpleasantvneutral_test_unpleasantAIvneutralAI",
    "train_unpleasantAIvneutralAI_test_unpleasantvneutral",
)

WITHIN_SPECS = {
    "pleasant_vs_neutral": ("natural", "pleasant", "neutral", "PL vs Nt, Natural"),
    "pleasantAI_vs_neutralAI": ("ai", "pleasantAI", "neutralAI", "PL vs Nt, AI"),
    "unpleasant_vs_neutral": ("natural", "unpleasant", "neutral", "UP vs Nt, Natural"),
    "unpleasantAI_vs_neutralAI": ("ai", "unpleasantAI", "neutralAI", "UP vs Nt, AI"),
}
CROSS_SPECS = {
    "train_pleasantvneutral_test_pleasantAIvneutralAI": (
        "natural", "ai", "pleasant", "neutral", "pleasantAI", "neutralAI",
        "PL vs Nt, trained on NA and tested on AI",
    ),
    "train_pleasantAIvneutralAI_test_pleasantvneutral": (
        "ai", "natural", "pleasantAI", "neutralAI", "pleasant", "neutral",
        "PL vs Nt, trained on AI and tested on NA",
    ),
    "train_unpleasantvneutral_test_unpleasantAIvneutralAI": (
        "natural", "ai", "unpleasant", "neutral", "unpleasantAI", "neutralAI",
        "UP vs Nt, trained on NA and tested on AI",
    ),
    "train_unpleasantAIvneutralAI_test_unpleasantvneutral": (
        "ai", "natural", "unpleasantAI", "neutralAI", "unpleasant", "neutral",
        "UP vs Nt, trained on AI and tested on NA",
    ),
}

WITHIN_COLORS = {
    "pleasant_vs_neutral": "#1f4e79",
    "pleasantAI_vs_neutralAI": "#8ecae6",
    "unpleasant_vs_neutral": "#b45f06",
    "unpleasantAI_vs_neutralAI": "#f6b26b",
}
CROSS_COLORS = {
    "train_pleasantvneutral_test_pleasantAIvneutralAI": "#1b7837",
    "train_pleasantAIvneutralAI_test_pleasantvneutral": "#a6dba0",
    "train_unpleasantvneutral_test_unpleasantAIvneutralAI": "#8b0000",
    "train_unpleasantAIvneutralAI_test_unpleasantvneutral": "#f4a3a3",
}


def parse_mricrogl_label_file(label_file: str) -> Dict[int, str]:
    labels = {}
    with open(label_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                roi_id = int(float(parts[0]))
            except ValueError:
                continue
            roi_name = parts[1]
            if roi_id > 0 and roi_name != "*.*.*.*.*":
                labels[roi_id] = roi_name
    return labels


def load_stim_names_for_subject(sub: str, runs: Iterable[str], onset_base_dir: str) -> List[str]:
    sub_onset_dir = Path(onset_base_dir) / sub / "LogFiles"
    stim_names = []
    for run_name in runs:
        fpath = sub_onset_dir / f"{run_name}.mat"
        if not fpath.exists():
            alt_name = "Run" + str(int(run_name.replace("Run", "")))
            fpath = sub_onset_dir / f"{alt_name}.mat"
        if not fpath.exists():
            raise FileNotFoundError(f"Cannot find onset file for {sub}, {run_name}: {fpath}")

        mat = scipy.io.loadmat(fpath, squeeze_me=False)
        raw = mat["dataLog"]
        for row in range(1, raw.shape[0]):
            c = raw[row, 1]
            if c.size == 0:
                continue
            if str(c.flat[0]).strip() == "Stim on":
                stim_names.append(str(raw[row, 2].flat[0]).strip())
    return stim_names


def build_stimulus_lookup(cfg: Config) -> Tuple[pd.DataFrame, Dict[str, str]]:
    beta_labels = pd.read_csv(cfg.label_file, header=None, names=["beta_file", "category"])
    stim_template = load_stim_names_for_subject("Sub27", cfg.runs, cfg.onset_base_dir)
    if len(stim_template) != len(beta_labels):
        raise ValueError(f"Sub27 stimulus count ({len(stim_template)}) != beta count ({len(beta_labels)})")
    return beta_labels, dict(zip(stim_template, beta_labels["category"].tolist()))


def make_subject_beta_table(sub: str, cfg: Config, beta_labels: pd.DataFrame, stim_to_category: Dict[str, str]) -> pd.DataFrame:
    stim_names = load_stim_names_for_subject(sub, cfg.runs, cfg.onset_base_dir)
    if len(stim_names) != len(beta_labels):
        raise ValueError(f"{sub}: stimulus count ({len(stim_names)}) != beta count ({len(beta_labels)})")
    missing = [s for s in stim_names if s not in stim_to_category]
    if missing:
        raise KeyError(f"{sub}: {len(missing)} stimuli missing from lookup. First few: {missing[:5]}")
    return pd.DataFrame(
        {
            "beta_file": beta_labels["beta_file"].to_numpy(),
            "category": [stim_to_category[s] for s in stim_names],
            "stimulus": stim_names,
        }
    )


def get_reference_image(cfg: Config, beta_labels: pd.DataFrame) -> nib.Nifti1Image:
    beta_path = Path(cfg.beta_dir) / "Sub1" / str(beta_labels["beta_file"].iloc[0])
    if not beta_path.exists() and Path(str(beta_path) + ".gz").exists():
        beta_path = Path(str(beta_path) + ".gz")
    return nib.load(str(beta_path))


def build_kastner_roi_columns(cfg: Config, ref_img: nib.Nifti1Image) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, nib.Nifti1Image]]:
    atlas_img = nib.load(cfg.kastner_atlas_file)
    atlas_data = np.rint(atlas_img.get_fdata()).astype(np.int32)
    labels = parse_mricrogl_label_file(cfg.kastner_label_file)
    label_to_id = {name: roi_id for roi_id, name in labels.items()}

    roi_flat: Dict[str, np.ndarray] = {}
    roi_imgs: Dict[str, nib.Nifti1Image] = {}
    for roi, members in KASTNER_ROI_GROUPS.items():
        missing = [m for m in members if m not in label_to_id]
        if missing:
            raise ValueError(f"Missing Kastner labels for {roi}: {missing}")
        member_ids = [label_to_id[m] for m in members]
        mask_data = np.isin(atlas_data, member_ids).astype(np.uint8)
        mask_img = nib.Nifti1Image(mask_data, atlas_img.affine)
        resampled = resample_img(
            mask_img,
            target_affine=ref_img.affine,
            target_shape=ref_img.shape,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        flat = np.flatnonzero(resampled.get_fdata().ravel() > 0).astype(np.int64)
        if len(flat) > 0:
            roi_flat[roi] = flat
            roi_imgs[roi] = resampled

    plot_rois = [r for r in KASTNER_PLOT_ROIS if r in roi_flat]
    union_flat = np.unique(np.concatenate([roi_flat[r] for r in plot_rois])).astype(np.int64)
    flat_to_col = {int(flat): i for i, flat in enumerate(union_flat)}
    roi_cols = {
        roi: np.asarray([flat_to_col[int(flat)] for flat in roi_flat[roi]], dtype=np.int32)
        for roi in plot_rois
    }
    return roi_cols, union_flat, roi_imgs


def load_subject_union_betas(sub: str, bl: pd.DataFrame, cfg: Config, union_flat: np.ndarray) -> np.ndarray:
    sub_dir = Path(cfg.beta_dir) / sub
    X = np.empty((len(bl), len(union_flat)), dtype=np.float32)
    for i, beta_file in enumerate(bl["beta_file"].values):
        beta_path = sub_dir / str(beta_file)
        if not beta_path.exists() and Path(str(beta_path) + ".gz").exists():
            beta_path = Path(str(beta_path) + ".gz")
        if not beta_path.exists():
            raise FileNotFoundError(beta_path)
        X[i] = nib.load(str(beta_path)).get_fdata(dtype=np.float32).ravel()[union_flat]
    return X


def split_conditions(X: np.ndarray, bl: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {cond: X[bl.index[bl["category"] == cond].to_numpy()] for cond in ALL_CATEGORIES}


def remove_invalid_voxels(cond_data: Dict[str, np.ndarray], min_voxels: int) -> Tuple[Dict[str, np.ndarray], np.ndarray, str]:
    n_vox = next(iter(cond_data.values())).shape[1]
    valid = np.ones(n_vox, dtype=bool)
    for cond in ALL_CATEGORIES:
        valid &= np.all(np.isfinite(cond_data[cond]), axis=0)
    stacked = np.vstack([cond_data[c] for c in ALL_CATEGORIES])
    valid &= np.isfinite(np.var(stacked, axis=0))
    valid &= np.var(stacked, axis=0) > 0
    if int(valid.sum()) < min_voxels:
        return cond_data, valid, f"too_few_valid_voxels:{int(valid.sum())}"
    return {c: cond_data[c][:, valid].astype(np.float32) for c in ALL_CATEGORIES}, valid, "ok"


def safe_pattern_zscore(X: np.ndarray) -> np.ndarray:
    Xz = zscore(np.asarray(X, dtype=np.float32), axis=1, nan_policy="omit")
    return np.nan_to_num(Xz, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_trial_zscore(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[~np.isfinite(std)] = 1.0
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_trial_zscore(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Xz = (X - mean) / std
    return np.nan_to_num(Xz, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def preprocess_train_test(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_train = safe_pattern_zscore(X_train)
    X_test = safe_pattern_zscore(X_test)
    mean, std = fit_trial_zscore(X_train)
    return apply_trial_zscore(X_train, mean, std), apply_trial_zscore(X_test, mean, std)


def haufe_pattern_from_training_data(X_train: np.ndarray, decision: np.ndarray) -> np.ndarray:
    Xc = X_train.astype(np.float64) - np.mean(X_train, axis=0, keepdims=True)
    dc = decision.astype(np.float64) - np.mean(decision)
    denom = float(np.dot(dc, dc))
    if denom <= 0 or not np.isfinite(denom):
        return np.zeros(X_train.shape[1], dtype=np.float32)
    return (Xc.T @ dc / denom).astype(np.float32)


def make_linear_svm(cfg: Config):
    if cfg.svm_backend == "linear_svc":
        return LinearSVC(
            C=cfg.clf_c,
            dual="auto",
            max_iter=cfg.linear_svc_max_iter,
            random_state=cfg.seed_base,
        )
    if cfg.svm_backend == "svc":
        return SVC(kernel="linear", C=cfg.clf_c)
    raise ValueError(f"Unknown svm_backend: {cfg.svm_backend}")


def select_feature_columns(selector: str, X_train: np.ndarray, y_train: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    n_features = X_train.shape[1]
    if selector == "none":
        return np.arange(n_features, dtype=np.int32), np.full(n_features, np.nan, dtype=np.float32)

    rank_model = make_linear_svm(cfg)
    rank_model.fit(X_train, y_train)
    if selector == "svm_top50":
        scores = np.abs(rank_model.coef_[0]).astype(np.float32)
    elif selector == "haufe_top50":
        haufe = haufe_pattern_from_training_data(X_train, rank_model.decision_function(X_train))
        scores = np.abs(haufe).astype(np.float32)
    else:
        raise ValueError(f"Unknown selector: {selector}")

    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    n_select = max(cfg.min_voxels, int(np.ceil(n_features * cfg.top_fraction)))
    n_select = min(n_select, n_features)
    cols = np.argpartition(scores, -n_select)[-n_select:]
    cols = cols[np.argsort(scores[cols])[::-1]].astype(np.int32)
    return cols, scores


def balanced_indices(n1: int, n2: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    n = min(n1, n2)
    return rng.choice(n1, n, replace=False), rng.choice(n2, n, replace=False)


def averaged_class_patterns(X: np.ndarray, n_groups: int, rng: np.random.RandomState) -> np.ndarray:
    n_groups = max(1, min(n_groups, len(X)))
    shuffled = rng.permutation(len(X))
    chunks = np.array_split(shuffled, n_groups)
    return np.vstack([X[chunk].mean(axis=0) for chunk in chunks]).astype(np.float32)


def erp_style_feature_data(d1: np.ndarray, d0: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    X = np.vstack([d1, d0]).astype(np.float32)
    X = safe_pattern_zscore(X)
    mean, std = fit_trial_zscore(X)
    X = apply_trial_zscore(X, mean, std)
    d1_z = X[: len(d1)]
    d0_z = X[len(d1):]

    rng = np.random.RandomState(cfg.seed_base)
    d1_avg = averaged_class_patterns(d1_z, cfg.n_avg_groups, rng)
    d0_avg = averaged_class_patterns(d0_z, cfg.n_avg_groups, rng)
    X_avg = np.vstack([d1_avg, d0_avg]).astype(np.float32)
    y_avg = np.array([1] * len(d1_avg) + [0] * len(d0_avg), dtype=np.int8)
    return X_avg, y_avg


def fit_score_one_fold(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    selector: str,
    cfg: Config,
    fixed_selected_cols: np.ndarray | None = None,
) -> Tuple[float, np.ndarray]:
    X_train, X_test = preprocess_train_test(X_train_raw, X_test_raw)
    if fixed_selected_cols is None:
        selected_cols, _ = select_feature_columns(selector, X_train, y_train, cfg)
    else:
        selected_cols = fixed_selected_cols
    clf = make_linear_svm(cfg)
    clf.fit(X_train[:, selected_cols], y_train)
    return float(clf.score(X_test[:, selected_cols], y_test)), selected_cols


def fixed_feature_columns(d1: np.ndarray, d0: np.ndarray, selector: str, cfg: Config) -> np.ndarray | None:
    if selector == "none" or cfg.feature_scope != "fixed":
        return None

    if cfg.feature_basis == "single_trial":
        X = np.vstack([d1, d0]).astype(np.float32)
        y = np.array([1] * len(d1) + [0] * len(d0), dtype=np.int8)
        X = safe_pattern_zscore(X)
        mean, std = fit_trial_zscore(X)
        X = apply_trial_zscore(X, mean, std)
    elif cfg.feature_basis == "erp_avg":
        X, y = erp_style_feature_data(d1, d0, cfg)
    else:
        raise ValueError(f"Unknown feature_basis: {cfg.feature_basis}")

    selected_cols, _ = select_feature_columns(selector, X, y, cfg)
    return selected_cols


def add_selection_count(
    selection_counts: Dict[Tuple[str, str, str], np.ndarray],
    selection_denoms: Dict[Tuple[str, str, str], int],
    comp: str,
    selector: str,
    local_cols: np.ndarray,
    n_vox: int,
    analysis: str,
) -> None:
    if selector == "none":
        return
    key = (analysis, comp, selector)
    if key not in selection_counts:
        selection_counts[key] = np.zeros(n_vox, dtype=np.uint16)
        selection_denoms[key] = 0
    selection_counts[key][local_cols] += 1
    selection_denoms[key] += 1


def decode_within_single_trial(
    d1: np.ndarray,
    d0: np.ndarray,
    comp: str,
    source: str,
    roi: str,
    cfg: Config,
) -> Tuple[List[dict], Dict[Tuple[str, str, str], np.ndarray], Dict[Tuple[str, str, str], int]]:
    fold_rows: List[dict] = []
    selection_counts: Dict[Tuple[str, str, str], np.ndarray] = {}
    selection_denoms: Dict[Tuple[str, str, str], int] = {}
    n_vox = d1.shape[1]
    fixed_by_selector = {selector: fixed_feature_columns(d1, d0, selector, cfg) for selector in cfg.selectors}

    for repeat in range(cfg.n_repeats):
        rng = np.random.RandomState(cfg.seed_base + repeat)
        idx1, idx0 = balanced_indices(len(d1), len(d0), rng)
        X = np.vstack([d1[idx1], d0[idx0]]).astype(np.float32)
        y = np.array([1] * len(idx1) + [0] * len(idx0), dtype=np.int8)
        skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed_base + repeat)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            for selector in cfg.selectors:
                acc, selected_cols = fit_score_one_fold(
                    X[train_idx], y[train_idx], X[test_idx], y[test_idx], selector, cfg,
                    fixed_selected_cols=fixed_by_selector[selector],
                )
                fold_rows.append(
                    {
                        "analysis": "within",
                        "subject": None,
                        "roi": roi,
                        "comparison": comp,
                        "source": source,
                        "train_source": source,
                        "test_source": source,
                        "selector": selector,
                        "feature_scope": cfg.feature_scope,
                        "feature_basis": cfg.feature_basis,
                        "repeat": repeat,
                        "fold": fold,
                        "accuracy": acc,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "n_voxels": int(n_vox),
                        "n_selected": int(len(selected_cols)),
                    }
                )
                add_selection_count(selection_counts, selection_denoms, comp, selector, selected_cols, n_vox, "within")
    return fold_rows, selection_counts, selection_denoms


def decode_cross_single_trial(
    train_d1: np.ndarray,
    train_d0: np.ndarray,
    test_d1: np.ndarray,
    test_d0: np.ndarray,
    comp: str,
    train_source: str,
    test_source: str,
    roi: str,
    cfg: Config,
) -> Tuple[List[dict], Dict[Tuple[str, str, str], np.ndarray], Dict[Tuple[str, str, str], int]]:
    fold_rows: List[dict] = []
    selection_counts: Dict[Tuple[str, str, str], np.ndarray] = {}
    selection_denoms: Dict[Tuple[str, str, str], int] = {}
    n_vox = train_d1.shape[1]
    fixed_by_selector = {
        selector: fixed_feature_columns(train_d1, train_d0, selector, cfg)
        for selector in cfg.selectors
    }

    for repeat in range(cfg.n_repeats):
        rng = np.random.RandomState(cfg.seed_base + 10000 + repeat)
        tr_i1, tr_i0 = balanced_indices(len(train_d1), len(train_d0), rng)
        te_i1, te_i0 = balanced_indices(len(test_d1), len(test_d0), rng)
        X_train_source = np.vstack([train_d1[tr_i1], train_d0[tr_i0]]).astype(np.float32)
        y_train_source = np.array([1] * len(tr_i1) + [0] * len(tr_i0), dtype=np.int8)
        X_test = np.vstack([test_d1[te_i1], test_d0[te_i0]]).astype(np.float32)
        y_test = np.array([1] * len(te_i1) + [0] * len(te_i0), dtype=np.int8)

        skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed_base + 10000 + repeat)
        for fold, (train_idx, _) in enumerate(skf.split(X_train_source, y_train_source)):
            for selector in cfg.selectors:
                acc, selected_cols = fit_score_one_fold(
                    X_train_source[train_idx], y_train_source[train_idx], X_test, y_test, selector, cfg,
                    fixed_selected_cols=fixed_by_selector[selector],
                )
                fold_rows.append(
                    {
                        "analysis": "cross",
                        "subject": None,
                        "roi": roi,
                        "comparison": comp,
                        "source": f"{train_source}_to_{test_source}",
                        "train_source": train_source,
                        "test_source": test_source,
                        "selector": selector,
                        "feature_scope": cfg.feature_scope,
                        "feature_basis": cfg.feature_basis,
                        "repeat": repeat,
                        "fold": fold,
                        "accuracy": acc,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(y_test)),
                        "n_voxels": int(n_vox),
                        "n_selected": int(len(selected_cols)),
                    }
                )
                add_selection_count(selection_counts, selection_denoms, comp, selector, selected_cols, n_vox, "cross")
    return fold_rows, selection_counts, selection_denoms


def merge_counts(
    dest_counts: Dict[Tuple[str, str, str], np.ndarray],
    dest_denoms: Dict[Tuple[str, str, str], int],
    src_counts: Dict[Tuple[str, str, str], np.ndarray],
    src_denoms: Dict[Tuple[str, str, str], int],
) -> None:
    for key, arr in src_counts.items():
        if key not in dest_counts:
            dest_counts[key] = arr.copy()
            dest_denoms[key] = src_denoms[key]
        else:
            dest_counts[key] += arr
            dest_denoms[key] += src_denoms[key]


def process_subject(
    sub: str,
    cfg: Config,
    beta_labels: pd.DataFrame,
    stim_to_category: Dict[str, str],
    roi_cols: Dict[str, np.ndarray],
    union_flat: np.ndarray,
    out_dir: Path,
    overwrite: bool,
) -> dict:
    per_subject_dir = out_dir / "per_subject"
    per_subject_dir.mkdir(parents=True, exist_ok=True)
    result_path = per_subject_dir / f"{sub}_kastner_singletrial_results.pkl"
    if result_path.exists() and not overwrite:
        with open(result_path, "rb") as f:
            return pickle.load(f)

    bl = make_subject_beta_table(sub, cfg, beta_labels, stim_to_category)
    X_union = load_subject_union_betas(sub, bl, cfg, union_flat)
    cond_union = split_conditions(X_union, bl)

    fold_rows: List[dict] = []
    status_rows: List[dict] = []
    selection_payload = {}

    for roi in KASTNER_PLOT_ROIS:
        if roi not in roi_cols:
            status_rows.append({"subject": sub, "roi": roi, "status": "missing_roi_mask", "n_valid_voxels": 0})
            continue
        cols = roi_cols[roi]
        cond_raw = {c: cond_union[c][:, cols] for c in ALL_CATEGORIES}
        cond_data, valid, status = remove_invalid_voxels(cond_raw, cfg.min_voxels)
        roi_flat_valid = union_flat[cols][valid]
        status_rows.append({"subject": sub, "roi": roi, "status": status, "n_valid_voxels": int(len(roi_flat_valid))})
        if status != "ok":
            continue

        roi_counts: Dict[Tuple[str, str, str], np.ndarray] = {}
        roi_denoms: Dict[Tuple[str, str, str], int] = {}

        for comp in WITHIN_ORDER:
            source, c1, c0, _ = WITHIN_SPECS[comp]
            rows, counts, denoms = decode_within_single_trial(cond_data[c1], cond_data[c0], comp, source, roi, cfg)
            fold_rows.extend(rows)
            merge_counts(roi_counts, roi_denoms, counts, denoms)

        for comp in CROSS_ORDER:
            train_source, test_source, tr1, tr0, te1, te0, _ = CROSS_SPECS[comp]
            rows, counts, denoms = decode_cross_single_trial(
                cond_data[tr1], cond_data[tr0], cond_data[te1], cond_data[te0],
                comp, train_source, test_source, roi, cfg,
            )
            fold_rows.extend(rows)
            merge_counts(roi_counts, roi_denoms, counts, denoms)

        selection_payload[roi] = {
            "roi_flat_valid": roi_flat_valid.astype(np.int64),
            "counts": roi_counts,
            "denoms": roi_denoms,
        }

    for row in fold_rows:
        row["subject"] = sub

    result = {
        "subject": sub,
        "status": "ok",
        "fold_rows": fold_rows,
        "status_rows": status_rows,
        "selection_payload": selection_payload,
        "condition_counts": bl["category"].value_counts().to_dict(),
    }

    with open(result_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    del X_union, cond_union
    gc.collect()
    return result


def sem(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def p_gt_chance(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2:
        return np.nan
    return float(stats.ttest_1samp(x, 0.5, alternative="greater").pvalue)


def summarize_results(fold_df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = [
        "subject", "analysis", "roi", "comparison", "source", "train_source",
        "test_source", "selector", "feature_scope", "feature_basis",
    ]
    subject_df = (
        fold_df.groupby(group_cols, dropna=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            accuracy_median=("accuracy", "median"),
            n_fits=("accuracy", "size"),
            n_train_mean=("n_train", "mean"),
            n_test_mean=("n_test", "mean"),
            n_voxels=("n_voxels", "first"),
            n_selected=("n_selected", "first"),
        )
        .reset_index()
    )
    subject_df["included_in_group"] = ~subject_df["subject"].isin(cfg.group_exclude_subjects)
    group_cols2 = [
        "analysis", "roi", "comparison", "source", "train_source",
        "test_source", "selector", "feature_scope", "feature_basis",
    ]
    group_df = (
        subject_df[subject_df["included_in_group"]]
        .groupby(group_cols2, dropna=False)
        .agg(
            n_subjects=("subject", "nunique"),
            accuracy_mean=("accuracy_mean", "mean"),
            accuracy_sem=("accuracy_mean", sem),
            accuracy_sd=("accuracy_mean", "std"),
            accuracy_median=("accuracy_mean", "median"),
            p_gt_chance=("accuracy_mean", p_gt_chance),
            n_voxels_mean=("n_voxels", "mean"),
            n_selected_mean=("n_selected", "mean"),
        )
        .reset_index()
    )
    return subject_df, group_df


def mean_sem_from_subject_df(subject_df: pd.DataFrame, analysis: str, comp: str, selector: str, rois: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    means = []
    sems = []
    for roi in rois:
        values = subject_df[
            (subject_df["included_in_group"])
            & (subject_df["analysis"] == analysis)
            & (subject_df["comparison"] == comp)
            & (subject_df["selector"] == selector)
            & (subject_df["roi"] == roi)
        ]["accuracy_mean"].astype(float)
        means.append(float(values.mean()) if len(values) else np.nan)
        sems.append(float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan)
    return np.asarray(means), np.asarray(sems)


def hatch_for_comp(comp: str) -> str | None:
    is_ai_style = ("AI_vs" in comp) or comp.startswith("train_pleasantAI") or comp.startswith("train_unpleasantAI")
    return "///" if is_ai_style else None


def add_percent_guide_lines(ax, low: float = 0.40, high: float = 0.85) -> None:
    for value in np.arange(low, high + 0.001, 0.05):
        is_chance = np.isclose(value, 0.50)
        ax.axhline(
            y=value,
            color="#4A4A4A" if is_chance else "#D0D0D0",
            linestyle="--",
            linewidth=2.0 if is_chance else 0.8,
            alpha=0.95 if is_chance else 0.75,
            zorder=0,
        )


def selector_label(selector: str) -> str:
    return {
        "none": "no feature selection",
        "svm_top50": "SVM-weight selected voxels",
        "haufe_top50": "Haufe-pattern selected voxels",
    }.get(selector, selector)


def plot_same_style(
    subject_df: pd.DataFrame,
    selector: str,
    analysis: str,
    out_file: Path,
    rois: Sequence[str] = KASTNER_PLOT_ROIS,
) -> None:
    if analysis == "within":
        comp_order = WITHIN_ORDER
        colors = WITHIN_COLORS
        labels = {comp: WITHIN_SPECS[comp][3] for comp in WITHIN_ORDER}
        title = "Within Source Decoding"
    elif analysis == "cross":
        comp_order = CROSS_ORDER
        colors = CROSS_COLORS
        labels = {comp: CROSS_SPECS[comp][6] for comp in CROSS_ORDER}
        title = "Cross Source Decoding"
    else:
        raise ValueError(analysis)

    x = np.arange(len(rois))
    width = min(0.18, 0.80 / len(comp_order))
    fig, ax = plt.subplots(figsize=(max(18, len(rois) * 1.15), 7))

    for ci, comp in enumerate(comp_order):
        means, sems = mean_sem_from_subject_df(subject_df, analysis, comp, selector, rois)
        offset = (ci - (len(comp_order) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=sems,
            color=colors[comp],
            label=labels[comp],
            hatch=hatch_for_comp(comp),
            capsize=3,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )

    add_percent_guide_lines(ax)
    ax.set_ylabel("Accuracy", fontsize=24)
    ax.set_xlabel("Region of Interest", fontsize=24)
    ax.set_title(title, fontsize=26, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.tick_params(axis="y", labelsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(rois, rotation=45, ha="right", fontsize=18)
    ax.set_ylim([0.40, 0.85])
    legend = ax.legend(fontsize=15, loc="lower right", frameon=True, facecolor="white", framealpha=0.85, edgecolor="gray")
    legend.get_frame().set_linewidth(0.8)
    plt.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def save_selection_frequency_maps(results: List[dict], cfg: Config, ref_img: nib.Nifti1Image, out_dir: Path) -> None:
    if not cfg.save_selection_frequency_maps:
        return
    map_dir = out_dir / "selected_voxel_frequency_maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    group_accum: Dict[Tuple[str, str, str, str], List[np.ndarray]] = {}
    index_rows = []
    full_size = int(np.prod(ref_img.shape))
    included = set(cfg.subjects) - set(cfg.group_exclude_subjects)

    for result in results:
        sub = result["subject"]
        for roi, payload in result.get("selection_payload", {}).items():
            roi_flat = payload["roi_flat_valid"]
            for key, counts in payload["counts"].items():
                analysis, comp, selector = key
                denom = payload["denoms"][key]
                freq = np.zeros(full_size, dtype=np.float32)
                freq[roi_flat] = counts.astype(np.float32) / float(denom)
                full_key = (roi, analysis, comp, selector)
                if sub in included:
                    group_accum.setdefault(full_key, []).append(freq)
                index_rows.append(
                    {
                        "subject": sub,
                        "roi": roi,
                        "analysis": analysis,
                        "comparison": comp,
                        "selector": selector,
                        "n_selection_records": int(denom),
                    }
                )

    pd.DataFrame(index_rows).to_csv(map_dir / "selection_frequency_map_index.csv", index=False)
    for (roi, analysis, comp, selector), maps in group_accum.items():
        if not maps:
            continue
        mean_freq = np.nanmean(np.vstack(maps), axis=0).reshape(ref_img.shape)
        safe_name = sanitize(f"group_{roi}_{analysis}_{comp}_{selector}")
        img = nib.Nifti1Image(mean_freq.astype(np.float32), ref_img.affine, ref_img.header)
        nib.save(img, str(map_dir / f"{safe_name}_selection_frequency.nii.gz"))


def load_erp_reference_subject_level(cfg: Config) -> pd.DataFrame:
    path = Path(cfg.existing_erp_results)
    if not path.exists():
        return pd.DataFrame()
    with open(path, "rb") as f:
        all_results = pickle.load(f)
    avg_mode = all_results.get("avg_z_mode", "voxel_trial_source_zscore")
    rows = []
    for subject in all_results["subs"]:
        if subject in cfg.group_exclude_subjects:
            continue
        for analysis, result_key, comp_order in [
            ("within", "within_avg", WITHIN_ORDER),
            ("cross", "cross_avg", CROSS_ORDER),
        ]:
            result_dict = all_results[result_key]
            for comp in comp_order:
                for roi in KASTNER_PLOT_ROIS:
                    value = result_dict.get(subject, {}).get(comp, {}).get(avg_mode, {}).get(roi, np.nan)
                    rows.append(
                        {
                            "subject": subject,
                            "analysis": analysis,
                            "roi": roi,
                            "comparison": comp,
                            "selector": "erp_style_avg",
                            "accuracy_mean": value,
                            "included_in_group": True,
                        }
                    )
    return pd.DataFrame(rows)


def save_outputs(results: List[dict], cfg: Config, ref_img: nib.Nifti1Image, out_dir: Path) -> None:
    status_rows = []
    fold_rows = []
    for result in results:
        status_rows.extend(result.get("status_rows", []))
        fold_rows.extend(result.get("fold_rows", []))
    status_df = pd.DataFrame(status_rows)
    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        raise RuntimeError("No fold-level results were produced")

    subject_df, group_df = summarize_results(fold_df, cfg)
    erp_subject_df = load_erp_reference_subject_level(cfg)

    status_df.to_csv(out_dir / "kastner_singletrial_subject_roi_status.csv", index=False)
    fold_df.to_csv(out_dir / "kastner_singletrial_fold_level_results.csv", index=False)
    subject_df.to_csv(out_dir / "kastner_singletrial_subject_level_results.csv", index=False)
    group_df.to_csv(out_dir / "kastner_singletrial_group_summary.csv", index=False)
    if not erp_subject_df.empty:
        erp_subject_df.to_csv(out_dir / "erp_style_reference_subject_level_same_rois.csv", index=False)

    with pd.ExcelWriter(out_dir / "kastner_singletrial_top50_results.xlsx") as writer:
        status_df.to_excel(writer, sheet_name="subject_roi_status", index=False)
        subject_df.to_excel(writer, sheet_name="singletrial_subject", index=False)
        group_df.to_excel(writer, sheet_name="singletrial_group", index=False)
        if not erp_subject_df.empty:
            erp_subject_df.to_excel(writer, sheet_name="erp_reference", index=False)

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    for selector in cfg.selectors:
        plot_same_style(subject_df, selector, "within", plot_dir / f"within_kastner_wang_singletrial_{selector}.png")
        plot_same_style(subject_df, selector, "cross", plot_dir / f"cross_kastner_wang_singletrial_{selector}.png")
    if not erp_subject_df.empty:
        plot_same_style(erp_subject_df, "erp_style_avg", "within", plot_dir / "within_kastner_wang_erp_reference_replotted.png")
        plot_same_style(erp_subject_df, "erp_style_avg", "cross", plot_dir / "cross_kastner_wang_erp_reference_replotted.png")

    save_selection_frequency_maps(results, cfg, ref_img, out_dir)

    with open(out_dir / "kastner_singletrial_top50_results.pkl", "wb") as f:
        pickle.dump(
            {
                "config": asdict(cfg),
                "subject_roi_status": status_df,
                "fold_level": fold_df,
                "subject_level": subject_df,
                "group_summary": group_df,
                "erp_reference_subject_level": erp_subject_df,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", help="Optional subject subset, e.g. Sub1 Sub2")
    parser.add_argument("--selectors", nargs="*", choices=("none", "svm_top50", "haufe_top50"), help="Selectors to run")
    parser.add_argument("--n-repeats", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--top-fraction", type=float, default=None, help="Fraction of voxels to select for top-k selectors")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-frequency-maps", action="store_true")
    parser.add_argument("--svm-backend", choices=("linear_svc", "svc"), default=None)
    parser.add_argument("--linear-svc-max-iter", type=int, default=None)
    parser.add_argument(
        "--feature-scope",
        choices=("nested", "fixed"),
        default=None,
        help="nested selects voxels inside each fold; fixed reuses one top-k voxel set across folds",
    )
    parser.add_argument(
        "--feature-basis",
        choices=("single_trial", "erp_avg"),
        default=None,
        help="For fixed scope, rank voxels from all single trials or from ERP-style averaged trial chunks",
    )
    parser.add_argument(
        "--n-avg-groups",
        type=int,
        default=None,
        help="Number of averaged chunks per class for --feature-basis erp_avg",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    if args.subjects:
        cfg.subjects = tuple(args.subjects)
    if args.selectors:
        cfg.selectors = tuple(args.selectors)
    if args.n_repeats is not None:
        cfg.n_repeats = args.n_repeats
    if args.n_folds is not None:
        cfg.n_folds = args.n_folds
    if args.top_fraction is not None:
        cfg.top_fraction = args.top_fraction
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.no_frequency_maps:
        cfg.save_selection_frequency_maps = False
    if args.svm_backend:
        cfg.svm_backend = args.svm_backend
    if args.linear_svc_max_iter is not None:
        cfg.linear_svc_max_iter = args.linear_svc_max_iter
    if args.feature_scope:
        cfg.feature_scope = args.feature_scope
    if args.feature_basis:
        cfg.feature_basis = args.feature_basis
    if args.n_avg_groups is not None:
        cfg.n_avg_groups = args.n_avg_groups

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    warnings.filterwarnings("ignore", category=FutureWarning)
    beta_labels, stim_to_category = build_stimulus_lookup(cfg)
    ref_img = get_reference_image(cfg, beta_labels)
    roi_cols, union_flat, roi_imgs = build_kastner_roi_columns(cfg, ref_img)
    mask_dir = out_dir / "roi_masks"
    mask_dir.mkdir(exist_ok=True)
    for roi, img in roi_imgs.items():
        if roi in KASTNER_PLOT_ROIS:
            nib.save(img, str(mask_dir / f"{roi}_mask.nii.gz"))

    print(f"Output directory: {out_dir}")
    print(f"Subjects: {len(cfg.subjects)}")
    print(f"Selectors: {cfg.selectors}")
    print(f"Top fraction: {cfg.top_fraction}")
    print(f"SVM backend: {cfg.svm_backend}")
    print(f"Feature scope: {cfg.feature_scope}")
    print(f"Feature basis: {cfg.feature_basis}")
    print(f"ERP avg groups per class: {cfg.n_avg_groups}")
    print(f"Kastner plot ROIs: {', '.join([r for r in KASTNER_PLOT_ROIS if r in roi_cols])}")
    print(f"Union ROI voxels loaded per subject: {len(union_flat):,}")
    print(f"Repeats x folds: {cfg.n_repeats} x {cfg.n_folds}")

    if args.n_jobs == 1:
        results = [
            process_subject(sub, cfg, beta_labels, stim_to_category, roi_cols, union_flat, out_dir, args.overwrite)
            for sub in tqdm(cfg.subjects, desc="Subjects")
        ]
    else:
        results = Parallel(n_jobs=args.n_jobs, prefer="processes")(
            delayed(process_subject)(sub, cfg, beta_labels, stim_to_category, roi_cols, union_flat, out_dir, args.overwrite)
            for sub in cfg.subjects
        )

    save_outputs(results, cfg, ref_img, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
