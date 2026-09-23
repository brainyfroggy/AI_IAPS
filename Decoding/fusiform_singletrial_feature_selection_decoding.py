"""Balanced Fusiform single-trial decoding with nested top-50 feature selection.

This script is a first standalone version of the leakage-safe single-trial
analysis requested after the ROI Avg(random)/ERP-style decoding work. It keeps
the existing subject/log-file labeling conventions, starts with bilateral AAL3
Fusiform, and compares:

- no feature selection
- top 50% voxels selected by absolute training-fold SVM weight
- top 50% voxels selected by absolute training-fold Haufe pattern

Feature selection is always fit on the current training fold only.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io
from joblib import Parallel, delayed
from nilearn.image import resample_img
from scipy import stats
from scipy.stats import zscore
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm


PROJECT_ROOT = Path(r"N:\Experimental_Data\yujunchen\projects\AI_IAPS")
DECODING_ROOT = PROJECT_ROOT / "Decoding"


@dataclass
class Config:
    beta_dir: str = str(PROJECT_ROOT / "GLM_singletrial" / "betas")
    label_file: str = str(PROJECT_ROOT / "GLM_singletrial" / "beta_groups.csv")
    onset_base_dir: str = r"N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\DataRecording"
    aal3_atlas_file: str = r"N:\Experimental_Data\yujunchen\projects\data\masks\AAL3\AAL3v1.nii.gz"
    aal3_label_file: str = r"C:\MRIcroGL\Resources\atlas\AAL3v1.nii.txt"
    output_dir: str = str(DECODING_ROOT / "results" / "fusiform_singletrial_top50_feature_selection")
    existing_erp_summary_csv: str = str(
        DECODING_ROOT
        / "results"
        / "fusiform_cross_vs_haufe_discrepancy_audit"
        / "fusiform_summary_cross_vs_haufe_audit.csv"
    )
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
    n_repeats: int = 20
    n_folds: int = 4
    seed_base: int = 42
    top_fraction: float = 0.50
    min_voxels: int = 10
    roi_name: str = "Fusiform"
    clf_c: float = 1.0


NATURAL_CATS = ("pleasant", "neutral", "unpleasant")
AI_CATS = ("pleasantAI", "neutralAI", "unpleasantAI")
ALL_CATEGORIES = NATURAL_CATS + AI_CATS

CONTRASTS = (
    {
        "contrast_family": "pleasant_vs_neutral",
        "label": "Pleasant vs Neutral",
        "natural": ("pleasant", "neutral"),
        "ai": ("pleasantAI", "neutralAI"),
    },
    {
        "contrast_family": "unpleasant_vs_neutral",
        "label": "Unpleasant vs Neutral",
        "natural": ("unpleasant", "neutral"),
        "ai": ("unpleasantAI", "neutralAI"),
    },
)

SELECTORS = ("none", "svm_top50", "haufe_top50")


def parse_mricrogl_label_file(label_file: str) -> Dict[int, str]:
    labels: Dict[int, str] = {}
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
    stim_names: List[str] = []

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
    template_stim = load_stim_names_for_subject("Sub27", cfg.runs, cfg.onset_base_dir)
    if len(template_stim) != len(beta_labels):
        raise ValueError(
            f"Sub27 stimulus count ({len(template_stim)}) != beta label count ({len(beta_labels)})"
        )
    return beta_labels, dict(zip(template_stim, beta_labels["category"].tolist()))


def make_subject_beta_table(
    sub: str, cfg: Config, beta_labels: pd.DataFrame, stim_to_category: Dict[str, str]
) -> pd.DataFrame:
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
    first_beta = Path(cfg.beta_dir) / "Sub1" / str(beta_labels["beta_file"].iloc[0])
    if not first_beta.exists() and Path(str(first_beta) + ".gz").exists():
        first_beta = Path(str(first_beta) + ".gz")
    return nib.load(str(first_beta))


def build_bilateral_fusiform_mask(cfg: Config, ref_img: nib.Nifti1Image) -> Tuple[nib.Nifti1Image, np.ndarray]:
    labels = parse_mricrogl_label_file(cfg.aal3_label_file)
    name_to_id = {name: roi_id for roi_id, name in labels.items()}
    missing = [name for name in ("Fusiform_L", "Fusiform_R") if name not in name_to_id]
    if missing:
        raise ValueError(f"Missing AAL3 Fusiform labels: {missing}")

    atlas_img = nib.load(cfg.aal3_atlas_file)
    atlas_data = np.rint(atlas_img.get_fdata()).astype(np.int32)
    fusiform_data = np.isin(atlas_data, [name_to_id["Fusiform_L"], name_to_id["Fusiform_R"]])
    fusiform_img = nib.Nifti1Image(fusiform_data.astype(np.uint8), atlas_img.affine)
    resampled = resample_img(
        fusiform_img,
        target_affine=ref_img.affine,
        target_shape=ref_img.shape,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    voxel_idx = np.flatnonzero(resampled.get_fdata().ravel() > 0).astype(np.int64)
    if len(voxel_idx) < cfg.min_voxels:
        raise ValueError(f"Fusiform mask has only {len(voxel_idx)} voxels after resampling")
    return resampled, voxel_idx


def load_subject_roi_betas(sub: str, bl: pd.DataFrame, cfg: Config, voxel_idx: np.ndarray) -> np.ndarray:
    sub_dir = Path(cfg.beta_dir) / sub
    X = np.empty((len(bl), len(voxel_idx)), dtype=np.float32)
    for i, beta_file in enumerate(bl["beta_file"].values):
        beta_path = sub_dir / str(beta_file)
        if not beta_path.exists() and Path(str(beta_path) + ".gz").exists():
            beta_path = Path(str(beta_path) + ".gz")
        if not beta_path.exists():
            raise FileNotFoundError(beta_path)
        X[i] = nib.load(str(beta_path)).get_fdata(dtype=np.float32).ravel()[voxel_idx]
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


def select_feature_columns(
    selector: str, X_train: np.ndarray, y_train: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, np.ndarray]:
    n_features = X_train.shape[1]
    if selector == "none":
        cols = np.arange(n_features, dtype=np.int32)
        return cols, np.full(n_features, np.nan, dtype=np.float32)

    rank_model = SVC(kernel="linear", C=cfg.clf_c)
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


def fit_score_one_fold(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    selector: str,
    cfg: Config,
) -> Tuple[float, np.ndarray, np.ndarray]:
    X_train, X_test = preprocess_train_test(X_train_raw, X_test_raw)
    selected_cols, score_vector = select_feature_columns(selector, X_train, y_train, cfg)
    clf = SVC(kernel="linear", C=cfg.clf_c)
    clf.fit(X_train[:, selected_cols], y_train)
    acc = float(clf.score(X_test[:, selected_cols], y_test))
    return acc, selected_cols, score_vector


def decode_within_single_trial(
    d1: np.ndarray,
    d2: np.ndarray,
    source: str,
    contrast: Dict[str, object],
    cfg: Config,
) -> Tuple[List[dict], List[dict]]:
    fold_rows: List[dict] = []
    selection_records: List[dict] = []

    for repeat in range(cfg.n_repeats):
        rng = np.random.RandomState(cfg.seed_base + repeat)
        idx1, idx2 = balanced_indices(len(d1), len(d2), rng)
        X = np.vstack([d1[idx1], d2[idx2]]).astype(np.float32)
        y = np.array([1] * len(idx1) + [0] * len(idx2), dtype=np.int8)
        skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed_base + repeat)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for selector in SELECTORS:
                acc, selected_cols, score_vector = fit_score_one_fold(
                    X_train_raw, y_train, X_test_raw, y_test, selector, cfg
                )
                row = {
                    "analysis": "within",
                    "subject": None,
                    "roi": cfg.roi_name,
                    "contrast_family": contrast["contrast_family"],
                    "label": contrast["label"],
                    "source": source,
                    "train_source": source,
                    "test_source": source,
                    "train_condition_1": contrast[source][0],
                    "train_condition_0": contrast[source][1],
                    "test_condition_1": contrast[source][0],
                    "test_condition_0": contrast[source][1],
                    "selector": selector,
                    "repeat": repeat,
                    "fold": fold,
                    "accuracy": acc,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_voxels": int(X.shape[1]),
                    "n_selected": int(len(selected_cols)),
                }
                fold_rows.append(row)
                if selector != "none":
                    selection_records.append({**row, "selected_cols": selected_cols, "scores": score_vector})

    return fold_rows, selection_records


def decode_cross_single_trial(
    train_d1: np.ndarray,
    train_d2: np.ndarray,
    test_d1: np.ndarray,
    test_d2: np.ndarray,
    train_source: str,
    test_source: str,
    contrast: Dict[str, object],
    cfg: Config,
) -> Tuple[List[dict], List[dict]]:
    fold_rows: List[dict] = []
    selection_records: List[dict] = []

    for repeat in range(cfg.n_repeats):
        rng = np.random.RandomState(cfg.seed_base + 10000 + repeat)
        tr_i1, tr_i2 = balanced_indices(len(train_d1), len(train_d2), rng)
        te_i1, te_i2 = balanced_indices(len(test_d1), len(test_d2), rng)

        X_train_source = np.vstack([train_d1[tr_i1], train_d2[tr_i2]]).astype(np.float32)
        y_train_source = np.array([1] * len(tr_i1) + [0] * len(tr_i2), dtype=np.int8)
        X_test_raw = np.vstack([test_d1[te_i1], test_d2[te_i2]]).astype(np.float32)
        y_test = np.array([1] * len(te_i1) + [0] * len(te_i2), dtype=np.int8)

        skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed_base + 10000 + repeat)
        for fold, (train_idx, _) in enumerate(skf.split(X_train_source, y_train_source)):
            X_train_raw = X_train_source[train_idx]
            y_train = y_train_source[train_idx]

            for selector in SELECTORS:
                acc, selected_cols, score_vector = fit_score_one_fold(
                    X_train_raw, y_train, X_test_raw, y_test, selector, cfg
                )
                row = {
                    "analysis": "cross",
                    "subject": None,
                    "roi": cfg.roi_name,
                    "contrast_family": contrast["contrast_family"],
                    "label": contrast["label"],
                    "source": f"{train_source}_to_{test_source}",
                    "train_source": train_source,
                    "test_source": test_source,
                    "train_condition_1": contrast[train_source][0],
                    "train_condition_0": contrast[train_source][1],
                    "test_condition_1": contrast[test_source][0],
                    "test_condition_0": contrast[test_source][1],
                    "selector": selector,
                    "repeat": repeat,
                    "fold": fold,
                    "accuracy": acc,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(y_test)),
                    "n_voxels": int(X_train_source.shape[1]),
                    "n_selected": int(len(selected_cols)),
                }
                fold_rows.append(row)
                if selector != "none":
                    selection_records.append({**row, "selected_cols": selected_cols, "scores": score_vector})

    return fold_rows, selection_records


def process_subject(
    sub: str,
    cfg: Config,
    beta_labels: pd.DataFrame,
    stim_to_category: Dict[str, str],
    voxel_idx_full: np.ndarray,
    out_dir: Path,
    overwrite: bool = False,
) -> dict:
    per_subject_dir = out_dir / "per_subject"
    per_subject_dir.mkdir(parents=True, exist_ok=True)
    result_path = per_subject_dir / f"{sub}_results.pkl"
    if result_path.exists() and not overwrite:
        with open(result_path, "rb") as f:
            return pickle.load(f)

    bl = make_subject_beta_table(sub, cfg, beta_labels, stim_to_category)
    X = load_subject_roi_betas(sub, bl, cfg, voxel_idx_full)
    cond_raw = split_conditions(X, bl)
    cond_data, valid_local, status = remove_invalid_voxels(cond_raw, cfg.min_voxels)
    voxel_idx_valid = voxel_idx_full[valid_local]

    if status != "ok":
        result = {
            "subject": sub,
            "status": status,
            "fold_rows": [],
            "selection_records": [],
            "voxel_idx_valid": voxel_idx_valid,
            "condition_counts": bl["category"].value_counts().to_dict(),
        }
        with open(result_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        return result

    fold_rows: List[dict] = []
    selection_records: List[dict] = []

    for contrast in CONTRASTS:
        for source in ("natural", "ai"):
            c1, c0 = contrast[source]
            rows, records = decode_within_single_trial(cond_data[c1], cond_data[c0], source, contrast, cfg)
            fold_rows.extend(rows)
            selection_records.extend(records)

        rows, records = decode_cross_single_trial(
            cond_data[contrast["natural"][0]],
            cond_data[contrast["natural"][1]],
            cond_data[contrast["ai"][0]],
            cond_data[contrast["ai"][1]],
            "natural",
            "ai",
            contrast,
            cfg,
        )
        fold_rows.extend(rows)
        selection_records.extend(records)

        rows, records = decode_cross_single_trial(
            cond_data[contrast["ai"][0]],
            cond_data[contrast["ai"][1]],
            cond_data[contrast["natural"][0]],
            cond_data[contrast["natural"][1]],
            "ai",
            "natural",
            contrast,
            cfg,
        )
        fold_rows.extend(rows)
        selection_records.extend(records)

    for row in fold_rows:
        row["subject"] = sub
    for record in selection_records:
        record["subject"] = sub

    result = {
        "subject": sub,
        "status": "ok",
        "fold_rows": fold_rows,
        "selection_records": selection_records,
        "voxel_idx_valid": voxel_idx_valid,
        "condition_counts": bl["category"].value_counts().to_dict(),
    }

    save_subject_selection_npz(sub, selection_records, voxel_idx_valid, out_dir)
    with open(result_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    del X, cond_raw, cond_data
    gc.collect()
    return result


def record_context_key(record: dict) -> str:
    return "|".join(
        [
            str(record["analysis"]),
            str(record["contrast_family"]),
            str(record["train_source"]),
            str(record["test_source"]),
            str(record["selector"]),
        ]
    )


def save_subject_selection_npz(
    sub: str, selection_records: List[dict], voxel_idx_valid: np.ndarray, out_dir: Path
) -> None:
    selection_dir = out_dir / "selected_voxels"
    selection_dir.mkdir(parents=True, exist_ok=True)
    if not selection_records:
        return

    n_records = len(selection_records)
    n_vox = len(voxel_idx_valid)
    mask_matrix = np.zeros((n_records, n_vox), dtype=np.uint8)
    score_matrix = np.zeros((n_records, n_vox), dtype=np.float32)
    metadata_rows = []

    for i, record in enumerate(selection_records):
        cols = record["selected_cols"]
        mask_matrix[i, cols] = 1
        score_matrix[i] = record["scores"]
        metadata_rows.append(
            {
                key: value
                for key, value in record.items()
                if key not in {"selected_cols", "scores"}
            }
        )
        metadata_rows[-1]["selection_record"] = i
        metadata_rows[-1]["context_key"] = record_context_key(record)

    np.savez_compressed(
        selection_dir / f"{sub}_selected_voxel_records.npz",
        voxel_idx_valid=voxel_idx_valid.astype(np.int64),
        selected_mask_matrix=mask_matrix,
        selector_score_matrix=score_matrix,
    )
    pd.DataFrame(metadata_rows).to_csv(selection_dir / f"{sub}_selection_record_index.csv", index=False)


def summarize_results(fold_df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = [
        "subject", "analysis", "roi", "contrast_family", "label", "source",
        "train_source", "test_source", "selector",
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

    def sem(x: pd.Series) -> float:
        x = x.dropna()
        return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan

    def p_gt_chance(x: pd.Series) -> float:
        x = x.dropna()
        if len(x) < 2:
            return np.nan
        return float(stats.ttest_1samp(x, 0.5, alternative="greater").pvalue)

    group_cols2 = [
        "analysis", "roi", "contrast_family", "label", "source",
        "train_source", "test_source", "selector",
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


def save_frequency_maps(
    results: List[dict],
    cfg: Config,
    ref_img: nib.Nifti1Image,
    out_dir: Path,
) -> None:
    freq_dir = out_dir / "selected_voxel_frequency_maps"
    freq_dir.mkdir(parents=True, exist_ok=True)

    accum: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    subject_accum_rows = []

    for result in results:
        if result.get("status") != "ok":
            continue
        sub = result["subject"]
        voxel_idx_valid = result["voxel_idx_valid"]
        for record in result["selection_records"]:
            key = record_context_key(record)
            subject_key = f"{sub}|{key}"
            selected = np.zeros(len(voxel_idx_valid), dtype=np.float32)
            selected[record["selected_cols"]] = 1.0

            if subject_key not in accum:
                accum[subject_key] = np.zeros(int(np.prod(ref_img.shape)), dtype=np.float32)
                counts[subject_key] = 0
            accum[subject_key][voxel_idx_valid] += selected
            counts[subject_key] += 1

    group_contexts: Dict[str, List[np.ndarray]] = {}
    for subject_key, count in counts.items():
        sub, analysis, contrast, train_source, test_source, selector = subject_key.split("|")
        full_freq = accum[subject_key] / float(count)
        context_key = "|".join([analysis, contrast, train_source, test_source, selector])
        group_contexts.setdefault(context_key, []).append(full_freq)
        subject_accum_rows.append(
            {
                "subject": sub,
                "analysis": analysis,
                "contrast_family": contrast,
                "train_source": train_source,
                "test_source": test_source,
                "selector": selector,
                "n_selection_records": count,
            }
        )

    pd.DataFrame(subject_accum_rows).to_csv(freq_dir / "frequency_map_index.csv", index=False)

    for context_key, maps in group_contexts.items():
        analysis, contrast, train_source, test_source, selector = context_key.split("|")
        safe_name = f"{analysis}_{contrast}_{train_source}_to_{test_source}_{selector}"
        stack = np.vstack(maps)
        mean_freq = np.nanmean(stack, axis=0).reshape(ref_img.shape)
        freq_img = nib.Nifti1Image(mean_freq.astype(np.float32), ref_img.affine, ref_img.header)
        nib.save(freq_img, str(freq_dir / f"group_{safe_name}_selection_frequency.nii.gz"))

        mask50 = (mean_freq >= 0.50).astype(np.uint8)
        mask_img = nib.Nifti1Image(mask50, ref_img.affine, ref_img.header)
        nib.save(mask_img, str(freq_dir / f"group_{safe_name}_frequency_ge50_mask.nii.gz"))


def load_erp_reference(cfg: Config) -> pd.DataFrame:
    path = Path(cfg.existing_erp_summary_csv)
    if not path.exists():
        return pd.DataFrame()
    erp = pd.read_csv(path)
    rows = []
    for _, row in erp.iterrows():
        for source, col in [
            ("natural", "within_natural_accuracy_mean"),
            ("ai", "within_ai_accuracy_mean"),
            ("natural_to_ai", "cross_natural_to_ai_accuracy_mean"),
            ("ai_to_natural", "cross_ai_to_natural_accuracy_mean"),
        ]:
            analysis = "within" if source in {"natural", "ai"} else "cross"
            if source == "natural_to_ai":
                train_source, test_source = "natural", "ai"
            elif source == "ai_to_natural":
                train_source, test_source = "ai", "natural"
            else:
                train_source = test_source = source
            rows.append(
                {
                    "analysis": analysis,
                    "contrast_family": row["contrast_family"],
                    "label": row["label"],
                    "source": source,
                    "train_source": train_source,
                    "test_source": test_source,
                    "selector": "erp_style_avg",
                    "accuracy_mean": row[col],
                    "accuracy_sem": row.get(col.replace("_mean", "_sem"), np.nan),
                    "n_subjects": row.get("n_subjects", np.nan),
                }
            )
    return pd.DataFrame(rows)


def make_plots(group_df: pd.DataFrame, cfg: Config, out_dir: Path) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    erp_ref = load_erp_reference(cfg)
    plot_df = group_df.copy()
    plot_df["series"] = plot_df["selector"].map(
        {
            "none": "Single trial: no selection",
            "svm_top50": "Single trial: SVM top 50%",
            "haufe_top50": "Single trial: Haufe top 50%",
        }
    )
    plot_cols = [
        "analysis", "contrast_family", "label", "source", "train_source", "test_source",
        "series", "accuracy_mean", "accuracy_sem", "n_subjects",
    ]
    plot_df = plot_df[plot_cols]
    if not erp_ref.empty:
        erp_plot = erp_ref.rename(columns={"selector": "series"})[plot_cols]
        erp_plot["series"] = "ERP-style Avg(random)"
        plot_df = pd.concat([plot_df, erp_plot], ignore_index=True)

    series_order = [
        "ERP-style Avg(random)",
        "Single trial: no selection",
        "Single trial: SVM top 50%",
        "Single trial: Haufe top 50%",
    ]
    colors = {
        "ERP-style Avg(random)": "#595959",
        "Single trial: no selection": "#2f7ebc",
        "Single trial: SVM top 50%": "#2c9c69",
        "Single trial: Haufe top 50%": "#c76f2b",
    }

    for analysis in ("within", "cross"):
        sub_df = plot_df[plot_df["analysis"] == analysis].copy()
        if sub_df.empty:
            continue
        contexts = (
            sub_df[["contrast_family", "label", "source", "train_source", "test_source"]]
            .drop_duplicates()
            .sort_values(["contrast_family", "train_source", "test_source"])
            .to_dict("records")
        )
        fig, axes = plt.subplots(
            len(contexts),
            1,
            figsize=(9, max(3.1, 2.55 * len(contexts))),
            sharex=True,
            constrained_layout=True,
        )
        if len(contexts) == 1:
            axes = [axes]

        for ax, ctx in zip(axes, contexts):
            ctx_df = sub_df[
                (sub_df["contrast_family"] == ctx["contrast_family"])
                & (sub_df["train_source"] == ctx["train_source"])
                & (sub_df["test_source"] == ctx["test_source"])
            ]
            x = np.arange(len(series_order))
            means = []
            sems = []
            for series in series_order:
                row = ctx_df[ctx_df["series"] == series]
                means.append(float(row["accuracy_mean"].iloc[0]) if len(row) else np.nan)
                sems.append(float(row["accuracy_sem"].iloc[0]) if len(row) else np.nan)
            ax.bar(
                x,
                means,
                yerr=sems,
                color=[colors[s] for s in series_order],
                edgecolor="#222222",
                linewidth=0.6,
                capsize=3,
            )
            ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1)
            ax.set_ylim(0.35, 1.0)
            ax.set_ylabel("Accuracy")
            title_source = (
                ctx["source"]
                if analysis == "within"
                else f"{ctx['train_source']} to {ctx['test_source']}"
            )
            ax.set_title(f"{ctx['label']} | {title_source}", fontsize=11)
            ax.grid(axis="y", alpha=0.22)
            ax.set_xticks(x)
            ax.set_xticklabels(series_order, rotation=20, ha="right")

        fig.suptitle(f"Bilateral Fusiform {analysis.capitalize()} Decoding", fontsize=13)
        fig.savefig(plot_dir / f"fusiform_singletrial_feature_selection_vs_erp_{analysis}.png", dpi=180)
        plt.close(fig)

    compact = group_df.copy()
    for analysis in ("within", "cross"):
        sub_df = compact[compact["analysis"] == analysis]
        if sub_df.empty:
            continue
        pivot = sub_df.pivot_table(
            index=["contrast_family", "train_source", "test_source"],
            columns="selector",
            values="accuracy_mean",
        )
        fig, ax = plt.subplots(figsize=(8.5, max(3, 0.55 * len(pivot) + 1.6)), constrained_layout=True)
        im = ax.imshow(pivot[["none", "svm_top50", "haufe_top50"]].to_numpy(), vmin=0.45, vmax=0.95, cmap="viridis")
        ax.set_yticks(np.arange(len(pivot)))
        ax.set_yticklabels([f"{a} | {b}->{c}" for a, b, c in pivot.index], fontsize=9)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(["No selection", "SVM top 50%", "Haufe top 50%"], rotation=20, ha="right")
        for i in range(len(pivot)):
            for j, value in enumerate(pivot[["none", "svm_top50", "haufe_top50"]].to_numpy()[i]):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value < 0.7 else "black")
        ax.set_title(f"Single-trial Fusiform {analysis} accuracy")
        fig.colorbar(im, ax=ax, label="Accuracy")
        fig.savefig(plot_dir / f"fusiform_singletrial_feature_selection_heatmap_{analysis}.png", dpi=180)
        plt.close(fig)


def write_outputs(results: List[dict], cfg: Config, ref_img: nib.Nifti1Image, out_dir: Path) -> None:
    fold_rows = []
    status_rows = []
    for result in results:
        status_rows.append(
            {
                "subject": result["subject"],
                "status": result["status"],
                "n_valid_voxels": int(len(result.get("voxel_idx_valid", []))),
                "condition_counts": json.dumps(result.get("condition_counts", {}), sort_keys=True),
            }
        )
        fold_rows.extend(result.get("fold_rows", []))

    pd.DataFrame(status_rows).to_csv(out_dir / "subject_processing_status.csv", index=False)
    if not fold_rows:
        raise RuntimeError("No fold-level decoding rows were produced")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "fusiform_singletrial_fold_level_results.csv", index=False)

    subject_df, group_df = summarize_results(fold_df, cfg)
    subject_df.to_csv(out_dir / "fusiform_singletrial_subject_level_results.csv", index=False)
    group_df.to_csv(out_dir / "fusiform_singletrial_group_summary.csv", index=False)

    with pd.ExcelWriter(out_dir / "fusiform_singletrial_feature_selection_results.xlsx") as writer:
        pd.DataFrame(status_rows).to_excel(writer, sheet_name="subject_status", index=False)
        subject_df.to_excel(writer, sheet_name="subject_level", index=False)
        group_df.to_excel(writer, sheet_name="group_summary", index=False)
        load_erp_reference(cfg).to_excel(writer, sheet_name="erp_reference", index=False)

    save_frequency_maps(results, cfg, ref_img, out_dir)
    make_plots(group_df, cfg, out_dir)

    with open(out_dir / "fusiform_singletrial_feature_selection_results.pkl", "wb") as f:
        pickle.dump(
            {
                "config": asdict(cfg),
                "subject_status": status_rows,
                "fold_level": fold_df,
                "subject_level": subject_df,
                "group_summary": group_df,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", help="Optional subject subset, e.g. Sub1 Sub2")
    parser.add_argument("--n-repeats", type=int, default=None, help="Override the default repeat count")
    parser.add_argument("--n-folds", type=int, default=None, help="Override the default fold count")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel subject jobs")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite per-subject checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    if args.subjects:
        cfg.subjects = tuple(args.subjects)
    if args.n_repeats is not None:
        cfg.n_repeats = args.n_repeats
    if args.n_folds is not None:
        cfg.n_folds = args.n_folds
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    warnings.filterwarnings("ignore", category=FutureWarning)
    beta_labels, stim_to_category = build_stimulus_lookup(cfg)
    ref_img = get_reference_image(cfg, beta_labels)
    fusiform_mask, voxel_idx_full = build_bilateral_fusiform_mask(cfg, ref_img)
    nib.save(fusiform_mask, str(out_dir / "bilateral_fusiform_roi_mask.nii.gz"))

    print(f"Output directory: {out_dir}")
    print(f"Subjects: {len(cfg.subjects)}")
    print(f"Bilateral Fusiform voxels before subject validity filtering: {len(voxel_idx_full):,}")
    print(f"Repeats x folds: {cfg.n_repeats} x {cfg.n_folds}")

    if args.n_jobs == 1:
        results = [
            process_subject(sub, cfg, beta_labels, stim_to_category, voxel_idx_full, out_dir, args.overwrite)
            for sub in tqdm(cfg.subjects, desc="Subjects")
        ]
    else:
        results = Parallel(n_jobs=args.n_jobs, prefer="processes")(
            delayed(process_subject)(sub, cfg, beta_labels, stim_to_category, voxel_idx_full, out_dir, args.overwrite)
            for sub in cfg.subjects
        )

    write_outputs(results, cfg, ref_img, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
