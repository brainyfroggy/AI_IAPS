#!/usr/bin/env python3
"""Phase C of the Stelzer et al. 2013 / Bo et al. 2021 permutation replication.

1. Aggregate all Phase B (subject, ROI) CSVs into subject_null_pools.npz -- the reusable
   artifact (30 subjects x 17 ROIs x 8 contrasts x 100 shuffles). This is what's expensive
   to build (Phase A+B, ~3h); everything below is seconds of arithmetic on top of it, so
   any future change to alpha, resampling count, or test statistic can rerun from here
   without repeating Phase A/B.

2. Group-level resampling, exactly as Bo et al. 2021 describe it: for each of the 100,000
   rounds, draw ONE value at random from each of the 30 subjects' null pools and average
   -> one group-level null draw. Vectorized: no Python-level loop over the 100,000 rounds.

3. Per cell (17 ROI x 8 contrast = 136), report:
   - observed accuracy (pulled from the existing 30-repeat production result, unchanged)
   - null mean/SD of the group-resampled distribution
   - threshold accuracies at p=0.05 / 0.01 / 0.001 (the paper's own reporting convention)
   - raw empirical p-value (one-tailed, "how often does resampled noise beat the real
     result", with the standard +1 correction so p is never exactly 0)
   - BH-FDR q-value, computed from the raw p-values within each contrast across its 17 ROIs
   - boolean significance flags at 0.05/0.01/0.001 for BOTH raw p and FDR q, so any
     significance level can be selected after the fact without rerunning anything
   - a conservativeness diagnostic: the observed statistic averages 30 CV repeats (low
     variance) but each null draw is a SINGLE CV pass (higher variance) -- this reports
     the ratio so the reader can judge how much power was traded away for fidelity to the
     paper's literal procedure.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import ROI_ORDER, CONTRASTS  # noqa: E402
from aal3_common import load_aal3_kept_regions  # noqa: E402

N_GROUP_DRAWS = 100_000


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def aggregate_null_pools(null_pools_dir: Path, subjects: list[int],
                          roi_order: tuple[str, ...] = ROI_ORDER,
                          contrast_order: tuple[str, ...] | None = None) -> dict[tuple[str, str], np.ndarray]:
    """Returns {(roi, contrast): (n_subjects, n_perms) array}, subject order == `subjects`."""
    if contrast_order is None:
        contrast_order = tuple(c[0] for c in CONTRASTS)
    files = sorted(glob.glob(str(null_pools_dir / "sub*_*.csv")))
    print(f"found {len(files)} per-(subject,roi) null-pool files")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["subnum"] = df["subject"].str.replace("Sub", "").astype(int)

    pools: dict[tuple[str, str], np.ndarray] = {}
    for roi in roi_order:
        for contrast in contrast_order:
            sub = df[(df.roi == roi) & (df.contrast == contrast)]
            mat = np.full((len(subjects), 100), np.nan, dtype=np.float32)
            for i, s in enumerate(subjects):
                vals = sub[sub.subnum == s].sort_values("permutation_index")["null_accuracy"].to_numpy()
                if len(vals) != 100:
                    raise RuntimeError(f"sub{s} {roi} {contrast}: expected 100 perms, got {len(vals)}")
                mat[i] = vals
            pools[(roi, contrast)] = mat
    return pools


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--null-pools-dir", type=Path,
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/stelzer_permutation/null_pools"))
    ap.add_argument("--observed-csv", type=Path,
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/roi_decoding_random10fold/group/group_roi_stats.csv"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/stelzer_permutation/group"))
    ap.add_argument("--n-draws", type=int, default=N_GROUP_DRAWS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subjects", type=int, nargs="+",
                    default=[1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                             24, 25, 26, 27, 28, 29, 30, 31, 33, 34])
    ap.add_argument("--rois", nargs="+", default=None)
    ap.add_argument("--roi-manifest", type=Path, default=None,
                    help="AAL3-style kept-regions CSV (id, roi_name, kept); overrides --rois")
    ap.add_argument("--contrast-kind", choices=["within", "cross"], default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.roi_manifest is not None:
        rois = [name for _rid, name in load_aal3_kept_regions(args.roi_manifest)]
    elif args.rois is not None:
        rois = args.rois
    else:
        rois = list(ROI_ORDER)
    contrast_order = tuple(c[0] for c in CONTRASTS if args.contrast_kind is None or c[1] == args.contrast_kind)

    pools = aggregate_null_pools(args.null_pools_dir, args.subjects, tuple(rois), contrast_order)
    n_subjects = len(args.subjects)

    # persist the reusable artifact
    npz_payload = {f"{roi}__{contrast}": mat for (roi, contrast), mat in pools.items()}
    np.savez_compressed(args.out_dir / "subject_null_pools.npz", **npz_payload)
    print(f"saved subject_null_pools.npz ({len(pools)} cells x {n_subjects} subjects x 100 perms)")

    observed = pd.read_csv(args.observed_csv)
    observed = observed.set_index(["contrast", "roi"])

    rng = np.random.default_rng(args.seed)
    rows = []
    for contrast in contrast_order:
        pvals_raw = []
        cell_rows = []
        for roi in rois:
            null_pool = pools[(roi, contrast)]  # (n_subjects, 100)
            obs_row = observed.loc[(contrast, roi)]
            obs_acc = float(obs_row["mean_accuracy"])
            obs_std_across_repeats = None  # not stored in group_roi_stats.csv; see subject file if needed

            # vectorized group-level resampling: draw one perm index per subject, n_draws times
            idx = rng.integers(0, 100, size=(args.n_draws, n_subjects))
            row_idx = np.arange(n_subjects)[None, :]
            drawn = null_pool[row_idx, idx]  # (n_draws, n_subjects)
            group_null = drawn.mean(axis=1)  # (n_draws,)

            null_mean = float(group_null.mean())
            null_sd = float(group_null.std(ddof=1))
            thresh_p05, thresh_p01, thresh_p001 = np.percentile(group_null, [95, 99, 99.9])

            p_raw = float((np.sum(group_null >= obs_acc) + 1) / (args.n_draws + 1))
            pvals_raw.append(p_raw)

            # per-subject single-shuffle null SD, for the conservativeness diagnostic
            single_shuffle_sd = float(np.nanstd(null_pool.astype(np.float64)))

            cell_rows.append({
                "contrast": contrast, "roi": roi, "n_subjects": n_subjects,
                "observed_accuracy": obs_acc,
                "null_mean": null_mean, "null_sd": null_sd,
                "threshold_p05": float(thresh_p05), "threshold_p01": float(thresh_p01),
                "threshold_p001": float(thresh_p001),
                "p_value_raw": p_raw,
                "sig_raw_p05": p_raw < 0.05, "sig_raw_p01": p_raw < 0.01, "sig_raw_p001": p_raw < 0.001,
                "single_shuffle_within_subject_sd": single_shuffle_sd,
            })
        q_raw = benjamini_hochberg(np.asarray(pvals_raw))
        for cell, q in zip(cell_rows, q_raw):
            cell["q_value_fdr"] = float(q)
            cell["sig_fdr_q05"] = bool(q < 0.05)
            cell["sig_fdr_q01"] = bool(q < 0.01)
            cell["sig_fdr_q001"] = bool(q < 0.001)
            rows.append(cell)
        print(f"  {contrast}: done", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(args.out_dir / "group_permutation_results.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for label, col in [("raw p<0.05", "sig_raw_p05"), ("raw p<0.01", "sig_raw_p01"),
                       ("raw p<0.001", "sig_raw_p001"), ("FDR q<0.05", "sig_fdr_q05"),
                       ("FDR q<0.01", "sig_fdr_q01"), ("FDR q<0.001", "sig_fdr_q001")]:
        print(f"  {label:14s}: {int(result[col].sum())}/{len(result)} significant")

    # null-mean sanity check: should sit at ~50% everywhere
    off = (result["null_mean"] - 0.5).abs()
    print(f"\nnull_mean sanity: max |null_mean - 0.5| = {off.max()*100:.3f} pts "
          f"(should be small; large values would indicate a bug)")

    print(f"\nsaved {args.out_dir / 'group_permutation_results.csv'}")
    print("STELZER_GROUP_RESAMPLE_COMPLETE")


if __name__ == "__main__":
    main()
