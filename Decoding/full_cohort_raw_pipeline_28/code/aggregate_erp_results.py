#!/usr/bin/env python3
"""Deterministically aggregate validated per-subject ERP decoding outputs.

By default aggregation fails unless the exact 28-subject frozen cohort is
present.  ``--allow-partial`` is an explicit progress-review mode; every
reported group size is still derived from subject rows and printed in labels.

Optional NIfTI exports are atlas-filled ROI summary maps.  They are not
voxelwise searchlight or whole-brain decoding maps, and both filenames and JSON
sidecars state that distinction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from cohort_analysis_common import (
    COHORT_ROOT,
    CONTRASTS,
    DEFAULT_FREEZE,
    HISTORICAL_METHOD,
    PIPELINES,
    ROI_ORDER,
    RUNWISE_METHOD,
    WITHIN_CONTRASTS,
    atomic_write_json,
    dynamic_pipeline_labels,
    load_freeze,
    sha256_file,
    subject_label,
    validate_fold_result_rows,
    validate_roi_count_rows,
    validate_subject_result_rows,
)
from decode_erp_subject import PILOT_SCRIPT, load_pilot_module


DECODING_ROOT = COHORT_ROOT.parent
DEFAULT_HISTORICAL = (
    DECODING_ROOT
    / "results"
    / "decoding_multisub_avg_voxel_trial_source_zscore_aal3_all.pkl"
)

CONTRAST_LABELS = {
    "within_natural_pleasant_vs_neutral": "PL vs Nt, Natural",
    "within_ai_pleasant_vs_neutral": "PL vs Nt, AI",
    "within_natural_unpleasant_vs_neutral": "UP vs Nt, Natural",
    "within_ai_unpleasant_vs_neutral": "UP vs Nt, AI",
    "train_natural_pleasant_vs_neutral_test_ai": "PL vs Nt, Natural to AI",
    "train_ai_pleasant_vs_neutral_test_natural": "PL vs Nt, AI to Natural",
    "train_natural_unpleasant_vs_neutral_test_ai": "UP vs Nt, Natural to AI",
    "train_ai_unpleasant_vs_neutral_test_natural": "UP vs Nt, AI to Natural",
}


def sem(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if len(array) < 2 or not np.all(np.isfinite(array)):
        return float("nan")
    return float(np.std(array, ddof=1) / math.sqrt(len(array)))


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            [
                "erp_method",
                "pipeline",
                "analysis_kind",
                "contrast",
                "contrast_label",
                "roi",
            ],
            sort=False,
        )
        .agg(
            n_subjects=("subject", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            sem_accuracy=("accuracy", sem),
        )
        .reset_index()
    )


def validate_summary_n(
    summary: pd.DataFrame,
    expected_n: int,
    name: str,
    expected_pipelines: int = 3,
) -> None:
    if len(summary) != expected_pipelines * len(CONTRASTS) * len(ROI_ORDER):
        raise ValueError(
            f"{name} has an incomplete {expected_pipelines} x 8 x 17 summary grid"
        )
    if set(summary["n_subjects"].astype(int)) != {expected_n}:
        raise ValueError(f"{name} does not use n={expected_n} in every cell")


def discover_subject_dirs(roots: list[Path]) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Subject-result root does not exist: {root}")
        for directory in sorted(root.iterdir()):
            if not directory.is_dir():
                continue
            match = re.fullmatch(r"Sub(\d+)", directory.name)
            if not match:
                continue
            subject = int(match.group(1))
            if subject in found:
                raise ValueError(
                    f"Duplicate Sub{subject} in {found[subject]} and {directory}"
                )
            found[subject] = directory
    return found


def load_subject_bundle(directory: Path, subject: int, freeze_sha: str) -> dict:
    validation_path = directory / "validation.json"
    provenance_path = directory / "provenance.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if validation.get("status") != "complete" or provenance.get("status") != "complete":
        raise ValueError(f"Sub{subject} is not marked complete")
    if provenance.get("request", {}).get("freeze", {}).get("sha256") != freeze_sha:
        raise ValueError(f"Sub{subject} was decoded under a different analysis freeze")

    historical_subject = pd.read_csv(
        directory / "historical_avg_random_subject_results.csv"
    )
    historical_folds = pd.read_csv(
        directory / "historical_avg_random_fold_results.csv"
    )
    runwise_subject = pd.read_csv(directory / "runwise_loro_subject_results.csv")
    runwise_folds = pd.read_csv(directory / "runwise_loro_fold_results.csv")
    counts = pd.read_csv(directory / "roi_voxel_counts.csv")
    validate_subject_result_rows(historical_subject, subject, HISTORICAL_METHOD)
    validate_fold_result_rows(historical_folds, subject, HISTORICAL_METHOD)
    validate_subject_result_rows(runwise_subject, subject, RUNWISE_METHOD)
    validate_fold_result_rows(runwise_folds, subject, RUNWISE_METHOD)
    validate_roi_count_rows(counts, subject)
    return {
        "historical_subject": historical_subject,
        "historical_folds": historical_folds,
        "runwise_subject": runwise_subject,
        "runwise_folds": runwise_folds,
        "counts": counts,
        "provenance": provenance,
        "directory": directory,
    }


def validate_combined_unique(frame: pd.DataFrame, fold: bool, name: str) -> None:
    keys = ["erp_method", "subject", "pipeline", "contrast", "roi"]
    if fold:
        if set(frame["erp_method"]) == {HISTORICAL_METHOD}:
            keys += ["repeat", "fold"]
        elif set(frame["erp_method"]) == {RUNWISE_METHOD}:
            keys += ["held_out_run"]
        else:
            raise ValueError(f"{name} mixes methods")
    if frame.duplicated(keys).any():
        raise ValueError(f"{name} contains duplicate exact keys")


def save_grid(
    summary: pd.DataFrame,
    pipelines: tuple[str, ...],
    path: Path,
    title: str,
) -> None:
    """Render only the 17 frozen Kastner/Wang ROIs with dynamic sample sizes."""

    labels = dynamic_pipeline_labels(summary)
    colors = {
        "within": ["#1f4e79", "#8ecae6", "#b45f06", "#f6b26b"],
        "cross": ["#1b7837", "#a6dba0", "#8b0000", "#f4a3a3"],
    }
    orders = {
        "within": [value for value in CONTRASTS if value in WITHIN_CONTRASTS],
        "cross": [value for value in CONTRASTS if value not in WITHIN_CONTRASTS],
    }
    figure, axes = plt.subplots(
        len(pipelines),
        2,
        figsize=(24, 3.6 * len(pipelines) + 1.4),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    legend_handles: dict[str, list] = {}
    x = np.arange(len(ROI_ORDER))
    width = 0.18
    for row_index, pipeline in enumerate(pipelines):
        for column_index, kind in enumerate(("within", "cross")):
            axis = axes[row_index, column_index]
            subset = summary[
                (summary["pipeline"] == pipeline)
                & (summary["analysis_kind"] == kind)
            ]
            containers = []
            for contrast_index, (contrast, color) in enumerate(
                zip(orders[kind], colors[kind])
            ):
                rows = (
                    subset[subset["contrast"] == contrast]
                    .set_index("roi")
                    .loc[list(ROI_ORDER)]
                )
                container = axis.bar(
                    x + (contrast_index - 1.5) * width,
                    rows["mean_accuracy"],
                    width,
                    # SEM is undefined for an explicit n=1 progress review.
                    # Plot a zero-length error bar while retaining NaN in CSV.
                    yerr=np.nan_to_num(
                        rows["sem_accuracy"].to_numpy(dtype=float), nan=0.0
                    ),
                    color=color,
                    edgecolor="black",
                    linewidth=0.3,
                    capsize=1.5,
                    alpha=0.9,
                )
                containers.append(container)
            if row_index == 0:
                legend_handles[kind] = containers
                axis.set_title(f"{kind.title()} source", fontweight="bold")
            axis.axhline(0.5, color="0.35", linestyle="--", linewidth=1)
            axis.set_ylim(0.40, 0.90)
            axis.grid(axis="y", alpha=0.18)
            if column_index == 0:
                axis.set_ylabel(f"{labels[pipeline]}\nAccuracy")
            if row_index == len(pipelines) - 1:
                axis.set_xticks(x)
                axis.set_xticklabels(ROI_ORDER, rotation=45, ha="right")
                axis.tick_params(axis="x", labelbottom=True)
    figure.suptitle(title, fontsize=18, fontweight="bold", y=0.995)
    for location, kind in ((0.28, "within"), (0.75, "cross")):
        figure.legend(
            legend_handles[kind],
            [CONTRAST_LABELS[value] for value in orders[kind]],
            loc="upper center",
            bbox_to_anchor=(location, 0.975),
            ncol=2,
            fontsize=9,
        )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_mricrogl_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                label = int(float(parts[0]))
            except ValueError:
                continue
            if label > 0 and parts[1] != "*.*.*.*.*":
                labels[parts[1]] = label
    return labels


def export_roi_value_maps(
    summary: pd.DataFrame,
    atlas_path: Path,
    labels_path: Path,
    output: Path,
) -> int:
    """Export atlas-filled mean-accuracy maps, explicitly not voxelwise maps."""

    atlas_image = nib.load(str(atlas_path))
    atlas_data = np.rint(atlas_image.get_fdata()).astype(np.int32)
    label_ids = parse_mricrogl_labels(labels_path)
    members = {
        "V1v": ("V1v",),
        "V1d": ("V1d",),
        "V2v": ("V2v",),
        "V2d": ("V2d",),
        "V3v": ("V3v",),
        "V3d": ("V3d",),
        "hV4": ("hV4",),
        "V3a": ("V3a",),
        "V3b": ("V3b",),
        "IPS": ("IPS0", "IPS1", "IPS2", "IPS3", "IPS4", "IPS5"),
        "LO1": ("LO1",),
        "LO2": ("LO2",),
        "hMT": ("hMT",),
        "VO1": ("VO1",),
        "VO2": ("VO2",),
        "PHC1": ("PHC1",),
        "PHC2": ("PHC2",),
    }
    missing = sorted(
        name for names in members.values() for name in names if name not in label_ids
    )
    if missing:
        raise ValueError(f"Atlas label file is missing: {missing}")

    output.mkdir(parents=True)
    n_written = 0
    group_columns = ["erp_method", "pipeline", "contrast"]
    for (method, pipeline, contrast), group in summary.groupby(
        group_columns, sort=False
    ):
        rows = group.set_index("roi").loc[list(ROI_ORDER)]
        values = np.zeros(atlas_data.shape, dtype=np.float32)
        for roi in ROI_ORDER:
            ids = [label_ids[name] for name in members[roi]]
            values[np.isin(atlas_data, ids)] = float(rows.loc[roi, "mean_accuracy"])
        header = atlas_image.header.copy()
        header.set_data_dtype(np.float32)
        header["descrip"] = b"Atlas-filled ROI mean accuracy; not voxelwise"
        stem = (
            f"method-{method}_pipeline-{pipeline}_contrast-{contrast}_"
            "desc-kastner_roi_mean_accuracy"
        )
        nifti_path = output / f"{stem}.nii.gz"
        nib.save(nib.Nifti1Image(values, atlas_image.affine, header), nifti_path)
        sidecar = {
            "MapType": "atlas-filled ROI group-summary map",
            "NotVoxelwiseDecoding": True,
            "Statistic": "subject-level group mean classification accuracy",
            "ChanceLevel": 0.5,
            "OutsideSelectedROIValue": 0.0,
            "Atlas": "Kastner/Wang",
            "ROIs": list(ROI_ORDER),
            "ERPMethod": method,
            "Pipeline": pipeline,
            "Contrast": contrast,
            "NSubjects": int(rows["n_subjects"].iloc[0]),
            "SourceSummary": "group_summary.csv",
        }
        atomic_write_json(output / f"{stem}.json", sidecar)
        n_written += 1
    return n_written


def run(args: argparse.Namespace) -> None:
    target = args.output.resolve()
    if target.exists():
        raise FileExistsError(f"Aggregation output already exists: {target}")
    freeze = load_freeze(args.freeze)
    freeze_sha = sha256_file(args.freeze)
    expected_subjects = [int(value) for value in freeze["included_subjects"]]
    discovered = discover_subject_dirs(args.subject_results_root)
    requested = sorted(args.subjects) if args.subjects else sorted(discovered)
    if not args.allow_partial:
        if requested != sorted(expected_subjects):
            raise ValueError(
                f"Full aggregation requires frozen subjects {sorted(expected_subjects)}; "
                f"requested/discovered {requested}"
            )
        if set(discovered) != set(expected_subjects):
            raise ValueError(
                f"Subject roots must contain exactly the frozen cohort; "
                f"missing={sorted(set(expected_subjects) - set(discovered))}, "
                f"unexpected={sorted(set(discovered) - set(expected_subjects))}"
            )
    if not requested:
        raise ValueError("No subject results were found")
    if not set(requested).issubset(set(expected_subjects)):
        raise ValueError("Partial aggregation contains a subject outside the freeze")
    if not set(requested).issubset(set(discovered)):
        raise ValueError("One or more requested subjects have no result directory")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        bundles = {
            subject: load_subject_bundle(discovered[subject], subject, freeze_sha)
            for subject in requested
        }
        historical_subject = pd.concat(
            [bundles[value]["historical_subject"] for value in requested],
            ignore_index=True,
        )
        historical_folds = pd.concat(
            [bundles[value]["historical_folds"] for value in requested],
            ignore_index=True,
        )
        runwise_subject = pd.concat(
            [bundles[value]["runwise_subject"] for value in requested],
            ignore_index=True,
        )
        runwise_folds = pd.concat(
            [bundles[value]["runwise_folds"] for value in requested],
            ignore_index=True,
        )
        counts = pd.concat(
            [bundles[value]["counts"] for value in requested], ignore_index=True
        )
        validate_combined_unique(
            historical_subject, False, "historical subject results"
        )
        validate_combined_unique(
            historical_folds, True, "historical fold results"
        )
        validate_combined_unique(runwise_subject, False, "runwise subject results")
        validate_combined_unique(runwise_folds, True, "runwise fold results")
        if counts.duplicated(["subject", "pipeline", "roi"]).any():
            raise ValueError("Combined ROI counts contain duplicate keys")

        all_subject = pd.concat(
            [historical_subject, runwise_subject], ignore_index=True
        )
        all_summary = summarize(all_subject)
        historical_summary = all_summary[
            all_summary["erp_method"] == HISTORICAL_METHOD
        ].copy()
        runwise_summary = all_summary[
            all_summary["erp_method"] == RUNWISE_METHOD
        ].copy()
        validate_summary_n(historical_summary, len(requested), "historical summary")
        validate_summary_n(runwise_summary, len(requested), "runwise summary")

        pilot = load_pilot_module(args.pilot_script)
        reference = pilot.load_historical(args.historical_result)
        reference.insert(0, "erp_method", HISTORICAL_METHOD)
        reference_for_reproduction = reference[
            reference["subject"].isin(map(subject_label, requested))
        ]
        legacy = historical_subject[
            historical_subject["pipeline"] == "legacy_lsa_8mm"
        ]
        keys = ["subject", "analysis_kind", "contrast", "roi"]
        reproduction = legacy.merge(
            reference_for_reproduction,
            on=keys,
            suffixes=("_recomputed", "_attached"),
            validate="one_to_one",
        )
        reproduction["abs_difference"] = (
            reproduction["accuracy_recomputed"]
            - reproduction["accuracy_attached"]
        ).abs()
        expected_reproduction_rows = len(requested) * len(CONTRASTS) * len(ROI_ORDER)
        if len(reproduction) != expected_reproduction_rows:
            raise ValueError("Historical reproduction comparison is incomplete")
        tolerance = float(pilot.REPRODUCTION_TOLERANCE)
        reproduction_passed = bool(
            (reproduction["abs_difference"] <= tolerance).all()
        )
        if not reproduction_passed:
            raise RuntimeError(
                "Legacy exact-historical reproduction exceeded the frozen tolerance"
            )

        historical_reference_summary = summarize(reference)
        historical_plot_summary = pd.concat(
            [historical_reference_summary, historical_summary], ignore_index=True
        )
        validate_summary_n(
            historical_reference_summary,
            len(freeze["included_subjects"]),
            "attached historical reference summary",
            expected_pipelines=1,
        )

        all_subject.to_csv(staging / "all_subject_results.csv", index=False)
        historical_folds.to_csv(
            staging / "historical_avg_random_fold_results.csv", index=False
        )
        runwise_folds.to_csv(staging / "runwise_loro_fold_results.csv", index=False)
        counts.to_csv(staging / "roi_voxel_counts.csv", index=False)
        all_summary.to_csv(staging / "group_summary.csv", index=False)
        reproduction.to_csv(
            staging / "legacy_historical_reproduction.csv", index=False
        )

        collapsed_subject = (
            all_subject.groupby(
                ["erp_method", "pipeline", "subject", "analysis_kind"],
                sort=False,
            )["accuracy"]
            .mean()
            .reset_index()
        )
        overall = (
            all_subject.groupby(
                ["erp_method", "pipeline", "subject"], sort=False
            )["accuracy"]
            .mean()
            .reset_index()
            .assign(analysis_kind="overall")
        )
        collapsed_subject = pd.concat(
            [collapsed_subject, overall], ignore_index=True
        )
        collapsed_summary = (
            collapsed_subject.groupby(
                ["erp_method", "pipeline", "analysis_kind"], sort=False
            )
            .agg(
                n_subjects=("subject", "nunique"),
                mean_accuracy=("accuracy", "mean"),
                sem_accuracy=("accuracy", sem),
            )
            .reset_index()
        )
        collapsed_subject.to_csv(
            staging / "collapsed_subject_results.csv", index=False
        )
        collapsed_summary.to_csv(
            staging / "collapsed_group_summary.csv", index=False
        )

        delta_source = all_subject.pivot(
            index=["erp_method", "subject", "analysis_kind", "contrast", "roi"],
            columns="pipeline",
            values="accuracy",
        ).reset_index()
        delta_rows = []
        for pipeline in ("glmsingle_typed_0mm", "glmsingle_typed_3mm"):
            working = delta_source[
                ["erp_method", "subject", "analysis_kind", "contrast", "roi"]
            ].copy()
            working["comparison_pipeline"] = pipeline
            working["accuracy_delta_vs_legacy"] = (
                delta_source[pipeline] - delta_source["legacy_lsa_8mm"]
            )
            delta_rows.append(working)
        deltas = pd.concat(delta_rows, ignore_index=True)
        deltas.to_csv(staging / "paired_pipeline_differences.csv", index=False)

        save_grid(
            historical_plot_summary,
            (
                "historical_28sub_original",
                "legacy_lsa_8mm",
                "glmsingle_typed_0mm",
                "glmsingle_typed_3mm",
            ),
            staging / "kastner_roi_historical_avg_random.png",
            "Kastner/Wang ROI decoding — historical Avg(random)",
        )
        save_grid(
            runwise_summary,
            PIPELINES,
            staging / "kastner_roi_runwise_loro.png",
            "Kastner/Wang ROI decoding — run-wise LORO means",
        )

        map_count = 0
        if args.export_roi_maps:
            map_summary = pd.concat(
                [historical_reference_summary, all_summary], ignore_index=True
            )
            map_count = export_roi_value_maps(
                map_summary,
                args.atlas,
                args.atlas_labels,
                staging / "roi_value_maps_not_voxelwise",
            )

        report = f"""# Full-cohort ERP decoding aggregation

- Analysis ID: `{freeze['analysis_id']}`
- Subjects: {', '.join(subject_label(value) for value in requested)}
- Number of subjects: **{len(requested)}**
- Mode: {'partial progress review' if args.allow_partial else 'frozen complete cohort'}
- Pipelines: legacy LS-A 8 mm; GLMsingle Type-D 0 mm; Type-D +3 mm
- ROIs: the 17 frozen Kastner/Wang regions only
- Historical method: exact Avg(random), descriptive/transductive as frozen
- Primary method: run-wise LORO condition means; run controlled, not identity held out
- Attached-result reproduction: passed; maximum absolute difference = {reproduction['abs_difference'].max():.8f}, tolerance = {tolerance:.8f}
- ROI-value NIfTI maps: {map_count}; these are atlas-filled group summaries, **not voxelwise decoding maps**

Group means and SEMs are calculated from subject-level accuracies, never from fold rows.
"""
        (staging / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")
        provenance = {
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_id": freeze["analysis_id"],
            "freeze": {
                "path": str(args.freeze.resolve()),
                "sha256": freeze_sha,
            },
            "subjects": [subject_label(value) for value in requested],
            "n_subjects": len(requested),
            "allow_partial": bool(args.allow_partial),
            "subject_directories": {
                subject_label(value): str(discovered[value].resolve())
                for value in requested
            },
            "row_counts": {
                "all_subject_results": len(all_subject),
                "historical_fold_results": len(historical_folds),
                "runwise_fold_results": len(runwise_folds),
                "roi_voxel_counts": len(counts),
                "group_summary": len(all_summary),
            },
            "historical_reproduction": {
                "n_rows": len(reproduction),
                "max_abs_difference": float(reproduction["abs_difference"].max()),
                "tolerance": tolerance,
                "passed": reproduction_passed,
            },
            "roi_value_map_count": map_count,
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        }
        atomic_write_json(staging / "provenance.json", provenance)
        validation = {
            "status": "complete",
            "exact_subject_set": requested == sorted(expected_subjects),
            "n_subjects": len(requested),
            "all_subject_keys_complete": True,
            "all_fold_keys_complete": True,
            "all_summary_cells_have_same_subject_n": True,
            "historical_reproduction_passed": reproduction_passed,
            "only_17_frozen_rois": True,
            "roi_value_maps_are_atlas_filled_not_voxelwise": bool(
                args.export_roi_maps
            ),
        }
        atomic_write_json(staging / "validation.json", validation)
        os.replace(staging, target)
        print(f"COMPLETE: {target}", flush=True)
    except Exception:
        print(f"FAILED; preserved staging directory: {staging}", file=sys.stderr)
        raise


def build_parser() -> argparse.ArgumentParser:
    default_atlas = Path(
        r"C:\MRIcroGL\Resources\atlas\kastner.nii.gz"
        if os.name == "nt"
        else "/mnt/c/MRIcroGL/Resources/atlas/kastner.nii.gz"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject-results-root", type=Path, nargs="+", required=True
    )
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--historical-result", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--pilot-script", type=Path, default=PILOT_SCRIPT)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--export-roi-maps", action="store_true")
    parser.add_argument("--atlas", type=Path, default=default_atlas)
    parser.add_argument("--atlas-labels", type=Path)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.atlas_labels is None:
        args.atlas_labels = args.atlas.with_name("kastner.nii.txt")
    return args


if __name__ == "__main__":
    run(parse_args())
