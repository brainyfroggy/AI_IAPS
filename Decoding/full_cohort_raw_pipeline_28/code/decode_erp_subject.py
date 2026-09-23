#!/usr/bin/env python3
"""Atomically decode one frozen-cohort subject in all ERP-style branches.

Each subject is an independent resumable unit.  A completed output is skipped
only when ``--resume`` is supplied and its request signature exactly matches
the current freeze, code, atlas, manifests, 600 legacy beta files, and Type-D
inputs.  Partial work is never published at the requested output path.

The immutable pilot module supplies the already-validated numerical routines:
the exact historical Avg(random) implementation and run-wise LORO condition
means.  This entry point deliberately writes no group summaries or plots;
those are produced by the separate deterministic aggregation program.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from cohort_analysis_common import (
    COHORT_ROOT,
    DEFAULT_FREEZE,
    HISTORICAL_METHOD,
    PIPELINES,
    ROI_ORDER,
    RUNWISE_METHOD,
    atomic_write_json,
    directory_file_set_signature,
    load_freeze,
    quick_file_signature,
    sha256_file,
    stable_json_sha256,
    subject_label,
    validate_fold_result_rows,
    validate_roi_count_rows,
    validate_subject_in_freeze,
    validate_subject_result_rows,
)


PROJECT_ROOT = COHORT_ROOT.parent.parent
DECODING_ROOT = COHORT_ROOT.parent
PILOT_CODE = DECODING_ROOT / "pilot_raw_pipeline_sub4_6" / "code"
PILOT_SCRIPT = PILOT_CODE / "decode_erp_style_compare.py"
DEFAULT_LEGACY_ROOT = PROJECT_ROOT / "GLM_singletrial" / "betas"
DEFAULT_LEGACY_GROUPS = PROJECT_ROOT / "GLM_singletrial" / "beta_groups.csv"
DEFAULT_LEGACY_MANIFEST = DECODING_ROOT / "stimuli_600trials.csv"
DEFAULT_GLMSINGLE_ROOT = COHORT_ROOT / "derivatives" / "glmsingle"


def load_pilot_module(path: Path = PILOT_SCRIPT) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Immutable pilot ERP module is missing: {path}")
    code_dir = str(path.parent)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    module_name = "ai_iaps_immutable_pilot_erp"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import immutable pilot ERP module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def validate_pinned_contract(pilot: ModuleType, freeze: dict) -> dict:
    historical = freeze["erp_decoding"]["methods"][HISTORICAL_METHOD]
    runwise = freeze["erp_decoding"]["methods"][RUNWISE_METHOD]
    checks = {
        "roi_order": tuple(pilot.ROI_ORDER) == ROI_ORDER,
        "contrasts": tuple(item[0] for item in pilot.CONTRASTS)
        == tuple(freeze["contrasts"]),
        "historical_mode": pilot.MODE == "voxel_trial_source_zscore",
        "historical_repeats": int(pilot.N_REPEATS) == int(historical["repeats"]),
        "historical_folds": int(pilot.N_FOLDS) == int(historical["folds"]),
        "historical_seed": int(pilot.SEED) == int(historical["seed"]),
        "historical_train_averages": int(pilot.N_AVG_GROUPS)
        == int(historical["within_source_train_averages_per_class"]),
        "runwise_outer_folds": int(runwise["outer_folds"]) == 10,
        "pipelines": PIPELINES
        == ("legacy_lsa_8mm", "glmsingle_typed_0mm", "glmsingle_typed_3mm"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Pilot/freeze ERP contract mismatch: {failed}")
    return checks


def resolve_glmsingle_subject_root(args: argparse.Namespace) -> Path:
    if args.glmsingle_subject_root is not None:
        return args.glmsingle_subject_root
    return args.glmsingle_root / subject_label(args.subject)


def load_beta_groups(
    pilot: ModuleType, groups_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    manifest = pilot.normalize_manifest(manifest_path)
    groups = pd.read_csv(groups_path, header=None, names=["beta_file", "category"])
    if len(groups) != 600 or groups["beta_file"].duplicated().any():
        raise ValueError("beta_groups.csv must contain 600 unique beta files")
    expected_category = manifest["valence"].astype(str) + np.where(
        manifest["source"].to_numpy() == "ai", "AI", ""
    )
    if not np.array_equal(
        groups["category"].astype(str).to_numpy(),
        np.asarray(expected_category, dtype=str),
    ):
        raise ValueError("beta_groups.csv categories do not align with the manifest")
    return groups, groups["beta_file"].astype(str).tolist(), manifest


def request_record(args: argparse.Namespace, pilot: ModuleType, freeze: dict) -> dict:
    glmsingle_root = resolve_glmsingle_subject_root(args)
    groups, beta_files, _ = load_beta_groups(
        pilot, args.legacy_groups, args.legacy_manifest
    )
    del groups
    legacy_directory = args.legacy_root / subject_label(args.subject)
    legacy_paths = [legacy_directory / name for name in beta_files]
    missing = [str(path) for path in legacy_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing legacy beta files, first: {missing[:3]}")

    typed = glmsingle_root / "glmsingle" / pilot.TYPE_D_NAME
    glmsingle_inputs = {
        "typed_hdf5": typed,
        "analysis_mask": glmsingle_root / "analysis_mask.nii.gz",
        "flat_mask_indices": glmsingle_root / "flat_mask_indices.npy",
        "trial_manifest": glmsingle_root / "trial_manifest.tsv",
        "validation": glmsingle_root / "validation.json",
        "provenance": glmsingle_root / "provenance.json",
    }
    missing_glm = [str(path) for path in glmsingle_inputs.values() if not path.is_file()]
    if missing_glm:
        raise FileNotFoundError(f"Missing GLMsingle inputs: {missing_glm}")

    record = {
        "analysis_id": freeze["analysis_id"],
        "subject": args.subject,
        "freeze": {
            "path": str(args.freeze.resolve()),
            "sha256": sha256_file(args.freeze),
        },
        "entry_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "pilot_script": {
            "path": str(args.pilot_script.resolve()),
            "sha256": sha256_file(args.pilot_script),
        },
        "pilot_decode_compare": {
            "path": str((args.pilot_script.parent / "decode_compare.py").resolve()),
            "sha256": sha256_file(args.pilot_script.parent / "decode_compare.py"),
        },
        "atlas": quick_file_signature(args.atlas),
        "atlas_labels": quick_file_signature(args.atlas_labels),
        "legacy_groups": quick_file_signature(args.legacy_groups),
        "legacy_manifest": quick_file_signature(args.legacy_manifest),
        "legacy_directory": str(legacy_directory.resolve()),
        "legacy_beta_set": directory_file_set_signature(legacy_paths),
        "glmsingle_subject_root": str(glmsingle_root.resolve()),
        "glmsingle_inputs": {
            name: quick_file_signature(path) for name, path in glmsingle_inputs.items()
        },
    }
    record["request_signature_sha256"] = stable_json_sha256(record)
    return record


def with_method(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "erp_method", method)
    return output


def _stable_sort(frame: pd.DataFrame, fold: bool = False) -> pd.DataFrame:
    pipeline_order = {value: index for index, value in enumerate(PIPELINES)}
    contrast_order = {
        value: index
        for index, value in enumerate(
            (
                "within_natural_pleasant_vs_neutral",
                "within_ai_pleasant_vs_neutral",
                "within_natural_unpleasant_vs_neutral",
                "within_ai_unpleasant_vs_neutral",
                "train_natural_pleasant_vs_neutral_test_ai",
                "train_ai_pleasant_vs_neutral_test_natural",
                "train_natural_unpleasant_vs_neutral_test_ai",
                "train_ai_unpleasant_vs_neutral_test_natural",
            )
        )
    }
    roi_order = {value: index for index, value in enumerate(ROI_ORDER)}
    working = frame.assign(
        _pipeline=frame["pipeline"].map(pipeline_order),
        _contrast=frame.get("contrast", pd.Series(0, index=frame.index)).map(
            contrast_order
        ),
        _roi=frame["roi"].map(roi_order),
    )
    columns = ["_pipeline", "_contrast", "_roi"]
    if fold and "repeat" in working:
        columns.extend(["repeat", "fold"])
    if fold and "held_out_run" in working:
        columns.append("held_out_run")
    return working.sort_values(columns, kind="stable").drop(
        columns=["_pipeline", "_contrast", "_roi"]
    ).reset_index(drop=True)


def validate_and_write(
    staging: Path,
    subject: int,
    historical_subject: pd.DataFrame,
    historical_folds: pd.DataFrame,
    runwise_subject: pd.DataFrame,
    runwise_folds: pd.DataFrame,
    counts: pd.DataFrame,
) -> dict:
    historical_subject = _stable_sort(
        with_method(historical_subject, HISTORICAL_METHOD)
    )
    historical_folds = _stable_sort(
        with_method(historical_folds, HISTORICAL_METHOD), fold=True
    )
    runwise_subject = _stable_sort(with_method(runwise_subject, RUNWISE_METHOD))
    runwise_folds = _stable_sort(
        with_method(runwise_folds, RUNWISE_METHOD), fold=True
    )
    counts = _stable_sort(counts)

    validate_subject_result_rows(historical_subject, subject, HISTORICAL_METHOD)
    validate_fold_result_rows(historical_folds, subject, HISTORICAL_METHOD)
    validate_subject_result_rows(runwise_subject, subject, RUNWISE_METHOD)
    validate_fold_result_rows(runwise_folds, subject, RUNWISE_METHOD)
    validate_roi_count_rows(counts, subject)

    outputs = {
        "historical_avg_random_subject_results.csv": historical_subject,
        "historical_avg_random_fold_results.csv": historical_folds,
        "runwise_loro_subject_results.csv": runwise_subject,
        "runwise_loro_fold_results.csv": runwise_folds,
        "roi_voxel_counts.csv": counts,
    }
    for name, frame in outputs.items():
        frame.to_csv(staging / name, index=False, lineterminator="\n")
    return {
        "historical_subject_rows": len(historical_subject),
        "historical_fold_rows": len(historical_folds),
        "runwise_subject_rows": len(runwise_subject),
        "runwise_fold_rows": len(runwise_folds),
        "roi_count_rows": len(counts),
    }


def validate_existing_output(target: Path, subject: int) -> None:
    """Re-read and validate every resumable table before declaring a skip."""

    historical_subject = pd.read_csv(
        target / "historical_avg_random_subject_results.csv"
    )
    historical_folds = pd.read_csv(
        target / "historical_avg_random_fold_results.csv"
    )
    runwise_subject = pd.read_csv(target / "runwise_loro_subject_results.csv")
    runwise_folds = pd.read_csv(target / "runwise_loro_fold_results.csv")
    counts = pd.read_csv(target / "roi_voxel_counts.csv")
    validate_subject_result_rows(historical_subject, subject, HISTORICAL_METHOD)
    validate_fold_result_rows(historical_folds, subject, HISTORICAL_METHOD)
    validate_subject_result_rows(runwise_subject, subject, RUNWISE_METHOD)
    validate_fold_result_rows(runwise_folds, subject, RUNWISE_METHOD)
    validate_roi_count_rows(counts, subject)


def run(args: argparse.Namespace) -> None:
    freeze = load_freeze(args.freeze)
    validate_subject_in_freeze(args.subject, freeze)
    pilot = load_pilot_module(args.pilot_script)
    contract = validate_pinned_contract(pilot, freeze)
    request = request_record(args, pilot, freeze)

    target = args.output.resolve()
    if target.exists():
        if not args.resume:
            raise FileExistsError(f"Output exists; use --resume to verify and skip: {target}")
        validation_path = target / "validation.json"
        provenance_path = target / "provenance.json"
        if not validation_path.is_file() or not provenance_path.is_file():
            raise RuntimeError(f"Existing target is not a complete resumable result: {target}")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if validation.get("status") != "complete":
            raise RuntimeError(f"Existing target is not marked complete: {target}")
        if provenance.get("request_signature_sha256") != request["request_signature_sha256"]:
            raise RuntimeError("Existing output request signature differs; use a new destination")
        validate_existing_output(target, args.subject)
        print(f"RESUME-SKIP: verified complete matching output {target}", flush=True)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        np.random.seed(int(freeze["erp_decoding"]["methods"][HISTORICAL_METHOD]["seed"]))
        atlas = pilot.load_atlas(args.atlas, args.atlas_labels)
        _, beta_files, legacy_manifest = load_beta_groups(
            pilot, args.legacy_groups, args.legacy_manifest
        )

        historical_rows: list[dict] = []
        historical_fold_rows: list[dict] = []
        runwise_rows: list[dict] = []
        runwise_fold_rows: list[dict] = []
        count_rows: list[dict] = []

        legacy_directory = args.legacy_root / subject_label(args.subject)
        print(f"Sub{args.subject}: legacy LS-A 8 mm", flush=True)
        legacy_data, legacy_rois = pilot.load_legacy_roi_data(
            args.subject, legacy_directory, beta_files, atlas
        )
        rows, folds = pilot.decode_branch(
            args.subject,
            "legacy_lsa_8mm",
            legacy_data,
            legacy_rois,
            legacy_manifest,
        )
        historical_rows.extend(rows)
        historical_fold_rows.extend(folds)
        rows, folds = pilot.decode_branch_runwise_safe(
            args.subject,
            "legacy_lsa_8mm",
            legacy_data,
            legacy_rois,
            legacy_manifest,
        )
        runwise_rows.extend(rows)
        runwise_fold_rows.extend(folds)
        count_rows.extend(legacy_rois.count_rows)
        del legacy_data
        gc.collect()

        glmsingle_root = resolve_glmsingle_subject_root(args)
        glmsingle_manifest = pilot.normalize_manifest(
            glmsingle_root / "trial_manifest.tsv"
        )
        pilot.assert_manifests_aligned(legacy_manifest, glmsingle_manifest)
        print(f"Sub{args.subject}: GLMsingle Type-D 0/+3 mm", flush=True)
        unsmoothed, smoothed, glmsingle_rois = pilot.load_glmsingle_roi_variants(
            args.subject, glmsingle_root, atlas
        )
        for pipeline, values in (
            ("glmsingle_typed_0mm", unsmoothed),
            ("glmsingle_typed_3mm", smoothed),
        ):
            rows, folds = pilot.decode_branch(
                args.subject, pipeline, values, glmsingle_rois, glmsingle_manifest
            )
            historical_rows.extend(rows)
            historical_fold_rows.extend(folds)
            rows, folds = pilot.decode_branch_runwise_safe(
                args.subject, pipeline, values, glmsingle_rois, glmsingle_manifest
            )
            runwise_rows.extend(rows)
            runwise_fold_rows.extend(folds)
            for item in glmsingle_rois.count_rows:
                copied = dict(item)
                copied["pipeline"] = pipeline
                count_rows.append(copied)
        del unsmoothed, smoothed
        gc.collect()

        row_counts = validate_and_write(
            staging,
            args.subject,
            pd.DataFrame(historical_rows),
            pd.DataFrame(historical_fold_rows),
            pd.DataFrame(runwise_rows),
            pd.DataFrame(runwise_fold_rows),
            pd.DataFrame(count_rows),
        )
        provenance = {
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_id": freeze["analysis_id"],
            "subject": subject_label(args.subject),
            "request_signature_sha256": request["request_signature_sha256"],
            "request": request,
            "pinned_contract": contract,
            "methods": {
                HISTORICAL_METHOD: freeze["erp_decoding"]["methods"][HISTORICAL_METHOD],
                RUNWISE_METHOD: freeze["erp_decoding"]["methods"][RUNWISE_METHOD],
            },
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "row_counts": row_counts,
        }
        atomic_write_json(staging / "provenance.json", provenance)
        validation = {
            "status": "complete",
            "subject": subject_label(args.subject),
            "pipelines": list(PIPELINES),
            "roi_order": list(ROI_ORDER),
            "row_counts": row_counts,
            "all_accuracy_values_finite_and_in_0_1": True,
            "unique_keys_complete": True,
            "published_atomically": True,
        }
        atomic_write_json(staging / "validation.json", validation)
        os.replace(staging, target)
        print(f"COMPLETE: {target}", flush=True)
    except Exception:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        failed = target.parent / f".{target.name}.failed-{timestamp}-{staging.name.rsplit('-', 1)[-1]}"
        if staging.exists():
            try:
                os.replace(staging, failed)
                print(f"FAILED; preserved staging at {failed}", file=sys.stderr, flush=True)
            except OSError:
                print(f"FAILED; preserved staging at {staging}", file=sys.stderr, flush=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    default_atlas = Path(
        r"C:\MRIcroGL\Resources\atlas\kastner.nii.gz"
        if os.name == "nt"
        else "/mnt/c/MRIcroGL/Resources/atlas/kastner.nii.gz"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--legacy-groups", type=Path, default=DEFAULT_LEGACY_GROUPS)
    parser.add_argument("--legacy-manifest", type=Path, default=DEFAULT_LEGACY_MANIFEST)
    parser.add_argument("--glmsingle-root", type=Path, default=DEFAULT_GLMSINGLE_ROOT)
    parser.add_argument("--glmsingle-subject-root", type=Path)
    parser.add_argument("--atlas", type=Path, default=default_atlas)
    parser.add_argument("--atlas-labels", type=Path)
    parser.add_argument("--pilot-script", type=Path, default=PILOT_SCRIPT)
    parser.add_argument("--resume", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.atlas_labels is None:
        args.atlas_labels = args.atlas.with_name("kastner.nii.txt")
    return args


if __name__ == "__main__":
    run(parse_args())
