#!/usr/bin/env python3
"""Validate, QC, checksum, and safely retire fMRIPrep subject work products."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import fmriprep_sdc_workflow as workflow

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - surfaced as a clear production gate
    nib = None


PILOT_CODE = Path(__file__).resolve().parents[2] / "pilot_raw_pipeline_sub4_6" / "code"
DEFAULT_QUANT_SCRIPT = PILOT_CODE / "quantify_fmriprep_tsnr.py"
DEFAULT_VISUAL_SCRIPT = PILOT_CODE / "render_fmriprep_visual_qc.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_immutable(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(f".{path.name}.staging-{os.getpid()}")
    stage.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    stage.replace(path)


def count_tsv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return sum(1 for _ in stream) - 1


def confound_motion(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames or "framewise_displacement" not in reader.fieldnames:
            raise RuntimeError(f"Missing framewise_displacement column: {path}")
        values = []
        for row in reader:
            raw = str(row.get("framewise_displacement", "")).strip().lower()
            if raw in {"", "n/a", "na", "nan"}:
                continue
            value = float(raw)
            if not np.isfinite(value) or value < 0:
                raise RuntimeError(f"Invalid framewise displacement {raw!r} in {path}")
            values.append(value)
    array = np.asarray(values, dtype=float)
    return {
        "valid_fd_frames": int(array.size),
        "fd_gt_0p5_frames": int(np.sum(array > 0.5)),
        "fd_gt_0p5_fraction": float(np.mean(array > 0.5)) if array.size else 0.0,
        "mean_fd_mm": float(np.mean(array)) if array.size else None,
        "median_fd_mm": float(np.median(array)) if array.size else None,
        "max_fd_mm": float(np.max(array)) if array.size else None,
    }


def _spatial_reference_matches(value: object, expected: str) -> bool:
    if isinstance(value, str):
        return expected in value
    if isinstance(value, list):
        return any(_spatial_reference_matches(item, expected) for item in value)
    if isinstance(value, dict):
        return any(expected in str(key) or _spatial_reference_matches(item, expected) for key, item in value.items())
    return False


def _sidecar_space_matches(sidecar: Mapping, expected: str) -> bool:
    """Accept the encodings emitted by pinned fMRIPrep 25.1.3.

    Its preprocessed BOLD JSON normally records template space in ``Resolution``
    and ``Sources`` rather than a ``SpatialReference`` key.
    """
    if _spatial_reference_matches(sidecar.get("SpatialReference"), expected):
        return True
    if expected in str(sidecar.get("Resolution", "")):
        return True
    return _spatial_reference_matches(sidecar.get("Sources"), expected)


def validate_derivative(
    derivative: Path,
    plan: workflow.SubjectPlan,
    sdc_config: Mapping,
) -> dict:
    if nib is None:
        raise RuntimeError("nibabel is required for derivative grid validation")
    derivative = workflow.require_n_drive(derivative, "fMRIPrep derivative", allow_missing=False)
    dataset_path = derivative / "dataset_description.json"
    dataset = workflow.read_json(dataset_path)
    generated = dataset.get("GeneratedBy")
    if not isinstance(generated, list):
        raise RuntimeError("Derivative dataset_description lacks GeneratedBy")
    matches = [
        item for item in generated
        if isinstance(item, dict)
        and str(item.get("Name", "")).lower() == "fmriprep"
        and str(item.get("Version")) == str(sdc_config["fmriprep"]["version"])
    ]
    if not matches:
        raise RuntimeError(
            f"Derivative does not identify fMRIPrep {sdc_config['fmriprep']['version']}"
        )
    subject_root = derivative / f"sub-{plan.subject_label}"
    expected_space = str(sdc_config["fmriprep"]["output_space"])
    expected_resolution = int(sdc_config["fmriprep"]["output_resolution"])
    reference_shape = None
    reference_affine = None
    records = []
    for session in plan.sessions:
        func = subject_root / f"ses-{session.session}" / "func"
        for run in session.runs:
            prefix = f"sub-{plan.subject_label}_ses-{session.session}_task-iaps_run-{run:02d}"
            bold_path = func / f"{prefix}_space-{expected_space}_res-{expected_resolution}_desc-preproc_bold.nii.gz"
            bold_json = bold_path.with_name(bold_path.name.replace(".nii.gz", ".json"))
            mask_path = func / f"{prefix}_space-{expected_space}_res-{expected_resolution}_desc-brain_mask.nii.gz"
            confounds = func / f"{prefix}_desc-confounds_timeseries.tsv"
            required = [bold_path, bold_json, mask_path, confounds]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing required derivative files: {missing}")
            bold = nib.load(str(bold_path))
            mask = nib.load(str(mask_path))
            if len(bold.shape) != 4 or bold.shape[3] < 2:
                raise RuntimeError(f"Invalid preprocessed BOLD shape {bold.shape}: {bold_path}")
            if mask.shape != bold.shape[:3] or not np.allclose(mask.affine, bold.affine, atol=1e-5):
                raise RuntimeError(f"BOLD/mask grid mismatch for {prefix}")
            if reference_shape is None:
                reference_shape = bold.shape[:3]
                reference_affine = bold.affine
            elif bold.shape[:3] != reference_shape or not np.allclose(
                bold.affine, reference_affine, atol=1e-5
            ):
                raise RuntimeError(f"MNI grid differs across runs: {bold_path}")
            rows = count_tsv_rows(confounds)
            if rows != bold.shape[3]:
                raise RuntimeError(
                    f"Confound rows ({rows}) do not match BOLD volumes ({bold.shape[3]}): {bold_path}"
                )
            sidecar = workflow.read_json(bold_json)
            if not _sidecar_space_matches(sidecar, expected_space):
                raise RuntimeError(f"Wrong or missing template-space metadata in {bold_json}")
            observed_method = workflow.sdc_method(
                derivative, plan.subject_label, session.session, run
            )
            if not workflow.method_matches(session.mode, observed_method):
                raise RuntimeError(
                    f"Expected {session.mode} SDC for {prefix}; summary says {observed_method!r}"
                )
            sdc_reportlet = subject_root / "figures" / f"{prefix}_desc-sdc_bold.svg"
            coreg_reportlet = subject_root / "figures" / f"{prefix}_desc-coreg_bold.svg"
            carpet_reportlet = subject_root / "figures" / f"{prefix}_desc-carpetplot_bold.svg"
            missing_reportlets = [
                str(path) for path in (sdc_reportlet, coreg_reportlet, carpet_reportlet)
                if not path.is_file()
            ]
            if missing_reportlets:
                raise FileNotFoundError(f"Missing visual reportlets: {missing_reportlets}")
            records.append(
                {
                    "session": session.session,
                    "run": run,
                    "expected_sdc_mode": session.mode,
                    "reported_sdc_method": observed_method,
                    "bold": str(bold_path),
                    "mask": str(mask_path),
                    "confounds": str(confounds),
                    "shape": list(bold.shape),
                    "voxel_sizes_mm": [float(value) for value in bold.header.get_zooms()[:3]],
                    "motion": confound_motion(confounds),
                    "reportlets": {
                        "sdc": str(sdc_reportlet),
                        "coreg": str(coreg_reportlet),
                        "carpetplot": str(carpet_reportlet),
                    },
                }
            )
    observed_pairs = {(item["session"], item["run"]) for item in records}
    expected_pairs = {
        (session.session, run) for session in plan.sessions for run in session.runs
    }
    if observed_pairs != expected_pairs or len(records) != 10:
        raise RuntimeError(
            f"Derivative run set mismatch: expected={sorted(expected_pairs)}, observed={sorted(observed_pairs)}"
        )
    return {
        "status": "pass",
        "generated_at": utc_now(),
        "subject": plan.subject,
        "derivative_root": str(derivative),
        "dataset_description_sha256": workflow.sha256_file(dataset_path),
        "fmriprep_version": str(sdc_config["fmriprep"]["version"]),
        "output_space": expected_space,
        "output_resolution": expected_resolution,
        "run_count": len(records),
        "grid_shape": list(reference_shape),
        "grid_affine": np.asarray(reference_affine).tolist(),
        "runs": records,
        "note": "Automated completeness, grid, SDC-marker, and motion checks passed. This does not constitute visual acceptance.",
    }


def run_qc_bundle(
    derivative: Path,
    plan: workflow.SubjectPlan,
    output: Path,
    python_executable: Path,
    quantitative_script: Path,
    visual_script: Path,
    inkscape: Path | None,
) -> dict:
    derivative = workflow.require_n_drive(derivative, "fMRIPrep derivative", allow_missing=False)
    output = workflow.require_n_drive(output, "QC output")
    if output.exists():
        raise FileExistsError(output)
    for path in (python_executable, quantitative_script, visual_script):
        if not path.is_file():
            raise FileNotFoundError(path)
    stage = output.with_name(f".{output.name}.staging-{os.getpid()}")
    stage.mkdir(parents=True)
    commands: list[list[str]] = []
    try:
        quant_output = stage / "quantitative"
        quant_command = [
            str(python_executable),
            str(quantitative_script),
            "--fmriprep-root",
            str(derivative),
            "--subject",
            str(plan.subject),
            "--output",
            str(quant_output),
            "--space",
            "MNI152NLin6Asym",
            "--resolution",
            "2",
            "--expected-runs",
            *[str(run) for run in range(1, 11)],
        ]
        commands.append(quant_command)
        subprocess.run(quant_command, check=True)
        visual_outputs = []
        for session in plan.sessions:
            destination = stage / "visual" / f"ses-{session.session}"
            command = [
                str(python_executable),
                str(visual_script),
                "--fmriprep-root",
                str(derivative),
                "--subject",
                str(plan.subject),
                "--session",
                session.session,
                "--expected-runs",
                *[str(run) for run in session.runs],
                "--output",
                str(destination),
                "--contact-prefix",
                f"sub-{plan.subject_label}_ses-{session.session}",
            ]
            if inkscape is not None:
                command += ["--inkscape", str(inkscape)]
            commands.append(command)
            subprocess.run(command, check=True)
            visual_outputs.append(str(destination.relative_to(stage)))

        attestation_path = stage / "visual_review_attestation.tsv"
        with attestation_path.open("w", encoding="utf-8", newline="") as stream:
            fields = [
                "subject",
                "session",
                "run",
                "expected_sdc_mode",
                "sdc_review",
                "coreg_review",
                "carpet_motion_review",
                "overall_decision",
                "reviewer",
                "reviewed_at_utc",
                "notes",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for session in plan.sessions:
                for run in session.runs:
                    writer.writerow(
                        {
                            "subject": f"Sub{plan.subject_label}",
                            "session": f"ses-{session.session}",
                            "run": f"run-{run:02d}",
                            "expected_sdc_mode": session.mode,
                            "sdc_review": "PENDING",
                            "coreg_review": "PENDING",
                            "carpet_motion_review": "PENDING",
                            "overall_decision": "PENDING",
                            "reviewer": "",
                            "reviewed_at_utc": "",
                            "notes": "",
                        }
                    )
        receipt = {
            "status": "artifacts_generated_human_review_pending",
            "generated_at": utc_now(),
            "subject": plan.subject,
            "derivative_root": str(derivative),
            "commands": commands,
            "quantitative_output": str(quant_output.relative_to(stage)),
            "visual_outputs": visual_outputs,
            "attestation": str(attestation_path.relative_to(stage)),
            "warning": "Contact sheets are review aids. A human must edit the attestation to ACCEPT every run; generation alone is not acceptance.",
        }
        (stage / "qc_bundle_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        stage.replace(output)
        return receipt
    except Exception:
        raise


def validate_attestation(path: Path, plan: workflow.SubjectPlan) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    expected = {
        (f"ses-{session.session}", f"run-{run:02d}")
        for session in plan.sessions
        for run in session.runs
    }
    observed = {(row.get("session", ""), row.get("run", "")) for row in rows}
    if len(rows) != 10 or observed != expected:
        raise RuntimeError("Visual attestation does not contain exactly the expected 10 runs")
    review_fields = ("sdc_review", "coreg_review", "carpet_motion_review", "overall_decision")
    failures = []
    for row in rows:
        if any(str(row.get(field, "")).strip().upper() != "ACCEPT" for field in review_fields):
            failures.append((row.get("session"), row.get("run"), "not fully accepted"))
        if not str(row.get("reviewer", "")).strip() or not str(row.get("reviewed_at_utc", "")).strip():
            failures.append((row.get("session"), row.get("run"), "missing reviewer/time"))
    if failures:
        raise RuntimeError(f"Visual-QC gate is incomplete or rejected: {failures}")
    return {
        "status": "pass",
        "path": str(path.resolve()),
        "sha256": workflow.sha256_file(path),
        "rows": len(rows),
        "reviewers": sorted({row["reviewer"].strip() for row in rows}),
    }


def create_checksum_manifest(roots: Sequence[tuple[str, Path]], output: Path) -> dict:
    output = workflow.require_n_drive(output, "checksum manifest")
    if output.exists():
        raise FileExistsError(output)
    entries = []
    for label, root in roots:
        root = workflow.require_n_drive(root, f"archive root {label}", allow_missing=False)
        for path in workflow.iter_files(root):
            entries.append(
                {
                    "root_label": label,
                    "root": str(root),
                    "relative_path": str(path.relative_to(root)).replace("\\", "/"),
                    "size_bytes": path.stat().st_size,
                    "sha256": workflow.sha256_file(path),
                }
            )
    entries.sort(key=lambda item: (item["root_label"], item["relative_path"]))
    payload = {
        "schema_version": 1,
        "status": "complete",
        "generated_at": utc_now(),
        "file_count": len(entries),
        "roots": {label: str(root.resolve()) for label, root in roots},
        "entries": entries,
    }
    write_json_immutable(output, payload)
    verify_checksum_manifest(output)
    return payload


def verify_checksum_manifest(path: Path) -> dict:
    payload = workflow.read_json(path)
    if payload.get("status") != "complete" or payload.get("file_count") != len(payload.get("entries", [])):
        raise RuntimeError("Malformed checksum manifest")
    failures = []
    roots = {key: Path(value) for key, value in payload["roots"].items()}
    for entry in payload["entries"]:
        target = roots[entry["root_label"]] / Path(entry["relative_path"])
        if not target.is_file():
            failures.append(f"missing:{target}")
            continue
        if target.stat().st_size != int(entry["size_bytes"]):
            failures.append(f"size:{target}")
            continue
        if workflow.sha256_file(target) != entry["sha256"]:
            failures.append(f"sha256:{target}")
    if failures:
        raise RuntimeError(f"Checksum verification failed: {failures[:20]}")
    return {
        "status": "pass",
        "manifest": str(path.resolve()),
        "manifest_sha256": workflow.sha256_file(path),
        "file_count": len(payload["entries"]),
    }


def retire_work_volumes(
    launch_receipt: Path,
    archive_manifest: Path,
    attestation: Path,
    plan: workflow.SubjectPlan,
    execute: bool,
) -> dict:
    launch = workflow.read_json(launch_receipt)
    if launch.get("status") != "pass" or launch.get("plan", {}).get("subject") != plan.subject:
        raise RuntimeError("Launch receipt does not match subject or did not pass")
    archive = verify_checksum_manifest(archive_manifest)
    visual = validate_attestation(attestation, plan)
    volumes = launch.get("work_volumes_retained_pending_archive_and_visual_qc")
    expected = {
        f"ai_iaps_full28_fmriprep_work_sub{plan.subject_label}_{branch.name}"
        for branch in plan.branches
    }
    if not isinstance(volumes, list) or set(volumes) != expected:
        raise RuntimeError(f"Launch receipt work-volume set mismatch: {volumes}; expected {sorted(expected)}")
    commands = [["docker", "volume", "rm", volume] for volume in sorted(volumes)]
    if execute:
        for command in commands:
            subprocess.run(command, check=True)
    return {
        "status": "retired" if execute else "dry_run",
        "subject": plan.subject,
        "archive_verification": archive,
        "visual_attestation": visual,
        "commands": commands,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-config", type=Path, default=workflow.DEFAULT_COHORT_CONFIG)
    parser.add_argument("--sdc-config", type=Path, default=workflow.DEFAULT_SDC_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-derivative")
    validate.add_argument("--subject", type=int, required=True)
    validate.add_argument("--derivative-root", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)

    qc = commands.add_parser("qc-bundle")
    qc.add_argument("--subject", type=int, required=True)
    qc.add_argument("--derivative-root", type=Path, required=True)
    qc.add_argument("--output", type=Path, required=True)
    qc.add_argument("--python-executable", type=Path, required=True)
    qc.add_argument("--quantitative-script", type=Path, default=DEFAULT_QUANT_SCRIPT)
    qc.add_argument("--visual-script", type=Path, default=DEFAULT_VISUAL_SCRIPT)
    qc.add_argument("--inkscape", type=Path)

    attest = commands.add_parser("validate-attestation")
    attest.add_argument("--subject", type=int, required=True)
    attest.add_argument("--attestation", type=Path, required=True)

    manifest = commands.add_parser("create-manifest")
    manifest.add_argument("--derivative-root", type=Path, required=True)
    manifest.add_argument("--qc-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify-manifest")
    verify.add_argument("--manifest", type=Path, required=True)

    retire = commands.add_parser("retire-work-volumes")
    retire.add_argument("--subject", type=int, required=True)
    retire.add_argument("--launch-receipt", type=Path, required=True)
    retire.add_argument("--archive-manifest", type=Path, required=True)
    retire.add_argument("--attestation", type=Path, required=True)
    retire.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cohort, sdc = workflow.load_configs(args.cohort_config, args.sdc_config)
        plan = workflow.subject_plan(cohort, args.subject) if hasattr(args, "subject") else None
        if args.command == "validate-derivative":
            payload = validate_derivative(args.derivative_root, plan, sdc)
            write_json_immutable(args.output, payload)
        elif args.command == "qc-bundle":
            payload = run_qc_bundle(
                args.derivative_root,
                plan,
                args.output,
                args.python_executable,
                args.quantitative_script,
                args.visual_script,
                args.inkscape,
            )
        elif args.command == "validate-attestation":
            payload = validate_attestation(args.attestation, plan)
        elif args.command == "create-manifest":
            payload = create_checksum_manifest(
                (("fmriprep", args.derivative_root), ("qc", args.qc_root)), args.output
            )
        elif args.command == "verify-manifest":
            payload = verify_checksum_manifest(args.manifest)
        else:
            payload = retire_work_volumes(
                args.launch_receipt,
                args.archive_manifest,
                args.attestation,
                plan,
                args.execute,
            )
        print(json.dumps(payload, indent=2))
        return 0
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
