#!/usr/bin/env python3
"""Fail-closed visual-QC, archive, and final subject gates.

The commands in this module are deliberately resumable only through immutable
evidence.  Visual artifacts are generated for every acquisition session but
the visual stage exits non-zero until a reviewer explicitly accepts every run.
The archive stage hashes the N:-resident derivative and QC trees before it can
retire the exact Docker work volumes named by the fMRIPrep launch receipt.  The
final subject gate independently re-reads the archive, GLMsingle Type-D HDF5,
ERP tables, and hash-chained production ledger.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Callable, Mapping, Sequence

import h5py
import nibabel as nib
import numpy as np
import pandas as pd

import fmriprep_postprocess as fpost
import fmriprep_sdc_workflow as workflow
from audit_stage_runner import verify_attempt_evidence, verify_event_mirror
from cohort_analysis_common import (
    CONTRASTS,
    HISTORICAL_METHOD,
    PIPELINES,
    ROI_ORDER,
    RUNWISE_METHOD,
    quick_file_signature,
    sha256_file,
    stable_json_sha256,
    validate_fold_result_rows,
    validate_roi_count_rows,
    validate_subject_result_rows,
)
from production_control import (
    build_control_identity,
    latest_stage_completions,
    read_ledger,
    validate_approval,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_VISUAL_SCRIPT = (
    ROOT.parent / "pilot_raw_pipeline_sub4_6" / "code" / "render_fmriprep_visual_qc.py"
)
VISUAL_MANIFEST = "visual_artifact_manifest.json"
ATTESTATION = "visual_review_attestation.tsv"
VISUAL_RECEIPT = "visual_gate_receipt.json"
ARCHIVE_MANIFEST = "required_products_sha256.json"
ARCHIVE_RECEIPT = "archive_receipt.json"
TYPE_D_NAME = "TYPED_FITHRF_GLMDENOISE_RR.hdf5"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
VOLUME_RE = re.compile(r"ai_iaps_full28_fmriprep_work_sub\d{2}_[a-z0-9_]+")
VISUAL_CATEGORIES = (
    "sdc_background",
    "sdc_foreground",
    "coreg_background",
    "coreg_foreground",
    "rois",
    "carpetplots",
)
ATTESTATION_FIELDS = (
    "subject",
    "session",
    "run",
    "expected_sdc_mode",
    "visual_manifest_sha256",
    "sdc_review",
    "coreg_review",
    "carpet_motion_review",
    "overall_decision",
    "reviewer",
    "reviewed_at_utc",
    "notes",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def write_json_immutable(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace immutable evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(f".{path.name}.staging-{os.getpid()}")
    if stage.exists():
        raise FileExistsError(stage)
    stage.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(stage, path)


def load_visual_module(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("ai_iaps_pinned_visual_qc", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import visual-QC renderer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _wsl_call(
    wsl: Path, distribution: str, argv: Sequence[str], *, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(wsl), "-d", distribution, "--", *map(str, argv)],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def verify_wsl_renderer(
    wsl: Path,
    distribution: str,
    renderer: str,
    expected_sha256: str,
) -> dict:
    if not wsl.is_file() or not SHA256_RE.fullmatch(expected_sha256):
        raise RuntimeError("Invalid WSL renderer identity arguments")
    digest = _wsl_call(wsl, distribution, ("sha256sum", renderer), timeout=30)
    if digest.returncode != 0:
        raise RuntimeError(f"Cannot hash WSL renderer: {digest.stderr.strip()}")
    observed = digest.stdout.strip().split()[0] if digest.stdout.strip() else ""
    if observed != expected_sha256:
        raise RuntimeError(
            f"WSL renderer SHA-256 changed: {observed}; expected {expected_sha256}"
        )
    version = _wsl_call(wsl, distribution, (renderer, "--version"), timeout=30)
    if version.returncode != 0:
        raise RuntimeError(f"Cannot execute WSL renderer: {version.stderr.strip()}")
    return {
        "wsl_executable": str(wsl.resolve()),
        "wsl_executable_sha256": sha256_file(wsl),
        "distribution": distribution,
        "renderer": renderer,
        "renderer_sha256": observed,
        "renderer_version": (version.stdout or version.stderr).strip(),
    }


def _wsl_path(wsl: Path, distribution: str, path: Path) -> str:
    """Map N: (including its resolved UNC form) to the mounted /mnt/n path.

    ``wslpath`` receives UNC arguments through Windows interop after lossy
    backslash handling on this host.  Resolve the one approved analysis share
    against the actual N: mapping instead, then prove the resulting path is
    visible inside the selected distribution.
    """

    del wsl, distribution  # Kept in the signature for the renderer callback.
    absolute = Path(os.path.abspath(path))
    n_root = Path("N:\\").resolve()
    resolved = absolute.resolve(strict=False)
    try:
        relative = resolved.relative_to(n_root)
    except ValueError as error:
        raise RuntimeError(
            f"Visual renderer inputs/outputs must resolve inside the N: share: {path}"
        ) from error
    linux = Path("/mnt/n") / Path(*relative.parts)
    return linux.as_posix()


def build_wsl_render_function(
    wsl: Path, distribution: str, renderer: str
) -> Callable[[Path, Path], None]:
    def render(source: Path, output: Path) -> None:
        linux_source = _wsl_path(wsl, distribution, source)
        linux_output = _wsl_path(wsl, distribution, output)
        result = _wsl_call(
            wsl,
            distribution,
            (
                renderer,
                linux_source,
                "--export-type=png",
                f"--export-filename={linux_output}",
                "--export-background=white",
                "--export-background-opacity=255",
            ),
            timeout=180,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"WSL Inkscape failed for {source}: {result.stderr.strip()}"
            )

    return render


@contextmanager
def n_drive_tempfiles(root: Path):
    """Route Python temporary files to an explicit N:-resident directory."""

    temp_root = root / ".visual_renderer_temp"
    temp_root.mkdir(parents=True, exist_ok=False)
    previous = tempfile.tempdir
    tempfile.tempdir = str(temp_root)
    try:
        yield
    finally:
        tempfile.tempdir = previous
        try:
            temp_root.rmdir()
        except OSError:
            # A failed renderer may leave evidence; never recursively delete it.
            pass


def _expected_visual_pngs(session: str, runs: Sequence[int]) -> set[str]:
    names = set()
    for run in runs:
        names.update(
            {
                f"run-{run:02d}_sdc_background.png",
                f"run-{run:02d}_sdc_foreground.png",
                f"run-{run:02d}_coreg_background.png",
                f"run-{run:02d}_coreg_foreground.png",
                f"run-{run:02d}_rois.png",
                f"run-{run:02d}_carpetplot.png",
            }
        )
    prefix = f"sub-{{subject}}_ses-{session}"
    names.update(f"{prefix}_{category}_contact.png" for category in VISUAL_CATEGORIES)
    return names


def inspect_visual_session(
    destination: Path,
    derivative_root: Path,
    plan: workflow.SubjectPlan,
    session: workflow.SessionSpec,
) -> list[Path]:
    provenance_path = destination / "provenance.json"
    table_path = destination / "visual_qc_manifest.tsv"
    if not provenance_path.is_file() or not table_path.is_file():
        raise FileNotFoundError(f"Incomplete visual-QC session bundle: {destination}")
    provenance = read_json(provenance_path)
    if (
        int(provenance.get("subject", -1)) != plan.subject
        or str(provenance.get("session")) != session.session
        or provenance.get("task") != "iaps"
        or [int(value) for value in provenance.get("expected_runs", [])]
        != list(session.runs)
        or Path(str(provenance.get("source_fmriprep_root"))).resolve()
        != derivative_root.resolve()
    ):
        raise RuntimeError(f"Visual-QC provenance does not match the subject plan: {destination}")
    contacts = provenance.get("contact_sheets")
    if not isinstance(contacts, dict) or set(contacts) != set(VISUAL_CATEGORIES):
        raise RuntimeError(f"Visual-QC contact-sheet set is not exact: {destination}")
    expected_contacts = {
        category: f"sub-{plan.subject_label}_ses-{session.session}_{category}_contact.png"
        for category in VISUAL_CATEGORIES
    }
    if contacts != expected_contacts:
        raise RuntimeError(f"Visual-QC contact filenames differ: {destination}")

    sources = provenance.get("source_reportlets")
    expected_source_keys = {
        (run, description)
        for run in session.runs
        for description in ("sdc", "coreg", "rois", "carpetplot")
    }
    if not isinstance(sources, list) or {
        (int(item.get("run", -1)), str(item.get("description"))) for item in sources
    } != expected_source_keys:
        raise RuntimeError(f"Visual-QC source reportlet set is not exact: {destination}")
    source_by_key = {}
    figures_root = (
        derivative_root / f"sub-{plan.subject_label}" / "figures"
    ).resolve()
    for item in sources:
        run = int(item["run"])
        description = str(item["description"])
        path = Path(str(item["path"])).resolve()
        expected_name = (
            f"sub-{plan.subject_label}_ses-{session.session}_task-iaps_"
            f"run-{run:02d}_desc-{description}_bold.svg"
        )
        if path.parent != figures_root or path.name != expected_name:
            raise RuntimeError(f"Visual-QC reportlet path is not exact: {path}")
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise RuntimeError(f"Visual-QC source reportlet changed: {path}")
        source_by_key[(run, description)] = (path, item["sha256"])

    with table_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    expected_row_keys = {
        (run, description, layer)
        for run in session.runs
        for description, layers in (
            ("sdc", ("background", "foreground")),
            ("coreg", ("background", "foreground")),
            ("rois", ("static",)),
            ("carpetplot", ("static",)),
        )
        for layer in layers
    }
    observed_row_keys = {
        (int(row["run"]), row["description"], row["layer"]) for row in rows
    }
    if len(rows) != 6 * len(session.runs) or observed_row_keys != expected_row_keys:
        raise RuntimeError(f"Visual-QC rendered-image matrix is not exact: {destination}")
    for row in rows:
        run = int(row["run"])
        description = row["description"]
        layer = row["layer"]
        source = Path(row["source_svg"]).resolve()
        output = destination / row["output_png"]
        expected_source, expected_hash = source_by_key[(run, description)]
        suffix = f"_{layer}" if layer != "static" else ""
        expected_output_name = f"run-{run:02d}_{description}{suffix}.png"
        if (
            source != expected_source
            or row["source_sha256"] != expected_hash
            or row["output_png"] != expected_output_name
            or not source.is_file()
            or sha256_file(source) != row["source_sha256"]
        ):
            raise RuntimeError(f"Rendered-image source changed: {source}")
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError(f"Rendered PNG is missing/empty: {output}")
        if int(row["width_pixels"]) < 1 or int(row["height_pixels"]) < 1:
            raise RuntimeError(f"Rendered PNG dimensions are invalid: {output}")

    expected_pngs = {
        name.format(subject=plan.subject_label)
        for name in _expected_visual_pngs(session.session, session.runs)
    }
    observed_pngs = {path.name for path in destination.glob("*.png")}
    if observed_pngs != expected_pngs:
        raise RuntimeError(
            f"Visual-QC PNG set mismatch in {destination}: "
            f"missing={sorted(expected_pngs-observed_pngs)}, "
            f"unexpected={sorted(observed_pngs-expected_pngs)}"
        )
    expected_files = {provenance_path, table_path} | {
        destination / name for name in expected_pngs
    }
    actual_files = {path for path in destination.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError(f"Unexpected visual-QC files in {destination}")
    return sorted(actual_files)


def visual_manifest_payload(
    output: Path,
    derivative_root: Path,
    plan: workflow.SubjectPlan,
    renderer_identity: Mapping[str, object],
) -> dict:
    files: list[Path] = []
    for session in plan.sessions:
        files.extend(
            inspect_visual_session(
                output / "visual" / f"ses-{session.session}",
                derivative_root,
                plan,
                session,
            )
        )
    entries = [
        {
            "relative_path": str(path.relative_to(output)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(files)
    ]
    return {
        "schema_version": 1,
        "status": "rendered_human_review_pending",
        "subject": plan.subject,
        "derivative_root": str(derivative_root.resolve()),
        "sessions": [
            {
                "session": item.session,
                "expected_sdc_mode": item.mode,
                "runs": list(item.runs),
            }
            for item in plan.sessions
        ],
        "renderer_identity": dict(renderer_identity),
        "file_count": len(entries),
        "files": entries,
    }


def validate_visual_manifest(
    output: Path, derivative_root: Path, plan: workflow.SubjectPlan
) -> dict:
    expected_top_level = {"visual", VISUAL_MANIFEST, ATTESTATION}
    if (output / VISUAL_RECEIPT).exists():
        expected_top_level.add(VISUAL_RECEIPT)
    observed_top_level = {path.name for path in output.iterdir()}
    if observed_top_level != expected_top_level:
        raise RuntimeError(
            "Visual-QC root file set is not exact: "
            f"missing={sorted(expected_top_level-observed_top_level)}, "
            f"unexpected={sorted(observed_top_level-expected_top_level)}"
        )
    path = output / VISUAL_MANIFEST
    payload = read_json(path)
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "rendered_human_review_pending"
        or payload.get("subject") != plan.subject
        or Path(str(payload.get("derivative_root"))).resolve()
        != derivative_root.resolve()
    ):
        raise RuntimeError("Visual artifact manifest identity is invalid")
    # Re-inspect both the rendered matrices and all source reportlet hashes.
    current = visual_manifest_payload(
        output, derivative_root, plan, payload.get("renderer_identity", {})
    )
    for key in ("sessions", "renderer_identity", "file_count", "files"):
        if current[key] != payload.get(key):
            raise RuntimeError(f"Visual artifact manifest changed at field {key}")
    return {
        "status": "pass",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "file_count": payload["file_count"],
    }


def write_pending_attestation(path: Path, plan: workflow.SubjectPlan, manifest_hash: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=ATTESTATION_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for session in plan.sessions:
            for run in session.runs:
                writer.writerow(
                    {
                        "subject": f"Sub{plan.subject_label}",
                        "session": f"ses-{session.session}",
                        "run": f"run-{run:02d}",
                        "expected_sdc_mode": session.mode,
                        "visual_manifest_sha256": manifest_hash,
                        "sdc_review": "PENDING",
                        "coreg_review": "PENDING",
                        "carpet_motion_review": "PENDING",
                        "overall_decision": "PENDING",
                        "reviewer": "",
                        "reviewed_at_utc": "",
                        "notes": "",
                    }
                )


def validate_attestation(
    path: Path, plan: workflow.SubjectPlan, visual_manifest_sha256: str
) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != ATTESTATION_FIELDS:
            raise RuntimeError("Visual attestation columns are not exact")
        rows = list(reader)
    expected = {
        (f"ses-{session.session}", f"run-{run:02d}"): session.mode
        for session in plan.sessions
        for run in session.runs
    }
    observed = {(row["session"], row["run"]): row["expected_sdc_mode"] for row in rows}
    if len(rows) != 10 or observed != expected:
        raise RuntimeError("Visual attestation does not contain the exact 10-run plan")
    failures = []
    for row in rows:
        key = (row["session"], row["run"])
        if row["subject"] != f"Sub{plan.subject_label}":
            failures.append((*key, "wrong subject"))
        if row["visual_manifest_sha256"] != visual_manifest_sha256:
            failures.append((*key, "wrong visual manifest hash"))
        for field in (
            "sdc_review",
            "coreg_review",
            "carpet_motion_review",
            "overall_decision",
        ):
            if row[field].strip().upper() != "ACCEPT":
                failures.append((*key, f"{field} is not ACCEPT"))
        if not row["reviewer"].strip():
            failures.append((*key, "missing reviewer"))
        try:
            timestamp = datetime.fromisoformat(row["reviewed_at_utc"].replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                raise ValueError("timezone missing")
        except ValueError:
            failures.append((*key, "invalid reviewed_at_utc"))
    if failures:
        raise RuntimeError(
            "Visual-QC remains pending/rejected; explicit ACCEPT is required for every "
            f"run. First failures: {failures[:20]}"
        )
    return {
        "status": "pass",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "reviewers": sorted({row["reviewer"].strip() for row in rows}),
        "visual_manifest_sha256": visual_manifest_sha256,
    }


def visual_gate(args: argparse.Namespace) -> dict:
    cohort, _ = workflow.load_configs(args.cohort_config, args.sdc_config)
    plan = workflow.subject_plan(cohort, args.subject)
    derivative = workflow.require_n_drive(
        args.derivative_root, "fMRIPrep derivative", allow_missing=False
    )
    output = workflow.require_n_drive(args.output, "visual-QC output")
    renderer_identity = verify_wsl_renderer(
        args.wsl_executable,
        args.wsl_distribution,
        args.wsl_renderer,
        args.wsl_renderer_sha256,
    )
    stale_stages = sorted(output.parent.glob(f".{output.name}.staging-*"))
    if stale_stages:
        raise RuntimeError(
            f"Visual-QC has preserved staging evidence requiring manual audit: {stale_stages}"
        )
    if not output.exists():
        module = load_visual_module(args.visual_script)
        stage = output.with_name(f".{output.name}.staging-{os.getpid()}")
        if stage.exists():
            raise FileExistsError(stage)
        stage.mkdir(parents=True)
        try:
            render = build_wsl_render_function(
                args.wsl_executable, args.wsl_distribution, args.wsl_renderer
            )
            with n_drive_tempfiles(stage):
                for session in plan.sessions:
                    module.run_visual_qc(
                        derivative,
                        plan.subject,
                        session.session,
                        stage / "visual" / f"ses-{session.session}",
                        expected_runs=session.runs,
                        task="iaps",
                        contact_prefix=(
                            f"sub-{plan.subject_label}_ses-{session.session}"
                        ),
                        render_svg=render,
                    )
            payload = visual_manifest_payload(stage, derivative, plan, renderer_identity)
            write_json_immutable(stage / VISUAL_MANIFEST, payload)
            write_pending_attestation(
                stage / ATTESTATION, plan, sha256_file(stage / VISUAL_MANIFEST)
            )
            os.replace(stage, output)
        except Exception:
            # Preserve a non-empty failed staging directory for audit.
            raise
    manifest = validate_visual_manifest(output, derivative, plan)
    attestation = validate_attestation(output / ATTESTATION, plan, manifest["sha256"])
    receipt_path = output / VISUAL_RECEIPT
    payload = {
        "schema_version": 1,
        "status": "pass",
        "generated_at": utc_now(),
        "subject": plan.subject,
        "visual_manifest": manifest,
        "attestation": attestation,
        "note": "Rendering alone never passes this gate; all 10 run rows were explicitly accepted.",
    }
    if receipt_path.exists():
        existing = read_json(receipt_path)
        for key in ("schema_version", "status", "subject", "visual_manifest", "attestation"):
            if existing.get(key) != payload.get(key):
                raise RuntimeError(f"Existing visual gate receipt differs at {key}")
        return existing
    write_json_immutable(receipt_path, payload)
    return payload


def docker_volume_exists(docker: Path, volume: str) -> bool:
    if not docker.is_file():
        raise FileNotFoundError(docker)
    if not VOLUME_RE.fullmatch(volume):
        raise RuntimeError(f"Unsafe Docker volume name: {volume!r}")
    result = subprocess.run(
        [str(docker), "volume", "inspect", "--format", "{{.Name}}", volume],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        if result.stdout.strip() != volume:
            raise RuntimeError(f"Docker returned a different volume identity: {result.stdout!r}")
        return True
    if result.returncode == 1 and "no such volume" in result.stderr.lower():
        return False
    raise RuntimeError(f"Docker volume inspection failed: {result.stderr.strip()}")


def _expected_volumes(plan: workflow.SubjectPlan) -> list[str]:
    return sorted(
        f"ai_iaps_full28_fmriprep_work_sub{plan.subject_label}_{branch.name}"
        for branch in plan.branches
    )


def validate_launch_receipt(
    path: Path,
    plan: workflow.SubjectPlan,
    cohort_config: Path,
    sdc_config: Path,
) -> dict:
    launch = read_json(path)
    expected_plan = workflow.plan_payload(plan, cohort_config, sdc_config)
    if (
        launch.get("status") != "pass"
        or launch.get("plan") != expected_plan
        or launch.get("work_volumes_retained_pending_archive_and_visual_qc")
        != _expected_volumes(plan)
    ):
        raise RuntimeError("fMRIPrep launch receipt/volume identity does not match the plan")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "work_volumes": _expected_volumes(plan),
    }


def verify_checksum_manifest_exact(path: Path) -> dict:
    """Verify hashes and reject both missing and newly added archive files."""

    verification = fpost.verify_checksum_manifest(path)
    payload = read_json(path)
    roots = {label: Path(value) for label, value in payload.get("roots", {}).items()}
    expected: dict[str, set[str]] = {label: set() for label in roots}
    for entry in payload.get("entries", []):
        label = entry.get("root_label")
        relative = str(entry.get("relative_path", ""))
        if label not in expected or not relative or relative in expected[label]:
            raise RuntimeError("Archive manifest has a malformed/duplicate entry")
        expected[label].add(relative)
    for label, root in roots.items():
        current = {
            str(file.relative_to(root)).replace("\\", "/")
            for file in workflow.iter_files(root)
        }
        if current != expected[label]:
            raise RuntimeError(
                f"Archive file set changed for {label}: "
                f"missing={sorted(expected[label]-current)[:20]}, "
                f"unexpected={sorted(current-expected[label])[:20]}"
            )
    return verification


def _validate_retirement_receipt(
    path: Path,
    subject: int,
    volume: str,
    manifest_hash: str,
    launch_hash: str,
    docker: Path,
) -> dict:
    value = read_json(path)
    expected = {
        "schema_version": 1,
        "status": "removed",
        "subject": subject,
        "volume": volume,
        "archive_manifest_sha256": manifest_hash,
        "launch_receipt_sha256": launch_hash,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise RuntimeError(f"Retirement receipt {path} differs at {key}")
    if docker_volume_exists(docker, volume):
        raise RuntimeError(f"Retired Docker volume has been recreated: {volume}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "volume": volume}


def retire_exact_work_volumes(
    docker: Path,
    plan: workflow.SubjectPlan,
    launch: Mapping[str, object],
    archive_manifest: Path,
    receipt_root: Path,
) -> list[dict]:
    if not docker.is_file():
        raise FileNotFoundError(docker)
    manifest = verify_checksum_manifest_exact(archive_manifest)
    manifest_hash = str(manifest["manifest_sha256"])
    launch_hash = str(launch["sha256"])
    receipts = []
    receipt_root.mkdir(parents=True, exist_ok=True)
    expected_receipt_names = {
        f"{volume}.json" for volume in _expected_volumes(plan)
    }
    unexpected = {path.name for path in receipt_root.iterdir()} - expected_receipt_names
    if unexpected or any(path.is_dir() for path in receipt_root.iterdir()):
        raise RuntimeError(
            f"Work-volume retirement directory is not exact: unexpected={sorted(unexpected)}"
        )
    for volume in _expected_volumes(plan):
        receipt = receipt_root / f"{volume}.json"
        if receipt.exists():
            receipts.append(
                _validate_retirement_receipt(
                    receipt,
                    plan.subject,
                    volume,
                    manifest_hash,
                    launch_hash,
                    docker,
                )
            )
            continue
        # The N:-resident checksum tree is re-read immediately before deletion.
        verify_checksum_manifest_exact(archive_manifest)
        if not docker_volume_exists(docker, volume):
            raise RuntimeError(
                f"Docker work volume is absent without a retirement receipt: {volume}"
            )
        result = subprocess.run(
            [str(docker), "volume", "rm", volume],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Docker refused exact volume retirement: {result.stderr.strip()}")
        if result.stdout.strip() != volume:
            raise RuntimeError(
                f"Docker removal output did not confirm the exact volume: {result.stdout!r}"
            )
        if docker_volume_exists(docker, volume):
            raise RuntimeError(f"Docker volume still exists after removal: {volume}")
        value = {
            "schema_version": 1,
            "status": "removed",
            "removed_at": utc_now(),
            "subject": plan.subject,
            "volume": volume,
            "archive_manifest_sha256": manifest_hash,
            "launch_receipt_sha256": launch_hash,
            "docker_executable": str(docker.resolve()),
            "docker_executable_sha256": sha256_file(docker),
            "stdout": result.stdout.strip(),
        }
        write_json_immutable(receipt, value)
        receipts.append(
            _validate_retirement_receipt(
                receipt,
                plan.subject,
                volume,
                manifest_hash,
                launch_hash,
                docker,
            )
        )
    verify_checksum_manifest_exact(archive_manifest)
    observed_receipt_names = {path.name for path in receipt_root.iterdir() if path.is_file()}
    if observed_receipt_names != expected_receipt_names:
        raise RuntimeError("Work-volume retirement receipt set is incomplete")
    return receipts


def archive_gate(args: argparse.Namespace) -> dict:
    if not args.execute_retirement:
        raise RuntimeError("Archive stage requires the explicit --execute-retirement guard")
    cohort, sdc = workflow.load_configs(args.cohort_config, args.sdc_config)
    plan = workflow.subject_plan(cohort, args.subject)
    derivative = workflow.require_n_drive(
        args.derivative_root, "fMRIPrep derivative", allow_missing=False
    )
    quantitative = workflow.require_n_drive(
        args.quantitative_root, "quantitative QC", allow_missing=False
    )
    visual = workflow.require_n_drive(args.visual_root, "visual QC", allow_missing=False)
    archive = workflow.require_n_drive(args.output, "archive output")
    archive.mkdir(parents=True, exist_ok=True)
    allowed_archive_names = {
        ARCHIVE_MANIFEST,
        ARCHIVE_RECEIPT,
        "work_volume_retirements",
    }
    unexpected_archive_names = {
        path.name for path in archive.iterdir()
    } - allowed_archive_names
    if unexpected_archive_names:
        raise RuntimeError(
            f"Archive destination contains unexpected entries: {sorted(unexpected_archive_names)}"
        )

    derivative_validation = fpost.validate_derivative(derivative, plan, sdc)
    visual_manifest = validate_visual_manifest(visual, derivative, plan)
    attestation = validate_attestation(
        visual / ATTESTATION, plan, visual_manifest["sha256"]
    )
    visual_receipt = read_json(visual / VISUAL_RECEIPT)
    if (
        visual_receipt.get("status") != "pass"
        or visual_receipt.get("visual_manifest") != visual_manifest
        or visual_receipt.get("attestation") != attestation
    ):
        raise RuntimeError("Visual gate receipt does not bind the accepted artifacts")
    expected_launch_path = derivative / "logs" / "production_launch_receipt.json"
    if args.launch_receipt.resolve() != expected_launch_path.resolve():
        raise RuntimeError("Launch receipt path is outside the exact derivative location")
    launch = validate_launch_receipt(
        args.launch_receipt, plan, args.cohort_config, args.sdc_config
    )

    manifest_path = archive / ARCHIVE_MANIFEST
    roots = (
        ("fmriprep", derivative),
        ("quantitative_qc", quantitative),
        ("visual_qc", visual),
    )
    if not manifest_path.exists():
        fpost.create_checksum_manifest(roots, manifest_path)
    archive_verification = verify_checksum_manifest_exact(manifest_path)
    raw_manifest = read_json(manifest_path)
    if raw_manifest.get("roots") != {
        label: str(path.resolve()) for label, path in roots
    } or {item.get("root_label") for item in raw_manifest.get("entries", [])} != {
        label for label, _ in roots
    }:
        raise RuntimeError("Archive checksum root set is not exact")
    retirements = retire_exact_work_volumes(
        args.docker_executable,
        plan,
        launch,
        manifest_path,
        archive / "work_volume_retirements",
    )
    payload = {
        "schema_version": 1,
        "status": "verified_and_work_volumes_retired",
        "generated_at": utc_now(),
        "subject": plan.subject,
        "derivative_validation": derivative_validation,
        "visual_manifest": visual_manifest,
        "visual_attestation": attestation,
        "launch_receipt": launch,
        "archive_verification": archive_verification,
        "retirements": retirements,
        "required_volumes": _expected_volumes(plan),
    }
    receipt_path = archive / ARCHIVE_RECEIPT
    if receipt_path.exists():
        existing = read_json(receipt_path)
        for key in (
            "schema_version",
            "status",
            "subject",
            "visual_manifest",
            "visual_attestation",
            "launch_receipt",
            "archive_verification",
            "retirements",
            "required_volumes",
        ):
            if existing.get(key) != payload.get(key):
                raise RuntimeError(f"Existing archive receipt differs at {key}")
        return existing
    write_json_immutable(receipt_path, payload)
    return payload


def verify_glmsingle(
    root: Path,
    subject: int,
    freeze: Mapping[str, object],
    freeze_path: Path,
    cohort_config_path: Path,
    expected_session_indicators: Sequence[int],
) -> dict:
    provenance_path = root / "provenance.json"
    validation_path = root / "validation.json"
    manifest_path = root / "trial_manifest.tsv"
    indices_path = root / "flat_mask_indices.npy"
    mask_path = root / "analysis_mask.nii.gz"
    typed_path = root / "glmsingle" / TYPE_D_NAME
    required = (
        provenance_path,
        validation_path,
        manifest_path,
        indices_path,
        mask_path,
        typed_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing GLMsingle final-gate inputs: {missing}")
    provenance = read_json(provenance_path)
    validation = read_json(validation_path)
    frozen_glm = freeze["glmsingle"]
    entry = provenance.get("full_cohort_entry_point")
    fracridge = entry.get("fracridge_vendor_validation") if isinstance(entry, dict) else None
    glm_vendor = provenance.get("glmsingle_vendor_validation")
    freeze_path = freeze_path.resolve()
    cohort_config_path = cohort_config_path.resolve()
    frozen_fracs = np.asarray(frozen_glm["fractional_ridge_grid"], dtype=float)
    observed_fracs = np.asarray(provenance.get("fracs", []), dtype=float)
    expected_session_indicators = [int(value) for value in expected_session_indicators]
    if (
        provenance.get("subject") != subject
        or provenance.get("glmsingle_commit") != frozen_glm["commit"]
        or provenance.get("n_pcs") != frozen_glm["candidate_noise_pcs"]
        or observed_fracs.shape != frozen_fracs.shape
        or not np.allclose(observed_fracs, frozen_fracs, rtol=0, atol=1e-12)
        or provenance.get("seed")
        != freeze["random_seed_for_matched_nonhistorical_code"]
        or provenance.get("space") != "MNI152NLin6Asym"
        or not isinstance(entry, dict)
        or entry.get("production_contract_enforced") is not True
        or Path(str(entry.get("freeze_path"))).resolve() != freeze_path
        or entry.get("freeze_sha256") != sha256_file(freeze_path)
        or Path(str(entry.get("cohort_config_path"))).resolve() != cohort_config_path
        or entry.get("cohort_config_sha256") != sha256_file(cohort_config_path)
        or not isinstance(fracridge, dict)
        or fracridge.get("validated") is not True
        or fracridge.get("record_sha256")
        != frozen_glm["fracridge_vendor"]["record_sha256"]
        or not isinstance(glm_vendor, dict)
        or glm_vendor.get("validated") is not True
        or glm_vendor.get("commit") != frozen_glm["commit"]
        or glm_vendor.get("source_sha256") != frozen_glm["source_tree_sha256"]
        or [item.get("sha256") for item in glm_vendor.get("patches", [])]
        != frozen_glm["ordered_patch_sha256"]
    ):
        raise RuntimeError("GLMsingle provenance does not match the frozen production model")
    indicators = entry.get("session_indicators")
    if (
        not isinstance(indicators, list)
        or [int(value) for value in indicators] != expected_session_indicators
        or [int(value) for value in provenance.get("sessionindicator", [])]
        != expected_session_indicators
    ):
        raise RuntimeError("GLMsingle provenance has incorrect session indicators")
    run_n_volumes = provenance.get("run_n_volumes")
    if (
        not isinstance(run_n_volumes, list)
        or len(run_n_volumes) != 10
        or any(int(value) < 2 for value in run_n_volumes)
    ):
        raise RuntimeError("GLMsingle provenance has invalid per-run volume counts")

    n_voxels = int(provenance.get("n_voxels", 0))
    if n_voxels < 1:
        raise RuntimeError("GLMsingle provenance has no analysis voxels")
    expected_shape = (n_voxels, 1, 1, 600)
    if (
        tuple(validation.get("betasmd_shape", [])) != expected_shape
        or validation.get("betasmd_all_values_finite") is not True
    ):
        raise RuntimeError("GLMsingle validation JSON lacks 600 finite Type-D betas")
    with h5py.File(typed_path, "r") as handle:
        required_datasets = {"betasmd", "HRFindex", "FRACvalue", "R2"}
        if not required_datasets.issubset(handle.keys()):
            raise RuntimeError("Type-D HDF5 is missing a required dataset")
        betas = handle["betasmd"]
        if tuple(betas.shape) != expected_shape:
            raise RuntimeError(f"Type-D beta shape changed: {betas.shape}")
        if str(betas.dtype) != "float32" or validation.get("betasmd_dtype") != "float32":
            raise RuntimeError(f"Type-D beta dtype must be float32, found {betas.dtype}")
        beta_min, beta_max = np.inf, -np.inf
        for start in range(0, n_voxels, 4096):
            block = np.asarray(betas[start : start + 4096, ..., :])
            if not np.isfinite(block).all():
                raise RuntimeError(f"Nonfinite Type-D beta in voxel block starting {start}")
            beta_min = min(beta_min, float(block.min()))
            beta_max = max(beta_max, float(block.max()))
        for name in ("HRFindex", "FRACvalue", "R2"):
            values = np.asarray(handle[name])
            if values.shape != (n_voxels, 1, 1) or not np.isfinite(values).all():
                raise RuntimeError(f"Invalid/nonfinite Type-D {name}")
            if name == "FRACvalue":
                distances = np.abs(
                    values.astype(float).reshape(-1, 1) - frozen_fracs.reshape(1, -1)
                )
                if not np.all(np.min(distances, axis=1) <= 1e-6):
                    raise RuntimeError("Type-D FRACvalue is outside the frozen ridge grid")
    if not (
        np.isclose(beta_min, float(validation.get("betasmd_min")), rtol=1e-6, atol=1e-6)
        and np.isclose(beta_max, float(validation.get("betasmd_max")), rtol=1e-6, atol=1e-6)
    ):
        raise RuntimeError("Type-D min/max differ from validation.json")
    indices = np.load(indices_path, allow_pickle=False)
    if indices.ndim != 1 or len(indices) != n_voxels or len(np.unique(indices)) != n_voxels:
        raise RuntimeError("GLMsingle flat mask index set is invalid")
    mask = np.asarray(nib.load(mask_path).dataobj) > 0
    if (
        int(mask.sum()) != n_voxels
        or not np.array_equal(indices.astype(np.int64), np.flatnonzero(mask.ravel()))
    ):
        raise RuntimeError("GLMsingle analysis mask voxel count differs from Type-D")

    trials = pd.read_csv(manifest_path, sep="\t")
    if (
        len(trials) != 600
        or "beta_index" not in trials
        or trials["beta_index"].astype(int).tolist() != list(range(600))
        or "run" not in trials
        or trials.groupby("run").size().to_dict() != {run: 60 for run in range(1, 11)}
        or "image_id" not in trials
        or len(trials["image_id"].astype(str).unique()) != 120
        or not (trials["image_id"].astype(str).value_counts() == 5).all()
        or "onset" not in trials
        or not np.isfinite(pd.to_numeric(trials["onset"], errors="coerce")).all()
    ):
        raise RuntimeError("GLMsingle trial manifest is not the exact 600-trial chronology")
    return {
        "status": "pass",
        "subject": subject,
        "n_voxels": n_voxels,
        "trial_count": 600,
        "typed_shape": list(expected_shape),
        "typed_all_finite": True,
        "typed_min": beta_min,
        "typed_max": beta_max,
        "typed_size_bytes": typed_path.stat().st_size,
        "provenance_sha256": sha256_file(provenance_path),
        "validation_sha256": sha256_file(validation_path),
        "trial_manifest_sha256": sha256_file(manifest_path),
        "analysis_mask_sha256": sha256_file(mask_path),
        "flat_mask_indices_sha256": sha256_file(indices_path),
    }


def verify_erp(root: Path, subject: int, freeze_path: Path, glmsingle_root: Path) -> dict:
    tables = {
        HISTORICAL_METHOD + "_subject": root / "historical_avg_random_subject_results.csv",
        HISTORICAL_METHOD + "_fold": root / "historical_avg_random_fold_results.csv",
        RUNWISE_METHOD + "_subject": root / "runwise_loro_subject_results.csv",
        RUNWISE_METHOD + "_fold": root / "runwise_loro_fold_results.csv",
        "roi_counts": root / "roi_voxel_counts.csv",
    }
    required = [*tables.values(), root / "validation.json", root / "provenance.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing ERP final-gate inputs: {missing}")
    frames = {name: pd.read_csv(path) for name, path in tables.items()}
    validate_subject_result_rows(frames[HISTORICAL_METHOD + "_subject"], subject, HISTORICAL_METHOD)
    validate_fold_result_rows(frames[HISTORICAL_METHOD + "_fold"], subject, HISTORICAL_METHOD)
    validate_subject_result_rows(frames[RUNWISE_METHOD + "_subject"], subject, RUNWISE_METHOD)
    validate_fold_result_rows(frames[RUNWISE_METHOD + "_fold"], subject, RUNWISE_METHOD)
    validate_roi_count_rows(frames["roi_counts"], subject)

    validation = read_json(root / "validation.json")
    provenance = read_json(root / "provenance.json")
    freeze = read_json(freeze_path)
    if (
        validation.get("status") != "complete"
        or validation.get("subject") != f"Sub{subject}"
        or tuple(validation.get("pipelines", [])) != PIPELINES
        or tuple(validation.get("roi_order", [])) != ROI_ORDER
        or validation.get("all_accuracy_values_finite_and_in_0_1") is not True
        or provenance.get("status") != "complete"
        or provenance.get("subject") != f"Sub{subject}"
        or provenance.get("analysis_id") != freeze.get("analysis_id")
        or set(provenance.get("methods", {}))
        != {HISTORICAL_METHOD, RUNWISE_METHOD}
    ):
        raise RuntimeError("ERP validation/provenance identity is incomplete")
    request = provenance.get("request")
    if not isinstance(request, dict):
        raise RuntimeError("ERP provenance request is missing")
    embedded_signature = request.get("request_signature_sha256")
    unsigned = dict(request)
    unsigned.pop("request_signature_sha256", None)
    if (
        embedded_signature != stable_json_sha256(unsigned)
        or provenance.get("request_signature_sha256") != embedded_signature
        or request.get("analysis_id") != freeze.get("analysis_id")
        or request.get("subject") != subject
        or Path(str(request.get("freeze", {}).get("path"))).resolve()
        != freeze_path.resolve()
        or request.get("freeze", {}).get("sha256") != sha256_file(freeze_path)
        or Path(str(request.get("glmsingle_subject_root"))).resolve()
        != glmsingle_root.resolve()
    ):
        raise RuntimeError("ERP request signature/provenance no longer matches its inputs")
    glmsingle_inputs = request.get("glmsingle_inputs", {})
    if set(glmsingle_inputs) != {
        "typed_hdf5",
        "analysis_mask",
        "flat_mask_indices",
        "trial_manifest",
        "validation",
        "provenance",
    }:
        raise RuntimeError("ERP request does not pin the exact GLMsingle input set")
    for name, record in glmsingle_inputs.items():
        if not isinstance(record, dict) or "path" not in record:
            raise RuntimeError(f"ERP GLMsingle request record is malformed: {name}")
        if quick_file_signature(Path(record["path"])) != record:
            raise RuntimeError(f"ERP GLMsingle input changed after decoding: {name}")
    expected_rows = {
        HISTORICAL_METHOD + "_subject": 3 * 8 * 17,
        RUNWISE_METHOD + "_subject": 3 * 8 * 17,
        "roi_counts": 3 * 17,
    }
    for name, count in expected_rows.items():
        if len(frames[name]) != count:
            raise RuntimeError(f"ERP table {name} has {len(frames[name])} rows; expected {count}")
    expected_row_counts = {
        "historical_subject_rows": len(frames[HISTORICAL_METHOD + "_subject"]),
        "historical_fold_rows": len(frames[HISTORICAL_METHOD + "_fold"]),
        "runwise_subject_rows": len(frames[RUNWISE_METHOD + "_subject"]),
        "runwise_fold_rows": len(frames[RUNWISE_METHOD + "_fold"]),
        "roi_count_rows": len(frames["roi_counts"]),
    }
    if (
        provenance.get("row_counts") != expected_row_counts
        or validation.get("row_counts") != expected_row_counts
    ):
        raise RuntimeError("ERP provenance/validation row counts differ from the tables")
    return {
        "status": "pass",
        "subject": subject,
        "pipelines": list(PIPELINES),
        "methods": [HISTORICAL_METHOD, RUNWISE_METHOD],
        "roi_count": len(ROI_ORDER),
        "contrasts": list(CONTRASTS),
        "table_rows": {name: len(frame) for name, frame in frames.items()},
        "files": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in required
        },
    }


def verify_archive(
    archive: Path,
    derivative: Path,
    quantitative: Path,
    visual: Path,
    plan: workflow.SubjectPlan,
    docker: Path,
    visual_manifest: Mapping[str, object],
    visual_attestation: Mapping[str, object],
) -> dict:
    manifest_path = archive / ARCHIVE_MANIFEST
    receipt_path = archive / ARCHIVE_RECEIPT
    expected_archive_top = {
        ARCHIVE_MANIFEST,
        ARCHIVE_RECEIPT,
        "work_volume_retirements",
    }
    if {path.name for path in archive.iterdir()} != expected_archive_top:
        raise RuntimeError("Final archive directory file set is not exact")
    verification = verify_checksum_manifest_exact(manifest_path)
    raw = read_json(manifest_path)
    expected_roots = {
        "fmriprep": str(derivative.resolve()),
        "quantitative_qc": str(quantitative.resolve()),
        "visual_qc": str(visual.resolve()),
    }
    if raw.get("roots") != expected_roots:
        raise RuntimeError("Archive roots differ from final-gate inputs")
    receipt = read_json(receipt_path)
    derivative_validation = receipt.get("derivative_validation", {})
    if (
        receipt.get("status") != "verified_and_work_volumes_retired"
        or receipt.get("subject") != plan.subject
        or receipt.get("archive_verification") != verification
        or receipt.get("required_volumes") != _expected_volumes(plan)
        or receipt.get("visual_manifest") != dict(visual_manifest)
        or receipt.get("visual_attestation") != dict(visual_attestation)
        or derivative_validation.get("status") != "pass"
        or derivative_validation.get("subject") != plan.subject
        or derivative_validation.get("run_count") != 10
    ):
        raise RuntimeError("Archive receipt is invalid")
    launch = receipt.get("launch_receipt", {})
    launch_path = Path(str(launch.get("path", "")))
    if (
        not launch_path.is_file()
        or launch.get("sha256") != sha256_file(launch_path)
        or launch.get("work_volumes") != _expected_volumes(plan)
    ):
        raise RuntimeError("Archived fMRIPrep launch receipt is invalid")
    checked = []
    receipt_root = archive / "work_volume_retirements"
    expected_receipt_names = {
        f"{volume}.json" for volume in _expected_volumes(plan)
    }
    if (
        not receipt_root.is_dir()
        or {path.name for path in receipt_root.iterdir()} != expected_receipt_names
        or any(path.is_dir() for path in receipt_root.iterdir())
    ):
        raise RuntimeError("Final work-volume retirement receipt set is not exact")
    for volume, record in zip(
        _expected_volumes(plan), receipt.get("retirements", []), strict=True
    ):
        path = Path(str(record.get("path", "")))
        if record.get("sha256") != sha256_file(path):
            raise RuntimeError(f"Retirement receipt hash changed: {path}")
        checked.append(
            _validate_retirement_receipt(
                path,
                plan.subject,
                volume,
                verification["manifest_sha256"],
                str(launch.get("sha256")),
                docker,
            )
        )
    return {
        "status": "pass",
        "manifest": verification,
        "receipt_sha256": sha256_file(receipt_path),
        "retirements": checked,
    }


def verify_ledger(
    subject: int,
    freeze_path: Path,
    contract_path: Path,
    approval_path: Path,
    log_root: Path,
) -> dict:
    approval, approval_record, freeze, contract, identities = validate_approval(
        approval_path, freeze_path, contract_path
    )
    if subject not in approval["production_subjects"]:
        raise RuntimeError(f"Sub{subject} is not in the approved production cohort")
    _, identity = build_control_identity(freeze["analysis_id"], approval_record, identities)
    state = read_ledger(log_root / "ledger.jsonl")
    verify_event_mirror(log_root, state)
    verify_attempt_evidence(state)
    completions = latest_stage_completions(state.events)
    order = list(contract["stage_order"])
    if order[-1] != "subject_gate":
        raise RuntimeError("subject_gate is not the final production stage")
    prior = order[:-1]
    completion_hashes = {}
    for stage in prior:
        event = completions.get((subject, stage))
        if (
            event is None
            or event.get("status") != "passed"
            or event.get("control_identity_sha256") != identity
        ):
            raise RuntimeError(f"Ledger lacks an identity-matched passed {stage} stage")
        completion_hashes[stage] = event["event_sha256"]
    wrong_identity_events = [
        (event.get("sequence"), event.get("stage"), event.get("event"))
        for event in state.events
        if event.get("subject") == subject
        and event.get("control_identity_sha256") != identity
    ]
    if wrong_identity_events:
        raise RuntimeError(
            "Subject ledger contains events under another control identity: "
            f"{wrong_identity_events}"
        )
    if completions.get((subject, "subject_gate")) is not None:
        raise RuntimeError("subject_gate already has a completion event")
    started_ids = {
        str(event["attempt_id"]): event
        for event in state.events
        if event.get("event") == "started"
        and event.get("subject") == subject
        and event.get("stage") == "subject_gate"
    }
    completed_ids = {
        str(event["attempt_id"])
        for event in state.events
        if event.get("event") == "completed"
        and event.get("subject") == subject
        and event.get("stage") == "subject_gate"
    }
    incomplete = sorted(set(started_ids) - completed_ids)
    if len(incomplete) != 1:
        raise RuntimeError(
            f"Expected exactly the current subject_gate attempt in the ledger; found {incomplete}"
        )
    current = started_ids[incomplete[0]]
    if current.get("control_identity_sha256") != identity:
        raise RuntimeError("Current subject_gate attempt uses a different control identity")
    attempt_path = Path(str(current["attempt_dir"])) / "attempt.json"
    attempt = read_json(attempt_path)
    if any(
        attempt.get(key) != current.get(key)
        for key in ("attempt_id", "subject", "stage", "control_identity_sha256")
    ):
        raise RuntimeError("Current subject_gate attempt.json differs from the ledger")
    return {
        "status": "pass",
        "control_identity_sha256": identity,
        "ledger_head_at_gate": state.head_sha256,
        "prior_stage_completion_event_sha256": completion_hashes,
        "current_subject_gate_started_event_sha256": current["event_sha256"],
        "current_attempt_id": current["attempt_id"],
    }


def subject_gate(args: argparse.Namespace) -> dict:
    cohort, _ = workflow.load_configs(args.cohort_config, args.sdc_config)
    plan = workflow.subject_plan(cohort, args.subject)
    freeze = read_json(args.freeze)
    if freeze.get("analysis_id") != "ai_iaps_full28_sdc_glmsingle_erp_v1":
        raise RuntimeError("Unexpected frozen analysis identity")
    derivative_root = workflow.require_n_drive(
        args.derivative_root, "final-gate fMRIPrep derivative", allow_missing=False
    )
    quantitative_root = workflow.require_n_drive(
        args.quantitative_root, "final-gate quantitative QC", allow_missing=False
    )
    visual_root = workflow.require_n_drive(
        args.visual_root, "final-gate visual QC", allow_missing=False
    )
    archive_root = workflow.require_n_drive(
        args.archive_root, "final-gate archive", allow_missing=False
    )
    glmsingle_root = workflow.require_n_drive(
        args.glmsingle_root, "final-gate GLMsingle", allow_missing=False
    )
    erp_root = workflow.require_n_drive(
        args.erp_root, "final-gate ERP", allow_missing=False
    )
    output = workflow.require_n_drive(args.output, "final-gate receipt")
    stale_receipts = sorted(output.parent.glob(f".{output.name}.staging-*"))
    if stale_receipts:
        raise RuntimeError(
            f"Final gate has preserved staging evidence requiring manual audit: {stale_receipts}"
        )
    visual_manifest = validate_visual_manifest(visual_root, derivative_root, plan)
    attestation = validate_attestation(
        visual_root / ATTESTATION, plan, visual_manifest["sha256"]
    )
    archive = verify_archive(
        archive_root,
        derivative_root,
        quantitative_root,
        visual_root,
        plan,
        args.docker_executable,
        visual_manifest,
        attestation,
    )
    glmsingle = verify_glmsingle(
        glmsingle_root,
        args.subject,
        freeze,
        args.freeze,
        args.cohort_config,
        [
            next(
                int(session.session)
                for session in plan.sessions
                if run in session.runs
            )
            for run in range(1, 11)
        ],
    )
    erp = verify_erp(
        erp_root, args.subject, args.freeze.resolve(), glmsingle_root
    )
    ledger = verify_ledger(
        args.subject,
        args.freeze,
        args.contract,
        args.approval,
        args.log_root,
    )
    payload = {
        "schema_version": 1,
        "status": "pass",
        "generated_at": utc_now(),
        "analysis_id": freeze["analysis_id"],
        "subject": args.subject,
        "visual_manifest": visual_manifest,
        "visual_attestation": attestation,
        "archive": archive,
        "glmsingle": glmsingle,
        "erp": erp,
        "ledger": ledger,
    }
    write_json_immutable(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-config", type=Path, required=True)
    parser.add_argument("--sdc-config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    visual = commands.add_parser("visual-qc")
    visual.add_argument("--subject", type=int, required=True)
    visual.add_argument("--derivative-root", type=Path, required=True)
    visual.add_argument("--output", type=Path, required=True)
    visual.add_argument("--visual-script", type=Path, default=DEFAULT_VISUAL_SCRIPT)
    visual.add_argument("--wsl-executable", type=Path, required=True)
    visual.add_argument("--wsl-distribution", required=True)
    visual.add_argument("--wsl-renderer", required=True)
    visual.add_argument("--wsl-renderer-sha256", required=True)

    archive = commands.add_parser("archive")
    archive.add_argument("--subject", type=int, required=True)
    archive.add_argument("--derivative-root", type=Path, required=True)
    archive.add_argument("--quantitative-root", type=Path, required=True)
    archive.add_argument("--visual-root", type=Path, required=True)
    archive.add_argument("--launch-receipt", type=Path, required=True)
    archive.add_argument("--output", type=Path, required=True)
    archive.add_argument("--docker-executable", type=Path, required=True)
    archive.add_argument("--execute-retirement", action="store_true")

    gate = commands.add_parser("subject-gate")
    gate.add_argument("--subject", type=int, required=True)
    gate.add_argument("--freeze", type=Path, required=True)
    gate.add_argument("--contract", type=Path, required=True)
    gate.add_argument("--approval", type=Path, required=True)
    gate.add_argument("--log-root", type=Path, required=True)
    gate.add_argument("--derivative-root", type=Path, required=True)
    gate.add_argument("--quantitative-root", type=Path, required=True)
    gate.add_argument("--visual-root", type=Path, required=True)
    gate.add_argument("--archive-root", type=Path, required=True)
    gate.add_argument("--glmsingle-root", type=Path, required=True)
    gate.add_argument("--erp-root", type=Path, required=True)
    gate.add_argument("--docker-executable", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "visual-qc":
            payload = visual_gate(args)
        elif args.command == "archive":
            payload = archive_gate(args)
        else:
            payload = subject_gate(args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (
        FileNotFoundError,
        FileExistsError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
        OSError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
