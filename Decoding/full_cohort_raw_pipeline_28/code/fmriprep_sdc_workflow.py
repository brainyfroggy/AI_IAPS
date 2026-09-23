#!/usr/bin/env python3
"""Fail-closed, session-aware fMRIPrep orchestration for the remaining cohort.

The module deliberately separates planning/preflight from execution.  It never
launches fMRIPrep unless ``run-subject --execute`` is supplied and an immutable
preflight receipt matches the current BIDS inputs byte-for-byte.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


# Do not resolve the mapped N: drive to its UNC target. Docker Desktop could not
# bind that Windows network path, while its Ubuntu-22.04 integration can bind
# the same bytes through /mnt/n.  Keeping the drive-qualified spelling here is
# therefore part of the container-launch contract, not cosmetic path handling.
HERE = Path(os.path.abspath(__file__)).parent
ROOT = HERE.parent
DEFAULT_COHORT_CONFIG = ROOT / "config" / "cohort.json"
DEFAULT_SDC_CONFIG = ROOT / "config" / "fmriprep_sdc_config.json"
DEFAULT_SYN_FILTER = ROOT / "config" / "syn_fmap_bids_filter.json"
DEFAULT_PROBE = HERE / "sdcflows_estimator_probe.py"
MEASURED_MODES = {"pepolar", "gre"}
ALL_MODES = MEASURED_MODES | {"syn"}
RUN_RE = re.compile(r"_run-(\d+)_")
SUMMARY_RE = re.compile(
    r"susceptibility\s+distortion\s+correction\s*:\s*([^<\r\n]+)", re.I
)


@dataclass(frozen=True)
class SessionSpec:
    session: str
    mode: str
    runs: tuple[int, ...]


@dataclass(frozen=True)
class BranchSpec:
    name: str
    mode: str
    sessions: tuple[str, ...]
    runs: tuple[int, ...]
    input_kind: str
    syn_filter: bool


@dataclass(frozen=True)
class SubjectPlan:
    subject: int
    subject_label: str
    strategy: str
    sessions: tuple[SessionSpec, ...]
    branches: tuple[BranchSpec, ...]


def branch_payload(branch: BranchSpec) -> dict:
    """Return the canonical JSON representation used in persisted receipts."""

    return {
        "name": branch.name,
        "mode": branch.mode,
        "sessions": list(branch.sessions),
        "runs": list(branch.runs),
        "input_kind": branch.input_kind,
        "syn_filter": branch.syn_filter,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_configs(cohort_path: Path, sdc_path: Path) -> tuple[dict, dict]:
    cohort = read_json(cohort_path)
    sdc = read_json(sdc_path)
    if cohort.get("schema_version") != 2:
        raise ValueError(f"Unsupported cohort schema: {cohort.get('schema_version')}")
    if sdc.get("schema_version") != 1:
        raise ValueError(f"Unsupported fMRIPrep SDC schema: {sdc.get('schema_version')}")
    if sdc.get("analysis_id") != "ai_iaps_full28_sdc_glmsingle_erp_v1":
        raise ValueError("Unexpected analysis_id in fMRIPrep SDC config")
    image = str(sdc["fmriprep"]["image"])
    digest = str(sdc["fmriprep"]["image_digest"])
    if not image.endswith("@" + digest):
        raise ValueError("Pinned fMRIPrep image reference and digest disagree")
    launcher = sdc.get("docker_launcher")
    if not isinstance(launcher, dict):
        raise ValueError("Missing docker_launcher configuration")
    expected_launcher = {
        "mode": "wsl",
        "windows_wsl_executable": r"C:\Windows\System32\wsl.exe",
        "distribution": "Ubuntu-22.04",
        "linux_docker_executable": "/usr/bin/docker",
        "persistent_bind_prefix": "/mnt/n",
    }
    for key, expected in expected_launcher.items():
        if launcher.get(key) != expected:
            raise ValueError(
                f"Unsupported Docker launcher {key}: {launcher.get(key)!r}; expected {expected!r}"
            )
    if launcher["distribution"] != sdc["validator"]["wsl_distribution"]:
        raise ValueError("Docker and BIDS-validator WSL distributions must match")
    return cohort, sdc


def subject_plan(cohort: Mapping, subject: int) -> SubjectPlan:
    remaining = {int(item) for item in cohort["cohort"]["remaining_subjects"]}
    if subject not in remaining:
        raise ValueError(f"Sub{subject} is not one of the frozen remaining 25 subjects")
    entry = cohort["subjects"].get(str(subject))
    if not isinstance(entry, dict):
        raise ValueError(f"No session plan exists for Sub{subject}")
    sessions: list[SessionSpec] = []
    seen_runs: set[int] = set()
    for raw in entry.get("sessions", []):
        label = f"{int(raw['session']):02d}"
        mode = str(raw["sdc_mode"]).lower()
        runs = tuple(int(item) for item in raw["experimental_runs"])
        if mode not in ALL_MODES:
            raise ValueError(f"Unsupported SDC mode for Sub{subject} ses-{label}: {mode}")
        if not runs or len(set(runs)) != len(runs):
            raise ValueError(f"Invalid run list for Sub{subject} ses-{label}: {runs}")
        overlap = seen_runs.intersection(runs)
        if overlap:
            raise ValueError(f"Runs assigned twice for Sub{subject}: {sorted(overlap)}")
        seen_runs.update(runs)
        sessions.append(SessionSpec(label, mode, runs))
    if seen_runs != set(range(1, 11)):
        raise ValueError(f"Sub{subject} does not cover experimental runs 1-10: {sorted(seen_runs)}")
    sessions.sort(key=lambda item: item.session)
    modes = {item.mode for item in sessions}
    if modes <= MEASURED_MODES and len(modes) != 1:
        raise ValueError(
            f"Untested measured-method mixture for Sub{subject}: {sorted(modes)}; add an explicit policy"
        )
    if modes <= MEASURED_MODES:
        mode = next(iter(modes))
        branches = (
            BranchSpec(
                "all_measured",
                mode,
                tuple(item.session for item in sessions),
                tuple(sorted(seen_runs)),
                "canonical",
                False,
            ),
        )
        strategy = "direct_all_measured_sessions"
    elif len(sessions) == 1 and modes == {"syn"}:
        item = sessions[0]
        branches = (
            BranchSpec("all_syn", "syn", (item.session,), item.runs, "canonical", True),
        )
        strategy = "direct_single_syn_session"
    elif "syn" in modes and modes.intersection(MEASURED_MODES):
        branches = tuple(
            BranchSpec(
                f"ses_{item.session}_{item.mode}",
                item.mode,
                (item.session,),
                item.runs,
                "isolated_session",
                item.mode == "syn",
            )
            for item in sessions
        )
        strategy = "isolated_mixed_session_branches"
    else:
        raise ValueError(f"No approved branch policy for Sub{subject}: {sorted(modes)}")
    return SubjectPlan(subject, f"{subject:02d}", strategy, tuple(sessions), branches)


def plan_payload(plan: SubjectPlan, cohort_path: Path, sdc_path: Path) -> dict:
    payload = asdict(plan)
    payload["cohort_config"] = str(cohort_path.resolve())
    payload["cohort_config_sha256"] = sha256_file(cohort_path)
    payload["sdc_config"] = str(sdc_path.resolve())
    payload["sdc_config_sha256"] = sha256_file(sdc_path)
    payload["syn_filter_sha256"] = sha256_file(DEFAULT_SYN_FILTER)
    payload["probe_sha256"] = sha256_file(DEFAULT_PROBE)
    payload["workflow_sha256"] = sha256_file(Path(__file__))
    payload["plan_sha256"] = canonical_json_hash(asdict(plan))
    return payload


def require_n_drive(path: Path, description: str, allow_missing: bool = True) -> Path:
    absolute = Path(os.path.abspath(path))
    n_root = Path("N:\\").resolve()
    try:
        same_n_storage = absolute.resolve(strict=False) == n_root or n_root in absolute.resolve(strict=False).parents
    except OSError:
        same_n_storage = False
    if absolute.drive.upper() != "N:" and not same_n_storage:
        raise ValueError(f"{description} must be on N:, not {absolute}")
    if not allow_missing and not absolute.exists():
        raise FileNotFoundError(f"{description} does not exist: {absolute}")
    return absolute


def expected_input_path(plan: SubjectPlan, branch: BranchSpec, canonical: Path, branch_root: Path) -> Path:
    return canonical if branch.input_kind == "canonical" else branch_root / branch.name


def expected_output_path(branch: BranchSpec, output_root: Path) -> Path:
    return output_root / "branches" / branch.name


def copy_or_link(src: Path, dst: Path, mode: str) -> str:
    if dst.exists():
        raise FileExistsError(f"Refusing to replace {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode in {"hardlink", "auto"}:
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(src, dst)
    return "copy"


def iter_files(root: Path) -> list[Path]:
    return sorted(item for item in root.rglob("*") if item.is_file())


def build_isolated_branch(
    canonical: Path,
    destination: Path,
    plan: SubjectPlan,
    branch: BranchSpec,
    link_mode: str,
) -> dict:
    if branch.input_kind != "isolated_session" or len(branch.sessions) != 1:
        raise ValueError(f"Not an isolated-session branch: {branch}")
    canonical = require_n_drive(canonical, "canonical BIDS root", allow_missing=False)
    destination = require_n_drive(destination, "branch BIDS destination")
    if destination.exists():
        raise FileExistsError(f"Branch destination must be absent: {destination}")
    subject_dir = canonical / f"sub-{plan.subject_label}"
    if not subject_dir.is_dir():
        raise FileNotFoundError(subject_dir)
    session = branch.sessions[0]
    selected_session = subject_dir / f"ses-{session}"
    if not selected_session.is_dir():
        raise FileNotFoundError(selected_session)
    stage = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if stage.exists():
        raise FileExistsError(stage)

    required_root = ["dataset_description.json", "participants.tsv", "task-iaps_events.json"]
    optional_root = ["README", ".bidsignore"]
    selected: list[Path] = []
    for name in required_root:
        path = canonical / name
        if not path.is_file():
            raise FileNotFoundError(path)
        selected.append(path)
    selected.extend(canonical / name for name in optional_root if (canonical / name).is_file())
    # Every T1w is retained in both branches so both fMRIPrep branches build the
    # same subject anatomical reference. Functional and fmap inputs remain
    # strictly session-isolated.
    for session_dir in sorted(subject_dir.glob("ses-*")):
        anat = session_dir / "anat"
        if anat.is_dir():
            selected.extend(iter_files(anat))
    func = selected_session / "func"
    if not func.is_dir():
        raise FileNotFoundError(func)
    selected.extend(iter_files(func))
    fmap = selected_session / "fmap"
    if branch.mode == "syn":
        if fmap.exists() and iter_files(fmap):
            raise RuntimeError(f"SyN branch unexpectedly has same-session fieldmaps: {fmap}")
    else:
        if not fmap.is_dir() or not iter_files(fmap):
            raise RuntimeError(f"Measured branch lacks same-session fieldmaps: {fmap}")
        selected.extend(iter_files(fmap))
    if len(selected) != len(set(selected)):
        raise RuntimeError("Duplicate input path in isolated branch plan")

    counts = {"hardlink": 0, "copy": 0}
    try:
        for src in selected:
            rel = src.relative_to(canonical)
            method = copy_or_link(src, stage / rel, link_mode)
            counts[method] += 1
        participants = stage / "participants.tsv"
        with participants.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        chosen = [row for row in rows if row.get("participant_id") == f"sub-{plan.subject_label}"]
        if len(chosen) != 1:
            raise RuntimeError("participants.tsv does not contain exactly the selected participant")
        # Break a possible hard link before rewriting.
        participants.unlink()
        with participants.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(chosen[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(chosen)
        stage.replace(destination)
    except Exception:
        # Preserve a nonempty stage for diagnosis. Empty stages can be removed.
        if stage.exists() and not any(stage.iterdir()):
            stage.rmdir()
        raise
    return {
        "destination": str(destination),
        "session": session,
        "mode": branch.mode,
        "files": len(selected),
        "materialization": counts,
    }


def _normalize_values(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(item) for item in value}
    return {str(value)}


def _opposite_phase_encoding(first: str, second: str) -> bool:
    return first.rstrip("-") == second.rstrip("-") and first.endswith("-") != second.endswith("-")


def run_number(path: Path) -> int:
    match = RUN_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse run entity: {path}")
    return int(match.group(1))


def audit_bids_branch(bids_root: Path, plan: SubjectPlan, branch: BranchSpec) -> dict:
    bids_root = require_n_drive(bids_root, "BIDS branch", allow_missing=False)
    if not (bids_root / "dataset_description.json").is_file():
        raise FileNotFoundError(f"Missing dataset_description.json under {bids_root}")
    subject_root = bids_root / f"sub-{plan.subject_label}"
    if not subject_root.is_dir():
        raise FileNotFoundError(subject_root)
    observed_subjects = sorted(path.name for path in bids_root.glob("sub-*") if path.is_dir())
    if observed_subjects != [f"sub-{plan.subject_label}"]:
        raise RuntimeError(f"BIDS branch contains wrong subjects: {observed_subjects}")
    expected_runs_by_session = {
        item.session: set(item.runs) for item in plan.sessions if item.session in branch.sessions
    }
    observed_bolds: dict[str, dict[int, Path]] = {}
    associations: dict[str, dict] = {}
    all_field_ids: dict[str, str] = {}

    # All subject T1w acquisitions must be retained in isolated branches.
    expected_anat_sessions = {item.session for item in plan.sessions}
    observed_anat_sessions = {
        path.parent.parent.name.removeprefix("ses-")
        for path in subject_root.glob("ses-*/anat/*_T1w.nii.gz")
    }
    if observed_anat_sessions != expected_anat_sessions:
        raise RuntimeError(
            f"T1w session set mismatch: expected {sorted(expected_anat_sessions)}, "
            f"observed {sorted(observed_anat_sessions)}"
        )

    for session in branch.sessions:
        func = subject_root / f"ses-{session}" / "func"
        hits = sorted(func.glob(f"sub-{plan.subject_label}_ses-{session}_task-iaps_run-*_bold.nii.gz"))
        by_run = {run_number(path): path for path in hits}
        if len(by_run) != len(hits) or set(by_run) != expected_runs_by_session[session]:
            raise RuntimeError(
                f"BOLD run mismatch for ses-{session}: expected {sorted(expected_runs_by_session[session])}, "
                f"observed {sorted(by_run)}"
            )
        observed_bolds[session] = by_run
        mode = next(item.mode for item in plan.sessions if item.session == session)
        expected_id = f"sub-{plan.subject_label}_ses-{session}_fmap0"
        expected_intended = {
            "bids::" + str(path.relative_to(bids_root)).replace("\\", "/") for path in hits
        }
        for bold in hits:
            sidecar = bold.with_name(bold.name.replace(".nii.gz", ".json"))
            events = bold.with_name(bold.name.replace("_bold.nii.gz", "_events.tsv"))
            if not sidecar.is_file() or not events.is_file():
                raise FileNotFoundError(f"Missing BOLD sidecar/events for {bold}")
            metadata = read_json(sidecar)
            if metadata.get("PhaseEncodingDirection") not in {"i", "i-", "j", "j-", "k", "k-"}:
                raise RuntimeError(f"Invalid PhaseEncodingDirection in {sidecar}")
            if "TotalReadoutTime" not in metadata and "EffectiveEchoSpacing" not in metadata:
                raise RuntimeError(f"Missing readout metadata in {sidecar}")
            sources = _normalize_values(metadata.get("B0FieldSource"))
            if mode == "syn" and sources:
                raise RuntimeError(f"SyN BOLD references a measured fieldmap: {sidecar}: {sources}")
            if mode in MEASURED_MODES and sources != {expected_id}:
                raise RuntimeError(
                    f"Measured BOLD has wrong B0FieldSource in {sidecar}: {sources}; expected {expected_id}"
                )

        fmap_dir = subject_root / f"ses-{session}" / "fmap"
        fmap_jsons = sorted(fmap_dir.glob("*.json")) if fmap_dir.is_dir() else []
        if mode == "syn":
            if fmap_jsons or (fmap_dir.exists() and iter_files(fmap_dir)):
                raise RuntimeError(f"SyN session contains fieldmaps: {fmap_dir}")
            associations[session] = {"mode": mode, "fieldmaps": 0}
            continue
        expected_count = 2 if mode == "pepolar" else 3
        if len(fmap_jsons) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} {mode} fieldmap sidecars for ses-{session}; found {fmap_jsons}"
            )
        directions: list[str] = []
        roles: set[str] = set()
        for sidecar in fmap_jsons:
            metadata = read_json(sidecar)
            identifiers = _normalize_values(metadata.get("B0FieldIdentifier"))
            intended = _normalize_values(metadata.get("IntendedFor"))
            if identifiers != {expected_id}:
                raise RuntimeError(f"Wrong B0FieldIdentifier in {sidecar}: {identifiers}")
            if intended != expected_intended:
                raise RuntimeError(
                    f"IntendedFor mismatch in {sidecar}; expected same-session BOLDs only"
                )
            previous = all_field_ids.setdefault(expected_id, session)
            if previous != session:
                raise RuntimeError(f"B0FieldIdentifier reused across sessions: {expected_id}")
            if sidecar.name.endswith("_epi.json"):
                roles.add("epi")
                directions.append(str(metadata.get("PhaseEncodingDirection", "")))
            elif sidecar.name.endswith("_magnitude1.json"):
                roles.add("magnitude1")
            elif sidecar.name.endswith("_magnitude2.json"):
                roles.add("magnitude2")
            elif sidecar.name.endswith("_phasediff.json"):
                roles.add("phasediff")
                echo1 = float(metadata.get("EchoTime1", 0))
                echo2 = float(metadata.get("EchoTime2", 0))
                if not 0 < echo1 < echo2:
                    raise RuntimeError(f"Invalid phasediff echo times in {sidecar}: {echo1}, {echo2}")
        if mode == "pepolar":
            if roles != {"epi"} or len(directions) != 2 or not _opposite_phase_encoding(*directions):
                raise RuntimeError(f"Invalid PEPOLAR pair for ses-{session}: roles={roles}, PE={directions}")
        elif roles != {"magnitude1", "magnitude2", "phasediff"}:
            raise RuntimeError(f"Invalid GRE roles for ses-{session}: {sorted(roles)}")
        associations[session] = {
            "mode": mode,
            "fieldmaps": len(fmap_jsons),
            "b0_field_identifier": expected_id,
            "intended_for": sorted(expected_intended),
        }

    # Isolated branches may include other sessions' anatomy, but never their
    # functional or fieldmap data.
    unselected_func_or_fmap = []
    for session_dir in subject_root.glob("ses-*"):
        session = session_dir.name.removeprefix("ses-")
        if session in branch.sessions:
            continue
        for name in ("func", "fmap"):
            path = session_dir / name
            if path.exists() and iter_files(path):
                unselected_func_or_fmap.append(str(path))
    if unselected_func_or_fmap:
        raise RuntimeError(
            f"Cross-session functional/fieldmap content entered branch: {unselected_func_or_fmap}"
        )
    return {
        "status": "pass",
        "subject": plan.subject,
        "branch": branch.name,
        "mode": branch.mode,
        "sessions": list(branch.sessions),
        "runs": sorted(run for runs in observed_bolds.values() for run in runs),
        "associations": associations,
    }


def tree_manifest(root: Path, destination: Path) -> dict:
    lines = []
    for path in iter_files(root):
        rel = str(path.relative_to(root)).replace("\\", "/")
        lines.append(f"{sha256_file(path)}\t{path.stat().st_size}\t{rel}")
    text = "\n".join(lines) + "\n"
    destination.write_text(text, encoding="utf-8", newline="\n")
    return {
        "file_count": len(lines),
        "manifest": str(destination),
        "manifest_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "tree_sha256": canonical_json_hash(lines),
    }


def windows_to_wsl(path: Path) -> str:
    absolute = Path(os.path.abspath(path))
    drive = absolute.drive.rstrip(":").lower()
    if absolute.drive.startswith("\\\\"):
        # Imports entered through a resolved N: sys.path can spell this exact
        # project as its backing UNC share.  Re-map only paths proven to be
        # beneath the current N: target; arbitrary UNC mounts remain rejected.
        n_target = Path("N:\\").resolve(strict=False)
        try:
            relative = absolute.relative_to(n_target)
        except ValueError as error:
            raise ValueError(f"Cannot map non-N UNC path to WSL: {absolute}") from error
        tail = relative.as_posix().lstrip("/")
        return f"/mnt/n/{tail}"
    if len(drive) != 1 or not drive.isalpha():
        raise ValueError(f"Cannot map path to WSL: {absolute}")
    tail = absolute.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{tail}"


def docker_launcher(sdc: Mapping) -> list[str]:
    """Return the only approved Docker CLI route for N: bind mounts."""
    launcher = sdc["docker_launcher"]
    if launcher.get("mode") != "wsl":
        raise ValueError(f"Unsupported Docker launcher mode: {launcher.get('mode')!r}")
    return [
        str(launcher["windows_wsl_executable"]),
        "-d",
        str(launcher["distribution"]),
        "--",
        str(launcher["linux_docker_executable"]),
    ]


def docker_bind_mount(path: Path, destination: str, read_only: bool = False) -> str:
    """Build fail-closed ``--mount`` syntax from a Windows drive path.

    ``--mount`` is deliberate: unlike ``-v``, Docker refuses a missing bind
    source instead of silently creating a directory.  Persistent inputs and
    outputs are constrained to N: by their callers; the read-only FreeSurfer
    license is the sole C: bind.
    """
    source = windows_to_wsl(path)
    value = f"type=bind,src={source},dst={destination}"
    if read_only:
        value += ",readonly"
    return value


def validator_errors(payload: object) -> int:
    if not isinstance(payload, dict):
        raise ValueError("BIDS validator output is not an object")
    issues = payload.get("issues")
    if isinstance(issues, dict):
        issues = issues.get("issues")
    if not isinstance(issues, list):
        raise ValueError("BIDS validator output has no issues list")
    return sum(
        isinstance(item, dict) and str(item.get("severity", "")).lower() == "error"
        for item in issues
    )


def run_bids_validator(bids_root: Path, raw_output: Path, sdc: Mapping) -> dict:
    validator = sdc["validator"]
    version_command = [
        r"C:\Windows\System32\wsl.exe",
        "-d",
        str(validator["wsl_distribution"]),
        "--",
        str(validator["linux_binary"]),
        "--version",
    ]
    version_result = subprocess.run(
        version_command, check=False, text=True, capture_output=True
    )
    version_text = (version_result.stdout + "\n" + version_result.stderr).strip()
    expected_version = str(validator["expected_version"])
    if version_result.returncode != 0 or expected_version not in version_text:
        raise RuntimeError(
            f"BIDS validator version gate failed: expected {expected_version!r}, "
            f"returncode={version_result.returncode}, output={version_text!r}"
        )
    command = [
        r"C:\Windows\System32\wsl.exe",
        "-d",
        str(validator["wsl_distribution"]),
        "--",
        str(validator["linux_binary"]),
        "--format",
        "json_pp",
        "--outfile",
        windows_to_wsl(raw_output),
        windows_to_wsl(bids_root),
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"BIDS validator failed ({completed.returncode}): {completed.stdout}\n{completed.stderr}"
        )
    payload = read_json(raw_output)
    errors = validator_errors(payload)
    if errors:
        raise RuntimeError(f"BIDS validator reported {errors} errors for {bids_root}")
    return {
        "status": "pass",
        "version_command": version_command,
        "version_output": version_text,
        "expected_version": expected_version,
        "command": command,
        "report": str(raw_output),
        "report_sha256": sha256_file(raw_output),
        "error_count": 0,
    }


def docker_probe_command(
    bids_root: Path,
    plan: SubjectPlan,
    branch: BranchSpec,
    sdc: Mapping,
    container_name: str,
) -> list[str]:
    command = docker_launcher(sdc) + [
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--name",
        container_name,
        "--mount",
        docker_bind_mount(bids_root, "/data", read_only=True),
        "--mount",
        docker_bind_mount(DEFAULT_PROBE, "/probe.py", read_only=True),
    ]
    if branch.syn_filter:
        command += [
            "--mount",
            docker_bind_mount(DEFAULT_SYN_FILTER, "/bids-filter.json", read_only=True),
        ]
    command += [
        "--entrypoint",
        "python",
        str(sdc["fmriprep"]["image"]),
        "/probe.py",
        "--bids-root",
        "/data",
        "--subject",
        plan.subject_label,
        "--expected-mode",
        branch.mode,
    ]
    for session in branch.sessions:
        command += ["--session", session]
    if branch.syn_filter:
        command += ["--filter-file", "/bids-filter.json"]
    return command


def run_docker_probe(command: Sequence[str]) -> dict:
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"sdcflows estimator probe failed ({completed.returncode}):\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    marker = "SDC_PROBE_JSON="
    candidates = [line[len(marker):] for line in completed.stdout.splitlines() if line.startswith(marker)]
    if len(candidates) != 1:
        raise RuntimeError(f"Estimator probe emitted no unique JSON receipt: {completed.stdout}")
    payload = json.loads(candidates[0])
    if payload.get("status") != "pass":
        raise RuntimeError(f"Estimator probe did not pass: {payload}")
    return {
        "status": "pass",
        "command": list(command),
        "probe": payload,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def preflight_subject(
    plan: SubjectPlan,
    cohort_path: Path,
    sdc_path: Path,
    canonical_bids: Path,
    branch_root: Path,
    output: Path,
    link_mode: str,
    run_external: bool,
) -> dict:
    cohort, sdc = load_configs(cohort_path, sdc_path)
    del cohort
    canonical_bids = require_n_drive(canonical_bids, "canonical BIDS root", allow_missing=False)
    branch_root = require_n_drive(branch_root, "branch BIDS root")
    output = require_n_drive(output, "SDC preflight output")
    if output.exists():
        raise FileExistsError(f"Preflight output must be absent: {output}")
    stage = output.with_name(f".{output.name}.staging-{os.getpid()}")
    stage.mkdir(parents=True)
    records = []
    try:
        for branch in plan.branches:
            bids_root = expected_input_path(plan, branch, canonical_bids, branch_root)
            materialization = None
            if branch.input_kind == "isolated_session":
                if bids_root.exists():
                    raise FileExistsError(
                        f"Refusing a stale isolated branch; retire or audit it first: {bids_root}"
                    )
                materialization = build_isolated_branch(
                    canonical_bids, bids_root, plan, branch, link_mode
                )
            local = audit_bids_branch(bids_root, plan, branch)
            branch_stage = stage / branch.name
            branch_stage.mkdir()
            manifest = tree_manifest(bids_root, branch_stage / "input_tree.sha256.tsv")
            external = None
            if run_external:
                validator = run_bids_validator(
                    bids_root,
                    branch_stage / "bids_validator.json",
                    sdc,
                )
                probe_command = docker_probe_command(
                    bids_root,
                    plan,
                    branch,
                    sdc,
                    f"ai-iaps-sdc-probe-sub{plan.subject_label}-{branch.name}",
                )
                probe = run_docker_probe(probe_command)
                external = {"validator": validator, "sdcflows_probe": probe}
            record = {
                "branch": branch_payload(branch),
                "bids_root": str(bids_root),
                "materialization": materialization,
                "local_audit": local,
                "input_identity": manifest,
                "external": external,
            }
            (branch_stage / "receipt.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
            records.append(record)
        summary = {
            "status": "pass" if run_external else "dry_run_local_only",
            "generated_at": utc_now(),
            "plan": plan_payload(plan, cohort_path, sdc_path),
            "external_gates_executed": run_external,
            "branches": records,
        }
        (stage / "preflight_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        stage.replace(output)
        return summary
    except Exception:
        raise


def verify_preflight_receipt(
    receipt_root: Path,
    plan: SubjectPlan,
    cohort_path: Path,
    sdc_path: Path,
) -> dict:
    summary_path = receipt_root / "preflight_summary.json"
    summary = read_json(summary_path)
    if summary.get("status") != "pass" or not summary.get("external_gates_executed"):
        raise RuntimeError("Production requires a completed external BIDS + exact-image SDC preflight")
    current_plan = plan_payload(plan, cohort_path, sdc_path)
    for field in (
        "plan_sha256",
        "cohort_config_sha256",
        "sdc_config_sha256",
        "syn_filter_sha256",
        "probe_sha256",
        "workflow_sha256",
    ):
        if summary["plan"].get(field) != current_plan.get(field):
            raise RuntimeError(f"Preflight receipt is stale: {field} changed")
    for record in summary.get("branches", []):
        bids_root = Path(record["bids_root"])
        with tempfile.TemporaryDirectory(dir=receipt_root) as temp:
            temp_manifest = Path(temp) / "current.tsv"
            current = tree_manifest(bids_root, temp_manifest)
        if current["tree_sha256"] != record["input_identity"]["tree_sha256"]:
            raise RuntimeError(f"BIDS branch changed after preflight: {bids_root}")
    return summary


def verify_docker_image(image: str, expected_digest: str, sdc: Mapping) -> dict:
    command = docker_launcher(sdc) + [
        "image",
        "inspect",
        image,
        "--format",
        "{{json .RepoDigests}}",
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Pinned Docker image is unavailable: {completed.stderr}")
    repo_digests = json.loads(completed.stdout.strip())
    expected = image
    if expected not in repo_digests or not expected.endswith("@" + expected_digest):
        raise RuntimeError(f"Docker image digest mismatch: {repo_digests}; expected {expected}")
    return {"command": command, "repo_digests": repo_digests}


def c_storage_gate(minimum: int) -> dict:
    usage = shutil.disk_usage("C:\\")
    if usage.free < minimum:
        raise RuntimeError(
            f"C: has {usage.free} free bytes; production gate requires at least {minimum}"
        )
    return {"path": "C:\\", "free_bytes": usage.free, "minimum_free_bytes": minimum}


def fmriprep_command(
    bids_root: Path,
    output: Path,
    license_file: Path,
    work_volume: str,
    plan: SubjectPlan,
    branch: BranchSpec,
    sdc: Mapping,
) -> list[str]:
    settings = sdc["fmriprep"]
    command = docker_launcher(sdc) + [
        "run",
        "--rm",
        "--pull",
        "never",
        "--name",
        f"ai-iaps-fmriprep-sub{plan.subject_label}-{branch.name}",
        "--mount",
        docker_bind_mount(bids_root, "/data", read_only=True),
        "--mount",
        docker_bind_mount(output, "/out"),
        "--mount",
        docker_bind_mount(license_file, "/opt/freesurfer/license.txt", read_only=True),
        "--mount",
        f"type=volume,src={work_volume},dst=/work",
    ]
    if branch.syn_filter:
        command += [
            "--mount",
            docker_bind_mount(DEFAULT_SYN_FILTER, "/bids-filter.json", read_only=True),
        ]
    command += [
        str(settings["image"]),
        "/data",
        "/out",
        "participant",
        "--participant-label",
        plan.subject_label,
        "--fs-license-file",
        "/opt/freesurfer/license.txt",
        "--fs-no-reconall",
        "--use-syn-sdc",
        "error" if branch.mode == "syn" else "warn",
        "--output-spaces",
        f"{settings['output_space']}:res-{settings['output_resolution']}",
        "--nthreads",
        str(settings["nthreads"]),
        "--omp-nthreads",
        str(settings["omp_nthreads"]),
        "--mem-mb",
        str(settings["memory_mb"]),
        "--low-mem",
        "--stop-on-first-crash",
        "--skip-bids-validation",
        "-w",
        "/work",
    ]
    if branch.syn_filter:
        command += ["--bids-filter-file", "/bids-filter.json"]
    return command


def sdc_method(root: Path, subject_label: str, session: str, run: int) -> str:
    report = root / f"sub-{subject_label}" / "figures" / (
        f"sub-{subject_label}_ses-{session}_task-iaps_run-{run:02d}_desc-summary_bold.html"
    )
    if not report.is_file():
        raise FileNotFoundError(f"Missing completed summary report: {report}")
    text = report.read_text(encoding="utf-8", errors="replace")
    match = SUMMARY_RE.search(text)
    if not match:
        raise RuntimeError(f"No SDC method in {report}")
    return match.group(1).strip()


def method_matches(mode: str, observed: str) -> bool:
    value = observed.lower()
    if mode == "pepolar":
        return "pepolar" in value and "syn" not in value
    if mode == "syn":
        return ("syn" in value or "fieldmap-less" in value or "flb" in value) and "none" not in value
    if mode == "gre":
        return ("fieldmap" in value or "fmb" in value) and all(
            marker not in value for marker in ("fieldmap-less", "syn", "pepolar", "none")
        )
    return False


def basic_derivative_gate(root: Path, plan: SubjectPlan, branch: BranchSpec) -> dict:
    if not (root / "dataset_description.json").is_file():
        raise FileNotFoundError(root / "dataset_description.json")
    if not (root / f"sub-{plan.subject_label}.html").is_file():
        raise FileNotFoundError(root / f"sub-{plan.subject_label}.html")
    required = (
        "_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
        "_space-MNI152NLin6Asym_res-2_desc-preproc_bold.json",
        "_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz",
        "_desc-confounds_timeseries.tsv",
    )
    session_records = []
    for session in branch.sessions:
        expected_runs = next(item.runs for item in plan.sessions if item.session == session)
        mode = next(item.mode for item in plan.sessions if item.session == session)
        func = root / f"sub-{plan.subject_label}" / f"ses-{session}" / "func"
        for suffix in required:
            hits = sorted(func.glob(f"sub-{plan.subject_label}_ses-{session}_task-iaps_run-*{suffix}"))
            observed = {run_number(path) for path in hits}
            if len(hits) != len(observed) or observed != set(expected_runs):
                raise RuntimeError(
                    f"Derivative run mismatch for ses-{session} suffix {suffix}: {sorted(observed)}"
                )
        methods = {run: sdc_method(root, plan.subject_label, session, run) for run in expected_runs}
        wrong = {run: value for run, value in methods.items() if not method_matches(mode, value)}
        if wrong:
            raise RuntimeError(f"Wrong SDC method for Sub{plan.subject} ses-{session}: {wrong}")
        session_records.append({"session": session, "mode": mode, "methods": methods})
    return {"status": "pass", "sessions": session_records}


def _copy_tree_selected(source: Path, destination: Path, relative_files: Iterable[Path], mode: str) -> None:
    for rel in relative_files:
        copy_or_link(source / rel, destination / rel, mode)


def merge_mixed_branches(
    plan: SubjectPlan,
    branch_outputs: Mapping[str, Path],
    destination: Path,
    link_mode: str,
) -> dict:
    if plan.strategy != "isolated_mixed_session_branches":
        raise ValueError("Merge is only valid for mixed-session plans")
    destination = require_n_drive(destination, "final fMRIPrep derivative")
    if destination.exists():
        raise FileExistsError(destination)
    for branch in plan.branches:
        basic_derivative_gate(branch_outputs[branch.name], plan, branch)
    stage = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    preferred = next(
        (branch for branch in plan.branches if branch.mode in MEASURED_MODES), plan.branches[0]
    )
    preferred_root = branch_outputs[preferred.name]
    try:
        for name in ("dataset_description.json", ".bidsignore", "README", "CHANGES"):
            src = preferred_root / name
            if src.is_file():
                copy_or_link(src, stage / name, link_mode)
        subject_rel = Path(f"sub-{plan.subject_label}")
        preferred_subject = preferred_root / subject_rel
        # Subject-level anatomical outputs must be byte-identical because every
        # branch receives the exact same complete T1w set and pinned runtime.
        preferred_anat = preferred_subject / "anat"
        for other in plan.branches:
            other_anat = branch_outputs[other.name] / subject_rel / "anat"
            preferred_map = {
                str(path.relative_to(preferred_anat)): sha256_file(path)
                for path in iter_files(preferred_anat)
            }
            other_map = {
                str(path.relative_to(other_anat)): sha256_file(path)
                for path in iter_files(other_anat)
            }
            if other_map != preferred_map:
                raise RuntimeError(
                    f"Branches produced different subject anatomical references: "
                    f"{preferred.name} versus {other.name}; do not merge"
                )
        _copy_tree_selected(
            preferred_root,
            stage,
            [path.relative_to(preferred_root) for path in iter_files(preferred_anat)],
            link_mode,
        )
        # Copy each branch's exact session derivatives and session-tagged reportlets.
        for branch in plan.branches:
            source = branch_outputs[branch.name]
            for session in branch.sessions:
                session_root = source / subject_rel / f"ses-{session}"
                _copy_tree_selected(
                    source,
                    stage,
                    [path.relative_to(source) for path in iter_files(session_root)],
                    link_mode,
                )
                figures = source / subject_rel / "figures"
                selected_figures = [
                    path.relative_to(source)
                    for path in iter_files(figures)
                    if f"_ses-{session}_" in path.name
                ]
                _copy_tree_selected(source, stage, selected_figures, link_mode)
            report_bundle = stage / "logs" / "branch_reports" / branch.name
            for relroot in (
                Path(f"sub-{plan.subject_label}.html"),
                subject_rel / "figures",
                subject_rel / "log",
                Path("logs"),
            ):
                src = source / relroot
                if src.is_file():
                    copy_or_link(src, report_bundle / relroot, link_mode)
                elif src.is_dir():
                    for item in iter_files(src):
                        copy_or_link(item, report_bundle / item.relative_to(source), link_mode)
        report_links = "\n".join(
            f'<li><a href="logs/branch_reports/{branch.name}/sub-{plan.subject_label}.html">'
            f'{branch.name}: sessions {", ".join(branch.sessions)} ({branch.mode})</a></li>'
            for branch in plan.branches
        )
        (stage / f"sub-{plan.subject_label}.html").write_text(
            "<!doctype html><html><head><meta charset=\"utf-8\"><title>"
            f"Sub-{plan.subject_label} merged fMRIPrep reports</title></head><body>"
            "<h1>Session-specific fMRIPrep branch reports</h1>"
            "<p>This subject required isolated branches to prevent cross-session "
            "fieldmap borrowing. Review every branch report below.</p><ul>"
            f"{report_links}</ul></body></html>\n",
            encoding="utf-8",
        )
        provenance = {
            "generated_at": utc_now(),
            "subject": plan.subject,
            "strategy": plan.strategy,
            "all_subject_anatomicals_in_every_branch": True,
            "session_sources": {
                session: {
                    "branch": branch.name,
                    "mode": branch.mode,
                    "root": str(branch_outputs[branch.name]),
                }
                for branch in plan.branches
                for session in branch.sessions
            },
            "branch_report_bundles": {
                branch.name: f"logs/branch_reports/{branch.name}/sub-{plan.subject_label}.html"
                for branch in plan.branches
            },
        }
        provenance_path = stage / "logs" / "merge_provenance.json"
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        # Validate each session separately because the synthetic branch mode is mixed.
        for item in plan.sessions:
            one = BranchSpec("merged", item.mode, (item.session,), item.runs, "merged", item.mode == "syn")
            basic_derivative_gate(stage, plan, one)
        stage.replace(destination)
        return provenance
    except Exception:
        raise


def run_subject(
    plan: SubjectPlan,
    cohort_path: Path,
    sdc_path: Path,
    preflight_root: Path,
    output_root: Path,
    final_root: Path,
    license_file: Path,
    execute: bool,
    link_mode: str,
) -> dict:
    _, sdc = load_configs(cohort_path, sdc_path)
    output_root = require_n_drive(output_root, "fMRIPrep branch output root")
    final_root = require_n_drive(final_root, "final fMRIPrep output")
    if not license_file.is_file():
        raise FileNotFoundError(license_file)
    summary = verify_preflight_receipt(preflight_root, plan, cohort_path, sdc_path)
    storage = c_storage_gate(int(sdc["storage"]["minimum_c_free_bytes_before_launch"]))
    image = verify_docker_image(
        str(sdc["fmriprep"]["image"]), str(sdc["fmriprep"]["image_digest"]), sdc
    )
    if final_root.exists():
        raise FileExistsError(f"Final derivative already exists: {final_root}")
    commands = []
    branch_outputs: dict[str, Path] = {}
    for record, branch in zip(summary["branches"], plan.branches, strict=True):
        if record["branch"] != branch_payload(branch):
            raise RuntimeError(f"Preflight branch order/content mismatch for {branch.name}")
        bids_root = Path(record["bids_root"])
        output = expected_output_path(branch, output_root)
        if output.exists():
            raise FileExistsError(f"Branch output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        work_volume = f"ai_iaps_full28_fmriprep_work_sub{plan.subject_label}_{branch.name}"
        command = fmriprep_command(
            bids_root, output, license_file.resolve(), work_volume, plan, branch, sdc
        )
        commands.append({"branch": branch.name, "argv": command, "work_volume": work_volume})
        branch_outputs[branch.name] = output
        if execute:
            launcher = docker_launcher(sdc)
            exists = subprocess.run(
                launcher + ["volume", "inspect", work_volume],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0
            if exists:
                raise RuntimeError(f"Refusing stale Docker work volume: {work_volume}")
            subprocess.run(launcher + ["volume", "create", work_volume], check=True)
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"fMRIPrep branch {branch.name} failed with code {completed.returncode}; "
                    f"work volume retained as {work_volume}"
                )
            basic_derivative_gate(output, plan, branch)
    if not execute:
        # Dry-run output directories are control artifacts only; remove the
        # empty directories we just created so no one can mistake them for runs.
        for output in branch_outputs.values():
            output.rmdir()
        if output_root.exists() and not any(output_root.iterdir()):
            output_root.rmdir()
        return {"status": "dry_run", "storage": storage, "image": image, "commands": commands}

    if plan.strategy == "isolated_mixed_session_branches":
        merge = merge_mixed_branches(plan, branch_outputs, final_root, link_mode)
    else:
        only = plan.branches[0]
        basic_derivative_gate(branch_outputs[only.name], plan, only)
        branch_outputs[only.name].replace(final_root)
        merge = {"strategy": "atomic_rename_single_branch", "source_branch": only.name}
    receipt = {
        "status": "pass",
        "generated_at": utc_now(),
        "plan": plan_payload(plan, cohort_path, sdc_path),
        "storage_gate": storage,
        "docker_image_gate": image,
        "commands": commands,
        "freesurfer_license": {
            "path": str(license_file.resolve()),
            "size_bytes": license_file.stat().st_size,
            "sha256": sha256_file(license_file),
        },
        "final_root": str(final_root),
        "publication": merge,
        "work_volumes_retained_pending_archive_and_visual_qc": [
            item["work_volume"] for item in commands
        ],
    }
    receipt_path = final_root / "logs" / "production_launch_receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def all_plans_payload(cohort: Mapping, cohort_path: Path, sdc_path: Path) -> dict:
    subjects = [int(item) for item in cohort["cohort"]["remaining_subjects"]]
    plans = [plan_payload(subject_plan(cohort, subject), cohort_path, sdc_path) for subject in subjects]
    return {
        "generated_at": utc_now(),
        "remaining_subject_count": len(plans),
        "subjects": plans,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--cohort-config", type=Path, default=DEFAULT_COHORT_CONFIG)
    root.add_argument("--sdc-config", type=Path, default=DEFAULT_SDC_CONFIG)
    commands = root.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--subject", type=int)
    plan.add_argument("--output", type=Path)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--subject", type=int, required=True)
    preflight.add_argument("--canonical-bids-root", type=Path, required=True)
    preflight.add_argument("--branch-root", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--link-mode", choices=("hardlink", "copy", "auto"), default="auto")
    preflight.add_argument(
        "--run-external-gates",
        action="store_true",
        help="Run official BIDS validator and pinned-container sdcflows discovery",
    )

    run = commands.add_parser("run-subject")
    run.add_argument("--subject", type=int, required=True)
    run.add_argument("--preflight-root", type=Path, required=True)
    run.add_argument("--branch-output-root", type=Path, required=True)
    run.add_argument("--final-root", type=Path, required=True)
    run.add_argument("--license-file", type=Path, required=True)
    run.add_argument("--link-mode", choices=("hardlink", "copy", "auto"), default="auto")
    run.add_argument("--execute", action="store_true")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        cohort, _ = load_configs(args.cohort_config, args.sdc_config)
        if args.command == "plan":
            payload = (
                plan_payload(subject_plan(cohort, args.subject), args.cohort_config, args.sdc_config)
                if args.subject is not None
                else all_plans_payload(cohort, args.cohort_config, args.sdc_config)
            )
            text = json.dumps(payload, indent=2) + "\n"
            if args.output:
                if args.output.exists():
                    raise FileExistsError(args.output)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text, encoding="utf-8")
            else:
                print(text, end="")
            return 0
        plan = subject_plan(cohort, args.subject)
        if args.command == "preflight":
            result = preflight_subject(
                plan,
                args.cohort_config,
                args.sdc_config,
                args.canonical_bids_root,
                args.branch_root,
                args.output,
                args.link_mode,
                args.run_external_gates,
            )
            print(f"SDC_PREFLIGHT_{result['status'].upper()}: {args.output}")
            return 0 if result["status"] == "pass" else 3
        result = run_subject(
            plan,
            args.cohort_config,
            args.sdc_config,
            args.preflight_root,
            args.branch_output_root,
            args.final_root,
            args.license_file,
            args.execute,
            args.link_mode,
        )
        print(json.dumps(result, indent=2))
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
