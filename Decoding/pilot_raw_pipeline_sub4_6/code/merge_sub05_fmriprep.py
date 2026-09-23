#!/usr/bin/env python3
"""Assemble Sub-05's session-specific fMRIPrep derivatives.

The all-session branch supplies measured reverse-PE correction for March
(ses-02).  The isolated February branch supplies fieldmap-less SyN correction
for ses-01.  This script makes a new derivative root and refuses to mix in the
uncorrected February outputs from the all-session branch.  It also requires,
verifies, and archives the exact BIDS filter that enabled February SyN
estimator discovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


def copy_or_link(src: Path, dst: Path, hardlink: bool) -> None:
    if dst.exists():
        raise FileExistsError(f"Refusing to replace an existing destination file: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def iter_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


EXPECTED_RUNS = {
    "ses-01": {1, 3, 4, 5},
    "ses-02": {2, 6, 7, 8, 9, 10},
}

REQUIRED_RUN_SUFFIXES = (
    "_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
    "_space-MNI152NLin6Asym_res-2_desc-preproc_bold.json",
    "_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz",
    "_desc-confounds_timeseries.tsv",
)

FEB_BIDS_FILTER_TEMPLATE = Path(__file__).with_name("sub05_feb_syn_bids_filter.json")
FEB_BIDS_FILTER_ARCHIVE = Path("logs/merge_branches/february_syn/bids_filter.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_feb_bids_filter(path: Path) -> tuple[Path, str]:
    """Require the exact pinned filter and return its resolved path and SHA-256."""
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"February BIDS filter is missing: {source}")
    if not FEB_BIDS_FILTER_TEMPLATE.is_file():
        raise FileNotFoundError(
            f"Pinned February BIDS filter template is missing: {FEB_BIDS_FILTER_TEMPLATE}"
        )
    expected_sha256 = sha256_file(FEB_BIDS_FILTER_TEMPLATE)
    observed_sha256 = sha256_file(source)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            "February BIDS filter does not match the pinned analysis-control file: "
            f"{source} has SHA-256 {observed_sha256}, expected {expected_sha256}"
        )
    return source, observed_sha256


def run_number(path: Path) -> int:
    match = re.search(r"_run-(\d+)_", path.name)
    if not match:
        raise RuntimeError(f"Cannot parse run number from {path}")
    return int(match.group(1))


def session_runs(root: Path, suffix: str) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for session in EXPECTED_RUNS:
        func = root / "sub-05" / session / "func"
        hits = sorted(func.glob(f"sub-05_{session}_task-iaps_run-*{suffix}"))
        result[session] = {run_number(path) for path in hits}
        if len(hits) != len(result[session]):
            raise RuntimeError(f"Duplicate run files for {session}, suffix {suffix}: {hits}")
    return result


def require_runs(root: Path, expected: dict[str, set[int]], label: str) -> None:
    for suffix in REQUIRED_RUN_SUFFIXES:
        observed = session_runs(root, suffix)
        if observed != expected:
            raise RuntimeError(
                f"{label} has the wrong session/run set for {suffix}: {observed}; "
                f"expected {expected}"
            )


def require_sdc_marker(root: Path, session: str, runs: set[int], marker: str) -> None:
    figures = root / "sub-05" / "figures"
    for run in sorted(runs):
        report = figures / (
            f"sub-05_{session}_task-iaps_run-{run:02d}_desc-summary_bold.html"
        )
        if not report.is_file():
            raise FileNotFoundError(f"Missing completed fMRIPrep summary report: {report}")
        text = report.read_text(encoding="utf-8", errors="replace")
        match = re.search(
            r"susceptibility\s+distortion\s+correction\s*:\s*([^<\r\n]+)",
            text,
            flags=re.IGNORECASE,
        )
        method = match.group(1).strip().lower() if match else ""
        expected = marker.lower()
        opposite = {"pepolar": "syn", "syn": "pepolar"}.get(expected)
        if expected not in method or (opposite is not None and opposite in method):
            raise RuntimeError(
                f"Expected exclusive SDC marker {marker!r} on the summary method line in "
                f"{report}; found {method!r}. Inspect the branch before merging"
            )


def copy_report_bundle(source: Path, destination: Path, hardlink: bool) -> None:
    """Preserve a report and its relative assets without mixing branch figures."""
    report = source / "sub-05.html"
    if not report.is_file():
        raise FileNotFoundError(f"Completed fMRIPrep HTML report is missing: {report}")
    copy_or_link(report, destination / "sub-05.html", hardlink)
    for relative_root in (Path("sub-05") / "figures", Path("sub-05") / "log", Path("logs")):
        tree = source / relative_root
        if not tree.exists():
            continue
        for src in iter_files(tree):
            copy_or_link(src, destination / src.relative_to(source), hardlink)


def assemble(
    all_sessions: Path,
    feb_syn: Path,
    destination: Path,
    hardlink: bool,
    *,
    feb_bids_filter: Path,
) -> None:
    all_sessions = all_sessions.resolve()
    feb_syn = feb_syn.resolve()
    destination = destination.resolve()
    feb_bids_filter, feb_bids_filter_sha256 = validate_feb_bids_filter(feb_bids_filter)
    for root in (all_sessions, feb_syn):
        if not root.is_dir():
            raise FileNotFoundError(root)
    if all_sessions == feb_syn:
        raise ValueError("The all-session and February-SyN roots must be different")
    if all_sessions in feb_syn.parents or feb_syn in all_sessions.parents:
        raise ValueError("The two input derivative roots must not contain one another")
    if any(destination == root or root in destination.parents for root in (all_sessions, feb_syn)):
        raise ValueError("Destination must not be an input root or a child of an input root")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Destination must be absent or empty: {destination}")

    if not (all_sessions / "dataset_description.json").is_file():
        raise FileNotFoundError(all_sessions / "dataset_description.json")
    require_runs(all_sessions, EXPECTED_RUNS, "All-session branch")
    feb_expected = {"ses-01": EXPECTED_RUNS["ses-01"], "ses-02": set()}
    require_runs(feb_syn, feb_expected, "February SyN branch")
    require_sdc_marker(all_sessions, "ses-02", EXPECTED_RUNS["ses-02"], "pepolar")
    require_sdc_marker(feb_syn, "ses-01", EXPECTED_RUNS["ses-01"], "syn")

    destination.mkdir(parents=True, exist_ok=True)

    # A derivative root only needs the derivative metadata. Do not copy other
    # subjects (or their top-level reports) if all_sessions is a shared root.
    for name in ("dataset_description.json", ".bidsignore", "README", "CHANGES"):
        src = all_sessions / name
        if src.is_file():
            copy_or_link(src, destination / name, hardlink)

    # Copy only Sub-05 from the all-session branch, excluding its known no-SDC
    # February derivatives and reportlets.
    all_subject = all_sessions / "sub-05"
    for src in iter_files(all_subject):
        rel_subject = src.relative_to(all_subject)
        if rel_subject.parts[0] in {"ses-01", "log"}:
            continue
        if rel_subject.parts[0] == "figures" and "_ses-01_" in src.name:
            continue
        copy_or_link(src, destination / "sub-05" / rel_subject, hardlink)

    # Bring in only the February functional derivatives and their SyN
    # reportlets. Anatomical derivatives remain from the all-session branch;
    # both branches use the identical subject-level T1w input and pinned image.
    feb_subject = feb_syn / "sub-05"
    february_files: list[Path] = []
    for src in iter_files(feb_subject):
        rel_subject = src.relative_to(feb_subject)
        if rel_subject.parts[0] == "ses-01" or (
            rel_subject.parts[0] == "figures" and "_ses-01_" in src.name
        ):
            february_files.append(src)
    if not february_files:
        raise RuntimeError("February SyN branch contains no sub-05/ses-01 files")
    for src in february_files:
        rel_subject = src.relative_to(feb_subject)
        copy_or_link(src, destination / "sub-05" / rel_subject, hardlink)

    # A renamed HTML report can silently point at the other branch's figures.
    # Preserve each complete report bundle in its own relative directory.
    report_root = destination / "logs" / "merge_branches"
    copy_report_bundle(all_sessions, report_root / "all_sessions", hardlink)
    copy_report_bundle(feb_syn, report_root / "february_syn", hardlink)

    # Preserve the small analysis-control file as an independent copy rather
    # than a hard link so later cleanup or mutation of the local input cannot
    # change the archived provenance artifact.
    archived_filter = destination / FEB_BIDS_FILTER_ARCHIVE
    copy_or_link(feb_bids_filter, archived_filter, hardlink=False)
    archived_filter_sha256 = sha256_file(archived_filter)
    source_sha256_after_copy = sha256_file(feb_bids_filter)
    if not (
        archived_filter_sha256
        == source_sha256_after_copy
        == feb_bids_filter_sha256
    ):
        raise RuntimeError(
            "February BIDS filter changed while it was being archived or its archived "
            "copy failed SHA-256 verification"
        )

    require_runs(destination, EXPECTED_RUNS, "Merged derivative")
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "all_sessions_source": str(all_sessions),
        "february_syn_source": str(feb_syn),
        "session_assignment": {
            "ses-01": "February fieldmap-less SyN; runs 01,03,04,05",
            "ses-02": "March measured reverse-PE PEPOLAR; runs 02,06,07,08,09,10",
        },
        "report_bundles": {
            "all_sessions": "logs/merge_branches/all_sessions/sub-05.html",
            "february_syn": "logs/merge_branches/february_syn/sub-05.html",
        },
        "february_bids_filter": {
            "source_path": str(feb_bids_filter),
            "archived_relative_path": FEB_BIDS_FILTER_ARCHIVE.as_posix(),
            "sha256": feb_bids_filter_sha256,
        },
    }
    (destination / "logs" / "merge_sub05_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"Assembled {destination}: 4 February SyN runs + 6 March PEPOLAR runs; "
        "10 preprocessed BOLD files"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("all_sessions", type=Path)
    parser.add_argument("feb_syn", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--feb-bids-filter",
        type=Path,
        required=True,
        help="Exact Sub-05 February fMRIPrep BIDS filter to verify and archive",
    )
    parser.add_argument("--hardlink", action="store_true")
    args = parser.parse_args()
    assemble(
        args.all_sessions,
        args.feb_syn,
        args.destination,
        args.hardlink,
        feb_bids_filter=args.feb_bids_filter,
    )


if __name__ == "__main__":
    main()
