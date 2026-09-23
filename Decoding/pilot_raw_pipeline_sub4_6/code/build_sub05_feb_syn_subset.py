#!/usr/bin/env python3
"""Build the isolated Sub-05 February BIDS input used for SyN SDC.

Sub-05's February acquisition (ses-01; runs 01, 03, 04, and 05) has no
same-session reverse-phase EPI.  The March reverse-phase pair belongs only to
ses-02.  fMRIPrep sees the March fieldmap at subject scope, so ``--use-syn-sdc
warn`` alone does not activate fieldmap-less SyN for February.  This script
creates a minimal, valid BIDS subset containing the March T1w and February
functional data but no fieldmaps or March BOLD data.

The anatomical image intentionally remains sessionless.  fMRIPrep 25.1.3's
sdcflows discovery otherwise narrows the anatomical query to the functional
session and misses this valid subject-level T1w.  Run this subset with
``sub05_feb_syn_bids_filter.json`` so the ``fmap`` discovery query uses
PyBIDS ``Query.OPTIONAL`` for the session entity.  This preserves the original
BIDS labels while allowing sessionless anatomy to be paired with ses-01 BOLD.

On one NTFS volume, ``--hardlink`` avoids another ~1 GB copy.  The destination
is required to be absent or empty so a stale or mixed input cannot be reused.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path


REQUIRED_ROOT_FILES = (
    "dataset_description.json",
    "participants.tsv",
    "task-iaps_events.json",
)
OPTIONAL_ROOT_FILES = (
    "README",
)
EXPECTED_RUNS = (1, 3, 4, 5)
FILTER_TEMPLATE = Path(__file__).with_name("sub05_feb_syn_bids_filter.json")


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


def require_one(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {description}; found {paths}")
    return paths[0]


def planned_files(source: Path) -> list[Path]:
    """Validate the full input plan before creating the destination."""
    selected: list[Path] = []
    for name in REQUIRED_ROOT_FILES:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"Required BIDS root file is missing: {path}")
        # participants.tsv must be regenerated with only sub-05. Hard-linking
        # the full pilot table would make this single-subject subset invalid,
        # and rewriting a hard link would corrupt the canonical source table.
        if name != "participants.tsv":
            selected.append(path)
    for name in OPTIONAL_ROOT_FILES:
        path = source / name
        if path.is_file():
            selected.append(path)

    participants = (source / "participants.tsv").read_text(encoding="utf-8")
    if "sub-05" not in participants.split():
        raise RuntimeError("participants.tsv does not contain sub-05")

    anat = source / "sub-05" / "anat"
    selected.append(require_one(list(anat.glob("sub-05_T1w.nii.gz")), "Sub-05 T1w NIfTI"))
    selected.append(require_one(list(anat.glob("sub-05_T1w.json")), "Sub-05 T1w sidecar"))
    session_anat = list((source / "sub-05").glob("ses-*/anat/*_T1w.nii.gz"))
    if session_anat:
        raise RuntimeError(
            "Sub-05 staging expects one truthful sessionless T1w; found "
            f"session-labelled alternatives: {session_anat}"
        )

    func = source / "sub-05" / "ses-01" / "func"
    observed_runs: set[int] = set()
    for path in func.glob("sub-05_ses-01_task-iaps_run-*_bold.nii.gz"):
        match = re.search(r"_run-(\d+)_", path.name)
        if not match:
            raise RuntimeError(f"Cannot parse run number from {path}")
        observed_runs.add(int(match.group(1)))
    if observed_runs != set(EXPECTED_RUNS):
        raise RuntimeError(
            f"Expected February runs {list(EXPECTED_RUNS)}; found {sorted(observed_runs)}"
        )

    for run in EXPECTED_RUNS:
        prefix = f"sub-05_ses-01_task-iaps_run-{run:02d}"
        bold = require_one(list(func.glob(f"{prefix}_bold.nii.gz")), f"run-{run:02d} BOLD")
        sidecar = require_one(list(func.glob(f"{prefix}_bold.json")), f"run-{run:02d} sidecar")
        events = require_one(list(func.glob(f"{prefix}_events.tsv")), f"run-{run:02d} events")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if "B0FieldSource" in metadata:
            raise RuntimeError(
                f"February sidecar unexpectedly references a measured fieldmap: {sidecar}"
            )
        selected.extend((bold, sidecar, events))

    if (source / "sub-05" / "ses-01" / "fmap").exists():
        raise RuntimeError("Sub-05 February unexpectedly contains a fieldmap directory")
    return selected


def write_subject_participants(source: Path, destination: Path) -> Path:
    source_path = source / "participants.tsv"
    with source_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames or "participant_id" not in reader.fieldnames:
            raise RuntimeError(f"participants.tsv lacks participant_id: {source_path}")
        rows = [row for row in reader if row.get("participant_id") == "sub-05"]
        fieldnames = list(reader.fieldnames)
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one sub-05 row in participants.tsv; found {len(rows)}"
        )

    destination_path = destination / "participants.tsv"
    with destination_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return destination_path


def build_subset(source: Path, destination: Path, hardlink: bool) -> list[Path]:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"BIDS source does not exist: {source}")
    if destination == source or source in destination.parents:
        raise ValueError("Destination must not be the source or a child of the source")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Destination must be absent or empty to prevent mixed inputs: {destination}"
        )
    selected = planned_files(source)
    filter_destination = destination.with_name(
        f"{destination.name}_fmriprep_bids_filter.json"
    )
    if not FILTER_TEMPLATE.is_file():
        raise FileNotFoundError(f"Pinned fMRIPrep BIDS filter is missing: {FILTER_TEMPLATE}")
    if filter_destination.exists():
        raise FileExistsError(
            f"Refusing to replace an existing fMRIPrep BIDS filter: {filter_destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = [write_subject_participants(source, destination)]
    for src in selected:
        rel = src.relative_to(source)
        dst = destination / rel
        copy_or_link(src, dst, hardlink)
        copied.append(dst)

    # Keep this analysis-control file outside the BIDS dataset so it does not
    # need a .bidsignore rule and cannot be mistaken for acquisition metadata.
    shutil.copy2(FILTER_TEMPLATE, filter_destination)

    forbidden = list(destination.rglob("*fmap*")) + list(destination.rglob("*ses-02*"))
    if forbidden:
        raise RuntimeError(f"Forbidden March/fieldmap content entered subset: {forbidden}")
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Full pilot BIDS root")
    parser.add_argument("destination", type=Path, help="New isolated BIDS root")
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="Try NTFS hard links first and fall back to copying",
    )
    args = parser.parse_args()
    copied = build_subset(args.source, args.destination, args.hardlink)
    n_bold = sum(path.name.endswith("_bold.nii.gz") for path in copied)
    print(f"Created {args.destination} with {len(copied)} files and {n_bold} BOLD runs")
    print(
        "fMRIPrep BIDS filter: "
        f"{args.destination.with_name(args.destination.name + '_fmriprep_bids_filter.json')}"
    )


if __name__ == "__main__":
    main()
