#!/usr/bin/env python3
"""Add full NIfTI SHA-256 values to a reviewed exact-path selection draft.

This script never changes approval_status. It checkpoints progress and never
modifies raw data. Production promotion remains a separate human audit action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cohortlib import (
    SELECTION_COLUMNS,
    load_config,
    read_selection,
    safe_relative_path,
    sha256_file,
    sidecar_for_nifti,
    verify_file_identity,
    verify_stimuli_identity,
    write_tsv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "cohort.json"
DEFAULT_INPUT = ROOT / "audit" / "source_inventory" / "review_draft_selection.tsv"
DEFAULT_OUTPUT = ROOT / "audit" / "source_inventory" / "hashed_review_draft_selection.tsv"


def frozen_path(root: Path, relpath: str, label: str) -> Path:
    path, problem = safe_relative_path(root, relpath)
    if problem or path is None:
        raise RuntimeError(f"Unsafe exact {label} path {relpath!r}: {problem}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    raw_root = args.raw_root or Path(config["paths"]["raw_root"])
    log_root = Path(config["paths"]["log_root"])
    stimuli_csv = Path(config["paths"]["stimuli_csv"])
    verify_stimuli_identity(config, stimuli_csv)
    rows = read_selection(args.input)
    if any(row["approval_status"] == "approved" for row in rows):
        raise RuntimeError("Input must remain a review draft; refusing already-approved rows")
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite completed hash freeze: {args.output}")
    checkpoint = args.output.with_name(args.output.name + ".partial")
    if checkpoint.exists():
        if not args.resume:
            raise RuntimeError(f"Checkpoint exists; inspect it and use --resume: {checkpoint}")
        prior = {row["asset_id"]: row for row in read_selection(checkpoint)}
        if set(prior) != {row["asset_id"] for row in rows}:
            raise RuntimeError("Checkpoint asset IDs differ from current input")
        for row in rows:
            old = prior[row["asset_id"]]
            for field in SELECTION_COLUMNS:
                if field == "source_sha256":
                    continue
                if old[field] != row[field]:
                    raise RuntimeError(
                        f"Checkpoint field {field} changed for {row['asset_id']}: "
                        f"{old[field]!r} != {row[field]!r}"
                    )
            row["source_sha256"] = old["source_sha256"]
    elif args.resume:
        raise RuntimeError(f"--resume requested but no checkpoint exists: {checkpoint}")

    for row in rows:
        source = frozen_path(raw_root, row["source_relpath"], "NIfTI")
        sidecar = frozen_path(raw_root, row["source_json_relpath"], "source JSON")
        if sidecar.resolve() != sidecar_for_nifti(source).resolve():
            raise RuntimeError(f"Frozen JSON is not the NIfTI sidecar for {row['asset_id']}")
        verify_file_identity(
            sidecar,
            row["source_json_size_bytes"],
            row["source_json_sha256"],
            label=f"source JSON for {row['asset_id']}",
        )
        if row["asset_type"] == "bold":
            onset = frozen_path(log_root, row["onset_mat_relpath"], "onset MAT")
            verify_file_identity(
                onset,
                row["onset_mat_size_bytes"],
                row["onset_mat_sha256"],
                label=f"onset MAT for {row['asset_id']}",
            )

    total = len(rows)
    for index, row in enumerate(rows, start=1):
        source = frozen_path(raw_root, row["source_relpath"], "NIfTI")
        if row["source_sha256"]:
            verify_file_identity(
                source,
                row["source_size_bytes"],
                row["source_sha256"],
                label=f"previously frozen NIfTI for {row['asset_id']}",
            )
            continue
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != int(row["source_size_bytes"]):
            raise RuntimeError(f"Source size changed for {row['asset_id']}")
        print(f"HASH {index}/{total} {row['asset_id']} {source}", flush=True)
        row["source_sha256"] = sha256_file(source)
        if index % args.checkpoint_every == 0:
            write_tsv(checkpoint, rows, SELECTION_COLUMNS)
    write_tsv(checkpoint, rows, SELECTION_COLUMNS)
    checkpoint.replace(args.output)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": total,
        "all_hashes_frozen": all(bool(row["source_sha256"]) for row in rows),
        "all_source_json_hashes_frozen": all(bool(row["source_json_sha256"]) for row in rows),
        "all_bold_onset_hashes_frozen": all(
            row["asset_type"] != "bold" or bool(row["onset_mat_sha256"]) for row in rows
        ),
        "stimuli_csv_sha256": config["input_identities"]["stimuli_600trials_csv"]["sha256"],
        "approved_rows": sum(row["approval_status"] == "approved" for row in rows),
        "note": "Hashes are frozen, but rows remain unapproved until separate audit promotion.",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
