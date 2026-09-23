#!/usr/bin/env python3
"""Exact-image sdcflows estimator-discovery gate.

This file is mounted read-only into the pinned fMRIPrep container.  It is not
intended to run in the host Python environment.  The caller has already
performed a stricter BIDS-metadata association audit; this probe independently
requires fMRIPrep's bundled sdcflows to discover the expected estimator type.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bids import BIDSLayout
from bids.layout import Query
from sdcflows import fieldmaps as fm
from sdcflows.utils.wrangler import find_estimators


def _decode_query(value):
    if isinstance(value, list):
        return [_decode_query(item) for item in value]
    if isinstance(value, dict):
        return {key: _decode_query(item) for key, item in value.items()}
    if isinstance(value, str) and value.startswith("<Query.") and value.endswith(">"):
        return getattr(Query, value[7:-4])
    return value


def _method_name(estimator) -> str:
    return str(estimator.method).rsplit(".", 1)[-1].lower()


def _as_values(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    return {str(value)}


def _source_path(source) -> str:
    return str(getattr(source, "path", source))


def _source_metadata(source) -> dict:
    value = getattr(source, "metadata", {})
    return value if isinstance(value, dict) else {}


def probe(bids_root: Path, subject: str, expected_mode: str, sessions: list[str], filter_file: Path | None) -> dict:
    layout = BIDSLayout(str(bids_root), validate=False)
    bolds = layout.get(
        subject=subject,
        datatype="func",
        suffix="bold",
        extension=[".nii", ".nii.gz"],
        scope="raw",
    )
    expected_bolds = {str(item.path) for item in bolds}
    if not expected_bolds:
        raise RuntimeError("No BOLD inputs were discovered")

    filters = None
    if filter_file is not None:
        payload = json.loads(filter_file.read_text(encoding="utf-8"))
        filters = {
            key: _decode_query(value)
            for key, value in payload.get("fmap", {}).items()
        }

    fm.clear_registry()
    kwargs = {"layout": layout, "subject": subject}
    if filters is not None:
        kwargs["bids_filters"] = filters
    if expected_mode == "syn":
        kwargs.update({"fmapless": {"bold"}, "force_fmapless": True})
    estimators = find_estimators(**kwargs)
    records = []
    for estimator in estimators:
        identifiers = set()
        sources = []
        for source in estimator.sources:
            metadata = _source_metadata(source)
            identifiers.update(_as_values(metadata.get("B0FieldIdentifier")))
            sources.append(
                {
                    "path": _source_path(source),
                    "b0_field_identifier": sorted(_as_values(metadata.get("B0FieldIdentifier"))),
                    "intended_for": sorted(_as_values(metadata.get("IntendedFor"))),
                }
            )
        bids_id = getattr(estimator, "bids_id", None)
        if bids_id:
            identifiers.add(str(bids_id))
        records.append(
            {
                "method": _method_name(estimator),
                "identifiers": sorted(identifiers),
                "sources": sources,
            }
        )

    if not records:
        raise RuntimeError("sdcflows discovered no estimators")

    if expected_mode == "syn":
        if any(record["method"] != "anat" for record in records):
            raise RuntimeError(f"Expected only ANAT/SyN estimators; found {records}")
        covered = {
            source["path"]
            for record in records
            for source in record["sources"]
            if source["path"].endswith(("_bold.nii", "_bold.nii.gz"))
        }
        if covered != expected_bolds:
            raise RuntimeError(
                f"SyN estimator coverage mismatch; missing={sorted(expected_bolds - covered)}, "
                f"extra={sorted(covered - expected_bolds)}"
            )
    else:
        # The pinned fMRIPrep/sdcflows image represents a Siemens GRE
        # magnitude1+magnitude2+phasediff field map as ``phasediff``.  The
        # older ``mapped`` label denotes an already-computed field map and is
        # not the BIDS input used by this cohort.
        expected_method = {"pepolar": "pepolar", "gre": "phasediff"}[expected_mode]
        matching = [record for record in records if record["method"] == expected_method]
        if len(matching) != len(sessions):
            raise RuntimeError(
                f"Expected {len(sessions)} {expected_method} estimators, found {len(matching)}: {records}"
            )
        expected_ids = {f"sub-{subject}_ses-{session}_fmap0" for session in sessions}
        observed_ids = {
            identifier
            for record in matching
            for identifier in record["identifiers"]
        }
        if not expected_ids.issubset(observed_ids):
            raise RuntimeError(
                f"Measured estimator IDs mismatch; expected={sorted(expected_ids)}, "
                f"observed={sorted(observed_ids)}"
            )

    payload = {
        "status": "pass",
        "subject": subject,
        "expected_mode": expected_mode,
        "sessions": sessions,
        "bold_count": len(expected_bolds),
        "estimators": records,
    }
    print("SDC_PROBE_JSON=" + json.dumps(payload, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", type=Path, default=Path("/data"))
    parser.add_argument("--subject", required=True)
    parser.add_argument("--expected-mode", choices=("pepolar", "gre", "syn"), required=True)
    parser.add_argument("--session", action="append", required=True)
    parser.add_argument("--filter-file", type=Path)
    args = parser.parse_args()
    probe(args.bids_root, args.subject, args.expected_mode, args.session, args.filter_file)


if __name__ == "__main__":
    main()
