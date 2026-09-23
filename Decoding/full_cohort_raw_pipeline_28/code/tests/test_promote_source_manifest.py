from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from cohortlib import SELECTION_COLUMNS  # noqa: E402
from promote_source_manifest import (  # noqa: E402
    validate_draft_equivalence,
    validate_frozen_rows,
)


def row(asset_id: str, asset_type: str, subject: int, status: str) -> dict[str, str]:
    result = {column: "" for column in SELECTION_COLUMNS}
    result.update(
        {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "subject": str(subject),
            "approval_status": status,
            "source_sha256": "a" * 64,
            "source_json_sha256": "b" * 64,
        }
    )
    if asset_type == "bold":
        result["onset_mat_sha256"] = "c" * 64
    return result


class DraftEquivalenceTests(unittest.TestCase):
    def test_only_nifti_hash_may_change_after_review(self) -> None:
        reviewed = [row("a", "t1w", 1, "proposed_unique")]
        reviewed[0]["source_sha256"] = ""
        hashed = copy.deepcopy(reviewed)
        hashed[0]["source_sha256"] = "d" * 64
        validate_draft_equivalence(reviewed, hashed)

    def test_path_change_is_rejected(self) -> None:
        reviewed = [row("a", "t1w", 1, "proposed_unique")]
        hashed = copy.deepcopy(reviewed)
        hashed[0]["source_relpath"] = "changed.nii.gz"
        with self.assertRaisesRegex(RuntimeError, "source_relpath changed"):
            validate_draft_equivalence(reviewed, hashed)

    def test_duplicate_asset_is_rejected(self) -> None:
        reviewed = [row("a", "t1w", 1, "proposed_unique")]
        with self.assertRaisesRegex(RuntimeError, "Duplicate asset_id"):
            validate_draft_equivalence(reviewed + reviewed, reviewed)


class FrozenRowsTests(unittest.TestCase):
    def test_invalid_hash_is_rejected_before_source_io(self) -> None:
        rows: list[dict[str, str]] = []
        subjects = [1, 2, 7, 9, *range(11, 32)]
        # Synthetic rows match only the aggregate production gates. Detailed
        # acquisition semantics are covered by cohortlib's existing tests.
        for index in range(250):
            rows.append(row(f"b{index}", "bold", subjects[index % 25], "proposed_unique"))
        for index in range(96):
            rows.append(row(f"f{index}", "fmap", subjects[index % 25], "proposed_unique"))
        for index in range(48):
            rows.append(row(f"t{index}", "t1w", subjects[index % 25], "proposed_unique"))
        for item in rows[312:]:
            item["approval_status"] = "reviewed_candidate"
        rows[0]["source_sha256"] = ""
        config = {"cohort": {"remaining_subjects": subjects}}
        with self.assertRaisesRegex(RuntimeError, "Missing/invalid NIfTI"):
            validate_frozen_rows(config, rows)


if __name__ == "__main__":
    unittest.main()
