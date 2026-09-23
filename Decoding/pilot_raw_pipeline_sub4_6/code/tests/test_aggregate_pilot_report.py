"""Focused tests for the opt-in final-report production gate."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import aggregate_pilot_report as reporter  # noqa: E402


class CompletePilotDesignTests(unittest.TestCase):
    def test_expected_design_has_144_unique_map_rows(self) -> None:
        expected = reporter._expected_production_keys()
        self.assertEqual(sum(expected.values()), 144)
        self.assertTrue(all(count == 1 for count in expected.values()))

    def test_forbidden_diagnostic_paths_fail_closed(self) -> None:
        for marker in reporter.FORBIDDEN_PRODUCTION_PATH_MARKERS:
            with self.subTest(marker=marker):
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    reporter._reject_forbidden_paths([Path("production") / marker / "Sub6"])


class AccuracyMapValidationTests(unittest.TestCase):
    def _write_inputs(self, root: Path, dtype: np.dtype = np.float32) -> tuple[Path, dict, pd.Series]:
        affine = np.eye(4)
        mask = np.zeros((2, 2, 2), dtype=np.uint8)
        mask[0, 0, 0] = 1
        mask[1, 1, 1] = 1
        mask_path = root / "analysis_mask.nii.gz"
        nib.save(nib.Nifti1Image(mask, affine), str(mask_path))

        values = np.full(mask.shape, np.nan, dtype=dtype)
        values[0, 0, 0] = 0.25
        values[1, 1, 1] = 0.75
        map_path = root / "contrast_accuracy.nii.gz"
        nib.save(nib.Nifti1Image(values, affine), str(map_path))
        row = pd.Series(
            {
                "map_path": str(map_path),
                "n_centers_valid": 2,
                "n_input_voxels": 2,
                "n_searchlight_centers": 2,
            }
        )
        return root / "map_summary.csv", {"mask_path": str(mask_path)}, row

    def test_float32_finite_in_mask_and_nan_outside_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            source, metadata, row = self._write_inputs(Path(tmp_name))
            reporter._validate_accuracy_map(row, source, metadata, {})

    def test_non_float32_map_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            source, metadata, row = self._write_inputs(Path(tmp_name), np.float64)
            with self.assertRaisesRegex(RuntimeError, "not float32"):
                reporter._validate_accuracy_map(row, source, metadata, {})


class DefaultGridProvenanceTests(unittest.TestCase):
    def test_exact_default_grid_passes_and_coarse_grid_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            beta_root = root / "glmsingle_sdc_defaultgrid" / "Sub4"
            beta_path = beta_root / "glmsingle" / "TYPED_FITHRF_GLMDENOISE_RR.hdf5"
            beta_path.parent.mkdir(parents=True)
            beta_path.touch()
            provenance_path = beta_root / "provenance.json"
            provenance_path.write_text(
                json.dumps({"fracs": list(reporter.DEFAULT_GLMSINGLE_FRACS)}),
                encoding="utf-8",
            )
            metadata = {
                "beta_path": str(beta_path),
                "mask_path": str(root / "mask.nii.gz"),
                "manifest_path": str(root / "trials.tsv"),
            }
            reporter._validate_default_grid_glmsingle(root / "map_summary.csv", metadata)

            provenance_path.write_text(
                json.dumps({"fracs": [1.0, 0.75, 0.5, 0.25, 0.1]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "not the 20-value default"):
                reporter._validate_default_grid_glmsingle(root / "map_summary.csv", metadata)


class DecoderProvenanceLookupTests(unittest.TestCase):
    def test_nested_summary_finds_subject_root_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            subject_root = Path(tmp_name) / "Sub6"
            source = (
                subject_root
                / "glmsingle_typed"
                / "additional_smoothing-3mm"
                / "leave-one-run-out"
                / "map_summary.csv"
            )
            source.parent.mkdir(parents=True)
            payload = {
                "algorithm": {"map_outside_centers": "NaN"},
                "branches": [{"name": "glmsingle_typed"}],
            }
            provenance_path = subject_root / "provenance.json"
            provenance_path.write_text(json.dumps(payload), encoding="utf-8")

            found_path, found_payload = reporter._load_decoder_provenance(source)

            self.assertEqual(found_path, provenance_path)
            self.assertEqual(found_payload, payload)

    def test_missing_nested_provenance_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            source = (
                Path(tmp_name)
                / "Sub6"
                / "glmsingle_typed"
                / "additional_smoothing-0mm"
                / "identity-groupkfold"
                / "map_summary.csv"
            )
            source.parent.mkdir(parents=True)
            with self.assertRaisesRegex(FileNotFoundError, "searched from the retained summary"):
                reporter._load_decoder_provenance(source)


if __name__ == "__main__":
    unittest.main()
