"""Focused regression tests for the pilot GLMsingle wrapper.

These tests use only small synthetic arrays/files; they never load participant
BOLD data or launch a full pilot fit.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import nibabel as nib
import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import run_glmsingle as runner  # noqa: E402
from glmsingle.check_inputs import check_inputs  # noqa: E402
from glmsingle.glmsingle import _merge_extra_regressors  # noqa: E402


class MaskedInputTests(unittest.TestCase):
    def test_masked_two_dimensional_data_accepts_mixed_run_lengths(self) -> None:
        data = [np.ones((7, 224)), np.full((7, 218), 2.0)]
        design = []
        for n_time in (224, 218):
            item = np.zeros((n_time, 2), dtype=int)
            item[2, 0] = 1
            item[7, 1] = 1
            design.append(item)

        checked_data, checked_design, xyz = check_inputs(data, design)

        self.assertIs(xyz, False)
        self.assertEqual(
            [item.shape for item in checked_data],
            [(7, 1, 1, 224), (7, 1, 1, 218)],
        )
        self.assertTrue(all(item.dtype == np.float32 for item in checked_data))
        self.assertEqual([item.shape for item in checked_design], [(224, 2), (218, 2)])


class DiagnosticFirRunPairingTests(unittest.TestCase):
    def test_mixed_run_lengths_reach_real_fir_estimator_with_matching_nuisance(self) -> None:
        """The run-wise FIR must not reuse run 1's nuisance matrix.

        Stop immediately after the diagnostic FIR so this remains a focused,
        fast test.  The two FIR calls themselves execute the real pinned
        ``glm_estimatemodel`` implementation rather than a wrapper stub.
        """
        vendor = importlib.import_module("glmsingle.glmsingle")
        real_estimator = vendor.glm_estimatemodel
        run_lengths = (224, 218)
        designs = []
        data = []
        nuisance = []
        for run, n_time in enumerate(run_lengths):
            design = np.zeros((n_time, 1), dtype=np.float32)
            design[[8, 56, 104, 152], 0] = 1
            time = np.arange(n_time, dtype=np.float32)
            signal = np.convolve(
                design[:, 0], np.asarray([0.0, 1.0, 0.5], dtype=np.float32)
            )[:n_time]
            data.append(
                np.vstack(
                    [
                        100.0 + (voxel + 1) * signal + 0.01 * time + run
                        for voxel in range(4)
                    ]
                ).astype(np.float32)
            )
            nuisance.append(
                np.linspace(-1.0, 1.0, n_time, dtype=np.float32)[:, None]
            )
            designs.append(design)

        observed = []

        class DiagnosticComplete(RuntimeError):
            pass

        def record_real_fir(*args, **kwargs):
            if args[4] != "fir":
                raise DiagnosticComplete
            result = real_estimator(*args, **kwargs)
            opt = args[7]
            observed.append(
                (
                    args[0].shape[0],
                    args[1].shape[-1],
                    len(opt["extra_regressors"]),
                    opt["extra_regressors"][0].shape[0],
                    list(opt["maxpolydeg"]),
                )
            )
            return result

        params = {
            "extra_regressors": nuisance,
            "maxpolydeg": [0, 1],
            "firdelay": 5,
            "wantlibrary": 0,
            "wantglmdenoise": 0,
            "wantfracridge": 0,
            "wantfileoutputs": [0, 0, 0, 0],
            "wantmemoryoutputs": [0, 0, 0, 0],
            "wantpercentbold": 0,
        }
        with tempfile.TemporaryDirectory() as tmp_name:
            previous = Path.cwd()
            os.chdir(tmp_name)
            try:
                with mock.patch.object(
                    vendor, "glm_estimatemodel", side_effect=record_real_fir
                ):
                    with self.assertRaises(DiagnosticComplete):
                        vendor.GLM_single(params).fit(
                            designs,
                            data,
                            3.0,
                            1.8,
                            outputdir=str(Path(tmp_name) / "output"),
                            figuredir=None,
                        )
            finally:
                os.chdir(previous)

        self.assertEqual(
            observed,
            [
                (224, 224, 1, 224, [0]),
                (218, 218, 1, 218, [1]),
            ],
        )


class RunLengthValidationTests(unittest.TestCase):
    def _records(self, root: Path, lengths: list[int]) -> list[dict[str, Path]]:
        records = []
        for run, n_time in enumerate(lengths, start=1):
            bold = root / f"run-{run:02d}_bold.nii.gz"
            nib.save(
                nib.Nifti1Image(
                    np.zeros((1, 1, 1, n_time), dtype=np.float32), np.eye(4)
                ),
                bold,
            )
            records.append({"bold": bold})
        return records

    def test_sub05_expected_mixed_lengths_are_accepted_in_run_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            records = self._records(
                Path(tmp_name), list(runner.EXPECTED_RUN_N_VOLUMES[5])
            )
            self.assertEqual(
                runner.validate_run_n_volumes(records, 5),
                [224, 218, 224, 224, 224, 218, 218, 218, 218, 218],
            )

    def test_sub05_blanket_224_assumption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            records = self._records(Path(tmp_name), [224] * 10)
            with self.assertRaisesRegex(RuntimeError, "expected.*224, 218"):
                runner.validate_run_n_volumes(records, 5)


class NuisanceMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.external = np.arange(16, dtype=np.float32).reshape(8, 2)
        self.pcs = np.arange(24, dtype=np.float32).reshape(8, 3)

    def test_external_regressors_survive_zero_pc_candidate(self) -> None:
        merged = _merge_extra_regressors(self.external, self.pcs, 0)
        np.testing.assert_array_equal(merged, self.external)

    def test_external_regressors_survive_zero_pc_final_fit(self) -> None:
        # The CV and final-fit branches call the same helper, making the
        # zero-PC invariant explicit and independently testable.
        merged = _merge_extra_regressors(self.external, self.pcs, 0)
        self.assertEqual(merged.shape, (8, 2))

    def test_external_and_pc_regressors_are_concatenated(self) -> None:
        merged = _merge_extra_regressors(self.external, self.pcs, 2)
        np.testing.assert_array_equal(merged, np.c_[self.external, self.pcs[:, :2]])

    def test_default_false_produces_none_at_zero_pcs(self) -> None:
        self.assertIsNone(_merge_extra_regressors(False, self.pcs, 0))
        np.testing.assert_array_equal(
            _merge_extra_regressors(False, self.pcs, 2), self.pcs[:, :2]
        )


class DesignChronologyTests(unittest.TestCase):
    def test_manifest_matches_glmsingle_onset_scan_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            func = root / "sub-04" / "ses-01" / "func"
            func.mkdir(parents=True)
            records = []
            start_time = 0.45

            for run in range(1, 11):
                bold = root / f"run-{run:02d}_bold.nii.gz"
                nib.save(
                    nib.Nifti1Image(np.zeros((1, 1, 1, 64), dtype=np.float32), np.eye(4)),
                    bold,
                )
                metadata = root / f"run-{run:02d}_bold.json"
                metadata.write_text(
                    json.dumps({"RepetitionTime": runner.TR, "StartTime": start_time}),
                    encoding="utf-8",
                )
                records.append({"bold": bold, "json": metadata})

                offset = 0 if run % 2 else 60
                events = pd.DataFrame(
                    {
                        "onset": start_time + np.arange(60) * runner.TR,
                        "duration": runner.STIMDUR,
                        "trial_type": "synthetic",
                        "image_id": [f"image-{offset + index:03d}" for index in range(60)],
                    }
                )
                # Deliberately violate chronological TSV row order. The wrapper
                # must still produce a manifest aligned to GLMsingle beta order.
                events = events.iloc[::-1].reset_index(drop=True)
                events.to_csv(
                    func / f"sub-04_ses-01_task-iaps_run-{run:02d}_events.tsv",
                    sep="\t",
                    index=False,
                )

            designs, manifest, starts = runner.build_design(records, root, 4)

            self.assertEqual(starts, [start_time] * 10)
            np.testing.assert_array_equal(manifest["beta_index"], np.arange(600))
            for run in range(1, 11):
                run_rows = manifest.loc[manifest["run"] == run]
                self.assertTrue(run_rows["onset"].is_monotonic_increasing)
                self.assertEqual(int(designs[run - 1].sum()), 60)
                emitted = [
                    int(np.flatnonzero(designs[run - 1][row])[0])
                    for row in np.flatnonzero(designs[run - 1].sum(axis=1))
                ]
                self.assertEqual(emitted, run_rows["design_column"].tolist())


class Hdf5ValidationTests(unittest.TestCase):
    def _write_valid(self, path: Path, n_voxels: int) -> None:
        with h5py.File(path, "w") as handle:
            handle.create_dataset(
                "betasmd", data=np.zeros((n_voxels, 1, 1, 600), dtype=np.float32)
            )
            for key in ("HRFindex", "FRACvalue", "R2"):
                handle.create_dataset(key, data=np.zeros((n_voxels, 1, 1), dtype=np.float32))
            # GLMsingle's HDF5 helper stores scalar outputs as attributes.
            handle.attrs["pcnum"] = 0

    def test_pcnum_attribute_and_masked_beta_axes_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            path = Path(tmp_name) / "typed.hdf5"
            self._write_valid(path, 7)
            result = runner.validate_hdf5(path, 7)
            self.assertEqual(result["betasmd_shape"], (7, 1, 1, 600))
            self.assertEqual(result["pcnum"], 0)
            self.assertIn("pcnum", result["attributes"])

    def test_transposed_beta_axes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            path = Path(tmp_name) / "typed.hdf5"
            self._write_valid(path, 7)
            with h5py.File(path, "a") as handle:
                del handle["betasmd"]
                handle.create_dataset("betasmd", data=np.zeros((600, 7), dtype=np.float32))
            with self.assertRaisesRegex(RuntimeError, "Unexpected beta shape"):
                runner.validate_hdf5(path, 7)

    def test_nonfinite_beta_beyond_old_probe_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            path = Path(tmp_name) / "typed.hdf5"
            self._write_valid(path, 130)
            with h5py.File(path, "a") as handle:
                handle["betasmd"][129, 0, 0, 599] = np.nan
            with self.assertRaisesRegex(RuntimeError, "masked voxel 129, trial 599"):
                runner.validate_hdf5(path, 130, beta_block_voxels=17)

    def test_fraction_outside_requested_grid_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            path = Path(tmp_name) / "typed.hdf5"
            self._write_valid(path, 7)
            with h5py.File(path, "a") as handle:
                handle["FRACvalue"][:] = 0.25
                handle["FRACvalue"][6, 0, 0] = 0.33
            with self.assertRaisesRegex(RuntimeError, "outside the requested"):
                runner.validate_hdf5(path, 7, allowed_fracs=np.array([0.25, 0.5]))

    def test_float32_fraction_is_reported_with_canonical_grid_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            path = Path(tmp_name) / "typed.hdf5"
            self._write_valid(path, 7)
            with h5py.File(path, "a") as handle:
                handle["FRACvalue"][:] = np.float32(0.05)
            result = runner.validate_hdf5(
                path,
                7,
                allowed_fracs=np.asarray([0.05, 0.1], dtype=np.float64),
            )
            self.assertEqual(result["fractional_ridge_distribution"], {"0.05": 7})
            self.assertEqual(result["minimum_requested_fraction"], 0.05)
            self.assertEqual(result["minimum_fraction_selected_count"], 7)
            self.assertEqual(result["minimum_fraction_selected_proportion"], 1.0)

    def test_default_grid_metadata_is_canonical(self) -> None:
        self.assertEqual(len(runner.DEFAULT_FRACS), 20)
        self.assertEqual(runner.DEFAULT_FRACS[1], 0.95)
        self.assertEqual(runner.DEFAULT_FRACS[-1], 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
