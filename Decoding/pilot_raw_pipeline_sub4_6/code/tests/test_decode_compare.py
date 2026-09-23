"""Regression tests for production decoding I/O."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import nibabel as nib
import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import decode_compare as decoder  # noqa: E402


class GLMsingleBetaLoadingTests(unittest.TestCase):
    def test_voxel_slab_loading_preserves_c_order_and_remainder(self) -> None:
        """Full-trial slabs must reproduce GLMsingle's voxel/trial mapping exactly."""

        spatial_shape = (33, 17, 17)
        n_trials = 3
        n_voxels = int(np.prod(spatial_shape))
        source = np.arange(n_voxels * n_trials, dtype=np.float32).reshape(
            spatial_shape + (n_trials,)
        )

        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            mask_path = root / "analysis_mask.nii.gz"
            nib.save(
                nib.Nifti1Image(np.ones(spatial_shape, dtype=np.uint8), np.eye(4)),
                str(mask_path),
            )
            indices_path = root / "flat_mask_indices.npy"
            np.save(indices_path, np.arange(n_voxels, dtype=np.int64))
            beta_path = root / decoder.TYPE_D_NAME
            with h5py.File(beta_path, "w") as handle:
                handle.create_dataset("betasmd", data=source)

            spec = decoder.BranchSpec(
                "synthetic_glmsingle",
                "glmsingle_type_d_hdf5",
                beta_path,
                mask_path,
                root / "unused_manifest.tsv",
                (0.0,),
                indices_path,
            )
            # A tiny byte target forces many slabs and a non-divisible final slab.
            with mock.patch.object(decoder, "HDF5_TARGET_READ_BYTES", 10_000):
                loaded = decoder.load_glmsingle_betas(spec, n_trials, trial_block=1)

        expected = source.reshape(n_voxels, n_trials).T
        np.testing.assert_array_equal(loaded.data, expected)
        self.assertEqual(loaded.data.dtype, np.dtype(np.float32))
        self.assertTrue(loaded.data.flags.c_contiguous)


class RunExclusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = decoder._synthetic_manifest(n_runs=5, identities_per_cell=2)
        self.contrast = decoder.CONTRAST_BY_NAME[
            "within_natural_pleasant_vs_neutral"
        ]

    def _assert_run_absent(self, folds: list[decoder.Fold], run: int) -> None:
        for fold in folds:
            used = np.concatenate([fold.train_index, fold.test_index])
            self.assertNotIn(run, set(self.frame.iloc[used]["run"].astype(int)))

    def test_loro_excludes_run_from_training_testing_and_held_out_folds(self) -> None:
        folds = decoder.leave_one_run_out_folds(
            self.frame, self.contrast, excluded_runs=[3]
        )

        self.assertEqual(
            [fold.fold_id for fold in folds],
            ["run-01", "run-02", "run-04", "run-05"],
        )
        self.assertEqual([len(fold.train_index) for fold in folds], [12] * 4)
        self.assertEqual([len(fold.test_index) for fold in folds], [4] * 4)
        self._assert_run_absent(folds, 3)

    def test_identity_folds_also_exclude_run_from_training_and_testing(self) -> None:
        folds = decoder.identity_group_folds(
            self.frame,
            self.contrast,
            n_splits=2,
            seed=20260728,
            excluded_runs=[3],
        )

        self.assertEqual(len(folds), 2)
        self._assert_run_absent(folds, 3)

    def test_run_exclusion_validation_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate excluded runs"):
            decoder.leave_one_run_out_folds(
                self.frame, self.contrast, excluded_runs=[3, 3]
            )
        with self.assertRaisesRegex(ValueError, "not present"):
            decoder.leave_one_run_out_folds(
                self.frame, self.contrast, excluded_runs=[8]
            )
        with self.assertRaisesRegex(ValueError, "at least two"):
            decoder.leave_one_run_out_folds(
                self.frame, self.contrast, excluded_runs=[1, 2, 3, 4]
            )


if __name__ == "__main__":
    unittest.main()
