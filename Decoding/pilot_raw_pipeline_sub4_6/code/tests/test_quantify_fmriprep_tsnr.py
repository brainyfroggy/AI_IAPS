"""Synthetic tests for quantitative fMRIPrep temporal-SNR QC."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import quantify_fmriprep_tsnr as tsnr_qc  # noqa: E402


SHAPE = (3, 2, 2)
AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])


def write_run(
    root: Path,
    *,
    run: int,
    session: int,
    n_volumes: int,
    mask: np.ndarray,
    affine: np.ndarray = AFFINE,
    confound_rows: int | None = None,
    nan_voxel: tuple[int, int, int] | None = None,
) -> None:
    func = root / "sub-05" / f"ses-{session:02d}" / "func"
    prefix = f"sub-05_ses-{session:02d}_task-iaps_run-{run:02d}"
    time = np.arange(n_volumes, dtype=np.float32) - (n_volumes - 1) / 2
    scale = np.arange(1, np.prod(SHAPE) + 1, dtype=np.float32).reshape(SHAPE)
    data = 1000.0 + scale[..., None] * time
    data[0, 0, 1, :] = 1000.0  # Positive mean but zero temporal SD: invalid tSNR.
    if nan_voxel is not None:
        data[nan_voxel + (0,)] = np.nan

    bold_path = func / (
        prefix + "_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    )
    bold_path.parent.mkdir(parents=True, exist_ok=True)
    bold_image = nib.Nifti1Image(data.astype(np.float32), affine)
    bold_image.header.set_zooms((2.0, 2.0, 2.0, 1.8))
    nib.save(bold_image, bold_path)

    mask_path = func / (
        prefix + "_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz"
    )
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), mask_path)
    (func / (prefix + "_space-MNI152NLin6Asym_res-2_desc-preproc_bold.json")).write_text(
        '{"RepetitionTime": 1.8}\n', encoding="utf-8"
    )
    row_count = n_volumes if confound_rows is None else confound_rows
    (func / (prefix + "_desc-confounds_timeseries.tsv")).write_text(
        "framewise_displacement\n" + "\n".join("0" for _ in range(row_count)) + "\n",
        encoding="utf-8",
    )


def make_two_run_derivative(
    root: Path,
    *,
    second_affine: np.ndarray = AFFINE,
    second_confound_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mask_one = np.ones(SHAPE, dtype=np.uint8)
    mask_two = np.ones(SHAPE, dtype=np.uint8)
    mask_one[0, 0, 0] = 0
    mask_two[0, 0, 0] = 0  # Outside every run.
    mask_two[1, 0, 0] = 0  # Covered only in run 1.
    write_run(root, run=1, session=1, n_volumes=5, mask=mask_one)
    write_run(
        root,
        run=2,
        session=2,
        n_volumes=7,
        mask=mask_two,
        affine=second_affine,
        confound_rows=second_confound_rows,
        nan_voxel=(2, 1, 1),
    )
    return mask_one, mask_two


class QuantitativeTsnrTests(unittest.TestCase):
    def test_mixed_lengths_float32_maps_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            derivative = root / "fmriprep"
            output = root / "reports" / "Sub5"
            make_two_run_derivative(derivative)
            input_bytes = {
                path.relative_to(derivative): path.read_bytes()
                for path in derivative.rglob("*")
                if path.is_file()
            }

            result = tsnr_qc.run_qc(
                derivative,
                5,
                output,
                expected_runs=(1, 2),
                z_chunk=1,
            )

            self.assertEqual(result, output.resolve())
            self.assertTrue(output.is_dir())
            self.assertFalse((output.parent / ".Sub5.staging").exists())
            maps = sorted(output.rglob("*.nii.gz"))
            self.assertEqual(len(maps), 8)  # Two maps/run plus four aggregate maps.
            for path in maps:
                image = nib.load(path)
                self.assertEqual(np.dtype(image.get_data_dtype()), np.dtype(np.float32))
                self.assertEqual(image.shape, SHAPE)
                self.assertTrue(np.array_equal(image.affine, AFFINE))

            run_tsnr_paths = sorted((output / "per_run").glob("*_desc-tsnr_map.nii.gz"))
            self.assertEqual(len(run_tsnr_paths), 2)
            run_tsnr = [np.asarray(nib.load(path).dataobj) for path in run_tsnr_paths]
            aggregate = np.asarray(
                nib.load(next(output.glob("*_desc-mean_tsnr_map.nii.gz"))).dataobj
            )
            both_valid = (1, 1, 1)
            self.assertAlmostEqual(
                float(aggregate[both_valid]),
                float((run_tsnr[0][both_valid] + run_tsnr[1][both_valid]) / 2),
                places=5,
            )
            self.assertTrue(np.isnan(aggregate[0, 0, 1]))  # Constant in both runs.

            mask_coverage = np.asarray(
                nib.load(next(output.glob("*_desc-brainmask_coverage_fraction_map.nii.gz"))).dataobj
            )
            valid_coverage = np.asarray(
                nib.load(next(output.glob("*_desc-valid_tsnr_coverage_fraction_map.nii.gz"))).dataobj
            )
            self.assertTrue(np.isnan(mask_coverage[0, 0, 0]))
            self.assertEqual(float(mask_coverage[1, 0, 0]), 0.5)
            self.assertEqual(float(mask_coverage[both_valid]), 1.0)
            self.assertEqual(float(valid_coverage[0, 0, 1]), 0.0)
            self.assertEqual(float(valid_coverage[2, 1, 1]), 0.5)

            with (output / "tsnr_summary.tsv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(rows), 3)
            self.assertEqual([int(row["n_volumes"]) for row in rows[:2]], [5, 7])
            self.assertEqual(rows[2]["row_type"], "aggregate_equal_run_mean")
            self.assertEqual(int(rows[2]["n_volumes"]), 12)

            provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["expected_runs"], [1, 2])
            self.assertEqual([run["n_volumes"] for run in provenance["runs"]], [5, 7])
            self.assertEqual(provenance["preprocessing_applied_by_this_utility"], [])
            self.assertEqual(
                input_bytes,
                {
                    path.relative_to(derivative): path.read_bytes()
                    for path in derivative.rglob("*")
                    if path.is_file()
                },
            )

    def test_exact_affine_mismatch_fails_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            derivative = root / "fmriprep"
            output = root / "reports" / "Sub5"
            wrong_affine = AFFINE.copy()
            wrong_affine[0, 3] = 0.25
            make_two_run_derivative(derivative, second_affine=wrong_affine)

            with self.assertRaisesRegex(RuntimeError, "affine mismatch"):
                tsnr_qc.run_qc(derivative, 5, output, expected_runs=(1, 2), z_chunk=1)
            self.assertFalse(output.exists())
            self.assertTrue((output.parent / ".Sub5.staging").is_dir())

    def test_confound_length_mismatch_fails_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            derivative = root / "fmriprep"
            output = root / "reports" / "Sub5"
            make_two_run_derivative(derivative, second_confound_rows=6)

            with self.assertRaisesRegex(RuntimeError, r"Confound rows \(6\).*volumes \(7\)"):
                tsnr_qc.run_qc(derivative, 5, output, expected_runs=(1, 2), z_chunk=1)
            self.assertFalse(output.exists())
            self.assertTrue((output.parent / ".Sub5.staging").is_dir())


if __name__ == "__main__":
    unittest.main(verbosity=2)
