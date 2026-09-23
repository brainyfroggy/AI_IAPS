#!/usr/bin/env python3
"""Run a tiny end-to-end GLMsingle Type-D fit on masked 2-D data.

This is deliberately synthetic and small. It exercises both zero-PC nuisance
merge call sites, actual GLM fitting, and HDF5 serialization without reading
participant data.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import run_glmsingle as runner  # noqa: E402
from glmsingle.glmsingle import GLM_single  # noqa: E402
from glmsingle.hrf.gethrf import getcanonicalhrf  # noqa: E402


def synthetic_inputs() -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(20260728)
    n_voxels = 24
    n_time = 80
    onsets = [5, 22, 41, 62]
    designs = []
    data = []
    nuisance = []
    hrf = getcanonicalhrf(runner.STIMDUR, runner.TR)

    for run in range(2):
        design = np.zeros((n_time, 2), dtype=np.float32)
        for trial, onset in enumerate(onsets):
            design[onset, (trial + run) % 2] = 1
        event_prediction = np.convolve(design.sum(axis=1), hrf, mode="full")[:n_time]
        external = np.sin(np.linspace(0, 3 * np.pi, n_time, dtype=np.float32))[:, None]
        amplitudes = np.linspace(0.7, 2.0, n_voxels, dtype=np.float32)[:, None]
        external_weights = np.linspace(-1.0, 1.0, n_voxels, dtype=np.float32)[:, None]
        signal = (
            100.0
            + amplitudes * event_prediction[None, :]
            + external_weights * external.T
            + rng.normal(0, 0.15, size=(n_voxels, n_time))
        )
        designs.append(design)
        data.append(signal.astype(np.float32))
        nuisance.append(external)
    return designs, data, nuisance


def main() -> None:
    designs, data, nuisance = synthetic_inputs()
    module = importlib.import_module("glmsingle.glmsingle")
    original_merge = module._merge_extra_regressors
    merge_calls = []

    def recording_merge(external, pc_regressors, n_pc):
        result = original_merge(external, pc_regressors, n_pc)
        merge_calls.append((n_pc, np.asarray(external).copy(), np.asarray(result).copy()))
        return result

    module._merge_extra_regressors = recording_merge
    params = {
        "wantlibrary": 0,
        "wantglmdenoise": 1,
        "wantfracridge": 1,
        "n_pcs": 0,
        "pcstop": 1.05,
        "fracs": np.asarray([1.0], dtype=np.float32),
        "wantautoscale": 0,
        "wantpercentbold": 0,
        "xvalscheme": np.arange(2, dtype=np.int64),
        "sessionindicator": np.ones(2, dtype=np.int64),
        "extra_regressors": nuisance,
        "maxpolydeg": [1, 1],
        "chunklen": 12,
        "wantfileoutputs": [0, 1, 0, 1],
        "wantmemoryoutputs": [0, 0, 0, 0],
        "wanthdf5": 1,
        "brainexclude": False,
        "pcR2cutoffmask": 1,
        "seed": 20260728,
    }

    previous = Path.cwd()
    try:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            os.chdir(tmp)
            GLM_single(params).fit(
                designs,
                data,
                runner.STIMDUR,
                runner.TR,
                outputdir=str(tmp / "outputs"),
                figuredir=None,
            )
            typed = tmp / "outputs" / "TYPED_FITHRF_GLMDENOISE_RR.hdf5"
            with h5py.File(typed, "r") as handle:
                assert handle["betasmd"].shape == (24, 1, 1, 8)
                assert int(handle.attrs["pcnum"]) == 0
                assert np.all(np.isfinite(handle["betasmd"]))
    finally:
        module._merge_extra_regressors = original_merge
        os.chdir(previous)

    # Two runs times two call sites is the lower bound when n_pcs == 0.
    zero_pc_calls = [call for call in merge_calls if call[0] == 0]
    assert len(zero_pc_calls) >= 4, f"Expected both zero-PC branches, saw {len(zero_pc_calls)} calls"
    for _, external, merged in zero_pc_calls:
        np.testing.assert_array_equal(merged, external)
    print(
        "Synthetic GLMsingle smoke passed: 2-D masked input, zero-PC nuisance "
        f"retention ({len(zero_pc_calls)} calls), and Type-D HDF5 shape."
    )


if __name__ == "__main__":
    main()
