# GLMsingle pilot checks

These checks are synthetic and do not read participant BOLD data.

From the project root in the pilot WSL environment:

```bash
export PYTHONPATH="$PWD/.codex_work/pilot_pipeline/vendor/GLMsingle"
python -m unittest discover -s Decoding/pilot_raw_pipeline_sub4_6/code/tests -v
python Decoding/pilot_raw_pipeline_sub4_6/code/tests/smoke_glmsingle_synthetic.py
```

The unit suite checks Sub5 February subset/merge isolation, SDC gates, and
checksum-pinned BIDS-filter archival/provenance,
Sub5's mixed 224/218-volume run sequence, masked 2-D mixed-length input
handling, sequential mixed-length temporal-SNR QC with exact-grid and
confound-length fail-closed gates, events-to-beta chronology, zero-PC nuisance-
regressor merging, Type-D HDF5 axes/attributes, exact one-pass voxel-slab
loading into the decoder, fail-closed read-only rendering of the exact
fMRIPrep SDC/coreg/ROI/carpet-reportlet matrix (including deterministic flicker-
layer selection), and the final report's strict row/path/default-grid/NIfTI
integrity gates. The smoke script
additionally performs a tiny real GLMsingle Type-D fit and verifies that both
zero-PC code paths retain the supplied nuisance regressors.

The vendored GLMsingle checkout is pinned to commit
`1ab54a65edd3ea41a6133d4b4ecb78a9c7296684`. Its reproducible local changes
are stored as two ordered, checksum-pinned patches:
`../patches/glmsingle_zero_pc_extra_regressors.patch` and
`../patches/glmsingle_mixed_run_fir_pairing.patch`. The production wrapper
reconstructs the source from the pinned commit and both patches, then verifies
that reconstruction against the live vendor source before fitting.
