# Sub4–6 raw-to-decoding pilot runbook

This is the reproducible production path for the 1.8-mm acquisition pilot. It
compares the existing SPM LS-A betas with new fMRIPrep + GLMsingle Type-D betas
using the same leakage-safe decoder. The pilot is exploratory (`n=3`); it is a
pipeline screen, not group-level inference.

The project-wide default cohort remains 28 subjects, excluding Sub3 (Emily),
Sub8 (earrings), and Sub10 (not usable). This runbook only concerns Sub4, Sub5,
and Sub6.

## Fixed inputs and versions

| Item | Fixed value |
|---|---|
| Raw MRI | `N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\rawfMRI\IAPS-DEV` |
| Onset logs | `N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\DataRecording\Sub<id>\LogFiles\Run<run>.mat` |
| Stimulus manifest | `N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\stimuli_600trials.csv` |
| Pilot root | `N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\pilot_raw_pipeline_sub4_6` |
| Fast local Docker root | `C:\Users\yujunchen\.cache\ai_iaps_fmriprep` |
| fMRIPrep | `nipreps/fmriprep:25.1.3`; image ID/digest `sha256:4e5cfd99f6d80a9ef10a87929f8e74e4caf9dc108b49551eb23c775e61cd16f7` |
| Output space | `MNI152NLin6Asym:res-2` |
| GLMsingle commit | `1ab54a65edd3ea41a6133d4b4ecb78a9c7296684`; ordered patch SHA-256 values: zero-PC nuisance retention `b8f109d07f5d22299ae80d1303fa29fa55eec9ffaf555a63c885a8764ca625d4`, mixed-run FIR pairing `62a8a803ca4d6b1e459064a177a65a45d3fc152f02dd29dc94dac12a17fd729c` |
| Primary CV | Leave one run out (LORO) |
| Sensitivity CV | Stimulus-identity GroupKFold |
| Searchlight | 5-mm radius, diagonal LDA, training-fold-only scaling |
| New-beta smoothing | 0 and 3 mm FWHM, mask normalized and applied trial by trial |

Use local `C:` storage for Docker input, output, and named work volumes. Keep
large Docker/fMRIPrep jobs sequential unless free disk and memory are checked
again. Do not remove a named work volume until its derivative is complete,
visually checked, archived, and consumed by GLMsingle.

## Session-specific distortion correction

| Subject/session | Runs | Correction used |
|---|---:|---|
| Sub4 ses-01 | 01–10 | Same-session measured AP/PA reverse-PE; PEPOLAR |
| Sub5 ses-01 (February) | 01, 03, 04, 05 | No same-session fieldmap; isolated BIDS subset and fieldmap-less SyN |
| Sub5 ses-02 (March) | 02, 06–10 | Same-session measured AP/PA reverse-PE; PEPOLAR |
| Sub6 ses-01 | 01–05 | Same-session measured AP/PA reverse-PE; PEPOLAR |
| Sub6 ses-02 | 06–10 | Same-session measured AP/PA reverse-PE; PEPOLAR |

Sub5's T1w was acquired in March and is intentionally reused as the
subject-level anatomical image for both branches. Never associate the March
reverse-PE pair with February BOLD. In the full Sub5 tree, fMRIPrep sees a
subject-level measured fieldmap and does **not** activate SyN for February;
those full-tree February derivatives must never enter GLMsingle. The merge
script enforces the split above and checks the fMRIPrep summary report markers.

## 1. Code gates

Run these before production or after any code change:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash -lc "source /home/yujun/.cache/ai_iaps_pilot_venv/bin/activate && cd /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS && python -m unittest discover Decoding/pilot_raw_pipeline_sub4_6/code/tests -p 'test_*.py' -v"

wsl.exe -d Ubuntu-22.04 -- bash -lc "source /home/yujun/.cache/ai_iaps_pilot_venv/bin/activate && python /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/code/decode_compare.py --self-test"
```

Gate: all orchestration/GLMsingle tests pass and the decoder prints
`SELF-TEST PASSED`. The decoder self-test includes matched decimal identities
such as `2900.1` and `2900.1_1`; identity parsing must not collapse them to
`2900`. A real 5,000-voxel GLMsingle-to-decoder integration result is retained
under `pilot_raw_pipeline_sub4_6/smoke/decode_integration_sub06_5k` for code
integration only, never for production aggregation.

Also verify the GLMsingle checkout:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash -lc "git -C /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/.codex_work/pilot_pipeline/vendor/GLMsingle rev-parse HEAD"
```

Gate: the printed commit is the pinned commit in the table above. At production
runtime, `run_glmsingle.py` additionally verifies both ordered patch hashes,
reconstructs the patched source from the pinned commit, and requires that its
SHA-256 exactly match the live vendor source before fitting.

```powershell
docker image inspect nipreps/fmriprep:25.1.3 --format '{{.Id}}'
```

Gate: the image ID is the pinned digest in the table above.

## 2. Build and validate the corrected BIDS input

For a fresh local staging tree (the destination must be absent or unfinished
work must be audited first):

```powershell
wsl.exe -d Ubuntu-22.04 -- bash -lc "python3 /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/code/build_pilot_bids.py --output /mnt/c/Users/yujunchen/.cache/ai_iaps_fmriprep/bids"
```

The builder reads the raw MRI and MATLAB onset logs, requires exact source
matches, checks all 60 stimuli per run against `stimuli_600trials.csv`, and
writes `code/pilot_source_manifest.tsv`. It must produce 10 runs and 600 trials
per subject.

Validate the full tree with the official BIDS Validator. Install the official
precompiled validator in a separate environment so the analysis environment
does not change:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash -lc "python3 -m venv /home/yujun/.cache/ai_iaps_bids_validator_venv && source /home/yujun/.cache/ai_iaps_bids_validator_venv/bin/activate && python -m pip install bids-validator-deno==3.0.1"

wsl.exe -d Ubuntu-22.04 -- bash -lc "source /home/yujun/.cache/ai_iaps_bids_validator_venv/bin/activate && bids-validator-deno --format json_pp --outfile /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/reports/pilot_bids_validator_v3.0.1.json /mnt/c/Users/yujunchen/.cache/ai_iaps_fmriprep/bids"
```

Gate: zero errors. Review every warning; do not merely rely on fMRIPrep's
`--skip-bids-validation` flag.

If the local staging tree already existed, rerun the builder with `--refresh`
before analysis and verify that all 30 local `events.tsv` files are byte-identical
to the canonical pilot BIDS tree on `N:`. A stale staging tree can otherwise mix
event-table schemas even though fMRIPrep itself ignores event files. GLMsingle
should read its small events files from the canonical `N:` BIDS root below.

## 3. Run fMRIPrep

The wrapper is synchronous, pins all production options, uses a read-only BIDS
mount, and preserves work in a named Docker volume. It refuses a non-empty
output or existing work volume unless `-Resume` is supplied. `-DryRun` has no
filesystem or Docker-volume side effects.

Set paths once in PowerShell:

```powershell
$Code = 'N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\pilot_raw_pipeline_sub4_6\code'
$Local = 'C:\Users\yujunchen\.cache\ai_iaps_fmriprep'
$Bids = "$Local\bids"
$License = 'C:\Users\yujunchen\.cache\ai_iaps_fmriprep_license.txt'
```

Run or resume one full subject at a time:

```powershell
& "$Code\run_fmriprep_container.ps1" -ParticipantLabel 04 -BidsRoot $Bids -OutputRoot "$Local\out_sub04" -WorkVolume ai_iaps_fmriprep_work_sub04 -LicenseFile $License -NThreads 5 -OmpNThreads 2 -MemoryMb 14000 -Resume

& "$Code\run_fmriprep_container.ps1" -ParticipantLabel 05 -BidsRoot $Bids -OutputRoot "$Local\out_sub05" -WorkVolume ai_iaps_fmriprep_work_sub05 -LicenseFile $License -NThreads 4 -OmpNThreads 2 -MemoryMb 10000 -Resume

& "$Code\run_fmriprep_container.ps1" -ParticipantLabel 06 -BidsRoot $Bids -OutputRoot "$Local\out" -WorkVolume ai_iaps_fmriprep_work_sub06 -LicenseFile $License -Resume
```

For a genuinely fresh output and work volume, omit `-Resume`. Never launch a
second writer for the same output root. A stopped `--rm` container can be
resumed because the named volume and output directory survive.

### Sub5 February SyN branch

Build an input containing only the March T1w and February functional runs. It
contains no March BOLD and no fieldmap, and its generated `participants.tsv`
contains only `sub-05`:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash -lc "python3 /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/code/build_sub05_feb_syn_subset.py /mnt/c/Users/yujunchen/.cache/ai_iaps_fmriprep/bids /mnt/c/Users/yujunchen/.cache/ai_iaps_fmriprep/bids_sub05_feb_syn --hardlink"

wsl.exe -d Ubuntu-22.04 -- bash -lc "source /home/yujun/.cache/ai_iaps_bids_validator_venv/bin/activate && bids-validator-deno --format json_pp --outfile /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/reports/sub05_feb_syn_bids_validator_v3.0.1.json /mnt/c/Users/yujunchen/.cache/ai_iaps_fmriprep/bids_sub05_feb_syn"
```

Gate: the builder reports exactly four BOLD runs (01, 03, 04, 05), the
validator reports zero errors, and there is no `fmap` or `ses-02` path. The
builder also writes the analysis-control file
`$Local\bids_sub05_feb_syn_fmriprep_bids_filter.json` outside the BIDS root.

This filter is required with fMRIPrep 25.1.3/sdcflows 2.13.1/PyBIDS 0.19.0. With only
session-labelled BOLD and a valid subject-level (sessionless) T1w, sdcflows
otherwise derives `sessions=['01']` and then incorrectly narrows its T1w query
to `session=['01']`, returning no fieldmap-less estimator. The pinned filter
sets only the `fmap` discovery query's session entity to PyBIDS
`Query.OPTIONAL`, which includes sessionless anatomy and ses-01 BOLD without
renaming or falsely session-labelling either acquisition.

Run the exact-image, no-output preflight before the expensive branch:

```powershell
$SynFilter = "$Local\bids_sub05_feb_syn_fmriprep_bids_filter.json"
& "$Code\run_fmriprep_container.ps1" -ParticipantLabel 05 -BidsRoot "$Local\bids_sub05_feb_syn" -OutputRoot "$Local\out_sub05_feb_syn" -WorkVolume ai_iaps_fmriprep_work_sub05_feb_syn -ContainerName ai-iaps-fmriprep-sub05-feb-syn -LicenseFile $License -NThreads 6 -OmpNThreads 2 -MemoryMb 26000 -BidsFilterFile $SynFilter -SynSdcMode error -RequireSynAnatEstimator -PreflightOnly
```

Gate: `SyN ANAT estimator preflight passed: 4 estimators cover all 4 BOLD
inputs`. This command does not create the output directory or work volume.

Run the isolated branch:

```powershell
& "$Code\run_fmriprep_container.ps1" -ParticipantLabel 05 -BidsRoot "$Local\bids_sub05_feb_syn" -OutputRoot "$Local\out_sub05_feb_syn" -WorkVolume ai_iaps_fmriprep_work_sub05_feb_syn -ContainerName ai-iaps-fmriprep-sub05-feb-syn -LicenseFile $License -NThreads 6 -OmpNThreads 2 -MemoryMb 26000 -BidsFilterFile $SynFilter -SynSdcMode error -RequireSynAnatEstimator
```

If this exact branch was interrupted and both paths were confirmed, add
`-Resume`. The isolated branch deliberately uses `error`, not `warn`, so a
future estimator-discovery regression fails during workflow construction.

Provenance caveat: Sub5's sessionless T1w was acquired in March and is reused
as the anatomical reference for the February BOLD runs. The compatibility
filter does not change that acquisition fact or relabel the T1w as February;
retain the filter and both branch logs with the merged derivative.

### fMRIPrep completion and visual-QC gate

For each usable subject root, require one preprocessed BOLD, mask, preprocessed
JSON, and confounds TSV per expected run, plus a completed `sub-XX.html`.

```powershell
(Get-ChildItem "$Local\out_sub04\sub-04" -Recurse -Filter '*_desc-preproc_bold.nii.gz').Count
(Get-ChildItem "$Local\out\sub-06" -Recurse -Filter '*_desc-preproc_bold.nii.gz').Count
(Get-ChildItem "$Local\out_sub05_feb_syn\sub-05" -Recurse -Filter '*_desc-preproc_bold.nii.gz').Count
```

Gate: counts are 10, 10, and 4. The full Sub5 branch must contain 10 outputs,
but only its six March outputs will be used.

Sub5 has an intentional mixed temporal length. In run-number order, both the
raw and selected/merged preprocessed BOLDs must have
`[224, 218, 224, 224, 224, 218, 218, 218, 218, 218]` volumes: the four February
runs (01/03/04/05) have 224, while the six March runs (02/06-10) have 218.
Require each run's confounds table to have the same number of rows as that
run's BOLD. Do not pad, crop, or impose 224 volumes on the March runs.

Open the HTML and SDC/coregistration reportlets. Every Sub4/Sub6 run and every
Sub5 March run must say `PEB/PEPOLAR`; every isolated Sub5 February run must say
fieldmap-less `SyN`. Check before/after distortion overlays, EPI-to-T1/MNI
alignment, brain coverage, signal dropout, and motion. SDC addresses geometric
susceptibility distortion; it cannot restore signal already lost to dropout or
fix low temporal SNR.

## 4. Merge Sub5 without leaking the no-SDC February branch

The destination must be absent or empty:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash -lc "python3 /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/code/merge_sub05_fmriprep.py /mnt/c/Users/yujunchen/.cache/ai_iaps_fmriprep/out_sub05 /mnt/c/Users/yujunchen/.cache/ai_iaps_fmriprep/out_sub05_feb_syn /mnt/c/Users/yujunchen/.cache/ai_iaps_fmriprep/out_sub05_combined --feb-bids-filter /mnt/c/Users/yujunchen/.cache/ai_iaps_fmriprep/bids_sub05_feb_syn_fmriprep_bids_filter.json --hardlink"
```

The merge fails unless it finds:

- February runs 01/03/04/05 with `SyN` in each summary report;
- March runs 02/06/07/08/09/10 with `PEPOLAR` in each summary report;
- exactly 10 preprocessed BOLDs, masks, JSON sidecars, and confounds files;
- a February BIDS filter whose SHA-256 exactly matches the pinned
  `code/sub05_feb_syn_bids_filter.json` analysis-control file;
- no unrelated subject copied from a shared derivative root.

The two complete report bundles remain separate at
`out_sub05_combined/logs/merge_branches/{all_sessions,february_syn}/sub-05.html`,
the verified filter is independently copied to
`out_sub05_combined/logs/merge_branches/february_syn/bids_filter.json`, and
`logs/merge_sub05_provenance.json` records the source assignment plus the
filter's resolved source path, archived relative path, and full SHA-256. The
merge verifies the archived copy against the source checksum. There is
deliberately no misleading canonical `sub-05.html` assembled from mixed
reportlets.

### Archive the verified Sub5 merge

Archive the merged derivative before GLMsingle or cleanup. The canonical
shared derivative root keeps the combined subject at `fmriprep_sdc/sub-05` and
keeps `logs/merge_branches` plus `logs/merge_sub05_provenance.json` at the same
relative paths recorded by the merge provenance. Because the archived February
filter is inside `logs/merge_branches/february_syn`, it is copied, compared, and
included in the canonical manifest by the same transaction.

Use the Windows-native archiver below. Do not copy from WSL to the mapped `N:`
share with `rsync`: that route can fail on temporary-file creation or ownership
metadata even when the source is valid. `archive_sub05.ps1` instead uses
`robocopy` on Windows, refuses every occupied canonical or staging target,
requires matching derivative-root metadata, hashes every source/staged file,
atomically moves the verified trees into place on the archive volume, writes a
UTF-8/LF SHA-256 manifest, and verifies every canonical file plus the pinned
February BIDS-filter provenance. It never overwrites the existing Sub4/Sub6
archive or root metadata.

```powershell
$ArchiveRoot = 'N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\pilot_raw_pipeline_sub4_6\fmriprep_sdc'

& "$Code\archive_sub05.ps1" `
    -CombinedRoot "$Local\out_sub05_combined" `
    -ArchiveRoot $ArchiveRoot `
    -PinnedFebruaryBidsFilter "$Code\sub05_feb_syn_bids_filter.json"
```

For the validated Sub5 merge, successful completion prints exactly this gate
summary (the hexadecimal hash is lowercase):

```text
ARCHIVE_PASS subject_files=341 manifest_lines=489 filter_sha256=aca68315de44714d07bc182cccae64a8a619485ff2ea2734b588cc6667751195
```

Gate: `sub-05` contains exactly 10 target-space preprocessed BOLD NIfTIs, 10
matching JSON sidecars, 10 masks, and 10 confounds files: four ses-01 files and
six ses-02 files for every suffix. Both branch HTML reports and all their assets
exist under `logs/merge_branches`, the merge provenance exists, the SHA-256
manifest verifies, and canonical `fmriprep_sdc/sub-05.html` is absent. Require
`logs/merge_branches/february_syn/bids_filter.json` to exist and require its
SHA-256 to equal `february_bids_filter.sha256` in
`logs/merge_sub05_provenance.json`; that value must also equal the pinned filter
checksum. Do not remove either local branch or its Docker work volume until this
archive gate and visual QC have passed.

### Quantitative tSNR and valid-coverage QC

Run the read-only quantitative QC utility against the canonical archive after
that subject's archive and visual-QC gates pass. It loads each compressed run
once and computes it in bounded z-slabs, requires the exact run set, exact
BOLD/mask grid equality, a binary nonempty mask, a common TR, and one confounds
row per native BOLD volume.
It deliberately applies no detrending, filtering, smoothing, nuisance
regression, censoring, or temporal padding. Thus the maps describe the supplied
fMRIPrep series rather than creating another preprocessing branch.

```powershell
$QcSubject = 5
wsl.exe -d Ubuntu-22.04 -- bash -lc "source /home/yujun/.cache/ai_iaps_pilot_venv/bin/activate && python /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/code/quantify_fmriprep_tsnr.py --subject $QcSubject --fmriprep-root /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/fmriprep_sdc --space MNI152NLin6Asym --resolution 2 --expected-runs 1 2 3 4 5 6 7 8 9 10 --output /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/reports/fmriprep_quantitative_qc/Sub$QcSubject"
if ($LASTEXITCODE -ne 0) { throw "Quantitative fMRIPrep QC failed for Sub$QcSubject" }
```

Set `$QcSubject` to 4, 5, or 6 and use a new output path each time. A complete
10-run result contains 10 per-run tSNR maps, 10 per-run valid-voxel maps, four
aggregate maps (equal-run mean tSNR, brain-mask coverage fraction, valid-tSNR
coverage fraction, and valid-run count), an 11-row `tsnr_summary.tsv`, and
`provenance.json`. Every map is float32 `.nii.gz`; per-run tSNR uses temporal
mean divided by sample temporal SD (`ddof=1`), and the aggregate weights each
run equally despite Sub5's mixed 224/218-volume lengths. Invalid/out-of-mask
locations remain `NaN` where appropriate. These files belong only under
`reports/fmriprep_quantitative_qc`; never pass this root as a strict decoding
summary root.

## 5. Estimate GLMsingle Type-D betas

Run one memory-heavy fit at a time. Outputs must not exist. The wrapper uses
120 shared image-identity columns, acquisition-session indicators, 24P motion,
FD > 0.5 one-hot spikes, voxelwise HRF selection, GLMdenoise, and a 20-value
fractional-ridge grid from `1.00` through `0.05` in steps of `0.05`. The
commands intentionally omit `--fracs`, which invokes that default grid. A
five-value Sub6 sensitivity fit (`1, .75, .5, .25, .1`) selected the weakest
tested regularization in 99.91% of voxels and is retained only under the
versioned coarse-grid output roots. The 20-value Sub6 fit still selected its
`0.05` boundary in 99.03% of voxels. This limitation must be reported; the
standard grid is retained because its decoded accuracies were stable relative
to the coarse fit (mean absolute whole-mask difference 0.00422, maximum 0.01).

For performance, stage each completed derivative from Windows into WSL ext4
before fitting, then require a checksum dry-run with no output. The GLMsingle
provenance records this staged input path. Example for Sub6:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash -lc "mkdir -p /home/yujun/ai_iaps_pilot_inputs/fmriprep_sub06 && rsync -a /mnt/c/Users/yujunchen/.cache/ai_iaps_fmriprep/out/sub-06/ /home/yujun/ai_iaps_pilot_inputs/fmriprep_sub06/sub-06/"

wsl.exe -d Ubuntu-22.04 -- bash -lc "rsync -rcn --delete --itemize-changes /mnt/c/Users/yujunchen/.cache/ai_iaps_fmriprep/out/sub-06/ /home/yujun/ai_iaps_pilot_inputs/fmriprep_sub06/sub-06/ > /tmp/sub06_fmriprep_rsync_check.txt && test ! -s /tmp/sub06_fmriprep_rsync_check.txt"
```

Use the analogous `fmriprep_sub04/sub-04` staging root for Sub4. For Sub5,
the source **must** be the checksum-verified canonical combined derivative—
never `out_sub05`, whose February runs lack SDC. The staging command refuses
both an existing final path and an abandoned staging path, checksum-compares
all files, and then atomically renames the staging directory on WSL ext4:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash -lc 'set -euo pipefail; src=/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/fmriprep_sdc/sub-05; final=/home/yujun/ai_iaps_pilot_inputs/fmriprep_sub05; stage=/home/yujun/ai_iaps_pilot_inputs/.fmriprep_sub05-staging; test -d "\$src"; test ! -e "\$final"; test ! -e "\$stage"; mkdir -p "\$stage/sub-05"; rsync -a "\$src/" "\$stage/sub-05/"; rsync -rcn --delete --itemize-changes "\$src/" "\$stage/sub-05/" > /tmp/sub05_combined_fmriprep_rsync_check.txt; test ! -s /tmp/sub05_combined_fmriprep_rsync_check.txt; mv -T "\$stage" "\$final"'
if ($LASTEXITCODE -ne 0) { throw 'Sub5 WSL staging or checksum comparison failed' }
```

Keep `/home/yujun/ai_iaps_pilot_inputs/fmriprep_sub05/sub-05` until final
strict aggregation has passed, because the GLMsingle provenance records this
input root.

```powershell
wsl.exe -d Ubuntu-22.04 -- bash -lc "source /home/yujun/.cache/ai_iaps_pilot_venv/bin/activate && python /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/code/run_glmsingle.py --subject 4 --fmriprep-root /home/yujun/ai_iaps_pilot_inputs/fmriprep_sub04 --bids-root /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/bids --space MNI152NLin6Asym --output /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/glmsingle_sdc_defaultgrid/Sub4 --n-pcs 10 --chunklen 10000"

wsl.exe -d Ubuntu-22.04 -- bash -lc "source /home/yujun/.cache/ai_iaps_pilot_venv/bin/activate && python /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/code/run_glmsingle.py --subject 5 --fmriprep-root /home/yujun/ai_iaps_pilot_inputs/fmriprep_sub05 --bids-root /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/bids --space MNI152NLin6Asym --output /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/glmsingle_sdc_defaultgrid/Sub5 --n-pcs 10 --chunklen 10000"

wsl.exe -d Ubuntu-22.04 -- bash -lc "source /home/yujun/.cache/ai_iaps_pilot_venv/bin/activate && python /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/code/run_glmsingle.py --subject 6 --fmriprep-root /home/yujun/ai_iaps_pilot_inputs/fmriprep_sub06 --bids-root /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/bids --space MNI152NLin6Asym --output /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/glmsingle_sdc_defaultgrid/Sub6 --n-pcs 10 --chunklen 10000"
```

Gate for every subject:

- `validation.json` exists, reports `betasmd_shape = [n_voxels,1,1,600]`, and
  confirms that every beta is finite;
- `glmsingle/TYPED_FITHRF_GLMDENOISE_RR.hdf5` exists;
- `trial_manifest.tsv` has 600 rows and 60 rows per run;
- `motion_qc.tsv` has 10 rows;
- `provenance.json` records the pinned commit, ordered patch hashes and verified
  source state, session indicators, all 20 ridge fractions, 10 candidate PCs,
  and the exact 10 input BOLDs;
- for Sub5, `provenance.json.run_n_volumes` is exactly
  `[224, 218, 224, 224, 224, 218, 218, 218, 218, 218]`; GLMsingle sizes each
  run's design and nuisance matrices independently;
- every `FRACvalue` belongs to the requested grid; report its distribution and
  the fraction at the `0.05` boundary.

Do not use `--overwrite` on a production result. Choose a new versioned output
if a rerun is required.

## 6. Decode the new betas with the matched decoder

Each destination must be new. Omitting `--contrasts` runs all four within-source
and four directional cross-source contrasts.

The matched existing-beta baseline is already complete under
`decoding_corrected_legacy/Sub4`, `Sub5`, and `Sub6`. If reproducing it from
scratch, run the following only when those destinations are absent (otherwise
choose a new versioned root and pass that root to the reporter):

```powershell
wsl.exe -d Ubuntu-22.04 -- bash -lc 'source /home/yujun/.cache/ai_iaps_pilot_venv/bin/activate && for s in 4 5 6; do python /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/code/decode_compare.py --subject "$s" --only legacy --output "/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/decoding_corrected_legacy/Sub$s" --legacy-smoothing-mm 0 --cv leave-one-run-out identity-groupkfold --classifier diaglda --variance-shrinkage .1 --radius-mm 5 --center-step 1 --n-jobs 6 || exit; done'
```

Gate: each legacy subject has 16 map-summary rows (8 contrasts × 2 CV schemes)
and 16 float32 NIfTI maps.

Before decoding GLMsingle, copy exactly the six decoder/runtime files
(`analysis_mask.nii.gz`, `flat_mask_indices.npy`, `trial_manifest.tsv`,
`provenance.json`, `validation.json`, and the Type-D HDF5) to a hidden
same-parent WSL staging directory. Require exact inventory, full SHA-256
agreement before and after the copy, the same deep HDF5/mask/manifest gate used
on the source, and an atomic rename to
`/home/yujun/ai_iaps_pilot_inputs/glmsingle_sdc_defaultgrid/SubN`. Never decode
while the GLMsingle producer is active, and never reuse an abandoned staging
directory blindly. Keep these checksum-identical local stages through strict
aggregation.

```powershell
wsl.exe -d Ubuntu-22.04 -- bash -lc "source /home/yujun/.cache/ai_iaps_pilot_venv/bin/activate && python /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/code/decode_compare.py --subject 4 --only glmsingle --glmsingle-dir /home/yujun/ai_iaps_pilot_inputs/glmsingle_sdc_defaultgrid/Sub4 --output /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/decoding_new_sdc/Sub4 --new-smoothing-mm 0 3 --cv leave-one-run-out identity-groupkfold --classifier diaglda --variance-shrinkage .1 --radius-mm 5 --center-step 1 --n-jobs 6"

wsl.exe -d Ubuntu-22.04 -- bash -lc "source /home/yujun/.cache/ai_iaps_pilot_venv/bin/activate && python /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/code/decode_compare.py --subject 5 --only glmsingle --glmsingle-dir /home/yujun/ai_iaps_pilot_inputs/glmsingle_sdc_defaultgrid/Sub5 --output /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/decoding_new_sdc/Sub5 --new-smoothing-mm 0 3 --cv leave-one-run-out identity-groupkfold --classifier diaglda --variance-shrinkage .1 --radius-mm 5 --center-step 1 --n-jobs 6"

wsl.exe -d Ubuntu-22.04 -- bash -lc "source /home/yujun/.cache/ai_iaps_pilot_venv/bin/activate && python /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/code/decode_compare.py --subject 6 --only glmsingle --glmsingle-dir /home/yujun/ai_iaps_pilot_inputs/glmsingle_sdc_defaultgrid/Sub6 --output /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/decoding_new_sdc/Sub6 --new-smoothing-mm 0 3 --cv leave-one-run-out identity-groupkfold --classifier diaglda --variance-shrinkage .1 --radius-mm 5 --center-step 1 --n-jobs 6"
```

Gate: every subject has 32 `map_summary.csv` rows (8 contrasts × 2 CV schemes ×
2 smoothing levels), complete fold summaries, and 32 float32 `.nii.gz` maps.
Maps must contain `NaN` outside evaluated centers, never quantized `uint8`.
Training scaling and classifier parameters are fold-local.

Historical ERP-style maps remain useful context, but their random validation
and averaging make them non-equivalent to the matched LORO comparison.

## 7. Build the side-by-side report and group brain maps

Use explicit production roots even though the reporter's default scan is now
restricted to `decoding_corrected_legacy` and `decoding_new_sdc`. It does not
scan `smoke`, preventing the integration result from contaminating production.
The output must be new (or use `--refresh` only on artifacts owned by the
reporter).

```powershell
wsl.exe -d Ubuntu-22.04 -- bash -lc "source /home/yujun/.cache/ai_iaps_pilot_venv/bin/activate && python /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/code/aggregate_pilot_report.py --summary-root /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/decoding_corrected_legacy --summary-root /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/decoding_new_sdc --require-complete-pilot --output /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/reports/final_sub4-6_v1"
```

Gate:

- `REPORT.md`, `side_by_side_long.csv`, `side_by_side_wide.csv`,
  `branch_summary.csv`, and `group_map_manifest.csv` exist;
- the opt-in completeness gate passes exactly 16 legacy map rows plus 120 fold
  rows per subject, and exactly 32 GLMsingle map rows plus 240 fold rows per
  subject;
- no source path contains `smoke`, `coarsegrid`, or `failed`, and every
  `glmsingle_typed` source has the 20-value default-grid provenance;
- every source accuracy map is float32, has finite in-mask values in `[0,1]`,
  contains NaN outside evaluated centers, and agrees with its recorded valid
  center count;
- group mean-accuracy and valid-count maps exist for each comparable variant;
- mean-accuracy maps are float32 `.nii.gz` and count maps preserve validity;
- the primary decision is based on LORO; identity-GroupKFold is sensitivity.

Legacy and GLMsingle maps are on different MNI grids. Compare scalar and
spatial summaries directly, but do not subtract voxelwise maps without an
explicit resampling plan. The reporter requires
`--allow-approximate-resampling` before creating such a difference and records
that approximation in provenance.

## Final output locations

| Output | Location |
|---|---|
| Corrected BIDS | `pilot_raw_pipeline_sub4_6/bids` (canonical) and local Docker staging tree |
| fMRIPrep derivatives | `pilot_raw_pipeline_sub4_6/fmriprep_sdc/sub-04`, `sub-05`, and `sub-06`; Sub5 branch reports/provenance are under `fmriprep_sdc/logs` |
| Quantitative fMRIPrep tSNR QC | `pilot_raw_pipeline_sub4_6/reports/fmriprep_quantitative_qc/Sub4`, `Sub5`, and `Sub6` |
| GLMsingle Type-D (20-value primary grid) | `pilot_raw_pipeline_sub4_6/glmsingle_sdc_defaultgrid/Sub4`, `Sub5`, `Sub6` |
| GLMsingle coarse-grid sensitivity | `pilot_raw_pipeline_sub4_6/glmsingle_sdc/Sub6` |
| Matched legacy decoding | `pilot_raw_pipeline_sub4_6/decoding_corrected_legacy/Sub4`, `Sub5`, `Sub6` |
| New decoding + float32 maps | `pilot_raw_pipeline_sub4_6/decoding_new_sdc/Sub4`, `Sub5`, `Sub6` |
| Side-by-side tables and group maps | the versioned directory under `pilot_raw_pipeline_sub4_6/reports` |

An output is complete only when its verification gate passes; the presence of
a directory or partial Docker work tree is not evidence of completion.
