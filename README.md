# AI_IAPS

## Purpose

This repository holds the analysis code for a multi-modal fMRI study comparing
brain responses to natural (IAPS) versus AI-generated pleasant / unpleasant /
neutral images, in a cohort of up to 30 subjects. The experimental design
crosses two image sources (natural IAPS photographs vs. AI-generated images)
with three emotional valence categories (pleasant, unpleasant, neutral), and
asks whether — and how — multivariate fMRI patterns evoked by AI-generated
emotional images resemble those evoked by real photographs of the same
valence.

The code here covers the full analysis chain used to answer that question:

- First- and single-trial-level GLM modeling (SPM12 and GLMsingle) to turn
  preprocessed BOLD timeseries into per-condition and per-trial beta estimates.
- Multivariate pattern decoding (within- and cross-image-source, ROI-based and
  whole-brain searchlight) to test whether a classifier trained on one image
  source generalizes to the other.
- Representational similarity analysis (RSA) comparing representational
  geometry across sources and valence categories.
- ROI/atlas overlap analysis of group-level univariate contrasts against
  standard brain atlases.
- Eye-tracking (pupillometry) analysis of physiological arousal responses
  during scanning.
- Behavioral analysis of subjects' own valence/arousal ratings of the stimuli.

**This repository contains only analysis code and documentation.** The
underlying data — raw and preprocessed neuroimaging data, single-trial beta
maps, eye-tracking recordings, stimulus images, behavioral rating
spreadsheets, and all intermediate/derived numerical results (~1.14 TB across
roughly 114,000 files) — is **not included** and is not suitable for public
hosting. It lives on institutional storage (University of Florida / Ding Lab
network storage) and is available to collaborators on request. Paths
referenced inside scripts and docs (e.g. `N:\Experimental_Data\...`,
`F:\yujun\projects\...`) point to that storage and will not resolve outside
the lab environment — see **How to Use** below for how to supply your own
data in the layout the code expects.

This code documents an active, multi-stage research pipeline (pilot cohort →
full cohort; GLM → GLMsingle → decoding → searchlight → ROI tables) rather
than a single, one-shot script. Several directories represent different,
still-valid stages of that pipeline (e.g. an earlier, smaller-cohort pilot run
alongside a later full-cohort run) rather than superseded/broken duplicates.

## Contents

```
AI_IAPS/
├── new_pipeline/        Current/active analysis pipeline (GLMsingle-centric)
├── GLM/                 First-level & second-level SPM GLM modeling (notebooks + .m)
├── GLM_singletrial/     Single-trial ("LS-A" style) SPM GLM for MVPA betas
├── Decoding/            MVPA decoding notebooks/scripts, incl. full raw-data
│                        preprocessing pipelines and searchlight analyses
├── ROI table/           ROI/atlas overlap analysis for univariate contrasts
├── Eyetracking/         Pupillometry preprocessing and analysis (MATLAB)
├── RDM/                 Representational dissimilarity matrix comparison
└── behavior/            Behavioral image-rating analysis
```

| Directory | Contents |
|---|---|
| `new_pipeline/` | Current, most actively maintained pipeline. `docs/` and top-level `*.md` files (`DESIGN_FREEZE.md`, `FINDINGS_AND_REPRODUCTION.md`, `COMPLETION_REPORT.md`, `WHOLEBRAIN_SEARCHLIGHT_TASK.md`) document design decisions and findings — `FINDINGS_AND_REPRODUCTION.md` in particular walks through why GLMsingle initially underperformed SPM and how the production GLMsingle configuration (fixed/canonical HRF + GLMdenoise + fracridge ridge regression) was arrived at. `code/` holds the pipeline scripts, organized by stage: GLM fitting (`run_glmsingle_*.py`, `categorical_glm_*.py`), ROI decoding (`decode_roi_*.py` + matching `aggregate_roi_decoding_*.py` + `*_wave_launcher.py` job dispatchers), permutation-based significance testing (`erp_permute_worker.py`, `stelzer_*.py`, `permutation_null_check.py`, Stelzer et al. 2013-style with Benjamini-Hochberg FDR), AAL3 whole-brain/cross-source analysis (`aal3_*.py`), univariate/group contrasts, diagnostics/plotting, and subject-onboarding utilities (`onboard_sub33.py`, etc.). (`config/` and `manifests/` hold per-run data manifests, not code, and are not part of the pipeline logic.) |
| `GLM/` | MATLAB/SPM first-level & second-level GLM: conventional run-wise models (`FirstLevel_AI_IAPS*.m`, `GLM_AI_IAPS*.m`) plus notebooks walking through activation maps, contrast maps, and single-trial beta activation maps (`Step1`–`Step3`). |
| `GLM_singletrial/` | Single-trial ("least-squares-all") SPM GLM used to generate the per-trial beta maps consumed by the decoding analyses, plus notebooks for categorizing/visualizing those beta files. (Beta NIfTI outputs themselves are data, not code, and are excluded.) |
| `Decoding/` | The bulk of the MVPA work. Top-level notebooks/scripts cover decoding pilots and iterations (`decoding_*.ipynb`), cross-source accuracy comparisons, searchlight cross-decoding, and a Kastner-atlas single-trial feature-selection decoding script. `archive/` holds earlier iterations of these notebooks kept for reference/provenance. Four subfolders are self-contained raw-data-to-decoding pipelines, independent of `new_pipeline/`: `full_cohort_raw_pipeline_28/` (production pipeline for the full 28+ subject cohort: BIDS conversion, fMRIPrep-based SDC preprocessing, GLMsingle, post-processing gates), `pilot_raw_pipeline_sub4_6/` (the same raw pipeline validated on the smaller subjects 4–6 pilot cohort — a complete, still-valid earlier stage, not a discarded draft), `sub4_spatial_sensitivity_32/` (native-space vs. MNI-space control analysis for subject 4), and `sub4_wholebrain_aal3_searchlight/` (whole-brain AAL3 searchlight decoding for subject 4). |
| `ROI table/` | Scripts computing overlap between statistical-map ROIs (from group-level univariate contrasts) and reference brain atlases (HCP-MMP1, Schaefer-400, AAL3, Kastner/Wang, Harvard-Oxford), producing ROI overlap/summary tables. `atlases/README.md` documents the atlas cache folder convention; atlas data files and the vendored AAL SPM12 toolbox are excluded (see Dependencies). |
| `Eyetracking/` | MATLAB scripts for pupillometry preprocessing/analysis: epoching pupil size around stimulus onset (`pupil_epoch_*.m`), timecourse extraction (`pupil_timecourse.m`), single-subject fixation/pupil QC checks (`test_fixation.m`, `test_pupil.m`), and a combined summary notebook (`main.ipynb`). Per-subject `.mat`/`.edf` files are excluded. |
| `RDM/` | `rdm_comparison.ipynb` — representational dissimilarity matrix comparison across conditions/sources. |
| `behavior/` | `behavioral.ipynb` — analysis of behavioral valence/arousal ratings collected alongside the fMRI sessions. Underlying rating spreadsheets are excluded. |

## How to Use

This is a research code export, not a plug-and-play package — it is shared so
the analysis methodology is transparent and reproducible, not so it can be
run end-to-end without modification. Several scripts contain absolute lab
paths and subject-specific onboarding logic that will need to be adjusted for
a new environment. That said, the overall pipeline flow is:

```
 raw fMRI / eye-tracking / behavioral data (not included, supplied separately)
        │
        ▼
 GLM/ , GLM_singletrial/  (SPM12 first-level + single-trial GLM)
   or new_pipeline/code/run_glmsingle_*.py  (GLMsingle single-trial GLM)
        │  produces per-condition / per-trial beta (NIfTI) maps
        ▼
 Decoding/ , new_pipeline/code/decode_roi_*.py , RDM/
   (MVPA decoding, searchlight, RSA — consume the beta maps above)
        │  produces per-subject/group decoding accuracies, RSA matrices
        ▼
 ROI table/   (aggregates statistical-map results against atlases)

 Eyetracking/  and  behavior/  run independently, over their own raw
   pupillometry / rating data, and are not inputs to the fMRI steps above.
```

1. **Supply the data.** The scripts expect data under per-subject folders
   named `SubN` (e.g. `Sub1`, `Sub11`), generally organized as:
   - Raw functional/anatomical NIfTIs and behavioral log files under a
     `DataRecording/SubN/` tree (see e.g. paths referenced in
     `GLM/Step3_singletrial_GLM.ipynb`, `Eyetracking/test_pupil.m`), with
     per-run onset/log files under a `LogFiles/` subfolder and eye-tracking
     `.edf` recordings under an `Eyetracking/` subfolder.
   - The `full_cohort_raw_pipeline_28/` and `pilot_raw_pipeline_sub4_6/`
     sub-pipelines instead expect (and will build) a standard **BIDS**
     dataset (`dataset_description.json`, `sub-XX/...`) — see each
     sub-pipeline's own `RUNBOOK.md` / `PIPELINE_RUNBOOK.md` for the exact
     expected/produced directory tree and subject-exclusion rules.
   - Single-trial beta maps produced by `GLM_singletrial/` are expected under
     `GLM_singletrial/betas/SubN/beta_XXXXX.nii` by the decoding scripts that
     consume them (e.g. `Decoding/misc.ipynb`, `Decoding/decoding_erp_rerun.ipynb`).
   - A stimulus/trial metadata CSV (`stimuli_600trials.csv`, columns
     including trial group/condition labels) is read by most decoding
     notebooks to align trials with condition labels.
   - Reference atlas files (`.nii.gz`/`.xml`) go under `ROI table/atlases/`;
     the vendored AAL SPM12 toolbox goes alongside your SPM12 install (see
     Dependencies).
2. **Run the GLM stage** (`GLM/`, `GLM_singletrial/`, or
   `new_pipeline/code/run_glmsingle_wave.py` and related scripts) to fit
   first-level and/or single-trial models and produce beta maps.
3. **Run the decoding/RSA stage** (`Decoding/`, `new_pipeline/code/decode_roi_*.py`,
   `RDM/rdm_comparison.ipynb`) against those beta maps to get within- and
   cross-source decoding accuracies, searchlight maps, and RDMs.
4. **Run `ROI table/`** scripts to summarize group-level statistical maps by
   atlas region.
5. **Eyetracking (`Eyetracking/`) and behavior (`behavior/`) analyses** are
   independent of the steps above and only need their own raw pupillometry
   recordings / rating spreadsheets.

Each sub-pipeline directory that has one documents its own setup and stage
gates in more detail in its own `RUNBOOK.md`/`PIPELINE_RUNBOOK.md`,
`docs/`, or top-level `*.md` files — consult those before attempting to run
a specific stage.

## Dependencies

**Python** (standard scientific/neuroimaging stack seen across the repo):
`numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` (`sklearn`),
`statsmodels`, `nibabel`, `nilearn`, `h5py`, `joblib`, `tqdm`, `natsort`,
`threadpoolctl`, `mlxtend`, `openpyxl`, `python-pptx` (`pptx`), `Pillow` (`PIL`),
`IPython`, `pybids` (`bids`), `rsatoolbox` (RDM/RSA analysis), and
`sdcflows`/`fmriprep` for the raw-data preprocessing pipelines under
`Decoding/full_cohort_raw_pipeline_28/` and `Decoding/pilot_raw_pipeline_sub4_6/`
(fMRIPrep itself is invoked as an external/Dockerized tool, not a Python
import). No `requirements.txt`/environment file is included; install these
with `pip`/`conda` as needed for the stage you're running.

**MATLAB / external toolboxes** (not vendored in this repo):
- **[GLMsingle](https://github.com/cvnlab/GLMsingle)** (Python package
  `glmsingle`) — single-trial GLM estimation with per-voxel HRF fitting,
  GLMdenoise, and ridge-regularized (fracridge) beta estimation. Install
  separately per its own documentation.
- **SPM12** — used throughout `GLM/`, `GLM_singletrial/`, and several
  `Decoding/` pipelines for first-level/single-trial modeling.
- **AAL SPM12 toolbox** — third-party Automated Anatomical Labeling atlas
  toolbox for SPM12, used as a reference atlas in the ROI overlap analysis
  (`ROI table/`). Not authored by this project; obtain it from its original
  distribution and place it alongside your SPM12 toolbox directory.
- **EyeLink `edfmex`** — MATLAB MEX interface for reading SR Research EyeLink
  `.edf` eye-tracking recordings, used by the `Eyetracking/` scripts.

## Note on scope

This is a code export intended to document and preserve the analysis
methodology. Some scripts reference absolute paths on lab storage or contain
subject-specific onboarding logic; they are shared for transparency and
reproducibility of method rather than as a plug-and-play package.
