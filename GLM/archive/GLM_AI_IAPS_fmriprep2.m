%%% 1st-Level Categorical Design for all sessions/subjects (SPM12)
clear; clc;

% ---- SPM init ----
spm('Defaults','fMRI');
spm_get_defaults('cmdline', true);

% ========== CONFIGURATION ==========
imgpath      = 'N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\rawfMRI\MyIAPS_BIDS\derivatives\fmriprep';
onsetpath    = 'N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\DataRecording\';
outputPath   = 'N:\Experimental_Data\yujunchen\projects\AI_IAPS\GLM\betas_fmriprep';

subjects     = { 'Sub5' };   % <- renamed to avoid conflict with MATLAB subs()
runs         = {'Run01','Run02','Run03','Run04','Run05','Run06','Run07','Run08','Run09','Run10'};
TR           = 1.8;
rnam         = {'X','Y','Z','x','y','z'};

% ---- explicitly set scans per run ----
% nscan_by_run = [224,224,224,224,224,224,224,224,224,224];
nscan_by_run = [224,218,224,224,224,218,218,218,218,218];

cond_names   = {'Pleasant','Neutral','Unpleasant','PleasantAI','NeutralAI','UnpleasantAI'};
ncond        = numel(cond_names);

% ===================================

for i = 1:numel(subjects)  % SUBJECT loop
    % fresh SPM state each subject
    spm('Defaults','fMRI');
    spm_get_defaults('cmdline', true);

    sub_num  = regexp(subjects{i}, '\d+', 'match', 'once');
    sub_bids = sprintf('sub-%s', sub_num);

    % Output folder
    outputDir = fullfile(outputPath, subjects{i});
    if ~exist(outputDir, 'dir'), mkdir(outputDir); end

    SPM   = [];                      % reset per subject
    files = cell(1, numel(runs));    % holds full paths to run NIfTIs

    for j = 1:numel(runs)  % RUN loop
        % ---- locate functional NIfTI ----
        if j <= 6
            ses = 1;
        else
            ses = 2;
        end
        ses_folder = fullfile(imgpath, sub_bids, sprintf('ses-%d', ses), 'func');
        run_label  = sprintf('%02d', j);

        nii_pattern = sprintf('%s_ses-%d_task-iaps_run-%s_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_sm8.nii', ...
                              sub_bids, ses, run_label);
        niifile = dir(fullfile(ses_folder, nii_pattern));

        if isempty(niifile)
            gzfile = dir(fullfile(ses_folder, [nii_pattern '.gz']));
            if ~isempty(gzfile)
                try
                    gunzip(fullfile(ses_folder, gzfile(1).name), ses_folder);
                catch ME
                    warning('gunzip failed for %s: %s', gzfile(1).name, ME.message);
                end
                niifile = dir(fullfile(ses_folder, nii_pattern));
            end
        end

        if isempty(niifile)
            error('Missing NIfTI for %s %s (expected %s)', subjects{i}, runs{j}, nii_pattern);
        end
        files{j} = fullfile(ses_folder, niifile(1).name);

        % ---- onsets for this run ----
        onsetDir = fullfile(onsetpath, subjects{i}, 'LogFiles');
        S = load(fullfile(onsetDir, ['processed_' runs{j} '.mat']));  % expects 'outputOnset'
        outputOnset = S.outputOnset;

        % six conditions
        Onset1 = cell2mat(outputOnset(:,1));
        Onset2 = cell2mat(outputOnset(:,2));
        Onset3 = cell2mat(outputOnset(:,3));
        Onset4 = cell2mat(outputOnset(:,4));
        Onset5 = cell2mat(outputOnset(:,5));
        Onset6 = cell2mat(outputOnset(:,6));
        B = {Onset1, Onset2, Onset3, Onset4, Onset5, Onset6};

        % durations (edit if not 3s blocks)
        Dur = ones(numel(Onset1), ncond) * 3;  % seconds
        D   = {Dur(:,1), Dur(:,2), Dur(:,3), Dur(:,4), Dur(:,5), Dur(:,6)};

        % build SPM.Sess(j)
        for c = 1:ncond
            SPM.Sess(j).U(c).name = {cond_names{c}};
            SPM.Sess(j).U(c).ons  = B{c};
            SPM.Sess(j).U(c).dur  = D{c};
            SPM.Sess(j).U(c).P(1).name = 'none';
        end

        % ---- motion regressors ----
        rp_file = sprintf('rp_%s_ses-%d_task-iaps_run-%s_bold.txt', sub_bids, ses, run_label);
        rp_path = fullfile(ses_folder, rp_file);
        if exist(rp_path, 'file')
            mov = load(rp_path);
            if size(mov,1) ~= nscan_by_run(j)
                warning('RP rows (%d) != nscan_by_run(%d) for %s %s', size(mov,1), nscan_by_run(j), subjects{i}, runs{j});
            end
            SPM.Sess(j).C.C    = mov;
            SPM.Sess(j).C.name = rnam;
        else
            warning('Motion regressor not found for %s %s, filling zeros', subjects{i}, runs{j});
            SPM.Sess(j).C.C    = zeros(nscan_by_run(j), 6);
            SPM.Sess(j).C.name = rnam;
        end

        % session-level/filter/variance
        SPM.xGX.iGXcalc     = 'Scaling';
        SPM.xX.K(j).HParam  = 128;     % seconds
        SPM.xVi.form        = 'AR(1)';
    end

    % ---- design (basis, timing) ----
    SPM.xBF.name     = 'hrf';    % Canonical HRF
    SPM.xBF.length   = 32;
    SPM.xBF.order    = 1;
    SPM.xBF.T        = 64;
    SPM.xBF.T0       = 32;
    SPM.xBF.UNITS    = 'secs';
    SPM.xBF.Volterra = 1;

    % scans & timing
    SPM.nscan  = nscan_by_run;   % explicit vector
    SPM.xY.P   = char(files(:)); % safe padding to longest filename
    SPM.xY.RT  = TR;

    % ---- tell SPM where to write (no cd) ----
    SPM.swd = outputDir;

    % ---- build & estimate ----
    SPM = spm_fmri_spm_ui(SPM);
    save(fullfile(outputDir,'SPM_design_pre_est.mat'), 'SPM', '-v7.3');  % optional for debugging

    SPM = spm_spm(SPM);  % writes SPM.mat, beta_*.nii, ResMS.nii, mask, etc.
    save(fullfile(outputDir,'SPM_estimated.mat'), 'SPM', '-v7.3');       % optional snapshot

    clear SPM S outputOnset Onset1 Onset2 Onset3 Onset4 Onset5 Onset6 B D files;
end

disp('All subjects finished!');
