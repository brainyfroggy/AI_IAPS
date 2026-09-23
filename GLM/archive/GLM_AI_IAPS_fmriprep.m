%%% 1st Level Categorical Design for all sessions/subjects
clear;
spm('defaults','FMRI')

global defaults;

% ========== CONFIGURATION ==========
imgpath   = 'N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\rawfMRI\MyIAPS_BIDS\derivatives\fmriprep';
onsetpath = 'N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\DataRecording\';
outputPath = 'N:\Experimental_Data\yujunchen\projects\AI_IAPS\GLM\betas_fmriprep';

subs = { 'Sub2'  };
% subs = { 'Sub7' , 'Sub8', 'Sub9', 'Sub11' 'Sub12'};
% subs = {'Sub22','Sub23','Sub24'};
runs = {'Run01','Run02','Run03','Run04','Run05','Run06','Run07','Run08','Run09','Run10'};
% runs = {'Run01','Run02','Run03','Run04','Run05','Run06'};
TR = 1.8;
rnam = {'X','Y','Z','x','y','z'};
nscan_per_run = 224;  % Confirmed from fMRIPrep output
% nscan_by_run = [218,218,218,218,218,218,218,218,218,218];
nscan_by_run = [224,224,224,224,224,224,224,224,224,224];

for i = 1:numel(subs) %SUBJECT
    % Get subject BIDS label, e.g. 'sub-14'
    sub_num = regexp(subs{i}, '\d+', 'match', 'once');
    sub_bids = sprintf('sub-%s', sub_num);

    % Output folder
    outputDir = fullfile(outputPath, subs{i});
    if ~exist(outputDir, 'dir'), mkdir(outputDir); end

    % Preallocate file arrays
    tmp = cell(1, numel(runs));
    SPM = [];  % Reset SPM for each subject

    for j = 1:numel(runs)
        % ----- Locate functional NIfTI -----
        if j <= 6
            ses = 1;
            ses_folder = fullfile(imgpath, sub_bids, 'ses-1', 'func');
        else
            ses = 2;
            ses_folder = fullfile(imgpath, sub_bids, 'ses-2', 'func');
        end
        run_num   = j;
        run_label = sprintf('%02d', run_num);
        
        % Prefer smoothed NLin6Asym .nii; if only .nii.gz exists, unzip it once.
        nii_pattern = sprintf('%s_ses-%d_task-iaps_run-%s_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_sm8.nii', ...
                              sub_bids, ses, run_label);
        niifile = dir(fullfile(ses_folder, nii_pattern));
        
        if isempty(niifile)
            gz_pattern = [nii_pattern '.gz'];
            gzfile = dir(fullfile(ses_folder, gz_pattern));
            if ~isempty(gzfile)
                try
                    gunzip(fullfile(ses_folder, gzfile(1).name), ses_folder);  % write .nii next to it
                catch ME
                    warning('gunzip failed for %s: %s', gzfile(1).name, ME.message);
                end
                niifile = dir(fullfile(ses_folder, nii_pattern));  % re-check for the .nii
            end
        end
        
        if isempty(niifile)
            warning('Missing NIfTI for %s %s', subs{i}, runs{j});
            tmp{j} = repmat(' ', 1, 150);
            continue
        end
        
        tmp{j}  = repmat(' ', 1, 150);
        thisfile = fullfile(ses_folder, niifile(1).name);
        tmp{j}(1:length(thisfile)) = thisfile;
        % ----- Load onsets -----
        onsetDir = fullfile(onsetpath, subs{i}, 'LogFiles');
        load(fullfile(onsetDir, ['processed_' runs{j} '.mat'])); % outputOnset
        Onset1=cell2mat(outputOnset(:,1));
        Onset2=cell2mat(outputOnset(:,2));
        Onset3=cell2mat(outputOnset(:,3));
        Onset4=cell2mat(outputOnset(:,4));
        Onset5=cell2mat(outputOnset(:,5));
        Onset6=cell2mat(outputOnset(:,6));
        B={Onset1,Onset2,Onset3,Onset4,Onset5,Onset6};
        Dur = ones(length(Onset1),6)*3; % change if not 10 trials per run
        D={Dur(:,1),Dur(:,2),Dur(:,3),Dur(:,4),Dur(:,5),Dur(:,6)};
        condition_nameCell = {'Pleasant','Neutral','Unpleasant', 'PleasantAI','NeutralAI','UnpleasantAI'};
        ncon=6;

        for c = 1:ncon
            SPM.Sess(j).U(c).name = {condition_nameCell{c}};
            SPM.Sess(j).U(c).ons = B{c};
            SPM.Sess(j).U(c).dur = D{c};
            SPM.Sess(j).U(c).P(1).name = 'none';
        end

        % ----- Include movement regressors -----
        rp_file = sprintf('rp_%s_ses-%d_task-iaps_run-%s_bold.txt', sub_bids, ses, run_label);
        rp_path = fullfile(ses_folder, rp_file);
        if exist(rp_path, 'file')
            mov = load(rp_path);
            SPM.Sess(j).C.C = mov;
            SPM.Sess(j).C.name = rnam;
        else
            warning('Motion regressor not found for %s %s', subs{i}, runs{j});
            SPM.Sess(j).C.C = zeros(nscan_per_run, 6); % or leave empty
            SPM.Sess(j).C.name = rnam;
        end

        SPM.xGX.iGXcalc = 'Scaling';
        SPM.xX.K(j).HParam = 128;
        SPM.xVi.form = 'AR(1)';
    end

    % Pad all NIfTI paths
    max_cols = max(cellfun(@(x) size(x,2), tmp));
    for j = 1:numel(runs)
        if size(tmp{j},2) < max_cols
            tmp{j} = [tmp{j} repmat(' ', size(tmp{j},1), max_cols - size(tmp{j},2))];
        end
    end

    % ========== Specify HRF to avoid GUI ==========
    SPM.xBF.name = 'hrf';        % Canonical HRF
    SPM.xBF.length = 32;         % or 32.0513, both are fine
    SPM.xBF.order = 1;
    SPM.xBF.T = 64;
    SPM.xBF.T0 = 32;
    SPM.xBF.UNITS = 'secs';
    SPM.xBF.Volterra = 1;

    SPM.nscan = nscan_by_run;
    SPM.xY.P = char(tmp{:});
    SPM.xY.RT = TR;
    SPM = spm_fmri_spm_ui(SPM);
    cd(outputDir);
    SPM = spm_spm(SPM);
    clear SPM;
end
disp('All subjects finished!');
