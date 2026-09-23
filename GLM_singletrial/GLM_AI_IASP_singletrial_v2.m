%%% Improved 1st Level Single-Trial GLM with Condition Labeling
% For beta series correlation analysis with proper condition identification
% Key improvements:
% - Loads condition labels from CSV
% - No global scaling (prevents removing real differences)
% - Adjusted microtime parameters for multiband
% - Automatic scan detection
% - Saves beta mapping for easy extraction
%%%

clc;
clear;

%% ========== PATHS AND PARAMETERS ==========
cwd = 'N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI';
subs = {'Sub31'};  % Add more subjects as: {'Sub15', 'Sub16', 'Sub17', ...}
runs = {'Run01','Run02','Run03','Run04','Run05','Run06','Run07','Run08','Run09','Run10'};
nses = 10;

% Data paths
imgpath = 'N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\preprocessed_fMRIdata\dev-sess';
onsetpath = 'N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\DataRecording';
conditionFile = 'N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\stimuli_600trials.csv';
outputPath = 'N:\Experimental_Data\yujunchen\projects\AI_IAPS\GLM_singletrial\betas2';

% Acquisition parameters
TR = 1.80;
rnam = {'X','Y','Z','x','y','z'};  % Movement parameter names

%% ========== LOAD CONDITION MAPPING ==========
fprintf('Loading condition mapping from CSV...\n');
condTable = readtable(conditionFile);
fprintf('Loaded %d trials with condition labels\n', height(condTable));

% Check what columns we have
fprintf('Columns in condition file: ');
disp(condTable.Properties.VariableNames);

%% ========== MAIN SUBJECT LOOP ==========
for i = 1:numel(subs)
    fprintf('\n========================================\n');
    fprintf('Processing Subject: %s\n', subs{i});
    fprintf('========================================\n');
    
    subDir = fullfile(imgpath, subs{i});
    if ~exist(subDir, 'dir')
        error('Subject directory not found: %s', subDir);
    end
    cd(subDir);

    outputDir = fullfile(outputPath, subs{i});
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    % Detect actual number of scans and find max length for padding
    max_len = 0;
    actual_nscans = zeros(1, nses);
    fprintf('\nDetecting number of scans per run:\n');
    
    for j = 1:numel(runs)
        runDir = fullfile(subDir, runs{j});
        files_j = spm_select('fplist', runDir, '^swar.*\.nii');
        max_len = max(max_len, size(files_j, 2));
        actual_nscans(j) = size(files_j, 1);
        fprintf('  %s: %d volumes\n', runs{j}, actual_nscans(j));
    end
    
    % Initialize beta mapping structure
    betaMapping = [];
    betaIdx = 1;
    
    %% ========== RUN/SESSION LOOP ==========
    for j = 1:numel(runs)
        fprintf('\n--- Processing %s ---\n', runs{j});
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        % Basis functions and timing parameters %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        SPM.xBF.name = 'hrf';
        SPM.xBF.length = 32.0513;
        SPM.xBF.order = 1;
        
        % IMPROVED: Adjusted for multiband acquisition
        % With multiband factor 2 and 64 slices, you have 32 unique slice timings
        SPM.xBF.T = 32;     % Changed from 64 to 32 for multiband=2
        SPM.xBF.T0 = 16;    % Changed from 32 to 16 (middle slice)
        
        SPM.xBF.UNITS = 'secs';
        SPM.xBF.Volterra = 1;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Trial Specification: fMRI data     %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        runDir = fullfile(subDir, runs{j});
        files_j = spm_select('fplist', runDir, '^swar.*\.nii');
        
        if isempty(files_j)
            error('No fMRI files found in %s', runDir);
        end
        
        % Pad files to max length
        pad_length = max_len - size(files_j, 2);
        padded_files = [files_j, repmat(' ', size(files_j, 1), pad_length)];
        tmp{j} = padded_files;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Load Onset Information              %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        onsetDir = fullfile(onsetpath, subs{i}, 'LogFiles');
        onset_file = fullfile(onsetDir, sprintf('spm_singletrial_%s.mat', runs{j}));
        
        if ~exist(onset_file, 'file')
            error('Onset file not found: %s', onset_file);
        end
        
        onset_data = load(onset_file);
        onset = struct2cell(onset_data);
        onset = table2cell(onset{1});
        
        fprintf('  Found %d trials in onset file\n', size(onset, 1));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Match Conditions from CSV          %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Get trials for this run from condition table
        runTrials = condTable(condTable.run == j, :);
        
        if height(runTrials) ~= size(onset, 1)
            warning('Trial count mismatch: %d in onset, %d in condition file', ...
                    size(onset, 1), height(runTrials));
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Create Design Matrix               %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ncon = size(onset, 1);  % Number of conditions/trials
        conditionCell = cell(1, ncon);
        
        for k = 1:ncon
            % Get onset time from onset file
            onsets{1, k} = onset{k, 2};
            
            % Get image name (for matching if needed)
            imgName = onset{k, 1};
            if contains(imgName, '.')
                [~, imgName, ~] = fileparts(imgName);
            end
            
            % Find corresponding condition from CSV
            % The OriginalOrder in CSV is 0-based across all runs
            trialGlobalIdx = (j-1) * 60 + (k-1);  % 0-599
            
            % Find matching row
            matchIdx = find(runTrials.OriginalOrder == trialGlobalIdx);
            
            if ~isempty(matchIdx)
                condLabel = runTrials.group{matchIdx(1)};
                % Create regressor name with condition and trial number
                conditionCell{k} = sprintf('%s_run%02d_trial%02d', condLabel, j, k);
                
                % Save to beta mapping
                betaMapping(betaIdx).subject = subs{i};
                betaMapping(betaIdx).run = j;
                betaMapping(betaIdx).trial = k;
                betaMapping(betaIdx).condition = condLabel;
                betaMapping(betaIdx).onset = onsets{1, k};
                betaMapping(betaIdx).image = imgName;
                betaMapping(betaIdx).beta_file = sprintf('beta_%04d.nii', betaIdx);
                betaMapping(betaIdx).regressor_name = conditionCell{k};
                betaIdx = betaIdx + 1;
            else
                % Fallback if no match found
                warning('No condition match for run %d trial %d (global idx %d)', ...
                        j, k, trialGlobalIdx);
                conditionCell{k} = sprintf('unknown_run%02d_trial%02d', j, k);
            end
        end

        % Set up regressors
        B = onsets;  % Onset times
        D = num2cell(ones(1, ncon) * 3);  % Duration = 3 seconds per trial

        for c = 1:ncon
            SPM.Sess(j).U(c).name = {conditionCell{1, c}};
            SPM.Sess(j).U(c).ons = B{1, c};
            SPM.Sess(j).U(c).dur = D{1, c};
            SPM.Sess(j).U(c).P(1).name = 'none';
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Movement Parameters                %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        currDir = fullfile(subDir, runs{j});
        cd(currDir);
        
        % Find movement parameter file
        fn = spm_select('list', currDir, '^rp.*\.txt');
        if isempty(fn)
            error('Movement parameter file not found in %s', currDir);
        end
        
        [r1, r2, r3, r4, r5, r6] = textread(fn, '%f%f%f%f%f%f');
        SPM.Sess(j).C.C = [r1 r2 r3 r4 r5 r6];
        SPM.Sess(j).C.name = rnam;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % GLM Settings                       %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % CRITICAL CHANGE: No global scaling to preserve condition differences
        SPM.xGX.iGXcalc = 'None';  % Changed from 'Scaling' to 'None'
        
        SPM.xX.K(j).HParam = 128;  % High-pass filter cutoff
        SPM.xVi.form = 'AR(1)';     % Serial correlation correction
        
        clear fn r1 r2 r3 r4 r5 r6
    end

    %% ========== MODEL SPECIFICATION & ESTIMATION ==========
    % Use actual detected scan numbers
    SPM.nscan = actual_nscans;
    fprintf('\nUsing scan numbers: [%s]\n', num2str(actual_nscans));
    
    SPM.xY.P = cat(1, tmp{:});
    SPM.xY.RT = TR;
    
    % Run SPM model
    fprintf('\nSpecifying and estimating model...\n');
    cd(outputDir);
    SPM = spm_fmri_spm_ui(SPM);
    SPM = spm_spm(SPM);
    
    %% ========== SAVE BETA MAPPING ==========
    % Save mapping for easy beta extraction
    save(fullfile(outputDir, 'beta_mapping.mat'), 'betaMapping');
    fprintf('\nSaved beta_mapping.mat with %d entries\n', length(betaMapping));
    
    % Create summary text file
    fid = fopen(fullfile(outputDir, 'beta_mapping_summary.txt'), 'w');
    fprintf(fid, 'Beta Mapping Summary for %s\n', subs{i});
    fprintf(fid, '=====================================\n\n');
    
    % Count trials per condition
    conditions = unique({betaMapping.condition});
    fprintf(fid, 'Conditions found:\n');
    fprintf('\nCondition Summary:\n');
    
    for c = 1:length(conditions)
        count = sum(strcmp({betaMapping.condition}, conditions{c}));
        fprintf(fid, '  %s: %d trials\n', conditions{c}, count);
        fprintf('  %s: %d trials\n', conditions{c}, count);
    end
    
    % Write first few entries as examples
    fprintf(fid, '\nFirst 10 beta mappings:\n');
    fprintf(fid, 'Beta#  Run  Trial  Condition      Onset    Image\n');
    fprintf(fid, '-----  ---  -----  ------------  -------  ----------\n');
    
    for b = 1:min(10, length(betaMapping))
        fprintf(fid, '%04d   %2d    %2d    %-12s  %6.2f   %s\n', ...
                b, betaMapping(b).run, betaMapping(b).trial, ...
                betaMapping(b).condition, betaMapping(b).onset, ...
                betaMapping(b).image);
    end
    
    fclose(fid);
    
    % Clear large variables
    clear SPM tmp;
end

%% ========== FINAL SUMMARY ==========
fprintf('\n========================================\n');
fprintf('GLM COMPLETE\n');
fprintf('========================================\n');
fprintf('Output directory: %s\n', outputPath);
fprintf('\nKey improvements implemented:\n');
fprintf('  1. Condition labels added to each trial\n');
fprintf('  2. Global scaling disabled (preserves condition differences)\n');
fprintf('  3. Microtime parameters adjusted for multiband acquisition\n');
fprintf('  4. Automatic scan detection\n');
fprintf('  5. Beta mapping saved for easy extraction\n');
fprintf('\nNext steps:\n');
fprintf('  1. Check beta_mapping_summary.txt for condition counts\n');
fprintf('  2. Use beta_mapping.mat to extract betas by condition\n');
fprintf('  3. Run second-level analysis or decoding\n');