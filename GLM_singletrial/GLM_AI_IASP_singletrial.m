%%% 1st Level Categorical Design with GLM modeling each trial in one run
% Code developed for subsequent beta series correlation analysis
% Code runs for a single subject 
%%%
clc;
clear;

cwd = 'N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI';
% subs = {'Sub5'};
% subs = {'Sub15', 'Sub16','Sub17', 'Sub18', 'Sub19', 'Sub20', 'Sub21', 'Sub22', 'Sub23', 'Sub24','Sub25', 'Sub26', 'Sub27', 'Sub28', 'Sub29', 'Sub30', 'Sub31'};
subs = { 'Sub17'};
runs = {'Run01','Run02','Run03','Run04','Run05','Run06','Run07','Run08','Run09','Run10'};
nses = 10;
% runs = {'Run10'};
% nses = 1;

imgpath = 'N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\preprocessed_fMRIdata\dev-sess';   % Working directory
onsetpath = 'N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\DataRecording';% onset directory

imgdir = dir(imgpath);
onsetdir = dir(onsetpath);

TR = 1.80;
outputPath = 'N:\Experimental_Data\yujunchen\projects\AI_IAPS\GLM_singletrial\betas';
rnam = {'X','Y','Z','x','y','z'}; 


for i = 1:numel(subs) %SUBJECT
    subDir = [imgpath, '\', subs{i}];
    if ~exist(subDir, 'dir')
        mkdir(subDir);
    end
    cd(subDir);

    outputDir = [outputPath, '\', subs{i}];
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    max_len = 0;
    for j = 1:numel(runs)
        runDir = fullfile(subDir, runs{j});
        files_j = spm_select('fplist', runDir, '^swar.*\.nii');
        max_len = max(max_len, size(files_j, 2));
    end
    
    for  j = 1:numel(runs) %SESSION, OR RUN
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        % Basis functions and timing parameters %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        SPM.xBF.name = 'hrf';       % name of basis function
        SPM.xBF.length = 32.0513;   % length in seconds of basis
        SPM.xBF.order = 1;          % order of basis set
        SPM.xBF.T = 64;             % number of subdivisions of TR
        SPM.xBF.T0 = 32;            % first time bin (see slice timing)
        SPM.xBF.UNITS = 'secs';    % options: 'scans'|'secs' for onsets
        SPM.xBF.Volterra = 1;       % order of convolution

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Trial Specification: fMRI data     %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % load preprocessed data
        runDir = fullfile(subDir, runs{j});
        % get this run’s files
        files_j = spm_select('fplist', runDir, '^swar.*\.nii');
        % pad them out to max_len
        pad_length   = max_len - size(files_j,2);
        padded_files = [files_j, repmat(' ', size(files_j,1), pad_length)];
        tmp{j}       = padded_files;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Trial Specification: Design Matrix %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Load the .mat file for stimulus onset timings
        onsetDir = [onsetpath, '\', subs{i} ,'\LogFiles'];
        cd(onsetDir);
        onset_data = load([onsetDir,'\','spm_singletrial_', runs{j},'.mat']);
        onset = struct2cell(onset_data);
        onset = table2cell(onset{1});

        % stimuli onset
        ncon=numel(onset(:,1));
        conditionCell = cell(1,ncon );
        for k = 1:numel(onset(:,1))
%             onsets{1,k} = onset(k,1); % stimuli name
            onsets{1,k} = onset{k,2};
            conditionCell{k} = onset{k,1};
        end

        % onsets
        B = onsets;

        % duration
        D = num2cell(ones(1,60)*3);

        % 
%         SPM.nscan = ones(1,nses)*218;

        %
        for c = 1:ncon
            SPM.Sess(j).U(c).name = {conditionCell{1,c}};    
            SPM.Sess(j).U(c).ons = B{1,c};
            SPM.Sess(j).U(c).dur = D{1,c};
            SPM.Sess(j).U(c).P(1).name = 'none'; 
        end

        % Include movement parameters
        currDir = strcat(subDir,'\',runs{j});
        cd(currDir);
        fn = spm_select('list',currDir,'^rp.*\.txt');
        [r1,r2,r3,r4,r5,r6] = textread(fn,'%f%f%f%f%f%f');
        SPM.Sess(j).C.C = [r1 r2 r3 r4 r5 r6];
        SPM.Sess(j).C.name = rnam;

        SPM.xGX.iGXcalc = 'Scaling';     % Global Normalization: OPTIONS: 'Scaling'
        SPM.xX.K(j).HParam = 128;     % HPF cutoff 128 s
        SPM.xVi.form = 'AR(1)';    % Adjustment for Intrinsic serial correlation: OPTIONS: 'none'|'AR(1)+w'
        clear fn r1 r2 r3 r4 r5 r6
%         clear outputOnset B D;
    end

%     SPM.nscan = ones(1,nses)*224;
    SPM.nscan = [218,218,218,218,218,218,218,218,218,218];
    SPM.xY.P = cat(1,tmp{:});
    SPM.xY.RT = TR;
    SPM = spm_fmri_spm_ui(SPM);
    cd(outputDir);
    SPM = spm_spm(SPM);
    clear SPM ;
end
