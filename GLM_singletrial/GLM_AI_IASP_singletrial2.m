spm%%% 1st Level Categorical Design with GLM modeling each trial in one run
% Code developed for subsequent beta series correlation analysis
% Code runs for a single subject 
%%%
clc;
clear;

cwd = 'F:\yujun\projects\LAB_IAPS_AI';
% subs = {'Sub5'};
subs = {'Sub11'};
runs = {'Run01','Run02','Run03','Run04','Run05','Run06','Run07','Run08','Run09','Run10'};
nses = 10;

imgpath = 'F:\yujun\projects\LAB_IAPS_AI\preprocessed_fMRIdata\dev-sess';   % Working directory
onsetpath = 'F:\yujun\projects\LAB_IAPS_AI\DataRecording';% onset directory

imgdir = dir(imgpath);
onsetdir = dir(onsetpath);

TR = 1.80;
outputPath = 'F:\yujun\projects\AI_IAPS\GLM_singletrial\betas';
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
%             conditionCell{k} = sprintf('trial_%d', k);
            conditionCell{k} = sprintf('trial_%d_run_%d', k, j);
        end

        % onsets
        B = onsets;

        % duration
        D = num2cell(ones(1,60)*3);

        % 
%         SPM.nscan = ones(1,nses)*218;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Trial Specification: fMRI data     %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

       % Get the last relevant scan based on the onset data
%         last_relevant_scan = ceil((max(cell2mat(onset(:, 2))) + 3) / 1.8)+1; % Assuming the onset data is in the second column

        % load preprocessed data
        dir{j} = strcat(subDir,'\',runs{j});
%         tmp{j} = repmat(' ', 1, 105);
%         tmp{j} = spm_select('fplist',dir{j},'^swar.*\.nii'); 
        % Get the filenames
        files = spm_select('fplist', dir{j}, '^swar.*\.nii');
%         num_relevant_scan = max(min(last_relevant_scan, size(files, 1)), 207);
        num_relevant_scan = 207;
        files = files(2:num_relevant_scan, :);
%         num_relevant_scans(j) = num_relevant_scan
        % Determine padding length
        pad_length = max(0, 150 - size(files, 2));
        % Pad filenames with trailing spaces
        padded_files = [files, repmat(' ', size(files, 1), pad_length)];
        % Assign to tmp
        tmp{j} = padded_files;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Trial Specification: Design Matrix %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
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
        SPM.Sess(j).C.C = [r1(2:207) r2(2:207) r3(2:207) r4(2:207) r5(2:207) r6(2:207)];
        SPM.Sess(j).C.name = rnam;

        SPM.xGX.iGXcalc = 'Scaling';     % Global Normalization: OPTIONS: 'Scaling'
        SPM.xX.K(j).HParam = 128;     % HPF cutoff 128 s
        SPM.xVi.form = 'AR(1)';    % Adjustment for Intrinsic serial correlation: OPTIONS: 'none'|'AR(1)+w'
        clear fn r1 r2 r3 r4 r5 r6
%         clear outputOnset B D;
    end

%     SPM.nscan = ones(1,nses)*224;
    SPM.nscan = [218,218,218,218,218,218,218,218,218,218];
%     SPM.nscan = [num_relevant_scans(1),
%         num_relevant_scans(2),
%         num_relevant_scans(3),
%         num_relevant_scans(4),
%         num_relevant_scans(5),
%         num_relevant_scans(6),
%         num_relevant_scans(7),
%         num_relevant_scans(8),
%         num_relevant_scans(9),
%         num_relevant_scans(10)];
    SPM.xY.P = cat(1,tmp{:});
    SPM.xY.RT = TR;
    SPM = spm_fmri_spm_ui(SPM);
    cd(outputDir);
    SPM = spm_spm(SPM);
    clear SPM ;
end
