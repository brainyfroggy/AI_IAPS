%%% 1st Level Categorical Design for all sessions/subjects
clear;
spm('defaults','FMRI')

global defaults;

imgpath = 'N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\preprocessed_fMRIdata\dev-sess';   % Working directory
onsetpath = 'N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\DataRecording\';% onset directory
outputPath = 'F:\yujun\projects\AI_IAPS\GLM\betas';

imgdir = dir(imgpath);
onsetdir = dir(onsetpath);

subs = {'Sub13'};
runs = {'Run01','Run02','Run03','Run04','Run05','Run06'};

nses = 6;   
% runs = {'Run01'};
% nscan = [224];
% nses = 1;    

% subs = {'Sub4_T1ref'};
% subs_onset = {'Sub4'};
% runs = {'Run01'};
% nscan = [224];
% nses = 1;   % Total number of sessions/runs to be modeled, 1 in this case.
                   
TR = 1.8;
rnam = {'X','Y','Z','x','y','z'}; 

for i = 1:numel(subs) %SUBJECT
    
    subDir = [imgpath, '\', subs{i}];
    if ~exist(subDir, 'dir')
        mkdir(subDir);
    end
    cd(subDir);
    
    outputDir = [outputPath, '\', [subs{i} '_s1']];
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

        %  SPM.nscan = ones(1,nses)*224;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Trial Specification: Design Matrix %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % load preprocessed data
        dir{j} = strcat(subDir,'\',runs{j});
%         tmp{j} = repmat(' ', 1, 115);
%         tmp{j} = spm_select('fplist',dir{j},'^swar.*\.nii');  

        % Get the filenames
        files = spm_select('fplist', dir{j}, '^swar.*\.nii');
        % Determine padding length
        pad_length = max(0, 150 - size(files, 2));
        % Pad filenames with trailing spaces
        padded_files = [files, repmat(' ', size(files, 1), pad_length)];
        % Assign to tmp
        tmp{j} = padded_files;
        
        % Load the .mat file for stimulus onset timings
        onsetDir = [onsetpath, '\', subs{i} ,'\LogFiles'];
        cd(onsetDir);
        load([onsetDir,'\','processed_', runs{j},'.mat']);
        Onset1=cell2mat(outputOnset(:,1));
        Onset2=cell2mat(outputOnset(:,2));
        Onset3=cell2mat(outputOnset(:,3));
        Onset4=cell2mat(outputOnset(:,4));
        Onset5=cell2mat(outputOnset(:,5));
        Onset6=cell2mat(outputOnset(:,6));
%         load([onsetDir,'\','adj_processed_', runs{j},'.mat']);
%         Onset1=cell2mat(outputOnset_adj(:,1));
%         Onset2=cell2mat(outputOnset_adj(:,2));
%         Onset3=cell2mat(outputOnset_adj(:,3));
%         Onset4=cell2mat(outputOnset_adj(:,4));
%         Onset5=cell2mat(outputOnset_adj(:,5));
%         Onset6=cell2mat(outputOnset_adj(:,6));
        
        B={Onset1,Onset2,Onset3,Onset4,Onset5,Onset6};
        
        Dur = ones(10,6)*3;
        Dur1=Dur(:,1);
        Dur2=Dur(:,2);
        Dur3=Dur(:,3);
        Dur4=Dur(:,4);
        Dur5=Dur(:,5);
        Dur6=Dur(:,6);
        D={Dur1,Dur2,Dur3,Dur4,Dur5,Dur6};
        

        
        condition_nameCell = {'Pleasant','Neutral','Unpleasant', 'PleasantAI','NeutralAI','UnpleasantAI'};
        ncon=6;
        
        for c = 1:ncon
            SPM.Sess(j).U(c).name = {condition_nameCell{1,c}};    
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
    
    max_cols = 0;
    for j = 1:numel(runs)
        max_cols = max(max_cols, size(tmp{j},2));
    end

    % Pad each cell's file list to have the same number of columns.
    for j = 1:numel(runs)
        pad_length = max_cols - size(tmp{j},2);
        if pad_length > 0
            tmp{j} = [tmp{j}, repmat(' ', size(tmp{j},1), pad_length)];
        end
    end

    %  SPM.nscan = ones(1,nses)*224;
    SPM.nscan = [218,218,218,218,218,218];
    SPM.xY.P = cat(1,tmp{:});
    SPM.xY.RT = TR;
    SPM = spm_fmri_spm_ui(SPM);
    cd(outputDir);
    SPM = spm_spm(SPM);
    clear SPM ;
   
end