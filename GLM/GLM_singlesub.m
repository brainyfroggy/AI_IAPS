%%% 1st Level Categorical Design for all sessions/subjects
clear;
spm('defaults','FMRI')

global defaults

mainpath = dir('F:\yujun\projects\LAB_IAPS_AI\prepfMRI\testsession\subject1');   % Working directory
   
runs = ['RUN1']
   
behavdir = dir(strcat('F:\yujun\projects\LAB_IAPS_AI\onset','\*.mat'));        % File Directory for the stimulus onset timings
        
% sessionID = 1;                   % The session number from 1 -> 3; change it accordingly
nscan = [340;344;340];           % Number of scans for each of the three sessions
nses = 1;                       % Total number of sessions/runs to be modeled, 3 in this case, habituation, acquisition, and extinction; each modeled separately
TR = 1.98;

for i = 3:20
    for j = 2:2                % Session ID: 1->3
        currDir = ['C:\Conditioning\MRIData\',mainpath(i).name,'\spm\',runs(j,:)];
        cd(currDir);
        mkdir('Non_smoothed GLM Beta series');
        outputdir = [currDir '\Non_smoothed GLM Beta series'];

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Basis functions and timing parameters %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        SPM.nscan          = nscan(j);
        
        SPM.xBF.name = 'hrf';       % name of basis function
        SPM.xBF.length = 32.0513;   % length in seconds of basis
        SPM.xBF.order = 1;          % order of basis set
        SPM.xBF.T = 36;             % number of subdivisions of TR
        SPM.xBF.T0 = 18;            % first time bin (see slice timing)
        SPM.xBF.UNITS = 'scans';    % options: 'scans'|'secs' for onsets
        SPM.xBF.Volterra = 1;       % order of convolution
%         SPM.xBF.dt % length of time bin in seconds
%         SPM.xBF.bf % basis set matrix

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Trial Specification: Design Matrix %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        onsetDir = ['C:\Conditioning\BehavioralData'];
        cd(onsetDir);
        load(behavdir((i-3)*3+j+1).name);    % Load the .mat file for stimulus onset timings
        
        time_by_Order(1,:) = [sots{1} sots{2} sots{3}];
        Condi_number = size (time_by_Order,2);
        for n = 1:Condi_number
            if n<=length(sots{1})
                time_by_Order(2,n) = -1;
            else if n<=length(sots{1})+length(sots{2})
                    time_by_Order(2,n) = 1;
                else if n<=Condi_number
                        time_by_Order(2,n) = 0;
                    end
                end
            end
        end
        
        Sorted_designmatrix = sortrows(time_by_Order',1);
        Sorted_designmatrix = Sorted_designmatrix';
        
        condition_name = cell(3,Condi_number);
        for y=1:Condi_number
            B = {y};
            [condition_name{2,y}] = deal(B);
        end
        
        for c = 1:Condi_number
            SPM.Sess(1).U(c).name = condition_name{j,c};    
            SPM.Sess(1).U(c).ons = Sorted_designmatrix(1,c);
            SPM.Sess(1).U(c).dur = 0;
            SPM.Sess(1).U(c).P(1).name = 'none';       % Parametric Modulation; 'none' for now
        end


    % Include movement parameters
        cd(currDir);

        rnam = {'X','Y','Z','x','y','z'};
        fn = spm_select('list',currDir,'^rp.*\.txt');
        [r1,r2,r3,r4,r5,r6] = textread(fn,'%f%f%f%f%f%f');
        SPM.Sess(1).C.C = [r1 r2 r3 r4 r5 r6];
        SPM.Sess(1).C.name = rnam;

    % Global Normalization: OPTIONS: 'Scaling'
    %-------------------------------------------------------------
        SPM.xGX.iGXcalc = 'Scaling';
        
    % low frequency confound: high-pass cutoff (secs) [Inf = no filtering]
    %-------------------------------------------------------------
        SPM.xX.K(1).HParam = 128;
        
    % intrinsic autocorrelations: OPTIONS: 'none'|'AR(1) + w'
    %-------------------------------------------------------------
        SPM.xVi.form       = 'AR(1)';
        
    % Specify data image files
        tmp{1} = spm_select('fplist',currDir,'^wars.*\.img');
        
        SPM.xY.P = cat(1,tmp{:});
        SPM.xY.RT = TR;

    % Configure design matrix
        SPM = spm_fmri_spm_ui(SPM);

    % Estimate parameters
        cd(outputdir);
        SPM = spm_spm(SPM);
        clear SPM;
        
    %%  Save the design matrix
%     Savename= strcat('DesignMatrix_',mainpath(i).name(23:27),runs(j,:));
%     eval([Savename '=Sorted_designmatrix']); 
%     savepath = 'C:\Users\Siyang\Documents\Conditioning\';
%     EEG_BOLD = [savepath 'Designmatrix.mat'];
%     save(EEG_BOLD,Savename,'-append');
    end
 end
