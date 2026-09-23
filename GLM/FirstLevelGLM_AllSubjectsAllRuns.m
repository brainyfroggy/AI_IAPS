%%% 1st Level Categorical Design for all sessions/subjects
clear;

cwd = ['C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\MRIData\20120813-Conditioning-Sub02\spm\';
       'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\MRIData\20120815-Conditioning-Sub03\spm\';
       'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\MRIData\20120816_Conditioning_Sub04\spm\';
       'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\MRIData\20120818_Conditioning_Sub05\spm\';
       'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\MRIData\20120820_Conditioning_Sub06\spm\';
       'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\MRIData\20120918-Conditioning-Sub07\spm\';
       'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\MRIData\20120919-Conditioning-Sub08\spm\';
       'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\MRIData\20120920-Conditioning-Sub09\spm\';
       'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\MRIData\20120926-Conditioning-Sub10\spm\';
       'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\MRIData\20120927-Conditioning-Sub11\spm\';
       'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\MRIData\20121001-Conditioning-Sub12\spm\';
       'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\MRIData\20121002-Conditioning-Sub13\spm\';
       'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\MRIData\20121003-Conditioning-Sub14\spm\';
       'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\MRIData\20121004-Conditioning-Sub15\spm\';];   % Working directory
   
runs = ['run1\';
        'run2\';
        'run3\';];
   
behavdir = ['C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub01_Run1.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub01_Run2.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub01_Run3.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub02_Run1.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub02_Run2.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub02_Run3.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub04_Run1.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub04_Run2.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub04_Run3.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub05_Run1.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub05_Run2.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub05_Run3.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub06_Run1.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub06_Run2.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub06_Run3.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub07_Run1.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub07_Run2.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub07_Run3.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub08_Run1.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub08_Run2.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub08_Run3.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub09_Run1.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub09_Run2.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub09_Run3.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub10_Run1.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub10_Run2.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub10_Run3.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub11_Run1.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub11_Run2.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub11_Run3.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub12_Run1.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub12_Run2.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub12_Run3.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub13_Run1.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub13_Run2.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub13_Run3.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub14_Run1.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub14_Run2.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub14_Run3.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub15_Run1.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub15_Run2.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub15_Run3.mat';];        % File Directory for the stimulus onset timings
        
% sessionID = 1;                   % The session number from 1 -> 3; change it accordingly
ncon = [2,3,2];                        % Number of Conditions you wish to model: CS+, CS-
cname = cell(3,3); 
cname{1,1} = 'CS-'; cname{1,2} = 'CS+';
cname{2,1} = 'CS-'; cname{2,2} = 'CS+unpair';cname{2,3} = 'CS+pair';
cname{3,1} = 'CS-'; cname{3,2} = 'CS+';            % Name of each condition; make sure the order is the same with what's stored in 'sots.mat';

nsess = 3;                       % Total number of sessions/runs to be modeled, 3 in this case, habituation, acquisition, and extinction; each modeled separately
TR = 1.98;

nscan = [340;344;340];           % Number of scans for each of the three sessions

%-------------------------------------------------------------------------%
% Start specifying the design matrix and parameters of the 1st level GLM  %
% for each session separately                                             %
%-------------------------------------------------------------------------%
for i = 1:size(cwd,1)
    for ses = 1:nsess                % Session ID: 1->3
        currDir = [cwd(i,:),runs(ses,:)];
        cd(currDir);
        
        swd = [currDir 'glm1'];
        SPM.nscan = nscan(ses);    % Get the # of scans within the current session to be analyzed

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Basis functions and timing parameters %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        SPM.xBF.name = 'hrf';
        SPM.xBF.order = 1;
        SPM.xBF.length = 32.0513;
        SPM.xBF.T = 36;              % Number of slices. This goes into the Microtime Resolution
        SPM.xBF.T0 = 18;             % Middle slice as the reference slice. This goes into the Microtime Onset
        SPM.xBF.UNITS = 'scans';     % OPTIONS: 'scans'|'secs'
        SPM.xBF.Volterra = 1;        % OPTIONS: 1|2 = order of convolution

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Trial Specification: Design Matrix %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        load(behavdir((i-1)*3+ses,:));    % Load the .mat file for stimulus onset timings
        for c = 1:ncon(ses)
            SPM.Sess(1).U(c).name = {cname{ses,c}};    % Use 1 for Sess index now for separate models
            SPM.Sess(1).U(c).ons = sots{c};
            SPM.Sess(1).U(c).dur = 0;
            SPM.Sess(1).U(c).P(1).name = 'none';       % Parametric Modulation; 'none' for now
        end


    % Include movement parameters

        rnam = {'X','Y','Z','x','y','z'};
        fn = spm_select('list',currDir,'^rp.*\.txt');
        [r1,r2,r3,r4,r5,r6] = textread(fn,'%f%f%f%f%f%f');
        SPM.Sess(1).C.C = [r1 r2 r3 r4 r5 r6];
        SPM.Sess(1).C.name = rnam;

    % Global Normalization: OPTIONS: 'Scaling'
        SPM.xGX.iGXcalc = 'Scaling';

    % Low frequency confound: high-pass cutoff (secs) [Inf = no filtering]

        SPM.xX.K(1).HParam = 128;     % HPF cutoff 128 s


    % Adjustment for Intrinsic serial correlation: OPTIONS: 'none'|'AR(1)+w'
        SPM.xVi.form = 'AR(1)';

    % Specify data image files

        tmp{1} = spm_select('fplist',currDir,'^swas.*\.img');

        SPM.xY.P = cat(1,tmp{:});
        SPM.xY.RT = TR;

    % Configure design matrix
        SPM = spm_fmri_spm_ui(SPM);

    % Estimate parameters
        cd(swd);
        SPM = spm_spm(SPM);
        clear SPM;
    end
end
