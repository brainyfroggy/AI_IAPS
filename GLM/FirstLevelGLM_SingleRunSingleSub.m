%%% 1st Level Categorical Design for one session/subject
clear;

cwd = ['C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\MRIData\20121201-Conditioning-Sub16\spm\run1\';
       'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\MRIData\20121201-Conditioning-Sub16\spm\run2\';
       'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\MRIData\20121201-Conditioning-Sub16\spm\run3\';];   % Working directory
   
behavdir = ['C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub16_Run1.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub16_Run2.mat';
            'C:\Users\Vladimir Liu\MyProjects\EEG_fMRI\Conditioning\BehavioralData\Sub16_Run3.mat';];        % File Directory for the stimulus onset timings
        
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
for ses = 1:3                % Session ID: 1->3
    cd(cwd(ses,:));


    swd = [cwd(ses,:) 'glm_SingleRun'];
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


    load(behavdir(ses,:));    % Load the .mat file for stimulus onset timings
    for c = 1:ncon(ses)
        SPM.Sess(1).U(c).name = {cname{ses,c}};    % Use 1 for Sess index now for separate models
        SPM.Sess(1).U(c).ons = sots{c};
        SPM.Sess(1).U(c).dur = 0;
        SPM.Sess(1).U(c).P(1).name = 'none';       % Parametric Modulation; 'none' for now
    end


% Include movement parameters

    rnam = {'X','Y','Z','x','y','z'};
    fn = spm_select('list',cwd(ses,:),'^rp.*\.txt');
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

    tmp{1} = spm_select('fplist',cwd(ses,:),'^swas.*\.img');

    SPM.xY.P = cat(1,tmp{:});
    SPM.xY.RT = TR;

% Configure design matrix
    SPM = spm_fmri_spm_ui(SPM);

% Estimate parameters
    cd(swd);
    SPM = spm_spm(SPM);
    clear SPM;
end











