% Specify the path to the folder containing the SPM.mat file
subject = 'Sub7';
spm_dir = fullfile('F:\yujun\projects\AI_IAPS\GLM_singletrial\betas', subject);

% Load the SPM.mat file
spm_file = fullfile(spm_dir, 'SPM.mat');
load(spm_file);

% Plot the design matrix
figure;
spm_DesRep('DesMtx', SPM.xX.name);
title('Design Matrix');


% Define the subject and runs
subject = 'Sub7';
runs = {'Run01', 'Run02', 'Run03', 'Run04', 'Run05', 'Run06', 'Run07', 'Run08', 'Run09', 'Run10'};

% Specify the path to the folder containing the SPM.mat file
spm_dir = fullfile('F:\yujun\projects\AI_IAPS\GLM_singletrial\betas', subject);

% Load the SPM.mat file
spm_file = fullfile(spm_dir, 'SPM.mat');
load(spm_file);

% Plot the design matrix
figure;
imagesc(SPM.xX.X);
xlabel('Predictors');
ylabel('Time (Volumes)');
title('Design Matrix');
colorbar;
