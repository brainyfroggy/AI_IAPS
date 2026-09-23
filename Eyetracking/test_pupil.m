% Define the base path to your EDF files
basePath = 'F:\yujun\projects\LAB_IAPS_AI\DataRecording\Sub1\Eyetracking';
onset_basePath = 'F:\yujun\projects\LAB_IAPS_AI\DataRecording\Sub1\LogFiles';
iapsFilePath = 'F:\yujun\projects\LAB_IAPS_AI\data\iaps_emotion_category_v2.csv';


% Load the IAPS emotion category data
iapsData = readtable(iapsFilePath);

% Initialize an array to hold the EDF file paths for the 10 runs
edfFilePaths = cell(1, 10);
for i = 1:10
    edfFilePaths{i} = fullfile(basePath, sprintf('Run%02d.edf', i));
end

logFilePaths = cell(1, 10);
for i = 1:10
    logFilePaths{i} = fullfile(onset_basePath, sprintf('Run%d.mat', i));
end

% Check if the processed data file exists
processedDataFile = 'sub1_pupil_data.mat';
if exist(processedDataFile, 'file')
    % Load the processed data file
    load(processedDataFile, 'edfStructs');
else
    % Initialize cell array to hold EDF structs for each run
    edfStructs = cell(1, 10);

    % Loop through each EDF file, read the data, and process it
    for i = 1:10
        % Load the EDF file
        edfStructs{i} = edfmex(edfFilePaths{i});
    end

    % Save the EDF structs to a file
    save(processedDataFile, 'edfStructs');
end

% Check if the processed data file exists
logDataFile = 'sub1_onset_data.mat';
if exist(logDataFile, 'file')
    % Load the processed data file
    load(logDataFile, 'onsetStructs');
else
    % Initialize cell array to hold EDF structs for each run
    onsetStructs = cell(1, 10);

    % Loop through each EDF file, read the data, and process it
    for i = 1:10
        % Load the EDF file
        onsetStructs{i} = load(logFilePaths{i});
    end

    % Save the EDF structs to a file
    save(logDataFile, 'onsetStructs');
end


% Define the smoothing window size (e.g., 50 samples) for moving average
windowSize = 500;
figure;
% Plot the pupil size data for each run in a 2x5 subplot layout
% figure('Position', [100, 100, 1400, 800]); % Set figure size
for i = 1:10
    % Calculate the onset time points of 'Stim on' relative to the last 'ITI end'
    stimulation= onsetStructs{i}.dataLog;
    ItiendTimes  = stimulation(strcmp(stimulation(:, 2), 'ITI end'),4);

    
    % Get the numeric value of the last 'ITI end' time
    lastItiEnd = ItiendTimes{end};

    % Get the 'Stim on' times as numeric values
    stimOnTimes = stimulation(strcmp(stimulation(:, 2), 'Stim on'), 4);
    stimOnTimesNumeric = cell2mat(stimOnTimes);

    % Calculate the onset time points of 'Stim on' relative to the last 'ITI end'
    stimOnsets = lastItiEnd - stimOnTimesNumeric;
    
    % Extract the pupil size data from the EDF struct
    pupilSizeData = edfStructs{i}.FSAMPLE.pa(1, :);

    % Remove all 0 pupil size values
    nonZeroIndices = pupilSizeData ~=0 ;
    pupilSizeData = pupilSizeData(nonZeroIndices);

    % Smooth the pupil size data using a moving average filter
    smoothedPupilSizeData = movmean(pupilSizeData, windowSize);

    % Find local maxima and minima
    [~, maxIndices] = findpeaks(smoothedPupilSizeData);
    [~, minIndices] = findpeaks(-smoothedPupilSizeData);
    localMaxima = smoothedPupilSizeData(maxIndices);
    localMinima = smoothedPupilSizeData(minIndices);

    % Extract the number of samples
    numSamples = length(pupilSizeData);

    % Define the sampling rate (samples per second)
    samplingRate = 1000;

    % Create a time vector in seconds
    timeVector = (1:numSamples) / samplingRate;
    
    timeVectorEnd = timeVector(end);
    stimOnsetsAlign = timeVectorEnd - stimOnsets;

    subplot(5, 2, i);
%     plot(timeVector, pupilSizeData, 'b', 'LineWidth', 0.5);
    
    plot(timeVector(1:length(smoothedPupilSizeData)), smoothedPupilSizeData, 'Color', [0.8 0.8 0.8], 'LineWidth', 1.5);
    hold on;
    
%     plot(timeVector(maxIndices), localMaxima, 'ro', 'MarkerSize', 5); % Mark local maxima
%     plot(timeVector(minIndices), localMinima, 'go', 'MarkerSize', 5); % Mark local minima


    % Mark the onset time points of 'Stim on' with green circles
    for j = 1:length(stimOnsetsAlign)
        if stimOnsetsAlign(j) <= timeVector(end) % Ensure the onset is within the time range of the plot
            plot(stimOnsetsAlign(j), interp1(timeVector, pupilSizeData, stimOnsetsAlign(j)), 'bo', 'MarkerSize', 8);
        end
    end
    
    hold off;
    xlabel('Time (seconds)');
    ylabel('Pupil Size');
    title(sprintf('Run %d', i));
    legend('Smoothed Data', 'Onsets');
    grid on;
    axis tight; % Fit the axis tightly around the data
end
sgtitle('Pupil Size Over Time for 10 Runs (After Removing 0 Values), Sub 1');