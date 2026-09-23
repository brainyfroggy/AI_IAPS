% Define the base path to your EDF files and LogFiles
basePathEDF = 'F:\yujun\projects\LAB_IAPS_AI\DataRecording\Subject_test_el\Eyetracking';
basePathLog = 'F:\yujun\projects\LAB_IAPS_AI\DataRecording\Subject_test_el\LogFiles';
iapsFilePath = 'F:\yujun\projects\LAB_IAPS_AI\data\iaps_emotion_category_v2.csv';

% Load the IAPS emotion category data
iapsData = readtable(iapsFilePath);

% Initialize an array to hold the EDF file paths for the 10 runs
edfFilePaths = cell(1, 10);
logFilePaths = cell(1, 10);
for i = 1:10
    edfFilePaths{i} = fullfile(basePathEDF, sprintf('Run%02d.edf', i));
    logFilePaths{i} = fullfile(basePathLog, sprintf('Run%d.mat', i));
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

% Define the smoothing window size (e.g., 500 samples) for moving average
windowSize = 500;

% Initialize the figure for plotting
figure;


% Plot the pupil size data for each run in a 2x5 subplot layout
for i = 1:10
    % Load the LogFiles data for the current run
    logData = onsetStructs{i};

    % Extract the necessary columns from the LogFiles data
    stimulation = logData.dataLog;
    itiEndTimes = stimulation(strcmp(stimulation(:, 2), 'ITI end'), 4);

    % Get the numeric value of the last 'ITI end' time
    lastItiEnd = itiEndTimes{end};

    % Get the 'Stim on' times as numeric values
    stimOnTimes = stimulation(strcmp(stimulation(:, 2), 'Stim on'), 4);
    stimOnTimesNumeric = cell2mat(stimOnTimes);

    % Get the 'Code' values for the stimuli
    stimCodes = stimulation(strcmp(stimulation(:, 2), 'Stim on'), 3);

    % Calculate the onset time points of 'Stim on' relative to the last 'ITI end'
    stimOnsets = lastItiEnd - stimOnTimesNumeric;

    % Extract the pupil size data from the EDF struct
    pupilSizeData = edfStructs{i}.FSAMPLE.pa(1, :);

    % Replace 0 values with NaN to ignore them in the moving average
    pupilSizeData(pupilSizeData == 0) = NaN;

    % Smooth the pupil size data using a moving average filter, ignoring NaNs
    smoothedPupilSizeData = movmean(pupilSizeData, windowSize, 'omitnan');

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

    % Plot the data
    subplot(5, 2, i);
%     plot(timeVector, pupilSizeData, 'b', 'LineWidth', 0.5);
    
    plot(timeVector(1:length(smoothedPupilSizeData)), smoothedPupilSizeData, 'Color', [0.8 0.8 0.8], 'LineWidth', 1.5);
    hold on;
    % Mark the local maxima and minima
%     plot(timeVector(maxIndices), localMaxima, 'ro', 'MarkerSize', 5); % Mark local maxima
%     plot(timeVector(minIndices), localMinima, 'go', 'MarkerSize', 5); % Mark local minima

    % Join stimulation data with iapsData
    joinedData = innerjoin(table(stimCodes, stimOnsetsAlign), iapsData, 'LeftKeys', 'stimCodes', 'RightKeys', 'img');

    % Mark the onset time points of 'Stim on' with different colors and markers
    for j = 1:height(joinedData)
        if joinedData.stimOnsetsAlign(j) <= timeVector(end) % Ensure the onset is within the time range of the plot
            % Find the pupil size at the onset time
            pupilSizeAtOnset = interp1(timeVector, smoothedPupilSizeData, joinedData.stimOnsetsAlign(j), 'linear', 'extrap');
            
            % If pupil size at onset is NaN or 0, find the nearest non-NaN value
            if isnan(pupilSizeAtOnset) || pupilSizeAtOnset == 0
                onsetIndex = find(timeVector >= joinedData.stimOnsetsAlign(j), 1);
                leftIndex = onsetIndex - 1;
                rightIndex = onsetIndex + 1;

                while leftIndex > 0 && (isnan(smoothedPupilSizeData(leftIndex)) || smoothedPupilSizeData(leftIndex) == 0)
                    leftIndex = leftIndex - 1;
                end
                while rightIndex <= length(smoothedPupilSizeData) && (isnan(smoothedPupilSizeData(rightIndex)) || smoothedPupilSizeData(rightIndex) == 0)
                    rightIndex = rightIndex + 1;
                end

                if leftIndex > 0 && rightIndex <= length(smoothedPupilSizeData)
                    if abs(timeVector(leftIndex) - joinedData.stimOnsetsAlign(j)) <= abs(timeVector(rightIndex) - joinedData.stimOnsetsAlign(j))
                        pupilSizeAtOnset = smoothedPupilSizeData(leftIndex);
                    else
                        pupilSizeAtOnset = smoothedPupilSizeData(rightIndex);
                    end
                elseif leftIndex > 0
                    pupilSizeAtOnset = smoothedPupilSizeData(leftIndex);
                elseif rightIndex <= length(smoothedPupilSizeData)
                    pupilSizeAtOnset = smoothedPupilSizeData(rightIndex);
                else
                    pupilSizeAtOnset = NaN; % If no non-NaN value is found
                end
            end

            if ~isnan(pupilSizeAtOnset)
                color = 'b'; % Default color
                if strcmp(joinedData.emotion_type{j}, 'pleasant')
                    color = 'g';
                elseif strcmp(joinedData.emotion_type{j}, 'unpleasant')
                    color = 'r';
                end

                marker = 'o'; % Default marker
                markerFaceColor = color; % Default solid color
                if strcmp(joinedData.AI{j}, 'YES')
                    markerFaceColor = 'none'; % Hollow marker for AI 'Yes'
                end

                plot(joinedData.stimOnsetsAlign(j), pupilSizeAtOnset, ...
                     'Marker', marker, 'MarkerEdgeColor', color, 'MarkerFaceColor', markerFaceColor, 'MarkerSize', 8, 'LineWidth', 2);
            end
            
        end
    end

    hold off;
    xlabel('Time (seconds)');
    ylabel('Pupil Size');
    title(sprintf('Run %d', i));
%     legend( 'Smoothed Data', 'pleasant-NA', 'neutral-NA', 'unpleasant-NA', 'pleasant-AI', 'neutral-AI', 'unpleasant-AI');
    grid on;
    axis tight; % Fit the axis tightly around the data
end

% Add a super title for the figure
sgtitle('Pupil Size Over Time for 10 Runs, SUBJECT1 (smoothed with 0.5 sec window)');



% Add the legend
subplot(5, 2, 1);
legendHandles = [];
legendLabels = {'Smoothed Data', 'Pleasant', 'Neutral', 'Unpleasant', 'Pleasant-AI', 'Neutral-AI', 'Unpleasant-AI'};

% Plot dummy data for the legend
hold on;
legendHandles(1) = plot(NaN, NaN, 'Color', [0.8 0.8 0.8], 'LineWidth', 1.5);
legendHandles(2) = plot(NaN, NaN, 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 8, 'LineWidth', 2);
legendHandles(3) = plot(NaN, NaN, 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 8, 'LineWidth', 2);
legendHandles(4) = plot(NaN, NaN, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8, 'LineWidth', 2);
legendHandles(5) = plot(NaN, NaN, 'go', 'MarkerFaceColor', 'none', 'MarkerSize', 8, 'LineWidth', 2);
legendHandles(6) = plot(NaN, NaN, 'bo', 'MarkerFaceColor', 'none', 'MarkerSize', 8, 'LineWidth', 2);
legendHandles(7) = plot(NaN, NaN, 'ro', 'MarkerFaceColor', 'none', 'MarkerSize', 8, 'LineWidth', 2);
hold off;

legend(legendHandles, legendLabels, 'Location', 'best');





