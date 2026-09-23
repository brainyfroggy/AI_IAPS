% Define the base path to your EDF files and LogFiles
basePathEDF = 'N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\DataRecording\Sub23\Eyetracking';
basePathLog = 'N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\DataRecording\Sub23\LogFiles';
iapsFilePath = 'N:\Experimental_Data\yujunchen\projects\LAB_IAPS_AI\data\iaps_emotion_category_v2.csv';

% Load the IAPS emotion category data
iapsData = readtable(iapsFilePath);

% Number of runs
numRuns = 10;

% Define the time window
preOnset = -1000; % 1000 ms before onset
postOnset = 3000; % 3000 ms after onset
windowLength = (postOnset - preOnset); % 4000 ms

% Initialize an array to hold the EDF file paths for the 10 runs
edfFilePaths = cell(1, 10);
logFilePaths = cell(1, 10);
for i = 1:10
    edfFilePaths{i} = fullfile(basePathEDF, sprintf('Run%02d.edf', i));
    logFilePaths{i} = fullfile(basePathLog, sprintf('Run%d.mat', i));
end

% Check if the processed data file exists
processedDataFile = 'sub23_pupil_data.mat';
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
logDataFile = 'sub2_onset_data.mat';
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

% Initialize a cell array to hold the extracted data tables
extractedDataTables = cell(numRuns, 1);

% Loop through each run
for i = 1:numRuns
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
%     smoothedPupilSizeData = movmean(pupilSizeData, windowSize, 'omitnan');


    % Extract the number of samples
    numSamples = length(pupilSizeData);

    % Define the sampling rate (samples per second)
    samplingRate = 1000;

    % Create a time vector in seconds
    timeVector = (1:numSamples) / samplingRate;

    timeVectorEnd = timeVector(end);
    stimOnsetsAlign = timeVectorEnd - stimOnsets;
    
    % Mark the local maxima and minima
%     plot(timeVector(maxIndices), localMaxima, 'ro', 'MarkerSize', 5); % Mark local maxima
%     plot(timeVector(minIndices), localMinima, 'go', 'MarkerSize', 5); % Mark local minima

    % Join stimulation data with iapsData
    joinedData = innerjoin(table(stimCodes, stimOnsetsAlign), iapsData, 'LeftKeys', 'stimCodes', 'RightKeys', 'img');

    % Sort the resulting table by stimCodes
    sortedJoinedData = sortrows(joinedData, 'stimOnsetsAlign');
    

    % Initialize an array to hold the extracted pupil sizes
    extractedPupilSizes = zeros(height(sortedJoinedData), windowLength);
    
    % Loop through each onset to extract the data
    for j = 1:height(sortedJoinedData)
        % Get the onset time
        onsetTime = sortedJoinedData.stimOnsetsAlign(j);

        % Define the time window around the onset
        startTime = onsetTime + preOnset / 1000; % convert ms to seconds
        endTime = onsetTime + postOnset / 1000;  % convert ms to seconds

        % Find the indices in the time vector corresponding to this window
        startIndex = find(timeVector >= startTime, 1, 'first');
        endIndex = find(timeVector <= endTime, 1, 'last');

        % Extract the pupil size data for this window
        if endIndex - startIndex + 1 == windowLength
            extractedPupilSizes(j, :) = pupilSizeData(startIndex:endIndex);
        else
            % Handle cases where the window length does not match the expected length
            extractedPupilSizes(j, 1:(endIndex - startIndex + 1)) = pupilSizeData(startIndex:endIndex);
        end
    end

    % Create a table for the extracted data with row names as stimCodes
    extractedDataTables{i} = array2table(extractedPupilSizes, 'RowNames', sortedJoinedData.stimCodes);

end


% Initialize an empty array to hold the combined data
combinedData = [];

% Loop through each run and concatenate the tables
for run = 1:numRuns
    % Convert the table in the cell to an array and concatenate it
    combinedData = [combinedData; table2array(extractedDataTables{run})];
end

% Create a final table with appropriate row names
% Create row names based on the stimCodes and run number
allStimCodes = [];
for run = 1:numRuns
    currentStimCodes = strcat('Run', num2str(run), '_', extractedDataTables{run}.Properties.RowNames);
    allStimCodes = [allStimCodes; currentStimCodes];
end

% Convert the combined data to a table with the row names
finalExtractedData = array2table(combinedData, 'RowNames', allStimCodes);


% Baseline correction
% Calculate the mean of the first 1000 time points for each row
baselineMeans = nanmean(combinedData(:, 1:1000), 2);

% Subtract the mean from each time point in the row to baseline correct
baselineCorrectedData = combinedData - baselineMeans;

% Convert the baseline-corrected data to a table with the row names
finalBaselineCorrectedData = array2table(baselineCorrectedData, 'RowNames', allStimCodes);



stimCodes_runs = finalBaselineCorrectedData.Properties.RowNames;
% Extract the part after the first underscore for matching
extractedStimCodes = cellfun(@(x) x(find(x == '_', 1, 'first')+1:end), stimCodes_runs, 'UniformOutput', false);

% Initialize containers for each group
groups = unique(iapsData.group);
groupTables = containers.Map;

for i = 1:length(groups)
    groupName = groups{i};
    groupTables(groupName) = table();
end

% Loop through each extractedStimCode in finalBaselineCorrectedData
for i = 1:length(extractedStimCodes)
    currentStimCode = extractedStimCodes{i};
    
    % Find the matching row in iapsData
    matchingRow = find(strcmp(iapsData.img, currentStimCode), 1);
    
    if ~isempty(matchingRow)
        % Get the group of the current stimCode
        currentGroup = iapsData.group{matchingRow};
        
        % Append the current row to the corresponding group table
        groupTables(currentGroup) = [groupTables(currentGroup); finalBaselineCorrectedData(i, :)];
    end
end

% Create tables with names based on the groups
for i = 1:length(groups)
    groupName = groups{i};
    groupTableName = ['BaselineCorrData_' groupName ];
    assignin('base', groupTableName, groupTables(groupName));
end



% Convert tables to arrays
neutralAI_array = table2array(BaselineCorrData_neutralAI);
pleasantAI_array = table2array(BaselineCorrData_pleasantAI);
unpleasantAI_array = table2array(BaselineCorrData_unpleasantAI);

% Calculate the mean of each array along the rows
avg_neutralAI = mean(neutralAI_array, 1, 'omitnan');
avg_pleasantAI = mean(pleasantAI_array, 1, 'omitnan');
avg_unpleasantAI = mean(unpleasantAI_array, 1, 'omitnan');

% Create a time vector for the x-axis
time_vector = -1000:2999;

% Plot the three lines on the same plot
figure;
hold on;
plot(time_vector, avg_neutralAI, 'b');
plot(time_vector, avg_pleasantAI, 'g');
plot(time_vector, avg_unpleasantAI, 'r');

% Add labels and title
xlabel('Time (ms)');
ylabel('Average Pupil Size');
title('Average Pupil Size Across Conditions');
legend('Neutral__AI', 'Pleasant__AI', 'Unpleasant__AI');



% Convert tables to arrays
neutral_array = table2array(BaselineCorrData_neutral);
pleasant_array = table2array(BaselineCorrData_pleasant);
unpleasant_array = table2array(BaselineCorrData_unpleasant);

% Calculate the mean of each array along the rows
avg_neutral = mean(neutral_array, 1, 'omitnan');
avg_pleasant = mean(pleasant_array, 1, 'omitnan');
avg_unpleasant = mean(unpleasant_array, 1, 'omitnan');

% Create a time vector for the x-axis
time_vector = -1000:2999;

% Plot the three lines on the same plot
figure;
hold on;
plot(time_vector, avg_neutral, 'b');
plot(time_vector, avg_pleasant, 'g');
plot(time_vector, avg_unpleasant, 'r');

% Add labels and title
xlabel('Time (ms)');
ylabel('Average Pupil Size');
title('Average Pupil Size Across Conditions');
legend('Neutral', 'Pleasant', 'Unpleasant');


