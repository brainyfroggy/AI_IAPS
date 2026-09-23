% Define the base path to your EDF files and LogFiles
basePathEDF = 'F:\yujun\projects\LAB_IAPS_AI\DataRecording\Sub1\Eyetracking';
basePathLog = 'F:\yujun\projects\LAB_IAPS_AI\DataRecording\Sub1\LogFiles';
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

% Define the time window around stimulus onset (in seconds)
preStimulus = 0.1; % 0.1 seconds before stimulus
postStimulus = 0.1; % 0.1 seconds after stimulus
epochDuration = preStimulus + postStimulus; % Total duration of the epoch

% Define the sampling rate (samples per second)
samplingRate = 1000; % Assuming the sampling rate is 1000 Hz
epochSamples = epochDuration * samplingRate; % Number of samples in the epoch

% Initialize a cell array to hold the result data
resultData = cell(10, 1);

% Loop through each run
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

    % Create a time vector in seconds
    numSamples = length(pupilSizeData);
    timeVector = (1:numSamples) / samplingRate;
    timeVectorEnd = timeVector(end);
    stimOnsetsAlign = timeVectorEnd - stimOnsets;

    % Join stimulation data with iapsData
    joinedData = innerjoin(table(stimCodes, stimOnsetsAlign, 'VariableNames', {'stimCodes', 'stimOnsetsAlign'}), iapsData, 'LeftKeys', 'stimCodes', 'RightKeys', 'img');

    % Initialize arrays to store the averaged pupil sizes, categories, and AI values
    avgPupilSizes = NaN(height(joinedData), 1);
    categories = cell(height(joinedData), 1);
    AI = cell(height(joinedData), 1);

    % Epoch the data and calculate the average pupil size
    for j = 1:height(joinedData)
        if joinedData.stimOnsetsAlign(j) <= timeVector(end) % Ensure the onset is within the time range of the data
            % Find the index of the onset time
            onsetIndex = find(timeVector >= joinedData.stimOnsetsAlign(j), 1);

            % Define the start and end indices for the epoch
            startIndex = max(onsetIndex - preStimulus * samplingRate, 1);
            endIndex = min(onsetIndex + postStimulus * samplingRate, numSamples);

            % Extract the epoch
            epochData = pupilSizeData(startIndex:endIndex);

            % Handle NaN values in the epoch
            if all(isnan(epochData))
                % Find the nearest non-NaN value before or after the epoch
                leftIndex = startIndex - 1;
                rightIndex = endIndex + 1;

                while leftIndex > 0 && isnan(pupilSizeData(leftIndex))
                    leftIndex = leftIndex - 1;
                end
                while rightIndex <= numSamples && isnan(pupilSizeData(rightIndex))
                    rightIndex = rightIndex + 1;
                end

                if leftIndex > 0 && rightIndex <= numSamples
                    if abs(timeVector(leftIndex) - joinedData.stimOnsetsAlign(j)) <= abs(timeVector(rightIndex) - joinedData.stimOnsetsAlign(j))
                        epochData(:) = pupilSizeData(leftIndex);
                    else
                        epochData(:) = pupilSizeData(rightIndex);
                    end
                elseif leftIndex > 0
                    epochData(:) = pupilSizeData(leftIndex);
                elseif rightIndex <= numSamples
                    epochData(:) = pupilSizeData(rightIndex);
                else
                    epochData(:) = NaN; % If no non-NaN value is found
                end
            end

            % Calculate the average pupil size
            avgPupilSizes(j) = mean(epochData, 'omitnan');
            categories{j} = joinedData.emotion_type{j};
            AI{j} = joinedData.AI{j};
        end
    end

    % Store the results in the resultData cell array
    resultData{i} = table(joinedData.stimCodes, joinedData.stimOnsetsAlign, categories, AI, avgPupilSizes, ...
                          'VariableNames', {'StimulusCode', 'StimOnset', 'Category', 'AI', 'AvgPupilSize'});
    
    % Sort the result data for the current run by 'StimOnset' to keep the original onset order
    resultData{i} = sortrows(resultData{i}, 'StimOnset');
end

% Combine all runs into a single table
combinedResults = vertcat(resultData{:});

% Display the combined results
% disp(combinedResults);

% combinedResults_sub2 = combinedResults
% combinedResults_sub1 = combinedResults(end-539:end, :);

% Define categories for comparison
categories = {'pleasant', 'neutral', 'unpleasant'};
AIValues = {'NO', 'YES'};

% Initialize a figure for the plots
figure;


% Set the significance level
alpha = 0.05;

% Counter for subplot indexing
plotIndex = 1;

for aiIdx = 1:length(AIValues)
    for category = 1:2 % Compare 'pleasant' and 'unpleasant' against 'neutral'
        % Extract data for the current AI value and categories
        aiValue = AIValues{aiIdx};
        if category == 1
            category1 = 'pleasant';
            category2 = 'neutral';
        else
            category1 = 'unpleasant';
            category2 = 'neutral';
        end

        % Filter data based on AI value and categories
        data1 = combinedResults.AvgPupilSize(strcmp(combinedResults.Category, category1) & strcmp(combinedResults.AI, aiValue));
        data2 = combinedResults.AvgPupilSize(strcmp(combinedResults.Category, category2) & strcmp(combinedResults.AI, aiValue));

        % Calculate means
        mean1 = mean(data1, 'omitnan');
        mean2 = mean(data2, 'omitnan');

        % Perform t-test
        [h, p] = ttest2(data1, data2, 'Alpha', alpha);

        % Create the subplot
        subplot(2, 2, plotIndex);
        bar([1, 2], [mean1, mean2]);
        hold on;

        % Add significance marker
        maxY = max(mean1, mean2) + 0.5; % Add extra space to the max Y value
        if h == 1
            % Significant difference
            sigLabel = '*';
            sigText = sprintf('p = %.4f', p);
        else
            % No significant difference
            sigLabel = 'ns';
            sigText = sprintf('p = %.4f', p);
        end
        % Place the p-value and significance indicator on the bottom right of the plot
        text(0.8, 0.1, sigLabel, 'HorizontalAlignment', 'right', 'FontSize', 14, 'FontWeight', 'bold', 'Units', 'normalized');
        text(0.8, 0.05, sigText, 'HorizontalAlignment', 'right', 'FontSize', 10, 'Units', 'normalized');


        % Add error bars
        error1 = std(data1, 'omitnan') / sqrt(length(data1));
        error2 = std(data2, 'omitnan') / sqrt(length(data2));
        errorbar([1, 2], [mean1, mean2], [error1, error2], 'k', 'LineStyle', 'none');

        % Add labels and title
        set(gca, 'XTick', [1, 2], 'XTickLabel', {category1, category2});
        ylabel('Average Pupil Size');
        title(sprintf('%s vs %s (AI = %s)', category1, category2, aiValue));
        
        % Increment the plot index
        plotIndex = plotIndex + 1;

        hold off;
    end
end

% Add a super title for the figure
sgtitle('Comparison of Average Pupil Sizes');


% Ensure that all subplots have enough space for annotations
set(gcf, 'Position', [100, 100, 1200, 800]); % Adjust figure size as needed



