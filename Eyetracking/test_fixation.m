
% Specify the path to your EDF file
edfFilePath = 'F:\yujun\projects\LAB_IAPS_AI\DataRecording\Sub2\Eyetracking\Run03.edf';
% % edfFilePath = 'N:\Experimental_Data\Qiang Yang\New_feature\EyeTracking\Sub05\Session01\Run01.edf';
edfStruct = edfmex(edfFilePath);

% Load the image names from the text file
stimulusFile = 'F:\yujun\projects\LAB_IAPS_AI\data\stimulus_v2\r4.txt';
imageNames = importdata(stimulusFile);

% Construct the full path to the first image
imagePath = fullfile('F:\yujun\projects\LAB_IAPS_AI\data\IAPS_AI_120photos_v2', imageNames{2});

% Load the image
img = imread(imagePath);

% Resize the image to 1920 by 1080
gray_image = rgb2gray(img);

% Get the size of the grayscale image
[imgHeight, imgWidth] = size(gray_image);

% Create a black background of size 1920 by 1080
background = zeros(1080, 1920, 'uint8'); % 'uint8' for a grayscale image

% Calculate the position to place the image in the center of the background
startRow = floor((1080 - imgHeight) / 2) + 1;
startCol = floor((1920 - imgWidth) / 2) + 1;

% Place the grayscale image in the center of the background
background(startRow:startRow + imgHeight - 1, startCol:startCol + imgWidth - 1) = gray_image;

% Display the image
figure;
imshow(background);
title('First Image Stimulus');
hold on;

% Extract fixation and saccade events
fixationEvents = edfStruct.FEVENT([edfStruct.FEVENT.type] == 7);
saccadeEvents = edfStruct.FEVENT([edfStruct.FEVENT.type] == 5);

% Determine the start time
start_time = min([fixationEvents.sttime]);
end_time = max([fixationEvents.sttime]);
% Define the time window from 5 seconds to 10 seconds (5000 ms to 10000 ms)
% timeWindowStart = start_time +1000; % 5 seconds in milliseconds
% timeWindowEnd = start_time + 50000; % 10 seconds in milliseconds

timeWindowStart = start_time ; % 5 seconds in milliseconds
timeWindowEnd = end_time; % 10 seconds in milliseconds
% Filter fixation events within the time window
fixationsInTimeWindow = fixationEvents([fixationEvents.sttime] >= timeWindowStart & [fixationEvents.sttime] <= timeWindowEnd);
% Plot fixation points
for i = 1:length(fixationsInTimeWindow)
    fixation = fixationsInTimeWindow(i);
    plot(fixation.gstx, fixation.gsty, 'r.', 'MarkerSize', 10); % plot start of fixation
end
% % Filter saccade events within the time window
% saccadesInTimeWindow = saccadeEvents([saccadeEvents.sttime] >= timeWindowStart & [saccadeEvents.sttime] <= timeWindowEnd);
% 
% % Plot saccade points
% for i = 1:length(saccadesInTimeWindow)
%     saccade = saccadesInTimeWindow(i);
%     plot([saccade.gstx, saccade.genx], [saccade.gsty, saccade.geny], 'b-'); % plot saccade
% end
hold off;