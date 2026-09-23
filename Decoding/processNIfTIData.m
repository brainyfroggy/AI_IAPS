function reshapedData = processNIfTIData(mainpath, SubNum, folderName)
    % Define the directory containing NIfTI files
    Nt_dir = strcat(mainpath,'\Sub',num2str(SubNum),folderName);
    
    % List all NIfTI files in the directory
    Nt_files = dir(fullfile(Nt_dir, '*.nii'));
    Nt_files = Nt_files(~[Nt_files.isdir]); % Exclude directories
    
    % Concatenate folder path with file names
    fullFileNames = fullfile({Nt_files.folder}, {Nt_files.name});
    niiDataArray = cell(numel(fullFileNames), 1);
    
    % Loop through each file and load its data
    for i = 1:numel(fullFileNames)
        % Full file path for the current NIfTI file
        filePath = fullFileNames{i};
        
        % Read the NIfTI file and store its data
        niiData = niftiread(filePath);
        
        % Flatten the 3D array into a column vector
        flattenedData = reshape(niiData, [], 1);
        
        % Store the flattened data in the cell array
        niiDataArray{i} = flattenedData;
    end
    
    % Concatenate the flattened data to form a matrix
    reshapedData = cell2mat(niiDataArray');
end