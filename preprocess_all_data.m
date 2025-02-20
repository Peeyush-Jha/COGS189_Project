function preprocess_all_data()
    clear; clc; close all;
    
    % ----------- 1) Define the folder pairs (input -> output) -----------
    folderPairs = {
        'ADHD_part1',    'ADHD_part1_preprocessed';
        'ADHD_part2',    'ADHD_part2_preprocessed';
        'Control_part1', 'Control_part1_preprocessed';
        'Control_part2', 'Control_part2_preprocessed'
    };
    
    % ----------- 2) Define Preprocessing Parameters -----------
    fs = 128;               % Sampling frequency
    lowCut = 1;             % Bandpass lower cutoff (Hz)
    highCut = 40;           % Bandpass upper cutoff (Hz)
    filterOrder = 4;        % Order of the Butterworth filter
    artifactThreshold = 1000;  % Simple amplitude threshold (µV)
    
    % Some filters require >= 3*(2*filterOrder) samples to run filtfilt
    minLengthForFilter = 3 * (2 * filterOrder);
    
    % Design the Butterworth filter once
    [b, a] = butter(filterOrder, [lowCut, highCut] / (fs/2), 'bandpass');
    
    % ----------- 3) Loop Over All Folder Pairs -----------
    for iPair = 1:size(folderPairs,1)
        
        inputDir  = folderPairs{iPair, 1};
        outputDir = folderPairs{iPair, 2};
        
        % Create output directory if needed
        if ~exist(outputDir, 'dir')
            mkdir(outputDir);
        end
        
        % List all .mat files in the input directory
        fileList = dir(fullfile(inputDir, '*.mat'));
        fprintf('\n=== Processing folder: %s ===\n', inputDir);
        
        if isempty(fileList)
            warning('No .mat files found in: %s', inputDir);
            continue;
        end
        
        % 3A) Process Each File
        for iFile = 1:length(fileList)
            
            inFile = fileList(iFile).name;
            inPath = fullfile(fileList(iFile).folder, inFile);
            
            fprintf('\n  -> Processing file [%d/%d]: %s\n', ...
                iFile, length(fileList), inFile);
            
            % ----------- Load Data -----------
            dataStruct = load(inPath);
            fn = fieldnames(dataStruct);
            rawData = dataStruct.(fn{1});  % e.g. dataStruct.v1p
            
            % Ensure channels x time orientation
            [rows, cols] = size(rawData);
            if rows < cols
                % If we suspect it's time x channels, transpose so
                % it becomes channels x time
                rawData = rawData';
                [rows, cols] = size(rawData);
            end
            % Now rawData is (channels x time)
            channels   = rows;
            timePoints = cols;
            
            % ----------- Filter (if data is long enough) -----------
            if timePoints < minLengthForFilter
                % If the data is too short to safely run filtfilt,
                % we skip the filter step
                fprintf('    * WARNING: Data length (%d) < %d required. Skipping filter.\n', ...
                    timePoints, minLengthForFilter);
                filteredData = rawData;
            else
                % Apply filtfilt channel-by-channel
                filteredData = zeros(size(rawData));
                for ch = 1:channels
                    filteredData(ch,:) = filtfilt(b, a, rawData(ch,:));
                end
            end
            
            % ----------- Artifact Rejection by Threshold -----------
            artifactMask = abs(filteredData) > artifactThreshold;
            cleanedData = filteredData;
            cleanedData(artifactMask) = NaN;
            
            % ----------- Channel-wise Normalization -----------
            normalizedData = zeros(size(cleanedData));
            for ch = 1:channels
                chanData = cleanedData(ch,:);
                goodIdx  = ~isnan(chanData);
                
                if any(goodIdx)
                    mu       = mean(chanData(goodIdx));
                    sigmaVal = std(chanData(goodIdx));
                    if sigmaVal < 1e-10
                        sigmaVal = 1;  % avoid /0
                    end
                    normalizedData(ch,goodIdx) = (chanData(goodIdx) - mu) / sigmaVal;
                else
                    % If entire channel is NaN, remain zeros or NaNs
                    normalizedData(ch,:) = NaN;
                end
            end
            
            preprocessedData = normalizedData;
            
            % ----------- Save Output -----------
            [~, baseName] = fileparts(inFile);
            outFileName = [baseName, '_preprocessed.mat'];
            outPath = fullfile(outputDir, outFileName);
            
            save(outPath, 'preprocessedData', 'fs');
            fprintf('    -> Saved: %s\n', outFileName);
        end
        
        fprintf('=== Finished folder: %s ===\n', inputDir);
    end
    
    fprintf('\n*** All folders processed! ***\n');
end
