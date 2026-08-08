% export_to_csv_traces.m
%
% Exports image-level fluorescence traces into CSV files for Python.
%
% The labels and grouping match export_to_csv_transitions_fixed.m:
%   - pre_act_dataSmall.mat: image_group = all, stages Pre/Task
%   - pre_post_dataSmall.mat: image_group = familiar/novel, stages Pre/Post
%   - image_type = Full/Occl
%
% Output columns:
% image_group, image_idx_original, image_idx_within_group, neuron_idx,
% image_type, stage, time, response

clear
clc

fprintf('Exporting transition traces with original image/stage labels...\n');

outputDir = fullfile('..', 'context_contrasting', 'data_analysis');
if ~exist(outputDir, 'dir')
    outputDir = '.';
end

%% ------------------------------------------------------------------------
% 1) pre_act_dataSmall.mat: Pre -> Task, 4 images, no familiar/novel split
% -------------------------------------------------------------------------
matFile = 'pre_act_dataSmall.mat';
if exist(matFile, 'file')
    fprintf('\nLoading %s...\n', matFile);
    S = load(matFile);

    tables = {};
    imageTypes = {'Full', 'Occl'};
    stages = {'Pre', 'Task'};

    for no = 1:numel(imageTypes)
        imageType = imageTypes{no};

        for st = 1:numel(stages)
            stage = stages{st};

            arrayName = sprintf('img%sResMnPop%s', imageType, stage);
            if ~isfield(S, arrayName)
                error('Variable %s not found in %s.', arrayName, matFile);
            end

            imgArray = S.(arrayName);  % time x images x neurons

            % IMPORTANT: pre_act uses vecAxPre for Pre and vecAx for Task.
            if strcmp(stage, 'Pre')
                timeAxis = S.vecAxPre;
            else
                timeAxis = S.vecAx;
            end

            baseWindow = timeAxis < 0;
            traces = baseline_subtracted_traces(imgArray, baseWindow);

            tables{end+1} = trace_table( ...
                traces, ...
                timeAxis, ...
                'all', ...
                1:size(traces, 2), ...
                1:size(traces, 2), ...
                imageType, ...
                stage ...
            ); %#ok<SAGROW>
        end
    end

    T = vertcat(tables{:});
    outFile = fullfile(outputDir, 'transitions_act_traces.csv');
    writetable(T, outFile);
    fprintf('Saved %s with %d rows.\n', outFile, height(T));
else
    warning('%s not found. Skipping act trace export.', matFile);
end

%% ------------------------------------------------------------------------
% 2) pre_post_dataSmall.mat: Pre -> Post, separate familiar and novel plots
% -------------------------------------------------------------------------
matFile = 'pre_post_dataSmall.mat';
if exist(matFile, 'file')
    fprintf('\nLoading %s...\n', matFile);
    S = load(matFile);

    tables = {};
    imageTypes = {'Full', 'Occl'};
    stages = {'Pre', 'Post'};

    imageGroups = struct();
    imageGroups(1).name = 'familiar';
    imageGroups(1).idx = [1 2 4 5];
    imageGroups(2).name = 'novel';
    imageGroups(2).idx = [3 6];

    timeAxis = S.vecAx;
    baseWindow = timeAxis < 0;

    for g = 1:numel(imageGroups)
        groupName = imageGroups(g).name;
        imgIdxGroup = imageGroups(g).idx;

        for no = 1:numel(imageTypes)
            imageType = imageTypes{no};

            for st = 1:numel(stages)
                stage = stages{st};

                arrayName = sprintf('img%sResMnPop%s', imageType, stage);
                if ~isfield(S, arrayName)
                    error('Variable %s not found in %s.', arrayName, matFile);
                end

                imgArrayAll = S.(arrayName);  % time x 6 images x neurons
                imgArray = imgArrayAll(:, imgIdxGroup, :);
                traces = baseline_subtracted_traces(imgArray, baseWindow);

                tables{end+1} = trace_table( ...
                    traces, ...
                    timeAxis, ...
                    groupName, ...
                    imgIdxGroup, ...
                    1:size(traces, 2), ...
                    imageType, ...
                    stage ...
                ); %#ok<SAGROW>
            end
        end
    end

    T = vertcat(tables{:});
    outFile = fullfile(outputDir, 'transitions_post_traces.csv');
    writetable(T, outFile);
    fprintf('Saved %s with %d rows.\n', outFile, height(T));
else
    warning('%s not found. Skipping post trace export.', matFile);
end

fprintf('\nDone.\n');

%% ------------------------------------------------------------------------
% Local helper functions
% -------------------------------------------------------------------------
function traces = baseline_subtracted_traces(imgArray, baseWindow)
    % imgArray: time x images x neurons
    baseline = mean(imgArray(baseWindow, :, :), 1, 'omitnan');
    traces = imgArray - baseline;
end

function T = trace_table(traces, timeAxis, groupName, originalImgIdx, withinImgIdx, imageType, stage)
    % traces: time x images x neurons
    timeAxis = timeAxis(:);
    originalImgIdx = originalImgIdx(:);
    withinImgIdx = withinImgIdx(:);
    nTime = size(traces, 1);
    nImages = size(traces, 2);
    nNeurons = size(traces, 3);

    if numel(timeAxis) ~= nTime
        error('timeAxis length (%d) does not match trace time dimension (%d).', numel(timeAxis), nTime);
    end
    if numel(originalImgIdx) ~= nImages || numel(withinImgIdx) ~= nImages
        error('Image index vectors must match the trace image dimension (%d).', nImages);
    end

    [timeGrid, imageGrid, neuronGrid] = ndgrid(1:nTime, 1:nImages, 1:nNeurons);

    T = table( ...
        repmat({groupName}, numel(traces), 1), ...
        originalImgIdx(imageGrid(:)), ...
        withinImgIdx(imageGrid(:)), ...
        neuronGrid(:), ...
        repmat({imageType}, numel(traces), 1), ...
        repmat({stage}, numel(traces), 1), ...
        timeAxis(timeGrid(:)), ...
        traces(:), ...
        'VariableNames', { ...
            'image_group', ...
            'image_idx_original', ...
            'image_idx_within_group', ...
            'neuron_idx', ...
            'image_type', ...
            'stage', ...
            'time', ...
            'response' ...
        } ...
    );
end
