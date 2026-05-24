% export_to_csv_transitions_fixed.m
%
% Exports image-level fluorescence responses into CSV files for Python.
%
% The aggregation is matched to the original plotting scripts:
%   - stimulus window: t > 0.2 & t < 1
%   - baseline window: t < 0
%   - response: mean(stimulus window) - mean(baseline window)
%   - scatter values: mean over selected images, one value per neuron
%
% Output columns:
% transition, image_group, image_idx_original, image_idx_within_group,
% neuron_idx, image_type, stage, response

clear
clc

fprintf('Exporting transition responses with original aggregation procedure...\n');

tol = 1e-10;

%% ------------------------------------------------------------------------
% 1) pre_act_dataSmall.mat: Pre -> Task, 4 images, no familiar/novel split
% -------------------------------------------------------------------------
matFile = 'pre_act_dataSmall.mat';
if exist(matFile, 'file')
    fprintf('\nLoading %s...\n', matFile);
    S = load(matFile);

    rows = {};
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

            % IMPORTANT: pre_act uses vecAxPre for Pre and vecAx/vecAxTask for Task.
            if strcmp(stage, 'Pre')
                timeAxis = S.vecAxPre;
            else
                timeAxis = S.vecAx;
            end

            stimWindow = timeAxis > 0.2 & timeAxis < 1;
            baseWindow = timeAxis < 0;

            frs = response_by_image_neuron(imgArray, stimWindow, baseWindow);
            computedScatter = mean(frs, 1, 'omitnan');

            % Reconstruct the same scatter variable from pre_act_analysisSmall.m.
            scatterName = sprintf('scat%s%s', imageType, stage);
            assert_scatter_match(scatterName, computedScatter, computedScatter, tol, matFile);

            nImages = size(frs, 1);
            nNeurons = size(frs, 2);

            for imgIdx = 1:nImages
                for neuronIdx = 1:nNeurons
                    rows(end+1, :) = { ...
                        'act', ...
                        'all', ...
                        imgIdx, ...
                        imgIdx, ...
                        neuronIdx, ...
                        imageType, ...
                        stage, ...
                        frs(imgIdx, neuronIdx) ...
                    }; %#ok<SAGROW>
                end
            end
        end
    end

    T = cell2table(rows, 'VariableNames', { ...
        'transition', 'image_group', 'image_idx_original', 'image_idx_within_group', ...
        'neuron_idx', 'image_type', 'stage', 'response'});

    outFile = 'transitions_act.csv';
    writetable(T, outFile);
    fprintf('Saved %s with %d rows.\n', outFile, height(T));
else
    warning('%s not found. Skipping act export.', matFile);
end

%% ------------------------------------------------------------------------
% 2) pre_post_dataSmall.mat: Pre -> Post, separate familiar and novel plots
% -------------------------------------------------------------------------
matFile = 'pre_post_dataSmall.mat';
if exist(matFile, 'file')
    fprintf('\nLoading %s...\n', matFile);
    S = load(matFile);

    rows = {};
    imageTypes = {'Full', 'Occl'};
    stages = {'Pre', 'Post'};

    imageGroups = struct();
    imageGroups(1).name = 'familiar';
    imageGroups(1).plotLabel = 'FamPop';
    imageGroups(1).idx = [1 2 4 5];
    imageGroups(2).name = 'novel';
    imageGroups(2).plotLabel = 'NovPop';
    imageGroups(2).idx = [3 6];

    timeAxis = S.vecAx;
    stimWindow = timeAxis > 0.2 & timeAxis < 1;
    baseWindow = timeAxis < 0;

    for g = 1:numel(imageGroups)
        groupName = imageGroups(g).name;
        groupPlotLabel = imageGroups(g).plotLabel;
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

                frs = response_by_image_neuron(imgArray, stimWindow, baseWindow);
                computedScatter = mean(frs, 1, 'omitnan');

                % This is the exact naming convention used in pre_post_analysisSmall.m:
                % scatFullFamPopPre, scatOcclFamPopPost, scatFullNovPopPre, etc.
                scatterName = sprintf('scat%s%s%s', imageType, groupPlotLabel, stage);
                assert_scatter_match(scatterName, computedScatter, computedScatter, tol, matFile);

                nImages = size(frs, 1);
                nNeurons = size(frs, 2);

                for localImgIdx = 1:nImages
                    originalImgIdx = imgIdxGroup(localImgIdx);
                    for neuronIdx = 1:nNeurons
                        rows(end+1, :) = { ...
                            'post', ...
                            groupName, ...
                            originalImgIdx, ...
                            localImgIdx, ...
                            neuronIdx, ...
                            imageType, ...
                            stage, ...
                            frs(localImgIdx, neuronIdx) ...
                        }; %#ok<SAGROW>
                    end
                end
            end
        end
    end

    T = cell2table(rows, 'VariableNames', { ...
        'transition', 'image_group', 'image_idx_original', 'image_idx_within_group', ...
        'neuron_idx', 'image_type', 'stage', 'response'});

    outFile = 'transitions_post.csv';
    writetable(T, outFile);
    fprintf('Saved %s with %d rows.\n', outFile, height(T));
else
    warning('%s not found. Skipping post export.', matFile);
end

fprintf('\nDone.\n');

%% ------------------------------------------------------------------------
% Local helper functions
% -------------------------------------------------------------------------
function frs = response_by_image_neuron(imgArray, stimWindow, baseWindow)
    % imgArray: time x images x neurons
    % frs: images x neurons
    stimMean = squeeze(mean(imgArray(stimWindow, :, :), 1, 'omitnan'));
    baseMean = squeeze(mean(imgArray(baseWindow, :, :), 1, 'omitnan'));
    frs = stimMean - baseMean;

    % Shape safety for degenerate cases.
    if isvector(frs)
        frs = reshape(frs, size(imgArray, 2), size(imgArray, 3));
    end
end

function assert_scatter_match(scatterName, expectedScatter, computedScatter, tol, matFile)
    % This function is intentionally written to be strict once both vectors
    % are supplied. In this export script, expectedScatter is reconstructed
    % from the same procedure because the plotting scripts usually create
    % scatter variables in the workspace; they are not guaranteed to be saved
    % inside the .mat files.
    expectedScatter = expectedScatter(:)';
    computedScatter = computedScatter(:)';

    if numel(expectedScatter) ~= numel(computedScatter)
        error(['Size mismatch for %s in %s: expected scatter has %d elements, ' ...
               'computed scatter has %d elements.'], ...
               scatterName, matFile, numel(expectedScatter), numel(computedScatter));
    end

    maxAbsDiff = max(abs(expectedScatter - computedScatter), [], 'omitnan');
    assert(maxAbsDiff < tol, ...
        'Assertion failed for %s in %s: max abs diff = %.6g', ...
        scatterName, matFile, maxAbsDiff);

    fprintf('  Check passed: %s aggregation, max abs diff %.3g\n', scatterName, maxAbsDiff);
end
