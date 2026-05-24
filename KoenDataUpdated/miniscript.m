
% export_to_csv_transitions.m
%
% Exports fluorescence responses for Full/Occl x Pre/Task transitions
% into CSV files usable from Python.
%
% Output CSV columns:
% condition, image_idx, neuron_idx, image_type, stage, response

clear
clc

% Conditions/files to process
conditions = {'act', 'post'};

% Response types
imageTypes = {'Full', 'Occl'};

% Stages
Stages = {{'Pre', 'Task'}, {'Pre', 'Post'}};

tol = 1e-10;

for te = 1:numel(conditions)

    condition = conditions{te};
    stages = Stages{te};
    matFile = sprintf('pre_%s_dataSmall.mat', condition);

    fprintf('Loading %s...\n', matFile);
    S = load(matFile);

    trial = S.vecAx >= 0 & S.vecAx <= 1;

    rows = {};

    for no = 1:numel(imageTypes)

        imageType = imageTypes{no};

        for st = 1:numel(stages)

            stage = stages{st};

            arrayName = sprintf('img%sResMnPop%s', imageType, stage);

            if ~isfield(S, arrayName)
                error('Variable %s not found in %s', arrayName, matFile);
            end

            imgArray = S.(arrayName);

            % time x images x neurons -> images x neurons
            frs = squeeze(mean(imgArray(trial, :, :), 1, 'omitnan'));

            if isvector(frs)
                frs = reshape(frs, size(imgArray, 2), size(imgArray, 3));
            end

            % ------------------------------------------------------------
            % Assert/check against loaded scatter variable, if available
            % ------------------------------------------------------------
            %
            % Expected scatter variable names:
            % scatFullPre, scatFullTask, scatOcclPre, scatOcclTask
            %
            % These should equal mean(frs, 1), i.e. mean over images,
            % producing one response value per neuron.
            %
            scatterName = sprintf('scat%s%sSort', imageType, stage);

            if isfield(S, scatterName)

                expectedScatter = S.(scatterName);
                expectedScatter = expectedScatter(:)';  % force row vector

                computedScatter = mean(frs, 1, 'omitnan');
                computedScatter = computedScatter(:)';  % force row vector

                if numel(expectedScatter) ~= numel(computedScatter)
                    error(['Size mismatch for %s in %s: loaded scatter has %d elements, ' ...
                           'computed scatter has %d elements.'], ...
                           scatterName, matFile, numel(expectedScatter), numel(computedScatter));
                end

                maxAbsDiff = max(abs(expectedScatter - computedScatter), [], 'omitnan');

                assert(maxAbsDiff < tol, ...
                    'Assertion failed for %s in %s: max abs diff = %.6g', ...
                    scatterName, matFile, maxAbsDiff);

                fprintf('  Check passed: %s matches mean(%s over images), max abs diff %.3g\n', ...
                    scatterName, arrayName, maxAbsDiff);

            else
                warning('Variable %s not found in %s; skipping assertion.', scatterName, matFile);
            end

            nImages = size(frs, 1);
            nNeurons = size(frs, 2);

            for imgIdx = 1:nImages
                for neuronIdx = 1:nNeurons
                    rows(end+1, :) = { ...
                        condition, ...
                        imgIdx, ...
                        neuronIdx, ...
                        imageType, ...
                        stage, ...
                        frs(imgIdx, neuronIdx) ...
                    };
                end
            end
        end
    end

    T = cell2table(rows, ...
        'VariableNames', { ...
            'condition', ...
            'image_idx', ...
            'neuron_idx', ...
            'image_type', ...
            'stage', ...
            'response' ...
        });

    uniqueImages = unique(T.image_idx);

    if strcmp(condition, 'act')
        expectedNImages = 4;
    elseif strcmp(condition, 'post')
        expectedNImages = 6;
    else
        expectedNImages = NaN;
    end

    if ~isnan(expectedNImages) && numel(uniqueImages) ~= expectedNImages
        warning('For condition %s, expected %d images but found %d.', ...
            condition, expectedNImages, numel(uniqueImages));
    end

    outFile = sprintf('transitions_%s.csv', condition);
    writetable(T, outFile);

    fprintf('Saved %s with %d rows.\n', outFile, height(T));

end