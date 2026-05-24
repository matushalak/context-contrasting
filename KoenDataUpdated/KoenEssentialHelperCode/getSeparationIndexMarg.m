function [p_value, permutation_matrix, real_count, separation_index] = getSeparationIndexMarg(data, num_permutations, stdMult, plotResults)
    % analyzeNeuronDistribution
    % Analyzes the distribution of neurons in a 2D space and performs a permutation test
    % to check if the count in the top right quadrant is lower than expected by chance (i.e. a marginal distribution).
    % Optionally, plots the permutation distribution and the scatter plot with highlighted top right quadrant.
    %
    % Input:
    % data - Nx2 matrix of neuron data
    % num_permutations - Number of permutations for the permutation test
    % numSquares - Number of squares along each axis (default: 3)
    % plotResults (optional) - Boolean to control whether to plot results
    %
    % Outputs:
    % p_value - The p-value from the permutation test
    % permutation_matrix - Matrix of counts from the permutation test
    % real_count - The count of neurons in the top right quadrant in the actual data
    % isolation_index - The isolation index based on the real and permutation data

    % for reproducibility
    rng(1)

    if nargin < 4
        plotResults = false;
    end

    % define edges for top right square
    dataStd = std(data, 0, 1); % calculate std
%     stdMult = 1; % std multiplier for edge definition
    x_edge = mean(data(:,1))+dataStd(1)*stdMult;
    y_edge = mean(data(:,2))+dataStd(2)*stdMult;
%     z = stdMult;
% 
%     x_edge = 0.5;
%     y_edge = 0.5;
    
    % Identify the top right square
    top_right = data(:,1) > x_edge & data(:,2) > y_edge;
    real_count = sum(top_right);

    numCells = length(data);
    sortX = sort(data(:,1));
    sortY = sort(data(:,2));

    % Permutation Test
    perm_counts = zeros(num_permutations, 1);
    for i = 1:num_permutations
%         rand_x = randsample(data(:,1), size(data, 1));
%         rand_y = randsample(data(:,2), size(data, 1));
        
        uniform_x = zeros(numCells,1);
        uniform_y = zeros(numCells,1);

        % for loop to find uniformly distributed values in our vectors
        for j = 1:numCells
        
            % first for NO (x axis)
            randVal = rand*(numCells-1); % random value between 0-1 as index to find position on x axis (no),
            randValInt = floor(randVal)+1; % we consider the first data point as 0, so we make it 1 to use as index
            x1 = sortX(randValInt); % x value for floored integer of randVal
            x2 = sortX(randValInt+1); % x value for one point higher
            difX12 = x2-x1; % difference
            uniform_x(j) = x1 + difX12*(randVal-(floor(randVal))); % adjusted x value
        
            % then for O (y axis)
            randVal = rand*(numCells-1); % random value between 0-1 as index to find position on x axis (no),
            randValInt = floor(randVal)+1; % we consider the first data point as 0, so we make it 1 to use as index
            x1 = sortY(randValInt); % x value for floored integer of randVal
            x2 = sortY(randValInt+1); % x value for one point higher
            difX12 = x2-x1; % difference
            uniform_y(j) = x1 + difX12*(randVal-(floor(randVal))); % adjusted x value
        end

        % count values in top right quadrant
        perm_counts(i) = sum(uniform_x > x_edge & uniform_y > y_edge);
    end

    % Calculate p-value and isolation index
    extreme_lower = sum(perm_counts <= real_count);
    extreme_higher = sum(perm_counts >= real_count);
%     p_value = (min(extreme_lower, extreme_higher)) / num_permutations;

    p_value = paretoEstBi(perm_counts/size(data,1), real_count/size(data,1));

    if isempty(p_value)
        p_value = 0;
    end
    
    permutation_matrix = perm_counts;
    separation_index = 1 - (real_count / size(data, 1)) / (mean(perm_counts) / size(data, 1));

    % Optional Plot
    if plotResults
        figure('Position',[424         392        1265         420])

        % Subplot 1: Permutation Distribution
        subplot(1, 2, 1);
        histogram(perm_counts, 'Normalization', 'probability');
        xline(real_count, 'r', 'LineWidth', 2);
        text(max(xlim)*0.6, max(ylim)*0.75, sprintf('Real Count: %d\np-value: %.3f\nIsolation Index: %.3f', real_count, p_value, separation_index), 'VerticalAlignment', 'bottom');
        title('Permutation Distribution');
        xlabel('Count in Top Right Quadrant');
        ylabel('Probability');

        % Subplot 2: Scatter Plot with Highlighted Top Right Quadrant
        subplot(1, 2, 2);
        scatter(data(~top_right,1), data(~top_right,2), 'filled');
        hold on;
        scatter(data(top_right,1), data(top_right,2), 'filled', 'r');
        refline(1)
        for edge = x_edge
            line([edge, edge], ylim, 'Color', 'k', 'LineStyle', '--');
        end
        for edge = y_edge
            line(xlim, [edge, edge], 'Color', 'k', 'LineStyle', '--');
        end
        title('Neuron Distribution with Division Lines');
        xlabel('Nonoccluded');
        ylabel('Occluded');
        hold off;
    end
end
