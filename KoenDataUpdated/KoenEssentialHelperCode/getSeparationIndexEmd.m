function [p_value, perm_emd, real_emd] = getSeparationIndexEmd(data, num_permutations, plotResults)
    % analyzeNeuronDistribution
    % Analyzes the distribution of neurons in a 2D space and performs a permutation test
    % to check if the count in the top right quadrant is lower than expected by chance.
    % Optionally, plots the permutation distribution and the scatter plot with highlighted top right quadrant.
    %
    % Input:
    % data - Nx2 matrix of neuron data
    % num_permutations - Number of permutations for the permutation test
    % plotResults (optional) - Boolean to control whether to plot results
    %
    % Outputs:
    % p_value - The p-value from the permutation test
    % permutation_matrix - Matrix of counts from the permutation test
    % real_emd - The count of neurons in the top right quadrant in the actual data

    % for reproducibility
    rng(1)

    if nargin < 4
        plotResults = false;
    end
    
    % Identify the top right square
    real_emd = getEmd(data(:,1), data(:,2));

%     % Permutation Test
%     perm_emd = zeros(num_permutations, 1);
%     for i = 1:num_permutations
%         rand_x = randsample(data(:,1), size(data, 1));
%         rand_y = randsample(data(:,2), size(data, 1));
%         perm_emd(i) = getEmd(rand_x, rand_y);
%     end


    numCells = length(data);
    sortX = sort(data(:,1));
    sortY = sort(data(:,2));

    % Permutation Test
    perm_emd = zeros(num_permutations, 1);
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
        perm_emd(i) = getEmd(uniform_x, uniform_y);
    end

%     % Calculate p-value and isolation index
%     extreme_lower = sum(perm_emd <= real_emd);
%     extreme_higher = sum(perm_emd >= real_emd);
%     p_value = (min(extreme_lower, extreme_higher)) / num_permutations;

    p_value = paretoEstBi(perm_emd/size(data,1), real_emd/size(data,1));

    if isempty(p_value)
        p_value = 0;
    end
    
%     permutation_matrix = perm_emd;
%     separation_index = 1 - (real_emd / size(data, 1)) / (mean(perm_emd) / size(data, 1));

    % Optional Plot
    if plotResults
        figure('Position',[424         392        1265         420])

        % Permutation Distribution
        histogram(perm_emd, 'Normalization', 'probability');
        xline(real_emd, 'r', 'LineWidth', 2);
%         text(max(xlim)*0.6, max(ylim)*0.75, sprintf('Real Emd: %d\np-value: %.3f\nIsolation Index: %.3f', real_emd, p_value, separation_index), 'VerticalAlignment', 'bottom');
        text(max(xlim)*0.6, max(ylim)*0.75, sprintf('Real Emd: %d\np-value: %.3f', real_emd, p_value), 'VerticalAlignment', 'bottom');
        title('Permutation Distribution');
        xlabel('Count in Top Right Quadrant');
        ylabel('Probability');

    end
end
