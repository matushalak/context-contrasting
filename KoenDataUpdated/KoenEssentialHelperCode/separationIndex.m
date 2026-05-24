function [dblRealMean, vecPermMean, upperRightCount, upperRightCountPerm ,pValue] = separationIndex(x, y, nPerms, enablePlotting)
% separationIndex calculates to what degree x and y are separated in space.
% It computes the real median, permutation median, and the p-value. Optionally,
% it can plot the results based on the 'enablePlotting' flag.
%
% Inputs:
%   x - An array of data points representing the first set of values.
%   y - An array of data points representing the second set of values.
%   nPerms - An integer indicating the number of permutations for the test.
%   enablePlotting - (optional) A boolean flag to enable/disable plotting.
%
% Outputs:
%   realMedian - The median of the sorted means where the count in the upper right
%                quadrant is less than the overall median count. This provides a measure
%                of central tendency in the spatial separation of x and y.
%   permMedian - An array of medians from each permutation iteration, representing
%                the distribution of medians under the null hypothesis.
%   pValue - The p-value quantifying the probability of observing a median as extreme
%            as the real median under the null hypothesis.

if nargin < 4
    enablePlotting = false;
end

% Calculate the mean of x and y
meanValues = (x + y) / 2;

% Sort the mean values
[sortedMeans, ~] = sort(meanValues, 'ascend');
sortedMeans = sortedMeans'; % transpose data

% Initialize the count of points in the upper right quadrant
upperRightCount = zeros(length(sortedMeans), 1);

% For each mean value, from low to high, we count the nr of points to the
% top right of that value
for j = 1:length(sortedMeans)
    upperRightCount(j) = sum(x > sortedMeans(j) & y > sortedMeans(j));
end

if enablePlotting
    figure;
    title('Upper Right Quadrant Count vs. Sorted Mean Values');
    xlabel('Sorted Mean Values');
    ylabel('Count');
end

% % Calculate the real median
% realMedian = sortedMeans(find(upperRightCount < median(upperRightCount), 1));
y_m = (upperRightCount(1:(end-1)) + upperRightCount(2:end))/2;
x_m = (sortedMeans(1:(end-1)) + sortedMeans(2:end))/2;
dblRealMean = sum(y_m .* x_m)/sum(y_m);

% Permutation analysis
vecPermMean = zeros(nPerms, 1);
upperRightCountPerm = zeros(length(sortedMeans), 1);

for k = 1:nPerms
    xR = randsample(x, length(x));
    yR = randsample(y, length(y));
    meanPerm = (xR + yR) / 2;
    [sortedPerm, ~] = sort(meanPerm, 'ascend');
    sortedPerm = sortedPerm'; % transpose data
    
    for j = 1:length(sortedMeans)
        upperRightCountPerm(j) = sum(xR > sortedPerm(j) & yR > sortedPerm(j));
    end

    if enablePlotting
        plot(sortedPerm, upperRightCountPerm, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.2); hold on
    end

%     permMedians(k) = sortedPerm(find(upperRightCountPerm(:, k) < median(upperRightCountPerm(:, k)), 1));

    y_m = (upperRightCountPerm(1:(end-1)) + upperRightCountPerm(2:end))/2;
    x_m = (sortedPerm(1:(end-1)) + sortedPerm(2:end))/2;
    vecPermMean(k) = sum(y_m .* x_m)/sum(y_m);

end

% plot the real mean
if enablePlotting
    plot(sortedMeans, upperRightCount, 'r', 'LineWidth', 2);
end

% Calculate the p-value
pValue = paretoEstBi(vecPermMean, dblRealMean);

end
