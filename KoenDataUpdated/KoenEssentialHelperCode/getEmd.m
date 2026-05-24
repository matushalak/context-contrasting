function fval = getEmd(x,y)

% function that gets the Emd using the matlab file exchange emd function
% https://nl.mathworks.com/matlabcentral/fileexchange/22962-the-earth-mover-s-distance
% input is x and y values of your 2D matrix (e.g. nonoccluded vs occluded
% reponses)
% output is the emd value


% Assuming x and y are your matrices of points
n = size(x, 1); % Number of points in x
m = size(y, 1); % Number of points in y

% Create equal weights for each point
W1 = ones(n, 1) / n;
W2 = ones(m, 1) / m;

% Use the EMD function
[~, fval] = emdist(x, y, W1, W2, @gdf);

% Display the EMD value
disp(fval);


end