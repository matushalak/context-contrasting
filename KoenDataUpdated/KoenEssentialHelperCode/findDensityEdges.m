function [x_edge, y_edge] = findDensityEdges(data)
    % findDensityEdges
    % Finds the x and y edges based on significant drops in density
    %
    % Input:
    % data - Nx2 matrix of neuron data
    %
    % Outputs:
    % x_edge - x value where a significant drop in density occurs
    % y_edge - y value where a significant drop in density occurs

    % Kernel Density Estimation
    [bandwidth, density, X, Y] = kde2d(data);

    % Sum the densities along x and y axes
    density_x = sum(density, 1); % Sum along y-axis (rows)
    density_y = sum(density, 2); % Sum along x-axis (columns)

    % Find the significant drop points in density
    x_edge = findDropPoint(X(1, :), density_x);
    y_edge = findDropPoint(Y(:, 1), density_y);

end

