function edge = findDropPoint(axis_values, density)
    % Find the point where there's a significant drop in density

    % Calculate the gradient (rate of change) of the density
    gradient_density = diff(density);

    % Find the point where the gradient changes significantly
    [~, idx] = min(gradient_density);

    % Corresponding axis value where the significant drop occurs
    edge = axis_values(idx);
end