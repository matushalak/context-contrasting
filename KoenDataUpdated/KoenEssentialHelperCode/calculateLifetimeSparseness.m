function sparseness = calculateLifetimeSparseness(responses)
    % CALCULATELIFETIMESPARSENESS calculates the lifetime sparseness of neuronal responses.
    %
    % Lifetime sparseness is a measure that indicates how selectively a neuron 
    % responds to a set of stimuli. It is calculated in such a way that a neuron 
    % responding equally to all stimuli will have a low sparseness value, while 
    % a neuron responding strongly to only one stimulus and weakly to others 
    % will have a high sparseness value.
    %
    % The lifetime sparseness (L) is calculated using the formula:
    % L = 1 - ((ΣRi)^2 / (N * ΣRi^2))
    % where Ri is the response to the i-th stimulus, and N is the total number of stimuli.
    %
    % Input:
    %   responses - A matrix where each row represents a neuron and each
    %               column represents a different stimulus. Each element in
    %               the matrix is the response of a neuron to a particular stimulus.
    %
    % Output:
    %   sparseness - A column vector where each element is the lifetime sparseness
    %                of the corresponding neuron. Higher values indicate more
    %                selective (sparse) responses to the set of stimuli.
    % 
    % Koen Seignette & chatGPT 4.0, 2024-01-04

    % Ensure that responses is a matrix
    if ~ismatrix(responses)
        error('Input must be a matrix where rows represent neurons and columns represent stimuli.');
    end

    % Calculate the number of stimuli (columns of the responses matrix)
    N = size(responses, 2);

    % Sum of responses for each neuron across all stimuli
    sumResponses = sum(responses, 2);

    % Sum of squared responses for each neuron
    sumSquaredResponses = sum(responses.^2, 2);

    % Calculate lifetime sparseness for each neuron
    sparseness = 1 - ((sumResponses.^2) ./ (N * sumSquaredResponses));
end
