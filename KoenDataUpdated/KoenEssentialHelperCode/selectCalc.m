function selectivity = selectCalc(dataMat)
% calculates selectivity of responses to different stimuli
% Koen Seignette - 13-01-2022
% INPUT:
% dataMat; mean response per stimulus x ROIs (e.g. 4 x 100)
% OUTPUT:
% selectivity; selectivity index per ROI

nStims = size(dataMat,1); % nr of stimuli
nRois = size(dataMat,2); % nr of ROIs
top = zeros(nStims,1); % top of division
bottom = zeros(nStims,1); % bottom of division
selectivity = zeros(nRois,1); % pre-allocate selectivity vector
% calculate selectivity per ROI
for j = 1:nRois % ROIs
    for i = 1:nStims% stimuli
        top(i) = abs(dataMat(i,j))/nStims;
        bottom(i) = (dataMat(i,j)^2)/nStims;
    end
    topSum = sum(top)^2;
    bottomSum = sum(bottom);
    selectivity(j) = 1-(topSum/bottomSum); % selectivity value
end

end