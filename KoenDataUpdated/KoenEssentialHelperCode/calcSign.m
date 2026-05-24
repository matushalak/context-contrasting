function [dblRes, hVal] = calcSign(dataRes, vecSp, vecSt, alphaVal)
% calculates whether a response is significant over trials
% INPUT:
% dataRes; response traces (frames x trials).
% vecSp; vector of zeros and ones that defines the prestim window of frames
% vecSt; vector of zeros and ones that defines the stim window of frames
% OUTPUT:
% pVal; pval of the response
% dblRes; size of the average response over trials
% Koen Seignette, 2022-02-26

% [hVal] = ttest(mean(dataRes(vecSp,:)), mean(dataRes(vecSt,:)),'Alpha',alphaVal);
[hVal] = ttest(mean(dataRes(vecSp,:)), mean(dataRes(vecSt,:)),'Alpha', alphaVal);
% [~,hVal] = signrank(mean(dataRes(vecSp,:)), mean(dataRes(vecSt,:)),'Alpha', alphaVal);
dblRes = mean(mean(dataRes(vecSt,:))-mean(dataRes(vecSp,:)));

end
