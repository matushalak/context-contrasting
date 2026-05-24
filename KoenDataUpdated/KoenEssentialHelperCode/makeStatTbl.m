% Function that compares group 1-2, 1-3 and 2-3 using coefTest on an LME
% INPUT
% lmeModel: linear mixed effects model 
% OUTPUT
% statTbl: table containing p-values (row 1) and F-statistics (row 2) for each comparison 
% Koen Seignette, 20230206

% function statTbl = makeStatTbl(lmeModel)
% 
% statTbl(1).onetwo = coefTest(lmeModel,[0 1 0])*2; % p-value corrected for 3 mult comparisons (Tukey)
% [~, statTbl(2).onetwo] = coefTest(lmeModel,[0 1 0]);
% 
% statTbl(1).onethree = coefTest(lmeModel,[0 0 1])*2; % p-value corrected for 3 mult comparisons (Tukey)
% [~, statTbl(2).onethree] = coefTest(lmeModel,[0 0 1]);
% 
% statTbl(1).twothree = coefTest(lmeModel,[0 1 -1])*2; % p-value corrected for 3 mult comparisons (Tukey)
% [~, statTbl(2).twothree] = coefTest(lmeModel,[0 1 -1]);
% 
% end

function statTbl = makeStatTbl(lmeModel)
% Post hoc pairwise contrasts from LMEM with 3-level condition factor

% Pre vs Post: [0 1 0]
p_raw = coefTest(lmeModel, [0 1 0]);
statTbl(1).comparison = 'Pre vs Post';
statTbl(1).p_uncorrected = p_raw;
statTbl(1).p_corrected = min(p_raw * 2, 1);

% Pre vs Task: [0 0 1]
p_raw = coefTest(lmeModel, [0 0 1]);
statTbl(2).comparison = 'Pre vs Task';
statTbl(2).p_uncorrected = p_raw;
statTbl(2).p_corrected = min(p_raw * 2, 1);

% Post vs Task: [0 1 -1]
p_raw = coefTest(lmeModel, [0 1 -1]);
statTbl(3).comparison = 'Post vs Task';
statTbl(3).p_uncorrected = p_raw;
statTbl(3).p_corrected = min(p_raw * 2, 1);
end
