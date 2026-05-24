%% LDA analysis on population data from muckli experiments pre vs post training
%
%	Version History:
%	2022-02-14	Created by Koen Seignette
%   2022-07-22 update on some significance calculations and average traces

clear all;
close all;
clc

%% 
% cd('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\')
load('pre_post_dataSmall.mat') % L23

%% Initialize and organise data


% imgNrs = [1 2 4 5]; % image nrs to decode, trained images
imgNrs = [1 2 3 4 5 6]; % image nrs to decode, all images
% imgNrs = [3 6]; % image nrs to decode, untrained images
% imgNrs = [4 5]; % image nrs to decode
nImgs = length(imgNrs);
performanceChance = 100/nImgs;
nTrials = 20; % nr of trials shown per image
trainFrac = 0.5; % on what fraction would you like to train the decoder (0.8 is good)
rfDistVec = 2; % Minimum distance away from occluder edge

vecAxSp = vecAx<0; % spontaneous activity window
vecAxSt = vecAx>0.2 & vecAx<1; % stim window
alphaVal = 0.99; % significance value for cells to be included
% zscoreVal = 0; % minimum zscore value for cells to be included
if nfiles == 6
    rsqThresh = 0.33; % 0.33 for L2/3
else
    rsqThresh = 0.15; % 0.15 for L5
end
% bratThresh = 1.5;
snrThresh = 4; % snr threshold for RF
useSpikingData = 0; % deconvolved (1) or df/f (0)
regressRun = false; % regress out running? Only for CaSigCorrected, not for spikes
doZscore = true; % in case you want to work with zscored data instead of dff



%% in case you calculated responses to 6 images, plot pre vs post (separate by image type)
% color pallets for plotting
col1 = [0,0,0]; % black
col2 = [131, 197, 190]/255; % blue/greenish

save_fig = false;

famIdx = [1 2 4 5];
% famIdx = [1 2];
% famIdx = [1 2 3 4 5 6]; % INCLUDING NOVEL IMAGES IN SELECTION
novIdx = [3 6];

imgFullResFamPre = squeeze(mean(imgFullResMnPopPre(:,famIdx,:),2));
imgFullResFamPreBsl = imgFullResFamPre-mean(imgFullResFamPre(vecAxSp,:));
imgFullResNovPre = squeeze(mean(imgFullResMnPopPre(:,novIdx,:),2));
imgFullResNovPreBsl = imgFullResNovPre-mean(imgFullResNovPre(vecAxSp,:));
imgOcclResFamPre = squeeze(mean(imgOcclResMnPopPre(:,famIdx,:),2));
imgOcclResFamPreBsl = imgOcclResFamPre-mean(imgOcclResFamPre(vecAxSp,:));
imgOcclResNovPre = squeeze(mean(imgOcclResMnPopPre(:,novIdx,:),2));
imgOcclResNovPreBsl = imgOcclResNovPre-mean(imgOcclResNovPre(vecAxSp,:));

imgFullResFamPost = squeeze(mean(imgFullResMnPopPost(:,famIdx,:),2));
imgFullResFamPostBsl = imgFullResFamPost-mean(imgFullResFamPost(vecAxSp,:));
imgFullResNovPost = squeeze(mean(imgFullResMnPopPost(:,novIdx,:),2));
imgFullResNovPostBsl = imgFullResNovPost-mean(imgFullResNovPost(vecAxSp,:));
imgOcclResFamPost = squeeze(mean(imgOcclResMnPopPost(:,famIdx,:),2));
imgOcclResFamPostBsl = imgOcclResFamPost-mean(imgOcclResFamPost(vecAxSp,:));
imgOcclResNovPost = squeeze(mean(imgOcclResMnPopPost(:,novIdx,:),2));
imgOcclResNovPostBsl = imgOcclResNovPost-mean(imgOcclResNovPost(vecAxSp,:));

scatFullFamPopPre = mean(imgFullResFamPreBsl(vecAxSt,:));
scatFullNovPopPre = mean(imgFullResNovPreBsl(vecAxSt,:));
scatOcclFamPopPre = mean(imgOcclResFamPreBsl(vecAxSt,:));
scatOcclNovPopPre = mean(imgOcclResNovPreBsl(vecAxSt,:));
scatFullFamPopPost = mean(imgFullResFamPostBsl(vecAxSt,:));
scatFullNovPopPost = mean(imgFullResNovPostBsl(vecAxSt,:));
scatOcclFamPopPost = mean(imgOcclResFamPostBsl(vecAxSt,:));
scatOcclNovPopPost = mean(imgOcclResNovPostBsl(vecAxSt,:));

sz = 10;
cPre = [0.2 0.2 0.2];
cPost = col2;

% plot traces and scatters in one figure
figure('Position', [101          97        1482         836])
clear t s g
% traces
t(1) = subplot(3,5,1);
shadedErrorBar(vecAx,mean(imgFullResFamPreBsl,2)...
    ,std(imgFullResFamPreBsl,0,2)/sqrt(size(imgFullResFamPreBsl,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAx,mean(imgOcclResFamPreBsl,2)...
    ,std(imgOcclResFamPreBsl,0,2)/sqrt(size(imgOcclResFamPreBsl,2)), 'lineProps', 'r');
xlabel('Time (s)'), ylabel('dF/F (%)'), xticks(-1:1:3), xlim([-1 3]), title('Pre Fam')
t(2) = subplot(3,5,2);
shadedErrorBar(vecAx,mean(imgFullResFamPostBsl,2)...
    ,std(imgFullResFamPostBsl,0,2)/sqrt(size(imgFullResFamPostBsl,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAx,mean(imgOcclResFamPostBsl,2)...
    ,std(imgOcclResFamPostBsl,0,2)/sqrt(size(imgOcclResFamPostBsl,2)), 'lineProps', 'r');
xlabel('Time (s)'), title('Post Fam')
t(3) = subplot(3,5,3);
shadedErrorBar(vecAx,mean(imgFullResNovPreBsl,2)...
    ,std(imgFullResNovPreBsl,0,2)/sqrt(size(imgFullResNovPreBsl,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAx,mean(imgOcclResNovPreBsl,2)...
    ,std(imgOcclResNovPreBsl,0,2)/sqrt(size(imgOcclResNovPreBsl,2)), 'lineProps', 'r');
xlabel('Time (s)'), ylabel('dF/F (%)'), xticks(-1:1:3), xlim([-1 3]), title('Pre Nov')
t(4) = subplot(3,5,4);
shadedErrorBar(vecAx,mean(imgFullResNovPostBsl,2)...
    ,std(imgFullResNovPostBsl,0,2)/sqrt(size(imgFullResNovPostBsl,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAx,mean(imgOcclResNovPostBsl,2)...
    ,std(imgOcclResNovPostBsl,0,2)/sqrt(size(imgOcclResNovPostBsl,2)), 'lineProps', 'r');
xlabel('Time (s)'), title('Post Nov')
% scatters
s(1) = subplot(3,5,6);
scatter(scatFullFamPopPre, scatOcclFamPopPre, sz, cPre, 'filled'); refline(1), xlabel('Full'), ylabel('Occl'), title('Fam Pre')
s(2) = subplot(3,5,7);
scatter(scatFullFamPopPost,scatOcclFamPopPost , sz, cPost, 'filled'); refline(1), title('Fam Post')
s(3) = subplot(3,5,8);
scatter(scatFullNovPopPre, scatOcclNovPopPre, sz, cPre, 'filled'); refline(1), title('Nov Pre')
s(4) = subplot(3,5,9);
scatter(scatFullNovPopPost, scatOcclNovPopPost, sz, cPost, 'filled'); refline(1), title('Nov Post')
% mean box plot
subplot(3,5,5)
boxchart([ones(size(scatFullFamPopPre)), ones(size(scatFullFamPopPost))+1, ...
    ones(size(scatFullNovPopPre))+2, ones(size(scatFullNovPopPost))+3, ...
    ones(size(scatOcclFamPopPre))+5, ones(size(scatOcclFamPopPost))+6, ...
    ones(size(scatOcclNovPopPre))+7, ones(size(scatOcclNovPopPost))+8], ...
    [scatFullFamPopPre, scatFullFamPopPost, scatFullNovPopPre, ...
    scatFullNovPopPost, scatOcclFamPopPre, scatOcclFamPopPost...
    scatOcclNovPopPre, scatOcclNovPopPost], 'MarkerStyle','none'), hold on
xlim([0 10]), ylabel('Response dF/F (%)'), xticks([1 2 3 4 5 6 7 8]), %if nfiles == 6, ylim([-10 15]), elseif nfiles == 5, ylim([-5 35]), end
xticklabels({'PreFamFull', 'PostFamFull','PreNovFull', 'PostNovFull', ...
    'PreFamOccl', 'PostFamOccl','PreNovOccl', 'PostNovOccl'}), xtickangle(45)
% mean scat/bar
subplot(3,5,10)
scatter([1 2 3 4],[mean(scatFullFamPopPre) mean(scatFullFamPopPost) mean(scatFullNovPopPre) mean(scatFullNovPopPost)], 35, 'k', 'LineWidth', 2), hold on
scatter([6 7 8 9],[mean(scatOcclFamPopPre) mean(scatOcclFamPopPost) mean(scatOcclNovPopPre) mean(scatOcclNovPopPost)], 35, col2, 'LineWidth', 2)                
er = errorbar([1 2 3 4],[mean(scatFullFamPopPre) mean(scatFullFamPopPost) mean(scatFullNovPopPre) mean(scatFullNovPopPost)], ...
    [calcSem(scatFullFamPopPre) calcSem(scatFullFamPopPost) calcSem(scatFullNovPopPre) calcSem(scatFullNovPopPost)] ...
    ,[calcSem(scatFullFamPopPre) calcSem(scatFullFamPopPost) calcSem(scatFullNovPopPre) calcSem(scatFullNovPopPost)]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
er = errorbar([6 7 8 9],[mean(scatOcclFamPopPre) mean(scatOcclFamPopPost) mean(scatOcclNovPopPre) mean(scatOcclNovPopPost)], ...
    [calcSem(scatOcclFamPopPre) calcSem(scatOcclFamPopPost) calcSem(scatOcclNovPopPre) calcSem(scatOcclNovPopPost)] ...
    ,[calcSem(scatOcclFamPopPre) calcSem(scatOcclFamPopPost) calcSem(scatOcclNovPopPre) calcSem(scatOcclNovPopPost)]);    
er.Color = col2; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 10]), ylabel('Response dF/F (%)'), xticks([1 2 3 4 5 6 7 8]), if nfiles == 6, ylim([-0.1 0.8]), elseif nfiles == 5, ylim([0 1]), end
xticklabels({'PreFamFull', 'PostFamFull','PreNovFull', 'PostNovFull', ...
    'PreFamOccl', 'PostFamOccl','PreNovOccl', 'PostNovOccl'}), xtickangle(45)
% compare pre vs post single cell level
g(1) = subplot(3,5,11);
scatter(scatFullFamPopPre, scatFullFamPopPost, sz, cPre, 'filled'); refline(1), xlabel('Pre'), ylabel('Post'),title('Full fam')
g(2) = subplot(3,5,12);
scatter(scatOcclFamPopPre, scatOcclFamPopPost , sz, cPost, 'filled'); refline(1), xlabel('Pre'), ylabel('Post'),title('Occl fam')
g(3) = subplot(3,5,13);
scatter(scatFullNovPopPre, scatFullNovPopPost, sz, cPre, 'filled'); refline(1), xlabel('Pre'), ylabel('Post'),title('Full nov')
g(4) = subplot(3,5,14);
scatter(scatOcclNovPopPre, scatOcclNovPopPost, sz, cPost, 'filled'); refline(1), xlabel('Pre'), ylabel('Post'),title('Occl nov')

% axes
for j = 1:length(t)
    t(j).YLim = [-0.1 0.7]; t(j).YTick = -0.1:0.2:0.7; t(j).XLim = [-1 3]; t(j).XTick = -1:1:3;
end
for j = 1:length(s)
    mn =  round(min([s(:).YLim s(:).XLim]));
    mx =  round(max([s(:).YLim s(:).XLim]));
    % s(j).YLim = [mn mx]; s(j).YTick = mn:20:mx; s(j).XLim = [mn mx]; s(j).XTick = mn:20:mx;
    s(j).YLim = [-1 3]; s(j).YTick = -1:1:3; s(j).XLim = [-1 3]; s(j).XTick = -1:1:3;
end
for j = 1:length(g)
    mn =  round(min([g(:).YLim g(:).XLim]));
    mx =  round(max([g(:).YLim g(:).XLim]));
    % g(j).YLim = [mn mx]; g(j).YTick = mn:20:mx; g(j).XLim = [mn mx]; g(j).XTick = mn:20:mx;
    g(j).YLim = [-1 3]; g(j).YTick = -1:1:3; g(j).XLim = [-1 3]; g(j).XTick = -1:1:3;
end

%%%%%% snake plots / imagesc of all cells, average responses
% sort on trace of preference pre training fam
traceToSortFamPre = imgFullResFamPreBsl;
[MniPre] = mean(traceToSortFamPre(vecAxSt,:));
[~,RsortedMnFamPre] = sort(MniPre,'descend');
[~, MxiPre] = max(traceToSortFamPre);
[~,RsortedMxFamPre] = sort(MxiPre,'ascend');

% sort on trace of preference pre training nov
traceToSortNovPre = imgFullResFamPreBsl;
[MniPre] = mean(traceToSortNovPre(vecAxSt,:));
[~,RsortedMnNovPre] = sort(MniPre,'descend');
[~, MxiPre] = max(traceToSortNovPre);
[~,RsortedMxNovPre] = sort(MxiPre,'ascend');

% sort on trace of preference post training fam
traceToSortFamPost = imgFullResFamPreBsl;
[MniPost] = mean(traceToSortFamPost(vecAxSt,:));
[~,RsortedMnFamPost] = sort(MniPost,'descend');
[~, MxiPost] = max(traceToSortFamPost);
[~,RsortedMxFamPost] = sort(MxiPost,'ascend');

% sort on trace of preference post training nov
traceToSortNovPost = imgFullResFamPreBsl;
[MniPost] = mean(traceToSortNovPost(vecAxSt,:));
[~,RsortedMnNovPost] = sort(MniPost,'descend');
[~, MxiPost] = max(traceToSortNovPost);
[~,RsortedMxNovPost] = sort(MxiPost,'ascend');

% plot with each condition in separate subplot, axes are similar scaling
clear p
figure('Position', [ 76         263        1577         713])
p(1) = subplot(1,9,1);
imagesc(vecAx, [], imgFullResFamPreBsl(:, RsortedMnFamPre)')
title('Pre Fam Full'),  xlabel('Time (s)'), ylabel('Neurons'), set(gca,'TickDir','out'), box off
p(2) = subplot(1,9,2);
imagesc(vecAx, [], imgOcclResFamPreBsl(:, RsortedMnFamPre)')
title('Pre Fam Occl'),  xlabel('Time (s)'), set(gca,'TickDir','out', 'YTickLabel', []), box off
p(3) = subplot(1,9,3);
imagesc(vecAx, [], imgFullResFamPostBsl(:, RsortedMnFamPost)')
title('Post Fam Full'),  xlabel('Time (s)'), ylabel('Neurons'), set(gca,'TickDir','out'), box off
p(4) = subplot(1,9,4);
imagesc(vecAx, [], imgOcclResFamPostBsl(:, RsortedMnFamPost)')
title('Post Fam Occl'),  xlabel('Time (s)'), set(gca,'TickDir','out', 'YTickLabel', []), box off

p(5) = subplot(1,9,6);
imagesc(vecAx, [], imgFullResNovPreBsl(:, RsortedMnNovPre)')
title('Pre Nov Full'),  xlabel('Time (s)'), ylabel('Neurons'), set(gca,'TickDir','out'), box off
p(6) = subplot(1,9,7);
imagesc(vecAx, [], imgOcclResNovPreBsl(:, RsortedMnNovPre)')
title('Pre Nov Occl'),  xlabel('Time (s)'), set(gca,'TickDir','out', 'YTickLabel', []), box off
p(7) = subplot(1,9,8);
imagesc(vecAx, [], imgFullResNovPostBsl(:, RsortedMnNovPost)')
title('Post Nov Full'),  xlabel('Time (s)'), ylabel('Neurons'), set(gca,'TickDir','out'), box off
p(8) = subplot(1,9,9);
imagesc(vecAx, [], imgOcclResNovPostBsl(:, RsortedMnNovPost)')
title('Post Nov Occl'),  xlabel('Time (s)'), set(gca,'TickDir','out', 'YTickLabel', []), box off
allCLim = get(p, {'CLim'});
allCLim = cat(2, allCLim{:});
set(p, 'CLim', [min(allCLim), max(allCLim)]);
% if nfiles == 6
%     set(p, 'CLim', [-5, 30]); % for L2/3
% elseif nfiles == 5
%     set(p, 'CLim', [-5, 30]); % for L5
% end
if nfiles == 6
    set(p, 'CLim', [-1, 3]); % for L2/3
elseif nfiles == 5
    set(p, 'CLim', [-1, 3]); % for L5
end
% colormap(flipud(gray))
colormap(hot)
subplot(1,9,5)
% axis off, colormap(flipud(gray)), caxis([-1 3]), colorbar
axis off, colormap(hot), caxis([-1 3]), colorbar

if save_fig
    func_save_fig('L23_ImagescSeparateChronicOG')
    func_save_fig('L5_ImagescSeparate')
end

%% stats
warning on

% full fam LMEM
% --- Paired LMEM: Difference (Pre - Post), mouse as random effect ---

% mouse IDs (should be same length for Pre and Post)
mouseID = categorical(mouseIDPre(:));  % same neurons, same mice

% ==== FULL FAM ====
diff_fullFam = scatFullFamPopPre(:) - scatFullFamPopPost(:);
tbl_diff_fullFam = table(diff_fullFam, mouseID, ...
    'VariableNames', {'Response', 'MouseID'});
lme_diff_fullFam = fitlme(tbl_diff_fullFam, 'Response ~ 1 + (1|MouseID)', ...
    'CheckHessian', true, 'FitMethod', 'REML', 'StartMethod', 'random');
[fe_fullFam, ~, stats_fullFam] = fixedEffects(lme_diff_fullFam);

% ==== OCCL FAM ====
diff_occlFam = scatOcclFamPopPre(:) - scatOcclFamPopPost(:);
tbl_diff_occlFam = table(diff_occlFam, mouseID, ...
    'VariableNames', {'Response', 'MouseID'});
lme_diff_occlFam = fitlme(tbl_diff_occlFam, 'Response ~ 1 + (1|MouseID)', ...
    'CheckHessian', true, 'FitMethod', 'REML', 'StartMethod', 'random');
[fe_occlFam, ~, stats_occlFam] = fixedEffects(lme_diff_occlFam);

% ==== FULL NOV ====
diff_fullNov = scatFullNovPopPre(:) - scatFullNovPopPost(:);
tbl_diff_fullNov = table(diff_fullNov, mouseID, ...
    'VariableNames', {'Response', 'MouseID'});
lme_diff_fullNov = fitlme(tbl_diff_fullNov, 'Response ~ 1 + (1|MouseID)', ...
    'CheckHessian', true, 'FitMethod', 'REML', 'StartMethod', 'random');
[fe_fullNov, ~, stats_fullNov] = fixedEffects(lme_diff_fullNov);

% ==== OCCL NOV ====
diff_occlNov = scatOcclNovPopPre(:) - scatOcclNovPopPost(:);
tbl_diff_occlNov = table(diff_occlNov, mouseID, ...
    'VariableNames', {'Response', 'MouseID'});
lme_diff_occlNov = fitlme(tbl_diff_occlNov, 'Response ~ 1 + (1|MouseID)', ...
    'CheckHessian', true, 'FitMethod', 'REML', 'StartMethod', 'random');
[fe_occlNov, ~, stats_occlNov] = fixedEffects(lme_diff_occlNov);


%% color coded chronic plot FAMILIAR (FIGURE S7) + displacement diagnostics

% Normalize PRE responses for color coding
fullNorm = normalize(scatFullFamPopPre, 'range');
occlNorm = normalize(scatOcclFamPopPre, 'range');

% Saturation thresholds
fullCutoff = 0.6;
occlCutoff = 0.6;

% Apply cutoff-based scaling
fullClamped = min(fullNorm, fullCutoff) / fullCutoff;
occlClamped = min(occlNorm, occlCutoff) / occlCutoff;

% Color coding: full → red, occl → blue
colors = [fullClamped(:), zeros(length(fullNorm),1), occlClamped(:)];

% Sort for plotting
magnitude = sqrt(fullNorm.^2 + occlNorm.^2);
[~, sortIdx] = sort(magnitude, 'ascend');
colors = colors(sortIdx, :);
scatFullFamPopPreSort = scatFullFamPopPre(sortIdx);
scatOcclFamPopPreSort = scatOcclFamPopPre(sortIdx);
scatFullFamPopPostSort = scatFullFamPopPost(sortIdx);
scatOcclFamPopPostSort = scatOcclFamPopPost(sortIdx);

if any(ismember(famIdx, 6))
    xPreLab = 'NO pre'; yPreLab = 'O pre';
    xPostLab = 'NO post'; yPostLab = 'O post';
    fprintf('included image 6 in famIdx, which is a novel image.\nOther analysis does not take this into account\n')
else
    xPreLab = 'NO fam pre'; yPreLab = 'O fam pre';
    xPostLab = 'NO fam post'; yPostLab = 'O fam post';
end

make_pre_post_displacement_diagnostics( ...
    scatFullFamPopPreSort, scatOcclFamPopPreSort, ...
    scatFullFamPopPostSort, scatOcclFamPopPostSort, ...
    colors, 'Familiar', xPreLab, yPreLab, xPostLab, yPostLab);

if save_fig
    func_save_fig('L23_chronic_Prepost_scatter_colorcodedOccltask_displacement_familiar')
    func_save_fig('L5_chronic_Prepost_scatter_colorcodedOccltask_displacement_familiar')
end

% Create colorbar figure (un-normalized version)
figure('Position', [1203 495 400 345]);
fullRange = linspace(min(scatFullFamPopPreSort), max(scatFullFamPopPreSort), 256);
occlRange = linspace(min(scatOcclFamPopPreSort), max(scatOcclFamPopPreSort), 256);
[fullGrid, occlGrid] = meshgrid(fullRange, occlRange);
fullClampedGrid = min(fullGrid, fullCutoff) / fullCutoff;
occlClampedGrid = min(occlGrid, occlCutoff) / occlCutoff;
colorGrid = cat(3, fullClampedGrid, zeros(size(fullClampedGrid)), occlClampedGrid);
image(fullRange, occlRange, colorGrid); axis xy;
xlabel('Full response pre');
ylabel('Occl response pre');
title('Familiar color blending: Red = Full, Blue = Occl');
set(gca, 'XTickMode', 'auto', 'YTickMode', 'auto');

if save_fig
    func_save_fig('L23_chronic_Prepost_scatter_colorcodedOccltask_colorbar_familiar')
    func_save_fig('L5_chronic_Prepost_scatter_colorcodedOccltask_colorbar_familiar')
end


%% color coded chronic plot NOVEL + displacement diagnostics

% Normalize PRE responses for color coding
fullNorm = normalize(scatFullNovPopPre, 'range');
occlNorm = normalize(scatOcclNovPopPre, 'range');

% Saturation thresholds
fullCutoff = 0.6;
occlCutoff = 0.6;

% Apply cutoff-based scaling
fullClamped = min(fullNorm, fullCutoff) / fullCutoff;
occlClamped = min(occlNorm, occlCutoff) / occlCutoff;

% Color coding: full → red, occl → blue
colors = [fullClamped(:), zeros(length(fullNorm),1), occlClamped(:)];

% Sort for plotting
magnitude = sqrt(fullNorm.^2 + occlNorm.^2);
[~, sortIdx] = sort(magnitude, 'ascend');
colors = colors(sortIdx, :);
scatFullNovPopPreSort = scatFullNovPopPre(sortIdx);
scatOcclNovPopPreSort = scatOcclNovPopPre(sortIdx);
scatFullNovPopPostSort = scatFullNovPopPost(sortIdx);
scatOcclNovPopPostSort = scatOcclNovPopPost(sortIdx);

make_pre_post_displacement_diagnostics( ...
    scatFullNovPopPreSort, scatOcclNovPopPreSort, ...
    scatFullNovPopPostSort, scatOcclNovPopPostSort, ...
    colors, 'Novel', 'NO nov pre', 'O nov pre', 'NO nov post', 'O nov post');

if save_fig
    func_save_fig('L23_chronic_Prepost_scatter_colorcodedOccltask_displacement_novel')
    func_save_fig('L5_chronic_Prepost_scatter_colorcodedOccltask_displacement_novel')
end

% Create colorbar figure (un-normalized version)
figure('Position', [1203 495 400 345]);
fullRange = linspace(min(scatFullNovPopPreSort), max(scatFullNovPopPreSort), 256);
occlRange = linspace(min(scatOcclNovPopPreSort), max(scatOcclNovPopPreSort), 256);
[fullGrid, occlGrid] = meshgrid(fullRange, occlRange);
fullClampedGrid = min(fullGrid, fullCutoff) / fullCutoff;
occlClampedGrid = min(occlGrid, occlCutoff) / occlCutoff;
colorGrid = cat(3, fullClampedGrid, zeros(size(fullClampedGrid)), occlClampedGrid);
image(fullRange, occlRange, colorGrid); axis xy;
xlabel('Full response pre');
ylabel('Occl response pre');
title('Novel color blending: Red = Full, Blue = Occl');
set(gca, 'XTickMode', 'auto', 'YTickMode', 'auto');

%% New heatmap plot to identify response of matched neurons by Leander

fullCutoff = 0.6;
occlCutoff = 0.6;

% Apply cutoff-based scaling
fullFamPreNorm  = normalize(scatFullFamPopPre, 'range');
occlFamPreNorm  = normalize(scatOcclFamPopPre, 'range');
fullFamPostNorm = normalize(scatFullFamPopPost, 'range');
occlFamPostNorm = normalize(scatOcclFamPopPost, 'range');
fullNovPreNorm  = normalize(scatFullNovPopPre, 'range');
occlNovPreNorm  = normalize(scatOcclNovPopPre, 'range');
fullNovPostNorm = normalize(scatFullNovPopPost, 'range');
occlNovPostNorm = normalize(scatOcclNovPopPost, 'range');
fullFamPreClamped = min(fullFamPreNorm, fullCutoff) / fullCutoff;
occlFamPreClamped  = min(occlFamPreNorm, occlCutoff) / occlCutoff;
fullFamPostClamped = min(fullFamPostNorm, fullCutoff) / fullCutoff;
occlFamPostClamped = min(occlFamPostNorm, occlCutoff) / occlCutoff;
fullNovPreClamped  = min(fullNovPreNorm, fullCutoff) / fullCutoff;
occlNovPreClamped  = min(occlNovPreNorm, occlCutoff) / occlCutoff;
fullNovPostClamped = min(fullNovPostNorm, fullCutoff) / fullCutoff;
occlNovPostClamped = min(occlNovPostNorm, occlCutoff) / occlCutoff;

% Color coding: full → red, occl → blue
nrois = length(fullNorm);
colorsFamPre  = cat(3, fullFamPreClamped(:),  zeros(nrois, 1), occlFamPreClamped(:));
colorsFamPost = cat(3, fullFamPostClamped(:), zeros(nrois, 1), occlFamPostClamped(:));
colorsNovPre  = cat(3, fullNovPreClamped(:),  zeros(nrois, 1), occlNovPreClamped(:));
colorsNovPost = cat(3, fullNovPostClamped(:), zeros(nrois, 1), occlNovPostClamped(:));
colors = cat(2, colorsFamPre, colorsFamPost, colorsNovPre, colorsNovPost);
xLabels = {'fam pre', 'fam post', 'nov pre', 'nov post'};
% colors = cat(2, colorsFamPre, colorsNovPre, colorsFamPost, colorsNovPost);
% xLabels = {'fam pre', 'nov pre', 'fam post', 'nov post'};

% sort the rois
% vals = sqrt(fullFamPreNorm.^2 + occlFamPreNorm.^2); sortExplain = {'sorted on response magnitude'; 'familiarImg-pre'};
vals = fullFamPreNorm - occlFamPreNorm; sortExplain = {'sorted on difference between'; 'NO-familiarImg-pre'; 'O-familiarImg-pre'};
% vals = fullNovPreNorm - occlNovPreNorm; sortExplain = {'sorted on difference between'; 'NO-novelImg-pre'; 'O-novelImg-pre'};
[~, sortIdx] = sort(vals, 'ascend');
colors = colors(sortIdx, :, :);

figure('Position', [680 374 467 504])
hax = subplot(1,2,1);
imagesc(colors)
set(hax, 'XTick', 1:4, 'XTickLabel', xLabels)
ylabel('roi (N)')
title(sortExplain)

hax = subplot(7,2,[4 8]);
image(fullRange, occlRange, colorGrid);
title({'Color blending:';  'Red = Full response'; 'Blue = Occl response'});
axis xy;
xlabel('Full response');
ylabel('Occl response');

% SaveImg('png', 'fam-nov_chronic_pre-post_NO-O_activity_differentColSort')

%% selectivity of neurons that increased responsiveness

% increase of >0.3, and minimum 0.1 response post (to avoid neurons that
% started negative pre and ended only a bit positive post)
ixUp = scatFullFamPopPost>0.2 & (scatFullFamPopPost-scatFullFamPopPre)>0.3;
nCells = sum(ixUp)
mnSel = mean(scatFullFamPopPost(ixUp))




%% final selectivity for NO novel

%%%% THIS ONE WENT INTO REVISION PAPER
fullPopPre = squeeze(mean(imgFullResMnPopPre(vecAxSt,:,:))-mean(imgFullResMnPopPre(vecAxSp,:,:)));
% occlPopPre = squeeze(mean(imgOcclResMnPopPre(vecAxSt,:,:))-mean(imgOcclResMnPopPre(vecAxSp,:,:)));
% occlPopPost = squeeze(mean(imgOcclResMnPopPost(vecAxSt,:,:))-mean(imgOcclResMnPopPost(vecAxSp,:,:)));

fullPopPost = squeeze(mean(imgFullResMnPopPost(vecAxSt,:,:))-mean(imgFullResMnPopPost(vecAxSp,:,:)));

siNovPost = NaN(8, size(fullPopPost, 2));
pairs = [3 1; 3 2; 3 4; 3 5; 6 1; 6 2; 6 4; 6 5];

for j = 1:8
    for i = 1:size(fullPopPost, 2)
        x = fullPopPost(pairs(j, 1), i);
        y = fullPopPost(pairs(j, 2), i);
        if x > 0 && y > 0
            siNovPost(j, i) = (x - y) / (x + y);
%             siNovPost(j, i) = (x - y);
        end
    end
end

siNovPostMn = nanmean(siNovPost);

figure('Position',[717   397   527   477])
% mask = scatFullNovPopPost > 0 & any(~isnan(siNovPost), 1);
mask = any(~isnan(siNovPost), 1);

% x = siNovPostMn(mask);
% y = scatFullNovPopPost(mask)-scatFullNovPopPre(mask);
% mask = any(~isnan(siNovPost), 1);
x = siNovPostMn(mask);
y = scatFullNovPopPost(mask)-scatFullNovPopPre(mask);
% x = siNovPostMn;
% y = scatFullNovPopPost-scatFullNovPopPre;


% Control for post magnitude explicitly
[ r_part, p_part ] = partialcorr(x', y', scatFullNovPopPost(mask)', 'type','Spearman');
fprintf('Partial Spearman rho = %.3f, p = %.3g\n', r_part, p_part);


scatter(x, y, 'filled', 'k'); refline;
xlabel('Expert Selectivity Index (familiar to novel)');
ylabel('Response strenght change (expert - naive, novel)');
title('Selectivity vs. Response change (Positive Responders)');
% ylim([-0.5 4])

% Compute correlation
[r, p] = corr(x', y', 'type', 'Pearson');

% Add text to plot
text(min(x) + 0.05 * range(x), max(y) - 0.05 * range(y), ...
    sprintf('R = %.2f, p = %.3g', r, p), ...
    'FontSize', 12, 'FontWeight', 'bold');


if save_fig
    func_save_fig('L23_scatter_novelSelectivity_vs_novelResIncrease')
    func_save_fig('L5_scatter_novelSelectivity_vs_novelResIncrease')
end

%%
% --- Compute fullPop for PRE and POST (same as your original style)
fullPopPre  = squeeze(mean(imgFullResMnPopPre(vecAxSt,:,:))  - mean(imgFullResMnPopPre(vecAxSp,:,:)));
fullPopPost = squeeze(mean(imgFullResMnPopPost(vecAxSt,:,:)) - mean(imgFullResMnPopPost(vecAxSp,:,:)));

% --- Selectivity indices (POST)
siNovPost = NaN(8, size(fullPopPost, 2));
pairs = [3 1; 3 2; 3 4; 3 5; 6 1; 6 2; 6 4; 6 5];
for j = 1:8
    for i = 1:size(fullPopPost, 2)
        x = fullPopPost(pairs(j, 1), i);
        y = fullPopPost(pairs(j, 2), i);
        if x > 0 && y > 0
            siNovPost(j, i) = (x - y) / (x + y);
%             siNovPost(j, i) = (x - y);
        end
    end
end
siNovPostMn = nanmean(siNovPost);

% --- Selectivity indices (PRE) - same computation as POST
siNovPre = NaN(8, size(fullPopPre,  2));
for j = 1:8
    for i = 1:size(fullPopPre, 2)
        x = fullPopPre(pairs(j, 1), i);
        y = fullPopPre(pairs(j, 2), i);
        if x > 0 && y > 0
            siNovPre(j, i) = (x - y) / (x + y);
%             siNovPre(j, i) = (x - y);
        end
    end
end
siNovPreMn = nanmean(siNovPre);

% -------------------- Two subplots: POST (left) and PRE (right) --------------------
figure('Position',[200 300 1100 450])

% --- POST subplot (your original analysis) ---
subplot(1,2,1)
% maskPost = any(~isnan(siNovPost), 1);
maskPost = scatFullNovPopPost > 0 & any(~isnan(siNovPost), 1);

x = siNovPostMn(maskPost);
y = scatFullNovPopPost(maskPost)-scatFullNovPopPre(maskPost);

% Control for post magnitude explicitly (POST)
[ r_part, p_part ] = partialcorr(x', y', scatFullNovPopPost(maskPost)', 'type','Spearman');
fprintf('POST: Partial Spearman rho = %.3f, p = %.3g\n', r_part, p_part);

scatter(x, y, 'filled', 'k'); refline;
xlabel('Expert Selectivity Index (familiar to novel) - POST');
ylabel('Response strenght change (expert - naive, novel)');
title('Selectivity vs. Response change (POST)');
[rP_post, pP_post] = corr(x', y', 'type', 'Pearson');
text(min(x) + 0.05 * range(x), max(y) - 0.05 * range(y), ...
    sprintf('R = %.2f, p = %.3g', rP_post, pP_post), ...
    'FontSize', 12, 'FontWeight', 'bold');

% --- PRE subplot (same analysis but using PRE SI) ---
subplot(1,2,2)
% maskPre = any(~isnan(siNovPre), 1);
maskPre = scatFullNovPopPre > 0 & any(~isnan(siNovPre), 1);
x = siNovPreMn(maskPre);
y = scatFullNovPopPost(maskPre)-scatFullNovPopPre(maskPre);

% Control for post magnitude explicitly (PRE analysis too)
[ r_part_pre, p_part_pre ] = partialcorr(x', y', scatFullNovPopPost(maskPre)', 'type','Spearman');
fprintf('PRE:  Partial Spearman rho = %.3f, p = %.3g\n', r_part_pre, p_part_pre);

scatter(x, y, 'filled', 'k'); refline;
xlabel('Expert Selectivity Index (familiar to novel) - PRE');
ylabel('Response strenght change (expert - naive, novel)');
title('Selectivity vs. Response change (PRE)');
[rP_pre, pP_pre] = corr(x', y', 'type', 'Pearson');
text(min(x) + 0.05 * range(x), max(y) - 0.05 * range(y), ...
    sprintf('R = %.2f, p = %.3g', rP_pre, pP_pre), ...
    'FontSize', 12, 'FontWeight', 'bold');



% -------------------- SI(pre) vs SI(post) per neuron --------------------
figure('Position',[717 397 527 477])
x = siNovPreMn;
y = siNovPostMn;
% maskBoth = ~isnan(x) & ~isnan(y);
maskBoth = ~isnan(x) & ~isnan(y) & scatFullNovPopPre>0 & scatFullNovPopPost>0;

scatFullNovPopPre > 0

scatter(x(maskBoth), y(maskBoth), 'filled', 'k'); hold on;
mn = min([x(maskBoth), y(maskBoth)], [], 'all');
mx = max([x(maskBoth), y(maskBoth)], [], 'all');
plot([mn mx], [mn mx], 'k:'); % unity line
scatter(mean(x(maskBoth)), mean(y(maskBoth)), 50,'filled','r');
hold off

xlabel('Selectivity Index - PRE');
ylabel('Selectivity Index - POST');
title('Neuron-wise change in selectivity (PRE vs POST)');



% -------------------- Response (pre) vs Response (post) per neuron --------------------
figure('Position',[717 397 527 477])
x = scatFullNovPopPre;
y = scatFullNovPopPost;

scatter(x, y, 'filled', 'k'); hold on;
% mn = min([x(maskBoth), y(maskBoth)], [], 'all');
% mx = max([x(maskBoth), y(maskBoth)], [], 'all');
% plot([mn mx], [mn mx], 'k:'); % unity line
refline(1)
scatter(mean(x), mean(y), 50,'filled','r');
hold off

xlabel('Response PRE');
ylabel('Response POST');
title('Neuron-wise change in response to novel (PRE vs POST)');



%% control analysis for pre

fullPopPre = squeeze(mean(imgFullResMnPopPre(vecAxSt,:,:))-mean(imgFullResMnPopPre(vecAxSp,:,:)));

siNovPre = NaN(8, size(fullPopPre, 2));
pairs = [3 1; 3 2; 3 4; 3 5; 6 1; 6 2; 6 4; 6 5];

for j = 1:8
    for i = 1:size(fullPopPre, 2)
        x = fullPopPre(pairs(j, 1), i);
        y = fullPopPre(pairs(j, 2), i);
        if x > 0 && y > 0
            siNovPre(j, i) = (x - y) / (x + y);
        end
    end
end

siNovPreMn = nanmean(siNovPre);

figure
mask = scatFullNovPopPre > 0 & any(~isnan(siNovPre), 1);
% mask = any(~isnan(siNovPre), 1);
x = siNovPreMn(mask);
y = scatFullNovPopPre(mask);

scatter(x, y, 'filled', 'k'); refline;
xlabel('Naive Selectivity Index (familiar to novel)');
ylabel('Response Strength for Novel Naive');
title('Selectivity vs. Response Strength (Positive Responders)');
% ylim([-0.5 4])

% Compute correlation
[r, p] = corr(x', y', 'type', 'Pearson');

% Add text to plot
text(min(x) + 0.05 * range(x), max(y) - 0.05 * range(y), ...
    sprintf('R = %.2f, p = %.3g', r, p), ...
    'FontSize', 12, 'FontWeight', 'bold');

if save_fig
    func_save_fig('L23_scatter_novelSelectivity_vs_novelResStrength')
    func_save_fig('L5_scatter_novelSelectivity_vs_novelResStrength')
end

%% analysis for post

fullPopPost = squeeze(mean(imgFullResMnPopPost(vecAxSt,:,:))-mean(imgFullResMnPopPost(vecAxSp,:,:)));

siNovPost = NaN(8, size(fullPopPost, 2));
pairs = [3 1; 3 2; 3 4; 3 5; 6 1; 6 2; 6 4; 6 5];

for j = 1:8
    for i = 1:size(fullPopPost, 2)
        x = fullPopPost(pairs(j, 1), i);
        y = fullPopPost(pairs(j, 2), i);
        if x > 0 && y > 0
            siNovPost(j, i) = (x - y) / (x + y);
        end
    end
end

siNovPostMn = nanmean(siNovPost);

figure
mask = scatFullNovPopPost > 0 & any(~isnan(siNovPost), 1);
% mask = any(~isnan(siNovPost), 1);
x = siNovPostMn(mask);
y = scatFullNovPopPost(mask);

scatter(x, y, 'filled', 'k'); refline;
xlabel('Naive Selectivity Index (familiar to novel)');
ylabel('Response Strength for Novel Naive');
title('Selectivity vs. Response Strength (Positive Responders)');
% ylim([-0.5 4])

% Compute correlation
[r, p] = corr(x', y', 'type', 'Pearson');

% Add text to plot
text(min(x) + 0.05 * range(x), max(y) - 0.05 * range(y), ...
    sprintf('R = %.2f, p = %.3g', r, p), ...
    'FontSize', 12, 'FontWeight', 'bold');

if save_fig
    func_save_fig('L23_scatter_novelSelectivity_vs_novelResStrength')
    func_save_fig('L5_scatter_novelSelectivity_vs_novelResStrength')
end

%% selectivity for NO novel pre versus Novelty response post

fullPopPre = squeeze(mean(imgFullResMnPopPre(vecAxSt,:,:))-mean(imgFullResMnPopPre(vecAxSp,:,:)));
occlPopPre = squeeze(mean(imgOcclResMnPopPre(vecAxSt,:,:))-mean(imgOcclResMnPopPre(vecAxSp,:,:)));
fullPopPost = squeeze(mean(imgFullResMnPopPost(vecAxSt,:,:))-mean(imgFullResMnPopPost(vecAxSp,:,:)));
occlPopPost = squeeze(mean(imgOcclResMnPopPost(vecAxSt,:,:))-mean(imgOcclResMnPopPost(vecAxSp,:,:)));

siNovPre = zeros(1,size(fullPopPre,2));
for i = 1:size(fullPopPre,2)
    % Compute selectivity index for stimuli 3 and 6
    R3 = fullPopPre(3,i);
    R6 = fullPopPre(6,i);
    total_response = sum(fullPopPre(:,i));

    siNovPre(i) = (R3 + R6) / total_response;

end
siNovPre(siNovPre<-3)=-3;
siNovPre(siNovPre>3)=3;

figure, scatter(siNovPre, scatFullNovPopPost-scatFullNovPopPre), refline
xlabel('Selectivity full nov pre'), ylabel('Full nov post - full nov pre')


fullPopPre = squeeze(mean(imgFullResMnPopPre(vecAxSt,:,:))-mean(imgFullResMnPopPre(vecAxSp,:,:)));
occlPopPre = squeeze(mean(imgOcclResMnPopPre(vecAxSt,:,:))-mean(imgOcclResMnPopPre(vecAxSp,:,:)));
fullPopPost = squeeze(mean(imgFullResMnPopPost(vecAxSt,:,:))-mean(imgFullResMnPopPost(vecAxSp,:,:)));
occlPopPost = squeeze(mean(imgOcclResMnPopPost(vecAxSt,:,:))-mean(imgOcclResMnPopPost(vecAxSp,:,:)));

siNovPre = zeros(1,size(fullPopPre,2));
siNovPost = zeros(1,size(fullPopPre,2));
for i = 1:size(fullPopPre,2)
    % Compute selectivity index for stimuli 3 and 6
    R1 = fullPopPre(1,i);
    R2 = fullPopPre(2,i);
    R3 = fullPopPre(3,i);
    R4 = fullPopPre(4,i);
    R5 = fullPopPre(5,i);
    R6 = fullPopPre(6,i);
    total_response = sum(fullPopPre(:,i));

    siNovPre(i) = ((mean(fullPopPre([3 6],i)-mean(fullPopPre([1 2 4 5],i))))) / ((mean(fullPopPre([3 6],i)+mean(fullPopPre([1 2 4 5],i)))));
    siNovPost(i) = ((mean(fullPopPost([3 6],i)-mean(fullPopPost([1 2 4 5],i))))) / ((mean(fullPopPost([3 6],i)+mean(fullPopPost([1 2 4 5],i)))));

end
siNovPre(siNovPre<-1)=-1;
siNovPre(siNovPre>1)=1;
siNovPost(siNovPost<-1)=-1;
siNovPost(siNovPost>1)=1;

ix = mean(fullPopPost([3 6],:)>0)&mean(fullPopPost([1 2 4 5],:))

% each image with each image

figure, scatter(siNovPre, scatFullNovPopPost-scatFullNovPopPre), refline
xlabel('Selectivity full nov pre'), ylabel('Full nov post - full nov pre')


% third way
fullPopPre = squeeze(mean(imgFullResMnPopPre(vecAxSt,:,:))-mean(imgFullResMnPopPre(vecAxSp,:,:)));
occlPopPre = squeeze(mean(imgOcclResMnPopPre(vecAxSt,:,:))-mean(imgOcclResMnPopPre(vecAxSp,:,:)));
fullPopPost = squeeze(mean(imgFullResMnPopPost(vecAxSt,:,:))-mean(imgFullResMnPopPost(vecAxSp,:,:)));
occlPopPost = squeeze(mean(imgOcclResMnPopPost(vecAxSt,:,:))-mean(imgOcclResMnPopPost(vecAxSp,:,:)));

siNovPre = zeros(1,size(fullPopPre,2));
siNovPost = zeros(1,size(fullPopPre,2));
for i = 1:size(fullPopPre,2)
    % Compute selectivity index for stimuli 3 and 6
    R1 = fullPopPre(1,i);
    R2 = fullPopPre(2,i);
    R3 = fullPopPre(3,i);
    R4 = fullPopPre(4,i);
    R5 = fullPopPre(5,i);
    R6 = fullPopPre(6,i);
    total_response = sum(fullPopPre(:,i));

    siNovPre(i) = ((mean(fullPopPre([3 6],i)-mean(fullPopPre([1 2 4 5],i)))));
    siNovPost(i) = ((mean(fullPopPost([3 6],i)-mean(fullPopPost([1 2 4 5],i)))));

end
% siNovPre(siNovPre<-1)=-1;
% siNovPre(siNovPre>1)=1;

figure, scatter(siNovPre, scatFullNovPopPost-scatFullNovPopPre), refline
xlabel('Selectivity full nov pre'), ylabel('Full nov post - full nov pre')

%% correlation full vs occl pre vs post

fullPopPre = squeeze(mean(imgFullResMnPopPre(vecAxSt,:,:))-mean(imgFullResMnPopPre(vecAxSp,:,:)));
occlPopPre = squeeze(mean(imgOcclResMnPopPre(vecAxSt,:,:))-mean(imgOcclResMnPopPre(vecAxSp,:,:)));
fullPopPost = squeeze(mean(imgFullResMnPopPost(vecAxSt,:,:))-mean(imgFullResMnPopPost(vecAxSp,:,:)));
occlPopPost = squeeze(mean(imgOcclResMnPopPost(vecAxSt,:,:))-mean(imgOcclResMnPopPost(vecAxSp,:,:)));


rPre = zeros(size(fullPopPre,2),1);
rPost = zeros(size(fullPopPre,2),1);
im = [1 2 4 5];

for i = 1:size(fullPopPre,2)
    rPre(i) = corr(fullPopPre(im,i), occlPopPre(im,i));
    rPost(i) = corr(fullPopPost(im,i), occlPopPost(im,i));
end

ix = mean(fullPopPost(im,:))>0.5;
figure
histogram(rPre(ix),20), hold on
histogram(rPost(ix),20)

[p,h] = ttest2(rPre(ix), rPost(ix))


%% delta O novel vs delta O familliar at single cell level

%%%% THIS WENT INTO REVISION PAPER

fullPopPre = squeeze(mean(imgFullResMnPopPre(vecAxSt,:,:))-mean(imgFullResMnPopPre(vecAxSp,:,:)));
occlPopPre = squeeze(mean(imgOcclResMnPopPre(vecAxSt,:,:))-mean(imgOcclResMnPopPre(vecAxSp,:,:)));
fullPopPost = squeeze(mean(imgFullResMnPopPost(vecAxSt,:,:))-mean(imgFullResMnPopPost(vecAxSp,:,:)));
occlPopPost = squeeze(mean(imgOcclResMnPopPost(vecAxSt,:,:))-mean(imgOcclResMnPopPost(vecAxSp,:,:)));

fam = [1 2 4 5];
nov = [3 6];

dOfam = mean(occlPopPost(fam,:)-occlPopPre(fam,:));
dOnov = mean(occlPopPost(nov,:)-occlPopPre(nov,:));

sz = 40;

figure('Position',[717   397   527   477])
scatter(dOfam, dOnov, sz, 'k', 'filled')
refline(1), refline
xlabel('delta O fam'), ylabel('delta O nov')

% Compute correlation
[R, P] = corr(dOfam(:), dOnov(:));  % Make sure both are column vectors

% Add text to the plot
textLocX = min(dOfam) + 0.05 * range(dOfam);
textLocY = max(dOnov) - 0.1 * range(dOnov);
text(textLocX, textLocY, sprintf('R = %.3f, p = %.3g', R, P), 'FontSize', 12, 'FontWeight', 'bold')


if save_fig
    func_save_fig('L23_scatter_chronic_post-pre_famvsnov')
    func_save_fig('L5_scatter_novelSelectivity_vs_novelResStrength')
end

%% revision analysis

% do NO responders become more responsive to O after training?
ix = scatFullFamPopPre>0.5;

figure
scatter(scatOcclFamPopPre(ix), scatOcclFamPopPost(ix)), hold on
scatter(mean(scatOcclFamPopPre(ix)), mean(scatOcclFamPopPost(ix)), 'filled')
xlim([-0.5 2.5]),ylim([-0.5 2.5])
refline(1)
title('Response for NO naive responders')
xlabel('Naive fam occl')
ylabel('Expert fam occl')

figure
subplot(1,2,1)
scatter(scatFullFamPopPre, scatOcclFamPopPre), hold on
scatter(scatFullFamPopPre(ix), scatOcclFamPopPre(ix), 'filled')
xlim([-0.5 2.5]), ylim([-0.5 2.5]), refline(1)
subplot(1,2,2)
scatter(scatFullFamPopPost, scatOcclFamPopPost), hold on
scatter(scatFullFamPopPost(ix), scatOcclFamPopPost(ix), 'filled')
xlim([-0.5 2.5]), ylim([-0.5 2.5]), refline(1)


% do NO responders become more responsive to O after training?
ix = scatFullFamPopPre<0.5;

figure('Position', [ 365         231        1051         370])
subplot(1,2,1)
scatter(scatFullFamPopPre, scatOcclFamPopPre), hold on
scatter(scatFullFamPopPre(ix), scatOcclFamPopPre(ix), 'filled')
xlim([-1 3]), ylim([-1 3]), refline(1), xlabel('NO'), ylabel('O'), title('Pre')
subplot(1,2,2)
scatter(scatFullFamPopPost, scatOcclFamPopPost), hold on
scatter(scatFullFamPopPost(ix), scatOcclFamPopPost(ix), 'filled')
xlim([-1 3]), ylim([-1 3]), refline(1), xlabel('NO'), ylabel('O'), title('Post')


%% selectivity / lifetime sparseness

ix = famIdx;

imgFullPre = squeeze(mean(imgFullResMnPopPre(vecAxSt,ix,:)))-squeeze(mean(imgFullResMnPopPre(vecAxSp,ix,:)));
imgFullPost = squeeze(mean(imgFullResMnPopPost(vecAxSt,ix,:)))-squeeze(mean(imgFullResMnPopPost(vecAxSp,ix,:)));
imgOcclPre = squeeze(mean(imgOcclResMnPopPre(vecAxSt,ix,:)))-squeeze(mean(imgOcclResMnPopPre(vecAxSp,ix,:)));
imgOcclPost = squeeze(mean(imgOcclResMnPopPost(vecAxSt,ix,:)))-squeeze(mean(imgOcclResMnPopPost(vecAxSp,ix,:)));

% lifetime sparseness
sparsenessFullPre = calculateLifetimeSparseness(imgFullPre')';
sparsenessFullPost = calculateLifetimeSparseness(imgFullPost')';

sparsenessOcclPre = calculateLifetimeSparseness(imgOcclPre')';
sparsenessOcclPost = calculateLifetimeSparseness(imgOcclPost')';

% threshold = 0.5;
threshold = -1000;

mouseIDPreFull = mouseIDPre(scatFullFamPopPre>threshold);
mouseIDPostFull = mouseIDPost(scatFullFamPopPost>threshold);
mouseIDPreOccl = mouseIDPre(scatOcclFamPopPre>threshold);
mouseIDPostOccl = mouseIDPost(scatOcclFamPopPost>threshold);

sparsenessFullPre = sparsenessFullPre(scatFullFamPopPre>threshold);
sparsenessFullPost = sparsenessFullPost(scatFullFamPopPost>threshold);

sparsenessOcclPre = sparsenessOcclPre(scatOcclFamPopPre>threshold);
sparsenessOcclPost = sparsenessOcclPost(scatOcclFamPopPost>threshold);

figure
subplot(1,2,1)
scatter([1 2],[mean(sparsenessFullPre) mean(sparsenessFullPost)], sz, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2],[mean(sparsenessFullPre) mean(sparsenessFullPost)], ...
    [calcSem(sparsenessFullPre) calcSem(sparsenessFullPost)] ...
    ,[calcSem(sparsenessFullPre) calcSem(sparsenessFullPost)]);
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 3]), ylim([0 0.65]), ylabel('Sparseness NO'), xticks([1 2])
xticklabels({'N','E'})
subplot(1,2,2)
scatter([1 2],[mean(sparsenessOcclPre) mean(sparsenessOcclPost)], sz, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2],[mean(sparsenessOcclPre) mean(sparsenessOcclPost)], ...
    [calcSem(sparsenessOcclPre) calcSem(sparsenessOcclPost)] ...
    ,[calcSem(sparsenessOcclPre) calcSem(sparsenessOcclPost)]);
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 3]), ylim([0 0.65]), ylabel('Sparseness O'), xticks([1 2])
xticklabels({'N','E'})

%%%% fam
[rDiff, pDiff] = corrcoef(sparsenessFullPre, scatFullFamPopPre - scatFullFamPopPost);
[rRes, pRes]   = corrcoef(sparsenessFullPre, scatFullFamPopPre);
figure('Position', [249 514 1124 373])
% --- Subplot 1: Difference (naive - expert)
subplot(1,2,1)
x1 = sparsenessFullPre;
y1 = scatFullFamPopPre - scatFullFamPopPost;
scatter(x1, y1, 30, 'filled', 'k'), hold on; refline, ylim([-1.5 3])
ylabel('NO fam naive - NO fam expert'), xlabel('NO fam selectivity (naive)')
% Place text in top-right corner
xlim1 = xlim; ylim1 = ylim;
text(xlim1(2) - 0.05 * range(xlim1), ylim1(2) - 0.05 * range(ylim1), ...
    sprintf('r = %.2f\np = %.3f', rDiff(1,2), pDiff(1,2)), ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', 10)
% --- Subplot 2: Raw naive
subplot(1,2,2)
x2 = sparsenessFullPre;
y2 = scatFullFamPopPre;
scatter(x2, y2, 30, 'filled', 'k'), hold on; refline, ylim([-1.5 3])
ylabel('NO fam naive'), xlabel('NO fam selectivity (naive)')
% Place text in top-right corner
xlim2 = xlim; ylim2 = ylim;
text(xlim2(2) - 0.05 * range(xlim2), ylim2(2) - 0.05 * range(ylim2), ...
    sprintf('r = %.2f\np = %.3f', rRes(1,2), pRes(1,2)), ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', 10)


%%%% novel
[rDiff, pDiff] = corrcoef(sparsenessFullPre, scatFullNovPopPre - scatFullNovPopPost);
[rRes, pRes]   = corrcoef(sparsenessFullPre, scatFullNovPopPre);
figure('Position', [249 514 1124 373])
% --- Subplot 1: Difference (naive - expert)
subplot(1,2,1)
x1 = sparsenessFullPre;
y1 = scatFullNovPopPre - scatFullNovPopPost;
scatter(x1, y1, 30, 'filled', 'k'), hold on; refline, ylim([-1.5 3])
ylabel('NO nov naive - NO nov expert'), xlabel('NO nov selectivity (naive)')
% Place text in top-right corner
xlim1 = xlim; ylim1 = ylim;
text(xlim1(2) - 0.05 * range(xlim1), ylim1(2) - 0.05 * range(ylim1), ...
    sprintf('r = %.2f\np = %.3f', rDiff(1,2), pDiff(1,2)), ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', 10)
% --- Subplot 2: Raw naive
subplot(1,2,2)
x2 = sparsenessFullPre;
y2 = scatFullNovPopPre;
scatter(x2, y2, 30, 'filled', 'k'), hold on; refline, ylim([-1.5 3])
ylabel('NO nov naive'), xlabel('NO nov selectivity (naive)')
% Place text in top-right corner
xlim2 = xlim; ylim2 = ylim;
text(xlim2(2) - 0.05 * range(xlim2), ylim2(2) - 0.05 * range(ylim2), ...
    sprintf('r = %.2f\np = %.3f', rRes(1,2), pRes(1,2)), ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', 10)

mouseID = categorical(mouseIDPreFull(:));  % same neurons, same mice

% ==== FULL FAM ====
diff_fullFamSpars = sparsenessFullPre(:) - sparsenessFullPost(:);
tbl_diff_fullFamSpars = table(diff_fullFamSpars, mouseID, ...
    'VariableNames', {'Response', 'MouseID'});
lme_diff_fullFamSpars = fitlme(tbl_diff_fullFamSpars, 'Response ~ 1 + (1|MouseID)', ...
    'CheckHessian', true, 'FitMethod', 'REML', 'StartMethod', 'random');
[fe_fullFam, ~, stats_fullFam] = fixedEffects(lme_diff_fullFamSpars);




%% lifetime sparseness for neurons that start responding to NO fam after training

% respond after but not before training
% ixUp = scatFullFamPopPost>0.4 & scatFullFamPopPre<0.4;
ixUp = scatFullFamPopPost>0.2 & (scatFullFamPopPost-scatFullFamPopPre)>0.5;
ixDown = scatFullFamPopPost>0.2 & (scatFullFamPopPost-scatFullFamPopPre)<0;

% ix = sparsenessFullPost>0.6;
ix = scatFullFamPopPost>0.5;
inc = scatFullFamPopPost-scatFullFamPopPre;
% figure
% scatter(scatFullFamPopPre(ix), scatFullFamPopPost(ix))

% Group 1: ~ixUp
means1 = [nanmean(sparsenessFullPost(ixUp)), nanmean(sparsenessFullPost(ixDown))];
sems1  = [nanstd(sparsenessFullPost(ixUp)) / sqrt(sum(~isnan(sparsenessFullPost(ixDown)))), ...
          nanstd(sparsenessFullPost(ixUp)) / sqrt(sum(~isnan(sparsenessFullPost(ixDown))))];


% 
% % Group 1: ~ixUp
% means1 = [nanmean(sparsenessFullPre(ixUp)), nanmean(sparsenessFullPost(ixUp))];
% sems1  = [nanstd(sparsenessFullPre(ixUp)) / sqrt(sum(~isnan(sparsenessFullPre(ixUp)))), ...
%           nanstd(sparsenessFullPost(ixUp)) / sqrt(sum(~isnan(sparsenessFullPost(ixUp))))];
% % Group 2: ixUp
% means2 = [nanmean(sparsenessFullPre(ixDown)), nanmean(sparsenessFullPost(ixDown))];
% sems2  = [nanstd(sparsenessFullPre(ixDown)) / sqrt(sum(~isnan(sparsenessFullPre(ixDown)))), ...
%           nanstd(sparsenessFullPost(ixDown)) / sqrt(sum(~isnan(sparsenessFullPost(ixDown))))];
figure
% Bar plots
bar([1 2], means1, 'FaceAlpha', 0.5); hold on
% bar([4 5], means2, 'FaceAlpha', 0.5); hold on

% Error bars
errorbar([1 2], means1, sems1, 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 8)
% errorbar([4 5], means2, sems2, 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 8)

% Optional formatting
xticks([1 2 4 5])
xticklabels({'Pre ↓', 'Post ↓', 'Pre ↑', 'Post ↑'})
ylabel('Sparseness')
box off


% mouseIDPre = [];
% mouseIDPost = [];
% for i = 1:nfiles
%     % prepare some data for linear mixed model effect
%     mouseIDPre = [mouseIDPre zeros(1,length(datastructPre(i).scatFull))+i];
%     mouseIDPost = [mouseIDPost zeros(1,length(datastructPost(i).scatFull))+i];
% end
% 
% data = cat(2, sparsenessOcclPre,sparsenessOcclPost, sparsenessOcclTask)';
% mouseID = categorical(cat(2, mouseIDPreOccl,mouseIDPostOccl,mouseIDTaskOccl))';
% condition = categorical(cat(1, ones(length(mouseIDPreOccl),1),ones(length(mouseIDPostOccl),1)+1,ones(length(mouseIDTaskOccl),1)+2));
% clear statTbl, statTbl = table(data, mouseID, condition);
% lmeSpars = fitlme(statTbl, 'data ~ condition + (1|mouseID)', 'CheckHessian', 1,'FitMethod', 'REML', 'StartMethod', 'random');
% statsSparsOccl = anova(lmeSpars,'DFMethod','Satterthwaite');
% statTblSparsOccl = makeStatTbl(lmeSpars);


figure('Position', [509   252   560   420])
scatter(scatFullFamPopPre, scatFullFamPopPost)
refline(1), xlabel('Full pre'), ylabel('Full task')
% xlim([-0.5 2.5]), ylim([-0.5 2.5])


%% In case of comparing familiar vs novel and full vs occl

% sort on trace of preference pre training, could be either of the 2 for live figure
traceToSortPre = imgFullResFamPreBsl;
% traceToSortPre = imgFullResNovPreBsl;
[MniPre] = mean(traceToSortPre(vecAxSt,:));
[~,RsortedMnPre] = sort(MniPre,'descend');

traceToSortPost = imgFullResFamPostBsl;
% traceToSortPost = imgFullResNovPostBsl;
[MniPost] = mean(traceToSortPost(vecAxSt,:));
[~,RsortedMnPost] = sort(MniPost,'descend');

clear p
p(1) = subplot(1,4,1);
figure('Position', [223.4000  105.8000  608.0000  980.0000])
p(1) = subplot(1,4,1);
imagesc(vecAx,[],imgFullResFamPreBsl(:,RsortedMnPre)')
p(2) = subplot(1,4,2);
imagesc(vecAx,[],imgOcclResFamPreBsl(:,RsortedMnPre)')
p(3) = subplot(1,4,3);
imagesc(vecAx,[],imgFullResNovPreBsl(:,RsortedMnPre)')
p(4) = subplot(1,4,4);
imagesc(vecAx,[],imgOcclResNovPreBsl(:,RsortedMnPre)')
allCLim = get(p, {'CLim'});
allCLim = cat(2, allCLim{:});
% set(p, 'CLim', [min(allCLim), max(allCLim)]);
set(p, 'CLim', [-5, 60]);
colormap hot

clear q
figure('Position', [881.8000  104.2000  608.0000  980.0000])
q(1) = subplot(1,4,1);
imagesc(vecAx,[],imgFullResFamPostBsl(:,RsortedMnPost)')
q(2) = subplot(1,4,2);
imagesc(vecAx,[],imgOcclResFamPostBsl(:,RsortedMnPost)')
q(3) = subplot(1,4,3);
imagesc(vecAx,[],imgFullResNovPostBsl(:,RsortedMnPost)')
q(4) = subplot(1,4,4);
imagesc(vecAx,[],imgOcclResNovPostBsl(:,RsortedMnPost)')
allCLim = get(q, {'CLim'});
allCLim = cat(2, allCLim{:});
% set(q, 'CLim', [min(allCLim), max(allCLim)]);
set(q, 'CLim', [-5, 60]);
colormap hot


figure
% scatter(mean(imgFullResNovPostBsl(vecAxSt,:)), mean(imgOcclResNovPostBsl(vecAxSt,:)))
% scatter(mean(imgFullResFamPostBsl(vecAxSt,:)), mean(imgOcclResFamPostBsl(vecAxSt,:)))
% scatter(mean(imgFullResFamPostBsl(vecAxSt,:)), mean(imgOcclResNovPostBsl(vecAxSt,:)))
% scatter(mean(imgOcclResFamPostBsl(vecAxSt,:)), mean(imgOcclResNovPostBsl(vecAxSt,:)))
% scatter(mean(imgFullResFamPostBsl(vecAxSt,:)), mean(imgFullResNovPostBsl(vecAxSt,:)))
% scatter(mean(imgFullResFamPostBsl(vecAxSt,:)), mean(imgOcclResFamPostBsl(vecAxSt,:)))
scatter(mean(imgFullResNovPostBsl(vecAxSt,:)), mean(imgOcclResFamPostBsl(vecAxSt,:)))
refline(1,0)
refline
xline(0)
yline(0)


%% new selectivity measure based on Poort et al nat neuro

imgFullResPopPreTr = reshape(imgFullResPopPre,[size(imgFullResPopPre,1),size(imgFullResPopPre,2)*size(imgFullResPopPre,3),size(imgFullResPopPre,4)]);
imgOcclResPopPreTr = reshape(imgOcclResPopPre,[size(imgOcclResPopPre,1),size(imgOcclResPopPre,2)*size(imgOcclResPopPre,3),size(imgOcclResPopPre,4)]);
imgFullResPopPreTrMn = mean(squeeze(mean(imgFullResPopPreTr(vecAxSt,:,:))-mean(imgFullResPopPreTr(vecAxSp,:,:))));
imgOcclResPopPreTrMn = mean(squeeze(mean(imgOcclResPopPreTr(vecAxSt,:,:))-mean(imgOcclResPopPreTr(vecAxSp,:,:))));
imgFullResPopPreTrSd = std(squeeze(mean(imgFullResPopPreTr(vecAxSt,:,:))-mean(imgFullResPopPreTr(vecAxSp,:,:))),[],1);
imgOcclResPopPreTrSd = std(squeeze(mean(imgOcclResPopPreTr(vecAxSt,:,:))-mean(imgOcclResPopPreTr(vecAxSp,:,:))),[],1);

imgFullResPopPostTr = reshape(imgFullResPopPost,[size(imgFullResPopPost,1),size(imgFullResPopPost,2)*size(imgFullResPopPost,3),size(imgFullResPopPost,4)]);
imgOcclResPopPostTr = reshape(imgOcclResPopPost,[size(imgOcclResPopPost,1),size(imgOcclResPopPost,2)*size(imgOcclResPopPost,3),size(imgOcclResPopPost,4)]);
imgFullResPopPostTrMn = mean(squeeze(mean(imgFullResPopPostTr(vecAxSt,:,:))-mean(imgFullResPopPostTr(vecAxSp,:,:))));
imgOcclResPopPostTrMn = mean(squeeze(mean(imgOcclResPopPostTr(vecAxSt,:,:))-mean(imgOcclResPopPostTr(vecAxSp,:,:))));
imgFullResPopPostTrSd = std(squeeze(mean(imgFullResPopPostTr(vecAxSt,:,:))-mean(imgFullResPopPostTr(vecAxSp,:,:))),[],1);
imgOcclResPopPostTrSd = std(squeeze(mean(imgOcclResPopPostTr(vecAxSt,:,:))-mean(imgOcclResPopPostTr(vecAxSp,:,:))),[],1);

nTrial = size(imgFullResPopPreTr,2)-1;
SIpre = zeros(size(imgFullResPopPreTr,3),1);
for i = 1:size(imgFullResPopPreTr,3)
    Sp = ((nTrial*imgFullResPopPreTrSd(:,i)^2)+(nTrial*imgOcclResPopPreTrSd(:,i)^2))/(2*nTrial);
    SIpre(i) = (imgFullResPopPreTrMn(:,i) - imgOcclResPopPreTrMn(:,i)) / Sp;
end
nTrial = size(imgFullResPopPostTr,2)-1;
SIpost = zeros(size(imgFullResPopPostTr,3),1);
for i = 1:size(imgFullResPopPostTr,3)
    Sp = ((nTrial*imgFullResPopPostTrSd(:,i)^2)+(nTrial*imgOcclResPopPostTrSd(:,i)^2))/(2*nTrial);
    SIpost(i) = (imgFullResPopPostTrMn(:,i) - imgOcclResPopPostTrMn(:,i)) / Sp;
end

SIprePop = sqrt(mean(SIpre.^2));
SIpostPop = sqrt(mean(SIpost.^2));

figure, 
histogram(SIpre, 'Normalization', 'Probability'), hold on
histogram(SIpost, 'Normalization', 'Probability')

figure
scatter(SIpre, max([imgFullResPopPreTrMn; imgOcclResPopPreTrMn]))
figure
scatter(SIpost, max([imgFullResPopPostTrMn; imgOcclResPopPostTrMn]))

%% % %======== Local helper functions for displacement diagnostics ========% % %
function make_pre_post_displacement_diagnostics(preX, preY, postX, postY, colors, figName, xPreLab, yPreLab, xPostLab, yPostLab)
    % Creates:
    % 1) a 4-panel main figure: pre, post, individual displacements, post k-means + mean displacements
    % 2) a 3-panel angle diagnostics figure: angles colored by pre color, angles colored by angle clusters, mean angle-cluster vectors

    k = 3;
    sz = 45;

    preX = preX(:); preY = preY(:); postX = postX(:); postY = postY(:);
    dx = postX - preX;
    dy = postY - preY;

    % Main 4-panel figure
    figure('Position', [35 360 1900 520]);

    ax1 = subplot(1,4,1);
    scatter(ax1, preX, preY, sz, colors, 'filled'); hold(ax1, 'on')
    format_response_axis(ax1);
    draw_diag45_local(ax1);
    xlabel(ax1, xPreLab); ylabel(ax1, yPreLab);
    title(ax1, [figName ' pre — colored by pre response blend']);

    ax2 = subplot(1,4,2);
    scatter(ax2, postX, postY, sz, colors, 'filled'); hold(ax2, 'on')
    format_response_axis(ax2);
    draw_diag45_local(ax2);
    xlabel(ax2, xPostLab); ylabel(ax2, yPostLab);
    title(ax2, [figName ' post — colored by pre response blend']);

    ax3 = subplot(1,4,3);
    hold(ax3, 'on')
    scatter(ax3, preX, preY, sz * 0.35, colors, 'filled', 'MarkerFaceAlpha', 0.35);
    for iNeuron = 1:numel(preX)
        draw_vector_arrow(ax3, preX(iNeuron), preY(iNeuron), dx(iNeuron), dy(iNeuron), colors(iNeuron,:), 0.35, 0.055, 0.020);
    end
    format_response_axis(ax3);
    draw_diag45_local(ax3);
    xlabel(ax3, '\Delta starts at pre NO'); ylabel(ax3, '\Delta starts at pre O');
    title(ax3, [figName ' individual pre \rightarrow post displacements']);

    ax4 = subplot(1,4,4);
    draw_post_kmeans_mean_displacements(ax4, preX, preY, postX, postY, k, figName, xPostLab, yPostLab);

    % Separate angle-diagnostics figure
    angleClusterIdx = cluster_displacement_angles(dx, dy, k);
    clusterColors = lines(k);

    figure('Position', [70 70 1500 470]);

    ax5 = subplot(1,3,1);
    draw_displacement_angle_circle(ax5, dx, dy, colors, [], [figName ' displacement angles — pre color']);

    ax6 = subplot(1,3,2);
    draw_displacement_angle_circle(ax6, dx, dy, clusterColors, angleClusterIdx, [figName ' displacement angles — angle clusters']);

    ax7 = subplot(1,3,3);
    draw_angle_cluster_mean_vectors(ax7, dx, dy, angleClusterIdx, clusterColors, [figName ' mean vectors by angle cluster']);
end

function draw_post_kmeans_mean_displacements(ax, preX, preY, postX, postY, k, figName, xPostLab, yPostLab)
    axes(ax); hold(ax, 'on')
    dx = postX - preX;
    dy = postY - preY;
    taskXY = [postX(:), postY(:)];
    valid = all(isfinite(taskXY), 2) & isfinite(dx(:)) & isfinite(dy(:));

    clusterIdxFull = nan(size(postX(:)));
    if sum(valid) >= k
        clusterIdxFull(valid) = kmeans(taskXY(valid,:), k, 'Replicates', 50, 'MaxIter', 1000, 'Display', 'off');
    else
        warning('Not enough valid points for k-means in %s post-position clustering.', figName);
    end

    clusterColors = lines(k);
    hLeg = gobjects(k,1);
    legTxt = cell(k,1);

    for ci = 1:k
        ix = clusterIdxFull == ci;
        if ~any(ix)
            continue
        end
        hLeg(ci) = scatter(ax, postX(ix), postY(ix), 45, clusterColors(ci,:), 'filled', 'MarkerFaceAlpha', 0.75);

        meanPreX = mean(preX(ix), 'omitnan');
        meanPreY = mean(preY(ix), 'omitnan');
        meanDx = mean(dx(ix), 'omitnan');
        meanDy = mean(dy(ix), 'omitnan');
        draw_vector_arrow(ax, meanPreX, meanPreY, meanDx, meanDy, clusterColors(ci,:), 0.65, 0.070, 0.026);
        legTxt{ci} = sprintf('K%d, n = %d', ci, sum(ix));
    end

    format_response_axis(ax);
    draw_diag45_local(ax);
    xlabel(ax, xPostLab); ylabel(ax, yPostLab);
    title(ax, [figName ' post k-means + mean displacements']);

    validLeg = isgraphics(hLeg);
    if any(validLeg)
        legend(ax, hLeg(validLeg), legTxt(validLeg), 'Location', 'best');
    end
end

function angleClusterIdxFull = cluster_displacement_angles(dx, dy, k)
    theta = atan2(dy(:), dx(:));
    valid = isfinite(theta) & hypot(dx(:), dy(:)) > eps;
    angleClusterIdxFull = nan(size(theta));

    if sum(valid) >= k
        angleXY = [cos(theta(valid)), sin(theta(valid))];
        angleClusterIdxFull(valid) = kmeans(angleXY, k, 'Replicates', 50, 'MaxIter', 1000, 'Display', 'off');
    else
        warning('Not enough valid displacement vectors for angle-based k-means.');
    end
end

function draw_displacement_angle_circle(ax, dx, dy, colors, clusterIdx, ttl)
    axes(ax); hold(ax, 'on')
    theta = atan2(dy(:), dx(:));
    valid = isfinite(theta) & hypot(dx(:), dy(:)) > eps;
    x = cos(theta);
    y = sin(theta);

    th = linspace(0, 2*pi, 400);
    plot(ax, cos(th), sin(th), 'k-', 'LineWidth', 0.5);
    plot(ax, [-1.1 1.1], [0 0], 'k:', 'LineWidth', 0.5);
    plot(ax, [0 0], [-1.1 1.1], 'k:', 'LineWidth', 0.5);

    if isempty(clusterIdx)
        scatter(ax, x(valid), y(valid), 42, colors(valid,:), 'filled', 'MarkerFaceAlpha', 0.85);
    else
        k = size(colors, 1);
        hLeg = gobjects(k,1);
        legTxt = cell(k,1);
        for ci = 1:k
            ix = valid & clusterIdx(:) == ci;
            if ~any(ix)
                continue
            end
            hLeg(ci) = scatter(ax, x(ix), y(ix), 42, colors(ci,:), 'filled', 'MarkerFaceAlpha', 0.85);
            legTxt{ci} = sprintf('A%d, n = %d', ci, sum(ix));
        end
        validLeg = isgraphics(hLeg);
        if any(validLeg)
            legend(ax, hLeg(validLeg), legTxt(validLeg), 'Location', 'bestoutside');
        end
    end

    axis(ax, 'equal')
    xlim(ax, [-1.15 1.15]); ylim(ax, [-1.15 1.15]);
    xticks(ax, -1:0.5:1); yticks(ax, -1:0.5:1);
    xlabel(ax, 'cos(angle)'); ylabel(ax, 'sin(angle)');
    title(ax, ttl);
    box(ax, 'off')
end

function draw_angle_cluster_mean_vectors(ax, dx, dy, angleClusterIdx, clusterColors, ttl)
    axes(ax); hold(ax, 'on')
    k = size(clusterColors, 1);
    hLeg = gobjects(k,1);
    legTxt = cell(k,1);

    dx = dx(:);
    dy = dy(:);
    angleClusterIdx = angleClusterIdx(:);

    maxAbs = max(abs([dx(:); dy(:)]), [], 'omitnan');
    if isempty(maxAbs) || ~isfinite(maxAbs) || maxAbs == 0
        maxAbs = 1;
    end
    lim = max(0.25, maxAbs * 1.15);

    plot(ax, [-lim lim], [0 0], 'k:', 'LineWidth', 0.5);
    plot(ax, [0 0], [-lim lim], 'k:', 'LineWidth', 0.5);

    % Individual neurons in displacement space, clustered by displacement angle.
    % Each point is one neuron: x = post NO - pre NO, y = post O - pre O.
    for ci = 1:k
        ix = angleClusterIdx == ci & isfinite(dx) & isfinite(dy);
        if ~any(ix)
            continue
        end
        hLeg(ci) = scatter(ax, dx(ix), dy(ix), 42, clusterColors(ci,:), ...
            'filled', 'MarkerFaceAlpha', 0.60, 'MarkerEdgeColor', 'none');
        legTxt{ci} = sprintf('A%d, n = %d', ci, sum(ix));
    end

    % Overlay the mean displacement vector for each angle cluster.
    for ci = 1:k
        ix = angleClusterIdx == ci & isfinite(dx) & isfinite(dy);
        if ~any(ix)
            continue
        end
        meanDx = mean(dx(ix), 'omitnan');
        meanDy = mean(dy(ix), 'omitnan');
        draw_vector_arrow(ax, 0, 0, meanDx, meanDy, clusterColors(ci,:), 0.65, 0.070, 0.026);
    end

    xlim(ax, [-lim lim]); ylim(ax, [-lim lim]);
    axis(ax, 'square')
    xlabel(ax, '\Delta NO'); ylabel(ax, '\Delta O');
    title(ax, ttl);
    validLeg = isgraphics(hLeg);
    if any(validLeg)
        legend(ax, hLeg(validLeg), legTxt(validLeg), 'Location', 'best');
    end
    box(ax, 'off')
end
function h = draw_vector_arrow(ax, x0, y0, dx, dy, colorVal, shaftWidth, headLength, headWidth)
    % Draw a vector with an independently controllable thin shaft and thicker arrowhead.
    % headLength/headWidth are fractions of the current axis range.
    if nargin < 7 || isempty(shaftWidth), shaftWidth = 0.5; end
    if nargin < 8 || isempty(headLength), headLength = 0.070; end
    if nargin < 9 || isempty(headWidth), headWidth = 0.026; end

    if ~all(isfinite([x0 y0 dx dy]))
        h = gobjects(1);
        return
    end

    x1 = x0 + dx;
    y1 = y0 + dy;
    vLen = hypot(dx, dy);
    if vLen <= eps
        h = plot(ax, x0, y0, '.', 'Color', colorVal, 'MarkerSize', 8);
        return
    end

    xl = xlim(ax); yl = ylim(ax);
    if any(~isfinite([xl yl])) || diff(xl) == 0 || diff(yl) == 0
        rangeScale = 1;
    else
        rangeScale = mean([diff(xl), diff(yl)]);
    end
    hl = min(headLength * rangeScale, 0.45 * vLen);
    hw = headWidth * rangeScale;

    ux = dx / vLen;
    uy = dy / vLen;
    px = -uy;
    py = ux;

    xb = x1 - hl * ux;
    yb = y1 - hl * uy;

    h = plot(ax, [x0 xb], [y0 yb], '-', 'Color', colorVal, 'LineWidth', shaftWidth, 'Clipping', 'on');
    patch(ax, [x1, xb + hw*px, xb - hw*px], [y1, yb + hw*py, yb - hw*py], colorVal, ...
        'EdgeColor', colorVal, 'FaceColor', colorVal, 'FaceAlpha', 0.9, 'Clipping', 'on');
end

function format_response_axis(ax)
    xlim(ax, [-1 3]); ylim(ax, [-1 3]);
    xticks(ax, -1:1:3); yticks(ax, -1:1:3);
    axis(ax, 'square')
    box(ax, 'off')
end

function draw_diag45_local(ax)
    axes(ax); hold(ax,'on');
    xl = xlim(ax); yl = ylim(ax);
    mn = max(xl(1), yl(1));
    mx = min(xl(2), yl(2));
    plot(ax, [mn mx], [mn mx], 'k--', 'LineWidth', 0.5, 'Clipping','on');
end
