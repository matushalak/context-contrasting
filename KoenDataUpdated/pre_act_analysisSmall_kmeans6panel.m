% This plots the data for chronically matched cells from naive and active 
% task recordings. 110 ROIs of 6 mice: 
% 1. Archer
% 2. Charyl
% 3. Gilette
% 4. Lana
% 5. Mallory
% 6. Pam
% 
% Koen Seignette, made small data and quick plot by Leander de Kraker
% Original script: occl_preactive_chronic_rfssep_zscorefinal_revisions
% 2026-4-16
% 
%% 
clear
clc
close all

%%
load('pre_act_dataSmall.mat')

%% plotting
% color pallets for plotting
col1 = [0, 0 ,0]; % black
col2 = [131, 197, 190]/255; % blue/greenish

imgIdx = [1 2 3 4];
% imgIdx = [4];

vecAxSp = vecAx<0; % spontaneous activity window
vecAxSt = vecAx>0.2 & vecAx<1; % stim window

imgFullRes = squeeze(nanmean(imgFullResMnPop(:,imgIdx,:),2));
imgFullResBsl = imgFullRes-nanmean(imgFullRes(vecAxSp,:));
imgOcclRes = squeeze(nanmean(imgOcclResMnPop(:,imgIdx,:),2));
imgOcclResBsl = imgOcclRes-nanmean(imgOcclRes(vecAxSp,:));
scatFullPop = nanmean(imgFullResBsl(vecAxSt,:));
scatOcclPop = nanmean(imgOcclResBsl(vecAxSt,:));

% cut off value just for plotting purposes
scatFullPopCut = scatFullPop;
scatOcclPopCut = scatOcclPop;
mnValCut = -0.5; % min val for cutting for plotting
mxValCut = 2.5; % max val for cutting for plotting
scatFullPopCut(scatFullPopCut>mxValCut)=mxValCut+0.5;
scatFullPopCut(scatFullPopCut<mnValCut)=mnValCut-0.5;
scatOcclPopCut(scatOcclPopCut>mxValCut)=mxValCut+0.5;
scatOcclPopCut(scatOcclPopCut<mnValCut)=mnValCut-0.5;

sz = 8;
cPre = [0 0 0];
cPost = col2;

% plot traces and scatters in one figure
figure('Position', [87         321        1635         579])
clear t s
% traces
t(1) = subplot(2,5,1);
shadedErrorBar(vecAx,mean(imgFullResBsl,2)...
    ,std(imgFullResBsl,0,2)/sqrt(size(imgFullResBsl,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAx,mean(imgOcclResBsl,2)...
    ,std(imgOcclResBsl,0,2)/sqrt(size(imgOcclResBsl,2)), 'lineProps', 'r');
xlabel('Time (s)'), ylabel('Z-scored dF/F'), title('Active'), xlim([-1 3]),
% scatters
s(1) = subplot(2,5,6);
scatter(scatFullPopCut, scatOcclPopCut, sz, cPre, 'filled'); refline(1), 
xlabel('Full res'), ylabel('Occl res')
% mean box plot
% subplot(2,5,3)
% boxchart([ones(size(scatFullPop)), ones(size(scatOcclPop))+1], ...
%     [scatFullPop, scatOcclPop], 'MarkerStyle','none'), hold on
% xlim([0 3]), ylabel('Response dF/F (%)'), xticks([1 2]), if nfiles == 6, ylim([-10 10]), elseif nfiles == 5, ylim([-5 35]), end
% xticklabels({'Full', 'Occl'}), xtickangle(45), 
% mean scat/bar
subplot(2,5,7)
scatter([1 2],[mean(scatFullPop) mean(scatOcclPop)], 35, 'k', 'LineWidth', 2), hold on
er = errorbar([1 2],[mean(scatFullPop) mean(scatOcclPop)], ...
    [calcSem(scatFullPop) calcSem(scatOcclPop)] ...
    ,[calcSem(scatFullPop) calcSem(scatOcclPop)]);
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 3]), ylabel('Z-scored dF/F'), xticks([1 2]), if nfiles == 6, ylim([0 0.4]), elseif nfiles == 5, ylim([0 0.4]), end
xticklabels({'Full', 'Occl'}), 
xtickangle(45), 

% Adjusting y-axes for subplots 1-4
yMax = max([ylim(t)]);
yMin = min([ylim(t)]);
set(t(1), 'YLim', [-0.1 0.6]);
set(t(1), 'XLim', [-1 3])

% axes
for j = 1:length(s)
    s(j).YLim = [-1 3]; 
    s(j).YTick = -1:0.5:3; 
    s(j).XLim = [-1 3]; 
    s(j).XTick = -1:0.5:3;
end

%%%%%% snake plots / imagesc of all cells, average responses
% sort on trace of preference pre training fam
traceToSort = imgFullResBsl;
[Mni] = mean(traceToSort(vecAxSt,:));
[~,RsortedMn] = sort(Mni,'descend');
[~, Mxi] = max(traceToSort);
[~,RsortedMx] = sort(Mxi,'ascend');

% plot with each condition in separate subplot, axes are similar scaling
clear p
figure('Position', [1079         265         485         713])
p(1) = subplot(1,3,1);
imagesc(vecAx, [], imgFullResBsl(:, RsortedMn)')
title('Pre Fam Full'),  xlabel('Time (s)'), ylabel('Neurons'),xlim([-1 3]), set(gca,'TickDir','out'), box off
p(2) = subplot(1,3,2);
imagesc(vecAx, [], imgOcclResBsl(:, RsortedMn)')
title('Pre Fam Occl'),  xlabel('Time (s)'), xlim([-1 3]), set(gca,'TickDir','out', 'YTickLabel', []), box off
mn = -0.4; 
mx = 3;
set(p, 'CLim', [mn, mx]); % for L2/3
colormap hot
subplot(1,3,3)
axis off, colormap hot, caxis([mn mx]), colorbar

ffTask = scatFullPop;
ofTask = scatOcclPop;

% we put values <-1 at -1 and >1 at 1 after SI calculation. Doesn't matter whether you do it like this or whether you make
% negative values at 0 before calculating selectivity index.
siFamTask = (ffTask-ofTask)./(ffTask+ofTask); 
siFamTask(isnan(siFamTask))=0; 
siFamTask(siFamTask<-1)=-1; 
siFamTask(siFamTask>1)=1;

% some more plotting
alpha = 1;
sz = 10;
cPre = [0.2 0.2 0.2];

% % correlate absolute selectivity to response strength (max of full or occl)
siFamPreAbs = abs(siFamTask);
mxfPre = max([ffTask; ofTask]);

figure
subplot(1,2,1)
scatter(1,nanmean(siFamPreAbs), 45, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar(1,nanmean(siFamPreAbs), calcSem(siFamPreAbs), calcSem(siFamPreAbs));    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 2]), ylim([0 0.7]), ylabel('Absolute SI fam')
xticks(1), xticklabels({'Task'}), xtickangle(45), 

figure('Position', [266   311   705   685])
subplot(2,2,1)
scatter(siFamPreAbs, mxfPre, 20, 'k', 'filled'), ylim([-1 3.5])
title('Fam pre'),xlabel('SI'), ylabel('Max response'), 

% divide in two bin, <0.5 si and >0.5 si for statistics
thres = 0.50001;
mxFamPreLow = mxfPre(siFamPreAbs<thres);
mxFamPreHigh = mxfPre(siFamPreAbs>thres);
figure('Position', [680   430   392   548])
scatter([1 2],[nanmean(mxFamPreLow) nanmean(mxFamPreHigh)], 45, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2],[nanmean(mxFamPreLow) nanmean(mxFamPreHigh)], ...
    [calcSem(mxFamPreLow) calcSem(mxFamPreHigh)] ...
    ,[calcSem(mxFamPreLow) calcSem(mxFamPreHigh)]);    
er.Color = [0 0 0]; 
er.LineStyle = 'none'; 
er.LineWidth = 2; 
er.CapSize = 0;
xlim([0 3]), ylabel('Response'), xticks([1 2]), ylim([0 0.7])
xticklabels({'PreFamLow', 'PreFamHigh'}), xtickangle(45), 

% divide in two bins based on response strenght, then look at SI
thres = 1.00001; % in df/f
siFamPreLow = siFamPreAbs(mxfPre<thres);
siFamPreHigh = siFamPreAbs(mxfPre>thres);
figure('Position', [680   430   392   548])
scatter([1 2],[nanmean(siFamPreLow) nanmean(siFamPreHigh)], 45, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2],[nanmean(siFamPreLow) nanmean(siFamPreHigh)], ...
    [calcSem(siFamPreLow) calcSem(siFamPreHigh)] ...
    ,[calcSem(siFamPreLow) calcSem(siFamPreHigh)]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 3]), ylabel('Response'), xticks([1 2]), ylim([0 1])
xticklabels({'TaskFamLow', 'TaskFamHigh'}); 
xtickangle(45), 

figure('Position', [421   301   800   317])
subplot(1,2,1);
% histogram(siFamPre, lims(1):0.5:lims(2),'Normalization', 'probability', 'FaceColor', 'k'), hold on
% histogram(siFamPost, lims(1):0.5:lims(2),'Normalization', 'probability', 'FaceColor', 'w', 'EdgeColor', 'k')
histogram(siFamTask, -1:0.1:1,'Normalization', 'probability', 'FaceColor', 'k'), hold on
xline(mean(siFamTask)), ylabel('Relative frequency'), xlabel('Selectivity index'),title('Familiar'),

figure('Position', [87         278        1635         673])
subplot(2,5,1)
scatter(siFamPreAbs, mxfPre, 15, 'k', 'filled'), ylim([-1 3.5])
title('Fam pre'),xlabel('SI'), ylabel('Max response'), 
subplot(2,5,3)
boxchart([ones(size(mxFamPreLow)), 2*ones(size(mxFamPreHigh))], ...
          [mxFamPreLow, mxFamPreHigh], 'MarkerStyle', 'none');
ylabel('Max response'); xlim([0 10]); xticks([1 2 3 4  6 7 8 9]);
xticklabels({'TaskFamLow', 'TaskFamHigh'}); 
xtickangle(45); ylim([0 1]); ;
if nfiles == 6
    ylim([-1 2.5]);
elseif nfiles == 5
    ylim([-0.3 1.8]);
end
;

subplot(2,5,4);
boxchart([ones(size(siFamPreLow)), 2*ones(size(siFamPreHigh))], ...
          [siFamPreLow, siFamPreHigh], 'MarkerStyle', 'none');
ylabel('Selectivity'); xlim([0 10]); xticks([1 2 3 4  6 7 8 9]);
xticklabels({'TaskFamLow', 'TaskFamHigh'}); 
xtickangle(45); ylim([0 1]); ;

[p,h] = ranksum(siFamPreLow,siFamPreHigh)

[p,h] = ranksum(mxFamPreLow, mxFamPreHigh)

%% pre dataset
% famIdx = [1 2 4 5];

vecAxPreSt = vecAxPre>0.2 & vecAxPre<1;
vecAxPreSp = vecAxPre<0;

vecAxTask = vecAx;
vecAxTaskSp = vecAxTask<0;
vecAxTaskSt = vecAxTask>0.2 & vecAxTask<1;

imgFullResMnPopPreBsl = imgFullResMnPopPre-mean(imgFullResMnPopPre(vecAxPreSp,:,:));
imgOcclResMnPopPreBsl = imgOcclResMnPopPre-mean(imgOcclResMnPopPre(vecAxPreSp,:,:));
imgFullResMnPopTaskBsl = imgFullResMnPopTask-mean(imgFullResMnPopTask(vecAxTaskSp,:,:));
imgOcclResMnPopTaskBsl = imgOcclResMnPopTask-mean(imgOcclResMnPopTask(vecAxTaskSp,:,:));

imgFullResMnPopPreBslMn = squeeze(mean(imgFullResMnPopPreBsl,2));
imgOcclResMnPopPreBslMn = squeeze(mean(imgOcclResMnPopPreBsl,2));
imgFullResMnPopTaskBslMn = squeeze(mean(imgFullResMnPopTaskBsl,2));
imgOcclResMnPopTaskBslMn = squeeze(mean(imgOcclResMnPopTaskBsl,2));

imgFullPre = squeeze(mean(imgFullResMnPopPre(vecAxPreSt,:,:)))-squeeze(mean(imgFullResMnPopPre(vecAxPreSp,:,:)));
imgFullTask = squeeze(mean(imgFullResMnPopTask(vecAxTaskSt,:,:)))-squeeze(mean(imgFullResMnPopTask(vecAxTaskSp,:,:)));
imgOcclPre = squeeze(mean(imgOcclResMnPopPre(vecAxPreSt,:,:)))-squeeze(mean(imgOcclResMnPopPre(vecAxPreSp,:,:)));
imgOcclTask = squeeze(mean(imgOcclResMnPopTask(vecAxTaskSt,:,:)))-squeeze(mean(imgOcclResMnPopTask(vecAxTaskSp,:,:)));

scatFullPre = mean(imgFullPre);
scatFullTask = mean(imgFullTask);
scatOcclPre = mean(imgOcclPre);
scatOcclTask = mean(imgOcclTask);

% lifetime sparseness
sparsenessFullPre = calculateLifetimeSparseness(imgFullPre')';
sparsenessFullTask = calculateLifetimeSparseness(imgFullTask')';

sparsenessOcclPre = calculateLifetimeSparseness(imgOcclPre')';
sparsenessOcclTask = calculateLifetimeSparseness(imgOcclTask')';

% ixUp = scatFullFamPopPost>0.4 & scatFullFamPopPre<0.4;
ixUp = scatFullTask>0.2 & (scatFullTask-scatFullPre)>0.1;
ixDown = scatFullTask>0.2 & (scatFullTask-scatFullPre)<0.1;

mean(sparsenessFullTask(ixUp))

sum(ixUp)
sum(ixDown)

ix = scatFullTask>0.2;
inc = scatFullTask-scatFullPre;

figure('Position', [1090         247         560         420])
scatter(sparsenessFullTask(ix), inc(ix))
refline(1), refline, xlabel('Sparseness full task'), ylabel('Response full task - response full pre')

% Group 1: ~ixUp
means1 = [nanmean(sparsenessFullTask(ixUp)), nanmean(sparsenessFullTask(ixDown))];
sems1  = [nanstd(sparsenessFullTask(ixUp)) / sqrt(sum(~isnan(sparsenessFullTask(ixDown)))), ...
          nanstd(sparsenessFullTask(ixUp)) / sqrt(sum(~isnan(sparsenessFullTask(ixDown))))];

figure('Position', [509   252   560   420])
% Bar plots
bar([1 2], means1, 'FaceAlpha', 0.5); hold on
% Error bars
errorbar([1 2], means1, sems1, 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 8)

%% chronic colored plot pre vs active (PAPER FIGURE 2I)

sz = 20;
cPre = [0.2 0.2 0.2];
cPost = col2;

% plot traces and scatters in one figure
figure('Position', [364   231   896   621])
clear t s g

% traces
t(1) = subplot(2,2,1);
shadedErrorBar(vecAxPre,mean(imgFullResMnPopPreBslMn,2)...
    ,std(imgFullResMnPopPreBslMn,0,2)/sqrt(size(imgFullResMnPopPreBslMn,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAxPre,mean(imgOcclResMnPopPreBslMn,2)...
    ,std(imgOcclResMnPopPreBslMn,0,2)/sqrt(size(imgOcclResMnPopPreBslMn,2)), 'lineProps', 'r');
xlabel('Time (s)'), ylabel('dF/F (%)'), xticks(-1:1:3), xlim([-1 3]), title('Pre Fam'), 
t(2) = subplot(2,2,2);
shadedErrorBar(vecAx,mean(imgFullResMnPopTaskBslMn,2)...
    ,std(imgFullResMnPopTaskBslMn,0,2)/sqrt(size(imgFullResMnPopTaskBslMn,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAx,mean(imgOcclResMnPopTaskBslMn,2)...
    ,std(imgOcclResMnPopTaskBslMn,0,2)/sqrt(size(imgOcclResMnPopTaskBslMn,2)), 'lineProps', 'r');
xlabel('Time (s)'), title('Post Fam'), 
% scatters
s(1) = subplot(2,2,3);
scatter(scatFullPre, scatOcclPre, sz, cPre, 'filled'); refline(1), xlabel('Full'), ylabel('Occl'), title('Pre'), 
s(2) = subplot(2,2,4);
scatter(scatFullTask,scatOcclTask , sz, cPost, 'filled'); refline(1), title('Task'), 
for j = 1:length(s)
    s(j).YLim = [-1 3]; 
    s(j).YTick = -1:0.5:3; 
    s(j).XLim = [-1 3]; 
    s(j).XTick = -1:0.5:3;
end
for j = 1:length(t)
    t(j).YLim = [-0.2 0.6]; 
    t(j).YTick = -0.2:0.1:0.6; 
    t(j).XLim = [-1 3]; 
    t(j).XTick = -1:0.5:3;
end

% Normalize pre responses
fullNorm = normalize(scatFullPre, 'range');
occlNorm = normalize(scatOcclPre, 'range');

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
scatFullPreSort = scatFullPre(sortIdx);
scatOcclPreSort = scatOcclPre(sortIdx);
scatFullTaskSort = scatFullTask(sortIdx);
scatOcclTaskSort = scatOcclTask(sortIdx);

figure('Position', [20 380 2300 520]);
sz = 45;
angleClusterK = 3;

% ---------- Pre ----------
axPreColored = subplot(1,6,1);
scatter(axPreColored, scatFullPreSort, scatOcclPreSort, sz, colors, 'filled');
hold(axPreColored, 'on')
xlim(axPreColored, [-1 3]); ylim(axPreColored, [-1 3]);
xticks(axPreColored, -1:1:3); yticks(axPreColored, -1:1:3);
draw_diag45(axPreColored);
xlabel(axPreColored, 'NO fam pre'); ylabel(axPreColored, 'O fam pre');
title(axPreColored, 'Pre — colored by pre response blend');
axis(axPreColored, 'square')

% ---------- Task ----------
axTaskColored = subplot(1,6,2);
scatter(axTaskColored, scatFullTaskSort, scatOcclTaskSort, sz, colors, 'filled');  % same colors
hold(axTaskColored, 'on')
xlim(axTaskColored, [-1 3]); ylim(axTaskColored, [-1 3]);
xticks(axTaskColored, -1:1:3); yticks(axTaskColored, -1:1:3);
draw_diag45(axTaskColored);
xlabel(axTaskColored, 'NO fam task'); ylabel(axTaskColored, 'O fam task');
title(axTaskColored, 'Task — colored by pre response blend');
axis(axTaskColored, 'square')

% ---------- Individual displacement vectors: Pre -> Task ----------
axDispColored = subplot(1,6,3);
draw_colored_displacements(axDispColored, scatFullPreSort, scatOcclPreSort, ...
    scatFullTaskSort, scatOcclTaskSort, colors, sz, 'Individual Pre -> Task displacement');

% ---------- k-means clusters on Task scatter + mean displacement vectors ----------
axKMeansColored = subplot(1,6,4);
draw_kmeans_mean_displacements(axKMeansColored, scatFullPreSort, scatOcclPreSort, ...
    scatFullTaskSort, scatOcclTaskSort, colors, 3, sz, 'k = 3 Task clusters: mean displacement');

% ---------- Displacement vector angles on unit circle ----------
axAngleCircleColored = subplot(1,6,5);
draw_displacement_angle_circle(axAngleCircleColored, scatFullPreSort, scatOcclPreSort, ...
    scatFullTaskSort, scatOcclTaskSort, colors, sz, 'Displacement vector angles');

% ---------- Angular clusters + mean displacement vectors ----------
axAngleClusterColored = subplot(1,6,6);
draw_angle_cluster_mean_displacements(axAngleClusterColored, scatFullPreSort, scatOcclPreSort, ...
    scatFullTaskSort, scatOcclTaskSort, angleClusterK, sz, 'Angle clusters: Task points + mean vectors');

% Create colorbar figure (un-normalized version)
figure('Position', [1203         495         400         345]);

% Define range for raw values
fullRange = linspace(min(scatFullPreSort), max(scatFullPreSort), 256);
occlRange = linspace(min(scatOcclPreSort), max(scatOcclPreSort), 256);
[fullGrid, occlGrid] = meshgrid(fullRange, occlRange);

% Apply cutoff-based clamping
fullClamped = min(fullGrid, fullCutoff) / fullCutoff;
occlClamped = min(occlGrid, occlCutoff) / occlCutoff;

% Build RGB color image (Red = fullClamped, Blue = occlClamped)
colorGrid = cat(3, fullClamped, zeros(size(fullClamped)), occlClamped);

% Display the color grid
image(fullRange, occlRange, colorGrid);
axis xy;
xlabel('Full response naive');
ylabel('Occl response naive');
title('Color blending: Red = Full, Blue = Occl');
set(gca, 'XTickMode', 'auto', 'YTickMode', 'auto');

%% just the neurons that we indicate in the paper:
% ---- Setup & original multi-panel figure ----

% ---- Selection handling ----
selection = [1:110];           % original neuron indices to highlight in THIS order
% % selection = [32 43 76 51 21 80 91 29 82 4 76 71 1 3 44 21 90 4 9]; % quick Leander selection
selection = selection(ismember(selection, sortIdx)); % keep only IDs that exist

sz = 20;
cPre = [0.2 0.2 0.2];
cPost = col2;                 % assumes this is defined elsewhere

figure('Position', [364   231   896   621])
clear t s g

% traces
t(1) = subplot(2,2,1);
shadedErrorBar(vecAxPre, mean(imgFullResMnPopPreBslMn,2) ...
    , std(imgFullResMnPopPreBslMn,0,2)/sqrt(size(imgFullResMnPopPreBslMn,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAxPre, mean(imgOcclResMnPopPreBslMn,2) ...
    , std(imgOcclResMnPopPreBslMn,0,2)/sqrt(size(imgOcclResMnPopPreBslMn,2)), 'lineProps', 'r');
xlabel('Time (s)'), ylabel('dF/F (%)'), xticks(-1:1:3), xlim([-1 3]), title('Pre Fam'), 

t(2) = subplot(2,2,2);
shadedErrorBar(vecAx, mean(imgFullResMnPopTaskBslMn,2) ...
    , std(imgFullResMnPopTaskBslMn,0,2)/sqrt(size(imgFullResMnPopTaskBslMn,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAx, mean(imgOcclResMnPopTaskBslMn,2) ...
    , std(imgOcclResMnPopTaskBslMn,0,2)/sqrt(size(imgOcclResMnPopTaskBslMn,2)), 'lineProps', 'r');
xlabel('Time (s)'), title('Post Fam'), 

% scatters (raw, before color coding)
s(1) = subplot(2,2,3);
scatter(scatFullPre, scatOcclPre, sz, cPre, 'filled'); draw_diag45(gca); xlabel('Full'), ylabel('Occl'), title('Pre'), 
s(2) = subplot(2,2,4);
scatter(scatFullTask, scatOcclTask, sz, cPost, 'filled'); draw_diag45(gca); title('Task'), 

for j = 1:length(s)
    s(j).YLim = [-1 3]; s(j).YTick = -1:0.5:3; s(j).XLim = [-1 3]; s(j).XTick = -1:0.5:3;
end
for j = 1:length(t)
    t(j).YLim = [-0.2 0.6]; t(j).YTick = -0.2:0.1:0.6; t(j).XLim = [-1 3]; t(j).XTick = -1:0.5:3;
end

% ---- Build colors from PRE responses, then sort (defines plotting order) ----
% Force column vectors for shape safety
scatFullPre  = scatFullPre(:);
scatOcclPre  = scatOcclPre(:);
scatFullTask = scatFullTask(:);
scatOcclTask = scatOcclTask(:);

% Normalize pre responses (for color blend)
fullNorm = normalize(scatFullPre, 'range');
occlNorm = normalize(scatOcclPre, 'range');

% Saturation thresholds
fullCutoff = 0.6;
occlCutoff = 0.6;

% Apply cutoff-based scaling
fullClamped = min(fullNorm, fullCutoff) / fullCutoff;
occlClamped = min(occlNorm, occlCutoff) / occlCutoff;

% Color coding: full → red, occl → blue
N = numel(fullNorm);
colors = [fullClamped, zeros(N,1), occlClamped];

% Sort for plotting (ascending magnitude) — THIS defines original 2nd-plot order
magnitude = sqrt(fullNorm.^2 + occlNorm.^2);
[~, sortIdx] = sort(magnitude, 'ascend');           % sorted position -> original neuron index
sortIdx = sortIdx(:);                                % column

colors           = colors(sortIdx, :);
scatFullPreSort  = scatFullPre(sortIdx);
scatOcclPreSort  = scatOcclPre(sortIdx);
scatFullTaskSort = scatFullTask(sortIdx);
scatOcclTaskSort = scatOcclTask(sortIdx);

% For each sorted position, does it belong to selection? 'selOrderIdx' gives selection number (1..K)
[selMaskSorted, selOrderIdx] = ismember(sortIdx, selection);   % selOrderIdx==0 => not selected
selMaskSorted = selMaskSorted(:);
selOrderIdx   = selOrderIdx(:);

% Plot order for ALL-points figure: non-selected first (preserve order), then selected (on top)
idxNonSel = find(~selMaskSorted);
idxSel    = find(selMaskSorted);
plotOrder = [idxNonSel(:); idxSel(:)];

% Reordered copies for plotting (leave *Sort arrays intact)
colorsPlot         = colors(plotOrder, :);
scatFullPrePlot    = scatFullPreSort(plotOrder);
scatOcclPrePlot    = scatOcclPreSort(plotOrder);
scatFullTaskPlot   = scatFullTaskSort(plotOrder);
scatOcclTaskPlot   = scatOcclTaskSort(plotOrder);

% Convenience: selection positions & numbering in the *sorted* space
selPos_sorted   = find(selMaskSorted);               % positions within *Sort arrays
selNums_inOrder = selOrderIdx(selPos_sorted);        % 1..K, in same order as selPos_sorted

% ---- ALL NEURONS: colored scatter with selection on top + leader lines (numbers) on PRE & TASK ----
figure('Position', [20 380 2300 520]);
sz = 45;
angleClusterK = 3;

% ---------- Pre (ALL) ----------
ax1 = subplot(1,6,1);
scatter(ax1, scatFullPrePlot,  scatOcclPrePlot,  sz, colorsPlot, 'filled'); hold(ax1,'on')
xlim(ax1, [-1 3]); ylim(ax1, [-1 3]); xticks(ax1, -1:1:3); yticks(ax1, -1:1:3);
draw_diag45(ax1);
xlabel(ax1,'NO fam pre'); ylabel(ax1,'O fam pre'); title(ax1,'Colored by pre response blend');
axis(ax1, 'square')

% Leader lines + selection numbers on PRE (use *Sort coords for the selected points)
preX_sel  = scatFullPreSort(selPos_sorted);
preY_sel  = scatOcclPreSort(selPos_sorted);
draw_leader_labels(ax1, preX_sel, preY_sel, selNums_inOrder);

% ---------- Task (ALL) ----------
ax2 = subplot(1,6,2);
scatter(ax2, scatFullTaskPlot, scatOcclTaskPlot, sz, colorsPlot, 'filled'); hold(ax2,'on')
xlim(ax2, [-1 3]); ylim(ax2, [-1 3]); xticks(ax2, -1:1:3); yticks(ax2, -1:1:3);
draw_diag45(ax2);
xlabel(ax2,'NO fam task'); ylabel(ax2,'O fam task'); title(ax2,'Colored by pre response blend');
axis(ax2, 'square')

% Leader lines + selection numbers on TASK
taskX_sel = scatFullTaskSort(selPos_sorted);
taskY_sel = scatOcclTaskSort(selPos_sorted);
draw_leader_labels(ax2, taskX_sel, taskY_sel, selNums_inOrder);

% ---------- Individual displacement vectors: Pre -> Task ----------
ax2b = subplot(1,6,3);
draw_colored_displacements(ax2b, scatFullPreSort, scatOcclPreSort, ...
    scatFullTaskSort, scatOcclTaskSort, colors, sz, 'Individual Pre -> Task displacement');

% ---------- k-means clusters on Task scatter + mean displacement vectors ----------
ax2c = subplot(1,6,4);
draw_kmeans_mean_displacements(ax2c, scatFullPreSort, scatOcclPreSort, ...
    scatFullTaskSort, scatOcclTaskSort, colors, 3, sz, 'k = 3 Task clusters: mean displacement');

% ---------- Displacement vector angles on unit circle (ALL) ----------
ax2d = subplot(1,6,5);
draw_displacement_angle_circle(ax2d, scatFullPreSort, scatOcclPreSort, ...
    scatFullTaskSort, scatOcclTaskSort, colors, sz, 'Displacement vector angles');

% ---------- Angular clusters + mean displacement vectors (ALL) ----------
ax2e = subplot(1,6,6);
draw_angle_cluster_mean_displacements(ax2e, scatFullPreSort, scatOcclPreSort, ...
    scatFullTaskSort, scatOcclTaskSort, angleClusterK, sz, 'Angle clusters: Task points + mean vectors');

% ---- SELECTION-ONLY: same colors + order as original sorted plot, with selection numbers ----
colorsSel   = colors(selPos_sorted, :);
preX_only   = scatFullPreSort(selPos_sorted);
preY_only   = scatOcclPreSort(selPos_sorted);
taskX_only  = scatFullTaskSort(selPos_sorted);
taskY_only  = scatOcclTaskSort(selPos_sorted);
selNumsLab  = selNums_inOrder;   % 1..K

figure('Position', [20 80 2300 520]);
sz = 55;
angleClusterK = 3;

% ---------- Pre (SELECTION ONLY) ----------
ax3 = subplot(1,6,1);
scatter(ax3, preX_only, preY_only, sz, colorsSel, 'filled'); hold(ax3,'on')
xlim(ax3, [-1 3]); ylim(ax3, [-1 3]); xticks(ax3, -1:1:3); yticks(ax3, -1:1:3);
draw_diag45(ax3);
xlabel(ax3,'NO fam pre'); ylabel(ax3,'O fam pre');
title(ax3,'Selection only — colored by pre response blend');
axis(ax3, 'square')
draw_leader_labels(ax3, preX_only, preY_only, selNumsLab);

% ---------- Task (SELECTION ONLY) ----------
ax4 = subplot(1,6,2);
scatter(ax4, taskX_only, taskY_only, sz, colorsSel, 'filled'); hold(ax4,'on')
xlim(ax4, [-1 3]); ylim(ax4, [-1 3]); xticks(ax4, -1:1:3); yticks(ax4, -1:1:3);
draw_diag45(ax4);
xlabel(ax4,'NO fam task'); ylabel(ax4,'O fam task');
title(ax4,'Selection only — colored by pre response blend');
axis(ax4, 'square')
draw_leader_labels(ax4, taskX_only, taskY_only, selNumsLab);

% ---------- Individual displacement vectors: Pre -> Task (SELECTION ONLY) ----------
ax5 = subplot(1,6,3);
draw_colored_displacements(ax5, preX_only, preY_only, taskX_only, taskY_only, ...
    colorsSel, sz, 'Selection only — individual displacement');
draw_leader_labels(ax5, taskX_only, taskY_only, selNumsLab);

% ---------- k-means clusters on Task scatter + mean displacement vectors (SELECTION ONLY) ----------
ax6 = subplot(1,6,4);
draw_kmeans_mean_displacements(ax6, preX_only, preY_only, taskX_only, taskY_only, ...
    colorsSel, 3, sz, 'Selection only — k = 3 mean displacement');

% ---------- Displacement vector angles on unit circle (SELECTION ONLY) ----------
ax7 = subplot(1,6,5);
draw_displacement_angle_circle(ax7, preX_only, preY_only, taskX_only, taskY_only, ...
    colorsSel, sz, 'Selection only — displacement angles');

% ---------- Angular clusters + mean displacement vectors (SELECTION ONLY) ----------
ax8 = subplot(1,6,6);
draw_angle_cluster_mean_displacements(ax8, preX_only, preY_only, taskX_only, taskY_only, ...
    angleClusterK, sz, 'Selection only — angle clusters');
% ---- Colorbar figure (un-normalized axes, using sorted ranges) ----
figure('Position', [1203 495 400 345]);
fullRange = linspace(min(scatFullPreSort), max(scatFullPreSort), 256);
occlRange = linspace(min(scatOcclPreSort), max(scatOcclPreSort), 256);
[fullGrid, occlGrid] = meshgrid(fullRange, occlRange);
fullClampedGrid = min(fullGrid, fullCutoff) / fullCutoff;
occlClampedGrid = min(occlGrid, occlCutoff) / occlCutoff;
colorGrid = cat(3, fullClampedGrid, zeros(size(fullClampedGrid)), occlClampedGrid);
image(fullRange, occlRange, colorGrid); axis xy
xlabel('Full response naive'); ylabel('Occl response naive');
title('Color blending: Red = Full, Blue = Occl');
set(gca, 'XTickMode', 'auto', 'YTickMode', 'auto');

%% Build data arrays
% fullDataPre = cat(4, datastructPre(:).imgFullRes);   % [frames x images x reps x neurons]
% occlDataPre = cat(4, datastructPre(:).imgOcclRes);
% 
% fullDataTask = cat(4, datastructActive(:).imgFullRes);
% occlDataTask = cat(4, datastructActive(:).imgOcclRes);

fullDataPre = imgFullResMnPopPre;   % [frames x images x reps x neurons]
occlDataPre = imgOcclResMnPopPre;

fullDataTask = imgFullResMnPopTask;
occlDataTask = imgOcclResMnPopTask;

% Setup
nNeurons = size(fullDataPre, 3);
nImgs = size(fullDataPre, 2);
nPerPage = 15;
% Optional: only plot selected neurons
if exist('selection', 'var') && ~isempty(selection)
    ix = selection(:);  % force to column vector
else
    ix = 1:nNeurons;
end
nPages = ceil(length(ix) / nPerPage);

for pg = 1:nPages
%     figure('Position', [74         266        1800         654]);
    figure('Position', [650 42 999 954])
    
    % Get neuron indices for this page
    startIdx = (pg-1)*nPerPage + 1;
    endIdx = min(pg*nPerPage, nNeurons);
    ixPage = ix(startIdx : min(endIdx, length(ix)));
    nThisPage = length(ixPage);
    
    for i = 1:nThisPage
        neuronIdx = ixPage(i);
        yValsNeuron = [];  % to collect for shared ylim
        
        mouseNamei = mouseName{roiMouseOrigin(neuronIdx)};
        matchIDi = roiMatchID(neuronIdx);
        roiIDi = roiID(neuronIdx,:);

        for j = 1:nImgs
            % --- Pre traces ---
            traceFullPre = squeeze(fullDataPre(:, j, neuronIdx) - mean(fullDataPre(vecAxPreSp, j, neuronIdx)));
            traceOcclPre = squeeze(occlDataPre(:, j, neuronIdx) - mean(occlDataPre(vecAxPreSp, j, neuronIdx)));
            
            % Collect for y-limits
            yValsNeuron = [yValsNeuron; traceFullPre(:); traceOcclPre(:)];
            
            % Plot Pre
            subplot(nPerPage, 9, (i-1)*9 + j);
            hold on;
            shadedErrorBar(vecAxPre, mean(traceFullPre, 2), std(traceFullPre, 0, 2)/sqrt(size(traceFullPre,2)), 'lineProps', 'k');
            shadedErrorBar(vecAxPre, mean(traceOcclPre, 2), std(traceOcclPre, 0, 2)/sqrt(size(traceOcclPre,2)), 'lineProps', 'r');
            box off; axis off;
            
            % Patch: 0–1s for Pre
            patchHandle = patch([0 1 1 0], [0 0 1 1], 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
            uistack(patchHandle, 'bottom');
            
            if j == 1
                line([0 0], [0 1], 'Color', 'k');
                line([0 1], [0 0], 'Color', 'k');
            elseif j == 2
                 title(sprintf('ROI %d. ', i+((pg-1)*nPerPage)));
                 % title(sprintf('selected %d. Mouse %s. match %d. Roi ID: %d. ', neuronIdx, mouseNamei, matchIDi, roiIDi(1)));
            end
        end
        
        for j = 1:nImgs
            % --- Task traces ---
            traceFullTask = squeeze(fullDataTask(:, j, neuronIdx) - mean(fullDataTask(vecAxTaskSp, j, neuronIdx)));
            traceOcclTask = squeeze(occlDataTask(:, j, neuronIdx) - mean(occlDataTask(vecAxTaskSp, j, neuronIdx)));
            
            % Collect for y-limits
            yValsNeuron = [yValsNeuron; traceFullTask(:); traceOcclTask(:)];
            
            % Plot Task
            subplot(nPerPage, 9, (i-1)*9 + 5 + j);
            hold on;
            shadedErrorBar(vecAxTask, mean(traceFullTask, 2), std(traceFullTask, 0, 2)/sqrt(size(traceFullTask,2)), 'lineProps', 'k');
            shadedErrorBar(vecAxTask, mean(traceOcclTask, 2), std(traceOcclTask, 0, 2)/sqrt(size(traceOcclTask,2)), 'lineProps', 'r');
            box off; axis off;
            
            % Patch: 0–2s for Task
            patchHandle = patch([0 2 2 0], [0 0 1 1], 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
            uistack(patchHandle, 'bottom');
            
            if j == 1
                line([0 0], [0 1], 'Color', 'k');
                line([0 1], [0 0], 'Color', 'k');
            elseif j == 2
%                 title(sprintf('Mouse %s. match %d. Roi ID: %d. ', mouseNamei, matchIDi, roiIDi(2)));
            end
        end

        % Apply common y-limits based on mean ± SEM
        yMean = [];
        ySEM = [];

        for j = 1:nImgs
            % Pre
            traceFullPre = squeeze(fullDataPre(:, j, neuronIdx) - mean(fullDataPre(vecAxPreSp, j, neuronIdx)));
            traceOcclPre = squeeze(occlDataPre(:, j, neuronIdx) - mean(occlDataPre(vecAxPreSp, j, neuronIdx)));
            yMean = [yMean; mean(traceFullPre, 2); mean(traceOcclPre, 2)];
            ySEM  = [ySEM;  std(traceFullPre, 0, 2)/sqrt(size(traceFullPre, 2)); ...
                std(traceOcclPre, 0, 2)/sqrt(size(traceOcclPre, 2))];
            
            % Task
            traceFullTask = squeeze(fullDataTask(:, j, neuronIdx) - mean(fullDataTask(vecAxTaskSp, j, neuronIdx)));
            traceOcclTask = squeeze(occlDataTask(:, j, neuronIdx) - mean(occlDataTask(vecAxTaskSp, j, neuronIdx)));
            yMean = [yMean; mean(traceFullTask, 2); mean(traceOcclTask, 2)];
            ySEM  = [ySEM;  std(traceFullTask, 0, 2)/sqrt(size(traceFullTask, 2)); ...
                std(traceOcclTask, 0, 2)/sqrt(size(traceOcclTask, 2))];
        end

        % Final limits from mean ± SEM
        yLower = yMean - ySEM;
        yUpper = yMean + ySEM;
        yLims = [min(yLower(:)), max(yUpper(:))];
        for col = [1:4, 6:9]
            subplot(nPerPage, 9, (i-1)*9 + col);
            ylim(yLims);
        end
    end

    sgtitle(sprintf('Neurons %d–%d', startIdx, endIdx));
    if pg<nPages
        disp('Click to continue...');
        % pause;
    end
end

%% % %======== Local helper functions (work inside a script) ========% % % 


function draw_colored_displacements(ax, preX, preY, taskX, taskY, colors, sz, plotTitle)
    % Draw one Pre -> Task displacement vector per neuron.
    % Shafts are intentionally thin; arrowheads are drawn as filled patches so
    % they appear thicker than the shaft.
    axes(ax); hold(ax, 'on');
    preX = preX(:); preY = preY(:); taskX = taskX(:); taskY = taskY(:);
    dx = taskX - preX;
    dy = taskY - preY;

    scatter(ax, preX, preY, sz * 0.35, colors, 'filled', 'MarkerFaceAlpha', 0.25);
    for ii = 1:numel(preX)
        draw_vector_arrow(ax, preX(ii), preY(ii), dx(ii), dy(ii), colors(ii,:), 0.35, 0.10, 0.07);
    end

    xlim(ax, [-1 3]); ylim(ax, [-1 3]);
    xticks(ax, -1:1:3); yticks(ax, -1:1:3);
    draw_diag45(ax);
    xlabel(ax, 'NO fam'); ylabel(ax, 'O fam');
    title(ax, plotTitle);
    axis(ax, 'square');
end


function draw_kmeans_mean_displacements(ax, preX, preY, taskX, taskY, colors, k, sz, plotTitle)
    % k-means is performed on the Task scatterplot coordinates [NO_task, O_task].
    % For each Task cluster, plot the average Pre -> Task displacement vector.
    axes(ax); hold(ax, 'on');
    preX = preX(:); preY = preY(:); taskX = taskX(:); taskY = taskY(:);

    valid = isfinite(preX) & isfinite(preY) & isfinite(taskX) & isfinite(taskY);
    preXv = preX(valid); preYv = preY(valid); taskXv = taskX(valid); taskYv = taskY(valid);
    colorsv = colors(valid, :);

    if numel(taskXv) < k
        text(ax, 0.5, 0.5, sprintf('Need at least %d valid points for k-means', k), ...
            'Units', 'normalized', 'HorizontalAlignment', 'center');
        xlim(ax, [-1 3]); ylim(ax, [-1 3]);
        title(ax, plotTitle); axis(ax, 'square');
        return
    end

    taskXY = [taskXv, taskYv];
    rng(1); % reproducible cluster assignment
    clusterIdx = kmeans(taskXY, k, 'Replicates', 50, 'MaxIter', 1000, 'Display', 'off');
    clusterColors = lines(k);

    % Show individual Task points with their original pre-response blend colors.
    scatter(ax, taskXv, taskYv, sz * 0.35, colorsv, 'filled', 'MarkerFaceAlpha', 0.20);

    for ci = 1:k
        ixC = clusterIdx == ci;
        if ~any(ixC)
            continue
        end

        % Cluster outline on the Task scatterplot.
        scatter(ax, taskXv(ixC), taskYv(ixC), sz * 0.65, ...
            'MarkerEdgeColor', clusterColors(ci,:), ...
            'MarkerFaceColor', 'none', ...
            'LineWidth', 0.8);

        % Average vector: starts at mean Pre location of neurons in this Task cluster,
        % ends at mean Task location of those same neurons.
        meanPreX  = mean(preXv(ixC),  'omitnan');
        meanPreY  = mean(preYv(ixC),  'omitnan');
        meanTaskX = mean(taskXv(ixC), 'omitnan');
        meanTaskY = mean(taskYv(ixC), 'omitnan');
        meanDx = meanTaskX - meanPreX;
        meanDy = meanTaskY - meanPreY;

        scatter(ax, meanPreX, meanPreY, sz * 1.0, clusterColors(ci,:), 'filled', ...
            'MarkerEdgeColor', 'k', 'LineWidth', 0.75);
        draw_vector_arrow(ax, meanPreX, meanPreY, meanDx, meanDy, clusterColors(ci,:), 1.1, 0.18, 0.13);
        text(ax, meanTaskX, meanTaskY, sprintf('  C%d, n=%d', ci, sum(ixC)), ...
            'Color', clusterColors(ci,:), 'FontWeight', 'bold', 'FontSize', 9, ...
            'Clipping', 'on');
    end

    xlim(ax, [-1 3]); ylim(ax, [-1 3]);
    xticks(ax, -1:1:3); yticks(ax, -1:1:3);
    draw_diag45(ax);
    xlabel(ax, 'NO fam'); ylabel(ax, 'O fam');
    title(ax, plotTitle);
    axis(ax, 'square');
end



function draw_displacement_angle_circle(ax, preX, preY, taskX, taskY, colors, sz, plotTitle)
    % Plot the angle of each Pre -> Task displacement vector on the unit circle.
    axes(ax); hold(ax, 'on');
    preX = preX(:); preY = preY(:); taskX = taskX(:); taskY = taskY(:);
    dx = taskX - preX;
    dy = taskY - preY;
    valid = isfinite(dx) & isfinite(dy) & hypot(dx, dy) > eps;
    theta = atan2(dy(valid), dx(valid));
    colorsv = colors(valid, :);

    th = linspace(-pi, pi, 361);
    plot(ax, cos(th), sin(th), 'k-', 'LineWidth', 0.5);
    plot(ax, [-1.1 1.1], [0 0], 'k:', 'LineWidth', 0.5);
    plot(ax, [0 0], [-1.1 1.1], 'k:', 'LineWidth', 0.5);

    scatter(ax, cos(theta), sin(theta), sz * 0.65, colorsv, 'filled', ...
        'MarkerFaceAlpha', 0.75, 'MarkerEdgeColor', 'none');

    xlim(ax, [-1.15 1.15]); ylim(ax, [-1.15 1.15]);
    axis(ax, 'square');
    xticks(ax, [-1 0 1]); yticks(ax, [-1 0 1]);
    xlabel(ax, 'cos(angle)'); ylabel(ax, 'sin(angle)');
    title(ax, plotTitle);
end


function [angleClusterIdxFull, angleClusterColors] = cluster_displacement_angles(preX, preY, taskX, taskY, k)
    % Circular clustering: k-means on [cos(theta), sin(theta)], not raw theta.
    preX = preX(:); preY = preY(:); taskX = taskX(:); taskY = taskY(:);
    dx = taskX - preX;
    dy = taskY - preY;
    valid = isfinite(preX) & isfinite(preY) & isfinite(taskX) & isfinite(taskY) & hypot(dx, dy) > eps;

    angleClusterIdxFull = nan(size(preX));
    angleClusterColors = lines(k);

    if sum(valid) < k
        return
    end

    theta = atan2(dy(valid), dx(valid));
    angleXY = [cos(theta), sin(theta)];
    rng(2); % reproducible angular cluster assignment
    angleClusterIdxFull(valid) = kmeans(angleXY, k, 'Replicates', 50, 'MaxIter', 1000, 'Display', 'off');
end


function draw_angle_cluster_mean_displacements(ax, preX, preY, taskX, taskY, k, sz, plotTitle)
    % Cluster neurons by displacement-vector angle and color Task points by those clusters.
    % Then plot the average Pre -> Task displacement vector for each angular cluster.
    axes(ax); hold(ax, 'on');
    preX = preX(:); preY = preY(:); taskX = taskX(:); taskY = taskY(:);
    dx = taskX - preX;
    dy = taskY - preY;

    [angleClusterIdx, clusterColors] = cluster_displacement_angles(preX, preY, taskX, taskY, k);
    valid = ~isnan(angleClusterIdx);

    if sum(valid) < k
        text(ax, 0.5, 0.5, sprintf('Need at least %d non-zero displacement vectors', k), ...
            'Units', 'normalized', 'HorizontalAlignment', 'center');
        xlim(ax, [-1 3]); ylim(ax, [-1 3]);
        title(ax, plotTitle); axis(ax, 'square');
        return
    end

    % Task scatter, colored by angular displacement cluster.
    for ci = 1:k
        ixC = angleClusterIdx == ci;
        scatter(ax, taskX(ixC), taskY(ixC), sz * 0.65, clusterColors(ci,:), 'filled', ...
            'MarkerFaceAlpha', 0.65, 'MarkerEdgeColor', 'k', 'LineWidth', 0.25);
    end

    % Mean displacement vector for each angular cluster.
    for ci = 1:k
        ixC = angleClusterIdx == ci;
        if ~any(ixC)
            continue
        end

        meanPreX = mean(preX(ixC), 'omitnan');
        meanPreY = mean(preY(ixC), 'omitnan');
        meanDx = mean(dx(ixC), 'omitnan');
        meanDy = mean(dy(ixC), 'omitnan');
        meanTaskX = meanPreX + meanDx;
        meanTaskY = meanPreY + meanDy;

        scatter(ax, meanPreX, meanPreY, sz * 1.05, clusterColors(ci,:), 'filled', ...
            'MarkerEdgeColor', 'k', 'LineWidth', 0.8);
        draw_vector_arrow(ax, meanPreX, meanPreY, meanDx, meanDy, clusterColors(ci,:), 1.1, 0.18, 0.13);
        text(ax, meanTaskX, meanTaskY, sprintf('  A%d, n=%d', ci, sum(ixC)), ...
            'Color', clusterColors(ci,:), 'FontWeight', 'bold', 'FontSize', 9, ...
            'Clipping', 'on');
    end

    xlim(ax, [-1 3]); ylim(ax, [-1 3]);
    xticks(ax, -1:1:3); yticks(ax, -1:1:3);
    draw_diag45(ax);
    xlabel(ax, 'NO fam task'); ylabel(ax, 'O fam task');
    title(ax, plotTitle);
    axis(ax, 'square');
end

function draw_vector_arrow(ax, x0, y0, dx, dy, color, shaftLW, headLenFrac, headWidth)
    % Draw an arrow in data coordinates with separate control over shaft and head.
    % shaftLW controls the thin line. headLenFrac/headWidth control the larger head.
    if ~all(isfinite([x0, y0, dx, dy]))
        return
    end

    L = hypot(dx, dy);
    if L < eps
        return
    end

    ux = dx / L;
    uy = dy / L;
    px = -uy;
    py = ux;

    % Keep heads visible without making short arrows ridiculous.
    headLen = min(max(headLenFrac * L, 0.035), 0.18);
    headWid = min(max(headWidth, 0.035), 0.16);

    xTip = x0 + dx;
    yTip = y0 + dy;
    xBase = xTip - headLen * ux;
    yBase = yTip - headLen * uy;

    % Thin shaft stops at the base of the head.
    plot(ax, [x0 xBase], [y0 yBase], '-', ...
        'Color', color, 'LineWidth', shaftLW, 'Clipping', 'on');

    % Thick filled head.
    patch(ax, [xTip, xBase + headWid * px, xBase - headWid * px], ...
              [yTip, yBase + headWid * py, yBase - headWid * py], ...
              color, 'EdgeColor', color, 'LineWidth', 0.5, 'Clipping', 'on');
end
function draw_leader_labels(ax, x, y, nums, offsetPct)
    % Draw thin leader lines from (x,y) to small offset labels showing "nums".
    % ax        : target axes handle
    % x, y      : column vectors of point coordinates
    % nums      : selection numbers (1..K) to display
    % offsetPct : optional, fraction of axis range for label offset (default 0.06)
    if nargin < 5 || isempty(offsetPct), offsetPct = 0.06; end
    axes(ax); hold(ax,'on');
    x = x(:); y = y(:); nums = nums(:);
    xl = xlim(ax); yl = ylim(ax);
    cx = mean(xl); cy = mean(yl);
    dx = offsetPct * diff(xl);
    dy = offsetPct * diff(yl);
    for k = 1:numel(nums)
        x0 = x(k); y0 = y(k);
        sx = sign(x0 - cx); 
        if sx == 0
            sx = 1; 
        end
        sy = sign(y0 - cy); 
        if sy == 0
            sy = 1; 
        end
        x1 = x0 + sx*dx; y1 = y0 + sy*dy;   % label anchor
        % clamp to axes limits
        x1 = min(max(x1, xl(1)), xl(2));
        y1 = min(max(y1, yl(1)), yl(2));
        % thin leader line
        plot(ax, [x0 x1], [y0 y1], '-', 'Color', [0.25 0.25 0.25], 'LineWidth', 0.5, 'Clipping','on');
        % text label (selection number)
        text(ax, x1, y1, sprintf('%d', nums(k)), ...
            'FontSize', 9, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'Color', [0.15 0.15 0.15], 'BackgroundColor', 'none', 'Clipping','on');
    end
end

function draw_diag45(ax)
    % Draw a y=x dashed line within current axis limits (more robust than refline in scripts)
    axes(ax); hold(ax,'on');
    xl = xlim(ax); yl = ylim(ax);
    mn = max(xl(1), yl(1));
    mx = min(xl(2), yl(2));
    plot(ax, [mn mx], [mn mx], 'k--', 'LineWidth', 0.5, 'Clipping','on');
end