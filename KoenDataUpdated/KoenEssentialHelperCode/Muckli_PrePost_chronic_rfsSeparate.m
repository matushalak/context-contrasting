%% LDA analysis on population data from muckli experiments pre vs post training
%
%	Version History:
%	2022-02-14	Created by Koen Seignette
%   2022-07-22 update on some significance calculations and average traces

%#ok<*UNRCH> 
%#ok<*SAGROW> 

clear all

% rmpath('D:\GitHub\2Pimage\imageVolumeToolbox')
% rmpath('D:\GitHub\InVivoTools-master\General')

filenamesPre = {};
filepathsPre = {};
selecting = true; % true as long as files are being selected
i = 0;
while selecting
    i = i + 1;

    str = sprintf('load pretraining Res file %d. Press cancel when done', i);
    [filenamesPre{i}, filepathsPre{i}] = uigetfile('*Res.mat', str);

    if filenamesPre{i} == 0 % Cancel is pressed probably: stop with selecting
        filenamesPre(i) = [];
        filepathsPre(i) = [];
        selecting = false;
    end
end
filenamesPre = filenamesPre';
filepathsPre = filepathsPre';
% nfiles = length(filenamesPre); % The number of files that have been selected

filenamesPost = {};
filepathsPost = {};
selecting = true; % true as long as files are being selected
i = 0;
while selecting
    i = i + 1;

    str = sprintf('load posttraining Res file %d. Press cancel when done', i);
    [filenamesPost{i}, filepathsPost{i}] = uigetfile('*Res.mat', str);

    if filenamesPost{i} == 0 % Cancel is pressed probably: stop with selecting
        filenamesPost(i) = [];
        filepathsPost(i) = [];
        selecting = false;
    end
end
filenamesPost = filenamesPost';
filepathsPost = filepathsPost';

filenamesChronic = {};
filepathsChronic = {};
selecting = true; % true as long as files are being selected
i = 0;
while selecting
    i = i + 1;

    str = sprintf('load chronic mat file %d. Press cancel when done', i);
    [filenamesChronic{i}, filepathsChronic{i}] = uigetfile('*_chronic.mat', str);

    if filenamesChronic{i} == 0 % Cancel is pressed probably: stop with selecting
        filenamesChronic(i) = [];
        filepathsChronic(i) = [];
        selecting = false;
    end
end
filenamesChronic = filenamesChronic';
filepathsChronic = filepathsChronic';

nfiles = length(filenamesPost); % The number of files that have been selected


%% load data
clearvars -except nfiles filepathsPre filenamesPre filepathsPost filenamesPost filepathsChronic filenamesChronic

% Load the main files
fprintf('\nloading in %d files:\n', nfiles)
for i = 1:nfiles % Backwards to create final size on first loop
    fprintf('\nloading files for mouse %d...',i)
    pnPre = filepathsPre{i};
    fnPre = filenamesPre{i};
    load([pnPre fnPre]);
    datastructPre(i).info = info;
    datastructPre(i).Res = Res;
    datastructPre(i).log = info.Stim.Parameters.StimSmp;

    pnPost = filepathsPost{i};
    fnPost = filenamesPost{i};
    load([pnPost fnPost]);
    datastructPost(i).info = info;
    datastructPost(i).Res = Res;
    datastructPost(i).log = info.Stim.Parameters.StimSmp;

    pnChronic = filepathsChronic{i};
    fnChronic = filenamesChronic{i};
    load([pnChronic fnChronic], 'linkMat');
    datastructChronic(i).linkMat = linkMat;

    fprintf('\nsuccesfully loaded files for mouse %d\n',i)
end
fprintf('\nsuccesfully loaded all files\n')


%% or just load files
clear all

% load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\datastructPrePostGrayL5_rfsSeparate_chronic.mat') % L5
% load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\PrePostGrayL23Chronic.mat') % L23 % OUTDATED RES FILES 


load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\PrePostGrayL23ChronicSeparateNewRFs.mat') % L23
% load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\PrePostGrayL5ChronicSeparateNewRFs.mat') % L5

%% Initialize and organise data
clearvars -except datastructPre datastructPost filenamesPre...
    filenamesPost filepathsPre filepathsPost nfiles doDecoding...
    imgNrs nImgs performanceChance nTrials nReps nBoots trainFrac...
    filepathsChronic filenamesChronic datastructChronic

% still figure out how to deal with neurons that have a good RF in either
% pre or post but not both.

% imgNrs = [1 2 4 5]; % image nrs to decode, trained images
imgNrs = [1 2 3 4 5 6]; % image nrs to decode, all images
% imgNrs = [3 6]; % image nrs to decode, untrained images
% imgNrs = [4 5]; % image nrs to decode
nImgs = length(imgNrs);
performanceChance = 100/nImgs;
nTrials = 20; % nr of trials shown per image
trainFrac = 0.5; % on what fraction would you like to train the decoder (0.8 is good)
rfDistVec = 2; % Minimum distance away from occluder edge
vecAx = datastructPre(1).Res.ax;
vecAxSp = vecAx<0; % spontaneous activity window
vecAxSt = vecAx>0.2 & vecAx<1; % stim window
alphaVal = 0.99; % significance value for cells to be included
% zscoreVal = 0; % minimum zscore value for cells to be included
if nfiles == 6
    rsqThresh = 0.33; % 0.33 for L2/3, 0.15 for L5
else
    rsqThresh = 0.15; % 0.33 for L2/3, 0.15 for L5
end
% bratThresh = 1.5;
snrThresh = 4; % snr threshold for RF
useSpikingData = 0; % deconvolved (1) or df/f (0)
regressRun = false; % regress out running? Only for CaSigCorrected, not for spikes
doZscore = true; % in case you want to work with zscored data instead of dff


% for loop with decoding etc.
for i = 1:nfiles
    %[2 3 5] % slower after training
    %[1 4 6] % faster after training

    %%%%%%%%%% PRE DATASET %%%%%%%%%%
    % calculation of RF distances to occluder and inclusion criteria
    info = datastructPre(i).info;
    linkMat = datastructChronic(i).linkMat; % chronically matched neurons
    clear linkMatIncl
    linkMatIncl(:,1) = linkMat(:,1); % first column is pre training
    linkMatIncl(:,2) = linkMat(:,3); % third column is post training (L2/3)
    linkMatIncl = linkMatIncl(all(linkMatIncl,2),:); % remove rows with at least one zero

    nRois = length(info.rois);
    rfOnDist = zeros(nRois,1);
    rfOffDist = zeros(nRois,1);
    rfOnGlmIncl = zeros(nRois,1);
    rfOffGlmIncl = zeros(nRois,1);
    for n = 1:nRois
        fwhmOn = info.rois(n).onFWHM;
        fwhmOff = info.rois(n).offFWHM;
        azi = info.rois(n).azi;
        ele = info.rois(n).ele;
        rfsz = info.rois(n).rfsz;
%         brat = info.rois(n).BRAT;
        rsq = info.rois(n).RSQ;
        snr = info.rois(n).SNR;

        aziOnDist = -(azi(1)+fwhmOn/2); % because azi values are negative on left side of screen
        aziOffDist = -(azi(2)+fwhmOff/2); % because azi values are negative on left side of screen
        eleOnDist = ele(1)-fwhmOn/2;
        eleOffDist = ele(2)-fwhmOff/2;

        rfOnDist(n) = min(aziOnDist, eleOnDist);
        rfOffDist(n) = min(aziOffDist, eleOffDist);
        if rsq(1)>rsqThresh && snr(1)>snrThresh
            rfOnGlmIncl(n) = 1;
        end
        if rsq(2)>rsqThresh && snr(2)>snrThresh
            rfOffGlmIncl(n) = 1;
        end
    end
    % inclusion of neurons based on RF properties
    onCrit = rfOnDist>rfDistVec & logical(rfOnGlmIncl);
    offCrit = rfOffDist>rfDistVec & logical(rfOffGlmIncl);
    rfInclPre = onCrit | offCrit; % either a good ON or OFF receptive field
   
    
    %%%%%%%%%% POST DATASET %%%%%%%%%%

    % calculation of RF distances to occluder and inclusion criteria
    info = datastructPost(i).info;
    nRois = length(info.rois);
    rfOnDist = zeros(nRois,1);
    rfOffDist = zeros(nRois,1);
    rfOnGlmIncl = zeros(nRois,1);
    rfOffGlmIncl = zeros(nRois,1);
    for n = 1:nRois
        fwhmOn = info.rois(n).onFWHM;
        fwhmOff = info.rois(n).offFWHM;
        azi = info.rois(n).azi;
        ele = info.rois(n).ele;
        rfsz = info.rois(n).rfsz;
%         brat = info.rois(n).BRAT;
        rsq = info.rois(n).RSQ;
        snr = info.rois(n).SNR;

        aziOnDist = -(azi(1)+fwhmOn/2); % because azi values are negative on left side of screen
        aziOffDist = -(azi(2)+fwhmOff/2); % because azi values are negative on left side of screen
        eleOnDist = ele(1)-fwhmOn/2;
        eleOffDist = ele(2)-fwhmOff/2;

        rfOnDist(n) = min(aziOnDist, eleOnDist);
        rfOffDist(n) = min(aziOffDist, eleOffDist);
        if rsq(1)>rsqThresh && snr(1)>snrThresh
            rfOnGlmIncl(n) = 1;
        end
        if rsq(2)>rsqThresh && snr(2)>snrThresh
            rfOffGlmIncl(n) = 1;
        end
    end
    % inclusion of neurons based on RF properties
    onCrit = rfOnDist>rfDistVec & logical(rfOnGlmIncl);
    offCrit = rfOffDist>rfDistVec & logical(rfOffGlmIncl);
    rfInclPost = onCrit | offCrit; % either a good ON or OFF receptive field
%     rfInclPost = true(size(rfInclPost)); % in case you want to just select based on pre RFs

%     rfsPre = find(rfIncl)
%     linkIncl = false(length(rfInclPost),1);
%     linkIncl(linkMatIncl(:,2)) = true; % post is second column of linkMatIncl

%     [val,pos] = intersect(find(rfIncl),linkMatIncl(:,1));
   
%     linkInclPre = false(length(rfInclPre),1);
%     linkInclPre(linkMatIncl(:,1)) = true; % pre is first column of linkMatIncl


%%%%%%% checking this section
% note that you can base the selection of the neurons on the pre training
% dataset, on the post training dataset, or on both (such that neurons
% should have good RFs in both datasets, rather than in either pre or post)

% check which neurons remain if selecting only for pre
[~,pos] = intersect(linkMatIncl(:,1), find(rfInclPre));
linkMatInclAfterPre = linkMatIncl(pos,:);

% check which neurons remain if selecting only for post
[~,pos] = intersect(linkMatIncl(:,2), find(rfInclPost));
linkMatInclAfterPost = linkMatIncl(pos,:);

% check which neurons remain if selecting for both
[~,pos] = intersect(linkMatInclAfterPre(:,2), find(rfInclPost));
linkMatInclAfterBoth = linkMatInclAfterPre(pos,:);
 
linkMatUsed = linkMatInclAfterPre;
% linkMatUsed = linkMatInclAfterPost;

%%%%%%% checking section above
        
    
    % responses sorted to match across mice
    Res = datastructPre(i).Res;

    if regressRun
        tempTrace = Res.CaSigCorrected; 
        % regress out running speed per trial
        clear r
        x = Res.speed; % run speed for this session
        for g = 1:size(tempTrace,3)
            y = squeeze(tempTrace(:,:,g)); % get trace for ROI
            lme = fitlm(x(:),y(:)); % model fit
            r(:,g) = lme.Residuals.Raw; % get residuals for this ROI
        end
        Res.CaSigCorrected = reshape(r, size(tempTrace)); % back to original matrix
    end

    matTrialTypes = datastructPre(i).log; % trialtypes
    [~, dataSortidx] = sortrows(matTrialTypes', [1 2]);
    matTrialTypesSort = matTrialTypes(:,dataSortidx);
    % create index to select only images in 'imgNrs' to decode on
    imgIdx = false(1,size(matTrialTypesSort,2));
    for n = 1:nImgs
        imgIdx(matTrialTypesSort(1,:)==imgNrs(n))=1;
    end

%     if useSpikingData
%         CaResSort = Res.CaDeconCorrected(:,dataSortidx,rfIncl&linkIncl); % reordering
%     else
%         if regressRun
%             CaResSort = Res.CaSigCorrected(:,dataSortidx,rfIncl&linkIncl); % reordering, regressRun does subtract 1 already
%         else
%             CaResSort = Res.CaSigCorrected(:,dataSortidx,rfIncl&linkIncl)-1; % reordering and subtract 1
%         end
%     end

if useSpikingData
%     CaResSort = Res.CaDeconCorrected(:,dataSortidx,linkMatInclAfterPre(:,1)); % reordering
    %         CaResSort = Res.CaDeconCorrected(:,dataSortidx,linkMatInclAfterPost(:,1)); % reordering
            CaResSort = Res.CaDeconCorrected(:,dataSortidx,linkMatUsed(:,1)); % reordering
else
    if regressRun
%         CaResSort = Res.CaSigCorrected(:,dataSortidx,linkMatInclAfterPre(:,1)); % reordering, regressRun does subtract 1 already
        %             CaResSort = Res.CaSigCorrected(:,dataSortidx,linkMatInclAfterPost(:,1)); % reordering, regressRun does subtract 1 already
                      CaResSort = Res.CaSigCorrected(:,dataSortidx,linkMatUsed(:,1));
    else
%         CaResSort = Res.CaSigCorrected(:,dataSortidx,linkMatInclAfterPre(:,1))-1; % reordering and subtract 1
        %             CaResSort = Res.CaSigCorrected(:,dataSortidx,linkMatInclAfterPost(:,1))-1; % reordering and subtract 1
                CaResSort = Res.CaSigCorrected(:,dataSortidx,linkMatUsed(:,1))-1; %
    end
    if doZscore
        caPop = [];
        for g = 1:size(CaResSort,3)
            ca = CaResSort(:,:,g);
            sz = size(ca);
            ca = zscore(ca(:));
            ca = reshape(ca, sz);
            caPop = cat(3, caPop,ca);
        end
    end
end

    if doZscore
        if exist('caPop', 'var') && ~isempty(caPop)
            CaResSort = caPop(:,imgIdx,:); % subselect images
        end
    else
        CaResSort = CaResSort(:,imgIdx,:); % subselect images
    end

    matTrialTypesIncl = matTrialTypesSort(:,imgIdx); % subselect images
    runSpeed = repmat(Res.speed(:,dataSortidx),1,1,size(CaResSort,3));
    runSpeed = runSpeed(:,imgIdx,:);

    % trace matrices (frames x imgs x trials x rois)
    imgFullRes = zeros(size(CaResSort,1), nImgs, nTrials, size(CaResSort,3)); % pre-allocate
    imgOcclRes = zeros(size(CaResSort,1), nImgs, nTrials, size(CaResSort,3)); % pre-allocate
    for j = 1:nImgs
        imgIdxFull = find(matTrialTypesIncl(1,:)==imgNrs(j) & matTrialTypesIncl(2,:)==0);
        imgIdxOccl = find(matTrialTypesIncl(1,:)==imgNrs(j) & matTrialTypesIncl(2,:)==1);
        imgFullRes(:,j,:,:) = CaResSort(:,imgIdxFull,:);
        imgOcclRes(:,j,:,:) = CaResSort(:,imgIdxOccl,:);
    end

    % data for decoding
    matData = squeeze(mean(CaResSort(vecAxSt,:,:)))-squeeze(mean(CaResSort(vecAxSp,:,:)));

    if size(matData,1)==1
        matData = matData'; % with only 1 ROI the dimensions get messed up
    end

    % calculate significance of responses
    hValFull = zeros(nImgs, size(imgFullRes,4));
    hValOccl = zeros(nImgs, size(imgOcclRes,4));

    for k = 1:size(imgFullRes,4)
        for j = 1:nImgs
            [~, hValFull(j,k)] = calcSign(squeeze(imgFullRes(:,j,:,k)), vecAxSp, vecAxSt, alphaVal);
            [~, hValOccl(j,k)] = calcSign(squeeze(imgOcclRes(:,j,:,k)), vecAxSp, vecAxSt, alphaVal);
        end
    end
    hValFull(isnan(hValFull))=0;
    hValOccl(isnan(hValOccl))=0;

    clear fullSign occlSign
    hValCrit = hValFull|hValOccl;

    imgFullResMn = squeeze(mean(imgFullRes,3));
    imgOcclResMn = squeeze(mean(imgOcclRes,3));
    
    n = 0;
    for k = 1:size(imgFullResMn,3)
        if sum(hValCrit(:,k))>0
            n = n+1;
%             fullSign(:,n) = squeeze(mean(imgFullResMn(:,hValCrit(:,k),k),2)); % take only significant images
%             occlSign(:,n) = squeeze(mean(imgOcclResMn(:,hValCrit(:,k),k),2)); % take only significant images
            fullSign(:,n) = squeeze(mean(imgFullResMn(:,:,k),2)); % take all images, not just significant ones
            occlSign(:,n) = squeeze(mean(imgOcclResMn(:,:,k),2)); % take all images, not just significant ones
        end
    end

    % recalculate with only significant neurons
    imgFullResMn = squeeze(mean(imgFullRes(:,:,:,sum(hValCrit)>0),3));
    imgOcclResMn = squeeze(mean(imgOcclRes(:,:,:,sum(hValCrit)>0),3));

    % zscore trace per ROI per image
%     imgFullResMnZs = (imgFullResMn-mean(imgFullResMn(vecAxSp,:,:)))./std(imgFullResMn(vecAxSp,:,:),0,1);
%     imgOcclResMnZs = (imgOcclResMn-mean(imgOcclResMn(vecAxSp,:,:)))./std(imgOcclResMn(vecAxSp,:,:),0,1);
    
%     mn = squeeze(min(min(cat(2,imgFullResMnZs,imgOcclResMnZs)))); % max zscore val
%     mx = squeeze(max(max(cat(2,imgFullResMnZs,imgOcclResMnZs)))); % min zscore val

%     idx = mn<-zscoreVal | mx>zscoreVal;
% 
%     % include only neurons with decent zscores
%     imgFullResMn = imgFullResMn(:,:,idx);
%     imgOcclResMn = imgOcclResMn(:,:,idx);
%     fullSign = fullSign(:,idx);
%     occlSign = occlSign(:,idx);
    
    % response strength per ROI
    if exist('fullSign', 'var')
        scatFull = squeeze(mean(fullSign(vecAxSt,:)))-squeeze(mean(fullSign(vecAxSp,:)));
        scatOccl = squeeze(mean(occlSign(vecAxSt,:)))-squeeze(mean(occlSign(vecAxSp,:)));
    else
        scatFull = [];
        scatOccl = [];
        fullSign = [];
        occlSign = [];
    end
%     scatFull = squeeze(mean(fullSign(vecAxSt,:)));
%     scatOccl = squeeze(mean(occlSign(vecAxSt,:)));

    datastructPre(i).rfIncl = rfInclPre;
    datastructPre(i).CaResSort = CaResSort;
    datastructPre(i).runSpeed = runSpeed;
    datastructPre(i).rfOnGlmIncl = rfOnGlmIncl;
    datastructPre(i).rfOffGlmIncl = rfOffGlmIncl;
    datastructPre(i).rfOnDist = rfOnDist;
    datastructPre(i).rfOffDist = rfOffDist;
%     datastructPre(i).rfGlmIncl = rfGlmIncl;
%     datastructPre(i).rfDist = rfDist;
    datastructPre(i).matTrialTypesIncl = matTrialTypesIncl;
    datastructPre(i).matTrialTypes = matTrialTypes;
    datastructPre(i).hValFull = hValFull;
    datastructPre(i).hValOccl = hValOccl;
    datastructPre(i).fullSign = fullSign;
    datastructPre(i).occlSign = occlSign;
    datastructPre(i).imgFullRes = imgFullRes;
    datastructPre(i).imgOcclRes = imgOcclRes;
    datastructPre(i).imgFullResMn = imgFullResMn;
    datastructPre(i).imgOcclResMn = imgOcclResMn;
    datastructPre(i).matData = matData;
    datastructPre(i).scatFull = scatFull;
    datastructPre(i).scatOccl = scatOccl;
%     datastructPre(i).azi = azi;
%     datastructPre(i).ele = ele;
%     datastructPre(i).rfsz = rfsz;
    


    % responses sorted to match across mice
    Res = datastructPost(i).Res;

    if regressRun
        tempTrace = Res.CaSigCorrected;
        % regress out running speed per trial
        clear r
        x = Res.speed; % run speed for this session
        for g = 1:size(tempTrace,3)
            y = squeeze(tempTrace(:,:,g)); % get trace for ROI
            lme = fitlm(x(:),y(:)); % model fit
            r(:,g) = lme.Residuals.Raw; % get residuals for this ROI
        end
        Res.CaSigCorrected = reshape(r, size(tempTrace)); % back to original matrix
    end

    matTrialTypes = datastructPost(i).log; % trialtypes
    [~, dataSortidx] = sortrows(matTrialTypes', [1 2]);
    matTrialTypesSort = matTrialTypes(:,dataSortidx);
    % create index to select only images in 'imgNrs' to decode on
    imgIdx = false(1,size(matTrialTypesSort,2));
    for n = 1:nImgs
        imgIdx(matTrialTypesSort(1,:)==imgNrs(n))=1;
    end

%     if useSpikingData
%         CaResSort = Res.CaDeconCorrected(:,dataSortidx,rfIncl&linkIncl); % reordering
%     else
%         if regressRun
%             CaResSort = Res.CaSigCorrected(:,dataSortidx,rfIncl&linkIncl); % reordering, regressRun does subtract 1 already
%         else
%             CaResSort = Res.CaSigCorrected(:,dataSortidx,rfIncl&linkIncl)-1; % reordering and subtract 1
%         end
%     end
if useSpikingData
%     CaResSort = Res.CaDeconCorrected(:,dataSortidx,linkMatInclAfterPre(:,2)); % reordering
    %         CaResSort = Res.CaDeconCorrected(:,dataSortidx,linkMatInclAfterPost(:,2)); % reordering
            CaResSort = Res.CaDeconCorrected(:,dataSortidx,linkMatUsed(:,2)); % reordering
else
    if regressRun
%         CaResSort = Res.CaSigCorrected(:,dataSortidx,linkMatInclAfterPre(:,2)); % reordering, regressRun does subtract 1 already
        %             CaResSort = Res.CaSigCorrected(:,dataSortidx,linkMatInclAfterPost(:,2)); % reordering, regressRun does subtract 1 already
                      CaResSort = Res.CaSigCorrected(:,dataSortidx,linkMatUsed(:,2));
    else
%         CaResSort = Res.CaSigCorrected(:,dataSortidx,linkMatInclAfterPre(:,2))-1; % reordering and subtract 1
        %             CaResSort = Res.CaSigCorrected(:,dataSortidx,linkMatInclAfterPost(:,2))-1; % reordering and subtract 1
                CaResSort = Res.CaSigCorrected(:,dataSortidx,linkMatUsed(:,2))-1; %
    end

    if doZscore
        caPop = [];
        for g = 1:size(CaResSort,3)
            ca = CaResSort(:,:,g);
            sz = size(ca);
            ca = zscore(ca(:));
            ca = reshape(ca, sz);
            caPop = cat(3, caPop,ca);
        end
    end
end

    if doZscore
        if exist('caPop', 'var') && ~isempty(caPop)
            CaResSort = caPop(:,imgIdx,:); % subselect images
        end
    else
        CaResSort = CaResSort(:,imgIdx,:); % subselect images
    end
    matTrialTypesIncl = matTrialTypesSort(:,imgIdx); % subselect images
    runSpeed = repmat(Res.speed(:,dataSortidx),1,1,size(CaResSort,3));
    runSpeed = runSpeed(:,imgIdx,:);

    % trace matrices (frames x imgs x trials x rois)
    imgFullRes = zeros(size(CaResSort,1), nImgs, nTrials, size(CaResSort,3)); % pre-allocate
    imgOcclRes = zeros(size(CaResSort,1), nImgs, nTrials, size(CaResSort,3)); % pre-allocate
    for j = 1:nImgs
        imgIdxFull = find(matTrialTypesIncl(1,:)==imgNrs(j) & matTrialTypesIncl(2,:)==0);
        imgIdxOccl = find(matTrialTypesIncl(1,:)==imgNrs(j) & matTrialTypesIncl(2,:)==1);
        imgFullRes(:,j,:,:) = CaResSort(:,imgIdxFull,:);
        imgOcclRes(:,j,:,:) = CaResSort(:,imgIdxOccl,:);
    end

    % data for decoding
    matData = squeeze(mean(CaResSort(vecAxSt,:,:)))-squeeze(mean(CaResSort(vecAxSp,:,:)));

    if size(matData,1)==1
        matData = matData'; % with only 1 ROI the dimensions get messed up
    end

    % calculate significance of responses
    hValFull = zeros(nImgs, size(imgFullRes,4));
    hValOccl = zeros(nImgs, size(imgOcclRes,4));

    for k = 1:size(imgFullRes,4)
        for j = 1:nImgs
            [~, hValFull(j,k)] = calcSign(squeeze(imgFullRes(:,j,:,k)), vecAxSp, vecAxSt, alphaVal);
            [~, hValOccl(j,k)] = calcSign(squeeze(imgOcclRes(:,j,:,k)), vecAxSp, vecAxSt, alphaVal);
        end
    end

    hValFull(isnan(hValFull))=0;
    hValOccl(isnan(hValOccl))=0;

    clear fullSign occlSign
    hValCrit = hValFull|hValOccl;

    imgFullResMn = squeeze(mean(imgFullRes,3));
    imgOcclResMn = squeeze(mean(imgOcclRes,3));

    n = 0;
    for k = 1:size(imgFullResMn,3)
        if sum(hValCrit(:,k))>0
            n = n+1;
%             fullSign(:,n) = squeeze(mean(imgFullResMn(:,hValCrit(:,k),k),2)); % take only significant images
%             occlSign(:,n) = squeeze(mean(imgOcclResMn(:,hValCrit(:,k),k),2)); % take only significant images
            fullSign(:,n) = squeeze(mean(imgFullResMn(:,:,k),2)); % take all images, not just significant ones
            occlSign(:,n) = squeeze(mean(imgOcclResMn(:,:,k),2)); % take all images, not just significant ones
        end
    end
    
    % recalculate with only significant neurons
    imgFullResMn = squeeze(mean(imgFullRes(:,:,:,sum(hValCrit)>0),3));
    imgOcclResMn = squeeze(mean(imgOcclRes(:,:,:,sum(hValCrit)>0),3));

    % zscore trace per ROI per image
    imgFullResMnZs = (imgFullResMn-mean(imgFullResMn(vecAxSp,:,:)))./std(imgFullResMn(vecAxSp,:,:),0,1);
    imgOcclResMnZs = (imgOcclResMn-mean(imgOcclResMn(vecAxSp,:,:)))./std(imgOcclResMn(vecAxSp,:,:),0,1);
    
    mn = squeeze(min(min(cat(2,imgFullResMnZs,imgOcclResMnZs)))); % max zscore val
    mx = squeeze(max(max(cat(2,imgFullResMnZs,imgOcclResMnZs)))); % min zscore val

%     idx = mn<-zscoreVal | mx>zscoreVal;
% 
%     % include only neurons with decent zscores
%     imgFullResMn = imgFullResMn(:,:,idx);
%     imgOcclResMn = imgOcclResMn(:,:,idx);
%     fullSign = fullSign(:,idx);
%     occlSign = occlSign(:,idx);
    

    % response strength per ROI
    if exist('fullSign', 'var')
        scatFull = squeeze(mean(fullSign(vecAxSt,:)))-squeeze(mean(fullSign(vecAxSp,:)));
        scatOccl = squeeze(mean(occlSign(vecAxSt,:)))-squeeze(mean(occlSign(vecAxSp,:)));
    else
        scatFull = [];
        scatOccl = [];
        fullSign = [];
        occlSign = [];
    end
%     scatFull = squeeze(mean(fullSign(vecAxSt,:)));
%     scatOccl = squeeze(mean(occlSign(vecAxSt,:)));

    datastructPost(i).rfIncl = rfInclPost;
    datastructPost(i).CaResSort = CaResSort;
    datastructPost(i).runSpeed = runSpeed;
    datastructPost(i).rfOnGlmIncl = rfOnGlmIncl;
    datastructPost(i).rfOffGlmIncl = rfOffGlmIncl;
    datastructPost(i).rfOnDist = rfOnDist;
    datastructPost(i).rfOffDist = rfOffDist;
%     datastructPost(i).rfGlmIncl = rfGlmIncl;
%     datastructPost(i).rfDist = rfDist;
    datastructPost(i).matTrialTypesIncl = matTrialTypesIncl;
    datastructPost(i).matTrialTypes = matTrialTypes;
    datastructPost(i).hValFull = hValFull;
    datastructPost(i).hValOccl = hValOccl;
    datastructPost(i).fullSign = fullSign;
    datastructPost(i).occlSign = occlSign;
    datastructPost(i).imgFullRes = imgFullRes;
    datastructPost(i).imgOcclRes = imgOcclRes;
    datastructPost(i).imgFullResMn = imgFullResMn;
    datastructPost(i).imgOcclResMn = imgOcclResMn;
    datastructPost(i).matData = matData;
    datastructPost(i).scatFull = scatFull;
    datastructPost(i).scatOccl = scatOccl;
%     datastructPost(i).azi = azi;
%     datastructPost(i).ele = ele;
%     datastructPost(i).rfsz = rfsz;

    disp(i)
end




%% in case you calculated responses to 6 images, plot pre vs post (separate by image type)
% color pallets for plotting
col1 = [0,0,0]; % black
col2 = [131, 197, 190]/255; % blue/greenish

save_fig = false;

famIdx = [1 2 4 5];
% famIdx = [1 2];
novIdx = [3 6];

imgFullResMnPopPre = datastructPre(1).imgFullResMn;
imgOcclResMnPopPre = datastructPre(1).imgOcclResMn;
imgFullResMnPopPost = datastructPost(1).imgFullResMn;
imgOcclResMnPopPost = datastructPost(1).imgOcclResMn;

for i = 2:nfiles
    imgFullResMnPopPre = cat(3, imgFullResMnPopPre, datastructPre(i).imgFullResMn);
    imgOcclResMnPopPre = cat(3, imgOcclResMnPopPre, datastructPre(i).imgOcclResMn);
    imgFullResMnPopPost = cat(3, imgFullResMnPopPost, datastructPost(i).imgFullResMn);
    imgOcclResMnPopPost = cat(3, imgOcclResMnPopPost, datastructPost(i).imgOcclResMn);
end

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
xlabel('Time (s)'), ylabel('dF/F (%)'), xticks(-1:1:3), xlim([-1 3]), title('Pre Fam'), % figClean
t(2) = subplot(3,5,2);
shadedErrorBar(vecAx,mean(imgFullResFamPostBsl,2)...
    ,std(imgFullResFamPostBsl,0,2)/sqrt(size(imgFullResFamPostBsl,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAx,mean(imgOcclResFamPostBsl,2)...
    ,std(imgOcclResFamPostBsl,0,2)/sqrt(size(imgOcclResFamPostBsl,2)), 'lineProps', 'r');
xlabel('Time (s)'), title('Post Fam'), % figClean
t(3) = subplot(3,5,3);
shadedErrorBar(vecAx,mean(imgFullResNovPreBsl,2)...
    ,std(imgFullResNovPreBsl,0,2)/sqrt(size(imgFullResNovPreBsl,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAx,mean(imgOcclResNovPreBsl,2)...
    ,std(imgOcclResNovPreBsl,0,2)/sqrt(size(imgOcclResNovPreBsl,2)), 'lineProps', 'r');
xlabel('Time (s)'), ylabel('dF/F (%)'), xticks(-1:1:3), xlim([-1 3]), title('Pre Nov'), % figClean
t(4) = subplot(3,5,4);
shadedErrorBar(vecAx,mean(imgFullResNovPostBsl,2)...
    ,std(imgFullResNovPostBsl,0,2)/sqrt(size(imgFullResNovPostBsl,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAx,mean(imgOcclResNovPostBsl,2)...
    ,std(imgOcclResNovPostBsl,0,2)/sqrt(size(imgOcclResNovPostBsl,2)), 'lineProps', 'r');
xlabel('Time (s)'), title('Post Nov'), % figClean
% scatters
s(1) = subplot(3,5,6);
scatter(scatFullFamPopPre, scatOcclFamPopPre, sz, cPre, 'filled'); refline(1), xlabel('Full'), ylabel('Occl'), title('Fam Pre'), % figClean
s(2) = subplot(3,5,7);
scatter(scatFullFamPopPost,scatOcclFamPopPost , sz, cPost, 'filled'); refline(1), title('Fam Post'), % figClean
s(3) = subplot(3,5,8);
scatter(scatFullNovPopPre, scatOcclNovPopPre, sz, cPre, 'filled'); refline(1), title('Nov Pre'), % figClean
s(4) = subplot(3,5,9);
scatter(scatFullNovPopPost, scatOcclNovPopPost, sz, cPost, 'filled'); refline(1), title('Nov Post'), % figClean
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
    'PreFamOccl', 'PostFamOccl','PreNovOccl', 'PostNovOccl'}), xtickangle(45), % figClean
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
xlim([0 10]), ylabel('Response dF/F (%)'), xticks([1 2 3 4 5 6 7 8]), 
if nfiles == 6, ylim([-0.1 0.8]), 
elseif nfiles == 5, ylim([0 1]), 
end
xticklabels({'PreFamFull', 'PostFamFull','PreNovFull', 'PostNovFull', ...
    'PreFamOccl', 'PostFamOccl','PreNovOccl', 'PostNovOccl'}), xtickangle(45), % figClean
% compare pre vs post single cell level
g(1) = subplot(3,5,11);
scatter(scatFullFamPopPre, scatFullFamPopPost, sz, cPre, 'filled'); refline(1), xlabel('Pre'), ylabel('Post'),title('Full fam'),% figClean
g(2) = subplot(3,5,12);
scatter(scatOcclFamPopPre, scatOcclFamPopPost , sz, cPost, 'filled'); refline(1), xlabel('Pre'), ylabel('Post'),title('Occl fam'),% figClean
g(3) = subplot(3,5,13);
scatter(scatFullNovPopPre, scatFullNovPopPost, sz, cPre, 'filled'); refline(1), xlabel('Pre'), ylabel('Post'),title('Full nov'),% figClean
g(4) = subplot(3,5,14);
scatter(scatOcclNovPopPre, scatOcclNovPopPost, sz, cPost, 'filled'); refline(1), xlabel('Pre'), ylabel('Post'),title('Occl nov'),% figClean

% axes
if nfiles == 6
    for j = 1:length(t)
        t(j).YLim = [-0.1 0.7]; t(j).YTick = -0.1:0.2:0.7; t(j).XLim = [-1 3]; t(j).XTick = -1:1:3;
    end
    for j = 1:length(s)
%         mn =  round(min([s(:).YLim s(:).XLim]));
%         mx =  round(max([s(:).YLim s(:).XLim]));
        % s(j).YLim = [mn mx]; s(j).YTick = mn:20:mx; s(j).XLim = [mn mx]; s(j).XTick = mn:20:mx;
        s(j).YLim = [-1 3]; s(j).YTick = -1:1:3; s(j).XLim = [-1 3]; s(j).XTick = -1:1:3;
    end
    for j = 1:length(g)
%         mn =  round(min([g(:).YLim g(:).XLim]));
%         mx =  round(max([g(:).YLim g(:).XLim]));
        % g(j).YLim = [mn mx]; g(j).YTick = mn:20:mx; g(j).XLim = [mn mx]; g(j).XTick = mn:20:mx;
        g(j).YLim = [-1 3]; g(j).YTick = -1:1:3; g(j).XLim = [-1 3]; g(j).XTick = -1:1:3;
    end
elseif nfiles == 5
    for j = 1:length(t)
        t(j).YLim = [-0.2 1.4]; t(j).YTick = -0.2:0.2:1.4; t(j).XLim = [-1 3]; t(j).XTick = -1:1:3;
    end
    for j = 1:length(s)
%         mn =  round(min([s(:).YLim s(:).XLim]));
%         mx =  round(max([s(:).YLim s(:).XLim]));
        % s(j).YLim = [mn mx]; s(j).YTick = mn:20:mx; s(j).XLim = [mn mx]; s(j).XTick = mn:20:mx;
        s(j).YLim = [-0.5 3.5]; s(j).YTick = -0.5:0.5:3.5; s(j).XLim = [-0.5 3.5]; s(j).XTick = -0.5:0.5:3.5;
    end
    for j = 1:length(g)
%         mn =  round(min([g(:).YLim g(:).XLim]));
%         mx =  round(max([g(:).YLim g(:).XLim]));
        % g(j).YLim = [mn mx]; g(j).YTick = mn:20:mx; g(j).XLim = [mn mx]; g(j).XTick = mn:20:mx;
        g(j).YLim = [-0.5 3.5]; g(j).YTick = -0.5:0.5:3.5; g(j).XLim = [-0.5 3.5]; g(j).XTick = -0.5:0.5:3.5;
    end
end

if save_fig
    func_save_fig('L23_traceAndScatter_Chronic_og')
    func_save_fig('L5_traceAndScatterAndBoxSeparate')
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

mouseIDPre = [];
mouseIDPost = [];
for i = 1:nfiles
    % prepare some data for linear mixed model effect
    mouseIDPre = [mouseIDPre zeros(1,length(datastructPre(i).scatFull))+i];
    mouseIDPost = [mouseIDPost zeros(1,length(datastructPost(i).scatFull))+i];
end

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


%% color coded chronic plot

% Normalize Post responses
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

figure('Position', [45         439        1146         471]);
sz = 45;
% Post
subplot(1,2,1)
scatter(scatFullFamPopPreSort, scatOcclFamPopPreSort, sz, colors, 'filled');
hold on
xlim([-1 3]); ylim([-1 3]);
xticks(-1:1:3),yticks(-1:1:3)
refline(1)
xlabel('NO fam pre'); ylabel('O fam pre');
title('Colored by Post response blend');
% figClean

% Task
subplot(1,2,2)
scatter(scatFullFamPopPostSort, scatOcclFamPopPostSort, sz, colors, 'filled');  % same colors
hold on
xlim([-1 3]); ylim([-1 3]);
xticks(-1:1:3),yticks(-1:1:3)
refline(1)
xlabel('NO fam post'); ylabel('O fam post');
title('Colored by Post response blend');
% figClean

if save_fig
    func_save_fig('L23_chronic_Prepost_scatter_colorcodedOccltask')
    func_save_fig('L5_chronic_Prepost_scatter_colorcodedOccltask')
end

% Create colorbar figure (un-normalized version)
figure('Position', [1203         495         400         345]);

% Define range for raw values
fullRange = linspace(min(scatFullFamPopPreSort), max(scatFullFamPopPreSort), 256);
occlRange = linspace(min(scatOcclFamPopPreSort), max(scatOcclFamPopPreSort), 256);
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

if save_fig
    func_save_fig('L23_chronic_Prepost_scatter_colorcodedOccltask_colorbar')
    func_save_fig('L5_chronic_Prepost_scatter_colorcodedOccltask_colorbar')
end


%% color coded chronic plot

% Normalize Post responses
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

figure('Position', [45         439        1146         471]);
sz = 45;
% Post
subplot(1,2,1)
scatter(scatFullNovPopPreSort, scatOcclNovPopPreSort, sz, colors, 'filled');
hold on
xlim([-1 3]); ylim([-1 3]);
xticks(-1:1:3),yticks(-1:1:3)
refline(1)
xlabel('NO nov pre'); ylabel('O nov pre');
title('Colored by Post response blend');

% Task
subplot(1,2,2)
scatter(scatFullNovPopPostSort, scatOcclNovPopPostSort, sz, colors, 'filled');  % same colors
hold on
xlim([-1 3]); ylim([-1 3]);
xticks(-1:1:3),yticks(-1:1:3)
refline(1)
xlabel('NO nov post'); ylabel('O nov post');
title('Colored by Post response blend');


% Create colorbar figure (un-normalized version)
figure('Position', [1203         495         400         345]);

% Define range for raw values
fullRange = linspace(min(scatFullNovPopPreSort), max(scatFullNovPopPreSort), 256);
occlRange = linspace(min(scatOcclNovPopPreSort), max(scatOcclNovPopPreSort), 256);
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
% figClean

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

% figClean

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

% figClean

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

% figClean

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
% figClean

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
xlim([-1 3]), ylim([-1 3]), refline(1), xlabel('NO'), ylabel('O'), title('Pre'), % figClean
subplot(1,2,2)
scatter(scatFullFamPopPost, scatOcclFamPopPost), hold on
scatter(scatFullFamPopPost(ix), scatOcclFamPopPost(ix), 'filled')
xlim([-1 3]), ylim([-1 3]), refline(1), xlabel('NO'), ylabel('O'), title('Post'), % figClean


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

% LMEM
mouseIDPre = [];
mouseIDPost = [];
for i = 1:nfiles
    % prepare some data for linear mixed model effect
    mouseIDPre = [mouseIDPre zeros(1,length(datastructPre(i).scatFull))+i];
    mouseIDPost = [mouseIDPost zeros(1,length(datastructPost(i).scatFull))+i];
end

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
xticklabels({'N','E'}), % figClean
subplot(1,2,2)
scatter([1 2],[mean(sparsenessOcclPre) mean(sparsenessOcclPost)], sz, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2],[mean(sparsenessOcclPre) mean(sparsenessOcclPost)], ...
    [calcSem(sparsenessOcclPre) calcSem(sparsenessOcclPost)] ...
    ,[calcSem(sparsenessOcclPre) calcSem(sparsenessOcclPost)]);
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 3]), ylim([0 0.65]), ylabel('Sparseness O'), xticks([1 2])
xticklabels({'N','E'}), % figClean

%%%% fam
[rDiff, pDiff] = corrcoef(sparsenessFullPre, scatFullFamPopPre - scatFullFamPopPost);
[rRes, pRes]   = corrcoef(sparsenessFullPre, scatFullFamPopPre);
figure('Position', [249 514 1124 373])
% --- Subplot 1: Difference (naive - expert)
subplot(1,2,1)
x1 = sparsenessFullPre;
y1 = scatFullFamPopPre - scatFullFamPopPost;
scatter(x1, y1, 30, 'filled', 'k'), hold on; refline, ylim([-1.5 3])
ylabel('NO fam naive - NO fam expert'), xlabel('NO fam selectivity (naive)'), % figClean
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
ylabel('NO fam naive'), xlabel('NO fam selectivity (naive)'), % figClean
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
ylabel('NO nov naive - NO nov expert'), xlabel('NO nov selectivity (naive)'), % figClean
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
ylabel('NO nov naive'), xlabel('NO nov selectivity (naive)'), % figClean
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





% occl  LMEM
data = cat(2, sparsenessOcclPre,sparsenessOcclPost, sparsenessOcclTask)';
mouseID = categorical(cat(2, mouseIDPreOccl,mouseIDPostOccl,mouseIDTaskOccl))';
condition = categorical(cat(1, ones(length(mouseIDPreOccl),1),ones(length(mouseIDPostOccl),1)+1,ones(length(mouseIDTaskOccl),1)+2));
clear statTbl, statTbl = table(data, mouseID, condition);
lmeSpars = fitlme(statTbl, 'data ~ condition + (1|mouseID)', 'CheckHessian', 1,'FitMethod', 'REML', 'StartMethod', 'random');
statsSparsOccl = anova(lmeSpars,'DFMethod','Satterthwaite');
statTblSparsOccl = makeStatTbl(lmeSpars);

if save_fig
    func_save_fig('L23_lifetimeSpars_og')
    func_save_fig('L5_lifetimeSpars_og')
end

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


%% Dampening or sharpening analyses
% order responses to the four images per neuron, sort in ascending order,
% average over neurons, pre vs post. See review de Lange fig 2.
ix = famIdx;

clear imgFullPreSem imgFullPostSem imgOcclPreSem imgOcclPostSem
imgFullPre = sort(squeeze(mean(imgFullResMnPopPre(vecAxSt,ix,:)))-squeeze(mean(imgFullResMnPopPre(vecAxSp,ix,:))), 'ascend');
imgFullPreMn = mean(imgFullPre,2); for i = 1:size(imgFullPre,1), imgFullPreSem(i) = calcSem(imgFullPre(i,:));end
imgFullPost = sort(squeeze(mean(imgFullResMnPopPost(vecAxSt,ix,:)))-squeeze(mean(imgFullResMnPopPost(vecAxSp,ix,:))), 'ascend');
imgFullPostMn = mean(imgFullPost,2); for i = 1:size(imgFullPost,1), imgFullPostSem(i) = calcSem(imgFullPost(i,:));end
imgOcclPre = sort(squeeze(mean(imgOcclResMnPopPre(vecAxSt,ix,:)))-squeeze(mean(imgOcclResMnPopPre(vecAxSp,ix,:))), 'ascend');
imgOcclPreMn = mean(imgOcclPre,2); for i = 1:size(imgOcclPre,1), imgOcclPreSem(i) = calcSem(imgOcclPre(i,:));end
imgOcclPost = sort(squeeze(mean(imgOcclResMnPopPost(vecAxSt,ix,:)))-squeeze(mean(imgOcclResMnPopPost(vecAxSp,ix,:))), 'ascend');
imgOcclPostMn = mean(imgOcclPost,2); for i = 1:size(imgOcclPost,1), imgOcclPostSem(i) = calcSem(imgOcclPost(i,:));end

figure('Position', [650         456        1191         420])
subplot(1,2,1)
bar(imgFullPreMn, 'FaceAlpha', 0.5), hold on
er = errorbar(imgFullPreMn,imgFullPreSem); 
er.Color = [0 0.4470 0.7410]; er.LineStyle = 'none'; er.LineWidth = 2; title('Average response per condition')
bar(imgFullPostMn, 'FaceAlpha', 0.5)
er = errorbar(imgFullPostMn,imgFullPostSem); 
er.Color = [0.9290 0.6940 0.1250]; er.LineStyle = 'none'; er.LineWidth = 2; title('Average response per condition')
title('Response strength per full image')
ylabel('dF/F')
subplot(1,2,2)
bar(imgOcclPreMn, 'FaceAlpha', 0.5), hold on
er = errorbar(imgOcclPreMn,imgOcclPreSem); 
er.Color = [0 0.4470 0.7410]; er.LineStyle = 'none'; er.LineWidth = 2; title('Average response per condition')
bar(imgOcclPostMn, 'FaceAlpha', 0.5)
er = errorbar(imgOcclPostMn,imgOcclPostSem); 
er.Color = [0.9290 0.6940 0.1250]; er.LineStyle = 'none'; er.LineWidth = 2; title('Average response per condition')
title('Response strength per Occl image')
ylabel('dF/F')

% selectivity across images per ROI
ix = famIdx;

imgFullPre = sort(squeeze(mean(imgFullResMnPopPre(vecAxSt,ix,:)))-squeeze(mean(imgFullResMnPopPre(vecAxSp,ix,:))), 'ascend');
imgFullPost = sort(squeeze(mean(imgFullResMnPopPost(vecAxSt,ix,:)))-squeeze(mean(imgFullResMnPopPost(vecAxSp,ix,:))), 'ascend');
imgOcclPre = sort(squeeze(mean(imgOcclResMnPopPre(vecAxSt,ix,:)))-squeeze(mean(imgOcclResMnPopPre(vecAxSp,ix,:))), 'ascend');
imgOcclPost = sort(squeeze(mean(imgOcclResMnPopPost(vecAxSt,ix,:)))-squeeze(mean(imgOcclResMnPopPost(vecAxSp,ix,:))), 'ascend');

selecFullPre = selectCalc(imgFullPre);
selecFullPost = selectCalc(imgFullPost);
selecOcclPre = selectCalc(imgOcclPre);
selecOcclPost = selectCalc(imgOcclPost);

figure('Position', [650         456        1191         420])
subplot(1,2,1)
bar(1, mean(selecFullPre), 'FaceAlpha', 0.5), hold on
er = errorbar(1, mean(selecFullPre),calcSem(selecFullPre)); 
er.Color = [0 0.4470 0.7410]; er.LineStyle = 'none'; er.LineWidth = 2; title('Average response per condition')
bar(2, mean(selecFullPost), 'FaceAlpha', 0.5)
er = errorbar(2, mean(selecFullPost),calcSem(selecFullPost)); 
er.Color = [0.9290 0.6940 0.1250]; er.LineStyle = 'none'; er.LineWidth = 2; title('Average response per condition')
title('Full selectivity')
ylabel('Selectivity')
subplot(1,2,2)
bar(1, mean(selecOcclPre), 'FaceAlpha', 0.5), hold on
er = errorbar(1, mean(selecOcclPre),calcSem(selecOcclPre)); 
er.Color = [0 0.4470 0.7410]; er.LineStyle = 'none'; er.LineWidth = 2; title('Average response per condition')
bar(2, mean(selecOcclPost), 'FaceAlpha', 0.5)
er = errorbar(2, mean(selecOcclPost),calcSem(selecOcclPost)); 
er.Color = [0.9290 0.6940 0.1250]; er.LineStyle = 'none'; er.LineWidth = 2; title('Average response per condition')
title('Occl selectivity')
ylabel('Selectivity')


% ANOVA
% based on one way anova method from Paolo, which does a one way anova
% per neurons across images
imgFullResPopPre = datastructPre(1).imgFullRes*100;
imgOcclResPopPre = datastructPre(1).imgOcclRes*100;
imgFullResPopPost = datastructPost(1).imgFullRes*100;
imgOcclResPopPost = datastructPost(1).imgOcclRes*100;

for i = 2:nfiles
    imgFullResPopPre = cat(4, imgFullResPopPre, datastructPre(i).imgFullRes*100);
    imgOcclResPopPre = cat(4, imgOcclResPopPre, datastructPre(i).imgOcclRes*100);
    imgFullResPopPost = cat(4, imgFullResPopPost, datastructPost(i).imgFullRes*100);
    imgOcclResPopPost = cat(4, imgOcclResPopPost, datastructPost(i).imgOcclRes*100);
end

clear fFullPre fOcclPre fFullPost fOcclPost
for j = 1:size(imgFullResPopPre,4)
    [~, F] = anova1(squeeze(mean(imgFullResPopPre(vecAxSt,famIdx,:,j))-mean(imgFullResPopPre(vecAxSp,famIdx,:,j)))',[],'off');
    fFullPre(j) = F{2,5};
    [~, F] = anova1(squeeze(mean(imgOcclResPopPre(vecAxSt,famIdx,:,j))-mean(imgOcclResPopPre(vecAxSp,famIdx,:,j)))',[],'off');
    fOcclPre(j) = F{2,5};
end
for j = 1:size(imgFullResPopPost,4)
    [~, F] = anova1(squeeze(mean(imgFullResPopPost(vecAxSt,famIdx,:,j))-mean(imgFullResPopPost(vecAxSp,famIdx,:,j)))',[],'off');
    fFullPost(j) = F{2,5};
    [~, F] = anova1(squeeze(mean(imgOcclResPopPost(vecAxSt,famIdx,:,j))-mean(imgOcclResPopPost(vecAxSp,famIdx,:,j)))',[],'off');
    fOcclPost(j) = F{2,5};
end

figure('Position', [1067         436         204         407])
scatter([1 2 4 5],[mean(fFullPre) mean(fFullPost) mean(fOcclPre) mean(fOcclPost)], 30, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2 4 5],[mean(fFullPre) mean(fFullPost) mean(fOcclPre) mean(fOcclPost)], ...
    [calcSem(fFullPre) calcSem(fFullPost) calcSem(fOcclPre) calcSem(fOcclPost)] ...
    ,[calcSem(fFullPre) calcSem(fFullPost) calcSem(fOcclPre) calcSem(fOcclPost)]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 6]), ylabel('Anova F-stat'), xticks([1 2 4 5])
xticklabels({'Full Pre', 'Full Post','Occl Pre', 'Occl Post'}), xtickangle(45), % figClean


% based on variance from Paolo
% Variance: So, Ri is the average variance across repetitions. 
% So you compute R_all(i) = var(R(i,:)) for each stimulus i (so it�s the variance across the 20 reps), 
% and then you average R_all (which is 6 numbers) to get Ri. 
% And then you compute the total variance by just considering all the data (so 20 reps x 6 stims, not just 6 values).
%  variance = ( var(R(:)) - sum(var(Ri)) ) / var(R(:))

clear varValFullPre varValOcclPre varValFullPost varValOcclPost
for j = 1:size(imgFullResPopPre,4)
    temp = squeeze(mean(imgFullResPopPre(vecAxSt,:,:,j))-mean(imgFullResPopPre(vecAxSp,:,:,j)));
    varValFullPre(j) = calcVar(temp);
    temp = squeeze(mean(imgOcclResPopPre(vecAxSt,:,:,j))-mean(imgOcclResPopPre(vecAxSp,:,:,j)));
    varValOcclPre(j) = calcVar(temp);
end
for j = 1:size(imgFullResPopPost,4)
    temp = squeeze(mean(imgFullResPopPost(vecAxSt,:,:,j))-mean(imgFullResPopPost(vecAxSp,:,:,j)));
    varValFullPost(j) = calcVar(temp);
    temp = squeeze(mean(imgOcclResPopPost(vecAxSt,:,:,j))-mean(imgOcclResPopPost(vecAxSp,:,:,j)));
    varValOcclPost(j) = calcVar(temp);
end
figure('Position', [1067         436         204         407])
scatter([1 2 4 5],[mean(varValFullPre) mean(varValFullPost) mean(varValOcclPre) mean(varValOcclPost)], 30, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2 4 5],[mean(varValFullPre) mean(varValFullPost) mean(varValOcclPre) mean(varValOcclPost)], ...
    [calcSem(varValFullPre) calcSem(varValFullPost) calcSem(varValOcclPre) calcSem(varValOcclPost)] ...
    ,[calcSem(varValFullPre) calcSem(varValFullPost) calcSem(varValOcclPre) calcSem(varValOcclPost)]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 6]), ylabel('Variance'), xticks([1 2 4 5])
xticklabels({'Full Pre', 'Full Post','Occl Pre', 'Occl Post'}), xtickangle(45), % figClean


%% selectivity of response versus response strength pre vs post training
% mean and max
varValFullPre(varValFullPre>20)=10;
% mean
resStrFullPre = mean(imgFullPre);
resStrFullPost = mean(imgFullPost);
resStrOcclPre = mean(imgOcclPre);
resStrOcclPost = mean(imgOcclPost);

% % or max
% resStrFullPre = max(imgFullPre);
% resStrFullPost = max(imgFullPost);
% resStrOcclPre = max(imgOcclPre);
% resStrOcclPost = max(imgOcclPost);


% versus lifetime sparseness or variance (Paolo)
figure('Position', [360   228   989   656])
subplot(2,2,1)
scatter(selecFullPre, resStrFullPre, 20, 'k', 'filled'), refline, ylabel('Resp strength'), xlabel('variance'), title('Full pre')
subplot(2,2,2)
scatter(selecFullPost, resStrFullPost, 20, 'k', 'filled'), refline, ylabel('Resp strength'), xlabel('variance'), title('Full post')
subplot(2,2,3)
scatter(selecOcclPre, resStrOcclPre, 20, 'k', 'filled'), refline, ylabel('Resp strength'), xlabel('variance'), title('Occl pre')
subplot(2,2,4)
scatter(selecOcclPost, resStrOcclPost, 20, 'k', 'filled'), refline, ylabel('Resp strength'), xlabel('variance'), title('Occl post')
% figure('Position', [360   228   989   656])
% subplot(2,2,1)
% scatter(varValFullPre, resStrFullPre, 20, 'k', 'filled'), refline, ylabel('Resp strength'), xlabel('variance'), title('Full pre')
% subplot(2,2,2)
% scatter(varValFullPost, resStrFullPost, 20, 'k', 'filled'), refline, ylabel('Resp strength'), xlabel('variance'), title('Full post')
% subplot(2,2,3)
% scatter(varValOcclPre, resStrOcclPre, 20, 'k', 'filled'), refline, ylabel('Resp strength'), xlabel('variance'), title('Occl pre')
% subplot(2,2,4)
% scatter(varValOcclPost, resStrOcclPost, 20, 'k', 'filled'), refline, ylabel('Resp strength'), xlabel('variance'), title('Occl post')

% colors = cmapL([0 0 0;0 0 0; 1 0 0], 256);
% lims = [-10 10];
% fullColors = squeeze(SetLimits(resStrFullPre-resStrFullPost, lims, colors));
% occlColors = squeeze(SetLimits(resStrOcclPre-resStrOcclPost, lims, colors));
% 
% figure('Position', [360   228   989   348])
% subplot(1,2,1)
% scatter(varValFullPre, varValFullPost, 20, fullColors, 'filled'), refline(1), ylabel('variance post'), xlabel('variance pre'), title('Full, color = strength diff')
% subplot(1,2,2)
% scatter(varValOcclPre, varValOcclPost, 20, occlColors, 'filled'), refline(1), ylabel('variance post'), xlabel('variance pre'), title('Occl, color = strength diff')

figure('Position', [360   228   989   348])
subplot(1,2,1)
scatter(varValFullPre, resStrFullPre-resStrFullPost, 20, 'filled'), refline, ylabel('Res pre-post'), xlabel('variance pre'), title('Full')
subplot(1,2,2)
scatter(varValOcclPre, resStrOcclPre-resStrOcclPost, 20, 'filled'), refline, ylabel('Res pre-post'), xlabel('variance pre'), title('Occl')

figure('Position', [360   228   989   348])
subplot(1,2,1)
scatter(varValFullPost, resStrFullPre-resStrFullPost, 20, 'filled'), refline, ylabel('Res pre-post'), xlabel('variance post'), title('Full')
subplot(1,2,2)
scatter(varValOcclPost, resStrOcclPre-resStrOcclPost, 20, 'filled'), refline, ylabel('Res pre-post'), xlabel('variance post'), title('Occl')

figure('Position', [360   228   989   348])
subplot(1,2,1)
scatter(varValFullPre, varValFullPost, 20, 'filled'), refline, ylabel('Var post'), xlabel('variance pre'), title('Full')
subplot(1,2,2)
scatter(varValOcclPre, varValOcclPost, 20, 'filled'), refline, ylabel('Var post'), xlabel('variance pre'), title('Occl')

figure('Position', [360   228   989   656])
subplot(2,2,1)
scatter(selecFullPre, resStrFullPre-resStrFullPost, 20, 'k', 'filled'), refline, ylabel('Resp strength'), xlabel('variance'), title('Full pre')
subplot(2,2,2)
scatter(selecFullPost, resStrFullPost, 20, 'k', 'filled'), refline, ylabel('Resp strength'), xlabel('variance'), title('Full post')
subplot(2,2,3)
scatter(selecOcclPre, resStrOcclPre, 20, 'k', 'filled'), refline, ylabel('Resp strength'), xlabel('variance'), title('Occl pre')
subplot(2,2,4)
scatter(selecOcclPost, resStrOcclPost, 20, 'k', 'filled'), refline, ylabel('Resp strength'), xlabel('variance'), title('Occl post')




%% Traces OLD STUFF!!!!

fullSignPopPre = datastructPre(1).fullSign*100;
occlSignPopPre = datastructPre(1).occlSign*100;
fullSignPopPost = datastructPost(1).fullSign*100;
occlSignPopPost = datastructPost(1).occlSign*100;

for i = 2:nfiles
    fullSignPopPre = cat(2, fullSignPopPre, datastructPre(i).fullSign*100);
    occlSignPopPre = cat(2, occlSignPopPre, datastructPre(i).occlSign*100);
    fullSignPopPost = cat(2, fullSignPopPost, datastructPost(i).fullSign*100);
    occlSignPopPost = cat(2, occlSignPopPost, datastructPost(i).occlSign*100);
end

% 
% fullSignPopPre = normalize(fullSignPopPre,1,'norm');
% occlSignPopPre = normalize(occlSignPopPre,1,'norm');
% fullSignPopPost = normalize(fullSignPopPost,1,'norm');
% occlSignPopPost = normalize(occlSignPopPost,1,'norm');

% average trace for each condition
figure('Position', [306         334        1148         454])
s(1) = subplot(1,2,1);
shadedErrorBar(vecAx,mean(fullSignPopPre,2)-mean(mean(fullSignPopPre(vecAxSp,:)),2)...
    ,std(fullSignPopPre,0,2)/sqrt(size(fullSignPopPre,2)), 'lineProps', 'k');
hold on
shadedErrorBar(vecAx,mean(occlSignPopPre,2)-mean(mean(occlSignPopPre(vecAxSp,:)),2)...
    ,std(occlSignPopPre,0,2)/sqrt(size(occlSignPopPre,2)), 'lineProps', 'r');
xlabel('Time (s)')
ylabel('dF/F (%)')
xticks(-1:1:3)
xlim([-1 3])
title('Mean response pre training')
% set(gca, 'LineWidth', 1, 'FontSize', 12)
box off, % figClean
s(2) = subplot(1,2,2);
shadedErrorBar(vecAx,mean(fullSignPopPost,2)-mean(mean(fullSignPopPost(vecAxSp,:)),2)...
    ,std(fullSignPopPost,0,2)/sqrt(size(fullSignPopPost,2)), 'lineProps', 'k');
hold on
shadedErrorBar(vecAx,mean(occlSignPopPost,2)-mean(mean(occlSignPopPost(vecAxSp,:)),2)...
    ,std(occlSignPopPost,0,2)/sqrt(size(occlSignPopPost,2)), 'lineProps', 'r');
xlabel('Time (s)')
ylabel('dF/F (%)')
xticks(-1:1:3)
xlim([-1 3])
% set(gca, 'LineWidth', 1, 'FontSize', 12)
box off, % figClean
title('Mean response post training')
% mnY = min(min(s(:).YLim));mxY = max(max(s(:).YLim));
% s(1).YLim = [mnY mxY]; s(2).YLim = [mnY mxY];
% s(1).YLim = [-1 4]; s(2).YLim = [-1 4]; % for L2/3
s(1).YLim = [-1 18]; s(2).YLim = [-1 18]; % for L5


fullSignPopPreBsl = fullSignPopPre-mean(fullSignPopPre(vecAxSp,:));
occlSignPopPreBsl = occlSignPopPre-mean(occlSignPopPre(vecAxSp,:));
fullSignPopPostBsl = fullSignPopPost-mean(fullSignPopPost(vecAxSp,:));
occlSignPopPostBsl = occlSignPopPost-mean(occlSignPopPost(vecAxSp,:));

% sort on trace of preference pre training, could be either of the 2 for live figure
traceToSortPre = fullSignPopPreBsl;
[MniPre] = mean(traceToSortPre(vecAxSt,:));
[~,RsortedMnPre] = sort(MniPre,'descend');

[~, MxiPre] = max(traceToSortPre);
[~,RsortedMxPre] = sort(MxiPre,'ascend');

% sort on trace of preference post training, could be either of the 2 for live figure
traceToSortPost = fullSignPopPostBsl;
[MniPost] = mean(traceToSortPost(vecAxSt,:));
[~,RsortedMnPost] = sort(MniPost,'descend');

[~, MxiPost] = max(traceToSortPost);
[~,RsortedMxPost] = sort(MxiPost,'ascend');



% plot with each condition in separate subplot, axes are similar scaling
clear p
figure('Position', [377 125 1126 818])
p(1) = subplot(1,4,1);
imagesc(vecAx, [], fullSignPopPreBsl(:, RsortedMnPre)')
title('Pre Full'),  colorbar, xlabel('Time (s)'), ylabel('Neurons'), set(gca,'TickDir','out');
p(2) = subplot(1,4,2);
imagesc(vecAx, [], occlSignPopPreBsl(:, RsortedMnPre)')
title('Pre Occl'), colorbar, set(gca,'TickDir','out');
p(3) = subplot(1,4,3);
imagesc(vecAx, [], fullSignPopPostBsl(:, RsortedMnPre)')
title('Post Full'), colorbar, set(gca,'TickDir','out');
p(4) = subplot(1,4,4);
imagesc(vecAx, [], occlSignPopPostBsl(:, RsortedMnPre)')
title('Post Occl'), colorbar, set(gca,'TickDir','out');
allCLim = get(p, {'CLim'});
allCLim = cat(2, allCLim{:});
set(p, 'CLim', [min(allCLim), max(allCLim)]);
% set(p, 'CLim', [-5, 60]); % for L2/3
set(p, 'CLim', [-5, 45]); % for L5
colormap hot


% figure, 
% subplot(1,2,1)
% scatter(1:size(fullSignPopPreBsl,2),mean(fullSignPopPreBsl(vecAxSt,RsortedMnPre)))
% hold on, scatter(1:size(fullSignPopPreBsl,2),mean(occlSignPopPreBsl(vecAxSt,RsortedMnPre)))
% refline
% subplot(1,2,2)
% scatter(1:size(fullSignPopPostBsl,2),mean(fullSignPopPostBsl(vecAxSt,RsortedMnPost)))
% hold on, scatter(1:size(fullSignPopPostBsl,2),mean(occlSignPopPostBsl(vecAxSt,RsortedMnPost)))
% refline


% % plot with each condition in separate subplot, axes are similar scaling
% clear q
% figure('Position', [377 125 1126 818])
% q(1) = subplot(1,4,1);
% imagesc(vecAx, [], fullSignPopPreBsl(:, RsortedMxPre)')
% title('Pre Full'),  colorbar, xlabel('Time (s)'), ylabel('Neurons')
% q(2) = subplot(1,4,2);
% imagesc(vecAx, [], occlSignPopPreBsl(:, RsortedMxPre)')
% title('Pre Occl'), colorbar
% q(3) = subplot(1,4,3);
% imagesc(vecAx, [], fullSignPopPostBsl(:, RsortedMxPost)')
% title('Post Full'), colorbar
% q(4) = subplot(1,4,4);
% imagesc(vecAx, [], occlSignPopPostBsl(:, RsortedMxPost)')
% title('Post Occl'), colorbar
% allCLim = get(q, {'CLim'});
% allCLim = cat(2, allCLim{:});
% set(q, 'CLim', [min(allCLim), max(allCLim)]);
% % set(q, 'CLim', [-0.05, 0.6]);
% colormap hot
% 
%% single cells

figure
for i = 1:size(fullSignPopPre,2)

    s1 = subplot(1,2,1);
    plot(vecAx, fullSignPopPre(:,i)-mean(fullSignPopPre(vecAxSp,i))), hold on
    plot(vecAx, occlSignPopPre(:,i)-mean(occlSignPopPre(vecAxSp,i))), hold off
    
    s2 = subplot(1,2,2);
    plot(vecAx, fullSignPopPost(:,i)-mean(fullSignPopPost(vecAxSp,i))), hold on
    plot(vecAx, occlSignPopPost(:,i)-mean(occlSignPopPost(vecAxSp,i))), hold off
%     s3 = subplot(2,2,3);
%     plot(vecAx, squeeze(mean(imgFullResMnPopPre(:,imgIdx2,idxPre),2))-mean(squeeze(mean(imgFullResMnPopPre(vecAxSp,imgIdx2,idxPre),2))))
%     s4 = subplot(2,2,4);
%     plot(vecAx, squeeze(mean(imgFullResMnPopPost(:,imgIdx2,idxPost),2))-mean(squeeze(mean(imgFullResMnPopPost(vecAxSp,imgIdx2,idxPost),2))))

    ylimMx = max([s1.YLim s2.YLim]);
    ylimMn = min([s1.YLim s2.YLim]);
    s1.YLim = [ylimMn ylimMx];
    s2.YLim = [ylimMn ylimMx];
pause
end





%% In case of comparing familiar vs novel and full vs occl
% famIdx = [1 2 4 5];
famIdx = [1 2 4 5];
novIdx = [3 6];
% novIdx = [4 5];

imgFullResMnPopPre = datastructPre(1).imgFullResMn*100;
imgOcclResMnPopPre = datastructPre(1).imgOcclResMn*100;
imgFullResMnPopPost = datastructPost(1).imgFullResMn*100;
imgOcclResMnPopPost = datastructPost(1).imgOcclResMn*100;

for i = 2:nfiles
    imgFullResMnPopPre = cat(3, imgFullResMnPopPre, datastructPre(i).imgFullResMn*100);
    imgOcclResMnPopPre = cat(3, imgOcclResMnPopPre, datastructPre(i).imgOcclResMn*100);
    imgFullResMnPopPost = cat(3, imgFullResMnPopPost, datastructPost(i).imgFullResMn*100);
    imgOcclResMnPopPost = cat(3, imgOcclResMnPopPost, datastructPost(i).imgOcclResMn*100);
end

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


%% percentage and mean response of significant neurons
percFullSignPre = zeros(nfiles,1);
percOcclSignPre = zeros(nfiles,1);
percOverlapPre = zeros(nfiles,1);
percFullSignPost = zeros(nfiles,1);
percOcclSignPost = zeros(nfiles,1);
percOverlapPost = zeros(nfiles,1);

for i = 1:nfiles
    hValFullPre = datastructPre(i).hValFull;
    hValOcclPre = datastructPre(i).hValOccl;
    hValFullPost = datastructPost(i).hValFull;
    hValOcclPost = datastructPost(i).hValOccl;
    percFullSignPre(i) = sum(sum(hValFullPre)>0&sum(hValOcclPre)==0)/length(hValFullPre)*100;
    percOcclSignPre(i) = sum(sum(hValOcclPre)>0&sum(hValFullPre)==0)/length(hValOcclPre)*100;
    percOverlapPre(i) = sum(sum(hValFullPre)>0&sum(hValOcclPre)>0)/length(hValFullPre)*100;
    percFullSignPost(i) = sum(sum(hValFullPost)>0&sum(hValOcclPost)==0)/length(hValFullPost)*100;
    percOcclSignPost(i) = sum(sum(hValOcclPost)>0&sum(hValFullPost)==0)/length(hValOcclPost)*100;
    percOverlapPost(i) = sum(sum(hValFullPost)>0&sum(hValOcclPost)>0)/length(hValFullPost)*100;

end


% figure 
% subplot(1,3,1)
% scatter(ones(nfiles,1), percFullSignPre) ,hold on 
% scatter(ones(nfiles,1)+1, percFullSignPost) 
% plot(1:2, [percFullSignPre percFullSignPost])
% xlim([0 3])
% subplot(1,3,2)
% scatter(ones(nfiles,1), percOcclSignPre) ,hold on 
% scatter(ones(nfiles,1)+1, percOcclSignPost) 
% plot(1:2, [percOcclSignPre percOcclSignPost])
% xlim([0 3])
% subplot(1,3,3)
% scatter(ones(nfiles,1), percOverlapPre) ,hold on 
% scatter(ones(nfiles,1)+1, percOverlapPost) 
% plot(1:2, [percOverlapPre percOverlapPost])
% xlim([0 3])

% per mouse
figure('Position', [819   262   417   696])
hold on
bar(1, mean(percFullSignPre), 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2)
bar(2, mean(percFullSignPost), 'FaceColor', col2, 'EdgeColor', 'k', 'LineWidth', 2)
plot(1:2,[percFullSignPre,percFullSignPost], 'Color', [0 0 0 0.5], 'LineWidth', 1);
% scatter(ones(length(percFullSignPre),1), percFullSignPre, 'LineWidth', 1, 'markerfacecolor',col1,'markeredgecolor','k')
% scatter(ones(length(percFullSignPost),1)+1, percFullSignPost, 'LineWidth', 1, 'markerfacecolor',col2,'markeredgecolor','k')
bar(4, mean(percOcclSignPre), 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2)
bar(5, mean(percOcclSignPost), 'FaceColor', col2, 'EdgeColor', 'k', 'LineWidth', 2)
plot(4:5,[percOcclSignPre,percOcclSignPost], 'Color', [0 0 0 0.5], 'LineWidth', 1);
% scatter(ones(length(percOcclSignPre),1)+3, percOcclSignPre, 'LineWidth', 1, 'markerfacecolor',col1,'markeredgecolor','k')
% scatter(ones(length(percOcclSignPost),1)+4, percOcclSignPost, 'LineWidth', 1, 'markerfacecolor',col2,'markeredgecolor','k')
% bar(7, mean(percOverlapPre), 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2)
% bar(8, mean(percOverlapPost), 'FaceColor', col2, 'EdgeColor', 'k', 'LineWidth', 2)
% plot(7:8,[percOverlapPre,percOverlapPost], 'Color', [0 0 0 0.5], 'LineWidth', 1);
% scatter(ones(length(percOverlapPre),1)+6, percOverlapPre, 'LineWidth', 1, 'markerfacecolor',col1,'markeredgecolor','k')
% scatter(ones(length(percOverlapPost),1)+7, percOverlapPost, 'LineWidth', 1, 'markerfacecolor',col2,'markeredgecolor','k')
% xticks([1 2 4 5 7 8])
xticks([1 2 4 5])
set(gca, 'XTickLabels', [])
% xlim([0 9])
xlim([0 6])
ylim([0 70]) % for L2/3
% ylim([0 90]) % for L5
ylabel('Percentage significantly responding neurons')
% title('Mean response onset latency')
% figClean


% per mouse
figure('Position', [819   262   417   696])
hold on
bar(1, mean(percFullSignPre), 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2)
bar(2, mean(percFullSignPost), 'FaceColor', col2, 'EdgeColor', 'k', 'LineWidth', 2)
plot(1:2,[percFullSignPre,percFullSignPost], 'Color', [0 0 0 0.5], 'LineWidth', 1);
% scatter(ones(length(percFullSignPre),1), percFullSignPre, 'LineWidth', 1, 'markerfacecolor',col1,'markeredgecolor','k')
% scatter(ones(length(percFullSignPost),1)+1, percFullSignPost, 'LineWidth', 1, 'markerfacecolor',col2,'markeredgecolor','k')
bar(4, mean(percOcclSignPre), 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2)
bar(5, mean(percOcclSignPost), 'FaceColor', col2, 'EdgeColor', 'k', 'LineWidth', 2)
plot(4:5,[percOcclSignPre,percOcclSignPost], 'Color', [0 0 0 0.5], 'LineWidth', 1);
% scatter(ones(length(percOcclSignPre),1)+3, percOcclSignPre, 'LineWidth', 1, 'markerfacecolor',col1,'markeredgecolor','k')
% scatter(ones(length(percOcclSignPost),1)+4, percOcclSignPost, 'LineWidth', 1, 'markerfacecolor',col2,'markeredgecolor','k')
bar(7, mean(percOverlapPre), 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2)
bar(8, mean(percOverlapPost), 'FaceColor', col2, 'EdgeColor', 'k', 'LineWidth', 2)
plot(7:8,[percOverlapPre,percOverlapPost], 'Color', [0 0 0 0.5], 'LineWidth', 1);
% scatter(ones(length(percOverlapPre),1)+6, percOverlapPre, 'LineWidth', 1, 'markerfacecolor',col1,'markeredgecolor','k')
% scatter(ones(length(percOverlapPost),1)+7, percOverlapPost, 'LineWidth', 1, 'markerfacecolor',col2,'markeredgecolor','k')
xticks([1 2 4 5 7 8])
% xticks([1 2 4 5])
set(gca, 'XTickLabels', [])
xlim([0 9])
% xlim([0 6])
ylim([0 70]) % for L2/3
% ylim([0 90]) % for L5
ylabel('Percentage significantly responding neurons')
% title('Mean response onset latency')
% figClean

% [pFull] = signrank(percFullSignPre, percFullSignPost)
% [pOccl] = signrank(percOcclSignPre, percOcclSignPost)
[~, pFull] = ttest(percFullSignPre, percFullSignPost)
[~, pOccl] = ttest(percOcclSignPre, percOcclSignPost)
[~ ,pBoth] = ttest(percOverlapPre, percOverlapPost)
% [pFull] = signrank(percFullSignPre, percFullSignPost)
% [pOccl] = signrank(percOcclSignPre, percOcclSignPost)
% [pBoth] = signrank(percOverlapPre, percOverlapPost)



%% mean response of neurons incl double sign responders

% WORK ON THIS
for i = 1:nfiles
    hValFullPre = datastructPre(i).hValFull;
    hValOcclPre = datastructPre(i).hValOccl;
    idxPre = sum(hValFullPre)>0&sum(hValOcclPre)==0;
   
    hValFullPost = datastructPost(i).hValFull;
    hValOcclPost = datastructPost(i).hValOccl;
    idxPost = sum(hValFullPost)>0&sum(hValOcclPost)==0;

end

%% Timing analysis
% Timing analysis
thres = 2;
start = 0; finish = 0.5;

latPreFull = zeros(nfiles,1);
latPreOccl = zeros(nfiles,1);
latPostFull = zeros(nfiles,1);
latPostOccl = zeros(nfiles,1);

postDataFull = [];
postDataOccl = [];
mouseIDPost = [];
% imgFullResMnPopPre = datastructPre(1).imgFullResMn;
% imgOcclResMnPopPre = datastructPre(1).imgOcclResMn;
% imgFullResMnPopPost = datastructPost(1).imgFullResMn;
% imgOcclResMnPopPost = datastructPost(1).imgOcclResMn;
% 
% for i = 2:nfiles
%     imgFullResMnPopPre = cat(3, imgFullResMnPopPre, datastructPre(i).imgFullResMn);  
%     imgOcclResMnPopPre = cat(3, imgOcclResMnPopPre, datastructPre(i).imgOcclResMn);  
%     imgFullResMnPopPost = cat(3, imgFullResMnPopPost, datastructPost(i).imgFullResMn);  
%     imgOcclResMnPopPost = cat(3, imgOcclResMnPopPost, datastructPost(i).imgOcclResMn);  
% end
% 
for i = 1:nfiles
%     % get significance matrices
%     hValFullPre = datastructPre(i).hValFull;
%     hValOcclPre = datastructPre(i).hValOccl;
%     hValFullPost = datastructPost(i).hValFull;
%     hValOcclPost = datastructPost(i).hValOccl;




    fullSignPre = datastructPre(i).fullSign;
    occlSignPre = datastructPre(i).occlSign;
    fullSignPost = datastructPost(i).fullSign;
    occlSignPost = datastructPost(i).occlSign;

    imgFullResMnPre = datastructPre(i).imgFullResMn;
    imgOcclResMnPre = datastructPre(i).imgOcclResMn;
    imgFullResMnPost = datastructPost(i).imgFullResMn;
    imgOcclResMnPost = datastructPost(i).imgOcclResMn;





%     % include neuron if if has significant response to either full or occl
%     hValCritPre = sum(hValFullPre|hValOcclPre)>0;
%     hValCritPost = sum(hValFullPost|hValOcclPost)>0;

    % calculate latency based on average trace of significant neurons, average
    % neurons per mouse
    latPreFull(i) = nanmean(calcOnsetLatency(squeeze(mean(imgFullResMnPre,2)), vecAx, thres, start, finish));
    latPreOccl(i) = nanmean(calcOnsetLatency(squeeze(mean(imgOcclResMnPre,2)), vecAx, thres, start, finish));
    latPostFull(i) = nanmean(calcOnsetLatency(squeeze(mean(imgFullResMnPost,2)), vecAx, thres, start, finish));
    latPostOccl(i) = nanmean(calcOnsetLatency(squeeze(mean(imgOcclResMnPost,2)), vecAx, thres, start, finish));
%     latPreFull(i) = mean(calcOnsetLatency(fullSignPre, vecAx, thres, start, finish));
%     latPreOccl(i) = mean(calcOnsetLatency(occlSignPre, vecAx, thres, start, finish));
%     latPostFull(i) = mean(calcOnsetLatency(fullSignPost, vecAx, thres, start, finish));
%     latPostOccl(i) = mean(calcOnsetLatency(occlSignPost, vecAx, thres, start, finish));


    postDataFull = [postDataFull; calcOnsetLatency(squeeze(mean(imgFullResMnPost,2)), vecAx, thres, start, finish)];
    postDataOccl = [postDataOccl; calcOnsetLatency(squeeze(mean(imgOcclResMnPost,2)), vecAx, thres, start, finish)];
    mouseIDPost = [mouseIDPost zeros(1,size(imgFullResMnPost,3))+i];


end



% per mouse
figure('Position', [968   274   534   696])
hold on
bar(1, mean(latPreFull), 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2)
bar(2, mean(latPostFull), 'FaceColor', col2, 'EdgeColor', 'k', 'LineWidth', 2)
plot(1:2,[latPreFull,latPostFull], 'Color', [0 0 0 0.5], 'LineWidth', 1);
% scatter(ones(length(latPreFull),1), latPreFull, 'LineWidth', 1, 'markerfacecolor',col1,'markeredgecolor','k')
% scatter(ones(length(latPostFull),1)+1, latPostFull, 'LineWidth', 1, 'markerfacecolor',col2,'markeredgecolor','k')
bar(4, mean(latPreOccl), 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2)
bar(5, mean(latPostOccl), 'FaceColor', col2, 'EdgeColor', 'k', 'LineWidth', 2)
plot(4:5,[latPreOccl,latPostOccl], 'Color', [0 0 0 0.5], 'LineWidth', 1);
% scatter(ones(length(latPreOccl),1)+3, latPreOccl, 'LineWidth', 1, 'markerfacecolor',col1,'markeredgecolor','k')
% scatter(ones(length(latPostOccl),1)+4, latPostOccl, 'LineWidth', 1, 'markerfacecolor',col2,'markeredgecolor','k')
xticks([1 2 4 5])
set(gca, 'XTickLabels', [])
xlim([0 6])
ylabel('Mean onset latency (s)')
title('Mean response onset latency')
% figClean

% per mouse
figure('Position', [968   274   534   696])
hold on
bar(1, mean(latPostFull), 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2)
bar(2, mean(latPostOccl), 'FaceColor', 'w', 'EdgeColor', 'r', 'LineWidth', 2)
plot(1:2,[latPostFull,latPostOccl], 'Color', [0 0 0 0.5], 'LineWidth', 1);
xticks([1 2])
set(gca, 'XTickLabels', [])
xlim([0 3])
ylabel('Mean onset latency (s)')
title('Full vs Occl post training')
% figClean


figure('Position',[1188         578         342         432])
bar([1 2],[nanmean(postDataFull) nanmean(postDataOccl)] , 'FaceColor', 'w', 'EdgeColor', 'k', 'FaceAlpha', 0.6, 'LineWidth', 2)                
hold on
er = errorbar([1 2],[nanmean(postDataFull) nanmean(postDataOccl)],[0 0],[nanstd(postDataFull,[],1) nanstd(postDataOccl,[],1)]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; title('Average response per condition')
xticks([1 2]), xticklabels({'Full', 'Occl'}), ylabel('Mean onset latency (s)')
title('Full vs Occl post training')
% figClean

xtickangle(45)



% full LMEM
postData = cat(1, postDataFull,postDataOccl);
mouseID = categorical(cat(2, mouseIDPost,mouseIDPost))';
condition = categorical(cat(1, ones(length(mouseIDPost),1),ones(length(mouseIDPost),1)+1));
tblPost = table(postData, mouseID, condition);
lmePost = fitlme(tblPost, 'postData ~ condition + (1|mouseID)');
statsPost = anova(lmePost,'DFMethod','Satterthwaite')


%% Clustering
fullSignPopPre = datastructPre(1).fullSign;
occlSignPopPre = datastructPre(1).occlSign;
fullSignPopPost = datastructPost(1).fullSign;
occlSignPopPost = datastructPost(1).occlSign;
matDataPopPre = datastructPre(1).matData;
matDataPopPost = datastructPost(1).matData;
imgFullResMnPopPre = datastructPre(1).imgFullResMn;
imgOcclResMnPopPre = datastructPre(1).imgOcclResMn;
imgFullResMnPopPost = datastructPost(1).imgFullResMn;
imgOcclResMnPopPost = datastructPost(1).imgOcclResMn;


for i = 2:nfiles
    fullSignPopPre = cat(2, fullSignPopPre, datastructPre(i).fullSign);
    occlSignPopPre = cat(2, occlSignPopPre, datastructPre(i).occlSign);
    fullSignPopPost = cat(2, fullSignPopPost, datastructPre(i).fullSign);
    occlSignPopPost = cat(2, occlSignPopPost, datastructPre(i).occlSign);
    matDataPopPre = cat(2, matDataPopPre, datastructPre(i).matData);
    matDataPopPost = cat(2, matDataPopPost, datastructPost(i).matData); 

    imgFullResMnPopPre = cat(3, imgFullResMnPopPre, datastructPre(i).imgFullResMn);  
    imgOcclResMnPopPre = cat(3, imgOcclResMnPopPre, datastructPre(i).imgOcclResMn);  
    imgFullResMnPopPost = cat(3, imgFullResMnPopPost, datastructPost(i).imgFullResMn);  
    imgOcclResMnPopPost = cat(3, imgOcclResMnPopPost, datastructPost(i).imgOcclResMn);  
end

fullPre = squeeze(mean(imgFullResMnPopPre,2))-mean(squeeze(mean(imgFullResMnPopPre(vecAxSp,:,:),2)));
occlPre = squeeze(mean(imgOcclResMnPopPre,2))-mean(squeeze(mean(imgOcclResMnPopPre(vecAxSp,:,:),2)));
fullPost = squeeze(mean(imgFullResMnPopPost,2))-mean(squeeze(mean(imgFullResMnPopPost(vecAxSp,:,:),2)));
occlPost = squeeze(mean(imgOcclResMnPopPost,2))-mean(squeeze(mean(imgOcclResMnPopPost(vecAxSp,:,:),2)));

tracePre = reshape(cat(2, mean(imgFullResMnPopPre(vecAxSt,:,:),2), mean(imgOcclResMnPopPre(vecAxSt,:,:),2)),[],size(imgFullResMnPopPre,3));
tracePost = reshape(cat(2, mean(imgFullResMnPopPost(vecAxSt,:,:),2), mean(imgOcclResMnPopPost(vecAxSt,:,:),2)),[],size(imgFullResMnPopPost,3));

col = ['b', 'r', 'y'];

clusterOn = tracePre;
TimeCourses = normalize(clusterOn, 1, 'range')'; % 'norm' or 'zscore' or 'range'
Time = 1:size(TimeCourses, 2);
Linkages = linkage(TimeCourses, 'Ward');
NumberOfClusters = 3;
TargetOrder = 1:NumberOfClusters;
[~, ClusterNumbers] = plot_task_cluster_dendrogram(NumberOfClusters, Linkages, TimeCourses, TargetOrder, Time, 'range', 'Activity (range)');
idx = ClusterNumbers;

fullPreZs = normalize(fullPre, 1, 'range');
occlPreZs = normalize(occlPre, 1, 'range');

figure('Position', [517         547        1086         380])
for i = 1:length(unique(idx))
    cluster = idx==i; hold on
    s(1) = subplot(1,2,1);
    shadedErrorBar(vecAx,mean(fullPreZs(:,cluster),2)...
        ,std(fullPreZs(:,cluster),0,2)/sqrt(size(fullPreZs(:,cluster),2)), 'lineProps', col(i));
    title('Fullscreen')
    s(2) = subplot(1,2,2);
    shadedErrorBar(vecAx,mean(occlPreZs(:,cluster),2)...
        ,std(occlPreZs(:,cluster),0,2)/sqrt(size(occlPreZs(:,cluster),2)), 'lineProps', col(i));
    title('Occluded')
end
xlabel('Time (s)')
xline(0, 'Color', 'black', 'LineStyle','--', 'LineWidth', 2)
set(gca,'TickDir','out');
mn = min(min(s.YLim)); 
mx = max(max(s.YLim));
s([1 2]).YLim = [mn mx]; 
% s(2).YLim = [mn mx];


clusterOn = tracePost;
TimeCourses = normalize(clusterOn, 1, 'range')'; % 'norm' or 'zscore' or 'range'
Time = 1:size(TimeCourses, 2);
Linkages = linkage(TimeCourses, 'Ward');
NumberOfClusters = 3;
TargetOrder = 1:NumberOfClusters;
[~, ClusterNumbers] = plot_task_cluster_dendrogram(NumberOfClusters, Linkages, TimeCourses, TargetOrder, Time, 'range', 'Activity (range)');
idx = ClusterNumbers;

fullPostZs = normalize(fullPost, 1, 'range');
occlPostZs = normalize(occlPost, 1, 'range');

figure('Position', [517         547        1086         380])
for i = 1:length(unique(idx))
    cluster = idx==i; hold on
    s(1) = subplot(1,2,1);
    shadedErrorBar(vecAx,mean(fullPostZs(:,cluster),2)...
        ,std(fullPostZs(:,cluster),0,2)/sqrt(size(fullPostZs(:,cluster),2)), 'lineProps', col(i));
    title('Fullscreen')
    s(2) = subplot(1,2,2);
    shadedErrorBar(vecAx,mean(occlPostZs(:,cluster),2)...
        ,std(occlPostZs(:,cluster),0,2)/sqrt(size(occlPostZs(:,cluster),2)), 'lineProps', col(i));
    title('Occluded')
end
xlabel('Time (s)')
xline(0, 'Color', 'black', 'LineStyle','--', 'LineWidth', 2)
set(gca,'TickDir','out');
mn = min(min(s.YLim)); 
mx = max(max(s.YLim));
s([1 2]).YLim = [mn mx]; 
% s(2).YLim = [mn mx];


%% DECODING
doDecoding = 1;
doPlotting = 1;
nReps = 500;
nBoots = 1000;
famIdx = [1 2 4 5];
novIdx = [3 6];
trialTypes = matTrialTypesIncl;

matDataPopPre = datastructPre(1).matData;
matDataPopPost = datastructPost(1).matData;

for i = 2:nfiles
    matDataPopPre = cat(2, matDataPopPre, datastructPre(i).matData);
    matDataPopPost = cat(2, matDataPopPost, datastructPost(i).matData);
end

idx = famIdx; % which images to include? famidx/novidx?
ix = 1:6;
ix(idx) = [];
for i = 1:length(ix)
    rmv = trialTypes(1,:)==ix(i);
    trialTypes(:,rmv)=[];
    matDataPopPre(rmv,:)=[];
    matDataPopPost(rmv,:)=[];
end

if doDecoding
    % Pre
%     [pFFPre, pOOPre, pFOPre, pOFPre,pFFpermPre, pOOpermPre,pFOpermPre, pOFpermPre, dPredictFullPre, dPredictOcclPre]...
%         = doMuckliDecodingLDA(matDataPopPre, matTrialTypesIncl, trainFrac, nReps, nBoots, 0);
% 
%     % Post
%     [pFFPost, pOOPost, pFOPost, pOFPost,pFFpermPost, pOOpermPost,pFOpermPost, pOFpermPost, dPredictFullPost, dPredictOcclPost]...
%         = doMuckliDecodingLDA(matDataPopPost, matTrialTypesIncl, trainFrac, nReps, nBoots, 0);

    [pFFPre, pOOPre, pFOPre, pOFPre,pFFpermPre, pOOpermPre,pFOpermPre, pOFpermPre,cMatFFPre, cMatOOPre, cMatFOPre, cMatOFPre,...
        cMatFFpermPre, cMatOOpermPre, cMatFOpermPre, cMatOFpermPre, dPredictFullPre, dPredictOcclPre]...
        = doMuckliDecodingLDAblock2(matDataPopPre, trialTypes, trainFrac, nReps, nBoots, 0);

    % Post
    [pFFPost, pOOPost, pFOPost, pOFPost,pFFpermPost, pOOpermPost,pFOpermPost, pOFpermPost, cMatFFPost, cMatOOPost, cMatFOPost, cMatOFPost,...
        cMatFFpermPost, cMatOOpermPost, cMatFOpermPost, cMatOFpermPost,dPredictFullPost, dPredictOcclPost]...
        = doMuckliDecodingLDAblock2(matDataPopPost, trialTypes, trainFrac, nReps, nBoots, 0);
end

if doPlotting
    % PLOTTING
    figure('Position', [170   400   560   420])
    histogram(pOOPre, length(unique(pOOPre)), 'Normalization', 'Probability', 'FaceColor', col1, 'EdgeColor', col1, 'LineWidth', 2);
    hold on
    histogram(pOOPost, length(unique(pOOPost)), 'Normalization', 'Probability', 'FaceColor', col2, 'EdgeColor', col2, 'LineWidth', 2);
    xlim([0 100])
    xline(mean(pOOPre), 'Color', col1, 'LineWidth', 2), xline(mean(pOOPost), 'Color', col2, 'LineWidth', 2)
    xlabel('Decoding accuracy')
    ylabel('Fraction of decoding runs')
    % figClean
    figure('Position', [749   395   560   420])
    h1 = cdfplot(pOOPre);
    hold on
    h2 = cdfplot(pOOPost);
    xlim([0 100])
    xlabel('Decoding accuracy')
    ylabel('Cumulative density function')
    set(h1,'LineWidth',2, 'Color', col1)
    set(h2,'LineWidth',2, 'Color', col2)
    box off, grid off
    % figClean

    % permutation test - not chronically matched (for chronic matching you need to pair it, so take
    % paired difference between each of the 500 points pre vs post, average
    % those for each run, take those 1000 runs as your permutation data.
    realDiff = mean(pOOPost)-mean(pOOPre);
    nPerms = 1000;
    permDist = zeros(nPerms,1);
    M = zeros(length(pOOPre)*2,2);
    M(:,1) = cat(1, pOOPre, pOOPost);
    M(1:length(pOOPre),2)=1;
    M(length(pOOPre)+1:end,2)=2;
    for perm = 1:nPerms
        Mperm = M(randperm(size(M,1)),2);
        permDist(perm) = mean(M(Mperm==2,1))-mean(M(Mperm==1,1));
    end

    % plot results permutation test
    figure('Position', [1315         397         559         419])
    histogram(permDist, 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(realDiff, 'r', 'LineWidth', 2);
    pval = (sum(permDist>realDiff))/nPerms;
    title(sprintf('p = %.3f', pval)), xlabel('Decoding difference (post-pre)'), ylabel('Relative count')
    legend({'Permutation difference', 'Real difference'})
    legend boxoff
    % figClean

    % weights from decoding
    figure
    subplot(2,2,1), histogram(mean(dPredictOcclPre,2))
    subplot(2,2,2), histogram(mean(dPredictOcclPost,2))
    subplot(2,2,3), histogram(mean(dPredictFullPre,2))
    subplot(2,2,4), histogram(mean(dPredictFullPost,2))

    pFFPreMn = mean(pFFPre);
    pFFPreSEM = std(pFFPre);
    pFFPostMn = mean(pFFPost);
    pFFPostSEM = std(pFFPost);
    pOOPreMn = mean(pOOPre);
    pOOPreSEM = std(pOOPre);
    pOOPostMn = mean(pOOPost);
    pOOPostSEM = std(pOOPost);
    pFOPreMn = mean(pFOPre);
    pFOPreSEM = std(pFOPre);
    pFOPostMn = mean(pFOPost);
    pFOPostSEM = std(pFOPost);
    pOFPreMn = mean(pOFPre);
    pOFPreSEM = std(pOFPre);
    pOFPostMn = mean(pOFPost);
    pOFPostSEM = std(pOFPost);
    

    offset = [-0.33 0 0.33];
    figure('Position', [490   146   754   766])
    bar([1 2 4 5 7 8 10 11],[pFFPreMn pFFPostMn pOOPreMn pOOPostMn pFOPreMn pFOPostMn pOFPreMn pOFPostMn], 'FaceColor', 'w', 'EdgeColor', 'k', 'FaceAlpha', 0.6, 'LineWidth', 2)
    hold on
    er = errorbar([1 2 4 5 7 8 10 11],[pFFPreMn pFFPostMn pOOPreMn pOOPostMn pFOPreMn pFOPostMn pOFPreMn pOFPostMn]...
        ,[0 0 0 0 0 0 0 0],[pFFPreSEM pFFPostSEM pOOPreSEM pOOPostSEM pFOPreSEM pFOPostSEM pOFPreSEM pOFPostSEM]);
    er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2;
%     plot([1+offset 2+offset 4+offset 5+offset 7+offset 8+offset 10+offset 11+offset], ...
%         [mean(pFFpermPre) mean(pFFpermPost) mean(pOOpermPre) mean(pOOpermPost) ...
%         mean(pFOpermPre) mean(pFOpermPost) mean(pOFpermPre) mean(pOFpermPost)],'Color',[0 0 0],'LineWidth',2)
    xticks([1 2 4 5 7 8 10 11]), xticklabels({'F-F', 'F-F', 'O-O', 'O-O', 'F-O', 'F-O', 'O-F', 'O-F'}), ylabel('MM response post - pre'),% figClean
    title('Decoding accuracy')
    ylim([0 120])
    yline(100/length(idx))

    if save_fig
%         func_save_fig('L23_decodingBars')
        func_save_fig('L5_decodingBars_4imgs')
    end

    % Permutation plots
    figure('Position', [301   129   453   695])
    hold on
    subplot(4,2,1)
    histogram(pFFpermPre, length(unique(pFFpermPre)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pFFPreMn, 'r', 'LineWidth', 2);
%     pval = (sum(pFFpermPre>pFFPreMn))/nBoots;
    pval = paretoEst(pFFpermPre/100, pFFPreMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100])
    %     legend({'Permutation data', 'Real data'})
    ylabel('Fraction of runs')
    set(gca, 'LineWidth', 2, 'FontSize', 12)
    box off
    %     figure('Position', [   211   201   540   844])
    subplot(4,2,3)
    histogram(pOOpermPre, length(unique(pOOpermPre)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pOOPreMn, 'r', 'LineWidth', 2);
%     pval = (sum(pOOpermPre>pOOPreMn))/nBoots;
    pval = paretoEst(pOOpermPre/100, pOOPreMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100])
    %     legend({'Permutation data', 'Real data'})
    ylabel('Fraction of runs')
    set(gca, 'LineWidth', 2, 'FontSize', 12)
    box off

    %     figure('Position', [   211   201   540   844])
    subplot(4,2,5)
    histogram(pFOpermPre, length(unique(pFOpermPre)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pFOPreMn, 'r', 'LineWidth', 2);
%     pval = (sum(pFOpermPre>pFOPreMn))/nBoots;
    pval = paretoEst(pFOpermPre/100, pFOPreMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100])
    %     legend({'Permutation data', 'Real data'})
    ylabel('Fraction of runs')
    set(gca, 'LineWidth', 2, 'FontSize', 12)
    box off
    %     figure('Position', [   211   201   540   844])
    subplot(4,2,7)
    histogram(pOFpermPre, length(unique(pOFpermPre)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pOFPreMn, 'r', 'LineWidth', 2);
%     pval = (sum(pOFpermPre>pOFPreMn))/nBoots;
    pval = paretoEst(pOFpermPre/100, pOFPreMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100])
    xlabel('Decoding accuracy')
    %     legend({'Permutation data', 'Real data'})
    ylabel('Fraction of runs')
    set(gca, 'LineWidth', 2, 'FontSize', 12)
    box off

    %     figure('Position', [   211   201   540   844])
    subplot(4,2,2)
    histogram(pFFpermPost, length(unique(pFFpermPost)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pFFPostMn, 'r', 'LineWidth', 2);
%     pval = (sum(pFFpermPost>pFFPostMn))/nBoots;
    pval = paretoEst(pFFpermPost/100, pFFPostMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100])
    %     legend({'Permutation data', 'Real data'})
    set(gca, 'LineWidth', 2, 'FontSize', 12)
    box off
    %     figure('Position', [   211   201   540   844])
    subplot(4,2,4)
    histogram(pOOpermPost, length(unique(pOOpermPost)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pOOPostMn, 'r', 'LineWidth', 2);
%     pval = (sum(pOOpermPost>pOOPostMn))/nBoots;
    pval = paretoEst(pOOpermPost/100, pOOPostMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100])
    %     legend({'Permutation data', 'Real data'})
    set(gca, 'LineWidth', 2, 'FontSize', 12)
    box off

    %     figure('Position', [   211   201   540   844])
    subplot(4,2,6)
    histogram(pFOpermPost, length(unique(pFOpermPost)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pFOPostMn, 'r', 'LineWidth', 2);
%     pval = (sum(pFOpermPost>pFOPostMn))/nBoots;
    pval = paretoEst(pFOpermPost/100, pFOPostMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100])
    %     legend({'Permutation data', 'Real data'})
    set(gca, 'LineWidth', 2, 'FontSize', 12)
    box off
    %     figure('Position', [   211   201   540   844])
    subplot(4,2,8)
    histogram(pOFpermPost, length(unique(pOFpermPost)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pOFPostMn, 'r', 'LineWidth', 2);
%     pval = (sum(pOFpermPost>pOFPostMn))/nBoots;
    pval = paretoEst(pOFpermPost/100, pOFPostMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100])
    xlabel('Decoding accuracy')
    %     legend({'Permutation data', 'Real data'})
    set(gca, 'LineWidth', 2, 'FontSize', 12)
    box off

    if save_fig
        func_save_fig('L23_decodingHists')
        func_save_fig('L5_decodingHists_4imgs')
    end


    cMatFFPreSum = sum(cMatFFPre,3);
    cMatFFPreSum = cMatFFPreSum./sum(cMatFFPreSum,1)*100;
    cMatOOPreSum = sum(cMatOOPre,3);
    cMatOOPreSum = cMatOOPreSum./sum(cMatOOPreSum,1)*100;
    cMatFOPreSum = sum(cMatFOPre,3);
    cMatFOPreSum = cMatFOPreSum./sum(cMatFOPreSum,1)*100;
    cMatOFPreSum = sum(cMatOFPre,3);
    cMatOFPreSum = cMatOFPreSum./sum(cMatOFPreSum,1)*100;
    cMatFFpermPreSum = sum(cMatFFpermPre,3);
    cMatFFpermPreSum = cMatFFpermPreSum./sum(cMatFFpermPreSum,1)*100;
    cMatOOpermPreSum = sum(cMatOOpermPre,3);
    cMatOOpermPreSum = cMatOOpermPreSum./sum(cMatOOpermPreSum,1)*100;
    cMatFOpermPreSum = sum(cMatFOpermPre,3);
    cMatFOpermPreSum = cMatFOpermPreSum./sum(cMatFOpermPreSum,1)*100;
    cMatOFpermPreSum = sum(cMatOFpermPre,3);
    cMatOFpermPreSum = cMatOFpermPreSum./sum(cMatOFpermPreSum,1)*100;

    cMatFFPostSum = sum(cMatFFPost,3);
    cMatFFPostSum = cMatFFPostSum./sum(cMatFFPostSum,1)*100;
    cMatOOPostSum = sum(cMatOOPost,3);
    cMatOOPostSum = cMatOOPostSum./sum(cMatOOPostSum,1)*100;
    cMatFOPostSum = sum(cMatFOPost,3);
    cMatFOPostSum = cMatFOPostSum./sum(cMatFOPostSum,1)*100;
    cMatOFPostSum = sum(cMatOFPost,3);
    cMatOFPostSum = cMatOFPostSum./sum(cMatOFPostSum,1)*100;
    cMatFFpermPostSum = sum(cMatFFpermPost,3);
    cMatFFpermPostSum = cMatFFpermPostSum./sum(cMatFFpermPostSum,1)*100;
    cMatOOpermPostSum = sum(cMatOOpermPost,3);
    cMatOOpermPostSum = cMatOOpermPostSum./sum(cMatOOpermPostSum,1)*100;
    cMatFOpermPostSum = sum(cMatFOpermPost,3);
    cMatFOpermPostSum = cMatFOpermPostSum./sum(cMatFOpermPostSum,1)*100;
    cMatOFpermPostSum = sum(cMatOFpermPost,3);
    cMatOFpermPostSum = cMatOFpermPostSum./sum(cMatOFpermPostSum,1)*100;

    figure('Position', [380         401        1123         454])
    s(1) = subplot(2,4,1);
    imagesc(cMatFFPreSum), xlabel('Predicted'), ylabel('Shown'), title('Pre')
    s(2) = subplot(2,4,2);
    imagesc(cMatOOPreSum)
    s(3) = subplot(2,4,3);
    imagesc(cMatFOPreSum)
    s(4) = subplot(2,4,4);
    imagesc(cMatOFPreSum)
    s(5) = subplot(2,4,5);
    imagesc(cMatFFpermPreSum)
    s(6) = subplot(2,4,6);
    imagesc(cMatOOpermPreSum)
    s(7) = subplot(2,4,7);
    imagesc(cMatFOpermPreSum)
    s(8) = subplot(2,4,8);
    imagesc(cMatOFpermPreSum)
    set(s, 'CLim', [0, 100]);    

    figure('Position', [380         401        1123         454])
    s(1) = subplot(2,4,1);
    imagesc(cMatFFPostSum), xlabel('Predicted'), ylabel('Shown'), title('Post')
    s(2) = subplot(2,4,2);
    imagesc(cMatOOPostSum)
    s(3) = subplot(2,4,3);
    imagesc(cMatFOPostSum)
    s(4) = subplot(2,4,4);
    imagesc(cMatOFPostSum)
    s(5) = subplot(2,4,5);
    imagesc(cMatFFpermPostSum)
    s(6) = subplot(2,4,6);
    imagesc(cMatOOpermPostSum)
    s(7) = subplot(2,4,7);
    imagesc(cMatFOpermPostSum)
    s(8) = subplot(2,4,8);
    imagesc(cMatOFpermPostSum)
    set(s, 'CLim', [0, 100]); 

    figure('Position', [454         178        1027         650]) 
    subplot(2,2,1)
    scatter(mean(dPredictFullPre,2), mean(dPredictFullPost,2), 10, 'k', 'filled')
    refline(1), xlabel('delta pre'), ylabel('delta post'),title('Full')
    subplot(2,2,2)
    scatter(mean(dPredictOcclPre,2), mean(dPredictOcclPost,2), 10, 'k', 'filled')
    refline(1), xlabel('delta pre'), ylabel('delta post'),title('Occl')
    subplot(2,2,3)
    scatter(mean(dPredictFullPre,2), mean(dPredictOcclPre,2), 10, 'k', 'filled')
    refline(1), xlabel('delta full'), ylabel('delta occl'),title('Pre')
    subplot(2,2,4)
    scatter(mean(dPredictFullPost,2), mean(dPredictOcclPost,2), 10, 'k', 'filled')
    refline(1), xlabel('delta full'), ylabel('delta occl'),title('Post')
    
end

%% decoding on subpopulation of neurons
doDecoding = 1;
doPlotting = 1;
nReps = 500;
nBoots = 1000;
famIdx = [1 2 4 5];
novIdx = [3 6];
trialTypes = matTrialTypesIncl;

matDataPopPre = datastructPre(1).matData;
matDataPopPost = datastructPost(1).matData;

for i = 2:nfiles
    matDataPopPre = cat(2, matDataPopPre, datastructPre(i).matData);
    matDataPopPost = cat(2, matDataPopPost, datastructPost(i).matData);
end

idx = famIdx; % which images to include? famidx/novidx?
ix = 1:6;
ix(idx) = [];
for i = 1:length(ix)
    rmv = trialTypes(1,:)==ix(i);
    trialTypes(:,rmv)=[];
    matDataPopPre(rmv,:)=[];
    matDataPopPost(rmv,:)=[];
end

% ix = scatOcclFamPopPost>0.5 & scatFullFamPopPost<0.2;
% ix = scatOcclFamPopPost>0.2;
ix = scatFullFamPopPre>0.5;

matDataPopPre = matDataPopPre(:,ix);
matDataPopPost = matDataPopPost(:,ix);

if doDecoding
    % Pre
%     [pFFPre, pOOPre, pFOPre, pOFPre,pFFpermPre, pOOpermPre,pFOpermPre, pOFpermPre, dPredictFullPre, dPredictOcclPre]...
%         = doMuckliDecodingLDA(matDataPopPre, matTrialTypesIncl, trainFrac, nReps, nBoots, 0);
% 
%     % Post
%     [pFFPost, pOOPost, pFOPost, pOFPost,pFFpermPost, pOOpermPost,pFOpermPost, pOFpermPost, dPredictFullPost, dPredictOcclPost]...
%         = doMuckliDecodingLDA(matDataPopPost, matTrialTypesIncl, trainFrac, nReps, nBoots, 0);

    [pFFPre, pOOPre, pFOPre, pOFPre,pFFpermPre, pOOpermPre,pFOpermPre, pOFpermPre,cMatFFPre, cMatOOPre, cMatFOPre, cMatOFPre,...
        cMatFFpermPre, cMatOOpermPre, cMatFOpermPre, cMatOFpermPre, dPredictFullPre, dPredictOcclPre]...
        = doMuckliDecodingLDACrossDecodingRevisions(matDataPopPre, matDataPopPost, trialTypes, trainFrac, nReps, nBoots, 0);

% %         Post
%         [pFFPost, pOOPost, pFOPost, pOFPost,pFFpermPost, pOOpermPost,pFOpermPost, pOFpermPost, cMatFFPost, cMatOOPost, cMatFOPost, cMatOFPost,...
%             cMatFFpermPost, cMatOOpermPost, cMatFOpermPost, cMatOFpermPost,dPredictFullPost, dPredictOcclPost]...
%             = doMuckliDecodingLDACrossDecodingRevisions(matDataPopPre, matDataPopPost, trialTypes, trainFrac, nReps, nBoots, 0);
end

if doPlotting
    % PLOTTING
    figure('Position', [170   400   560   420])
    histogram(pOOPre, length(unique(pOOPre)), 'Normalization', 'Probability', 'FaceColor', col1, 'EdgeColor', col1, 'LineWidth', 2);
    hold on
    histogram(pOOPost, length(unique(pOOPost)), 'Normalization', 'Probability', 'FaceColor', col2, 'EdgeColor', col2, 'LineWidth', 2);
    xlim([0 100])
    xline(mean(pOOPre), 'Color', col1, 'LineWidth', 2), xline(mean(pOOPost), 'Color', col2, 'LineWidth', 2)
    xlabel('Decoding accuracy')
    ylabel('Fraction of decoding runs')
    % figClean
    figure('Position', [749   395   560   420])
    h1 = cdfplot(pOOPre);
    hold on
    h2 = cdfplot(pOOPost);
    xlim([0 100])
    xlabel('Decoding accuracy')
    ylabel('Cumulative density function')
    set(h1,'LineWidth',2, 'Color', col1)
    set(h2,'LineWidth',2, 'Color', col2)
    box off, grid off
    % figClean

    % permutation test - not chronically matched (for chronic matching you need to pair it, so take
    % paired difference between each of the 500 points pre vs post, average
    % those for each run, take those 1000 runs as your permutation data.
    realDiff = mean(pOOPost)-mean(pOOPre);
    nPerms = 1000;
    permDist = zeros(nPerms,1);
    M = zeros(length(pOOPre)*2,2);
    M(:,1) = cat(1, pOOPre, pOOPost);
    M(1:length(pOOPre),2)=1;
    M(length(pOOPre)+1:end,2)=2;
    for perm = 1:nPerms
        Mperm = M(randperm(size(M,1)),2);
        permDist(perm) = mean(M(Mperm==2,1))-mean(M(Mperm==1,1));
    end

    % plot results permutation test
    figure('Position', [1315         397         559         419])
    histogram(permDist, 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(realDiff, 'r', 'LineWidth', 2);
    pval = (sum(permDist>realDiff))/nPerms;
    title(sprintf('p = %.3f', pval)), xlabel('Decoding difference (post-pre)'), ylabel('Relative count')
    legend({'Permutation difference', 'Real difference'})
    legend boxoff
    % figClean

    % weights from decoding
    figure
    subplot(2,2,1), histogram(mean(dPredictOcclPre,2))
    subplot(2,2,2), histogram(mean(dPredictOcclPost,2))
    subplot(2,2,3), histogram(mean(dPredictFullPre,2))
    subplot(2,2,4), histogram(mean(dPredictFullPost,2))

    pFFPreMn = mean(pFFPre);
    pFFPreSEM = std(pFFPre);
    pFFPostMn = mean(pFFPost);
    pFFPostSEM = std(pFFPost);
    pOOPreMn = mean(pOOPre);
    pOOPreSEM = std(pOOPre);
    pOOPostMn = mean(pOOPost);
    pOOPostSEM = std(pOOPost);
    pFOPreMn = mean(pFOPre);
    pFOPreSEM = std(pFOPre);
    pFOPostMn = mean(pFOPost);
    pFOPostSEM = std(pFOPost);
    pOFPreMn = mean(pOFPre);
    pOFPreSEM = std(pOFPre);
    pOFPostMn = mean(pOFPost);
    pOFPostSEM = std(pOFPost);
    

    offset = [-0.33 0 0.33];
    figure('Position', [490   146   754   766])
    bar([1 2 4 5 7 8 10 11],[pFFPreMn pFFPostMn pOOPreMn pOOPostMn pFOPreMn pFOPostMn pOFPreMn pOFPostMn], 'FaceColor', 'w', 'EdgeColor', 'k', 'FaceAlpha', 0.6, 'LineWidth', 2)
    hold on
    er = errorbar([1 2 4 5 7 8 10 11],[pFFPreMn pFFPostMn pOOPreMn pOOPostMn pFOPreMn pFOPostMn pOFPreMn pOFPostMn]...
        ,[0 0 0 0 0 0 0 0],[pFFPreSEM pFFPostSEM pOOPreSEM pOOPostSEM pFOPreSEM pFOPostSEM pOFPreSEM pOFPostSEM]);
    er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2;
%     plot([1+offset 2+offset 4+offset 5+offset 7+offset 8+offset 10+offset 11+offset], ...
%         [mean(pFFpermPre) mean(pFFpermPost) mean(pOOpermPre) mean(pOOpermPost) ...
%         mean(pFOpermPre) mean(pFOpermPost) mean(pOFpermPre) mean(pOFpermPost)],'Color',[0 0 0],'LineWidth',2)
    xticks([1 2 4 5 7 8 10 11]), xticklabels({'F-F', 'F-F', 'O-O', 'O-O', 'F-O', 'F-O', 'O-F', 'O-F'}), ylabel('MM response post - pre'),% figClean
    title('Decoding accuracy')
    ylim([0 120])
    yline(100/length(idx))

    if save_fig
%         func_save_fig('L23_decodingBars')
        func_save_fig('L5_decodingBars_4imgs')
    end

    % Permutation plots
    figure('Position', [301   129   453   695])
    hold on
    subplot(4,2,1)
    histogram(pFFpermPre, length(unique(pFFpermPre)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pFFPreMn, 'r', 'LineWidth', 2);
%     pval = (sum(pFFpermPre>pFFPreMn))/nBoots;
    pval = paretoEst(pFFpermPre/100, pFFPreMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100])
    %     legend({'Permutation data', 'Real data'})
    ylabel('Fraction of runs')
    set(gca, 'LineWidth', 2, 'FontSize', 12)
    box off
    %     figure('Position', [   211   201   540   844])
    subplot(4,2,3)
    histogram(pOOpermPre, length(unique(pOOpermPre)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pOOPreMn, 'r', 'LineWidth', 2);
%     pval = (sum(pOOpermPre>pOOPreMn))/nBoots;
    pval = paretoEst(pOOpermPre/100, pOOPreMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100])
    %     legend({'Permutation data', 'Real data'})
    ylabel('Fraction of runs')
    set(gca, 'LineWidth', 2, 'FontSize', 12)
    box off

    %     figure('Position', [   211   201   540   844])
    subplot(4,2,5)
    histogram(pFOpermPre, length(unique(pFOpermPre)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pFOPreMn, 'r', 'LineWidth', 2);
%     pval = (sum(pFOpermPre>pFOPreMn))/nBoots;
    pval = paretoEst(pFOpermPre/100, pFOPreMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100])
    %     legend({'Permutation data', 'Real data'})
    ylabel('Fraction of runs')
    set(gca, 'LineWidth', 2, 'FontSize', 12)
    box off
    %     figure('Position', [   211   201   540   844])
    subplot(4,2,7)
    histogram(pOFpermPre, length(unique(pOFpermPre)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pOFPreMn, 'r', 'LineWidth', 2);
%     pval = (sum(pOFpermPre>pOFPreMn))/nBoots;
    pval = paretoEst(pOFpermPre/100, pOFPreMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100])
    xlabel('Decoding accuracy')
    %     legend({'Permutation data', 'Real data'})
    ylabel('Fraction of runs')
    set(gca, 'LineWidth', 2, 'FontSize', 12)
    box off

    %     figure('Position', [   211   201   540   844])
    subplot(4,2,2)
    histogram(pFFpermPost, length(unique(pFFpermPost)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pFFPostMn, 'r', 'LineWidth', 2);
%     pval = (sum(pFFpermPost>pFFPostMn))/nBoots;
    pval = paretoEst(pFFpermPost/100, pFFPostMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100])
    %     legend({'Permutation data', 'Real data'})
    set(gca, 'LineWidth', 2, 'FontSize', 12)
    box off
    %     figure('Position', [   211   201   540   844])
    subplot(4,2,4)
    histogram(pOOpermPost, length(unique(pOOpermPost)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pOOPostMn, 'r', 'LineWidth', 2);
%     pval = (sum(pOOpermPost>pOOPostMn))/nBoots;
    pval = paretoEst(pOOpermPost/100, pOOPostMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100])
    %     legend({'Permutation data', 'Real data'})
    set(gca, 'LineWidth', 2, 'FontSize', 12)
    box off

    %     figure('Position', [   211   201   540   844])
    subplot(4,2,6)
    histogram(pFOpermPost, length(unique(pFOpermPost)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pFOPostMn, 'r', 'LineWidth', 2);
%     pval = (sum(pFOpermPost>pFOPostMn))/nBoots;
    pval = paretoEst(pFOpermPost/100, pFOPostMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100])
    %     legend({'Permutation data', 'Real data'})
    set(gca, 'LineWidth', 2, 'FontSize', 12)
    box off
    %     figure('Position', [   211   201   540   844])
    subplot(4,2,8)
    histogram(pOFpermPost, length(unique(pOFpermPost)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pOFPostMn, 'r', 'LineWidth', 2);
%     pval = (sum(pOFpermPost>pOFPostMn))/nBoots;
    pval = paretoEst(pOFpermPost/100, pOFPostMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100])
    xlabel('Decoding accuracy')
    %     legend({'Permutation data', 'Real data'})
    set(gca, 'LineWidth', 2, 'FontSize', 12)
    box off

    if save_fig
        func_save_fig('L23_decodingHists')
        func_save_fig('L5_decodingHists_4imgs')
    end


    cMatFFPreSum = sum(cMatFFPre,3);
    cMatFFPreSum = cMatFFPreSum./sum(cMatFFPreSum,1)*100;
    cMatOOPreSum = sum(cMatOOPre,3);
    cMatOOPreSum = cMatOOPreSum./sum(cMatOOPreSum,1)*100;
    cMatFOPreSum = sum(cMatFOPre,3);
    cMatFOPreSum = cMatFOPreSum./sum(cMatFOPreSum,1)*100;
    cMatOFPreSum = sum(cMatOFPre,3);
    cMatOFPreSum = cMatOFPreSum./sum(cMatOFPreSum,1)*100;
    cMatFFpermPreSum = sum(cMatFFpermPre,3);
    cMatFFpermPreSum = cMatFFpermPreSum./sum(cMatFFpermPreSum,1)*100;
    cMatOOpermPreSum = sum(cMatOOpermPre,3);
    cMatOOpermPreSum = cMatOOpermPreSum./sum(cMatOOpermPreSum,1)*100;
    cMatFOpermPreSum = sum(cMatFOpermPre,3);
    cMatFOpermPreSum = cMatFOpermPreSum./sum(cMatFOpermPreSum,1)*100;
    cMatOFpermPreSum = sum(cMatOFpermPre,3);
    cMatOFpermPreSum = cMatOFpermPreSum./sum(cMatOFpermPreSum,1)*100;

    cMatFFPostSum = sum(cMatFFPost,3);
    cMatFFPostSum = cMatFFPostSum./sum(cMatFFPostSum,1)*100;
    cMatOOPostSum = sum(cMatOOPost,3);
    cMatOOPostSum = cMatOOPostSum./sum(cMatOOPostSum,1)*100;
    cMatFOPostSum = sum(cMatFOPost,3);
    cMatFOPostSum = cMatFOPostSum./sum(cMatFOPostSum,1)*100;
    cMatOFPostSum = sum(cMatOFPost,3);
    cMatOFPostSum = cMatOFPostSum./sum(cMatOFPostSum,1)*100;
    cMatFFpermPostSum = sum(cMatFFpermPost,3);
    cMatFFpermPostSum = cMatFFpermPostSum./sum(cMatFFpermPostSum,1)*100;
    cMatOOpermPostSum = sum(cMatOOpermPost,3);
    cMatOOpermPostSum = cMatOOpermPostSum./sum(cMatOOpermPostSum,1)*100;
    cMatFOpermPostSum = sum(cMatFOpermPost,3);
    cMatFOpermPostSum = cMatFOpermPostSum./sum(cMatFOpermPostSum,1)*100;
    cMatOFpermPostSum = sum(cMatOFpermPost,3);
    cMatOFpermPostSum = cMatOFpermPostSum./sum(cMatOFpermPostSum,1)*100;

    figure('Position', [380         401        1123         454])
    s(1) = subplot(2,4,1);
    imagesc(cMatFFPreSum), xlabel('Predicted'), ylabel('Shown'), title('Pre')
    s(2) = subplot(2,4,2);
    imagesc(cMatOOPreSum)
    s(3) = subplot(2,4,3);
    imagesc(cMatFOPreSum)
    s(4) = subplot(2,4,4);
    imagesc(cMatOFPreSum)
    s(5) = subplot(2,4,5);
    imagesc(cMatFFpermPreSum)
    s(6) = subplot(2,4,6);
    imagesc(cMatOOpermPreSum)
    s(7) = subplot(2,4,7);
    imagesc(cMatFOpermPreSum)
    s(8) = subplot(2,4,8);
    imagesc(cMatOFpermPreSum)
    set(s, 'CLim', [0, 100]);    

    figure('Position', [380         401        1123         454])
    s(1) = subplot(2,4,1);
    imagesc(cMatFFPostSum), xlabel('Predicted'), ylabel('Shown'), title('Post')
    s(2) = subplot(2,4,2);
    imagesc(cMatOOPostSum)
    s(3) = subplot(2,4,3);
    imagesc(cMatFOPostSum)
    s(4) = subplot(2,4,4);
    imagesc(cMatOFPostSum)
    s(5) = subplot(2,4,5);
    imagesc(cMatFFpermPostSum)
    s(6) = subplot(2,4,6);
    imagesc(cMatOOpermPostSum)
    s(7) = subplot(2,4,7);
    imagesc(cMatFOpermPostSum)
    s(8) = subplot(2,4,8);
    imagesc(cMatOFpermPostSum)
    set(s, 'CLim', [0, 100]); 

    figure('Position', [454         178        1027         650]) 
    subplot(2,2,1)
    scatter(mean(dPredictFullPre,2), mean(dPredictFullPost,2), 10, 'k', 'filled')
    refline(1), xlabel('delta pre'), ylabel('delta post'),title('Full')
    subplot(2,2,2)
    scatter(mean(dPredictOcclPre,2), mean(dPredictOcclPost,2), 10, 'k', 'filled')
    refline(1), xlabel('delta pre'), ylabel('delta post'),title('Occl')
    subplot(2,2,3)
    scatter(mean(dPredictFullPre,2), mean(dPredictOcclPre,2), 10, 'k', 'filled')
    refline(1), xlabel('delta full'), ylabel('delta occl'),title('Pre')
    subplot(2,2,4)
    scatter(mean(dPredictFullPost,2), mean(dPredictOcclPost,2), 10, 'k', 'filled')
    refline(1), xlabel('delta full'), ylabel('delta occl'),title('Post')
    
end



%% Locomotion speed

runSpeedMsPre = zeros(nfiles,1);
runSpeedMsPost = zeros(nfiles,1);

for i = 1:nfiles
    runSpeedMsPre(i) = mean(datastructPre(i).info.Run.Speed);
    runSpeedMsPost(i) = mean(datastructPost(i).info.Run.Speed);
end

figure
plot(1:2, [runSpeedMsPre, runSpeedMsPost]')
xlim([0 3])
title('Locomotion speed')


%% RF location and responses/decoding contribution

rfOffDistPrePop = cat(1,[],datastructPre(:).rfOffDist);
rfOnDistPrePop = cat(1,[],datastructPre(:).rfOnDist);
rfOffDistPostPop = cat(1,[],datastructPost(:).rfOffDist);
rfOnDistPostPop = cat(1,[],datastructPost(:).rfOnDist);
rfInclPopPre = logical(cat(1,[],datastructPre(:).rfIncl));
rfInclPopPost = logical(cat(1,[],datastructPost(:).rfIncl));

rfOffDistPrePop = rfOffDistPrePop(rfInclPopPre);
rfOnDistPrePop = rfOnDistPrePop(rfInclPopPre);
rfOffDistPostPop = rfOffDistPostPop(rfInclPopPost);
rfOnDistPostPop = rfOnDistPostPop(rfInclPopPost);

% RF location with responses
figure
subplot(2,4,1)
scatter(rfOffDistPrePop, scatFullPopPre), refline
subplot(2,4,5)
scatter(rfOffDistPrePop, scatOcclPopPre), refline
subplot(2,4,2)
scatter(rfOnDistPrePop, scatFullPopPre), refline
subplot(2,4,6)
scatter(rfOnDistPrePop, scatOcclPopPre), refline
subplot(2,4,3)
scatter(rfOffDistPostPop, scatFullPopPost), refline
subplot(2,4,7)
scatter(rfOffDistPostPop, scatOcclPopPost), refline
subplot(2,4,4)
scatter(rfOnDistPostPop, scatFullPopPost), refline
subplot(2,4,8)
scatter(rfOnDistPostPop, scatOcclPopPost), refline

% RF location with decoding contribution
figure
subplot(2,4,1)
scatter(rfOffDistPrePop, mean(dPredictFullPre,2)), refline
subplot(2,4,5)
scatter(rfOffDistPrePop, mean(dPredictOcclPre,2)), refline
subplot(2,4,2)
scatter(rfOnDistPrePop, mean(dPredictFullPre,2)), refline
subplot(2,4,6)
scatter(rfOnDistPrePop, mean(dPredictOcclPre,2)), refline
subplot(2,4,3)
scatter(rfOffDistPostPop, mean(dPredictFullPost,2)), refline
subplot(2,4,7)
scatter(rfOffDistPostPop, mean(dPredictOcclPost,2)), refline
subplot(2,4,4)
scatter(rfOnDistPostPop, mean(dPredictFullPost,2)), refline
subplot(2,4,8)
scatter(rfOnDistPostPop, mean(dPredictOcclPost,2)), refline

%% MASK clustering
rgbFull = zeros(size(datastructPost(j).info.Mask, 1), size(datastructPost(j).info.Mask, 2), 3, nfiles);
rgbOccl = zeros(size(datastructPost(j).info.Mask, 1), size(datastructPost(j).info.Mask, 2), 3, nfiles);
histvalsOccl = [];
histvalsFull = [];
histvalsAll  = [];
histvalsIncl = [];
direction = -45;

for j = 1:nfiles
    info = datastructPost(j).info;
    hValOccl = datastructPost(j).hValOccl;
    hValFull = datastructPost(j).hValFull;
    rfIncl = datastructPost(j).rfIncl;
    Mask = info.Mask;
    PP = struct();
    PP.Cnt = length(info.rois);
    for i = 1:PP.Cnt
        PP.Con(i).x = info.rois(i).x;
        PP.Con(i).y = info.rois(i).y;
        PP.P(:,i) = [info.rois(i).px'; info.rois(i).py'];
    end
    toInclude = find(rfIncl);
%     toDelete = find(~rfIncl);

%     PP.Con(toDelete) = [];
%     PP.P(:,toDelete) = [];
%     PP.Cnt = size(PP.P,2);
%     Mask(ismember(Mask, toDelete)) = 0;
%     v = unique(Mask(:));
%     v = v(2:end);
%     for i = 1:length(v)
%         if v(i) ~= i
%             Mask(Mask == v(i)) = i;
%         end
%     end

    %%
    % plotIdx = sum(hValOccl)+1;
    idxOccl = (sum(hValOccl)>0&sum(hValFull)==0)+2;
    idxFull = (sum(hValFull)>0&sum(hValOccl)==0)+2;
    
    valsOccl = ones(PP.Cnt, 1);
    valsOccl(toInclude) = idxOccl;
    valsFull = ones(PP.Cnt, 1);
    valsFull(toInclude) = idxFull;

    figure('Position',[640   164   1000   832])
    weights = [];
    colorway = 'index';
    red = [];
    dotSize = [];
    doDots = false;
    doCons = false;
    colors = [0.5 0.5 0.5; 0 1 0; 1 0 0];
    subplot(2,2,1)
    rgbOccl(:,:,:,j) = RoiVal2Img(Mask, PP, valsOccl, weights, colors, colorway, red, dotSize, doDots, doCons);
    title(sprintf('Mouse number: %d. occl', j))

    subplot(2,2,2)
    rgbFull(:,:,:,j) = RoiVal2Img(Mask, PP, valsFull, weights, colors, colorway, red, dotSize, doDots, doCons);
    title('Full screen response')

    Mask = imrotate(info.Mask, direction);

    com = GetRoiCoM(Mask);
    x = com(:,1);

    bins = 0:50:900;

%     subplot(2,2,1)
%     RoiVal2Img(Mask, PP, valsOccl, weights, colors, colorway, red, dotSize, doDots, doCons);
%     subplot(2,2,2)
%     RoiVal2Img(Mask, PP, valsFull, weights, colors, colorway, red, dotSize, doDots, doCons);
    histvalsOccl = [histvalsOccl; x(toInclude(sum(hValOccl)>0&sum(hValFull)==0))];
    histvalsFull = [histvalsFull; x(toInclude(sum(hValFull)>0&sum(hValOccl)==0))];
    histvalsAll  = [histvalsAll; x];
    histvalsIncl = [histvalsIncl; x(toInclude)];

    subplot(2,2,3)
    histogram(x(toInclude(sum(hValOccl)>0)), bins, 'Normalization', 'Probability', 'FaceColor', [1 0 0])
    hold on
    histogram(x(toInclude), bins, 'Normalization', 'Probability', 'FaceColor', [0 1 0])
    histogram(x, bins, 'Normalization', 'Probability', 'FaceColor', [0.8 0.8 0.8], 'FaceAlpha', 0.1)
    xlim([1 size(Mask,2)])
    colorbar

    subplot(2,2,4)
    histogram(x(toInclude(sum(hValFull)>0)), bins, 'Normalization', 'Probability', 'FaceColor', [1 0 0])
    hold on
    histogram(x(toInclude), bins, 'Normalization', 'Probability', 'FaceColor', [0 1 0])
    histogram(x, bins, 'Normalization', 'Probability', 'FaceColor', [0.8 0.8 0.8], 'FaceAlpha', 0.1)
    xlim([1 size(Mask,2)])
    colorbar



end
%

figure('Position',[640   164   1000   832])
subplot(2,2,1)
imagesc(sum(rgbOccl, 4))
subplot(2,2,2)
imagesc(sum(rgbFull, 4))    

subplot(2,2,3)
histogram(histvalsOccl, bins, 'Normalization', 'Probability', 'FaceColor', [1 0 0])
hold on
histogram(histvalsIncl, bins, 'Normalization', 'Probability', 'FaceColor', [0 1 0])
histogram(histvalsAll, bins, 'Normalization', 'Probability', 'FaceColor', [0.8 0.8 0.8], 'FaceAlpha', 0.1)
xlim([1 size(Mask,2)])
colorbar

subplot(2,2,4)
histogram(histvalsFull, bins, 'Normalization', 'Probability', 'FaceColor', [1 0 0])
hold on
histogram(histvalsIncl, bins, 'Normalization', 'Probability', 'FaceColor', [0 1 0])
histogram(histvalsAll, bins, 'Normalization', 'Probability', 'FaceColor', [0.8 0.8 0.8], 'FaceAlpha', 0.1)
xlim([1 size(Mask,2)])
colorbar

[~,pFullOccl] = kstest2(histvalsFull, histvalsOccl)
[~,pFull] = kstest2(histvalsFull, histvalsIncl)
[~,pOccl] = kstest2(histvalsOccl, histvalsIncl)

%% Find spatial direction that matters for ROI properties

direction = -45;
Mask = imrotate(info.Mask, direction);

figure
imagesc(Mask)
com = GetRoiCoM(Mask);
x = com(:,1);

bins = 0:50:900;

figure('Position',[640   164   600   832])
subplot(2,2,1)
RoiVal2Img(Mask, PP, valsOccl, weights, colors, colorway, red, dotSize, doDots, doCons);
subplot(2,2,2)
toInclude = find(rfIncl);
histogram(x(toInclude(sum(hValOccl)>0)), bins, 'Normalization', 'Probability', 'FaceColor', [1 0 0])
hold on
% histogram(x(toInclude(sum(hValFull)>0)), bins, 'Normalization', 'Probability', 'FaceColor', [0 0 1])
histogram(x(toInclude), bins, 'Normalization', 'Probability', 'FaceColor', [0 1 0])
histogram(x, bins, 'Normalization', 'Probability', 'FaceColor', [0.8 0.8 0.8])
xlim([1 size(Mask,2)])
colorbar


%% new selectivity measure based on Poort et al nat neuro

imgFullResPopPre = datastructPre(1).imgFullRes;
imgOcclResPopPre = datastructPre(1).imgOcclRes;
imgFullResPopPost = datastructPost(1).imgFullRes;
imgOcclResPopPost = datastructPost(1).imgOcclRes;

for k = 2:nfiles
    imgFullResPopPre = cat(4, imgFullResPopPre, datastructPre(k).imgFullRes);
    imgOcclResPopPre = cat(4, imgOcclResPopPre, datastructPre(k).imgOcclRes);
    imgFullResPopPost = cat(4, imgFullResPopPost, datastructPost(k).imgFullRes);
    imgOcclResPopPost = cat(4, imgOcclResPopPost, datastructPost(k).imgOcclRes);
end

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


[B, I] = maxk(scatOcclPopPre,round(0.2*length(scatOcclPopPre)));
% [B, I] = maxk(scatFullPopPre,round(0.2*length(scatFullPopPre)));
mean(SIpre(I))
mean(SIpre)


[B, I] = maxk(scatOcclPopPost,round(0.2*length(scatOcclPopPost)));
% [B, I] = maxk(scatFullPopPost,round(0.2*length(scatFullPopPost)));
mean(SIpost(I))
mean(SIpost)


%% plot RFs

rfInclPopPre = datastructPre(1).rfIncl;
rfInclPopPost = datastructPost(1).rfIncl;
for i = 2:nfiles
    rfInclPopPre = cat(1, rfInclPopPre, datastructPre(i).rfIncl);
    rfInclPopPost = cat(1, rfInclPopPost, datastructPost(i).rfIncl);

end

aziOnPopPre = [];
eleOnPopPre = [];
rfszOnPopPre = [];
aziOffPopPre = [];
eleOffPopPre = [];
rfszOffPopPre = [];

aziOnPopPost = [];
eleOnPopPost = [];
rfszOnPopPost = [];
aziOffPopPost = [];
eleOffPopPost = [];
rfszOffPopPost = [];

for k = 1:nfiles
    roisPre = datastructPre(k).info.rois;
    nRoisPre = length(roisPre);
    aziOnPre = zeros(nRoisPre,1);
    eleOnPre = zeros(nRoisPre,1);
    rfszOnPre = zeros(nRoisPre,1);
    aziOffPre = zeros(nRoisPre,1);
    eleOffPre = zeros(nRoisPre,1);
    rfszOffPre = zeros(nRoisPre,1);
    for j = 1:nRoisPre
        aziOnPre(j) = roisPre(j).azi(1);
        eleOnPre(j) = roisPre(j).ele(1);
        rfszOnPre(j) = roisPre(j).rfsz(1);
        aziOffPre(j) = roisPre(j).azi(2);
        eleOffPre(j) = roisPre(j).ele(2);
        rfszOffPre(j) = roisPre(j).rfsz(2);
    end
    aziOnPopPre = [aziOnPopPre; aziOnPre];
    eleOnPopPre = [eleOnPopPre; eleOnPre];
    rfszOnPopPre = [rfszOnPopPre; rfszOnPre];
    aziOffPopPre = [aziOffPopPre; aziOffPre];
    eleOffPopPre = [eleOffPopPre; eleOffPre];
    rfszOffPopPre = [rfszOffPopPre; rfszOffPre];
    

    roisPost = datastructPost(k).info.rois;
    nRoisPost = length(roisPost);
    aziOnPost = zeros(nRoisPost,1);
    eleOnPost = zeros(nRoisPost,1);
    rfszOnPost = zeros(nRoisPost,1);
    aziOffPost = zeros(nRoisPost,1);
    eleOffPost = zeros(nRoisPost,1);
    rfszOffPost = zeros(nRoisPost,1);
    for j = 1:nRoisPost
        aziOnPost(j) = roisPost(j).azi(1);
        eleOnPost(j) = roisPost(j).ele(1);
        rfszOnPost(j) = roisPost(j).rfsz(1);
        aziOffPost(j) = roisPost(j).azi(2);
        eleOffPost(j) = roisPost(j).ele(2);
        rfszOffPost(j) = roisPost(j).rfsz(2);
    end
    aziOnPopPost = [aziOnPopPost; aziOnPost];
    eleOnPopPost = [eleOnPopPost; eleOnPost];
    rfszOnPopPost = [rfszOnPopPost; rfszOnPost];
    aziOffPopPost = [aziOffPopPost; aziOffPost];
    eleOffPopPost = [eleOffPopPost; eleOffPost];
    rfszOffPopPost = [rfszOffPopPost; rfszOffPost];

end


% all RFs
figure('Position', [198         198        1448         786])
subplot(2,2,1)
viscircles([aziOnPopPre, eleOnPopPre], rfszOnPopPre, 'Color', [0.8 0.8 0.8], 'LineWidth', 0.3); 
subplot(2,2,3)
viscircles([aziOffPopPre, eleOffPopPre], rfszOffPopPre, 'Color', [0.8 0.8 0.8], 'LineWidth', 0.3); 
subplot(2,2,2)
viscircles([aziOnPopPost, eleOnPopPost], rfszOnPopPost, 'Color', [0.8 0.8 0.8], 'LineWidth', 0.3); 
subplot(2,2,4)
viscircles([aziOffPopPost, eleOffPopPost], rfszOffPopPost, 'Color', [0.8 0.8 0.8], 'LineWidth', 0.3); 

figure
viscircles([mean(aziOnPopPre), mean(eleOnPopPre)], mean(rfszOnPopPre),'Color', [0 0 0])
hold on
viscircles([mean(aziOnPopPost), mean(eleOnPopPost)], mean(rfszOnPopPost),'Color', [1 0 0])

figure 
viscircles([mean(aziOffPopPre), mean(eleOffPopPre)], mean(rfszOffPopPre),'Color', [0 0 0])
hold on
viscircles([mean(aziOffPopPost), mean(eleOffPopPost)], mean(rfszOffPopPost),'Color', [1 0 0])


% only included RFs
figure('Position', [198         198        1448         786])
subplot(2,2,1)
viscircles([aziOnPopPre(rfInclPopPre), eleOnPopPre(rfInclPopPre)], rfszOnPopPre(rfInclPopPre), 'Color', [0.8 0.8 0.8], 'LineWidth', 0.3); 
subplot(2,2,3)
viscircles([aziOffPopPre(rfInclPopPre), eleOffPopPre(rfInclPopPre)], rfszOffPopPre(rfInclPopPre), 'Color', [0.8 0.8 0.8], 'LineWidth', 0.3); 
subplot(2,2,2)
viscircles([aziOnPopPost(rfInclPopPost), eleOnPopPost(rfInclPopPost)], rfszOnPopPost(rfInclPopPost), 'Color', [0.8 0.8 0.8], 'LineWidth', 0.3); 
subplot(2,2,4)
viscircles([aziOffPopPost(rfInclPopPost), eleOffPopPost(rfInclPopPost)], rfszOffPopPost(rfInclPopPost), 'Color', [0.8 0.8 0.8], 'LineWidth', 0.3); 

figure
viscircles([mean(aziOnPopPre(rfInclPopPre)), mean(eleOnPopPre(rfInclPopPre))], mean(rfszOnPopPre(rfInclPopPre)),'Color', [0 0 0])
hold on
viscircles([mean(aziOnPopPost(rfInclPopPost)), mean(eleOnPopPost(rfInclPopPost))], mean(rfszOnPopPost(rfInclPopPost)),'Color', [1 0 0])

figure 
viscircles([mean(aziOffPopPre(rfInclPopPre)), mean(eleOffPopPre(rfInclPopPre))], mean(rfszOffPopPre(rfInclPopPre)),'Color', [0 0 0])
hold on
viscircles([mean(aziOffPopPost(rfInclPopPost)), mean(eleOffPopPost(rfInclPopPost))], mean(rfszOffPopPost(rfInclPopPost)),'Color', [1 0 0])


%% RF properties of subsets of neurons
rfszOnPopPreIncl = rfszOnPopPre(rfInclPopPre);
rfszOffPopPreIncl = rfszOffPopPre(rfInclPopPre);
rfszOnPopPostIncl = rfszOnPopPost(rfInclPopPost);
rfszOffPopPostIncl = rfszOffPopPost(rfInclPopPost);

figure
% scatter(rfszOnPopPreIncl, scatOcclPopPre)
% scatter(rfszOnPopPreIncl, scatFullPopPre)
subplot(1,2,1)
scatter(rfszOnPopPostIncl, scatFullPopPost)
subplot(1,2,2)
scatter(rfszOnPopPostIncl, scatOcclPopPost)

%% calculate selectivity over images

selecPreFull = zeros(nfiles,1);
selecPreOccl = zeros(nfiles,1);
selecPostFull = zeros(nfiles,1);
selecPostOccl = zeros(nfiles,1);

fullDataPre = [];
occlDataPre = [];
mouseIDPre = [];

fullDataPost = [];
occlDataPost = [];
mouseIDPost = [];


for i = 1:nfiles
    % prepare some data for linear mixed model effect
    fullDataPre = [fullDataPre; selectCalc(squeeze(mean(datastructPre(i).imgFullResMn(vecAxSt,:,:)))-squeeze(mean(datastructPre(i).imgFullResMn(vecAxSp,:,:))))];
    occlDataPre = [occlDataPre; selectCalc(squeeze(mean(datastructPre(i).imgOcclResMn(vecAxSt,:,:)))-squeeze(mean(datastructPre(i).imgOcclResMn(vecAxSp,:,:))))];
    mouseIDPre = [mouseIDPre zeros(1,size(datastructPre(i).imgFullResMn,3))+i];

    fullDataPost = [fullDataPost; selectCalc(squeeze(mean(datastructPost(i).imgFullResMn(vecAxSt,:,:)))-squeeze(mean(datastructPost(i).imgFullResMn(vecAxSp,:,:))))];
    occlDataPost = [occlDataPost; selectCalc(squeeze(mean(datastructPost(i).imgOcclResMn(vecAxSt,:,:)))-squeeze(mean(datastructPost(i).imgOcclResMn(vecAxSp,:,:))))];
    mouseIDPost = [mouseIDPost zeros(1,size(datastructPost(i).imgFullResMn,3))+i];

    selecPreFull(i) = mean(selectCalc(squeeze(mean(datastructPre(i).imgFullResMn(vecAxSt,:,:)))-squeeze(mean(datastructPre(i).imgFullResMn(vecAxSp,:,:)))));
    selecPreOccl(i) = mean(selectCalc(squeeze(mean(datastructPre(i).imgOcclResMn(vecAxSt,:,:)))-squeeze(mean(datastructPre(i).imgOcclResMn(vecAxSp,:,:)))));
    selecPostFull(i) = mean(selectCalc(squeeze(mean(datastructPost(i).imgFullResMn(vecAxSt,:,:)))-squeeze(mean(datastructPost(i).imgFullResMn(vecAxSp,:,:)))));
    selecPostOccl(i) = mean(selectCalc(squeeze(mean(datastructPost(i).imgOcclResMn(vecAxSt,:,:)))-squeeze(mean(datastructPost(i).imgOcclResMn(vecAxSp,:,:)))));
end


% per mouse
figure('Position', [968   274   534   696])
hold on
bar(1, mean(selecPreFull), 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2)
bar(2, mean(selecPostFull), 'FaceColor', col2, 'EdgeColor', 'k', 'LineWidth', 2)
plot(1:2,[selecPreFull,selecPostFull], 'Color', [0 0 0 0.5], 'LineWidth', 1);
% scatter(ones(length(selecPreFull),1), selecPreFull, 'LineWidth', 1, 'markerfacecolor',col1,'markeredgecolor','k')
% scatter(ones(length(selecPostFull),1)+1, selecPostFull, 'LineWidth', 1, 'markerfacecolor',col2,'markeredgecolor','k')
bar(4, mean(selecPreOccl), 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2)
bar(5, mean(selecPostOccl), 'FaceColor', col2, 'EdgeColor', 'k', 'LineWidth', 2)
plot(4:5,[selecPreOccl,selecPostOccl], 'Color', [0 0 0 0.5], 'LineWidth', 1);
% scatter(ones(length(selecPreOccl),1)+3, selecPreOccl, 'LineWidth', 1, 'markerfacecolor',col1,'markeredgecolor','k')
% scatter(ones(length(selecPostOccl),1)+4, selecPostOccl, 'LineWidth', 1, 'markerfacecolor',col2,'markeredgecolor','k')
xticks([1 2 4 5])
set(gca, 'XTickLabels', [])
xlim([0 6])
ylabel('Mean selectivity')
title('Mean selectivity')
% figClean

bins = 0.2:0.01:0.35;

figure
subplot(1,2,1)
histogram(selecPreFull, bins, 'Normalization', 'Probability')
hold on
histogram(selecPostFull, bins, 'Normalization', 'Probability')
subplot(1,2,2)
histogram(selecPreOccl, bins, 'Normalization', 'Probability')
hold on
histogram(selecPostOccl, bins, 'Normalization', 'Probability')


% full LMEM
fullDataPre = cat(1, fullDataPre,fullDataPost);
mouseID = categorical(cat(2, mouseIDPre,mouseIDPost))';
training = categorical(cat(1, ones(length(mouseIDPre),1),ones(length(mouseIDPost),1)+1));
tblFull = table(fullDataPre, mouseID, training);
lmeFull = fitlme(tblFull, 'fullData ~ training + (1|mouseID)');
statsFull = anova(lmeFull,'DFMethod','Satterthwaite')

% occl LMEM
occlDataPre = cat(1, occlDataPre,occlDataPost);
mouseID = categorical(cat(2, mouseIDPre,mouseIDPost))';
training = categorical(cat(1, ones(length(mouseIDPre),1),ones(length(mouseIDPost),1)+1));
tblOccl = table(occlDataPre, mouseID, training);
lmeOccl = fitlme(tblOccl, 'occlData ~ training + (1|mouseID)');
statsOccl = anova(lmeOccl,'DFMethod','Satterthwaite')

%% plot example responses
imgFullResMnPopPre = datastructPre(1).imgFullResMn*100;
imgOcclResMnPopPre = datastructPre(1).imgOcclResMn*100;
imgFullResMnPopPost = datastructPost(1).imgFullResMn*100;
imgOcclResMnPopPost = datastructPost(1).imgOcclResMn*100;


for i = 2:nfiles
    imgFullResMnPopPre = cat(3, imgFullResMnPopPre, datastructPre(i).imgFullResMn*100);  
    imgOcclResMnPopPre = cat(3, imgOcclResMnPopPre, datastructPre(i).imgOcclResMn*100);  
    imgFullResMnPopPost = cat(3, imgFullResMnPopPost, datastructPost(i).imgFullResMn*100);  
    imgOcclResMnPopPost = cat(3, imgOcclResMnPopPost, datastructPost(i).imgOcclResMn*100);  
end
 
imgFullResMnPopPre = imgFullResMnPopPre-mean(imgFullResMnPopPre(vecAxSp,:,:));
imgOcclResMnPopPre = imgOcclResMnPopPre-mean(imgOcclResMnPopPre(vecAxSp,:,:));
imgFullResMnPopPost = imgFullResMnPopPost-mean(imgFullResMnPopPost(vecAxSp,:,:));
imgOcclResMnPopPost = imgOcclResMnPopPost-mean(imgOcclResMnPopPost(vecAxSp,:,:));



% 
% figure('Position',[250         399        1502         599])
% for j = 1:size(imgFullResMnPopPre,3)
%     mnFull = min(min(imgFullResMnPopPre(:,:,j)));
%     mxFull = max(max(imgFullResMnPopPre(:,:,j)));
%     mnOccl = min(min(imgOcclResMnPopPre(:,:,j)));
%     mxOccl = max(max(imgOcclResMnPopPre(:,:,j)));
%     
%     for i = 1:nImgs
%         subplot(2, nImgs,i);
%         plot(vecAx, imgFullResMnPopPre(:,i,j), 'k')
%         ylim([min([mnFull mnOccl]) max([mxFull mxOccl])])
%         xlim([min(vecAx) max(vecAx)])
%     end
%     
%     for i = 1:nImgs
%         subplot(2,nImgs,i+nImgs);
%         plot(vecAx, imgOcclResMnPopPre(:,i,j), 'k')
%         ylim([min([mnFull mnOccl]) max([mxFull mxOccl])])
%         xlim([min(vecAx) max(vecAx)])
%     end
%     neuron = j
%     pause
% end


% % plot response strength sorted on full screen images per ROI, applied to
% % occluded images to see if 'tuning curves' align.
% imgFullResMnPopPreMn = squeeze(mean(imgFullResMnPopPre(vecAxSt,:,:)));
% imgOcclResMnPopPreMn = squeeze(mean(imgOcclResMnPopPre(vecAxSt,:,:)));
% figure('Position',[414   617   566   303])
% for j = 1:size(imgFullResMnPopPre,3)
% 
%     [~,Rsorted] = sort(imgFullResMnPopPreMn(:,j), 'descend');
%     full = imgFullResMnPopPreMn(Rsorted,j);
%     occl = imgOcclResMnPopPreMn(Rsorted,j);
%     
%     plot(full, 'LineWidth', 2)
%     hold on
%     plot(occl, 'LineWidth', 2)
%     title(['Neuron  ' num2str(j)])
%     xlim([0 5]), xticks(1:4)
%     ylabel('df/f (%)'), hold off
%     yline(0, '--r', 'LineWidth', 1);    
%     pause
% end


% plot average 'tuning curve' for full vs occluded
imgFullResMnPopPreMn = squeeze(mean(imgFullResMnPopPre(vecAxSt,:,:)));
imgOcclResMnPopPreMn = squeeze(mean(imgOcclResMnPopPre(vecAxSt,:,:)));

full = zeros(nImgs,size(imgFullResMnPopPre,3));
occl = zeros(nImgs,size(imgFullResMnPopPre,3));
for j = 1:size(imgFullResMnPopPre,3)
    [~,Rsorted] = sort(imgFullResMnPopPreMn(:,j), 'descend');
    full(:,j) = imgFullResMnPopPreMn(Rsorted,j);
    [~,Rsorted] = sort(imgOcclResMnPopPreMn(:,j), 'descend');
    occl(:,j) = imgOcclResMnPopPreMn(Rsorted,j);
end

% plot average 'tuning curve' for full vs occluded
% imgFullResMnPopPostMn = squeeze(mean(imgFullResMnPopPost(vecAxSt,:,:)));
% imgOcclResMnPopPostMn = squeeze(mean(imgOcclResMnPopPost(vecAxSt,:,:)));
% full = zeros(nImgs,size(imgFullResMnPopPost,3));
% occl = zeros(nImgs,size(imgFullResMnPopPost,3));
% for j = 1:size(imgFullResMnPopPost,3)
%     [~,Rsorted] = sort(imgFullResMnPopPostMn(:,j), 'descend');
%     full(:,j) = imgFullResMnPopPostMn(Rsorted,j);
%     [~,Rsorted] = sort(imgOcclResMnPopPostMn(:,j), 'descend');
%     occl(:,j) = imgOcclResMnPopPostMn(Rsorted,j);
% end

% fullBsl = full-min(full);
% fullNrm = fullBsl./max(fullBsl);
% occlBsl = occl-min(full);
% occlNrm = occlBsl./max(fullBsl);
% fullNrm = full./max(full);
% occlNrm = occl./max(full);

% figure
% % plot(mean(fullNrm,2), 'LineWidth', 2)
% plot(mean(full,2), 'LineWidth', 2)
% hold on
% % plot(mean(occlNrm,2), 'LineWidth', 2)
% plot(mean(occl,2), 'LineWidth', 2)
% title('Average tuning curve over neurons')
% xlim([0 5]), xticks(1:4)
% ylabel('df/f (%)'), hold off
% yline(0, '--r', 'LineWidth', 1);
% legend({'Full','Occl'})

% in case you want to remove data
mx = max(full);
md = median(mx);
idx = mx<md;
full(:,idx)=[];
occl(:,idx)=[];

% fullBsl = full-min(full);
% fullNrm = fullBsl./max(fullBsl);
% occlBsl = occl-min(full);
% occlNrm = occlBsl./max(fullBsl);
% fullNrm = full./max(full);
% occlNrm = occl./max(full);
fullNrm = full;
occlNrm = occl;



figure
% errorscatter(1:4,mean(fullNrm,2),std(fullNrm,[],2),[0 0 0]);    
errorscatter(1:4,mean(fullNrm,2),std(fullNrm,[],2)./sqrt(length(fullNrm)),[0 0 0]);    
hold on
plot(1:4,mean(fullNrm,2),'k', 'LineWidth', 2)
% errorscatter(1:4,mean(occlNrm,2),std(occlNrm,[],2), [1 0 0]);    
errorscatter(1:4,mean(occlNrm,2),std(occlNrm,[],2)./sqrt(length(occlNrm)), [1 0 0]);    
plot(1:4,mean(occlNrm,2),'r', 'LineWidth', 2)
xlim([0 5]), xticks(1:4)
ylabel('df/f (%)'), hold off
yline(0, '--r', 'LineWidth', 1);
legend({'Full','Occl'})


%% plot individual cells and chronically matched (without shadederror bar still)

nNeurons = size(imgFullResMnPopPreBsl, 3);
nPerPage = 10;
nPages = ceil(nNeurons / nPerPage);
nFrames = size(imgFullResMnPopPreBsl, 1);
tPre = vecAxPre;     % x-axis for Pre
tTask = vecAxTask;   % x-axis for Task

for pg = 1:nPages
    figure('Position', [100 100 1800 900]);

    % Indices of neurons for this page
    startIdx = (pg-1)*nPerPage + 1;
    endIdx = min(pg*nPerPage, nNeurons);
    nThisPage = endIdx - startIdx + 1;

    for n = 1:nThisPage
        neuronIdx = startIdx + n - 1;

        % === Compute consistent y-limits for this neuron ===
        yVals = [];

        for stim = 1:4
            yVals = [yVals;
                imgFullResMnPopPreBsl(:, stim, neuronIdx);
                imgOcclResMnPopPreBsl(:, stim, neuronIdx);
                imgFullResMnPopTaskBsl(:, stim, neuronIdx);
                imgOcclResMnPopTaskBsl(:, stim, neuronIdx)];
        end

        yLimNeuron = [min(yVals(:)), max(yVals(:))];

        for stim = 1:4
            % ----- PRE (columns 1–4) -----
            subplot(nPerPage, 9, (n-1)*9 + stim);
            plot(tPre, imgFullResMnPopPreBsl(:, stim, neuronIdx), 'k'); hold on;
            plot(tPre, imgOcclResMnPopPreBsl(:, stim, neuronIdx), 'r');
            ylim(yLimNeuron);
            if stim == 1
                ylabel(['Neuron ' num2str(neuronIdx)]);
            end
            if n == 1
                title(['Pre Img ' num2str(stim)]);
            end

            % ----- TASK (columns 6–9) -----
            subplot(nPerPage, 9, (n-1)*9 + 5 + stim);
            plot(tTask, imgFullResMnPopTaskBsl(:, stim, neuronIdx), 'k'); hold on;
            plot(tTask, imgOcclResMnPopTaskBsl(:, stim, neuronIdx), 'r');
            ylim(yLimNeuron);
            if n == 1
                title(['Task Img ' num2str(stim)]);
            end
        end
    end

    sgtitle(['Neurons ' num2str(startIdx) '–' num2str(endIdx)]);
    disp('Click to continue...');
    pause;
end


% imgIdx1 = [1 2 4 5]; % trained images
% imgIdx2 = [3 6]; % untrained images
% 
% figure
% for i = 1:length(linkMat2)
%     idxPre = linkMat2(i,1);
%     idxPost = linkMat2(i,2);
% 
%     s1 = subplot(2,2,1);
%     plot(vecAx, squeeze(mean(imgFullResMnPopPre(:,imgIdx1,idxPre),2))-mean(squeeze(mean(imgFullResMnPopPre(vecAxSp,imgIdx1,idxPre),2))))
%     s2 = subplot(2,2,2);
%     plot(vecAx, squeeze(mean(imgFullResMnPopPost(:,imgIdx1,idxPost),2))-mean(squeeze(mean(imgFullResMnPopPost(vecAxSp,imgIdx1,idxPost),2))))
%     s3 = subplot(2,2,3);
%     plot(vecAx, squeeze(mean(imgFullResMnPopPre(:,imgIdx2,idxPre),2))-mean(squeeze(mean(imgFullResMnPopPre(vecAxSp,imgIdx2,idxPre),2))))
%     s4 = subplot(2,2,4);
%     plot(vecAx, squeeze(mean(imgFullResMnPopPost(:,imgIdx2,idxPost),2))-mean(squeeze(mean(imgFullResMnPopPost(vecAxSp,imgIdx2,idxPost),2))))
% 
%     ylimMx = max([s1.YLim s2.YLim s3.YLim s4.YLim]);
%     ylimMn = min([s1.YLim s2.YLim s3.YLim s4.YLim]);
%     s1.YLim = [ylimMn ylimMx];
%     s2.YLim = [ylimMn ylimMx];
%     s3.YLim = [ylimMn ylimMx];
%     s4.YLim = [ylimMn ylimMx];
% pause
% end

