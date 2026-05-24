%% LDA analysis on population data from muckli experiments post vs active
%
%	Version History:
%	2023-06-28	Created by Koen Seignette

clear all


filenamesActive = {};
filepathsActive = {};
selecting = true; % true as long as files are being selected
i = 0;
while selecting
    i = i + 1;

    str = sprintf('load posttraining Res file %d. Press cancel when done', i);
    [filenamesActive{i}, filepathsActive{i}] = uigetfile('*Res.mat', str);

    if filenamesActive{i} == 0 % Cancel is pressed probably: stop with selecting
        filenamesActive(i) = [];
        filepathsActive(i) = [];
        selecting = false;
    end

end
filenamesActive = filenamesActive';
filepathsActive = filepathsActive';

nfiles = length(filenamesActive); % The number of files that have been selected

%%
clearvars -except nfiles filenamesActive filepathsActive

% Load the main files
fprintf('\nloading in %d files:\n', nfiles)
for i = 1:nfiles % Backwards to create final size on first loop
    fprintf('\nloading files for mouse %d...',i)
    fnActive = filenamesActive{i};
    pnActive = filepathsActive{i};
    load([pnActive fnActive]);

    datastructActive(i).info = info;
    datastructActive(i).Res = Res;
    datastructActive(i).log = info.Stim.log;

    fprintf('\nsuccesfully loaded files for mouse %d\n',i)
end
fprintf('\nsuccesfully loaded all files\n')


%% load existing datastructs
clear all
% 
% % L2/3
load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\PrePostGrayL23SeparateNewRFs.mat', 'datastructPre')
load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\postActiveGray\datastructChronicL23.mat')
load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\postActiveGray\datastructActiveL23.mat')

% % you can do it like this also for L5 but remember you also did RF mapping
% % for L5 after the active session immediately
% load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\PrePostGrayL5ChronicSeparateNewRFs.mat', 'datastructPre', 'datastructChronic')
% load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\ActiveGray\datastructActiveL5.mat')

%% Initialize and organise data
clearvars -except datastructPre datastructActive datastructChronic nfiles

imgNrsActive = [1 2 3 4]; % image nrs to decode, all images
imgNrsPre = [1 2 4 5]; % image nrs to decode, all images
nImgs = length(imgNrsActive);
performanceChance = 100/nImgs;
nTrials = 20; % nr of trials shown per image
trainFrac = 0.5; % on what fraction would you like to train the decoder (0.8 is good)
rfDistVec = 2; % Minimum distance away from occluder edge
vecAx = datastructActive(1).Res.ax;
vecAxSp = vecAx<0; % spontaneous activity window
vecAxSt = vecAx>0.2 & vecAx<1; % stim window
alphaVal = 0.99; % significance value for cells to be included
if nfiles == 6
    rsqThresh = 0.33; % 0.33 for L2/3, 0.15 for L5
else
    rsqThresh = 0.15; % 0.33 for L2/3, 0.15 for L5
end
snrThresh = 4; % snr threshold for RF
useSpikingData = 0; % deconvolved (1) or df/f (0)
doZscore = true; % if you want to use zscore data
regressRun = false; % regress out running? Only for CaSigCorrected, not for spikes
runNan = false;
runThres = 2;

roiMatchPresentBoth = cell(nfiles, 1); % index in the linkMat at which ROIs were present for the requested recordings (2 and 4)
roiMouseOrigin = []; % in the final data, which mouse does the signal come from?
roiID     = []; % in the final data, what original ROI idx belongs to it?
roiMatchID     = []; % in the final data, which Match was this ROI in the linkmat?
mouseName = cell(nfiles, 1);

% for loop with decoding etc.
for i = 1:nfiles
    
    mouseName{i} = strsplit(datastructActive(i).info.strfp, '\');
    mouseName{i} = mouseName{i}{end-1};
    
    
    linkMat = datastructChronic(i).linkMat; % chronically matched neurons
    clear linkMatIncl
%     linkMatIncl(:,1) = linkMat(:,2); % second column is pre training
    linkMatIncl(:,1) = linkMat(:,1); % FIRST column is pre training
    linkMatIncl(:,2) = linkMat(:,4); % fourth column is active
    roiMatchPresentBoth{i} = find(all(linkMatIncl,2));
    linkMatIncl = linkMatIncl(roiMatchPresentBoth{i},:); % remove rows with at least one zero
    
    % calculation of RF distances to occluder and inclusion criteria based
    % on post dataset
    info = datastructPre(i).info;
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
    % we save this on and offCrit but that's just based on the RFs, not on
    % the chronic matching so we still don't know for the plotting of the
    % RFs which was also there chronically.

    %     rfsPre = find(rfIncl)
    %     linkIncl = false(length(rfInclPost),1);
    %     linkIncl(linkMatIncl(:,2)) = true; % post is second column of linkMatIncl

    [val, posRF, posLink] = intersect(find(rfInclPre),linkMatIncl(:,1));

    roiMouseOrigin = cat(1, roiMouseOrigin, i*ones(length(val), 1));
    roiID = cat(1, roiID, linkMatIncl(posLink,:));
    roiMatchID = cat(1, roiMatchID, posLink);

    %     linkInclPre = false(length(rfInclPre),1);
    %     linkInclPre(linkMatIncl(:,1)) = true; % pre is first column of linkMatIncl


    %     % check which neurons remain
    %     [~,pos] = intersect(linkMatIncl(:,1), find(rfInclPre));
    %     linkMatInclAfterPre = linkMatIncl(pos,:);
    %
    %     [~,pos] = intersect(linkMatInclAfterPre(:,2), find(rfInclPost));
    %     linkMatInclAfterPost = linkMatInclAfterPre(pos,:);

    % resample motionlog into runningtrials
    stimlog = datastructActive(i).log.stimlog(1:160,1);
    motionlog = datastructActive(i).log.motionlog;
    desiredNumFrames = length(vecAx);
    stimlogCut = datastructActive(i).log.stimlog(1:160,1);
    % Initialize a matrix to store the resampled trial data
    trialData = NaN(length(vecAx), length(stimlogCut));
    % Loop over each trial
    for j = 1:length(stimlogCut)
        % Determine the time window for the current trial
        startTime = stimlogCut(j) + vecAx(1);
        endTime = stimlogCut(j) + vecAx(end);
        % Find the corresponding indices in the motionlog
        startIndex = find(motionlog(:, 2) >= startTime, 1, 'first');
        endIndex = find(motionlog(:, 2) <= endTime, 1, 'last');
        % Extract the trial data
        trialFrames = motionlog(startIndex:endIndex, 1);
        % Resample the trial data to have desiredNumFrames frames
        resampledFrames = interp1(linspace(0, 1, numel(trialFrames)), trialFrames, linspace(0, 1, desiredNumFrames));
        % Store the resampled trial data in the matrix
        trialData(:, j) = resampledFrames;
    end
    
    % responses sorted to match across mice
    Res = datastructActive(i).Res; % from active session
    Res.speed = trialData;
    
    if regressRun
        tempTrace = Res.CaSigCorrected(:,1:160,:);
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

    %     matTrialTypes = datastructActive(i).log; % trialtypes
    %     [~, dataSortidx] = sortrows(matTrialTypes', [1 2]);
    %     matTrialTypesSort = matTrialTypes(:,dataSortidx);

    matTrialTypes = datastructActive(i).log.stimlog(1:160,7:8)';
    [~, dataSortidx] = sortrows(matTrialTypes', [1 2]);
    matTrialTypesSort = matTrialTypes(:,dataSortidx);

    % create index to select only images in 'imgNrs' to decode on
    imgIdx = false(1,size(matTrialTypesSort,2));
    for n = 1:nImgs
        imgIdx(matTrialTypesSort(1,:)==imgNrsActive(n))=1;
    end

    if useSpikingData
        CaResSort = Res.CaDeconCorrected(:,dataSortidx,linkMatIncl(posLink,2)); % reordering
    else
        if regressRun
            CaResSort = Res.CaSigCorrected(:,dataSortidx,linkMatIncl(posLink,2)); % reordering, regressRun does subtract 1 already
        else
            CaResSort = Res.CaSigCorrected(:,dataSortidx,linkMatIncl(posLink,2))-1; % reordering and subtract 1
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

%     CaResSort = CaResSort(:,imgIdx,:); % subselect images
    matTrialTypesIncl = matTrialTypesSort(:,imgIdx); % subselect images
    runSpeed = repmat(Res.speed(:,dataSortidx),1,1,size(CaResSort,3));
    runSpeed = runSpeed(:,imgIdx,:);
    if runNan % in case of runtrial removing
        if ~isempty(runSpeed) % some mice might not have any neurons left
            runTrials = mean(runSpeed(vecAxSt,:,1))>runThres; % get trials in which average runspeed in vecAxSt > threshold
            %             runTrials = mean(runSpeed(vecAx>-3&vecAx<2,:,1))>runThres; % slightly bigger window for testing
            CaResSort(:,runTrials,:) = NaN; % remove those trials
        end
    end

    % trace matrices (frames x imgs x trials x rois)
    imgFullRes = zeros(size(CaResSort,1), nImgs, nTrials, size(CaResSort,3)); % pre-allocate
    imgOcclRes = zeros(size(CaResSort,1), nImgs, nTrials, size(CaResSort,3)); % pre-allocate
    for j = 1:nImgs
        imgIdxFull = find(matTrialTypesIncl(1,:)==imgNrsActive(j) & matTrialTypesIncl(2,:)==0);
        imgIdxOccl = find(matTrialTypesIncl(1,:)==imgNrsActive(j) & matTrialTypesIncl(2,:)==1);
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

    imgFullResMn = squeeze(nanmean(imgFullRes,3)); % mean over trials
    imgOcclResMn = squeeze(nanmean(imgOcclRes,3)); % mean over trials
    fullSign = squeeze(nanmean(imgFullResMn,2)); % mean over images
    occlSign = squeeze(nanmean(imgOcclResMn,2)); % mean over images

    % response strength per ROI
    if exist('fullSign', 'var')
        if useSpikingData
            scatFull = squeeze(nanmean(fullSign(vecAxSt,:)));
            scatOccl = squeeze(nanmean(occlSign(vecAxSt,:)));
        else
            scatFull = squeeze(nanmean(fullSign(vecAxSt,:)))-squeeze(nanmean(fullSign(vecAxSp,:)));
            scatOccl = squeeze(nanmean(occlSign(vecAxSt,:)))-squeeze(nanmean(occlSign(vecAxSp,:)));
        end
    else
        scatFull = [];
        scatOccl = [];
        fullSign = [];
        occlSign = [];
    end


    datastructActive(i).CaResSort = CaResSort;
    datastructActive(i).matTrialTypesIncl = matTrialTypesIncl;
    datastructActive(i).matTrialTypes = matTrialTypes;
    datastructActive(i).hValFull = hValFull;
    datastructActive(i).hValOccl = hValOccl;
    datastructActive(i).fullSign = fullSign;
    datastructActive(i).occlSign = occlSign;
    datastructActive(i).imgFullRes = imgFullRes;
    datastructActive(i).imgOcclRes = imgOcclRes;
    datastructActive(i).imgFullResMn = imgFullResMn;
    datastructActive(i).imgOcclResMn = imgOcclResMn;
    datastructActive(i).matData = matData;
    datastructActive(i).scatFull = scatFull;
    datastructActive(i).scatOccl = scatOccl;

    datastructPre(i).rfOnGlmIncl = rfOnGlmIncl(val);
    datastructPre(i).rfOffGlmIncl = rfOffGlmIncl(val);
    datastructPre(i).rfIncl = rfInclPre(val);
    datastructPre(i).rfOnDist = rfOnDist(val);
    datastructPre(i).rfOffDist = rfOffDist(val);
    datastructPre(i).onCrit = onCrit(val);
    datastructPre(i).offCrit = offCrit(val);

    for g = 1:nRois
        azi(g,:) = info.rois(g).azi;
        ele(g,:) = info.rois(g).ele;
        rfsz(g,:) = info.rois(g).rfsz;
        fwhmOn(g,:) = info.rois(g).onFWHM;
        fwhmOff(g,:) = info.rois(g).offFWHM;
    end

    datastructPre(i).azi = azi;
    datastructPre(i).ele = ele;
    datastructPre(i).rfsz = rfsz;
    datastructPre(i).fwhmOn = fwhmOn;
    datastructPre(i).fwhmOff = fwhmOff;

    datastructPre(i).aziIncl = azi(val,:);
    datastructPre(i).eleIncl = ele(val,:);
    datastructPre(i).rfszIncl = rfsz(val,:);
    datastructPre(i).fwhmOnIncl = fwhmOn(val,:);
    datastructPre(i).fwhmOffIncl = fwhmOff(val,:);


    %%%%%%% now to redo pre datastruct with matched neurons
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
        imgIdx(matTrialTypesSort(1,:)==imgNrsPre(n))=1;
    end


    if regressRun
        CaResSort = Res.CaSigCorrected(:,dataSortidx,val);
    else
        CaResSort = Res.CaSigCorrected(:,dataSortidx,val)-1; %
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
        imgIdxFull = find(matTrialTypesIncl(1,:)==imgNrsPre(j) & matTrialTypesIncl(2,:)==0);
        imgIdxOccl = find(matTrialTypesIncl(1,:)==imgNrsPre(j) & matTrialTypesIncl(2,:)==1);
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
    datastructPre(i).CaResSort = CaResSort;    
    %     datastructPre(i).azi = azi;
    %     datastructPre(i).ele = ele;
    %     datastructPre(i).rfsz = rfsz;
    
    disp(i)
    
    %     figure, imagesc(vecAx,[],trialData')
    %     pause
end

% color pallets for plotting
col1 = [0,0,0]; % black
col2 = [131, 197, 190]/255; % blueish
col3 = [0,0,1]; % blue
col4 = [1,0,0]; % red
col5 = [202, 103, 2]/255; % red brownish


%% plotting
% color pallets for plotting
col1 = [0,0,0]; % black
col2 = [131, 197, 190]/255; % blue/greenish
% col3 = [0,0,1]; % blue
% col4 = [1,0,0]; % red
% col5 = [202, 103, 2]/255; % red brownish

save_fig = false;

imgIdx = [1 2 3 4];
% imgIdx = [4];

imgFullResMnPop = datastructActive(1).imgFullResMn;
imgOcclResMnPop = datastructActive(1).imgOcclResMn;

for i = 2:nfiles
    imgFullResMnPop = cat(3, imgFullResMnPop, datastructActive(i).imgFullResMn);
    imgOcclResMnPop = cat(3, imgOcclResMnPop, datastructActive(i).imgOcclResMn);
end

if useSpikingData
    imgFullRes = squeeze(nanmean(imgFullResMnPop(:,imgIdx,:),2));
    imgFullResBsl = imgFullRes;
    imgOcclRes = squeeze(nanmean(imgOcclResMnPop(:,imgIdx,:),2));
    imgOcclResBsl = imgOcclRes;
    scatFullPop = nanmean(imgFullResBsl(vecAxSt,:));
    scatOcclPop = nanmean(imgOcclResBsl(vecAxSt,:));
else
    imgFullRes = squeeze(nanmean(imgFullResMnPop(:,imgIdx,:),2));
    imgFullResBsl = imgFullRes-nanmean(imgFullRes(vecAxSp,:));
    imgOcclRes = squeeze(nanmean(imgOcclResMnPop(:,imgIdx,:),2));
    imgOcclResBsl = imgOcclRes-nanmean(imgOcclRes(vecAxSp,:));
    scatFullPop = nanmean(imgFullResBsl(vecAxSt,:));
    scatOcclPop = nanmean(imgOcclResBsl(vecAxSt,:));
end


% cut off value just for plotting purposes
scatFullPopCut = scatFullPop;
scatOcclPopCut = scatOcclPop;

if nfiles == 5
    mnValCut = -0.5; % min val for cutting for plotting
    mxValCut = 2.5; % max val for cutting for plotting
elseif nfiles == 6
    mnValCut = -0.5; % min val for cutting for plotting
    mxValCut = 2.5; % max val for cutting for plotting
end

scatFullPopCut(scatFullPopCut>mxValCut)=mxValCut+0.5;scatFullPopCut(scatFullPopCut<mnValCut)=mnValCut-0.5;
scatOcclPopCut(scatOcclPopCut>mxValCut)=mxValCut+0.5;scatOcclPopCut(scatOcclPopCut<mnValCut)=mnValCut-0.5;

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
xlabel('Time (s)'), ylabel('Z-scored dF/F'), title('Active'), xlim([-1 3]),figClean
% scatters
s(1) = subplot(2,5,6);
scatter(scatFullPopCut, scatOcclPopCut, sz, cPre, 'filled'); refline(1), figClean
xlabel('Full res'), ylabel('Occl res')
% mean box plot
% subplot(2,5,3)
% boxchart([ones(size(scatFullPop)), ones(size(scatOcclPop))+1], ...
%     [scatFullPop, scatOcclPop], 'MarkerStyle','none'), hold on
% xlim([0 3]), ylabel('Response dF/F (%)'), xticks([1 2]), if nfiles == 6, ylim([-10 10]), elseif nfiles == 5, ylim([-5 35]), end
% xticklabels({'Full', 'Occl'}), xtickangle(45), figClean
% mean scat/bar
subplot(2,5,7)
scatter([1 2],[mean(scatFullPop) mean(scatOcclPop)], 35, 'k', 'LineWidth', 2), hold on
er = errorbar([1 2],[mean(scatFullPop) mean(scatOcclPop)], ...
    [calcSem(scatFullPop) calcSem(scatOcclPop)] ...
    ,[calcSem(scatFullPop) calcSem(scatOcclPop)]);
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 3]), ylabel('Z-scored dF/F'), xticks([1 2]), if nfiles == 6, ylim([0 0.4]), elseif nfiles == 5, ylim([0 0.4]), end
xticklabels({'Full', 'Occl'}), xtickangle(45), figClean

% Adjusting y-axes for subplots 1-4
yMax = max([ylim(t)]);
yMin = min([ylim(t)]);
% set(t(1:4), 'YLim', [yMin yMax]);
if nfiles == 6
    set(t(1), 'YLim', [-0.1 0.6]);
elseif nfiles == 5
    set(t(1), 'YLim', [-0.1 1]);
end
set(t(1), 'XLim', [-1 3])

% axes
if nfiles == 6
    for j = 1:length(s)
        s(j).YLim = [-1 3]; s(j).YTick = -1:0.5:3; s(j).XLim = [-1 3]; s(j).XTick = -1:0.5:3;
    end
elseif nfiles == 5
    for j = 1:length(s)
%         s(j).YLim = [-1 3]; s(j).YTick = -1:0.5:3; s(j).XLim = [-1 3]; s(j).XTick = -1:0.5:3;
        s(j).YLim = [-0.5 3]; s(j).YTick = -0.5:0.5:3; s(j).XLim = [-0.5 3]; s(j).XTick = -0.5:0.5:3;
    end
end
% 
% % axes
% if nfiles == 6
%     for j = 1:length(t)
%         t(j).YLim = [-1 6]; t(j).YTick = -1:1:6; t(j).XLim = [-1 3]; t(j).XTick = -1:1:3;
%     end
%     for j = 1:length(s)
%         s(j).YLim = [-15 35]; s(j).YTick = -10:10:30; s(j).XLim = [-15 35]; s(j).XTick = -10:10:30;
%     end
% elseif nfiles == 5
%     for j = 1:length(t)
%         t(j).YLim = [-1 10]; t(j).YTick = 0:2:10; t(j).XLim = [-1 3]; t(j).XTick = -1:1:3;
%     end
%     for j = 1:length(s)
%         s(j).YLim = [-5 35]; s(j).YTick = -5:5:30; s(j).XLim = [-5 35]; s(j).XTick = -5:5:30;
%     end
% end

if save_fig
    func_save_fig('L23_ActiveTraceAndScatterAndBox')
    func_save_fig('L5_ActiveTraceAndScatterAndBox')
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
% allCLim = get(p, {'CLim'});
% allCLim = cat(2, allCLim{:});
% set(p, 'CLim', [min(allCLim), max(allCLim)]);
if nfiles == 6
    mn = -0.4; mx = 3;
    set(p, 'CLim', [mn, mx]); % for L2/3
elseif nfiles == 5
    mn = -0.4; mx = 3;
    set(p, 'CLim', [mn, mx]); % for L5
end
% colormap(flipud(gray))
colormap hot
subplot(1,3,3)
% axis off, colormap(flipud(gray)), caxis([mn mx]), colorbar
axis off, colormap hot, caxis([mn mx]), colorbar

if save_fig
    func_save_fig('L23_ActiveImagesc')
    func_save_fig('L5_ActiveImagesc')
end


ffTask = scatFullPop;% ffPre(ffPre<0) = 0;
ofTask = scatOcclPop;% ofPre(ofPre<0) = 0;

% we put values <-1 at -1 and >1 at 1 after SI calculation. Doesn't matter whether you do it like this or whether you make
% negative values at 0 before calculating selectivity index.
siFamTask = (ffTask-ofTask)./(ffTask+ofTask); siFamTask(isnan(siFamTask))=0; siFamTask(siFamTask<-1)=-1; siFamTask(siFamTask>1)=1;

% colors = cmapL([0 0 1;0 0 0; 1 0 0], 256);
% lims = [-1 1];
% siFamPreColors = squeeze(SetLimits(siFamTask, lims, colors));

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
xticks(1), xticklabels({'Task'}), xtickangle(45), figClean

if save_fig
    func_save_fig('L23_absSIprepost')
end

figure('Position', [266   311   705   685])
subplot(2,2,1)
scatter(siFamPreAbs, mxfPre, 20, 'k', 'filled'), ylim([-1 3.5])
title('Fam pre'),xlabel('SI'), ylabel('Max response'), figClean


if save_fig
    func_save_fig('L23_SIvsMAXRes')
    func_save_fig('L5_SIvsMAXRes')
end

% divide in two bin, <0.5 si and >0.5 si for statistics
thres = 0.50001;
mxFamPreLow = mxfPre(siFamPreAbs<thres);
mxFamPreHigh = mxfPre(siFamPreAbs>thres);
figure('Position', [680   430   392   548])
scatter([1 2],[nanmean(mxFamPreLow) nanmean(mxFamPreHigh)], 45, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2],[nanmean(mxFamPreLow) nanmean(mxFamPreHigh)], ...
    [calcSem(mxFamPreLow) calcSem(mxFamPreHigh)] ...
    ,[calcSem(mxFamPreLow) calcSem(mxFamPreHigh)]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 3]), ylabel('Response'), xticks([1 2]), ylim([0 0.7])
xticklabels({'PreFamLow', 'PreFamHigh'}), xtickangle(45), figClean
if save_fig
    func_save_fig('L23_SIvsMAXResBinned')
    func_save_fig('L5_SIvsMAXResBinned')
end

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
xtickangle(45), figClean
if save_fig
    func_save_fig('L23_MAXResVsSIBinned')
end

figure('Position', [421   301   800   317])
subplot(1,2,1);
% histogram(siFamPre, lims(1):0.5:lims(2),'Normalization', 'probability', 'FaceColor', 'k'), hold on
% histogram(siFamPost, lims(1):0.5:lims(2),'Normalization', 'probability', 'FaceColor', 'w', 'EdgeColor', 'k')
histogram(siFamTask, -1:0.1:1,'Normalization', 'probability', 'FaceColor', 'k'), hold on
xline(mean(siFamTask)), ylabel('Relative frequency'), xlabel('Selectivity index'),title('Familiar'),figClean

if save_fig
    func_save_fig('L23_ActiveSIhists')
end

figure('Position', [87         278        1635         673])
subplot(2,5,1)
scatter(siFamPreAbs, mxfPre, 15, 'k', 'filled'), ylim([-1 3.5])
title('Fam pre'),xlabel('SI'), ylabel('Max response'), figClean
subplot(2,5,3)
boxchart([ones(size(mxFamPreLow)), 2*ones(size(mxFamPreHigh))], ...
          [mxFamPreLow, mxFamPreHigh], 'MarkerStyle', 'none');
ylabel('Max response'); xlim([0 10]); xticks([1 2 3 4  6 7 8 9]);
xticklabels({'TaskFamLow', 'TaskFamHigh'}); 
xtickangle(45); ylim([0 1]); figClean;
if nfiles == 6
    ylim([-1 2.5]);
elseif nfiles == 5
    ylim([-0.3 1.8]);
end
figClean;

subplot(2,5,4);
boxchart([ones(size(siFamPreLow)), 2*ones(size(siFamPreHigh))], ...
          [siFamPreLow, siFamPreHigh], 'MarkerStyle', 'none');
ylabel('Selectivity'); xlim([0 10]); xticks([1 2 3 4  6 7 8 9]);
xticklabels({'TaskFamLow', 'TaskFamHigh'}); 
xtickangle(45); ylim([0 1]); figClean;

if save_fig
    func_save_fig('L23_ActiveMAXResVsSIplots')
    func_save_fig('L5_ActiveMAXResVsSIplots')
end

[p,h] = ranksum(siFamPreLow,siFamPreHigh)

[p,h] = ranksum(mxFamPreLow, mxFamPreHigh)


%% pre dataset
% famIdx = [1 2 4 5];

vecAxPre = datastructPre(1).Res.ax';
vecAxPreSt = vecAxPre>0.2 & vecAxPre<1;
vecAxPreSp = vecAxPre<0;

vecAxTask = vecAx;
vecAxTaskSp = vecAxTask<0;
vecAxTaskSt = vecAxTask>0.2 & vecAxTask<1;

imgFullResMnPopPre = datastructPre(1).imgFullResMn;
imgOcclResMnPopPre = datastructPre(1).imgOcclResMn;
imgFullResMnPopTask = datastructActive(1).imgFullResMn;
imgOcclResMnPopTask = datastructActive(1).imgOcclResMn;

for i = 2:nfiles
    imgFullResMnPopPre = cat(3, imgFullResMnPopPre, datastructPre(i).imgFullResMn);
    imgOcclResMnPopPre = cat(3, imgOcclResMnPopPre, datastructPre(i).imgOcclResMn);
    imgFullResMnPopTask = cat(3, imgFullResMnPopTask, datastructActive(i).imgFullResMn);
    imgOcclResMnPopTask = cat(3, imgOcclResMnPopTask, datastructActive(i).imgOcclResMn);
end

imgFullResMnPopPreBsl = imgFullResMnPopPre-mean(imgFullResMnPopPre(vecAxPreSp,:,:));
imgOcclResMnPopPreBsl = imgOcclResMnPopPre-mean(imgOcclResMnPopPre(vecAxPreSp,:,:));
imgFullResMnPopTaskBsl = imgFullResMnPopTask-mean(imgFullResMnPopTask(vecAxTaskSp,:,:));
imgOcclResMnPopTaskBsl = imgOcclResMnPopTask-mean(imgOcclResMnPopTask(vecAxTaskSp,:,:));

imgFullResMnPopPreBslMn = squeeze(mean(imgFullResMnPopPreBsl,2));
imgOcclResMnPopPreBslMn = squeeze(mean(imgOcclResMnPopPreBsl,2));
imgFullResMnPopTaskBslMn = squeeze(mean(imgFullResMnPopTaskBsl,2));
imgOcclResMnPopTaskBslMn = squeeze(mean(imgOcclResMnPopTaskBsl,2));

% ix = famIdx;

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

% ix = sparsenessFullPost>0.6;
ix = scatFullTask>0.2;
inc = scatFullTask-scatFullPre;

figure('Position', [1090         247         560         420])
scatter(sparsenessFullTask(ix), inc(ix))
refline(1), refline, xlabel('Sparseness full task'), ylabel('Response full task - response full pre')

% Group 1: ~ixUp
means1 = [nanmean(sparsenessFullTask(ixUp)), nanmean(sparsenessFullTask(ixDown))];
sems1  = [nanstd(sparsenessFullTask(ixUp)) / sqrt(sum(~isnan(sparsenessFullTask(ixDown)))), ...
          nanstd(sparsenessFullTask(ixUp)) / sqrt(sum(~isnan(sparsenessFullTask(ixDown))))];


% 
% % Group 1: ~ixUp
% means1 = [nanmean(sparsenessFullPre(ixUp)), nanmean(sparsenessFullPost(ixUp))];
% sems1  = [nanstd(sparsenessFullPre(ixUp)) / sqrt(sum(~isnan(sparsenessFullPre(ixUp)))), ...
%           nanstd(sparsenessFullPost(ixUp)) / sqrt(sum(~isnan(sparsenessFullPost(ixUp))))];
% % Group 2: ixUp
% means2 = [nanmean(sparsenessFullPre(ixDown)), nanmean(sparsenessFullPost(ixDown))];
% sems2  = [nanstd(sparsenessFullPre(ixDown)) / sqrt(sum(~isnan(sparsenessFullPre(ixDown)))), ...
%           nanstd(sparsenessFullPost(ixDown)) / sqrt(sum(~isnan(sparsenessFullPost(ixDown))))];
figure('Position', [509   252   560   420])
% Bar plots
bar([1 2], means1, 'FaceAlpha', 0.5); hold on
% bar([4 5], means2, 'FaceAlpha', 0.5); hold on

% Error bars
errorbar([1 2], means1, sems1, 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 8)
% errorbar([4 5], means2, sems2, 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 8)


% %% simple scatter plots
% 
% figure('Position', [509   252   560   420])
% scatter(scatFullPre, scatFullTask)
% refline(1), xlabel('Full pre'), ylabel('Full task')
% xlim([-0.5 2.5]), ylim([-0.5 2.5])
% 
% 
% 
% 
% 
% %% some plotting
% figure
% scatter(scatFullPopPre, scatFullPop), refline(1)
% xlim([-1 3]),ylim([-1 3])
% 
% figure
% scatter(scatFullPopPre, scatOcclPopPre), refline(1)
% xlim([-1 3]),ylim([-1 3])
% 
% figure
% scatter(scatFullPop, scatOcclPop), refline(1)
% xlim([-1 3]),ylim([-1 3])

%% chronic colored plot pre vs active
% close all

sz = 20;
cPre = [0.2 0.2 0.2];
cPost = col2;
vecAx = datastructActive(1).Res.ax';
% plot traces and scatters in one figure
figure('Position', [364   231   896   621])
clear t s g
% traces
t(1) = subplot(2,2,1);
shadedErrorBar(vecAxPre,mean(imgFullResMnPopPreBslMn,2)...
    ,std(imgFullResMnPopPreBslMn,0,2)/sqrt(size(imgFullResMnPopPreBslMn,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAxPre,mean(imgOcclResMnPopPreBslMn,2)...
    ,std(imgOcclResMnPopPreBslMn,0,2)/sqrt(size(imgOcclResMnPopPreBslMn,2)), 'lineProps', 'r');
xlabel('Time (s)'), ylabel('dF/F (%)'), xticks(-1:1:3), xlim([-1 3]), title('Pre Fam'), figClean
t(2) = subplot(2,2,2);
shadedErrorBar(vecAx,mean(imgFullResMnPopTaskBslMn,2)...
    ,std(imgFullResMnPopTaskBslMn,0,2)/sqrt(size(imgFullResMnPopTaskBslMn,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAx,mean(imgOcclResMnPopTaskBslMn,2)...
    ,std(imgOcclResMnPopTaskBslMn,0,2)/sqrt(size(imgOcclResMnPopTaskBslMn,2)), 'lineProps', 'r');
xlabel('Time (s)'), title('Post Fam'), figClean
% scatters
s(1) = subplot(2,2,3);
scatter(scatFullPre, scatOcclPre, sz, cPre, 'filled'); refline(1), xlabel('Full'), ylabel('Occl'), title('Pre'), figClean
s(2) = subplot(2,2,4);
scatter(scatFullTask,scatOcclTask , sz, cPost, 'filled'); refline(1), title('Task'), figClean
for j = 1:length(s)
    s(j).YLim = [-1 3]; s(j).YTick = -1:0.5:3; s(j).XLim = [-1 3]; s(j).XTick = -1:0.5:3;
end
for j = 1:length(t)
    t(j).YLim = [-0.2 0.6]; t(j).YTick = -0.2:0.1:0.6; t(j).XLim = [-1 3]; t(j).XTick = -1:0.5:3;
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

figure('Position', [45         439        1146         471]);
sz = 45;
% Pre
subplot(1,2,1)
scatter(scatFullPreSort, scatOcclPreSort, sz, colors, 'filled');
hold on
xlim([-1 3]); ylim([-1 3]);
xticks(-1:1:3),yticks(-1:1:3)
refline(1)
xlabel('NO fam pre'); ylabel('O fam pre');
title('Colored by pre response blend');
figClean

% Task
subplot(1,2,2)
scatter(scatFullTaskSort, scatOcclTaskSort, sz, colors, 'filled');  % same colors
hold on
xlim([-1 3]); ylim([-1 3]);
xticks(-1:1:3),yticks(-1:1:3)
refline(1)
xlabel('NO fam task'); ylabel('O fam task');
title('Colored by pre response blend');
figClean

if save_fig
    func_save_fig('L23_chronic_pretask_scatter_colorcodedOccltask')
    func_save_fig('L5_chronic_pretask_scatter_colorcodedOccltask')
end

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

if save_fig
    func_save_fig('L23_chronic_pretask_scatter_colorcodedOccltask_colorbar')
    func_save_fig('L5_chronic_pretask_scatter_colorcodedOccltask_colorbar')
end

%% just the neurons that we indicate in the paper:
% ---- Setup & original multi-panel figure ----
sz = 20;
cPre = [0.2 0.2 0.2];
cPost = col2;                 % assumes this is defined elsewhere
vecAx = datastructActive(1).Res.ax';

figure('Position', [364   231   896   621])
clear t s g

% traces
t(1) = subplot(2,2,1);
shadedErrorBar(vecAxPre, mean(imgFullResMnPopPreBslMn,2) ...
    , std(imgFullResMnPopPreBslMn,0,2)/sqrt(size(imgFullResMnPopPreBslMn,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAxPre, mean(imgOcclResMnPopPreBslMn,2) ...
    , std(imgOcclResMnPopPreBslMn,0,2)/sqrt(size(imgOcclResMnPopPreBslMn,2)), 'lineProps', 'r');
xlabel('Time (s)'), ylabel('dF/F (%)'), xticks(-1:1:3), xlim([-1 3]), title('Pre Fam'), figClean

t(2) = subplot(2,2,2);
shadedErrorBar(vecAx, mean(imgFullResMnPopTaskBslMn,2) ...
    , std(imgFullResMnPopTaskBslMn,0,2)/sqrt(size(imgFullResMnPopTaskBslMn,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAx, mean(imgOcclResMnPopTaskBslMn,2) ...
    , std(imgOcclResMnPopTaskBslMn,0,2)/sqrt(size(imgOcclResMnPopTaskBslMn,2)), 'lineProps', 'r');
xlabel('Time (s)'), title('Post Fam'), figClean

% scatters (raw, before color coding)
s(1) = subplot(2,2,3);
scatter(scatFullPre, scatOcclPre, sz, cPre, 'filled'); draw_diag45(gca); xlabel('Full'), ylabel('Occl'), title('Pre'), figClean
s(2) = subplot(2,2,4);
scatter(scatFullTask, scatOcclTask, sz, cPost, 'filled'); draw_diag45(gca); title('Task'), figClean

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

% ---- Selection handling ----
% selection = [33 78 11 20 118 54 105 101];           % original neuron% indices to highlight in THIS order BAD MATCHING
% selection = [1:110];
selection = [1, 65, 84, 90, 91, 3, 77, 56, 50];
selection = selection(ismember(selection, sortIdx)); % keep only IDs that exist

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
figure('Position', [45 439 1146 471]);
sz = 45;

% ---------- Pre (ALL) ----------
ax1 = subplot(1,2,1);
scatter(ax1, scatFullPrePlot,  scatOcclPrePlot,  sz, colorsPlot, 'filled'); hold(ax1,'on')
xlim(ax1, [-1 3]); ylim(ax1, [-1 3]); xticks(ax1, -1:1:3); yticks(ax1, -1:1:3);
draw_diag45(ax1);
xlabel(ax1,'NO fam pre'); ylabel(ax1,'O fam pre'); title(ax1,'Colored by pre response blend');
figClean
% Leader lines + selection numbers on PRE (use *Sort coords for the selected points)
preX_sel  = scatFullPreSort(selPos_sorted);
preY_sel  = scatOcclPreSort(selPos_sorted);
draw_leader_labels(ax1, preX_sel, preY_sel, selNums_inOrder);

% ---------- Task (ALL) ----------
ax2 = subplot(1,2,2);
scatter(ax2, scatFullTaskPlot, scatOcclTaskPlot, sz, colorsPlot, 'filled'); hold(ax2,'on')
xlim(ax2, [-1 3]); ylim(ax2, [-1 3]); xticks(ax2, -1:1:3); yticks(ax2, -1:1:3);
draw_diag45(ax2);
xlabel(ax2,'NO fam task'); ylabel(ax2,'O fam task'); title(ax2,'Colored by pre response blend');
figClean
% Leader lines + selection numbers on TASK
taskX_sel = scatFullTaskSort(selPos_sorted);
taskY_sel = scatOcclTaskSort(selPos_sorted);
draw_leader_labels(ax2, taskX_sel, taskY_sel, selNums_inOrder);

% ---- SELECTION-ONLY: same colors + order as original sorted plot, with selection numbers ----
colorsSel   = colors(selPos_sorted, :);
preX_only   = scatFullPreSort(selPos_sorted);
preY_only   = scatOcclPreSort(selPos_sorted);
taskX_only  = scatFullTaskSort(selPos_sorted);
taskY_only  = scatOcclTaskSort(selPos_sorted);
selNumsLab  = selNums_inOrder;   % 1..K

figure('Position', [45 100 1146 471]);
sz = 55;

% ---------- Pre (SELECTION ONLY) ----------
ax3 = subplot(1,2,1);
scatter(ax3, preX_only, preY_only, sz, colorsSel, 'filled'); hold(ax3,'on')
xlim(ax3, [-1 3]); ylim(ax3, [-1 3]); xticks(ax3, -1:1:3); yticks(ax3, -1:1:3);
draw_diag45(ax3);
xlabel(ax3,'NO fam pre'); ylabel(ax3,'O fam pre');
title(ax3,'Selection only — colored by pre response blend');
draw_leader_labels(ax3, preX_only, preY_only, selNumsLab);
figClean

% ---------- Task (SELECTION ONLY) ----------
ax4 = subplot(1,2,2);
scatter(ax4, taskX_only, taskY_only, sz, colorsSel, 'filled'); hold(ax4,'on')
xlim(ax4, [-1 3]); ylim(ax4, [-1 3]); xticks(ax4, -1:1:3); yticks(ax4, -1:1:3);
draw_diag45(ax4);
xlabel(ax4,'NO fam task'); ylabel(ax4,'O fam task');
title(ax4,'Selection only — colored by pre response blend');
draw_leader_labels(ax4, taskX_only, taskY_only, selNumsLab);
figClean

% ---- Optional saves ----
if exist('save_fig','var') && save_fig
    func_save_fig('L23_chronic_pretask_scatter_colorcodedOccltask_withlabels')
    func_save_fig('L23_chronic_pretask_scatter_colorcodedOccltask_selection_leaders')
    func_save_fig('L5_chronic_pretask_scatter_colorcodedOccltask_all_leaders')
    func_save_fig('L5_chronic_pretask_scatter_colorcodedOccltask_selection_leaders')
end

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


%% same but for patent figure

% Define colors based on cutoff
cutoff = 0.6;
colors = repmat([0 0 0], length(scatFullPre), 1); % default black
colors(scatFullPre > cutoff, :) = repmat([0.7 0.7 0.7], sum(scatFullPre > cutoff), 1); % gray

% Sort if you want (optional)
magnitude = sqrt(scatFullPre.^2 + scatOcclPre.^2);
[~, sortIdx] = sort(magnitude, 'ascend');
colors = colors(sortIdx, :);
scatFullPre = scatFullPre(sortIdx);
scatOcclPre = scatOcclPre(sortIdx);
scatFullTask = scatFullTask(sortIdx);
scatOcclTask = scatOcclTask(sortIdx);

figure('Position', [45 439 1146 471]);
sz = 45;

% Pre
subplot(1,2,1)
scatter(scatFullPre, scatOcclPre, sz, colors, 'filled');
hold on
xlim([-1 3]); ylim([-1 3]);
xticks(-1:1:3), yticks(-1:1:3)
refline(1)
xlabel('NO fam pre'); ylabel('O fam pre');
title('Gray if Full > 0.6, else black');
figClean

% Task
subplot(1,2,2)
scatter(scatFullTask, scatOcclTask, sz, colors, 'filled'); % same colors as pre
hold on
xlim([-1 3]); ylim([-1 3]);
xticks(-1:1:3), yticks(-1:1:3)
refline(1)
xlabel('NO fam task'); ylabel('O fam task');
title('Gray if Full > 0.6, else black');
figClean

if save_fig
    func_save_fig('L23_chronic_pretask_scatter_colorcodedOccltask_colorbar_patent')
    func_save_fig('L5_chronic_pretask_scatter_colorcodedOccltask_colorbar_patent')
end


%% rfdist vs occl response

rfInclPostPop = cat(1, datastructPre(:).rfIncl);
rfOnDistPostPop = cat(1, datastructPre(:).rfOnDist);
rfOffDistPostPop = cat(1, datastructPre(:).rfOffDist);
onCritPostPop = cat(1, datastructPre(:).onCrit);
offCritPostPop = cat(1, datastructPre(:).offCrit);

aziOnPostIncl = cat(1, datastructPre(:).aziIncl);
aziOnPostIncl = aziOnPostIncl(:,1);
aziOffPostIncl = cat(1, datastructPre(:).aziIncl);
aziOffPostIncl = aziOffPostIncl(:,2);
eleOnPostIncl = cat(1, datastructPre(:).eleIncl);
eleOnPostIncl = eleOnPostIncl(:,1);
eleOffPostIncl = cat(1, datastructPre(:).eleIncl);
eleOffPostIncl = eleOffPostIncl(:,2);

aziOnPost = cat(1, datastructPre(:).azi);
aziOnPost = aziOnPost(:,1);
aziOffPost = cat(1, datastructPre(:).azi);
aziOffPost = aziOffPost(:,2);
eleOnPost = cat(1, datastructPre(:).ele);
eleOnPost = eleOnPost(:,1);
eleOffPost = cat(1, datastructPre(:).ele);
eleOffPost = eleOffPost(:,2);

rfValPost = zeros(length(onCritPostPop),1);
for i = 1:length(onCritPostPop)
    if onCritPostPop(i) && offCritPostPop(i)
        rfValPost(i) = 1;
    elseif onCritPostPop(i)
        rfValPost(i) = 1;
    elseif offCritPostPop(i)
        rfValPost(i) = 2;
    end
end

% plot RFs before and after exclusion pre and post training
sz = 12;
figure('Position', [323         281        1040         585])
subplot(2,2,3)
scatter(aziOnPost, eleOnPost, sz, 'filled', 'r'), xlim([-60 60]), ylim([-45 45]), hold on
scatter(aziOffPost, eleOffPost, sz, 'filled', 'b'), xlim([-60 60]), ylim([-45 45])
% Only the good RFs, either ON or OFF depending on the neuron
subplot(2,2,4)
scatter(aziOnPostIncl(rfValPost==1), eleOnPostIncl(rfValPost==1), sz, 'filled', 'r'), xlim([-60 60]), ylim([-45 45]), hold on
scatter(aziOffPostIncl(rfValPost==2), eleOffPostIncl(rfValPost==2), sz, 'filled', 'b'), xlim([-60 60]), ylim([-45 45])

if save_fig
    func_save_fig('L23_RFselection')
end

% % plot correlation between distance to Occluder edge and occl responses
rfValPostIncl = rfValPost;%(rfInclPostPop);
rfDistPostData = zeros(length(rfOnDistPostPop),1);
rfDistPostData(rfValPostIncl==1) = rfOnDistPostPop(rfValPostIncl==1); % 1 equals ON, 2 equals OFF
rfDistPostData(rfValPostIncl==2) = rfOffDistPostPop(rfValPostIncl==2);
[rDistVsOccl, pDistVsOccl] = corrcoef(rfDistPostData, scatOcclPop);

figure('Position', [96   226   560   420])
fits = polyfit(rfDistPostData, scatOcclPop,1);
fit1 = polyval(fits,rfDistPostData);
vr = scatter(rfDistPostData, scatOcclPop, sz, 'k', 'filled'); hold on
plot(rfDistPostData, fit1, 'r', 'LineWidth', 1.5)
ylabel('Occl res'), xlabel('Dist to edge'), title('RFdist vs Occl res')
text(5,2,sprintf('r=%.3f',rDistVsOccl(2))), text(5,1.5,sprintf('p=%.3f', pDistVsOccl(2)))
figClean

if save_fig
    func_save_fig('L23_RFdistVsOcclResActive')
end

%% correlation plots
% correlations are done on all data, plotting is done with 'cut off data' 
save_fig = false;

corrFullPrep = squeeze(mean(imgFullResMnPop(vecAxSt,:,:))-mean(imgFullResMnPop(vecAxSp,:,:)));
corrOcclPrep = squeeze(mean(imgOcclResMnPop(vecAxSt,:,:))-mean(imgOcclResMnPop(vecAxSp,:,:)));
mnValCut = -100; % min val for cutting for plotting
mxValCut = 300; % max val for cutting for plotting
sz = 7; % scatter size for plotting

%%%%%%%% correlation plots full/full, occl/occl
% possible correlations for full/full and occl/occl
c = nchoosek(1:4,2);
nrComs = length(c);

% full post
figure('Position', [400    42   120   954])
for i = 1:nrComs
    subplot(8,1,i)
    x = corrFullPrep(c(i,1),:);
    y = corrFullPrep(c(i,2),:);
    x1 = x; x1(x1>mxValCut)=mxValCut;x1(x1<mnValCut)=mnValCut;
    y1 = y; y1(y1>mxValCut)=mxValCut;y1(y1<mnValCut)=mnValCut;
    scatter(x1,y1,sz,'filled', 'k')
%     if nfiles==6
%         xlim([mnValCut-10 mxValCut+10]), ylim([-mnValCut-10 mxValCut+10])
%     end
    refline(1), xline(0), yline(0)
    [Rp(i),Pp(i)]=corr(x',y', 'Type', 'Pearson');
    [Rs(i),Ps(i)]=corr(x',y', 'Type', 'Spearman');
%     if i == 1
%         title('Full post corrs')
%     end
    title(sprintf('%d vs %d', c(i,1), c(i,2)))
%     text(-15,35,sprintf('Rp=%.2f', Rp(i))), text(15,35,sprintf('Pp=%.2f', Pp(i)))
%     text(-15,27,sprintf('Rs=%.2f', Rs(i))), text(15,27,sprintf('Ps=%.2f', Ps(i)))
%     xticks(mnValCut-10:10:mxValCut+10), yticks(mnValCut-10:10:mxValCut+10)
    xticks(''), yticks('')
%     if i ~= 1
%         xticklabels(''), yticklabels('')
%     end
end
RpFull = Rp; PpFull = Pp; RsFull = Rs; PsFull = Ps;
if save_fig, func_save_fig('L23_FullCorrs'), end

% occl post
figure('Position', [900    42   120   954])
for i = 1:nrComs
    subplot(8,1,i)
    x = corrOcclPrep(c(i,1),:);
    y = corrOcclPrep(c(i,2),:);
    x1 = x; x1(x1>mxValCut)=mxValCut;x1(x1<mnValCut)=mnValCut;
    y1 = y; y1(y1>mxValCut)=mxValCut;y1(y1<mnValCut)=mnValCut;
    scatter(x1,y1,sz,'filled', 'k')
%     if nfiles==6
%         xlim([mnValCut-10 mxValCut+10]), ylim([-mnValCut-10 mxValCut+10])
%     end
    refline(1), xline(0), yline(0)
    [Rp(i),Pp(i)]=corr(x',y', 'Type', 'Pearson');
    [Rs(i),Ps(i)]=corr(x',y', 'Type', 'Spearman');
%     if i == 1
%         title('Occl post corrs')
%     end
    title(sprintf('%d vs %d', c(i,1), c(i,2)))
%     text(-15,35,sprintf('Rp=%.2f', Rp(i))), text(15,35,sprintf('Pp=%.2f', Pp(i)))
%     text(-15,27,sprintf('Rs=%.2f', Rs(i))), text(15,27,sprintf('Ps=%.2f', Ps(i)))
%     xticks(mnValCut-10:10:mxValCut+10), yticks(mnValCut-10:10:mxValCut+10)
    xticks(''), yticks('')
%     if i ~= 1
%         xticklabels(''), yticklabels('')
%     end
end
RpOccl = Rp; PpOccl = Pp; RsOccl = Rs; PsOccl = Ps;
if save_fig, func_save_fig('L23_OcclCorrs'), end


%%%%%% this section calculates the same correlation but now between all
%%%%%% full and occluded images so that we can compare full-full and
%%%%%% occl-occl versus full-occl
% possible correlations for full/occl
% possible options: 1-1, 1-2, 1-3, 1-4 etc, so 4x4 = 16 options?
clear c
c(:,1) = [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4];
c(:,2) = [1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4];
nrComs = length(c);
order = [1:2:nrComs 2:2:nrComs];

clear Rp Pp Rs Ps
figure('Position', [425    42   249   954])
for i = 1:nrComs
    subplot(8,nrComs/8,order(i))
    x = corrFullPrep(c(i,1),:);
    y = corrOcclPrep(c(i,2),:);
    x1 = x; x1(x1>mxValCut)=mxValCut;x1(x1<mnValCut)=mnValCut;
    y1 = y; y1(y1>mxValCut)=mxValCut;y1(y1<mnValCut)=mnValCut;
    scatter(x1,y1,sz,'filled', 'k')
%     if nfiles==6
%         xlim([mnValCut-10 mxValCut+10]), ylim([-mnValCut-10 mxValCut+10])
%     end
    refline(1), xline(0), yline(0)
    [Rp(i),Pp(i)]=corr(x',y', 'Type', 'Pearson');
    [Rs(i),Ps(i)]=corr(x',y', 'Type', 'Spearman');
%     if i == 1
%         title('Full-Occl post corrs')
%     end
    title(sprintf('%d vs %d', c(i,1), c(i,2)))
%     text(-15,35,sprintf('Rp=%.2f', Rp(i))), text(15,35,sprintf('Pp=%.2f', Pp(i)))
%     text(-15,27,sprintf('Rs=%.2f', Rs(i))), text(15,27,sprintf('Ps=%.2f', Ps(i)))
%     xticks(mnValCut-10:10:mxValCut+10), yticks(mnValCut-10:10:mxValCut+10)
    xticks(''), yticks('')
%     if i ~= 1
%         xticklabels(''), yticklabels('')
%     end
end
RpFullOccl = Rp; PpFullOccl = Pp; RsFullOccl = Rs; PsFullOccl = Ps;
if save_fig, func_save_fig('L23_FullOcclCorrs'), end

%%%% plot quantification
figure('Position', [ 1067         436         476         407])
subplot(1,2,1)
scatter([1 2 3],[mean(RpFull) mean(RpOccl) mean(RpFullOccl)], 45, 'k', 'LineWidth', 2), hold on
er = errorbar([1 2 3],[mean(RpFull) mean(RpOccl) mean(RpFullOccl)], ...
    [calcSem(RpFull) calcSem(RpOccl) calcSem(RpFullOccl)] ...
    ,[calcSem(RpFull) calcSem(RpOccl) calcSem(RpFullOccl)]);    
    er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 4]), ylim([-0.2 1]), ylabel('Pearson'), xticks([1 2 3]), title('Pearson')
xticklabels({'NO-NO', 'O-O', 'NO-O'}), xtickangle(45), figClean
subplot(1,2,2)
scatter([1 2 3],[mean(RsFull) mean(RsOccl) mean(RsFullOccl)], 45, 'k', 'LineWidth', 2), hold on
er = errorbar([1 2 3],[mean(RsFull) mean(RsOccl) mean(RsFullOccl)], ...
    [calcSem(RsFull) calcSem(RsOccl) calcSem(RsFullOccl)] ...
    ,[calcSem(RsFull) calcSem(RsOccl) calcSem(RsFullOccl)]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 4]), ylim([-0.2 1]), ylabel('Spearman'), xticks([1 2 4 5 7 8]), title('Spearman')
xticklabels({'NO-NO', 'O-O', 'NO-O'}), xtickangle(45), figClean
save_fig = false;
if save_fig, func_save_fig('L23_CorrQuants'), end





%% DECODING
rng(1)

save_fig = false;
col1 = [0,0,0]; % black
col2 = [131, 197, 190]/255; % blue/greenish
doDecoding = 1;
doPlotting = 1;
nReps = 200; % nr of iterations for training/testing on subsample in order to sample everything
nBoots = 1000; % nr of iterations for the permutation test

matDataPop = datastructActive(1).matData;

for i = 2:nfiles
    matDataPop = cat(2, matDataPop, datastructActive(i).matData);
end

if doDecoding
    % Pre
    [pFF, pOO, pFO, pOF,pFFperm, pOOperm,pFOperm, pOFperm,cMatFF, cMatOO, cMatFO, cMatOF,...
        cMatFFperm, cMatOOperm, cMatFOperm, cMatOFperm, dPredictFull, dPredictOccl]...
        = doMuckliDecodingLDAblock2(matDataPop, matTrialTypesIncl, trainFrac, nReps, nBoots, 0);
end

if doPlotting
    % PLOTTING
    figure('Position', [170   400   560   420])
    histogram(pOO, length(unique(pOO)), 'Normalization', 'Probability', 'FaceColor', col1, 'EdgeColor', col1, 'LineWidth', 2);
    xlim([0 100])
    xline(mean(pOO), 'Color', col1, 'LineWidth', 2)
    xlabel('Decoding accuracy')
    ylabel('Fraction of decoding runs')
    figClean
    figure('Position', [749   395   560   420])
    h1 = cdfplot(pOO);
    xlim([0 100])
    xlabel('Decoding accuracy')
    ylabel('Cumulative density function')
    set(h1,'LineWidth',2, 'Color', col1)
    box off, grid off
    figClean

    % weights from decoding
    figure
    subplot(2,1,1), histogram(mean(dPredictOccl,2))
    subplot(2,1,2), histogram(mean(dPredictFull,2))

    pFFMn = mean(pFF);
    pFFSEM = std(pFF);
    pOOMn = mean(pOO);
    pOOSEM = std(pOO);
    pFOMn = mean(pFO);
    pFOSEM = std(pFO);
    pOFMn = mean(pOF);
    pOFSEM = std(pOF);

    figure('Position', [423   199   294   505])
    bar([1 2 4 5],[pFFMn pOOMn pFOMn pOFMn], 'FaceColor', 'w', 'EdgeColor', 'k', 'FaceAlpha', 0.6, 'LineWidth', 2)
    hold on
    er = errorbar([1 2 4 5],[pFFMn pOOMn pFOMn pOFMn]...
        ,[0 0 0 0],[pFFSEM pOOSEM pFOSEM pOFSEM]);
    er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2;
    xticks([1 2 4 5]), xticklabels({'F-F', 'O-O', 'F-O', 'O-F'}), ylabel('Decoding accuracy'),figClean
    title('Decoding accuracy')
    ylim([0 120])
    yline(100/nImgs)

    if save_fig
        func_save_fig('L23_DecodingBarsActive')
    end


    % Permutation plots
    figure('Position', [ 763   195   294   505])
    hold on
    subplot(4,1,1)
    histogram(pFFperm, length(unique(pFFperm)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pFFMn, 'r', 'LineWidth', 2);
    %     pval = (sum(pFFperm>pFFMn))/nBoots;
    pval = paretoEst(pFFperm/100, pFFMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100])
    %     legend({'Permutation data', 'Real data'})
    ylabel('Fraction of runs')
    set(gca, 'LineWidth', 2, 'FontSize', 12)
    box off
    %     figure('Position', [   211   201   540   844])
    subplot(4,1,2)
    histogram(pOOperm, length(unique(pOOperm)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pOOMn, 'r', 'LineWidth', 2);
    %     pval = (sum(pOOperm>pOOMn))/nBoots;
    pval = paretoEst(pOOperm/100, pOOMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100])
    %     legend({'Permutation data', 'Real data'})
    ylabel('Fraction of runs')
    set(gca, 'LineWidth', 2, 'FontSize', 12)
    box off

    %     figure('Position', [   211   201   540   844])
    subplot(4,1,3)
    histogram(pFOperm, length(unique(pFOperm)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pFOMn, 'r', 'LineWidth', 2);
    %     pval = (sum(pFOperm>pFOMn))/nBoots;
    pval = paretoEst(pFOperm/100, pFOMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100])
    %     legend({'Permutation data', 'Real data'})
    ylabel('Fraction of runs')
    set(gca, 'LineWidth', 2, 'FontSize', 12)
    box off
    %     figure('Position', [   211   201   540   844])
    subplot(4,1,4)
    histogram(pOFperm, length(unique(pOFperm)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pOFMn, 'r', 'LineWidth', 2);
    %     pval = (sum(pOFperm>pOFMn))/nBoots;
    pval = paretoEst(pOFperm/100, pOFMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100])
    xlabel('Decoding accuracy')
    %     legend({'Permutation data', 'Real data'})
    ylabel('Fraction of runs')
    set(gca, 'LineWidth', 2, 'FontSize', 12)
    box off


    if save_fig
        func_save_fig('L23_DecodingHistsActive')
    end


    cMatFFSum = sum(cMatFF,3);
    cMatFFSum = cMatFFSum./sum(cMatFFSum,1)*100;
    cMatOOSum = sum(cMatOO,3);
    cMatOOSum = cMatOOSum./sum(cMatOOSum,1)*100;
    cMatFOSum = sum(cMatFO,3);
    cMatFOSum = cMatFOSum./sum(cMatFOSum,1)*100;
    cMatOFSum = sum(cMatOF,3);
    cMatOFSum = cMatOFSum./sum(cMatOFSum,1)*100;
    cMatFFpermSum = sum(cMatFFperm,3);
    cMatFFpermSum = cMatFFpermSum./sum(cMatFFpermSum,1)*100;
    cMatOOpermSum = sum(cMatOOperm,3);
    cMatOOpermSum = cMatOOpermSum./sum(cMatOOpermSum,1)*100;
    cMatFOpermSum = sum(cMatFOperm,3);
    cMatFOpermSum = cMatFOpermSum./sum(cMatFOpermSum,1)*100;
    cMatOFpermSum = sum(cMatOFperm,3);
    cMatOFpermSum = cMatOFpermSum./sum(cMatOFpermSum,1)*100;

    figure('Position', [380         401        1123         454])
    s(1) = subplot(2,4,1);
    imagesc(cMatFFSum), xlabel('Predicted'), ylabel('Shown')
    s(2) = subplot(2,4,2);
    imagesc(cMatOOSum)
    s(3) = subplot(2,4,3);
    imagesc(cMatFOSum)
    s(4) = subplot(2,4,4);
    imagesc(cMatOFSum)
    s(5) = subplot(2,4,5);
    imagesc(cMatFFpermSum)
    s(6) = subplot(2,4,6);
    imagesc(cMatOOpermSum)
    s(7) = subplot(2,4,7);
    imagesc(cMatFOpermSum)
    s(8) = subplot(2,4,8);
    imagesc(cMatOFpermSum)
    set(s, 'CLim', [0, 100]);

end


%%

doDecoding = 1;
doPlotting = 1;
nReps = 200;
nBoots = 1000;
famIdx = [1 2 4 5];
novIdx = [3 6];
trialTypes = matTrialTypesIncl;

matDataPopPre = datastructPre(1).matData;
matDataPop = datastructActive(1).matData;

for i = 2:nfiles
    matDataPopPre = cat(2, matDataPopPre, datastructPre(i).matData);
    matDataPop = cat(2, matDataPop, datastructActive(i).matData);
end

idx = famIdx; % which images to include? famidx/novidx?
ix = 1:6;
ix(idx) = [];
for i = 1:length(ix)
    rmv = trialTypes(1,:)==ix(i);
    trialTypes(:,rmv)=[];
    matDataPopPre(rmv,:)=[];
end

% ix = scatOcclFamPopPost>0.5 & scatFullFamPopPost<0.2;
ix = scatOcclPop>0.5;
% ix = scatFullFamPopPre>0.5;

matDataPopPre = matDataPopPre(:,ix);
matDataPop = matDataPop(:,ix);

if doDecoding

    [pFFPre, pOOPre, pFOPre, pOFPre,pFFpermPre, pOOpermPre,pFOpermPre, pOFpermPre,cMatFFPre, cMatOOPre, cMatFOPre, cMatOFPre,...
        cMatFFpermPre, cMatOOpermPre, cMatFOpermPre, cMatOFpermPre, dPredictFullPre, dPredictOcclPre]...
        = doMuckliDecodingLDACrossDecodingRevisions(matDataPopPre, matDataPop, trialTypes, trainFrac, nReps, nBoots, 0);

%         Post
%         [pFFPost, pOOPost, pFOPost, pOFPost,pFFpermPost, pOOpermPost,pFOpermPost, pOFpermPost, cMatFFPost, cMatOOPost, cMatFOPost, cMatOFPost,...
%             cMatFFpermPost, cMatOOpermPost, cMatFOpermPost, cMatOFpermPost,dPredictFullPost, dPredictOcclPost]...
%             = doMuckliDecodingLDACrossDecodingRevisions(matDataPopPre, matDataPop, trialTypes, trainFrac, nReps, nBoots, 0);
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
    figClean
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
    figClean

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
    figClean

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
    xticks([1 2 4 5 7 8 10 11]), xticklabels({'F-F', 'F-F', 'O-O', 'O-O', 'F-O', 'F-O', 'O-F', 'O-F'}), ylabel('MM response post - pre'),figClean
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


%% decoding over time (takes a long time!)
% note that this does not take into account yet if you want to include only
% a subset of the images

doDecoding = 1;
doPlotting = 1;
nReps = 100; % nr of iterations for training/testing on subsample in order to sample everything
nBoots = 100; % nr of iterations for the permutation test
% nReps = 1; % nr of iterations for training/testing on subsample in order to sample everything
% nBoots = 10; % nr of iterations for the permutation test

sigBin = 1:248; % bins for time decoding (depend on your axes)
if nfiles == 5
    sigBin = 1:124; % has fewer samples extracted
end

nBins = length(sigBin); % number of bins

caResPopPre = datastructPre(1).CaResSort;
for i = 2:nfiles % concatenate all ROIs
    caResPopPre = cat(3, caResPopPre, datastructPre(i).CaResSort);
end
% baseline correct
caResPopPre = caResPopPre-mean(caResPopPre(vecAxSp,:,:));
% smooth with movmean 3 frames
for i = 1:size(caResPopPre,3)
    for j = 1:size(caResPopPre,2)
        %         caResPopPre(:,j,i) = smoothG(caResPopPre(:,j,i), 2);
        data = squeeze(caResPopPre(:,j,i));
        caResPopPre(:,j,i) = smoothdata(data, 'movmean', 3);
    end
end
sigBinResPre = zeros(nBins, size(caResPopPre,2), size(caResPopPre,3));
for sample = 1:nBins
    stBin = sigBin(sample);
    sigBinResPre(sample,:,:) = caResPopPre(stBin,:,:); % get response magnitudes per trial
end


%%%% task data
caResPopPost = datastructActive(1).CaResSort;
for i = 2:nfiles % concatenate all ROIs
    caResPopPost = cat(3, caResPopPost, datastructActive(i).CaResSort);
end
% baseline correct
caResPopPost = caResPopPost-mean(caResPopPost(vecAxSp,:,:));
% smooth with movmean 3 frames
for i = 1:size(caResPopPost,3)
    for j = 1:size(caResPopPost,2)
        %         caResPopPost(:,j,i) = smoothG(caResPopPost(:,j,i), 2);
        data = squeeze(caResPopPost(:,j,i));
        caResPopPost(:,j,i) = smoothdata(data, 'movmean', 3);
    end
end
sigBinResPost = zeros(nBins, size(caResPopPost,2), size(caResPopPost,3));
for sample = 1:nBins
    stBin = sigBin(sample);
    sigBinResPost(sample,:,:) = caResPopPost(stBin,:,:); % get response magnitudes per trial
end

ix = scatOcclPop>0.5;
% ix = scatFullFamPopPre>0.5;

sigBinResPre = sigBinResPre(:,:,ix);
sigBinResPost = sigBinResPost(:,:,ix);

if doDecoding
    % decoding
    clear pFFPreTime pOOPreTime pFOPreTime pOFPreTime pFFpermPreTime...
        pOOpermPreTime pFOpermPreTime pOFpermPreTime dPredictFullPreTime dPredictOcclPreTime
    for j = 1:nBins
        dataPre = squeeze(sigBinResPre(j,:,:));
        dataPost = squeeze(sigBinResPost(j,:,:));
        [pFFPreTime(j,:), pOOPreTime(j,:), pFOPreTime(j,:), pOFPreTime(j,:),...
            pFFpermPreTime(j,:), pOOpermPreTime(j,:),pFOpermPreTime(j,:), pOFpermPreTime(j,:),~, ~, ~, ~,...
            ~, ~, ~, ~, dPredictFullPreTime(j,:,:), dPredictOcclPreTime(j,:,:)]...
            = doMuckliDecodingLDACrossDecodingRevisions(dataPre, dataPost, matTrialTypesIncl, trainFrac, nReps, nBoots, 0);
    disp(j)
    end
end

if doPlotting
    clear pvalFFPre pvalOOPre pvalFOPre pvalOFPre pvalFFPost pvalOOPost pvalFOPost pvalOFPost
    for i = 1:size(pFFPreTime,1)

        %         calculate significance over trace using pareto tail estimation (Paolo)
        pvalFFPre(i) = paretoEst(pFFpermPreTime(i,:)/100, mean(pFFPreTime(i,:)/100,2));
        pvalOOPre(i) = paretoEst(pOOpermPreTime(i,:)/100, mean(pOOPreTime(i,:)/100,2));
        pvalFOPre(i) = paretoEst(pFOpermPreTime(i,:)/100, mean(pFOPreTime(i,:)/100,2));
        pvalOFPre(i) = paretoEst(pOFpermPreTime(i,:)/100, mean(pOFPreTime(i,:)/100,2));

    end

    pFFPreTimeSign = zeros(size(pFFPreTime,1),1);% pFFPreTimeSign(pvalFFPre<0.05)=1;
    pOOPreTimeSign = zeros(size(pFFPreTime,1),1);% pOOPreTimeSign(pvalOOPre<0.05)=1;
    pFOPreTimeSign = zeros(size(pFFPreTime,1),1);% pFOPreTimeSign(pvalFOPre<0.05)=1;
    pOFPreTimeSign = zeros(size(pFFPreTime,1),1);% pOFPreTimeSign(pvalOFPre<0.05)=1;

    signSz = 1;
    topSign1 = 110;
    botSign1 = 105;
    topSign2 = 57;
    botSign2 = 55;


    figure('Position', [83    74   773   627])
    clear s
    s(1) = subplot(2,2,1); yline(performanceChance), hold on
    shadedErrorBar(vecAx,mean(pFFPreTime,2)...
        ,std(pFFPreTime,0,2)/sqrt(size(pFFPreTime,2)), 'lineProps', 'b'); hold on
    shadedErrorBar(vecAx,mean(pOOPreTime,2)...
        ,std(pOOPreTime,0,2)/sqrt(size(pOOPreTime,2)), 'lineProps', 'r'); hold on, xline(0)
    plot(vecAx(find(pOOPreTimeSign)), ones(sum(pOOPreTimeSign),1)+botSign1, 'square', 'MarkerSize', signSz, 'MarkerFaceColor',[1 0 0])
    plot(vecAx(find(pFFPreTimeSign)), ones(sum(pFFPreTimeSign),1)+topSign1, 'square', 'MarkerSize', signSz, 'MarkerFaceColor',[0 0 1])
    legend({'','','', 'FF','','','','OO'}, 'Location', 'best'), legend boxoff, title('Pre cross val'), xlabel('Time (s)'), ylabel('Decoding accuracy (%)'), figClean
    s(2) = subplot(2,2,2); yline(performanceChance), hold on
    shadedErrorBar(vecAx,mean(pFOPreTime,2)...
        ,std(pFOPreTime,0,2)/sqrt(size(pFOPreTime,2)), 'lineProps', 'b'); hold on
    shadedErrorBar(vecAx,mean(pOFPreTime,2)...
        ,std(pOFPreTime,0,2)/sqrt(size(pOFPreTime,2)), 'lineProps', 'r'); hold on, xline(0)
    plot(vecAx(find(pOFPreTimeSign)), ones(sum(pOFPreTimeSign),1)+botSign2, 'square', 'MarkerSize', signSz, 'MarkerFaceColor',[1 0 0])
    plot(vecAx(find(pFOPreTimeSign)), ones(sum(pFOPreTimeSign),1)+topSign2, 'square', 'MarkerSize', signSz, 'MarkerFaceColor',[0 0 1])
    legend({'','','', 'FO','','','','OF'}, 'Location', 'best'), legend boxoff,title('Pre cross decoding'), xlabel('Time (s)'), ylabel('Decoding accuracy (%)'), figClean
    s(3) = subplot(2,2,3); yline(performanceChance), hold on
    shadedErrorBar(vecAx,mean(pFFpermPreTime,2)...
        ,std(pFFpermPreTime,0,2)/sqrt(size(pFFpermPreTime,2)), 'lineProps', 'b'); hold on
    shadedErrorBar(vecAx,mean(pOOpermPreTime,2)...
        ,std(pOOpermPreTime,0,2)/sqrt(size(pOOpermPreTime,2)), 'lineProps', 'r'); hold on, xline(0)
    xlabel('Time (s)'), ylabel('Decoding accuracy (%)'), figClean
    s(4) = subplot(2,2,4); yline(performanceChance), hold on
    shadedErrorBar(vecAx,mean(pFOpermPreTime,2)...
        ,std(pFOpermPreTime,0,2)/sqrt(size(pFOpermPreTime,2)), 'lineProps', 'b'); hold on
    shadedErrorBar(vecAx,mean(pOFpermPreTime,2)...
        ,std(pOFpermPreTime,0,2)/sqrt(size(pOFpermPreTime,2)), 'lineProps', 'r'); hold on, xline(0)
    xlabel('Time (s)'), ylabel('Decoding accuracy (%)'), figClean

    % for g = 1:length(s)
    %     s(g).YLim = [0 140]; s(g).YTick = 0:20:140; s(g).XLim = [-1 3]; s(g).XTick = -1:3;
    % end
    g = 1;
    s(g).YLim = [0 120]; s(g).YTick = 0:20:120; s(g).XLim = [-1 3]; s(g).XTick = -1:3;
    g = 2;
    s(g).YLim = [10 60]; s(g).YTick = 10:10:60; s(g).XLim = [-1 3]; s(g).XTick = -1:3;

    for g = 1:length(s)
        s(g).TickDir = 'in';
    end

    if save_fig
        func_save_fig('L23_timeDecodingActive')
        func_save_fig('L5_timeDecodingActive')
%         func_save_fig('L5_traceAndScatterAndBoxActive')
    end


end

%% plot some examples
famIdx = 1:4;

figure('Position', [363         520        1112         169])
for g = 3:nfiles
    for i = 1:size(datastructActive(g).imgFullRes,4)
        delete(subplot(1,length(famIdx),1:length(famIdx)))
        for j = 1:length(famIdx)
            traceFull = squeeze(datastructActive(g).imgFullRes(:,famIdx(j),:,i)-mean(datastructActive(g).imgFullRes(vecAxSp,famIdx(j),:,i)))*100;
            traceOccl = squeeze(datastructActive(g).imgOcclRes(:,famIdx(j),:,i)-mean(datastructActive(g).imgOcclRes(vecAxSp,famIdx(j),:,i)))*100;
            s(j) = subplot(1,length(famIdx),j);
            shadedErrorBar(vecAx,mean(traceFull,2)...
                ,std(traceFull,0,2)/sqrt(size(traceFull,2)), 'lineProps', 'k');
            shadedErrorBar(vecAx,mean(traceOccl,2)...
                ,std(traceOccl,0,2)/sqrt(size(traceOccl,2)), 'lineProps', 'r');
            if j ~= 1
                set(gca,'Visible','off')
            else
                xlabel('Time (s)')
                ylabel('Response')
            end
        end
        mn = min([s(:).YLim]);
        mx = max([s(:).YLim]);
        for j = 1:length(s)
            s(j).YLim = [mn mx]; s(j).XLim = [-1 3]; s(j).XTick = -1:1:3;
        end
        figClean
        pause
    end
end

%%
%%%%% lifetime sparseness
imgFullResPop = cat(4,datastructActive(:).imgFullRes);
imgOcclResPop = cat(4,datastructActive(:).imgOcclRes);

% Assuming 'responses' is a matrix where each row corresponds to a neuron's responses to n stimuli
varValFull = zeros(size(imgFullResPop,4),1);
varValOccl = zeros(size(imgFullResPop,4),1);

n = 4;
for j = 1:size(imgFullResPop,4)
    responses = squeeze(mean(imgFullResPop(vecAxSt,famIdx,:,j))-mean(imgFullResPop(vecAxSp,famIdx,:,j)));
    mean_response = mean(responses, 2);  % Mean response across stimuli
    varValFull(j) = 1 - ((sum(mean_response) / n)^2) / (sum(mean_response.^2) / n);    
    responses = squeeze(mean(imgOcclResPop(vecAxSt,famIdx,:,j))-mean(imgOcclResPop(vecAxSp,famIdx,:,j)));
    mean_response = mean(responses, 2);  % Mean response across stimuli
    varValOccl(j) = 1 - ((sum(mean_response) / n)^2) / (sum(mean_response.^2) / n);
end 

% Calculate lifetime sparseness
figure('Position',[680   481   300   497])
scatter([1 2],[nanmean(varValFull) nanmean(varValOccl)], 45, 'k', 'LineWidth', 2), hold on          
er = errorbar([1 2],[nanmean(varValFull) nanmean(varValOccl)], ...
    [calcSem(varValFull) calcSem(varValOccl)] ...
    ,[calcSem(varValFull) calcSem(varValOccl)]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0; xlim([0 3]), ylim([0 0.7])
ylabel('Sparseness'), figCleansel

if save_fig
    func_save_fig('L23_lifetimesparsenessFamActive')
end




%% SECTION TO PLOT NEURONS FOR IN PAPER, their traces
%% plot individual cells with shadederrobar and matched
clear selection

% % % % selection = [14 19 25 33 41 59 71 78 83 89 92 105 113 120]; %
% % % % selection = [33 78 11 20 25 54 105 101 114 89 14]; %
% % % % selection = [34 70 118 133];
% % % % selection = [33 78 11 20 54 105 101];

% FINAL NEURONS CHOSEN FOR PAPER:
% selection = [33 78 11 20 118 54 105 101]; % Neurons chosen for paper
% selection = [1, 65, 84, 90, 91, 3, 77, 56, 50];

% Build data arrays
fullDataPre = cat(4, datastructPre(:).imgFullRes);   % [frames x images x reps x neurons]
occlDataPre = cat(4, datastructPre(:).imgOcclRes);

fullDataTask = cat(4, datastructActive(:).imgFullRes);
occlDataTask = cat(4, datastructActive(:).imgOcclRes);

% Setup
nNeurons = size(fullDataPre, 4);
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
            traceFullPre = squeeze(fullDataPre(:, j, :, neuronIdx) - mean(fullDataPre(vecAxPreSp, j, :, neuronIdx)));
            traceOcclPre = squeeze(occlDataPre(:, j, :, neuronIdx) - mean(occlDataPre(vecAxPreSp, j, :, neuronIdx)));
            
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
%                 title(sprintf('ROI %d. ', neuronIdx));
%                 title(sprintf('selected %d. Mouse %s. match %d. Roi ID: %d. ', neuronIdx, mouseNamei, matchIDi, roiIDi(1)));
            end
        end

        for j = 1:nImgs
            % --- Task traces ---
            traceFullTask = squeeze(fullDataTask(:, j, :, neuronIdx) - mean(fullDataTask(vecAxTaskSp, j, :, neuronIdx)));
            traceOcclTask = squeeze(occlDataTask(:, j, :, neuronIdx) - mean(occlDataTask(vecAxTaskSp, j, :, neuronIdx)));

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
            traceFullPre = squeeze(fullDataPre(:, j, :, neuronIdx) - mean(fullDataPre(vecAxPreSp, j, :, neuronIdx)));
            traceOcclPre = squeeze(occlDataPre(:, j, :, neuronIdx) - mean(occlDataPre(vecAxPreSp, j, :, neuronIdx)));
            yMean = [yMean; mean(traceFullPre, 2); mean(traceOcclPre, 2)];
            ySEM  = [ySEM;  std(traceFullPre, 0, 2)/sqrt(size(traceFullPre, 2)); ...
                std(traceOcclPre, 0, 2)/sqrt(size(traceOcclPre, 2))];

            % Task
            traceFullTask = squeeze(fullDataTask(:, j, :, neuronIdx) - mean(fullDataTask(vecAxTaskSp, j, :, neuronIdx)));
            traceOcclTask = squeeze(occlDataTask(:, j, :, neuronIdx) - mean(occlDataTask(vecAxTaskSp, j, :, neuronIdx)));
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
        pause;
    end
end

if save_fig
    func_save_fig('L23_preActiveChronicExampleTraces_4')
end

%%

% ===== Local helper functions (work inside a script) =====
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
    sx = sign(x0 - cx); if sx == 0, sx = 1; end
    sy = sign(y0 - cy); if sy == 0, sy = 1; end
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

