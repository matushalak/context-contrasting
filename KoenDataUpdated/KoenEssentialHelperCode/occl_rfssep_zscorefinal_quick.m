% Data analysis for naive, expert and task occluded image stimuli
%
% 
% LDA analysis on population data from muckli experiments pre vs post training
%
%	Version History:
%	2022-02-14	Created by Koen Seignette
%   2022-07-22 update on some significance calculations and average traces
%   2023-05-16 updated for nov vs fam and for rfs combined
%   2026-04-17 Shortened a lot by Leander

clear all

warning on

% load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\dataPrePostGraySeparateRFs20230516.mat') % L23
% load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\datastructPrePostGrayL5_rfsSeparate.mat') % L5
% load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\PrePostGrayL23SeparateNewRFs.mat') % L23
% load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\datastructPrePostGrayL5_rfsSeparate.mat') % L5

% newest:
% % % % load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\PrePostGrayL23ChronicSeparateNewRFs.mat') % L23

% load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\PrePostGrayL5ChronicSeparateNewRFs.mat') % 5
load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\PrePostGrayL23ChronicSeparateNewRFsPupil.mat') % L23, with new pupil from Huub

%% Initialize and organise data
clearvars -except datastructPre datastructPost filenamesPre...
    filenamesPost filepathsPre filepathsPost nfiles doDecoding...
    imgNrs nImgs performanceChance nTrials nReps nBoots trainFrac 

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
vecAxRunSt = vecAx>0.2 & vecAx<1; % stim window
% vecAxSt = vecAx>0 & vecAx<0.2; % stim window
alphaVal = 0.9999999; % significance value for cells to be included
if nfiles == 6
    rsqThresh = 0.33; % 0.33 for L2/3, 0.15 for L5
else
    rsqThresh = 0.15; % 0.33 for L2/3, 0.15 for L5
end
% bratThresh = 1.5;
snrThresh = 4; % snr threshold for RF
useSpikingData = 0; % deconvolved (1) or df/f (0)

doZscore = true; % in case you want to work with zscored data instead of dff
smoothDecoding = false;

regressRun = false; % regress out running? Only for CaSigCorrected, not for spikes
runNan = false;
runThres = 2;

removeSaccade = false; % remove trials in which pupil moves?
veloThresh = 0.5; % threshold for calling a frame a movement frame

% for loop with decoding etc.
for i = 1:nfiles
    %[2 3 5] % slower after training
    %[1 4 6] % faster after training

    %%%%%%%%%% PRE DATASET %%%%%%%%%%

    % calculation of RF distances to occluder and inclusion criteria
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
%     onCrit = rfOnDist>rfDistVec & logical(rfOnGlmIncl);
%     offCrit = rfOffDist>rfDistVec & logical(rfOffGlmIncl);

    onCrit = rfOnDist>rfDistVec & logical(rfOnGlmIncl);
    offCrit = rfOffDist>rfDistVec & logical(rfOffGlmIncl);

    rfIncl = onCrit | offCrit; % either a good ON or OFF receptive field

    % responses sorted to match across mice
    Res = datastructPre(i).Res;

    %%% extra analysis: extract R2 for this and for shuffled data and show
    %%% that r2 for real data higher is than shuffled
    if regressRun
        tempTrace = Res.CaSigCorrected;
        % regress out running speed per trial
        clear r
        x = Res.speed; % run speed for this session
        for g = 1:size(tempTrace,3)
            y = squeeze(tempTrace(:,:,g)); % get trace for ROI
            lme = fitlm(x(:),y(:)); % model fit
%             lme = fitlm(zscore(x(:)),zscore(y(:))); % zscored then model
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

    if useSpikingData
        CaResSort = Res.CaDeconCorrected(:,dataSortidx,rfIncl); % reordering
    else
        if regressRun
            CaResSort = Res.CaSigCorrected(:,dataSortidx,rfIncl); % reordering, regressRun does subtract 1 already
        else
            CaResSort = Res.CaSigCorrected(:,dataSortidx,rfIncl)-1; %
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
    
    % % pupil data from normal Chris analysis
    % if isfield(Res, 'eye')
    %     CaResEye = Res.eye.sz(:,imgIdx); % pupil size data, subselected
    %     CaResEyeLoc = Res.eye.pos(:,imgIdx,:); % pupil location data, subselected
    % else
    %     CaResEye = nan([size(CaResSort,1), size(CaResSort,2)]);
    %     CaResEyeLoc = nan([size(CaResSort,1), size(CaResSort,2), 2]);
    % end

    % pupil data from facemap
    % extract trials first
    eventFrames = datastructPre(i).info.frame;
    framesBef = 30; % frames before stimulus
    framesAft = 93;

    % % pupil area data
    % pupilArea = datastructPre(i).pupil{1, 1}.area;
    % eventsEpochs = get_eventsEpochs(pupilArea,eventFrames, framesBef, framesAft);
    % CaResEye = eventsEpochs(:,imgIdx); % subselect trials
    % 
    % % pupil position data
    % pupilX = zscore(datastructPre(i).pupil{1, 1}.com(:,1));
    % pupilY = zscore(datastructPre(i).pupil{1, 1}.com(:,2));
    % xDataTrials = get_eventsEpochs(pupilX,eventFrames, framesBef, framesAft);
    % yDataTrials = get_eventsEpochs(pupilY,eventFrames, framesBef, framesAft);
    % CaResEyeLoc(:,:,1) = xDataTrials(:,imgIdx); % subselect trials for x data
    % CaResEyeLoc(:,:,2) = yDataTrials(:,imgIdx); % subselect trials for x data

    % % find eye movement frames
    % vx = diff(pupilX);  % velocity in x
    % vy = diff(pupilY);  % velocity in y
    % v = sqrt(vx.^2 + vy.^2); % Compute the overall velocity magnitude
    % v = v-median(v); % to make baseline at 0
    % eyeMovs = zeros(size(pupilX));
    % eyeMovs(find(v>veloThresh))=1; % eye movement frames
    % eyeMovMat = get_eventsEpochs(eyeMovs,eventFrames, framesBef, framesAft); % align to stims
    % CaResEyeMov = eyeMovMat(:,imgIdx); % subselect trials
    % velocityMat = get_eventsEpochs(v, eventFrames, framesBef, framesAft); % align to stims

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
    
    if ~isempty(runSpeed) % some mice might not have any neurons left
        runTrials = nanmean(runSpeed(vecAxRunSt,:,1))>runThres; % get trials in which average runspeed in vecAxSt > 1 cm/s
    else
        runTrials = [];
    end
    if runNan % in case of runtrial removing
        if ~isempty(runSpeed) % some mice might not have any neurons left
%             fracRunTrialsPre = sum(runTrials)/length(runTrials)  
%             runTrials = mean(runSpeed(vecAx>-0.5&vecAx<1.5,:,1))>runThres; % slightly bigger window for testing
            CaResSort(:,runTrials,:) = NaN; % remove those trials
        end
    end

    % define saccade trials as trials in which there was eye movement
    % anywhere in the stimulus period
    % sacTrials = mean(eyeMovMat(vecAxSt,:))>0;
    % 
    % if removeSaccade % in case of saccade trial removing
    %     CaResSort(:,sacTrials,:) = NaN; % remove those trials
    % end

    % trace matrices (frames x imgs x trials x rois)
    imgFullRes = zeros(size(CaResSort,1), nImgs, nTrials, size(CaResSort,3)); % pre-allocate
    imgOcclRes = zeros(size(CaResSort,1), nImgs, nTrials, size(CaResSort,3)); % pre-allocate
    % eyeFullRes = zeros(size(CaResSort,1), nImgs, nTrials); % pre-allocate
    % eyeOcclRes = zeros(size(CaResSort,1), nImgs, nTrials); % pre-allocate
    % eyeFullPos = zeros(size(CaResSort,1), nImgs, nTrials,2); % pre-allocate
    % eyeOcclPos = zeros(size(CaResSort,1), nImgs, nTrials,2); % pre-allocate
    % eyeFullMov = zeros(size(CaResSort,1), nImgs, nTrials);
    % eyeOcclMov = zeros(size(CaResSort,1), nImgs, nTrials);
    % eyeFullVel = zeros(size(CaResSort,1), nImgs, nTrials);
    % eyeOcclVel = zeros(size(CaResSort,1), nImgs, nTrials);
    for j = 1:nImgs
        imgIdxFull = find(matTrialTypesIncl(1,:)==imgNrs(j) & matTrialTypesIncl(2,:)==0);
        imgIdxOccl = find(matTrialTypesIncl(1,:)==imgNrs(j) & matTrialTypesIncl(2,:)==1);
        imgFullRes(:,j,:,:) = CaResSort(:,imgIdxFull,:);
        imgOcclRes(:,j,:,:) = CaResSort(:,imgIdxOccl,:);
        % eyeFullPos(:,j,:,1) = CaResEyeLoc(:,imgIdxFull,1); % pupil size res x
        % eyeFullPos(:,j,:,2) = CaResEyeLoc(:,imgIdxFull,2); % pupil size res x
        % eyeOcclPos(:,j,:,1) = CaResEyeLoc(:,imgIdxOccl,1); % pupil size res x
        % eyeOcclPos(:,j,:,2) = CaResEyeLoc(:,imgIdxOccl,2); % pupil size res x
        % eyeFullRes(:,j,:) = CaResEye(:,imgIdxFull); % pupil size full stims
        % eyeOcclRes(:,j,:) = CaResEye(:,imgIdxOccl); % pupil size full stims
        % eyeFullMov(:,j,:) = CaResEyeMov(:,imgIdxFull); % pupil movement
        % eyeOcclMov(:,j,:) = CaResEyeMov(:,imgIdxOccl); % pupil movement
        % eyeFullVel(:,j,:) = velocityMat(:,imgIdxFull); % pupil movement
        % eyeOcclVel(:,j,:) = velocityMat(:,imgIdxOccl); % pupil movement
    end

    % data for decoding
    if smoothDecoding
        temp = smoothG(CaResSort,1);
        matData = squeeze(nanmean(temp(vecAxSt,:,:)))-squeeze(nanmean(temp(vecAxSp,:,:)));
    else
        matData = squeeze(nanmean(CaResSort(vecAxSt,:,:)))-squeeze(nanmean(CaResSort(vecAxSp,:,:)));
    end

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
        scatFull = squeeze(nanmean(fullSign(vecAxSt,:)))-squeeze(nanmean(fullSign(vecAxSp,:)));
        scatOccl = squeeze(nanmean(occlSign(vecAxSt,:)))-squeeze(nanmean(occlSign(vecAxSp,:)));
    else
        scatFull = [];
        scatOccl = [];
        fullSign = [];
        occlSign = [];
    end

    datastructPre(i).rfIncl = rfIncl;
    datastructPre(i).CaResSort = CaResSort;
    datastructPre(i).runSpeed = runSpeed;
    datastructPre(i).rfOnGlmIncl = rfOnGlmIncl;
    datastructPre(i).rfOffGlmIncl = rfOffGlmIncl;
    datastructPre(i).rfOnDist = rfOnDist;
    datastructPre(i).rfOffDist = rfOffDist;
    datastructPre(i).onCrit = onCrit;
    datastructPre(i).offCrit = offCrit;
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
    % datastructPre(i).eyeFullRes = eyeFullRes;
    % datastructPre(i).eyeOcclRes = eyeOcclRes;
    % datastructPre(i).eyeFullPos = eyeFullPos;
    % datastructPre(i).eyeOcclPos = eyeOcclPos;  
    % datastructPre(i).eyeFullMov = eyeFullMov;
    % datastructPre(i).eyeOcclMov = eyeOcclMov;  
    % datastructPre(i).eyeFullVel = eyeFullVel;
    % datastructPre(i).eyeOcclVel = eyeOcclVel;  

    datastructPre(i).matData = matData;
    datastructPre(i).scatFull = scatFull;
    datastructPre(i).scatOccl = scatOccl;
%     if runNan
        datastructPre(i).runTrials = runTrials;
%     end
    

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
    rfIncl = onCrit | offCrit; % either a good ON or OFF receptive field
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
%             lme = fitlm(zscore(x(:)),zscore(y(:))); % zscored then model
            r(:,g) = lme.Residuals.Raw; % get residuals for this ROI
            scatter(x(:),y(:))
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

    if useSpikingData
        CaResSort = Res.CaDeconCorrected(:,dataSortidx,rfIncl); % reordering
    else
        if regressRun
            CaResSort = Res.CaSigCorrected(:,dataSortidx,rfIncl); % reordering, regressRun does subtract 1 already
        else
            CaResSort = Res.CaSigCorrected(:,dataSortidx,rfIncl)-1; %
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

    % if isfield(Res, 'eye')
    %     CaResEye = Res.eye.sz(:,imgIdx); % pupil size data, subselected
    % else
    %     CaResEye = nan([size(CaResSort,1), size(CaResSort,2)]);
    % end
    
    % pupil data from facemap
    % extract trials first
    eventFrames = datastructPost(i).info.frame;
    framesBef = 30; % frames before stimulus
    framesAft = 93;

    % % pupil area data
    % pupilArea = datastructPost(i).pupil{1, 1}.area;
    % eventsEpochs = get_eventsEpochs(pupilArea,eventFrames, framesBef, framesAft);
    % CaResEye = eventsEpochs(:,imgIdx); % subselect trials

    % % pupil position data
    % pupilX = zscore(datastructPost(i).pupil{1, 1}.com(:,1));
    % pupilY = zscore(datastructPost(i).pupil{1, 1}.com(:,2));
    % xDataTrials = get_eventsEpochs(pupilX,eventFrames, framesBef, framesAft);
    % yDataTrials = get_eventsEpochs(pupilY,eventFrames, framesBef, framesAft);
    % CaResEyeLoc(:,:,1) = xDataTrials(:,imgIdx); % subselect trials for x data
    % CaResEyeLoc(:,:,2) = yDataTrials(:,imgIdx); % subselect trials for x data
    % 
    % % find eye movement frames
    % vx = diff(pupilX);  % velocity in x
    % vy = diff(pupilY);  % velocity in y
    % v = sqrt(vx.^2 + vy.^2); % Compute the overall velocity magnitude
    % v = v-median(v); % to make baseline at 0
    % eyeMovs = zeros(size(pupilX));
    % eyeMovs(find(v>veloThresh))=1; % eye movement frames
    % eyeMovMat = get_eventsEpochs(eyeMovs,eventFrames, framesBef, framesAft); % align to stims
    % CaResEyeMov = eyeMovMat(:,imgIdx); % subselect trials
    % velocityMat = get_eventsEpochs(v, eventFrames, framesBef, framesAft); % align to stims

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
    
    if ~isempty(runSpeed) % some mice might not have any neurons left
        runTrials = nanmean(runSpeed(vecAxRunSt,:,1))>runThres; % get trials in which average runspeed in vecAxSt > 1 cm/s
    else
        runTrials = [];
    end
    if runNan % in case of runtrial removing
        if ~isempty(runSpeed) % some mice might not have any neurons left
%             fracRunTrialsPre = sum(runTrials)/length(runTrials)  
%             runTrials = mean(runSpeed(vecAx>-0.5&vecAx<1.5,:,1))>runThres; % slightly bigger window for testing
            CaResSort(:,runTrials,:) = NaN; % remove those trials
        end
    end

    % % define saccade trials as trials in which there was eye movement
    % % anywhere in the stimulus period
    % sacTrials = mean(eyeMovMat(vecAxSt,:))>0;
    % 
    % if removeSaccade % in case of saccade trial removing
    %     CaResSort(:,sacTrials,:) = NaN; % remove those trials
    % end

    % trace matrices (frames x imgs x trials x rois)
    imgFullRes = zeros(size(CaResSort,1), nImgs, nTrials, size(CaResSort,3)); % pre-allocate
    imgOcclRes = zeros(size(CaResSort,1), nImgs, nTrials, size(CaResSort,3)); % pre-allocate
    % eyeFullRes = zeros(size(CaResSort,1), nImgs, nTrials); % pre-allocate
    % eyeOcclRes = zeros(size(CaResSort,1), nImgs, nTrials); % pre-allocate
    % eyeFullPos = zeros(size(CaResSort,1), nImgs, nTrials,2); % pre-allocate
    % eyeOcclPos = zeros(size(CaResSort,1), nImgs, nTrials,2); % pre-allocate
    % eyeFullMov = zeros(size(CaResSort,1), nImgs, nTrials);
    % eyeOcclMov = zeros(size(CaResSort,1), nImgs, nTrials);
    % eyeFullVel = zeros(size(CaResSort,1), nImgs, nTrials);
    % eyeOcclVel = zeros(size(CaResSort,1), nImgs, nTrials);
    for j = 1:nImgs
        imgIdxFull = find(matTrialTypesIncl(1,:)==imgNrs(j) & matTrialTypesIncl(2,:)==0);
        imgIdxOccl = find(matTrialTypesIncl(1,:)==imgNrs(j) & matTrialTypesIncl(2,:)==1);
        imgFullRes(:,j,:,:) = CaResSort(:,imgIdxFull,:);
        imgOcclRes(:,j,:,:) = CaResSort(:,imgIdxOccl,:);
        % eyeFullPos(:,j,:,1) = CaResEyeLoc(:,imgIdxFull,1); % pupil size res x
        % eyeFullPos(:,j,:,2) = CaResEyeLoc(:,imgIdxFull,2); % pupil size res x
        % eyeOcclPos(:,j,:,1) = CaResEyeLoc(:,imgIdxOccl,1); % pupil size res x
        % eyeOcclPos(:,j,:,2) = CaResEyeLoc(:,imgIdxOccl,2); % pupil size res x
        % eyeFullRes(:,j,:) = CaResEye(:,imgIdxFull); % pupil size full stims
        % eyeOcclRes(:,j,:) = CaResEye(:,imgIdxOccl); % pupil size full stims
        % eyeFullMov(:,j,:) = CaResEyeMov(:,imgIdxFull); % pupil movement
        % eyeOcclMov(:,j,:) = CaResEyeMov(:,imgIdxOccl); % pupil movement
        % eyeFullVel(:,j,:) = velocityMat(:,imgIdxFull); % pupil movement
        % eyeOcclVel(:,j,:) = velocityMat(:,imgIdxOccl); % pupil movement
    end
    % data for decoding
    if smoothDecoding
        temp = smoothG(CaResSort,1);
        matData = squeeze(nanmean(temp(vecAxSt,:,:)))-squeeze(nanmean(temp(vecAxSp,:,:)));
    else
        matData = squeeze(nanmean(CaResSort(vecAxSt,:,:)))-squeeze(nanmean(CaResSort(vecAxSp,:,:)));
    end

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
        scatFull = squeeze(nanmean(fullSign(vecAxSt,:)))-squeeze(nanmean(fullSign(vecAxSp,:)));
        scatOccl = squeeze(nanmean(occlSign(vecAxSt,:)))-squeeze(nanmean(occlSign(vecAxSp,:)));
    else
        scatFull = [];
        scatOccl = [];
        fullSign = [];
        occlSign = [];
    end

    datastructPost(i).rfIncl = rfIncl;
    datastructPost(i).CaResSort = CaResSort;
    datastructPost(i).runSpeed = runSpeed;
    datastructPost(i).rfOnGlmIncl = rfOnGlmIncl;
    datastructPost(i).rfOffGlmIncl = rfOffGlmIncl;
    datastructPost(i).rfOnDist = rfOnDist;
    datastructPost(i).rfOffDist = rfOffDist;
    datastructPost(i).onCrit = onCrit;
    datastructPost(i).offCrit = offCrit;
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
    % datastructPost(i).eyeFullRes = eyeFullRes;
    % datastructPost(i).eyeOcclRes = eyeOcclRes;
    % datastructPost(i).eyeFullPos = eyeFullPos;
    % datastructPost(i).eyeOcclPos = eyeOcclPos;  
    % datastructPost(i).eyeFullMov = eyeFullMov;
    % datastructPost(i).eyeOcclMov = eyeOcclMov; 
    % datastructPost(i).eyeFullVel = eyeFullVel;
    % datastructPost(i).eyeOcclVel = eyeOcclVel;  
    datastructPost(i).matData = matData;
    datastructPost(i).scatFull = scatFull;
    datastructPost(i).scatOccl = scatOccl;

%     if runNan
        datastructPost(i).runTrials = runTrials; %#ok<*SAGROW> 
%     end
    disp(i)
end


%% in case you calculated responses to 6 images, plot pre vs post (separate by image type)
% color pallets for plotting
col1 = [0,0,0]; % black
col2 = [131, 197, 190]/255; % blue/greenish
% col3 = [0,0,1]; % blue
% col4 = [1,0,0]; % red
% col5 = [202, 103, 2]/255; % red brownish

save_fig = false;

famIdx = [1 2 4 5];
novIdx = [3 6];
% famIdx = [1 2];
% novIdx = [3 6];

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

% === Pre: Full, Occluded, Familiar and Novel ===

% AVERAGED responses (original)
imgFullResFamPre = squeeze(nanmean(imgFullResMnPopPre(:,famIdx,:),2));
imgFullResFamPreBsl = imgFullResFamPre - nanmean(imgFullResFamPre(vecAxSp,:));
imgFullResNovPre = squeeze(nanmean(imgFullResMnPopPre(:,novIdx,:),2));
imgFullResNovPreBsl = imgFullResNovPre - nanmean(imgFullResNovPre(vecAxSp,:));

imgOcclResFamPre = squeeze(nanmean(imgOcclResMnPopPre(:,famIdx,:),2));
imgOcclResFamPreBsl = imgOcclResFamPre - nanmean(imgOcclResFamPre(vecAxSp,:));
imgOcclResNovPre = squeeze(nanmean(imgOcclResMnPopPre(:,novIdx,:),2));
imgOcclResNovPreBsl = imgOcclResNovPre - nanmean(imgOcclResNovPre(vecAxSp,:));

% PER-IMAGE responses (new)
imgFullResFamPre_each = squeeze(imgFullResMnPopPre(:,famIdx,:));  % [neurons x 4 x time]
imgFullResFamPreBsl_each = imgFullResFamPre_each - nanmean(imgFullResFamPre_each(vecAxSp,:,:), 1);

imgFullResNovPre_each = squeeze(imgFullResMnPopPre(:,novIdx,:));
imgFullResNovPreBsl_each = imgFullResNovPre_each - nanmean(imgFullResNovPre_each(vecAxSp,:,:), 1);

imgOcclResFamPre_each = squeeze(imgOcclResMnPopPre(:,famIdx,:));
imgOcclResFamPreBsl_each = imgOcclResFamPre_each - nanmean(imgOcclResFamPre_each(vecAxSp,:,:), 1);

imgOcclResNovPre_each = squeeze(imgOcclResMnPopPre(:,novIdx,:));
imgOcclResNovPreBsl_each = imgOcclResNovPre_each - nanmean(imgOcclResNovPre_each(vecAxSp,:,:), 1);

% === Post: Full, Occluded, Familiar and Novel ===

% AVERAGED responses (original)
imgFullResFamPost = squeeze(nanmean(imgFullResMnPopPost(:,famIdx,:),2));
imgFullResFamPostBsl = imgFullResFamPost - nanmean(imgFullResFamPost(vecAxSp,:));
imgFullResNovPost = squeeze(nanmean(imgFullResMnPopPost(:,novIdx,:),2));
imgFullResNovPostBsl = imgFullResNovPost - nanmean(imgFullResNovPost(vecAxSp,:));

imgOcclResFamPost = squeeze(nanmean(imgOcclResMnPopPost(:,famIdx,:),2));
imgOcclResFamPostBsl = imgOcclResFamPost - nanmean(imgOcclResFamPost(vecAxSp,:));
imgOcclResNovPost = squeeze(nanmean(imgOcclResMnPopPost(:,novIdx,:),2));
imgOcclResNovPostBsl = imgOcclResNovPost - nanmean(imgOcclResNovPost(vecAxSp,:));

% PER-IMAGE responses (new)
imgFullResFamPost_each = squeeze(imgFullResMnPopPost(:,famIdx,:));
imgFullResFamPostBsl_each = imgFullResFamPost_each - nanmean(imgFullResFamPost_each(vecAxSp,:,:), 1);

imgFullResNovPost_each = squeeze(imgFullResMnPopPost(:,novIdx,:));
imgFullResNovPostBsl_each = imgFullResNovPost_each - nanmean(imgFullResNovPost_each(vecAxSp,:,:), 1);

imgOcclResFamPost_each = squeeze(imgOcclResMnPopPost(:,famIdx,:));
imgOcclResFamPostBsl_each = imgOcclResFamPost_each - nanmean(imgOcclResFamPost_each(vecAxSp,:,:), 1);

imgOcclResNovPost_each = squeeze(imgOcclResMnPopPost(:,novIdx,:));
imgOcclResNovPostBsl_each = imgOcclResNovPost_each - nanmean(imgOcclResNovPost_each(vecAxSp,:,:), 1);

% === Population Response Measures ===

% AVERAGED across familiar/novel images
scatFullFamPopPre  = nanmean(imgFullResFamPreBsl(vecAxSt,:));
scatFullNovPopPre  = nanmean(imgFullResNovPreBsl(vecAxSt,:));
scatOcclFamPopPre  = nanmean(imgOcclResFamPreBsl(vecAxSt,:));
scatOcclNovPopPre  = nanmean(imgOcclResNovPreBsl(vecAxSt,:));

scatFullFamPopPost = nanmean(imgFullResFamPostBsl(vecAxSt,:));
scatFullNovPopPost = nanmean(imgFullResNovPostBsl(vecAxSt,:));
scatOcclFamPopPost = nanmean(imgOcclResFamPostBsl(vecAxSt,:));
scatOcclNovPopPost = nanmean(imgOcclResNovPostBsl(vecAxSt,:));

% PER-IMAGE population responses (size: [4 x neurons])
scatFullFamPopPre_each  = squeeze(nanmean(imgFullResFamPreBsl_each(vecAxSt,:,:), 1));
scatFullNovPopPre_each  = squeeze(nanmean(imgFullResNovPreBsl_each(vecAxSt,:,:), 1));
scatOcclFamPopPre_each  = squeeze(nanmean(imgOcclResFamPreBsl_each(vecAxSt,:,:), 1));
scatOcclNovPopPre_each  = squeeze(nanmean(imgOcclResNovPreBsl_each(vecAxSt,:,:), 1));

scatFullFamPopPost_each = squeeze(nanmean(imgFullResFamPostBsl_each(vecAxSt,:,:), 1));
scatFullNovPopPost_each = squeeze(nanmean(imgFullResNovPostBsl_each(vecAxSt,:,:), 1));
scatOcclFamPopPost_each = squeeze(nanmean(imgOcclResFamPostBsl_each(vecAxSt,:,:), 1));
scatOcclNovPopPost_each = squeeze(nanmean(imgOcclResNovPostBsl_each(vecAxSt,:,:), 1));


% cut off value just for plotting purposes
scatFullFamPopPreCut = scatFullFamPopPre; 
scatOcclFamPopPreCut = scatOcclFamPopPre; 
scatFullFamPopPostCut = scatFullFamPopPost; 
scatOcclFamPopPostCut = scatOcclFamPopPost; 
scatFullNovPopPreCut = scatFullNovPopPre; 
scatOcclNovPopPreCut = scatOcclNovPopPre; 
scatFullNovPopPostCut = scatFullNovPopPost; 
scatOcclNovPopPostCut = scatOcclNovPopPost; 

% if nfiles == 5
%     mnValCut = -0.5; % min val for cutting for plotting
%     mxValCut = 5; % max val for cutting for plotting
%     scatFullFamPopPreCut(scatFullFamPopPreCut>mxValCut)=mxValCut;scatFullFamPopPreCut(scatFullFamPopPreCut<mnValCut)=mnValCut;
%     scatOcclFamPopPreCut(scatOcclFamPopPreCut>mxValCut)=mxValCut;scatOcclFamPopPreCut(scatOcclFamPopPreCut<mnValCut)=mnValCut;
%     scatFullFamPopPostCut(scatFullFamPopPostCut>mxValCut)=mxValCut; scatFullFamPopPostCut(scatFullFamPopPostCut<mnValCut)=mnValCut;
%     scatOcclFamPopPostCut(scatOcclFamPopPostCut>mxValCut)=mxValCut; scatOcclFamPopPostCut(scatOcclFamPopPostCut<mnValCut)=mnValCut;
%     scatFullNovPopPreCut(scatFullNovPopPreCut>mxValCut)=mxValCut;scatFullNovPopPreCut(scatFullNovPopPreCut<mnValCut)=mnValCut;
%     scatOcclNovPopPreCut(scatOcclNovPopPreCut>mxValCut)=mxValCut;scatOcclNovPopPreCut(scatOcclNovPopPreCut<mnValCut)=mnValCut;
%     scatFullNovPopPostCut(scatFullNovPopPostCut>mxValCut)=mxValCut;scatFullNovPopPostCut(scatFullNovPopPostCut<mnValCut)=mnValCut;
%     scatOcclNovPopPostCut(scatOcclNovPopPostCut>mxValCut)=mxValCut;scatOcclNovPopPostCut(scatOcclNovPopPostCut<mnValCut)=mnValCut;
% elseif nfiles == 6
    mnValCut = -0.5; % min val for cutting for plotting
    mxValCut = 2.5; % max val for cutting for plotting
    scatFullFamPopPreCut(scatFullFamPopPreCut>mxValCut)=mxValCut+0.5;scatFullFamPopPreCut(scatFullFamPopPreCut<mnValCut)=mnValCut-0.5;
    scatOcclFamPopPreCut(scatOcclFamPopPreCut>mxValCut)=mxValCut+0.5;scatOcclFamPopPreCut(scatOcclFamPopPreCut<mnValCut)=mnValCut-0.5;
    scatFullFamPopPostCut(scatFullFamPopPostCut>mxValCut)=mxValCut+0.5; scatFullFamPopPostCut(scatFullFamPopPostCut<mnValCut)=mnValCut-0.5;
    scatOcclFamPopPostCut(scatOcclFamPopPostCut>mxValCut)=mxValCut+0.5; scatOcclFamPopPostCut(scatOcclFamPopPostCut<mnValCut)=mnValCut-0.5;
    scatFullNovPopPreCut(scatFullNovPopPreCut>mxValCut)=mxValCut+0.5;scatFullNovPopPreCut(scatFullNovPopPreCut<mnValCut)=mnValCut-0.5;
    scatOcclNovPopPreCut(scatOcclNovPopPreCut>mxValCut)=mxValCut+0.5;scatOcclNovPopPreCut(scatOcclNovPopPreCut<mnValCut)=mnValCut-0.5;
    scatFullNovPopPostCut(scatFullNovPopPostCut>mxValCut)=mxValCut+0.5;scatFullNovPopPostCut(scatFullNovPopPostCut<mnValCut)=mnValCut-0.5;
    scatOcclNovPopPostCut(scatOcclNovPopPostCut>mxValCut)=mxValCut+0.5;scatOcclNovPopPostCut(scatOcclNovPopPostCut<mnValCut)=mnValCut-0.5;
% end

sz = 8;
% cPre = [0.2 0.2 0.2];
% cPost = col2;
cPre = [0 0 0];
cPost = [0 0 0];

% plot traces and scatters in one figure
% figure('Position', [87         278        1635         673], 'Renderer', 'painters')
figure('Position', [87         278        1635         673])
clear t s
% traces
t(1) = subplot(2,5,1);
shadedErrorBar(vecAx,nanmean(imgFullResFamPreBsl,2)...
    ,nanstd(imgFullResFamPreBsl,0,2)/sqrt(size(imgFullResFamPreBsl,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAx,nanmean(imgOcclResFamPreBsl,2)...
    ,nanstd(imgOcclResFamPreBsl,0,2)/sqrt(size(imgOcclResFamPreBsl,2)), 'lineProps', 'r');
xlabel('Time (s)'), ylabel('Response'), xticks(-1:1:3), xlim([-1 3]), title('Pre Fam'), 
t(2) = subplot(2,5,2);
shadedErrorBar(vecAx,nanmean(imgFullResFamPostBsl,2)...
    ,nanstd(imgFullResFamPostBsl,0,2)/sqrt(size(imgFullResFamPostBsl,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAx,nanmean(imgOcclResFamPostBsl,2)...
    ,nanstd(imgOcclResFamPostBsl,0,2)/sqrt(size(imgOcclResFamPostBsl,2)), 'lineProps', 'r');
xlabel('Time (s)'), title('Post Fam'), 
t(3) = subplot(2,5,3);
shadedErrorBar(vecAx,nanmean(imgFullResNovPreBsl,2)...
    ,nanstd(imgFullResNovPreBsl,0,2)/sqrt(size(imgFullResNovPreBsl,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAx,nanmean(imgOcclResNovPreBsl,2)...
    ,nanstd(imgOcclResNovPreBsl,0,2)/sqrt(size(imgOcclResNovPreBsl,2)), 'lineProps', 'r');
xlabel('Time (s)'), xticks(-1:1:3), xlim([-1 3]), title('Pre Nov'), 
t(4) = subplot(2,5,4);
shadedErrorBar(vecAx,nanmean(imgFullResNovPostBsl,2)...
    ,nanstd(imgFullResNovPostBsl,0,2)/sqrt(size(imgFullResNovPostBsl,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAx,nanmean(imgOcclResNovPostBsl,2)...
    ,nanstd(imgOcclResNovPostBsl,0,2)/sqrt(size(imgOcclResNovPostBsl,2)), 'lineProps', 'r');
xlabel('Time (s)'), title('Post Nov'), 
% scatters
s(1) = subplot(2,5,6);
scatter(scatFullFamPopPreCut, scatOcclFamPopPreCut, sz, cPre, 'filled'); refline(1), ylabel('Occl'), xlabel('Full'),
s(2) = subplot(2,5,7);
scatter(scatFullFamPopPostCut,scatOcclFamPopPostCut , sz, cPost, 'filled'); refline(1), 
s(3) = subplot(2,5,8);
scatter(scatFullNovPopPreCut, scatOcclNovPopPreCut, sz, cPre, 'filled'); refline(1), 
s(4) = subplot(2,5,9);
scatter(scatFullNovPopPostCut, scatOcclNovPopPostCut, sz, cPost, 'filled'); refline(1), 
% nanmean box plot
subplot(2,5,5)
boxchart([ones(size(scatFullFamPopPre)), ones(size(scatFullFamPopPost))+1, ...
    ones(size(scatFullNovPopPre))+2, ones(size(scatFullNovPopPost))+3, ...
    ones(size(scatOcclFamPopPre))+5, ones(size(scatOcclFamPopPost))+6, ...
    ones(size(scatOcclNovPopPre))+7, ones(size(scatOcclNovPopPost))+8], ...
    [scatFullFamPopPre, scatFullFamPopPost, scatFullNovPopPre, ...
    scatFullNovPopPost, scatOcclFamPopPre, scatOcclFamPopPost...
    scatOcclNovPopPre, scatOcclNovPopPost], 'MarkerStyle','none'), hold on
xlim([0 10]), ylabel('Response'), xticks([1 2 3 4 5 6 7 8]);% if nfiles == 6, ylim([-5 10]), elseif nfiles == 5, ylim([-5 35]), end
xticklabels({'PreFamFull', 'PostFamFull','PreNovFull', 'PostNovFull', ...
    'PreFamOccl', 'PostFamOccl','PreNovOccl', 'PostNovOccl'}), xtickangle(45), 
% nanmean scat/bar
subplot(2,5,10)
scatter([1 2 3 4],[nanmean(scatFullFamPopPre) nanmean(scatFullFamPopPost) nanmean(scatFullNovPopPre) nanmean(scatFullNovPopPost)], 35, 'k', 'LineWidth', 2), hold on
scatter([6 7 8 9],[nanmean(scatOcclFamPopPre) nanmean(scatOcclFamPopPost) nanmean(scatOcclNovPopPre) nanmean(scatOcclNovPopPost)], 35, 'r', 'LineWidth', 2)                
er = errorbar([1 2 3 4],[nanmean(scatFullFamPopPre) nanmean(scatFullFamPopPost) nanmean(scatFullNovPopPre) nanmean(scatFullNovPopPost)], ...
    [calcSem(scatFullFamPopPre) calcSem(scatFullFamPopPost) calcSem(scatFullNovPopPre) calcSem(scatFullNovPopPost)] ...
    ,[calcSem(scatFullFamPopPre) calcSem(scatFullFamPopPost) calcSem(scatFullNovPopPre) calcSem(scatFullNovPopPost)]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
er = errorbar([6 7 8 9],[nanmean(scatOcclFamPopPre) nanmean(scatOcclFamPopPost) nanmean(scatOcclNovPopPre) nanmean(scatOcclNovPopPost)], ...
    [calcSem(scatOcclFamPopPre) calcSem(scatOcclFamPopPost) calcSem(scatOcclNovPopPre) calcSem(scatOcclNovPopPost)] ...
    ,[calcSem(scatOcclFamPopPre) calcSem(scatOcclFamPopPost) calcSem(scatOcclNovPopPre) calcSem(scatOcclNovPopPost)]);    
er.Color = [1 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 10]), ylabel('Response'), xticks([1 2 3 4 5 6 7 8]);% if nfiles == 6, ylim([0 5]), elseif nfiles == 5, ylim([0 7]), end
xticklabels({'PreFamFull', 'PostFamFull','PreNovFull', 'PostNovFull', ...
    'PreFamOccl', 'PostFamOccl','PreNovOccl', 'PostNovOccl'}), xtickangle(45), 


% Adjusting y-axes for subplots 1-4
yMax = max([ylim(t(1)), ylim(t(2)), ylim(t(3)), ylim(t(4))]);
yMin = min([ylim(t(1)), ylim(t(2)), ylim(t(3)), ylim(t(4))]);
% set(t(1:4), 'YLim', [yMin yMax]);
if nfiles == 6
    set(t(1:4), 'YLim', [-0.1 0.6]);
elseif nfiles == 5
    set(t(1:4), 'YLim', [-0.1 1]);
end
set(t(1:4), 'XLim', [-1 3])

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

if save_fig
    func_save_fig('L23_traceAndScatterAndBoxSeparate')
    func_save_fig('L5_traceAndScatterAndBoxSeparate')
end

%
sz = 8;

figure('Position', [ 87         278        1635         551])
clear t s
s(1) = subplot(2,5,1);
scatter(scatOcclFamPopPostCut, scatOcclNovPopPostCut, sz, cPre, 'filled'); refline(1), xlabel('Occl Fam'), ylabel('Occl Nov'),
s(2) = subplot(2,5,2);
scatter(scatFullFamPopPostCut, scatFullNovPopPostCut, sz, cPre, 'filled'); refline(1), xlabel('Full Fam'), ylabel('Full Nov'),
s(3) = subplot(2,5,3);
scatter(scatOcclFamPopPostCut, scatFullNovPopPostCut, sz, cPre, 'filled'); refline(1), xlabel('Occl Fam'), ylabel('Full Nov'), 

if nfiles == 6
    for j = 1:length(s)
        s(j).YLim = [-1 3]; s(j).YTick = -1:1:3; s(j).XLim = [-1 3]; s(j).XTick = -1:1:3;
    end
elseif nfiles == 5
    for j = 1:length(s)
%         s(j).YLim = [-1 3]; s(j).YTick = -1:0.5:3; s(j).XLim = [-1 3]; s(j).XTick = -1:0.5:3;
        s(j).YLim = [-0.5 3]; s(j).YTick = -0.5:0.5:3; s(j).XLim = [-0.5 3]; s(j).XTick = -0.5:0.5:3;
    end
end

if save_fig
    func_save_fig('L23_scattersFamVsNov')
    func_save_fig('L5_scattersFamVsNov')
end



figure('Position', [ 87         278        1635         551])
clear t s
s(1) = subplot(1,2,1);
scatter(scatFullNovPopPreCut, scatOcclNovPopPreCut, 25, 'k', 'filled'); refline(1), xlabel('Full nov pre'), ylabel('Occl nov pre'),
s(2) = subplot(1,2,2);
scatter(scatFullNovPopPostCut, scatOcclNovPopPostCut, 25, 'k', 'filled'); refline(1), xlabel('Full nov post'), ylabel('Occl nov post'),

if save_fig
    func_save_fig('L23_scattersNovNOvsOprepost')
    func_save_fig('L5_scattersNovNOvsOprepost')
end



%%%%%% snake plots / imagesc of all cells, average responses
% sort on trace of preference pre training fam
traceToSortFamPre = imgFullResFamPreBsl;
[MniPre] = nanmean(traceToSortFamPre(vecAxSt,:));
[~,RsortedMnFamPre] = sort(MniPre,'descend');
[~, MxiPre] = max(traceToSortFamPre);
[~,RsortedMxFamPre] = sort(MxiPre,'ascend');

% sort on trace of preference pre training nov
traceToSortNovPre = imgFullResNovPreBsl;
[MniPre] = nanmean(traceToSortNovPre(vecAxSt,:));
[~,RsortedMnNovPre] = sort(MniPre,'descend');
[~, MxiPre] = max(traceToSortNovPre);
[~,RsortedMxNovPre] = sort(MxiPre,'ascend');

% sort on trace of preference post training fam
traceToSortFamPost = imgFullResFamPostBsl;
[MniPost] = nanmean(traceToSortFamPost(vecAxSt,:));
[~,RsortedMnFamPost] = sort(MniPost,'descend');
[~, MxiPost] = max(traceToSortFamPost);
[~,RsortedMxFamPost] = sort(MxiPost,'ascend');

% sort on trace of preference post training nov
traceToSortNovPost = imgFullResNovPostBsl;
[MniPost] = nanmean(traceToSortNovPost(vecAxSt,:));
[~,RsortedMnNovPost] = sort(MniPost,'descend');
[~, MxiPost] = max(traceToSortNovPost);
[~,RsortedMxNovPost] = sort(MxiPost,'ascend');

% plot with each condition in separate subplot, axes are similar scaling
clear p
figure('Position', [76         263        1577         426])
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
%     mn = -5; mx = 20;
%     set(p, 'CLim', [mn, mx]); % for L2/3
% elseif nfiles == 5
%     mn = -5; mx = 20;
%     set(p, 'CLim', [mn, mx]); % for L5
% end
if nfiles == 6
    mn = -0.4; mx = 3;
    set(p, 'CLim', [mn, mx]); % for L2/3
elseif nfiles == 5
    mn = -0.4; mx = 3;
    set(p, 'CLim', [mn, mx]); % for L5
end
% colormap(flipud(gray))
colormap hot
subplot(1,9,5)
% axis off, colormap(flipud(gray)), caxis([mn mx]), colorbar
axis off, colormap hot, caxis([mn mx]), colorbar

if save_fig
    func_save_fig('L23_ImagescSeparate')
    func_save_fig('L5_ImagescSeparate')
end

% siFamPre = scatFullFamPopPre-scatOcclFamPopPre;
% siFamPost = scatFullFamPopPost-scatOcclFamPopPost;
% siNovPre = scatFullNovPopPre-scatOcclNovPopPre;
% siNovPost = scatFullNovPopPost-scatOcclNovPopPost;
% 
% colors = cmapL([0 0 1;0 0 0; 1 0 0], 256);
% lims = [-15 15];
% siFamPreColors = squeeze(SetLimits(siFamPre, lims, colors));
% siFamPostColors = squeeze(SetLimits(siFamPost, lims, colors));
% siNovPreColors = squeeze(SetLimits(siNovPre, lims, colors));
% siNovPostColors = squeeze(SetLimits(siNovPost, lims, colors));
% 
% siFamPre(siFamPre<lims(1))=lims(1);
% siFamPre(siFamPre>lims(2))=lims(2);
% siFamPost(siFamPost<lims(1))=lims(1);
% siFamPost(siFamPost>lims(2))=lims(2);
% siNovPre(siNovPre<lims(1))=lims(1);
% siNovPre(siNovPre>lims(2))=lims(2);
% siNovPost(siNovPost<lims(1))=lims(1);
% siNovPost(siNovPost>lims(2))=lims(2);

ffPre = scatFullFamPopPre;% ffPre(ffPre<0) = 0;
ofPre = scatOcclFamPopPre;% ofPre(ofPre<0) = 0;
fnPre = scatFullNovPopPre;% fnPre(fnPre<0) = 0;
onPre = scatOcclNovPopPre;% onPre(onPre<0) = 0;
ffPost = scatFullFamPopPost;% ffPost(ffPost<0) = 0;
ofPost = scatOcclFamPopPost;% ofPost(ofPost<0) = 0;
fnPost = scatFullNovPopPost;% fnPost(fnPost<0) = 0;
onPost = scatOcclNovPopPost;% onPost(onPost<0) = 0;

% we put values <-1 at -1 and >1 at 1 after SI calculation. Doesn't matter whether you do it like this or whether you make
% negative values at 0 before calculating selectivity index.
siFamPre = (ffPre-ofPre)./(ffPre+ofPre); siFamPre(isnan(siFamPre))=0; siFamPre(siFamPre<-1)=-1; siFamPre(siFamPre>1)=1;
siFamPost = (ffPost-ofPost)./(ffPost+ofPost); siFamPost(isnan(siFamPost))=0; siFamPost(siFamPost<-1)=-1; siFamPost(siFamPost>1)=1;
siNovPre = (fnPre-onPre)./(fnPre+onPre); siNovPre(isnan(siNovPre))=0; siNovPre(siNovPre<-1)=-1; siNovPre(siNovPre>1)=1;
siNovPost = (fnPost-onPost)./(fnPost+onPost); siNovPost(isnan(siNovPost))=0; siNovPost(siNovPost<-1)=-1; siNovPost(siNovPost>1)=1;

colors = cmapL([0 0 1;0 0 0; 1 0 0], 256);
lims = [-1 1];
siFamPreColors = squeeze(SetLimits(siFamPre, lims, colors));
siFamPostColors = squeeze(SetLimits(siFamPost, lims, colors));
siNovPreColors = squeeze(SetLimits(siNovPre, lims, colors));
siNovPostColors = squeeze(SetLimits(siNovPost, lims, colors));

% some more plotting
alpha = 1;
sz = 10;
cPre = [0.2 0.2 0.2];
cPost = col2;

clear h
figure('Position', [421   301   800   317])
h(1) = subplot(1,2,1);
histogram(siFamPre, -1:0.1:1,'Normalization', 'probability', 'FaceColor', 'k'), hold on
histogram(siFamPost, -1:0.1:1,'Normalization', 'probability', 'FaceColor', 'w', 'EdgeColor', 'k')
xline([mean(siFamPre) mean(siFamPost)]), ylabel('Relative frequency'), xlabel('Selectivity index'),title('Familiar'),
h(2) = subplot(1,2,2);
histogram(siNovPre, -1:0.1:1,'Normalization', 'probability', 'FaceColor', 'k'), hold on
histogram(siNovPost, -1:0.1:1,'Normalization', 'probability', 'FaceColor', 'w', 'EdgeColor', 'k')
xline([mean(siNovPre) mean(siNovPost)]), ylabel('Relative frequency'), xlabel('Selectivity index'),title('Novel'),
linkaxes(h)

if save_fig
    func_save_fig('L23_SIhistsFam')
    func_save_fig('L5_SIhistsFam')
end


% % correlate absolute selectivity to response strength (max of full or occl)
% siFamPre = (ffPre-ofPre)./(ffPre+ofPre); siFamPre(isnan(siFamPre))=0;
siFamPreAbs = abs(siFamPre);
% siFamPost = (ffPost-ofPost)./(ffPost+ofPost); siFamPost(isnan(siFamPost))=0;
siFamPostAbs = abs(siFamPost);
% siNovPre = (fnPre-onPre)./(fnPre+onPre); siNovPre(isnan(siNovPre))=0;
siNovPreAbs = abs(siNovPre);
% siNovPost = (fnPost-onPost)./(fnPost+onPost); siNovPost(isnan(siNovPost))=0;
siNovPostAbs = abs(siNovPost);

mxfPre = max([ffPre; ofPre]);
mxnPre = max([fnPre; onPre]);
mxfPost = max([ffPost; ofPost]);
mxnPost = max([fnPost; onPost]);

figure
subplot(1,2,1)
scatter([1 2],[nanmean(siFamPreAbs) nanmean(siFamPostAbs)], 45, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2],[nanmean(siFamPreAbs) nanmean(siFamPostAbs)], ...
    [calcSem(siFamPreAbs) calcSem(siFamPostAbs)] ...
    ,[calcSem(siFamPreAbs) calcSem(siFamPostAbs)]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 3]), ylabel('Absolute SI fam'), xticks([1 2])
xticklabels({'Pre','Post'}), xtickangle(45), 
subplot(1,2,2)
scatter([1 2],[nanmean(siNovPreAbs) nanmean(siNovPostAbs)], 45, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2],[nanmean(siNovPreAbs) nanmean(siNovPostAbs)], ...
    [calcSem(siNovPreAbs) calcSem(siNovPostAbs)] ...
    ,[calcSem(siNovPreAbs) calcSem(siNovPostAbs)]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 3]), ylabel('Absolute SI nov'), xticks([1 2])
xticklabels({'Pre','Post'}), xtickangle(45), 

if save_fig
    func_save_fig('L23_absSIfamnov')
end

figure('Position', [87         278        1635         673])
subplot(2,5,1)
scatter(siFamPreAbs, mxfPre, 15, 'k', 'filled'), ylim([-1 3.5])
title('Fam pre'),xlabel('SI'), ylabel('Max response'), 
subplot(2,5,2)
scatter(siFamPostAbs, mxfPost, 15, 'k', 'filled'), ylim([-1 3.5])
title('Fam post'),xlabel('SI'), ylabel('Max response'), 
subplot(2,5,6)
scatter(siNovPreAbs, mxnPre, 15, 'k', 'filled'), ylim([-1 3.5])
title('Nov pre'),xlabel('SI'), ylabel('Max response'), 
subplot(2,5,7)
scatter(siNovPostAbs, mxnPost, 15, 'k', 'filled'), ylim([-1 3.5])
title('Nov post'),xlabel('SI'), ylabel('Max response'), 

% divide in two bin, <0.5 si and >0.5 si for statistics
thres = 0.50001;
mxFamPreLow = mxfPre(siFamPreAbs<thres);
mxFamPreHigh = mxfPre(siFamPreAbs>thres);
mxFamPostLow = mxfPost(siFamPostAbs<thres);
mxFamPostHigh = mxfPost(siFamPostAbs>thres);
mxNovPreLow = mxnPre(siNovPreAbs<thres);
mxNovPreHigh = mxnPre(siNovPreAbs>thres);
mxNovPostLow = mxnPost(siNovPostAbs<thres);
mxNovPostHigh = mxnPost(siNovPostAbs>thres);

subplot(2,5,3)
% scatter([1 2 3 4],[nanmean(mxFamPreLow) nanmean(mxFamPreHigh) nanmean(mxFamPostLow) nanmean(mxFamPostHigh)], 45, 'k', 'filled', 'LineWidth', 2), hold on
% scatter([6 7 8 9],[nanmean(mxNovPreLow) nanmean(mxNovPreHigh) nanmean(mxNovPostLow) nanmean(mxNovPostHigh)], 45, col2, 'filled', 'LineWidth', 2)                
% er = errorbar([1 2 3 4],[nanmean(mxFamPreLow) nanmean(mxFamPreHigh) nanmean(mxFamPostLow) nanmean(mxFamPostHigh)], ...
%     [calcSem(mxFamPreLow) calcSem(mxFamPreHigh) calcSem(mxFamPostLow) calcSem(mxFamPostHigh)] ...
%     ,[calcSem(mxFamPreLow) calcSem(mxFamPreHigh) calcSem(mxFamPostLow) calcSem(mxFamPostHigh)]);    
% er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
% er = errorbar([6 7 8 9],[nanmean(mxNovPreLow) nanmean(mxNovPreHigh) nanmean(mxNovPostLow) nanmean(mxNovPostHigh)], ...
%     [calcSem(mxNovPreLow) calcSem(mxNovPreHigh) calcSem(mxNovPostLow) calcSem(mxNovPostHigh)] ...
%     ,[calcSem(mxNovPreLow) calcSem(mxNovPreHigh) calcSem(mxNovPostLow) calcSem(mxNovPostHigh)]);    
% er.Color = col2; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
% xlim([0 10]), ylabel('Response dF/F (%)'), xticks([1 2 3 4 6 7 8 9]), ylim([0 1])
% xticklabels({'PreFamLow', 'PreFamHigh','PostFamLow', 'PostFamHigh', ...
%     'PreNovLow', 'PreNovHigh','PostNovLow', 'PostNovHigh'});
% xtickangle(45), 
% % if save_fig
% %     func_save_fig('L23_SIvsMAXResBinned')
% %     func_save_fig('L5_SIvsMAXResBinned')
% % end
% Grouping for boxchart
boxchart([ones(size(mxFamPreLow)), 2*ones(size(mxFamPreHigh)), 3*ones(size(mxFamPostLow)), 4*ones(size(mxFamPostHigh)), ...
          6*ones(size(mxNovPreLow)), 7*ones(size(mxNovPreHigh)), 8*ones(size(mxNovPostLow)), 9*ones(size(mxNovPostHigh))], ...
          [mxFamPreLow, mxFamPreHigh, mxFamPostLow, mxFamPostHigh, ...
           mxNovPreLow, mxNovPreHigh, mxNovPostLow, mxNovPostHigh], 'MarkerStyle', 'none');
ylabel('Max response'); xlim([0 10]); xticks([1 2 3 4  6 7 8 9]);
xticklabels({'PreFamLow', 'PreFamHigh','PostFamLow', 'PostFamHigh', 'PreNovLow', 'PreNovHigh','PostNovLow', 'PostNovHigh'});
xtickangle(45);
if nfiles == 6
    ylim([-1 2.5]);
elseif nfiles == 5
    ylim([-0.3 1.8]);
end
;

% divide in two bins based on response strenght, then look at SI
thres = 0.5000001; % in response zscore
siFamPreLow = siFamPreAbs(mxfPre<thres);
siFamPreHigh = siFamPreAbs(mxfPre>thres);
siFamPostLow = siFamPostAbs(mxfPost<thres);
siFamPostHigh = siFamPostAbs(mxfPost>thres);
siNovPreLow = siNovPreAbs(mxnPre<thres);
siNovPreHigh = siNovPreAbs(mxnPre>thres);
siNovPostLow = siNovPostAbs(mxnPost<thres);
siNovPostHigh = siNovPostAbs(mxnPost>thres);
% figure('Position', [680   430   334   548])
subplot(2,5,4)
% scatter([1 2 3 4],[nanmean(siFamPreLow) nanmean(siFamPreHigh) nanmean(siFamPostLow) nanmean(siFamPostHigh)], 45, 'k', 'LineWidth', 2), hold on
% scatter([6 7 8 9],[nanmean(siNovPreLow) nanmean(siNovPreHigh) nanmean(siNovPostLow) nanmean(siNovPostHigh)], 45, col2, 'LineWidth', 2)                
% er = errorbar([1 2 3 4],[nanmean(siFamPreLow) nanmean(siFamPreHigh) nanmean(siFamPostLow) nanmean(siFamPostHigh)], ...
%     [calcSem(siFamPreLow) calcSem(siFamPreHigh) calcSem(siFamPostLow) calcSem(siFamPostHigh)] ...
%     ,[calcSem(siFamPreLow) calcSem(siFamPreHigh) calcSem(siFamPostLow) calcSem(siFamPostHigh)]);    
% er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
% er = errorbar([6 7 8 9],[nanmean(siNovPreLow) nanmean(siNovPreHigh) nanmean(siNovPostLow) nanmean(siNovPostHigh)], ...
%     [calcSem(siNovPreLow) calcSem(siNovPreHigh) calcSem(siNovPostLow) calcSem(siNovPostHigh)] ...
%     ,[calcSem(siNovPreLow) calcSem(siNovPreHigh) calcSem(siNovPostLow) calcSem(siNovPostHigh)]);    
% er.Color = col2; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
% xlim([0 10]), ylabel('SI'), xticks([1 2 3 4 6 7 8 9]), ylim([0 1])
% % xticklabels({'PreFamLow', 'PreFamHigh','PostFamLow', 'PostFamHigh', ...
% %     'PreNovLow', 'PreNovHigh','PostNovLow', 'PostNovHigh'});
% xtickangle(45), 
% Grouping for boxchart
subplot(2,5,4);
boxchart([ones(size(siFamPreLow)), 2*ones(size(siFamPreHigh)), 3*ones(size(siFamPostLow)), 4*ones(size(siFamPostHigh)), ...
          6*ones(size(siNovPreLow)), 7*ones(size(siNovPreHigh)), 8*ones(size(siNovPostLow)), 9*ones(size(siNovPostHigh))], ...
          [siFamPreLow, siFamPreHigh, siFamPostLow, siFamPostHigh, ...
           siNovPreLow, siNovPreHigh, siNovPostLow, siNovPostHigh], 'MarkerStyle', 'none');
ylabel('Selectivity'); xlim([0 10]); xticks([1 2 3 4  6 7 8 9]);
xticklabels({'PreFamLow', 'PreFamHigh','PostFamLow', 'PostFamHigh', 'PreNovLow', 'PreNovHigh','PostNovLow', 'PostNovHigh'}); 
xtickangle(45); ylim([0 1]); ;

if save_fig
    func_save_fig('L23_MAXResVsSIplots')
    func_save_fig('L5_MAXResVsSIplots')
end


% note that we load data in here, so if you change parameters in this script
% this will not be taken into account for this particular plot with active
% data
if nfiles == 6
    if regressRun
        load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\postActiveTraceAndScatDataL23regressrun.mat')
    elseif runNan
        load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\postActiveTraceAndScatDataL23nanrun.mat')
    else
        load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\postActiveTraceAndScatDataL23zscore.mat')
    end
elseif nfiles == 5
    % original based on the way we select RFs
%     if regressRun
%         load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\postActiveTraceAndScatDataL5regressrun.mat')
%     elseif runNan
%         load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\postActiveTraceAndScatDataL5nanrun.mat')
%     else
%         load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\postActiveTraceAndScatDataL5zscored.mat')
%     end
% new RF based on post active session RF mapping
    if regressRun
        load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\postActiveTraceAndScatDataL5regressrunv2.mat')
    elseif runNan
        load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\postActiveTraceAndScatDataL5nanrunv2.mat')
    else
        load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\postActiveTraceAndScatDataL5zscorev2.mat')
    end
    
end

% load in active data
if nfiles == 6
    load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\datastructActiveL23.mat')
elseif nfiles == 5
%     load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\datastructActiveL5.mat')
    load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\datastructActiveL5v2.mat')
end

siFamTask = (scatFullPop-scatOcclPop)./(scatFullPop+scatOcclPop); siFamTask(isnan(siFamTask))=0; siFamTask(siFamTask<-1)=-1; siFamTask(siFamTask>1)=1;

siThres = 0.8;
fracFullPre = sum(siFamPre>siThres)/length(siFamPre);
fracOcclPre = sum(siFamPre<-siThres)/length(siFamPre);
fracFullPost = sum(siFamPost>siThres)/length(siFamPost);
fracOcclPost = sum(siFamPost<-siThres)/length(siFamPost);
fracFullTask = sum(siFamTask>siThres)/length(siFamTask);
fracOcclTask = sum(siFamTask<-siThres)/length(siFamTask);

figure('Position', [618   428   841   318])
subplot(1,3,1)
histogram(siFamPre,-1:0.1:1,'Normalization', 'probability'), hold on
histogram(siFamPost,-1:0.1:1,'Normalization', 'probability')
histogram(siFamTask,-1:0.1:1,'Normalization', 'probability')
subplot(1,3,2)
boxchart([ones(size(siFamPre)), ones(size(siFamPost))+1, ones(size(siFamTask))+2], ...
    [siFamPre, siFamPost, siFamTask], 'MarkerStyle','none'), hold on
xlim([0 4]), ylabel('Response'), xticks([1 2 3]); ylim([-1.2 1.2])
xticklabels({'SI pre', 'SI post', 'SI task'}), xtickangle(45), 
subplot(1,3,3)
bar([1 2 3 5 6 7 9 10 11],[fracFullPre fracFullPost fracFullTask fracOcclPre fracOcclPost fracOcclTask ...
    fracFullPre+fracOcclPre fracFullPost+fracOcclPost fracFullTask+fracOcclTask])

n_comparisons = 3; % for bonferroni correction
% full
[p,~] = ranksum(siFamPre, siFamPost); pSIPrePost = p*n_comparisons
[p,~] = ranksum(siFamPre, siFamTask); pSIPreTask = p*n_comparisons
[p,~] = ranksum(siFamPost, siFamTask); pSIPostTask = p*n_comparisons

siFullPreMs = zeros(nfiles,1);
siFullPostMs = zeros(nfiles,1);
siFullTaskMs = zeros(nfiles,1);
siOcclPreMs = zeros(nfiles,1);
siOcclPostMs = zeros(nfiles,1);
siOcclTaskMs = zeros(nfiles,1);
for i = 1:nfiles
   siFullPreMs(i) = sum((datastructPre(i).scatFull-datastructPre(i).scatOccl)./(datastructPre(i).scatFull+datastructPre(i).scatOccl)>siThres)/length(datastructPre(i).scatFull);
   siFullPostMs(i) = sum((datastructPost(i).scatFull-datastructPost(i).scatOccl)./(datastructPost(i).scatFull+datastructPost(i).scatOccl)>siThres)/length(datastructPost(i).scatFull);
   siFullTaskMs(i) = sum((datastructActiveRes(i).scatFull-datastructActiveRes(i).scatOccl)./(datastructActiveRes(i).scatFull+datastructActiveRes(i).scatOccl)>siThres)/length(datastructActiveRes(i).scatFull);
   siOcclPreMs(i) = sum((datastructPre(i).scatFull-datastructPre(i).scatOccl)./(datastructPre(i).scatFull+datastructPre(i).scatOccl)<-siThres)/length(datastructPre(i).scatFull);
   siOcclPostMs(i) = sum((datastructPost(i).scatFull-datastructPost(i).scatOccl)./(datastructPost(i).scatFull+datastructPost(i).scatOccl)<-siThres)/length(datastructPost(i).scatFull);
   siOcclTaskMs(i) = sum((datastructActiveRes(i).scatFull-datastructActiveRes(i).scatOccl)./(datastructActiveRes(i).scatFull+datastructActiveRes(i).scatOccl)<-siThres)/length(datastructActiveRes(i).scatFull);
end

figure
plot([1 2 3], [siFullPreMs'; siFullPostMs'; siFullTaskMs']), hold on
plot([5 6 7], [siOcclPreMs'; siOcclPostMs'; siOcclTaskMs'])
plot([9 10 11], [siFullPreMs'+siOcclPreMs'; siFullPostMs'+siOcclPostMs'; siFullTaskMs'+siOcclTaskMs'])
xlim([0 12])

figure('Position', [356 408 1280 318])
subplot(1,5,1)
bc1 = boxchart([ones(size(scatFullFamPopPre)), ones(size(scatFullFamPopPost))+1, ones(size(scatFullPop))+2], ...
    [scatFullFamPopPre, scatFullFamPopPost, scatFullPop], 'MarkerStyle','none', 'WhiskerLineColor', 'w');
hold on
meanValues1 = [mean(scatFullFamPopPre), mean(scatFullFamPopPost), mean(scatFullPop)];
for i = 1:length(meanValues1)
    plot([i-0.25 i+0.25], [meanValues1(i) meanValues1(i)], 'r-', 'LineWidth',2) % Draws mean line
end
xlim([0 4]), ylabel('Response'), xticks([1 2 3]);
if nfiles == 6, ylim([-0.2 0.7]), elseif nfiles == 5, ylim([-0.1 0.9]), end
xticklabels({'PreFamFull', 'PostFamFull', 'TaskFamFull'}), xtickangle(45)
subplot(1,5,2)
bc2 = boxchart([ones(size(scatOcclFamPopPre)), ones(size(scatOcclFamPopPost))+1, ones(size(scatOcclPop))+2], ...
    [scatOcclFamPopPre, scatOcclFamPopPost, scatOcclPop], 'MarkerStyle','none', 'WhiskerLineColor', 'w');
hold on
meanValues2 = [mean(scatOcclFamPopPre), mean(scatOcclFamPopPost), mean(scatOcclPop)];
for i = 1:length(meanValues2)
    plot([i-0.25 i+0.25], [meanValues2(i) meanValues2(i)], 'r-', 'LineWidth',2) % Draws mean line
end
xlim([0 4]), ylabel('Response'), xticks([1 2 3]);
if nfiles == 6, ylim([-0.1 0.4]), elseif nfiles == 5, ylim([-0.1 0.9]), end
xticklabels({'PreFamOccl', 'PostFamOccl', 'TaskFamOccl'}), xtickangle(45)

subplot(1,5,3)
scatter([1 2 3 5 6 7],[median(scatFullFamPopPre) median(scatFullFamPopPost) median(scatFullPop) median(scatOcclFamPopPre) median(scatOcclFamPopPost) median(scatOcclPop)], 50, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2 3 5 6 7],[median(scatFullFamPopPre) median(scatFullFamPopPost) median(scatFullPop) median(scatOcclFamPopPre) median(scatOcclFamPopPost) median(scatOcclPop)], ...
    [calcSem(scatFullFamPopPre) calcSem(scatFullFamPopPost) calcSem(scatFullPop) calcSem(scatOcclFamPopPre) calcSem(scatOcclFamPopPost) calcSem(scatOcclPop)] ...
    ,[calcSem(scatFullFamPopPre) calcSem(scatFullFamPopPost) calcSem(scatFullPop) calcSem(scatOcclFamPopPre) calcSem(scatOcclFamPopPost) calcSem(scatOcclPop)]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 8]), ylabel('Z-score df/f'), xticks([1 2 3 5 6 7])
xticklabels({'Naive', 'Expert', 'Task', 'Naive', 'Expert', 'Task'}), xtickangle(45), 
subplot(1,5,4)
scatter([1 2 3 5 6 7],[mean(scatFullFamPopPre) mean(scatFullFamPopPost) mean(scatFullPop) mean(scatOcclFamPopPre) mean(scatOcclFamPopPost) mean(scatOcclPop)], 50, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2 3 5 6 7],[mean(scatFullFamPopPre) mean(scatFullFamPopPost) mean(scatFullPop) mean(scatOcclFamPopPre) mean(scatOcclFamPopPost) mean(scatOcclPop)], ...
    [calcSem(scatFullFamPopPre) calcSem(scatFullFamPopPost) calcSem(scatFullPop) calcSem(scatOcclFamPopPre) calcSem(scatOcclFamPopPost) calcSem(scatOcclPop)] ...
    ,[calcSem(scatFullFamPopPre) calcSem(scatFullFamPopPost) calcSem(scatFullPop) calcSem(scatOcclFamPopPre) calcSem(scatOcclFamPopPost) calcSem(scatOcclPop)]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 8]), ylabel('Z-score df/f'), xticks([1 2 3 5 6 7])
xticklabels({'Naive', 'Expert', 'Task', 'Naive', 'Expert', 'Task'}), xtickangle(45), 

if save_fig
    func_save_fig('L23_quantification')
    func_save_fig('L5_quantification')
    func_save_fig('L5_boxplots')
end

sz = 70;
figure('Position', [618   376   355   370])
scatter([1 2 3 5 6 7],[nanmean(scatFullFamPopPre) nanmean(scatFullFamPopPost) nanmean(scatFullPop) nanmean(scatOcclFamPopPre) nanmean(scatOcclFamPopPost) nanmean(scatOcclPop)], sz, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2 3 5 6 7],[nanmean(scatFullFamPopPre) nanmean(scatFullFamPopPost) nanmean(scatFullPop) nanmean(scatOcclFamPopPre) nanmean(scatOcclFamPopPost) nanmean(scatOcclPop)], ...
    [calcSem(scatFullFamPopPre) calcSem(scatFullFamPopPost) calcSem(scatFullPop) calcSem(scatOcclFamPopPre) calcSem(scatOcclFamPopPost) calcSem(scatOcclPop)] ...
    ,[calcSem(scatFullFamPopPre) calcSem(scatFullFamPopPost) calcSem(scatFullPop) calcSem(scatOcclFamPopPre) calcSem(scatOcclFamPopPost) calcSem(scatOcclPop)]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 8]), if nfiles == 6, ylim([0 0.4]), else, ylim([0 0.6]), end,  ylabel('Z-score df/f'), xticks([1 2 3 5 6 7])
xticklabels({'Naive', 'Expert', 'Task', 'Naive', 'Expert', 'Task'}), xtickangle(45), 

if save_fig
    func_save_fig('L23_quantification_errorbar')
    func_save_fig('L5_quantification_errorbar')
    func_save_fig('L5_boxplots')
end

sz = 70;
figure('Position', [87         278        1635         673])
subplot(2,5,1);
scatter([1 2 4 5],[nanmean(scatFullNovPopPre) nanmean(scatFullNovPopPost) nanmean(scatOcclNovPopPre) nanmean(scatOcclNovPopPost) ], sz, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2 4 5],[nanmean(scatFullNovPopPre) nanmean(scatFullNovPopPost) nanmean(scatOcclNovPopPre) nanmean(scatOcclNovPopPost)], ...
    [calcSem(scatFullNovPopPre) calcSem(scatFullNovPopPost) calcSem(scatOcclNovPopPre) calcSem(scatOcclNovPopPost)] ...
    ,[calcSem(scatFullNovPopPre) calcSem(scatFullNovPopPost) calcSem(scatOcclNovPopPre) calcSem(scatOcclNovPopPost)]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 6]), if nfiles == 6, ylim([0 0.5]), else, ylim([0 1]), end,  ylabel('Z-score df/f'), xticks([1 2 4 5])
xticklabels({'Naive', 'Expert', 'Naive', 'Expert'}), xtickangle(45), 

if save_fig
    func_save_fig('L23_quantification_novel_errorbar')
    func_save_fig('L5_quantification_novel_errorbar')
    func_save_fig('L5_boxplots')
end


mouseIDPre = [];
mouseIDPost = [];
mouseIDTask = [];
for i = 1:nfiles
    % prepare some data for linear mixed model effect
    mouseIDPre = [mouseIDPre zeros(1,length(datastructPre(i).scatFull))+i];
    mouseIDPost = [mouseIDPost zeros(1,length(datastructPost(i).scatFull))+i];
    mouseIDTask = [mouseIDTask zeros(1,length(datastructActiveRes(i).scatFull))+i];
end

% full fam LMEM
data = cat(2, scatFullFamPopPre,scatFullFamPopPost, scatFullPop)';
mouseID = categorical(cat(2, mouseIDPre,mouseIDPost,mouseIDTask))';
condition = categorical(cat(1, ones(length(mouseIDPre),1),ones(length(mouseIDPost),1)+1,ones(length(mouseIDTask),1)+2));
clear statTbl, statTbl = table(data, mouseID, condition);
lmeFullFam = fitlme(statTbl, 'data ~ condition + (1|mouseID)', 'CheckHessian', 1,'FitMethod', 'REML', 'StartMethod', 'random');
statsFullFam = anova(lmeFullFam,'DFMethod','Satterthwaite');
statTblFull = makeStatTbl(lmeFullFam);

% occl fam LMEM
data = cat(2, scatOcclFamPopPre,scatOcclFamPopPost, scatOcclPop)';
mouseID = categorical(cat(2, mouseIDPre,mouseIDPost,mouseIDTask))';
condition = categorical(cat(1, ones(length(mouseIDPre),1),ones(length(mouseIDPost),1)+1,ones(length(mouseIDTask),1)+2));
clear statTbl, statTbl = table(data, mouseID, condition);
lmeOcclFam = fitlme(statTbl, 'data ~ condition + (1|mouseID)', 'CheckHessian', 1,'FitMethod', 'REML', 'StartMethod', 'random');
statsOcclFam = anova(lmeOcclFam,'DFMethod','Satterthwaite');
statTblOccl = makeStatTbl(lmeOcclFam);

% full nov LMEM
data = cat(2, scatFullNovPopPre,scatFullNovPopPost)';
mouseID = categorical(cat(2, mouseIDPre,mouseIDPost))';
condition = categorical(cat(1, ones(length(mouseIDPre),1),ones(length(mouseIDPost),1)+1));
clear statTbl, statTbl = table(data, mouseID, condition);
lmeFullNov = fitlme(statTbl, 'data ~ condition + (1|mouseID)', 'CheckHessian', 1,'FitMethod', 'REML', 'StartMethod', 'random');
statsFullNov = anova(lmeFullNov,'DFMethod','Satterthwaite');

% occl nov LMEM
data = cat(2, scatOcclNovPopPre,scatOcclNovPopPost)';
mouseID = categorical(cat(2, mouseIDPre,mouseIDPost))';
condition = categorical(cat(1, ones(length(mouseIDPre),1),ones(length(mouseIDPost),1)+1));
clear statTbl, statTbl = table(data, mouseID, condition);
lmeOcclNov = fitlme(statTbl, 'data ~ condition + (1|mouseID)', 'CheckHessian', 1,'FitMethod', 'REML', 'StartMethod', 'random');
statsOcclNov = anova(lmeOcclNov,'DFMethod','Satterthwaite');


%%%%% only for comparison pre vs post with more RF distance
% full fam LMEM
data = cat(2, scatFullFamPopPre,scatFullFamPopPost)';
mouseID = categorical(cat(2, mouseIDPre,mouseIDPost))';
condition = categorical(cat(1, ones(length(mouseIDPre),1),ones(length(mouseIDPost),1)+1));
clear statTbl, statTbl = table(data, mouseID, condition);
lmeFullFam = fitlme(statTbl, 'data ~ condition + (1|mouseID)', 'CheckHessian', 1,'FitMethod', 'REML', 'StartMethod', 'random');
statsFullFam = anova(lmeFullFam,'DFMethod','Satterthwaite');

% occl fam LMEM
data = cat(2, scatOcclFamPopPre,scatOcclFamPopPost)';
mouseID = categorical(cat(2, mouseIDPre,mouseIDPost))';
condition = categorical(cat(1, ones(length(mouseIDPre),1),ones(length(mouseIDPost),1)+1));
clear statTbl, statTbl = table(data, mouseID, condition);
lmeOcclFam = fitlme(statTbl, 'data ~ condition + (1|mouseID)', 'CheckHessian', 1,'FitMethod', 'REML', 'StartMethod', 'random');
statsOcclFam = anova(lmeOcclFam,'DFMethod','Satterthwaite');


%% selectivity vs response strength

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
maskPre = scatFullNovPopPre > 0 & any(~isnan(siNovPre), 1);
% maskPre = any(~isnan(siNovPre), 1);
xPre = siNovPreMn(maskPre);
yPre = scatFullNovPopPre(maskPre);

scatter(xPre, yPre, 'filled', 'k'); refline;
xlabel('Naive Selectivity Index (familiar to novel)');
ylabel('Response Strength for Novel Naive');
title('Selectivity vs. Response Strength (>0.5 Responders)');
% ylim([-0.5 4])

% Compute correlation
[r, p] = corr(xPre', yPre', 'type', 'Pearson');

% Add text to plot
text(min(xPre) + 0.05 * range(xPre), max(yPre) - 0.05 * range(yPre), ...
    sprintf('R = %.2f, p = %.3g', r, p), ...
    'FontSize', 12, 'FontWeight', 'bold');

if save_fig
    func_save_fig('L23_scatter_novelSelectivity_vs_novelResStrength')
    func_save_fig('L5_scatter_novelSelectivity_vs_novelResStrength')
end


% analysis for post
%%% selectivity vs response strength

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
maskPost = scatFullNovPopPost > 0 & any(~isnan(siNovPost), 1);
% maskPost = any(~isnan(siNovPost), 1);
xPost = siNovPostMn(maskPost);
yPost = scatFullNovPopPost(maskPost);

scatter(xPost, yPost, 'filled', 'k'); refline;
xlabel('Expert Selectivity Index (familiar to novel)');
ylabel('Response Strength for Novel Expert');
title('Selectivity vs. Response Strength (>0.5 Responders)');
% ylim([-0.5 4])

% Compute correlation
[r, p] = corr(xPost', yPost', 'type', 'Pearson');

% Add text to plot
text(min(xPost) + 0.05 * range(xPost), max(yPost) - 0.05 * range(yPost), ...
    sprintf('R = %.2f, p = %.3g', r, p), ...
    'FontSize', 12, 'FontWeight', 'bold');

if save_fig
    func_save_fig('L23_scatter_novelSelectivity_vs_novelResStrength')
    func_save_fig('L5_scatter_novelSelectivity_vs_novelResStrength')
end

% avg selectivity for high responders (>0.5)
ixPre = yPre>0.5;
siNovPreMnIx = xPre(ixPre);
ixPost = yPost>0.5;
siNovPostMnIx = xPost(ixPost);

figure('Position', [293   502   320   378])

scatter([1 2],[nanmean(siNovPreMnIx) nanmean(siNovPostMnIx)], 80, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2],[nanmean(siNovPreMnIx) nanmean(siNovPostMnIx)], ...
    [calcSem(siNovPreMnIx) calcSem(siNovPostMnIx)] ...
    ,[calcSem(siNovPreMnIx) calcSem(siNovPostMnIx)]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 3]), ylabel('Expert selectivity for novel stimuli'), xticks([1 2])
xticklabels({'Naive','Expert'}), xtickangle(45), 
ylim([0 0.5])


if save_fig
    func_save_fig('L23_errorbar_selectivityToNovel_preVsPost')
    func_save_fig('L5_errorbar_selectivityToNovel')
end

% stats
mouseIDPre = [];
mouseIDPost = [];
for i = 1:nfiles
    % prepare some data for linear mixed model effect
    mouseIDPre = [mouseIDPre zeros(1,length(datastructPre(i).scatFull))+i];
    mouseIDPost = [mouseIDPost zeros(1,length(datastructPost(i).scatFull))+i];
end

mouseIDPre = mouseIDPre(maskPre);
mouseIDPre = mouseIDPre(ixPre);
mouseIDPost = mouseIDPost(maskPost);
mouseIDPost = mouseIDPost(ixPost);

warning on

% full fam LMEM
data = cat(2, siNovPreMnIx,siNovPostMnIx)';
mouseID = categorical(cat(2, mouseIDPre,mouseIDPost))';
condition = categorical(cat(1, ones(length(mouseIDPre),1),ones(length(mouseIDPost),1)+1));
clear statTbl, statTbl = table(data, mouseID, condition);
lmeSINovPreVsPost = fitlme(statTbl, 'data ~ condition + (1|mouseID)', 'CheckHessian', 1,'FitMethod', 'REML', 'StartMethod', 'random');
statsSINovPreVsPost = anova(lmeSINovPreVsPost,'DFMethod','Satterthwaite');

%% interaction effect between cell type and condition (naive vs expert)

% load L2/3 data
load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\scatOcclFamPrePostL23.mat')
occlPreL23 = scatOcclFamPopPre;
occlPostL23 = scatOcclFamPopPost;
mouseIDPreL23 = mouseIDPre;
mouseIDPostL23 = mouseIDPost;

% load L5 data
load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\scatOcclFamPrePostL5.mat')
occlPreL5 = scatOcclFamPopPre;
occlPostL5 = scatOcclFamPopPost;
mouseIDPreL5 = mouseIDPre;
mouseIDPostL5 = mouseIDPost;

warning on

% Combine Group 1 and Group 2 data
dataGroup1 = cat(2, occlPreL23, occlPostL23)';
dataGroup2 = cat(2, occlPreL5, occlPostL5)';
data = cat(1, dataGroup1, dataGroup2);

mouseIDGroup1 = categorical(cat(2, mouseIDPreL23, mouseIDPostL23))';
mouseIDGroup2 = categorical(cat(2, mouseIDPreL5, mouseIDPostL5))';
mouseID = cat(1, mouseIDGroup1, mouseIDGroup2);

conditionGroup1 = categorical(cat(1, ones(length(occlPreL23),1), ones(length(occlPostL23),1) + 1));
conditionGroup2 = categorical(cat(1, ones(length(occlPreL5),1), ones(length(occlPostL5),1) + 1));
condition = cat(1, conditionGroup1, conditionGroup2);

% Create a group variable
group = cat(1, repmat(categorical({'Group1'}), length(dataGroup1), 1), repmat(categorical({'Group2'}), length(dataGroup2), 1));

% Create the table
statTbl = table(data, mouseID, condition, group);

% Fit the LME model including group, condition, and their interaction
lme = fitlme(statTbl, 'data ~ condition*group + (1|mouseID)', 'CheckHessian', 1, 'FitMethod', 'REML', 'StartMethod', 'random');

% Get ANOVA table with Satterthwaite's method for degrees of freedom
stats = anova(lme, 'DFMethod', 'Satterthwaite');

% Display the results
disp(stats);

%% LATEST ADAPTATION ANALYSIS FOR IN PAPER
% famIdx = [1 2 4 5];
% % famIdx = [1 2 3 4 5 6];
% novIdx = [3 6];

% plot over trials
warning on 

if nfiles == 5
    if regressRun
        load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\traces_regressrun_ActiveL5.mat')
    else
        load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\traces_noregressrun_ActiveL5.mat')
    end
end

if nfiles == 6
    vecAxTaskSp = vecAxTask<0;
    vecAxTaskSt = vecAxTask>0.2 & vecAxTask<1;
end

% adaptation plots
cfg = struct('nfiles', nfiles, ...
             'datastructPre', datastructPre, ...
             'datastructPost', datastructPost, ...
             'datastructTask', datastructTask, ...
             'vecAxSt', vecAxSt, 'vecAxSp', vecAxSp, ...
             'vecAxTaskSt', vecAxTaskSt, 'vecAxTaskSp', vecAxTaskSp);

cfg.respThreshold = 0;
cfg.smoothed = true;
cfg.smoothVal = 0.5;

[stats, figH] = adaptation_plots(cfg);

if save_fig
    func_save_fig('L23_og_resOverTrials')
    func_save_fig('L23_og_resOverTrials_quantonly')
    func_save_fig('L5_og_resOverTrials')
    func_save_fig('L5_og_resOverTrials_quantonly')
end





%% SOME OLDER STUFF
famIdx = [1 2 4 5];
% famIdx = [1 2 3 4 5 6];
novIdx = [3 6];

% plot over trials
warning on 

if nfiles == 5
    if regressRun
        load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\traces_regressrun_ActiveL5.mat')
    else
        load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\traces_noregressrun_ActiveL5.mat')
    end
end

if nfiles == 6
    vecAxTaskSp = vecAxTask<0;
    vecAxTaskSt = vecAxTask>0.2 & vecAxTask<1;
end

imgFullResPre = [];
imgOcclResPre = [];
imgFullResPost = [];
imgOcclResPost = [];
imgFullResTask = [];
imgOcclResTask = [];

for i = 1:nfiles
    imgFullResPre = cat(4, imgFullResPre, datastructPre(i).imgFullRes);
    imgOcclResPre = cat(4, imgOcclResPre, datastructPre(i).imgOcclRes);
    imgFullResPost = cat(4, imgFullResPost, datastructPost(i).imgFullRes);
    imgOcclResPost = cat(4, imgOcclResPost, datastructPost(i).imgOcclRes);
    imgFullResTask = cat(4, imgFullResTask, datastructTask(i).imgFullRes);
    imgOcclResTask = cat(4, imgOcclResTask, datastructTask(i).imgOcclRes);
end

clear imgFullResPreMnFamMn imgOcclResPreMnFamMn imgFullResPostMnFamMn... 
imgFullResPreMnNovMn imgOcclResPreMnNovMn imgFullResPostMnNovMn...
imgFullResTaskMnFamMn imgOcclResTaskMnFamMn imgOcclResPostMnFamMn imgOcclResPostMnNovMn

fh = 1:5;
sh = 16:20;

respThreshold = -1000;
smoothed = true; % do you want to smooth the traces
smoothVal = 1; % by how much?
sub = 1:20;

imgFullResPreMn = squeeze(mean(imgFullResPre(vecAxSt,:,:,:)))-squeeze(mean(imgFullResPre(vecAxSp,:,:,:)));
imgFullResPreMnFam = squeeze(mean(imgFullResPreMn(famIdx,:,:)));
if smoothed, imgFullResPreMnFam = smoothG(imgFullResPreMnFam,smoothVal); end
imgFullResPreMnFam = imgFullResPreMnFam(:,mean(imgFullResPreMnFam(sub,:))>respThreshold);
mouseIdxFullPreFam = mouseIDPre(mean(imgFullResPreMnFam(sub,:))>respThreshold);
imgFullResPreMnFamMn(1,:) = mean(imgFullResPreMnFam(fh,:));
imgFullResPreMnFamMn(2,:) = mean(imgFullResPreMnFam(sh,:));
% imgFullResPreMnFamMn = imgFullResPreMnFamMn./imgFullResPreMnFamMn(1,:);
imgFullResPreMnNov = squeeze(mean(imgFullResPreMn(novIdx,:,:)));
if smoothed, imgFullResPreMnNov = smoothG(imgFullResPreMnNov,smoothVal);
imgFullResPreMnNov = imgFullResPreMnNov(:,mean(imgFullResPreMnNov(sub,:))>respThreshold); end
mouseIdxFullPreNov = mouseIDPre(mean(imgFullResPreMnNov(sub,:))>respThreshold);
imgFullResPreMnNovMn(1,:) = mean(imgFullResPreMnNov(fh,:));
imgFullResPreMnNovMn(2,:) = mean(imgFullResPreMnNov(sh,:));

% --- Occluded Pre ---
imgOcclResPreMn = squeeze(mean(imgOcclResPre(vecAxSt,:,:,:))) - squeeze(mean(imgOcclResPre(vecAxSp,:,:,:)));

imgOcclResPreMnFam = squeeze(mean(imgOcclResPreMn(famIdx,:,:)));
if smoothed, imgOcclResPreMnFam = smoothG(imgOcclResPreMnFam,smoothVal); end
imgOcclResPreMnFam = imgOcclResPreMnFam(:, mean(imgOcclResPreMnFam(sub,:)) > respThreshold);
mouseIdxOcclPreFam = mouseIDPre(mean(imgOcclResPreMnFam(sub,:)) > respThreshold);
imgOcclResPreMnFamMn(1,:) = mean(imgOcclResPreMnFam(fh,:));
imgOcclResPreMnFamMn(2,:) = mean(imgOcclResPreMnFam(sh,:));

imgOcclResPreMnNov = squeeze(mean(imgOcclResPreMn(novIdx,:,:)));
if smoothed, imgOcclResPreMnNov = smoothG(imgOcclResPreMnNov,smoothVal); end
imgOcclResPreMnNov = imgOcclResPreMnNov(:, mean(imgOcclResPreMnNov(sub,:)) > respThreshold);
mouseIdxOcclPreNov = mouseIDPre(mean(imgOcclResPreMnNov(sub,:)) > respThreshold);
imgOcclResPreMnNovMn(1,:) = mean(imgOcclResPreMnNov(fh,:));
imgOcclResPreMnNovMn(2,:) = mean(imgOcclResPreMnNov(sh,:));

% --- Full Post ---
imgFullResPostMn = squeeze(mean(imgFullResPost(vecAxSt,:,:,:))) - squeeze(mean(imgFullResPost(vecAxSp,:,:,:)));

imgFullResPostMnFam = squeeze(mean(imgFullResPostMn(famIdx,:,:)));
if smoothed, imgFullResPostMnFam = smoothG(imgFullResPostMnFam,smoothVal); end
imgFullResPostMnFam = imgFullResPostMnFam(:, mean(imgFullResPostMnFam(sub,:)) > respThreshold);
mouseIdxFullPostFam = mouseIDPost(mean(imgFullResPostMnFam(sub,:)) > respThreshold);
imgFullResPostMnFamMn(1,:) = mean(imgFullResPostMnFam(fh,:));
imgFullResPostMnFamMn(2,:) = mean(imgFullResPostMnFam(sh,:));

imgFullResPostMnNov = squeeze(mean(imgFullResPostMn(novIdx,:,:)));
if smoothed, imgFullResPostMnNov = smoothG(imgFullResPostMnNov,smoothVal); end
imgFullResPostMnNov = imgFullResPostMnNov(:, mean(imgFullResPostMnNov(sub,:)) > respThreshold);
mouseIdxFullPostNov = mouseIDPost(mean(imgFullResPostMnNov(sub,:)) > respThreshold);
imgFullResPostMnNovMn(1,:) = mean(imgFullResPostMnNov(fh,:));
imgFullResPostMnNovMn(2,:) = mean(imgFullResPostMnNov(sh,:));

% --- Occluded Post ---
imgOcclResPostMn = squeeze(mean(imgOcclResPost(vecAxSt,:,:,:))) - squeeze(mean(imgOcclResPost(vecAxSp,:,:,:)));

imgOcclResPostMnFam = squeeze(mean(imgOcclResPostMn(famIdx,:,:)));
if smoothed, imgOcclResPostMnFam = smoothG(imgOcclResPostMnFam,smoothVal); end
imgOcclResPostMnFam = imgOcclResPostMnFam(:, mean(imgOcclResPostMnFam(sub,:)) > respThreshold);
mouseIdxOcclPostFam = mouseIDPost(mean(imgOcclResPostMnFam(sub,:)) > respThreshold);
imgOcclResPostMnFamMn(1,:) = mean(imgOcclResPostMnFam(fh,:));
imgOcclResPostMnFamMn(2,:) = mean(imgOcclResPostMnFam(sh,:));

imgOcclResPostMnNov = squeeze(mean(imgOcclResPostMn(novIdx,:,:)));
if smoothed, imgOcclResPostMnNov = smoothG(imgOcclResPostMnNov,smoothVal); end
imgOcclResPostMnNov = imgOcclResPostMnNov(:, mean(imgOcclResPostMnNov(sub,:)) > respThreshold);
mouseIdxOcclPostNov = mouseIDPost(mean(imgOcclResPostMnNov(sub,:)) > respThreshold);
imgOcclResPostMnNovMn(1,:) = mean(imgOcclResPostMnNov(fh,:));
imgOcclResPostMnNovMn(2,:) = mean(imgOcclResPostMnNov(sh,:));

% --- Full Task (Fam only) ---
imgFullResTaskMn = squeeze(mean(imgFullResTask(vecAxTaskSt,:,:,:))) - squeeze(mean(imgFullResTask(vecAxTaskSp,:,:,:)));
imgFullResTaskMnFam = squeeze(mean(imgFullResTaskMn, 1));
if smoothed, imgFullResTaskMnFam = smoothG(imgFullResTaskMnFam,smoothVal); end
imgFullResTaskMnFam = imgFullResTaskMnFam(:, mean(imgFullResTaskMnFam(sub,:)) > respThreshold);
mouseIdxFullTaskFam = mouseIDTask(mean(imgFullResTaskMnFam(sub,:)) > respThreshold);
imgFullResTaskMnFamMn(1,:) = mean(imgFullResTaskMnFam(fh,:));
imgFullResTaskMnFamMn(2,:) = mean(imgFullResTaskMnFam(sh,:));

% --- Occluded Task (Fam only) ---
imgOcclResTaskMn = squeeze(mean(imgOcclResTask(vecAxTaskSt,:,:,:))) - squeeze(mean(imgOcclResTask(vecAxTaskSp,:,:,:)));
imgOcclResTaskMnFam = squeeze(mean(imgOcclResTaskMn, 1));
if smoothed, imgOcclResTaskMnFam = smoothG(imgOcclResTaskMnFam,smoothVal); end
imgOcclResTaskMnFam = imgOcclResTaskMnFam(:, mean(imgOcclResTaskMnFam(sub,:)) > respThreshold);
mouseIdxOcclTaskFam = mouseIDTask(mean(imgOcclResTaskMnFam(sub,:)) > respThreshold);
imgOcclResTaskMnFamMn(1,:) = mean(imgOcclResTaskMnFam(fh,:));
imgOcclResTaskMnFamMn(2,:) = mean(imgOcclResTaskMnFam(sh,:));

clear s t

figure('Position', [293         309        1164         571])
s(1) = subplot(2,4,1);
shadedErrorBar(1:20,mean(imgFullResPreMnFam,2)...
    ,std(imgFullResPreMnFam,0,2)/sqrt(size(imgFullResPreMnFam,2)), 'lineProps', 'b'); hold on
shadedErrorBar(1:20,mean(imgFullResPreMnNov,2)...
    ,std(imgFullResPreMnNov,0,2)/sqrt(size(imgFullResPreMnNov,2)), 'lineProps', 'r'); 
ylabel('Response'), xlabel('Trials'), title('NO Pre'), 
s(2) = subplot(2,4,2);
shadedErrorBar(1:20,mean(imgFullResPostMnFam,2)...
    ,std(imgFullResPostMnFam,0,2)/sqrt(size(imgFullResPostMnFam,2)), 'lineProps', 'b'); hold on
shadedErrorBar(1:20,mean(imgFullResPostMnNov,2)...
    ,std(imgFullResPostMnNov,0,2)/sqrt(size(imgFullResPostMnNov,2)), 'lineProps', 'r'); 
ylabel('Response'), xlabel('Trials'), title('NO Post'), 
s(3) = subplot(2,4,3);
shadedErrorBar(1:20, mean(imgFullResTaskMnFam,2), ...
    std(imgFullResTaskMnFam,0,2)/sqrt(size(imgFullResTaskMnFam,2)), 'lineProps', 'b'); 
ylabel('Response'), xlabel('Trials'), title('NO Task'), 


s(4) = subplot(2,4,5);
shadedErrorBar(1:20,mean(imgOcclResPreMnFam,2)...
    ,std(imgOcclResPreMnFam,0,2)/sqrt(size(imgOcclResPreMnFam,2)), 'lineProps', 'b'); hold on
shadedErrorBar(1:20,mean(imgOcclResPreMnNov,2)...
    ,std(imgOcclResPreMnNov,0,2)/sqrt(size(imgOcclResPreMnNov,2)), 'lineProps', 'r'); 
ylabel('Response'), xlabel('Trials'), title('O Pre'), 
s(5) = subplot(2,4,6);
shadedErrorBar(1:20,mean(imgOcclResPostMnFam,2)...
    ,std(imgOcclResPostMnFam,0,2)/sqrt(size(imgOcclResPostMnFam,2)), 'lineProps', 'b'); hold on
shadedErrorBar(1:20,mean(imgOcclResPostMnNov,2)...
    ,std(imgOcclResPostMnNov,0,2)/sqrt(size(imgOcclResPostMnNov,2)), 'lineProps', 'r'); 
ylabel('Response'), xlabel('Trials'), title('O Post'), 
s(6) = subplot(2,4,7);
shadedErrorBar(1:20, mean(imgOcclResTaskMnFam,2), ...
    std(imgOcclResTaskMnFam,0,2)/sqrt(size(imgOcclResTaskMnFam,2)), 'lineProps', 'b'); 
ylabel('Response'), xlabel('Trials'), title('O Task'), 


t(1) = subplot(2,4,4);
scatter([1 2],[nanmean(imgFullResPreMnFamMn(1,:)) nanmean(imgFullResPreMnFamMn(2,:))], 45, 'b', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2],[nanmean(imgFullResPreMnFamMn(1,:)) nanmean(imgFullResPreMnFamMn(2,:))], ...
    [calcSem(imgFullResPreMnFamMn(1,:)) calcSem(imgFullResPreMnFamMn(2,:))] ...
    ,[calcSem(imgFullResPreMnFamMn(1,:)) calcSem(imgFullResPreMnFamMn(2,:))]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
plot([1 2],[nanmean(imgFullResPreMnFamMn(1,:)) nanmean(imgFullResPreMnFamMn(2,:))],'b')
scatter([4 5],[nanmean(imgFullResPreMnNovMn(1,:)) nanmean(imgFullResPreMnNovMn(2,:))], 45, 'r', 'filled', 'LineWidth', 2), hold on
er = errorbar([4 5],[nanmean(imgFullResPreMnNovMn(1,:)) nanmean(imgFullResPreMnNovMn(2,:))], ...
    [calcSem(imgFullResPreMnNovMn(1,:)) calcSem(imgFullResPreMnNovMn(2,:))] ...
    ,[calcSem(imgFullResPreMnNovMn(1,:)) calcSem(imgFullResPreMnNovMn(2,:))]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;xlim([0 12])
plot([4 5],[nanmean(imgFullResPreMnNovMn(1,:)) nanmean(imgFullResPreMnNovMn(2,:))],'r')
scatter([7 8],[nanmean(imgFullResPostMnFamMn(1,:)) nanmean(imgFullResPostMnFamMn(2,:))], 45, 'b', 'filled', 'LineWidth', 2), hold on
er = errorbar([7 8],[nanmean(imgFullResPostMnFamMn(1,:)) nanmean(imgFullResPostMnFamMn(2,:))], ...
    [calcSem(imgFullResPostMnFamMn(1,:)) calcSem(imgFullResPostMnFamMn(2,:))] ...
    ,[calcSem(imgFullResPostMnFamMn(1,:)) calcSem(imgFullResPostMnFamMn(2,:))]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
plot([7 8],[nanmean(imgFullResPostMnFamMn(1,:)) nanmean(imgFullResPostMnFamMn(2,:))],'b')
scatter([10 11],[nanmean(imgFullResPostMnNovMn(1,:)) nanmean(imgFullResPostMnNovMn(2,:))], 45, 'r', 'filled', 'LineWidth', 2), hold on
er = errorbar([10 11],[nanmean(imgFullResPostMnNovMn(1,:)) nanmean(imgFullResPostMnNovMn(2,:))], ...
    [calcSem(imgFullResPostMnNovMn(1,:)) calcSem(imgFullResPostMnNovMn(2,:))] ...
    ,[calcSem(imgFullResPostMnNovMn(1,:)) calcSem(imgFullResPostMnNovMn(2,:))]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
plot([10 11],[nanmean(imgFullResPostMnNovMn(1,:)) nanmean(imgFullResPostMnNovMn(2,:))],'r')
scatter([13 14],[nanmean(imgFullResTaskMnFamMn(1,:)) nanmean(imgFullResTaskMnFamMn(2,:))], 45, 'b', 'filled', 'LineWidth', 2), hold on
er = errorbar([13 14],[nanmean(imgFullResTaskMnFamMn(1,:)) nanmean(imgFullResTaskMnFamMn(2,:))], ...
    [calcSem(imgFullResTaskMnFamMn(1,:)) calcSem(imgFullResTaskMnFamMn(2,:))] ...
    ,[calcSem(imgFullResTaskMnFamMn(1,:)) calcSem(imgFullResTaskMnFamMn(2,:))]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;xlim([0 15])
plot([13 14],[nanmean(imgFullResTaskMnFamMn(1,:)) nanmean(imgFullResTaskMnFamMn(2,:))],'b')
ylabel('Response'), 

t(2) = subplot(2,4,8);
scatter([1 2],[nanmean(imgOcclResPreMnFamMn(1,:)) nanmean(imgOcclResPreMnFamMn(2,:))], 45, 'b', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2],[nanmean(imgOcclResPreMnFamMn(1,:)) nanmean(imgOcclResPreMnFamMn(2,:))], ...
    [calcSem(imgOcclResPreMnFamMn(1,:)) calcSem(imgOcclResPreMnFamMn(2,:))] ...
    ,[calcSem(imgOcclResPreMnFamMn(1,:)) calcSem(imgOcclResPreMnFamMn(2,:))]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
plot([1 2],[nanmean(imgOcclResPreMnFamMn(1,:)) nanmean(imgOcclResPreMnFamMn(2,:))],'b')
scatter([4 5],[nanmean(imgOcclResPreMnNovMn(1,:)) nanmean(imgOcclResPreMnNovMn(2,:))], 45, 'r', 'filled', 'LineWidth', 2), hold on
er = errorbar([4 5],[nanmean(imgOcclResPreMnNovMn(1,:)) nanmean(imgOcclResPreMnNovMn(2,:))], ...
    [calcSem(imgOcclResPreMnNovMn(1,:)) calcSem(imgOcclResPreMnNovMn(2,:))] ...
    ,[calcSem(imgOcclResPreMnNovMn(1,:)) calcSem(imgOcclResPreMnNovMn(2,:))]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
plot([4 5],[nanmean(imgOcclResPreMnNovMn(1,:)) nanmean(imgOcclResPreMnNovMn(2,:))],'r')
scatter([7 8],[nanmean(imgOcclResPostMnFamMn(1,:)) nanmean(imgOcclResPostMnFamMn(2,:))], 45, 'b', 'filled', 'LineWidth', 2), hold on
er = errorbar([7 8],[nanmean(imgOcclResPostMnFamMn(1,:)) nanmean(imgOcclResPostMnFamMn(2,:))], ...
    [calcSem(imgOcclResPostMnFamMn(1,:)) calcSem(imgOcclResPostMnFamMn(2,:))] ...
    ,[calcSem(imgOcclResPostMnFamMn(1,:)) calcSem(imgOcclResPostMnFamMn(2,:))]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
plot([7 8],[nanmean(imgOcclResPostMnFamMn(1,:)) nanmean(imgOcclResPostMnFamMn(2,:))],'b')
scatter([10 11],[nanmean(imgOcclResPostMnNovMn(1,:)) nanmean(imgOcclResPostMnNovMn(2,:))], 45, 'r', 'filled', 'LineWidth', 2), hold on
er = errorbar([10 11],[nanmean(imgOcclResPostMnNovMn(1,:)) nanmean(imgOcclResPostMnNovMn(2,:))], ...
    [calcSem(imgOcclResPostMnNovMn(1,:)) calcSem(imgOcclResPostMnNovMn(2,:))] ...
    ,[calcSem(imgOcclResPostMnNovMn(1,:)) calcSem(imgOcclResPostMnNovMn(2,:))]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
plot([10 11],[nanmean(imgOcclResPostMnNovMn(1,:)) nanmean(imgOcclResPostMnNovMn(2,:))],'r')
scatter([13 14],[nanmean(imgOcclResTaskMnFamMn(1,:)) nanmean(imgOcclResTaskMnFamMn(2,:))], 45, 'b', 'filled', 'LineWidth', 2), hold on
er = errorbar([13 14],[nanmean(imgOcclResTaskMnFamMn(1,:)) nanmean(imgOcclResTaskMnFamMn(2,:))], ...
    [calcSem(imgOcclResTaskMnFamMn(1,:)) calcSem(imgOcclResTaskMnFamMn(2,:))] ...
    ,[calcSem(imgOcclResTaskMnFamMn(1,:)) calcSem(imgOcclResTaskMnFamMn(2,:))]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;xlim([0 15])
plot([13 14],[nanmean(imgOcclResTaskMnFamMn(1,:)) nanmean(imgOcclResTaskMnFamMn(2,:))],'b')
ylabel('Response'), ;%, ylim([0 0.5])
for j = 1:3
    s(j).YLim = [0 1.5];
end
for j = 4:6
    s(j).YLim = [-0.05 0.45];
end
t(1).YLim = [0 1.2];
t(2).YLim = [0 0.32];


% === T-Test Analysis on FH vs SH per condition ===

% Prepare conditions and data matrices
condList = {
    'FullPreFam', imgFullResPreMnFamMn;
    'FullPreNov', imgFullResPreMnNovMn;
    'FullPostFam', imgFullResPostMnFamMn;
    'FullPostNov', imgFullResPostMnNovMn;
    'OcclPreFam', imgOcclResPreMnFamMn;
    'OcclPreNov', imgOcclResPreMnNovMn;
    'OcclPostFam', imgOcclResPostMnFamMn;
    'OcclPostNov', imgOcclResPostMnNovMn;
    'FullTaskFam', imgFullResTaskMnFamMn;
    'OcclTaskFam', imgOcclResTaskMnFamMn;
};

fprintf('\n--- Paired t-tests: FH vs SH ---\n');
fhShTResults = cell(size(condList,1), 5);

for i = 1:size(condList,1)
    condName = condList{i,1};
    mat = condList{i,2};

    if isempty(mat) || size(mat,2) < 2
        fprintf('%s: Skipped (not enough data)\n', condName);
        continue
    end

    fh = mat(1,:);
    sh = mat(2,:);

    [~, p, ~, stats] = ttest(fh, sh);
    fhShTResults(i,:) = {condName, mean(fh), mean(sh), stats.tstat, p};
    fprintf('%s: FH = %.3f, SH = %.3f, t(%d) = %.2f, p = %.4f\n', condName, mean(fh), mean(sh), stats.df, stats.tstat, p);

    % Add significance stars to existing plots
    if p < 0.001
        stars = '***';
    elseif p < 0.01
        stars = '**';
    elseif p < 0.05
        stars = '*';
    else
        stars = '';
    end

%     if ~isempty(stars)
%         % Find subplot column to place text above SH
%         % Each pair is plotted with spacing: 1-2, 4-5, 7-8, 10-11, 13-14
%         subplotCol = (i-1)*3 + 2;
%         try
%             subplotIdx = findobj(gcf, 'Type', 'axes', '-and', 'Tag', sprintf('t%d', ceil(i/5)));
%             axes(subplotIdx);
%         catch
%             subplot(2,4,4 + floor((i-1)/5)*4); % fallback placement
%         end
%         y = mean(sh) + 0.05;
%         x = [2, 5, 8, 11, 14];
%         xIdx = mod(i-1, 5);
%         text(x(xIdx+1), y, stars, 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
%     end
end

% Convert to table
fhShStatsTable = cell2table(fhShTResults, 'VariableNames', {'Condition', 'MeanFH', 'MeanSH', 'tStat', 'pValue'});

% Optional: save to .mat or print summary
% save('fhShStatsTable.mat', 'fhShStatsTable');

% View table
disp(fhShStatsTable);



%%
famIdx = [1 2 4 5];
novIdx = [3 6];

% plot over trials
warning on 

if nfiles == 5
    if regressRun
        load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\traces_regressrun_ActiveL5.mat')
    else
        load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\traces_noregressrun_ActiveL5.mat')
    end
end

if nfiles == 6
    vecAxTaskSp = vecAxTask<0;
    vecAxTaskSt = vecAxTask>0.2 & vecAxTask<1;
end

imgFullResPre = [];
imgOcclResPre = [];
imgFullResPost = [];
imgOcclResPost = [];
imgFullResTask = [];
imgOcclResTask = [];

for i = 1:nfiles
    imgFullResPre = cat(4, imgFullResPre, datastructPre(i).imgFullRes);
    imgOcclResPre = cat(4, imgOcclResPre, datastructPre(i).imgOcclRes);
    imgFullResPost = cat(4, imgFullResPost, datastructPost(i).imgFullRes);
    imgOcclResPost = cat(4, imgOcclResPost, datastructPost(i).imgOcclRes);
    imgFullResTask = cat(4, imgFullResTask, datastructTask(i).imgFullRes);
    imgOcclResTask = cat(4, imgOcclResTask, datastructTask(i).imgOcclRes);
end

clear imgFullResPreMnFamMn imgOcclResPreMnFamMn imgFullResPostMnFamMn... 
imgFullResPreMnNovMn imgOcclResPreMnNovMn imgFullResPostMnNovMn...
imgFullResTaskMnFamMn imgOcclResTaskMnFamMn imgOcclResPostMnFamMn imgOcclResPostMnNovMn

fh = 1:5;
sh = 16:20;

respThreshold = 0;
smoothed = true; % do you want to smooth the traces
smoothVal = 0.5; % by how much?
sub = 1:20;

imgFullResPreMn = squeeze(mean(imgFullResPre(vecAxSt,:,:,:)))-squeeze(mean(imgFullResPre(vecAxSp,:,:,:)));
imgFullResPreMnFam = squeeze(mean(imgFullResPreMn(famIdx,:,:)));
if smoothed, imgFullResPreMnFam = smoothG(imgFullResPreMnFam,smoothVal); end
imgFullResPreMnFam = imgFullResPreMnFam(:,mean(imgFullResPreMnFam(sub,:))>respThreshold);
mouseIdxFullPreFam = mouseIDPre(mean(imgFullResPreMnFam(sub,:))>respThreshold);
imgFullResPreMnFamMn(1,:) = mean(imgFullResPreMnFam(fh,:));
imgFullResPreMnFamMn(2,:) = mean(imgFullResPreMnFam(sh,:));
% imgFullResPreMnFamMn = imgFullResPreMnFamMn./imgFullResPreMnFamMn(1,:);
imgFullResPreMnNov = squeeze(mean(imgFullResPreMn(novIdx,:,:)));
if smoothed, imgFullResPreMnNov = smoothG(imgFullResPreMnNov,smoothVal);
imgFullResPreMnNov = imgFullResPreMnNov(:,mean(imgFullResPreMnNov(sub,:))>respThreshold); end
mouseIdxFullPreNov = mouseIDPre(mean(imgFullResPreMnNov(sub,:))>respThreshold);
imgFullResPreMnNovMn(1,:) = mean(imgFullResPreMnNov(fh,:));
imgFullResPreMnNovMn(2,:) = mean(imgFullResPreMnNov(sh,:));

% --- Occluded Pre ---
imgOcclResPreMn = squeeze(mean(imgOcclResPre(vecAxSt,:,:,:))) - squeeze(mean(imgOcclResPre(vecAxSp,:,:,:)));

imgOcclResPreMnFam = squeeze(mean(imgOcclResPreMn(famIdx,:,:)));
if smoothed, imgOcclResPreMnFam = smoothG(imgOcclResPreMnFam,smoothVal); end
imgOcclResPreMnFam = imgOcclResPreMnFam(:, mean(imgOcclResPreMnFam(sub,:)) > respThreshold);
mouseIdxOcclPreFam = mouseIDPre(mean(imgOcclResPreMnFam(sub,:)) > respThreshold);
imgOcclResPreMnFamMn(1,:) = mean(imgOcclResPreMnFam(fh,:));
imgOcclResPreMnFamMn(2,:) = mean(imgOcclResPreMnFam(sh,:));

imgOcclResPreMnNov = squeeze(mean(imgOcclResPreMn(novIdx,:,:)));
if smoothed, imgOcclResPreMnNov = smoothG(imgOcclResPreMnNov,smoothVal); end
imgOcclResPreMnNov = imgOcclResPreMnNov(:, mean(imgOcclResPreMnNov(sub,:)) > respThreshold);
mouseIdxOcclPreNov = mouseIDPre(mean(imgOcclResPreMnNov(sub,:)) > respThreshold);
imgOcclResPreMnNovMn(1,:) = mean(imgOcclResPreMnNov(fh,:));
imgOcclResPreMnNovMn(2,:) = mean(imgOcclResPreMnNov(sh,:));

% --- Full Post ---
imgFullResPostMn = squeeze(mean(imgFullResPost(vecAxSt,:,:,:))) - squeeze(mean(imgFullResPost(vecAxSp,:,:,:)));

imgFullResPostMnFam = squeeze(mean(imgFullResPostMn(famIdx,:,:)));
if smoothed, imgFullResPostMnFam = smoothG(imgFullResPostMnFam,smoothVal); end
imgFullResPostMnFam = imgFullResPostMnFam(:, mean(imgFullResPostMnFam(sub,:)) > respThreshold);
mouseIdxFullPostFam = mouseIDPost(mean(imgFullResPostMnFam(sub,:)) > respThreshold);
imgFullResPostMnFamMn(1,:) = mean(imgFullResPostMnFam(fh,:));
imgFullResPostMnFamMn(2,:) = mean(imgFullResPostMnFam(sh,:));

imgFullResPostMnNov = squeeze(mean(imgFullResPostMn(novIdx,:,:)));
if smoothed, imgFullResPostMnNov = smoothG(imgFullResPostMnNov,smoothVal); end
imgFullResPostMnNov = imgFullResPostMnNov(:, mean(imgFullResPostMnNov(sub,:)) > respThreshold);
mouseIdxFullPostNov = mouseIDPost(mean(imgFullResPostMnNov(sub,:)) > respThreshold);
imgFullResPostMnNovMn(1,:) = mean(imgFullResPostMnNov(fh,:));
imgFullResPostMnNovMn(2,:) = mean(imgFullResPostMnNov(sh,:));

% --- Occluded Post ---
imgOcclResPostMn = squeeze(mean(imgOcclResPost(vecAxSt,:,:,:))) - squeeze(mean(imgOcclResPost(vecAxSp,:,:,:)));

imgOcclResPostMnFam = squeeze(mean(imgOcclResPostMn(famIdx,:,:)));
if smoothed, imgOcclResPostMnFam = smoothG(imgOcclResPostMnFam,smoothVal); end
imgOcclResPostMnFam = imgOcclResPostMnFam(:, mean(imgOcclResPostMnFam(sub,:)) > respThreshold);
mouseIdxOcclPostFam = mouseIDPost(mean(imgOcclResPostMnFam(sub,:)) > respThreshold);
imgOcclResPostMnFamMn(1,:) = mean(imgOcclResPostMnFam(fh,:));
imgOcclResPostMnFamMn(2,:) = mean(imgOcclResPostMnFam(sh,:));

imgOcclResPostMnNov = squeeze(mean(imgOcclResPostMn(novIdx,:,:)));
if smoothed, imgOcclResPostMnNov = smoothG(imgOcclResPostMnNov,smoothVal); end
imgOcclResPostMnNov = imgOcclResPostMnNov(:, mean(imgOcclResPostMnNov(sub,:)) > respThreshold);
mouseIdxOcclPostNov = mouseIDPost(mean(imgOcclResPostMnNov(sub,:)) > respThreshold);
imgOcclResPostMnNovMn(1,:) = mean(imgOcclResPostMnNov(fh,:));
imgOcclResPostMnNovMn(2,:) = mean(imgOcclResPostMnNov(sh,:));

% --- Full Task (Fam only) ---
imgFullResTaskMn = squeeze(mean(imgFullResTask(vecAxTaskSt,:,:,:))) - squeeze(mean(imgFullResTask(vecAxTaskSp,:,:,:)));
imgFullResTaskMnFam = squeeze(mean(imgFullResTaskMn, 1));
if smoothed, imgFullResTaskMnFam = smoothG(imgFullResTaskMnFam,smoothVal); end
imgFullResTaskMnFam = imgFullResTaskMnFam(:, mean(imgFullResTaskMnFam(sub,:)) > respThreshold);
mouseIdxFullTaskFam = mouseIDTask(mean(imgFullResTaskMnFam(sub,:)) > respThreshold);
imgFullResTaskMnFamMn(1,:) = mean(imgFullResTaskMnFam(fh,:));
imgFullResTaskMnFamMn(2,:) = mean(imgFullResTaskMnFam(sh,:));

% --- Occluded Task (Fam only) ---
imgOcclResTaskMn = squeeze(mean(imgOcclResTask(vecAxTaskSt,:,:,:))) - squeeze(mean(imgOcclResTask(vecAxTaskSp,:,:,:)));
imgOcclResTaskMnFam = squeeze(mean(imgOcclResTaskMn, 1));
if smoothed, imgOcclResTaskMnFam = smoothG(imgOcclResTaskMnFam,smoothVal); end
imgOcclResTaskMnFam = imgOcclResTaskMnFam(:, mean(imgOcclResTaskMnFam(sub,:)) > respThreshold);
mouseIdxOcclTaskFam = mouseIDTask(mean(imgOcclResTaskMnFam(sub,:)) > respThreshold);
imgOcclResTaskMnFamMn(1,:) = mean(imgOcclResTaskMnFam(fh,:));
imgOcclResTaskMnFamMn(2,:) = mean(imgOcclResTaskMnFam(sh,:));


clear s t

figure('Position', [293         309        1164         571])
s(1) = subplot(2,4,1);
shadedErrorBar(1:20,mean(imgFullResPreMnFam,2)...
    ,std(imgFullResPreMnFam,0,2)/sqrt(size(imgFullResPreMnFam,2)), 'lineProps', 'b'); hold on
shadedErrorBar(1:20,mean(imgFullResPreMnNov,2)...
    ,std(imgFullResPreMnNov,0,2)/sqrt(size(imgFullResPreMnNov,2)), 'lineProps', 'r'); 
ylabel('Response'), xlabel('Trials'), title('NO Pre'), 
s(2) = subplot(2,4,2);
shadedErrorBar(1:20,mean(imgFullResPostMnFam,2)...
    ,std(imgFullResPostMnFam,0,2)/sqrt(size(imgFullResPostMnFam,2)), 'lineProps', 'b'); hold on
shadedErrorBar(1:20,mean(imgFullResPostMnNov,2)...
    ,std(imgFullResPostMnNov,0,2)/sqrt(size(imgFullResPostMnNov,2)), 'lineProps', 'r'); 
ylabel('Response'), xlabel('Trials'), title('NO Post'), 
s(3) = subplot(2,4,3);
shadedErrorBar(1:20, mean(imgFullResTaskMnFam,2), ...
    std(imgFullResTaskMnFam,0,2)/sqrt(size(imgFullResTaskMnFam,2)), 'lineProps', 'b'); 
ylabel('Response'), xlabel('Trials'), title('NO Task'), 


s(4) = subplot(2,4,5);
shadedErrorBar(1:20,mean(imgOcclResPreMnFam,2)...
    ,std(imgOcclResPreMnFam,0,2)/sqrt(size(imgOcclResPreMnFam,2)), 'lineProps', 'b'); hold on
shadedErrorBar(1:20,mean(imgOcclResPreMnNov,2)...
    ,std(imgOcclResPreMnNov,0,2)/sqrt(size(imgOcclResPreMnNov,2)), 'lineProps', 'r'); 
ylabel('Response'), xlabel('Trials'), title('O Pre'), 
s(5) = subplot(2,4,6);
shadedErrorBar(1:20,mean(imgOcclResPostMnFam,2)...
    ,std(imgOcclResPostMnFam,0,2)/sqrt(size(imgOcclResPostMnFam,2)), 'lineProps', 'b'); hold on
shadedErrorBar(1:20,mean(imgOcclResPostMnNov,2)...
    ,std(imgOcclResPostMnNov,0,2)/sqrt(size(imgOcclResPostMnNov,2)), 'lineProps', 'r'); 
ylabel('Response'), xlabel('Trials'), title('O Post'), 
s(6) = subplot(2,4,7);
shadedErrorBar(1:20, mean(imgOcclResTaskMnFam,2), ...
    std(imgOcclResTaskMnFam,0,2)/sqrt(size(imgOcclResTaskMnFam,2)), 'lineProps', 'b'); 
ylabel('Response'), xlabel('Trials'), title('O Task'), 


t(1) = subplot(2,4,4);
scatter([1 2],[nanmean(imgFullResPreMnFamMn(1,:)) nanmean(imgFullResPreMnFamMn(2,:))], 45, 'b', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2],[nanmean(imgFullResPreMnFamMn(1,:)) nanmean(imgFullResPreMnFamMn(2,:))], ...
    [calcSem(imgFullResPreMnFamMn(1,:)) calcSem(imgFullResPreMnFamMn(2,:))] ...
    ,[calcSem(imgFullResPreMnFamMn(1,:)) calcSem(imgFullResPreMnFamMn(2,:))]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
plot([1 2],[nanmean(imgFullResPreMnFamMn(1,:)) nanmean(imgFullResPreMnFamMn(2,:))],'b')
scatter([4 5],[nanmean(imgFullResPreMnNovMn(1,:)) nanmean(imgFullResPreMnNovMn(2,:))], 45, 'r', 'filled', 'LineWidth', 2), hold on
er = errorbar([4 5],[nanmean(imgFullResPreMnNovMn(1,:)) nanmean(imgFullResPreMnNovMn(2,:))], ...
    [calcSem(imgFullResPreMnNovMn(1,:)) calcSem(imgFullResPreMnNovMn(2,:))] ...
    ,[calcSem(imgFullResPreMnNovMn(1,:)) calcSem(imgFullResPreMnNovMn(2,:))]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;xlim([0 12])
plot([4 5],[nanmean(imgFullResPreMnNovMn(1,:)) nanmean(imgFullResPreMnNovMn(2,:))],'r')
scatter([7 8],[nanmean(imgFullResPostMnFamMn(1,:)) nanmean(imgFullResPostMnFamMn(2,:))], 45, 'b', 'filled', 'LineWidth', 2), hold on
er = errorbar([7 8],[nanmean(imgFullResPostMnFamMn(1,:)) nanmean(imgFullResPostMnFamMn(2,:))], ...
    [calcSem(imgFullResPostMnFamMn(1,:)) calcSem(imgFullResPostMnFamMn(2,:))] ...
    ,[calcSem(imgFullResPostMnFamMn(1,:)) calcSem(imgFullResPostMnFamMn(2,:))]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
plot([7 8],[nanmean(imgFullResPostMnFamMn(1,:)) nanmean(imgFullResPostMnFamMn(2,:))],'b')
scatter([10 11],[nanmean(imgFullResPostMnNovMn(1,:)) nanmean(imgFullResPostMnNovMn(2,:))], 45, 'r', 'filled', 'LineWidth', 2), hold on
er = errorbar([10 11],[nanmean(imgFullResPostMnNovMn(1,:)) nanmean(imgFullResPostMnNovMn(2,:))], ...
    [calcSem(imgFullResPostMnNovMn(1,:)) calcSem(imgFullResPostMnNovMn(2,:))] ...
    ,[calcSem(imgFullResPostMnNovMn(1,:)) calcSem(imgFullResPostMnNovMn(2,:))]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
plot([10 11],[nanmean(imgFullResPostMnNovMn(1,:)) nanmean(imgFullResPostMnNovMn(2,:))],'r')
scatter([13 14],[nanmean(imgFullResTaskMnFamMn(1,:)) nanmean(imgFullResTaskMnFamMn(2,:))], 45, 'b', 'filled', 'LineWidth', 2), hold on
er = errorbar([13 14],[nanmean(imgFullResTaskMnFamMn(1,:)) nanmean(imgFullResTaskMnFamMn(2,:))], ...
    [calcSem(imgFullResTaskMnFamMn(1,:)) calcSem(imgFullResTaskMnFamMn(2,:))] ...
    ,[calcSem(imgFullResTaskMnFamMn(1,:)) calcSem(imgFullResTaskMnFamMn(2,:))]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;xlim([0 15])
plot([13 14],[nanmean(imgFullResTaskMnFamMn(1,:)) nanmean(imgFullResTaskMnFamMn(2,:))],'b')
ylabel('Response'), 

t(2) = subplot(2,4,8);
scatter([1 2],[nanmean(imgOcclResPreMnFamMn(1,:)) nanmean(imgOcclResPreMnFamMn(2,:))], 45, 'b', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2],[nanmean(imgOcclResPreMnFamMn(1,:)) nanmean(imgOcclResPreMnFamMn(2,:))], ...
    [calcSem(imgOcclResPreMnFamMn(1,:)) calcSem(imgOcclResPreMnFamMn(2,:))] ...
    ,[calcSem(imgOcclResPreMnFamMn(1,:)) calcSem(imgOcclResPreMnFamMn(2,:))]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
plot([1 2],[nanmean(imgOcclResPreMnFamMn(1,:)) nanmean(imgOcclResPreMnFamMn(2,:))],'b')
scatter([4 5],[nanmean(imgOcclResPreMnNovMn(1,:)) nanmean(imgOcclResPreMnNovMn(2,:))], 45, 'r', 'filled', 'LineWidth', 2), hold on
er = errorbar([4 5],[nanmean(imgOcclResPreMnNovMn(1,:)) nanmean(imgOcclResPreMnNovMn(2,:))], ...
    [calcSem(imgOcclResPreMnNovMn(1,:)) calcSem(imgOcclResPreMnNovMn(2,:))] ...
    ,[calcSem(imgOcclResPreMnNovMn(1,:)) calcSem(imgOcclResPreMnNovMn(2,:))]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
plot([4 5],[nanmean(imgOcclResPreMnNovMn(1,:)) nanmean(imgOcclResPreMnNovMn(2,:))],'r')
scatter([7 8],[nanmean(imgOcclResPostMnFamMn(1,:)) nanmean(imgOcclResPostMnFamMn(2,:))], 45, 'b', 'filled', 'LineWidth', 2), hold on
er = errorbar([7 8],[nanmean(imgOcclResPostMnFamMn(1,:)) nanmean(imgOcclResPostMnFamMn(2,:))], ...
    [calcSem(imgOcclResPostMnFamMn(1,:)) calcSem(imgOcclResPostMnFamMn(2,:))] ...
    ,[calcSem(imgOcclResPostMnFamMn(1,:)) calcSem(imgOcclResPostMnFamMn(2,:))]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
plot([7 8],[nanmean(imgOcclResPostMnFamMn(1,:)) nanmean(imgOcclResPostMnFamMn(2,:))],'b')
scatter([10 11],[nanmean(imgOcclResPostMnNovMn(1,:)) nanmean(imgOcclResPostMnNovMn(2,:))], 45, 'r', 'filled', 'LineWidth', 2), hold on
er = errorbar([10 11],[nanmean(imgOcclResPostMnNovMn(1,:)) nanmean(imgOcclResPostMnNovMn(2,:))], ...
    [calcSem(imgOcclResPostMnNovMn(1,:)) calcSem(imgOcclResPostMnNovMn(2,:))] ...
    ,[calcSem(imgOcclResPostMnNovMn(1,:)) calcSem(imgOcclResPostMnNovMn(2,:))]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
plot([10 11],[nanmean(imgOcclResPostMnNovMn(1,:)) nanmean(imgOcclResPostMnNovMn(2,:))],'r')
scatter([13 14],[nanmean(imgOcclResTaskMnFamMn(1,:)) nanmean(imgOcclResTaskMnFamMn(2,:))], 45, 'b', 'filled', 'LineWidth', 2), hold on
er = errorbar([13 14],[nanmean(imgOcclResTaskMnFamMn(1,:)) nanmean(imgOcclResTaskMnFamMn(2,:))], ...
    [calcSem(imgOcclResTaskMnFamMn(1,:)) calcSem(imgOcclResTaskMnFamMn(2,:))] ...
    ,[calcSem(imgOcclResTaskMnFamMn(1,:)) calcSem(imgOcclResTaskMnFamMn(2,:))]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;xlim([0 15])
plot([13 14],[nanmean(imgOcclResTaskMnFamMn(1,:)) nanmean(imgOcclResTaskMnFamMn(2,:))],'b')
ylabel('Response'), ;%, ylim([0 0.5])
for j = 1:3
    s(j).YLim = [0 1.5];
end
for j = 4:6
%     s(j).YLim = [-0.05 0.45];
    s(j).YLim = [0 1.5];
end
t(1).YLim = [0 1.2];
t(2).YLim = [0 1.2];
% t(2).YLim = [0 0.32];

if save_fig
    func_save_fig('L23_og_resOverTrials')
    func_save_fig('L5_og_resOverTrials')
end

%
conds = {
    'FullPreFam',   imgFullResPreMnFam,   mouseIDPre;
    'FullPreNov',   imgFullResPreMnNov,   mouseIDPre;
    'FullPostFam',  imgFullResPostMnFam,  mouseIDPost;
    'FullPostNov',  imgFullResPostMnNov,  mouseIDPost;
    'OcclPreFam',   imgOcclResPreMnFam,   mouseIDPre;
    'OcclPreNov',   imgOcclResPreMnNov,   mouseIDPre;
    'OcclPostFam',  imgOcclResPostMnFam,  mouseIDPost;
    'OcclPostNov',  imgOcclResPostMnNov,  mouseIDPost;
    'FullTaskFam',  imgFullResTaskMnFam,  mouseIDTask;
    'OcclTaskFam',  imgOcclResTaskMnFam,  mouseIDTask;
};

resultsLMEM = cell(size(conds,1), 5);
resultsTTest = cell(size(conds,1), 5);

fprintf('--- Slope Analysis with and without Random Effects (filtered) ---\n');

for i = 1:size(conds,1)
    condName  = conds{i,1};
    dataMat   = conds{i,2};  % [time x neurons]
    mouseIDs  = conds{i,3};  % [1 x neurons] or [1 x n]

    % Filter: neurons with mean response > threshold
    meanResp = mean(dataMat, 1);
    validIdx = meanResp > respThreshold;

    if sum(validIdx) < 2
        warning('%s: Skipping (too few neurons after filtering)', condName);
        resultsLMEM(i,:) = {condName, NaN, NaN, NaN, NaN};
        resultsTTest(i,:) = {condName, NaN, NaN, NaN, NaN};
        continue
    end

    dataMat = dataMat(:, validIdx);
    mouseIDs = categorical(mouseIDs(validIdx));
    [t, n] = size(dataMat);
    x = (1:t)';

    slopes = nan(n,1);
    for j = 1:n
        p = polyfit(x, dataMat(:,j), 1);
        slopes(j) = p(1);
    end

    % -- LMEM
    T = table(slopes(:), mouseIDs(:), 'VariableNames', {'slope','mouseID'});
    try
        lme = fitlme(T, 'slope ~ 1 + (1|mouseID)', ...
            'FitMethod', 'REML', 'CheckHessian', true, 'StartMethod', 'random');
        [fe,~,stats] = fixedEffects(lme);

        resultsLMEM(i,:) = {condName, fe, stats.tStat, stats.DF, stats.pValue};
        fprintf('[LME] %s: slope = %.4f, t(%d) = %.2f, p = %.4f\n', ...
            condName, fe, stats.DF, stats.tStat, stats.pValue);
    catch
        warning('%s: LMEM failed.', condName);
        resultsLMEM(i,:) = {condName, NaN, NaN, NaN, NaN};
    end

    % -- t-test alternative
    mdl = fitlm(ones(n,1), slopes);
    coeff = mdl.Coefficients;
    resultsTTest(i,:) = {
        condName, ...
        coeff.Estimate(1), ...
        coeff.tStat(1), ...
        mdl.DFE, ...
        coeff.pValue(1)};
    fprintf('[t-test] %s: slope = %.4f, t(%d) = %.2f, p = %.4f\n', ...
        condName, coeff.Estimate(1), mdl.DFE, coeff.tStat(1), coeff.pValue(1));
end

% Convert to tables
statsL23adaptationFiltered = cell2table(resultsLMEM, ...
    'VariableNames', {'Condition', 'MeanSlope', 'tStat', 'DF', 'pValue'});
statsL23adaptationFilteredTTest = cell2table(resultsTTest, ...
    'VariableNames', {'Condition', 'MeanSlope', 'tStat', 'DF', 'pValue'});

% Choose which results table to visualize:
plotTable = statsL23adaptationFilteredTTest;  % or statsL23adaptationFiltered for LMEM

% Extract info
condsToPlot = plotTable.Condition;
meanSlopes = plotTable.MeanSlope;
tStats     = plotTable.tStat;
df         = plotTable.DF;
pValues    = plotTable.pValue;

% Categorize conditions
famIdx  = contains(condsToPlot, 'Fam') & ~contains(condsToPlot, 'Task');
novIdx  = contains(condsToPlot, 'Nov');
taskIdx = contains(condsToPlot, 'Task');

figure('Position', [310    90   684   400])

categories = {
    'Full', contains(plotTable.Condition, 'Full');
    'Occl', contains(plotTable.Condition, 'Occl');
};

for k = 1:2  % Only Full and Occl
    subplot(1,2,k)
    catName = categories{k,1};
    idx = categories{k,2};

    conds = condsToPlot(idx);
    slopes = meanSlopes(idx);
    tVals = tStats(idx);
    dfs = df(idx);
    pVals = pValues(idx);

    hold on
    for i = 1:length(slopes)
        % Color by category
        if contains(catName, 'Full')
            col = [0 0 0];  % black
        else
            col = [1 0 0];  % red
        end

        % Plot slope + SEM
        scatter(i, slopes(i), 60, col, 'filled')
        if ~isnan(tVals(i)) && ~isnan(dfs(i))
            sem = abs(slopes(i) ./ tVals(i));
            errorbar(i, slopes(i), sem, 'Color', col, 'LineWidth', 1.5, 'CapSize', 0)
        end

        % Annotate significance asterisks
        if ~isnan(pVals(i))
            if pVals(i) < 0.001
                stars = '***';
            elseif pVals(i) < 0.01
                stars = '**';
            elseif pVals(i) < 0.05
                stars = '*';
            else
                stars = '';
            end

            if ~isempty(stars)
                yOffset = 0.009;
                text(i, slopes(i) - yOffset, stars, ...
                    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold')
            end
        end
    end

    % Formatting
    ylim([-0.05 0.01])
    xlim([0 6])
    set(gca, 'YDir', 'reverse')  % More negative = more adaptation
    yline(0, '--', 'Color', [0.6 0.6 0.6])
    xticks(1:length(conds))
    xticklabels(conds)
    xtickangle(45)
    ylabel('Adaptation slope (flipped)')
    title(catName)
    
end

if save_fig
    func_save_fig('L23_adaptation_slopes_errorbar')
    func_save_fig('L5_adaptation_slopes_errorbar')
end

%% First half vs second half
warning on

condData = {
    'FullPreFam',   imgFullResPreMnFamMn,   mouseIdxFullPreFam;
    'FullPreNov',   imgFullResPreMnNovMn,   mouseIdxFullPreNov;
    'FullPostFam',  imgFullResPostMnFamMn,  mouseIdxFullPostFam;
    'FullPostNov',  imgFullResPostMnNovMn,  mouseIdxFullPostNov;
    'OcclPreFam',   imgOcclResPreMnFamMn,   mouseIdxOcclPreFam;
    'OcclPreNov',   imgOcclResPreMnNovMn,   mouseIdxOcclPreNov;
    'OcclPostFam',  imgOcclResPostMnFamMn,  mouseIdxOcclPostFam;
    'OcclPostNov',  imgOcclResPostMnNovMn,  mouseIdxOcclPostNov;
    'FullTaskFam',  imgFullResTaskMnFamMn,  mouseIdxFullTaskFam;
    'OcclTaskFam',  imgOcclResTaskMnFamMn,  mouseIdxOcclTaskFam;
};


resultsLMEM_fhsh = cell(size(condData,1), 5);

fprintf('--- LMEM on First-Half vs Second-Half Response ---\n');

for i = 1:size(condData,1)
    condName = condData{i,1};
    respMat  = condData{i,2}; % [2 x neurons]
    mouseIDs = condData{i,3};

    if isempty(respMat) || size(respMat,2) < 2
        warning('%s: Skipping (not enough data)', condName);
        resultsLMEM_fhsh(i,:) = {condName, NaN, NaN, NaN, NaN};
        continue;
    end

    % Format for LMEM
    n = size(respMat,2);
    response = respMat(:);
    timepoint = categorical([repmat({'fh'}, 1, n), repmat({'sh'}, 1, n)]');
    mouseID = categorical([mouseIDs, mouseIDs]');

    tbl = table(response, timepoint, mouseID);

    try
        lme = fitlme(tbl, 'response ~ timepoint + (1|mouseID)');
        [fe, ~, stats] = fixedEffects(lme);

        resultsLMEM_fhsh(i,:) = {condName, fe(2), stats.tStat(2), stats.DF(2), stats.pValue(2)};
        fprintf('[LMEM] %s: Δ = %.4f, t(%d) = %.2f, p = %.4f\n', ...
            condName, fe(2), stats.DF(2), stats.tStat(2), stats.pValue(2));
    catch
        warning('%s: LMEM failed.', condName);
        resultsLMEM_fhsh(i,:) = {condName, NaN, NaN, NaN, NaN};
    end
end

% Store in a table
adaptStats_fhsh = cell2table(resultsLMEM_fhsh, ...
    'VariableNames', {'Condition', 'Delta', 'tStat', 'DF', 'pValue'});

resultsTTest_fhsh = cell(size(condData,1), 5);

fprintf('--- Paired t-test on First-Half vs Second-Half Responses ---\n');

for i = 1:size(condData,1)
    condName = condData{i,1};
    respMat  = condData{i,2}; % [2 x neurons]

    if isempty(respMat) || size(respMat,2) < 2
        warning('%s: Skipping (not enough data)', condName);
        resultsTTest_fhsh(i,:) = {condName, NaN, NaN, NaN, NaN};
        continue;
    end

    fh = respMat(1,:);
    sh = respMat(2,:);

    [~, p, ~, stats] = ttest(fh, sh);

    delta = mean(sh - fh);  % average change from fh to sh
    resultsTTest_fhsh(i,:) = {condName, delta, stats.tstat, stats.df, p};

    fprintf('[t-test] %s: Δ = %.4f, t(%d) = %.2f, p = %.4f\n', ...
        condName, delta, stats.df, stats.tstat, p);
end

adaptStatsTTest_fhsh = cell2table(resultsTTest_fhsh, ...
    'VariableNames', {'Condition', 'Delta', 'tStat', 'DF', 'pValue'});






% Optional save
% save('adaptationStats_filtered.mat', 'statsL5adaptationFiltered', 'statsL5adaptationFilteredTTest');

%% final adaptation for fh vs sh ttest for paper?
% -------------------- Setup & Loading --------------------
warning on

famIdx = [1 2 4 5];
novIdx = [3 6];

if nfiles == 5
    if regressRun
        load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\traces_regressrun_ActiveL5.mat')
    else
        load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\traces_noregressrun_ActiveL5.mat')
    end
elseif nfiles == 6
    vecAxTaskSp = vecAxTask < 0;
    vecAxTaskSt = vecAxTask > 0.2 & vecAxTask < 1;
end

% Collect across files (keeps dims: time x neurons x files -> 4th dim)
imgFullResPre  = [];
imgOcclResPre  = [];
imgFullResPost = [];
imgOcclResPost = [];
imgFullResTask = [];
imgOcclResTask = [];

for i = 1:nfiles
    imgFullResPre  = cat(4, imgFullResPre,  datastructPre(i).imgFullRes);
    imgOcclResPre  = cat(4, imgOcclResPre,  datastructPre(i).imgOcclRes);
    imgFullResPost = cat(4, imgFullResPost, datastructPost(i).imgFullRes);
    imgOcclResPost = cat(4, imgOcclResPost, datastructPost(i).imgOcclRes);
    imgFullResTask = cat(4, imgFullResTask, datastructTask(i).imgFullRes);
    imgOcclResTask = cat(4, imgOcclResTask, datastructTask(i).imgOcclRes);
end

% -------------------- Parameters --------------------
fh = 1:10;              % first half trials
sh = 11:20;             % second half trials
respThreshold = 0;  % filter bad/non-responsive units
smoothed = true;        % smooth traces?
smoothVal = 0.5;        % smoothing amount
sub = 1:20;             % window for mean filtering
save_fig = exist('save_fig','var') && save_fig;  % keep prior behavior if defined

% -------------------- Helper: build condition matrices --------------------
% Mean over stimulus vs spacer (trial-averaged timecourses), then fam/nov split
clear img*Mn* mouseIdx*

% --- Full Pre ---
imgFullResPreMn = squeeze(mean(imgFullResPre(vecAxSt,:,:,:))) - squeeze(mean(imgFullResPre(vecAxSp,:,:,:)));

imgFullResPreMnFam = squeeze(mean(imgFullResPreMn(famIdx,:,:)));
if smoothed, imgFullResPreMnFam = smoothG(imgFullResPreMnFam, smoothVal); end
imgFullResPreMnFam = imgFullResPreMnFam(:, mean(imgFullResPreMnFam(sub,:)) > respThreshold);
mouseIdxFullPreFam = mouseIDPre(mean(imgFullResPreMnFam(sub,:)) > respThreshold);
imgFullResPreMnFamMn = [mean(imgFullResPreMnFam(fh,:)); mean(imgFullResPreMnFam(sh,:))];

imgFullResPreMnNov = squeeze(mean(imgFullResPreMn(novIdx,:,:)));
if smoothed, imgFullResPreMnNov = smoothG(imgFullResPreMnNov, smoothVal); end
imgFullResPreMnNov = imgFullResPreMnNov(:, mean(imgFullResPreMnNov(sub,:)) > respThreshold);
mouseIdxFullPreNov = mouseIDPre(mean(imgFullResPreMnNov(sub,:)) > respThreshold);
imgFullResPreMnNovMn = [mean(imgFullResPreMnNov(fh,:)); mean(imgFullResPreMnNov(sh,:))];

% --- Occluded Pre ---
imgOcclResPreMn = squeeze(mean(imgOcclResPre(vecAxSt,:,:,:))) - squeeze(mean(imgOcclResPre(vecAxSp,:,:,:)));

imgOcclResPreMnFam = squeeze(mean(imgOcclResPreMn(famIdx,:,:)));
if smoothed, imgOcclResPreMnFam = smoothG(imgOcclResPreMnFam, smoothVal); end
imgOcclResPreMnFam = imgOcclResPreMnFam(:, mean(imgOcclResPreMnFam(sub,:)) > respThreshold);
mouseIdxOcclPreFam = mouseIDPre(mean(imgOcclResPreMnFam(sub,:)) > respThreshold);
imgOcclResPreMnFamMn = [mean(imgOcclResPreMnFam(fh,:)); mean(imgOcclResPreMnFam(sh,:))];

imgOcclResPreMnNov = squeeze(mean(imgOcclResPreMn(novIdx,:,:)));
if smoothed, imgOcclResPreMnNov = smoothG(imgOcclResPreMnNov, smoothVal); end
imgOcclResPreMnNov = imgOcclResPreMnNov(:, mean(imgOcclResPreMnNov(sub,:)) > respThreshold);
mouseIdxOcclPreNov = mouseIDPre(mean(imgOcclResPreMnNov(sub,:)) > respThreshold);
imgOcclResPreMnNovMn = [mean(imgOcclResPreMnNov(fh,:)); mean(imgOcclResPreMnNov(sh,:))];

% --- Full Post ---
imgFullResPostMn = squeeze(mean(imgFullResPost(vecAxSt,:,:,:))) - squeeze(mean(imgFullResPost(vecAxSp,:,:,:)));

imgFullResPostMnFam = squeeze(mean(imgFullResPostMn(famIdx,:,:)));
if smoothed, imgFullResPostMnFam = smoothG(imgFullResPostMnFam, smoothVal); end
imgFullResPostMnFam = imgFullResPostMnFam(:, mean(imgFullResPostMnFam(sub,:)) > respThreshold);
mouseIdxFullPostFam = mouseIDPost(mean(imgFullResPostMnFam(sub,:)) > respThreshold);
imgFullResPostMnFamMn = [mean(imgFullResPostMnFam(fh,:)); mean(imgFullResPostMnFam(sh,:))];

imgFullResPostMnNov = squeeze(mean(imgFullResPostMn(novIdx,:,:)));
if smoothed, imgFullResPostMnNov = smoothG(imgFullResPostMnNov, smoothVal); end
imgFullResPostMnNov = imgFullResPostMnNov(:, mean(imgFullResPostMnNov(sub,:)) > respThreshold);
mouseIdxFullPostNov = mouseIDPost(mean(imgFullResPostMnNov(sub,:)) > respThreshold);
imgFullResPostMnNovMn = [mean(imgFullResPostMnNov(fh,:)); mean(imgFullResPostMnNov(sh,:))];

% --- Occluded Post ---
imgOcclResPostMn = squeeze(mean(imgOcclResPost(vecAxSt,:,:,:))) - squeeze(mean(imgOcclResPost(vecAxSp,:,:,:)));

imgOcclResPostMnFam = squeeze(mean(imgOcclResPostMn(famIdx,:,:)));
if smoothed, imgOcclResPostMnFam = smoothG(imgOcclResPostMnFam, smoothVal); end
imgOcclResPostMnFam = imgOcclResPostMnFam(:, mean(imgOcclResPostMnFam(sub,:)) > respThreshold);
mouseIdxOcclPostFam = mouseIDPost(mean(imgOcclResPostMnFam(sub,:)) > respThreshold);
imgOcclResPostMnFamMn = [mean(imgOcclResPostMnFam(fh,:)); mean(imgOcclResPostMnFam(sh,:))];

imgOcclResPostMnNov = squeeze(mean(imgOcclResPostMn(novIdx,:,:)));
if smoothed, imgOcclResPostMnNov = smoothG(imgOcclResPostMnNov, smoothVal); end
imgOcclResPostMnNov = imgOcclResPostMnNov(:, mean(imgOcclResPostMnNov(sub,:)) > respThreshold);
mouseIdxOcclPostNov = mouseIDPost(mean(imgOcclResPostMnNov(sub,:)) > respThreshold);
imgOcclResPostMnNovMn = [mean(imgOcclResPostMnNov(fh,:)); mean(imgOcclResPostMnNov(sh,:))];

% --- Full Task (Fam only) ---
imgFullResTaskMn = squeeze(mean(imgFullResTask(vecAxTaskSt,:,:,:))) - squeeze(mean(imgFullResTask(vecAxTaskSp,:,:,:)));
imgFullResTaskMnFam = squeeze(mean(imgFullResTaskMn, 1));
if smoothed, imgFullResTaskMnFam = smoothG(imgFullResTaskMnFam, smoothVal); end
imgFullResTaskMnFam = imgFullResTaskMnFam(:, mean(imgFullResTaskMnFam(sub,:)) > respThreshold);
mouseIdxFullTaskFam = mouseIDTask(mean(imgFullResTaskMnFam(sub,:)) > respThreshold);
imgFullResTaskMnFamMn = [mean(imgFullResTaskMnFam(fh,:)); mean(imgFullResTaskMnFam(sh,:))];

% --- Occluded Task (Fam only) ---
imgOcclResTaskMn = squeeze(mean(imgOcclResTask(vecAxTaskSt,:,:,:))) - squeeze(mean(imgOcclResTask(vecAxTaskSp,:,:,:)));
imgOcclResTaskMnFam = squeeze(mean(imgOcclResTaskMn, 1));
if smoothed, imgOcclResTaskMnFam = smoothG(imgOcclResTaskMnFam, smoothVal); end
imgOcclResTaskMnFam = imgOcclResTaskMnFam(:, mean(imgOcclResTaskMnFam(sub,:)) > respThreshold);
mouseIdxOcclTaskFam = mouseIDTask(mean(imgOcclResTaskMnFam(sub,:)) > respThreshold);
imgOcclResTaskMnFamMn = [mean(imgOcclResTaskMnFam(fh,:)); mean(imgOcclResTaskMnFam(sh,:))];

% -------------------- (1) Response-over-trials “slope” plots --------------------
figure('Position',[293 309 1164 571])

% Full visibility: Pre/Post/Task
s(1) = subplot(2,4,1);
shadedErrorBar(1:20, mean(imgFullResPreMnFam,2),  std(imgFullResPreMnFam,0,2)/sqrt(size(imgFullResPreMnFam,2)), 'lineProps','b'); hold on
shadedErrorBar(1:20, mean(imgFullResPreMnNov,2),  std(imgFullResPreMnNov,0,2)/sqrt(size(imgFullResPreMnNov,2)), 'lineProps','r');
ylabel('Response'), xlabel('Trials'), title('NO Pre'), 

s(2) = subplot(2,4,2);
shadedErrorBar(1:20, mean(imgFullResPostMnFam,2), std(imgFullResPostMnFam,0,2)/sqrt(size(imgFullResPostMnFam,2)), 'lineProps','b'); hold on
shadedErrorBar(1:20, mean(imgFullResPostMnNov,2), std(imgFullResPostMnNov,0,2)/sqrt(size(imgFullResPostMnNov,2)), 'lineProps','r');
ylabel('Response'), xlabel('Trials'), title('NO Post'), 

s(3) = subplot(2,4,3);
shadedErrorBar(1:20, mean(imgFullResTaskMnFam,2), std(imgFullResTaskMnFam,0,2)/sqrt(size(imgFullResTaskMnFam,2)), 'lineProps','b');
ylabel('Response'), xlabel('Trials'), title('NO Task'), 

% Occluded: Pre/Post/Task
s(4) = subplot(2,4,5);
shadedErrorBar(1:20, mean(imgOcclResPreMnFam,2),  std(imgOcclResPreMnFam,0,2)/sqrt(size(imgOcclResPreMnFam,2)), 'lineProps','b'); hold on
shadedErrorBar(1:20, mean(imgOcclResPreMnNov,2),  std(imgOcclResPreMnNov,0,2)/sqrt(size(imgOcclResPreMnNov,2)), 'lineProps','r');
ylabel('Response'), xlabel('Trials'), title('O Pre'), 

s(5) = subplot(2,4,6);
shadedErrorBar(1:20, mean(imgOcclResPostMnFam,2), std(imgOcclResPostMnFam,0,2)/sqrt(size(imgOcclResPostMnFam,2)), 'lineProps','b'); hold on
shadedErrorBar(1:20, mean(imgOcclResPostMnNov,2), std(imgOcclResPostMnNov,0,2)/sqrt(size(imgOcclResPostMnNov,2)), 'lineProps','r');
ylabel('Response'), xlabel('Trials'), title('O Post'), 

s(6) = subplot(2,4,7);
shadedErrorBar(1:20, mean(imgOcclResTaskMnFam,2), std(imgOcclResTaskMnFam,0,2)/sqrt(size(imgOcclResTaskMnFam,2)), 'lineProps','b');
ylabel('Response'), xlabel('Trials'), title('O Task'), 

for j = 1:6
    s(j).YLim = [0 1.5]; % adjust if needed
end

% -------------------- (2) fh vs sh scatter panels (means ± SEM) --------------------
% Build condition lists for plotting
fullConds = {
    'FullPreFam',  imgFullResPreMnFamMn;
    'FullPreNov',  imgFullResPreMnNovMn;
    'FullPostFam', imgFullResPostMnFamMn;
    'FullPostNov', imgFullResPostMnNovMn;
    'FullTaskFam', imgFullResTaskMnFamMn;
};

occlConds = {
    'OcclPreFam',  imgOcclResPreMnFamMn;
    'OcclPreNov',  imgOcclResPreMnNovMn;
    'OcclPostFam', imgOcclResPostMnFamMn;
    'OcclPostNov', imgOcclResPostMnNovMn;
    'OcclTaskFam', imgOcclResTaskMnFamMn;
};

clear axFHSH

% --- fh vs sh scatter panels (means ± SEM) ---
axFHSH = gobjects(1,2);

axFHSH(1) = subplot(2,4,4); hold on
plot_fh_sh_block(fullConds, 'b'); title('NO (fh vs sh)'), ylabel('Response'), 
xlim([0 15]); ylim(axFHSH(1), [0 1.2]);

axFHSH(2) = subplot(2,4,8); hold on
plot_fh_sh_block(occlConds, 'b'); title('O (fh vs sh)'), ylabel('Response'), 
xlim([0 15]); ylim(axFHSH(2), [0 1.2]);

% Update the annotation calls to use the new handles:
annotate_fh_sh_stars(axFHSH(1), fullConds(:,1), adaptStatsTTest_fhsh);
annotate_fh_sh_stars(axFHSH(2), occlConds(:,1), adaptStatsTTest_fhsh);

% -------------------- (3) Paired t-tests on fh vs sh (neurons) --------------------
condData = {
    'FullPreFam',   imgFullResPreMnFamMn;
    'FullPreNov',   imgFullResPreMnNovMn;
    'FullPostFam',  imgFullResPostMnFamMn;
    'FullPostNov',  imgFullResPostMnNovMn;
    'OcclPreFam',   imgOcclResPreMnFamMn;
    'OcclPreNov',   imgOcclResPreMnNovMn;
    'OcclPostFam',  imgOcclResPostMnFamMn;
    'OcclPostNov',  imgOcclResPostMnNovMn;
    'FullTaskFam',  imgFullResTaskMnFamMn;
    'OcclTaskFam',  imgOcclResTaskMnFamMn;
};

resultsTTest_fhsh = cell(size(condData,1), 5);
fprintf('--- Paired t-test on First-Half vs Second-Half Responses ---\n');

for i = 1:size(condData,1)
    condName = condData{i,1};
    respMat  = condData{i,2}; % [2 x neurons]

    if isempty(respMat) || size(respMat,2) < 2
        warning('%s: Skipping (not enough data)', condName);
        resultsTTest_fhsh(i,:) = {condName, NaN, NaN, NaN, NaN};
        continue
    end

    fhv = respMat(1,:);
    shv = respMat(2,:);
    [~, p, ~, stats] = ttest(fhv, shv);  % paired across neurons
    delta = mean(shv - fhv);             % average change (sh - fh)
    resultsTTest_fhsh(i,:) = {condName, delta, stats.tstat, stats.df, p};

    fprintf('[t-test] %s: Δ = %.4f, t(%d) = %.2f, p = %.4f\n', ...
        condName, delta, stats.df, stats.tstat, p);
end

adaptStatsTTest_fhsh = cell2table(resultsTTest_fhsh, ...
    'VariableNames', {'Condition','Delta','tStat','DF','pValue'});

% ---- Optional: annotate stars on the fh/sh panels using these p-values ----
annotate_fh_sh_stars(axFHSH(1), fullConds(:,1), adaptStatsTTest_fhsh);
annotate_fh_sh_stars(axFHSH(2), occlConds(:,1), adaptStatsTTest_fhsh);

if save_fig
    func_save_fig('L23_resOverTrials'); func_save_fig('L5_resOverTrials');
end




%% selectivity/sparsity and timing plotting
save_fig = false;

% ix = novIdx;
ix = famIdx;

% load in active data
if nfiles == 6
    load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\postActiveTraceDataForCorrsL23zscored.mat')
elseif nfiles == 5
    load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\postActiveTraceDataForCorrsL5zscoredv2.mat')
end
vecAxTaskSt = vecAxTask>0.2 & vecAxTask<1;
vecAxTaskSp = vecAxTask<0;

imgFullPre = squeeze(mean(imgFullResMnPopPre(vecAxSt,ix,:)))-squeeze(mean(imgFullResMnPopPre(vecAxSp,ix,:)));
imgFullPost = squeeze(mean(imgFullResMnPopPost(vecAxSt,ix,:)))-squeeze(mean(imgFullResMnPopPost(vecAxSp,ix,:)));
imgFullTask = squeeze(mean(imgFullResMnPop(vecAxTaskSt,:,:)))-squeeze(mean(imgFullResMnPop(vecAxTaskSp,:,:)));
imgOcclPre = squeeze(mean(imgOcclResMnPopPre(vecAxSt,ix,:)))-squeeze(mean(imgOcclResMnPopPre(vecAxSp,ix,:)));
imgOcclPost = squeeze(mean(imgOcclResMnPopPost(vecAxSt,ix,:)))-squeeze(mean(imgOcclResMnPopPost(vecAxSp,ix,:)));
imgOcclTask = squeeze(mean(imgOcclResMnPop(vecAxTaskSt,:,:)))-squeeze(mean(imgOcclResMnPop(vecAxTaskSp,:,:)));

% lifetime sparseness
sparsenessFullPre = calculateLifetimeSparseness(imgFullPre')';
sparsenessFullPost = calculateLifetimeSparseness(imgFullPost')';
sparsenessFullTask = calculateLifetimeSparseness(imgFullTask')';

sparsenessOcclPre = calculateLifetimeSparseness(imgOcclPre')';
sparsenessOcclPost = calculateLifetimeSparseness(imgOcclPost')';
sparsenessOcclTask = calculateLifetimeSparseness(imgOcclTask')';

mouseIDPre = [];
mouseIDPost = [];
mouseIDTask = [];
for i = 1:nfiles
    % prepare some data for linear mixed model effect
    mouseIDPre = [mouseIDPre zeros(1,length(datastructPre(i).scatFull))+i];
    mouseIDPost = [mouseIDPost zeros(1,length(datastructPost(i).scatFull))+i];
    mouseIDTask = [mouseIDTask zeros(1,length(datastructActiveRes(i).scatFull))+i];
end

% full fam LMEM
data = cat(2, sparsenessFullPre,sparsenessFullPost, sparsenessFullTask)';
mouseID = categorical(cat(2, mouseIDPre,mouseIDPost,mouseIDTask))';
condition = categorical(cat(1, ones(length(mouseIDPre),1),ones(length(mouseIDPost),1)+1,ones(length(mouseIDTask),1)+2));
clear statTbl, statTbl = table(data, mouseID, condition);
lmeSpars = fitlme(statTbl, 'data ~ condition + (1|mouseID)', 'CheckHessian', 1,'FitMethod', 'REML', 'StartMethod', 'random');
statsSpars = anova(lmeSpars,'DFMethod','Satterthwaite');
statTblSpars = makeStatTbl(lmeSpars);

% plot passive and active data together and timing differences zoom
% latency estimation is done on a part of the whole trace only, as it's
% about the onset. Taking longer sections hampers the fitting. We fit two
% gaussian curves according to Papale et al. 2023 and Kirchberger et al 2023.
% We also normalize between 0 and 1 and then subtract the baseline. The curve
% should start around 0 for the fitting.

loadData = 1;
sqrThres = 0.5; % R2 threshold to only include well fitted neurons

if ~loadData % if we did the analysis already we can load the data
    latFullTask = zeros(size(imgFullResBsl,2),1);
    latOcclTask = zeros(size(imgOcclResBsl,2),1);
    sqrFullTask = zeros(size(imgFullResBsl,2),1);
    sqrOcclTask = zeros(size(imgOcclResBsl,2),1);
    ax = vecAxTask>-0.4&vecAxTask<0.8; % same as cnn
    newAx = vecAxTask(ax);

    for i = 1:size(imgFullResBsl,2)
        trace = normalize(imgFullResBsl(ax,i),'range');
        x = trace-mean(trace(newAx<0));
        [latFullTask(i), ~, sqrFullTask(i)] = calcLatencyFitInterp(x,newAx,1,0);
        trace = normalize(imgOcclResBsl(ax,i),'range');
        x = trace-mean(trace(newAx<0));
        [latOcclTask(i), ~, sqrOcclTask(i)] = calcLatencyFitInterp(x,newAx,1,0);
    end
else
    if nfiles == 6
        load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\timingOnsetDataL23.mat')
    else
        load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\timingOnsetDataL5.mat')
    end
end

% only include well fitted neurons and neurons with a positive response
posResFull = mean(imgFullResBsl(vecAxTaskSt,:))>0; % positive response
posResOccl = mean(imgOcclResBsl(vecAxTaskSt,:))>0; % positive response
inclFull = sqrFullTask>sqrThres&posResFull'; % good fit (sqr) and positive response
inclOccl = sqrOcclTask>sqrThres&posResOccl'; % good fit (sqr) and positive response
latFullTaskIncl = latFullTask(inclFull)*1000;
latOcclTaskIncl = latOcclTask(inclOccl)*1000;

sz = 70;
figure('Position', [618   459   838   287])
subplot(1,4,1:2)
shadedErrorBar(vecAxTask,mean(imgFullResBsl,2)...
    ,std(imgFullResBsl,0,2)/sqrt(size(imgFullResBsl,2)), 'lineProps', 'b'); hold on
shadedErrorBar(vecAxTask,mean(imgOcclResBsl,2)...
    ,std(imgOcclResBsl,0,2)/sqrt(size(imgOcclResBsl,2)), 'lineProps', 'r');
ylim([-0.02 0.3])
ylabel('Response'), xlim([0 0.3])
xline(nanmean(latFullTaskIncl)/1000), xline(nanmean(latOcclTaskIncl)/1000)

subplot(1,4,3)
scatter([1 2],[nanmean(latFullTaskIncl) nanmean(latOcclTaskIncl)], sz, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2],[nanmean(latFullTaskIncl) nanmean(latOcclTaskIncl)], ...
    [calcSem(latFullTaskIncl) calcSem(latOcclTaskIncl)] ...
    ,[calcSem(latFullTaskIncl) calcSem(latOcclTaskIncl)]);
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 3]), ylabel('Onset (ms)'), xticks([1 2]), 
ylim([170 270])
xticklabels({'NO','O'}), 
subplot(1,4,4)
scatter([1 2 3],[mean(sparsenessFullPre) mean(sparsenessFullPost) mean(sparsenessFullTask)], sz, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2 3],[mean(sparsenessFullPre) mean(sparsenessFullPost) mean(sparsenessFullTask)], ...
    [calcSem(sparsenessFullPre) calcSem(sparsenessFullPost) calcSem(sparsenessFullTask)] ...
    ,[calcSem(sparsenessFullPre) calcSem(sparsenessFullPost) calcSem(sparsenessFullTask)]);
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 4]), ylim([0.4 0.65]), ylabel('Sparseness NO'), xticks([1 2 3])
xticklabels({'N','E', 'T'}), 

if save_fig
    func_save_fig('L23_timingAndSelectivity')
    func_save_fig('L5_timingAndSelectivity')
end

% LMEM
mouseIDPre = [];
mouseIDPost = [];
mouseIDTask = [];
for i = 1:nfiles
    % prepare some data for linear mixed model effect
    mouseIDPre = [mouseIDPre zeros(1,length(datastructPre(i).scatFull))+i];
    mouseIDPost = [mouseIDPost zeros(1,length(datastructPost(i).scatFull))+i];
    mouseIDTask = [mouseIDTask zeros(1,length(datastructActiveRes(i).scatFull))+i];
end

% latency LMEM
data = cat(1, latFullTaskIncl,latOcclTaskIncl);
mouseID = categorical(cat(2, mouseIDTask(inclFull),mouseIDTask(inclOccl)))';
condition = categorical(cat(1, ones(length(mouseIDTask(inclFull)),1),ones(length(mouseIDTask(inclOccl)),1)+1));
clear statTbl, statTbl = table(data, mouseID, condition);

% first try for L5 gives hessian error due to outliers
lmeLatency = fitlme(statTbl, 'data ~ condition + (1|mouseID)', 'CheckHessian', 1,'FitMethod', 'REML', 'StartMethod', 'random');

%     % first try gives hessian error due to outliers
%     lmeLatency = fitlme(statTbl, 'data ~ condition + (1|mouseID)', 'CheckHessian', 1,'FitMethod', 'REML', 'StartMethod', 'random');

% %     % first remove mouse nr 3, not enough neurons
% %     rmv = statTbl.mouseID=='3';
% %     statTbl(rmv,:) = [];

if nfiles == 5
    %  remove outliers for L5
    zsRes = zscore(lmeLatency.residuals);
    rmv = zsRes<-2.5|zsRes>2.5; % index to remove
    statTbl(rmv,:) = []; % remove outliers

    % retry
    lmeLatency = fitlme(statTbl, 'data ~ condition + (1|mouseID)', 'CheckHessian', 1,'FitMethod', 'REML', 'StartMethod', 'random');
end



% %% isolation index
% data = [scatFullFamPopPre; scatOcclFamPopPre]';
% % data = [scatFullFamPopPost; scatOcclFamPopPost]';
% data = [scatFullPop; scatOcclPop]';
% % data = [scatFullNovPopPost; scatOcclNovPopPost]';
% plotResults = true;
% num_permutations = 10000;
% stdMult = 1;
% % [p_value, permutation_matrix, real_count, isolation_index] = getSeparationIndex(data, num_permutations, stdMult, plotResults);
% [p_value, permutation_matrix, real_count, isolation_index] = getSeparationIndexMarg(data, num_permutations, stdMult, plotResults);
% 
% if save_fig
%     func_save_fig('L23_Pre_separationIndex')
%     func_save_fig('L23_Post_separationIndex')
%     func_save_fig('L23_Task_separationIndex')
%     func_save_fig('L5_Pre_ROI1_G2I28')
% end
% 
% %% isolation index but with edges based on spontaneous activity
% %%% THIS IS WRONG STILL %%%%
% 
% imgFullRes = [];
% imgOcclRes = [];
% for i = 1:nfiles
%     imgFullRes = cat(3, imgFullRes, datastructActiveRes(i).imgFullResMn);
%     imgOcclRes = cat(3, imgOcclRes, datastructActiveRes(i).imgOcclResMn);
% end
% 
% imgFullResTask = squeeze(mean(imgFullRes,2));
% imgOcclResTask = squeeze(mean(imgOcclRes,2));
% vecAxTaskSp = vecAxTask<0;
% 
% plotResults = 1;
% num_permutations = 10000;
% 
% % fam naive
% data = [scatFullFamPopPre; scatOcclFamPopPre]';
% dataSpont = [mean(imgFullResFamPre(vecAxSp,:)); mean(imgOcclResFamPre(vecAxSp,:))]';
% % edges = [max(dataSpont(:,1)); max(dataSpont(:,2))];
% edges = [prctile(dataSpont(:,1),0.95); prctile(dataSpont(:,2),0.95)];
% [pFamNaive, ~, ~, siFamNaive] = getSeparationIndexSpont(data, num_permutations, edges, plotResults); % separation index
% [crFamNaive, cpFamNaive] = corrcoef(scatFullFamPopPre, scatOcclFamPopPre); % correlation
% 
% % fam expert
% data = [scatFullFamPopPost; scatOcclFamPopPost]';
% dataSpont = [mean(imgFullResFamPost(vecAxSp,:)); mean(imgOcclResFamPost(vecAxSp,:))]';
% % edges = [max(dataSpont(:,1)); max(dataSpont(:,2))];
% edges = [prctile(dataSpont(:,1),[0 0.95]); prctile(dataSpont(:,2),[0 0.95])];
% [pFamExpert, ~, ~, siFamExpert] = getSeparationIndexSpont(data, num_permutations, edges, plotResults); % separation index
% [crFamExpert, cpFamExpert] = corrcoef(scatFullFamPopPost, scatOcclFamPopPost); % correlation
% 
% % fam task
% data = [scatFullPop; scatOcclPop]';
% dataSpont = [mean(imgFullResTask(vecAxTaskSp,:)); mean(imgOcclResTask(vecAxTaskSp,:))]';
% % edges = [max(dataSpont(:,1)); max(dataSpont(:,2))];
% edges = [prctile(dataSpont(:,1),0.95); prctile(dataSpont(:,2),0.95)];
% [pFamTask, ~, ~, siFamTask] = getSeparationIndexSpont(data, num_permutations, edges, plotResults); % separation index
% [crTask, cpTask] = corrcoef(scatFullPop, scatOcclPop); % correlation
% 
% % nov naive
% data = [scatFullNovPopPre; scatOcclNovPopPre]';
% dataSpont = [mean(imgFullResNovPre(vecAxSp,:)); mean(imgOcclResNovPre(vecAxSp,:))]';
% % edges = [max(dataSpont(:,1)); max(dataSpont(:,2))];
% edges = [prctile(dataSpont(:,1),0.95); prctile(dataSpont(:,2),0.95)];
% [pNovNaive, ~, ~, siNovNaive] = getSeparationIndexSpont(data, num_permutations, edges, plotResults); % separation index
% [crNovNaive, cpNovNaive] = corrcoef(scatFullNovPopPre, scatOcclNovPopPre); % correlation
% 
% % nov expert
% data = [scatFullNovPopPost; scatOcclNovPopPost]';
% dataSpont = [mean(imgFullResNovPost(vecAxSp,:)); mean(imgOcclResNovPost(vecAxSp,:))]';
% % edges = [max(dataSpont(:,1)); max(dataSpont(:,2))];
% edges = [prctile(dataSpont(:,1),0.95); prctile(dataSpont(:,2),0.95)];
% [pNovExpert, ~, ~, siNovExpert] = getSeparationIndexSpont(data, num_permutations, edges, plotResults); % separation index
% [crNovExpert, cpNovExpert] = corrcoef(scatFullNovPopPost, scatOcclNovPopPost); % correlation
% 
% % O fam vs O novel in expert
% data = [scatOcclFamPopPost; scatOcclNovPopPost]';
% dataSpont = [mean(imgOcclResFamPost(vecAxSp,:)); mean(imgOcclResNovPost(vecAxSp,:))]';
% edges = [max(dataSpont(:,1)); max(dataSpont(:,2))];
% [pFamONovOExpert, ~, ~, siFamONovOExpert] = getSeparationIndexSpont(data, num_permutations, edges, plotResults); % separation index
% [crFamONovOExpert, cpFamONovOExpert] = corrcoef(scatOcclFamPopPost, scatOcclNovPopPost); % correlation
% 
% % NO fam vs NO novel in expert
% data = [scatFullFamPopPost; scatFullNovPopPost]';
% dataSpont = [mean(imgFullResFamPost(vecAxSp,:)); mean(imgFullResNovPost(vecAxSp,:))]';
% % edges = [max(dataSpont(:,1)); max(dataSpont(:,2))];
% edges = [prctile(dataSpont(:,1),0.95); prctile(dataSpont(:,2),0.95)];
% [pFamNONovNOExpert, ~, ~, siFamNONovNOExpert] = getSeparationIndexSpont(data, num_permutations, edges, plotResults); % separation index
% [crFamNONovNOExpert, cpFamNONovNOExpert] = corrcoef(scatFullFamPopPost, scatFullNovPopPost); % correlation
% 
% % NO fam vs NO novel in expert
% data = [scatOcclFamPopPost; scatFullNovPopPost]';
% dataSpont = [mean(imgOcclResFamPost(vecAxSp,:)); mean(imgFullResNovPost(vecAxSp,:))]';
% % edges = [max(dataSpont(:,1)); max(dataSpont(:,2))];
% edges = [prctile(dataSpont(:,1),0.95); prctile(dataSpont(:,2),0.95)];
% [pFamONovNOExpert, ~, ~, siFamONovNOExpert] = getSeparationIndexSpont(data, num_permutations, edges, plotResults); % separation index
% [crFamONovNOExpert, cpFamONovNOExpert] = corrcoef(scatOcclFamPopPost, scatFullNovPopPost); % correlation
% 
% 
% 
% 
% % some other ways to define edges and stdMult as backup:
% % stdMult = 1;
% % edges = [quantile(dataSpont(:,1),0.95); quantile(dataSpont(:,2),0.95)];
% % edges = [quantile(data(:,1),0.9); quantile(data(:,2),0.9)];
% % edges = [0.5; 0.5];
% % [p_value, perm_emd, real_emd] = getSeparationIndexEmd(data, num_permutations, plotResults);

%% lifetime sparseness maar dan voor alleen responsive cellen
save_fig = false;
warning on

% ix = novIdx;
ix = famIdx;

% load in active data
if nfiles == 6
    load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\postActiveTraceDataForCorrsL23zscored.mat')
elseif nfiles == 5
    load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\postActiveTraceDataForCorrsL5zscoredv2.mat')
end
vecAxTaskSt = vecAxTask>0.2 & vecAxTask<1;
vecAxTaskSp = vecAxTask<0;

imgFullPre = squeeze(mean(imgFullResMnPopPre(vecAxSt,ix,:)))-squeeze(mean(imgFullResMnPopPre(vecAxSp,ix,:)));
imgFullPost = squeeze(mean(imgFullResMnPopPost(vecAxSt,ix,:)))-squeeze(mean(imgFullResMnPopPost(vecAxSp,ix,:)));
imgFullTask = squeeze(mean(imgFullResMnPop(vecAxTaskSt,:,:)))-squeeze(mean(imgFullResMnPop(vecAxTaskSp,:,:)));
imgOcclPre = squeeze(mean(imgOcclResMnPopPre(vecAxSt,ix,:)))-squeeze(mean(imgOcclResMnPopPre(vecAxSp,ix,:)));
imgOcclPost = squeeze(mean(imgOcclResMnPopPost(vecAxSt,ix,:)))-squeeze(mean(imgOcclResMnPopPost(vecAxSp,ix,:)));
imgOcclTask = squeeze(mean(imgOcclResMnPop(vecAxTaskSt,:,:)))-squeeze(mean(imgOcclResMnPop(vecAxTaskSp,:,:)));

% lifetime sparseness
sparsenessFullPre = calculateLifetimeSparseness(imgFullPre')';
sparsenessFullPost = calculateLifetimeSparseness(imgFullPost')';
sparsenessFullTask = calculateLifetimeSparseness(imgFullTask')';

sparsenessOcclPre = calculateLifetimeSparseness(imgOcclPre')';
sparsenessOcclPost = calculateLifetimeSparseness(imgOcclPost')';
sparsenessOcclTask = calculateLifetimeSparseness(imgOcclTask')';

threshold = 0.3; % used 0.3 for paper

% LMEM
mouseIDPre = [];
mouseIDPost = [];
mouseIDTask = [];
for i = 1:nfiles
    % prepare some data for linear mixed model effect
    mouseIDPre = [mouseIDPre zeros(1,length(datastructPre(i).scatFull))+i];
    mouseIDPost = [mouseIDPost zeros(1,length(datastructPost(i).scatFull))+i];
    mouseIDTask = [mouseIDTask zeros(1,length(datastructActiveRes(i).scatFull))+i];
end

mouseIDPreFull = mouseIDPre(scatFullFamPopPre>threshold);
mouseIDPostFull = mouseIDPost(scatFullFamPopPost>threshold);
mouseIDTaskFull = mouseIDTask(scatFullPop>threshold);
mouseIDPreOccl = mouseIDPre(scatOcclFamPopPre>threshold);
mouseIDPostOccl = mouseIDPost(scatOcclFamPopPost>threshold);
mouseIDTaskOccl = mouseIDTask(scatOcclPop>threshold);

sparsenessFullPre = sparsenessFullPre(scatFullFamPopPre>threshold);
sparsenessFullPost = sparsenessFullPost(scatFullFamPopPost>threshold);
sparsenessFullTask = sparsenessFullTask(scatFullPop>threshold);

sparsenessOcclPre = sparsenessOcclPre(scatOcclFamPopPre>threshold);
sparsenessOcclPost = sparsenessOcclPost(scatOcclFamPopPost>threshold);
sparsenessOcclTask = sparsenessOcclTask(scatOcclPop>threshold);

sz = 90;

figure('Position', [680   558   353   420])
subplot(1,2,1)
scatter([1 2 3],[mean(sparsenessFullPre) mean(sparsenessFullPost) mean(sparsenessFullTask)], sz, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2 3],[mean(sparsenessFullPre) mean(sparsenessFullPost) mean(sparsenessFullTask)], ...
    [calcSem(sparsenessFullPre) calcSem(sparsenessFullPost) calcSem(sparsenessFullTask)] ...
    ,[calcSem(sparsenessFullPre) calcSem(sparsenessFullPost) calcSem(sparsenessFullTask)]);
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 4]), ylim([0 0.65]), ylabel('Sparseness NO'), xticks([1 2 3])
xticklabels({'N','E', 'T'}), 

subplot(1,2,2)
scatter([1 2 3],[mean(sparsenessOcclPre) mean(sparsenessOcclPost) mean(sparsenessOcclTask)], sz, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2 3],[mean(sparsenessOcclPre) mean(sparsenessOcclPost) mean(sparsenessOcclTask)], ...
    [calcSem(sparsenessOcclPre) calcSem(sparsenessOcclPost) calcSem(sparsenessOcclTask)] ...
    ,[calcSem(sparsenessOcclPre) calcSem(sparsenessOcclPost) calcSem(sparsenessOcclTask)]);
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 4]), ylim([0 0.65]), ylabel('Sparseness O'), xticks([1 2 3])
xticklabels({'N','E', 'T'}), 

% full  LMEM
data = cat(2, sparsenessFullPre,sparsenessFullPost, sparsenessFullTask)';
mouseID = categorical(cat(2, mouseIDPreFull,mouseIDPostFull,mouseIDTaskFull))';
condition = categorical(cat(1, ones(length(mouseIDPreFull),1),ones(length(mouseIDPostFull),1)+1,ones(length(mouseIDTaskFull),1)+2));
clear statTbl, statTbl = table(data, mouseID, condition);
lmeSpars = fitlme(statTbl, 'data ~ condition + (1|mouseID)', 'CheckHessian', 1,'FitMethod', 'REML', 'StartMethod', 'random');
statsSparsFull = anova(lmeSpars,'DFMethod','Satterthwaite');
statTblSparsFull = makeStatTbl(lmeSpars);

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




%% isolation index but with std as the border

imgFullRes = [];
imgOcclRes = [];
for i = 1:nfiles
    imgFullRes = cat(3, imgFullRes, datastructActiveRes(i).imgFullResMn);
    imgOcclRes = cat(3, imgOcclRes, datastructActiveRes(i).imgOcclResMn);
end

imgFullResTask = squeeze(mean(imgFullRes,2));
imgOcclResTask = squeeze(mean(imgOcclRes,2));
vecAxTaskSp = vecAxTask<0;

plotResults = true;
num_permutations = 10000;
stdMult = 0.75;

% fam naive
data = [scatFullFamPopPre; scatOcclFamPopPre]';
[pFamNaive, ~, ~, siFamNaive] = getSeparationIndex(data, num_permutations, stdMult, plotResults); % separation index
[crFamNaive, cpFamNaive] = corr(data(:,1), data(:,2), 'Type' ,'Pearson'); % correlation
[crSpFamNaive, cpSpFamNaive] = corr(data(:,1), data(:,2), 'Type' ,'Spearman'); % correlation

% fam expert
data = [scatFullFamPopPost; scatOcclFamPopPost]';
[pFamExpert, ~, ~, siFamExpert] = getSeparationIndex(data, num_permutations, stdMult, plotResults); % separation index
[crFamExpert, cpFamExpert] = corr(data(:,1), data(:,2), 'Type' ,'Pearson'); % correlation
[crSpFamExpert, cpSpFamExpert] = corr(data(:,1), data(:,2), 'Type' ,'Spearman'); % correlation

% fam task
data = [scatFullPop; scatOcclPop]';
[pFamTask, ~, ~, siFamTask] = getSeparationIndex(data, num_permutations, stdMult, plotResults); % separation index
[crTask, cpTask] = corr(data(:,1), data(:,2), 'Type' ,'Pearson'); % correlation
[crSpFamTask, cpSpFamTask] = corr(data(:,1), data(:,2), 'Type' ,'Spearman'); % correlation

% nov naive
data = [scatFullNovPopPre; scatOcclNovPopPre]';
[pNovNaive, ~, ~, siNovNaive] = getSeparationIndex(data, num_permutations, stdMult, plotResults); % separation index
[crNovNaive, cpNovNaive] = corr(data(:,1), data(:,2), 'Type' ,'Pearson'); % correlation
[crSpNovNaive, cpSpNovNaive] = corr(data(:,1), data(:,2), 'Type' ,'Spearman'); % correlation

% nov expert
data = [scatFullNovPopPost; scatOcclNovPopPost]';
[pNovExpert, ~, ~, siNovExpert] = getSeparationIndex(data, num_permutations, stdMult, plotResults); % separation index
[crNovExpert, cpNovExpert] = corr(data(:,1), data(:,2), 'Type' ,'Pearson'); % correlation
[crSpNovExpert, cpSpNovExpert] = corr(data(:,1), data(:,2), 'Type' ,'Spearman'); % correlation

% O fam vs O novel in expert
data = [scatOcclFamPopPost; scatOcclNovPopPost]';
[pFamONovOExpert, ~, ~, siFamONovOExpert] = getSeparationIndex(data, num_permutations, stdMult, plotResults); % separation index
[crFamONovOExpert, cpFamONovOExpert] = corr(data(:,1), data(:,2), 'Type' ,'Pearson'); % correlation
[crSpFamONovOExpert, cpSpFamONovOExpert] = corr(data(:,1), data(:,2), 'Type' ,'Spearman'); % correlation

% NO fam vs NO novel in expert
data = [scatFullFamPopPost; scatFullNovPopPost]';
[pFamNONovNOExpert, ~, ~, siFamNONovNOExpert] = getSeparationIndex(data, num_permutations, stdMult, plotResults); % separation index
[crFamNONovNOExpert, cpFamNONovNOExpert] = corr(data(:,1), data(:,2), 'Type' ,'Pearson'); % correlation
[crSpNONovNOExpert, cpSpNONovNOExpert] = corr(data(:,1), data(:,2), 'Type' ,'Spearman'); % correlation

% O fam vs NO novel in expert
data = [scatOcclFamPopPost; scatFullNovPopPost]';
[pFamONovNOExpert, ~, ~, siFamONovNOExpert] = getSeparationIndex(data, num_permutations, stdMult, plotResults); % separation index
[crFamONovNOExpert, cpFamONovNOExpert] = corr(data(:,1), data(:,2), 'Type' ,'Pearson'); % correlation
[crSpFamONovNOExpert, cpSpFamONovNOExpert] = corr(data(:,1), data(:,2), 'Type' ,'Spearman'); % correlation

%% sum of stds in arms of the L shape

doPlotting = 1;

data = [scatFullFamPopPre; scatOcclFamPopPre]';
sumOfStds = calculateSumOfStds(data(:,1), data(:,2), doPlotting); disp(sumOfStds);

% fam expert
data = [scatFullFamPopPost; scatOcclFamPopPost]';
sumOfStds = calculateSumOfStds(data(:,1), data(:,2), doPlotting); disp(sumOfStds);

% fam task
data = [scatFullPop; scatOcclPop]';
sumOfStds = calculateSumOfStds(data(:,1), data(:,2), doPlotting); disp(sumOfStds);

% nov naive
data = [scatFullNovPopPre; scatOcclNovPopPre]';
sumOfStds = calculateSumOfStds(data(:,1), data(:,2), doPlotting); disp(sumOfStds);

% nov expert
data = [scatFullNovPopPost; scatOcclNovPopPost]';
sumOfStds = calculateSumOfStds(data(:,1), data(:,2), doPlotting); disp(sumOfStds);

% O fam vs O novel in expert
data = [scatOcclFamPopPost; scatOcclNovPopPost]';
sumOfStds = calculateSumOfStds(data(:,1), data(:,2), doPlotting); disp(sumOfStds);

% NO fam vs NO novel in expert
data = [scatFullFamPopPost; scatFullNovPopPost]';
sumOfStds = calculateSumOfStds(data(:,1), data(:,2), doPlotting); disp(sumOfStds);

% O fam vs NO novel in expert
data = [scatOcclFamPopPost; scatFullNovPopPost]';
sumOfStds = calculateSumOfStds(data(:,1), data(:,2), doPlotting); disp(sumOfStds);






%% tests
% data = [scatFullFamPopPre; scatOcclFamPopPre]';
data = [scatFullFamPopPost; scatOcclFamPopPost]';
% data = [scatFullPop; scatOcclPop]';

[real_counts, fractions] = plotQuadrantDistribution(data, stdMult, plotResults)



%% comparison to marginal distribution
numdiv = 4;
plotting = true;

[pval, chisquare, freq, dgf] = calcMargDistChi(scatFullFamPopPre,scatOcclFamPopPre, numdiv, plotting)
[pval, chisquare, freq, dgf] = calcMargDistChi(scatFullFamPopPost,scatOcclFamPopPost, numdiv, plotting)
[pval, chisquare, freq, dgf] = calcMargDistChi(scatFullPop,scatOcclPop, numdiv, plotting)

%%

data = [scatFullFamPopPost; scatOcclFamPopPost]';
[pFamExpert, ~, ~, siFamExpert] = getSeparationIndexChi(data, num_permutations, stdMult, plotResults); % separation index
[crFamExpert, cpFamExpert] = corrcoef(scatFullFamPopPost, scatOcclFamPopPost); % correlation

  %%  
%   x = randn(1,1000);
%   y = randn(1,1000);

  x = scatFullPop;
  y = scatOcclPop;
%   x = scatFullFamPopPre;
%   y = scatOcclFamPopPre;
%   x = scatFullFamPopPost;
%   y = scatOcclFamPopPost;

  numdiv = 4;
  plotting = true;

  [pval, chisquare, freq, dgf, pvalSeparation, chiSquareSeparation, observed, expected] = calcMargDistChiAndSep(x, y, numdiv, plotting)

  



%% Manier van Jorrit based on entire distribution

x = scatFullFamPopPre;
y = scatOcclFamPopPre;
x = scatFullFamPopPost;
y = scatOcclFamPopPost;
% x = scatFullPop;
% y = scatOcclPop;
x = scatFullNovPopPost;
y = scatOcclNovPopPost;

nPerms = 10000;
enablePlotting = 1;

[realMedian, permMedians, upperRightCount, upperRightCountPerm, pValue] = separationIndex2(x, y, nPerms, enablePlotting);
figure, histogram(permMedians), xline(realMedian)
pValue


%% fraction of neurons responding above a threshold
thresh = 0.5;

fracFullNaive = sum(scatFullFamPopPre>thresh)/length(scatOcclFamPopPre);
fracFullExpert = sum(scatFullFamPopPost>thresh)/length(scatFullFamPopPost);
fracFullTask = sum(scatFullPop>thresh)/length(scatFullPop);

fracOcclNaive = sum(scatOcclFamPopPre>thresh)/length(scatOcclFamPopPre);
fracOcclExpert = sum(scatOcclFamPopPost>thresh)/length(scatOcclFamPopPost);
fracOcclTask = sum(scatOcclPop>thresh)/length(scatOcclPop);

fracFullOcclNaive = sum(scatFullFamPopPre>thresh&scatOcclFamPopPre>thresh)/length(scatOcclFamPopPre);
fracFullOcclExpert = sum(scatFullFamPopPost>thresh&scatOcclFamPopPost>thresh)/length(scatOcclFamPopPost);
fracFullOcclTask = sum(scatFullPop>thresh&scatOcclPop>thresh)/length(scatOcclPop);

figure
bar([1 2 3 5 6 7 9 10 11], ...
    [fracFullNaive fracFullExpert fracFullTask ...
    fracOcclNaive fracOcclExpert fracOcclTask ...
    fracFullOcclNaive fracFullOcclExpert fracFullOcclTask])

[h,pChi2stat, chi2stat,df] = prop_test([sum(scatFullFamPopPre>thresh) sum(scatFullFamPopPost>thresh)] , [length(scatFullFamPopPre) length(scatFullFamPopPost)], false)
[h,pChi2stat, chi2stat,df] = prop_test([sum(scatFullPop>thresh) sum(scatFullFamPopPost>thresh)] , [length(scatFullPop) length(scatFullFamPopPost)], false)
[h,pChi2stat, chi2stat,df] = prop_test([sum(scatFullFamPopPre>thresh) sum(scatFullPop>thresh)] , [length(scatFullFamPopPre) length(scatFullPop)], false)

[h,pChi2stat, chi2stat,df] = prop_test([sum(scatOcclFamPopPre>thresh) sum(scatOcclFamPopPost>thresh)] , [length(scatOcclFamPopPre) length(scatOcclFamPopPost)], false)
[h,pChi2stat, chi2stat,df] = prop_test([sum(scatOcclPop>thresh) sum(scatOcclFamPopPost>thresh)] , [length(scatOcclPop) length(scatOcclFamPopPost)], false)
[h,pChi2stat, chi2stat,df] = prop_test([sum(scatOcclFamPopPre>thresh) sum(scatOcclPop>thresh)] , [length(scatOcclFamPopPre) length(scatOcclPop)], false)

[h,pChi2stat, chi2stat,df] = prop_test([sum(scatFullFamPopPre>thresh&scatOcclFamPopPre>thresh) sum(scatFullFamPopPost>thresh&scatOcclFamPopPost>thresh)], ...
    [length(scatOcclFamPopPre) length(scatOcclFamPopPost)], false)
[h,pChi2stat, chi2stat,df] = prop_test([sum(scatFullFamPopPre>thresh&scatOcclFamPopPre>thresh) sum(scatFullPop>thresh&scatOcclPop>thresh)], ...
    [length(scatFullFamPopPre) length(scatFullPop)], false)
[h,pChi2stat, chi2stat,df] = prop_test([sum(scatFullFamPopPost>thresh&scatOcclFamPopPost>thresh) sum(scatFullPop>thresh&scatOcclPop>thresh)], ...
    [length(scatFullFamPopPost) length(scatFullPop)], false)

% per mouse:
fracFullNaive = zeros(nfiles,1);
fracFullExpert = zeros(nfiles,1);
fracFullTask = zeros(nfiles,1);

fracOcclNaive = zeros(nfiles,1);
fracOcclExpert = zeros(nfiles,1);
fracOcclTask = zeros(nfiles,1);

fracFullOcclNaive = zeros(nfiles,1);
fracFullOcclExpert = zeros(nfiles,1);
fracFullOcclTask = zeros(nfiles,1);

for i = 1:nfiles
    full = datastructPre(i).scatFull;
    occl = datastructPre(i).scatOccl;
    fracFullNaive(i) = sum(full>thresh)/length(full);
    fracOcclNaive(i) = sum(occl>thresh)/length(full);
    fracFullOcclNaive(i) = sum(full>thresh & occl>thresh)/length(full);

    full = datastructPost(i).scatFull;
    occl = datastructPost(i).scatOccl;
    fracFullExpert(i) = sum(full>thresh)/length(full);
    fracOcclExpert(i) = sum(occl>thresh)/length(full);
    fracFullOcclExpert(i) = sum(full>thresh & occl>thresh)/length(full);
    
    full = datastructActiveRes(i).scatFull;
    occl = datastructActiveRes(i).scatOccl;
    fracFullTask(i) = sum(full>thresh)/length(full);
    fracOcclTask(i) = sum(occl>thresh)/length(full);
    fracFullOcclTask(i) = sum(full>thresh & occl>thresh)/length(full);
end

figure, 
subplot(1,2,1)
bar([mean(fracFullNaive) mean(fracFullExpert) mean(fracFullTask)]), hold on
plot([fracFullNaive fracFullExpert fracFullTask]')
subplot(1,2,2)
bar([mean(fracOcclNaive) mean(fracOcclExpert) mean(fracOcclTask)]), hold on
plot([fracOcclNaive fracOcclExpert fracOcclTask]')


figure, plot([fracFullOcclNaive fracFullOcclExpert fracFullOcclTask]')


%% single cell responses naive

fullData = cat(4, datastructPre(:).imgFullRes);
occlData = cat(4, datastructPre(:).imgOcclRes);

fullData = fullData(:,famIdx,:,:);
occlData = occlData(:,famIdx,:,:);

nCells = 8; % nr of cells per plot
nPlots = 20; % how many plots in total?

for k = 1:nPlots % run the figure plot 10 times
% for k = 1 % run the figure plot 1 time
ix = randsample(1:size(fullData,4),nCells);

% ix = [300 236 639 628 505 542 120 139]; % L2/3 examples 1
% ix = [ 427 441 295 321 23 788]; % L2/3 examples 2

% ix = [391 277 113 86 184 385 96 256 ]; % L5 examples 1
% ix = [ 315 217 396]; % L5 examples 2

n = 0;

figure('Position', [336   104   543   809])
for i = 1:length(ix)
    clear s
    for j = 1:4
        traceFull = squeeze(fullData(:,j,:,ix(i))-mean(fullData(vecAxSp,j,:,ix(i))));
        traceOccl = squeeze(occlData(:,j,:,ix(i))-mean(occlData(vecAxSp,j,:,ix(i))));
        s(j) = subplot(nCells,4,j+n*4);
        shadedErrorBar(vecAx,mean(traceFull,2)...
            ,std(traceFull,0,2)/sqrt(size(traceFull,2)), 'lineProps', 'k');
        shadedErrorBar(vecAx,mean(traceOccl,2)...
            ,std(traceOccl,0,2)/sqrt(size(traceOccl,2)), 'lineProps', 'r');
        box off
        axis off
        if j == 1
            title(sprintf('ROI %d', ix(i)))
            line([0 0],[0 1],'Color','k');
            line([0 1],[0 0],'Color','k');
        end
        if i == 1
            ylims = ylim;
            patchHandle = patch([0 1 1 0], [0 0 1 1], 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
            % Use uistack to move the patch to the bottom of the stack
            uistack(patchHandle, 'bottom');
        end
    end
    mn = min([s(:).YLim]);
    mx = max([s(:).YLim]);
    for j = 1:length(s)
        s(j).YLim = [mn mx]; s(j).XLim = [-1 3]; s(j).XTick = -1:1:3;
    end

    n = n+1;
end
% pause
if nfiles == 6
    func_save_fig(sprintf('L23_naive_batch-%d', k+40))
%     func_save_fig(sprintf('L23_naive_batch-%d', 2))
else
%     func_save_fig(sprintf('L5_naive_batch-%d', k))
    func_save_fig(sprintf('L5_naive_batch-%d', 2))
end

end

%% single cell responses expert

fullData = cat(4, datastructPost(:).imgFullRes);
occlData = cat(4, datastructPost(:).imgOcclRes);

fullData = fullData(:,famIdx,:,:);
occlData = occlData(:,famIdx,:,:);

nCells = 8; % nr of cells per plot
nPlots = 20; % how many plots in total?

% for k = 1:nPlots % run the figure plot 10 times
for k = 1 % run the figure plot 1 time

% ix = randsample(1:size(fullData,4),nCells);

% ix = [480 375 98 688 248 56 102 337]; % L2/3 examples 1
% ix = [261 707 844 721]; % L2/3 examples 2

% ix = [59 287 16 8 134 220 35 125]; % L5 examples 1
ix = [30 223 17]; % L5 examples 2

n = 0;

figure('Position', [336   104   543   809])
for i = 1:length(ix)
    clear s
    for j = 1:4
        traceFull = squeeze(fullData(:,j,:,ix(i))-mean(fullData(vecAxSp,j,:,ix(i))));
        traceOccl = squeeze(occlData(:,j,:,ix(i))-mean(occlData(vecAxSp,j,:,ix(i))));
        s(j) = subplot(nCells,4,j+n*4);
        shadedErrorBar(vecAx,mean(traceFull,2)...
            ,std(traceFull,0,2)/sqrt(size(traceFull,2)), 'lineProps', 'k');
        shadedErrorBar(vecAx,mean(traceOccl,2)...
            ,std(traceOccl,0,2)/sqrt(size(traceOccl,2)), 'lineProps', 'r');

        box off
        axis off
        if j == 1
            title(sprintf('ROI %d', ix(i)))
            line([0 0],[0 1],'Color','k');
            line([0 1],[0 0],'Color','k');
        end
        if i == 1
            ylims = ylim;
            patchHandle = patch([0 1 1 0], [0 0 1 1], 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
            % Use uistack to move the patch to the bottom of the stack
            uistack(patchHandle, 'bottom');
        end
    end
    mn = min([s(:).YLim]);
    mx = max([s(:).YLim]);
    for j = 1:length(s)
        s(j).YLim = [mn mx]; s(j).XLim = [-1 3]; s(j).XTick = -1:1:3;
    end

    n = n+1;
end
% pause
if nfiles == 6
    func_save_fig(sprintf('L23_expert_batch-%d', k))
%     func_save_fig(sprintf('L23_expert_batch-%d', 2))
else
%     func_save_fig(sprintf('L5_expert_batch-%d', k))
    func_save_fig(sprintf('L5_expert_batch-%d', 2))
end
end

%% single cell responses task
if nfiles == 6
    load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\postActiveTraceDataForCorrsL23zscored.mat')
elseif nfiles == 5
    load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\postActiveTraceDataForCorrsL5zscoredv2.mat')
end

vecAxTaskSt = vecAxTask>0.2 & vecAxTask<1;
vecAxTaskSp = vecAxTask<0;

fullData = imgFullRes;
occlData = imgOcclRes;

nCells = 8; % nr of cells per plot
nPlots = 20; % how many plots in total?

% for k = 1:nPlots % run the figure plot 10 times
for k = 1 % run the figure plot 1 time
% ix = randsample(1:size(fullData,4),nCells);

% ix = [260 143 94 298 24 9 100 232 ]; % L2/3 examples 1
% ix = [204 85 177 130 120 123 247 32]; % L2/3 examples 2

% ix = [347 36 122 174 105 342 94 54]; % L5 examples 1
ix = [90 186 250 142 338 25 361 ]; % L5 examples 2

n = 0;

figure('Position', [336   104   543   809])
for i = 1:length(ix)
    clear s
    for j = 1:4
        traceFull = squeeze(fullData(:,j,:,ix(i))-mean(fullData(vecAxTaskSp,j,:,ix(i))));
        traceOccl = squeeze(occlData(:,j,:,ix(i))-mean(occlData(vecAxTaskSp,j,:,ix(i))));
        s(j) = subplot(nCells,4,j+n*4);
        shadedErrorBar(vecAxTask,mean(traceFull,2)...
            ,std(traceFull,0,2)/sqrt(size(traceFull,2)), 'lineProps', 'k');
        shadedErrorBar(vecAxTask,mean(traceOccl,2)...
            ,std(traceOccl,0,2)/sqrt(size(traceOccl,2)), 'lineProps', 'r');
        box off
        axis off
        if j == 1
            title(sprintf('ROI %d', ix(i)))
            line([0 0],[0 1],'Color','k');
            line([0 1],[0 0],'Color','k');
        end
        if i == 1
            ylims = ylim;
            patchHandle = patch([0 2 2 0], [0 0 1 1], 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
            % Use uistack to move the patch to the bottom of the stack
            uistack(patchHandle, 'bottom');
        end
    end
    mn = min([s(:).YLim]);
    mx = max([s(:).YLim]);
    for j = 1:length(s)
        s(j).YLim = [mn mx]; s(j).XLim = [-1 3]; s(j).XTick = -1:1:3;
    end

    n = n+1;
end
% pause
if nfiles == 6
%     func_save_fig(sprintf('L23_task_batch-%d', k+20))
    func_save_fig(sprintf('L23_task_batch-%d', k))
%     func_save_fig(sprintf('L23_task_batch-%d', 2))
else
%     func_save_fig(sprintf('L5_task_batch-%d', k))
    func_save_fig(sprintf('L5_task_batch-%d', 2))
end

end


%% single cell responses task
cd('\\vs01\mvp\Shared\Koen\FIGURES\Muckli\Manuscript\v5\Figure_1_setup')
imagefiles= dir('*.bmp');
nrOfImgs = length(imagefiles); % Number of files found

vecAxTaskSt = vecAxTask>0.2 & vecAxTask<1;
vecAxTaskSp = vecAxTask<0;

images = [];
for i=1:nrOfImgs
    %     disp(i)
    currentfilename = imagefiles(i).name;
    currentimage = imread(currentfilename);
    images{i} = currentimage;
end

if nfiles == 6
    load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\postActiveTraceDataForCorrsL23zscored.mat')
elseif nfiles == 5
    load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\postActiveTraceDataForCorrsL5zscoredv2.mat')
end

imagesOccl = images;
for i = 1:length(images)
    temp = images{i};
    temp(1:end/2,1:end/2)=128;
    imagesOccl{i} = temp;
end

figure('Position', [336         134        1112         461])
subplot(3,4,1), imagesc(images{1}), colormap gray, axis off
subplot(3,4,2), imagesc(images{2}), colormap gray, axis off
subplot(3,4,3), imagesc(images{4}), colormap gray, axis off
subplot(3,4,4), imagesc(images{5}), colormap gray, axis off
subplot(3,4,5), imagesc(imagesOccl{1}), hold on, colormap gray, axis off
subplot(3,4,6), imagesc(imagesOccl{2}), colormap gray, axis off
subplot(3,4,7), imagesc(imagesOccl{4}), colormap gray, axis off
subplot(3,4,8), imagesc(imagesOccl{5}), colormap gray, axis off
    
    for i = 1:size(imgFullResMnPop,3)
        delete(subplot(3,4,9:12))
        for j = 1:4
            traceFull = squeeze(imgFullResMnPop(:,j,i)-mean(imgFullResMnPop(vecAxSp,j,i)));
            traceOccl = squeeze(imgOcclResMnPop(:,j,i)-mean(imgOcclResMnPop(vecAxSp,j,i)));
            s(j) = subplot(3,4,j+8);
            shadedErrorBar(vecAxTask,mean(traceFull,2)...
                ,std(traceFull,0,2)/sqrt(size(traceFull,2)), 'lineProps', 'k');
            shadedErrorBar(vecAxTask,mean(traceOccl,2)...
                ,std(traceOccl,0,2)/sqrt(size(traceOccl,2)), 'lineProps', 'r');
            if j ~= 1
                set(gca,'Visible','off')
            else
                xlabel('Time (s)')
                ylabel('Response')
                title(sprintf('ROI %d', i))
            end
        end
        mn = min([s(:).YLim]);
        mx = max([s(:).YLim]);
        for j = 1:length(s)
            s(j).YLim = [mn mx]; s(j).XLim = [-1 3]; s(j).XTick = -1:1:3;
        end

        
        pause
    end

%% average across neurons per image
vecAxTaskSt = vecAxTask>0.2 & vecAxTask<1;
vecAxTaskSp = vecAxTask<0;

figure('Position', [363         520        1112         169])
clear s 
for i = 1:4 
    s(i) = subplot(1,4,i); 
    traceFull = squeeze(imgFullResMnPop(:,i,:)-mean(imgFullResMnPop(vecAxTaskSp,i,:))); hold on
    traceOccl = squeeze(imgOcclResMnPop(:,i,:)-mean(imgOcclResMnPop(vecAxTaskSp,i,:)));
            
    shadedErrorBar(vecAxTask,mean(traceFull,2)...
        ,std(traceFull,0,2)/sqrt(size(traceFull,2)), 'lineProps', 'k');
    shadedErrorBar(vecAxTask,mean(traceOccl,2)...
        ,std(traceOccl,0,2)/sqrt(size(traceOccl,2)), 'lineProps', 'r');

    xlabel('Time (s)')
    ylabel('Response')
    title('Average across all neurons')
end
mn = min([s(:).YLim]);
mx = max([s(:).YLim]);
for j = 1:length(s)
    s(j).YLim = [mn mx]; s(j).XLim = [-1 3]; s(j).XTick = -1:1:3;
end

%% Scatter plots per image
cd('\\vs01\mvp\Shared\Koen\FIGURES\Muckli\Manuscript\v5\Figure_1_setup')
imagefiles= dir('*.bmp');
nrOfImgs = length(imagefiles); % Number of files found

vecAxTaskSt = vecAxTask>0.2 & vecAxTask<1;
vecAxTaskSp = vecAxTask<0;

scatFullResMnPopImg = squeeze(mean(imgFullResMnPop(vecAxTaskSt,:,:))-mean(imgFullResMnPop(vecAxTaskSp,:,:)));
scatOcclResMnPopImg = squeeze(mean(imgOcclResMnPop(vecAxTaskSt,:,:))-mean(imgOcclResMnPop(vecAxTaskSp,:,:)));
% scatFullResMnPopImg = squeeze(mean(imgFullResMnPopPost(vecAxTaskSt,famIdx,:))-mean(imgFullResMnPopPost(vecAxTaskSp,famIdx,:)));
% scatOcclResMnPopImg = squeeze(mean(imgOcclResMnPopPost(vecAxTaskSt,famIdx,:))-mean(imgOcclResMnPopPost(vecAxTaskSp,famIdx,:)));

% remove non-responsive neurons?
ixFull = sum(scatFullResMnPopImg>0.5)>0;
ixOccl = sum(scatOcclResMnPopImg>0.5)>0;
scatFullResMnPopImg = scatFullResMnPopImg(:,ixFull|ixOccl);
scatOcclResMnPopImg = scatOcclResMnPopImg(:,ixFull|ixOccl);

images = [];
for i=1:nrOfImgs
    %     disp(i)
    currentfilename = imagefiles(i).name;
    currentimage = imread(currentfilename);
    images{i} = currentimage;
end

if nfiles == 6
    load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\postActiveTraceDataForCorrsL23zscored.mat')
elseif nfiles == 5
    load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\postActiveTraceDataForCorrsL5zscoredv2.mat')
end

imagesOccl = images;
for i = 1:length(images)
    temp = images{i};
    temp(1:end/2,1:end/2)=128;
    imagesOccl{i} = temp;
end

figure('Position', [145         323        1329         605])
subplot(3,4,1), imagesc(images{1}), colormap gray, axis off
subplot(3,4,2), imagesc(images{2}), colormap gray, axis off
subplot(3,4,3), imagesc(images{4}), colormap gray, axis off
subplot(3,4,4), imagesc(images{5}), colormap gray, axis off
subplot(3,4,5), imagesc(imagesOccl{1}), hold on, colormap gray, axis off
subplot(3,4,6), imagesc(imagesOccl{2}), colormap gray, axis off
subplot(3,4,7), imagesc(imagesOccl{4}), colormap gray, axis off
subplot(3,4,8), imagesc(imagesOccl{5}), colormap gray, axis off

sz = 8;
clear s
s(1) = subplot(3,4,9);
scatter(scatFullResMnPopImg(1,:), scatOcclResMnPopImg(1,:), sz, cPre, 'filled'); refline(1), xlabel('NO'), ylabel('O'), 
s(2) = subplot(3,4,10);
scatter(scatFullResMnPopImg(2,:), scatOcclResMnPopImg(2,:), sz, cPre, 'filled'); refline(1), xlabel('NO'), ylabel('O'), 
s(3) = subplot(3,4,11);
scatter(scatFullResMnPopImg(3,:), scatOcclResMnPopImg(3,:), sz, cPre, 'filled'); refline(1), xlabel('NO'), ylabel('O'), 
s(4) = subplot(3,4,12);
scatter(scatFullResMnPopImg(4,:), scatOcclResMnPopImg(4,:), sz, cPre, 'filled'); refline(1), xlabel('NO'), ylabel('O'), 

if nfiles == 6
    for j = 1:length(s)
        s(j).YLim = [-1 6]; s(j).YTick = -1:1:6; s(j).XLim = [-1 6]; s(j).XTick = -1:1:6;
    end
elseif nfiles == 5
    for j = 1:length(s)
%         s(j).YLim = [-1 3]; s(j).YTick = -1:0.5:3; s(j).XLim = [-1 3]; s(j).XTick = -1:0.5:3;
%         s(j).YLim = [-0.5 3]; s(j).YTick = -0.5:0.5:3; s(j).XLim = [-0.5 3]; s(j).XTick = -0.5:0.5:3;
        s(j).YLim = [-1 4]; s(j).YTick = -1:1:4; s(j).XLim = [-1 4]; s(j).XTick = -1:1:4;
    end
end

if save_fig
    func_save_fig('L23_scattersFamVsNov')
    func_save_fig('L5_scattersFamVsNov')
end


% selectivity index per image, then
siTask = zeros(4,length(scatFullResMnPopImg));
for j = 1:4
    siTask(j,:) = (scatFullResMnPopImg(j,:)-scatOcclResMnPopImg(j,:))./(scatFullResMnPopImg(j,:)+scatOcclResMnPopImg(j,:));
end
siTask(siTask<-1) = -1;
siTask(siTask>1) = 1;

%%
load('data.mat')
x1 = x;
y1 = y;

load('data2.mat')
x2 = x;
y2 = y;


adi1 = abs(x1 - y1) ./ (abs(x1) + abs(y1));
adi2 = abs(x2 - y2) ./ (abs(x2) + abs(y2));

if nfiles == 6
    mouseIDTaskL23 = [];
    for i = 1:nfiles
        % prepare some data for linear mixed model effect
        mouseIDTaskL23 = [mouseIDTaskL23 zeros(1,length(datastructActiveRes(i).scatFull))+i];
    end
else
    mouseIDTaskL5 = [];
    for i = 1:nfiles
        % prepare some data for linear mixed model effect
        mouseIDTaskL5 = [mouseIDTaskL5 zeros(1,length(datastructActiveRes(i).scatFull))+i];
    end
end

% full fam LMEM
data = cat(2, adi1,adi2)';
mouseID = categorical(cat(2, mouseIDTaskL23,mouseIDTaskL5))';
condition = categorical(cat(1, ones(length(mouseIDTaskL23),1),ones(length(mouseIDTaskL5),1)+1));
clear statTbl, statTbl = table(data, mouseID, condition);
lmeFullFam = fitlme(statTbl, 'data ~ condition + (1|mouseID)', 'CheckHessian', 1,'FitMethod', 'REML', 'StartMethod', 'random');
statsFullFam = anova(lmeFullFam,'DFMethod','Satterthwaite');

figure('Position', [316         515        1123         420])
boxchart([ones(size(adi1)), ones(size(adi2))+1], ...
    [adi1, adi2], 'MarkerStyle','none'); hold on
xlim([0 3]), ylabel('Response'), xticks([1 2])
xticklabels({'L2/3', 'L5'}), xtickangle(45); 

figure
scatter([1 2],[nanmean(adi1) nanmean(adi2)], sz, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2],[nanmean(adi1) nanmean(adi2)], ...
    [calcSem(adi1) calcSem(adi2)] ...
    ,[calcSem(adi1) calcSem(adi2)]);
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 3]), ylim([0.5 0.9]), ylabel('ADI'), xticks([1 2]), 
xticklabels({'L2/3','L5'}), 

%% Dampening or sharpening analyses
% order responses to the four images per neuron, sort in ascending order,
% average over neurons, pre vs post. See review de Lange fig 2.
ix = famIdx;
% ix = novIdx;

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
er = errorbar(imgFullPreMn,imgFullPreSem); hold on
er.Color = [0 0.4470 0.7410]; er.LineStyle = 'none'; er.LineWidth = 2; title('Average response per condition')
er = errorbar(imgFullPostMn,imgFullPostSem); 
er.Color = [0.9290 0.6940 0.1250]; er.LineStyle = 'none'; er.LineWidth = 2; title('Average response per condition')
plot(imgFullPreMn)
plot(imgFullPostMn), ylim([-0.2 1])
title('Response strength per full image')
ylabel('dF/F')
subplot(1,2,2)
er = errorbar(imgOcclPreMn,imgOcclPreSem); hold on
er.Color = [0 0.4470 0.7410]; er.LineStyle = 'none'; er.LineWidth = 2; title('Average response per condition')
er = errorbar(imgOcclPostMn,imgOcclPostSem); 
er.Color = [0.9290 0.6940 0.1250]; er.LineStyle = 'none'; er.LineWidth = 2; title('Average response per condition')
plot(imgOcclPreMn)
plot(imgOcclPostMn), ylim([-0.2 1])
title('Response strength per Occl image')
ylabel('dF/F')


%% Running speed analyses
rocImgNrs = [1 2 4 5];
% rocImgNrs = [3 6];
nImgsRoc = length(rocImgNrs);
occlusion = 1; % 1 = occl images, 0 = full images
    
% familiar images and occluded trials
imgIdxPreRun = false(1,size(matTrialTypesSort,2));
imgIdxPostRun = false(1,size(matTrialTypesSort,2));
for n = 1:nImgsRoc
    imgIdxPreRun(matTrialTypesSort(1,:)==rocImgNrs(n)&matTrialTypesSort(2,:)==occlusion)=1; % one of the selected images as well as occl images
    imgIdxPostRun(matTrialTypesSort(1,:)==rocImgNrs(n)&matTrialTypesSort(2,:)==occlusion)=1; % one of the selected images as well as occl images
end

% run data for all mice
runSpeedPrePop = zeros(size(datastructPre(1).runSpeed,1), size(datastructPre(1).runSpeed,2),nfiles);
runSpeedPostPop = zeros(size(datastructPost(1).runSpeed,1), size(datastructPost(1).runSpeed,2),nfiles);
matDataPopPre = zeros(size(datastructPre(1).matData,1),nfiles);
matDataPopPost = zeros(size(datastructPost(1).matData,1),nfiles);
for i = 1:nfiles
    runSpeedPrePop(:,:,i) = datastructPre(i).runSpeed(:,:,1);
    runSpeedPostPop(:,:,i) = datastructPost(i).runSpeed(:,:,1);
    matDataPopPre(:,i) = nanmean(datastructPre(i).matData,2);
    matDataPopPost(:,i) = nanmean(datastructPost(i).matData,2);
end

% select right data from running trials
% runSpeedMnPostPop = squeeze(mean(runSpeedPostPop(vecAxSt,imgIdxPostRun,:))-mean(runSpeedPostPop(vecAxSp,imgIdxPostRun,:)));
runSpeedMnPrePop = squeeze(nanmean(runSpeedPrePop(vecAxSt,imgIdxPreRun,:)));
runSpeedMnPostPop = squeeze(nanmean(runSpeedPostPop(vecAxSt,imgIdxPostRun,:)));
occlDataPrePop = matDataPopPre(imgIdxPreRun,:);
occlDataPostPop = matDataPopPost(imgIdxPostRun,:);

% % % here we plot the average occl response over all ROIs per mouse
% figure, for i = 1:6, scatter(runSpeedMnPrePop(:,i),occlDataPrePop(:,i)),refline,hold on, pause,end
% figure, for i = 1:6, scatter(runSpeedMnPostPop(:,i),occlDataPostPop(:,i)),refline,hold on, pause,end

%%%%% now calculate correlation between running and occl response for each
%%%%% neuron so that we can compare occl responders with nonoccluded
%%%%% responders
matDataCellPre = cat(2,datastructPre(:).matData);
matDataCellPre = matDataCellPre(imgIdxPreRun,:);
matDataCellPost = cat(2,datastructPost(:).matData);
matDataCellPost = matDataCellPost(imgIdxPostRun,:);
runSpeedCellPre = cat(3,datastructPre(:).runSpeed);
runSpeedCellPre = squeeze(nanmean(runSpeedCellPre(vecAxSt,imgIdxPreRun,:)));
runSpeedCellPost = cat(3,datastructPost(:).runSpeed);
runSpeedCellPost = squeeze(nanmean(runSpeedCellPost(vecAxSt,imgIdxPostRun,:)));

% Naive mice
rRunPre = zeros(size(matDataCellPre,2),1);
pRunPre = zeros(size(matDataCellPre,2),1);
for i = 1:size(matDataCellPre,2)
    x = runSpeedCellPre(:,i);
    y = matDataCellPre(:,i);
    [rRunPre(i),pRunPre(i)]=corr(x,y, 'Type', 'Pearson');
end

% Expert mice
rRunPost = zeros(size(matDataCellPost,2),1);
pRunPost = zeros(size(matDataCellPost,2),1);
for i = 1:size(matDataCellPost,2)
    x = runSpeedCellPost(:,i);
    y = matDataCellPost(:,i);
    [rRunPost(i),pRunPost(i)]=corr(x,y, 'Type', 'Pearson');
end

mean(rRunPost(scatOcclFamPopPost>1))
mean(rRunPost(scatFullFamPopPost>1))

runThres = 1;
clear occlStatPost occlRunPost
for i = 1:nfiles
    occlStatPost(i) = nanmean(occlDataPostPop(runSpeedMnPostPop(:,i)<runThres,i));
    occlRunPost(i) = nanmean(occlDataPostPop(runSpeedMnPostPop(:,i)>runThres,i));
%     fullStatPost(i) = mean(fullDataPostPop(runSpeedMnPostPop(:,i)<runThres,i));
%     fullRunPost(i) = mean(fullDataPostPop(runSpeedMnPostPop(:,i)>runThres,i));
end
figure('Position', [680   475   353   503]) 
% subplot(1,2,2)
scatter(ones(5,1), occlStatPost), hold on, scatter(ones(5,1)+1, occlRunPost)
xlim([0 3]), ylim([0 1]), ylabel('Occl response')
xticks([1 2]), xticklabels({'Stat','Run'})
% subplot(1,2,2)
% scatter(ones(6,1), fullStatPost), hold on, scatter(ones(6,1)+1, fullRunPost)
% xlim([0 3]), ylim([0 2]), ylabel('Occl response')
% xticks([1 2]), xticklabels({'Naive','Expert'})

%% Locomotion modulation index full vs occl
rocImgNrs = [1 2 4 5];
% rocImgNrs = [3 6];
nImgsRoc = length(rocImgNrs);
% occlusion = 1; % 1 = occl images, 0 = full images
    
% % familiar images and occluded trials
% imgIdxPreRun = false(1,size(matTrialTypesSort,2));
% imgIdxPostRun = false(1,size(matTrialTypesSort,2));
% for n = 1:nImgsRoc
%     imgIdxPreRun(matTrialTypesSort(1,:)==rocImgNrs(n)&matTrialTypesSort(2,:)==occlusion)=1; % one of the selected images as well as occl images
%     imgIdxPostRun(matTrialTypesSort(1,:)==rocImgNrs(n)&matTrialTypesSort(2,:)==occlusion)=1; % one of the selected images as well as occl images
% end

runTrialsPopPre = [];
runTrialsPopPost = [];
for i = 1:nfiles
    run = datastructPre(i).runTrials';
    runTrialsPopPre = cat(2,runTrialsPopPre, repmat(run, [1, sum(datastructPre(i).rfIncl)]));
    run = datastructPost(i).runTrials';
    runTrialsPopPost = cat(2,runTrialsPopPost, repmat(run, [1, sum(datastructPost(i).rfIncl)]));
end

matDataCellPre = cat(2,datastructPre(:).matData);
matDataCellPost = cat(2,datastructPost(:).matData);

fullPreRun = zeros(size(runTrialsPopPre,2),1);
occlPreRun = zeros(size(runTrialsPopPre,2),1);
fullPreStat = zeros(size(runTrialsPopPre,2),1);
occlPreStat = zeros(size(runTrialsPopPre,2),1);
for i = 1:size(runTrialsPopPre,2)
    fullPreRun(i) = mean(matDataCellPre(runTrialsPopPre(:,i)'&matTrialTypesSort(2,:)==0,i)); % nonoccluded images
    occlPreRun(i) = mean(matDataCellPre(runTrialsPopPre(:,i)'&matTrialTypesSort(2,:)==1,i)); % occluded images
    fullPreStat(i) = mean(matDataCellPre(~runTrialsPopPre(:,i)'&matTrialTypesSort(2,:)==0,i)); % nonoccluded images
    occlPreStat(i) = mean(matDataCellPre(~runTrialsPopPre(:,i)'&matTrialTypesSort(2,:)==1,i)); % occluded images
end
fullPostRun = zeros(size(runTrialsPopPost,2),1);
occlPostRun = zeros(size(runTrialsPopPost,2),1);
fullPostStat = zeros(size(runTrialsPopPost,2),1);
occlPostStat = zeros(size(runTrialsPopPost,2),1);
for i = 1:size(runTrialsPopPost,2)
    fullPostRun(i) = mean(matDataCellPost(runTrialsPopPost(:,i)'&matTrialTypesSort(2,:)==0,i));
    occlPostRun(i) = mean(matDataCellPost(runTrialsPopPost(:,i)'&matTrialTypesSort(2,:)==1,i));
    fullPostStat(i) = mean(matDataCellPost(~runTrialsPopPost(:,i)'&matTrialTypesSort(2,:)==0,i));
    occlPostStat(i) = mean(matDataCellPost(~runTrialsPopPost(:,i)'&matTrialTypesSort(2,:)==1,i));
end

figure('Position', [316         515        1123         420])
subplot(1,3,1)
boxchart([ones(size(fullPostStat')), ones(size(fullPostRun'))+1], ...
    [fullPostStat', fullPostRun'], 'MarkerStyle','none'); hold on
xlim([0 3]), ylim([-0.5 1.5]), ylabel('Response'), xticks([1 2])
xticklabels({'Pre', 'Post'}), xtickangle(45); 
subplot(1,3,2)
boxchart([ones(size(occlPostStat')), ones(size(occlPostRun'))+1], ...
    [occlPostStat', occlPostRun'], 'MarkerStyle','none'); hold on
xlim([0 3]), ylim([-0.5 1.5]), ylabel('Response'), xticks([1 2])
xticklabels({'Pre', 'Post'}), xtickangle(45); 
subplot(1,3,3)
boxchart([ones(size(fullPostStat')), ones(size(fullPostRun'))+1], ...
    [fullPostRun'-fullPostStat', occlPostRun'-occlPostStat'], 'MarkerStyle','none'); hold on
xlim([0 3]), ylim([-0.5 1.5]), ylabel('Response difference (run-stat)'), xticks([1 2])
xticklabels({'Full', 'Occl'}), xtickangle(45); 


LMIfullPre = (fullPreRun-fullPreStat)./(fullPreRun+fullPreStat);
LMIocclPre = (occlPreRun-occlPreStat)./(occlPreRun+occlPreStat);
LMIfullPre(LMIfullPre>1)=1; LMIfullPre(LMIfullPre<-1)=-1;
LMIocclPre(LMIocclPre>1)=1; LMIocclPre(LMIocclPre<-1)=-1;

LMIfullPost = (fullPostRun-fullPostStat)./(fullPostRun+fullPostStat);
LMIocclPost = (occlPostRun-occlPostStat)./(occlPostRun+occlPostStat);
LMIfullPost(LMIfullPost>1)=1; LMIfullPost(LMIfullPost<-1)=-1;
LMIocclPost(LMIocclPost>1)=1; LMIocclPost(LMIocclPost<-1)=-1;

figure('Position', [316         515        1123         420])
subplot(1,3,1)
boxchart([ones(size(LMIfullPre')), ones(size(LMIfullPost'))+1], ...
    [LMIfullPre', LMIfullPost'], 'MarkerStyle','none'); hold on
xlim([0 3]), ylim([-1.2 1.2]), ylabel('LMI full'), xticks([1 2])
xticklabels({'Pre', 'Post'}), xtickangle(45); 
subplot(1,3,2)
boxchart([ones(size(LMIocclPre')), ones(size(LMIocclPost'))+1], ...
    [LMIocclPre', LMIocclPost'], 'MarkerStyle','none'); hold on
xlim([0 3]), ylim([-1.2 1.2]), ylabel('LMI occl'), xticks([1 2])
xticklabels({'Pre', 'Post'}), xtickangle(45); 

figure('Position', [316         515        1123         420])
subplot(1,3,1)
boxchart([ones(size(LMIfullPre')), ones(size(LMIocclPre'))+1], ...
    [LMIfullPre', LMIocclPre'], 'MarkerStyle','none'); hold on
xlim([0 3]), ylim([-1.2 1.2]), ylabel('LMI Pre'), xticks([1 2])
xticklabels({'Full', 'Occl'}), xtickangle(45); 
subplot(1,3,2)
boxchart([ones(size(LMIfullPost')), ones(size(LMIocclPost'))+1], ...
    [LMIfullPost', LMIocclPost'], 'MarkerStyle','none'); hold on
xlim([0 3]), ylim([-1.2 1.2]), ylabel('LMI Post'), xticks([1 2])
xticklabels({'Full', 'Occl'}), xtickangle(45); 

figure('Position', [942   339   296   493])
scatter([1 2 4 5],[nanmean(LMIfullPre) nanmean(LMIocclPre) nanmean(LMIfullPost) nanmean(LMIocclPost)], 35, 'k', 'LineWidth', 2), hold on
er = errorbar([1 2 4 5],[nanmean(LMIfullPre) nanmean(LMIocclPre) nanmean(LMIfullPost) nanmean(LMIocclPost)], ...
    [calcSem(LMIfullPre) calcSem(LMIocclPre) calcSem(LMIfullPost) calcSem(LMIocclPost)] ...
    ,[calcSem(LMIfullPre) calcSem(LMIocclPre) calcSem(LMIfullPost) calcSem(LMIocclPost)]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 6]), ylim([0 0.4]), 

% [pRunMod,~] = ranksum(LMIfull, LMIoccl)

if save_fig
    func_save_fig('L23_LMIpost')
    func_save_fig('L5_LMIpost')
end

% naive full vs occl lmi LMEM
data = cat(2, LMIfullPre',LMIocclPre')';
mouseID = categorical(cat(2, mouseIDPre,mouseIDPre))';
condition = categorical(cat(1, ones(length(mouseIDPre),1),ones(length(mouseIDPre),1)+1));
clear statTbl, statTbl = table(data, mouseID, condition);
lmeNaiveLmi = fitlme(statTbl, 'data ~ condition + (1|mouseID)', 'CheckHessian', 1,'FitMethod', 'REML', 'StartMethod', 'random');
statsNaiveLmi = anova(lmeNaiveLmi,'DFMethod','Satterthwaite');

% expert full vs occl lmi LMEM
data = cat(2, LMIfullPost',LMIocclPost')';
mouseID = categorical(cat(2, mouseIDPost,mouseIDPost))';
condition = categorical(cat(1, ones(length(mouseIDPost),1),ones(length(mouseIDPost),1)+1));
clear statTbl, statTbl = table(data, mouseID, condition);
lmeFullLmi = fitlme(statTbl, 'data ~ condition + (1|mouseID)', 'CheckHessian', 1,'FitMethod', 'REML', 'StartMethod', 'random');
statsFullLmi = anova(lmeFullLmi,'DFMethod','Satterthwaite');

% full naive vs expert lmi LMEM
data = cat(2, LMIfullPre',LMIfullPost')';
mouseID = categorical(cat(2, mouseIDPre,mouseIDPost))';
condition = categorical(cat(1, ones(length(mouseIDPre),1),ones(length(mouseIDPost),1)+1));
clear statTbl, statTbl = table(data, mouseID, condition);
lmeFullLmi = fitlme(statTbl, 'data ~ condition + (1|mouseID)', 'CheckHessian', 1,'FitMethod', 'REML', 'StartMethod', 'random');
statsFullLmi = anova(lmeFullLmi,'DFMethod','Satterthwaite');

% occl naive vs expert lmi LMEM
data = cat(2, LMIocclPre',LMIocclPost')';
mouseID = categorical(cat(2, mouseIDPre,mouseIDPost))';
condition = categorical(cat(1, ones(length(mouseIDPre),1),ones(length(mouseIDPost),1)+1));
clear statTbl, statTbl = table(data, mouseID, condition);
lmeOcclLmi = fitlme(statTbl, 'data ~ condition + (1|mouseID)', 'CheckHessian', 1,'FitMethod', 'REML', 'StartMethod', 'random');
statsOcclLmi = anova(lmeOcclLmi,'DFMethod','Satterthwaite');

% find out whether NO responders have higher LMI than O responders
postRun = zeros(size(runTrialsPopPost,2),1);
postStat = zeros(size(runTrialsPopPost,2),1);
for i = 1:size(runTrialsPopPost,2)
    postRun(i) = mean(matDataCellPost(logical(runTrialsPopPost(:,i)),i));
    postStat(i) = mean(matDataCellPost(logical(~runTrialsPopPost(:,i)),i));
end

LMIpost = (postRun-postStat)./(postRun+postStat);
LMIpost(LMIpost<-1)=-1;
LMIpost(LMIpost>1)=1;

fulls = scatFullFamPopPost > 0.5;
occls = scatOcclFamPopPost > 0.5;

figure('Position', [316         515        1123         420])
subplot(1,3,1)
boxchart([ones(size(LMIpost(fulls)')), ones(size(LMIpost(occls)'))+1], ...
    [LMIpost(fulls)', LMIpost(occls)'], 'MarkerStyle','none'); hold on
xlim([0 3]), ylim([-1.2 1.2]), ylabel('LMI Pre'), xticks([1 2])
xticklabels({'Fulls', 'Occls'}), xtickangle(45); 
subplot(1,3,2)
scatter([1 2],[nanmean(LMIpost(fulls)') nanmean(LMIpost(occls)')], 35, 'k', 'LineWidth', 2), hold on
er = errorbar([1 2],[nanmean(LMIpost(fulls)') nanmean(LMIpost(occls)')], ...
    [calcSem(LMIpost(fulls)') calcSem(LMIpost(occls)')] ...
    ,[calcSem(LMIpost(fulls)') calcSem(LMIpost(occls)')]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 4]), ylim([0 0.4]), 

figure, 
subplot(1,2,1)
scatter(LMIpost, scatFullFamPopPost), refline
subplot(1,2,2)
scatter(LMIpost, scatOcclFamPopPost), refline



%% correlation plots
% correlations are done on all data, plotting is done with 'cut off data' 
save_fig = false;

ix = famIdx;
ix = novIdx;

% load in active data
if nfiles == 6
    load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\postActiveTraceDataForCorrsL23zscored.mat')
elseif nfiles == 5
    load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\postActiveTraceDataForCorrsL5zscoredv2.mat')
end
vecAxTaskSt = vecAxTask>0.2 & vecAxTask<1;
vecAxTaskSp = vecAxTask<0;

corrFullPrePrep = squeeze(nanmean(imgFullResMnPopPre(vecAxSt,ix,:))-nanmean(imgFullResMnPopPre(vecAxSp,ix,:)));
corrOcclPrePrep = squeeze(nanmean(imgOcclResMnPopPre(vecAxSt,ix,:))-nanmean(imgOcclResMnPopPre(vecAxSp,ix,:)));
corrFullPostPrep = squeeze(nanmean(imgFullResMnPopPost(vecAxSt,ix,:))-nanmean(imgFullResMnPopPost(vecAxSp,ix,:)));
corrOcclPostPrep = squeeze(nanmean(imgOcclResMnPopPost(vecAxSt,ix,:))-nanmean(imgOcclResMnPopPost(vecAxSp,ix,:)));
corrFullTaskPrep = squeeze(nanmean(imgFullResMnPop(vecAxTaskSt,:,:))-nanmean(imgFullResMnPop(vecAxTaskSp,:,:)));
corrOcclTaskPrep = squeeze(nanmean(imgOcclResMnPop(vecAxTaskSt,:,:))-nanmean(imgOcclResMnPop(vecAxTaskSp,:,:)));

mnValCut = -0.5; % min val for cutting for plotting
mxValCut = 2.5; % max val for cutting for plotting
sz = 7; % scatter size for plotting

%%%%%%%% correlation plots full/full, occl/occl
% possible correlations for full/full and occl/occl
c = nchoosek(1:length(ix),2);
nrComs = size(c,1);

% full pre
clear Rp Pp Rs Ps
figure('Position', [200    42   120   954])
for i = 1:nrComs
    subplot(8,1,i)
    x = corrFullPrePrep(c(i,1),:);
    y = corrFullPrePrep(c(i,2),:);
    x1 = x; x1(x1>mxValCut)=mxValCut+0.5;x1(x1<mnValCut)=mnValCut-0.5;
    y1 = y; y1(y1>mxValCut)=mxValCut+0.5;y1(y1<mnValCut)=mnValCut-0.5;
    scatter(x1,y1,sz,'filled', 'k')
%     if nfiles==6
        xlim([mnValCut-0.5 mxValCut+0.5]), ylim([mnValCut-0.5 mxValCut+0.5])
%     end
    refline(1), xline(0), yline(0)
    [Rp(i),Pp(i)]=corr(x',y', 'Type', 'Pearson');
    [Rs(i),Ps(i)]=corr(x',y', 'Type', 'Spearman');
    title(sprintf('%d vs %d', c(i,1), c(i,2)))
%     text(-15,35,sprintf('Rp=%.2f', Rp(i))), text(15,35,sprintf('Pp=%.2f', Pp(i)))
%     text(-15,27,sprintf('Rs=%.2f', Rs(i))), text(15,27,sprintf('Ps=%.2f', Ps(i)))
%     xticks(mnValCut-10:10:mxValCut+10), yticks(mnValCut-10:10:mxValCut+10)
    xticks(''), yticks('')
%     if i ~= 1
%         xticklabels(''), yticklabels('')
%     end
end
RpPreFull = Rp; PpPreFull = Pp; RsPreFull = Rs; PsPreFull = Ps;
% if save_fig, func_save_fig('L23_FullPreCorrs'), end
if save_fig, func_save_fig('L5_FullPreCorrs'), end

% full post
clear Rp Pp Rs Ps
figure('Position', [400    42   120   954])
for i = 1:nrComs
    subplot(8,1,i)
    x = corrFullPostPrep(c(i,1),:);
    y = corrFullPostPrep(c(i,2),:);
    x1 = x; x1(x1>mxValCut)=mxValCut+0.5;x1(x1<mnValCut)=mnValCut-0.5;
    y1 = y; y1(y1>mxValCut)=mxValCut+0.5;y1(y1<mnValCut)=mnValCut-0.5;
    scatter(x1,y1,sz,'filled', 'k')
%     if nfiles==6
        xlim([mnValCut-0.5 mxValCut+0.5]), ylim([mnValCut-0.5 mxValCut+0.5])
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
RpPostFull = Rp; PpPostFull = Pp; RsPostFull = Rs; PsPostFull = Ps;
% if save_fig, func_save_fig('L23_FullPostCorrs'), end
if save_fig, func_save_fig('L5_FullPostCorrs'), end

% Task
clear Rp Pp Rs Ps
figure('Position', [400    42   120   954])
for i = 1:nrComs
    subplot(8,1,i)
    x = corrFullTaskPrep(c(i,1),:);
    y = corrFullTaskPrep(c(i,2),:);
    x1 = x; x1(x1>mxValCut)=mxValCut+0.5;x1(x1<mnValCut)=mnValCut-0.5;
    y1 = y; y1(y1>mxValCut)=mxValCut+0.5;y1(y1<mnValCut)=mnValCut-0.5;
    scatter(x1,y1,sz,'filled', 'k')
%     if nfiles==6
        xlim([mnValCut-0.5 mxValCut+0.5]), ylim([mnValCut-0.5 mxValCut+0.5])
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
RpTaskFull = Rp; PpTaskFull = Pp; RsTaskFull = Rs; PsTaskFull = Ps;
% if save_fig, func_save_fig('L23_FullTaskCorrs'), end
if save_fig, func_save_fig('L5_FullTaskCorrs'), end

% occl pre
clear Rp Pp Rs Ps
figure('Position', [700    42   120   954])
for i = 1:nrComs
    subplot(8,1,i)
    x = corrOcclPrePrep(c(i,1),:);
    y = corrOcclPrePrep(c(i,2),:);
    x1 = x; x1(x1>mxValCut)=mxValCut+0.5;x1(x1<mnValCut)=mnValCut-0.5;
    y1 = y; y1(y1>mxValCut)=mxValCut+0.5;y1(y1<mnValCut)=mnValCut-0.5;
    scatter(x1,y1,sz,'filled', 'k')
%     if nfiles==6
        xlim([mnValCut-0.5 mxValCut+0.5]), ylim([mnValCut-0.5 mxValCut+0.5])
%     end
    refline(1), xline(0), yline(0)
    [Rp(i),Pp(i)]=corr(x',y', 'Type', 'Pearson');
    [Rs(i),Ps(i)]=corr(x',y', 'Type', 'Spearman');
%     if i == 1
%         title('Occl pre corrs')
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
RpPreOccl = Rp; PpPreOccl = Pp; RsPreOccl = Rs; PsPreOccl = Ps;
% if save_fig, func_save_fig('L23_OcclPreCorrs'), end
if save_fig, func_save_fig('L5_OcclPreCorrs'), end

% occl post
clear Rp Pp Rs Ps
figure('Position', [900    42   120   954])
for i = 1:nrComs
    subplot(8,1,i)
    x = corrOcclPostPrep(c(i,1),:);
    y = corrOcclPostPrep(c(i,2),:);
    x1 = x; x1(x1>mxValCut)=mxValCut+0.5;x1(x1<mnValCut)=mnValCut-0.5;
    y1 = y; y1(y1>mxValCut)=mxValCut+0.5;y1(y1<mnValCut)=mnValCut-0.5;
    scatter(x1,y1,sz,'filled', 'k')
%     if nfiles==6
        xlim([mnValCut-0.5 mxValCut+0.5]), ylim([mnValCut-0.5 mxValCut+0.5])
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
RpPostOccl = Rp; PpPostOccl = Pp; RsPostOccl = Rs; PsPostOccl = Ps;
% if save_fig, func_save_fig('L23_OcclPostCorrs'), end
if save_fig, func_save_fig('L5_OcclPostCorrs'), end

% Task
clear Rp Pp Rs Ps
figure('Position', [400    42   120   954])
for i = 1:nrComs
    subplot(8,1,i)
    x = corrOcclTaskPrep(c(i,1),:);
    y = corrOcclTaskPrep(c(i,2),:);
    x1 = x; x1(x1>mxValCut)=mxValCut+0.5;x1(x1<mnValCut)=mnValCut-0.5;
    y1 = y; y1(y1>mxValCut)=mxValCut+0.5;y1(y1<mnValCut)=mnValCut-0.5;
    scatter(x1,y1,sz,'filled', 'k')
%     if nfiles==6
        xlim([mnValCut-0.5 mxValCut+0.5]), ylim([mnValCut-0.5 mxValCut+0.5])
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
RpTaskOccl = Rp; PpTaskOccl = Pp; RsTaskOccl = Rs; PsTaskOccl = Ps;
% if save_fig, func_save_fig('L23_OcclTaskCorrs'), end
if save_fig, func_save_fig('L5_OcclTaskCorrs'), end


%%%%%% this section calculates the same correlation but now between all
%%%%%% full and occluded images so that we can compare full-full and
%%%%%% occl-occl versus full-occl
% possible correlations for full/occl
% possible options: 1-1, 1-2, 1-3, 1-4 etc, so 4x4 = 16 options?
clear c
c(:,1) = [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4];
c(:,2) = [1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4];
% c(:,1) = [1 1 2 2]; % for nov images
% c(:,2) = [1 2 1 2];
nrComs = length(c);
order = [1:2:nrComs 2:2:nrComs];

clear Rp Pp Rs Ps
figure('Position', [125    42   249   954])
for i = 1:nrComs
    subplot(8,nrComs/8,order(i))
%     subplot(2,2,order(i))
    x = corrFullPrePrep(c(i,1),:);
    y = corrOcclPrePrep(c(i,2),:);
    x1 = x; x1(x1>mxValCut)=mxValCut+0.5;x1(x1<mnValCut)=mnValCut-0.5;
    y1 = y; y1(y1>mxValCut)=mxValCut+0.5;y1(y1<mnValCut)=mnValCut-0.5;
    scatter(x1,y1,sz,'filled', 'k')
%     if nfiles==6
        xlim([mnValCut-0.5 mxValCut+0.5]), ylim([mnValCut-0.5 mxValCut+0.5])
%     end
    refline(1), xline(0), yline(0)
    [Rp(i),Pp(i)]=corr(x',y', 'Type', 'Pearson');
    [Rs(i),Ps(i)]=corr(x',y', 'Type', 'Spearman');
%     if i == 1
%         title('Full-Occl pre corrs')
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
RpPreFullOccl = Rp; PpPreFullOccl = Pp; RsPreFullOccl = Rs; PsPreFullOccl = Ps;
% if save_fig, func_save_fig('L23_FullOcclPreCorrs'), end
if save_fig, func_save_fig('L5_FullOcclPreCorrs'), end

clear Rp Pp Rs Ps
figure('Position', [425    42   249   954])
for i = 1:nrComs
    subplot(8,nrComs/8,order(i))
%     subplot(2,2,order(i))
    x = corrFullPostPrep(c(i,1),:);
    y = corrOcclPostPrep(c(i,2),:);
    x1 = x; x1(x1>mxValCut)=mxValCut+0.5;x1(x1<mnValCut)=mnValCut-0.5;
    y1 = y; y1(y1>mxValCut)=mxValCut+0.5;y1(y1<mnValCut)=mnValCut-0.5;
    scatter(x1,y1,sz,'filled', 'k')
%     if nfiles==6
        xlim([mnValCut-0.5 mxValCut+0.5]), ylim([mnValCut-0.5 mxValCut+0.5])
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
RpPostFullOccl = Rp; PpPostFullOccl = Pp; RsPostFullOccl = Rs; PsPostFullOccl = Ps;
% if save_fig, func_save_fig('L23_FullOcclPostCorrs'), end
if save_fig, func_save_fig('L5_FullOcclPostCorrs'), end

clear Rp Pp Rs Ps
figure('Position', [425    42   249   954])
for i = 1:nrComs
    subplot(8,nrComs/8,order(i))
%     subplot(2,2,order(i))
    x = corrFullTaskPrep(c(i,1),:);
    y = corrOcclTaskPrep(c(i,2),:);
    x1 = x; x1(x1>mxValCut)=mxValCut+0.5;x1(x1<mnValCut)=mnValCut-0.5;
    y1 = y; y1(y1>mxValCut)=mxValCut+0.5;y1(y1<mnValCut)=mnValCut-0.5;
    scatter(x1,y1,sz,'filled', 'k')
%     if nfiles==6
        xlim([mnValCut-0.5 mxValCut+0.5]), ylim([mnValCut-0.5 mxValCut+0.5])
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
RpTaskFullOccl = Rp; PpTaskFullOccl = Pp; RsTaskFullOccl = Rs; PsTaskFullOccl = Ps;
% if save_fig, func_save_fig('L23_FullOcclTaskCorrs'), end
if save_fig, func_save_fig('L5_FullOcclTaskCorrs'), end

figure('Position', [618   428   841   318])
subplot(1,3,1)
boxchart([ones(size(RpPreFull)), ones(size(RpPreOccl))+1, ones(size(RpPreFullOccl))+2], ...
    [RpPreFull, RpPreOccl, RpPreFullOccl], 'MarkerStyle','none'), hold on
xlim([0 4]), ylabel('Correlation'), xticks([1 2 3]); ylim([-0.3 1])
xticklabels({'PreN', 'PreO', 'PreNO'}), xtickangle(45), 

subplot(1,3,2)
boxchart([ones(size(RpPostFull)), ones(size(RpPostOccl))+1, ones(size(RpPostFullOccl))+2], ...
    [RpPostFull, RpPostOccl, RpPostFullOccl], 'MarkerStyle','none'), hold on
xlim([0 4]), ylabel('Correlation'), xticks([1 2 3]); ylim([-0.3 1])
xticklabels({'PostN', 'PostO', 'PostNO'}), xtickangle(45), 

subplot(1,3,3)
boxchart([ones(size(RpTaskFull)), ones(size(RpTaskOccl))+1, ones(size(RpTaskFullOccl))+2], ...
    [RpTaskFull, RpTaskOccl, RpTaskFullOccl], 'MarkerStyle','none'), hold on
xlim([0 4]), ylabel('Correlation'), xticks([1 2 3]); ylim([-0.3 1])
xticklabels({'TaskN', 'TaskO', 'TaskNO'}), xtickangle(45), 

save_fig = false;
if save_fig
    func_save_fig('L23_CorrQuants')
    func_save_fig('L5_CorrQuants')
end


%%%%%% stats for pearson
n_comparisons = 3; % for bonferroni correction
% pre
[p,~] = ranksum(RpPreFull, RpPreOccl); pPreFO = p*n_comparisons
[p,~] = ranksum(RpPreFull, RpPreFullOccl); pPreFFO = p*n_comparisons
[p,~] = ranksum(RpPreOccl, RpPreFullOccl); pPreOFO = p*n_comparisons

% post
[p,~] = ranksum(RpPostFull, RpPostOccl); pPostFO = p*n_comparisons
[p,~] = ranksum(RpPostFull, RpPostFullOccl); pPostFFO = p*n_comparisons
[p,~] = ranksum(RpPostOccl, RpPostFullOccl); pPostOFO = p*n_comparisons

% task
[p,~] = ranksum(RpTaskFull, RpTaskOccl); pTaskFO = p*n_comparisons
[p,~] = ranksum(RpTaskFull, RpTaskFullOccl); pTaskFFO = p*n_comparisons
[p,~] = ranksum(RpTaskOccl, RpTaskFullOccl); pTaskOFO = p*n_comparisons


%% correlations per mouse

%%%%% CHECK BELOW IN THE FOR LOOP IF YOU USE L2/3 DATA OR L5, YOU HAVE TO
%%%%% CHANGE STUFF THERE

% correlations are done on all data, plotting is done with 'cut off data' 
save_fig = false;

% ix = novIdx;
ix = famIdx;

% load in active data
if nfiles == 6
    load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\datastructActiveL23.mat')
elseif nfiles == 5
    load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\datastructActiveL5v2.mat')
end
vecAxTaskSt = vecAxTask>0.2 & vecAxTask<1;
vecAxTaskSp = vecAxTask<0;

RpPreFullMs = nan(nfiles,1);
RpPreOcclMs = nan(nfiles,1);
RpPreFullOcclMs = nan(nfiles,1);
RpPostFullMs = nan(nfiles,1);
RpPostOcclMs = nan(nfiles,1);
RpPostFullOcclMs = nan(nfiles,1);
RpTaskFullMs = nan(nfiles,1);
RpTaskOcclMs = nan(nfiles,1);
RpTaskFullOcclMs = nan(nfiles,1);
RpPreFullOcclBasicMs = nan(nfiles,1);
RpPostFullOcclBasicMs = nan(nfiles,1);
RpTaskFullOcclBasicMs = nan(nfiles,1);


% for k = 1:nfiles
for k = [1 2 4 5] % for L5, because mouse 3 only has 1 neuron
    
    c = nchoosek(1:length(ix),2);
    nrComs = size(c,1);
    % full pre
    prepFull = squeeze(nanmean(datastructPre(k).imgFullResMn(vecAxSt,ix,:))-nanmean(datastructPre(k).imgFullResMn(vecAxSp,ix,:)));
    clear Rp
    for i = 1:nrComs
        x = prepFull(c(i,1),:);
        y = prepFull(c(i,2),:);
        Rp(i)=corr(x',y', 'Type', 'Pearson');
    end
    RpPreFullMs(k) = mean(Rp);

    % full post
    prepFull = squeeze(nanmean(datastructPost(k).imgFullResMn(vecAxSt,ix,:))-nanmean(datastructPost(k).imgFullResMn(vecAxSp,ix,:)));
    clear Rp
    for i = 1:nrComs
        x = prepFull(c(i,1),:);
        y = prepFull(c(i,2),:);
        if size(x,2)<1
            Rp(i)=NaN;
        else
            Rp(i)=corr(x',y', 'Type', 'Pearson');
        end
    end
    RpPostFullMs(k) = mean(Rp);

    % full task
    prepFull = squeeze(nanmean(datastructActiveRes(k).imgFullResMn(vecAxTaskSt,:,:))-nanmean(datastructActiveRes(k).imgFullResMn(vecAxTaskSp,:,:)));
    clear Rp
    for i = 1:nrComs
        x = prepFull(c(i,1),:);
        y = prepFull(c(i,2),:);
        if size(x,2)<1
            Rp(i)=NaN;
        else
            Rp(i)=corr(x',y', 'Type', 'Pearson');
        end
    end
    RpTaskFullMs(k) = mean(Rp);

    % occl pre
    prepOccl = squeeze(nanmean(datastructPre(k).imgOcclResMn(vecAxSt,ix,:))-nanmean(datastructPre(k).imgOcclResMn(vecAxSp,ix,:)));
    clear Rp
    for i = 1:nrComs
        x = prepOccl(c(i,1),:);
        y = prepOccl(c(i,2),:);
        Rp(i)=corr(x',y', 'Type', 'Pearson');
    end
    RpPreOcclMs(k) = mean(Rp);

    % occl post
    prepOccl = squeeze(nanmean(datastructPost(k).imgOcclResMn(vecAxSt,ix,:))-nanmean(datastructPost(k).imgOcclResMn(vecAxSp,ix,:)));
    clear Rp
    for i = 1:nrComs
        x = prepOccl(c(i,1),:);
        y = prepOccl(c(i,2),:);
        if size(x,2)<1
            Rp(i)=NaN;
        else
            Rp(i)=corr(x',y', 'Type', 'Pearson');
        end
    end
    RpPostOcclMs(k) = mean(Rp);

    % occl task
    prepOccl = squeeze(nanmean(datastructActiveRes(k).imgOcclResMn(vecAxTaskSt,:,:))-nanmean(datastructActiveRes(k).imgOcclResMn(vecAxTaskSp,:,:)));
    clear Rp
    for i = 1:nrComs
        x = prepOccl(c(i,1),:);
        y = prepOccl(c(i,2),:);
        if size(x,2)<1
            Rp(i)=NaN;
        else
            Rp(i)=corr(x',y', 'Type', 'Pearson');
        end
    end
    RpTaskOcclMs(k) = mean(Rp);


    % possible correlations for full/occl
    % possible options: 1-1, 1-2, 1-3, 1-4 etc, so 4x4 = 16 options?
    clear c
    if length(ix)==4
        c(:,1) = [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4];
        c(:,2) = [1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4];
    else
        c(:,1) = [1 1 2 2];
        c(:,2) = [1 2 1 2];
    end
    nrComs = length(c);

    % full-occl pre
    prepFull = squeeze(nanmean(datastructPre(k).imgFullResMn(vecAxSt,ix,:))-nanmean(datastructPre(k).imgFullResMn(vecAxSp,ix,:)));
    prepOccl = squeeze(nanmean(datastructPre(k).imgOcclResMn(vecAxSt,ix,:))-nanmean(datastructPre(k).imgOcclResMn(vecAxSp,ix,:)));
    clear Rp
    for i = 1:nrComs
        x = prepFull(c(i,1),:);
        y = prepOccl(c(i,2),:);
        Rp(i)=corr(x',y', 'Type', 'Pearson');
    end
    RpPreFullOcclMs(k) = mean(Rp);

    % full-occl post
    prepFull = squeeze(nanmean(datastructPost(k).imgFullResMn(vecAxSt,ix,:))-nanmean(datastructPost(k).imgFullResMn(vecAxSp,ix,:)));
    prepOccl = squeeze(nanmean(datastructPost(k).imgOcclResMn(vecAxSt,ix,:))-nanmean(datastructPost(k).imgOcclResMn(vecAxSp,ix,:)));
    clear Rp
    for i = 1:nrComs
        x = prepFull(c(i,1),:);
        y = prepOccl(c(i,2),:);
        if size(x,2)<1
            Rp(i)=NaN;
        else
            Rp(i)=corr(x',y', 'Type', 'Pearson');
        end
    end
    RpPostFullOcclMs(k) = mean(Rp);

    % full-occl task
    prepFull = squeeze(nanmean(datastructActiveRes(k).imgFullResMn(vecAxTaskSt,:,:))-nanmean(datastructActiveRes(k).imgFullResMn(vecAxTaskSp,:,:)));
    prepOccl = squeeze(nanmean(datastructActiveRes(k).imgOcclResMn(vecAxTaskSt,:,:))-nanmean(datastructActiveRes(k).imgOcclResMn(vecAxTaskSp,:,:)));
    c = nchoosek(1:length(ix),2);
    nrComs = size(c,1);
    clear Rp
    for i = 1:nrComs
        x = prepFull(c(i,1),:);
        y = prepOccl(c(i,2),:);
        if size(x,2)<1
            Rp(i)=NaN;
        else
            Rp(i)=corr(x',y', 'Type', 'Pearson');
        end
    end
    RpTaskFullOcclMs(k) = mean(Rp);

end

for k = 1:nfiles
    prepFull = datastructPre(k).scatFull;
    prepOccl = datastructPre(k).scatOccl;
    RpPreFullOcclBasicMs(k) = corr(prepFull',prepOccl', 'Type', 'Pearson');

    prepFull = datastructPost(k).scatFull;
    prepOccl = datastructPost(k).scatOccl;
    if isempty(prepFull)
        RpPostFullOcclBasicMs(k)=NaN;
    else
        RpPostFullOcclBasicMs(k) = corr(prepFull',prepOccl', 'Type', 'Pearson');
    end
    
    prepFull = datastructActiveRes(k).scatFull;
    prepOccl = datastructActiveRes(k).scatOccl;
    if isempty(prepFull)
        RpTaskFullOcclBasicMs(k) = NaN;
    else
        RpTaskFullOcclBasicMs(k) = corr(prepFull',prepOccl', 'Type', 'Pearson');
    end
end

figure('Position',[305         279        1419         395])
subplot(1,4,1)
bar([nanmean(RpPreFullMs), nanmean(RpPostFullMs), nanmean(RpTaskFullMs)], 'FaceColor', 'w', 'EdgeColor', 'k', 'FaceAlpha', 0.6, 'LineWidth', 2)
hold on
plot([RpPreFullMs, RpPostFullMs, RpTaskFullMs]', 'k', 'LineWidth', 1), ylim([-0.3 1])
xticklabels({'Naive','Expert', 'Task'}),ylabel('Correlation coefficient'),title('NO-NO'),
subplot(1,4,2)
bar([nanmean(RpPreOcclMs), nanmean(RpPostOcclMs), nanmean(RpTaskOcclMs)], 'FaceColor', 'w', 'EdgeColor', 'k', 'FaceAlpha', 0.6, 'LineWidth', 2)
hold on
plot([RpPreOcclMs, RpPostOcclMs, RpTaskOcclMs]', 'k', 'LineWidth', 1), ylim([-0.3 1]),xticklabels({'Naive','Expert', 'Task'}),title('O-O'),
subplot(1,4,3)
bar([nanmean(RpPreFullOcclMs), nanmean(RpPostFullOcclMs), nanmean(RpTaskFullOcclMs)], 'FaceColor', 'w', 'EdgeColor', 'k', 'FaceAlpha', 0.6, 'LineWidth', 2)
hold on
plot([RpPreFullOcclMs, RpPostFullOcclMs, RpTaskFullOcclMs]', 'k', 'LineWidth', 1), ylim([-0.3 1]),xticklabels({'Naive','Expert', 'Task'}),title('NO-O'),
subplot(1,4,4)
bar([nanmean(RpPreFullOcclBasicMs), nanmean(RpPostFullOcclBasicMs), nanmean(RpTaskFullOcclBasicMs)], 'FaceColor', 'w', 'EdgeColor', 'k', 'FaceAlpha', 0.6, 'LineWidth', 2)
hold on
plot([RpPreFullOcclBasicMs, RpPostFullOcclBasicMs, RpTaskFullOcclBasicMs]', 'k', 'LineWidth', 1), ylim([-0.3 1]),xticklabels({'Naive','Expert', 'Task'}),title('NO-O single'),

if save_fig
    func_save_fig('L23_corrsPerMouse1')
    func_save_fig('L5_corrsPerMouse1')
    func_save_fig('L23_FullOcclTaskCorrs')
end

figure('Position',[305         279        1419         395])
subplot(1,4,1)
bar([nanmean(RpPreOcclMs), nanmean(RpPreFullMs), nanmean(RpPreFullOcclMs)], 'FaceColor', 'w', 'EdgeColor', 'k', 'FaceAlpha', 0.6, 'LineWidth', 2)
hold on
plot([RpPreOcclMs, RpPreFullMs, RpPreFullOcclMs]', 'k', 'LineWidth', 1), ylim([-0.3 1])
xticklabels({'O-O','NO-NO', 'NO-O'}),ylabel('Correlation coefficient'),title('Naive'),
subplot(1,4,2)
bar([nanmean(RpPostOcclMs), nanmean(RpPostFullMs), nanmean(RpPostFullOcclMs)], 'FaceColor', 'w', 'EdgeColor', 'k', 'FaceAlpha', 0.6, 'LineWidth', 2)
hold on
plot([RpPostOcclMs, RpPostFullMs, RpPostFullOcclMs]', 'k', 'LineWidth', 1), ylim([-0.3 1]),xticklabels({'O-O','NO-NO', 'NO-O'}),title('Expert'),
subplot(1,4,3)
bar([nanmean(RpTaskOcclMs), nanmean(RpTaskFullMs), nanmean(RpTaskFullOcclMs)], 'FaceColor', 'w', 'EdgeColor', 'k', 'FaceAlpha', 0.6, 'LineWidth', 2)
hold on
plot([RpTaskOcclMs, RpTaskFullMs, RpTaskFullOcclMs]', 'k', 'LineWidth', 1), ylim([-0.3 1]),xticklabels({'O-O','NO-NO', 'NO-O'}),title('Task'),

if save_fig
    func_save_fig('L23_corrsPerMouse2')
    func_save_fig('L5_corrsPerMouse2')
    func_save_fig('L23_FullOcclTaskCorrs')
end

% for novel we don't have task
% figure('Position',[305         279        1419         395])
% subplot(1,4,1)
% bar([nanmean(RpPreFullMs), nanmean(RpPostFullMs)], 'FaceColor', 'w', 'EdgeColor', 'k', 'FaceAlpha', 0.6, 'LineWidth', 2)
% hold on
% plot([RpPreFullMs, RpPostFullMs]', 'k', 'LineWidth', 1), ylim([-0.4 0.1])
% xticklabels({'Naive','Expert'}),ylabel('Correlation coefficient'),title('NO-NO'),
% 
% if save_fig
%     func_save_fig('L23_corrsPerMouse2')
%     func_save_fig('L5_FullprepostCorrsPerMouse')
% end
% 
% [~,p] = ttest(RpPreFullMs, RpPostFullMs)
% 

data = [RpPreOcclMs, RpPreFullMs, RpPreFullOcclMs];
% data = [RpPreFullMs, RpPreOcclMs, RpPreFullOcclMs];
[p,tbl,stats] = anova1(data);
multcompare(stats)

data = [RpPostOcclMs, RpPostFullMs, RpPostFullOcclMs];
% data = [RpPostFullMs, RpPostOcclMs, RpPostFullOcclMs];
[p,tbl,stats] = anova1(data);
multcompare(stats)
% 
data = [RpTaskOcclMs, RpTaskFullMs, RpTaskFullOcclMs];
% data = [RpTaskFullMs, RpTaskOcclMs, RpTaskFullOcclMs];
[p,tbl,stats] = anova1(data);
multcompare(stats)
% 

data = [RpPreFullMs, RpPostFullMs, RpTaskFullMs];
[p,tbl,stats] = anova1(data);
multcompare(stats)

%% rfdist vs occl response

rfInclPrePop = cat(1, datastructPre(:).rfIncl);
rfInclPostPop = cat(1, datastructPost(:).rfIncl);
rfOnDistPrePop = cat(1, datastructPre(:).rfOnDist);
rfOffDistPrePop = cat(1, datastructPre(:).rfOffDist);
rfOnDistPostPop = cat(1, datastructPost(:).rfOnDist);
rfOffDistPostPop = cat(1, datastructPost(:).rfOffDist);
onCritPrePop = cat(1, datastructPre(:).onCrit);
offCritPrePop = cat(1, datastructPre(:).offCrit);
onCritPostPop = cat(1, datastructPost(:).onCrit);
offCritPostPop = cat(1, datastructPost(:).offCrit);

rfOnDistPrePop = rfOnDistPrePop(rfInclPrePop);
rfOffDistPrePop = rfOffDistPrePop(rfInclPrePop);
rfOnDistPostPop = rfOnDistPostPop(rfInclPostPop);
rfOffDistPostPop = rfOffDistPostPop(rfInclPostPop);

fwhmOnPre = [];
fwhmOffPre = [];
aziOnPre = [];
aziOffPre = [];
eleOnPre = [];
eleOffPre = [];
fwhmOnPost = [];
fwhmOffPost = [];
aziOnPost = [];
aziOffPost = [];
eleOnPost = [];
eleOffPost = [];

for i = 1:nfiles
    fwhmOnPre = cat(2, fwhmOnPre, [datastructPre(i).info.rois(:).onFWHM]);
    fwhmOffPre = cat(2, fwhmOffPre, [datastructPre(i).info.rois(:).offFWHM]);
    val = [datastructPre(i).info.rois(:).azi];
    aziOnPre = cat(2,aziOnPre,val(1:2:end));
    aziOffPre = cat(2,aziOffPre,val(2:2:end));
    val = [datastructPre(i).info.rois(:).ele];
    eleOnPre = cat(2, eleOnPre, val(1:2:end));
    eleOffPre = cat(2, eleOffPre, val(2:2:end));

    fwhmOnPost = cat(2, fwhmOnPost, [datastructPost(i).info.rois(:).onFWHM]);
    fwhmOffPost = cat(2, fwhmOffPost, [datastructPost(i).info.rois(:).offFWHM]);
    val = [datastructPost(i).info.rois(:).azi];
    aziOnPost = cat(2,aziOnPost,val(1:2:end));
    aziOffPost = cat(2,aziOffPost,val(2:2:end));
    val = [datastructPost(i).info.rois(:).ele];
    eleOnPost = cat(2, eleOnPost, val(1:2:end));
    eleOffPost = cat(2, eleOffPost, val(2:2:end));
end

figure
subplot(2,2,1)
scatter(rfOnDistPrePop, scatOcclFamPopPre, sz, cPre, 'filled'); ylabel('Occl res'), xlabel('Dist to edge'), title('ON pre'),xlim([0 50]),
subplot(2,2,2)
scatter(rfOffDistPrePop, scatOcclFamPopPre, sz, cPre, 'filled'); ylabel('Occl res'), xlabel('Dist to edge'), title('OFF pre'),xlim([0 50]),
subplot(2,2,3)
scatter(rfOnDistPostPop, scatOcclFamPopPost, sz, cPre, 'filled'); ylabel('Occl res'), xlabel('Dist to edge'), title('ON post'),xlim([0 50]),
subplot(2,2,4)
scatter(rfOffDistPostPop, scatOcclFamPopPost, sz, cPre, 'filled'); ylabel('Occl res'), xlabel('Dist to edge'), title('OFF post'),xlim([0 50]),

% choose ON (1) or OFF (2) based on crit values, take ON if both are okay
rfValPre = zeros(length(onCritPrePop),1);
for i = 1:length(onCritPrePop)
    if onCritPrePop(i) && offCritPrePop(i)
        rfValPre(i) = 1;
    elseif onCritPrePop(i)
        rfValPre(i) = 1;
    elseif offCritPrePop(i)
        rfValPre(i) = 2;
    end
end
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
sz = 10;
figure('Position', [323         281        1040         585])
% both ON and OFF RFs for all neurons, so each neuron gets 2 dots
subplot(2,2,1)
scatter(aziOnPre, eleOnPre, sz, 'filled', 'r'), xlim([-60 60]), ylim([-45 45]), hold on
scatter(aziOffPre, eleOffPre, sz, 'filled', 'b'), xlim([-60 60]), ylim([-45 45])
% Only the good RFs, either ON or OFF depending on the neuron
subplot(2,2,2)
scatter(aziOnPre(rfValPre==1), eleOnPre(rfValPre==1), sz, 'filled', 'r'), xlim([-60 60]), ylim([-45 45]), hold on
scatter(aziOffPre(rfValPre==2), eleOffPre(rfValPre==2), sz, 'filled', 'b'), xlim([-60 60]), ylim([-45 45])
subplot(2,2,3)
scatter(aziOnPost, eleOnPost, sz, 'filled', 'r'), xlim([-60 60]), ylim([-45 45]), hold on
scatter(aziOffPost, eleOffPost, sz, 'filled', 'b'), xlim([-60 60]), ylim([-45 45])
% Only the good RFs, either ON or OFF depending on the neuron
subplot(2,2,4)
scatter(aziOnPost(rfValPost==1), eleOnPost(rfValPost==1), sz, 'filled', 'r'), xlim([-60 60]), ylim([-45 45]), hold on
scatter(aziOffPost(rfValPost==2), eleOffPost(rfValPost==2), sz, 'filled', 'b'), xlim([-60 60]), ylim([-45 45])

if save_fig
    func_save_fig('L23_RFselection')
    func_save_fig('L5_RFselection')
end

% plot RFs before and after exclusion pre and post training
sz = 9;
figure('Position', [323         281        1040         585])
% both ON and OFF RFs for all neurons, so each neuron gets 2 dots
subplot(2,2,1)
scatter(aziOnPre(rfValPre==0), eleOnPre(rfValPre==0), sz, 'filled', 'k'), xlim([-60 60]), ylim([-45 45]), hold on
scatter(aziOffPre(rfValPre==0), eleOffPre(rfValPre==0), sz, 'filled', 'k'), xlim([-60 60]), ylim([-45 45])
subplot(2,2,2)
scatter(aziOnPre(rfValPre==1), eleOnPre(rfValPre==1), sz, 'filled', 'g'), xlim([-60 60]), ylim([-45 45]), hold on
scatter(aziOffPre(rfValPre==2), eleOffPre(rfValPre==2), sz, 'filled', 'g'), xlim([-60 60]), ylim([-45 45])

if save_fig
    func_save_fig('L23_RFselection')
    func_save_fig('L5_RFselection')
end

% plot correlation between distance to Occluder edge and occl responses
rfValPostIncl = rfValPost(rfInclPostPop);
aziOnPostIncl = aziOnPost(rfInclPostPop);
eleOnPostIncl = eleOnPost(rfInclPostPop);
aziOffPostIncl = aziOffPost(rfInclPostPop);
eleOffPostIncl = eleOffPost(rfInclPostPop);
aziData = zeros(length(aziOnPostIncl),1);
eleData = zeros(length(eleOnPostIncl),1);
aziData(rfValPostIncl==1) = aziOnPostIncl(rfValPostIncl==1); % 1 equals ON, 2 equals OFF
aziData(rfValPostIncl==2) = aziOffPostIncl(rfValPostIncl==2);
eleData(rfValPostIncl==1) = eleOnPostIncl(rfValPostIncl==1);
eleData(rfValPostIncl==2) = eleOffPostIncl(rfValPostIncl==2);
rfDistPostData = zeros(length(rfOnDistPostPop),1);
rfDistPostData(rfValPostIncl==1) = rfOnDistPostPop(rfValPostIncl==1); % 1 equals ON, 2 equals OFF
rfDistPostData(rfValPostIncl==2) = rfOffDistPostPop(rfValPostIncl==2);

% figure
% scatter(rfDistPostData, scatOcclFamPopPost, sz, cPre, 'filled'); ylabel('Occl res'), xlabel('Dist to edge') 
% title('RFdist vs Occl res'),
% [rDistVsOccl, pDistVsOccl] = corrcoef(rfDistPreData, scatOcclFamPopPre);
[rDistVsOccl, pDistVsOccl] = corrcoef(rfDistPostData, scatOcclFamPopPost);

figure('Position', [96   226   560   420])
fits = polyfit(rfDistPostData, scatOcclFamPopPost,1);
fit1 = polyval(fits,rfDistPostData);
vr = scatter(rfDistPostData, scatOcclFamPopPost, sz, 'k', 'filled'); hold on
plot(rfDistPostData, fit1, 'r', 'LineWidth', 1.5)
ylabel('Occl res'), xlabel('Dist to edge'), title('RFdist vs Occl res')
text(5,2,sprintf('r=%.3f',rDistVsOccl(2))), text(5,1.5,sprintf('p=%.3f', pDistVsOccl(2)))


if save_fig
    func_save_fig('L23_RFdistVsOcclRes')
    func_save_fig('L5_RFdistVsOcclRes')
end


%% rfdist vs occl response for a subpopulation of neurons

rfInclPrePop = cat(1, datastructPre(:).rfIncl);
rfInclPostPop = cat(1, datastructPost(:).rfIncl);
rfOnDistPrePop = cat(1, datastructPre(:).rfOnDist);
rfOffDistPrePop = cat(1, datastructPre(:).rfOffDist);
rfOnDistPostPop = cat(1, datastructPost(:).rfOnDist);
rfOffDistPostPop = cat(1, datastructPost(:).rfOffDist);
onCritPrePop = cat(1, datastructPre(:).onCrit);
offCritPrePop = cat(1, datastructPre(:).offCrit);
onCritPostPop = cat(1, datastructPost(:).onCrit);
offCritPostPop = cat(1, datastructPost(:).offCrit);

rfOnDistPrePop = rfOnDistPrePop(rfInclPrePop);
rfOffDistPrePop = rfOffDistPrePop(rfInclPrePop);
rfOnDistPostPop = rfOnDistPostPop(rfInclPostPop);
rfOffDistPostPop = rfOffDistPostPop(rfInclPostPop);

fwhmOnPre = [];
fwhmOffPre = [];
aziOnPre = [];
aziOffPre = [];
eleOnPre = [];
eleOffPre = [];
fwhmOnPost = [];
fwhmOffPost = [];
aziOnPost = [];
aziOffPost = [];
eleOnPost = [];
eleOffPost = [];

for i = 1:nfiles
    fwhmOnPre = cat(2, fwhmOnPre, [datastructPre(i).info.rois(:).onFWHM]);
    fwhmOffPre = cat(2, fwhmOffPre, [datastructPre(i).info.rois(:).offFWHM]);
    val = [datastructPre(i).info.rois(:).azi];
    aziOnPre = cat(2,aziOnPre,val(1:2:end));
    aziOffPre = cat(2,aziOffPre,val(2:2:end));
    val = [datastructPre(i).info.rois(:).ele];
    eleOnPre = cat(2, eleOnPre, val(1:2:end));
    eleOffPre = cat(2, eleOffPre, val(2:2:end));

    fwhmOnPost = cat(2, fwhmOnPost, [datastructPost(i).info.rois(:).onFWHM]);
    fwhmOffPost = cat(2, fwhmOffPost, [datastructPost(i).info.rois(:).offFWHM]);
    val = [datastructPost(i).info.rois(:).azi];
    aziOnPost = cat(2,aziOnPost,val(1:2:end));
    aziOffPost = cat(2,aziOffPost,val(2:2:end));
    val = [datastructPost(i).info.rois(:).ele];
    eleOnPost = cat(2, eleOnPost, val(1:2:end));
    eleOffPost = cat(2, eleOffPost, val(2:2:end));
end

% choose ON (1) or OFF (2) based on crit values, take ON if both are okay
rfValPre = zeros(length(onCritPrePop),1);
for i = 1:length(onCritPrePop)
    if onCritPrePop(i) && offCritPrePop(i)
        rfValPre(i) = 1;
    elseif onCritPrePop(i)
        rfValPre(i) = 1;
    elseif offCritPrePop(i)
        rfValPre(i) = 2;
    end
end
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
sz = 9;

scatter(aziOnPost(rfValPost==1), eleOnPost(rfValPost==1), sz, 'filled', 'g'), xlim([-60 60]), ylim([-45 45]), hold on
scatter(aziOffPost(rfValPost==2), eleOffPost(rfValPost==2), sz, 'filled', 'g'), xlim([-60 60]), ylim([-45 45])


aziOnPostAll = aziOnPost(rfValPost==1 & rfInclPostPop);
eleOnPostAll = eleOnPost(rfValPost==1 & rfInclPostPop);

aziOffPostAll = aziOnPost(rfValPost==1 & rfInclPostPop);
eleOffPostAll = eleOnPost(rfValPost==1 & rfInclPostPop);



if save_fig
    func_save_fig('L23_RFselection')
    func_save_fig('L5_RFselection')
end

% plot correlation between distance to Occluder edge and occl responses
rfValPostIncl = rfValPost(rfInclPostPop);
aziOnPostIncl = aziOnPost(rfInclPostPop);
eleOnPostIncl = eleOnPost(rfInclPostPop);
aziOffPostIncl = aziOffPost(rfInclPostPop);
eleOffPostIncl = eleOffPost(rfInclPostPop);
aziData = zeros(length(aziOnPostIncl),1);
eleData = zeros(length(eleOnPostIncl),1);
aziData(rfValPostIncl==1) = aziOnPostIncl(rfValPostIncl==1); % 1 equals ON, 2 equals OFF
aziData(rfValPostIncl==2) = aziOffPostIncl(rfValPostIncl==2);
eleData(rfValPostIncl==1) = eleOnPostIncl(rfValPostIncl==1);
eleData(rfValPostIncl==2) = eleOffPostIncl(rfValPostIncl==2);
rfDistPostData = zeros(length(rfOnDistPostPop),1);
rfDistPostData(rfValPostIncl==1) = rfOnDistPostPop(rfValPostIncl==1); % 1 equals ON, 2 equals OFF
rfDistPostData(rfValPostIncl==2) = rfOffDistPostPop(rfValPostIncl==2);

%%%% plot only neurons that had postExcl responses
postExcl = scatFullFamPopPost>maxResThres & scatOcclFamPopPost>maxResThres;

figure('Position', [96         325        1100         321])
subplot(1,2,1)
scatter(rfDistPostData, scatOcclFamPopPost, sz, 'k', 'filled'); hold on
scatter(rfDistPostData(postExcl), scatOcclFamPopPost(postExcl), sz, 'r', 'filled'); hold on
ylabel('Occl res'), xlabel('Dist to edge'), title('RFdist vs Occl res')

subplot(1,2,2)
hold on
inclIdx = find(rfInclPostPop);  % 884 x 1 indices into the full 1807
exclIdx = inclIdx(postExcl);  % Indices into full 1807 for excluded neurons
% Logical masks for full data (1807 x 1)
isOn  = rfValPost == 1;
isOff = rfValPost == 2;

% Exclusion masks (within full size)
isExclOn  = false(size(rfValPost));
isExclOff = false(size(rfValPost));
isExclOn(exclIdx)  = rfValPost(exclIdx) == 1;
isExclOff(exclIdx) = rfValPost(exclIdx) == 2;

% Plot ALL valid ON/OFF neurons in green
scatter(aziOnPost(isOn), eleOnPost(isOn), sz, 'filled', 'g')
scatter(aziOffPost(isOff), eleOffPost(isOff), sz, 'filled', 'g')

% Overlay EXCLUDED ON/OFF neurons in red
scatter(aziOnPost(isExclOn), eleOnPost(isExclOn), sz, 'filled', 'r')
scatter(aziOffPost(isExclOff), eleOffPost(isExclOff), sz, 'filled', 'r')

xlim([-60 60]); ylim([-45 45])
% legend('All', 'Excluded');


if save_fig
    func_save_fig('L23_RFdistExclNeurons')
end

sz = 15;

figure
scatter(scatFullFamPopPostCut(~postExcl), scatOcclFamPopPostCut(~postExcl), sz, 'g', 'filled'); 
hold on
scatter(scatFullFamPopPostCut(postExcl), scatOcclFamPopPostCut(postExcl), sz, 'r', 'filled');
xlim([-1 3]), ylim([-1 3])
refline(1), xlabel('Full post'), ylabel('Occl post'), 

if save_fig
    func_save_fig('L23_ScatterPostExclNeurons')
end


%% RF size differences between NO and O responders
% Parameters
zThresh = 0.2;

% Filter to only included neurons
rfVal = rfValPost(rfInclPostPop);             % 884 x 1
fwhmOn = fwhmOnPost(rfInclPostPop);           % 884 x 1
fwhmOff = fwhmOffPost(rfInclPostPop);         % 884 x 1
scatFull = scatFullFamPopPost;                % 884 x 1
scatOccl = scatOcclFamPopPost;                % 884 x 1

% Build mouse IDs for all neurons
mouseIDPost = [];
for i = 1:nfiles
    mouseIDPost = [mouseIDPost; repmat(i, length(datastructPost(i).scatFull), 1)];
end
mouseID = mouseIDPost;         % 884 x 1

% Masks to assign neurons to unique conditions (prefer ON if both pass)
onMaskFull  = (rfVal == 1) & ((scatFull > zThresh) & (scatOccl < zThresh))';
onMaskOccl  = (rfVal == 1) & ((scatOccl > zThresh) & (scatFull < zThresh))';
offMaskFull = (rfVal == 2) & ((scatFull > zThresh) & (scatOccl < zThresh))';
offMaskOccl = (rfVal == 2) & ((scatOccl > zThresh) & (scatFull < zThresh))';

% Extract RF sizes
onFull  = fwhmOn(onMaskFull);
onOccl  = fwhmOn(onMaskOccl);
offFull = fwhmOff(offMaskFull);
offOccl = fwhmOff(offMaskOccl);

% Extract corresponding mouse IDs
mouseID_onFull  = mouseID(onMaskFull);
mouseID_onOccl  = mouseID(onMaskOccl);
mouseID_offFull = mouseID(offMaskFull);
mouseID_offOccl = mouseID(offMaskOccl);

% Combine into one dataset
data = [onFull offFull onOccl offOccl]';
mouseID_all = categorical([
    mouseID_onFull;
    mouseID_offFull;
    mouseID_onOccl;
    mouseID_offOccl
]);

% Condition: 1 = Full, 2 = Occl
condition = categorical([
    ones(length(onFull) + length(offFull), 1);         % Full
    ones(length(onOccl) + length(offOccl), 1) * 2       % Occl
]);

% Build table and fit LMEM
statTbl = table(data, mouseID_all, condition);
lme = fitlme(statTbl, 'data ~ condition + (1|mouseID_all)', ...
    'CheckHessian', 1, 'FitMethod', 'REML', 'StartMethod', 'random');

% Output stats
stats = anova(lme, 'DFMethod', 'Satterthwaite');

fullRF = [onFull offFull];
occlRF = [onOccl offOccl];

figure('Position', [680   566   257   412])
scatter([1 2],[nanmean(fullRF) nanmean(occlRF)], 35, 'k', 'LineWidth', 2), hold on
er = errorbar([1 2],[nanmean(fullRF) nanmean(occlRF)], ...
    [calcSem(fullRF) calcSem(occlRF)] ...
    ,[calcSem(fullRF) calcSem(occlRF)]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 3]), 
if nfiles == 6, ylim([10 20]), elseif nfiles == 5, ylim([5 15]), end

xticks([1 2]), xticklabels({'NO cells','O cells'})


if save_fig
    func_save_fig('L23_NOvsOcellsRFs')
end



%% DECODING
rng(1) % for reproducibility

% close all

minThresh = false; % minimum trheshold for response to include in decoding?
minResThres = 0.5;

maxThresh = false; % both should be lower
maxResThres = 0.3; % than this threshold

bothThresh = false;
bothResThres = 0.3;


% load active data, note that this is only relevant if you decode from
% familiar images. If you also want novel images you get errors.
if nfiles == 6
    load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\matDataDecodingActive.mat', 'matDataDecoding') % no need to load in trialtypes
elseif nfiles == 5
    load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\matDataDecodingActive.mat', 'matDataDecoding')
end

doDecoding = 1;
doPlotting = 1;
nReps = 200;
nBoots = 1000;
famIdx = [1 2 4 5];
% famIdx = [2 4];
novIdx = [3 6];
trialTypes = matTrialTypesIncl;
gammOpt = 0; % 0 for paper

% g = 6;
% matDataPopPre = datastructPre(g).matData;
% matDataPopPost = datastructPost(g).matData;
% matDataPopTask = matDataDecoding(g).matData;


matDataPopPre = datastructPre(1).matData;
matDataPopPost = datastructPost(1).matData;
matDataPopTask = matDataDecoding(1).matData;

for i = 2:nfiles
    matDataPopPre = cat(2, matDataPopPre, datastructPre(i).matData);
    matDataPopPost = cat(2, matDataPopPost, datastructPost(i).matData);
    matDataPopTask = cat(2, matDataPopTask, matDataDecoding(i).matData);
end

% if minThresh % either full or occl should be higher than thresh
%     preIncl = scatFullFamPopPre>minResThres | scatOcclFamPopPre>minResThres;
%     matDataPopPre = matDataPopPre(:,preIncl);
%     postIncl = scatFullFamPopPost>minResThres | scatOcclFamPopPost>minResThres;
%     matDataPopPost = matDataPopPost(:,postIncl);
%     taskIncl = scatFullPop>minResThres | scatOcclPop>minResThres;
%     matDataPopTask = matDataPopTask(:,taskIncl);
% end

if maxThresh % cells that respond > thesh for full and occl should be removed
    preExcl = scatFullFamPopPre>maxResThres & scatOcclFamPopPre>maxResThres;
    matDataPopPre(:,preExcl) = [];
    postExcl = scatFullFamPopPost>maxResThres & scatOcclFamPopPost>maxResThres;
    matDataPopPost(:,postExcl) = [];
    taskExcl = scatFullPop>maxResThres & scatOcclPop>maxResThres;
    matDataPopTask(:,taskExcl) = [];
end

% if minBothThresh % both should be higher than thresh
%     preIncl = scatFullFamPopPre>bothResThres & scatOcclFamPopPre>bothResThres;
%     matDataPopPre = matDataPopPre(:,preIncl);
%     postIncl = scatFullFamPopPost>bothResThres & scatOcclFamPopPost>bothResThres;
%     matDataPopPost = matDataPopPost(:,postIncl);
%     taskIncl = scatFullPop>bothResThres & scatOcclPop>bothResThres;
%     matDataPopTask = matDataPopTask(:,taskIncl);
% end


idx = famIdx; % which images to include? famidx/novidx?
% idx = novIdx; % which images to include? famidx/novidx?
% idx = 1:6; % which images to include? famidx/novidx?
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
    [pFFPre, pOOPre, pFOPre, pOFPre,pFFpermPre, pOOpermPre,pFOpermPre, pOFpermPre,cMatFFPre, cMatOOPre, cMatFOPre, cMatOFPre,...
        cMatFFpermPre, cMatOOpermPre, cMatFOpermPre, cMatOFpermPre, dPredictFullPre, dPredictOcclPre]...
        = doMuckliDecodingLDAblock2(matDataPopPre, trialTypes, trainFrac, nReps, nBoots, gammOpt);
    % Post
    [pFFPost, pOOPost, pFOPost, pOFPost,pFFpermPost, pOOpermPost,pFOpermPost, pOFpermPost, cMatFFPost, cMatOOPost, cMatFOPost, cMatOFPost,...
        cMatFFpermPost, cMatOOpermPost, cMatFOpermPost, cMatOFpermPost,dPredictFullPost, dPredictOcclPost]...
        = doMuckliDecodingLDAblock2(matDataPopPost, trialTypes, trainFrac, nReps, nBoots, gammOpt);
    % Task
    [pFFTask, pOOTask, pFOTask, pOFTask,pFFpermTask, pOOpermTask,pFOpermTask, pOFpermTask,cMatFFTask, cMatOOTask, cMatFOTask, cMatOFTask,...
        cMatFFperm, cMatOOperm, cMatFOperm, cMatOFperm, dPredictFullTask, dPredictOcclTask]...
        = doMuckliDecodingLDAblock2(matDataPopTask, trialTypes, trainFrac, nReps, nBoots, gammOpt);
end

if doPlotting
    % PLOTTING
% permutation tests - not chronically matched (for chronic matching you need to pair it, so take
% paired difference between each of the 500 points pre vs post, average
% those for each run, take those 1000 runs as your permutation data.

realDiffOO = mean(pOOPost)-mean(pOOPre); % Compute the observed difference in means
realDiffFO = mean(pFOPost)-mean(pFOPre); % Compute the observed difference in means
realDiffOF = mean(pOFPost)-mean(pOFPre); % Compute the observed difference in means
all_data_OO = [pOOPost; pOOPre]; % Concatenate the data
all_data_FO = [pFOPost; pFOPre]; % Concatenate the data
all_data_OF = [pOFPost; pOFPre]; % Concatenate the data

nPerms = 1000; % Number of permutations

% Initialize arrays to store permuted differences
permuted_diffs_OO = zeros(nPerms, 1);
permuted_diffs_FO = zeros(nPerms, 1);
permuted_diffs_OF = zeros(nPerms, 1);

% Permutation test
for i = 1:nPerms
    % Randomly permute the data & compute the permuted difference in means
    permuted_data = all_data_OO(randperm(length(all_data_OO)));
    permuted_diffs_OO(i) = mean(permuted_data(1:nReps)) - mean(permuted_data(nReps+1:end));

    permuted_data = all_data_FO(randperm(length(all_data_FO)));
    permuted_diffs_FO(i) = mean(permuted_data(1:nReps)) - mean(permuted_data(nReps+1:end));

    permuted_data = all_data_OF(randperm(length(all_data_OF)));
    permuted_diffs_OF(i) = mean(permuted_data(1:nReps)) - mean(permuted_data(nReps+1:end));
end
% plot results permutation test
figure('Position', [353    86   663   845])
subplot(3,2,1)
histogram(pOOPre, length(unique(pOOPre)), 'Normalization', 'Probability', 'FaceColor', col1, 'EdgeColor', col1, 'LineWidth', 2); hold on
histogram(pOOPost, length(unique(pOOPost)), 'Normalization', 'Probability', 'FaceColor', col2, 'EdgeColor', col2, 'LineWidth', 2);
xlim([0 100]), xline(mean(pOOPre), 'Color', col1, 'LineWidth', 2), xline(mean(pOOPost), 'Color', col2, 'LineWidth', 2)
xlabel('Decoding accuracy'), ylabel('Fraction of decoding runs'), 
subplot(3,2,2)
histogram(permuted_diffs_OO, 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
xline(realDiffOO, 'r', 'LineWidth', 2); pval = paretoEst(permuted_diffs_OO/100, realDiffOO/100);
title(sprintf('p = %.3f', pval)), xlabel('Decoding difference (post-pre)'), ylabel('Relative count')
legend({'Permutation difference', 'Real difference'}); legend boxoff; 

subplot(3,2,3)
histogram(pFOPre, length(unique(pFOPre)), 'Normalization', 'Probability', 'FaceColor', col1, 'EdgeColor', col1, 'LineWidth', 2); hold on
histogram(pFOPost, length(unique(pFOPost)), 'Normalization', 'Probability', 'FaceColor', col2, 'EdgeColor', col2, 'LineWidth', 2);
xlim([0 100]), xline(mean(pFOPre), 'Color', col1, 'LineWidth', 2), xline(mean(pFOPost), 'Color', col2, 'LineWidth', 2)
xlabel('Decoding accuracy'), ylabel('Fraction of decoding runs'), 
subplot(3,2,4)
histogram(permuted_diffs_FO, 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
xline(realDiffFO, 'r', 'LineWidth', 2); pval = paretoEst(permuted_diffs_FO/100, realDiffFO/100);
title(sprintf('p = %.3f', pval)), xlabel('Decoding difference (post-pre)'), ylabel('Relative count')
legend({'Permutation difference', 'Real difference'}); legend boxoff; 

subplot(3,2,5)
histogram(pOFPre, length(unique(pOFPre)), 'Normalization', 'Probability', 'FaceColor', col1, 'EdgeColor', col1, 'LineWidth', 2); hold on
histogram(pOFPost, length(unique(pOFPost)), 'Normalization', 'Probability', 'FaceColor', col2, 'EdgeColor', col2, 'LineWidth', 2);
xlim([0 100]), xline(mean(pOFPre), 'Color', col1, 'LineWidth', 2), xline(mean(pOFPost), 'Color', col2, 'LineWidth', 2)
xlabel('Decoding accuracy'), ylabel('Fraction of decoding runs'), 
subplot(3,2,6)
histogram(permuted_diffs_OF, 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
xline(realDiffOF, 'r', 'LineWidth', 2); pval = paretoEst(permuted_diffs_OF/100, realDiffOF/100);
title(sprintf('p = %.3f', pval)), xlabel('Decoding difference (post-pre)'), ylabel('Relative count')
legend({'Permutation difference', 'Real difference'}); legend boxoff; 

    if save_fig
        func_save_fig('L23_decodingPermTestsNaiveExpert_6imgs')
%         func_save_fig('L5_decodingBars_4imgs')
    end


% statistics on pOOpre vs pOOpost, this is more stringent but better
pvalOO = paretoEst(pOOPre/100, mean(pOOPost/100));
pvalFO = paretoEst(pFOPre/100, mean(pFOPost/100));
pvalOF = paretoEst(pOFPre/100, mean(pOFPost/100));


%     % weights from decoding
figure, histogram(mean(dPredictOcclPre,2), 'Normalization', 'Probability'), hold on, histogram(mean(dPredictOcclPost,2), 'Normalization', 'Probability'), xlabel('Occl weight'), ylabel('frac of cells')
figure, histogram(mean(dPredictFullPre,2), 'Normalization', 'Probability'), hold on, histogram(mean(dPredictFullPost,2), 'Normalization', 'Probability'), xlabel('Full weight'), ylabel('frac of cells')

% 
%     if save_fig
%         func_save_fig('L23_decodingweights')
% %         func_save_fig('L5_decodingBars_4imgs')
%     end

    pFFPreMn = mean(pFFPre);pFFPreSEM = std(pFFPre);
    pFFPostMn = mean(pFFPost);pFFPostSEM = std(pFFPost);
    pFFTaskMn = mean(pFFTask);pFFTaskSEM = std(pFFTask);

    pOOPreMn = mean(pOOPre);pOOPreSEM = std(pOOPre);
    pOOPostMn = mean(pOOPost);pOOPostSEM = std(pOOPost);
    pOOTaskMn = mean(pOOTask);pOOTaskSEM = std(pOOTask);

    pFOPreMn = mean(pFOPre);pFOPreSEM = std(pFOPre);
    pFOPostMn = mean(pFOPost);pFOPostSEM = std(pFOPost);
    pFOTaskMn = mean(pFOTask);pFOTaskSEM = std(pFOTask);

    pOFPreMn = mean(pOFPre);pOFPreSEM = std(pOFPre);
    pOFPostMn = mean(pOFPost);pOFPostSEM = std(pOFPost);
    pOFTaskMn = mean(pOFTask);pOFTaskSEM = std(pOFTask);
    
%     figure('Position', [490   146   754   766])
%     bar([1 2 4 5 7 8 10 11],[pFFPreMn pFFPostMn pOOPreMn pOOPostMn pFOPreMn pFOPostMn pOFPreMn pOFPostMn], 'FaceColor', 'w', 'EdgeColor', 'k', 'FaceAlpha', 0.6, 'LineWidth', 2)
%     hold on
%     er = errorbar([1 2 4 5 7 8 10 11],[pFFPreMn pFFPostMn pOOPreMn pOOPostMn pFOPreMn pFOPostMn pOFPreMn pOFPostMn]...
%         ,[0 0 0 0 0 0 0 0],[pFFPreSEM pFFPostSEM pOOPreSEM pOOPostSEM pFOPreSEM pFOPostSEM pOFPreSEM pOFPostSEM]);
%     er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2;
%     xticks([1 2 4 5 7 8 10 11]), xticklabels({'F-F', 'F-F', 'O-O', 'O-O', 'F-O', 'F-O', 'O-F', 'O-F'}), ylabel('MM response post - pre'),
%     title('Decoding accuracy')
%     ylim([0 120])
%     yline(100/length(idx))

    figure('Position', [301   129   947   695])
    bar([1 2 3],[pFFPreMn pFFPostMn pFFTaskMn], 'FaceColor', 'w', 'EdgeColor', 'k', 'FaceAlpha', 0.6, 'LineWidth', 2), hold on
    er = errorbar([1 2 3],[pFFPreMn pFFPostMn pFFTaskMn],[pFFPreSEM pFFPostSEM pFFTaskSEM],[pFFPreSEM pFFPostSEM pFFTaskSEM]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2;
    
    bar([5 6 7],[pOOPreMn pOOPostMn pOOTaskMn], 'FaceColor', 'w', 'EdgeColor', 'k', 'FaceAlpha', 0.6, 'LineWidth', 2), hold on
    er = errorbar([5 6 7],[pOOPreMn pOOPostMn pOOTaskMn],[pOOPreSEM pOOPostSEM pOOTaskSEM],[pOOPreSEM pOOPostSEM pOOTaskSEM]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2;
    
    bar([9 10 11],[pFOPreMn pFOPostMn pFOTaskMn], 'FaceColor', 'w', 'EdgeColor', 'k', 'FaceAlpha', 0.6, 'LineWidth', 2), hold on
    er = errorbar([9 10 11],[pFOPreMn pFOPostMn pFOTaskMn],[pFOPreSEM pFOPostSEM pFOTaskSEM],[pFOPreSEM pFOPostSEM pFOTaskSEM]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2;

    bar([13 14 15],[pOFPreMn pOFPostMn pOFTaskMn], 'FaceColor', 'w', 'EdgeColor', 'k', 'FaceAlpha', 0.6, 'LineWidth', 2), hold on
    er = errorbar([13 14 15],[pOFPreMn pOFPostMn pOFTaskMn],[pOFPreSEM pOFPostSEM pOFTaskSEM],[pOFPreSEM pOFPostSEM pOFTaskSEM]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2;

    xticks([2 6 10 14]), xticklabels({'NO-NO', 'O-O', 'NO-O', 'O-NO'}), ylabel('Decoding accuracy'),
    ylim([0 120])
    yline(100/length(idx))

    if save_fig
        func_save_fig('L23_decodingBars_4imgs')
        func_save_fig('L23_decodingBars_6imgs')
        func_save_fig('L5_decodingBars_4imgs')
    end

    figure('Position', [301   367   397   457])
    bar([1 2 3 4],[pFFTaskMn pOOTaskMn pFOTaskMn pOFTaskMn], 'FaceColor', 'w', 'EdgeColor', 'k', 'FaceAlpha', 0.6, 'LineWidth', 2), hold on
    er = errorbar([1 2 3 4],[pFFTaskMn pOOTaskMn pFOTaskMn pOFTaskMn],[pFFTaskSEM pOOTaskSEM pFOTaskSEM pOFTaskSEM],[pFFTaskSEM pOOTaskSEM pFOTaskSEM pOFTaskSEM]); er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2;
    ylabel('Decoding accuracy'), xticklabels({'NO-NO','O-O','NO-O','O-NO'}), yline(100/length(idx)), 

    if save_fig
        func_save_fig('L23_decodingBarsTask_4imgs')
        func_save_fig('L5_decodingBarsTask_4imgs')
    end

    % Permutation plots
    figure('Position', [301   129   877   695])
    hold on
    subplot(4,3,1)
    histogram(pFFpermPre, length(unique(pFFpermPre)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pFFPreMn, 'r', 'LineWidth', 2);
    pval = paretoEst(pFFpermPre/100, pFFPreMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100]), ylabel('Fraction of runs'), set(gca, 'LineWidth', 2, 'FontSize', 12), box off
    subplot(4,3,2)
    histogram(pFFpermPost, length(unique(pFFpermPost)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pFFPostMn, 'r', 'LineWidth', 2);
    pval = paretoEst(pFFpermPost/100, pFFPostMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100]), ylabel('Fraction of runs'), set(gca, 'LineWidth', 2, 'FontSize', 12), box off
    subplot(4,3,3)
    histogram(pFFpermTask, length(unique(pFFpermTask)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pFFPostMn, 'r', 'LineWidth', 2);
    pval = paretoEst(pFFpermTask/100, pFFTaskMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100]), ylabel('Fraction of runs'), set(gca, 'LineWidth', 2, 'FontSize', 12), box off

    subplot(4,3,4)
    histogram(pOOpermPre, length(unique(pOOpermPre)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pOOPreMn, 'r', 'LineWidth', 2);
    pval = paretoEst(pOOpermPre/100, pOOPreMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100]), ylabel('Fraction of runs'), set(gca, 'LineWidth', 2, 'FontSize', 12), box off
    subplot(4,3,5)
    histogram(pOOpermPost, length(unique(pOOpermPost)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pOOPostMn, 'r', 'LineWidth', 2);
    pval = paretoEst(pOOpermPost/100, pOOPostMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100]), ylabel('Fraction of runs'), set(gca, 'LineWidth', 2, 'FontSize', 12), box off
    subplot(4,3,6)
    histogram(pOOpermTask, length(unique(pOOpermTask)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pOOTaskMn, 'r', 'LineWidth', 2);
    pval = paretoEst(pOOpermTask/100, pOOTaskMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100]), ylabel('Fraction of runs'), set(gca, 'LineWidth', 2, 'FontSize', 12), box off
    
    subplot(4,3,7)
    histogram(pFOpermPre, length(unique(pFOpermPre)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pFOPreMn, 'r', 'LineWidth', 2);
    pval = paretoEst(pFOpermPre/100, pFOPreMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100]), ylabel('Fraction of runs'), set(gca, 'LineWidth', 2, 'FontSize', 12), box off
    subplot(4,3,8)
    histogram(pFOpermPost, length(unique(pFOpermPost)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pFOPostMn, 'r', 'LineWidth', 2);
    pval = paretoEst(pFOpermPost/100, pFOPostMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100]), ylabel('Fraction of runs'), set(gca, 'LineWidth', 2, 'FontSize', 12), box off
    subplot(4,3,9)
    histogram(pFOpermTask, length(unique(pFOpermTask)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pFOTaskMn, 'r', 'LineWidth', 2);
    pval = paretoEst(pFOpermTask/100, pFOTaskMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100]), ylabel('Fraction of runs'), set(gca, 'LineWidth', 2, 'FontSize', 12), box off
    
    subplot(4,3,10)
    histogram(pOFpermPre, length(unique(pOFpermPre)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pOFPreMn, 'r', 'LineWidth', 2);
    pval = paretoEst(pOFpermPre/100, pOFPreMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100]), ylabel('Fraction of runs'), set(gca, 'LineWidth', 2, 'FontSize', 12), box off
    subplot(4,3,11)
    histogram(pOFpermPost, length(unique(pOFpermPost)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pOFPostMn, 'r', 'LineWidth', 2);
    pval = paretoEst(pOFpermPost/100, pOFPostMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100]), ylabel('Fraction of runs'), set(gca, 'LineWidth', 2, 'FontSize', 12), box off
    subplot(4,3,12)
    histogram(pOFpermTask, length(unique(pOFpermTask)), 'Normalization', 'Probability', 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2);
    xline(pOFTaskMn, 'r', 'LineWidth', 2);
    pval = paretoEst(pOFpermTask/100, pOFTaskMn/100);
    title(sprintf('p = %.3f', pval))
    xlim([0 100]), ylabel('Fraction of runs'), set(gca, 'LineWidth', 2, 'FontSize', 12), box off

    if save_fig
        func_save_fig('L23_decodingHists_4imgs')
        func_save_fig('L5_decodingHists_4imgs')
        func_save_fig('L23_decodingHists_6imgs')
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
    colormap gray
    set(s, 'CLim', [0, 100]);  

    if save_fig
        func_save_fig('L23_decodingConfMatsNaive')
        func_save_fig('L5_decodingConfMatsNaive')
    end
    
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
    colormap gray
    set(s, 'CLim', [0, 100]);   

    if save_fig
        func_save_fig('L23_decodingConfMatsExpert')
        func_save_fig('L5_decodingConfMatsExpert')
    end
end


%% an plot results of single cell contributions
figure('Position', [133 202 1575 732])
% better colors
% colorM = cmapL([0 0 0; 0.8 0.8 0.8], 100); % sorting is descending
% colorM = cmapL([0 0 0; [144, 169, 85]/255; [236 243 158]/255], 100); % sorting is descending
% colorM = cmapL([0 0 0; [220, 47, 2]/255; [234, 226, 183]/255], 100); % sorting is descending

% we change the colormap in the end anyway, but you can use this function
% to make a nicer one if needed
colorM = cmapL([0 0 0; [220, 47, 2]/255; [255, 233, 78]/255], 100); % sorting is descending

sz = 10;
s(1) = subplot(3,4,1);
scatCol = mean(ffPre);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPreCut(sortIndex), scatOcclFamPopPreCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), title('NO-NO'), colormap(colorM), colorbar
s(2) = subplot(3,4,2);
scatCol = mean(ooPre);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPreCut(sortIndex), scatOcclFamPopPreCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), title('O-O'), colormap(colorM), colorbar
s(3) = subplot(3,4,3);
scatCol = mean(foPre);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPreCut(sortIndex), scatOcclFamPopPreCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), title('NO-O'), colormap(colorM), colorbar
s(4) = subplot(3,4,4);
scatCol = mean(ofPre);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPreCut(sortIndex), scatOcclFamPopPreCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), title('O-NO'), colormap(colorM), colorbar
s(5) = subplot(3,4,5);
scatCol = mean(ffPost);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPostCut(sortIndex), scatOcclFamPopPostCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar
s(6) = subplot(3,4,6);
scatCol = mean(ooPost);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPostCut(sortIndex), scatOcclFamPopPostCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar
s(7) = subplot(3,4,7);
scatCol = mean(foPost);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPostCut(sortIndex), scatOcclFamPopPostCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar
s(8) = subplot(3,4,8);
scatCol = mean(ofPost);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPostCut(sortIndex), scatOcclFamPopPostCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar
s(9) = subplot(3,4,9);
scatCol = mean(ffTask);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullPopCut(sortIndex), scatOcclPopCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(hot), colorbar
s(10) = subplot(3,4,10);
scatCol = mean(ooTask);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullPopCut(sortIndex), scatOcclPopCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar
s(11) = subplot(3,4,11);
scatCol = mean(foTask);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullPopCut(sortIndex), scatOcclPopCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar
s(12) = subplot(3,4,12);
scatCol = mean(ofTask);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullPopCut(sortIndex), scatOcclPopCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar

hotMap = flipud(hot);
nColors = size(hotMap, 1); % Number of colors in the colormap
cutOffIndex = floor(nColors * 0.3); % Calculate cutoff index to remove the bottom 20%
modifiedHotMap = hotMap(cutOffIndex:end, :); % Keep the upper part of the colormap
colormap(modifiedHotMap); % cut off map

% Adjust subplot properties

for j = 1:length(s)
    if nfiles == 6
        labels = {'-1','','0','','1','','2','','>2.5'};
        s(j).YLim = [-1 3]; s(j).YTick = -1:0.5:3; s(j).YTickLabel = labels;
        s(j).XLim = [-1 3]; s(j).XTick = -1:0.5:3; s(j).XTickLabel = labels;
        s(j).CLim = [25 50]; % Adjust according to your data range
        s(j).XTickLabelRotation = 0;
    else
        labels = {'','0','','1','','2','','>2.5'};
        s(j).YLim = [-0.5 3]; s(j).YTick = -0.5:0.5:3; s(j).YTickLabel = labels;
        s(j).XLim = [-0.5 3]; s(j).XTick = -0.5:0.5:3; s(j).XTickLabel = labels;
        s(j).CLim = [25 50]; % Adjust according to your data range
        s(j).XTickLabelRotation = 0;
    end
end


%% decoding plotting of increase naive vs expert pairs of images nov vs fam

load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\decodingDataNaiveVsExpertIncrease.mat')

figure, bar(1, diffnov), hold on, bar(2, difffam)
hold on, scatter(ones(6,1)+1,difffams)

[h, p] = ttest(difffams, diffnov)
[h, p] = ranksum(difffams, diffnov)

%% decoding increase naive vs expert pairs of images, new, for revisions

if nfiles == 6
    load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\matDataDecodingActive.mat', 'matDataDecoding') % no need to load in trialtypes
elseif nfiles == 5
    load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\matDataDecodingActive.mat', 'matDataDecoding')
end

matDataPopPre = datastructPre(1).matData;
matDataPopPost = datastructPost(1).matData;
matDataPopTask = matDataDecoding(1).matData;

for i = 2:nfiles
    matDataPopPre = cat(2, matDataPopPre, datastructPre(i).matData);
    matDataPopPost = cat(2, matDataPopPost, datastructPost(i).matData);
    matDataPopTask = cat(2, matDataPopTask, matDataDecoding(i).matData);
end

trialTypes = matTrialTypesIncl;

% Setup
rng(1)

famIdxAll = [1 2 4 5];
novIdx = [3 6];
famCombs = nchoosek(famIdxAll, 2);
nCombs = size(famCombs, 1);

% Preallocate
pFamiliarPre = zeros(nCombs,1);
pFamiliarPost = zeros(nCombs,1);
pFamiliarTask = zeros(nCombs,1);

trainFrac = 0.5;
nReps = 50;
nBoots = 1;

% Familiar decoding
for c = 1:nCombs
    idx = famCombs(c,:);
    excludeIdx = setdiff(1:6, idx);
    
    types = trialTypes;
    rmv = ismember(types(1,:), excludeIdx);
    
    typesUse = types;
    typesUse(:,rmv) = [];
    dataPre = matDataPopPre;
    dataPost = matDataPopPost;
    dataTask = matDataPopTask;
    dataPre(rmv,:) = [];
    dataPost(rmv,:) = [];

    % Run pOO decoding
    [~, pOOPre, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = ...
        doMuckliDecodingLDAblock2(dataPre, typesUse, trainFrac, nReps, nBoots, 0);
    [~, pOOPost, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = ...
        doMuckliDecodingLDAblock2(dataPost, typesUse, trainFrac, nReps, nBoots, 0);
    [~, pOOTask, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = ...
        doMuckliDecodingLDAblock2(dataTask, typesUse, trainFrac, nReps, nBoots, 0);
    
    % Store mean accuracy
    pFamiliarPre(c) = mean(pOOPre);
    pFamiliarPost(c) = mean(pOOPost);
    pFamiliarTask(c) = mean(pOOTask);
end

% Novel decoding (only pre and post)
idx = novIdx;
excludeIdx = setdiff(1:6, idx);
rmv = ismember(trialTypes(1,:), excludeIdx);

types = trialTypes;
types(:, rmv) = [];
dataPre = matDataPopPre;
dataPost = matDataPopPost;
dataPre(rmv,:) = [];
dataPost(rmv,:) = [];

[~, pOOPreNov, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = ...
    doMuckliDecodingLDAblock2(dataPre, types, trainFrac, nReps, nBoots, 1);
[~, pOOPostNov, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = ...
    doMuckliDecodingLDAblock2(dataPost, types, trainFrac, nReps, nBoots, 1);

pNovelPre = mean(pOOPreNov);
pNovelPost = mean(pOOPostNov);

% Calculate improvements
deltaPostPreFam = pFamiliarPost - pFamiliarPre;
deltaTaskPreFam = pFamiliarTask - pFamiliarPre;
deltaPostPreNov = pNovelPost - pNovelPre;

% Stats
[~, p_fam_vs_nov] = ttest(deltaPostPreFam, deltaPostPreNov);
[~, p_task_vs_post] = ttest(deltaTaskPreFam, deltaPostPreFam);


% Plotting
figure('Position', [500   400   600   696])
hold on

% Plot bars
bar(1, deltaPostPreNov, 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2)  % Novel
bar(2, mean(deltaPostPreFam), 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2)  % Familiar post-pre
bar(3, mean(deltaTaskPreFam), 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 2)  % Familiar task-pre

% Error bars
errorbar(2, mean(deltaPostPreFam), std(deltaPostPreFam), 'k', 'LineWidth', 1.5, 'CapSize', 10)
errorbar(3, mean(deltaTaskPreFam), std(deltaTaskPreFam), 'k', 'LineWidth', 1.5, 'CapSize', 10)

% dots
scatter(ones(size(deltaPostPreFam,2),1)*2, deltaPostPreFam, 40, 'k', 'filled')
scatter(ones(size(deltaTaskPreFam,2),1)*3, deltaTaskPreFam, 40, 'k', 'filled')

% Labels
xticks([1 2 3])
xticklabels({'Novel Δpost-pre', 'Familiar Δpost-pre', 'Familiar Δtask-pre'})
ylabel('Decoding accuracy change (%)')
title(sprintf('Fam>Nov p=%.3f | Task>Post p=%.3f', p_fam_vs_nov, p_task_vs_post))
yline(0, '--k'), ylim([-10 50]), box off, 

if save_fig
    func_save_fig('L23_decodingIncreasePreVsPostVsTask')
    func_save_fig('L5_singleCellContToScatterGray')
end

%% DECODING using one neuron out to get contribution per neuron
% we first perform decoding with all neurons. Then one by one leave a
% neuron out and calculate the difference in average decoding. The
% difference is the contribution for that neuron.
% takes a long time, probably ~30 hours at 1000 nReps.
% should also try it the other way around, decoding with only 1 neuron at a
% time to get the individual contribution of that neuron.

rng(1) % for reproducibility

doDecoding = 1;
doPlotting = 1;
nReps = 200;
nBoots = 1;
famIdx = [1 2 4 5];
novIdx = [3 6];
trialTypes = matTrialTypesIncl;

matDataPopPre = datastructPre(1).matData;
matDataPopPost = datastructPost(1).matData;
matDataPopTask = matDataDecoding(1).matData;

for i = 2:nfiles
    matDataPopPre = cat(2, matDataPopPre, datastructPre(i).matData);
    matDataPopPost = cat(2, matDataPopPost, datastructPost(i).matData);
    matDataPopTask = cat(2, matDataPopTask, matDataDecoding(i).matData);
end

idx = famIdx; % which images to include? famidx/novidx?
% idx = 1:6; % which images to include? famidx/novidx?
ix = 1:6;
ix(idx) = [];
for i = 1:length(ix)
    rmv = trialTypes(1,:)==ix(i);
    trialTypes(:,rmv)=[];
    matDataPopPre(rmv,:)=[];
    matDataPopPost(rmv,:)=[];
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

%     % pre training
%     ffPre = zeros(nReps, size(matDataPopPre,2));
%     ooPre = zeros(nReps, size(matDataPopPre,2));
%     foPre = zeros(nReps, size(matDataPopPre,2));
%     ofPre = zeros(nReps, size(matDataPopPre,2));
%     for neuron = 1:size(matDataPopPre,2)
%         data = matDataPopPre;
%         data(:,neuron)=[]; % remove neuron
%         [ff, oo, fo, of] = doMuckliDecodingLDAblock2(data, trialTypes, trainFrac, nReps, nBoots, 0);
%         ffPre(:,neuron) = ff; ooPre(:,neuron) = oo; foPre(:,neuron) = fo; ofPre(:,neuron) = of;
%         disp(neuron)
%     end
%     % now for all neurons included:
%     [pFFPre, pOOPre, pFOPre, pOFPre] = doMuckliDecodingLDAblock2(matDataPopPre, trialTypes, trainFrac, nReps, nBoots, 0);
% 
%     % post training
%     ffPost = zeros(nReps, size(matDataPopPost,2));
%     ooPost = zeros(nReps, size(matDataPopPost,2));
%     foPost = zeros(nReps, size(matDataPopPost,2));
%     ofPost = zeros(nReps, size(matDataPopPost,2));
%     for neuron = 1:size(matDataPopPost,2)
%         data = matDataPopPost;
%         data(:,neuron)=[]; % remove neuron
%         [ff, oo, fo, of] = doMuckliDecodingLDAblock2(data, trialTypes, trainFrac, nReps, nBoots, 0);
%         ffPost(:,neuron) = ff; ooPost(:,neuron) = oo; foPost(:,neuron) = fo; ofPost(:,neuron) = of;
%         disp(neuron)
%     end
%     % now for all neurons included:
%     [pFFPost, pOOPost, pFOPost, pOFPost] = doMuckliDecodingLDAblock2(matDataPopPost, trialTypes, trainFrac, nReps, nBoots, 0);

    % Task
    ffTask = zeros(nReps, size(matDataPopTask,2));
    ooTask = zeros(nReps, size(matDataPopTask,2));
    foTask = zeros(nReps, size(matDataPopTask,2));
    ofTask = zeros(nReps, size(matDataPopTask,2));
    for neuron = 1:size(matDataPopTask,2)
        data = matDataPopTask;
        data(:,neuron)=[]; % remove neuron
        [ff, oo, fo, of] = doMuckliDecodingLDAblock2(data, trialTypes, trainFrac, nReps, nBoots, 0);
        ffTask(:,neuron) = ff; ooTask(:,neuron) = oo; foTask(:,neuron) = fo; ofTask(:,neuron) = of;
        disp(neuron)
    end
    % now for all neurons included:
    [pFFTask, pOOTask, pFOTask, pOFTask] = doMuckliDecodingLDAblock2(matDataPopTask, trialTypes, trainFrac, nReps, nBoots, 0);

end

% contFFPre = mean(pFFPre)-mean(ffPre);
% contOOPre = mean(pOOPre)-mean(ooPre);
% contFOPre = mean(pFOPre)-mean(foPre);
% contOFPre = mean(pOFPre)-mean(ofPre);
% contFFPost = mean(pFFPost)-mean(ffPost);
% contOOPost = mean(pOOPost)-mean(ooPost);
% contFOPost = mean(pFOPost)-mean(foPost);
% contOFPost = mean(pOFPost)-mean(ofPost);
contFFTask = mean(pFFTask)-mean(ffTask);
contOOTask = mean(pOOTask)-mean(ooTask);
contFOTask = mean(pFOTask)-mean(foTask);
contOFTask = mean(pOFTask)-mean(ofTask);
% contFFPreRel = (mean(pFFPre)-mean(ffPre))./mean(pFFPre);
% contOOPreRel = (mean(pOOPre)-mean(ooPre))./mean(pOOPre);
% contFOPreRel = (mean(pFOPre)-mean(foPre))./mean(pFOPre);
% contOFPreRel = (mean(pOFPre)-mean(ofPre))./mean(pOFPre);
% contFFPostRel = (mean(pFFPost)-mean(ffPost))./mean(pFFPost);
% contOOPostRel = (mean(pOOPost)-mean(ooPost))./mean(pOOPost);
% contFOPostRel = (mean(pFOPost)-mean(foPost))./mean(pFOPost);
% contOFPostRel = (mean(pOFPost)-mean(ofPost))./mean(pOFPost);

if doPlotting
    figure
    histogram(contOOPreRel, 'normalization', 'probability'), hold on
    histogram(contOOPostRel, 'normalization', 'probability')
    xlabel('OO contribution'), ylabel('Nr of cells'), legend({'Naive','Expert'})

    figure
    histogram(contOOPre, 'normalization', 'probability'), hold on
    histogram(contOOPost, 'normalization', 'probability')
    xlabel('OO contribution'), ylabel('Nr of cells'), legend({'Naive','Expert'})


    bins = -2:0.1:2;
    figure
    histogram(contOOPost, bins, 'normalization' , 'probability'), hold on
    histogram(contFOPost, bins, 'normalization' , 'probability')
    histogram(contOFPost, bins, 'normalization' , 'probability')

    figure,
    scatter(scatFullFamPopPost(contOFPost<0.1), scatOcclFamPopPost(contOFPost<0.1), sz, 'filled'),hold on
    scatter(scatFullFamPopPost(contOFPost>0.1), scatOcclFamPopPost(contOFPost>0.1), sz, 'filled')

    figure,
    scatter(scatFullFamPopPost(contOOPost<0.1), scatOcclFamPopPost(contOOPost<0.1), sz, 'filled'),hold on
    scatter(scatFullFamPopPost(contOOPost>0.1), scatOcclFamPopPost(contOOPost>0.1), sz, 'filled')

    figure,
    scatter(scatFullFamPopPre(contOOPre<0.1), scatOcclFamPopPre(contOOPre<0.1), sz, 'filled'),hold on
    scatter(scatFullFamPopPre(contOOPre>0.1), scatOcclFamPopPre(contOOPre>0.1), sz, 'filled')

    [rDistVsOccl, pDistVsOccl] = corrcoef(rfDistPostData, contOOPost);

    figure('Position', [96   226   560   420])
    fits = polyfit(rfDistPostData, contOOPost,1);
    fit1 = polyval(fits,rfDistPostData);
    vr = scatter(rfDistPostData, contOOPost, sz, 'k', 'filled'); hold on
    plot(rfDistPostData, fit1, 'r', 'LineWidth', 1.5)
    ylabel('O contribution'), xlabel('Dist to edge'), title('RFdist vs Occl res')
    text(5,25,sprintf('r=%.3f',rDistVsOccl(2))), text(5,20,sprintf('p=%.3f', pDistVsOccl(2)))
    
end

%% decoding with only 1 neuron at a time
% we decode with 1 neuron at a time, to see what the decoding accuracy is
% for each neuron. We use that as an estimate of the neuron's usefullness
% for decoding.

rng(1) % for reproducibility

% load active data, note that this is only relevant if you decode from
% familiar images. If you also want novel images you get errors.
if nfiles == 6
    load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\matDataDecodingActive.mat', 'matDataDecoding') % no need to load in trialtypes
elseif nfiles == 5
    load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\matDataDecodingActive.mat', 'matDataDecoding')
end

doDecoding = 1;
doPlotting = 1;
nReps = 200;
nBoots = 1;
famIdx = [1 2 4 5];
novIdx = [3 6];
trialTypes = matTrialTypesIncl;

matDataPopPre = datastructPre(1).matData;
matDataPopPost = datastructPost(1).matData;
matDataPopTask = matDataDecoding(1).matData;

for i = 2:nfiles
    matDataPopPre = cat(2, matDataPopPre, datastructPre(i).matData);
    matDataPopPost = cat(2, matDataPopPost, datastructPost(i).matData);
    matDataPopTask = cat(2, matDataPopTask, matDataDecoding(i).matData);
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

    % pre training
    ffPre = zeros(nReps, size(matDataPopPre,2));
    ooPre = zeros(nReps, size(matDataPopPre,2));
    foPre = zeros(nReps, size(matDataPopPre,2));
    ofPre = zeros(nReps, size(matDataPopPre,2));

    ffPermPre = zeros(nBoots, size(matDataPopPre,2));
    ooPermPre = zeros(nBoots, size(matDataPopPre,2));
    foPermPre = zeros(nBoots, size(matDataPopPre,2));
    ofPermPre = zeros(nBoots, size(matDataPopPre,2));

    for neuron = 1:size(matDataPopPre,2)
        data = matDataPopPre(:,neuron);
%         data(:,neuron)=[]; % remove neuron
        [ff, oo, fo, of, ffPerm, ooPerm, foPerm, ofPerm] = doMuckliDecodingLDAblock2(data, trialTypes, trainFrac, nReps, nBoots, 0);
        ffPre(:,neuron) = ff; ooPre(:,neuron) = oo; foPre(:,neuron) = fo; ofPre(:,neuron) = of;
        ffPermPre(:,neuron) = ffPerm; ooPermPre(:,neuron) = ooPerm; foPermPre(:,neuron) = foPerm; ofPermPre(:,neuron) = ofPerm;
        disp(neuron)
    end
%     % now for all neurons included:
%     [pFFPre, pOOPre, pFOPre, pOFPre] = doMuckliDecodingLDAblock2(matDataPopPre, trialTypes, trainFrac, nReps, nBoots, 0);

    % post training
    ffPost = zeros(nReps, size(matDataPopPost,2));
    ooPost = zeros(nReps, size(matDataPopPost,2));
    foPost = zeros(nReps, size(matDataPopPost,2));
    ofPost = zeros(nReps, size(matDataPopPost,2));

    ffPermPost = zeros(nBoots, size(matDataPopPost,2));
    ooPermPost = zeros(nBoots, size(matDataPopPost,2));
    foPermPost = zeros(nBoots, size(matDataPopPost,2));
    ofPermPost = zeros(nBoots, size(matDataPopPost,2));

    for neuron = 1:size(matDataPopPost,2)
        data = matDataPopPost(:,neuron);
%         data(:,neuron)=[]; % remove neuron
        [ff, oo, fo, of, ffPerm, ooPerm, foPerm, ofPerm] = doMuckliDecodingLDAblock2(data, trialTypes, trainFrac, nReps, nBoots, 0);
        ffPost(:,neuron) = ff; ooPost(:,neuron) = oo; foPost(:,neuron) = fo; ofPost(:,neuron) = of;
        ffPermPost(:,neuron) = ffPerm; ooPermPost(:,neuron) = ooPerm; foPermPost(:,neuron) = foPerm; ofPermPost(:,neuron) = ofPerm;
        disp(neuron)
    end
%     % now for all neurons included:
%     [pFFPost, pOOPost, pFOPost, pOFPost] = doMuckliDecodingLDAblock2(matDataPopPost, trialTypes, trainFrac, nReps, nBoots, 0);

    % task
    ffTask = zeros(nReps, size(matDataPopTask,2));
    ooTask = zeros(nReps, size(matDataPopTask,2));
    foTask = zeros(nReps, size(matDataPopTask,2));
    ofTask = zeros(nReps, size(matDataPopTask,2));

    ffPermTask = zeros(nBoots, size(matDataPopTask,2));
    ooPermTask = zeros(nBoots, size(matDataPopTask,2));
    foPermTask = zeros(nBoots, size(matDataPopTask,2));
    ofPermTask = zeros(nBoots, size(matDataPopTask,2));

    for neuron = 1:size(matDataPopTask,2)
        data = matDataPopTask(:,neuron);
%         data(:,neuron)=[]; % remove neuron
        [ff, oo, fo, of, ffPerm, ooPerm, foPerm, ofPerm] = doMuckliDecodingLDAblock2(data, trialTypes, trainFrac, nReps, nBoots, 0);
        ffTask(:,neuron) = ff; ooTask(:,neuron) = oo; foTask(:,neuron) = fo; ofTask(:,neuron) = of;
        ffPermTask(:,neuron) = ffPerm; ooPermTask(:,neuron) = ooPerm; foPermTask(:,neuron) = foPerm; ofPermTask(:,neuron) = ofPerm;
        disp(neuron)
    end


end

%% an plot results of single cell decoding
figure('Position', [133 202 1575 732])
% better colors
% colorM = cmapL([0 0 0; 0.8 0.8 0.8], 100); % sorting is descending
% colorM = cmapL([0 0 0; [144, 169, 85]/255; [236 243 158]/255], 100); % sorting is descending
% colorM = cmapL([0 0 0; [220, 47, 2]/255; [234, 226, 183]/255], 100); % sorting is descending

% we change the colormap in the end anyway, but you can use this function
% to make a nicer one if needed
colorM = cmapL([0 0 0; [220, 47, 2]/255; [255, 233, 78]/255], 100); % sorting is descending

sz = 10;
s(1) = subplot(3,4,1);
scatCol = mean(ffPre);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPreCut(sortIndex), scatOcclFamPopPreCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), title('NO-NO'), colormap(colorM), colorbar
s(2) = subplot(3,4,2);
scatCol = mean(ooPre);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPreCut(sortIndex), scatOcclFamPopPreCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), title('O-O'), colormap(colorM), colorbar
s(3) = subplot(3,4,3);
scatCol = mean(foPre);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPreCut(sortIndex), scatOcclFamPopPreCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), title('NO-O'), colormap(colorM), colorbar
s(4) = subplot(3,4,4);
scatCol = mean(ofPre);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPreCut(sortIndex), scatOcclFamPopPreCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), title('O-NO'), colormap(colorM), colorbar
s(5) = subplot(3,4,5);
scatCol = mean(ffPost);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPostCut(sortIndex), scatOcclFamPopPostCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar
s(6) = subplot(3,4,6);
scatCol = mean(ooPost);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPostCut(sortIndex), scatOcclFamPopPostCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar
s(7) = subplot(3,4,7);
scatCol = mean(foPost);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPostCut(sortIndex), scatOcclFamPopPostCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar
s(8) = subplot(3,4,8);
scatCol = mean(ofPost);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPostCut(sortIndex), scatOcclFamPopPostCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar
s(9) = subplot(3,4,9);
scatCol = mean(ffTask);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullPopCut(sortIndex), scatOcclPopCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(hot), colorbar
s(10) = subplot(3,4,10);
scatCol = mean(ooTask);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullPopCut(sortIndex), scatOcclPopCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar
s(11) = subplot(3,4,11);
scatCol = mean(foTask);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullPopCut(sortIndex), scatOcclPopCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar
s(12) = subplot(3,4,12);
scatCol = mean(ofTask);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullPopCut(sortIndex), scatOcclPopCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar

hotMap = flipud(hot);
nColors = size(hotMap, 1); % Number of colors in the colormap
cutOffIndex = floor(nColors * 0.3); % Calculate cutoff index to remove the bottom 20%
modifiedHotMap = hotMap(cutOffIndex:end, :); % Keep the upper part of the colormap
colormap(modifiedHotMap); % cut off map

% Adjust subplot properties

for j = 1:length(s)
    if nfiles == 6
        labels = {'-1','','0','','1','','2','','>2.5'};
        s(j).YLim = [-1 3]; s(j).YTick = -1:0.5:3; s(j).YTickLabel = labels;
        s(j).XLim = [-1 3]; s(j).XTick = -1:0.5:3; s(j).XTickLabel = labels;
        s(j).CLim = [25 50]; % Adjust according to your data range
        s(j).XTickLabelRotation = 0;
    else
        labels = {'','0','','1','','2','','>2.5'};
        s(j).YLim = [-0.5 3]; s(j).YTick = -0.5:0.5:3; s(j).YTickLabel = labels;
        s(j).XLim = [-0.5 3]; s(j).XTick = -0.5:0.5:3; s(j).XTickLabel = labels;
        s(j).CLim = [25 50]; % Adjust according to your data range
        s(j).XTickLabelRotation = 0;
    end
end

if save_fig
    func_save_fig('L23_singleCellContToScatterGray')
    func_save_fig('L5_singleCellContToScatterGray')
end

figure('Position', [680   328   320   650])
boxchart([ones(size(mean(ffTask))), ones(size(mean(ooTask)))+1, ones(size(mean(foTask)))+2, ones(size(mean(ofTask)))+3], ...
    [mean(ffTask), mean(ooTask), mean(foTask), mean(ofTask)], 'MarkerStyle','none'), hold on
xlim([0 5]), ylabel('Decoding accuracy per cell (%)'), xticks([1 2 3 4]); ylim([10 60]), yline(25)
xticklabels({'NO-NO', 'O-O', 'NO-O', 'O-NO'}), xtickangle(45), 

if save_fig
    func_save_fig('L23_singleCellContBoxplot')
    func_save_fig('L5_singleCellContBoxplot')
end





% bins = 0:2:100;
% 
% figure('Position', [47         341        1766         262])
% subplot(1,4,1)
% histogram(mean(ffPre), bins, 'normalization', 'probability'), hold on
% histogram(mean(ffPost), bins, 'normalization', 'probability')
% xline(mean(mean(ffPermPre))), xline(mean(mean(ffPermPost)))
% xlabel('NO-NO contribution'), ylabel('Nr of cells'), legend({'Naive','Expert'})
% subplot(1,4,2)
% histogram(mean(ooPre), bins, 'normalization', 'probability'), hold on
% histogram(mean(ooPost), bins, 'normalization', 'probability')
% xline(mean(mean(ooPermPre))), xline(mean(mean(ooPermPost)))
% xlabel('O-O contribution'), ylabel('Nr of cells'), legend({'Naive','Expert'})
% subplot(1,4,3)
% histogram(mean(foPre), bins, 'normalization', 'probability'), hold on
% histogram(mean(foPost), bins, 'normalization', 'probability')
% xline(mean(mean(foPermPre))), xline(mean(mean(foPermPost)))
% xlabel('NO-O contribution'), ylabel('Nr of cells'), legend({'Naive','Expert'})
% subplot(1,4,4)
% histogram(mean(ofPre), bins, 'normalization', 'probability'), hold on
% histogram(mean(ofPost), bins, 'normalization', 'probability')
% xline(mean(mean(ofPermPre))), xline(mean(mean(ofPermPost)))
% xlabel('O-NO contribution'), ylabel('Nr of cells'), legend({'Naive','Expert'})

%% some average plotting and calculations
ix = scatOcclPop>0.5 | scatFullPop>0.5;
mnFFPre = mean(mean(ffPre(:,ix)))
mnOOPre = mean(mean(ooPre(:,ix)))
mnFOPre = mean(mean(foPre(:,ix)))
mnOFPre = mean(mean(ofPre(:,ix)))

ix = scatOcclFamPopPost>0.5 | scatFullFamPopPost>0.5;
mnFFPost = mean(mean(ffPost(:,ix)))
mnOOPost = mean(mean(ooPost(:,ix)))
mnFOPost = mean(mean(foPost(:,ix)))
mnOFPost = mean(mean(ofPost(:,ix)))

ix = scatOcclPop>0.5 | scatFullPop>0.5;
mnFFtask = mean(mean(ffTask(:,ix)))
mnOOtask = mean(mean(ooTask(:,ix)))
mnFOtask = mean(mean(foTask(:,ix)))
mnOFtask = mean(mean(ofTask(:,ix)))

%% or just load preanalysed data
if nfiles == 6
    load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\DecodingData\cellContributionData_oneCell_4imgs_preposttaskL23.mat')
else
    load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\DecodingData\cellContributionData_oneCell_4imgs_preposttaskL5.mat')
end

figure
boxchart([ones(size(mean(ffPre))), ones(size(mean(ffPost)))+1, ones(size(mean(ffTask)))+2], ...
    [mean(ffPre), mean(ffPost), mean(ffTask)], 'MarkerStyle','none'), hold on
boxchart([ones(size(mean(ooPre)))+4, ones(size(mean(ooPost)))+5, ones(size(mean(ooTask)))+6], ...
    [mean(ooPre), mean(ooPost), mean(ooTask)], 'MarkerStyle','none'), hold on
boxchart([ones(size(mean(foPre)))+8, ones(size(mean(foPost)))+9, ones(size(mean(foTask)))+10], ...
    [mean(foPre), mean(foPost), mean(foTask)], 'MarkerStyle','none'), hold on
boxchart([ones(size(mean(ofPre)))+12, ones(size(mean(ofPost)))+13, ones(size(mean(ofTask)))+14], ...
    [mean(ofPre), mean(ofPost), mean(ofTask)], 'MarkerStyle','none'), hold on
xlim([0 16]), ylim([10 60]), ylabel('Decoding accuracy per neuron'), xticks([1 2 3 5 6 7 9 10 11 13 14 15]);
xticklabels({'', 'NO-NO','', '','O-O', '', '','NO-O', '', '','O-NO', '',}), xtickangle(45), 

if save_fig
    func_save_fig('L23_contributionPerCell')
    func_save_fig('L5_contributionPerCell')
end

% also plotted sorted based on contribution so that they come out more
% clearly
figure('Position', [133 202 1575 732])
% better colors
% colorM = cmapL([0 0 0; 0.8 0.8 0.8], 100); % sorting is descending
% colorM = cmapL([0 0 0; [144, 169, 85]/255; [236 243 158]/255], 100); % sorting is descending
% colorM = cmapL([0 0 0; [220, 47, 2]/255; [234, 226, 183]/255], 100); % sorting is descending

% we change the colormap in the end anyway, but you can use this function
% to make a nicer one if needed
colorM = cmapL([0 0 0; [220, 47, 2]/255; [255, 233, 78]/255], 100); % sorting is descending

sz = 10;
s(1) = subplot(3,4,1);
scatCol = mean(ffPre);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPreCut(sortIndex), scatOcclFamPopPreCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), title('NO-NO'), colormap(colorM), colorbar
s(2) = subplot(3,4,2);
scatCol = mean(ooPre);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPreCut(sortIndex), scatOcclFamPopPreCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), title('O-O'), colormap(colorM), colorbar
s(3) = subplot(3,4,3);
scatCol = mean(foPre);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPreCut(sortIndex), scatOcclFamPopPreCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), title('NO-O'), colormap(colorM), colorbar
s(4) = subplot(3,4,4);
scatCol = mean(ofPre);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPreCut(sortIndex), scatOcclFamPopPreCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), title('O-NO'), colormap(colorM), colorbar
s(5) = subplot(3,4,5);
scatCol = mean(ffPost);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPostCut(sortIndex), scatOcclFamPopPostCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar
s(6) = subplot(3,4,6);
scatCol = mean(ooPost);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPostCut(sortIndex), scatOcclFamPopPostCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar
s(7) = subplot(3,4,7);
scatCol = mean(foPost);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPostCut(sortIndex), scatOcclFamPopPostCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar
s(8) = subplot(3,4,8);
scatCol = mean(ofPost);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullFamPopPostCut(sortIndex), scatOcclFamPopPostCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar
s(9) = subplot(3,4,9);
scatCol = mean(ffTask);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullPopCut(sortIndex), scatOcclPopCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(hot), colorbar
s(10) = subplot(3,4,10);
scatCol = mean(ooTask);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullPopCut(sortIndex), scatOcclPopCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar
s(11) = subplot(3,4,11);
scatCol = mean(foTask);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullPopCut(sortIndex), scatOcclPopCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar
s(12) = subplot(3,4,12);
scatCol = mean(ofTask);
[sortedScatCol, sortIndex] = sort(scatCol);
scatter(scatFullPopCut(sortIndex), scatOcclPopCut(sortIndex), sz, sortedScatCol, 'filled');
refline(1), 
xlabel('Full res'), ylabel('Occl res'), colormap(colorM), colorbar

hotMap = flipud(hot);
nColors = size(hotMap, 1); % Number of colors in the colormap
cutOffIndex = floor(nColors * 0.3); % Calculate cutoff index to remove the bottom 20%
modifiedHotMap = hotMap(cutOffIndex:end, :); % Keep the upper part of the colormap
colormap(modifiedHotMap); % cut off map

% Adjust subplot properties

for j = 1:length(s)
    if nfiles == 6
        labels = {'-1','','0','','1','','2','','>2.5'};
        s(j).YLim = [-1 3]; s(j).YTick = -1:0.5:3; s(j).YTickLabel = labels;
        s(j).XLim = [-1 3]; s(j).XTick = -1:0.5:3; s(j).XTickLabel = labels;
        s(j).CLim = [25 50]; % Adjust according to your data range
        s(j).XTickLabelRotation = 0;
    else
        labels = {'','0','','1','','2','','>2.5'};
        s(j).YLim = [-0.5 3]; s(j).YTick = -0.5:0.5:3; s(j).YTickLabel = labels;
        s(j).XLim = [-0.5 3]; s(j).XTick = -0.5:0.5:3; s(j).XTickLabel = labels;
        s(j).CLim = [25 50]; % Adjust according to your data range
        s(j).XTickLabelRotation = 0;
    end
end

if save_fig
    func_save_fig('L23_singleCellContToScatterGray')
    func_save_fig('L5_singleCellContToScatterGray')
end

figure('Position', [680   328   320   650])
boxchart([ones(size(mean(ffTask))), ones(size(mean(ooTask)))+1, ones(size(mean(foTask)))+2, ones(size(mean(ofTask)))+3], ...
    [mean(ffTask), mean(ooTask), mean(foTask), mean(ofTask)], 'MarkerStyle','none'), hold on
xlim([0 5]), ylabel('Decoding accuracy per cell (%)'), xticks([1 2 3 4]); ylim([10 60]), yline(25)
xticklabels({'NO-NO', 'O-O', 'NO-O', 'O-NO'}), xtickangle(45), 

if save_fig
    func_save_fig('L23_singleCellContBoxplot')
    func_save_fig('L5_singleCellContBoxplot')
end



%%
% plot with thresholded values of significant decoding. Note that this is
% based on only 100 perms, should use more if we want to make this figure
figure('Position', [133         202        1199         732])
warning off
% Define your threshold and colors
threshold = 0.05;  % Replace with your threshold
% colorAbove = [56, 176, 0]/255;  % Greenish color for values above threshold
colorBelow = [56, 176, 0]/255;  % Greenish color for values above threshold
colorAbove = [1 0 0];  % Redish color for values below threshold
% colorAbove = [0 0 0];  % Greenish color for values above threshold
% colorBelow = [0.5 0.5 0.5];  % Redish color for values below threshold
faceAlphaValue = 0.5;  % Set the FaceAlpha value
sz = 20;
% Subplot 1
s(1) = subplot(3,4,1);
scatCol = mean(ffPre); clear pval, for i = 1:size(scatCol,2), pval(i) = paretoEst(ffPermPre(:,i)/100, scatCol(i)/100); end
customColors = thresholdColors(pval, threshold, colorAbove, colorBelow);
scatter(scatFullFamPopPreCut, scatOcclFamPopPreCut, sz, customColors, 'filled', 'MarkerFaceAlpha', faceAlphaValue);
refline(1); ; xlabel('NO response'); ylabel('O response'); title('NO-NO');
% Subplot 2
s(2) = subplot(3,4,2);
scatCol = mean(ooPre); clear pval, for i = 1:size(scatCol,2), pval(i) = paretoEst(ooPermPre(:,i)/100, scatCol(i)/100); end
customColors = thresholdColors(pval, threshold, colorAbove, colorBelow);
scatter(scatFullFamPopPreCut, scatOcclFamPopPreCut, sz, customColors, 'filled', 'MarkerFaceAlpha', faceAlphaValue);
refline(1); ; xlabel('NO response'); ylabel('O response'); title('O-O');
% Subplot 3
s(3) = subplot(3,4,3);
scatCol = mean(foPre); clear pval, for i = 1:size(scatCol,2), pval(i) = paretoEst(foPermPre(:,i)/100, scatCol(i)/100); end
customColors = thresholdColors(pval, threshold, colorAbove, colorBelow);
scatter(scatFullFamPopPreCut, scatOcclFamPopPreCut, sz, customColors, 'filled', 'MarkerFaceAlpha', faceAlphaValue);
refline(1); ; xlabel('NO response'); ylabel('O response'); title('NO-O');
% Subplot 4
s(4) = subplot(3,4,4);
scatCol = mean(ofPre); clear pval, for i = 1:size(scatCol,2), pval(i) = paretoEst(ofPermPre(:,i)/100, scatCol(i)/100); end
customColors = thresholdColors(pval, threshold, colorAbove, colorBelow);
scatter(scatFullFamPopPreCut, scatOcclFamPopPreCut, sz, customColors, 'filled', 'MarkerFaceAlpha', faceAlphaValue);
refline(1); ; xlabel('NO response'); ylabel('O response'); title('O-NO');
% Subplot 5
s(5) = subplot(3,4,5);
scatCol = mean(ffPost); clear pval, for i = 1:size(scatCol,2), pval(i) = paretoEst(ffPermPost(:,i)/100, scatCol(i)/100); end
customColors = thresholdColors(pval, threshold, colorAbove, colorBelow);
scatter(scatFullFamPopPostCut, scatOcclFamPopPostCut, sz, customColors, 'filled', 'MarkerFaceAlpha', faceAlphaValue);
refline(1); ; xlabel('NO response'); ylabel('O response');
% Subplot 6
s(6) = subplot(3,4,6);
scatCol = mean(ooPost); clear pval, for i = 1:size(scatCol,2), pval(i) = paretoEst(ooPermPost(:,i)/100, scatCol(i)/100); end
customColors = thresholdColors(pval, threshold, colorAbove, colorBelow);
scatter(scatFullFamPopPostCut, scatOcclFamPopPostCut, sz, customColors, 'filled', 'MarkerFaceAlpha', faceAlphaValue);
refline(1); ; xlabel('NO response'); ylabel('O response');
% Subplot 7
s(7) = subplot(3,4,7);
scatCol = mean(foPost); clear pval, for i = 1:size(scatCol,2), pval(i) = paretoEst(foPermPost(:,i)/100, scatCol(i)/100); end
customColors = thresholdColors(pval, threshold, colorAbove, colorBelow);
scatter(scatFullFamPopPostCut, scatOcclFamPopPostCut, sz, customColors, 'filled', 'MarkerFaceAlpha', faceAlphaValue);
refline(1); ; xlabel('NO response'); ylabel('O response');
% Subplot 8
s(8) = subplot(3,4,8);
scatCol = mean(ofPost); clear pval, for i = 1:size(scatCol,2), pval(i) = paretoEst(ofPermPost(:,i)/100, scatCol(i)/100); end
customColors = thresholdColors(pval, threshold, colorAbove, colorBelow);
scatter(scatFullFamPopPostCut, scatOcclFamPopPostCut, sz, customColors, 'filled', 'MarkerFaceAlpha', faceAlphaValue);
refline(1); ; xlabel('NO response'); ylabel('O response');
% Subplot 9
s(9) = subplot(3,4,9);
scatCol = mean(ffTask); clear pval, for i = 1:size(scatCol,2), pval(i) = paretoEst(ffPermTask(:,i)/100, scatCol(i)/100); end
customColors = thresholdColors(pval, threshold, colorAbove, colorBelow);
scatter(scatFullPop, scatOcclPop, sz, customColors, 'filled', 'MarkerFaceAlpha', faceAlphaValue);
refline(1); ; xlabel('NO response'); ylabel('O response');
% Subplot 10
s(10) = subplot(3,4,10);
scatCol = mean(ooTask); clear pval, for i = 1:size(scatCol,2), pval(i) = paretoEst(ooPermTask(:,i)/100, scatCol(i)/100); end
customColors = thresholdColors(pval, threshold, colorAbove, colorBelow);
scatter(scatFullPop, scatOcclPop, sz, customColors, 'filled', 'MarkerFaceAlpha', faceAlphaValue);
refline(1); ; xlabel('NO response'); ylabel('O response');
% Subplot 11
s(11) = subplot(3,4,11);
scatCol = mean(foTask); clear pval, for i = 1:size(scatCol,2), pval(i) = paretoEst(foPermTask(:,i)/100, scatCol(i)/100); end
customColors = thresholdColors(pval, threshold, colorAbove, colorBelow);
scatter(scatFullPop, scatOcclPop, sz, customColors, 'filled', 'MarkerFaceAlpha', faceAlphaValue);
refline(1); ; xlabel('NO response'); ylabel('O response');
% Subplot 12
s(12) = subplot(3,4,12);
scatCol = mean(ofTask); clear pval, for i = 1:size(scatCol,2), pval(i) = paretoEst(ofPermTask(:,i)/100, scatCol(i)/100); end
customColors = thresholdColors(pval, threshold, colorAbove, colorBelow);
scatter(scatFullPop, scatOcclPop, sz, customColors, 'filled', 'MarkerFaceAlpha', faceAlphaValue);
refline(1); ; xlabel('NO response'); ylabel('O response');
warning on

% Adjust axes for all subplots
for j = 1:length(s)
    s(j).YLim = [-1 3]; s(j).YTick = -1:0.5:3; s(j).XLim = [-1 3]; s(j).XTick = -1:0.5:3;
end

%% decoding with adding n-neurons every time to get an accuracy plot as a function of added neurons (similar as voxels in case of humans?)

% load active data, note that this is only relevant if you decode from
% familiar images. If you also want novel images you get errors.
if nfiles == 6
    load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\prePostGrayCopy\matDataDecodingActive.mat', 'matDataDecoding') % no need to load in trialtypes
elseif nfiles == 5
    load('D:\2Pdata\Koen\Muckli\Analyses\Rbp4\prePostGrayCopy\matDataDecodingActive.mat', 'matDataDecoding')
end

rng(1) % for reproducibility

doDecoding = 1;
nReps = 20; % nr of 50/50 train/test repetitions
nBoots = 1; % nr of boots for chance level (not important here)
nRepeats = 20; % how many times do we want to create the curve
nCellsStep = 5; % nr of cells per step added

ffPre = [];
ooPre = [];
foPre = [];
ofPre = [];
famIdx = [1 2 4 5];
novIdx = [3 6];
trialTypes = matTrialTypesIncl;

matDataPopPre = datastructPre(1).matData;
matDataPopPost = datastructPost(1).matData;
matDataPopTask = matDataDecoding(1).matData;

for chunk = 2:nfiles
    matDataPopPre = cat(2, matDataPopPre, datastructPre(chunk).matData);
    matDataPopPost = cat(2, matDataPopPost, datastructPost(chunk).matData);
    matDataPopTask = cat(2, matDataPopTask, matDataDecoding(chunk).matData);
end

idx = famIdx; % which images to include? famidx/novidx?
% idx = 1:6; % which images to include? famidx/novidx?
ix = 1:6;
ix(idx) = [];
for chunk = 1:length(ix)
    rmv = trialTypes(1,:)==ix(chunk);
    trialTypes(:,rmv)=[];
    matDataPopPre(rmv,:)=[];
    matDataPopPost(rmv,:)=[];
end

if doDecoding
    [ffPre, ooPre, foPre, ofPre] = runChunkDecoding(matDataPopPre, nCellsStep, nRepeats, trialTypes, trainFrac, nReps, nBoots);
    [ffPost, ooPost, foPost, ofPost] = runChunkDecoding(matDataPopPost, nCellsStep, nRepeats, trialTypes, trainFrac, nReps, nBoots);
    [ffTask, ooTask, foTask, ofTask] = runChunkDecoding(matDataPopTask, nCellsStep, nRepeats, trialTypes, trainFrac, nReps, nBoots);
end

%%
% Plot results
trialWidth = 0.5;
avgWidth = 3;
trialc1 = [1 0 0 0.2]; % L2/3 color
% trialc2 = [1 0 0 0.2]; % L5 color
avgc1 = [1 0 0]; % L2/3 color
% avgc2 = [1 0 0]; % L2/3 color

titles = {'NO-NO','O-O','NO-O','O-NO',    'NO-NO','O-O','NO-O','O-NO',    'NO-NO','O-O','NO-O','O-NO'};

% figure('Position', [225         235        1505         650]) 
s(1) = subplot(3,4,1);
plot(squeeze(mean(ffPre)), 'Color', trialc1, 'LineWidth', trialWidth), hold on, yline(25)
hold on, plot(mean(squeeze(mean(ffPre)),2), 'Color', avgc1, 'LineWidth', avgWidth), xlabel('Nr of cell groups'), ylabel('Decoding accuracy')

s(2) = subplot(3,4,2);
plot(squeeze(mean(ooPre)), 'Color', trialc1, 'LineWidth', trialWidth), hold on, yline(25)
hold on, plot(mean(squeeze(mean(ooPre)),2), 'Color', avgc1, 'LineWidth', avgWidth), xlabel('Nr of cell groups'), ylabel('Decoding accuracy')

s(3) = subplot(3,4,3);
plot(squeeze(mean(foPre)), 'Color', trialc1, 'LineWidth', trialWidth), hold on, yline(25)
hold on, plot(mean(squeeze(mean(foPre)),2), 'Color', avgc1, 'LineWidth', avgWidth), xlabel('Nr of cell groups'), ylabel('Decoding accuracy')

s(4) = subplot(3,4,4);
plot(squeeze(mean(ofPre)), 'Color', trialc1, 'LineWidth', trialWidth), hold on, yline(25)
hold on, plot(mean(squeeze(mean(ofPre)),2), 'Color', avgc1, 'LineWidth', avgWidth), xlabel('Nr of cell groups'), ylabel('Decoding accuracy')

s(5) = subplot(3,4,5);
plot(squeeze(mean(ffPost)), 'Color', trialc1, 'LineWidth', trialWidth), hold on, yline(25)
hold on, plot(mean(squeeze(mean(ffPost)),2), 'Color', avgc1, 'LineWidth', avgWidth), xlabel('Nr of cell groups'), ylabel('Decoding accuracy')

s(6) = subplot(3,4,6);
plot(squeeze(mean(ooPost)), 'Color', trialc1, 'LineWidth', trialWidth), hold on, yline(25)
hold on, plot(mean(squeeze(mean(ooPost)),2), 'Color', avgc1, 'LineWidth', avgWidth), xlabel('Nr of cell groups'), ylabel('Decoding accuracy')

s(7) = subplot(3,4,7);
plot(squeeze(mean(foPost)), 'Color', trialc1, 'LineWidth', trialWidth), hold on, yline(25)
hold on, plot(mean(squeeze(mean(foPost)),2), 'Color', avgc1, 'LineWidth', avgWidth), xlabel('Nr of cell groups'), ylabel('Decoding accuracy')

s(8) = subplot(3,4,8);
plot(squeeze(mean(ofPost)), 'Color', trialc1, 'LineWidth', trialWidth), hold on, yline(25)
hold on, plot(mean(squeeze(mean(ofPost)),2), 'Color', avgc1, 'LineWidth', avgWidth), xlabel('Nr of cell groups'), ylabel('Decoding accuracy')

s(9) = subplot(3,4,9);
plot(squeeze(mean(ffTask)), 'Color', trialc1, 'LineWidth', trialWidth), hold on, yline(25)
hold on, plot(mean(squeeze(mean(ffTask)),2), 'Color', avgc1, 'LineWidth', avgWidth), xlabel('Nr of cell groups'), ylabel('Decoding accuracy')

s(10) = subplot(3,4,10);
plot(squeeze(mean(ooTask)), 'Color', trialc1, 'LineWidth', trialWidth), hold on, yline(25)
hold on, plot(mean(squeeze(mean(ooTask)),2), 'Color', avgc1, 'LineWidth', avgWidth), xlabel('Nr of cell groups'), ylabel('Decoding accuracy')

s(11) = subplot(3,4,11);
plot(squeeze(mean(foTask)), 'Color', trialc1, 'LineWidth', trialWidth), hold on, yline(25)
hold on, plot(mean(squeeze(mean(foTask)),2), 'Color', avgc1, 'LineWidth', avgWidth), xlabel('Nr of cell groups'), ylabel('Decoding accuracy')

s(12) = subplot(3,4,12);
plot(squeeze(mean(ofTask)), 'Color', trialc1, 'LineWidth', trialWidth), hold on, yline(25)
hold on, plot(mean(squeeze(mean(ofTask)),2), 'Color', avgc1, 'LineWidth', avgWidth), xlabel('Nr of cell groups'), ylabel('Decoding accuracy')

for j = 1:length(s)
    s(j).XLim = [0 180];
    s(j).YLim = [10 100];
    s(j).Title.String = titles{j};
end

if save_fig
    func_save_fig('L23vsL5_chunkDecoding')
    func_save_fig('L5_singleCellContBoxplot')
end


%% decoding over time (takes a long time, frame by frame!)
% smoothes the data a bit
rng(1) % for reproducibility

nReps = 100; % nr of iterations for training/testing on subsample in order to sample everything
nBoots = 1000; % nr of iterations for the permutation test
trainFrac = 0.5; % on what fraction to train?
famIdx = [1 2 4 5];
novIdx = [3 6];
idx = famIdx; % which images to include? famidx/novidx?
idx = [1 2 3 4 5 6]; % which images to include? famidx/novidx?
trialTypes = matTrialTypesIncl;

sigBin = 1:124; % bins for time decoding (depend on your axes)
nBins = length(sigBin); % number of bins

caResPopPre = datastructPre(1).CaResSort;
caResPopPost = datastructPost(1).CaResSort;
for i = 2:nfiles % concatenate all ROIs
    caResPopPre = cat(3, caResPopPre, datastructPre(i).CaResSort);
    caResPopPost = cat(3, caResPopPost, datastructPost(i).CaResSort);
end

% baseline correct and smooth?
caResPopPre = caResPopPre-mean(caResPopPre(vecAxSp,:,:));
figure, plot(vecAx, squeeze(mean(mean(caResPopPre,2),3)));
for i = 1:size(caResPopPre,3)
    for j = 1:size(caResPopPre,2)
        %         caResPopPre(:,j,i) = smoothG(caResPopPre(:,j,i), 2);
        data = squeeze(caResPopPre(:,j,i));
        caResPopPre(:,j,i) = smoothdata(data, 'movmean', 3);
    end
end
hold on, plot(vecAx, squeeze(mean(mean(caResPopPre,2),3)));

caResPopPost = caResPopPost-mean(caResPopPost(vecAxSp,:,:));
figure, plot(vecAx, squeeze(mean(mean(caResPopPost,2),3)));
for i = 1:size(caResPopPost,3)
    for j = 1:size(caResPopPost,2)
%         caResPopPost(:,j,i) = smoothG(caResPopPost(:,j,i), 2);
        data = squeeze(caResPopPost(:,j,i));
        caResPopPost(:,j,i) = smoothdata(data, 'movmean', 3);
    end
end
hold on, plot(vecAx, squeeze(mean(mean(caResPopPost,2),3)));

ix = 1:6;
ix(idx) = [];
% remove the other images from all data
for i = 1:length(ix)
    rmv = trialTypes(1,:)==ix(i);
    trialTypes(:,rmv)=[];
    caResPopPre(:,rmv,:)=[];
    caResPopPost(:,rmv,:)=[];
end

performanceChance = 100/length(idx);

sigBinResPre = zeros(nBins, size(caResPopPre,2), size(caResPopPre,3));
sigBinResPost = zeros(nBins, size(caResPopPost,2), size(caResPopPost,3));
for sample = 1:nBins
    stBin = sigBin(sample);
    sigBinResPre(sample,:,:) = caResPopPre(stBin,:,:); % get response magnitudes per trial
    sigBinResPost(sample,:,:) = caResPopPost(stBin,:,:); % get response magnitudes per trial
end

% Pre decoding
clear pFFPreTime pOOPreTime pFOPreTime pOFPreTime pFFpermPreTime...
pOOpermPreTime pFOpermPreTime pOFpermPreTime dPredictFullPreTime dPredictOcclPreTime
for j = 1:nBins
    data = squeeze(sigBinResPre(j,:,:));
    [pFFPreTime(j,:), pOOPreTime(j,:), pFOPreTime(j,:), pOFPreTime(j,:),...
        pFFpermPreTime(j,:), pOOpermPreTime(j,:),pFOpermPreTime(j,:), pOFpermPreTime(j,:),~, ~, ~, ~,...
        ~, ~, ~, ~, dPredictFullPreTime(j,:,:), dPredictOcclPreTime(j,:,:)]...
        = doMuckliDecodingLDAblock2(data, trialTypes, trainFrac, nReps, nBoots, 0);
    disp(j)
end

% Post decoding
clear pFFPostTime pOOPostTime pFOPostTime pOFPostTime pFFpermPostTime...
    pOOpermPostTime pFOpermPostTime pOFpermPostTime dPredictFullPostTime dPredictOcclPostTime
for j = 1:nBins
    data = squeeze(sigBinResPost(j,:,:));
    [pFFPostTime(j,:), pOOPostTime(j,:), pFOPostTime(j,:), pOFPostTime(j,:),...
        pFFpermPostTime(j,:), pOOpermPostTime(j,:),pFOpermPostTime(j,:), pOFpermPostTime(j,:),~, ~, ~, ~,...
        ~, ~, ~, ~, dPredictFullPostTime(j,:,:), dPredictOcclPostTime(j,:,:)]...
        = doMuckliDecodingLDAblock2(data, trialTypes, trainFrac, nReps, nBoots, 0);
    disp(j)
end

%% plotting shadederrorbar

save_fig = false;

clear pvalFFPre pvalOOPre pvalFOPre pvalOFPre pvalFFPost pvalOOPost pvalFOPost pvalOFPost
for i = 1:size(pFFPreTime,1)
    % calculate significance over trace using pareto tail estimation (Paolo)
    pvalFFPre(i) = paretoEst(pFFpermPreTime(i,:)/100, mean(pFFPreTime(i,:)/100,2));
    pvalOOPre(i) = paretoEst(pOOpermPreTime(i,:)/100, mean(pOOPreTime(i,:)/100,2));
    pvalFOPre(i) = paretoEst(pFOpermPreTime(i,:)/100, mean(pFOPreTime(i,:)/100,2));
    pvalOFPre(i) = paretoEst(pOFpermPreTime(i,:)/100, mean(pOFPreTime(i,:)/100,2));
    pvalFFPost(i) = paretoEst(pFFpermPostTime(i,:)/100, mean(pFFPostTime(i,:)/100,2));
    pvalOOPost(i) = paretoEst(pOOpermPostTime(i,:)/100, mean(pOOPostTime(i,:)/100,2));
    pvalFOPost(i) = paretoEst(pFOpermPostTime(i,:)/100, mean(pFOPostTime(i,:)/100,2));
    pvalOFPost(i) = paretoEst(pOFpermPostTime(i,:)/100, mean(pOFPostTime(i,:)/100,2));
end

pFFPreTimeSign = zeros(size(pFFPreTime,1),1); pFFPreTimeSign(pvalFFPre<0.05)=1;
pOOPreTimeSign = zeros(size(pFFPreTime,1),1); pOOPreTimeSign(pvalOOPre<0.05)=1;
pFOPreTimeSign = zeros(size(pFFPreTime,1),1); pFOPreTimeSign(pvalFOPre<0.05)=1;
pOFPreTimeSign = zeros(size(pFFPreTime,1),1); pOFPreTimeSign(pvalOFPre<0.05)=1;
pFFPostTimeSign = zeros(size(pFFPostTime,1),1); pFFPostTimeSign(pvalFFPost<0.05)=1;
pOOPostTimeSign = zeros(size(pFFPostTime,1),1); pOOPostTimeSign(pvalOOPost<0.05)=1;
pFOPostTimeSign = zeros(size(pFFPostTime,1),1); pFOPostTimeSign(pvalFOPost<0.05)=1;
pOFPostTimeSign = zeros(size(pFFPostTime,1),1); pOFPostTimeSign(pvalOFPost<0.05)=1;

% signSz = 1;
% topSign1 = 110;
% botSign1 = 105;
% topSign2 = 38;
% botSign2 = 36;

signSz = 1;
topSign1 = 110;
botSign1 = 105;
if nfiles == 6
    topSign2 = 57;
    botSign2 = 55;
elseif nfiles == 5
    topSign2 = 78;
    botSign2 = 74;
end

figure('Position', [104         223        1656         627])
clear s
s(1) = subplot(2,4,1); yline(performanceChance), hold on
shadedErrorBar(vecAx,mean(pFFPreTime,2)...
    ,std(pFFPreTime,0,2)/sqrt(size(pFFPreTime,2)), 'lineProps', 'b'); hold on
shadedErrorBar(vecAx,mean(pOOPreTime,2)...
    ,std(pOOPreTime,0,2)/sqrt(size(pOOPreTime,2)), 'lineProps', 'r'); hold on, xline(0)
plot(vecAx(find(pOOPreTimeSign)), ones(sum(pOOPreTimeSign),1)+botSign1, 'square', 'MarkerSize', signSz, 'MarkerFaceColor',[1 0 0])
plot(vecAx(find(pFFPreTimeSign)), ones(sum(pFFPreTimeSign),1)+topSign1, 'square', 'MarkerSize', signSz, 'MarkerFaceColor',[0 0 1])
legend({'','','', 'FF','','','','OO'}, 'Location', 'best'), legend boxoff, title('Pre cross val'), xlabel('Time (s)'), ylabel('Decoding accuracy (%)'), 
s(2) = subplot(2,4,2); yline(performanceChance), hold on
shadedErrorBar(vecAx,mean(pFOPreTime,2)...
    ,std(pFOPreTime,0,2)/sqrt(size(pFOPreTime,2)), 'lineProps', 'b'); hold on
shadedErrorBar(vecAx,mean(pOFPreTime,2)...
    ,std(pOFPreTime,0,2)/sqrt(size(pOFPreTime,2)), 'lineProps', 'r'); hold on, xline(0)
plot(vecAx(find(pOFPreTimeSign)), ones(sum(pOFPreTimeSign),1)+botSign2, 'square', 'MarkerSize', signSz, 'MarkerFaceColor',[1 0 0])
plot(vecAx(find(pFOPreTimeSign)), ones(sum(pFOPreTimeSign),1)+topSign2, 'square', 'MarkerSize', signSz, 'MarkerFaceColor',[0 0 1])
legend({'','','', 'FO','','','','OF'}, 'Location', 'best'), legend boxoff,title('Pre cross decoding'), xlabel('Time (s)'), ylabel('Decoding accuracy (%)'), 
s(3) = subplot(2,4,3); yline(performanceChance), hold on
shadedErrorBar(vecAx,mean(pFFPostTime,2)...
    ,std(pFFPostTime,0,2)/sqrt(size(pFFPostTime,2)), 'lineProps', 'b'); hold on
shadedErrorBar(vecAx,mean(pOOPostTime,2)...
    ,std(pOOPostTime,0,2)/sqrt(size(pOOPostTime,2)), 'lineProps', 'r'); hold on, xline(0)
plot(vecAx(find(pOOPostTimeSign)), ones(sum(pOOPostTimeSign),1)+botSign1, 'square', 'MarkerSize', signSz, 'MarkerFaceColor',[1 0 0])
plot(vecAx(find(pFFPostTimeSign)), ones(sum(pFFPostTimeSign),1)+topSign1, 'square', 'MarkerSize', signSz, 'MarkerFaceColor',[0 0 1])
title('Post cross val'), xlabel('Time (s)'), ylabel('Decoding accuracy (%)'), 
s(4) = subplot(2,4,4); yline(performanceChance), hold on
shadedErrorBar(vecAx,mean(pFOPostTime,2)...
    ,std(pFOPostTime,0,2)/sqrt(size(pFOPostTime,2)), 'lineProps', 'b'); hold on
shadedErrorBar(vecAx,mean(pOFPostTime,2)...
    ,std(pOFPostTime,0,2)/sqrt(size(pOFPostTime,2)), 'lineProps', 'r'); hold on, xline(0)
plot(vecAx(find(pOFPostTimeSign)), ones(sum(pOFPostTimeSign),1)+botSign2, 'square', 'MarkerSize', signSz, 'MarkerFaceColor',[1 0 0])
plot(vecAx(find(pFOPostTimeSign)), ones(sum(pFOPostTimeSign),1)+topSign2, 'square', 'MarkerSize', signSz, 'MarkerFaceColor',[0 0 1])
title('Post cross decoding'), xlabel('Time (s)'), ylabel('Decoding accuracy (%)'), 
s(5) = subplot(2,4,5); yline(performanceChance), hold on
shadedErrorBar(vecAx,mean(pFFpermPreTime,2)...
    ,std(pFFpermPreTime,0,2)/sqrt(size(pFFpermPreTime,2)), 'lineProps', 'b'); hold on
shadedErrorBar(vecAx,mean(pOOpermPreTime,2)...
    ,std(pOOpermPreTime,0,2)/sqrt(size(pOOpermPreTime,2)), 'lineProps', 'r'); hold on, xline(0)
xlabel('Time (s)'), ylabel('Decoding accuracy (%)'), 
s(6) = subplot(2,4,6); yline(performanceChance), hold on
shadedErrorBar(vecAx,mean(pFOpermPreTime,2)...
    ,std(pFOpermPreTime,0,2)/sqrt(size(pFOpermPreTime,2)), 'lineProps', 'b'); hold on
shadedErrorBar(vecAx,mean(pOFpermPreTime,2)...
    ,std(pOFpermPreTime,0,2)/sqrt(size(pOFpermPreTime,2)), 'lineProps', 'r'); hold on, xline(0)
xlabel('Time (s)'), ylabel('Decoding accuracy (%)'), 
s(7) = subplot(2,4,7); yline(performanceChance), hold on
shadedErrorBar(vecAx,mean(pFFpermPostTime,2)...
    ,std(pFFpermPostTime,0,2)/sqrt(size(pFFpermPostTime,2)), 'lineProps', 'b'); hold on
shadedErrorBar(vecAx,mean(pOOpermPostTime,2)...
    ,std(pOOpermPostTime,0,2)/sqrt(size(pOOpermPostTime,2)), 'lineProps', 'r'); hold on, xline(0)
xlabel('Time (s)'), ylabel('Decoding accuracy (%)'), 
s(8) = subplot(2,4,8); yline(performanceChance), hold on
shadedErrorBar(vecAx,mean(pFOpermPostTime,2)...
    ,std(pFOpermPostTime,0,2)/sqrt(size(pFOpermPostTime,2)), 'lineProps', 'b'); hold on
shadedErrorBar(vecAx,mean(pOFpermPostTime,2)...
    ,std(pOFpermPostTime,0,2)/sqrt(size(pOFpermPostTime,2)), 'lineProps', 'r'); hold on, xline(0)
xlabel('Time (s)'), ylabel('Decoding accuracy (%)'), 

% for g = 1:length(s)
%     s(g).YLim = [0 140]; s(g).YTick = 0:20:140; s(g).XLim = [-1 3]; s(g).XTick = -1:3;
% end
for g = [1 3 5 6 7 8]
    s(g).YLim = [0 120]; s(g).YTick = 0:20:120; s(g).XLim = [-1 3]; s(g).XTick = -1:3;
end
if nfiles == 6
for g = [2 4]
    s(g).YLim = [10 60]; s(g).YTick = 10:10:60; s(g).XLim = [-1 3]; s(g).XTick = -1:3;
end
elseif nfiles == 5
for g = [2 4]
    s(g).YLim = [10 80]; s(g).YTick = 10:10:80; s(g).XLim = [-1 3]; s(g).XTick = -1:3;
end
end
for g = 1:length(s)
    s(g).TickDir = 'in';
end

if save_fig
    func_save_fig('L23_timeDecoding_prepost_4imgs')
    func_save_fig('L5_timeDecoding_prepost_4imgs')

    func_save_fig('L23_timeDecoding_prepost_6imgs')
    func_save_fig('L5_timeDecoding_prepost_6imgs')
end


%% selectivity/sparsity

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




% based on one way anova method from Paolo, which does a one way anova
% per neurons across images
imgFullResPrePop = cat(4,datastructPre(:).imgFullRes);
imgOcclResPrePop = cat(4,datastructPre(:).imgOcclRes);
imgFullResPostPop = cat(4,datastructPost(:).imgFullRes);
imgOcclResPostPop = cat(4,datastructPost(:).imgOcclRes);
clear fFullPre fOcclPre fFullPost fOcclPost
for j = 1:size(imgFullResPrePop,4)
    [~, F] = anova1(squeeze(mean(imgFullResPrePop(vecAxSt,famIdx,:,j))-mean(imgFullResPrePop(vecAxSp,famIdx,:,j)))',[],'off');
    fFullPre(j) = F{2,5};
    [~, F] = anova1(squeeze(mean(imgOcclResPrePop(vecAxSt,famIdx,:,j))-mean(imgOcclResPrePop(vecAxSp,famIdx,:,j)))',[],'off');
    fOcclPre(j) = F{2,5};
end
for j = 1:size(imgFullResPostPop,4)
    [~, F] = anova1(squeeze(mean(imgFullResPostPop(vecAxSt,famIdx,:,j))-mean(imgFullResPostPop(vecAxSp,famIdx,:,j)))',[],'off');
    fFullPost(j) = F{2,5};
    [~, F] = anova1(squeeze(mean(imgOcclResPostPop(vecAxSt,famIdx,:,j))-mean(imgOcclResPostPop(vecAxSp,famIdx,:,j)))',[],'off');
    fOcclPost(j) = F{2,5};

end
figure('Position', [1067         436         204         407])
scatter([1 2 4 5],[mean(fFullPre) mean(fFullPost) mean(fOcclPre) mean(fOcclPost)], 30, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2 4 5],[mean(fFullPre) mean(fFullPost) mean(fOcclPre) mean(fOcclPost)], ...
    [calcSem(fFullPre) calcSem(fFullPost) calcSem(fOcclPre) calcSem(fOcclPost)] ...
    ,[calcSem(fFullPre) calcSem(fFullPost) calcSem(fOcclPre) calcSem(fOcclPost)]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 6]), ylabel('Anova F-stat'), xticks([1 2 4 5])
xticklabels({'Full Pre', 'Full Post','Occl Pre', 'Occl Post'}), xtickangle(45), 

if save_fig
%     func_save_fig('L23_AnovaFstat')
    func_save_fig('L5_AnovaFstat')
end

% based on variance from Paolo
% Variance: So, Ri is the average variance across repetitions. 
% So you compute R_all(i) = var(R(i,:)) for each stimulus i (so its the variance across the 20 reps), 
% and then you average R_all (which is 6 numbers) to get Ri. 
% And then you compute the total variance by just considering all the data (so 20 reps x 6 stims, not just 6 values).
%  variance = ( var(R(:)) - sum(var(Ri)) ) / var(R(:))

clear varValFullPre varValOcclPre varValFullPost varValOcclPost
for j = 1:size(imgFullResPrePop,4)
    temp = squeeze(mean(imgFullResPrePop(vecAxSt,:,:,j))-mean(imgFullResPrePop(vecAxSp,:,:,j)));
    varValFullPre(j) = calcVar(temp);
    temp = squeeze(mean(imgOcclResPrePop(vecAxSt,:,:,j))-mean(imgOcclResPrePop(vecAxSp,:,:,j)));
    varValOcclPre(j) = calcVar(temp);
end
for j = 1:size(imgFullResPostPop,4)
    temp = squeeze(mean(imgFullResPostPop(vecAxSt,:,:,j))-mean(imgFullResPostPop(vecAxSp,:,:,j)));
    varValFullPost(j) = calcVar(temp);
    temp = squeeze(mean(imgOcclResPostPop(vecAxSt,:,:,j))-mean(imgOcclResPostPop(vecAxSp,:,:,j)));
    varValOcclPost(j) = calcVar(temp);
end
figure('Position', [1067         436         204         407])
scatter([1 2 4 5],[mean(varValFullPre) mean(varValFullPost) mean(varValOcclPre) mean(varValOcclPost)], 30, 'k', 'filled', 'LineWidth', 2), hold on
er = errorbar([1 2 4 5],[mean(varValFullPre) mean(varValFullPost) mean(varValOcclPre) mean(varValOcclPost)], ...
    [calcSem(varValFullPre) calcSem(varValFullPost) calcSem(varValOcclPre) calcSem(varValOcclPost)] ...
    ,[calcSem(varValFullPre) calcSem(varValFullPost) calcSem(varValOcclPre) calcSem(varValOcclPost)]);    
er.Color = [0 0 0]; er.LineStyle = 'none'; er.LineWidth = 2; er.CapSize = 0;
xlim([0 6]), ylabel('Variance'), xticks([1 2 4 5])
xticklabels({'Full Pre', 'Full Post','Occl Pre', 'Occl Post'}), xtickangle(45), 

if save_fig
%     func_save_fig('L23_Variance')
    func_save_fig('L5_Variance')
end

mouseIDPre = [];
mouseIDPost = [];
mouseIDTask = [];
for i = 1:nfiles
    % prepare some data for linear mixed model effect
    mouseIDPre = [mouseIDPre zeros(1,length(datastructPre(i).scatFull))+i];
    mouseIDPost = [mouseIDPost zeros(1,length(datastructPost(i).scatFull))+i];
    mouseIDTask = [mouseIDTask zeros(1,length(datastructActiveRes(i).scatFull))+i];
end

% full fam LMEM
data = cat(2, varValFullPre,varValFullPost)';
mouseID = categorical(cat(2, mouseIDPre,mouseIDPost))';
condition = categorical(cat(1, ones(length(mouseIDPre),1),ones(length(mouseIDPost),1)+1));
clear statTbl, statTbl = table(data, mouseID, condition);
lmeFullFam = fitlme(statTbl, 'data ~ condition + (1|mouseID)', 'CheckHessian', 1,'FitMethod', 'REML', 'StartMethod', 'random');
statsFullFam = anova(lmeFullFam,'DFMethod','Satterthwaite');
statTblFull = makeStatTbl(lmeFullFam);

%%

% ==================== Local helper functions ====================
function plot_fh_sh_block(condCell, colorChar)
% Expects condCell = { name, [2 x N]; ... }
% Plots pairs (fh, sh) with mean ± SEM; clusters spaced by +3 on x-axis
    k = size(condCell,1);
    for ii = 1:k
        xpair = [1 2] + 3*(ii-1);
        dat   = condCell{ii,2};
        if isempty(dat) || size(dat,2) < 2, continue; end

        fhv = dat(1,:); shv = dat(2,:);
        mFH = nanmean(fhv); mSH = nanmean(shv);
        eFH = calcSem(fhv); eSH = calcSem(shv);

        scatter(xpair(1), mFH, 45, colorChar, 'filled', 'LineWidth', 2); hold on
        er = errorbar(xpair(1), mFH, eFH, 'k', 'LineStyle','none', 'LineWidth', 2, 'CapSize', 0); %#ok<NASGU>
        scatter(xpair(2), mSH, 45, colorChar, 'filled', 'LineWidth', 2);
        er = errorbar(xpair(2), mSH, eSH, 'k', 'LineStyle','none', 'LineWidth', 2, 'CapSize', 0); %#ok<NASGU>
        plot(xpair, [mFH mSH], colorChar, 'LineWidth', 1.5)

        % Label cluster with condition name (optional: shorten)
        if contains(condCell{ii,1}, 'Pre'),  lab = 'Pre';
        elseif contains(condCell{ii,1}, 'Post'), lab = 'Post';
        else, lab = 'Task'; end
        if contains(condCell{ii,1}, 'Fam'), lab = [lab '-Fam']; end
        if contains(condCell{ii,1}, 'Nov'), lab = [lab '-Nov']; end
        text(mean(xpair), min([mFH mSH]) - 0.02, lab, 'HorizontalAlignment','center', 'FontSize', 9)
    end
    xlim([0 3*k+0.5])
    ylabel('Response'); xlabel('Half'); box off
end

function annotate_fh_sh_stars(ax, namesInOrder, statsTbl)
% Adds significance stars above each pair using adaptStatsTTest_fhsh
    axes(ax); %#ok<LAXES>
    k = numel(namesInOrder);
    for ii = 1:k
        condName = string(namesInOrder{ii});
        row = statsTbl(strcmp(statsTbl.Condition, condName), :);
        if isempty(row) || isnan(row.pValue), continue; end

        % x positions for this pair
        xpair = [1 2] + 3*(ii-1);

        % y position slightly above higher of the two points at those x
        ylimCurr = ylim;
        yvals = get_points_at_x(xpair); % visual query (see helper below)
        yTop = max(yvals) + 0.02*(ylimCurr(2)-ylimCurr(1));

        % stars by p-value
        p = row.pValue;
        if p < 1e-3, stars = '***';
        elseif p < 1e-2, stars = '**';
        elseif p < 5e-2, stars = '*';
        else, stars = ''; end
        if ~isempty(stars)
            text(mean(xpair), yTop, stars, 'HorizontalAlignment','center', 'FontSize', 12, 'FontWeight','bold')
        end
    end
end

function yvals = get_points_at_x(xpair)
% crude readback of last plotted means at xpair: finds scatter objects
    yvals = [NaN NaN];
    h = findobj(gca,'Type','Scatter');
    if isempty(h), return; end
    % Take last plotted for the cluster (works with code order above)
    for i = 1:2
        for hh = h.'
            X = get(hh,'XData'); Y = get(hh,'YData');
            idx = find(abs(X - xpair(i)) < 1e-9, 1, 'last');
            if ~isempty(idx), yvals(i) = Y(idx); break; end
        end
    end
end
