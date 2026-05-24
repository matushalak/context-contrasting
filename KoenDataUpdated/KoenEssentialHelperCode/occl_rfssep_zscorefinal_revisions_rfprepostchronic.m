%% or just load files
clear all

warning on

load('D:\2Pdata\Koen\Muckli\Analyses\Kazu\revisions\dataPrePostRevisionsComplete_occlGray_processed.mat')
nfiles = length(datastructPre);

%% Initialize and organise data
clearvars -except datastructPre datastructPost filenamesPre...
    filenamesPost filepathsPre filepathsPost nfiles doDecoding...
    imgNrs nImgs performanceChance nTrials nReps nBoots trainFrac 

imgNrs = [1 2 3 4 5 6]; % image nrs to include, all images
nImgs = length(imgNrs);
performanceChance = 100/nImgs;
nTrials = 20; % nr of trials shown per image
trainFrac = 0.5; % on what fraction would you like to train the decoder (0.5 is good)
rfDistVec = 2; % Minimum distance away from occluder edge
vecAx = datastructPre(1).Res.ax;
vecAxSp = vecAx<0; % spontaneous activity window
vecAxSt = vecAx>0.2 & vecAx<1; % stim window
vecAxRunSt = vecAx>0.2 & vecAx<1; % stim window
alphaVal = 0.9999999; % significance value for cells to be included
rsqThresh = 0.33; % 0.33 for L2/3
snrThresh = 4; % snr threshold for RF
useSpikingData = 0; % deconvolved (1) or df/f (0)
doZscore = true; % in case you want to work with zscored data instead of dff
smoothDecoding = false;
regressRun = false; % regress out running? Only for CaSigCorrected, not for spikes
runNan = false;
runThres = 2;

for i = 1:nfiles

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

    % correct for lack of 31 hz in 1 mouse
    if strcmp(filenamesPre(i), 'Fctwente_20241007_005_normcorr_SPSIG_Res.mat') && size(datastructPre(i).Res.CaSigCorrected,1)<100
        sigs = datastructPre(i).Res.CaSigCorrected;
        runs = datastructPre(i).Res.speed;
        % Define the original time vector (size: 1x63)
        original_time = datastructPre(i).Res.ax;

        % Define the new time vector
        new_time = datastructPost(1).Res.ax;

        % Initialize the new matrix to hold the interpolated data
        new_sigs = zeros(length(new_time), size(sigs,2), size(sigs,3));
        new_runs = zeros(length(new_time), size(sigs,2));
        % Loop over each column and plane (240x403) to interpolate along the first dimension (time)
        for k = 1:size(sigs,2)
            for j = 1:size(sigs,3)
                % Perform 1D interpolation along the first dimension
                new_sigs(:, k, j) = interp1(original_time, sigs(:, k, j), new_time, 'linear');
            end
                % Perform 1D interpolation along the first dimension
                new_runs(:, k) = interp1(original_time, runs(:, k), new_time, 'linear');
        end 
        new_sigs(1,:,:) = new_sigs(2,:,:);
        new_runs(1,:) = new_runs(2,:);
        datastructPre(i).Res.CaSigCorrected = new_sigs;
        datastructPre(i).Res.Speed = new_runs;
    end

    % correct for lack of 31 hz in 1 mouse
    if size(datastructPre(i).Res.speed,1)<100
        runs = datastructPre(i).Res.speed;
        % Define the original time vector (size: 1x63)
        original_time = datastructPre(i).Res.ax;

        % Define the new time vector
        new_time = datastructPost(1).Res.ax;

        % Initialize the new matrix to hold the interpolated data
        new_runs = zeros(length(new_time), size(runs,2));
        % Loop over each column and plane (240x403) to interpolate along the first dimension (time)
        for k = 1:size(runs,2)
            % Perform 1D interpolation along the first dimension
            new_runs(:, k) = interp1(original_time, runs(:, k), new_time, 'linear');
        end 
        new_runs(1,:) = new_runs(2,:);
        datastructPre(i).Res.speed = new_runs;
    end

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
   
%     % correct for lack of 31 hz in 1 mouse
%     if strcmp(filenamesPost(i), 'Fctwente_20241007_005_normcorr_SPSIG_Res.mat') && size(datastructPost(i).Res.CaSigCorrected,1)<100
%         Res = datastructPre(i).Res;
% 
%     end

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
    % 
    % % pupil position data
    % pupilX = zscore(datastructPost(i).pupil{1, 1}.com(:,1));
    % pupilY = zscore(datastructPost(i).pupil{1, 1}.com(:,2));
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
        datastructPost(i).runTrials = runTrials;
%     end
    disp(i)
end


%% in case you calculated responses to 6 images, plot pre vs post (separate by image type)

vecAx = datastructPost(1).Res.ax;
vecAxSp = vecAx<0; % spontaneous activity window
vecAxSt = vecAx>0.2 & vecAx<1; % stim window

save_fig = false;

famIdx = [1 2 4 5];
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
xlabel('Time (s)'), ylabel('Response'), xticks(-1:1:3), xlim([-1 3]), title('Pre Fam'), figClean
t(2) = subplot(2,5,2);
shadedErrorBar(vecAx,nanmean(imgFullResFamPostBsl,2)...
    ,nanstd(imgFullResFamPostBsl,0,2)/sqrt(size(imgFullResFamPostBsl,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAx,nanmean(imgOcclResFamPostBsl,2)...
    ,nanstd(imgOcclResFamPostBsl,0,2)/sqrt(size(imgOcclResFamPostBsl,2)), 'lineProps', 'r');
xlabel('Time (s)'), title('Post Fam'), figClean
t(3) = subplot(2,5,3);
shadedErrorBar(vecAx,nanmean(imgFullResNovPreBsl,2)...
    ,nanstd(imgFullResNovPreBsl,0,2)/sqrt(size(imgFullResNovPreBsl,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAx,nanmean(imgOcclResNovPreBsl,2)...
    ,nanstd(imgOcclResNovPreBsl,0,2)/sqrt(size(imgOcclResNovPreBsl,2)), 'lineProps', 'r');
xlabel('Time (s)'), xticks(-1:1:3), xlim([-1 3]), title('Pre Nov'), figClean
t(4) = subplot(2,5,4);
shadedErrorBar(vecAx,nanmean(imgFullResNovPostBsl,2)...
    ,nanstd(imgFullResNovPostBsl,0,2)/sqrt(size(imgFullResNovPostBsl,2)), 'lineProps', 'k'); hold on
shadedErrorBar(vecAx,nanmean(imgOcclResNovPostBsl,2)...
    ,nanstd(imgOcclResNovPostBsl,0,2)/sqrt(size(imgOcclResNovPostBsl,2)), 'lineProps', 'r');
xlabel('Time (s)'), title('Post Nov'), figClean
% scatters
s(1) = subplot(2,5,6);
scatter(scatFullFamPopPreCut, scatOcclFamPopPreCut, sz, cPre, 'filled'); refline(1), ylabel('Occl'), xlabel('Full'),figClean
s(2) = subplot(2,5,7);
scatter(scatFullFamPopPostCut,scatOcclFamPopPostCut , sz, cPost, 'filled'); refline(1), figClean
s(3) = subplot(2,5,8);
scatter(scatFullNovPopPreCut, scatOcclNovPopPreCut, sz, cPre, 'filled'); refline(1), figClean
s(4) = subplot(2,5,9);
scatter(scatFullNovPopPostCut, scatOcclNovPopPostCut, sz, cPost, 'filled'); refline(1), figClean
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
    'PreFamOccl', 'PostFamOccl','PreNovOccl', 'PostNovOccl'}), xtickangle(45), figClean
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
    'PreFamOccl', 'PostFamOccl','PreNovOccl', 'PostNovOccl'}), xtickangle(45), figClean


% Adjusting y-axes for subplots 1-4
yMax = max([ylim(t(1)), ylim(t(2)), ylim(t(3)), ylim(t(4))]);
yMin = min([ylim(t(1)), ylim(t(2)), ylim(t(3)), ylim(t(4))]);
% set(t(1:4), 'YLim', [yMin yMax]);
% if nfiles == 6
    set(t(1:4), 'YLim', [-0.1 0.6]);
% elseif nfiles == 5
%     set(t(1:4), 'YLim', [-0.1 1]);
% end
set(t(1:4), 'XLim', [-1 3])

for j = 1:length(s)
    s(j).YLim = [-1 3]; s(j).YTick = -1:0.5:3; s(j).XLim = [-1 3]; s(j).XTick = -1:0.5:3;
end

if save_fig
    func_save_fig('traceAndScatterAndBox')
end

%
sz = 8;

figure('Position', [ 87         278        1635         551])
clear t s
s(1) = subplot(2,5,1);
scatter(scatOcclFamPopPostCut, scatOcclNovPopPostCut, sz, cPre, 'filled'); refline(1), xlabel('Occl Fam'), ylabel('Occl Nov'),figClean
s(2) = subplot(2,5,2);
scatter(scatFullFamPopPostCut, scatFullNovPopPostCut, sz, cPre, 'filled'); refline(1), xlabel('Full Fam'), ylabel('Full Nov'),figClean
s(3) = subplot(2,5,3);
scatter(scatOcclFamPopPostCut, scatFullNovPopPostCut, sz, cPre, 'filled'); refline(1), xlabel('Occl Fam'), ylabel('Full Nov'), figClean

for j = 1:length(s)
    s(j).YLim = [-1 3]; s(j).YTick = -1:1:3; s(j).XLim = [-1 3]; s(j).XTick = -1:1:3;
end

if save_fig
    func_save_fig('L23_scattersFamVsNov')
    func_save_fig('L5_scattersFamVsNov')
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
mn = -0.4; mx = 3;
set(p, 'CLim', [mn, mx]); % for L5
colormap hot
subplot(1,9,5)
axis off, colormap hot, caxis([mn mx]), colorbar

if save_fig
    func_save_fig('ImagescSeparate')
end

mouseIDPre = [];
mouseIDPost = [];
for i = 1:nfiles
    % prepare some data for linear mixed model effect
    mouseIDPre = [mouseIDPre zeros(1,length(datastructPre(i).scatFull))+i];
    mouseIDPost = [mouseIDPost zeros(1,length(datastructPost(i).scatFull))+i];
end

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


%% RF response strength vs neuron identity

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
snrOnPre = [];
snrOffPre = [];
resMatPre = [];
fwhmOnPost = [];
fwhmOffPost = [];
aziOnPost = [];
aziOffPost = [];
eleOnPost = [];
eleOffPost = [];
snrOnPost = [];
snrOffPost = [];
resMatPost = [];

for i = 1:nfiles
    fwhmOnPre = cat(2, fwhmOnPre, [datastructPre(i).info.rois(:).onFWHM]);
    fwhmOffPre = cat(2, fwhmOffPre, [datastructPre(i).info.rois(:).offFWHM]);
    val = [datastructPre(i).info.rois(:).azi];
    aziOnPre = cat(2,aziOnPre,val(1:2:end));
    aziOffPre = cat(2,aziOffPre,val(2:2:end));
    val = [datastructPre(i).info.rois(:).ele];
    eleOnPre = cat(2, eleOnPre, val(1:2:end));
    eleOffPre = cat(2, eleOffPre, val(2:2:end));
    val = [datastructPre(i).info.rois(:).SNR];
    snrOnPre = cat(2, snrOnPre, val(1:2:end));
    snrOffPre = cat(2, snrOffPre, val(2:2:end));
    snrOnPre = cat(2, snrOnPre, val(1:2:end));
    snrOffPre = cat(2, snrOffPre, val(2:2:end));
    resMatPre = cat(3, resMatPre, squeeze(datastructPre(i).info.rois(1).resMat(:,:,1,:))); % only taking ON RF here

    fwhmOnPost = cat(2, fwhmOnPost, [datastructPost(i).info.rois(:).onFWHM]);
    fwhmOffPost = cat(2, fwhmOffPost, [datastructPost(i).info.rois(:).offFWHM]);
    val = [datastructPost(i).info.rois(:).azi];
    aziOnPost = cat(2,aziOnPost,val(1:2:end));
    aziOffPost = cat(2,aziOffPost,val(2:2:end));
    val = [datastructPost(i).info.rois(:).ele];
    eleOnPost = cat(2, eleOnPost, val(1:2:end));
    eleOffPost = cat(2, eleOffPost, val(2:2:end));
    val = [datastructPost(i).info.rois(:).SNR];
    snrOnPost = cat(2, snrOnPost, val(1:2:end));
    snrOffPost = cat(2, snrOffPost, val(2:2:end));
    resMatPost = cat(3, resMatPost, squeeze(datastructPost(i).info.rois(1).resMat(:,:,1,:))); % only taking ON RF here
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

snrOnPreIncl = snrOnPre(rfInclPrePop);
snrOffPreIncl = snrOffPre(rfInclPrePop);
snrOnPostIncl = snrOnPost(rfInclPostPop);
snrOffPostIncl = snrOffPost(rfInclPostPop);
resMatPreIncl = resMatPre(:,:,rfInclPrePop);
resMatPostIncl = resMatPost(:,:,rfInclPostPop);
fwhmOffPreIncl = fwhmOffPre(rfInclPrePop);
fwhmOffPostIncl = fwhmOffPost(rfInclPostPop);
fwhmOnPreIncl = fwhmOnPre(rfInclPrePop);
fwhmOnPostIncl = fwhmOnPost(rfInclPostPop);

% what neurons to include for different response properties?
fullsFamPre = scatFullFamPopPre>0.5;
fullsNovPre = scatFullNovPopPre>0.5;
occlsFamPre = scatOcclFamPopPre>0.5;
occlsNovPre = scatOcclNovPopPre>0.5;
fullsFamPost = scatFullFamPopPost>0.5;
fullsNovPost = scatFullNovPopPost>0.5;
occlsFamPost = scatOcclFamPopPost>0.5;
occlsNovPost = scatOcclNovPopPost>0.5;

% this just looks at ON RFs:
resMatFullsFamPre = resMatPreIncl(:,:,fullsFamPre);
resMatFullsNovPre = resMatPreIncl(:,:,fullsNovPre);
resMatOcclsFamPre = resMatPreIncl(:,:,occlsFamPre);
resMatOcclsNovPre = resMatPreIncl(:,:,occlsNovPre);
resMatFullsFamPost = resMatPostIncl(:,:,fullsFamPost);
resMatFullsNovPost = resMatPostIncl(:,:,fullsNovPost);
resMatOcclsFamPost = resMatPostIncl(:,:,occlsFamPost);
resMatOcclsNovPost = resMatPostIncl(:,:,occlsNovPost);

% max response per neuron
mxFullsFamPre = max(max(resMatFullsFamPre));
mxFullsFamPost = max(max(resMatFullsFamPost));
mxFullsNovPre = max(max(resMatFullsNovPre));
mxFullsNovPost = max(max(resMatFullsNovPost));
mxOcclsFamPre = max(max(resMatOcclsFamPre));
mxOcclsFamPost = max(max(resMatOcclsFamPost));
mxOcclsNovPre = max(max(resMatOcclsNovPre));
mxOcclsNovPost = max(max(resMatOcclsNovPost));


