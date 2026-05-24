%% movieDecoding runs Jorrit Montijn's decoder on calcium data from movie presentations
%   Based on runPlaceDecodingSpeedMult20190603, uses only the decoding part
%
%	Version History:
%   2019-08-21  adapted from Jorrit's scripts by Koen Seignette

%% load data
[strFile , strPath] = uigetfile('*_SPSIG.mat');   
load([strPath strFile], 'sig', 'den', 'decon', 'freq');
% load([strPath strFile(1:end-19) '_eye.mat'], 'eye', 'time');
load([strPath strFile(1:end-10) '.mat']);
load([strPath strFile(1:end-19) '_log.mat']);
% load([strPath strFile(1:end-19) '_quadrature.mat']);

%% prepare
% movfreq = Parameters.movies(1).framerate;
movfreq = 29.79;
numofmovframes = floor(Parameters.time*movfreq);
signew = sig; %backup
sig = sig'; %transpose for decoder
mov = zeros(1, length(sig)); %create movie vector for decoder

for i = 1:length(info.frame)
    mov(info.frame(i):info.frame(i)+numofmovframes-1) = 1:numofmovframes;
end


%% decode
%bin location data
dblBinStep = 0.01;
vecBins = (-dblBinStep/2):dblBinStep:1;
vecX = matOutX(:,4);
vecResps = vecY;
[vecCounts,vecMeans,vecSDs,cellVals,cellIDs] = makeBins(vecX,vecX,vecBins);
intFirstRemBin = find(vecBins>0.5);%find(vecCounts < 100,1);
cellRems = cellIDs(intFirstRemBin:end);
cellBins = cellIDs(1:(intFirstRemBin-1));
intLocBins = numel(cellBins)+1;
vecLocBin = nan(size(vecX));
for intLocBin=1:(intLocBins-1)
	vecLocBin(cellBins{intLocBin}) = intLocBin;
end
vecLocBin(cell2vec(cellRems)) = intLocBins;

%make figure
figure;
for intDecodeType=1%:2
	if intDecodeType==1
		matData=matAct;
		strTitle = strTitFile;
	else
		matData=matActClean;
		strTitle = 'gnm-subtracted';
	end
	%decode
	[dblPerformance,vecDecodedIndexCV,matMahalDistsCV,dblMeanErrorDegs,matConfusion] = ...
		doCrossValidatedDecodingMD(matData,vecLocBin,[]);
	
	% plot
	%{
	subplot(2,2,1+(intDecodeType-1)*2);
	imagesc(matConfusion);colormap(hot)
	ylabel('Decoded location');
	xlabel('Real location');
	title(sprintf('Raw pdf, %s',strTitle),'interpreter','none');
	colorbar;
	fixfig;
	grid off;
	%}
	%normalize
	vecObservedCounts = accumarray(vecLocBin,ones(size(vecLocBin)));
	matConfusionNorm = bsxfun(@rdivide,matConfusion,vecObservedCounts');
	vecConfCount = sum(matConfusionNorm,1); %should all be one
	matCN2 = matConfusionNorm;
	matCN2(end,end) = 0;
	dblMax = max(matCN2(:));
	
	%calculate MSE
	intBins = size(matConfusion,1);
	matErrDist = (abs((1:intBins) - 1 * (1:intBins)'));
	matError = matErrDist .* matConfusionNorm;
	dblMSE = mean(matError(:).^2);
	
	%subplot(2,2,2+(intDecodeType-1)*2);
	imagesc(matConfusionNorm,[0 dblMax]);colormap(hot)
	ylabel('Decoded location');
	xlabel('Real location');
	title(sprintf('MSE=%.3f, Norm pdf, %s',dblMSE,strTitle),'interpreter','none');
	colorbar;
	fixfig;
	grid off;
end
pause(0.5);

%save fig
%drawnow;
%jFig = get(handle(gcf), 'JavaFrame');
%jFig.setMaximized(true);
%figure(gcf);
%drawnow;
strFig = ['LocationDecoding_' strFile '_' getDate];
pause(0.5);
%save
export_fig([strFigPath strFig '.tif']);
export_fig([strFigPath strFig '.pdf']);

%% save data
save(['workspace_' strFile '_' getDate '.mat']);