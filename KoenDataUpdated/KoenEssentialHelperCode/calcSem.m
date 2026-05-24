% calculates SEM using std and sample size, omitting nans and infs

function sem = calcSem(vecData)

sampSize = sum(~isnan(vecData)&(~isinf(vecData)));
vecData(isnan(vecData))=[];
vecData(isinf(vecData))=[];
stdData = std(vecData);
sem = stdData/sqrt(sampSize);

end