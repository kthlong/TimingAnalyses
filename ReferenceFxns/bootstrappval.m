function [pvalue, meanpval] = bootstrappval(realdata,modeldata,iterations)

realdata = realdata(:);
realdata = realdata(~isnan(realdata));
if ~(isempty(realdata) | sum(realdata) == 0)
modeldata = modeldata(:);
modeldata = modeldata(~isnan(modeldata));


realisless = datasample(realdata,iterations,'replace',true) > datasample(modeldata,iterations,'replace',true);
for itInd = 1:iterations
    meanisless(itInd) = nanmean(realdata) > nanmean(datasample(modeldata,10));
end


pvalue = nanmean(realisless);
meanpval = nanmean(meanisless);

else
    pvalue = nan;
    meanpval = nan;
end
end