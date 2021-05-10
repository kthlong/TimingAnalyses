function [ISIdistnorm,isibins] = makeISI_onlyone(spikes,startTime,endTime,binWidth,norm)

if nargin < 5
    norm = 1;
end

spikes2keep = spikes(spikes>=startTime & spikes<=endTime);
nspikes = length(spikes2keep);
if nspikes <2
    ISIdistnorm = nan; isibins = nan;
else
    nbins = (endTime-startTime)/binWidth;
    ISIvec = spikes2keep(2:end)-spikes2keep(1:end-1);
    
    [isidist, isibins] = histcounts(ISIvec,linspace(0,endTime-startTime,nbins));
    
    if norm == 1
        ISIdistnorm = isidist./length(ISIvec);
    else
        ISIdistnorm = isidist;
    end
end




end