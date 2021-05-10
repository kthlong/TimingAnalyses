function [ISIdistnorm,isibins] = makeISI(spikes,startTime,endTime,nbins,norm)

if nargin < 5
    norm = 1;
end

spikes2keep = spikes(spikes>=startTime & spikes<=endTime);
nspikes = length(spikes2keep);
spikeMat = repmat(spikes2keep,1,nspikes);
ISIs = spikeMat' - spikeMat;
ISIs(ISIs<=0) = nan;
ISIvec = ISIs(:);
ISIvec = ISIvec(~isnan(ISIvec));

[isidist, isibins] = histcounts(ISIvec,linspace(0,1.8,nbins));

if norm == 1
    ISIdistnorm = isidist./length(ISIvec);
else
    ISIdistnorm = isidist;
end