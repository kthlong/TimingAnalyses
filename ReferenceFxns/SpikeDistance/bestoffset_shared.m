function shift = bestoffset_shared(shortspikes,longwindowspikes,slide,duration)

offsets = [0:.001:slide];

if iscell(shortspikes)
    shortspikes = cell2mat(shortspikes);
    longwindowspikes = cell2mat(longwindowspikes);
end

for offInd = 1:length(offsets)
    longstart = offsets(offInd);
    longend = longstart + duration;
    longspikes = longwindowspikes(longwindowspikes>=longstart & longwindowspikes<=longend)-longstart;
    cospikes(offInd,:,:,:,:) = nsharedspikes(longspikes,shortspikes,.001); 
end    

[~, bestoffsets] = max(cospikes,[],1);
shift = squeeze(bestoffsets);
shift = offsets(shift);