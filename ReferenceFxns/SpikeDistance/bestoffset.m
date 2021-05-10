function offsets = bestoffset(shortspikes,longwindowspikes,slide,length)

offsets = [0:.01:slide];

for offInd = 1:size(offsets,2)
    longstart = offsets(offInd);
    longend = longstart + length;
    longspikes = cellfun(@(x) x(x>=longstart & x<=longend)-longstart,longwindowspikes,'uniformoutput',0);
    cospikes(offInd,:,:,:,:) = ncoincident(shortspikes,longspikes,length);
end    

[~, bestoffsets] = max(cospikes,[],1);
offsets = squeeze(bestoffsets);