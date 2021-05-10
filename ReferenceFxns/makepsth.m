function psth = makepsth(spikes,binEdges,gaussWidth)

psth = zeros(length(binEdges)-1,1);

for spInd = 1:length(spikes)
    psth  = psth + diff(normcdf(binEdges, spikes(spInd),gaussWidth))';
end