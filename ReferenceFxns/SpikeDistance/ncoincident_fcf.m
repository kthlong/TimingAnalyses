function nshared = ncoincident_fcf(spiketrain1,spiketrain2,windowlength)
%fcf = for cellfun

edges = [0:.004:windowlength];

hist1 = histc(spiketrain1,edges);
hist2 = histc(spiketrain2,edges);
hist1(cellfun(@isempty,hist1)) = {nan};
hist2(cellfun(@isempty,hist2)) = {nan};

nshared = cellfun(@(x,y) countspikes(x,y),hist1,hist2);
