function nshared = ncoincident(spiketrain1,spiketrain2,windowlength)

edges = [0:.004:windowlength];

hist1 = cellfun(@(x) histc(x,edges),spiketrain1,'uniformoutput',0);
hist2 = cellfun(@(x) histc(x,edges),spiketrain2,'uniformoutput',0);
hist1(cellfun(@isempty,hist1)) = {nan};
hist2(cellfun(@isempty,hist2)) = {nan};

nshared = cellfun(@(x,y) countspikes(x,y),hist1,hist2);
