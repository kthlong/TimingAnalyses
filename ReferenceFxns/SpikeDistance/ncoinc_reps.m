function nshared = ncoinc_reps(reps,windowlength,nreps)
% this is just for peripheral data
% reps = squeeze(spikes(x,x,x,:))
% windowlength = 1s or something like that
% nreps = squeeze(nmask(x,x,x))

edges = [0:.004:windowlength];

    hist1 = cellfun(@(x) histc(x,edges),reps(1),'uniformoutput',0);
    hist2 = cellfun(@(x) histc(x,edges),reps(2),'uniformoutput',0);
    hist3 = cellfun(@(x) histc(x,edges),reps(3),'uniformoutput',0);
    hist4 = cellfun(@(x) histc(x,edges),reps(4),'uniformoutput',0);
    
    hist1(cellfun(@isempty,hist1)) = {nan};
    hist2(cellfun(@isempty,hist2)) = {nan};
    hist3(cellfun(@isempty,hist3)) = {nan};
    hist4(cellfun(@isempty,hist4)) = {nan};
    
    nshared12 = cellfun(@(x,y) countspikes(x,y),hist1,hist2);
    nshared13 = cellfun(@(x,y) countspikes(x,y),hist1,hist3);
    nshared23 = cellfun(@(x,y) countspikes(x,y),hist2,hist3);
    nshared14 = cellfun(@(x,y) countspikes(x,y),hist1,hist4);
    nshared24 = cellfun(@(x,y) countspikes(x,y),hist2,hist4);
    nshared34 = cellfun(@(x,y) countspikes(x,y),hist3,hist4);
    

if nreps == 2
    nshared = nshared12;
    
elseif nreps ==3
    nshared = nshared12+nshared13+nshared23;
    
elseif nreps == 4
    nshared = nshared12+nshared13+nshared23+nshared14+nshared24+nshared34;
end