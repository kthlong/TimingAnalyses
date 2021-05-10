function [aligneddata,ncoincspikes] = alignspikes(data,rates,minT,maxT,method)

slideVals = [-.05:.001:.05];
repDim = 3;

[~,highreprate] = max(rates,[],repDim);

maskmat = ones(size(data,1),size(data,2),size(data,3))==0;

for cellInd = 1:size(data,1)
    for tInd = 1:size(data,2)
        refspiketrain(cellInd,tInd) = data(cellInd,tInd,highreprate(cellInd,tInd));
        maskmat(cellInd,tInd,highreprate) = true;
    end
end

for slideInd = 1:length(slideVals)
    presentrep(:,:,:,slideInd) = cellfun(@(x) x+slideVals(slideInd),data,'uniformoutput',0);
end

if strcmp(method,'ncoincspikes')
    ncoincspikesmat = cellfun(@(x,y) coincspikes(x,y),repmat(refspiketrain,1,1,size(data,3),length(slideVals)),presentrep);

    [ncoincspikes,offsets] = max(ncoincspikesmat,[],repDim+1);
    ncoincspikes(maskmat) = nan;
    offsets(isnan(ncoincspikes)) = find(slideVals==0);

   
    
elseif strcmp(method,'spikeDist')
    spikeDistMat = cellfun(@(x,y) minispikedist(x,y),repmat(refspiketrain,1,1,size(data,3),length(slideVals)),presentrep);
    [~,offsets] = min(spikeDistMat,[],repDim+1);
    ncoincspikes = 'na';
end

offsets(maskmat) = find(slideVals==0);
offsets_vals = slideVals(offsets);
aligneddata = cellfun(@(x,y) x+y,squeeze(data),squeeze(num2cell(offsets_vals)),'uniformoutput',0);



function ncoincspikes = coincspikes(repref,presentrep)
    bintrainref = histc(repref,minT:.002:maxT);
    bintraincomp = histc(presentrep,minT:.002:maxT);
    jointspikes = bintrainref+bintraincomp;
    ncoincspikes = length(find(jointspikes>bintrainref & jointspikes>bintraincomp));
    if ncoincspikes == 0
        ncoincspikes = nan; 
    end
end


function dist = minispikedist(spike1_tmp,spike2_tmp)
     dists_tmp = spkdl([spike1_tmp; spike2_tmp],[1 length(spike1_tmp)+1],[length(spike1_tmp) length(spike1_tmp)+length(spike2_tmp)],1/.001);
     dist = dists_tmp(2);
end

end