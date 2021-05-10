function TVPoisson = maketvpoisson(data,jitters,minT,maxT)

for binInd = 1:length(jitters)
    binSize = jitters(binInd);
    thispsth = makemeanpsth(data,binSize,minT,maxT);
    meanpsth = squeeze(nanmean(thispsth,3)./binSize);
    for cellInd = 1:size(data,1)
        for tInd = 1:size(data,2)
            psthcells{cellInd, tInd} = squeeze(meanpsth(cellInd,tInd,:));
        end
    end
    
    tvpspikes = cellfun(@(x) poissonspikes_timevaried(x,binSize,minT,maxT), repmat(psthcells,1,1,size(data,3)),'uniformoutput',0);
    
    jitteredrates = cellfun(@(x) length(x(x>=minT & x<=maxT)),tvpspikes);
    repRates = cellfun(@(x) length(x(x>=minT & x<=maxT)),data);
    
    TVPoisson(:,:,:,binInd) = cellfun(@(x,y,z) ratematchjitter(x,y,z,minT,maxT),tvpspikes,num2cell(jitteredrates),num2cell(repRates),'uniformoutput',0);
%     TVPoisson(:,:,:,binInd) = tvpspikes;
end

TVPoisson = cellfun(@(x) makecolumn(x), TVPoisson,'uniformoutput',0);

% functions
function newspikes = ratematchjitter(spiketrain,presentrate,ratetomatch,minT,maxT)
deltaRate = presentrate - ratetomatch;
if deltaRate < 0
    newspikes = sort([spiketrain' datasample(minT:.001:maxT,abs(deltaRate))]);
elseif deltaRate > 0
    spInd = find(spiketrain>=minT & spiketrain<=maxT);
    sptodelete = datasample(spInd,deltaRate,'replace',false);
    spiketrain(sptodelete) = [];
    newspikes = spiketrain;
else
    newspikes = spiketrain;
end
end


end