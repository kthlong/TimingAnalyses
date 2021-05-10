function ratematchedspikes = ratematchedjitters(data,minT,maxT,repDim,jitterVal)

allreps = 1:size(data,repDim);
repRates = cellfun(@(x) length(x(x>=minT & x<=maxT)), data);
repsPresent_logic = cellfun(@(x) ~isempty(x),data);

% Make sample data matrix
for cellInd = 1:size(data,1)
    for tInd = 1:size(data,2)
        availreps = allreps(squeeze(repsPresent_logic(cellInd,tInd,:)));
        if ~isempty(availreps)
            randomrep = datasample(availreps,1);
            sampleddata(cellInd,tInd) = data(cellInd,tInd,randomrep);
        else
            sampleddata(cellInd,tInd) = {nan};
        end
    end
end

sampleddata = repmat(sampleddata,1,1,size(data,repDim));
jittereddata = cellfun(@(x) makejitteredreps(x,jitterVal),sampleddata,'uniformoutput',0);
jitteredrates = cellfun(@(x) length(x(x>=minT & x<=maxT)),jittereddata);

ratematchedspikes = cellfun(@(x,y,z) ratematchjitter(x,y,z,minT,maxT),jittereddata,num2cell(jitteredrates),num2cell(repRates),'uniformoutput',0);




%% functions
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