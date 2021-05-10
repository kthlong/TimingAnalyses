function [jittereddata] = makejitteredreps_ratematched(spikes,rates,jittervals)

[maxspikes,identifyrep] = max(rates,[],3);

unjiterredspikes = cellfun(@(x,y) x(:,:,y), spikes, num2cell(identifyrep),'uniformoutput',0)

for tInd = 1:size(spikes,1)
    for repInd = 1:5
        spikes2keep = datasample(1:maxrate(tInd),rates(tInd,repInd),'replace',false);
        maxspikes = spikes{tInd,whichrep(tInd)};
        unjitteredspikes{tInd,repInd} = maxspikes(spikes2keep);
    end
end

unjitteredspikes = repmat(unjitteredspikes,1,1,length(jittervals));
jittervals_reform = permute(repmat(num2cell(1:10),59,1,5),[1 3 2]);

jitterVals = cellfun(@(x,y) normrnd(0,jittervals(y),size(x)),unjitteredspikes,jittervals_reform,'uniformoutput',0);

jittereddata = cellfun(@(x,y) x+y,unjitteredspikes,jitterVals,'uniformoutput',0);


