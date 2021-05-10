function spiketimes = poissonspikes(originalspikes,minT,maxT)

rate = length(find(originalspikes>=minT & originalspikes<=maxT));

timeStepS = 0.0001;  
spikesPerS = rate;                   
durationS = maxT;                     
times = minT:timeStepS:durationS;		

vt = rand(size(times));

spikes = (spikesPerS*timeStepS) > vt;

spiketimes = times(find(spikes))';
poissrate = length(spiketimes(spiketimes>=minT & spiketimes<=maxT));
spiketimes = ratematchjitter(spiketimes,poissrate,rate,minT,maxT);
spiketimes= makecolumn(spiketimes);


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