function [testData] = toySpikes(meanfiringrates, frequency)


if nargin < 1
meanfiringrates = datasample([60:5:80],nneurons);
elseif nargin < 2
    frequency = 2;
end

dt = .001;
timeVals = round(-1:dt:3.5,3);
adjust = 2*pi;
wavelength = 1/frequency;
% wavelength = .5;
stimTime = round(0:.001:2,3);
stimulus_template = sin(adjust/wavelength*stimTime);
stimulus_template(stimulus_template<=0) = 0;


nneurons = 5;
nreps = 5;  
offsets = .01:.02:.2;

for offInd = 1:length(offsets)
            phaseoffset = offsets(offInd);
            stimstart = round(phaseoffset,3); stimend = round(stimstart + 2,3);
        stimulus = zeros(1,length(timeVals));
            stimulus(timeVals>=stimstart & timeVals<=stimend) = stimulus_template;
    for repInd = 1:5
        spikeDiscrete = zeros(1,length(timeVals));
            spikeProb = 100*stimulus*meanfiringrates(1)*dt+meanfiringrates(1)*dt*20;
            hitormiss = datasample(1:100,length(spikeProb));
        spikeDiscrete(spikeProb>hitormiss) = 1;
        theseSpikes(offInd,repInd) = {timeVals(spikeDiscrete==1)};
        theseRates(offInd,repInd) = length(find(spikeDiscrete))/(timeVals(end)-timeVals(1));
    end
end
% figure
testData.Spikes = theseSpikes;
testData.Rates = theseRates;
testData.Offsets = repmat(offsets,nneurons,1);
testData.BaseRate = meanfiringrates;
testData.Stimulus = stimulus_template;
testData.TimeVec = timeVals;
% raster(theseSpikes)