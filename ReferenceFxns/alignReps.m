function [alignedspikes, offsets] = alignReps(spikes,maxSlide)

if nargin < 2
    maxSlide = 100;
end

if size(spikes,1) ~= 1
    disp('error: only input one cell at a time!')
end


% testspikes
testspikes = [0:.05:1];
bumps = 0:.1:1;
bumps2 = 0:.2:1;
phaseshifts = [.01:.01:.05];
for repInd = 1:5
    phase = phaseshifts(repInd);
    thesespikes = testspikes;
    for bumpInd = 1:length(bumps)
        thesespikes = [thesespikes,normrnd(bumps(bumpInd)+phase,.005,1,10)];
    end
    spikes{repInd} = thesespikes;
end
figure();
raster(spikes)



for slideInd = 1:length(slideVals)
    slide = slideVals(slideInd);
    
    
    
end
    





