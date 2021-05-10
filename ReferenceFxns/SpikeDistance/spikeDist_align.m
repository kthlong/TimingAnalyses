
function [mindist,offset] = spikeDist_align(spike1, spike2, q,minT,maxT,maxSlide,stepSize)
% THIS IS THE MOST UP TO DATE SPIKEDIST FUNCTION, March 2020

if nargin < 6
    maxSlide = .05;
    stepSize = .001;
end


if isempty(spike1) | isempty(spike2)
    mindist = nan;
    offset = nan;
elseif any(isnan(spike1)) | any(isnan(spike2))
    mindist = nan;
    offset = nan;
    
else
    spike1 = makecolumn(spike1);
    spike2 = makecolumn(spike2);
    spike1start = minT + maxSlide/2;
    spike1end = maxT - maxSlide/2;
    spike1_tmp = spike1(spike1>=spike1start & spike1<=spike1end)-spike1start;
    offsets = 0:stepSize:maxSlide;
    winLength = maxT-minT-maxSlide;
    
    for slideInd = 1:length(offsets)
        s2start = minT + offsets(slideInd); s2end = s2start + winLength;
        spike2_tmp = spike2(spike2>=s2start & spike2<=s2end)-s2start;
        dists_tmp = spkdl([spike1_tmp; spike2_tmp],[1 length(spike1_tmp)+1],[length(spike1_tmp) length(spike1_tmp)+length(spike2_tmp)],q);
        dists(slideInd) = dists_tmp(2);
    end
    
    [mindist,offsetInd]  = min(dists);
    offset = offsets(offsetInd);
end

% the above if statements correct a problem that arose anytime one input
% was an empty vector and the other a nan. this combination should now
% always result in a nan output instead of a numerical distance.