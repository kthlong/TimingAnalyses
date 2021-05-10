function [ISIhistogram] = ISIhist(spikes,window)

if nargin > 1
    spikes = spikes(spikes >= min(window) & spikes <= max(window));
else
    spikes = spikes(spikes>=0);
end

spiketimes = repmat(spikes,1,length(spikes));
ISIs = abs(triu(spiketimes - spiketimes'));
ISIs = ISIs(ISIs~=0);
ISIs_all = ISIs(:);
ISIhistogram = histc(ISIs_all,[0:.05:3]);

