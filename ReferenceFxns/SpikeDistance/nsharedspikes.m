function coinspikes = nsharedspikes(spikes1,spikes2,resolution)

if nargin <3
    resolution = .004;
end

if iscell(spikes1)
    spikes1 = cell2mat(spikes1);
    spikes2 = cell2mat(spikes2);
end

if length(spikes1) > length(spikes2)
    refspikes = spikes2;
    compspikes = spikes1;
else
    refspikes = spikes1;
    compspikes = spikes2;
end

coinspikes = 0;

for ind1 = 1:length(refspikes)
    spike = refspikes(ind1);
    if ~isempty(find(compspikes >= spike- resolution/2 & compspikes <= spike+resolution/2))
        coinspikes = coinspikes + 1;
    end
end

