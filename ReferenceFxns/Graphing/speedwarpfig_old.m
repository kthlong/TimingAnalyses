function [spikevec, n] = speedwarpfig_old(spikes,texture,offsets,minT,maxT)
% UNIVERSAL

spikes = squeeze(spikes(:,texture,:,:));
speeds = [60 80 100 120];
% nneurons = size(spikes,1);
colors = [0 150 0; 24 128 192; 150 0 0; 255 150 0]/255; % green black orange
repInd = 1;
nspeeds = size(spikes,1);
nreps = size(spikes,2);
minT = .1;
speedInds = 1:nspeeds;
if nspeeds == 3
    colors = colors([1, 2, 4],:);
    speeds = [40 80 120];
    minT = 0;
end
% speedInds = nspeeds:-1:1;

if nargin < 4
    offsets = zeros(nspeeds,nreps);
else
    spikes = cellfun(@(x,y) x(x <= (maxT + y) & x >= (minT + y)) - y,spikes,num2cell(offsets),'uniformoutput',0);
end

for speedInd = speedInds
    % plot characteristics
    speed = speeds(speedInd);
    hold on;
    box off
    xlabel('time (s)')
    ylabel('rep')
    % plot for each neuron
%     for neuron = 1:nneurons
        for rep = 1:nreps
            currentmat = cell2mat(spikes(speedInd,rep));
            if ~isnan(currentmat)
                numspikes = length(currentmat);
                first = (1:3:3*numspikes);
                toplot = [];
                if numspikes > 0
                    for i = 1:numspikes
                        toplot(first(i),:) = [currentmat(i), repInd];
                        toplot(first(i)+1,:) = [currentmat(i), repInd+.9];
                        toplot(first(i)+2,:) = [currentmat(i), NaN];
                    end
                    plot(toplot(:,1),toplot(:,2),'-','color',colors(speedInd,:),'linewidth',2)
                end
                repInd = repInd + 1;
            end
        end
%     end
end
set(gca,'visible','off');
plot([0 .1],[0.9 0.9],'linewidth',2,'color','k')
set(gcf, 'Position', [1000, 500, 500, 200]);