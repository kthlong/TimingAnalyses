function [spikevec, n] = speedwarpfig_new(spikes,minT,maxT,offsets)
% UNIVERSAL
% figure();
nneurons = size(spikes,1);
nspeeds = size(spikes,3);
ntextures = size(spikes,2);
nreps = size(spikes,4); repInd = 1;
colors = [0 150 0; 24 128 192; 150 0 0; 255 150 0]/255; % green black orange

speeds = [60 80 100 120];
if nspeeds == 3
    colors = colors([1, 2, 4],:);
    speeds = [40 80 120];
end

if nargin < 4
    refspikes = cellfun(@(x) x(x <= maxT & x >= minT) - minT,squeeze(spikes(1,1,2,1)),'uniformoutput',0); % 2,1
    for nInd = 1:nneurons
        for tInd = 1:ntextures
            for speedInd = 1:nspeeds
                for repInd = 1:nreps
                    offsets(nInd,tInd,speedInd,repInd) = bestoffset_shared(refspikes,squeeze(spikes(nInd,tInd,speedInd,repInd)),.05,maxT-minT);
                end
            end
        end
    end
end

spikes = cellfun(@(x,y) x(x <= (maxT + y) & x >= (minT + y)) - (minT + y),spikes,num2cell(offsets),'uniformoutput',0);


count = 0;
for neuron = 1:nneurons
    for texture = 1:ntextures
        for speedInd = nspeeds:-1:1
            % plot for each neuron
            %     for neuron = 1:nneurons
            for rep = 1:nreps
                currentmat = squeeze(cell2mat(spikes(neuron,texture,speedInd,rep)));
                if ~isnan(currentmat)
                    count = count + 1;
                    numspikes = length(currentmat);
                    first = (1:3:3*numspikes);
                    toplot = [];
                    if numspikes > 0
                        for i = 1:numspikes
                            toplot(first(i),:) = [currentmat(i), count];
                            toplot(first(i)+1,:) = [currentmat(i), count+.9];
                            toplot(first(i)+2,:) = [currentmat(i), NaN];
                        end
                        plot(toplot(:,1),toplot(:,2),'-','color',colors(speedInd,:),'linewidth',2); hold on;
                    end
                end
            end
        end
    end
end
set(gca,'visible','off');
plot([0 .1],[0.9 0.9],'linewidth',2,'color','k')
set(gcf, 'Position', [1000, 500, 500, 200]);
xlim([0 maxT-minT])