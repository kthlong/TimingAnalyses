function [spikevec, n] = raster_warp(spikes,texture,nreps)
% UNIVERSAL

spikes = permute(spikes(:,texture,:,:),[1 3 4 2]);
nspeeds = size(spikes,3);
nneurons = size(spikes,1);

figure()
for speed = 1:nspeeds
    % plot characteristics
    subplot(nspeeds,1,speed)
    hold on;
    title(speed)
    ylim([0 length(spikes)])
    xlim([0 1.1])
    box off
    xlabel('time (s)')
    ylabel('trial')
    % plot for each neuron
    for neuron = 1:nneurons
        for rep = 1:nreps
            currentmat = cell2mat(spikes(neuron,speed,rep));
            numspikes = length(currentmat);
            first = (1:3:3*numspikes);
            toplot = [];
            if numspikes > 0
                for i = 1:numspikes
                    toplot(first(i),:) = [currentmat(i), neuron + (rep-1)/nreps];
                    toplot(first(i)+1,:) = [currentmat(i), neuron + rep/nreps];
                    toplot(first(i)+2,:) = [currentmat(i), NaN];
                end                
                plot(toplot(:,1),toplot(:,2),'-','color','k')
                spikevec{neuron,texture,rep} = toplot;
            else
                spikevec{neuron,texture,rep} = [nan nan];
            end
        end
    end
end
n = neuron;
end