function raster(data,color,plottitle)

if nargin < 2
    color = parula(size(data,1));
end

hold on;
for neuron = 1:size(data,1)
    for texture = 1:size(data,2)
        for rep = 1:size(data,3)
            currentmat = cell2mat(data(neuron,texture,rep));
            numspikes = length(currentmat);
            first = (1:3:3*numspikes);
            toplot = [];
            if numspikes > 0
                for i = 1:numspikes
                    ID = (neuron-1)*60 + (texture-1)*5 + (rep-1);
                    toplot(first(i),:) = [currentmat(i), ID];
                    toplot(first(i)+1,:) = [currentmat(i), ID + .9];
                    toplot(first(i)+2,:) = [currentmat(i), NaN];
                end
                plot(toplot(:,1),toplot(:,2),'-','color',color(neuron,:),'linewidth',2)
                %         xlim([0 0.5])
            end
        end
    end
end
ylim([1 ID+1])
box off
xlabel('time (ms)')
% ylabel('neuron index')


if nargin == 3
    title(plottitle)
end
end
