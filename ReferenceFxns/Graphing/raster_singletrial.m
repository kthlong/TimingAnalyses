function raster_singletrial(data,yMin,yMax,color)

if nargin < 4
    color = 'k';
end

if ~exist('yMin')
    yMin = 0;
end

if ~exist('yMax')
    yMax = .9;
end

hold on;

if iscell(data)
data = cell2mat(data);
end

numspikes = length(data);
first = (1:3:3*numspikes);
toplot = [];
    if numspikes > 0
        for i = 1:numspikes          
            toplot(first(i),:) = [data(i), yMin];
            toplot(first(i)+1,:) = [data(i), yMax];
            toplot(first(i)+2,:) = [data(i), NaN];
        end                  
        plot(toplot(:,1),toplot(:,2),'-','color',color,'linewidth',2)
    end
end
