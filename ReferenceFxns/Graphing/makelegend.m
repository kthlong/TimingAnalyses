function makelegend(legendstrings,title)
%UNIVERSAL

% Create the legend
hLegend = legend(legendstrings);  
legend boxoff
 
% Create an invisible axes at the same position as the legend
hLegendAxes = axes('Parent',hLegend.Parent, 'Units',hLegend.Units, 'Position',hLegend.Position, ...
                   'XTick',[] ,'YTick',[], 'Color','none', 'YColor','none', 'XColor','none', 'HandleVisibility','off', 'HitTest','off');
 
% Add the axes title (will appear directly above the legend box)
hTitle = title(hLegendAxes, title, 'FontWeight','normal', 'FontSize',10);  % Default is bold-11, which is too large
 
% Link between some property values of the legend and the new axes
hLinks = linkprop([hLegend,hLegendAxes], {'Units', 'Position', 'Visible'});
% persist hLinks, otherwise they will stop working when they go out of scope
setappdata(hLegendAxes, 'listeners', hLinks);
 
% Add destruction event listener (no need to persist here - this is done by addlistener)
addlistener(hLegend, 'ObjectBeingDestroyed', @(h,e)delete(hLegendAxes));