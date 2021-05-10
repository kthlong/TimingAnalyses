function mastertitle(titlehere,fs)
% UNIVERSAL

if nargin < 2
    fs = 12;
end

set(gcf,'NextPlot','add');
axes;
h = title(titlehere);
set(gca,'Visible','off');
set(h,'Visible','on');
set(h,'fontsize',fs);