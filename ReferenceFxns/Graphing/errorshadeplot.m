function linespecs = errorshadeplot(x,y,dy,colorval)

if nargin < 4
%     colorval = [.5 .5 .5];
    colorval = 'k';
end

fill([x';flipud(x')],[y'+dy';flipud(y'-dy')],colorval,'linestyle','none');
alpha(.1)
linespecs = line(x,y,'linewidth',2,'color',colorval);
box off;
set(gca,'XScale','log');