function errorshadeplot_nonlog(x,y,dy,colorval)

if nargin < 4
%     colorval = [.5 .5 .5];
    colorval = 'k';
end

fill([x';flipud(x')],[y'+dy';flipud(y'-dy')],colorval,'linestyle','none');
alpha(.5)
line(x,y,'linewidth',2,'color',colorval)
box off;
