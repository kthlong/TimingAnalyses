function spiketiming(slowwind,fastwind,offset)

window1unmod = cellfun(@(x) x(x <= .75),slowwind,'uniformoutput',0);
figure(); hold on;
subplot(2,1,1)
raster(window1unmod,'r')
raster(fastwind,'k')
title(num2str(spikeDist([window1unmod{:}],[fastwind{:}],1/4)))

window1mod = cellfun(@(x) x(x>=offset & x<=offset + .75) - offset,slowwind,'uniformoutput',0);
subplot(2,1,2)
raster(window1mod,'r')
raster(fastwind,'k')
title(num2str(spikeDist([window1mod{:}],[fastwind{:}],1/4)))
