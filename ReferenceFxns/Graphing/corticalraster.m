function corticalraster(warp,spikes,spikes_warped,minT,maxT,nmask,nspeeds,spdcolors)

%% CORTICAL RASTERS
% CORTICAL A1 = 4, A2 = 50, A3 = 27 or 29; minT = .1, maxT = 1.1
% Make one each for warped / unwarped
cell = 4;
tInd = 7;
minT = .1;
maxT = 1.1;

if warp == 1
    spikes_relevant = spikes_warped;
else
    spikes_relevant = spikes;
end

spikes_within = cellfun(@(x) x( x>= minT & x <= maxT) - minT, spikes_relevant(cell,tInd,:,1), 'uniformoutput',0);

QCdata = find(nmask(:,3)>=1 & nmask(:,4)>=1);
cellInd = find(QCdata == cell);
load(['bestoffsets_' num2str(3) '_' num2str(4) '.mat']);
offset = bestoffsets{tInd}; offset = offset(cellInd,tInd);
spikes_within(:,2) = cellfun(@(x) x ( x >= offset & x <= offset + (maxT-minT)) - offset, spikes_relevant(cell,tInd,2,1),'uniformoutput',0);

QCdata = find(nmask(:,1)>=1 & nmask(:,4)>=1);
cellInd = find(QCdata == cell);
load(['bestoffsets_' num2str(1) '_' num2str(4) '.mat']);
offset = bestoffsets{tInd}; offset = offset(cell,tInd);
spikes_within(:,1) = cellfun(@(x) x ( x >= offset & x <= offset + (maxT-minT)) - offset, spikes_relevant(cell,tInd,1,1),'uniformoutput',0);

QCdata = find(nmask(:,2)>=1 & nmask(:,4)>=1);
cellInd = find(QCdata == cell);
load(['bestoffsets_' num2str(2) '_' num2str(4) '.mat']);
offset = bestoffsets{tInd}; offset = offset(cell,tInd);
spikes_within(:,1) = cellfun(@(x) x ( x >= offset & x <= offset + (maxT-minT)) - offset, spikes_relevant(cell,tInd,1,1),'uniformoutput',0);

figure();
speedwarpfig(spikes_within,1);
if warp == 1
    saveas(gcf,['rasterfig_cell_' num2str(cell) '_warped'],'svg')
else
    saveas(gcf,['rasterfig_cell_' num2str(cell) '_UNwarped'],'svg')
end

figure()
rates_within = permute(cellfun(@(x) length(x)/(maxT-minT), spikes_within),[3 1 2]);
for speed = 1:nspeeds
    hold on;
    bar(speed,rates_within(speed),.9,'facecolor',spdcolors(speed,:),'edgecolor','none');
end
bar(4.7,50,.05,'facecolor','k')
box off;
ylabel('rate')
set(gcf, 'Position', [1000, 500, 400, 150]);
set(gca,'xTicklabel',{[]})
set(gca,'visible','off');
ylim([0 200])
if warp == 1
    saveas(gcf,['ratefig_cell_' num2str(cell) '_warped'],'svg')
else
    saveas(gcf,['ratefig_cell_' num2str(cell) '_UNwarped'],'svg')
end