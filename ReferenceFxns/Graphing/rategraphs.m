function rategraphs(rates,QCdata,color)
% UNIVERSAL
% Parameters:
ntextures = size(rates,2);
nspeeds = size(rates,3);

if nspeeds == 3
    speeds = {'40','80','120'};
    spdlist = [40 80 120];
    colors = [0 150 0; 0 0 0; 255 150 0]/255; % green black orange
else
    speeds = {'120' '100' '80' '60'};
    spdlist = [120 100 80 60];
    colors = [0 150 0; 150 0 0; 0 0 0; 255 150 0]/255; % green red black orange

end

for speed = 1:nspeeds
    meanrates(speed,:) = nanmean(nanmean(rates(QCdata,:,speed,:),4),1);
end

[a ind] = sort(meanrates(2,:));

hold on;
%
subplot(3,1,1)
p = plot(1:ntextures,meanrates(:,ind),'linewidth',3);
for speed = 1:nspeeds
    p(speed).Color = colors(speed,:);
end
box off
xlabel('texture'); xlim([1 ntextures])
ylabel('rate'); ylim([0 120])
legend(speeds,'location','NW'); legend boxoff;
%
subplot(3,1,2)
hold on;
plot(spdlist,nanmean(meanrates,2),'linewidth',3,'color',color)
xlabel('scan speed')
ylabel('mean rate'); ylim([20 60])
%
subplot(3,1,3)
hold on;
cellmeans = nanmean(nanmean(nanmean(rates(QCdata,:,:,:),4),2),3);
normalized = permute(nanmean(nanmean(rates(QCdata,:,:,:),4),2),[1 3 2])./repmat(cellmeans,1,nspeeds);
plot(spdlist, nanmean(normalized,1),'linewidth',3,'color',color)
box off
ylabel('norm mean rate')
xlabel('scan speed')
ylim([.5 1.5])
