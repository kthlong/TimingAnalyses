function arearategraphs(rates,QCdata,color)
% UNIVERSAL

% Parameters:
ntextures = size(rates,2);
nspeeds = size(rates,3);
if nspeeds == 4
    speeds = [40 60 80 120];
else
    speeds = [40 80 120];
end

for speed = 1:nspeeds
    meanrates(speed,:) = nanmean(nanmean(rates(QCdata,:,speed,:),4),1);
end

[a ind] = sort(meanrates(2,:));

%
subplot(2,1,1)
hold on;
plot(speeds,nanmean(meanrates,2),'linewidth',3,'color',color)
ylabel('mean rate');
%
subplot(2,1,2)
hold on;
cellmeans = nanmean(nanmean(nanmean(rates(QCdata,:,:,:),4),2),3);
normalized = permute(nanmean(nanmean(rates(QCdata,:,:,:),4),2),[1 3 2])./repmat(cellmeans,1,nspeeds);
plot(speeds, nanmean(normalized,1),'linewidth',3,'color',color)
box off
ylabel('norm mean rate');
xlabel('scan speed')