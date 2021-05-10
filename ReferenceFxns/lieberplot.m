function lieberplot(data1,data2,se,theseColors)

if nargin<4
theseColors = [.8 .8 .8; .7137 .5529 .2745];
end

firstmean = nanmean(data1);
secondmean = nanmean(data2);

if strcmp(se,'se')
firstse = nanstd(data1)./sqrt(length(data1));
secondse = nanstd(data2)./sqrt(length(data2));
else
firstse = nanstd(data1);
secondse = nanstd(data2);
end

dots1 = .1*randn(size(data1))+ones(size(data1));
dots2 = .1*randn(size(data2))+2*ones(size(data2));

figure; hold on;
b1 = bar(1,firstmean);
b1.FaceColor = theseColors(1,:);
b2 = bar(2,secondmean);
b2.FaceColor = theseColors(2,:);
b1dot = scatter(dots1,data1,[],.4*theseColors(1,:),'filled');
b2dot = scatter(dots2,data2,[],.4*theseColors(2,:),'filled');
e1 = errorbar(1,firstmean,firstse,'color','k','linewidth',2);
e2 = errorbar(2,secondmean,secondse,'color','k','linewidth',2);
xticks([1 2]);
xticklabels({'data1' 'data2'})
ylabel('values')
