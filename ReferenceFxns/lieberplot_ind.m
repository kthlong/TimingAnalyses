function lieberplot_ind(data,dataInd,SAinds,PCinds)

data1 = data;
theseColors = [.625 .625 .625];
firstmean = nanmean(data1);
colorSA = [  0    0.5977    0.2969];
colorPC = [  0.9961    0.5000         0];

firstse = nanstd(data1)./sqrt(length(data1));

dots1 = .1*randn(size(data1))+dataInd*ones(size(data1));

hold on;
b1 = bar(dataInd,firstmean);
b1.FaceColor = theseColors(1,:);

b1dot = scatter(dots1,data1,[],.4*theseColors(1,:),'filled');
e1 = errorbar(dataInd,firstmean,firstse,'color','k','linewidth',2);

if nargin>2
scatter(b1dot.XData(SAinds),b1dot.YData(SAinds),[],colorSA,'filled')
scatter(b1dot.XData(PCinds),b1dot.YData(PCinds),[],colorPC,'filled')
end

xticks([1:dataInd]);
ylabel('values')
