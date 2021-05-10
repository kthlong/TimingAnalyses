function qvalPerfPlot(data,QCdata,timeResArray)
% CORTICAL
% Parameters
% afftypes = {'SA1','RA','PC'};
% load('periphData.mat');
timeResArray = timeResArray.*1000;
affcolors = [0 153 76; 0 76 153; 255 128 0]./255;
linecolors = affcolors*.8;

if length(size(data{1})) == 3 % single neuron
    % Performance Per Afferent Type
    for affInd = 1:length(afftypes)
        aff = afftypes{affInd};
        affMask = find(~cellfun(@isempty,strfind(periphData.type(QCdata),aff)));

        for qInd = 1:length(timeResArray)
            qresults = data{qInd};
            meanperf = nanmean(nanmean(qresults(:,affMask,:),3),1);
            stdperf(qInd) = std(meanperf)/sqrt(length(affMask));
            perf(qInd) = nanmean(meanperf);
        end
        hold on;
        patch([timeResArray(1:end-1),timeResArray(end-1:-1:1)],[perf(1:end-1)+stdperf(1:end-1),perf(end-1:-1:1)-stdperf(end-1:-1:1)],affcolors(affInd,:),'EdgeColor','none')
        lines(affInd) = plot(timeResArray,perf,'-','linewidth',3,'color',linecolors(affInd,:));
    end
    legend(lines, afftypes);
    title('Single Afferent Spike Timing Classification') 
    legend boxoff;
    
elseif length(size(data{1})) == 2 % population
    % Population Performance
    for qInd = 1:length(timeResArray)
        meanperf = nanmean(data{qInd},1);
        stdperf(qInd) = std(meanperf)/sqrt(length(QCdata));
        perf(qInd) = nanmean(meanperf);
    end
    hold on;
    patch([timeResArray(1:end-1),timeResArray(end-1:-1:1)],[perf(1:end-1)+stdperf(1:end-1),perf(end-1:-1:1)-stdperf(end-1:-1:1)],[.75 .75 .75],'EdgeColor','none') % Standard Error
    plot(timeResArray,perf,'-','linewidth',3,'color','k') % Mean
    title('Population Spike Timing Classification')
end

% Formatting
ylim([0 1]);
xlabel('Time Resolution (ms)')
ylabel('Performance')
set(gca,'XScale','log');
set(gca,'xticklabel',[1 10 100 1000])
xlim([min(timeResArray) max(timeResArray)]);
box off