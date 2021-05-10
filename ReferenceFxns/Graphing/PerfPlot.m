function PerfPlot(data,QCdata)
%PERIPHERAL

% Parameters
afftypes = {'RA','SA1','PC'};
load('periphData');

if length(size(data{1})) == 3 % single neuron
    % Performance Per Afferent Type
    for affInd = 1:length(afftypes)
        aff = afftypes{affInd};
        for speed = 1:3
            QCD = QCdata{speed};
            affMask = find(~cellfun(@isempty,strfind(periphData.type(QCD),aff)));
            sdata = data{speed};
            meanperf = nanmean(nanmean(sdata(:,affMask,:),3),1);
            stdperf = std(meanperf);
            perf(speed) = nanmean(meanperf);
        end
        % Plot
        hold on;
        plot([40 80 120],perf,'linewidth',3);
    end
    legend(afftypes); 
    legend boxoff;
    title('Single Afferent Spike Timing Classification')
    
elseif length(size(data{1})) == 2 % population
    % Population Performance
    for speed = 1:3
        meanperf = nanmean(data{speed},1);
        stdperf(speed) = std(meanperf);
        perf(speed) = nanmean(meanperf);
    end
    hold on;
    plot([40 80 120],perf,'-','linewidth',3,'color','k')
    title('Population Spike Timing Classification')
end

% Formatting
ylim([0 1]);
xlabel('Time Resolution (ms)')
ylabel('Performance')
box off