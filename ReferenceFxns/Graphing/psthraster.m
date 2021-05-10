function psthraster(cell,tInd,spikes1,spikes2,nmask,trainspeed,testspeed)

nreps1 = nmask(cell,tInd,trainspeed);
nreps2 = nmask(cell,tInd,testspeed);
colors = [0 150 0; 24 128 192; 150 0 0; 255 150 0]/255; % green black orange

counter = 0;

% TRAIN SPIKES
figure(); subplot(2,1,1); hold on; box off; xlabel('time (s)'); ylabel('rep')
for repInd = 1:nreps1
    % plot for each neuron
    currentmat = cell2mat(spikes1(cell,tInd,repInd));
    if ~isnan(currentmat)
        numspikes = size(currentmat,1);
        first = (1:3:3*numspikes);
        toplot = [];
        if numspikes > 0
            for i = 1:numspikes
                toplot(first(i),:) = [currentmat(i), repInd];
                toplot(first(i)+1,:) = [currentmat(i), repInd+.9];
                toplot(first(i)+2,:) = [currentmat(i), NaN];
            end
            plot(toplot(:,1),toplot(:,2),'-','color',colors(1,:),'linewidth',2)
        end
    end
end

% TEST SPIKES
for repInd = 1:nreps2
    % plot for each neuron
    currentmat = cell2mat(spikes2(cell,tInd,repInd));
    if ~isnan(currentmat)
        numspikes = size(currentmat,1);
        first = (1:3:3*numspikes);
        toplot = [];
        if numspikes > 0
            for i = 1:numspikes
                toplot(first(i),:) = [currentmat(i), repInd+nreps1];
                toplot(first(i)+1,:) = [currentmat(i), repInd+nreps1+.9];
                toplot(first(i)+2,:) = [currentmat(i), NaN];
            end
            plot(toplot(:,1),toplot(:,2),'-','color',colors(2,:),'linewidth',2)
        end
    end
end
set(gca,'visible','off');
plot([0 .1],[0 0],'linewidth',2,'color','k')
set(gcf, 'Position', [1000, 500, 500, 200]);

% PSTH
hist1 = cellfun(@(x) histogram(x,200), squeeze(spikes1(cell,tInd,:)),'uniformoutput',0);
hist2 = cellfun(@(x) histogram(x,200), squeeze(spikes2(cell,tInd,:)),'uniformoutput',0);
for repInd = 1:nreps1
    hist_train(repInd,:,:) = hist1{repInd}.Values;
end
for repInd = 1:nreps2
    hist_test(repInd,:,:) = hist2{repInd}.Values;
end

hist_train = detrend(squeeze(mean(hist_train)));
hist_test = detrend(squeeze(mean(hist_test)));

subplot(2,1,2); hold on;
plot(hist_train,'color',colors(1,:,:),'linewidth',2);
plot(hist_test,'color',colors(2,:,:),'linewidth',2);
xlim([1 20])
set(gca,'visible','off')

