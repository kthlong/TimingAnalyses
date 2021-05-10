% SpikeDist_APCTiming_SPEED_July2020

%% 0. Load everything
close all; clear all;

for hide = 1
% path
addpath(genpath('Users/klong/Dropbox/Matlab/New folder/Spike_Timing_APC'))
addpath(genpath('Users/klong/Dropbox/Moonshine_NeuralData/MatLab'))

% load data
load('cdaData.mat');
load('periphData.mat');
load('roughData_mod.mat');
load('periphtexturelab.mat')
load('pRegCoeff.mat')
load('dissimData.mat')

% general parameters
minT = .1;
maxT = .6;
qvals = [.001, .002, .005, .01, .02, .05, .1,.2,.5,inf];
jitters = [.001,.002,.005,.008,.01,.02,.05,.1,.2];

% Spikes and Textures
% cdaSpikes = cdaData.fullRun.data;
% cdaTextures = cdaData.fullRun.textures;
% cdaRates = corticalSpikeTiming(cdaSpikes,'rate',minT,maxT,'windowsOFF');
% cdaRates = cdaRates.rate;
% perSpikes = squeeze(periphData.allSpikes(2,:,:,:));
% perTextures = periphtexturelabs;
% perRough = roughData.roughMean;
% perRates = corticalSpikeTiming(perSpikes,'rate',minT,maxT,'windowsOFF');
% perRates = perRates.rate;

roughTextures = periphtexturelabs;
cdaRough = roughData.cdaroughVal;

% Submodality (this is already within the data structures)
% subcell_high = pRegCoeff > .8;
% PClikes = find(subcell_high(1,:));
% SAlikes = find(subcell_high(3,:));
% PCs = find(strcmp(periphData.type,'PC'));
% SAs = find(strcmp(periphData.type,'SA1'));

% Colors
colorPC = [255,128,0]./256;
colorSA = [0 153 76]./256;
colorRate = [0 153 153]./256;
colorTiming = [255 51 51]./256;
colorGrey = [160 160 160]./256;
colorJitters = parula(9);

% load generated data
load('cda.mat')
load('cdaJit.mat')
load('cdaPoiss.mat')
load('cdaTVPoiss.mat')
load('per.mat');
% load('LPC.mat');

% QC data
QCmask = cellfun(@(x) ~isvector(x), cda.spikes);

end

%% Set up data structure
% perSpeedNorm.spikes = periphData.allSpikes;
% perSpeedWarped.spikes = cellfun(@(x,y) x.*(y/80),perSpeedNorm.spikes,num2cell(repmat(periphData.vel,1,39,55,4)),'uniformoutput',0);

% thisStruct          = cdaWarped;

thisStruct.minT     = minT;
thisStruct.maxT     = maxT;
spikes = cdaData.speedRun.data;
QCmask = cellfun(@(x) ~isvector(x), spikes);
spikes(QCmask) = {nan};
speeds = permute(repmat(cdaData.speedRun.speeds',1,size(spikes,1),size(spikes,2),size(spikes,4)),[2 3 1 4]);
warpedspikes   = cellfun(@(x,y) x.*(y/80),spikes,num2cell(speeds),'uniformoutput',0);

%% 3. [Plot] Plot Raster: one cell, multiple textures
% for tIndplot = 1:length(speedTextures)
    tIndplot = 39;
figure;
data = cdawarped;
data.spikes = data.spikes(:,:,:,1);
data.rates = data.rates(:,:,:,1);
jitInd = 1;
speedTextures = find(~isnan(data.rates(:,1,1)));
thiscell = speedTextures(tIndplot);
thiscolor = colorGrey;
tInds = [1 2 3 4 5 6 7 8 9];
definedtitle = ['cortical cell ' num2str(tIndplot)];

squeeze(nanmean(data.rates(thiscell,tInds,:),3))

% -------------------------------------------------------------------------
for hide = 1
    data.spikes = data.spikes(:,:,:,jitInd);
textureNames = data.textures;
ncells = size(data.spikes,1);
ntextures = size(data.spikes,2);
nreps=  size(data.spikes,3);

for tIndInd = 1:length(tInds)
    tInd = tInds(tIndInd);
subplot(round(sqrt(length(tInds))),round(length(tInds)/round(sqrt(length(tInds)))),tIndInd);
hold on;
for repInd = 1:nreps
    raster_singletrial(data.spikes(thiscell,tInd,repInd),repInd-1,repInd-.1,thiscolor)
    ylim([0 nreps])
end

xlim([minT 2])
axis off;
title(textureNames{tInd})
end
hold on; plot([.1 .3],[0 0],'-','linewidth',2,'color','k')


mastertitle([definedtitle ': ' num2str(thiscell)])
set(gcf,'Position',[100 100 800 200])

end
% end



%% spike dist distmats
for trainInd = 1:4
    for testInd = 1:4
        filename = ['spikeDist_train' num2str(trainInd) '_test' num2str(testInd)];
        distmat = cstClassification_speed(squeeze(warpedspikes(:,:,trainInd,:)),squeeze(warpedspikes(:,:,testInd,:)),'SpikeDist',2,3,minT,maxT,qvals);
    end
end
save(filename,'distmat')

%% spike dist results
for trainInd = 1:4
    for testInd = 1:4
        filename = ['spikeDist_train' num2str(trainInd) '_test' num2str(testInd)];
        load(filename)
        distmat = squeeze(fixdistmat(distmat));
        results = GetResults(nanmean(distmat,4));
        meanresults = squeeze(nanmean(results,[1 2]));
        allresults(:,:,trainInd,testInd) = meanresults;
    end
end

allresultsmax = squeeze(max(allresults,[],2));
allresultsmaxmean = squeeze(nanmean(allresultsmax,1));
imagesc(allresultsmaxmean);

% save('spikedistResults.mat','allresults');

%% rate distmats
output = corticalSpikeTiming(unwarpedspikes,'rate',minT,maxT,'windowsOFF','scrambleOFF');
rates = num2cell(output.rate);
reprates = permute(repmat(rates,1,1,1,1,4),[1 2 4 3 5]);

distmat = cstClassification_speed_withoutbins(reprates,'EucDist',2,3,minT,maxT,bins);

%% rate results
distmat = fixdistmat(distmat);
results = GetResults(nanmean(distmat,4));
meanresults = squeeze(nanmean(results,[1 2]));
figure; imagesc(squeeze(nanmean(meanresults,1)))
caxis([.13 .32])
title('rate - warped')

%% psth output
 output = corticalSpikeTiming(unwarpedspikes,'psth',minT,maxT,'windowsOFF','scrambleOFF');
 psth = output.psth;
 bins = output.psthGaussWidths;
 speedCells = find(~QCmask(:,1,1,5));
 QCmask_rep = permute(repmat(QCmask,1,1,1,1,15),[5 1 2 3 4]);
 psth(QCmask_rep) = {repmat(nan,500,1)};
 psth = psth(:,speedCells,:,:,:);
 reppsth = permute(repmat(psth,1,1,1,1,1,4),[1 2 3 4 6 5]);
 
 
 %% psth distmat

distmat = cstClassification_speed(reppsth,'XCov',3,6,minT,maxT,bins);
 
%% psth results
distmat = fixdistmat(distmat);
results = GetResults_cov(nanmean(distmat,4));
meanresults = squeeze(nanmean(results,[1 2]));
meanresults = reshape(meanresults,15,49,4,4);
maxmeanresults = squeeze(max(nanmean(meanresults,2),[],1));
figure; imagesc(maxmeanresults);
caxis([.15 .5])
title('psth warped')

%% psth figure; mean across conditions
distmat = fixdistmat(distmat);
results = GetResults_cov(nanmean(distmat,4));
meanresults = squeeze(nanmean(results,[1 2]));
meanresults = reshape(meanresults,15,49,4,4);

clear withinall
clear acrossall
% NEED TO ADD STD ERROR
allspeeds = 1:4;
for spInd = 1:4
    nonspInd = allspeeds(allspeeds~=spInd);
    withinall(spInd,:,:) = meanresults(:,:,spInd,spInd);
    acrossall(spInd,:,:,:) = meanresults(:,:,spInd,nonspInd);
end

withinall = reshape(permute(withinall,[2 1 3]),15,4*49);
acrossall = reshape(permute(acrossall,[2 1 3 4]),15,4*49*3);


figure;
errorshadeplot(output.psthGaussWidths,nanmean(withinall,2)',nanstd(withinall,[],2)'./sqrt(size(withinall,2)))
hold on;
errorshadeplot(output.psthGaussWidths,nanmean(acrossall,2)',nanstd(acrossall,[],2)'./sqrt(size(acrossall,2)))
ylim([0 .5])
xticks(output.psthGaussWidths)
xticklabels(output.psthGaussWidths*1000)

% psth bar graph
% allspeeds = 1:4;
% for spInd = 1:4
%     nonspInd = allspeeds(allspeeds~=spInd);
%     withinmaxmean(spInd) = maxmeanresults(spInd,spInd);
%     acrossmaxmean(spInd,:) = maxmeanresults(spInd,nonspInd);
% end
% bar(withinmaxmean)

%% psth violin plot: warped vs not
distmat = fixdistmat(distmat);
results = GetResults_cov(nanmean(distmat,4));
meanresults = squeeze(nanmean(results,[1 2]));
meanresults = reshape(meanresults,15,49,4,4);
maxmeanresults = squeeze(max(meanresults,[],1));

clear withinallmax
clear acrossallmax
% NEED TO ADD STD ERROR
allspeeds = 1:4;
for spInd = 1:4
    nonspInd = allspeeds(allspeeds~=spInd);
    withinallmax(spInd,:) = maxmeanresults(:,spInd,spInd);
    acrossallmax(spInd,:,:) = maxmeanresults(:,spInd,nonspInd);
end

withinallmax = reshape(withinallmax,4*49,1);
acrossallmax = reshape(acrossallmax,4*49*3,1);
figure; hold on;
Violin(withinallmax,1);
Violin(acrossallmax,2);
xticks([1 2])
xticklabels({'within' 'across'})
ylim([0 1])


withinallmax = reshape(withinallmax,4*49,1);
acrossallmax = reshape(acrossallmax,4*49*3,1);
figure; hold on;
bar(1,nanmean(withinallmax));
plot([1 1],[nanmean(withinallmax)+nanstd(withinallmax), nanmean(withinallmax)-nanstd(withinallmax)],'color','k')
bar(2,nanmean(acrossallmax));
plot([2 2],[nanmean(acrossallmax)+nanstd(acrossallmax), nanmean(acrossallmax)-nanstd(acrossallmax)],'color','k')

xticks([1 2])
xticklabels({'within' 'across'})
ylim([0 1])


%% rate violin plot within across
distmat = fixdistmat(distmat);
results = GetResults(nanmean(distmat,4));
meanresults = squeeze(nanmean(results,[1 2]));


clear withinallmax
clear acrossallmax
% NEED TO ADD STD ERROR
allspeeds = 1:4;
for spInd = 1:4
    nonspInd = allspeeds(allspeeds~=spInd);
    withinallmax(spInd,:) = meanresults(:,spInd,spInd);
    acrossallmax(spInd,:,:) = meanresults(:,spInd,nonspInd);
end

withinallmax = reshape(withinallmax,4*141,1);
acrossallmax = reshape(acrossallmax,4*141*3,1);
figure; hold on;
bar(1,nanmean(withinallmax));
plot([1 1],[nanmean(withinallmax)+nanstd(withinallmax), nanmean(withinallmax)-nanstd(withinallmax)],'color','k')
bar(2,nanmean(acrossallmax));
plot([2 2],[nanmean(acrossallmax)+nanstd(acrossallmax), nanmean(acrossallmax)-nanstd(acrossallmax)],'color','k')

xticks([1 2])
xticklabels({'within' 'across'})
ylim([0 1])


%% spike dist distmat
repspikes = repmat(warpedspikes,1,1,1,1,4);
repspikes = permute(repspikes,[1 2 4 3 5]);
distmat = cstClassification_speed_withoutbins(repspikes,'SpikeDist',2,3,minT,maxT,bins);
distmat = distmat(:,:,:,:,speedCells,:,:,:);

%% QC MASK
QCmask = cellfun(@(x) ~isvector(x) | isempty(x) | all(isnan(x)), spikes);
QCmask = QCmask(speedCells,:,:,:);
QCmask = permute(repmat(QCmask,1,1,1,1,10,4,5, 15),[2 5 4 7 1 3 6 8]);
distmat(QCmask) = nan;

% NOT NECESSARY

%% spikedist results
distmat = fixdistmat(distmat);
results = GetResults(nanmean(distmat,4));
meanresults = squeeze(nanmean(results,[1 2]));
meanresults = reshape(meanresults,49,4,4,15);
maxmeanresults = squeeze(max(nanmean(meanresults,1),[],4));
figure; imagesc(maxmeanresults);
caxis([.1 .8])
title('spikedist warped')

%% RATE Population!
ratedistmat = distmat(:,:,:,:,speedcells,:,:);
ratedistmat = fixdistmat(ratedistmat);

for popSize = 1:49
    for itInd = 1:50
    theseCells = datasample(1:length(speedCells),popSize,'replace',false);
    results = GetResults(nanmean(ratedistmat(:,:,:,:,theseCells,:,:),[4 5]));
    meanresults = nanmean(results,[1 2]);
    popresults(popSize,itInd,:,:) = squeeze(meanresults);
    end
end
 
meanpopresults = squeeze(nanmean(popresults,2));
stdpopresults = squeeze(nanstd(popresults,[],2));
figure;
for trainInd = 1:4
subplot(4,1,trainInd)
plot(squeeze(meanpopresults(:,:,trainInd)))
ylim([0 1])
% for testInd = 1:4
%     hold on;
% e=errorshadeplot(1:49,squeeze(meanpopresults(:,testInd,trainInd))',stdpopresults(:,testInd,trainInd)',colors(testInd,:));
end
box off;
legend 120 100 80 60

%% rate/timing/psth distmats!
load('rate_unwarped.mat');
distmat = distmat(:,:,:,:,speedCells,:,:,:);
distmat = fixdistmat(distmat);
ratedistmat = distmat./200;
load('PSTH_denseWarped.mat');
psthdistmat = squeeze(distmat(:,:,:,:,8,:,:,:)*-1+1);
alldistmat = cat(9,ratedistmat,psthdistmat);
finaldistmat = squeeze(nanmean(alldistmat,9));
%% rate/timing/psth Population Plot

for popSize = 1:49
    for itInd = 1:100
    theseCells = datasample(1:length(speedCells),popSize,'replace',false);
    theseCellsextra = datasample(1:length(speedCells),popSize,'replace',false);
    results = GetResults(nanmean(finaldistmat(:,:,:,:,[theseCells theseCellsextra],:,:),[4 5]));
    rateresults = GetResults(nanmean(ratedistmat(:,:,:,:,theseCells,:,:),[4 5]));
    psthresults = GetResults(nanmean(psthdistmat(:,:,:,:,theseCells,:,:),[4 5]));
    meanresults = nanmean(results,[1 2]);
    meanrate    = nanmean(rateresults,[1 2]);
    meanpsth    = nanmean(psthresults,[1 2]);
    popresults(1,popSize,itInd,:,:) = squeeze(meanresults);
    popresults(2,popSize,itInd,:,:) = squeeze(meanrate);
    popresults(3,popSize,itInd,:,:) = squeeze(meanpsth);
    end
end
 
meanpopresults = squeeze(nanmean(popresults,3));
figure;
for trainInd = 1:4
    testMask = 1:4;
    testMask = find(trainInd ~= testMask);
    withinspeed(:,:,trainInd) = squeeze(nanmean(meanpopresults(:,:,trainInd,trainInd),3));
    acrossspeed(:,:,trainInd) = squeeze(nanmean(meanpopresults(:,:,testMask,trainInd),3));
end
for groupInd = 1:3
    plot(1:49,nanmean(acrossspeed(groupInd,:,:),3)); hold on;
end


%% TIMING Population!
spikedistmat = fixdistmat(distmat);

for popSize = 1:49
    for itInd = 1:50
    theseCells = datasample(1:length(speedCells),popSize,'replace',false);
    results = GetResults(nanmean(spikedistmat(:,:,:,:,theseCells,:,:,10),[4 5]));
    meanresults = nanmean(results,[1 2]);
    meanresults = reshape(meanresults,4,4,15);
    popresults(popSize,itInd,:,:,:) = squeeze(meanresults);
    end
end
 
meanpopresults = squeeze(nanmean(popresults,2));
figure;
for trainInd = 1:4
subplot(4,1,trainInd)
plot(squeeze(meanpopresults(:,:,trainInd)))
box off;
end
legend 120 100 80 60

%% combined Population!
ratedistmat = fixdistmat(ratedistmat);

for popSize = 1:49
    for itInd = 1:50
    theseCells = datasample(1:length(speedCells),popSize,'replace',false);
    results = GetResults(nanmean(alldistmat(:,:,:,:,theseCells,:,:),[4 5]));
    meanresults = nanmean(results,[1 2]);
    popresults(popSize,itInd,:,:) = squeeze(meanresults);
    end
end
 
meanpopresults = squeeze(nanmean(popresults,2));
stdpopresults = squeeze(nanstd(popresults,[],2));
figure;
for trainInd = 1:4
subplot(4,1,trainInd)
plot(squeeze(meanpopresults(:,trainInd,:)))
ylim([0 1])
% for testInd = 1:4
%     hold on;
% e=errorshadeplot(1:49,squeeze(meanpopresults(:,testInd,trainInd))',stdpopresults(:,testInd,trainInd)',colors(testInd,:));
end
box off;
legend 120 100 80 60
    
    
%% Single cells across resolutions: SPIKEDIST
spikedistmat = fixdistmat(distmat(:,:,:,:,speedCells,:,:,:));
results = GetResults(nanmean(spikedistmat,4));
meanresults = squeeze(nanmean(results,[1 2 3]));
figure;
for trainInd = 1:4
    for testInd = 1:4
        subplot(4,4,trainInd + (testInd-1)*4)
        plot(bins,squeeze(meanresults(testInd,trainInd,:)))
    set(gca,'XScale','log')   
    ylim([0 1])
    end
end

%% Single cells across resolutions: PSTH
psthdistmat = fixdistmat(distmat);
results = GetResults_cov(nanmean(psthdistmat,4));
meanresults = squeeze(nanmean(results,[1 2]));
        groupresults = squeeze(nanmean(meanresults,2));
        grouperror = squeeze(nanstd(meanresults,[],2))./7;
        colors = parula(4);
figure;
for trainInd = 1:4 
    subplot(2,2,trainInd)
        for testInd = 1:4
%     subplot(4,4,trainInd + (testInd-1)*4)
        e = errorshadeplot(1000*bins,groupresults(:,trainInd,testInd)',grouperror(:,trainInd,testInd)',colors(testInd,:)); 
        hold on;
%         plot(bins,squeeze(meanresults(:,testInd,trainInd))); hold on;
    end
    set(gca,'XScale','log')
    ylim([0 .5])
    xticks(bins([1,4:2:end])*1000)
    box off;
end

%% Population speed
load('PSTH_denseWarped.mat')
clear zmat_timing;
distmat = fixdistmat(distmat);
distmat = squeeze(nanmean(distmat,4));

for spInd = 1:4
    for spInd2 = 1:4
        for resInd = 1:15
            straightmat = nanzscore(reshape(distmat(:,:,:,resInd,:,spInd,spInd2),10*10*5,49));
            zmat_timing(:,:,:,resInd,spInd,spInd2,:) = reshape(straightmat,10,10,5,49);
        end
    end
end


% % % 
load('rate_unwarped.mat')
clear zmat_rate;
distmat = fixdistmat(distmat);
distmat = squeeze(nanmean(distmat(:,:,:,:,speedCells,:,:),4));

for spInd = 1:4
    for spInd2 = 1:4
        straightmat = nanzscore(reshape(distmat(:,:,:,:,spInd,spInd2),10*10*5,49));
        zmat_rate(:,:,:,spInd,spInd2,:) = reshape(straightmat,10,10,5,49);
    end
end

resInd = 8; clear both; clear popresults_r; clear popresults_t; clear popresults_both;
for popSize = 1:length(speedCells)
    for itInd = 1:1000
        theseCells = datasample(1:length(speedCells),popSize,'replace',false);
        poptiming = squeeze(nanmean(zmat_timing(:,:,:,resInd,:,:,theseCells),[7]));
        poprate = squeeze(nanmean(zmat_rate(:,:,:,:,:,theseCells),[6]));
        both(:,:,:,:,:,1) = poprate; both(:,:,:,:,:,2) = -poptiming;
        popboth = nanmean(both,6);
        popresults_r(popSize,itInd,:,:) = squeeze(nanmean(GetResults(poprate),[1 2]));
        popresults_t(popSize,itInd,:,:) = squeeze(nanmean(GetResults_cov(poptiming),[1 2]));
        popresults_both(popSize,itInd,:,:) = squeeze(nanmean(GetResults(popboth),[1 2]));
    end
end


for spInd = 1:4
    popresults_r(:,:,spInd,spInd) = nan;
    popresults_t(:,:,spInd,spInd) = nan;
    popresults_both(:,:,spInd,spInd) = nan;
end

rateresults = squeeze(nanmean(popresults_r,[3 4]));
timingresults = squeeze(nanmean(popresults_t,[3 4]));
bothresults = squeeze(nanmean(popresults_both,[3 4]));
    
    figure; hold on;
    plot(nanmean(rateresults,[2]));
    plot(nanmean(timingresults,[2]))
    plot(nanmean(bothresults,[2]))
    legend rate timing both
    legend boxoff
    
    figure; hold on;
    errorshadeplot_nonlog(1:49,nanmean(rateresults,[2])',nanstd(rateresults,[],2)',colorTiming);
    errorshadeplot_nonlog(1:49,nanmean(timingresults,[2])',nanstd(timingresults,[],2)',colorRate);
    errorshadeplot_nonlog(1:49,nanmean(bothresults,[2])',nanstd(bothresults,[],2)');
    ylim([0 1])

    
    
    %% Population speed // control with twice as many rate cells 
load('PSTH_denseWarped.mat')
clear zmat_timing;
distmat = fixdistmat(distmat);
distmat = squeeze(nanmean(distmat,4));

for spInd = 1:4
    for spInd2 = 1:4
        for resInd = 1:15
            straightmat = nanzscore(reshape(distmat(:,:,:,resInd,:,spInd,spInd2),10*10*5,49));
            zmat_timing(:,:,:,resInd,spInd,spInd2,:) = reshape(straightmat,10,10,5,49);
        end
    end
end


% % % 
load('rate_unwarped.mat')
clear zmat_rate;
distmat = fixdistmat(distmat);
distmat = squeeze(nanmean(distmat(:,:,:,:,speedCells,:,:),4));

for spInd = 1:4
    for spInd2 = 1:4
        straightmat = nanzscore(reshape(distmat(:,:,:,:,spInd,spInd2),10*10*5,49));
        zmat_rate(:,:,:,spInd,spInd2,:) = reshape(straightmat,10,10,5,49);
    end
end

resInd = 8; clear both; clear popresults_r; clear popresults_t; clear popresults_both;
for popSize = 1:length(speedCells)
    for itInd = 1:1000
        theseCells = datasample(1:length(speedCells),popSize,'replace',false);
        theseExtraCells = datasample(1:length(speedCells),popSize,'replace',false);
        poptiming = squeeze(nanmean(zmat_timing(:,:,:,resInd,:,:,theseCells),[7]));
        poprate = squeeze(nanmean(zmat_rate(:,:,:,:,:,theseCells),[6]));
        poprate2 = squeeze(nanmean(zmat_rate(:,:,:,:,:,theseExtraCells),[6]));
        both(:,:,:,:,:,1) = poprate; both(:,:,:,:,:,2) = poprate2;
        popboth = nanmean(both,6);
        popresults_r(popSize,itInd,:,:) = squeeze(nanmean(GetResults(poprate),[1 2]));
        popresults_t(popSize,itInd,:,:) = squeeze(nanmean(GetResults_cov(poptiming),[1 2]));
        popresults_both(popSize,itInd,:,:) = squeeze(nanmean(GetResults(popboth),[1 2]));
    end
end


for spInd = 1:4
    popresults_r(:,:,spInd,spInd) = nan;
    popresults_t(:,:,spInd,spInd) = nan;
    popresults_both(:,:,spInd,spInd) = nan;
end

rateresults = squeeze(nanmean(popresults_r,[3 4]));
timingresults = squeeze(nanmean(popresults_t,[3 4]));
bothresults = squeeze(nanmean(popresults_both,[3 4]));
    
    figure; hold on;
    plot(nanmean(rateresults,[2]));
    plot(nanmean(timingresults,[2]))
    plot(nanmean(bothresults,[2]))
    legend rate timing both
    legend boxoff
    
    figure; hold on;
    errorshadeplot_nonlog(1:49,nanmean(rateresults,[2])',nanstd(rateresults,[],2)',colorTiming);
    errorshadeplot_nonlog(1:49,nanmean(timingresults,[2])',nanstd(timingresults,[],2)',colorRate);
    errorshadeplot_nonlog(1:49,nanmean(bothresults,[2])',nanstd(bothresults,[],2)');
    ylim([0 1])
    
    %%
    data = per;
    
    lieberplot(data.matchedRATEresults(data.PCs),data.matchedRATEresults(data.SAs),'sd',[colorPC;colorSA])
    title('Rate'); ylim([0 .5])
    lieberplot(data.matchedPSTHresults(data.PCs,5),data.matchedPSTHresults(data.SAs,5),'sd',[colorPC;colorSA])
    title('Timing'); ylim([0 .8])
    
    lieberplot(cda.matchedRATEresults,per.matchedRATEresults,'sd'); title('Rate'); ylim([0 .5]);
    lieberplot(cda.matchedPSTHresults(:,5),per.matchedPSTHresults(:,5),'sd'); title('Timing'); ylim([0 1])

 %% Make data mask
% thisStruct = perSpeedWarped;
thisStruct.mask = cellfun(@isempty,thisStruct.spikes);
thisStruct.spikes(nodatamask) = {nan};



%% Make new data structure


% RATE
output = corticalSpikeTiming(thisStruct.spikes,'rate',minT,maxT,'windowsOFF','scrambleOFF');
thisStruct.rate = output.rate;
thisStruct.rate(thisStruct.mask) = nan;

distmat = cstClassification(num2cell(thisStruct.rate),'EucDist',3,4,minT,maxT);
distmat = fixdistmat(distmat);
thisStruct.ratedistmat = distmat;


% ISI
output = corticalSpikeTiming(thisStruct.spikes,'isi',minT,maxT,'windowsOFF','scrambleOFF');
output.isi(repmat(permute(thisStruct.mask,[5 1 2 3 4]),size(output.isi,1),1,1,1,1)) = {nan};
thisStruct.isi = output.isi;
thisStruct.isibins = output.isibins;

distmat = cstClassification(thisStruct.isi,'EucDist',4,5,minT,maxT);
distmat = fixdistmat(distmat);
thisStruct.isidistmat = distmat;

% PSTH
output = corticalSpikeTiming(thisStruct.spikes,'psth',minT,maxT,'windowsOFF','scrambleOFF');
output.psth(repmat(permute(thisStruct.mask,[5 1 2 3 4]),size(output.psth,1),1,1,1,1)) = {nan};
thisStruct.psth = output.psth;
thisStruct.psthbins = output.psthGaussWidths;

distmat = cstClassification(reshape(permute(thisStruct.psth,[1 3 4 5 2]),15,39,55,12),'XCov',3,4,minT,maxT);
distmat = reshape(distmat,55,55,4,3,4,3,15,39);
distmat = permute(distmat,[1 2 3 5 8 7 4 6]);
distmat = fixdistmat(distmat);
results = GetResults_cov(nanmean(distmat,4));
meanresults = squeeze(nanmean(results,[1 2 3]));
meanresults = reshape(meanresults,15,3,3);
thisStruct.PSTHmat = distmat;
thisStruct.PSTHresults = results;
thisStruct.PSTHmeanresults = meanresults;

% SPIKE DIST
distmat = cstClassification(reshape(permute(thisStruct.spikes,[2 3 4 1]),39,55,12),'SpikeDist',2,3,minT,maxT);
distmat = reshape(distmat,55,55,4,3,4,3,39,9);
distmat = permute(distmat,[1 2 3 5 7 8 4 6]);
distmat = fixdistmat(distmat);
results = GetResults(nanmean(distmat,4));
meanresults = squeeze(nanmean(results,[1 2 3]));
meanresults = reshape(meanresults,9,3,3);
thisStruct.spikedistmat = distmat;
thisStruct.spikedistresults = results;
thisStruct.spikedistmeanresults = meanresults;

%% Population RESULTS
% Spike Dist
popresults = GetResults(nanmean(thisStruct.spikedistmat,[4 5]));
meanpopresults = squeeze(nanmean(popresults,[1 2]));
meanpopresults = reshape(meanpopresults,9,3,3);
thisStruct.spikedistPOPresults = meanpopresults;

% PSTH
popresults = GetResults_cov(nanmean(thisStruct.PSTHmat,[4 5]));
meanpopresults = squeeze(nanmean(popresults,[1 2]));
meanpopresults = reshape(meanpopresults,15,3,3);
thisStruct.psthPOPresults = meanpopresults;

%% Figures from above
% SPIKE DIST
bins = cda.SpikeDistbins;
results = thisStruct.spikedistPOPresults;

speeds = [40 80 120];
figure;
for trainI = 1:3
    subplot(3,1,trainI)
    for testI = 1:3
        semilogx(bins,results(:,testI,trainI),'linewidth',2); hold on;
        box off;
        xlabel('temporal resolution (ms)')
        ylabel('% correct')
        ylim([0 1])
        xticks([.0001 .0002 .0005 cda.SpikeDistbins])
        xticklabels([.0001 .0002 .0005 cda.SpikeDistbins]*1000)
    end
    title(['train at ' num2str(speeds(trainI))])
end
legend 40 80 120
legend boxoff

% PSTH
bins = thisStruct.psthbins;
results = thisStruct.psthPOPresults;

speeds = [40 80 120];
figure;
for trainI = 1:3
    subplot(3,1,trainI)
    for testI = 1:3
        semilogx(bins,results(:,testI,trainI),'linewidth',2); hold on;
        box off;
        xlabel('temporal resolution (ms)')
        ylabel('% correct')
        ylim([0 1])
        xticks([.0001 .0002 .0005 cda.SpikeDistbins])
        xticklabels([.0001 .0002 .0005 cda.SpikeDistbins]*1000)
    end
    title(['train at ' num2str(speeds(trainI))])
end
legend 40 80 120
legend boxoff


