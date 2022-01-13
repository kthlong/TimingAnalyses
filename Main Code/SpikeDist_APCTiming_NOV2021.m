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
load('periphtexturelab.mat') % peripheral texture names
load('pRegCoeff.mat') % PC RA SA
load('dissimData.mat')
load('newcdaJit.mat')
load('perception.mat')

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
colorRA =  [ 0.0508    0.4453    0.7266];
colorTiming = [0 153 153]./256;
colorRate = [255 51 51]./256;
colorGrey = [160 160 160]./256;
colorJitters = parula(9);

% load generated data
load('cda.mat')
load('newcdaJit.mat')
load('cdaPoiss_extrareps.mat')
load('cdaTVPoiss.mat')
load('per.mat');

% QC data
QCmask = cellfun(@(x) ~isvector(x), cda.spikes);
QCmask_peripheral = cellfun(@(x) ~isvector(x), per.spikes);

end


%% [SKIP, already saved] Create Pseudo Data
% generate jittered data
for hide = 1
jitters = [.0001 .0002 .0005 .001,.002,.005,.008,.01,.02,.05,.1,.2];
for jitInd = 1:length(jitters)
    cdaJitSpikes(:,:,:,jitInd) = ...
        ratematchedjitters_minrep(cda.spikes,minT,maxT,3,jitters(jitInd));
%     perJitSpikes(:,:,:,jitInd) = ...
%         ratematchedjitters(per.spikes,minT,maxT,3,jitters(jitInd));
end

% generate poisson data
% cdaPoiss = cellfun(@(x) poissonspikes(x,minT,maxT), cda.spikes,'uniformoutput',0);
% perPoiss = cellfun(@(x) poissonspikes(x,minT,maxT), num2cell(per.rates),'uniformoutput',0);

% generate time-varying poisson data
% cdaTVPoisson = maketvpoisson(cdaSpikes,jitters,minT,maxT);
% perTVPoisson = maketvpoisson(perSpikes,jitters,minT,maxT);
end

%%  [Plot] Plot Raster: any number of cells, one texture
figure;
data = cda;
jitInd = 1;
timingcells = find(max(cda.PSTHresults,[],2)>.3);
choosecells = cda.RAs;
thesecells = choosecells(1:9);
thiscolor = colorGrey;
for tInd = [2 26 22 25 33 41 38 47 52]
    figure
definedtitle = ['cortical time-varied poisson '];

% -------------------------------------------------------------------------
for hide = 1
    data.spikes = data.spikes(:,:,:,jitInd);
textureNames = data.textures;
ncells = size(data.spikes,1);
ntextures = size(data.spikes,2);
nreps=  size(data.spikes,3);

for cellInd = 1:length(thesecells)
subplot(round(sqrt(length(thesecells))),round(length(thesecells)/round(sqrt(length(thesecells)))),cellInd);
hold on;
for repInd = 1:nreps
    raster_singletrial(data.spikes(thesecells(cellInd),tInd,repInd),repInd-1,repInd-.1,thiscolor)
    ylim([0 nreps])
end
xlim([minT maxT])
axis off;
end

mastertitle([definedtitle ': ' textureNames{tInd}])
set(gcf,'Position',[100 100 800 200])
end
end

%%  [Plot] Plot Raster: any number of cells, one texture, all stacked
figure;
data = cda;
jitInd = 1;
timingcells = find(max(cda.PSTHresults,[],2)>.3);
[~,inds] = sort(nanmean(cda.rates(cda.SAs,:,:),[2 3]))
SAs = cda.SAs(inds);
choosecells = SAs;
thesecells = choosecells(15:24);
thiscolor = colorGrey;
tInds = [2 26 22 25 33 41 38 47 52];
for tInd = [2 26 22 25 33 41 38 47 52]
    figure
definedtitle = ['cortical time-varied poisson '];

% -------------------------------------------------------------------------
for hide = 1
    data.spikes = data.spikes(:,:,:,jitInd);
textureNames = data.textures;
ncells = size(data.spikes,1);
ntextures = size(data.spikes,2);
nreps=  size(data.spikes,3);

for cellInd = 1:length(thesecells)
subplot(length(thesecells),1,cellInd);
hold on;
for repInd = 1:nreps
    raster_singletrial(data.spikes(thesecells(cellInd),tInd,repInd),repInd-1,repInd-.1,thiscolor)
    ylim([0 nreps])
end
xlim([minT maxT])
hold on;
plot([.1 .2],[0 0],'-','color','r')
axis off;
end

mastertitle([definedtitle ': ' textureNames{tInd}])
set(gcf,'Position',[100 100 800 200])
end
end

%% 3. [Plot] Plot Raster: one cell, multiple textures
figure;
data = cda; % cdawarped
jitInd = 1;
thiscell = 86; % 84
thiscolor = colorGrey;
tInds = [2 26 22 25 33 41 38 47 52]; % 2 26 22 25 33 41/42 38/51 47 52
tInds = [35 59];
definedtitle = ['cortical cell '];

squeeze(nanmean(cda.rates(thiscell,tInds,:),3))

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
hold on;
plot([.1 .2],[0 0],'-','color','k','linewidth',3)
xlim([minT maxT])
axis off;
title(textureNames{tInd})
end



mastertitle([definedtitle ': ' num2str(thiscell)])
set(gcf,'Position',[100 100 800 200])
end






%% 3. [Plot] Plot Raster: one cell, multiple textures
for thiscell = 1:141
close all;
figure;
data = cda; % cdawarped
jitInd = 1;
% thiscell = 8; % 84
thiscolor = colorGrey;
tInds = [2 26 22 25 33 41 38 47 52]; % 2 26 22 25 33 41/42 38/51 47 52
definedtitle = ['cortical cell '];

squeeze(nanmean(cda.rates(thiscell,tInds,:),3))

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
xlim([minT maxT])
axis off;
title(textureNames{tInd})
end

mastertitle([definedtitle ': ' num2str(thiscell)])
set(gcf,'Position',[100 100 800 200])
end
saveas(gcf,['Rasters/' num2str(thiscell) '.png'])
end

%% rep 2 rep distance

data = jitteredmin;

% -------------------------------------------------------------------------
for hide = 1
ncells = size(data.spikes,1);
ntextures = size(data.spikes,2);
nreps=  size(data.spikes,3);
njits = size(data.spikes,4);
qvals = [.001, .002, .005, .01, .02, .05, inf];

repSpikeDist = nan(nreps,nreps,ncells,ntextures,njits);
for refRep = 1:nreps
    for crossrefRep = 1:nreps
        tic
        if refRep < crossrefRep
            for qInd = 1:length(qvals)
        repSpikeDist(refRep,crossrefRep,:,:,:,qInd) = ...
            cellfun(@(x,y) spikeDist_align(x,y,1/qvals(qInd),minT,maxT),data.spikes(:,:,refRep,:),data.spikes(:,:,crossrefRep,:));
            end
        end
        toc
    end
end
end

jitteredmin.repSpikeDist = repSpikeDist;
clear repSpikeDist;
save('jitteredmin.mat','jitteredmin')

%% Plot rate x  variability
data = jitteredmin;
ismodel = 1;
% figure; 
hold on;
thesecells = data.allcells;
colorHere = 'r';
tempRes = 2;

for hide = 1
    if ismodel == 1
        thiscolor = colorGrey;
        thiscolor = colorHere;
    else
        thiscolor = colorHere;
    end
    rSD = fixdistmat(data.repSpikeDist);
    rSD = squeeze(nanmean(data.repSpikeDist,[1 2 4]));
    x = nanmean(data.rates(thesecells,:,:),[2 3]);
    for jitInd = 1:9
        subplot(3,3,jitInd)
        if ismodel == 1
            y = rSD(thesecells,jitInd,tempRes);
        else
            y = rSD(thesecells,tempRes);
        end
    plot(x,y,'o','color',thiscolor);
    coeff = polyfit(x,y,2);
    xFit = linspace(min(x),max(x),100);
    yFit = polyval(coeff,xFit);
    hold on;
    plot(xFit,yFit,'--','color',thiscolor);
    xlabel('rate')
    ylabel('temporal variability')
%     ylim([0 20])
    box off;
    title(jitters(jitInd))
    end
end

%% Each cell's matched jitter?
realr2r = squeeze(nanmean(cda.repSpikeDist,[1 2 4]));
jitterr2r = squeeze(nanmean(newcdaJit.repSpikeDist,[1 2 4]));
poissr2r = squeeze(nanmean(cdaPoiss_extrareps.repSpikeDist,[1 2 4]));

% [~,badcells] = sort(cellres,'descend')
% thesecells = badcells(131:141)

thesecells = 1:141;
qvals_noinf = [qvals(1:end-1) .5];
thisColor = [0 0 0];
jitters = newcdaJit.jitters;

figure; hold on;
for jitInd = 1:12
    semilogx(qvals_noinf,squeeze(nanmean(jitterr2r(thesecells,jitInd,:),1)));
end
semilogx(qvals_noinf,squeeze(nanmean(realr2r(thesecells,:),1)),'linewidth',2,'color',thisColor);
semilogx(qvals_noinf,squeeze(nanmean(poissr2r(thesecells,:),1)),'linewidth',2,'color',[.8 .8 .8]);


set(gca,'XScale','log')
xlabel('qval')
ylabel('temporal variability')
title('real data (black), poisson (grey), jittered (.1 ms -> 20 ms)')

tempRes = 2;
diff_fromjit = abs(permute(jitterr2r,[1 3 2]) - repmat(realr2r,1,1,12));
[cellBest,bestRes] = min(diff_fromjit(:,tempRes,:),[],3);
figure;
Violin(jitters(bestRes),1)
Violin(jitters(bestRes(cda.PCs)),2)
Violin(jitters(bestRes(cda.SAs)),3)

realandjit = cat(3,permute(jitterr2r,[1 3 2]),realr2r);
[~,jitOrder] = sort(realandjit,3);
for cellInd = 1:141
    cellres(cellInd) = find(jitOrder(cellInd,tempRes,:) == 13);
end
betweenjits =  1000*(jitters - .5*[.00005, diff(jitters)]);
betweenjits(13) = 300;
figure;
Violin(betweenjits(cellres),1)
Violin(betweenjits(cellres(cda.PCs)),2)
Violin(betweenjits(cellres(cda.SAs)),3)
set(gca,'YScale','log')

figure;
cdfplot(betweenjits(cellres))
set(gca,'XScale','log')
xticks([1 10 100])
xticklabels([1 10 100])
title('cdf of best resolutions')
xlabel('resolution')
ylabel('fraction of cells')

hold on;
cdfplot(betweenjits(cellres(cda.PCs)))
cdfplot(betweenjits(cellres(cda.SAs)))
set(gca,'XScale','log')
xlabel('temporal resolution (ms)')
ylabel('proportion');
xlim([.1 500])
xticks(jitters*1000)
box off;



[cellMax,cellRes]  = max(cda.PSTHresults,[],2);
cellRes = cda.PSTHbins(cellRes);
figure;
cdfplot(cellRes*1000)
hold on;
cdfplot(1000*cellRes(cda.PCs))
cdfplot(1000*cellRes(cda.SAs))
set(gca,'XScale','log')
xlabel('temporal resolution (ms)')
ylabel('proportion');
xlim([.1 500])
xticks(jitters*1000)
box off;

[cellMax,cellRes]  = max(cda.PSTHresults,[],2);
cellRes = cda.PSTHbins(cellRes);
figure; hold on;
sAll = scatter(betweenjits(cellres),1000*cellRes,'filled','k');
sPC = scatter(betweenjits(cellres(cda.SAs)),1000*cellRes(cda.SAs),[],colorSA,'filled');
sSA = scatter(betweenjits(cellres(cda.PCs)),1000*cellRes(cda.PCs),[],colorPC,'filled');
sAll.MarkerFaceAlpha = .5;
sPC.MarkerFaceAlpha = .5;
sSA.MarkerFaceAlpha = .5;
set(gca,'XScale','log')
set(gca,'YScale','log')
plot([.01 5000], [.01 5000],'--','color','k')
box off;
xlim([.01 5000])
ylim([.01 5000])
xlabel('jitter resolution')
ylabel('psth classification best res')
xticks(jitters*1000)
yticks(jitters*1000)


howfaroff = abs(betweenjits(cellres) - 1000*cellRes);
figure;
scatter(cellMax, howfaroff,'filled','k')
set(gca,'YScale','log')
box off;
yticks(jitters*1000)
ylim([0.05 500])
hold on;
scatter(cellMax(cda.PCs), howfaroff(cda.PCs),[],colorPC,'filled')
scatter(cellMax(cda.SAs), howfaroff(cda.SAs),[],colorSA,'filled')
xlabel('classification performance')
ylabel('difference in resolution prediction')

figure;
cdfplot(howfaroff(find(cellMax>=0.3)))
box off;
xlabel('absolute difference (ms)')
ylabel('proportion')
title('psth error for cells with classification >= 30%')

% spike distance


[cellMax,cellRes]  = max(cda.SpikeDistresults,[],2);
cellRes = cda.SpikeDistbins(cellRes);


figure;
cdfplot(cellRes*1000)
hold on;
cdfplot(1000*cellRes(cda.PCs))
cdfplot(1000*cellRes(cda.SAs))
set(gca,'XScale','log')
xlabel('temporal resolution (ms)')
ylabel('proportion');
xlim([.1 500])
xticks(jitters*1000)
box off;


figure; hold on;
sAll = scatter(betweenjits(cellres),1000*cellRes,'filled','k');
sPC = scatter(betweenjits(cellres(cda.SAs)),1000*cellRes(cda.SAs),[],colorSA,'filled');
sSA = scatter(betweenjits(cellres(cda.PCs)),1000*cellRes(cda.PCs),[],colorPC,'filled');
sAll.MarkerFaceAlpha = .5;
sPC.MarkerFaceAlpha = .5;
sSA.MarkerFaceAlpha = .5;
set(gca,'XScale','log')
set(gca,'YScale','log')
plot([.01 5000], [.01 5000],'--','color','k')
box off;
xlim([.01 5000])
ylim([.01 5000])
xlabel('jitter resolution')
ylabel('spikedist classification best res')
xticks(jitters*1000)
yticks(jitters*1000)


howfaroff = abs(betweenjits(cellres) - 1000*cellRes);
figure;
scatter(cellMax, howfaroff,'filled','k')
set(gca,'YScale','log')
box off;
yticks(jitters*1000)
ylim([0.05 500])
hold on;
scatter(cellMax(cda.PCs), howfaroff(cda.PCs),[],colorPC,'filled')
scatter(cellMax(cda.SAs), howfaroff(cda.SAs),[],colorSA,'filled')
xlabel('classification performance')
ylabel('difference in resolution prediction')

figure;
cdfplot(howfaroff(cellMax>=0.2))
box off;
xlabel('absolute difference (ms)')
ylabel('proportion')
title('spd error for cells with classification >= 30%')
%% NEW R2R PLOT
r2r_percell = squeeze(nanmean(cda.repSpikeDist,[1 2]));
poissr2r = squeeze(nanmean(cdaPoiss.repSpikeDist,[1 2]));
r2r_percell  = poissr2r;
% r2r_percell_jit = squeeze(nanmean(newcdaJit.repSpikeDist,[1 2]));
r2r_percell_jit = squeeze(nanmean(jitteredmin.repSpikeDist,[1 2]));

thiscell = 86;
[cellMax,cellRes]  = max(cda.PSTHresults,[],2);

thesecolors = lines(12);
figure;
for jitInd = 1:12
    subplot(3,4,jitInd)
    theserates = squeeze(nanmean(cda.rates(thiscell,:,:),[1 3]));
    realdist   = squeeze(r2r_percell(thiscell,:,2));
    jitdist    = squeeze(r2r_percell_jit(thiscell,:,jitInd,2));
    plot(theserates,realdist./theserates,'o','color',colorGrey); hold on;
    coeffs = polyfit(theserates,realdist./theserates, 1);
    fittedX = linspace(min(theserates), max(theserates), 200);
    fittedY = polyval(coeffs, fittedX);
    plot(fittedX,fittedY,'--','color',colorGrey)

    plot(theserates,jitdist./theserates,'o','color',thesecolors(jitInd,:))
    coeffs = polyfit(theserates,jitdist./theserates, 1);
    fittedX = linspace(min(theserates), max(theserates), 200);
    fittedY = polyval(coeffs, fittedX);
    plot(fittedX,fittedY,'--','color',thesecolors(jitInd,:))
    box off;
    title(['jitter: ' num2str(newcdaJit.jitters(jitInd)*1000) 'ms'])
end

jitdist = squeeze(r2r_percell_jit(thiscell,:,:,2));
diffjit = jitdist - repmat(realdist',1,12);
figure;
for jitInd = 4:12
    Violin(diffjit(:,jitInd),newcdaJit.jitters(jitInd)*1000,'Width',(newcdaJit.jitters(jitInd)*1000)/4);
end
set(gca,'XScale','log')
hold on;
plot([1 200],[0 0],'--','color','k')
xticks(cda.PSTHbins*1000)
plot([cda.PSTHbins(cellRes(thiscell))*1000 cda.PSTHbins(cellRes(thiscell))*1000],[-1 1],'--','color','k')


theseCells = 86;
realdist   = squeeze(nanmean(r2r_percell(theseCells,:,2),2));
jitdist = squeeze(nanmean(r2r_percell_jit(theseCells,:,:,2),2));
diffjit = jitdist - repmat(realdist,1,12);
figure;
for jitInd = 4:12
    v = Violin(diffjit(:,jitInd),newcdaJit.jitters(jitInd)*1000,'Width',(newcdaJit.jitters(jitInd)*1000)/4);
end
set(gca,'XScale','log')
hold on;
plot([1 200],[0 0],'--','color','k')
xticks(cda.PSTHbins*1000)
plot([cda.PSTHbins(cellRes(thiscell))*1000 cda.PSTHbins(cellRes(thiscell))*1000],[-1 1],'--','color','k')


%

theseCells = find(cdaData.area==3);
realdist   = squeeze(nanmean(r2r_percell(theseCells,:,2),2));
jitdist = squeeze(nanmean(r2r_percell_jit(theseCells,:,:,2),2));
diffjit = jitdist - repmat(realdist,1,12);
figure; hold on;
for jitInd = 4:12
    v = scatter(repmat(newcdaJit.jitters(jitInd)*1000,size(diffjit,1),1),diffjit(:,jitInd),'filled','k');
    v.MarkerFaceAlpha = 0.3;
    m = scatter(newcdaJit.jitters(jitInd)*1000,nanmean(diffjit(:,jitInd),1),'filled','k');
    m.SizeData = 60;
    mean_prec(jitInd) = nanmean(diffjit(:,jitInd),1);
end
set(gca,'XScale','log')
hold on;
plot([1 200],[0 0],'--','color','k')
xticks(cda.PSTHbins*1000)
plot([cda.PSTHbins(cellRes(thiscell))*1000 cda.PSTHbins(cellRes(thiscell))*1000],[-1 1],'--','color','k')


coeffs = polyfit(newcdaJit.jitters(4:12)*1000,mean_prec(4:12),1);
xvals = linspace(newcdaJit.jitters(4)*1000,newcdaJit.jitters(12)*1000,500);
yvals = polyval(coeffs,xvals);
plot(xvals,yvals,'-','color','r')

%% HEATMAPS CONFUSION MATRICES POISSON VS REAL
distmat = fixdistmat(distmat);

figure;
[~,tInd] = sort(nanmean(cda.rates(86,:,:),[1 3]));
imagesc(nanmean(distmat(tInd,tInd,:,:,5,86),[3 4 6]),[0.5 2.5])
title('example cell')
figure;
[~,tInd] = sort(nanmean(cda.rates(:,:,:),[1 3]));
imagesc(nanmean(distmat(tInd,tInd,:,:,5,:),[3 4 6]),[.5 2])
title('all cells')
figure;
[~,tInd] = sort(nanmean(cda.rates(cda.PCs,:,:),[1 3]));
imagesc(nanmean(distmat(tInd,tInd,:,:,5,cda.PCs),[3 4 6]),[0.5 2.5])

title('PC-like cells')

%% All cells, new plot, R2R calculate actual resolution
% r2r_percell = squeeze(nanmedian(rep2rep.SPD_real,[1 2]));
% poissr2r = squeeze(nanmedian(rep2rep.SPD_poiss,[1 2]));
% r2r_percell_jit = squeeze(nanmedian(newcdaJit.repSpikeDist,[1 2]));

r2r_percell = squeeze(nanmean(cda.repSpikeDist,[1 2]));
poissr2r = squeeze(nanmean(cdaPoiss.repSpikeDist,[1 2]));
r2r_percell_jit = squeeze(nanmean(newcdaJit.repSpikeDist,[1 2]));

tempInd = 2;
theseCells = 86;
[cellMax,cellRes]  = max(cda.PSTHresults,[],2);
whichInd = 1; % (one cell = 1, multiple cells = 2;)

thesecolors = lines(12);
figure; overallInd = 0;
for jitInd = [4:7]
    overallInd = overallInd+1;
    subplot(2,2,overallInd)
    theserates = squeeze(nanmean(cda.rates(theseCells,:,:),[whichInd 3]));
    realdist   = squeeze(nanmean(r2r_percell(theseCells,:,tempInd),whichInd));
    jitdist    = squeeze(nanmean(r2r_percell_jit(theseCells,:,jitInd,tempInd),whichInd));
    poissdist  = squeeze(nanmean(poissr2r(theseCells,:,tempInd),whichInd));
    
    if whichInd == 1
        realdist = realdist';
        jitdist = jitdist';
        poissdist = poissdist';
        theserates = theserates';
    end
    
    % Real
    sR = scatter(theserates,realdist./theserates,'filled','CData',[0 0 0],'MarkerFaceAlpha',.3); hold on;
    coeffs = polyfit(theserates,realdist./theserates, 1);
    fittedX = linspace(min(theserates), max(theserates), 200);
    fittedY = polyval(coeffs, fittedX);
    plot(fittedX,fittedY,'--','color','k')
%     cf = fit(theserates,realdist./theserates,'exp1'); hold on;
%     p = plot(cf);
%     p.Color = [0 0 0];
%     p.LineStyle = '--';

    % Jit
    sJ = scatter(theserates,jitdist./theserates,'filled','CData',thesecolors(jitInd,:),'MarkerFaceAlpha',.3);
    coeffs = polyfit(theserates,jitdist./theserates,1);
    fittedY = polyval(coeffs,fittedX);
    plot(fittedX,fittedY,'--','color',thesecolors(jitInd,:));
%     cf = fit(theserates,jitdist./theserates,'exp2'); hold on;
%     p = plot(cf);
%     p.Color = thesecolors(jitInd,:);
%     p.LineStyle = '--';
    
    
    % Poiss
    sP = scatter(theserates,poissdist./theserates,'filled','CData',colorGrey,'MarkerFaceAlpha',.3); hold on;
    coeffs = polyfit(theserates,poissdist./theserates, 1);
    fittedX = linspace(min(theserates), max(theserates), 200);
    fittedY = polyval(coeffs, fittedX);
    plot(fittedX,fittedY,'--','color',colorGrey)
%     cf = fit(theserates,poissdist./theserates,'exp2'); hold on;
%     p = plot(cf);
%     p.Color = colorGrey;
%     p.LineStyle = '--';
 
    % Format
    box off;
    title(['jitter: ' num2str(newcdaJit.jitters(jitInd)*1000) 'ms'])
    legend off;
    ylim([0 1])
end

    if whichInd == 2
        theserates = squeeze(nanmean(cda.rates(theseCells,:,:),[whichInd 3]));
        realdist   = squeeze(nanmedian(r2r_percell(theseCells,:,tempInd),whichInd))./theserates;
        jitdist    = squeeze(nanmedian(r2r_percell_jit(theseCells,:,:,tempInd),whichInd))./repmat(theserates,1,12);    
        poissdist  = squeeze(nanmedian(poissr2r(theseCells,:,tempInd),whichInd))./theserates;
        diffjit_r  = jitdist - repmat(realdist,1,12);
        diffjit_p  = jitdist - repmat(poissdist,1,12);
    else
        theserates = squeeze(nanmean(cda.rates(theseCells,:,:),[whichInd 3]))';
        realdist   = squeeze(nanmedian(r2r_percell(theseCells,:,tempInd),whichInd))'./theserates;
        jitdist    = squeeze(nanmedian(r2r_percell_jit(theseCells,:,:,tempInd),whichInd))./repmat(theserates,1,12);    
        poissdist  = squeeze(nanmedian(poissr2r(theseCells,:,tempInd),whichInd))'./theserates;
        diffjit_r  = jitdist - repmat(realdist,1,12);
        diffjit_p  = jitdist - repmat(poissdist,1,12);
    end

figure;
for jitInd = [4,5,6,8,9,10,11,12]
    v2 = Violin(diffjit_p(:,jitInd),newcdaJit.jitters(jitInd)*1000,'Width',(newcdaJit.jitters(jitInd)*1000)/4,'ViolinColor',colorGrey);
    v = Violin(diffjit_r(:,jitInd),newcdaJit.jitters(jitInd)*1000,'Width',(newcdaJit.jitters(jitInd)*1000)/4,'ViolinColor',[0 0 0]);
end
set(gca,'XScale','log')
hold on;
plot([1 200],[0 0],'--','color','k')
xticks(cda.PSTHbins*1000)


%%
r2r_percell = squeeze(nanmean(rep2rep.PSTH_real,[1 2]));
r2r_poiss   = squeeze(nanmean(rep2rep.PSTH_poiss,[1 2]));
tempInd = 5;
cellInd = 86;

figure;
scatter(nanmean(cda.rates(:,:,:),[2 3]),nanmean(r2r_percell(:,:,tempInd),2),'filled','CData',[0 0 0]);
hold on;
scatter(nanmean(cda.rates(:,:,:),[2 3]),nanmean(r2r_poiss(:,:,tempInd),2),'filled','CData',colorGrey);



%%
for repInd = 1:5
    cda.repSpikeDist([repInd:5],repInd,:,:,:) = nan;
    newcdaJit.repSpikeDist([repInd:5],repInd,:,:,:,:) = nan;
    
end

for repInd = 1:50
    cdaPoiss_extrareps.repSpikeDist([repInd:50],repInd,:,:,:) = nan;
end

%% Repeatab
close all;
r2r_percell = squeeze(nanmean(cda.repSpikeDist,[1 2]));
poissr2r = squeeze(nanmean(cdaPoiss.repSpikeDist,[1 2]));
r2r_percell_jit = squeeze(nanmean(jitteredmin.repSpikeDist,[1 2]));

% thisInd = 5;

figure;
tempInd = 2;
poissTrue = 0;

allJit_vals(2:13) = newcdaJit.jitters+[diff(newcdaJit.jitters),.2]./2;
allJit_vals(1) = .00005;

for poissTrue = 0:1
for cellInd = 1:141
realdist   = squeeze(r2r_percell(cellInd,:,tempInd));
jitdist = squeeze(r2r_percell_jit(cellInd,:,:,tempInd));
poissdist = squeeze(poissr2r(cellInd,:,tempInd));
if poissTrue == 1
    realdist = poissdist;
end
diffjit = jitdist - repmat(realdist',1,12);
diffjit(diffjit<0) = 0;
diffjit(diffjit>0) = +1;

for tInd = 1:59
    if sum(diffjit(tInd,:)) == 0
        bestInd(tInd) = 13;
    elseif diffjit(tInd,1) == 1
        bestInd(tInd) = 1;
    else
        bestInd(tInd) = min(find(diff(diffjit(tInd,:))));
    end
end

thesePrecisions(cellInd,:) = allJit_vals(bestInd);
if poissTrue == 0
[h_real(cellInd,:),bins] = hist(thesePrecisions(cellInd,:),allJit_vals);
else
[h_poiss(cellInd,:),bins] = hist(thesePrecisions(cellInd,:),allJit_vals);
end
plot(bins,h_real(cellInd,:),'-')

end
end

cellInd  = 86;
figure; hold on;
plot(bins,h_real(cellInd,:)/59)
plot(bins,h_poiss(cellInd,:)/59)
set(gca,'XScale','log')
xticks([cda.PSTHbins])
xticklabels(cda.PSTHbins*1000)

diff_frompoiss = bsxfun(@(x,y) KLDiv(x,y),h_real,h_poiss);
figure;
scatter(squeeze(nanmean(cda.rates,[2 3])),diff_frompoiss,'filled')
xlabel('firing rate (hz)')
ylabel('JS Div: real vs poisson distributions')
box off;
hold on;
s1 = scatter(squeeze(nanmean(cda.rates(86,:,:),[2 3])),diff_frompoiss(86),'filled');

% mean
for cellInd = 1:141
    cellRes(cellInd) = sum(h_real(cellInd,:)./59.*bins);
    cellRes_poiss(cellInd) = sum(h_poiss(cellInd,:)./59.*bins);
end

% median
for cellInd = 1:141
    cellRes(cellInd) = bins(min(find(cumsum(h_real(cellInd,:)) >= 59/2)));
    cellRes_poiss(cellInd) = bins(min(find(cumsum(h_poiss(cellInd,:)) >= 59/2)));
end

figure; hold on;
s0 = scatter(squeeze(nanmean(cda.rates,[2 3])),cellRes,'filled');
s0.CData = [0 0 0];
set(gca,'YScale','log')
yticks(cda.PSTHbins)
yticklabels(cda.PSTHbins*1000)
box off;
xlabel('firing rate')
ylabel('jitter-determined rep2rep resolution')
s1 = scatter(squeeze(nanmean(cda.rates(cda.PCs,:,:),[2 3])),cellRes(cda.PCs),'filled');
s2 = scatter(squeeze(nanmean(cda.rates(cda.SAs,:,:),[2 3])),cellRes(cda.SAs),'filled');
s1.CData = colorPC; s2.CData = colorSA;



figure; hold on;
s0 = scatter(squeeze(nanmean(cda.rates(find(cdaData.area==3),:,:),[2 3])),cellRes(find(cdaData.area==3)),'filled');
set(gca,'YScale','log')
yticks(cda.PSTHbins)
yticklabels(cda.PSTHbins*1000)
box off;
xlabel('firing rate')
ylabel('jitter-determined rep2rep resolution')
s1 = scatter(squeeze(nanmean(cda.rates(find(cdaData.area==1),:,:),[2 3])),cellRes(find(cdaData.area==1)),'filled');
s2 = scatter(squeeze(nanmean(cda.rates(find(cdaData.area==2),:,:),[2 3])),cellRes(find(cdaData.area==2)),'filled');


figure;
cdfplot(cellRes);
hold on;
cdfplot(cellRes_poiss);
grid off; box off;
set(gca,'XScale','log');
xticks(cda.PSTHbins);
xticklabels(cda.PSTHbins*1000);

cellRes_min = cellRes;
cellRes_poiss_min = cellRes_poiss;

% allcellRes(:,thisInd) = cellRes;
% allcellRes_poiss(:,thisInd) = cellRes_poiss';

%% DIFFERENCE IN DETERMINED JITTER RESOLUTIONS
figure;
cdfplot(abs(cellRes_max([1:55,57:141])-cellRes_min([1:55,57:141])));
hold on;
cdfplot(abs(cellRes_max(cda.PCs)-cellRes_min(cda.PCs)));
cdfplot(abs(cellRes_max(cda.SAs)-cellRes_min(cda.SAs)));
set(gca,'XScale','log')
xticks([.0001 .0002 .0005 .001 .002 .005 .01 .02 .05 .1])
xticklabels([.1 .2 .5 1 2 5 10 20 50 100])

figure;
scatter(cellRes_max,cellRes_min,'filled')
set(gca,'XScale','log')
set(gca,'YScale','log')
xlim([.00001 1])

figure;
cdfplot(cellRes_poiss_max([1:55,57:141]));
hold on;
cdfplot(cellRes_poiss_min([1:55,57:141]));
legend max min maxpoiss minpoiss

%% Classification Figure that explains methods

load('output_psth.mat')
figure;
for gInd = 1:9
subplot(10,1,gInd)
plot(output.psth{gInd,86,35,2})
title(output.psthGaussWidths(gInd)*1000)
end

%% 6. [Plot] Rep by Rep spike distance
data = cda;
qInd = 1;
    
    for hide = 1
        if size(data.spikes,4) == 1
            
            figure;
            data.repSpikeDist =data.repSpikeDist(:,:,:,:,qInd);
            textureSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 3]));
            cellSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 4]));
            
            subplot(2,2,1)
            Violin(textureSpikeDist,1);
            hold on;
            Violin(cellSpikeDist,2);
            xticks([1 2]);
            xticklabels({'across textures' 'across cells'})
            ylabel('avg spike distance')
            title('spike distance across reps')
            
            subplot(2,2,2)
            title('Spike Distance by Submodality')
            PCSpikeDist = squeeze(nanmean(data.repSpikeDist(:,:,data.PCs,:),[1 2 4]));
            SASpikeDist = squeeze(nanmean(data.repSpikeDist(:,:,data.SAs,:),[1 2 4]));
            vpc = Violin(PCSpikeDist,1);
            vpc.ViolinColor = colorPC;
            vsa = Violin(SASpikeDist,2);
            vsa.ViolinColor = colorSA;
            xticks([1 2])
            xticklabels({'PC like' 'SA like'})
            
            subplot(2,2,3); hold on;
            title('firing rate x spike Dist')
            cellrate = nanmean(data.rates,[2 3]);
            p1 = plot(cellrate,cellSpikeDist,'o','color',colorGrey);
            p1.MarkerFaceColor = colorGrey;
            p2 = plot(cellrate(data.PCs),cellSpikeDist(data.PCs),'o','color',colorPC);
            p2.MarkerFaceColor = colorPC;
            p3 = plot(cellrate(data.SAs),cellSpikeDist(data.SAs),'o','color',colorSA);
            p3.MarkerFaceColor = colorSA;
            box off;
            xlabel('cell avg firing rate (Hz)')
            ylabel('cell avg spike distance')
            
            subplot(2,2,4); hold on;
            title('roughness x spikeDist, across cells')
            p1 = plot(data.rough(~isnan(data.rough)),textureSpikeDist(~isnan(data.rough)),'o','color',colorGrey);
            p1.MarkerFaceColor = colorGrey;
            xlabel('roughness')
            ylabel('spikeDist')
            mastertitle(data.title)

            
            figure(); hold on;
            PCtextureSpikeDist = squeeze(nanmean(data.repSpikeDist(:,:,data.PCs,~isnan(data.rough)),[1 2 3]));
            SAtextureSpikeDist = squeeze(nanmean(data.repSpikeDist(:,:,data.SAs,~isnan(data.rough)),[1 2 3]));
            title('roughness x spikeDist, across cells')
            subplot(2,1,1)
            p1 = plot(data.rough(~isnan(data.rough)),textureSpikeDist(~isnan(data.rough)),'o','color',colorGrey);
            p1.MarkerFaceColor = colorGrey;
            box off;
            subplot(2,1,2); hold on;
            p2 = plot(data.rough(~isnan(data.rough)),PCtextureSpikeDist,'o','color',colorPC);
            p2.MarkerFaceColor = colorPC;
            p3 = plot(data.rough(~isnan(data.rough)),SAtextureSpikeDist,'o','color',colorSA);
            p3.MarkerFaceColor = colorSA;
            xlabel('roughness')
            ylabel('spikeDist')
            mastertitle(data.title)
        end
    end
    
%%    
for old2= 1
%% X. MDS of distmat
distmat = fixdistmat(distmat);
distmat = nanmean(distmat,[3 4 5]);
for tInd1 = 1:size(distmat,1)
    distmat(tInd1,tInd1) = 0;
end

[Y,eigvals] = cmdscale(distmat);
figure;
plot(Y(:,1),Y(:,2),'o')
text(Y(:,1)+.1,Y(:,2),cda.textures(dissimData.textInd))

%% SEARCHTERM Rate across areas
load('Rate_Euc_cda.mat');
distmat = fixdistmat(distmat);

for areaInd = [3 1 2]
areaCells = find(cdaData.area == areaInd);
for popSize = 1:25
for itInd = 1:20
    chosencells = datasample(areaCells,popSize,'replace',false);
    popdistmat = nanmean(distmat(:,:,:,:,chosencells),5);
    results = GetResults(nanmean(popdistmat,3));
    itResults(areaInd,popSize,itInd) = nanmean(results,[1 2]);
end
end
    meanrates(areaInd,:) = nanmean(cda.rates(areaCells,:,:),[ 2 3]);
end

figure;
areacolors = lines(3);
for areaInd = 1:3
    hold on;
    errorshadeplot_nonlog(1:25,squeeze(nanmean(itResults(areaInd,:,:),3)),squeeze(nanstd(itResults(areaInd,:,:),[],3)),areacolors(areaInd,:))
end

figure;
areacolors = lines(3);
areacolors = areacolors([2 3 1],:);
for areaInd = 1:3
    areaCells = find(cdaData.area == areaInd);
    areaRates = nanmean(cda.rates(areaCells,:,:),[2 3]);
    areaPerf = cda.RATEresults(areaCells);
    hold on;
    plot(areaRates,areaPerf,'o','color',areacolors(areaInd,:));
end

figure;
for areaInd = 1:3
subplot(3,1,areaInd)
areaCells = find(cdaData.area == areaInd);
areaRates=nanmean(cda.rates(areaCells,:,:),[2 3]);
hist(areaCells);
xlim([0 180]); ylim([0 15])
end

figure; hold on;
areaInd = 2
    areaCells = find(cdaData.area == areaInd);
    areaPerf = cda.RATEresults(areaCells);
    Violin(areaPerf,3);



%% How well do PSTH across cells correlate?
for qInd = 1:9
for cellInd = 1:141
    for tInd = 1:59
        for repInd = 1:5
            thispsth(repInd,:) = output.psth{qInd,cellInd,tInd,repInd};
        end
            thesepsths(qInd,cellInd,tInd) = nanmean(thispsth,1);
    end
end
end

%% 7. Compare real data to poisson
for hide = 1
data = cda;
textureSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 3]));
cellSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 4]));

figure;
errorshadeplot(1:length(qvals),nanmean(cellSpikeDist,1),nanstd(cellSpikeDist,[],1),'k')
xticks(1:length(qvals));
xticklabels(1000*qvals)
ylabel('avg spike distance across reps (error across cells)')
title('spike distance across reps')
xlabel('qvals')

data = cdaPoiss;
textureSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 3]));
cellSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 4]));

hold on;
errorshadeplot(1:length(qvals),nanmean(cellSpikeDist,1),nanstd(cellSpikeDist,[],1),colorTiming)
xticks(1:length(qvals));
xticklabels(1000*qvals)
ylabel('avg spike distance across reps (error across cells)')
title('spike distance across reps')
xlabel('qvals')

% Submodality

data = cda;
figure;
title('Spike Distance by Submodality')
PCSpikeDist = squeeze(nanmean(data.repSpikeDist(:,:,data.PCs,:,:),[1 2 4]));
SASpikeDist = squeeze(nanmean(data.repSpikeDist(:,:,data.SAs,:,:),[1 2 4]));
subplot(2,1,1); hold on;
errorshadeplot(1:length(qvals),nanmean(PCSpikeDist,1),nanstd(PCSpikeDist,[],1),'k')
xticks(1:length(qvals));
xticklabels(1000*qvals)
ylabel('avg spike distance across reps (error across cells)')
title('spike distance across reps')
xlabel('qvals')
subplot(2,1,2); hold on;
errorshadeplot(1:length(qvals),nanmean(SASpikeDist,1),nanstd(SASpikeDist,[],1),'k')
xticks(1:length(qvals));
xticklabels(1000*qvals)
ylabel('avg spike distance across reps (error across cells)')
title('spike distance across reps')
xlabel('qvals')

data = cdaPoiss;
title('Spike Distance by Submodality')
PCSpikeDist = squeeze(nanmean(data.repSpikeDist(:,:,data.PCs,:,:),[1 2 4]));
SASpikeDist = squeeze(nanmean(data.repSpikeDist(:,:,data.SAs,:,:),[1 2 4]));
subplot(2,1,1); hold on;
errorshadeplot(1:length(qvals),nanmean(PCSpikeDist,1),nanstd(PCSpikeDist,[],1),colorPC)
xticks(1:length(qvals));
xticklabels(1000*qvals)
ylabel('avg spike distance across reps (error across cells)')
title('spike distance across reps')
xlabel('qvals')
subplot(2,1,2); hold on;
errorshadeplot(1:length(qvals),nanmean(SASpikeDist,1),nanstd(SASpikeDist,[],1),colorSA)
xticks(1:length(qvals));
xticklabels(1000*qvals)
ylabel('avg spike distance across reps (error across cells)')
title('spike distance across reps')
xlabel('qvals')

end

%% 8. Compare real data to t-v poisson
for hide = 1
figure;
data = cdaTVPoiss;
textureSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 3]));
cellSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 4]));
for jitInd = 1:length(jitters)
    hold on;
    errorshadeplot(1:length(qvals),squeeze(nanmean(cellSpikeDist(:,jitInd,:),1))',squeeze(nanstd(cellSpikeDist(:,jitInd,:),[],1))',colorJitters(jitInd,:))
end
data = cda;
textureSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 3]));
cellSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 4]));
errorshadeplot(1:length(qvals),nanmean(cellSpikeDist,1),nanstd(cellSpikeDist,[],1),'k')
xticks(1:length(qvals));
xticklabels(1000*qvals)
ylabel('avg spike distance across reps (error across cells)')
title('spike distance across reps')
xlabel('qvals')

% THATS HARD TO SEE, so let's plot it without error
figure;
data = cdaTVPoiss;
textureSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 3]));
cellSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 4]));
for jitInd = 1:length(jitters)
    v(jitInd) = Violin(cellSpikeDist(:,jitInd,1),jitInd);
    v(jitInd).ViolinColor = colorJitters(jitInd,:);
    hold on;
end
data = cda;
textureSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 3]));
cellSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 4]));
loglog(1:length(qvals),nanmean(cellSpikeDist,1),'color','k')
xticks(1:length(qvals));
xticklabels(1000*qvals)
ylabel('avg spike distance across reps (error across cells)')
title('spike distance across reps')
xlabel('qvals')



% Submodality

data = cda;
figure;
title('Spike Distance by Submodality')
PCSpikeDist = squeeze(nanmean(data.repSpikeDist(:,:,data.PCs,:,:),[1 2 4]));
SASpikeDist = squeeze(nanmean(data.repSpikeDist(:,:,data.SAs,:,:),[1 2 4]));
subplot(2,1,1); hold on;
errorshadeplot(1:length(qvals),nanmean(PCSpikeDist,1),nanstd(PCSpikeDist,[],1),'k')
xticks(1:length(qvals));
xticklabels(1000*qvals)
ylabel('avg spike distance across reps (error across cells)')
title('spike distance across reps')
xlabel('qvals')
subplot(2,1,2); hold on;
errorshadeplot(1:length(qvals),nanmean(SASpikeDist,1),nanstd(SASpikeDist,[],1),'k')
xticks(1:length(qvals));
xticklabels(1000*qvals)
ylabel('avg spike distance across reps (error across cells)')
title('spike distance across reps')
xlabel('qvals')

data = cdaPoiss;
title('Spike Distance by Submodality')
PCSpikeDist = squeeze(nanmean(data.repSpikeDist(:,:,data.PCs,:,:),[1 2 4]));
SASpikeDist = squeeze(nanmean(data.repSpikeDist(:,:,data.SAs,:,:),[1 2 4]));
subplot(2,1,1); hold on;
errorshadeplot(1:length(qvals),nanmean(PCSpikeDist,1),nanstd(PCSpikeDist,[],1),colorPC)
xticks(1:length(qvals));
xticklabels(1000*qvals)
ylabel('avg spike distance across reps (error across cells)')
title('spike distance across reps')
xlabel('qvals')
subplot(2,1,2); hold on;
errorshadeplot(1:length(qvals),nanmean(SASpikeDist,1),nanstd(SASpikeDist,[],1),colorSA)
xticks(1:length(qvals));
xticklabels(1000*qvals)
ylabel('avg spike distance across reps (error across cells)')
title('spike distance across reps')
xlabel('qvals')

end

%% 9. Compare real data to jittered
for hide = 1
figure;
data = cdaJit;
textureSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 3]));
cellSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 4]));
for jitInd = 1:length(jitters)
    hold on;
    errorshadeplot(1:length(qvals),squeeze(nanmean(cellSpikeDist(:,jitInd,:),1))',squeeze(nanstd(cellSpikeDist(:,jitInd,:),[],1))',colorJitters(jitInd,:))
end
data = cda;
textureSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 3]));
cellSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 4]));
errorshadeplot(1:length(qvals),nanmean(cellSpikeDist,1),nanstd(cellSpikeDist,[],1),'k')
xticks(1:length(qvals));
xticklabels(1000*qvals)
ylabel('avg spike distance across reps (error across cells)')
title('spike distance across reps')
xlabel('qvals')

% THATS HARD TO SEE, so let's plot it without error
figure;
data = cdaJit;
textureSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 3]));
cellSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 4]));
for jitInd = 1:length(jitters)
    p = plot(1:length(qvals),squeeze(nanmean(cellSpikeDist(:,jitInd,:),1))','color',colorJitters(jitInd,:),'linewidth',2);
    p.Color(4) = .4;
        hold on;
end
data = cda;
textureSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 3]));
cellSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 4]));
plot(1:length(qvals),nanmean(cellSpikeDist,1),'color','k','linewidth',3)
xticks(1:length(qvals));
xticklabels(1000*qvals)
ylabel('avg spike distance across reps (error across cells)')
title('spike distance across reps')
xlabel('qvals')

legend([cellfun(@(x) num2str(x),num2cell(1000*jitters),'uniformoutput',0) {'real'}])
legend boxoff
box off

% Submodality
figure;
subplot(2,1,1);
data = cdaJit;
cells = cdaJit.PCs;
cellSpikeDist = squeeze(nanmean(data.repSpikeDist(:,:,cells,:,:,:),[1 2 4]));
for jitInd = 1:length(jitters)
    p = plot(1:length(qvals),squeeze(nanmean(cellSpikeDist(:,jitInd,:),1))','color',colorJitters(jitInd,:),'linewidth',2);
    p.Color(4) = .4;
        hold on;
end
data = cda;
textureSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 3]));
cellSpikeDist = squeeze(nanmean(data.repSpikeDist(:,:,cells,:,:,:),[1 2 4]));
plot(1:length(qvals),nanmean(cellSpikeDist,1),'color',colorPC,'linewidth',3)
xticks(1:length(qvals));
xticklabels(1000*qvals)
ylabel('avg spike distance across reps (error across cells)')
title('spike distance across reps')
xlabel('qvals')
legend([cellfun(@(x) num2str(x),num2cell(1000*jitters),'uniformoutput',0) {'PC'}])
legend boxoff
box off

subplot(2,1,2);
data = cdaJit;
cells = cdaJit.SAs;
cellSpikeDist = squeeze(nanmean(data.repSpikeDist(:,:,cells,:,:,:),[1 2 4]));
for jitInd = 1:length(jitters)
    p = plot(1:length(qvals),squeeze(nanmean(cellSpikeDist(:,jitInd,:),1))','color',colorJitters(jitInd,:),'linewidth',2);
    p.Color(4) = .4;
        hold on;
end
data = cda;
textureSpikeDist = squeeze(nanmean(data.repSpikeDist,[1 2 3]));
cellSpikeDist = squeeze(nanmean(data.repSpikeDist(:,:,cells,:,:,:),[1 2 4]));
plot(1:length(qvals),nanmean(cellSpikeDist,1),'color',colorSA,'linewidth',3)
xticks(1:length(qvals));
xticklabels(1000*qvals)
ylabel('avg spike distance across reps (error across cells)')
title('spike distance across reps')
xlabel('qvals')
legend([cellfun(@(x) num2str(x),num2cell(1000*jitters),'uniformoutput',0) {'SA'}])
legend boxoff
box off

end

%% 10. repxrep rate
cdaRate = load('Rate_Euc_cda.mat');
cda.distmat = fixdistmat(cdaRate.distmat);
PoissRate = load('Rate_Euc_cdaPoiss.mat');
cdaPoiss.distmat = fixdistmat(PoissRate.distmat);

for data = [cda cdaPoiss]

    for hide = 1
            
            figure;
            distmat = data.distmat;
            cellSpikeDist = squeeze(nanmean(distmat,[1 2 3]));
            textureSpikeDist = squeeze(nanmean(distmat,[2 3 4]));
            subplot(2,2,1)
            Violin(textureSpikeDist,1);
            hold on;
            Violin(cellSpikeDist,2);
            xticks([1 2]);
            xticklabels({'across textures' 'across cells'})
            ylabel('avg euc dist')
            title('rate diff across reps')
            
            subplot(2,2,2)
            title('Rate Diff by Submodality')
            PCSpikeDist = squeeze(nanmean(distmat(:,:,:,data.PCs),[1 2 3]));
            SASpikeDist = squeeze(nanmean(distmat(:,:,:,data.SAs),[1 2 3]));
            vpc = Violin(PCSpikeDist,1);
            vpc.ViolinColor = colorPC;
            vsa = Violin(SASpikeDist,2);
            vsa.ViolinColor = colorSA;
            xticks([1 2])
            xticklabels({'PC like' 'SA like'})
            
            subplot(2,2,3); hold on;
            title('firing rate x rate diff')
            cellrate = nanmean(data.rates,[2 3]);
            p1 = plot(cellrate,cellSpikeDist,'o','color',colorGrey);
            p1.MarkerFaceColor = colorGrey;
            p2 = plot(cellrate(data.PCs),cellSpikeDist(data.PCs),'o','color',colorPC);
            p2.MarkerFaceColor = colorPC;
            p3 = plot(cellrate(data.SAs),cellSpikeDist(data.SAs),'o','color',colorSA);
            p3.MarkerFaceColor = colorSA;
            box off;
            xlabel('cell avg firing rate (Hz)')
            ylabel('cell avg rate diff')
            
            subplot(2,2,4); hold on;
            title('roughness x rate diff, across cells')
            p1 = plot(data.rough(~isnan(data.rough)),textureSpikeDist(~isnan(data.rough)),'o','color',colorGrey);
            p1.MarkerFaceColor = colorGrey;
            xlabel('roughness')
            ylabel('rate diff')
            mastertitle(data.title)

        end
end
    

end
%% 11. [KEEP] Create results matrices for SINGLE cells
distmat = fixdistmat(distmat);
results = squeeze(GetResults_cov(nanmean(distmat,4)));
results = squeeze(nanmean(results,[1 2]));
size(results)

% cdaJit.SpikeDistresults(:,:,jitInd) = results;
% cdaPoiss.ISIbins = cda.ISIbins;

%% 12. [KEEP] Create results matrices for POPULATIONS of cells
data= cda;
size(distmat)
distmat = fixdistmat(distmat);
cells2include = bestcells;
[cellMax, cellRes ] =max(data.SpikeDistresults,[],2);
for cellInd = 1:length(cells2include)
    alldistmat(:,:,:,:,cellInd) = distmat(:,:,:,:,cellInd,cellRes(cellInd));
end
distmat= alldistmat;

for popSize = 1:length(cells2include)
    for itInd = 1:1000
    popCells = datasample(cells2include,popSize,'replace',false);
    results = squeeze(GetResults(nanmean(distmat(:,:,:,:,popCells),[4 5])));
    popresults_spd(popSize,itInd) = squeeze(nanmean(results,[1 2]));
        end
    end


% cdaJit.SpikeDistresults(:,:,jitInd) = results;
% cdaPoiss.ISIbins = cda.ISIbins;

figure;
errorshadeplot_nonlog([1:size(popresults_spd,1)],nanmean(popresults_spd,2)',nanstd(popresults_spd,[],2)')
ylim([0 1])

%% [KEEP] Create results matrices
for trainForce = 1:3
    for testForce = 1:3
        distmat = squeeze(realdistmat(:,:,:,:,trainForce,testForce,:,:));
size(distmat)
distmat = fixdistmat(distmat);
results = squeeze(GetResults(nanmean(distmat,[4 5])));
results = squeeze(nanmean(results,[1 2]));
forceresults_spd(trainForce,testForce,:,:) = results;
    end
end

%% Plot heatmap
theseCells = cda.RAs;
thisdistmat = distmat(:,:,:,:,5,theseCells);
theseRates = squeeze(nanmean(cda.rates(theseCells,:,:),[1 3]));
[~,rateOrder] = sort(theseRates);

figure;
imagesc(nanmean(thisdistmat(rateOrder,rateOrder,:,:,:,:,:),[3 4 5 6]))
caxis([0.5 2.5])
axis off

%% [KEEP] Plot results (for ISI, PSTH, or SpikeDist)
figure;
data = per;
results = per.PSTHresults;
bins = cda.PSTHbins;
% bins = [.001 .002 .005 .01 .02 .05 .1 .2 .5];
% bins = finalqvals;
% bins = qvalsFinal;
    % FOR SPIKEDIST ONLY:
%     bins(bins == Inf) = 1;
%     binlabels = [num2cell(bins(1:end-1)*1000) {'Inf'}];
titleHere = 'spikeDist classification, cortical data';
hasJitters = 0;
modeldata = 0;

for hide = 1
if hasJitters == 0
hold on;
colors = [colorRA;colorPC; colorSA; [0 0 0]];
choosecells = [{data.RAs} {data.PCs} {data.SAs} {data.allcells}];
% choosecells = [{cdaData.area == 1} {cdaData.area==2} {cdaData.area==3} {cda.allcells}]

for groupInd = 1:4
    cells = choosecells{groupInd};
    groupresults = nanmean(results(cells,:),1);
    grouperror   = nanstd(results(cells,:),[],1)./(sqrt(length(cells)));
    e = errorshadeplot(1000*bins,groupresults,grouperror,colors(groupInd,:));
    box off; hold on;
    if modeldata == 1
        e.LineStyle = '--';
        e.Color(4) = .3;
    end
end
xticks([qvals(1:end-1)*1000 1000])
if exist('binlabels')
    xticklabels([qvals(1:end-1)*1000 Inf])
end
ylim([0 .5])


xlabel('temporal resolution (ms)')
ylabel('single cell classification performance')
clear binlabels;
title(titleHere)

else
for jitInd = 1:4
subplot(2,2,jitInd)
hold on;
colors = [0 0 0;colorPC; colorSA];
choosecells = [{cda.allcells} {cda.PCs} {cda.SAs}];

for groupInd = 1:3
    cells = choosecells{groupInd};
    groupresults = nanmean(results(cells,:,jitInd),1);
    grouperror   = nanstd(results(cells,:,jitInd),[],1)./(sqrt(length(cells)));
    e = errorshadeplot(1000*bins,groupresults,grouperror,colors(groupInd,:));
    box off; hold on;
    if modeldata == 1
    e.LineStyle = '--';
    e.Color(4) = .3;
    end
end
xticks([1000*bins])
if exist('binlabels')
    xticklabels([num2cell(bins(1:end-1)) {'Inf'}])
end
ylim([0 1])


xlabel('temporal resolution (ms)')
ylabel('single cell classification performance')
clear binlabels;
title(jitters(jitInd))
mastertitle(titleHere)
end
end
end

figure; hold on;
[cellMax,cellRes] = max_conserv(results,2);
cellRes = bins(cellRes);
% cells2keep = find(cellMax < 20/59);
% cellRes(cells2keep) = nan;
for groupInd = 1:4
    cells = choosecells{groupInd};
    l = cdfplot(cellRes(cells));
    l.Color = colors(groupInd,:);
end
set(gca,'XScale','log')
xticks(cda.PSTHbins)
xticklabels(1000*cda.PSTHbins);

%% [KEEP] Plot results (for ISI, PSTH, or SpikeDist) ONE INDIVIDUAL CELL
% load('cdaFull.mat')
figure;
data = cda;
results = data.PSTHresults;
bins = data.PSTHbins;
cell = inds(70); %100 % 86 IS THE EXAMPLE CELL IN THE PAPER
modeldata = 0;

% bins = finalqvals;
% bins = qvalsFinal;
    % FOR SPIKEDIST ONLY:
%     bins(bins == Inf) = 1;
%     binlabels = [num2cell(bins(1:end-1)*1000) {'Inf'}];
titleHere = 'spikeDist classification, cortical data';
hasJitters = 0;

    groupresults = nanmean(results(cell,:),1);
    e = errorshadeplot(bins,groupresults,zeros(1,length(groupresults)),[0 0 0]);
    box off; hold on;
    if modeldata == 1
        e.LineStyle = '--';
        e.Color(4) = .3;
    end

    xticks(cda.PSTHbins)
xticklabels(1000*cda.PSTHbins)
ylim([0 .5])


xlabel('temporal resolution (ms)')
ylabel('single cell classification performance')
clear binlabels;
title(titleHere)

hold on;
data = cdaPoiss;
results = data.PSTHresults;
bins = data.PSTHbins;
modeldata = 1;
    % FOR SPIKEDIST ONLY:
%     bins(bins == Inf) = 1;
%     binlabels = [num2cell(bins(1:end-1)*1000) {'Inf'}];


    groupresults = nanmean(results(cell,:),1);
    e = errorshadeplot(bins,groupresults,zeros(1,length(groupresults)),[0 0 0]);
    box off; hold on;
    if modeldata == 1
        e.LineStyle = '--';
        e.Color(4) = .3;
    end

%% Violin plot comparing performance
 data = cda;   
figure;
v1 = Violin(data.RATEresults,1)
% plot(v1.ScatterPlot.XData(1,86),v1.ScatterPlot.YData(1,86),'x','color','w')
v2 = Violin(max(data.PSTHresults,[],2),2)
% plot(v2.ScatterPlot.XData(1,86),v2.ScatterPlot.YData(1,86),'x','color','w')
v3 = Violin(max(data.SpikeDistresults,[],2),3)
% plot(v3.ScatterPlot.XData(1,86),v3.ScatterPlot.YData(1,86),'x','color','w')

%% Optimal 

%% Plot PSTH / SpikeDist results ACROSS AREAS
figure;
data = cda;
results = cda.PSTHresults;
bins = cda.PSTHbins;
    % FOR SPIKEDIST ONLY:
%     bins(bins == Inf) = 1;
%     binlabels = [num2cell(bins(1:end-1)*1000) {'Inf'}];
titleHere = 'spikeDist classification, cortical data';
hasJitters = 0;
modeldata = 0;

for hide = 1
if hasJitters == 0
hold on;
colors = [parula(3)];
choosecells = [{find(cdaData.area == 3)} {find(cdaData.area == 1)} {find(cdaData.area == 2)}];

for groupInd = 1:3
    cells = choosecells{groupInd};
    groupresults = nanmean(results(cells,:),1);
    grouperror   = nanstd(results(cells,:),[],1)./(sqrt(length(cells)));
    e = errorshadeplot(1000*bins,groupresults,grouperror,colors(groupInd,:));
    box off; hold on;
    if modeldata == 1
        e.LineStyle = '--';
        e.Color(4) = .3;
    end
end
xticks([qvals(1:end-1)*1000 1000])
if exist('binlabels')
    xticklabels([qvals(1:end-1)*1000 Inf])
end
ylim([0 .5])


xlabel('temporal resolution (ms)')
ylabel('single cell classification performance')
clear binlabels;
title(titleHere)

else
for jitInd = 1:4
subplot(2,2,jitInd)
hold on;
colors = [0 0 0;colorPC; colorSA];
choosecells = [{cda.allcells} {cda.PCs} {cda.SAs}];

for groupInd = 1:3
    cells = choosecells{groupInd};
    groupresults = nanmean(results(cells,:,jitInd),1);
    grouperror   = nanstd(results(cells,:,jitInd),[],1)./(sqrt(length(cells)));
    e = errorshadeplot(1000*bins,groupresults,grouperror,colors(groupInd,:));
    box off; hold on;
    if modeldata == 1
    e.LineStyle = '--';
    e.Color(4) = .3;
    end
end
xticks([1000*bins])
if exist('binlabels')
    xticklabels([num2cell(bins(1:end-1)) {'Inf'}])
end
ylim([0 1])


xlabel('temporal resolution (ms)')
ylabel('single cell classification performance')
clear binlabels;
title(jitters(jitInd))
mastertitle(titleHere)
end
end
end
%% [Plot, Skip, KEEP] Compare PSTH, Rate, ISI, and SpikeDist between Poisson and Real data
for hide = 1
figure;
% PSTH

realPSTHresults = max(cda.PSTHresults,[],2);
PoissPSTHresults = max(cdaPoiss.PSTHresults,[],2);

subplot(2,2,1)
plot(realPSTHresults,PoissPSTHresults,'o')
ylim([0 1])
xlim([0 1])
box off;
xlabel('Single cell classification')
ylabel('Poisson matched cell classification')
title('PSTH comparison')

% ISI

realISIresults = max(cda.ISIresults,[],2);
PoissISIresults = max(cdaPoiss.ISIresults,[],2);

subplot(2,2,2)
plot(realISIresults,PoissISIresults,'o')
ylim([0 1])
xlim([0 1])
box off;
xlabel('Single cell classification')
ylabel('Poisson matched cell classification')
title('ISI comparison')


% RATE

realRateresults = cda.RATEresults;
PoissRateresults = cdaPoiss.RATEresults;

subplot(2,2,3)
plot(realRateresults,PoissRateresults,'o')
ylim([0 1])
xlim([0 1])
box off;
xlabel('Single cell classification')
ylabel('Poisson matched cell classification')
title('Rate comparison')


% SpikeDist

realSpikeDistresults = max(cda.SpikeDistresults(:,1:3),[],2);
PoissSpikeDistresults = max(cdaPoiss.SpikeDistresults(:,1:3),[],2);

subplot(2,2,4)
plot(realSpikeDistresults,PoissSpikeDistresults,'o')
ylim([0 1])
xlim([0 1])
box off;
xlabel('Single cell classification')
ylabel('Poisson matched cell classification')
title('SpikeDist comparison')
end

%% [Plot] Best temporal resolution using SpikeDist
figure;
data = cda;
[cellMax,cellRes]  = max(data.PSTHresults,[],2);
cellRes = data.PSTHbins(cellRes);

% [cellMax,cellRes]  = max(meanresults,[],2);
% cellRes = qvalsFinal(cellRes);

cellRes(isnan(cellMax)) = nan;
cellRes(cellMax<.3) = nan;
% cellRes = cellRes(cdaData.area==2)
% [cellMax,cellRes]  = max(meanresults,[],2);
% cellRes = qvals_new(cellRes);

hold on;
cAll = cdfplot(cellRes); hold on;
cAll.Color = colorGrey;
cAll.LineWidth = 2;
cPC = cdfplot(cellRes(data.PCs)); hold on;
cPC.Color = colorPC;
cPC.LineWidth = 2;
cSA = cdfplot(cellRes(data.SAs));
cSA.Color = colorSA;
cSA.LineWidth = 2;
box off; grid off;
xlabel('temporal resolution')
ylabel('fraction of cells')
title('temporal resolution of peripheral cells')

set(gca,'XScale','log')
xticks([.0001 .0002 .0005 cda.SpikeDistbins]);
xticklabels([.0001 .0002 .0005 cda.SpikeDistbins .5]*1000);

%% [Plot] Best temporal resolution using PSTH
% Get rid of low firing rates


data = cda;
% lowratecells = find(nanmean(data.rates,[2 3]) < 40);
% data.PSTHresults(lowratecells,:) = nan;
[cellMax,cellRes]  = max(data.PSTHresults,[],2);
cellRes = data.PSTHbins(cellRes);
cellRes(isnan(cellMax)) = nan;
cellRes(cellMax<0.1) = nan;

for cellInd = 1:length(cellMax)
    possibleresolutions = find(data.PSTHresults(cellInd,:) == cellMax(cellInd));
    cellMax(cellInd) = max(possibleresolutions);
end

% cellRes(cellMax<0.1) = nan;
% cells2exclude = find(cdaData.area~=2);
% cellRes(cells2exclude) = nan;
% % 
% cellRes = cellRes(cdaData.area==3)
% [cellMax,cellRes]  = max(meanresults,[],2);
% cellRes = qvals_new(cellRes);

figure;
hold on;
cAll = cdfplot(cellRes); hold on;
cAll.Color = colorGrey;
cAll.LineWidth = 2;
cPC = cdfplot(cellRes(data.PCs)); hold on;
cPC.Color = colorPC;
cPC.LineWidth = 2;
cSA = cdfplot(cellRes(data.SAs));
cSA.Color = colorSA;
cSA.LineWidth = 2;
box off; grid off;
xlabel('temporal resolution')
ylabel('fraction of cells')
title('PSTH temporal resolution of cortical cells')

 set(gca,'XScale','log')
xticks(data.PSTHbins);
xticklabels(data.PSTHbins*1000);

%% the general scatter
x= maxCoeff;
y = cellRes;
cda.RAs = pRegCoeff(2,:) > .8;

figure; hold on;
scatter(x,y,'filled','k')
scatter(x(cda.PCs),y(cda.PCs),[],colorPC,'filled')
scatter(x(cda.SAs),y(cda.SAs),[],colorSA,'filled')
scatter(x(cda.RAs),y(cda.RAs),[],'b','filled')

%% texture effects
textureprecision = squeeze(nanmean(cda.repSpikeDist(:,:,:,:,2),[1 2 3]));


%% 
data = per;
figure;
plot(cellRes,cellMax,'o')
set(gca,'XScale','log')
figure;
plot(cellRes(data.PCs),cellMax(data.PCs),'o','color',colorPC)
hold on;
plot(cellRes(data.SAs),cellMax(data.SAs),'o','color',colorSA)
set(gca,'XScale','log')
box off;
xticks(qvals)
xticklabels(qvals*1000)

%% Best temporal resolution using SpikeDist FOR AREAS
data = cda;
[cellMax,cellRes]  = max(data.SpikeDistresults,[],2);
cellRes = data.SpikeDistbins(cellRes);
cells2exclude = find(cdaData.area~=2);
cellRes(cells2exclude) = nan;

figure
subplot(3,1,3)
cAll = cdfplot(cellRes); hold on;
cAll.Color = colorGrey;
cAll.LineWidth = 2;
cPC = cdfplot(cellRes(data.PCs)); hold on;
cPC.Color = colorPC;
cPC.LineWidth = 2;
cSA = cdfplot(cellRes(data.SAs));
cSA.Color = colorSA;
cSA.LineWidth = 2;
box off; grid off;
xlabel('temporal resolution')
ylabel('fraction of cells')
title('temporal resolution of area 2')

xlim([0 .5])
set(gca,'XScale','log')
xticks(qvals)
xticklabels(qvals*1000)

% Violin(cellRes,1)

%% Best temporal resolution using SpikeDist FOR AREAS
data = cda;
[cellMax,cellRes]  = max(data.PSTHresults,[],2);
cellRes = data.PSTHbins(cellRes);
cells2exclude = find(cdaData.area~=2);
cellRes(cells2exclude) = nan;

figure
subplot(3,1,3)
cAll = cdfplot(cellRes); hold on;
cAll.Color = colorGrey;
cAll.LineWidth = 2;
cPC = cdfplot(cellRes(data.PCs)); hold on;
cPC.Color = colorPC;
cPC.LineWidth = 2;
cSA = cdfplot(cellRes(data.SAs));
cSA.Color = colorSA;
cSA.LineWidth = 2;
box off; grid off;
xlabel('temporal resolution')
ylabel('fraction of cells')
title('temporal resolution of area 2')

xlim([0 .5])
set(gca,'XScale','log')
xticks(qvals)
xticklabels(qvals*1000)

% Violin(cellRes,1)

%% Does the texture set influence temporal resolution?

distmat = fixdistmat(distmat);
nsampletextures = 10;
nInds = 30;

for tInd = 1:nInds
    tSet = datasample(1:size(distmat,1),nsampletextures,'replace',false);
    sampleDistMat = distmat(tSet,tSet,:,:,:,:,:,:,:,:,:);
    sampleResults = GetResults(nanmean(sampleDistMat,4));
    sampleResults_Ind(:,:,tInd) = squeeze(nanmean(sampleResults,[1 2]));
end
    
%% Confusion Matrix
figure;

for qInd = 1:9
distmat_here = fixdistmat(distmat(:,:,:,:,:,qInd));
rates = data.rates;
[~,rateOrder] = sort(nanmean(rates,[1 3]));

subplot(3,3,qInd);
imagesc(nanmean(distmat_here(rateOrder,rateOrder,:,:,:),[3 4 5]))
title(data.SpikeDistbins(qInd))
end

%% Perception heatmap
figure;

% for qInd = 1:9
distmat_here = fixdistmat(distmat(:,:,:,:,data.PCs));
order = dissimData.textInd;

% subplot(3,3,qInd);
imagesc(nanmean(distmat_here(order,order,:,:,:),[3 4 5]))
% title(data.SpikeDistbins(qInd))
% end

%% Perception correlation of distmats
data = cda;
ncells = size(data.spikes,1);
thisdistmat = squeeze(nanmean(fixdistmat(distmat(order,order,:,:,:,3)),[3 4]));
thisdistmat = reshape(thisdistmat,169,141);
perceptdistmat = reshape(dissimData.mat,169,1);

for cellInd = 1:ncells
    subplot(12,12,cellInd)
    plot(zscore(thisdistmat(:,cellInd)),perceptdistmat,'.')
    regress(perceptdistmat,zscore(thisdistmat(:,cellInd)))
    box off
end
    

%% [IMPORTANT KEEP] Fit perceptual data
% Note: Load a distmat!
data = cda;
dissimPerceptual = dissimData.mat;
for tInd = 1:13
dissimPerceptual(tInd,tInd) = nan;
distmat(dissimData.textInd(tInd),dissimData.textInd(tInd),:,:,:,:) = nan;
end
dissimPerceptual = dissimPerceptual(:);

for qInd1 = 1:9
for qInd2 = 1:9
perceptualdistmat = distmat(dissimData.textInd,dissimData.textInd,:,:,:,:);
fun = @(b,inputdata) b(1)*inputdata(1,:)+b(2)*inputdata(2,:)+b(3);
PCdata = reshape(permute(squeeze(nanmean(perceptualdistmat(:,:,:,:,data.PCs(1:5),qInd1),[3 4])),[3 1 2]),[5,size(perceptualdistmat,1)^2]);
SAdata = reshape(permute(squeeze(nanmean(perceptualdistmat(:,:,:,:,data.SAs(1:5),qInd2),[3 4])),[3 1 2]),[5,size(perceptualdistmat,1)^2]);

PCdata= nanmean(PCdata,1)';
SAdata=  nanmean(SAdata,1)';

[b,~,~,~,stats(:,qInd1,qInd2)] = regress(dissimPerceptual',[ones(size(PCdata))  PCdata SAdata]);
end
end

figure;
imagesc(squeeze(stats(1,:,:)))

for cellInd = 1:141
for qInd1 = 1:9
perceptualdistmat = nanmean(distmat(dissimData.textInd,dissimData.textInd,:,:,cellInd,qInd),[3 4]);
fun = @(b,inputdata) b(1)*inputdata(1,:,b(4))+b(2)*inputdata(2,:,b(5))+b(3);
singlecelldata= perceptualdistmat(:);

[b,~,~,~,stats(:,cellInd,qInd)] = regress(dissimPerceptual',[ones(size(singlecelldata))  singlecelldata]);
end
end

figure;
for qInd = 1:9
for cellInd = 1:141
perceptualdistmat = nanmean(distmat(dissimData.textInd,dissimData.textInd,:,:,cellInd,qInd),[3 4]);
[~,~,~,~,stats(cellInd,:)] = regress(dissimPerceptual,[ones(size(perceptualdistmat(:))),perceptualdistmat(:)]);
end

subplot(3,3,qInd)
vAll = Violin(stats(:,1),1); hold on;
vAll.ViolinColor = colorGrey;
vPc = Violin(stats(cda.PCs,1),2);
vPc.ViolinColor = colorPC;
vSA = Violin(stats(cda.SAs,1),3);
vSA.ViolinColor = colorSA;
xticks([1 2 3])
xticklabels({'all' 'PC' 'SA'})
end


figure;
plot(cda.SpikeDistbins,squeeze(stats(1,cda.PCs,:)))


%% still workin on this
b0 = [1,1,1];
for itInd = 1:100
b(:,:,:,itInd) = lsqcurvefit(fun,b0,[PCdata; SAdata],dissimPerceptual);
end

alldatavec = linspace(min(alldata',[],1),max(alldata',[],1));
plot(alldata',dissimPerceptual,'ko',alldatavec,fun(b(:,:,:,1)',alldatavec),'b-')
%% Compare single cell rate vs timing classification 
figure;
plot(cda.RATEresults, max(cda.SpikeDistresults,[],2),'o');
box off;
xlim([0 1])
ylim([0 1])
xlabel('rate classification')
ylabel('spike dist timing classification')
title('single cell classification')

    
    
%% 10. Plot Bootstrap to Poisson extrarep data
modeldata = cdaPoiss_extrareps.repSpikeDist;
realdata = cda.repSpikeDist;
for cellInd = 1:141
    for tInd = 1:59
        for qInd = 1:7
            [pvals(cellInd,tInd,qInd) meanpvals(cellInd,tInd,qInd)] = bootstrappval(realdata(:,:,cellInd,tInd,qInd),modeldata(:,:,cellInd,tInd,qInd),500);
        end
    end
end

figure
for tInd = 1:59
    subplot(8,8,tInd)
v1= Violin(meanpvals(cda.PCs,tInd,3),1); v1.ViolinColor = colorPC;
v2= Violin(meanpvals(cda.SAs,tInd,3),2); v2.ViolinColor = colorSA;
v3= Violin(meanpvals(cda.allcells,tInd,3),3); v3.ViolinColor = colorGrey;
xticks([1 2 3])
xticklabels({'PC' 'SA' 'All'});
ylabel('p-value')
end

%% 11 Bootstrap values
data = cda;
bsdata = cdaPoiss_extrareps;

for cellInd = 1:141
    for tInd = 1:59
        datapoints = data.repSpikeDist(:,:,cellInd,tInd);
        datapoints = datapoints(~isnan(datapoints));
        distribution = bsdata.repSpikeDist(:,:,cellInd,tInd);
        distribution = distribution(~isnan(distribution));
        pvals{cellInd,tInd} = bootstrap_pval(datapoints(:),distribution(:));
    end
end


%% MOVE ABOVE: Classification
distmat = cstClassification(cdaJit.spikes,'SpikeDist',2,3,minT,maxT);


%% MOVE ABOVE: Classification Plot (not spike distance, but resolution values)
for jitInd = 1:9
results = GetResults(nanmean(distmat(:,:,:,:,:,:,jitInd),4));
subplot(3,3,jitInd)
errorshadeplot(qvals,squeeze(nanmean(results,[1 2 4]))',squeeze(nanstd(results,[],[1 2 4]))./sqrt(141));
box off;
ylim([0 1])
xlabel('resolution')
ylabel('single cell classifiation')
title(jitters(jitInd))
end

%% MOVE ABOVE: Classification Plot ISI
hold on;
distmat= fixdistmat(distmat);
results = GetResults(nanmean(distmat(:,:,:,:,:,:),4));
errorshadeplot(output.isibins,squeeze(nanmean(results,[1 2 4]))',squeeze(nanstd(results,[],[1 2 4]))./sqrt(141));
box off;
ylim([0 1])
xlabel('resolution')
ylabel('single cell classifiation')
title('ISI')

%% MOVE ABOVE: Classification Plot (Violin, rate)
results = GetResults(nanmean(distmat,4));
figure;
Violin(squeeze(nanmean(results,[1 2 4])),2);

errorshadeplot(qvals,squeeze(nanmean(results,[1 2 4]))',squeeze(nanstd(results,[],[1 2 4]))./sqrt(141)');
box off;
ylim([0 1])
xlabel('resolution')
ylabel('single cell classifiation')


%% MOVE ABOVE: PSTH plot
for tInd = sortIDs(49:50)
cellInds = cda.allcells;
titleHere = ['realdata all, texture ' cda.textures(tInd)];

figure;
for resInd = 1:9
    subplot(9,1,resInd)
    thispsth = squeeze(nanmean(cell2mat(output.psth(resInd,cellInds,tInd,:)),[2 4]));
    plot(linspace(minT,maxT,length(thispsth)),thispsth,'linewidth',2);
    axis off;
end
mastertitle(titleHere)
end

%% MOVE ABOVE: PSTH Classification Plot
data = cda;
% load('PSTH_XCov_cda.mat')
size(distmat)
% load('output_psth.mat')

for repInd = 1:5
        distmat(:,:,repInd,repInd,:,:,:,:,:) = nan;
end

%---
% All cells
gaussWidth = makecolumn(output.psthGaussWidths);

results = GetResults_cov(nanmean(distmat,3));
results = squeeze(nanmean(results,[1 2]));
meanresults = squeeze(nanmean(results,2));
stdresults = squeeze(nanstd(results,[],2));
figure;
errorshadeplot(gaussWidth,meanresults,stdresults,'k');
hold on;

meanresults = squeeze(nanmean(results(:,data.PCs),2));
stdresults = squeeze(nanstd(results,[],2));
errorshadeplot(gaussWidth,meanresults,stdresults,colorPC);

meanresults = squeeze(nanmean(results(:,data.SAs),2));
stdresults = squeeze(nanstd(results,[],2));
errorshadeplot(gaussWidth,meanresults,stdresults,colorSA);

xticks(gaussWidth)
xticklabels(gaussWidth*1000)
xlabel('resolution (ms)')
ylabel('single cell classification')














%% Cell metrics
figure;
for jitInd = 1
load('repSpikeDist_cortical_JITTERED.mat');
spdist_jit = squeeze(repSpikeDist(:,:,:,:,jitInd,:));

load('repSpikeDist_cortical.mat');

cellSpikeDist = squeeze(nanmean(repSpikeDist,[1 2 4]));
jitSpikeDist = squeeze(nanmean(spdist_jit,[1 2 4]));

adjcellSpikeDist = (cellSpikeDist+eps)./(cellSpikeDist+jitSpikeDist+eps);

subplot(3,3,jitInd); hold on;
for qInd=  1:7
    Violin(adjcellSpikeDist(:,qInd),qInd); hold on;
end

ylim([0 1])
xlabel('q value of spike distance')
ylabel('r2r spd / (r2r spd + jittered spd)')
xticks([1:7])
xticklabels(qvals)

title(['jittered: ' num2str(jitters(jitInd))])
end
%% Cell metrics: poisson
load('repSpikeDist_cortical_POISSON.mat');
spdist_jit = squeeze(repSpikeDist(:,:,:,:,:));

load('repSpikeDist_cortical.mat');

cellSpikeDist = squeeze(nanmean(repSpikeDist,[1 2 4]));
jitSpikeDist = squeeze(nanmean(spdist_jit,[1 2 4]));

adjcellSpikeDist = (cellSpikeDist+eps-jitSpikeDist)./(cellSpikeDist+jitSpikeDist+eps);

figure; hold on;
for qInd=  1:7
    Violin(adjcellSpikeDist(:,qInd),qInd); hold on;
end

ylim([-1 1])
xlabel('q value of spike distance')
ylabel('(r2r spd - poisson spd) / (r2r spd + poisson spd)')
xticks([1:7])
xticklabels(qvals)

title(['poisson model'])

%% how do spike distances change across q values?
figure;
for jitInd = 1:9
%     load('repSpikeDist_cortical_JITTERED.mat');
    load('repSpikeDist_cortical_TVP.mat');

    spdist_jit = squeeze(repSpikeDist(:,:,:,:,jitInd,:));

load('repSpikeDist_cortical.mat');

cellSpikeDist = squeeze(nanmean(repSpikeDist,[1 2 4]));
jitSpikeDist = squeeze(nanmean(spdist_jit,[1 2 4]));

adjcellSpikeDist = (cellSpikeDist+eps)./(cellSpikeDist+jitSpikeDist+eps);

subplot(3,3,jitInd); hold on;
for qInd=  1:7
    vReal = Violin(cellSpikeDist(:,qInd),qInd); hold on;
    vJit = Violin(jitSpikeDist(:,qInd),qInd+.5); 
    vReal.ViolinColor = 'k';
    vJit.ViolinColor = colorGrey;
end

ylim([0 150])
xlabel('q value of spike distance')
ylabel('spikeDist')
xticks([1:7])
xticklabels(qvals)

title(['jittered: ' num2str(jitters(jitInd))])
end

figure;
for jitInd = 1:9
    load('repSpikeDist_cortical_JITTERED.mat');
    spdist_jit = squeeze(repSpikeDist(:,:,:,:,jitInd,:));

load('repSpikeDist_cortical.mat');

cellSpikeDist = squeeze(nanmean(repSpikeDist,[1 2 4]));
jitSpikeDist = squeeze(nanmean(spdist_jit,[1 2 4]));

adjcellSpikeDist = (cellSpikeDist+eps)./(cellSpikeDist+jitSpikeDist+eps);

subplot(3,3,jitInd); hold on;
errorshadeplot_nonlog(1:length(qvals),nanmean(cellSpikeDist,1),nanstd(cellSpikeDist,1),colorTiming)
errorshadeplot_nonlog(1:length(qvals),nanmean(jitSpikeDist,1),nanstd(jitSpikeDist,1),'k')

ylim([0 50])
xlabel('q value of spike distance')
ylabel('spikeDist')
xticks([1:7])
xticklabels(qvals)

title(['jittered: ' num2str(jitters(jitInd))])
end


%% how do spike distances change across q values? POISSON
figure;

    load('repSpikeDist_cortical_POISSON.mat');
    spdist_jit = squeeze(repSpikeDist(:,:,:,:,:));

load('repSpikeDist_cortical.mat');
repSpikeDist = repSpikeDist(:,:,:,:,:);
cellSpikeDist = squeeze(nanmean(repSpikeDist,[1 2 4]));
jitSpikeDist = squeeze(nanmean(spdist_jit,[1 2 4]));

adjcellSpikeDist = (cellSpikeDist+eps)./(cellSpikeDist+jitSpikeDist+eps);

figure; hold on;
for qInd=  1:7
    vReal = Violin(cellSpikeDist(:,qInd),qInd); hold on;
    vJit = Violin(jitSpikeDist(:,qInd),qInd+.5); 
    vReal.ViolinColor = 'k';
    vJit.ViolinColor = colorGrey;
end

ylim([0 40])
xlabel('q value of spike distance')
ylabel('spikeDist')
xticks([1:7])
xticklabels(qvals)

title(['poisson'])

adjcellSpikeDist = (cellSpikeDist+eps)./(cellSpikeDist+jitSpikeDist+eps);

figure; hold on;
errorshadeplot_nonlog(1:length(qvals),nanmean(cellSpikeDist,1),nanstd(cellSpikeDist,1),colorTiming)
errorshadeplot_nonlog(1:length(qvals),nanmean(jitSpikeDist,1),nanstd(jitSpikeDist,1),'k')

ylim([0 100])
xlabel('q value of spike distance')
ylabel('spikeDist')
xticks([1:7])
xticklabels(qvals)

title(['poisson'])



%% eh delete
[minVals,cellInds] = sort(cellSpikeDist,1,'ascend');
figure;
for tInd = 1:ntextures
    subplot(8,8,tInd); hold on;
    for repInd = 1:nreps
        raster_singletrial(data(cellInd(1),tInd,repInd),repInd-1,repInd-.1,'k')
        ylim([1 5])
        axis off
    end
end

%%
figure;
thisInd = 1;
 for jitInd = [4,5,6,8,9]
     thisInd = thisInd+1;
     subplot(7,1,thisInd)
     raster(newcdaJit.spikes(86,35,1:5,jitInd),'k'); xlim([minT maxT]); ylim([0 5]);
     axis off
 end
 subplot(7,1,1)
 raster(cda.spikes(86,35,1:5),'k'); xlim([minT maxT]); ylim([0 5]);
 axis off
 subplot(7,1,7)
 raster(cdaPoiss_extrareps.spikes(86,35,1:5),'k'); xlim([minT maxT]); ylim([0 5]);
 axis off;
     
     
%% cortical figure
textureSpikeDist = squeeze(nanmean(repSpikeDist,[1 2 3]));
cellSpikeDist = squeeze(nanmean(repSpikeDist,[1 2 4]));

figure();
Violin(textureSpikeDist,1);
hold on;
Violin(cellSpikeDist,2);
xticks([1 2]);
xticklabels({'across textures' 'across cells'})
ylabel('avg spike distance')
title('spike distance across reps')

figure();
title('Spike Distance by Submodality')
PCSpikeDist = squeeze(nanmean(repSpikeDist(:,:,PClikes,:),[1 2 4]));
SASpikeDist = squeeze(nanmean(repSpikeDist(:,:,SAlikes,:),[1 2 4]));
vpc = Violin(PCSpikeDist,1);
vpc.ViolinColor = colorPC;
vsa = Violin(SASpikeDist,2);
vsa.ViolinColor = colorSA;
xticks([1 2])
xticklabels({'PC like' 'SA like'})

figure(); hold on;
title('firing rate x spike Dist')
cellrate = nanmean(rates,[2 3]);
p1 = plot(cellrate,cellSpikeDist,'o','color',colorGrey);
p1.MarkerFaceColor = colorGrey;
p2 = plot(cellrate(PClikes),cellSpikeDist(PClikes),'o','color',colorPC);
p2.MarkerFaceColor = colorPC;
p3 = plot(cellrate(SAlikes),cellSpikeDist(SAlikes),'o','color',colorSA);
p3.MarkerFaceColor = colorSA;
box off;
xlabel('cell avg firing rate (Hz)')
ylabel('cell avg spike distance')

figure(); hold on;
title('roughness x spikeDist, across cells')
p1 = plot(cdaRough(~isnan(cdaRough)),textureSpikeDist(~isnan(cdaRough)),'o','color',colorGrey);
p1.MarkerFaceColor = colorGrey;
xlabel('roughness')
ylabel('spikeDist')

figure(); hold on;
PCtextureSpikeDist = squeeze(nanmean(repSpikeDist(:,:,PClikes,~isnan(roughness)),[1 2 3]));
SAtextureSpikeDist = squeeze(nanmean(repSpikeDist(:,:,SAlikes,~isnan(roughness)),[1 2 3]));
title('roughness x spikeDist, across cells')
% p1 = plot(perRough,textureSpikeDist,'o','color',colorGrey);
% p1.MarkerFaceColor = colorGrey;
p2 = plot(roughness(~isnan(roughness)),PCtextureSpikeDist,'o','color',colorPC);
p2.MarkerFaceColor = colorPC;
p3 = plot(roughness(~isnan(roughness)),SAtextureSpikeDist,'o','color',colorSA);
p3.MarkerFaceColor = colorSA;
xlabel('roughness')
ylabel('spikeDist')

%% peripheral
textureSpikeDist = squeeze(nanmean(repSpikeDist,[1 2 3]));
cellSpikeDist = squeeze(nanmean(repSpikeDist,[1 2 4]));

figure();
Violin(textureSpikeDist,1);
hold on;
Violin(cellSpikeDist,2);
xticks([1 2]);
xticklabels({'across textures' 'across cells'})
ylabel('avg spike distance')
title('spike distance across reps')

figure();
title('Spike Distance by Submodality')
PCSpikeDist = squeeze(nanmean(repSpikeDist(:,:,PCs,:),[1 2 4]));
SASpikeDist = squeeze(nanmean(repSpikeDist(:,:,SAs,:),[1 2 4]));
vpc = Violin(PCSpikeDist,1);
vpc.ViolinColor = colorPC;
vsa = Violin(SASpikeDist,2);
vsa.ViolinColor = colorSA;
xticks([1 2])
xticklabels({'PCs' 'SAs'})

figure(); hold on;
title('firing rate x spike Dist')
cellrate = nanmean(rates,[2 3]);
p1 = plot(cellrate,cellSpikeDist,'o','color',colorGrey);
p1.MarkerFaceColor = colorGrey;
p2 = plot(cellrate(PCs),cellSpikeDist(PCs),'o','color',colorPC);
p2.MarkerFaceColor = colorPC;
p3 = plot(cellrate(SAs),cellSpikeDist(SAs),'o','color',colorSA);
p3.MarkerFaceColor = colorSA;
box off;
xlabel('cell avg firing rate (Hz)')
ylabel('cell avg spike distance')

figure(); hold on;
title('roughness x spikeDist, across cells')
p1 = plot(perRough,textureSpikeDist,'o','color',colorGrey);
p1.MarkerFaceColor = colorGrey;
xlabel('roughness')
ylabel('spikeDist')

figure(); hold on;
PCtextureSpikeDist = squeeze(nanmean(repSpikeDist(:,:,PCs,:),[1 2 3]));
SAtextureSpikeDist = squeeze(nanmean(repSpikeDist(:,:,SAs,:),[1 2 3]));
title('roughness x spikeDist, across cells')
% p1 = plot(perRough,textureSpikeDist,'o','color',colorGrey);
% p1.MarkerFaceColor = colorGrey;
p2 = plot(perRough,PCtextureSpikeDist,'o','color',colorPC);
p2.MarkerFaceColor = colorPC;
p3 = plot(perRough,SAtextureSpikeDist,'o','color',colorSA);
p3.MarkerFaceColor = colorSA;
xlabel('roughness')
ylabel('spikeDist')

%% cortical jittered
textureSpikeDist = squeeze(nanmean(repSpikeDist,[1 2 3]));
cellSpikeDist = squeeze(nanmean(repSpikeDist,[1 2 4]));
njits = size(repSpikeDist,5);
nqvals = size(repSpikeDist,6);

figure();
for jitInd = 1:njits
hold on;
Violin(cellSpikeDist(:,jitInd),jitInd);
end
xticks(1:njits);
xticklabels(jitters)
ylabel('avg spike distance')
title('spike distance across reps')
xlabel('jittered amount')

figure();
title('Spike Distance by Submodality')
for jitInd = 1:njits
PCSpikeDist = squeeze(nanmean(repSpikeDist(:,:,PClikes,:,jitInd),[1 2 4]));
SASpikeDist = squeeze(nanmean(repSpikeDist(:,:,SAlikes,:,jitInd),[1 2 4]));
vpc = Violin(PCSpikeDist,jitInd);
vpc.ViolinColor = colorPC;
vsa = Violin(SASpikeDist,jitInd + .5);
vsa.ViolinColor = colorSA;
end
xticks(1:njits+1);
xticklabels([jitters {'real'}]);

figure(); 
title('firing rate x spike Dist')
cellrate = nanmean(rates,[2 3]);
for jitInd = 1:njits
    subplot(3,3,jitInd); hold on;
    title(jitters(jitInd))
p1 = plot(cellrate,cellSpikeDist(:,jitInd),'o','color',colorGrey);
p1.MarkerFaceColor = colorGrey;
p2 = plot(cellrate(PClikes),cellSpikeDist(PClikes,jitInd),'o','color',colorPC);
p2.MarkerFaceColor = colorPC;
p3 = plot(cellrate(SAlikes),cellSpikeDist(SAlikes,jitInd),'o','color',colorSA);
p3.MarkerFaceColor = colorSA;
box off;
xlabel('cell avg firing rate (Hz)')
ylabel('cell avg spike distance')
end

figure(); hold on;
title('roughness x spikeDist, across cells')
p1 = plot(cdaRough(~isnan(cdaRough)),textureSpikeDist(~isnan(cdaRough)),'o','color',colorGrey);
p1.MarkerFaceColor = colorGrey;
xlabel('roughness')
ylabel('spikeDist')

figure(); hold on;
PCtextureSpikeDist = squeeze(nanmean(repSpikeDist(:,:,PClikes,~isnan(roughness)),[1 2 3]));
SAtextureSpikeDist = squeeze(nanmean(repSpikeDist(:,:,SAlikes,~isnan(roughness)),[1 2 3]));
title('roughness x spikeDist, across cells')
% p1 = plot(perRough,textureSpikeDist,'o','color',colorGrey);
% p1.MarkerFaceColor = colorGrey;
p2 = plot(roughness(~isnan(roughness)),PCtextureSpikeDist,'o','color',colorPC);
p2.MarkerFaceColor = colorPC;
p3 = plot(roughness(~isnan(roughness)),SAtextureSpikeDist,'o','color',colorSA);
p3.MarkerFaceColor = colorSA;
xlabel('roughness')
ylabel('spikeDist')

%% peripheral jittered
textureSpikeDist = squeeze(nanmean(repSpikeDist,[1 2 3]));
cellSpikeDist = squeeze(nanmean(repSpikeDist,[1 2 4]));
njits = size(repSpikeDist,5);

figure();
for jitInd = 1:njits
hold on;
Violin(cellSpikeDist(:,jitInd),jitInd);
end
xticks(1:njits);
xticklabels(jitters)
ylabel('avg spike distance')
title('spike distance across reps')
xlabel('jittered amount')

figure();
title('Spike Distance by Submodality')
for jitInd = 1:njits
PCSpikeDist = squeeze(nanmean(repSpikeDist(:,:,PCs,:,jitInd),[1 2 4]));
SASpikeDist = squeeze(nanmean(repSpikeDist(:,:,SAs,:,jitInd),[1 2 4]));
vpc = Violin(PCSpikeDist,jitInd);
vpc.ViolinColor = colorPC;
vsa = Violin(SASpikeDist,jitInd + .5);
vsa.ViolinColor = colorSA;
end
xticks(1:njits+1);
xticklabels([jitters {'real'}]);

figure(); 
title('firing rate x spike Dist')
cellrate = nanmean(rates,[2 3]);
for jitInd = 1:njits
    subplot(3,3,jitInd); hold on;
    title(jitters(jitInd))
p1 = plot(cellrate,cellSpikeDist(:,jitInd),'o','color',colorGrey);
p1.MarkerFaceColor = colorGrey;
p2 = plot(cellrate(PCs),cellSpikeDist(PCs,jitInd),'o','color',colorPC);
p2.MarkerFaceColor = colorPC;
p3 = plot(cellrate(SAs),cellSpikeDist(SAs,jitInd),'o','color',colorSA);
p3.MarkerFaceColor = colorSA;
box off;
xlabel('cell avg firing rate (Hz)')
ylabel('cell avg spike distance')
end

figure(); hold on;
title('roughness x spikeDist, across cells')
p1 = plot(cdaRough(~isnan(cdaRough)),textureSpikeDist(~isnan(cdaRough)),'o','color',colorGrey);
p1.MarkerFaceColor = colorGrey;
xlabel('roughness')
ylabel('spikeDist')

figure(); hold on;
PCtextureSpikeDist = squeeze(nanmean(repSpikeDist(:,:,PClikes,~isnan(roughness)),[1 2 3]));
SAtextureSpikeDist = squeeze(nanmean(repSpikeDist(:,:,SAlikes,~isnan(roughness)),[1 2 3]));
title('roughness x spikeDist, across cells')
% p1 = plot(perRough,textureSpikeDist,'o','color',colorGrey);
% p1.MarkerFaceColor = colorGrey;
p2 = plot(roughness(~isnan(roughness)),PCtextureSpikeDist,'o','color',colorPC);
p2.MarkerFaceColor = colorPC;
p3 = plot(roughness(~isnan(roughness)),SAtextureSpikeDist,'o','color',colorSA);
p3.MarkerFaceColor = colorSA;
xlabel('roughness')
ylabel('spikeDist')




%% JITTER ONLY: classification
    distmat = cstClassification(output.psth,'EucDist',3,4,minT,maxT);
%%
PCresults = squeeze(nanmean(results(:,:,cda.PCs,:),[1 2]));
PCerror   = nanstd(PCresults,[],1)./(sqrt(length(PCerror)));
SAresults = squeeze(nanmean(results(:,:,cda.SAs,:),[1 2]));
semilogx(1000*qvals,PCresults,'linewidth',2,'color',colorPC);
semilogx(1000*qvals,SAresults,'linewidth',2,'color',colorSA);

%% Crazy classification scheme
distmatTime = [];

%% Predicting Perception

load('Rate_Euc_cda.mat');
ratedistmat= fixdistmat(distmat);

load('PSTH_XCov_cda_expanded.mat');
% load('spikeDist_cda_ALLQs.mat')
PSTHdistmat=  fixdistmat(distmat);
[cellMax,cellRes]  = max(cda.PSTHresults,[],2);


load('dissimData.mat');
perceptualDissim = dissimData.mat;

ratedistmat = ratedistmat(dissimData.textInd,dissimData.textInd,:,:,:);
PSTHdistmat = PSTHdistmat(dissimData.textInd,dissimData.textInd,:,:,:,:);
for cellInd = 1:141
    PSTHdistmat_best(:,:,:,:,cellInd) = PSTHdistmat(:,:,:,:,cellRes(cellInd),cellInd);
end

for tInd = 1:13
    ratedistmat(tInd,tInd,:,:,:) = nan;
    PSTHdistmat_best(tInd,tInd,:,:,:,:) = nan;
    perceptualDissim(tInd,tInd) = nan;
end

meanPSTHdistmat = reshape(nanmean(PSTHdistmat_best(:,:,:,:,cda.PCs),[3 4 5]),13*13,1);
meanPerceptualdistmat = reshape(perceptualDissim,13*13,1);
meanratedistmat = reshape(nanmean(ratedistmat(:,:,:,:,cda.PCs),[3 4 5]),13*13,1);


figure;
subplot(2,1,1)
plot(meanratedistmat,meanPerceptualdistmat,'o','color','r')
xlabel('rate distance')
ylabel('percpetual distance')

box off;
subplot(2,1,2)
plot(meanPSTHdistmat,meanPerceptualdistmat,'o','color','b')
xlabel('timing distance')
ylabel('percpetual distance')
box off;

% BEST FIT
figure;
x = meanratedistmat(~isnan(meanratedistmat));
y = meanPerceptualdistmat(~isnan(meanPerceptualdistmat));
% Get coefficients of a line fit through the data.
coefficients = polyfit(x, y, 1);
% Create a new x axis with exactly 1000 points (or whatever you want).
xFit = linspace(min(x), max(x), 1000);
% Get the estimated yFit value for each of those 1000 new x locations.
yFit = polyval(coefficients , xFit);
% Plot everything.
plot(x, y, 'b.', 'MarkerSize', 15); % Plot training data.
hold on; % Set hold on so the next plot does not blow away the one we just drew.
plot(xFit, yFit, 'r-', 'LineWidth', 2); % Plot fitted line.
box off;
[b] = regress(y,x)

rateresiduals = abs(y - polyval(coefficients, x));

figure;
x = meanPSTHdistmat(~isnan(meanPSTHdistmat));
y = meanPerceptualdistmat(~isnan(meanPerceptualdistmat));
% Get coefficients of a line fit through the data.
coefficients = polyfit(x, y, 1);
% Create a new x axis with exactly 1000 points (or whatever you want).
xFit = linspace(min(x), max(x), 1000);
% Get the estimated yFit value for each of those 1000 new x locations.
yFit = polyval(coefficients , xFit);
% Plot everything.
plot(x, y, 'b.', 'MarkerSize', 15); % Plot training data.
hold on; % Set hold on so the next plot does not blow away the one we just drew.
plot(xFit, yFit, 'r-', 'LineWidth', 2); % Plot fitted line.
box off;
[b] = regress(y,x)

figure;
plot(rateresiduals,x,'o')

%% PERCEPTUAL DATA
chosencells= 1:141;
ncells = length(chosencells);

load('Rate_Euc_cda.mat');
ratedistmat= fixdistmat(distmat);

load('PSTH_XCov_cda_expanded.mat');
% load('spikeDist_cda_ALLQs.mat')
PSTHdistmat=  fixdistmat(distmat);
[cellMax,cellRes]  = max(cda.PSTHresults,[],2);


load('dissimData.mat');
perceptualDissim = dissimData.mat;

ratedistmat = ratedistmat(dissimData.textInd,dissimData.textInd,:,:,:);
PSTHdistmat = PSTHdistmat(dissimData.textInd,dissimData.textInd,:,:,:,:);
for cellInd = 1:141
    PSTHdistmat_best(:,:,:,:,cellInd) = PSTHdistmat(:,:,:,:,cellRes(cellInd),cellInd);
end

for tInd = 1:13
    ratedistmat(tInd,tInd,:,:,:) = nan;
    PSTHdistmat_best(tInd,tInd,:,:,:,:) = nan;
    perceptualDissim(tInd,tInd) = nan;
end

meanPSTHdistmat = reshape(nanmean(PSTHdistmat_best(:,:,:,:,chosencells),[3 4]),13*13,ncells);
meanPerceptualdistmat = reshape(perceptualDissim,13*13,1);
meanratedistmat = reshape(nanmean(ratedistmat(:,:,:,:,chosencells),[3 4]),13*13,ncells);

meanPSTHdistmat = meanPSTHdistmat(~isnan(meanPSTHdistmat(:,1)),:);
meanPerceptualdistmat = meanPerceptualdistmat(~isnan(meanPerceptualdistmat));
meanratedistmat = meanratedistmat(~isnan(meanratedistmat(:,1)),:);

%%
figure;
clear X; clear stats;
X(:,:,1) = ones(size(meanPSTHdistmat));
X(:,:,2) = meanratedistmat;
X(:,:,3) = meanPSTHdistmat;
X = permute(X,[1 3 2]);

for cellInd = 1:141
    [coeffs(cellInd,:),~,~,~,stats(cellInd,:)] = regress(meanPerceptualdistmat,X(:,:,cellInd));
end

subplot(1,3,1)
v1 = Violin(stats(:,1),1);
v1.ViolinColor = 'k';
v2 = Violin(stats(cda.PCs,1),2);
v2.ViolinColor = colorPC;
v3 = Violin(stats(cda.SAs,1),3);
v3.ViolinColor = colorSA;
title('All')
ylim([0 1])
ylabel('r2 value, single cell')
xticks([1 2 3])
xticklabels({'all', 'PC', 'SA'})

clear X; clear stats;
X(:,:,1) = ones(size(meanPSTHdistmat));
X(:,:,2) = meanratedistmat;
% X(:,:,3) = meanPSTHdistmat;
X = permute(X,[1 3 2]);

for cellInd = 1:141
    [~,~,rate_r(cellInd,:),~,stats(cellInd,:)] = regress(meanPerceptualdistmat,X(:,:,cellInd));
end

subplot(1,3,2)
v1 = Violin(stats(:,1),1);
v1.ViolinColor = colorTiming;
v2 = Violin(stats(cda.PCs,1),2);
v2.ViolinColor = colorPC;
v3 = Violin(stats(cda.SAs,1),3);
v3.ViolinColor = colorSA;
title('Rate')
ylim([0 1])
xticks([1 2 3])
xticklabels({'all', 'PC', 'SA'})

clear X; clear stats;
X(:,:,1) = ones(size(meanPSTHdistmat));
X(:,:,2) = meanPSTHdistmat;

X = permute(X,[1 3 2]);

for cellInd = 1:141
    [~,~,timing_r(cellInd,:),~,stats(cellInd,:)] = regress(meanPerceptualdistmat,X(:,:,cellInd));
end

subplot(1,3,3)
v1 = Violin(stats(:,1),1);
v1.ViolinColor = colorRate;
v2 = Violin(stats(cda.PCs,1),2);
v2.ViolinColor = colorPC;
v3 = Violin(stats(cda.SAs,1),3);
v3.ViolinColor = colorSA;
title('Timing')
ylim([0 1])
xticks([1 2 3])
xticklabels({'all', 'PC', 'SA'})

%
figure;
hold on;
plot(squeeze(nanmean(abs(rate_r(cda.PCs,:)),1)),squeeze(nanmean(abs(timing_r(cda.PCs,:)),1)),'o')
plot([0 1], [0 1],'--')
ylabel('timing residual')
xlabel('rate residual')

%% quad model population
% PC, SA, rate, timing
% cross validated, predict left out texture pair from all others
chosencells= cda.PCs;

for hide = 1
ncells = length(chosencells);

load('Rate_Euc_cda.mat');
ratedistmat= fixdistmat(distmat);

load('PSTH_XCov_cda_expanded.mat');
% load('spikeDist_cda_ALLQs.mat')
PSTHdistmat=  fixdistmat(distmat);
[cellMax,cellRes]  = max(cda.PSTHresults,[],2);


load('dissimData.mat');
perceptualDissim = dissimData.mat;

ratedistmat = ratedistmat(dissimData.textInd,dissimData.textInd,:,:,:);
PSTHdistmat = PSTHdistmat(dissimData.textInd,dissimData.textInd,:,:,:,:);
for cellInd = 1:141
    PSTHdistmat_best(:,:,:,:,cellInd) = PSTHdistmat(:,:,:,:,cellRes(cellInd),cellInd);
end

for tInd = 1:13
    ratedistmat(tInd,tInd,:,:,:) = nan;
    PSTHdistmat_best(tInd,tInd,:,:,:,:) = nan;
    perceptualDissim(tInd,tInd) = nan;
end

meanPSTHdistmat = reshape(nanmean(PSTHdistmat_best(:,:,:,:,chosencells),[3 4]),13*13,ncells);
meanPerceptualdistmat = reshape(perceptualDissim,13*13,1);
meanratedistmat = reshape(nanmean(ratedistmat(:,:,:,:,chosencells),[3 4]),13*13,ncells);

meanPSTHdistmat = meanPSTHdistmat(~isnan(meanPSTHdistmat(:,1)),:);
meanPerceptualdistmat = meanPerceptualdistmat(~isnan(meanPerceptualdistmat));
meanratedistmat = meanratedistmat(~isnan(meanratedistmat(:,1)),:);
end

PC_timing = meanPSTHdistmat;
PC_rate = meanratedistmat;
y = meanPerceptualdistmat;

chosencells= cda.SAs;

for hide = 1
ncells = length(chosencells);

load('Rate_Euc_cda.mat');
ratedistmat= fixdistmat(distmat);

load('PSTH_XCov_cda_expanded.mat');
% load('spikeDist_cda_ALLQs.mat')
PSTHdistmat=  fixdistmat(distmat);
[cellMax,cellRes]  = max(cda.PSTHresults,[],2);


load('dissimData.mat');
perceptualDissim = dissimData.mat;

ratedistmat = ratedistmat(dissimData.textInd,dissimData.textInd,:,:,:);
PSTHdistmat = PSTHdistmat(dissimData.textInd,dissimData.textInd,:,:,:,:);
for cellInd = 1:141
    PSTHdistmat_best(:,:,:,:,cellInd) = PSTHdistmat(:,:,:,:,cellRes(cellInd),cellInd);
end

for tInd = 1:13
    ratedistmat(tInd,tInd,:,:,:) = nan;
    PSTHdistmat_best(tInd,tInd,:,:,:,:) = nan;
    perceptualDissim(tInd,tInd) = nan;
end

meanPSTHdistmat = reshape(nanmean(PSTHdistmat_best(:,:,:,:,chosencells),[3 4]),13*13,ncells);
meanPerceptualdistmat = reshape(perceptualDissim,13*13,1);
meanratedistmat = reshape(nanmean(ratedistmat(:,:,:,:,chosencells),[3 4]),13*13,ncells);

meanPSTHdistmat = meanPSTHdistmat(~isnan(meanPSTHdistmat(:,1)),:);
meanPerceptualdistmat = meanPerceptualdistmat(~isnan(meanPerceptualdistmat));
meanratedistmat = meanratedistmat(~isnan(meanratedistmat(:,1)),:);
end

SA_timing = meanPSTHdistmat;
SA_rate = meanratedistmat;

thisInd = 0;
thesecolors = parula(12);

thisInd = thisInd+1;
for itInd = 1:12
    X = squeeze(nanmean(cat(3,PC_timing(:,1:itInd),PC_rate(:,1:itInd),SA_timing(:,1:itInd),SA_rate(:,1:itInd)),2));
    X(:,5) = ones(size(X,1),1);
    [B,BINT,R,RINT,STATS(itInd,:)] = regress(y,X);
%     popSizeX(:,:,itInd) = X;
end
% figure;
plot(1:12,STATS(:,1),'color',thesecolors(thisInd,:))
hold on;

legend PCtime PCrate SAtime SArate PC SA timing rate all bestcombo

%% quad model population: MSE
% PC, SA, rate, timing
% cross validated, predict left out texture pair from all others
% 
% for hide = 1
% ncells = 141;
% 
% load('Rate_Euc_cda.mat');
% ratedistmat= fixdistmat(distmat);
% 
% load('PSTH_XCov_cda_expanded.mat');
% % load('spikeDist_cda_ALLQs.mat')
% PSTHdistmat=  fixdistmat(distmat);
% [cellMax,cellRes]  = max(cda.PSTHresults,[],2);
% 
% 
% load('dissimData.mat');
% perceptualDissim = dissimData.mat;
% 
% ratedistmat = ratedistmat(dissimData.textInd,dissimData.textInd,:,:,:);
% PSTHdistmat = PSTHdistmat(dissimData.textInd,dissimData.textInd,:,:,:,:);
% for cellInd = 1:141
%     PSTHdistmat_best(:,:,:,:,cellInd) = PSTHdistmat(:,:,:,:,cellRes(cellInd),cellInd);
% end
% 
% for tInd = 1:13
%     ratedistmat(tInd,tInd,:,:,:) = nan;
%     PSTHdistmat_best(tInd,tInd,:,:,:,:) = nan;
%     perceptualDissim(tInd,tInd) = nan;
% end
% 
% meanPSTHdistmat = reshape(nanmean(PSTHdistmat_best,[3 4]),13*13,ncells);
% meanPerceptualdistmat = reshape(perceptualDissim,13*13,1);
% meanratedistmat = reshape(nanmean(ratedistmat,[3 4]),13*13,ncells);
% 
% meanPSTHdistmat = meanPSTHdistmat(~isnan(meanPSTHdistmat(:,1)),:);
% meanPerceptualdistmat = meanPerceptualdistmat(~isnan(meanPerceptualdistmat));
% meanratedistmat = meanratedistmat(~isnan(meanratedistmat(:,1)),:);
% end

dist_timing = perception.dist_timing;
dist_rate = perception.dist_rate;
y = perception.dissim_vec;

% timingCells = find(cdaData.area==3); %56 cells
% rateCells = find(cdaData.area==1); %31 cells
% 
timingCells = cda.RAs;
rateCells = cda.SAs;

% alltimingCells = find(cellMax>.4); %50 cells
% timingCells = alltimingCells(cdaData.area(alltimingCells) == 1);
% rateCells = alltimingCells(cdaData.area(alltimingCells) == 3);

for popSize = 1:12
    for itInd = 1:500
    thesePCs = datasample(timingCells,popSize);
    theseSAs = datasample(rateCells,popSize);
    X = squeeze(nanmean(cat(3,dist_timing(:,thesePCs),dist_rate(:,thesePCs),dist_timing(:,theseSAs),dist_rate(:,theseSAs)),2));
    X(:,5) = ones(size(X,1),1);
    for loInd = 1:156
        allbut = ones(156,1);
        allbut(loInd) = 0;
        allbut = find(allbut);
        [B,BINT,R,RINT,STATS(itInd,:)] = regress(y(allbut),X(allbut,:));
        actual = y(loInd);
        predicted = B(1)*X(loInd,1) + B(2) * X(loInd,2) + B(3) * X(loInd,3) + B(4) * X(loInd,4) + B(5);
        predictionerror(popSize,loInd,itInd,1) = (actual-predicted)^2;
        allpredictions(popSize,loInd,itInd,1) = predicted;
%         other models
        B_PC = regress(y(allbut),X(allbut,[1 2 5]));
            predicted_PC = B_PC(1) * X(loInd,1) + B_PC(2) * X(loInd,2) + B_PC(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,2) = (actual-predicted_PC)^2;
            allpredictions(popSize,loInd,itInd,2) = predicted_PC;
        B_SA = regress(y(allbut),X(allbut,[3 4 5]));
            predicted_SA = B_SA(1) * X(loInd,3) + B_SA(2) * X(loInd,4) + B_SA(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,3) = (actual-predicted_SA)^2;
            allpredictions(popSize,loInd,itInd,3) = predicted_SA;
        B_t = regress(y(allbut),X(allbut,[1 3 5]));
            predicted_t = B_t(1) * X(loInd,1) + B_t(2) * X(loInd,3) + B_t(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,4) = (actual-predicted_t)^2;
            allpredictions(popSize,loInd,itInd,4) = predicted_t;
        B_r = regress(y(allbut),X(allbut,[2 4 5]));
            predicted_r = B_r(1) * X(loInd,2) + B_r(2) * X(loInd,4) + B_r(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,5) = (actual-predicted_r)^2;
            allpredictions(popSize,loInd,itInd,5) = predicted_r;
        B_optimal = regress(y(allbut),X(allbut,[1 4 5]));
            predicted_optimal = B_optimal(1) * X(loInd,1) + B_optimal(2) * X(loInd,4) + B_optimal(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,6) = (actual-predicted_optimal)^2;
            optimalpredictions(popSize,loInd,itInd) = predicted_optimal;
            allpredictions(popSize,loInd,itInd,6) = predicted_optimal;
        modelName = {'all' 'PC' 'SA' 'timing' 'rate' 'optimal'};
        
    end
    end
end

figure; hold on;
colorVals = [ 0 0 0; colorPC; colorSA; colorRate; colorTiming; colorGrey];
for errorModel = 1:6
plot([1:12],nanmean(predictionerror(:,:,:,errorModel),[2 3])','linewidth',2,'color',colorVals(errorModel,:));
end
legend(modelName)
legend boxoff
ylim([ 0 .18]);
xlabel('population size')
ylabel('MSE')

figure;
 hold on;
for errorModel = 1:6
errorshadeplot_nonlog([1:12],nanmean(predictionerror(:,:,:,errorModel),[2 3])',nanstd(predictionerror(:,:,:,errorModel),[],[2 3])',colorVals(errorModel,:));
end
legend(modelName)
legend boxoff
ylim([ 0 .18]);
xlabel('population size')
ylabel('MSE')

figure;
for errorModel = 1:6
subplot(3,2,errorModel)
predicted = squeeze(nanmean(allpredictions(12,:,:,errorModel),3))';

scatter(predicted,y,'filled','CData',colorVals(errorModel,:))
title(modelName(errorModel))
hold on;
plot([.5 2],[.5 2],'--','color','k')
xlim([0.5 2]); ylim([0.5 2])
end


%% Every submodality combination

dist_timing = perception.dist_timing;
dist_rate = perception.dist_rate;
y = perception.dissim_vec;

% timingCells = find(cdaData.area==3); %56 cells
% rateCells = find(cdaData.area==1); %31 cells
% 
RAs = cda.RAs;
SAs = cda.SAs;
PCs = cda.PCs;

% alltimingCells = find(cellMax>.4); %50 cells
% timingCells = alltimingCells(cdaData.area(alltimingCells) == 1);
% rateCells = alltimingCells(cdaData.area(alltimingCells) == 3);

for popSize = 1:12
    for itInd = 1:500
    thesePCs = datasample(PCs,popSize,'replace',false);
    theseSAs = datasample(SAs,popSize,'replace',false);
    theseRAs = datasample(RAs,popSize,'replace',false);
    X = squeeze(nanmean(cat(3,dist_timing(:,thesePCs),dist_rate(:,thesePCs),dist_timing(:,theseSAs),dist_rate(:,theseSAs),dist_timing(:,theseRAs),dist_rate(:,theseRAs)),2));
    X(:,7) = ones(size(X,1),1);
    for loInd = 1:156
        allbut = ones(156,1);
        allbut(loInd) = 0;
        allbut = find(allbut);
        [B,BINT,R,RINT,STATS(itInd,:)] = regress(y(allbut),X(allbut,:));
        actual = y(loInd);
        predicted = B(1)*X(loInd,1) + B(2) * X(loInd,2) + B(3) * X(loInd,3) + B(4) * X(loInd,4) + B(5) * X(loInd,5) + B(6) * X(loInd,6) + B(7);
        predictionerror(popSize,loInd,itInd,1) = (actual-predicted)^2;
        allpredictions(popSize,loInd,itInd,1) = predicted;
%         other models
        B_PC = regress(y(allbut),X(allbut,[1 2 7]));
            predicted_PC = B_PC(1) * X(loInd,1) + B_PC(2) * X(loInd,2) + B_PC(3) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,2) = (actual-predicted_PC)^2;
            allpredictions(popSize,loInd,itInd,2) = predicted_PC;
        B_SA = regress(y(allbut),X(allbut,[3 4 7]));
            predicted_SA = B_SA(1) * X(loInd,3) + B_SA(2) * X(loInd,4) + B_SA(3) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,3) = (actual-predicted_SA)^2;
            allpredictions(popSize,loInd,itInd,3) = predicted_SA;
        B_RA = regress(y(allbut),X(allbut,[5 6 7]));
            predicted_RA = B_RA(1) * X(loInd,5) + B_RA(2) * X(loInd,6) + B_RA(3) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,4) = (actual-predicted_RA)^2;
            allpredictions(popSize,loInd,itInd,4) = predicted_RA;
            
        
        B_t = regress(y(allbut),X(allbut,[1 3 5 7])); %
            predicted_t = B_t(1) * X(loInd,1) + B_t(2) * X(loInd,3) + B_t(3) * X(loInd,5) + B_t(4) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,5) = (actual-predicted_t)^2;
            allpredictions(popSize,loInd,itInd,5) = predicted_t;
            
        B_r = regress(y(allbut),X(allbut,[2 4 6 7])); %
            predicted_r = B_r(1) * X(loInd,2) + B_r(2) * X(loInd,4) + B_r(3) * X(loInd,6) + B_r(4) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,6) = (actual-predicted_r)^2;
            allpredictions(popSize,loInd,itInd,6) = predicted_r;
        B_optimal = regress(y(allbut),X(allbut,[1 4 7])); %
            predicted_optimal = B_optimal(1) * X(loInd,1) + B_optimal(2) * X(loInd,4) + B_optimal(3) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,7) = (actual-predicted_optimal)^2;
            optimalpredictions(popSize,loInd,itInd) = predicted_optimal;
            allpredictions(popSize,loInd,itInd,7) = predicted_optimal;
        B_PCRA = regress(y(allbut),X(allbut,[1 6 7])); %
            predicted_optimal = B_PCRA(1) * X(loInd,1) + B_PCRA(2) * X(loInd,6) + B_PCRA(3) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,8) = (actual-predicted_optimal)^2;
            allpredictions(popSize,loInd,itInd,8) = predicted_optimal;
        B_RASA = regress(y(allbut),X(allbut,[5 4 7])); %
            predicted_optimal = B_RASA(1) * X(loInd,5) + B_RASA(2) * X(loInd,4) + B_RASA(3) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,9) = (actual-predicted_optimal)^2;
            allpredictions(popSize,loInd,itInd,9) = predicted_optimal;  
            
        B_PCRASA = regress(y(allbut),X(allbut,[1 4 6 7]));
            predicted_optimal = B_PCRASA(1) * X(loInd,1) + B_PCRASA(2) * X(loInd,4) + B_PCRASA(3) * X(loInd,6) + B_PCRASA(4) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,10) = (actual-predicted_optimal)^2;
            allpredictions(popSize,loInd,itInd,10) = predicted_optimal;
            
        B_PCandrate = regress(y(allbut),X(allbut,[1 2 4 6 7]));
            predicted_optimal = B_PCandrate(1) * X(loInd,1) + B_PCandrate(2) * X(loInd,2) + B_PCandrate(3) * X(loInd,4) + B_PCandrate(4) * X(loInd,6) + B_PCandrate(5) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,11) = (actual-predicted_optimal)^2;
            allpredictions(popSize,loInd,itInd,11) = predicted_optimal;
            
            
        B_12 = regress(y(allbut),X(allbut,[1 2 3 4 6 7]));
            predicted_optimal = B_12(1) * X(loInd,1) + B_12(2) * X(loInd,2) + B_12(3) * X(loInd,3) + B_12(4) * X(loInd,4) + B_12(5) * X(loInd,6) + B_12(6) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,12) = (actual-predicted_optimal)^2;
            allpredictions(popSize,loInd,itInd,12) = predicted_optimal;
            
         B_13 = regress(y(allbut),X(allbut,[1 2 4 5 6 7]));
            predicted_optimal = B_13(1) * X(loInd,1) + B_13(2) * X(loInd,2) + B_13(3) * X(loInd,4) + B_13(4) * X(loInd,5) + B_13(5) * X(loInd,6) + B_13(6) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,13) = (actual-predicted_optimal)^2;
            allpredictions(popSize,loInd,itInd,13) = predicted_optimal;
            
                 B_14 = regress(y(allbut),X(allbut,[ 2 3 4 5 6 7]));
            predicted_optimal = B_14(1) * X(loInd,2) + B_14(2) * X(loInd,3) + B_14(3) * X(loInd,4) + B_14(4) * X(loInd,5) + B_14(5) * X(loInd,6) + B_14(6) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,14) = (actual-predicted_optimal)^2;
            allpredictions(popSize,loInd,itInd,14) = predicted_optimal;    
            
        modelName = {'all' 'PC' 'SA' 'RA' 'timing_all' 'rate_all' 'PCSA' 'PCRA' 'RASA' 'PCRASA' 'B_PCandrate' 'allbutRAtiming' 'allbutSAtiming' 'allbutPCtiming'};
        
    end
    end
end

figure; hold on;
colorVals = [ 0 0 0; colorPC; colorSA; colorRA; colorRate; colorTiming; colorGrey; colorGrey*.6; colorGrey*.4; colorGrey*.2; colorPC*.8; colorSA*.8; colorRA*.8; colorPC*.6];
for errorModel = 1:length(modelName)
plot([1:12],nanmean(predictionerror(:,:,:,errorModel),[2 3])','linewidth',2,'color',colorVals(errorModel,:));
end
legend(modelName)
legend boxoff
ylim([ 0 .18]);
xlabel('population size')
ylabel('MSE')

figure; thisInd = 0;
for errorModel = [1 5 6 4 2 3 7 8 9 10 11 12 13 14]
    thisInd = thisInd + 1;
subplot(5,3,thisInd)
predicted = squeeze(nanmean(allpredictions(12,:,:,errorModel),3))';

scatter(predicted,y,80,'filled','CData',colorVals(errorModel,:))
title(modelName(errorModel))
hold on;
plot([0 2.5],[0 2.5],'--','color','k')
xlim([0 2.5]); ylim([0 2.5])
end

groupMSE = squeeze(nanmean(predictionerror(10,:,1:100,:),2));
[p,t,stats] = kruskalwallis(groupMSE);
multcompare(stats)

figure;
[errormeans,meanInd]=sort(squeeze(nanmean(predictionerror(10,:,1:100,:),[2 3])),'descend');
thisInd = 0;
for errorModel = 1:length(meanInd)
    errorModelInd = meanInd(errorModel);
    lieberplot_ind_std(squeeze(nanmean(predictionerror(10,:,1:100,errorModelInd),2)),errorModel);
    [p,h,stats] = ranksum(groupMSE(1:10,1),groupMSE(1:10,errorModelInd),'tail','left');
    text(errorModel,.17,num2str(round(p*100,2)))
end
xticklabels(modelName(meanInd))


%% Every submodality combination: z-scored

dist_timing = perception.dist_timing;
dist_rate == perception.dist_rate;
y = perception.dissim_vec;

% timingCells = find(cdaData.area==3); %56 cells
% rateCells = find(cdaData.area==1); %31 cells
% 
RAs = cda.RAs;
SAs = cda.SAs;
PCs = cda.PCs;

% alltimingCells = find(cellMax>.4); %50 cells
% timingCells = alltimingCells(cdaData.area(alltimingCells) == 1);
% rateCells = alltimingCells(cdaData.area(alltimingCells) == 3);

for popSize = 10
    for itInd = 1:500
    thesePCs = datasample(PCs,popSize,'replace',false);
    theseSAs = datasample(SAs,popSize,'replace',false);
    theseRAs = datasample(RAs,popSize,'replace',false);
    X = squeeze(nanmean(cat(3,dist_timing(:,thesePCs),dist_rate(:,thesePCs),dist_timing(:,theseSAs),dist_rate(:,theseSAs),dist_timing(:,theseRAs),dist_rate(:,theseRAs)),2));
    X(:,7) = ones(size(X,1),1);
    for loInd = 1:156
        allbut = ones(156,1);
        allbut(loInd) = 0;
        allbut = find(allbut);
        [B,BINT,R,RINT,STATS(itInd,:)] = regress(y(allbut),X(allbut,:));
        actual = y(loInd);
        predicted = B(1)*X(loInd,1) + B(2) * X(loInd,2) + B(3) * X(loInd,3) + B(4) * X(loInd,4) + B(5) * X(loInd,5) + B(6) * X(loInd,6) + B(7);
        predictionerror(popSize,loInd,itInd,1) = (actual-predicted)^2;
        allpredictions(popSize,loInd,itInd,1) = predicted;
%         other models
        B_PC = regress(y(allbut),X(allbut,[1 2 7]));
            predicted_PC = B_PC(1) * X(loInd,1) + B_PC(2) * X(loInd,2) + B_PC(3) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,2) = (actual-predicted_PC)^2;
            allpredictions(popSize,loInd,itInd,2) = predicted_PC;
        B_SA = regress(y(allbut),X(allbut,[3 4 7]));
            predicted_SA = B_SA(1) * X(loInd,3) + B_SA(2) * X(loInd,4) + B_SA(3) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,3) = (actual-predicted_SA)^2;
            allpredictions(popSize,loInd,itInd,3) = predicted_SA;
        B_RA = regress(y(allbut),X(allbut,[5 6 7]));
            predicted_RA = B_RA(1) * X(loInd,5) + B_RA(2) * X(loInd,6) + B_RA(3) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,4) = (actual-predicted_RA)^2;
            allpredictions(popSize,loInd,itInd,4) = predicted_RA;
            
        
        B_t = regress(y(allbut),X(allbut,[1 3 5 7])); %
            predicted_t = B_t(1) * X(loInd,1) + B_t(2) * X(loInd,3) + B_t(3) * X(loInd,5) + B_t(4) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,5) = (actual-predicted_t)^2;
            allpredictions(popSize,loInd,itInd,5) = predicted_t;
            
        B_r = regress(y(allbut),X(allbut,[2 4 6 7])); %
            predicted_r = B_r(1) * X(loInd,2) + B_r(2) * X(loInd,4) + B_r(3) * X(loInd,6) + B_r(4) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,6) = (actual-predicted_r)^2;
            allpredictions(popSize,loInd,itInd,6) = predicted_r;
        B_optimal = regress(y(allbut),X(allbut,[1 4 7])); %
            predicted_optimal = B_optimal(1) * X(loInd,1) + B_optimal(2) * X(loInd,4) + B_optimal(3) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,7) = (actual-predicted_optimal)^2;
            optimalpredictions(popSize,loInd,itInd) = predicted_optimal;
            allpredictions(popSize,loInd,itInd,7) = predicted_optimal;
        B_PCRA = regress(y(allbut),X(allbut,[1 6 7])); %
            predicted_optimal = B_PCRA(1) * X(loInd,1) + B_PCRA(2) * X(loInd,6) + B_PCRA(3) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,8) = (actual-predicted_optimal)^2;
            allpredictions(popSize,loInd,itInd,8) = predicted_optimal;
        B_RASA = regress(y(allbut),X(allbut,[5 4 7])); %
            predicted_optimal = B_RASA(1) * X(loInd,5) + B_RASA(2) * X(loInd,4) + B_RASA(3) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,9) = (actual-predicted_optimal)^2;
            allpredictions(popSize,loInd,itInd,9) = predicted_optimal;  
            
        B_PCRASA = regress(y(allbut),X(allbut,[1 4 6 7]));
            predicted_optimal = B_PCRASA(1) * X(loInd,1) + B_PCRASA(2) * X(loInd,4) + B_PCRASA(3) * X(loInd,6) + B_PCRASA(4) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,10) = (actual-predicted_optimal)^2;
            allpredictions(popSize,loInd,itInd,10) = predicted_optimal;
            
        B_PCandrate = regress(y(allbut),X(allbut,[1 2 4 6 7]));
            predicted_optimal = B_PCandrate(1) * X(loInd,1) + B_PCandrate(2) * X(loInd,2) + B_PCandrate(3) * X(loInd,4) + B_PCandrate(4) * X(loInd,6) + B_PCandrate(5) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,11) = (actual-predicted_optimal)^2;
            allpredictions(popSize,loInd,itInd,11) = predicted_optimal;
            
            
        B_12 = regress(y(allbut),X(allbut,[1 2 3 4 6 7]));
            predicted_optimal = B_12(1) * X(loInd,1) + B_12(2) * X(loInd,2) + B_12(3) * X(loInd,3) + B_12(4) * X(loInd,4) + B_12(5) * X(loInd,6) + B_12(6) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,12) = (actual-predicted_optimal)^2;
            allpredictions(popSize,loInd,itInd,12) = predicted_optimal;
            
         B_13 = regress(y(allbut),X(allbut,[1 2 4 5 6 7]));
            predicted_optimal = B_13(1) * X(loInd,1) + B_13(2) * X(loInd,2) + B_13(3) * X(loInd,4) + B_13(4) * X(loInd,5) + B_13(5) * X(loInd,6) + B_13(6) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,13) = (actual-predicted_optimal)^2;
            allpredictions(popSize,loInd,itInd,13) = predicted_optimal;
            
                 B_14 = regress(y(allbut),X(allbut,[ 2 3 4 5 6 7]));
            predicted_optimal = B_14(1) * X(loInd,2) + B_14(2) * X(loInd,3) + B_14(3) * X(loInd,4) + B_14(4) * X(loInd,5) + B_14(5) * X(loInd,6) + B_14(6) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,14) = (actual-predicted_optimal)^2;
            allpredictions(popSize,loInd,itInd,14) = predicted_optimal;    
            
        modelName = {'all' 'PC' 'SA' 'RA' 'timing_all' 'rate_all' 'PCSA' 'PCRA' 'RASA' 'PCRASA' 'B_PCandrate' 'allbutRAtiming' 'allbutSAtiming' 'allbutPCtiming'};
        
    end
    end
end

figure; hold on;
colorVals = [ 0 0 0; colorPC; colorSA; colorRA; colorRate; colorTiming; colorGrey; colorGrey*.6; colorGrey*.4; colorGrey*.2; colorPC*.8; colorSA*.8; colorRA*.8; colorPC*.6];
for errorModel = 1:length(modelName)
plot([1:12],nanmean(predictionerror(:,:,:,errorModel),[2 3])','linewidth',2,'color',colorVals(errorModel,:));
end
legend(modelName)
legend boxoff
ylim([ 0 .18]);
xlabel('population size')
ylabel('MSE')

figure; thisInd = 0;
for errorModel = [1 5 6 4 2 3 7 8 9 10 11 12 13 14]
    thisInd = thisInd + 1;
subplot(5,3,thisInd)
predicted = squeeze(nanmean(allpredictions(10,:,:,errorModel),3))';

scatter(predicted,y,'filled','CData',colorVals(errorModel,:))
title(modelName(errorModel))
hold on;
plot([.5 2],[.5 2],'--','color','k')
xlim([0.5 2]); ylim([0.5 2])
end

groupMSE = squeeze(nanmean(predictionerror(10,:,1:100,:),2));
[p,t,stats] = kruskalwallis(groupMSE);
multcompare(stats)

figure;
[errormeans,meanInd]=sort(squeeze(nanmean(predictionerror(10,:,1:100,:),[2 3])),'descend');
thisInd = 0;
for errorModel = 1:length(meanInd)
    errorModelInd = meanInd(errorModel);
    lieberplot_ind_std(squeeze(nanmean(predictionerror(10,:,1:100,errorModelInd),2)),errorModel);
    [p,h,stats] = ranksum(groupMSE(1:10,1),groupMSE(1:10,errorModelInd),'tail','left');
    text(errorModel,.17,num2str(round(p*100,2)))
end
xticklabels(modelName(meanInd))


%% Create subsampled distribution of standardized regression coefficients
dist_timing = perception.dist_timing;
dist_rate == perception.dist_rate;
y = perception.dissim_vec;

% timingCells = find(cdaData.area==3); %56 cells
% rateCells = find(cdaData.area==1); %31 cells
% 
RAs = cda.RAs;
SAs = cda.SAs;
PCs = cda.PCs;
all = cda.allcells;

popSize = 10;
clear B;
for itInd = 1:1000
thesePCs = datasample(PCs,popSize,'replace',false);
    theseSAs = datasample(SAs,popSize,'replace',false);
    theseRAs = datasample(RAs,popSize,'replace',false);
    
     X = squeeze(nanmean(cat(3,dist_timing(:,thesePCs),dist_rate(:,thesePCs),dist_timing(:,theseSAs),dist_rate(:,theseSAs),dist_timing(:,theseRAs),dist_rate(:,theseRAs)),2));
    X(:,7) = ones(size(X,1),1);
    
    mdl = fitlm(nanzscore(X(allbut,[1 2 3 4 5 7]),[],1),y(allbut,:));
    B(:,itInd) = mdl.Coefficients.Estimate;
    R2(:,itInd) = mdl.Rsquared.Adjusted;
end

[coeffVal,coeffOrder] = sort(nanmean(abs(B(2:end,:)),2),'descend');
mdlFactors = {'PC timing' 'PC rate' 'SA timing' 'SA rate' 'RA timing' 'null'};
mdlFactors(coeffOrder)'
coeffVal
%%
timingCells = cda.RAs;
rateCells = cda.SAs;


for popSize = 12
    for itInd = 1:500
    thesePCs = datasample(timingCells,popSize,'Replace',false);
    theseSAs = datasample(rateCells,popSize,'Replace',false);
    theseAlls = datasample(cda.allcells,popSize,'replace',false);
    X = squeeze(nanmean(cat(3,dist_timing(:,thesePCs),dist_rate(:,thesePCs),dist_timing(:,theseSAs),dist_timing(:,theseAlls)),2));
    X(:,5) = ones(size(X,1),1);
    for loInd = 1:156
        allbut = ones(156,1);
        allbut(loInd) = 0;
        allbut = find(allbut);
        [B,BINT,R,RINT,STATS(itInd,:)] = regress(y(allbut),X(allbut,:));
        actual = y(loInd);
        predicted = B(1)*X(loInd,1) + B(2) * X(loInd,2) + B(3) * X(loInd,3) + B(4) * X(loInd,4) + B(5);
        predictionerror(popSize,loInd,itInd,1) = (actual-predicted)^2;
        allpredictions(popSize,loInd,itInd,1) = predicted;
%         other models
        B_PC = regress(y(allbut),X(allbut,[1 5]));
            predicted_PC = B_PC(1) * X(loInd,1)  + B_PC(2) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,2) = (actual-predicted_PC)^2;
            allpredictions(popSize,loInd,itInd,2) = predicted_PC;
        B_SA = regress(y(allbut),X(allbut,[3 5]));
            predicted_SA = B_SA(1) * X(loInd,3) + B_SA(2) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,3) = (actual-predicted_SA)^2;
            allpredictions(popSize,loInd,itInd,3) = predicted_SA;
        B_t = regress(y(allbut),X(allbut,[4 5]));
            predicted_t = B_t(1) * X(loInd,4) + B_t(2) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,4) = (actual-predicted_t)^2;
            allpredictions(popSize,loInd,itInd,4) = predicted_t;
        B_r = regress(y(allbut),X(allbut,[2 5]));
            predicted_r = B_r(1) * X(loInd,2) + B_r(2) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,5) = (actual-predicted_r)^2;
            allpredictions(popSize,loInd,itInd,5) = predicted_r;
        B_optimal = regress(y(allbut),X(allbut,[1 4 5]));
            predicted_optimal = B_optimal(1) * X(loInd,1) + B_optimal(2) * X(loInd,4) + B_optimal(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,6) = (actual-predicted_optimal)^2;
            optimalpredictions(popSize,loInd,itInd) = predicted_optimal;
            allpredictions(popSize,loInd,itInd,6) = predicted_optimal;
        modelName = {'all' 'PC timing' 'SA timing' 'All timing' 'PC rate' 'optimal'};
        
    end
    end
end

figure; hold on;
colorVals = [ 0 0 0; colorPC; colorSA; colorRate; colorTiming; colorGrey];
for errorModel = 1:6
scatter([12],nanmean(predictionerror(12,:,:,errorModel),[2 3])',50,colorVals(errorModel,:),'filled');
end
legend(modelName)
legend boxoff
ylim([ 0 .15]);
xlabel('population size')
ylabel('MSE')

%% subsampled iteration of adjusted r2 values
for popSize = 10
    for itInd = 1:1000
    thesePCs = datasample(PCs,popSize,'replace',false);
    theseSAs = datasample(SAs,popSize,'replace',false);
    theseRAs = datasample(RAs,popSize,'replace',false);
    X = squeeze(nanmean(cat(3,dist_timing(:,thesePCs),dist_rate(:,thesePCs),dist_timing(:,theseSAs),dist_rate(:,theseSAs),dist_timing(:,theseRAs),dist_rate(:,theseRAs)),2));
    X(:,7) = ones(size(X,1),1);
    
    thesePCs = datasample(cda.allcells,popSize,'replace',false);
    theseSAs = datasample(cda.allcells,popSize,'replace',false);
    theseRAs = datasample(cda.allcells,popSize,'replace',false);
    X_all = squeeze(nanmean(cat(3,dist_timing(:,thesePCs),dist_rate(:,thesePCs),dist_timing(:,theseSAs),dist_rate(:,theseSAs),dist_timing(:,theseRAs),dist_rate(:,theseRAs)),2));
    X_all(:,7) = ones(size(X,1),1);
    
    X_uni = squeeze(nanmean(dist_rate(:,[thesePCs theseSAs theseRAs]),2));
    X_uni(:,2) = ones(size(X_uni,1),1);
    
    for loInd = 1:156
        allbut = ones(156,1);
        allbut(loInd) = 0;
        allbut = find(allbut);
    
    mdl = fitlm(X(allbut,[2 4 6 7]),y(allbut,:));
    mdl_all = fitlm(X_all(allbut,[2 4 6 7]),y(allbut,:));
    mdl_uni = fitlm(X_uni(allbut,[1 2]),y(allbut,:));
    
    submodality_adjr2(itInd,1,loInd) = mdl.Rsquared.Adjusted;
    submodality_adjr2(itInd,2,loInd) = mdl_all.Rsquared.Adjusted;
    submodality_adjr2(itInd,3,loInd) = mdl_uni.Rsquared.Adjusted;
    
    MSE(itInd,1,loInd) = mdl.RMSE^2;
    MSE(itInd,2,loInd) = mdl_all.RMSE^2;
    MSE(itInd,3,loInd) = mdl_uni.RMSE^2;
    end
    end
end

figure;
Violin(nanmean(submodality_adjr2(:,1,:),3),1)
Violin(nanmean(submodality_adjr2(:,2,:),3),2)
Violin(nanmean(submodality_adjr2(:,3,:),3),3)
xticks([1 2 3])
xticklabels({'Submodality' 'Random subpopulations' 'Univariate rate'})
ylabel('Adjusted R^2')
title('Samples of 10 cells')
    
%% AREA PERCEPTION

dist_timing = perception.dist_timing;
dist_rate = perception.dist_rate;
y = perception.dissim_vec;

area3Cells = find(cdaData.area==3); %56 cells
area2Cells = find(cdaData.area==1); %31 cells
area1Cells = find(cdaData.area==3);


for popSize = 1:25
    for itInd = 1:500
    thesePCs = datasample(area1Cells,popSize);
    theseSAs = datasample(area2Cells,popSize);
    theseRAs = datasample(area3Cells,popSize);
    X = squeeze(nanmean(cat(3,dist_timing(:,thesePCs),dist_rate(:,thesePCs),dist_timing(:,theseSAs),dist_rate(:,theseSAs),dist_timing(:,theseRAs),dist_rate(:,theseRAs)),2));
    X(:,7) = ones(size(X,1),1);
    for loInd = 1:156
        allbut = ones(156,1);
        allbut(loInd) = 0;
        allbut = find(allbut);
        [B,BINT,R,RINT,STATS(itInd,:)] = regress(y(allbut),X(allbut,:));
        actual = y(loInd);
        predicted = B(1)*X(loInd,1) + B(2) * X(loInd,2) + B(3) * X(loInd,3) + B(4) * X(loInd,4) + B(5) * X(loInd,5) + B(6) * X(loInd,6) + B(7);
        predictionerror(popSize,loInd,itInd,1) = (actual-predicted)^2;
        allpredictions(popSize,loInd,itInd) = predicted;
%         other models
        B_PC = regress(y(allbut),X(allbut,[1 2 7]));
            predicted_PC = B_PC(1) * X(loInd,1) + B_PC(2) * X(loInd,2) + B_PC(3) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,2) = (actual-predicted_PC)^2;
        B_SA = regress(y(allbut),X(allbut,[3 4 7]));
            predicted_SA = B_SA(1) * X(loInd,3) + B_SA(2) * X(loInd,4) + B_SA(3) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,3) = (actual-predicted_SA)^2;
        B_t = regress(y(allbut),X(allbut,[5 6 7]));
            predicted_t = B_t(1) * X(loInd,5) + B_t(2) * X(loInd,6) + B_t(3) * X(loInd,7);
            predictionerror(popSize,loInd,itInd,4) = (actual-predicted_t)^2;
        modelName = {'all' '3b' '1' '2'};
        
    end
    end
end

figure; hold on;
colorVals = [ 0 0 0; colorPC; colorSA; colorRate; colorTiming; colorGrey];
for errorModel = 1:4
plot([1:25],nanmean(predictionerror(:,:,:,errorModel),[2 3])','linewidth',2,'color',colorVals(errorModel,:));
end
legend(modelName)
legend boxoff
ylim([ 0 .15]);
xlabel('population size')
ylabel('MSE')



%% fancy plot
figure;
scatter(squeeze(nanmean(allpredictions(12,:,:),3)),y,'filled')
hold on;
plot([0 2.5],[0 2.5],'--','color','k')


%% quad model population: MSE PERIPHERAL
% PC, SA, rate, timing
% cross validated, predict left out texture pair from all others

for hide = 1
ncells = 39;

load('rate_euc_per.mat')
ratedistmat= fixdistmat(distmat);

load('PSTH_XCov_distmat_CORRECT.mat')
% load('spikeDist_cda_ALLQs.mat')
PSTHdistmat=  fixdistmat(distmat);
[cellMax,cellRes]  = max(per.PSTHresults,[],2);


load('dissimData.mat');
perceptualDissim = dissimData.mat;

ratedistmat = ratedistmat(dissimData.pTextInd,dissimData.pTextInd,:,:,:);
PSTHdistmat = PSTHdistmat(dissimData.pTextInd,dissimData.pTextInd,:,:,:,:);
for cellInd = 1:ncells
    PSTHdistmat_best(:,:,:,:,cellInd) = PSTHdistmat(:,:,:,:,cellRes(cellInd),cellInd);
end

for tInd = 1:13
    ratedistmat(tInd,tInd,:,:,:) = nan;
    PSTHdistmat_best(tInd,tInd,:,:,:,:) = nan;
    perceptualDissim(tInd,tInd) = nan;
end

meanPSTHdistmat = reshape(nanmean(PSTHdistmat_best,[3 4]),13*13,ncells);
meanPerceptualdistmat = reshape(perceptualDissim,13*13,1);
meanratedistmat = reshape(nanmean(ratedistmat,[3 4]),13*13,ncells);

meanPSTHdistmat = meanPSTHdistmat(~isnan(meanPSTHdistmat(:,1)),:);
meanPerceptualdistmat = meanPerceptualdistmat(~isnan(meanPerceptualdistmat));
meanratedistmat = meanratedistmat(~isnan(meanratedistmat(:,1)),:);
end

dist_timing = meanPSTHdistmat;
dist_rate = meanratedistmat;
y = meanPerceptualdistmat;

% timingCells = find(cellMax>.4); %56 cells
% rateCells = find(cda.RATEresults>.1); %31 cells

timingCells = per.PCs;
rateCells = per.SAs;
% 
% alltimingCells = find(cellMax>.4); %50 cells
% timingCells = alltimingCells(cdaData.area(alltimingCells) == 1);
% rateCells = alltimingCells(cdaData.area(alltimingCells) == 3);

for popSize = 1:7
    for itInd = 1:100
    thesePCs = datasample(timingCells,popSize);
    theseSAs = datasample(rateCells,popSize);
    X = squeeze(nanmean(cat(3,dist_timing(:,thesePCs),dist_rate(:,thesePCs),dist_timing(:,theseSAs),dist_rate(:,theseSAs)),2));
    X(:,5) = ones(size(X,1),1);
    for loInd = 1:156
        allbut = ones(156,1);
        allbut(loInd) = 0;
        allbut = find(allbut);
        [B,BINT,R,RINT,STATS(itInd,:)] = regress(y(allbut),X(allbut,:));
        actual = y(loInd);
        predicted = B(1)*X(loInd,1) + B(2) * X(loInd,2) + B(3) * X(loInd,3) + B(4) * X(loInd,4) + B(5);
        predictionerror(popSize,loInd,itInd,1) = (actual-predicted)^2;
        % other models
        B_PC = regress(y(allbut),X(allbut,[1 2 5]));
            predicted_PC = B_PC(1) * X(loInd,1) + B_PC(2) * X(loInd,2) + B_PC(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,2) = (actual-predicted_PC)^2;
        B_SA = regress(y(allbut),X(allbut,[3 4 5]));
            predicted_SA = B_SA(1) * X(loInd,3) + B_SA(2) * X(loInd,4) + B_SA(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,3) = (actual-predicted_SA)^2;
        B_t = regress(y(allbut),X(allbut,[1 3 5]));
            predicted_t = B_t(1) * X(loInd,1) + B_t(2) * X(loInd,3) + B_t(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,4) = (actual-predicted_t)^2;
        B_r = regress(y(allbut),X(allbut,[2 4 5]));
            predicted_r = B_r(1) * X(loInd,2) + B_r(2) * X(loInd,4) + B_r(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,5) = (actual-predicted_r)^2;
        B_optimal = regress(y(allbut),X(allbut,[1 4 5]));
            predicted_optimal = B_optimal(1) * X(loInd,1) + B_optimal(2) * X(loInd,4) + B_optimal(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,6) = (actual-predicted_optimal)^2;
        modelName = {'all' 'PC' 'SA' 'timing' 'rate' 'optimal'};
        
    end
    end
end

figure; hold on;
colorVals = [ 0 0 0; colorPC; colorSA; colorRate; colorTiming; colorGrey];
for errorModel = 1:6
plot([1:7],nanmean(predictionerror(:,:,:,errorModel),[2 3])','linewidth',2,'color',colorVals(errorModel,:));
end
legend(modelName)
legend boxoff
ylim([ 0 .15]);
xlabel('population size')
ylabel('MSE')

figure;
 hold on;
for errorModel = 1:6
errorshadeplot_nonlog([1:7],nanmean(predictionerror(:,:,:,errorModel),[2 3])',nanstd(predictionerror(:,:,:,errorModel),[],[2 3])',colorVals(errorModel,:));
end
legend(modelName)
legend boxoff
ylim([ 0 .15]);
xlabel('population size')
ylabel('MSE')

%% SINGLE CELL MSE
data = cda;
for cellInd = 1:141
    thisCell = cellInd;
    X = squeeze(nanmean(cat(3,perception.dist_timing(:,thisCell),perception.dist_rate(:,thisCell)),2));
    X(:,3) = ones(size(X,1),1);
    for loInd = 1:156
        allbut = ones(156,1);
        allbut(loInd) = 0;
        allbut = find(allbut);
        % complete
        [B,BINT,R,RINT,STATS] = regress(y(allbut),X(allbut,:));
        actual = y(loInd);
        predicted = B(1)*X(loInd,1) + B(2) * X(loInd,2) + B(3);
        predictionerror(cellInd,loInd,1) = (actual-predicted)^2;
        % rate
        [B,BINT,R,RINT,STATS] = regress(y(allbut),X(allbut,[2 3]));
        predicted =  B(1) * X(loInd,2) + B(2);
        predictionerror(cellInd,loInd,2) = (actual-predicted)^2;
        % timing
        [B,BINT,R,RINT,STATS] = regress(y(allbut),X(allbut,[1 3]));
        predicted = B(1)*X(loInd,1) + B(2);
        predictionerror(cellInd,loInd,3) = (actual-predicted)^2;
    end
end

figure;
pe = squeeze(nanmean(predictionerror,2));
lieberplot_ind(pe(:,2),1,data.SAs,data.PCs);
lieberplot_ind(pe(:,3),2,data.SAs,data.PCs);
lieberplot_ind(pe(:,1),3,data.SAs,data.PCs);
xticks([1:3]); xticklabels({'rate' 'timing' 'both'})
ylabel('MSE')

%% quad model population: MSE---- ZSCORE DISTMAT
% PC, SA, rate, timing
% cross validated, predict left out texture pair from all others

for hide = 1
ncells = 141;

load('ZSCORE_rate_euc_cda.mat');
ratedistmat = zs_ratedistmat;
load('ZSCORE_psth_xcov_cda.mat');
PSTHdistmat = zs_timingdistmat;
[cellMax,cellRes]  = max(cda.PSTHresults,[],2);


load('dissimData.mat');
perceptualDissim = dissimData.mat;

ratedistmat = ratedistmat(dissimData.textInd,dissimData.textInd,:,:);
PSTHdistmat_best = PSTHdistmat(dissimData.textInd,dissimData.textInd,:,:);
% for cellInd = 1:141
%     PSTHdistmat_best(:,:,:,:,cellInd) = PSTHdistmat(:,:,:,:,cellRes(cellInd),cellInd);
% end

for tInd = 1:13
    ratedistmat(tInd,tInd,:,:,:) = nan;
    PSTHdistmat_best(tInd,tInd,:,:,:,:) = nan;
    perceptualDissim(tInd,tInd) = nan;
end

meanPSTHdistmat = reshape(nanmean(PSTHdistmat_best,[3]),13*13,ncells);
meanPerceptualdistmat = reshape(perceptualDissim,13*13,1);
meanratedistmat = reshape(nanmean(ratedistmat,[3]),13*13,ncells);

meanPSTHdistmat = meanPSTHdistmat(~isnan(meanPSTHdistmat(:,1)),:);
meanPerceptualdistmat = meanPerceptualdistmat(~isnan(meanPerceptualdistmat));
meanratedistmat = meanratedistmat(~isnan(meanratedistmat(:,1)),:);
end

dist_timing = meanPSTHdistmat;
dist_rate = meanratedistmat;
y = meanPerceptualdistmat;

timingCells = find(cellMax>.4); %56 cells
rateCells = find(cda.RATEresults>.1); %31 cells

timingCells = cda.PCs;
rateCells = cda.SAs;
% 
% alltimingCells = find(cellMax>.4); %50 cells
% timingCells = alltimingCells(cdaData.area(alltimingCells) == 1);
% rateCells = alltimingCells(cdaData.area(alltimingCells) == 3);

for popSize = 1:12
    for itInd = 1:500
    thesePCs = datasample(timingCells,popSize);
    theseSAs = datasample(rateCells,popSize);
    X = squeeze(nanmean(cat(3,dist_timing(:,thesePCs),dist_rate(:,thesePCs),dist_timing(:,theseSAs),dist_rate(:,theseSAs)),2));
    X(:,5) = ones(size(X,1),1);
    for loInd = 1:156
        allbut = ones(156,1);
        allbut(loInd) = 0;
        allbut = find(allbut);
        [B,BINT,R,RINT,STATS(itInd,:)] = regress(y(allbut),X(allbut,:));
        actual = y(loInd);
        predicted = B(1)*X(loInd,1) + B(2) * X(loInd,2) + B(3) * X(loInd,3) + B(4) * X(loInd,4) + B(5);
        predictionerror(popSize,loInd,itInd,1) = (actual-predicted)^2;
        % other models
        B_PC = regress(y(allbut),X(allbut,[1 2 5]));
            predicted_PC = B_PC(1) * X(loInd,1) + B_PC(2) * X(loInd,2) + B_PC(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,2) = (actual-predicted_PC)^2;
        B_SA = regress(y(allbut),X(allbut,[3 4 5]));
            predicted_SA = B_SA(1) * X(loInd,3) + B_SA(2) * X(loInd,4) + B_SA(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,3) = (actual-predicted_SA)^2;
        B_t = regress(y(allbut),X(allbut,[1 3 5]));
            predicted_t = B_t(1) * X(loInd,1) + B_t(2) * X(loInd,3) + B_t(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,4) = (actual-predicted_t)^2;
        B_r = regress(y(allbut),X(allbut,[2 4 5]));
            predicted_r = B_r(1) * X(loInd,2) + B_r(2) * X(loInd,4) + B_r(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,5) = (actual-predicted_r)^2;
        B_optimal = regress(y(allbut),X(allbut,[1 4 5]));
            predicted_optimal = B_optimal(1) * X(loInd,1) + B_optimal(2) * X(loInd,4) + B_optimal(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,6) = (actual-predicted_optimal)^2;
        modelName = {'all' 'PC' 'SA' 'timing' 'rate' 'optimal'};
        
    end
    end
end

figure; hold on;
colorVals = [ 0 0 0; colorPC; colorSA; colorRate; colorTiming; colorGrey];
for errorModel = 1:6
plot([1:12],nanmean(predictionerror(:,:,:,errorModel),[2 3])','linewidth',2,'color',colorVals(errorModel,:));
end
legend(modelName)
legend boxoff
ylim([ 0 .15]);
xlabel('population size')
ylabel('MSE')

figure;
 hold on;
for errorModel = 1:6
errorshadeplot_nonlog([1:12],nanmean(predictionerror(:,:,:,errorModel),[2 3])',nanstd(predictionerror(:,:,:,errorModel),[],[2 3])',colorVals(errorModel,:));
end
legend(modelName)
legend boxoff
ylim([ 0 .15]);
xlabel('population size')
ylabel('MSE')

%% quad model population: fitlm, adjusted R2
% PC, SA, rate, timing
% cross validated, predict left out texture pair from all others
chosencells= cda.PCs;
chosencells = find(cdaData.area==2);

for hide = 1
ncells = length(chosencells);

load('Rate_Euc_cda.mat');
ratedistmat= fixdistmat(distmat);

load('PSTH_XCov_cda_expanded.mat');
% load('spikeDist_cda_ALLQs.mat')
PSTHdistmat=  fixdistmat(distmat);
[cellMax,cellRes]  = max(cda.PSTHresults,[],2);


load('dissimData.mat');
perceptualDissim = dissimData.mat;

ratedistmat = ratedistmat(dissimData.textInd,dissimData.textInd,:,:,:);
PSTHdistmat = PSTHdistmat(dissimData.textInd,dissimData.textInd,:,:,:,:);
for cellInd = 1:141
    PSTHdistmat_best(:,:,:,:,cellInd) = PSTHdistmat(:,:,:,:,cellRes(cellInd),cellInd);
end

for tInd = 1:13
    ratedistmat(tInd,tInd,:,:,:) = nan;
    PSTHdistmat_best(tInd,tInd,:,:,:,:) = nan;
    perceptualDissim(tInd,tInd) = nan;
end

meanPSTHdistmat = reshape(nanmean(PSTHdistmat_best(:,:,:,:,chosencells),[3 4]),13*13,ncells);
meanPerceptualdistmat = reshape(perceptualDissim,13*13,1);
meanratedistmat = reshape(nanmean(ratedistmat(:,:,:,:,chosencells),[3 4]),13*13,ncells);

meanPSTHdistmat = meanPSTHdistmat(~isnan(meanPSTHdistmat(:,1)),:);
meanPerceptualdistmat = meanPerceptualdistmat(~isnan(meanPerceptualdistmat));
meanratedistmat = meanratedistmat(~isnan(meanratedistmat(:,1)),:);
end

PC_timing = meanPSTHdistmat;
PC_rate = meanratedistmat;
y = meanPerceptualdistmat;

chosencells= cda.SAs;
chosencells = datasample(1:141,50);
chosencells = find(cdaData.area==2);

for hide = 1
ncells = length(chosencells);

load('Rate_Euc_cda.mat');
ratedistmat= fixdistmat(distmat);

load('PSTH_XCov_cda_expanded.mat');
% load('spikeDist_cda_ALLQs.mat')
PSTHdistmat=  fixdistmat(distmat);
[cellMax,cellRes]  = max(cda.PSTHresults,[],2);


load('dissimData.mat');
perceptualDissim = dissimData.mat;

ratedistmat = ratedistmat(dissimData.textInd,dissimData.textInd,:,:,:);
PSTHdistmat = PSTHdistmat(dissimData.textInd,dissimData.textInd,:,:,:,:);
for cellInd = 1:141
    PSTHdistmat_best(:,:,:,:,cellInd) = PSTHdistmat(:,:,:,:,cellRes(cellInd),cellInd);
end

for tInd = 1:13
    ratedistmat(tInd,tInd,:,:,:) = nan;
    PSTHdistmat_best(tInd,tInd,:,:,:,:) = nan;
    perceptualDissim(tInd,tInd) = nan;
end

meanPSTHdistmat = reshape(nanmean(PSTHdistmat_best(:,:,:,:,chosencells),[3 4]),13*13,ncells);
meanPerceptualdistmat = reshape(perceptualDissim,13*13,1);
meanratedistmat = reshape(nanmean(ratedistmat(:,:,:,:,chosencells),[3 4]),13*13,ncells);

meanPSTHdistmat = meanPSTHdistmat(~isnan(meanPSTHdistmat(:,1)),:);
meanPerceptualdistmat = meanPerceptualdistmat(~isnan(meanPerceptualdistmat));
meanratedistmat = meanratedistmat(~isnan(meanratedistmat(:,1)),:);
end

SA_timing = meanPSTHdistmat;
SA_rate = meanratedistmat;

for popSize = 1:12
    for itInd = 1:100
    thesePCs = datasample(1:length(cda.PCs),popSize);
    theseSAs = datasample(1:length(cda.SAs),popSize);
    X = squeeze(nanmean(cat(3,PC_timing(:,thesePCs),PC_rate(:,thesePCs),SA_timing(:,theseSAs),SA_rate(:,theseSAs)),2));
    X(:,5) = ones(size(X,1),1);
    mdl = fitlm(X,y);
    adjr2(popSize,itInd,1) = mdl.Rsquared.Adjusted;
    mdl_PC = fitlm(X(:,[1 2 5]),y);
    mdl_SA = fitlm(X(:,[3 4 5]),y);
    mdl_timing = fitlm(X(:,[1 3 5]),y);
    mdl_rate = fitlm(X(:,[2 4 5]),y);
    mdl_optimal = fitlm(X(:,[1 4 5]),y);
    adjr2(popSize,itInd,2:6) = [mdl_PC.Rsquared.Adjusted;mdl_SA.Rsquared.Adjusted;mdl_timing.Rsquared.Adjusted;mdl_rate.Rsquared.Adjusted;mdl_optimal.Rsquared.Adjusted];
   end
end

modelName = {'all' 'PC' 'SA' 'timing' 'rate' 'optimal'};


figure; hold on;
colorVals = [ 0 0 0; colorPC; colorSA; colorRate; colorTiming; colorGrey];
for errorModel = 1:6
plot([1:12],nanmean(adjr2(:,:,errorModel),2)','linewidth',2,'color',colorVals(errorModel,:));
end
legend(modelName)
legend boxoff
ylim([ 0 1]);
xlabel('population size')
ylabel('adjusted r2')



%% quad model population: fitlm, adjusted R2, simplified
% PC, SA, rate, timing
% cross validated, predict left out texture pair from all others

for hide = 1
ncells = 141;

load('Rate_Euc_cda.mat');
ratedistmat= fixdistmat(distmat);

load('PSTH_XCov_cda_expanded.mat');
% load('spikeDist_cda_ALLQs.mat')
PSTHdistmat=  fixdistmat(distmat);
[cellMax,cellRes]  = max(cda.PSTHresults,[],2);


load('dissimData.mat');
perceptualDissim = dissimData.mat;

ratedistmat = ratedistmat(dissimData.textInd,dissimData.textInd,:,:,:);
PSTHdistmat = PSTHdistmat(dissimData.textInd,dissimData.textInd,:,:,:,:);
for cellInd = 1:141
    PSTHdistmat_best(:,:,:,:,cellInd) = PSTHdistmat(:,:,:,:,cellRes(cellInd),cellInd);
end

for tInd = 1:13
    ratedistmat(tInd,tInd,:,:,:) = nan;
    PSTHdistmat_best(tInd,tInd,:,:,:,:) = nan;
    perceptualDissim(tInd,tInd) = nan;
end

meanPSTHdistmat = reshape(nanmean(PSTHdistmat_best,[3 4]),13*13,ncells);
meanPerceptualdistmat = reshape(perceptualDissim,13*13,1);
meanratedistmat = reshape(nanmean(ratedistmat,[3 4]),13*13,ncells);

meanPSTHdistmat = meanPSTHdistmat(~isnan(meanPSTHdistmat(:,1)),:);
meanPerceptualdistmat = meanPerceptualdistmat(~isnan(meanPerceptualdistmat));
meanratedistmat = meanratedistmat(~isnan(meanratedistmat(:,1)),:);
end

dist_timing = meanPSTHdistmat;
dist_rate = meanratedistmat;
y = meanPerceptualdistmat;


alltimingCells = find(cellMax>.4); %50 cells
timingCells = alltimingCells(cdaData.area(alltimingCells) == 1);
rateCells = alltimingCells(cdaData.area(alltimingCells) == 3);
% rateCells = find(cda.RATEresults>.1); %31 cells
% 
% timingCells = cda.PCs;
% rateCells = cda.SAs;


for popSize = 1:10
    for itInd = 1:100
    thesePCs = datasample(timingCells,popSize,'replace',false);
    theseSAs = datasample(rateCells,popSize,'replace',false);
    X = squeeze(nanmean(cat(3,dist_timing(:,thesePCs),dist_rate(:,thesePCs),dist_timing(:,theseSAs),dist_rate(:,theseSAs)),2));
    X(:,5) = ones(size(X,1),1);
    mdl = fitlm(X,y);
    rawr2(popSize,itInd,1) = mdl.Rsquared.Ordinary;
    adjr2(popSize,itInd,1) = mdl.Rsquared.Adjusted;
    
    mdl_PC = fitlm(X(:,[1 2 5]),y);
    mdl_SA = fitlm(X(:,[3 4 5]),y);
    mdl_timing = fitlm(X(:,[1 3 5]),y);
    mdl_rate = fitlm(X(:,[2 4 5]),y);
    mdl_optimal = fitlm(X(:,[1 4 5]),y);
    rawr2(popSize,itInd,2:6) = [mdl_PC.Rsquared.Ordinary;mdl_SA.Rsquared.Ordinary;mdl_timing.Rsquared.Ordinary;mdl_rate.Rsquared.Ordinary;mdl_optimal.Rsquared.Ordinary];

    adjr2(popSize,itInd,2:6) = [mdl_PC.Rsquared.Adjusted;mdl_SA.Rsquared.Adjusted;mdl_timing.Rsquared.Adjusted;mdl_rate.Rsquared.Adjusted;mdl_optimal.Rsquared.Adjusted];
   end
end

modelName = {'all' 'PC' 'SA' 'timing' 'rate' 'optimal'};


figure; hold on;
colorVals = [ 0 0 0; colorPC; colorSA; colorRate; colorTiming; colorGrey];
for errorModel = 1:6
plot([1:popSize],nanmean(adjr2(1:popSize,:,errorModel),2)','linewidth',2,'color',colorVals(errorModel,:));
end
legend(modelName)
legend boxoff
ylim([ 0 1]);
xlabel('population size')
ylabel('adjusted r2')

figure; hold on;
colorVals = [ 0 0 0; colorPC; colorSA; colorRate; colorTiming; colorGrey];
for errorModel = 1:6
plot([1:popSize],nanmean(rawr2(:,:,errorModel),2)','linewidth',2,'color',colorVals(errorModel,:));
end
legend(modelName)
legend boxoff
ylim([ 0 1]);
xlabel('population size')
ylabel('raw r2')

%% adj r2 for full population

for hide = 1
ncells = 141;

load('Rate_Euc_cda.mat');
ratedistmat= fixdistmat(distmat);

load('PSTH_XCov_cda_expanded.mat');
% load('spikeDist_cda_ALLQs.mat')
PSTHdistmat=  fixdistmat(distmat);
[cellMax,cellRes]  = max(cda.PSTHresults,[],2);


load('dissimData.mat');
perceptualDissim = dissimData.mat;

ratedistmat = ratedistmat(dissimData.textInd,dissimData.textInd,:,:,:);
PSTHdistmat = PSTHdistmat(dissimData.textInd,dissimData.textInd,:,:,:,:);
for cellInd = 1:141
    PSTHdistmat_best(:,:,:,:,cellInd) = PSTHdistmat(:,:,:,:,cellRes(cellInd),cellInd);
end

for tInd = 1:13
    ratedistmat(tInd,tInd,:,:,:) = nan;
    PSTHdistmat_best(tInd,tInd,:,:,:,:) = nan;
    perceptualDissim(tInd,tInd) = nan;
end

meanPSTHdistmat = reshape(nanmean(PSTHdistmat_best,[3 4]),13*13,ncells);
meanPerceptualdistmat = reshape(perceptualDissim,13*13,1);
meanratedistmat = reshape(nanmean(ratedistmat,[3 4]),13*13,ncells);

meanPSTHdistmat = meanPSTHdistmat(~isnan(meanPSTHdistmat(:,1)),:);
meanPerceptualdistmat = meanPerceptualdistmat(~isnan(meanPerceptualdistmat));
meanratedistmat = meanratedistmat(~isnan(meanratedistmat(:,1)),:);
end

dist_timing = meanPSTHdistmat;
dist_rate = meanratedistmat;
y = meanPerceptualdistmat;

allCells = 1:141;

for popSize = 1:141
    for itInd = 1:500
    thesePCs = datasample(allCells,popSize,'replace',false);
    X = squeeze(nanmean(cat(3,dist_timing(:,thesePCs),dist_rate(:,thesePCs)),2));
    X(:,3) = ones(size(X,1),1);
    mdl = fitlm(X,y);
    rawr2(popSize,itInd,1) = mdl.Rsquared.Ordinary;
    adjr2(popSize,itInd,1) = mdl.Rsquared.Adjusted;
    
    mdl_timing = fitlm(X(:,[1 3]),y);
    mdl_rate = fitlm(X(:,[2 3]),y);
    rawr2(popSize,itInd,2:3) = [mdl_timing.Rsquared.Ordinary;mdl_rate.Rsquared.Ordinary];

    adjr2(popSize,itInd,2:3) = [mdl_timing.Rsquared.Adjusted;mdl_rate.Rsquared.Adjusted];
   end
end

modelName = {'all' 'timing' 'rate'};


figure; hold on;
colorVals = [ 0 0 0; colorRate; colorTiming; colorGrey];
for errorModel = 1:3
plot([1:popSize],nanmean(adjr2(:,:,errorModel),2)','.','linewidth',2,'color',colorVals(errorModel,:));
plot([1:popSize],movmean(nanmean(adjr2(:,:,errorModel),2)',5),'color',colorVals(errorModel,:))
end
legend(modelName)
legend boxoff
ylim([ 0 1]);
xlabel('population size')
ylabel('adjusted r2')

figure; hold on;
colorVals = [ 0 0 0; colorPC; colorSA; colorRate; colorTiming; colorGrey];
for errorModel = 1:6
plot([1:popSize],nanmean(rawr2(:,:,errorModel),2)','linewidth',2,'color',colorVals(errorModel,:));
end
legend(modelName)
legend boxoff
ylim([ 0 1]);
xlabel('population size')
ylabel('raw r2')

%% TWO DIM SCATTER
% X = nanmean(nanzscore(perception.dist_timing(inds,cda.RAs),[],1),2);

% X = nanmean(perception.dist_timing_per(inds,per.RAs),2);
% x_name = 'RA timing correlation';
y = nanmean(nanzscore(perception.dist_timing(inds,cda.RAs),[],1),2);
y_name = 'RA-like timing correlation';
% X = nanmean(cda.rates(cda.SAs,cda.matchedtextureInds,:),[1 3]);
% x_name = 'SA1-like rates';


% y = abs(perception.deltarough_vec(inds));
% y_name = 'Delta Roughness';
% y = perception.dissim_vec(inds);
% y_name = 'Perceptual dissimilarity';
% y = nanmean(nanzscore(perception.dist_timing(inds,cda.PCs),[],1),2);
% y_name = 'PC timing correlation';
% % y = nanmean(cda.rates(cda.RAs,cda.matchedtextureInds,:),[1 3]);
% % y_name = 'RA cortical rates';
% y = nanmean(cda.rates(cda.RAs,cda.matchedtextureInds,:),[1 3]);
% y_name = 'RA-like afferent rates';
X = nanmean(nanzscore(perception.dist_timing_per(inds,per.RAs),[],1),2);
x_name = 'RA afferent timing correlation'




figure;
scatter(X,y,30,colorRA,'filled');
mdl = fitlm(X,y);
hold on;
fitx = [min(X):.001:max(X)];
coeffs = table2array(mdl.Coefficients);
fity = fitx*coeffs(2,1) + coeffs(1,1);
plot(fitx,fity,'--','color',colorGrey)
title(['adjusted r2: ' num2str(mdl.Rsquared.Adjusted)])
xlabel(x_name)
ylabel(y_name)
adjr2 = mdl.Rsquared.Adjusted;

%% TWO DIM SCATTER, per cell
theseCells = per.RAs;
clear adjr2
X_all = nanzscore(perception.dist_timing_per(inds,theseCells),[],1);
x_name = 'PC timing correlation';
% y = abs(perception.deltarough_vec(inds));
% y_name = 'Delta Roughness';
y = perception.dissim_vec(inds);
y_name = 'Perceptual dissimilarity';


figure;
hold on;
thisColor = lines(length(theseCells));
for cellInd = 1:length(theseCells)
scatter(X_all(:,cellInd),y,50,thisColor(cellInd,:),'filled');
X = X_all(:,cellInd);
mdl = fitlm(X,y);
hold on;
fitx = [min(X):.001:max(X)];
coeffs = table2array(mdl.Coefficients);
fity = fitx*coeffs(2,1) + coeffs(1,1);
plot(fitx,fity,'--','color',thisColor(cellInd,:))
title(['adjusted r2: ' num2str(mdl.Rsquared.Adjusted)])
xlabel(x_name)
ylabel(y_name)
adjr2(cellInd) = mdl.Rsquared.Adjusted;
end

%% THREE DIM SCATTER
X = cat(2,nanmean(perception.dist_rate(:,cda.SAs),2),nanmean(perception.dist_timing(:,cda.PCs),2));
x_name = 'SA rate';
x2_name = 'PC timing';
% y = abs(perception.deltarough_vec);
% z_name = 'Delta Roughness';
y = perception.dissim_vec;
z_name = 'Perceptual dissimilarity';
thiscolor = colorGrey;

figure;
scatter3(X(:,1),X(:,2),y,[],thiscolor,'filled');
X(:,3) = ones(size(X(:,1)));
mdl = fitlm(X,y);
hold on;
% fitx = [min(X):.1:max(X)];
% coeffs = table2array(mdl.Coefficients);
% fity = fitx*coeffs(2,1) + coeffs(1,1);
% plot(fitx,fity,'--','color',colorGrey)
title(['adjusted r2: ' num2str(mdl.Rsquared.Adjusted)])
xlabel(x_name)
ylabel(x2_name)
zlabel(z_name)

%% MSE of individual cells, cross validated

for hide = 1
ncells = 39;

load('rate_euc_per.mat')
ratedistmat= fixdistmat(distmat);

load('PSTH_XCov_distmat_CORRECT.mat')
% load('spikeDist_cda_ALLQs.mat')
PSTHdistmat=  fixdistmat(distmat);
[cellMax,cellRes]  = max(per.PSTHresults,[],2);


load('dissimData.mat');
perceptualDissim = dissimData.mat;

ratedistmat = ratedistmat(dissimData.pTextInd,dissimData.pTextInd,:,:,:);
PSTHdistmat = PSTHdistmat(dissimData.pTextInd,dissimData.pTextInd,:,:,:,:);
for cellInd = 1:ncells
    PSTHdistmat_best(:,:,:,:,cellInd) = PSTHdistmat(:,:,:,:,cellRes(cellInd),cellInd);
end

for tInd = 1:13
    ratedistmat(tInd,tInd,:,:,:) = nan;
    PSTHdistmat_best(tInd,tInd,:,:,:,:) = nan;
    perceptualDissim(tInd,tInd) = nan;
end

meanPSTHdistmat = reshape(nanmean(PSTHdistmat_best,[3 4]),13*13,ncells);
meanPerceptualdistmat = reshape(perceptualDissim,13*13,1);
meanratedistmat = reshape(nanmean(ratedistmat,[3 4]),13*13,ncells);

meanPSTHdistmat = meanPSTHdistmat(~isnan(meanPSTHdistmat(:,1)),:);
meanPerceptualdistmat = meanPerceptualdistmat(~isnan(meanPerceptualdistmat));
meanratedistmat = meanratedistmat(~isnan(meanratedistmat(:,1)),:);
end

dist_timing = meanPSTHdistmat;
dist_rate = meanratedistmat;
y = meanPerceptualdistmat;

% timingCells = find(cellMax>.4); %56 cells
% rateCells = find(cda.RATEresults>.1); %31 cells

timingCells = per.PCs;
rateCells = per.SAs;
% 
% alltimingCells = find(cellMax>.4); %50 cells
% timingCells = alltimingCells(cdaData.area(alltimingCells) == 1);
% rateCells = alltimingCells(cdaData.area(alltimingCells) == 3);

for popSize = 1:7
    for itInd = 1:100
    thesePCs = datasample(timingCells,popSize);
    theseSAs = datasample(rateCells,popSize);
    X = squeeze(nanmean(cat(3,dist_timing(:,thesePCs),dist_rate(:,thesePCs),dist_timing(:,theseSAs),dist_rate(:,theseSAs)),2));
    X(:,5) = ones(size(X,1),1);
    for loInd = 1:156
        allbut = ones(156,1);
        allbut(loInd) = 0;
        allbut = find(allbut);
        [B,BINT,R,RINT,STATS(itInd,:)] = regress(y(allbut),X(allbut,:));
        actual = y(loInd);
        predicted = B(1)*X(loInd,1) + B(2) * X(loInd,2) + B(3) * X(loInd,3) + B(4) * X(loInd,4) + B(5);
        predictionerror(popSize,loInd,itInd,1) = (actual-predicted)^2;
        % other models
        B_PC = regress(y(allbut),X(allbut,[1 2 5]));
            predicted_PC = B_PC(1) * X(loInd,1) + B_PC(2) * X(loInd,2) + B_PC(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,2) = (actual-predicted_PC)^2;
        B_SA = regress(y(allbut),X(allbut,[3 4 5]));
            predicted_SA = B_SA(1) * X(loInd,3) + B_SA(2) * X(loInd,4) + B_SA(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,3) = (actual-predicted_SA)^2;
        B_t = regress(y(allbut),X(allbut,[1 3 5]));
            predicted_t = B_t(1) * X(loInd,1) + B_t(2) * X(loInd,3) + B_t(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,4) = (actual-predicted_t)^2;
        B_r = regress(y(allbut),X(allbut,[2 4 5]));
            predicted_r = B_r(1) * X(loInd,2) + B_r(2) * X(loInd,4) + B_r(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,5) = (actual-predicted_r)^2;
        B_optimal = regress(y(allbut),X(allbut,[1 4 5]));
            predicted_optimal = B_optimal(1) * X(loInd,1) + B_optimal(2) * X(loInd,4) + B_optimal(3) * X(loInd,5);
            predictionerror(popSize,loInd,itInd,6) = (actual-predicted_optimal)^2;
        modelName = {'all' 'PC' 'SA' 'timing' 'rate' 'optimal'};
        
    end
    end
end

figure; hold on;
colorVals = [ 0 0 0; colorPC; colorSA; colorRate; colorTiming; colorGrey];
for errorModel = 1:6
plot([1:7],nanmean(predictionerror(:,:,:,errorModel),[2 3])','linewidth',2,'color',colorVals(errorModel,:));
end
legend(modelName)
legend boxoff
ylim([ 0 .15]);
xlabel('population size')
ylabel('MSE')

figure;
 hold on;
for errorModel = 1:6
errorshadeplot_nonlog([1:7],nanmean(predictionerror(:,:,:,errorModel),[2 3])',nanstd(predictionerror(:,:,:,errorModel),[],[2 3])',colorVals(errorModel,:));
end
legend(modelName)
legend boxoff
ylim([ 0 .15]);
xlabel('population size')
ylabel('MSE')


%% PCA of dissim matrices
[coeff, score, latent] = pca(perception.dist_rate);
varexplained_rate = latent ./ sum(latent);
[coeff, score, latent] = pca(perception.dist_timing);
varexplained_timing = latent ./ sum(latent);
varexplained_rate(1:4)
varexplained_timing(1:4)

[coeff, score, latent] = pca(perception.dist_timing(:,cda.PCs));
varexplained_timing_PC = latent ./ sum(latent);
varexplained_timing_PC(1:4)

[coeff, score, latent] = pca(perception.dist_timing(:,cda.SAs(1:12)));
varexplained_timing_SA = latent ./ sum(latent);
varexplained_timing_SA(1:4)

%% area figures
[bestClass] = max(cda.PSTHresults,[],2);

figure;
areas = [3 1 2];
for areaInd = 1:3
    thisArea = areas(areaInd);
    bestClass_mod = bestClass;
    bestClass_mod(cdaData.area~=thisArea) = nan;
    lieberplot_ind(bestClass_mod,areaInd,intersect(cda.PCs,find(cdaData.area==thisArea)),intersect(cda.SAs,find(cdaData.area==thisArea)),intersect(cda.RAs,find(cdaData.area==thisArea)))
end

figure;
areas = [3 1 2];
for areaInd = 1:3
    thisArea = areas(areaInd);
    bestClass_mod = bestClass./cda.RATEresults;
    bestClass_mod(cdaData.area~=thisArea) = nan;
    lieberplot_ind(bestClass_mod,areaInd,intersect(cda.PCs,find(cdaData.area==thisArea)),intersect(cda.SAs,find(cdaData.area==thisArea)),intersect(cda.RAs,find(cdaData.area==thisArea)))
end

%%
Xrows = {'PC_t' 'PC_r' 'SA_t' 'SA_r'};
mdl
stepwisefit(X,y)

variablecoeff = corrcoef(X,'Rows','complete');
figure;
imagesc(abs(variablecoeff(1:4,1:4)));
xticks([1:4])
yticks([1:4])
xticklabels(Xrows); yticklabels(Xrows);

%%
Xrows = {'PC_t' 'PC_r' 'SA_t' 'SA_r'};
thisInd = 2;
for otherInd = [1 3 4]
figure;
plot(X(:,thisInd),X(:,otherInd),'o')
ylabel(Xrows(otherInd))
xlabel(Xrows(thisInd))
box off;
end

%%
chosencells = 1:141;
for hide = 1
ncells = length(chosencells);

load('Rate_Euc_cda.mat');
ratedistmat= fixdistmat(distmat);

load('PSTH_XCov_cda_expanded.mat');
% load('spikeDist_cda_ALLQs.mat')
PSTHdistmat=  fixdistmat(distmat);
[cellMax,cellRes]  = max(cda.PSTHresults,[],2);


load('dissimData.mat');
perceptualDissim = dissimData.mat;

ratedistmat = ratedistmat(dissimData.textInd,dissimData.textInd,:,:,:);
PSTHdistmat = PSTHdistmat(dissimData.textInd,dissimData.textInd,:,:,:,:);
for cellInd = 1:141
    PSTHdistmat_best(:,:,:,:,cellInd) = PSTHdistmat(:,:,:,:,cellRes(cellInd),cellInd);
end

for tInd = 1:13
    ratedistmat(tInd,tInd,:,:,:) = nan;
    PSTHdistmat_best(tInd,tInd,:,:,:,:) = nan;
    perceptualDissim(tInd,tInd) = nan;
end

meanPSTHdistmat = reshape(nanmean(PSTHdistmat_best(:,:,:,:,chosencells),[3 4]),13*13,ncells);
meanPerceptualdistmat = reshape(perceptualDissim,13*13,1);
meanratedistmat = reshape(nanmean(ratedistmat(:,:,:,:,chosencells),[3 4]),13*13,ncells);

meanPSTHdistmat = meanPSTHdistmat(~isnan(meanPSTHdistmat(:,1)),:);
meanPerceptualdistmat = meanPerceptualdistmat(~isnan(meanPerceptualdistmat));
meanratedistmat = meanratedistmat(~isnan(meanratedistmat(:,1)),:);
end

figure;
clear X; clear stats; clear coeffs;
X(:,:,1) = ones(size(meanPSTHdistmat));
X(:,:,2) = meanratedistmat;
X(:,:,3) = meanPSTHdistmat;
X = permute(X,[1 3 2]);

for cellInd = 1:141
    [coeffs(cellInd,:),~,~,~,stats(cellInd,:)] = regress(meanPerceptualdistmat,X(:,:,cellInd));
end

figure;
scatter3(X(:,2,86),X(:,3,86),meanPerceptualdistmat,'filled')
hold on
x1fit = min(X(:,2,86)):.1:max(X(:,2,86));
x2fit = min(X(:,3,86)):.01:max(X(:,3,86));
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
YFIT = coeffs(86,1) + coeffs(86,2)*X1FIT + coeffs(86,3)*X2FIT
mesh(X1FIT,X2FIT,YFIT)
xlabel('Rate')
ylabel('Timing')
zlabel('Perceptual dissimilarity')
view(50,10)
hold off

%% check spike distance

%% check afferents

%% dig into textures that drive this
load('roughData.mat')

dissimRough = roughData.roughMean(dissimData.textInd);
roughDiff = abs(repmat(dissimRough,1,13) - repmat(dissimRough,1,13)');
for tInd = 1:13
roughDiff(tInd,tInd) = nan;
end
roughDiff = roughDiff(~isnan(roughDiff));

figure;
subplot(3,1,1)
plot(roughDiff,meanPerceptualdistmat,'o')
box off;
xlabel('diference in Roughness')
ylabel('perceptual dissimilarity')

subplot(3,1,2)
plot(roughDiff,nanmean(meanPSTHdistmat(:,cda.PCs),2),'o')
box off;
xlabel('diference in Roughness')
ylabel('timing dissimilarity')

subplot(3,1,3)
plot(roughDiff,nanmean(meanratedistmat(:,cda.PCs),2),'o')
box off;
xlabel('diference in Roughness')
ylabel('rate dissimilarity')


dissimRates = nanmean(cda.rates(:,dissimData.textInd,:),3);
figure;
plot(squeeze(nanmean(dissimRates(:,:),1)),dissimRough,'o')
xlabel('PC rates')
ylabel('roughness')
box off;

[~,sortInd] = sort(dissimRough,'descend');
dissimTextures = cda.textures(dissimData.textInd);

%%
[~,inds] = unique(perception.dissim_vec);
chosencells = find(nanmean(cda.rates,[2 3])>40);

cellcorrelation = corrcoef(perception.dist_timing(inds,chosencells));
[~,PCsort] = sort(pRegCoeff(1,chosencells));
figure
imagesc(abs(cellcorrelation(PCsort,PCsort)))

for cellInd = 1:141
    cellcorrelation = corrcoef(perception.dist_rate(inds,cellInd),perception.dist_timing(inds,cellInd));
    cellcorrs(cellInd) = cellcorrelation(2,1);
end

% All textures

load('Rate_Euc_cda.mat');
ratedistmat= fixdistmat(distmat);

load('PSTH_XCov_cda_expanded.mat');
% load('spikeDist_cda_ALLQs.mat')
PSTHdistmat=  fixdistmat(distmat);
[cellMax,cellRes]  = max(cda.PSTHresults,[],2);


for cellInd = 1:141
    PSTHdistmat_best(:,:,:,:,cellInd) = PSTHdistmat(:,:,:,:,cellRes(cellInd),cellInd);
end

for tInd = 1:59
    ratedistmat(tInd,tInd,:,:,:) = nan;
    PSTHdistmat_best(tInd,tInd,:,:,:,:) = nan;
end

meanPSTHdistmat = reshape(nanmean(PSTHdistmat_best,[3 4]),59*59,ncells);
meanratedistmat = reshape(nanmean(ratedistmat,[3 4]),59*59,ncells);

meanPSTHdistmat = meanPSTHdistmat(~isnan(meanPSTHdistmat(:,1)),:);
meanratedistmat = meanratedistmat(~isnan(meanratedistmat(:,1)),:);

for cellInd = 1:141
    cellcorrelation = corrcoef(meanratedistmat(:,cellInd),meanPSTHdistmat(:,cellInd));
    cellcorrs(cellInd) = cellcorrelation(2,1);
end

cellcorrelation = corrcoef(meanratedistmat(:,chosencells));
[~,PCsort] = sort(pRegCoeff(1,chosencells));
figure
imagesc(abs(cellcorrelation(PCsort,PCsort)))



%% PERIPHERY VS CORTEX (SHARED TEXTURE SET)

matchedtextures = nan(59,1);
for tInd = 1:59
    if any(strcmp(cda.textures{tInd},per.textures))
        matchedtextures(tInd) = find(strcmp(cda.textures{tInd},per.textures));
    end
end

% CHANGE NAMES OF TEXTURES TO MATCH, THIS IS DONE NOW! -------------------
% cdaunmatchedtextures = cda.textures(find(isnan(matchedtextures)))
% perunmatchedtextures = per.textures(setdiff(1:55,matchedtextures(~isnan(matchedtextures))))'

% % 'Grating - 5mm' --> '5 mm Grating'
% % 'Grating - 1mm' --> '1 mm Grating'
% % 'Flag/Banner'   --> 'Nylon'
% % 'Drapery Tape (Foam Side)' --> 'Foam (Drapery Tape)'
% % '20% Wool Felt' --> '20 Percent Wool Felt'

% per.textures(find(strcmp('Grating - 5mm',per.textures))) = {'5 mm Grating'};
% per.textures(find(strcmp('Grating - 1mm',per.textures))) = {'1 mm Grating'};
% per.textures(find(strcmp('Flag/Banner',per.textures))) = {'Nylon'};
% per.textures(find(strcmp('Drapery Tape (Foam Side)',per.textures))) = {'Foam (Drapery Tape)'};
% per.textures(find(strcmp('20% Wool Felt',per.textures))) = {'20 Percent Wool Felt'};
% -------------------------------------------------------------------------

% cdaTextures = find(~isnan(matchedtextures));
% perTextures = matchedtextures(~isnan(matchedtextures));
distmat = fixdistmat(distmat);
results = GetResults_cov(nanmean(distmat(cda.matchedtextureInds,cda.matchedtextureInds,:,:,:,:),4));
meanresults = squeeze(nanmean(results,[1 2]))';
% 
% cda.matchedPSTHresults = meanresults;

distmat = fixdistmat(distmat);
distmat = distmat(cda.matchedtextureInds,cda.matchedtextureInds,:,:,:,:,:,:);


for itInd = 1:100
    theseCells = datasample(per.PCs,5,'replace',false);
    results = GetResults_cov(nanmean(distmat(:,:,:,:,5,theseCells),[4 5 6]));
    popresults(itInd) = nanmean(results,[1 2]);
end


for itInd = 1:100
    theseCells = datasample(cda.SAs,5,'replace',false);
    results = GetResults(nanmean(distmat(:,:,:,:,theseCells),[4 5]));
    popresults(itInd) = nanmean(results,[1 2]);
end

%% SUPPLEMENTARY FIGURE 7

[cellMax_cda, cellRes_cda] = max(cda.matchedPSTHresults,[],2);
[cellMax_per, cellRes_per] = max(per.matchedPSTHresults,[],2);

cellRes_cda = cda.PSTHbins(cellRes_cda);
cellRes_per = cda.PSTHbins(cellRes_per);

% cellMax_cda(nanmean(cda.rates,[2 3])<20) = nan;
% cellMax_per(nanmean(per.rates,[2 3])<20) = nan;

% cellRes_cda(cellMax_cda<.3) = nan;
% cellRes_per(cellMax_per<.3) = nan;


figure; hold on;
Violin(cellMax_cda,1);
Violin(cellMax_per,2);
xticks([1 2]); xticklabels({'cortical' 'peripheral'})
ylim([0 1]); ylabel('classification performance')
plot([0 3],[1/24 1/24],'--','color','k')

% SMALL POPULATION OF PERIPHERAL CELLS
% for itInd = 1:100
%     theseCells = datasample(1:141,2);
%     results = GetResults_cov(nanmean(distmat(cda.matchedtextureInds,cda.matchedtextureInds,:,:,:,theseCells),[4 6]));
%     allmeanresults(:,itInd) = nanmean(results,[1 2]);
% end
% [cellMax_perPop, cellRes_perPop] = max(allmeanresults,[],1);
% figure; hold on;
% Violin(cellMax_cda,1);
% Violin(cellMax_perPop,2);
% xticks([1 2]); xticklabels({'cortical' 'peripheral'})
% ylim([0 1]); ylabel('classification performance')
% plot([0 3],[1/24 1/24],'--','color','k')

lieberplot(cellMax_cda,cellMax_per,'sd',repmat(colorGrey,2,1))
xticklabels({'cortical' 'peripheral'})
ylabel('classification performance')
ylim([0 1])

% TEMPORAL RESOLUTION
figure;
cdfplot(cellRes_cda);
hold on;
cdfplot(cellRes_per)
legend cortical peripheral
grid off; box off; legend boxoff;
xlabel('temporal resolution (ms)')
ylabel('proportion of cells')

figure; hold on;
cdfplot(cellRes_cda(cda.PCs))
cdfplot(cellRes_cda(cda.SAs))
cdfplot(cellRes_per(per.PCs))
cdfplot(cellRes_per(per.SAs))
legend cPC cSA pPC pSA
grid off; box off; legend boxoff;
xlabel('temporal resolution (ms)')
ylabel('proportion of cells')

% Overall plot
data = cda;
figure; hold on;
errorshadeplot(cda.PSTHbins,nanmean(data.matchedPSTHresults(data.RAs,:),1),nanstd(data.matchedPSTHresults(data.RAs,:),[],1)./sqrt(length(data.RAs)),'k')
errorshadeplot(cda.PSTHbins,nanmean(data.matchedPSTHresults(data.PCs,:),1),nanstd(data.matchedPSTHresults(data.PCs,:),[],1)./sqrt(length(data.PCs)),colorPC)
errorshadeplot(cda.PSTHbins,nanmean(data.matchedPSTHresults(data.SAs,:),1),nanstd(data.matchedPSTHresults(data.SAs,:),[],1)./sqrt(length(data.SAs)),colorSA)
xticks([.1 .2 .5 1 2 5 10 20 50 100 200 500]./1000)
xticklabels([.1 .2 .5 1 2 5 10 20 50 100 200 500])
xlabel('temporal resolution (ms)')
ylabel('classification performance')
ylim([0 .5])
yticks([0 .25 .5])

% Ascending plot
data = per;
figure; hold on;
errorshadeplot(cda.PSTHbins,nanmean(data.matchedPSTHresults(data.PCs,:),1),nanstd(data.matchedPSTHresults(data.PCs,:),[],1)./sqrt(length(data.PCs)),colorPC)
errorshadeplot(cda.PSTHbins,nanmean(data.matchedPSTHresults(data.SAs,:),1),nanstd(data.matchedPSTHresults(data.SAs,:),[],1)./sqrt(length(data.SAs)),colorSA)

data = cda; lineColors = lines(3);
for areaInd = [3 1 2]
errorshadeplot(cda.PSTHbins,nanmean(data.matchedPSTHresults(cdaData.area==areaInd,:),1),nanstd(data.matchedPSTHresults(cdaData.area==areaInd,:),[],1)./sqrt(length(find(cdaData.area==areaInd))),lineColors(areaInd,:))
end

xticks([.1 .2 .5 1 2 5 10 20 50 100 200 500]./1000)
xticklabels([.1 .2 .5 1 2 5 10 20 50 100 200 500])
xlabel('temporal resolution (ms)')
ylabel('classification performance')
ylim([0 .45])
yticks([0 .2 .4])
legend({'' 'PC' '' 'SA' '' '3b' '' '1' '' '2'})

% CDF
figure; hold on;
cdfplot(cellRes_cda(cdaData.area == 3))
cdfplot(cellRes_cda(cdaData.area == 1))
cdfplot(cellRes_cda(cdaData.area == 2))
cdfplot(cellRes_per(per.PCs))
cdfplot(cellRes_per(per.SAs))
legend 3b 1 2 PC SA
grid off; box off; legend boxoff;
xlabel('temporal resolution (ms)')
ylabel('proportion of cells')
set(gca,'XScale','log')
xticks([.0001 .001 .01 .1 .5])
xticklabels([.0001 .001 .01 .1 .5]*1000)

%% SUPPLEMENTARY FIGURE 6
corticalsubpop = cda.others;
peripheralsubpop = per.RAs;
clear popResults_cdaPSTH popResults_perPSTH popResults_perrate popResults_cdarate

% lieberplot(cda.matchedRATEresults,per.matchedRATEresults,'sd',repmat(colorGrey,2,1))
% xticklabels({'cortical' 'peripheral'});

load('Rate_Euc_cda.mat')
distmat = fixdistmat(distmat);
cdarate = distmat(cda.matchedtextureInds,cda.matchedtextureInds,:,:,:);

load('rate_euc_per.mat')
distmat = fixdistmat(distmat);
perrate = distmat(per.matchedtextureInds,per.matchedtextureInds,:,:,:);
results = GetResults(nanmean(perrate,4));
meanresults = squeeze(nanmean(results,[1 2]));

load('PSTH_XCov_cda_expanded.mat')
distmat = fixdistmat(distmat);
cdaPSTH_all = distmat(cda.matchedtextureInds,cda.matchedtextureInds,:,:,:,:,:,:);
[cellMax,cellRes]  = max(cda.PSTHresults,[],2);
for cellInd = 1:141
    cdaPSTH(:,:,:,:,cellInd) = cdaPSTH_all(:,:,:,:,cellRes(cellInd),cellInd);
end

load('PSTH_XCov_distmat_CORRECT.mat')
distmat = fixdistmat(distmat);
perPSTH_all = distmat(per.matchedtextureInds,per.matchedtextureInds,:,:,:,:,:);
[cellMax,cellRes]  = max(per.PSTHresults,[],2);
for cellInd = 1:39
    perPSTH(:,:,:,:,cellInd) = perPSTH_all(:,:,:,:,cellRes(cellInd),cellInd);
end

% population analysis: rate

for popSize = 1:length(corticalsubpop)
    for itInd = 1:100
        theseCells = datasample(corticalsubpop,popSize,'replace',false);
        cdarateresults = GetResults(nanmean(cdarate(:,:,:,:,theseCells),[4 5]));
        popResults_cdarate(popSize,itInd) = nanmean(cdarateresults,[1 2]);
    end
end

for popSize = 1:length(peripheralsubpop)
    for itInd = 1:100
        theseCells = datasample(peripheralsubpop,popSize,'replace',false);
        perrateresults = GetResults(nanmean(perrate(:,:,:,:,theseCells),[4 5]));
        popResults_perrate(popSize,itInd) = nanmean(perrateresults,[1 2]);
    end
end


% population analysis: timing

for popSize = 1:length(corticalsubpop)
    for itInd = 1:100
        theseCells = datasample(corticalsubpop,popSize,'replace',false);
        cdaPSTHresults = GetResults_cov(nanmean(cdaPSTH(:,:,:,:,theseCells),[4 5]));
        popResults_cdaPSTH(popSize,itInd) = nanmean(cdaPSTHresults,[1 2]);
    end
end

for popSize = 1:length(peripheralsubpop)
    for itInd = 1:100
        theseCells = datasample(peripheralsubpop,popSize,'replace',false);
        perPSTHresults = GetResults_cov(nanmean(perPSTH(:,:,:,:,theseCells),[4 5]));
        popResults_perPSTH(popSize,itInd) = nanmean(perPSTHresults,[1 2]);
    end
end


figure;
hold on;
errorshadeplot_nonlog([1:length(corticalsubpop)],nanmean(popResults_cdarate,2)',nanstd(popResults_cdarate,[],2)',colorRate);
errorshadeplot_nonlog([1:length(corticalsubpop)],nanmean(popResults_cdaPSTH,2)',nanstd(popResults_cdaPSTH,[],2)',colorTiming);
errorshadeplot_nonlog([1:length(peripheralsubpop)],nanmean(popResults_perrate,2)',nanstd(popResults_perrate,[],2)',colorRate,'--');
errorshadeplot_nonlog([1:length(peripheralsubpop)],nanmean(popResults_perPSTH,2)',nanstd(popResults_perPSTH,[],2)',colorTiming,'--');

ylim([0 1])

