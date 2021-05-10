function output = corticalSpikeTiming(data,method,minT,maxT,windowsOFF,scrambleOn,bins)
% this function is meant to be the first stage of processing for spike time
% data of any size. it makes no assumptions about the dimensions in the
% data. you must provide data, the output(s) you'd like, the minT, and the
% maxT. you can, optionally, specify the temporal resolution(s) you'd like
% to see in ISIs or PSTHs by specifying 'bins'. you can also use
% 'windowsOFF' to prevent this from assuming you want these generated at
% different window sizes (which I recommend)


if exist('windowsOFF') & strcmp(windowsOFF,'windowsOFF')
    windows = maxT-minT;
else
    windows = [.005,.01,.02,.05,.1,.2,.5,1,1.5,maxT-minT];
    windows = unique(windows(windows <= (maxT-minT)));
end

if exist('scrambleOn') & strcmp(scrambleOn,'scrambleOn')
    % THIS NEEDS ATTENTION! This has to be custom set!! %
    maxT_mod = 1.8; 
    [data,starttime] = scramblespiketimes(data,minT,maxT_mod,windows);
    output.starttime = starttime;
end

output.windows = windows;
output.range = [minT maxT];

QCmask = cellfun(@(x) ~isvector(x) | any(isnan(x)), data);

methodoptions = {'rate','fft','isi','psth','spikes'};

%% RATE
if strcmp(method,'rate')
    for windInd = 1:length(windows)
        endT = minT + windows(windInd);
        output.rate(windInd,:,:,:,:,:,:) = cellfun(@(x) length(x(x>=minT & x<=endT))/(endT-minT),data);
    end
    output.rate = squeeze(output.rate);
    output.rate(QCmask) = nan;

end

%% SPIKES
if strcmp(method,'spikes')
    output.spikes = data;
    data(QCmask) = {nan};
end

%% FFT
% NO WINDOWS
if strcmp(method,'fft')
 
    for windInd = 1
            windmaxT = minT + windows(windInd);
            nBins       = 800*(windmaxT-minT);
            edges       = linspace(minT,windmaxT,nBins+1);
            maxF        = nBins/(2*(windmaxT-minT));
            xFreq       = linspace(-maxF, maxF, nBins+1);
            xFreq       = xFreq(1:end-1);
            fftInd        = xFreq >= 0;
    
        cutdata         = cellfun(@(x)(x(x>=minT & x <= (windmaxT))), data, 'UniformOutput', false);
        bindata         = cellfun(@(x) Histc(x,edges), cutdata, 'UniformOutput', false);
        outputfft       = cellfun(@(x) abs(fftshift(fft(x-repmat(mean(x,1), [size(x,1) 1 1 1]),[],1),1)), bindata,'uniformoutput',0);
        matfft          = cell2mat(shiftdim(outputfft,-1));
%         windfft{windInd} = outputfft(fftInd,:,:,:,:,:);
    end
      
    QCmask_rep              = permute(repmat(QCmask,[1 1 1 size(matfft,1)]),[4 1 2 3]);
    matfft(QCmask_rep)      = nan;
    outputfft(QCmask)       = {nan};
    
    output.fftFreqs = xFreq;
    output.fft      = outputfft;
    output.matfft      = matfft;


end

%% ISI

if strcmp(method,'isi')
    
    if ~exist('bins')
        bins = [.0001,.0002,.0005,.001,.002,.005,.01,.02,.05,.1,.2,.5];
    end

    isimat = num2cell(nan([length(windows),length(bins),size(data)]));    

    for windInd = 1:length(windows)
        thesebins = bins(bins<windows(windInd)/2);
        for binInd = 1:length(thesebins)
            binWidth = thesebins(binInd);
            isimat(windInd,binInd,:,:,:,:,:) = ...
               cellfun(@(x) makeISI_onlyone(x,minT,(minT + windows(windInd)),binWidth,1),data,'uniformoutput',0);
        end
    end

    output.isi     = squeeze(isimat);
    output.isidim  = {'windows','binWidths','givenDims'};
    output.isibins = bins;
end

%% PSTH

if strcmp(method,'psth')
    
    if ~exist('bins')
      bins = [.001,.002,.005,.01,.02,.05,.1,.2,.5];
    end

    binSize = 1/5000;
    binEdges = minT:binSize:maxT;
    binCenters = binEdges(1:end-1) + binSize/2;


    for binInd = 1:length(bins)
        gaussWidth = bins(binInd);
        allpsth(binInd,:,:,:,:,:,:) = cellfun(@(x) makepsth(x,binEdges,gaussWidth),data,'uniformoutput',0);
    end

QCmask_rep              = permute(repmat(QCmask,[1 1 1 1 size(allpsth,1)]),[5 1 2 3 4]);
allpsth(QCmask_rep)      = {repmat(nan,size(allpsth{1,1,1,1,1,1,1}))};
output.psth = allpsth;
output.psthGaussWidths = bins;


end


