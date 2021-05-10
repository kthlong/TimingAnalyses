function SNR_timevec = SNR2_spikes(spiketrain, timebin)

if narg < 2
    timebin = .01;
end

PSTH = hist(spiketrain{:},[0:.001:1]);
slidingrate = movmean(PSTH,10);
slidingvar = movstd(PSTH,10);

SNR2_spikes = slidingrate.^2 ./ slidingvar;