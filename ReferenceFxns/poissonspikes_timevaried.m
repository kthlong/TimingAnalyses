function spiketimes = poissonspikes_timevaried(psth,binSize,minT,maxT)


repfactor = binSize/.0001;
repfactor = 1;
psth_mod =repmat(psth./repfactor,1,repfactor)';
psth_mod = psth_mod(:)';

times = minT:.0001:maxT;
times = times(1:end-1);
vt = rand(size(times));

spikes = (psth_mod*binSize) > vt;

spiketimes = times(find(spikes))';

end

