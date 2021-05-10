% wrapper for spkdl.c
% 
% spikes is a cell array of spike time vectors (should be line vectors)
% q is the cost for moving a spike per unit time 
% (cost to add or remove spike is 1)

function dist=spkd_wrapper(spikes,q)

spikes=spikes(:);
l=cellfun(@length,spikes);

STIME=[spikes{:}]';
SSTART=cumsum([1;l(1:end-1)]);
SEND=cumsum(l);

idx=SEND-SSTART<0;

dist=spkdl(STIME,SSTART,SEND,q);
dist=reshape(dist,length(spikes),[]);