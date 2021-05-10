function jitteredspikes = makejitteredreps(spiketrain,amountjittered)

pd = makedist('Normal',0,amountjittered);
jitters = random(pd,size(spiketrain));
jitteredspikes = spiketrain + jitters;


% Rate-matched repetitions