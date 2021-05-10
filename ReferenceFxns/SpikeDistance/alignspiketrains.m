function alignspiketrains(spiketrains,nspiketrains,timeoffset,windowlength)

spiketrains = squeeze(spikes(1,1,2,:));
nspiketrains = nmask(1,1,2);
timeoffset = -.01:.001:.01;
spikeoffsets = cellfun(@(x) x + offset, spiketrains

if nspiketrains < 2
    best_offset = 0;
    
else
    for rep1off = 1:length(timeoffset)
        
        
        off1 = timeoffset(rep1off);
        rep1 = cellfun(@(x) x + off1, spiketrains(1),'uniformoutput',0);
        for rep2off = 1:length(timeoffset)
            off2 = timeoffset(rep2off);
            rep2 = cellfun(@(x) x+off2, spiketrains(2),'uniformoutput',0);
            for rep3off = 1:length(timeoffset)
                off3 = timeoffset(rep3off);
                rep3 = cellfun(@(x) x+off3, spiketrains(3),'uniformoutput',0);
                for rep4off = 1:length(timeoffset)
                    off4 = timeoffset(rep4off);
                    rep4 = cellfun(@(x) x+off4, spiketrains(4),'uniformoutput',0);
                    nshared(rep1off,rep2off,rep3off,rep4off) = ncoinc_reps([rep1,rep2,rep3,rep4],windowlength,nspiketrains);
                end
            end
        end
    end
    
    

    
end

% %% old code
%         [spikeD, oo] = min(cell2mat(spikeDmat(:)));
%     [offset1, offset2] = ind2sub(size(spikeDmat),oo)
% 
%     spikeDmat = num2cell(zeros(noffsets*dimensions));
%     spiketrain_mat = repmat(spiketrains,1,length(timeoffset));
%     timeoffset_mat = repmat(timeoffset, 4, 1);
%     spiketrain_mat = cellfun(@(x,y) x(x >= y & x <= (y + .5)) - y, spiketrain_mat, num2cell(timeoffset_mat), 'uniformoutput',0);
%     
%     % Make a matrix of all possible combinations of spike trains
%     possiblecombinations = nchoosek(1:nspiketrains,2);
%     
%     for offset_ind1 = 1:length(timeoffset)
%         for offset_ind2 = 1:length(timeoffset)
%             for offset_ind3 = 1:length(timeoffset)
%                 for offset_ind4 = 1:length(timeoffset)
%                     for pair_ind = 1:size(possiblecombinations,1)
%                         pair = possiblecombinations(pair_ind,:);
%                         allpairs{pair
%                         spikeDmat{offset_ind1,offset_ind2, offset_ind3, offset_ind4} = spikeDist(spiketrain_mat{pair(1),offset_ind1},spiketrain_mat{pair(2),offset_ind2},q);
%                     end
%                 end
%             end
%         end
%     end
%     
%     spikeDist(spiketrain_mat{1,1},spiketrain_mat{2,1},1/4);
%     
%     
%     spikecell{1} = spiketrains(1:nspiketrains);
%     spiketrainmat = repmat(spikecell,noffsets*dimensions);
% 
%     offsetdims = noffsets*dimensions;
%     offsetdims(1,2) = 1;
%     offsetmat = repmat(timeoffset,offsetdims);
%     for rep = 1:nspiketrains
%         order = circshift([1:nspiketrains],[rep,0]);
%         offsetmat{rep} = repmat(timeoffset,offsetdims);
%     end
%     offset1 = repmat(timeoffset,noffset*[ones(1,nspiketrains));
% 
% 
%     firstrep = cellfun(@(x) x(x <= maxT & x >= minT) - minT+ slide/2, spiketrains(:,:,:,1), 'uniformoutput',0); % FIRST SPIKE AT t = 0
%     for rep = 1:nspiketrains
%         for offInd = 1:length(timeoffset)
%             slidingstart = timeoffset(offInd);
%             slidingend = slidingstart + windowlength;
%             slowspikes = cellfun(@(x) x(x <= slowend & x >= slowstart)-slowstart,spikes_within(:,:,slowspeed,:),'uniformoutput',0);
% 
% 
%     for tInd = 1:ntextures
%         for offInd = 1:length(timeoffset)
%             slowstart = timeoffset(offInd);
%             slowend = slowstart + windowlength;
%             slowspikes = cellfun(@(x) x(x <= slowend & x >= slowstart)-slowstart,spikes_within(:,:,slowspeed,:),'uniformoutput',0);
%             nsharedspikes(offInd,:,:,:) = ncoincident(repmat(fastspikes(:,tInd,:,:),1,ntextures,1,1),slowspikes,windowlength);
%         end
%         [overlaps, offsets] = max(nsharedspikes,[],1);
%         overlaps = permute(overlaps,[2 3 4 1]);
%         offsets = permute(offsets,[2 3 4 1]);
%         offsets = timeoffset(offsets);
%         bestoffsets{tInd} = offsets;
%         bestoverlaps{tInd} = overlaps;
%         clear offsets; clear overlaps;
%     end
% 
%     save(['bestoffsets_' num2str(slowspeed) '_' num2str(fastspeed) '.mat'],'bestoffsets');
%     save(['bestoverlaps_' num2str(slowspeed) '_' num2str(fastspeed) '.mat'],'bestoverlaps');