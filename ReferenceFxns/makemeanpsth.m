function meanpsth = makemeanpsth(data,binSize,minT,maxT)
% METHOD ONE
% binEdges = minT:binSize:maxT;
% 
% meanpsth = nan(size(data,1),size(data,2),size(data,3),size(binEdges,2)-1);
% 
% for cellInd = 1:size(data,1)
%     for tInd = 1:size(data,2)
%         for repInd = 1:size(data,3)
%             thesespikes = data{cellInd,tInd,repInd};
%             if length(thesespikes(thesespikes>=minT & thesespikes<=maxT)) > 0
%                 thispsth = squeeze(histcounts(thesespikes,binEdges));
%                 if size(thispsth,1) == 1
%                     meanpsth(cellInd,tInd,repInd,:) = thispsth';
%                 elseif size(thispsth,2) == 1
%                     meanpsth(cellInd,tInd,repInd,:) = thispsth;
%                 else
%                     pause;
%                 end
%             end
%         end
%     end
% end

% METHOD TWO
binEdges = minT:.0001:maxT;
nbins = binSize/.0001;

meanpsth = nan(size(data,1),size(data,2),size(data,3),size(binEdges,2)-1);

for cellInd = 1:size(data,1)
    for tInd = 1:size(data,2)
        for repInd = 1:size(data,3)
            thesespikes = data{cellInd,tInd,repInd};
            if length(thesespikes(thesespikes>=minT & thesespikes<=maxT)) > 0
                binspikes = squeeze(histcounts(thesespikes,binEdges));
                thispsth = movmean(binspikes,nbins);
                if size(thispsth,1) == 1
                    meanpsth(cellInd,tInd,repInd,:) = thispsth';
                elseif size(thispsth,2) == 1
                    meanpsth(cellInd,tInd,repInd,:) = thispsth;
                else
                    pause;
                end
            end
        end
    end
end





end
                
