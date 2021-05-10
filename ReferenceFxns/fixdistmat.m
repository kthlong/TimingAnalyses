function [distmat] = fixdistmat(distmat)


        for repInd = 1:size(distmat,3)
             distmat(:,:,repInd,repInd,:,:,:,:,:,:,:,:,:,:) = nan;
        end



end