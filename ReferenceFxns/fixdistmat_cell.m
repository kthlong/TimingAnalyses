function [distmat] = fixdistmat_cell(distmat)


        for repInd = 1:size(distmat,3)
             distmat(:,:,repInd,repInd,:,:,:,:,:) = {nan};
        end



end