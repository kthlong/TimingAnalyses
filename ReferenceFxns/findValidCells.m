function [emptymask,cells2keep] = findValidCells(data,cellDim,threshold)

if nargin < 3
    threshold = 1;
end

emptymask = cellfun(@(x) ~(isempty(x)),data);

cells2keep = [];
    for cellInd = 1:size(data,cellDim)
        if sum(emptymask(cellInd,:,:,:)) >= threshold
            cells2keep = [cells2keep cellInd];
        end
    end