function [misclass_names, misclass] = errornames(responses,texture_names)

correctresponses = repmat(1:59,5,1)';
misclassified_ind = find(responses ~= correctresponses);
misclass = num2cell([responses(misclassified_ind),correctresponses(misclassified_ind)]); % column 1 = guessed, column 2 = correct
misclass_names = cellfun(@(x) texture_names(x), misclass);