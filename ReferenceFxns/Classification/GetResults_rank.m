
function [results responses rank] = GetResults_rank(distmat)

[~,dataResponses] = sort(distmat,2); % for each row (test texture), rank guesses
responses = squeeze(dataResponses(:,1,:,:,:,:,:)); % keep only the top response
ignore = isnan(distmat); % flag responses to ignore (nans)
responses(squeeze(ignore(:,1,:,:))) = NaN; % get rid of those nans
nResp = size(responses);
expectedResponses = repmat([1:nResp(1)]',[1 nResp(2:end)]); % correct responses
results = (responses == expectedResponses); % find those that match
results = double(results); % convert from logicals
results(squeeze(ignore(:,1,:,:))) = NaN; % get rid of nans

correct = repmat([1:55]',1,55,39);
rank = [correct == dataResponses];
[~,ranking] = sort(rank,2,'descend');
ranking = squeeze(ranking(:,1,:,:,:,:));