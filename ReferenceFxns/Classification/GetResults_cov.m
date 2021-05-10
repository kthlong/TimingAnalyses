
function [results responses] = GetResults_cov(distmat)

[~,dataResponses] = sort(distmat,2,'descend'); % for each row (test texture), rank guesses (MAX COVARIANCE)
responses = squeeze(dataResponses(:,1,:,:,:,:,:,:,:)); % keep only the top response
ignore = isnan(distmat); % flag responses to ignore (nans)
responses(squeeze(ignore(:,1,:,:))) = NaN; % get rid of those nans
nResp = size(responses);
expectedResponses = repmat([1:nResp(1)]',[1 nResp(2:end)]); % correct responses
results = (responses == expectedResponses); % find those that match
results = double(results); % convert from logicals
results(squeeze(ignore(:,1,:,:))) = NaN; % get rid of nans
