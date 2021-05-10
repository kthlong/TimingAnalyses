function zscores = nanzscores_keepdim(X,keepdim)

if nargin < 2
    keepdim = 1;
end

ndims = length(size(X));
alldims = 1:ndims;
nobs = size(X,keepdim);
X_shaped = permute(X,[keepdim alldims(alldims~=keepdim)]);

for obsInd = 1:nobs
    obsmat = X_shaped(obsInd,:,:,:,:,:,:);
    obsmat = obsmat(:);
    xmu(obsInd) = nanmean(obsmat);
    xsigma(obsInd) = nanstd(obsmat);
end
    
zscores =   bsxfun(@minus,X,xmu');
% zscores = X;
zscores = bsxfun(@rdivide,zscores,xsigma');
