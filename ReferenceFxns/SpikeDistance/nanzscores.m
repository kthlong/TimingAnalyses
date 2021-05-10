function zscores = nanzscores(X,dim)

if nargin < 2
    dim = 1;
end

xmu     =   nanmean(X,dim);
xsigma  =   nanstd(X,0,dim);
zscores =   bsxfun(@minus,X,xmu);
zscores = bsxfun(@rdivide,zscores,xsigma);
