function dist=KLDiv(P,Q)
%  dist = KLDiv(P,Q) Kullback-Leibler divergence of two discrete probability
%  distributions
%  P and Q  are automatically normalised to have the sum of one on rows
% have the length of one at each 
% P =  n x nbins
% Q =  1 x nbins or n x nbins(one to one)
% dist = n x 1

P = P+eps;
Q = Q+eps;

if isnan(P) | isnan(Q)
    dist = nan;

elseif size(P,2)~=size(Q,2)
    error('the number of columns in P and Q should be the same');
    
elseif sum(~isfinite(P(:))) + sum(~isfinite(Q(:)))
   error('the inputs contain non-finite values!') 
   

% normalizing the P and Q
elseif size(Q,1)==1
    Q = Q ./sum(Q);
    P = P ./repmat(sum(P,2),[1 size(P,2)]);
    dist =  sum(P.*log2(P./repmat(Q,[size(P,1) 1])),2);
    
%     sum(p[i] * log2(p[i]/q[i])
    
elseif size(Q,1)==size(P,1)
    
    Q = Q ./repmat(sum(Q,2),[1 size(Q,2)]);
    P = P ./repmat(sum(P,2),[1 size(P,2)]);
    dist =  sum(P.*log2(P./Q),2);
end

% resolving the case when P(i)==0
dist(isnan(dist))=0;

