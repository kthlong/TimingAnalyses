function N = Histc(X,EDGES)

N = histc(X,EDGES);
N = reshape( N(1:end-1), [],1);
if(isempty(N))
   N = zeros(length(EDGES)-1,1);
end