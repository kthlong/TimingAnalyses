

function newM = deplane(M)

[X Y] = meshgrid(1:size(M,2),1:size(M,1));

inp = [reshape(ones(size(M)), 1, []); reshape(X, 1, []); reshape(Y, 1, [])]';

p = mvregress(inp, reshape(M, [],1));

% myfun = @(params)(params(1) + ...
%                   params(2)*reshape(X, 1, []) + ...
%                   params(3)*reshape(Y, 1, []) - ...
%                   reshape(M, 1, []));
% 
% lsqnonlin(myfun, [1 1 1])

newM = M - reshape(p(1) + ...
            p(2)*reshape(X,[],1) + ...
            p(3).*reshape(Y, [], 1), size(M,1), size(M,2));
        
end