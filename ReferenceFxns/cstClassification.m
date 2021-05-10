function distmat = cstClassification(data,distancemetric,varDim,repDim,minT,maxT,qval)

nCategories = size(data,varDim);
nReps = size(data,repDim);
allotherdims = 1:length(size(data));  allotherdims = allotherdims(allotherdims~= varDim & allotherdims~= repDim);
trainMatrix(1,1,:,:,:,:,:,:) = permute(data,[varDim repDim allotherdims]);
manualtoggle = 0;

if manualtoggle == 0
trainMatrix = repmat(trainMatrix,nCategories,nReps,1,1,1,1,1,1,1);
trainMatrix = permute(trainMatrix,[3 1 4 2 5:length(size(trainMatrix))]);
testMatrix = permute(trainMatrix,[2 1 4 3 5:length(size(trainMatrix))]);
else 
trainMatrix = repmat(trainMatrix,1,nReps,1,1,1,1,1,1,1);
trainMatrix = permute(trainMatrix,[3 1 4 2 5:length(size(trainMatrix))]);
testMatrix = permute(trainMatrix,[1 2 4 3 5:length(size(trainMatrix))]);
end

% distmat: testCat x trainCat x testRep x trainRep x all others


%% Spike Distance: if input is spike times
if strcmp(distancemetric,'SpikeDist')
    if ~exist('qval')
        qval = .002;
        qmat(1,1,1,1,1,1,:) = num2cell(1./[.001 .002 .005 .01 .02 .05 .1 .2 Inf]);
    else
        qmat(1,1,1,1,1,1,:) = num2cell(1./qval);
    end
    testMatrix = fixdistmat_cell(repmat(testMatrix,1,1,1,1,1,1,length(qmat)));
    trainMatrix = fixdistmat_cell(repmat(trainMatrix,1,1,1,1,1,1,length(qmat)));
    qmat = fixdistmat_cell(repmat(qmat,size(testMatrix,1:6)));
    distmat = cellfun(@(x,y,q) spikeDist_align(x,y,q,minT,maxT),testMatrix,trainMatrix,qmat);


%% Rate Abs Value: if input is rates in a double mat

elseif strcmp(distancemetric,'AbsoluteValue') & ~iscell(data)
    distmat = bsxfun(@(x,y) absdist(x,y), testMatrix, trainMatrix); % RATE

%% Any other input
elseif strcmp(distancemetric,'EucDist')
    distmat = cellfun(@(x,y) eucdist(x,y), testMatrix, trainMatrix); % symmetric
    
elseif strcmp(distancemetric,'DKL')
    distmat = cellfun(@(x,y) KLDiv(x,y), trainMatrix, testMatrix); % distance of test from train
    
elseif strcmp(distancemetric,'DJS')
    distmat = cellfun(@(x,y) JSDiv(x,y), trainMatrix, testMatrix); % symmetric
    
elseif strcmp(distancemetric,'AbsoluteValue') & iscell(data)
    distmat = cellfun(@(x,y) absdist(x,y), testMatrix, trainMatrix); % symmetric
    
elseif strcmp(distancemetric,'XCov')
    distmat = cellfun(@(x,y) maxxcov(x,y), trainMatrix,testMatrix); % symmetric
else
    distmat = nan;
end





%% DISTANCE VALUES--- -----------------------------------------------------
% Euclidian Dist
function dist = eucdist(x,y)
    if isnan(x) | isnan(y)
        dist = nan;
    else
    dist = sqrt(sum((x-y).^2));
    end
end

% KLDiv
function dist=KLDiv(P,Q)
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

    elseif size(Q,1)==size(P,1)

        Q = Q ./repmat(sum(Q,2),[1 size(Q,2)]);
        P = P ./repmat(sum(P,2),[1 size(P,2)]);
        dist =  sum(P.*log2(P./Q),2);
    end

    % resolving the case when P(i)==0
    dist(isnan(dist))=0;
end

% JSDiv
function dist =JSDiv(P,Q)
    P = P+eps;
    Q = Q+eps;

    if isnan(P) | isnan(Q)
        dist = nan;
    elseif size(P,2)~=size(Q,2)
        error('the number of columns in P and Q should be the same');
    else

        % normalizing the P and Q
        Q = Q ./sum(Q);
        Q = repmat(Q,[size(P,1) 1]);
        P = P ./repmat(sum(P,2),[1 size(P,2)]);

        M = 0.5.*(P + Q);

        dist = 0.5.*KLDiv(P,M) + 0.5*KLDiv(Q,M);
    end
end

% AbsVal Div
function dist = absdist(x,y)
    dist = abs(x-y);
end

% Max XCov
function dist = maxxcov(x,y)
    if isnan(x) | isnan(y)
        dist = nan;
    else
    dist = max(xcov(x,y));
    end
end


end
