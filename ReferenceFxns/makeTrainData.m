function trainSpikes = makeTrainData(spikedata,varDim,repDim)

nCategories = size(data,varDim);
nReps = size(data,repDim);
allotherdims = 1:length(size(data));  allotherdims = allotherdims(allotherdims~= varDim & allotherdims~= repDim);
nDims = length(allotherdims)+2;
trainMatrix_premean(1,:,:,:,:,:,:) = permute(data,[varDim repDim allotherdims]);
trainMat = cell2mat(trainMatrix_premean);


for repInd = 1:nReps
    repMask = ones(nReps,1);
    repMask(repInd) = nan;
    repMask = repMask == 1;
    reps = find(repMask);
    trainSpikes(repInd,:,:,:) = 


trainMatrix_premean(1,:,:,:,:,:,:) = permute(data,[varDim repDim allotherdims]);
trainMatrix_premean = repmat(trainMatrix_premean,nCategories,1,1,1,1,1,1,1);
trainMatrix_premean = permute(trainMatrix_premean,[6 1 2 3 4 5]);
trainMatrix_premean = cell2mat(trainMatrix_premean);

for repInd = 1:nReps
    repMask = ones(nReps,1);
    repMask(repInd) = nan;
    repMask = repMask == 1;
    reps = find(repMask);
     
    rep1 = trainMatrix_premean(:,:,reps(1),:,:,:,:,:);
    rep2 = trainMatrix_premean(:,:,reps(2),:,:,:,:,:);
    rep3 = trainMatrix_premean(:,:,reps(3),:,:,:,:,:);
    rep4 = trainMatrix_premean(:,:,reps(4),:,:,:,:,:);
    trainMatrix(:,:,repInd,:,:,:,:,:,:) = cellfun(@(r1,r2,r3,r4) nanmean([r1,r2,r3,r4],2), rep1,rep2,rep3,rep4,'uniformoutput',false);

trainMatrix = repmat(trainMatrix,nCategories,nReps,1,1,1,1,1,1,1);
trainMatrix = permute(trainMatrix,[3 1 4 2 5:length(size(trainMatrix))]);
testMatrix = permute(trainMatrix,[2 1 4 3 5:length(size(trainMatrix))]);
