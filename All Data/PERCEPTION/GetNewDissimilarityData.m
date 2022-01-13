function dissimData = GetNewDissimilarityData()


roughData = GetRoughData();
newRoughData = GetNewRoughData();


subjectNotesOnline = ...
    'https://docs.google.com/spreadsheets/d/1T3-IxjhNBGARjuO9vNvYWTngH-nEhgi43OE1gM_xyB0/edit#gid=577847773';


filepath = 'C:\somlab\justin_hub\CorticalTexture\roughness\NewDissimilarityFiles';
file = 'TextureDataBase.csv';

formatString        = '%s%s%s%s%s%s%s%s';

fid                 = fopen([filepath '\' file], 'rt');
headers             = textscan(fid, formatString, 1, 'Delimiter',',');
data                = textscan(fid, formatString, 'Delimiter',',');
fclose(fid);

textureDatabase     = data;

allGoodTextInd      = GetAllGoodTextInd();



databaseTextInd     = cellfun(@str2num, textureDatabase{1});
databaseTextNames   = textureDatabase{3};


filepath = 'C:\somlab\justin_hub\CorticalTexture\roughness\NewDissimilarityFiles';
formatStringHeader  = '%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s';
formatString        = '%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d';
    
files = ls([filepath '\*rsp*']);

N = size(files,1);
fileList = cell(N,1);

nText       = 60;
nRep        = 6;
allDissim   = nan(N,nText,nText,nRep);
smallDissim = cell(N,1);

allNames    = cell(N,1);
allTList    = cell(N,1);
allTNames   = cell(N,1);

for i=1:N
    file = strtrim(files(i,:));
    fileList{i} = file;
    
    allNames{i} = file([9 10]);
    
    fileList{i};
    
    fid                 = fopen([filepath '\' file], 'rt');
    headers             = textscan(fid, formatStringHeader, 1);
    data                = textscan(fid, formatString);
    fclose(fid);
    
    posInd  = find(cellfun(@(x)strcmp(x, 'PositionOnDrum'), headers));
    tList   = [data{posInd}];
    [uniqueTList, uniqueInd] = unique(tList(:));
    allTList{i}     = uniqueTList;
    
    posInd          = find(cellfun(@(x)strcmp(x, 'PatternName'), headers));
    namesIndList    = [data{posInd}];
    namesIndList    = namesIndList(uniqueInd);
    
    nList           = length(namesIndList);
    
    stringList = cell(nList,1);
    for j = 1:length(namesIndList)
        ind = (databaseTextInd == namesIndList(j));
        stringList{j} = databaseTextNames{ind};
    end
    
    allTNames{i} = stringList;

    dissimInd   = find(cellfun(@(x)strcmp(x, 'Response'), headers));
    thisDissim  = data{dissimInd};
    repInd      = find(cellfun(@(x)strcmp(x, 'RepNum'), headers));
    thisRep     = data{repInd};
    for j = 1:size(thisDissim,1)
        tInds = sort(tList(j,:));
        while ~isnan(allDissim(i, tInds(1), tInds(2), thisRep(j)))
            thisRep(j) = thisRep(j)+1;
        end
        allDissim(i, tInds(1), tInds(2), thisRep(j)) = thisDissim(j);
    end
    smallDissim{i} = squeeze( allDissim(i, uniqueTList, uniqueTList, :) );
end


subjNameList    = unique(allNames);
nSubj           = length(subjNameList);
allDissimData   = cell(2,nSubj);

for sInd = 1:nSubj
    name = subjNameList{sInd};
    ind = find(strcmp(allNames,name));
    
    
    bigMat = [];
    for fInd = 1:3
        for repInd = 1:3
            thisMat = smallDissim{ind(1+fInd)}(:,:,repInd);
            if 1 == sum(isnan(thisMat(:))) / length(thisMat(:))
                continue;
            end
            bigMat = cat(3, bigMat, thisMat);
        end
    end
    allDissimData{1,sInd} = bigMat;
    
    
    allDissimData{2,sInd} = smallDissim{ind(5)};
end

fullDissimMat = [];
smallDissimMat = [];
for i=1:size(allDissimData,2)
    thisMat = allDissimData{1,i};
    thisMat = thisMat ./ nanmean(thisMat(:));
    fullDissimMat = cat(4,fullDissimMat,thisMat);
    
    thisMat = allDissimData{2,i}(:,:,1:3);
    thisMat = thisMat ./ nanmean(thisMat(:));
    smallDissimMat = cat(4,smallDissimMat,thisMat);
end

fullDissimMat   = permute(fullDissimMat, [1 2 4 3]);
smallDissimMat  = permute(smallDissimMat, [1 2 4 3]);

meanDissimMat       = nanmean(nanmean(fullDissimMat,4),3);
meanSmallDissimMat  = nanmean(nanmean(smallDissimMat,4),3);
N = size(meanDissimMat,1);
for i=1:N
    for j=(i+1):N
        meanDissimMat(j,i) = meanDissimMat(i,j);
    end
end

N = size(meanSmallDissimMat,1);
for i=1:N
    for j=(i+1):N
        meanSmallDissimMat(j,i) = meanSmallDissimMat(i,j);
    end
end

textureNames = newRoughData.textureNames;
newTextureNames = textureNames;

newTextureNames{2} = 'Flag Banner';
newTextureNames{21} = 'Suede Cuddle (suede side)';
newTextureNames{26} = 'Denim Stretch';
newTextureNames{27} = 'Silver Satin';
newTextureNames{29} = 'Metallic Silk (Silver)';
newTextureNames{36} = '#1 Orange/Red Upholstery';
newTextureNames{38} = 'Corduroy (Black/thick stripe)';
newTextureNames{51} = 'Corduroy (Black/thin stripe)';
newTextureNames{52} = 'Embossed Dots 4mm';
newTextureNames{53} = 'Embossed Dots 5mm';

nNames = cellfun(@length, allTNames);

nameList = allTNames{ find(nNames == 13, 1) };
tcList = [];
for i=1:length(nameList)
    ind = find( strcmp(newTextureNames, nameList{i}) );
    tcList(i) = ind;
end

tcList;

tpList = [];

for i=1:length(tcList)
    tpList(i) = allGoodTextInd(1,allGoodTextInd(2,:) == tcList(i));
end



nameList = allTNames{ find(nNames == 5, 1) };
smallTcList = [];
for i=1:length(nameList)
    ind = find( strcmp(newTextureNames, nameList{i}) );
    smallTcList(i) = ind;
end

smallTpList = [];

for i=1:length(smallTcList)
    smallTpList(i) = allGoodTextInd(1,allGoodTextInd(2,:) == smallTcList(i));
end



dissimData              = [];
dissimData.textInd      = tcList;
dissimData.pTextInd     = tpList;
dissimData.textureNames = nameList';
dissimData.subNames     = subjNameList;
dissimData.mat          = meanDissimMat;
dissimData.fullMat      = fullDissimMat;

dissimData.smallTList   = smallTcList;
dissimData.smallPTList  = smallTpList;
dissimData.smallMat     = meanSmallDissimMat;
dissimData.smallFullMat = smallDissimMat;



