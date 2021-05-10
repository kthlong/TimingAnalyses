function plotprofilom(tID)

figure;
N = 4;
path = '\\bsl-somsrv1\data\raw\per\profilometry\';
filename = strcat(path, tID);

tempProf = csvread(filename, 16, 2);
xRange = 1:N:size(tempProf,2);
yRange = 1:N:size(tempProf,1);
profIm = deplane(tempProf(yRange, xRange));
newmap = contrast(profIm);
colormap(newmap);
imagesc(profIm);
axis off;
