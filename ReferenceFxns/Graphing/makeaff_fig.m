function makeaff_fig(QCdata,cells2include,max)
% Include QCdata, cells2include if desired
% Makes a plot with filled backgrounds for PCs then RAs then SA1s.

load('periphData.mat');

if nargin > 2
    cells = QCdata(cells2include);
elseif nargin > 1
    cells = QCdata(cells2include);
    max=1;
elseif nargin == 1
    cells = QCdata;
else
    cells = 1:39;
end
    
[~, afforder] = sort(periphData.type(cells));
firstRA = find(strcmp(periphData.type(cells(afforder)),'RA'),1,'first')-.5;
firstSA = find(strcmp(periphData.type(cells(afforder)),'SA1'),1,'first')-.5;
lastSA = find(strcmp(periphData.type(cells(afforder)),'SA1'),1,'last')+.5;

colors = parula(3);
linecolors = colors*.8;


figure()
h = patch([.5 firstRA firstRA .5],[0 0 max max],colors(1,:));
h.FaceAlpha = .2;
h.LineStyle = 'none';
h = patch([firstRA  firstSA firstSA firstRA],[0 0 max max],colors(2,:));
h.FaceAlpha = .2;
h.LineStyle = 'none';
h = patch([firstSA lastSA lastSA firstSA],[0 0 max max],colors(3,:));
h.FaceAlpha = .2;
h.LineStyle = 'none';