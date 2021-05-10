function columndata = makecolumn(data)

if size(data,1) == 1
    columndata = data';
elseif size(data,2) == 1
    columndata = data;
else
    error('more than 1D data!')
end
