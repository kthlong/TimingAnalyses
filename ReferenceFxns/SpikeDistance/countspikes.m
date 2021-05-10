function nspikes = countspikes(hist1,hist2)

if size(hist1) ~= size(hist2)
    hist1 = hist1';
end
if size(hist1) ~= size(hist2)
    nspikes = nan;
end

nspikes = sum(bsxfun(@min,hist1,hist2));