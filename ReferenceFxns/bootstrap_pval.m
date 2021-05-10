function pval = bootstrap_pval(datapoints,distribution)

distribution = distribution(~isnan(distribution));
distributionflat = sort(distribution(:));
pval = [];
for dataInd = 1:length(datapoints)
    datapoint = datapoints(dataInd);
    if isnan(datapoint)
        pval(dataInd) =  nan;
    else
nabove = length(find(distributionflat < datapoint)) + length(find(distributionflat == datapoint))/2;
ntotal = length(distributionflat);

pval(dataInd) = nabove/ntotal;
    end
end