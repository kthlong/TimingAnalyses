function [newspiketimes,starttimes] = scramblespiketimes(data,minT,maxT_mod,windowlength)

starttimes = reshape(datasample([minT:.0001:(maxT_mod-windowlength)],prod(size(data))),size(data,1),size(data,2),size(data,3),size(data,4));

newspiketimes = cellfun(@(x,y) x(x>=y & x<=(y+windowlength))-y+minT,data,num2cell(starttimes),'uniformoutput',0);