function [ pltcount ] = PltDecisionSurf( X, y, w, pltstep, pltcount, t )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%Problem 1b. Plot decision surface.
X1plt = [min(X(:,2)):pltstep:max(X(:,2))];
X2plt = [min(X(:,3)):pltstep:max(X(:,3))];
[Xplt,Yplt] = meshgrid(X1plt,X2plt);
[N, M] = size(Xplt);
for n = 1:N
for m = 1:M
   Z(n,m) = sign(dot([1,Xplt(n,m),Yplt(n,m)],w));
end
end


subplot(4,1,pltcount);
hold on;
contourf(Xplt,Yplt,Z);
map = [255,185,185; 61,77,220];
map = map/255;
colormap(map)
s = gscatter(X(:,2),X(:,3),y,[],'o',4);
title(t)
hold off;
pltcount = pltcount + 1;

end

