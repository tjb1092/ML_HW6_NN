function [ pltcount ] = PltDecisionSurf( X, y, w, pltcount, t )
%Problem 1b. Plot decision surface.

%Create a mesh spanning all of the data points.
pltstep = 0.01;
X1plt = [min(X(:,2)):pltstep:max(X(:,2))];
X2plt = [min(X(:,3)):pltstep:max(X(:,3))];
[Xplt,Yplt] = meshgrid(X1plt,X2plt);
[N, M] = size(Xplt);

%Compute the decision surface using a thresholded feedforward propagation.
for n = 1:N
    for m = 1:M
       Z(n,m) = sign([1,Xplt(n,m),Yplt(n,m)]*w);
    end
end

%Plot the decision surface as a filled contour plot.
subplot(4,1,pltcount);
hold on;
contourf(Xplt,Yplt,Z);

%Change colors to a red/blue color.
map = [255,185,185; 61,77,220];
map = map/255;
colormap(map)

%Plot the labeled data on top of the contour plot.
gscatter(X(:,2),X(:,3),y,[],'o',4);
title(t)
xlabel('X_1');
ylabel('X_2');
hold off;
pltcount = pltcount + 1;
end

