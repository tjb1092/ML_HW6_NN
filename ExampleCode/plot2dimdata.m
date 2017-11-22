function plot2dimdata(D,T, fp, fn)
% INVOKE AS: plot(D, fp, fn)where fp/fn are of the type 'r/g/b/y/m/k'
% plot a two dimensional data according to their labels
% D(i,:): coordinates of points to be plotted
% T(i) = +1 or -1
% +1 points are plotted with format fp
% -1 points are plotted with format fn

%%% insert code to check that data is indeed as supposed to be: 
%%  a matrix of dimensions N x 2 for N data points
%% T a vector of length N with values +1/-1
[rd, cd]=size(D);
% figure; hold on;
posindex=find(T==1);
negindex=find(T==-1);
plot(D(posindex,1), D(posindex,2), fp);
plot(D(negindex,1), D(negindex,2), fn);
end
