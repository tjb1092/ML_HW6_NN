% Cleanup
clear all, clc, close all;


% Problem 1
[X, y] = dataGen(100);

Eta = logspace(-7,-3,5);
%Eta = 1e-5;
viewing = [5, 10, 50, 100];  % Epochs to plot decision surface.

pltstep = 0.01;
batchmode = true;
delta = true;
num_epochs = 10000;

for eta_i = 1:length(Eta)
    figure();
    pltcount = 1;
    w = rand([3,1])-0.5 *ones([3,1]);
    delW = zeros(size(w));

    for epoch = 1:num_epochs
        for i = 1:length(X)
            o(i) = dot(X(i,:),w);  % Delta rule is unthresholded dot product.
        end
        
        %o = FeedForward(X, w, batchmode, delta);
        %[E(epoch,eta_i), w, delW] = BackProp(w, delW, o, y, Eta(eta_i);
        
        E(epoch,eta_i) = (1/(2*length(X)))*sum((y-o).^2);  %Error E.
        delW = delW + Eta(eta_i) * ((y - o) * X)';
        w = w + delW;
        
        if ismember(epoch, viewing)
            %Problme 1b. Plot decision surface.

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
            map = [255/255, 185/255, 185/255; 61/255,77/255,220/255];
            colormap(map)
            s = gscatter(X(:,2),X(:,3),y,[],'o',4);
            title(['Eta: ' num2str(Eta(eta_i)) ' | Epoch: ' num2str(epoch)])
            hold off;
            pltcount = pltcount + 1;        
        end
    end
end

%Problem 1a. Plot E vs. Epoch for different learning rates.
figure();
hold on;
[~,A] = size(E);
for i = 1:A
    plot(1:num_epochs,E(:,i));
end
grid on;
xlabel('Epoch');
ylabel('E');
title('E per Epoch');
legend(num2str(Eta(1)),num2str(Eta(2)),num2str(Eta(3)),num2str(Eta(4)),num2str(Eta(5)))
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Problem 1d.
clc,clear all
[X, y] = dataGen(100);
Eta = logspace(-5,-1,5);
E_Thresh = 0.01;
num_epochs = 100;

figure();
hold on;
for eta_i = 1:length(Eta)
    timer = tic;
    update_cnt = 0;
    epoch = 0;
    first = 1;
    clear E;
    w = rand([3,1])-0.5 *ones([3,1]);
    
    while (first || E(update_cnt) > E_Thresh) && epoch <= num_epochs               
        first = 0;
        epoch = epoch + 1;
        
        for i = 1:length(X)
            update_cnt = update_cnt + 1;
            o = dot(X(i,:),w);  % Delta rule is unthresholded dot product.
            E(update_cnt) = 0.5*(y(i)-o).^2;  %Error E.
            if(E(update_cnt) < E_Thresh)
                break;
            end
            w = w + Eta(eta_i) * (y(i) - o) * X(i,:)';

        end
    end

    iterative_time(eta_i) = toc(timer);
    iter_num_updates(eta_i) = update_cnt;
    plot(1:update_cnt,E);
end

iterative_time
iter_num_updates
