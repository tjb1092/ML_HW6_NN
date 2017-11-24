% Cleanup
clear all, clc, close all;

% Problem 1

[X_train, y_train] = dataGen(100);
[X_val, y_val] = dataGen(50);
figure,gscatter(X_train(:,2),X_train(:,3),y_train);
title('Generated Data');

Eta = logspace(-6,-2,5);
viewing = [5, 10, 50, 100];  % Epochs to plot decision surface.

pltflg = 1;
pltstep = 0.01;
num_epochs = 100;

for eta_i = 1:length(Eta)
    if(pltflg)
        figure();
        pltcount = 1;
    end
    w = rand([3,1])-0.5 *ones([3,1]);
    delW = zeros(size(w));

    for epoch = 1:num_epochs

        o = (X_train * w)';
        
        delW = Eta(eta_i) * ((y_train - o) * X_train)';
        w = w + delW;
        
        o_val = (X_val * w)';
        E(epoch,eta_i) = (1/(2*length(X_val)))*sum((y_val-o_val).^2);  %Error E.
        
        if ismember(epoch, viewing)
            %Problem 1b. Plot decision surface.
           if(pltflg)
               X1plt = [min(X_train(:,2)):pltstep:max(X_train(:,2))];
               X2plt = [min(X_train(:,3)):pltstep:max(X_train(:,3))];
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
               s = gscatter(X_train(:,2),X_train(:,3),y_train,[],'o',4);
               title(['Eta: ' num2str(Eta(eta_i)) ' | Epoch: ' num2str(epoch)])
               hold off;
               pltcount = pltcount + 1;
           end
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
%% Problem 1d.
clc,clear all
[X_train, y_train] = dataGen(100);
[X_val, y_val] = dataGen(50);

Eta = logspace(-4,-1,4);
E_Thresh = 0.0001;
num_epochs = 100;


for eta_i = 1:length(Eta)
    timer = tic;
    update_cnt = 0;
    epoch = 0;
    first = 1;
    clear E;
    w = rand([3,1])-0.5 *ones([3,1]);
    endFlg = 0;
    while first || (epoch < num_epochs && ~endFlg)            
        
        epoch = epoch + 1;
        
        for i = 1:length(X_train)
            update_cnt = update_cnt + 1;
            o = X_train(i,:)*w;  % Delta rule is unthresholded dot product.
            
            w = w + Eta(eta_i) * (y_train(i) - o) * X_train(i,:)';

            o_val = (X_val * w)';
            E(update_cnt) = (1/2)*sum((y_val-o_val).^2);  %Error E.
            if(~first && abs(E(update_cnt) - E(update_cnt-1)) < E_Thresh)
              endFlg = 1;
              break;
            else
              if first
                first = 0;
              end
            end
        end      
    end
    err{eta_i} = E;
    iterative_time(eta_i) = toc(timer);
    iter_num_updates(eta_i) = update_cnt;
end

figure();
hold on;
[~,A] = size(err);
m = 0;
for i = 1:A
    %find max length.
    if max(length(err{1,i})) > m
      m = length(err{1,i});
    end
end

for i = 1:A
    x_plt = 1:length(err{1,i});
    n = m-length(x_plt);  % how many short are we?
    x_plt = [x_plt,nan(1,n)];         % add that many NaNs
    y_plt = [err{1,i}, nan(1,n)];
    plot(x_plt, y_plt);
end
grid on;
xlabel('Epoch');
ylabel('E');
title('E per Epoch');
legend(num2str(Eta(1)),num2str(Eta(2)),num2str(Eta(3)),num2str(Eta(4)))
hold off;


iterative_time
iter_num_updates


%% Now try on batch mode.

for eta_i = 1:length(Eta)
    timer = tic;
    update_cnt = 0;
    epoch = 0;
    first = 1;
    clear E;
    w = rand([3,1])-0.5 *ones([3,1]);
    delW = zeros(size(w));
    endFlg = 0;
    while first || (epoch < num_epochs && ~endFlg)            
        
        epoch = epoch + 1;

            o = (X_train*w)';  % Delta rule is unthresholded dot product.
            
            w = w + Eta(eta_i) * (y_train(i) - o) * X_train(i,:)';

            o_val = (X_val * w)';
            E(update_cnt) = (1/2)*sum((y_val-o_val).^2);  %Error E.
            if(~first && abs(E(update_cnt) - E(update_cnt-1)) < E_Thresh)
              endFlg = 1;
              break;
            else
              if first
                first = 0;
              end
            end
        end      
    end
    err{eta_i} = E;
    iterative_time(eta_i) = toc(timer);
    iter_num_updates(eta_i) = update_cnt;
end

figure();
hold on;
[~,A] = size(err);
m = 0;
for i = 1:A
    %find max length.
    if max(length(err{1,i})) > m
      m = length(err{1,i});
    end
end

for i = 1:A
    x_plt = 1:length(err{1,i});
    n = m-length(x_plt);  % how many short are we?
    x_plt = [x_plt,nan(1,n)];         % add that many NaNs
    y_plt = [err{1,i}, nan(1,n)];
    plot(x_plt, y_plt);
end
grid on;
xlabel('Epoch');
ylabel('E');
title('E per Epoch');
legend(num2str(Eta(1)),num2str(Eta(2)),num2str(Eta(3)),num2str(Eta(4)))
hold off;


iterative_time
iter_num_updates
