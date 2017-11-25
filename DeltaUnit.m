function [ iterative_time, iter_num_updates ] = DeltaUnit( X_train, X_val, y_train, y_val, Eta,...
    E_Thresh, num_epochs, isIterative, pltDS)


viewing = [5, 10, 50, 100];  % Epochs to plot decision surface.
pltstep = 0.01;


for eta_i = 1:length(Eta)
    
    if(pltDS)
        figure();
        pltcount = 1;
    end
    w = rand([3,1])-0.5 *ones([3,1]); % I should be able to generalize the 3. 
    delW = zeros(size(w));
        timer = tic;
    update_cnt = 0;
    epoch = 0;
    first = 1;
    clear E;
    endFlg = 0;
    
    while first || (epoch < num_epochs && ~endFlg)
        
        epoch = epoch + 1;
        
        
        if isIterative
            for i = 1:length(X_train)
                update_cnt = update_cnt + 1;
                o = X_train(i,:)*w;  % Delta rule is unthresholded dot product.

                w = w + Eta(eta_i) * (y_train(i) - o) * X_train(i,:)';

                o_val = (X_val * w)';
                E(update_cnt) = (1/2)*sum((y_val-o_val).^2);  %Error E per update.
                if(~first && abs(E(update_cnt) - E(update_cnt-1)) < E_Thresh)
                  endFlg = 1;
                  break;
                else
                  if first
                    first = 0;
                  end
                end
            end  
        else     
            o = (X_train * w)';

            delW = Eta(eta_i) * ((y_train - o) * X_train)';
            w = w + delW;

            o_val = (X_val * w)';
            update_cnt = update_cnt + 1;
            E(update_cnt) = (1/(2*length(X_val)))*sum((y_val-o_val).^2);  %Error E per epoch update.
            
            if(~first && abs(E(update_cnt) - E(update_cnt-1)) < E_Thresh)
              endFlg = 1;
              break;
            else
              if first
                first = 0;
              end
            end
        end
        
        if ismember(epoch, viewing) && pltDS
            %Problem 1b. Plot decision surface.
            t = ['Eta: ' num2str(Eta(eta_i)) ' | Epoch: ' num2str(epoch)];
            pltcount = PltDecisionSurf( X_train, y_train, w, pltstep, pltcount, t );
        end
        
        err{eta_i} = E;
        iterative_time(eta_i) = toc(timer);
        iter_num_updates(eta_i) = update_cnt;
    end
end

%Problem 1a. Plot E vs. Epoch for different learning rates.
PltE(err, Eta);
end

