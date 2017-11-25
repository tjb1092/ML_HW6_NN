function [ iterative_time, iter_num_updates ] = DeltaUnit( data,...
    Eta, E_Thresh, alpha, adapt, num_epochs, flags)

viewing = [5, 10, 50, 100];  % Epochs to plot decision surface.
pltstep = 0.01;

for eta_i = 1:length(Eta)
    
    if(flags.pltDS)
        figure();
        pltcount = 1;
    end
    
    w = rand([3,1])-0.5 *ones([3,1]); % I should be able to generalize the 3. 
  
    timer = tic;
    update_cnt = 0;
    epoch = 0;
    first = 1;
    clear E;
    endFlg = 0;
   
    while first || (epoch < num_epochs && ~endFlg)
        
        epoch = epoch + 1;
        
        %Apply Eta change per epoch. 
        if flags.learnType == 0
            %Constant Learning Rate. Nothing.
            eta = Eta(eta_i);                  
        elseif flags.learnType == 1
            %Decaying Learning Rate.
            eta = alpha^epoch * Eta(eta_i);

        elseif flags.learnType == 2
            % Adaptive Learning Rate.
            o_val = (data.X_val * w)';
            if first
                prevW = w;
                prevAddE = (1/(2*length(data.X_val)))*sum((data.y_val-o_val).^2);  %Error E per epoch update.
                eta = Eta(eta_i);
            else
                newE = (1/(2*length(data.X_val)))*sum((data.y_val-o_val).^2);
                relChange = abs(prevAddE-newE)/prevAddE;  % Compute Relative change.
                if newE > prevAddE && relChange > adapt.Thresh
                    w = prevW; %Discard the new w.
                    eta(epoch) = eta(end) * adapt.d;
                elseif newE < prevAddE
                    eta(epoch) = eta(end) * adapt.D;                   
                else
                    % else don't change anything 
                    % (i.e. greater than prev, but less than thresh).
                    eta(epoch) = eta(end);
                end
            end
        end
        
        
        if flags.isIncremental
            if flags.learnType == 2
                disp('Not meant for incremental approach'); 
                return
            end
            
            for i = 1:length(data.X_train)
                update_cnt = update_cnt + 1;
                o = data.X_train(i,:)*w;  % Delta rule is unthresholded dot product.
                
            w = BackProp(data.X_train(i,:), data.y_train(i), w, o, eta, flags.p3);

            o_val = (data.X_val * w)';
            E(update_cnt) = (1/(2*length(data.X_val)))*sum((data.y_val-o_val).^2);  %Error E per update.
            
            if(~first && abs(E(update_cnt) - E(update_cnt-1)) < E_Thresh)
              %This wasn't nested in a function to ensure the break works
              %correctly.
              endFlg = 1;
              break;
            else
              if first
                first = 0;
              end
            end
            end  
        else 
            
            if flags.p3
                o = ([data.X_train(:,1), (data.X_train(:,2:3) + data.X_train(:,2:3).^2)]*w)';
            else
                o = (data.X_train * w)'; % Forward Propagation
            end
            
            %Indexing eta(end) is for adaptive rates where
            %We want to see how the eta adapts over time.         
            [w] = BackProp(data.X_train, data.y_train, w, o, eta(end), flags.p3);  
            o_val = (data.X_val * w)';
            update_cnt = update_cnt + 1;
            E(update_cnt) = (1/(2*length(data.X_val)))*sum((data.y_val-o_val).^2);  %Error E per epoch update.
            
            if(~first && abs(E(update_cnt) - E(update_cnt-1)) < E_Thresh)
              %This wasn't nested in a function to ensure the break works
              %correctly.
              endFlg = 1;
              break;
            else
              if first
                first = 0;
              end
            end
        end
        
        if ismember(epoch, viewing) && flags.pltDS
            %Problem 1b. Plot decision surface.
            t = ['Eta: ' num2str(Eta(eta_i)) ' | Epoch: ' num2str(epoch)];
            pltcount = PltDecisionSurf( data.X_train, data.y_train, w, pltstep, pltcount, t );
        end
        
        err{eta_i} = E;
        iterative_time(eta_i) = toc(timer);
        iter_num_updates(eta_i) = update_cnt;

    end
    
    if flags.learnType == 2
        figure, plot(1:epoch,eta);
        title('Adaptive eta');
    end
    
end

%Problem 1a. Plot E vs. Epoch for different learning rates.
PltE(err, Eta);
end

