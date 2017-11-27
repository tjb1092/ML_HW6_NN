function [ iterative_time, iter_num_updates ] = DeltaUnit( data,...
    Eta, E_Thresh, alpha, adapt, num_epochs, flags)

viewing = [5, 10, 50, 100];  % Epochs to plot decision surface.

%For each learning rate to test,
for eta_i = 1:length(Eta)
    
    %If visualizing decision surface, make a plot at the beginning of the
    %run and populate it as the DS changes.
    if(flags.pltDS)
        figure();
        pltcount = 1;
    end
    
    w = rand([3,1])-0.5 *ones([3,1]); % Initialize to small values above and below zero.  
  
    timer = tic;  % Start timer for the learning rate.
    
    %Initialize & clear necessary variables for the training process.
    update_cnt = 0;
    epoch = 0;
    first = 1;
    clear E;
    endFlg = 0;
    clear eta;
    
    % While the end conditions are not met, continue training.
    while first || (epoch < num_epochs && ~endFlg)
        
        epoch = epoch + 1;  % Increment epoch #.
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute Learning Rate. 
        
        %Apply Eta change per epoch. 
        if flags.learnType == 0
        %Constant Learning Rate. Nothing.
        eta = Eta(eta_i);
        
        elseif flags.learnType == 1
            %Problem 2a. Decaying Learning Rate.
            eta(epoch) = alpha^epoch * Eta(eta_i);

        elseif flags.learnType == 2
            % Problem 2b. Adaptive Learning Rate.
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
                    eta(epoch) = eta(end) * adapt.d;    %Decrease the learning rate slightly
                elseif newE < prevAddE
                    eta(epoch) = eta(end) * adapt.D;    %Increase the learning rate slightly. Keep weights.                   
                else
                    % else don't change anything 
                    % (i.e. greater than prev, but less than thresh).
                    eta(epoch) = eta(end);
                end
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Problem 1d. Incremental Gradient Descent.
        if flags.isIncremental
            if flags.learnType == 2
                disp('Not meant for incremental approach'); 
                return
            end
            
            %Update weights for each data point.
            for i = 1:length(data.X_train)
                update_cnt = update_cnt + 1;
                %o = data.X_train(i,:)*w;  % Delta rule is unthresholded dot product.
                if flags.p3
                    %Problem 3. Forward propagation of quadradic gradient
                    %descent. The bias term was removed from the addition &
                    %squaring operations, and was thus prepended to X. 
                    o = ([data.X_train(i,1), (data.X_train(i,2:3) + data.X_train(i,2:3).^2)]*w)';
                else
                    o = (data.X_train(i,:) * w)'; % Forward Propagation
                end
            
                
                
                w = BackProp(data.X_train(i,:), data.y_train(i), w, o, eta(end), flags.p3);

                o_val = (data.X_val * w)';
                E(update_cnt) = (1/(2*length(data.X_val)))*sum((data.y_val-o_val).^2);  %Error E per update.
                
                %End condition. Change in Error is smaller than threshold.
                if(~first && abs(E(update_cnt) - E(update_cnt-1)) < E_Thresh)
                  %This wasn't nested in a function to ensure the break works
                  %correctly.
                  endFlg = 1;
                  break;
                else
                  %flip flag
                  if first
                    first = 0;
                  end
                end
            end  
        else 
            
            if flags.p3
                %Problem 3. Forward propagation of quadradic gradient
                %descent. The bias term was removed from the addition &
                %squaring operations, and was thus prepended to X. 
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
            t = ['\eta : ' num2str(Eta(eta_i)) ' | Epoch: ' num2str(epoch)];
            pltcount = PltDecisionSurf( data.X_train, data.y_train, w, pltcount, t );
        end
        
        err{eta_i} = E; %Append error for the training. Using a cell array b/c the lengths may not be equal.
        iterative_time(eta_i) = toc(timer); %Store duration of training.
        iter_num_updates(eta_i) = update_cnt; % Store number of weight updates.

    end
    
    if flags.learnType == 2 || flags.learnType == 1
        %Plot eta to see how it adapts during training.        
        figure, plot(1:epoch,eta); 
        if flags.learnType == 1
            L_type = 'Decaying ';
        else
            L_type = 'Adaptive ';
        end
        
        if flags.isIncremental
            Mode = 'Incremental Mode: ';
        else
            Mode = 'Batch Mode: ';
        end
        
        title(strcat(Mode,' ',L_type,' Learning Rate. \eta start: ',num2str(Eta(eta_i))));
        xlabel('Epoch');
        ylabel('\eta');
    end
    
end

%Problem 1a. Plot E vs. Epoch for different learning rates.
PltE(err, Eta, flags);
end

