%% Cleanup
clear all, clc, close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Problem 1
fprintf('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\nProblem 1a-c. Batch mode utilizing the delta training rule.\n\n');

[data.X_train, data.y_train] = dataGen(100);
[data.X_val, data.y_val] = dataGen(50);

figure,gscatter(data.X_train(:,2),data.X_train(:,3),data.y_train);
title('Generated Data');

Eta = logspace(-6,-2,5);
num_epochs = 100;
E_Thresh = 0;  % Forces it to run all 100 epochs.

%Problem Flags
flags.pltDS = 1;  % Want to view the decision surface as it trains.
flags.learnType = 0;
flags.isIncremental = 0;
flags.p3 = 0;

DeltaUnit(data, Eta, E_Thresh, 0, 0, num_epochs, flags);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Problem 1d.
fprintf('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\nProblem 1d. Comparing Batch and Incremental Update Modes.\n\n');

clear all;
[data.X_train, data.y_train] = dataGen(100);
[data.X_val, data.y_val] = dataGen(50);

E_Thresh = 0.0001;
num_epochs = 100;

%Stochastic Gradient Descent.
Eta = logspace(-5,-1,5); % Can run @ 0.1 without destabalizing.
flags.isIncremental = 1;
flags.learnType = 0;
flags.pltDS = 0;
flags.p3 = 0;

[ time, num_updates ] = DeltaUnit(data, Eta, E_Thresh,...
    0, 0, num_epochs, flags);

fprintf('\nIncremental Stochastic Gradient Descent:\n')
for i = 1:length(Eta)
    fprintf('Eta: %0.5f | Runtime (s): %0.4f | # Weight Updates: %i\n',Eta(i), time(i), num_updates(i));
end

clear flags;

% Batch Mode
flags.isIncremental = 0;
flags.learnType = 0;
flags.pltDS = 0;
flags.p3 = 0;
Eta = logspace(-5,-2,4); %Running it much higher than 0.01 can lead to unstable results.
[ time, num_updates ] = DeltaUnit( data, Eta, E_Thresh,...
    0, 0, num_epochs, flags);

fprintf('\nBatch Mode Gradient Descent:\n')
for i = 1:length(Eta)
    fprintf('Eta: %0.5f | Runtime (s): %0.4f | # Weight Updates: %i\n',Eta(i), time(i), num_updates(i));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Problem 2

%2a. Decaying Learning Rates.
fprintf('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\nProblem 2a. Decaying Learning Rates.\n\n');

clear all;

[data.X_train, data.y_train] = dataGen(100);
[data.X_val, data.y_val] = dataGen(50);

E_Thresh = 0.0001;
num_epochs = 100;

ISGD_flags.isIncremental = 0;
ISGD_flags.learnType = 1;
ISGD_flags.pltDS = 0;
ISGD_flags.p3 = 0;

BSGD_flags.isIncremental = 0;
BSGD_flags.learnType = 1;
BSGD_flags.pltDS = 0;
BSGD_flags.p3 = 0;

alpha = linspace(0.7,0.99,3);  %Starting decay rates.

for n = 1:length(alpha)
    Eta = logspace(-5,-2,4);
    [ time, num_updates ] = DeltaUnit( data, Eta, E_Thresh,...
        alpha(n),0, num_epochs, ISGD_flags);
    
    fprintf('\nIncremental Mode With Decaying Learning Rates:\n')
    for i = 1:length(Eta)
        fprintf('Eta: %0.5f | Runtime (s): %0.4f | # Weight Updates: %i\n',Eta(i), time(i), num_updates(i));
    end
    
    Eta = logspace(-5,-1,5);
    [ time, num_updates ] = DeltaUnit( data, Eta, E_Thresh,...
        alpha(n),0, num_epochs, BSGD_flags);
    
    fprintf('\nBatch Mode With Decaying Learning Rates:\n')
    for i = 1:length(Eta)
        fprintf('Eta: %0.5f | Runtime (s): %0.4f | # Weight Updates: %i\n',Eta(i), time(i), num_updates(i));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2b. Adaptive rates
fprintf('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\nProblem 2a. Decaying Learning Rates.\n\n');

clear all;
[data.X_train, data.y_train] = dataGen(100);
[data.X_val, data.y_val] = dataGen(50);

E_Thresh = 0.1;
num_epochs = 100;

% Flags & problem specific hyper-parameters.
flags.pltDS = 0;
flags.isIncremental = 0;
flags.learnType = 2;
flags.p3 = 0;

adapt.d = 0.9;
adapt.D = 1.02;
adapt.Thresh = 0.05;
Eta = logspace(-5,-2,4);
Eta = 1;
[ time, num_updates ] = DeltaUnit( data, Eta, E_Thresh,...
    0,adapt, num_epochs, flags );

fprintf('\nBatch Mode With Adaptive Learning Rates:\n')

fprintf('\nAdaptive Learning Parameters:\n')
fprintf('d: %0.1f | D: %0.2f | thresh: %0.2f\n',adapt.d, adapt.D, adapt.Thresh)
fprintf('\nAdaptive Learning Performance:\n')
for i = 1:length(Eta)
    fprintf('Eta: %0.5f | Runtime (s): %0.4f | # Weight Updates: %i\n',Eta(i), time(i), num_updates(i));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3 Quadratic Gradient Descent.
clear all;
fprintf('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\nProblem 3. Derived Quadratic Gradient Descent.\n');

% Generate sigmoid data from provided code snippet. Reshape into struct.
Data = gen_sigmoid_classes(100);
data.X_train = [ones(length(Data),1),Data(:,1:2)];
data.y_train = Data(:,3)';
Data = gen_sigmoid_classes(50);
data.X_val = [ones(length(Data),1),Data(:,1:2)];
data.y_val = Data(:,3)';
figure,gscatter(data.X_train(:,2),data.X_train(:,3),data.y_train);
title('Generated Sigmoid Data');

E_Thresh = 0.0001;
num_epochs = 100;

%Flags & problem specific hyper-parameters.
flags.pltDS = 0;
flags.isIncremental = 0;
flags.learnType = 0;
flags.p3 = 1;

Eta = logspace(-5,-2,4);
[ time, num_updates ] = DeltaUnit( data, Eta, E_Thresh,...
    0,0, num_epochs, flags);

fprintf('\nBatch Mode With Quadratic Gradient Descent:\n')
for i = 1:length(Eta)
    fprintf('Eta: %0.5f | Runtime (s): %0.4f | # Weight Updates: %i\n',Eta(i), time(i), num_updates(i));
end


