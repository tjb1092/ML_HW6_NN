% Cleanup
clear all, clc, close all;

% Problem 1

[X_train, y_train] = dataGen(100);
[X_val, y_val] = dataGen(50);

figure,gscatter(X_train(:,2),X_train(:,3),y_train);
title('Generated Data');

Eta = logspace(-6,-2,5);
isIncremental = 0;
num_epochs = 100;
E_Thresh = 0;  % Forces it to run all 100 epochs.
learnType = 0;
pltDS = 1;
DeltaUnit( X_train, X_val, y_train, y_val, Eta,...
    E_Thresh, 0, 0, num_epochs, isIncremental, learnType, pltDS, 0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Problem 1d.
clc,clear all;
[X_train, y_train] = dataGen(100);
[X_val, y_val] = dataGen(50);
E_Thresh = 0.0001;
num_epochs = 100;

%Stochastic Gradient Descent.
Eta = logspace(-5,-1,5); % Can run @ 0.1 without destabalizing.
isIncremental = 1;
learnType = 0;
pltDS = 0;
[ iterative_time, iter_num_updates ] = DeltaUnit( X_train, X_val,...
    y_train, y_val, Eta, E_Thresh, 0, 0, num_epochs, isIncremental, learnType, pltDS, 0)

% Cute fprintf statement.

% Batch Mode
isIncremental = 0;
learnType = 0;
pltDS = 0;
Eta = logspace(-5,-2,4); %Running it much higher than 0.01 can lead to unstable results.
[ iterative_time, iter_num_updates ] = DeltaUnit( X_train, X_val,...
    y_train, y_val, Eta, E_Thresh, 0, 0, num_epochs, isIncremental, learnType, pltDS, 0)

% Cute fprintf statement.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Problem 2

%2a. Decaying Learning Rates.
clc,clear all;
[X_train, y_train] = dataGen(100);
[X_val, y_val] = dataGen(50);
E_Thresh = 0.0001;
num_epochs = 100;

isIncremental = 0;
learnType = 1;
pltDS = 0;
alpha = linspace(0.7,0.99,3);  %Starting decay rates.

for n = 1:length(alpha)
    Eta = logspace(-5,-2,4);
    [ iterative_time, iter_num_updates ] = DeltaUnit( X_train, X_val,...
        y_train, y_val, Eta, E_Thresh, alpha(n),0,...
        num_epochs, isIncremental, learnType, pltDS, 0 )
    Eta = logspace(-5,-1,5);
    [ iterative_time, iter_num_updates ] = DeltaUnit( X_train, X_val,...
        y_train, y_val, Eta, E_Thresh, alpha(n),0,...
        num_epochs, 1, learnType, pltDS, 0 )
end

%2b. Adaptive rates
clc,clear all;
[X_train, y_train] = dataGen(100);
[X_val, y_val] = dataGen(50);
E_Thresh = 0.1;
num_epochs = 100;

%Flags & problem specific hyper-parameters.
pltDS = 0;
isIncremental = 0;
learnType = 2;
adapt.d = 0.9;
adapt.D = 1.02;
adapt.Thresh = 0.05;
Eta = logspace(-5,-2,4);
Eta = 1
[ iterative_time, iter_num_updates ] = DeltaUnit( X_train, X_val,...
    y_train, y_val, Eta, E_Thresh, 0,adapt,...
    num_epochs, isIncremental, learnType, pltDS, 0 )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%3 Quadratic Gradient Descent.
clc,clear all;
data = gen_sigmoid_classes(100);
X_train = [ones(length(data),1),data(:,1:2)];
y_train = data(:,3)';
data = gen_sigmoid_classes(50);
X_val = [ones(length(data),1),data(:,1:2)];
y_val = data(:,3)';

figure,gscatter(X_train(:,2),X_train(:,3),y_train);
title('Generated Sigmoid Data');

E_Thresh = 0.0001;
num_epochs = 100;

%Flags & problem specific hyper-parameters.
pltDS = 0;
isIncremental = 0;
learnType = 0;

Eta = logspace(-5,-2,4);
[ iterative_time, iter_num_updates ] = DeltaUnit( X_train, X_val,...
    y_train, y_val, Eta, E_Thresh, 0,0,...
    num_epochs, isIncremental, learnType, pltDS, 1 )