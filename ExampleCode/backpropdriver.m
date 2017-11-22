clear all;
load('toydatax');
load('toydatay');
Target = toydatay';
[P,N] = size(toydatax)
[PT, M] = size(Target)
L = [N, 1, M]
eta = 0.01;
alpha = 0.1;
errorbound = 0.01;
epochsbound = 10^5;
y = backpropagation(toydatax,Target, L,eta,alpha, errorbound, epochsbound);
