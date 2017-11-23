function [ X, y ] = dataGen( N )
% Generates N data points with X = [x1 x2] and y {-1, +1}
% Data is classified as positive if x1 + 2*x2 -2 > 0.

% Randomly define x1
x1 = rand([N,1]);

% Pull from a normal distribution that is centered on
% (2-mu_x1)/2 with the sigma of x1's distribution. 
% This should approximately make the ratio of pos/neg 50/50.
x2 = normrnd((2-mean(x1))/2,std(x1),N,1);
sum((x1 + 2*x2 -2) > 0)/N  % Validating the ratio of pos/neg.
 
% Assign pos and neg labels
for n = 1:N
   if (x1(n) + 2*x2(n) -2) > 0
      y(n) = 1;
   else
      y(n) = -1;
   end
end

% Concat the input features into X matrix. 
% Prepend array of 1's for bias.
X = [ones(N,1), x1, x2];

end

