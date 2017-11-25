function [w] = BackProp(X, y, w, o, Eta, p3)
%   Performs backpropagation. The various Eta's are determined outside
%   This provides a common way to backprop regardless of input shape and
%   type of learning. 

if p3
    w = w + (Eta * (y - o) * [X(:,1), (X(:,2:3) + X(:,2:3).^2)])';
else
    w = w + (Eta * (y - o) * X)';
end

