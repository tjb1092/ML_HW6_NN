function [w] = BackProp(X, y, w, o, Eta)
%   Performs backpropagation. The various Eta's are determined outside
%   This provides a common way to backprop regardless of input shape and
%   type of learning. 

w = w + (Eta * (y - o) * X)';

end

