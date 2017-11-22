function [ o ] = FeedForward( X, w, batch, delta)
% Computes the feedforward pass of the perceptron units.

if batch 
    for i = 1:length(X)
        if delta
            o(i) = dot(X(i,:),w);  % Delta rule is unthresholded dot product.
        else
            o(i) = sign(dot(X(i,:),w));
        end
    end
end

