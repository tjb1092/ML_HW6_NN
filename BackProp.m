function [ E, w, delW ] = BackProp( w, delW, o, y, Eta)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

E = .5 * sum((sign(o)-y).^2);  %Error E.
delW = delW + Eta * ((y - sign(o)) * X)';
w = w + delW;
end

