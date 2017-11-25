function [] = PltE( err, Eta )
%Problem 1a. Plot E vs. Epoch for different learning rates.

% Need some subplots for Problem 2a.

figure();
hold on;
[~,A] = size(err);
m = 0;

%Compensation for plots w/ differing lengths. Will find max series and
%append m-n NaN values where m = the max length and n = length of each
%series.

for i = 1:A
    %find max length.
    if max(length(err{1,i})) > m
      m = length(err{1,i});
    end
end

for i = 1:A
    x_plt = 1:length(err{1,i});
    n = m-length(x_plt);  % how many short are we?
    x_plt = [x_plt,nan(1,n)];         % add that many NaNs
    y_plt = [err{1,i}, nan(1,n)];
    plot(x_plt, y_plt);
end

l_array{1} = num2str(Eta(1));
for i = 2:length(Eta)
    l_array{i} = num2str(Eta(i));
end

grid on;

%Need an if statement to change between number of updates and epochs. 
xlabel('Epoch');
ylabel('E');
title('E per Epoch');
legend(l_array)
hold off;
end

