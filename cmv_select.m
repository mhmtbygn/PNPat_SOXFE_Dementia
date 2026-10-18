function [cm, yhat, acc8] = cmv_select(outcomes, y, combinations)
% CMV_SELECT  Combinational majority voting with greedy selection (Equations 41-48).
%
%   outcomes : n x 4 classifier-based outcomes
%   cm.members : outcomes combined by the selected result (one index = a
%                classifier-based outcome, several = a voted outcome)

ng = size(outcomes, 2);
nc = numel(combinations);
acc8 = zeros(ng + nc, 1);
vo = zeros(size(outcomes, 1), nc);
for r = 1:ng
    acc8(r) = mean(outcomes(:, r) == y);                      % Equation 45
end
for r = 1:nc
    vo(:, r) = mode(outcomes(:, combinations{r}), 2);         % Equations 41-44
    acc8(ng + r) = mean(vo(:, r) == y);                       % Equation 46
end
[~, id] = max(acc8);                                          % Equation 47
if id <= ng                                                   % Equation 48
    cm.members = id;
    yhat = outcomes(:, id);
else
    cm.members = combinations{id - ng};
    yhat = vo(:, id - ng);
end
cm.selected_index = id;
end
