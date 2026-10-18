function [tk, yhat, acc_all] = tknn_select(X, y, folds, cfg)
% TKNN_SELECT  tkNN with iterative majority voting and greedy selection (Equations 29-40).
%
%   [tk, yhat, acc_all] = tknn_select(X, y, folds, cfg)
%   The 30 parameter-based outcomes and the 28 IMV outcomes are computed with
%   out-of-fold predictions on the given folds; the most accurate of the 58
%   outcomes is selected.
%   tk.params  : the 30 parameter settings
%   tk.members : parameter settings combined by the selected outcome
%                (one index = parameter-based outcome, several = IMV outcome)

P = tknn_params(cfg);
np = numel(P);
n = numel(y);
po = zeros(n, np);
acc = zeros(np, 1);
for p = 1:np                                                  % Equation 33
    po(:, p) = knn_oof_predict(X, y, folds, P(p).k, P(p).distance, P(p).weight);
    acc(p) = mean(po(:, p) == y);                             % Equation 35
end
[~, order] = sort(acc, 'descend');                            % Equation 36
L = cfg.tknn.imv_range(cfg.tknn.imv_range <= np);
vo = zeros(n, numel(L));
accv = zeros(numel(L), 1);
for j = 1:numel(L)                                            % Equation 37
    vo(:, j) = mode(po(:, order(1:L(j))), 2);
    accv(j) = mean(vo(:, j) == y);                            % Equation 38
end
acc_all = [acc; accv];
[~, id] = max(acc_all);                                       % Equation 39
if id <= np                                                   % Equation 40
    tk.members = id;
    yhat = po(:, id);
else
    tk.members = order(1:L(id - np)).';
    yhat = vo(:, id - np);
end
tk.params = P;
tk.selected_index = id;
end
