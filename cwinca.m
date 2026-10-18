function [sel, info] = cwinca(Xn, y, folds, cfg)
% CWINCA  Cumulative weight-based iterative NCA feature selection (Equations 12-23).
%
%   [sel, info] = cwinca(Xn, y, folds, cfg)
%   Xn    : normalized feature matrix (observations x features)
%   folds : fold vector used to compute the kNN loss of each candidate subset
%   sel   : indices of the selected features, in descending NCA weight order
%           (these indices generate the DLob sentence, Equation 23)

mdl = fscnca(Xn, y, cfg.nca_args{:});                       % Equation 14
w = mdl.FeatureWeights(:);
[ws, qin] = sort(w, 'descend');
if sum(ws) <= 0
    cw = (1:numel(ws)).' / numel(ws);
else
    cw = cumsum(ws) / sum(ws);                              % Equation 15
end
start_i = find(cw > cfg.cwinca.start_th, 1, 'first');      % Equation 16
stop_i  = find(cw <= cfg.cwinca.stop_th, 1, 'last');       % Equation 17
if isempty(start_i)
    start_i = 1;
end
if isempty(stop_i) || stop_i < start_i
    stop_i = start_i;
end

qs = start_i:stop_i;
loss = zeros(numel(qs), 1);
for j = 1:numel(qs)                                         % Equations 18-20
    yhat = knn_oof_predict(Xn(:, qin(1:qs(j))), y, folds, cfg.cwinca.knn_k, ...
                           cfg.cwinca.knn_distance, cfg.cwinca.knn_weight);
    loss(j) = mean(yhat ~= y);
end
[~, ix] = min(loss);                                        % Equation 21
sel = qin(1:qs(ix));                                        % Equations 22-23

info.weights = w;
info.ranked  = qin;
info.cumw    = cw;
info.start   = start_i;
info.stop    = stop_i;
info.loss    = loss;
info.n_selected = numel(sel);
end
