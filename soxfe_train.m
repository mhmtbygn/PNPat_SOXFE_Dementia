function [model, yhat, report] = soxfe_train(Feat, y, folds, cfg)
% SOXFE_TRAIN  Fit the SOXFE model (normalization, CWINCA, tkNN, CMV).
%
%   [model, yhat, report] = soxfe_train(Feat, y, folds, cfg)
%   Feat  : 1x4 cell of feature matrices (observations x 361)
%   y     : labels
%   folds : fold vector used for every selection step (CWINCA loss, tkNN,
%           IMV, CMV). All labels passed here are used for selection.
%   yhat  : out-of-fold predictions of the selected final outcome
%   report: accuracies of the four classifier-based and four voted outcomes

y = y(:);
ng = numel(Feat);
outcomes = zeros(numel(y), ng);
model.g = cell(1, ng);
for g = 1:ng
    [Xn, mn, mx] = minmax_normalize(Feat{g}, [], [], cfg.norm_eps);
    [sel, cinfo] = cwinca(Xn, y, folds, cfg);
    [tk, outcomes(:, g), acc_all] = tknn_select(Xn(:, sel), y, folds, cfg);
    model.g{g} = struct('mn', mn, 'mx', mx, 'sel', sel(:).', 'tk', tk, ...
                        'cwinca', cinfo, 'tknn_acc', acc_all, ...
                        'Xtr', Xn(:, sel), 'ytr', y);
    if cfg.verbose
        fprintf('    F%d: %d features selected, outcome accuracy %.2f%%\n', ...
                g, numel(sel), 100 * mean(outcomes(:, g) == y));
    end
end
[model.cmv, yhat, acc8] = cmv_select(outcomes, y, cfg.cmv.combinations);
report.outcome_acc = acc8(1:ng);
report.voted_acc   = acc8(ng+1:end);
report.outcomes    = outcomes;
report.n_selected  = cellfun(@(m) numel(m.sel), model.g);
end
