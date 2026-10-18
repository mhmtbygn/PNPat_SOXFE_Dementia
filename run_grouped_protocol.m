function R = run_grouped_protocol(Feat, y, groups, class_names, cfg, outer_type)
% RUN_GROUPED_PROTOCOL  Participant-independent evaluation with nested selection.
%
%   outer_type = 'loso'  : leave-one-subject-out (one participant per outer fold)
%              = 'group' : participant-grouped cfg.kfold-fold cross-validation
%   In every outer fold, normalization, CWINCA, tkNN/IMV and CMV selection are
%   fitted on the training participants only, using participant-grouped inner
%   folds (cfg.inner_kfold). The held-out participants are predicted once and
%   their labels are used only to compute the metrics (Algorithm A1).

t0 = tic;
y = y(:);
groups = groups(:);
outer = make_folds(y, groups, outer_type, cfg.kfold, cfg.seed);
yhat = zeros(size(y));
nsel = zeros(max(outer), numel(Feat));
for k = unique(outer).'
    te = (outer == k);
    tr = ~te;
    if cfg.verbose
        fprintf('  outer fold %d/%d: %d training / %d test segments\n', ...
                k, max(outer), sum(tr), sum(te));
    end
    inner = make_folds(y(tr), groups(tr), 'group', cfg.inner_kfold, cfg.seed + k);
    Ftr = cellfun(@(A) A(tr, :), Feat, 'UniformOutput', false);
    Fte = cellfun(@(A) A(te, :), Feat, 'UniformOutput', false);
    model = soxfe_train(Ftr, y(tr), inner, cfg);
    yhat(te) = soxfe_predict(model, Fte, cfg);
    nsel(k, :) = cellfun(@(m) numel(m.sel), model.g);
end
if strcmpi(outer_type, 'loso')
    R.protocol = 'loso';
else
    R.protocol = 'nested';
end
R.class_names = class_names;
R.folds = outer;
R.ypred = yhat;
R.ytrue = y;
R.groups = groups;
R.metrics = compute_metrics(y, yhat, numel(class_names));
[R.participants, yvote, ytrue_p] = participant_results(y, yhat, groups);
R.metrics_participant = compute_metrics(ytrue_p, yvote, numel(class_names));
R.n_selected = nsel;
R.seconds = toc(t0);
end
