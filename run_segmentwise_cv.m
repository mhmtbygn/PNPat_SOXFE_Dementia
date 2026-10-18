function R = run_segmentwise_cv(Feat, y, groups, class_names, cfg)
% RUN_SEGMENTWISE_CV  Segment-wise 10-fold protocol of Table 3 (participant-mixed).
%
%   Folds are assigned to segments without participant grouping, and every
%   selection step uses the out-of-fold predictions of these folds on the
%   full data set. The reported accuracies are therefore optimistic and do
%   not measure generalization to unseen participants.

t0 = tic;
folds = make_folds(y, groups, 'segment', cfg.kfold, cfg.seed);
[model, yhat, rep] = soxfe_train(Feat, y, folds, cfg);
R.protocol = 'segment';
R.class_names = class_names;
R.folds = folds;
R.ypred = yhat;
R.ytrue = y;
R.groups = groups;
R.metrics = compute_metrics(y, yhat, numel(class_names));
R.outcome_acc = 100 * rep.outcome_acc;
R.voted_acc = 100 * rep.voted_acc;
R.outcome_gm = zeros(size(rep.outcomes, 2), 1);
for g = 1:size(rep.outcomes, 2)
    Mg = compute_metrics(y, rep.outcomes(:, g), numel(class_names));
    R.outcome_gm(g) = Mg.gm;
end
R.n_selected = rep.n_selected;
R.model = model;
R.seconds = toc(t0);
end
