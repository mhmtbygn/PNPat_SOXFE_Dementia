function yhat = knn_oof_predict(X, y, folds, k, distance, weight)
% KNN_OOF_PREDICT  Out-of-fold kNN predictions for a given fold vector.

yhat = zeros(size(y));
for f = unique(folds).'
    te = (folds == f);
    tr = ~te;
    mdl = fitcknn(X(tr, :), y(tr), 'NumNeighbors', k, ...
                  'Distance', distance, 'DistanceWeight', weight);
    yhat(te) = predict(mdl, X(te, :));
end
end
