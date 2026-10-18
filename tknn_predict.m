function yte = tknn_predict(tk, Xtr, ytr, Xte)
% TKNN_PREDICT  Apply a selected tkNN outcome to new observations.
m = tk.members;
pred = zeros(size(Xte, 1), numel(m));
for j = 1:numel(m)
    p = tk.params(m(j));
    mdl = fitcknn(Xtr, ytr, 'NumNeighbors', p.k, 'Distance', p.distance, ...
                  'DistanceWeight', p.weight);
    pred(:, j) = predict(mdl, Xte);
end
yte = mode(pred, 2);
end
