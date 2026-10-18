function yte = soxfe_predict(model, Feat_te, cfg)
% SOXFE_PREDICT  Predict new observations with a trained SOXFE model.
ng = numel(model.g);
out = zeros(size(Feat_te{1}, 1), ng);
for g = 1:ng
    m = model.g{g};
    Xn = minmax_normalize(Feat_te{g}, m.mn, m.mx, cfg.norm_eps);
    out(:, g) = tknn_predict(m.tk, m.Xtr, m.ytr, Xn(:, m.sel));
end
yte = mode(out(:, model.cmv.members), 2);
end
