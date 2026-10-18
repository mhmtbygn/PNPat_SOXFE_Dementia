function P = tknn_params(cfg)
% TKNN_PARAMS  The 30 kNN parameter settings of tkNN (Equations 30-34).
P = struct('distance', {}, 'weight', {}, 'k', {});
for a = 1:numel(cfg.tknn.distances)
    for b = 1:numel(cfg.tknn.weights)
        for c = 1:numel(cfg.tknn.k)
            P(end+1) = struct('distance', cfg.tknn.distances{a}, ...
                              'weight', cfg.tknn.weights{b}, ...
                              'k', cfg.tknn.k(c)); %#ok<AGROW>
        end
    end
end
end
