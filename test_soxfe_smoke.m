% TEST_SOXFE_SMOKE  Quick check of the pipeline on synthetic data (about 1-3 min).
%
%   Runs feature extraction, DLob index checks, the segment-wise protocol and
%   the nested participant-grouped protocol on a small synthetic data set.
%   It does not use the dementia data and its accuracies have no meaning.

clear; clc;
addpath(fileparts(mfilename('fullpath')));
cfg = soxfe_config();
cfg.verbose = false;
cfg.kfold = 3;
cfg.inner_kfold = 3;
C = 19;

% ---- 1. feature extraction and Equation 11 / 25-26 consistency
X = randn(C, 200);
Fx = extract_features(X, C);
assert(isequal(size(Fx), [4 C*C]), 'feature size');
assert(all(abs(sum(reshape(Fx(1, :), C, C), 1) - 1) < 1e-12 | ...
           sum(reshape(Fx(1, :), C, C), 1) == 0), 'row normalization');
[ts1, ~] = bct(X);
T = accumarray([ts1(2:end) ts1(1:end-1)], 1, [C C]);
z = randi(C * C);
a = ceil(z / C); b = mod(z - 1, C) + 1;
Tn = T ./ max(sum(T, 2), 1);
assert(abs(Fx(1, z) - Tn(a, b)) < 1e-12, 'index mapping of Equations 25-26');
[sym, sentence] = dlob_sentence([1 19 20 361], cfg.dlob_lut, C);
assert(isequal(sym, {'FL','FL','FL','OR','FR','FL','OR','OR'}), 'DLob mapping');
[H, counts] = dlob_analysis(sym, cfg.dlob_symbols);
assert(sum(counts) == 8 && H > 0, 'DLob analysis');
fprintf('Feature extraction and DLob checks passed (%s).\n', sentence);

% ---- 2. synthetic participants: 3 classes x 4 participants x 6 segments
nper = 4; nseg = 6; L = 600;
Feat = {[], [], [], []}; y = []; groups = [];
pid = 0;
for cls = 1:3
    for p = 1:nper
        pid = pid + 1;
        gain = ones(C, 1); gain(cls * 5) = 3;          % class-dependent channel
        for s = 1:nseg
            Xs = gain .* randn(C, L) + 0.5 * randn(C, 1);
            Fs = extract_features(Xs, C);
            for g = 1:4, Feat{g}(end+1, :) = Fs(g, :); end
            y(end+1, 1) = cls; groups(end+1, 1) = pid; %#ok<SAGROW>
        end
    end
end

R1 = run_segmentwise_cv(Feat, y, groups, {'AD', 'FD', 'HC'}, cfg);
assert(numel(R1.ypred) == numel(y) && all(ismember(R1.ypred, 1:3)), 'segment-wise output');
fprintf('Segment-wise protocol: accuracy %.2f%%, selected features %s\n', ...
        R1.metrics.acc, mat2str(R1.n_selected));

R2 = run_grouped_protocol(Feat, y, groups, {'AD', 'FD', 'HC'}, cfg, 'group');
for q = 1:pid
    assert(numel(unique(R2.folds(groups == q))) == 1, 'participant split across outer folds');
end
assert(height(R2.participants) == pid, 'participant table');
fprintf('Nested grouped protocol: accuracy %.2f%% (segments), %.2f%% (participants)\n', ...
        R2.metrics.acc, R2.metrics_participant.acc);

M = compute_metrics([1 1 2 2]', [1 2 2 2]', 2);
assert(abs(M.acc - 75) < 1e-9 && abs(M.recall(1) - 50) < 1e-9 && abs(M.npv(1) - 2/3*100) < 1e-9, 'metrics');
fprintf('All smoke tests passed.\n');
