% =========================================================================
% MAIN  PNPat/BCT-based SOXFE model for EEG dementia classification
% =========================================================================
% Implementation accompanying the manuscript
%   "Self-Organized Explainable Dementia Detection from Electroencephalography
%    Signals Using Directed Lobish and Previous-Next Pattern"
%
% Pipeline: BCT + PNPat feature extraction -> CWINCA feature selection ->
%           DLob explainable outputs -> tkNN (IMV, greedy) -> CMV fusion
% Protocols: segment-wise 10-fold CV (participant-mixed, Table 3),
%            leave-one-subject-out (LOSO) and nested participant-grouped
%            10-fold CV (Tables 4-6, Algorithm A1).
% Data: OpenNeuro ds004504 (https://openneuro.org/datasets/ds004504).
% =========================================================================

clc; clear; close all;
addpath(fileparts(mfilename('fullpath')));

cfg = soxfe_config();
% cfg.dataset_root = 'D:\data\ds004504';   % <-- set the dataset path here
% cfg.protocols = {'segment', 'nested'};   % LOSO retrains the model 88 times per case

if ~exist(cfg.results_dir, 'dir'), mkdir(cfg.results_dir); end
feature_file = fullfile(cfg.results_dir, 'features.mat');

% ------------------------------------------------------------- features
if exist(feature_file, 'file')
    fprintf('Loading features from %s\n', feature_file);
    load(feature_file, 'Feat', 'y', 'groups', 'meta');
else
    fprintf('Extracting BCT + PNPat features from %s\n', cfg.dataset_root);
    t0 = tic;
    [Feat, y, groups, meta] = load_ds004504_features(cfg);
    fprintf('Feature extraction: %.1f s (%.4f s per segment)\n', toc(t0), toc(t0) / numel(y));
    save(feature_file, 'Feat', 'y', 'groups', 'meta', '-v7.3');
end

% ------------------------------------------------------------- cases
RES = cell(1, max(cfg.cases));
for c = cfg.cases
    [idx, yc, names] = case_data(y, c);
    Fc = cellfun(@(A) A(idx, :), Feat, 'UniformOutput', false);
    gc = groups(idx);
    fprintf('\n===== Case %d (%s): %d segments, %d participants =====\n', ...
            c, strjoin(names, ' vs '), numel(yc), numel(unique(gc)));
    if any(strcmp(cfg.protocols, 'segment'))
        fprintf('-- Segment-wise %d-fold CV (participant-mixed)\n', cfg.kfold);
        RES{c}.segment = run_segmentwise_cv(Fc, yc, gc, names, cfg);
        fprintf('   accuracy %.2f%%\n', RES{c}.segment.metrics.acc);
    end
    if any(strcmp(cfg.protocols, 'loso'))
        fprintf('-- LOSO with nested participant-grouped selection\n');
        RES{c}.loso = run_grouped_protocol(Fc, yc, gc, names, cfg, 'loso');
        fprintf('   accuracy %.2f%% (segments), %.2f%% (participants)\n', ...
                RES{c}.loso.metrics.acc, RES{c}.loso.metrics_participant.acc);
    end
    if any(strcmp(cfg.protocols, 'nested'))
        fprintf('-- Nested participant-grouped %d-fold CV\n', cfg.kfold);
        RES{c}.nested = run_grouped_protocol(Fc, yc, gc, names, cfg, 'group');
        fprintf('   accuracy %.2f%% (segments), %.2f%% (participants)\n', ...
                RES{c}.nested.metrics.acc, RES{c}.nested.metrics_participant.acc);
    end
    save(fullfile(cfg.results_dir, 'results.mat'), 'RES', 'cfg', '-v7.3');
end

export_results(RES, meta, cfg);
fprintf('\nResults written to %s\n', cfg.results_dir);
