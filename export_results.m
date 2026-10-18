function export_results(RES, meta, cfg)
% EXPORT_RESULTS  Write result tables (CSV) to cfg.results_dir.
%
%   summary_metrics.csv      accuracy (Wilson 95% CI), GM, BA, macro-F1 per case/protocol/level
%   classwise_metrics.csv    sensitivity, specificity, precision, NPV, F1 per class
%   confusion_matrices.csv   confusion matrix entries
%   per_participant_<protocol>_case<k>.csv
%   folds_<protocol>_case<k>.csv  participant and fold of each segment
%   selected_features.csv, dlob_*.csv/txt (segment-wise models)

out = cfg.results_dir;
if ~exist(out, 'dir'), mkdir(out); end
ids = meta.participant_id;

S = {}; Cw = {}; CMr = {};
for c = 1:numel(RES)
    if isempty(RES{c}), continue; end
    protos = fieldnames(RES{c});
    for p = 1:numel(protos)
        R = RES{c}.(protos{p});
        levels = {'segment', R.metrics};
        if isfield(R, 'metrics_participant')
            levels(end+1, :) = {'participant', R.metrics_participant}; %#ok<AGROW>
        end
        for l = 1:size(levels, 1)
            M = levels{l, 2};
            S(end+1, :) = {c, R.protocol, levels{l, 1}, M.n, M.acc, M.ci(1), M.ci(2), ...
                           M.gm, M.ba, M.macro_f1, R.seconds}; %#ok<AGROW>
            for k = 1:numel(R.class_names)
                Cw(end+1, :) = {c, R.protocol, levels{l, 1}, R.class_names{k}, M.recall(k), ...
                                M.specificity(k), M.precision(k), M.npv(k), M.f1(k)}; %#ok<AGROW>
                for j = 1:numel(R.class_names)
                    CMr(end+1, :) = {c, R.protocol, levels{l, 1}, R.class_names{k}, ...
                                     R.class_names{j}, M.cm(k, j)}; %#ok<AGROW>
                end
            end
        end
        Tf = table(ids(R.groups), R.ytrue, R.folds, R.ypred, 'VariableNames', ...
                   {'participant', 'true_class', 'fold', 'predicted_class'});
        writetable(Tf, fullfile(out, sprintf('folds_%s_case%d.csv', R.protocol, c)));
        if isfield(R, 'participants')
            Tp = R.participants;
            Tp.participant = ids(cellfun(@str2double, Tp.participant));
            writetable(Tp, fullfile(out, sprintf('per_participant_%s_case%d.csv', R.protocol, c)));
        end
    end
end
writetable(cell2table(S, 'VariableNames', {'case', 'protocol', 'level', 'n', 'accuracy', ...
    'ci_low', 'ci_high', 'gm', 'ba', 'macro_f1', 'seconds'}), fullfile(out, 'summary_metrics.csv'));
writetable(cell2table(Cw, 'VariableNames', {'case', 'protocol', 'level', 'class', ...
    'sensitivity', 'specificity', 'precision', 'npv', 'f1'}), fullfile(out, 'classwise_metrics.csv'));
writetable(cell2table(CMr, 'VariableNames', {'case', 'protocol', 'level', 'true_class', ...
    'predicted_class', 'count'}), fullfile(out, 'confusion_matrices.csv'));

% DLob outputs of the segment-wise models (Tables 7, 8, 10-14 and A1)
Fs = {}; Hs = {};
fid = fopen(fullfile(out, 'dlob_sentences.txt'), 'w');
for c = 1:numel(RES)
    if isempty(RES{c}) || ~isfield(RES{c}, 'segment'), continue; end
    model = RES{c}.segment.model;
    Tavg = zeros(numel(cfg.dlob_symbols));
    for g = 1:numel(model.g)
        sel = model.g{g}.sel;
        [sym, sentence] = dlob_sentence(sel, cfg.dlob_lut, numel(cfg.channels));
        [H, ~, T] = dlob_analysis(sym, cfg.dlob_symbols);
        Tavg = Tavg + T / numel(model.g);
        Fs(end+1, :) = {c, g, numel(sel), 100 * (1 - numel(sel) / numel(cfg.channels)^2)}; %#ok<AGROW>
        Hs(end+1, :) = {c, g, H}; %#ok<AGROW>
        fprintf(fid, 'Case %d, outcome %d: %s\n', c, g, sentence);
    end
    Tt = array2table(Tavg, 'VariableNames', cfg.dlob_symbols, 'RowNames', cfg.dlob_symbols);
    writetable(Tt, fullfile(out, sprintf('dlob_transition_case%d.csv', c)), 'WriteRowNames', true);
end
fclose(fid);
if ~isempty(Fs)
    writetable(cell2table(Fs, 'VariableNames', {'case', 'outcome', 'n_selected', 'reduction_pct'}), ...
               fullfile(out, 'selected_features.csv'));
    writetable(cell2table(Hs, 'VariableNames', {'case', 'outcome', 'entropy_bits'}), ...
               fullfile(out, 'dlob_entropy.csv'));
end
end
