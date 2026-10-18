function [Feat, y, groups, meta] = load_ds004504_features(cfg)
% LOAD_DS004504_FEATURES  Segment OpenNeuro ds004504 recordings and extract features.
%
%   [Feat, y, groups, meta] = load_ds004504_features(cfg)
%   Feat   : 1x4 cell, Feat{g} is (number of segments) x 361
%   y      : segment labels (1 = AD, 2 = FD, 3 = HC)
%   groups : participant index of each segment (1..88)
%   meta   : participant table (id, group, sex, age, MMSE, number of segments)
%
%   Recordings are divided into non-overlapping segments of cfg.segment_len
%   samples; an incomplete final segment is discarded.

root = cfg.dataset_root;
P = readtable(fullfile(root, 'participants.tsv'), 'FileType', 'text', ...
              'Delimiter', '\t', 'TextType', 'char');
C = numel(cfg.channels);
L = cfg.segment_len;
nP = height(P);

Feat = {[], [], [], []};
y = [];
groups = [];
nseg = zeros(nP, 1);
for i = 1:nP
    pid = P.participant_id{i};
    if cfg.use_derivatives
        set_file = fullfile(root, 'derivatives', pid, 'eeg', ...
                            sprintf('%s_task-%s_eeg.set', pid, cfg.task_name));
    else
        set_file = fullfile(root, pid, 'eeg', ...
                            sprintf('%s_task-%s_eeg.set', pid, cfg.task_name));
    end
    [data, labels, fs] = read_eeglab_set(set_file);
    if fs ~= cfg.fs
        error('Sampling rate of %s is %g Hz, expected %g Hz.', pid, fs, cfg.fs);
    end
    order = zeros(1, C);
    for c = 1:C
        hit = find(strcmpi(labels, cfg.channels{c}), 1);
        if isempty(hit)
            error('Channel %s not found for %s.', cfg.channels{c}, pid);
        end
        order(c) = hit;
    end
    data = data(order, :);

    label = find(strcmp(cfg.group_codes, P.Group{i}));
    ns = floor(size(data, 2) / L);
    Fi = zeros(ns, 4, C * C);
    for s = 1:ns
        Fi(s, :, :) = reshape(extract_features(data(:, (s-1)*L+1 : s*L), C), [1 4 C*C]);
    end
    for g = 1:4
        Feat{g} = [Feat{g}; reshape(Fi(:, g, :), ns, C * C)];
    end
    y = [y; repmat(label, ns, 1)];          %#ok<AGROW>
    groups = [groups; repmat(i, ns, 1)];    %#ok<AGROW>
    nseg(i) = ns;
    if cfg.verbose
        fprintf('  %s (%s): %d segments\n', pid, P.Group{i}, ns);
    end
end

meta = P;
meta.n_segments = nseg;
if cfg.verbose
    for k = 1:numel(cfg.class_names)
        fprintf('  %s: %d participants, %d segments\n', cfg.class_names{k}, ...
                sum(strcmp(P.Group, cfg.group_codes{k})), sum(y == k));
    end
    fprintf('  Total: %d segments\n', numel(y));
end
end
