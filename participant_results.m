function [T, yvote, ytrue_p] = participant_results(ytrue, ypred, groups, participant_ids)
% PARTICIPANT_RESULTS  Per-participant accuracy and majority-vote prediction.
%
%   T : table with one row per participant (id, true class, number of
%       segments, correctly classified segments, accuracy (%), voted class)

[ug, ~, gi] = unique(groups(:));
np = numel(ug);
nseg = accumarray(gi, 1);
ncor = accumarray(gi, double(ytrue(:) == ypred(:)));
ytrue_p = accumarray(gi, ytrue(:), [], @(v) mode(v));
yvote = accumarray(gi, ypred(:), [], @(v) mode(v));
if nargin < 4 || isempty(participant_ids)
    ids = arrayfun(@(v) sprintf('%d', v), ug, 'UniformOutput', false);
else
    ids = participant_ids(ug);
    ids = ids(:);
end
T = table(ids, ytrue_p, nseg, ncor, 100 * ncor ./ nseg, yvote, ...
          'VariableNames', {'participant', 'true_class', 'n_segments', ...
                            'n_correct', 'accuracy', 'voted_class'});
assert(height(T) == np);
end
