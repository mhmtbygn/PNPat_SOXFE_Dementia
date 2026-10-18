function folds = make_folds(y, groups, type, k, seed)
% MAKE_FOLDS  Fold index of every segment.
%
%   folds = make_folds(y, groups, type, k, seed)
%   type = 'segment' : stratified segment-wise k-fold (participants may be
%                      split across folds; protocol of Table 3)
%          'group'   : participant-grouped k-fold; participants are shuffled
%                      within each class and dealt to the folds in turn, so
%                      all segments of a participant belong to one fold
%          'loso'    : one fold per participant
%   The random seed makes the assignment reproducible.

y = y(:);
groups = groups(:);
rng(seed, 'twister');
switch lower(type)
    case 'segment'
        c = cvpartition(y, 'KFold', k);
        folds = zeros(numel(y), 1);
        for f = 1:c.NumTestSets
            folds(test(c, f)) = f;
        end
    case 'group'
        [ug, ~, gi] = unique(groups);
        gy = accumarray(gi, y, [], @(v) mode(v));
        k = min(k, numel(ug));
        gf = zeros(numel(ug), 1);
        next = 0;
        for c = unique(gy).'
            members = find(gy == c);
            members = members(randperm(numel(members)));
            gf(members) = mod(next + (0:numel(members)-1).', k) + 1;
            next = next + numel(members);
        end
        folds = gf(gi);
    case 'loso'
        [~, ~, folds] = unique(groups);
    otherwise
        error('Unknown fold type %s.', type);
end
end
