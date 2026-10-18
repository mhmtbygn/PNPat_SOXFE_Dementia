function [ts1, ts2] = bct(X)
% BCT  Bipartite Channel Transformer (Equations 1-4).
%
%   [ts1, ts2] = bct(X)
%   X   : channels x samples EEG segment
%   ts1 : channel identities sorted by ascending amplitude at each sample,
%         concatenated over time (length channels*samples)
%   ts2 : the same for descending amplitude

[~, id1] = sort(X, 1, 'ascend');     % Equation 2
[~, id2] = sort(X, 1, 'descend');    % Equation 3
ts1 = id1(:);                        % Equation 4 (column-major = per sample)
ts2 = id2(:);
end
