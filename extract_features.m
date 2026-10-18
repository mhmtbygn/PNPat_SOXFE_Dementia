function F = extract_features(X, C)
% EXTRACT_FEATURES  BCT + PNPat features of one EEG segment.
%
%   F = extract_features(X, C)
%   X : channels x samples segment (samples x channels is transposed automatically)
%   F : 4 x C^2 matrix; rows are the feature vectors F1-F4 of Figure 1
%       F1 = PNPat-previous(TS1), F2 = PNPat-next(TS1),
%       F3 = PNPat-previous(TS2), F4 = PNPat-next(TS2)

if nargin < 2
    C = 19;
end
if size(X, 1) ~= C && size(X, 2) == C
    X = X.';
end
[ts1, ts2] = bct(X);
F = [pnpat(ts1, C); pnpat(ts2, C)];
end
