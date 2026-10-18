function [Xn, mn, mx] = minmax_normalize(X, mn, mx, epsv)
% MINMAX_NORMALIZE  Min-max normalization (Equation 13).
%   [Xn, mn, mx] = minmax_normalize(X)             fits and applies
%   Xn = minmax_normalize(X, mn, mx)               applies stored parameters

if nargin < 4
    epsv = 1e-12;
end
if nargin < 2 || isempty(mn)
    mn = min(X, [], 1);
    mx = max(X, [], 1);
end
Xn = (X - mn) ./ (mx - mn + epsv);
end
