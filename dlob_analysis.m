function [H, counts, T] = dlob_analysis(symbols, alphabet)
% DLOB_ANALYSIS  Entropy, symbol histogram and transition table of a DLob sentence.
%
%   H      : Shannon entropy of the symbol distribution (bits)
%   counts : number of occurrences of each symbol of the alphabet
%   T      : transition table; T(i, j) counts symbol i followed by symbol j

[~, idx] = ismember(symbols, alphabet);
if any(idx == 0)
    error('dlob_analysis:symbol', 'Unknown DLob symbol.');
end
K = numel(alphabet);
counts = accumarray(idx(:), 1, [K 1]).';
p = counts(counts > 0) / sum(counts);
H = -sum(p .* log2(p));
T = accumarray([idx(1:end-1).' idx(2:end).'], 1, [K K]);
end
