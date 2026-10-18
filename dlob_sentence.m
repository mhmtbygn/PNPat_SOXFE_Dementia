function [symbols, sentence] = dlob_sentence(sel, lut, C)
% DLOB_SENTENCE  DLob sentence of a selected feature index list (Equations 24-28).
%
%   Each selected feature index z (1..C^2) encodes a transition between the
%   channels a = ceil(z/C) and b = mod(z-1, C) + 1 (Equations 25-26), and
%   contributes the two symbols lut{a}, lut{b}.

if nargin < 3
    C = numel(lut);
end
sel = sel(:).';
a = ceil(sel / C);                    % Equation 25
b = mod(sel - 1, C) + 1;              % Equation 26
symbols = cell(1, 2 * numel(sel));
symbols(1:2:end) = lut(a);            % Equation 27
symbols(2:2:end) = lut(b);            % Equation 28
sentence = strjoin(symbols, '');
end
