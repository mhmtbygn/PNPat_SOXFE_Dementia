function f = pnpat(ts, C)
% PNPAT  Previous-Next Pattern features of one transformed signal (Equations 5-11).
%
%   f = pnpat(ts, C)
%   ts : transformed signal (channel identities, values 1..C)
%   C  : number of channels (19)
%   f  : 2 x C^2 matrix; row 1 = previous-transition features,
%        row 2 = next-transition features
%
%   Transition tables are row-normalized (Equation 10) and vectorized in
%   row-major order, so that feature index z corresponds to the table entry
%   (a, b) with a = ceil(z/C) and b = mod(z-1, C) + 1 (Equations 25-26).

ts = ts(:);
cur  = ts(2:end);
prev = ts(1:end-1);
tt_prev = accumarray([cur,  prev], 1, [C C]);   % Equations 6/8: tt(ts(j), ts(j-1))
tt_next = accumarray([prev, cur],  1, [C C]);   % Equations 7/9: tt(ts(k), ts(k+1))

f = [vec_rows(row_normalize(tt_prev)); vec_rows(row_normalize(tt_next))];
end

function T = row_normalize(T)
s = sum(T, 2);
s(s == 0) = 1;
T = T ./ s;                                      % Equation 10
end

function v = vec_rows(T)
v = reshape(T.', 1, []);                         % Equation 11 (row-major)
end
