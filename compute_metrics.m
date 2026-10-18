function M = compute_metrics(ytrue, ypred, nclass)
% COMPUTE_METRICS  Classification metrics from true and predicted labels.
%
%   M.cm      : confusion matrix (rows = true class, columns = predicted)
%   M.acc     : accuracy (%), M.ci = Wilson 95% interval (%)
%   M.ba, M.gm, M.macro_f1
%   M.recall, M.specificity, M.precision, M.npv, M.f1 : per class (one-vs-rest, %)

ytrue = ytrue(:); ypred = ypred(:);
if nargin < 3
    nclass = max([ytrue; ypred]);
end
cm = accumarray([ytrue ypred], 1, [nclass nclass]);
n = sum(cm(:));
tp = diag(cm);
fn = sum(cm, 2) - tp;
fp = sum(cm, 1).' - tp;
tn = n - tp - fn - fp;
recall = tp ./ max(tp + fn, 1);
spec = tn ./ max(tn + fp, 1);
prec = tp ./ max(tp + fp, 1);
npv = tn ./ max(tn + fn, 1);
f1 = 2 * prec .* recall ./ max(prec + recall, eps);

M.cm = cm;
M.n = n;
M.acc = 100 * sum(tp) / n;
M.ci = 100 * wilson_ci(sum(tp), n);
M.recall = 100 * recall;
M.specificity = 100 * spec;
M.precision = 100 * prec;
M.npv = 100 * npv;
M.f1 = 100 * f1;
M.ba = mean(M.recall);
M.gm = 100 * prod(recall) ^ (1 / nclass);
M.macro_f1 = mean(M.f1);
end

function ci = wilson_ci(k, n)
z = 1.959963984540054;
p = k / n;
den = 1 + z^2 / n;
c = (p + z^2 / (2 * n)) / den;
h = z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / den;
ci = [c - h, c + h];
end
