function [idx, yc, class_names] = case_data(y, case_id)
% CASE_DATA  Segments and labels of the five classification cases.
%
%   Case 1: AD vs FD vs HC            (labels 1, 2, 3)
%   Case 2: Dementia (AD+FD) vs HC    (1 = dementia, 2 = HC)
%   Case 3: AD vs FD                  (1 = AD, 2 = FD)
%   Case 4: AD vs HC                  (1 = AD, 2 = HC)
%   Case 5: FD vs HC                  (1 = FD, 2 = HC)
%   For the binary cases, class 1 is the positive class.

y = y(:);
switch case_id
    case 1
        idx = true(size(y));
        yc = y;
        class_names = {'AD', 'FD', 'HC'};
    case 2
        idx = true(size(y));
        yc = 1 + (y == 3);
        class_names = {'Dementia', 'HC'};
    case 3
        idx = (y == 1) | (y == 2);
        yc = y(idx);
        class_names = {'AD', 'FD'};
    case 4
        idx = (y == 1) | (y == 3);
        yc = 1 + (y(idx) == 3);
        class_names = {'AD', 'HC'};
    case 5
        idx = (y == 2) | (y == 3);
        yc = 1 + (y(idx) == 3);
        class_names = {'FD', 'HC'};
    otherwise
        error('Unknown case %d.', case_id);
end
end
