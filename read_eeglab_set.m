function [data, labels, fs] = read_eeglab_set(set_file)
% READ_EEGLAB_SET  Read an EEGLAB .set file (with or without .fdt) without EEGLAB.
%
%   [data, labels, fs] = read_eeglab_set(set_file)
%   data   : channels x samples (double)
%   labels : 1 x channels cell array of channel labels
%   fs     : sampling rate (Hz)

S = load(set_file, '-mat');
if isfield(S, 'EEG')
    EEG = S.EEG;
else
    EEG = S;
end

if ischar(EEG.data) || isstring(EEG.data)
    fdt_file = fullfile(fileparts(set_file), char(EEG.data));
    fid = fopen(fdt_file, 'r', 'ieee-le');
    if fid < 0
        error('read_eeglab_set:fdt', 'Cannot open data file %s', fdt_file);
    end
    data = fread(fid, [double(EEG.nbchan), Inf], 'float32=>double');
    fclose(fid);
else
    data = double(EEG.data);
end
data = reshape(data, double(EEG.nbchan), []);
labels = {EEG.chanlocs.labels};
fs = double(EEG.srate);
end
