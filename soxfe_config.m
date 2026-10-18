function cfg = soxfe_config()
% SOXFE_CONFIG  Default settings of the PNPat/BCT-based SOXFE model.
%
%   cfg = soxfe_config() returns the parameters used in the manuscript
%   "Self-Organized Explainable Dementia Detection from Electroencephalography
%   Signals Using Directed Lobish and Previous-Next Pattern".
%   Edit the returned struct (or this file) to change paths or settings.

% ---------------------------------------------------------------- data
cfg.dataset_root  = fullfile(pwd, 'ds004504');   % OpenNeuro ds004504 root folder
cfg.use_derivatives = true;                      % true: preprocessed recordings (derivatives/)
cfg.task_name     = 'eyesclosed';
cfg.fs            = 500;                         % sampling frequency (Hz)
cfg.segment_sec   = 15;                          % non-overlapping segment length (s)
cfg.segment_len   = cfg.fs * cfg.segment_sec;    % 7500 samples
cfg.channels = {'Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4', ...
                'T4','T5','P3','Pz','P4','T6','O1','O2'};
cfg.group_codes   = {'A','F','C'};               % participants.tsv Group column
cfg.class_names   = {'AD','FD','HC'};            % label 1, 2, 3

% ---------------------------------------------------------------- DLob
% One symbol per channel, in the channel order above (Equation 24).
cfg.dlob_lut = {'FL','FR','FL','FL','Fz','FR','FR','TL','CL','Cz','CR', ...
                'TR','TL','PL','Pz','PR','TR','OL','OR'};
cfg.dlob_symbols = {'CL','CR','Cz','FL','FR','Fz','OL','OR','PL','PR','Pz','TL','TR'};

% ---------------------------------------------------------------- CWINCA
cfg.cwinca.start_th = 0.85;                      % Equation 16
cfg.cwinca.stop_th  = 0.9999;                    % Equation 17
cfg.cwinca.knn_k        = 1;                     % loss classifier (Equation 20)
cfg.cwinca.knn_distance = 'cityblock';
cfg.cwinca.knn_weight   = 'equal';
cfg.nca_args = {'Solver', 'lbfgs'};              % passed to fscnca
cfg.norm_eps = 1e-12;                            % epsilon of Equation 13

% ---------------------------------------------------------------- tkNN
cfg.tknn.distances = {'cityblock', 'euclidean', 'cosine'};   % Equation 30
cfg.tknn.weights   = {'squaredinverse', 'equal'};            % Equation 31
cfg.tknn.k         = 1:5;                                    % Equation 32
cfg.tknn.imv_range = 3:30;                                   % Equation 37

% ---------------------------------------------------------------- CMV
% Combinations of the four classifier-based outcomes (Table 3).
cfg.cmv.combinations = {[1 2 3], [1 3 4], [2 3 4], [1 2 3 4]};

% ---------------------------------------------------------------- validation
cfg.cases       = 1:5;                           % classification cases (see case_data.m)
cfg.protocols   = {'segment', 'loso', 'nested'};
cfg.kfold       = 10;                            % segment-wise and outer grouped folds
cfg.inner_kfold = 10;                            % inner participant-grouped folds
cfg.seed        = 1;                             % random seed for all fold assignments

% ---------------------------------------------------------------- output
cfg.results_dir = fullfile(pwd, 'results');
cfg.verbose     = true;
end
