function iter_compare_worker(algo, iters, K, out_file, data_file, data_ratio, anchor_mode, cfg)
%ITER_COMPARE_WORKER 任务2 的执行体（由 iter_compare_main 并行拉起，也可单独调用）
%
%   iter_compare_worker('DMLKF', [10 300], 4, 'RESULT\_parts\part_1.csv')
%   iter_compare_worker('V1',    30,        4, 'RESULT\_parts\part_0.csv')
%
%   algo        : 'DMLKF_C'（对 iters 里每个值各跑一次；也接受旧的 'DMLKF' 写法）或 'V1'（只跑一次，迭代数记 30）
%   iters       : DMLKF 的 max_iter 列表（NaN 会被跳过）
%   K           : 车间邻居数
%   out_file    : 结果 CSV，列 = algorithm,iterations,pos_rmse,mean_gn_iter,cap_frac,sec,ok,note
%                 （中间文件，主脚本汇总后会删除）
%   data_file   : 数据集路径（默认 Data\Trj_Veh6_Anc4_3D.mat）
%   data_ratio  : 数据截取比例（默认 0.2 = 只用数据集前 20%，约 2001 步；传 1 = 全量）
%   anchor_mode : 'full'（默认，全部车用全部基站）或 'tiered'（30% 全基站 / 50% 半基站 / 20% 无基站）
%   cfg         : （可选）参数结构体，字段名 = 类的公开属性名，脚本里这样传：
%                   cfg = struct('max_step', 0.5, 'epsilon', 1e-5, 'ALPHA_SAFETY', 0.2);
%                   iter_compare_worker('DMLKF_C', [10 20], 4, out, data, 0.2, 'full', cfg);
%                 DMLKF_C 可设：max_iter / epsilon / beta_inv / max_step / ALPHA_SAFETY
%                              / alpha_const / print_flag / diag_flag
%                              / IMU_Sigma_a / IMU_Sigma_w / UWB_sigma_anc / UWB_sigma_rel
%                 DMLKF_V1 可设：max_iter / epsilon / beta_inv / max_step
%                              / IMU_Sigma_a / IMU_Sigma_w / UWB_sigma_anc / UWB_sigma_rel
%                 （只传你想改的字段即可；改 ALPHA_SAFETY 时会自动重算 alpha_const）
%
%   注意：V1(DMLKF_V1) 与 DMLKF 使用完全相同的车间掩码、基站掩码与观测，
%         唯一差别是集中式 GN vs 分布式 GN。

if nargin < 4 || isempty(out_file),    out_file = fullfile(tempdir, 'iter_compare_part.csv'); end
if nargin < 5 || isempty(data_file),   data_file = default_data_file(); end
if nargin < 6 || isempty(data_ratio),  data_ratio = 0.2; end   % 任务约定：只用数据集的 20%
if nargin < 7 || isempty(anchor_mode), anchor_mode = 'full'; end
if nargin < 8,                         cfg = struct(); end

add_paths();

% ---------------------------------------------------------------- 数据准备
D = load_case(data_file, data_ratio);
V2V_Mask    = sym_k_mask(D.V, K);
Anchor_Mask = anchor_mask(D.V, D.A, anchor_mode);

algo = upper(char(algo));
if strcmp(algo, 'V1'), iters = 30; end     % V1 只跑一次，迭代数记为 30

fid = fopen(out_file, 'w');
fprintf(fid, 'algorithm,iterations,pos_rmse,mean_gn_iter,cap_frac,sec,ok,note\n');

for it = iters(:)'
    if isnan(it), continue; end
    note = ''; ok = 0; rmse_p = NaN; mean_it = NaN; cap_frac = NaN; sec = NaN;
    try
        [rmse_p, mean_it, cap_frac, sec] = run_once(D, algo, it, V2V_Mask, Anchor_Mask, cfg);
        ok = 1;
    catch err
        sec  = NaN;
        note = strrep(strrep(sprintf('%s: %s', err.identifier, err.message), ',', ';'), '%', 'pct');
    end
    if strcmp(algo, 'V1'), iter_label = 30; else, iter_label = it; end
    fprintf(fid, '%s,%d,%.6f,%.2f,%.4f,%.1f,%d,%s\n', ...
            algo, iter_label, rmse_p, mean_it, cap_frac, sec, ok, note);
end
fclose(fid);

fid = fopen([out_file '.done'], 'w');
fprintf(fid, 'done\n');
fclose(fid);
end

% ================================================================ 单次滤波
function [rmse_p, mean_it, cap_frac, sec] = run_once(D, algo, max_iter, V2V_Mask, Anchor_Mask, cfg)
switch algo
    case {'DMLKF', 'DMLKF_C'}
        kf = DMLKF_C(D.V, D.A, D.anchors, D.dt, D.p0, D.v0, D.R0, V2V_Mask, max_iter);
        kf.print_flag = 0;                       % 不在子进程里刷未收敛警告
        kf.beta_inv = 0.1;
        kf.max_step = Inf; 

    case 'V1'
        kf = DMLKF_V1(D.V, D.A, D.anchors, D.dt, D.p0, D.v0, D.R0);
        kf.beta_inv = 0.01;
        kf.max_iter = 30;
        kf.max_step = Inf;
    otherwise
        error('未知算法: %s', algo);
end
kf = apply_cfg(kf, cfg);      % 脚本传入的参数（可选）

N = D.N_steps;
est_p = zeros(N, 3, D.V);
for i = 1:D.V
    est_p(1, :, i) = D.p0(3*i-2 : 3*i)';
end

uwb_idx = 2;
UWB_Time_Vec = D.UWB_Time_Vec;
t0 = tic;
for k = 2:N
    acc_m  = zeros(3, D.V);
    gyro_m = zeros(3, D.V);
    for i = 1:D.V
        nm = sprintf('V%d', i);
        acc_m(:, i)  = D.traj.(nm).IMU_acc_m(k-1, :)'  - D.bias_comp_ratio * D.traj.(nm).IMU_bias_a_true(k-1, :)';
        gyro_m(:, i) = D.traj.(nm).IMU_gyro_m(k-1, :)' - D.bias_comp_ratio * D.traj.(nm).IMU_bias_w_true(k-1, :)';
    end
    kf.predict(acc_m, gyro_m);

    curr_time = D.traj.V1.Time_true(k);
    if uwb_idx <= numel(UWB_Time_Vec) && abs(curr_time - UWB_Time_Vec(uwb_idx)) < 1e-5
        uwb_anc = zeros(D.V, D.A);
        uwb_rel = zeros(D.V, D.V);
        for i = 1:D.V
            nm = sprintf('V%d', i);
            uwb_anc(i, :) = D.traj.(nm).UWB_Anchor(uwb_idx, 2:end);
            uwb_rel(i, :) = D.traj.(nm).UWB_Relative(uwb_idx, 2:end);
        end
        uwb_anc(Anchor_Mask == 0) = NaN;
        uwb_rel(V2V_Mask    == 0) = NaN;
        kf.update(uwb_anc, uwb_rel);
        uwb_idx = uwb_idx + 1;
    end

    for i = 1:D.V
        est_p(k, :, i) = kf.Nodes{i}.p';
    end
end
sec = toc(t0);

rmse_each = zeros(D.V, 1);
for i = 1:D.V
    rmse_each(i) = sqrt(mean(sum((est_p(:, :, i) - D.true_p(:, :, i)).^2, 2)));
end
rmse_p = mean(rmse_each);

mean_it = NaN; cap_frac = NaN;
if isprop(kf, 'iter_hist') && ~isempty(kf.iter_hist)
    mean_it  = mean(kf.iter_hist);
    cap_frac = mean(kf.iter_hist >= kf.max_iter);   % 迭代被上限截断的比例（"跑满"的比例）
end
end

% ================================================================ 数据与拓扑
function D = load_case(data_file, data_ratio)
S = load(data_file, 'trajectories', 'anchors', 'Vehicle_num', 'Anchor_num');
D = struct();
D.traj  = S.trajectories;
D.anchors = S.anchors;
D.V     = S.Vehicle_num;
D.A     = S.Anchor_num;
D.bias_comp_ratio = 1;                     % 零偏全补偿（与 DMLKF_Test 一致，两算法相同）

N_total = numel(D.traj.V1.Time_true);
N = max(2, round(N_total * data_ratio));
D.N_steps = N;
D.N_steps_total = N_total;
D.dt = D.traj.V1.Time_true(2) - D.traj.V1.Time_true(1);

p0 = zeros(3*D.V, 1); v0 = zeros(3*D.V, 1); R0 = zeros(3, 3, D.V);
D.true_p = zeros(N, 3, D.V);
for i = 1:D.V
    nm = sprintf('V%d', i);
    p0(3*i-2 : 3*i) = [D.traj.(nm).X_true(1);  D.traj.(nm).Y_true(1);  D.traj.(nm).Z_true(1)];
    v0(3*i-2 : 3*i) = [D.traj.(nm).Vx_true(1); D.traj.(nm).Vy_true(1); D.traj.(nm).Vz_true(1)];
    R0(:, :, i)     = D.traj.(nm).R_true(:, :, 1);
    D.true_p(:, :, i) = [D.traj.(nm).X_true(1:N), D.traj.(nm).Y_true(1:N), D.traj.(nm).Z_true(1:N)];
end
D.p0 = p0; D.v0 = v0; D.R0 = R0;
D.UWB_Time_Vec = D.traj.V1.UWB_Anchor(:, 1);
end

function M = sym_k_mask(V, K)
%SYM_K_MASK 严格 K 度且对称的环状车间拓扑（与新版 DMLKF_Test 的写法一致）
%   偶数 K：±1..±K/2；奇数 K：±1..±K/2 再补一辆对径车（V 为偶数时成立）
K = min(K, V - 1);
M = zeros(V, V);
K_half = floor(K / 2);
for i = 1:V
    for d = 1:K_half
        M(i, mod(i + d - 1, V) + 1) = 1;
        M(i, mod(i - d - 1, V) + 1) = 1;
    end
    if mod(K, 2) ~= 0
        M(i, mod(i + V/2 - 1, V) + 1) = 1;
    end
end
M(logical(eye(V))) = 0;
assert(isequal(M, M'), 'V2V mask is not symmetric');
end

function A = anchor_mask(V, A_num, mode)
switch lower(mode)
    case 'full'
        A = ones(V, A_num);
    case 'tiered'      % 与 DMLKF_Test.m 一致：30% 全基站 / 50% 半基站 / 20% 无基站
        n1 = round(0.3 * V); n2 = round(0.5 * V);
        A = ones(V, A_num);
        for i = 1:V
            if i <= n1
                % 全基站
            elseif i <= n1 + n2
                for anc_idx = 1:A_num
                    if mod(anc_idx, 2) ~= 0, A(i, anc_idx) = 0; end
                end
            else
                A(i, :) = 0;
            end
        end
    otherwise
        error('未知基站掩码模式: %s', mode);
end
end

function f = default_data_file()
this_dir = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(fileparts(this_dir)));   % ...\TEST\GN_compare\iter_compare -> 工程根
f = fullfile(root, 'Data', 'Trj_Veh6_Anc4_3D.mat');
end

function add_paths()
this_dir = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(fileparts(this_dir)));
addpath(root, this_dir, ...
        fullfile(root, 'MLKF', 'DMLKF'), ...
        fullfile(root, 'MLKF', 'CMLKF'), ...
        fullfile(root, 'Data'));
end

function kf = apply_cfg(kf, cfg)
%APPLY_CFG 把脚本传入的参数结构体写进滤波器对象（字段名 = 类的公开属性名）
%   只写存在的属性、跳过空字段，因此可以只传你要改的那几个参数。
%   例： cfg = struct('max_step', 0.5, 'epsilon', 1e-5, 'ALPHA_SAFETY', 0.2);
if nargin < 2 || isempty(cfg), return; end
fn = fieldnames(cfg);
for q = 1:numel(fn)
    name = fn{q};
    if ~isprop(kf, name), continue; end      % 该类没有这个属性就跳过
    val = cfg.(name);
    if isempty(val), continue; end
    kf.(name) = val;
end

% DMLKF_C 的固定步长 alpha_const 是在构造函数里由 ALPHA_SAFETY 与 lambda_2 一次算好的：
% 构造之后再改 ALPHA_SAFETY 不会自动生效，这里补一次重算
% （除非你在 cfg 里直接指定了 alpha_const）
if isprop(kf, 'ALPHA_SAFETY') && isfield(cfg, 'ALPHA_SAFETY') && ~isempty(cfg.ALPHA_SAFETY) ...
        && ~(isfield(cfg, 'alpha_const') && ~isempty(cfg.alpha_const))
    kf.alpha_const = min(1.0, max(0.01, ...
        kf.ALPHA_SAFETY * (1 - kf.lambda_2) / (1 + sqrt(kf.lambda_2))));
end
end
