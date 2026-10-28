function T = iter_compare_earlystop_check(data_ratio, K)
%ITER_COMPARE_EARLYSTOP_CHECK 诊断：为什么"迭代不足"的分布式 GN 看起来比集中式 V1 更好
%
%   两个对照：
%   1) 把**集中式 V1 自己**的 GN 迭代上限 max_iter 调小（默认 30）：
%      DMLKF_V1 的代码一个字没改，只是给它公开参数 max_iter 传不同值。
%      如果 V1 早停后 RMSE 也掉到它自己收敛值以下，说明这个"更好"与分布式无关，
%      而是"没算完 = 对测量更新做阻尼"的早停正则化效应。
%   2) 偏差/方差分解：RMSE^2 = bias^2 + std^2
%      bias = |误差的时间均值|（系统性滞后），std = 围绕该均值的波动（噪声放大）。
%      lag = 误差在真实速度方向上的投影均值（>0 表示估计落后于真值）。
%
%   用法： iter_compare_earlystop_check          % 默认 20% 数据，K = 4
%          iter_compare_earlystop_check(0.2, 4)

if nargin < 1 || isempty(data_ratio), data_ratio = 0.2; end
if nargin < 2 || isempty(K),          K = 4; end

this_dir = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(fileparts(this_dir)));
addpath(this_dir, fileparts(this_dir), ...
        fullfile(root, 'MLKF', 'DMLKF'), fullfile(root, 'MLKF', 'CMLKF'), fullfile(root, 'Data'));

data_file = fullfile(root, 'Data', 'Trj_Veh8_Anc4_3D.mat');
D = load_case(data_file, data_ratio);
V2V_Mask    = sym_k_mask(D.V, K);
Anchor_Mask = true(D.V, D.A);

fprintf('数据：8 车 4 基站（前 %.0f%% = %d 步），K = %d，掩码/噪声参数与任务2 完全一致\n\n', ...
        100*data_ratio, D.N_steps, K);

cfgs = { 'V1', 1; 'V1', 2; 'V1', 5; 'V1', 10; 'V1', 20; 'V1', 30; 'DMLKF_C', 10; 'DMLKF_C', 50 };

rows = cell(size(cfgs, 1), 1);
fprintf(' %-9s | %8s | %9s | %10s | %10s | %10s | %9s | %7s\n', ...
        'algo', 'GN iter', 'RMSE(m)', 'bias(mm)', 'std(mm)', 'lag(mm)', 'cap hit', 'sec');
fprintf(' %s\n', repmat('-', 1, 94));
for q = 1:size(cfgs, 1)
    algo = cfgs{q, 1}; it = cfgs{q, 2};

    switch algo
        case 'V1'
            kf = DMLKF_V1(D.V, D.A, D.anchors, D.dt, D.p0, D.v0, D.R0);
            kf.max_iter = it;            % 只设公开参数，未改动 DMLKF_V1 的代码
        case 'DMLKF_C'
            kf = DMLKF_C(D.V, D.A, D.anchors, D.dt, D.p0, D.v0, D.R0, V2V_Mask, it);
            kf.print_flag = 0;
    end

    [est_p, sec] = run_filter(kf, D, V2V_Mask, Anchor_Mask);
    m = error_stats(est_p, D);

    cap_frac = NaN;
    if isprop(kf, 'iter_hist') && ~isempty(kf.iter_hist)
        cap_frac = mean(kf.iter_hist >= kf.max_iter);
    end
    if isfinite(cap_frac)
        cap_txt = sprintf('%.4f%%', 100*cap_frac);
    else
        cap_txt = '-';
    end

    fprintf(' %-9s | %8d | %9.4f | %10.4f | %10.4f | %10.4f | %9s | %7.1f\n', ...
            algo, it, m.rmse, 1000*m.bias, 1000*m.std, 1000*m.lag, cap_txt, sec);
    rows{q} = sprintf('%s,%d,%.4f,%.4f,%.4f,%.4f', algo, it, m.rmse, 1000*m.bias, 1000*m.std, 1000*m.lag);
end
fprintf(' %s\n', repmat('-', 1, 94));
fprintf(' bias = |误差时间均值|（系统滞后），std = 围绕该均值的波动（噪声放大）\n');
fprintf(' lag  = 误差在真实速度方向上的投影均值（>0 表示估计落后于真值）\n');
fprintf(' 原始 UWB 测距噪声水平 = %.4f m（供对照）\n\n', uwb_noise_level(D));

T = rows;
end

% ================================================================ 滤波
function [est_p, sec] = run_filter(kf, D, V2V_Mask, Anchor_Mask)
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
        acc_m(:, i)  = D.traj.(nm).IMU_acc_m(k-1, :)'  - D.traj.(nm).IMU_bias_a_true(k-1, :)';
        gyro_m(:, i) = D.traj.(nm).IMU_gyro_m(k-1, :)' - D.traj.(nm).IMU_bias_w_true(k-1, :)';
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
        uwb_anc(~Anchor_Mask) = NaN;
        uwb_rel(~V2V_Mask)    = NaN;
        kf.update(uwb_anc, uwb_rel);
        uwb_idx = uwb_idx + 1;
    end

    for i = 1:D.V
        est_p(k, :, i) = kf.Nodes{i}.p';
    end
end
sec = toc(t0);
end

% ================================================================ 误差统计
function m = error_stats(est_p, D)
V = D.V;
rmse_v = zeros(V,1); bias_v = zeros(V,1); std_v = zeros(V,1); lag_v = zeros(V,1);
for i = 1:V
    e = est_p(:, :, i) - D.true_p(:, :, i);        % N x 3
    rmse_v(i) = sqrt(mean(sum(e.^2, 2)));
    eb = mean(e, 1);                               % 误差的时间均值（系统偏差）
    bias_v(i) = norm(eb);
    std_v(i)  = sqrt(mean(sum((e - eb).^2, 2)));

    v = D.true_v(:, :, i);
    vn = sqrt(sum(v.^2, 2));
    vn(vn < 1e-9) = 1;
    vhat = v ./ vn;
    lag_v(i) = mean(sum(e .* vhat, 2));
end
m = struct('rmse', mean(rmse_v), 'bias', mean(bias_v), 'std', mean(std_v), 'lag', mean(lag_v));
end

function lvl = uwb_noise_level(D)
%UWB_NOISE_LEVEL 原始基站测距的均方根误差（m），给出"测量本身的噪声水平"
dt = D.dt;
se = 0; cnt = 0;
for m = 1:numel(D.UWB_Time_Vec)
    idx = round(D.UWB_Time_Vec(m) / dt) + 1;
    if idx < 1 || idx > D.N_steps, continue; end
    for i = 1:D.V
        nm = sprintf('V%d', i);
        r  = D.traj.(nm).UWB_Anchor(m, 2:end);
        pt = [D.traj.(nm).X_true(idx); D.traj.(nm).Y_true(idx); D.traj.(nm).Z_true(idx)];
        rt = sqrt(sum((D.anchors - pt').^2, 2))';
        d  = r(:)' - rt;
        d  = d(isfinite(d));
        se = se + sum(d.^2); cnt = cnt + numel(d);
    end
end
if cnt > 0, lvl = sqrt(se / cnt); else, lvl = NaN; end
end

% ================================================================ 数据与拓扑
function D = load_case(data_file, data_ratio)
S = load(data_file, 'trajectories', 'anchors', 'Vehicle_num', 'Anchor_num');
D = struct();
D.traj    = S.trajectories;
D.anchors = S.anchors;
D.V       = S.Vehicle_num;
D.A       = S.Anchor_num;

N_total = numel(D.traj.V1.Time_true);
N = max(2, round(N_total * data_ratio));
D.N_steps = N;
D.dt = D.traj.V1.Time_true(2) - D.traj.V1.Time_true(1);

p0 = zeros(3*D.V, 1); v0 = zeros(3*D.V, 1); R0 = zeros(3, 3, D.V);
D.true_p = zeros(N, 3, D.V);
D.true_v = zeros(N, 3, D.V);
for i = 1:D.V
    nm = sprintf('V%d', i);
    p0(3*i-2 : 3*i) = [D.traj.(nm).X_true(1);  D.traj.(nm).Y_true(1);  D.traj.(nm).Z_true(1)];
    v0(3*i-2 : 3*i) = [D.traj.(nm).Vx_true(1); D.traj.(nm).Vy_true(1); D.traj.(nm).Vz_true(1)];
    R0(:, :, i)     = D.traj.(nm).R_true(:, :, 1);
    D.true_p(:, :, i) = [D.traj.(nm).X_true(1:N), D.traj.(nm).Y_true(1:N), D.traj.(nm).Z_true(1:N)];
    D.true_v(:, :, i) = [D.traj.(nm).Vx_true(1:N), D.traj.(nm).Vy_true(1:N), D.traj.(nm).Vz_true(1:N)];
end
D.p0 = p0; D.v0 = v0; D.R0 = R0;
D.UWB_Time_Vec = D.traj.V1.UWB_Anchor(:, 1);
end

function M = sym_k_mask(V, K)
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
