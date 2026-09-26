function C_Veh_Anc_Compare(Vehicle_num, Anchor_num)
%C_VEH_ANC_COMPARE  满邻居下 CMLKF 与 DMLKF_V1 的"迭代数 - RMSE"对比（单进程，无并行）
%
%   用法：  C_Veh_Anc_Compare            % 默认 8 车辆、4 基站
%           C_Veh_Anc_Compare(8, 4)
%
%   所有参数都是"先用 7 个入参构造对象，再直接给公有属性赋值"注入的，不改类文件：
%       UWB_sigma_anc = UWB_sigma_rel = 0.18
%       IMU_Sigma_a   = 0.15^2 * eye(3)      IMU_Sigma_w = 0.025^2 * eye(3)
%       beta_inv = 50    epsilon = 1e-4    max_step = Inf
%       bias_comp_ratio = 1（零偏完全补偿）    邻居数 = Vehicle_num - 1（满邻居）
%
%   两个场景各输出一份 CSV 到 RESULT 目录，文件已存在则直接读取、跳过运行：
%       C_Veh%d_Anc%d_nosie.csv   含 UWB 噪声数据集， iter = 1 / 5 / 10 / 15 / 20
%       C_Veh%d_Anc%d_pure.csv    无 UWB 噪声数据集，iter = 1 / 5 / 10 / 20 / 50 / 100

if nargin < 1 || isempty(Vehicle_num), Vehicle_num = 8; end
if nargin < 2 || isempty(Anchor_num),  Anchor_num  = 4; end

% ---------------------------------------------------------------- 路径
this_dir = fileparts(mfilename('fullpath'));
root     = fileparts(fileparts(this_dir));             % ...\DMLKF_CODE
data_dir = fullfile(this_dir, 'Data');
res_dir  = fullfile(this_dir, 'RESULT');

if ~exist(res_dir, 'dir'), mkdir(res_dir); end
addpath(fullfile(root, 'MLKF', 'CMLKF'), fullfile(root, 'MLKF', 'DMLKF'));

% ---------------------------------------------------------------- 固定参数
cfg = struct();
cfg.K        = Vehicle_num - 1;      % 满邻居
cfg.bias     = 1.0;                  % 零偏完全补偿
cfg.beta_inv = 100;
cfg.epsilon  = 1e-4;
cfg.max_step = Inf;
cfg.sig_anc  = 0.1;
cfg.sig_rel  = 0.1;
cfg.proc_a   = 0.15;
cfg.proc_w   = 0.025;
algos = {'CMLKF', 'DMLKF_V1'};

% ---------------------------------------------------------------- 场景定义
scen = struct();
scen(1).tag   = 'nosie';
scen(1).file  = sprintf('Trj_Veh%d_Anc%d_noise.mat', Vehicle_num, Anchor_num);
scen(1).iters = [1 5 10 15 20 30 40];
scen(1).title = sprintf('With UWB noise (Veh%d / Anc%d, full neighbors)', Vehicle_num, Anchor_num);

scen(2).tag   = 'pure';
scen(2).file  = sprintf('Trj_Veh%d_Anc%d_pure.mat', Vehicle_num, Anchor_num);
scen(2).iters = [1 5 10 20 50 100];
scen(2).title = sprintf('Noise-free UWB (Veh%d / Anc%d, full neighbors)', Vehicle_num, Anchor_num);

fprintf('\n================================================================================\n');
fprintf(' CMLKF vs DMLKF_V1  |  Veh=%d  Anc=%d  neighbors=%d  bias=%.1f\n', ...
        Vehicle_num, Anchor_num, cfg.K, cfg.bias);
fprintf(' beta_inv=%g  epsilon=%g  max_step=%g  UWB sigma=%.2f  IMU sigma_a=%.2f\n', ...
        cfg.beta_inv, cfg.epsilon, cfg.max_step, cfg.sig_anc, cfg.proc_a);
fprintf('================================================================================\n');

for s = 1:numel(scen)
    csv_path = fullfile(res_dir, sprintf('C_Veh%d_Anc%d_%s.csv', ...
                                         Vehicle_num, Anchor_num, scen(s).tag));
    
    fprintf('\n---------------- 场景 %d/%d：%s ----------------\n', s, numel(scen), scen(s).title);
    
    if exist(csv_path, 'file')
        T = readtable(csv_path);
        fprintf(' 已存在结果文件，直接读取并跳过运行：\n   %s\n', csv_path);
    else
        data_file = fullfile(data_dir, scen(s).file);
        if ~exist(data_file, 'file')
            error('找不到数据集：%s', data_file);
        end
        fprintf(' 数据集：%s\n', data_file);
        
        T = run_scenario(data_file, scen(s).tag, algos, scen(s).iters, cfg, Vehicle_num, Anchor_num);
        writetable(T, csv_path);
        fprintf(' 结果已写出：%s\n', csv_path);
    end
    
    print_scenario(T, scen(s));
    
    plot_scenario(T, scen(s), fullfile(res_dir, sprintf('C_Veh%d_Anc%d_%s_curve.png', ...
                                                        Vehicle_num, Anchor_num, scen(s).tag)));
end
fprintf('\n全部完成。结果目录：%s\n\n', res_dir);
end

% ================================================================ 跑一个场景
function T = run_scenario(data_file, tag, algos, iters, cfg, Vehicle_num, Anchor_num)
D = load_dataset(data_file, cfg.bias, Vehicle_num, Anchor_num);
nA = numel(algos); nI = numel(iters);
ests = cell(nA, nI);
rows = {};

for ai = 1:nA
    for ii = 1:nI
        it = iters(ii);
        fprintf(' ... %-9s iter=%-4d ', algos{ai}, it);
        
        t0 = tic;
        % 忽略不再需要的姿态和速度估计输出
        [est_p, ~, ~, n_uwb] = run_one(D, algos{ai}, it, cfg);
        sec = toc(t0);
        
        ests{ai, ii} = est_p;
        rp = calc_rmse(D, est_p);
        
        rows{end+1} = struct( ...                                        %#ok<AGROW>
            'scenario',    tag, ...
            'dataset',     D.data_name, ...
            'algo',        algos{ai}, ...
            'iter',        it, ...
            'neighbors',   cfg.K, ...
            'bias_comp_ratio', cfg.bias, ...
            'beta_inv',    cfg.beta_inv, ...
            'epsilon',     cfg.epsilon, ...
            'max_step',    cfg.max_step, ...
            'uwb_sigma_anc', cfg.sig_anc, ...
            'uwb_sigma_rel', cfg.sig_rel, ...
            'imu_sigma_a', cfg.proc_a, ...
            'imu_sigma_w', cfg.proc_w, ...
            'rmse_p',      rp, ...
            'div_p_max',   NaN, ...
            'n_uwb',       n_uwb, ...
            'sec',         sec);
            
        fprintf('RMSE p=%.6f   [%.1fs]\n', rp, sec);
    end
end

% 同 iter 下两算法的最大状态差（验证"应该一致"）
for ii = 1:nI
    d = NaN;
    if nA >= 2 && ~isempty(ests{1, ii}) && ~isempty(ests{2, ii})
        d = max(abs(ests{1, ii}(:) - ests{2, ii}(:)));
    end
    for ai = 1:nA
        rows{(ai-1)*nI + ii}.div_p_max = d;
    end
end
T = struct2table([rows{:}], 'AsArray', true);
end

% ================================================================ 单次运行
function [est_p, est_v, est_R, n_uwb] = run_one(D, algo, iter, cfg)
V = D.V; A = D.A; N = D.N;

% --- 用 7 个入参构造对象（噪声用可选入参先给一份，随后再用公有属性覆盖） ---
Noise = struct('IMU_Sigma_a', cfg.proc_a^2 * eye(3), ...
               'IMU_Sigma_w', cfg.proc_w^2 * eye(3), ...
               'UWB_sigma_anc', cfg.sig_anc, 'UWB_sigma_rel', cfg.sig_rel);
               
switch upper(algo)
    case 'CMLKF'
        kf = CMLKF(V, A, D.anchors, D.dt, D.p0, D.v0, D.R0, Noise);
    case 'DMLKF_V1'
        kf = DMLKF_V1(V, A, D.anchors, D.dt, D.p0, D.v0, D.R0, Noise);
    otherwise
        error('未知算法：%s', algo);
end

% --- 构造之后直接给公有属性赋值 ---
kf.max_iter       = iter;
kf.epsilon        = cfg.epsilon;
kf.max_step       = cfg.max_step;
if isprop(kf, 'beta_inv'), kf.beta_inv = cfg.beta_inv; end
kf.UWB_sigma_anc  = cfg.sig_anc;
kf.UWB_sigma_rel  = cfg.sig_rel;
kf.IMU_Sigma_a    = cfg.proc_a^2 * eye(3);
kf.IMU_Sigma_w    = cfg.proc_w^2 * eye(3);

% --- 掩码：满邻居 + 全基站 ---
V2V_Mask    = ones(V, V) - eye(V);      % 8 车满邻居 = 7 个邻居
Anchor_Mask = ones(V, A);

% --- 驱动循环 ---
est_p = zeros(N, 3, V);
est_v = zeros(N, 3, V);
est_R = zeros(3, 3, V, N);

for i = 1:V
    est_p(1, :, i)    = D.p0(3*i-2 : 3*i)';
    est_v(1, :, i)    = D.v0(3*i-2 : 3*i)';
    est_R(:, :, i, 1) = D.R0(:, :, i);
end

uwb_idx = 2;
n_uwb   = 0;

for k = 2:N
    acc = zeros(3, V); gyr = zeros(3, V);
    for i = 1:V
        nm = D.names{i};
        acc(:, i) = D.traj.(nm).IMU_acc_m(k-1, :)' ...
                    - cfg.bias * D.traj.(nm).IMU_bias_a_true(k-1, :)';
        gyr(:, i) = D.traj.(nm).IMU_gyro_m(k-1, :)' ...
                    - cfg.bias * D.traj.(nm).IMU_bias_w_true(k-1, :)';
    end
    
    kf.predict(acc, gyr);
    
    if uwb_idx <= numel(D.UWB_Time_Vec) && ...
       abs(D.traj.V1.Time_true(k) - D.UWB_Time_Vec(uwb_idx)) < 1e-5
       
        anc = zeros(V, A); rel = zeros(V, V);
        for i = 1:V
            nm = D.names{i};
            anc(i, :) = D.traj.(nm).UWB_Anchor(uwb_idx, 2:end);
            rel(i, :) = D.traj.(nm).UWB_Relative(uwb_idx, 2:end);
        end
        
        anc(Anchor_Mask == 0) = NaN;
        rel(V2V_Mask    == 0) = NaN;
        kf.update(anc, rel);
        uwb_idx = uwb_idx + 1;
        n_uwb   = n_uwb + 1;
    end
    
    for i = 1:V
        if strcmpi(algo, 'CMLKF')
            est_p(k, :, i)    = kf.p(3*i-2 : 3*i)';
            est_v(k, :, i)    = kf.v(3*i-2 : 3*i)';
            est_R(:, :, i, k) = kf.R(:, :, i);
        else
            est_p(k, :, i)    = kf.Nodes{i}.p';
            est_v(k, :, i)    = kf.Nodes{i}.v';
            est_R(:, :, i, k) = kf.Nodes{i}.R;
        end
    end
end
end

% ================================================================ 数据读取
function D = load_dataset(data_file, bias_comp_ratio, Vehicle_num, Anchor_num)
[~, data_name] = fileparts(data_file);
S = load(data_file);
traj = S.trajectories;
V = Vehicle_num; A = Anchor_num;
names = arrayfun(@(i) sprintf('V%d', i), 1:V, 'UniformOutput', false);

N  = numel(traj.V1.Time_true);
dt_imu = traj.V1.Time_true(2) - traj.V1.Time_true(1);

p0 = zeros(3*V, 1); v0 = zeros(3*V, 1); R0 = zeros(3, 3, V);
true_p = zeros(N, 3, V); true_v = zeros(N, 3, V); true_R = zeros(3, 3, V, N);

for i = 1:V
    nm = names{i};
    p0(3*i-2 : 3*i) = [traj.(nm).X_true(1);  traj.(nm).Y_true(1);  traj.(nm).Z_true(1)];
    v0(3*i-2 : 3*i) = [traj.(nm).Vx_true(1); traj.(nm).Vy_true(1); traj.(nm).Vz_true(1)];
    R0(:, :, i)     = traj.(nm).R_true(:, :, 1);
    
    true_p(:, :, i) = [traj.(nm).X_true(1:N), traj.(nm).Y_true(1:N), traj.(nm).Z_true(1:N)];
    true_v(:, :, i) = [traj.(nm).Vx_true(1:N), traj.(nm).Vy_true(1:N), traj.(nm).Vz_true(1:N)];
    true_R(:, :, i, :) = traj.(nm).R_true(:, :, 1:N);
end

D = struct('data_name', data_name, 'traj', traj, 'anchors', S.anchors, ...
           'V', V, 'A', A, 'names', {names}, ...
           'N', N, 'dt', dt_imu, 'p0', p0, 'v0', v0, 'R0', R0, ...
           'true_p', true_p, 'true_v', true_v, 'true_R', true_R, ...
           'UWB_Time_Vec', traj.V1.UWB_Anchor(:, 1), 'bias_comp_ratio', bias_comp_ratio);
end

% ================================================================ RMSE (仅计算位置)
function rmse_p = calc_rmse(D, est_p)
V = D.V;
rp = zeros(V, 1);
for i = 1:V
    rp(i) = sqrt(mean(sum((est_p(:, :, i) - D.true_p(:, :, i)).^2, 2)));
end
rmse_p = mean(rp);
end

% ================================================================ 打印 (仅展示位置)
function print_scenario(T, sc)
iters = sc.iters;
fprintf('\n 迭代数   |    CMLKF-RMSE   DMLKF_V1-RMSE      差值 \n');
fprintf(' %s\n', repmat('-', 1, 60));
for it = iters
    i1 = find(T.iter == it & strcmp(T.algo, 'CMLKF'), 1);
    i2 = find(T.iter == it & strcmp(T.algo, 'DMLKF_V1'), 1);
    if isempty(i1) || isempty(i2), continue; end
    fprintf(' %-8d | %12.6f  %12.6f  %11.2e\n', ...
            it, T.rmse_p(i1), T.rmse_p(i2), T.rmse_p(i1) - T.rmse_p(i2));
end

p1 = zeros(numel(iters), 1); p2 = p1;
for q = 1:numel(iters)
    p1(q) = T.rmse_p(T.iter == iters(q) & strcmp(T.algo, 'CMLKF'));
    p2(q) = T.rmse_p(T.iter == iters(q) & strcmp(T.algo, 'DMLKF_V1'));
end

[b1, k1] = min(p1);
dm = max(abs(p1 - p2));
fprintf('\n 两算法最大 RMSE 差 = %.3e m（iter=%d）→ %s\n', dm, iters(find(abs(p1-p2) == dm, 1)), ...
        tern(dm < 1e-6, '一致（满邻居下应当完全相等）', '不一致，需要检查'));
fprintf(' CMLKF    曲线：极小 %.6f 在 iter=%d；末点 iter=%d 为 %.6f（相对极小 %+.2f%%）→ %s\n', ...
        b1, iters(k1), iters(end), p1(end), 100*(p1(end)-b1)/b1, shape_name(p1));
fprintf(' DMLKF_V1 曲线：极小 %.6f 在 iter=%d；末点 iter=%d 为 %.6f（相对极小 %+.2f%%）→ %s\n', ...
        min(p2), iters(find(p2 == min(p2), 1)), iters(end), p2(end), ...
        100*(p2(end)-min(p2))/min(p2), shape_name(p2));
end

function s = shape_name(p)
if all(diff(p) < 0)
    s = '全程单调下降';
elseif p(end) < p(1)
    s = '总体下降（含局部回升）';
else
    s = '先下降后上升';
end
end

function s = tern(c, a, b)
if c, s = a; else, s = b; end
end

% ================================================================ 绘图 (仅绘制位置)
function plot_scenario(T, sc, png_path)
iters = sc.iters;
getv = @(algo, field) arrayfun(@(it) T.(field)(T.iter == it & strcmp(T.algo, algo)), iters);
pC = getv('CMLKF', 'rmse_p');    pD = getv('DMLKF_V1', 'rmse_p');

% 调整了图窗尺寸，只画一张图
f = figure('Name', sc.title, 'Color', 'w', 'Position', [100 100 550 420]);

plot(iters, pC, '-o', 'LineWidth', 1.8, 'MarkerSize', 6, 'Color', [0 0.45 0.74]); hold on;
plot(iters, pD, '--s', 'LineWidth', 1.8, 'MarkerSize', 6, 'Color', [0.85 0.33 0.10]);
grid on; box on;
xlabel('Gauss-Newton iterations'); ylabel('Position RMSE (m)');
title('Position RMSE');
xlim([0, max(iters) + 1]);
legend('CMLKF', 'DMLKF V1', 'Location', 'northeast');

sgtitle(sc.title, 'FontWeight', 'bold');
saveas(f, png_path);
fprintf(' 曲线图已保存：%s\n', png_path);
end