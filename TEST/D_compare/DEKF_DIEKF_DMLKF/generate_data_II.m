% =========================================================================
% 脚本名称: generate_data_II.m
% 功能描述: 用于 DEKF/DIEKF/DIEKF_ADMM/DMLKF 对比实验的 II 级(近场立体)数据生成器。
%           20x20x6m 封闭场地，基站部署在四周墙壁(高度分布在 1m, 3m, 5m 左右)；
%           多车轨迹为贯穿场地中心与四象限的 "正交直线 + 90度圆弧转弯" 立体分层巡航轨迹。
% =========================================================================
clc; clear; close all;

%% 1. 全局参数配置
% -------------------------------------------------------------------------
% 1.1 传感器采样频率与周期配置
F_imu  = 100;                                   % IMU 采样频率 (Hz)
F_uwb  = 10;                                    % UWB 采样频率 (Hz)
dt_imu = 1 / F_imu;                             % IMU 采样周期 (s)
dt_uwb = 1 / F_uwb;                             % UWB 采样周期 (s)
uwb_downsample_factor = round(dt_uwb / dt_imu); % UWB 相对于 IMU 的降采样倍数

if abs(dt_uwb / dt_imu - uwb_downsample_factor) > 1e-12
    error('The UWB period must be an integer multiple of the IMU period.');
end

% 1.2 场景规模与批次时长配置
Vehicle_num   = 8;                              % 车辆总数 [1, 18]
Anchor_num    = 4;                              % 启用的 UWB 基站数量 [1, 30]
t_end         = 60;                             % 单批次仿真时长 (s)
N_steps       = round(t_end / dt_imu) + 1;      % 单批次 IMU 总步数
repeat_time   = 1;                              % 独立重复批次数量
N_steps_total = N_steps * repeat_time;          % 所有批次拼接后的总步数

% 1.3 数据集标识配置
dataset_id = 'DEKF_DIEKF_DMLKF_60_II';

if Vehicle_num < 1 || Vehicle_num > 18, error('Vehicle_num must be in [1, 18].'); end
if Anchor_num < 1 || Anchor_num > 30,   error('Anchor_num must be in [1, 30].'); end
if repeat_time < 1 || round(repeat_time) ~= repeat_time
    error('repeat_time must be a positive integer.');
end


%% 2. 随机种子、传感器噪声与初始位置误差配置
% -------------------------------------------------------------------------
% 2.1 初始化随机种子
rng('shuffle');
rng_state = rng;
base_seed = double(rng_state.Seed);

% 2.2 UWB 测距噪声标准差 (m)
UWB_noise_params.sigma_anc = 0.1;               % 车-基站 (V2A) 测距噪声
UWB_noise_params.sigma_rel = 0.1;               % 车-车 (V2V) 相对测距噪声

% 2.3 IMU 测量噪声与零偏随机游走标准差
IMU_noise_params.sigma_na = 0.05;               % 加速度计测量白噪声
IMU_noise_params.sigma_nw = 0.005;              % 陀螺仪测量白噪声
IMU_noise_params.sigma_ba = 0.005;              % 加速度计零偏随机游走
IMU_noise_params.sigma_bw = 0.0005;             % 陀螺仪零偏随机游走

% 2.4 初始位置误差标准差配置 (m)
position_prior_std_xy = 1.5;                    % XY 平面初始位置偏差标准差
position_prior_std_z  = 1.0;                    % Z 轴初始位置偏差标准差
std_vec = [position_prior_std_xy; position_prior_std_xy; position_prior_std_z];


%% 3. 内嵌场景几何与真实轨迹生成
% -------------------------------------------------------------------------
test_dir = fileparts(mfilename('fullpath'));

[trajectory_template, anchors] = env_setup_crossing_corridors( ...
    Vehicle_num, Anchor_num, N_steps, dt_imu, t_end);

truth_position = zeros(N_steps, 3, Vehicle_num);
for n = 1:Vehicle_num
    v_name = sprintf('V%d', n);
    truth_position(:, 1, n) = trajectory_template.(v_name).X_true;
    truth_position(:, 2, n) = trajectory_template.(v_name).Y_true;
    truth_position(:, 3, n) = trajectory_template.(v_name).Z_true;
end


%% 4. 多批次传感器测量数据与初始误差采样
% -------------------------------------------------------------------------
trajectories           = struct();
batch_seeds            = base_seed + (0:repeat_time - 1);
initial_position_error = zeros(3, Vehicle_num, repeat_time);

for r = 1:repeat_time
    rng(batch_seeds(r), 'twister');
    
    batch_trajectories = generate_imu_uwb_batch( ...
        trajectory_template, truth_position, anchors, ...
        N_steps, dt_imu, uwb_downsample_factor, ...
        IMU_noise_params, UWB_noise_params);

    if r == 1
        trajectories = batch_trajectories;
    else
        trajectories = concatenate_trajectories( ...
            trajectories, batch_trajectories);
    end

    % 采样当前批次各车辆的初始位置偏移
    initial_position_error(:, :, r) = std_vec .* randn(3, Vehicle_num);
end

% 组装 scenario 结构体 (供外部对比脚本自动检测识别 II 级数据集)
scenario.position_prior_std_vec = std_vec;
scenario.initial_position_error = initial_position_error;
scenario.dataset_id             = dataset_id;
scenario.base_seed              = base_seed;
scenario.batch_seeds            = batch_seeds;


%% 5. 三维场景与轨迹可视化
% -------------------------------------------------------------------------
figure('Name', 'Level-II Scene: 20x20x6m Orthogonal Corridors & Perimeter Anchors', ...
    'Position', [100, 100, 900, 700], 'Color', 'w');
hold on; grid on; axis equal;
xlabel('X 轴位置 (m)');
ylabel('Y 轴位置 (m)');
zlabel('Z 轴位置 (m)');
title(sprintf('20x20x6m 场景：%d 车正交 90°转弯分层轨迹 与 %d 个周界基站', ...
    Vehicle_num, Anchor_num));

% 5.1 绘制 20x20x6 场景三维边界线框
X_max = 20;
Y_max = 20;
Z_max = 6;
line([0, X_max, X_max, 0, 0], [0, 0, Y_max, Y_max, 0], ...
    [0, 0, 0, 0, 0], 'Color', [0.5, 0.5, 0.5], 'LineStyle', '--');
line([0, X_max, X_max, 0, 0], [0, 0, Y_max, Y_max, 0], ...
    [Z_max, Z_max, Z_max, Z_max, Z_max], ...
    'Color', [0.5, 0.5, 0.5], 'LineStyle', '--');
for corner = [0, X_max]
    for side = [0, Y_max]
        line([corner, corner], [side, side], [0, Z_max], ...
            'Color', [0.5, 0.5, 0.5], 'LineStyle', '--');
    end
end

% 5.2 绘制四周墙壁 UWB 基站位置
h_anchor = plot3(anchors(:, 1), anchors(:, 2), anchors(:, 3), ...
    '^', 'MarkerSize', 11, 'MarkerFaceColor', 'r', ...
    'MarkerEdgeColor', 'k', 'LineWidth', 1.5, ...
    'DisplayName', 'UWB 基站');

% 5.3 绘制各车辆第一批次 (1:N_steps) 三维真实轨迹及起止点
colors = lines(Vehicle_num);
h_traj = zeros(1, Vehicle_num);
for n = 1:Vehicle_num
    v_name = sprintf('V%d', n);
    v_data = trajectories.(v_name);
    X_plot = v_data.X_true(1:N_steps);
    Y_plot = v_data.Y_true(1:N_steps);
    Z_plot = v_data.Z_true(1:N_steps);
    
    h_traj(n) = plot3(X_plot, Y_plot, Z_plot, ...
        'Color', colors(n, :), 'LineWidth', 2.2, ...
        'DisplayName', sprintf('V%d', n));
    % 起点标记 (圆圈)
    plot3(X_plot(1), Y_plot(1), Z_plot(1), ...
        'o', 'MarkerSize', 7, 'MarkerFaceColor', colors(n, :), ...
        'Color', 'k');
    % 终点标记 (星号)
    plot3(X_plot(end), Y_plot(end), Z_plot(end), ...
        '*', 'MarkerSize', 9, 'Color', colors(n, :), 'LineWidth', 1.5);
end
xlim([-1, 21]); ylim([-1, 21]); zlim([0, 6]);
view(-35, 30);
legend([h_anchor, h_traj], 'Location', 'northeastoutside');
hold off;


%% 6. 数据集保存与摘要打印
% -------------------------------------------------------------------------
save_dir = fullfile(test_dir, 'data');
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

mat_name  = sprintf('Trj_Veh%d_Anc%d_%s.mat', ...
    Vehicle_num, Anchor_num, dataset_id);
save_path = fullfile(save_dir, mat_name);

save(save_path, 'trajectories', 'anchors', ...
    'IMU_noise_params', 'UWB_noise_params', ...
    'Vehicle_num', 'Anchor_num', 't_end', ...
    'N_steps', 'N_steps_total', 'repeat_time', ...
    'uwb_downsample_factor', 'batch_seeds', 'scenario');

fprintf('\nLevel-II Data generation completed.\n');
fprintf('Vehicles: %d, Perimeter Anchors: %d (Z in [%.2f, %.2f] m)\n', ...
    Vehicle_num, Anchor_num, min(anchors(:, 3)), max(anchors(:, 3)));
fprintf('Initial Position Std: [%.2f, %.2f, %.2f] m\n', ...
    std_vec(1), std_vec(2), std_vec(3));
fprintf('Per batch: %.1f s, %d IMU steps, batches: %d\n', ...
    t_end, N_steps, repeat_time);
fprintf('Dataset ID: %s\n', dataset_id);
fprintf('Saved to: %s\n', save_path);


%% ======================= 局部函数区域 =======================

function [trajectories, anchors] = env_setup_crossing_corridors( ...
        Vehicle_num, Anchor_num, N_steps, dt_imu, t_end)
    % 20x20x6m 封闭场地：
    % 1) 30 个候选基站严格部署在四周墙壁 (X=0, X=20, Y=0, Y=20)，高度在 1m, 3m, 5m 左右交错分布；
    % 2) 4 组正交贯穿中心的 "直线 + 90度圆弧转弯" 航线，车辆分布在 1.5m, 3.0m, 4.5m 三层高度上。

    % --- 1. 四周墙壁上的 30 个周界基站坐标池 (Z 分布在 1.0m ~ 5.0m 之间) ---
    all_anchors_pool = [
        % 1~4: 四角基站 (涵盖 1m, 3m, 5m 立体高度落差)
          0.0,   0.0,  1.0;  %  1: 左下角 (低层 1.0m)
         20.0,  20.0,  5.0;  %  2: 右上角 (高层 5.0m)
          0.0,  20.0,  2.5;  %  3: 左上角 (中层 2.5m)
         20.0,   0.0,  3.8;  %  4: 右下角 (中高 3.8m)
        % 5~8: 四面墙中心点
         10.0,   0.0,  4.8;  %  5: 南墙中点
         20.0,  10.0,  1.2;  %  6: 东墙中点
         10.0,  20.0,  3.2;  %  7: 北墙中点
          0.0,  10.0,  4.5;  %  8: 西墙中点
        % 9~12: 四面墙 1/4 分位点
          5.0,   0.0,  2.8;  %  9: 南墙 5m
         20.0,   5.0,  5.0;  % 10: 东墙 5m
         15.0,  20.0,  1.1;  % 11: 北墙 15m
          0.0,  15.0,  3.5;  % 12: 西墙 15m
        % 13~16: 四面墙 3/4 分位点
         15.0,   0.0,  1.4;  % 13: 南墙 15m
         20.0,  15.0,  3.0;  % 14: 东墙 15m
          5.0,  20.0,  4.9;  % 15: 北墙 5m
          0.0,   5.0,  2.2;  % 16: 西墙 5m
        % 17~20: 四面墙 1/8 分位点
          2.5,   0.0,  4.2;  % 17: 南墙 2.5m
         20.0,   2.5,  1.5;  % 18: 东墙 2.5m
         17.5,  20.0,  3.0;  % 19: 北墙 17.5m
          0.0,  17.5,  4.8;  % 20: 西墙 17.5m
        % 21~24: 四面墙 3/8 分位点
          7.5,   0.0,  1.2;  % 21: 南墙 7.5m
         20.0,   7.5,  4.4;  % 22: 东墙 7.5m
         12.5,  20.0,  2.6;  % 23: 北墙 12.5m
          0.0,  12.5,  1.3;  % 24: 西墙 12.5m
        % 25~28: 四面墙 5/8 分位点
         12.5,   0.0,  3.4;  % 25: 南墙 12.5m
         20.0,  12.5,  2.0;  % 26: 东墙 12.5m
          7.5,  20.0,  1.2;  % 27: 北墙 7.5m
          0.0,   7.5,  5.0;  % 28: 西墙 7.5m
        % 29~30: 南北墙 7/8 对称点
         17.5,   0.0,  2.8;  % 29: 南墙 17.5m
          2.5,  20.0,  3.2;  % 30: 北墙 2.5m
    ];
    anchors = all_anchors_pool(1:Anchor_num, :);

    % --- 2. 定义 4 条正交贯穿中心的 "直线 + 90度圆弧转弯" 基础航线 ---
    % 所有直线段均严格平行于 X 轴或 Y 轴，每次转弯严格为 +90°(pi/2) 或 -90°(-pi/2)
    % 参数结构: [起点X0, 起点Y0, 初始航向psi0, 第一转角dpsi1, 第二转角dpsi2]
    route_templates = [
        % 航线 1 (左下 -> 中心 -> 右中): 正东直行 -> 左转90°(正北)穿中心 -> 右转90°(正东)直行
         3.0,  4.0,   0.0,       pi/2,  -pi/2;
        % 航线 2 (左上 -> 中心 -> 右中): 正东直行 -> 右转90°(正南)穿中心 -> 左转90°(正东)直行
         3.0, 16.0,   0.0,      -pi/2,   pi/2;
        % 航线 3 (右下 -> 中心 -> 中上): 正北直行 -> 左转90°(正西)穿中心 -> 右转90°(正北)直行
        16.0,  3.0,   pi/2,      pi/2,  -pi/2;
        % 航线 4 (右上 -> 中心 -> 左中): 正西直行 -> 左转90°(正南)穿中心 -> 右转90°(正西)直行
        17.0, 16.0,   pi,        pi/2,  -pi/2
    ];

    time     = (0:N_steps-1)' * dt_imu;
    total_L  = 18.0;                 % 每辆车在 t_end 内的总巡航路程 (m)
    v_cruise = total_L / t_end;      % 巡航线速度 (m/s)

    % 严格 3 段正交直线 + 2 处 90° 圆弧转弯的时间节点
    t1_s = 0.25 * t_end;  t1_e = 0.45 * t_end;  dT1 = t1_e - t1_s;
    t2_s = 0.55 * t_end;  t2_e = 0.75 * t_end;  dT2 = t2_e - t2_s;

    % 水平俯仰与横滚角设为 0
    roll_max_rad  = 0;
    pitch_max_rad = 0;

    trajectories = struct();

    for n = 1:Vehicle_num
        % 将车辆轮流分配至 4 条正交走廊航线上
        fam_idx = mod(n - 1, 4) + 1;
        tier_id = floor((n - 1) / 4);

        X0_base = route_templates(fam_idx, 1);
        Y0_base = route_templates(fam_idx, 2);
        psi0    = route_templates(fam_idx, 3);
        dpsi1   = route_templates(fam_idx, 4);
        dpsi2   = route_templates(fam_idx, 5);

        % 同航线簇车辆给予平行车道微偏移 (间距 1.0m)，保持横平竖直
        lat_dir = psi0 + pi/2;
        lon_off = (mod(tier_id, 2) - 0.5) * 1.0 * (tier_id > 0);
        lat_off = (floor((tier_id + 1)/2) * (-1)^tier_id) * 1.0;

        X_start = X0_base + lon_off * cos(psi0) + lat_off * cos(lat_dir);
        Y_start = Y0_base + lon_off * sin(psi0) + lat_off * sin(lat_dir);

        % 生成平滑 90° 转弯航向角序列 (匀速圆弧转弯，带余弦角速度平滑包络)
        psi_vec     = zeros(N_steps, 1);
        psi_dot_vec = zeros(N_steps, 1);

        for k = 1:N_steps
            t = time(k);
            if t <= t1_s
                psi_vec(k)     = psi0;
                psi_dot_vec(k) = 0;
            elseif t <= t1_e
                tau1           = (t - t1_s) / dT1;
                psi_vec(k)     = psi0 + dpsi1 * (tau1 - sin(2*pi*tau1)/(2*pi));
                psi_dot_vec(k) = (dpsi1 / dT1) * (1 - cos(2*pi*tau1));
            elseif t <= t2_s
                psi_vec(k)     = psi0 + dpsi1;
                psi_dot_vec(k) = 0;
            elseif t <= t2_e
                tau2           = (t - t2_s) / dT2;
                psi_vec(k)     = psi0 + dpsi1 + dpsi2 * (tau2 - sin(2*pi*tau2)/(2*pi));
                psi_dot_vec(k) = (dpsi2 / dT2) * (1 - cos(2*pi*tau2));
            else
                psi_vec(k)     = psi0 + dpsi1 + dpsi2;
                psi_dot_vec(k) = 0;
            end
        end

        % 水平速度、加速度与位置积分
        Vx = v_cruise * cos(psi_vec);
        Vy = v_cruise * sin(psi_vec);
        Ax = -v_cruise * sin(psi_vec) .* psi_dot_vec;
        Ay =  v_cruise * cos(psi_vec) .* psi_dot_vec;

        X_pos = zeros(N_steps, 1);
        Y_pos = zeros(N_steps, 1);
        X_pos(1) = X_start;
        Y_pos(1) = Y_start;
        for k = 1:N_steps-1
            X_pos(k+1) = X_pos(k) + Vx(k) * dt_imu + 0.5 * Ax(k) * (dt_imu^2);
            Y_pos(k+1) = Y_pos(k) + Vy(k) * dt_imu + 0.5 * Ay(k) * (dt_imu^2);
        end

        % 垂直高度落差与振荡：适配 6m 场地的等比例三层高度配置 (1.5m, 3.0m, 4.5m ± 0.8m，严格处于 0.7~5.3m 内)
        v_rate = 3;
        Zc     = 1.5 + mod(n, 3) * 1.5;
        Az     = 0.8;
        w_z    = (0.04 + 0.003 * n) * v_rate;
        phi_z  = n * (2*pi / Vehicle_num);

        Z_pos  = Zc + Az * sin(w_z * time + phi_z);
        Vz     = Az * w_z * cos(w_z * time + phi_z);
        Az_acc = -Az * (w_z^2) * sin(w_z * time + phi_z);

        % 生成姿态旋转矩阵序列 R_true (此时 roll=0, pitch=0，纯水平航向旋转)
        R_true  = zeros(3, 3, N_steps);
        phase_n = 2*pi*(n - 1) / max(1, Vehicle_num);
        for k = 1:N_steps
            t = time(k);
            roll_k  = roll_max_rad  * sin(0.6 * t + phase_n);
            pitch_k = pitch_max_rad * cos(0.4 * t + phase_n);
            R_true(:, :, k) = euler_to_rot_mat(roll_k, pitch_k, psi_vec(k));
        end

        v_name = sprintf('V%d', n);
        trajectories.(v_name).Time_true  = time;
        trajectories.(v_name).X_true     = X_pos;
        trajectories.(v_name).Y_true     = Y_pos;
        trajectories.(v_name).Z_true     = Z_pos;
        trajectories.(v_name).Vx_true    = Vx;
        trajectories.(v_name).Vy_true    = Vy;
        trajectories.(v_name).Vz_true    = Vz;
        trajectories.(v_name).Theta_true = psi_vec;
        trajectories.(v_name).A_true     = [Ax, Ay, Az_acc];
        trajectories.(v_name).R_true     = R_true;
    end
end

function batch = generate_imu_uwb_batch( ...
        trajectory_template, truth_position, anchors, ...
        N_steps, dt_imu, uwb_downsample_factor, ...
        IMU_noise_params, UWB_noise_params)
    
    Vehicle_num = size(truth_position, 3);
    Anchor_num  = size(anchors, 1);
    batch       = trajectory_template;
    g_vec       = [0; 0; -9.81];

    % --- 1. 生成 100 Hz IMU 测量数据 ---
    for n = 1:Vehicle_num
        v_name = sprintf('V%d', n);
        veh    = batch.(v_name);

        omega_body_ideal = zeros(N_steps, 3);
        a_body_ideal     = zeros(N_steps, 3);

        for k = 1:N_steps
            R_k     = veh.R_true(:, :, k);
            a_world = veh.A_true(k, :)';
            a_body_ideal(k, :) = (R_k' * (a_world - g_vec))';

            if k < N_steps
                dR = R_k' * veh.R_true(:, :, k+1);
                omega_body_ideal(k, :) = (log_SO3_vec(dR) / dt_imu)';
            else
                omega_body_ideal(k, :) = omega_body_ideal(k-1, :);
            end
        end

        ba_init = (rand(1, 3) - 0.5) * 0.1;
        bw_init = (rand(1, 3) - 0.5) * 0.01;
        b_a = ba_init + cumsum( ...
            randn(N_steps, 3) * IMU_noise_params.sigma_ba * sqrt(dt_imu), 1);
        b_w = bw_init + cumsum( ...
            randn(N_steps, 3) * IMU_noise_params.sigma_bw * sqrt(dt_imu), 1);

        veh.IMU_Time        = veh.Time_true;
        veh.IMU_acc_m       = a_body_ideal + b_a + ...
            randn(N_steps, 3) * IMU_noise_params.sigma_na;
        veh.IMU_gyro_m      = omega_body_ideal + b_w + ...
            randn(N_steps, 3) * IMU_noise_params.sigma_nw;
        veh.IMU_bias_a_true = b_a;
        veh.IMU_bias_w_true = b_w;
        batch.(v_name)      = veh;
    end

    % --- 2. 生成 10 Hz UWB 测量数据 ---
    idx_uwb      = 1:uwb_downsample_factor:N_steps;
    t_uwb        = trajectory_template.V1.Time_true(idx_uwb);
    N_uwb        = length(t_uwb);
    pos_true_uwb = truth_position(idx_uwb, :, :);

    for n = 1:Vehicle_num
        v_name = sprintf('V%d', n);

        UWB_Anchor       = zeros(N_uwb, 1 + Anchor_num);
        UWB_Anchor(:, 1) = t_uwb;
        for a_idx = 1:Anchor_num
            delta     = squeeze(pos_true_uwb(:, :, n)) - anchors(a_idx, :);
            dist_true = sqrt(sum(delta.^2, 2));
            UWB_Anchor(:, 1 + a_idx) = dist_true + ...
                randn(N_uwb, 1) * UWB_noise_params.sigma_anc;
        end
        batch.(v_name).UWB_Anchor = UWB_Anchor;

        UWB_Relative       = zeros(N_uwb, 1 + Vehicle_num);
        UWB_Relative(:, 1) = t_uwb;
        for j = 1:Vehicle_num
            if n == j
                UWB_Relative(:, 1 + j) = NaN;
            else
                delta = squeeze(pos_true_uwb(:, :, n)) - ...
                    squeeze(pos_true_uwb(:, :, j));
                dist_true = sqrt(sum(delta.^2, 2));
                UWB_Relative(:, 1 + j) = dist_true + ...
                    randn(N_uwb, 1) * UWB_noise_params.sigma_rel;
            end
        end
        batch.(v_name).UWB_Relative = UWB_Relative;
    end
end

function total = concatenate_trajectories(total, batch)
    Vehicle_num = numel(fieldnames(total));
    for n = 1:Vehicle_num
        v_name = sprintf('V%d', n);
        fields = fieldnames(batch.(v_name));
        for q = 1:numel(fields)
            field_name = fields{q};
            if strcmp(field_name, 'R_true')
                total.(v_name).(field_name) = cat(3, ...
                    total.(v_name).(field_name), ...
                    batch.(v_name).(field_name));
            else
                total.(v_name).(field_name) = cat(1, ...
                    total.(v_name).(field_name), ...
                    batch.(v_name).(field_name));
            end
        end
    end
end

function R = euler_to_rot_mat(phi, th, psi)
    cp = cos(phi); sp = sin(phi);
    ct = cos(th);  st = sin(th);
    cy = cos(psi); sy = sin(psi);
    R = [ct*cy, sp*st*cy - cp*sy, cp*st*cy + sp*sy;
         ct*sy, sp*st*sy + cp*cy, cp*st*sy - sp*cy;
         -st,   sp*ct,            cp*ct];
end

function v = log_SO3_vec(R)
    tr    = max(-1, min(3, trace(R)));
    theta = acos((tr - 1) / 2);
    if theta < 1e-7
        S = 0.5 * (R - R');
    else
        S = (theta / (2 * sin(theta))) * (R - R');
    end
    v = [S(3, 2); S(1, 3); S(2, 1)];
end