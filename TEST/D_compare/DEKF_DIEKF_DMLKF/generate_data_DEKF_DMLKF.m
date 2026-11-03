% =========================================================================
% 脚本名称: generate_data_DEKF_DMLKF.m
% 功能描述: 用于 DEKF/DIEKF/DMLKF 参数对比实验的独立数据生成器。
%           本文件整合了 Data/generate_data_3D.m 与 Data/env_setup.m，
%           轨迹与 Data/env_setup.m 完全一致，基站池从 10 个扩展至 15 个，
%           运行时无需依赖外部 Data/ 目录。
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
Vehicle_num   = 8;                              % 车辆总数
Anchor_num    = 4;                              % 启用的 UWB 基站数量
t_end         = 20;                             % 单批次仿真时长 (s)
N_steps       = round(t_end / dt_imu) + 1;      % 单批次 IMU 总步数
repeat_time   = 4;                              % 独立重复批次数量
N_steps_total = N_steps * repeat_time;          % 所有批次拼接后的总步数

% 1.3 数据集标识配置
% 相同的 dataset_id 会直接覆盖同名 MAT 文件，但数据是不一样的
dataset_id = 'DEKF_DIEKF_DMLKF_20_4';

if repeat_time < 1 || round(repeat_time) ~= repeat_time
    error('repeat_time must be a positive integer.');
end


%% 2. 随机种子与传感器噪声配置
% -------------------------------------------------------------------------
% 2.1 初始化随机种子（每次运行均生成新种子，即便 dataset_id 未改变）
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


%% 3. 内嵌场景几何与真实轨迹生成
% -------------------------------------------------------------------------
test_dir = fileparts(mfilename('fullpath'));

% 3.1 生成理想轨迹模板与基站坐标
[trajectory_template, anchors] = env_setup_embedded( ...
    Vehicle_num, Anchor_num, N_steps, dt_imu, t_end);

if size(anchors, 1) ~= Anchor_num
    error('Embedded scene returned %d anchors, but Anchor_num is %d.', ...
        size(anchors, 1), Anchor_num);
end

% 3.2 提取单批次全网车辆真实位置矩阵 (N_steps x 3 x Vehicle_num)
truth_position = zeros(N_steps, 3, Vehicle_num);
for n = 1:Vehicle_num
    v_name = sprintf('V%d', n);
    truth_position(:, 1, n) = trajectory_template.(v_name).X_true;
    truth_position(:, 2, n) = trajectory_template.(v_name).Y_true;
    truth_position(:, 3, n) = trajectory_template.(v_name).Z_true;
end


%% 4. 多批次传感器测量数据生成与拼接
% -------------------------------------------------------------------------
trajectories = struct();
batch_seeds  = base_seed + (0:repeat_time - 1); % 为每个批次分配独立随机种子

for r = 1:repeat_time
    rng(batch_seeds(r), 'twister');
    
    % 生成当前批次带噪声的 IMU 与 UWB 数据
    batch_trajectories = generate_imu_uwb_batch( ...
        trajectory_template, truth_position, anchors, ...
        N_steps, dt_imu, uwb_downsample_factor, ...
        IMU_noise_params, UWB_noise_params);

    % 拼接各批次数据
    if r == 1
        trajectories = batch_trajectories;
    else
        trajectories = concatenate_trajectories( ...
            trajectories, batch_trajectories);
    end
end


%% 5. 三维场景与轨迹可视化
% -------------------------------------------------------------------------
figure('Name', 'Data scene DEKF/DIEKF/DMLKF test data', ...
    'Position', [100, 100, 900, 700]);
hold on; grid on; axis equal;
xlabel('X position (m)');
ylabel('Y position (m)');
zlabel('Z position (m)');
title(sprintf('%d vehicles, %d anchors: embedded Data-style geometry', ...
    Vehicle_num, Anchor_num));

% 5.1 绘制场景三维边界线框
X_max = 120;
Y_max = 120;
Z_max = 15;
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

% 5.2 绘制 UWB 基站位置
h_anchor = plot3(anchors(:, 1), anchors(:, 2), anchors(:, 3), ...
    '^', 'MarkerSize', 13, 'MarkerFaceColor', 'r', ...
    'MarkerEdgeColor', 'k', 'LineWidth', 1.5, ...
    'DisplayName', 'Anchors');

% 5.3 绘制各车辆三维真实轨迹及起止点
colors = lines(Vehicle_num);
h_traj = zeros(1, Vehicle_num);
for n = 1:Vehicle_num
    v_name = sprintf('V%d', n);
    v_data = trajectories.(v_name);
    
    % 轨迹曲线
    h_traj(n) = plot3(v_data.X_true, v_data.Y_true, v_data.Z_true, ...
        'Color', colors(n, :), 'LineWidth', 2.5, ...
        'DisplayName', sprintf('V%d', n));
    % 起点标记 (圆圈)
    plot3(v_data.X_true(1), v_data.Y_true(1), v_data.Z_true(1), ...
        'o', 'MarkerSize', 8, 'MarkerFaceColor', colors(n, :), ...
        'Color', colors(n, :));
    % 终点标记 (星号)
    plot3(v_data.X_true(end), v_data.Y_true(end), v_data.Z_true(end), ...
        '*', 'MarkerSize', 10, 'Color', colors(n, :));
end
view(3);
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
    'uwb_downsample_factor', 'batch_seeds');

fprintf('\nData generation completed.\n');
fprintf('Geometry source: embedded env_setup_embedded\n');
fprintf('Vehicles: %d, anchors: %d\n', Vehicle_num, Anchor_num);
fprintf('Anchor Z range: %.3f to %.3f m\n', ...
    min(anchors(:, 3)), max(anchors(:, 3)));
fprintf('Per batch: %.1f s, %d IMU steps, batches: %d\n', ...
    t_end, N_steps, repeat_time);
fprintf('Total IMU steps: %d\n', N_steps_total);
fprintf('Total UWB steps: %d\n', ...
    size(trajectories.V1.UWB_Anchor, 1));
fprintf('Dataset ID: %s\n', dataset_id);
fprintf('Random base seed: %u\n', base_seed);
fprintf('Saved to: %s\n', save_path);


%% ======================= 局部函数区域 =======================

% -------------------------------------------------------------------------
% 函数: generate_imu_uwb_batch
% 功能: 为单个批次生成带噪声与随机游走零偏的 100Hz IMU 和 10Hz UWB 测量数据
% -------------------------------------------------------------------------
function batch = generate_imu_uwb_batch( ...
        trajectory_template, truth_position, anchors, ...
        N_steps, dt_imu, uwb_downsample_factor, ...
        IMU_noise_params, UWB_noise_params)
    
    Vehicle_num = size(truth_position, 3);
    Anchor_num  = size(anchors, 1);
    batch       = trajectory_template;

    % --- 1. 生成 100 Hz IMU 测量数据 ---
    for n = 1:Vehicle_num
        v_name = sprintf('V%d', n);
        veh    = batch.(v_name);

        % 计算理想机体系角速度
        theta_unwrapped  = unwrap(veh.Theta_true);
        wz_true          = gradient(theta_unwrapped, dt_imu);
        omega_body_ideal = [zeros(N_steps, 2), wz_true];

        % 计算理想机体系比力 (去除重力加速度并投影至机体系)
        a_body_ideal = zeros(N_steps, 3);
        g_vec        = [0; 0; -9.81];
        for k = 1:N_steps
            R_k     = veh.R_true(:, :, k);
            a_world = [veh.A_true(k, 1);
                       veh.A_true(k, 2);
                       veh.A_true(k, 3)];
            a_body_ideal(k, :) = (R_k' * (a_world - g_vec))';
        end

        % 生成初始零偏与随机游走零偏序列
        ba_init = (rand(1, 3) - 0.5) * 0.1;
        bw_init = (rand(1, 3) - 0.5) * 0.01;
        b_a = ba_init + cumsum( ...
            randn(N_steps, 3) * IMU_noise_params.sigma_ba * sqrt(dt_imu), 1);
        b_w = bw_init + cumsum( ...
            randn(N_steps, 3) * IMU_noise_params.sigma_bw * sqrt(dt_imu), 1);

        % 叠加零偏与高斯白噪声，保存 IMU 数据
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

        % 2.1 生成车对基站 (V2A) 测距数据 (第 1 列为时间戳)
        UWB_Anchor       = zeros(N_uwb, 1 + Anchor_num);
        UWB_Anchor(:, 1) = t_uwb;
        for a_idx = 1:Anchor_num
            delta     = squeeze(pos_true_uwb(:, :, n)) - anchors(a_idx, :);
            dist_true = sqrt(sum(delta.^2, 2));
            UWB_Anchor(:, 1 + a_idx) = dist_true + ...
                randn(N_uwb, 1) * UWB_noise_params.sigma_anc;
        end
        batch.(v_name).UWB_Anchor = UWB_Anchor;

        % 2.2 生成车对车 (V2V) 相对测距数据 (第 1 列为时间戳，自测距置为 NaN)
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

% -------------------------------------------------------------------------
% 函数: concatenate_trajectories
% 功能: 将新生成的单批次轨迹与传感器数据追加拼接到总结构体中
% -------------------------------------------------------------------------
function total = concatenate_trajectories(total, batch)
    Vehicle_num = numel(fieldnames(total));
    for n = 1:Vehicle_num
        v_name = sprintf('V%d', n);
        fields = fieldnames(batch.(v_name));
        for q = 1:numel(fields)
            field_name = fields{q};
            if strcmp(field_name, 'R_true')
                % 姿态旋转矩阵沿第 3 维拼接
                total.(v_name).(field_name) = cat(3, ...
                    total.(v_name).(field_name), ...
                    batch.(v_name).(field_name));
            else
                % 其余二维序列沿第 1 维（时间行）拼接
                total.(v_name).(field_name) = cat(1, ...
                    total.(v_name).(field_name), ...
                    batch.(v_name).(field_name));
            end
        end
    end
end

% -------------------------------------------------------------------------
% 函数: env_setup_embedded
% 功能: 内嵌的 Data/env_setup.m 场景模型副本。
%       前 10 个基站与 Data/env_setup.m 完全一致，第 11-15 个为内部扩展基站。
% -------------------------------------------------------------------------
function [trajectories, anchors] = env_setup_embedded( ...
        Vehicle_num, Anchor_num, N_steps, dt_imu, t_end)

    % --- 1. 定义候选基站坐标池 (15x3) ---
    all_anchors_pool = [
          0,   0, 1.0;  % 1: 左下角
        120, 120, 1.0;  % 2: 右上角
          0, 120, 1.0;  % 3: 左上角
        120,   0, 1.0;  % 4: 右下角
         60,  60, 1.0;  % 5: 中心点
         60,   0, 1.0;  % 6: 下边界中点
          0,  60, 1.0;  % 7: 左边界中点
        120,  60, 2.0;  % 8: 右边界中点
         60, 120, 1.0;  % 9: 上边界中点
         30,  90, 2.0;  % 10: 左上内部偏移点
         90,  30, 1.5;  % 11: 右下内部偏移点
         30,  30, 1.5;  % 12: 左下内部偏移点
         90,  90, 1.5;  % 13: 右上内部偏移点
         20,  60, 1.5;  % 14: 左中内部偏移点
        100,  60, 1.5   % 15: 右中内部偏移点
    ];

    if Anchor_num > size(all_anchors_pool, 1) || Anchor_num < 1
        error('Anchor_num must be in the range 1-%d.', ...
            size(all_anchors_pool, 1));
    end
    anchors = all_anchors_pool(1:Anchor_num, :);

    if Vehicle_num > 18 || Vehicle_num < 1
        error('Vehicle_num must be in the range 1-18.');
    end

    % --- 2. 解析生成各车辆的三维真实运动轨迹 ---
    v_rate       = 2;
    trajectories = struct();

    for n = 1:Vehicle_num
        % 为每辆车分配唯一的径向振荡频率与空间初相
        R0    = 20 + 1.5 * n;
        Ar    = 5 + 0.2 * n;
        w_r   = (0.03 + 0.002 * n) * v_rate;
        phi_r = n * (2*pi/Vehicle_num);

        % 奇偶编号车辆采用相反的环绕运动方向
        dir_flag = (-1)^n;
        w_a      = (dir_flag * (0.015 + 0.001 * n)) * v_rate;
        phi_a    = n * (pi/2);

        % 将车辆分布在三个标称高度层上并进行垂向正弦振荡
        Zc    = 4 + mod(n, 3) * 3.5;
        Az    = 2.0;
        w_z   = (0.04 + 0.003 * n) * v_rate;
        phi_z = n * (2*pi/Vehicle_num);

        P_true     = zeros(N_steps, 3);
        V_true     = zeros(N_steps, 3);
        A_true     = zeros(N_steps, 3);
        Theta_true = zeros(N_steps, 1);

        for k = 1:N_steps
            t = (k - 1) * dt_imu;

            % 解析计算动态半径及其一阶、二阶导数
            R      = R0 + Ar * sin(w_r * t + phi_r);
            R_dot  = Ar * w_r * cos(w_r * t + phi_r);
            R_ddot = -Ar * w_r^2 * sin(w_r * t + phi_r);

            % 解析计算环绕角度及其导数
            alpha     = w_a * t + phi_a;
            alpha_dot = w_a;

            % 三维真实位置
            X = 60 + R * cos(alpha);
            Y = 60 + R * sin(alpha);
            Z = Zc + Az * sin(w_z * t + phi_z);

            % 三维真实速度
            Vx = R_dot * cos(alpha) - R * alpha_dot * sin(alpha);
            Vy = R_dot * sin(alpha) + R * alpha_dot * cos(alpha);
            Vz = Az * w_z * cos(w_z * t + phi_z);

            % 三维真实加速度
            Ax     = (R_ddot - R * alpha_dot^2) * cos(alpha) ...
                - (2 * R_dot * alpha_dot) * sin(alpha);
            Ay     = (R_ddot - R * alpha_dot^2) * sin(alpha) ...
                + (2 * R_dot * alpha_dot) * cos(alpha);
            Az_acc = -Az * w_z^2 * sin(w_z * t + phi_z);

            % 计算连续的航向角 (消除 +-pi 跳变)
            th_raw = atan2(Vy, Vx);
            if k == 1
                th_curr = th_raw;
            else
                diff    = th_raw - Theta_true(k - 1);
                diff    = mod(diff + pi, 2*pi) - pi;
                th_curr = Theta_true(k - 1) + diff;
            end

            P_true(k, :)   = [X, Y, Z];
            V_true(k, :)   = [Vx, Vy, Vz];
            A_true(k, :)   = [Ax, Ay, Az_acc];
            Theta_true(k)  = th_curr;
        end

        % 写入车辆轨迹结构体
        v_name = sprintf('V%d', n);
        trajectories.(v_name).Time_true  = (0:dt_imu:t_end)';
        trajectories.(v_name).X_true     = P_true(:, 1);
        trajectories.(v_name).Y_true     = P_true(:, 2);
        trajectories.(v_name).Z_true     = P_true(:, 3);
        trajectories.(v_name).Vx_true    = V_true(:, 1);
        trajectories.(v_name).Vy_true    = V_true(:, 2);
        trajectories.(v_name).Vz_true    = V_true(:, 3);
        trajectories.(v_name).Theta_true = Theta_true;
        trajectories.(v_name).A_true     = A_true;

        % 生成对应的三维姿态旋转矩阵序列
        R_true = zeros(3, 3, N_steps);
        for k = 1:N_steps
            th = Theta_true(k);
            R_true(:, :, k) = [
                cos(th), -sin(th), 0;
                sin(th),  cos(th), 0;
                0,       0,        1
            ];
        end
        trajectories.(v_name).R_true = R_true;
    end
end