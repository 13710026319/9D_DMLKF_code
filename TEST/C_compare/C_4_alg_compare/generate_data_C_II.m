% generate_data_C_II.m
% 20x20m 封闭环境周界基站（四周布站、中间无基站）+ "三段直线 + 两处 90度平滑转弯" 真实巡航轨迹生成脚本
% 用于 CEKF / CIEKF / CMLKF / CPF 四算法对比及论文轨迹可视化截图
clc; clear; close all;

%% ==================== 1. 全局参数与环境配置 ====================
% --- 采样率配置 ---
F_imu  = 100;
F_uwb  = 10;
dt_imu = 1 / F_imu;
dt_uwb = 1 / F_uwb;
uwb_downsample_factor = round(dt_uwb / dt_imu);

if mod(dt_uwb, dt_imu) ~= 0
    error('UWB 采样周期必须是 IMU 采样周期的整数倍！');
end

% --- 仿真时间与批次配置 ---
t_end         = 40;          % 固定每个批次仿真时长 40s
N_steps       = round(t_end / dt_imu) + 1;
repeat_time   = 3;           % 生成的独立批次数量
N_steps_total = N_steps * repeat_time;

% --- 实体数量配置 ---
Vehicle_num = 1;             % 车辆/无人机数量 [1, 18]
Anchor_num  = 12;            % UWB基站数量 [1, 30]
if Vehicle_num < 1 || Vehicle_num > 18, error('Vehicle_num 必须在 [1, 18] 范围内。'); end
if Anchor_num < 1 || Anchor_num > 30,   error('Anchor_num 必须在 [1, 30] 范围内。'); end
if repeat_time < 1 || round(repeat_time) ~= repeat_time, error('repeat_time 必须是正整数。'); end

% --- 文件标识与随机种子配置 ---
dataset_id = 'C_4_alg_40_II'; % 文件名后缀 ID (可直接供 compare_C_4_alg 读取)
rng('shuffle');              
rng_state  = rng;
base_seed  = rng_state.Seed;

% --- 传感器噪声配置 ---
UWB_noise_params.sigma_anc = 0.01;
UWB_noise_params.sigma_rel = 0.01;

IMU_noise_params.sigma_na  = 0.25;
IMU_noise_params.sigma_nw  = 0.05;
IMU_noise_params.sigma_ba  = 0.05;
IMU_noise_params.sigma_bw  = 0.01;

% --- 初始状态先验误差标准差配置 ---
position_prior_std_xy  = 1.5;   % XY 轴初始位置偏移标准差 (m)
position_prior_std_z   = 1.0;   % Z 轴初始位置偏移标准差 (m)
velocity_prior_std     = 0.5;   % 初始速度偏移标准差 (m/s)
attitude_prior_std_deg = 20.0;  % 初始三轴姿态偏移标准差 (deg)

std_vec     = [position_prior_std_xy; position_prior_std_xy; position_prior_std_z];
att_std_rad = deg2rad(attitude_prior_std_deg);

%% ==================== 2. 环境与真值轨迹生成 ====================
[trajectory_template, anchors, truth_position] = env_setup_corridor( ...
    Vehicle_num, Anchor_num, N_steps, dt_imu, t_end);

%% ==================== 3. 传感器噪声与初始误差多批次采样 ====================
trajectories           = struct();
anchors_all            = zeros(Anchor_num, 3, repeat_time);
initial_position_error = zeros(3, Vehicle_num, repeat_time);
initial_velocity_error = zeros(3, Vehicle_num, repeat_time);
initial_attitude_error = zeros(3, Vehicle_num, repeat_time);
batch_seeds            = double(base_seed) + (0:repeat_time-1);

for r = 1:repeat_time
    rng(batch_seeds(r), 'twister'); 
    anchors_all(:, :, r) = anchors;
    
    batch_trajectories = generate_imu_uwb_batch( ...
        trajectory_template, truth_position, anchors, ...
        N_steps, dt_imu, uwb_downsample_factor, ...
        IMU_noise_params, UWB_noise_params);

    if r == 1
        trajectories = batch_trajectories;
    else
        trajectories = concatenate_trajectories(trajectories, batch_trajectories);
    end

    initial_position_error(:, :, r) = std_vec .* randn(3, Vehicle_num);
    initial_velocity_error(:, :, r) = velocity_prior_std * randn(3, Vehicle_num);
    initial_attitude_error(:, :, r) = att_std_rad * randn(3, Vehicle_num);
end

% 组装场景参数字典
scenario.position_prior_std_vec = std_vec;
scenario.velocity_prior_std     = velocity_prior_std;
scenario.attitude_prior_std_deg = attitude_prior_std_deg;
scenario.initial_position_error = initial_position_error;
scenario.initial_velocity_error = initial_velocity_error;
scenario.initial_attitude_error = initial_attitude_error;
scenario.dataset_id             = dataset_id;
scenario.base_seed              = double(base_seed);
scenario.batch_seeds            = double(batch_seeds);

%% ==================== 4. 3D 轨迹可视化 ====================
figure('Name', 'Multi-Agent 室内周界基站与巡航轨迹可视化', 'Position', [100, 100, 880, 680], 'Color', 'w');
hold on; grid on; axis equal;
xlabel('X 轴位置 (m)'); ylabel('Y 轴位置 (m)'); zlabel('高度 Z (m)');
title(sprintf('20x20m 封闭场景：%d机 直线+两处转弯巡航轨迹 与 %d 个周界基站', Vehicle_num, Anchor_num));

% 绘制 20x20x6 的立体边界框
X_max = 20; Y_max = 20; Z_max = 6;
line([0, X_max, X_max, 0, 0], [0, 0, Y_max, Y_max, 0], [0, 0, 0, 0, 0], 'Color', [0.5,0.5,0.5], 'LineStyle', '--');
line([0, X_max, X_max, 0, 0], [0, 0, Y_max, Y_max, 0], [Z_max, Z_max, Z_max, Z_max, Z_max], 'Color', [0.5,0.5,0.5], 'LineStyle', '--');
for corner = [0, X_max]
    for side = [0, Y_max]
        line([corner, corner], [side, side], [0, Z_max], 'Color', [0.5,0.5,0.5], 'LineStyle', '--');
    end
end

% 绘制四周墙壁基站
h_anchor = plot3(anchors(:,1), anchors(:,2), anchors(:,3), '^', 'MarkerSize', 10, ...
    'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, 'DisplayName', 'UWB 基站');

% 绘制真值轨迹 (第一批次)
colors = lines(Vehicle_num);
h_traj = zeros(1, Vehicle_num);
for n = 1:Vehicle_num
    v_data = trajectories.(sprintf('V%d', n));
    X_plot = v_data.X_true(1:N_steps);
    Y_plot = v_data.Y_true(1:N_steps);
    Z_plot = v_data.Z_true(1:N_steps);
    
    h_traj(n) = plot3(X_plot, Y_plot, Z_plot, 'Color', colors(n,:), 'LineWidth', 2.2, ...
        'DisplayName', sprintf('车辆 V%d', n));
    plot3(X_plot(1), Y_plot(1), Z_plot(1), 'o', 'MarkerSize', 7, 'MarkerFaceColor', colors(n,:), 'Color', 'k');
    plot3(X_plot(end), Y_plot(end), Z_plot(end), 'p', 'MarkerSize', 9, 'MarkerFaceColor', colors(n,:), 'Color', 'k');
end
xlim([-1, 21]); ylim([-1, 21]); zlim([0, 6]);
view(-35, 28);
legend([h_anchor, h_traj], 'Location', 'northeastoutside');
hold off;

%% ==================== 5. 数据保存 ====================
save_dir = fullfile(fileparts(mfilename('fullpath')), 'data');
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

trajectories_mat_name = sprintf('Trj_Veh%d_Anc%d_%s.mat', Vehicle_num, Anchor_num, dataset_id);
save_path = fullfile(save_dir, trajectories_mat_name);

save(save_path, 'trajectories', 'anchors', 'anchors_all', ...
    'IMU_noise_params', 'UWB_noise_params', ...
    'Vehicle_num', 'Anchor_num', 't_end', ...
    'N_steps', 'N_steps_total', 'repeat_time', ...
    'uwb_downsample_factor', 'scenario');

fprintf('\n======================================================\n');
fprintf('20x20m 周界基站 + 两处转弯自然轨迹数据生成成功！\n');
fprintf('文件后缀 ID: %s\n', dataset_id);
fprintf('车辆数: %d，周界基站数: %d\n', Vehicle_num, Anchor_num);
fprintf('初始误差标准差: Pos=[%.2f, %.2f, %.2f]m, Vel=%.2fm/s, Att=%.1fdeg\n', ...
    std_vec(1), std_vec(2), std_vec(3), velocity_prior_std, attitude_prior_std_deg);
fprintf('UWB测距噪声: %.3f m\n', UWB_noise_params.sigma_anc);
fprintf('文件已保存至: %s\n', save_path);
fprintf('======================================================\n\n');


%% ======================= 局部函数区域 =======================

function [trajectories, anchors, truth_position] = env_setup_corridor( ...
        Vehicle_num, Anchor_num, N_steps, dt_imu, t_end)
    % 20x20m 封闭室内环境：
    % 1) 30个基站严格部署在四周外墙 (X=0, X=20, Y=0, Y=20)，场地中间无任何基站；
    % 2) 轨迹为自然规整的 "三段直线 + 两处 90度平滑圆弧转弯" (U型室内巡航走廊)，伴随平缓高度起伏。

    % --- 1. 显式定义四周外墙上的 30 个周界基站坐标池 (高度限制在 1.0m ~ 2.0m) ---
    all_anchors_pool = [
        % 1~4: 四面墙顶角 (4个角点)
          0.0,   0.0,  1.0;  %  1: 左下角 (南墙-西墙交点)
         20.0,  20.0,  2.0;  %  2: 右上角 (北墙-东墙交点)
          0.0,  20.0,  1.3;  %  3: 左上角 (北墙-西墙交点)
         20.0,   0.0,  1.7;  %  4: 右下角 (南墙-东墙交点)
        % 5~8: 四面墙正中心点
         10.0,   0.0,  1.9;  %  5: 南墙中点
         20.0,  10.0,  1.1;  %  6: 东墙中点
         10.0,  20.0,  1.8;  %  7: 北墙中点
          0.0,  10.0,  1.2;  %  8: 西墙中点
        % 9~12: 四面墙 1/4 分位点 (第一组对称周界点)
          5.0,   0.0,  1.4;  %  9: 南墙左四分点
         20.0,   5.0,  1.8;  % 10: 东墙下四分点
         15.0,  20.0,  1.2;  % 11: 北墙右四分点
          0.0,  15.0,  1.6;  % 12: 西墙上四分点
        % 13~16: 四面墙 3/4 分位点 (第二组对称周界点)
         15.0,   0.0,  1.2;  % 13: 南墙右四分点
         20.0,  15.0,  1.6;  % 14: 东墙上四分点
          5.0,  20.0,  1.9;  % 15: 北墙左四分点
          0.0,   5.0,  1.1;  % 16: 西墙下四分点
        % 17~20: 四面墙 1/8 分位点
          2.5,   0.0,  1.7;  % 17: 南墙 2.5m 处
         20.0,   2.5,  1.3;  % 18: 东墙 2.5m 处
         17.5,  20.0,  1.5;  % 19: 北墙 17.5m 处
          0.0,  17.5,  1.9;  % 20: 西墙 17.5m 处
        % 21~24: 四面墙 3/8 分位点
          7.5,   0.0,  1.1;  % 21: 南墙 7.5m 处
         20.0,   7.5,  1.9;  % 22: 东墙 7.5m 处
         12.5,  20.0,  1.4;  % 23: 北墙 12.5m 处
          0.0,  12.5,  1.7;  % 24: 西墙 12.5m 处
        % 25~28: 四面墙 5/8 分位点
         12.5,   0.0,  1.6;  % 25: 南墙 12.5m 处
         20.0,  12.5,  1.2;  % 26: 东墙 12.5m 处
          7.5,  20.0,  1.1;  % 27: 北墙 7.5m 处
          0.0,   7.5,  1.8;  % 28: 西墙 7.5m 处
        % 29~30: 南北墙 7/8 对称收尾点
         17.5,   0.0,  1.5;  % 29: 南墙 17.5m 处
          2.5,  20.0,  1.5;  % 30: 北墙 2.5m 处
    ];
    anchors = all_anchors_pool(1:Anchor_num, :);

    % --- 2. 生成基准中心轨迹 (40s 内严格 3 段直线 + 2 处 90度平滑转弯) ---
    % 时间分段 (总长 40s):
    %   [0,  11]s : 第 1 段直线 (向东行驶)
    %   [11, 17]s : 第 1 处 90° 平滑左转弯 (由东转向北)
    %   [17, 23]s : 第 2 段直线 (向北行驶)
    %   [23, 29]s : 第 2 处 90° 平滑左转弯 (由北转向西)
    %   [29, 40]s : 第 3 段直线 (向西行驶)
    time = (0:N_steps-1)' * dt_imu;
    v_cruise = 0.72; % 巡航线速度 (m/s)
    t1_s = 11.0; t1_e = 17.0; dT1 = t1_e - t1_s;
    t2_s = 23.0; t2_e = 29.0; dT2 = t2_e - t2_s;

    psi_c      = zeros(N_steps, 1);
    psi_dot_c  = zeros(N_steps, 1);
    psi_ddot_c = zeros(N_steps, 1);

    for k = 1:N_steps
        t = time(k);
        if t <= t1_s
            % 第 1 段直线 (航向 0 rad)
            psi_c(k)      = 0;
            psi_dot_c(k)  = 0;
            psi_ddot_c(k) = 0;
        elseif t <= t1_e
            % 第 1 处 90° 转弯 (余弦平滑过渡，C^2 连续)
            tau_1         = (t - t1_s) / dT1;
            psi_c(k)      = (pi/2) * (tau_1 - sin(2*pi*tau_1) / (2*pi));
            psi_dot_c(k)  = (pi / (2*dT1)) * (1 - cos(2*pi*tau_1));
            psi_ddot_c(k) = (pi^2 / (dT1^2)) * sin(2*pi*tau_1);
        elseif t <= t2_s
            % 第 2 段直线 (航向 pi/2 rad)
            psi_c(k)      = pi/2;
            psi_dot_c(k)  = 0;
            psi_ddot_c(k) = 0;
        elseif t <= t2_e
            % 第 2 处 90° 转弯 (由 pi/2 平滑转至 pi)
            tau_2         = (t - t2_s) / dT2;
            psi_c(k)      = pi/2 + (pi/2) * (tau_2 - sin(2*pi*tau_2) / (2*pi));
            psi_dot_c(k)  = (pi / (2*dT2)) * (1 - cos(2*pi*tau_2));
            psi_ddot_c(k) = (pi^2 / (dT2^2)) * sin(2*pi*tau_2);
        else
            % 第 3 段直线 (航向 pi rad)
            psi_c(k)      = pi;
            psi_dot_c(k)  = 0;
            psi_ddot_c(k) = 0;
        end
    end

    % 基准高度平滑拱形变化 (从 2.8m 平缓升至 4.0m 再回到 2.8m)
    z_base   = 2.8;
    z_arch   = 1.2;
    w_z      = pi / t_end;
    Z_c      = z_base + z_arch * sin(w_z * time);
    Vz_c     = z_arch * w_z * cos(w_z * time);
    Az_c     = -z_arch * (w_z^2) * sin(w_z * time);

    % 精确积分中心线水平位置 (起点设在场地左下内部 (4.2, 4.2))
    Vx_c = v_cruise * cos(psi_c);
    Vy_c = v_cruise * sin(psi_c);
    Ax_c = -v_cruise * sin(psi_c) .* psi_dot_c;
    Ay_c =  v_cruise * cos(psi_c) .* psi_dot_c;

    X_c = zeros(N_steps, 1);
    Y_c = zeros(N_steps, 1);
    X_c(1) = 4.2;
    Y_c(1) = 4.2;
    for k = 1:N_steps-1
        X_c(k+1) = X_c(k) + Vx_c(k) * dt_imu + 0.5 * Ax_c(k) * (dt_imu^2);
        Y_c(k+1) = Y_c(k) + Vy_c(k) * dt_imu + 0.5 * Ay_c(k) * (dt_imu^2);
    end

    % --- 3. 多车平行编队偏移与内部三维姿态生成 (支持 1 ~ 18 辆车) ---
    % 采用固定全局偏移，使每辆车的轨迹都是完全平行的 "三段直线 + 两处 90°转弯"，没有任何交叉打结
    n_cols  = min(3, max(1, ceil(sqrt(Vehicle_num))));
    n_rows  = ceil(Vehicle_num / n_cols);
    spacing_xy = 1.1;
    spacing_z  = 0.30;

    roll_max_rad  = deg2rad(28); % 内部横滚角变化幅度 (用于激发 SO3 vs 欧拉角差异)
    pitch_max_rad = deg2rad(70); % 内部俯仰角变化幅度

    trajectories   = struct();
    truth_position = zeros(N_steps, 3, Vehicle_num);

    for n = 1:Vehicle_num
        col_idx = mod(n - 1, n_cols);
        row_idx = floor((n - 1) / n_cols);
        dx_n = (col_idx - (n_cols - 1)/2) * spacing_xy;
        dy_n = (row_idx - (n_rows - 1)/2) * spacing_xy;
        dz_n = (mod(n - 1, 3) - 1) * spacing_z;

        P_true = [X_c + dx_n, Y_c + dy_n, Z_c + dz_n];
        V_true = [Vx_c,       Vy_c,       Vz_c];
        A_true = [Ax_c,       Ay_c,       Az_c];

        R_true = zeros(3, 3, N_steps);
        phase_n = 2*pi*(n - 1) / max(1, Vehicle_num);
        for k = 1:N_steps
            t = time(k);
            roll_k  = roll_max_rad  * sin(0.7 * t + phase_n);
            pitch_k = pitch_max_rad * cos(0.45 * t + phase_n);
            R_true(:, :, k) = euler_to_rot_mat(roll_k, pitch_k, psi_c(k));
        end

        truth_position(:, :, n) = P_true;
        v_name = sprintf('V%d', n);
        trajectories.(v_name).Time_true  = time;
        trajectories.(v_name).X_true     = P_true(:, 1);
        trajectories.(v_name).Y_true     = P_true(:, 2);
        trajectories.(v_name).Z_true     = P_true(:, 3);
        trajectories.(v_name).Vx_true    = V_true(:, 1);
        trajectories.(v_name).Vy_true    = V_true(:, 2);
        trajectories.(v_name).Vz_true    = V_true(:, 3);
        trajectories.(v_name).Theta_true = psi_c;
        trajectories.(v_name).A_true     = A_true;
        trajectories.(v_name).R_true     = R_true;
    end
end

function batch = generate_imu_uwb_batch(trajectory_template, ...
        truth_position, anchors, N_steps, dt_imu, ...
        uwb_downsample_factor, IMU_noise_params, UWB_noise_params)
    
    Vehicle_num = size(truth_position, 3);
    Anchor_num  = size(anchors, 1);
    batch       = trajectory_template;
    g_vec       = [0; 0; -9.81];

    % 1. 生成 100Hz IMU 测量 (严格由流形旋转真值 R_true 反解机体系角速度与比力)
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
                R_next = veh.R_true(:, :, k+1);
                dR     = R_k' * R_next;
                omega_body_ideal(k, :) = (log_SO3_vec(dR) / dt_imu)';
            else
                omega_body_ideal(k, :) = omega_body_ideal(k-1, :);
            end
        end

        ba_init = (rand(1, 3) - 0.5) * 0.1;
        bw_init = (rand(1, 3) - 0.5) * 0.01;
        b_a = ba_init + cumsum(randn(N_steps, 3) * IMU_noise_params.sigma_ba * sqrt(dt_imu), 1);
        b_w = bw_init + cumsum(randn(N_steps, 3) * IMU_noise_params.sigma_bw * sqrt(dt_imu), 1);

        veh.IMU_Time        = veh.Time_true;
        veh.IMU_acc_m       = a_body_ideal    + b_a + randn(N_steps, 3) * IMU_noise_params.sigma_na;
        veh.IMU_gyro_m      = omega_body_ideal + b_w + randn(N_steps, 3) * IMU_noise_params.sigma_nw;
        veh.IMU_bias_a_true = b_a;
        veh.IMU_bias_w_true = b_w;
        batch.(v_name)      = veh;
    end

    % 2. 生成 10Hz UWB 测距
    idx_uwb      = 1:uwb_downsample_factor:N_steps;
    t_uwb        = trajectory_template.V1.Time_true(idx_uwb);
    N_uwb        = length(t_uwb);
    pos_true_uwb = truth_position(idx_uwb, :, :);

    for n = 1:Vehicle_num
        v_name = sprintf('V%d', n);

        UWB_Anchor = zeros(N_uwb, 1 + Anchor_num);
        UWB_Anchor(:, 1) = t_uwb;
        for a_idx = 1:Anchor_num
            delta     = squeeze(pos_true_uwb(:, :, n)) - anchors(a_idx, :);
            dist_true = sqrt(sum(delta.^2, 2));
            UWB_Anchor(:, 1 + a_idx) = dist_true + randn(N_uwb, 1) * UWB_noise_params.sigma_anc;
        end
        batch.(v_name).UWB_Anchor = UWB_Anchor;

        UWB_Relative = zeros(N_uwb, 1 + Vehicle_num);
        UWB_Relative(:, 1) = t_uwb;
        for j = 1:Vehicle_num
            if n == j
                UWB_Relative(:, 1 + j) = NaN;
            else
                delta     = squeeze(pos_true_uwb(:, :, n)) - squeeze(pos_true_uwb(:, :, j));
                dist_true = sqrt(sum(delta.^2, 2));
                UWB_Relative(:, 1 + j) = dist_true + randn(N_uwb, 1) * UWB_noise_params.sigma_rel;
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
                    total.(v_name).(field_name), batch.(v_name).(field_name));
            else
                total.(v_name).(field_name) = cat(1, ...
                    total.(v_name).(field_name), batch.(v_name).(field_name));
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