% generate_data_CMLKF_CPF.m
% 20x20m 近场场地、3D 俯仰/横滚机动编队 UWB/IMU 多批次仿真数据生成脚本
% 用于 CEKF / CIEKF / CMLKF / CPF 四算法对比
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
t_end         = 40;          % 每个批次的仿真时长 (s)
N_steps       = round(t_end / dt_imu) + 1;
repeat_time   = 1;           % 生成的独立批次数量
N_steps_total = N_steps * repeat_time;

% --- 实体数量配置 ---
Vehicle_num = 2;             % 车辆/无人机数量 [1, 18]
Anchor_num  = 12;            % UWB基站数量 [1, 30]
if Vehicle_num < 1 || Vehicle_num > 18, error('Vehicle_num 必须在 [1, 18] 范围内。'); end
if Anchor_num < 1 || Anchor_num > 30,   error('Anchor_num 必须在 [1, 30] 范围内。'); end
if repeat_time < 1 || round(repeat_time) ~= repeat_time, error('repeat_time 必须是正整数。'); end

% --- 文件标识与随机种子配置 ---
dataset_id = 'C_4_alg_40_4'; % 文件名后缀 ID
rng('shuffle');              
rng_state  = rng;
base_seed  = rng_state.Seed;

% --- 传感器噪声配置 ---
UWB_noise_params.sigma_anc = 0.01;
UWB_noise_params.sigma_rel = 0.01;

% IMU_noise_params.sigma_na  = 0.05;
% IMU_noise_params.sigma_nw  = 0.005;
% IMU_noise_params.sigma_ba  = 0.005;
% IMU_noise_params.sigma_bw  = 0.0005;

IMU_noise_params.sigma_na  = 0.25;
IMU_noise_params.sigma_nw  = 0.05;
IMU_noise_params.sigma_ba  = 0.05;
IMU_noise_params.sigma_bw  = 0.01;

% --- 初始状态先验误差标准差配置 ---
position_prior_std_xy  = 1;   % XY 轴初始位置偏移标准差 (m)
position_prior_std_z   = 1;   % Z 轴初始位置偏移标准差 (m)
velocity_prior_std     = 0.5;   % 初始速度偏移标准差 (m/s)
attitude_prior_std_deg = 10.0;  % 初始三轴姿态偏移标准差 (deg)

std_vec     = [position_prior_std_xy; position_prior_std_xy; position_prior_std_z];
att_std_rad = deg2rad(attitude_prior_std_deg);

%% ==================== 2. 环境与真值轨迹生成 ====================
[trajectory_template, anchors, truth_position] = env_setup_E3( ...
    Vehicle_num, Anchor_num, N_steps, dt_imu, t_end);

%% ==================== 3. 传感器噪声与初始误差多批次采样 ====================
trajectories            = struct();
anchors_all             = zeros(Anchor_num, 3, repeat_time);
initial_position_error  = zeros(3, Vehicle_num, repeat_time);
initial_velocity_error  = zeros(3, Vehicle_num, repeat_time);
initial_attitude_error  = zeros(3, Vehicle_num, repeat_time); % 李代数旋转向量误差 (rad)
batch_seeds             = double(base_seed) + (0:repeat_time-1);

for r = 1:repeat_time
    rng(batch_seeds(r), 'twister'); 
    anchors_all(:, :, r) = anchors;
    
    % 生成单批次的 IMU 和 UWB 测量
    batch_trajectories = generate_imu_uwb_batch( ...
        trajectory_template, truth_position, anchors, ...
        N_steps, dt_imu, uwb_downsample_factor, ...
        IMU_noise_params, UWB_noise_params);

    if r == 1
        trajectories = batch_trajectories;
    else
        trajectories = concatenate_trajectories(trajectories, batch_trajectories);
    end

    % 每个批次独立采样初始位置、速度与三维姿态偏差
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
figure('Name', 'Multi-Agent 20x20m 拓扑与 3D 机动轨迹', 'Position', [100, 100, 900, 700]);
hold on; grid on; axis equal;
xlabel('X 轴位置 (m)'); ylabel('Y 轴位置 (m)'); zlabel('高度 Z (m)');
title(sprintf('30x30m 场地：%d机 3D 波浪机动编队 与 %d 个近场基站', Vehicle_num, Anchor_num));

% 绘制 20x20x8 的立体边界框
X_max = 20; Y_max = 20; Z_max = 8;
line([0, X_max, X_max, 0, 0], [0, 0, Y_max, Y_max, 0], [0, 0, 0, 0, 0], 'Color', [0.5,0.5,0.5], 'LineStyle', '--');
line([0, X_max, X_max, 0, 0], [0, 0, Y_max, Y_max, 0], [Z_max, Z_max, Z_max, Z_max, Z_max], 'Color', [0.5,0.5,0.5], 'LineStyle', '--');
for corner = [0, X_max]
    for side = [0, Y_max]
        line([corner, corner], [side, side], [0, Z_max], 'Color', [0.5,0.5,0.5], 'LineStyle', '--');
    end
end

h_anchor = plot3(anchors(:,1), anchors(:,2), anchors(:,3), '^', 'MarkerSize', 10, ...
    'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, 'DisplayName', 'UWB 基站');

colors = lines(Vehicle_num);
h_traj = zeros(1, Vehicle_num);
for n = 1:Vehicle_num
    v_data = trajectories.(sprintf('V%d', n));
    X_plot = v_data.X_true(1:N_steps);
    Y_plot = v_data.Y_true(1:N_steps);
    Z_plot = v_data.Z_true(1:N_steps);
    
    h_traj(n) = plot3(X_plot, Y_plot, Z_plot, 'Color', colors(n,:), 'LineWidth', 2, ...
        'DisplayName', sprintf('无人机 V%d', n));
    plot3(X_plot(1), Y_plot(1), Z_plot(1), 'o', 'MarkerSize', 6, 'MarkerFaceColor', colors(n,:), 'Color', colors(n,:));
    plot3(X_plot(end), Y_plot(end), Z_plot(end), '*', 'MarkerSize', 8, 'Color', colors(n,:));
end
view(3); 
legend([h_anchor, h_traj(1)], 'Location', 'northeastoutside');
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
fprintf('30x30m 近场 3D 机动多批次仿真数据生成成功！\n');
fprintf('文件后缀 ID: %s\n', dataset_id);
fprintf('车辆数: %d，基站数: %d\n', Vehicle_num, Anchor_num);
fprintf('初始误差标准差: Pos=[%.2f, %.2f, %.2f]m, Vel=%.2fm/s, Att=%.1fdeg\n', ...
    std_vec(1), std_vec(2), std_vec(3), velocity_prior_std, attitude_prior_std_deg);
fprintf('UWB测距噪声: %.3f m\n', UWB_noise_params.sigma_anc);
fprintf('文件已保存至: %s\n', save_path);
fprintf('======================================================\n\n');


%% ======================= 局部函数区域 =======================

function batch = generate_imu_uwb_batch(trajectory_template, ...
        truth_position, anchors, N_steps, dt_imu, ...
        uwb_downsample_factor, IMU_noise_params, UWB_noise_params)
    
    Vehicle_num = size(truth_position, 3);
    Anchor_num  = size(anchors, 1);
    batch       = trajectory_template;
    g_vec       = [0; 0; -9.81];

    % 1. 生成 3D IMU 测量 (100Hz，严格由流形旋转真值 R_true 反解机体系角速度与比力)
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
        veh.IMU_acc_m       = a_body_ideal   + b_a + randn(N_steps, 3) * IMU_noise_params.sigma_na;
        veh.IMU_gyro_m      = omega_body_ideal + b_w + randn(N_steps, 3) * IMU_noise_params.sigma_nw;
        veh.IMU_bias_a_true = b_a;
        veh.IMU_bias_w_true = b_w;
        batch.(v_name)      = veh;
    end

    % 2. 生成 UWB 测距 (10Hz)
    idx_uwb      = 1:uwb_downsample_factor:N_steps;
    t_uwb        = trajectory_template.V1.Time_true(idx_uwb);
    N_uwb        = length(t_uwb);
    pos_true_uwb = truth_position(idx_uwb, :, :);

    for n = 1:Vehicle_num
        v_name = sprintf('V%d', n);

        % 基站测距
        UWB_Anchor = zeros(N_uwb, 1 + Anchor_num);
        UWB_Anchor(:, 1) = t_uwb;
        for a_idx = 1:Anchor_num
            delta     = squeeze(pos_true_uwb(:, :, n)) - anchors(a_idx, :);
            dist_true = sqrt(sum(delta.^2, 2));
            UWB_Anchor(:, 1 + a_idx) = dist_true + randn(N_uwb, 1) * UWB_noise_params.sigma_anc;
        end
        batch.(v_name).UWB_Anchor = UWB_Anchor;

        % 相对测距
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

function [trajectories, anchors, truth_position] = env_setup_E3( ...
        Vehicle_num, Anchor_num, N_steps, dt_imu, t_end)
    % 20x20场地
    
    center_xy         = 10.0;  % [修改] 20x20m 场地中心点坐标 (10, 10)
    formation_spacing = 1.2;   % [修改] 相应缩小水平编队间距
    nominal_height    = 4.0;   % 保持不变
    vehicle_z_step    = 0.35;  % 保持不变
    orbit_radius      = 5.0;   % [修改] 编队中心主轨道半径 (m)
    radius_amplitude  = 0.8;   % [修改] 轨道半径周期呼吸幅度 (m)
    radius_rate       = 0.18;  % 保持不变
    angular_rate      = 0.35;  % 保持不变
    z_amplitude       = 1.2;   % 保持不变
    z_rate            = 0.45;  % 保持不变

    % 三维姿态机动幅度 (打破水平零倾角退化区)
    % roll_max_rad      = deg2rad(28); % 最大横滚角 28 deg
    % pitch_max_rad     = deg2rad(24); % 最大俯仰角 24 deg
    roll_max_rad      = deg2rad(70); % 最大横滚角 50 deg
    pitch_max_rad     = deg2rad(80); % 最大俯仰角 60 deg

    % 显式定义 20x20m 场地内的 30 个固定基站池
    all_anchors_pool = [
        % --- 1~4: 外围四角大包围 ---
          0.0,   0.0,  1.0;  %  1: 左下角
         20.0,  20.0,  2.0;  %  2: 右上角
          0.0,  20.0,  1.3;  %  3: 左上角
         20.0,   0.0,  1.7;  %  4: 右下角
        % --- 5~9: 场地中心与四边线中点 ---
         10.0,  10.0,  1.9;  %  5: 场地正中央 (近场核心点)
         10.0,   0.0,  1.1;  %  6: 下边线中点
          0.0,  10.0,  1.8;  %  7: 左边线中点
         20.0,  10.0,  1.2;  %  8: 右边线中点
         10.0,  20.0,  1.6;  %  9: 上边线中点
        % --- 10~13: 内圈四象限中心 ---
          5.0,  15.0,  1.1;  % 10: 左上内象限
         15.0,   5.0,  1.9;  % 11: 右下内象限
          5.0,   5.0,  1.8;  % 12: 左下内象限
         15.0,  15.0,  1.2;  % 13: 右上内象限
        % --- 14~17: 内圈十字正交点 ---
         10.0,   5.0,  2.0;  % 14: 正南轨道点
          5.0,  10.0,  1.0;  % 15: 正西轨道点
         15.0,  10.0,  1.7;  % 16: 正东轨道点
         10.0,  15.0,  1.3;  % 17: 正北轨道点
        % --- 18~21: 外边界四分对称点 (第一组) ---
          5.0,   0.0,  1.6;  % 18: 下边左四分点
         15.0,  20.0,  1.4;  % 19: 上边右四分点
          0.0,   5.0,  1.2;  % 20: 左边下四分点
         20.0,  15.0,  1.8;  % 21: 右边上四分点
        % --- 22~25: 外边界四分对称点 (第二组) ---
         15.0,   0.0,  1.3;  % 22: 下边右四分点
          5.0,  20.0,  1.7;  % 23: 上边左四分点
          0.0,  15.0,  1.9;  % 24: 左边上四分点
         20.0,   5.0,  1.1;  % 25: 右边下四分点
        % --- 26~30: 核心轨道内侧近场加密环 ---
          7.5,   7.5,  1.4;  % 26: 内侧左下
         12.5,  12.5,  1.6;  % 27: 内侧右上
          7.5,  12.5,  2.0;  % 28: 内侧左上
         12.5,   7.5,  1.0;  % 29: 内侧右下
         10.0,   7.5,  1.5;  % 30: 内侧正南
    ];

    anchors = all_anchors_pool(1:Anchor_num, :);

    % 编队偏移量分配 (支持 1 ~ 18 辆车)
    n_columns       = min(4, max(1, ceil(sqrt(Vehicle_num))));
    n_rows          = ceil(Vehicle_num / n_columns);
    offsets_tangent = zeros(Vehicle_num, 1);
    offsets_radial  = zeros(Vehicle_num, 1);
    offsets_z       = zeros(Vehicle_num, 1);
    for n = 1:Vehicle_num
        column = mod(n - 1, n_columns);
        row    = floor((n - 1) / n_columns);
        offsets_tangent(n) = (column - (n_columns - 1)/2) * formation_spacing;
        offsets_radial(n)  = (row - (n_rows - 1)/2) * formation_spacing;
        offsets_z(n)       = vehicle_z_step * (mod(n - 1, 5) - 2);
    end

    time           = (0:N_steps-1)' * dt_imu;
    trajectories   = struct();
    truth_position = zeros(N_steps, 3, Vehicle_num);

    for n = 1:Vehicle_num
        P_true     = zeros(N_steps, 3);
        V_true     = zeros(N_steps, 3);
        A_true     = zeros(N_steps, 3);
        Theta_true = zeros(N_steps, 1);
        R_true     = zeros(3, 3, N_steps);
        
        o_t     = offsets_tangent(n);
        o_r     = offsets_radial(n);
        o_z     = offsets_z(n);
        z_phase = 2*pi*(n - 1) / max(1, Vehicle_num);

        for k = 1:N_steps
            t     = time(k);
            theta = angular_rate * t + 0.3;
            er    = [cos(theta); sin(theta); 0];
            et    = [-sin(theta); cos(theta); 0];

            radius      = orbit_radius + radius_amplitude * sin(radius_rate * t);
            radius_dot  = radius_amplitude * radius_rate * cos(radius_rate * t);
            radius_ddot = -radius_amplitude * radius_rate^2 * sin(radius_rate * t);

            center = [center_xy + radius * cos(theta); ...
                      center_xy + radius * sin(theta); ...
                      nominal_height + z_amplitude * sin(z_rate * t + z_phase) + o_z];
            
            center_v = [radius_dot * cos(theta) - radius * angular_rate * sin(theta); ...
                        radius_dot * sin(theta) + radius * angular_rate * cos(theta); ...
                        z_amplitude * z_rate * cos(z_rate * t + z_phase)];
                    
            center_a = [(radius_ddot - radius * angular_rate^2) * cos(theta) - 2 * radius_dot * angular_rate * sin(theta); ...
                        (radius_ddot - radius * angular_rate^2) * sin(theta) + 2 * radius_dot * angular_rate * cos(theta); ...
                        -z_amplitude * z_rate^2 * sin(z_rate * t + z_phase)];

            offset   = o_t * et + o_r * er;
            offset_v = -o_t * angular_rate * er + o_r * angular_rate * et;
            offset_a = -angular_rate^2 * offset;

            P_true(k, :)  = (center + offset)';
            V_true(k, :)  = (center_v + offset_v)';
            A_true(k, :)  = (center_a + offset_a)';
            yaw_k         = theta + pi/2;
            Theta_true(k) = yaw_k;

            % 生成真实的 3D 姿态旋转矩阵 R_true (含显著的 Roll 与 Pitch 动态变化)
            roll_k  = roll_max_rad  * sin(0.7 * t + z_phase);
            pitch_k = pitch_max_rad * cos(z_rate * t + z_phase);
            R_true(:, :, k) = euler_to_rot_mat(roll_k, pitch_k, yaw_k);
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
        trajectories.(v_name).Theta_true = Theta_true;
        trajectories.(v_name).A_true     = A_true;
        trajectories.(v_name).R_true     = R_true;
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