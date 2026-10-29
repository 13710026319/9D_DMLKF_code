% generate_data_CEKF_CMLKF.m
% 近共面基站、近距离编队 UWB/IMU 多批次仿真数据生成脚本
% 用于CMLKF VS CEKF 的数据集生成脚本，侧重于严重非线性场景+初始状态未校准（先验差）
clc; clear; close all;

%% ==================== 1. 全局参数与环境配置 ====================
% --- 采样率配置 ---
F_imu = 100;
F_uwb = 10;
dt_imu = 1 / F_imu;
dt_uwb = 1 / F_uwb;
uwb_downsample_factor = round(dt_uwb / dt_imu);

if mod(dt_uwb, dt_imu) ~= 0
    error('UWB 采样周期必须是 IMU 采样周期的整数倍！');
end

% --- 仿真时间与批次配置 ---
t_end = 30;                  % 每个批次的仿真时长 (s)
N_steps = round(t_end / dt_imu) + 1;
repeat_time = 5;             % 生成的独立批次数量
N_steps_total = N_steps * repeat_time;

% --- 文件标识与随机种子配置 (核心修改点) ---
dataset_id = 'CEKF_CMLKF_CIEKF_30_1';          % 文件名后缀 ID，用于标识数据集

% --- 实体数量配置 ---
Vehicle_num = 4;             % 车辆/无人机数量 [4, 12]
Anchor_num  = 12;            % UWB基站数量 [4, 20]
% if Vehicle_num < 4 || Vehicle_num > 12, error('Vehicle_num 必须在 [4, 12] 范围内。'); end
% if Anchor_num < 4 || Anchor_num > 20,   error('Anchor_num 必须在 [4, 20] 范围内。'); end
if repeat_time < 1 || round(repeat_time) ~= repeat_time, error('repeat_time 必须是正整数。'); end


% 使用基于时间的随机种子，确保每次运行哪怕 dataset_id 相同，生成的数据也完全不同
rng('shuffle');              
rng_state = rng;
base_seed = rng_state.Seed;  % 记录本次运行的真实基准种子，以便存入数据中追溯

% --- 噪声与先验误差配置 ---
UWB_noise_params.sigma_anc = 0.01;
UWB_noise_params.sigma_rel = 0.01;

IMU_noise_params.sigma_na = 0.05;
IMU_noise_params.sigma_nw = 0.005;
IMU_noise_params.sigma_ba = 0.005;
IMU_noise_params.sigma_bw = 0.0005;

% XY和Z轴的初始偏差标准差，会给初始位置注入该标准差下的随机位置偏移，并以此标准差构建初始状态协方差矩阵
position_prior_std_xy = 2;
position_prior_std_z  = 1.5;

std_vec = [position_prior_std_xy; position_prior_std_xy; position_prior_std_z];

%% ==================== 2. 环境与真值轨迹生成 ====================
% 提示：真值轨迹和基站几何在所有批次中保持一致，仅生成一次
[trajectory_template, anchors, truth_position] = env_setup_E3( ...
    Vehicle_num, Anchor_num, N_steps, dt_imu, t_end);

%% ==================== 3. 传感器噪声多批次采样 ====================
trajectories = struct();
anchors_all  = zeros(Anchor_num, 3, repeat_time);
initial_position_error = zeros(3, Vehicle_num, repeat_time);
standardized_initial_position_error = zeros(3, Vehicle_num, repeat_time);
batch_seeds = double(base_seed) + (0:repeat_time-1); % 强制转换为 double，避免类型冲突

for r = 1:repeat_time
    % 为当前批次设置独立种子，确保不同批次数据不同
    rng(batch_seeds(r), 'twister'); 
    anchors_all(:, :, r) = anchors;
    
    % 生成单批次的 IMU 和 UWB 测量
    batch_trajectories = generate_imu_uwb_batch( ...
        trajectory_template, truth_position, anchors, ...
        N_steps, dt_imu, uwb_downsample_factor, ...
        IMU_noise_params, UWB_noise_params);

    % 数据拼接
    if r == 1
        trajectories = batch_trajectories;
    else
        trajectories = concatenate_trajectories(trajectories, batch_trajectories);
    end

    % 每个批次独立采样宽而一致的初始位置误差
    standardized_initial_position_error(:, :, r) = randn(3, Vehicle_num);
    initial_position_error(:, :, r) = std_vec .* standardized_initial_position_error(:, :, r);
end

% 组装场景参数字典
scenario.position_prior_std_vec = std_vec;                 % XYZ 三轴的初始位置先验标准差向量 (留给滤波器做 P0 初始化)
scenario.initial_position_error = initial_position_error;  % 实际施加在每个批次的初始位置误差真值
scenario.dataset_id             = dataset_id;              % 当前数据集的文件标识后缀
scenario.base_seed              = double(base_seed);       % 本次运行底层生成的基础随机数种子
scenario.batch_seeds            = double(batch_seeds);     % 内部各独立批次所使用的衍生随机数种子数组
%% ==================== 4. 3D 轨迹可视化 ====================
figure('Name', 'Multi-Agent 拓扑与轨迹可视化', 'Position', [100, 100, 900, 700]);
hold on; grid on; axis equal;
xlabel('X 轴位置 (m)'); ylabel('Y 轴位置 (m)'); zlabel('高度 Z (m)');
title(sprintf('%d机 贴地高速编队 与 高空共面基站', Vehicle_num));

% 绘制 120x120x10 的立体边界框 (适配 E3 的 (60,60) 中心点坐标)
X_max = 120; Y_max = 120; Z_max = 10;
line([0, X_max, X_max, 0, 0], [0, 0, Y_max, Y_max, 0], [0, 0, 0, 0, 0], 'Color', [0.5,0.5,0.5], 'LineStyle', '--');
line([0, X_max, X_max, 0, 0], [0, 0, Y_max, Y_max, 0], [Z_max, Z_max, Z_max, Z_max, Z_max], 'Color', [0.5,0.5,0.5], 'LineStyle', '--');
for corner = [0, X_max]
    for side = [0, Y_max]
        line([corner, corner], [side, side], [0, Z_max], 'Color', [0.5,0.5,0.5], 'LineStyle', '--');
    end
end

% 绘制天花板基站
h_anchor = plot3(anchors(:,1), anchors(:,2), anchors(:,3), '^', 'MarkerSize', 10, ...
    'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, 'DisplayName', 'UWB 基站');

% 绘制真值轨迹 (仅取前 N_steps 个点，即第一批次，避免多批次重复绘制重叠图形)
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
legend([h_anchor, h_traj(1)], 'Location', 'northeastoutside'); % 为简化图例，只显示基站和一个车辆
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
fprintf('E3 多批次 3D 仿真数据生成成功！\n');
fprintf('文件后缀 ID: %s (已覆盖/生成新数据)\n', dataset_id);
fprintf('底层基础种子: %d (使用动态时间生成，保证每次运行数据不同)\n', base_seed);
fprintf('车辆数: %d，基站数: %d\n', Vehicle_num, Anchor_num);
fprintf('每批时长: %.1f s，每批步数: %d，重复批次: %d\n', t_end, N_steps, repeat_time);
fprintf('UWB测距噪声: %.3f m\n', UWB_noise_params.sigma_anc);
fprintf('文件已保存至: %s\n', save_path);
fprintf('======================================================\n\n');


%% ======================= 局部函数区域 =======================

function batch = generate_imu_uwb_batch(trajectory_template, ...
        truth_position, anchors, N_steps, dt_imu, ...
        uwb_downsample_factor, IMU_noise_params, UWB_noise_params)
    % 为单个批次生成加噪的 IMU 和 UWB 测量
    
    Vehicle_num = size(truth_position, 3);
    Anchor_num = size(anchors, 1);
    batch = trajectory_template;

    % 1. 生成 IMU 测量 (100Hz)
    for n = 1:Vehicle_num
        v_name = sprintf('V%d', n);
        veh = batch.(v_name);

        theta_unwrapped = unwrap(veh.Theta_true);
        wz_true = gradient(theta_unwrapped, dt_imu);
        omega_body_ideal = [zeros(N_steps, 2), wz_true];

        a_body_ideal = zeros(N_steps, 3);
        g_vec = [0; 0; -9.81];
        for k = 1:N_steps
            R_k = veh.R_true(:, :, k);
            a_world = [veh.A_true(k, 1); veh.A_true(k, 2); veh.A_true(k, 3)];
            a_body_ideal(k, :) = (R_k' * (a_world - g_vec))';
        end

        ba_init = (rand(1, 3) - 0.5) * 0.1;
        bw_init = (rand(1, 3) - 0.5) * 0.01;
        b_a = ba_init + cumsum(randn(N_steps, 3) * IMU_noise_params.sigma_ba * sqrt(dt_imu), 1);
        b_w = bw_init + cumsum(randn(N_steps, 3) * IMU_noise_params.sigma_bw * sqrt(dt_imu), 1);

        veh.IMU_Time = veh.Time_true;
        veh.IMU_acc_m = a_body_ideal + b_a + randn(N_steps, 3) * IMU_noise_params.sigma_na;
        veh.IMU_gyro_m = omega_body_ideal + b_w + randn(N_steps, 3) * IMU_noise_params.sigma_nw;
        veh.IMU_bias_a_true = b_a;
        veh.IMU_bias_w_true = b_w;
        batch.(v_name) = veh;
    end

    % 2. 生成 UWB 测距 (10Hz)
    idx_uwb = 1:uwb_downsample_factor:N_steps;
    t_uwb = trajectory_template.V1.Time_true(idx_uwb);
    N_uwb = length(t_uwb);
    pos_true_uwb = truth_position(idx_uwb, :, :);

    for n = 1:Vehicle_num
        v_name = sprintf('V%d', n);

        % 基站测距
        UWB_Anchor = zeros(N_uwb, 1 + Anchor_num);
        UWB_Anchor(:, 1) = t_uwb;
        for a_idx = 1:Anchor_num
            delta = squeeze(pos_true_uwb(:, :, n)) - anchors(a_idx, :);
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
                delta = squeeze(pos_true_uwb(:, :, n)) - squeeze(pos_true_uwb(:, :, j));
                dist_true = sqrt(sum(delta.^2, 2));
                UWB_Relative(:, 1 + j) = dist_true + randn(N_uwb, 1) * UWB_noise_params.sigma_rel;
            end
        end
        batch.(v_name).UWB_Relative = UWB_Relative;
    end
end

function total = concatenate_trajectories(total, batch)
    % 沿时间维拼接多个批次的轨迹和测量
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
    % E3 近共面基站与近距离编队轨迹初始化
    
    formation_spacing = 2.5;          % 相邻车辆的水平编队间距
    anchor_height = 5.0;              % 基站名义高度
    anchor_z_half_span = 0.25;        % 基站高度相对名义值的单侧错落幅度
    vehicle_z_step = 0.5;             % 相邻车辆之间的高度差
    anchor_radial_offsets = [11, 12]; % 基站相对编队轨道半径的交替径向偏移
    orbit_radius = 20.0;              % 编队中心轨道半径
    radius_amplitude = 2.0;           % 轨道半径周期变化幅度
    radius_rate = 0.06;               % 轨道半径周期变化角频率
    angular_rate = 0.12;              % 编队中心绕场地旋转角速度
    z_amplitude = 0.15;               % 每辆车的高度周期变化幅度

    % 基站生成 (带高度错落)
    anchors = zeros(Anchor_num, 3);
    for a = 1:Anchor_num
        angle = 0.3 + 2*pi*(a - 1)/Anchor_num;
        offset_index = mod(a - 1, numel(anchor_radial_offsets)) + 1;
        radial_offset = anchor_radial_offsets(offset_index);
        anchors(a, :) = [ ...
            60 + (orbit_radius + radial_offset)*cos(angle), ...
            60 + (orbit_radius + radial_offset)*sin(angle), ...
            anchor_height + anchor_z_half_span*sin(1.7*(a - 1) + 0.3)];
    end

    % 编队偏移量分配
    n_columns = min(4, max(1, ceil(sqrt(Vehicle_num))));
    n_rows = ceil(Vehicle_num / n_columns);
    offsets_tangent = zeros(Vehicle_num, 1);
    offsets_radial = zeros(Vehicle_num, 1);
    offsets_z = zeros(Vehicle_num, 1);
    for n = 1:Vehicle_num
        column = mod(n - 1, n_columns);
        row = floor((n - 1) / n_columns);
        offsets_tangent(n) = (column - (n_columns - 1)/2) * formation_spacing;
        offsets_radial(n) = (row - (n_rows - 1)/2) * formation_spacing;
        offsets_z(n) = vehicle_z_step * (mod(n - 1, 5) - 2);
    end

    % 轨迹演算
    time = (0:N_steps-1)' * dt_imu;
    trajectories = struct();
    truth_position = zeros(N_steps, 3, Vehicle_num);

    for n = 1:Vehicle_num
        P_true = zeros(N_steps, 3);
        V_true = zeros(N_steps, 3);
        A_true = zeros(N_steps, 3);
        Theta_true = zeros(N_steps, 1);
        
        o_t = offsets_tangent(n);
        o_r = offsets_radial(n);
        o_z = offsets_z(n);
        z_phase = 2*pi*(n - 1) / Vehicle_num;

        for k = 1:N_steps
            t = time(k);
            theta = angular_rate*t + 0.3;
            er = [cos(theta); sin(theta); 0];
            et = [-sin(theta); cos(theta); 0];

            radius = orbit_radius + radius_amplitude*sin(radius_rate*t);
            radius_dot = radius_amplitude*radius_rate*cos(radius_rate*t);
            radius_ddot = -radius_amplitude*radius_rate^2 * sin(radius_rate*t);

            center = [60 + radius*cos(theta); ...
                      60 + radius*sin(theta); ...
                      anchor_height + z_amplitude*sin(0.10*t + z_phase) + o_z];
            
            center_v = [radius_dot*cos(theta) - radius*angular_rate*sin(theta); ...
                        radius_dot*sin(theta) + radius*angular_rate*cos(theta); ...
                        z_amplitude*0.10*cos(0.10*t + z_phase)];
                    
            center_a = [(radius_ddot - radius*angular_rate^2)*cos(theta) - 2*radius_dot*angular_rate*sin(theta); ...
                        (radius_ddot - radius*angular_rate^2)*sin(theta) + 2*radius_dot*angular_rate*cos(theta); ...
                        -z_amplitude*0.10^2*sin(0.10*t + z_phase)];

            offset = o_t*et + o_r*er;
            offset_v = -o_t*angular_rate*er + o_r*angular_rate*et;
            offset_a = -angular_rate^2*offset;

            P_true(k, :) = (center + offset)';
            V_true(k, :) = (center_v + offset_v)';
            A_true(k, :) = (center_a + offset_a)';
            Theta_true(k) = theta + pi/2;
        end

        truth_position(:, :, n) = P_true;
        v_name = sprintf('V%d', n);
        trajectories.(v_name).Time_true = time;
        trajectories.(v_name).X_true = P_true(:, 1);
        trajectories.(v_name).Y_true = P_true(:, 2);
        trajectories.(v_name).Z_true = P_true(:, 3);
        trajectories.(v_name).Vx_true = V_true(:, 1);
        trajectories.(v_name).Vy_true = V_true(:, 2);
        trajectories.(v_name).Vz_true = V_true(:, 3);
        trajectories.(v_name).Theta_true = Theta_true;
        trajectories.(v_name).A_true = A_true;

        R_true = zeros(3, 3, N_steps);
        for k = 1:N_steps
            theta_k = Theta_true(k);
            R_true(:, :, k) = [ ...
                cos(theta_k), -sin(theta_k), 0; ...
                sin(theta_k),  cos(theta_k), 0; ...
                0,             0,            1];
        end
        trajectories.(v_name).R_true = R_true;
    end
end