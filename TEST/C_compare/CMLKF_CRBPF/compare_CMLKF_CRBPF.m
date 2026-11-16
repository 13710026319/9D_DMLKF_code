% compare_CEKF_CMLKF_CRBPF.m
% 在同一组多批次数据上比较 CEKF、CMLKF 和 CRBPF 的位置 RMSE。
% 三个算法共享相同的初始状态误差、先验参数和输入测量。
% 评价指标逻辑：
% 1、计算每辆车在全程时间上的 RMSE
% 2、各车在多个批次上进行 RMSE 平均
% 3、将所有车辆的均值结果再次求平均，得到各算法的最终评价 RMSE

clc; clear; close all;

%% ==================== 1. 测试参数配置 ====================
Vehicle_num = 6;
Anchor_num  = 10;
dataset_id  = 'CMLKF_CPF_30';
run_flag    = 1;          % 当前脚本暂不读取历史结果
save_flag   = 0;          % 暂时不保存 CSV
figure_save = 0;          % 暂时不保存图片
bias_comp_ratio = 1.0;    % 1.0 表示完全补偿 IMU 零偏
data_ratio      = 1.0;    % 每个批次使用的数据比例

% --- CMLKF 参数 ---
cml_max_iter = 60;
cml_epsilon  = 1e-8;
cml_beta_inv = 100;
cml_max_step = Inf;

% --- CRBPF 参数 ---
crbpf_num_particles = 300;
rng_base_seed = 20261028;

% CRBPF 当前使用较宽的观测噪声，后续再单独调参。
crbpf_noise.IMU_Sigma_a   = (0.25)^2 * eye(3);
crbpf_noise.IMU_Sigma_w   = (0.015)^2 * eye(3);
crbpf_noise.UWB_sigma_anc = 0.08;
crbpf_noise.UWB_sigma_rel = 0.08;

fprintf('\n========== 开始运行 CEKF、CMLKF 与 CRBPF 算法对比 ==========\n');
fprintf('目标数据集: 车辆 %d, 基站 %d, 后缀标识: %s\n', ...
    Vehicle_num, Anchor_num, dataset_id);

%% ==================== 2. 路径与文件配置 ====================
test_dir      = fileparts(mfilename('fullpath'));
compare_root  = fileparts(test_dir);
repo_root     = fileparts(fileparts(compare_root));
data_dir      = fullfile(compare_root, 'CMLKF_CRBPF', 'data');
result_dir    = fullfile(test_dir, 'result');

if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

addpath(fullfile(repo_root, 'EKF', 'CEKF'));
addpath(fullfile(repo_root, 'MLKF', 'CMLKF'));
addpath(fullfile(repo_root, 'RBPF', 'CRBPF'));

data_file   = fullfile(data_dir, sprintf( ...
    'Trj_Veh%d_Anc%d_%s.mat', Vehicle_num, Anchor_num, dataset_id));
result_file = fullfile(result_dir, sprintf( ...
    '%s_Veh%d_Anc%d_metrics.mat', dataset_id, Vehicle_num, Anchor_num));

% 当前不读取历史结果，保留 run_flag 以便后续扩展。
if run_flag ~= 1
    error('当前版本仅支持 run_flag=1。');
end

%% ==================== 3. 加载多批次数据集 ====================
if ~exist(data_file, 'file')
    error('未找到数据集文件：%s\n请检查数据目录和 dataset_id。', data_file);
end

S = load(data_file, 'trajectories', 'anchors_all', ...
    'Vehicle_num', 'Anchor_num', 'scenario', ...
    'IMU_noise_params', 'UWB_noise_params', ...
    't_end', 'N_steps', 'N_steps_total', 'repeat_time', ...
    'uwb_downsample_factor');

if S.Vehicle_num ~= Vehicle_num || S.Anchor_num ~= Anchor_num
    error('数据集中的车辆数或基站数与测试参数不一致。');
end

dt_imu_sample = S.trajectories.V1.Time_true(2) - ...
    S.trajectories.V1.Time_true(1);
N_steps_from_tend = round(S.t_end / dt_imu_sample) + 1;

if N_steps_from_tend ~= S.N_steps
    error('数据集中的 N_steps 与 t_end、采样周期不一致。');
end

repeat_count = S.N_steps_total / N_steps_from_tend;
if round(repeat_count) ~= repeat_count || round(repeat_count) ~= S.repeat_time
    error('数据集总步数或重复次数校验失败。');
end
repeat_count = round(repeat_count);

N_steps_use = max(2, round(S.N_steps * data_ratio));
N_uwb_per_batch = size(S.trajectories.V1.UWB_Anchor, 1) / repeat_count;

prior_std_vec      = S.scenario.position_prior_std_vec;
velocity_prior_std = 0.1;
attitude_std_deg   = 1.0;

% CEKF/CMLKF 使用数据文件中的 IMU/UWB 噪声参数。
ekf_imu_sigma_a   = S.IMU_noise_params.sigma_na^2 * eye(3);
ekf_imu_sigma_w   = S.IMU_noise_params.sigma_nw^2 * eye(3);
ekf_uwb_sigma_anc = S.UWB_noise_params.sigma_anc;
ekf_uwb_sigma_rel = S.UWB_noise_params.sigma_rel;

fprintf('数据集加载成功：%d 个批次，每批 %d 步。\n', ...
    repeat_count, N_steps_use);

%% ==================== 4. 逐批次运行三个滤波器 ====================
% 存储格式：矩阵维度为 (批次数 repeat_count) x (车辆数 Vehicle_num)
% 其中每个元素代表某批次下、某辆车在整个时间轴上的均方根误差 (RMSE)
CEKF_batch_veh_rmse  = zeros(repeat_count, Vehicle_num);
CMLKF_batch_veh_rmse = zeros(repeat_count, Vehicle_num);
CRBPF_batch_veh_rmse = zeros(repeat_count, Vehicle_num);

for r = 1:repeat_count
    fprintf('正在处理第 %d/%d 批次数据...\n', r, repeat_count);
    trajectories = extract_batch(S.trajectories, r, S.N_steps, N_uwb_per_batch);
    batch_anchors = S.anchors_all(:, :, r);
    initial_position_error = S.scenario.initial_position_error(:, :, r);
    batch_seed = rng_base_seed + r - 1;
    
    % --- 运行 CEKF ---
    CEKF_batch_veh_rmse(r, :) = run_filter('CEKF', trajectories, batch_anchors, ...
        N_steps_use, S.IMU_noise_params, S.UWB_noise_params, ...
        bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
        attitude_std_deg, initial_position_error, ...
        cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step, ...
        crbpf_num_particles, crbpf_noise, batch_seed, ...
        ekf_imu_sigma_a, ekf_imu_sigma_w, ...
        ekf_uwb_sigma_anc, ekf_uwb_sigma_rel);
    
    % --- 运行 CMLKF ---
    CMLKF_batch_veh_rmse(r, :) = run_filter('CMLKF', trajectories, batch_anchors, ...
        N_steps_use, S.IMU_noise_params, S.UWB_noise_params, ...
        bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
        attitude_std_deg, initial_position_error, ...
        cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step, ...
        crbpf_num_particles, crbpf_noise, batch_seed, ...
        ekf_imu_sigma_a, ekf_imu_sigma_w, ...
        ekf_uwb_sigma_anc, ekf_uwb_sigma_rel);
    
    % --- 运行 CRBPF ---
    CRBPF_batch_veh_rmse(r, :) = run_filter('CRBPF', trajectories, batch_anchors, ...
        N_steps_use, S.IMU_noise_params, S.UWB_noise_params, ...
        bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
        attitude_std_deg, initial_position_error, ...
        cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step, ...
        crbpf_num_particles, crbpf_noise, batch_seed, ...
        ekf_imu_sigma_a, ekf_imu_sigma_w, ...
        ekf_uwb_sigma_anc, ekf_uwb_sigma_rel);
end

%% ==================== 5. 结果整理 ====================
% 步骤2：各车在各批次上做RMSE平均
% 得到大小为 (1 x Vehicle_num) 的向量，代表每辆车多批次后的平均 RMSE
CEKF_veh_avg  = mean(CEKF_batch_veh_rmse, 1);
CMLKF_veh_avg = mean(CMLKF_batch_veh_rmse, 1);
CRBPF_veh_avg = mean(CRBPF_batch_veh_rmse, 1);

% 步骤3：各算法对各车做平均，得到该算法最终 RMSE
CEKF_final_rmse  = mean(CEKF_veh_avg);
CMLKF_final_rmse = mean(CMLKF_veh_avg);
CRBPF_final_rmse = mean(CRBPF_veh_avg);

% 打印结果
fprintf('\n========== 最终算法对比 RMSE ==========\n');
fprintf('数据集：Veh%d_Anc%d，后缀标识=%s\n', Vehicle_num, Anchor_num, dataset_id);
fprintf('聚合逻辑：单车全程均方根 -> 各车多批次平均 -> 所有车取均值\n\n');

fprintf('【各单车多批次平均 RMSE (米)】:\n');
fprintf('Vehicle ID |   CEKF   |  CMLKF   |  CRBPF\n');
fprintf('-------------------------------------------\n');
for i = 1:Vehicle_num
    fprintf('   V%d      |  %.4f  |  %.4f  |  %.4f\n', ...
        i, CEKF_veh_avg(i), CMLKF_veh_avg(i), CRBPF_veh_avg(i));
end
fprintf('-------------------------------------------\n');
fprintf(' 算法最终  |  %.4f  |  %.4f  |  %.4f\n', ...
    CEKF_final_rmse, CMLKF_final_rmse, CRBPF_final_rmse);
fprintf('===========================================\n\n');

if save_flag
    save(result_file, 'CEKF_batch_veh_rmse', 'CMLKF_batch_veh_rmse', ...
         'CRBPF_batch_veh_rmse', 'CEKF_veh_avg', 'CMLKF_veh_avg', ...
         'CRBPF_veh_avg', 'CEKF_final_rmse', 'CMLKF_final_rmse', 'CRBPF_final_rmse');
    fprintf('---> 评价指标已成功保存至:\n%s\n', result_file);
end


%% ======================= 局部函数区域 =======================

function batch_trajectories = extract_batch(trajectories, batch_id, ...
        N_steps, N_uwb_per_batch)
    % 从完整多批次轨迹数据中截取指定单批次数据。
    Vehicle_num = numel(fieldnames(trajectories));
    batch_trajectories = struct();
    idx_imu = (batch_id - 1) * N_steps + (1:N_steps);
    idx_uwb = (batch_id - 1) * N_uwb_per_batch + (1:N_uwb_per_batch);
    for n = 1:Vehicle_num
        v_name = sprintf('V%d', n);
        source = trajectories.(v_name);
        fields = fieldnames(source);
        for q = 1:numel(fields)
            field_name = fields{q};
            if strcmp(field_name, 'R_true')
                batch_trajectories.(v_name).(field_name) = ...
                    source.(field_name)(:, :, idx_imu);
            elseif strncmp(field_name, 'UWB_', 4)
                batch_trajectories.(v_name).(field_name) = ...
                    source.(field_name)(idx_uwb, :);
            else
                batch_trajectories.(v_name).(field_name) = ...
                    source.(field_name)(idx_imu, :);
            end
        end
    end
end


function veh_rmse = run_filter(method_name, trajectories, anchors, ...
        N_steps, IMU_noise_params, UWB_noise_params, ...
        bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
        attitude_std_deg, initial_position_error, ...
        cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step, ...
        crbpf_num_particles, crbpf_noise, batch_seed, ...
        ekf_imu_sigma_a, ekf_imu_sigma_w, ...
        ekf_uwb_sigma_anc, ekf_uwb_sigma_rel)
    
    Vehicle_num = numel(fieldnames(trajectories));
    Anchor_num = size(anchors, 1);
    dt_imu = trajectories.V1.Time_true(2) - trajectories.V1.Time_true(1);
    
    % 三个算法接收完全相同的初始位置误差。
    p0 = zeros(3 * Vehicle_num, 1);
    v0 = zeros(3 * Vehicle_num, 1);
    R0 = zeros(3, 3, Vehicle_num);
    for i = 1:Vehicle_num
        name = sprintf('V%d', i);
        p0(3*i-2:3*i) = [trajectories.(name).X_true(1); ...
                         trajectories.(name).Y_true(1); ...
                         trajectories.(name).Z_true(1)] + ...
                         initial_position_error(:, i);
        v0(3*i-2:3*i) = [trajectories.(name).Vx_true(1); ...
                         trajectories.(name).Vy_true(1); ...
                         trajectories.(name).Vz_true(1)];
        R0(:, :, i) = trajectories.(name).R_true(:, :, 1);
    end
    
    switch method_name
        case 'CEKF'
            kf = CEKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);
            configure_standard_noise(kf, ekf_imu_sigma_a, ...
                ekf_imu_sigma_w, ekf_uwb_sigma_anc, ekf_uwb_sigma_rel);
        case 'CMLKF'
            kf = CMLKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);
            kf.is_GN    = 0;
            kf.max_iter = cml_max_iter;
            kf.epsilon  = cml_epsilon;
            kf.beta_inv = cml_beta_inv;
            kf.max_step = cml_max_step;
            configure_standard_noise(kf, ekf_imu_sigma_a, ...
                ekf_imu_sigma_w, ekf_uwb_sigma_anc, ekf_uwb_sigma_rel);
        case 'CRBPF'
            rng(batch_seed, 'twister');
            kf = CRBPF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);
            kf.set_particle_count(crbpf_num_particles);
            configure_crbpf_noise(kf, crbpf_noise);
        otherwise
            error('未知滤波器类型：%s', method_name);
    end
    
    % 三个算法使用相同的位置、速度和姿态先验标准差。
    initialize_filter_prior(kf, method_name, p0, v0, R0, ...
        prior_std_vec, velocity_prior_std, attitude_std_deg);
    
    % 记录各个车辆在每一时刻的误差平方：矩阵维度 (N_steps x Vehicle_num)
    err_squared_series = zeros(N_steps, Vehicle_num);
    err_squared_series(1, :) = get_vehicle_sq_err(kf.p, trajectories, 1, Vehicle_num);
    
    uwb_time = trajectories.V1.UWB_Anchor(:, 1);
    uwb_idx = 2;
    
    for k = 2:N_steps
        acc_m = zeros(3, Vehicle_num);
        gyro_m = zeros(3, Vehicle_num);
        for i = 1:Vehicle_num
            name = sprintf('V%d', i);
            acc_m(:, i) = trajectories.(name).IMU_acc_m(k-1, :)' - ...
                bias_comp_ratio * trajectories.(name).IMU_bias_a_true(k-1, :)';
            gyro_m(:, i) = trajectories.(name).IMU_gyro_m(k-1, :)' - ...
                bias_comp_ratio * trajectories.(name).IMU_bias_w_true(k-1, :)';
        end
        kf.predict(acc_m, gyro_m);
        
        current_time = trajectories.V1.Time_true(k);
        if uwb_idx <= length(uwb_time) && ...
                abs(current_time - uwb_time(uwb_idx)) < 1e-5
            uwb_anc = zeros(Vehicle_num, Anchor_num);
            uwb_rel = zeros(Vehicle_num, Vehicle_num);
            for i = 1:Vehicle_num
                name = sprintf('V%d', i);
                uwb_anc(i, :) = ...
                    trajectories.(name).UWB_Anchor(uwb_idx, 2:end);
                uwb_rel(i, :) = ...
                    trajectories.(name).UWB_Relative(uwb_idx, 2:end);
            end
            kf.update(uwb_anc, uwb_rel);
            uwb_idx = uwb_idx + 1;
        end
        
        % 记录当前时刻各车的误差平方
        err_squared_series(k, :) = get_vehicle_sq_err(kf.p, trajectories, k, Vehicle_num);
    end
    
    % 步骤1完成：计算单批次内，每辆车贯穿时间轴的 RMSE (返回大小 1 x Vehicle_num)
    veh_rmse = sqrt(mean(err_squared_series, 1));
end


function configure_standard_noise(kf, imu_sigma_a, imu_sigma_w, ...
        uwb_sigma_anc, uwb_sigma_rel)
    kf.IMU_Sigma_a   = imu_sigma_a;
    kf.IMU_Sigma_w   = imu_sigma_w;
    kf.UWB_sigma_anc = uwb_sigma_anc;
    kf.UWB_sigma_rel = uwb_sigma_rel;
end


function configure_crbpf_noise(kf, crbpf_noise)
    kf.IMU_Sigma_a   = crbpf_noise.IMU_Sigma_a;
    kf.IMU_Sigma_w   = crbpf_noise.IMU_Sigma_w;
    kf.UWB_sigma_anc = crbpf_noise.UWB_sigma_anc;
    kf.UWB_sigma_rel = crbpf_noise.UWB_sigma_rel;
    
    tau = kf.dt_imu;
    F_blk = [eye(3), tau * eye(3); zeros(3), eye(3)];
    F_cell = repmat({F_blk}, 1, kf.Vehicle_num);
    kf.F_global = blkdiag(F_cell{:});
    
    Q_blk = [(tau^4)/4 * kf.IMU_Sigma_a, ...
             (tau^3)/2 * kf.IMU_Sigma_a; ...
             (tau^3)/2 * kf.IMU_Sigma_a, ...
             (tau^2)   * kf.IMU_Sigma_a];
    Q_cell = repmat({Q_blk}, 1, kf.Vehicle_num);
    kf.Q_global = blkdiag(Q_cell{:});
end


function initialize_filter_prior(kf, method_name, p0, v0, R0, ...
        prior_std_vec, velocity_prior_std, attitude_std_deg)
    if strcmp(method_name, 'CRBPF')
        pv_prior_block = blkdiag(diag(prior_std_vec.^2), ...
            velocity_prior_std^2 * eye(3));
        pv_blocks = repmat({pv_prior_block}, 1, kf.Vehicle_num);
        pv_prior_global = blkdiag(pv_blocks{:});
        X_init = zeros(6 * kf.Vehicle_num, 1);
        for i = 1:kf.Vehicle_num
            X_init(6*i-5:6*i-3) = p0(3*i-2:3*i);
            X_init(6*i-2:6*i)   = v0(3*i-2:3*i);
        end
        for n = 1:kf.Np
            kf.P_Xl(:, n) = X_init;
            kf.P_Pl(:, :, n) = pv_prior_global;
        end
        
        attitude_std = deg2rad(attitude_std_deg);
        for n = 1:kf.Np
            for i = 1:kf.Vehicle_num
                delta_theta = attitude_std * randn(3, 1);
                kf.P_R(:, :, i, n) = ...
                    R0(:, :, i) * kf.exp_SO3(delta_theta);
            end
        end
        kf.p = p0;
        kf.v = v0;
        kf.R = R0;
    else
        prior_block = blkdiag(diag(prior_std_vec.^2), ...
            velocity_prior_std^2 * eye(3), ...
            deg2rad(attitude_std_deg)^2 * eye(3));
        prior_blocks = repmat({prior_block}, 1, kf.Vehicle_num);
        kf.Sigma = blkdiag(prior_blocks{:});
    end
end


function sq_errs = get_vehicle_sq_err(position, trajectories, k, Vehicle_num)
    % 计算每一时刻各个车辆的位置误差平方 (返回 1 x Vehicle_num 的向量)
    sq_errs = zeros(1, Vehicle_num);
    for i = 1:Vehicle_num
        name = sprintf('V%d', i);
        p_true = [trajectories.(name).X_true(k); ...
                  trajectories.(name).Y_true(k); ...
                  trajectories.(name).Z_true(k)];
        p_est = position(3*i-2:3*i);
        sq_errs(i) = sum((p_est - p_true).^2);
    end
end