% compare_CEKF_CMLKF_CRBPF.m
% 在同一组多批次数据上比较 CEKF、CMLKF 和 CRBPF 的位置 RMSE。
% 三个算法共享相同的初始状态误差、先验参数和输入测量。

clc; clear; close all;

%% ==================== 1. 测试参数配置 ====================
Vehicle_num = 6;
Anchor_num  = 10;
dataset_id  = 'CMLKF_CRBPF_30';

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
    '%s_Veh%d_Anc%d.csv', dataset_id, Vehicle_num, Anchor_num));
figure_file = fullfile(result_dir, sprintf( ...
    '%s_Veh%d_Anc%d.png', dataset_id, Vehicle_num, Anchor_num));

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
batch_metrics = struct('batch', {}, ...
    'CEKF_total', {}, 'CEKF_post2', {}, 'CEKF_final', {}, ...
    'CMLKF_total', {}, 'CMLKF_post2', {}, 'CMLKF_final', {}, ...
    'CRBPF_total', {}, 'CRBPF_post2', {}, 'CRBPF_final', {});

long_batch = zeros(repeat_count * N_steps_use, 1);
long_time  = zeros(repeat_count * N_steps_use, 1);
long_CEKF  = zeros(repeat_count * N_steps_use, 1);
long_CMLKF = zeros(repeat_count * N_steps_use, 1);
long_CRBPF = zeros(repeat_count * N_steps_use, 1);
row_offset = 0;

for r = 1:repeat_count
    fprintf('正在处理第 %d/%d 批次数据...\n', r, repeat_count);

    trajectories = extract_batch(S.trajectories, r, S.N_steps, ...
        N_uwb_per_batch);
    batch_anchors = S.anchors_all(:, :, r);
    initial_position_error = S.scenario.initial_position_error(:, :, r);
    batch_seed = rng_base_seed + r - 1;

    CEKF_rmse = run_filter('CEKF', trajectories, batch_anchors, ...
        N_steps_use, S.IMU_noise_params, S.UWB_noise_params, ...
        bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
        attitude_std_deg, initial_position_error, ...
        cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step, ...
        crbpf_num_particles, crbpf_noise, batch_seed, ...
        ekf_imu_sigma_a, ekf_imu_sigma_w, ...
        ekf_uwb_sigma_anc, ekf_uwb_sigma_rel);

    CMLKF_rmse = run_filter('CMLKF', trajectories, batch_anchors, ...
        N_steps_use, S.IMU_noise_params, S.UWB_noise_params, ...
        bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
        attitude_std_deg, initial_position_error, ...
        cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step, ...
        crbpf_num_particles, crbpf_noise, batch_seed, ...
        ekf_imu_sigma_a, ekf_imu_sigma_w, ...
        ekf_uwb_sigma_anc, ekf_uwb_sigma_rel);

    CRBPF_rmse = run_filter('CRBPF', trajectories, batch_anchors, ...
        N_steps_use, S.IMU_noise_params, S.UWB_noise_params, ...
        bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
        attitude_std_deg, initial_position_error, ...
        cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step, ...
        crbpf_num_particles, crbpf_noise, batch_seed, ...
        ekf_imu_sigma_a, ekf_imu_sigma_w, ...
        ekf_uwb_sigma_anc, ekf_uwb_sigma_rel);

    local_time = trajectories.V1.Time_true(1:N_steps_use);
    CEKF_metrics  = calculate_metrics(local_time, CEKF_rmse);
    CMLKF_metrics = calculate_metrics(local_time, CMLKF_rmse);
    CRBPF_metrics = calculate_metrics(local_time, CRBPF_rmse);

    batch_metrics(end + 1) = struct( ... %#ok<SAGROW>
        'batch', r, ...
        'CEKF_total', CEKF_metrics.total, ...
        'CEKF_post2', CEKF_metrics.post2, ...
        'CEKF_final', CEKF_metrics.final, ...
        'CMLKF_total', CMLKF_metrics.total, ...
        'CMLKF_post2', CMLKF_metrics.post2, ...
        'CMLKF_final', CMLKF_metrics.final, ...
        'CRBPF_total', CRBPF_metrics.total, ...
        'CRBPF_post2', CRBPF_metrics.post2, ...
        'CRBPF_final', CRBPF_metrics.final);

    idx = row_offset + (1:N_steps_use);
    long_batch(idx) = r;
    long_time(idx)  = local_time;
    long_CEKF(idx)  = CEKF_rmse;
    long_CMLKF(idx) = CMLKF_rmse;
    long_CRBPF(idx) = CRBPF_rmse;
    row_offset = row_offset + N_steps_use;
end

%% ==================== 5. 结果整理 ====================
T = table(long_batch, long_time, long_CEKF, long_CMLKF, long_CRBPF, ...
    'VariableNames', {'batch_id', 'time_s', ...
    'CEKF_position_rmse_m', 'CMLKF_position_rmse_m', ...
    'CRBPF_position_rmse_m'});

% 当前不保存任何结果。
if save_flag
    writetable(T, result_file);
    fprintf('\n---> 结果数据已成功保存至:\n%s\n', result_file);
end

metrics_table = struct2table(batch_metrics);
print_metrics_table(metrics_table, Vehicle_num, Anchor_num, dataset_id);

% 暂时不生成图片。
% plot_average_rmse(T, figure_save, figure_file);

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

function rmse_series = run_filter(method_name, trajectories, anchors, ...
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

    rmse_series = zeros(N_steps, 1);
    rmse_series(1) = position_rmse(kf.p, trajectories, 1, Vehicle_num);
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

        rmse_series(k) = position_rmse(kf.p, trajectories, k, Vehicle_num);
    end
end

function configure_standard_noise(kf, imu_sigma_a, imu_sigma_w, ...
        uwb_sigma_anc, uwb_sigma_rel)
    % 所有噪声参数均由当前脚本显式写入。
    kf.IMU_Sigma_a   = imu_sigma_a;
    kf.IMU_Sigma_w   = imu_sigma_w;
    kf.UWB_sigma_anc = uwb_sigma_anc;
    kf.UWB_sigma_rel = uwb_sigma_rel;
end

function configure_crbpf_noise(kf, crbpf_noise)
    % CRBPF 暂时使用较宽的 UWB 噪声，避免粒子权重过快退化。
    kf.IMU_Sigma_a   = crbpf_noise.IMU_Sigma_a;
    kf.IMU_Sigma_w   = crbpf_noise.IMU_Sigma_w;
    kf.UWB_sigma_anc = crbpf_noise.UWB_sigma_anc;
    kf.UWB_sigma_rel = crbpf_noise.UWB_sigma_rel;

    % CRBPF 构造时会根据加速度噪声预先构建 Q_global。
    % 此处同步重建，确保后续调整 IMU_Sigma_a 后确实生效。
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
        % CRBPF 的线性子状态是 [p; v]，对应 CEKF/CMLKF 初始
        % 协方差块中的位置、速度部分，数值保持完全一致。
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

        % 以相同姿态先验标准差初始化姿态粒子。
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

function value = position_rmse(position, trajectories, k, Vehicle_num)
    % 当前时刻所有车辆的位置均方根误差。
    squared_sum = 0;
    for i = 1:Vehicle_num
        name = sprintf('V%d', i);
        p_true = [trajectories.(name).X_true(k), ...
                  trajectories.(name).Y_true(k), ...
                  trajectories.(name).Z_true(k)];
        p_est = position(3*i-2:3*i)';
        squared_sum = squared_sum + sum((p_est - p_true).^2);
    end
    value = sqrt(squared_sum / Vehicle_num);
end

function metrics = calculate_metrics(time, rmse_series)
    post2_mask = time >= 2 - 1e-12;
    metrics.total  = sqrt(mean(rmse_series.^2));
    metrics.post2  = sqrt(mean(rmse_series(post2_mask).^2));
    metrics.final  = rmse_series(end);
end

function print_metrics_table(T, Vehicle_num, Anchor_num, dataset_id)
    fprintf('\n========== CEKF / CMLKF / CRBPF 位置 RMSE ==========\n');
    fprintf('数据集：Veh%d_Anc%d，后缀标识=%s\n\n', ...
        Vehicle_num, Anchor_num, dataset_id);
    fprintf(['批次 | CEKF总 CEKF2s后 CEKF最终 | ' ...
             'CMLKF总 CMLKF2s后 CMLKF最终 | ' ...
             'CRBPF总 CRBPF2s后 CRBPF最终\n']);
    fprintf(['--------------------------------------------------' ...
             '----------------------------\n']);
    for i = 1:height(T)
        fprintf(['%4d | %6.4f %7.4f %7.4f | ' ...
                 '%6.4f %8.4f %8.4f | ' ...
                 '%6.4f %8.4f %8.4f\n'], ...
            T.batch(i), ...
            T.CEKF_total(i), T.CEKF_post2(i), T.CEKF_final(i), ...
            T.CMLKF_total(i), T.CMLKF_post2(i), T.CMLKF_final(i), ...
            T.CRBPF_total(i), T.CRBPF_post2(i), T.CRBPF_final(i));
    end
    fprintf(['--------------------------------------------------' ...
             '----------------------------\n']);
    fprintf(['平均 | %6.4f %7.4f %7.4f | ' ...
             '%6.4f %8.4f %8.4f | ' ...
             '%6.4f %8.4f %8.4f\n'], ...
        mean(T.CEKF_total), mean(T.CEKF_post2), mean(T.CEKF_final), ...
        mean(T.CMLKF_total), mean(T.CMLKF_post2), mean(T.CMLKF_final), ...
        mean(T.CRBPF_total), mean(T.CRBPF_post2), mean(T.CRBPF_final));
    fprintf('====================================================\n\n');
end

function plot_average_rmse(T, figure_save, figure_file)
    % 按时间点计算各批次的平均 RMSE，并绘制三个算法的对比曲线。
    avg_table = varfun(@mean, T, ...
        'InputVariables', {'CEKF_position_rmse_m', ...
                           'CMLKF_position_rmse_m', ...
                           'CRBPF_position_rmse_m'}, ...
        'GroupingVariables', 'time_s');

    time_series = avg_table.time_s;
    avg_CEKF  = avg_table.mean_CEKF_position_rmse_m;
    avg_CMLKF = avg_table.mean_CMLKF_position_rmse_m;
    avg_CRBPF = avg_table.mean_CRBPF_position_rmse_m;

    figure('Name', 'Average RMSE Over Time', ...
        'Position', [150, 150, 800, 500]);
    plot(time_series, avg_CEKF, 'b-', ...
        'LineWidth', 1.5, 'DisplayName', 'CEKF');
    hold on; grid on;
    plot(time_series, avg_CMLKF, 'r-', ...
        'LineWidth', 1.5, 'DisplayName', 'CMLKF');
    plot(time_series, avg_CRBPF, 'g-', ...
        'LineWidth', 1.5, 'DisplayName', 'CRBPF');

    xlabel('Time (s)');
    ylabel('RMSE (m)');
    legend('Location', 'northeast');

    if figure_save
        saveas(gcf, figure_file);
        fprintf('---> 图片已成功保存至:\n%s\n', figure_file);
    end
end
