% compare_CMLKF_CRBPF_parallel.m
% 并行比较 CEKF、CMLKF 与 5 组不同 IMU 噪声的 CRBPF。
% CEKF/CMLKF 每批只运行一次，5 组 CRBPF 共用同一数据与随机种子。

clc; clear; close all;

%% ==================== 1. 实验参数 ====================
Vehicle_num = 6;
Anchor_num  = 10;
dataset_id  = 'CMLKF_CRBPF_30';

bias_comp_ratio = 1.0;
data_ratio      = 1.0;

% --- CMLKF 参数 ---
cml_max_iter = 60;
cml_epsilon  = 1e-8;
cml_beta_inv = 100;
cml_max_step = Inf;

% --- CRBPF 公共参数 ---
rng_base_seed       = 20261028;
crbpf_num_particles = 300;
crbpf_batch_size    = 40;
crbpf_neff_ratio    = 0.65;
crbpf_rough_coeff   = 0.05;
crbpf_min_jitter    = 0.001;

% --- CRBPF 五组 IMU 噪声扫描 ---
% 前四组固定 UWB=0.03，逐步增大 IMU 过程噪声；
% 第五组在最大 IMU 设置下进一步测试更尖锐的 UWB 噪声。
config_names = ["CRBPF_IMU_a005_w0005", ...
                "CRBPF_IMU_a0075_w00075", ...
                "CRBPF_IMU_a010_w001", ...
                "CRBPF_IMU_a015_w00125", ...
                "CRBPF_IMU_a020_w0015_UWB001"];
crbpf_imu_sigma_a_std = [0.05, 0.075, 0.10, 0.15, 0.20];
crbpf_imu_sigma_w_std = [0.005, 0.0075, 0.010, 0.0125, 0.015];
crbpf_uwb_sigmas = [0.03, 0.03, 0.03, 0.03, 0.01];
num_configs = numel(crbpf_imu_sigma_a_std);

save_flag = 1;

%% ==================== 2. 路径与数据 ====================
test_dir     = fileparts(mfilename('fullpath'));
compare_root = fileparts(test_dir);
repo_root    = fileparts(fileparts(compare_root));
data_dir     = fullfile(test_dir, 'data');
result_dir   = fullfile(test_dir, 'result');

addpath(fullfile(repo_root, 'EKF', 'CEKF'));
addpath(fullfile(repo_root, 'MLKF', 'CMLKF'));
addpath(fullfile(repo_root, 'RBPF', 'CRBPF'));

if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

data_file = fullfile(data_dir, sprintf( ...
    'Trj_Veh%d_Anc%d_%s.mat', Vehicle_num, Anchor_num, dataset_id));
if ~exist(data_file, 'file')
    error('未找到数据集文件：%s', data_file);
end

S = load(data_file, 'trajectories', 'anchors_all', ...
    'Vehicle_num', 'Anchor_num', 'scenario', ...
    'IMU_noise_params', 'UWB_noise_params', ...
    't_end', 'N_steps', 'N_steps_total', 'repeat_time', ...
    'uwb_downsample_factor');

if S.Vehicle_num ~= Vehicle_num || S.Anchor_num ~= Anchor_num
    error('数据集中的车辆数或基站数与实验参数不一致。');
end

dt_imu_sample = S.trajectories.V1.Time_true(2) - ...
    S.trajectories.V1.Time_true(1);
N_steps_from_tend = round(S.t_end / dt_imu_sample) + 1;
repeat_count = S.N_steps_total / N_steps_from_tend;
if round(repeat_count) ~= repeat_count || ...
        round(repeat_count) ~= S.repeat_time
    error('数据集总步数或重复次数校验失败。');
end
repeat_count = round(repeat_count);

N_steps_use = max(2, round(S.N_steps * data_ratio));
N_uwb_per_batch = size(S.trajectories.V1.UWB_Anchor, 1) / repeat_count;

prior_std_vec      = S.scenario.position_prior_std_vec;
velocity_prior_std = 0.1;
attitude_std_deg   = 1.0;

ekf_imu_sigma_a   = S.IMU_noise_params.sigma_na^2 * eye(3);
ekf_imu_sigma_w   = S.IMU_noise_params.sigma_nw^2 * eye(3);
ekf_uwb_sigma_anc = S.UWB_noise_params.sigma_anc;
ekf_uwb_sigma_rel = S.UWB_noise_params.sigma_rel;

% 提前提取各批次，避免每个并行任务重复切割整个数据集。
batch_trajectories = cell(repeat_count, 1);
batch_anchors = cell(repeat_count, 1);
batch_initial_errors = cell(repeat_count, 1);
for r = 1:repeat_count
    batch_trajectories{r} = extract_batch(S.trajectories, r, ...
        S.N_steps, N_uwb_per_batch);
    batch_anchors{r} = S.anchors_all(:, :, r);
    batch_initial_errors{r} = ...
        S.scenario.initial_position_error(:, :, r);
end

fprintf('\n========== 开始并行 CRBPF 参数扫描 ==========\n');
fprintf('数据集：Veh%d_Anc%d，%d 批，每批 %d 步\n', ...
    Vehicle_num, Anchor_num, repeat_count, N_steps_use);
fprintf('数据 UWB 噪声：%.3f m\n', ekf_uwb_sigma_anc);
fprintf('CRBPF UWB 噪声配置：%s m\n', mat2str(crbpf_uwb_sigmas));
fprintf('CRBPF IMU 加速度计噪声扫描：%s m/s^2\n', ...
    mat2str(crbpf_imu_sigma_a_std));
fprintf('CRBPF IMU 陀螺仪噪声扫描：%s rad/s\n', ...
    mat2str(crbpf_imu_sigma_w_std));

pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('Processes');
end
fprintf('并行工作进程数：%d\n', pool.NumWorkers);

batch_trajectories_const = parallel.pool.Constant(batch_trajectories);
batch_anchors_const = parallel.pool.Constant(batch_anchors);
batch_initial_errors_const = parallel.pool.Constant(batch_initial_errors);

%% ==================== 3. 构造并行任务 ====================
% 每批包含：CEKF、CMLKF 和 5 组 CRBPF，共 7 个任务。
tasks_per_batch = 2 + num_configs;
num_tasks = repeat_count * tasks_per_batch;

task_kind = zeros(num_tasks, 1);       % 1=CEKF, 2=CMLKF, 3=CRBPF
task_batch = zeros(num_tasks, 1);
task_config = zeros(num_tasks, 1);

for r = 1:repeat_count
    base = (r - 1) * tasks_per_batch;
    task_kind(base + 1) = 1;
    task_kind(base + 2) = 2;
    task_kind(base + 3:base + tasks_per_batch) = 3;
    task_batch(base + 1:base + tasks_per_batch) = r;
    task_config(base + 3:base + tasks_per_batch) = 1:num_configs;
end

task_rmse = cell(num_tasks, 1);
task_status = strings(num_tasks, 1);
task_error = strings(num_tasks, 1);
task_elapsed = zeros(num_tasks, 1);

%% ==================== 4. 并行运行 ====================
total_timer = tic;
parfor task_idx = 1:num_tasks
    r = task_batch(task_idx);
    kind = task_kind(task_idx);
    cfg_idx = task_config(task_idx);
    trajectories = batch_trajectories_const.Value{r};
    anchors = batch_anchors_const.Value{r};
    initial_position_error = batch_initial_errors_const.Value{r};
    batch_seed = rng_base_seed + r - 1;
    task_timer = tic;
    method_name = '';
    method_cfg = struct();

    try
        switch kind
            case 1
                method_name = 'CEKF';
                method_cfg = struct();

            case 2
                method_name = 'CMLKF';
                method_cfg = struct( ...
                    'cml_max_iter', cml_max_iter, ...
                    'cml_epsilon', cml_epsilon, ...
                    'cml_beta_inv', cml_beta_inv, ...
                    'cml_max_step', cml_max_step);

            case 3
                method_name = 'CRBPF';
                method_cfg = struct( ...
                    'num_particles', crbpf_num_particles, ...
                    'uwb_sigma_anc', crbpf_uwb_sigmas(cfg_idx), ...
                    'uwb_sigma_rel', crbpf_uwb_sigmas(cfg_idx), ...
                    'imu_sigma_a', ...
                        crbpf_imu_sigma_a_std(cfg_idx)^2 * eye(3), ...
                    'imu_sigma_w', ...
                        crbpf_imu_sigma_w_std(cfg_idx)^2 * eye(3), ...
                    'batch_size', crbpf_batch_size, ...
                    'neff_ratio', crbpf_neff_ratio, ...
                    'rough_coeff', crbpf_rough_coeff, ...
                    'min_jitter', crbpf_min_jitter, ...
                    'rng_seed', batch_seed);
        end

        task_rmse{task_idx} = run_filter( ...
            method_name, method_cfg, trajectories, anchors, N_steps_use, ...
            bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
            attitude_std_deg, initial_position_error, ...
            ekf_imu_sigma_a, ekf_imu_sigma_w, ...
            ekf_uwb_sigma_anc, ekf_uwb_sigma_rel);
        task_status(task_idx) = "ok";

    catch ME
        task_rmse{task_idx} = nan(N_steps_use, 1);
        task_status(task_idx) = string(ME.identifier);
        if strlength(task_status(task_idx)) == 0
            task_status(task_idx) = "failed";
        end
        task_error(task_idx) = string(ME.message);
    end

    task_elapsed(task_idx) = toc(task_timer);
end

fprintf('并行计算完成，总耗时 %.1f 秒。\n', toc(total_timer));

%% ==================== 5. 整理每时刻结果 ====================
num_rows = repeat_count * N_steps_use;
long_batch = zeros(num_rows, 1);
long_time = zeros(num_rows, 1);
long_CEKF = zeros(num_rows, 1);
long_CMLKF = zeros(num_rows, 1);
long_CRBPF = cell(1, num_configs);
for c = 1:num_configs
    long_CRBPF{c} = zeros(num_rows, 1);
end

row_offset = 0;
for r = 1:repeat_count
    base = (r - 1) * tasks_per_batch;
    local_time = batch_trajectories{r}.V1.Time_true(1:N_steps_use);
    idx = row_offset + (1:N_steps_use);

    long_batch(idx) = r;
    long_time(idx) = local_time;
    long_CEKF(idx) = task_rmse{base + 1};
    long_CMLKF(idx) = task_rmse{base + 2};
    for c = 1:num_configs
        long_CRBPF{c}(idx) = task_rmse{base + 2 + c};
    end
    row_offset = row_offset + N_steps_use;
end

per_time_data = [long_batch, long_time, long_CEKF, long_CMLKF, ...
    cell2mat(long_CRBPF)];
base_names = {'batch_id', 'time_s', ...
    'CEKF_position_rmse_m', 'CMLKF_position_rmse_m'};
crbpf_names = cell(1, num_configs);
for c = 1:num_configs
    crbpf_names{c} = char(config_names(c) + "_position_rmse_m");
end
per_time_names = [base_names, crbpf_names];
T = array2table(per_time_data, 'VariableNames', per_time_names);

%% ==================== 6. 汇总指标 ====================
num_summary_rows = repeat_count * tasks_per_batch;
summary_rows = repmat(struct('batch', 0, 'algorithm', "", ...
    'total', NaN, 'post2', NaN, 'final', NaN, ...
    'status', "", 'error', "", 'elapsed_s', NaN), ...
    num_summary_rows, 1);

summary_idx = 0;
for r = 1:repeat_count
    base = (r - 1) * tasks_per_batch;
    local_time = batch_trajectories{r}.V1.Time_true(1:N_steps_use);

    for offset = 1:tasks_per_batch
        task_idx = base + offset;
        if offset == 1
            algorithm = "CEKF";
        elseif offset == 2
            algorithm = "CMLKF";
        else
            algorithm = config_names(offset - 2);
        end

        series = task_rmse{task_idx};
        if all(isnan(series))
            metrics = struct('total', NaN, 'post2', NaN, 'final', NaN);
        else
            metrics = calculate_metrics(local_time, series);
        end

        summary_idx = summary_idx + 1;
        summary_rows(summary_idx) = struct( ...
            'batch', r, ...
            'algorithm', algorithm, ...
            'total', metrics.total, ...
            'post2', metrics.post2, ...
            'final', metrics.final, ...
            'status', task_status(task_idx), ...
            'error', task_error(task_idx), ...
            'elapsed_s', task_elapsed(task_idx));
    end
end

summary_table = struct2table(summary_rows);

config_table = table(config_names(:), ...
    crbpf_uwb_sigmas(:), ...
    crbpf_imu_sigma_a_std(:), crbpf_imu_sigma_w_std(:), ...
    repmat(crbpf_num_particles, num_configs, 1), ...
    repmat(crbpf_batch_size, num_configs, 1), ...
    repmat(crbpf_neff_ratio, num_configs, 1), ...
    repmat(crbpf_rough_coeff, num_configs, 1), ...
    'VariableNames', {'config_name', 'uwb_sigma', ...
    'imu_sigma_a_std', 'imu_sigma_w_std', ...
    'num_particles', 'batch_size', 'neff_ratio', 'rough_coeff'});

print_summary(summary_table);

%% ==================== 7. 保存扫描结果 ====================
if save_flag
    per_time_file = fullfile(result_dir, sprintf( ...
        '%s_Veh%d_Anc%d_imu_parallel_per_time.csv', ...
        dataset_id, Vehicle_num, Anchor_num));
    summary_file = fullfile(result_dir, sprintf( ...
        '%s_Veh%d_Anc%d_imu_parallel_summary.csv', ...
        dataset_id, Vehicle_num, Anchor_num));
    config_file = fullfile(result_dir, sprintf( ...
        '%s_Veh%d_Anc%d_imu_parallel_configs.csv', ...
        dataset_id, Vehicle_num, Anchor_num));
    mat_file = fullfile(result_dir, sprintf( ...
        '%s_Veh%d_Anc%d_imu_parallel.mat', ...
        dataset_id, Vehicle_num, Anchor_num));

    writetable(T, per_time_file);
    writetable(summary_table, summary_file);
    writetable(config_table, config_file);
    save(mat_file, 'T', 'summary_table', 'config_table', ...
        'crbpf_uwb_sigmas', 'crbpf_imu_sigma_a_std', ...
        'crbpf_imu_sigma_w_std', 'task_status', 'task_error');

    fprintf('结果已保存至：%s\n', result_dir);
end

delete(pool);

%% ======================= 局部函数 =======================

function batch_trajectories = extract_batch(trajectories, batch_id, ...
        N_steps, N_uwb_per_batch)
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

function rmse_series = run_filter(method_name, method_cfg, ...
        trajectories, anchors, N_steps, bias_comp_ratio, ...
        prior_std_vec, velocity_prior_std, attitude_std_deg, ...
        initial_position_error, ekf_imu_sigma_a, ekf_imu_sigma_w, ...
        ekf_uwb_sigma_anc, ekf_uwb_sigma_rel)

    Vehicle_num = numel(fieldnames(trajectories));
    Anchor_num = size(anchors, 1);
    dt_imu = trajectories.V1.Time_true(2) - trajectories.V1.Time_true(1);

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
            kf.max_iter = method_cfg.cml_max_iter;
            kf.epsilon  = method_cfg.cml_epsilon;
            kf.beta_inv = method_cfg.cml_beta_inv;
            kf.max_step = method_cfg.cml_max_step;
            configure_standard_noise(kf, ekf_imu_sigma_a, ...
                ekf_imu_sigma_w, ekf_uwb_sigma_anc, ekf_uwb_sigma_rel);

        case 'CRBPF'
            rng(method_cfg.rng_seed, 'twister');
            kf = CRBPF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);
            kf.set_particle_count(method_cfg.num_particles);
            kf.batch_size = method_cfg.batch_size;
            kf.neff_ratio = method_cfg.neff_ratio;
            kf.rough_coeff = method_cfg.rough_coeff;
            kf.min_att_jitter = method_cfg.min_jitter;
            configure_crbpf_noise(kf, method_cfg.imu_sigma_a, ...
                method_cfg.imu_sigma_w, method_cfg.uwb_sigma_anc, ...
                method_cfg.uwb_sigma_rel);

        otherwise
            error('未知滤波器类型：%s', method_name);
    end

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
    kf.IMU_Sigma_a   = imu_sigma_a;
    kf.IMU_Sigma_w   = imu_sigma_w;
    kf.UWB_sigma_anc = uwb_sigma_anc;
    kf.UWB_sigma_rel = uwb_sigma_rel;
end

function configure_crbpf_noise(kf, imu_sigma_a, imu_sigma_w, ...
        uwb_sigma_anc, uwb_sigma_rel)
    kf.IMU_Sigma_a   = imu_sigma_a;
    kf.IMU_Sigma_w   = imu_sigma_w;
    kf.UWB_sigma_anc = uwb_sigma_anc;
    kf.UWB_sigma_rel = uwb_sigma_rel;

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

function value = position_rmse(position, trajectories, k, Vehicle_num)
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
    metrics.total = sqrt(mean(rmse_series.^2));
    metrics.post2 = sqrt(mean(rmse_series(post2_mask).^2));
    metrics.final = rmse_series(end);
end

function print_summary(summary_table)
    fprintf('\n========== 并行扫描结果 ==========\n');
    fprintf('批次 | 算法                  | 总RMSE    2s后RMSE   最终RMSE | 状态\n');
    fprintf('-------------------------------------------------------------------\n');
    for i = 1:height(summary_table)
        fprintf('%4d | %-20s | %9.4f  %9.4f  %9.4f | %s\n', ...
            summary_table.batch(i), ...
            summary_table.algorithm(i), ...
            summary_table.total(i), ...
            summary_table.post2(i), ...
            summary_table.final(i), ...
            summary_table.status(i));
    end
    fprintf('===================================================================\n');
end
