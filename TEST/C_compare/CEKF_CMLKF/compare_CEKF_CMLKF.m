% compare_CEKF_CMLKF.m
% 在同一多批次数据集上比较 CEKF 和 CMLKF 的位置 RMSE
clc; clear; close all;

%% ==================== 1. 测试参数配置 ====================
Vehicle_num = 6;
Anchor_num  = 10;
dataset_id  = 'CEKF_CMLKF_30';      % 可以是任意字符串，例如 '002' 或 'CEKF_CMLKF_002'

run_flag    = 1;          % 0：结果存在时直接读取；1：重新运行
save_flag   = 1;          % 1：保存各批次、各时刻的位置 RMSE (CSV)
figure_save = 0;          % 0：不保存图片；1：保存平均 RMSE 曲线图 (PNG)

bias_comp_ratio = 1.0;    % 1.0 表示完全补偿 IMU 零偏
data_ratio      = 1.0;    % 每个批次使用的数据比例

% --- CMLKF 滤波器参数 ---
cml_max_iter = 60;
cml_epsilon  = 1e-8;
cml_beta_inv = 100;
cml_max_step = Inf;

fprintf('\n================ 开始运行 CEKF 与 CMLKF 算法对比 ================\n');
fprintf('目标数据集: 车辆 %d, 基站 %d, 后缀标识: %s\n', Vehicle_num, Anchor_num, dataset_id);

%% ==================== 2. 路径与文件配置 ====================
test_dir   = fileparts(mfilename('fullpath'));
repo_root  = fileparts(fileparts(fileparts(test_dir)));
data_dir   = fullfile(test_dir, 'data');
result_dir = fullfile(test_dir, 'results');

if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

addpath(fullfile(repo_root, 'EKF', 'CEKF'));
addpath(fullfile(repo_root, 'MLKF', 'CMLKF'));

% 文件名统一规范
data_file   = fullfile(data_dir, sprintf('Trj_Veh%d_Anc%d_%s.mat', Vehicle_num, Anchor_num, dataset_id));
result_file = fullfile(result_dir, sprintf('%s_Veh%d_Anc%d.csv', dataset_id,Vehicle_num, Anchor_num));
figure_file = fullfile(result_dir, sprintf('%s_Veh%d_Anc%d.png', dataset_id,Vehicle_num, Anchor_num));

% 若关闭运行且结果已存在，直接进入分析与绘图环节
if run_flag == 0 && exist(result_file, 'file')
    fprintf('检测到 run_flag=0 且历史数据存在，直接读取 CSV 进行分析...\n');
    T = readtable(result_file);
    metrics_table = summarize_from_long_table(T);
    print_metrics_table(metrics_table, Vehicle_num, Anchor_num, dataset_id);
    plot_average_rmse(T, figure_save, figure_file); % 绘图
    return;
end

%% ==================== 3. 加载多批次数据集 ====================
if ~exist(data_file, 'file')
    error('未找到数据集文件：%s\n请先运行 generate_data_3D.m。', data_file);
end

S = load(data_file, 'trajectories', 'anchors_all', ...
    'Vehicle_num', 'Anchor_num', 'scenario', ...
    'IMU_noise_params', 'UWB_noise_params', ...
    't_end', 'N_steps', 'N_steps_total', 'repeat_time', ...
    'uwb_downsample_factor');

if S.Vehicle_num ~= Vehicle_num || S.Anchor_num ~= Anchor_num
    error('数据集中的车辆数或基站数与测试参数不一致。');
end

% 数据合法性与维度校验
dt_imu_sample = S.trajectories.V1.Time_true(2) - S.trajectories.V1.Time_true(1);
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
velocity_prior_std = 0.1; % 初始速度先验标准差 (m/s)，用于初始化协方差 P0
attitude_std_deg   = 1.0; % 初始姿态角先验标准差 (度)，用于初始化协方差 P0

%% ==================== 4. 逐批次运行 CEKF 和 CMLKF ====================
batch_metrics = struct('batch', {}, ...
    'CEKF_total', {}, 'CEKF_post2', {}, 'CEKF_final', {}, ...
    'CMLKF_total', {}, 'CMLKF_post2', {}, 'CMLKF_final', {});

% 预分配内存用于长表保存
long_batch = zeros(repeat_count * N_steps_use, 1);
long_time  = zeros(repeat_count * N_steps_use, 1);
long_CEKF  = zeros(repeat_count * N_steps_use, 1);
long_CMLKF = zeros(repeat_count * N_steps_use, 1);
row_offset = 0;

for r = 1:repeat_count
    fprintf('正在处理第 %d/%d 批次数据...\n', r, repeat_count);
    
    trajectories = extract_batch(S.trajectories, r, S.N_steps, N_uwb_per_batch);
    batch_anchors = S.anchors_all(:, :, r);
    initial_position_error = S.scenario.initial_position_error(:, :, r);

    % 运行 CEKF
    CEKF_rmse = run_filter('CEKF', trajectories, batch_anchors, ...
        N_steps_use, S.IMU_noise_params, S.UWB_noise_params, ...
        bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
        attitude_std_deg, initial_position_error, ...
        cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step);

    % 运行 CMLKF
    CMLKF_rmse = run_filter('CMLKF', trajectories, batch_anchors, ...
        N_steps_use, S.IMU_noise_params, S.UWB_noise_params, ...
        bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
        attitude_std_deg, initial_position_error, ...
        cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step);

    % 计算指标并记录
    local_time = trajectories.V1.Time_true(1:N_steps_use);
    CEKF_metrics  = calculate_metrics(local_time, CEKF_rmse);
    CMLKF_metrics = calculate_metrics(local_time, CMLKF_rmse);

    batch_metrics(end + 1) = struct( ... %#ok<SAGROW>
        'batch', r, ...
        'CEKF_total', CEKF_metrics.total, 'CEKF_post2', CEKF_metrics.post2, 'CEKF_final', CEKF_metrics.final, ...
        'CMLKF_total', CMLKF_metrics.total, 'CMLKF_post2', CMLKF_metrics.post2, 'CMLKF_final', CMLKF_metrics.final);

    % 填入长表以便保存 CSV 和后续求平均绘图
    idx = row_offset + (1:N_steps_use);
    long_batch(idx) = r;
    long_time(idx)  = local_time;
    long_CEKF(idx)  = CEKF_rmse;
    long_CMLKF(idx) = CMLKF_rmse;
    row_offset = row_offset + N_steps_use;
end

%% ==================== 5. 数据保存与绘图 ====================
T = table(long_batch, long_time, long_CEKF, long_CMLKF, ...
    'VariableNames', {'batch_id', 'time_s', 'CEKF_position_rmse_m', 'CMLKF_position_rmse_m'});

% 保存 CSV
if save_flag
    writetable(T, result_file);
    fprintf('\n---> 结果数据已成功保存至:\n%s\n', result_file);
end

% 打印汇总表格
metrics_table = struct2table(batch_metrics);
print_metrics_table(metrics_table, Vehicle_num, Anchor_num, dataset_id);

% 绘制并（可选）保存平均 RMSE 曲线
% plot_average_rmse(T, figure_save, figure_file);


%% ======================= 局部函数区域 =======================

function batch_trajectories = extract_batch(trajectories, batch_id, N_steps, N_uwb_per_batch)
    % 从完整多批次轨迹数据中，截取指定单个批次的数据片段
    Vehicle_num = numel(fieldnames(trajectories));
    batch_trajectories = struct();
    idx_imu = (batch_id - 1)*N_steps + (1:N_steps);
    idx_uwb = (batch_id - 1)*N_uwb_per_batch + (1:N_uwb_per_batch);

    for n = 1:Vehicle_num
        v_name = sprintf('V%d', n);
        source = trajectories.(v_name);
        fields = fieldnames(source);
        for q = 1:numel(fields)
            field_name = fields{q};
            if strcmp(field_name, 'R_true')
                batch_trajectories.(v_name).(field_name) = source.(field_name)(:, :, idx_imu);
            elseif strncmp(field_name, 'UWB_', 4)
                batch_trajectories.(v_name).(field_name) = source.(field_name)(idx_uwb, :);
            else
                batch_trajectories.(v_name).(field_name) = source.(field_name)(idx_imu, :);
            end
        end
    end
end

function rmse_series = run_filter(method_name, trajectories, anchors, ...
        N_steps, IMU_noise_params, UWB_noise_params, ...
        bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
        attitude_std_deg, initial_position_error, ...
        cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step)
    % 初始化并运行指定的滤波器 (CEKF 或 CMLKF)，返回随时间变化的 RMSE 数组
    Vehicle_num = numel(fieldnames(trajectories));
    Anchor_num = size(anchors, 1);
    dt_imu = trajectories.V1.Time_true(2) - trajectories.V1.Time_true(1);

    p0 = zeros(3*Vehicle_num, 1);
    v0 = zeros(3*Vehicle_num, 1);
    R0 = zeros(3, 3, Vehicle_num);
    for i = 1:Vehicle_num
        name = sprintf('V%d', i);
        p0(3*i-2:3*i) = [trajectories.(name).X_true(1); ...
                         trajectories.(name).Y_true(1); ...
                         trajectories.(name).Z_true(1)] + initial_position_error(:, i);
        v0(3*i-2:3*i) = [trajectories.(name).Vx_true(1); ...
                         trajectories.(name).Vy_true(1); ...
                         trajectories.(name).Vz_true(1)];
        R0(:, :, i) = trajectories.(name).R_true(:, :, 1);
    end

    if strcmp(method_name, 'CEKF')
        kf = CEKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);
    else
        kf = CMLKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);
        kf.is_GN    = 0;
        kf.max_iter = cml_max_iter;
        kf.epsilon  = cml_epsilon;
        kf.beta_inv = cml_beta_inv;
        kf.max_step = cml_max_step;
    end

    kf.IMU_Sigma_a   = IMU_noise_params.sigma_na^2 * eye(3);
    kf.IMU_Sigma_w   = IMU_noise_params.sigma_nw^2 * eye(3);
    kf.UWB_sigma_anc = UWB_noise_params.sigma_anc;
    kf.UWB_sigma_rel = UWB_noise_params.sigma_rel;

    prior_block = blkdiag(diag(prior_std_vec.^2), velocity_prior_std^2 * eye(3), deg2rad(attitude_std_deg)^2 * eye(3));
    prior_blocks = repmat({prior_block}, 1, Vehicle_num);
    kf.Sigma = blkdiag(prior_blocks{:});

    rmse_series = zeros(N_steps, 1);
    rmse_series(1) = position_rmse(kf.p, trajectories, 1, Vehicle_num);
    uwb_time = trajectories.V1.UWB_Anchor(:, 1);
    uwb_idx = 2;

    for k = 2:N_steps
        acc_m = zeros(3, Vehicle_num);
        gyro_m = zeros(3, Vehicle_num);
        for i = 1:Vehicle_num
            name = sprintf('V%d', i);
            acc_m(:, i)  = trajectories.(name).IMU_acc_m(k-1, :)' - bias_comp_ratio * trajectories.(name).IMU_bias_a_true(k-1, :)';
            gyro_m(:, i) = trajectories.(name).IMU_gyro_m(k-1, :)' - bias_comp_ratio * trajectories.(name).IMU_bias_w_true(k-1, :)';
        end
        kf.predict(acc_m, gyro_m);

        current_time = trajectories.V1.Time_true(k);
        if uwb_idx <= length(uwb_time) && abs(current_time - uwb_time(uwb_idx)) < 1e-5
            uwb_anc = zeros(Vehicle_num, Anchor_num);
            uwb_rel = zeros(Vehicle_num, Vehicle_num);
            for i = 1:Vehicle_num
                name = sprintf('V%d', i);
                uwb_anc(i, :) = trajectories.(name).UWB_Anchor(uwb_idx, 2:end);
                uwb_rel(i, :) = trajectories.(name).UWB_Relative(uwb_idx, 2:end);
            end
            kf.update(uwb_anc, uwb_rel);
            uwb_idx = uwb_idx + 1;
        end

        rmse_series(k) = position_rmse(kf.p, trajectories, k, Vehicle_num);
    end
end

function value = position_rmse(position, trajectories, k, Vehicle_num)
    % 计算当前时刻 k 所有车辆估计位置与真值的均方根误差 (RMSE)
    squared_sum = 0;
    for i = 1:Vehicle_num
        name = sprintf('V%d', i);
        p_true = [trajectories.(name).X_true(k), trajectories.(name).Y_true(k), trajectories.(name).Z_true(k)];
        p_est = position(3*i-2:3*i)';
        squared_sum = squared_sum + sum((p_est - p_true).^2);
    end
    value = sqrt(squared_sum / Vehicle_num);
end

function metrics = calculate_metrics(time, rmse_series)
    % 根据位置 RMSE 时间序列，计算 3 种汇总指标（总RMSE、收敛后RMSE、最终RMSE）
    post2_mask = time >= 2 - 1e-12;
    metrics.total = sqrt(mean(rmse_series.^2));
    metrics.post2 = sqrt(mean(rmse_series(post2_mask).^2));
    metrics.final = rmse_series(end);
end

function metrics_table = summarize_from_long_table(T)
    % 从已读取的长格式表格 T 中重新计算每个批次的汇总指标
    batch_ids = unique(T.batch_id);
    rows = struct('batch', {}, 'CEKF_total', {}, 'CEKF_post2', {}, 'CEKF_final', {}, ...
        'CMLKF_total', {}, 'CMLKF_post2', {}, 'CMLKF_final', {});
    for r = batch_ids'
        idx = T.batch_id == r;
        CEKF_metrics = calculate_metrics(T.time_s(idx), T.CEKF_position_rmse_m(idx));
        CMLKF_metrics = calculate_metrics(T.time_s(idx), T.CMLKF_position_rmse_m(idx));
        rows(end + 1) = struct('batch', r, ... %#ok<SAGROW>
            'CEKF_total', CEKF_metrics.total, 'CEKF_post2', CEKF_metrics.post2, 'CEKF_final', CEKF_metrics.final, ...
            'CMLKF_total', CMLKF_metrics.total, 'CMLKF_post2', CMLKF_metrics.post2, 'CMLKF_final', CMLKF_metrics.final);
    end
    metrics_table = struct2table(rows);
end

function plot_average_rmse(T, figure_save, figure_file)
    % 按时间点计算各个批次的平均 RMSE，并绘制对比曲线图
    
    % 使用 varfun 按时间进行分组求均值
    avg_table = varfun(@mean, T, 'InputVariables', {'CEKF_position_rmse_m', 'CMLKF_position_rmse_m'}, ...
                       'GroupingVariables', 'time_s');
    
    time_series = avg_table.time_s;
    avg_CEKF    = avg_table.mean_CEKF_position_rmse_m;
    avg_CMLKF   = avg_table.mean_CMLKF_position_rmse_m;
    
    % 绘图
    figure('Name', 'Average RMSE Over Time', 'Position', [150, 150, 800, 500]);
    plot(time_series, avg_CEKF, 'b-', 'LineWidth', 1.5, 'DisplayName', 'CEKF');
    hold on; grid on;
    plot(time_series, avg_CMLKF, 'r-', 'LineWidth', 1.5, 'DisplayName', 'CMLKF');
    
    xlabel('Time (s)'); 
    ylabel('RMSE (m)');
    legend('Location', 'northeast');
    
    if figure_save
        saveas(gcf, figure_file);
        fprintf('---> 图片已成功保存至:\n%s\n', figure_file);
    end
end

function print_metrics_table(T, Vehicle_num, Anchor_num, dataset_id)
    % 在终端控制台美观地打印每个批次的指标，以及整体平均指标
    fprintf('\n================ CEKF / CMLKF 位置 RMSE ================\n');
    fprintf('数据集：Veh%d_Anc%d，后缀标识=%s\n\n', Vehicle_num, Anchor_num, dataset_id);
    fprintf('批次 | CEKF总  CEKF2s后  CEKF最终 | CMLKF总  CMLKF2s后  CMLKF最终\n');
    fprintf('--------------------------------------------------------------\n');
    for i = 1:height(T)
        fprintf('%4d | %7.4f  %8.4f  %8.4f | %8.4f  %9.4f  %9.4f\n', ...
            T.batch(i), T.CEKF_total(i), T.CEKF_post2(i), T.CEKF_final(i), ...
            T.CMLKF_total(i), T.CMLKF_post2(i), T.CMLKF_final(i));
    end
    fprintf('--------------------------------------------------------------\n');
    fprintf('平均 | %7.4f  %8.4f  %8.4f | %8.4f  %9.4f  %9.4f\n', ...
        mean(T.CEKF_total), mean(T.CEKF_post2), mean(T.CEKF_final), ...
        mean(T.CMLKF_total), mean(T.CMLKF_post2), mean(T.CMLKF_final));
    fprintf('==============================================================\n\n');
end