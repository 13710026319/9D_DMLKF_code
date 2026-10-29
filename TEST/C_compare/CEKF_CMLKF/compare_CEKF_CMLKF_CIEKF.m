% compare_CEKF_CMLKF_CIEKF.m
% 在同一多批次数据集上比较 CEKF、CMLKF 和 CIEKF 在不同基站数量下的位置 RMSE
clc; clear; close all;

%% ==================== 1. 测试参数配置 ====================
Vehicle_num     = 6;
Anchor_num      = 10;
Anchor_data_num = 4:10  % 测试使用的基站数量，可以是标量(如10)或向量(如4:10)
dataset_id      = 'CEKF_CMLKF_CIEKF_30'; 

run_flag        = 1;     % 0：结果存在时直接读取；1：重新运行
save_flag       = 1;     % 1：保存各基站数下批次平均的位置 RMSE (CSV)

% --- 算法端与滤波器相关参数 ---
bias_comp_ratio    = 1.0; % 1.0 表示完全补偿 IMU 零偏
data_ratio         = 1.0; % 每个批次使用的数据比例
velocity_prior_std = 0.1; % 初始速度先验标准差 (m/s)
attitude_std_deg   = 1.0; % 初始姿态角先验标准差 (度)

% --- CMLKF 与 CIEKF 迭代参数 ---
cml_max_iter = 60;        % 最大迭代次数 (CMLKF与CIEKF共用)
cml_epsilon  = 1e-8;      % 迭代收敛阈值 (CMLKF与CIEKF共用)
cml_beta_inv = 100;       % CMLKF 专属
cml_max_step = 1;       % CMLKF 专属

% 参数校验
if max(Anchor_data_num) > Anchor_num
    error('Anchor_data_num 中的最大值 (%d) 不能超过总基站数 Anchor_num (%d)！', max(Anchor_data_num), Anchor_num);
end

fprintf('\n================ 开始运行 CEKF, CMLKF, CIEKF 基站数量对比 ================\n');
fprintf('目标数据集: 车辆 %d, 总基站 %d, 测试基站范围 %d-%d, 后缀: %s\n', ...
    Vehicle_num, Anchor_num, min(Anchor_data_num), max(Anchor_data_num), dataset_id);

%% ==================== 2. 路径与文件配置 ====================
test_dir   = fileparts(mfilename('fullpath'));
repo_root  = fileparts(fileparts(fileparts(test_dir)));
data_dir   = fullfile(test_dir, 'data');
result_dir = fullfile(test_dir, 'results');

if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

% 添加各个滤波器的路径
addpath(fullfile(repo_root, 'EKF', 'CEKF'));
addpath(fullfile(repo_root, 'MLKF', 'CMLKF'));
addpath(fullfile(repo_root, 'IEKF', 'CIEKF')); 

% 按照要求的新格式命名结果文件 (开始基站数 - 结束基站数)
data_file   = fullfile(data_dir, sprintf('Trj_Veh%d_Anc%d_%s.mat', Vehicle_num, Anchor_num, dataset_id));
result_file = fullfile(result_dir, sprintf('%s_Veh%d_Anc%d-%d.csv', dataset_id, Vehicle_num, min(Anchor_data_num), max(Anchor_data_num)));

% 若关闭运行且结果已存在，直接读取 CSV 并打印汇总表
if run_flag == 0 && exist(result_file, 'file')
    fprintf('\n检测到 run_flag=0 且历史数据存在，直接读取最终 CSV 汇总表...\n');
    summary_table = readtable(result_file);
    print_summary_table(summary_table, Vehicle_num, dataset_id);
    return;
end

%% ==================== 3. 加载多批次数据集 ====================
if ~exist(data_file, 'file')
    error('未找到数据集文件：%s\n请先运行数据集生成脚本。', data_file);
end

S = load(data_file, 'trajectories', 'anchors_all', ...
    'Vehicle_num', 'Anchor_num', 'scenario', ...
    'IMU_noise_params', 'UWB_noise_params', ...
    't_end', 'N_steps', 'N_steps_total', 'repeat_time', ...
    'uwb_downsample_factor');

if S.Vehicle_num ~= Vehicle_num || S.Anchor_num ~= Anchor_num
    error('数据集中的车辆数或总基站数与测试参数不一致。');
end

dt_imu_sample = S.trajectories.V1.Time_true(2) - S.trajectories.V1.Time_true(1);
N_steps_from_tend = round(S.t_end / dt_imu_sample) + 1;
if N_steps_from_tend ~= S.N_steps
    error('数据集中的 N_steps 与 t_end、采样周期不一致。');
end

repeat_count = round(S.N_steps_total / N_steps_from_tend);
N_steps_use = max(2, round(S.N_steps * data_ratio));
N_uwb_per_batch = size(S.trajectories.V1.UWB_Anchor, 1) / repeat_count;

% 读取共有的位置先验
prior_std_vec = S.scenario.position_prior_std_vec;

%% ==================== 4. 外层循环：遍历指定基站数量 ====================
% 初始化汇总表格所需的结构体数组
summary_metrics = struct('Anchor_Count', {}, ...
    'CEKF_total_avg', {}, 'CEKF_post2_avg', {}, 'CEKF_final_avg', {}, ...
    'CMLKF_total_avg', {}, 'CMLKF_post2_avg', {}, 'CMLKF_final_avg', {}, ...
    'CIEKF_total_avg', {}, 'CIEKF_post2_avg', {}, 'CIEKF_final_avg', {});

for anc_idx = 1:length(Anchor_data_num)
    current_anc_num = Anchor_data_num(anc_idx);
    
    fprintf('\n==========================================================================================\n');
    fprintf('>>> 正在测试基站数量: %d <<<\n', current_anc_num);
    
    batch_metrics = struct('batch', {}, ...
        'CEKF_total', {}, 'CEKF_post2', {}, 'CEKF_final', {}, ...
        'CMLKF_total', {}, 'CMLKF_post2', {}, 'CMLKF_final', {}, ...
        'CIEKF_total', {}, 'CIEKF_post2', {}, 'CIEKF_final', {});

    % 内层循环：逐批次处理
    for r = 1:repeat_count
        fprintf('  - 正在处理第 %d/%d 批次数据...\n', r, repeat_count);
        
        trajectories = extract_batch(S.trajectories, r, S.N_steps, N_uwb_per_batch);
        % 关键截取：只选取前 current_anc_num 个基站坐标
        batch_anchors = S.anchors_all(1:current_anc_num, :, r); 
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
    
        % 运行 CIEKF
        CIEKF_rmse = run_filter('CIEKF', trajectories, batch_anchors, ...
            N_steps_use, S.IMU_noise_params, S.UWB_noise_params, ...
            bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
            attitude_std_deg, initial_position_error, ...
            cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step);
    
        % 计算指标并记录
        local_time = trajectories.V1.Time_true(1:N_steps_use);
        CEKF_metrics  = calculate_metrics(local_time, CEKF_rmse);
        CMLKF_metrics = calculate_metrics(local_time, CMLKF_rmse);
        CIEKF_metrics = calculate_metrics(local_time, CIEKF_rmse);
    
        batch_metrics(end + 1) = struct( ... %#ok<SAGROW>
            'batch', r, ...
            'CEKF_total', CEKF_metrics.total, 'CEKF_post2', CEKF_metrics.post2, 'CEKF_final', CEKF_metrics.final, ...
            'CMLKF_total', CMLKF_metrics.total, 'CMLKF_post2', CMLKF_metrics.post2, 'CMLKF_final', CMLKF_metrics.final, ...
            'CIEKF_total', CIEKF_metrics.total, 'CIEKF_post2', CIEKF_metrics.post2, 'CIEKF_final', CIEKF_metrics.final);
    end

    % 打印当前基站数量下各批次的对比结果
    T_batch = struct2table(batch_metrics);
    print_batch_table(T_batch, Vehicle_num, current_anc_num, dataset_id);
    
    % 将平均值存入汇总结构体中
    summary_metrics(end + 1) = struct( ... %#ok<SAGROW>
        'Anchor_Count', current_anc_num, ...
        'CEKF_total_avg', mean(T_batch.CEKF_total), ...
        'CEKF_post2_avg', mean(T_batch.CEKF_post2), ...
        'CEKF_final_avg', mean(T_batch.CEKF_final), ...
        'CMLKF_total_avg', mean(T_batch.CMLKF_total), ...
        'CMLKF_post2_avg', mean(T_batch.CMLKF_post2), ...
        'CMLKF_final_avg', mean(T_batch.CMLKF_final), ...
        'CIEKF_total_avg', mean(T_batch.CIEKF_total), ...
        'CIEKF_post2_avg', mean(T_batch.CIEKF_post2), ...
        'CIEKF_final_avg', mean(T_batch.CIEKF_final));
end

%% ==================== 5. 数据保存与汇总输出 ====================
summary_table = struct2table(summary_metrics);

% 保存最终的汇总 CSV
if save_flag
    writetable(summary_table, result_file);
    fprintf('\n---> 最终基站变化汇总结果已成功保存至:\n%s\n', result_file);
end

% 打印全部基站跑完后的汇总对比表
print_summary_table(summary_table, Vehicle_num, dataset_id);


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
    % 初始化并运行指定的滤波器，返回随时间变化的 RMSE 数组
    Vehicle_num = numel(fieldnames(trajectories));
    % 此时输入的 anchors 已经是被截取后的，所以可以直接用 size 获取当前的基站数
    Current_Anchor_num = size(anchors, 1); 
    dt_imu = trajectories.V1.Time_true(2) - trajectories.V1.Time_true(1);

    % 构建相同的初始状态
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

    % 根据名字实例化具体的滤波器类
    if strcmp(method_name, 'CEKF')
        kf = CEKF(Vehicle_num, Current_Anchor_num, anchors, dt_imu, p0, v0, R0);
    elseif strcmp(method_name, 'CMLKF')
        kf = CMLKF(Vehicle_num, Current_Anchor_num, anchors, dt_imu, p0, v0, R0);
        kf.is_GN    = 0;
        kf.max_iter = cml_max_iter;
        kf.epsilon  = cml_epsilon;
        kf.beta_inv = cml_beta_inv;
        kf.max_step = cml_max_step;
    elseif strcmp(method_name, 'CIEKF')
        kf = CIEKF(Vehicle_num, Current_Anchor_num, anchors, dt_imu, p0, v0, R0);
        kf.max_iter = cml_max_iter; 
        kf.tol      = cml_epsilon;  
    end

    % 设置统一的噪声参数和初始协方差
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

    % 滤波主循环
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
            uwb_anc = zeros(Vehicle_num, Current_Anchor_num);
            uwb_rel = zeros(Vehicle_num, Vehicle_num);
            for i = 1:Vehicle_num
                name = sprintf('V%d', i);
                % 关键截取：只提取前 Current_Anchor_num 个基站的观测值
                % UWB_Anchor 第一列是时间，第二列开始是基站数据
                uwb_anc(i, :) = trajectories.(name).UWB_Anchor(uwb_idx, 2:(1 + Current_Anchor_num));
                uwb_rel(i, :) = trajectories.(name).UWB_Relative(uwb_idx, 2:end);
            end
            kf.update(uwb_anc, uwb_rel);
            uwb_idx = uwb_idx + 1;
        end

        rmse_series(k) = position_rmse(kf.p, trajectories, k, Vehicle_num);
    end
end

function value = position_rmse(position, trajectories, k, Vehicle_num)
    % 计算当前时刻 RMSE
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
    % 计算 3 种汇总指标
    post2_mask = time >= 2 - 1e-12;
    metrics.total = sqrt(mean(rmse_series.^2));
    metrics.post2 = sqrt(mean(rmse_series(post2_mask).^2));
    metrics.final = rmse_series(end);
end

function print_batch_table(T, Vehicle_num, Anchor_num, dataset_id)
    % 打印单个基站数目下各批次的对比结果
    fprintf('\n------> 局部统计 (车辆: %d, 当前基站数: %d, 后缀: %s) <------\n', Vehicle_num, Anchor_num, dataset_id);
    fprintf('批次 | CEKF总   CEKF>2s  CEKF最终 | CMLKF总  CMLKF>2s CMLKF最终 | CIEKF总  CIEKF>2s CIEKF最终\n');
    fprintf('------------------------------------------------------------------------------------------\n');
    for i = 1:height(T)
        fprintf('%4d | %7.4f  %7.4f  %8.4f | %7.4f  %8.4f  %9.4f | %7.4f  %8.4f  %9.4f\n', ...
            T.batch(i), T.CEKF_total(i), T.CEKF_post2(i), T.CEKF_final(i), ...
            T.CMLKF_total(i), T.CMLKF_post2(i), T.CMLKF_final(i), ...
            T.CIEKF_total(i), T.CIEKF_post2(i), T.CIEKF_final(i));
    end
    fprintf('------------------------------------------------------------------------------------------\n');
    fprintf('平均 | %7.4f  %7.4f  %8.4f | %7.4f  %8.4f  %9.4f | %7.4f  %8.4f  %9.4f\n\n', ...
        mean(T.CEKF_total), mean(T.CEKF_post2), mean(T.CEKF_final), ...
        mean(T.CMLKF_total), mean(T.CMLKF_post2), mean(T.CMLKF_final), ...
        mean(T.CIEKF_total), mean(T.CIEKF_post2), mean(T.CIEKF_final));
end

function print_summary_table(T, Vehicle_num, dataset_id)
    % 打印最终按基站数汇总对比表
    fprintf('\n======================== 终极汇总：平均位置 RMSE 随基站数量的变化 ========================\n');
    fprintf('数据集：Veh%d，后缀标识=%s\n\n', Vehicle_num, dataset_id);
    fprintf('基站数 | CEKF总   CEKF>2s  CEKF最终 | CMLKF总  CMLKF>2s CMLKF最终 | CIEKF总  CIEKF>2s CIEKF最终\n');
    fprintf('------------------------------------------------------------------------------------------\n');
    for i = 1:height(T)
        fprintf('%6d | %7.4f  %7.4f  %8.4f | %7.4f  %8.4f  %9.4f | %7.4f  %8.4f  %9.4f\n', ...
            T.Anchor_Count(i), ...
            T.CEKF_total_avg(i), T.CEKF_post2_avg(i), T.CEKF_final_avg(i), ...
            T.CMLKF_total_avg(i), T.CMLKF_post2_avg(i), T.CMLKF_final_avg(i), ...
            T.CIEKF_total_avg(i), T.CIEKF_post2_avg(i), T.CIEKF_final_avg(i));
    end
    fprintf('==========================================================================================\n\n');
end