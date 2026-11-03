% =========================================================================
% 脚本名称: compare_CEKF_CMLKF_CIEKF.m
% 功能描述: 在同一多批次数据集上比较 CEKF、CMLKF 和 CIEKF 在不同基站数量下的位置 RMSE
%           (采用单车 RMSE 算术平均逻辑，最终汇总并保存各基站数下的全网平均 RMSE)
% =========================================================================
clc; clear; close all;

%% ==================== 1. 测试参数配置 ====================
Vehicle_num     = 4;
Anchor_num      = 12;
Anchor_data_num = 4:12;  % 测试使用的基站数量，可以是标量(如10)或向量(如4:12)
dataset_id      = 'CEKF_CMLKF_CIEKF_30_5'; 

save_flag       = 1;  % 是否保存各基站数下平均位置 RMSE 汇总表至 CSV
run_flag        = 0;     % 1: 运行实验; 0: 仅读取并打印已有结果

% --- 算法端与滤波器相关参数 ---
bias_comp_ratio    = 1;   % 1.0 表示完全补偿 IMU 零偏
data_ratio         = 1.0; % 每个批次使用的数据比例
velocity_prior_std = 0.1; % 初始速度先验标准差 (m/s)
attitude_std_deg   = 1.0; % 初始姿态角先验标准差 (度)

% --- CMLKF 与 CIEKF 迭代参数 ---
cml_max_iter = 60;        % 最大迭代次数 (CMLKF)
ekf_max_iter = 60;        % 最大迭代次数 (IEKF)
cml_epsilon  = 1e-6;      % 迭代收敛阈值 (CMLKF与CIEKF共用)
cml_beta_inv = 100;       % CMLKF 专属 越小则优化过程牛顿步走的越慢但是稳定 需要更多迭代
cml_max_step = 1;         % CMLKF 专属

% 如果要让MLKF用少的迭代数跑则用下面的参数 但有时可能不稳定
% cml_beta_inv = 10;       % CMLKF 专属
% cml_max_step = inf;      % CMLKF 专属

% 参数校验
if max(Anchor_data_num) > Anchor_num
    error('Anchor_data_num 中的最大值 (%d) 不能超过总基站数 Anchor_num (%d)！', ...
        max(Anchor_data_num), Anchor_num);
end


%% ==================== 2. 路径与文件配置 ====================
test_dir   = fileparts(mfilename('fullpath'));
repo_root  = fileparts(fileparts(fileparts(test_dir)));
data_dir   = fullfile(test_dir, 'data');
result_dir = fullfile(test_dir, 'results');

% 添加各个滤波器的路径
addpath(fullfile(repo_root, 'EKF', 'CEKF'));
addpath(fullfile(repo_root, 'MLKF', 'CMLKF'));
addpath(fullfile(repo_root, 'IEKF', 'CIEKF')); 

% 按照要求格式命名数据与结果文件 (开始基站数 - 结束基站数)
data_file   = fullfile(data_dir, sprintf( ...
    'Trj_Veh%d_Anc%d_%s.mat', Vehicle_num, Anchor_num, dataset_id));
result_file = fullfile(result_dir, sprintf( ...
    '%s_Veh%d_Anc%d-%d.csv', dataset_id, Vehicle_num, min(Anchor_data_num), max(Anchor_data_num)));

% 仅读取模式处理 (run_flag == 0)
if run_flag == 0
    if exist(result_file, 'file')
        summary_table = readtable(result_file);
        print_summary_table(summary_table, Vehicle_num, dataset_id);
    else
        fprintf('当前数据集结果不存在，请先将 run_flag 设为 1 并运行实验得到结果。\n');
    end
    return;
end

fprintf('\n================ 开始运行 CEKF, CMLKF, CIEKF 基站数量对比 ================\n');
fprintf('目标数据集: 车辆 %d, 总基站 %d, 测试基站范围 %d-%d, 后缀: %s\n', ...
    Vehicle_num, Anchor_num, min(Anchor_data_num), max(Anchor_data_num), dataset_id);


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

dt_imu_sample     = S.trajectories.V1.Time_true(2) - S.trajectories.V1.Time_true(1);
N_steps_from_tend = round(S.t_end / dt_imu_sample) + 1;
if N_steps_from_tend ~= S.N_steps
    error('数据集中的 N_steps 与 t_end、采样周期不一致。');
end

repeat_count    = round(S.N_steps_total / N_steps_from_tend);
N_steps_use     = max(2, round(S.N_steps * data_ratio));
N_uwb_per_batch = size(S.trajectories.V1.UWB_Anchor, 1) / repeat_count;

% 读取共有的位置先验
prior_std_vec = S.scenario.position_prior_std_vec;


%% ==================== 4. 外层循环：遍历指定基站数量 ====================
num_anc_tests = length(Anchor_data_num);
summary_metrics = repmat(struct( ...
    'Anchor_Count', 0, ...
    'CEKF', 0, ...
    'CMLKF', 0, ...
    'CIEKF', 0), num_anc_tests, 1);

for anc_idx = 1:num_anc_tests
    current_anc_num = Anchor_data_num(anc_idx);
    
    fprintf('\n=============================================================================\n');
    fprintf('>>> 正在测试基站数量: %d <<<\n', current_anc_num);
    
    % 记录当前基站数下，每辆车在各个批次中的位置 RMSE (Vehicle_num x repeat_count)
    rmse_p_cekf_all  = zeros(Vehicle_num, repeat_count);
    rmse_p_cmlkf_all = zeros(Vehicle_num, repeat_count);
    rmse_p_ciekf_all = zeros(Vehicle_num, repeat_count);

    % 内层循环：逐批次处理
    for r = 1:repeat_count
        fprintf('  - 正在处理第 %d/%d 批次数据...\n', r, repeat_count);
        
        trajectories = extract_batch(S.trajectories, r, S.N_steps, N_uwb_per_batch);
        % 关键截取：只选取前 current_anc_num 个基站坐标
        batch_anchors          = S.anchors_all(1:current_anc_num, :, r); 
        initial_position_error = S.scenario.initial_position_error(:, :, r);
    
        % 运行 CEKF (返回各车在该批次下的位置 RMSE 向量: Vehicle_num x 1)
        rmse_p_cekf_all(:, r) = run_filter('CEKF', trajectories, batch_anchors, ...
            N_steps_use, S.IMU_noise_params, S.UWB_noise_params, ...
            bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
            attitude_std_deg, initial_position_error, ...
            cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step);
    
        % 运行 CMLKF
        rmse_p_cmlkf_all(:, r) = run_filter('CMLKF', trajectories, batch_anchors, ...
            N_steps_use, S.IMU_noise_params, S.UWB_noise_params, ...
            bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
            attitude_std_deg, initial_position_error, ...
            cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step);
    
        % 运行 CIEKF
        rmse_p_ciekf_all(:, r) = run_filter('CIEKF', trajectories, batch_anchors, ...
            N_steps_use, S.IMU_noise_params, S.UWB_noise_params, ...
            bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
            attitude_std_deg, initial_position_error, ...
            ekf_max_iter, cml_epsilon, cml_beta_inv, cml_max_step);
    end

    % 1. 计算当前基站数下，各车在所有批次上的平均位置 RMSE (Vehicle_num x 1)
    rmse_p_cekf  = mean(rmse_p_cekf_all, 2);
    rmse_p_cmlkf = mean(rmse_p_cmlkf_all, 2);
    rmse_p_ciekf = mean(rmse_p_ciekf_all, 2);

    % 2. 打印当前基站数量下的各车平均位置 RMSE 表 (即表 2)
    print_vehicle_table(rmse_p_cekf, rmse_p_cmlkf, rmse_p_ciekf, ...
        Vehicle_num, current_anc_num, repeat_count);
    
    % 3. 将各车 RMSE 求算术平均 (Avg)，存入最终汇总结构体中 (每个基站数一行)
    summary_metrics(anc_idx) = struct( ...
        'Anchor_Count', current_anc_num, ...
        'CEKF',  mean(rmse_p_cekf), ...
        'CMLKF', mean(rmse_p_cmlkf), ...
        'CIEKF', mean(rmse_p_ciekf));
end


%% ==================== 5. 数据保存与汇总输出 ====================
summary_table = struct2table(summary_metrics);

% 保存最终的汇总 CSV
if save_flag
    if ~exist(result_dir, 'dir')
        mkdir(result_dir);
    end
    writetable(summary_table, result_file);
    fprintf('\n---> 最终基站变化汇总结果已成功保存至:\n%s\n', result_file);
end

% 打印全部基站跑完后的汇总对比表
print_summary_table(summary_table, Vehicle_num, dataset_id);


%% ======================= 局部函数区域 =======================

% -------------------------------------------------------------------------
% 函数: extract_batch
% 功能: 从完整多批次轨迹数据中，截取指定单个批次的数据片段
% -------------------------------------------------------------------------
function batch_trajectories = extract_batch(trajectories, batch_id, N_steps, N_uwb_per_batch)
    Vehicle_num        = numel(fieldnames(trajectories));
    batch_trajectories = struct();
    idx_imu            = (batch_id - 1)*N_steps + (1:N_steps);
    idx_uwb            = (batch_id - 1)*N_uwb_per_batch + (1:N_uwb_per_batch);

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

% -------------------------------------------------------------------------
% 函数: run_filter
% 功能: 初始化并运行指定的滤波器，返回该批次中每辆车各自的位置 RMSE (Vehicle_num x 1)
% -------------------------------------------------------------------------
function rmse_p = run_filter(method_name, trajectories, anchors, ...
        N_steps, IMU_noise_params, UWB_noise_params, ...
        bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
        attitude_std_deg, initial_position_error, ...
        cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step)
    
    Vehicle_num        = numel(fieldnames(trajectories));
    Current_Anchor_num = size(anchors, 1); 
    dt_imu             = trajectories.V1.Time_true(2) - trajectories.V1.Time_true(1);

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

    prior_block  = blkdiag(diag(prior_std_vec.^2), velocity_prior_std^2 * eye(3), deg2rad(attitude_std_deg)^2 * eye(3));
    prior_blocks = repmat({prior_block}, 1, Vehicle_num);
    kf.Sigma     = blkdiag(prior_blocks{:});

    % 预分配各车三维位置估计存储空间 (N_steps x 3 x Vehicle_num)
    est_p = zeros(N_steps, 3, Vehicle_num);
    for i = 1:Vehicle_num
        est_p(1, :, i) = kf.p(3*i-2 : 3*i)';
    end

    uwb_time = trajectories.V1.UWB_Anchor(:, 1);
    uwb_idx  = 2;

    % 滤波主循环
    for k = 2:N_steps
        acc_m  = zeros(3, Vehicle_num);
        gyro_m = zeros(3, Vehicle_num);
        for i = 1:Vehicle_num
            name         = sprintf('V%d', i);
            acc_m(:, i)  = trajectories.(name).IMU_acc_m(k-1, :)' - bias_comp_ratio * trajectories.(name).IMU_bias_a_true(k-1, :)';
            gyro_m(:, i) = trajectories.(name).IMU_gyro_m(k-1, :)' - bias_comp_ratio * trajectories.(name).IMU_bias_w_true(k-1, :)';
        end
        kf.predict(acc_m, gyro_m);

        current_time = trajectories.V1.Time_true(k);
        if uwb_idx <= length(uwb_time) && abs(current_time - uwb_time(uwb_idx)) < 1e-5
            uwb_anc = zeros(Vehicle_num, Current_Anchor_num);
            uwb_rel = zeros(Vehicle_num, Vehicle_num);
            for i = 1:Vehicle_num
                name          = sprintf('V%d', i);
                % 关键截取：只提取前 Current_Anchor_num 个基站的观测值
                uwb_anc(i, :) = trajectories.(name).UWB_Anchor(uwb_idx, 2:(1 + Current_Anchor_num));
                uwb_rel(i, :) = trajectories.(name).UWB_Relative(uwb_idx, 2:end);
            end
            kf.update(uwb_anc, uwb_rel);
            uwb_idx = uwb_idx + 1;
        end

        % 记录当前时刻各车辆位置估计值
        for i = 1:Vehicle_num
            est_p(k, :, i) = kf.p(3*i-2 : 3*i)';
        end
    end

    % 计算每辆车在整个时间序列 (1~N_steps) 上的位置 RMSE
    rmse_p = zeros(Vehicle_num, 1);
    for i = 1:Vehicle_num
        name   = sprintf('V%d', i);
        true_p = [trajectories.(name).X_true(1:N_steps), ...
                  trajectories.(name).Y_true(1:N_steps), ...
                  trajectories.(name).Z_true(1:N_steps)];
        rmse_p(i) = sqrt(mean(sum((est_p(:, :, i) - true_p).^2, 2)));
    end
end

% -------------------------------------------------------------------------
% 函数: print_vehicle_table
% 功能: 打印单个基站数目下，各车辆在所有批次上的平均位置 RMSE 表 (表 2 格式)
% -------------------------------------------------------------------------
function print_vehicle_table(rmse_cekf, rmse_cmlkf, rmse_ciekf, Vehicle_num, Anchor_num, repeat_count)
    fprintf('\n------> 基站数 = %d：各车辆位置 RMSE 在 %d 个批次上的平均值 (单位: m) <------\n', ...
        Anchor_num, repeat_count);
    fprintf(' VehID  |    CEKF     |    CMLKF    |    CIEKF\n');
    fprintf('-----------------------------------------------------------------------------\n');
    for i = 1:Vehicle_num
        fprintf('  %2d    |   %8.4f  |   %8.4f  |   %8.4f\n', ...
            i, rmse_cekf(i), rmse_cmlkf(i), rmse_ciekf(i));
    end
    fprintf('-----------------------------------------------------------------------------\n');
    fprintf('  Avg   |   %8.4f  |   %8.4f  |   %8.4f\n', ...
        mean(rmse_cekf), mean(rmse_cmlkf), mean(rmse_ciekf));
end

% -------------------------------------------------------------------------
% 函数: print_summary_table
% 功能: 打印最终按基站数汇总的对比表 (每行对应一个基站数下的各车算术平均 RMSE)
% -------------------------------------------------------------------------
function print_summary_table(T, Vehicle_num, dataset_id)
    fprintf('\n============== 终极汇总：各车平均位置 RMSE 随基站数量的变化 (单位: m) ==============\n');
    fprintf('数据集：Veh%d，后缀标识=%s\n\n', Vehicle_num, dataset_id);
    fprintf(' 基站数 |    CEKF     |    CMLKF    |    CIEKF\n');
    fprintf('------------------------------------------------------------------------------------\n');
    for i = 1:height(T)
        fprintf(' %6d |   %8.4f  |   %8.4f  |   %8.4f\n', ...
            T.Anchor_Count(i), T.CEKF(i), T.CMLKF(i), T.CIEKF(i));
    end
    fprintf('====================================================================================\n\n');
end