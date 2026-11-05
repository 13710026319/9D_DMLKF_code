% =========================================================================
% 脚本名称: compare_DEKF_DIEKF_DMLKF.m
% 功能描述: 四算法同台对比测试脚本：DEKF vs. DIEKF vs. DIEKF_ADMM vs. DMLKF
%           (仅计算、打印并保存各车辆在多批次上的平均位置 RMSE 表)
% =========================================================================
clc; clear; close all;

%% 1. 测试场景与公共参数配置
% -------------------------------------------------------------------------
% 1.1 场景规模与拓扑参数
Vehicle_num = 8;                        % 协同车辆总数
Anchor_num  = 4;                        % UWB 基站总数
neighbor_k  = 4;                        % 环形拓扑中的车间通信邻居数

% 1.2 数据与误差配置
% 0.7 为保留 30% 未知零偏以破坏先验 (1.0 为完全补偿)
bias_comp_ratio = 1;                    % IMU 零偏补偿比例
data_ratio      = 1;                  % 每个批次的数据截取比例

% 1.3 数据集与运行控制标志
dataset_id = 'DEKF_DIEKF_DMLKF_15_1';  % 数据集标识
save_flag  = 1;                         % 是否保存实验结果至 CSV
run_flag   = 1;                         % 1: 运行实验; 0: 仅读取并打印已有结果

% 1.4 算法超参数与迭代配置
DIEKF_max_iter     = 3;                % DIEKF 本地高斯牛顿最大迭代次数

DMLKF_max_iter     = 10;                % DMLKF 最大迭代次数
DMLKF_beta_inv     = 70;                % DMLKF 正则化步长参数
DMLKF_print_flag   = 0;                 % DMLKF 调试信息打印开关

DIEKF_ADMM_iter    = 10;               % DIEKF_ADMM 外层 ADMM 迭代次数
DIEKF_ADMM_GN_iter = 3;                % DIEKF_ADMM 内层高斯牛顿迭代次数
DIEKF_ADMM_rho     = 1.2;                % DIEKF_ADMM 惩罚参数


%% 2. 路径配置与数据集加载
% -------------------------------------------------------------------------
% 2.1 构造文件读写路径
test_dir    = fileparts(mfilename('fullpath'));
data_dir    = fullfile(test_dir, 'data');
result_dir  = fullfile(test_dir, 'results');
data_file   = fullfile(data_dir, ...
    sprintf('Trj_Veh%d_Anc%d_%s.mat', Vehicle_num, Anchor_num, dataset_id));
result_file = fullfile(result_dir, sprintf( ...
    '%s_Veh%d_Anc%d.csv', dataset_id, Vehicle_num, Anchor_num));

% 2.2 仅读取模式处理 (run_flag == 0)
if run_flag == 0
    if exist(result_file, 'file')
        opts = detectImportOptions(result_file);
        opts.VariableTypes{1} = 'string';
        saved_result = readtable(result_file, opts);
        
        if ~all(ismember({'DIEKF', 'DIEKF_ADMM'}, saved_result.Properties.VariableNames))
            error('检测到旧版 CSV 结果文件（缺少 DIEKF 或 DIEKF_ADMM 列），请先将 run_flag 设为 1 重新运行并保存结果！');
        end
        
        % 尝试获取批次数量以保持表头打印完全一致
        if exist(data_file, 'file')
            S_meta = load(data_file, 'repeat_time');
            repeat_count = double(S_meta.repeat_time);
        else
            repeat_count = NaN;
        end
        print_vehicle_results(saved_result, repeat_count);
    else
        fprintf('当前数据集结果不存在，请先将 run_flag 设为 1 并运行实验得到结果。\n');
    end
    return;
end

% 2.3 加载数据集文件
if ~exist(data_file, 'file')
    error('未找到数据集文件: %s\n请先运行生成脚本生成该数据！', data_file);
end

S = load(data_file, 'trajectories', 'anchors', ...
    'IMU_noise_params', 'UWB_noise_params', ...
    'Vehicle_num', 'Anchor_num', 'N_steps', 'N_steps_total', ...
    'repeat_time', 't_end');

% 2.4 数据集参数一致性校验
if S.Vehicle_num ~= Vehicle_num || S.Anchor_num ~= Anchor_num
    error('数据集中的车辆数或基站数与测试参数不一致。');
end

anchors       = S.anchors;
trajectories  = S.trajectories;
repeat_count  = double(S.repeat_time);
N_steps_batch = double(S.N_steps);
N_steps_total = numel(trajectories.V1.Time_true);
N_uwb_total   = size(trajectories.V1.UWB_Anchor, 1);

if repeat_count < 1 || round(repeat_count) ~= repeat_count
    error('数据集中的 repeat_time 必须为正整数。');
end
if S.N_steps_total ~= N_steps_total || ...
        N_steps_total ~= N_steps_batch * repeat_count
    error('多批次数据的步数不一致，请检查数据集生成脚本。');
end
if mod(N_uwb_total, repeat_count) ~= 0
    error('UWB 总帧数无法被 repeat_time 整除。');
end

% 2.5 计算单批次步数与采样周期
N_uwb_per_batch = N_uwb_total / repeat_count;
N_steps_use     = max(2, round(N_steps_batch * data_ratio));
N_steps_use     = min(N_steps_use, N_steps_batch);
dt_imu          = trajectories.V1.Time_true(2) - trajectories.V1.Time_true(1);

fprintf('数据集加载成功: %s\n', data_file);
fprintf('>> 共 %d 个独立批次，每批 %d 步；本次每批使用 %d 步 (%.1f%%)\n', ...
    repeat_count, N_steps_batch, N_steps_use, data_ratio * 100);

% 2.6 提取传感器噪声协方差参数（确保四个算法条件完全一致）
Noise.IMU_Sigma_a   = S.IMU_noise_params.sigma_na^2 * eye(3);
Noise.IMU_Sigma_w   = S.IMU_noise_params.sigma_nw^2 * eye(3);
Noise.UWB_sigma_anc = 0.1;
Noise.UWB_sigma_rel = 0.1;


%% 3. 生成静态拓扑掩码 (三档基站 + 对称环形车间拓扑)
% -------------------------------------------------------------------------
[Anchor_Mask, V2V_Mask] = build_topology( ...
    Vehicle_num, Anchor_num, neighbor_k);


%% 4. 逐批次独立运行四个滤波器
% -------------------------------------------------------------------------
rmse_p_dekf_all       = zeros(Vehicle_num, repeat_count);
rmse_p_diekf_all      = zeros(Vehicle_num, repeat_count);
rmse_p_diekf_admm_all = zeros(Vehicle_num, repeat_count);
rmse_p_dmlkf_all      = zeros(Vehicle_num, repeat_count);

for r = 1:repeat_count
    fprintf('  - 正在处理第 %d/%d 批次数据...\n', r, repeat_count);

    % 截取当前批次轨迹数据
    batch_trajectories = extract_batch( ...
        trajectories, r, N_steps_batch, N_uwb_per_batch);

    % 运行四种滤波算法并获取各车 RMSE
    batch_result = run_batch( ...
        batch_trajectories, anchors, Anchor_Mask, V2V_Mask, ...
        Vehicle_num, Anchor_num, N_steps_use, bias_comp_ratio, Noise, ...
        DIEKF_max_iter, ...
        DMLKF_max_iter, DMLKF_beta_inv, DMLKF_print_flag, ...
        DIEKF_ADMM_iter, DIEKF_ADMM_GN_iter, DIEKF_ADMM_rho);

    % 记录当前批次各车辆的位置 RMSE
    rmse_p_dekf_all(:, r)       = batch_result.rmse_p_dekf;
    rmse_p_diekf_all(:, r)      = batch_result.rmse_p_diekf;
    rmse_p_diekf_admm_all(:, r) = batch_result.rmse_p_diekf_admm;
    rmse_p_dmlkf_all(:, r)      = batch_result.rmse_p_dmlkf;
end


%% 5. 构造“各车辆批次平均 + 全网平均(Avg)”结果表并保存
% -------------------------------------------------------------------------
% 计算每辆车在所有批次上的平均位置 RMSE
rmse_p_dekf       = mean(rmse_p_dekf_all, 2);
rmse_p_diekf      = mean(rmse_p_diekf_all, 2);
rmse_p_diekf_admm = mean(rmse_p_diekf_admm_all, 2);
rmse_p_dmlkf      = mean(rmse_p_dmlkf_all, 2);

% 构造包含 VehID (1~Vehicle_num) 及最后一行 Avg 的结果表
veh_labels = [string((1:Vehicle_num)'); "Avg"];
result_table = table( ...
    veh_labels, ...
    [rmse_p_dekf;       mean(rmse_p_dekf)], ...
    [rmse_p_diekf;      mean(rmse_p_diekf)], ...
    [rmse_p_diekf_admm; mean(rmse_p_diekf_admm)], ...
    [rmse_p_dmlkf;      mean(rmse_p_dmlkf)], ...
    'VariableNames', {'VehID', 'DEKF', 'DIEKF', 'DIEKF_ADMM', 'DMLKF'});

if save_flag
    if ~exist(result_dir, 'dir')
        mkdir(result_dir);
    end
    writetable(result_table, result_file);
    fprintf('\n---> 各车辆位置 RMSE 结果已保存至:\n%s\n', result_file);
end


%% 6. 打印各车辆平均位置 RMSE 结果表
% -------------------------------------------------------------------------
print_vehicle_results(result_table, repeat_count);


%% ======================= 局部函数区域 =======================

% -------------------------------------------------------------------------
% 函数: print_vehicle_results
% 功能: 打印各车辆在所有批次上的平均位置 RMSE 表格
% -------------------------------------------------------------------------
function print_vehicle_results(T, repeat_count)
    if ~isnan(repeat_count)
        fprintf('\n=================== 各车辆位置 RMSE 在 %d 个批次上的平均值 (单位: m) ===================\n', repeat_count);
    else
        fprintf('\n=================== 各车辆位置 RMSE 在多个批次上的平均值 (单位: m) ===================\n');
    end
    fprintf(' VehID  |    DEKF     |    DIEKF    |  DIEKF_ADMM |    DMLKF\n');
    fprintf('----------------------------------------------------------------------------------------\n');
    
    for i = 1:height(T)
        veh_label = string(T.VehID(i));
        if veh_label == "Avg" || veh_label == "Average"
            fprintf('----------------------------------------------------------------------------------------\n');
            fprintf('  Avg   |   %8.4f  |   %8.4f  |   %8.4f  |   %8.4f\n', ...
                T.DEKF(i), T.DIEKF(i), T.DIEKF_ADMM(i), T.DMLKF(i));
        else
            veh_id = str2double(veh_label);
            if ~isnan(veh_id)
                fprintf('  %2d    |   %8.4f  |   %8.4f  |   %8.4f  |   %8.4f\n', ...
                    veh_id, T.DEKF(i), T.DIEKF(i), T.DIEKF_ADMM(i), T.DMLKF(i));
            else
                fprintf('  %-4s  |   %8.4f  |   %8.4f  |   %8.4f  |   %8.4f\n', ...
                    veh_label, T.DEKF(i), T.DIEKF(i), T.DIEKF_ADMM(i), T.DMLKF(i));
            end
        end
    end
    fprintf('========================================================================================\n\n');
end

% -------------------------------------------------------------------------
% 函数: extract_batch
% 功能: 从拼接数据中截取一个完整批次；每个批次的时间都从 0 重新开始
% -------------------------------------------------------------------------
function batch_trajectories = extract_batch( ...
        trajectories, batch_id, N_steps, N_uwb_per_batch)
    
    Vehicle_num        = numel(fieldnames(trajectories));
    batch_trajectories = struct();
    idx_imu            = (batch_id - 1) * N_steps + (1:N_steps);
    idx_uwb            = (batch_id - 1) * N_uwb_per_batch + (1:N_uwb_per_batch);

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

% -------------------------------------------------------------------------
% 函数: build_topology
% 功能: 构建车辆与基站的观测掩码矩阵，以及车间对称环形通信拓扑掩码矩阵
% -------------------------------------------------------------------------
function [Anchor_Mask, V2V_Mask] = build_topology( ...
        Vehicle_num, Anchor_num, neighbor_k)
    
    % --- 1. 基站拓扑：30% 全基站、50% 半基站、其余仅相对测距 ---
    N_tier1     = round(0.3 * Vehicle_num);
    N_tier2     = round(0.5 * Vehicle_num);
    Anchor_Mask = ones(Vehicle_num, Anchor_num);
    for i = 1:Vehicle_num
        if i <= N_tier1
            % Tier 1: 保持全 1
        elseif i <= N_tier1 + N_tier2
            for anc_idx = 1:Anchor_num
                if mod(anc_idx, 2) ~= 0
                    Anchor_Mask(i, anc_idx) = 0;
                end
            end
        else
            Anchor_Mask(i, :) = 0;
        end
    end

    % --- 2. 车间拓扑：严格对称的静态环形拓扑 ---
    K_degree = min(neighbor_k, Vehicle_num - 1);
    if mod(Vehicle_num, 2) ~= 0 && mod(K_degree, 2) ~= 0
        error(['图论限制: 车辆总数(%d)与邻居数(%d)均为奇数，' ...
            '数学上无法构成完全对称的双向测距拓扑！'], ...
            Vehicle_num, K_degree);
    end

    V2V_Mask = zeros(Vehicle_num, Vehicle_num);
    K_half   = floor(K_degree / 2);
    for i = 1:Vehicle_num
        for d = 1:K_half
            idx_forward  = mod(i + d - 1, Vehicle_num) + 1;
            idx_backward = mod(i - d - 1, Vehicle_num) + 1;
            V2V_Mask(i, idx_forward)  = 1;
            V2V_Mask(i, idx_backward) = 1;
        end
        if mod(K_degree, 2) ~= 0
            idx_opposite = mod(i + Vehicle_num/2 - 1, Vehicle_num) + 1;
            V2V_Mask(i, idx_opposite) = 1;
        end
    end
    V2V_Mask(logical(eye(Vehicle_num))) = 0;

    if ~isequal(V2V_Mask, V2V_Mask')
        error('V2V_Mask 矩阵非对称，请检查拓扑生成逻辑！');
    end
end

% -------------------------------------------------------------------------
% 函数: run_batch
% 功能: 在单个批次数据上同步执行 DEKF、DIEKF、DIEKF_ADMM、DMLKF 的预测与更新，并计算各车 RMSE
% -------------------------------------------------------------------------
function result = run_batch( ...
        trajectories, anchors, Anchor_Mask, V2V_Mask, ...
        Vehicle_num, Anchor_num, N_steps, bias_comp_ratio, Noise, ...
        DIEKF_max_iter, ...
        DMLKF_max_iter, DMLKF_beta_inv, DMLKF_print_flag, ...
        DIEKF_ADMM_iter, DIEKF_ADMM_GN_iter, DIEKF_ADMM_rho)
    
    dt_imu = trajectories.V1.Time_true(2) - trajectories.V1.Time_true(1);

    % --- 1. 提取全网车辆初始真实状态 ---
    p0 = zeros(3 * Vehicle_num, 1);
    v0 = zeros(3 * Vehicle_num, 1);
    R0 = zeros(3, 3, Vehicle_num);
    for i = 1:Vehicle_num
        v_name = sprintf('V%d', i);
        p0(3*i-2 : 3*i) = [
            trajectories.(v_name).X_true(1);
            trajectories.(v_name).Y_true(1);
            trajectories.(v_name).Z_true(1)];
        v0(3*i-2 : 3*i) = [
            trajectories.(v_name).Vx_true(1);
            trajectories.(v_name).Vy_true(1);
            trajectories.(v_name).Vz_true(1)];
        R0(:, :, i) = trajectories.(v_name).R_true(:, :, 1);
    end

    % --- 2. 实例化并配置四个滤波器 ---
    kf_dekf = DEKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);
    kf_dekf.IMU_Sigma_a   = Noise.IMU_Sigma_a;
    kf_dekf.IMU_Sigma_w   = Noise.IMU_Sigma_w;
    kf_dekf.UWB_sigma_anc = Noise.UWB_sigma_anc;
    kf_dekf.UWB_sigma_rel = Noise.UWB_sigma_rel;

    kf_diekf = DIEKF(Vehicle_num, Anchor_num, anchors, dt_imu, ...
        p0, v0, R0, DIEKF_max_iter, Noise);
    kf_diekf.IMU_Sigma_a   = Noise.IMU_Sigma_a;
    kf_diekf.IMU_Sigma_w   = Noise.IMU_Sigma_w;
    kf_diekf.UWB_sigma_anc = Noise.UWB_sigma_anc;
    kf_diekf.UWB_sigma_rel = Noise.UWB_sigma_rel;
    kf_diekf.max_iter      = DIEKF_max_iter;

    kf_diekf_admm = DIEKF_ADMM(Vehicle_num, Anchor_num, anchors, dt_imu, ...
        p0, v0, R0, V2V_Mask, DIEKF_ADMM_iter, Noise);
    kf_diekf_admm.IMU_Sigma_a   = Noise.IMU_Sigma_a;
    kf_diekf_admm.IMU_Sigma_w   = Noise.IMU_Sigma_w;
    kf_diekf_admm.UWB_sigma_anc = Noise.UWB_sigma_anc;
    kf_diekf_admm.UWB_sigma_rel = Noise.UWB_sigma_rel;
    kf_diekf_admm.ADMM_iter     = DIEKF_ADMM_iter;
    kf_diekf_admm.GN_iter       = DIEKF_ADMM_GN_iter;
    kf_diekf_admm.rho           = DIEKF_ADMM_rho;
    if isprop(kf_diekf_admm, 'M_in')
        kf_diekf_admm.M_in = DIEKF_ADMM_GN_iter;
    end
    if isprop(kf_diekf_admm, 'max_iter')
        kf_diekf_admm.max_iter = DIEKF_ADMM_iter;
    end

    kf_dmlkf = DMLKF(Vehicle_num, Anchor_num, anchors, dt_imu, ...
        p0, v0, R0, V2V_Mask, DMLKF_max_iter, Noise);
    kf_dmlkf.IMU_Sigma_a   = Noise.IMU_Sigma_a;
    kf_dmlkf.IMU_Sigma_w   = Noise.IMU_Sigma_w;
    kf_dmlkf.UWB_sigma_anc = Noise.UWB_sigma_anc;
    kf_dmlkf.UWB_sigma_rel = Noise.UWB_sigma_rel;
    kf_dmlkf.max_iter      = DMLKF_max_iter;
    kf_dmlkf.beta_inv      = DMLKF_beta_inv;
    kf_dmlkf.print_flag    = DMLKF_print_flag;

    % --- 3. 预分配状态估计存储空间并写入初值 ---
    est_p_dekf       = zeros(N_steps, 3, Vehicle_num);
    est_v_dekf       = zeros(N_steps, 3, Vehicle_num);
    est_R_dekf       = zeros(3, 3, Vehicle_num, N_steps);
    est_p_diekf      = zeros(N_steps, 3, Vehicle_num);
    est_v_diekf      = zeros(N_steps, 3, Vehicle_num);
    est_R_diekf      = zeros(3, 3, Vehicle_num, N_steps);
    est_p_diekf_admm = zeros(N_steps, 3, Vehicle_num);
    est_v_diekf_admm = zeros(N_steps, 3, Vehicle_num);
    est_R_diekf_admm = zeros(3, 3, Vehicle_num, N_steps);
    est_p_dmlkf      = zeros(N_steps, 3, Vehicle_num);
    est_v_dmlkf      = zeros(N_steps, 3, Vehicle_num);
    est_R_dmlkf      = zeros(3, 3, Vehicle_num, N_steps);

    for i = 1:Vehicle_num
        est_p_dekf(1, :, i)          = p0(3*i-2 : 3*i)';
        est_v_dekf(1, :, i)          = v0(3*i-2 : 3*i)';
        est_R_dekf(:, :, i, 1)       = R0(:, :, i);
        est_p_diekf(1, :, i)         = p0(3*i-2 : 3*i)';
        est_v_diekf(1, :, i)         = v0(3*i-2 : 3*i)';
        est_R_diekf(:, :, i, 1)      = R0(:, :, i);
        est_p_diekf_admm(1, :, i)    = p0(3*i-2 : 3*i)';
        est_v_diekf_admm(1, :, i)    = v0(3*i-2 : 3*i)';
        est_R_diekf_admm(:, :, i, 1) = R0(:, :, i);
        est_p_dmlkf(1, :, i)         = p0(3*i-2 : 3*i)';
        est_v_dmlkf(1, :, i)         = v0(3*i-2 : 3*i)';
        est_R_dmlkf(:, :, i, 1)      = R0(:, :, i);
    end

    % --- 4. 时间步主循环 (IMU 预测 + UWB 更新) ---
    uwb_idx      = 2;
    UWB_Time_Vec = trajectories.V1.UWB_Anchor(:, 1);
    for k = 2:N_steps
        acc_m  = zeros(3, Vehicle_num);
        gyro_m = zeros(3, Vehicle_num);
        for i = 1:Vehicle_num
            v_name      = sprintf('V%d', i);
            acc_m(:, i) = trajectories.(v_name).IMU_acc_m(k-1, :)' ...
                - bias_comp_ratio * trajectories.(v_name).IMU_bias_a_true(k-1, :)';
            gyro_m(:, i) = trajectories.(v_name).IMU_gyro_m(k-1, :)' ...
                - bias_comp_ratio * trajectories.(v_name).IMU_bias_w_true(k-1, :)';
        end

        kf_dekf.predict(acc_m, gyro_m);
        kf_diekf.predict(acc_m, gyro_m);
        kf_diekf_admm.predict(acc_m, gyro_m);
        kf_dmlkf.predict(acc_m, gyro_m);

        curr_time = trajectories.V1.Time_true(k);
        if uwb_idx <= length(UWB_Time_Vec) && ...
                abs(curr_time - UWB_Time_Vec(uwb_idx)) < 1e-5
            uwb_anc_raw = zeros(Vehicle_num, Anchor_num);
            uwb_rel_raw = zeros(Vehicle_num, Vehicle_num);
            for i = 1:Vehicle_num
                v_name            = sprintf('V%d', i);
                uwb_anc_raw(i, :) = ...
                    trajectories.(v_name).UWB_Anchor(uwb_idx, 2:end);
                uwb_rel_raw(i, :) = ...
                    trajectories.(v_name).UWB_Relative(uwb_idx, 2:end);
            end

            uwb_anc_masked                   = uwb_anc_raw;
            uwb_anc_masked(Anchor_Mask == 0) = NaN;
            uwb_rel_masked                   = uwb_rel_raw;
            uwb_rel_masked(V2V_Mask == 0)    = NaN;

            kf_dekf.update(uwb_anc_masked, uwb_rel_masked);
            kf_diekf.update(uwb_anc_masked, uwb_rel_masked);
            kf_diekf_admm.update(uwb_anc_masked, uwb_rel_masked);
            kf_dmlkf.update(uwb_anc_masked, uwb_rel_masked);
            uwb_idx = uwb_idx + 1;
        end

        for i = 1:Vehicle_num
            est_p_dekf(k, :, i)          = kf_dekf.Nodes{i}.p';
            est_v_dekf(k, :, i)          = kf_dekf.Nodes{i}.v';
            est_R_dekf(:, :, i, k)       = kf_dekf.Nodes{i}.R;
            est_p_diekf(k, :, i)         = kf_diekf.Nodes{i}.p';
            est_v_diekf(k, :, i)         = kf_diekf.Nodes{i}.v';
            est_R_diekf(:, :, i, k)      = kf_diekf.Nodes{i}.R;
            est_p_diekf_admm(k, :, i)    = kf_diekf_admm.Nodes{i}.p';
            est_v_diekf_admm(k, :, i)    = kf_diekf_admm.Nodes{i}.v';
            est_R_diekf_admm(:, :, i, k) = kf_diekf_admm.Nodes{i}.R;
            est_p_dmlkf(k, :, i)         = kf_dmlkf.Nodes{i}.p';
            est_v_dmlkf(k, :, i)         = kf_dmlkf.Nodes{i}.v';
            est_R_dmlkf(:, :, i, k)      = kf_dmlkf.Nodes{i}.R;
        end
    end

    % --- 5. 计算各车辆在当前批次内的全程 RMSE ---
    [result.rmse_p_dekf, rmse_v_dekf, rmse_att_dekf] = ...
        calc_rmse(trajectories, est_p_dekf, est_v_dekf, ...
        est_R_dekf, Vehicle_num, N_steps);
    [result.rmse_p_diekf, rmse_v_diekf, rmse_att_diekf] = ...
        calc_rmse(trajectories, est_p_diekf, est_v_diekf, ...
        est_R_diekf, Vehicle_num, N_steps);
    [result.rmse_p_diekf_admm, rmse_v_diekf_admm, rmse_att_diekf_admm] = ...
        calc_rmse(trajectories, est_p_diekf_admm, est_v_diekf_admm, ...
        est_R_diekf_admm, Vehicle_num, N_steps);
    [result.rmse_p_dmlkf, rmse_v_dmlkf, rmse_att_dmlkf] = ...
        calc_rmse(trajectories, est_p_dmlkf, est_v_dmlkf, ...
        est_R_dmlkf, Vehicle_num, N_steps);

    % 保留全状态指标，便于后续需要时扩展输出
    result.rmse_v_dekf         = rmse_v_dekf;
    result.rmse_v_diekf        = rmse_v_diekf;
    result.rmse_v_diekf_admm   = rmse_v_diekf_admm;
    result.rmse_v_dmlkf        = rmse_v_dmlkf;
    result.rmse_att_dekf       = rmse_att_dekf;
    result.rmse_att_diekf      = rmse_att_diekf;
    result.rmse_att_diekf_admm = rmse_att_diekf_admm;
    result.rmse_att_dmlkf      = rmse_att_dmlkf;
end

% -------------------------------------------------------------------------
% 函数: calc_rmse
% 功能: 计算每辆车在整个时间序列 (1~N_steps) 上的位置、速度和姿态 RMSE
% -------------------------------------------------------------------------
function [rmse_p, rmse_v, rmse_att] = calc_rmse( ...
        trajectories, est_p, est_v, est_R, Vehicle_num, N_steps)
    
    rmse_p   = zeros(Vehicle_num, 1);
    rmse_v   = zeros(Vehicle_num, 1);
    rmse_att = zeros(Vehicle_num, 1);
    for i = 1:Vehicle_num
        v_name = sprintf('V%d', i);
        true_p = [
            trajectories.(v_name).X_true(1:N_steps), ...
            trajectories.(v_name).Y_true(1:N_steps), ...
            trajectories.(v_name).Z_true(1:N_steps)];
        true_v = [
            trajectories.(v_name).Vx_true(1:N_steps), ...
            trajectories.(v_name).Vy_true(1:N_steps), ...
            trajectories.(v_name).Vz_true(1:N_steps)];

        rmse_p(i) = sqrt(mean(sum((est_p(:, :, i) - true_p).^2, 2)));
        rmse_v(i) = sqrt(mean(sum((est_v(:, :, i) - true_v).^2, 2)));

        err_att_seq = zeros(N_steps, 1);
        for k = 1:N_steps
            R_t            = trajectories.(v_name).R_true(:, :, k);
            R_e            = est_R(:, :, i, k);
            R_err          = R_t' * R_e;
            tr             = max(-1, min(3, trace(R_err)));
            err_att_seq(k) = acos((tr - 1) / 2) * (180 / pi);
        end
        rmse_att(i) = sqrt(mean(err_att_seq.^2));
    end
end