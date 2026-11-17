% compare_CEKF_CMLKF_CPF.m
% 在同一组多批次数据上比较 CEKF、CMLKF 和 CPF 在不同基站数量下的位置 RMSE。
% 三个算法共享相同的初始状态误差、基站观测掩码和输入测量。
% RMSE 计算逻辑：
% 1. 单车全程均方根：先计算每辆车在每个时刻的误差平方，在时间轴求平均后开根号；
% 2. 多批次平均：将每辆车的全程 RMSE 在各批次上求均值；
% 3. 算法最终平均：将各车的多批次平均 RMSE 再次求均值，得到最终评价指标。

clc; clear; close all;

%% ==================== 1. 测试参数配置 ====================
Vehicle_num    = 4;
Anchor_num     = 16;             % 数据集中的总基站数
Anchor_run_num = 3:10;           % 待测试的基站数量 (可为列表如 5:15 或单个数如 10)
dataset_id     = 'CMLKF_CPF_20_3';

run_flag    = 1;          % 当前脚本暂不读取历史结果
save_flag   = 0;          % 暂时不保存结果 (设为 1 则保存至 CSV)
figure_save = 0;          % 暂时不保存图片

bias_comp_ratio = 1.0;    % 1.0 表示完全补偿 IMU 零偏
data_ratio      = 1.0;    % 每个批次使用的数据比例

% --- CMLKF 参数 ---
cml_max_iter = 60;
cml_epsilon  = 1e-6;
cml_beta_inv = 100;
cml_max_step = Inf;

% --- CPF 参数 ---
rng_base_seed = 20261028;

cpf_params.num_particles          = 1000;
cpf_params.neff_ratio             = 0.6; % 重采样频率阈值
cpf_params.temper_ess_ratio       = 0.8; % 退火步长 ESS 阈值
cpf_params.n_mcmc                 = 10;  % 每个退火阶段 MH 次数
cpf_params.depletion_streak_limit = 3;

% CPF 噪声参数
cpf_params.noise.IMU_Sigma_a   = (0.05)^2 * eye(3);
cpf_params.noise.IMU_Sigma_w   = (0.005)^2 * eye(3);
cpf_params.noise.UWB_sigma_anc = 0.01;
cpf_params.noise.UWB_sigma_rel = 0.01;

% 校验 Anchor_run_num 合法性
Anchor_run_num = Anchor_run_num(:)';
if isempty(Anchor_run_num) || any(Anchor_run_num < 1) || ...
        any(Anchor_run_num > Anchor_num) || any(round(Anchor_run_num) ~= Anchor_run_num)
    error('Anchor_run_num 必须是 [1, %d] 范围内的正整数或正整数列表。', Anchor_num);
end

fprintf('\n========== 开始运行 CEKF、CMLKF 与 CPF 算法对比 ==========\n');
fprintf('目标数据集: 车辆 %d, 数据集总基站 %d, 测试基站数: [%s], 后缀标识: %s\n', ...
    Vehicle_num, Anchor_num, num2str(Anchor_run_num), dataset_id);

%% ==================== 2. 路径与文件配置 ====================
test_dir      = fileparts(mfilename('fullpath'));
compare_root  = fileparts(test_dir);
repo_root     = fileparts(fileparts(compare_root));

data_dir      = fullfile(test_dir, 'data');
result_dir    = fullfile(test_dir, 'result');

if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

addpath(fullfile(repo_root, 'EKF', 'CEKF'));
addpath(fullfile(repo_root, 'MLKF', 'CMLKF'));
addpath(fullfile(repo_root, 'PF', 'CPF'));

data_file   = fullfile(data_dir, sprintf( ...
    'Trj_Veh%d_Anc%d_%s.mat', Vehicle_num, Anchor_num, dataset_id));
result_file = fullfile(result_dir, sprintf( ...
    '%s_Veh%d_Anc%d_to_%d.csv', dataset_id, Vehicle_num, ...
    Anchor_run_num(1), Anchor_run_num(end)));

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

% CEKF/CMLKF/CPF 使用数据文件中的 IMU/UWB 噪声参数。
ekf_imu_sigma_a   = S.IMU_noise_params.sigma_na^2 * eye(3);
ekf_imu_sigma_w   = S.IMU_noise_params.sigma_nw^2 * eye(3);
ekf_uwb_sigma_anc = S.UWB_noise_params.sigma_anc;
ekf_uwb_sigma_rel = S.UWB_noise_params.sigma_rel;

fprintf('数据集加载成功：%d 个批次，每批 %d 步。\n', ...
    repeat_count, N_steps_use);

%% ==================== 4. 逐基站数量、逐批次运行三个滤波器 ====================
num_anc_configs = numel(Anchor_run_num);
summary_CEKF    = zeros(num_anc_configs, 1);
summary_CMLKF   = zeros(num_anc_configs, 1);
summary_CPF     = zeros(num_anc_configs, 1);

for a_idx = 1:num_anc_configs
    curr_anc_num = Anchor_run_num(a_idx);
    fprintf('\n-----------------------------------------------------------\n');
    fprintf('>>> 正在测试基站数量: %d / %d (进度: %d/%d) <<<\n', ...
        curr_anc_num, Anchor_num, a_idx, num_anc_configs);

    % 构建当前基站数量下的观测掩码 (保留前 curr_anc_num 个基站，其余置 0 屏蔽)
    anchor_mask = false(1, Anchor_num);
    anchor_mask(1:curr_anc_num) = true;

    % 用于存储当前基站数下各批次、各车的全程 RMSE。维度：(repeat_count, Vehicle_num)
    CEKF_batch_veh_rmse  = zeros(repeat_count, Vehicle_num);
    CMLKF_batch_veh_rmse = zeros(repeat_count, Vehicle_num);
    CPF_batch_veh_rmse   = zeros(repeat_count, Vehicle_num);

    for r = 1:repeat_count
        fprintf('  正在处理第 %d/%d 批次数据 (基站数=%d)...\n', ...
            r, repeat_count, curr_anc_num);
        
        trajectories = extract_batch(S.trajectories, r, S.N_steps, N_uwb_per_batch);
        batch_anchors = S.anchors_all(:, :, r);
        initial_position_error = S.scenario.initial_position_error(:, :, r);
        batch_seed = rng_base_seed + r - 1;
        
        % CEKF
        CEKF_batch_veh_rmse(r, :) = run_filter('CEKF', trajectories, batch_anchors, ...
            anchor_mask, N_steps_use, bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
            attitude_std_deg, initial_position_error, ...
            cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step, ...
            cpf_params, batch_seed, ...
            ekf_imu_sigma_a, ekf_imu_sigma_w, ...
            ekf_uwb_sigma_anc, ekf_uwb_sigma_rel);
        
        % CMLKF
        CMLKF_batch_veh_rmse(r, :) = run_filter('CMLKF', trajectories, batch_anchors, ...
            anchor_mask, N_steps_use, bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
            attitude_std_deg, initial_position_error, ...
            cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step, ...
            cpf_params, batch_seed, ...
            ekf_imu_sigma_a, ekf_imu_sigma_w, ...
            ekf_uwb_sigma_anc, ekf_uwb_sigma_rel);
        
        % CPF
        CPF_batch_veh_rmse(r, :) = run_filter('CPF', trajectories, batch_anchors, ...
            anchor_mask, N_steps_use, bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
            attitude_std_deg, initial_position_error, ...
            cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step, ...
            cpf_params, batch_seed, ...
            ekf_imu_sigma_a, ekf_imu_sigma_w, ...
            ekf_uwb_sigma_anc, ekf_uwb_sigma_rel);
    end

    % 步骤2：各车在多批次上的 RMSE 平均 (1 x Vehicle_num)
    CEKF_veh_avg  = mean(CEKF_batch_veh_rmse, 1);
    CMLKF_veh_avg = mean(CMLKF_batch_veh_rmse, 1);
    CPF_veh_avg   = mean(CPF_batch_veh_rmse, 1);

    % 步骤3：各算法对各车再做平均，得到当前基站数下该算法的最终 RMSE
    CEKF_final_rmse  = mean(CEKF_veh_avg);
    CMLKF_final_rmse = mean(CMLKF_veh_avg);
    CPF_final_rmse   = mean(CPF_veh_avg);

    summary_CEKF(a_idx)  = CEKF_final_rmse;
    summary_CMLKF(a_idx) = CMLKF_final_rmse;
    summary_CPF(a_idx)   = CPF_final_rmse;

    % 每个基站数完成后立即打印一次该基站数下的结果
    fprintf('\n========== 基站数 = %d 时的算法对比 RMSE ==========\n', curr_anc_num);
    fprintf('数据集：Veh%d_Anc%d (启用 %d 基站)，后缀标识=%s\n', ...
        Vehicle_num, Anchor_num, curr_anc_num, dataset_id);
    fprintf('Vehicle ID |   CEKF   |  CMLKF   |   CPF \n');
    fprintf('-------------------------------------------\n');
    for i = 1:Vehicle_num
        fprintf('   V%d      |  %.4f  |  %.4f  |  %.4f\n', ...
            i, CEKF_veh_avg(i), CMLKF_veh_avg(i), CPF_veh_avg(i));
    end
    fprintf('-------------------------------------------\n');
    fprintf(' 算法最终  |  %.4f  |  %.4f  |  %.4f\n', ...
        CEKF_final_rmse, CMLKF_final_rmse, CPF_final_rmse);
    fprintf('===================================================\n');
end

%% ==================== 5. 汇总结果输出与保存 ====================
if num_anc_configs > 1
    fprintf('\n================ 基站数量变化下三算法最终 RMSE 汇总 (单位: m) ================\n');
    fprintf(' Anchor Num |    CEKF    |   CMLKF    |    CPF     \n');
    fprintf('------------------------------------------------------------------------------\n');
    for a_idx = 1:num_anc_configs
        fprintf('     %2d     |   %.4f   |   %.4f   |   %.4f\n', ...
            Anchor_run_num(a_idx), summary_CEKF(a_idx), ...
            summary_CMLKF(a_idx), summary_CPF(a_idx));
    end
    fprintf('==============================================================================\n\n');
end

if save_flag
    Anchor_Num = Anchor_run_num(:);
    CEKF_RMSE  = summary_CEKF;
    CMLKF_RMSE = summary_CMLKF;
    CPF_RMSE   = summary_CPF;
    T = table(Anchor_Num, CEKF_RMSE, CMLKF_RMSE, CPF_RMSE);
    writetable(T, result_file);
    fprintf('---> 结果摘要数据已成功保存至:\n%s\n', result_file);
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
        anchor_mask, N_steps, bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
        attitude_std_deg, initial_position_error, ...
        cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step, ...
        cpf_params, batch_seed, ...
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
        case 'CPF'
            rng(batch_seed, 'twister');
            kf = CPF(Vehicle_num, Anchor_num, anchors, dt_imu, ...
                p0, v0, R0);
            configure_standard_noise(kf, ekf_imu_sigma_a, ...
                ekf_imu_sigma_w, ekf_uwb_sigma_anc, ekf_uwb_sigma_rel);
            configure_cpf(kf, cpf_params);
        otherwise
            error('未知滤波器类型：%s', method_name);
    end
    
    % 三个算法使用相同的位置、速度和姿态先验标准差。
    initialize_filter_prior(kf, method_name, prior_std_vec, ...
        velocity_prior_std, attitude_std_deg);
    
    % 用于记录每个车在每个时刻的平方误差
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
            % 统一应用基站数量掩码 (未启用的基站观测置为 NaN)
            uwb_anc(:, ~anchor_mask) = NaN;
            
            kf.update(uwb_anc, uwb_rel);
            uwb_idx = uwb_idx + 1;
        end
        
        % 记录当前时刻各车的误差平方
        err_squared_series(k, :) = get_vehicle_sq_err(kf.p, trajectories, k, Vehicle_num);
    end
    
    % 对每个车，在时间维度上求平均均方根误差，返回 1 x Vehicle_num 向量
    veh_rmse = sqrt(mean(err_squared_series, 1));
end

function sq_errs = get_vehicle_sq_err(position, trajectories, k, Vehicle_num)
    % 返回一个 1 x Vehicle_num 的向量，代表各个车在 k 时刻的位置误差平方
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

function configure_standard_noise(kf, imu_sigma_a, imu_sigma_w, ...
        uwb_sigma_anc, uwb_sigma_rel)
    % 所有噪声参数均由当前脚本显式写入。
    kf.IMU_Sigma_a   = imu_sigma_a;
    kf.IMU_Sigma_w   = imu_sigma_w;
    kf.UWB_sigma_anc = uwb_sigma_anc;
    kf.UWB_sigma_rel = uwb_sigma_rel;
end

function configure_cpf(kf, cpf_params)
    kf.set_particle_count(cpf_params.num_particles);
    kf.neff_ratio             = cpf_params.neff_ratio;
    kf.temper_ess_ratio       = cpf_params.temper_ess_ratio;
    kf.n_mcmc                 = cpf_params.n_mcmc;
    kf.depletion_streak_limit = cpf_params.depletion_streak_limit;
end

function initialize_filter_prior(kf, method_name, prior_std_vec, ...
        velocity_prior_std, attitude_std_deg)

    if strcmp(method_name, 'CPF')
        kf.init_prior(prior_std_vec, velocity_prior_std, deg2rad(attitude_std_deg));
    else
        prior_block = blkdiag(diag(prior_std_vec.^2), ...
            velocity_prior_std^2 * eye(3), ...
            deg2rad(attitude_std_deg)^2 * eye(3));
        prior_blocks = repmat({prior_block}, 1, kf.Vehicle_num);
        kf.Sigma = blkdiag(prior_blocks{:});
    end
end