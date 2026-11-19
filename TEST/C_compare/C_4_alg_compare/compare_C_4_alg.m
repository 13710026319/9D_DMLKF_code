% compare_C_4_alg.m
% 在同一组多批次数据上比较 CEKF、CIEKF、CMLKF 和 CPF 在不同基站数量下的位置 RMSE。
% 四个算法共享相同的初始状态误差、基站观测掩码、车间相对测距掩码和输入测量。
% 注意：该数据集生成后的数据目前测试效果很差，需要进一步修改此数据集！！

clc; clear; close all;

%% ==================== 1. 测试参数配置 ====================
Vehicle_num    = 1;
Anchor_num     = 12;             % 数据集中的总基站数
Anchor_run_num = 3:12;           % 待测试的基站数量 (可为列表如 4:12 或单个数如 10)
neighbor_num   = 0;              % 每辆车的相对测距邻居数量 (取值范围: 0 ~ Vehicle_num-1)
Anchor_mark    = 0;              % 0: Anchor_run_num表示每车基站数; 1: 表示全网所有车的总基站测距条数
dataset_id     = 'C_4_alg_40_II';

run_flag    = 1;          % 1: 重新运行滤波实验; 0: 直接读取已有 CSV 结果打印并作图
save_flag   = 1;          % 1: 保存结果至 CSV; 0: 不保存
figure_save = 1;          % 1: 保存图片至 PNG; 0: 仅显示不保存

bias_comp_ratio = 0.5;    % 1.0 表示完全补偿 IMU 零偏
data_ratio      = 1.0;    % 每个批次使用的数据比例

% --- CMLKF 参数 ---
cml_max_iter = 20;
cml_epsilon  = 1e-6;
cml_beta_inv = 10;         
cml_max_step = 1;

% --- CIEKF 参数 ---
iekf_max_iter = 20;

% --- CPF 参数 ---
rng_base_seed = 20261028;
cpf_params.num_particles          = 1000;
cpf_params.neff_ratio             = 0.6; % 重采样频率阈值
cpf_params.temper_ess_ratio       = 0.8; % 退火步长 ESS 阈值
cpf_params.n_mcmc                 = 10;  % 每个退火阶段 MH 次数
cpf_params.depletion_streak_limit = 3;

% --- 参数合法性校验 ---
Anchor_run_num = Anchor_run_num(:)';
if ~ismember(Anchor_mark, [0, 1])
    error('Anchor_mark 必须为 0 或 1。');
end
if Anchor_mark == 0
    max_anc_limit = Anchor_num;
else
    max_anc_limit = Vehicle_num * Anchor_num;
end
if isempty(Anchor_run_num) || any(Anchor_run_num < 1) || ...
        any(Anchor_run_num > max_anc_limit) || any(round(Anchor_run_num) ~= Anchor_run_num)
    error('当前 Anchor_mark=%d 下，Anchor_run_num 必须是 [1, %d] 范围内的正整数或列表。', ...
        Anchor_mark, max_anc_limit);
end
if isempty(neighbor_num) || ~isscalar(neighbor_num) || ...
        neighbor_num < 0 || neighbor_num > Vehicle_num - 1 || ...
        round(neighbor_num) ~= neighbor_num
    error('neighbor_num 必须是 [0, %d] 范围内的整数。', Vehicle_num - 1);
end

%% ==================== 2. 路径配置与 run_flag == 0 读取逻辑 ====================
test_dir     = fileparts(mfilename('fullpath'));
compare_root = fileparts(test_dir);
repo_root    = fileparts(fileparts(compare_root));

data_dir   = fullfile(test_dir, 'data');
result_dir = fullfile(test_dir, 'result');
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

base_result_name = sprintf('%s_Veh%d_Anc%d_to_%d', ...
    dataset_id, Vehicle_num, Anchor_run_num(1), Anchor_run_num(end));
data_file   = fullfile(data_dir, sprintf( ...
    'Trj_Veh%d_Anc%d_%s.mat', Vehicle_num, Anchor_num, dataset_id));
result_file = fullfile(result_dir, [base_result_name, '.csv']);
figure_file = fullfile(result_dir, [base_result_name, '.png']);

% 若 run_flag == 0，直接检查并读取已有 CSV 结果打印及绘图
if run_flag == 0
    if ~exist(result_file, 'file')
        error('未找到当前配置下的历史结果文件：\n%s\n请先将 run_flag 设为 1 运行实验！', result_file);
    end
    fprintf('检测到历史结果文件，正在加载：\n%s\n', result_file);
    T = readtable(result_file);
    if ~ismember('CIEKF_RMSE', T.Properties.VariableNames)
        error('已有 CSV 结果文件中缺少 CIEKF_RMSE 列，请将 run_flag 设为 1 重新运行生成！');
    end
    print_summary_table(T.Anchor_Num, T.CEKF_RMSE, T.CIEKF_RMSE, T.CMLKF_RMSE, T.CPF_RMSE);
    plot_anchor_rmse(T.Anchor_Num, T.CPF_RMSE, T.CMLKF_RMSE, T.CIEKF_RMSE, T.CEKF_RMSE, ...
        figure_save, figure_file);
    return;
end

addpath(fullfile(repo_root, 'EKF', 'CEKF'));
addpath(fullfile(repo_root, 'IEKF', 'CIEKF'));
addpath(fullfile(repo_root, 'MLKF', 'CMLKF'));
addpath(fullfile(repo_root, 'PF', 'CPF'));

fprintf('\n========== 开始运行 CEKF、CIEKF、CMLKF 与 CPF 四算法对比 ==========\n');
fprintf('目标数据集: 车辆 %d (邻居数 %d), 数据集总基站 %d, 测试基站数: [%s], 后缀标识: %s\n', ...
    Vehicle_num, neighbor_num, Anchor_num, num2str(Anchor_run_num), dataset_id);

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

dt_imu_sample     = S.trajectories.V1.Time_true(2) - S.trajectories.V1.Time_true(1);
N_steps_from_tend = round(S.t_end / dt_imu_sample) + 1;
if N_steps_from_tend ~= S.N_steps
    error('数据集中的 N_steps 与 t_end、采样周期不一致。');
end

repeat_count = S.N_steps_total / N_steps_from_tend;
if round(repeat_count) ~= repeat_count || round(repeat_count) ~= S.repeat_time
    error('数据集总步数或重复次数校验失败。');
end
repeat_count = round(repeat_count);

N_steps_use     = max(2, round(S.N_steps * data_ratio));
N_uwb_per_batch = size(S.trajectories.V1.UWB_Anchor, 1) / repeat_count;

prior_std_vec = S.scenario.position_prior_std_vec;
if isfield(S.scenario, 'velocity_prior_std')
    velocity_prior_std = S.scenario.velocity_prior_std;
else
    velocity_prior_std = 0.2;
end
if isfield(S.scenario, 'attitude_prior_std_deg')
    attitude_std_deg = S.scenario.attitude_prior_std_deg;
else
    attitude_std_deg = 10.0;
end

% 四个算法统一使用数据文件中的真值 IMU/UWB 噪声参数
ekf_imu_sigma_a   = S.IMU_noise_params.sigma_na^2 * eye(3);
ekf_imu_sigma_w   = S.IMU_noise_params.sigma_nw^2 * eye(3);
ekf_uwb_sigma_anc = S.UWB_noise_params.sigma_anc;
ekf_uwb_sigma_rel = S.UWB_noise_params.sigma_rel;

% 构建车间相对测距掩码
rel_mask = build_rel_mask(Vehicle_num, neighbor_num);

fprintf('数据集加载成功：%d 个批次，每批 %d 步。\n', repeat_count, N_steps_use);

%% ==================== 4. 逐基站数量、逐批次运行四个滤波器 ====================
num_anc_configs = numel(Anchor_run_num);
summary_CEKF    = zeros(num_anc_configs, 1);
summary_CIEKF   = zeros(num_anc_configs, 1);
summary_CMLKF   = zeros(num_anc_configs, 1);
summary_CPF     = zeros(num_anc_configs, 1);

for a_idx = 1:num_anc_configs
    curr_anc_num = Anchor_run_num(a_idx);
    fprintf('\n-----------------------------------------------------------\n');
    
    % 构建 Vehicle_num x Anchor_num 的二维基站观测掩码
    anchor_mask = false(Vehicle_num, Anchor_num);
    if Anchor_mark == 0
        % 模式 0：每辆车均取前 curr_anc_num 个基站
        anchor_mask(:, 1:curr_anc_num) = true;
        fprintf('>>> 正在测试每车基站数量: %d / %d (进度: %d/%d) <<<\n', ...
            curr_anc_num, Anchor_num, a_idx, num_anc_configs);
    else
        % 模式 1：curr_anc_num 为全网总基站测距数，均分给各车，余数按顺序分给前 rem_cnt 辆车
        base_cnt = floor(curr_anc_num / Vehicle_num);
        rem_cnt  = mod(curr_anc_num, Vehicle_num);
        for v_i = 1:Vehicle_num
            v_anc_cnt = base_cnt + (v_i <= rem_cnt);
            if v_anc_cnt > 0
                anchor_mask(v_i, 1:v_anc_cnt) = true;
            end
        end
        fprintf('>>> 正在测试全网总基站测距数: %d / %d (进度: %d/%d) <<<\n', ...
            curr_anc_num, Vehicle_num * Anchor_num, a_idx, num_anc_configs);
    end

    CEKF_batch_veh_rmse  = zeros(repeat_count, Vehicle_num);
    CIEKF_batch_veh_rmse = zeros(repeat_count, Vehicle_num);
    CMLKF_batch_veh_rmse = zeros(repeat_count, Vehicle_num);
    CPF_batch_veh_rmse   = zeros(repeat_count, Vehicle_num);

    for r = 1:repeat_count
        fprintf('  正在处理第 %d/%d 批次数据 (配置=%d)...\n', ...
            r, repeat_count, curr_anc_num);
        
        trajectories           = extract_batch(S.trajectories, r, S.N_steps, N_uwb_per_batch);
        batch_anchors          = S.anchors_all(:, :, r);
        initial_position_error = S.scenario.initial_position_error(:, :, r);
        if isfield(S.scenario, 'initial_velocity_error')
            initial_velocity_error = S.scenario.initial_velocity_error(:, :, r);
        else
            initial_velocity_error = zeros(3, Vehicle_num);
        end
        if isfield(S.scenario, 'initial_attitude_error')
            initial_attitude_error = S.scenario.initial_attitude_error(:, :, r);
        else
            initial_attitude_error = zeros(3, Vehicle_num);
        end
        batch_seed = rng_base_seed + r - 1;
        
        % CEKF
        CEKF_batch_veh_rmse(r, :) = run_filter('CEKF', trajectories, batch_anchors, ...
            anchor_mask, rel_mask, N_steps_use, bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
            attitude_std_deg, initial_position_error, initial_velocity_error, initial_attitude_error, ...
            cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step, ...
            iekf_max_iter, cpf_params, batch_seed, ...
            ekf_imu_sigma_a, ekf_imu_sigma_w, ekf_uwb_sigma_anc, ekf_uwb_sigma_rel);
        
        % CIEKF
        CIEKF_batch_veh_rmse(r, :) = run_filter('CIEKF', trajectories, batch_anchors, ...
            anchor_mask, rel_mask, N_steps_use, bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
            attitude_std_deg, initial_position_error, initial_velocity_error, initial_attitude_error, ...
            cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step, ...
            iekf_max_iter, cpf_params, batch_seed, ...
            ekf_imu_sigma_a, ekf_imu_sigma_w, ekf_uwb_sigma_anc, ekf_uwb_sigma_rel);
        
        % CMLKF
        CMLKF_batch_veh_rmse(r, :) = run_filter('CMLKF', trajectories, batch_anchors, ...
            anchor_mask, rel_mask, N_steps_use, bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
            attitude_std_deg, initial_position_error, initial_velocity_error, initial_attitude_error, ...
            cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step, ...
            iekf_max_iter, cpf_params, batch_seed, ...
            ekf_imu_sigma_a, ekf_imu_sigma_w, ekf_uwb_sigma_anc, ekf_uwb_sigma_rel);
        
        % CPF
        CPF_batch_veh_rmse(r, :) = run_filter('CPF', trajectories, batch_anchors, ...
            anchor_mask, rel_mask, N_steps_use, bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
            attitude_std_deg, initial_position_error, initial_velocity_error, initial_attitude_error, ...
            cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step, ...
            iekf_max_iter, cpf_params, batch_seed, ...
            ekf_imu_sigma_a, ekf_imu_sigma_w, ekf_uwb_sigma_anc, ekf_uwb_sigma_rel);
    end

    CEKF_veh_avg  = mean(CEKF_batch_veh_rmse, 1);
    CIEKF_veh_avg = mean(CIEKF_batch_veh_rmse, 1);
    CMLKF_veh_avg = mean(CMLKF_batch_veh_rmse, 1);
    CPF_veh_avg   = mean(CPF_batch_veh_rmse, 1);

    summary_CEKF(a_idx)  = mean(CEKF_veh_avg);
    summary_CIEKF(a_idx) = mean(CIEKF_veh_avg);
    summary_CMLKF(a_idx) = mean(CMLKF_veh_avg);
    summary_CPF(a_idx)   = mean(CPF_veh_avg);

    % 每个基站数完成后立即打印一次该基站数下的各车及平均结果
    fprintf('\n============== 基站配置 = %d 时的四算法对比 RMSE ==============\n', curr_anc_num);
    fprintf('数据集：Veh%d_Anc%d (配置=%d, 邻居数=%d)，后缀=%s\n', ...
        Vehicle_num, Anchor_num, curr_anc_num, neighbor_num, dataset_id);
    fprintf('Vehicle ID |   CEKF   |  CIEKF   |  CMLKF   |   CPF \n');
    fprintf('------------------------------------------------------\n');
    for i = 1:Vehicle_num
        fprintf('   V%d      |  %.4f  |  %.4f  |  %.4f  |  %.4f\n', ...
            i, CEKF_veh_avg(i), CIEKF_veh_avg(i), CMLKF_veh_avg(i), CPF_veh_avg(i));
    end
    fprintf('------------------------------------------------------\n');
    fprintf(' 算法最终  |  %.4f  |  %.4f  |  %.4f  |  %.4f\n', ...
        summary_CEKF(a_idx), summary_CIEKF(a_idx), summary_CMLKF(a_idx), summary_CPF(a_idx));
    fprintf('==============================================================\n');
end

%% ==================== 5. 汇总结果输出、保存与绘图 ====================
print_summary_table(Anchor_run_num, summary_CEKF, summary_CIEKF, summary_CMLKF, summary_CPF);

if save_flag
    Anchor_Num = Anchor_run_num(:);
    CEKF_RMSE  = summary_CEKF;
    CIEKF_RMSE = summary_CIEKF;
    CMLKF_RMSE = summary_CMLKF;
    CPF_RMSE   = summary_CPF;
    T = table(Anchor_Num, CEKF_RMSE, CIEKF_RMSE, CMLKF_RMSE, CPF_RMSE);
    writetable(T, result_file);
    fprintf('---> 结果摘要数据已成功保存至:\n%s\n', result_file);
end

plot_anchor_rmse(Anchor_run_num, summary_CPF, summary_CMLKF, summary_CIEKF, summary_CEKF, ...
    figure_save, figure_file);

%% ======================= 局部函数区域 =======================
function print_summary_table(anchor_nums, rmse_ekf, rmse_iekf, rmse_mlkf, rmse_pf)
    fprintf('\n==================== 基站数量变化下四算法最终 RMSE 汇总 (单位: m) ====================\n');
    fprintf(' Anchor Num |    CEKF    |   CIEKF    |   CMLKF    |    CPF     \n');
    fprintf('--------------------------------------------------------------------------------------\n');
    for i = 1:numel(anchor_nums)
        fprintf('     %2d     |   %.4f   |   %.4f   |   %.4f   |   %.4f\n', ...
            anchor_nums(i), rmse_ekf(i), rmse_iekf(i), rmse_mlkf(i), rmse_pf(i));
    end
    fprintf('======================================================================================\n\n');
end

function plot_anchor_rmse(anchor_nums, rmse_pf, rmse_mlkf, rmse_iekf, rmse_ekf, ...
        figure_save, figure_file)
    figure('Name', 'RMSE vs Anchor Number', 'Position', [150, 150, 720, 480], ...
        'Color', 'w', 'Visible', 'on');
    hold on; grid on; box on;

    plot(anchor_nums, rmse_pf,   '-s', 'LineWidth', 1.8, 'MarkerSize', 7, 'DisplayName', 'PF');
    plot(anchor_nums, rmse_mlkf, '-o', 'LineWidth', 1.8, 'MarkerSize', 7, 'DisplayName', 'MLKF');
    plot(anchor_nums, rmse_iekf, '-d', 'LineWidth', 1.8, 'MarkerSize', 7, 'DisplayName', 'IEKF');
    plot(anchor_nums, rmse_ekf,  '-^', 'LineWidth', 1.8, 'MarkerSize', 7, 'DisplayName', 'EKF');

    xticks(anchor_nums(:)');
    if numel(anchor_nums) > 1
        xlim([min(anchor_nums), max(anchor_nums)]);
    end
    xlabel('Anchor Number');
    ylabel('RMSE');
    legend('Location', 'best');
    hold off;
    drawnow;

    if figure_save
        exportgraphics(gcf, figure_file, 'Resolution', 300);
        fprintf('---> 对比曲线图片已成功保存至:\n%s\n', figure_file);
    end
end

function rel_mask = build_rel_mask(Vehicle_num, neighbor_num)
    rel_mask = false(Vehicle_num, Vehicle_num);
    if neighbor_num == 0
        return;
    end

    K_half = floor(neighbor_num / 2);
    for i = 1:Vehicle_num
        for d = 1:K_half
            idx_fwd = mod(i + d - 1, Vehicle_num) + 1;
            idx_bwd = mod(i - d - 1, Vehicle_num) + 1;
            rel_mask(i, idx_fwd) = true;
            rel_mask(i, idx_bwd) = true;
        end
        if mod(neighbor_num, 2) ~= 0
            if mod(Vehicle_num, 2) == 0
                idx_extra = mod(i + Vehicle_num/2 - 1, Vehicle_num) + 1;
            else
                idx_extra = mod(i + K_half, Vehicle_num) + 1;
            end
            rel_mask(i, idx_extra) = true;
        end
    end
    rel_mask(1:Vehicle_num+1:end) = false;
end

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

function veh_rmse = run_filter(method_name, trajectories, anchors, ...
        anchor_mask, rel_mask, N_steps, bias_comp_ratio, prior_std_vec, velocity_prior_std, ...
        attitude_std_deg, initial_position_error, initial_velocity_error, initial_attitude_error, ...
        cml_max_iter, cml_epsilon, cml_beta_inv, cml_max_step, ...
        iekf_max_iter, cpf_params, batch_seed, ...
        ekf_imu_sigma_a, ekf_imu_sigma_w, ...
        ekf_uwb_sigma_anc, ekf_uwb_sigma_rel)
    
    Vehicle_num = numel(fieldnames(trajectories));
    Anchor_num  = size(anchors, 1);
    dt_imu      = trajectories.V1.Time_true(2) - trajectories.V1.Time_true(1);
    
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
                         trajectories.(name).Vz_true(1)] + ...
                         initial_velocity_error(:, i);
        R0(:, :, i)   = trajectories.(name).R_true(:, :, 1) * ...
                        CMLKF.exp_SO3(initial_attitude_error(:, i));
    end
    
    switch method_name
        case 'CEKF'
            kf = CEKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);
            configure_standard_noise(kf, ekf_imu_sigma_a, ...
                ekf_imu_sigma_w, ekf_uwb_sigma_anc, ekf_uwb_sigma_rel);
        case 'CIEKF'
            kf = CIEKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);
            kf.max_iter = iekf_max_iter;
            kf.tol      = cml_epsilon;
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
            kf = CPF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);
            configure_standard_noise(kf, ekf_imu_sigma_a, ...
                ekf_imu_sigma_w, ekf_uwb_sigma_anc, ekf_uwb_sigma_rel);
            configure_cpf(kf, cpf_params);
        otherwise
            error('未知滤波器类型：%s', method_name);
    end
    
    initialize_filter_prior(kf, method_name, prior_std_vec, ...
        velocity_prior_std, attitude_std_deg);
    
    err_squared_series = zeros(N_steps, Vehicle_num);
    err_squared_series(1, :) = get_vehicle_sq_err(kf.p, trajectories, 1, Vehicle_num);
    
    uwb_time = trajectories.V1.UWB_Anchor(:, 1);
    uwb_idx  = 2;
    for k = 2:N_steps
        acc_m  = zeros(3, Vehicle_num);
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
                uwb_anc(i, :) = trajectories.(name).UWB_Anchor(uwb_idx, 2:end);
                uwb_rel(i, :) = trajectories.(name).UWB_Relative(uwb_idx, 2:end);
            end
            % 统一应用二维基站掩码与车间相对测距掩码
            uwb_anc(~anchor_mask) = NaN;
            uwb_rel(~rel_mask)    = NaN;
            
            kf.update(uwb_anc, uwb_rel);
            uwb_idx = uwb_idx + 1;
        end
        
        err_squared_series(k, :) = get_vehicle_sq_err(kf.p, trajectories, k, Vehicle_num);
    end
    
    veh_rmse = sqrt(mean(err_squared_series, 1));
end

function sq_errs = get_vehicle_sq_err(position, trajectories, k, Vehicle_num)
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