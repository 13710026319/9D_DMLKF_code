% compare_DEKF_DIEKF_DMLKF.m
% 三算法同台对比测试脚本：DEKF vs. DIEKF (IEKF-ADMM) vs. DMLKF
clc; clear; close all;

%% 1. 测试场景与公共参数配置
Vehicle_num = 8;            
Anchor_num  = 4;     
neighbor_k  = 4;   % 邻居数

% 0.7为保留 30% 未知零偏以破坏先验 (1.0为完全补偿)
bias_comp_ratio = 0.7; 

% 控制截取数据集的比例 (例如 0.2 表示只跑前 20% 的数据，1.0为全部)
data_ratio = 0.1;

dataset_id      = 'DEKF_DIEKF_DMLKF_100_1'; 

% 统一传感器噪声参数配置
Noise.IMU_Sigma_a   = (0.05)^2 * eye(3);
Noise.IMU_Sigma_w   = (0.005)^2 * eye(3);
Noise.UWB_sigma_anc = 0.1;
Noise.UWB_sigma_rel = 0.1;

% 迭代与正则化参数配置
DMLKF_max_iter  = 50;
DMLKF_beta_inv  = 50;

DIEKF_ADMM_iter = 300;
DIEKF_GN_iter   = 5;
DIEKF_rho       = 50;

% 数据集路径配置
data_dir  = 'E:\DMLKF_code\TEST\D_compare\DEKF_DIEKF_DMLKF\data';
data_file = fullfile(data_dir, sprintf('Trj_Veh%d_Anc%d_%s.mat', Vehicle_num, Anchor_num, dataset_id));

%% 2. 加载数据集
if ~exist(data_file, 'file')
    error('未找到数据集文件: %s\n请先运行生成脚本生成该数据！', data_file);
end
load(data_file, 'trajectories', 'anchors', 'IMU_noise_params', 'UWB_noise_params');
fprintf('数据集加载成功: %s\n', data_file);

N_steps_total = length(trajectories.V1.Time_true);
N_steps = max(2, round(N_steps_total * data_ratio));
fprintf('>> 实验设定的数据截取比例: %.1f%% (运行步数: %d / %d)\n', data_ratio * 100, N_steps, N_steps_total);
dt_imu = trajectories.V1.Time_true(2) - trajectories.V1.Time_true(1);

%% 3. 生成静态拓扑掩码 (三档基站 + 对称环形车间拓扑)
% A. 基站掩码 (30% 全基站, 50% 半基站, 20% 无基站)
N_tier1 = round(0.3 * Vehicle_num);
N_tier2 = round(0.5 * Vehicle_num);
Anchor_Mask = ones(Vehicle_num, Anchor_num);
for i = 1:Vehicle_num
    if i <= N_tier1
        % Tier 1: 保持全 1
    elseif i <= N_tier1 + N_tier2
        % Tier 2: 仅保留偶数基站
        for anc_idx = 1:Anchor_num
            if mod(anc_idx, 2) ~= 0, Anchor_Mask(i, anc_idx) = 0; end
        end
    else
        % Tier 3: 纯相对测距
        Anchor_Mask(i, :) = 0;
    end
end

% B. 车间掩码 (保证严格对称的静态环形拓扑)
K_degree = min(neighbor_k, Vehicle_num - 1); 
if mod(Vehicle_num, 2) ~= 0 && mod(K_degree, 2) ~= 0
    error(['图论限制: 车辆总数(%d)与邻居数(%d)均为奇数，数学上无法构成完全对称的双向测距拓扑！\n' ...
           ' ---> 解决方案: 请将 neighbor_k 或 Vehicle_num 修改为偶数。'], Vehicle_num, K_degree);
end

V2V_Mask = zeros(Vehicle_num, Vehicle_num);
K_half = floor(K_degree / 2);

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

%% 4. 初始化三个滤波器 (DEKF / DIEKF / DMLKF)
p0 = zeros(3 * Vehicle_num, 1);
v0 = zeros(3 * Vehicle_num, 1);
R0 = zeros(3, 3, Vehicle_num);
for i = 1:Vehicle_num
    v_name = sprintf('V%d', i);
    p_true_init = [trajectories.(v_name).X_true(1); trajectories.(v_name).Y_true(1); trajectories.(v_name).Z_true(1)];
    v_true_init = [trajectories.(v_name).Vx_true(1); trajectories.(v_name).Vy_true(1); trajectories.(v_name).Vz_true(1)];
    
    p0(3*i-2 : 3*i) = p_true_init(:);
    v0(3*i-2 : 3*i) = v_true_init(:);
    R0(:, :, i)     = trajectories.(v_name).R_true(:, :, 1); 
end

% --- (1) 实例化并配置 DEKF ---
kf_dekf = DEKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);
kf_dekf.IMU_Sigma_a   = Noise.IMU_Sigma_a;
kf_dekf.IMU_Sigma_w   = Noise.IMU_Sigma_w;
kf_dekf.UWB_sigma_anc = Noise.UWB_sigma_anc;
kf_dekf.UWB_sigma_rel = Noise.UWB_sigma_rel;

% --- (2) 实例化并配置 DIEKF ---
kf_diekf = DIEKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0, V2V_Mask, DIEKF_ADMM_iter, Noise);
kf_diekf.IMU_Sigma_a   = Noise.IMU_Sigma_a;
kf_diekf.IMU_Sigma_w   = Noise.IMU_Sigma_w;
kf_diekf.UWB_sigma_anc = Noise.UWB_sigma_anc;
kf_diekf.UWB_sigma_rel = Noise.UWB_sigma_rel;
kf_diekf.ADMM_iter     = DIEKF_ADMM_iter;
kf_diekf.GN_iter       = DIEKF_GN_iter;
kf_diekf.rho           = DIEKF_rho;
if isprop(kf_diekf, 'M_in'),     kf_diekf.M_in     = DIEKF_GN_iter;   end
if isprop(kf_diekf, 'max_iter'), kf_diekf.max_iter = DIEKF_ADMM_iter; end

% --- (3) 实例化并配置 DMLKF ---
kf_dmlkf = DMLKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0, V2V_Mask, DMLKF_max_iter, Noise);
kf_dmlkf.IMU_Sigma_a   = Noise.IMU_Sigma_a;
kf_dmlkf.IMU_Sigma_w   = Noise.IMU_Sigma_w;
kf_dmlkf.UWB_sigma_anc = Noise.UWB_sigma_anc;
kf_dmlkf.UWB_sigma_rel = Noise.UWB_sigma_rel;
kf_dmlkf.max_iter      = DMLKF_max_iter;
kf_dmlkf.beta_inv      = DMLKF_beta_inv;

% --- 结果存储空间初始化 ---
est_p_dekf  = zeros(N_steps, 3, Vehicle_num);
est_v_dekf  = zeros(N_steps, 3, Vehicle_num);
est_R_dekf  = zeros(3, 3, Vehicle_num, N_steps);

est_p_diekf = zeros(N_steps, 3, Vehicle_num);
est_v_diekf = zeros(N_steps, 3, Vehicle_num);
est_R_diekf = zeros(3, 3, Vehicle_num, N_steps);

est_p_dmlkf = zeros(N_steps, 3, Vehicle_num);
est_v_dmlkf = zeros(N_steps, 3, Vehicle_num);
est_R_dmlkf = zeros(3, 3, Vehicle_num, N_steps);

for i = 1:Vehicle_num
    est_p_dekf(1, :, i)  = p0(3*i-2 : 3*i)';
    est_v_dekf(1, :, i)  = v0(3*i-2 : 3*i)';
    est_R_dekf(:, :, i, 1)  = R0(:, :, i);
    
    est_p_diekf(1, :, i) = p0(3*i-2 : 3*i)';
    est_v_diekf(1, :, i) = v0(3*i-2 : 3*i)';
    est_R_diekf(:, :, i, 1) = R0(:, :, i);
    
    est_p_dmlkf(1, :, i) = p0(3*i-2 : 3*i)';
    est_v_dmlkf(1, :, i) = v0(3*i-2 : 3*i)';
    est_R_dmlkf(:, :, i, 1) = R0(:, :, i);
end

%% 5. 滤波主循环 (同步推进三个算法)
uwb_idx = 2; 
UWB_Time_Vec = trajectories.V1.UWB_Anchor(:, 1);
fprintf('开始同步运行 DEKF / DIEKF / DMLKF (共 %d 步)...\n', N_steps);

for k = 2:N_steps
    % --- A. 100Hz 预测过程 ---
    acc_m  = zeros(3, Vehicle_num);
    gyro_m = zeros(3, Vehicle_num);
    for i = 1:Vehicle_num
        v_name = sprintf('V%d', i);
        acc_m(:, i)  = trajectories.(v_name).IMU_acc_m(k-1, :)'  - bias_comp_ratio * trajectories.(v_name).IMU_bias_a_true(k-1, :)';
        gyro_m(:, i) = trajectories.(v_name).IMU_gyro_m(k-1, :)' - bias_comp_ratio * trajectories.(v_name).IMU_bias_w_true(k-1, :)';
    end
    
    kf_dekf.predict(acc_m, gyro_m);
    kf_diekf.predict(acc_m, gyro_m);
    kf_dmlkf.predict(acc_m, gyro_m);
    
    % --- B. 10Hz 更新过程 (严格应用相同的拓扑掩码) ---
    curr_time = trajectories.V1.Time_true(k);
    if uwb_idx <= length(UWB_Time_Vec) && abs(curr_time - UWB_Time_Vec(uwb_idx)) < 1e-5
        uwb_anc_raw = zeros(Vehicle_num, Anchor_num);
        uwb_rel_raw = zeros(Vehicle_num, Vehicle_num);
        for i = 1:Vehicle_num
            v_name = sprintf('V%d', i);
            uwb_anc_raw(i, :) = trajectories.(v_name).UWB_Anchor(uwb_idx, 2:end);
            uwb_rel_raw(i, :) = trajectories.(v_name).UWB_Relative(uwb_idx, 2:end);
        end
        
        uwb_anc_masked = uwb_anc_raw;
        uwb_anc_masked(Anchor_Mask == 0) = NaN;
        
        uwb_rel_masked = uwb_rel_raw;
        uwb_rel_masked(V2V_Mask == 0) = NaN;
        
        kf_dekf.update(uwb_anc_masked, uwb_rel_masked);
        kf_diekf.update(uwb_anc_masked, uwb_rel_masked);
        kf_dmlkf.update(uwb_anc_masked, uwb_rel_masked);
        
        uwb_idx = uwb_idx + 1;
    end
    
    % --- C. 提取三个算法的当前步估计结果 ---
    for i = 1:Vehicle_num
        est_p_dekf(k, :, i)     = kf_dekf.Nodes{i}.p';
        est_v_dekf(k, :, i)     = kf_dekf.Nodes{i}.v';
        est_R_dekf(:, :, i, k)  = kf_dekf.Nodes{i}.R;
        
        est_p_diekf(k, :, i)    = kf_diekf.Nodes{i}.p';
        est_v_diekf(k, :, i)    = kf_diekf.Nodes{i}.v';
        est_R_diekf(:, :, i, k) = kf_diekf.Nodes{i}.R;
        
        est_p_dmlkf(k, :, i)    = kf_dmlkf.Nodes{i}.p';
        est_v_dmlkf(k, :, i)    = kf_dmlkf.Nodes{i}.v';
        est_R_dmlkf(:, :, i, k) = kf_dmlkf.Nodes{i}.R;
    end
    
    if mod(k, 1000) == 0
        fprintf('已处理: %d / %d 步...\n', k, N_steps);
    end
end

%% 6. 计算三个算法的 RMSE
[rmse_p_dekf,  rmse_v_dekf,  rmse_att_dekf]  = calc_rmse(trajectories, est_p_dekf,  est_v_dekf,  est_R_dekf,  Vehicle_num, N_steps);
[rmse_p_diekf, rmse_v_diekf, rmse_att_diekf] = calc_rmse(trajectories, est_p_diekf, est_v_diekf, est_R_diekf, Vehicle_num, N_steps);
[rmse_p_dmlkf, rmse_v_dmlkf, rmse_att_dmlkf] = calc_rmse(trajectories, est_p_dmlkf, est_v_dmlkf, est_R_dmlkf, Vehicle_num, N_steps);

%% 7. 打印对比结果
fprintf('\n==================== 三算法位置 RMSE (Pos, 单位: m) 对比 ====================\n');
fprintf(' VehID  |    DEKF     |    DIEKF    |    DMLKF    \n');
fprintf('-----------------------------------------------------------------------------\n');
for i = 1:Vehicle_num
    fprintf('  %2d    |   %8.4f  |   %8.4f  |   %8.4f\n', i, rmse_p_dekf(i), rmse_p_diekf(i), rmse_p_dmlkf(i));
end
fprintf('-----------------------------------------------------------------------------\n');
fprintf('  Avg   |   %8.4f  |   %8.4f  |   %8.4f\n', mean(rmse_p_dekf), mean(rmse_p_diekf), mean(rmse_p_dmlkf));
fprintf('=============================================================================\n');

fprintf('\n==================== 三算法全状态平均 RMSE 汇总 =============================\n');
fprintf(' Algorithm |  Avg Pos (m)  |  Avg Vel (m/s)  |  Avg Att (deg)\n');
fprintf('-----------------------------------------------------------------------------\n');
fprintf('   DEKF    |    %8.4f   |     %8.4f    |     %8.4f\n', mean(rmse_p_dekf),  mean(rmse_v_dekf),  mean(rmse_att_dekf));
fprintf('   DIEKF   |    %8.4f   |     %8.4f    |     %8.4f\n', mean(rmse_p_diekf), mean(rmse_v_diekf), mean(rmse_att_diekf));
fprintf('   DMLKF   |    %8.4f   |     %8.4f    |     %8.4f\n', mean(rmse_p_dmlkf), mean(rmse_v_dmlkf), mean(rmse_att_dmlkf));
fprintf('=============================================================================\n\n');

%% ==== 局部辅助函数：计算单算法 RMSE ====
function [rmse_p, rmse_v, rmse_att] = calc_rmse(trajectories, est_p, est_v, est_R, Vehicle_num, N_steps)
    rmse_p   = zeros(Vehicle_num, 1);
    rmse_v   = zeros(Vehicle_num, 1);
    rmse_att = zeros(Vehicle_num, 1);
    for i = 1:Vehicle_num
        v_name = sprintf('V%d', i);
        true_p = [trajectories.(v_name).X_true(1:N_steps), ...
                  trajectories.(v_name).Y_true(1:N_steps), ...
                  trajectories.(v_name).Z_true(1:N_steps)];
        true_v = [trajectories.(v_name).Vx_true(1:N_steps), ...
                  trajectories.(v_name).Vy_true(1:N_steps), ...
                  trajectories.(v_name).Vz_true(1:N_steps)];
        
        rmse_p(i) = sqrt(mean(sum((est_p(:, :, i) - true_p).^2, 2)));
        rmse_v(i) = sqrt(mean(sum((est_v(:, :, i) - true_v).^2, 2)));
        
        err_att_seq = zeros(N_steps, 1);
        for k = 1:N_steps
            R_t = trajectories.(v_name).R_true(:, :, k);
            R_e = est_R(:, :, i, k);
            R_err = R_t' * R_e;
            tr = max(-1, min(3, trace(R_err)));
            err_att_seq(k) = acos((tr - 1) / 2) * (180 / pi);
        end
        rmse_att(i) = sqrt(mean(err_att_seq.^2));
    end
end