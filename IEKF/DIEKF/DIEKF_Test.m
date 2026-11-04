% DIEKF_Test.m 
% 同步测试 DIEKF_ADMM 与 纯局部 DIEKF 并分别输出 RMSE 结果
clc; clear; close all;

%% 1. 测试参数与运行配置
Vehicle_num = 8;            
Anchor_num  = 4;     
neighbor_k  = 4; % 邻居数

% 0.7为保留 30% 未知零偏以破坏先验 (1.0为完全补偿)
bias_comp_ratio = 1; 

% 控制截取数据集的比例 (例如 0.3 表示只跑前 30% 的数据，1.0为全部)
data_ratio  = 0.3;

% 路径配置
data_dir  = 'E:\DMLKF_code\Data';
data_file = fullfile(data_dir, sprintf('Trj_Veh%d_Anc%d_3D.mat', Vehicle_num, Anchor_num));

%% 2. 加载数据集
if ~exist(data_file, 'file')
    error('未找到数据集文件: %s\n请先运行生成脚本生成该数据！', data_file);
end
load(data_file, 'trajectories', 'anchors', 'IMU_noise_params', 'UWB_noise_params');
fprintf('数据集加载成功，开始运行 DIEKF_ADMM 与 DIEKF 算法...\n');

N_steps_total = length(trajectories.V1.Time_true);
N_steps = max(2, round(N_steps_total * data_ratio)); % 计算截断步数(至少保证有2步)
fprintf('>> 实验设定的数据截取比例: %.1f%%\n', data_ratio * 100);
fprintf('>> 实际运行步数 / 总步数: %d / %d\n', N_steps, N_steps_total);
dt_imu = trajectories.V1.Time_true(2) - trajectories.V1.Time_true(1);

%% 3. 生成静态拓扑掩码 (三档基站 + K-Regular 车间)
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

% B. 车间掩码 (对称环形拓扑)
K_degree = min(neighbor_k, Vehicle_num - 1); 

if mod(Vehicle_num, 2) ~= 0 && mod(K_degree, 2) ~= 0
    error(['图论限制: 车辆总数(%d)与邻居数(%d)均为奇数，数学上无法构成完全对称的双向测距拓扑！\n' ...
           ' ---> 解决方案: 请在脚本顶部的测试参数中，将 neighbor_k 修改为偶数，或者将 Vehicle_num 修改为偶数。'], ...
           Vehicle_num, K_degree);
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

%% 4. 初始化 DIEKF_ADMM 与 DIEKF 滤波器
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

% (1) 实例化并配置 DIEKF_ADMM
kf_admm = DIEKF_ADMM(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0, V2V_Mask);
kf_admm.IMU_Sigma_a   = (0.07)^2 * eye(3);
kf_admm.IMU_Sigma_w   = (0.007)^2 * eye(3);
kf_admm.UWB_sigma_anc = 0.1;
kf_admm.UWB_sigma_rel = 0.1;
kf_admm.rho           = 50;
kf_admm.ADMM_iter     = 300;
kf_admm.GN_iter       = 5;

% (2) 实例化并配置 纯局部 DIEKF
kf_diekf = DIEKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);
kf_diekf.IMU_Sigma_a   = (0.07)^2 * eye(3);
kf_diekf.IMU_Sigma_w   = (0.007)^2 * eye(3);
kf_diekf.UWB_sigma_anc = 0.1;
kf_diekf.UWB_sigma_rel = 0.1;
kf_diekf.max_iter      = 50;

% 结果存储空间
est_p_admm  = zeros(N_steps, 3, Vehicle_num);
est_v_admm  = zeros(N_steps, 3, Vehicle_num);
est_R_admm  = zeros(3, 3, Vehicle_num, N_steps);

est_p_diekf = zeros(N_steps, 3, Vehicle_num);
est_v_diekf = zeros(N_steps, 3, Vehicle_num);
est_R_diekf = zeros(3, 3, Vehicle_num, N_steps);

for i = 1:Vehicle_num
    est_p_admm(1, :, i)  = p0(3*i-2 : 3*i)';
    est_v_admm(1, :, i)  = v0(3*i-2 : 3*i)';
    est_R_admm(:, :, i, 1)  = R0(:, :, i);
    
    est_p_diekf(1, :, i) = p0(3*i-2 : 3*i)';
    est_v_diekf(1, :, i) = v0(3*i-2 : 3*i)';
    est_R_diekf(:, :, i, 1) = R0(:, :, i);
end

%% 5. 滤波主循环
uwb_idx = 2; 
UWB_Time_Vec = trajectories.V1.UWB_Anchor(:, 1);
fprintf('开始迭代仿真 (共 %d 步)...\n', N_steps);
for k = 2:N_steps
    % --- A. 100Hz 预测过程 ---
    acc_m  = zeros(3, Vehicle_num);
    gyro_m = zeros(3, Vehicle_num);
    for i = 1:Vehicle_num
        v_name = sprintf('V%d', i);
        acc_m(:, i)  = trajectories.(v_name).IMU_acc_m(k-1, :)'  - bias_comp_ratio * trajectories.(v_name).IMU_bias_a_true(k-1, :)';
        gyro_m(:, i) = trajectories.(v_name).IMU_gyro_m(k-1, :)' - bias_comp_ratio * trajectories.(v_name).IMU_bias_w_true(k-1, :)';
    end
    kf_admm.predict(acc_m, gyro_m);
    kf_diekf.predict(acc_m, gyro_m);
    
    % --- B. 10Hz 更新过程 (严格应用掩码过滤) ---
    curr_time = trajectories.V1.Time_true(k);
    if uwb_idx <= length(UWB_Time_Vec) && abs(curr_time - UWB_Time_Vec(uwb_idx)) < 1e-5
        uwb_anc_raw = zeros(Vehicle_num, Anchor_num);
        uwb_rel_raw = zeros(Vehicle_num, Vehicle_num);
        for i = 1:Vehicle_num
            v_name = sprintf('V%d', i);
            uwb_anc_raw(i, :) = trajectories.(v_name).UWB_Anchor(uwb_idx, 2:end);
            uwb_rel_raw(i, :) = trajectories.(v_name).UWB_Relative(uwb_idx, 2:end);
        end
        
        % 应用拓扑掩码，屏蔽不连通的边
        uwb_anc_masked = uwb_anc_raw;
        uwb_anc_masked(Anchor_Mask == 0) = NaN;
        
        uwb_rel_masked = uwb_rel_raw;
        uwb_rel_masked(V2V_Mask == 0) = NaN;
        
        % 输入两个滤波器
        kf_admm.update(uwb_anc_masked, uwb_rel_masked);
        kf_diekf.update(uwb_anc_masked, uwb_rel_masked);
        uwb_idx = uwb_idx + 1;
    end
    
    % --- C. 提取当前步沙盒内的局部估计结果 ---
    for i = 1:Vehicle_num
        est_p_admm(k, :, i)     = kf_admm.Nodes{i}.p';
        est_v_admm(k, :, i)     = kf_admm.Nodes{i}.v';
        est_R_admm(:, :, i, k)  = kf_admm.Nodes{i}.R;
        
        est_p_diekf(k, :, i)    = kf_diekf.Nodes{i}.p';
        est_v_diekf(k, :, i)    = kf_diekf.Nodes{i}.v';
        est_R_diekf(:, :, i, k) = kf_diekf.Nodes{i}.R;
    end
    
    if mod(k, 2000) == 0
        fprintf('已处理: %d / %d 步...\n', k, N_steps);
    end
end

%% 6. 分别计算并输出两个算法的 RMSE
[rmse_p_admm,  rmse_v_admm,  rmse_att_admm]  = calc_rmse(trajectories, est_p_admm,  est_v_admm,  est_R_admm,  Vehicle_num, N_steps);
[rmse_p_diekf, rmse_v_diekf, rmse_att_diekf] = calc_rmse(trajectories, est_p_diekf, est_v_diekf, est_R_diekf, Vehicle_num, N_steps);

print_results('DIEKF_ADMM', Vehicle_num, rmse_p_admm,  rmse_v_admm,  rmse_att_admm,  mean(rmse_p_admm),  mean(rmse_v_admm),  mean(rmse_att_admm));
print_results('DIEKF',      Vehicle_num, rmse_p_diekf, rmse_v_diekf, rmse_att_diekf, mean(rmse_p_diekf), mean(rmse_v_diekf), mean(rmse_att_diekf));

%% ==== 局部辅助函数 ====
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

function print_results(alg_name, Vehicle_num, rmse_p, rmse_v, rmse_att, m_p, m_v, m_att)
    fprintf('\n================ %s RMSE 结果 ================\n', alg_name);
    fprintf(' VehID  |  Pos(m)  |  Vel(m/s)  |  Att(deg)\n');
    fprintf('-----------------------------------------------------\n');
    for i = 1:Vehicle_num
        fprintf('  %2d    |  %6.4f  |   %6.4f   |  %6.4f\n', i, rmse_p(i), rmse_v(i), rmse_att(i));
    end
    fprintf('-----------------------------------------------------\n');
    fprintf('  Avg   |  %6.4f  |   %6.4f   |  %6.4f\n', m_p, m_v, m_att);
    fprintf('=====================================================\n\n');
end