% DGN_DN_Test.m 
% 测试分布式高斯牛顿 (D-GN) 与 分布式精确牛顿 (D-N) 
% 使用 DMLKF_C 对比，目的是在iter_compare_main上让分布式的趋于集中式
clc; clear; close all;

%% 1. 测试参数与运行配置
Vehicle_num = 8;            
Anchor_num  = 4;     
neighbor_k  = 4; % 邻居数

% 故意保留一定未知零偏以破坏先验，凸显纯测距优化优势 (1.0为完全补偿)
bias_comp_ratio = 1; 

% 控制截取数据集的比例
data_ratio  = 0.2;

% 路径配置
data_dir = 'E:\DMLKF_code\TEST\GN_compare\Data';
data_file = fullfile(data_dir, sprintf('Trj_Veh%d_Anc%d_pure.mat', Vehicle_num, Anchor_num));

%% 2. 加载数据集
if ~exist(data_file, 'file')
    error('未找到数据集文件: %s\n请先运行生成脚本生成该数据！', data_file);
end
load(data_file, 'trajectories', 'anchors', 'IMU_noise_params', 'UWB_noise_params');
fprintf('数据集加载成功，开始对比 D-GN 与 D-N 算法...\n');

N_steps_total = length(trajectories.V1.Time_true);
N_steps = max(2, round(N_steps_total * data_ratio)); 
fprintf('>> 实验设定的数据截取比例: %.1f%%\n', data_ratio * 100);
fprintf('>> 实际运行步数 / 总步数: %d / %d\n', N_steps, N_steps_total);
dt_imu = trajectories.V1.Time_true(2) - trajectories.V1.Time_true(1);

%% 3. 生成静态拓扑掩码
% A. 基站掩码 (修改：取消分级制度，所有车辆均可观测所有基站)
Anchor_Mask = ones(Vehicle_num, Anchor_num);

% B. 车间掩码 (静态环形拓扑)
K_degree = min(neighbor_k, Vehicle_num - 1); 
V2V_Mask = zeros(Vehicle_num, Vehicle_num);
K_fwd = ceil(K_degree / 2); 
K_bwd = floor(K_degree / 2); 
for i = 1:Vehicle_num
    for d = 1:K_fwd
        idx_forward = mod(i + d - 1, Vehicle_num) + 1;
        V2V_Mask(i, idx_forward) = 1;
    end
    for d = 1:K_bwd
        idx_backward = mod(i - d - 1, Vehicle_num) + 1;
        V2V_Mask(i, idx_backward) = 1;
    end
end
V2V_Mask(logical(eye(Vehicle_num))) = 0;

%% 4. 初始化滤波器 (使用 DMLKF_C)
p0 = zeros(3 * Vehicle_num, 1);
v0 = zeros(3 * Vehicle_num, 1);
R0 = zeros(3, 3, Vehicle_num);
for i = 1:Vehicle_num
    v_name = sprintf('V%d', i);
    p0(3*i-2 : 3*i) = [trajectories.(v_name).X_true(1); trajectories.(v_name).Y_true(1); trajectories.(v_name).Z_true(1)];
    v0(3*i-2 : 3*i) = [trajectories.(v_name).Vx_true(1); trajectories.(v_name).Vy_true(1); trajectories.(v_name).Vz_true(1)];
    R0(:, :, i)     = trajectories.(v_name).R_true(:, :, 1); 
end

max_iter = 50; % 分布式最大迭代次数

% 分别实例化两个类 (使用同一套代码，仅改变 is_GN 标志)
kf_DGN = DMLKF_C(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0, V2V_Mask, max_iter);
kf_DN  = DMLKF_C(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0, V2V_Mask, max_iter);

% [核心设置] 控制优化方法
kf_DGN.is_GN = 1; % 使用分布式高斯-牛顿
kf_DN.is_GN  = 0; % 使用分布式精确牛顿 (含二阶残差项)

% 测试修改参数 (DN需要更大的beta_inv 且在UWB参数符合实际时优于DGN)

kf_DN.max_step = Inf;
kf_DN.beta_inv = 10;

kf_DGN.beta_inv = 0.1;
kf_DGN.max_step = Inf; 


% 仅存储位置结果用于对比
est_p_DGN = zeros(N_steps, 3, Vehicle_num);
est_p_DN  = zeros(N_steps, 3, Vehicle_num);
for i = 1:Vehicle_num
    est_p_DGN(1, :, i) = p0(3*i-2 : 3*i)';
    est_p_DN(1, :, i)  = p0(3*i-2 : 3*i)';
end

%% 5. 滤波主循环 (双算法同步运行)
uwb_idx = 2; 
UWB_Time_Vec = trajectories.V1.UWB_Anchor(:, 1);
fprintf('开始双算法迭代仿真...\n');
for k = 2:N_steps
    % --- A. IMU 预测提取 ---
    acc_m = zeros(3, Vehicle_num);
    gyro_m = zeros(3, Vehicle_num);
    for i = 1:Vehicle_num
        v_name = sprintf('V%d', i);
        acc_m(:, i)  = trajectories.(v_name).IMU_acc_m(k-1, :)' - bias_comp_ratio * trajectories.(v_name).IMU_bias_a_true(k-1, :)';
        gyro_m(:, i) = trajectories.(v_name).IMU_gyro_m(k-1, :)' - bias_comp_ratio * trajectories.(v_name).IMU_bias_w_true(k-1, :)';
    end
    
    % 执行预测
    kf_DGN.predict(acc_m, gyro_m);
    kf_DN.predict(acc_m, gyro_m);
    
    % --- B. UWB 更新过程 ---
    curr_time = trajectories.V1.Time_true(k);
    if uwb_idx <= length(UWB_Time_Vec) && abs(curr_time - UWB_Time_Vec(uwb_idx)) < 1e-5
        uwb_anc_raw = zeros(Vehicle_num, Anchor_num);
        uwb_rel_raw = zeros(Vehicle_num, Vehicle_num);
        for i = 1:Vehicle_num
            v_name = sprintf('V%d', i);
            uwb_anc_raw(i, :) = trajectories.(v_name).UWB_Anchor(uwb_idx, 2:end);
            uwb_rel_raw(i, :) = trajectories.(v_name).UWB_Relative(uwb_idx, 2:end);
        end
        
        uwb_anc_masked = uwb_anc_raw;  uwb_anc_masked(Anchor_Mask == 0) = NaN;
        uwb_rel_masked = uwb_rel_raw;  uwb_rel_masked(V2V_Mask == 0) = NaN;
        
        % 执行更新
        kf_DGN.update(uwb_anc_masked, uwb_rel_masked);
        kf_DN.update(uwb_anc_masked, uwb_rel_masked);
        
        uwb_idx = uwb_idx + 1;
    end
    
    % --- C. 提取位置结果 ---
    for i = 1:Vehicle_num
        est_p_DGN(k, :, i) = kf_DGN.Nodes{i}.p';
        est_p_DN(k, :, i)  = kf_DN.Nodes{i}.p';
    end
    
    if mod(k, 2000) == 0
        fprintf('已处理: %d / %d 步...\n', k, N_steps);
    end
end

%% 6. 计算 RMSE 差异并打印
rmse_p_DGN = zeros(Vehicle_num, 1);
rmse_p_DN  = zeros(Vehicle_num, 1);

for i = 1:Vehicle_num
    v_name = sprintf('V%d', i);
    true_p = [trajectories.(v_name).X_true(1:N_steps), ...
              trajectories.(v_name).Y_true(1:N_steps), ...
              trajectories.(v_name).Z_true(1:N_steps)];
              
    rmse_p_DGN(i) = sqrt(mean(sum((est_p_DGN(:, :, i) - true_p).^2, 2)));
    rmse_p_DN(i)  = sqrt(mean(sum((est_p_DN(:, :, i)  - true_p).^2, 2)));
end
mean_rmse_DGN = mean(rmse_p_DGN);
mean_rmse_DN  = mean(rmse_p_DN);

print_comparison(Vehicle_num, rmse_p_DGN, rmse_p_DN, mean_rmse_DGN, mean_rmse_DN);

%% ==== 局部打印对比函数 ====
function print_comparison(Vehicle_num, rmse_DGN, rmse_DN, m_DGN, m_DN)
    fprintf('\n======================== 位置 RMSE (m) 对比 ========================\n');
    fprintf(' VehID  |  DGN (高斯-牛顿) |   DN (精确牛顿)  |   Diff (DGN - DN)\n');
    fprintf('--------------------------------------------------------------------\n');
    for i = 1:Vehicle_num
        diff = rmse_DGN(i) - rmse_DN(i);
        fprintf('  %2d    |      %6.4f       |      %6.4f      |    %6.4f\n', i, rmse_DGN(i), rmse_DN(i), diff);
    end
    fprintf('--------------------------------------------------------------------\n');
    diff_m = m_DGN - m_DN;
    fprintf('  Avg   |      %6.4f       |      %6.4f      |    %6.4f\n', m_DGN, m_DN, diff_m);
    fprintf('====================================================================\n');
end