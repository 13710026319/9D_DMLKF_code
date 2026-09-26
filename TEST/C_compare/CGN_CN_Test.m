% CGN_CN_Test.m 
% CMLKF 集中式高斯牛顿 (CGN) 与 集中式精确牛顿 (CN) 对比测试脚本
clc; clear; close all;

%% 1. 测试参数配置
Vehicle_num = 8;            
Anchor_num = 4;             

% 故意保留一定的 IMU 零偏 (1.0 为完全补偿，0.6 表示残留 40% 的偏置)
% 破坏系统先验置信度，更容易凸显不同测距优化算法的优劣
bias_comp_ratio = 1; 

% 控制截取数据集的比例 (例如 0.2 表示只跑前 20% 的数据，1.0为全部)
data_ratio = 0.2; 

% 路径配置 (使用你指定的数据集)
data_dir = 'E:\DMLKF_code\TEST\GN_compare\Data';
data_file = fullfile(data_dir, sprintf('Trj_Veh%d_Anc%d_subpure_1.mat', Vehicle_num, Anchor_num));
% data_dir = 'E:\DMLKF_code\Data';
% data_file = fullfile(data_dir, sprintf('Trj_Veh%d_Anc%d_3D.mat', Vehicle_num, Anchor_num));
%% 2. 加载数据集
if ~exist(data_file, 'file')
    error('未找到数据集文件: %s\n请确认路径及文件是否存在！', data_file);
end
load(data_file, 'trajectories', 'anchors', 'IMU_noise_params', 'UWB_noise_params');
fprintf('数据集加载成功，开始对比 CGN 与 CN 算法...\n');

N_steps_total = length(trajectories.V1.Time_true);
N_steps = max(2, round(N_steps_total * data_ratio)); % 计算截断步数
fprintf('>> 实验设定的数据截取比例: %.1f%%\n', data_ratio * 100);
fprintf('>> 实际运行步数 / 总步数: %d / %d\n', N_steps, N_steps_total);

dt_imu = trajectories.V1.Time_true(2) - trajectories.V1.Time_true(1);

%% 3. 初始化滤波器 (CGN 与 CN)
p0 = zeros(3 * Vehicle_num, 1);
v0 = zeros(3 * Vehicle_num, 1);
R0 = zeros(3, 3, Vehicle_num);
for i = 1:Vehicle_num
    v_name = sprintf('V%d', i);
    p0(3*i-2 : 3*i) = [trajectories.(v_name).X_true(1); trajectories.(v_name).Y_true(1); trajectories.(v_name).Z_true(1)];
    v0(3*i-2 : 3*i) = [trajectories.(v_name).Vx_true(1); trajectories.(v_name).Vy_true(1); trajectories.(v_name).Vz_true(1)];
    R0(:, :, i)     = trajectories.(v_name).R_true(:, :, 1);
end

% 实例化两个对象
cmlkf_cgn = CMLKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);
cmlkf_cn  = CMLKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);

% [核心设置] 将 cn 实例的优化方法标志位置为 0，启用精确牛顿法 (包含二阶导数项)
cmlkf_cn.is_GN = 0; 
cmlkf_cn.beta_inv = 1;
cmlkf_cn.max_iter = 1;

cmlkf_cgn.beta_inv = 0.1;
cmlkf_cgn.max_iter = 10;

% 预分配估计结果存储空间 (仅保留位置用于对比)
est_p_cgn = zeros(N_steps, 3, Vehicle_num);
est_p_cn  = zeros(N_steps, 3, Vehicle_num);

for i = 1:Vehicle_num
    est_p_cgn(1, :, i) = p0(3*i-2 : 3*i)';
    est_p_cn(1, :, i)  = p0(3*i-2 : 3*i)';
end

%% 4. 滤波主循环 (双算法同步运行)
uwb_idx = 2; 
UWB_Time_Vec = trajectories.V1.UWB_Anchor(:, 1);
fprintf('开始迭代仿真...\n');

for k = 2:N_steps
    % --- A. 100Hz 预测过程 ---
    acc_m = zeros(3, Vehicle_num);
    gyro_m = zeros(3, Vehicle_num);
    for i = 1:Vehicle_num
        v_name = sprintf('V%d', i);
        % 注入带有残留零偏的 IMU 数据
        acc_m(:, i)  = trajectories.(v_name).IMU_acc_m(k-1, :)' - bias_comp_ratio * trajectories.(v_name).IMU_bias_a_true(k-1, :)';
        gyro_m(:, i) = trajectories.(v_name).IMU_gyro_m(k-1, :)' - bias_comp_ratio * trajectories.(v_name).IMU_bias_w_true(k-1, :)';
    end
    
    % 两者分别预测
    cmlkf_cgn.predict(acc_m, gyro_m);
    cmlkf_cn.predict(acc_m, gyro_m);
    
    % --- B. 10Hz 更新过程 ---
    curr_time = trajectories.V1.Time_true(k);
    if uwb_idx <= length(UWB_Time_Vec) && abs(curr_time - UWB_Time_Vec(uwb_idx)) < 1e-5
        uwb_anc = zeros(Vehicle_num, Anchor_num);
        uwb_rel = zeros(Vehicle_num, Vehicle_num);
        for i = 1:Vehicle_num
            v_name = sprintf('V%d', i);
            uwb_anc(i, :) = trajectories.(v_name).UWB_Anchor(uwb_idx, 2:end);
            uwb_rel(i, :) = trajectories.(v_name).UWB_Relative(uwb_idx, 2:end);
        end
        
        % 两者分别融合更新
        cmlkf_cgn.update(uwb_anc, uwb_rel);
        cmlkf_cn.update(uwb_anc, uwb_rel);
        
        uwb_idx = uwb_idx + 1;
    end
    
    % --- C. 保存当前步估计结果 ---
    for i = 1:Vehicle_num
        est_p_cgn(k, :, i) = cmlkf_cgn.p(3*i-2 : 3*i)';
        est_p_cn(k, :, i)  = cmlkf_cn.p(3*i-2 : 3*i)';
    end
    
    % 进度打印
    if mod(k, 2000) == 0
        fprintf('已处理: %d / %d 步...\n', k, N_steps);
    end
end

%% 5. 计算 RMSE 并对比
rmse_p_cgn = zeros(Vehicle_num, 1);
rmse_p_cn  = zeros(Vehicle_num, 1);

for i = 1:Vehicle_num
    v_name = sprintf('V%d', i);
    true_p = [trajectories.(v_name).X_true(1:N_steps), ...
              trajectories.(v_name).Y_true(1:N_steps), ...
              trajectories.(v_name).Z_true(1:N_steps)];
              
    % 位置 RMSE
    rmse_p_cgn(i) = sqrt(mean(sum((est_p_cgn(:, :, i) - true_p).^2, 2)));
    rmse_p_cn(i)  = sqrt(mean(sum((est_p_cn(:, :, i)  - true_p).^2, 2)));
end

mean_rmse_cgn = mean(rmse_p_cgn);
mean_rmse_cn  = mean(rmse_p_cn);

% 打印对比结果
print_comparison(Vehicle_num, rmse_p_cgn, rmse_p_cn, mean_rmse_cgn, mean_rmse_cn);

%% ==== 局部打印对比函数 ====
function print_comparison(Vehicle_num, rmse_CGN, rmse_CN, m_CGN, m_CN)
    fprintf('\n======================== CMLKF 位置 RMSE (m) 对比 ========================\n');
    fprintf(' VehID  |  CGN (高斯-牛顿) |   CN (精确牛顿)  |   Diff (CGN - CN)\n');
    fprintf('--------------------------------------------------------------------------\n');
    for i = 1:Vehicle_num
        diff = rmse_CGN(i) - rmse_CN(i);
        fprintf('  %2d    |      %6.4f      |     %6.4f      |     %6.4f\n', i, rmse_CGN(i), rmse_CN(i), diff);
    end
    fprintf('--------------------------------------------------------------------------\n');
    diff_m = m_CGN - m_CN;
    fprintf('  Avg   |      %6.4f      |     %6.4f      |     %6.4f\n', m_CGN, m_CN, diff_m);
    fprintf('==========================================================================\n');
end