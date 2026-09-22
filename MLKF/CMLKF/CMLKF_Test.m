% CMLKF_Test.m 
% CMLKF集中式最大似然卡尔曼滤波算法 - 测试与评价脚本

clc; clear; close all;

%% 1. 测试参数配置
Vehicle_num = 8;            
Anchor_num = 4;             
run_flag = 1;   % 0: 若存在结果则直接打印不运行; 1: 强制重新运行并覆盖

% 路径配置
data_dir = 'E:\DMLKF_code\Data';
res_dir  = 'E:\DMLKF_code\MLKF\CMLKF\RESULT';
if ~exist(res_dir, 'dir')
    mkdir(res_dir);
end

data_file = fullfile(data_dir, sprintf('Trj_Veh%d_Anc%d_3D.mat', Vehicle_num, Anchor_num));
res_file  = fullfile(res_dir, sprintf('CMLKF_Veh%d_Anc%d.mat', Vehicle_num, Anchor_num));

%% 2. 检查结果文件是否存在 (run_flag 机制)
if run_flag == 0 && exist(res_file, 'file')
    fprintf('检测到结果文件已存在，直接加载并打印结果 (若需重跑请置 run_flag=1)...\n');
    load(res_file, 'rmse_p', 'rmse_v', 'rmse_att', 'mean_rmse_p', 'mean_rmse_v', 'mean_rmse_att');
    print_results(Vehicle_num, rmse_p, rmse_v, rmse_att, mean_rmse_p, mean_rmse_v, mean_rmse_att);
    return;
end

%% 3. 加载数据集
if ~exist(data_file, 'file')
    error('未找到数据集文件: %s\n请先运行生成脚本生成该数据！', data_file);
end
load(data_file, 'trajectories', 'anchors', 'IMU_noise_params', 'UWB_noise_params');
fprintf('数据集加载成功，开始运行 CMLKF 算法...\n');

N_steps = length(trajectories.V1.Time_true);
dt_imu = trajectories.V1.Time_true(2) - trajectories.V1.Time_true(1);

%% 4. 初始化 CMLKF 滤波器
p0 = zeros(3 * Vehicle_num, 1);
v0 = zeros(3 * Vehicle_num, 1);
R0 = zeros(3, 3, Vehicle_num);

for i = 1:Vehicle_num
    v_name = sprintf('V%d', i);
    p0(3*i-2 : 3*i) = [trajectories.(v_name).X_true(1); trajectories.(v_name).Y_true(1); trajectories.(v_name).Z_true(1)];
    v0(3*i-2 : 3*i) = [trajectories.(v_name).Vx_true(1); trajectories.(v_name).Vy_true(1); trajectories.(v_name).Vz_true(1)];
    R0(:, :, i)     = trajectories.(v_name).R_true(:, :, 1);
end

kf = CMLKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);

% 预分配估计结果存储空间
est_p = zeros(N_steps, 3, Vehicle_num);
est_v = zeros(N_steps, 3, Vehicle_num);
est_R = zeros(3, 3, Vehicle_num, N_steps);
for i = 1:Vehicle_num
    est_p(1, :, i) = p0(3*i-2 : 3*i)';
    est_v(1, :, i) = v0(3*i-2 : 3*i)';
    est_R(:, :, i, 1) = R0(:, :, i);
end

%% 5. 滤波主循环
uwb_idx = 2; % UWB 从第2个历元(idx_uwb)开始融合，第1个是初始时刻
UWB_Time_Vec = trajectories.V1.UWB_Anchor(:, 1);

for k = 2:N_steps
    % --- A. 100Hz 预测过程 (扣除偏置) ---
    acc_m = zeros(3, Vehicle_num);
    gyro_m = zeros(3, Vehicle_num);
    for i = 1:Vehicle_num
        v_name = sprintf('V%d', i);
        % 在此处模拟IMU扰动以及一定的偏置累加影响 运行时间越长差距越大
        acc_m(:, i)  = trajectories.(v_name).IMU_acc_m(k-1, :)' - 0.5*trajectories.(v_name).IMU_bias_a_true(k-1, :)';
        gyro_m(:, i) = trajectories.(v_name).IMU_gyro_m(k-1, :)' - 0.5*trajectories.(v_name).IMU_bias_w_true(k-1, :)';
    end
    kf.predict(acc_m, gyro_m);
    
    % --- B. 10Hz 更新过程 (异步融合) ---
    curr_time = trajectories.V1.Time_true(k);
    if uwb_idx <= length(UWB_Time_Vec) && abs(curr_time - UWB_Time_Vec(uwb_idx)) < 1e-5
        uwb_anc = zeros(Vehicle_num, Anchor_num);
        uwb_rel = zeros(Vehicle_num, Vehicle_num);
        for i = 1:Vehicle_num
            v_name = sprintf('V%d', i);
            uwb_anc(i, :) = trajectories.(v_name).UWB_Anchor(uwb_idx, 2:end);
            uwb_rel(i, :) = trajectories.(v_name).UWB_Relative(uwb_idx, 2:end);
        end
        kf.update(uwb_anc, uwb_rel);
        uwb_idx = uwb_idx + 1;
    end
    
    % --- C. 保存当前步估计结果 ---
    for i = 1:Vehicle_num
        est_p(k, :, i) = kf.p(3*i-2 : 3*i)';
        est_v(k, :, i) = kf.v(3*i-2 : 3*i)';
        est_R(:, :, i, k) = kf.R(:, :, i);
    end
end

%% 6. 计算 RMSE
rmse_p = zeros(Vehicle_num, 1);
rmse_v = zeros(Vehicle_num, 1);
rmse_att = zeros(Vehicle_num, 1);

for i = 1:Vehicle_num
    v_name = sprintf('V%d', i);
    % 取真值
    true_p = [trajectories.(v_name).X_true, trajectories.(v_name).Y_true, trajectories.(v_name).Z_true];
    true_v = [trajectories.(v_name).Vx_true, trajectories.(v_name).Vy_true, trajectories.(v_name).Vz_true];
    
    % 位置和速度 RMSE
    rmse_p(i) = sqrt(mean(sum((est_p(:, :, i) - true_p).^2, 2)));
    rmse_v(i) = sqrt(mean(sum((est_v(:, :, i) - true_v).^2, 2)));
    
    % 姿态 RMSE (角度误差)
    err_att_seq = zeros(N_steps, 1);
    for k = 1:N_steps
        R_t = trajectories.(v_name).R_true(:, :, k);
        R_e = est_R(:, :, i, k);
        R_err = R_t' * R_e;
        tr = max(-1, min(3, trace(R_err))); % 防溢出限幅
        err_att_seq(k) = acos((tr - 1) / 2) * (180 / pi);
    end
    rmse_att(i) = sqrt(mean(err_att_seq.^2));
end

mean_rmse_p = mean(rmse_p);
mean_rmse_v = mean(rmse_v);
mean_rmse_att = mean(rmse_att);

%% 7. 保存结果文件并打印
save(res_file, 'est_p', 'est_v', 'est_R', 'rmse_p', 'rmse_v', 'rmse_att', ...
               'mean_rmse_p', 'mean_rmse_v', 'mean_rmse_att');
fprintf('运行完成，结果已保存至: %s\n', res_file);

print_results(Vehicle_num, rmse_p, rmse_v, rmse_att, mean_rmse_p, mean_rmse_v, mean_rmse_att);


%% ==== 局部打印辅助函数 ====
function print_results(Vehicle_num, rmse_p, rmse_v, rmse_att, m_p, m_v, m_att)
    fprintf('\n================== CMLKF RMSE 结果 ==================\n');
    fprintf(' VehID  |  Pos(m)  |  Vel(m/s)  |  Att(deg)\n');
    fprintf('-----------------------------------------------------\n');
    for i = 1:Vehicle_num
        fprintf('  %2d    |  %6.4f  |   %6.4f   |  %6.4f\n', i, rmse_p(i), rmse_v(i), rmse_att(i));
    end
    fprintf('-----------------------------------------------------\n');
    fprintf('  Avg   |  %6.4f  |   %6.4f   |  %6.4f\n', m_p, m_v, m_att);
    fprintf('=====================================================\n\n');
end