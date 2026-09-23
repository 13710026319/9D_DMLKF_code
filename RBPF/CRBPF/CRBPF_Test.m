% CRBPF_Test.m 
% CRBPF (集中式 Rao-Blackwellized 粒子滤波) - 测试与评价脚本
clc; clear; close all;

%% 1. 测试参数与运行配置
Vehicle_num = 5;            
Anchor_num  = 4;             
run_flag    = 1;   % 0: 若存在结果则直接打印不运行; 1: 强制重新运行并覆盖
save_flag = 0;

% 故意保留 30% 未知零偏以破坏先验，考验粒子群抗漂移能力 (1.0为完全补偿)
bias_comp_ratio = 0.5; 

% 数据集截取比例
data_ratio  = 0.3;

% 路径配置
data_dir = 'E:\DMLKF_code\Data';
res_dir  = 'E:\DMLKF_code\RBPF\CRBPF\RESULT';
if ~exist(res_dir, 'dir')
    mkdir(res_dir);
end

data_file = fullfile(data_dir, sprintf('Trj_Veh%d_Anc%d_3D_1.mat', Vehicle_num, Anchor_num));
res_file  = fullfile(res_dir, sprintf('CRBPF_Veh%d_Anc%d.mat', Vehicle_num, Anchor_num));

%% 2. 检查结果文件是否存在
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
fprintf('数据集加载成功，开始运行 CRBPF 算法...\n');

N_steps_total = length(trajectories.V1.Time_true);
N_steps = max(2, round(N_steps_total * data_ratio)); 
fprintf('>> 实验设定的数据截取比例: %.1f%%\n', data_ratio * 100);
fprintf('>> 实际运行步数 / 总步数: %d / %d\n', N_steps, N_steps_total);

dt_imu = trajectories.V1.Time_true(2) - trajectories.V1.Time_true(1);

%% 4. 初始化 CRBPF 滤波器 (使用完美真值初始化，不加偏航角扰动)
p0 = zeros(3 * Vehicle_num, 1);
v0 = zeros(3 * Vehicle_num, 1);
R0 = zeros(3, 3, Vehicle_num);

for i = 1:Vehicle_num
    v_name = sprintf('V%d', i);
    p_true_init = [trajectories.(v_name).X_true(1); trajectories.(v_name).Y_true(1); trajectories.(v_name).Z_true(1)];
    v_true_init = [trajectories.(v_name).Vx_true(1); trajectories.(v_name).Vy_true(1); trajectories.(v_name).Vz_true(1)];
    
    p0(3*i-2 : 3*i) = p_true_init(:);
    v0(3*i-2 : 3*i) = v_true_init(:);
    
    % 直接使用真值姿态
    R0(:, :, i) = trajectories.(v_name).R_true(:, :, 1); 
end

kf = CRBPF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);
kf.set_particle_count(1500)

% 结果存储空间
est_p = zeros(N_steps, 3, Vehicle_num);
est_v = zeros(N_steps, 3, Vehicle_num);
est_R = zeros(3, 3, Vehicle_num, N_steps);
for i = 1:Vehicle_num
    est_p(1, :, i) = p0(3*i-2 : 3*i)';
    est_v(1, :, i) = v0(3*i-2 : 3*i)';
    est_R(:, :, i, 1) = R0(:, :, i);
end

%% 5. 滤波主循环
uwb_idx = 2; 
UWB_Time_Vec = trajectories.V1.UWB_Anchor(:, 1);

fprintf('开始迭代仿真 (粒子数: %d, 共 %d 步)...\n', kf.Np, N_steps);
for k = 2:N_steps
    % --- A. 100Hz 预测过程 ---
    acc_m = zeros(3, Vehicle_num);
    gyro_m = zeros(3, Vehicle_num);
    for i = 1:Vehicle_num
        v_name = sprintf('V%d', i);
        % 注入指定比例的残留零偏 
        acc_m(:, i)  = trajectories.(v_name).IMU_acc_m(k-1, :)' - bias_comp_ratio * trajectories.(v_name).IMU_bias_a_true(k-1, :)';
        gyro_m(:, i) = trajectories.(v_name).IMU_gyro_m(k-1, :)' - bias_comp_ratio * trajectories.(v_name).IMU_bias_w_true(k-1, :)';
    end
    
    kf.predict(acc_m, gyro_m);
    
    % --- B. 10Hz 更新过程 (集中式：使用全连接数据) ---
    curr_time = trajectories.V1.Time_true(k);
    if uwb_idx <= length(UWB_Time_Vec) && abs(curr_time - UWB_Time_Vec(uwb_idx)) < 1e-5
        uwb_anc_raw = zeros(Vehicle_num, Anchor_num);
        uwb_rel_raw = zeros(Vehicle_num, Vehicle_num);
        for i = 1:Vehicle_num
            v_name = sprintf('V%d', i);
            uwb_anc_raw(i, :) = trajectories.(v_name).UWB_Anchor(uwb_idx, 2:end);
            uwb_rel_raw(i, :) = trajectories.(v_name).UWB_Relative(uwb_idx, 2:end);
        end
        
        % CRBPF 为集中式算法，直接传入完整的 Raw 数据进行全局权重更新
        kf.update(uwb_anc_raw, uwb_rel_raw);
        uwb_idx = uwb_idx + 1;
    end
    
    % --- C. 提取当前步 MMSE 估计结果 ---
    for i = 1:Vehicle_num
        est_p(k, :, i) = kf.p(3*i-2 : 3*i)';
        est_v(k, :, i) = kf.v(3*i-2 : 3*i)';
        est_R(:, :, i, k) = kf.R(:, :, i);
    end
    
    % 打印进度条
    if mod(k, max(1, round(N_steps/10))) == 0
        fprintf('已处理: %d / %d 步 (%.1f%%)...\n', k, N_steps, (k/N_steps)*100);
    end
end

%% 6. 计算 RMSE
rmse_p = zeros(Vehicle_num, 1);
rmse_v = zeros(Vehicle_num, 1);
rmse_att = zeros(Vehicle_num, 1);

for i = 1:Vehicle_num
    v_name = sprintf('V%d', i);
    
    % 截取真值
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

mean_rmse_p = mean(rmse_p);
mean_rmse_v = mean(rmse_v);
mean_rmse_att = mean(rmse_att);

%% 7. 保存结果文件并打印
if save_flag
    save(res_file, 'est_p', 'est_v', 'est_R', 'rmse_p', 'rmse_v', 'rmse_att', ...
                   'mean_rmse_p', 'mean_rmse_v', 'mean_rmse_att');
    fprintf('运行完成，结果已保存至: %s\n', res_file);
end
print_results(Vehicle_num, rmse_p, rmse_v, rmse_att, mean_rmse_p, mean_rmse_v, mean_rmse_att);

%% ==== 局部打印辅助函数 ====
function print_results(Vehicle_num, rmse_p, rmse_v, rmse_att, m_p, m_v, m_att)
    fprintf('\n================== CRBPF RMSE 结果 ==================\n');
    fprintf(' VehID  |  Pos(m)  |  Vel(m/s)  |  Att(deg)\n');
    fprintf('-----------------------------------------------------\n');
    for i = 1:Vehicle_num
        fprintf('  %2d    |  %6.4f  |   %6.4f   |  %6.4f\n', i, rmse_p(i), rmse_v(i), rmse_att(i));
    end
    fprintf('-----------------------------------------------------\n');
    fprintf('  Avg   |  %6.4f  |   %6.4f   |  %6.4f\n', m_p, m_v, m_att);
    fprintf('=====================================================\n\n');
end