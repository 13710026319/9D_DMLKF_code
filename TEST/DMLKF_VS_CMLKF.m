% DMLKF_VS_CMLKF.m
% 集中式 CMLKF 与 分布式 DMLKF 对比测试脚本
% 验证目的：在全基站观测下，DMLKF 随车间邻居数增加，逐渐逼近 CMLKF (理论最优下界)
clc; clear; close all;

%% 1. 实验参数与运行配置
Vehicle_num = 8;            
Anchor_num  = 4;             
K_test_list = 2:7; % 邻居数

% 保留一定比例的零偏误差以破坏 IMU 先验，凸显测距优化优势 (1.0为完全补偿, 0.7为保留30%漂移)
bias_comp_ratio = 1; 

% =========================================================================
% [数据集截取比例] (例如: 0.2 表示只取前 20% 的数据进行测试，1.0 为完整测试)
data_ratio  = 1;
% =========================================================================

% 路径配置
data_dir = 'E:\DMLKF_code\Data';
data_file = fullfile(data_dir, sprintf('Trj_Veh%d_Anc%d_3D.mat', Vehicle_num, Anchor_num));

%% 2. 加载数据集与截断处理
if ~exist(data_file, 'file')
    error('未找到数据集文件: %s\n请先运行生成脚本生成该数据！', data_file);
end
load(data_file, 'trajectories', 'anchors', 'IMU_noise_params', 'UWB_noise_params');
fprintf('数据集加载成功！开始对照实验...\n');

% 根据截取比例计算实际运行步数
N_steps_total = length(trajectories.V1.Time_true);
N_steps = max(2, round(N_steps_total * data_ratio));
dt_imu = trajectories.V1.Time_true(2) - trajectories.V1.Time_true(1);

fprintf('>> 实验设定数据截取比例: %.1f%%\n', data_ratio * 100);
fprintf('>> 实际运行步数 / 总步数: %d / %d 步\n\n', N_steps, N_steps_total);

%% 3. 初始化通用状态与真值提取
p0 = zeros(3 * Vehicle_num, 1);
v0 = zeros(3 * Vehicle_num, 1);
R0 = zeros(3, 3, Vehicle_num);

% 预分配真值矩阵 (长度严格限制为 N_steps 防止报错)
true_p_all = zeros(N_steps, 3, Vehicle_num);
true_v_all = zeros(N_steps, 3, Vehicle_num);

for i = 1:Vehicle_num
    v_name = sprintf('V%d', i);
    p0(3*i-2 : 3*i) = [trajectories.(v_name).X_true(1); trajectories.(v_name).Y_true(1); trajectories.(v_name).Z_true(1)];
    v0(3*i-2 : 3*i) = [trajectories.(v_name).Vx_true(1); trajectories.(v_name).Vy_true(1); trajectories.(v_name).Vz_true(1)];
    R0(:, :, i)     = trajectories.(v_name).R_true(:, :, 1); % 不加姿态扰动，使用真值
    
    % 仅截取前 N_steps 的轨迹真值
    true_p_all(:, :, i) = [trajectories.(v_name).X_true(1:N_steps), trajectories.(v_name).Y_true(1:N_steps), trajectories.(v_name).Z_true(1:N_steps)];
    true_v_all(:, :, i) = [trajectories.(v_name).Vx_true(1:N_steps), trajectories.(v_name).Vy_true(1:N_steps), trajectories.(v_name).Vz_true(1:N_steps)];
end

% 存储最终比对结果的容器 [Mean_Pos_RMSE, Mean_Vel_RMSE, Mean_Att_RMSE]
% 第一行存 CMLKF, 后续存不同 K 下的 DMLKF

results_summary = zeros(1 + length(K_test_list), 3); 

%% ========================================================================
%% 第一部分：运行集中式 CMLKF (绝对最优基准)
%% ========================================================================
fprintf('====================================================\n');
fprintf('[Baseline] 正在运行 集中式 CMLKF (全基站 + 全连通车间)...\n');
fprintf('====================================================\n');

kf_cmlkf = CMLKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);

% 预分配估计容器，长度限制为 N_steps
est_p_cmlkf = zeros(N_steps, 3, Vehicle_num);
est_v_cmlkf = zeros(N_steps, 3, Vehicle_num);
est_R_cmlkf = zeros(3, 3, Vehicle_num, N_steps);
for i=1:Vehicle_num
    est_p_cmlkf(1,:,i)=p0(3*i-2:3*i)'; 
    est_v_cmlkf(1,:,i)=v0(3*i-2:3*i)'; 
    est_R_cmlkf(:,:,i,1)=R0(:,:,i); 
end

uwb_idx = 2; 
UWB_Time_Vec = trajectories.V1.UWB_Anchor(:, 1);

for k = 2:N_steps
    acc_m = zeros(3, Vehicle_num); gyro_m = zeros(3, Vehicle_num);
    for i = 1:Vehicle_num
        v_name = sprintf('V%d', i);
        acc_m(:, i)  = trajectories.(v_name).IMU_acc_m(k-1, :)' - bias_comp_ratio * trajectories.(v_name).IMU_bias_a_true(k-1, :)';
        gyro_m(:, i) = trajectories.(v_name).IMU_gyro_m(k-1, :)' - bias_comp_ratio * trajectories.(v_name).IMU_bias_w_true(k-1, :)';
    end
    kf_cmlkf.predict(acc_m, gyro_m);
    
    curr_time = trajectories.V1.Time_true(k);
    if uwb_idx <= length(UWB_Time_Vec) && abs(curr_time - UWB_Time_Vec(uwb_idx)) < 1e-5
        uwb_anc_raw = zeros(Vehicle_num, Anchor_num); uwb_rel_raw = zeros(Vehicle_num, Vehicle_num);
        for i = 1:Vehicle_num
            v_name = sprintf('V%d', i);
            uwb_anc_raw(i, :) = trajectories.(v_name).UWB_Anchor(uwb_idx, 2:end);
            uwb_rel_raw(i, :) = trajectories.(v_name).UWB_Relative(uwb_idx, 2:end);
        end
        % CMLKF 使用全连接数据 (不做任何掩码截断)
        kf_cmlkf.update(uwb_anc_raw, uwb_rel_raw);
        uwb_idx = uwb_idx + 1;
    end
    for i=1:Vehicle_num
        est_p_cmlkf(k,:,i)=kf_cmlkf.p(3*i-2:3*i)'; 
        est_v_cmlkf(k,:,i)=kf_cmlkf.v(3*i-2:3*i)'; 
        est_R_cmlkf(:,:,i,k)=kf_cmlkf.R(:,:,i); 
    end
end

% 结算 CMLKF RMSE
[cmlkf_mean_p, cmlkf_mean_v, cmlkf_mean_att] = calc_mean_rmse(Vehicle_num, N_steps, est_p_cmlkf, est_v_cmlkf, est_R_cmlkf, true_p_all, true_v_all, trajectories);
results_summary(1, :) = [cmlkf_mean_p, cmlkf_mean_v, cmlkf_mean_att];
fprintf('-> CMLKF 运行完毕 | 平均位置RMSE: %.4f m | 速度: %.4f | 姿态: %.4f\n\n', cmlkf_mean_p, cmlkf_mean_v, cmlkf_mean_att);


%% ========================================================================
%% 第二部分：循环运行分布式 DMLKF (全基站 + 不同邻居数 K)
%% ========================================================================
Anchor_Mask_Full = ones(Vehicle_num, Anchor_num); % DMLKF 统一使用全基站

for test_idx = 1:length(K_test_list)
    K = K_test_list(test_idx);
    fprintf('====================================================\n');
    fprintf('[Experiment %d] 正在运行 DMLKF (邻居数 K = %d)...\n', test_idx, K);
    
    % 生成车间掩码 (奇数前大后小分配法)
    V2V_Mask = generate_v2v_mask(Vehicle_num, K);
    
    kf_dmlkf = DMLKF_V1(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);
    
    % 预分配估计容器，长度限制为 N_steps
    est_p_dmlkf = zeros(N_steps, 3, Vehicle_num);
    est_v_dmlkf = zeros(N_steps, 3, Vehicle_num);
    est_R_dmlkf = zeros(3, 3, Vehicle_num, N_steps);
    for i=1:Vehicle_num
        est_p_dmlkf(1,:,i)=p0(3*i-2:3*i)'; 
        est_v_dmlkf(1,:,i)=v0(3*i-2:3*i)'; 
        est_R_dmlkf(:,:,i,1)=R0(:,:,i); 
    end

    uwb_idx = 2; 
    for k = 2:N_steps
        acc_m = zeros(3, Vehicle_num); gyro_m = zeros(3, Vehicle_num);
        for i = 1:Vehicle_num
            v_name = sprintf('V%d', i);
            acc_m(:, i)  = trajectories.(v_name).IMU_acc_m(k-1, :)' - bias_comp_ratio * trajectories.(v_name).IMU_bias_a_true(k-1, :)';
            gyro_m(:, i) = trajectories.(v_name).IMU_gyro_m(k-1, :)' - bias_comp_ratio * trajectories.(v_name).IMU_bias_w_true(k-1, :)';
        end
        kf_dmlkf.predict(acc_m, gyro_m);
        
        curr_time = trajectories.V1.Time_true(k);
        if uwb_idx <= length(UWB_Time_Vec) && abs(curr_time - UWB_Time_Vec(uwb_idx)) < 1e-5
            uwb_anc_raw = zeros(Vehicle_num, Anchor_num); uwb_rel_raw = zeros(Vehicle_num, Vehicle_num);
            for i = 1:Vehicle_num
                v_name = sprintf('V%d', i);
                uwb_anc_raw(i, :) = trajectories.(v_name).UWB_Anchor(uwb_idx, 2:end);
                uwb_rel_raw(i, :) = trajectories.(v_name).UWB_Relative(uwb_idx, 2:end);
            end
            
            % DMLKF 应用稀疏化掩码
            uwb_anc_masked = uwb_anc_raw; uwb_anc_masked(Anchor_Mask_Full == 0) = NaN;
            uwb_rel_masked = uwb_rel_raw; uwb_rel_masked(V2V_Mask == 0) = NaN;
            
            kf_dmlkf.update(uwb_anc_masked, uwb_rel_masked);
            uwb_idx = uwb_idx + 1;
        end
        
        for i=1:Vehicle_num
            est_p_dmlkf(k,:,i)=kf_dmlkf.Nodes{i}.p'; 
            est_v_dmlkf(k,:,i)=kf_dmlkf.Nodes{i}.v'; 
            est_R_dmlkf(:,:,i,k)=kf_dmlkf.Nodes{i}.R; 
        end
    end
    
    % 结算当前 K 下的 RMSE
    [mean_p, mean_v, mean_att] = calc_mean_rmse(Vehicle_num, N_steps, est_p_dmlkf, est_v_dmlkf, est_R_dmlkf, true_p_all, true_v_all, trajectories);
    results_summary(1 + test_idx, :) = [mean_p, mean_v, mean_att];
    fprintf('-> DMLKF (K=%d) 运行完毕 | 平均位置RMSE: %.4f m | 速度: %.4f | 姿态: %.4f\n\n', K, mean_p, mean_v, mean_att);
end


%% ========================================================================
%% 第三部分：输出最终比对汇总表
%% ========================================================================
fprintf('\n');
fprintf('===================================================================\n');
fprintf('                     CMLKF vs DMLKF 最终对比总结表                 \n');
fprintf('===================================================================\n');
fprintf(' Method    | Neighbors (K) |  Pos RMSE (m) | Vel RMSE (m/s) | Att RMSE (deg)\n');
fprintf('-------------------------------------------------------------------\n');
fprintf(' CMLKF     | Full (%d)      |   %10.4f  |   %10.4f   |   %10.4f\n', (Vehicle_num - 1), results_summary(1,1), results_summary(1,2), results_summary(1,3));
fprintf('-------------------------------------------------------------------\n');
for idx = 1:length(K_test_list)
    fprintf(' DMLKF     | %-13d |   %10.4f  |   %10.4f   |   %10.4f\n', ...
        K_test_list(idx), results_summary(1+idx, 1), results_summary(1+idx, 2), results_summary(1+idx, 3));
end
fprintf('===================================================================\n\n');


%% ==== 局部辅助函数：生成奇数前大后小的 V2V Mask ====
function V2V_Mask = generate_v2v_mask(V_num, K)
    V2V_Mask = zeros(V_num, V_num);
    K_fwd = ceil(K / 2);  % 前面(序号变大方向)分大头
    K_bwd = floor(K / 2); % 后面(序号变小方向)分小头
    for i = 1:V_num
        for d = 1:K_fwd
            idx_fwd = mod(i + d - 1, V_num) + 1;
            V2V_Mask(i, idx_fwd) = 1;
        end
        for d = 1:K_bwd
            idx_bwd = mod(i - d - 1, V_num) + 1;
            V2V_Mask(i, idx_bwd) = 1;
        end
    end
    V2V_Mask(logical(eye(V_num))) = 0; % 自身置0
end

%% ==== 局部辅助函数：统一结算全局平均 RMSE ====
function [mean_p, mean_v, mean_att] = calc_mean_rmse(V_num, N_steps, est_p, est_v, est_R, true_p, true_v, trajectories)
    rmse_p = zeros(V_num, 1);
    rmse_v = zeros(V_num, 1);
    rmse_att = zeros(V_num, 1);
    
    for i = 1:V_num
        rmse_p(i) = sqrt(mean(sum((est_p(:, :, i) - true_p(:, :, i)).^2, 2)));
        rmse_v(i) = sqrt(mean(sum((est_v(:, :, i) - true_v(:, :, i)).^2, 2)));
        
        v_name = sprintf('V%d', i);
        err_att_seq = zeros(N_steps, 1);
        for k = 1:N_steps
            % 这里的真值提取直接使用索引 k 也是安全的，因为 k 被限制在 N_steps 内
            R_t = trajectories.(v_name).R_true(:, :, k);
            R_e = est_R(:, :, i, k);
            R_err = R_t' * R_e;
            tr = max(-1, min(3, trace(R_err)));
            err_att_seq(k) = acos((tr - 1) / 2) * (180 / pi);
        end
        rmse_att(i) = sqrt(mean(err_att_seq.^2));
    end
    mean_p = mean(rmse_p);
    mean_v = mean(rmse_v);
    mean_att = mean(rmse_att);
end