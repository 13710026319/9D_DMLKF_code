% DEKF_Test.m 
% DEKF (9D 分布式扩展卡尔曼滤波) - 测试与评价脚本 (Baseline对比)
clc; clear; close all;

%% 1. 测试参数与运行配置
Vehicle_num = 8;            
Anchor_num  = 4;             
run_flag    = 1;   % 0: 若存在结果则直接打印不运行; 1: 强制重新运行并覆盖

% 故意保留未知零偏以破坏先验 (1.0为完全补偿，0.7为补偿70%保留30%漂移)
bias_comp_ratio = 1; 

% 控制截取数据集的比例 (例如 0.8 表示只跑前 80% 的数据，1.0为全部)
data_ratio  = 0.2;

% 路径配置 (修改为 DEKF 的保存路径)
data_dir = 'E:\DMLKF_code\Data';
res_dir  = 'E:\DMLKF_code\EKF\DEKF\RESULT';
if ~exist(res_dir, 'dir')
    mkdir(res_dir);
end

data_file = fullfile(data_dir, sprintf('Trj_Veh%d_Anc%d_3D.mat', Vehicle_num, Anchor_num));
res_file  = fullfile(res_dir, sprintf('DEKF_Veh%d_Anc%d.mat', Vehicle_num, Anchor_num));

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
fprintf('数据集加载成功，开始运行 DEKF 算法...\n');

N_steps_total = length(trajectories.V1.Time_true);
N_steps = max(2, round(N_steps_total * data_ratio)); % 计算截断步数(至少保证有2步)
fprintf('>> 实验设定的数据截取比例: %.1f%%\n', data_ratio * 100);
fprintf('>> 实际运行步数 / 总步数: %d / %d\n', N_steps, N_steps_total);

dt_imu = trajectories.V1.Time_true(2) - trajectories.V1.Time_true(1);

%% 4. 生成静态拓扑掩码 (三档基站 + K-Regular 车间)
% A. 基站掩码 (30% 全基站, 50% 半基站, 20% 无基站)
N_tier1 = round(0.3 * Vehicle_num);
N_tier2 = round(0.5 * Vehicle_num);
Anchor_Mask = ones(Vehicle_num, Anchor_num);
for i = 1:Vehicle_num
    if i <= N_tier1
        % Tier 1: 保持全 1
    elseif i <= N_tier1 + N_tier2
        % Tier 2: 仅保留偶数基站
        for k = 1:Anchor_num
            if mod(k, 2) ~= 0, Anchor_Mask(i, k) = 0; end
        end
    else
        % Tier 3: 纯相对测距
        Anchor_Mask(i, :) = 0;
    end
end

% B. 车间掩码 (K=4 静态环形拓扑)
K_degree = min(4, Vehicle_num - 1); 
V2V_Mask = zeros(Vehicle_num, Vehicle_num);
for i = 1:Vehicle_num
    for d = 1:floor(K_degree/2)
        idx_forward = mod(i + d - 1, Vehicle_num) + 1;
        idx_backward = mod(i - d - 1, Vehicle_num) + 1;
        V2V_Mask(i, idx_forward) = 1;
        V2V_Mask(i, idx_backward) = 1;
    end
end
V2V_Mask(logical(eye(Vehicle_num))) = 0; % 自身对自身设0

%% 5. 初始化 DEKF 滤波器 (使用列向量锁定避免维度崩溃)
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

% 实例化 DEKF 基准算法
kf = DEKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);

% 结果存储空间
est_p = zeros(N_steps, 3, Vehicle_num);
est_v = zeros(N_steps, 3, Vehicle_num);
est_R = zeros(3, 3, Vehicle_num, N_steps);
for i = 1:Vehicle_num
    est_p(1, :, i) = p0(3*i-2 : 3*i)';
    est_v(1, :, i) = v0(3*i-2 : 3*i)';
    est_R(:, :, i, 1) = R0(:, :, i);
end

%% 6. 滤波主循环
uwb_idx = 2; 
UWB_Time_Vec = trajectories.V1.UWB_Anchor(:, 1);

fprintf('开始迭代仿真 (共 %d 步)...\n', N_steps);
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
        
        % 输入滤波器
        kf.update(uwb_anc_masked, uwb_rel_masked);
        uwb_idx = uwb_idx + 1;
    end
    
    % --- C. 提取当前步沙盒内的局部估计结果 ---
    for i = 1:Vehicle_num
        est_p(k, :, i) = kf.Nodes{i}.p';
        est_v(k, :, i) = kf.Nodes{i}.v';
        est_R(:, :, i, k) = kf.Nodes{i}.R;
    end
    
    if mod(k, 2000) == 0
        fprintf('已处理: %d / %d 步...\n', k, N_steps);
    end
end

%% 7. 计算 RMSE
rmse_p = zeros(Vehicle_num, 1);
rmse_v = zeros(Vehicle_num, 1);
rmse_att = zeros(Vehicle_num, 1);
for i = 1:Vehicle_num
    v_name = sprintf('V%d', i);
    
    % 截取前 N_steps 的轨迹真值，保证与 est_p 的维度一致
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

%% 8. 保存结果文件并打印
% 解除保存注释，以保证后续 run_flag = 0 正常工作
% save(res_file, 'est_p', 'est_v', 'est_R', 'rmse_p', 'rmse_v', 'rmse_att', ...
%                'mean_rmse_p', 'mean_rmse_v', 'mean_rmse_att', ...
%                'Anchor_Mask', 'V2V_Mask');
% fprintf('运行完成，结果已保存至: %s\n', res_file);

print_results(Vehicle_num, rmse_p, rmse_v, rmse_att, mean_rmse_p, mean_rmse_v, mean_rmse_att);

%% ==== 局部打印辅助函数 ====
function print_results(Vehicle_num, rmse_p, rmse_v, rmse_att, m_p, m_v, m_att)
    fprintf('\n================== DEKF RMSE 结果 ==================\n');
    fprintf(' VehID  |  Pos(m)  |  Vel(m/s)  |  Att(deg)\n');
    fprintf('----------------------------------------------------\n');
    for i = 1:Vehicle_num
        fprintf('  %2d    |  %6.4f  |   %6.4f   |  %6.4f\n', i, rmse_p(i), rmse_v(i), rmse_att(i));
    end
    fprintf('----------------------------------------------------\n');
    fprintf('  Avg   |  %6.4f  |   %6.4f   |  %6.4f\n', m_p, m_v, m_att);
    fprintf('====================================================\n\n');
end