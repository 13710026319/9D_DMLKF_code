% DMLKF_CL_Test.m
% 集中式 CMLKF 与 L级通信分布式 DMLKF_LC 对比测试脚本
% 验证目的：在固定拓扑下，随着逻辑通信等级 L 的提升，DMLKF_LC 完全等价于 CMLKF
% [测量掩码] CMLKF 的量测集完全由传入矩阵的 NaN 图样决定，因此本脚本按 K_degree
%            为 CMLKF 施加与 DMLKF_LC 完全相同的相对测距掩码（只改调用侧，
%            不修改 CMLKF 类本身）。这样两者吃同一张测量图，L 拉满后的等价性
%            才是有意义的对比；关掉掩码则退化为"观测上界"参考。
clc; clear; close all;

%% 1. 实验参数与运行配置
run_flag  = 1;   % 0: 若存在结果则直接加载绘制不运行; 1: 强制重新运行算法
save_flag = 1;   % 0: 不保存结果; 1: 保存 csv, mat 和 fig

Vehicle_num = 8;            
Anchor_num  = 4;             
K_degree    = 4; % 暂定1-hop物理邻居数为3

% 保留一定比例的零偏误差以破坏 IMU 先验，凸显测距优化优势
bias_comp_ratio = 1; 
data_ratio  = 0.2; % 数据集截取比例

% 路径配置
data_dir = 'E:\DMLKF_code\TEST\GN_compare\Data';
data_file = fullfile(data_dir, sprintf('Trj_Veh%d_Anc%d_pure.mat', Vehicle_num, Anchor_num));

res_dir = 'E:\DMLKF_code\TEST\DMLKF_CL_Test\RESULT';
if save_flag && ~exist(res_dir, 'dir')
    mkdir(res_dir);
end
res_mat_file = fullfile(res_dir, sprintf('Result_Veh%d_K%d.mat', Vehicle_num, K_degree));
res_csv_file = fullfile(res_dir, sprintf('Result_Veh%d_K%d.csv', Vehicle_num, K_degree));

%% 2. 拓扑生成与最大通信等级 L_max 计算
% 生成绝对对称的固定 V2V_Mask
V2V_Mask = generate_symmetric_mask(Vehicle_num, K_degree);

% =========================================================================
% [L_max 计算公式与说明]
% 理论上，L_max 是网络的“图直径 (Graph Diameter)”。
% 对于总节点数 N，1-hop规则度数为 K 的对称循环图：
% 1. 若 K 为偶数 (仅前后相连)，L_max = ceil((N-1) / K) (近似)
% 2. 若 K 为奇数 (额外连接对径点)，L_max 显著降低。
% 在代码实现中，直接使用邻接矩阵 (Adj) 的幂运算求连通度：
% 若 (Adj + I)^L 中没有 0 元素，说明 L 跳可达全网，此时的 L 即为 L_max。
% =========================================================================
Adj = V2V_Mask;
L_max = 1;          %固定为1，下面Reach自动计算最大需要的L
Reach = Adj + eye(Vehicle_num);
while ~all(Reach(:))
    L_max = L_max + 1;
    Reach = Reach * (Adj + eye(Vehicle_num));
end
fprintf('>> 拓扑分析: 车辆数 N=%d, 物理邻居数 K=%d\n', Vehicle_num, K_degree);
fprintf('>> 全网覆盖最大通信等级 L_max = %d\n\n', L_max);

%% 2.5 统一测量掩码（CMLKF 与 DMLKF_LC 必须同图，否则对比不公平）
% 基站绝对测距：两者都不掩码，保留全部 Anchor_num 个基站
Anchor_Mask_Full = ones(Vehicle_num, Anchor_num);
% 车间相对测距：只保留由 K_degree 决定的物理边，与 DMLKF_LC 用的 V2V_Mask 完全一致
Rel_Mask = V2V_Mask;

% [开关] 是否给 CMLKF 施加与 DMLKF_LC 完全相同的相对测距掩码
%   1 = 施加：CMLKF 与 DMLKF_LC 吃同一张测量图，比的是"分布式融合 vs 集中式融合"，
%             L 拉满 (L = L_max) 后两者应收敛到同一个集中式解；
%   0 = 不施加：CMLKF 吃全量相对测距（观测更多），只能当观测上界参考，不是同题对照。
cmlkf_rel_mask_flag = 1;

L_test_list = 1:L_max; % 测试的 L 列表

% L_test_list = L_max; % 测试的 L 列表

%% 3. 运行算法 (受 run_flag 控制)
if run_flag || ~exist(res_mat_file, 'file')
    % --- 加载数据集 ---
    if ~exist(data_file, 'file')
        error('未找到数据集文件: %s', data_file);
    end
    load(data_file, 'trajectories', 'anchors', 'IMU_noise_params', 'UWB_noise_params');
    
    N_steps_total = length(trajectories.V1.Time_true);
    N_steps = max(2, round(N_steps_total * data_ratio));
    dt_imu = trajectories.V1.Time_true(2) - trajectories.V1.Time_true(1);
    
    p0 = zeros(3 * Vehicle_num, 1);
    v0 = zeros(3 * Vehicle_num, 1);
    R0 = zeros(3, 3, Vehicle_num);
    true_p_all = zeros(N_steps, 3, Vehicle_num);
    true_v_all = zeros(N_steps, 3, Vehicle_num);
    
    for i = 1:Vehicle_num
        v_name = sprintf('V%d', i);
        p0(3*i-2 : 3*i) = [trajectories.(v_name).X_true(1); trajectories.(v_name).Y_true(1); trajectories.(v_name).Z_true(1)];
        v0(3*i-2 : 3*i) = [trajectories.(v_name).Vx_true(1); trajectories.(v_name).Vy_true(1); trajectories.(v_name).Vz_true(1)];
        R0(:, :, i)     = trajectories.(v_name).R_true(:, :, 1); 
        true_p_all(:, :, i) = [trajectories.(v_name).X_true(1:N_steps), trajectories.(v_name).Y_true(1:N_steps), trajectories.(v_name).Z_true(1:N_steps)];
        true_v_all(:, :, i) = [trajectories.(v_name).Vx_true(1:N_steps), trajectories.(v_name).Vy_true(1:N_steps), trajectories.(v_name).Vz_true(1:N_steps)];
    end
    
    % 结果容器: 行数 = 1(CMLKF) + length(L_test_list)
    results_summary = zeros(1 + length(L_test_list), 3); 
    
    %% ====================================================================
    %% 3.1 运行集中式 CMLKF (绝对最优基准)
    %% ====================================================================
    fprintf('====================================================\n');
    fprintf('[Baseline] 正在运行 集中式 CMLKF ...\n');
    fprintf('====================================================\n');
    
    kf_cmlkf = CMLKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);
    kf_cmlkf.beta_inv = 0.01;
    kf_cmlkf.max_iter = 30;
    
    est_p_cmlkf = zeros(N_steps, 3, Vehicle_num);
    est_v_cmlkf = zeros(N_steps, 3, Vehicle_num);
    est_R_cmlkf = zeros(3, 3, Vehicle_num, N_steps);
    for i=1:Vehicle_num
        est_p_cmlkf(1,:,i)=p0(3*i-2:3*i)'; est_v_cmlkf(1,:,i)=v0(3*i-2:3*i)'; est_R_cmlkf(:,:,i,1)=R0(:,:,i); 
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
            % --- 与 DMLKF_LC 完全相同的测量掩码（只改调用侧，不动 CMLKF 类本身）---
            uwb_anc_cmlkf = uwb_anc_raw;
            uwb_anc_cmlkf(Anchor_Mask_Full == 0) = NaN;
            uwb_rel_cmlkf = uwb_rel_raw;
            if cmlkf_rel_mask_flag
                uwb_rel_cmlkf(Rel_Mask == 0) = NaN;
            end
            kf_cmlkf.update(uwb_anc_cmlkf, uwb_rel_cmlkf);
            uwb_idx = uwb_idx + 1;
        end
        for i=1:Vehicle_num
            est_p_cmlkf(k,:,i)=kf_cmlkf.p(3*i-2:3*i)'; 
            est_v_cmlkf(k,:,i)=kf_cmlkf.v(3*i-2:3*i)'; 
            est_R_cmlkf(:,:,i,k)=kf_cmlkf.R(:,:,i); 
        end
    end
    
    [cmlkf_mean_p, cmlkf_mean_v, cmlkf_mean_att] = calc_mean_rmse(Vehicle_num, N_steps, est_p_cmlkf, est_v_cmlkf, est_R_cmlkf, true_p_all, true_v_all, trajectories);
    results_summary(1, :) = [cmlkf_mean_p, cmlkf_mean_v, cmlkf_mean_att];
    fprintf('-> CMLKF 运行完毕 | 位置RMSE: %.4f m | 速度: %.4f | 姿态: %.4f\n\n', cmlkf_mean_p, cmlkf_mean_v, cmlkf_mean_att);
    
    %% ====================================================================
    %% 3.2 循环运行 DMLKF_LC (固定K，递增L)
    %% ====================================================================
    % (Anchor_Mask_Full / Rel_Mask 已在第 2.5 节统一定义，CMLKF 与 DMLKF_LC 共用同一张掩码)
    
    for test_idx = 1:length(L_test_list)
        L = L_test_list(test_idx);
        fprintf('====================================================\n');
        fprintf('[Experiment %d] 正在运行 DMLKF_LC (通信等级 L = %d)...\n', test_idx, L);
        
        kf_dmlkf = DMLKF_LC(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0, V2V_Mask, 200, L);
        kf_dmlkf.beta_inv = 0.01;
        kf_dmlkf.epsilon  = 1e-8;    % 关键；不设就是类默认的 1e-4
        kf_dmlkf.max_step = 1;       % 可留 1 或 Inf，无影响
        
        est_p_dmlkf = zeros(N_steps, 3, Vehicle_num);
        est_v_dmlkf = zeros(N_steps, 3, Vehicle_num);
        est_R_dmlkf = zeros(3, 3, Vehicle_num, N_steps);
        for i=1:Vehicle_num
            est_p_dmlkf(1,:,i)=p0(3*i-2:3*i)'; est_v_dmlkf(1,:,i)=v0(3*i-2:3*i)'; est_R_dmlkf(:,:,i,1)=R0(:,:,i); 
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
                
                uwb_anc_masked = uwb_anc_raw; uwb_anc_masked(Anchor_Mask_Full == 0) = NaN;
                uwb_rel_masked = uwb_rel_raw; uwb_rel_masked(Rel_Mask == 0) = NaN;
                
                kf_dmlkf.update(uwb_anc_masked, uwb_rel_masked);
                uwb_idx = uwb_idx + 1;
            end
            
            for i=1:Vehicle_num
                est_p_dmlkf(k,:,i)=kf_dmlkf.Nodes{i}.p'; 
                est_v_dmlkf(k,:,i)=kf_dmlkf.Nodes{i}.v'; 
                est_R_dmlkf(:,:,i,k)=kf_dmlkf.Nodes{i}.R; 
            end
        end
        
        [mean_p, mean_v, mean_att] = calc_mean_rmse(Vehicle_num, N_steps, est_p_dmlkf, est_v_dmlkf, est_R_dmlkf, true_p_all, true_v_all, trajectories);
        results_summary(1 + test_idx, :) = [mean_p, mean_v, mean_att];
        fprintf('-> DMLKF_LC (L=%d) 运行完毕 | 位置RMSE: %.4f m | 速度: %.4f | 姿态: %.4f\n\n', L, mean_p, mean_v, mean_att);
    end
    
    if save_flag
        save(res_mat_file, 'results_summary', 'L_test_list', 'L_max', 'Vehicle_num', 'K_degree', ...
             'cmlkf_rel_mask_flag');
        % 导出 CSV
        T = table();
        if cmlkf_rel_mask_flag
            cmlkf_label = 'CMLKF(mask)';
        else
            cmlkf_label = 'CMLKF';
        end
        T.Method = [cellstr(cmlkf_label); repmat({'DMLKF_LC'}, length(L_test_list), 1)];
        T.L_Level = [NaN; L_test_list'];
        T.Pos_RMSE = results_summary(:, 1);
        T.Vel_RMSE = results_summary(:, 2);
        T.Att_RMSE = results_summary(:, 3);
        writetable(T, res_csv_file);
        fprintf('>> 数据已保存至: %s\n', res_dir);
    end
else
    fprintf('>> run_flag=0，加载已有结果: %s\n', res_mat_file);
    load(res_mat_file, 'results_summary', 'L_test_list', 'L_max');
end

%% 4. 终端总结输出
fprintf('\n');
fprintf('===================================================================\n');
fprintf('                CMLKF vs DMLKF_LC 最终对比总结表                 \n');
fprintf('===================================================================\n');
if cmlkf_rel_mask_flag
    fprintf(' CMLKF 测量集   : 与 DMLKF_LC 相同的 K=%d 相对测距掩码 (同图对比)\n', K_degree);
else
    fprintf(' CMLKF 测量集   : 全量相对测距 (观测更多, 非同图对比)\n');
end
fprintf(' Method    | Comm Level (L) |  Pos RMSE (m) | Vel RMSE (m/s) | Att RMSE (deg)\n');
fprintf('-------------------------------------------------------------------\n');
if cmlkf_rel_mask_flag
    cmlkf_row = 'CMLKF(mask)';
else
    cmlkf_row = 'CMLKF';
end
fprintf(' %-11s| Full (Central) |   %10.4f  |   %10.4f   |   %10.4f\n', ...
    cmlkf_row, results_summary(1,1), results_summary(1,2), results_summary(1,3));
fprintf('-------------------------------------------------------------------\n');
for idx = 1:length(L_test_list)
    fprintf(' DMLKF_LC  | %-14d |   %10.4f  |   %10.4f   |   %10.4f\n', ...
        L_test_list(idx), results_summary(1+idx, 1), results_summary(1+idx, 2), results_summary(1+idx, 3));
end
fprintf('===================================================================\n\n');

%% 5. 绘制 RMSE vs L 趋势图
figure('Name', 'Position RMSE vs Communication Level L', 'Color', 'w', 'Position', [200, 200, 700, 500]);
hold on; grid on;

% 绘制 DMLKF_LC 的曲线
plot(L_test_list, results_summary(2:end, 1), '-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', '#0072BD', 'Color', '#0072BD');

% 绘制 CMLKF 的基准线
yline(results_summary(1, 1), '--r', 'LineWidth', 2, 'LabelHorizontalAlignment', 'left');

xlabel('Communication Level (L-hop)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Mean Position RMSE (m)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('DMLKF\\_LC vs CMLKF (N=%d, K=%d)', Vehicle_num, K_degree), 'FontSize', 14);

% 强制 x 轴刻度为整数
xticks(L_test_list);
xlim([0.8, max(L_test_list)+0.2]);

legend({'DMLKF\_LC', 'CMLKF (Lower Bound)'}, 'FontSize', 12, 'Location', 'northeast');
set(gca, 'FontSize', 11, 'GridAlpha', 0.3);

if save_flag
    fig_file = fullfile(res_dir, sprintf('LC_Veh%d_K%d.png', Vehicle_num, K_degree));
    % 使用 exportgraphics 替代 saveas，并设置 300 分辨率
    exportgraphics(gcf, fig_file, 'Resolution', 300);
    fprintf('>> 图表已保存至: %s\n', fig_file);
end

%% ==== 局部辅助函数：生成奇数/偶数皆对称的 V2V Mask ====
function V2V_Mask = generate_symmetric_mask(V_num, K)
    V2V_Mask = zeros(V_num, V_num);
    K_half = floor(K / 2); 
    
    for i = 1:V_num
        % 1. 绝对对称地连接前后各 K_half 个节点
        for d = 1:K_half
            idx_forward = mod(i + d - 1, V_num) + 1;
            idx_backward = mod(i - d - 1, V_num) + 1;
            V2V_Mask(i, idx_forward) = 1;
            V2V_Mask(i, idx_backward) = 1;
        end
        
        % 2. 处理 K 为奇数的情况：连接圆环正对面的节点
        if mod(K, 2) ~= 0
            if mod(V_num, 2) ~= 0
                error('图论限制: 当每辆车的邻居数 K 为奇数时，车辆总数必须为偶数！');
            end
            idx_opposite = mod(i + V_num/2 - 1, V_num) + 1;
            V2V_Mask(i, idx_opposite) = 1;
        end
    end
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
