% =========================================================================
% 脚本名称: DMLKF_L_Test.m
% 功能描述: 集中式 CMLKF 与 L级通信分布式 DMLKF_L 对比测试脚本
% 验证目的: 在固定拓扑下，随着逻辑通信等级 L 的提升，DMLKF_L 完全等价于 CMLKF
% [测量掩码] 为 CMLKF 施加与 DMLKF_L 完全相同的相对测距掩码（只改调用侧，
%            不修改 CMLKF 类本身），保证两者在同一张测量图下进行公平对比。
% =========================================================================
clc; clear; close all;

%% 1. 实验参数与运行配置
Vehicle_num = 12;            
Anchor_num  = 5;             
K_degree    = 4;        % 1-hop 物理邻居数

% 保留一定比例的零偏误差以破坏 IMU 先验，凸显测距优化优势
bias_comp_ratio = 1; 
data_ratio      = 0.2;  % 数据集截取比例

% 运行与保存控制标志
save_flag = 0;          % 1: 保存汇总表(CSV)与对比图(PNG); 0: 不保存
run_flag  = 1;          % 1: 运行实验; 0: 仅读取已有结果并打印、绘图

% 路径配置（锁定在当前脚本所在的 DMLKF_L_Test 文件夹下）
test_dir    = fileparts(mfilename('fullpath'));
data_dir    = fullfile(test_dir, 'Data');
result_dir  = fullfile(test_dir, 'Result');

data_file   = fullfile(data_dir, sprintf( ...
    'Trj_Veh%d_Anc%d_pure.mat', Vehicle_num, Anchor_num));
result_file = fullfile(result_dir, sprintf( ...
    'DMLKF_L_Veh%d_Anc%d_K%d.csv', Vehicle_num, Anchor_num, K_degree));
plot_file   = fullfile(result_dir, sprintf( ...
    'DMLKF_L_Veh%d_Anc%d_K%d.png', Vehicle_num, Anchor_num, K_degree));

% 仅读取模式处理 (run_flag == 0)
if run_flag == 0
    if exist(result_file, 'file')
        opts = detectImportOptions(result_file);
        opts.VariableTypes{1} = 'string';
        summary_table = readtable(result_file, opts);
        print_summary_table(summary_table, K_degree);
        plot_summary_results(summary_table, plot_file, save_flag);
    else
        fprintf('当前配置的结果文件不存在，请先将 run_flag 设为 1 并运行实验得到结果。\n');
    end
    return;
end


%% 2. 拓扑生成与最大通信等级 L_max 计算
% 生成绝对对称的固定 V2V_Mask
V2V_Mask = generate_symmetric_mask(Vehicle_num, K_degree);

% =========================================================================
% [L_max 计算公式与说明]
% 理论上，L_max 是网络的“图直径 (Graph Diameter)”。
% 在代码实现中，直接使用邻接矩阵 (Adj) 的幂运算求连通度：
% 若 (Adj + I)^L 中没有 0 元素，说明 L 跳可达全网，此时的 L 即为 L_max。
% =========================================================================
Adj   = V2V_Mask;
L_max = 1;
Reach = Adj + eye(Vehicle_num);
while ~all(Reach(:))
    L_max = L_max + 1;
    Reach = Reach * (Adj + eye(Vehicle_num));
end
fprintf('>> 拓扑分析: 车辆数 N=%d, 物理邻居数 K=%d\n', Vehicle_num, K_degree);
fprintf('>> 全网覆盖最大通信等级 L_max = %d, save_flag = %d\n\n', L_max, save_flag);

% 统一测量掩码（CMLKF 与 DMLKF_L 共用同一张测量图，保证公平对比）
Anchor_Mask_Full = ones(Vehicle_num, Anchor_num);
Rel_Mask         = V2V_Mask;

L_test_list      = 1:L_max; % 测试的 L 列表


%% 3. 加载数据集与初始化状态
if ~exist(data_file, 'file')
    error('未找到数据集文件: %s\n请先运行生成脚本生成该数据！', data_file);
end
load(data_file, 'trajectories', 'anchors', 'IMU_noise_params', 'UWB_noise_params');

N_steps_total = length(trajectories.V1.Time_true);
N_steps       = max(2, round(N_steps_total * data_ratio));
dt_imu        = trajectories.V1.Time_true(2) - trajectories.V1.Time_true(1);

p0         = zeros(3 * Vehicle_num, 1);
v0         = zeros(3 * Vehicle_num, 1);
R0         = zeros(3, 3, Vehicle_num);
true_p_all = zeros(N_steps, 3, Vehicle_num);
true_v_all = zeros(N_steps, 3, Vehicle_num);

for i = 1:Vehicle_num
    v_name              = sprintf('V%d', i);
    p0(3*i-2 : 3*i)     = [trajectories.(v_name).X_true(1); trajectories.(v_name).Y_true(1); trajectories.(v_name).Z_true(1)];
    v0(3*i-2 : 3*i)     = [trajectories.(v_name).Vx_true(1); trajectories.(v_name).Vy_true(1); trajectories.(v_name).Vz_true(1)];
    R0(:, :, i)         = trajectories.(v_name).R_true(:, :, 1); 
    true_p_all(:, :, i) = [trajectories.(v_name).X_true(1:N_steps), trajectories.(v_name).Y_true(1:N_steps), trajectories.(v_name).Z_true(1:N_steps)];
    true_v_all(:, :, i) = [trajectories.(v_name).Vx_true(1:N_steps), trajectories.(v_name).Vy_true(1:N_steps), trajectories.(v_name).Vz_true(1:N_steps)];
end

% 结果容器: 行数 = 1(CMLKF) + length(L_test_list)
results_summary = zeros(1 + length(L_test_list), 3); 


%% ========================================================================
%% 4.1 运行集中式 CMLKF (同图测量最优基准)
%% ========================================================================
fprintf('====================================================\n');
fprintf('[Baseline] 正在运行 集中式 CMLKF ...\n');
fprintf('====================================================\n');

kf_cmlkf          = CMLKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0);
kf_cmlkf.beta_inv = 0.01;
kf_cmlkf.max_iter = 30;

est_p_cmlkf = zeros(N_steps, 3, Vehicle_num);
est_v_cmlkf = zeros(N_steps, 3, Vehicle_num);
est_R_cmlkf = zeros(3, 3, Vehicle_num, N_steps);
for i = 1:Vehicle_num
    est_p_cmlkf(1, :, i)    = p0(3*i-2:3*i)';
    est_v_cmlkf(1, :, i)    = v0(3*i-2:3*i)';
    est_R_cmlkf(:, :, i, 1) = R0(:, :, i); 
end

uwb_idx      = 2; 
UWB_Time_Vec = trajectories.V1.UWB_Anchor(:, 1);

for k = 2:N_steps
    acc_m  = zeros(3, Vehicle_num);
    gyro_m = zeros(3, Vehicle_num);
    for i = 1:Vehicle_num
        v_name       = sprintf('V%d', i);
        acc_m(:, i)  = trajectories.(v_name).IMU_acc_m(k-1, :)' - bias_comp_ratio * trajectories.(v_name).IMU_bias_a_true(k-1, :)';
        gyro_m(:, i) = trajectories.(v_name).IMU_gyro_m(k-1, :)' - bias_comp_ratio * trajectories.(v_name).IMU_bias_w_true(k-1, :)';
    end
    kf_cmlkf.predict(acc_m, gyro_m);
    
    curr_time = trajectories.V1.Time_true(k);
    if uwb_idx <= length(UWB_Time_Vec) && abs(curr_time - UWB_Time_Vec(uwb_idx)) < 1e-5
        uwb_anc_raw = zeros(Vehicle_num, Anchor_num);
        uwb_rel_raw = zeros(Vehicle_num, Vehicle_num);
        for i = 1:Vehicle_num
            v_name            = sprintf('V%d', i);
            uwb_anc_raw(i, :) = trajectories.(v_name).UWB_Anchor(uwb_idx, 2:end);
            uwb_rel_raw(i, :) = trajectories.(v_name).UWB_Relative(uwb_idx, 2:end);
        end
        
        % 施加与 DMLKF_L 完全相同的测量掩码，保证公平同图对比
        uwb_anc_cmlkf = uwb_anc_raw;
        uwb_anc_cmlkf(Anchor_Mask_Full == 0) = NaN;
        uwb_rel_cmlkf = uwb_rel_raw;
        uwb_rel_cmlkf(Rel_Mask == 0) = NaN;
        
        kf_cmlkf.update(uwb_anc_cmlkf, uwb_rel_cmlkf);
        uwb_idx = uwb_idx + 1;
    end
    for i = 1:Vehicle_num
        est_p_cmlkf(k, :, i)    = kf_cmlkf.p(3*i-2:3*i)'; 
        est_v_cmlkf(k, :, i)    = kf_cmlkf.v(3*i-2:3*i)'; 
        est_R_cmlkf(:, :, i, k) = kf_cmlkf.R(:, :, i); 
    end
end

[cmlkf_mean_p, cmlkf_mean_v, cmlkf_mean_att] = calc_mean_rmse( ...
    Vehicle_num, N_steps, est_p_cmlkf, est_v_cmlkf, est_R_cmlkf, true_p_all, true_v_all, trajectories);
results_summary(1, :) = [cmlkf_mean_p, cmlkf_mean_v, cmlkf_mean_att];
fprintf('-> CMLKF 运行完毕 | 位置RMSE: %.4f m | 速度: %.4f | 姿态: %.4f\n\n', ...
    cmlkf_mean_p, cmlkf_mean_v, cmlkf_mean_att);


%% ========================================================================
%% 4.2 循环运行 DMLKF_L (固定 K，递增 L)
%% ========================================================================
for test_idx = 1:length(L_test_list)
    L = L_test_list(test_idx);
    fprintf('====================================================\n');
    fprintf('[Experiment %d] 正在运行 DMLKF_L (通信等级 L = %d)...\n', test_idx, L);
    
    % prue数据集不会有RMSE回升，200次迭代完全足够收敛，可以测试更小的iter提高运行速度
    kf_dmlkf_max_iter = 70; % 70次也够 在12车5基站数据集下
    kf_dmlkf          = DMLKF_L(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0, V2V_Mask, kf_dmlkf_max_iter, L); 
    kf_dmlkf.beta_inv = 0.01;
    kf_dmlkf.epsilon  = 1e-8;    % 关键；不设就是类默认的 1e-4
    kf_dmlkf.max_step = 1;       % 可留 1 或 Inf，无影响
    
    est_p_dmlkf = zeros(N_steps, 3, Vehicle_num);
    est_v_dmlkf = zeros(N_steps, 3, Vehicle_num);
    est_R_dmlkf = zeros(3, 3, Vehicle_num, N_steps);
    for i = 1:Vehicle_num
        est_p_dmlkf(1, :, i)    = p0(3*i-2:3*i)';
        est_v_dmlkf(1, :, i)    = v0(3*i-2:3*i)';
        est_R_dmlkf(:, :, i, 1) = R0(:, :, i); 
    end
    
    uwb_idx = 2; 
    for k = 2:N_steps
        acc_m  = zeros(3, Vehicle_num);
        gyro_m = zeros(3, Vehicle_num);
        for i = 1:Vehicle_num
            v_name       = sprintf('V%d', i);
            acc_m(:, i)  = trajectories.(v_name).IMU_acc_m(k-1, :)' - bias_comp_ratio * trajectories.(v_name).IMU_bias_a_true(k-1, :)';
            gyro_m(:, i) = trajectories.(v_name).IMU_gyro_m(k-1, :)' - bias_comp_ratio * trajectories.(v_name).IMU_bias_w_true(k-1, :)';
        end
        kf_dmlkf.predict(acc_m, gyro_m);
        
        curr_time = trajectories.V1.Time_true(k);
        if uwb_idx <= length(UWB_Time_Vec) && abs(curr_time - UWB_Time_Vec(uwb_idx)) < 1e-5
            uwb_anc_raw = zeros(Vehicle_num, Anchor_num);
            uwb_rel_raw = zeros(Vehicle_num, Vehicle_num);
            for i = 1:Vehicle_num
                v_name            = sprintf('V%d', i);
                uwb_anc_raw(i, :) = trajectories.(v_name).UWB_Anchor(uwb_idx, 2:end);
                uwb_rel_raw(i, :) = trajectories.(v_name).UWB_Relative(uwb_idx, 2:end);
            end
            
            uwb_anc_masked = uwb_anc_raw; uwb_anc_masked(Anchor_Mask_Full == 0) = NaN;
            uwb_rel_masked = uwb_rel_raw; uwb_rel_masked(Rel_Mask == 0) = NaN;
            
            kf_dmlkf.update(uwb_anc_masked, uwb_rel_masked);
            uwb_idx = uwb_idx + 1;
        end
        
        for i = 1:Vehicle_num
            est_p_dmlkf(k, :, i)    = kf_dmlkf.Nodes{i}.p'; 
            est_v_dmlkf(k, :, i)    = kf_dmlkf.Nodes{i}.v'; 
            est_R_dmlkf(:, :, i, k) = kf_dmlkf.Nodes{i}.R; 
        end
    end
    
    [mean_p, mean_v, mean_att] = calc_mean_rmse( ...
        Vehicle_num, N_steps, est_p_dmlkf, est_v_dmlkf, est_R_dmlkf, true_p_all, true_v_all, trajectories);
    results_summary(1 + test_idx, :) = [mean_p, mean_v, mean_att];
    fprintf('-> DMLKF_L (L=%d) 运行完毕 | 位置RMSE: %.4f m | 速度: %.4f | 姿态: %.4f\n\n', ...
        L, mean_p, mean_v, mean_att);
end


%% ========================================================================
%% 5. 构造汇总表、保存、终端打印与绘图
%% ========================================================================
method_col = ["CMLKF(mask)"; repmat("DMLKF_L", length(L_test_list), 1)];
l_col      = [NaN; L_test_list(:)];

summary_table = table( ...
    method_col, ...
    l_col, ...
    results_summary(:, 1), ...
    results_summary(:, 2), ...
    results_summary(:, 3), ...
    'VariableNames', {'Method', 'L_Level', 'Pos_RMSE', 'Vel_RMSE', 'Att_RMSE'});

% 按需保存 CSV 结果文件
if save_flag
    if ~exist(result_dir, 'dir')
        mkdir(result_dir);
    end
    writetable(summary_table, result_file);
    fprintf('>> 汇总数据已保存至: %s\n', result_file);
end

% 终端总结输出
print_summary_table(summary_table, K_degree);

% 绘制 RMSE vs L 趋势图并按需保存 PNG
plot_summary_results(summary_table, plot_file, save_flag);


%% ======================= 局部辅助函数区域 =======================

% -------------------------------------------------------------------------
% 函数: print_summary_table
% 功能: 打印 CMLKF vs DMLKF_L 最终对比总结表 (终端显示为 DMLKF_L)
% -------------------------------------------------------------------------
function print_summary_table(T, K_degree)
    cmlkf_row  = find(startsWith(string(T.Method), "CMLKF"), 1);
    dmlkf_rows = find(startsWith(string(T.Method), "DMLKF"));

    fprintf('\n');
    fprintf('===================================================================\n');
    fprintf('                CMLKF vs DMLKF_L 最终对比总结表                  \n');
    fprintf('===================================================================\n');
    fprintf(' CMLKF 测量集   : 与 DMLKF_L 相同的 K=%d 相对测距掩码 (同图对比)\n', K_degree);
    fprintf(' Method    | Comm Level (L) |  Pos RMSE (m) | Vel RMSE (m/s) | Att RMSE (deg)\n');
    fprintf('-------------------------------------------------------------------\n');
    if ~isempty(cmlkf_row)
        fprintf(' %-11s| Full (Central) |   %10.4f  |   %10.4f   |   %10.4f\n', ...
            'CMLKF(mask)', T.Pos_RMSE(cmlkf_row), T.Vel_RMSE(cmlkf_row), T.Att_RMSE(cmlkf_row));
    end
    fprintf('-------------------------------------------------------------------\n');
    for i = 1:length(dmlkf_rows)
        r = dmlkf_rows(i);
        fprintf(' DMLKF_L   | %-14d |   %10.4f  |   %10.4f   |   %10.4f\n', ...
            T.L_Level(r), T.Pos_RMSE(r), T.Vel_RMSE(r), T.Att_RMSE(r));
    end
    fprintf('===================================================================\n\n');
end

% -------------------------------------------------------------------------
% 函数: plot_summary_results
% 功能: 绘制 RMSE vs communicational level 趋势图并按需保存 PNG
% -------------------------------------------------------------------------
function plot_summary_results(T, plot_file, do_save)
    cmlkf_row = find(startsWith(string(T.Method), "CMLKF"), 1);
    dmlkf_T   = T(startsWith(string(T.Method), "DMLKF"), :);
    dmlkf_T   = sortrows(dmlkf_T, 'L_Level');

    fig = figure('Name', 'Position RMSE vs Communication Level L', ...
        'Color', 'w', 'Position', [200, 200, 700, 500]);
    hold on; grid on; box on;

    % 绘制 DMLKF 的曲线
    plot(dmlkf_T.L_Level, dmlkf_T.Pos_RMSE, '-o', ...
        'LineWidth', 2, 'MarkerSize', 8, ...
        'MarkerFaceColor', '#0072BD', 'Color', '#0072BD', ...
        'DisplayName', 'DMLKF');

    % 绘制 CMLKF 的基准线
    if ~isempty(cmlkf_row)
        yline(T.Pos_RMSE(cmlkf_row), '--r', ...
            'LineWidth', 2, 'LabelHorizontalAlignment', 'left', ...
            'DisplayName', 'CMLKF');
    end

    % 坐标轴、空字符串标题与图例配置
    xlabel('communicational level', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('RMSE', 'FontSize', 12, 'FontWeight', 'bold');
    title('');

    % 强制 x 轴刻度为整数
    xticks(dmlkf_T.L_Level);
    xlim([0.8, max(dmlkf_T.L_Level) + 0.2]);

    legend({'DMLKF', 'CMLKF'}, 'FontSize', 12, 'Location', 'northeast');
    set(gca, 'FontSize', 11, 'GridAlpha', 0.3);

    if do_save
        plot_dir = fileparts(plot_file);
        if ~exist(plot_dir, 'dir')
            mkdir(plot_dir);
        end
        exportgraphics(fig, plot_file, 'Resolution', 300);
        fprintf('>> 图表已保存至: %s\n', plot_file);
    end
end

% -------------------------------------------------------------------------
% 函数: generate_symmetric_mask
% 功能: 生成奇数/偶数皆对称的 V2V Mask
% -------------------------------------------------------------------------
function V2V_Mask = generate_symmetric_mask(V_num, K)
    V2V_Mask = zeros(V_num, V_num);
    K_half   = floor(K / 2); 
    
    for i = 1:V_num
        % 1. 绝对对称地连接前后各 K_half 个节点
        for d = 1:K_half
            idx_forward  = mod(i + d - 1, V_num) + 1;
            idx_backward = mod(i - d - 1, V_num) + 1;
            V2V_Mask(i, idx_forward)  = 1;
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

% -------------------------------------------------------------------------
% 函数: calc_mean_rmse
% 功能: 统一结算全局平均 RMSE
% -------------------------------------------------------------------------
function [mean_p, mean_v, mean_att] = calc_mean_rmse(V_num, N_steps, est_p, est_v, est_R, true_p, true_v, trajectories)
    rmse_p   = zeros(V_num, 1);
    rmse_v   = zeros(V_num, 1);
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
    mean_p   = mean(rmse_p);
    mean_v   = mean(rmse_v);
    mean_att = mean(rmse_att);
end