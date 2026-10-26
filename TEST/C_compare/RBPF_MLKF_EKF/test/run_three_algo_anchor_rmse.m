%% CRBPF / CMLKF / CEKF 位置 RMSE 随基站数变化
% 用法：在 MATLAB 中进入本 test 目录，运行
%   run_three_algo_anchor_rmse
%
% 修改 noise_cfg 中对应算法的 4 个噪声参数即可重复实验：
%   IMU_Sigma_a、IMU_Sigma_w 是 3x3 协方差，通常写成 (sigma)^2*eye(3)；
%   UWB_sigma_anc、UWB_sigma_rel 是标准差。
%
% run_flag：1=重新运行；0=若完整 CSV 已存在则直接读取并重新绘图。
% save_flag：1=保存最终 CSV 和 PNG；0=只在内存中计算，不保存结果。
% 不保存任何单次滤波估计或中间 MAT 文件。
clc;
clear;
close all;

%% 1. 运行配置
bias_comp_ratio = 0.1;
data_ratio      = 0.9;
run_flag        = 1;
save_flag       = 1;

%% 2. 外部噪声参数
% CRBPF
noise_cfg.CRBPF.IMU_Sigma_a   = (0.25)^2  * eye(3);
noise_cfg.CRBPF.IMU_Sigma_w   = (0.015)^2 * eye(3);
noise_cfg.CRBPF.UWB_sigma_anc = 0.08;
noise_cfg.CRBPF.UWB_sigma_rel = 0.08;

% CEKF
noise_cfg.CEKF.IMU_Sigma_a   = (0.05)^2  * eye(3);
noise_cfg.CEKF.IMU_Sigma_w   = (0.005)^2 * eye(3);
noise_cfg.CEKF.UWB_sigma_anc = 0.035;
noise_cfg.CEKF.UWB_sigma_rel = 0.035;

% CMLKF
noise_cfg.CMLKF.IMU_Sigma_a   = (0.05)^2  * eye(3);
noise_cfg.CMLKF.IMU_Sigma_w   = (0.005)^2 * eye(3);
noise_cfg.CMLKF.UWB_sigma_anc = 0.035;
noise_cfg.CMLKF.UWB_sigma_rel = 0.035;

%% 3. 路径和输出文件
this_dir      = fileparts(mfilename('fullpath'));
exp_root      = fileparts(this_dir);
project_root  = fileparts(fileparts(fileparts(exp_root)));
data_file     = fullfile(exp_root, 'data', 'Trj_Veh6_Anc30_3D.mat');
result_dir    = fullfile(exp_root, 'result');
csv_file      = fullfile(result_dir, 'ThreeAlgo_Anchor_Num_RMSE.csv');
png_file      = fullfile(result_dir, 'ThreeAlgo_Anchor_Num_RMSE.png');

addpath(this_dir);
addpath(fullfile(project_root, 'EKF',  'CEKF'));
addpath(fullfile(project_root, 'MLKF', 'CMLKF'));
addpath(fullfile(project_root, 'RBPF', 'CRBPF'));

expected_anchor_nums = (4:30)';

%% 4. run_flag=0：优先读取已有完整 CSV
if run_flag == 0 && exist(csv_file, 'file')
    try
        R_cached = readmatrix(csv_file);
        cache_ok = size(R_cached, 2) == 4 && ...
                   isequal(R_cached(:, 1), expected_anchor_nums);
    catch
        cache_ok = false;
    end

    if cache_ok
        fprintf('读取已有结果：%s\n', csv_file);
        if save_flag
            save_three_algo_outputs(R_cached, csv_file, png_file);
        end
        print_rmse_table(R_cached);
        return;
    end
    warning('已有 CSV 不完整或格式不符，将重新运行实验。');
end

if ~exist(data_file, 'file')
    error('找不到数据集：%s', data_file);
end
if save_flag && ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

%% 5. 启动并行池；不可用时自动串行
use_parallel = ~isempty(ver('parallel')) && ...
               license('test', 'Distrib_Computing_Toolbox');
if use_parallel
    try
        pool = gcp('nocreate');
        if isempty(pool)
            pool = parpool('Processes');
        end
        fprintf('并行计算：%d 个 worker。\n', pool.NumWorkers);
    catch ME
        warning('ThreeAlgo:ParallelStartFailed', ...
                '并行池启动失败，改用串行计算：%s', ME.message);
        use_parallel = false;
    end
end
if ~use_parallel
    fprintf('未启用并行计算，串行运行。\n');
end

%% 6. 基站数按 5 个一组提交并行任务
anchor_groups = {4:8, 9:13, 14:18, 19:23, 24:28, 29:30};
group_results = repmat(struct('anchor_nums', [], 'rmse', []), ...
                       1, numel(anchor_groups));

if use_parallel
    parfor g = 1:numel(anchor_groups)
        group_results(g) = run_three_algo_anchor_group( ...
            anchor_groups{g}, data_file, bias_comp_ratio, data_ratio, ...
            noise_cfg, project_root);
    end
else
    for g = 1:numel(anchor_groups)
        group_results(g) = run_three_algo_anchor_group( ...
            anchor_groups{g}, data_file, bias_comp_ratio, data_ratio, ...
            noise_cfg, project_root);
    end
end

%% 7. 汇总并保存
R = zeros(sum(cellfun(@numel, anchor_groups)), 4);
row = 0;
for g = 1:numel(group_results)
    n = numel(group_results(g).anchor_nums);
    R(row+1:row+n, :) = [group_results(g).anchor_nums(:), ...
                         group_results(g).rmse'];
    row = row + n;
end
R = sortrows(R, 1);

if ~isequal(R(:, 1), expected_anchor_nums)
    error('汇总结果未完整覆盖基站数 4:30。');
end

print_rmse_table(R);

if save_flag
    save_three_algo_outputs(R, csv_file, png_file);
    fprintf('\nCSV：%s\nPNG：%s\n', csv_file, png_file);
else
    fprintf('\nsave_flag=0：本次结果未保存。\n');
end

%% 控制台表格
function print_rmse_table(R)
    fprintf('\nAnc |   CRBPF |   CMLKF |    CEKF\n');
    fprintf('----+---------+---------+--------\n');
    for r = 1:size(R, 1)
        fprintf('%3d | %7.4f | %7.4f | %7.4f\n', ...
                R(r, 1), R(r, 2), R(r, 3), R(r, 4));
    end
end
