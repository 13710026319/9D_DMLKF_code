clc; clear; close all;

%% Parameters
Vehicle_num = 8;
Anchor_num  = 4;
neighbor_k  = 7;

dataset_name    = 'Trj_Veh8_Anc4_pure.mat';
bias_comp_ratio = 1;
data_ratio      = 0.2;

run_flag  = 1;
save_flag = 1;

V1_max_iter = 30;
DMLKF_iter_groups = {
    [1 2 3], ...
    [5 7 10], ...
    [15 20 25], ...
    [30 35], ...
    [50]
};

V1_beta_inv       = 0.01;
V1_max_step       = Inf;
DMLKF_beta_inv     = 0.2;
DMLKF_max_step     = Inf;
DMLKF_ALPHA_SAFETY = 1;
DMLKF_epsilon      = 1e-4;

max_wait_hours = 6;

%% Paths and result signature
test_dir     = fileparts(mfilename('fullpath'));
old_test_dir = fileparts(test_dir);
project_root = fileparts(fileparts(old_test_dir));
data_file    = fullfile(old_test_dir, 'Data', dataset_name);
result_dir   = fullfile(old_test_dir, 'Result');

[all_dmlkf_iters, active_groups] = prepare_iterations( ...
    DMLKF_iter_groups, V1_max_iter);

result_tag = make_result_tag(dataset_name, Vehicle_num, Anchor_num, ...
    neighbor_k, data_ratio, bias_comp_ratio, V1_max_iter, all_dmlkf_iters);
result_file = fullfile(result_dir, [result_tag '.csv']);
plot_file   = fullfile(result_dir, [result_tag '.png']);

if run_flag == 0
    if exist(result_file, 'file')
        T = readtable(result_file, 'VariableNamingRule', 'preserve', ...
            'TextType', 'string');
        print_results(T);
        plot_results(T, result_file, plot_file, neighbor_k, Vehicle_num, ...
            Anchor_num, data_ratio, save_flag);
    else
        fprintf('未找到当前配置的保存结果，请先将 run_flag 设为 1 并运行实验。\n');
    end
    return;
end

if ~isfile(data_file)
    error('找不到数据集: %s', data_file);
end

cfg = struct();
cfg.data_file          = data_file;
cfg.Vehicle_num        = Vehicle_num;
cfg.Anchor_num         = Anchor_num;
cfg.neighbor_k         = neighbor_k;
cfg.bias_comp_ratio    = bias_comp_ratio;
cfg.data_ratio         = data_ratio;
cfg.V1_max_iter        = V1_max_iter;
cfg.V1_beta_inv        = V1_beta_inv;
cfg.V1_max_step        = V1_max_step;
cfg.DMLKF_beta_inv     = DMLKF_beta_inv;
cfg.DMLKF_max_step     = DMLKF_max_step;
cfg.DMLKF_ALPHA_SAFETY = DMLKF_ALPHA_SAFETY;
cfg.DMLKF_epsilon      = DMLKF_epsilon;

%% Parallel jobs
tmp_dir = fullfile(test_dir, ...
    ['newton_iter_tmp_' char(datetime('now', 'Format', 'yyyyMMdd_HHmmssSSS'))]);
mkdir(tmp_dir);

cfg_file = fullfile(tmp_dir, 'config.mat');
save(cfg_file, 'cfg');

jobs = struct('mode', {}, 'iters', {}, 'tag', {});
jobs(end+1) = struct('mode', 'V1', 'iters', V1_max_iter, 'tag', 'V1');
for q = 1:numel(active_groups)
    jobs(end+1) = struct('mode', 'DMLKF', ...
        'iters', active_groups{q}, 'tag', sprintf('DMLKF_%d', q)); %#ok<AGROW>
end

matlab_exe = fullfile(matlabroot, 'bin', 'matlab.exe');
if ~isfile(matlab_exe)
    error('找不到 matlab.exe: %s', matlab_exe);
end

fprintf('数据集: %s\n', data_file);
fprintf('V1 max_iter: %d\n', V1_max_iter);
fprintf('DMLKF max_iter: %s\n', mat2str(all_dmlkf_iters));
fprintf('启动并行任务: %d\n', numel(jobs));

try
    for q = 1:numel(jobs)
        jobs(q).part_file = fullfile(tmp_dir, [jobs(q).tag '.csv']);
        jobs(q).done_file = [jobs(q).part_file '.done'];
        jobs(q).log_file  = fullfile(tmp_dir, [jobs(q).tag '.log']);

        statement = sprintf(['addpath(%s); Newton_iter_worker(%s,%s,%s,%s);'], ...
            quote_text(test_dir), quote_text(jobs(q).mode), ...
            mat2str(jobs(q).iters), quote_text(jobs(q).part_file), ...
            quote_text(cfg_file));
        cmd = sprintf(['start "newton_iter_%d" /B "%s" ' ...
            '-singleCompThread -batch "%s" > "%s" 2>&1'], q, ...
            matlab_exe, statement, jobs(q).log_file);

        status = system(cmd);
        if status ~= 0
            error('并行任务启动失败: %s', jobs(q).tag);
        end
        pause(1);
    end

    wait_for_jobs(jobs, max_wait_hours);
    T = collect_results(jobs, all_dmlkf_iters, V1_max_iter);
catch err
    stop_workers(tmp_dir);
    cleanup_temp(tmp_dir, data_file);
    rethrow(err);
end

cleanup_temp(tmp_dir, data_file);
print_results(T);

if save_flag
    if ~exist(result_dir, 'dir')
        mkdir(result_dir);
    end
    writetable(T, result_file);
    fprintf('结果已保存: %s\n', result_file);
end

plot_results(T, result_file, plot_file, neighbor_k, Vehicle_num, ...
    Anchor_num, data_ratio, save_flag);

%% Local functions
function [all_iters, active_groups] = prepare_iterations(groups, v1_iter)
if ~iscell(groups) || numel(groups) ~= 5
    error('DMLKF_iter_groups 必须是包含 5 个列表的 cell 数组。');
end
if ~is_positive_integer(v1_iter)
    error('V1_max_iter 必须是正整数。');
end

active_groups = {};
for q = 1:numel(groups)
    g = groups{q};
    if isempty(g)
        continue;
    end
    if ~isnumeric(g) || ~isvector(g) || any(~isfinite(g)) || ...
            any(g <= 0) || any(g ~= round(g))
        error('DMLKF_iter_groups{%d} 必须是正整数向量或空数组。', q);
    end
    active_groups{end+1} = g(:)'; %#ok<AGROW>
end

if isempty(active_groups)
    error('至少需要一个非空的 DMLKF 迭代列表。');
end

all_iters = sort([active_groups{:}]);
if numel(all_iters) ~= numel(unique(all_iters))
    error('DMLKF 迭代列表中不能出现重复的 max_iter。');
end
end

function tf = is_positive_integer(v)
tf = isnumeric(v) && isscalar(v) && isfinite(v) && v > 0 && v == round(v);
end

function tag = make_result_tag(dataset_name, V_num, A_num, K, ...
        data_ratio, bias_ratio, v1_iter, dml_iters)
[~, stem] = fileparts(dataset_name);
stem = regexprep(stem, '[^0-9A-Za-z_-]', '_');
ratio_txt = number_tag(data_ratio);
bias_txt  = number_tag(bias_ratio);
iter_txt  = strjoin(string(dml_iters), '-');
tag = sprintf('Newton_iter_%s_V%d_A%d_K%d_r%s_b%s_V1-%d_DML-%s', ...
    stem, V_num, A_num, K, ratio_txt, bias_txt, v1_iter, iter_txt);
end

function s = number_tag(v)
s = strrep(sprintf('%.6g', v), '.', 'p');
s = strrep(s, '-', 'm');
end

function s = quote_text(v)
v = char(v);
s = ['''' strrep(v, '''', '''''') ''''];
end

function wait_for_jobs(jobs, max_wait_hours)
timer_id = tic;
while true
    done_count = 0;
    for q = 1:numel(jobs)
        if isfile(jobs(q).done_file)
            done_count = done_count + 1;
        end
    end
    if done_count == numel(jobs)
        return;
    end
    if toc(timer_id) > max_wait_hours * 3600
        error('等待并行任务超时，完成 %d/%d。', done_count, numel(jobs));
    end
    pause(5);
end
end

function T = collect_results(jobs, expected_iters, v1_iter)
parts = table();
for q = 1:numel(jobs)
    if ~isfile(jobs(q).part_file)
        error('并行任务未生成结果: %s', jobs(q).tag);
    end
    current = readtable(jobs(q).part_file, ...
        'VariableNamingRule', 'preserve', 'TextType', 'string');
    if isempty(parts)
        parts = current;
    else
        parts = [parts; current]; %#ok<AGROW>
    end
end

failed = parts.ok ~= 1 | ~isfinite(parts.pos_rmse_m);
if any(failed)
    fprintf('\n失败任务:\n');
    disp(parts(failed, {'algorithm', 'max_iter', 'note'}));
    error('存在未成功完成的滤波任务。');
end

v1_rows = find(parts.algorithm == "V1");
if numel(v1_rows) ~= 1 || parts.max_iter(v1_rows) ~= v1_iter
    error('V1 结果缺失或不唯一。');
end

dml_rows = parts.algorithm == "DMLKF";
dml_iters = sort(parts.max_iter(dml_rows));
if ~isequal(dml_iters(:)', expected_iters(:)')
    error('DMLKF 结果数量与配置不一致。');
end

v1_rmse = parts.pos_rmse_m(v1_rows);
dml_table = sortrows(parts(dml_rows, :), 'max_iter');

algorithm = [string("V1"); dml_table.algorithm];
max_iter = [parts.max_iter(v1_rows); dml_table.max_iter];
pos_rmse_m = [v1_rmse; dml_table.pos_rmse_m];

T = table(algorithm, max_iter, pos_rmse_m);
end

function print_results(T)
fprintf('\n%-8s | %10s | %12s\n', ...
    'algorithm', 'max_iter', 'RMSE(m)');
fprintf('%s\n', repmat('-', 1, 38));
for q = 1:height(T)
    fprintf('%-8s | %10d | %12.4f\n', ...
        T.algorithm(q), T.max_iter(q), T.pos_rmse_m(q));
end
fprintf('%s\n\n', repmat('-', 1, 38));
end

function plot_results(T, result_file, plot_file, K, V_num, A_num, ...
        data_ratio, do_save)
v1_row = find(T.algorithm == "V1", 1);
dml = T(T.algorithm == "DMLKF", :);
dml = sortrows(dml, 'max_iter');

fig = figure('Name', 'Newton iteration comparison', 'Color', 'w');
hold on; grid on; box on;
plot(dml.max_iter, dml.pos_rmse_m, 'o-', 'LineWidth', 1.8, ...
    'MarkerSize', 7, 'MarkerFaceColor', 'w', ...
    'Color', [0.85 0.20 0.15], 'DisplayName', 'DMLKF');

v1_rmse = T.pos_rmse_m(v1_row);
yline(v1_rmse, '--', 'LineWidth', 1.6, ...
    'Color', [0.10 0.35 0.80], ...
    'Label', sprintf('V1: %.4f', v1_rmse), ...
    'LabelHorizontalAlignment', 'left', ...
    'LabelVerticalAlignment', 'bottom', ...
    'DisplayName', sprintf('V1 max_iter = %d', T.max_iter(v1_row)));

for q = 1:height(dml)
    text(dml.max_iter(q), dml.pos_rmse_m(q), ...
        sprintf('  %.4f', dml.pos_rmse_m(q)), ...
        'FontSize', 8, 'VerticalAlignment', 'bottom', ...
        'Interpreter', 'none');
end

set(gca, 'XScale', 'log');
xticks(dml.max_iter);
xticklabels(cellstr(string(dml.max_iter)));
xlim([min(dml.max_iter)/1.35, max(dml.max_iter)*1.45]);
ytickformat('%.4f');
xlabel('DMLKF max_iter', 'Interpreter', 'none');
ylabel('Position RMSE (m)');
title(sprintf(['Position RMSE vs. max_iter ' ...
    '(K = %d, %d vehicles, %d anchors, %.0f%% data)'], ...
    K, V_num, A_num, 100*data_ratio), 'Interpreter', 'none');
legend('Location', 'southeast');

if do_save
    exportgraphics(fig, plot_file, 'Resolution', 150);
    fprintf('图片已保存: %s\n', plot_file);
end
end

function cleanup_temp(tmp_dir, data_file)
if ~strcmpi(fileparts(tmp_dir), fileparts(data_file)) && ...
        contains(tmp_dir, 'newton_iter_tmp_') && exist(tmp_dir, 'dir')
    rmdir(tmp_dir, 's');
end
end

function stop_workers(tmp_dir)
if ~ispc
    return;
end
cmd = sprintf(['powershell -NoProfile -Command "Get-CimInstance ' ...
    'Win32_Process | Where-Object { $_.Name -like ''matlab*'' -and ' ...
    '$_.CommandLine -like ''*%s*'' } | ForEach-Object { ' ...
    'Stop-Process -Id $_.ProcessId -Force }"'], tmp_dir);
system(cmd);
end
