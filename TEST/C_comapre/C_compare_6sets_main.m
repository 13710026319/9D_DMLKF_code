%% C_compare_6sets_main.m
% =========================================================================
%  4 车 / 6 个数据集 / 每个数据集基站数 2~20 / 三算法对比
%
%  目的：找出"CMLKF 相对 CRBPF 的差距随基站数收缩最明显"的那一组数据
%        （即起始三算法差距大、后续 CMLKF 明显趋近 CRBPF，CEKF 始终留有余量）。
%
%  * 数据集：Data\C_compare\Trj_Veh4_Anc20_3D_<k>.mat，k = 1..6
%            不存在时用 c_compare_gen_dataset(4, 20, ..., 20260924+k) 生成
%            （同一 env_setup、同一套噪声参数，只换随机种子）
%  * CRBPF 直接使用算法文件里的粒子数（当前为 300），脚本不做任何设置
%  * bias_comp_ratio = 0.6，data_ratio = 100%
%  * 结果只保存位置 RMSE；分段结果在汇总后自动删除
%
%  运行：直接运行本脚本，会并行拉起 6x2 = 12 个 MATLAB 进程。
% =========================================================================

clc;

vehicle_num  = 4;
anchor_total = 20;
ds_list      = 1:6;
anchor_lo    = 2;
anchor_hi    = 20;
segments     = [anchor_lo 11; 12 anchor_hi];   % 每个数据集切两段并行
step_cap     = Inf;
clean_start  = true;
keep_parts   = false;      % false = 汇总后删除分段结果，只留总结果
max_wait_h   = 8;

this_dir = fileparts(mfilename('fullpath'));
if isempty(this_dir), this_dir = pwd; end
root_dir = fullfile(this_dir, 'RESULT', sprintf('Veh%d_Anc%d', vehicle_num, anchor_total));
if ~exist(root_dir, 'dir'), mkdir(root_dir); end

matlab_exe = fullfile(matlabroot, 'bin', 'matlab.exe');
if ~exist(matlab_exe, 'file'), error('找不到 matlab.exe: %s', matlab_exe); end
addpath(this_dir);

fprintf('================ 4 车 / %d 个数据集 / 基站 %d~%d ================\n', ...
        numel(ds_list), anchor_lo, anchor_hi);
fprintf(' 输出目录 : %s\n', root_dir);
fprintf(' 分段方案 : 每个数据集 %s 两段，共 %d 个并行进程\n', ...
        mat2str(segments), numel(ds_list)*size(segments,1));

%% ---------------- 1. 准备数据集 ----------------
data_files = cell(numel(ds_list), 1);
for k = ds_list
    data_files{k} = fullfile('E:\DMLKF_code', 'Data', 'C_compare', ...
        sprintf('Trj_Veh%d_Anc%d_3D_%d.mat', vehicle_num, anchor_total, k));
    if ~exist(data_files{k}, 'file')
        fprintf('生成数据集 ds%d ...\n', k);
        c_compare_gen_dataset(vehicle_num, anchor_total, data_files{k}, 20260924 + k);
    else
        fprintf('数据集 ds%d 已存在\n', k);
    end
end

%% ---------------- 2. 清空旧结果 ----------------
if clean_start
    for k = ds_list
        d = fullfile(root_dir, sprintf('ds%d', k));
        if exist(d, 'dir')
            old = [dir(fullfile(d, 'seg_*.csv')); dir(fullfile(d, 'seg_*.done'))];
            for f = 1:numel(old)
                delete(fullfile(d, old(f).name));
            end
        end
    end
    old = [dir(fullfile(root_dir, 'C_compare_*.csv')); dir(fullfile(root_dir, 'C_compare_*.png'))];
    for f = 1:numel(old)
        delete(fullfile(root_dir, old(f).name));
    end
end

%% ---------------- 3. 并行启动 ----------------
if isfinite(step_cap), capstr = sprintf('%d', round(step_cap)); else, capstr = 'Inf'; end

jobs = {};   % {ds, a1, a2, out_dir, log_file}
for k = ds_list
    ds_dir = fullfile(root_dir, sprintf('ds%d', k));
    log_dir = fullfile(ds_dir, 'logs');
    if ~exist(ds_dir, 'dir'), mkdir(ds_dir); end
    if ~exist(log_dir, 'dir'), mkdir(log_dir); end
    for s = 1:size(segments, 1)
        jobs{end+1} = {k, segments(s,1), segments(s,2), ds_dir, ...
                       fullfile(log_dir, sprintf('seg_%d_%d.log', segments(s,1), segments(s,2)))}; %#ok<SAGROW>
    end
end

for j = 1:numel(jobs)
    k = jobs{j}{1}; a1 = jobs{j}{2}; a2 = jobs{j}{3};
    stmt = sprintf(['addpath(''%s''); c_compare_worker(%d, %d, %s, ''%s'', true, ''%s'');'], ...
                   this_dir, a1, a2, capstr, jobs{j}{4}, data_files{k});
    cmd = sprintf('start "c6s%02d" /B "%s" -singleCompThread -batch "%s" > "%s" 2>&1', ...
                  j, matlab_exe, stmt, jobs{j}{5});
    st = system(cmd);
    if st ~= 0
        warning('任务 %d (ds%d, 基站 %d~%d) 启动失败', j, k, a1, a2);
    end
    pause(1.5);
end
fprintf('已启动 %d 个进程\n', numel(jobs));

%% ---------------- 4. 等待 ----------------
fprintf('\n等待全部进程结束 ...\n');
t0 = tic; last = -1;
while true
    done = 0;
    for j = 1:numel(jobs)
        if exist(fullfile(jobs{j}{4}, sprintf('seg_%d_%d.done', jobs{j}{2}, jobs{j}{3})), 'file')
            done = done + 1;
        end
    end
    el = toc(t0);
    if done ~= last
        fprintf('  [%6.1f min] 完成 %2d / %d\n', el/60, done, numel(jobs));
        last = done;
    end
    if done == numel(jobs), break; end
    if el > max_wait_h*3600, warning('等待超时，按已完成部分汇总'); break; end
    pause(20);
end
fprintf('总耗时 %.1f 分钟\n', toc(t0)/60);

%% ---------------- 5. 汇总 ----------------
c_compare_6sets_collect(root_dir, true, ~keep_parts);
