function T = iter_compare_main(K, iter_list, data_ratio, dataset_name, save_png, cfg)
%ITER_COMPARE_MAIN 任务2：固定邻居数 K 下扫描分布式 GN 的 max_iter，
%                   看 RMSE 是否随 iter 增大逼近集中式 GN（V1 基准线）
%
% 直接运行（默认配置，即任务2 要求的这一组）：
%     iter_compare_main
%
% 自定义调用：
%     iter_compare_main(4, [10 20 30 40 50 80 100 150], 0.2)         % 默认参数
%     iter_compare_main(4, [10 20 30], 0.2, 'Trj_Veh6_Anc4_3D.mat')  % 换数据集
%     iter_compare_main(4, [], 0.2, 'Trj_Veh8_Anc4_3D.mat', false)   % 只打印不存图
%     % 想要自己设参数（例如把固定步长安全系数改成 0.2、max_step 改成 0.5）：
%     cfg = struct('ALPHA_SAFETY', 0.2, 'max_step', 0.5, 'epsilon', 1e-5);
%     iter_compare_main(4, [], 0.2, 'Trj_Veh8_Anc4_3D.mat', true, cfg)
%
% 默认配置：Data\Trj_Veh8_Anc4_3D.mat（8 车 4 基站），K = 4，
%       iter = [10 20 30 40 50 80 100 150]，data_ratio = 0.2（数据集前 20%），
%       基站全连通；V1 基线与 DMLKF_C 各点使用完全相同的这段数据与同一张车间掩码。
%
% 结果文件按"车数 + 基站数"自动命名：
%       RESULT\iter_<V>V_<A>A.csv     例如 RESULT\iter_8V_4A.csv
%   如果传了 cfg，文件名会再带上参数标签，例如
%       RESULT\iter_8V_4A_ALPHA_SAFETY0p2_max_step0p5.csv
%   这样不同参数组的结果不会互相覆盖，复用检查也仍然有效。
%
% 运行前先检查该文件是否存在：
%     * 已存在  -> 直接读取、打印、绘图，不再重跑（要强制重跑请先删除该文件）
%     * 不存在  -> 启动并行计算：1 个 V1 基准进程 + iter 列表按"小 + 大"配对，
%                  [10 20 30 40 50 80 100 150] -> (10,150) (20,100) (30,80) (40,50)
%
% 输出：
%       RESULT\iter_<V>V_<A>A.csv   3 列：algorithm,iterations,pos_rmse_m
%                                   （保留 4 位小数；V1 行的 iterations 记 30）
%       RESULT\iter_<V>V_<A>A.png   save_png = true 时保存，图例英文
%       中间分片放在 RESULT\_parts\，汇总后自动删除
%
% 打印、绘图与保存的数值统一保留小数点后 4 位。

if nargin < 1 || isempty(K),            K = 7; end % 邻居数
if nargin < 2 || isempty(iter_list),    iter_list = [10 20 30 40 50 80 100 150]; end
if nargin < 3 || isempty(data_ratio),   data_ratio = 0.2; end   % 任务约定：只用数据集的 20%
if nargin < 4 || isempty(dataset_name), dataset_name = 'Trj_Veh8_Anc4_pure.mat'; end
if nargin < 5 || isempty(save_png),     save_png = true; end
if nargin < 6,                          cfg = struct(); end

anchor_mode = 'full';   % 任务2 固定口径：基站全连通
iter_list   = sort(iter_list(:))';

% ---------------- 路径与数据集元信息 ----------------
this_dir = fileparts(mfilename('fullpath'));
gn_dir   = fileparts(this_dir);
root     = fileparts(fileparts(gn_dir));
addpath(root, this_dir, gn_dir, ...
        fullfile(root, 'MLKF', 'DMLKF'), fullfile(root, 'MLKF', 'CMLKF'), ...
        fullfile(root, 'Data'));

data_file = fullfile(root, 'TEST', 'GN_compare', 'Data', dataset_name);
if ~exist(data_file, 'file'), error('找不到数据集: %s', data_file); end

% 车辆/基站数（决定结果文件名，也用于打印与图标题）
V_num = NaN; A_num = NaN;
try
    S_meta = load(data_file, 'Vehicle_num', 'Anchor_num');
    V_num = S_meta.Vehicle_num;
    A_num = S_meta.Anchor_num;
    clear S_meta
catch
end
if ~isfinite(V_num) || ~isfinite(A_num)
    error('无法从数据集读取 Vehicle_num / Anchor_num: %s', data_file);
end

res_dir  = fullfile(this_dir, 'RESULT');
part_dir = fullfile(res_dir, '_parts');
if ~exist(res_dir, 'dir'), mkdir(res_dir); end

out_csv = fullfile(res_dir, sprintf('iter_%dV_%dA%s.csv', V_num, A_num, cfg_tag(cfg)));
out_png = fullfile(res_dir, sprintf('iter_%dV_%dA%s.png', V_num, A_num, cfg_tag(cfg)));

fprintf('=========== 任务2: %d 车 %d 基站，K = %d 下的 max_iter 扫描 ===========\n', ...
        V_num, A_num, K);
fprintf(' 数据集   : %s\n', data_file);
fprintf(' iter 列表: %s\n', mat2str(iter_list));
fprintf(' 数据比例 : %g%s（只截取时间序列前段，两算法同口径）\n', data_ratio, steps_txt(data_file, data_ratio));
fprintf(' 基站掩码 : %s\n', anchor_mode);
fprintf(' 结果文件 : %s\n\n', out_csv);
if ~isempty(cfg)
    fprintf(' 自定义参数: %s\n\n', cfg_to_str(cfg));
end

% ================================================================
% 已有结果：直接读取、打印、绘图，不重跑
% ================================================================
if exist(out_csv, 'file')
    fprintf(' [复用] 已存在 %d 车 %d 基站 的结果文件，直接读取（不重跑）。\n', V_num, A_num);
    fprintf('        如需重新计算，请先删除: %s\n\n', out_csv);

    T = readtable(out_csv, 'VariableNamingRule', 'preserve', 'TextType', 'string');
    if ~ismember('pos_rmse_m', T.Properties.VariableNames)
        error('结果文件缺少 pos_rmse_m 列: %s', out_csv);
    end
    T = normalize_result_table(T);
else

    % ============================================================
    % 没有结果：启动并行计算
    % ============================================================
    if ~exist(part_dir, 'dir'), mkdir(part_dir); end
    old = [dir(fullfile(part_dir, '*.csv')); dir(fullfile(part_dir, '*.done'))];
    for q = 1:numel(old), delete(fullfile(part_dir, old(q).name)); end

    n_iter = numel(iter_list);

    % 并行分组：小 + 大 配对
    lo = iter_list(1:ceil(n_iter/2));
    hi = flipud(iter_list(ceil(n_iter/2)+1:end)');
    groups = cell(numel(lo), 1);
    for q = 1:numel(lo)
        if q <= numel(hi), groups{q} = [lo(q), hi(q)]; else, groups{q} = lo(q); end
    end

    matlab_exe = fullfile(matlabroot, 'bin', 'matlab.exe');
    if ~exist(matlab_exe, 'file'), error('找不到 matlab.exe: %s', matlab_exe); end

    jobs = struct('tag', {}, 'stmt', {}, 'csv', {}, 'done', {});

    tag = 'V1';
    stmt = sprintf(['addpath(''%s'',''%s''); iter_compare_worker(''V1'',30,%d,''%s'',''%s'',%g,''%s'',%s);'], ...
                   this_dir, gn_dir, K, fullfile(part_dir, [tag '.csv']), data_file, data_ratio, ...
                   anchor_mode, cfg_to_str(cfg));
    jobs(end+1) = struct('tag', tag, 'stmt', stmt, ...
                         'csv', fullfile(part_dir, [tag '.csv']), 'done', fullfile(part_dir, [tag '.csv.done']));

    for q = 1:numel(groups)
        g   = groups{q};
        tag = sprintf('DMLKF_group%d', q);
        itxt = ['[' strjoin(cellstr(string(g)), ',') ']'];
        stmt = sprintf(['addpath(''%s'',''%s''); iter_compare_worker(''DMLKF_C'',%s,%d,''%s'',''%s'',%g,''%s'',%s);'], ...
                       this_dir, gn_dir, itxt, K, fullfile(part_dir, [tag '.csv']), data_file, data_ratio, ...
                       anchor_mode, cfg_to_str(cfg));
        jobs(end+1) = struct('tag', tag, 'stmt', stmt, ...
                             'csv', fullfile(part_dir, [tag '.csv']), 'done', fullfile(part_dir, [tag '.csv.done'])); %#ok<AGROW>
    end

    fprintf(' 并行分组 : %s  + V1 基线\n\n', strjoin(cellfun(@(g) ['(' strjoin(cellstr(string(g)), ',') ')'], ...
            groups, 'UniformOutput', false), ' '));

    for j = 1:numel(jobs)
        cmd = sprintf('start "ic%02d" /B "%s" -singleCompThread -batch "%s" > NUL 2>&1', ...
                      j, matlab_exe, jobs(j).stmt);
        st = system(cmd);
        if st ~= 0, warning('作业 %s 启动失败', jobs(j).tag); end
        pause(1.5);
    end
    fprintf(' 已启动 %d 个进程（1 个 V1 + %d 个分组），等待全部完成 ...\n', numel(jobs), numel(groups));

    t0 = tic; last = -1; max_wait_h = 6;
    try
        while true
            done_cnt = 0;
            for j = 1:numel(jobs)
                if exist(jobs(j).done, 'file') || exist(fullfile(part_dir, [jobs(j).tag '.done']), 'file')
                    done_cnt = done_cnt + 1;
                end
            end
            el = toc(t0);
            if done_cnt ~= last
                fprintf('  [%6.1f min] 完成 %d / %d\n', el/60, done_cnt, numel(jobs));
                last = done_cnt;
            end
            if done_cnt == numel(jobs), break; end
            if el > max_wait_h*3600
                warning('等待超时（%g h），按已完成部分汇总', max_wait_h);
                break;
            end
            pause(20);
        end
    catch err
        % Ctrl+C / 停止：子进程是独立进程，不会被自动结束，这里尽力回收
        fprintf('\n[中断] 已停止等待，正在回收本次启动的后台 worker ...\n');
        kill_workers_by_parts(part_dir);
        fprintf('[中断] 若仍有残留，请在 MATLAB 里运行 iter_compare_stop 再确认一次。\n');
        rethrow(err);
    end
    fprintf(' 并行总耗时 %.1f 分钟\n\n', toc(t0)/60);

    T = read_parts(jobs);
    if isempty(T), error('没有读到任何结果，检查 %s', part_dir); end
    T = normalize_result_table(T);

    % 写出结果 CSV（保留 4 位小数）
    fid = fopen(out_csv, 'w');
    fprintf(fid, 'algorithm,iterations,pos_rmse_m\n');
    for q = 1:height(T)
        fprintf(fid, '%s,%d,%.4f\n', T.algorithm(q), T.iterations(q), T.pos_rmse(q));
    end
    fclose(fid);
    fprintf(' 已写出: %s\n', out_csv);

    % 清理中间文件
    stale = [dir(fullfile(part_dir, '*.csv')); dir(fullfile(part_dir, '*.done'))];
    for q = 1:numel(stale), delete(fullfile(part_dir, stale(q).name)); end
    try
        rmdir(part_dir);
    catch
    end
end

% ---------------- 打印（4 位小数）----------------
print_results(T, K);

% ---------------- 绘图（英文图例；绘图逻辑在 iter_compare_plot.m）----------------
if save_png
    iter_compare_plot(out_csv, out_png, K, V_num, A_num, data_ratio);
else
    iter_compare_plot(out_csv, '', K, V_num, A_num, data_ratio);
end
end

% ================================================================ 打印
function print_results(T, K)
is_v1 = strcmp(T.algorithm, 'V1');
v1_rmse = NaN;
if any(is_v1), v1_rmse = T.pos_rmse(find(is_v1, 1)); end
T = [T(is_v1, :); sortrows(T(~is_v1, :), 'iterations')];

fprintf('=============== 结果（位置 RMSE，全网平均，保留 4 位小数）===============\n');
fprintf(' %-8s | %10s | %12s | %12s | %12s | %10s\n', ...
        'algo', 'iterations', 'RMSE(m)', 'vs V1(mm)', 'mean GN iter', 'cap hit');
fprintf(' %s\n', repmat('-', 1, 92));
for q = 1:height(T)
    if strcmp(T.algorithm(q), 'V1')
        fprintf(' %-8s | %10d | %12.4f | %12s | %12s | %12s\n', ...
                T.algorithm(q), T.iterations(q), T.pos_rmse(q), 'baseline', '-', '-');
    else
        gap = T.pos_rmse(q) - v1_rmse;
        if isfinite(T.mean_gn_iter(q))
            it_txt = sprintf('%.4f', T.mean_gn_iter(q));
        else
            it_txt = '-';
        end
        if isfinite(T.cap_frac(q))
            cap_txt = sprintf('%.4f%%', 100*T.cap_frac(q));
        else
            cap_txt = '-';
        end
        fprintf(' %-8s | %10d | %12.4f | %12.4f | %12s | %10s\n', ...
                T.algorithm(q), T.iterations(q), T.pos_rmse(q), 1000*gap, it_txt, cap_txt);
    end
end
fprintf(' %s\n', repmat('-', 1, 92));
fprintf(' V1 基准（集中式 GN，K = %d）= %.4f m —— 分布式 GN 完全收敛时应当逼近的 RMSE\n', K, v1_rmse);

dm = T(~strcmp(T.algorithm, 'V1'), :);
if ~isempty(dm) && isfinite(v1_rmse)
    fprintf(' 最大 iter（%d）：RMSE = %.4f m，与 V1 差 %+.4f mm\n', ...
            dm.iterations(end), dm.pos_rmse(end), 1000*(dm.pos_rmse(end) - v1_rmse));
    if all(isfinite(dm.cap_frac))
        if dm.cap_frac(end) < 0.005
            fprintf(' [判定] 最大 iter 时已基本没有"跑满"的更新（跑满比例 %.4f%%），\n', 100*dm.cap_frac(end));
            fprintf('        即 D-GN 内循环已收敛，曲线进入平台并与 V1 基准线重合。\n');
        else
            fprintf(' [判定] 最大 iter 时仍有 %.4f%% 的更新跑满预算，说明迭代次数还不够。\n', ...
                    100*dm.cap_frac(end));
        end
    end
end
fprintf('\n');
end

% ================================================================ 工具
function s = cfg_tag(cfg)
%CFG_TAG 由参数结构体生成文件名标签（空的 cfg 返回空串，即沿用默认文件名）
if isempty(cfg), s = ''; return; end
fn = sort(fieldnames(cfg));
parts = cell(numel(fn), 1);
for q = 1:numel(fn)
    v = cfg.(fn{q});
    if ischar(v), vs = v; else, vs = mat2str(v); end
    vs = regexprep(vs, '[^0-9A-Za-z]', 'p');    % 0.5 -> 0p5
    parts{q} = sprintf('%s%s', fn{q}, vs);
end
s = ['_' strjoin(parts, '_')];
end

function s = cfg_to_str(cfg)
%CFG_TO_STR 把参数结构体序列化成可嵌入批处理命令的 MATLAB 表达式
if isempty(cfg), s = 'struct()'; return; end
fn = sort(fieldnames(cfg));
parts = cell(numel(fn), 1);
for q = 1:numel(fn)
    v = cfg.(fn{q});
    if ischar(v)
        vs = sprintf('''%s''', v);
    else
        vs = mat2str(v);
    end
    parts{q} = sprintf('''%s'',%s', fn{q}, vs);
end
s = ['struct(' strjoin(parts, ',') ')'];
end

function T = normalize_result_table(T)
% 统一列名：algorithm / iterations / pos_rmse / mean_gn_iter / cap_frac / sec / ok / note
v = T.Properties.VariableNames;
if ismember('pos_rmse_m', v), T.Properties.VariableNames{'pos_rmse_m'} = 'pos_rmse'; end
n = height(T);
if ~ismember('mean_gn_iter', v), T.mean_gn_iter = NaN(n, 1); end
if ~ismember('cap_frac',     v), T.cap_frac     = NaN(n, 1); end
if ~ismember('sec',          v), T.sec          = NaN(n, 1); end
if ~ismember('ok',           v), T.ok           = ones(n, 1); end
if ~ismember('note',         v), T.note         = strings(n, 1); end
end

function s = steps_txt(data_file, data_ratio)
s = '';
try
    S_info = load(data_file, 'trajectories');
    N_total = numel(S_info.trajectories.V1.Time_true);
    s = sprintf(' = %d steps', max(2, round(N_total * data_ratio)));
    clear S_info
catch
end
end

function T = read_parts(jobs)
rows = table();
for j = 1:numel(jobs)
    f = jobs(j).csv;
    if ~exist(f, 'file'), continue; end
    try
        t = readtable(f, 'VariableNamingRule', 'preserve', 'TextType', 'string');
        rows = [rows; t]; %#ok<AGROW>
    catch
    end
end
T = rows;
end

function kill_workers_by_parts(part_dir)
%KILL_WORKERS_BY_PARTS 按"命令行里带本实验 _parts 路径"精确结束后台 worker
%   只匹配 iter_compare_main 自己拉起的子进程，不影响用户已打开的 MATLAB 会话
cmd = sprintf(['powershell -NoProfile -Command "Get-CimInstance Win32_Process | ' ...
               'Where-Object { $_.Name -like ''matlab*'' -and $_.CommandLine -like ''*%s*'' } | ' ...
               'ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"'], part_dir);
try
    system(cmd);
catch
end
end
