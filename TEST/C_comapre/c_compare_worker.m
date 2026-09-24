function c_compare_worker(a_from, a_to, step_cap, out_dir, use_rel, data_file)
% 并行分段的执行体：对基站数 a = a_from : a_to 逐个执行
%
%   调用 CRBPF / CMLKF / CEKF 三个算法（纯调用，不修改算法任何参数），
%   bias_comp_ratio = 0.6，data_ratio = 1.0（数据集 100% 使用），
%   6 车辆，数据始终来自同一个 35 基站数据集，
%   基站数 a 表示只使用该数据集中的前 a 个基站观测。
%
% 结果逐行追加写入 out_dir\seg_<a_from>_<a_to>.csv，
% 全部完成后写出 out_dir\seg_<a_from>_<a_to>.done。
%
% 用法：
%   c_compare_worker(4, 8)                     % 基站 4..8，全量数据
%   c_compare_worker(4, 8, 300)                % 只跑前 300 步（冒烟测试）
%   c_compare_worker(4, 4, [], dir, false)     % 诊断：不用车-车相对测距
%   c_compare_worker(4, 8, [], dir, true, mat) % 指定数据集（默认 6 车 35 基站）
%
% 结果 CSV 只保存位置 RMSE（速度/姿态只在日志里打印，用于分析）。

if nargin < 1 || isempty(a_from),   a_from   = 4;    end
if nargin < 2 || isempty(a_to),     a_to     = 8;    end
if nargin < 3 || isempty(step_cap), step_cap = Inf;  end
if nargin < 4 || isempty(out_dir)
    out_dir = fullfile(fileparts(mfilename('fullpath')), 'RESULT');
end
if nargin < 5 || isempty(use_rel),  use_rel  = true; end
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% ---------------- 实验配置 ----------------
bias_comp_ratio = 0.6;      % 零偏扣除比例（0.6 = 保留 40% 残留零偏）
data_ratio      = 1.0;      % 数据集使用比例
root_dir        = 'E:\DMLKF_code';
if nargin < 6 || isempty(data_file)
    data_file = fullfile(root_dir, 'Data', 'C_compare', 'Trj_Veh6_Anc35_3D_10.mat');
end
algorithms      = {'CRBPF', 'CMLKF', 'CEKF'};

this_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(root_dir, 'Data'));
addpath(fullfile(root_dir, 'RBPF', 'CRBPF'));
addpath(fullfile(root_dir, 'MLKF', 'CMLKF'));
addpath(fullfile(root_dir, 'EKF',  'CEKF'));
addpath(this_dir);

% ---------------- 载入唯一数据集 ----------------
S = load(data_file, 'trajectories', 'anchors', 'Vehicle_num', 'Anchor_num');
traj        = S.trajectories;
anchors_all = S.anchors;
V           = S.Vehicle_num;
A_total     = S.Anchor_num;
if a_to > A_total
    error('数据集只有 %d 个基站，无法请求到 %d 个', A_total, a_to);
end

N_steps_total = numel(traj.V1.Time_true);
N_steps = max(2, round(N_steps_total * data_ratio));
if isfinite(step_cap)
    N_steps = min(N_steps, max(2, round(step_cap)));
end
dt_imu = traj.V1.Time_true(2) - traj.V1.Time_true(1);

% ---------------- 真值初始化（与三个 Test 脚本一致） ----------------
p0 = zeros(3*V, 1);
v0 = zeros(3*V, 1);
R0 = zeros(3, 3, V);
for i = 1:V
    v_name = sprintf('V%d', i);
    p0(3*i-2 : 3*i) = [traj.(v_name).X_true(1);  traj.(v_name).Y_true(1);  traj.(v_name).Z_true(1)];
    v0(3*i-2 : 3*i) = [traj.(v_name).Vx_true(1); traj.(v_name).Vy_true(1); traj.(v_name).Vz_true(1)];
    R0(:, :, i)     = traj.(v_name).R_true(:, :, 1);
end

seg_tag  = sprintf('seg_%d_%d', a_from, a_to);
csv_file = fullfile(out_dir, [seg_tag '.csv']);
log_note = @(fmt, varargin) fprintf([sprintf('[%s] ', seg_tag) fmt], varargin{:});

log_note('start  anchors %d..%d  Veh=%d  steps=%d/%d  bias_comp=%.2f  data=%.0f%%  rel_ranging=%d\n', ...
         a_from, a_to, V, N_steps, N_steps_total, bias_comp_ratio, data_ratio*100, use_rel);

if ~exist(csv_file, 'file')
    fid = fopen(csv_file, 'w');
    fprintf(fid, 'anchor_num,algorithm,rmse_p,sec,ok,note\n');
    fclose(fid);
end

% ---------------- 主循环：基站数 x 算法 ----------------
for a = a_from:a_to
    anchors = anchors_all(1:a, :);          % 只取前 a 个基站

    for c = 1:numel(algorithms)
        alg  = algorithms{c};
        note = '';
        rp = NaN; rv = NaN; ra = NaN; sec = NaN; ok = 0;

        t_start = tic;
        try
            kf = feval(alg, V, a, anchors, dt_imu, p0, v0, R0);
            [rp, rv, ra, sec] = c_compare_loop(kf, traj, V, a, N_steps, bias_comp_ratio, use_rel);
            ok = 1;
            clear kf
        catch err
            sec  = toc(t_start);
            note = strrep(sprintf('%s: %s', err.identifier, err.message), ',', ';');
            note = strrep(note, '%', 'pct');      % 避免 %% 在 fprintf 里被当作格式符
            log_note('a=%d %-5s FAILED: %s\n', a, alg, note);
        end

        fid = fopen(csv_file, 'a');
        fprintf(fid, '%d,%s,%.6f,%.1f,%d,%s\n', a, alg, rp, sec, ok, note);
        fclose(fid);

        if ok
            log_note('a=%2d %-5s pos=%.4f vel=%.4f att=%.4f  (%.1f s)\n', a, alg, rp, rv, ra, sec);
        end
        drawnow;   % 让日志及时刷出
    end
end

fid = fopen(fullfile(out_dir, [seg_tag '.done']), 'w');
fprintf(fid, 'finished %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fclose(fid);
log_note('DONE\n');
end
