function Newton_iter_worker(mode, iters, out_file, cfg_file)
% Run one V1 job or one DMLKF max_iter group.

S = load(cfg_file, 'cfg');
cfg = S.cfg;
add_paths(cfg.data_file);

fid = fopen(out_file, 'w');
if fid < 0
    error('无法创建结果文件: %s', out_file);
end
fprintf(fid, 'algorithm,max_iter,pos_rmse_m,seconds,ok,note\n');

try
    D = load_case(cfg.data_file, cfg.Vehicle_num, cfg.Anchor_num, ...
        cfg.data_ratio, cfg.bias_comp_ratio);
    V2V_Mask = sym_k_mask(D.V, cfg.neighbor_k);

    mode = upper(char(mode));
    if ~ismember(mode, {'V1', 'DMLKF'})
        error('未知算法模式: %s', mode);
    end

    for it = iters(:)'
        try
            [rmse_p, elapsed] = run_filter(mode, it, D, V2V_Mask, cfg);
            fprintf(fid, '%s,%d,%.12g,%.3f,1,\n', ...
                mode, it, rmse_p, elapsed);
        catch err
            note = clean_note(err);
            fprintf(fid, '%s,%d,NaN,NaN,0,%s\n', mode, it, note);
        end
    end
catch err
    note = clean_note(err);
    it = NaN;
    if ~isempty(iters), it = iters(1); end
    fprintf(fid, '%s,%d,NaN,NaN,0,%s\n', upper(char(mode)), it, note);
end

fclose(fid);
done_fid = fopen([out_file '.done'], 'w');
if done_fid >= 0
    fprintf(done_fid, 'done\n');
    fclose(done_fid);
end
end

function D = load_case(data_file, V_expected, A_expected, data_ratio, bias_ratio)
S = load(data_file, 'trajectories', 'anchors', ...
    'Vehicle_num', 'Anchor_num');
if S.Vehicle_num ~= V_expected || S.Anchor_num ~= A_expected
    error('数据集车辆数或基站数与配置不一致。');
end

D = struct();
D.traj = S.trajectories;
D.anchors = S.anchors;
D.V = S.Vehicle_num;
D.A = S.Anchor_num;
D.bias_comp_ratio = bias_ratio;

N_total = numel(D.traj.V1.Time_true);
D.N_steps = max(2, round(N_total * data_ratio));
D.N_steps = min(D.N_steps, N_total);
D.dt = D.traj.V1.Time_true(2) - D.traj.V1.Time_true(1);

p0 = zeros(3 * D.V, 1);
v0 = zeros(3 * D.V, 1);
R0 = zeros(3, 3, D.V);
D.true_p = zeros(D.N_steps, 3, D.V);
for i = 1:D.V
    name = sprintf('V%d', i);
    p0(3*i-2 : 3*i) = [D.traj.(name).X_true(1); ...
        D.traj.(name).Y_true(1); D.traj.(name).Z_true(1)];
    v0(3*i-2 : 3*i) = [D.traj.(name).Vx_true(1); ...
        D.traj.(name).Vy_true(1); D.traj.(name).Vz_true(1)];
    R0(:, :, i) = D.traj.(name).R_true(:, :, 1);
    D.true_p(:, :, i) = [D.traj.(name).X_true(1:D.N_steps), ...
        D.traj.(name).Y_true(1:D.N_steps), ...
        D.traj.(name).Z_true(1:D.N_steps)];
end

D.p0 = p0;
D.v0 = v0;
D.R0 = R0;
D.UWB_Time_Vec = D.traj.V1.UWB_Anchor(:, 1);
end

function [rmse_p, elapsed] = run_filter(mode, max_iter, D, V2V_Mask, cfg)
switch mode
    case 'V1'
        kf = DMLKF_V1(D.V, D.A, D.anchors, D.dt, ...
            D.p0, D.v0, D.R0);
        kf.max_iter = max_iter;
        kf.beta_inv = cfg.V1_beta_inv;
        kf.max_step = cfg.V1_max_step;
    case 'DMLKF'
        kf = DMLKF(D.V, D.A, D.anchors, D.dt, ...
            D.p0, D.v0, D.R0, V2V_Mask, max_iter);
        kf.beta_inv = cfg.DMLKF_beta_inv;
        kf.max_step = cfg.DMLKF_max_step;
        kf.ALPHA_SAFETY = cfg.DMLKF_ALPHA_SAFETY;
        kf.alpha_const = min(1.0, max(0.01, ...
            kf.ALPHA_SAFETY * (1 - kf.lambda_2) / ...
            (1 + sqrt(kf.lambda_2))));
        kf.epsilon = cfg.DMLKF_epsilon;
        kf.print_flag = 0;
end

est_p = zeros(D.N_steps, 3, D.V);
for i = 1:D.V
    est_p(1, :, i) = D.p0(3*i-2 : 3*i)';
end

uwb_idx = 2;
timer_id = tic;
for k = 2:D.N_steps
    acc_m = zeros(3, D.V);
    gyro_m = zeros(3, D.V);
    for i = 1:D.V
        name = sprintf('V%d', i);
        acc_m(:, i) = D.traj.(name).IMU_acc_m(k-1, :)' - ...
            D.bias_comp_ratio * D.traj.(name).IMU_bias_a_true(k-1, :)';
        gyro_m(:, i) = D.traj.(name).IMU_gyro_m(k-1, :)' - ...
            D.bias_comp_ratio * D.traj.(name).IMU_bias_w_true(k-1, :)';
    end
    kf.predict(acc_m, gyro_m);

    current_time = D.traj.V1.Time_true(k);
    if uwb_idx <= numel(D.UWB_Time_Vec) && ...
            abs(current_time - D.UWB_Time_Vec(uwb_idx)) < 1e-5
        uwb_anc = zeros(D.V, D.A);
        uwb_rel = zeros(D.V, D.V);
        for i = 1:D.V
            name = sprintf('V%d', i);
            uwb_anc(i, :) = D.traj.(name).UWB_Anchor(uwb_idx, 2:end);
            uwb_rel(i, :) = D.traj.(name).UWB_Relative(uwb_idx, 2:end);
        end
        uwb_rel(V2V_Mask == 0) = NaN;
        kf.update(uwb_anc, uwb_rel);
        uwb_idx = uwb_idx + 1;
    end

    for i = 1:D.V
        est_p(k, :, i) = kf.Nodes{i}.p';
    end
end
elapsed = toc(timer_id);

rmse_each = zeros(D.V, 1);
for i = 1:D.V
    error_vector = est_p(:, :, i) - D.true_p(:, :, i);
    rmse_each(i) = sqrt(mean(sum(error_vector.^2, 2)));
end
rmse_p = mean(rmse_each);
end

function M = sym_k_mask(V, K)
K = min(K, V - 1);
M = zeros(V, V);
K_half = floor(K / 2);
for i = 1:V
    for d = 1:K_half
        M(i, mod(i + d - 1, V) + 1) = 1;
        M(i, mod(i - d - 1, V) + 1) = 1;
    end
    if mod(K, 2) ~= 0
        M(i, mod(i + V/2 - 1, V) + 1) = 1;
    end
end
M(logical(eye(V))) = 0;
if ~isequal(M, M')
    error('V2V mask 不是对称矩阵。');
end
end

function paths = add_paths(data_file)
test_dir = fileparts(mfilename('fullpath'));
old_test_dir = fileparts(fileparts(test_dir));
project_root = fileparts(old_test_dir);
paths = {project_root, fullfile(project_root, 'MLKF', 'DMLKF')};
addpath(paths{:});
if ~isfile(data_file)
    error('找不到数据集: %s', data_file);
end
end

function note = clean_note(err)
note = sprintf('%s: %s', err.identifier, err.message);
note = strrep(note, ',', ';');
note = strrep(note, newline, ' ');
end
