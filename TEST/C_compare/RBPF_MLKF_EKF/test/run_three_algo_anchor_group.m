function out = run_three_algo_anchor_group(anchor_nums, data_file, ...
                                           bias_comp_ratio, data_ratio, ...
                                           noise_cfg, project_root)
% 在同一个 worker 中运行一组基站数下的 CRBPF、CMLKF 和 CEKF。

addpath(fullfile(project_root, 'EKF',  'CEKF'));
addpath(fullfile(project_root, 'MLKF', 'CMLKF'));
addpath(fullfile(project_root, 'RBPF', 'CRBPF'));

S = load(data_file, 'trajectories', 'anchors', 'Vehicle_num', 'Anchor_num');
traj = S.trajectories;
V = double(S.Vehicle_num);
A_data = double(S.Anchor_num);

if any(anchor_nums > A_data)
    error('请求的基站数超过数据集上限 %d。', A_data);
end

N_steps_total = numel(traj.V1.Time_true);
N_steps = max(2, round(N_steps_total * data_ratio));
dt_imu = traj.V1.Time_true(2) - traj.V1.Time_true(1);
vehicle_names = arrayfun(@(i) sprintf('V%d', i), 1:V, ...
                         'UniformOutput', false);

% 真值初始化和位置真值只计算一次，供本组的全部实验复用。
p0 = zeros(3*V, 1);
v0 = zeros(3*V, 1);
R0 = zeros(3, 3, V);
true_p = zeros(N_steps, 3, V);
for i = 1:V
    name = vehicle_names{i};
    obs = traj.(name);
    p0(3*i-2 : 3*i) = [obs.X_true(1); obs.Y_true(1); obs.Z_true(1)];
    v0(3*i-2 : 3*i) = [obs.Vx_true(1); obs.Vy_true(1); obs.Vz_true(1)];
    R0(:, :, i) = obs.R_true(:, :, 1);
    true_p(:, :, i) = [obs.X_true(1:N_steps), ...
                       obs.Y_true(1:N_steps), ...
                       obs.Z_true(1:N_steps)];
end

uwb_time = traj.V1.UWB_Anchor(:, 1);
alg_names = {'CRBPF', 'CMLKF', 'CEKF'};
rmse_by_anchor = zeros(numel(anchor_nums), numel(alg_names));

for ai = 1:numel(anchor_nums)
    anchor_num = double(anchor_nums(ai));
    anchors = S.anchors(1:anchor_num, :);
    rmse_one_anchor = zeros(1, numel(alg_names));

    for alg_idx = 1:numel(alg_names)
        alg_name = alg_names{alg_idx};

        switch alg_name
            case 'CRBPF'
                kf = CRBPF(V, anchor_num, anchors, dt_imu, p0, v0, R0);
            case 'CMLKF'
                kf = CMLKF(V, anchor_num, anchors, dt_imu, p0, v0, R0);
            case 'CEKF'
                kf = CEKF(V, anchor_num, anchors, dt_imu, p0, v0, R0);
        end

        % 四个噪声参数由运行脚本提供，其他滤波参数保持类默认值。
        np = noise_cfg.(alg_name);
        kf.IMU_Sigma_a   = np.IMU_Sigma_a;
        kf.IMU_Sigma_w   = np.IMU_Sigma_w;
        kf.UWB_sigma_anc = np.UWB_sigma_anc;
        kf.UWB_sigma_rel = np.UWB_sigma_rel;

        if strcmp(alg_name, 'CRBPF')
            % CRBPF 构造时缓存了 Q_global；按其原有公式同步外部 IMU 噪声。
            tau = kf.dt_imu;
            Q_blk = [(tau^4)/4 * kf.IMU_Sigma_a, (tau^3)/2 * kf.IMU_Sigma_a; ...
                     (tau^3)/2 * kf.IMU_Sigma_a,     tau^2   * kf.IMU_Sigma_a];
            Q_cell = repmat({Q_blk}, 1, kf.Vehicle_num);
            kf.Q_global = blkdiag(Q_cell{:});

            % 单次实验内固定随机流，使结果可复现；粒子随机性仍然保留。
            rng(20261026, 'threefry');
        end

        est_p = zeros(N_steps, 3, V);
        for i = 1:V
            est_p(1, :, i) = p0(3*i-2 : 3*i)';
        end

        uwb_idx = 2;
        for k = 2:N_steps
            acc_m  = zeros(3, V);
            gyro_m = zeros(3, V);
            for i = 1:V
                obs = traj.(vehicle_names{i});
                acc_m(:, i) = obs.IMU_acc_m(k-1, :)' ...
                    - bias_comp_ratio * obs.IMU_bias_a_true(k-1, :)';
                gyro_m(:, i) = obs.IMU_gyro_m(k-1, :)' ...
                    - bias_comp_ratio * obs.IMU_bias_w_true(k-1, :)';
            end
            kf.predict(acc_m, gyro_m);

            curr_time = traj.V1.Time_true(k);
            if uwb_idx <= numel(uwb_time) && ...
                    abs(curr_time - uwb_time(uwb_idx)) < 1e-5
                uwb_anc = zeros(V, anchor_num);
                uwb_rel = zeros(V, V);
                for i = 1:V
                    obs = traj.(vehicle_names{i});
                    uwb_anc(i, :) = obs.UWB_Anchor(uwb_idx, 2:anchor_num+1);
                    uwb_rel(i, :) = obs.UWB_Relative(uwb_idx, 2:end);
                end
                kf.update(uwb_anc, uwb_rel);
                uwb_idx = uwb_idx + 1;
            end

            for i = 1:V
                est_p(k, :, i) = kf.p(3*i-2 : 3*i)';
            end
        end

        vehicle_rmse = zeros(V, 1);
        for i = 1:V
            pos_error = est_p(:, :, i) - true_p(:, :, i);
            vehicle_rmse(i) = sqrt(mean(sum(pos_error.^2, 2)));
        end
        rmse_one_anchor(alg_idx) = mean(vehicle_rmse);
    end

    rmse_by_anchor(ai, :) = rmse_one_anchor;
    fprintf('Anc %2d | CRBPF %.4f | CMLKF %.4f | CEKF %.4f\n', ...
            anchor_num, rmse_one_anchor(1), rmse_one_anchor(2), ...
            rmse_one_anchor(3));
end

out.anchor_nums = double(anchor_nums(:));
out.rmse = rmse_by_anchor';  % 3 x 组内基站数，按 CRBPF、CMLKF、CEKF
end
