function [rmse_p, rmse_v, rmse_att, sec] = c_compare_loop(kf, traj, V, A, N_steps, bias_comp_ratio, use_rel)
% 三个算法共用的滤波主循环 + RMSE 计算。
%
% 与 CRBPF_Test / CMLKF_Test / CEKF_Test 中的循环完全一致，只有一点不同：
%   基站观测只取前 A 个 (anchors 的 1:A 行, uwb_anc 的 1:A 列)，
%   车辆之间的相对测距 V x V 保持全量使用。
%
% 三个滤波器暴露的接口相同：
%   kf.predict(acc_m, gyro_m) / kf.update(uwb_anc, uwb_rel) / kf.p / kf.v / kf.R
%
% 本函数只做“调用”，不读写任何算法内部参数。
%
% use_rel (可选，默认 true)：是否使用车-车相对测距。
%   置为 false 时相对测距全部传 NaN（三个算法都会跳过 NaN 观测），
%   用于诊断“观测真的稀缺”时的算法差距，不影响正常实验。

if nargin < 7 || isempty(use_rel), use_rel = true; end

vn = cell(1, V);
for i = 1:V
    vn{i} = sprintf('V%d', i);
end

t0 = tic;

est_p = zeros(N_steps, 3, V);
est_v = zeros(N_steps, 3, V);
est_R = zeros(3, 3, V, N_steps);
for i = 1:V
    est_p(1, :, i) = [traj.(vn{i}).X_true(1), traj.(vn{i}).Y_true(1), traj.(vn{i}).Z_true(1)];
    est_v(1, :, i) = [traj.(vn{i}).Vx_true(1), traj.(vn{i}).Vy_true(1), traj.(vn{i}).Vz_true(1)];
    est_R(:, :, i, 1) = traj.(vn{i}).R_true(:, :, 1);
end

uwb_idx = 2;                                  % UWB 从第 2 个历元开始融合
UWB_Time_Vec = traj.V1.UWB_Anchor(:, 1);

for k = 2:N_steps
    % --- A. 100 Hz 预测：只扣除 bias_comp_ratio 比例的零偏 ---
    acc_m  = zeros(3, V);
    gyro_m = zeros(3, V);
    for i = 1:V
        acc_m(:, i)  = traj.(vn{i}).IMU_acc_m(k-1, :)'  - bias_comp_ratio * traj.(vn{i}).IMU_bias_a_true(k-1, :)';
        gyro_m(:, i) = traj.(vn{i}).IMU_gyro_m(k-1, :)' - bias_comp_ratio * traj.(vn{i}).IMU_bias_w_true(k-1, :)';
    end
    kf.predict(acc_m, gyro_m);

    % --- B. 10 Hz 更新：前 A 个基站的测距 + 全部车-车相对测距 ---
    curr_time = traj.V1.Time_true(k);
    if uwb_idx <= length(UWB_Time_Vec) && abs(curr_time - UWB_Time_Vec(uwb_idx)) < 1e-5
        uwb_anc_raw = zeros(V, A);
        for i = 1:V
            uwb_anc_raw(i, :) = traj.(vn{i}).UWB_Anchor(uwb_idx, 2:1+A);
        end
        if use_rel
            uwb_rel_raw = zeros(V, V);
            for i = 1:V
                uwb_rel_raw(i, :) = traj.(vn{i}).UWB_Relative(uwb_idx, 2:end);
            end
        else
            uwb_rel_raw = nan(V, V);      % 诊断模式：不使用车-车相对测距
        end
        kf.update(uwb_anc_raw, uwb_rel_raw);
        uwb_idx = uwb_idx + 1;
    end

    % --- C. 保存当前步估计 ---
    for i = 1:V
        est_p(k, :, i) = kf.p(3*i-2 : 3*i)';
        est_v(k, :, i) = kf.v(3*i-2 : 3*i)';
        est_R(:, :, i, k) = kf.R(:, :, i);
    end
end
sec = toc(t0);

% --- RMSE (与三个 Test 脚本口径一致：先按车算，再对车取平均) ---
rmse_p   = zeros(V, 1);
rmse_v   = zeros(V, 1);
rmse_att = zeros(V, 1);
for i = 1:V
    true_p = [traj.(vn{i}).X_true(1:N_steps),  traj.(vn{i}).Y_true(1:N_steps),  traj.(vn{i}).Z_true(1:N_steps)];
    true_v = [traj.(vn{i}).Vx_true(1:N_steps), traj.(vn{i}).Vy_true(1:N_steps), traj.(vn{i}).Vz_true(1:N_steps)];

    rmse_p(i) = sqrt(mean(sum((est_p(:, :, i) - true_p).^2, 2)));
    rmse_v(i) = sqrt(mean(sum((est_v(:, :, i) - true_v).^2, 2)));

    err_att_seq = zeros(N_steps, 1);
    for k = 1:N_steps
        R_err = traj.(vn{i}).R_true(:, :, k)' * est_R(:, :, i, k);
        tr = max(-1, min(3, trace(R_err)));
        err_att_seq(k) = acos((tr - 1) / 2) * (180 / pi);
    end
    rmse_att(i) = sqrt(mean(err_att_seq.^2));
end

rmse_p   = mean(rmse_p);
rmse_v   = mean(rmse_v);
rmse_att = mean(rmse_att);
end
