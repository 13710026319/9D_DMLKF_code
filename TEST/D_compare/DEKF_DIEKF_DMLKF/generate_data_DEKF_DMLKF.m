% generate_data_DEKF_DMLKF.m
% Standalone data generator for DEKF/DIEKF/DMLKF parameter studies.
%
% This file integrates Data/generate_data_3D.m and Data/env_setup.m.
% The trajectory follows Data/env_setup.m exactly, and the anchor pool is
% extended from 10 to 15 points. There is no runtime dependency on Data/.

clc; clear; close all;

%% ==================== 1. Global configuration ====================
F_imu = 100;
F_uwb = 10;
dt_imu = 1 / F_imu;
dt_uwb = 1 / F_uwb;
uwb_downsample_factor = round(dt_uwb / dt_imu);

if abs(dt_uwb / dt_imu - uwb_downsample_factor) > 1e-12
    error('The UWB period must be an integer multiple of the IMU period.');
end

Vehicle_num = 8;
Anchor_num  = 4;
t_end       = 100;
N_steps     = round(t_end / dt_imu) + 1;
repeat_time = 1;
N_steps_total = N_steps * repeat_time;

% The same dataset_id intentionally overwrites the same MAT file.
% Randomization is based on rng('shuffle'), never on this identifier.
dataset_id = 'DEKF_DIEKF_DMLKF_100_1';

if repeat_time < 1 || round(repeat_time) ~= repeat_time
    error('repeat_time must be a positive integer.');
end

%% ==================== 2. Random seed and noise configuration ====================
% Every execution creates a new random seed, even when dataset_id is unchanged.
rng('shuffle');
rng_state = rng;
base_seed = double(rng_state.Seed);

UWB_noise_params.sigma_anc = 0.1;
UWB_noise_params.sigma_rel = 0.1;

IMU_noise_params.sigma_na = 0.05;
IMU_noise_params.sigma_nw = 0.005;
IMU_noise_params.sigma_ba = 0.005;
IMU_noise_params.sigma_bw = 0.0005;

%% ==================== 3. Embedded scene geometry ====================
test_dir = fileparts(mfilename('fullpath'));
[trajectory_template, anchors] = env_setup_embedded( ...
    Vehicle_num, Anchor_num, N_steps, dt_imu, t_end);

if size(anchors, 1) ~= Anchor_num
    error('Embedded scene returned %d anchors, but Anchor_num is %d.', ...
        size(anchors, 1), Anchor_num);
end

truth_position = zeros(N_steps, 3, Vehicle_num);
for n = 1:Vehicle_num
    v_name = sprintf('V%d', n);
    truth_position(:, 1, n) = trajectory_template.(v_name).X_true;
    truth_position(:, 2, n) = trajectory_template.(v_name).Y_true;
    truth_position(:, 3, n) = trajectory_template.(v_name).Z_true;
end

%% ==================== 4. Sensor generation ====================
trajectories = struct();
batch_seeds = base_seed + (0:repeat_time - 1);

for r = 1:repeat_time
    rng(batch_seeds(r), 'twister');
    batch_trajectories = generate_imu_uwb_batch( ...
        trajectory_template, truth_position, anchors, ...
        N_steps, dt_imu, uwb_downsample_factor, ...
        IMU_noise_params, UWB_noise_params);

    if r == 1
        trajectories = batch_trajectories;
    else
        trajectories = concatenate_trajectories( ...
            trajectories, batch_trajectories);
    end
end

%% ==================== 5. Visualization ====================
figure('Name', 'Data scene DEKF/DIEKF/DMLKF test data', ...
    'Position', [100, 100, 900, 700]);
hold on; grid on; axis equal;
xlabel('X position (m)');
ylabel('Y position (m)');
zlabel('Z position (m)');
title(sprintf('%d vehicles, %d anchors: embedded Data-style geometry', ...
    Vehicle_num, Anchor_num));

X_max = 120;
Y_max = 120;
Z_max = 15;
line([0, X_max, X_max, 0, 0], [0, 0, Y_max, Y_max, 0], ...
    [0, 0, 0, 0, 0], 'Color', [0.5, 0.5, 0.5], 'LineStyle', '--');
line([0, X_max, X_max, 0, 0], [0, 0, Y_max, Y_max, 0], ...
    [Z_max, Z_max, Z_max, Z_max, Z_max], ...
    'Color', [0.5, 0.5, 0.5], 'LineStyle', '--');
for corner = [0, X_max]
    for side = [0, Y_max]
        line([corner, corner], [side, side], [0, Z_max], ...
            'Color', [0.5, 0.5, 0.5], 'LineStyle', '--');
    end
end

h_anchor = plot3(anchors(:, 1), anchors(:, 2), anchors(:, 3), ...
    '^', 'MarkerSize', 13, 'MarkerFaceColor', 'r', ...
    'MarkerEdgeColor', 'k', 'LineWidth', 1.5, ...
    'DisplayName', 'Anchors');

colors = lines(Vehicle_num);
h_traj = zeros(1, Vehicle_num);
for n = 1:Vehicle_num
    v_name = sprintf('V%d', n);
    v_data = trajectories.(v_name);
    h_traj(n) = plot3(v_data.X_true, v_data.Y_true, v_data.Z_true, ...
        'Color', colors(n, :), 'LineWidth', 2.5, ...
        'DisplayName', sprintf('V%d', n));
    plot3(v_data.X_true(1), v_data.Y_true(1), v_data.Z_true(1), ...
        'o', 'MarkerSize', 8, 'MarkerFaceColor', colors(n, :), ...
        'Color', colors(n, :));
    plot3(v_data.X_true(end), v_data.Y_true(end), v_data.Z_true(end), ...
        '*', 'MarkerSize', 10, 'Color', colors(n, :));
end
view(3);
legend([h_anchor, h_traj], 'Location', 'northeastoutside');
hold off;

%% ==================== 6. Save ====================
save_dir = fullfile(test_dir, 'data');
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

mat_name = sprintf('Trj_Veh%d_Anc%d_%s.mat', ...
    Vehicle_num, Anchor_num, dataset_id);
save_path = fullfile(save_dir, mat_name);

save(save_path, 'trajectories', 'anchors', ...
    'IMU_noise_params', 'UWB_noise_params', ...
    'Vehicle_num', 'Anchor_num', 't_end', ...
    'N_steps', 'N_steps_total', 'repeat_time', ...
    'uwb_downsample_factor', 'batch_seeds');

fprintf('\nData generation completed.\n');
fprintf('Geometry source: embedded env_setup_embedded\n');
fprintf('Vehicles: %d, anchors: %d\n', Vehicle_num, Anchor_num);
fprintf('Anchor Z range: %.3f to %.3f m\n', ...
    min(anchors(:, 3)), max(anchors(:, 3)));
fprintf('Per batch: %.1f s, %d IMU steps, batches: %d\n', ...
    t_end, N_steps, repeat_time);
fprintf('Total IMU steps: %d\n', N_steps_total);
fprintf('Total UWB steps: %d\n', ...
    size(trajectories.V1.UWB_Anchor, 1));
fprintf('Dataset ID: %s\n', dataset_id);
fprintf('Random base seed: %u\n', base_seed);
fprintf('Saved to: %s\n', save_path);

%% ======================= Local functions =======================
function batch = generate_imu_uwb_batch( ...
        trajectory_template, truth_position, anchors, ...
        N_steps, dt_imu, uwb_downsample_factor, ...
        IMU_noise_params, UWB_noise_params)
    Vehicle_num = size(truth_position, 3);
    Anchor_num = size(anchors, 1);
    batch = trajectory_template;

    % 100 Hz IMU measurements.
    for n = 1:Vehicle_num
        v_name = sprintf('V%d', n);
        veh = batch.(v_name);

        theta_unwrapped = unwrap(veh.Theta_true);
        wz_true = gradient(theta_unwrapped, dt_imu);
        omega_body_ideal = [zeros(N_steps, 2), wz_true];

        a_body_ideal = zeros(N_steps, 3);
        g_vec = [0; 0; -9.81];
        for k = 1:N_steps
            R_k = veh.R_true(:, :, k);
            a_world = [veh.A_true(k, 1);
                       veh.A_true(k, 2);
                       veh.A_true(k, 3)];
            a_body_ideal(k, :) = (R_k' * (a_world - g_vec))';
        end

        ba_init = (rand(1, 3) - 0.5) * 0.1;
        bw_init = (rand(1, 3) - 0.5) * 0.01;
        b_a = ba_init + cumsum( ...
            randn(N_steps, 3) * IMU_noise_params.sigma_ba * sqrt(dt_imu), 1);
        b_w = bw_init + cumsum( ...
            randn(N_steps, 3) * IMU_noise_params.sigma_bw * sqrt(dt_imu), 1);

        veh.IMU_Time = veh.Time_true;
        veh.IMU_acc_m = a_body_ideal + b_a + ...
            randn(N_steps, 3) * IMU_noise_params.sigma_na;
        veh.IMU_gyro_m = omega_body_ideal + b_w + ...
            randn(N_steps, 3) * IMU_noise_params.sigma_nw;
        veh.IMU_bias_a_true = b_a;
        veh.IMU_bias_w_true = b_w;
        batch.(v_name) = veh;
    end

    % 10 Hz UWB measurements.
    idx_uwb = 1:uwb_downsample_factor:N_steps;
    t_uwb = trajectory_template.V1.Time_true(idx_uwb);
    N_uwb = length(t_uwb);
    pos_true_uwb = truth_position(idx_uwb, :, :);

    for n = 1:Vehicle_num
        v_name = sprintf('V%d', n);

        UWB_Anchor = zeros(N_uwb, 1 + Anchor_num);
        UWB_Anchor(:, 1) = t_uwb;
        for a_idx = 1:Anchor_num
            delta = squeeze(pos_true_uwb(:, :, n)) - anchors(a_idx, :);
            dist_true = sqrt(sum(delta.^2, 2));
            UWB_Anchor(:, 1 + a_idx) = dist_true + ...
                randn(N_uwb, 1) * UWB_noise_params.sigma_anc;
        end
        batch.(v_name).UWB_Anchor = UWB_Anchor;

        UWB_Relative = zeros(N_uwb, 1 + Vehicle_num);
        UWB_Relative(:, 1) = t_uwb;
        for j = 1:Vehicle_num
            if n == j
                UWB_Relative(:, 1 + j) = NaN;
            else
                delta = squeeze(pos_true_uwb(:, :, n)) - ...
                    squeeze(pos_true_uwb(:, :, j));
                dist_true = sqrt(sum(delta.^2, 2));
                UWB_Relative(:, 1 + j) = dist_true + ...
                    randn(N_uwb, 1) * UWB_noise_params.sigma_rel;
            end
        end
        batch.(v_name).UWB_Relative = UWB_Relative;
    end
end

function total = concatenate_trajectories(total, batch)
    Vehicle_num = numel(fieldnames(total));
    for n = 1:Vehicle_num
        v_name = sprintf('V%d', n);
        fields = fieldnames(batch.(v_name));
        for q = 1:numel(fields)
            field_name = fields{q};
            if strcmp(field_name, 'R_true')
                total.(v_name).(field_name) = cat(3, ...
                    total.(v_name).(field_name), ...
                    batch.(v_name).(field_name));
            else
                total.(v_name).(field_name) = cat(1, ...
                    total.(v_name).(field_name), ...
                    batch.(v_name).(field_name));
            end
        end
    end
end

function [trajectories, anchors] = env_setup_embedded( ...
        Vehicle_num, Anchor_num, N_steps, dt_imu, t_end)
    % Embedded copy of the Data/env_setup.m scene model.
    % The first ten anchors match Data/env_setup.m exactly. Anchors 11-15
    % are the five requested internal expansion points.

    all_anchors_pool = [
          0,   0, 1.0;  % 1: lower-left corner
        120, 120, 1.0;  % 2: upper-right corner
          0, 120, 1.0;  % 3: upper-left corner
        120,   0, 1.0;  % 4: lower-right corner
         60,  60, 1.0;  % 5: center
         60,   0, 1.0;  % 6: bottom midpoint
          0,  60, 1.0;  % 7: left midpoint
        120,  60, 2.0;  % 8: right midpoint
         60, 120, 1.0;  % 9: top midpoint
         30,  90, 2.0;  % 10: upper-left internal offset
         90,  30, 1.5;  % 11: lower-right internal offset
         30,  30, 1.5;  % 12: lower-left internal offset
         90,  90, 1.5;  % 13: upper-right internal offset
         20,  60, 1.5;  % 14: left-middle internal offset
        100,  60, 1.5   % 15: right-middle internal offset
    ];

    if Anchor_num > size(all_anchors_pool, 1) || Anchor_num < 1
        error('Anchor_num must be in the range 1-%d.', ...
            size(all_anchors_pool, 1));
    end
    anchors = all_anchors_pool(1:Anchor_num, :);

    if Vehicle_num > 18 || Vehicle_num < 1
        error('Vehicle_num must be in the range 1-18.');
    end

    v_rate = 2;
    trajectories = struct();

    for n = 1:Vehicle_num
        % Assign each vehicle a unique radial frequency and spatial phase.
        R0 = 20 + 1.5 * n;
        Ar = 5 + 0.2 * n;
        w_r = (0.03 + 0.002 * n) * v_rate;
        phi_r = n * (2*pi/Vehicle_num);

        % Use opposite orbit directions for odd and even vehicles.
        dir_flag = (-1)^n;
        w_a = (dir_flag * (0.015 + 0.001 * n)) * v_rate;
        phi_a = n * (pi/2);

        % Spread vehicles across three nominal altitude layers.
        Zc = 4 + mod(n, 3) * 3.5;
        Az = 2.0;
        w_z = (0.04 + 0.003 * n) * v_rate;
        phi_z = n * (2*pi/Vehicle_num);

        P_true = zeros(N_steps, 3);
        V_true = zeros(N_steps, 3);
        A_true = zeros(N_steps, 3);
        Theta_true = zeros(N_steps, 1);

        for k = 1:N_steps
            t = (k - 1) * dt_imu;

            % Analytic dynamic radius.
            R = R0 + Ar * sin(w_r * t + phi_r);
            R_dot = Ar * w_r * cos(w_r * t + phi_r);
            R_ddot = -Ar * w_r^2 * sin(w_r * t + phi_r);

            % Analytic orbit angle.
            alpha = w_a * t + phi_a;
            alpha_dot = w_a;

            X = 60 + R * cos(alpha);
            Y = 60 + R * sin(alpha);
            Z = Zc + Az * sin(w_z * t + phi_z);

            Vx = R_dot * cos(alpha) - R * alpha_dot * sin(alpha);
            Vy = R_dot * sin(alpha) + R * alpha_dot * cos(alpha);
            Vz = Az * w_z * cos(w_z * t + phi_z);

            Ax = (R_ddot - R * alpha_dot^2) * cos(alpha) ...
                - (2 * R_dot * alpha_dot) * sin(alpha);
            Ay = (R_ddot - R * alpha_dot^2) * sin(alpha) ...
                + (2 * R_dot * alpha_dot) * cos(alpha);
            Az_acc = -Az * w_z^2 * sin(w_z * t + phi_z);

            th_raw = atan2(Vy, Vx);
            if k == 1
                th_curr = th_raw;
            else
                diff = th_raw - Theta_true(k - 1);
                diff = mod(diff + pi, 2*pi) - pi;
                th_curr = Theta_true(k - 1) + diff;
            end

            P_true(k, :) = [X, Y, Z];
            V_true(k, :) = [Vx, Vy, Vz];
            A_true(k, :) = [Ax, Ay, Az_acc];
            Theta_true(k) = th_curr;
        end

        v_name = sprintf('V%d', n);
        trajectories.(v_name).Time_true = (0:dt_imu:t_end)';
        trajectories.(v_name).X_true = P_true(:, 1);
        trajectories.(v_name).Y_true = P_true(:, 2);
        trajectories.(v_name).Z_true = P_true(:, 3);
        trajectories.(v_name).Vx_true = V_true(:, 1);
        trajectories.(v_name).Vy_true = V_true(:, 2);
        trajectories.(v_name).Vz_true = V_true(:, 3);
        trajectories.(v_name).Theta_true = Theta_true;
        trajectories.(v_name).A_true = A_true;

        R_true = zeros(3, 3, N_steps);
        for k = 1:N_steps
            th = Theta_true(k);
            R_true(:, :, k) = [
                cos(th), -sin(th), 0;
                sin(th),  cos(th), 0;
                0,       0,        1
            ];
        end
        trajectories.(v_name).R_true = R_true;
    end
end
