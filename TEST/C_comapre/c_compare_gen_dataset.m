function out_file = c_compare_gen_dataset(Vehicle_num, Anchor_num, out_file, seed)
% 生成 C_compare 实验用的数据集。
%
% 与 Data\generate_data_3D.m 完全同源（同一个 env_setup、同一套噪声参数，
% IMU 100Hz / UWB 10Hz / 100s），只是把车辆数、基站数、保存路径参数化，
% 并去掉了绘图，方便批量生成。
%
%   c_compare_gen_dataset()                                  % 4 车 30 基站
%   c_compare_gen_dataset(6, 35, 'E:\...\Trj_Veh6_Anc35_3D_10.mat', 20260934)
%
% 默认保存到 E:\DMLKF_code\Data\C_compare\Trj_Veh{V}_Anc{A}_3D_1.mat

if nargin < 1 || isempty(Vehicle_num), Vehicle_num = 4;  end
if nargin < 2 || isempty(Anchor_num),  Anchor_num  = 30; end
if nargin < 4, seed = []; end

root_dir = 'E:\DMLKF_code';
addpath(fullfile(root_dir, 'Data'));

if nargin < 3 || isempty(out_file)
    out_file = fullfile(root_dir, 'Data', 'C_compare', ...
                        sprintf('Trj_Veh%d_Anc%d_3D_1.mat', Vehicle_num, Anchor_num));
end
out_dir = fileparts(out_file);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% ---------------- 全局参数（与 generate_data_3D.m 一致） ----------------
F_imu  = 100;
F_uwb  = 10;
dt_imu = 1 / F_imu;
dt_uwb = 1 / F_uwb;
uwb_downsample_factor = round(dt_uwb / dt_imu);
if mod(dt_uwb, dt_imu) ~= 0
    error('UWB 采样周期必须是 IMU 采样周期的整数倍！');
end
t_end   = 100;
N_steps = round(t_end / dt_imu) + 1;

% ---------------- 噪声参数（与 generate_data_3D.m 完全一致） ----------------
IMU_noise_params.sigma_na = 0.05;
IMU_noise_params.sigma_nw = 0.005;
IMU_noise_params.sigma_ba = 0.005;
IMU_noise_params.sigma_bw = 0.0005;
UWB_noise_params.sigma_anc = 0.1;
UWB_noise_params.sigma_rel = 0.1;

if ~isempty(seed), rng(seed); end

fprintf('生成数据集: Veh=%d Anc=%d N_steps=%d seed=%s -> %s\n', ...
        Vehicle_num, Anchor_num, N_steps, mat2str(seed), out_file);

% ---------------- 真值轨迹与基站布局 ----------------
[trajectories, anchors] = env_setup(Vehicle_num, Anchor_num, N_steps, dt_imu, t_end);

% ---------------- 100Hz IMU（零偏游走 + 白噪声） ----------------
for n = 1:Vehicle_num
    v_name = sprintf('V%d', n);
    veh = trajectories.(v_name);

    theta_unwrapped = unwrap(veh.Theta_true);
    wz_true         = gradient(theta_unwrapped, dt_imu);
    omega_body_ideal = [zeros(N_steps, 2), wz_true];

    a_body_ideal = zeros(N_steps, 3);
    g_vec = [0; 0; -9.81];
    for k = 1:N_steps
        R_k    = veh.R_true(:, :, k);
        a_world = [veh.A_true(k, 1); veh.A_true(k, 2); veh.A_true(k, 3)];
        a_body_ideal(k, :) = (R_k' * (a_world - g_vec))';
    end

    ba_init = (rand(1, 3) - 0.5) * 0.1;
    bw_init = (rand(1, 3) - 0.5) * 0.01;
    b_a = ba_init + cumsum(randn(N_steps, 3) * IMU_noise_params.sigma_ba * sqrt(dt_imu), 1);
    b_w = bw_init + cumsum(randn(N_steps, 3) * IMU_noise_params.sigma_bw * sqrt(dt_imu), 1);

    acc_noise  = randn(N_steps, 3) * IMU_noise_params.sigma_na;
    gyro_noise = randn(N_steps, 3) * IMU_noise_params.sigma_nw;

    trajectories.(v_name).IMU_Time        = veh.Time_true;
    trajectories.(v_name).IMU_acc_m       = a_body_ideal + b_a + acc_noise;
    trajectories.(v_name).IMU_gyro_m      = omega_body_ideal + b_w + gyro_noise;
    trajectories.(v_name).IMU_bias_a_true = b_a;
    trajectories.(v_name).IMU_bias_w_true = b_w;
end

% ---------------- 10Hz UWB（基站测距 + 车-车相对测距） ----------------
idx_uwb = 1:uwb_downsample_factor:N_steps;
t_uwb   = trajectories.V1.Time_true(idx_uwb);
N_uwb   = length(t_uwb);

pos_true_uwb = zeros(N_uwb, 3, Vehicle_num);
for n = 1:Vehicle_num
    v_name = sprintf('V%d', n);
    pos_true_uwb(:, 1, n) = trajectories.(v_name).X_true(idx_uwb);
    pos_true_uwb(:, 2, n) = trajectories.(v_name).Y_true(idx_uwb);
    pos_true_uwb(:, 3, n) = trajectories.(v_name).Z_true(idx_uwb);
end

for n = 1:Vehicle_num
    v_name = sprintf('V%d', n);

    UWB_Anchor = zeros(N_uwb, 1 + Anchor_num);
    UWB_Anchor(:, 1) = t_uwb;
    for a_idx = 1:Anchor_num
        dx = pos_true_uwb(:, 1, n) - anchors(a_idx, 1);
        dy = pos_true_uwb(:, 2, n) - anchors(a_idx, 2);
        dz = pos_true_uwb(:, 3, n) - anchors(a_idx, 3);
        UWB_Anchor(:, 1 + a_idx) = sqrt(dx.^2 + dy.^2 + dz.^2) + randn(N_uwb, 1) * UWB_noise_params.sigma_anc;
    end
    trajectories.(v_name).UWB_Anchor = UWB_Anchor;

    UWB_Relative = zeros(N_uwb, 1 + Vehicle_num);
    UWB_Relative(:, 1) = t_uwb;
    for j = 1:Vehicle_num
        if n == j
            UWB_Relative(:, 1 + j) = NaN;
        else
            dx = pos_true_uwb(:, 1, n) - pos_true_uwb(:, 1, j);
            dy = pos_true_uwb(:, 2, n) - pos_true_uwb(:, 2, j);
            dz = pos_true_uwb(:, 3, n) - pos_true_uwb(:, 3, j);
            UWB_Relative(:, 1 + j) = sqrt(dx.^2 + dy.^2 + dz.^2) + randn(N_uwb, 1) * UWB_noise_params.sigma_rel;
        end
    end
    trajectories.(v_name).UWB_Relative = UWB_Relative;
end

save(out_file, 'trajectories', 'anchors', 'IMU_noise_params', 'UWB_noise_params', ...
     'Vehicle_num', 'Anchor_num');
fprintf('已保存: %s (%.1f MB)\n', out_file, dir(out_file).bytes/1e6);
end
