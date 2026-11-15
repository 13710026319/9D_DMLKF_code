classdef CPF < handle
    % CPF - Centralized Particle Filter (集中式粒子滤波)
    % 核心原理：在 9I 维全局联合状态空间 (位置3I、速度3I、姿态SO(3)^I) 直接撒粒子进行蒙特卡洛采样与重采样
    % 严格对齐 Centralized Particle Filter 算法文档，支持渐进式批处理与严重贫化自动监测终止
    
    properties
        Vehicle_num
        Anchor_num
        anchors
        dt_imu
        g_vec
        
        % 传感器噪声参数 (支持外部脚本直接修改，实时生效)
        IMU_Sigma_a
        IMU_Sigma_w
        UWB_sigma_anc
        UWB_sigma_rel
        
        % 粒子滤波核心参数
        Np               % 粒子总数 (支持外部直接赋值修改或调用 set_particle_count)
        neff_ratio       % 重采样阈值比例 (默认 0.5)
        rough_coeff      % 重采样后 Roughening 自适应扩散系数
        min_pos_jitter   % 位置抖动绝对下限 (m)
        min_vel_jitter   % 速度抖动绝对下限 (m/s)
        min_att_jitter   % 姿态抖动绝对下限 (rad)
        
        % 批处理与贫化监控参数
        batch_size               % 渐进式权重更新的批大小 (设为大数如 1000 即为单步全量更新)
        depletion_streak_limit   % 连续严重贫化警告次数上限，超过则终止本次运行
        depletion_streak         % 当前连续严重贫化警告计数
        update_count             % 累计 UWB 更新次数
        resample_count           % 累计重采样次数
        severe_count             % 累计严重贫化警告次数
        
        % 粒子内部核心状态变量 (9I 维全局联合粒子云)
        % P_p   : 3I x Np       (所有粒子的全局位置向量)
        % P_v   : 3I x Np       (所有粒子的全局速度向量)
        % P_R   : 3 x 3 x I x Np(所有粒子的全局姿态旋转矩阵)
        % P_logw: 1 x Np        (所有粒子的对数权重)
        P_p
        P_v
        P_R
        P_logw
        
        % 供外部调用的 MMSE 后验估计值 (与 CMLKF / CEKF 接口完全对齐)
        p
        v
        R
    end
    
    methods
        function obj = CPF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0, Noise)
            obj.Vehicle_num = Vehicle_num;
            obj.Anchor_num  = Anchor_num;
            obj.anchors     = anchors;
            obj.dt_imu      = dt_imu;
            obj.g_vec       = [0; 0; -9.81];
            
            % 1. 默认噪声参数配置 (支持外部随时覆盖)
            obj.IMU_Sigma_a   = (0.25)^2 * eye(3);      
            obj.IMU_Sigma_w   = (0.025)^2 * eye(3);     
            obj.UWB_sigma_anc = 0.18;                  
            obj.UWB_sigma_rel = 0.18;                  
            
            if nargin >= 8 && ~isempty(Noise) && isstruct(Noise)
                if isfield(Noise, 'IMU_Sigma_a'),   obj.IMU_Sigma_a   = Noise.IMU_Sigma_a;   end
                if isfield(Noise, 'IMU_Sigma_w'),   obj.IMU_Sigma_w   = Noise.IMU_Sigma_w;   end
                if isfield(Noise, 'UWB_sigma_anc'), obj.UWB_sigma_anc = Noise.UWB_sigma_anc; end
                if isfield(Noise, 'UWB_sigma_rel'), obj.UWB_sigma_rel = Noise.UWB_sigma_rel; end
            end
            
            % 2. 粒子与抗贫化默认配置
            obj.Np             = 500;
            obj.neff_ratio     = 0.5;     % 对应文档 N_thresh = 0.5 * N_p
            obj.rough_coeff    = 0.02;    % 重采样后自适应扩散比例
            obj.min_pos_jitter = 1e-4;    % 位置最小抖动下限 (m)
            obj.min_vel_jitter = 1e-4;    % 速度最小抖动下限 (m/s)
            obj.min_att_jitter = 1e-4;    % 姿态最小抖动下限 (rad)
            
            % 3. 批处理与贫化监控配置
            obj.batch_size             = 1000; % 默认大数即单历元全量更新；可调小以启用渐进批处理
            obj.depletion_streak_limit = 3;    % 连续超过 3 次严重贫化则报错终止
            obj.depletion_streak       = 0;
            obj.update_count           = 0;
            obj.resample_count         = 0;
            obj.severe_count           = 0;
            
            % 4. 初始化外部输出状态与粒子云
            obj.p = p0(:);
            obj.v = v0(:);
            obj.R = R0;
            
            obj.set_particle_count(obj.Np);
        end
        
        function set_particle_count(obj, Np_new)
            % 根据当前状态 (p, v, R) 初始化/重置 Np_new 个粒子
            I_num  = obj.Vehicle_num;
            obj.Np = Np_new;
            
            obj.P_p    = repmat(obj.p(:), 1, Np_new);
            obj.P_v    = repmat(obj.v(:), 1, Np_new);
            obj.P_R    = repmat(obj.R, 1, 1, 1, Np_new);
            obj.P_logw = log(1 / Np_new) * ones(1, Np_new);
        end
        
        function predict(obj, acc_m, gyro_m)
            % 预测步：对应文档 Section II (公式 6-12)
            % 若外部直接修改了 obj.Np，自动同步粒子数组大小
            if size(obj.P_p, 2) ~= obj.Np
                obj.set_particle_count(obj.Np);
            end
            
            I_num = obj.Vehicle_num;
            tau   = obj.dt_imu;
            Np_curr = obj.Np;
            
            % 实时根据当前 IMU_Sigma_a 和 IMU_Sigma_w 计算噪声采样下三角阵 (修复 CRBPF 静态噪声 Bug)
            L_a = chol((obj.IMU_Sigma_a + obj.IMU_Sigma_a') / 2, 'lower');
            L_w = chol((obj.IMU_Sigma_w + obj.IMU_Sigma_w') / 2, 'lower');
            
            for i = 1:I_num
                idx_3 = 3*i-2 : 3*i;
                ai = acc_m(:, i);
                wi = gyro_m(:, i);
                
                % 批量独立采样第 i 辆车在所有粒子上的过程噪声 (式 6)
                e_a_all = L_a * randn(3, Np_curr);
                e_w_all = L_w * randn(3, Np_curr);
                
                for k = 1:Np_curr
                    R_old = obj.P_R(:, :, i, k);
                    v_old = obj.P_v(idx_3, k);
                    p_old = obj.P_p(idx_3, k);
                    
                    % 角速度增量与流形传播 (式 7, 9)
                    phi_k = tau * (wi - e_w_all(:, k));
                    obj.P_R(:, :, i, k) = R_old * obj.exp_SO3(phi_k);
                    
                    % 导航系加速度与速度、位置传播 (式 8, 10, 11)
                    a_n = R_old * (ai - e_a_all(:, k)) + obj.g_vec;
                    obj.P_v(idx_3, k) = v_old + tau * a_n;
                    obj.P_p(idx_3, k) = p_old + tau * v_old + 0.5 * (tau^2) * a_n;
                end
            end
            % 粒子先验权重保持不变: w_{t|t+1}^{(k)} = w_{t|t}^{(k)} (式 12)
            
            % 同步更新外部 MMSE 估计
            obj.update_mmse_estimate();
        end
        
        function update(obj, uwb_anc, uwb_rel)
            % 更新步：对应文档 Section III (公式 13-24)
            if size(obj.P_p, 2) ~= obj.Np
                obj.set_particle_count(obj.Np);
            end
            
            I_num = obj.Vehicle_num;
            obj.update_count = obj.update_count + 1;
            
            % --- 1. 提取所有有效观测 (非 NaN) ---
            meas_vals  = [];
            meas_types = []; % 1: 基站测距, 2: 车间相对测距
            meas_nodes = []; % [i, l] 或 [i, j]
            
            for i = 1:I_num
                for l = 1:obj.Anchor_num
                    if ~isnan(uwb_anc(i, l))
                        meas_vals  = [meas_vals; uwb_anc(i, l)];
                        meas_types = [meas_types; 1];
                        meas_nodes = [meas_nodes; i, l];
                    end
                end
                for j = 1:I_num
                    if j ~= i && ~isnan(uwb_rel(i, j))
                        meas_vals  = [meas_vals; uwb_rel(i, j)];
                        meas_types = [meas_types; 2];
                        meas_nodes = [meas_nodes; i, j];
                    end
                end
            end
            
            M = length(meas_vals);
            if M == 0, return; end
            
            % 实时读取当前 UWB 测量方差
            var_anc = obj.UWB_sigma_anc^2;
            var_rel = obj.UWB_sigma_rel^2;
            
            R_var_all = zeros(M, 1);
            R_var_all(meas_types == 1) = var_anc;
            R_var_all(meas_types == 2) = var_rel;
            
            % --- 2. 渐进式批处理更新 (当 batch_size >= M 时即为单步全量更新) ---
            batch_sz = max(1, obj.batch_size);
            if batch_sz < M
                shuffle_idx = randperm(M);
                meas_vals   = meas_vals(shuffle_idx);
                meas_types  = meas_types(shuffle_idx);
                meas_nodes  = meas_nodes(shuffle_idx, :);
                R_var_all   = R_var_all(shuffle_idx);
            end
            
            num_batches = ceil(M / batch_sz);
            
            for b = 1:num_batches
                idx_start = (b - 1) * batch_sz + 1;
                idx_end   = min(b * batch_sz, M);
                
                % 在对数域累加当前批次所有测量的联合对数似然 (式 13-17，向量化计算)
                log_lik_batch = zeros(1, obj.Np);
                
                for m = idx_start:idx_end
                    z_meas = meas_vals(m);
                    sig2   = R_var_all(m);
                    i      = meas_nodes(m, 1);
                    p_i_particles = obj.P_p(3*i-2 : 3*i, :); % 3 x Np
                    
                    if meas_types(m) == 1
                        % 基站测距预测 (式 13)
                        l_idx = meas_nodes(m, 2);
                        c_l   = obj.anchors(l_idx, :)';
                        diff  = p_i_particles - c_l;
                    else
                        % 车间相对测距预测 (式 14：直接取同一粒子内的双车位置差)
                        j     = meas_nodes(m, 2);
                        p_j_particles = obj.P_p(3*j-2 : 3*j, :); % 3 x Np
                        diff  = p_i_particles - p_j_particles;
                    end
                    
                    dist_pred = sqrt(sum(diff.^2, 1)); % 1 x Np
                    res_sq    = (z_meas - dist_pred).^2;
                    log_lik_batch = log_lik_batch - 0.5 * (res_sq / sig2) - 0.5 * log(2 * pi * sig2);
                end
                
                % 权重更新与归一化 (式 17-18)
                obj.P_logw = obj.P_logw + log_lik_batch;
                max_logw   = max(obj.P_logw);
                w_norm     = exp(obj.P_logw - max_logw);
                w_norm     = w_norm / sum(w_norm);
                obj.P_logw = log(max(w_norm, realmin));
                
                % 评估有效粒子数 N_eff (式 22)
                N_eff = 1 / sum(w_norm.^2);
                
                % 贫化监测与重采样判断 (式 23-24)
                if N_eff < obj.neff_ratio * obj.Np
                    if N_eff / obj.Np < 0.1
                        obj.severe_count     = obj.severe_count + 1;
                        obj.depletion_streak = obj.depletion_streak + 1;
                        if obj.depletion_streak > obj.depletion_streak_limit
                            error('CPF:SevereParticleDepletion', ...
                                  ['连续 %d 次出现粒子严重贫化警告 (UWB 更新序号 %d, 累计 %d 次)，' ...
                                   '已终止本次运行以便调整参数。'], ...
                                  obj.depletion_streak, obj.update_count, obj.severe_count);
                        end
                    else
                        obj.depletion_streak = 0;
                    end
                    
                    % 若还有后续批次，先提取当前后验均值再重采样与微抖动
                    obj.resample_and_roughen(w_norm);
                else
                    obj.depletion_streak = 0;
                end
            end
            
            % --- 3. 提取最终后验状态估计 (式 19-21) ---
            obj.update_mmse_estimate();
        end
        
        function resample_and_roughen(obj, w_norm)
            obj.resample_count = obj.resample_count + 1;
            Np_curr = obj.Np;
            
            % A. 系统重采样 (Systematic Resampling, 式 23-24)
            c = cumsum(w_norm(:));
            u = ((0 : Np_curr - 1)' + rand()) / Np_curr;
            idx = zeros(1, Np_curr);
            i = 1;
            for j = 1:Np_curr
                while i < Np_curr && u(j) > c(i)
                    i = i + 1;
                end
                idx(j) = i;
            end
            
            obj.P_p    = obj.P_p(:, idx);
            obj.P_v    = obj.P_v(:, idx);
            obj.P_R    = obj.P_R(:, :, :, idx);
            obj.P_logw = log(1 / Np_curr) * ones(1, Np_curr);
            
            % B. 自适应微抖动 (Roughening，防止多批次重采样时粒子坍缩为单点)
            if obj.rough_coeff > 0 || obj.min_pos_jitter > 0 || obj.min_vel_jitter > 0 || obj.min_att_jitter > 0
                % 位置与速度抖动
                std_p = std(obj.P_p, 0, 2);
                std_v = std(obj.P_v, 0, 2);
                jit_p = obj.rough_coeff * std_p + obj.min_pos_jitter;
                jit_v = obj.rough_coeff * std_v + obj.min_vel_jitter;
                obj.P_p = obj.P_p + jit_p .* randn(size(obj.P_p));
                obj.P_v = obj.P_v + jit_v .* randn(size(obj.P_v));
                
                % SO(3) 姿态流形抖动
                for v_idx = 1:obj.Vehicle_num
                    R_sum = sum(obj.P_R(:, :, v_idx, :), 4);
                    [U, ~, V] = svd(R_sum);
                    R_mean = U * diag([1, 1, det(U * V')]) * V';
                    
                    err_sq_sum = 0;
                    for k = 1:Np_curr
                        err_vec = obj.log_SO3(R_mean' * obj.P_R(:, :, v_idx, k));
                        err_sq_sum = err_sq_sum + sum(err_vec.^2);
                    end
                    sigma_theta = sqrt(err_sq_sum / Np_curr);
                    jit_att = obj.rough_coeff * sigma_theta + obj.min_att_jitter;
                    
                    if jit_att > 0
                        for k = 1:Np_curr
                            dtheta = jit_att * randn(3, 1);
                            obj.P_R(:, :, v_idx, k) = obj.P_R(:, :, v_idx, k) * obj.exp_SO3(dtheta);
                        end
                    end
                end
            end
        end
        
        function update_mmse_estimate(obj)
            % 计算加权后验均值与 SO(3) 正交投影均值 (式 19-21)
            max_logw = max(obj.P_logw);
            w_norm   = exp(obj.P_logw - max_logw);
            w_norm   = w_norm(:) / sum(w_norm); % Np x 1
            
            % 位置与速度加权平均 (式 19)
            obj.p = obj.P_p * w_norm;
            obj.v = obj.P_v * w_norm;
            
            % 旋转矩阵加权平均与 SVD 正交投影 (式 20-21)
            for i = 1:obj.Vehicle_num
                R_i_all = reshape(obj.P_R(:, :, i, :), 9, obj.Np);
                R_bar   = reshape(R_i_all * w_norm, 3, 3);
                [U, ~, V] = svd(R_bar);
                obj.R(:, :, i) = U * diag([1, 1, det(U * V')]) * V';
            end
        end
        
        % =========================================================
        % 辅助函数：李代数和流形运算操作
        % =========================================================
        function S = skew(~, v)
            S = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
        end
        
        function v = log_SO3(~, R)
            tr = trace(R);
            theta = acos(max(-1, min(1, (tr - 1) / 2)));
            if theta < 1e-6
                v = zeros(3, 1);
            else
                w_skew = (theta / (2 * sin(theta))) * (R - R');
                v = [w_skew(3, 2); w_skew(1, 3); w_skew(2, 1)];
            end
        end
        
        function R = exp_SO3(obj, v)
            theta = norm(v);
            if theta < 1e-6
                R = eye(3) + obj.skew(v);
            else
                n = v / theta; S = obj.skew(n);
                R = eye(3) + sin(theta) * S + (1 - cos(theta)) * (S * S);
            end
        end
    end
end