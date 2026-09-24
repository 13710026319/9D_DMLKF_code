classdef CRBPF < handle
    % CRBPF - 集中式 Rao-Blackwellized 粒子滤波
    % 核心原理：在姿态 SO(3) 空间撒粒子进行非线性采样，在位置-速度(6I)空间使用解析线性卡尔曼跟踪。
    % 集中式优势：完美维护全局 6I x 6I 协方差矩阵，从数学上精确处理所有车辆的交叉相关性，无需任何保守方差膨胀。
    
    properties
        Vehicle_num
        Anchor_num
        anchors
        dt_imu
        g_vec
        
        IMU_Sigma_a
        IMU_Sigma_w
        UWB_sigma_anc
        UWB_sigma_rel
        
        % 粒子滤波核心参数
        Np               % 粒子总数
        neff_ratio       % 重采样阈值比例
        rough_coeff      % roughening 扩散系数
        min_att_jitter   % roughening 绝对下限 (rad)
        
        % 批处理与贫化监控参数
        batch_size               % 渐进式权重更新的批大小 (等于单历元总测量数时即全量更新)
        depletion_streak_limit   % 连续贫化警告次数上限，超过则终止本次运行
        depletion_streak         % 当前连续贫化警告计数
        update_count             % 累计 UWB 更新次数
        resample_count           % 累计重采样次数
        severe_count             % 累计严重贫化警告次数
        
        % 全局线性动力学矩阵
        F_global
        Q_global
        
        % 粒子内部核心状态变量 (采用数组结构以提升 MATLAB 运算速度)
        % P_Xl: 6I x Np     (所有粒子的全局位置速度向量)
        % P_Pl: 6I x 6I x Np(所有粒子的全局位置速度协方差矩阵)
        % P_R : 3 x 3 x I x Np (所有粒子的姿态矩阵)
        % P_logw: 1 x Np    (所有粒子的对数权重)
        P_Xl
        P_Pl
        P_R
        P_logw
        
        % 供外部调用的 MMSE 估计值 (与 CEKF/DMLKF 接口对齐)
        p
        v
        R
    end
    
    methods
        function obj = CRBPF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0)
            obj.Vehicle_num = Vehicle_num;
            obj.Anchor_num = Anchor_num;
            obj.anchors = anchors;
            obj.dt_imu = dt_imu;
            obj.g_vec = [0; 0; -9.81];
            
            % 1. 设置严格的噪声配置
            obj.IMU_Sigma_a = (0.25)^2 * eye(3);      
            obj.IMU_Sigma_w = (0.015)^2 * eye(3);     
            obj.UWB_sigma_anc = 0.08;                  
            obj.UWB_sigma_rel = 0.08;                  
            
            % 2. 粒子与抗贫化配置
            obj.Np = 300;            
            obj.neff_ratio = 0.65;     % 重采样门槛
            obj.rough_coeff = 0.05;    % 重采样后注入噪声的扩散程度
            obj.min_att_jitter = 0.001; % 重采样后注入的抖动下限
            
            % 2b. 批处理与贫化监控配置
            obj.batch_size = 40;              % 渐进式批处理窗口 (40 = 单历元全量测量)
            obj.depletion_streak_limit = 3;   % 连续 4 次贫化警告后仍继续则终止本次运行
            obj.depletion_streak = 0;
            obj.update_count = 0;
            obj.resample_count = 0;
            obj.severe_count = 0;
            
            % 3. 构建全局线性卡尔曼矩阵 (F_global 和 Q_global)
            % 由于位置速度更新在条件姿态下是严格线性的，且 Q 独立于具体姿态近似为常数块
            I_num = obj.Vehicle_num;
            tau = obj.dt_imu;
            
            F_blk = [eye(3), tau*eye(3); zeros(3), eye(3)];
            F_cell = repmat({F_blk}, 1, I_num);
            obj.F_global = blkdiag(F_cell{:});
            
            Q_blk = [ (tau^4)/4 * obj.IMU_Sigma_a, (tau^3)/2 * obj.IMU_Sigma_a;
                      (tau^3)/2 * obj.IMU_Sigma_a, (tau^2)   * obj.IMU_Sigma_a ];
            Q_cell = repmat({Q_blk}, 1, I_num);
            obj.Q_global = blkdiag(Q_cell{:});
            
            % 4. 粒子系统初始化
            obj.P_Xl = zeros(6*I_num, obj.Np);
            obj.P_Pl = zeros(6*I_num, 6*I_num, obj.Np);
            obj.P_R  = zeros(3, 3, I_num, obj.Np);
            obj.P_logw = log(1 / obj.Np) * ones(1, obj.Np);
            
            Xl_init = zeros(6*I_num, 1);
            for i = 1:I_num
                Xl_init(6*i-5 : 6*i-3) = p0(3*i-2 : 3*i);
                Xl_init(6*i-2 : 6*i)   = v0(3*i-2 : 3*i);
            end
            
            P_l_init_cell = repmat({blkdiag((0.1^2)*eye(3), (0.1^2)*eye(3))}, 1, I_num);
            P_l_init = blkdiag(P_l_init_cell{:});
            
            for k = 1:obj.Np
                obj.P_Xl(:, k) = Xl_init;
                obj.P_Pl(:, :, k) = P_l_init;
                obj.P_R(:, :, :, k) = R0;
            end
            
            % 初始化外部输出状态
            obj.p = p0;
            obj.v = v0;
            obj.R = R0;
        end
        
        function set_particle_count(obj, Np_new)
            % 用当前 MMSE 状态重新铺设粒子群 (须在构造后、滤波开始前调用)
            I_num = obj.Vehicle_num;
            obj.Np = Np_new;
            
            obj.P_Xl = zeros(6*I_num, Np_new);
            obj.P_Pl = zeros(6*I_num, 6*I_num, Np_new);
            obj.P_R  = zeros(3, 3, I_num, Np_new);
            obj.P_logw = log(1 / Np_new) * ones(1, Np_new);
            
            Xl_init = zeros(6*I_num, 1);
            for i = 1:I_num
                Xl_init(6*i-5 : 6*i-3) = obj.p(3*i-2 : 3*i);
                Xl_init(6*i-2 : 6*i)   = obj.v(3*i-2 : 3*i);
            end
            
            P_l_init_cell = repmat({blkdiag((0.1^2)*eye(3), (0.1^2)*eye(3))}, 1, I_num);
            P_l_init = blkdiag(P_l_init_cell{:});
            
            for k = 1:Np_new
                obj.P_Xl(:, k) = Xl_init;
                obj.P_Pl(:, :, k) = P_l_init;
                obj.P_R(:, :, :, k) = obj.R;
            end
        end
        
        function predict(obj, acc_m, gyro_m)
            % 预测步：姿态非线性采样 (Lie Group) + 线性解析传播 (Kalman)
            I_num = obj.Vehicle_num;
            tau = obj.dt_imu;
            
            % 预计算角速度噪声的协方差下三角阵 (用于批量采样)
            % 积分后角度误差 delta_phi ~ N(0, tau^2 * Sigma_w)
            L_w = chol(tau^2 * obj.IMU_Sigma_w, 'lower');
            
            for k = 1:obj.Np
                B_global_k = zeros(6*I_num, 1);
                
                for i = 1:I_num
                    ai = acc_m(:, i);
                    wi = gyro_m(:, i);
                    R_old = obj.P_R(:, :, i, k);
                    
                    % A. 姿态流形采样 (引入颗粒随机性)
                    delta_phi = L_w * randn(3, 1); 
                    R_new = R_old * obj.exp_SO3(tau * wi) * obj.exp_SO3(delta_phi);
                    obj.P_R(:, :, i, k) = R_new;
                    
                    % B. 组装该粒子的特定条件控制输入 B
                    % a_world = R_old * a_body + g
                    a_world = R_old * ai + obj.g_vec;
                    B_global_k(6*i-5 : 6*i-3) = 0.5 * tau^2 * a_world;
                    B_global_k(6*i-2 : 6*i)   = tau * a_world;
                end
                
                % C. 解析线性卡尔曼时间更新
                obj.P_Xl(:, k) = obj.F_global * obj.P_Xl(:, k) + B_global_k;
                obj.P_Pl(:, :, k) = obj.F_global * obj.P_Pl(:, :, k) * obj.F_global' + obj.Q_global;
            end
            
            % 同步更新外部 MMSE 估计
            obj.update_mmse_estimate();
        end
        
        function update(obj, uwb_anc, uwb_rel)
            I_num = obj.Vehicle_num;
            
            obj.update_count = obj.update_count + 1;
            
            % --- 1. 动态提取所有有效观测并组装成全局观测向量 ---
            meas_vals = [];
            meas_types = []; % 1: anchor, 2: relative
            meas_nodes = []; % [i, k] or [i, j]
            
            for i = 1:I_num
                for k = 1:obj.Anchor_num
                    if ~isnan(uwb_anc(i, k))
                        meas_vals = [meas_vals; uwb_anc(i, k)];
                        meas_types = [meas_types; 1];
                        meas_nodes = [meas_nodes; i, k];
                    end
                end
                for j = 1:I_num
                    if j ~= i && ~isnan(uwb_rel(i, j))
                        meas_vals = [meas_vals; uwb_rel(i, j)];
                        meas_types = [meas_types; 2];
                        meas_nodes = [meas_nodes; i, j];
                    end
                end
            end
            
            M = length(meas_vals);
            if M == 0, return; end % 无有效观测
            
            % 组装观测噪声方差向量
            R_diag_all = zeros(M, 1);
            R_diag_all(meas_types == 1) = obj.UWB_sigma_anc^2;
            R_diag_all(meas_types == 2) = obj.UWB_sigma_rel^2;
            
            % ==========================================================
            % [核心升级] 渐进式批处理更新 (Progressive Mini-Batch Update)
            % 解决高维测量导致的高斯似然尖锐与粒子权重崩塌问题
            % ==========================================================
            batch_size = obj.batch_size; % 批大小由属性控制 (40 = 单历元全量更新)
            
            % 随机打乱测量顺序，消除始终先融合某基站带来的顺序偏置 (Order Bias)
            shuffle_idx = randperm(M);
            meas_vals  = meas_vals(shuffle_idx);
            meas_types = meas_types(shuffle_idx);
            meas_nodes = meas_nodes(shuffle_idx, :);
            R_diag_all = R_diag_all(shuffle_idx);
            
            num_batches = ceil(M / batch_size);
            
            for b = 1 : num_batches
                % 获取当前批次的测量索引
                idx_start = (b - 1) * batch_size + 1;
                idx_end   = min(b * batch_size, M);
                m_batch   = idx_end - idx_start + 1;
                
                % 提取当前批次的数据
                batch_vals  = meas_vals(idx_start : idx_end);
                batch_types = meas_types(idx_start : idx_end);
                batch_nodes = meas_nodes(idx_start : idx_end, :);
                R_UWB_batch = diag(R_diag_all(idx_start : idx_end));
                
                % --- 遍历粒子执行该小批次的 EKF 更新与权重计算 ---
                for k = 1:obj.Np
                    Xl_k = obj.P_Xl(:, k);
                    Pl_k = obj.P_Pl(:, :, k);
                    
                    y_pred = zeros(m_batch, 1);
                    H_k    = zeros(m_batch, 6*I_num);
                    
                    for m = 1:m_batch
                        i = batch_nodes(m, 1);
                        p_i = Xl_k(6*i-5 : 6*i-3);
                        
                        if batch_types(m) == 1
                            % 基站观测
                            anc_idx = batch_nodes(m, 2);
                            delta = p_i - obj.anchors(anc_idx, :)';
                            d = max(norm(delta), 1e-4);
                            y_pred(m) = d;
                            H_k(m, 6*i-5 : 6*i-3) = (delta / d)';
                        else
                            % 相对观测
                            j = batch_nodes(m, 2);
                            p_j = Xl_k(6*j-5 : 6*j-3);
                            delta = p_i - p_j;
                            d = max(norm(delta), 1e-4);
                            y_pred(m) = d;
                            u_ij = delta / d;
                            H_k(m, 6*i-5 : 6*i-3) = u_ij';
                            H_k(m, 6*j-5 : 6*j-3) = -u_ij';
                        end
                    end
                    
                    % EKF 更新步
                    innovation = batch_vals - y_pred;
                    S = H_k * Pl_k * H_k' + R_UWB_batch;
                    S = (S + S') / 2 + 1e-8 * eye(m_batch); % 数值保护
                    
                    K_gain = (Pl_k * H_k') / S; % 这里只是一个最大 5x5 的矩阵求逆，极快！
                    
                    obj.P_Xl(:, k) = Xl_k + K_gain * innovation;
                    I_KH = eye(6*I_num) - K_gain * H_k;
                    Pl_new = I_KH * Pl_k * I_KH' + K_gain * R_UWB_batch * K_gain';
                    obj.P_Pl(:, :, k) = (Pl_new + Pl_new') / 2;
                    
                    % 计算当前批次的似然 (完全不使用退火，保留纯正高斯数学)
                    inv_S_r = S \ innovation;
                    try
                        L_chol = chol(S);
                        log_det_S = 2 * sum(log(diag(L_chol)));
                    catch
                        log_det_S = sum(log(eig(S))); 
                    end
                    
                    log_likelihood = -0.5 * (innovation' * inv_S_r) - 0.5 * log_det_S;
                    obj.P_logw(k) = obj.P_logw(k) + log_likelihood;
                end
                
                % --- 批次间的权重归一化与防贫化重采样 ---
                max_logw = max(obj.P_logw);
                w_norm = exp(obj.P_logw - max_logw);
                w_norm = w_norm / sum(w_norm);
                obj.P_logw = log(w_norm); % 保存回对数域
                
                N_eff = 1 / sum(w_norm.^2);
                
                % 如果在这个小批次中发生了退化，立刻启动重采样与抖动 (Roughening)
                % 此时优秀的粒子会被复制，并在下一批次测量中继续接受考验
                if N_eff < obj.neff_ratio * obj.Np
                    if N_eff / obj.Np < 0.1
                        obj.severe_count = obj.severe_count + 1;
                        obj.depletion_streak = obj.depletion_streak + 1;
                        if obj.depletion_streak > obj.depletion_streak_limit
                            error('CRBPF:SevereParticleDepletion', ...
                                  ['连续 %d 次出现粒子贫化警告 (UWB 更新序号 %d, 累计 %d 次)，' ...
                                   '已终止本次运行以便调整参数。'], ...
                                  obj.depletion_streak, obj.update_count, obj.severe_count);
                        end
                    else
                        obj.depletion_streak = 0;
                    end
                    obj.resample_and_roughen(w_norm);
                else
                    obj.depletion_streak = 0;
                end
            end
            
            % --- 最终提取本次完整观测更新后的最优 MMSE 状态 ---
            obj.update_mmse_estimate();
        end
        
        function resample_and_roughen(obj, w_norm)
            obj.resample_count = obj.resample_count + 1;
            
            % A. 确定性系统重采样 (Systematic Resampling)
            c = cumsum(w_norm);
            u = (0 : obj.Np - 1)' / obj.Np + rand() / obj.Np;
            idx = zeros(1, obj.Np);
            i = 1;
            for j = 1:obj.Np
                while i <= obj.Np && u(j) > c(i)
                    i = i + 1;
                end
                idx(j) = min(i, obj.Np);
            end
            
            % 复制提取幸存粒子
            obj.P_Xl = obj.P_Xl(:, idx);
            obj.P_Pl = obj.P_Pl(:, :, idx);
            obj.P_R  = obj.P_R(:, :, :, idx);
            obj.P_logw = log(1 / obj.Np) * ones(1, obj.Np);
            
            % B. 姿态抗贫化 (Roughening)
            % 基于粒子群在姿态流形上的分散程度，注入自适应抖动，防止粒子在多车同构下退化成一个点
            for v_idx = 1:obj.Vehicle_num
                % 1. 求解该车辆所有粒子的平均姿态
                R_sum = sum(obj.P_R(:, :, v_idx, :), 4);
                [U, ~, V] = svd(R_sum);
                R_mean = U * V';
                
                % 2. 评估粒子群姿态的李代数方差 (标准差)
                err_sq_sum = 0;
                for k = 1:obj.Np
                    R_k = obj.P_R(:, :, v_idx, k);
                    err_vec = obj.log_SO3(R_mean' * R_k);
                    err_sq_sum = err_sq_sum + norm(err_vec)^2;
                end
                sigma_theta = sqrt(err_sq_sum / obj.Np);
                
                % 3. 自适应生成注入抖动标准差 (Roughing Jitter)
                jitter_std = obj.rough_coeff * sigma_theta + obj.min_att_jitter;
                
                % 4. 逐粒子注入抖动
                for k = 1:obj.Np
                    delta_theta = jitter_std * randn(3, 1);
                    obj.P_R(:, :, v_idx, k) = obj.P_R(:, :, v_idx, k) * obj.exp_SO3(delta_theta);
                end
            end
        end
        
        function update_mmse_estimate(obj)
            % 外部 MMSE 提取器：计算当前粒子群的加权平均值
            w_norm = exp(obj.P_logw);
            
            Xl_mmse = zeros(6 * obj.Vehicle_num, 1);
            for k = 1:obj.Np
                Xl_mmse = Xl_mmse + w_norm(k) * obj.P_Xl(:, k);
            end
            
            for i = 1:obj.Vehicle_num
                obj.p(3*i-2 : 3*i) = Xl_mmse(6*i-5 : 6*i-3);
                obj.v(3*i-2 : 3*i) = Xl_mmse(6*i-2 : 6*i);
                
                % 旋转矩阵加权平均并正交化 (Chordal Mean)
                R_w_sum = zeros(3, 3);
                for k = 1:obj.Np
                    R_w_sum = R_w_sum + w_norm(k) * obj.P_R(:, :, i, k);
                end
                [U, ~, V] = svd(R_w_sum);
                obj.R(:, :, i) = U * V';
            end
        end
        
        % =========================================================
        % 辅助函数：李代数和流形运算操作
        % =========================================================
        function S = skew(~, v)
            S = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
        end
        
        function v = log_SO3(~, R)
            % 从 SO(3) 映射到 so(3) 李代数向量
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
