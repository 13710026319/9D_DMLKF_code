classdef DMLKF_V1 < handle
    % DMLKF_V1 - 预言家下界版 (Oracle Bound) 分布式最大似然卡尔曼滤波
    % 核心验证目标：隔离 D-GN 优化误差。
    % 机制：使用上帝视角的集中式 GN 解出完美的全局 MLE 状态和联合海森矩阵；
    % 随后每个节点按分布式拓扑提取属于自己 U_i 的子矩阵，执行局部分布式舒尔补融合。
    
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
        
        max_iter
        epsilon
        beta_inv % LM 阻尼系数 (保证 H 矩阵正定下限)
        max_step % 防爆墙：牛顿迭代最大移动阈值
        
        Nodes 
    end
    
    methods
        function obj = DMLKF_V1(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0)
            obj.Vehicle_num = Vehicle_num;
            obj.Anchor_num = Anchor_num;
            obj.anchors = anchors;
            obj.dt_imu = dt_imu;
            obj.g_vec = [0; 0; -9.81];
            
            obj.IMU_Sigma_a = (0.07)^2 * eye(3);
            obj.IMU_Sigma_w = (0.007)^2 * eye(3);
            obj.UWB_sigma_anc = 0.1;
            obj.UWB_sigma_rel = 0.1;
            
            obj.max_iter = 10;
            obj.epsilon  = 1e-4;
            obj.beta_inv = 0.1;  
            obj.max_step = 1; 
            
            Sigma_0 = blkdiag((0.1^2)*eye(3), (0.1^2)*eye(3), ((pi/180)^2)*eye(3));
            obj.Nodes = cell(Vehicle_num, 1);
            for i = 1:Vehicle_num
                obj.Nodes{i}.p = p0(3*i-2 : 3*i);
                obj.Nodes{i}.v = v0(3*i-2 : 3*i);
                obj.Nodes{i}.R = R0(:, :, i);
                obj.Nodes{i}.Sigma = Sigma_0;
            end
        end
        
        function predict(obj, acc_m, gyro_m)
            % 独立执行本地 IMU 预测步
            tau = obj.dt_imu;
            I3 = eye(3); O3 = zeros(3);
            Q_local = blkdiag(obj.IMU_Sigma_a, obj.IMU_Sigma_w);
            
            for i = 1:obj.Vehicle_num
                ai = acc_m(:, i);
                wi = gyro_m(:, i);
                Ri_old = obj.Nodes{i}.R;
                vi_old = obj.Nodes{i}.v;
                pi_old = obj.Nodes{i}.p;
                
                Ri_new = Ri_old * obj.exp_SO3(tau * wi);
                vi_new = vi_old + tau * (Ri_old * ai + obj.g_vec);
                pi_new = pi_old + tau * vi_old + 0.5 * tau^2 * (Ri_old * ai + obj.g_vec);
                
                a_skew = obj.skew(ai);
                A_i = [I3, tau * I3, -0.5 * tau^2 * Ri_old * a_skew;
                       O3, I3,       -tau * Ri_old * a_skew;
                       O3, O3,       obj.exp_SO3(-tau * wi)]; 
                W_i = [-0.5 * tau^2 * Ri_old, O3;
                       -tau * Ri_old,         O3;
                       O3,                   -tau * obj.Jr_SO3(tau * wi)];
                   
                Q_i = W_i * Q_local * W_i';
                
                obj.Nodes{i}.p = pi_new;
                obj.Nodes{i}.v = vi_new;
                obj.Nodes{i}.R = Ri_new;
                Sigma_new = A_i * obj.Nodes{i}.Sigma * A_i' + Q_i;
                obj.Nodes{i}.Sigma = (Sigma_new + Sigma_new') / 2;
            end
        end
        
        function update(obj, uwb_anc, uwb_rel)
            I_num = obj.Vehicle_num;
            
            % --- 0. 冻结全局先验状态与协方差缓存 (防止 Data Incest) ---
            p_prior = zeros(3, I_num);
            Sigma_prior_cache = cell(I_num, 1);
            for i = 1:I_num
                p_prior(:, i) = obj.Nodes{i}.p;
                Sigma_prior_cache{i} = obj.Nodes{i}.Sigma;
            end
            
            % ==========================================================
            % 1. 上帝视角的完美集中式 MLE 优化 (Oracle Global GN)
            % 完全放弃先验，求解无任何分布式截断误差的最优解与全局海森
            % ==========================================================
            p_iter = p_prior;
            H_glob_final = zeros(3*I_num, 3*I_num);
            
            for iter = 1:obj.max_iter
                H_glob = zeros(3*I_num, 3*I_num);
                g_glob = zeros(3*I_num, 1);
                
                for i = 1:I_num
                    p_i = p_iter(:, i);
                    
                    % 基站测距
                    for k = 1:obj.Anchor_num
                        z = uwb_anc(i, k);
                        if ~isnan(z)
                            delta = p_i - obj.anchors(k, :)';
                            d = max(norm(delta), 1e-4);
                            u_vec = delta / d;
                            
                            sig2 = obj.UWB_sigma_anc^2;
                            grad = (1/sig2) * (1 - z/d) * delta;
                            hess = (1/sig2) * (u_vec * u_vec'); % GN近似
                            
                            idx_i = 3*i-2 : 3*i;
                            g_glob(idx_i) = g_glob(idx_i) + grad;
                            H_glob(idx_i, idx_i) = H_glob(idx_i, idx_i) + hess;
                        end
                    end
                    
                    % 相对测距
                    for j = 1:I_num
                        z = uwb_rel(i, j);
                        if ~isnan(z) && j ~= i
                            p_j = p_iter(:, j);
                            delta = p_i - p_j;
                            d = max(norm(delta), 1e-4);
                            u_vec = delta / d;
                            
                            sig2 = obj.UWB_sigma_rel^2;
                            g_i =  (1/sig2) * (1 - z/d) * delta;
                            g_j = -(1/sig2) * (1 - z/d) * delta;
                            hess = (1/sig2) * (u_vec * u_vec');
                            
                            idx_i = 3*i-2 : 3*i;
                            idx_j = 3*j-2 : 3*j;
                            
                            g_glob(idx_i) = g_glob(idx_i) + g_i;
                            g_glob(idx_j) = g_glob(idx_j) + g_j;
                            
                            H_glob(idx_i, idx_i) = H_glob(idx_i, idx_i) + hess;
                            H_glob(idx_j, idx_j) = H_glob(idx_j, idx_j) + hess;
                            H_glob(idx_i, idx_j) = H_glob(idx_i, idx_j) - hess;
                            H_glob(idx_j, idx_i) = H_glob(idx_j, idx_i) - hess;
                        end
                    end
                end
                
                H_glob = (H_glob + H_glob') / 2;
                H_glob_final = H_glob; % 缓存最后一次求出的全局完美海森
                
                % Levenberg-Marquardt LM 下降
                H_reg = H_glob + obj.beta_inv * eye(3*I_num);
                dp_glob = H_reg \ g_glob;
                
                % 步长防护与防爆
                if any(isnan(dp_glob(:))) || any(isinf(dp_glob(:))), dp_glob = zeros(3*I_num, 1); end
                for i = 1:I_num
                    idx_i = 3*i-2 : 3*i;
                    step_i = dp_glob(idx_i);
                    if norm(step_i) > obj.max_step
                        dp_glob(idx_i) = step_i * (obj.max_step / norm(step_i));
                    end
                end
                
                p_iter = p_iter - reshape(dp_glob, 3, I_num);
                
                if max(abs(dp_glob)) < obj.epsilon
                    break;
                end
            end
            p_MLE_perfect = p_iter; % 获取到预言家视角的完美坐标解
            
            % ==========================================================
            % 2. 模拟分布式融合架构 (局部完美提取 + 块对角先验 + 舒尔补)
            % ==========================================================
            for i = 1:I_num
                % 构建节点 i 的局部子网络 U_i
                neighbors = find(~isnan(uwb_rel(i, :)) | ~isnan(uwb_rel(:, i)'));
                neighbors = setdiff(neighbors, i);
                U_i = [i, neighbors]; % 严格保证 ego 节点 i 排在第一位
                Ui_len = length(U_i);
                
                if Ui_len == 0, continue; end
                
                % A. 从完美全局结果中抠出属于 U_i 的局部状态误差与联合海森阵
                Lambda_3D = zeros(3*Ui_len, 3*Ui_len);
                s_dn_vec = zeros(3*Ui_len, 1);
                
                for c_idx = 1:Ui_len
                    c = U_i(c_idx);
                    % 从集中式 MLE 解中提取该节点的完美局部误差
                    s_dn_vec(3*c_idx-2 : 3*c_idx) = p_MLE_perfect(:, c) - p_prior(:, c);
                    for d_idx = 1:Ui_len
                        d = U_i(d_idx);
                        % 从集中式海森中抠出属于 U_i 的主子矩阵
                        Lambda_3D(3*c_idx-2:3*c_idx, 3*d_idx-2:3*d_idx) = H_glob_final(3*c-2:3*c, 3*d-2:3*d);
                    end
                end
                Lambda_3D = (Lambda_3D + Lambda_3D') / 2;
                
                % 保证局部提取的海森阵半正定
                [V_lam, D_lam] = eig(Lambda_3D);
                eig_lam = diag(D_lam); eig_lam(eig_lam < 0) = 0;
                Lambda_3D = V_lam * diag(eig_lam) * V_lam';
                
                % B. 提升至 9D 空间
                Pi_mat = kron(eye(Ui_len), [eye(3), zeros(3,3), zeros(3,3)]);
                Lambda_9D = Pi_mat' * Lambda_3D * Pi_mat;
                lambda_9D = Pi_mat' * (Lambda_3D * s_dn_vec);
                
                % C. 构建块对角分布式的“残缺先验” (导致其弱于 CMLKF 的根源)
                Gamma_prior = zeros(9*Ui_len, 9*Ui_len);
                for c_idx = 1:Ui_len
                    c = U_i(c_idx);
                    Sigma_c = (Sigma_prior_cache{c} + Sigma_prior_cache{c}') / 2 + 1e-12 * eye(9);
                    Gamma_prior(9*c_idx-8 : 9*c_idx, 9*c_idx-8 : 9*c_idx) = eye(9) / Sigma_c;
                end
                
                % D. 局部信息融合
                Gamma_post = Gamma_prior + Lambda_9D;
                gamma_post = lambda_9D; 
                
                % E. 舒尔补边缘化，提纯 ego 节点
                Gamma_ii = Gamma_post(1:9, 1:9);
                Gamma_iN = Gamma_post(1:9, 10:end);
                Gamma_Ni = Gamma_post(10:end, 1:9);
                Gamma_NN = Gamma_post(10:end, 10:end);
                
                gamma_i = gamma_post(1:9);
                gamma_N = gamma_post(10:end);
                
                if isempty(Gamma_NN)
                    Gamma_ego = Gamma_ii;
                    gamma_ego = gamma_i;
                else
                    Gamma_NN_reg = Gamma_NN + 1e-8 * eye(size(Gamma_NN));
                    Gamma_ego = Gamma_ii - Gamma_iN * (Gamma_NN_reg \ Gamma_Ni);
                    gamma_ego = gamma_i - Gamma_iN * (Gamma_NN_reg \ gamma_N);
                end
                
                Gamma_ego = (Gamma_ego + Gamma_ego') / 2 + 1e-10 * eye(9);
                Sigma_ego = eye(9) / Gamma_ego;
                Sigma_ego = (Sigma_ego + Sigma_ego') / 2;
                
                theta_ego = Gamma_ego \ gamma_ego;
                
                if any(isnan(theta_ego)) || any(isinf(theta_ego))
                    theta_ego = zeros(9, 1);
                    Sigma_ego = Sigma_prior_cache{i}; 
                end
                
                % F. 状态流形收回
                dp   = theta_ego(1:3);
                dv   = theta_ego(4:6);
                dphi = theta_ego(7:9);
                
                obj.Nodes{i}.p = obj.Nodes{i}.p + dp;
                obj.Nodes{i}.v = obj.Nodes{i}.v + dv;
                R_new = obj.Nodes{i}.R * obj.exp_SO3(dphi);
                [U, ~, V] = svd(R_new);
                obj.Nodes{i}.R = U * V';
                obj.Nodes{i}.Sigma = Sigma_ego;
            end
        end
        
        % =========================================================
        % 辅助函数：李代数和流形运算操作
        % =========================================================
        function S = skew(~, v)
            S = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
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
        
        function J = Jr_SO3(obj, v)
            theta = norm(v);
            if theta < 1e-6
                J = eye(3) - 0.5 * obj.skew(v);
            else
                n = v / theta; S = obj.skew(n);
                J = eye(3) - (1 - cos(theta))/theta * S + (theta - sin(theta))/theta * (S * S);
            end
        end
    end
end