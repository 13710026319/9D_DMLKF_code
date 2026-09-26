classdef DMLKF_V1 < handle
    % DMLKF_V1 - Change 1 理论验证版 (Oracle Bound) 与CMLKF对比

    % 1. 优化步：使用集中式 GN 解出完美的全局 MLE 状态和联合海森矩阵。
    % 2. 预测步：构建全局 A 和 Q，自然传递节点与邻居之间的交叉协方差。
    % 3. 更新步：彻底抛弃舒尔补边缘化！直接提取包含自身的邻居的局部协方差块，
   
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
        function obj = DMLKF_V1(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0, Noise)
            obj.Vehicle_num = Vehicle_num;
            obj.Anchor_num = Anchor_num;
            obj.anchors = anchors;
            obj.dt_imu = dt_imu;
            obj.g_vec = [0; 0; -9.81];
            
            obj.IMU_Sigma_a = (0.25)^2 * eye(3);      
            obj.IMU_Sigma_w = (0.025)^2 * eye(3);     
            obj.UWB_sigma_anc = 0.18;                  
            obj.UWB_sigma_rel = 0.18;   

            % ==== [可选] 外部噪声参数输入 ====
            % 用法： N.IMU_Sigma_a = (0.05)^2*eye(3);  N.IMU_Sigma_w = (0.005)^2*eye(3);
            %        N.UWB_sigma_anc = 0.18;  N.UWB_sigma_rel = 0.18;
            %        kf = DMLKF_V1(V, A, anchors, dt, p0, v0, R0, N);
            % 只覆盖传入的字段；不传（或传空）时完全保持上面的默认值，行为与以前一致。
            if nargin >= 8 && ~isempty(Noise) && isstruct(Noise)
                if isfield(Noise, 'IMU_Sigma_a'),   obj.IMU_Sigma_a   = Noise.IMU_Sigma_a;   end
                if isfield(Noise, 'IMU_Sigma_w'),   obj.IMU_Sigma_w   = Noise.IMU_Sigma_w;   end
                if isfield(Noise, 'UWB_sigma_anc'), obj.UWB_sigma_anc = Noise.UWB_sigma_anc; end
                if isfield(Noise, 'UWB_sigma_rel'), obj.UWB_sigma_rel = Noise.UWB_sigma_rel; end
            end
            
            obj.max_iter = 40;
            obj.epsilon  = 1e-4;
            obj.beta_inv = 50;  
            obj.max_step = Inf; 
            
            % [修改点] 初始化：每个节点维护一个全网的 9I x 9I 协方差矩阵视图
            Sigma_0_blk = blkdiag((0.1^2)*eye(3), (0.1^2)*eye(3), ((pi/180)^2)*eye(3));
            Sigma_0_full = kron(eye(Vehicle_num), Sigma_0_blk);
            
            obj.Nodes = cell(Vehicle_num, 1);
            for i = 1:Vehicle_num
                obj.Nodes{i}.p = p0(3*i-2 : 3*i);
                obj.Nodes{i}.v = v0(3*i-2 : 3*i);
                obj.Nodes{i}.R = R0(:, :, i);
                obj.Nodes{i}.Sigma_full = Sigma_0_full; % 本地缓存的包含交叉协方差的视图
            end
        end
        
        function predict(obj, acc_m, gyro_m)
            tau = obj.dt_imu;
            I3 = eye(3); O3 = zeros(3);
            Q_local = blkdiag(obj.IMU_Sigma_a, obj.IMU_Sigma_w);
            
            % [修改点] 预测步：构建全局动力学矩阵 A_global 和 Q_global
            % 目的：让节点间的"交叉协方差"能够按照真实的物理模型进行自然传播
            A_global = zeros(9*obj.Vehicle_num, 9*obj.Vehicle_num);
            Q_global = zeros(9*obj.Vehicle_num, 9*obj.Vehicle_num);
            
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
                
                idx_9D = 9*i-8 : 9*i;
                A_global(idx_9D, idx_9D) = A_i;
                Q_global(idx_9D, idx_9D) = Q_i;
                
                obj.Nodes{i}.p = pi_new;
                obj.Nodes{i}.v = vi_new;
                obj.Nodes{i}.R = Ri_new;
            end
            
            % 执行全局协方差预测，完美保留非对角线上的交叉相关性
            for i = 1:obj.Vehicle_num
                Sigma_pred = A_global * obj.Nodes{i}.Sigma_full * A_global' + Q_global;
                obj.Nodes{i}.Sigma_full = (Sigma_pred + Sigma_pred') / 2;
            end
        end
        
        function update(obj, uwb_anc, uwb_rel)
            I_num = obj.Vehicle_num;
            
            % --- 0. 冻结全局先验状态 ---
            p_prior = zeros(3, I_num);
            for i = 1:I_num
                p_prior(:, i) = obj.Nodes{i}.p;
            end
            
            % ==========================================================
            % 1. 上帝视角的完美集中式 MLE 优化 (保持不变)
            % 获取无优化近似误差的完美全局海森 H_glob_final 与状态 p_MLE_perfect
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
                            hess = (1/sig2) * (u_vec * u_vec'); 
                            
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
                H_glob_final = H_glob; 
                
                H_reg = H_glob + obj.beta_inv * eye(3*I_num);
                dp_glob = H_reg \ g_glob;
                
                if any(isnan(dp_glob(:))) || any(isinf(dp_glob(:))), dp_glob = zeros(3*I_num, 1); end
                step_norm = norm(dp_glob);
                if step_norm > obj.max_step
                    dp_glob = dp_glob * (obj.max_step / step_norm);
                end
                
                % 状态更新
                p_iter = p_iter - reshape(dp_glob, 3, I_num);
                
                % 将原来的 max(abs(dp_glob)) 改为 norm(dp_glob)
                if norm(dp_glob) < obj.epsilon
                    break; 
                end
            end
            p_MLE_perfect = p_iter; 
            
            % ==========================================================
            % 2. 模拟 Change 1 分布式融合架构 (保留全联通交叉相关性)
            % ==========================================================
            for i = 1:I_num
                % 构建节点 i 的局部子网络 U_i
                neighbors = find(~isnan(uwb_rel(i, :)) | ~isnan(uwb_rel(:, i)'));
                neighbors = setdiff(neighbors, i);
                U_i = [i, neighbors]; % 严格保证 ego 节点 i 排在第一位
                Ui_len = length(U_i);
                
                if Ui_len == 0, continue; end
                
                % 获取 U_i 对应在全局矩阵中的 9D 和 3D 索引
                idx_9D_Ui = zeros(1, 9*Ui_len);
                idx_3D_Ui = zeros(1, 3*Ui_len);
                for c_idx = 1:Ui_len
                    c = U_i(c_idx);
                    idx_9D_Ui(9*c_idx-8 : 9*c_idx) = 9*c-8 : 9*c;
                    idx_3D_Ui(3*c_idx-2 : 3*c_idx) = 3*c-2 : 3*c;
                end
                
                % A. 从完美集中式海森中，精准抠出属于 U_i 的局部海森阵
                % 由于提取顺序服从 U_i，ego 节点 i 必然位于矩阵左上角！
                Lambda_3D = H_glob_final(idx_3D_Ui, idx_3D_Ui);
                Lambda_3D = (Lambda_3D + Lambda_3D') / 2;
                
                [V_lam, D_lam] = eig(Lambda_3D);
                eig_lam = diag(D_lam); eig_lam(eig_lam < 0) = 0;
                Lambda_3D = V_lam * diag(eig_lam) * V_lam';
                
                % 从集中式 MLE 解中提取该局部的完美误差向量
                s_dn_vec = zeros(3*Ui_len, 1);
                for c_idx = 1:Ui_len
                    c = U_i(c_idx);
                    s_dn_vec(3*c_idx-2 : 3*c_idx) = p_MLE_perfect(:, c) - p_prior(:, c);
                end
                
                % B. 将 3D 位置信息提升至 9D 联合空间 (用 0 填充速度和姿态块)
                Pi_mat = kron(eye(Ui_len), [eye(3), zeros(3,3), zeros(3,3)]);
                Lambda_9D = Pi_mat' * Lambda_3D * Pi_mat;
                lambda_9D = Pi_mat' * (Lambda_3D * s_dn_vec);
                
                % C. [Change 1 核心] 直接从本地全局视图提取包含所有交叉协方差的真实先验块
                % 这彻底抛弃了导致发散的完全对角阵假设 (Block-diagonal Prior)
                Sigma_prior_Ui = obj.Nodes{i}.Sigma_full(idx_9D_Ui, idx_9D_Ui);
                Sigma_prior_Ui = (Sigma_prior_Ui + Sigma_prior_Ui') / 2 + 1e-12 * eye(9*Ui_len);
                
                % D. [Change 1 核心] 执行次优融合 (Suboptimal Fusion)
                % 抛弃舒尔补边缘化！直接使用完整局部逆矩阵相加。
                % 若拓扑为全连接(U_i 包含所有人)，此处数学公式严格等于集中式 Information Filter
                Gamma_prior_Ui = eye(size(Sigma_prior_Ui)) / Sigma_prior_Ui;
                Gamma_post_Ui  = Gamma_prior_Ui + Lambda_9D;
                Gamma_post_Ui  = (Gamma_post_Ui + Gamma_post_Ui') / 2 + 1e-10 * eye(9*Ui_len);
                
                Sigma_post_Ui = eye(size(Gamma_post_Ui)) / Gamma_post_Ui;
                Sigma_post_Ui = (Sigma_post_Ui + Sigma_post_Ui') / 2;
                
                theta_Ui = Gamma_post_Ui \ lambda_9D;
                
                % E. 状态流形收回 (仅提取左上角属于 ego 节点 i 的那 9 维更新量)
                theta_ego = theta_Ui(1:9);
                
                if any(isnan(theta_ego)) || any(isinf(theta_ego))
                    theta_ego = zeros(9, 1);
                    Sigma_post_Ui = Sigma_prior_Ui; % 异常回退保护
                end
                
                dp   = theta_ego(1:3);
                dv   = theta_ego(4:6);
                dphi = theta_ego(7:9);
                
                obj.Nodes{i}.p = obj.Nodes{i}.p + dp;
                obj.Nodes{i}.v = obj.Nodes{i}.v + dv;
                R_new = obj.Nodes{i}.R * obj.exp_SO3(dphi);
                [U, ~, V] = svd(R_new);
                obj.Nodes{i}.R = U * V';
                
                % F. [Change 1 核心] 更新本地的世界观缓存
                % 将融合后的完整块(包含保留的交叉协方差)写回本地全局视图，
                % 这就是"局部认知不一致"发生的根源(节点互相写不同块)，但完全保证了数学上的保守性。
                obj.Nodes{i}.Sigma_full(idx_9D_Ui, idx_9D_Ui) = Sigma_post_Ui;
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
