classdef DIEKF_ADMM < handle
    % DIEKF - Distributed Iterated Extended Kalman Filter via Consensus ADMM (IEKF-ADMM)
    % 基于 1 跳物理邻居子网的分布式高斯-牛顿迭代扩展卡尔曼滤波算法 (内层局部 GN-IEKF + 外层 Consensus ADMM)
    
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
        
        ADMM_iter          % 外层 ADMM 最大通信迭代次数 (K)
        GN_iter = 5          % 内层局部 GN 迭代次数 (M_in)
        rho  = 50         % ADMM 二次惩罚参数 (\rho)
        epsilon           % 收敛阈值
        max_step          % 最大步长限制 (默认 inf)
        
        diag_flag = 1       
        last_iter = 0
        last_err  = NaN
        iter_hist = []      
        res_hist  = []      
        last_clip = 0       
        clip_hist = []      
        
        print_flag = 0;   % 打印残差收敛警告
        Nodes 
        
        % --- 1-hop 固定拓扑相关量 ---
        L_hop = 1         % 固定为 1-hop 物理邻居通信
        V2V_Mask          % 物理通信连通掩码
        N_list            % N_i: 每个节点的物理 1 跳邻居集合
        U_list            % U_i = {i} \cup N_i (ego 强制排在第1位)
        U_set             % U_set(i,c)=true 表示节点 i 在追踪变量 c
        W_c               % W_c(i,j,c): 逐变量的单变量双随机 MH 权重字典
        V_c_size          % |V_c|: 全网追踪变量 c 的节点总数 (预计算常数)
    end
    
    methods
        function obj = DIEKF_ADMM(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0, V2V_Mask, max_iter, varargin)
            if nargin < 9 || isempty(max_iter)
                max_iter = 400;
            end
            
            % 兼容 10 参数 (..., max_iter, Noise) 或 11 参数 (..., max_iter, L_hop, Noise) 调用
            Noise = [];
            if ~isempty(varargin)
                if isstruct(varargin{1})
                    Noise = varargin{1};
                elseif length(varargin) >= 2 && isstruct(varargin{2})
                    Noise = varargin{2};
                end
            end

            obj.Vehicle_num = Vehicle_num;
            obj.Anchor_num  = Anchor_num;
            obj.anchors     = anchors;
            obj.dt_imu      = dt_imu;
            obj.g_vec       = [0; 0; -9.81];
            obj.L_hop       = 1; % 严格固定为 1-hop
            obj.V2V_Mask    = V2V_Mask;
            
            % 传感器噪声基准参数 (与 DMLKF 对齐)
            obj.IMU_Sigma_a   = (0.25)^2 * eye(3);      
            obj.IMU_Sigma_w   = (0.025)^2 * eye(3);     
            obj.UWB_sigma_anc = 0.18;                  
            obj.UWB_sigma_rel = 0.18;   

            % 外部噪声与算法参数覆盖
            if ~isempty(Noise) && isstruct(Noise)
                if isfield(Noise, 'IMU_Sigma_a'),   obj.IMU_Sigma_a   = Noise.IMU_Sigma_a;   end
                if isfield(Noise, 'IMU_Sigma_w'),   obj.IMU_Sigma_w   = Noise.IMU_Sigma_w;   end
                if isfield(Noise, 'UWB_sigma_anc'), obj.UWB_sigma_anc = Noise.UWB_sigma_anc; end
                if isfield(Noise, 'UWB_sigma_rel'), obj.UWB_sigma_rel = Noise.UWB_sigma_rel; end
                if isfield(Noise, 'rho'),           obj.rho           = Noise.rho;           end
                if isfield(Noise, 'M_in'),          obj.GN_iter          = Noise.M_in;          end
            end

            obj.ADMM_iter = max_iter;   
            obj.epsilon  = 1e-4; 
            obj.max_step = inf;  

            I_num = Vehicle_num;
            
            % =========================================================
            % [1] 固定物理拓扑与 1 跳索引集 U_i = {i} \cup N_i 生成
            % =========================================================
            obj.N_list = cell(I_num, 1);
            obj.U_list = cell(I_num, 1);
            obj.U_set  = false(I_num, I_num);
            
            for i = 1:I_num
                obj.N_list{i} = setdiff(find(V2V_Mask(i, :) ~= 0), i);
                u_i_nodes = [i, obj.N_list{i}]; % 自身(ego)强制排在第一位
                obj.U_list{i} = u_i_nodes;
                obj.U_set(i, u_i_nodes) = true;
            end
            
            % =========================================================
            % [2] 预计算单变量子网覆盖常数 |V_c| (用于式 19 先验合同缩放)
            % =========================================================
            obj.V_c_size = zeros(I_num, 1);
            for c = 1:I_num
                obj.V_c_size(c) = sum(obj.U_set(:, c));
            end
            
            % =========================================================
            % [3] 构建单变量双随机 MH 权重字典 W_c (对应文档式 39)
            % =========================================================
            obj.W_c = zeros(I_num, I_num, I_num);
            for c = 1:I_num
                Uc_nodes = find(obj.U_set(:, c));
                for i = Uc_nodes'
                    % d_c^i = | {l \in N_i | c \in U_l } |
                    d_c_i = sum(obj.U_set(obj.N_list{i}, c));
                    sum_w = 0;
                    comm_neighbors = intersect(obj.N_list{i}, Uc_nodes);
                    for j = comm_neighbors'
                        d_c_j = sum(obj.U_set(obj.N_list{j}, c));
                        w = 1 / (1 + max(d_c_i, d_c_j));
                        obj.W_c(i, j, c) = w;
                        sum_w = sum_w + w;
                    end
                    obj.W_c(i, i, c) = 1 - sum_w;
                end
            end
            
            % =========================================================
            % [4] 节点状态与联合协方差初始化
            % =========================================================
            Sigma_0 = blkdiag((0.1^2)*eye(3), (0.1^2)*eye(3), ((pi/180)^2)*eye(3));
            obj.Nodes = cell(Vehicle_num, 1);
            for i = 1:Vehicle_num
                obj.Nodes{i}.p = p0(3*i-2 : 3*i);
                obj.Nodes{i}.v = v0(3*i-2 : 3*i);
                obj.Nodes{i}.R = R0(:, :, i);
                
                Ui_len = length(obj.U_list{i});
                Sigma0_cell = repmat({Sigma_0}, 1, Ui_len);
                obj.Nodes{i}.SigmaJ = blkdiag(Sigma0_cell{:});
            end
        end
        
        function predict(obj, acc_m, gyro_m)
            % 先验传播：(t) -> (t+1|t) (对应文档第 III, IV 节)
            tau = obj.dt_imu;
            I3 = eye(3); O3 = zeros(3);
            Q_local = blkdiag(obj.IMU_Sigma_a, obj.IMU_Sigma_w);
            I_num = obj.Vehicle_num;
            
            A_all = cell(I_num, 1);
            Q_all = cell(I_num, 1);
            
            for i = 1:I_num
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
                
                A_all{i} = A_i;
                Q_all{i} = (Q_i + Q_i') / 2;
            end
            
            % 按跟踪集 U_i 组装并推进联合先验协方差 (式 17)
            for i = 1:I_num
                Ui_nodes = obj.U_list{i};
                Ui_len = length(Ui_nodes);
                
                Ac_cell = cell(1, Ui_len);
                Qc_cell = cell(1, Ui_len);
                for c_idx = 1:Ui_len
                    c = Ui_nodes(c_idx);
                    Ac_cell{c_idx} = A_all{c};
                    Qc_cell{c_idx} = Q_all{c};
                end
                Abar_i = blkdiag(Ac_cell{:});
                Qbar_i = blkdiag(Qc_cell{:});
                
                SigmaJ_new = Abar_i * obj.Nodes{i}.SigmaJ * Abar_i' + Qbar_i;
                obj.Nodes{i}.SigmaJ = (SigmaJ_new + SigmaJ_new') / 2;
            end
        end
        
        function update(obj, uwb_anc, uwb_rel)
            I_num  = obj.Vehicle_num;
            U_list = obj.U_list;
            N_list = obj.N_list;
            U_set  = obj.U_set;
            
            p_prior = zeros(3, I_num);
            for i = 1:I_num
                p_prior(:, i) = obj.Nodes{i}.p;
            end
            
            % ========================================================
            % --- [1] 联合先验信息矩阵求逆与保正定合同缩放 (式 18-19) ---
            % ========================================================
            Gamma_prior_bar   = cell(I_num, 1);
            Gamma_prior_tilde = cell(I_num, 1);
            Pi_cell           = cell(I_num, 1);
            
            for i = 1:I_num
                Ui_nodes = U_list{i};
                Ui_len   = length(Ui_nodes);
                dim_i    = 9 * Ui_len;
                
                Sigma_prior_joint = obj.Nodes{i}.SigmaJ;
                Sigma_prior_joint = (Sigma_prior_joint + Sigma_prior_joint') / 2 + 1e-12 * eye(dim_i);
                G_bar = eye(dim_i) / Sigma_prior_joint;
                G_bar = (G_bar + G_bar') / 2;
                Gamma_prior_bar{i} = G_bar;
                
                % 构造合同缩放对角向量 D^i = blockdiag(|V_c|^{-1/2} I_9) (式 19)
                d_scale = zeros(dim_i, 1);
                for c_idx = 1:Ui_len
                    c = Ui_nodes(c_idx);
                    d_scale(9*c_idx-8 : 9*c_idx) = 1 / sqrt(obj.V_c_size(c));
                end
                G_tilde = (d_scale * d_scale') .* G_bar;
                Gamma_prior_tilde{i} = (G_tilde + G_tilde') / 2;
                
                Pi_cell{i} = kron(eye(Ui_len), [eye(3), zeros(3,3), zeros(3,3)]);
            end
            
            % ========================================================
            % --- [2] ADMM 变量初始化 (式 25) ---
            % ========================================================
            % Theta(:, c, i): 节点 i 对变量 c 的 9D 误差状态估计 \theta_k^{i,c}
            % Z(:, c, i)    : 节点 i 对变量 c 的 9D 共识辅助变量 z_k^{i,c}
            % Lam(:, c, i)  : 节点 i 对变量 c 的 9D 拉格朗日对偶乘子 \lambda_k^{i,c}
            Theta = zeros(9, I_num, I_num);
            Z     = zeros(9, I_num, I_num);
            Lam   = zeros(9, I_num, I_num);
            
            n_clip = 0;
            rho_val = obj.rho;
            
            % ========================================================
            % --- [3] 内层局部 GN-IEKF 与外层 Consensus ADMM 迭代 ---
            % ========================================================
            for iter = 1:obj.ADMM_iter
                Theta_next = Theta;
                Z_next     = Z;
                Lam_next   = Lam;
                
                % ---- [Step A]: 内层局部 GN-IEKF 原变量更新 (式 26-37) ----
                for i = 1:I_num
                    Ui_nodes = U_list{i};
                    Ui_len   = length(Ui_nodes);
                    if Ui_len == 0, continue; end
                    dim_i    = 9 * Ui_len;
                    Pi_mat   = Pi_cell{i};
                    
                    % 组装当前外层步 k 的局部向量 \bar{\theta}_k^i, \bar{z}_k^i, \bar{\lambda}_k^i
                    theta_m = zeros(dim_i, 1);
                    z_vec   = zeros(dim_i, 1);
                    lam_vec = zeros(dim_i, 1);
                    for c_idx = 1:Ui_len
                        c = Ui_nodes(c_idx);
                        idx_9 = 9*c_idx-8 : 9*c_idx;
                        theta_m(idx_9) = Theta(:, c, i);
                        z_vec(idx_9)   = Z(:, c, i);
                        lam_vec(idx_9) = Lam(:, c, i);
                    end
                    
                    % 内层 Gauss-Newton 循环 m = 1, ..., M_in
                    for m = 1:obj.GN_iter
                        s_m = Pi_mat * theta_m; % 3D 位置误差向量 (式 21)
                        
                        % 计算 3D UWB GN 梯度与 GN 海森矩阵 (式 26-34)
                        [g_uwb, Omega_GN] = obj.eval_local_gn(i, s_m, Ui_nodes, p_prior, uwb_anc, uwb_rel);
                        
                        % 组装 9D 总梯度与 9D 总 GN 海森矩阵 (式 35-36)
                        g_total = Gamma_prior_tilde{i} * theta_m + Pi_mat' * g_uwb ...
                                  + lam_vec + rho_val * (theta_m - z_vec);
                        H_total = Gamma_prior_tilde{i} + Pi_mat' * Omega_GN * Pi_mat ...
                                  + rho_val * eye(dim_i);
                        H_total = (H_total + H_total') / 2;
                        
                        % 求解内层 GN 步 (式 37)
                        dtheta = H_total \ g_total;
                        if any(isnan(dtheta(:))) || any(isinf(dtheta(:)))
                            dtheta = zeros(size(dtheta));
                        end
                        
                        % 步长裁剪保护 (默认 max_step = inf)
                        for c_idx = 1:Ui_len
                            idx_p = 9*c_idx-8 : 9*c_idx-6;
                            step_p = dtheta(idx_p);
                            if norm(step_p) > obj.max_step
                                dtheta(idx_p) = step_p * (obj.max_step / norm(step_p));
                                n_clip = n_clip + 1;
                            end
                        end
                        
                        theta_m = theta_m - dtheta;
                    end
                    
                    % 写回 \bar{\theta}_{k+1}^i
                    for c_idx = 1:Ui_len
                        c = Ui_nodes(c_idx);
                        Theta_next(:, c, i) = theta_m(9*c_idx-8 : 9*c_idx);
                    end
                end
                
                % ---- [Step B]: 外层 1-跳状态交换与 z, \lambda 更新 (式 38-41) ----
                max_res_primal = 0;
                for i = 1:I_num
                    Ui_nodes = U_list{i};
                    Ui_len   = length(Ui_nodes);
                    if Ui_len == 0, continue; end
                    
                    diff_vec = zeros(9 * Ui_len, 1);
                    for c_idx = 1:Ui_len
                        c = Ui_nodes(c_idx);
                        
                        % 双随机 MH 加权平均计算 z_{k+1}^{i,c} (式 40)
                        sum_z = zeros(9, 1);
                        Uc_nodes = find(U_set(:, c));
                        comm_nodes = intersect(Uc_nodes, [i, N_list{i}]);
                        for j_idx = 1:length(comm_nodes)
                            j = comm_nodes(j_idx);
                            sum_z = sum_z + obj.W_c(i, j, c) * Theta_next(:, c, j);
                        end
                        Z_next(:, c, i) = sum_z;
                        
                        % 对偶乘子更新 \lambda_{k+1}^{i,c} (式 41)
                        res_ic = Theta_next(:, c, i) - Z_next(:, c, i);
                        Lam_next(:, c, i) = Lam(:, c, i) + rho_val * res_ic;
                        diff_vec(9*c_idx-8 : 9*c_idx) = res_ic;
                    end
                    max_res_primal = max(max_res_primal, norm(diff_vec));
                end
                
                % 综合判据：兼顾状态一致性残差 ||\theta_{k+1}^i - z_{k+1}^i|| 与迭代步差
                step_diff = max(abs(Z_next(:) - Z(:)));
                err = max(max_res_primal, step_diff);
                
                Theta = Theta_next;
                Z     = Z_next;
                Lam   = Lam_next;
                
                if err < obj.epsilon
                    break;
                end
            end
            
            if obj.diag_flag
                obj.last_iter = iter;
                obj.last_err  = err;
                obj.iter_hist(end+1, 1) = iter;
                obj.res_hist(end+1, 1)  = err;
                obj.last_clip = n_clip;
                obj.clip_hist(end+1, 1) = n_clip;
            end
            if iter == obj.ADMM_iter && err >= obj.epsilon && obj.print_flag
                fprintf('警告: DIEKF 节点未在%d次内收敛, 残差=%.6f\n', obj.ADMM_iter, err);
            end
            
            % ========================================================
            % --- [4] 后验协方差更新与本车状态流形回退 (式 42-46) ---
            % ========================================================
            for i = 1:I_num
                Ui_nodes = U_list{i};
                Ui_len   = length(Ui_nodes);
                if Ui_len == 0, continue; end
                dim_i    = 9 * Ui_len;
                Pi_mat   = Pi_cell{i};
                
                % 提取收敛后的全状态误差估计 \bar{\theta}_{t|t}^i = \bar{z}_{k+1}^i
                theta_joint = zeros(dim_i, 1);
                for c_idx = 1:Ui_len
                    c = Ui_nodes(c_idx);
                    theta_joint(9*c_idx-8 : 9*c_idx) = Z(:, c, i);
                end
                
                if any(isnan(theta_joint)) || any(isinf(theta_joint))
                    theta_joint = zeros(dim_i, 1);
                end
                
                % 在收敛位置 s_{t|t}^i = \bar{\pi}^i \bar{\theta}_{t|t}^i 处评估局部 GN 信息矩阵 (式 42)
                s_final = Pi_mat * theta_joint;
                [~, Omega_GN_final] = obj.eval_local_gn(i, s_final, Ui_nodes, p_prior, uwb_anc, uwb_rel);
                
                Gamma_post = Gamma_prior_bar{i} + Pi_mat' * Omega_GN_final * Pi_mat;
                Gamma_post = (Gamma_post + Gamma_post') / 2 + 1e-10 * eye(dim_i);
                
                SigmaJ_post = eye(dim_i) / Gamma_post;
                SigmaJ_post = (SigmaJ_post + SigmaJ_post') / 2;
                
                % --- 本车名义状态回退 (Ego Retraction, 式 44-46) ---
                theta_ego = theta_joint(1:9);
                dp   = theta_ego(1:3);
                dv   = theta_ego(4:6);
                dphi = theta_ego(7:9);
                
                obj.Nodes{i}.p = obj.Nodes{i}.p + dp;
                obj.Nodes{i}.v = obj.Nodes{i}.v + dv;
                R_new = obj.Nodes{i}.R * obj.exp_SO3(dphi);
                [U_svd, ~, V_svd] = svd(R_new);
                obj.Nodes{i}.R = U_svd * V_svd';
                
                % --- ESKF 协方差重置 (Reset Jacobian) ---
                G_reset_joint = eye(dim_i);
                for c_idx = 1:Ui_len
                    dphi_c = theta_joint(9*c_idx-2 : 9*c_idx);
                    G_reset_joint(9*c_idx-2 : 9*c_idx, 9*c_idx-2 : 9*c_idx) = eye(3) - 0.5 * obj.skew(dphi_c);
                end
                
                SigmaJ_post = G_reset_joint * SigmaJ_post * G_reset_joint';
                obj.Nodes{i}.SigmaJ = (SigmaJ_post + SigmaJ_post') / 2;
            end
        end
        
        function [g_uwb, Omega_GN] = eval_local_gn(obj, i, s_vec, Ui_nodes, p_prior_all, uwb_anc, uwb_rel)
            % ==============================================================
            % 计算节点 i 本地 3D 位置空间的 GN 梯度与一阶 GN 信息矩阵 (式 26-34)
            % ==============================================================
            num_vars = length(Ui_nodes);
            g_uwb    = zeros(3 * num_vars, 1);
            Omega_GN = zeros(3 * num_vars, 3 * num_vars);
            
            idx_ego = find(Ui_nodes == i);
            r_ego   = 3*idx_ego-2 : 3*idx_ego;
            p_i     = p_prior_all(:, i) + s_vec(r_ego);
            
            % 1. 基站绝对测距 (式 26-27, 31, 33)
            sig2_anc = obj.UWB_sigma_anc^2;
            for l = 1:obj.Anchor_num
                z = uwb_anc(i, l);
                if ~isnan(z)
                    delta = p_i - obj.anchors(l, :)';
                    d = norm(delta);
                    if d < 1e-4, d = 1e-4; end
                    r_il  = z - d;
                    u_vec = delta / d;
                    
                    % 一阶 GN 梯度与海森块 (不含二阶曲率补偿项)
                    g_uwb(r_ego) = g_uwb(r_ego) - (1 / sig2_anc) * r_il * u_vec;
                    Omega_GN(r_ego, r_ego) = Omega_GN(r_ego, r_ego) + (1 / sig2_anc) * (u_vec * u_vec');
                end
            end
            
            % 2. 1-跳邻居相对测距 (式 28-30, 31-34)
            sig2_rel = obj.UWB_sigma_rel^2;
            for j_idx = 1:length(obj.N_list{i})
                j = obj.N_list{i}(j_idx);
                z = uwb_rel(i, j);
                if ~isnan(z)
                    idx_nbr = find(Ui_nodes == j);
                    r_nbr   = 3*idx_nbr-2 : 3*idx_nbr;
                    p_j     = p_prior_all(:, j) + s_vec(r_nbr);
                    
                    delta = p_i - p_j;
                    d = norm(delta);
                    if d < 1e-4, d = 1e-4; end
                    r_ij  = z - d;
                    u_vec = delta / d;
                    
                    g_ij   = (1 / sig2_rel) * r_ij * u_vec;
                    omega_ij = (1 / sig2_rel) * (u_vec * u_vec');
                    
                    g_uwb(r_ego) = g_uwb(r_ego) - g_ij;
                    g_uwb(r_nbr) = g_uwb(r_nbr) + g_ij;
                    
                    Omega_GN(r_ego, r_ego) = Omega_GN(r_ego, r_ego) + omega_ij;
                    Omega_GN(r_nbr, r_nbr) = Omega_GN(r_nbr, r_nbr) + omega_ij;
                    Omega_GN(r_ego, r_nbr) = Omega_GN(r_ego, r_nbr) - omega_ij;
                    Omega_GN(r_nbr, r_ego) = Omega_GN(r_nbr, r_ego) - omega_ij;
                end
            end
            
            Omega_GN = (Omega_GN + Omega_GN') / 2;
        end
        
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