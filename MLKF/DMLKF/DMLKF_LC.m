classdef DMLKF_LC < handle
    % DMLKF_LC - L-hop Distributed Maximum Likelihood Kalman Filter
    % 基于固定通信拓扑的 L 级通信扩展版分布式高斯-牛顿卡尔曼滤波算法
    % 集中式版，无自适应步长，用于与集中式GN (CMLKF,DMLKF_V1,DMLKF_C) 比较
    
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
        beta_inv 
        max_step 
        
        ALPHA_SAFETY = 1   % 固定步长的缩放系数，可根据实际调整
        alpha_const         % 依据拓扑结构计算出的固定步长

        diag_flag = 1       
        last_iter = 0
        last_err  = NaN
        iter_hist = []      
        res_hist  = []      
        last_clip = 0       
        clip_hist = []      
        
        print_flag = 1; 
        Nodes 
        
        % --- L-hop 固定拓扑相关量 ---
        L_hop       % 设定的通信逻辑等级
        V2V_Mask    % 物理通信连通掩码
        N_list      % N_i^{(1)}: 每个节点的物理 1 跳邻居集合
        U_list      % U_i: 每个节点的 L 跳逻辑索引集合 (ego 强制排在第1位)
        U_set       % U_set(i,c)=true 表示节点 i 在追踪变量 c
        W_c         % W_c(i,j,c): 逐变量的 MH 权重字典
        W_cd        % W_cd(i,j,c,d): 交叉边(海森块)的 MH 权重字典
        V_cd_size   % |V_{c,d}|: 全网同时追踪 (c,d) 的节点总数(预计算常数)
        
        W_global    
        lambda_2    
        is_GN       
    end
    
    methods
        function obj = DMLKF_LC(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0, V2V_Mask, max_iter, L_hop, Noise)
            obj.is_GN = 1;

            if nargin < 9 || isempty(max_iter)
                max_iter = 200;
            end
            if nargin < 10 || isempty(L_hop)
                L_hop = 1; % 默认降级为 1-hop
            end

            obj.Vehicle_num = Vehicle_num;
            obj.Anchor_num = Anchor_num;
            obj.anchors = anchors;
            obj.dt_imu = dt_imu;
            obj.g_vec = [0; 0; -9.81];
            obj.L_hop = L_hop;
            obj.V2V_Mask = V2V_Mask;
            
            % 传感器噪声基准参数
            obj.IMU_Sigma_a = (0.25)^2 * eye(3);      
            obj.IMU_Sigma_w = (0.025)^2 * eye(3);     
            obj.UWB_sigma_anc = 0.18;                  
            obj.UWB_sigma_rel = 0.18;   

            % 外部噪声参数覆盖
            if nargin >= 11 && ~isempty(Noise) && isstruct(Noise)
                if isfield(Noise, 'IMU_Sigma_a'),   obj.IMU_Sigma_a   = Noise.IMU_Sigma_a;   end
                if isfield(Noise, 'IMU_Sigma_w'),   obj.IMU_Sigma_w   = Noise.IMU_Sigma_w;   end
                if isfield(Noise, 'UWB_sigma_anc'), obj.UWB_sigma_anc = Noise.UWB_sigma_anc; end
                if isfield(Noise, 'UWB_sigma_rel'), obj.UWB_sigma_rel = Noise.UWB_sigma_rel; end
            end

            obj.max_iter = max_iter;   
            obj.epsilon  = 1e-4; 
            obj.beta_inv = 50;  
            obj.max_step = 1;  

            I_num = Vehicle_num;
            
            % =========================================================
            % [1] 固定物理拓扑与 L 跳逻辑索引集生成 (离线只算一次)
            % =========================================================
            obj.N_list = cell(I_num, 1);
            obj.U_list = cell(I_num, 1);
            obj.U_set  = false(I_num, I_num);
            
            for i = 1:I_num
                obj.N_list{i} = setdiff(find(V2V_Mask(i, :) ~= 0), i);
            end
            
            for i = 1:I_num
                N_l = obj.N_list{i}; % 初始化为 1-hop 集合
                % 递归生成 L-hop 集合
                for l = 2:obj.L_hop
                    N_next = N_l;
                    for j = N_l
                        N_next = union(N_next, obj.N_list{j});
                    end
                    N_l = setdiff(N_next, i);
                end
                u_i_nodes = [i, N_l]; % 自身(ego)必须强制排第一位
                obj.U_list{i} = u_i_nodes;
                obj.U_set(i, u_i_nodes) = true;
            end
            
            % =========================================================
            % [2] 预计算子网覆盖常数 |V_{c,d}| 
            % =========================================================
            obj.V_cd_size = zeros(I_num, I_num);
            for c = 1:I_num
                for d = 1:I_num
                    % 只统计对角块和存在物理边的块
                    if c == d || V2V_Mask(c, d) ~= 0
                        count = 0;
                        for v = 1:I_num
                            if obj.U_set(v, c) && obj.U_set(v, d)
                                count = count + 1;
                            end
                        end
                        obj.V_cd_size(c, d) = count;
                    end
                end
            end
            
            % =========================================================
            % [3] 构建单变量 MH 权重字典 W_c
            % =========================================================
            obj.W_c = zeros(I_num, I_num, I_num);
            for c = 1:I_num
                Uc_nodes = find(obj.U_set(:, c));
                for i = Uc_nodes'
                    % d_c^i = | {l \in N_i^{(1)} | c \in U_l } |
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
            % [4] 构建双变量(海森矩阵) MH 权重字典 W_cd
            % =========================================================
            obj.W_cd = zeros(I_num, I_num, I_num, I_num);
            for c = 1:I_num
                for d = 1:I_num
                    % 严格稀疏性限制：仅针对有物理测距边的变量对
                    if c ~= d && V2V_Mask(c, d) == 0
                        continue; 
                    end
                    Ucd_nodes = find(obj.U_set(:, c) & obj.U_set(:, d));
                    for i = Ucd_nodes'
                        d_cd_i = sum(obj.U_set(obj.N_list{i}, c) & obj.U_set(obj.N_list{i}, d));
                        sum_w = 0;
                        comm_neighbors = intersect(obj.N_list{i}, Ucd_nodes);
                        for j = comm_neighbors'
                            d_cd_j = sum(obj.U_set(obj.N_list{j}, c) & obj.U_set(obj.N_list{j}, d));
                            w = 1 / (1 + max(d_cd_i, d_cd_j));
                            obj.W_cd(i, j, c, d) = w;
                            sum_w = sum_w + w;
                        end
                        obj.W_cd(i, i, c, d) = 1 - sum_w;
                    end
                end
            end
            
            % --- 全局拓扑权重矩阵与代数连通度 lambda_2 ---
            Adj_global = zeros(I_num, I_num);
            for i = 1:I_num
                Adj_global(i, obj.N_list{i}) = 1;
            end
            obj.W_global = zeros(I_num, I_num);
            for i = 1:I_num
                deg_i = sum(Adj_global(i,:));
                sum_w_global = 0;
                for j = find(Adj_global(i,:))
                    deg_j = sum(Adj_global(j,:));
                    w_g = 1 / (1 + max(deg_i, deg_j));
                    obj.W_global(i,j) = w_g;
                    sum_w_global = sum_w_global + w_g;
                end
                obj.W_global(i,i) = 1 - sum_w_global;
            end
            eig_W = sort(real(eig(obj.W_global)), 'descend');
            if length(eig_W) >= 2
                obj.lambda_2 = eig_W(2);
            else
                obj.lambda_2 = 0.5;
            end
            obj.lambda_2 = max(0.01, min(0.99, obj.lambda_2));

            % 静态稳定步长
            obj.alpha_const = min(1.0, max(0.01, ...
                obj.ALPHA_SAFETY * (1 - obj.lambda_2) / (1 + sqrt(obj.lambda_2))));
            
            % --- 节点状态与联合协方差初始化 ---
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
            % 先验传播：(t) -> (t+1|t)
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
            
            % 按逻辑跟踪集 U_i 组装并整体推进巨型联合协方差
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
            I_num = obj.Vehicle_num;
            U_list = obj.U_list;
            N_list = obj.N_list;
            U_set  = obj.U_set;
            
            p_prior = zeros(3, I_num);
            for i = 1:I_num, p_prior(:, i) = obj.Nodes{i}.p; end
            
            % --- D-GN 初始化 ---
            S = zeros(3, I_num, I_num);
            G = zeros(3, I_num, I_num);
            H_mat = zeros(3, 3, I_num, I_num, I_num); 
            
            for i = 1:I_num
                if isempty(U_list{i}), continue; end
                [g_init, h_init] = obj.eval_local_cost(i, zeros(3*length(U_list{i}),1), U_list{i}, p_prior, uwb_anc, uwb_rel);
                for c_idx = 1:length(U_list{i})
                    c = U_list{i}(c_idx);
                    G(:, c, i) = g_init{c_idx};
                    for d_idx = 1:length(U_list{i})
                        d = U_list{i}(d_idx);
                        % 仅对合法物理边赋值，非边严格保持为0
                        if c == d || obj.V2V_Mask(c,d) ~= 0
                            H_mat(:, :, c, d, i) = h_init{c_idx, d_idx};
                        end
                    end
                end
            end

            alpha_c = obj.alpha_const;
            n_clip = 0;   
            
            % ========================================================
            % --- 分布式高斯-牛顿迭代 (Two-Phase Communication) ---
            % ========================================================
            for iter = 1:obj.max_iter
                S_next = S; G_next = zeros(size(G)); H_next = zeros(size(H_mat));
                
                % [Phase 1]: 状态一致性更新，计算 s_{k+1}
                for i = 1:I_num
                    Ui_len = length(U_list{i});
                    if Ui_len == 0, continue; end
                    
                    G_blk = zeros(3*Ui_len, 1);
                    H_blk = zeros(3*Ui_len, 3*Ui_len);
                    for c_idx = 1:Ui_len
                        c = U_list{i}(c_idx);
                        G_blk(3*c_idx-2 : 3*c_idx) = G(:, c, i);
                        for d_idx = 1:Ui_len
                            d = U_list{i}(d_idx);
                            if c == d || obj.V2V_Mask(c,d) ~= 0
                                H_blk(3*c_idx-2:3*c_idx, 3*d_idx-2:3*d_idx) = H_mat(:, :, c, d, i);
                            end
                        end
                    end
                    
                    H_blk = (H_blk + H_blk') / 2;
                    [V, D] = eig(H_blk);
                    eig_vals = diag(D);
                    eig_vals(eig_vals < obj.beta_inv) = obj.beta_inv;
                    H_reg = V * diag(eig_vals) * V';
                    
                    ds = H_reg \ G_blk;
                    if any(isnan(ds(:))) || any(isinf(ds(:))), ds = zeros(size(ds)); end
                    
                    for c_idx = 1:Ui_len
                        idx_r = 3*c_idx-2 : 3*c_idx;
                        step_c = ds(idx_r);
                        if norm(step_c) > obj.max_step
                            ds(idx_r) = step_c * (obj.max_step / norm(step_c));
                            n_clip = n_clip + 1;
                        end
                    end
                    
                    for c_idx = 1:Ui_len
                        c = U_list{i}(c_idx);
                        sum_s = zeros(3,1);
                        Uc_nodes = find(U_set(:, c));
                        comm_nodes = intersect(Uc_nodes, [i, N_list{i}]); 
                        for j = comm_nodes'
                            sum_s = sum_s + obj.W_c(i, j, c) * S(:, c, j);
                        end
                        S_next(:, c, i) = sum_s - alpha_c * ds(3*c_idx-2 : 3*c_idx);
                    end
                end
                
                % 评估基于最新状态 s_{k+1} 产生的局部梯度/海森差值
                g_new_all = cell(I_num,1); h_new_all = cell(I_num,1);
                g_old_all = cell(I_num,1); h_old_all = cell(I_num,1);
                for i = 1:I_num
                    if isempty(U_list{i}), continue; end
                    s_curr_vec = zeros(3*length(U_list{i}), 1);
                    s_next_vec = zeros(3*length(U_list{i}), 1);
                    for c_idx = 1:length(U_list{i})
                        c = U_list{i}(c_idx);
                        s_curr_vec(3*c_idx-2 : 3*c_idx) = S(:, c, i);
                        s_next_vec(3*c_idx-2 : 3*c_idx) = S_next(:, c, i);
                    end
                    [g_old, h_old] = obj.eval_local_cost(i, s_curr_vec, U_list{i}, p_prior, uwb_anc, uwb_rel);
                    [g_new, h_new] = obj.eval_local_cost(i, s_next_vec, U_list{i}, p_prior, uwb_anc, uwb_rel);
                    g_new_all{i} = g_new; h_new_all{i} = h_new;
                    g_old_all{i} = g_old; h_old_all{i} = h_old;
                end
                
                % [Phase 2]: 梯度和海森矩阵的动态追踪更新 (G_{k+1}, H_{k+1})
                for i = 1:I_num
                    Ui_len = length(U_list{i});
                    if Ui_len == 0, continue; end
                    
                    for c_idx = 1:Ui_len
                        c = U_list{i}(c_idx);
                        Uc_nodes = find(U_set(:, c));
                        comm_nodes_c = intersect(Uc_nodes, [i, N_list{i}]); 
                        
                        % 梯度追踪更新
                        sum_g = zeros(3,1);
                        for j = comm_nodes_c'
                            idx_c_in_j = find(U_list{j} == c);
                            term_g = G(:, c, j) + g_new_all{j}{idx_c_in_j} - g_old_all{j}{idx_c_in_j};
                            sum_g = sum_g + obj.W_c(i, j, c) * term_g;
                        end
                        G_next(:, c, i) = sum_g;
                        
                        % 海森矩阵追踪更新 (严格实施物理图稀疏限制)
                        for d_idx = 1:Ui_len
                            d = U_list{i}(d_idx);
                            if c ~= d && obj.V2V_Mask(c,d) == 0
                                continue; 
                            end
                            
                            Ud_nodes = find(U_set(:, d));
                            Ucd_nodes = intersect(Uc_nodes, Ud_nodes);
                            comm_nodes_cd = intersect(Ucd_nodes, [i, N_list{i}]);
                            
                            sum_h = zeros(3,3);
                            for j = comm_nodes_cd'
                                idx_c_j = find(U_list{j} == c);
                                idx_d_j = find(U_list{j} == d);
                                term_h = H_mat(:, :, c, d, j) + h_new_all{j}{idx_c_j, idx_d_j} - h_old_all{j}{idx_c_j, idx_d_j};
                                
                                w_cd = obj.W_cd(i, j, c, d);
                                sum_h = sum_h + w_cd * term_h;
                            end
                            H_next(:, :, c, d, i) = sum_h;
                        end
                    end
                end
                
                err = max(abs(S_next(:) - S(:)));
                S = S_next; G = G_next; H_mat = H_next;
                if err < obj.epsilon, break; end
            end

            if obj.diag_flag
                obj.last_iter = iter;
                obj.last_err  = err;
                obj.iter_hist(end+1, 1) = iter;
                obj.res_hist(end+1, 1)  = err;
                obj.last_clip = n_clip;
                obj.clip_hist(end+1, 1) = n_clip;
            end
            if iter == obj.max_iter && err >= obj.epsilon && obj.print_flag
                fprintf('警告: 节点未在%d次内收敛, 残差=%.6f\n', obj.max_iter, err);
            end
            
            % ========================================================
            % --- 5. Posterior Information Fusion & Ego Retraction ---
            % ========================================================
            Sigma_prior_cache = cell(I_num, 1);
            for c = 1:I_num
                Sigma_prior_cache{c} = obj.Nodes{c}.SigmaJ;
            end
            
            for i = 1:I_num
                Ui_len = length(U_list{i});
                if Ui_len == 0, continue; end
                
                % --- 似然信息重构与升维 ---
                Lambda_3D = zeros(3*Ui_len, 3*Ui_len);
                s_dn_vec = zeros(3*Ui_len, 1);
                for c_idx = 1:Ui_len
                    c = U_list{i}(c_idx);
                    s_dn_vec(3*c_idx-2 : 3*c_idx) = S(:, c, i);
                    
                    for d_idx = 1:Ui_len
                        d = U_list{i}(d_idx);
                        % 仅对存在物理联系的块进行重构
                        if c == d || obj.V2V_Mask(c, d) ~= 0
                            vcd_size = obj.V_cd_size(c, d);
                            Lambda_3D(3*c_idx-2:3*c_idx, 3*d_idx-2:3*d_idx) = vcd_size * H_mat(:, :, c, d, i);
                        end
                    end
                end
                Lambda_3D = (Lambda_3D + Lambda_3D') / 2; 
                
                [V_lam, D_lam] = eig(Lambda_3D);
                eig_lam = diag(D_lam);
                eig_lam(eig_lam < 0) = 0;   
                Lambda_3D = V_lam * diag(eig_lam) * V_lam';
                
                Pi_mat = kron(eye(Ui_len), [eye(3), zeros(3,3), zeros(3,3)]);
                Lambda_9D = Pi_mat' * Lambda_3D * Pi_mat;
                lambda_9D = Pi_mat' * (Lambda_3D * s_dn_vec);
                
                % --- 精度矩阵求逆与后验融合 ---
                dim_i = 9 * Ui_len;
                Sigma_prior_joint = Sigma_prior_cache{i};
                Sigma_prior_joint = (Sigma_prior_joint + Sigma_prior_joint') / 2 + 1e-12 * eye(dim_i);
                Gamma_prior = eye(dim_i) / Sigma_prior_joint;
                
                Gamma_post = Gamma_prior + Lambda_9D;
                gamma_post = lambda_9D; 
                Gamma_post = (Gamma_post + Gamma_post') / 2 + 1e-10 * eye(dim_i); 
                
                SigmaJ_post = eye(dim_i) / Gamma_post;
                SigmaJ_post = (SigmaJ_post + SigmaJ_post') / 2;
                theta_joint = Gamma_post \ gamma_post;
                
                if any(isnan(theta_joint)) || any(isinf(theta_joint))
                    theta_joint = zeros(dim_i, 1);
                    SigmaJ_post = Sigma_prior_cache{i};
                end
                
                % --- 本车名义状态回退 (Ego Retraction) ---
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
                % 为了严谨对齐流形上的切空间，依据李代数构建针对局部状态的投影矩阵
                G_reset_joint = eye(dim_i);
                for c_idx = 1:Ui_len
                    % 各自的姿态微调将引起切空间的轻微扭曲
                    dphi_c = theta_joint(9*c_idx-2 : 9*c_idx);
                    G_reset_joint(9*c_idx-2 : 9*c_idx, 9*c_idx-2 : 9*c_idx) = eye(3) - 0.5 * obj.skew(dphi_c);
                end
                
                SigmaJ_post = G_reset_joint * SigmaJ_post * G_reset_joint';
                obj.Nodes{i}.SigmaJ = (SigmaJ_post + SigmaJ_post') / 2;
            end
        end
        
        function [g_list, h_list] = eval_local_cost(obj, i, s_vec, Ui_nodes, p_prior_all, uwb_anc, uwb_rel)
            num_vars = length(Ui_nodes);
            g_list = cell(num_vars, 1);
            h_list = cell(num_vars, num_vars);
            for c=1:num_vars, g_list{c} = zeros(3,1); end
            for c=1:num_vars, for d=1:num_vars, h_list{c,d} = zeros(3,3); end; end
            
            idx_ego = find(Ui_nodes == i);
            p_i = p_prior_all(:, i) + s_vec(3*idx_ego-2 : 3*idx_ego);
            
            % 1. 基站绝对测距
            for k = 1:obj.Anchor_num
                z = uwb_anc(i, k);
                if ~isnan(z)
                    delta = p_i - obj.anchors(k, :)';
                    d = norm(delta);
                    if d < 1e-4, d = 1e-4; end
                    u_vec = delta / d; 
                    sig2 = obj.UWB_sigma_anc^2;
                    grad = (1/sig2) * (1 - z/d) * delta; 
                    
                    if obj.is_GN
                        hess = (1/sig2) * (u_vec * u_vec'); 
                    else
                        hess = (1/sig2) * (u_vec * u_vec' + (1 - z/d) * (eye(3) - u_vec * u_vec')); 
                    end
                    
                    g_list{idx_ego} = g_list{idx_ego} + grad;
                    h_list{idx_ego, idx_ego} = h_list{idx_ego, idx_ego} + hess;
                end
            end
            
            % 2. 相对节点测距 (严格限制在 1-hop 物理邻居内评估)
            for j = obj.N_list{i}'
                z = uwb_rel(i, j);
                if ~isnan(z)
                    idx_nbr = find(Ui_nodes == j);
                    p_j = p_prior_all(:, j) + s_vec(3*idx_nbr-2 : 3*idx_nbr);
                    delta = p_i - p_j;
                    d = norm(delta);
                    if d < 1e-4, d = 1e-4; end
                    u_vec = delta / d;
                    sig2 = obj.UWB_sigma_rel^2;
                    g_i =  (1/sig2) * (1 - z/d) * delta;
                    g_j = -(1/sig2) * (1 - z/d) * delta;
                    
                    if obj.is_GN
                        h_ii = (1/sig2) * (u_vec * u_vec'); 
                    else
                        h_ii = (1/sig2) * (u_vec * u_vec' + (1 - z/d) * (eye(3) - u_vec * u_vec')); 
                    end
                    
                    g_list{idx_ego} = g_list{idx_ego} + g_i;
                    g_list{idx_nbr} = g_list{idx_nbr} + g_j;
                    h_list{idx_ego, idx_ego} = h_list{idx_ego, idx_ego} + h_ii;
                    h_list{idx_nbr, idx_nbr} = h_list{idx_nbr, idx_nbr} + h_ii;
                    h_list{idx_ego, idx_nbr} = h_list{idx_ego, idx_nbr} - h_ii;
                    h_list{idx_nbr, idx_ego} = h_list{idx_nbr, idx_ego} - h_ii;
                end
            end
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