classdef DMLKF_DN < handle
    % DMLKF - 由DMLKF_D该来的DN框架，使用论文内的自适应步长
    % 优化框架：已将原来的 Gauss-Newton (J^T*J) 近似修改为使用精确二阶导数的 Distributed Newton (DN) 框架
    
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
        
        print_flag = 1; 
        Nodes 
        
        % 固定拓扑相关量，构造函数中一次性算好，update()/predict() 直接复用
        N_list      % 每个节点的物理邻居集合
        U_list      % 每个节点的局部索引集合 U_i = {i} U N_i（ego 恒排在第1位）
        U_set       % U_set(i,c)=true 表示节点i维护变量c
        W_c         % 逐变量 Metropolis-Hastings 权重矩阵
        W_global    % 全局拓扑权重矩阵（用于 lambda_2 及步长追踪 Eq33）
        lambda_2    % 全局通信图的代数连通度（第二大特征值）
    end
    
    methods
        function obj = DMLKF_DN(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0, V2V_Mask)
            % 因为 Ui 全程不变，拓扑只需要在这里解析一次
            obj.Vehicle_num = Vehicle_num;
            obj.Anchor_num = Anchor_num;
            obj.anchors = anchors;
            obj.dt_imu = dt_imu;
            obj.g_vec = [0; 0; -9.81];
            
            obj.IMU_Sigma_a = (0.05)^2 * eye(3);
            obj.IMU_Sigma_w = (0.005)^2 * eye(3);
            obj.UWB_sigma_anc = 0.1;
            obj.UWB_sigma_rel = 0.1;
            
            obj.max_iter = 40;   
            obj.epsilon  = 0.01; 
            obj.beta_inv = 100;  
            obj.max_step = 0.06;  
            
            I_num = Vehicle_num;
            
            % --- [固定拓扑解析] 只在构造时做一次 ---
            obj.N_list = cell(I_num, 1);
            obj.U_list = cell(I_num, 1);
            obj.U_set  = false(I_num, I_num);
            for i = 1:I_num
                neighbors = find(V2V_Mask(i, :) ~= 0);
                neighbors = setdiff(neighbors, i);
                obj.N_list{i} = neighbors;
                u_i_nodes = [i, setdiff(neighbors, i)]; % ego 强制排第一位
                obj.U_list{i} = u_i_nodes;
                obj.U_set(i, u_i_nodes) = true;
            end
            
            % --- 构建 MH 权重矩阵 W_c (Eq 30) ---
            obj.W_c = zeros(I_num, I_num, I_num);
            for c = 1:I_num
                Uc_nodes = find(obj.U_set(:, c));
                for ii = 1:length(Uc_nodes)
                    i = Uc_nodes(ii);
                    deg_i = length(intersect(Uc_nodes, obj.N_list{i}));
                    sum_w = 0;
                    neighbors_in_Uc = intersect(Uc_nodes, obj.N_list{i});
                    for jj = 1:length(neighbors_in_Uc)
                        j = neighbors_in_Uc(jj);
                        deg_j = length(intersect(Uc_nodes, obj.N_list{j}));
                        w = 1 / (1 + max(deg_i, deg_j));
                        obj.W_c(i, j, c) = w;
                        sum_w = sum_w + w;
                    end
                    obj.W_c(i, i, c) = 1 - sum_w;
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
            tau = obj.dt_imu;
            I3 = eye(3); O3 = zeros(3);
            Q_local = blkdiag(obj.IMU_Sigma_a, obj.IMU_Sigma_w);
            I_num = obj.Vehicle_num;
            
            % 第一遍：算好每个节点自己的名义状态积分 + 局部 A_c, Q_c
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
            
            % 第二遍：按 U_i 拼装 Abar_i / Qbar_i，整体传播联合协方差 SigmaJ
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
            W_c    = obj.W_c;
            W_global = obj.W_global;
            lambda_2 = obj.lambda_2;
            
            p_prior = zeros(3, I_num);
            for i = 1:I_num, p_prior(:, i) = obj.Nodes{i}.p; end
            
            % --- D-Newton 初始化 ---
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
                        H_mat(:, :, c, d, i) = h_init{c_idx, d_idx};
                    end
                end
            end

            % --- 分布式步长估计初始化 (Eq 33) ---
            R_est = zeros(3, 3, I_num);
            r_prev = zeros(3, 3, I_num);
            alpha_nodes = zeros(I_num, 1);
            for i = 1:I_num
                R_est(:,:,i) = eye(3); 
                alpha_nodes(i) = (1 - lambda_2) / (1 + sqrt(lambda_2)); 
            end

            % --- 分布式牛顿 (D-Newton) 迭代 ---
            for iter = 1:obj.max_iter
                S_next = S; G_next = zeros(size(G)); H_next = zeros(size(H_mat));
                            
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
                            H_blk(3*c_idx-2:3*c_idx, 3*d_idx-2:3*d_idx) = H_mat(:, :, c, d, i);
                        end
                    end
                    
                    H_blk = (H_blk + H_blk') / 2;
                    % 【注】精确牛顿法中 H_blk 可能是非正定的，这里必须投影回正定域以保证非升方向
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
                        end
                    end
                    
                    for c_idx = 1:Ui_len
                        c = U_list{i}(c_idx);
                        sum_s = zeros(3,1);
                        Uc_nodes = find(U_set(:, c));
                        comm_nodes = intersect(Uc_nodes, [i, N_list{i}]); 
                        for j = comm_nodes'
                            sum_s = sum_s + W_c(i, j, c) * S(:, c, j);
                        end
                        S_next(:, c, i) = sum_s - alpha_nodes(i) * ds(3*c_idx-2 : 3*c_idx);
                    end
                end
                
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
                
                for i = 1:I_num
                    Ui_len = length(U_list{i});
                    if Ui_len == 0, continue; end
                    
                    for c_idx = 1:Ui_len
                        c = U_list{i}(c_idx);
                        Uc_nodes = find(U_set(:, c));
                        comm_nodes_c = intersect(Uc_nodes, [i, N_list{i}]); 
                        
                        sum_g = zeros(3,1);
                        for j = comm_nodes_c'
                            idx_c_in_j = find(U_list{j} == c);
                            term_g = G(:, c, j) + g_new_all{j}{idx_c_in_j} - g_old_all{j}{idx_c_in_j};
                            sum_g = sum_g + W_c(i, j, c) * term_g;
                        end
                        G_next(:, c, i) = sum_g;
                        
                        for d_idx = 1:Ui_len
                            d = U_list{i}(d_idx);
                            Ud_nodes = find(U_set(:, d));
                            Ucd_nodes = intersect(Uc_nodes, Ud_nodes);
                            comm_nodes_cd = intersect(Ucd_nodes, [i, N_list{i}]);
                            
                            sum_h = zeros(3,3);
                            for j = comm_nodes_cd'
                                idx_c_j = find(U_list{j} == c);
                                idx_d_j = find(U_list{j} == d);
                                term_h = H_mat(:, :, c, d, j) + h_new_all{j}{idx_c_j, idx_d_j} - h_old_all{j}{idx_c_j, idx_d_j};
                                
                                deg_i_cd = length(intersect(Ucd_nodes, N_list{i})); 
                                if j ~= i
                                    deg_j_cd = length(intersect(Ucd_nodes, N_list{j}));
                                    w_cd = 1 / (1 + max(deg_i_cd, deg_j_cd));
                                else
                                    sum_w_l = 0;
                                    neighbors_in_Ucd = intersect(Ucd_nodes, N_list{i});
                                    for l_idx = 1:length(neighbors_in_Ucd)
                                        l = neighbors_in_Ucd(l_idx);
                                        deg_l_cd = length(intersect(Ucd_nodes, N_list{l}));
                                        sum_w_l = sum_w_l + 1 / (1 + max(deg_i_cd, deg_l_cd));
                                    end
                                    w_cd = 1 - sum_w_l;
                                end
                                sum_h = sum_h + w_cd * term_h;
                            end
                            H_next(:, :, c, d, i) = sum_h;
                        end
                    end
                end
                
                % --- 分布式理论最优步长更新 ---
                R_est_next = zeros(3, 3, I_num);
                for i = 1:I_num
                    if isempty(U_list{i}), continue; end
                    idx_ego = find(U_list{i} == i);
                    if isempty(idx_ego), continue; end
                    
                    h_local_ego = h_new_all{i}{idx_ego, idx_ego}; 
                    H_cons_ego = H_next(:, :, idx_ego, idx_ego, i); 
                    H_cons_reg = H_cons_ego + obj.beta_inv * eye(3);
                    r_curr = h_local_ego / H_cons_reg; 
                    
                    if iter == 1
                        r_prev(:,:,i) = r_curr; 
                    end
                    
                    sum_R = zeros(3,3);
                    comm_nodes_global = [i, N_list{i}]; 
                    for j = comm_nodes_global
                        sum_R = sum_R + W_global(i, j) * R_est(:,:,j);
                    end
                    
                    R_est_next(:,:,i) = sum_R + r_curr - r_prev(:,:,i);
                    r_prev(:,:,i) = r_curr; 
                    
                    R_i = R_est_next(:,:,i);
                    try
                        norm_R = norm(R_i, 2);
                        norm_invR = norm(inv(R_i + 1e-8*eye(3)), 2);
                        s_i = 0.5 * (norm_R + 1 / norm_invR);
                    catch
                        s_i = 1.0;
                    end
                    s_i = max(0.1, min(10, s_i)); 
                    
                    alpha_opt = (1 - lambda_2) / (1 + sqrt(s_i * lambda_2));
                    alpha_nodes(i) = max(0.05, min(1.0, alpha_opt)); 
                end
                R_est = R_est_next;

                err = max(abs(S_next(:) - S(:)));
                S = S_next; G = G_next; H_mat = H_next;
                if err < obj.epsilon, break; end
            end
            if iter == obj.max_iter && err >= obj.epsilon && obj.print_flag
                fprintf('警告: 节点未在%d次内收敛, 残差=%.6f\n', obj.max_iter, err);
            end
            
            % ========================================================
            % --- 5. Posterior Fusion ---
            % ========================================================
            Sigma_prior_cache = cell(I_num, 1);
            for c = 1:I_num
                Sigma_prior_cache{c} = obj.Nodes{c}.SigmaJ;
            end
            
            for i = 1:I_num
                Ui_len = length(U_list{i});
                if Ui_len == 0, continue; end
                
                Lambda_3D = zeros(3*Ui_len, 3*Ui_len);
                s_dn_vec = zeros(3*Ui_len, 1);
                for c_idx = 1:Ui_len
                    c = U_list{i}(c_idx);
                    s_dn_vec(3*c_idx-2 : 3*c_idx) = S(:, c, i);
                    Uc_nodes = find(U_set(:, c));
                    for d_idx = 1:Ui_len
                        d = U_list{i}(d_idx);
                        Ud_nodes = find(U_set(:, d));
                        Ucd_size = length(intersect(Uc_nodes, Ud_nodes));
                        Lambda_3D(3*c_idx-2:3*c_idx, 3*d_idx-2:3*d_idx) = Ucd_size * H_mat(:, :, c, d, i);
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
                
                theta_ego = theta_joint(1:9);
                
                dp   = theta_ego(1:3);
                dv   = theta_ego(4:6);
                dphi = theta_ego(7:9);
                
                obj.Nodes{i}.p = obj.Nodes{i}.p + dp;
                obj.Nodes{i}.v = obj.Nodes{i}.v + dv;
                R_new = obj.Nodes{i}.R * obj.exp_SO3(dphi);
                [U, ~, V] = svd(R_new);
                obj.Nodes{i}.R = U * V';
                
                obj.Nodes{i}.SigmaJ = SigmaJ_post;
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
            
            for k = 1:obj.Anchor_num
                z = uwb_anc(i, k);
                if ~isnan(z)
                    delta = p_i - obj.anchors(k, :)';
                    d = norm(delta);
                    if d < 1e-4, d = 1e-4; end
                    u_vec = delta / d; 
                    sig2 = obj.UWB_sigma_anc^2;
                    grad = (1/sig2) * (1 - z/d) * delta; 
                    
                    % ==========================================
                    % [DGN -> DN 修改] 加入完整的二阶导数项 (1-z/d)*(I-uu^T)
                    % ==========================================
                    hess = (1/sig2) * (u_vec * u_vec' + (1 - z/d) * (eye(3) - u_vec * u_vec')); 
                    
                    g_list{idx_ego} = g_list{idx_ego} + grad;
                    h_list{idx_ego, idx_ego} = h_list{idx_ego, idx_ego} + hess;
                end
            end
            
            for j = 1:obj.Vehicle_num
                z = uwb_rel(i, j);
                if ~isnan(z) && j ~= i
                    idx_nbr = find(Ui_nodes == j);
                    p_j = p_prior_all(:, j) + s_vec(3*idx_nbr-2 : 3*idx_nbr);
                    delta = p_i - p_j;
                    d = norm(delta);
                    if d < 1e-4, d = 1e-4; end
                    u_vec = delta / d;
                    sig2 = obj.UWB_sigma_rel^2;
                    g_i =  (1/sig2) * (1 - z/d) * delta;
                    g_j = -(1/sig2) * (1 - z/d) * delta;
                    
                    % ==========================================
                    % [DGN -> DN 修改] 加入完整的二阶导数项 (1-z/d)*(I-uu^T)
                    % ==========================================
                    h_ii = (1/sig2) * (u_vec * u_vec' + (1 - z/d) * (eye(3) - u_vec * u_vec')); 
                    
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