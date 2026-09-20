classdef DMLKF < handle
    % DMLKF - 9D Distributed Maximum Likelihood Kalman Filter
    % 终极修复版：融入 GN 海森近似、时序冻结缓存、步长截断与全面舒尔补数值保护
    
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
        alpha    % 学习率/步长
        max_step % [新增防护] 单次牛顿迭代最大移动阈值(防爆墙)
        
        Nodes 
    end
    
    methods
        function obj = DMLKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0)
            obj.Vehicle_num = Vehicle_num;
            obj.Anchor_num = Anchor_num;
            obj.anchors = anchors;
            obj.dt_imu = dt_imu;
            obj.g_vec = [0; 0; -9.81];
            
            obj.IMU_Sigma_a = (0.07)^2 * eye(3);
            obj.IMU_Sigma_w = (0.007)^2 * eye(3);
            obj.UWB_sigma_anc = 0.1;
            obj.UWB_sigma_rel = 0.1;
            
            obj.max_iter = 40;
            obj.epsilon  = 0.01;
            obj.beta_inv = 0.1;  
            obj.alpha    = 1;  
            obj.max_step = 0.2;  % [防护] 每次迭代单节点最多移动 1 米
            
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
            
            % --- 1. 动态拓扑解析 ---
            U_set = false(I_num, I_num); 
            N_list = cell(I_num, 1);
            U_list = cell(I_num, 1);
            for i = 1:I_num
                neighbors = find(~isnan(uwb_rel(i, :)) | ~isnan(uwb_rel(:, i)'));
                neighbors = setdiff(neighbors, i); 
                N_list{i} = neighbors;
                u_i_nodes = unique([i, neighbors]);
                u_i_nodes = [i, setdiff(u_i_nodes, i)]; 
                U_list{i} = u_i_nodes;
                U_set(i, u_i_nodes) = true;
            end
            
            p_prior = zeros(3, I_num);
            for i = 1:I_num, p_prior(:, i) = obj.Nodes{i}.p; end
            
            % --- 2. 构建 MH 权重矩阵 W_c ---
            W_c = zeros(I_num, I_num, I_num);
            for c = 1:I_num
                Uc_nodes = find(U_set(:, c));
                for ii = 1:length(Uc_nodes)
                    i = Uc_nodes(ii);
                    deg_i = length(intersect(Uc_nodes, N_list{i}));
                    sum_w = 0;
                    neighbors_in_Uc = intersect(Uc_nodes, N_list{i});
                    for jj = 1:length(neighbors_in_Uc)
                        j = neighbors_in_Uc(jj);
                        deg_j = length(intersect(Uc_nodes, N_list{j}));
                        w = 1 / (1 + max(deg_i, deg_j));
                        W_c(i, j, c) = w;
                        sum_w = sum_w + w;
                    end
                    W_c(i, i, c) = 1 - sum_w;
                end
            end
            
            % --- 3. D-GN 初始化 ---
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
            
            % --- 4. 分布式高斯-牛顿迭代 ---
            for iter = 1:obj.max_iter
                S_next = S; G_next = zeros(size(G)); H_next = zeros(size(H_mat));

                % 让步长随迭代衰减，早期小步试探、后期逐渐加大到接近真实Newton步，可以打破振荡。
                alpha_k = obj.alpha / (1 + 0.1 * (iter - 1));
                            
                % A. 本地计算与状态共识
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
                    
                    % 确保海森块对称且正定
                    H_blk = (H_blk + H_blk') / 2;
                    [V, D] = eig(H_blk);
                    eig_vals = diag(D);
                    eig_vals(eig_vals < obj.beta_inv) = obj.beta_inv;
                    H_reg = V * diag(eig_vals) * V';
                    
                    ds = H_reg \ G_blk;
                    
                    % [防护] 防爆墙：NaN 感染阻断与步长截断 (Step Clipping)
                    if any(isnan(ds(:))) || any(isinf(ds(:))), ds = zeros(size(ds)); end
                    for c_idx = 1:Ui_len
                        idx_r = 3*c_idx-2 : 3*c_idx;
                        step_c = ds(idx_r);
                        if norm(step_c) > obj.max_step
                            ds(idx_r) = step_c * (obj.max_step / norm(step_c));
                        end
                    end
                    
                    % 状态共识
                    for c_idx = 1:Ui_len
                        c = U_list{i}(c_idx);
                        sum_s = zeros(3,1);
                        Uc_nodes = find(U_set(:, c));
                        comm_nodes = intersect(Uc_nodes, [i, N_list{i}]); 
                        for j = comm_nodes'
                            sum_s = sum_s + W_c(i, j, c) * S(:, c, j);
                        end
                        S_next(:, c, i) = sum_s - alpha_k * ds(3*c_idx-2 : 3*c_idx);
                        
                    end
                end
                
                % B. 评估新局部梯度和海森
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
                
                % C. 动态梯度与海森追踪 (严格 MH 权重)
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
                                
                                % 严格按照 Eq 31 计算跨变量的 MH 通信度
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
                
                err = max(abs(S_next(:) - S(:)));
                S = S_next; G = G_next; H_mat = H_next;
                if err < obj.epsilon, break; end
            end
            % [诊断] 若始终跑满20次仍未收敛，说明H_mat远未逼近真值，Bug4的隐患会更严重
            % if iter == obj.max_iter && err >= obj.epsilon
            %     fprintf('警告: 节点未在%d次内收敛, 残差=%.6f\n', obj.max_iter, err);
            % end
            
            % --- 5. Posterior Fusion & Schur Marginalization ---
            % [防护] 时序冻结：提前缓存当前步的所有先验，切断 Data Incest 循环污染！
            Sigma_prior_cache = cell(I_num, 1);
            for c = 1:I_num
                Sigma_prior_cache{c} = obj.Nodes{c}.Sigma;
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
                
                % [新增防护] D-GN 在有限迭代内跟踪出的网络 Hessian 未必收敛到真正的
                % 半正定网络总信息矩阵，尤其在早期/拓扑突变时可能出现负特征值。
                % 直接把这样的 Lambda 加进 Gamma_post 会导致 Sigma_ego 出现负方差。
                % 用特征值下限投影强制 Lambda_3D 半正定，从源头保证下游 Gamma_post/
                % Gamma_NN/Gamma_ego 全部自动正定（正定矩阵的主子块与 Schur 补仍正定）。
                [V_lam, D_lam] = eig(Lambda_3D);
                eig_lam = diag(D_lam);
                eig_lam(eig_lam < 0) = 0;   % 半正定投影，不吃掉真实正信息
                Lambda_3D = V_lam * diag(eig_lam) * V_lam';
                
                Pi_mat = kron(eye(Ui_len), [eye(3), zeros(3,3), zeros(3,3)]);

                Lambda_9D = Pi_mat' * Lambda_3D * Pi_mat;
                lambda_9D = Pi_mat' * (Lambda_3D * s_dn_vec);
                
                Gamma_prior = zeros(9*Ui_len, 9*Ui_len);
                for c_idx = 1:Ui_len
                    c = U_list{i}(c_idx);
                    % [防护] 从快照缓存中提取，并加入 1e-12 微底座防止奇异
                    Sigma_c = (Sigma_prior_cache{c} + Sigma_prior_cache{c}') / 2 + 1e-12 * eye(9);
                    Gamma_prior(9*c_idx-8 : 9*c_idx, 9*c_idx-8 : 9*c_idx) = eye(9) / Sigma_c;
                end
                
                Gamma_post = Gamma_prior + Lambda_9D;
                gamma_post = lambda_9D; 
                
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
                    % [防护] 舒尔补内部正则化
                    Gamma_NN_reg = Gamma_NN + 1e-8 * eye(size(Gamma_NN));
                    Gamma_ego = Gamma_ii - Gamma_iN * (Gamma_NN_reg \ Gamma_Ni);
                    gamma_ego = gamma_i - Gamma_iN * (Gamma_NN_reg \ gamma_N);
                end
                
                Gamma_ego = (Gamma_ego + Gamma_ego') / 2 + 1e-10 * eye(9); % 最终兜底
                Sigma_ego = eye(9) / Gamma_ego;
                Sigma_ego = (Sigma_ego + Sigma_ego') / 2;
                
                theta_ego = Gamma_ego \ gamma_ego;
                
                % [防护] 若算出的误差向量崩溃，退回先验状态，防止系统性宕机
                if any(isnan(theta_ego)) || any(isinf(theta_ego))
                    theta_ego = zeros(9, 1);
                    Sigma_ego = Sigma_prior_cache{i}; 
                end
                
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
        
        function [g_list, h_list] = eval_local_cost(obj, i, s_vec, Ui_nodes, p_prior_all, uwb_anc, uwb_rel)
            num_vars = length(Ui_nodes);
            g_list = cell(num_vars, 1);
            h_list = cell(num_vars, num_vars);
            for c=1:num_vars, g_list{c} = zeros(3,1); end
            for c=1:num_vars, for d=1:num_vars, h_list{c,d} = zeros(3,3); end; end
            
            idx_ego = find(Ui_nodes == i);
            p_i = p_prior_all(:, i) + s_vec(3*idx_ego-2 : 3*idx_ego);
            
            % [核心修复] 使用 Gauss-Newton 半正定海森近似，斩断负曲率陷阱
            for k = 1:obj.Anchor_num
                z = uwb_anc(i, k);
                if ~isnan(z)
                    delta = p_i - obj.anchors(k, :)';
                    d = norm(delta);
                    if d < 1e-4, d = 1e-4; end
                    
                    u_vec = delta / d; 
                    sig2 = obj.UWB_sigma_anc^2;
                    grad = (1/sig2) * (1 - z/d) * delta; 
                    hess = (1/sig2) * (u_vec * u_vec'); % GN J^T*J
                    
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
                    h_ii = (1/sig2) * (u_vec * u_vec'); % GN J^T*J
                    
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