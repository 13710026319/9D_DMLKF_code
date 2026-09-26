classdef DMLKF_C < handle
    % DMLKF - 9D Distributed Maximum Likelihood Kalman Filter
    % 该算法参数目前用于与V1 集中式GN比较
    %  1) predict(): 按 Eq18-19 整体传播 U_i 上的联合先验协方差 SigmaJ（不再各节点独立传播9x9再拼block-diag）
    %  2) update() 第5部分: 按 Eq46-54 直接对联合精度矩阵求逆重构联合后验协方差（不再用 Schur 补边缘化丢弃互相关）
    %  3) 拓扑 U_i/N_i 全程固定，相关量（W_c, W_global, lambda_2）移至构造函数只计算一次
    
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

        % [固定步长设计] 分布式 GN 步长：构造时按通信拓扑算一次，全程恒定（无自适应、无衰减）
        %   alpha = ALPHA_SAFETY * (1 - lambda_2) / (1 + sqrt(lambda_2))
        %   lambda_2 为全局拓扑权重矩阵 W_global 的代数连通度（构造时算一次），
        %   邻居数 K 的影响已经包含在 lambda_2 里，因此不需要按 K 另外调参。
        %   自适应步长（Eq 34，按曲率 s_i 动态变化的那一支）只保留在 DMLKF_D 中。
        ALPHA_SAFETY = 0.3  % 固定步长的缩放系数，0.3通常最优，也可根据实际调整
        alpha_const         % 依据拓扑结构计算出的固定步长

        % [新增-调参用] D-GN 收敛诊断（只记录，不影响计算）
        diag_flag = 1       % 1 = 记录每次 update 的迭代次数与残差
        last_iter = 0
        last_err  = NaN
        iter_hist = []      % 每次 update 实际用掉的 D-GN 迭代次数
        res_hist  = []      % 每次 update 结束时的残差 max|ΔS|
        last_clip = 0       % [新增-调参用] 最近一次 update 内被 max_step 截断的变量次数
        clip_hist = []      % [新增-调参用] 每次 update 被 max_step 截断的变量次数
        
        print_flag = 1; 
        Nodes 
        
        % [新增] 固定拓扑相关量，构造函数中一次性算好，update()/predict() 直接复用
        N_list      % 每个节点的物理邻居集合
        U_list      % 每个节点的局部索引集合 U_i = {i} U N_i（ego 恒排在第1位）
        U_set       % U_set(i,c)=true 表示节点i维护变量c
        W_c         % 逐变量 Metropolis-Hastings 权重矩阵
        W_global    % 全局拓扑权重矩阵（构造时用于计算代数连通度 lambda_2）
        lambda_2    % 全局通信图的代数连通度（第二大特征值）
    end
    
    methods
        function obj = DMLKF_C(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0, V2V_Mask, max_iter, Noise)
            % [文档修改] 新增 V2V_Mask 输入：固定的车间通信邻接矩阵（对称，1=连通）
            % 因为 Ui 全程不变，拓扑只需要在这里解析一次

            if nargin < 9 || isempty(max_iter)
                max_iter = 200;
            end

            obj.Vehicle_num = Vehicle_num;
            obj.Anchor_num = Anchor_num;
            obj.anchors = anchors;
            obj.dt_imu = dt_imu;
            obj.g_vec = [0; 0; -9.81];
            
            % 以下为VS V1或者集中式GN时的参数
            obj.IMU_Sigma_a = (0.25)^2 * eye(3);      
            obj.IMU_Sigma_w = (0.025)^2 * eye(3);     
            obj.UWB_sigma_anc = 0.18;                  
            obj.UWB_sigma_rel = 0.18;   

            % ==== [可选] 外部噪声参数输入 ====
            % 用法： N.IMU_Sigma_a = (0.05)^2*eye(3);  N.IMU_Sigma_w = (0.005)^2*eye(3);
            %        N.UWB_sigma_anc = 0.18;  N.UWB_sigma_rel = 0.18;
            %        kf = DMLKF_C(V, A, anchors, dt, p0, v0, R0, V2V_Mask, max_iter, N);
            % 只覆盖传入的字段；不传（或传空）时完全保持上面的默认值，行为与以前一致。
            if nargin >= 10 && ~isempty(Noise) && isstruct(Noise)
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

            % --- 固定分布式步长（构造时算一次，全程恒定）---
            % 与集中式 GN 对比用的这一支不做自适应步长：直接用论文 Eq33 的静态步长
            % 乘一个固定安全系数，避免步长过大导致 GN 内循环震荡不收敛。
            obj.alpha_const = min(1.0, max(0.01, ...
                obj.ALPHA_SAFETY * (1 - obj.lambda_2) / (1 + sqrt(obj.lambda_2))));
            
            % --- 节点状态与联合协方差初始化 ---
            Sigma_0 = blkdiag((0.1^2)*eye(3), (0.1^2)*eye(3), ((pi/180)^2)*eye(3));
            obj.Nodes = cell(Vehicle_num, 1);
            for i = 1:Vehicle_num
                obj.Nodes{i}.p = p0(3*i-2 : 3*i);
                obj.Nodes{i}.v = v0(3*i-2 : 3*i);
                obj.Nodes{i}.R = R0(:, :, i);
                
                % [文档修改 Eq18] SigmaJ: 节点i维护的 U_i 上的联合先验协方差
                % 初始时刻各节点误差互不相关，故用 block-diag(Sigma_0) 初始化
                Ui_len = length(obj.U_list{i});
                Sigma0_cell = repmat({Sigma_0}, 1, Ui_len);
                obj.Nodes{i}.SigmaJ = blkdiag(Sigma0_cell{:});
            end
        end
        
        function predict(obj, acc_m, gyro_m)
            % [文档修改 Eq18-19] 联合先验协方差整体传播
            % 不再是"各节点独立传播9x9再block-diag拼接丢弃互相关"，
            % 而是用 Abar_i = blockdiag(A_c)_{c∈Ui} 对整块 SigmaJ 做精确传播
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
            
            % 第二遍：[核心修改] 按 U_i 拼装 Abar_i / Qbar_i，整体传播联合协方差 SigmaJ
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
            % [文档修改] 拓扑相关量（U_list/N_list/U_set/W_c/W_global/lambda_2）
            % 已在构造函数中算好，此处直接复用，不再每次重新解析
            I_num = obj.Vehicle_num;
            U_list = obj.U_list;
            N_list = obj.N_list;
            U_set  = obj.U_set;
            W_c    = obj.W_c;
            
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
                        H_mat(:, :, c, d, i) = h_init{c_idx, d_idx};
                    end
                end
            end

            % --- 固定分布式步长：构造时按拓扑算好，全程恒定 ---
            alpha_c = obj.alpha_const;

            % --- D-GN 迭代 (Section IV，未改动，与文档一致) ---
            n_clip = 0;   % [新增-调参用] 统计 max_step 截断次数
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
                            sum_s = sum_s + W_c(i, j, c) * S(:, c, j);
                        end
                        S_next(:, c, i) = sum_s - alpha_c * ds(3*c_idx-2 : 3*c_idx);
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
                
                err = max(abs(S_next(:) - S(:)));
                S = S_next; G = G_next; H_mat = H_next;
                if err < obj.epsilon, break; end
            end

            % [新增-调参] 记录本次 update 的 D-GN 收敛情况
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
            % --- 5. Posterior Fusion [文档修改 Eq46-54] ---
            % 不再做 Schur 补边缘化，改为对联合精度矩阵直接求逆，
            % 保留完整的跨节点互相关信息，供下一步 predict() 使用
            % ========================================================
            
            % [时序冻结] 缓存本次 update 开始前每个节点各自的联合先验 SigmaJ
            % 防止同一轮循环内 i 较大的节点用到 c<i 已经被更新过的"后验"
            Sigma_prior_cache = cell(I_num, 1);
            for c = 1:I_num
                Sigma_prior_cache{c} = obj.Nodes{c}.SigmaJ;
            end
            
            for i = 1:I_num
                Ui_len = length(U_list{i});
                if Ui_len == 0, continue; end
                
                % --- Λ (似然Fisher信息) 重构，Eq 39-41，未改动 ---
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
                
                % --- [文档修改 Eq46] 联合先验精度矩阵 = 整块联合协方差直接求逆 ---
                % 不再是"逐个节点9x9分别求逆再block-diag拼接"（旧版本丢弃互相关的根源）
                dim_i = 9 * Ui_len;
                Sigma_prior_joint = Sigma_prior_cache{i};
                Sigma_prior_joint = (Sigma_prior_joint + Sigma_prior_joint') / 2 + 1e-12 * eye(dim_i);
                Gamma_prior = eye(dim_i) / Sigma_prior_joint;
                
                % --- Eq 47-48：信息可加融合 ---
                Gamma_post = Gamma_prior + Lambda_9D;
                gamma_post = lambda_9D; 
                Gamma_post = (Gamma_post + Gamma_post') / 2 + 1e-10 * eye(dim_i); % 数值兜底
                
                % --- [文档修改 Eq49-50] 直接对整块联合精度矩阵求逆，不做 Schur 补 ---
                SigmaJ_post = eye(dim_i) / Gamma_post;
                SigmaJ_post = (SigmaJ_post + SigmaJ_post') / 2;
                theta_joint = Gamma_post \ gamma_post;
                
                if any(isnan(theta_joint)) || any(isinf(theta_joint))
                    theta_joint = zeros(dim_i, 1);
                    SigmaJ_post = Sigma_prior_cache{i};
                end
                
                % --- [文档修改 Eq51] 从联合误差向量中取出 ego 子向量（ego 恒为第1位）---
                theta_ego = theta_joint(1:9);
                
                dp   = theta_ego(1:3);
                dv   = theta_ego(4:6);
                dphi = theta_ego(7:9);
                
                obj.Nodes{i}.p = obj.Nodes{i}.p + dp;
                obj.Nodes{i}.v = obj.Nodes{i}.v + dv;
                R_new = obj.Nodes{i}.R * obj.exp_SO3(dphi);
                [U, ~, V] = svd(R_new);
                obj.Nodes{i}.R = U * V';
                
                % [文档修改] 存储完整联合后验协方差（含互相关块），供下一步 predict() 使用
                obj.Nodes{i}.SigmaJ = SigmaJ_post;
            end
        end
        
        function [g_list, h_list] = eval_local_cost(obj, i, s_vec, Ui_nodes, p_prior_all, uwb_anc, uwb_rel)
            % 未改动
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
                    hess = (1/sig2) * (u_vec * u_vec'); 
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
                    h_ii = (1/sig2) * (u_vec * u_vec'); 
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
