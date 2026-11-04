classdef DIEKF < handle
    % DIEKF - 9D Distributed Iterated Extended Kalman Filter
    % 纯分布式 IEKF 算法：采用 "Neighbor-as-Noisy-Anchor" 策略 + 本地高斯-牛顿(GN)迭代更新
    % 状态维度：9D（位置3、速度3、纯SO3姿态3）
    
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
        
        % 迭代控制参数
        max_iter  = 50    % 本地 GN 最大迭代次数
        epsilon   = 1e-6  % 迭代收敛阈值
        max_step  = inf   % 单步位置更新限幅
        
        % 诊断统计量
        diag_flag = 1       
        last_iter = 0
        last_err  = NaN
        iter_hist = []      
        res_hist  = []      
        
        % 分布式节点沙盒 (Sandbox)
        Nodes 
    end
    
    methods
        function obj = DIEKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0, max_iter, Noise)
            obj.Vehicle_num = Vehicle_num;
            obj.Anchor_num  = Anchor_num;
            obj.anchors     = anchors;
            obj.dt_imu      = dt_imu;
            obj.g_vec       = [0; 0; -9.81];
            
            % 默认噪声参数 (与 DEKF 保持一致)
            obj.IMU_Sigma_a   = (0.07)^2 * eye(3);      % sigma_na = 0.07
            obj.IMU_Sigma_w   = (0.007)^2 * eye(3);     % sigma_nw = 0.007
            obj.UWB_sigma_anc = 0.1;                    % sigma_anc = 0.1
            obj.UWB_sigma_rel = 0.1;                    % sigma_rel = 0.1
            
            if nargin >= 8 && ~isempty(max_iter)
                obj.max_iter = max_iter;
            end
            if nargin >= 9 && ~isempty(Noise) && isstruct(Noise)
                if isfield(Noise, 'IMU_Sigma_a'),   obj.IMU_Sigma_a   = Noise.IMU_Sigma_a;   end
                if isfield(Noise, 'IMU_Sigma_w'),   obj.IMU_Sigma_w   = Noise.IMU_Sigma_w;   end
                if isfield(Noise, 'UWB_sigma_anc'), obj.UWB_sigma_anc = Noise.UWB_sigma_anc; end
                if isfield(Noise, 'UWB_sigma_rel'), obj.UWB_sigma_rel = Noise.UWB_sigma_rel; end
            end
            
            % 实例化每个节点的本地内存
            Sigma_0 = blkdiag((0.1^2)*eye(3), (0.1^2)*eye(3), ((pi/180)^2)*eye(3));
            obj.Nodes = cell(Vehicle_num, 1);
            for i = 1:Vehicle_num
                obj.Nodes{i}.p     = p0(3*i-2 : 3*i);
                obj.Nodes{i}.v     = v0(3*i-2 : 3*i);
                obj.Nodes{i}.R     = R0(:, :, i);
                obj.Nodes{i}.Sigma = Sigma_0;
            end
        end
        
        function predict(obj, acc_m, gyro_m)
            % 独立执行本地 IMU 预测步 (与 DEKF 完全一致)
            tau = obj.dt_imu;
            I3 = eye(3); O3 = zeros(3);
            Q_local = blkdiag(obj.IMU_Sigma_a, obj.IMU_Sigma_w);
            
            for i = 1:obj.Vehicle_num
                ai = acc_m(:, i);
                wi = gyro_m(:, i);
                Ri_old = obj.Nodes{i}.R;
                vi_old = obj.Nodes{i}.v;
                pi_old = obj.Nodes{i}.p;
                
                % 标称状态积分
                Ri_new = Ri_old * obj.exp_SO3(tau * wi);
                vi_new = vi_old + tau * (Ri_old * ai + obj.g_vec);
                pi_new = pi_old + tau * vi_old + 0.5 * tau^2 * (Ri_old * ai + obj.g_vec);
                
                % 误差传递矩阵计算
                a_skew = obj.skew(ai);
                A_i = [I3, tau * I3, -0.5 * tau^2 * Ri_old * a_skew;
                       O3, I3,       -tau * Ri_old * a_skew;
                       O3, O3,       obj.exp_SO3(-tau * wi)]; 
                W_i = [-0.5 * tau^2 * Ri_old, O3;
                       -tau * Ri_old,         O3;
                       O3,                   -tau * obj.Jr_SO3(tau * wi)];
                   
                Q_i = W_i * Q_local * W_i';
                
                % 更新局部状态与协方差
                obj.Nodes{i}.p = pi_new;
                obj.Nodes{i}.v = vi_new;
                obj.Nodes{i}.R = Ri_new;
                Sigma_new = A_i * obj.Nodes{i}.Sigma * A_i' + Q_i;
                obj.Nodes{i}.Sigma = (Sigma_new + Sigma_new') / 2;
            end
        end
        
        function update(obj, uwb_anc, uwb_rel)
            I_num = obj.Vehicle_num;
            
            % --- 1. 缓存全局先验信息 (广播先验位置与 3x3 位置先验协方差) ---
            p_prior = zeros(3, I_num);
            Sigma_p_prior = cell(I_num, 1);
            for i = 1:I_num
                p_prior(:, i) = obj.Nodes{i}.p;
                Sigma_p_prior{i} = obj.Nodes{i}.Sigma(1:3, 1:3); 
            end
            
            max_iter_used = 0;
            max_err_final = 0;
            
            % --- 2. 遍历每个节点，执行完全独立的 Local GN-IEKF Update ---
            for i = 1:I_num
                p_i_prior = p_prior(:, i);
                Sigma_i   = obj.Nodes{i}.Sigma;
                
                % 初始化本地 9D 误差状态迭代变量 \theta^{(1)} = 0
                theta_m = zeros(9, 1);
                has_meas = false;
                
                for m = 1:obj.max_iter
                    % 当前迭代步的 3D 位置估计 p_i^{(m)} = \hat{p}_{t-1|t}^i + \delta p_i^{(m)}
                    p_i_curr = p_i_prior + theta_m(1:3);
                    
                    y_meas = [];
                    y_pred = [];
                    H_i    = [];
                    R_diag = [];
                    
                    % A. 基站绝对观测处理 (在当前迭代点 p_i_curr 处线性化)
                    for k = 1:obj.Anchor_num
                        z = uwb_anc(i, k);
                        if ~isnan(z)
                            delta = p_i_curr - obj.anchors(k, :)';
                            d = norm(delta);
                            if d < 1e-4, d = 1e-4; end
                            
                            y_meas = [y_meas; z];
                            y_pred = [y_pred; d];
                            
                            H_row = zeros(1, 9);
                            H_row(1:3) = (delta / d)';
                            H_i = [H_i; H_row];
                            
                            R_diag = [R_diag; obj.UWB_sigma_anc^2];
                        end
                    end
                    
                    % B. 车间相对观测处理 (Neighbor-as-Noisy-Anchor + 动态视线投影)
                    for j = 1:I_num
                        z = uwb_rel(i, j);
                        if ~isnan(z) && j ~= i
                            p_j = p_prior(:, j);
                            delta = p_i_curr - p_j;
                            d = norm(delta);
                            if d < 1e-4, d = 1e-4; end
                            
                            y_meas = [y_meas; z];
                            y_pred = [y_pred; d];
                            
                            % 当前迭代点处的单位视线向量与雅可比
                            u_ij = delta / d;
                            H_row = zeros(1, 9);
                            H_row(1:3) = u_ij';
                            H_i = [H_i; H_row];
                            
                            % 测量噪声膨胀：将邻居 j 的位置先验协方差沿当前视线方向 u_ij 投影
                            inflated_var = u_ij' * Sigma_p_prior{j} * u_ij;
                            R_eff = obj.UWB_sigma_rel^2 + inflated_var;
                            R_diag = [R_diag; R_eff];
                        end
                    end
                    
                    % 若该节点当前无任何可用观测，直接跳出
                    if isempty(y_meas)
                        has_meas = false;
                        break;
                    end
                    has_meas = true;
                    
                    % C. 高斯-牛顿 IEKF 更新步计算
                    r_vec = y_meas - y_pred;
                    R_mat = diag(R_diag);
                    
                    S = H_i * Sigma_i * H_i' + R_mat;
                    K = (Sigma_i * H_i') / S;
                    
                    % MAP 高斯-牛顿更新：当 m=1 (theta_m=0) 时退化为标准 DEKF: K * r_vec
                    theta_next = K * (r_vec + H_i * theta_m);
                    
                    % 步长限幅保护 (默认 max_step = inf)
                    dtheta = theta_next - theta_m;
                    if norm(dtheta(1:3)) > obj.max_step
                         scale = obj.max_step / norm(dtheta(1:3));
                         theta_next = theta_m + scale * dtheta;
                    end
                    
                    step_err = norm(theta_next - theta_m);
                    theta_m  = theta_next;
                    
                    max_iter_used = max(max_iter_used, m);
                    max_err_final = max(max_err_final, step_err);
                    
                    if step_err < obj.epsilon
                        break;
                    end
                end
                
                if ~has_meas
                    continue;
                end
                
                % D. 在收敛点处更新后验协方差 (使用最终迭代点重评估雅可比与 Joseph 形式)
                p_i_final = p_i_prior + theta_m(1:3);
                H_final    = [];
                R_diag_fin = [];
                for k = 1:obj.Anchor_num
                    z = uwb_anc(i, k);
                    if ~isnan(z)
                        delta = p_i_final - obj.anchors(k, :)';
                        d = max(norm(delta), 1e-4);
                        H_row = zeros(1, 9);
                        H_row(1:3) = (delta / d)';
                        H_final = [H_final; H_row];
                        R_diag_fin = [R_diag_fin; obj.UWB_sigma_anc^2];
                    end
                end
                for j = 1:I_num
                    z = uwb_rel(i, j);
                    if ~isnan(z) && j ~= i
                        delta = p_i_final - p_prior(:, j);
                        d = max(norm(delta), 1e-4);
                        u_ij = delta / d;
                        H_row = zeros(1, 9);
                        H_row(1:3) = u_ij';
                        H_final = [H_final; H_row];
                        R_eff = obj.UWB_sigma_rel^2 + u_ij' * Sigma_p_prior{j} * u_ij;
                        R_diag_fin = [R_diag_fin; R_eff];
                    end
                end
                R_mat_fin = diag(R_diag_fin);
                S_fin     = H_final * Sigma_i * H_final' + R_mat_fin;
                K_fin     = (Sigma_i * H_final') / S_fin;
                
                I9 = eye(9);
                Temp = I9 - K_fin * H_final;
                Sigma_new = Temp * Sigma_i * Temp' + K_fin * R_mat_fin * K_fin';
                Sigma_new = (Sigma_new + Sigma_new') / 2;
                
                % E. 状态流形收回 (Retraction)
                dp   = theta_m(1:3);
                dv   = theta_m(4:6);
                dphi = theta_m(7:9);
                
                obj.Nodes{i}.p = obj.Nodes{i}.p + dp;
                obj.Nodes{i}.v = obj.Nodes{i}.v + dv;
                
                R_new = obj.Nodes{i}.R * obj.exp_SO3(dphi);
                [U, ~, V] = svd(R_new);
                obj.Nodes{i}.R = U * V';
                
                obj.Nodes{i}.Sigma = Sigma_new;
            end
            
            if obj.diag_flag
                obj.last_iter = max_iter_used;
                obj.last_err  = max_err_final;
                obj.iter_hist(end+1, 1) = max_iter_used;
                obj.res_hist(end+1, 1)  = max_err_final;
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