classdef DEKF < handle
    % DEKF - 9D Distributed Extended Kalman Filter
    % 纯分布式 EKF 基准算法：采用 "Neighbor-as-Noisy-Anchor" 策略处理车间相对测距
    % 状态维度：9D（纯SO3姿态3、位置3、速度3）
    
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
        
        % 分布式节点沙盒 (Sandbox)
        Nodes 
    end
    
    methods
        function obj = DEKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0)
            % 构造函数接口与 DMLKF 保持完全一致
            obj.Vehicle_num = Vehicle_num;
            obj.Anchor_num = Anchor_num;
            obj.anchors = anchors;
            obj.dt_imu = dt_imu;
            obj.g_vec = [0; 0; -9.81];
            
            % 噪声参数 (与 DMLKF 中保持一致)
            obj.IMU_Sigma_a = (0.9)^2 * eye(3);      % sigma_na = 0.07
            obj.IMU_Sigma_w = (0.09)^2 * eye(3);     % sigma_nw = 0.007
            obj.UWB_sigma_anc = 0.4;                  % sigma_anc = 0.1
            obj.UWB_sigma_rel = 0.4;                  % sigma_rel = 0.1
            
            % 实例化每个节点的本地内存
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
            % 独立执行本地 IMU 预测步 (与 DMLKF 完全一致)
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
            
            % --- 1. 缓存全局先验信息 (切断 Data Incest 污染) ---
            % DEKF 更新时需要用到邻居的先验位置和先验协方差
            p_prior = zeros(3, I_num);
            Sigma_p_prior = cell(I_num, 1);
            for i = 1:I_num
                p_prior(:, i) = obj.Nodes{i}.p;
                % 仅提取邻居位置对应的 3x3 协方差块
                Sigma_p_prior{i} = obj.Nodes{i}.Sigma(1:3, 1:3); 
            end
            
            % --- 2. 遍历每个节点，执行完全独立的 Local EKF Update ---
            for i = 1:I_num
                p_i = p_prior(:, i);
                Sigma_i = obj.Nodes{i}.Sigma;
                
                y_meas = [];
                y_pred = [];
                H_i    = [];
                R_diag = [];
                
                % A. 基站绝对观测处理
                for k = 1:obj.Anchor_num
                    z = uwb_anc(i, k);
                    if ~isnan(z)
                        delta = p_i - obj.anchors(k, :)';
                        d = norm(delta);
                        if d < 1e-4, d = 1e-4; end
                        
                        y_meas = [y_meas; z];
                        y_pred = [y_pred; d];
                        
                        % 雅可比矩阵 (1x9)
                        H_row = zeros(1, 9);
                        H_row(1:3) = (delta / d)';
                        H_i = [H_i; H_row];
                        
                        % 测量噪声
                        R_diag = [R_diag; obj.UWB_sigma_anc^2];
                    end
                end
                
                % B. 车间相对观测处理 (Neighbor-as-Noisy-Anchor 核心机制)
                for j = 1:I_num
                    z = uwb_rel(i, j);
                    if ~isnan(z) && j ~= i
                        p_j = p_prior(:, j);
                        delta = p_i - p_j;
                        d = norm(delta);
                        if d < 1e-4, d = 1e-4; end
                        
                        y_meas = [y_meas; z];
                        y_pred = [y_pred; d];
                        
                        % 对自身位置的雅可比矩阵
                        u_ij = delta / d; % 单位方向向量
                        H_row = zeros(1, 9);
                        H_row(1:3) = u_ij';
                        H_i = [H_i; H_row];
                        
                        % 测量噪声膨胀：将邻居 j 的位置不确定度通过方向向量投影为附加测距噪声
                        inflated_var = u_ij' * Sigma_p_prior{j} * u_ij;
                        R_eff = obj.UWB_sigma_rel^2 + inflated_var;
                        R_diag = [R_diag; R_eff];
                    end
                end
                
                % 如果该节点当前没有任何观测，跳过更新
                if isempty(y_meas)
                    continue; 
                end
                
                % C. 标准 EKF 卡尔曼更新计算
                r_vec = y_meas - y_pred;
                R_mat = diag(R_diag);
                
                % 新息协方差矩阵 (Innovation Covariance)
                S = H_i * Sigma_i * H_i' + R_mat;
                
                % 卡尔曼增益 K = Sigma_i * H_i^T * S^{-1}
                % 使用 MATLAB 内置的左除 '\' 保证数值稳定性
                K = (Sigma_i * H_i') / S; 
                
                % 计算误差状态校正量
                delta_theta = K * r_vec;
                
                % D. 使用 Joseph 形式更新协方差，确保其始终对称正定
                I9 = eye(9);
                Temp = I9 - K * H_i;
                Sigma_new = Temp * Sigma_i * Temp' + K * R_mat * K';
                Sigma_new = (Sigma_new + Sigma_new') / 2; % 强制对称化
                
                % E. 状态流形收回 (Retraction)
                dp   = delta_theta(1:3);
                dv   = delta_theta(4:6);
                dphi = delta_theta(7:9);
                
                obj.Nodes{i}.p = obj.Nodes{i}.p + dp;
                obj.Nodes{i}.v = obj.Nodes{i}.v + dv;
                
                % SO(3) 姿态指数映射补偿
                R_new = obj.Nodes{i}.R * obj.exp_SO3(dphi);
                [U, ~, V] = svd(R_new); % 确保严格正交
                obj.Nodes{i}.R = U * V';
                
                obj.Nodes{i}.Sigma = Sigma_new;
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