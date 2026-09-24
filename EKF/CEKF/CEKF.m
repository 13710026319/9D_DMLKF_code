classdef CEKF < handle
    % CEKF - 集中式扩展卡尔曼滤波器 (Centralized Extended Kalman Filter)
    % 状态维度：9D x I（纯SO3姿态3、位置3、速度3，无偏置估计）
    % 作为 CMLKF 的严格对比基准，预测阶段相同，更新阶段采用一次泰勒展开与Joseph形式更新
    
    properties
        Vehicle_num       % 无人机数量 (I)
        Anchor_num        % 基站数量 (K)
        anchors           % 基站坐标 (K x 3 矩阵)
        dt_imu            % IMU 采样时间 (tau)
        g_vec             % 重力向量
        
        % ==== 状态变量 ====
        p                 % 全局位置向量 (3I x 1)
        v                 % 全局速度向量 (3I x 1)
        R                 % 全局姿态矩阵组 (3 x 3 x I)
        Sigma             % 全局误差状态协方差矩阵 (9I x 9I)
        
        % ==== 噪声与滤波器参数 ====
        IMU_Sigma_a       % 加速度计测量噪声协方差 (3x3)
        IMU_Sigma_w       % 陀螺仪测量噪声协方差 (3x3)
        UWB_sigma_anc     % 基站测距噪声标准差
        UWB_sigma_rel     % 相对测距噪声标准差
    end
    
    methods
        function obj = CEKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0)
            % 构造函数：初始化系统维度、参数与初始状态 (接口与CMLKF完全一致)
            obj.Vehicle_num = Vehicle_num;
            obj.Anchor_num = Anchor_num;
            obj.anchors = anchors;
            obj.dt_imu = dt_imu;
            obj.g_vec = [0; 0; -9.81];
            
            % 1. 设置合理的默认噪声参数
            obj.IMU_Sigma_a = (0.05)^2 * eye(3);      
            obj.IMU_Sigma_w = (0.005)^2 * eye(3);     
            obj.UWB_sigma_anc = 0.075;                  
            obj.UWB_sigma_rel = 0.075;                  
            
            % 2. 初始化状态
            obj.p = p0;
            obj.v = v0;
            obj.R = R0;
            
            % 3. 初始化误差协方差矩阵 Sigma0 (9I x 9I)
            Sigma_i_0 = blkdiag((0.1^2)*eye(3), (0.1^2)*eye(3), ((pi/180)^2)*eye(3));
            Sigma_cell = repmat({Sigma_i_0}, 1, Vehicle_num);
            obj.Sigma = blkdiag(Sigma_cell{:});
        end
        
        function predict(obj, acc_m, gyro_m)
            % 预测步骤 (Prediction) - 与 CMLKF 完全一致
            
            tau = obj.dt_imu;
            I3 = eye(3);
            O3 = zeros(3);
            
            A_t = zeros(9 * obj.Vehicle_num, 9 * obj.Vehicle_num);
            Q_t = zeros(9 * obj.Vehicle_num, 9 * obj.Vehicle_num);
            
            for i = 1:obj.Vehicle_num
                idx = 9*(i-1) + (1:9); % 当前车在 9I 状态向量中的索引
                
                ai = acc_m(:, i);
                wi = gyro_m(:, i);
                Ri_old = obj.R(:, :, i);
                vi_old = obj.v(3*i-2 : 3*i);
                pi_old = obj.p(3*i-2 : 3*i);
                
                % A. 标称状态物理积分传播
                Ri_new = Ri_old * obj.exp_SO3(tau * wi);
                vi_new = vi_old + tau * (Ri_old * ai + obj.g_vec);
                pi_new = pi_old + tau * vi_old + 0.5 * tau^2 * (Ri_old * ai + obj.g_vec);
                
                % B. 误差状态转移矩阵与噪声输入矩阵
                a_skew = obj.skew(ai);
                A_i = [I3, tau * I3, -0.5 * tau^2 * Ri_old * a_skew;
                       O3, I3,       -tau * Ri_old * a_skew;
                       O3, O3,       obj.exp_SO3(-tau * wi)]; 
                   
                W_i = [-0.5 * tau^2 * Ri_old, O3;
                       -tau * Ri_old,         O3;
                       O3,                   -tau * obj.Jr_SO3(tau * wi)];
                   
                % 本地过程噪声矩阵
                Q_local = blkdiag(obj.IMU_Sigma_a, obj.IMU_Sigma_w);
                Q_i = W_i * Q_local * W_i';
                
                % 存入全局矩阵
                A_t(idx, idx) = A_i;
                Q_t(idx, idx) = Q_i;
                
                % 更新状态
                obj.p(3*i-2 : 3*i) = pi_new;
                obj.v(3*i-2 : 3*i) = vi_new;
                obj.R(:, :, i) = Ri_new;
            end
            
            % C. 协方差传播
            obj.Sigma = A_t * obj.Sigma * A_t' + Q_t;
            obj.Sigma = (obj.Sigma + obj.Sigma') / 2; % 强制对称化
        end
        
        function update(obj, uwb_anc, uwb_rel)
            % 更新步骤 (Update) - 标准 EKF (一阶展开 + Joseph 形式更新)
            % uwb_anc: I x K 矩阵 
            % uwb_rel: I x I 矩阵 
            
            y_meas = [];       % 实际观测向量 y_{t+1}
            y_pred = [];       % 预测观测向量 h(\hat{p}_{t+1|t})
            H_mat = [];        % 全局观测雅可比矩阵 H (M x 9I)
            R_uwb_diag = [];   % 观测噪声对角元素
            
            p_prior = obj.p;
            
            % ==== 1. 在当前预测点处进行一阶展开构建雅可比矩阵 ====
            for i = 1:obj.Vehicle_num
                p_i = p_prior(3*i-2 : 3*i);
                idx_p_i = 9*(i-1) + (1:3); % i车位置分量在9I全局状态中的索引
                
                % (1) 基站测距观测
                for k = 1:obj.Anchor_num
                    meas = uwb_anc(i, k);
                    if ~isnan(meas)
                        c_k = obj.anchors(k, :)';
                        diff = p_i - c_k;
                        dist = norm(diff);
                        
                        y_meas = [y_meas; meas];
                        y_pred = [y_pred; dist];
                        R_uwb_diag = [R_uwb_diag; obj.UWB_sigma_anc^2];
                        
                        H_row = zeros(1, 9 * obj.Vehicle_num);
                        if dist > 1e-4
                            H_row(idx_p_i) = (diff / dist)';
                        end
                        H_mat = [H_mat; H_row];
                    end
                end
                
                % (2) 相对测距观测
                for j = 1:obj.Vehicle_num
                    if i ~= j
                        meas = uwb_rel(i, j);
                        if ~isnan(meas)
                            p_j = p_prior(3*j-2 : 3*j);
                            idx_p_j = 9*(j-1) + (1:3); % j车位置分量索引
                            
                            diff = p_i - p_j;
                            dist = norm(diff);
                            
                            y_meas = [y_meas; meas];
                            y_pred = [y_pred; dist];
                            R_uwb_diag = [R_uwb_diag; obj.UWB_sigma_rel^2];
                            
                            H_row = zeros(1, 9 * obj.Vehicle_num);
                            if dist > 1e-4
                                u_ij = diff / dist;
                                H_row(idx_p_i) = u_ij';
                                H_row(idx_p_j) = -u_ij';
                            end
                            H_mat = [H_mat; H_row];
                        end
                    end
                end
            end
            
            % 若无可用的有效观测，直接返回
            if isempty(y_meas)
                return; 
            end
            
            % ==== 2. 计算卡尔曼增益 ====
            r_vec = y_meas - y_pred;
            R_UWB = spdiags(R_uwb_diag, 0, length(y_meas), length(y_meas));
            
            % 新息协方差矩阵 S = H * Sigma * H' + R_UWB
            S = H_mat * obj.Sigma * H_mat' + R_UWB;
            
            % 计算卡尔曼增益 K (利用 MATLAB 内置求解器代替 inv(S) 保证稳定性)
            % K = Sigma * H' * inv(S) 等价于 (Sigma * H') / S
            K = (obj.Sigma * H_mat') / S; 
            
            % ==== 3. 更新误差状态 ====
            Delta_theta = K * r_vec;
            
            % ==== 4. 使用 Joseph 形式更新协方差矩阵 (严格保证正定) ====
            I9 = eye(9 * obj.Vehicle_num);
            Temp = I9 - K * H_mat;
            obj.Sigma = Temp * obj.Sigma * Temp' + K * R_UWB * K';
            obj.Sigma = (obj.Sigma + obj.Sigma') / 2; % 强制对称
            
            % ==== 5. 状态流形收回 (Retraction) ====
            for i = 1:obj.Vehicle_num
                idx_p = 9*(i-1) + (1:3);
                idx_v = 9*(i-1) + (4:6);
                idx_phi = 9*(i-1) + (7:9);
                
                dp = Delta_theta(idx_p);
                dv = Delta_theta(idx_v);
                dphi = Delta_theta(idx_phi);
                
                % 更新位置和速度
                obj.p(3*i-2 : 3*i) = obj.p(3*i-2 : 3*i) + dp;
                obj.v(3*i-2 : 3*i) = obj.v(3*i-2 : 3*i) + dv;
                
                % 姿态在 SO(3) 上的流形指数补偿
                R_new = obj.R(:, :, i) * obj.exp_SO3(dphi);
                
                % 数值稳定化：重正交化 (使用 SVD 保留最接近的合法 SO3)
                [U, ~, V] = svd(R_new);
                obj.R(:, :, i) = U * V';
            end
        end
    end
    
    methods(Static)
        % =========================================================
        % 辅助函数：李代数和流形运算操作 (与 CMLKF 一致)
        % =========================================================
        function S = skew(v)
            S = [   0, -v(3),  v(2);
                 v(3),     0, -v(1);
                -v(2),  v(1),     0];
        end
        
        function R = exp_SO3(v)
            theta = norm(v);
            if theta < 1e-6
                R = eye(3) + CEKF.skew(v);
            else
                n = v / theta;
                S = CEKF.skew(n);
                R = eye(3) + sin(theta) * S + (1 - cos(theta)) * (S * S);
            end
        end
        
        function J = Jr_SO3(v)
            theta = norm(v);
            if theta < 1e-6
                J = eye(3) - 0.5 * CEKF.skew(v);
            else
                n = v / theta;
                S = CEKF.skew(n);
                J = eye(3) - (1 - cos(theta)) / theta * S + (theta - sin(theta)) / theta * (S * S);
            end
        end
    end
end