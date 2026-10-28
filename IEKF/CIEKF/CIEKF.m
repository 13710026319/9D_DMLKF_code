classdef CIEKF < handle
    % CIEKF - 集中式迭代扩展卡尔曼滤波器 (Centralized Iterated Extended Kalman Filter)
    % 状态维度：9D x I（纯SO3姿态3、位置3、速度3，无偏置估计）
    % 预测阶段与 CEKF 完全一致。
    % 更新阶段采用高斯-牛顿(Gauss-Newton)迭代优化，在当前估计点反复展开以减少非线性误差，最后使用Joseph形式更新协方差。
    
    properties
        Vehicle_num       % 无人机数量 (I)
        Anchor_num        % 基站数量 (K)
        anchors           % 基站坐标 (K x 3 矩阵)
        dt_imu            % IMU 采样时间 (tau)
        g_vec             % 重力向量
        
        % ==== 迭代优化参数 ====
        max_iter          % 最大迭代次数
        tol               % 收敛阈值
        
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
        function obj = CIEKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0)
            % 构造函数：初始化系统维度、参数与初始状态
            obj.Vehicle_num = Vehicle_num;
            obj.Anchor_num = Anchor_num;
            obj.anchors = anchors;
            obj.dt_imu = dt_imu;
            obj.g_vec = [0; 0; -9.81];
            
            % IEKF 专属迭代参数
            obj.max_iter = 10;       % 默认最大迭代次数
            obj.tol = 1e-4;         % 默认迭代收敛阈值 (状态变化量的L2范数)
            
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
            % 预测步骤 (Prediction) - 与 CEKF 完全一致
            
            tau = obj.dt_imu;
            I3 = eye(3);
            O3 = zeros(3);
            
            A_t = zeros(9 * obj.Vehicle_num, 9 * obj.Vehicle_num);
            Q_t = zeros(9 * obj.Vehicle_num, 9 * obj.Vehicle_num);
            
            for i = 1:obj.Vehicle_num
                idx = 9*(i-1) + (1:9); 
                
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
            % 更新步骤 (Update) - IEKF 高斯-牛顿迭代更新
            
            % 记录先验状态 (Prior)
            p_prior = obj.p;
            v_prior = obj.v;
            R_prior = obj.R;
            Sigma_prior = obj.Sigma; % 迭代过程中的先验协方差矩阵保持不变
            
            % 初始化当前工作状态 (Operating point)
            p_op = p_prior;
            v_op = v_prior;
            R_op = R_prior;
            
            K = [];
            H_mat = [];
            R_UWB = [];
            
            % ==== 高斯-牛顿迭代过程 ====
            for iter = 1:obj.max_iter
                y_meas = [];       
                y_pred = [];       
                H_mat = [];        
                R_uwb_diag = [];   
                
                % 1. 在当前"操作点(op)"处重新计算雅可比矩阵和残差
                for i = 1:obj.Vehicle_num
                    p_i = p_op(3*i-2 : 3*i);
                    idx_p_i = 9*(i-1) + (1:3); 
                    
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
                                p_j = p_op(3*j-2 : 3*j); % 注意：这里用的是 p_op
                                idx_p_j = 9*(j-1) + (1:3); 
                                
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
                
                % 若无可用的有效观测，退出更新
                if isempty(y_meas)
                    return; 
                end
                
                % 2. 计算当前操作点与先验状态的误差 dx_diff = op 减 prior
                dx_diff = zeros(9 * obj.Vehicle_num, 1);
                for i = 1:obj.Vehicle_num
                    idx_p = 9*(i-1) + (1:3);
                    idx_v = 9*(i-1) + (4:6);
                    idx_phi = 9*(i-1) + (7:9);
                    
                    dx_diff(idx_p) = p_op(3*i-2 : 3*i) - p_prior(3*i-2 : 3*i);
                    dx_diff(idx_v) = v_op(3*i-2 : 3*i) - v_prior(3*i-2 : 3*i);
                    % SO(3) 上的李代数差：R_op = R_prior * exp(phi)  =>  phi = log(R_prior^T * R_op)
                    dx_diff(idx_phi) = obj.log_SO3(R_prior(:,:,i)' * R_op(:,:,i));
                end
                
                % 3. 计算卡尔曼增益 K 
                r_vec = y_meas - y_pred;
                R_UWB = spdiags(R_uwb_diag, 0, length(y_meas), length(y_meas));
                
                % S = H * Sigma_prior * H' + R
                S = H_mat * Sigma_prior * H_mat' + R_UWB;
                K = (Sigma_prior * H_mat') / S;
                
                % 4. IEKF 高斯-牛顿更新步长计算
                % 推导自MAP优化： Delta = K*(r + H*dx_diff) - dx_diff
                Delta_theta = K * (r_vec + H_mat * dx_diff) - dx_diff;
                
                % 5. 更新当前工作状态 (Retraction)
                for i = 1:obj.Vehicle_num
                    idx_p = 9*(i-1) + (1:3);
                    idx_v = 9*(i-1) + (4:6);
                    idx_phi = 9*(i-1) + (7:9);
                    
                    dp = Delta_theta(idx_p);
                    dv = Delta_theta(idx_v);
                    dphi = Delta_theta(idx_phi);
                    
                    p_op(3*i-2 : 3*i) = p_op(3*i-2 : 3*i) + dp;
                    v_op(3*i-2 : 3*i) = v_op(3*i-2 : 3*i) + dv;
                    
                    R_new = R_op(:, :, i) * obj.exp_SO3(dphi);
                    % SVD 重正交化
                    [U, ~, V] = svd(R_new);
                    R_op(:, :, i) = U * V';
                end
                
                % 6. 收敛判断：如果状态更新量足够小，则提前跳出迭代
                if norm(Delta_theta) < obj.tol
                    break;
                end
            end
            
            % ==== 迭代结束后，将最优状态赋予全局对象 ====
            obj.p = p_op;
            obj.v = v_op;
            obj.R = R_op;
            
            % ==== 使用最后一次迭代得到的 H_mat 和 K 来更新协方差 ====
            % 仍采用 Joseph 形式以严格保证协方差的正定性
            I9 = eye(9 * obj.Vehicle_num);
            Temp = I9 - K * H_mat;
            obj.Sigma = Temp * Sigma_prior * Temp' + K * R_UWB * K';
            obj.Sigma = (obj.Sigma + obj.Sigma') / 2; % 强制对称
        end
    end
    
    methods(Static)
        % =========================================================
        % 辅助函数：李代数和流形运算操作
        % =========================================================
        function S = skew(v)
            S = [   0, -v(3),  v(2);
                 v(3),     0, -v(1);
                -v(2),  v(1),     0];
        end
        
        function R = exp_SO3(v)
            theta = norm(v);
            if theta < 1e-6
                R = eye(3) + CIEKF.skew(v);
            else
                n = v / theta;
                S = CIEKF.skew(n);
                R = eye(3) + sin(theta) * S + (1 - cos(theta)) * (S * S);
            end
        end
        
        function J = Jr_SO3(v)
            theta = norm(v);
            if theta < 1e-6
                J = eye(3) - 0.5 * CIEKF.skew(v);
            else
                n = v / theta;
                S = CIEKF.skew(n);
                J = eye(3) - (1 - cos(theta)) / theta * S + (theta - sin(theta)) / theta * (S * S);
            end
        end
        
        function v = log_SO3(R)
            % SO(3) 的对数映射，将旋转矩阵转换回李代数向量 (与 exp_SO3 互逆)
            tr = trace(R);
            % 数值截断，防止 trace(R) 偶尔因浮点数误差超出 [-1, 3] 从而产生复数
            tr = max(min(tr, 3), -1);
            theta = acos((tr - 1) / 2);
            
            if theta < 1e-6
                % 泰勒展开近似
                S = 0.5 * (R - R');
                v = [S(3,2); S(1,3); S(2,1)];
            else
                % 完整映射公式
                S = (theta / (2 * sin(theta))) * (R - R');
                v = [S(3,2); S(1,3); S(2,1)];
            end
        end
    end
end