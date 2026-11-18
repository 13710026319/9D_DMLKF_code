classdef CEKF < handle
    % CEKF - 经典欧氏空间集中式扩展卡尔曼滤波器 (Classic Euclidean EKF)
    % 状态维度：9D x I 欧氏向量 [p; v; eta]，其中姿态采用 3D 欧拉角 eta = [roll; pitch; yaw] 表示
    % 预测与更新全程在平直欧氏空间 R^{9I} 中采用单步一阶泰勒展开与 Joseph 协方差更新，不含 SO(3) 流形运算
    
    properties
        Vehicle_num       % 无人机数量 (I)
        Anchor_num        % 基站数量 (K)
        anchors           % 基站坐标 (K x 3 矩阵)
        dt_imu            % IMU 采样时间 (tau)
        g_vec             % 重力向量
        
        % ==== 状态变量 ====
        p                 % 全局位置向量 (3I x 1)
        v                 % 全局速度向量 (3I x 1)
        eta               % 全局欧拉角向量 (3I x 1, [phi; theta; psi])
        R                 % 全局姿态矩阵组 (3 x 3 x I, 由欧拉角实时换算以兼容外部接口)
        Sigma             % 全局欧氏误差状态协方差矩阵 (9I x 9I)
        
        % ==== 噪声与滤波器参数 ====
        IMU_Sigma_a       % 加速度计测量噪声协方差 (3x3)
        IMU_Sigma_w       % 陀螺仪测量噪声协方差 (3x3)
        UWB_sigma_anc     % 基站测距噪声标准差
        UWB_sigma_rel     % 相对测距噪声标准差
    end
    
    methods
        function obj = CEKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0)
            % 构造函数：初始化系统维度、参数与初始欧氏状态 (接口与 CMLKF 完全一致)
            obj.Vehicle_num = Vehicle_num;
            obj.Anchor_num  = Anchor_num;
            obj.anchors     = anchors;
            obj.dt_imu      = dt_imu;
            obj.g_vec       = [0; 0; -9.81];
            
            % 1. 默认噪声参数
            obj.IMU_Sigma_a   = (0.05)^2 * eye(3);      
            obj.IMU_Sigma_w   = (0.005)^2 * eye(3);     
            obj.UWB_sigma_anc = 0.075;                  
            obj.UWB_sigma_rel = 0.075;                  
            
            % 2. 初始化欧氏状态 (将输入的初始旋转矩阵 R0 转换为 3D 欧拉角向量 eta)
            obj.p   = p0(:);
            obj.v   = v0(:);
            obj.eta = zeros(3 * Vehicle_num, 1);
            obj.R   = zeros(3, 3, Vehicle_num);
            for i = 1:Vehicle_num
                eta_i = CEKF.rot_to_euler(R0(:, :, i));
                obj.eta(3*i-2 : 3*i) = eta_i;
                obj.R(:, :, i)       = CEKF.euler_to_rot(eta_i);
            end
            
            % 3. 初始化欧氏状态协方差矩阵 Sigma0 (9I x 9I)
            Sigma_i_0  = blkdiag((0.1^2)*eye(3), (0.1^2)*eye(3), ((pi/180)^2)*eye(3));
            Sigma_cell = repmat({Sigma_i_0}, 1, Vehicle_num);
            obj.Sigma  = blkdiag(Sigma_cell{:});
        end
        
        function predict(obj, acc_m, gyro_m)
            % 预测步骤 (Prediction) - 经典欧氏空间欧拉角动力学与一阶泰勒线性化
            tau = obj.dt_imu;
            I3  = eye(3);
            O3  = zeros(3);
            
            A_t = zeros(9 * obj.Vehicle_num, 9 * obj.Vehicle_num);
            Q_t = zeros(9 * obj.Vehicle_num, 9 * obj.Vehicle_num);
            
            for i = 1:obj.Vehicle_num
                idx   = 9*(i-1) + (1:9);
                idx_3 = 3*i-2 : 3*i;
                
                ai      = acc_m(:, i);
                wi      = gyro_m(:, i);
                pi_old  = obj.p(idx_3);
                vi_old  = obj.v(idx_3);
                eta_old = obj.eta(idx_3);
                
                % 当前欧拉角对应的旋转矩阵 R(eta) 与角速度转换矩阵 T(eta)
                Ri_old = CEKF.euler_to_rot(eta_old);
                Ti_old = CEKF.euler_rate_matrix(eta_old);
                
                % A. 欧氏空间状态前向欧拉积分传播
                a_nav   = Ri_old * ai + obj.g_vec;
                pi_new  = pi_old + tau * vi_old + 0.5 * tau^2 * a_nav;
                vi_new  = vi_old + tau * a_nav;
                eta_new = CEKF.wrap_euler(eta_old + tau * (Ti_old * wi));
                
                % B. 欧氏空间偏导雅可比矩阵计算
                J_rot   = CEKF.dR_a_deta(eta_old, ai);
                J_euler = CEKF.dT_w_deta(eta_old, wi);
                
                A_i = [I3, tau * I3, 0.5 * tau^2 * J_rot;
                       O3, I3,       tau * J_rot;
                       O3, O3,       I3 + tau * J_euler];
                   
                W_i = [-0.5 * tau^2 * Ri_old, O3;
                       -tau * Ri_old,         O3;
                       O3,                   -tau * Ti_old];
                   
                Q_local = blkdiag(obj.IMU_Sigma_a, obj.IMU_Sigma_w);
                Q_i     = W_i * Q_local * W_i';
                
                A_t(idx, idx) = A_i;
                Q_t(idx, idx) = Q_i;
                
                % 更新欧氏状态并同步旋转矩阵属性
                obj.p(idx_3)   = pi_new;
                obj.v(idx_3)   = vi_new;
                obj.eta(idx_3) = eta_new;
                obj.R(:, :, i) = CEKF.euler_to_rot(eta_new);
            end
            
            % C. 协方差传播
            obj.Sigma = A_t * obj.Sigma * A_t' + Q_t;
            obj.Sigma = (obj.Sigma + obj.Sigma') / 2;
        end
        
        function update(obj, uwb_anc, uwb_rel)
            % 更新步骤 (Update) - 经典欧氏空间单步 EKF 更新
            y_meas     = [];       
            y_pred     = [];       
            H_mat      = [];        
            R_uwb_diag = [];   
            
            p_prior = obj.p;
            
            % ==== 1. 在当前预测点 p_prior 处进行单次一阶展开构建雅可比矩阵 ====
            for i = 1:obj.Vehicle_num
                p_i     = p_prior(3*i-2 : 3*i);
                idx_p_i = 9*(i-1) + (1:3);
                
                % (1) 基站测距观测
                for k = 1:obj.Anchor_num
                    meas = uwb_anc(i, k);
                    if ~isnan(meas)
                        c_k  = obj.anchors(k, :)';
                        diff = p_i - c_k;
                        dist = norm(diff);
                        
                        y_meas     = [y_meas; meas];
                        y_pred     = [y_pred; dist];
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
                            p_j     = p_prior(3*j-2 : 3*j);
                            idx_p_j = 9*(j-1) + (1:3);
                            
                            diff = p_i - p_j;
                            dist = norm(diff);
                            
                            y_meas     = [y_meas; meas];
                            y_pred     = [y_pred; dist];
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
            
            if isempty(y_meas)
                return; 
            end
            
            % ==== 2. 计算卡尔曼增益 (带防奇异正则化保护) ====
            r_vec = y_meas - y_pred;
            R_UWB = spdiags(R_uwb_diag, 0, length(y_meas), length(y_meas));
            
            S     = H_mat * obj.Sigma * H_mat' + R_UWB;
            S     = (S + S') / 2;
            reg_S = max(1e-4, 1e-5 * (trace(S) / length(y_meas)));
            S_reg = S + reg_S * eye(length(y_meas));
            K     = (obj.Sigma * H_mat') / S_reg;
            
            % ==== 3. 计算欧氏状态更新量 (带防 NaN 与限幅保护) ====
            Delta_x = K * r_vec;
            if any(isnan(Delta_x)) || any(isinf(Delta_x))
                Delta_x = zeros(size(Delta_x));
            end
            
            % ==== 4. 使用 Joseph 形式更新协方差矩阵 ====
            I9        = eye(9 * obj.Vehicle_num);
            Temp      = I9 - K * H_mat;
            obj.Sigma = Temp * obj.Sigma * Temp' + K * R_UWB * K';
            obj.Sigma = (obj.Sigma + obj.Sigma') / 2;
            
            % ==== 5. 纯欧氏空间向量加法更新状态 ====
            for i = 1:obj.Vehicle_num
                idx_p   = 9*(i-1) + (1:3);
                idx_v   = 9*(i-1) + (4:6);
                idx_eta = 9*(i-1) + (7:9);
                idx_3   = 3*i-2 : 3*i;
                
                dp   = Delta_x(idx_p);
                dv   = Delta_x(idx_v);
                deta = Delta_x(idx_eta);
                
                if norm(dp) > 2.0,   dp   = dp   * (2.0 / norm(dp));   end
                if norm(dv) > 3.0,   dv   = dv   * (3.0 / norm(dv));   end
                if norm(deta) > 0.4, deta = deta * (0.4 / norm(deta)); end
                
                obj.p(idx_3)   = obj.p(idx_3) + dp;
                obj.v(idx_3)   = obj.v(idx_3) + dv;
                obj.eta(idx_3) = CEKF.wrap_euler(obj.eta(idx_3) + deta);
                obj.R(:, :, i) = CEKF.euler_to_rot(obj.eta(idx_3));
            end
        end
    end
    
    methods(Static)
        % =========================================================
        % 辅助函数：欧氏空间 Z-Y-X 欧拉角运动学、解析偏导与奇异区保护
        % =========================================================
        function R = euler_to_rot(eta)
            phi = eta(1); th = eta(2); psi = eta(3);
            cp = cos(phi); sp = sin(phi);
            ct = cos(th);  st = sin(th);
            cy = cos(psi); sy = sin(psi);
            
            R = [ct*cy, sp*st*cy - cp*sy, cp*st*cy + sp*sy;
                 ct*sy, sp*st*sy + cp*cy, cp*st*sy - sp*cy;
                 -st,   sp*ct,            cp*ct];
        end
        
        function eta = rot_to_euler(R)
            st  = max(-1, min(1, -R(3, 1)));
            th  = asin(st);
            phi = atan2(R(3, 2), R(3, 3));
            psi = atan2(R(2, 1), R(1, 1));
            eta = [phi; th; psi];
        end
        
        function T = euler_rate_matrix(eta)
            phi = eta(1); th = eta(2);
            cp = cos(phi); sp = sin(phi);
            ct = cos(th);  st = sin(th);
            % 万向节死锁区安全截断保护 (牺牲大仰角精度换取数值稳定，防止 1/ct 爆炸)
            if abs(ct) < 0.25
                ct = sign(ct + eps) * 0.25;
            end
            tt = st / ct;
            T = [1, sp * tt,  cp * tt;
                 0, cp,      -sp;
                 0, sp / ct,  cp / ct];
        end
        
        function J_rot = dR_a_deta(eta, a)
            phi = eta(1); th = eta(2); psi = eta(3);
            cp = cos(phi); sp = sin(phi);
            ct = cos(th);  st = sin(th);
            cy = cos(psi); sy = sin(psi);
            
            dR_dphi = [0,  cp*st*cy + sp*sy, -sp*st*cy + cp*sy;
                       0,  cp*st*sy - sp*cy, -sp*st*sy - cp*cy;
                       0,  cp*ct,            -sp*ct];
                   
            dR_dth  = [-st*cy, sp*ct*cy, cp*ct*cy;
                       -st*sy, sp*ct*sy, cp*ct*sy;
                       -ct,    -sp*st,   -cp*st];
                   
            dR_dpsi = [-ct*sy, -sp*st*sy - cp*cy, -cp*st*sy + sp*cy;
                        ct*cy,  sp*st*cy - cp*sy,  cp*st*cy + sp*sy;
                        0,      0,                 0];
                    
            J_rot = [dR_dphi * a, dR_dth * a, dR_dpsi * a];
        end
        
        function J_euler = dT_w_deta(eta, w)
            phi = eta(1); th = eta(2);
            cp = cos(phi); sp = sin(phi);
            ct = cos(th);  st = sin(th);
            if abs(ct) < 0.25
                ct = sign(ct + eps) * 0.25;
            end
            tt = st / ct;
            wy = w(2); wz = w(3);
            
            dTw_dphi = [(cp * wy - sp * wz) * tt;
                        -sp * wy - cp * wz;
                        (cp * wy - sp * wz) / ct];
                    
            dTw_dth  = [(sp * wy + cp * wz) / (ct^2);
                        0;
                        (sp * wy + cp * wz) * st / (ct^2)];
                    
            J_euler  = [dTw_dphi, dTw_dth, zeros(3, 1)];
        end
        
        function eta = wrap_euler(eta)
            eta(1) = mod(eta(1) + pi, 2*pi) - pi;
            eta(2) = max(-deg2rad(75), min(deg2rad(75), eta(2)));
            eta(3) = mod(eta(3) + pi, 2*pi) - pi;
        end
    end
end