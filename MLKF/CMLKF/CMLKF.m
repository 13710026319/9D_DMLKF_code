classdef CMLKF < handle
    % CMLKF - 集中式最大似然卡尔曼滤波器 (Centralized Maximum Likelihood Kalman Filter)
    % 使用集中式高斯牛顿优化
    % 状态维度：9D x I（纯SO3姿态3、位置3、速度3，无偏置估计）
    % 对应 PDF 参考方程进行严格代码映射，具备高保真李群运算和数值稳定处理
    
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
        
        epsilon           % Gauss-Newton 迭代收敛阈值 (默认 1e-4)
        max_iter          % Gauss-Newton 最大迭代次数 (默认 10)
        Pi_mat            % 投影选择矩阵 \pi (3I x 9I)
        max_step 
        beta_inv          % LM 阻尼系数 (默认 1e-2，与 DMLKF_V1 的 beta_inv 对齐)

        is_GN             % 是否为高斯牛顿，1为C-GN, 0为C-N
    end
    
    methods
        function obj = CMLKF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0, Noise)
            % 优化函数是否为GN
            obj.is_GN = 1;

            % 构造函数：初始化系统维度、参数与初始状态
            obj.Vehicle_num = Vehicle_num;
            obj.Anchor_num = Anchor_num;
            obj.anchors = anchors;
            obj.dt_imu = dt_imu;
            obj.g_vec = [0; 0; -9.81];
            
            % 1. 模拟不可信的先验，将IMU的参数置信度稍调大
            obj.IMU_Sigma_a = (0.25)^2 * eye(3);      
            obj.IMU_Sigma_w = (0.025)^2 * eye(3);     
            obj.UWB_sigma_anc = 0.18;                  
            obj.UWB_sigma_rel = 0.18;                  

            % ==== [可选] 外部噪声参数输入 ====
            % 用法： N.IMU_Sigma_a = (0.05)^2*eye(3);  N.IMU_Sigma_w = (0.005)^2*eye(3);
            %        N.UWB_sigma_anc = 0.18;  N.UWB_sigma_rel = 0.18;
            %        kf = CMLKF(V, A, anchors, dt, p0, v0, R0, N);
            % 只覆盖传入的字段；不传（或传空）时完全保持上面的默认值，行为与以前一致。
            if nargin >= 8 && ~isempty(Noise) && isstruct(Noise)
                if isfield(Noise, 'IMU_Sigma_a'),   obj.IMU_Sigma_a   = Noise.IMU_Sigma_a;   end
                if isfield(Noise, 'IMU_Sigma_w'),   obj.IMU_Sigma_w   = Noise.IMU_Sigma_w;   end
                if isfield(Noise, 'UWB_sigma_anc'), obj.UWB_sigma_anc = Noise.UWB_sigma_anc; end
                if isfield(Noise, 'UWB_sigma_rel'), obj.UWB_sigma_rel = Noise.UWB_sigma_rel; end
            end
            obj.epsilon = 1e-4;
            obj.max_iter = 40;
            obj.max_step = Inf; 
            obj.beta_inv = 30;   

            % 2. 初始化状态
            % p0, v0 应为 3I x 1 列向量；R0 应为 3 x 3 x I 矩阵
            obj.p = p0;
            obj.v = v0;
            obj.R = R0;
            
            % 3. 初始化误差协方差矩阵 Sigma0 (9I x 9I)
            % 初始不确定度：位置0.1m, 速度0.1m/s, 姿态 1度(pi/180 rad)
            Sigma_i_0 = blkdiag((0.1^2)*eye(3), (0.1^2)*eye(3), ((pi/180)^2)*eye(3));
            Sigma_cell = repmat({Sigma_i_0}, 1, Vehicle_num);
            obj.Sigma = blkdiag(Sigma_cell{:});
            
            % 4. 构造子空间选择矩阵 Pi (\pi) (Eq 37, 38)
            % 将 9I 维误差状态映射到 3I 维位置子空间
            pi_cell = repmat({[eye(3), zeros(3,3), zeros(3,3)]}, 1, Vehicle_num);
            obj.Pi_mat = blkdiag(pi_cell{:});
        end
        
        function predict(obj, acc_m, gyro_m)
            % 预测步骤 (Prediction) - Eq 9-11, 26-32
            % acc_m: 3 x I (已去偏的真实加速度测量值)
            % gyro_m: 3 x I (已去偏的真实角速度测量值)
            
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
                
                % A. 标称状态物理积分传播 (Eq 9, 10, 11)
                Ri_new = Ri_old * obj.exp_SO3(tau * wi);
                vi_new = vi_old + tau * (Ri_old * ai + obj.g_vec);
                pi_new = pi_old + tau * vi_old + 0.5 * tau^2 * (Ri_old * ai + obj.g_vec);
                
                % B. 误差状态转移矩阵与噪声输入矩阵 (Eq 27, 28)
                a_skew = obj.skew(ai);
                A_i = [I3, tau * I3, -0.5 * tau^2 * Ri_old * a_skew;
                       O3, I3,       -tau * Ri_old * a_skew;
                       O3, O3,       obj.exp_SO3(-tau * wi)];  % exp(-tau*[w]x)
                   
                W_i = [-0.5 * tau^2 * Ri_old, O3;
                       -tau * Ri_old,         O3;
                       O3,                   -tau * obj.Jr_SO3(tau * wi)];
                   
                % 本地过程噪声矩阵 (Eq 29)
                Q_local = blkdiag(obj.IMU_Sigma_a, obj.IMU_Sigma_w);
                Q_i = W_i * Q_local * W_i';
                
                % 存入全局矩阵 (Eq 30, 31)
                A_t(idx, idx) = A_i;
                Q_t(idx, idx) = Q_i;
                
                % 更新状态
                obj.p(3*i-2 : 3*i) = pi_new;
                obj.v(3*i-2 : 3*i) = vi_new;
                obj.R(:, :, i) = Ri_new;
            end
            
            % C. 协方差传播 (Eq 32)
            obj.Sigma = A_t * obj.Sigma * A_t' + Q_t;
            obj.Sigma = (obj.Sigma + obj.Sigma') / 2; % 强制对称化，防数值漂移
        end
        
        function update(obj, uwb_anc, uwb_rel)
            % 更新与融合步骤 (Update and Fusion)
            % uwb_anc: I x K 矩阵 (含NaN代表无观测)
            % uwb_rel: I x I 矩阵 (含NaN代表无观测，自身对自身为NaN)
            
            p_prior = obj.p;       % 记录预测的先验位置 (p_{t+1|t})
            p_iter = p_prior;      % 迭代优化初始值 p^{(0)}
            
            % ==== 1. 迭代最大似然优化 (Iterative ML Optimization) ====
            for iter = 1:obj.max_iter
                y_meas = [];       % 实际观测向量 y_{t+1}
                y_pred = [];       % 预测观测向量 h(p^{(l)})
                H_l = [];          % 观测雅可比矩阵 H^{(l)}
                R_uwb_diag = [];   % 观测噪声对角元素
                
                % [条件分配]: 若为牛顿法，则分配二阶导数海森补偿矩阵
                if ~obj.is_GN
                    H_second_order = zeros(3 * obj.Vehicle_num, 3 * obj.Vehicle_num);
                end
                
                % 动态提取有效观测并构建雅可比/海森矩阵
                for i = 1:obj.Vehicle_num
                    p_i = p_iter(3*i-2 : 3*i);
                    
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
                            
                            H_row = zeros(1, 3 * obj.Vehicle_num);
                            if dist > 1e-4
                                u_ik = diff / dist;
                                H_row(3*i-2 : 3*i) = u_ik';
                                
                                % [如果为 N]: 计算基站观测的二阶导数补偿
                                if ~obj.is_GN
                                    hess_2nd = (1 / obj.UWB_sigma_anc^2) * (1 - meas/dist) * (eye(3) - u_ik * u_ik');
                                    idx_i = 3*i-2 : 3*i;
                                    H_second_order(idx_i, idx_i) = H_second_order(idx_i, idx_i) + hess_2nd;
                                end
                            end
                            H_l = [H_l; H_row];
                        end
                    end
                    
                    % (2) 相对测距观测
                    for j = 1:obj.Vehicle_num
                        if i ~= j
                            meas = uwb_rel(i, j);
                            if ~isnan(meas)
                                p_j = p_iter(3*j-2 : 3*j);
                                diff = p_i - p_j;
                                dist = norm(diff);
                                
                                y_meas = [y_meas; meas];
                                y_pred = [y_pred; dist];
                                R_uwb_diag = [R_uwb_diag; obj.UWB_sigma_rel^2];
                                
                                H_row = zeros(1, 3 * obj.Vehicle_num);
                                if dist > 1e-4
                                    u_ij = diff / dist;
                                    H_row(3*i-2 : 3*i) = u_ij';
                                    H_row(3*j-2 : 3*j) = -u_ij';
                                    
                                    % [如果为 N]: 计算相对观测的二阶导数补偿 (ii, jj, ij, ji)
                                    if ~obj.is_GN
                                        hess_2nd = (1 / obj.UWB_sigma_rel^2) * (1 - meas/dist) * (eye(3) - u_ij * u_ij');
                                        idx_i = 3*i-2 : 3*i;
                                        idx_j = 3*j-2 : 3*j;
                                        
                                        H_second_order(idx_i, idx_i) = H_second_order(idx_i, idx_i) + hess_2nd;
                                        H_second_order(idx_j, idx_j) = H_second_order(idx_j, idx_j) + hess_2nd;
                                        H_second_order(idx_i, idx_j) = H_second_order(idx_i, idx_j) - hess_2nd;
                                        H_second_order(idx_j, idx_i) = H_second_order(idx_j, idx_i) - hess_2nd;
                                    end
                                end
                                H_l = [H_l; H_row];
                            end
                        end
                    end
                end
                
                % 若无可用的有效观测，直接返回，跳过融合步骤
                if isempty(y_meas)
                    return; 
                end
                
                r_l = y_meas - y_pred;
                R_UWB_inv = spdiags(1 ./ R_uwb_diag, 0, length(y_meas), length(y_meas));
                
                % 基础的高斯-牛顿海森矩阵 (J^T * W * J)
                Omega_GN = H_l' * R_UWB_inv * H_l;
                b = H_l' * R_UWB_inv * r_l;
                
                % ==== 优化求解分支 ====
                if obj.is_GN
                    % 1. 高斯牛顿法 (CGN) + LM阻尼
                    delta_p = (Omega_GN + obj.beta_inv * eye(3 * obj.Vehicle_num)) \ b;
                else
                    % 2. 精确牛顿法 (CN)
                    Omega_Exact = Omega_GN + H_second_order;
                    Omega_Exact = (Omega_Exact + Omega_Exact') / 2; % 保证严格对称
                    
                    % 必须进行正定投影与特征值截断 (取代简单的LM阻尼加单位阵)
                    [V_eig, D_eig] = eig(Omega_Exact);
                    eig_vals = diag(D_eig);
                    eig_vals(eig_vals < obj.beta_inv) = obj.beta_inv; % 负特征值截断 & 阻尼
                    Omega_Exact_pd = V_eig * diag(eig_vals) * V_eig';
                    
                    delta_p = Omega_Exact_pd \ b;
                end

                % 增加防飞车步长限制（如果单次迭代移动超过 max_step，强制截断）
                step_norm = norm(delta_p);
                if step_norm > obj.max_step
                    delta_p = delta_p * (obj.max_step / step_norm);
                end
                
                % 位置更新
                p_iter = p_iter + delta_p;
                
                % 收敛判定
                if norm(delta_p) < obj.epsilon
                    break;
                end
            end
            
            % ==== 2. 似然信息提取 ====
            mu = p_iter - p_prior;
            
            % 【核心警告】：无论上面是 GN 还是 N，这里提取似然 Fisher 信息
            % 必须且只能用一阶雅可比 J^T * W * J，不能包含二阶导数项！
            Xi_inv = H_l' * R_UWB_inv * H_l;
            
            % ==== 3. 先验与似然融合 ====
            Lambda = obj.Pi_mat' * Xi_inv * obj.Pi_mat;
            lambda = obj.Pi_mat' * Xi_inv * mu;
            
            % 计算后验协方差 (Information Form)
            I9 = eye(9 * obj.Vehicle_num);
            Sigma_inv_prior = obj.Sigma \ I9;                     
            Sigma_post = (Sigma_inv_prior + Lambda) \ I9;         
            Sigma_post = (Sigma_post + Sigma_post') / 2;          
            
            % 全局状态校正量向量
            Delta_theta = Sigma_post * lambda;
            
            % ==== 4. 状态流形收回 (Retraction) ====
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
                
                % 数值稳定化：重正交化 (使用 SVD)
                [U, ~, V] = svd(R_new);
                obj.R(:, :, i) = U * V';
            end
            
            % 更新全局协方差矩阵
            obj.Sigma = Sigma_post;
        end
    end
    methods(Static)
        % =========================================================
        % 辅助函数：李代数和流形运算操作 (数值鲁棒版)
        % =========================================================
        
        function S = skew(v)
            % 反对称矩阵操作
            S = [   0, -v(3),  v(2);
                 v(3),     0, -v(1);
                -v(2),  v(1),     0];
        end
        
        function R = exp_SO3(v)
            % SO(3) 指数映射 (Rodrigues' Formula - Eq 7)
            theta = norm(v);
            if theta < 1e-6
                % 泰勒展开小角度近似，避免 0/0 奇异点
                R = eye(3) + CMLKF.skew(v);
            else
                n = v / theta;
                S = CMLKF.skew(n);
                R = eye(3) + sin(theta) * S + (1 - cos(theta)) * (S * S);
            end
        end
        
        function J = Jr_SO3(v)
            % SO(3) 右雅可比矩阵 (Right Jacobian - Eq 8)
            theta = norm(v);
            if theta < 1e-6
                % 泰勒展开小角度近似，避免 0/0 奇异点
                J = eye(3) - 0.5 * CMLKF.skew(v);
            else
                n = v / theta;
                S = CMLKF.skew(n);
                J = eye(3) - (1 - cos(theta)) / theta * S + (theta - sin(theta)) / theta * (S * S);
            end
        end
    end
end
