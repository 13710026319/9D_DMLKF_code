function [trajectories, anchors] = env_setup(Vehicle_num, Anchor_num, N_steps, dt_imu, t_end)
    % 广域空间范围：120m x 120m x 15m
    % 专为凸显 MLKF 优势设计的“解析级-三维变轨蜂群交叉模型”
    
    %% 1. 基站拓扑池定义 (1-30，专为 120x120 空间优化的高低交错布局)
    all_anchors_pool = [
          0,   0,  0;  120, 120,  0;   0, 120, 15; 120,   0, 15; % 1-4: 最大体积四面体(最强3D支撑)
        120,   0,  0;    0, 120,  0; 120, 120, 15;   0,   0, 15; % 5-8: 补齐八大角点
         60,  60,  0;   60,  60, 15;   0,  60,7.5; 120,  60,7.5; % 9-12: 地面、天花板、左右墙中心
         60,   0,7.5;   60, 120,7.5;  60,   0,  0;  60, 120,  0; % 13-16: 前后墙中心，底边中点
          0,  60,  0;  120,  60,  0;  60,   0, 15;  60, 120, 15; % 17-20: 其他底边、顶边中点
          0,  60, 15;  120,  60, 15;   0,   0,7.5; 120,   0,7.5; % 21-24: 顶边中点，垂直柱中点
          0, 120,7.5;  120, 120,7.5;  30,  30,  5;  90,  90, 10; % 25-28: 剩余垂直柱中点，内部悬空点
         30,  90, 10;   90,  30,  5;                             % 29-30: 内部悬空交叉点
    ];
    % all_anchors_pool = [
    %       0,   0,  1.0;  % 1: 左下角
    %     120, 120,  1.0;  % 2: 右上角
    %       0, 120,  1.0;  % 3: 左上角
    %     120,   0,  1.0;  % 4: 右下角 (前4个构成底层大包围，但完全缺乏Z轴张力)
    %      60,  60,  1.0;  % 5: 场地正中央
    %      60,   0,  1.0;  % 6: 下边线中点
    %       0,  60,  1.0;  % 7: 左边线中点
    %     120,  60,  2.0;  % 8: 右边线中点
    %      60, 120,  1.0;  % 9: 上边线中点
    %      30,  90,  2.0;  % 10: 内部偏置点
    % ];
    if Anchor_num > 30 || Anchor_num < 1
        error('输入的 Anchor_num 超出预设范围(1-30)！');
    end
    anchors = all_anchors_pool(1:Anchor_num, :);
    
    v_rate = 2;   % 速度
    %% 2. 车辆纯解析动态穿插网络 (1-18车自适应，绝不撞车、绝不越界)
    if Vehicle_num > 18 || Vehicle_num < 1
        error('输入的 Vehicle_num 超出预设范围(1-18)！');
    end
    
    trajectories = struct();
    for n = 1:Vehicle_num
        % --- 为第 n 辆车动态分配唯一的频域和空间参数 ---
        % 1. 轨道基础半径与变化幅度
        R0 = 20 + 1.5 * n;           % 基础轨道半径：21.5m ~ 47m
        Ar = 5 + 0.2 * n;            % 轨道形变振幅：5.2m ~ 8.6m
        w_r = (0.03 + 0.002 * n) * v_rate;      % 径向收缩频率 (决定多少时间往内收一次)
        phi_r = n * (2*pi/Vehicle_num); % 径向相位
        
        % 2. 环绕中心公转参数
        dir_flag = (-1)^n;           % 奇数车顺时针，偶数车逆时针，制造迎面相遇的高视线角率
        w_a = (dir_flag * (0.015 + 0.001 * n))* v_rate; % 环绕角速度 
        phi_a = n * (pi/2);          % 初始方位角错开
        
        % 3. 三维高度分层 (确保立体分布，弥补Z轴不可观)
        Zc = 4 + mod(n, 3) * 3.5;    % 三层核心高度：4.0m, 7.5m, 11.0m
        Az = 2.0;                    % 高低起伏 2m
        w_z = (0.04 + 0.003 * n) * v_rate;      % 垂直穿越频率
        phi_z = n * (2*pi/Vehicle_num);
        
        P_true = zeros(N_steps, 3);
        V_true = zeros(N_steps, 3);
        A_true = zeros(N_steps, 3);
        Theta_true = zeros(N_steps, 1);
        
        for k = 1:N_steps
            t = (k-1) * dt_imu;
            
            % ========================================================
            % 核心：完全解析的轨迹方程与一阶/二阶导数 (杜绝一切数值积分误差)
            % ========================================================
            
            % --- 极坐标动态半径 ---
            R = R0 + Ar * sin(w_r * t + phi_r);
            R_dot = Ar * w_r * cos(w_r * t + phi_r);
            R_ddot = -Ar * w_r^2 * sin(w_r * t + phi_r);
            
            % --- 极坐标角度 ---
            alpha = w_a * t + phi_a;
            alpha_dot = w_a;
            
            % --- 1. 位置 P_true ---
            X = 60 + R * cos(alpha);
            Y = 60 + R * sin(alpha);
            Z = Zc + Az * sin(w_z * t + phi_z);
            
            % --- 2. 速度 V_true (解析导数) ---
            Vx = R_dot * cos(alpha) - R * alpha_dot * sin(alpha);
            Vy = R_dot * sin(alpha) + R * alpha_dot * cos(alpha);
            Vz = Az * w_z * cos(w_z * t + phi_z);
            
            % --- 3. 加速度 A_true (包含严格的科里奥利力与向心力项) ---
            Ax = (R_ddot - R * alpha_dot^2) * cos(alpha) - (2 * R_dot * alpha_dot) * sin(alpha);
            Ay = (R_ddot - R * alpha_dot^2) * sin(alpha) + (2 * R_dot * alpha_dot) * cos(alpha);
            Az_acc = -Az * w_z^2 * sin(w_z * t + phi_z);
            
            % --- 4. 连续平滑的偏航角 Theta_true ---
            th_raw = atan2(Vy, Vx);
            if k == 1
                th_curr = th_raw;
            else
                % 保证偏航角在 -pi 到 pi 的连续无跳变展开 (Unwrap)
                diff = th_raw - Theta_true(k-1);
                diff = mod(diff + pi, 2*pi) - pi;
                th_curr = Theta_true(k-1) + diff;
            end
            
            % --- 数据赋值 ---
            P_true(k, :) = [X, Y, Z];
            V_true(k, :) = [Vx, Vy, Vz];
            A_true(k, :) = [Ax, Ay, Az_acc];
            Theta_true(k) = th_curr;
        end

        % 封装保存
        v_name = sprintf('V%d', n);
        trajectories.(v_name).Time_true = (0:dt_imu:t_end)';
        trajectories.(v_name).X_true = P_true(:, 1);
        trajectories.(v_name).Y_true = P_true(:, 2);
        trajectories.(v_name).Z_true = P_true(:, 3);
        trajectories.(v_name).Vx_true = V_true(:, 1);
        trajectories.(v_name).Vy_true = V_true(:, 2);
        trajectories.(v_name).Vz_true = V_true(:, 3);
        trajectories.(v_name).Theta_true = Theta_true;
        trajectories.(v_name).A_true = A_true;

        R_true = zeros(3, 3, N_steps);
        for k = 1:N_steps
            th = Theta_true(k);
            R_true(:, :, k) = [cos(th), -sin(th), 0; sin(th), cos(th), 0; 0, 0, 1];
        end
        trajectories.(v_name).R_true = R_true;
    end
end