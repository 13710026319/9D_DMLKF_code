function [trajectories, anchors] = env_setup(Vehicle_num, Anchor_num, N_steps, dt_imu, t_end)
    % 广域空间范围：120m x 120m x 10m
    % 专为凸显 MLKF 优势设计的“极端共面-浅仰角非线性测试模型”
    
    %% 1. 基站拓扑池定义 (所有基站强制 Z = 10m，制造极浅的仰角)
    all_anchors_pool = [
          0,   0, 10.0;  % 1: 左下角
        120, 120, 10.0;  % 2: 右上角
          0, 120, 10.0;  % 3: 左上角
        120,   0, 10.0;  % 4: 右下角 
         60,  60, 10.0;  % 5: 场地正中央
         60,   0, 10.0;  % 6: 下边线中点
          0,  60, 10.0;  % 7: 左边线中点
        120,  60, 10.0;  % 8: 右边线中点
         60, 120, 10.0;  % 9: 上边线中点
         30,  90, 10.0;  % 10: 内部偏置点
         90,  30, 10.0;  % 11: 内部偏置点
         30,  30, 10.0;  % 12: 内部偏置点
         90,  90, 10.0;  % 13: 内部偏置点
         20,  60, 10.0;  % 14: 额外补充
         60,  20, 10.0;  % 15: 额外补充
        100,  60, 10.0;  % 16: 额外补充
         60, 100, 10.0;  % 17: 额外补充
         40,  40, 10.0;  % 18: 额外补充
         80,  80, 10.0;  % 19: 额外补充
         80,  40, 10.0;  % 20: 额外补充
    ];
    if Anchor_num > 20 || Anchor_num < 4
        error('输入的 Anchor_num 建议在(4-20)范围内以符合极简测试需求！');
    end
    anchors = all_anchors_pool(1:Anchor_num, :);
    
    v_rate = 2;   % 速度乘子
    
    %% 2. 车辆纯解析动态穿插网络
    if Vehicle_num > 18 || Vehicle_num < 4
        error('输入的 Vehicle_num 建议在(4-18)范围内！');
    end
    
    trajectories = struct();
    for n = 1:Vehicle_num
        % --- 为第 n 辆车动态分配唯一的频域和空间参数 ---
        R0 = 20 + 1.5 * n;           % 基础轨道半径：21.5m ~ 47m
        Ar = 5 + 0.2 * n;            % 轨道形变振幅：5.2m ~ 8.6m
        w_r = (0.03 + 0.002 * n) * v_rate;      
        phi_r = n * (2*pi/Vehicle_num); 
        
        dir_flag = (-1)^n;           
        w_a = (dir_flag * (0.015 + 0.001 * n))* v_rate; 
        phi_a = n * (pi/2);          
        
        % =======================================================
        % 【修改核心】：Z轴被极端压扁
        % 车辆高度基础为 0.15m，振幅 0.15m，严格波动在 0 ~ 0.3m 之间
        % =======================================================
        Zc = 0.15;                   
        Az = 0.15;                    
        w_z = (0.04 + 0.003 * n) * v_rate;      
        phi_z = n * (2*pi/Vehicle_num);
        
        P_true = zeros(N_steps, 3);
        V_true = zeros(N_steps, 3);
        A_true = zeros(N_steps, 3);
        Theta_true = zeros(N_steps, 1);
        
        for k = 1:N_steps
            t = (k-1) * dt_imu;
            
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
            Z = Zc + Az * sin(w_z * t + phi_z); % Z轴在 0~0.3m 内变动
            
            % --- 2. 速度 V_true (解析导数) ---
            Vx = R_dot * cos(alpha) - R * alpha_dot * sin(alpha);
            Vy = R_dot * sin(alpha) + R * alpha_dot * cos(alpha);
            Vz = Az * w_z * cos(w_z * t + phi_z);
            
            % --- 3. 加速度 A_true ---
            Ax = (R_ddot - R * alpha_dot^2) * cos(alpha) - (2 * R_dot * alpha_dot) * sin(alpha);
            Ay = (R_ddot - R * alpha_dot^2) * sin(alpha) + (2 * R_dot * alpha_dot) * cos(alpha);
            Az_acc = -Az * w_z^2 * sin(w_z * t + phi_z);
            
            % --- 4. 连续平滑的偏航角 Theta_true ---
            th_raw = atan2(Vy, Vx);
            if k == 1
                th_curr = th_raw;
            else
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