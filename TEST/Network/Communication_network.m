% 拓扑网络生成与掩码测试脚本 (Demo)
% 总车辆数必须是偶数
clc; clear;
Vehicle_num = 10;
Anchor_num = 4; % 假设有4个基站

%% 1. 生成基站拓扑掩码 (Anchor_Mask: I x K)
% 比例: 30% Tier1, 50% Tier2, 20% Tier3
N_tier1 = round(0.3 * Vehicle_num);
N_tier2 = round(0.5 * Vehicle_num);
N_tier3 = Vehicle_num - N_tier1 - N_tier2;

Anchor_Mask = ones(Vehicle_num, Anchor_num); % 1表示保留，0表示变NaN
for i = 1:Vehicle_num
    if i <= N_tier1
        % Tier 1: 全基站 (全保留)
        % do nothing, remains 1
    elseif i <= N_tier1 + N_tier2
        % Tier 2: 部分基站 (例如仅保留偶数编号基站)
        for k = 1:Anchor_num
            if mod(k, 2) ~= 0
                Anchor_Mask(i, k) = 0;
            end
        end
    else
        % Tier 3: 纯相对测距 (无基站)
        Anchor_Mask(i, :) = 0;
    end
end

%% 2. 生成车间相对测距掩码 (V2V_Mask: I x I) (修复非对称Bug版)
K_degree = 5; % 测试奇数个邻居的情况
V2V_Mask = zeros(Vehicle_num, Vehicle_num);

% 提取对称部分
K_half = floor(K_degree / 2); 

for i = 1:Vehicle_num
    % 1. 绝对对称地连接前后各 K_half 个节点
    for d = 1:K_half
        idx_forward = mod(i + d - 1, Vehicle_num) + 1;
        idx_backward = mod(i - d - 1, Vehicle_num) + 1;
        V2V_Mask(i, idx_forward) = 1;
        V2V_Mask(i, idx_backward) = 1;
    end
    
    % 2. 处理 K 为奇数的情况：连接圆环正对面的节点
    if mod(K_degree, 2) ~= 0
        if mod(Vehicle_num, 2) ~= 0
            error('图论限制: 当每辆车的邻居数 K_degree 为奇数时，车辆总数 Vehicle_num 必须为偶数才能构成对称拓扑！');
        end
        % 找到对径节点 (Diametrically opposite node)
        idx_opposite = mod(i + Vehicle_num/2 - 1, Vehicle_num) + 1;
        V2V_Mask(i, idx_opposite) = 1;
    end
end

% 确保对角线自身到自身不通信 (为0)
V2V_Mask(logical(eye(Vehicle_num))) = 0;

% 安全校验：检查矩阵是否已经完全对称
if ~isequal(V2V_Mask, V2V_Mask')
    warning('V2V掩码矩阵非对称！请检查算法逻辑。');
else
    fprintf('检查通过：V2V掩码矩阵已完全对称。\n');
end

%% 3. 可视化打印拓扑结果
fprintf('=== 基站连接掩码 (Anchor Mask) ===\n');
fprintf(' (1: 有连接, 0: NaN无连接)\n');
disp(Anchor_Mask);

fprintf('=== 车间相对测距掩码 (V2V Mask) ===\n');
fprintf(' (1: 有连接, 0: NaN无连接)\n');
disp(V2V_Mask);

% 模拟实际应用 (在 DMLKF_Test.m 中提取数据时的做法)：
% 假设从数据集拿到了当前帧的全连接相对测距 uwb_rel_raw
uwb_rel_raw = rand(Vehicle_num, Vehicle_num) * 10 + 30; 
uwb_rel_raw(logical(eye(Vehicle_num))) = NaN;

% 应用掩码，把非连通边全部置为 NaN
uwb_rel_masked = uwb_rel_raw;
uwb_rel_masked(V2V_Mask == 0) = NaN;

% 这就是你要喂给 DMLKF 类的输入！
fprintf('=== 喂给算法的车间测距矩阵 (uwb_rel_masked) ===\n');
disp(uwb_rel_masked);