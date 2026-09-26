function out_png = iter_compare_plot(csv_file, out_png, K, V_num, A_num, data_ratio, show_fit)
%ITER_COMPARE_PLOT 任务2 曲线：位置 RMSE 随 GN 迭代上限变化（含 V1 集中式基准线）
%
%   直接运行（用默认路径）：
%       iter_compare_plot
%   或指定：
%       iter_compare_plot('...\RESULT\iter_compare.csv', '...\RESULT\iter_compare.png', 4, 8, 4, 0.2)
%
%   横轴：D-GN 迭代上限 max_iter（对数刻度），纵轴：位置 RMSE（m）。
%   红色实线 = 分布式 GN（DMLKF_C），蓝色虚线 = V1 集中式 GN 基准。
%   图例与标注全部为英文。
%
%   show_fit = true 时，额外叠加一条"收敛趋势拟合曲线"
%   （rmse = c - a*exp(-n/tau)，用 fminsearch 拟合）：原始测量点仍然保留，
%   拟合线在图例里明确标注为 fit，不冒充测量结果。

this_dir = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(csv_file), csv_file = fullfile(this_dir, 'RESULT', 'iter_compare.csv'); end
if nargin < 2, out_png = fullfile(this_dir, 'RESULT', 'iter_compare.png'); end   % 传 '' 表示只画不存
if nargin < 3 || isempty(K),          K = 4;   end
if nargin < 4 || isempty(V_num),      V_num = 8;   end
if nargin < 5 || isempty(A_num),      A_num = 4;   end
if nargin < 6 || isempty(data_ratio), data_ratio = 0.2; end
if nargin < 7 || isempty(show_fit),    show_fit = false; end

if ~exist(csv_file, 'file'), error('找不到结果 CSV: %s', csv_file); end
T = readtable(csv_file, 'VariableNamingRule', 'preserve', 'TextType', 'string');

is_v1 = T.algorithm == "V1";
v1_rmse = NaN;
if any(is_v1)
    v1_rmse = T.pos_rmse_m(find(is_v1, 1));
end

dm = T(~is_v1, :);
[it, ord] = sort(dm.iterations);
rm = dm.pos_rmse_m(ord);
it = it(:)'; rm = rm(:)';

fig = figure('Name', 'iter_compare', 'Color', 'w');
hold on; grid on; box on;

plot(it, rm, 'o-', 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', 'w', ...
     'Color', [0.85 0.20 0.15], 'DisplayName', 'Distributed GN (DMLKF\_C)');

if isfinite(v1_rmse)
    % 这里加入了 'Label' 属性及其对齐方式，用于在图表左侧显示 V1 的数值
    yline(v1_rmse, '--', 'LineWidth', 1.6, 'Color', [0.10 0.35 0.80], ...
          'Label', sprintf('V1: %.4f', v1_rmse), ...
          'LabelHorizontalAlignment', 'left', ...
          'LabelVerticalAlignment', 'bottom', ...
          'FontSize', 8, ...
          'DisplayName', sprintf('V1 baseline: centralized GN (K = %d)', K));
end

if show_fit
    [nf, yf, fit_ok] = exp_convergence_fit(it, rm);
    if fit_ok
        plot(nf, yf, '-', 'LineWidth', 1.6, 'Color', [0.45 0.10 0.55], ...
             'DisplayName', 'Distributed GN (convergence fit, not a measurement)');
    end
end

for q = 1:numel(it)
    text(it(q), rm(q), sprintf('  %.4f', rm(q)), 'FontSize', 8, ...
         'VerticalAlignment', 'bottom', 'Interpreter', 'none');
end

set(gca, 'XScale', 'log');
xticks(it);
xticklabels(cellstr(string(it)));
xlim([min(it)/1.35, max(it)*1.45]);          % 给末端数值标注留位置

vals = [rm(:); v1_rmse]; vals = vals(isfinite(vals));
if numel(vals) > 1
    pad = 0.14 * (max(vals) - min(vals));
    ylim([min(vals) - pad, max(vals) + 0.55*pad]);
end

xlabel('Gauss-Newton iteration budget (max\_iter)');
ylabel('Position RMSE (m)');
ytickformat('%.4f');                          % 坐标刻度也保留 4 位小数
title(sprintf('Task 2: Position RMSE vs. iteration budget (K = %d, %d vehicles, %d anchors, %.0f%% of data)', ...
      K, V_num, A_num, 100*data_ratio));
legend('Location', 'southeast');

if ~isempty(out_png)
    exportgraphics(fig, out_png, 'Resolution', 150);
    fprintf(' 已保存图片: %s\n', out_png);
end
end

% ================================================================ 趋势拟合
function [nf, yf, ok] = exp_convergence_fit(it, rm)
%EXP_CONVERGENCE_FIT 单调饱和收敛模型 rmse(n) = c - a*exp(-n/tau)（仅用于显示趋势）
%   在对数等距网格上返回拟合曲线；数据点太少或拟合异常时 ok = false。
nf = []; yf = []; ok = false;
n = it(:); y = rm(:);
if numel(n) < 4, return; end

c0   = max(y) * 1.02;
a0   = max(c0 - min(y), 1e-6);
tau0 = max(max(n) / 3, 1);
obj  = @(p) sum((y - (p(1) - abs(p(2)) .* exp(-n ./ max(abs(p(3)), 1e-6)))).^2);
opts = optimset('Display', 'off', 'MaxFunEvals', 2e4, 'MaxIter', 2e4);
p    = fminsearch(obj, [c0, a0, tau0], opts);

nf = logspace(log10(min(n)), log10(max(n)), 200)';
yf = p(1) - abs(p(2)) .* exp(-nf ./ max(abs(p(3)), 1e-6));
ok = all(isfinite(yf));
end