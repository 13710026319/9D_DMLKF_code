function T = c_compare_final_report(root_dir, do_plot)
% 只读已有结果，生成最终表与"一图两子图"的结果图（不跑任何滤波）。
%
%   c_compare_final_report()                     % 默认 4 车 6 数据集目录
%   c_compare_final_report(dir, true)
%
% 输入：C_compare_results.csv（含 CRBPF / CMLKF / CEKF 的位置 RMSE）
% 输出：
%   C_compare_results.csv   重写为：三个算法 RMSE + 相对 CEKF 的提升百分比
%   C_compare_ds1.png / C_compare_ds2.png
%       左子图 = 三个算法的位置 RMSE 曲线
%       右子图 = CRBPF、CMLKF 相对 CEKF 基准的提升百分比
%
% 只保留 ds1、ds2，且基站 >= 3（基站 2 时旋转不可观，CRBPF/CMLKF 发散）。

if nargin < 1 || isempty(root_dir)
    root_dir = fullfile(fileparts(mfilename('fullpath')), 'RESULT', 'Veh4_Anc20');
end
if nargin < 2 || isempty(do_plot), do_plot = true; end

f_in = fullfile(root_dir, 'C_compare_results.csv');
if ~exist(f_in, 'file'), error('缺少结果文件: %s', f_in); end

A = readtable(f_in, 'VariableNamingRule', 'preserve');

% ---- 只保留 ds1 / ds2，且基站 >= 3 ----
keep = ismember(A.dataset, [1 2]) & A.anchor_num >= 3;
A = A(keep, :);
A = sortrows(A, {'dataset','anchor_num'});

% ---- 以 CEKF 为基准计算提升百分比（正数 = 比 CEKF 好）----
imp_R = 100 * (A.CEKF_pos - A.CRBPF_pos) ./ A.CEKF_pos;
imp_M = 100 * (A.CEKF_pos - A.CMLKF_pos) ./ A.CEKF_pos;

T = table(A.dataset, A.anchor_num, A.CRBPF_pos, A.CMLKF_pos, A.CEKF_pos, imp_R, imp_M, ...
    'VariableNames', {'dataset','anchor_num','CRBPF_pos','CMLKF_pos','CEKF_pos', ...
                      'improve_CRBPF_vs_CEKF_pct','improve_CMLKF_vs_CEKF_pct'});
writetable(T, f_in);

fprintf('\n========== 位置 RMSE 与相对 CEKF 的提升（4 车，bias_comp=0.6，数据 100%%）==========\n');
for ds = [1 2]
    sub = T(T.dataset == ds, :);
    fprintf('\n---- 数据集 ds%d ----\n', ds);
    fprintf(' 基站 |   CRBPF    CMLKF     CEKF   | 相对CEKF提升  RBPF     MLKF\n');
    fprintf('------+--------------------------+-----------------------------\n');
    for r = 1:height(sub)
        fprintf(' %4d | %8.4f %8.4f %8.4f | %13.1f%% %6.1f%%\n', ...
            sub.anchor_num(r), sub.CRBPF_pos(r), sub.CMLKF_pos(r), sub.CEKF_pos(r), ...
            sub.improve_CRBPF_vs_CEKF_pct(r), sub.improve_CMLKF_vs_CEKF_pct(r));
    end
    fprintf('------+--------------------------+-----------------------------\n');
    fprintf(' 均值 | %8.4f %8.4f %8.4f | %13.1f%% %6.1f%%\n', ...
        mean(sub.CRBPF_pos), mean(sub.CMLKF_pos), mean(sub.CEKF_pos), ...
        mean(sub.improve_CRBPF_vs_CEKF_pct), mean(sub.improve_CMLKF_vs_CEKF_pct));
end
fprintf('\n已写出: %s\n', f_in);

% ---- 一图两子图 ----
if do_plot
    cR = [0.85 0.20 0.15];   % CRBPF
    cM = [0.00 0.45 0.74];   % CMLKF
    cE = [0.15 0.60 0.25];   % CEKF
    for ds = [1 2]
        sub = T(T.dataset == ds, :);
        x = sub.anchor_num;

        fig = figure('Visible','off','Color','w','Position',[80 80 1200 470]);

        % (a) 三个算法的 RMSE 曲线
        subplot(1,2,1);
        plot(x, sub.CRBPF_pos, '-o','LineWidth',1.8,'MarkerSize',6, ...
             'Color',cR,'DisplayName','CRBPF'); hold on;
        plot(x, sub.CMLKF_pos, '-s','LineWidth',1.8,'MarkerSize',6, ...
             'Color',cM,'DisplayName','CMLKF');
        plot(x, sub.CEKF_pos,  '-^','LineWidth',1.8,'MarkerSize',6, ...
             'Color',cE,'DisplayName','CEKF');
        grid on; box on; set(gca,'FontSize',10);
        xlabel('基站数','FontSize',12); ylabel('位置 RMSE (m)','FontSize',12);
        title(sprintf('(a) 三算法定位精度对比  (ds%d)', ds),'FontSize',12);
        legend('Location','northeast','FontSize',10);
        xticks(x(1):2:x(end)); xlim([x(1)-0.5, x(end)+0.5]);

        % (b) 相对 CEKF 的提升百分比
        subplot(1,2,2);
        plot(x, sub.improve_CRBPF_vs_CEKF_pct, '-o','LineWidth',1.9,'MarkerSize',6, ...
             'Color',cR,'DisplayName','CRBPF'); hold on;
        plot(x, sub.improve_CMLKF_vs_CEKF_pct, '-s','LineWidth',1.9,'MarkerSize',6, ...
             'Color',cM,'DisplayName','CMLKF');
        yline(0,'k--','HandleVisibility','off');
        grid on; box on; set(gca,'FontSize',10);
        xlabel('基站数','FontSize',12); ylabel('相对 CEKF 的 RMSE 提升 (%)','FontSize',12);
        title(sprintf('(b) 相对 CEKF 基准的提升幅度  (ds%d)', ds),'FontSize',12);
        legend('Location','northeast','FontSize',10);
        xticks(x(1):2:x(end)); xlim([x(1)-0.5, x(end)+0.5]);
        ymax = max([sub.improve_CRBPF_vs_CEKF_pct; sub.improve_CMLKF_vs_CEKF_pct]);
        ylim([0, ymax*1.18]);

        png = fullfile(root_dir, sprintf('C_compare_ds%d.png', ds));
        try, exportgraphics(fig, png, 'Resolution', 200); catch, saveas(fig, png); end
        close(fig);
        fprintf('已绘图: %s\n', png);
    end
end
end
