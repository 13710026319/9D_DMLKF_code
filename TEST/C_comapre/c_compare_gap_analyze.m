function S = c_compare_gap_analyze(root_dir, do_plot)
% 从"总结果" C_compare_RMSE_vs_Anchor.csv 重新做差距分析（不需要分段结果）。
%
%   c_compare_gap_analyze()                                  % 默认 4 车 6 数据集目录
%   c_compare_gap_analyze('E:\...\RESULT\Veh4_Anc20', true)
%
% 输出：
%   C_compare_gap_summary.csv  每个数据集一行（起始段 / 末端段 / 收缩量）
%   C_compare_RMSE_vs_Anchor.png / C_compare_mean.png  重新绘图
%
% 统计窗口：起始 = 基站 3~6，末端 = 基站 17~20。
% （基站 2 时绕两台基站连线的旋转不可观，CRBPF/CMLKF 会发散，单列不进统计。）

if nargin < 1 || isempty(root_dir)
    root_dir = fullfile(fileparts(mfilename('fullpath')), 'RESULT', 'Veh4_Anc20');
end
if nargin < 2 || isempty(do_plot), do_plot = true; end

algs = {'CRBPF','CMLKF','CEKF'};
csv_main = fullfile(root_dir, 'C_compare_RMSE_vs_Anchor.csv');
if ~exist(csv_main, 'file'), error('找不到总结果文件: %s', csv_main); end
T = readtable(csv_main, 'VariableNamingRule', 'preserve');

ds_list     = unique(T.dataset)';
anchor_list = unique(T.anchor_num)';
nDS = numel(ds_list); nA = numel(anchor_list);
P = nan(nDS, nA, 3);
for r = 1:height(T)
    k = find(ds_list == T.dataset(r), 1);
    i = find(anchor_list == T.anchor_num(r), 1);
    P(k,i,1) = T.CRBPF_pos(r); P(k,i,2) = T.CMLKF_pos(r); P(k,i,3) = T.CEKF_pos(r);
end

early = anchor_list >= 3 & anchor_list <= 6;
late  = anchor_list >= (max(anchor_list) - 3);
Rc = P(:,:,2) ./ P(:,:,1);
Re = P(:,:,3) ./ P(:,:,1);

S = table('Size', [nDS 10], 'VariableTypes', repmat({'double'},1,10), ...
    'VariableNames', {'dataset','CMLKF_ratio_early','CMLKF_ratio_late','CMLKF_drop', ...
                      'CMLKF_drop_rel','CEKF_ratio_early','CEKF_ratio_late','CEKF_drop', ...
                      'CRBPF_pos_late','CMLKF_pos_late'});
CEKF_late = nan(nDS,1);
for k = 1:nDS
    ce = mean(Rc(k,early),'omitnan'); cl = mean(Rc(k,late),'omitnan');
    ee = mean(Re(k,early),'omitnan'); el = mean(Re(k,late),'omitnan');
    S.dataset(k)=ds_list(k);
    S.CMLKF_ratio_early(k)=ce; S.CMLKF_ratio_late(k)=cl;
    S.CMLKF_drop(k)=ce-cl;     S.CMLKF_drop_rel(k)=(ce-cl)/ce;
    S.CEKF_ratio_early(k)=ee;  S.CEKF_ratio_late(k)=el; S.CEKF_drop(k)=ee-el;
    S.CRBPF_pos_late(k)=mean(P(k,late,1),'omitnan');
    S.CMLKF_pos_late(k)=mean(P(k,late,2),'omitnan');
    CEKF_late(k) = mean(P(k,late,3),'omitnan');
end
writetable(S, fullfile(root_dir, 'C_compare_gap_summary.csv'));

fprintf('\n============ 4 车 / 6 数据集：CMLKF、CEKF 相对 CRBPF 的差距 ============\n');
fprintf('（起始 = 基站 3~6 平均，末端 = 基站 17~20 平均；比值 >1 = 比 CRBPF 差）\n\n');
fprintf('数据集 | CMLKF起始  CMLKF末端   收缩量   收缩%%  | CEKF起始  CEKF末端 | 末端RMSE RBPF / MLKF / EKF\n');
fprintf('-------+-----------------------------------------+------------------+---------------------------\n');
for k = 1:nDS
    fprintf('  ds%d  | %8.3f %9.3f %9.3f %7.1f%% | %8.3f %8.3f | %.4f / %.4f / %.4f\n', ...
        S.dataset(k), S.CMLKF_ratio_early(k), S.CMLKF_ratio_late(k), ...
        S.CMLKF_drop(k), 100*S.CMLKF_drop_rel(k), ...
        S.CEKF_ratio_early(k), S.CEKF_ratio_late(k), ...
        S.CRBPF_pos_late(k), S.CMLKF_pos_late(k), CEKF_late(k));
end

[~, b1] = max(S.CMLKF_drop);        % 绝对收缩最大
[~, b2] = max(S.CMLKF_drop_rel);    % 相对收缩最大
fprintf('\n>> 绝对差距收缩最大 : ds%d  (%.3f -> %.3f，收缩 %.3f)\n', ...
    S.dataset(b1), S.CMLKF_ratio_early(b1), S.CMLKF_ratio_late(b1), S.CMLKF_drop(b1));
fprintf('>> 相对收缩最大     : ds%d  (%.3f -> %.3f，收缩 %.1f%%)\n', ...
    S.dataset(b2), S.CMLKF_ratio_early(b2), S.CMLKF_ratio_late(b2), 100*S.CMLKF_drop_rel(b2));

% ---------------- 绘图 ----------------
if do_plot
    cols = [0.85 0.20 0.15; 0.00 0.45 0.74; 0.15 0.60 0.25];
    nRow = ceil(nDS/3); nCol = min(nDS,3);
    fig = figure('Visible','off','Color','w','Position',[60 60 470*nCol 370*nRow]);
    for k = 1:nDS
        subplot(nRow,nCol,k);
        plot(anchor_list, P(k,:,1),'-o','LineWidth',1.5,'MarkerSize',5,'Color',cols(1,:)); hold on;
        plot(anchor_list, P(k,:,2),'-s','LineWidth',1.5,'MarkerSize',5,'Color',cols(2,:));
        plot(anchor_list, P(k,:,3),'-^','LineWidth',1.5,'MarkerSize',5,'Color',cols(3,:));
        set(gca,'YScale','log'); grid on; box on; set(gca,'FontSize',8);
        xlabel('基站数'); ylabel('位置 RMSE (m)');
        title(sprintf('ds%d', ds_list(k)));
        if k==1, legend({'CRBPF','CMLKF','CEKF'},'Location','northeast','FontSize',8); end
        xticks(anchor_list(1):2:anchor_list(end));
    end
    png1 = fullfile(root_dir, 'C_compare_RMSE_vs_Anchor.png');
    try, exportgraphics(fig, png1, 'Resolution', 180); catch, saveas(fig, png1); end
    close(fig);

    % 平均曲线：剔除基站=2（发散点），否则会压扁整张图
    keep = anchor_list >= 3;
    fig = figure('Visible','off','Color','w','Position',[80 80 900 540]);
    for c = 1:3
        Y = squeeze(P(:,keep,c));
        m = mean(Y,1,'omitnan'); s = std(Y,0,1,'omitnan');
        fill([anchor_list(keep), fliplr(anchor_list(keep))], ...
             [m-s, fliplr(m+s)], cols(c,:), 'FaceAlpha',0.12,'EdgeColor','none', ...
             'HandleVisibility','off'); hold on;
        plot(anchor_list(keep), m, '-o','LineWidth',1.8,'MarkerSize',6,'Color',cols(c,:), ...
             'DisplayName', algs{c});
    end
    grid on; box on; set(gca,'FontSize',10);
    xlabel('基站数','FontSize',12); ylabel('位置 RMSE (m)','FontSize',12);
    title('6 个数据集的平均（阴影 = \pm1 标准差；已剔除 2 基站的发散点）','FontSize',12);
    legend('Location','northeast','FontSize',11);
    xticks(anchor_list(keep(1)):2:anchor_list(end));
    png2 = fullfile(root_dir, 'C_compare_mean.png');
    try, exportgraphics(fig, png2, 'Resolution', 200); catch, saveas(fig, png2); end
    close(fig);
    fprintf('\n已绘图: %s\n已绘图: %s\n', png1, png2);
end
end
