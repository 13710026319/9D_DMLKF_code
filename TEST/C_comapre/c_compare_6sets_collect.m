function T = c_compare_6sets_collect(root_dir, do_plot, clean_parts)
% 汇总"4 车 / 6 个数据集 / 基站 2~20"实验。
%
% root_dir 下应有 ds1 ... ds6 子目录，每个里面有分段 CSV（含 anchor_num / algorithm / rmse_p）。
% 输出（全部只含位置 RMSE）：
%   C_compare_RMSE_vs_Anchor.csv   每个 (数据集, 基站数) 一行
%   C_compare_gap_summary.csv      每个数据集一行：CMLKF/CRBPF 差距的"起始-末端"变化
%   C_compare_RMSE_vs_Anchor.png   2x3 子图，每个数据集一张
%   C_compare_mean.png             6 个数据集平均 ± 标准差
% 最后（clean_parts = true）删除所有分段结果 seg_*.csv / seg_*.done。

if nargin < 1 || isempty(root_dir)
    root_dir = fullfile(fileparts(mfilename('fullpath')), 'RESULT', 'Veh4_Anc20');
end
if nargin < 2 || isempty(do_plot),     do_plot     = true;  end
if nargin < 3 || isempty(clean_parts), clean_parts = true;  end

algs = {'CRBPF', 'CMLKF', 'CEKF'};

ds_dirs = dir(fullfile(root_dir, 'ds*'));
ds_dirs = ds_dirs([ds_dirs.isdir]);
if isempty(ds_dirs)
    error('没有找到任何 ds* 子目录 (目录: %s)', root_dir);
end
nDS = numel(ds_dirs);
ds_id = zeros(nDS, 1);
for k = 1:nDS
    ds_id(k) = sscanf(ds_dirs(k).name, 'ds%d');
end
[ds_id, ord] = sort(ds_id);
ds_dirs = ds_dirs(ord);

% ---- 收集 ----
anchors_all = [];
RAW = [];       % [ds_idx, anchor, alg_idx, rmse, sec]
for k = 1:nDS
    files = dir(fullfile(root_dir, ds_dirs(k).name, 'seg_*.csv'));
    for f = 1:numel(files)
        Tf = readtable(fullfile(root_dir, ds_dirs(k).name, files(f).name), ...
                       'VariableNamingRule', 'preserve');
        if isempty(Tf), continue; end
        vn = Tf.Properties.VariableNames;
        if ~ismember('sec', vn), Tf.sec = nan(height(Tf), 1); end
        for r = 1:height(Tf)
            c = find(strcmp(Tf.algorithm{r}, algs), 1);
            if isempty(c) || Tf.ok(r) ~= 1, continue; end
            RAW(end+1, :) = [k, Tf.anchor_num(r), c, Tf.rmse_p(r), Tf.sec(r)]; %#ok<AGROW>
            anchors_all(end+1) = Tf.anchor_num(r);                               %#ok<AGROW>
        end
    end
end
if isempty(RAW), error('分段结果为空，无法汇总'); end

anchor_list = unique(anchors_all(:))';
nA = numel(anchor_list);
P  = nan(nDS, nA, numel(algs));
Se = nan(nDS, nA, numel(algs));
for r = 1:size(RAW, 1)
    ai = find(anchor_list == RAW(r, 2), 1);
    P(RAW(r,1), ai, RAW(r,3))  = RAW(r,4);
    Se(RAW(r,1), ai, RAW(r,3)) = RAW(r,5);
end

% ---- 宽表 CSV ----
rows = nDS * nA;
T = table('Size', [rows 8], ...
    'VariableTypes', {'double','double','double','double','double','double','double','double'}, ...
    'VariableNames', {'dataset','anchor_num','CRBPF_pos','CMLKF_pos','CEKF_pos', ...
                      'ratio_CMLKF_over_CRBPF','ratio_CEKF_over_CRBPF','CRBPF_sec'});
t = 0;
for k = 1:nDS
    for i = 1:nA
        t = t + 1;
        T.dataset(t) = ds_id(k);
        T.anchor_num(t) = anchor_list(i);
        T.CRBPF_pos(t) = P(k,i,1);
        T.CMLKF_pos(t) = P(k,i,2);
        T.CEKF_pos(t)  = P(k,i,3);
        T.ratio_CMLKF_over_CRBPF(t) = P(k,i,2)/P(k,i,1);
        T.ratio_CEKF_over_CRBPF(t)  = P(k,i,3)/P(k,i,1);
        T.CRBPF_sec(t) = Se(k,i,1);
    end
end
csv_main = fullfile(root_dir, 'C_compare_RMSE_vs_Anchor.csv');
writetable(T, csv_main);

% ---- 每个数据集的"差距变化"摘要 ----
% 注意：基站 = 2 时整个网络绕"两台基站连线"的旋转不可观，CRBPF/CMLKF 会发散
%       （RMSE 十几米，反而比 CEKF 差），所以统计窗口从 3 个基站开始。
early = anchor_list >= 3 & anchor_list <= 6;              % 起始段
late  = anchor_list >= (max(anchor_list) - 3);            % 末端段
R_c = P(:,:,2) ./ P(:,:,1);
R_e = P(:,:,3) ./ P(:,:,1);

S = table('Size', [nDS 10], ...
    'VariableTypes', repmat({'double'}, 1, 10), ...
    'VariableNames', {'dataset','CMLKF_ratio_early','CMLKF_ratio_late','CMLKF_drop', ...
                      'CEKF_ratio_early','CEKF_ratio_late','CEKF_drop', ...
                      'CRBPF_pos_late','CMLKF_pos_late','CEKF_pos_late'});
for k = 1:nDS
    ce = mean(R_c(k, early), 'omitnan');
    cl = mean(R_c(k, late),  'omitnan');
    ee = mean(R_e(k, early), 'omitnan');
    el = mean(R_e(k, late),  'omitnan');
    S.dataset(k) = ds_id(k);
    S.CMLKF_ratio_early(k) = ce;  S.CMLKF_ratio_late(k) = cl;  S.CMLKF_drop(k) = ce - cl;
    S.CEKF_ratio_early(k)  = ee;  S.CEKF_ratio_late(k)  = el;  S.CEKF_drop(k)  = ee - el;
    S.CRBPF_pos_late(k) = mean(P(k, late, 1), 'omitnan');
    S.CMLKF_pos_late(k) = mean(P(k, late, 2), 'omitnan');
    S.CEKF_pos_late(k)  = mean(P(k, late, 3), 'omitnan');
end
csv_gap = fullfile(root_dir, 'C_compare_gap_summary.csv');
writetable(S, csv_gap);

% ---- 控制台打印 ----
fprintf('\n================ 4 车 / 6 数据集：CMLKF 与 CEKF 相对 CRBPF 的差距 ================\n');
fprintf('数据集 | CMLKF起始  CMLKF末端  CMLKF变化 | CEKF起始  CEKF末端  CEKF变化 | 末端 RMSE (RBPF/MLKF/EKF)\n');
fprintf('-------+------------------------------------+----------------------------------+-----------------------------\n');
for k = 1:nDS
    fprintf('  ds%d  | %8.3f %9.3f %10.3f | %8.3f %8.3f %9.3f |  %.4f / %.4f / %.4f\n', ...
        S.dataset(k), S.CMLKF_ratio_early(k), S.CMLKF_ratio_late(k), S.CMLKF_drop(k), ...
        S.CEKF_ratio_early(k), S.CEKF_ratio_late(k), S.CEKF_drop(k), ...
        S.CRBPF_pos_late(k), S.CMLKF_pos_late(k), S.CEKF_pos_late(k));
end
fprintf('（起始 = 基站 %d~%d 平均，末端 = 基站 %d~%d 平均；比值 >1 表示该算法比 CRBPF 差）\n', ...
        min(anchor_list(early)), max(anchor_list(early)), ...
        min(anchor_list(late)),  max(anchor_list(late)));

[~, best] = max(S.CMLKF_drop);
fprintf('\n>> CMLKF 差距收缩最明显的数据集: ds%d (收缩 %.3f，末端比值 %.3f)\n', ...
        S.dataset(best), S.CMLKF_drop(best), S.CMLKF_ratio_late(best));
fprintf('\n---- ds%d 全部基站数明细 (位置 RMSE) ----\n', S.dataset(best));
fprintf(' 基站 |   CRBPF    CMLKF     CEKF   | CMLKF/RBPF  CEKF/RBPF\n');
for i = 1:nA
    fprintf(' %4d | %8.4f %8.4f %8.4f | %9.3f %10.3f\n', ...
        anchor_list(i), P(best,i,1), P(best,i,2), P(best,i,3), R_c(best,i), R_e(best,i));
end
fprintf('\n已写出: %s\n已写出: %s\n', csv_main, csv_gap);

% ---- 绘图 ----
if do_plot
    nRow = ceil(nDS/3); nCol = min(nDS,3);
    fig = figure('Visible','off','Color','w','Position',[60 60 480*nCol 380*nRow]);
    for k = 1:nDS
        subplot(nRow, nCol, k);
        plot(anchor_list, P(k,:,1), '-o','LineWidth',1.5,'MarkerSize',5,'Color',[0.85 0.20 0.15]); hold on;
        plot(anchor_list, P(k,:,2), '-s','LineWidth',1.5,'MarkerSize',5,'Color',[0.00 0.45 0.74]);
        plot(anchor_list, P(k,:,3), '-^','LineWidth',1.5,'MarkerSize',5,'Color',[0.15 0.60 0.25]);
        grid on; box on;
        xlabel('基站数'); ylabel('位置 RMSE (m)');
        title(sprintf('ds%d', S.dataset(k)));
        if k == 1, legend({'CRBPF','CMLKF','CEKF'}, 'Location','northeast','FontSize',8); end
        set(gca,'FontSize',8); xticks(anchor_list(1):2:anchor_list(end));
    end
    png1 = fullfile(root_dir, 'C_compare_RMSE_vs_Anchor.png');
    try, exportgraphics(fig, png1, 'Resolution', 180); catch, saveas(fig, png1); end
    close(fig);

    fig = figure('Visible','off','Color','w','Position',[80 80 900 540]);
    cols = [0.85 0.20 0.15; 0.00 0.45 0.74; 0.15 0.60 0.25];
    names = {'CRBPF','CMLKF','CEKF'};
    for c = 1:3
        Y = squeeze(P(:,:,c));
        m = mean(Y, 1, 'omitnan');  s = std(Y, 0, 1, 'omitnan');
        ok = ~isnan(m);
        fill([anchor_list(ok), fliplr(anchor_list(ok))], ...
             [m(ok)-s(ok), fliplr(m(ok)+s(ok))], cols(c,:), ...
             'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off'); hold on;
        plot(anchor_list(ok), m(ok), '-o','LineWidth',1.8,'MarkerSize',6,'Color',cols(c,:), ...
             'DisplayName', names{c});
    end
    grid on; box on; set(gca,'FontSize',10);
    xlabel('基站数','FontSize',12); ylabel('位置 RMSE (m)','FontSize',12);
    title('6 个数据集的平均（阴影 = \pm1 标准差）','FontSize',13);
    legend('Location','northeast','FontSize',11);
    xticks(anchor_list(1):2:anchor_list(end));
    png2 = fullfile(root_dir, 'C_compare_mean.png');
    try, exportgraphics(fig, png2, 'Resolution', 200); catch, saveas(fig, png2); end
    close(fig);
    fprintf('已绘图: %s\n已绘图: %s\n', png1, png2);
end

% ---- 清理分段结果（只保留总结果） ----
if clean_parts
    n = 0;
    for k = 1:nDS
        d = fullfile(root_dir, ds_dirs(k).name);
        f1 = dir(fullfile(d, 'seg_*.csv'));
        f2 = dir(fullfile(d, 'seg_*.done'));
        for f = [f1; f2]'
            delete(fullfile(d, f.name)); n = n + 1;
        end
    end
    fprintf('已删除 %d 个分段结果文件（只保留总结果）\n', n);
end
end
