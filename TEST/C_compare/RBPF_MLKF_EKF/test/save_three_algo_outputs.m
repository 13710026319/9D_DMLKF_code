function save_three_algo_outputs(R, csv_file, png_file)
% 写入四位小数的 RMSE CSV，并生成一张三曲线 PNG。

fid = fopen(csv_file, 'w', 'n', 'UTF-8');
if fid < 0
    error('无法写入 CSV：%s', csv_file);
end
cleanup_obj = onCleanup(@() fclose(fid));

fprintf(fid, 'Anchor_num,CRBPF_PosRMSE_m,CMLKF_PosRMSE_m,CEKF_PosRMSE_m\n');
for r = 1:size(R, 1)
    fprintf(fid, '%d,%.4f,%.4f,%.4f\n', ...
            R(r, 1), R(r, 2), R(r, 3), R(r, 4));
end
clear cleanup_obj;

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 720 480]);
plot(R(:, 1), R(:, 2), '-o', 'LineWidth', 1.8, ...
     'MarkerSize', 5, 'Color', [0.85 0.33 0.10]); hold on;
plot(R(:, 1), R(:, 3), '-s', 'LineWidth', 1.8, ...
     'MarkerSize', 5, 'Color', [0.00 0.45 0.74]);
plot(R(:, 1), R(:, 4), '-^', 'LineWidth', 1.8, ...
     'MarkerSize', 5, 'Color', [0.47 0.67 0.19]);
grid on;
box on;
xlim([R(1, 1), R(end, 1)]);
xticks(R(1, 1):2:R(end, 1));
xlabel('Number of anchors');
ylabel('Position RMSE (m)');
title('Position RMSE versus Number of Anchors');
legend({'CRBPF', 'CMLKF', 'CEKF'}, 'Location', 'northeast');
exportgraphics(fig, png_file, 'Resolution', 200);
close(fig);
end
