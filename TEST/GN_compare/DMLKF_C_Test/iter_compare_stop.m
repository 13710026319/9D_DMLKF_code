function iter_compare_stop(verbose)
%ITER_COMPARE_STOP 结束 iter_compare_main 留下的后台 worker 进程
%
%   背景：iter_compare_main 用 start /B 拉起独立 MATLAB 进程做并行。
%   在 MATLAB 里按 Ctrl+C / “停止”只中断主脚本自己的等待循环，
%   已经被系统接管的子进程不会随之退出，会继续跑并往 RESULT\_parts 写结果。
%
%   用法（在 MATLAB 里直接运行即可）：
%       iter_compare_stop          % 结束残留 worker，并清理未完成的中间分片
%       iter_compare_stop(false)   % 只结束进程，不打印细节
%
%   只会结束“命令行里带本实验 RESULT\_parts 路径”的进程，
%   不会影响你已打开的 MATLAB 会话，也不会动别的实验。

if nargin < 1 || isempty(verbose), verbose = true; end

this_dir = fileparts(mfilename('fullpath'));
part_dir = fullfile(this_dir, 'RESULT', '_parts');

cmd = sprintf(['powershell -NoProfile -Command "Get-CimInstance Win32_Process | ' ...
               'Where-Object { $_.Name -like ''matlab*'' -and $_.CommandLine -like ''*%s*'' } | ' ...
               'ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"'], part_dir);
[st, out] = system(cmd);
if verbose
    fprintf('[iter_compare_stop] 已结束残留 worker（退出码 %d）\n', st);
    if ~isempty(strtrim(out)), disp(out); end
end

if exist(part_dir, 'dir')
    f = [dir(fullfile(part_dir, '*.csv')); dir(fullfile(part_dir, '*.done'))];
    for q = 1:numel(f)
        try
            delete(fullfile(part_dir, f(q).name));
        catch
        end
    end
    if verbose, fprintf('[iter_compare_stop] 已清理中间分片: %s\n', part_dir); end
end
if verbose
    fprintf('[iter_compare_stop] 注意：RESULT 里只保留汇总后的 iter_compare.csv，重跑前无需手动清理。\n');
end
end
