# C_compare —— 三算法 × 基站数 对比实验

## 目的

在**同一个** 6 车辆数据集上，考察基站数从 4 增加到 35 时，
**CRBPF / CMLKF / CEKF** 三个算法的定位性能变化。

- 数据集：`E:\DMLKF_code\Data\C_compare\Trj_Veh6_Anc35_3D_10.mat`（唯一数据集，不改动）
- 基站数 = a 时，只取该数据集中**前 a 个基站**的测距观测（`anchors(1:a,:)` 与 `UWB_Anchor(:,2:1+a)`），
  车-车相对测距始终全量使用。
- `bias_comp_ratio = 0.6`，`data_ratio = 1.0`（100% 使用数据集）。
- 三个算法**纯调用**：脚本里不 `set` 任何参数，`CRBPF.m / CMLKF.m / CEKF.m` 一字未改。

## 文件

| 文件 | 作用 |
|---|---|
| `C_compare_main.m` | **总入口**。划分基站数分段 → 并行拉起多个 MATLAB 子进程 → 等待 → 汇总 → 打印 → 绘图 |
| `c_compare_worker.m` | 子脚本。负责一段连续基站数（如 4~8）内三个算法的运行，结果写到 `RESULT\seg_<a1>_<a2>.csv` |
| `c_compare_loop.m` | 三算法共用的滤波主循环 + RMSE 计算（与三个 `*_Test.m` 口径完全一致） |
| `c_compare_collect.m` | 汇总所有分段 CSV → 宽表 CSV、控制台表格、对比图 |

## 用法

```matlab
cd('E:\DMLKF_code\TEST\C_comapre');
C_compare_main          % 直接跑：默认 10 个并行进程，基站 4~35，全量数据
```

只跑某一段（调试/单机串行）：

```matlab
c_compare_worker(4, 8)          % 基站 4~8，三个算法，全量数据
c_compare_worker(4, 8, 300)     % 冒烟测试：只跑前 300 步
c_compare_collect               % 只用现有的 seg_*.csv 出表和出图
```

## 可调配置（`C_compare_main.m` 顶部）

| 变量 | 默认 | 说明 |
|---|---|---|
| `anchor_lo / anchor_hi` | 4 / 35 | 基站数范围 |
| `n_workers` | 10 | 并行进程数 = 分段数；机器空闲多可调高，内存吃紧就调低 |
| `step_cap` | `Inf` | `Inf` 跑满 10001 步；设小值可快速验证流程 |
| `clean_start` | `true` | 是否先清掉 `RESULT` 里上一轮的 `seg_*.csv` 与汇总结果 |

## 输出（`RESULT\`）

- `seg_<a1>_<a2>.csv` —— 每段原始结果：`anchor_num, algorithm, rmse_p, rmse_v, rmse_att, sec, ok, note`
- `C_compare_RMSE_vs_Anchor.csv` —— 汇总宽表：每个基站数一行，三个算法的位置/速度/姿态 RMSE 与耗时
- `C_compare_RMSE_vs_Anchor.png` —— 三联图：位置 / 速度 / 姿态 RMSE 随基站数变化
- `logs\seg_*.log` —— 每个子进程的完整日志

## 说明

- 并行方式是"多个独立 MATLAB 进程"（`start /B matlab.exe -singleCompThread -batch ...`），
  不依赖 Parallel Computing Toolbox，也不会干扰你自己已经打开的 MATLAB 会话。
- 单个 CRBPF 运行（Np=600、35 基站、10001 步）约需 17 分钟，是所有耗时的主导项；
  CMLKF / CEKF 各只需数秒到数十秒。因此分段时把**基站数多的段划得短一些**更均衡。
