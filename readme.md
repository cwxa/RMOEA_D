# RMOEA/D - Reinforcement Learning based MOEA/D for Bi-objective Fuzzy Flexible Job Shop Scheduling

基于强化学习的 MOEA/D 双目标模糊柔性作业车间调度求解器。

## 项目简介

复现论文 *A reinforcement learning based RMOEA/D for bi-objective fuzzy flexible job shop scheduling* 的核心算法，针对 **Brandimarte Mk01~Mk10** 实例进行双目标优化：

- **Makespan**：模糊最大完工时间，三角模糊数 (t1, t2, t3)
- **Workload**：总机器工作负载

### 核心创新

| 组件 | 机制 |
|------|------|
| **Q-PAS** | Q-learning 自适应选择邻域大小 T，动态平衡收敛性与多样性 |
| **RVNS** | 强化学习驱动变邻域搜索，5 种 LS 算子 + 滑动窗口成功/失败记忆 |

### 代码架构特点

- **继承复用**: `MOEADBaseline` 继承自 `RMOEAD`，通过构造函数参数特化（`fixed_T=10`, `enable_rvns=False`）消除重复代码
- **图表复用**: `ablation_visualization.py` 复用 `analysis_report.py` 标准图表生成函数，避免重复造轮子
- **统一日志**: 双路日志系统（控制台 INFO + 文件 DEBUG），关键变量与执行计时全面记录
- **单元测试**: `tests/test_refactor.py` 覆盖重构后核心功能的正确性验证

---

## 文件结构

```
RMOEA_D/
├── main.py                              # 唯一入口 (optimize / benchmark / ablation / run_all)
├── readme.md
│
├── src/rmoea_d/
│   ├── algorithm.py                     # RMOEA/D 主算法 (Q-PAS + RVNS + MOEA/D)
│   ├── moead_baseline.py                # MOEA/D 基线 (继承 RMOEAD，固定 T)
│   │
│   ├── core/                            # 核心模块
│   │   ├── fuzzy.py                     # 三角模糊数运算
│   │   ├── instance.py                  # Brandimarte 实例加载
│   │   ├── encoding.py                  # OS+MA 编解码
│   │   ├── operators.py                 # 遗传算子 (MIX3/POX/UX)
│   │   ├── moead.py                     # MOEA/D 框架 (权重/邻居/Tchebycheff)
│   │   ├── qlearning.py                 # Q-PAS 参数自适应
│   │   └── rvns.py                      # RVNS (5 LS 算子 + 记忆机制)
│   │
│   └── utils/                           # 工具 & 实验流水线
│       ├── run_all.py                   # ★ 一键并行全流程 (统一实验 + viz + 甘特图)
│       ├── experiment.py                # 统一实验脚本 (每次 seed 同时产出 4 种算法变体)
│       ├── benchmark.py                 # 对比实验 (RMOEA/D vs MOEA/D + 统计检验)
│       ├── ablation.py                  # 消融实验 (结构解耦验证)
│       ├── visualization.py             # 图表入口 (TFN 三线表 + PF/HV 对比 + 甘特图)
│       ├── tfn_table_charts.py          # ★ TFN 对比三线表 (SCI/booktabs 风格)
│       ├── comparison_charts.py         # PF 与 HV 对比图
│       ├── plot_helpers.py              # 共享绘图工具 (学术配色 / rcParams / 轴样式)
│       ├── ablation_visualization.py    # 消融图表辅助 (结果加载 / 多轮聚合)
│       ├── analysis_report.py           # 综合分析报告与统计检验图表
│       ├── metrics.py                   # HV / 非支配排序
│       ├── logger_setup.py              # 日志配置
│       ├── generate_test_cases.py       # 固定测试用例生成
│       ├── validate_solution.py         # 解验证
│       └── validate_operators.py        # 算子验证
│
├── scripts/                             # 脚本工具
│   └── build_ppt.py                     # 自动生成 4 阶段学术汇报 PPT
│
├── tests/                               # 单元测试
│   └── test_refactor.py                 # 重构验证 (MOEA/D 继承关系、Q-learning 策略、结果字段)
│
├── data/                                # Brandimarte 原始实例 (Mk01~Mk10.fjs)
├── test_cases/                          # 固定测试用例 (seed=42)
├── logs/                                # 详细日志文件 (rmoea_d_{ts}.log)
├── results/                             # 实验结果
│   ├── experiment/                      # ★ 统一实验主数据源 (每次 seed 产出 4 变体)
│   │   ├── mk01/
│   │   │   └── run_{seed}_rmoea_{exp_id}.json   # 含 rmoea_d/qpas_only/rvns_only/moea_d + fuzzy_makespan/fuzzy_workload
│   │   ├── ...
│   │   └── aggregate_{exp_id}.json      # 聚合统计
│   ├── benchmark/                       # 对比实验 (由 experiment 派生)
│   │   ├── mk01/
│   │   │   ├── RMOEA_D/                 #   per-run JSONs
│   │   │   └── MOEA_D/
│   │   └── benchmark_aggregate_{exp_id}.json
│   ├── ablation/                        # 消融实验 (由 experiment 派生)
│   │   └── mk01/ablation_results_{exp_id}.json
│   └── schedules/                       # 调度数据 (甘特图源)
│       └── mk01/mk01_full_schedule_run{idx}_{exp_id}.json
│
├── charts/                              # 可视化图表
│   ├── benchmark/
│   │   ├── benchmark_tfn_comparison_table.png   # ★ TFN 三线表
│   │   ├── hv_comparison.png
│   │   └── mk01/pareto_front_comparison.png     # per-instance, Mk01~Mk10
│   ├── ablation/
│   │   ├── ablation_tfn_comparison_table.png    # ★ TFN 三线表
│   │   ├── hv_comparison.png
│   │   └── mk01/pareto_front_comparison.png     # per-instance, Mk01~Mk10
│   └── schedules/                       # 甘特图 (--skip_gantt 时不生成)
│       └── mk01/...
│
└── docs/                                # 论文与设计文档
```

---

## 环境依赖

```powershell
pip install -r requirements.txt
# 或 pip 可编辑安装
pip install -e .
```

最低版本要求：Python >= 3.9, numpy >= 1.24, scipy >= 1.10, matplotlib >= 3.7
无上限约束：matplotlib >= 3.9（含 3.10）与 numpy 2.x 均已实测通过，`plot_helpers.get_colormap()` 已内置新旧版本兼容处理。

---

## 使用方法

### `run_all` — 一键并行全流程（推荐）

最简命令，10 实例的统一实验 + 可视化全部自动完成：

```powershell
python main.py run_all
```

核心参数：

```powershell
python main.py run_all `
    --instances mk01 mk02 mk03 mk04 mk05 mk06 mk07 mk08 mk09 mk10 `
    --n_pop 100 --max_gen 200 --n_runs 30 `
    --max_workers 4
```

Phase 跳过（调试/增量运行）：

```powershell
python main.py run_all --skip_viz                  # 仅跑实验，不出图
python main.py run_all --skip_gantt                # 跳过甘特图
python main.py run_all --skip_viz --skip_gantt     # 仅计算
```

完整参数列表见 `python main.py run_all --help`
（`--seed` / `--crossover_rate` / `--fixed_T` / `--data_dir` / `--output_dir` /
`--max_workers` / `--n_workers_inner` / `--timeout_per_task`）。

**统一实验 ID**：每次 `run_all` 自动生成 `rmoea_YYYYMMDD_HHMMSS` 贯穿所有文件、日志、图表脚注，便于追溯。

**统一实验设计（单次产出 4 变体）**：
- 每个 seed 只跑一次，**同时产出** `rmoea_d` / `qpas_only` / `rvns_only` / `moea_d` 四种变体
- 一次运行即同时满足对比实验与消融实验需求，**无需分别跑两遍**（旧版双实验流程已废弃）
- `results/experiment/` 为唯一主数据源，`results/benchmark/` 与 `results/ablation/`
  由它派生（兼容旧版路径），per-run JSON 同时供甘特图直接渲染，零重复计算
- 数据复用：benchmark 与消融共用同一批 run，节省约 **53% 算力**

**TFN 对比三线表**：`python main.py visualize` 会从 `results/experiment/` 读取
每种变体的 `fuzzy_makespan` / `fuzzy_workload`，生成两张 SCI/booktabs 风格的三线表
（分组表头 + 最优变体高亮 + 显著性标注）：
- `charts/benchmark/benchmark_tfn_comparison_table.png`（RMOEA/D vs MOEA/D，Wilcoxon 检验）
- `charts/ablation/ablation_tfn_comparison_table.png`（4 变体，Friedman 检验）

---

### 其他命令

```powershell
# 单实例优化
python main.py optimize --instance Mk01 --n_pop 100 --max_gen 200 --seed 42

# 对比实验 (RMOEA/D vs MOEA/D)
python main.py benchmark --instances mk01 mk10 --n_runs 30

# 消融实验
python main.py ablation --instances mk01 mk02 --n_runs 5

# 图表生成 (读取已有 results/ 数据 → TFN 三线表 + PF/HV 对比)
python main.py visualize

# 结果分析报告 (analysis_report)
python main.py analyze

# 生成学术汇报 PPT (4 个阶段)
python scripts\build_ppt.py --output-dir results\pptx

# 单独生成某一阶段 PPT
python scripts\build_ppt.py --phase 1   # 背景与问题
python scripts\build_ppt.py --phase 2   # 算法方法
python scripts\build_ppt.py --phase 3   # 实验设计
python scripts\build_ppt.py --phase 4   # 结果分析
```

---

## 并行架构

两级进程池，避免 CPU 过载：

```
外层 ProcessPool (--max_workers=4)
└─ mk01 ── 内层 ProcessPool → n_runs × 2 algorithms 并发
└─ mk02 ── 内层 ProcessPool → ...
└─ mk03 ── ...
└─ mk04 ── ...
```

内层 worker 数由环境自动计算：`max(1, min(tasks, total_cpu // outer_workers))`，确保不超核心数。

---

## RVNS 算子

| 算子 | 描述 |
|------|------|
| LS1 随机机器交换 | 随机选工序换到另一台候选机器 |
| LS2 最小时间选择 | 选工序换到加工时间最短的机器 |
| LS3 负载均衡调整 | 负载最大机器移走一个工序 |
| LS4 位置交换 | 随机交换两个工序位置 |
| LS5 位置插入 | 随机将工序插入另一位置 |

**记忆机制**：lp=40 代滑动窗口统计成功/失败次数，轮盘赌动态分配选择概率。

---

## 参数设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| n_pop | 100 | 种群大小 |
| max_gen | 200 | 最大代数 (benchmark & 消融统一) |
| crossover_rate | 0.8 | 交叉率 |
| Q-learning α | 0.4 | 学习率 |
| Q-learning γ | 0.6 | 折扣因子 |
| Q-learning ε | 0.8 | 探索率 |
| 邻域候选 T | {5,10,15,20} | Q-PAS 动作空间 |
| RVNS lp | 40 | 记忆窗口长度 |
| fixed_T | 10 | MOEA/D 固定邻域大小 |
| n_runs | 3~30 | 独立运行次数 |

---

## 算法流程

```
输入: Np, Gen, CR, Q-learning 参数, RVNS 参数
输出: 最终 PF + HV

1. 初始化权重向量 W (Np), Q-table (4×4), RVNS 记忆
2. 初始化种群 P (MIX3), 参考点 z*
3. For gen = 1..Gen:
   a. RVNS: 对每个解局部搜索, 更新记忆和选择概率
   b. Q-learning 选择 T, 重建邻域 B(i)
   c. MOEA/D 一代: POX+UX 交叉, 变异, 修复, 更新
   d. 更新 PF 和精英存档
   e. Q-learning 学习: 根据 ΔCV, ΔDV 更新 Q-table
4. 输出 PF 和 HV
```

---

## 统计检验

- **Wilcoxon 符号秩检验**：配对比较 RMOEA/D vs MOEA/D 的 HV/Makespan/Workload
- **Friedman 检验**：多算法整体排序
- **效应量** (Cohen's d / Cliff's delta)：量化差异幅度
- 报告格式：Mean ± Std [min, max, median]

---

## 故障排查

### Windows 平台 ProcessPoolExecutor 报错

```
RuntimeError: An attempt has been made to start a new process before the current process has finished...
```

**原因**：Windows 缺少 `fork()`，必须用 `if __name__ == '__main__'` 保护入口点。

**解决**：始终通过 `python main.py` 运行，不要从 Jupyter/IPython 交互式调用 `run_all.py` 等脚本。

---

### 日志文件过大

长期运行或大批量实验时，`logs/` 目录下的日志文件可能持续增长至数 GB。

**解决**：
- 手动清理：`Remove-Item logs\*.log`
- 未来版本将使用 `RotatingFileHandler` 自动轮转（10MB/5 个备份）

---

### `import rmoea_d` 失败 / ModuleNotFoundError

```
ModuleNotFoundError: No module named 'rmoea_d'
```

**原因**：Python 未找到 `src/` 下的包。

**解决**：
```powershell
# 方案1：将 src 加入 PYTHONPATH (PowerShell)
$env:PYTHONPATH = "src;$env:PYTHONPATH"

# 方案2：pip 可编辑安装（推荐）
pip install -e .
```

---

### 甘特图中文显示异常（方框/乱码）

**原因**：系统缺少中文字体。

**解决**：
- **Windows**：通常已安装 SimHei，无需额外操作
- **Linux**：`sudo apt install fonts-noto-cjk`
- **macOS**：系统自带苹方，或 `brew install font-noto-sans-cjk`

---

### 内存不足 (OOM)

大实例（Mk10: 20 工序 × 15 机器）配合大种群（n_pop=100）时，解向量和解码中间结果占用可观内存，30 轮并行运行可能触发 OOM。

**解决**：
```powershell
# 减少内层并行数
python main.py run_all --n_workers_inner 1 --max_workers 2

# 降低种群大小
python main.py run_all --n_pop 50
```

---

### 运行时间过长 / 任务挂起

**解决**：使用超时保护，超时任务自动跳过并记录警告：
```powershell
python main.py benchmark --instances mk01 --n_runs 30 --timeout_per_task 600
python main.py run_all --timeout_per_task 600
```

---

### 实验结果缺少某些轮次

**原因**：部分运行可能因超时或其他异常被跳过，日志中会记录 `[WARNING]` 信息。

**解决**：检查 `logs/` 目录中的日志文件，搜索 `Timeout` 或 `Error` 关键词定位失败原因。

---

### `matplotlib` 版本不兼容

使用 matplotlib >= 3.9 时，已弃用的 `plt.cm.get_cmap` 被移除，可能导致甘特图生成失败。

**解决**：项目已内置版本兼容处理（`rmoea_d.utils.plot_helpers.get_colormap()`，
优先走 `plt.colormaps[name]` 并回退到旧 API），matplotlib 3.7 ~ 3.10 均可直接使用，
无需降级或加版本上限。

---

## 参考文献

- *A reinforcement learning based RMOEA/D for bi-objective fuzzy flexible job shop scheduling*
- 数据集：Brandimarte Flexible Job Shop Scheduling Instances (Mk01~Mk10)