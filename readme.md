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
- **绘图复用**: `plot_helpers.py` 统一提供学术配色 / rcParams / 轴样式 / 数据来源脚注，
  被 `visualization.py`、`comparison_charts.py`、`analysis_report.py` 共用，消除重复代码
- **图表分工**: `visualization.py` 出 TFN 三线表 + PF/HV 对比；`analysis_report.py` 出带统计检验的
  对比 / 消融分析图；`ablation_visualization.py` 为独立运行的消融补充图（未被流水线调用）
- **统一日志**: 双路日志系统（控制台 INFO + 文件 DEBUG），关键变量与执行计时全面记录
- **单元测试**: `tests/test_refactor.py` 覆盖重构后核心功能的正确性验证

---

## 文件结构

```
RMOEA_D/
├── main.py                              # 唯一入口 (optimize / analyze / visualize / benchmark / ablation / run_all)
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
│       ├── run_all.py                   # ★ 一键并行全流程 (统一实验 + 可视化 + 统计分析 + 甘特图)
│       ├── experiment.py                # 统一实验脚本 (每次 seed 同时产出 4 种算法变体)
│       ├── benchmark.py                 # 对比实验 (RMOEA/D vs MOEA/D + 统计检验)
│       ├── ablation.py                  # 消融实验 (结构解耦验证)
│       ├── visualization.py             # 出图入口 (TFN 三线表 + PF/HV 对比；甘特图函数供 run_all 调用)
│       ├── tfn_table_charts.py          # ★ TFN 对比三线表 (SCI/booktabs 风格)
│       ├── comparison_charts.py         # PF 与 HV 对比图
│       ├── plot_helpers.py              # 共享绘图工具 (学术配色 / rcParams / 轴样式)
│       ├── ablation_visualization.py    # 消融补充图表 (独立运行, 流水线未调用)
│       ├── analysis_report.py           # 综合分析报告与统计检验图表 (run_all Phase 2b)
│       ├── metrics.py                   # HV / 非支配排序
│       ├── logger_setup.py              # 日志配置
│       ├── generate_test_cases.py       # 固定测试用例生成
│       ├── validate_solution.py         # 解验证
│       └── validate_operators.py        # 算子验证
│
├── scripts/                             # 脚本工具
│   ├── build_ppt.py                     # 自动生成 4 阶段学术汇报 PPT
│   ├── ablation_analysis.py             # ★ 消融分析 (参考集归一化 HV + 配对检验 + 2×2 因子分解)
│   ├── paper_table5_audit.py            # ★ 论文 Table 5 阶梯逐步骤审计 (符号检验 + 排名/效应量对照)
│   ├── t_leverage_sweep.py              # ★ T 杠杆 / Q-PAS / 局部搜索实验台 (分批·增量·可续跑)
│   ├── t_leverage_analysis.py           # ★ 上述实验台的分析 (参考集归一化 + 组件分解)
│   ├── qpas_audit.py                    # ★ Q-PAS 逐条审计 (式(13)不动点 / CV 尺度 / Q 表语义 / RNG)
│   ├── ablation_ladder.py               # ★ 论文 §5.4 六级阶梯实验台 (8 臂, 增量落盘, 断点续跑)
│   ├── ablation_ladder_analysis.py      # ★ 阶梯分析 (逐级 HV+Friedman / 相邻级配对 / 2×2 / T 分布)
│   ├── ladder_run_all.py                # ★ 阶梯分批驱动 (按实例切批, 超时重试)
│   ├── ladder_plot.py                   # ★ 阶梯 4 面板图 (与文档表格逐位同口径)
│   └── paper_cmp_plot.py                # ★ 论文对照图 (按种子交集配对)
│
├── tests/                               # 单元测试
│   └── test_refactor.py                 # 重构验证 + 消融/阶梯诊断修复的回归锁 (80 cases)
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
├── charts/                              # 可视化图表 (由 run_all 的 Viz 阶段产出)
│   ├── benchmark/
│   │   ├── benchmark_tfn_comparison_table.png   # ★ TFN 三线表      (visualization.py)
│   │   ├── hv_comparison.png                    # PF / HV 对比      (comparison_charts.py)
│   │   ├── mk01/pareto_front_comparison.png     # per-instance, Mk01~Mk10
│   │   └── benchmark_{hv,makespan,workload}_comparison_{exp_id}.png
│   │       benchmark_{cohens_d,pvalue_heatmap,effect_summary,stats_card}_{exp_id}.png
│   │                                            # 统计分析图表       (analysis_report.py)
│   ├── ablation/
│   │   ├── ablation_tfn_comparison_table.png    # ★ TFN 三线表      (visualization.py)
│   │   ├── hv_comparison.png
│   │   ├── mk01/pareto_front_comparison.png     # per-instance, Mk01~Mk10
│   │   ├── ablation_{hv,makespan,runtime}_comparison.png
│   │   │   ablation_{cohens_d_matrix,cohens_d_per_instance,pvalue_heatmap,stats_card}.png
│   │   │                                        # 统计分析图表       (analysis_report.py)
│   │   ├── paper_ladder_reproduction.png        # ★ 六级阶梯 4 面板  (ladder_plot.py)
│   │   ├── paper_vs_reproduction.png            # ★ 论文对照 3 面板  (paper_cmp_plot.py)
│   │   ├── mk10_t_leverage_qpas_fix.png         # T 杠杆 / Q-PAS 修复对照 (t_leverage_analysis.py)
│   │   └── qpas_implementation_audit.png        # Q-PAS 实现审计 3 面板   (qpas_audit.py)
│   └── schedules/                       # 甘特图 (--skip_gantt 时不生成)
│       └── mk01/...
│
└── docs/                                # 论文与设计文档
    ├── ablation-qpas-rvns-diagnosis.md  # ★ 消融诊断：4 个 bug + T 杠杆 + 六级阶梯复现 (§9)
    ├── paper-vs-reproduction.md         # ★ 与论文 (Li et al. 2022) 的逐条对照
    ├── qpas-implementation-audit.md     # ★ Q-PAS 实现审计 (逐条核对 + 6 处论文笔误)
    ├── rmoead-paper.md                  # 论文原文 (Markdown 抽取)
    ├── rmoead-paper-cn.md               # 论文中文翻译
    └── superpowers/                     # 设计规格 (specs/) 与实现计划 (plans/)
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

流水线阶段（实例级并行，`--max_workers` 控制外层进程数）：

| 阶段 | 内容 | 产出 | 跳过开关 |
|------|------|------|----------|
| Phase 1 | 统一实验：10 实例 × n_runs × 4 变体 | `results/experiment/` + `results/benchmark/`(聚合) | — |
| Phase 2 | 可视化 `visualization.py` | TFN 三线表 + PF/HV 对比 | `--skip_viz` |
| Phase 2b | 统计分析 `analysis_report.py` | 带检验的对比 / 消融分析图 | `--skip_viz` |
| Phase 3 | 甘特图并行渲染 | `charts/schedules/` | `--skip_gantt` |

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

# 消融结果分析 (参考集归一化 HV + 配对检验 + 2×2 因子分解)
python scripts\ablation_analysis.py --results_dir results\experiment --instance mk01

# 论文 (Li et al. 2022) 消融阶梯审计：逐步骤效应量 + 精确符号检验
python scripts\paper_table5_audit.py

# T 杠杆 / Q-PAS / 局部搜索对照实验台 (分批·增量落盘·可断点续跑)
python scripts\t_leverage_sweep.py --instance Mk10 --arms RandVNS,RVNSonly,RVNSonly_t3
python scripts\t_leverage_analysis.py --lab_json logs\_mk10_lab.json

# Q-PAS 实现审计：逐条核对论文 Algorithm 3 + 5 项数值验证
# (式(13) 不动点 / CV 尺度支配 / 状态翻转率 / Q 表语义 / RNG 共享)
python scripts\qpas_audit.py --instance Mk10 --seed 42

# —— 论文 §5.4 六级消融阶梯（8 臂 × 10 实例 × 30 seeds = 2400 runs）——
# 分批驱动：按实例切批 + 超时自动重试，配合实验台断点续跑可稳定跑完
python scripts\ladder_run_all.py --seeds 30 --workers 6
# 阶梯分析：逐级 HV+Friedman / 相邻级配对检验 / 2×2 因子分解 / T 分布 / 逐实例单调性
python scripts\ablation_ladder_analysis.py --lab_json logs\ablation_ladder.json
# 阶梯 4 面板图（与本文档表格逐位同口径）
python scripts\ladder_plot.py --lab_json logs\ablation_ladder.json
# 论文对照图（按种子交集配对；默认读 logs\_paper_audit.json + logs\_mk10_lab.json）
python scripts\paper_cmp_plot.py
```

---

## HV 归一化口径（重要）

跨算法比较 HV **必须**用同一套归一化边界，否则指标不可比。本项目统一采用
**参考集归一化（reference-set normalization）**：

1. 求解器内联用 `instance_hv_bounds(instance)` 由实例数据确定性推出边界
   （同一实例所有 run / 所有算法共用，结果 JSON 的 `hv_bounds` 字段）；
2. 统一实验跑完后，`experiment._retune_hv_reference_set()` 再用
   **该实例所有变体、所有 run 的前沿并集** 重算一次 `final_hv`，
   并写入 `hv_norm_bounds` / `hv_ref_point` / `hv_definition` 字段。

不要把 `final_hv` 与「用每条前沿自己的 min/max 归一化」得到的数值混用——
后者会把任意前沿拉伸到单位盒，指标对整体优劣不敏感，**无法区分算法优劣**。

## Q-PAS 奖励模式（`ql_reward_mode`）

| 值 | 奖励定义 | 说明 |
|----|----------|------|
| `"dv"` | ΔDV > 0 → 10，否则 0 | **默认**，严格照论文式 (20) |
| `"cv_dv"` | ΔCV>0 与 ΔDV>0 各计 5 分 | 把收敛性纳入奖励 |
| `"hv"` | ΔHV > 0 → 10，否则 0 | 奖励直接对齐最终评价指标 |
| `"hv_cont"` | 连续 HV 增量 | 二值奖励的连续化版本 |

四种模式两两差异均不显著（Mk01 上 Friedman p=0.86；Mk10 上亦然）。
**根因不是奖励设计，而是 Q-PAS 本身的收益上限**：Mk10 上 T 确实是强杠杆
（固定 T 扫参 Friedman p=4.7e-05\*\*\*，最优 T=50 落在论文候选集 {5,10,15,20} 之外），
但在四种语境下 Q-PAS 相对配对对照均未达显著（+1.56% / +0.56% / −1.49%，p≥0.13）。
详见 `docs/ablation-qpas-rvns-diagnosis.md` 与 `docs/paper-vs-reproduction.md`。

## Q-PAS 的 CV 归一化（`ql_cv_normalize`）

论文式 (14) 用**原始目标值**算 CV，且未规定归一化。当两个目标量纲悬殊时，
CV 被大量纲目标独占——实测 Mk10 上一次 200 代运行里
`f2` 占 `CV²` 的 **96.5%**、`f1` 仅 **3.5%**，而状态只由 `ΔCV` 的**符号**决定，
于是 makespan 对 Q-PAS 的状态完全隐形。把同一批前沿的两目标拉回同一尺度重算，
**44.7% 的历史状态会翻转**——状态划分不是尺度不变的。

| 值 | 行为 |
|----|------|
| `False`（**默认**） | 严格照论文，用原始目标值计算（复现优先） |
| `True` | 先用 `hv_bounds`（缺省用本代前沿范围）归一到同一尺度 |

**实测结论：打开它没有救回 Q-PAS**。Mk10 × 30 seeds 对照：
`0.82740 → 0.82458`（dv）、`0.83590 → 0.83091`（hv+wide），HV 略降且仍 n.s.
它是定义层面的瑕疵（值得写进复现说明），但不是 Q-PAS 失效的原因。

**真正的原因**：排除 `T=5` 后，T 的收益曲线几乎是平的（T10/15/20/50/100
极差仅 0.0168，而 run 间 σ 为 0.034~0.055）。T 的杠杆几乎全部来自
"别用 T=5"，而非"选到最优 T"。完整审计见 `docs/qpas-implementation-audit.md`。

## RVNS 算子选择模式（`rvns_mode`）

| 值 | 行为 | 说明 |
|----|------|------|
| `"rl"` | 按 SM/FM 轮盘赌选算子 | **默认**，论文 RVNS |
| `"random"` | 五算子等概率随机选 | 论文 §4.6 用法 (1)，即变体 RMOEA/D3 |

`rvns_mode` 的存在是为了把「加上局部搜索本身」与「RL 引导选算子」在消融里**分开**——
论文阶梯里随机 VNS 早在 D3 就位，所以它测的 RVNS 增益只是后者。
`ls_trials`（每代邻域尝试次数）是比算子选择更强的杠杆：

| 对照（Mk10, n=30） | ΔHV 相对 | p |
|---|---|---|
| 加上局部搜索本身（论文 `ls_trials=1`） | +5.68% | 1.8e-05 \*\*\* |
| RL 引导选算子 vs 随机选算子（`ls_trials=1`） | −0.25% | 0.53 n.s. |
| RL 引导选算子 vs 随机选算子（`ls_trials=3`） | **+1.86%** | **0.045 \*** |
| 邻域尝试 1 → 3 次 | **+4.68%** | 3.5e-05 \*\*\* |
详见 `docs/ablation-qpas-rvns-diagnosis.md`。

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