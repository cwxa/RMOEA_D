# RMOEA/D - Reinforcement Learning based MOEA/D for Bi-objective Fuzzy Flexible Job Shop Scheduling

基于强化学习的 MOEA/D 双目标模糊柔性作业车间调度求解器。

## 项目简介

本项目复现了论文《A reinforcement learning based RMOEA/D for bi-objective fuzzy flexible job shop scheduling》中的核心算法，针对 **Brandimarte 模糊柔性作业车间实例（Mk01~Mk10）** 进行双目标优化：

- **目标1**：模糊最大完工时间（Fuzzy Makespan），以三角模糊数 (t1, t2, t3) 表示
- **目标2**：总机器工作负载（Total Workload）

**核心创新**：
1. **Q-PAS**：通过 Q-learning 自适应选择邻域大小 T，动态平衡收敛性与多样性
2. **RVNS**：基于强化学习的变邻域搜索，包含 5 种局部搜索策略，通过成功/失败记忆动态调整选择概率

---

## 文件结构

```
RMOEA_D/
├── main.py                          # 唯一入口（支持子命令：optimize / benchmark / ablation / visualize）
│
├── src/rmoea_d/
│   ├── algorithm.py                 # RMOEA/D 主算法（Q-PAS + RVNS + MOEA/D）
│   ├── moead_baseline.py            # MOEA/D 基线算法（固定 T）
│   │
│   ├── core/                        # 核心模块
│   │   ├── fuzzy.py                 # 三角模糊数（TFN）运算
│   │   ├── instance.py              # Brandimarte 实例加载 + 测试用例生成
│   │   ├── encoding.py              # OS+MA 编码解码
│   │   ├── operators.py             # 遗传算子（MIX3/POX/UX/变异/修复）
│   │   ├── moead.py                 # MOEA/D 框架（权重/邻居/Tchebycheff）
│   │   ├── qlearning.py             # Q-PAS 参数自适应策略
│   │   └── rvns.py                  # RVNS 变邻域搜索（5种LS算子+记忆机制）
│   │
│   └── utils/                       # 工具模块
│       ├── benchmark.py             # 对比实验脚本（支持 --n_runs 多次独立运行 + Friedman/Wilcoxon 检验）
│       ├── ablation.py              # 消融实验脚本（Full RMOEA/D vs Q-PAS Only vs RVNS Only vs MOEA/D）
│       ├── visualization.py         # 可视化图表生成（输出到 charts/benchmark/）
│       ├── ablation_visualization.py # 消融实验可视化（输出到 charts/ablation/）
│       ├── metrics.py               # 非支配排序 + 超体积(HV)计算
│       ├── logger_setup.py          # 日志配置
│       └── generate_test_cases.py   # 生成固定测试用例
│       └── validate_solution.py     # 解验证工具
│
├── data/                            # Brandimarte 原始实例（Mk01~Mk10）
├── test_cases/                      # 固定测试用例（JSON，seed=42）
├── logs/                            # 详细日志文件
├── results/                         # 实验结果（按实例和算法分组存储）
│   └── mk01/
│       ├── RMOEA_D/
│       ├── MOEA_D/
│       └── benchmark_aggregate_*.json  # 多轮聚合统计结果
├── charts/                          # 可视化图表
│   ├── benchmark/                   # 对比实验图表（按实例分子目录）
│   │   ├── mk01/                   # MK01: pareto_front.png, fuzzy_pareto.png, convergence.png, fuzzy_range.png
│   │   ├── mk10/                   # MK10: 同上
│   │   ├── hv_comparison.png       # 跨实例HV对比
│   │   ├── makespan_comparison.png # 跨实例Makespan对比
│   │   ├── workload_comparison.png # 跨实例Workload对比
│   │   └── ...
│   └── ablation/                    # 消融实验图表（按实例分子目录）
│       ├── mk01/                   # MK01: convergence.png, hv_comparison.png, ...
│       └── ...
│
├── docs/                            # 论文与实验分析文档
│   ├── rmoead-paper.md
│   ├── rmoead-paper-cn.md
│   └── EXPERIMENT_SUMMARY.md
└── readme.md
```

---

## 环境依赖

- Python >= 3.8
- NumPy
- SciPy（用于统计检验：Wilcoxon / Friedman）
- Matplotlib（用于可视化）

```bash
pip install numpy scipy matplotlib
```

---

## 使用方法

### 查看帮助

```bash
python main.py --help               # 查看所有子命令
python main.py optimize --help      # 优化命令参数
python main.py benchmark --help     # 对比实验参数
python main.py ablation --help      # 消融实验参数
```

### 1. 运行单实例优化

```bash
# 默认参数
python main.py optimize

# 自定义参数
python main.py optimize --instance Mk01 --n_pop 100 --max_gen 200 --seed 42
```

### 2. 运行对比实验（RMOEA/D vs MOEA/D）

```bash
# 单实例 30 次独立运行 + 统计检验
python main.py benchmark --instances Mk01 --n_runs 30 --seed 42

# 多实例批量对比
python main.py benchmark --instances mk01 mk10 --n_pop 100 --max_gen 200 --n_runs 30

# 完整参数
python main.py benchmark --instances mk01 mk02 mk03 mk04 mk05 mk06 mk07 mk08 mk09 mk10 \
    --n_pop 100 --max_gen 200 --n_runs 30 --seed 42 \
    --crossover_rate 0.8 --fixed_T 10 --data_dir data --output_dir results
```

### 3. 运行消融实验

```bash
# 四种算法变体对比：Full RMOEA/D, Q-PAS Only, RVNS Only, MOEA/D
python main.py ablation --instances mk01 mk02 mk03 --n_pop 100 --max_gen 200
```

### 4. 生成可视化图表

```bash
python main.py visualize
```

图表将按数据类别分子目录输出：
- `charts/benchmark/` — 对比实验图表（按实例分 `mk01/`, `mk10/` 等子目录）
- `charts/ablation/` — 消融实验图表（按实例分子目录）

---

## RVNS 变邻域搜索

| 算子 | 描述 |
|------|------|
| LS1 随机机器交换 | 随机选一个工序，换到另一台候选机器 |
| LS2 最小时间选择 | 随机选一个工序，换到加工时间最短的机器 |
| LS3 负载均衡调整 | 找到负载最大的机器，移走一个工序 |
| LS4 位置交换 | 随机交换两个工序的位置 |
| LS5 位置插入 | 随机将工序插入到另一位置 |

**记忆机制**：lp=40 代滑动窗口内统计各算子成功/失败次数，轮盘赌动态分配选择概率。

---

## 参数设置

| 参数 | 值 |
|------|-----|
| 种群大小 Np | 100 |
| 最大代数 Gen | 200 |
| 交叉率 CR | 0.8 |
| Q-learning α | 0.4 |
| Q-learning γ | 0.6 |
| Q-learning ε | 0.8 |
| 邻域候选 T | {5, 10, 15, 20} |
| RVNS 记忆长度 lp | 40 |
| 独立运行次数 n_runs | 30（按论文要求） |

---

## 算法流程

```
输入: 种群大小 Np, 最大迭代代数 Gen, 交叉率 CR
      Q-learning 参数 α=0.4, γ=0.6, ε=0.8
      RVNS 参数: 5种LS算子, lp=40
输出: 最终非支配解集 PF，及其 HV 值

1. 初始化权重向量 W (Np 个)
2. 初始化 Q-table: 4 状态 × 4 动作
3. 初始化 RVNS: 成功/失败记忆, 初始概率均匀分布
4. 初始化种群 P (MIX3 规则)
5. 初始化参考点 z* = (min f1, min f2)
6. For gen = 1 to Gen:
   a. RVNS 局部搜索: 对每个解应用 RVNS，更新记忆和选择概率
   b. Q-learning 选择 T
   c. 重新计算邻居 B(i)
   d. MOEA/D 一代更新: POX+UX 交叉、变异、修复、更新
   e. 更新 PF 和精英存档
   f. Q-learning 学习: 根据 ΔCV, ΔDV 更新 Q-table
7. 输出最终 PF 和 HV
```

---

## 统计检验

按照论文要求，每算法每实例独立运行 30 次后进行统计检验：

- **Wilcoxon 符号秩检验**：配对比较 RMOEA/D 与 MOEA/D 的 HV 值，验证显著性差异
- **结果报告**：均值 ± 标准差（Mean ± Std），含 min/max/median

---

## 参考文献

- *A reinforcement learning based RMOEA/D for bi-objective fuzzy flexible job shop scheduling*
- 数据集：Brandimarte Flexible Job Shop Scheduling Instances (Mk01~Mk10)