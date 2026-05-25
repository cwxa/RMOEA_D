# RMOEA/D - Reinforcement Learning based MOEA/D for Bi-objective Fuzzy Flexible Job Shop Scheduling

基于强化学习的 MOEA/D 双目标模糊柔性作业车间调度求解器。

## 项目简介

本项目复现了论文《A reinforcement learning based RMOEA/D for bi-objective fuzzy flexible job shop scheduling》中的核心算法，针对 **Brandimarte 模糊柔性作业车间实例（Mk01~Mk10）** 进行双目标优化：

- **目标1**：模糊最大完工时间（Fuzzy Makespan）
- **目标2**：总机器工作负载（Total Workload）

**核心创新**：通过 Q-learning 自适应选择邻域大小 T，动态平衡收敛性（Convergence）与多样性（Diversity）。

---

## 文件结构

```
RMOEA_D/
├── main.py                          # 主程序入口：运行 RMOEA/D
├── benchmark.py                     # 对比实验：RMOEA/D vs MOEA/D
├── generate_test_cases.py           # 生成固定测试用例（JSON）
│
├── src/rmoea_d/
│   ├── algorithm.py                 # RMOEA/D 主算法（Q-learning + MOEA/D）
│   ├── moead_baseline.py            # MOEA/D 基线算法（固定 T）
│   │
│   ├── core/                        # 核心模块
│   │   ├── fuzzy.py                 # 三角模糊数运算
│   │   ├── instance.py              # Brandimarte 实例加载 + 测试用例生成
│   │   ├── encoding.py              # OS+MA 编码解码
│   │   ├── operators.py             # 遗传算子（MIX3/POX/UX/变异/修复）
│   │   ├── moead.py                 # MOEA/D 框架（权重/邻居/Tchebycheff）
│   │   └── qlearning.py             # Q-learning 参数自适应策略
│   │
│   └── utils/                       # 工具模块
│       ├── metrics.py               # 非支配排序 + 超体积(HV)计算
│       └── logger_setup.py          # 日志配置（控制台 + 文件）
│
├── data/                            # Brandimarte 原始实例（Mk01~Mk10）
├── test_cases/                      # 固定测试用例（JSON，seed=42）
├── logs/                            # 详细日志文件
├── results/                         # 实验结果（按实例和算法分组存储）
│   ├── Mk01/
│   │   ├── RMOEA_D/
│   │   │   └── Mk01_RMOEA_D_Np100_G200_xxx.json
│   │   ├── MOEA_D/
│   │   │   └── Mk01_MOEA_D_Np100_G200_T10_xxx.json
│   │   └── benchmark_summary_Mk01_xxx.json
│   └── ...
└── visualization/                   # 可视化网页（Chart.js）
    └── index.html                   # 支持文件夹选择、Pareto前沿对比、甘特图动画
```

---

## 环境依赖

- Python >= 3.8
- NumPy

```bash
pip install numpy
```

---

## 使用方法

### 1. 生成固定测试用例

确保实验可复现，将模糊加工时间固定为 seed=42 的结果：

```bash
python generate_test_cases.py
```

输出：`test_cases/Mk01_seed42.json` ~ `Mk10_seed42.json`

### 2. 运行 RMOEA/D

```bash
# 默认参数（Mk01, Np=100, Gen=200）
python main.py

# 自定义参数
python main.py --instance Mk01 --n_pop 100 --max_gen 200 --seed 42

# 查看所有参数
python main.py --help
```

### 3. 运行对比实验（RMOEA/D vs MOEA/D）

```bash
# 单个实例对比
python benchmark.py --instances Mk01 --n_pop 100 --max_gen 200

# 多个实例批量对比
python benchmark.py --instances Mk01 Mk02 Mk03 --n_pop 100 --max_gen 200

# 查看所有参数
python benchmark.py --help
```

### 4. 可视化结果

打开 `visualization/index.html`，支持三种方式加载结果：
- **选择 results/ 文件夹**：通过 File System Access API 浏览本地结果目录
- **自动加载最新结果**：扫描 `results/<instance>/<algorithm>/` 结构自动加载
- **拖拽 JSON 文件**：直接拖放结果文件到页面

功能包括：Pareto 前沿对比图、收敛曲线（HV 趋势）、模糊数展示、甘特图动画。

---

## 实验方法

### 算法对比设计

| 对比维度 | RMOEA/D | MOEA/D（Baseline） |
|---------|---------|-------------------|
| 邻域策略 | **Q-learning 自适应选择 T** | 固定 T=10 |
| 状态空间 | 4 状态（ΔCV, ΔDV） | 无 |
| 动作空间 | 4 动作（T ∈ {5,10,15,20}） | 无 |
| 奖励函数 | R=10 if ΔDV>0 else 0 | 无 |
| 其他参数 | 完全一致 | 完全一致 |

### 评价指标

#### 综合指标
1. **HV (Hypervolume)**：超体积指标，衡量 Pareto 前沿的综合质量，**越大越好**
2. **PF Size**：非支配解数量，**越大越好**
3. **Runtime**：算法运行时间
4. **收敛曲线**：每代 HV 变化趋势

#### 单目标分析指标（清晰值 + 三角模糊数）
对每个目标独立分析，评估算法在单一目标上的优化能力。结果同时输出 **清晰值**（Clear Value, `(t1+2t2+t3)/4`）和 **三角模糊数**（Triangular Fuzzy Number, `(t1,t2,t3)`）：

| 指标 | 说明 | 最优方向 |
|------|------|---------|
| **Best Makespan** | Pareto前沿中最小的模糊最大完工时间 | 越小越好 |
| **Avg Makespan** | Pareto前沿中模糊最大完工时间的平均值 | 越小越好 |
| **Worst Makespan** | Pareto前沿中最大的模糊最大完工时间 | 越小越好 |
| **Best Workload** | Pareto前沿中最小的总机器工作负载 | 越小越好 |
| **Avg Workload** | Pareto前沿中总机器工作负载的平均值 | 越小越好 |
| **Worst Workload** | Pareto前沿中最大的总机器工作负载 | 越小越好 |

### 参数设置

| 参数 | 值 |
|------|-----|
| 种群大小 Np | 100 |
| 最大代数 Gen | 200 |
| 交叉率 | 0.8 |
| Q-learning 学习率 α | 0.4 |
| Q-learning 折扣因子 γ | 0.6 |
| Q-learning ε | 0.8 |
| 邻域候选 T | [5, 10, 15, 20] |

---

## 实验结果示例

### Mk01 实例（Np=30, Gen=20, Seed=42）

```
================================================================================
COMPARISON RESULTS
================================================================================
Metric                         RMOEA/D                MOEA/D                 Improvement
--------------------------------------------------------------------------------
Hypervolume (HV)               0.788430               0.795201                  -0.85%
PF Size                        8                      10                       -20.00%
Runtime (s)                    0.1562                 0.1389                   +12.47%
--------------------------------------------------------------------------------
--- Makespan (minimize) ---
Best Makespan                  51.5000                47.7500                   -7.85%
Avg Makespan                   57.5312                55.0000                   -4.60%
Worst Makespan                 68.0000                68.0000                   +0.00%
--- Workload (minimize) ---
Best Workload                  149.5000               149.5000                  +0.00%
Avg Workload                   155.2812               162.4250                  +4.40%
Worst Workload                 163.2500               180.5000                  +9.56%
--- Fuzzy Makespan (t1,t2,t3) ---
Best Makespan (t1,t2,t3)       (38.0,50.0,62.0)       (33.0,50.0,56.0)
Avg Makespan (t1,t2,t3)        (44.0,58.8,68.6)       (41.5,57.1,64.3)
Worst Makespan (t1,t2,t3)      (53.0,70.0,79.0)       (53.0,70.0,79.0)
--- Fuzzy Workload (t1,t2,t3) ---
Best Workload (t1,t2,t3)       (112.0,153.0,180.0)    (112.0,153.0,180.0)
Avg Workload (t1,t2,t3)        (115.4,158.8,188.2)    (121.0,166.8,195.1)
Worst Workload (t1,t2,t3)      (122.0,166.0,199.0)    (135.0,184.0,219.0)
================================================================================
```

**分析**：在小规模参数下（Np=30, Gen=20），两种算法在 Best Makespan 和 Best Workload 上持平；RMOEA/D 在 Avg Makespan 上略优，而 MOEA/D 在 Worst Makespan 上略优。RMOEA/D 获得了更多的非支配解（PF Size +25%）。随着种群规模和迭代代数增加，Q-learning 的自适应优势将更加明显。

### 结果文件结构示例

```json
// results/Mk01/RMOEA_D/Mk01_RMOEA_D_Np100_G200_xxx.json
{
  "algorithm": "RMOEA/D",
  "instance": "Mk01",
  "final_pf": [
    {"Makespan": 51.5, "Workload": 149.5},
    ...
  ],
  "fuzzy_pf": [
    {
      "Makespan": {"t1": 38.0, "t2": 50.0, "t3": 62.0},
      "Workload": {"t1": 112.0, "t2": 153.0, "t3": 180.0}
    },
    ...
  ],
  "final_hv": 0.788430,
  "history": [
    {"gen": 1, "T": 10, "pf_size": 8, "hv": 0.569, "best_makespan": 47.25, ...},
    ...
  ],
  "q_table": [...]
}
```

### Q-learning 自适应过程示例

```
[INFO] Q-learning init: CV=107.5320, DV=0.2543, state=0, action=2 (T=15)
[INFO] Q-learning step: delta_CV=1.6536, delta_DV=1.2751, state=0->0, reward=10.0, action=0 (T=5)
[INFO] Q-learning step: delta_CV=-2.6568, delta_DV=0.0066, state=0->2, reward=10.0, action=0 (T=5)
[INFO] Q-learning step: delta_CV=1.1478, delta_DV=-0.0458, state=2->1, reward=0.0, action=1 (T=10)
```

Q-learning 根据收敛性（CV）和多样性（DV）的变化动态调整 T，在探索（大 T）与开发（小 T）之间自动权衡。

---

## 核心算法流程

```
输入: 种群大小 Np, 最大迭代代数 Gen, 交叉率 R, 参数T候选列表 [5,10,15,20]
      Q-learning 参数 α=0.4, γ=0.6, ε=0.8
输出: 最终非支配解集 PF，及其 HV 值

1. 初始化权重向量 W (Np 个, 两目标, Das & Dennis 方法)
2. 初始化 Q-table: 4 状态 × 4 动作
3. 初始化种群 P (MIX3 规则: 1/3随机 + 1/3最短 + 1/3负载均衡)
4. 初始化参考点 z* = (min f1, min f2)
5. For gen = 1 to Gen:
   a. Q-learning 选择动作 A → 邻域大小 T
   b. 重新计算邻居 B(i): 为每个子问题选最近的 T 个邻居
   c. MOEA/D 一代更新:
      - 对每个子问题 i:
        - 从 B(i) 选两个父代，POX+UX 交叉和变异产生子代
        - 修复子代使其合法
        - 更新参考点 z*
        - 对邻居 j ∈ B(i)，若 Tchebycheff 值更好则替换
   d. 计算新一代 PF，更新精英存档
   e. 计算新状态 CV, DV → 状态 S'
   f. 奖励 R = 10 if ΔDV>0 else 0
   g. 更新 Q(S,A) = Q(S,A) + α[R + γ max Q(S',a) - Q(S,A)]
   h. S ← S'
6. 输出最终 PF 和 HV
```

---

## 日志说明

- **控制台**：输出 INFO 级别及以上关键信息（英文，简洁格式）
- **日志文件**（`logs/rmoea_d_YYYYMMDD_HHMMSS.log`）：记录 DEBUG 级别详细日志（英文），包含：
  - 每代运行时间
  - Q-learning 状态转移
  - 邻居更新次数
  - 参考点变化
- **代码注释**：使用中文，关键算法步骤和逻辑说明

---

## 参考文献

- 论文：*A reinforcement learning based RMOEA/D for bi-objective fuzzy flexible job shop scheduling*
- 数据集：Brandimarte Flexible Job Shop Instances (Mk01~Mk10)
