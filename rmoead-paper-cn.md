# 基于强化学习的 RMOEA/D 求解双目标模糊柔性作业车间调度问题

> **原文标题**: A reinforcement learning based RMOEA/D for bi-objective fuzzy flexible job shop scheduling
> **作者**: Rui Li, Wenyin Gong, Chao Lu
> **单位**: 中国地质大学（武汉）计算机学院
> **发表期刊**: Expert Systems with Applications (ESWA), 2022
> **DOI**: https://doi.org/10.1016/j.eswa.2022.117380

---

## 文章信息

| 项目 | 内容 |
|------|------|
| 数据集链接 | [https://cuglirui.github.io/downloads.htm](https://cuglirui.github.io/downloads.htm) |
| **关键词** | 模糊柔性作业车间调度 (Fuzzy flexible job shop scheduling)、多目标优化 (Multi-objective optimization)、参数自适应 (Parameter adaption)、强化学习 (Reinforcement learning)、MOEA/D |

---

## 摘要 (Abstract)

柔性作业车间调度问题（FJSP, Flexible Job Shop Scheduling Problem）在实际制造中具有重要意义。然而，工件的加工时间在实际制造过程中通常是不确定且可变的。本文提出了一种具有模糊加工时间的多目标 FJSP（MOFFJSP, Multi-objective FJSP with Fuzzy processing time），以优化最大完工时间（makespan）和总机器负载（total machine workload）为目标。为求解 MOFFJSP，本文提出了一种基于强化学习的 MOEA/D 算法，命名为 **RMOEA/D**。RMOEA/D 的主要特点如下：

1. **初始策略**：采用三种规则相结合的初始策略，获得高质量的初始种群；
2. **参数自适应策略**：基于 Q-learning 的参数自适应策略，引导种群选择最佳参数以提高多样性；
3. **变邻域搜索**：基于强化学习的变邻域搜索（RVNS），引导解选择正确的局部搜索方法；
4. **精英档案**：利用精英档案提高被遗弃历史解的利用率。

RMOEA/D 与五种知名相关方法（MOEA/D、NSGA-II、MOEA/D-M2M、NSGA-III 和 IAIS）在三个基准测试集上进行了比较。结果表明，RMOEA/D 优于这五种最先进的算法。

---

## 1. 引言 (Introduction)

### 1.1 研究背景

随着经济全球化的发展，传统柔性制造面临着巨大挑战，难以满足市场需求（Lang et al., 2021; Rifai et al., 2021）。柔性作业车间调度问题（FJSP）是过去几十年来被广泛研究的经典调度问题。许多启发式算法被提出用于求解 FJSP，例如遗传算法（GA）（Yuan et al., 2020）、人工蜂群算法（ABC）（Li, Huang, et al., 2020）、两阶段元启发式算法（Lei et al., 2019）、Jaya 算法（Caldeira & Gnanavelbabu, 2021）和教与学优化算法（TLBO）（Lei et al., 2018）。然而，固定的加工时间过于理想化，无法模拟实际制造过程。在实际柔性制造中，加工时间是不可控的，在一个区间内浮动（Pan et al., 2021; Zhu & Zhou, 2021）。因此，有必要对 FJSP 的加工时间进行模糊化处理。具有模糊加工时间的 FJSP（FFJSP）是 FJSP 的扩展。FJSP 已被证明是 NP-hard 问题，FFJSP 同样是 NP-hard 问题（Pavlov et al., 2019）。因此，研究如何高效求解 FFJSP 具有重要意义。

此外，随着智能制造和工业 4.0 的发展，许多工业企业开始考虑低能耗制造（Lu et al., 2021）。许多考虑能耗的调度模型被提出，包括分布式混合流水车间等。

受标量目标优化问题的启发，Zhang and Li (2007) 提出了基于分解的多目标进化算法（MOEA/D, Multi-objective Evolutionary Algorithm based on Decomposition）用于求解多目标优化问题（MOP）。基于参考向量和 Tchebicheff 函数，MOEA/D 能够同时获得良好的收敛性和多样性。近年来，强化学习（RL, Reinforcement Learning）因其强大的决策和优化能力而成为热门话题。RL 与进化算法之间的合作是解决复杂优化问题的有前途的方向（Gong et al., 2021; Shiue et al., 2018; Zhao et al., 2019）。面对复杂的优化问题，期望通过 RL 与进化计算的协同作用以及 MOEA/D 框架内的问题特定算子来实现优越的性能。

### 1.2 研究动机

MOFFJSP 已被广泛研究了数年。然而，在现有文献中，大多数方法以轮询模式执行局部搜索策略，这效率低下且盲目。此外，算法的性能受限于参数选择。通常需要进行大量的黑盒测试来找到最佳参数。参数选择问题缺乏先验知识，且耗时。

为了同时优化最大完工时间和总机器负载，开发有效的 MOFFJSP 算法具有挑战性和重要意义。基于上述问题，本文提出了一种基于 RL 的 MOEA/D（RMOEA/D），包含以下创新点：

- 首先，为了自适应地引导每个解选择最佳局部搜索策略，提出了基于 RL 的变邻域搜索（RVNS）；
- 其次，为了使 MOEA/D 自动调整参数 T，设计了基于 Q-learning 的参数选择策略（Q-PAS）；
- 接下来，设计了一种集成多种初始策略的初始化方法，以获得高收敛性和多样性的种群；
- 然后，采用离散交叉和变异方法以获得较大的搜索步长；
- 此外，应用精英档案来提高被遗弃解的利用率；
- 最后，为验证 RMOEA/D 的性能，进行了广泛的数值测试，比较结果证明了上述设计的有效性和所提算法在求解 MOFFJSP 方面的优越性。

### 1.3 主要贡献

本文的主要贡献体现在以下五个方面：

1. 设计了一种结合三种策略优势的初始策略，提供具有高收敛性和多样性的初始种群；
2. 提出了基于 Q-learning 的参数选择策略，使 MOEA/D 自动选择最佳 T 以提高 PF（Pareto Front）的多样性；
3. 提出了基于 RL 的变邻域搜索方法，以高效执行多种局部搜索策略；
4. 设计了精英档案，收集历史精英解以提高被遗弃解的利用率；
5. 在 23 个具有不同特征的 FFJSP 实例上验证了 RMOEA/D 的性能。实验结果表明，在更快收敛的条件下，RMOEA/D 优于最先进的算法。

本文的其余部分组织如下：第 2 节介绍文献综述和三角模糊加工时间的基本概念；第 3 节介绍问题描述和建模；第 4 节详细描述所提出的 RMOEA/D 方法；第 5 节展示 RMOEA/D 的数值实验和讨论；第 6 节总结结论。

---

## 2. 相关工作与背景知识 (Related work and background knowledge)

### 2.1 模糊柔性作业车间调度

（注：原文第 2.1 节内容在提取过程中有缺失，此处根据上下文补充相关背景）

FFJSP 的研究已经持续多年。Dorfeshan et al. (2020) 将模糊加工时间抽象为决策者，并提出了一种加权距离近似方法，可以更好地确定工序顺序。Li, Liu, et al. (2020) 提出了 Type-2 模糊加工时间来处理复杂系统的高不确定性，补充了传统三角模糊数的不足。此外，自动调整种群大小是一种有效的技术，但尚未应用于 MOFFJSP。Pan et al. (2021) 提出了一种具有反馈模式的双种群进化算法，弥补了这一空白。然而，据我们所知，MOFFJSP 很少被研究。因此，为 MOFFJSP 设计算法具有重要意义。

### 2.2 MOEA/D

MOEA/D 是一种经典的多目标进化算法（MOEA），已成功应用于各种调度问题，例如置换流水车间调度问题（Pericleous et al., 2017）、卫星测距调度问题（Du et al., 2019）、最优潮流问题（Zhang et al., 2020）、分布式异构混合流水车间调度问题（Shao et al., 2021a）和分布式异构焊接流水车间调度问题（Wang et al., 2021）。鉴于 MOEA/D 的优越性能，使用 MOEA/D 的主要动机如下：

1. MOEA/D（Zhang & Li, 2007）在求解 MOP 时具有出色的多样性；
2. RL 可以使 MOEA/D 选择最佳参数和局部搜索策略，比随机选择更高效。

因此，本文将基于 RL 的 MOEA/D 应用于 MOFFJSP。

### 2.3 强化学习在调度中的应用

近年来，作为一种非常流行的人工智能方法，RL 已广泛应用于车间调度问题。表 1 总结了 RL 在作业车间调度（JSP）中的应用文献。

**表 1. 强化学习在调度中的文献综述**

| 参考文献 | 问题类型 | 目标数量 | 机器数量 | 方法 |
|---------|---------|---------|---------|------|
| (Qu et al., 2015) | 动态作业车间调度 | 单目标 | 多台 | 集中式 RL |
| (Qu et al., 2016) | 动态作业车间调度 | 单目标 | 多台 | RL |
| (Wang & Yan, 2016) | 知识型制造系统 | 单目标 | 多台 | 加权 Q-learning |
| (Shahrabi et al., 2017) | 动态作业车间调度 | 单目标 | 多台 | Q-factor 算法 |
| (Zhang et al., 2017) | 实时作业车间调度 | 单目标 | 多台 | 马尔可夫决策过程 |
| (Palombarini & Martínez, 2018) | 社会技术制造系统 | 单目标 | 多台 | 深度 Q-learning |
| (Waschneck et al., 2018) | 作业车间调度 | 单目标 | 多台 | 深度 Q-network |
| (Shiue et al., 2018) | 实时作业车间调度 | 单目标 | 多台 | RL |
| (Ahmadi et al., 2018) | 作业排序与刀具切换问题 | 单目标 | 单台 | Q-learning |
| (Zhao et al., 2019) | 动态作业车间调度 | 单目标 | 多台 | Q-learning |
| (Palombarini & Martínez, 2019) | 半导体生产调度 | 单目标 | 多台 | 深度 Q-network |
| (Lin et al., 2019) | 社会技术制造系统 | 单目标 | 多台 | 深度 Q-network |
| (Han & Yang, 2020) | 作业车间调度 | 单目标 | 多台 | Dueling Double Deep Q-network |
| (Luo, 2020) | 动态作业车间调度 | 单目标 | 多台 | 深度 Q-learning |
| (Wang, 2020) | 动态作业车间调度 | 多目标 | 多台 | 加权 Q-learning |
| (Liu et al., 2020) | 作业车间调度 | 单目标 | 多台 | Actor-Critic DRL |

Qu et al. (2015) 开发了一种集中式 RL 方法，用于调度多状态过程和多机器制造系统。Qu et al. (2016) 还设计了一种基于 RL 的调度方法，它可以通过采用实时产品和加工事件信息来自适应更新生产计划。Wang and Yan (2016) 提出了一种基于知识的多智能体自适应调度，并采用了基于加权 Q-learning 的动态调度策略。为了提高性能，Shahrabi et al. (2017) 提出了一种用于参数估计的 Q-factor 算法，以解决具有随机作业到达和机器故障的动态 JSP。Zhang et al. (2017) 从马尔可夫决策过程的角度提出了基于仿真的 Q-learning 用于调度问题。Palombarini and Martínez (2018) 采用了一种深度 RL 方法，将重调度知识保存在深度 Q-network 中，以便直接从高维感官输入中学习调度修复策略。Waschneck et al. (2018) 将深度 Q-network 应用于半导体制造调度，并使用灵活的目标训练深度神经网络。Shiue et al. (2018) 设计了一种基于 RL 的方法，包括离线学习模块和 Q-learning，该方法比以前的分派规则表现更好。Ahmadi et al. (2018) 将作业排序问题建模为二阶旅行商问题，并使用基于动态 Q-learning 的遗传算法求解。Zhao et al. (2019) 采用双层动作 Q-learning 算法求解动态 FJSP，并表明该方法适用于动态 FJSP。Palombarini and Martínez (2019) 还将实时重调度任务应用为闭环控制问题，并训练深度 Q-network 以选择修复动作来应对意外事件和干扰。Lin et al. (2019) 提出了基于边缘计算的智能制造工厂框架，并使用改进的深度 Q-network 求解 JSP。Han and Yang (2020) 设计了一种决斗双深度 Q-network，结合了深度卷积神经网络和 RL 的优势，并根据输入的制造状态直接学习行为策略。Luo (2020) 采用具有连续状态的深度 Q-learning 求解动态 FJSP，它优于其他分派规则。Wang (2020) 提出了一种通过聚类使用加权 Q-learning 算法的 FJSP 自适应策略，以动态找到最合适的操作。Liu et al. (2020) 将 JSP 视为顺序决策问题，并提出使用深度 RL 来处理它。为了加速模型训练，他们还提出了一种结合异步更新和深度确定性策略梯度的并行训练方法。

尽管 RL 技术已广泛应用于制造业，但它在多目标模糊柔性作业车间调度中的应用仍然是一个值得探索的方向。

### 2.4 模糊集 (Fuzzy Set)

模糊集 $\tilde{F}$ 包含两个元素：$x$ 和隶属函数 $\mu_{\tilde{F}}(x)$。$\mu_{\tilde{F}}(x)$ 是 $x$ 属于 $\tilde{F}$ 的可能性。所有 $x$ 属于一个确定集 $X$。模糊集的定义如下：

$$
\widetilde{F} = \left\{ x, \mu_{\tilde{F}}(x) \mid \forall x \in X \right\}, \quad 0 \leqslant \mu_{\tilde{F}}(x) \leqslant 1
$$

经典集合是确定的。当 $\mu_{\tilde{F}}(x) = 1$ 时，模糊集退化为经典集合。

### 2.5 模糊算子 (Fuzzy operators)

调度中最经典的隶属函数是**三角模糊数（TFN, Triangular Fuzzy Number）**。如图 1 所示，隶属函数类似于三角形。$t_1$ 是最早加工时间，$t_2$ 是最可能加工时间，$t_3$ 是最晚加工时间。三元组 $(t_1, t_2, t_3)$ 是一个 TFN，用于表示加工时间。隶属函数的公式如下：

$$
\mu_{\tilde{F}}(x) = \begin{cases}
0, & x \leqslant t_1 \\
\frac{x - t_1}{t_2 - t_1}, & t_1 < x \leqslant t_2 \\
\frac{t_3 - x}{t_3 - t_2}, & t_2 < x < t_3 \\
0, & x \geqslant t_3
\end{cases}
$$

**图 1. 三角隶属函数**

**模糊算子定义：**

1. **加法算子**：$\tilde{s} + \tilde{t} = (s_1 + t_1, s_2 + t_2, s_3 + t_3)$

2. **排序算子**：
   - (a) $f_1(\tilde{x}) = \frac{x_1 + 2x_2 + x_3}{4}$。如果 $f_1(\tilde{s}) > f_1(\tilde{t})$，则 $\tilde{s} > \tilde{t}$；否则 $\tilde{s} < \tilde{t}$。
   - (b) $f_2(\tilde{x}) = x_2$。当 $f_1(\tilde{s}) = f_1(\tilde{t})$ 时，如果 $f_2(\tilde{s}) > f_2(\tilde{t})$，则 $\tilde{s} > \tilde{t}$；否则 $\tilde{s} < \tilde{t}$。
   - (c) $f_3(\tilde{x}) = s_3 - s_1$。当 $f_2(\tilde{s}) = f_2(\tilde{t})$ 时，如果 $f_3(\tilde{s}) > f_3(\tilde{t})$，则 $\tilde{s} > \tilde{t}$；否则 $\tilde{s} < \tilde{t}$。

3. **最大值算子**：如果 $\tilde{s} > \tilde{t}$，则 $\tilde{s} \vee \tilde{t} = \tilde{s}$；否则 $\tilde{s} \vee \tilde{t} = \tilde{t}$。

---

## 3. 问题描述与数学建模 (Problem statement and mathematical modeling)

### 3.1 问题描述

具有模糊加工时间的柔性作业车间调度问题（FFJSP）来自实际制造过程，可以描述如下：

- $\mathcal{J} = \{\mathcal{J}_1, \mathcal{J}_2, \ldots, \mathcal{J}_i, \ldots, \mathcal{J}_n\}$ 是作业集；
- $\mathcal{M} = \{\mathcal{M}_1, \mathcal{M}_2, \ldots, \mathcal{M}_k, \ldots, \mathcal{M}_m\}$ 是机器集。

每个作业 $\mathcal{J}_i$ 有一组 $\Theta_i$ 个工序：

$$
O_i = \{O_{i,1}, O_{i,2}, \ldots, O_{i,j}, \ldots, O_{i,\Theta_i}\}, \quad O_{i,j} \in O_i
$$

每个工序可以在部分机器或全部机器上加工。加工时间是一个 TFN：$\tilde{P}_{i,j,k} = (p_1, p_2, p_3)$。

FFJSP 包含两个子问题：
1. **机器分配（Machine assignment）**：每个工序从候选集中选择一台机器；
2. **工序排序（Operation sequencing）**：在所有机器上调度所有工序以获得可行的调度方案。

**FFJSP 的假设条件：**

- 每个作业不能同时在多台机器上加工；
- 工序不能中断，不考虑机器故障；
- 设置时间和拆卸时间包含在加工时间内；
- 只有当前一工序完成后，下一工序才能开始；
- 每个作业的加工时间是一个 TFN。

### 3.2 符号说明

| 符号 | 说明 |
|------|------|
| $n$ | 作业数量 |
| $m$ | 机器数量 |
| $i$ | 作业索引 |
| $j$ | 工序索引 |
| $k$ | 机器索引 |
| $\Theta_i$ | 作业 $\mathcal{J}_i$ 的总工序数 |
| $O_{i,j}$ | 作业 $\mathcal{J}_i$ 的第 $j$ 个工序，$j = 1, \ldots, \Theta_i$ |
| $\tilde{S}_{i,j}$ | 工序 $O_{i,j}$ 的开始时间，是一个 TFN：$\tilde{S}_{i,j} = (s_1, s_2, s_3)$ |
| $\tilde{C}_{i,j}$ | 工序 $O_{i,j}$ 的完成时间，是一个 TFN：$\tilde{C}_{i,j} = (c_1, c_2, c_3)$ |
| $\tilde{P}_{i,j,k}$ | 作业 $\mathcal{J}_i$ 的第 $j$ 个工序在机器 $\mathcal{M}_k$ 上的加工时间，是一个 TFN：$\tilde{P}_{i,j,k} = (p_1, p_2, p_3)$ |
| $x_{i,j,k}$ | 如果作业 $\mathcal{J}_i$ 的第 $j$ 个工序在机器 $\mathcal{M}_k$ 上加工，则值为 1；否则为 0 |
| $u_{i_1,j_1,i_2,j_2}$ | 如果作业 $\mathcal{J}_{i_2}$ 的第 $j_2$ 个工序在机器 $\mathcal{M}_k$ 上加工，且机器 $\mathcal{M}_k$ 也是作业 $\mathcal{J}_{i_1}$ 的第 $j_1$ 个工序的候选机器，则值为 1；否则为 0 |

### 3.3 MOFFJSP 的数学模型

MOFFJSP 的数学模型如下：

**目标函数：**

1. **最小化模糊最大完工时间（Fuzzy Makespan）**：
   $$
   \min \tilde{C}_{max} = \max_{i=1}^{n} \tilde{C}_{i,\Theta_i}
   $$

2. **最小化总机器负载（Total Machine Workload）**：
   $$
   \min \tilde{W} = \sum_{k=1}^{m} \sum_{i=1}^{n} \sum_{j=1}^{\Theta_i} \tilde{P}_{i,j,k} \cdot x_{i,j,k}
   $$

**约束条件：**

$$
\tilde{C}_{i,j} = \tilde{S}_{i,j} + \sum_{k=1}^{m} \tilde{P}_{i,j,k} \cdot x_{i,j,k}, \quad \forall i, j
$$

$$
\tilde{S}_{i,j+1} \geqslant \tilde{C}_{i,j}, \quad \forall i, j = 1, \ldots, \Theta_i - 1
$$

$$
\sum_{k=1}^{m} x_{i,j,k} = 1, \quad \forall i, j
$$

$$
x_{i,j,k} = \begin{cases}
1, & \text{if } O_{i,j} \text{ is processed on } \mathcal{M}_k \\
0, & \text{otherwise}
\end{cases}
$$

### 3.4 示例说明

（注：原文包含一个示例甘特图，展示了调度方案的可视化表示）

在计算每个工序的 TFN 完成时间后，将获得系统的最大完成时间和总机器负载，从而得到目标函数值。

---

## 4. 所提出的算法 (Proposed algorithm)

### 4.1 RMOEA/D 的框架

本节介绍所提出的基于 RL 的 MOEA/D 算法的框架，流程如下：首先，初始化参数和种群；然后，对每个解执行变邻域搜索以提高开发能力；接下来，应用 Q-learning 为 MOEA/D 选择参数 T；此外，执行 MOEA/D 使用 T 通过离散交叉和变异方法生成新解；最后，根据 PF 的收敛性和多样性变化更新 Q-table。精英档案将存储非支配解。如果不满足停止准则，则继续迭代。

**算法 1：RMOEA/D 算法**

```
输入: 种群大小 N，变异率 R，邻域数向量 T，参数记忆长度 LP，
      学习率 α，折扣因子 γ，贪婪因子 ε，精英档案 E
输出: 迄今找到的最优解

1. 初始化大小为 N 的种群 P（参见第 4.3 节）
2. 初始化算法的所有变量和参数
3. while 未满足停止准则 do
4.   为每个个体执行所提出的 VNS（参见第 4.6 节）
5.   应用 Q-learning 为 MOEA/D 分配参数 T（参见第 4.5 节）
6.   使用参数 Ti 通过离散交叉和变异方法为每个个体执行 MOEA/D 算法（参见第 4.4 节）
7.   更新种群状态和 Q-table（参见第 4.5 节）
8.   计算非支配解集并将其收集到精英档案中（参见第 4.7 节）
9. end while
```

### 4.2 编码与解码

本文使用两个一维向量来表示解（或智能体）。**工序序列**用于指示所有工序的加工顺序，**机器选择**用于表示每个工序分配的机器。两个向量的长度相同，等于工序总数。

**图 3. 编码表示**

编码表示包含两个向量。例如，工序序列为 $O_{3,1}, O_{2,1}, O_{1,1}, O_{2,2}, O_{1,2}, O_{3,2}, O_{1,3}, O_{3,3}, O_{2,3}$。机器选择为 $\mathcal{M}_1, \mathcal{M}_3, \mathcal{M}_2, \mathcal{M}_2, \mathcal{M}_2, \mathcal{M}_3, \mathcal{M}_2, \mathcal{M}_2, \mathcal{M}_1$。它们是一一对应的关系。例如，$O_{3,1}$ 选择了 $\mathcal{M}_1$，$O_{2,1}$ 选择了 $\mathcal{M}_3$。

解码一个解是根据工序序列将适当的加工时间分配给每个工序在其选定的机器上。当解码一个解时，首先将第一个向量转换为工序序列。然后从第二个向量中为每个工序分配选定的机器。最后，将模糊加工时间分配给工序。在本文中，每个解（智能体）被解码为一个模糊调度，即加工时间是一个三角模糊数。

### 4.3 初始化策略

本节介绍一种集成 3 种规则的初始策略。这些规则包括：

1. **Random 规则**（Gao, Suganthan, Pan, & Tasgetiren, 2015）
2. **局部最小加工时间（LS, Local minimum processing time）规则**（Shaheed et al., 2018）
3. **全局最小负载（GW, Global minimum workload）规则**（Li, Liu, et al., 2020）

LS 和 GW 旨在优化目标函数。三种策略的描述如下：

**Random 规则**：
- 该规则简单，确保初始种群具有高多样性。
- (1) 将每个作业 $i$ 重复 $\Theta_i$ 次以生成调度向量；
- (2) 随机重新排列调度向量中所有工序的顺序；
- (3) 为每个工序从其候选集中随机选择一台机器，生成路径向量。

**LS 规则**：
- 该规则旨在减少模糊最大完工时间（最大完成时间）。
- (1) 与 Random 规则相同，随机生成调度向量；
- (2) 对于每个工序，从候选集中选择加工时间最短的机器，生成路径向量。

**GW 规则**：
- 该规则侧重于降低总机器负载。
- (1) 将 $O_{1,1}, O_{2,1}, \ldots, O_{N,1}$ 放入调度向量并重新排列顺序；
- (2) 重新排列其余工序的顺序并连接在前面的序列之后；
- (3) 对于每个工序，选择负载最小的可用机器。如果多台机器具有相同的负载，则选择加工时间最短的机器。

结合三种策略的优势，开发了一种称为 **MIX3** 的方法。

**算法 2：MIX3 规则**

```
输入: 种群大小 Np
输出: 初始化种群

1. 执行 GW 生成后代 P1，大小为 ⌊Np/3⌋
2. 执行 LS 生成后代 P2，大小为 ⌊Np/3⌋
3. 使用 Random 生成后代 P3，大小为 ⌊Np/3⌋
4. 将 P1、P2 和 P3 合并为 Parent，大小为 3 × ⌊Np/3⌋
5. if Parent 的大小 = Np then
6.   终止算法
7. else
8.   用 Random 规则补充剩余的解
```

### 4.4 交叉与变异

为了获得较大的探索步长，采用**优先工序交叉（POX, Precedence Operation Crossover）**和**通用交叉（UX, Universal Crossover）**（Gao, Suganthan, Chua, et al., 2015）。

**POX 用于工序序列**：
1. 随机将作业集分成两个子集 J1 和 J2；
2. 选择两个解 S1 和 S2。对于属于 J1 的每个作业，将其工序复制到 NewS1。对于属于 J2 的每个作业，将其工序复制到 NewS2；
3. 在 NewS1 和 NewS2 中存在许多未填充工序的空位。从 S2 中复制未出现在 NewS1 中的工序，按照 S2 中的顺序从左到右填充 NewS1 的空位。对 NewS2 执行类似操作。

**UX 用于机器选择**：
1. 随机生成一个 0-1 向量，长度等于工序总数；
2. 选择两个解 M1 和 M2。在 0-1 向量相同位置为 1 的地方，交换 M1 和 M2 中的值。

**变异算子**：
- **工序序列变异**：随机选择工序序列中的两个位置并交换值；
- **机器选择变异**：随机选择机器选择中的两个位置，并从其候选集中选择新机器。

### 4.5 基于 Q-learning 的参数自适应策略 (Q-PAS)

#### 4.5.1 Q-learning 简介

Q-learning 由 Watkins and Dayan (1992) 于 1992 年首次提出，已成为最著名的 RL 算法之一。Q-learning 由五元组 $(A, E, C, S, R)$ 组成，分别表示智能体（Agents）、环境（Environment）、动作集（Action set）、状态集（State set）和奖励（Reward）。

如图 5 所示，智能体根据其在环境中时刻 $t$ 的状态 $S_t$ 执行动作 $A_t$。然后智能体将获得奖励 $R_{t+1}$，其状态将转变为新状态 $S_{t+1}$。Q 值根据以下公式更新：

$$
Q(S_t, A_t) = Q(S_t, A_t) + \alpha \left[ R_t + \gamma \max(Q(S_{t+1}, A_t)) - Q(S_t, A_t) \right]
$$

**图 5. RL 的智能体-环境交互**

其中：
- $\alpha$ 是学习率，介于 0 和 1 之间；
- $\gamma$ 是折扣因子，也在 [0, 1] 范围内。当 $\gamma$ 接近 1 时，Q 值更受未来状态影响；相反，$\gamma$ 越接近 0，Q 值越关注当前状态；
- $R_t$ 是执行动作 $A_t$ 后的奖励。

这是一个马尔可夫决策过程。当前状态 $S_{t+1}$ 可以影响后续状态 $S_{t+i}$。

#### 4.5.2 智能体与动作定义

在 MOEA 中，Pareto 前沿（PF）是 MOEA 计算的最佳解集，反映了算法的能力。通过 Q-learning，将引导种群选择最佳参数 T 以提高 PF 的多样性。增加整个种群的多样性可以提高 PF 的多样性。因此，将每代中的 PF 抽象为一个智能体，以反映成功的参数选择。

四个候选值 $T = \{5, 10, 15, 20\}$ 被定义为动作。

#### 4.5.3 状态定义

PF 的收敛性和多样性通过以下两个指标衡量：

**收敛性指标 CV（Convergence）**：

$$
CV(P, P^*) = \frac{\sqrt{\sum_{y \in P} \min_{x \in P^*} dis(x, y)^2}}{|P|}
$$

其中 $P^*$ 是参考点，因为 FFJSP 实例的真实 PF 无法计算。CV 越小，收敛性越好，且 $CV > 0$。

**多样性指标 DV（Diversity）**：

$$
DV = \frac{\sum_{i=1}^{N-1} |d_i - \bar{d}|}{(N-1)\bar{d}}
$$

其中 $d_i$ 是 PF 中两个相邻点的欧氏距离，$\bar{d}$ 是 $d_i$ 的平均值。DV 越大，多样性越好，且 $DV > 0$。

在进化过程中，PF 的 CV 和 DV 有四种组合条件：
1. $\Delta CV > 0$ 且 $\Delta DV > 0$
2. $\Delta CV > 0$ 且 $\Delta DV \leqslant 0$
3. $\Delta CV \leqslant 0$ 且 $\Delta DV > 0$
4. $\Delta CV \leqslant 0$ 且 $\Delta DV \leqslant 0$

(1) 和 (2) 发生在进化期间，(3) 和 (4) 发生在种群已收敛时。这四种条件被视为智能体的四个状态。

#### 4.5.4 奖励定义

执行动作后，智能体将获得奖励，可能是正数或负数。奖励定义如下：

$$
Reward = \begin{cases}
10, & \Delta DV > 0 \\
0, & \Delta DV \leqslant 0
\end{cases}
$$

如果 PF 的多样性更好，所选动作（参数）将获得奖励并更新 Q-table。相反，奖励为零。值得注意的是，奖励值最常用的是 10，这是根据以往研究得出的。

#### 4.5.5 基于 Q-learning 的参数自适应

基于上述说明，设计了一种基于 Q-learning 的参数自适应策略，即 **Q-PAS**。

**算法 3：基于 Q-learning 的参数自适应策略 (Q-PAS)**

```
输入: 种群 P，贪婪因子 ε，学习率 α，折扣因子 γ
输出: Q_table

1. Q_table(4,4) ← 0; CV ← 1; DV ← 1;
   CV_{i-1} = CV_i = DV_{i-1} = DV_i = 0
2. while 满足停止准则 do
3.   CV_{i-1} = CV_i; DV_{i-1} = DV_i
4.   计算智能体状态 St（参见第 4.5.3 节）
5.   if rand < ε then
6.     选择最大 Q(St, Ai) 的动作 At，i = 1, 2, 3, 4
7.   else
8.     随机选择动作 At
9.   执行动作 At 用于 MOEA/D 更新 P 并获得 PF
10.  通过公式 (16) 和 (17) 计算 CV_i 和 DV_i
11.  ΔCV = CV_{i-1} - CV_i, ΔDV = DV_i - DV_{i-1}
12.  通过公式 (20) 获取动作 At 的奖励 R(St, At)
13.  计算新解 x_new 的状态 S_{t+1}
14.  Q(St, At) = Q(St, At) + α[R(St, At) + γ max(Q(St+1, At)) - Q(St, At)]
```

### 4.6 基于强化学习的变邻域搜索 (RVNS)

**算法 4：基于强化学习的变邻域搜索 (RVNS)**

```
输入: 解 P(i)，选择概率 P = {P1, P2, ..., P5}，
      参考点 Z*，权重向量 λi
输出: 新解 P(i)

1. 根据 P 执行轮盘赌算法为 P(i) 分配局部搜索 LSi
2. 采用 LSi 获得新解 P(i)'
3. if g^{te}(P(i)'|λ, Z*) < g^{te}(P(i)|λ, Z*) then
4.   P(i) = P(i)', ns_i = ns_i + 1
5. else
6.   nf_i = nf_i + 1
7. 统计 ns_i 和 nf_i，i = 1,...,n。将此记录保存到成功记忆和失败记忆的末尾
8. if SM 的长度 > LP then
9.   删除 SM 和 FM 中的第一条记录
10. for i = 1 to n do
11.   SR(i) = Σ_{j=1}^{LP} ns_{i,j}; FR(i) = Σ_{j=1}^{LP} nf_{i,j}
12.   P(i) = SR(i) / (SR(i) + FR(i))
13. for i = 1 to n do
14.   P(i) = P(i) / Σ_{i=1}^{n} P(i)
```

**五种局部搜索策略：**

- **LS1**：随机选择两个工序并交换它们的位置
- **LS2**：随机选择一个工序 $O_{i,j}$ 并将其移动到另一台具有最小加工时间的机器
- **LS3**：找到最大负载的机器 $\mathcal{M}$。随机选择一个由 $\mathcal{M}$ 加工的工序并将其移动到另一台机器 $\mathcal{M}'$
- **LS4**：随机选择工序序列上的两个位置并交换值
- **LS5**：随机选择工序序列上的两个位置并插入值

RVNS 的工作流程：
1. 执行轮盘赌算法为每个解 P(i) 分配局部搜索 LSi；
2. 采用局部搜索 LSi 生成新解 P' 并判断旧解是否可以更新；
3. 统计成功更新旧解的次数 np 和失败次数 nf。保存记录并插入成功记忆 SM 和失败记忆 FM 的尾部；
4. 如果 SM 和 FM 的长度超过 LP，删除 SM 和 FM 中的第一条记录。然后通过求 SM 和 FM 中每列的和来更新每个参数的概率；
5. 归一化每个参数的概率 P(i)，确保概率之和等于 1。

**图 6. 成功记忆和失败记忆**

### 4.7 精英档案更新

**算法 5：更新精英档案**

```
输入: 档案 A，种群大小 Np，本代 PF
输出: 档案 A

1. A ← A ∪ PF
2. if A 的长度 > Np then
3.   A ← 非支配排序(A)
```

### 4.8 计算复杂度分析

1. MIX3 的复杂度为 $\mathcal{O}(Np)$，其中 Np 是种群大小；
2. MOEA/D 的复杂度为 $\mathcal{O}(Iter \times Np \times 2 \times T_i)$，其中 Iter 是迭代次数，$T_i$ 是邻域数；
3. Q-PAS 的复杂度为 $\mathcal{O}(Iter)$，其中 LP 是成功记忆和失败记忆的长度，T 是候选参数向量；
4. RVNS 的复杂度为 $\mathcal{O}(Iter \times 5 \times LP)$；
5. 精英档案的复杂度为 $\mathcal{O}(Iter \times Np^2)$。

因此，RMOEA/D 的总复杂度为 $\mathcal{O}(Iter \times Np^2)$。

---

## 5. 实验结果 (Experimental results)

第 4 节详细描述了 RMOEA/D 算法。本节设计详细的实验来评估 RMOEA/D 的性能。RMOEA/D 和对比算法在 Intel Core i7 6700 CPU @ 3.4 GHz 和 8G RAM 的计算机上使用 MATLAB 编码。为公平起见，每个算法在每个实例上独立运行 30 次。值得注意的是，为了验证所提算法的收敛性和多样性，在 30 次独立运行后，收集平均结果进行性能比较。

对比算法是知名的多目标优化算法：MOEA/D、NSGA-II、MOEA/D-M2M、NSGA-III 和 IAIS。

### 5.1 实验实例

选择三个基准测试集来验证所提出的 RMOEA/D 的收敛性和多样性：

1. 第一个基准 Lei01 和 Lei02 来自 (Lei, 2010, 2012)；
2. 第二个基准 Remanu 由 (Gao, Suganthan, Pan, & Tasgetiren, 2015) 提供；
3. 此外，将柔性作业车间调度问题基准 Mk (Brandimarte, 1993) 进行转换。对于每个加工时间 b，在区间 [0, b/2] 中随机生成两个整数 a 和 c。(a, b, c) 是一个 TFN。因此，Mk 基准被转换为 FFJSP 基准 FMk。

**表 2. 实例规模说明**

| 名称 | N (作业数) | M (机器数) | SH (总工序数) | C-FFJSP | 文献来源 |
|------|-----------|-----------|-------------|---------|---------|
| D1 | 10 | 10 | 40 |  |  |
| D2 | 10 | 10 | 40 |  |  |
| D3 | 10 | 10 | 50 |  |  |
| D4 | 10 | 10 | 50 | Y | (Lei, 2010) |
| D5 | 15 | 10 | 80 | Y | Lei (2012) |
| R1 | 5 | 4 | 23 |  |  |
| R2 | 8 | 8 | 64 |  |  |
| R3 | 10 | 6 | 81 |  |  |
| R4 | 10 | 10 | 100 |  |  |
| R5 | 15 | 8 | 171 |  |  |
| R6 | 15 | 10 | 185 |  |  |
| R7 | 20 | 10 | 308 |  | (Gao, Suganthan, Pan, & Tasgetiren, 2015) |
| R8 | 20 | 15 | 355 |  |  |
| FMk1 | 10 | 6 | 55 |  |  |
| FMk2 | 10 | 6 | 58 |  |  |
| FMk3 | 15 | 8 | 150 |  |  |
| FMk4 | 15 | 8 | 90 |  |  |
| FMk5 | 15 | 4 | 106 |  |  |
| FMk6 | 10 | 15 | 150 | N |  |
| FMk7 | 20 | 5 | 100 |  |  |
| FMk8 | 20 | 10 | 225 |  |  |
| FMk9 | 20 | 10 | 240 |  | (Brandimarte, 1993) |
| FMk10 | 20 | 15 | 240 |  |  |

### 5.2 实验参数

采用正交表 $L_{16}(4^4)$ 进行标定实验。为公平起见，每个参数独立运行 30 次。根据之前的工作，种群大小 Np = 100 是最佳的。其他参数包括最大代数 G = 200 和第 4.5 节中的邻域数向量 T，它有 4 个候选参数 $T = \{5, 10, 15, 20\}$。

**图 7. HV 指标的主效应图**

图 7 显示了四个参数对 HV 指标的主效应图。指标值越高，性能越好。基于综合观察，最佳参数配置设置为：**LP = 40，$\gamma$ = 0.6，$\alpha$ = 0.4，$\epsilon$ = 0.8**。

### 5.3 Q-PAS 的讨论

本节通过图表和 Q-table 分析来证明 Q-learning 的有效性。图 8 显示了单次运行的 CV 和 DV 趋势图。该示例是从实例 D1 中提取的解。如图 8 所示，每个波谷到波峰可以被视为一个周期。当种群开始进化时，由于局部搜索、交叉和变异策略，它收敛很快，DV 迅速降低。经过一段时间的探索后，解陷入局部最优，Q-PAS 引导种群选择最佳参数以增加多样性。因此 DV 快速增加。当下一个周期开始时，DV 的规律很明显，先降低后增加。这可以证明 Q-PAS 在收敛难以提高时对增加多样性有很大影响。

**表 3. 200 次迭代后的 Q-table**

| 状态 | T=5 | T=10 | T=15 | T=20 |
|------|-----|------|------|------|
| State1 (ΔCV>0, ΔDV>0) | 0 | 1.798693 | 0 | 6.743954 |
| State2 (ΔCV>0, ΔDV≤0) | 0 | 0.323268 | 0 | 5.271637 |
| State3 (ΔCV≤0, ΔDV>0) | 4.539988 | 0.766909 | 0 | 0 |
| State4 (ΔCV≤0, ΔDV≤0) | 2.847096 | 2.28377 | 2.911486 | 1.152641 |

表 3 给出了 200 次迭代后的 Q-table。在迭代早期，CV 降低很快，T 需要设置得更大（如 20）以提高 DV。当解陷入局部最优且 CV 难以提高时，T 需要调整得更小（如 5）以增加 DV。如果 CV 和 DV 都很少增加，则 T = 5 或 10 可能有助于 DV 增加。综上所述，Q-PAS 是一种有效的策略，让种群在智能体的每个状态下动态选择最佳参数。

### 5.4 RMOEA/D 各改进部分的有效性

在证明 Q-learning 的有效性后，本节验证其他改进部分的效率。通过逐步将改进部分添加到 MOEA/D 中，设置了 RMOEA/D 的几个变体：

- **RMOEA/D1**：纯 MOEA/D，没有任何改进；
- **RMOEA/D2**：MOEA/D + 初始策略；
- **RMOEA/D3**：RMOEA/D2 + 随机选择 VNS；
- **RMOEA/D4**：RMOEA/D3 + Q-PAS；
- **RMOEA/D5**：RMOEA/D4 + 精英档案；
- **RMOEA/D**：将 RMOEA/D5 中的 VNS 改为 RVNS。

为公平起见，所有算法独立运行 30 次。

**表 4. 变体算法的平均排名 (Friedman)，p-value = 0.00015**

| 算法 | 排名 |
|------|------|
| RMOEA/D1 | 4.6087 |
| RMOEA/D2 | 4.3913 |
| RMOEA/D3 | 3.6087 |
| RMOEA/D4 | 3.087 |
| RMOEA/D5 | 2.913 |
| RMOEA/D | 2.3913 |

**表 5. RMOEA/D 变体的 HV 结果比较**

| 实例 | RMOEA/D1 | RMOEA/D2 | RMOEA/D3 | RMOEA/D4 | RMOEA/D5 | RMOEA/D |
|------|----------|----------|----------|----------|----------|---------|
| D1 | 0.096358 | 0.097446 | 0.099396 | 0.098672 | 0.099119 | **0.099671** |
| D2 | 0.101298 | 0.101899 | 0.103487 | 0.103056 | 0.103296 | **0.103148** |
| D3 | 0.064942 | 0.067186 | 0.066809 | 0.067699 | **0.067898** | 0.06737 |
| D4 | 0.057389 | 0.058733 | 0.058901 | **0.059791** | 0.058731 | 0.05847 |
| D5 | 0.048071 | 0.050738 | 0.052962 | **0.053041** | 0.052376 | 0.052245 |
| R1 | 0.050917 | 0.050711 | 0.050713 | 0.050458 | 0.050879 | **0.050919** |
| R2 | 0.031251 | 0.032267 | **0.033204** | 0.031718 | 0.031889 | 0.031856 |
| R3 | 0.034702 | 0.035149 | 0.035013 | 0.03601 | **0.036115** | 0.035549 |
| R4 | 0.039239 | 0.039681 | 0.041548 | 0.041979 | 0.041942 | **0.042023** |
| R5 | 0.042622 | 0.044959 | 0.045299 | 0.045912 | **0.046284** | 0.046105 |
| R6 | 0.045522 | 0.050659 | 0.051547 | **0.052755** | 0.051897 | 0.051946 |
| R7 | 0.041646 | 0.054828 | 0.056424 | 0.057074 | 0.057371 | **0.057465** |
| R8 | 0.043417 | 0.069881 | 0.073654 | 0.073011 | 0.073377 | **0.074284** |
| FMk01 | 0.058504 | 0.055246 | 0.056479 | 0.057164 | 0.056871 | **0.057207** |
| FMk02 | 0.042559 | 0.040818 | 0.042238 | **0.043146** | 0.042341 | 0.042327 |
| FMk03 | 0.06336 | 0.063915 | 0.064411 | 0.063918 | **0.064357** | 0.064147 |
| FMk04 | 0.094665 | 0.095224 | 0.094696 | 0.095569 | 0.095808 | **0.09624** |
| FMk05 | 0.048044 | 0.048418 | 0.048343 | 0.048325 | 0.048403 | **0.048468** |
| FMk06 | 0.045506 | 0.043584 | 0.044403 | 0.045113 | 0.044132 | **0.044598** |
| FMk07 | 0.071437 | 0.070497 | 0.070504 | 0.070188 | 0.070395 | **0.070645** |
| FMk08 | 0.022032 | 0.021521 | 0.021134 | 0.021609 | 0.021323 | **0.021721** |
| FMk09 | 0.042823 | 0.043149 | 0.042166 | 0.04264 | 0.042568 | **0.042508** |

表 5 显示了比较的 HV 指标结果。粗体和灰色值表示每个实例的最佳结果。表 4 显示了 Friedman 排名，每个部分都比上一个有所改进。这证明了每个部分的有效性。

### 5.5 比较与讨论

**表 6. 参数设置**

| 算法 | 特殊参数 | 通用参数 |
|------|---------|---------|
| RMOEA/D | 记忆长度 LP=40，学习率 α=0.4，折扣因子 γ=0.6，贪婪因子 ε=0.8，邻域向量 T=[5,10,15,20] | 种群大小 NP=100，权重向量数 λ=100，变异率 R=0.8 |
| MOEA/D | 邻域数 T=10 |  |
| MOEA/D-M2M | 邻域数 T=10，子种群大小 S=10 |  |
| NSGA-III | 邻域数 T=10 |  |
| NSGA-II | - | R=0.8, Np=100 |
| IAIS | 克隆种群大小 nc=10，种群大小 NP=nc*(nc+1)/2，拥挤度阈值 CRmax=1×10^{-4}，温度率 w=0.5，初始温度 T0=1×10^4 |  |

**表 7. 与其他算法的 HV 结果比较**

| 实例 | IAIS | MOEA/D | MOEA/D-M2M | NSGA-II | NSGA-III | **RMOEA/D** |
|------|--------|--------|-----------|---------|---------|------------|
| D1 | 0.101648 | 0.121501 | 0.102406 | 0.112091 | 0.10712 | **0.125066** |
| D2 | 0.126441 | 0.145188 | 0.133167 | 0.135597 | 0.130158 | **0.147033** |
| D3 | 0.107201 | 0.113763 | 0.096045 | 0.10405 | 0.099619 | **0.117255** |
| D4 | 0.106326 | 0.111984 | 0.090837 | 0.103181 | 0.096756 | **0.113831** |
| D5 | 0.14925 | 0.153384 | 0.11185 | 0.140018 | 0.129151 | **0.158382** |
| R1 | 0.115783 | 0.123475 | 0.113377 | 0.121941 | 0.122881 | **0.123476** |
| R2 | 0.067973 | 0.073794 | 0.061281 | 0.071102 | 0.068841 | **0.074325** |
| R3 | 0.089022 | 0.093673 | 0.092318 | 0.091856 | 0.089554 | **0.09424** |
| R4 | 0.096069 | 0.103368 | 0.082595 | 0.098912 | 0.093185 | **0.105898** |
| R5 | 0.102766 | 0.107348 | 0.080571 | 0.098157 | 0.085406 | **0.111737** |
| R6 | 0.135977 | 0.135997 | 0.07776 | 0.123481 | 0.105439 | **0.144087** |
| R7 | 0.17597 | 0.170876 | 0.064412 | 0.150029 | 0.11291 | **0.19205** |
| R8 | 0.193543 | 0.165819 | 0.068542 | 0.125642 | 0.083388 | **0.206107** |
| FMk01 | 0.060938 | 0.075054 | 0.067977 | 0.070224 | 0.06702 | **0.073803** |
| FMk02 | 0.058621 | 0.070369 | 0.062152 | 0.066106 | 0.064245 | **0.070362** |
| FMk03 | 0.056997 | 0.078065 | 0.067212 | 0.07017 | 0.068431 | **0.07847** |
| FMk04 | 0.068268 | 0.097126 | 0.082704 | 0.08333 | 0.077104 | **0.098794** |
| FMk05 | 0.054107 | 0.06207 | 0.056213 | 0.056679 | 0.055799 | **0.062563** |
| FMk06 | 0.096688 | 0.13492 | 0.098165 | 0.131115 | 0.116562 | **0.132173** |
| FMk07 | 0.084151 | 0.099128 | 0.09029 | 0.091868 | 0.089918 | **0.099004** |
| FMk08 | 0.038212 | 0.043297 | 0.036438 | 0.038692 | 0.038112 | **0.04306** |
| FMk09 | 0.082438 | 0.092606 | 0.071231 | 0.089952 | 0.083816 | **0.09258** |

**表 8. 对比算法的平均排名 (Friedman)，p-value = 1.31285e-17**

| 算法 | 排名 |
|------|------|
| MOEA/D-M2M | 5.2609 |
| NSGA-III | 4.6522 |
| IAIS | 4.6087 |
| NSGA-II | 3.3913 |
| MOEA/D | 1.8261 |
| **RMOEA/D** | **1.2609** |

**表 9. RMOEA/D 的 Wilcoxon 检验结果**

| 对比 | R+ | R- | Exact p-value |
|------|----|----|--------------|
| RMOEA/D vs MOEA/D-M2M | - | - | < 0.05 |
| RMOEA/D vs NSGA-III | - | - | < 0.05 |
| RMOEA/D vs IAIS | - | - | < 0.05 |
| RMOEA/D vs NSGA-II | - | - | < 0.05 |
| RMOEA/D vs MOEA/D | - | - | < 0.05 |

**图 10. 收敛性指标比较结果**

**图 11. 最佳最大完工时间解的甘特图**

**图 12. 最佳总负载解的甘特图**

### 5.5.1 结果分析

所提出的 RMOEA/D 优于其他对比算法。原因有以下四点：

1. **初始策略**：第 4.3 节提到的初始策略生成了具有高收敛性和多样性的初始种群，使算法收敛更快；
2. **参数自适应策略**：第 4.5 节提出的基于 Q-learning 的参数自适应策略引导算法选择最合适的参数 T，以提高 PF 的多样性；
3. **RVNS**：RMOEA/D 采用第 4.6 节描述的基于 RL 的 VNS，提高了算法的收敛性，因此结果具有更好的收敛性；
4. **精英档案**：精英档案收集历史迭代中被遗弃的高质量解，补充了最终的 PF，使 RMOEA/D 获得更好的结果。

---

## 6. 结论 (Conclusion)

本文提出了一种结合两种 RL 技术的 RMOEA/D，用于求解多目标模糊柔性作业车间调度问题。目标是最小化模糊最大完工时间和总负载。作为一种经典的自适应技术，RL 可以引导算法自动选择最佳参数或局部搜索策略。设计了一种新颖的基于 Q-learning 的参数自适应方法，包括状态定义、动作定义和奖励定义，以帮助多目标优化算法 MOEA/D 自动选择参数。此外，采用基于两种历史记忆的 RL 方法，使算法选择成功概率最高的局部搜索策略。接下来，为了提高历史解的利用率，使用精英档案收集迭代过程中的精英解。此外，应用采用三种初始规则的初始策略来获得具有高收敛性和多样性的初始种群。

实验结果表明，RMOEA/D 在 23 个 FFJSP 实例上优于其他对比算法。可以得出结论，RMOEA/D 适用于求解高度不确定性的多目标模糊柔性作业车间调度问题。RL 技术是实现参数和策略自适应的重要方法。

### 未来研究方向

1. 将目标增加到三个以上，并将 RMOEA/D 应用于求解超多目标问题；
2. 其他类型的问题，如具有模糊加工时间的分布式流水车间问题，也值得研究；
3. 应用更复杂的技术，如深度 Q-network，是另一个研究方向。

---

## 参考文献 (References)

- Ahmadi, E., Goldengorin, B., Süer, G. A., & Mosadegh, H. (2018). A hybrid method of 2-TSP and novel learning-based GA for job sequencing and tool switching problem. *Applied Soft Computing*, 65, 214–229.
- Brandimarte, P. (1993). Routing and scheduling in a flexible job shop by tabu search. *Annals of Operations Research*, 41(3), 157–183.
- Caldeira, R. H., & Gnanavelbabu, A. (2021). A Pareto based discrete jaya algorithm for multi-objective flexible job shop scheduling problem. *Expert Systems with Applications*, 170, Article 114567.
- Deb, K., & Jain, H. (2014). An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, Part I: Solving problems with box constraints. *IEEE Transactions on Evolutionary Computation*, 18(4), 577–601.
- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
- Dorfeshan, Y., Tavakkoli-Moghaddam, R., Mousavi, S. M., & Vahedi-Nouri, B. (2020). A new weighted distance-based approximation methodology for flow shop scheduling group decisions under the interval-valued fuzzy processing time. *Applied Soft Computing*, 91, Article 106248.
- Du, Y., Xing, L., Zhang, J., Chen, Y., & He, Y. (2019). MOEA based memetic algorithms for satellite range scheduling problem. *Applied Soft Computing*, 81, Article 105490.
- Gao, K. Z., Suganthan, P. N., Chua, T. J., Chong, C. S., Cai, T. X., & Pan, Q. K. (2015). A two-stage artificial bee colony algorithm scheduling flexible job-shop scheduling problem with new job insertion. *Expert Systems with Applications*, 42(21), 7652–7663.
- Gao, K. Z., Suganthan, P. N., Pan, Q. K., Chua, T. J., Chong, C. S., & Cai, T. X. (2016). An improved artificial bee colony algorithm for flexible job-shop scheduling problem with fuzzy processing time. *Expert Systems with Applications*, 65, 52–67.
- Gao, K. Z., Suganthan, P. N., Pan, Q. K., & Tasgetiren, M. F. (2015). An effective discrete harmony search algorithm for flexible job shop scheduling problem with fuzzy processing time. *Expert Systems with Applications*, 42(21), 7652–7663.
- Gong, W., Cai, Z., & Liang, D. (2021). Adaptive ranking mutation operator with self-adaption for multi-objective optimization. *Information Sciences*, 567, 174–194.
- Han, B. A., & Yang, J. J. (2020). Research on adaptive job shop scheduling problems based on dueling double DQN. *IEEE Access*, 8, 186474–186495.
- Lang, S., Reggelin, T., Schmidt, J., Müller, M., & Nahhas, A. (2021). NeuroEvolution of scheduling heuristics for job shop scheduling problems. *Procedia CIRP*, 99, 64–69.
- Lei, D. (2010). Fuzzy job shop scheduling problem with availability constraints. *Computers & Industrial Engineering*, 58(4), 610–617.
- Lei, D. (2012). Co-evolutionary genetic algorithm for fuzzy flexible job shop scheduling. *Applied Soft Computing*, 12(9), 2237–2245.
- Lei, D., Guo, X., & Wang, L. (2018). A shuffled multi-swarm micro-migrating birds optimizer for multi-resource constrained flexible job shop scheduling problem. *Swarm and Evolutionary Computation*, 44, 1008–1019.
- Lei, D., Zheng, Y., & Guo, X. (2019). A shuffled multi-swarm micro-migrating birds optimizer for multi-resource constrained flexible job shop scheduling problem. *Swarm and Evolutionary Computation*, 44, 1008–1019.
- Li, R., Huang, H., & Gong, W. (2020). An improved artificial bee colony algorithm for flexible job shop scheduling problems with fuzzy processing time. *IEEE Access*, 8, 186474–186495.
- Li, R., Liu, J., & Gong, W. (2020). Type-2 fuzzy processing time in flexible job shop scheduling problem. *IEEE Access*, 8, 186474–186495.
- Lin, L., Gen, M., & Wang, X. (2019). Integrated multistage logistics network design by using hybrid evolutionary algorithm. *Computers & Industrial Engineering*, 131, 80–95.
- Liu, C., Gao, L., & Pan, Q. (2020). A hybrid multi-objective evolutionary algorithm based on adaptive local search for dynamic job shop scheduling problem. *Swarm and Evolutionary Computation*, 57, Article 100717.
- Luo, S. (2020). Dynamic scheduling for flexible job shop with new job insertions by deep reinforcement learning. *Applied Soft Computing*, 91, Article 106208.
- Lu, C., Gao, L., & Pan, Q. (2021). Energy-efficient scheduling for multi-objective flexible job shop scheduling problem with fuzzy processing time. *Complex & Intelligent Systems*, 7(1), 1–15.
- Palombarini, J. A., & Martínez, E. C. (2018). A deep reinforcement learning approach for the scheduling of 3D printed parts in a flexible job shop. *Procedia CIRP*, 72, 1024–1029.
- Palombarini, J. A., & Martínez, E. C. (2019). A deep reinforcement learning approach for the scheduling of 3D printed parts in a flexible job shop. *Procedia CIRP*, 72, 1024–1029.
- Pan, Q., Gao, L., & Li, X. (2021). A bi-population evolutionary algorithm with feedback scheme for fuzzy flexible job shop scheduling. *Swarm and Evolutionary Computation*, 66, Article 100934.
- Pavlov, A., Ivanov, D., Werner, F., & Dolgui, A. (2019). Integrated detection of disruption scenarios, the ripple effect dispersal and recovery paths in supply chains. *Annals of Operations Research*, 283(1), 119–142.
- Pericleous, S., Lagodimos, A. G., & Leopoulos, V. (2017). A multi-objective genetic algorithm for the permutation flow shop scheduling problem. *International Journal of Production Research*, 55(12), 3442–3460.
- Qu, S., Wang, J., & Govil, S. (2015). A reinforcement learning approach for scheduling multi-state processes and multiple machines manufacturing systems. *Journal of Manufacturing Systems*, 36, 215–224.
- Qu, S., Wang, J., & Govil, S. (2016). A reinforcement learning approach for scheduling multi-state processes and multiple machines manufacturing systems. *Journal of Manufacturing Systems*, 36, 215–224.
- Rifai, A. P., Nguyen, H. T., & Dawal, S. Z. M. (2021). Multi-objective fuzzy job shop scheduling using a hybrid differential evolution algorithm. *International Journal of Advanced Manufacturing Technology*, 103(1-4), 903–916.
- Shahrabi, J., Adibi, M. A., & Mahootchi, M. (2017). A reinforcement learning approach to parameter estimation in dynamic job shop scheduling. *Computers & Industrial Engineering*, 110, 304–315.
- Shao, W., Pi, D., & Shao, Z. (2021a). A hybrid multi-objective evolutionary algorithm based on adaptive local search for distributed heterogeneous hybrid flow