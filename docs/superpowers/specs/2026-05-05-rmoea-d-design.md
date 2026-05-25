# RMOEA/D 算法复现设计文档

## 项目目标

复现论文《A reinforcement learning based RMOEA/D for bi-objective fuzzy flexible job shop scheduling》的核心算法框架（MOEA/D + Q-learning 参数自适应），针对 Brandimarte 模糊柔性作业车间实例（Mk01~Mk10）运行并输出 Pareto 前沿与 HV 指标。

## 技术栈

- Python 3.x
- numpy（数组运算、随机数）
- matplotlib（Pareto 前沿可视化，可选）
- 标准库：urllib, json, math

## 文件结构

```
RMOEA_D/
├── fuzzy.py          # 三角模糊数运算
├── instance.py       # Brandimarte 数据下载与解析
├── encoding.py       # 编码/解码与调度仿真
├── operators.py      # 初始化、交叉、变异、修复
├── moead.py          # MOEA/D 框架：权重、邻域、Tchebycheff
├── qlearning.py      # Q-PAS 自适应：CV/DV、状态、奖励
├── metrics.py        # 非支配排序、HV 计算
├── main.py           # 运行入口与 results.json 输出
└── results.json      # 最终输出（运行后生成）
```

## 模块职责

### fuzzy.py

`FuzzyNumber` 三元组 `(t1, t2, t3)`：
- 加法：分量分别相加
- 取大：按排序规则返回较大者
- 排序：三层规则 `f1 → f2 → f3`
- 清晰化：`(t1 + 2*t2 + t3) / 4`

### instance.py

- 从网络下载 Brandimarte 标准数据（Mk01~Mk10）
- 解析为 `Instance` 对象：工件数、机器数、每道工序的候选机器列表及确定时间 `b`
- 按论文规则生成模糊时间：对每个 `b`，`a ~ randint(0, b/2)`，`c ~ randint(0, b/2)`，模糊时间为 `(a, b, c)`，固定种子 `seed=42`

### encoding.py

`decode(os, ma, instance)`：
1. OS 转工序序列（工件号第几次出现即第几道工序）
2. 依次安排工序，维护每台机器的模糊时钟
3. 开始时间 = max(工件前一工序完成时间, 机器空闲时间)（模糊取大）
4. 完成时间 = 开始时间 + 加工时间（模糊加法）
5. 返回模糊 makespan 和总负载

### operators.py

- **MIX3 初始化**：
  - 1/3 Random：随机 OS，随机 MA
  - 1/3 LS：随机 OS，MA 选 `t2` 最小的机器
  - 1/3 GW：先排第一道工序，再排其余；MA 选增加负载最少的机器
- **POX 交叉**（OS）：按工件分组保留位置
- **UX 交叉**（MA）：均匀交换
- **变异**：OS 交换两位，MA 随机改机器
- **修复**：确保 OS 中各工件出现次数正确

### moead.py

- **权重向量**：Das & Dennis 方法生成 Np=100 个 2D 向量
- **邻域**：按权重欧氏距离选最近的 T 个
- **Tchebycheff**：`max(λ₁·|f₁-z₁|, λ₂·|f₂-z₂|)`，使用清晰化值
- **参考点 z***：每代取各目标清晰值的最小值
- **单代迭代**：对每个子问题选父代→交叉变异→修复→解码→更新邻域

### qlearning.py

- **状态**：4 种（ΔCV>0/≤0 × ΔDV>0/≤0）
- **动作**：T ∈ {5, 10, 15, 20}
- **奖励**：ΔDV > 0 时 10，否则 0
- **ε-greedy**：ε=0.8，按论文描述实现
- **更新**：`Q(S,A) ← Q(S,A) + α[R + γ·max Q(S',a) - Q(S,A)]`

### metrics.py

- **非支配排序**：合并存档后排序，保留前 Np 个
- **HV 计算**：前沿归一化到 [0,1]，参考点 (1,1)，使用 numpy 计算

### main.py

- 对 Mk01~Mk10 逐个运行
- 参数：Np=100, Gen=200, T=[5,10,15,20], α=0.4, γ=0.6, ε=0.8
- 交叉率 0.8，变异始终执行
- 输出 results.json：`{实例名: {hv: float, pf: [[makespan, workload], ...]}}`

## 关键约束

1. 模糊运算仅在调度仿真中使用，Tchebycheff/HV/CV/DV 均使用清晰化值
2. 每代开始前由 Q-learning 统一选择 T，然后重算所有邻域
3. 交叉率 0.8 控制是否执行交叉，变异始终执行
4. 所有实例共享固定随机种子 42（用于生成模糊时间）
5. 每次独立运行使用不同的随机种子（算法本身）

## 输出格式

```json
{
  "Mk01": {
    "hv": 0.073,
    "pf": [
      [2.54, 15.3],
      [3.12, 13.8]
    ]
  },
  "Mk02": { ... }
}
```
