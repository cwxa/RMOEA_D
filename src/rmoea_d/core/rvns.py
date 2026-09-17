"""
Variable Neighborhood Search based on Reinforcement Learning (RVNS).
基于强化学习的变邻域搜索模块。

论文 Section 4.6 描述的 RVNS：
- 5种局部搜索策略 (LS1~LS5)
- 成功/失败记忆 (SM/FM) 更新选择概率
- 轮盘赌选择局部搜索算子
"""

import numpy as np
import logging

from .encoding import decode, decode_crisp
from .operators import _get_op_index, _repair_ma_for_os

logger = logging.getLogger(__name__)


def _build_schedule_info(os_vec, ma_vec, instance):
    """
    解码并构建完整的调度信息，用于LS3（最大负载机器局部搜索）。
    使用预计算的 crisp_times 表，O(1) 查找加工时间。
    """
    n_jobs = instance["n_jobs"]
    n_machines = instance["n_machines"]
    jobs = instance["jobs"]
    crisp_times = instance["crisp_times"]

    op_counter = [0] * n_jobs
    job_ready = [0.0] * n_jobs
    machine_ready = [0.0] * n_machines
    machine_workload = [0.0] * n_machines

    schedule = []
    op_idx = 0

    for pos, job_id in enumerate(os_vec):
        oi = op_counter[job_id]
        op_counter[job_id] += 1
        chosen_m = ma_vec[op_idx]
        op_idx += 1

        # O(1) 数组索引替代线性扫描
        ptime_crisp = crisp_times[job_id][oi][chosen_m]

        start = job_ready[job_id] if job_ready[job_id] > machine_ready[chosen_m] else machine_ready[chosen_m]
        finish = start + ptime_crisp

        job_ready[job_id] = finish
        machine_ready[chosen_m] = finish
        machine_workload[chosen_m] += ptime_crisp

        schedule.append({
            "pos": pos,
            "job_id": job_id,
            "oi": oi,
            "machine": chosen_m,
            "ptime": ptime_crisp,
            "start": start,
            "finish": finish,
            "alts": jobs[job_id][oi],
        })

    makespan = max(job_ready)
    total_workload = sum(machine_workload)

    return schedule, makespan, total_workload, machine_workload


def ls1_swap_machine(os_vec, ma_vec, instance, rng):
    """
    LS1: Randomly select an operation and change its machine to another candidate.
    随机选择一个工序，将其机器改为候选集中的另一个随机机器。
    """
    if not os_vec:
        return os_vec.copy(), ma_vec.copy()

    ma_new = ma_vec.copy()
    idx = rng.randint(len(os_vec))
    job_id, oi = _get_op_index(os_vec, idx)
    alts = instance["jobs"][job_id][oi]

    current_m = ma_new[idx]
    candidates = [alt[0] for alt in alts if alt[0] != current_m]
    if candidates:
        ma_new[idx] = int(rng.choice(candidates))
    return os_vec.copy(), ma_new


def ls2_min_time_machine(os_vec, ma_vec, instance, rng):
    """
    LS2: Randomly select an operation and move it to another machine with minimum processing time.
    随机选择一个工序，将其移动到加工时间最短的机器上。
    """
    if not os_vec:
        return os_vec.copy(), ma_vec.copy()

    ma_new = ma_vec.copy()
    idx = rng.randint(len(os_vec))
    job_id, oi = _get_op_index(os_vec, idx)
    alts = instance["jobs"][job_id][oi]

    current_m = ma_new[idx]
    # 找t2（最可能时间）最小的候选机器，且不同于当前机器
    best_alt = None
    best_t2 = float('inf')
    for alt in alts:
        m, a, b, c = alt
        if m != current_m and b < best_t2:
            best_t2 = b
            best_alt = alt

    if best_alt is not None:
        ma_new[idx] = best_alt[0]
    return os_vec.copy(), ma_new


def ls3_max_workload_machine(os_vec, ma_vec, instance, rng):
    """
    LS3: Find the maximum workload machine. Randomly select an operation processed by it
    and move it to another machine.
    找到负载最大的机器，随机选择一个在其上加工的工序，移动到另一台机器。
    """
    if not os_vec:
        return os_vec.copy(), ma_vec.copy()

    # 先解码获取当前调度信息
    schedule, makespan, total_workload, machine_workload = _build_schedule_info(
        os_vec, ma_vec, instance
    )

    # 找最大负载机器
    max_m = int(np.argmax(machine_workload))
    if machine_workload[max_m] <= 0:
        return os_vec.copy(), ma_vec.copy()

    # 收集在max_m上加工的所有工序位置
    ops_on_max = [s for s in schedule if s["machine"] == max_m]
    if not ops_on_max:
        return os_vec.copy(), ma_vec.copy()

    # 随机选一个工序
    chosen = ops_on_max[rng.randint(len(ops_on_max))]
    pos = chosen["pos"]
    job_id = chosen["job_id"]
    oi = chosen["oi"]
    alts = chosen["alts"]

    # 换到另一个候选机器（随机选一个不同于当前的）
    current_m = ma_vec[pos]
    candidates = [alt[0] for alt in alts if alt[0] != current_m]
    if not candidates:
        return os_vec.copy(), ma_vec.copy()

    ma_new = ma_vec.copy()
    ma_new[pos] = int(rng.choice(candidates))
    return os_vec.copy(), ma_new


def ls4_swap_positions(os_vec, ma_vec, instance, rng):
    """
    LS4: Randomly choose two positions on the operation sequence and exchange the value.
    随机选择OS中两个位置并交换。
    """
    if len(os_vec) < 2:
        return os_vec.copy(), ma_vec.copy()

    os_new = os_vec.copy()
    ma_new = ma_vec.copy()
    i, j = rng.choice(len(os_new), 2, replace=False)
    os_new[i], os_new[j] = os_new[j], os_new[i]
    ma_new[i], ma_new[j] = ma_new[j], ma_new[i]
    return os_new, ma_new


def ls5_insert_position(os_vec, ma_vec, instance, rng):
    """
    LS5: Randomly choose two positions on the operation sequence and insert one before another.
    随机选择OS中两个位置，将其中一个插入到另一个前面。
    论文描述为"Randomly choose two positions on the operation sequence and..."，
    结合上下文应为插入操作（与LS4交换互补）。
    """
    if len(os_vec) < 2:
        return os_vec.copy(), ma_vec.copy()

    os_new = os_vec.copy()
    ma_new = ma_vec.copy()
    i, j = rng.choice(len(os_new), 2, replace=False)

    # 将位置i的元素插入到位置j之前
    elem_os = os_new.pop(i)
    elem_ma = ma_new.pop(i)

    # 如果i<j，pop后j的位置前移1；如果i>j，j位置不变
    insert_pos = j if i > j else j - 1
    os_new.insert(insert_pos, elem_os)
    ma_new.insert(insert_pos, elem_ma)

    return os_new, ma_new


# 局部搜索算子注册表
LOCAL_SEARCH_OPERATORS = [
    ls1_swap_machine,
    ls2_min_time_machine,
    ls3_max_workload_machine,
    ls4_swap_positions,
    ls5_insert_position,
]


class RVNS:
    """
    Variable Neighborhood Search based on RL.
    基于强化学习的变邻域搜索。

    维护成功记忆(SM)和失败记忆(FM)，通过轮盘赌动态选择局部搜索策略。
    """

    def __init__(self, n_operators=5, lp=40, ls_trials=1, mode="rl",
                 budget_mode="fixed", budget_pool=None, budget_target_mean=None):
        """
        Initialize RVNS.

        Parameters:
            n_operators: Number of local search operators (default 5)
            lp: Length of success/failure memory (default 40)
            ls_trials: 每个解每代最多尝试的邻域次数（论文 Algorithm 4 为 1，
                       即 first-improvement 的单步 VNS；调大可增强局部搜索强度）
            mode: "rl"（默认）按 SM/FM 轮盘赌选算子；"random" 等价于论文
                  Section 4.6 提到的用法 (1)——五个算子等概率随机选，
                  即论文变体阶梯里的 RMOEA/D3「randomly selection VNS」。
                  用于把「RL 引导选择」的净贡献「加上局部搜索本身」里剥离出来。
            budget_mode: 邻域尝试次数在种群内的**分配**方式（ABA，见下）。
            budget_pool: 允许的尝试次数档位，默认 ``[1, 3]``（最小档/最大档）。
            budget_target_mean: 每代每解**平均**预算上限。给定时，
                  每代总预算固定为 ``n_pop * budget_target_mean``，
                  因此不同分配策略之间的**总算力完全相同**。

        ── ABA：邻域搜索预算的自适应分配 ──────────────────────────────

        论文 Algorithm 4 是 first-improvement 单步 VNS：每个解每代只尝试 1 次
        邻域（``ls_trials=1``）就返回。实测把全局上限提到 3 是全场最大的单一
        新杠杆（+4.68%），但这**同时把算力提高了 2.2 倍**，属于"更狠"而非
        "更聪明"。ABA 要回答的是：**在总算力固定时，把预算集中投给哪些解**

        四档策略（``budget_mode``）：

        ==============  =====================================================
        "fixed"         所有解统一用 ``ls_trials`` 次上限（论文口径，默认）
        "pool_random"   每代总预算固定，随机挑一部分解升级到最大档（**异质性对照**）
        "pool_state"    同上，但优先升级「上一代局部搜索失败」的解（状态驱动）
        "pool_improved" 同上，但优先升级「上一代刚被改进过」的解（状态驱动，反极性）
        "pool_learn"    同上，升级偏好由 SM/FM 式信用统计**学习**得到
        ==============  =====================================================

        ``pool_improved`` 是**探索性**设定（post-hoc）：Mk10 实测显示
        「升级卡住的解」几乎买不到回报，「升级刚被改进过的解」的回报率高得多，
        因此补一个反极性臂直接检验方向。**在留出集确认之前不得作为主张。**

        ``pool_random`` 是必要的对照：若 ``pool_state`` 胜过 ``fixed``，
        必须能排除"预算的异质性本身就有用"这一解释，否则归因不成立。

        **算力相等是构造性保证的**：``plan_generation`` 每代只发放
        ``n_pop * budget_target_mean`` 份预算，超出即不发——因此各档策略
        的总预算逐代严格相等，差异只来自"发给谁"。
        """
        self.n_operators = n_operators
        self.lp = lp
        self.ls_trials = max(1, int(ls_trials))
        self.mode = mode
        # 成功记忆和失败记忆：每个元素是长度为n_operators的列表
        self.success_memory = []  # SM
        self.failure_memory = []  # FM
        # 当前选择概率
        self.probabilities = np.ones(n_operators) / n_operators

        # ── ABA：预算分配 ──
        self.budget_mode = str(budget_mode)
        pool = sorted({max(1, int(x)) for x in (budget_pool or [1, 3])})
        if len(pool) < 2:
            # 单档位等价于固定预算，直接退回 fixed，避免出现"名义自适应"
            self.budget_mode = "fixed"
            pool = [self.ls_trials, self.ls_trials]
        self.budget_pool = pool
        self.budget_target_mean = float(
            ls_trials if budget_target_mean is None else budget_target_mean)
        # 「上一代这个解是否被局部搜索改进过」——pool_state / pool_learn 的状态量
        # 初值 False（= 全部视为"还没成功过"），使第 1 代的升级名额由打散后的随机并列决定
        self._last_success = []
        # 分桶信用统计：按 stuck∈{0,1} 记录"升级确实带来回报"的次数
        self._up_succ = [0.0, 0.0]
        self._up_total = [0.0, 0.0]
        # 最近一次 apply_local_search 的记账，供 rvns_generation 归因使用
        self._last_trials_used = 0
        self._last_upgrade_paid = False
        # 代际累计，用于事后核对"算力是否真的相等"
        self.budget_stats = {
            "planned_budget": 0,   # 按上限发放的尝试次数合计
            "actual_evals": 0,     # 真实执行的邻域求值次数（命中即返回，可能少于上限）
            "upgrades": 0,         # 获得最大档预算的解次数
            "decisions": 0,        # 参与过预算决策的解次数
        }

    def select_operator(self, rng):
        """
        Roulette wheel selection of local search operator.
        轮盘赌选择局部搜索算子。
        """
        return rng.choice(self.n_operators, p=self.probabilities)

    def apply_local_search(self, os_vec, ma_vec, instance, weight, z, rng,
                            old_mc=None, old_wc=None, trials=None):
        """
        Apply one local search operator to a solution.
        对单个解应用局部搜索（论文 Algorithm 4）。

        接受准则按论文 Algorithm 4 第 3 行，用**子问题的 Tchebycheff 标量化值**：

            g^te(P' | λ, Z*) < g^te(P | λ, Z*)   →   更新解 P ← P'

        注意：这里**不能**用 Pareto 支配判定。双目标下，一个随机邻域同时不劣化
        两个目标的概率极低，支配判定会导致局部搜索在种群收敛后完全空转
        （实测接受率很快降到 0/100 并保持到结束）。λ 与 Z* 正是为此传入的。

        优化：接收已知旧解 crisp 值，消除热路径中重复的 decode 调用。
        """
        from .moead import tchebycheff  # 局部导入：避免模块级循环依赖

        if old_mc is None or old_wc is None:
            old_mc, old_wc = decode_crisp(os_vec, ma_vec, instance)

        old_g = tchebycheff((old_mc, old_wc), weight, z)

        # trials 由 ABA 的 plan_generation 逐解决定；None 时沿用全局 ls_trials
        n_trials = self.ls_trials if trials is None else max(1, int(trials))
        self._last_trials_used = 0
        self._last_upgrade_paid = False

        for t_i in range(n_trials):
            op_idx = self.select_operator(rng)
            ls_func = LOCAL_SEARCH_OPERATORS[op_idx]

            # Generate neighbor
            new_os, new_ma = ls_func(os_vec, ma_vec, instance, rng)

            # 修复MA合法性（OS变化后MA可能不合法，始终调用）
            new_ma = _repair_ma_for_os(new_os, new_ma, instance, rng)

            # Evaluate: 旧值复用入参，新值用 decode_crisp（零分配）
            new_mc, new_wc = decode_crisp(new_os, new_ma, instance)
            self.budget_stats["actual_evals"] += 1

            # 子问题标量化值是否改善（越小越好）
            new_g = tchebycheff((new_mc, new_wc), weight, z)
            success = new_g < old_g - 1e-12

            # Update memories
            self._update_memory(op_idx, success)

            if success:
                self._last_trials_used = t_i + 1
                # 只有第 2 次及以后命中，才算这次"升级"真买到了东西；
                # 第 1 次就命中说明升级名额被浪费了（首试本来就会成功）
                self._last_upgrade_paid = bool(t_i > 0)
                return new_os, new_ma, True

        self._last_trials_used = n_trials
        return os_vec, ma_vec, False

    def _update_memory(self, op_idx, success):
        """
        Update success/failure memory (deferred probability update).
        仅写入记忆，不立即重算概率。概率由 rvns_generation 代末统一更新，
        将 O(n_pop) 次 _update_probabilities 合并为 1 次。
        """
        ns_record = [0] * self.n_operators
        nf_record = [0] * self.n_operators
        if success:
            ns_record[op_idx] = 1
        else:
            nf_record[op_idx] = 1

        self.success_memory.append(ns_record)
        self.failure_memory.append(nf_record)

        if len(self.success_memory) > self.lp:
            self.success_memory.pop(0)
            self.failure_memory.pop(0)
        # 概率更新推迟到 rvns_generation 代末统一调用

    def _update_probabilities(self):
        """
        Recalculate selection probabilities based on SM and FM.
        根据成功/失败记忆重新计算选择概率。

        mode="random" 时保持等概率：算子选择退化为论文的用法 (1)，
        记忆仍然记录但不再影响选择，因此该配置就是 RMOEA/D3 的随机 VNS。
        """
        if self.mode == "random":
            self.probabilities = np.ones(self.n_operators) / self.n_operators
            return

        if not self.success_memory:
            self.probabilities = np.ones(self.n_operators) / self.n_operators
            return

        # 按列求和
        sr = np.sum(self.success_memory, axis=0).astype(float)
        fr = np.sum(self.failure_memory, axis=0).astype(float)

        # 计算原始概率
        raw_probs = np.zeros(self.n_operators)
        for i in range(self.n_operators):
            total = sr[i] + fr[i]
            if total > 0:
                raw_probs[i] = sr[i] / total
            else:
                # 从未尝试过的算子给一个小的基础概率
                raw_probs[i] = 0.1

        # 归一化确保和为1
        sum_probs = np.sum(raw_probs)
        if sum_probs > 0:
            self.probabilities = raw_probs / sum_probs
        else:
            self.probabilities = np.ones(self.n_operators) / self.n_operators

    def get_probabilities(self):
        """Return current operator selection probabilities."""
        return self.probabilities.tolist()

    # ──────────────────────────────────────────────────────────────
    # ABA：邻域搜索预算的自适应分配
    # ──────────────────────────────────────────────────────────────

    def plan_generation(self, n_pop, rng):
        """
        为一代发放邻域尝试预算，返回长度 ``n_pop`` 的 trials 上限数组。

        总算力**构造性相等**：先给每个解发最小档，再把
        ``n_pop * budget_target_mean - n_pop * base`` 份剩余预算按
        ``top - base`` 的步长发给被选中的解。因此四档策略之间
        总预算逐代严格相同，HV 差异只可能来自"发给谁"。
        """
        base = self.budget_pool[0]
        top = self.budget_pool[-1]
        budget = np.full(n_pop, base, dtype=np.int64)

        def _account():
            self.budget_stats["planned_budget"] += int(budget.sum())
            self.budget_stats["decisions"] += n_pop
            return budget

        if self.budget_mode == "fixed" or top <= base:
            budget[:] = self.ls_trials
            return _account()

        step = top - base
        remaining = int(round(n_pop * self.budget_target_mean)) - n_pop * base
        n_up = remaining // step
        if n_up <= 0:
            return _account()
        n_up = min(n_up, n_pop)

        if self.budget_mode == "pool_random":
            # 异质性对照：升级名额随机发放，与"发得准不准"无关
            score = rng.rand(n_pop)
        else:
            stuck = self._stuck_flags(n_pop)
            if self.budget_mode == "pool_learn":
                # 学习档：按「升级能买到改善」的历史概率排序，并列用微小噪声打散
                p = self.budget_propensities()
                score = p[stuck] + rng.rand(n_pop) * 1e-3
            elif self.budget_mode == "pool_improved":
                # 反极性（探索性）：优先升级"上一代刚被改进过"的解。
                # 注意：credit 统计本身带策略依赖（升级对象由当前策略决定），
                # 因此不能靠读 propensity 反推方向，必须直接跑这个臂来检验。
                score = (1 - stuck).astype(float) + rng.rand(n_pop) * 0.5
            else:  # "pool_state"：状态驱动，优先升级上一代没被改进过的解
                score = stuck.astype(float) + rng.rand(n_pop) * 0.5

        idx = np.argsort(-score)[:n_up]
        budget[idx] = top
        self.budget_stats["upgrades"] += int(n_up)
        return _account()

    def _stuck_flags(self, n_pop):
        """上一代各解是否未被局部搜索改进（长度对齐，缺省视为 stuck=1）。"""
        flags = np.ones(n_pop, dtype=np.int64)
        m = min(len(self._last_success), n_pop)
        for i in range(m):
            flags[i] = 0 if self._last_success[i] else 1
        return flags

    def budget_propensities(self):
        """按 stuck 状态估计「升级能买到改善」的概率（Laplace 平滑）。"""
        return np.array([(self._up_succ[s] + 1.0) / (self._up_total[s] + 2.0)
                         for s in (0, 1)])

    def _record_budget_outcome(self, stuck, paid):
        """把一次升级的结果写进分桶信用统计。"""
        s = 1 if stuck else 0
        self._up_total[s] += 1.0
        if paid:
            self._up_succ[s] += 1.0

    def get_budget_stats(self):
        """预算分配的可核对统计。

        ``mean_planned_trials`` 是各档策略之间**算力是否真相等**的判据；
        ``mean_actual_evals`` 是真实邻域求值次数（命中即返回，可能少于上限）。
        两者都要报——否则"等算力"只是声称。
        """
        st = dict(self.budget_stats)
        st["budget_mode"] = self.budget_mode
        st["budget_pool"] = list(self.budget_pool)
        st["budget_target_mean"] = self.budget_target_mean
        st["propensities"] = self.budget_propensities().tolist()
        st["upgrade_records"] = [list(self._up_total), list(self._up_succ)]
        d = st["decisions"]
        st["mean_planned_trials"] = (st["planned_budget"] / d) if d else 0.0
        st["mean_actual_evals"] = (st["actual_evals"] / d) if d else 0.0
        return st


def rvns_generation(population, objectives, weights, instance, z, rng, rvns):
    """
    Apply RVNS to each solution in the population (论文 Algorithm 4).
    对种群中每个解应用RVNS局部搜索。

    Parameters:
        population: list of (os, ma)
        objectives: list of (makespan_crisp, workload_crisp)
        weights: weight vectors
        instance: problem instance
        z: reference point
        rng: random state
        rvns: RVNS instance

    Returns:
        new_population, new_objectives, new_z
    """
    new_pop = [p for p in population]
    new_obj = [list(o) for o in objectives]
    z = list(z)

    n_pop = len(population)
    # ABA：先为整代发放预算（总算力固定），再逐解执行——
    # 顺序很关键，"先定总额再分配"才能保证各策略之间的算力严格相等。
    trials_plan = rvns.plan_generation(n_pop, rng)
    stuck_flags = rvns._stuck_flags(n_pop)

    success_count = 0
    improved_flags = []

    for i in range(len(population)):
        os_vec, ma_vec = new_pop[i]
        weight = weights[i]
        old_mc, old_wc = new_obj[i]  # 复用已解码的旧值，避免重复 decode

        t_i = int(trials_plan[i])
        new_os, new_ma, success = rvns.apply_local_search(
            os_vec, ma_vec, instance, weight, tuple(z), rng,
            old_mc=old_mc, old_wc=old_wc, trials=t_i,
        )

        # 预算信用只在"被升级过"的解上结算，且 payoff 要求成功发生在第 2 次及以后
        if rvns.budget_mode != "fixed" and t_i > rvns.budget_pool[0]:
            rvns._record_budget_outcome(
                bool(stuck_flags[i]), bool(success and rvns._last_upgrade_paid))

        if success:
            # 更新解和目标值（crisp-only）
            new_mc, new_wc = decode_crisp(new_os, new_ma, instance)
            new_pop[i] = (new_os, new_ma)
            new_obj[i] = [new_mc, new_wc]
            z[0] = min(z[0], new_mc)
            z[1] = min(z[1], new_wc)
            success_count += 1

        improved_flags.append(bool(success))

    # 供下一代 pool_state / pool_learn 判定"这个解是否卡住了"
    rvns._last_success = improved_flags

    # 代末统一更新概率（从 3000 次 → 30 次调用）
    rvns._update_probabilities()

    logger.debug("RVNS generation completed: %d/%d solutions improved",
                 success_count, len(population))
    return new_pop, [tuple(o) for o in new_obj], tuple(z)
