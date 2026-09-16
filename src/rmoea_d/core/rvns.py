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

    def __init__(self, n_operators=5, lp=40, ls_trials=1):
        """
        Initialize RVNS.

        Parameters:
            n_operators: Number of local search operators (default 5)
            lp: Length of success/failure memory (default 40)
            ls_trials: 每个解每代最多尝试的邻域次数（论文 Algorithm 4 为 1，
                       即 first-improvement 的单步 VNS；调大可增强局部搜索强度）
        """
        self.n_operators = n_operators
        self.lp = lp
        self.ls_trials = max(1, int(ls_trials))
        # 成功记忆和失败记忆：每个元素是长度为n_operators的列表
        self.success_memory = []  # SM
        self.failure_memory = []  # FM
        # 当前选择概率
        self.probabilities = np.ones(n_operators) / n_operators

    def select_operator(self, rng):
        """
        Roulette wheel selection of local search operator.
        轮盘赌选择局部搜索算子。
        """
        return rng.choice(self.n_operators, p=self.probabilities)

    def apply_local_search(self, os_vec, ma_vec, instance, weight, z, rng,
                            old_mc=None, old_wc=None):
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

        for _ in range(self.ls_trials):
            op_idx = self.select_operator(rng)
            ls_func = LOCAL_SEARCH_OPERATORS[op_idx]

            # Generate neighbor
            new_os, new_ma = ls_func(os_vec, ma_vec, instance, rng)

            # 修复MA合法性（OS变化后MA可能不合法，始终调用）
            new_ma = _repair_ma_for_os(new_os, new_ma, instance, rng)

            # Evaluate: 旧值复用入参，新值用 decode_crisp（零分配）
            new_mc, new_wc = decode_crisp(new_os, new_ma, instance)

            # 子问题标量化值是否改善（越小越好）
            new_g = tchebycheff((new_mc, new_wc), weight, z)
            success = new_g < old_g - 1e-12

            # Update memories
            self._update_memory(op_idx, success)

            if success:
                return new_os, new_ma, True

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
        """
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

    success_count = 0

    for i in range(len(population)):
        os_vec, ma_vec = new_pop[i]
        weight = weights[i]
        old_mc, old_wc = new_obj[i]  # 复用已解码的旧值，避免重复 decode

        new_os, new_ma, success = rvns.apply_local_search(
            os_vec, ma_vec, instance, weight, tuple(z), rng,
            old_mc=old_mc, old_wc=old_wc,
        )

        if success:
            # 更新解和目标值（crisp-only）
            new_mc, new_wc = decode_crisp(new_os, new_ma, instance)
            new_pop[i] = (new_os, new_ma)
            new_obj[i] = [new_mc, new_wc]
            z[0] = min(z[0], new_mc)
            z[1] = min(z[1], new_wc)
            success_count += 1

    # 代末统一更新概率（从 3000 次 → 30 次调用）
    rvns._update_probabilities()

    logger.debug("RVNS generation completed: %d/%d solutions improved",
                 success_count, len(population))
    return new_pop, [tuple(o) for o in new_obj], tuple(z)
