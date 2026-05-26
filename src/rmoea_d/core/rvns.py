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

from .encoding import decode
from .operators import _get_op_index, _repair_ma_for_os

logger = logging.getLogger(__name__)


def _build_schedule_info(os_vec, ma_vec, instance):
    """
    解码并构建完整的调度信息，用于局部搜索。
    返回每个工序的详细调度信息列表。
    """
    n_jobs = instance["n_jobs"]
    n_machines = instance["n_machines"]
    jobs = instance["jobs"]

    op_counter = [0] * n_jobs
    job_ready = [0.0] * n_jobs          # 清晰值时间
    machine_ready = [0.0] * n_machines
    machine_workload = [0.0] * n_machines

    schedule = []  # list of dicts
    op_idx = 0

    for pos, job_id in enumerate(os_vec):
        oi = op_counter[job_id]
        op_counter[job_id] += 1
        alts = jobs[job_id][oi]
        chosen_m = ma_vec[op_idx]
        op_idx += 1

        # 找到加工时间
        proc = None
        for alt in alts:
            if alt[0] == chosen_m:
                proc = alt
                break
        if proc is None:
            proc = alts[0]
            chosen_m = proc[0]

        _, a, b, c = proc
        ptime_crisp = (a + 2.0 * b + c) / 4.0

        start = max(job_ready[job_id], machine_ready[chosen_m])
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
            "alts": alts,
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

    def __init__(self, n_operators=5, lp=40):
        """
        Initialize RVNS.

        Parameters:
            n_operators: Number of local search operators (default 5)
            lp: Length of success/failure memory (default 40)
        """
        self.n_operators = n_operators
        self.lp = lp
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

    def apply_local_search(self, os_vec, ma_vec, instance, weight, z, rng):
        """
        Apply one local search operator to a solution.
        对单个解应用一次局部搜索。

        Parameters:
            os_vec, ma_vec: Current solution
            instance: Problem instance
            weight: Weight vector for Tchebycheff (lambda_i)
            z: Reference point
            rng: Random state

        Returns:
            new_os, new_ma, success_flag
        """
        op_idx = self.select_operator(rng)
        ls_func = LOCAL_SEARCH_OPERATORS[op_idx]

        # Generate neighbor
        new_os, new_ma = ls_func(os_vec, ma_vec, instance, rng)

        # 修复MA合法性（OS变化后）
        if new_os != os_vec:
            new_ma = _repair_ma_for_os(new_os, new_ma, instance, rng)

        # Evaluate old and new solution using dominance relation
        _, _, old_mc, old_wc = decode(os_vec, ma_vec, instance)
        _, _, new_mc, new_wc = decode(new_os, new_ma, instance)

        # Check if new solution dominates old solution
        # 新解支配旧解：新解在至少一个目标上严格更好，且在所有目标上不差
        success = (new_mc < old_mc and new_wc <= old_wc) or (new_mc <= old_mc and new_wc < old_wc)

        # Update memories
        self._update_memory(op_idx, success)

        if success:
            return new_os, new_ma, True
        else:
            return os_vec, ma_vec, False

    def _update_memory(self, op_idx, success):
        """
        Update success/failure memory and recalculate probabilities.
        更新成功/失败记忆并重新计算选择概率。
        """
        # 创建当前代的记录
        ns_record = [0] * self.n_operators
        nf_record = [0] * self.n_operators
        if success:
            ns_record[op_idx] = 1
        else:
            nf_record[op_idx] = 1

        # 添加到记忆尾部
        self.success_memory.append(ns_record)
        self.failure_memory.append(nf_record)

        # 如果超过长度限制，删除头部记录
        if len(self.success_memory) > self.lp:
            self.success_memory.pop(0)
            self.failure_memory.pop(0)

        # 重新计算概率
        self._update_probabilities()

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

        new_os, new_ma, success = rvns.apply_local_search(
            os_vec, ma_vec, instance, weight, tuple(z), rng
        )

        if success:
            # 更新解和目标值
            _, _, new_mc, new_wc = decode(new_os, new_ma, instance)
            new_pop[i] = (new_os, new_ma)
            new_obj[i] = [new_mc, new_wc]
            # 更新参考点
            z[0] = min(z[0], new_mc)
            z[1] = min(z[1], new_wc)
            success_count += 1

    logger.info("RVNS generation completed: %d/%d solutions improved",
                success_count, len(population))
    return new_pop, [tuple(o) for o in new_obj], tuple(z)
