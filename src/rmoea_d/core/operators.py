"""
Genetic operators: initialization, crossover, mutation, repair.
遗传算子模块：初始化、交叉、变异、修复。
"""

import numpy as np


def init_random(instance, rng):
    """Random initialization.
    随机初始化：随机打乱工序顺序，每个工序随机选一台候选机器。"""
    n_jobs = instance["n_jobs"]
    jobs = instance["jobs"]
    os = []
    for j in range(n_jobs):
        os.extend([j] * len(jobs[j]))
    rng.shuffle(os)

    op_counter = [0] * n_jobs
    ma = []
    for job_id in os:
        oi = op_counter[job_id]
        op_counter[job_id] += 1
        alts = jobs[job_id][oi]
        choice = rng.randint(len(alts))
        ma.append(alts[choice][0])
    return os, ma


def init_ls(instance, rng):
    """Least processing time initialization.
    最短时间初始化：随机工序顺序，每个工序选t2（最可能时间）最小的机器。"""
    n_jobs = instance["n_jobs"]
    jobs = instance["jobs"]
    os = []
    for j in range(n_jobs):
        os.extend([j] * len(jobs[j]))
    rng.shuffle(os)

    op_counter = [0] * n_jobs
    ma = []
    for job_id in os:
        oi = op_counter[job_id]
        op_counter[job_id] += 1
        alts = jobs[job_id][oi]
        # Pick machine with minimum t2 (most likely time)
        best = min(alts, key=lambda x: x[2])
        ma.append(best[0])
    return os, ma


def init_gw(instance, rng):
    """Global workload initialization.
    全局工作负载初始化：先排所有工件第一道工序，再排剩余工序；
    机器选择时优先选当前总负载增加最少的机器。"""
    n_jobs = instance["n_jobs"]
    jobs = instance["jobs"]
    # First: all first operations in random order
    first_ops = list(range(n_jobs))
    rng.shuffle(first_ops)
    # Then remaining ops
    remaining = []
    for j in range(n_jobs):
        remaining.extend([j] * (len(jobs[j]) - 1))
    rng.shuffle(remaining)
    os = first_ops + remaining

    machine_load = [0.0] * instance["n_machines"]
    op_counter = [0] * n_jobs
    ma = []
    for job_id in os:
        oi = op_counter[job_id]
        op_counter[job_id] += 1
        alts = jobs[job_id][oi]
        # Pick machine that adds least crisp workload
        best_alt = None
        best_total = float('inf')
        for alt in alts:
            m, a, b, c = alt
            crisp = (a + 2 * b + c) / 4.0
            new_total = machine_load[m] + crisp
            if new_total < best_total:
                best_total = new_total
                best_alt = alt
            elif new_total == best_total and b < best_alt[2]:
                best_alt = alt
        ma.append(best_alt[0])
        machine_load[best_alt[0]] += (best_alt[1] + 2 * best_alt[2] + best_alt[3]) / 4.0
    return os, ma


def init_mix3(instance, n_pop, rng):
    """MIX3 initialization: 1/3 random, 1/3 LS, 1/3 GW.
    MIX3初始化策略：三分之一随机、三分之一最短时间、三分之一全局负载均衡。"""
    population = []
    thirds = n_pop // 3
    for _ in range(thirds):
        population.append(init_random(instance, rng))
    for _ in range(thirds):
        population.append(init_ls(instance, rng))
    for _ in range(thirds):
        population.append(init_gw(instance, rng))
    while len(population) < n_pop:
        population.append(init_random(instance, rng))
    return population[:n_pop]


def pox_crossover(os1, os2, rng):
    """Precedence Operation Crossover for OS vectors.
    基于工件顺序的交叉(POX)：随机分组，保留一组工件位置，其余按另一父代顺序填充。"""
    n_jobs = max(os1) + 1 if os1 else 0
    if n_jobs == 0:
        return os1.copy(), os2.copy()

    jobs = list(range(n_jobs))
    rng.shuffle(jobs)
    split = rng.randint(1, n_jobs)
    group1 = set(jobs[:split])

    def make_child(p1, p2, keep_group):
        child = [None] * len(p1)
        # Keep positions of keep_group jobs from p1
        for i, job in enumerate(p1):
            if job in keep_group:
                child[i] = job
        # Fill remaining with other jobs from p2 in order
        other_jobs = [j for j in p2 if j not in keep_group]
        ptr = 0
        for i in range(len(child)):
            if child[i] is None:
                child[i] = other_jobs[ptr]
                ptr += 1
        return child

    c1 = make_child(os1, os2, group1)
    c2 = make_child(os2, os1, group1)
    return c1, c2


def ux_crossover(ma1, ma2, rng):
    """Uniform crossover for MA vectors.
    均匀交叉(UX)：随机掩码决定从哪个父代继承机器分配。"""
    mask = rng.randint(0, 2, size=len(ma1))
    c1 = [ma2[i] if mask[i] else ma1[i] for i in range(len(ma1))]
    c2 = [ma1[i] if mask[i] else ma2[i] for i in range(len(ma1))]
    return c1, c2


def mutate_os(os_vec, rng):
    """Swap two random positions in OS.
    OS变异：随机交换两个位置的工序。"""
    os_vec = os_vec.copy()
    i, j = rng.choice(len(os_vec), 2, replace=False)
    os_vec[i], os_vec[j] = os_vec[j], os_vec[i]
    return os_vec


def _get_op_index(os_vec, idx):
    """Get operation index (oi) for the job at position idx in OS.
    获取OS向量中位置idx对应的工序索引（该工件的第几道工序）。"""
    job_id = os_vec[idx]
    count = 0
    for i in range(idx):
        if os_vec[i] == job_id:
            count += 1
    return job_id, count


def _repair_ma_for_os(os_vec, ma_vec, instance, rng):
    """Repair MA to ensure each machine is valid for its corresponding operation in OS.
    修复MA向量：确保每个位置的机器对应该位置工序的候选机器集。
    使用预计算的 valid_machines 集合，O(1) 查找替代 O(n_alts) 遍历。
    """
    ma_vec = ma_vec.copy()
    op_counter = [0] * instance["n_jobs"]
    valid_machines = instance["valid_machines"]
    for idx, job_id in enumerate(os_vec):
        oi = op_counter[job_id]
        op_counter[job_id] += 1
        if ma_vec[idx] not in valid_machines[job_id][oi]:
            # 随机选一个合法机器（集合随机选择比列表重建快）
            ma_vec[idx] = rng.choice(list(valid_machines[job_id][oi]))
    return ma_vec


def mutate_ma(ma_vec, os_vec, instance, rng):
    """Change one operation's machine to another candidate.
    MA变异：随机选一个工序，将其机器改为候选集中的另一个随机可选机器。"""
    ma_vec = ma_vec.copy()
    idx = rng.randint(len(ma_vec))
    job_id, oi = _get_op_index(os_vec, idx)
    alts = instance["jobs"][job_id][oi]
    current_m = ma_vec[idx]
    candidates = [alt[0] for alt in alts if alt[0] != current_m]
    if candidates:
        ma_vec[idx] = int(rng.choice(candidates))
    return ma_vec


def repair_os(os_vec, instance):
    """Ensure OS has correct job counts.
    修复OS向量：确保每个工件出现的次数等于其工序数。"""
    n_jobs = instance["n_jobs"]
    expected_counts = [len(instance["jobs"][j]) for j in range(n_jobs)]
    actual_counts = [0] * n_jobs
    for j in os_vec:
        actual_counts[j] += 1

    # If counts match, return as-is
    if actual_counts == expected_counts:
        return os_vec.copy()

    # Build corrected OS
    missing = []
    for j in range(n_jobs):
        diff = expected_counts[j] - actual_counts[j]
        if diff > 0:
            missing.extend([j] * diff)

    result = os_vec.copy()
    # Remove excess
    for j in range(n_jobs):
        excess = actual_counts[j] - expected_counts[j]
        if excess > 0:
            removed = 0
            for i in range(len(result) - 1, -1, -1):
                if result[i] == j and removed < excess:
                    result[i] = -1
                    removed += 1

    # Fill missing
    rng = np.random.RandomState()
    rng.shuffle(missing)
    ptr = 0
    for i in range(len(result)):
        if result[i] == -1:
            result[i] = missing[ptr]
            ptr += 1
    return result
