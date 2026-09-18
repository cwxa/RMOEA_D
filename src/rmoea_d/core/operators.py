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


def init_os_spt(instance, rng, explore=0.25):
    """派工式初始化：最短加工时间优先（SPT）+ 随机探索。

    **与 MIX3 三条分支的本质区别**

    论文的 MIX3 是 1/3 random + 1/3 LS + 1/3 GW，三条分支**都把 OS 随机打乱**，
    只在**机器选择（MA 维度）**上做文章：

        init_random : 随机工序序 + 随机机器
        init_ls     : 随机工序序 + t2 最小机器
        init_gw     : 随机工序序 + 负载最轻机器

    也就是说，工序顺序这个维度**从未被初始化利用过**。这里改为用派工规则直接
    生成 OS：每一步从「各工件待排的下一道工序」中挑 t2 最短的一道入列。

    为避免同一 run 内 1/3 的个体完全同质（SPT 是确定性的），以 `explore`
    概率改选一个随机可用工件；`rng.rand()` 无条件调用一次，使随机流消耗稳定。
    """
    n_jobs = instance["n_jobs"]
    jobs = instance["jobs"]
    min_t2 = instance["min_t2"]

    op_counter = [0] * n_jobs
    available = list(range(n_jobs))
    os = []
    ma = []
    for _ in range(instance["total_ops"]):
        if rng.rand() < explore:
            j = available[rng.randint(len(available))]
        else:
            j = available[0]
            best_t = min_t2[j][op_counter[j]]
            for k in available[1:]:
                t = min_t2[k][op_counter[k]]
                if t < best_t:
                    best_t = t
                    j = k
        oi = op_counter[j]
        os.append(j)
        op_counter[j] = oi + 1
        if op_counter[j] >= len(jobs[j]):
            available.remove(j)
        alts = jobs[j][oi]
        ma.append(min(alts, key=lambda x: x[2])[0])
    return os, ma


def init_os_mwr(instance, rng, explore=0.25):
    """派工式初始化：剩余工作量最大优先（Most Work Remaining）+ 随机探索。

    与 `init_os_spt` 互补：每一步选「剩余工序 t2 之和」最大的工件入列，
    先把长工件清掉，避免它们在序列末尾堆积成 makespan 瓶颈。
    """
    n_jobs = instance["n_jobs"]
    jobs = instance["jobs"]
    min_t2 = instance["min_t2"]

    op_counter = [0] * n_jobs
    remaining = [sum(row) for row in min_t2]
    available = list(range(n_jobs))
    os = []
    ma = []
    for _ in range(instance["total_ops"]):
        if rng.rand() < explore:
            j = available[rng.randint(len(available))]
        else:
            j = available[0]
            best_r = remaining[j]
            for k in available[1:]:
                if remaining[k] > best_r:
                    best_r = remaining[k]
                    j = k
        oi = op_counter[j]
        os.append(j)
        op_counter[j] = oi + 1
        remaining[j] -= min_t2[j][oi]
        if op_counter[j] >= len(jobs[j]):
            available.remove(j)
        alts = jobs[j][oi]
        ma.append(min(alts, key=lambda x: x[2])[0])
    return os, ma


# 初始化变体注册表：名称 -> (random, ls, gw, spt, mwr) 的整数配比
#
# 论文口径只有 "mix3"（等价于原来的 init_mix3 硬编码实现）；
# 其余条目是**本轮新增的变体扫描对象** —— MIX3 是消融阶梯里最大的单一组件
# （实例边界口径 **+3.406%**；旧值 +14.46% 是盒口径，放大 4.9×，见 docs/new-arch-report.md §1），
# 却从未做过变体扫描；而它三条分支全部放弃了 OS 维度。
INIT_VARIANTS = {
    "mix3":        (1, 1, 1, 0, 0),   # 论文口径
    "random":      (1, 0, 0, 0, 0),   # = 论文 RMOEA/D1
    "mix3_spt":    (1, 1, 0, 1, 0),   # 用 OS-SPT 替换 GW 分支
    "mix3_mwr":    (1, 1, 0, 0, 1),   # 用 OS-MWR 替换 GW 分支
    "mix3_gw_spt": (1, 1, 1, 1, 0),   # 加一路 OS-SPT（4 等分）
    "half_random": (2, 1, 1, 0, 0),   # 提高随机占比
    "no_random":   (0, 1, 1, 0, 0),   # 去掉随机分支
}


def init_by_variant(instance, n_pop, rng, variant="mix3"):
    """按变体配比组合初始化策略。

    分桶用「最大余额法」，保证各桶之和恰好 n_pop；个体排列顺序固定为
        [random, ls, gw, spt, mwr]
    与原始 `init_mix3` 的 [random, ls, gw] 顺序一致 —— 因此
    `init_by_variant(inst, n, rng, "mix3")` 与 `init_mix3(inst, n, rng)`
    在同 seed 下**逐位相同**（见 tests 里的等价锁）。
    """
    if variant not in INIT_VARIANTS:
        raise ValueError("unknown init variant: %r (可选: %s)"
                         % (variant, sorted(INIT_VARIANTS)))
    ratios = INIT_VARIANTS[variant]
    total = sum(ratios)
    if total == 0:
        raise ValueError("variant %r 的配比全为 0" % variant)

    # 每桶 n_pop // total * ratio 个；**余数全部补进 random 桶并追加在末尾** ——
    # 这正是 init_mix3 的 `while len(population) < n_pop: append(init_random(...))`。
    # 必须逐字复刻：random / ls / gw 三者消耗的随机数**个数**不同，一旦余数的
    # 落点变了，rng 的消耗顺序就变，"mix3" 不再逐位等价于论文口径。
    base = n_pop // total
    counts = [base * r for r in ratios]
    rest = n_pop - sum(counts)

    generators = (init_random, init_ls, init_gw, init_os_spt, init_os_mwr)
    population = []
    for gen, c in zip(generators, counts):
        for _ in range(c):
            population.append(gen(instance, rng))
    for _ in range(rest):
        population.append(init_random(instance, rng))
    return population


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


def fallback_candidates(instance):
    """惰性预构建「候选机器列表」表：(job, oi) -> list，供不合法位置随机挑选。

    原实现每次现场 `list(valid_machines[j][oi])` 构造一次；POX 交叉后平均每次
    repair 有 ~11 处不合法，故这个转换是热路径。改成一次性构建并缓存。
    同一进程内 `list(set)` 的结果是确定的，故缓存内容与现场构造逐位相同。
    """
    tbl = instance.get("_fallback_candidates")
    if tbl is None:
        tbl = [[list(s) for s in job_valid]
               for job_valid in instance["valid_machines"]]
        instance["_fallback_candidates"] = tbl
    return tbl


def _repair_ma_for_os(os_vec, ma_vec, instance, rng):
    """Repair MA to ensure each machine is valid for its corresponding operation in OS.
    修复MA向量：确保每个位置的机器对应该位置工序的候选机器集。

    **两处等价改写（随机流逐位不变）**

    1. `rng.choice(list(set))` → `cand_list[rng.randint(len(cand_list))]`。
       `RandomState.choice` 即便 `size=None` 也走 `np.prod(size)` 通用路径，实测
       8.2us/次；`randint` 只要 2.0us。两者消耗的底层随机数序列**逐位相同**——
       `scripts/diag_rng_equiv.py` 对 k=1..15 逐位比对过 rng 状态缓冲（值 + buffer
       + 内部位置三者全同）。而 POX 交叉后平均每次调用有 ~11 处不合法，
       所以这是本模块最大的单点开销。
    2. `list(valid_machines[j][oi])` 预缓存进 `_fallback_candidates`，省掉集合转列表。

    **刻意不用 numpy 批量化**：实测（`scripts/profile_wall.py`）把 240 元素的检查
    循环换成 `argsort` + fancy-index 之后整体反而**慢 8–11%** —— numpy 每次调用的
    固定开销摊不平 240 次纯 Python 循环的成本。cProfile 会严重高估纯 Python 循环
    的相对成本（逐行插桩），据此做的优化决策是错的。

    等价性锁：`tests/test_refactor.py::TestRepairFastPath`。
    """
    ma_vec = ma_vec.copy()
    valid_machines = instance["valid_machines"]
    fallback = fallback_candidates(instance)
    op_counter = [0] * instance["n_jobs"]
    for idx, job_id in enumerate(os_vec):
        oi = op_counter[job_id]
        op_counter[job_id] = oi + 1
        if ma_vec[idx] not in valid_machines[job_id][oi]:
            # 随机选一个合法机器（集合随机选择比列表重建快）
            cand = fallback[job_id][oi]
            ma_vec[idx] = cand[rng.randint(len(cand))]
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
        ma_vec[idx] = candidates[rng.randint(len(candidates))]
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
