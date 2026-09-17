"""
MOEA/D framework: weights, neighborhood, Tchebycheff, generation update.
MOEA/D框架模块：权重向量、邻居结构、Tchebycheff分解、代际更新。
"""

import numpy as np
import time
import logging

from .encoding import decode_crisp
from .operators import pox_crossover, ux_crossover, mutate_os, mutate_ma, repair_os, _repair_ma_for_os

logger = logging.getLogger(__name__)


def generate_weights(n_pop):
    """Generate uniform weight vectors for 2 objectives (Das & Dennis).
    使用Das & Dennis方法生成两目标均匀权重向量。"""
    if n_pop == 1:
        return np.array([[0.5, 0.5]])
    weights = []
    for i in range(n_pop):
        w1 = i / (n_pop - 1)
        w2 = 1.0 - w1
        weights.append([w1, w2])
    return np.array(weights, dtype=float)


def compute_neighbors(weights, T):
    """Compute T nearest neighbors for each weight vector by Euclidean distance.
    基于欧氏距离为每个权重向量计算T个最近邻居。"""
    n = len(weights)
    B = []
    for i in range(n):
        dists = np.linalg.norm(weights - weights[i], axis=1)
        idx = np.argsort(dists)[:T]
        B.append(idx.tolist())
    return B


def tchebycheff(f, weight, z):
    """Tchebycheff scalarizing function using crisp values (minimization).
    Tchebycheff标量化函数，使用清晰值进行最小化。"""
    return max(weight[0] * abs(f[0] - z[0]), weight[1] * abs(f[1] - z[1]))


def moead_generation(population, objectives, weights, B, instance, z, crossover_rate, rng):
    """
    Execute one generation of MOEA/D.
    执行一代MOEA/D演化。

    population: list of (os, ma)
    objectives: list of (makespan_crisp, workload_crisp)
    weights: Np x 2 array
    B: list of neighbor index lists
    z: tuple (z1, z2) reference point
    crossover_rate: probability of crossover
    rng: numpy RandomState

    Returns: new_population, new_objectives, new_z
    """
    start_time = time.perf_counter()
    n_pop = len(population)
    new_pop = [p for p in population]
    new_obj = [list(o) for o in objectives]
    z = list(z)

    # weights 是 numpy 数组，`weights[j][0]` 每次都会产生 np.float64 标量，
    # 其算术比 Python float 慢数倍；邻居内层循环每代要跑 n_pop*T 次。
    # 转成 Python list-of-list 后省下的标量开销可观，数值完全一致（同一个 double）。
    w_list = weights.tolist() if hasattr(weights, "tolist") else weights

    update_count = 0

    for i in range(n_pop):
        neighbors = B[i]
        if len(neighbors) < 2:
            continue

        # Select two parents from neighborhood
        # 从邻居中随机选择两个父代
        p1_idx, p2_idx = rng.choice(neighbors, 2, replace=False)
        os1, ma1 = new_pop[p1_idx]
        os2, ma2 = new_pop[p2_idx]

        # Crossover with probability
        # 以一定概率进行交叉
        if rng.rand() < crossover_rate:
            child_os, _ = pox_crossover(os1, os2, rng)
            child_ma, _ = ux_crossover(ma1, ma2, rng)
            # Repair MA after crossover: OS changed but MA may not match
            # 交叉后修复MA：OS变化后MA可能不再匹配新位置的工序
            child_ma = _repair_ma_for_os(child_os, child_ma, instance, rng)
        else:
            child_os = os1.copy()
            child_ma = ma1.copy()

        # Mutation
        # 变异操作
        child_os = mutate_os(child_os, rng)
        child_os = repair_os(child_os, instance)
        # Repair MA after OS mutation: positions changed, machines may be invalid
        # OS变异后修复MA：位置变化后机器可能不再合法
        child_ma = _repair_ma_for_os(child_os, child_ma, instance, rng)
        child_ma = mutate_ma(child_ma, child_os, instance, rng)

        # Decode (crisp-only, 零 FuzzyNumber 分配)
        mc, wc = decode_crisp(child_os, child_ma, instance)
        f = [mc, wc]

        # Update reference point
        # 更新参考点
        z[0] = min(z[0], f[0])
        z[1] = min(z[1], f[1])

        # Update neighbors
        # 更新邻居解
        #
        # 内联 Tchebycheff（原先是 moead.tchebycheff 调用，60 代共 13.2 万次）：
        #     g = max(w0*|f0-z0|, w1*|f1-z1|)
        # z 在邻居循环内不变，故「新解」的 |f-z| 提到循环外只算一次。
        ns0 = abs(f[0] - z[0])
        ns1 = abs(f[1] - z[1])
        z0 = z[0]
        z1 = z[1]
        for j in neighbors:
            w = w_list[j]
            o = new_obj[j]
            old_g = w[0] * abs(o[0] - z0)
            t = w[1] * abs(o[1] - z1)
            if t > old_g:
                old_g = t
            new_g = w[0] * ns0
            t = w[1] * ns1
            if t > new_g:
                new_g = t
            if new_g < old_g:
                new_pop[j] = (child_os, child_ma)
                new_obj[j] = f
                update_count += 1

    elapsed = time.perf_counter() - start_time
    logger.debug("MOEA/D generation completed in %.4f s, neighbor updates=%d", elapsed, update_count)
    return new_pop, [tuple(o) for o in new_obj], tuple(z)
