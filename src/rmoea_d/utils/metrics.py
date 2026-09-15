"""
Metrics: non-dominated sorting and Hypervolume computation.
性能指标模块：非支配排序与超体积(HV)计算。
"""

import numpy as np


def dominates(a, b):
    """True if a dominates b (minimization).
    判断a是否支配b（最小化问题）。"""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def non_dominated_sort(front):
    """
    Return non-dominated (Pareto-optimal) solutions for 2D minimization.
    二维最小化问题的Kung算法：O(N log N)，仅适用于2目标。

    原理：
    1. 按 f1 升序排列（makespan从小到大）
    2. 从左到右扫描，维护已见最小 f2
    3. 若当前 f2 < 已见最小f2，则该点非支配（前序点f1更小但f2更大）
    4. 若当前 f2 >= 已见最小f2，则该点被某个前序点支配

    相比原算法O(N²)的逐一比较，扫描阶段仅O(N)。
    """
    if not front:
        return []
    # 按第一个目标升序排列
    sorted_f = sorted(front, key=lambda x: x[0])
    result = []
    best_f2 = float('inf')
    for p in sorted_f:
        if p[1] < best_f2:
            result.append(p)
            best_f2 = p[1]
    return result


def fast_non_dominated_sort(fronts):
    """NSGA-II style non-dominated sorting. Returns list of ranks.
    NSGA-II风格的快速非支配排序，返回分层前沿。"""
    if not fronts:
        return []
    n = len(fronts)
    domination_count = [0] * n
    dominated_solutions = [[] for _ in range(n)]
    rank = [0] * n
    fronts_result = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(fronts[i], fronts[j]):
                dominated_solutions[i].append(j)
                domination_count[j] += 1
            elif dominates(fronts[j], fronts[i]):
                dominated_solutions[j].append(i)
                domination_count[i] += 1
        if domination_count[i] == 0:
            rank[i] = 0
            fronts_result[0].append(i)

    i = 0
    while len(fronts_result[i]) > 0:
        next_front = []
        for p in fronts_result[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts_result.append(next_front)

    # Remove empty last front
    if len(fronts_result[-1]) == 0:
        fronts_result.pop()

    return fronts_result


def compute_hv(front, ref_point=(1.0, 1.0)):
    """
    Compute hypervolume for 2D minimization front.
    计算2D最小化问题的超体积指标。
    Front is normalized to [0,1] before computation.
    """
    if not front:
        return 0.0
    front = np.array(front, dtype=float)
    # Normalize to [0,1]
    mins = front.min(axis=0)
    maxs = front.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    norm_front = (front - mins) / ranges

    # Sort by first objective ascending
    idx = np.argsort(norm_front[:, 0])
    sorted_f = norm_front[idx]

    hv = 0.0
    prev_x = 0.0
    for i in range(len(sorted_f)):
        x = sorted_f[i, 0]
        y = sorted_f[i, 1]
        # Rectangle width * height contribution
        width = x - prev_x
        height = ref_point[1] - y
        if height > 0 and width > 0:
            hv += width * height
        prev_x = x
    return float(hv)
