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


def compute_hv(front, ref_point=(1.0, 1.0), norm_bounds=None):
    """
    Hypervolume of a 2D minimization front, measured against `ref_point`.

    参数
    ----
    front : iterable of (f1, f2)
    ref_point : (rx, ry)，归一化空间中的参考点（默认 (1,1)）
    norm_bounds : (lo, hi) 或 None
        **参与比较的所有前沿必须共用同一套归一化边界**（lo/hi 为各目标的
        下/上界，长度 2 的序列）。这是让 HV 在不同算法、不同 run 之间可比的
        前提。若传 None，则退回「用本前沿自己的 min/max 归一化」——这种口径
        把每条前沿都拉伸到单位盒，**指标对整体优劣不敏感**，只能衡量前沿
        形状，不能用来比较算法，仅保留以兼容旧调用。

    说明
    ----
    支配区域 = ∪_p [p, ref] 的并集（最小化）。前沿按 f1 升序后 f2 严格递减，
    故 x ∈ [x_i, x_{i+1}) 上的最低 y 即 y_i，于是

        HV = Σ_i (x_{i+1} - x_i) · (ry - y_i),   x_n := rx

    注意 **不含** [0, x_0) 段——那段没有任何解支配。
    """
    if front is None or len(front) == 0:
        return 0.0

    pts = np.asarray(front, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"front must be an Nx2 array, got shape {pts.shape}")

    # ── 归一化 ──
    if norm_bounds is not None:
        lo = np.asarray(norm_bounds[0], dtype=float)
        hi = np.asarray(norm_bounds[1], dtype=float)
    else:
        # 旧口径（不可用于跨算法比较）：本前沿自归一化
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
    rng = hi - lo
    rng = np.where(rng == 0.0, 1.0, rng)
    norm = (pts - lo) / rng

    # ── 裁掉落在参考点之外的点 ──
    rx, ry = float(ref_point[0]), float(ref_point[1])
    norm = norm[(norm[:, 0] < rx) & (norm[:, 1] < ry)]
    if len(norm) == 0:
        return 0.0

    # ── 取非支配子集（按 f1 升序，保留严格更小的 f2） ──
    norm = norm[np.argsort(norm[:, 0], kind="stable")]
    keep_x, keep_y = [], []
    best_y = float("inf")
    for x, y in norm:
        if y < best_y - 1e-15:
            keep_x.append(x)
            keep_y.append(y)
            best_y = y
    if not keep_x:
        return 0.0

    # ── 扫掠积分 ──
    hv = 0.0
    n = len(keep_x)
    for i in range(n):
        x_next = keep_x[i + 1] if i + 1 < n else rx
        width = x_next - keep_x[i]
        height = ry - keep_y[i]
        if width > 0.0 and height > 0.0:
            hv += width * height
    return float(hv)


def estimate_hv_bounds(fronts):
    """从多条前沿的并集估计一套共享归一化边界 (lo, hi)。

    用于离线的多算法比较：所有算法/run 共用该边界，HV 才可直接比大小。
    """
    non_empty = [np.asarray(f, dtype=float) for f in fronts if len(f) > 0]
    if not non_empty:
        return np.array([0.0, 0.0]), np.array([1.0, 1.0])
    allpts = np.vstack(non_empty)
    lo = allpts.min(axis=0)
    hi = allpts.max(axis=0)
    hi = np.where(hi - lo == 0.0, lo + 1.0, hi)
    return lo, hi


def instance_hv_bounds(instance):
    """由实例数据推出**确定性**的 HV 归一化边界（同一实例内所有 run 共用）。

    这样任何一次运行结束后写出的 HV 都是可比的，无需事后统一重算。

    lo = (临界路径下界, 全部工序最短时间之和)
    hi = (全部工序最长时间之和, 同上)   —— 两者都是 makespan/workload 的合法上界
    """
    crisp = instance["crisp_times"]
    n_jobs = instance["n_jobs"]

    ms_lb_per_job = []
    wl_lb = 0.0
    ub = 0.0
    for j in range(n_jobs):
        job_lb = 0.0
        for op in crisp[j]:
            valid = [t for t in op if t is not None]
            if not valid:
                continue
            job_lb += min(valid)
            wl_lb += min(valid)
            ub += max(valid)
        ms_lb_per_job.append(job_lb)

    ms_lb = max(ms_lb_per_job) if ms_lb_per_job else 0.0
    return np.array([ms_lb, wl_lb], dtype=float), np.array([ub, ub], dtype=float)

