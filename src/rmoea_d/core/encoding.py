"""
Encoding/decoding: OS+MA vectors to fuzzy objective values.
编码解码模块：OS+MA双层向量 ↔ 模糊目标值。
"""

from .fuzzy import FuzzyNumber, fuzzy_max, ZERO


def decode_crisp(os_vec, ma_vec, instance):
    """
    零分配解码：仅返回 crisp makespan 和 workload，用于进化热路径。
    使用展平为 list-of-lists 的 crisp_times 表，O(1) 数组索引，全程零 Python 对象分配。

    微优化：用 `zip` 同时迭代 os/ma（省掉第二个下标计数器 `op_idx_global` 与
    `ma_vec[i]` 索引），并把 `job_ready`/`machine_ready` 的读写改为「读一次、
    写一次」。数值路径与归约顺序完全不变。
    """
    n_jobs = instance["n_jobs"]
    n_machines = instance["n_machines"]
    crisp_times = instance["crisp_times"]

    op_counter = [0] * n_jobs
    job_ready = [0.0] * n_jobs
    machine_ready = [0.0] * n_machines
    total_workload = 0.0

    for job_id, chosen_m in zip(os_vec, ma_vec):
        oi = op_counter[job_id]
        op_counter[job_id] = oi + 1

        # O(1) 数组索引替代 dict.get，消除哈希计算
        ptime = crisp_times[job_id][oi][chosen_m]

        jr = job_ready[job_id]
        mr = machine_ready[chosen_m]
        start = jr if jr > mr else mr
        finish = start + ptime
        job_ready[job_id] = finish
        machine_ready[chosen_m] = finish
        total_workload += ptime

    makespan = job_ready[0]
    for t in job_ready[1:]:
        if t > makespan:
            makespan = t

    return makespan, total_workload


def decode(os_vec, ma_vec, instance):
    """
    Decode OS and MA vectors into fuzzy makespan and total workload.
    将OS和MA向量解码为模糊最大完工时间和总机器工作负载。

    Returns:
        (makespan, workload, makespan_crisp, workload_crisp)
        返回模糊数和清晰值，供算法不同环节使用。
    """
    makespan, total_workload, mc, wc, _ = _decode_inner(os_vec, ma_vec, instance)
    return makespan, total_workload, mc, wc


def decode_with_schedule(os_vec, ma_vec, instance):
    """
    Decode OS and MA vectors, returning detailed per-operation schedule.
    解码并返回每道工序的详细调度信息，用于甘特图可视化和结果分析。

    Returns:
        (makespan, workload, makespan_crisp, workload_crisp, schedule)
        schedule: list of dicts, each containing:
            - job_id: 工件编号 (0-indexed)
            - op_idx: 工序在该工件中的序号 (0-indexed)
            - machine: 所选机器编号
            - machine_label: 机器标签 (如 "M1")
            - start: 开始时间清晰值
            - finish: 结束时间清晰值
            - processing_time: 加工时间清晰值
            - fuzzy_start: (t1, t2, t3) 模糊开始时间
            - fuzzy_finish: (t1, t2, t3) 模糊结束时间
            - fuzzy_processing: (t1, t2, t3) 模糊加工时间
    """
    makespan, total_workload, mc, wc, schedule = _decode_inner(os_vec, ma_vec, instance)
    return makespan, total_workload, mc, wc, schedule


def _decode_inner(os_vec, ma_vec, instance):
    """
    Internal decoder: computes objectives and records per-operation schedule.
    内部解码器：计算目标值并记录每道工序的调度详情。

    os_vec: list of job indices (length = total_ops)
    ma_vec: list of machine indices (length = total_ops)
    """
    n_jobs = instance["n_jobs"]
    n_machines = instance["n_machines"]
    jobs = instance["jobs"]

    # 跟踪每个工件的下一道工序索引
    op_counter = [0] * n_jobs
    # 每个工件最后一道工序的完成时间（模糊数）
    job_ready = [ZERO] * n_jobs
    # 每台机器的就绪时间（模糊数）
    machine_ready = [ZERO] * n_machines

    total_workload = ZERO
    op_idx = 0
    schedule = []  # 收集每道工序的调度详情

    for pos, job_id in enumerate(os_vec):
        oi = op_counter[job_id]
        op_counter[job_id] += 1
        alts = jobs[job_id][oi]
        chosen_m = ma_vec[op_idx]
        op_idx += 1

        # 查找与所选机器匹配的候选方案
        proc = None
        for alt in alts:
            if alt[0] == chosen_m:
                proc = alt
                break
        if proc is None:
            # 回退：选第一个可用方案
            proc = alts[0]
            chosen_m = proc[0]

        _, a, b, c = proc
        ptime = FuzzyNumber(a, b, c)

        # 开始时间 = max(工件就绪时间, 机器就绪时间)
        start = fuzzy_max(job_ready[job_id], machine_ready[chosen_m])
        finish = start + ptime

        # 记录调度详情
        schedule.append({
            "job_id": job_id,
            "job_label": f"J{job_id + 1}",
            "op_idx": oi,
            "op_label": f"O{job_id + 1},{oi + 1}",
            "machine": chosen_m,
            "machine_label": f"M{chosen_m + 1}",
            "position": pos,  # 在OS向量中的位置
            "start": start.clear_value(),
            "finish": finish.clear_value(),
            "processing_time": ptime.clear_value(),
            "fuzzy_start": start.to_tuple(),
            "fuzzy_finish": finish.to_tuple(),
            "fuzzy_processing": ptime.to_tuple(),
        })

        job_ready[job_id] = finish
        machine_ready[chosen_m] = finish
        total_workload = total_workload + ptime

    # 最大完工时间 = 所有工件完成时间的最大值
    makespan = job_ready[0]
    for t in job_ready[1:]:
        makespan = fuzzy_max(makespan, t)

    return makespan, total_workload, makespan.clear_value(), total_workload.clear_value(), schedule
