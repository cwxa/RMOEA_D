"""
Encoding/decoding: OS+MA vectors to fuzzy objective values.
编码解码模块：OS+MA双层向量 ↔ 模糊目标值。
"""

from .fuzzy import FuzzyNumber, fuzzy_max, ZERO


def decode(os_vec, ma_vec, instance):
    """
    Decode OS and MA vectors into fuzzy makespan and total workload.
    将OS和MA向量解码为模糊最大完工时间和总机器工作负载。

    Returns:
        (makespan, workload, makespan_crisp, workload_crisp)
        返回模糊数和清晰值，供算法不同环节使用。

    os_vec: list of job indices (length = total_ops)
    ma_vec: list of machine indices (length = total_ops)
    """
    n_jobs = instance["n_jobs"]
    n_machines = instance["n_machines"]
    jobs = instance["jobs"]

    # Track next operation index for each job
    # 跟踪每个工件的下一道工序索引
    op_counter = [0] * n_jobs
    # Completion time of last scheduled op for each job (fuzzy)
    # 每个工件最后一道工序的完成时间（模糊数）
    job_ready = [ZERO] * n_jobs
    # Machine ready times (fuzzy)
    # 每台机器的就绪时间（模糊数）
    machine_ready = [ZERO] * n_machines

    total_workload = ZERO
    op_idx = 0

    for job_id in os_vec:
        oi = op_counter[job_id]
        op_counter[job_id] += 1
        alts = jobs[job_id][oi]
        chosen_m = ma_vec[op_idx]
        op_idx += 1

        # Find the alternative matching chosen machine
        # 查找与所选机器匹配的候选方案
        proc = None
        for alt in alts:
            if alt[0] == chosen_m:
                proc = alt
                break
        if proc is None:
            # Fallback: pick first available
            proc = alts[0]
            chosen_m = proc[0]

        _, a, b, c = proc
        ptime = FuzzyNumber(a, b, c)

        # Start time = max(job_ready, machine_ready)
        # 开始时间 = max(工件就绪时间, 机器就绪时间)
        start = fuzzy_max(job_ready[job_id], machine_ready[chosen_m])
        finish = start + ptime

        job_ready[job_id] = finish
        machine_ready[chosen_m] = finish
        total_workload = total_workload + ptime

    # Makespan = max of all job completion times
    # 最大完工时间 = 所有工件完成时间的最大值
    makespan = job_ready[0]
    for t in job_ready[1:]:
        makespan = fuzzy_max(makespan, t)

    return makespan, total_workload, makespan.clear_value(), total_workload.clear_value()
