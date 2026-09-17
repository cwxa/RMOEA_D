"""
Brandimarte FJSP instance loader and fuzzy processing time generator.
Brandimarte柔性作业车间实例加载器与模糊加工时间生成器。
"""

import os
import json
import numpy as np


def parse_fjs(text, seed=42):
    """Parse .fjs format and generate fuzzy processing times.
    解析.fjs格式并生成模糊加工时间。
    对每个确定时间b，生成a, c ~ randint(0, floor(b/2))，模糊时间为(a, b, c)。"""
    rng = np.random.RandomState(seed)
    lines = text.strip().splitlines()
    first = lines[0].split()
    n_jobs = int(first[0])
    n_machines = int(first[1])

    jobs = []
    line_idx = 1
    for _ in range(n_jobs):
        parts = lines[line_idx].split()
        line_idx += 1
        ptr = 0
        n_ops = int(parts[ptr])
        ptr += 1
        operations = []
        for _ in range(n_ops):
            n_alt = int(parts[ptr])
            ptr += 1
            alts = []
            for _ in range(n_alt):
                m = int(parts[ptr]) - 1  # convert to 0-indexed
                ptr += 1
                b = int(parts[ptr])
                ptr += 1
                # Generate fuzzy times: a in [b/2, b], c in [b, 3b/2]
                # 确保三角模糊数满足 t1 <= t2 <= t3，以b为最可能时间
                half = b // 2
                a = rng.randint(half, b + 1) if b > 0 else 0
                c = rng.randint(b, b + half + 1) if b > 0 else 0
                alts.append((m, a, b, c))
            operations.append(alts)
        jobs.append(operations)

    total_ops = sum(len(job) for job in jobs)

    # ── 预计算 crisp 加工时间表，避免热路径重复 (a+2b+c)/4 运算 ──
    # 展平为连续数组：crisp_times[job_idx][op_idx][machine_id] = crisp time (None if invalid)
    # 用 list.index 替代 dict.get，消除哈希开销 (~580k calls per 30 gens)
    crisp_times = []
    for job in jobs:
        job_table = []
        for op in job:
            op_table = [None] * n_machines
            for alt in op:
                m, a, b, c = alt
                op_table[m] = (a + 2.0 * b + c) / 4.0
            job_table.append(op_table)
        crisp_times.append(job_table)

    # ── 预计算每道工序的合法机器集合，供 _repair_ma_for_os O(1) 校验 ──
    # valid_machines[job_idx][op_idx] = set of valid machine IDs
    valid_machines = []
    for job in jobs:
        job_valid = []
        for op in job:
            job_valid.append({alt[0] for alt in op})
        valid_machines.append(job_valid)

    # ── 预计算每道工序的最小 t2（最可能时间）──
    # 供 OS 维度的派工式初始化（init_os_spt / init_os_mwr）O(1) 取用代表加工时间
    min_t2 = []
    for job in jobs:
        min_t2.append([min(alt[2] for alt in op) for op in job])

    return {
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "jobs": jobs,
        "total_ops": total_ops,
        "crisp_times": crisp_times,      # 供 decode_crisp() 零分配解码
        "valid_machines": valid_machines, # 供 _repair_ma_for_os O(1) 校验
        "min_t2": min_t2,                 # 供派工式初始化（OS 维度）
    }


def load_instance(name, data_dir="data", seed=42):
    """Load a Brandimarte instance from data directory.
    从数据目录加载Brandimarte实例。"""
    path = os.path.join(data_dir, f"{name}.fjs")
    if not os.path.exists(path):
        # Try alternative naming
        alt_path = os.path.join(data_dir, f"Brandimarte{name[2:]}.fjs")
        if os.path.exists(alt_path):
            path = alt_path
        else:
            raise FileNotFoundError(f"Instance file not found: {path}")
    with open(path, "r") as f:
        text = f.read()
    return parse_fjs(text, seed)


ALL_INSTANCES = ["Mk01", "Mk02", "Mk03", "Mk04", "Mk05",
                 "Mk06", "Mk07", "Mk08", "Mk09", "Mk10"]


def generate_test_cases(data_dir="data", output_dir="test_cases", seed=42):
    """
    Generate fixed test cases for all Mk instances and save to JSON.
    为所有Mk实例生成固定测试用例并保存为JSON，确保实验可复现。
    """
    os.makedirs(output_dir, exist_ok=True)
    test_cases = {}

    for name in ALL_INSTANCES:
        inst = load_instance(name, data_dir, seed)
        test_cases[name] = inst
        # Save individual file
        filepath = os.path.join(output_dir, f"{name}_seed{seed}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(inst, f, indent=2, ensure_ascii=False)

    # Save combined file
    combined_path = os.path.join(output_dir, f"all_instances_seed{seed}.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)

    print(f"Test cases generated: {len(test_cases)} instances saved to {output_dir}/")
    return test_cases


def load_test_case(name, test_case_dir="test_cases", seed=42):
    """Load a pre-generated test case from JSON.
    从JSON加载预生成的测试用例。"""
    filepath = os.path.join(test_case_dir, f"{name}_seed{seed}.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Test case not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
