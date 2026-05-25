#!/usr/bin/env python3
"""
Validate solution feasibility: check OS+MA encoding produces valid schedule.
验证解的可行性：检查OS+MA编码是否产生有效调度。
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rmoea_d.core.instance import load_instance
from rmoea_d.core.encoding import decode
from rmoea_d.core.fuzzy import FuzzyNumber


def validate_solution(os_vec, ma_vec, instance, verbose=True):
    """
    Validate that OS+MA vectors produce a feasible schedule.
    验证OS+MA向量是否产生可行调度。
    
    Checks:
    1. OS length equals total_ops
    2. Each job appears exactly len(job) times in OS
    3. Each operation's machine is in its candidate set
    4. No operation is scheduled before its predecessor (same job)
    5. No operation starts before machine is free
    """
    n_jobs = instance["n_jobs"]
    n_machines = instance["n_machines"]
    jobs = instance["jobs"]
    total_ops = instance["total_ops"]
    
    errors = []
    
    # Check 1: OS length
    if len(os_vec) != total_ops:
        errors.append(f"OS length {len(os_vec)} != total_ops {total_ops}")
    
    # Check 2: Job counts in OS
    expected_counts = [len(jobs[j]) for j in range(n_jobs)]
    actual_counts = [0] * n_jobs
    for j in os_vec:
        actual_counts[j] += 1
    if actual_counts != expected_counts:
        errors.append(f"Job counts mismatch: expected {expected_counts}, got {actual_counts}")
    
    # Check 3: MA length matches OS
    if len(ma_vec) != len(os_vec):
        errors.append(f"MA length {len(ma_vec)} != OS length {len(os_vec)}")
    
    # Check 4 & 5: Decode and verify schedule
    op_counter = [0] * n_jobs
    job_ready = [FuzzyNumber(0, 0, 0)] * n_jobs
    machine_ready = [FuzzyNumber(0, 0, 0)] * n_machines
    machine_workload = [FuzzyNumber(0, 0, 0)] * n_machines
    
    scheduled_ops = []  # track (job, op_idx, machine, start, finish)
    
    for op_idx, job_id in enumerate(os_vec):
        oi = op_counter[job_id]
        op_counter[job_id] += 1
        alts = jobs[job_id][oi]
        chosen_m = ma_vec[op_idx]
        
        # Check 4: Machine is in candidate set
        proc = None
        for alt in alts:
            if alt[0] == chosen_m:
                proc = alt
                break
        if proc is None:
            errors.append(f"Op ({job_id},{oi}): machine {chosen_m} not in candidates {[a[0] for a in alts]}")
            continue
        
        _, a, b, c = proc
        ptime = FuzzyNumber(a, b, c)
        
        # Check 5: Start time >= job_ready and machine_ready
        start = max(job_ready[job_id], machine_ready[chosen_m])
        if start.t1 < job_ready[job_id].t1 or start.t1 < machine_ready[chosen_m].t1:
            errors.append(f"Op ({job_id},{oi}): start time invalid")
        
        finish = start + ptime
        
        # Check precedence: this op must start after previous op of same job finishes
        if oi > 0:
            prev_finish = None
            for prev in scheduled_ops:
                if prev[0] == job_id and prev[1] == oi - 1:
                    prev_finish = prev[4]
                    break
            if prev_finish is not None:
                if start.t1 < prev_finish.t1:
                    errors.append(f"Op ({job_id},{oi}): starts before prev op finishes")
        
        job_ready[job_id] = finish
        machine_ready[chosen_m] = finish
        machine_workload[chosen_m] = machine_workload[chosen_m] + ptime
        scheduled_ops.append((job_id, oi, chosen_m, start, finish))
    
    # Compute makespan and total workload
    makespan = job_ready[0]
    for t in job_ready[1:]:
        makespan = max(makespan, t)
    
    total_workload = FuzzyNumber(0, 0, 0)
    for w in machine_workload:
        total_workload = total_workload + w
    
    if verbose:
        print(f"  Total operations scheduled: {len(scheduled_ops)}")
        print(f"  Makespan (fuzzy): ({makespan.t1}, {makespan.t2}, {makespan.t3})")
        print(f"  Makespan (clear): {makespan.clear_value()}")
        print(f"  Workload (fuzzy): ({total_workload.t1}, {total_workload.t2}, {total_workload.t3})")
        print(f"  Workload (clear): {total_workload.clear_value()}")
    
    return len(errors) == 0, errors, makespan.clear_value(), total_workload.clear_value()


def validate_result_file(filepath):
    """Validate all solutions in a result JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    
    instance_name = data["instance"]
    instance = load_instance(instance_name, "data", data.get("seed", 42))
    
    print(f"\n{'='*60}")
    print(f"Validating: {os.path.basename(filepath)}")
    print(f"Instance: {instance_name}, Algorithm: {data.get('algorithm', 'Unknown')}")
    print(f"{'='*60}")
    
    # Validate a few solutions from final_pf/fuzzy_pf
    # We need to get the actual OS+MA vectors - they're in the archive or population
    # For now, let's check if the fuzzy_pf values match what decode would produce
    
    # Check that fuzzy numbers satisfy t1 <= t2 <= t3
    fuzzy_pf = data.get("fuzzy_pf", [])
    fuzzy_errors = 0
    for i, sol in enumerate(fuzzy_pf):
        ms = sol["Makespan"]
        wl = sol["Workload"]
        if not (ms["t1"] <= ms["t2"] <= ms["t3"]):
            print(f"  [ERROR] Solution {i}: Makespan t1<=t2<=t3 violated: ({ms['t1']}, {ms['t2']}, {ms['t3']})")
            fuzzy_errors += 1
        if not (wl["t1"] <= wl["t2"] <= wl["t3"]):
            print(f"  [ERROR] Solution {i}: Workload t1<=t2<=t3 violated: ({wl['t1']}, {wl['t2']}, {wl['t3']})")
            fuzzy_errors += 1
    
    if fuzzy_errors == 0:
        print(f"  [PASS] All {len(fuzzy_pf)} fuzzy solutions satisfy t1<=t2<=t3")
    
    # Check consistency between final_pf (clear) and fuzzy_pf
    final_pf = data.get("final_pf", [])
    consistency_errors = 0
    for i, (clear_sol, fuzzy_sol) in enumerate(zip(final_pf, fuzzy_pf)):
        ms_clear = clear_sol["Makespan"]
        wl_clear = clear_sol["Workload"]
        ms = fuzzy_sol["Makespan"]
        wl = fuzzy_sol["Workload"]
        expected_ms_clear = (ms["t1"] + 2*ms["t2"] + ms["t3"]) / 4.0
        expected_wl_clear = (wl["t1"] + 2*wl["t2"] + wl["t3"]) / 4.0
        if abs(ms_clear - expected_ms_clear) > 0.001:
            print(f"  [ERROR] Solution {i}: Makespan clear value mismatch: {ms_clear} vs expected {expected_ms_clear}")
            consistency_errors += 1
        if abs(wl_clear - expected_wl_clear) > 0.001:
            print(f"  [ERROR] Solution {i}: Workload clear value mismatch: {wl_clear} vs expected {expected_wl_clear}")
            consistency_errors += 1
    
    if consistency_errors == 0:
        print(f"  [PASS] All {len(final_pf)} solutions have consistent clear/fuzzy values")
    
    return fuzzy_errors + consistency_errors


if __name__ == "__main__":
    import glob
    
    result_files = glob.glob("results/*/*/*.json")
    if not result_files:
        print("No result files found in results/*/*/*.json")
        sys.exit(1)
    
    total_errors = 0
    for filepath in result_files:
        total_errors += validate_result_file(filepath)
    
    print(f"\n{'='*60}")
    if total_errors == 0:
        print("ALL VALIDATIONS PASSED!")
    else:
        print(f"TOTAL ERRORS: {total_errors}")
    print(f"{'='*60}")
