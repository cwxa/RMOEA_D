#!/usr/bin/env python3
"""
Validate genetic operators produce feasible OS+MA pairs.
验证遗传算子是否始终产生合法的OS+MA配对。
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rmoea_d.core.instance import load_instance
from rmoea_d.core.encoding import decode
from rmoea_d.core.operators import (
    init_random, init_ls, init_gw, init_mix3,
    pox_crossover, ux_crossover, mutate_os, mutate_ma, repair_os, _repair_ma_for_os
)


def validate_os_counts(os_vec, instance):
    """Check OS has correct job counts."""
    expected = [len(instance["jobs"][j]) for j in range(instance["n_jobs"])]
    actual = [0] * instance["n_jobs"]
    for j in os_vec:
        actual[j] += 1
    return actual == expected


def validate_ma_for_os(os_vec, ma_vec, instance):
    """Check each MA entry is a valid machine for the corresponding operation."""
    op_counter = [0] * instance["n_jobs"]
    for idx, job_id in enumerate(os_vec):
        oi = op_counter[job_id]
        op_counter[job_id] += 1
        alts = instance["jobs"][job_id][oi]
        chosen_m = ma_vec[idx]
        valid_machines = [alt[0] for alt in alts]
        if chosen_m not in valid_machines:
            return False, f"Op ({job_id},{oi}) at idx {idx}: machine {chosen_m} not in {valid_machines}"
    return True, "OK"


def test_operator_roundtrip(instance, rng, n_trials=1000):
    """Test operators through many random generations."""
    errors = []
    
    # Initialize population
    pop = init_mix3(instance, 20, rng)
    
    for trial in range(n_trials):
        # Pick two parents
        i, j = rng.choice(len(pop), 2, replace=False)
        os1, ma1 = pop[i]
        os2, ma2 = pop[j]

        # Crossover (simulating fixed algorithm with _repair_ma_for_os)
        if rng.rand() < 0.9:
            child_os, _ = pox_crossover(os1, os2, rng)
            child_ma, _ = ux_crossover(ma1, ma2, rng)
            child_ma = _repair_ma_for_os(child_os, child_ma, instance, rng)
        else:
            child_os, child_ma = os1.copy(), ma1.copy()

        # Mutation
        child_os = mutate_os(child_os, rng)
        child_os = repair_os(child_os, instance)
        child_ma = _repair_ma_for_os(child_os, child_ma, instance, rng)
        child_ma = mutate_ma(child_ma, child_os, instance, rng)
        
        # Check after mutation
        if not validate_os_counts(child_os, instance):
            errors.append(f"Trial {trial}: After mutation OS counts invalid")
        
        ok, msg = validate_ma_for_os(child_os, child_ma, instance)
        if not ok:
            errors.append(f"Trial {trial}: After mutation invalid: {msg}")
        
        # Try decode
        try:
            decode(child_os, child_ma, instance)
        except Exception as e:
            errors.append(f"Trial {trial}: Decode failed: {e}")
    
    return errors


def main():
    rng = np.random.RandomState(42)
    
    instances = ["Mk01", "Mk02", "Mk03", "Mk04", "Mk05",
                 "Mk06", "Mk07", "Mk08", "Mk09", "Mk10"]
    
    all_errors = []
    for inst_name in instances:
        print(f"\nTesting {inst_name}...")
        instance = load_instance(inst_name, "data", 42)
        errors = test_operator_roundtrip(instance, rng, n_trials=500)
        if errors:
            print(f"  ERRORS: {len(errors)}")
            for e in errors[:5]:
                print(f"    - {e}")
            all_errors.extend(errors)
        else:
            print(f"  PASS: All 500 trials valid")
    
    print(f"\n{'='*60}")
    if all_errors:
        print(f"TOTAL ERRORS: {len(all_errors)}")
    else:
        print("ALL OPERATOR VALIDATIONS PASSED!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
