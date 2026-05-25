#!/usr/bin/env python3
"""
Benchmark script: Compare RMOEA/D vs MOEA/D on Brandimarte instances.
对比实验脚本：在Brandimarte实例上对比RMOEA/D与MOEA/D性能。

对比指标：
- HV (Hypervolume): 超体积，越大越好
- PF size: Pareto前沿大小
- Runtime: 运行时间
- Convergence trace: 收敛曲线
"""

import argparse
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rmoea_d.algorithm import RMOEAD
from rmoea_d.moead_baseline import MOEADBaseline
from rmoea_d.utils.logger_setup import setup_logging
from rmoea_d.core.instance import ALL_INSTANCES

import logging

logger = logging.getLogger(__name__)


def run_comparison(instance, n_pop, max_gen, crossover_rate, seed, fixed_T, data_dir):
    """
    Run both algorithms on the same instance with identical parameters.
    在相同实例和参数下运行两种算法。
    """
    print("\n" + "=" * 70)
    print(f"Instance: {instance} | Np={n_pop} | Gen={max_gen} | Seed={seed}")
    print("=" * 70)

    # Run RMOEA/D
    print("\n[1/2] Running RMOEA/D (with Q-learning)...")
    solver_r = RMOEAD(
        instance_name=instance,
        n_pop=n_pop,
        max_gen=max_gen,
        crossover_rate=crossover_rate,
        seed=seed,
        data_dir=data_dir,
    )
    results_r = solver_r.solve()

    # Run MOEA/D baseline
    print("\n[2/2] Running MOEA/D baseline (fixed T=%d)..." % fixed_T)
    solver_m = MOEADBaseline(
        instance_name=instance,
        n_pop=n_pop,
        max_gen=max_gen,
        crossover_rate=crossover_rate,
        fixed_T=fixed_T,
        seed=seed,
        data_dir=data_dir,
    )
    results_m = solver_m.solve()

    # Extract final PF objective statistics (crisp values)
    # 提取最终Pareto前沿的清晰值统计
    pf_r = results_r["final_pf"]
    pf_m = results_m["final_pf"]

    def _pf_stats(pf):
        if not pf:
            return {}
        import numpy as np
        # Support both old list format and new dict format with names
        # 支持旧的列表格式和新的带名称字典格式
        if isinstance(pf[0], dict):
            makespan_vals = [p["Makespan"] for p in pf]
            workload_vals = [p["Workload"] for p in pf]
        else:
            arr = np.array(pf)
            makespan_vals = arr[:, 0]
            workload_vals = arr[:, 1]
        return {
            "best_makespan": float(np.min(makespan_vals)),
            "best_workload": float(np.min(workload_vals)),
            "avg_makespan": float(np.mean(makespan_vals)),
            "avg_workload": float(np.mean(workload_vals)),
            "worst_makespan": float(np.max(makespan_vals)),
            "worst_workload": float(np.max(workload_vals)),
        }

    stats_r = _pf_stats(pf_r)
    stats_m = _pf_stats(pf_m)

    # Extract fuzzy PF statistics (triangular fuzzy numbers)
    # 提取模糊Pareto前沿的三角模糊数统计
    fuzzy_pf_r = results_r.get("fuzzy_pf", [])
    fuzzy_pf_m = results_m.get("fuzzy_pf", [])

    def _fuzzy_stats(fuzzy_pf):
        if not fuzzy_pf:
            return {}
        import numpy as np
        from rmoea_d.core.fuzzy import fuzzy_tuple_clear_value
        # fuzzy_pf format: [{"Makespan": {"t1":..,"t2":..,"t3":..}, "Workload": {...}}, ...]
        ms_t1 = [p["Makespan"]["t1"] for p in fuzzy_pf]
        ms_t2 = [p["Makespan"]["t2"] for p in fuzzy_pf]
        ms_t3 = [p["Makespan"]["t3"] for p in fuzzy_pf]
        wl_t1 = [p["Workload"]["t1"] for p in fuzzy_pf]
        wl_t2 = [p["Workload"]["t2"] for p in fuzzy_pf]
        wl_t3 = [p["Workload"]["t3"] for p in fuzzy_pf]
        ms_clear = [fuzzy_tuple_clear_value((t1, t2, t3)) for t1, t2, t3 in zip(ms_t1, ms_t2, ms_t3)]
        wl_clear = [fuzzy_tuple_clear_value((t1, t2, t3)) for t1, t2, t3 in zip(wl_t1, wl_t2, wl_t3)]
        return {
            "fuzzy_makespan": {
                "best": {"t1": float(np.min(ms_t1)), "t2": float(np.min(ms_t2)), "t3": float(np.min(ms_t3))},
                "avg": {"t1": float(np.mean(ms_t1)), "t2": float(np.mean(ms_t2)), "t3": float(np.mean(ms_t3))},
                "worst": {"t1": float(np.max(ms_t1)), "t2": float(np.max(ms_t2)), "t3": float(np.max(ms_t3))},
                "best_clear": float(np.min(ms_clear)),
                "avg_clear": float(np.mean(ms_clear)),
                "worst_clear": float(np.max(ms_clear)),
            },
            "fuzzy_workload": {
                "best": {"t1": float(np.min(wl_t1)), "t2": float(np.min(wl_t2)), "t3": float(np.min(wl_t3))},
                "avg": {"t1": float(np.mean(wl_t1)), "t2": float(np.mean(wl_t2)), "t3": float(np.mean(wl_t3))},
                "worst": {"t1": float(np.max(wl_t1)), "t2": float(np.max(wl_t2)), "t3": float(np.max(wl_t3))},
                "best_clear": float(np.min(wl_clear)),
                "avg_clear": float(np.mean(wl_clear)),
                "worst_clear": float(np.max(wl_clear)),
            },
        }

    fuzzy_stats_r = _fuzzy_stats(fuzzy_pf_r)
    fuzzy_stats_m = _fuzzy_stats(fuzzy_pf_m)

    # Compare results
    comparison = {
        "instance": instance,
        "n_pop": n_pop,
        "max_gen": max_gen,
        "seed": seed,
        "rmoea_d": {
            "final_hv": results_r["final_hv"],
            "pf_size": len(pf_r),
            "total_time": results_r["total_time"],
            "q_table": results_r.get("q_table", []),
            **stats_r,
            **fuzzy_stats_r,
        },
        "moea_d": {
            "final_hv": results_m["final_hv"],
            "pf_size": len(pf_m),
            "total_time": results_m["total_time"],
            "fixed_T": fixed_T,
            **stats_m,
            **fuzzy_stats_m,
        },
    }

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Metric':<30} {'RMOEA/D':<22} {'MOEA/D':<22} {'Improvement':<15}")
    print("-" * 80)

    hv_r = results_r["final_hv"]
    hv_m = results_m["final_hv"]
    hv_imp = ((hv_r - hv_m) / hv_m * 100) if hv_m > 0 else 0
    print(f"{'Hypervolume (HV)':<30} {hv_r:<22.6f} {hv_m:<22.6f} {hv_imp:>+14.2f}%")

    pf_size_r = len(pf_r)
    pf_size_m = len(pf_m)
    pf_imp = ((pf_size_r - pf_size_m) / pf_size_m * 100) if pf_size_m > 0 else 0
    print(f"{'PF Size':<30} {pf_size_r:<22} {pf_size_m:<22} {pf_imp:>+14.2f}%")

    time_r = results_r["total_time"]
    time_m = results_m["total_time"]
    time_imp = ((time_r - time_m) / time_m * 100) if time_m > 0 else 0
    print(f"{'Runtime (s)':<30} {time_r:<22.4f} {time_m:<22.4f} {time_imp:>+14.2f}%")
    print("-" * 80)

    # Per-objective comparison
    print(f"{'--- Makespan (minimize) ---':<80}")
    if stats_r and stats_m:
        bm_r, bm_m = stats_r["best_makespan"], stats_m["best_makespan"]
        bm_imp = ((bm_m - bm_r) / bm_m * 100) if bm_m > 0 else 0  # lower is better
        print(f"{'Best Makespan':<30} {bm_r:<22.4f} {bm_m:<22.4f} {bm_imp:>+14.2f}%")

        am_r, am_m = stats_r["avg_makespan"], stats_m["avg_makespan"]
        am_imp = ((am_m - am_r) / am_m * 100) if am_m > 0 else 0
        print(f"{'Avg Makespan':<30} {am_r:<22.4f} {am_m:<22.4f} {am_imp:>+14.2f}%")

        wm_r, wm_m = stats_r["worst_makespan"], stats_m["worst_makespan"]
        wm_imp = ((wm_m - wm_r) / wm_m * 100) if wm_m > 0 else 0
        print(f"{'Worst Makespan':<30} {wm_r:<22.4f} {wm_m:<22.4f} {wm_imp:>+14.2f}%")

    print(f"{'--- Workload (minimize) ---':<80}")
    if stats_r and stats_m:
        bw_r, bw_m = stats_r["best_workload"], stats_m["best_workload"]
        bw_imp = ((bw_m - bw_r) / bw_m * 100) if bw_m > 0 else 0
        print(f"{'Best Workload':<30} {bw_r:<22.4f} {bw_m:<22.4f} {bw_imp:>+14.2f}%")

        aw_r, aw_m = stats_r["avg_workload"], stats_m["avg_workload"]
        aw_imp = ((aw_m - aw_r) / aw_m * 100) if aw_m > 0 else 0
        print(f"{'Avg Workload':<30} {aw_r:<22.4f} {aw_m:<22.4f} {aw_imp:>+14.2f}%")

        ww_r, ww_m = stats_r["worst_workload"], stats_m["worst_workload"]
        ww_imp = ((ww_m - ww_r) / ww_m * 100) if ww_m > 0 else 0
        print(f"{'Worst Workload':<30} {ww_r:<22.4f} {ww_m:<22.4f} {ww_imp:>+14.2f}%")

    # Fuzzy objective comparison (triangular fuzzy numbers)
    # 模糊目标对比（三角模糊数）
    print(f"{'--- Fuzzy Makespan (t1,t2,t3) ---':<80}")
    if fuzzy_stats_r and fuzzy_stats_m:
        fm_r = fuzzy_stats_r["fuzzy_makespan"]
        fm_m = fuzzy_stats_m["fuzzy_makespan"]
        print(f"{'Best Makespan (t1,t2,t3)':<30} ({fm_r['best']['t1']:.1f},{fm_r['best']['t2']:.1f},{fm_r['best']['t3']:.1f}){'':>6} ({fm_m['best']['t1']:.1f},{fm_m['best']['t2']:.1f},{fm_m['best']['t3']:.1f})")
        print(f"{'Avg Makespan (t1,t2,t3)':<30} ({fm_r['avg']['t1']:.1f},{fm_r['avg']['t2']:.1f},{fm_r['avg']['t3']:.1f}){'':>6} ({fm_m['avg']['t1']:.1f},{fm_m['avg']['t2']:.1f},{fm_m['avg']['t3']:.1f})")
        print(f"{'Worst Makespan (t1,t2,t3)':<30} ({fm_r['worst']['t1']:.1f},{fm_r['worst']['t2']:.1f},{fm_r['worst']['t3']:.1f}){'':>6} ({fm_m['worst']['t1']:.1f},{fm_m['worst']['t2']:.1f},{fm_m['worst']['t3']:.1f})")

    print(f"{'--- Fuzzy Workload (t1,t2,t3) ---':<80}")
    if fuzzy_stats_r and fuzzy_stats_m:
        fw_r = fuzzy_stats_r["fuzzy_workload"]
        fw_m = fuzzy_stats_m["fuzzy_workload"]
        print(f"{'Best Workload (t1,t2,t3)':<30} ({fw_r['best']['t1']:.1f},{fw_r['best']['t2']:.1f},{fw_r['best']['t3']:.1f}){'':>6} ({fw_m['best']['t1']:.1f},{fw_m['best']['t2']:.1f},{fw_m['best']['t3']:.1f})")
        print(f"{'Avg Workload (t1,t2,t3)':<30} ({fw_r['avg']['t1']:.1f},{fw_r['avg']['t2']:.1f},{fw_r['avg']['t3']:.1f}){'':>6} ({fw_m['avg']['t1']:.1f},{fw_m['avg']['t2']:.1f},{fw_m['avg']['t3']:.1f})")
        print(f"{'Worst Workload (t1,t2,t3)':<30} ({fw_r['worst']['t1']:.1f},{fw_r['worst']['t2']:.1f},{fw_r['worst']['t3']:.1f}){'':>6} ({fw_m['worst']['t1']:.1f},{fw_m['worst']['t2']:.1f},{fw_m['worst']['t3']:.1f})")

    print("=" * 80)

    if hv_r > hv_m:
        print("Winner: RMOEA/D (higher HV)")
    elif hv_r < hv_m:
        print("Winner: MOEA/D (higher HV)")
    else:
        print("Tie: Equal HV")
    print("=" * 80)

    return comparison, results_r, results_m


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: RMOEA/D vs MOEA/D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--instances", type=str, nargs="+", default=["Mk01"],
        help="Instance names to benchmark (default: Mk01)"
    )
    parser.add_argument(
        "--n_pop", type=int, default=100,
        help="Population size (default: 100)"
    )
    parser.add_argument(
        "--max_gen", type=int, default=200,
        help="Maximum generations (default: 200)"
    )
    parser.add_argument(
        "--crossover_rate", type=float, default=0.8,
        help="Crossover rate (default: 0.8)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--fixed_T", type=int, default=10,
        help="Fixed T for MOEA/D baseline (default: 10)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Data directory (default: data)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Output directory (default: results)"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs",
        help="Log directory (default: logs)"
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(log_dir=args.log_dir)

    all_comparisons = []

    for instance in args.instances:
        comparison, results_r, results_m = run_comparison(
            instance=instance,
            n_pop=args.n_pop,
            max_gen=args.max_gen,
            crossover_rate=args.crossover_rate,
            seed=args.seed,
            fixed_T=args.fixed_T,
            data_dir=args.data_dir,
        )
        all_comparisons.append(comparison)

        # Save individual results with grouped structure
        ts = time.strftime("%Y%m%d_%H%M%S")

        r_dir = os.path.join(args.output_dir, instance, "RMOEA_D")
        os.makedirs(r_dir, exist_ok=True)
        r_path = os.path.join(r_dir, f"{instance}_RMOEA_D_Np{args.n_pop}_G{args.max_gen}_{ts}.json")
        with open(r_path, "w", encoding="utf-8") as f:
            json.dump(results_r, f, indent=2, ensure_ascii=False)

        m_dir = os.path.join(args.output_dir, instance, "MOEA_D")
        os.makedirs(m_dir, exist_ok=True)
        m_path = os.path.join(m_dir, f"{instance}_MOEA_D_Np{args.n_pop}_G{args.max_gen}_T{args.fixed_T}_{ts}.json")
        with open(m_path, "w", encoding="utf-8") as f:
            json.dump(results_m, f, indent=2, ensure_ascii=False)

    # Save summary to each instance directory (one summary per instance)
    # 将summary保存到各实例目录下
    for idx, inst in enumerate(args.instances):
        inst_summary_path = os.path.join(args.output_dir, inst, f"benchmark_summary_{inst}_{ts}.json")
        with open(inst_summary_path, "w", encoding="utf-8") as f:
            json.dump(all_comparisons[idx], f, indent=2, ensure_ascii=False)

    print(f"\nAll results saved to: {args.output_dir}/")
    print(f"  Structure: {args.output_dir}/<instance>/<algorithm>/result.json")
    print(f"  Summary saved to: {args.output_dir}/<instance>/benchmark_summary_<instance>_<timestamp>.json")


if __name__ == "__main__":
    main()
