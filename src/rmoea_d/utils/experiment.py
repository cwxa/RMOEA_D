#!/usr/bin/env python3
"""
统一实验脚本：单次运行同时产出 Benchmark + Ablation 的全部数据。
每个 seed 跑 4 种算法变体，一次运行两份报告。

算法变体:
  - rmoea_d  (Full RMOEA/D): Q-PAS + RVNS  → benchmark RMOEA/D + ablation "full"
  - moea_d   (MOEA/D Baseline): 纯 MOEA/D  → benchmark MOEA/D  + ablation "moead"
  - qpas_only (Q-PAS Only):     仅 Q-PAS    → ablation 专用
  - rvns_only (RVNS Only):      仅 RVNS     → ablation 专用

输出结构:
  results/experiment/{inst}/run_{seed}_{exp_id}.json   # per-run, 4 algos
  results/experiment/aggregate_{exp_id}.json            # 聚合统计
  results/schedules/{inst}/gantt_*.png                  # 甘特图 (不变)
"""

import argparse
import json
import os
import sys
import time
import traceback
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

# ── 禁止 BLAS/MKL 内部多线程 ──
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rmoea_d.algorithm import RMOEAD
from rmoea_d.moead_baseline import MOEADBaseline
from rmoea_d.utils.logger_setup import setup_logging

import logging
logger = logging.getLogger(__name__)

# ── 算法变体定义 ──
BENCHMARK_ALGOS = ["rmoea_d", "moea_d"]           # benchmark 对比
ABLATION_ALGOS  = ["rmoea_d", "qpas_only", "rvns_only", "moea_d"]  # 消融对比
# rmoea_d 在消融中别名 "full"
ALGO_ALIAS = {"rmoea_d": "full"}


# ═══════════════════════════════════════════════════════════
# 核心：单次运行 4 种算法变体
# ═══════════════════════════════════════════════════════════

def _extract_run(r, algo, instance, seed, n_pop, max_gen, elapsed, ql_params=None):
    """从 solver.solve() 返回的原始结果中提取标准化字段。"""
    pf = r.get("final_pf", [])
    if pf:
        if isinstance(pf[0], dict):
            ms = [p["Makespan"] for p in pf]
            wl = [p["Workload"] for p in pf]
        else:
            arr = np.array(pf)
            ms, wl = arr[:, 0].tolist(), arr[:, 1].tolist()
        best_ms = float(np.min(ms))
        best_wl = float(np.min(wl))
        avg_ms = float(np.mean(ms))
        avg_wl = float(np.mean(wl))
    else:
        best_ms = best_wl = avg_ms = avg_wl = 0.0

    # 模糊统计
    fuzzy_pf = r.get("fuzzy_pf", [])
    fuzzy_stats = {}
    if fuzzy_pf:
        from rmoea_d.core.fuzzy import fuzzy_tuple_clear_value
        fm_t1 = [p["Makespan"]["t1"] for p in fuzzy_pf]
        fm_t2 = [p["Makespan"]["t2"] for p in fuzzy_pf]
        fm_t3 = [p["Makespan"]["t3"] for p in fuzzy_pf]
        fw_t1 = [p["Workload"]["t1"] for p in fuzzy_pf]
        fw_t2 = [p["Workload"]["t2"] for p in fuzzy_pf]
        fw_t3 = [p["Workload"]["t3"] for p in fuzzy_pf]
        ms_clear = [fuzzy_tuple_clear_value((t1, t2, t3)) for t1, t2, t3 in zip(fm_t1, fm_t2, fm_t3)]
        wl_clear = [fuzzy_tuple_clear_value((t1, t2, t3)) for t1, t2, t3 in zip(fw_t1, fw_t2, fw_t3)]
        fuzzy_stats = {
            "fuzzy_makespan": {
                "best": {"t1": float(np.min(fm_t1)), "t2": float(np.min(fm_t2)), "t3": float(np.min(fm_t3))},
                "best_clear": float(np.min(ms_clear)),
            },
            "fuzzy_workload": {
                "best": {"t1": float(np.min(fw_t1)), "t2": float(np.min(fw_t2)), "t3": float(np.min(fw_t3))},
                "best_clear": float(np.min(wl_clear)),
            },
        }

    # 收敛历史
    history = []
    for h in r.get("history", []):
        history.append({
            "gen": h.get("gen", 0),
            "hv": h.get("hv", 0),
            "pf_size": h.get("pf_size", 0),
            "best_makespan": h.get("best_makespan", 0),
            "best_workload": h.get("best_workload", 0),
        })

    return {
        "algorithm": algo,
        "instance": instance,
        "seed": seed,
        "n_pop": n_pop,
        "max_gen": max_gen,
        "total_time": elapsed,
        "final_hv": r.get("final_hv", 0),
        "pf_size": len(pf),
        "best_makespan": best_ms,
        "best_workload": best_wl,
        "avg_makespan": avg_ms,
        "avg_workload": avg_wl,
        "final_pf": pf,
        "fuzzy_pf": fuzzy_pf,
        "schedules": r.get("schedules", []),
        "q_table": r.get("q_table", []),
        "history": history,
        "parameters": ql_params or {},
        **fuzzy_stats,
    }


def run_single(instance, n_pop, max_gen, seed, crossover_rate, fixed_T,
               data_dir, ql_params=None, rvns_ls_trials=1):
    """一次运行产出 4 种算法变体——benchmark + ablation 数据一举拿下。
    
    Returns:
        dict with keys: rmoea_d, moea_d, qpas_only, rvns_only
    """
    if ql_params is None:
        ql_params = {"alpha": 0.4, "gamma": 0.6, "epsilon": 0.8, "actions": [5, 10, 15, 20]}

    results = {}

    # 1. Full RMOEA/D (Q-PAS + RVNS) → benchmark rmoea + ablation "full"
    t0 = time.perf_counter()
    solver = RMOEAD(
        instance_name=instance, n_pop=n_pop, max_gen=max_gen,
        crossover_rate=crossover_rate, seed=seed, data_dir=data_dir,
        ql_alpha=ql_params["alpha"], ql_gamma=ql_params["gamma"],
        ql_epsilon=ql_params["epsilon"], ql_actions=ql_params["actions"],
        enable_rvns=True, rvns_ls_trials=rvns_ls_trials)
    r = solver.solve()
    results["rmoea_d"] = _extract_run(r, "rmoea_d", instance, seed, n_pop, max_gen,
                                       time.perf_counter() - t0, ql_params)

    # 2. Q-PAS Only (no RVNS)
    t0 = time.perf_counter()
    solver = RMOEAD(
        instance_name=instance, n_pop=n_pop, max_gen=max_gen,
        crossover_rate=crossover_rate, seed=seed, data_dir=data_dir,
        ql_alpha=ql_params["alpha"], ql_gamma=ql_params["gamma"],
        ql_epsilon=ql_params["epsilon"], ql_actions=ql_params["actions"],
        enable_rvns=False)
    r = solver.solve()
    results["qpas_only"] = _extract_run(r, "qpas_only", instance, seed, n_pop, max_gen,
                                         time.perf_counter() - t0, ql_params)

    # 3. RVNS Only (fixed T, no Q-PAS)
    t0 = time.perf_counter()
    solver = RMOEAD(
        instance_name=instance, n_pop=n_pop, max_gen=max_gen,
        crossover_rate=crossover_rate, seed=seed, data_dir=data_dir,
        fixed_T=fixed_T, enable_rvns=True, rvns_ls_trials=rvns_ls_trials)
    r = solver.solve()
    results["rvns_only"] = _extract_run(r, "rvns_only", instance, seed, n_pop, max_gen,
                                         time.perf_counter() - t0)

    # 4. MOEA/D Baseline (pure MOEA/D)
    t0 = time.perf_counter()
    solver = MOEADBaseline(
        instance_name=instance, n_pop=n_pop, max_gen=max_gen,
        crossover_rate=crossover_rate, fixed_T=fixed_T, seed=seed,
        data_dir=data_dir)
    r = solver.solve()
    results["moea_d"] = _extract_run(r, "moea_d", instance, seed, n_pop, max_gen,
                                      time.perf_counter() - t0)

    return results


# ═══════════════════════════════════════════════════════════
# 并行 Worker
# ═══════════════════════════════════════════════════════════

def _worker(args_tuple):
    """进程池 worker：运行单次统一实验 (picklable 顶层函数)。"""
    (instance, n_pop, max_gen, seed, crossover_rate, fixed_T,
     data_dir, ql_alpha, ql_gamma, ql_epsilon, ql_actions,
     rvns_ls_trials) = args_tuple
    ql_params = {
        "alpha": ql_alpha, "gamma": ql_gamma,
        "epsilon": ql_epsilon, "actions": list(ql_actions),
    }
    return run_single(instance, n_pop, max_gen, seed, crossover_rate,
                      fixed_T, data_dir, ql_params,
                      rvns_ls_trials=rvns_ls_trials)


# ═══════════════════════════════════════════════════════════
# 多轮并行运行
# ═══════════════════════════════════════════════════════════

def run_experiment(instances, n_pop, max_gen, n_runs, base_seed,
                   crossover_rate, fixed_T, data_dir, output_dir,
                   ql_alpha=0.4, ql_gamma=0.6, ql_epsilon=0.8,
                   ql_actions=None, n_workers=None, exp_id=None,
                   timeout_per_task=None, rvns_ls_trials=1):
    """统一实验：并行运行所有 instance × seed 任务，每个任务产出 4 种算法。
    
    Returns:
        all_runs:  {inst: [run_0, run_1, ...]}  每个 run 含 4 种算法
        agg:       benchmark 和 ablation 两份聚合统计
    """
    if ql_actions is None:
        ql_actions = [5, 10, 15, 20]
    if exp_id is None:
        exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if n_workers is None:
        parent_workers = int(os.environ.get("RMOEA_PARENT_WORKERS", "0"))
        total_cpus = os.cpu_count() or 4
        if parent_workers > 0:
            n_workers = max(1, min(n_runs, total_cpus // parent_workers))
        else:
            n_workers = min(total_cpus, n_runs)

    # ── 生成任务 ──
    tasks = []
    for inst in instances:
        for run_idx in range(n_runs):
            seed = base_seed + run_idx
            tasks.append((inst, n_pop, max_gen, seed, crossover_rate, fixed_T,
                          data_dir, ql_alpha, ql_gamma, ql_epsilon, tuple(ql_actions),
                          rvns_ls_trials))

    logger.info("Unified experiment: %d instances x %d runs x 4 algos = %d tasks → %d workers",
                len(instances), n_runs, len(tasks), n_workers)

    # ── 并行执行 ──
    all_runs = {inst: [] for inst in instances}
    completed = 0
    timed_out = 0
    t0 = time.time()
    milestone = max(1, len(tasks) // 5)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            inst = task[0]
            seed = task[3]
            try:
                run_data = future.result(timeout=timeout_per_task)
                run_data["_instance"] = inst
                run_data["_seed"] = seed
                all_runs[inst].append(run_data)
                completed += 1
            except FutureTimeoutError:
                timed_out += 1
                logger.warning("[%d/%d] Timeout: %s seed=%d > %.0fs",
                               completed + timed_out, len(tasks), inst, seed, timeout_per_task)
                future.cancel()
                continue
            except Exception as e:
                timed_out += 1
                logger.error("[%d/%d] Error: %s seed=%d: %s",
                             completed + timed_out, len(tasks), inst, seed, e)
                continue

            elapsed = time.time() - t0
            if completed % milestone == 0 or completed == len(tasks):
                logger.info("Experiment %d/%d tasks [%.0fs]", completed, len(tasks), elapsed)

            # ── 每完成一个实例的所有 run 就保存 per-run JSON ──
            if completed % n_runs == 0:
                _save_per_run(all_runs, output_dir, exp_id)

    if timed_out > 0:
        logger.warning("%d/%d tasks timed out or errored", timed_out, len(tasks))

    # ── 用「参考集」口径统一重算 HV ──
    _retune_hv_reference_set(all_runs)

    # ── 最终保存 ──
    _save_per_run(all_runs, output_dir, exp_id)

    # ── 聚合统计 ──
    agg = _aggregate(all_runs, instances, n_pop, max_gen, n_runs, base_seed)
    _save_aggregate(agg, output_dir, exp_id)

    return all_runs, agg


def _retune_hv_reference_set(all_runs, ref_point=(1.02, 1.02)):
    """把所有 run 的 `final_hv` 统一重算为「参考集归一化」口径。

    为什么必须重算：
      * `compute_hv` 在未给 norm_bounds 时用「每条前沿自己的 min/max」归一化，
        这会把任意前沿拉伸到单位盒，指标对整体优劣不敏感（只反映前沿形状），
        无法用于比较算法；
      * 求解器内联的 `instance_hv_bounds` 虽然固定可比，但 makespan 上界取
        「全部工序最长时间之和」，远松于真实取值，HV 分辨率被压扁（实测
        四个变体差异仅 ~0.002）。
    这里改用「同一实例下所有变体、所有 run 的前沿并集」作为归一化盒——
    这是多算法比较 HV 的标准做法，同实例内 HV 严格可比。
    """
    from rmoea_d.utils.metrics import compute_hv, estimate_hv_bounds

    for inst, runs in all_runs.items():
        fronts = []
        for rd in runs:
            for algo in ABLATION_ALGOS:
                r = rd.get(algo)
                if not r:
                    continue
                pf = r.get("final_pf") or []
                if not pf:
                    continue
                if isinstance(pf[0], dict):
                    fronts.append([[p["Makespan"], p["Workload"]] for p in pf])
                else:
                    fronts.append([[p[0], p[1]] for p in pf])
        if not fronts:
            continue

        lo, hi = estimate_hv_bounds(fronts)
        for rd in runs:
            for algo in ABLATION_ALGOS:
                r = rd.get(algo)
                if not r:
                    continue
                pf = r.get("final_pf") or []
                if not pf:
                    r["final_hv"] = 0.0
                    continue
                if isinstance(pf[0], dict):
                    pts = [[p["Makespan"], p["Workload"]] for p in pf]
                else:
                    pts = [[p[0], p[1]] for p in pf]
                r["final_hv"] = compute_hv(pts, ref_point=ref_point,
                                           norm_bounds=(lo, hi))
                r["hv_norm_bounds"] = [lo.tolist(), hi.tolist()]
                r["hv_ref_point"] = list(ref_point)
                r["hv_definition"] = ("reference-set normalization "
                                      "(union of all runs of this instance)")
        logger.info("HV re-tuned [%s]: lo=%s hi=%s (reference-set normalization)",
                    inst, lo.round(2).tolist(), hi.round(2).tolist())


# ═══════════════════════════════════════════════════════════
# 数据保存
# ═══════════════════════════════════════════════════════════

def _save_per_run(all_runs, output_dir, exp_id):
    """保存每个 run 的完整 JSON（含 4 种算法 + schedules）。"""
    import numpy as np

    def _to_json(o):
        """递归转换 numpy 类型为原生 Python 类型，确保 JSON 可序列化。"""
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: _to_json(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_json(v) for v in o]
        return o

    exp_dir = os.path.join(output_dir, "experiment")
    os.makedirs(exp_dir, exist_ok=True)
    for inst, runs in all_runs.items():
        inst_dir = os.path.join(exp_dir, inst)
        os.makedirs(inst_dir, exist_ok=True)
        for run_data in runs:
            seed = run_data.get("_seed", 0)
            path = os.path.join(inst_dir, f"run_{seed}_{exp_id}.json")
            # 去掉内部标记字段
            clean = {k: v for k, v in run_data.items() if not k.startswith("_")}
            clean["exp_id"] = exp_id
            clean["instance"] = inst
            clean = _to_json(clean)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(clean, f, indent=2, ensure_ascii=False)


def _aggregate(all_runs, instances, n_pop, max_gen, n_runs, base_seed):
    """从 all_runs 聚合出 benchmark 和 ablation 两份视图。"""
    def _stats(values):
        arr = np.array(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "_all": arr.tolist(),
        }

    benchmark_agg = {}
    ablation_agg = {}

    for inst in instances:
        runs = all_runs.get(inst, [])
        if not runs:
            continue

        # ── Benchmark 视图: rmoea_d vs moea_d ──
        bm = {"instance": inst, "n_pop": n_pop, "max_gen": max_gen,
              "n_runs": len(runs), "base_seed": base_seed}
        for algo in BENCHMARK_ALGOS:
            hvs = [r[algo]["final_hv"] for r in runs]
            times = [r[algo]["total_time"] for r in runs]
            bm_ms = [r[algo]["best_makespan"] for r in runs]
            bm_wl = [r[algo]["best_workload"] for r in runs]
            pf_sizes = [r[algo]["pf_size"] for r in runs]
            bm[algo] = {
                "hv_mean": float(np.mean(hvs)), "hv_std": float(np.std(hvs, ddof=1)) if len(hvs) > 1 else 0.0,
                "time_mean": float(np.mean(times)), "time_std": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
                "best_makespan_mean": float(np.mean(bm_ms)), "best_makespan_std": float(np.std(bm_ms, ddof=1)) if len(bm_ms) > 1 else 0.0,
                "best_workload_mean": float(np.mean(bm_wl)), "best_workload_std": float(np.std(bm_wl, ddof=1)) if len(bm_wl) > 1 else 0.0,
                "pf_size_mean": float(np.mean(pf_sizes)), "pf_size_std": float(np.std(pf_sizes, ddof=1)) if len(pf_sizes) > 1 else 0.0,
                "_hv_all": hvs, "_bm_all": bm_ms, "_bw_all": bm_wl,
            }
            # 模糊统计 (TFN 三参数 + clear value)
            fm_t1 = [r[algo].get("fuzzy_makespan", {}).get("best", {}).get("t1", 0) for r in runs]
            fm_t2 = [r[algo].get("fuzzy_makespan", {}).get("best", {}).get("t2", 0) for r in runs]
            fm_t3 = [r[algo].get("fuzzy_makespan", {}).get("best", {}).get("t3", 0) for r in runs]
            fm_clear = [r[algo].get("fuzzy_makespan", {}).get("best_clear", 0) for r in runs]
            fw_t1 = [r[algo].get("fuzzy_workload", {}).get("best", {}).get("t1", 0) for r in runs]
            fw_t2 = [r[algo].get("fuzzy_workload", {}).get("best", {}).get("t2", 0) for r in runs]
            fw_t3 = [r[algo].get("fuzzy_workload", {}).get("best", {}).get("t3", 0) for r in runs]
            fw_clear = [r[algo].get("fuzzy_workload", {}).get("best_clear", 0) for r in runs]
            bm[algo].update({
                "fuzzy_makespan_t1_mean": float(np.mean(fm_t1)),
                "fuzzy_makespan_t2_mean": float(np.mean(fm_t2)),
                "fuzzy_makespan_t3_mean": float(np.mean(fm_t3)),
                "fuzzy_makespan_best_clear_mean": float(np.mean(fm_clear)),
                "fuzzy_makespan_clear_mean": float(np.mean(fm_clear)),
                "fuzzy_workload_t1_mean": float(np.mean(fw_t1)),
                "fuzzy_workload_t2_mean": float(np.mean(fw_t2)),
                "fuzzy_workload_t3_mean": float(np.mean(fw_t3)),
                "fuzzy_workload_best_clear_mean": float(np.mean(fw_clear)),
                "fuzzy_workload_clear_mean": float(np.mean(fw_clear)),
            })
        benchmark_agg[inst] = bm

        # ── Ablation 视图: full vs qpas_only vs rvns_only vs moea_d ──
        ab = {}
        for algo in ABLATION_ALGOS:
            hvs = [r[algo]["final_hv"] for r in runs]
            times = [r[algo]["total_time"] for r in runs]
            ab[algo] = {
                "hv_mean": float(np.mean(hvs)), "hv_std": float(np.std(hvs, ddof=1)) if len(hvs) > 1 else 0.0,
                "time_mean": float(np.mean(times)), "time_std": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
                "runs": len(runs),
            }
        ablation_agg[inst] = ab

    return {
        "exp_id": "",
        "config": {"n_pop": n_pop, "max_gen": max_gen, "n_runs": n_runs, "base_seed": base_seed},
        "benchmark": benchmark_agg,
        "ablation": ablation_agg,
    }


def _save_aggregate(agg, output_dir, exp_id):
    """保存聚合统计 JSON。"""
    import numpy as np

    def _to_json(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {k: _to_json(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [_to_json(v) for v in o]
        return o

    agg["exp_id"] = exp_id
    exp_dir = os.path.join(output_dir, "experiment")
    os.makedirs(exp_dir, exist_ok=True)
    path = os.path.join(exp_dir, f"aggregate_{exp_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_json(agg), f, indent=2, ensure_ascii=False)
    logger.info("Aggregate saved: %s", path)


# ═══════════════════════════════════════════════════════════
# 统计检验
# ═══════════════════════════════════════════════════════════

def _wilcoxon(a, b):
    """安全的 Wilcoxon 检验，处理相同样本的除零问题。"""
    from scipy.stats import wilcoxon
    min_len = min(len(a), len(b))
    if min_len < 3:
        return None
    if np.allclose(a[:min_len], b[:min_len]):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False, "sig": "n.s."}
    try:
        stat, p = wilcoxon(a[:min_len], b[:min_len], zero_method='zsplit')
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        return {"statistic": float(stat), "p_value": float(p), "significant": p < 0.05, "sig": sig}
    except Exception:
        return None


def _cohens_d(a, b):
    """Cohen's d 效应量。"""
    d1 = np.array(a, dtype=float)
    d2 = np.array(b, dtype=float)
    pooled = np.sqrt((np.std(d1, ddof=1) ** 2 + np.std(d2, ddof=1) ** 2) / 2)
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(d1) - np.mean(d2)) / pooled)


def run_statistical_tests(agg):
    """对 benchmark 和 ablation 视图分别执行统计检验。"""
    stats = {
        "benchmark": {"wilcoxon_hv": {}, "wilcoxon_makespan": {}, "wilcoxon_workload": {}},
        "ablation": {"wilcoxon_hv": {}, "wilcoxon_makespan": {}, "wilcoxon_workload": {}},
    }

    # ── Benchmark: rmoea_d vs moea_d ──
    bm = agg.get("benchmark", {})
    for metric, key in [("wilcoxon_hv", "_hv_all"), ("wilcoxon_makespan", "_bm_all"), ("wilcoxon_workload", "_bw_all")]:
        per_instance = {}
        all_r, all_m = [], []
        for inst, data in bm.items():
            if "rmoea_d" not in data or "moea_d" not in data:
                continue
            r_vals = data["rmoea_d"].get(key, [])
            m_vals = data["moea_d"].get(key, [])
            if len(r_vals) < 3 or len(m_vals) < 3:
                per_instance[inst] = {"p_value": 1.0, "sig": "n.s.", "cohens_d": 0.0}
                continue
            w = _wilcoxon(r_vals, m_vals)
            d = _cohens_d(r_vals, m_vals)
            per_instance[inst] = {**(w or {}), "cohens_d": d}
            all_r.extend(r_vals)
            all_m.extend(m_vals)
        overall = _wilcoxon(all_r, all_m) if all_r and all_m else None
        stats["benchmark"][metric] = {"per_instance": per_instance, "overall": overall}

    # ── Ablation: 所有 pairwise 对比 ──
    ab = agg.get("ablation", {})
    for metric, field in [("wilcoxon_hv", "hv_mean"), ("wilcoxon_makespan", "best_makespan"), ("wilcoxon_workload", "best_workload")]:
        # 聚合所有实例的每个算法数据
        algo_data = {a: [] for a in ABLATION_ALGOS}
        for inst, data in ab.items():
            for a in ABLATION_ALGOS:
                if a in data:
                    algo_data[a].append(data[a].get(field.replace("wilcoxon_", "").replace("hv", "hv_mean"), 0))
        pairwise = {}
        for i, a1 in enumerate(ABLATION_ALGOS):
            for a2 in ABLATION_ALGOS[i + 1:]:
                vals1 = algo_data[a1]
                vals2 = algo_data[a2]
                if len(vals1) < 3 or len(vals2) < 3:
                    continue
                w = _wilcoxon(vals1, vals2)
                d = _cohens_d(vals1, vals2)
                pairwise[f"{a1}_vs_{a2}"] = {**(w or {}), "cohens_d": d}
        stats["ablation"][metric] = pairwise

    return stats


# ═══════════════════════════════════════════════════════════
# 报告生成
# ═══════════════════════════════════════════════════════════

def print_report(agg, stats):
    """打印实验报告。"""
    bm = agg.get("benchmark", {})
    ab = agg.get("ablation", {})

    print("\n" + "=" * 70)
    print("  UNIFIED EXPERIMENT REPORT")
    print("=" * 70)

    cfg = agg.get("config", {})
    print(f"  Config: n_pop={cfg.get('n_pop')}, max_gen={cfg.get('max_gen')}, "
          f"n_runs={cfg.get('n_runs')}, seed={cfg.get('base_seed')}")

    # ── Benchmark 对比 ──
    print("\n── Benchmark: RMOEA/D vs MOEA/D ──")
    print(f"{'Instance':<8} {'RMOEA/D HV':>12} {'MOEA/D HV':>12} {'ΔHV':>8} {'Win':>5}")
    print("-" * 50)
    for inst, data in bm.items():
        rhv = data.get("rmoea_d", {}).get("hv_mean", 0)
        mhv = data.get("moea_d", {}).get("hv_mean", 0)
        delta = rhv - mhv
        winner = "RMOEA" if delta > 0 else ("MOEA" if delta < 0 else "TIE")
        print(f"  {inst:<8} {rhv:>12.6f} {mhv:>12.6f} {delta:>+8.6f} {winner:>5}")

    # ── Ablation 对比 ──
    print("\n── Ablation: 4 Variants ──")
    header = f"{'Instance':<8}"
    for a in ABLATION_ALGOS:
        header += f" {a:>12}"
    print(header)
    print("-" * (8 + 13 * len(ABLATION_ALGOS)))
    for inst, data in ab.items():
        row = f"  {inst:<8}"
        for a in ABLATION_ALGOS:
            row += f" {data.get(a, {}).get('hv_mean', 0):>12.6f}"
        print(row)

    # ── 统计检验 ──
    bm_stats = stats.get("benchmark", {})
    wh = bm_stats.get("wilcoxon_hv", {}).get("overall", {})
    if wh:
        print(f"\n── Wilcoxon (Benchmark HV): p={wh.get('p_value', 1):.4f} "
              f"{wh.get('sig', 'n.s.')} ──")

    print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified Experiment: RMOEA/D Benchmark + Ablation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 完整实验 (benchmark + ablation)
  python experiment.py --instances Mk01 Mk02 --n_runs 30

  # 仅 benchmark (2 算法)
  python experiment.py --instances Mk01 --n_runs 30 --study benchmark

  # 仅 ablation (4 变体)
  python experiment.py --instances Mk01 --n_runs 30 --study ablation
        """,
    )
    parser.add_argument("--instances", type=str, nargs="+", default=["Mk01"])
    parser.add_argument("--n_pop", type=int, default=100)
    parser.add_argument("--max_gen", type=int, default=200)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crossover_rate", type=float, default=0.8)
    parser.add_argument("--fixed_T", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--timeout_per_task", type=float, default=None)
    parser.add_argument("--ql_alpha", type=float, default=0.4)
    parser.add_argument("--ql_gamma", type=float, default=0.6)
    parser.add_argument("--ql_epsilon", type=float, default=0.8)
    parser.add_argument("--ql_actions", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--rvns_ls_trials", type=int, default=1,
                        help="RVNS 每个解每代最多尝试的邻域次数 (论文 Algorithm 4 为 1)")
    parser.add_argument("--study", type=str, default="all",
                        choices=["all", "benchmark", "ablation"],
                        help="Which study to run (default: all)")
    args = parser.parse_args()

    setup_logging(log_dir=args.log_dir)
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 60)
    logger.info("Unified Experiment | ExpID=%s | Study=%s", exp_id, args.study)
    logger.info("Instances: %s | n_pop=%d max_gen=%d n_runs=%d",
                ','.join(args.instances), args.n_pop, args.max_gen, args.n_runs)
    logger.info("=" * 60)

    all_runs, agg = run_experiment(
        instances=args.instances,
        n_pop=args.n_pop,
        max_gen=args.max_gen,
        n_runs=args.n_runs,
        base_seed=args.seed,
        crossover_rate=args.crossover_rate,
        fixed_T=args.fixed_T,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        ql_alpha=args.ql_alpha,
        ql_gamma=args.ql_gamma,
        ql_epsilon=args.ql_epsilon,
        ql_actions=args.ql_actions,
        n_workers=args.n_workers,
        exp_id=exp_id,
        timeout_per_task=args.timeout_per_task,
        rvns_ls_trials=args.rvns_ls_trials,
    )

    if args.n_runs > 2:
        stats = run_statistical_tests(agg)
    else:
        stats = {"benchmark": {}, "ablation": {}}

    print_report(agg, stats)

    logger.info("Experiment complete! ExpID=%s", exp_id)


if __name__ == "__main__":
    main()