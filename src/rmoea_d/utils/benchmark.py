#!/usr/bin/env python3
"""
Benchmark script: RMOEA/D vs MOEA/D comparison on Brandimarte instances.
对比实验脚本：支持多次独立运行、统计分析和三角模糊数报告。

功能：
- 多轮独立运行 (--n_runs)，统计 mean ± std
- Friedman检验 + Wilcoxon符号秩检验
- 模糊目标 (t1,t2,t3) 的统计对比
"""

import argparse
import sys
import os
import json
import time
import math
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from scipy.stats import friedmanchisquare as _friedman_scipy

# ── 禁止 BLAS/MKL 内部多线程，避免与进程池冲突 ──
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Add the src directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rmoea_d.algorithm import RMOEAD
from rmoea_d.moead_baseline import MOEADBaseline
from rmoea_d.utils.logger_setup import setup_logging
from rmoea_d.core.instance import ALL_INSTANCES

import logging
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 统计检验模块（论文表4、表8的Friedman和Wilcoxon检验）
# ═══════════════════════════════════════════════════════════

def friedman_test(*args):
    """
    Friedman 秩和检验
    用于比较多个算法在多个实例上的性能是否显著不同
    H0: 所有算法表现相同
    """
    try:
        from scipy.stats import friedmanchisquare
        stat, p = friedmanchisquare(*args)
        return stat, p
    except ImportError:
        # 手动实现Friedman检验
        data = np.array(args)  # shape: (n_algorithms, n_instances)
        k = data.shape[0]  # 算法数
        n = data.shape[1]  # 实例数
        # 对每个实例内排名
        ranks = np.zeros_like(data, dtype=float)
        for j in range(n):
            col = data[:, j]
            # 排名(1=最小)，处理平局
            order = np.argsort(col)
            for rank, idx in enumerate(order):
                ranks[idx, j] = rank + 1
        R = ranks.sum(axis=1)  # 各算法秩和
        # Friedman统计量
        chi2 = (12 * n) / (k * (k + 1)) * np.sum((R - n * (k + 1) / 2) ** 2)
        from scipy.stats import chi2 as chi2_dist
        p = 1 - chi2_dist.cdf(chi2, k - 1)
        return chi2, p


def wilcoxon_test(sample1, sample2):
    """
    Wilcoxon 符号秩检验（配对）
    H0: 两个样本来自相同分布
    返回: (统计量, p值)
    """
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(sample1, sample2)
        return stat, p
    except ImportError:
        # 简易手动实现
        s1 = np.array(sample1, dtype=float)
        s2 = np.array(sample2, dtype=float)
        d = s1 - s2
        d = d[d != 0]  # 去掉差为0的
        if len(d) == 0:
            return 0, 1.0
        abs_d = np.abs(d)
        order = np.argsort(abs_d)
        ranks = np.zeros(len(d))
        for i, idx in enumerate(order):
            ranks[idx] = i + 1
        # 处理平局
        W = np.sum(ranks[d > 0])
        # 正态近似
        n = len(d)
        mean_W = n * (n + 1) / 4
        std_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        if std_W > 0:
            z = (W - mean_W) / std_W
            from scipy.stats import norm
            p = 2 * (1 - norm.cdf(abs(z)))
        else:
            p = 1.0
        return W, p


# ═══════════════════════════════════════════════════════════
# 单次运行与结果提取
# ═══════════════════════════════════════════════════════════

def run_single(instance, n_pop, max_gen, crossover_rate, seed, fixed_T, data_dir):
    """运行单次对比实验"""
    # RMOEA/D
    solver_r = RMOEAD(
        instance_name=instance, n_pop=n_pop, max_gen=max_gen,
        crossover_rate=crossover_rate, seed=seed, data_dir=data_dir)
    results_r = solver_r.solve()

    # MOEA/D baseline
    solver_m = MOEADBaseline(
        instance_name=instance, n_pop=n_pop, max_gen=max_gen,
        crossover_rate=crossover_rate, fixed_T=fixed_T, seed=seed, data_dir=data_dir)
    results_m = solver_m.solve()

    # 提取统计量
    def _pf_stats(pf):
        if not pf:
            return {}
        if isinstance(pf[0], dict):
            ms = [p["Makespan"] for p in pf]
            wl = [p["Workload"] for p in pf]
        else:
            arr = np.array(pf)
            ms, wl = arr[:, 0], arr[:, 1]
        return {
            "best_makespan": float(np.min(ms)),
            "best_workload": float(np.min(wl)),
            "avg_makespan": float(np.mean(ms)),
            "avg_workload": float(np.mean(wl)),
        }

    def _fuzzy_stats(fuzzy_pf):
        if not fuzzy_pf:
            return {}
        from rmoea_d.core.fuzzy import fuzzy_tuple_clear_value
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
                "best_clear": float(np.min(ms_clear)),
            },
            "fuzzy_workload": {
                "best": {"t1": float(np.min(wl_t1)), "t2": float(np.min(wl_t2)), "t3": float(np.min(wl_t3))},
                "best_clear": float(np.min(wl_clear)),
            },
        }

    pf_r = results_r["final_pf"]
    pf_m = results_m["final_pf"]
    stats_r = _pf_stats(pf_r)
    stats_m = _pf_stats(pf_m)
    fuzzy_stats_r = _fuzzy_stats(results_r.get("fuzzy_pf", []))
    fuzzy_stats_m = _fuzzy_stats(results_m.get("fuzzy_pf", []))

    return {
        "rmoea_d": {
            "final_hv": results_r["final_hv"],
            "pf_size": len(pf_r),
            "total_time": results_r["total_time"],
            "q_table": results_r.get("q_table", []),
            "final_pf": results_r.get("final_pf", []),
            "fuzzy_pf": results_r.get("fuzzy_pf", []),
            "history": results_r.get("history", []),
            **stats_r, **fuzzy_stats_r,
        },
        "moea_d": {
            "final_hv": results_m["final_hv"],
            "pf_size": len(pf_m),
            "total_time": results_m["total_time"],
            "final_pf": results_m.get("final_pf", []),
            "fuzzy_pf": results_m.get("fuzzy_pf", []),
            "history": results_m.get("history", []),
            **stats_m, **fuzzy_stats_m,
        },
        "results_r": results_r,
        "results_m": results_m,
    }


# ═══════════════════════════════════════════════════════════
# 多轮运行与聚合 (并行版)
# ═══════════════════════════════════════════════════════════

def _run_single_worker(args_tuple):
    """进程池 worker：运行单次对比实验 (picklable 顶层函数)"""
    inst, n_pop, max_gen, crossover_rate, seed, fixed_T, data_dir = args_tuple
    comp = run_single(inst, n_pop, max_gen, crossover_rate, seed, fixed_T, data_dir)
    return inst, seed, comp


def run_multi(instances, n_pop, max_gen, crossover_rate, base_seed, fixed_T,
              data_dir, n_runs, output_dir, n_workers=None):
    """多轮独立运行（并行版），返回聚合统计结果"""
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, n_runs)

    # ── 生成所有任务 (instance, seed) 对 ──
    tasks = []
    for inst in instances:
        for run_idx in range(n_runs):
            seed = base_seed + run_idx if n_runs > 1 else base_seed
            tasks.append((inst, n_pop, max_gen, crossover_rate, seed, fixed_T, data_dir))

    logger.debug("Running %d tasks (%d instances x %d runs) with %d workers",
                 len(tasks), len(instances), n_runs, n_workers)

    # ── 并行执行 ──
    all_runs = {inst: {"rmoea_d": [], "moea_d": [], "_full_results_r": []} for inst in instances}
    completed = 0
    t0 = time.time()
    milestone_interval = max(1, len(tasks) // 5)  # 每20%报告一次

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_single_worker, t): t for t in tasks}
        for future in as_completed(futures):
            inst, seed, comp = future.result()
            all_runs[inst]["rmoea_d"].append(comp["rmoea_d"])
            all_runs[inst]["moea_d"].append(comp["moea_d"])
            # 保留完整 solve() 结果（含 schedules 调度数据，用于甘特图生成）
            all_runs[inst]["_full_results_r"].append(comp.get("results_r", {}))
            completed += 1
            elapsed = time.time() - t0
            logger.debug("[%d/%d] %s seed=%d HV_R=%.4f HV_M=%.4f [%.0fs]",
                        completed, len(tasks), inst, seed,
                        comp['rmoea_d']['final_hv'], comp['moea_d']['final_hv'], elapsed)
            if completed % milestone_interval == 0 or completed == len(tasks):
                logger.info("Benchmark %d/%d tasks [%.0fs]", completed, len(tasks), elapsed)

    # ── 聚合统计 (mean ± std) ──
    agg_results = {}
    for inst in instances:
        agg = {"instance": inst, "n_runs": n_runs,
               "n_pop": n_pop, "max_gen": max_gen, "base_seed": base_seed}
        for key in ["rmoea_d", "moea_d"]:
            runs = all_runs[inst][key]
            hvs = [r["final_hv"] for r in runs]
            pf_sizes = [r["pf_size"] for r in runs]
            times_list = [r["total_time"] for r in runs]
            bm = [r["best_makespan"] for r in runs]
            bw = [r["best_workload"] for r in runs]

            # 模糊统计量
            fm_t1 = [r.get("fuzzy_makespan", {}).get("best", {}).get("t1", 0) for r in runs]
            fm_t2 = [r.get("fuzzy_makespan", {}).get("best", {}).get("t2", 0) for r in runs]
            fm_t3 = [r.get("fuzzy_makespan", {}).get("best", {}).get("t3", 0) for r in runs]
            fm_clear = [r.get("fuzzy_makespan", {}).get("best_clear", 0) for r in runs]
            fw_t1 = [r.get("fuzzy_workload", {}).get("best", {}).get("t1", 0) for r in runs]
            fw_t2 = [r.get("fuzzy_workload", {}).get("best", {}).get("t2", 0) for r in runs]
            fw_t3 = [r.get("fuzzy_workload", {}).get("best", {}).get("t3", 0) for r in runs]
            fw_clear = [r.get("fuzzy_workload", {}).get("best_clear", 0) for r in runs]

            agg[key] = {
                "hv_mean": float(np.mean(hvs)),
                "hv_std": float(np.std(hvs, ddof=1)) if n_runs > 1 else 0.0,
                "pf_size_mean": float(np.mean(pf_sizes)),
                "pf_size_std": float(np.std(pf_sizes, ddof=1)) if n_runs > 1 else 0.0,
                "time_mean": float(np.mean(times_list)),
                "time_std": float(np.std(times_list, ddof=1)) if n_runs > 1 else 0.0,
                "best_makespan_mean": float(np.mean(bm)),
                "best_makespan_std": float(np.std(bm, ddof=1)) if n_runs > 1 else 0.0,
                "best_workload_mean": float(np.mean(bw)),
                "best_workload_std": float(np.std(bw, ddof=1)) if n_runs > 1 else 0.0,
                # Fuzzy
                "fuzzy_makespan_t1_mean": float(np.mean(fm_t1)),
                "fuzzy_makespan_t2_mean": float(np.mean(fm_t2)),
                "fuzzy_makespan_t3_mean": float(np.mean(fm_t3)),
                "fuzzy_makespan_clear_mean": float(np.mean(fm_clear)),
                "fuzzy_workload_t1_mean": float(np.mean(fw_t1)),
                "fuzzy_workload_t2_mean": float(np.mean(fw_t2)),
                "fuzzy_workload_t3_mean": float(np.mean(fw_t3)),
                "fuzzy_workload_clear_mean": float(np.mean(fw_clear)),
            }
            # 保留每次运行的数据用于统计检验
            agg[key]["_hv_all"] = hvs
            agg[key]["_bm_all"] = bm
            agg[key]["_bw_all"] = bw
            agg[key]["_fm_t2_all"] = fm_t2
            agg[key]["_fw_t2_all"] = fw_t2

        agg_results[inst] = agg
        # 附加全局配置信息供报告使用
        agg_results[inst]["n_pop"] = n_pop
        agg_results[inst]["max_gen"] = max_gen
        agg_results[inst]["n_runs"] = n_runs
        agg_results[inst]["base_seed"] = base_seed

    return agg_results, all_runs


# ═══════════════════════════════════════════════════════════
# 统计检验执行
# ═══════════════════════════════════════════════════════════

def run_statistical_tests(agg_results):
    """执行Friedman和Wilcoxon检验（论文表4、表8）"""
    instances = sorted(agg_results.keys())
    n_inst = len(instances)
    n_algo = 2  # RMOEA/D vs MOEA/D

    # ── Friedman Test (基于每个实例各算法多次运行的HV均值) ──
    rmoea_hv_means = [agg_results[i]["rmoea_d"]["hv_mean"] for i in instances]
    moea_hv_means = [agg_results[i]["moea_d"]["hv_mean"] for i in instances]

    try:
        stat_f, p_f = friedman_test(rmoea_hv_means, moea_hv_means)
        friedman_valid = True
    except Exception:
        stat_f, p_f = 0, 1
        friedman_valid = False

    # ── Wilcoxon Signed-Rank Test (每个实例的多次HV配对) ──
    wilcoxon_results = {}
    for inst in instances:
        hv_r = np.array(agg_results[inst]["rmoea_d"]["_hv_all"])
        hv_m = np.array(agg_results[inst]["moea_d"]["_hv_all"])
        try:
            w_stat, w_p = wilcoxon_test(hv_r, hv_m)
            wilcoxon_results[inst] = {"R+": float(w_stat), "p_value": float(w_p),
                                        "sig": "yes" if w_p < 0.05 else "no"}
        except Exception:
            wilcoxon_results[inst] = {"R+": 0, "p_value": 1, "sig": "no"}

    # ── 总Wilcoxon (所有实例的HV值合并) ──
    all_hv_r = []
    all_hv_m = []
    for inst in instances:
        all_hv_r.extend(agg_results[inst]["rmoea_d"]["_hv_all"])
        all_hv_m.extend(agg_results[inst]["moea_d"]["_hv_all"])
    try:
        w_stat_total, w_p_total = wilcoxon_test(all_hv_r, all_hv_m)
    except Exception:
        w_stat_total, w_p_total = 0, 1

    return {
        "friedman": {
            "statistic": float(stat_f),
            "p_value": float(p_f),
            "significant": "yes" if p_f < 0.05 else "no",
        },
        "wilcoxon_per_instance": wilcoxon_results,
        "wilcoxon_overall": {
            "statistic": float(w_stat_total),
            "p_value": float(w_p_total),
            "significant": "yes" if w_p_total < 0.05 else "no",
        },
    }


# ═══════════════════════════════════════════════════════════
# 报告生成（含三角模糊数对比）
# ═══════════════════════════════════════════════════════════

def print_report(agg_results, stats_tests):
    """打印完整实验报告 — 详细表格写入日志文件，关键摘要打印到控制台"""
    instances = sorted(agg_results.keys())
    first = agg_results[instances[0]]

    # ── 详细报告 → logger.debug（文件） ──
    logger.debug("=" * 90)
    logger.debug("  RMOEA/D vs MOEA/D - EXPERIMENTAL RESULTS REPORT")
    logger.debug("  (Paper: A reinforcement learning based RMOEA/D for bi-objective FFJSP)")
    logger.debug("=" * 90)
    logger.debug("  Configuration: Np=%d, Gen=%d, Runs=%d, Seed=%d",
                 first['n_pop'], first['max_gen'], first['n_runs'], first['base_seed'])

    # 表1: HV对比
    logger.debug("  %s", "─" * 85)
    logger.debug("  TABLE 1: Hypervolume (HV) Comparison  [mean ± std, N=%d]", first['n_runs'])
    logger.debug("  %s", "─" * 85)
    logger.debug("  %-12s %-22s %-22s %-15s %s", "Instance", "RMOEA/D HV", "MOEA/D HV", "Improvement", "Winner")
    for inst in instances:
        a = agg_results[inst]
        hv_r, hv_rs = a["rmoea_d"]["hv_mean"], a["rmoea_d"]["hv_std"]
        hv_m, hv_ms = a["moea_d"]["hv_mean"], a["moea_d"]["hv_std"]
        imp = ((hv_r - hv_m) / hv_m * 100) if hv_m > 0 else 0
        winner = "RMOEA/D" if hv_r > hv_m else ("MOEA/D" if hv_m > hv_r else "Tie")
        logger.debug("  %-12s %.4f ±%.4f       %.4f ±%.4f      %+.2f%%          %s",
                     inst.upper(), hv_r, hv_rs, hv_m, hv_ms, imp, winner)

    # 表2: Makespan (Crisp)
    logger.debug("  %s", "─" * 85)
    logger.debug("  TABLE 2: Best Makespan Comparison (Crisp)  [mean ± std]")
    logger.debug("  %-12s %-22s %-22s %-15s", "Instance", "RMOEA/D", "MOEA/D", "Improvement")
    for inst in instances:
        a = agg_results[inst]
        mr, mrs = a["rmoea_d"]["best_makespan_mean"], a["rmoea_d"]["best_makespan_std"]
        mm, mms = a["moea_d"]["best_makespan_mean"], a["moea_d"]["best_makespan_std"]
        imp = ((mm - mr) / mm * 100) if mm > 0 else 0
        logger.debug("  %-12s %.2f ±%.2f           %.2f ±%.2f           %+.2f%%",
                     inst.upper(), mr, mrs, mm, mms, imp)

    # 表3: Fuzzy Makespan
    logger.debug("  %s", "─" * 85)
    logger.debug("  TABLE 3: Fuzzy Makespan - Triangular Fuzzy Number (t1, t2, t3)")
    logger.debug("       Format: t1(earliest) / t2(most-likely) / t3(latest)")
    logger.debug("  %-12s %-30s %-30s", "Instance", "RMOEA/D (t1/t2/t3)", "MOEA/D (t1/t2/t3)")
    for inst in instances:
        a = agg_results[inst]
        r = (f"{a['rmoea_d']['fuzzy_makespan_t1_mean']:.1f}/"
             f"{a['rmoea_d']['fuzzy_makespan_t2_mean']:.1f}/"
             f"{a['rmoea_d']['fuzzy_makespan_t3_mean']:.1f}")
        m = (f"{a['moea_d']['fuzzy_makespan_t1_mean']:.1f}/"
             f"{a['moea_d']['fuzzy_makespan_t2_mean']:.1f}/"
             f"{a['moea_d']['fuzzy_makespan_t3_mean']:.1f}")
        logger.debug("  %-12s %-30s %-30s", inst.upper(), r, m)

    # 表4: Fuzzy Workload
    logger.debug("  %s", "─" * 85)
    logger.debug("  TABLE 4: Fuzzy Workload - Triangular Fuzzy Number (t1, t2, t3)")
    logger.debug("  %-12s %-30s %-30s", "Instance", "RMOEA/D (t1/t2/t3)", "MOEA/D (t1/t2/t3)")
    for inst in instances:
        a = agg_results[inst]
        r = (f"{a['rmoea_d']['fuzzy_workload_t1_mean']:.1f}/"
             f"{a['rmoea_d']['fuzzy_workload_t2_mean']:.1f}/"
             f"{a['rmoea_d']['fuzzy_workload_t3_mean']:.1f}")
        m = (f"{a['moea_d']['fuzzy_workload_t1_mean']:.1f}/"
             f"{a['moea_d']['fuzzy_workload_t2_mean']:.1f}/"
             f"{a['moea_d']['fuzzy_workload_t3_mean']:.1f}")
        logger.debug("  %-12s %-30s %-30s", inst.upper(), r, m)

    # 表5: 运行时间
    logger.debug("  %s", "─" * 85)
    logger.debug("  TABLE 5: Runtime Comparison (seconds)  [mean ± std]")
    for inst in instances:
        a = agg_results[inst]
        tr, trs = a["rmoea_d"]["time_mean"], a["rmoea_d"]["time_std"]
        tm, tms = a["moea_d"]["time_mean"], a["moea_d"]["time_std"]
        logger.debug("  %-12s RMOEA/D: %.2f ±%.2fs      MOEA/D: %.2f ±%.2fs",
                     inst.upper(), tr, trs, tm, tms)

    # 统计检验
    ft = stats_tests.get("friedman", {})
    wt = stats_tests.get("wilcoxon_overall", {})
    wp = stats_tests.get("wilcoxon_per_instance", {})
    
    if ft:
        logger.debug("  %s", "=" * 85)
        logger.debug("  STATISTICAL TESTS (ref. Paper Table 8)")
        logger.debug("  Friedman Test (on HV means across instances): chi2=%.4f, p=%.6f, sig=%s",
                     ft.get('statistic', 0), ft.get('p_value', 1),
                     str(ft.get('significant', False)).upper())
    if wt:
        logger.debug("  Wilcoxon Signed-Rank Test (HV paired): R+=%.2f, p=%.6f, sig=%s",
                     wt.get('statistic', 0), wt.get('p_value', 1),
                     str(wt.get('significant', False)).upper())
    if wp:
        logger.debug("  Per-instance Wilcoxon results:")
        for inst in instances:
            w = wp[inst]
            sig_mark = "***" if w.get("sig") == "yes" else "   "
            logger.debug("    %s: R+=%.2f, p=%.4f %s", inst.upper(),
                        w.get('R+', 0), w.get('p_value', 1), sig_mark)

    # ── 关键摘要 → logger.info（控制台+文件） ──
    avg_hv_r = np.mean([agg_results[i]["rmoea_d"]["hv_mean"] for i in instances])
    avg_hv_m = np.mean([agg_results[i]["moea_d"]["hv_mean"] for i in instances])
    n_better = sum(1 for i in instances
                   if agg_results[i]["rmoea_d"]["hv_mean"] > agg_results[i]["moea_d"]["hv_mean"])
    logger.info("Report: RMOEA/D avgHV=%.4f | MOEA/D avgHV=%.4f | RMOEA/D wins %d/%d",
                avg_hv_r, avg_hv_m, n_better, len(instances))
    if ft and wt:
        logger.info("Stats: Friedman p=%.4f | Wilcoxon p=%.4f",
                    ft.get('p_value', 1), wt.get('p_value', 1))


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: RMOEA/D vs MOEA/D (multi-run, statistical tests)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run (original behavior)
  python benchmark.py --instances Mk01

  # 30 independent runs (paper standard)
  python benchmark.py --instances Mk01 --n_runs 30 --seed 42

  # Multiple instances, 30 runs each
  python benchmark.py --instances Mk01 Mk02 Mk03 --n_runs 30 --seed 42
        """,
    )
    parser.add_argument("--instances", type=str, nargs="+", default=["Mk01"])
    parser.add_argument("--n_pop", type=int, default=100)
    parser.add_argument("--max_gen", type=int, default=200)
    parser.add_argument("--crossover_rate", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (incremented per run)")
    parser.add_argument("--fixed_T", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of independent runs per instance (paper: 30)")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Parallel workers for n_runs (default: auto=min(cpu_count,n_runs))")
    args = parser.parse_args()

    setup_logging(log_dir=args.log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 多轮运行 (并行版)
    agg_results, all_runs = run_multi(
        instances=args.instances,
        n_pop=args.n_pop,
        max_gen=args.max_gen,
        crossover_rate=args.crossover_rate,
        base_seed=args.seed,
        fixed_T=args.fixed_T,
        data_dir=args.data_dir,
        n_runs=args.n_runs,
        output_dir=args.output_dir,
        n_workers=args.n_workers,
    )

    # 统计检验
    if args.n_runs > 2:
        stats_tests = run_statistical_tests(agg_results)
    else:
        stats_tests = {"friedman": {}, "wilcoxon_per_instance": {}, "wilcoxon_overall": {}}

    # 打印报告
    print_report(agg_results, stats_tests)

    # 保存聚合结果（JSON）
    save_data = {
        "timestamp": timestamp,
        "config": {
            "n_pop": args.n_pop, "max_gen": args.max_gen,
            "n_runs": args.n_runs, "base_seed": args.seed,
            "crossover_rate": args.crossover_rate, "fixed_T": args.fixed_T,
        },
        "instances": {},
        "statistical_tests": stats_tests,
    }
    for inst, agg in agg_results.items():
        # 去掉内部列表字段以保持JSON简洁
        clean = {k: v for k, v in agg.items()}
        for key in ["rmoea_d", "moea_d"]:
            if key in clean:
                clean[key] = {k: v for k, v in clean[key].items()
                               if not k.startswith("_")}
        save_data["instances"][inst] = clean

    output_path = os.path.join(args.output_dir,
                                f"benchmark_aggregate_{timestamp}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    logger.debug("Aggregate results saved to: %s", output_path)

    # 保存每次运行详细结果（原有格式）
    for inst in args.instances:
        for run_idx in range(args.n_runs):
            ts_run = time.strftime("%Y%m%d_%H%M%S")
            # 从all_runs获取原始数据 - 我们需要保存原始results_r和results_m
            # 由于原始solve()数据在run_single中返回但未全部存储，
            # 这里保存聚合后的每次运行摘要
            run_summary = {
                "instance": inst,
                "run": run_idx,
                "rmoea_d": all_runs[inst]["rmoea_d"][run_idx],
                "moea_d": all_runs[inst]["moea_d"][run_idx],
            }
            for algo_key, algo_dir in [("rmoea_d", "RMOEA_D"),
                                        ("moea_d", "MOEA_D")]:
                r_dir = os.path.join(args.output_dir, inst, algo_dir)
                os.makedirs(r_dir, exist_ok=True)
                r_path = os.path.join(r_dir,
                    f"{inst}_{algo_dir}_Np{args.n_pop}_G{args.max_gen}_run{run_idx}_{ts_run}.json")
                with open(r_path, "w", encoding="utf-8") as f:
                    json.dump(run_summary[algo_key], f, indent=2, ensure_ascii=False)

            # 保存单实例单次summary
            sum_dir = os.path.join(args.output_dir, inst)
            sum_path = os.path.join(sum_dir,
                f"benchmark_summary_{inst}_run{run_idx}_{ts_run}.json")
            comp = {
                "instance": inst, "run": run_idx,
                "n_pop": args.n_pop, "max_gen": args.max_gen, "seed": args.seed + run_idx,
                "rmoea_d": run_summary["rmoea_d"],
                "moea_d": run_summary["moea_d"],
            }
            with open(sum_path, "w", encoding="utf-8") as f:
                json.dump(comp, f, indent=2, ensure_ascii=False)

            # ── 保存完整 RMOEA/D 结果（含调度数据 schedules，用于甘特图生成）──
            full_r = all_runs[inst]["_full_results_r"][run_idx]
            if full_r and full_r.get("schedules"):
                rmo_dir = os.path.join(args.output_dir, inst, "RMOEA_D")
                os.makedirs(rmo_dir, exist_ok=True)
                full_path = os.path.join(rmo_dir,
                    f"{inst}_full_schedule_run{run_idx}_{ts_run}.json")
                with open(full_path, "w", encoding="utf-8") as f:
                    json.dump(full_r, f, indent=2, ensure_ascii=False)
                logger.debug("Full schedule saved: %s", full_path)

    logger.info("Benchmark done: %d runs × %d instances saved to %s/", 
                args.n_runs, len(args.instances), args.output_dir)


if __name__ == "__main__":
    main()