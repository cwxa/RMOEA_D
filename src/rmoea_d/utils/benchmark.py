#!/usr/bin/env python3
"""
Benchmark script: RMOEA/D vs MOEA/D comparison on Brandimarte instances.
对比实验脚本：支持多次独立运行、统计分析和三角模糊数报告。

.. deprecated::
    推荐使用 experiment.py 进行统一实验，单次运行同时产出 benchmark + ablation 数据。
    此脚本保留用于向后兼容和独立调试。
    run_all.py v4.0 已统一到 experiment.py，不再调用此脚本。

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
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
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
        import numpy as np
        # 若两样本完全相同，se=0 会导致除零 RuntimeWarning，直接返回 p=1.0
        if np.allclose(sample1, sample2):
            return 0.0, 1.0
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


def cohens_d(sample1, sample2):
    """Cohen's d 效应量 (pooled SD), 衡量两组样本的差异幅度。
    |d| < 0.2: negligible; 0.2-0.5: small; 0.5-0.8: medium; >0.8: large."""
    s1, s2 = np.array(sample1, dtype=float), np.array(sample2, dtype=float)
    n1, n2 = len(s1), len(s2)
    if n1 < 2 or n2 < 2:
        return 0.0
    v1, v2 = np.var(s1, ddof=1), np.var(s2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_sd < 1e-12:
        return 0.0
    return (np.mean(s1) - np.mean(s2)) / pooled_sd


def _wilcoxon_multi_metric(agg_results, metric_key):
    """对指定 metric 做 per-instance + overall Wilcoxon 检验。
    metric_key: '_hv_all' | '_bm_all' | '_bw_all' 等"""
    instances = sorted(agg_results.keys())
    wilcoxon_per = {}
    all_r, all_m = [], []
    for inst in instances:
        hv_r = np.array(agg_results[inst]["rmoea_d"][metric_key])
        hv_m = np.array(agg_results[inst]["moea_d"][metric_key])
        all_r.extend(hv_r.tolist())
        all_m.extend(hv_m.tolist())
        try:
            w_stat, w_p = wilcoxon_test(hv_r, hv_m)
            d = cohens_d(hv_r, hv_m)
            wilcoxon_per[inst] = {"R+": float(w_stat), "p_value": float(w_p),
                                   "cohens_d": float(d),
                                   "sig": _sig_level(w_p)}
        except Exception:
            wilcoxon_per[inst] = {"R+": 0, "p_value": 1, "cohens_d": 0.0, "sig": "ns"}
    try:
        w_stat_total, w_p_total = wilcoxon_test(all_r, all_m)
        d_total = cohens_d(all_r, all_m)
    except Exception:
        w_stat_total, w_p_total, d_total = 0, 1, 0.0
    return {
        "per_instance": wilcoxon_per,
        "overall": {"statistic": float(w_stat_total), "p_value": float(w_p_total),
                     "cohens_d": float(d_total),
                     "significant": "yes" if w_p_total < 0.05 else "no"},
    }


def _sig_level(p):
    """p值转显著性标记 (ASCII星号，兼容Windows终端)"""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


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
              data_dir, n_runs, output_dir, n_workers=None, timeout_per_task=None):
    """多轮独立运行（并行版），返回聚合统计结果
    
    Parameters:
        timeout_per_task: 单个任务最大执行时间（秒），超时后将跳过该任务并记录警告
    """
    if n_workers is None:
        # 优先用环境变量感知外层并发数，否则默认用全部 CPU
        parent_workers = int(os.environ.get("RMOEA_PARENT_WORKERS", "0"))
        if parent_workers > 0:
            total_cpus = os.cpu_count() or 4
            n_workers = max(1, min(n_runs, total_cpus // parent_workers))
        else:
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
    timed_out = 0
    t0 = time.time()
    milestone_interval = max(1, len(tasks) // 5)  # 每20%报告一次

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_single_worker, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                inst, seed, comp = future.result(timeout=timeout_per_task)
            except FutureTimeoutError:
                task = futures[future]
                timed_out += 1
                logger.warning("[%d/%d] Timeout: %s seed=%s > %.0fs, skipping",
                               completed + timed_out, len(tasks), task[0], task[5], timeout_per_task)
                future.cancel()
                continue
            except Exception as e:
                task = futures[future]
                timed_out += 1
                logger.error("[%d/%d] Error: %s seed=%s: %s, skipping",
                             completed + timed_out, len(tasks), task[0], task[5], e)
                continue
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

    if timed_out > 0:
        logger.warning("%d/%d tasks timed out or errored", timed_out, len(tasks))

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
    """执行完整的统计检验：Friedman + 多指标 Wilcoxon + 效应量 (Cohen's d)。
    覆盖 HV、Makespan、Workload 三个核心指标。"""
    instances = sorted(agg_results.keys())

    # ── Friedman Test (基于每实例各算法 HV 均值) ──
    rmoea_hv_means = [agg_results[i]["rmoea_d"]["hv_mean"] for i in instances]
    moea_hv_means = [agg_results[i]["moea_d"]["hv_mean"] for i in instances]
    try:
        stat_f, p_f = friedman_test(rmoea_hv_means, moea_hv_means)
        friedman_valid = True
    except Exception:
        stat_f, p_f = 0, 1
        friedman_valid = False

    # ── 多指标 Wilcoxon + 效应量 ──
    wilcoxon_hv = _wilcoxon_multi_metric(agg_results, "_hv_all")
    wilcoxon_ms = _wilcoxon_multi_metric(agg_results, "_bm_all")
    wilcoxon_wl = _wilcoxon_multi_metric(agg_results, "_bw_all")

    # ── 跨实例平均效应量 ──
    agg_effect = {}
    for metric_name, w_data in [("HV", wilcoxon_hv), ("Makespan", wilcoxon_ms),
                                  ("Workload", wilcoxon_wl)]:
        ds = [d["cohens_d"] for d in w_data["per_instance"].values()]
        agg_effect[metric_name] = {
            "mean_d": float(np.mean(ds)) if ds else 0.0,
            "median_d": float(np.median(ds)) if ds else 0.0,
            "min_d": float(np.min(ds)) if ds else 0.0,
            "max_d": float(np.max(ds)) if ds else 0.0,
            "n_large": sum(1 for d in ds if abs(d) > 0.8),
            "n_medium": sum(1 for d in ds if 0.5 < abs(d) <= 0.8),
            "n_small": sum(1 for d in ds if 0.2 < abs(d) <= 0.5),
            "n_negligible": sum(1 for d in ds if abs(d) <= 0.2),
        }

    return {
        "friedman": {
            "statistic": float(stat_f),
            "p_value": float(p_f),
            "significant": "yes" if p_f < 0.05 else "no",
        },
        "wilcoxon_hv": wilcoxon_hv,
        "wilcoxon_makespan": wilcoxon_ms,
        "wilcoxon_workload": wilcoxon_wl,
        "effect_size_summary": agg_effect,
    }


# ═══════════════════════════════════════════════════════════
# 报告生成（含三角模糊数对比）
# ═══════════════════════════════════════════════════════════

def print_report(agg_results, stats_tests):
    """输出实验报告：详细表格 → logger.debug (日志文件)，关键摘要 + 统计 → logger.info (控制台可见)。

    控制台输出结构：
      1. 整体摘要 (avgHV + 胜率)
      2. Friedman + Overall Wilcoxon 显著性
      3. Per-instance 快速一览表 (HV + Makespan + 显著性)
      4. 效应量汇总 (Cohen's d 分布)
    """
    instances = sorted(agg_results.keys())
    first = agg_results[instances[0]]
    n_runs = first.get('n_runs', 0)

    # ═══════════════════════════════════════════════════════
    # 控制台可见摘要 (logger.info)
    # ═══════════════════════════════════════════════════════

    avg_hv_r = np.mean([agg_results[i]["rmoea_d"]["hv_mean"] for i in instances])
    avg_hv_m = np.mean([agg_results[i]["moea_d"]["hv_mean"] for i in instances])
    n_better = sum(1 for i in instances
                   if agg_results[i]["rmoea_d"]["hv_mean"] > agg_results[i]["moea_d"]["hv_mean"])

    try:
        logger.info("")
        logger.info("=" * 78)
        logger.info("  RMOEA/D vs MOEA/D -- BENCHMARK STATISTICAL REPORT")
        logger.info("-" * 78)
        logger.info("  Config: Np=%d, Gen=%d, Runs=%d, Instances=%d",
                    first['n_pop'], first['max_gen'], n_runs, len(instances))
        logger.info("-" * 78)
        logger.info("  HV Summary")
        logger.info("    RMOEA/D  avgHV = %.4f", avg_hv_r)
        logger.info("    MOEA/D   avgHV = %.4f", avg_hv_m)
        logger.info("    RMOEA/D  wins  %d / %d  instances",
                    n_better, len(instances))
        logger.info("-" * 78)

        # ── 统计检验结果 ──
        ft = stats_tests.get("friedman", {})
        wh = stats_tests.get("wilcoxon_hv", {})
        wm = stats_tests.get("wilcoxon_makespan", {})
        ww = stats_tests.get("wilcoxon_workload", {})
        es = stats_tests.get("effect_size_summary", {})

        if ft:
            logger.info("  Friedman Test (on HV means):")
            logger.info("    chi^2 = %.4f, p = %.6f  %s",
                        ft.get('statistic', 0), ft.get('p_value', 1),
                        "<- SIGNIFICANT" if ft.get('significant') == 'yes' else "")

        logger.info("-" * 78)
        logger.info("  Wilcoxon Signed-Rank (Overall, paired by run)")
        for label, wd in [("HV", wh), ("Makespan", wm), ("Workload", ww)]:
            ov = wd.get("overall", {})
            sig_mark = "* SIGNIFICANT" if ov.get("significant") == "yes" else "not significant"
            logger.info("    %9s:  R+=%.1f  p=%.6f  d=%.3f  %s",
                        label, ov.get('statistic', 0), ov.get('p_value', 1),
                        ov.get('cohens_d', 0), sig_mark)

        # ── Per-instance 快速一览表 ──
        logger.info("-" * 78)
        logger.info("  Per-Instance: HV [Delta%]  |  Makespan [Delta%]  |  Wilcoxon p")
        for inst in instances:
            a = agg_results[inst]
            hv_r, hv_m = a["rmoea_d"]["hv_mean"], a["moea_d"]["hv_mean"]
            mr, mm = a["rmoea_d"]["best_makespan_mean"], a["moea_d"]["best_makespan_mean"]
            hv_delta = ((hv_r - hv_m) / hv_m * 100) if hv_m > 0 else 0
            ms_delta = ((mm - mr) / mm * 100) if mm > 0 else 0
            # 安全访问 per_instance，避免 KeyError
            wp_hv = wh.get("per_instance", {}).get(inst, {})
            wp_ms = wm.get("per_instance", {}).get(inst, {})
            logger.info("  %-5s  HV:%+.1f%% %s | MS:%+.1f%% %s | HVp=%.4f  d_HV=%.2f",
                        inst.upper(), hv_delta,
                        "R+" if hv_r > hv_m else ("M+" if hv_m > hv_r else "="),
                        ms_delta,
                        "R+" if mr < mm else ("M+" if mm < mr else "="),
                        wp_hv.get('p_value', 1), wp_hv.get('cohens_d', 0))

        # ── 效应量汇总 ──
        if es:
            logger.info("-" * 78)
            logger.info("  Cohen's d Effect Size Distribution (across instances)")
            for label in ["HV", "Makespan", "Workload"]:
                e = es[label]
                logger.info("    %9s:  mean|d|=%.3f  median|d|=%.3f  [L:%d M:%d S:%d N:%d]",
                            label, e["mean_d"], e["median_d"],
                            e["n_large"], e["n_medium"], e["n_small"], e["n_negligible"])
            logger.info("    (L=large>0.8  M=medium>0.5  S=small>0.2  N=negligible)")

        logger.info("=" * 78)
        logger.info("")
    except UnicodeEncodeError as e:
        # Windows 控制台编码问题处理
        logger.error("Console encoding error (Windows PowerShell issue): %s", str(e))
        logger.error("Report will be saved to log file instead")
        # 尝试输出简化版本（仅 ASCII）
        try:
            logger.info("RMOEA/D vs MOEA/D - BENCHMARK SUMMARY")
            logger.info("  Instances: %d, Runs: %d", len(instances), n_runs)
            logger.info("  RMOEA/D avgHV: %.4f, MOEA/D avgHV: %.4f", avg_hv_r, avg_hv_m)
        except:
            pass  # 如果还是失败，就跳过控制台输出
        # 确保详细报告写入日志文件
        logger.debug("=" * 90)
        logger.debug("RMOEA/D vs MOEA/D - FULL BENCHMARK REPORT")
        logger.debug("=" * 90)
        logger.debug("Config: Np=%d, Gen=%d, Runs=%d, Instances=%d",
                    first['n_pop'], first['max_gen'], n_runs, len(instances))
        logger.debug("RMOEA/D avgHV = %.4f, MOEA/D avgHV = %.4f, Wins = %d/%d",
                    avg_hv_r, avg_hv_m, n_better, len(instances))

    # ═══════════════════════════════════════════════════════
    # 详细表格 → logger.debug (日志文件)
    # ═══════════════════════════════════════════════════════
    logger.debug("=" * 90)
    logger.debug("  RMOEA/D vs MOEA/D - DETAILED EXPERIMENTAL RESULTS")
    logger.debug("  (Paper: A reinforcement learning based RMOEA/D for bi-objective FFJSP)")
    logger.debug("=" * 90)
    logger.debug("  Configuration: Np=%d, Gen=%d, Runs=%d, Seed=%d",
                 first['n_pop'], first['max_gen'], first['n_runs'], first['base_seed'])

    # 表1: HV对比 + Wilcoxon
    logger.debug("  %s", "─" * 85)
    logger.debug("  TABLE 1: Hypervolume (HV) + Wilcoxon  [mean ± std, N=%d]", n_runs)
    logger.debug("  %-8s %-22s %-22s %-10s %-14s %s", "Inst", "RMOEA/D", "MOEA/D", "Δ%", "Wilcoxon p", "Cohen's d")
    for inst in instances:
        a = agg_results[inst]
        hv_r, hv_rs = a["rmoea_d"]["hv_mean"], a["rmoea_d"]["hv_std"]
        hv_m, hv_ms = a["moea_d"]["hv_mean"], a["moea_d"]["hv_std"]
        imp = ((hv_r - hv_m) / hv_m * 100) if hv_m > 0 else 0
        wp = wh.get("per_instance", {}).get(inst, {})
        logger.debug("  %-8s %.4f ±%.4f     %.4f ±%.4f    %+6.1f%%  p=%.4f %s  d=%.3f",
                     inst.upper(), hv_r, hv_rs, hv_m, hv_ms, imp,
                     wp.get('p_value', 1), wp.get('sig', 'ns'), wp.get('cohens_d', 0))

    # 表2: Makespan + Wilcoxon
    logger.debug("  %s", "─" * 85)
    logger.debug("  TABLE 2: Best Makespan  [mean ± std]")
    logger.debug("  %-8s %-22s %-22s %-10s %-14s %s", "Inst", "RMOEA/D", "MOEA/D", "Δ%", "Wilcoxon p", "Cohen's d")
    for inst in instances:
        a = agg_results[inst]
        mr, mrs = a["rmoea_d"]["best_makespan_mean"], a["rmoea_d"]["best_makespan_std"]
        mm, mms = a["moea_d"]["best_makespan_mean"], a["moea_d"]["best_makespan_std"]
        imp = ((mm - mr) / mm * 100) if mm > 0 else 0
        wp = wm.get("per_instance", {}).get(inst, {})
        logger.debug("  %-8s %.2f ±%.2f         %.2f ±%.2f        %+6.1f%%  p=%.4f %s  d=%.3f",
                     inst.upper(), mr, mrs, mm, mms, imp,
                     wp.get('p_value', 1), wp.get('sig', 'ns'), wp.get('cohens_d', 0))

    # 表3: Fuzzy Makespan TFN
    logger.debug("  %s", "─" * 85)
    logger.debug("  TABLE 3: Fuzzy Makespan TFN (t1 / t2 / t3)")
    logger.debug("  %-8s %-32s %-32s", "Inst", "RMOEA/D", "MOEA/D")
    for inst in instances:
        a = agg_results[inst]
        r = (f"{a['rmoea_d']['fuzzy_makespan_t1_mean']:.1f}/"
             f"{a['rmoea_d']['fuzzy_makespan_t2_mean']:.1f}/"
             f"{a['rmoea_d']['fuzzy_makespan_t3_mean']:.1f}")
        m = (f"{a['moea_d']['fuzzy_makespan_t1_mean']:.1f}/"
             f"{a['moea_d']['fuzzy_makespan_t2_mean']:.1f}/"
             f"{a['moea_d']['fuzzy_makespan_t3_mean']:.1f}")
        logger.debug("  %-8s %-32s %-32s", inst.upper(), r, m)

    # 表4: Fuzzy Workload TFN
    logger.debug("  %s", "─" * 85)
    logger.debug("  TABLE 4: Fuzzy Workload TFN (t1 / t2 / t3)")
    logger.debug("  %-8s %-32s %-32s", "Inst", "RMOEA/D", "MOEA/D")
    for inst in instances:
        a = agg_results[inst]
        r = (f"{a['rmoea_d']['fuzzy_workload_t1_mean']:.1f}/"
             f"{a['rmoea_d']['fuzzy_workload_t2_mean']:.1f}/"
             f"{a['rmoea_d']['fuzzy_workload_t3_mean']:.1f}")
        m = (f"{a['moea_d']['fuzzy_workload_t1_mean']:.1f}/"
             f"{a['moea_d']['fuzzy_workload_t2_mean']:.1f}/"
             f"{a['moea_d']['fuzzy_workload_t3_mean']:.1f}")
        logger.debug("  %-8s %-32s %-32s", inst.upper(), r, m)

    # 表5: 运行时间
    logger.debug("  %s", "─" * 85)
    logger.debug("  TABLE 5: Runtime (seconds)  [mean ± std]")
    for inst in instances:
        a = agg_results[inst]
        tr, trs = a["rmoea_d"]["time_mean"], a["rmoea_d"]["time_std"]
        tm, tms = a["moea_d"]["time_mean"], a["moea_d"]["time_std"]
        logger.debug("  %-8s RMOEA/D: %.2f ±%.2f s      MOEA/D: %.2f ±%.2f s",
                     inst.upper(), tr, trs, tm, tms)

    logger.debug("  %s", "─" * 85)
    logger.debug("  Full statistical results saved in aggregate JSON.")


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
    parser.add_argument("--timeout_per_task", type=float, default=None,
                        help="Per-task timeout in seconds (None = no limit)")
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
        timeout_per_task=args.timeout_per_task,
    )

    # 统计检验
    if args.n_runs > 2:
        stats_tests = run_statistical_tests(agg_results)
    else:
        stats_tests = {"friedman": {}, "wilcoxon_hv": {}, "wilcoxon_makespan": {}, "wilcoxon_workload": {}, "effect_size_summary": {}}

    # 打印报告
    print_report(agg_results, stats_tests)

    # ── 目录结构: results/benchmark/{inst}/  +  results/schedules/{inst}/ ──
    bench_dir = os.path.join(args.output_dir, "benchmark")
    sched_dir = os.path.join(args.output_dir, "schedules")

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

    os.makedirs(bench_dir, exist_ok=True)
    # per-instance aggregate: 文件名含实例名防止子进程并发碰撞
    output_path = os.path.join(bench_dir,
                                f"benchmark_partial_{'_'.join(args.instances)}_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    logger.debug("Aggregate results saved to: %s", output_path)

    # 保存每次运行详细结果（原有格式）
    for inst in args.instances:
        inst_bench_dir = os.path.join(bench_dir, inst)
        inst_sched_dir = os.path.join(sched_dir, inst)
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
                r_dir = os.path.join(inst_bench_dir, algo_dir)
                os.makedirs(r_dir, exist_ok=True)
                r_path = os.path.join(r_dir,
                    f"{inst}_{algo_dir}_Np{args.n_pop}_G{args.max_gen}_run{run_idx}_{ts_run}.json")
                with open(r_path, "w", encoding="utf-8") as f:
                    json.dump(run_summary[algo_key], f, indent=2, ensure_ascii=False)

            # 保存单实例单次summary
            sum_path = os.path.join(inst_bench_dir,
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
                os.makedirs(inst_sched_dir, exist_ok=True)
                full_path = os.path.join(inst_sched_dir,
                    f"{inst}_full_schedule_run{run_idx}_{ts_run}.json")
                with open(full_path, "w", encoding="utf-8") as f:
                    json.dump(full_r, f, indent=2, ensure_ascii=False)
                logger.debug("Full schedule saved: %s", full_path)

    logger.info("Benchmark done: %d runs × %d instances saved to %s/ + %s/",
                args.n_runs, len(args.instances), bench_dir, sched_dir)


if __name__ == "__main__":
    main()