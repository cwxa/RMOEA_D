#!/usr/bin/env python3
"""
消融实验脚本：对比不同算法变体的性能
- 完整 RMOEA/D（Q-PAS + RVNS）
- 仅 Q-PAS（无 RVNS）
- 仅 RVNS（无 Q-PAS）
- 纯 MOEA/D（无 Q-PAS，无 RVNS）

.. deprecated::
    推荐使用 experiment.py 进行统一实验，单次运行同时产出 benchmark + ablation 数据。
    此脚本保留用于向后兼容和独立调试。
    run_all.py v4.0 已统一到 experiment.py，不再调用此脚本。
"""

import argparse
import json
import os
import time
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

# ── 禁止 BLAS/MKL 内部多线程，避免与进程池冲突 ──
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 添加路径
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rmoea_d.algorithm import RMOEAD
from rmoea_d.moead_baseline import MOEADBaseline
from rmoea_d.utils.logger_setup import setup_logging
import logging

logger = logging.getLogger(__name__)


def run_single_experiment(instance_name, n_pop, max_gen, seed, algorithm_type, 
                          ql_alpha=0.4, ql_gamma=0.6, ql_epsilon=0.8, ql_actions=[5, 10, 15, 20],
                          crossover_rate=0.8, data_dir="data"):
    """运行单次实验"""
    start_time = time.perf_counter()
    
    if algorithm_type == "full":
        # 完整 RMOEA/D（Q-PAS + RVNS）
        solver = RMOEAD(
            instance_name=instance_name,
            n_pop=n_pop,
            max_gen=max_gen,
            crossover_rate=crossover_rate,
            seed=seed,
            data_dir=data_dir,
            ql_alpha=ql_alpha,
            ql_gamma=ql_gamma,
            ql_epsilon=ql_epsilon,
            ql_actions=ql_actions,
            enable_rvns=True
        )
    elif algorithm_type == "qpas_only":
        # 仅 Q-PAS（无 RVNS）
        solver = RMOEAD(
            instance_name=instance_name,
            n_pop=n_pop,
            max_gen=max_gen,
            crossover_rate=crossover_rate,
            seed=seed,
            data_dir=data_dir,
            ql_alpha=ql_alpha,
            ql_gamma=ql_gamma,
            ql_epsilon=ql_epsilon,
            ql_actions=ql_actions,
            enable_rvns=False
        )
    elif algorithm_type == "rvns_only":
        # 仅 RVNS（无 Q-PAS，固定 T=10）
        solver = RMOEAD(
            instance_name=instance_name,
            n_pop=n_pop,
            max_gen=max_gen,
            crossover_rate=crossover_rate,
            seed=seed,
            data_dir=data_dir,
            fixed_T=10,  # 固定 T
            enable_rvns=True
        )
    elif algorithm_type == "moead":
        # 纯 MOEA/D（无 Q-PAS，无 RVNS）
        solver = MOEADBaseline(
            instance_name=instance_name,
            n_pop=n_pop,
            max_gen=max_gen,
            crossover_rate=crossover_rate,
            seed=seed,
            data_dir=data_dir
        )
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")
    
    results = solver.solve()
    elapsed_time = time.perf_counter() - start_time
    
    # 提取关键数据
    result_data = {
        "algorithm": algorithm_type,
        "instance": instance_name,
        "n_pop": n_pop,
        "max_gen": max_gen,
        "seed": seed,
        "total_time": elapsed_time,
        "final_hv": results["final_hv"],
        "pf_size": len(results["final_pf"]),
        "best_makespan": min(p["Makespan"] if isinstance(p, dict) else p[0] for p in results["final_pf"]),
        "best_workload": min(p["Workload"] if isinstance(p, dict) else p[1] for p in results["final_pf"]),
        "avg_makespan": np.mean([p["Makespan"] if isinstance(p, dict) else p[0] for p in results["final_pf"]]),
        "avg_workload": np.mean([p["Workload"] if isinstance(p, dict) else p[1] for p in results["final_pf"]]),
        "convergence": [],
        "parameters": {
            "ql_alpha": ql_alpha,
            "ql_gamma": ql_gamma,
            "ql_epsilon": ql_epsilon,
            "ql_actions": ql_actions,
            "crossover_rate": crossover_rate
        }
    }
    
    # 提取收敛曲线
    if "history" in results:
        for h in results["history"]:
            result_data["convergence"].append({
                "gen": h["gen"],
                "hv": h.get("hv", 0),
                "pf_size": h.get("pf_size", 0),
                "best_makespan": h.get("best_makespan", 0),
                "best_workload": h.get("best_workload", 0)
            })
    
    logger.debug("Experiment completed: %s on %s, HV=%.6f, Time=%.2fs",
                 algorithm_type, instance_name, result_data['final_hv'], elapsed_time)
    return result_data


# ═══════════════════════════════════════════════════════════
# 消融实验 并行版
# ═══════════════════════════════════════════════════════════

def _ablation_worker(args_tuple):
    """进程池 worker：运行单次消融实验 (picklable 顶层函数)"""
    (instance_name, n_pop, max_gen, seed, algorithm_type,
     ql_alpha, ql_gamma, ql_epsilon, ql_actions,
     crossover_rate, data_dir, run_idx) = args_tuple
    result = run_single_experiment(
        instance_name=instance_name, n_pop=n_pop, max_gen=max_gen,
        seed=seed, algorithm_type=algorithm_type,
        ql_alpha=ql_alpha, ql_gamma=ql_gamma, ql_epsilon=ql_epsilon,
        ql_actions=list(ql_actions), crossover_rate=crossover_rate,
        data_dir=data_dir)
    result["run"] = run_idx
    return instance_name, algorithm_type, run_idx, result


def run_ablation_study(instances, n_pop, max_gen, n_runs, algorithms, output_dir,
                       ql_alpha=0.4, ql_gamma=0.6, ql_epsilon=0.8,
                       ql_actions=None, crossover_rate=0.8, data_dir="data",
                       n_workers=None, exp_id=None, timeout_per_task=None):
    """并行运行完整的消融实验。
    exp_id: 统一实验ID，贯穿全流程文件名（None则自动生成时间戳）。
    timeout_per_task: 单个任务最大执行时间（秒），超时后将跳过该任务并记录警告。"""
    if ql_actions is None:
        ql_actions = [5, 10, 15, 20]
    if n_workers is None:
        # 优先用环境变量感知外层并发数，否则默认用全部 CPU
        parent_workers = int(os.environ.get("RMOEA_PARENT_WORKERS", "0"))
        if parent_workers > 0:
            total_cpus = os.cpu_count() or 4
            n_workers = max(1, min(n_runs * len(algorithms), total_cpus // parent_workers))
        else:
            n_workers = min(os.cpu_count() or 4, n_runs * len(algorithms))

    timestamp = exp_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {inst: {algo: [] for algo in algorithms} for inst in instances}

    for instance in instances:
        # ── 生成该实例的所有 (algorithm, run) 任务 ──
        tasks = []
        for algorithm in algorithms:
            for run_idx in range(n_runs):
                seed = 42 + run_idx
                tasks.append((
                    instance, n_pop, max_gen, seed, algorithm,
                    ql_alpha, ql_gamma, ql_epsilon, tuple(ql_actions),
                    crossover_rate, data_dir, run_idx,
                ))

        logger.debug("[%s] %d tasks (%d algos x %d runs) -> %d workers",
                     instance, len(tasks), len(algorithms), n_runs, n_workers)

        # ── 并行执行 ──
        completed = 0
        timed_out = 0
        t0 = time.time()
        milestone_interval = max(1, len(tasks) // 4)  # 每25%报告一次
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_ablation_worker, t): t for t in tasks}
            for future in as_completed(futures):
                try:
                    inst, algo, run_idx, result = future.result(timeout=timeout_per_task)
                except FutureTimeoutError:
                    task = futures[future]
                    timed_out += 1
                    logger.warning("[%d/%d] Timeout: %s %s run %d > %.0fs, skipping",
                                   completed + timed_out, len(tasks),
                                   task[0], task[4], task[11], timeout_per_task)
                    future.cancel()
                    continue
                except Exception as e:
                    task = futures[future]
                    timed_out += 1
                    logger.error("[%d/%d] Error: %s %s run %d: %s, skipping",
                                 completed + timed_out, len(tasks),
                                 task[0], task[4], task[11], e)
                    continue
                all_results[inst][algo].append(result)
                completed += 1
                elapsed = time.time() - t0
                logger.debug("[%d/%d] %s run %d HV=%.4f Time=%.1fs [%.0fs]",
                            completed, len(tasks), algo, run_idx + 1,
                            result['final_hv'], result['total_time'], elapsed)
                if completed % milestone_interval == 0 or completed == len(tasks):
                    logger.info("Ablation %s: %d/%d tasks [%.0fs]", instance, completed, len(tasks), elapsed)

            if timed_out > 0:
                logger.warning("Ablation %s: %d/%d tasks timed out or errored", instance, timed_out, len(tasks))

            # ── 每完成一个实例就保存中间结果 ──
            save_results(all_results, output_dir, timestamp, instance_tag=instance)

    # 保存最终结果 + 生成报告 (多实例一起跑时才写入统一JSON)
    if len(instances) > 1:
        save_results(all_results, output_dir, timestamp)
    generate_report(all_results, output_dir, timestamp,
                    instance=instances[0] if len(instances) == 1 else None)

    return all_results


def save_results(results, output_dir, timestamp, instance_tag=None):
    """保存实验结果。instance_tag指定时写入 output_dir/{instance_tag}/ 子目录"""
    if instance_tag:
        out = os.path.join(output_dir, instance_tag)
    else:
        out = output_dir
    os.makedirs(out, exist_ok=True)
    filename = os.path.join(out, f"ablation_results_{timestamp}.json")
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {filename}")


def generate_report(results, output_dir, timestamp, instance=None):
    """生成实验报告。instance指定时写入 output_dir/{instance}/ 子目录"""
    report_lines = [
        "=" * 80,
        "RMOEA/D 消融实验报告",
        "=" * 80,
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "=" * 80,
        "一、实验配置",
        "=" * 80,
    ]
    
    # 提取配置信息
    first_instance = list(results.keys())[0]
    first_algo = list(results[first_instance].keys())[0]
    first_run = results[first_instance][first_algo][0]
    
    report_lines.extend([
        f"种群大小: {first_run['n_pop']}",
        f"最大代数: {first_run['max_gen']}",
        f"重复次数: {len(results[first_instance][first_algo])}",
        f"交叉率: {first_run['parameters']['crossover_rate']}",
        "",
        "参与算法:",
        "\n".join([f"  - {algo}" for algo in results[first_instance].keys()]),
        "",
        "测试实例:",
        "\n".join([f"  - {inst}" for inst in results.keys()]),
        "",
        "=" * 80,
        "二、实验结果汇总",
        "=" * 80,
        "",
    ])
    
    # 表格表头
    report_lines.append(f"{'实例':<8} {'算法':<12} {'HV (mean±std)':<20} {'PF Size':<10} {'Time (s)':<12} {'Best Makespan':<15}")
    report_lines.append("-" * 80)
    
    # 填充数据
    for instance, algos in results.items():
        for algo, runs in algos.items():
            hvs = [r["final_hv"] for r in runs]
            pf_sizes = [r["pf_size"] for r in runs]
            times = [r["total_time"] for r in runs]
            best_makespans = [r["best_makespan"] for r in runs]
            
            report_lines.append(
                f"{instance:<8} {algo:<12} {np.mean(hvs):.4f}±{np.std(hvs):.4f} {np.mean(pf_sizes):.1f}±{np.std(pf_sizes):.1f} "
                f"{np.mean(times):.2f}±{np.std(times):.2f} {np.mean(best_makespans):.2f}±{np.std(best_makespans):.2f}"
            )
    
    report_lines.extend([
        "",
        "=" * 80,
        "三、统计检验分析",
        "=" * 80,
        "",
    ])

    for instance in results.keys():
        avail = [a for a in ["full", "qpas_only", "rvns_only", "moead"] if a in results[instance]]
        if len(avail) < 2:
            continue
        report_lines.append(f"\n【{instance}】")
        for i, a1 in enumerate(avail):
            for a2 in avail[i + 1:]:
                try:
                    hvs1 = [r["final_hv"] for r in results[instance][a1]]
                    hvs2 = [r["final_hv"] for r in results[instance][a2]]
                    if len(hvs1) < 3 or len(hvs2) < 3:
                        continue
                    from scipy.stats import wilcoxon
                    min_len = min(len(hvs1), len(hvs2))
                    # 若两样本完全相同，se=0 会导致除零 RuntimeWarning，直接返回 p=1.0
                    if np.allclose(hvs1[:min_len], hvs2[:min_len]):
                        p_val = 1.0
                    else:
                        _, p_val = wilcoxon(hvs1[:min_len], hvs2[:min_len], zero_method='zsplit')
                    d1 = np.array(hvs1[:min_len])
                    d2 = np.array(hvs2[:min_len])
                    pooled = np.sqrt((np.std(d1, ddof=1) ** 2 + np.std(d2, ddof=1) ** 2) / 2)
                    d = (np.mean(d1) - np.mean(d2)) / pooled if pooled > 0 else 0
                    sig = "★" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns")
                    report_lines.append(
                        f"  {a1} vs {a2}: p={p:.4f} {sig}, Cohen's d={d:+.3f}"
                    )
                except Exception:
                    pass

    report_lines.extend([
        "",
        "=" * 80,
        "四、算法对比",
        "=" * 80,
        "",
    ])
    
    # 计算相对于MOEA/D的提升
    for instance in results.keys():
        if "moead" in results[instance]:
            moead_hv = np.mean([r["final_hv"] for r in results[instance]["moead"]])
        else:
            moead_hv = None

        report_lines.append(f"\n【{instance}】")
        for algo in ["full", "qpas_only", "rvns_only"]:
            if algo in results[instance]:
                algo_hv = np.mean([r["final_hv"] for r in results[instance][algo]])
                if moead_hv is not None:
                    improvement = ((algo_hv - moead_hv) / moead_hv) * 100
                    report_lines.append(f"  {algo}: HV={algo_hv:.4f} ({improvement:+.2f}%)")
                else:
                    report_lines.append(f"  {algo}: HV={algo_hv:.4f}")
    
    report_lines.extend([
        "",
        "=" * 80,
        "五、结论",
        "=" * 80,
        ""
    ])
    
    # 总结
    all_hvs = {}
    for algo in ["full", "qpas_only", "rvns_only", "moead"]:
        all_hvs[algo] = []
        for instance in results.keys():
            if algo in results[instance]:
                all_hvs[algo].extend([r["final_hv"] for r in results[instance][algo]])

    # 筛选有数据的算法
    valid_algos = {k: v for k, v in all_hvs.items() if v}
    best_algo = max(valid_algos.keys(), key=lambda x: np.mean(valid_algos[x])) if valid_algos else "N/A"
    if best_algo != "N/A":
        report_lines.append(f"最佳算法: {best_algo}")
        report_lines.append(f"平均 HV: {np.mean(valid_algos[best_algo]):.4f}")

    # 全局统计检验 (pooled across all instances)
    report_lines.append("")
    report_lines.append("全局统计检验 (pooled across instances):")
    avail_global = sorted(valid_algos.keys())
    for i, a1 in enumerate(avail_global):
        for a2 in avail_global[i + 1:]:
            try:
                hvs1 = valid_algos[a1]
                hvs2 = valid_algos[a2]
                if len(hvs1) < 3 or len(hvs2) < 3:
                    continue
                from scipy.stats import wilcoxon
                min_len = min(len(hvs1), len(hvs2))
                # 若两样本完全相同，se=0 会导致除零 RuntimeWarning，直接返回 p=1.0
                if np.allclose(hvs1[:min_len], hvs2[:min_len]):
                    p_val = 1.0
                else:
                    _, p_val = wilcoxon(hvs1[:min_len], hvs2[:min_len], zero_method='zsplit')
                d1 = np.array(hvs1[:min_len])
                d2 = np.array(hvs2[:min_len])
                pooled = np.sqrt((np.std(d1, ddof=1) ** 2 + np.std(d2, ddof=1) ** 2) / 2)
                d = (np.mean(d1) - np.mean(d2)) / pooled if pooled > 0 else 0
                sig = "★" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns")
                report_lines.append(
                    f"  {a1} vs {a2}: p={p_val:.4f} {sig}, Cohen's d={d:+.3f}")
            except Exception:
                pass
    report_lines.append("")
    
    report = "\n".join(report_lines)
    
    # 保存报告
    report_out = os.path.join(output_dir, instance) if instance else output_dir
    os.makedirs(report_out, exist_ok=True)
    report_path = os.path.join(report_out, f"ablation_report_{timestamp}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.debug(report.replace('%', '%%'))  # 详细报告写入日志文件
    logger.info("Ablation report saved to %s", report_path)


def main():
    parser = argparse.ArgumentParser(description="RMOEA/D Ablation Study")
    
    # 实验配置
    parser.add_argument("--instances", type=str, nargs="+", default=["Mk01", "Mk02"],
                        help="Instance names")
    parser.add_argument("--n_pop", type=int, default=100, help="Population size")
    parser.add_argument("--max_gen", type=int, default=100, help="Maximum generations")
    parser.add_argument("--n_runs", type=int, default=3, help="Number of repetitions")
    parser.add_argument("--output_dir", type=str, default="results/ablation",
                        help="Output directory")
    
    # 算法选择
    parser.add_argument("--algorithms", type=str, nargs="+", 
                        default=["full", "qpas_only", "rvns_only", "moead"],
                        choices=["full", "qpas_only", "rvns_only", "moead"],
                        help="Algorithms to compare")
    
    # 参数配置
    parser.add_argument("--ql_alpha", type=float, default=0.4, help="Q-learning alpha")
    parser.add_argument("--ql_gamma", type=float, default=0.6, help="Q-learning gamma")
    parser.add_argument("--ql_epsilon", type=float, default=0.8, help="Q-learning epsilon")
    parser.add_argument("--ql_actions", type=int, nargs="+", default=[5, 10, 15, 20],
                        help="Q-learning candidate T values")
    parser.add_argument("--crossover_rate", type=float, default=0.8, help="Crossover rate")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Parallel workers (default: auto=min(cpu_count, n_runs*4))")
    parser.add_argument("--timeout_per_task", type=float, default=None,
                        help="Per-task timeout in seconds (None = no limit)")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(log_dir="logs")
    
    # 运行消融实验
    logger.info("Starting ablation study...")
    results = run_ablation_study(
        instances=args.instances,
        n_pop=args.n_pop,
        max_gen=args.max_gen,
        n_runs=args.n_runs,
        algorithms=args.algorithms,
        output_dir=args.output_dir,
        ql_alpha=args.ql_alpha,
        ql_gamma=args.ql_gamma,
        ql_epsilon=args.ql_epsilon,
        ql_actions=args.ql_actions,
        crossover_rate=args.crossover_rate,
        data_dir=args.data_dir,
        n_workers=args.n_workers,
        timeout_per_task=args.timeout_per_task,
    )

    # 日志: 消融实验完成；可视化由 ablation_visualization.py 统一负责
    logger.info("Ablation study completed! (Charts via ablation_visualization.py)")


if __name__ == "__main__":
    main()