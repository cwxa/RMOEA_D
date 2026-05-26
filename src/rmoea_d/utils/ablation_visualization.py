#!/usr/bin/env python3
"""
消融实验可视化脚本：生成对比图表
图表输出:
  charts/ablation/
    mk01/ mk02/ .../   ← 每实例收敛曲线
    hv_comparison.png, makespan_comparison.png, ...
"""

import os
import json
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

logger = logging.getLogger(__name__)

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150
rcParams['figure.figsize'] = (10, 6)
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10

CHARTS_BASE = 'charts'
ABLATION_DIR = os.path.join(CHARTS_BASE, 'ablation')


def load_ablation_results(results_dir='results/ablation'):
    """加载消融实验结果。支持多文件自动合并（多实例子进程并行产出）"""
    pattern = os.path.join(results_dir, 'ablation_results_*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        return None

    merged = {}
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            # data 格式: {instance: {algo: [runs]}}
            for inst, algos in data.items():
                if inst not in merged:
                    merged[inst] = {}
                for algo, runs in algos.items():
                    if algo not in merged[inst]:
                        merged[inst][algo] = []
                    merged[inst][algo].extend(runs)
        except Exception as e:
            logger.warning("Failed to load %s: %s", fpath, e)

    if not merged:
        return None
    logger.info("Loaded ablation results: %d file(s), %d instances, %d algos",
                len(files), len(merged), sum(len(v) for v in merged.values()))
    return merged


def plot_hv_comparison(results):
    """HV对比柱状图"""
    instances = sorted(results.keys())
    algorithms = ["full", "qpas_only", "rvns_only", "moead"]
    labels = ['Full RMOEA/D', 'Q-PAS Only', 'RVNS Only', 'MOEA/D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

    x = np.arange(len(instances))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, algo in enumerate(algorithms):
        hvs = []
        for instance in instances:
            if algo in results[instance] and results[instance][algo]:
                hvs.append(np.mean([r["final_hv"] for r in results[instance][algo]]))
            else:
                hvs.append(0)
        ax.bar(x + i * width, hvs, width, label=labels[i], color=colors[i], alpha=0.8)

    ax.set_xlabel('Instance')
    ax.set_ylabel('Hypervolume')
    ax.set_title('HV Comparison (Ablation Study)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([inst.upper() for inst in instances], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_DIR, 'hv_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.debug("  [HV] %s/hv_comparison.png", ABLATION_DIR)


def plot_convergence_curves(results):
    """每实例收敛曲线对比图"""
    algorithms = ["full", "qpas_only", "rvns_only", "moead"]
    labels = ['Full RMOEA/D', 'Q-PAS Only', 'RVNS Only', 'MOEA/D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

    for instance in sorted(results.keys()):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, algo in enumerate(algorithms):
            if algo in results[instance] and results[instance][algo]:
                convergence = results[instance][algo][0]["convergence"]
                gens = [c["gen"] for c in convergence]
                hvs = [c["hv"] for c in convergence]
                ax.plot(gens, hvs, label=labels[i], color=colors[i], linewidth=2.5)

        ax.set_xlabel('Generation')
        ax.set_ylabel('Hypervolume')
        ax.set_title(f'{instance.upper()} Convergence Curves (Ablation)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        inst_dir = os.path.join(ABLATION_DIR, instance)
        os.makedirs(inst_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(inst_dir, 'convergence.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.debug("  [Convergence] %s/convergence.png", inst_dir)


def plot_makespan_comparison(results):
    """Makespan对比柱状图"""
    instances = sorted(results.keys())
    algorithms = ["full", "qpas_only", "rvns_only", "moead"]
    labels = ['Full RMOEA/D', 'Q-PAS Only', 'RVNS Only', 'MOEA/D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

    x = np.arange(len(instances))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, algo in enumerate(algorithms):
        vals = []
        for instance in instances:
            if algo in results[instance] and results[instance][algo]:
                vals.append(np.mean([r["best_makespan"] for r in results[instance][algo]]))
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=labels[i], color=colors[i], alpha=0.8)

    ax.set_xlabel('Instance')
    ax.set_ylabel('Best Makespan')
    ax.set_title('Makespan Comparison (Ablation Study)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([inst.upper() for inst in instances], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_DIR, 'makespan_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.debug("  [Makespan] %s/makespan_comparison.png", ABLATION_DIR)


def plot_workload_comparison(results):
    """Workload对比柱状图"""
    instances = sorted(results.keys())
    algorithms = ["full", "qpas_only", "rvns_only", "moead"]
    labels = ['Full RMOEA/D', 'Q-PAS Only', 'RVNS Only', 'MOEA/D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

    x = np.arange(len(instances))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, algo in enumerate(algorithms):
        vals = []
        for instance in instances:
            if algo in results[instance] and results[instance][algo]:
                vals.append(np.mean([r["best_workload"] for r in results[instance][algo]]))
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=labels[i], color=colors[i], alpha=0.8)

    ax.set_xlabel('Instance')
    ax.set_ylabel('Best Workload')
    ax.set_title('Workload Comparison (Ablation Study)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([inst.upper() for inst in instances], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_DIR, 'workload_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.debug("  [Workload] %s/workload_comparison.png", ABLATION_DIR)


def plot_runtime_comparison(results):
    """运行时间对比图"""
    algorithms = ["full", "qpas_only", "rvns_only", "moead"]
    labels = ['Full RMOEA/D', 'Q-PAS Only', 'RVNS Only', 'MOEA/D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

    times = []
    for algo in algorithms:
        algo_times = []
        for instance in results.keys():
            if algo in results[instance] and results[instance][algo]:
                algo_times.extend([r["total_time"] for r in results[instance][algo]])
        times.append(np.mean(algo_times))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, times, color=colors, alpha=0.8)
    ax.set_ylabel('Average Runtime (s)')
    ax.set_title('Runtime Comparison (Ablation Study)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    for i, v in enumerate(times):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_DIR, 'runtime_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.debug("  [Runtime] %s/runtime_comparison.png", ABLATION_DIR)


def plot_pf_size_comparison(results):
    """PF规模对比图"""
    instances = sorted(results.keys())
    algorithms = ["full", "qpas_only", "rvns_only", "moead"]
    labels = ['Full RMOEA/D', 'Q-PAS Only', 'RVNS Only', 'MOEA/D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

    x = np.arange(len(instances))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, algo in enumerate(algorithms):
        vals = []
        for instance in instances:
            if algo in results[instance] and results[instance][algo]:
                vals.append(np.mean([r["pf_size"] for r in results[instance][algo]]))
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=labels[i], color=colors[i], alpha=0.8)

    ax.set_xlabel('Instance')
    ax.set_ylabel('PF Size')
    ax.set_title('PF Size Comparison (Ablation Study)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([inst.upper() for inst in instances], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_DIR, 'pf_size_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.debug("  [PF Size] %s/pf_size_comparison.png", ABLATION_DIR)


def plot_improvement_summary(results):
    """改进幅度汇总图（相对MOEA/D的HV提升%）"""
    instances = sorted(results.keys())
    algorithms = ["full", "qpas_only", "rvns_only"]
    labels = ['Full RMOEA/D', 'Q-PAS Only', 'RVNS Only']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    x = np.arange(len(instances))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, algo in enumerate(algorithms):
        improvements = []
        for instance in instances:
            if algo in results[instance] and "moead" in results[instance]:
                algo_hv = np.mean([r["final_hv"] for r in results[instance][algo]])
                moead_hv = np.mean([r["final_hv"] for r in results[instance]["moead"]])
                imp = ((algo_hv - moead_hv) / moead_hv) * 100 if moead_hv > 0 else 0
                improvements.append(imp)
            else:
                improvements.append(0)
        ax.bar(x + i * width, improvements, width, label=labels[i], color=colors[i], alpha=0.8)

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Instance')
    ax.set_ylabel('HV Improvement (%)')
    ax.set_title('Performance Improvement Over MOEA/D (Ablation Study)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([inst.upper() for inst in instances], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_DIR, 'improvement_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.debug("  [Improvement] %s/improvement_summary.png", ABLATION_DIR)


def main():
    results = load_ablation_results()
    if not results:
        logger.error("No ablation results found!")
        return

    logger.info("Loaded ablation results: %d instances", len(results))
    os.makedirs(ABLATION_DIR, exist_ok=True)

    logger.debug("=== Generating Ablation Comparison Charts ===")
    plot_hv_comparison(results)
    plot_makespan_comparison(results)
    plot_workload_comparison(results)
    plot_runtime_comparison(results)
    plot_pf_size_comparison(results)
    plot_improvement_summary(results)

    logger.debug("=== Generating Convergence Curves ===")
    plot_convergence_curves(results)

    logger.info("Ablation charts done: %s/", ABLATION_DIR)


if __name__ == '__main__':
    main()