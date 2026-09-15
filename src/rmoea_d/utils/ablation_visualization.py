#!/usr/bin/env python3
"""
消融实验可视化脚本：RMOEA/D 组件贡献分析（补充图表）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
本模块负责生成 analysis_report.py 中未覆盖的消融实验图表：
  • 收敛曲线 (convergence curves)
  • Workload 对比
  • PF 规模对比
  • 改进幅度汇总

HV / Makespan / Runtime 对比由 analysis_report.py 统一生成，
此处直接复用，避免重复造轮子。
"""

import os
import json
import glob
import logging
import numpy as np

# ── 强制使用非 GUI 后端，确保线程安全渲染 ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 共享绘图工具 ──
try:
    from .plot_helpers import (
        C_FULL, C_QPAS, C_RVNS, C_MOEAD,
        C_IMPROVE, C_DECLINE, C_GRAY, C_GRID,
        set_data_timestamp, source_footer, style_ax,
    )
except ImportError:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_helpers import (
        C_FULL, C_QPAS, C_RVNS, C_MOEAD,
        C_IMPROVE, C_DECLINE, C_GRAY, C_GRID,
        set_data_timestamp, source_footer, style_ax,
    )

logger = logging.getLogger(__name__)

# ── 消融实验专用常量 ──
ALGORITHMS   = ["full", "qpas_only", "rvns_only", "moead"]
ALGO_LABELS  = ['Full RMOEA/D', 'Q-PAS Only', 'RVNS Only', 'MOEA/D']
ALGO_COLORS  = [C_FULL, C_QPAS, C_RVNS, C_MOEAD]

CHARTS_BASE  = 'charts'
ABLATION_DIR = os.path.join(CHARTS_BASE, 'ablation')


def load_ablation_results(results_dir='results/ablation'):
    """加载消融实验结果。支持多文件自动合并（多实例子进程并行产出）
    目录结构: results/ablation/{instance}/ablation_results_{timestamp}.json
    """
    pattern = os.path.join(results_dir, '*', 'ablation_results_*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        pattern = os.path.join(results_dir, 'ablation_results_*.json')
        files = sorted(glob.glob(pattern))
    if not files:
        return None

    merged = {}
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
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
    if files:
        set_data_timestamp(os.path.basename(files[-1]).replace('ablation_results_', '').replace('.json', ''))
    logger.info("Loaded ablation results: %d file(s), %d instances, %d algos",
                len(files), len(merged), sum(len(v) for v in merged.values()))
    return merged


def _get_multi_run(values):
    """聚合多轮运行数据：返回 (均值, 标准差, 运行次数)"""
    if not values:
        return 0, 0, 0
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, len(arr)


def _plot_metric_bar(results, metric_key, ylabel, title, filename):
    """通用指标柱状图（Makespan / Workload / PF Size 等）"""
    instances = sorted(results.keys())
    x = np.arange(len(instances))
    width = 0.2
    has_error = any(len(results[inst].get(algo, [])) > 1
                    for inst in instances for algo in ALGORITHMS if algo in results[inst])

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, algo in enumerate(ALGORITHMS):
        vals, errs = [], []
        for inst in instances:
            runs = results[inst].get(algo, [])
            run_vals = [r.get(metric_key, 0) for r in runs] if runs else [0]
            mean_val, std_val, _ = _get_multi_run(run_vals)
            vals.append(mean_val)
            errs.append(std_val)
        ax.bar(x + i * width, vals, width, label=ALGO_LABELS[i],
               color=ALGO_COLORS[i], alpha=0.85, edgecolor='white', linewidth=0.5)
        if has_error and any(e > 0 for e in errs):
            ax.errorbar(x + i * width, vals, yerr=errs, fmt='none',
                        ecolor=C_GRAY, elinewidth=1.2, capsize=3, alpha=0.7)

    # 改进百分比（Full vs MOEA/D，越小越好则 check 方向翻转）
    for idx, inst in enumerate(instances):
        full_runs = results[inst].get("full", [])
        moead_runs = results[inst].get("moead", [])
        fv = [r.get(metric_key, 0) for r in full_runs]
        mv = [r.get(metric_key, 0) for r in moead_runs]
        if fv and mv:
            f_mean, m_mean = np.mean(fv), np.mean(mv)
            if m_mean > 0:
                imp = ((m_mean - f_mean) / m_mean) * 100
                clr = C_IMPROVE if imp > 0 else C_DECLINE
                ax.annotate(f'{imp:+.1f}%', (x[idx], max(f_mean, m_mean)),
                            textcoords="offset points", xytext=(0, 5),
                            ha='center', fontsize=7.5, color=clr, fontweight='bold')

    ax.set_xlabel('Instance')
    ax.set_ylabel(ylabel)
    nr = max(len(results[inst].get(algo, [])) for inst in instances for algo in ALGORITHMS if algo in results[inst])
    err_note = f' (N={nr}, mean ± std)' if has_error and nr > 1 else ''
    ax.set_title(f'{title} — Ablation Study{err_note}')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([inst.upper() for inst in instances], rotation=45, ha='right')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
    style_ax(ax)
    source_footer(fig, f'Ablation {title} · N={nr}')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(ABLATION_DIR, exist_ok=True)
    plt.savefig(os.path.join(ABLATION_DIR, filename), dpi=300)
    plt.close()
    logger.debug("  [%s] %s/%s", title, ABLATION_DIR, filename)


def plot_convergence_curves(results):
    """每实例收敛曲线对比图 — 多运行均值±标准差带状区域"""
    for instance in sorted(results.keys()):
        fig, ax = plt.subplots(figsize=(10, 6))
        n_runs_total = 0

        for i, algo in enumerate(ALGORITHMS):
            if algo not in results[instance] or not results[instance][algo]:
                continue

            n_runs = len(results[instance][algo])
            n_runs_total = max(n_runs_total, n_runs)

            all_gens = []
            all_hvs = []
            min_len = float('inf')
            for run in results[instance][algo]:
                conv = run.get("convergence", run.get("history", []))
                if not conv or not isinstance(conv, list):
                    continue
                gens = [c.get("gen", c.get("generation", idx)) for idx, c in enumerate(conv)]
                hvs = [c.get("hv", c.get("hypervolume", 0)) for c in conv]
                all_gens.append(gens)
                all_hvs.append(hvs)
                min_len = min(min_len, len(gens))

            if min_len == float('inf') or min_len <= 0:
                continue

            trimmed_hvs = np.array([h[:min_len] for h in all_hvs])
            mean_hvs = np.mean(trimmed_hvs, axis=0)
            std_hvs = np.std(trimmed_hvs, axis=0, ddof=1) if n_runs > 1 else np.zeros_like(mean_hvs)
            gens = all_gens[0][:min_len]

            ax.plot(gens, mean_hvs, label=f'{ALGO_LABELS[i]} (N={n_runs})',
                    color=ALGO_COLORS[i], linewidth=2.2, alpha=0.9)
            if n_runs > 1:
                ax.fill_between(gens, mean_hvs - std_hvs, mean_hvs + std_hvs,
                                color=ALGO_COLORS[i], alpha=0.08)
            ax.scatter([gens[-1]], [mean_hvs[-1]], c=ALGO_COLORS[i], s=25, zorder=5)

        ax.set_xlabel('Generation')
        ax.set_ylabel('Hypervolume (HV)')
        ax.set_title(f'{instance.upper()} — Convergence Curves (Ablation)'
                     f'{" — mean ± std" if n_runs_total > 1 else ""}')
        ax.legend(loc='lower right', framealpha=0.9, fontsize=8)
        style_ax(ax)

        inst_dir = os.path.join(ABLATION_DIR, instance)
        os.makedirs(inst_dir, exist_ok=True)
        source_footer(fig, f'Ablation · {instance.upper()} · N={n_runs_total}')
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        plt.savefig(os.path.join(inst_dir, 'convergence.png'), dpi=300)
        plt.close()
        logger.debug("  [Convergence] %s/convergence.png", inst_dir)


def plot_workload_comparison(results):
    """消融实验 Workload 对比柱状图"""
    _plot_metric_bar(results, "best_workload",
                     'Best Workload (Crisp)', 'Workload', 'workload_comparison.png')


def plot_pf_size_comparison(results):
    """消融实验 PF 规模对比柱状图"""
    _plot_metric_bar(results, "pf_size",
                     'PF Size (|PF|)', 'Pareto Front Size', 'pf_size_comparison.png')


def plot_improvement_summary(results):
    """改进幅度汇总图（相对 MOEA/D 的 HV 提升百分比）"""
    instances = sorted(results.keys())
    ablation_algos = ["full", "qpas_only", "rvns_only"]
    ablation_labels = ['Full RMOEA/D', 'Q-PAS Only', 'RVNS Only']
    ablation_colors = [C_FULL, C_QPAS, C_RVNS]

    x = np.arange(len(instances))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 7))

    bars_list = []
    for i, (algo, label, color) in enumerate(zip(ablation_algos, ablation_labels, ablation_colors)):
        improvements = []
        for inst in instances:
            if algo in results[inst] and "moead" in results[inst]:
                algo_runs = results[inst][algo]
                moead_runs = results[inst]["moead"]
                algo_hv = np.mean([r.get("final_hv", 0) for r in algo_runs]) if algo_runs else 0
                moead_hv = np.mean([r.get("final_hv", 0) for r in moead_runs]) if moead_runs else 0
                imp = ((algo_hv - moead_hv) / moead_hv) * 100 if moead_hv > 0 else 0
                improvements.append(imp)
            else:
                improvements.append(0)
        bars = ax.bar(x + i * width, improvements, width, label=label,
                      color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
        bars_list.append((bars, improvements))

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('Instance')
    ax.set_ylabel('HV Improvement over MOEA/D (%)')
    ax.set_title('Performance Improvement — Ablation Study\n(+ = Outperforms MOEA/D)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([inst.upper() for inst in instances], rotation=45, ha='right')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=8)

    for bars, improvements in bars_list:
        for bar, v in zip(bars, improvements):
            clr = C_IMPROVE if v > 0 else C_DECLINE
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.3 if v >= 0 else -0.3),
                    f'{v:+.1f}%',
                    ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=6.5, color=clr, fontweight='bold', rotation=90)

    style_ax(ax)
    source_footer(fig, 'Ablation Improvement vs MOEA/D')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(ABLATION_DIR, exist_ok=True)
    plt.savefig(os.path.join(ABLATION_DIR, 'improvement_summary.png'), dpi=300)
    plt.close()
    logger.debug("  [Improvement] %s/improvement_summary.png", ABLATION_DIR)


def main():
    results = load_ablation_results()
    if not results:
        logger.error("No ablation results found!")
        return

    logger.info("Loaded ablation results: %d instances", len(results))
    os.makedirs(ABLATION_DIR, exist_ok=True)

    logger.info("=== Generating Ablation Charts ===")
    # 本模块负责生成 analysis_report.py 未覆盖的补充图表
    # (HV / Makespan / Runtime 由 analysis_report.py 统一生成)
    plot_workload_comparison(results)
    plot_pf_size_comparison(results)
    plot_improvement_summary(results)
    plot_convergence_curves(results)

    logger.info("Ablation charts done — saved to %s/", ABLATION_DIR)


if __name__ == '__main__':
    main()
