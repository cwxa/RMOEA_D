#!/usr/bin/env python3
"""
RMOEA/D 完整实验分析可视化系统
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 每个图表均为独立文件，输出到 charts/benchmark/ 和 charts/ablation/
• 支持：基准实验 (RMOEA/D vs MOEA/D) + 消融实验 (Full vs Q-PAS Only vs RVNS Only)
• 统计检验：Wilcoxon 符号秩检验、Friedman 检验、Cohen's d 效应量
• 所有输出均为高分辨率 PNG 图表 (300 DPI)
"""

import os
import json
import glob
import logging
import time
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── 共享绘图工具（跨平台字体、配色、轴样式、脚注） ──
# 兼容模块导入 (from .plot_helpers) 和脚本直跑 (from plot_helpers)
try:
    from .plot_helpers import (
        C_RMOEA, C_MOEA, C_MOEAD, C_FULL, C_QPAS, C_RVNS,
        C_IMPROVE, C_DECLINE, C_GRAY, C_GRID,
        source_footer, style_ax,
    )
except ImportError:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_helpers import (
        C_RMOEA, C_MOEA, C_MOEAD, C_FULL, C_QPAS, C_RVNS,
        C_IMPROVE, C_DECLINE, C_GRAY, C_GRID,
        source_footer, style_ax,
    )

logger = logging.getLogger(__name__)

# ═══════════════════════════════════ 算法/配色映射 ═══════════════════════════════

BENCHMARK_ALGO_LABELS = {'rmoea_d': 'RMOEA/D', 'moea_d': 'MOEA/D'}
BENCHMARK_ALGO_COLORS = {'rmoea_d': C_RMOEA, 'moea_d': C_MOEA}
ABLATION_ALGO_LABELS = {'full': 'Full RMOEA/D', 'qpas_only': 'Q-PAS Only', 'rvns_only': 'RVNS Only', 'moead': 'MOEA/D'}
ABLATION_ALGO_COLORS = {'full': C_FULL, 'qpas_only': C_QPAS, 'rvns_only': C_RVNS, 'moead': C_MOEAD}


def _save_and_close(fig, filepath, log_msg=None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    if log_msg:
        logger.info(log_msg + " -> %s", os.path.relpath(filepath))


def _make_figure(title, figsize=(12, 7), extra=''):
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.97)
    source_footer(fig, extra)
    return fig, ax


# ═══════════════════════════════════ 统计检验 ═══════════════════════════════════

def _wilcoxon_test(a, b, alpha=0.05):
    from scipy.stats import wilcoxon
    import numpy as np
    min_len = min(len(a), len(b))
    if min_len < 3:
        return None
    try:
        # 若两样本完全相同，Wilcoxon 的 se=0 会导致 z=(r_plus-mn)/se 除零警告
        # 此时差异不显著，直接返回 p=1.0
        if np.allclose(a[:min_len], b[:min_len]):
            return {"statistic": 0.0, "p_value": 1.0, "significant": False, "sig": "n.s."}
        stat, p = wilcoxon(a[:min_len], b[:min_len], zero_method='zsplit')
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        return {"statistic": float(stat), "p_value": float(p), "significant": p < alpha, "sig": sig}
    except Exception:
        return None


def _cohens_d(a, b):
    d1 = np.array(a, dtype=float)
    d2 = np.array(b, dtype=float)
    pooled = np.sqrt((np.std(d1, ddof=1) ** 2 + np.std(d2, ddof=1) ** 2) / 2)
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(d1) - np.mean(d2)) / pooled)


def _friedman_test(*groups):
    from scipy.stats import friedmanchisquare
    try:
        stat, p = friedmanchisquare(*groups)
        return {"statistic": float(stat), "p_value": float(p), "significant": p < 0.05}
    except Exception:
        return None


def _effect_size_label(d):
    if abs(d) < 0.2:
        return "negligible"
    elif abs(d) < 0.5:
        return "small"
    elif abs(d) < 0.8:
        return "medium"
    else:
        return "large"


# ═══════════════════════════════════ 数据加载 ═══════════════════════════════════

def load_benchmark_data(results_dir='results/benchmark'):
    pattern = os.path.join(results_dir, "benchmark_aggregate_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None, None, None
    latest = files[-1]
    with open(latest, 'r') as f:
        data = json.load(f)
    instances = data.get('instances', {})
    stats_tests = data.get('statistical_tests', {})
    exp_id = data.get('exp_id', 'unknown')
    return instances, stats_tests, exp_id


def load_ablation_data(results_dir='results/ablation'):
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
    return merged if merged else None


def load_per_run_data(results_dir='results/benchmark'):
    all_runs = {}
    for d in sorted(glob.glob(os.path.join(results_dir, 'mk*', ''))):
        inst = os.path.basename(os.path.normpath(d))
        if not inst.startswith('mk'):
            continue
        all_runs[inst] = {}
        for algo_dir, algo_key in [('RMOEA_D', 'rmoea_d'), ('MOEA_D', 'moea_d')]:
            ad = os.path.join(d, algo_dir)
            run_files = sorted(glob.glob(os.path.join(ad, f'{inst}_{algo_dir}_*.json')))
            runs = []
            for rf in run_files:
                try:
                    with open(rf, 'r') as f:
                        runs.append(json.load(f))
                except Exception:
                    pass
            if runs:
                all_runs[inst][algo_key] = runs
    return all_runs if all_runs else None


# ═══════════════════════════════════ 基准实验图表 (每个图表独立文件) ═══════════════════════════════════

def chart_benchmark_hv_comparison(instances, stats_tests, exp_id, output_dir='charts'):
    """Hypervolume 对比柱状图 + Wilcoxon 显著性标注 (独立图表)."""
    bench_dir = os.path.join(output_dir, 'benchmark')
    sorted_inst = sorted(instances.keys())
    x = np.arange(len(sorted_inst))
    w = 0.35

    fig, ax = _make_figure(
        f'Hypervolume Comparison — RMOEA/D vs MOEA/D\nExperiment: {exp_id}',
        figsize=(14, 7), extra=f'HV · {exp_id}')

    rh, re, mh, me = [], [], [], []
    for inst in sorted_inst:
        r = instances[inst].get('rmoea_d', {})
        m = instances[inst].get('moea_d', {})
        rh.append(r.get('hv_mean', 0))
        re.append(r.get('hv_std', 0))
        mh.append(m.get('hv_mean', 0))
        me.append(m.get('hv_std', 0))

    ax.bar(x - w / 2, rh, w, yerr=re, label='RMOEA/D', color=C_RMOEA, alpha=0.88,
           capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': C_GRAY}, edgecolor='white', linewidth=0.5)
    ax.bar(x + w / 2, mh, w, yerr=me, label='MOEA/D', color=C_MOEA, alpha=0.88,
           capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': C_GRAY}, edgecolor='white', linewidth=0.5)

    if stats_tests and stats_tests.get('wilcoxon_hv'):
        for i, inst in enumerate(sorted_inst):
            pi = stats_tests['wilcoxon_hv'].get('per_instance', {}).get(inst, {})
            if pi.get('sig') != 'n.s.':
                ymax = max(rh[i] + re[i], mh[i] + me[i])
                ax.annotate(pi.get('sig', '*'), (i, ymax * 1.03), ha='center',
                            fontsize=14, color=C_DECLINE, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in sorted_inst], fontsize=11, fontweight='bold')
    ax.set_ylabel('Hypervolume', fontsize=12)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    style_ax(ax)

    winner_r = sum(1 for i, inst in enumerate(sorted_inst) if rh[i] >= mh[i])
    ax.set_title(f'Hypervolume Comparison — RMOEA/D vs MOEA/D\n'
                 f'RMOEA/D wins on {winner_r}/{len(sorted_inst)} instances | Experiment: {exp_id}',
                 fontsize=14, fontweight='bold')

    filepath = os.path.join(bench_dir, f'benchmark_hv_comparison_{exp_id}.png')
    _save_and_close(fig, filepath, "Benchmark HV comparison chart saved")


def chart_benchmark_makespan_comparison(instances, stats_tests, exp_id, output_dir='charts'):
    """Best Makespan 对比柱状图 + Wilcoxon 显著性标注 (独立图表)."""
    bench_dir = os.path.join(output_dir, 'benchmark')
    sorted_inst = sorted(instances.keys())
    x = np.arange(len(sorted_inst))
    w = 0.35

    fig, ax = _make_figure(
        f'Best Makespan Comparison — RMOEA/D vs MOEA/D\nExperiment: {exp_id}',
        figsize=(14, 7), extra=f'Makespan · {exp_id}')

    rh, re, mh, me = [], [], [], []
    for inst in sorted_inst:
        r = instances[inst].get('rmoea_d', {})
        m = instances[inst].get('moea_d', {})
        rh.append(r.get('best_makespan_mean', 0))
        re.append(r.get('best_makespan_std', 0))
        mh.append(m.get('best_makespan_mean', 0))
        me.append(m.get('best_makespan_std', 0))

    ax.bar(x - w / 2, rh, w, yerr=re, label='RMOEA/D', color=C_RMOEA, alpha=0.88,
           capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': C_GRAY}, edgecolor='white', linewidth=0.5)
    ax.bar(x + w / 2, mh, w, yerr=me, label='MOEA/D', color=C_MOEA, alpha=0.88,
           capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': C_GRAY}, edgecolor='white', linewidth=0.5)

    if stats_tests and stats_tests.get('wilcoxon_makespan'):
        for i, inst in enumerate(sorted_inst):
            pi = stats_tests['wilcoxon_makespan'].get('per_instance', {}).get(inst, {})
            if pi.get('sig') != 'n.s.':
                ymax = max(rh[i] + re[i], mh[i] + me[i])
                ax.annotate(pi.get('sig', '*'), (i, ymax * 1.03), ha='center',
                            fontsize=14, color=C_DECLINE, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in sorted_inst], fontsize=11, fontweight='bold')
    ax.set_ylabel('Best Makespan', fontsize=12)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    style_ax(ax)

    winner_r = sum(1 for i, inst in enumerate(sorted_inst) if rh[i] <= mh[i])
    ax.set_title(f'Best Makespan Comparison — RMOEA/D vs MOEA/D\n'
                 f'RMOEA/D wins on {winner_r}/{len(sorted_inst)} instances | Experiment: {exp_id}',
                 fontsize=14, fontweight='bold')

    filepath = os.path.join(bench_dir, f'benchmark_makespan_comparison_{exp_id}.png')
    _save_and_close(fig, filepath, "Benchmark Makespan comparison chart saved")


def chart_benchmark_workload_comparison(instances, stats_tests, exp_id, output_dir='charts'):
    """Best Workload 对比柱状图 + Wilcoxon 显著性标注 (独立图表)."""
    bench_dir = os.path.join(output_dir, 'benchmark')
    sorted_inst = sorted(instances.keys())
    x = np.arange(len(sorted_inst))
    w = 0.35

    fig, ax = _make_figure(
        f'Best Workload Comparison — RMOEA/D vs MOEA/D\nExperiment: {exp_id}',
        figsize=(14, 7), extra=f'Workload · {exp_id}')

    rh, re, mh, me = [], [], [], []
    for inst in sorted_inst:
        r = instances[inst].get('rmoea_d', {})
        m = instances[inst].get('moea_d', {})
        rh.append(r.get('best_workload_mean', 0))
        re.append(r.get('best_workload_std', 0))
        mh.append(m.get('best_workload_mean', 0))
        me.append(m.get('best_workload_std', 0))

    ax.bar(x - w / 2, rh, w, yerr=re, label='RMOEA/D', color=C_RMOEA, alpha=0.88,
           capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': C_GRAY}, edgecolor='white', linewidth=0.5)
    ax.bar(x + w / 2, mh, w, yerr=me, label='MOEA/D', color=C_MOEA, alpha=0.88,
           capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': C_GRAY}, edgecolor='white', linewidth=0.5)

    if stats_tests and stats_tests.get('wilcoxon_workload'):
        for i, inst in enumerate(sorted_inst):
            pi = stats_tests['wilcoxon_workload'].get('per_instance', {}).get(inst, {})
            if pi.get('sig') != 'n.s.':
                ymax = max(rh[i] + re[i], mh[i] + me[i])
                ax.annotate(pi.get('sig', '*'), (i, ymax * 1.03), ha='center',
                            fontsize=14, color=C_DECLINE, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in sorted_inst], fontsize=11, fontweight='bold')
    ax.set_ylabel('Best Workload', fontsize=12)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    style_ax(ax)

    winner_r = sum(1 for i, inst in enumerate(sorted_inst) if rh[i] <= mh[i])
    ax.set_title(f'Best Workload Comparison — RMOEA/D vs MOEA/D\n'
                 f'RMOEA/D wins on {winner_r}/{len(sorted_inst)} instances | Experiment: {exp_id}',
                 fontsize=14, fontweight='bold')

    filepath = os.path.join(bench_dir, f'benchmark_workload_comparison_{exp_id}.png')
    _save_and_close(fig, filepath, "Benchmark Workload comparison chart saved")


def chart_benchmark_cohens_d(instances, stats_tests, exp_id, output_dir='charts'):
    """Cohen's d 效应量柱状图 — 三个指标并列 (独立图表)."""
    bench_dir = os.path.join(output_dir, 'benchmark')
    sorted_inst = sorted(instances.keys())
    x = np.arange(len(sorted_inst))
    wd = 0.22

    fig, ax = _make_figure(
        f"Cohen's d Effect Size Per Instance — RMOEA/D vs MOEA/D\nExperiment: {exp_id}",
        figsize=(14, 7), extra=f'Cohen d · {exp_id}')

    d_hv, d_ms, d_wl = [], [], []
    for inst in sorted_inst:
        wh = stats_tests.get('wilcoxon_hv', {}).get('per_instance', {}).get(inst, {})
        wm = stats_tests.get('wilcoxon_makespan', {}).get('per_instance', {}).get(inst, {})
        ww = stats_tests.get('wilcoxon_workload', {}).get('per_instance', {}).get(inst, {})
        d_hv.append(wh.get('cohens_d', 0))
        d_ms.append(wm.get('cohens_d', 0))
        d_wl.append(ww.get('cohens_d', 0))

    ax.bar(x - wd, d_hv, wd, label="HV", color=C_RMOEA, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.bar(x, d_ms, wd, label="Makespan", color=C_MOEA, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.bar(x + wd, d_wl, wd, label="Workload", color=C_QPAS, alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axhline(y=0.8, color='#D62728', linestyle='--', alpha=0.6, linewidth=1.2, label='large (±0.8)')
    ax.axhline(y=-0.8, color='#D62728', linestyle='--', alpha=0.6, linewidth=1.2)
    ax.axhline(y=0.5, color='#FF7F0E', linestyle=':', alpha=0.6, linewidth=1.2, label='medium (±0.5)')
    ax.axhline(y=-0.5, color='#FF7F0E', linestyle=':', alpha=0.6, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in sorted_inst], fontsize=11, fontweight='bold')
    ax.set_ylabel("Cohen's d", fontsize=12)
    ax.set_title("Cohen's d Effect Size (d>0: RMOEA/D better on HV; d<0: better on Makespan/Workload)",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9, ncol=3, framealpha=0.9)
    style_ax(ax)

    filepath = os.path.join(bench_dir, f'benchmark_cohens_d_{exp_id}.png')
    _save_and_close(fig, filepath, "Benchmark Cohen's d chart saved")


def chart_benchmark_pvalue_heatmap(instances, stats_tests, exp_id, output_dir='charts'):
    """Wilcoxon p-value 热力图 — 三指标 × 实例 (独立图表)."""
    bench_dir = os.path.join(output_dir, 'benchmark')
    sorted_inst = sorted(instances.keys())
    n_inst = len(sorted_inst)

    fig, ax = _make_figure(
        f'Wilcoxon p-value Heatmap — RMOEA/D vs MOEA/D\nExperiment: {exp_id}',
        figsize=(14, 5), extra=f'p-value · {exp_id}')

    p_matrix = np.zeros((3, n_inst))
    annotations = np.empty((3, n_inst), dtype=object)
    metric_names = ['HV', 'Makespan', 'Workload']
    wt_keys = ['wilcoxon_hv', 'wilcoxon_makespan', 'wilcoxon_workload']

    for mi, (mk, mn) in enumerate(zip(wt_keys, metric_names)):
        for ii, inst in enumerate(sorted_inst):
            pi = stats_tests.get(mk, {}).get('per_instance', {}).get(inst, {})
            p_val = pi.get('p_value', 1.0)
            p_matrix[mi, ii] = min(p_val, 1.0)
            sig = pi.get('sig', 'ns')
            annotations[mi, ii] = f'{p_val:.3f}\n{sig}'

    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(p_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=0.15)
    ax.set_xticks(range(n_inst))
    ax.set_yticks(range(3))
    ax.set_xticklabels([i.upper() for i in sorted_inst], fontsize=11, fontweight='bold')
    ax.set_yticklabels(metric_names, fontsize=12, fontweight='bold')

    for mi in range(3):
        for ii in range(n_inst):
            color = 'white' if p_matrix[mi, ii] < 0.06 else 'black'
            ax.text(ii, mi, annotations[mi, ii], ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.9, label='p-value')
    ax.set_title('Wilcoxon p-value Heatmap (green = significant)',
                 fontsize=14, fontweight='bold')
    style_ax(ax)

    filepath = os.path.join(bench_dir, f'benchmark_pvalue_heatmap_{exp_id}.png')
    _save_and_close(fig, filepath, "Benchmark p-value heatmap saved")


def chart_benchmark_effect_summary(instances, stats_tests, exp_id, output_dir='charts'):
    """效应量汇总水平柱状图 (独立图表)."""
    bench_dir = os.path.join(output_dir, 'benchmark')

    fig, ax = _make_figure(
        f'Effect Size Summary — RMOEA/D vs MOEA/D\nExperiment: {exp_id}',
        figsize=(12, 5), extra=f'Effect Summary · {exp_id}')

    if stats_tests and stats_tests.get('effect_size_summary'):
        es = stats_tests['effect_size_summary']
        metric_labels = sorted(es.keys())
        y_positions = range(len(metric_labels))
        mean_ds = [es[m].get('mean_d', 0) for m in metric_labels]
        colors = [C_RMOEA if v > 0 else C_DECLINE for v in mean_ds]

        for yi, (m, md) in enumerate(zip(metric_labels, mean_ds)):
            ax.axhline(y=yi, color=C_GRID, alpha=0.3, linewidth=0.5)
            ax.barh(yi, md, height=0.5, color=colors[yi] if md != 0 else C_GRAY,
                    alpha=0.8, edgecolor='white', linewidth=0.5)
            ax.text(md + (0.02 if md >= 0 else -0.12), yi,
                    f'{md:+.3f}  [{es[m].get("min_d", 0):+.2f}, {es[m].get("max_d", 0):+.2f}]',
                    va='center', fontsize=10, fontweight='bold')

        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(metric_labels, fontsize=12, fontweight='bold')
        ax.set_xlabel("Cohen's d (mean ± range)", fontsize=12)
        ax.set_title('Effect Size Summary (Cross-Instance)', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No effect size data available', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color=C_GRAY)
        ax.axis('off')
    style_ax(ax)

    filepath = os.path.join(bench_dir, f'benchmark_effect_summary_{exp_id}.png')
    _save_and_close(fig, filepath, "Benchmark effect summary chart saved")


def chart_benchmark_stats_card(instances, stats_tests, exp_id, output_dir='charts'):
    """统计检验概览卡片 (独立图表)."""
    bench_dir = os.path.join(output_dir, 'benchmark')

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis('off')
    fig.suptitle(f'RMOEA/D vs MOEA/D — Statistical Tests Overview\nExperiment: {exp_id}',
                 fontsize=14, fontweight='bold')
    source_footer(fig, f'Stats Card · {exp_id}')

    card_lines = []
    card_lines.append("═══════════════════════════════════════════")
    card_lines.append("  STATISTICAL TESTS SUMMARY")
    card_lines.append("═══════════════════════════════════════════")
    card_lines.append("")

    if stats_tests and stats_tests.get('friedman'):
        f = stats_tests['friedman']
        sig_raw = f.get('significant', False)
        sig_bool = sig_raw in (True, 'yes', 'true')
        card_lines.append(f"  Friedman Test (Global):")
        card_lines.append(f"    chi2 = {f.get('statistic', 0):.4f}")
        card_lines.append(f"    p    = {f.get('p_value', 1):.6f}")
        card_lines.append(f"    -> {'SIGNIFICANT *' if sig_bool else 'not significant (n.s.)'}")
    card_lines.append("")

    for wt_key, label in [('wilcoxon_hv', 'Hypervolume'), ('wilcoxon_makespan', 'Makespan'), ('wilcoxon_workload', 'Workload')]:
        if stats_tests and stats_tests.get(wt_key):
            wt = stats_tests[wt_key]
            per_inst = wt.get('per_instance', {})
            sig_count = sum(1 for pi in per_inst.values() if pi.get('sig') != 'n.s.')
            ov = wt.get('overall', {})
            card_lines.append(f"  Wilcoxon ({label}):")
            card_lines.append(f"    Per-instance: {sig_count}/{len(per_inst)} significant")
            if ov:
                card_lines.append(f"    Overall: p={ov.get('p_value', 1):.6f} ({ov.get('sig', 'n.s.')})")
            sig_insts = [k.upper() for k, v in per_inst.items() if v.get('sig') != 'n.s.']
            card_lines.append(f"    Significant instances: {', '.join(sig_insts) if sig_insts else 'none'}")
        card_lines.append("")

    if stats_tests and stats_tests.get('effect_size_summary'):
        es = stats_tests['effect_size_summary']
        card_lines.append("  Cohen's d Effect Size (Cross-Instance):")
        for m in sorted(es.keys()):
            ei = es[m]
            card_lines.append(f"    {m}: mean={ei.get('mean_d', 0):+.3f} median={ei.get('median_d', 0):+.3f}")
            card_lines.append(f"         L:{ei.get('n_large', 0)} M:{ei.get('n_medium', 0)} "
                              f"S:{ei.get('n_small', 0)} N:{ei.get('n_negligible', 0)}")

    card_lines.append("")
    card_lines.append("═══════════════════════════════════════════")
    card_lines.append("  LEGEND")
    card_lines.append("═══════════════════════════════════════════")
    card_lines.append("  *** p<0.001   ** p<0.01   * p<0.05   n.s. not significant")
    card_lines.append("  |d|>=0.8 large   0.5<=|d|<0.8 medium   0.2<=|d|<0.5 small   |d|<0.2 negligible")

    card_text = "\n".join(card_lines)
    ax.text(0.5, 0.5, card_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='#F5F5F5', edgecolor=C_GRAY, alpha=0.95))

    filepath = os.path.join(bench_dir, f'benchmark_stats_card_{exp_id}.png')
    _save_and_close(fig, filepath, "Benchmark stats card saved")


# ═══════════════════════════════════ 消融实验图表 (每个图表独立文件) ═══════════════════════════════════

def chart_ablation_hv_comparison(ablation_data, output_dir='charts'):
    """消融实验 Hypervolume 分组柱状图 + 配对显著性标注 (独立图表)."""
    ab_dir = os.path.join(output_dir, 'ablation')
    avail_algos = sorted(set().union(*[set(d.keys()) for d in ablation_data.values()]))
    sorted_inst = sorted(ablation_data.keys())
    n_inst = len(sorted_inst)
    n_algos = len(avail_algos)
    x = np.arange(n_inst)
    width = max(0.15, 0.8 / n_algos)

    fig, ax = _make_figure(
        'RMOEA/D Ablation Study — Hypervolume Comparison\n(with Wilcoxon Pairwise Significance)',
        figsize=(14, 7), extra='Ablation HV')

    for i, algo in enumerate(avail_algos):
        hvs, errs = [], []
        for inst in sorted_inst:
            runs = ablation_data[inst].get(algo, [])
            vals = [r.get("final_hv", r.get("hv", 0)) for r in runs] if runs else [0]
            hvs.append(np.mean(vals))
            errs.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
        ax.bar(x + i * width, hvs, width, label=ABLATION_ALGO_LABELS.get(algo, algo),
               color=ABLATION_ALGO_COLORS.get(algo, C_GRAY), alpha=0.85,
               edgecolor='white', linewidth=0.5)
        if any(e > 0 for e in errs):
            ax.errorbar(x + i * width, hvs, yerr=errs, fmt='none',
                        ecolor=C_GRAY, elinewidth=1.0, capsize=3, alpha=0.6)

    for idx, inst in enumerate(sorted_inst):
        avail = sorted(ablation_data[inst].keys())
        y_offsets = {}
        for i, a1 in enumerate(avail):
            for a2 in avail[i + 1:]:
                hvs1 = [r["final_hv"] for r in ablation_data[inst][a1]]
                hvs2 = [r["final_hv"] for r in ablation_data[inst][a2]]
                wt = _wilcoxon_test(hvs1, hvs2)
                if wt and wt.get('sig') != 'n.s.':
                    mid_x = idx + (avail_algos.index(a1) + avail_algos.index(a2)) / 2 * width
                    max_hv = max(np.mean(hvs1), np.mean(hvs2))
                    key = round(mid_x, 2)
                    y_offsets[key] = y_offsets.get(key, 0) + 1
                    ann_y = max_hv * (1 + 0.025 * y_offsets[key])
                    ax.annotate(wt['sig'], (mid_x, ann_y), ha='center', fontsize=8,
                                color=C_DECLINE, fontweight='bold')

    ax.set_xticks(x + width * (n_algos - 1) / 2)
    ax.set_xticklabels([i.upper() for i in sorted_inst], fontsize=11, fontweight='bold')
    ax.set_ylabel('Hypervolume', fontsize=12)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    style_ax(ax)

    filepath = os.path.join(ab_dir, 'ablation_hv_comparison.png')
    _save_and_close(fig, filepath, "Ablation HV comparison chart saved")


def chart_ablation_makespan_comparison(ablation_data, output_dir='charts'):
    """消融实验 Best Makespan 分组柱状图 (独立图表)."""
    ab_dir = os.path.join(output_dir, 'ablation')
    avail_algos = sorted(set().union(*[set(d.keys()) for d in ablation_data.values()]))
    sorted_inst = sorted(ablation_data.keys())
    n_inst = len(sorted_inst)
    n_algos = len(avail_algos)
    x = np.arange(n_inst)
    width = max(0.15, 0.8 / n_algos)

    fig, ax = _make_figure(
        'RMOEA/D Ablation Study — Best Makespan Comparison',
        figsize=(14, 7), extra='Ablation Makespan')

    for i, algo in enumerate(avail_algos):
        mss, errs = [], []
        for inst in sorted_inst:
            runs = ablation_data[inst].get(algo, [])
            vals = [r.get("best_makespan", 99999) for r in runs] if runs else [0]
            mss.append(np.mean(vals))
            errs.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
        ax.bar(x + i * width, mss, width, label=ABLATION_ALGO_LABELS.get(algo, algo),
               color=ABLATION_ALGO_COLORS.get(algo, C_GRAY), alpha=0.85,
               edgecolor='white', linewidth=0.5)
        if any(e > 0 for e in errs):
            ax.errorbar(x + i * width, mss, yerr=errs, fmt='none',
                        ecolor=C_GRAY, elinewidth=1.0, capsize=3, alpha=0.6)

    ax.set_xticks(x + width * (n_algos - 1) / 2)
    ax.set_xticklabels([i.upper() for i in sorted_inst], fontsize=11, fontweight='bold')
    ax.set_ylabel('Best Makespan', fontsize=12)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    style_ax(ax)

    filepath = os.path.join(ab_dir, 'ablation_makespan_comparison.png')
    _save_and_close(fig, filepath, "Ablation Makespan comparison chart saved")


def chart_ablation_runtime_comparison(ablation_data, output_dir='charts'):
    """消融实验 Runtime 分组柱状图 (独立图表)."""
    ab_dir = os.path.join(output_dir, 'ablation')
    avail_algos = sorted(set().union(*[set(d.keys()) for d in ablation_data.values()]))
    sorted_inst = sorted(ablation_data.keys())
    n_inst = len(sorted_inst)
    n_algos = len(avail_algos)
    x = np.arange(n_inst)
    width = max(0.15, 0.8 / n_algos)

    fig, ax = _make_figure(
        'RMOEA/D Ablation Study — Runtime Comparison',
        figsize=(14, 7), extra='Ablation Runtime')

    for i, algo in enumerate(avail_algos):
        tms, errs = [], []
        for inst in sorted_inst:
            runs = ablation_data[inst].get(algo, [])
            vals = [r.get("total_time", r.get("time_mean", 0)) for r in runs] if runs else [0]
            tms.append(np.mean(vals))
            errs.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
        ax.bar(x + i * width, tms, width, label=ABLATION_ALGO_LABELS.get(algo, algo),
               color=ABLATION_ALGO_COLORS.get(algo, C_GRAY), alpha=0.85,
               edgecolor='white', linewidth=0.5)
        if any(e > 0 for e in errs):
            ax.errorbar(x + i * width, tms, yerr=errs, fmt='none',
                        ecolor=C_GRAY, elinewidth=1.0, capsize=3, alpha=0.6)

    ax.set_xticks(x + width * (n_algos - 1) / 2)
    ax.set_xticklabels([i.upper() for i in sorted_inst], fontsize=11, fontweight='bold')
    ax.set_ylabel('Runtime (s)', fontsize=12)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    style_ax(ax)

    filepath = os.path.join(ab_dir, 'ablation_runtime_comparison.png')
    _save_and_close(fig, filepath, "Ablation Runtime comparison chart saved")


def chart_ablation_cohens_d_matrix(ablation_data, output_dir='charts'):
    """消融实验 Cohen's d 矩阵热力图 — 全局配对比较 (独立图表)."""
    ab_dir = os.path.join(output_dir, 'ablation')
    avail_algos = sorted(set().union(*[set(d.keys()) for d in ablation_data.values()]))
    sorted_inst = sorted(ablation_data.keys())
    n_algos = len(avail_algos)

    fig, ax = _make_figure(
        "RMOEA/D Ablation Study — Cohen's d Matrix (Global, row vs col)",
        figsize=(9, 7), extra='Ablation Cohen d Matrix')

    all_hv = {algo: [] for algo in avail_algos}
    for inst in sorted_inst:
        for algo in avail_algos:
            if algo in ablation_data[inst]:
                all_hv[algo].extend([r["final_hv"] for r in ablation_data[inst][algo]])

    d_matrix = np.zeros((n_algos, n_algos))
    for i, a1 in enumerate(avail_algos):
        for j, a2 in enumerate(avail_algos):
            if i != j:
                d_matrix[i, j] = _cohens_d(all_hv[a1], all_hv[a2])

    im = ax.imshow(d_matrix, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    ax.set_xticks(range(n_algos))
    ax.set_yticks(range(n_algos))
    ax.set_xticklabels([ABLATION_ALGO_LABELS.get(a, a) for a in avail_algos],
                       rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels([ABLATION_ALGO_LABELS.get(a, a) for a in avail_algos], fontsize=9)

    for i in range(n_algos):
        for j in range(n_algos):
            color = 'white' if abs(d_matrix[i, j]) > 1.0 else 'black'
            ax.text(j, i, f'{d_matrix[i, j]:.2f}', ha='center', va='center',
                    fontsize=11, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.85, label="Cohen's d")
    style_ax(ax)

    filepath = os.path.join(ab_dir, 'ablation_cohens_d_matrix.png')
    _save_and_close(fig, filepath, "Ablation Cohen's d matrix chart saved")


def chart_ablation_cohens_d_per_instance(ablation_data, output_dir='charts'):
    """消融实验 每实例 Cohen's d vs 参考算法 (独立图表)."""
    ab_dir = os.path.join(output_dir, 'ablation')
    avail_algos = sorted(set().union(*[set(d.keys()) for d in ablation_data.values()]))
    sorted_inst = sorted(ablation_data.keys())
    n_inst = len(sorted_inst)
    n_algos = len(avail_algos)
    x = np.arange(n_inst)

    if n_algos < 2:
        return

    ref = avail_algos[0]
    fig, ax = _make_figure(
        f"RMOEA/D Ablation Study — Cohen's d per Instance (vs {ABLATION_ALGO_LABELS.get(ref, ref)})",
        figsize=(14, 7), extra='Ablation Cohen d Per Instance')

    sub_w = max(0.15, 0.7 / (n_algos - 1))
    for i, algo in enumerate(avail_algos[1:], 1):
        ds = []
        for inst in sorted_inst:
            if algo in ablation_data[inst] and ref in ablation_data[inst]:
                hvs_r = [r["final_hv"] for r in ablation_data[inst][ref]]
                hvs_a = [r["final_hv"] for r in ablation_data[inst][algo]]
                ds.append(_cohens_d(hvs_r, hvs_a))
            else:
                ds.append(0)
        ax.bar(x + (i - 1) * sub_w, ds, sub_w,
               label=f'{ABLATION_ALGO_LABELS.get(algo, algo)} vs {ABLATION_ALGO_LABELS.get(ref, ref)}',
               color=ABLATION_ALGO_COLORS.get(algo, C_GRAY), alpha=0.85,
               edgecolor='white', linewidth=0.5)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axhline(y=0.8, color='#D62728', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=-0.8, color='#D62728', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in sorted_inst], fontsize=11, fontweight='bold')
    ax.set_ylabel("Cohen's d", fontsize=12)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    style_ax(ax)

    filepath = os.path.join(ab_dir, 'ablation_cohens_d_per_instance.png')
    _save_and_close(fig, filepath, "Ablation Cohen's d per-instance chart saved")


def chart_ablation_pvalue_heatmap(ablation_data, output_dir='charts'):
    """消融实验 配对 Wilcoxon p-value 热力图 (独立图表)."""
    ab_dir = os.path.join(output_dir, 'ablation')
    avail_algos = sorted(set().union(*[set(d.keys()) for d in ablation_data.values()]))
    sorted_inst = sorted(ablation_data.keys())
    n_inst = len(sorted_inst)
    n_algos = len(avail_algos)

    if n_algos < 2:
        return

    n_pairs = n_algos * (n_algos - 1) // 2
    fig, ax = _make_figure(
        'RMOEA/D Ablation Study — Pairwise Wilcoxon p-value Heatmap\n(* = significant at p < 0.05)',
        figsize=(14, 5), extra='Ablation p-value')

    p_matrix = np.zeros((n_pairs, n_inst))
    pair_labels = []
    pi = 0
    for a1_idx, a1 in enumerate(avail_algos):
        for a2 in avail_algos[a1_idx + 1:]:
            pair_labels.append(f'{ABLATION_ALGO_LABELS.get(a1, a1)} v {ABLATION_ALGO_LABELS.get(a2, a2)}')
            for ii, inst in enumerate(sorted_inst):
                if a1 in ablation_data[inst] and a2 in ablation_data[inst]:
                    hvs1 = [r["final_hv"] for r in ablation_data[inst][a1]]
                    hvs2 = [r["final_hv"] for r in ablation_data[inst][a2]]
                    wt = _wilcoxon_test(hvs1, hvs2)
                    p_matrix[pi, ii] = wt['p_value'] if wt else 1.0
                else:
                    p_matrix[pi, ii] = 1.0
            pi += 1

    im = ax.imshow(p_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.15)
    ax.set_xticks(range(n_inst))
    ax.set_yticks(range(n_pairs))
    ax.set_xticklabels([i.upper() for i in sorted_inst], fontsize=10, fontweight='bold')
    ax.set_yticklabels(pair_labels, fontsize=8)

    for pi in range(n_pairs):
        for ii in range(n_inst):
            color = 'white' if p_matrix[pi, ii] < 0.06 else 'black'
            sig_star = ('***' if p_matrix[pi, ii] < 0.001 else
                        ('**' if p_matrix[pi, ii] < 0.01 else
                         ('*' if p_matrix[pi, ii] < 0.05 else '')))
            ax.text(ii, pi, sig_star, ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.9, label='p-value')
    style_ax(ax)

    filepath = os.path.join(ab_dir, 'ablation_pvalue_heatmap.png')
    _save_and_close(fig, filepath, "Ablation p-value heatmap chart saved")


def chart_ablation_stats_card(ablation_data, output_dir='charts'):
    """消融实验统计摘要卡片 (独立图表)."""
    ab_dir = os.path.join(output_dir, 'ablation')
    avail_algos = sorted(set().union(*[set(d.keys()) for d in ablation_data.values()]))
    sorted_inst = sorted(ablation_data.keys())
    n_inst = len(sorted_inst)
    n_algos = len(avail_algos)

    from scipy.stats import friedmanchisquare

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    fig.suptitle('Ablation Study — Statistical Tests Summary', fontsize=14, fontweight='bold')
    source_footer(fig, 'Ablation Stats Card')

    all_hv = {algo: [] for algo in avail_algos}
    for inst in sorted_inst:
        for algo in avail_algos:
            if algo in ablation_data[inst]:
                all_hv[algo].extend([r["final_hv"] for r in ablation_data[inst][algo]])

    card = ["═══════════════════════════════════════════"]
    card.append("  ABLATION STATISTICAL TESTS")
    card.append("═══════════════════════════════════════════")
    card.append("")
    card.append(f"  Algorithms: {' · '.join([ABLATION_ALGO_LABELS.get(a, a) for a in avail_algos])}")
    card.append(f"  Instances: {n_inst} ({', '.join(s.upper() for s in sorted_inst)})")
    card.append("")

    groups = []
    for algo in avail_algos:
        grp = []
        for inst in sorted_inst:
            if algo in ablation_data[inst]:
                grp.append(np.mean([r["final_hv"] for r in ablation_data[inst][algo]]))
            else:
                grp.append(0)
        groups.append(grp)

    card.append("--- Friedman Test ---")
    try:
        stat, p = friedmanchisquare(*groups)
        card.append(f"  chi2 = {stat:.4f}")
        card.append(f"  p    = {p:.6f}")
        card.append(f"  -> {'SIGNIFICANT *' if p < 0.05 else 'not significant (n.s.)'}")
    except Exception:
        card.append("  N/A")
    card.append("")

    card.append("--- Global Wilcoxon + Cohen's d ---")
    for i, a1 in enumerate(avail_algos):
        for a2 in avail_algos[i + 1:]:
            wt = _wilcoxon_test(all_hv[a1], all_hv[a2])
            d = _cohens_d(all_hv[a1], all_hv[a2])
            if wt:
                card.append(f"  {a1} vs {a2}:")
                card.append(f"    p={wt['p_value']:.6f} {wt['sig']}, Cohen's d={d:+.3f} ({_effect_size_label(d)})")
    card.append("")

    card.append("--- Per-Instance Significant Pairs ---")
    for inst in sorted_inst:
        avail = sorted(ablation_data[inst].keys())
        sig_pairs = []
        for i, a1 in enumerate(avail):
            for a2 in avail[i + 1:]:
                hvs1 = [r["final_hv"] for r in ablation_data[inst][a1]]
                hvs2 = [r["final_hv"] for r in ablation_data[inst][a2]]
                wt = _wilcoxon_test(hvs1, hvs2)
                if wt and wt.get('sig') != 'ns':
                    d = _cohens_d(hvs1, hvs2)
                    sig_pairs.append(f'{a1}v{a2}:p={wt["p_value"]:.4f} d={d:+.3f}')
        if sig_pairs:
            card.append(f"  {inst.upper()}: {' | '.join(sig_pairs)}")
    if not any(sig_pairs for inst in sorted_inst):
        card.append("  (none)")
    card.append("")
    card.append("═══════════════════════════════════════════")
    card.append("  *** p<0.001  ** p<0.01  * p<0.05")
    card.append("  |d|>=0.8 large  0.5<=|d|<0.8 medium")
    card.append("  0.2<=|d|<0.5 small  |d|<0.2 negligible")

    ax.text(0.5, 0.5, "\n".join(card), transform=ax.transAxes, fontsize=9,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='#FAFAFA', edgecolor=C_GRAY, alpha=0.95))

    filepath = os.path.join(ab_dir, 'ablation_stats_card.png')
    _save_and_close(fig, filepath, "Ablation stats card saved")


# ═══════════════════════════════════ 主入口 ═══════════════════════════════════

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("RMOEA/D 完整实验分析可视化系统")
    logger.info("所有图表输出到 charts/ 目录，每个图表为独立文件")
    logger.info("=" * 60)

    # ── 1. Benchmark 分析 (7 个独立图表) ──
    instances, stats_tests, exp_id = load_benchmark_data()
    if instances:
        logger.info("Loaded benchmark data: %d instances, exp_id=%s", len(instances), exp_id)

        if not stats_tests:
            logger.info("No stats_tests in JSON, computing from per-run data...")
            per_run = load_per_run_data()
            if per_run:
                from rmoea_d.utils.experiment import run_statistical_tests
                agg_raw = {}
                for inst in sorted(per_run.keys()):
                    agg = {}
                    for algo_key in ['rmoea_d', 'moea_d']:
                        if algo_key in per_run[inst]:
                            runs = per_run[inst][algo_key]
                            hvs = [r.get('final_hv', r.get('hv', 0)) for r in runs]
                            bm = [r.get('best_makespan', 0) for r in runs]
                            bw = [r.get('best_workload', 0) for r in runs]
                            agg[algo_key] = {
                                "hv_mean": float(np.mean(hvs)),
                                "hv_std": float(np.std(hvs, ddof=1)) if len(hvs) > 1 else 0.0,
                                "best_makespan_mean": float(np.mean(bm)),
                                "best_makespan_std": float(np.std(bm, ddof=1)) if len(bm) > 1 else 0.0,
                                "best_workload_mean": float(np.mean(bw)),
                                "best_workload_std": float(np.std(bw, ddof=1)) if len(bw) > 1 else 0.0,
                                "_hv_all": hvs, "_bm_all": bm, "_bw_all": bw,
                            }
                    agg_raw[inst] = agg
                # 用 experiment.py 的统一实现，再解包到 legacy 格式
                wrapped = {"benchmark": agg_raw, "ablation": {}}
                stats_tests = run_statistical_tests(wrapped).get("benchmark", {})
                # 补充 Friedman 检验 (experiment.py 的 run_statistical_tests 不包含)
                rmoea_hv_means = [agg_raw[i]["rmoea_d"]["hv_mean"] for i in sorted(agg_raw)]
                moea_hv_means = [agg_raw[i]["moea_d"]["hv_mean"] for i in sorted(agg_raw)]
                stats_tests["friedman"] = _friedman_test(rmoea_hv_means, moea_hv_means)
                logger.info("Computed stats_tests from %d per-run files", sum(len(v) for v in per_run.values()))

        # 生成基准实验的 7 个独立图表
        chart_benchmark_hv_comparison(instances, stats_tests, exp_id)
        chart_benchmark_makespan_comparison(instances, stats_tests, exp_id)
        chart_benchmark_workload_comparison(instances, stats_tests, exp_id)
        chart_benchmark_cohens_d(instances, stats_tests, exp_id)
        chart_benchmark_pvalue_heatmap(instances, stats_tests, exp_id)
        chart_benchmark_effect_summary(instances, stats_tests, exp_id)
        chart_benchmark_stats_card(instances, stats_tests, exp_id)

    # ── 2. Ablation 分析 (7 个独立图表) ──
    ablation_data = load_ablation_data()
    if ablation_data:
        logger.info("Loaded ablation data: %d instances", len(ablation_data))

        chart_ablation_hv_comparison(ablation_data)
        chart_ablation_makespan_comparison(ablation_data)
        chart_ablation_runtime_comparison(ablation_data)
        chart_ablation_cohens_d_matrix(ablation_data)
        chart_ablation_cohens_d_per_instance(ablation_data)
        chart_ablation_pvalue_heatmap(ablation_data)
        chart_ablation_stats_card(ablation_data)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("All charts generated in %.1fs", elapsed)
    logger.info("Output: charts/benchmark/ + charts/ablation/ (14 independent PNG files)")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()