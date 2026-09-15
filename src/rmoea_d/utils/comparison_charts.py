#!/usr/bin/env python3
"""
补充对比图表生成模块
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
在已有 TFN 表格图基础上，新增两组对比图：
  1. Pareto Front 对比图（Benchmark + Ablation，每实例一张）
  2. Hypervolume (HV) 对比图（Benchmark + Ablation，单张汇总）

数据来源：results/experiment/{inst}/run_*.json
"""

import os
import json
import logging
import time
from pathlib import Path

import numpy as np

# ── 强制使用非 GUI 后端，确保线程安全渲染 ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 共享绘图工具 ──
try:
    from .plot_helpers import (
        C_RMOEA, C_MOEA, C_FULL, C_QPAS, C_RVNS, C_MOEAD,
        C_IMPROVE, C_DECLINE, C_GRAY,
        source_footer, style_ax,
    )
except ImportError:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_helpers import (
        C_RMOEA, C_MOEA, C_FULL, C_QPAS, C_RVNS, C_MOEAD,
        C_IMPROVE, C_DECLINE, C_GRAY,
        source_footer, style_ax,
    )

logger = logging.getLogger(__name__)

# ═══════════════════════════════════ 常量 ═══════════════════════════════════

CHARTS_BASE = 'charts'
BENCHMARK_DIR = os.path.join(CHARTS_BASE, 'benchmark')
ABLATION_DIR = os.path.join(CHARTS_BASE, 'ablation')
EXPERIMENT_DIR = 'results/experiment'

ALGO_ALIAS = {
    'rmoea_d': 'RMOEA/D',
    'qpas_only': 'Q-PAS',
    'rvns_only': 'RVNS',
    'moea_d': 'MOEA/D',
}
ALGO_ORDER_BENCHMARK = ['rmoea_d', 'moea_d']
ALGO_ORDER_ABLATION = ['rmoea_d', 'qpas_only', 'rvns_only', 'moea_d']
ALGO_STYLE = {
    'rmoea_d': {'color': C_FULL, 'marker': 'o', 'linestyle': '-', 'size': 7},
    'qpas_only': {'color': C_QPAS, 'marker': 's', 'linestyle': '--', 'size': 7},
    'rvns_only': {'color': C_RVNS, 'marker': '^', 'linestyle': '-.', 'size': 7},
    'moea_d': {'color': C_MOEAD, 'marker': 'D', 'linestyle': ':', 'size': 6},
}


# ═══════════════════════════════════ 数据加载 ═══════════════════════════════════

def _extract_pf_points(pf):
    """统一解析 final_pf 中的 (makespan, workload) 清晰值。"""
    pts = []
    for p in pf:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            ms, wl = p[0], p[1]
        elif isinstance(p, dict):
            ms = p.get('Makespan', 0)
            wl = p.get('Workload', 0)
            if isinstance(ms, dict):
                ms = (ms.get('t1', 0) + 2 * ms.get('t2', 0) + ms.get('t3', 0)) / 4.0
            if isinstance(wl, dict):
                wl = (wl.get('t1', 0) + 2 * wl.get('t2', 0) + wl.get('t3', 0)) / 4.0
        else:
            continue
        pts.append((float(ms), float(wl)))
    return pts


def _filter_non_dominated(pts):
    """过滤非支配解（假设两个目标均为最小化）。"""
    pts = sorted(pts, key=lambda x: (x[0], x[1]))
    nd = []
    min_wl = float('inf')
    for ms, wl in pts:
        if wl < min_wl:
            nd.append((ms, wl))
            min_wl = wl
    return nd


def load_experiment_data(results_dir='results'):
    """
    加载 experiment 原始文件，返回：
      {instance: {algo: {'hv': [...], 'pf': [(ms, wl), ...]}}}
    其中 PF 选取该算法所有 run 中 HV 最高的一次，并做非支配过滤。
    """
    base = Path(results_dir) / 'experiment'
    if not base.exists():
        logger.error("Experiment directory not found: %s", base)
        return {}

    data = {}
    t0 = time.perf_counter()
    for inst_dir in sorted(base.glob('mk*')):
        inst = inst_dir.name.lower()
        data[inst] = {algo: {'hv': [], 'pf': []} for algo in ALGO_ORDER_ABLATION}
        run_files = sorted(inst_dir.glob('run_*.json'))
        logger.info("Loading PF/HV data: %d runs for %s", len(run_files), inst.upper())

        for run_file in run_files:
            try:
                with open(run_file, 'r', encoding='utf-8') as f:
                    run = json.load(f)
            except Exception as e:
                logger.warning("Failed to load %s: %s", run_file, e)
                continue

            for algo in ALGO_ORDER_ABLATION:
                if algo not in run:
                    continue
                entry = run[algo]
                data[inst][algo]['hv'].append(float(entry.get('final_hv', 0)))

                # 记录当前 run 的 HV，用于后续选择最优 PF
                pf = entry.get('final_pf', [])
                pts = _extract_pf_points(pf)
                if not hasattr(data[inst][algo], '_pf_candidates'):
                    data[inst][algo]['_pf_candidates'] = []
                data[inst][algo]['_pf_candidates'].append((float(entry.get('final_hv', 0)), pts))

        # 每个算法选取 HV 最高的一次运行作为该 instance 的代表 PF
        for algo in ALGO_ORDER_ABLATION:
            candidates = data[inst][algo].pop('_pf_candidates', [])
            if candidates:
                best_pts = max(candidates, key=lambda x: x[0])[1]
                data[inst][algo]['pf'] = _filter_non_dominated(best_pts)

    elapsed = time.perf_counter() - t0
    logger.info("PF/HV data loaded: %d instances in %.2fs", len(data), elapsed)
    return data


# ═══════════════════════════════════ Pareto Front 图 ═══════════════════════════════════

def _plot_pf(ax, data_by_algo, title, algo_order, show_legend=True):
    """在指定 ax 上绘制多个算法的 Pareto Front。"""
    for algo in algo_order:
        pts = data_by_algo.get(algo, {}).get('pf', [])
        if not pts:
            continue
        pts = sorted(pts, key=lambda x: x[0])
        ms_vals = [p[0] for p in pts]
        wl_vals = [p[1] for p in pts]
        style = ALGO_STYLE[algo]
        label = f"{ALGO_ALIAS[algo]} (|PF|={len(pts)})"
        ax.plot(ms_vals, wl_vals, color=style['color'], linestyle=style['linestyle'],
                linewidth=1.6, alpha=0.85, zorder=2)
        ax.scatter(ms_vals, wl_vals, c=style['color'], marker=style['marker'],
                   s=style['size'] ** 2 * 4, alpha=0.85, edgecolors='white',
                   linewidth=0.6, label=label, zorder=3)

    ax.set_xlabel('Crisp Makespan', fontsize=11)
    ax.set_ylabel('Crisp Total Workload', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    if show_legend:
        ax.legend(loc='upper right', framealpha=0.95, fontsize=8)
    style_ax(ax)


def plot_benchmark_pareto_fronts(data):
    """生成 Benchmark Pareto Front 对比图（每实例一张）。"""
    if not data:
        return
    insts = sorted(data.keys())
    for inst in insts:
        fig, ax = plt.subplots(figsize=(9, 6.5))
        _plot_pf(ax, data[inst], f'{inst.upper()} — Pareto Front Comparison',
                 ALGO_ORDER_BENCHMARK)
        source_footer(fig)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        out_dir = os.path.join(BENCHMARK_DIR, inst)
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, 'pareto_front_comparison.png')
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info("Benchmark PF comparison saved -> %s", filepath)


def plot_ablation_pareto_fronts(data):
    """生成 Ablation Pareto Front 对比图（每实例一张）。"""
    if not data:
        return
    insts = sorted(data.keys())
    for inst in insts:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        _plot_pf(ax, data[inst], f'{inst.upper()} — Ablation Pareto Front Comparison',
                 ALGO_ORDER_ABLATION)
        source_footer(fig)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        out_dir = os.path.join(ABLATION_DIR, inst)
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, 'pareto_front_comparison.png')
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info("Ablation PF comparison saved -> %s", filepath)


# ═══════════════════════════════════ HV 对比图 ═══════════════════════════════════

def _load_benchmark_hv(results_dir='results'):
    """从 benchmark aggregate JSON 加载 HV 均值与标准差。"""
    agg_files = sorted(Path(results_dir).glob('benchmark/benchmark_aggregate_*.json'))
    if not agg_files:
        logger.warning("No benchmark aggregate file found.")
        return {}
    with open(agg_files[-1], 'r', encoding='utf-8') as f:
        agg = json.load(f)
    insts = sorted([k for k in agg.get('instances', {}).keys() if k.startswith('mk')])
    data = {inst: {} for inst in insts}
    for inst in insts:
        for algo in ALGO_ORDER_BENCHMARK:
            entry = agg['instances'][inst].get(algo, {})
            data[inst][algo] = {
                'mean': float(entry.get('hv_mean', 0)),
                'std': float(entry.get('hv_std', 0)),
                'n_runs': int(entry.get('n_runs', 30)),
            }
    return data


def _load_ablation_hv(results_dir='results'):
    """从 ablation 结果文件聚合 HV 均值与标准差。
    注意：ablation 结果文件中的算法键为 'full'，对应本模块的 'rmoea_d'。"""
    base = Path(results_dir) / 'ablation'
    if not base.exists():
        logger.warning("No ablation results directory found.")
        return {}
    data = {}
    key_map = {'rmoea_d': 'full', 'qpas_only': 'qpas_only',
               'rvns_only': 'rvns_only', 'moea_d': 'moea_d'}
    for inst_dir in sorted(base.glob('mk*')):
        inst = inst_dir.name.lower()
        res_files = sorted(inst_dir.glob('ablation_results_*.json'))
        if not res_files:
            continue
        with open(res_files[-1], 'r', encoding='utf-8') as f:
            res = json.load(f)
        inst_data = res.get(inst, {})
        data[inst] = {}
        for algo in ALGO_ORDER_ABLATION:
            file_key = key_map[algo]
            runs = inst_data.get(file_key, [])
            hvs = [float(r.get('final_hv', 0)) for r in runs]
            data[inst][algo] = {
                'mean': float(np.mean(hvs)) if hvs else 0.0,
                'std': float(np.std(hvs, ddof=1)) if len(hvs) > 1 else 0.0,
                'n_runs': len(hvs),
            }
    return data


def _plot_hv_bar(ax, data, title, algo_order, ylabel='Hypervolume (HV)'):
    """绘制带误差棒的 HV 分组柱状图。"""
    insts = sorted(data.keys())
    n = len(insts)
    m = len(algo_order)
    x = np.arange(n)
    width = 0.7 / m

    for idx, algo in enumerate(algo_order):
        means = [data[inst][algo]['mean'] for inst in insts]
        stds = [data[inst][algo]['std'] for inst in insts]
        style = ALGO_STYLE[algo]
        offset = width * (idx - (m - 1) / 2)
        bars = ax.bar(x + offset, means, width, yerr=stds, label=ALGO_ALIAS[algo],
                      color=style['color'], alpha=0.85, capsize=3,
                      error_kw={'elinewidth': 1.2, 'ecolor': C_GRAY},
                      edgecolor='white', linewidth=0.5, zorder=2)

    ax.set_xlabel('Instance', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in insts])
    ax.legend(loc='upper right', framealpha=0.95, fontsize=9)
    ax.set_ylim(0, min(1.05, ax.get_ylim()[1] * 1.05))
    style_ax(ax)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', zorder=1)
    ax.set_axisbelow(True)


def plot_benchmark_hv(results_dir='results'):
    """生成 Benchmark HV 对比柱状图。"""
    data = _load_benchmark_hv(results_dir)
    if not data:
        return
    nr = list(data.values())[0]['rmoea_d'].get('n_runs', 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    _plot_hv_bar(ax, data, f'Benchmark HV Comparison (N={nr}, mean ± std)',
                 ALGO_ORDER_BENCHMARK)
    source_footer(fig, f'N={nr}')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    filepath = os.path.join(BENCHMARK_DIR, 'hv_comparison.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("Benchmark HV comparison saved -> %s", filepath)


def plot_ablation_hv(results_dir='results'):
    """生成 Ablation HV 对比柱状图。"""
    data = _load_ablation_hv(results_dir)
    if not data:
        return
    nr = list(data.values())[0]['rmoea_d'].get('n_runs', 1)
    fig, ax = plt.subplots(figsize=(14, 6))
    _plot_hv_bar(ax, data, f'Ablation Study HV Comparison (N={nr}, mean ± std)',
                 ALGO_ORDER_ABLATION)
    source_footer(fig, f'N={nr}')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(ABLATION_DIR, exist_ok=True)
    filepath = os.path.join(ABLATION_DIR, 'hv_comparison.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("Ablation HV comparison saved -> %s", filepath)


# ═══════════════════════════════════ 主入口 ═══════════════════════════════════

def main():
    """生成 PF 与 HV 两组补充对比图。"""
    data = load_experiment_data('results')
    if not data:
        logger.error("No experiment data found. Run `python main.py run_all` first.")
        return

    t0 = time.perf_counter()
    plot_benchmark_pareto_fronts(data)
    plot_ablation_pareto_fronts(data)
    plot_benchmark_hv('results')
    plot_ablation_hv('results')
    elapsed = time.perf_counter() - t0
    logger.info("Comparison charts done in %.2fs", elapsed)


if __name__ == '__main__':
    main()
