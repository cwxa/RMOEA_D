#!/usr/bin/env python3
"""
TFN（三角模糊数）对比表格图生成模块
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
只生成两张图表：
  1. charts/benchmark/benchmark_tfn_comparison_table.png
     — RMOEA/D vs MOEA/D 在 Mk01–Mk10 上的 Makespan/Workload TFN 平均值与显著性
  2. charts/ablation/ablation_tfn_comparison_table.png
     — Full / Q-PAS Only / RVNS Only / MOEA/D 在 Mk01–Mk10 上的 Makespan/Workload TFN 平均值与显著性

数据来源：results/experiment/{inst}/run_*.json（每轮运行已包含 4 种算法变体的 fuzzy_makespan / fuzzy_workload）

排版风格（SCI / booktabs 三线表）
--------------------------------
- 三线表：仅保留顶线、表头分隔线、底线，无竖线、无网格、无隔行底色；
- 分组多级表头：每个算法变体名称横跨 t₁/t₂/t₃ 三列，并以该变体主题色淡底 + 同色粗体文字标识；
- 最优值高亮：每行按清晰值 (t₁+2t₂+t₃)/4 判定最优变体，用其主题色极淡底纹 + 加粗数值标示；
- 衬线字体（Times New Roman + STIX 数学字体），显著性以 p 值上标星号表示；
- 脚注采用居中方法学说明 + 右下角数据来源戳。
"""

import os
import json
import glob
import logging
import time
from pathlib import Path

import numpy as np

# ── 强制使用非 GUI 后端，确保线程安全渲染 ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── 共享绘图工具 ──
try:
    from .plot_helpers import (
        C_RMOEA, C_MOEA, C_FULL, C_QPAS, C_RVNS, C_MOEAD,
        C_IMPROVE, C_DECLINE, C_GRAY, C_GRID,
        set_data_timestamp, source_footer, style_ax,
    )
except ImportError:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_helpers import (
        C_RMOEA, C_MOEA, C_FULL, C_QPAS, C_RVNS, C_MOEAD,
        C_IMPROVE, C_DECLINE, C_GRAY, C_GRID,
        set_data_timestamp, source_footer, style_ax,
    )

logger = logging.getLogger(__name__)

# ═══════════════════════════════════ 常量 ═══════════════════════════════════

CHARTS_BASE = 'charts'
BENCHMARK_DIR = os.path.join(CHARTS_BASE, 'benchmark')
ABLATION_DIR = os.path.join(CHARTS_BASE, 'ablation')
EXPERIMENT_DIR = 'results/experiment'

# 消融实验中 rmoea_d 的显示别名
ALGO_ALIAS = {
    'rmoea_d': 'Full',
    'qpas_only': 'Q-PAS',
    'rvns_only': 'RVNS',
    'moea_d': 'MOEA/D',
}
ALGO_ORDER_BENCHMARK = ['rmoea_d', 'moea_d']
ALGO_ORDER_ABLATION = ['rmoea_d', 'qpas_only', 'rvns_only', 'moea_d']
ALGO_COLORS = {
    'rmoea_d': C_FULL,
    'qpas_only': C_QPAS,
    'rvns_only': C_RVNS,
    'moea_d': C_MOEAD,
}

# Brandimarte 实例规模（工件数 × 机器数），取自 doc.md
SIZE_MAP = {
    'mk01': '10\u00d76', 'mk02': '10\u00d76', 'mk03': '15\u00d78', 'mk04': '15\u00d78',
    'mk05': '15\u00d74', 'mk06': '10\u00d710', 'mk07': '20\u00d75', 'mk08': '20\u00d710',
    'mk09': '20\u00d710', 'mk10': '20\u00d715',
}

# ── SCI/出版排版：衬线字体 + 数学字体 ──
SCI_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'STIXGeneral'],
    'mathtext.fontset': 'stix',
    'axes.grid': False,
    'axes.unicode_minus': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
}

# ── 表线中性色 ──
_RULE_DARK = '#1A1A1A'
_RULE_LIGHT = '#9A9A9A'
_SEP_LIGHT = '#D8D8D8'
_TEXT_DARK = '#1A1A1A'
_TEXT_SUB = '#5A5A5A'


# ═══════════════════════════════════ 数据加载 ═══════════════════════════════════

def load_experiment_tfn(results_dir='results'):
    """
    从 results/experiment/{inst}/run_*.json 加载每轮运行的 TFN 数据。
    返回结构：{instance: {algo: {metric: {'t1': [...], 't2': [...], 't3': [...]}}}}
    """
    base = Path(results_dir) / 'experiment'
    if not base.exists():
        logger.error("Experiment directory not found: %s", base)
        return {}

    data = {}
    t0 = time.perf_counter()
    for inst_dir in sorted(base.glob('mk*')):
        inst = inst_dir.name.lower()
        data[inst] = {}
        run_files = sorted(inst_dir.glob('run_*.json'))
        logger.info("Loading %d runs for %s", len(run_files), inst.upper())

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
                fm = entry.get('fuzzy_makespan', {})
                fw = entry.get('fuzzy_workload', {})

                if algo not in data[inst]:
                    data[inst][algo] = {
                        'makespan': {'t1': [], 't2': [], 't3': []},
                        'workload': {'t1': [], 't2': [], 't3': []},
                    }

                if isinstance(fm, dict) and 'best' in fm:
                    best = fm['best']
                    data[inst][algo]['makespan']['t1'].append(float(best.get('t1', 0)))
                    data[inst][algo]['makespan']['t2'].append(float(best.get('t2', 0)))
                    data[inst][algo]['makespan']['t3'].append(float(best.get('t3', 0)))

                if isinstance(fw, dict) and 'best' in fw:
                    best = fw['best']
                    data[inst][algo]['workload']['t1'].append(float(best.get('t1', 0)))
                    data[inst][algo]['workload']['t2'].append(float(best.get('t2', 0)))
                    data[inst][algo]['workload']['t3'].append(float(best.get('t3', 0)))

    elapsed = time.perf_counter() - t0
    logger.info("TFN data loaded: %d instances in %.2fs", len(data), elapsed)
    return data


def _aggregate_tfn(values):
    """计算一组 TFN 参数的 mean ± std。"""
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def compute_tfn_stats(data):
    """
    聚合为：{instance: {algo: {metric: {'t1': (mean, std), 't2': ..., 't3': ..., 'clear': [...]}}}}
    """
    stats = {}
    for inst, algos in data.items():
        stats[inst] = {}
        for algo, metrics in algos.items():
            stats[inst][algo] = {}
            for metric, tfn in metrics.items():
                t1_mean, t1_std = _aggregate_tfn(tfn['t1'])
                t2_mean, t2_std = _aggregate_tfn(tfn['t2'])
                t3_mean, t3_std = _aggregate_tfn(tfn['t3'])
                clear = [(a + 2 * b + c) / 4.0 for a, b, c in zip(tfn['t1'], tfn['t2'], tfn['t3'])]
                stats[inst][algo][metric] = {
                    't1': (t1_mean, t1_std),
                    't2': (t2_mean, t2_std),
                    't3': (t3_mean, t3_std),
                    'clear': clear,
                }
    return stats


# ═══════════════════════════════════ 统计检验 ═══════════════════════════════════

def _wilcoxon_test(a, b, alpha=0.05):
    """安全的 Wilcoxon 符号秩检验。"""
    from scipy.stats import wilcoxon
    min_len = min(len(a), len(b))
    if min_len < 3:
        return None, 'n.s.'
    try:
        if np.allclose(a[:min_len], b[:min_len]):
            return 1.0, 'n.s.'
        stat, p = wilcoxon(a[:min_len], b[:min_len], zero_method='zsplit')
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
        return float(p), sig
    except Exception as e:
        logger.debug("Wilcoxon failed: %s", e)
        return None, 'n.s.'


def _friedman_test(*groups):
    """Friedman 检验（≥3 组）。"""
    from scipy.stats import friedmanchisquare
    if len(groups) < 3 or any(len(g) < 2 for g in groups):
        return None, 'n.s.'
    try:
        # 保证等长
        min_len = min(len(g) for g in groups)
        groups = [g[:min_len] for g in groups]
        stat, p = friedmanchisquare(*groups)
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
        return float(p), sig
    except Exception as e:
        logger.debug("Friedman failed: %s", e)
        return None, 'n.s.'


# ═══════════════════════════════════ 排版工具 ═══════════════════════════════════

def _format_val(mean, std):
    """格式化为 mean±std，统一 1 位小数。"""
    return f'{mean:.1f}±{std:.1f}'


def _fmt_pvalue(p, sig):
    """p 值 + 上标显著性标记（booktabs 风格，SCI 常用写法）。"""
    if p is None:
        return 'n.a.'
    stars = {'***': '***', '**': '**', '*': '*', 'n.s.': r'\mathrm{n.s.}'}.get(sig, '')
    if stars:
        return f'$p$ = {p:.3f}$^{{{stars}}}$'
    return f'$p$ = {p:.3f}'


def _col_edges(n_var, w_inst, w_size, w_p, w_best, sub_per_var=3):
    """计算各列左右边界（x 归一化到 [0,1]），返回长度 = 列数 + 1。"""
    n_data = n_var * sub_per_var
    w_data = (1.0 - w_inst - w_size - w_p - w_best) / n_data
    edges = [0.0, w_inst, w_inst + w_size]
    for _ in range(n_data):
        edges.append(edges[-1] + w_data)
    edges.append(edges[-1] + w_p)
    edges.append(edges[-1] + w_best)
    return edges, n_data


def _draw_tfn_three_line_table(ax, *, panel_label, row_labels, sizes,
                               group_names, group_colors, cells, pvals, winners,
                               idx_inst=0, idx_size=1, head_fs=9.0, group_fs=9.8,
                               data_fs=8.0, tint_alpha=0.14):
    """
    绘制一张 SCI/booktabs 三线表。

    cells  : list[list[tuple]]  第 r 行、第 v 变体的 (t1_str, t2_str, t3_str)
    pvals  : list[str]          每行 Friedman p 值文本（含上标星号）
    winners: list[int]          每行最优变体索引（按清晰值）
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    n_rows = len(row_labels)
    n_var = len(group_names)
    sub_per_var = 3
    n_data = n_var * sub_per_var

    # ── 列宽 ──
    edges, _ = _col_edges(n_var, w_inst=0.062, w_size=0.052, w_p=0.112, w_best=0.086)
    # 列索引：0=Instance, 1=Size, 2..2+n_data-1=data, then Friedman, Best
    i_p = 2 + n_data
    i_best = i_p + 1

    def xc(i):
        return 0.5 * (edges[i] + edges[i + 1])

    # ── 纵向布局：自适应填满面板高度 ──
    y_top = 0.995
    y_bot = 0.020
    avail = y_top - y_bot
    group_h = 0.085 * avail
    sub_h = 0.070 * avail
    row_h = (avail - group_h - sub_h) / n_rows

    y_g0 = y_top
    y_g1 = y_g0 - group_h        # 分组行 / 子表头行分界
    y_s1 = y_g1 - sub_h          # 子表头行 / 数据行分界

    # ── 0) 最优变体列底纹（zorder 最低，先画）──
    for r in range(n_rows):
        yc = y_s1 - row_h * (r + 0.5)
        v = winners[r]
        x_l = edges[2 + v * sub_per_var]
        x_r = edges[2 + v * sub_per_var + sub_per_var]
        ax.add_patch(Rectangle((x_l, yc - row_h / 2), x_r - x_l, row_h,
                               facecolor=group_colors[v], alpha=tint_alpha,
                               edgecolor='none', zorder=0))

    # ── 1) 分组表头横跨色带 + 变体名（同色粗体）──
    for v, (name, color) in enumerate(zip(group_names, group_colors)):
        x_l = edges[2 + v * sub_per_var]
        x_r = edges[2 + v * sub_per_var + sub_per_var]
        ax.add_patch(Rectangle((x_l, y_g1), x_r - x_l, group_h,
                               facecolor=color, alpha=0.16, edgecolor='none', zorder=1))
        ax.text(0.5 * (x_l + x_r), 0.5 * (y_g0 + y_g1), name,
                ha='center', va='center', fontsize=group_fs,
                fontweight='bold', color=color, zorder=3)

    # ── 2) t₁/t₂/t₃ 子表头 ──
    for v in range(n_var):
        for s, sub in enumerate((r'$t_1$', r'$t_2$', r'$t_3$')):
            ax.text(xc(2 + v * sub_per_var + s), 0.5 * (y_g1 + y_s1), sub,
                    ha='center', va='center', fontsize=head_fs - 0.6,
                    color=_TEXT_SUB, zorder=3)

    # ── 3) 左侧跨两行的行标签 ──
    ax.text(xc(idx_inst), 0.5 * (y_g0 + y_s1), 'Instance',
            ha='center', va='center', fontsize=head_fs, fontweight='bold', color=_TEXT_DARK)
    ax.text(xc(idx_size), 0.5 * (y_g0 + y_s1), 'Size',
            ha='center', va='center', fontsize=head_fs, fontweight='bold', color=_TEXT_DARK)

    # ── 4) 右侧跨两行的列标签 ──
    ax.text(xc(i_p), 0.5 * (y_g0 + y_s1), 'Friedman test',
            ha='center', va='center', fontsize=head_fs, fontweight='bold', color=_TEXT_DARK)
    ax.text(xc(i_best), 0.5 * (y_g0 + y_s1), 'Best',
            ha='center', va='center', fontsize=head_fs, fontweight='bold', color=_TEXT_DARK)

    # ── 5) 数据行 ──
    for r in range(n_rows):
        yc = y_s1 - row_h * (r + 0.5)

        ax.text(xc(idx_inst), yc, row_labels[r], ha='center', va='center',
                fontsize=data_fs + 0.3, fontweight='bold', color=_TEXT_DARK, zorder=3)
        ax.text(xc(idx_size), yc, sizes[r], ha='center', va='center',
                fontsize=data_fs - 0.8, color=_TEXT_SUB, zorder=3)

        v_best = winners[r]
        for v in range(n_var):
            for s in range(sub_per_var):
                ax.text(xc(2 + v * sub_per_var + s), yc, cells[r][v][s],
                        ha='center', va='center', fontsize=data_fs, zorder=3,
                        color=_TEXT_DARK, fontweight=('bold' if v == v_best else 'normal'))

        ax.text(xc(i_p), yc, pvals[r], ha='center', va='center',
                fontsize=data_fs, color=_TEXT_DARK, zorder=3)
        ax.text(xc(i_best), yc, group_names[v_best], ha='center', va='center',
                fontsize=data_fs + 0.3, fontweight='bold', color=group_colors[v_best], zorder=3)

    # ── 6) 分组竖分隔线（极淡，仅在变体之间）──
    for v in range(1, n_var):
        x = edges[2 + v * sub_per_var]
        ax.plot([x, x], [y_g1, y_bot], color=_SEP_LIGHT, lw=0.6, alpha=0.85,
                solid_capstyle='butt', zorder=1)
    for i in (i_p, i_best):
        x = edges[i]
        ax.plot([x, x], [y_g1, y_bot], color=_SEP_LIGHT, lw=0.6, alpha=0.85,
                solid_capstyle='butt', zorder=1)

    # ── 7) 三线表横线 ──
    def hline(y, lw, color, x0=0.0, x1=1.0):
        ax.plot([x0, x1], [y, y], color=color, lw=lw,
                solid_capstyle='butt', zorder=4, clip_on=False)

    xd0, xd1 = edges[2], edges[2 + n_data]
    hline(y_g0, 1.6, _RULE_DARK)                      # 顶线
    hline(y_g1, 0.7, _RULE_LIGHT, x0=xd0, x1=xd1)     # 分组线（仅覆盖算法列，避免穿过跨行标签）
    hline(y_s1, 1.0, _RULE_DARK)                      # 表头与数据分界
    hline(y_bot, 1.6, _RULE_DARK)                      # 底线

    # ── 8) 面板标签 ──
    ax.set_title(panel_label, fontsize=11.2, fontweight='bold',
                 color=_TEXT_DARK, pad=9)


# ═══════════════════════════════════ 表格图生成 ═══════════════════════════════════

def _build_tfn_panel(stats, insts, algos, metric, test='friedman'):
    """为一个指标构建表格行数据。返回 (cells, pvals, winners, clear_means)。"""
    cells, pvals, winners = [], [], []
    for inst in insts:
        row_cells = []
        clears = []
        for algo in algos:
            s = stats[inst][algo][metric]
            row_cells.append((
                _format_val(*s['t1']),
                _format_val(*s['t2']),
                _format_val(*s['t3']),
            ))
            clears.append(s['clear'])
        cells.append(row_cells)

        if test == 'friedman':
            p, sig = _friedman_test(*clears)
        else:
            p, sig = _wilcoxon_test(clears[0], clears[-1])
        pvals.append(_fmt_pvalue(p, sig))
        winners.append(int(np.argmin([np.mean(c) for c in clears])))
    return cells, pvals, winners


def plot_ablation_tfn_table(data, output_dir='charts'):
    """生成消融实验四方案 TFN 对比表格图（SCI 三线表风格）。"""
    stats = compute_tfn_stats(data)
    insts = sorted(stats.keys())
    if not insts:
        logger.error("No ablation TFN data available.")
        return

    out_dir = os.path.join(output_dir, 'ablation')
    os.makedirs(out_dir, exist_ok=True)

    algos = ALGO_ORDER_ABLATION
    group_names = [ALGO_ALIAS[a] for a in algos]
    group_colors = [ALGO_COLORS[a] for a in algos]
    row_labels = [i.upper() for i in insts]
    sizes = [SIZE_MAP.get(i, '') for i in insts]

    cells_ms, p_ms, win_ms = _build_tfn_panel(stats, insts, algos, 'makespan', 'friedman')
    cells_wl, p_wl, win_wl = _build_tfn_panel(stats, insts, algos, 'workload', 'friedman')

    with plt.rc_context(SCI_RC):
        fig = plt.figure(figsize=(12.8, 8.8))
        gs = fig.add_gridspec(2, 1, left=0.030, right=0.985, top=0.925, bottom=0.105, hspace=0.14)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        _draw_tfn_three_line_table(
            ax1, panel_label='(a)  Fuzzy makespan  (mean ± std, 30 runs)',
            row_labels=row_labels, sizes=sizes,
            group_names=group_names, group_colors=group_colors,
            cells=cells_ms, pvals=p_ms, winners=win_ms)

        _draw_tfn_three_line_table(
            ax2, panel_label='(b)  Fuzzy workload  (mean ± std, 30 runs)',
            row_labels=row_labels, sizes=sizes,
            group_names=group_names, group_colors=group_colors,
            cells=cells_wl, pvals=p_wl, winners=win_wl)

        fig.suptitle('Ablation Study — Triangular Fuzzy Number Comparison '
                     'on Brandimarte Instances (Mk01–Mk10)',
                     fontsize=12.8, fontweight='bold', y=0.982, color=_TEXT_DARK)

        # 方法学脚注（居中）
        note_main = ('All entries are mean ± std over 30 independent runs; a lower value is better in every column.  '
                     '$t_1,t_2,t_3$ denote the optimistic, most-likely and pessimistic components of the TFN.')
        note_sub = (r'Significance from the Friedman test on the crisp value $(t_1+2t_2+t_3)/4$ across the four '
                    r'variants ($*\,p<0.05$, $**\,p<0.01$, $***\,p<0.001$, n.s. = not significant).')
        note_sub2 = ('Cells shaded in a variant colour and set in bold mark the best variant '
                     '(lowest crisp value) of that row.')
        fig.text(0.5, 0.072, note_main, ha='center', va='top', fontsize=7.3,
                 color=_TEXT_SUB, style='italic')
        fig.text(0.5, 0.051, note_sub, ha='center', va='top', fontsize=7.3,
                 color=_TEXT_SUB, style='italic')
        fig.text(0.5, 0.030, note_sub2, ha='center', va='top', fontsize=7.3,
                 color=_TEXT_SUB, style='italic')

        source_footer(fig, extra='booktabs-style three-line table | TFN ablation')

        filepath = os.path.join(out_dir, 'ablation_tfn_comparison_table.png')
        fig.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.06)
        plt.close(fig)
    logger.info("Ablation TFN table saved -> %s", filepath)


def plot_benchmark_tfn_table(data, output_dir='charts'):
    """生成 benchmark 双方案 TFN 对比表格图（SCI 三线表风格）。"""
    stats = compute_tfn_stats(data)
    insts = sorted(stats.keys())
    if not insts:
        logger.error("No benchmark TFN data available.")
        return

    out_dir = os.path.join(output_dir, 'benchmark')
    os.makedirs(out_dir, exist_ok=True)

    algos = ALGO_ORDER_BENCHMARK
    group_names = ['RMOEA/D', 'MOEA/D']
    group_colors = [C_RMOEA, C_MOEA]
    row_labels = [i.upper() for i in insts]
    sizes = [SIZE_MAP.get(i, '') for i in insts]

    cells_ms, p_ms, win_ms = _build_tfn_panel(stats, insts, algos, 'makespan', 'wilcoxon')
    cells_wl, p_wl, win_wl = _build_tfn_panel(stats, insts, algos, 'workload', 'wilcoxon')

    with plt.rc_context(SCI_RC):
        fig = plt.figure(figsize=(10.4, 8.4))
        gs = fig.add_gridspec(2, 1, left=0.035, right=0.985, top=0.925, bottom=0.110, hspace=0.15)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        _draw_tfn_three_line_table(
            ax1, panel_label='(a)  Fuzzy makespan  (mean ± std, 30 runs)',
            row_labels=row_labels, sizes=sizes,
            group_names=group_names, group_colors=group_colors,
            cells=cells_ms, pvals=p_ms, winners=win_ms)

        _draw_tfn_three_line_table(
            ax2, panel_label='(b)  Fuzzy workload  (mean ± std, 30 runs)',
            row_labels=row_labels, sizes=sizes,
            group_names=group_names, group_colors=group_colors,
            cells=cells_wl, pvals=p_wl, winners=win_wl)

        fig.suptitle('RMOEA/D versus MOEA/D — Triangular Fuzzy Number Comparison '
                     'on Brandimarte Instances (Mk01–Mk10)',
                     fontsize=12.4, fontweight='bold', y=0.982, color=_TEXT_DARK)

        note_main = ('All entries are mean ± std over 30 independent runs; a lower value is better in every column.  '
                     '$t_1,t_2,t_3$ denote the optimistic, most-likely and pessimistic components of the TFN.')
        note_sub = (r'Significance from the Wilcoxon signed-rank test on the crisp value $(t_1+2t_2+t_3)/4$ '
                    r'($*\,p<0.05$, $**\,p<0.01$, $***\,p<0.001$, n.s. = not significant).')
        note_sub2 = ('Cells shaded in a variant colour and set in bold mark the better algorithm of that row.')
        fig.text(0.5, 0.076, note_main, ha='center', va='top', fontsize=7.3,
                 color=_TEXT_SUB, style='italic')
        fig.text(0.5, 0.053, note_sub, ha='center', va='top', fontsize=7.3,
                 color=_TEXT_SUB, style='italic')
        fig.text(0.5, 0.031, note_sub2, ha='center', va='top', fontsize=7.3,
                 color=_TEXT_SUB, style='italic')

        source_footer(fig, extra='booktabs-style three-line table | TFN benchmark')

        filepath = os.path.join(out_dir, 'benchmark_tfn_comparison_table.png')
        fig.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.06)
        plt.close(fig)
    logger.info("Benchmark TFN table saved -> %s", filepath)


# ═══════════════════════════════════ 主入口 ═══════════════════════════════════

def main():
    """生成两张 TFN 对比表格图。"""
    # 尝试从最新的 aggregate 文件名提取时间戳，用于脚注
    agg_files = sorted(glob.glob('results/benchmark/benchmark_aggregate_*.json'))
    if agg_files:
        ts = os.path.basename(agg_files[-1]).replace('benchmark_aggregate_', '').replace('.json', '')
        set_data_timestamp(ts)
    else:
        set_data_timestamp()

    data = load_experiment_tfn('results')
    if not data:
        logger.error("No experiment data found. Run `python main.py run_all` first.")
        return

    t0 = time.perf_counter()
    plot_benchmark_tfn_table(data)
    plot_ablation_tfn_table(data)
    elapsed = time.perf_counter() - t0
    logger.info("TFN table charts done in %.2fs", elapsed)


if __name__ == '__main__':
    main()
