#!/usr/bin/env python3
"""
专业科研可视化脚本：RMOEA/D vs MOEA/D 对比图表
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 符合学术出版标准的高DPI、专业配色、完整标注
• 三角模糊数(TFN) (t₁, t₂, t₃) 三参数完整展示：
  下界值 t₁ (Earliest)、最可能值 t₂ (Most Likely)、上界值 t₃ (Latest)
• 支持单次运行和多轮独立运行聚合数据（mean ± std）
• 误差棒、Wilcoxon显著性标注、Friedman检验
• 所有图表保存到 charts 文件夹并含数据来源脚注
"""

import os
import json
import glob
import logging
import time
import numpy as np

# ── 强制使用非 GUI 后端，确保线程安全渲染 ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as mticker

# ── 共享绘图工具（跨平台字体、配色、轴样式、脚注） ──
# 兼容模块导入 (from .plot_helpers) 和脚本直跑 (from plot_helpers)
try:
    from .plot_helpers import (
        C_RMOEA, C_MOEA, C_TFN_T1, C_TFN_T3,
        C_IMPROVE, C_DECLINE, C_METRIC3, C_GRAY, C_GRID,
        set_data_timestamp, source_footer, style_ax, dedup_legend,
        get_colormap, _DATA_TIMESTAMP,
    )
except ImportError:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_helpers import (
        C_RMOEA, C_MOEA, C_TFN_T1, C_TFN_T3,
        C_IMPROVE, C_DECLINE, C_METRIC3, C_GRAY, C_GRID,
        set_data_timestamp, source_footer, style_ax, dedup_legend,
        get_colormap, _DATA_TIMESTAMP,
    )

logger = logging.getLogger(__name__)

# ═══════════════════════════════════ 模块专用常量 ═══════════════════════════════

CHARTS_BASE = 'charts'
BENCHMARK_DIR = os.path.join(CHARTS_BASE, 'benchmark')
GANNT_DIR = os.path.join(CHARTS_BASE, 'schedules')
RESULTS_BENCHMARK = 'results/benchmark'
RESULTS_SCHEDULES = 'results/schedules'


def _is_aggregate(agg, inst0):
    rd = agg.get(inst0, {}).get('rmoea_d', {})
    return isinstance(rd, dict) and 'hv_mean' in rd


# ──────────── 数据加载 ────────────

def _get_val(data, key, default=0):
    v = data.get(key, default)
    if isinstance(v, dict):
        return v.get('t2', v.get('best_clear', default))
    return v if v else default


def load_results(results_dir, instance):
    summary_path = os.path.join(results_dir, instance, f'benchmark_summary_{instance}_*.json')
    files = glob.glob(summary_path)
    if not files:
        return None
    with open(sorted(files)[-1], 'r') as f:
        data = json.load(f)
    for algo_key, algo_dir in [('rmoea_d', 'RMOEA_D'), ('moea_d', 'MOEA_D')]:
        detail_dir = os.path.join(results_dir, instance, algo_dir)
        if os.path.isdir(detail_dir):
            dp = os.path.join(detail_dir, f'{instance}_{algo_dir}_*.json')
            df = sorted(glob.glob(dp))
            if df:
                with open(df[-1], 'r') as dfile:
                    ad = json.load(dfile)
                for fk in ['final_pf', 'fuzzy_pf', 'history']:
                    if fk in ad and ad[fk]:
                        data.setdefault(algo_key, {})[fk] = ad[fk]
    return data


def load_all_results(results_dir=RESULTS_BENCHMARK):
    all_r = {}
    for d in sorted(glob.glob(os.path.join(results_dir, 'mk*', ''))):
        inst = os.path.basename(os.path.normpath(d))
        if not inst.startswith('mk'):
            continue
        data = load_results(results_dir, inst)
        if data:
            all_r[inst] = data
    return all_r


def load_aggregate_results(results_dir=RESULTS_BENCHMARK):
    pattern = os.path.join(results_dir, "benchmark_aggregate_*.json")
    files = glob.glob(pattern)
    if not files:
        return None, None
    latest = sorted(files)[-1]
    with open(latest, 'r') as f:
        data = json.load(f)
    set_data_timestamp(os.path.basename(latest).replace('benchmark_aggregate_', '').replace('.json', ''))
    inst_data = data.get('instances', {})
    stats_tests = data.get('statistical_tests', {})
    # 加载每实例详细数据
    for instance in inst_data:
        for algo_key, algo_dir in [('rmoea_d', 'RMOEA_D'), ('moea_d', 'MOEA_D')]:
            dd = os.path.join(results_dir, instance, algo_dir)
            dp = os.path.join(dd, f'{instance}_{algo_dir}_Np*_run0_*.json')
            for df in sorted(glob.glob(dp)):
                try:
                    with open(df, 'r') as f:
                        detail = json.load(f)
                    for field in ['final_pf', 'fuzzy_pf', 'history']:
                        if field in detail and detail[field]:
                            inst_data[instance].setdefault(algo_key, {})[field] = detail[field]
                    break
                except Exception as e:
                    logger.warning("Detail load fail %s/%s: %s", instance, algo_dir, e)
    return inst_data, stats_tests


def load_history(instance, algorithm, results_dir=RESULTS_BENCHMARK):
    pattern = os.path.join(results_dir, instance, algorithm, f'{instance}_{algorithm}_*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    try:
        with open(files[-1], 'r') as f:
            return json.load(f)
    except Exception:
        return None


# ════════════════════════════ 图表1: 传统 Pareto 前沿 ════════════════════════════

def plot_pareto_front_per_instance(all_results):
    for instance in sorted(all_results.keys()):
        if instance.startswith('_'):
            continue
        results = all_results[instance]
        fig, ax = plt.subplots(figsize=(9, 6.5))
        for algo_key, algo_label, color, marker, ms in [
            ('rmoea_d', 'RMOEA/D', C_RMOEA, 'o', 8),
            ('moea_d', 'MOEA/D', C_MOEA, '^', 7)
        ]:
            pf = results.get(algo_key, {}).get('final_pf', [])
            if not pf:
                continue
            ms_vals, wl_vals = [], []
            for p in pf:
                if isinstance(p, (list, tuple)):
                    ms_vals.append(p[0])
                    wl_vals.append(p[1])
                else:
                    mv = p.get('Makespan', p)
                    wv = p.get('Workload', 0)
                    ms_vals.append(mv['t2'] if isinstance(mv, dict) else mv)
                    wl_vals.append(wv['t2'] if isinstance(wv, dict) else wv)
            ax.scatter(ms_vals, wl_vals, c=color, marker=marker, s=ms**2 * 3,
                       alpha=0.75, edgecolors='white', linewidth=0.5,
                       label=f'{algo_label} (|PF|={len(pf)})', zorder=3)
        ax.set_xlabel('Crisp Makespan')
        ax.set_ylabel('Crisp Total Workload')
        ax.set_title(f'{instance.upper()} — Pareto Front\nDefuzzified: (t₁+2t₂+t₃)/4')
        ax.legend(loc='upper right', framealpha=0.9)
        style_ax(ax)
        source_footer(fig)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        d = os.path.join(BENCHMARK_DIR, instance)
        os.makedirs(d, exist_ok=True)
        plt.savefig(os.path.join(d, 'pareto_front.png'), dpi=300)
        plt.close()


# ════════════════════════════ 图表2: TFN 模糊 Pareto 前沿 ═══════════════════════

def plot_fuzzy_pareto_front(all_results):
    """
    三角模糊数Pareto前沿 — 核心TFN图表。
    左图: 带不确定性填充区域的Pareto前沿
    右图: 最佳解三参数分布对比
    """
    instances = sorted(all_results.keys())
    for instance in instances:
        if instance.startswith('_'):
            continue
        results = all_results[instance]
        fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))

        # ── 左图: TFN Pareto 前沿 ──
        ax = axes[0]
        for algo_key, algo_label, color in [
            ('rmoea_d', 'RMOEA/D', C_RMOEA),
            ('moea_d', 'MOEA/D', C_MOEA)
        ]:
            fuzzy_pf = results.get(algo_key, {}).get('fuzzy_pf', [])
            if not fuzzy_pf:
                continue
            ms_t1 = [p['Makespan']['t1'] for p in fuzzy_pf]
            ms_t2 = [p['Makespan']['t2'] for p in fuzzy_pf]
            ms_t3 = [p['Makespan']['t3'] for p in fuzzy_pf]
            wl_t1 = [p['Workload']['t1'] for p in fuzzy_pf]
            wl_t2 = [p['Workload']['t2'] for p in fuzzy_pf]
            wl_t3 = [p['Workload']['t3'] for p in fuzzy_pf]

            # 清晰值
            ms_crisp = [(t1 + 2*t2 + t3) / 4 for t1, t2, t3 in zip(ms_t1, ms_t2, ms_t3)]
            wl_crisp = [(t1 + 2*t2 + t3) / 4 for t1, t2, t3 in zip(wl_t1, wl_t2, wl_t3)]

            # 填充矩形 [t₁, t₃] 表示不确定性区域
            for i in range(len(fuzzy_pf)):
                ax.add_patch(plt.Rectangle(
                    (ms_t1[i], wl_t1[i]),
                    ms_t3[i] - ms_t1[i], wl_t3[i] - wl_t1[i],
                    facecolor=color, alpha=0.06, edgecolor='none', zorder=1))

            # t₂ 散点（最可能值）
            ax.scatter(ms_t2, wl_t2, c=color, marker='o', s=25, alpha=0.4,
                       edgecolors='none', zorder=2)
            # 清晰值散点（菱形）
            ax.scatter(ms_crisp, wl_crisp, c=color, marker='D', s=50, alpha=0.85,
                       edgecolors='white', linewidth=0.8, zorder=3,
                       label=f'{algo_label} Crisp (|PF|={len(fuzzy_pf)})')

        ax.set_xlabel('Makespan — TFN (t₁, t₂, t₃)')
        ax.set_ylabel('Workload — TFN (t₁, t₂, t₃)')
        ax.set_title(f'{instance.upper()} — TFN Pareto Front\nShaded=[t₁,t₃] uncertainty; ◆=crisp via (t₁+2t₂+t₃)/4',
                     fontsize=11)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(),
                  loc='upper right', framealpha=0.9, fontsize=8)
        style_ax(ax)

        # ── 右图: 最佳解 TFN 分布 ──
        ax2 = axes[1]
        y_positions, y_labels, bar_data, bar_colors = [], [], [], []
        for idx, (algo_key, algo_label, color) in enumerate([
            ('rmoea_d', 'RMOEA/D', C_RMOEA),
            ('moea_d', 'MOEA/D', C_MOEA)
        ]):
            fuzzy_pf = results.get(algo_key, {}).get('fuzzy_pf', [])
            if not fuzzy_pf:
                continue
            best_idx = min(range(len(fuzzy_pf)),
                           key=lambda i: fuzzy_pf[i]['Makespan']['t2'])
            best = fuzzy_pf[best_idx]
            y_positions.extend([idx * 3 + 0.8, idx * 3 + 1.8])
            y_labels.extend([f'{algo_label}\nMakespan', f'{algo_label}\nWorkload'])
            bar_colors.extend([color, color])
            bar_data.extend([
                (best['Makespan']['t1'], best['Makespan']['t2'], best['Makespan']['t3']),
                (best['Workload']['t1'], best['Workload']['t2'], best['Workload']['t3'])
            ])

        for y, (t1, t2, t3), color in zip(y_positions, bar_data, bar_colors):
            h = 0.55
            ax2.barh(y, t2 - t1, height=h, left=t1, color=color, alpha=0.22, edgecolor='none')
            ax2.barh(y, t3 - t2, height=h, left=t2, color=color, alpha=0.22, edgecolor='none')
            ymin = (y - h/2 - 0.15) / (max(y_positions) + 1)
            ymax = (y + h/2 + 0.15) / (max(y_positions) + 1)
            ax2.axvline(x=t2, ymin=ymin, ymax=ymax, color=color, linewidth=3.5, alpha=0.95)
            ax2.text(t1, y + 0.30, f't₁={t1:.0f}', fontsize=6.5, ha='center',
                     color=C_TFN_T1, fontweight='bold')
            ax2.text(t2, y - 0.30, f't₂={t2:.0f}', fontsize=7.5, ha='center',
                     color=color, fontweight='bold')
            ax2.text(t3, y + 0.30, f't₃={t3:.0f}', fontsize=6.5, ha='center',
                     color=C_TFN_T3, fontweight='bold')
            crisp = (t1 + 2*t2 + t3) / 4
            ax2.text(crisp, y - 0.06, '|', fontsize=6, ha='center', va='center',
                     color='black', alpha=0.55)

        ax2.set_yticks(y_positions)
        ax2.set_yticklabels(y_labels, fontsize=8)
        ax2.set_xlabel('Triangular Fuzzy Number Value (t₁, t₂, t₃)')
        ax2.set_title(f'{instance.upper()} — Best Solution TFN\n'
                      't₁=Earliest  t₂=Most Likely  t₃=Latest  | =crisp',
                      fontsize=11)
        from matplotlib.lines import Line2D
        ax2.legend(handles=[
            Line2D([0],[0], color=C_TFN_T1, lw=6, alpha=0.25, label='t₁ (Lower bound)'),
            Line2D([0],[0], color='black', lw=3, label='t₂ (Most Likely)'),
            Line2D([0],[0], color=C_TFN_T3, lw=6, alpha=0.25, label='t₃ (Upper bound)'),
        ], loc='lower right', fontsize=7, framealpha=0.9, ncol=3)
        style_ax(ax2)

        source_footer(fig)
        plt.tight_layout(rect=[0, 0.04, 1, 0.98])
        d = os.path.join(BENCHMARK_DIR, instance)
        os.makedirs(d, exist_ok=True)
        plt.savefig(os.path.join(d, 'fuzzy_pareto.png'), dpi=300)
        plt.close()


# ════════════════════════ 图表3: TFN Makespan 三参数对比 ═══════════════════════

def plot_fuzzy_makespan_comparison(agg_results):
    instances = sorted([k for k in agg_results if not k.startswith('_')])
    if not instances:
        return
    x = np.arange(len(instances))
    w = 0.12
    is_agg = _is_aggregate(agg_results, instances[0])
    fig, ax = plt.subplots(figsize=(14, 7))

    for ao, (ak, al, bc) in enumerate([
        ('rmoea_d', 'RMOEA/D', C_RMOEA), ('moea_d', 'MOEA/D', C_MOEA)
    ]):
        for to, (tk, tl) in enumerate([
            ('t1', 't₁ (Earliest)'), ('t2', 't₂ (Most Likely)'), ('t3', 't₃ (Latest)')
        ]):
            vals = []
            for inst in instances:
                r = agg_results[inst].get(ak, {})
                if is_agg:
                    v = r.get(f'fuzzy_makespan_{tk}_mean', 0)
                else:
                    fm = r.get('fuzzy_makespan', {})
                    v = fm.get('best', {}).get(tk, 0) if fm else 0
                vals.append(v if v else 0)
            pos = x + ao * w * 3 + to * w - w * 3
            alpha_v = 0.6 if tk == 't2' else 0.32
            ax.bar(pos, vals, w, color=bc, alpha=alpha_v,
                   edgecolor=bc if tk == 't2' else 'white',
                   linewidth=0.6 if tk == 't2' else 0.3,
                   label=f'{al} {tl}', zorder=2 if tk == 't2' else 1)

    ax.set_xlabel('Instance')
    ax.set_ylabel('Makespan — TFN Value')
    ax.set_title('Fuzzy Makespan: t₁(Earliest) / t₂(Most Likely) / t₃(Latest)')
    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in instances])
    dedup_legend(ax, ncol=2, fs=7)
    style_ax(ax)
    source_footer(fig)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(BENCHMARK_DIR, 'fuzzy_makespan_comparison.png'), dpi=300)
    plt.close()


# ════════════════════════ 图表4: TFN Workload 三参数对比 ═══════════════════════

def plot_fuzzy_workload_comparison(agg_results):
    instances = sorted([k for k in agg_results if not k.startswith('_')])
    if not instances:
        return
    x = np.arange(len(instances))
    w = 0.12
    is_agg = _is_aggregate(agg_results, instances[0])
    fig, ax = plt.subplots(figsize=(14, 7))

    for ao, (ak, al, bc) in enumerate([
        ('rmoea_d', 'RMOEA/D', C_RMOEA), ('moea_d', 'MOEA/D', C_MOEA)
    ]):
        for to, (tk, tl) in enumerate([
            ('t1', 't₁ (Earliest)'), ('t2', 't₂ (Most Likely)'), ('t3', 't₃ (Latest)')
        ]):
            vals = []
            for inst in instances:
                r = agg_results[inst].get(ak, {})
                if is_agg:
                    v = r.get(f'fuzzy_workload_{tk}_mean', 0)
                else:
                    fw = r.get('fuzzy_workload', {})
                    v = fw.get('best', {}).get(tk, 0) if fw else 0
                vals.append(v if v else 0)
            pos = x + ao * w * 3 + to * w - w * 3
            alpha_v = 0.6 if tk == 't2' else 0.32
            ax.bar(pos, vals, w, color=bc, alpha=alpha_v,
                   edgecolor=bc if tk == 't2' else 'white',
                   linewidth=0.6 if tk == 't2' else 0.3,
                   label=f'{al} {tl}', zorder=2 if tk == 't2' else 1)

    ax.set_xlabel('Instance')
    ax.set_ylabel('Workload — TFN Value')
    ax.set_title('Fuzzy Workload: t₁(Earliest) / t₂(Most Likely) / t₃(Latest)')
    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in instances])
    dedup_legend(ax, ncol=2, fs=7)
    style_ax(ax)
    source_footer(fig)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(BENCHMARK_DIR, 'fuzzy_workload_comparison.png'), dpi=300)
    plt.close()


# ════════════════════════ 图表5: 每实例 TFN 区间图 ═══════════════════════════

def plot_fuzzy_range_per_instance(agg_results):
    """
    每个实例单独绘制TFN不确定性区间。
    渐变填充 [t₁, t₃]，粗线 t₂，虚线 crisp。
    """
    instances = sorted([k for k in agg_results if not k.startswith('_')])
    if not instances:
        return
    is_agg = _is_aggregate(agg_results, instances[0])

    for instance in instances:
        r = agg_results[instance]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax_idx, (mk, ml) in enumerate([
            ('fuzzy_makespan', 'Makespan'), ('fuzzy_workload', 'Workload')
        ]):
            ax = axes[ax_idx]
            for i, (ak, al, c) in enumerate([
                ('rmoea_d', 'RMOEA/D', C_RMOEA), ('moea_d', 'MOEA/D', C_MOEA)
            ]):
                if is_agg:
                    t1v = r.get(ak, {}).get(f'{mk}_t1_mean', 0)
                    t2v = r.get(ak, {}).get(f'{mk}_t2_mean', 0)
                    t3v = r.get(ak, {}).get(f'{mk}_t3_mean', 0)
                else:
                    fm = r.get(ak, {}).get(mk, {})
                    best = fm.get('best', {}) if fm else {}
                    t1v, t2v, t3v = best.get('t1', 0), best.get('t2', 0), best.get('t3', 0)
                if t2v == 0:
                    continue

                xp = i * 4 + 1.5
                bw = 0.7

                # 渐变填充（t₁→t₃ 由浅到深再浅）
                grad = np.linspace(0, 1, 50).reshape(50, 1)
                ax.imshow(grad, aspect='auto', cmap=plt.cm.RdYlGn_r,
                          extent=[t1v, t3v, xp - bw/2, xp + bw/2],
                          alpha=0.35, vmin=0, vmax=1, zorder=1)

                # t₂ 粗线
                ax.plot([t2v, t2v], [xp - bw/2 - 0.1, xp + bw/2 + 0.1],
                        color=c, linewidth=4, alpha=0.9, zorder=3, solid_capstyle='round')

                # 标注 t₁, t₂, t₃
                off = bw/2 + 0.08
                ax.text(t1v, xp + off, f't₁={t1v:.0f}', fontsize=6.5, ha='center',
                        color=C_TFN_T1, fontweight='bold')
                ax.text(t2v, xp - off, f't₂={t2v:.0f}', fontsize=7.5, ha='center',
                        color=c, fontweight='bold')
                ax.text(t3v, xp + off, f't₃={t3v:.0f}', fontsize=6.5, ha='center',
                        color=C_TFN_T3, fontweight='bold')

                # crisp 虚线
                crisp = (t1v + 2*t2v + t3v) / 4
                ax.plot([crisp, crisp], [xp - bw/2 - 0.12, xp + bw/2 + 0.12],
                        color='black', linewidth=1.5, linestyle='--', alpha=0.5, zorder=4)
                ax.text(crisp, xp + bw/2 + 0.16, f'crisp={crisp:.0f}',
                        fontsize=6, ha='center', color='black', alpha=0.65)

            ax.set_yticks([1.5, 5.5])
            ax.set_yticklabels(['RMOEA/D', 'MOEA/D'], fontsize=10)
            ax.set_xlabel(f'{ml} — TFN (t₁, t₂, t₃)')
            ax.set_title(f'{instance.upper()} — {ml} TFN\nGradient=[t₁,t₃] | =t₂  -- =crisp', fontsize=10)
            ax.invert_yaxis()
            style_ax(ax)

        source_footer(fig)
        plt.tight_layout(rect=[0, 0.04, 1, 0.98])
        d = os.path.join(BENCHMARK_DIR, instance)
        os.makedirs(d, exist_ok=True)
        plt.savefig(os.path.join(d, 'fuzzy_range.png'), dpi=300)
        plt.close()


# ════════════════════════ 图表6: 收敛曲线 ═════════════════════════════════════

def plot_convergence_curve_per_instance(all_results):
    for instance in sorted(all_results.keys()):
        if instance.startswith('_'):
            continue
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for algo_dir, algo_label, color, ls in [
            ('RMOEA_D', 'RMOEA/D', C_RMOEA, '-'),
            ('MOEA_D', 'MOEA/D', C_MOEA, '--')
        ]:
            hist = load_history(instance, algo_dir)
            if hist and 'history' in hist and isinstance(hist['history'], list):
                hd = hist['history']
                gens = [h.get('gen', i) for i, h in enumerate(hd)]
                hvs = [h.get('hv', 0) for h in hd]
                ax.plot(gens, hvs, color=color, linewidth=2.2, linestyle=ls,
                        label=algo_label, alpha=0.85)
                ax.scatter([gens[-1]], [hvs[-1]], c=color, s=30, zorder=5)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Hypervolume (HV)')
        ax.set_title(f'{instance.upper()} — HV Convergence')
        ax.legend(loc='lower right', framealpha=0.9)
        style_ax(ax)
        source_footer(fig)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        d = os.path.join(BENCHMARK_DIR, instance)
        os.makedirs(d, exist_ok=True)
        plt.savefig(os.path.join(d, 'convergence.png'), dpi=300)
        plt.close()


# ════════════════════════ 图表7: HV 对比 ═══════════════════════════════════════

def plot_hv_comparison(agg_results):
    instances = sorted([k for k in agg_results if not k.startswith('_')])
    if not instances:
        return
    x = np.arange(len(instances))
    w = 0.35
    is_agg = _is_aggregate(agg_results, instances[0])

    if is_agg:
        rh = [agg_results[i].get('rmoea_d', {}).get('hv_mean', 0) for i in instances]
        re = [agg_results[i].get('rmoea_d', {}).get('hv_std', 0) for i in instances]
        mh = [agg_results[i].get('moea_d', {}).get('hv_mean', 0) for i in instances]
        me = [agg_results[i].get('moea_d', {}).get('hv_std', 0) for i in instances]
    else:
        rh = [agg_results[i].get('rmoea_d', {}).get('final_hv', 0) for i in instances]
        re = [0]*len(instances)
        mh = [agg_results[i].get('moea_d', {}).get('final_hv', 0) for i in instances]
        me = [0]*len(instances)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w/2, rh, w, yerr=re, label='RMOEA/D', color=C_RMOEA, alpha=0.85,
           capsize=4, error_kw={'elinewidth': 1.5, 'ecolor': C_GRAY},
           edgecolor='white', linewidth=0.5)
    ax.bar(x + w/2, mh, w, yerr=me, label='MOEA/D', color=C_MOEA, alpha=0.85,
           capsize=4, error_kw={'elinewidth': 1.5, 'ecolor': C_GRAY},
           edgecolor='white', linewidth=0.5)
    for i, (vr, vm) in enumerate(zip(rh, mh)):
        if vm > 0:
            imp = ((vr - vm) / vm * 100)
            clr = C_IMPROVE if imp > 0 else C_DECLINE
            ax.annotate(f'{"↑" if imp>0 else "↓"}{abs(imp):.1f}%',
                        (x[i], max(vr, vm)), textcoords="offset points",
                        xytext=(0, 6), ha='center', fontsize=8,
                        color=clr, fontweight='bold')
    ax.set_xlabel('Instance')
    ax.set_ylabel('Hypervolume (HV)')
    nr = agg_results.get('_meta', {}).get('n_runs', 1)
    ax.set_title(f'HV Comparison (N={nr}, mean ± std)' if is_agg else 'HV Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in instances])
    ax.legend(loc='upper left', framealpha=0.9)
    style_ax(ax)
    source_footer(fig, f'N={nr}')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(BENCHMARK_DIR, 'hv_comparison.png'), dpi=300)
    plt.close()


# ════════════════════════ 图表8-9: Makespan / Workload 清晰值对比 ═════════════

def _plot_crisp_comparison(agg_results, metric_key, title_label, filename):
    instances = sorted([k for k in agg_results if not k.startswith('_')])
    if not instances:
        return
    x = np.arange(len(instances))
    w = 0.35
    is_agg = _is_aggregate(agg_results, instances[0])
    mean_key = f'{metric_key}_mean'
    std_key = f'{metric_key}_std'

    rv, mv, re, me = [], [], [], []
    for inst in instances:
        rd = agg_results[inst].get('rmoea_d', {})
        md = agg_results[inst].get('moea_d', {})
        if is_agg:
            rv.append(rd.get(mean_key, 0))
            re.append(rd.get(std_key, 0))
            mv.append(md.get(mean_key, 0))
            me.append(md.get(std_key, 0))
        else:
            rv.append(_get_val(rd, metric_key))
            mv.append(_get_val(md, metric_key))
            re.append(0)
            me.append(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w/2, rv, w, yerr=re, label='RMOEA/D', color=C_RMOEA, alpha=0.85,
           capsize=4, error_kw={'elinewidth': 1.5, 'ecolor': C_GRAY},
           edgecolor='white', linewidth=0.5)
    ax.bar(x + w/2, mv, w, yerr=me, label='MOEA/D', color=C_MOEA, alpha=0.85,
           capsize=4, error_kw={'elinewidth': 1.5, 'ecolor': C_GRAY},
           edgecolor='white', linewidth=0.5)
    for i, (vr, vm) in enumerate(zip(rv, mv)):
        if vm > 0:
            imp = ((vm - vr) / vm * 100)
            clr = C_IMPROVE if imp > 0 else C_DECLINE
            ax.annotate(f'{imp:+.1f}%', (x[i], max(vr, vm)),
                        textcoords="offset points", xytext=(0, 5),
                        ha='center', fontsize=8, color=clr, fontweight='bold')
    ax.set_xlabel('Instance')
    ax.set_ylabel(f'{title_label} (Crisp)')
    nl = ' (mean ± std)' if is_agg else ''
    ax.set_title(f'Crisp {title_label} Comparison{nl}')
    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in instances])
    ax.legend(loc='upper left', framealpha=0.9)
    style_ax(ax)
    source_footer(fig)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(BENCHMARK_DIR, filename), dpi=300)
    plt.close()


def plot_makespan_comparison(agg_results):
    _plot_crisp_comparison(agg_results, 'best_makespan', 'Makespan', 'makespan_comparison.png')


def plot_workload_comparison(agg_results):
    _plot_crisp_comparison(agg_results, 'best_workload', 'Workload', 'workload_comparison.png')


# ════════════════════════ 图表10: 改进幅度汇总 ════════════════════════════════

def plot_improvement_summary(agg_results):
    instances = sorted([k for k in agg_results if not k.startswith('_')])
    if not instances:
        return
    x = np.arange(len(instances))
    w = 0.25
    hv_imp, ms_imp, wl_imp = [], [], []

    for inst in instances:
        r = agg_results[inst]
        hr = r.get('rmoea_d', {}).get('hv_mean', r.get('rmoea_d', {}).get('final_hv', 0))
        hm = r.get('moea_d', {}).get('hv_mean', r.get('moea_d', {}).get('final_hv', 0))
        mr = r.get('rmoea_d', {}).get('best_makespan_mean', _get_val(r.get('rmoea_d', {}), 'best_makespan'))
        mm = r.get('moea_d', {}).get('best_makespan_mean', _get_val(r.get('moea_d', {}), 'best_makespan'))
        wr = r.get('rmoea_d', {}).get('best_workload_mean', _get_val(r.get('rmoea_d', {}), 'best_workload'))
        wm = r.get('moea_d', {}).get('best_workload_mean', _get_val(r.get('moea_d', {}), 'best_workload'))
        hv_imp.append(((hr - hm) / hm * 100) if hm > 0 else 0)
        ms_imp.append(((mm - mr) / mm * 100) if mm > 0 else 0)
        wl_imp.append(((wm - wr) / wm * 100) if wm > 0 else 0)

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = [
        ax.bar(x - w, hv_imp, w, label='HV ↑', color=C_RMOEA, alpha=0.85, edgecolor='white', lw=0.5),
        ax.bar(x, ms_imp, w, label='Makespan ↓', color=C_IMPROVE, alpha=0.85, edgecolor='white', lw=0.5),
        ax.bar(x + w, wl_imp, w, label='Workload ↓', color=C_METRIC3, alpha=0.85, edgecolor='white', lw=0.5),
    ]
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, lw=0.8)
    for i in range(len(instances)):
        for j, vals in enumerate([hv_imp, ms_imp, wl_imp]):
            v = vals[i]
            bar = bars[j][i]
            clr = C_IMPROVE if v > 0 else C_DECLINE
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.3 if v >= 0 else -0.3),
                    f'{v:+.1f}%', ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=6.5, color=clr, fontweight='bold', rotation=90)
    ax.set_xlabel('Instance')
    ax.set_ylabel('Improvement (%) — RMOEA/D vs MOEA/D')
    ax.set_title('Performance Improvement Summary\n(+ = RMOEA/D outperforms MOEA/D)')
    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in instances])
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9, ncol=3)
    style_ax(ax)
    source_footer(fig)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(BENCHMARK_DIR, 'improvement_summary.png'), dpi=300)
    plt.close()


# ════════════════════════ 图表11: PF 大小对比 ═════════════════════════════════

def plot_pf_size_comparison(agg_results):
    instances = sorted([k for k in agg_results if not k.startswith('_')])
    if not instances:
        return
    x = np.arange(len(instances))
    w = 0.35
    is_agg = _is_aggregate(agg_results, instances[0])

    if is_agg:
        rp = [agg_results[i].get('rmoea_d', {}).get('pf_size_mean', 0) for i in instances]
        re = [agg_results[i].get('rmoea_d', {}).get('pf_size_std', 0) for i in instances]
        mp = [agg_results[i].get('moea_d', {}).get('pf_size_mean', 0) for i in instances]
        me = [agg_results[i].get('moea_d', {}).get('pf_size_std', 0) for i in instances]
    else:
        rp = [agg_results[i].get('rmoea_d', {}).get('pf_size', 0) for i in instances]
        re = [0]*len(instances)
        mp = [agg_results[i].get('moea_d', {}).get('pf_size', 0) for i in instances]
        me = [0]*len(instances)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w/2, rp, w, yerr=re, label='RMOEA/D', color=C_RMOEA, alpha=0.85,
           capsize=4, error_kw={'elinewidth': 1.5, 'ecolor': C_GRAY},
           edgecolor='white', linewidth=0.5)
    ax.bar(x + w/2, mp, w, yerr=me, label='MOEA/D', color=C_MOEA, alpha=0.85,
           capsize=4, error_kw={'elinewidth': 1.5, 'ecolor': C_GRAY},
           edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Instance')
    ax.set_ylabel('|PF| Size')
    nl = ' (mean ± std)' if is_agg else ''
    ax.set_title(f'Pareto Front Size Comparison{nl}')
    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in instances])
    ax.legend(loc='upper left', framealpha=0.9)
    style_ax(ax)
    source_footer(fig)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(BENCHMARK_DIR, 'pf_size_comparison.png'), dpi=300)
    plt.close()


# ════════════════════════ 图表12: 统计检验 ═════════════════════════════════════

def plot_statistical_tests(agg_results, stats_tests):
    if not stats_tests or 'wilcoxon_hv' not in stats_tests:
        return
    instances = sorted([k for k in agg_results if not k.startswith('_')])
    if not instances:
        return
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))

    ax = axes[0]
    x = np.arange(len(instances))
    w = 0.3
    for i, inst in enumerate(instances):
        r = agg_results[inst]
        hr, hrs = r.get('rmoea_d', {}).get('hv_mean', 0), r.get('rmoea_d', {}).get('hv_std', 0)
        hm, hms = r.get('moea_d', {}).get('hv_mean', 0), r.get('moea_d', {}).get('hv_std', 0)
        ax.bar(i - w/2, hr, w, yerr=hrs, color=C_RMOEA, alpha=0.8, capsize=3,
               error_kw={'elinewidth': 1.5, 'ecolor': C_GRAY},
               edgecolor='white', lw=0.5, label='RMOEA/D' if i == 0 else '')
        ax.bar(i + w/2, hm, w, yerr=hms, color=C_MOEA, alpha=0.8, capsize=3,
               error_kw={'elinewidth': 1.5, 'ecolor': C_GRAY},
               edgecolor='white', lw=0.5, label='MOEA/D' if i == 0 else '')
        wp = stats_tests.get('wilcoxon_hv', {}).get('per_instance', {}).get(inst, {})
        if wp.get('sig') not in ('ns', 'n.s.', None):
            ymax = max(hr + hrs, hm + hms)
            ax.annotate(wp.get('sig', '★'), (i, ymax * 1.03), ha='center', fontsize=16,
                        color=C_DECLINE, fontweight='bold')
            ax.annotate(f"p={wp.get('p_value', 0):.4f}", (i, ymax * 1.08),
                        ha='center', fontsize=7, color=C_DECLINE, style='italic')
    ax.set_xlabel('Instance')
    ax.set_ylabel('HV (mean ± std)')
    ax.set_title('HV with Wilcoxon Significance\n(★ p<.05  ★★ p<.01  ★★★ p<.001)')
    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in instances])
    ax.legend(loc='upper left', framealpha=0.9)
    style_ax(ax)

    ax2 = axes[1]
    ax2.axis('off')
    ft = stats_tests.get('friedman', {})
    wt = stats_tests.get('wilcoxon_hv', {}).get('overall', {})
    td = [
        ['Test', 'Statistic', 'p-value', 'Significant (α=0.05)'],
        ['Friedman', f"{ft.get('statistic', 'N/A'):.4f}" if ft.get('statistic') else 'N/A',
         f"{ft.get('p_value', 'N/A'):.6f}" if ft.get('p_value') else 'N/A',
         ft.get('significant', 'N/A').upper()],
        ['Wilcoxon', f"{wt.get('statistic', 'N/A'):.2f}" if wt.get('statistic') else 'N/A',
         f"{wt.get('p_value', 'N/A'):.6f}" if wt.get('p_value') else 'N/A',
         wt.get('significant', 'N/A').upper()],
    ]
    tbl = ax2.table(cellText=td, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.2, 0.2, 0.25])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.0)
    for j in range(4):
        tbl[0, j].set_facecolor(C_RMOEA)
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i in [1, 2]:
        if td[i][3] == 'YES':
            tbl[i, 3].set_text_props(color=C_IMPROVE, fontweight='bold')
    ax2.set_title('Statistical Tests (Friedman + Wilcoxon)', fontsize=12, fontweight='bold', pad=20)

    source_footer(fig, 'α=0.05')
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    plt.savefig(os.path.join(BENCHMARK_DIR, 'statistical_tests.png'), dpi=300)
    plt.close()


# ════════════════════════ 图表13: 运行时间对比 ════════════════════════════════

def plot_runtime_comparison(agg_results):
    instances = sorted([k for k in agg_results if not k.startswith('_')])
    if not instances:
        return
    x = np.arange(len(instances))
    w = 0.35
    is_agg = _is_aggregate(agg_results, instances[0])

    if is_agg:
        rt = [agg_results[i].get('rmoea_d', {}).get('time_mean', 0) for i in instances]
        re = [agg_results[i].get('rmoea_d', {}).get('time_std', 0) for i in instances]
        mt = [agg_results[i].get('moea_d', {}).get('time_mean', 0) for i in instances]
        me = [agg_results[i].get('moea_d', {}).get('time_std', 0) for i in instances]
    else:
        rt = [agg_results[i].get('rmoea_d', {}).get('total_time', 0) for i in instances]
        mt = [agg_results[i].get('moea_d', {}).get('total_time', 0) for i in instances]
        re = [0]*len(instances)
        me = [0]*len(instances)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w/2, rt, w, yerr=re, label='RMOEA/D', color=C_RMOEA, alpha=0.85,
           capsize=4, error_kw={'elinewidth': 1.5, 'ecolor': C_GRAY},
           edgecolor='white', linewidth=0.5)
    ax.bar(x + w/2, mt, w, yerr=me, label='MOEA/D', color=C_MOEA, alpha=0.85,
           capsize=4, error_kw={'elinewidth': 1.5, 'ecolor': C_GRAY},
           edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Instance')
    ax.set_ylabel('Runtime (s)')
    nl = ' (mean ± std)' if is_agg else ''
    ax.set_title(f'Runtime Comparison{nl}')
    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in instances])
    ax.legend(loc='upper left', framealpha=0.9)
    style_ax(ax)
    source_footer(fig)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(BENCHMARK_DIR, 'runtime_comparison.png'), dpi=300)
    plt.close()


# ════════════════════════ 图表14: 模糊清晰值对比 ══════════════════════════════

def plot_fuzzy_clear_value_comparison(agg_results):
    """TFN 清晰值 (t₁+2t₂+t₃)/4 对比。Makespan 和 Workload 分开两个子图。"""
    instances = sorted([k for k in agg_results if not k.startswith('_')])
    if not instances:
        return
    x = np.arange(len(instances))
    w = 0.2
    is_agg = _is_aggregate(agg_results, instances[0])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax_idx, (mk, yl) in enumerate([
        ('fuzzy_makespan_clear', 'Makespan Crisp'), ('fuzzy_workload_clear', 'Workload Crisp')
    ]):
        ax = axes[ax_idx]
        for ao, (ak, al, c) in enumerate([
            ('rmoea_d', 'RMOEA/D', C_RMOEA), ('moea_d', 'MOEA/D', C_MOEA)
        ]):
            vals = []
            for inst in instances:
                rd = agg_results[inst].get(ak, {})
                if is_agg:
                    ck = f'fuzzy_{"makespan" if "makespan" in mk else "workload"}_clear_mean'
                    vals.append(rd.get(ck, 0))
                else:
                    v = rd.get('fuzzy_makespan_clear' if 'makespan' in mk else 'fuzzy_workload_clear', 0)
                    vals.append(v if v else 0)
            ax.bar(x + ao * w - w/2, vals, w, color=c, alpha=0.85,
                   label=al, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Instance')
        ax.set_ylabel(yl)
        ax.set_title(f'{yl} — Defuzzified via (t₁+2t₂+t₃)/4')
        ax.set_xticks(x)
        ax.set_xticklabels([i.upper() for i in instances])
        ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
        style_ax(ax)

    source_footer(fig)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(BENCHMARK_DIR, 'fuzzy_clear_value_comparison.png'), dpi=300)
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# 主入口
# ═════════════════════════════════════════════════════════════════════════════

def main():
    """
    当前只生成两张 TFN 对比表格图：
      1. charts/benchmark/benchmark_tfn_comparison_table.png
      2. charts/ablation/ablation_tfn_comparison_table.png
    如需恢复旧版全套图表，请取消下方旧函数调用注释。
    """
    # ── 新增：TFN 表格图 ──
    try:
        from . import tfn_table_charts
    except ImportError:
        import tfn_table_charts
    tfn_table_charts.main()

    # ── 补充：PF 与 HV 对比图 ──
    try:
        from . import comparison_charts
    except ImportError:
        import comparison_charts
    comparison_charts.main()

    # ── 旧版全套图表（已停用）──
    # os.makedirs(BENCHMARK_DIR, exist_ok=True)
    # agg_data, stats_tests = load_aggregate_results()
    # is_aggregate = agg_data is not None and len(agg_data) > 1
    # if is_aggregate:
    #     all_results = agg_data
    #     nr = list(all_results.values())[0].get('n_runs', 1)
    #     logger.info("Loaded aggregate: %d instances (%d runs each)",
    #                 len([k for k in all_results if not k.startswith('_')]), nr)
    # else:
    #     all_results = load_all_results()
    #     if not all_results:
    #         logger.error("No experiment results found.")
    #         return
    #     stats_tests = {}
    #     logger.info("Loaded single-run: %d instances", len(all_results))
    #
    # logger.info("Generating charts... [%s]", "AGGREGATE" if is_aggregate else "SINGLE")
    # plot_pareto_front_per_instance(all_results)
    # plot_fuzzy_pareto_front(all_results)
    # plot_fuzzy_makespan_comparison(all_results)
    # plot_fuzzy_workload_comparison(all_results)
    # plot_fuzzy_range_per_instance(all_results)
    # plot_convergence_curve_per_instance(all_results)
    # plot_hv_comparison(all_results)
    # plot_makespan_comparison(all_results)
    # plot_workload_comparison(all_results)
    # plot_improvement_summary(all_results)
    # plot_pf_size_comparison(all_results)
    # plot_runtime_comparison(all_results)
    # if is_aggregate and stats_tests:
    #     plot_statistical_tests(all_results, stats_tests)
    # plot_fuzzy_clear_value_comparison(all_results)
    #
    # logger.info("Charts saved to %s/", BENCHMARK_DIR)


# ═════════════════════════════════════════════════════════════════════════════
# 甘特图模块
# ═════════════════════════════════════════════════════════════════════════════

def _compute_tfn_makespan(schedule):
    """从调度操作列表计算三角模糊数 makespan (t₁, t₂, t₃)。
    每个操作的 fuzzy_finish = [t₁, t₂, t₃]，整体 TFN makespan 取各维度的最大值。"""
    t1 = max((op.get("fuzzy_finish", [op["finish"]] * 3)[0] for op in schedule), default=0)
    t2 = max((op.get("fuzzy_finish", [op["finish"]] * 3)[1] for op in schedule), default=0)
    t3 = max((op.get("fuzzy_finish", [op["finish"]] * 3)[2] for op in schedule), default=0)
    return t1, t2, t3


def plot_gantt_chart(schedule, instance_name, output_dir=None):
    """绘制单个调度解甘特图（含 TFN 三参数标注）"""
    if not output_dir:
        output_dir = os.path.join(GANNT_DIR, instance_name)
    os.makedirs(output_dir, exist_ok=True)

    machines = sorted(set(op["machine_label"] for op in schedule),
                      key=lambda m: int(m[1:]))
    n_machines = len(machines)
    machine_idx = {m: i for i, m in enumerate(machines)}

    job_ids = sorted(set(op["job_id"] for op in schedule))
    n_jobs = len(job_ids)
    cmap = get_colormap('tab20', max(n_jobs, 20))
    job_color = {jid: cmap(i % 20) for i, jid in enumerate(job_ids)}

    makespan = max(op["finish"] for op in schedule)
    tfn_t1, tfn_t2, tfn_t3 = _compute_tfn_makespan(schedule)

    fig, ax = plt.subplots(figsize=(max(14, makespan / 30), max(5, n_machines * 0.5) + 1.2))

    for op in schedule:
        y = n_machines - 1 - machine_idx[op["machine_label"]]
        start = op["start"]
        duration = op["finish"] - op["start"]
        ax.barh(y, duration, height=0.7, left=start,
                color=job_color[op["job_id"]], edgecolor='black',
                linewidth=0.5, alpha=0.85)

    ax.set_yticks(range(n_machines))
    ax.set_yticklabels(reversed(machines))
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Machine', fontsize=12)
    ax.set_title(f'{instance_name.upper()} — Gantt Chart\n'
                 f'Crisp Makespan = {makespan:.2f}', fontsize=13, fontweight='bold')
    ax.set_xlim(0, makespan * 1.03)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    style_ax(ax)

    # ══════════════════════════════════════════════════════════════
    # TFN 不确定性区间可视化
    # ══════════════════════════════════════════════════════════════
    total_ms = makespan  # 统一变量名，供 TFN 标注使用
    # 灰色半透明带：[t₁, t₃] 表示模糊 makespan 的可能范围
    ax.axvspan(tfn_t1, tfn_t3, color='#2166AC', alpha=0.06, zorder=0)
    # 红色虚线：t₂ (最可能值，defuzzified makespan 的核心分量)
    ax.axvline(x=tfn_t2, color='#D6604D', linewidth=1.2, linestyle='--', alpha=0.6, zorder=2)

    # ══════════════════════════════════════════════════════════════
    # 图表下方信息面板：Job 图例 + TFN 参数 + 数据来源
    # ══════════════════════════════════════════════════════════════
    crisp_tfn = (tfn_t1 + 2 * tfn_t2 + tfn_t3) / 4

    # Job 色块图例（水平居中，位于图表下方）
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=job_color[jid], edgecolor='black',
                      linewidth=0.5, label=f'J{jid + 1}')
        for jid in job_ids
    ]
    fig.legend(handles=legend_handles, loc='upper center', fontsize=7,
               ncol=min(n_jobs, 10), title='Jobs', title_fontsize=8,
               frameon=True, framealpha=0.8, edgecolor=C_GRAY,
               bbox_to_anchor=(0.5, 0.14))

    # TFN 参数信息行（图例下方，紧凑单行）
    tfn_info = (
        f'TFN Makespan:  '
        f't\u2081 = {tfn_t1:.1f}  (Lower bound)  |  '
        f't\u2082 = {tfn_t2:.1f}  (Most likely)  |  '
        f't\u2083 = {tfn_t3:.1f}  (Upper bound)  |  '
        f'Crisp = {crisp_tfn:.1f}  =  (t\u2081 + 2t\u2082 + t\u2083) / 4  '
        f'[Interval: {tfn_t1:.0f} \u2013 {tfn_t3:.0f}]'
    )
    fig.text(0.5, 0.05, tfn_info, ha='center', va='center', fontsize=7.5,
             fontname='DejaVu Sans Mono',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F4F8',
                       edgecolor=C_GRAY, alpha=0.92))

    source_footer(fig, f'Instance: {instance_name}')
    # 为底部信息面板预留空间
    fig.subplots_adjust(bottom=0.22)
    filepath = os.path.join(output_dir, 'gantt_chart.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("Gantt: %s", filepath)
    return filepath


def generate_gantt_from_results(results, output_dir=None, tag=''):
    """从实验结果 JSON 生成甘特图（只画 makespan 最短的解）。
    tag: 用于区分同实例多次运行的文件名后缀（如 'run0', 'run1'）"""
    instance_name = results.get("instance", "Unknown")
    schedules_data = results.get("schedules", [])
    if not schedules_data:
        logger.warning("No schedule data for %s", instance_name)
        return []
    if not output_dir:
        output_dir = os.path.join(GANNT_DIR, instance_name)

    best_entry = min(schedules_data, key=lambda e: e["makespan_crisp"])
    schedule = best_entry["schedule"]
    makespan = best_entry["makespan_crisp"]
    index = best_entry.get("rank", "?")  # archive 中的序号 (1-based)

    # 计算 TFN makespan 三参数 (t₁, t₂, t₃)
    tfn_t1, tfn_t2, tfn_t3 = _compute_tfn_makespan(schedule)

    # 文件名含 run 标签 + TFN 三参数，唯一且便于识别
    prefix = f'gantt_{tag}_' if tag else 'gantt_'
    filepath = os.path.join(output_dir,
        f'{prefix}idx{index}_ms{makespan:.1f}_TFN[t1={tfn_t1:.0f},t2={tfn_t2:.0f},t3={tfn_t3:.0f}].png')
    os.makedirs(output_dir, exist_ok=True)
    _plot_gantt_internal(schedule, instance_name, index, makespan,
                         tfn_t1, tfn_t2, tfn_t3, filepath)
    logger.info("Gantt (ms=%.1f, TFN=[%.0f,%.0f,%.0f]): %s",
                makespan, tfn_t1, tfn_t2, tfn_t3, filepath)
    return [filepath]


def _plot_gantt_internal(schedule, instance_name, index, makespan,
                          tfn_t1, tfn_t2, tfn_t3, filepath):
    """内部甘特图渲染（线程安全：使用显式 Figure 引用保存/关闭）
    包含 TFN 三参数可视化：灰色不确定区间 [t₁,t₃] + 红色虚线标注 t₂"""
    machines = sorted(set(op["machine_label"] for op in schedule),
                      key=lambda m: int(m[1:]))
    n_machines = len(machines)
    machine_idx = {m: i for i, m in enumerate(machines)}
    job_ids = sorted(set(op["job_id"] for op in schedule))
    n_jobs = len(job_ids)
    cmap = get_colormap('tab20', max(n_jobs, 20))
    job_color = {jid: cmap(i % 20) for i, jid in enumerate(job_ids)}
    total_ms = max(op["finish"] for op in schedule)

    fig, ax = plt.subplots(figsize=(max(14, total_ms / 25), max(5, n_machines * 0.55) + 1.2))
    for op in schedule:
        y = n_machines - 1 - machine_idx[op["machine_label"]]
        dur = op["finish"] - op["start"]
        ax.barh(y, dur, height=0.7, left=op["start"],
                color=job_color[op["job_id"]], edgecolor='black',
                linewidth=0.5, alpha=0.85)
    ax.set_yticks(range(n_machines))
    ax.set_yticklabels(reversed(machines))
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Machine', fontsize=12)
    ax.set_title(f'{instance_name.upper()} — Gantt #{index}\n'
                 f'Crisp Makespan = {makespan:.2f}', fontsize=13, fontweight='bold')
    ax.set_xlim(0, total_ms * 1.03)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    style_ax(ax)

    # ══════════════════════════════════════════════════════════════
    # TFN 不确定性区间可视化
    # ══════════════════════════════════════════════════════════════
    # 灰色半透明带：[t₁, t₃] 表示模糊 makespan 的可能范围
    ax.axvspan(tfn_t1, tfn_t3, color='#2166AC', alpha=0.06, zorder=0)
    # 红色虚线：t₂ (最可能值，defuzzified makespan 的核心分量)
    ax.axvline(x=tfn_t2, color='#D6604D', linewidth=1.2, linestyle='--', alpha=0.6, zorder=2)

    # ══════════════════════════════════════════════════════════════
    # 图表下方信息面板：Job 图例 + TFN 参数 + 数据来源
    # ══════════════════════════════════════════════════════════════
    crisp_tfn = (tfn_t1 + 2 * tfn_t2 + tfn_t3) / 4

    # Job 色块图例（水平居中，位于图表下方）
    handles = [plt.Rectangle((0, 0), 1, 1, fc=job_color[jid],
                             edgecolor='black', linewidth=0.5, label=f'J{jid + 1}')
               for jid in job_ids]
    fig.legend(handles=handles, loc='upper center', fontsize=7,
               ncol=min(n_jobs, 10), title='Jobs', title_fontsize=8,
               frameon=True, framealpha=0.8, edgecolor=C_GRAY,
               bbox_to_anchor=(0.5, 0.14))

    # TFN 参数信息行（图例下方，紧凑单行）
    tfn_info = (
        f'TFN Makespan:  '
        f't\u2081 = {tfn_t1:.1f}  (Lower bound)  |  '
        f't\u2082 = {tfn_t2:.1f}  (Most likely)  |  '
        f't\u2083 = {tfn_t3:.1f}  (Upper bound)  |  '
        f'Crisp = {crisp_tfn:.1f}  =  (t\u2081 + 2t\u2082 + t\u2083) / 4  '
        f'[Interval: {tfn_t1:.0f} \u2013 {tfn_t3:.0f}]'
    )
    fig.text(0.5, 0.05, tfn_info, ha='center', va='center', fontsize=7.5,
             fontname='DejaVu Sans Mono',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F4F8',
                       edgecolor=C_GRAY, alpha=0.92))

    source_footer(fig, f'Instance: {instance_name}')
    # 为底部信息面板预留空间
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()