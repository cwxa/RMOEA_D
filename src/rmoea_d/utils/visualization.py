#!/usr/bin/env python3
"""
可视化脚本：RMOEA/D vs MOEA/D 对比图表
- 支持单次运行和多轮独立运行（30次）聚合数据
- 误差棒表示 mean±std
- 【三角模糊数(TFN)】凸显模糊Pareto前沿的不确定性区间(t1,t2,t3)
- 统计检验可视化（Friedman + Wilcoxon）
- 所有图表保存到 charts 文件夹
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

# 图表输出基础目录
CHARTS_BASE = 'charts'
BENCHMARK_DIR = os.path.join(CHARTS_BASE, 'benchmark')
rcParams['legend.fontsize'] = 10

# ──────────── 数据加载工具 ────────────

def load_results(results_dir, instance):
    """加载单个实例的单次/多次实验结果（合并summary和详细结果文件，含模糊数据）"""
    summary_path = os.path.join(results_dir, instance, f'benchmark_summary_{instance}_*.json')
    files = glob.glob(summary_path)
    if not files:
        return None
    latest_file = sorted(files)[-1]
    with open(latest_file, 'r') as f:
        summary = json.load(f)

    # 加载详细结果以获取 final_pf 和 fuzzy_pf
    for algo_key, algo_dir in [('rmoea_d', 'RMOEA_D'), ('moea_d', 'MOEA_D')]:
        pattern = os.path.join(results_dir, instance, algo_dir, f'{instance}_{algo_dir}_*.json')
        detail_files = glob.glob(pattern)
        if detail_files:
            detail = json.load(open(sorted(detail_files)[-1], 'r'))
            if 'final_pf' in detail:
                summary[algo_key]['final_pf'] = detail['final_pf']
            if 'fuzzy_pf' in detail:
                summary[algo_key]['fuzzy_pf'] = detail['fuzzy_pf']

    return summary


def load_aggregate_results(results_dir="results"):
    """加载多轮独立运行聚合结果（benchmark_aggregate_*.json）
    同时加载每实例的详细数据（final_pf, fuzzy_pf, history）用于Pareto/收敛图
    返回: (instances_dict, statistical_tests_dict)
    """
    pattern = os.path.join(results_dir, "benchmark_aggregate_*.json")
    files = glob.glob(pattern)
    if not files:
        return None, None
    latest = sorted(files)[-1]
    with open(latest, 'r') as f:
        data = json.load(f)

    instances_data = data.get("instances", {})

    # 补充加载每实例的详细数据（final_pf, fuzzy_pf, history）
    for instance in instances_data:
        for algo_key, algo_dir in [('rmoea_d', 'RMOEA_D'), ('moea_d', 'MOEA_D')]:
            # 加载任意一个run的详细结果（含final_pf, fuzzy_pf, history）
            detail_dir = os.path.join(results_dir, instance, algo_dir)
            if not os.path.isdir(detail_dir):
                continue
            # 尝试 run0，若不存在则取任意可用 json
            detail_pattern = os.path.join(detail_dir, f'{instance}_{algo_dir}_Np*_run0_*.json')
            detail_files = glob.glob(detail_pattern)
            if not detail_files:
                # 回退：取任意 run 文件
                detail_pattern = os.path.join(detail_dir, f'{instance}_{algo_dir}_Np*_*.json')
                detail_files = glob.glob(detail_pattern)
            if not detail_files:
                continue
            try:
                with open(sorted(detail_files)[-1], 'r') as f:
                    detail = json.load(f)
                # 将详细数据附加到聚合结果中
                for field in ['final_pf', 'fuzzy_pf', 'history']:
                    if field in detail and detail[field]:
                        instances_data[instance][algo_key][field] = detail[field]
            except Exception:
                pass

    return instances_data, data.get("statistical_tests", {})


def load_history(instance, algorithm):
    """加载算法的历史收敛记录"""
    pattern = os.path.join('results', instance, algorithm, f'{instance}_{algorithm}_*.json')
    files = glob.glob(pattern)
    if not files:
        return None
    with open(sorted(files)[-1], 'r') as f:
        return json.load(f)


def load_all_results():
    """加载所有实例的单次实验结果"""
    all_results = {}
    for instance in ['mk01', 'mk02', 'mk03', 'mk04', 'mk05',
                     'mk06', 'mk07', 'mk08', 'mk09', 'mk10']:
        result = load_results('results', instance)
        if result:
            all_results[instance] = result
    return all_results


def _get_val(data, key, default=0):
    """安全获取值，支持dict(含t1/t2/t3)和标量"""
    v = data.get(key, default)
    if isinstance(v, dict):
        return v.get('t2', v.get('best_clear', default))
    return v if v else default


# ──────────── 图表1: 传统Pareto前沿对比 ────────────

def plot_pareto_front_per_instance(all_results):
    """为每个实例绘制Pareto前沿对比图"""
    for instance in sorted(all_results.keys()):
        results = all_results[instance]
        fig, ax = plt.subplots()

        for algo_key, algo_label, color, marker in [
            ('rmoea_d', 'RMOEA/D', '#1f77b4', 'o'),
            ('moea_d', 'MOEA/D', '#ff7f0e', '^')
        ]:
            pf = results[algo_key].get('final_pf', [])
            if not pf:
                continue
            if isinstance(pf[0], dict):
                ms = [p.get('Makespan', 0) for p in pf]
                wl = [p.get('Workload', 0) for p in pf]
            else:
                ms = [p[0] for p in pf]
                wl = [p[1] for p in pf]
            ms = [x for x in ms if x > 0]
            wl = [x for x in wl if x > 0]
            if not ms:
                continue
            ax.scatter(ms, wl, s=80, c=color, marker=marker,
                       edgecolor='k', linewidth=1.5, alpha=0.85, label=algo_label)

        ax.set_xlabel('Makespan (Crisp)')
        ax.set_ylabel('Total Workload')
        ax.set_title(f'{instance.upper()} - Pareto Front Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        instance_dir = os.path.join(BENCHMARK_DIR, instance)
        os.makedirs(instance_dir, exist_ok=True)
        plt.savefig(os.path.join(instance_dir, 'pareto_front.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.debug("  [Pareto] %s/pareto_front.png", instance_dir)


# ──────────── 图表2: 三角模糊数Pareto前沿 (TFN Pareto Front) ────────────

def plot_fuzzy_pareto_front(all_results):
    """
    凸显三角模糊数的Pareto前沿对比图
    - 中心点为清晰值 (t1+2*t2+t3)/4
    - 误差棒/椭圆表示模糊不确定性区间 [t1, t3]
    - 子图: 同时展示Makespan和Workload的TFN分布
    """
    instances = sorted(all_results.keys())
    for instance in instances:
        results = all_results[instance]
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # 左图: 带模糊区间的Pareto前沿
        ax = axes[0]
        for algo_key, algo_label, color, marker in [
            ('rmoea_d', 'RMOEA/D', '#1f77b4', 'o'),
            ('moea_d', 'MOEA/D', '#ff7f0e', '^')
        ]:
            fuzzy_pf = results[algo_key].get('fuzzy_pf', [])
            if not fuzzy_pf:
                continue
            ms_t1 = [p['Makespan']['t1'] for p in fuzzy_pf]
            ms_t2 = [p['Makespan']['t2'] for p in fuzzy_pf]
            ms_t3 = [p['Makespan']['t3'] for p in fuzzy_pf]
            wl_t1 = [p['Workload']['t1'] for p in fuzzy_pf]
            wl_t2 = [p['Workload']['t2'] for p in fuzzy_pf]
            wl_t3 = [p['Workload']['t3'] for p in fuzzy_pf]

            # 误差棒: 下界 t2 - t1, 上界 t3 - t2
            x_err_low = [m2 - m1 for m1, m2 in zip(ms_t1, ms_t2)]
            x_err_high = [m3 - m2 for m2, m3 in zip(ms_t2, ms_t3)]
            y_err_low = [w2 - w1 for w1, w2 in zip(wl_t1, wl_t2)]
            y_err_high = [w3 - w2 for w2, w3 in zip(wl_t2, wl_t3)]

            ax.errorbar(ms_t2, wl_t2,
                        xerr=[x_err_low, x_err_high],
                        yerr=[y_err_low, y_err_high],
                        fmt=marker, color=color, alpha=0.7,
                        capsize=3, capthick=0.5, elinewidth=0.8,
                        markersize=8, markeredgecolor='k',
                        markeredgewidth=0.5, label=algo_label)

        ax.set_xlabel('Makespan (t2, with t1~t3 uncertainty)')
        ax.set_ylabel('Total Workload (t2, with t1~t3 uncertainty)')
        ax.set_title(f'{instance.upper()} - TFN Pareto Front\n'
                     '(t1,t2,t3) uncertainty intervals via error bars')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # 右图: TFN分布对比 - 选最佳Makespan解展示其三角模糊数
        ax2 = axes[1]
        y_positions = []
        y_labels = []
        colors_tfn = []
        bar_data = []

        for idx, (algo_key, algo_label, color) in enumerate([
            ('rmoea_d', 'RMOEA/D', '#1f77b4'),
            ('moea_d', 'MOEA/D', '#ff7f0e')
        ]):
            fuzzy_pf = results[algo_key].get('fuzzy_pf', [])
            if not fuzzy_pf:
                continue
            # 找makespan t2最小的解
            best_idx = min(range(len(fuzzy_pf)),
                           key=lambda i: fuzzy_pf[i]['Makespan']['t2'])
            best = fuzzy_pf[best_idx]
            ms = best['Makespan']
            wl = best['Workload']

            y_positions.extend([idx * 3 + 0, idx * 3 + 1])
            y_labels.extend([
                f'{algo_label}\nMakespan',
                f'{algo_label}\nWorkload'
            ])
            colors_tfn.extend([color, color])
            bar_data.extend([
                (ms['t1'], ms['t2'], ms['t3']),
                (wl['t1'], wl['t2'], wl['t3'])
            ])

        # 绘制TFN: 左侧区间(t2-t1), 中心t2, 右侧区间(t3-t2)
        for i, (y, (t1, t2, t3), color, label) in enumerate(
                zip(y_positions, bar_data, colors_tfn, y_labels)):
            # 左侧填充(t1到t2)
            ax2.barh(y, t2 - t1, height=0.6, left=t1, color=color, alpha=0.35)
            # 右侧填充(t2到t3)
            ax2.barh(y, t3 - t2, height=0.6, left=t2, color=color, alpha=0.35)
            # 中心线
            ax2.barh(y, 0.01, height=0.6, left=t2, color=color, alpha=1.0)
            # 标注t1, t2, t3
            ax2.text(t1, y + 0.32, f'{t1:.0f}', fontsize=7, ha='center', color='gray')
            ax2.text(t2, y - 0.32, f'{t2:.0f}', fontsize=8, ha='center',
                     fontweight='bold', color=color)
            ax2.text(t3, y + 0.32, f'{t3:.0f}', fontsize=7, ha='center', color='gray')

        ax2.set_yticks(y_positions)
        ax2.set_yticklabels(y_labels, fontsize=8)
        ax2.set_xlabel('Triangular Fuzzy Number Value (t1, t2, t3)')
        ax2.set_title(f'{instance.upper()} - Best Solution TFN Distribution\n'
                      'shaded=t1~t3 uncertainty, bold=t2 most-likely')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_axisbelow(True)

        plt.tight_layout()
        instance_dir = os.path.join(BENCHMARK_DIR, instance)
        os.makedirs(instance_dir, exist_ok=True)
        plt.savefig(os.path.join(instance_dir, 'fuzzy_pareto.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.debug("  [Fuzzy Pareto] %s/fuzzy_pareto.png", instance_dir)


# ──────────── 图表3: 模糊Makespan对比 (t1,t2,t3 三柱对比 + 误差棒) ────────────

def plot_fuzzy_makespan_comparison(agg_results):
    """
    对所有实例绘制三角模糊数Makespan的三维度对比 (t1, t2, t3)
    支持多轮聚合数据的mean±std误差棒
    """
    instances = sorted(agg_results.keys())
    x = np.arange(len(instances))
    width = 0.12

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {'rmoea_d': '#1f77b4', 'moea_d': '#ff7f0e'}

    # 检测是否为聚合数据（有mean/std字段）
    is_agg = isinstance(
        agg_results.get(instances[0], {}).get('rmoea_d', {}), dict
    ) and 'hv_mean' in agg_results[instances[0]].get('rmoea_d', {})

    for algo_offset, (algo_key, algo_label) in enumerate([
        ('rmoea_d', 'RMOEA/D'), ('moea_d', 'MOEA/D')
    ]):
        for t_offset, (t_key, t_label) in enumerate([
            ('t1', 't1 (Earliest)'), ('t2', 't2 (Most Likely)'), ('t3', 't3 (Latest)')
        ]):
            values = []
            errors = []
            for inst in instances:
                r = agg_results[inst][algo_key]
                if is_agg:
                    # 聚合数据: fuzzy_makespan_t1_mean 等
                    mean_key = f'fuzzy_makespan_{t_key}_mean'
                    v = r.get(mean_key, 0)
                    values.append(v if v else 0)
                    errors.append(0)
                else:
                    fm = r.get('fuzzy_makespan', {})
                    if fm and 'best' in fm:
                        values.append(fm['best'][t_key])
                    else:
                        values.append(0)
                    errors.append(0)

            pos = x + algo_offset * width * 3 + t_offset * width - width * 3
            bars = ax.bar(pos, values, width,
                          color=colors[algo_key],
                          alpha=0.5 + 0.25 * (2 - t_offset),  # t2最深
                          edgecolor=colors[algo_key],
                          linewidth=0.8,
                          label=f'{algo_label} {t_label}')

    ax.set_xlabel('Instance')
    ax.set_ylabel('Makespan (Triangular Fuzzy Number)')
    ax.set_title('Fuzzy Makespan Comparison: t1(Earliest)/t2(Most Likely)/t3(Latest)\n'
                 'Triangular Fuzzy Numbers reveal processing time uncertainty')
    ax.set_xticks(x)
    ax.set_xticklabels([inst.upper() for inst in instances], rotation=45, ha='right')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:6], labels[:6], ncol=2, fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_DIR, 'fuzzy_makespan_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.debug("  [Fuzzy Makespan] %s/fuzzy_makespan_comparison.png", BENCHMARK_DIR)


# ──────────── 图表4: 模糊Workload对比 (t1,t2,t3 三柱对比) ────────────

def plot_fuzzy_workload_comparison(agg_results):
    """
    对所有实例绘制三角模糊数Workload的三维度对比 (t1, t2, t3)
    """
    instances = sorted(agg_results.keys())
    x = np.arange(len(instances))
    width = 0.12

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {'rmoea_d': '#1f77b4', 'moea_d': '#ff7f0e'}

    is_agg = isinstance(
        agg_results.get(instances[0], {}).get('rmoea_d', {}), dict
    ) and 'hv_mean' in agg_results[instances[0]].get('rmoea_d', {})

    for algo_offset, (algo_key, algo_label) in enumerate([
        ('rmoea_d', 'RMOEA/D'), ('moea_d', 'MOEA/D')
    ]):
        for t_offset, (t_key, t_label) in enumerate([
            ('t1', 't1 (Earliest)'), ('t2', 't2 (Most Likely)'), ('t3', 't3 (Latest)')
        ]):
            values = []
            for inst in instances:
                r = agg_results[inst][algo_key]
                if is_agg:
                    mean_key = f'fuzzy_workload_{t_key}_mean'
                    v = r.get(mean_key, 0)
                    values.append(v if v else 0)
                else:
                    fw = r.get('fuzzy_workload', {})
                    if fw and 'best' in fw:
                        values.append(fw['best'][t_key])
                    else:
                        values.append(0)
            pos = x + algo_offset * width * 3 + t_offset * width - width * 3
            ax.bar(pos, values, width,
                   color=colors[algo_key],
                   alpha=0.5 + 0.25 * (2 - t_offset),
                   edgecolor=colors[algo_key],
                   linewidth=0.8,
                   label=f'{algo_label} {t_label}')

    ax.set_xlabel('Instance')
    ax.set_ylabel('Workload (Triangular Fuzzy Number)')
    ax.set_title('Fuzzy Workload Comparison: t1(Earliest)/t2(Most Likely)/t3(Latest)\n'
                 'Triangular Fuzzy Numbers reveal total machine load uncertainty')
    ax.set_xticks(x)
    ax.set_xticklabels([inst.upper() for inst in instances], rotation=45, ha='right')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:6], labels[:6], ncol=2, fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_DIR, 'fuzzy_workload_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.debug("  [Fuzzy Workload] %s/fuzzy_workload_comparison.png", BENCHMARK_DIR)


# ──────────── 图表5: 每个实例的TFN区间对比 ────────────

def plot_fuzzy_range_per_instance(agg_results):
    """
    每个实例单独绘制：用箱形风格展示模糊Makespan的不确定性区间 [t1, t3]
    主柱=t2（最可能值），须=t1和t3区间
    """
    instances = sorted(agg_results.keys())
    is_agg = isinstance(
        agg_results.get(instances[0], {}).get('rmoea_d', {}), dict
    ) and 'hv_mean' in agg_results[instances[0]].get('rmoea_d', {})

    for instance in instances:
        r = agg_results[instance]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, metric_key, metric_label in [
            (axes[0], 'fuzzy_makespan', 'Makespan'),
            (axes[1], 'fuzzy_workload', 'Workload')
        ]:
            for i, (algo_key, algo_label, color) in enumerate([
                ('rmoea_d', 'RMOEA/D', '#1f77b4'),
                ('moea_d', 'MOEA/D', '#ff7f0e')
            ]):
                if is_agg:
                    # 聚合数据：提取mean值
                    prefix = metric_key  # fuzzy_makespan or fuzzy_workload
                    t1_mean = r[algo_key].get(f'{prefix}_t1_mean', 0)
                    t2_mean = r[algo_key].get(f'{prefix}_t2_mean', 0)
                    t3_mean = r[algo_key].get(f'{prefix}_t3_mean', 0)
                    if t2_mean == 0:
                        continue
                    # 绘制单个TFN条
                    x_pos = i * 4 + 1.5
                    ax.bar(x_pos, t3_mean - t1_mean, 0.6, bottom=t1_mean,
                           color=color, alpha=0.25, edgecolor=color, linewidth=0.5)
                    ax.bar(x_pos, 0.01, 0.6, bottom=t2_mean,
                           color=color, alpha=1.0)
                    ax.plot([x_pos, x_pos], [t1_mean, t3_mean],
                            color=color, linewidth=2, alpha=0.8)
                    ax.text(x_pos, t3_mean + (t3_mean - t1_mean) * 0.03,
                            f'({t1_mean:.0f},{t2_mean:.0f},{t3_mean:.0f})',
                            ha='center', fontsize=8, color='darkblue' if color == '#1f77b4' else 'darkorange',
                            fontweight='bold', rotation=90)
                else:
                    fm = r[algo_key].get(metric_key, {})
                    if not fm or 'best' not in fm:
                        continue
                    best = fm['best']
                    # 单次数据
                    x_pos = i * 4 + 1.5
                    t1, t2, t3 = best['t1'], best['t2'], best['t3']
                    ax.bar(x_pos, t3 - t1, 0.6, bottom=t1,
                           color=color, alpha=0.25, edgecolor=color, linewidth=0.5)
                    ax.bar(x_pos, 0.01, 0.6, bottom=t2,
                           color=color, alpha=1.0)
                    ax.plot([x_pos, x_pos], [t1, t3],
                            color=color, linewidth=2, alpha=0.8)
                    ax.text(x_pos, t3 + (t3 - t1) * 0.03,
                            f'({t1:.0f},{t2:.0f},{t3:.0f})',
                            ha='center', fontsize=8, color='darkblue' if color == '#1f77b4' else 'darkorange',
                            fontweight='bold', rotation=90)

            ax.set_xticks([1.5, 5.5])
            ax.set_xticklabels(['RMOEA/D', 'MOEA/D'])
            ax.set_ylabel(f'{metric_label} (Triangular Fuzzy Number)')
            ax.set_title(f'{instance.upper()} - Fuzzy {metric_label} Distribution\n'
                         f'Bar=t1~t3 range, bold line=t2 most-likely')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)

        plt.tight_layout()
        instance_dir = os.path.join(BENCHMARK_DIR, instance)
        os.makedirs(instance_dir, exist_ok=True)
        plt.savefig(os.path.join(instance_dir, 'fuzzy_range.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.debug("  [Fuzzy Range] %s/fuzzy_range.png", instance_dir)


# ──────────── 图表6: 收敛曲线 ────────────

def plot_convergence_curve_per_instance(all_results):
    """每个实例的HV收敛曲线"""
    for instance in sorted(all_results.keys()):
        fig, ax = plt.subplots()
        for algo_dir, algo_label, color, ls in [
            ('RMOEA_D', 'RMOEA/D', '#1f77b4', '-'),
            ('MOEA_D', 'MOEA/D', '#ff7f0e', '--')
        ]:
            hist = load_history(instance, algo_dir)
            if hist and 'history' in hist:
                gens = [h['gen'] for h in hist['history']]
                hvs = [h.get('hv', 0) for h in hist['history']]
                ax.plot(gens, hvs, c=color, linewidth=2.5, linestyle=ls, label=algo_label)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Hypervolume (HV)')
        ax.set_title(f'{instance.upper()} - Convergence Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        instance_dir = os.path.join(BENCHMARK_DIR, instance)
        os.makedirs(instance_dir, exist_ok=True)
        plt.savefig(os.path.join(instance_dir, 'convergence.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.debug("  [Convergence] %s/convergence.png", instance_dir)


# ──────────── 图表7: HV对比（支持误差棒） ────────────

def plot_hv_comparison(agg_results):
    """HV总对比柱状图（支持多轮聚合的mean±std）"""
    instances = sorted(agg_results.keys())
    x = np.arange(len(instances))
    width = 0.35

    is_agg = isinstance(
        agg_results.get(instances[0], {}).get('rmoea_d', {}), dict
    ) and 'hv_mean' in agg_results[instances[0]].get('rmoea_d', {})

    if is_agg:
        rmoea_hv = [agg_results[i]['rmoea_d']['hv_mean'] for i in instances]
        rmoea_err = [agg_results[i]['rmoea_d']['hv_std'] for i in instances]
        moea_hv = [agg_results[i]['moea_d']['hv_mean'] for i in instances]
        moea_err = [agg_results[i]['moea_d']['hv_std'] for i in instances]
    else:
        rmoea_hv = [agg_results[i]['rmoea_d'].get('final_hv', 0) for i in instances]
        rmoea_err = [0] * len(instances)
        moea_hv = [agg_results[i]['moea_d'].get('final_hv', 0) for i in instances]
        moea_err = [0] * len(instances)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, rmoea_hv, width, yerr=rmoea_err,
           label='RMOEA/D', color='#1f77b4', alpha=0.85,
           capsize=4, error_kw={'elinewidth': 1.5})
    ax.bar(x + width / 2, moea_hv, width, yerr=moea_err,
           label='MOEA/D', color='#ff7f0e', alpha=0.85,
           capsize=4, error_kw={'elinewidth': 1.5})
    for i, (vr, vm) in enumerate(zip(rmoea_hv, moea_hv)):
        imp = ((vr - vm) / vm * 100) if vm > 0 else 0
        clr = 'green' if imp > 0 else 'red'
        ax.annotate(f'{imp:+.1f}%', (x[i], max(vr, vm)),
                    textcoords="offset points", xytext=(0, 8),
                    ha='center', fontsize=8, color=clr, fontweight='bold')
    ax.set_xlabel('Instance')
    ax.set_ylabel('Hypervolume')
    n_runs = agg_results[instances[0]].get('n_runs', 1)
    n_label = f' (N={n_runs}, mean±std)' if is_agg else ''
    ax.set_title(f'HV Comparison (RMOEA/D vs MOEA/D){n_label}')
    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in instances], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_DIR, 'hv_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.debug("  [HV] %s/hv_comparison.png", BENCHMARK_DIR)


# ──────────── 图表8: Makespan对比（清晰值 + 误差棒） ────────────

def plot_makespan_comparison(agg_results):
    """Makespan总对比（清晰值，支持mean±std）"""
    instances = sorted(agg_results.keys())
    x = np.arange(len(instances))
    width = 0.35

    is_agg = isinstance(
        agg_results.get(instances[0], {}).get('rmoea_d', {}), dict
    ) and 'best_makespan_mean' in agg_results[instances[0]].get('rmoea_d', {})

    rmoea_ms, moea_ms = [], []
    rmoea_err, moea_err = [], []
    for inst in instances:
        r = agg_results[inst]
        if is_agg:
            rmoea_ms.append(r['rmoea_d'].get('best_makespan_mean', 0))
            rmoea_err.append(r['rmoea_d'].get('best_makespan_std', 0))
            moea_ms.append(r['moea_d'].get('best_makespan_mean', 0))
            moea_err.append(r['moea_d'].get('best_makespan_std', 0))
        else:
            vr = r['rmoea_d'].get('best_makespan', 0)
            vm = r['moea_d'].get('best_makespan', 0)
            rmoea_ms.append(_get_val(r['rmoea_d'], 'best_makespan'))
            moea_ms.append(_get_val(r['moea_d'], 'best_makespan'))
            rmoea_err.append(0)
            moea_err.append(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, rmoea_ms, width, yerr=rmoea_err,
           label='RMOEA/D', color='#1f77b4', alpha=0.85,
           capsize=4, error_kw={'elinewidth': 1.5})
    ax.bar(x + width / 2, moea_ms, width, yerr=moea_err,
           label='MOEA/D', color='#ff7f0e', alpha=0.85,
           capsize=4, error_kw={'elinewidth': 1.5})
    for i, (vr, vm) in enumerate(zip(rmoea_ms, moea_ms)):
        if vm > 0:
            imp = ((vm - vr) / vm * 100)
            clr = 'green' if imp > 0 else 'red'
            ax.annotate(f'{imp:+.1f}%', (x[i], max(vr, vm)),
                        textcoords="offset points", xytext=(0, 5),
                        ha='center', fontsize=8, color=clr)
    ax.set_xlabel('Instance')
    ax.set_ylabel('Best Makespan (Crisp)')
    n_label = f' (mean±std)' if is_agg else ''
    ax.set_title(f'Makespan Comparison{n_label}')
    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in instances], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_DIR, 'makespan_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.debug("  [Makespan] %s/makespan_comparison.png", BENCHMARK_DIR)


# ──────────── 图表9: Workload对比 ────────────

def plot_workload_comparison(agg_results):
    """Workload总对比（支持mean±std）"""
    instances = sorted(agg_results.keys())
    x = np.arange(len(instances))
    width = 0.35

    is_agg = isinstance(
        agg_results.get(instances[0], {}).get('rmoea_d', {}), dict
    ) and 'best_workload_mean' in agg_results[instances[0]].get('rmoea_d', {})

    rmoea_wl, moea_wl = [], []
    rmoea_err, moea_err = [], []
    for inst in instances:
        r = agg_results[inst]
        if is_agg:
            rmoea_wl.append(r['rmoea_d'].get('best_workload_mean', 0))
            rmoea_err.append(r['rmoea_d'].get('best_workload_std', 0))
            moea_wl.append(r['moea_d'].get('best_workload_mean', 0))
            moea_err.append(r['moea_d'].get('best_workload_std', 0))
        else:
            rmoea_wl.append(_get_val(r['rmoea_d'], 'best_workload'))
            moea_wl.append(_get_val(r['moea_d'], 'best_workload'))
            rmoea_err.append(0)
            moea_err.append(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, rmoea_wl, width, yerr=rmoea_err,
           label='RMOEA/D', color='#1f77b4', alpha=0.85,
           capsize=4, error_kw={'elinewidth': 1.5})
    ax.bar(x + width / 2, moea_wl, width, yerr=moea_err,
           label='MOEA/D', color='#ff7f0e', alpha=0.85,
           capsize=4, error_kw={'elinewidth': 1.5})
    for i, (vr, vm) in enumerate(zip(rmoea_wl, moea_wl)):
        if vm > 0:
            imp = ((vm - vr) / vm * 100)
            clr = 'green' if imp > 0 else 'red'
            ax.annotate(f'{imp:+.1f}%', (x[i], max(vr, vm)),
                        textcoords="offset points", xytext=(0, 5),
                        ha='center', fontsize=8, color=clr)
    ax.set_xlabel('Instance')
    ax.set_ylabel('Best Workload')
    n_label = f' (mean±std)' if is_agg else ''
    ax.set_title(f'Workload Comparison{n_label}')
    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in instances], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_DIR, 'workload_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.debug("  [Workload] %s/workload_comparison.png", BENCHMARK_DIR)


# ──────────── 图表10: 改进幅度汇总 ────────────

def plot_improvement_summary(agg_results):
    """改进幅度汇总（支持mean数据）"""
    instances = sorted(agg_results.keys())
    x = np.arange(len(instances))
    width = 0.25

    is_agg = isinstance(
        agg_results.get(instances[0], {}).get('rmoea_d', {}), dict
    ) and 'hv_mean' in agg_results[instances[0]].get('rmoea_d', {})

    hv_imp, ms_imp, wl_imp = [], [], []
    for inst in instances:
        r = agg_results[inst]
        if is_agg:
            hr = r['rmoea_d']['hv_mean']
            hm = r['moea_d']['hv_mean']
            mr = r['rmoea_d'].get('best_makespan_mean', 0)
            mm = r['moea_d'].get('best_makespan_mean', 0)
            wr = r['rmoea_d'].get('best_workload_mean', 0)
            wm = r['moea_d'].get('best_workload_mean', 0)
        else:
            hr = r['rmoea_d'].get('final_hv', 0)
            hm = r['moea_d'].get('final_hv', 0)
            mr = _get_val(r['rmoea_d'], 'best_makespan')
            mm = _get_val(r['moea_d'], 'best_makespan')
            wr = _get_val(r['rmoea_d'], 'best_workload')
            wm = _get_val(r['moea_d'], 'best_workload')

        hv_imp.append(((hr - hm) / hm * 100) if hm > 0 else 0)
        ms_imp.append(((mm - mr) / mm * 100) if mm > 0 else 0)
        wl_imp.append(((wm - wr) / wm * 100) if wm > 0 else 0)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width, hv_imp, width, label='HV Improvement (%)', color='#1f77b4', alpha=0.85)
    ax.bar(x, ms_imp, width, label='Makespan Improvement (%)', color='#2ca02c', alpha=0.85)
    ax.bar(x + width, wl_imp, width, label='Workload Improvement (%)', color='#9467bd', alpha=0.85)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Instance')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Performance Improvement Summary (RMOEA/D vs MOEA/D)')
    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in instances], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_DIR, 'improvement_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.debug("  [Improvement] %s/improvement_summary.png", BENCHMARK_DIR)


# ──────────── 图表11: PF大小对比 ────────────

def plot_pf_size_comparison(agg_results):
    """PF大小对比（支持mean±std）"""
    instances = sorted(agg_results.keys())
    x = np.arange(len(instances))
    width = 0.35

    is_agg = isinstance(
        agg_results.get(instances[0], {}).get('rmoea_d', {}), dict
    ) and 'pf_size_mean' in agg_results[instances[0]].get('rmoea_d', {})

    if is_agg:
        rmoea_ps = [agg_results[i]['rmoea_d']['pf_size_mean'] for i in instances]
        rmoea_err = [agg_results[i]['rmoea_d']['pf_size_std'] for i in instances]
        moea_ps = [agg_results[i]['moea_d']['pf_size_mean'] for i in instances]
        moea_err = [agg_results[i]['moea_d']['pf_size_std'] for i in instances]
    else:
        rmoea_ps = [agg_results[i]['rmoea_d'].get('pf_size', 0) for i in instances]
        rmoea_err = [0] * len(instances)
        moea_ps = [agg_results[i]['moea_d'].get('pf_size', 0) for i in instances]
        moea_err = [0] * len(instances)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, rmoea_ps, width, yerr=rmoea_err,
           label='RMOEA/D', color='#1f77b4', alpha=0.85,
           capsize=4, error_kw={'elinewidth': 1.5})
    ax.bar(x + width / 2, moea_ps, width, yerr=moea_err,
           label='MOEA/D', color='#ff7f0e', alpha=0.85,
           capsize=4, error_kw={'elinewidth': 1.5})
    ax.set_xlabel('Instance')
    ax.set_ylabel('PF Size')
    n_label = f' (mean±std)' if is_agg else ''
    ax.set_title(f'Pareto Front Size Comparison{n_label}')
    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in instances], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_DIR, 'pf_size_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.debug("  [PF Size] %s/pf_size_comparison.png", BENCHMARK_DIR)


# ──────────── 图表12: 统计检验可视化 ────────────

def plot_statistical_tests(agg_results, stats_tests):
    """绘制统计检验结果图表"""
    if not stats_tests or 'wilcoxon_per_instance' not in stats_tests:
        logger.warning("No statistical test data available, skipping stats chart.")
        return

    instances = sorted(agg_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图: 每个实例的HV boxplot风格（mean±std）
    ax = axes[0]
    x = np.arange(len(instances))
    width = 0.3
    for i, inst in enumerate(instances):
        r = agg_results[inst]
        hv_r_mean = r['rmoea_d'].get('hv_mean', 0)
        hv_r_std = r['rmoea_d'].get('hv_std', 0)
        hv_m_mean = r['moea_d'].get('hv_mean', 0)
        hv_m_std = r['moea_d'].get('hv_std', 0)

        ax.bar(i - width / 2, hv_r_mean, width, yerr=hv_r_std,
               color='#1f77b4', alpha=0.8, capsize=3, label='RMOEA/D' if i == 0 else '')
        ax.bar(i + width / 2, hv_m_mean, width, yerr=hv_m_std,
               color='#ff7f0e', alpha=0.8, capsize=3, label='MOEA/D' if i == 0 else '')

        # Wilcoxon显著性标记
        w = stats_tests.get('wilcoxon_per_instance', {}).get(inst, {})
        if w.get('sig') == 'yes':
            ymax = max(hv_r_mean + hv_r_std, hv_m_mean + hv_m_std)
            ax.annotate('***', (i, ymax * 1.02), ha='center', fontsize=14,
                        color='red', fontweight='bold')
            ax.annotate(f"p={w.get('p_value', 0):.4f}", (i, ymax * 1.06),
                        ha='center', fontsize=7, color='darkred')

    ax.set_xlabel('Instance')
    ax.set_ylabel('Hypervolume (mean ± std)')
    ax.set_title('HV Comparison with Wilcoxon Significance\n(*** p<0.05, paired Wilcoxon test)')
    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in instances])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # 右图: 统计检验结果表格
    ax2 = axes[1]
    ax2.axis('off')

    ft = stats_tests.get('friedman', {})
    wt = stats_tests.get('wilcoxon_overall', {})

    table_data = [
        ['Test', 'Statistic', 'p-value', 'Significant (α=0.05)'],
        ['Friedman', f"{ft.get('statistic', 'N/A'):.4f}" if ft.get('statistic') else 'N/A',
         f"{ft.get('p_value', 'N/A'):.6f}" if ft.get('p_value') else 'N/A',
         ft.get('significant', 'N/A').upper()],
        ['Wilcoxon (Overall)', f"{wt.get('statistic', 'N/A'):.2f}" if wt.get('statistic') else 'N/A',
         f"{wt.get('p_value', 'N/A'):.6f}" if wt.get('p_value') else 'N/A',
         wt.get('significant', 'N/A').upper()],
    ]

    table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.25, 0.2, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    # 表头样式
    for j in range(4):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax2.set_title('Statistical Test Results\n(Friedman & Wilcoxon Signed-Rank)', fontsize=13, pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_DIR, 'statistical_tests.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.debug("  [Stats] %s/statistical_tests.png", BENCHMARK_DIR)


# ──────────── 图表13: 运行时间对比 ────────────

def plot_runtime_comparison(agg_results):
    """运行时间对比（支持mean±std）"""
    instances = sorted(agg_results.keys())
    x = np.arange(len(instances))
    width = 0.35

    is_agg = isinstance(
        agg_results.get(instances[0], {}).get('rmoea_d', {}), dict
    ) and 'time_mean' in agg_results[instances[0]].get('rmoea_d', {})

    if is_agg:
        rmoea_t = [agg_results[i]['rmoea_d']['time_mean'] for i in instances]
        rmoea_err = [agg_results[i]['rmoea_d']['time_std'] for i in instances]
        moea_t = [agg_results[i]['moea_d']['time_mean'] for i in instances]
        moea_err = [agg_results[i]['moea_d']['time_std'] for i in instances]
    else:
        rmoea_t = [agg_results[i]['rmoea_d'].get('total_time', 0) for i in instances]
        rmoea_err = [0] * len(instances)
        moea_t = [agg_results[i]['moea_d'].get('total_time', 0) for i in instances]
        moea_err = [0] * len(instances)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, rmoea_t, width, yerr=rmoea_err,
           label='RMOEA/D', color='#1f77b4', alpha=0.85,
           capsize=4, error_kw={'elinewidth': 1.5})
    ax.bar(x + width / 2, moea_t, width, yerr=moea_err,
           label='MOEA/D', color='#ff7f0e', alpha=0.85,
           capsize=4, error_kw={'elinewidth': 1.5})
    ax.set_xlabel('Instance')
    ax.set_ylabel('Runtime (seconds)')
    n_label = f' (mean±std)' if is_agg else ''
    ax.set_title(f'Runtime Comparison{n_label}')
    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in instances], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_DIR, 'runtime_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.debug("  [Runtime] %s/runtime_comparison.png", BENCHMARK_DIR)


# ──────────── 图表14: 模糊清晰值对比 ────────────

def plot_fuzzy_clear_value_comparison(agg_results):
    """三角模糊数清晰值(crisp value)对比: (t1+2*t2+t3)/4"""
    instances = sorted(agg_results.keys())
    x = np.arange(len(instances))
    width = 0.2

    is_agg = isinstance(
        agg_results.get(instances[0], {}).get('rmoea_d', {}), dict
    ) and 'hv_mean' in agg_results[instances[0]].get('rmoea_d', {})

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, (metric, ylabel) in enumerate([
        ('fuzzy_makespan_clear', 'Makespan Crisp Value'),
        ('fuzzy_workload_clear', 'Workload Crisp Value')
    ]):
        ax = axes[ax_idx]
        # RMOEA/D: t1, t2, t3 bars
        for algo_offset, (algo_key, algo_label, color) in enumerate([
            ('rmoea_d', 'RMOEA/D', '#1f77b4'),
            ('moea_d', 'MOEA/D', '#ff7f0e')
        ]):
            vals = []
            for inst in instances:
                r = agg_results[inst][algo_key]
                if is_agg:
                    # 聚合数据: fuzzy_makespan_clear_mean, fuzzy_workload_clear_mean
                    clear_key = f'fuzzy_{"makespan" if "makespan" in metric else "workload"}_clear_mean'
                    v = r.get(clear_key, 0)
                else:
                    v = r.get(metric.replace('fuzzy_', '').replace('_clear', ''), 0)
                    if isinstance(v, dict):
                        v = v.get('best_clear', v.get('t2', 0))
                vals.append(v if v else 0)
            pos = x + algo_offset * width - width / 2
            ax.bar(pos, vals, width, color=color, alpha=0.85,
                   label=algo_label, edgecolor='k', linewidth=0.5)

        ax.set_xlabel('Instance')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} - Crisp Value Comparison\n(clear value = (t1+2*t2+t3)/4)')
        ax.set_xticks(x)
        ax.set_xticklabels([i.upper() for i in instances], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_DIR, 'fuzzy_clear_value_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.debug("  [Fuzzy Clear] %s/fuzzy_clear_value_comparison.png", BENCHMARK_DIR)


# ──────────── 主入口 ────────────

def main():
    """主入口: 自动检测数据格式并生成所有图表"""
    os.makedirs(BENCHMARK_DIR, exist_ok=True)

    # 1. 尝试加载多轮聚合数据
    agg_data, stats_tests = load_aggregate_results()
    is_aggregate = agg_data is not None and len(agg_data) > 0

    if is_aggregate:
        all_results = agg_data
        n_runs_val = list(all_results.values())[0].get('n_runs', 1)
        logger.info("Loaded aggregate results: %d instances (%d runs each)",
                     len(all_results), n_runs_val)
        logger.debug("Instances: %s", [i.upper() for i in sorted(all_results.keys())])
    else:
        # 回退到单次实验结果
        all_results = load_all_results()
        if not all_results:
            logger.error("No experiment results found. Run benchmark first.")
            return
        stats_tests = {}
        logger.info("Loaded single-run results: %d instances", len(all_results))
        logger.debug("Instances: %s", [i.upper() for i in sorted(all_results.keys())])

    logger.info("Generating visualization charts... [%s]",
                "AGGREGATE (mean±std)" if is_aggregate else "SINGLE RUN")

    # ── Pareto前沿 ──
    logger.debug("=== 1. Pareto Front Comparison ===")
    plot_pareto_front_per_instance(all_results)

    # ── 三角模糊数Pareto前沿 ──
    logger.debug("=== 2. Triangular Fuzzy Number (TFN) Pareto Front ===")
    plot_fuzzy_pareto_front(all_results)

    # ── 模糊Makespan对比 ──
    logger.debug("=== 3. Fuzzy Makespan Comparison (t1, t2, t3) ===")
    plot_fuzzy_makespan_comparison(all_results)

    # ── 模糊Workload对比 ──
    logger.debug("=== 4-9 Chart sections ===")
    plot_fuzzy_workload_comparison(all_results)

    # ── 每实例TFN区间 ──
    logger.debug("=== 5. Fuzzy Range Per Instance ===")
    plot_fuzzy_range_per_instance(all_results)

    # ── 收敛曲线 ──
    logger.debug("=== 6. Convergence Curves ===")
    plot_convergence_curve_per_instance(all_results)

    # ── 汇总对比 ──
    logger.debug("=== 7. Summary Comparison Charts ===")
    plot_hv_comparison(all_results)
    plot_makespan_comparison(all_results)
    plot_workload_comparison(all_results)
    plot_improvement_summary(all_results)
    plot_pf_size_comparison(all_results)
    plot_runtime_comparison(all_results)

    # ── 统计检验 ──
    if is_aggregate and stats_tests:
        logger.debug("=== 8. Statistical Tests Visualization ===")
        plot_statistical_tests(all_results, stats_tests)

    # ── 模糊清晰值对比 ──
    logger.debug("=== 9. Fuzzy Crisp Value Comparison ===")
    plot_fuzzy_clear_value_comparison(all_results)

    n_charts = len(all_results) * 5 + 8
    logger.info("Visualization done: %d+ charts in %s/", n_charts, BENCHMARK_DIR)


# ══════════════════════════════════════════════════════════════
# 甘特图模块 (Gantt Chart)
# ══════════════════════════════════════════════════════════════

GANNT_DIR = os.path.join(CHARTS_BASE, 'schedules')


def plot_gantt_chart(schedule, instance_name, output_dir=None):
    """
    Draw Gantt chart for a single scheduling solution.
    为单个调度解绘制甘特图。

    Parameters:
        schedule: list of dicts from decode_with_schedule(),
                  每项包含: machine_label, job_label, op_label,
                  start, finish, processing_time, job_id
        instance_name: 实例名称 (如 "Mk01")
        output_dir: 输出目录 (默认 charts/schedules/<instance>/)
    """
    if not output_dir:
        output_dir = os.path.join(GANNT_DIR, instance_name)
    os.makedirs(output_dir, exist_ok=True)

    # 按机器分组及机器编号排序
    machines = sorted(set(op["machine_label"] for op in schedule),
                      key=lambda m: int(m[1:]))
    n_machines = len(machines)
    machine_idx = {m: i for i, m in enumerate(machines)}

    # 找出一共多少个不同的job，分配颜色
    job_ids = sorted(set(op["job_id"] for op in schedule))
    n_jobs = len(job_ids)
    cmap = plt.cm.get_cmap('tab20', max(n_jobs, 20))
    job_color = {jid: cmap(i % 20) for i, jid in enumerate(job_ids)}

    # 计算makespan
    makespan = max(op["finish"] for op in schedule)

    fig, ax = plt.subplots(figsize=(max(14, makespan / 30), max(5, n_machines * 0.5)))

    for op in schedule:
        m_label = op["machine_label"]
        y = n_machines - 1 - machine_idx[m_label]  # 从上到下M1, M2, ...
        color = job_color[op["job_id"]]
        start = op["start"]
        duration = op["finish"] - op["start"]

        # 绘制矩形条
        bar = ax.barh(y, duration, height=0.7, left=start,
                      color=color, edgecolor='black', linewidth=0.5, alpha=0.85)

        # 在条上标注工序标签 (如果条足够宽)
        if duration > makespan * 0.015:
            ax.text(start + duration / 2, y, op["op_label"],
                    ha='center', va='center', fontsize=6.5,
                    fontweight='bold', color='black')

        # 在条两端标注开始/结束时间
        if duration > makespan * 0.03:
            ax.text(start + 0.5, y + 0.32, f'{start:.1f}',
                    fontsize=5.5, ha='left', va='bottom', color='gray', rotation=45)
            ax.text(op["finish"] - 0.5, y + 0.32, f'{op["finish"]:.1f}',
                    fontsize=5.5, ha='right', va='bottom', color='gray', rotation=45)

    # 坐标轴设置
    ax.set_yticks(range(n_machines))
    ax.set_yticklabels(reversed(machines))
    ax.set_xlabel('Time (Crisp Value)', fontsize=12)
    ax.set_ylabel('Machine', fontsize=12)
    ax.set_title(f'{instance_name.upper()} - Gantt Chart (RMakespan={makespan:.2f})',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, makespan * 1.03)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.invert_yaxis()  # M1在最上方

    # Job颜色图例
    legend_handles = []
    for jid in job_ids:
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=job_color[jid],
                          edgecolor='black', linewidth=0.5, label=f'J{jid + 1}')
        )
    ax.legend(handles=legend_handles, loc='upper right',
              fontsize=7, ncol=min(n_jobs, 8), title='Jobs',
              title_fontsize=8, framealpha=0.8)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'gantt_chart.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Gantt chart saved: %s", filepath)
    return filepath


def generate_gantt_from_results(results, output_dir=None):
    """
    Generate Gantt charts from experiment results JSON.
    从实验结果JSON生成甘特图。

    Parameters:
        results: dict from RMOEAD.save_results() or loaded from JSON
        output_dir: 输出目录 (默认 charts/schedules/<instance>/)
    """
    instance_name = results.get("instance", "Unknown")
    schedules_data = results.get("schedules", [])

    if not schedules_data:
        logger.warning("No schedule data in results for %s, skip Gantt chart.", instance_name)
        return []

    if not output_dir:
        output_dir = os.path.join(GANNT_DIR, instance_name)

    chart_paths = []
    # 为每个非支配解画甘特图
    for entry in schedules_data:
        rank = entry["rank"]
        schedule = entry["schedule"]
        makespan = entry["makespan_crisp"]

        filename = f'gantt_rank{rank}_makespan{makespan:.0f}.png'
        filepath = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)

        plot_gantt_chart_internal(schedule, instance_name, rank, makespan, filepath)
        chart_paths.append(filepath)

    logger.info("Generated %d Gantt charts for %s in %s",
                len(chart_paths), instance_name, output_dir)
    return chart_paths


def plot_gantt_chart_internal(schedule, instance_name, rank, makespan, filepath):
    """
    Internal Gantt chart renderer (used by both entry points).
    内部甘特图渲染函数，被两个入口共用。
    """
    machines = sorted(set(op["machine_label"] for op in schedule),
                      key=lambda m: int(m[1:]))
    n_machines = len(machines)
    machine_idx = {m: i for i, m in enumerate(machines)}

    job_ids = sorted(set(op["job_id"] for op in schedule))
    n_jobs = len(job_ids)
    cmap = plt.cm.get_cmap('tab20', max(n_jobs, 20))
    job_color = {jid: cmap(i % 20) for i, jid in enumerate(job_ids)}

    total_makespan = max(op["finish"] for op in schedule)

    fig, ax = plt.subplots(figsize=(max(14, total_makespan / 25), max(5, n_machines * 0.55)))

    for op in schedule:
        m_label = op["machine_label"]
        y = n_machines - 1 - machine_idx[m_label]
        color = job_color[op["job_id"]]
        start = op["start"]
        duration = op["finish"] - op["start"]

        ax.barh(y, duration, height=0.7, left=start,
                color=color, edgecolor='black', linewidth=0.5, alpha=0.85)

        # 标注工序 (条足够宽时才标)
        if duration > total_makespan * 0.012:
            ax.text(start + duration / 2, y, op["op_label"],
                    ha='center', va='center', fontsize=6.5,
                    fontweight='bold', color='black')

    ax.set_yticks(range(n_machines))
    ax.set_yticklabels(reversed(machines))
    ax.set_xlabel('Time (Crisp Value)', fontsize=12)
    ax.set_ylabel('Machine', fontsize=12)
    ax.set_title(f'{instance_name.upper()} - Gantt Chart #{rank} '
                 f'(Crisp Makespan={makespan:.2f})',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, total_makespan * 1.03)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    # Job图例
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=job_color[jid],
                      edgecolor='black', linewidth=0.5, label=f'J{jid + 1}')
        for jid in job_ids
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              fontsize=7, ncol=min(n_jobs, 8), title='Jobs',
              title_fontsize=8, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()