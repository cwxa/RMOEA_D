#!/usr/bin/env python3
"""
Shared plotting utilities for RMOEA/D visualization modules.
共享绘图工具模块：消除 visualization.py / analysis_report.py / ablation_visualization.py 之间的重复代码。

Consolidates:
  - Academic color palette (color-blind friendly, ColorBrewer / Nature style)
  - Matplotlib rcParams (font fallback, DPI, figure defaults)
  - _style_ax() — consistent axis styling
  - _source_footer() — data source footnote
  - _get_colormap() — matplotlib version-compatible colormap
  - _dedup_legend() — legend deduplication
  - _set_data_timestamp() / _DATA_TIMESTAMP — global timestamp
"""

import os
import time
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ═══════════════════════════════════ 全局学术样式 ═══════════════════════════════

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        plt.style.use('ggplot')

# ── 跨平台字体回退链：Windows → Linux → macOS ──
rcParams['font.sans-serif'] = [
    'DejaVu Sans',        # 通用英文字体（所有平台）
    'Noto Sans CJK SC',   # Linux 主流中文字体
    'WenQuanYi Micro Hei', # Linux 备选中文
    'SimHei',             # Windows 中文字体
    'Microsoft YaHei',    # Windows 现代中文
    'PingFang SC',        # macOS 中文字体
    'Heiti SC',           # macOS 备选中文
    'Arial',              # 通用回退
]
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.1
rcParams['figure.figsize'] = (10, 6)
rcParams['axes.titlesize'] = 13
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.labelsize'] = 11
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['lines.linewidth'] = 2.0
rcParams['lines.markersize'] = 6
rcParams['errorbar.capsize'] = 3

# ═══════════════════════════════════ 学术配色 ═══════════════════════════════════
# Color-blind friendly, 源自 ColorBrewer / Nature 风格

C_RMOEA    = '#2166AC'   # 深蓝 — RMOEA/D
C_MOEA     = '#D6604D'   # 暖橙 — MOEA/D
C_MOEAD    = '#984EA3'   # 紫   — MOEA/D (消融)
C_FULL     = '#2166AC'   # 深蓝 — Full RMOEA/D (消融)
C_QPAS     = '#D6604D'   # 暖橙 — Q-PAS Only
C_RVNS     = '#4DAF4A'   # 绿   — RVNS Only
C_TFN_T1   = '#92C5DE'   # 浅蓝 — t₁ (Earliest)
C_TFN_T3   = '#F4A582'   # 浅粉 — t₃ (Latest)
C_IMPROVE  = '#4DAF4A'   # 绿 — 改进正值
C_DECLINE  = '#E41A1C'   # 红 — 改进负值
C_METRIC3  = '#984EA3'   # 紫 — 第三指标
C_GRAY     = '#7F7F7F'   # 灰 — 辅助
C_GRID     = '#E0E0E0'   # 极浅灰 — 网格

# ═══════════════════════════════════ 全局时间戳 ═══════════════════════════════

_DATA_TIMESTAMP = ''


def set_data_timestamp(ts=''):
    """Set global timestamp for chart footnotes.
    设置图表脚注的全局时间戳。"""
    global _DATA_TIMESTAMP
    _DATA_TIMESTAMP = ts or time.strftime('%Y-%m-%d %H:%M:%S')


def source_footer(fig, extra='', source_label='RMOEA/D Results'):
    """Add data source footnote to figure bottom-right.
    在图表右下角添加数据来源脚注。"""
    ts = _DATA_TIMESTAMP or time.strftime('%Y-%m-%d %H:%M:%S')
    s = f'Source: {source_label} | Generated: {ts}'
    if extra:
        s += f' | {extra}'
    fig.text(0.99, 0.005, s, ha='right', va='bottom', fontsize=5.5,
             color=C_GRAY, style='italic', alpha=0.6)


def style_ax(ax):
    """Apply consistent academic axis styling.
    应用统一的学术风格坐标轴样式。"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(C_GRAY)
    ax.spines['bottom'].set_color(C_GRAY)
    ax.tick_params(colors=C_GRAY)
    ax.grid(True, alpha=0.3, color=C_GRID, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)


def dedup_legend(ax, ncol=2, fs=7, loc='upper left', framealpha=0.9):
    """Deduplicate legend entries by label, preserving order.
    按标签去重图例条目，保持首次出现顺序。"""
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    uh, ul = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            uh.append(h)
            ul.append(l)
    ax.legend(uh, ul, ncol=ncol, fontsize=fs, loc=loc, framealpha=framealpha)


# ── matplotlib 版本兼容：get_cmap 自 3.7 弃用，3.9+ 移除 ──

def get_colormap(name, n_colors=None):
    """Get colormap with matplotlib version compatibility.
    兼容 matplotlib 新旧版本的 colormap 获取。"""
    try:
        cmap = plt.colormaps[name]
    except AttributeError:
        cmap = plt.cm.get_cmap(name)
    if n_colors is not None:
        return cmap.resampled(n_colors)
    return cmap