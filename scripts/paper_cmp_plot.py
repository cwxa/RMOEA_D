# -*- coding: utf-8 -*-
"""论文（Li et al., ESWA 203:117380）与复现结果的对照图。

  (a) 论文自己 Table 4 的阶梯：Friedman 排名增益
  (b) 论文自己 Table 5 的逐实例平均 ΔHV（附精确符号检验）
  (c) Mk10 复现的组件分解：同强度配对比较，含邻域尝试次数对照

(a)(b) 并置是为了暴露一个反差：论文按排名把 Q-PAS 与 RVNS 记为一等功臣
（各自 0.52 名），但这两人的逐实例效应量在全场最小、符号检验不显著。
(c) 则给出复现侧真正的贡献归属，以及 RL 引导选算子唯一显著的语境。

前置输入（**先跑这两个脚本**）：

    python scripts/paper_table5_audit.py          # -> logs/_paper_audit.json
    python scripts/t_leverage_sweep.py --instance Mk10 --arms T05,T10,...,Full_t3
                                                  # -> logs/_mk10_lab.json

用法：

    python scripts/paper_cmp_plot.py
    #   -> charts/ablation/paper_vs_reproduction.png
    #   -> logs/_paper_vs_repro.json

注：本脚本原先放在 `logs/_plot_paper_cmp.py`，而 `logs/` 在 `.gitignore` 里 ——
文档把它写进"复现方式"会让新克隆的仓库**没有这个文件**。已提升为正式脚本。
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from rmoea_d.utils.metrics import compute_hv, estimate_hv_bounds  # noqa: E402

REF = (1.02, 1.02)
OUT = os.path.join(ROOT, "charts", "ablation")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.default": "regular",
    "font.size": 9,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "0.25",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.color": "0.25",
    "ytick.color": "0.25",
    "savefig.bbox": "tight",
})

C_POS, C_NEG, C_NS = "#3B6BA5", "#C1554B", "#9AA3AB"
C_RANK, C_ACC = "#B9C4CE", "#C97A22"


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def paired_stats(hv, a, b, min_pairs=3):
    """按 seed 取交集做配对，返回 ``(相对增幅 %, SEM %, Wilcoxon p)``。

    不能两条臂各取各的 seed 再按位置相减 —— 两条臂的 seed 集不同时，
    numpy 要么直接抛 broadcast 错、要么在长度可整除时**静默算错**
    （长度 1 会被广播到整条臂）。这与实验台聚合端查出的是同一类缺陷。
    交集不足 ``min_pairs`` 时返回 ``nan`` 而不是硬算。
    """
    common = sorted(set(hv[a]) & set(hv[b]))
    if len(common) < min_pairs:
        return float("nan"), float("nan"), float("nan")
    va = np.array([hv[a][s] for s in common], float)
    vb = np.array([hv[b][s] for s in common], float)
    d = va - vb
    if d.std(ddof=1) <= 0:
        return 100.0 * d.mean() / vb.mean(), 0.0, float("nan")
    sem = 100.0 * d.std(ddof=1) / np.sqrt(len(d)) / vb.mean()
    return 100.0 * d.mean() / vb.mean(), sem, float(stats.wilcoxon(va, vb)[1])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit_json", default=os.path.join(ROOT, "logs", "_paper_audit.json"),
                    help="paper_table5_audit.py 的输出")
    ap.add_argument("--lab_json", default=os.path.join(ROOT, "logs", "_mk10_lab.json"),
                    help="t_leverage_sweep.py 的输出（Mk10 组件分解）")
    ap.add_argument("--out", default=os.path.join(OUT, "paper_vs_reproduction.png"))
    ap.add_argument("--out_json", default=os.path.join(ROOT, "logs", "_paper_vs_repro.json"))
    args = ap.parse_args()

    for label, path, how in [("论文审计", args.audit_json, "scripts/paper_table5_audit.py"),
                             ("Mk10 组件分解", args.lab_json,
                              "scripts/t_leverage_sweep.py --instance Mk10 ...")]:
        if not os.path.exists(path):
            print(f"[错误] 找不到{label}输入 {path}\n        请先运行: {how}")
            return 1

    # ══════════════════════════════════════════════════════════════════════
    audit = json.load(open(args.audit_json, encoding="utf-8"))

    STEP_KEYS = ["initial strategy (MIX3)", "randomly-selected VNS",
                 "Q-PAS  <-- under audit", "elite archive",
                 "RVNS (replaces rand. VNS)"]
    STEP_TICK = ["init\nstrategy", "random\nVNS", "Q-PAS", "elite\narchive", "RVNS"]

    mean_rel = [audit["steps"][k]["mean_rel"] for k in STEP_KEYS]
    ps = [audit["steps"][k]["p_sign"] for k in STEP_KEYS]
    RANK = [4.6087, 4.3913, 3.6087, 3.0870, 2.9130, 2.3913]
    rank_gain = [RANK[i] - RANK[i + 1] for i in range(5)]

    fig = plt.figure(figsize=(9.6, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.30], hspace=0.60, wspace=0.30)

    # ── (a) 排名增益 ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(5)
    ax.bar(xs, rank_gain, width=0.60, color=C_RANK, edgecolor="0.3", linewidth=0.8, zorder=3)
    for i, v in enumerate(rank_gain):
        ax.text(i, v + 0.020, f"{v:.2f}", ha="center", va="bottom", fontsize=8.3,
                fontweight="bold" if i in (1, 2, 4) else "normal",
                color="0.15" if i in (1, 2, 4) else "0.45")
    ax.annotate("", xy=(1, max(rank_gain) * 1.16), xytext=(4, max(rank_gain) * 1.16),
                arrowprops=dict(arrowstyle="-", color=C_ACC, lw=1.1, ls=":"))
    ax.text(2.5, max(rank_gain) * 1.20, "Q-PAS ties RVNS as the largest step",
            ha="center", va="bottom", fontsize=7.5, color=C_ACC, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(STEP_TICK, fontsize=7.8)
    ax.set_ylabel("Friedman rank gain\n(drop in average rank)", fontsize=8.6)
    ax.set_title("(a) What the paper credits, by rank\n"
                 "its Table 4: Friedman p = 1.5e-4 over 23 instances",
                 fontsize=9.2, pad=6, loc="left")
    ax.set_ylim(0, max(rank_gain) * 1.40)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    clean(ax)

    # ── (b) 逐实例效应量 + 符号检验 ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    cols = [C_NS if p >= 0.05 else (C_POS if v > 0 else C_NEG) for v, p in zip(mean_rel, ps)]
    ax.bar(xs, mean_rel, width=0.60, color=cols, edgecolor="0.3", linewidth=0.8, zorder=3)
    ax.axhline(0, color="0.3", lw=0.9, zorder=2)
    for i, (v, p) in enumerate(zip(mean_rel, ps)):
        ax.text(i, v + (0.10 if v >= 0 else -0.20), f"{v:+.2f}%", ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8.1, fontweight="bold" if p < 0.05 else "normal",
                color="0.15" if p < 0.05 else "0.42")
        ax.text(i, min(mean_rel) - 0.74, stars(p), ha="center", va="top", fontsize=8.0,
                color="0.15" if p < 0.05 else "0.5")
    ax.set_xticks(xs)
    ax.set_xticklabels(STEP_TICK, fontsize=7.8)
    ax.set_ylabel("mean per-instance ΔHV (%)", fontsize=8.6)
    ax.set_title("(b) What each step actually buys\n"
                 "its own Table 5: exact sign test, same 23 instances",
                 fontsize=9.2, pad=6, loc="left")
    ax.set_ylim(min(mean_rel) - 1.50, max(mean_rel) * 1.20)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    clean(ax)
    ax.annotate("largest rank gain,\nsmallest effect  (n.s.)", xy=(2.28, mean_rel[2] + 0.05),
                xytext=(3.05, max(mean_rel) * 0.62), fontsize=7.4, color=C_ACC,
                ha="left", va="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_ACC, lw=1.0))

    # ══════════════════════════════════════════════════════════════════════
    # (c) Mk10 复现的组件分解
    # ══════════════════════════════════════════════════════════════════════
    rows = json.load(open(args.lab_json, encoding="utf-8"))
    fronts = [np.asarray(r["final_pf"], float) for r in rows if r["final_pf"]]
    lo, hi = estimate_hv_bounds(fronts)
    hv = collections.defaultdict(dict)
    for r in rows:
        hv[r["label"]][r["seed"]] = compute_hv(np.asarray(r["final_pf"], float),
                                               ref_point=REF, norm_bounds=(lo, hi))

    def pair(a, b):
        """按 seed 交集做配对（见模块级 ``paired_stats``）。"""
        return paired_stats(hv, a, b)

    ITEMS = [
        ("Q-PAS alone vs fixed $T$=10", "QPAS2_hv_wide", "T10",
         "Q-PAS 单独（无局部搜索）", 0),
        ("random-pick LS vs $T$=10", "RandVNS", "T10",
         "加上局部搜索本身（算子随机选）", 0),
        ("RL-guided pick vs random pick", "RVNSonly", "RandVNS",
         "RL 引导选算子 替代 随机选算子", 0),
        ("add Q-PAS on top of LS", "Full", "RVNSonly",
         "在已有局部搜索上再加 Q-PAS", 0),
        ("random-pick LS vs $T$=10", "RandVNS_t3", "T10",
         "加上局部搜索本身（算子随机选）", 1),
        ("neighborhood tries 1 → 3", "RVNSonly_t3", "RVNSonly",
         "把每代邻域尝试 1 次提到 3 次", 1),
        ("RL-guided pick vs random pick", "RVNSonly_t3", "RandVNS_t3",
         "RL 引导选算子 替代 随机选算子", 1),
        ("add Q-PAS on top of LS", "Full_t3", "RVNSonly_t3",
         "在已有局部搜索上再加 Q-PAS", 1),
    ]

    ax = fig.add_subplot(gs[1, :])
    ys = np.arange(len(ITEMS))[::-1]
    for y, (lbl, a, b, _, g) in zip(ys, ITEMS):
        v, e, p = pair(a, b)
        c = C_NS if not (p < 0.05) else (C_POS if v > 0 else C_NEG)
        ax.errorbar(v, y, xerr=e, fmt="o", ms=6.6, mfc=c, mec="0.25", mew=0.8,
                    ecolor="0.42", elinewidth=1.1, capsize=3.0, zorder=4)
        ax.text(v + np.sign(v) * (e + 0.30), y + 0.24,
                f"{v:+.2f}%", fontsize=8.1, ha="left" if v >= 0 else "right",
                va="center", fontweight="bold" if p < 0.05 else "normal",
                color="0.15" if p < 0.05 else "0.42")
        ax.text(10.55, y, stars(p), ha="right", va="center", fontsize=8.4,
                fontweight="bold" if p < 0.05 else "normal",
                color="0.20" if p < 0.05 else "0.5")

    ax.axvline(0, color="0.3", lw=1.0, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([it[0] for it in ITEMS], fontsize=8.4)
    ax.set_xlim(-4.2, 11.1)
    ax.set_ylim(-0.58, 7.98)
    ax.set_xticks([-4, -2, 0, 2, 4, 6, 8, 10])
    ax.set_xlabel("ΔHV relative to the paired control (%)", fontsize=8.8)
    ax.set_title("(c) Mk10 reproduction: each component isolated at matched settings",
                 fontsize=9.2, pad=6, loc="left")
    clean(ax)
    ax.spines["left"].set_visible(True)
    ax.axhline(3.5, color="0.72", lw=1.0, ls="--", zorder=1)
    ax.text(-4.05, 7.62, "neighborhood tries per generation  $ls\\_trials$ = 1"
                         "   (paper's Algorithm 4)", fontsize=7.6, color="0.42")
    ax.text(-4.05, 3.24, "$ls\\_trials$ = 3", fontsize=7.6, color="0.42")
    ax.text(0.0, -0.185, "Mk10, $N_p$=100, $G$=200, n=30 seeds per arm, 28 arms / 840 runs  |  "
                         "HV by reference-set normalization, ref = (1.02, 1.02)  |  "
                         "error bars: SEM of the paired difference, Wilcoxon signed-rank",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.9, color="0.5")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=300, facecolor="white")
    print("written:", args.out)

    summary = {
        "paper_source": "Li, R. et al., Expert Systems With Applications 203 (2022) 117380, Tables 3-5, 7",
        "paper_steps": {k: audit["steps"][k] for k in STEP_KEYS},
        "paper_rank_gain": dict(zip(STEP_KEYS, rank_gain)),
        "mk10_pairs": {},
        "hv_bounds": {"lo": list(map(float, lo)), "hi": list(map(float, hi)), "ref": list(REF)},
        "pairing": "seed intersection per arm pair",
    }
    for (lbl, a, b, _, g) in ITEMS:
        v, e, p = pair(a, b)
        summary["mk10_pairs"][f"{a} vs {b}"] = {
            "label": lbl, "ls_trials": g, "dHV_pct": v, "sem_pct": e, "p": p}
    json.dump(summary, open(args.out_json, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)

    print("\n-- Mk10 component decomposition (n=30) --")
    for (lbl, a, b, _, g), y in zip(ITEMS, ys):
        v, e, p = pair(a, b)
        print(f"  ls_trials={g}  {a + ' vs ' + b:<32} {v:+7.2f}% ±{e:5.2f}  "
              f"p={p:<9.4g} {stars(p)}")
    print("\n-- paper's own ladder (Table 5) --")
    for k, v, p in zip(STEP_KEYS, mean_rel, ps):
        print(f"  {k:<30} {v:+7.2f}%  sign p={p:<9.4g} {stars(p)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
