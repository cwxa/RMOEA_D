# -*- coding: utf-8 -*-
"""论文 6 级消融阶梯的复现图（SCI 审美）。

输入：``scripts/ablation_ladder.py`` 的 ``logs/ablation_ladder.json``
      （可选 ``logs/ablation_ladder.json.analysis.json``）

输出：``charts/ablation/paper_ladder_reproduction.png``

四个面板：

  (a) 逐级 HV（相对 D1 的增量 %）—— 论文 Table 5 的复现位；每级标注与 D1 的
      Wilcoxon p（逐 seed 配对，跨实例合并）。
  (b) Friedman 平均排名 —— 本项目复现 vs 论文 Table 4 的并置柱状图。
  (c) 每组件隔离的效应量森林图 —— 相邻两级配对 ΔHV%±SEM，星号表示显著性。
  (d) Q-PAS 实际选到的 T 分布 —— 只画真在自适应选 T 的臂（固定 T 臂恒 100%
      会把尺度压扁）；若各柱接近 25%，说明 Q-PAS 的选择与均匀随机无异。

配色沿用 ``plot_helpers.py`` 的学术色板：深蓝 = 改进且显著，暖橙 = 改进不显著，
灰 = 不显著，红 = 变差。
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
OUT_DIR = os.path.join(ROOT, "charts", "ablation")

PAPER_ORDER = ["D1", "D2", "D3", "D4", "D5", "RMOEAD"]
PAPER_TABLE4 = {"D1": 4.6087, "D2": 4.3913, "D3": 3.6087,
                "D4": 3.0870, "D5": 2.9130, "RMOEAD": 2.3913}
STEP_COMPONENT = {
    ("D1", "D2"): "MIX3 init",
    ("D2", "D3"): "random VNS",
    ("D3", "D4"): "Q-PAS",
    ("D4", "D5"): "elite archive",
    ("D5", "RMOEAD"): "RVNS (RL pick)",
}

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
C_PAPER, C_OURS = "#B9C4CE", "#C97A22"


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load(lab_json):
    rows = json.load(open(lab_json, encoding="utf-8"))
    instances = sorted({r["instance"] for r in rows})
    fronts = collections.defaultdict(list)
    for r in rows:
        if r.get("final_pf"):
            fronts[r["instance"]].append(np.asarray(r["final_pf"], float))
    bounds = {i: estimate_hv_bounds(fronts[i]) for i in instances}
    hv = collections.defaultdict(dict)
    for r in rows:
        lo, hi = bounds[r["instance"]]
        pf = np.asarray(r["final_pf"], float) if r.get("final_pf") else np.empty((0, 2))
        hv[(r["instance"], r["label"])][r["seed"]] = (
            compute_hv(pf, ref_point=REF, norm_bounds=(lo, hi)) if len(pf) else 0.0)
    labels = sorted({r["label"] for r in rows})
    ladder = [l for l in PAPER_ORDER if l in labels]
    return rows, instances, hv, ladder


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lab_json", default=os.path.join(
        ROOT, "logs", "ablation_ladder.json"))
    ap.add_argument("--out", default=os.path.join(
        OUT_DIR, "paper_ladder_reproduction.png"))
    args = ap.parse_args()

    if not os.path.exists(args.lab_json):
        print("[错误] 找不到", args.lab_json)
        return 1
    rows, instances, hv, ladder = load(args.lab_json)
    if len(ladder) < 2:
        print("[错误] 至少需要两条臂")
        return 1

    # 先剔掉「一条数据都没有」的臂（否则空集会污染下面的交集，导致所有实例都被判空）
    ladder = [l for l in ladder if any(hv[(i, l)] for i in instances)]
    if len(ladder) < 2:
        print("[错误] 数据里可用的臂不足两条")
        return 1
    # 逐实例、跨臂取交集的 seed 集：与 ablation_ladder_analysis.py 同口径，
    # 否则 Friedman 会把「不同 seed 的观测」放在一起排名（部分完成的数据集尤其明显）。
    seeds_ci = {i: sorted(set.intersection(*[set(hv[(i, l)]) for l in ladder]) or set())
                for i in instances}
    # 交集为空的实例（某些臂在这个实例上一条都没跑）无法跨臂比较，剔除而不是算出 nan
    empty = [i for i in instances if not seeds_ci[i]]
    if empty:
        print("跳过尚未跑完的实例:", empty)
        instances = [i for i in instances if i not in empty]
    if not instances:
        print("[错误] 没有可用于跨臂比较的实例")
        return 1
    M = {l: np.array([np.mean([hv[(i, l)][s] for s in seeds_ci[i]])
                      for i in instances]) for l in ladder}

    def paired_runs(a, b):
        out = []
        for inst in instances:
            sa, sb = hv[(inst, a)], hv[(inst, b)]
            for s in sorted(set(sa) & set(sb)):
                out.append(sa[s] - sb[s])
        return np.array(out)

    # ── 相对 D1 的逐级增量（%）──
    # 口径必须与 ablation_ladder_analysis.py 完全一致：**mean(Δ) / mean(base)**
    # （比值之比），而不是 mean(Δ / base)（各实例相对增幅的平均）。
    # 两者在大效应上能差 2 个百分点（D1→D2：+14.46% vs +16.47%），
    # 会让同一张图的 (a) 面板与 (c) 面板、以及图与文档表格互相打架。
    base = ladder[0]
    rel, rel_err, p_vs_base = [], [], []
    for l in ladder:
        d = np.array([M[l][k] - M[base][k] for k in range(len(instances))])
        r = 100.0 * d.mean() / M[base].mean()
        e = 100.0 * d.std(ddof=1) / np.sqrt(len(d)) / M[base].mean() \
            if len(d) > 1 else 0.0
        rel.append(r)
        rel_err.append(e)
        pr = paired_runs(l, base) if l != base else np.zeros(3)
        p_vs_base.append(float("nan") if l == base else
                         (float(stats.wilcoxon(pr)[1]) if len(pr) >= 3
                          and pr.std() > 0 else float("nan")))

    # ── Friedman 排名（需要 >=2 实例）──
    ranks_ours = None
    if len(instances) >= 2:
        mat = np.array([M[l] for l in ladder])
        ranks_ours = np.mean([stats.rankdata(-mat[:, k])
                              for k in range(mat.shape[1])], axis=0)

    fig = plt.figure(figsize=(9.8, 8.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 0.78, 1.10],
                          hspace=0.62, wspace=0.28)

    # ── (a) 逐级 HV 增量 ──
    ax = fig.add_subplot(gs[0, :])
    xs = np.arange(len(ladder))
    cols = []
    for k, l in enumerate(ladder):
        if l == base:
            cols.append("0.60")
        elif rel[k] <= 0:
            cols.append(C_NEG)
        elif p_vs_base[k] < 0.05:
            cols.append(C_POS)
        else:
            cols.append(C_NS)
    ax.bar(xs, rel, width=0.62, color=cols, edgecolor="0.3", linewidth=0.8, zorder=3)
    for k, (v, e) in enumerate(zip(rel, rel_err)):
        ax.errorbar(k, v, yerr=e, fmt="none", ecolor="0.35", elinewidth=1.0,
                    capsize=2.6, zorder=4)
        ax.text(k, v + (e + 0.16 if v >= 0 else -(e + 0.16)),
                f"{v:+.2f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8.0,
                fontweight="bold" if (p_vs_base[k] < 0.05) else "normal",
                color="0.15" if p_vs_base[k] < 0.05 else "0.42")
        if k > 0:
            ax.text(k, min(rel) - 1.55, stars(p_vs_base[k]), ha="center",
                    va="top", fontsize=8.0,
                    color="0.20" if p_vs_base[k] < 0.05 else "0.55")
    ax.axhline(0, color="0.3", lw=0.9, zorder=2)
    ax.set_xticks(xs)
    LONG = {"D1": "RMOEA/D1", "D2": "RMOEA/D2", "D3": "RMOEA/D3",
            "D4": "RMOEA/D4", "D5": "RMOEA/D5", "RMOEAD": "RMOEA/D"}
    ax.set_xticklabels([f"{l}\n{LONG.get(l, l)}" for l in ladder], fontsize=7.9)
    ax.set_ylabel("ΔHV vs RMOEA/D1 (%)", fontsize=8.8)
    ax.set_title("(a) Paper's ablation ladder, reproduced  —  each bar adds one component",
                 fontsize=9.4, pad=7, loc="left")
    if len(instances) > 1:
        ax.set_ylim(min(rel) - 2.4, max(rel) + max(rel_err) + 1.0)
    else:
        ax.set_ylim(min(rel) - 2.4, max(rel) * 1.35 + 0.6)
    clean(ax)

    # ── (b) Friedman 排名并置 ──
    ax = fig.add_subplot(gs[1, 0])
    if ranks_ours is not None:
        w = 0.38
        ax.bar(xs - w / 2, [PAPER_TABLE4[l] for l in ladder], width=w,
               color=C_PAPER, edgecolor="0.35", linewidth=0.7, zorder=3,
               label="paper Table 4")
        ax.bar(xs + w / 2, ranks_ours, width=w, color=C_OURS, edgecolor="0.35",
               linewidth=0.7, zorder=3, label="this reproduction")
        ax.legend(fontsize=7.4, loc="upper right", framealpha=0.92)
        ax.set_ylim(0, max(max(PAPER_TABLE4.values()), ranks_ours.max()) * 1.22)
    else:
        ax.text(0.5, 0.5, "single instance\n(no Friedman)", ha="center",
                va="center", transform=ax.transAxes, fontsize=9, color="0.5")
    ax.set_xticks(xs)
    ax.set_xticklabels([l.replace("RMOEAD", "RMOE/D") for l in ladder], fontsize=7.6)
    ax.set_ylabel("mean Friedman rank\n(1 = best)", fontsize=8.4)
    ax.set_title("(b) Ranking: paper vs reproduction", fontsize=9.2,
                 pad=6, loc="left")
    clean(ax)

    # ── (c) 每组件隔离的效应量森林图 ──
    ax = fig.add_subplot(gs[1, 1])
    items = list(zip(ladder[:-1], ladder[1:]))
    ys = np.arange(len(items))[::-1]
    # 先算齐所有效应量，再统一决定数值标签放哪一侧 —— 单遍判断无法知道
    # 该点相对整条 x 轴的位置，靠近 0 的点会把标签压到 y 轴刻度上。
    stats_rows = []
    for a, b in items:
        # 统一符号约定：加上这一级的组件的效应 = HV(b) − HV(a)，>0 即正贡献
        d = np.array([M[b][k] - M[a][k] for k in range(len(instances))])
        # 同 (a) 面板：比值之比，与 ablation_ladder_analysis.py 的 rel_pct 逐位一致
        relv = 100.0 * d.mean() / M[a].mean()
        e = 100.0 * d.std(ddof=1) / np.sqrt(len(d)) / M[a].mean() \
            if len(d) > 1 else 0.0
        pr = paired_runs(b, a)
        p = float(stats.wilcoxon(pr)[1]) if len(pr) >= 3 and pr.std() > 0 \
            else float("nan")
        stats_rows.append((relv, e, p))
    span = max((abs(r[0]) + r[1] for r in stats_rows), default=1.0) or 1.0
    for y, (a, b), (relv, e, p) in zip(ys, items, stats_rows):
        c = C_NS if not (p < 0.05) else (C_POS if relv > 0 else C_NEG)
        ax.errorbar(relv, y, xerr=e, fmt="o", ms=6.4, mfc=c, mec="0.25",
                    mew=0.8, ecolor="0.42", elinewidth=1.1, capsize=3.0, zorder=4)
        # 点落在左侧 30% 区间时，标签一律放右边，否则会压到 y 轴刻度文字上
        put_right = (relv - e) < 0.30 * span
        xoff = (e + 0.18 * span) if put_right else -(e + 0.18 * span)
        ax.text(relv + xoff, y, f"{relv:+.2f}%",
                fontsize=7.8, ha="left" if put_right else "right",
                va="center",
                fontweight="bold" if p < 0.05 else "normal",
                color="0.15" if p < 0.05 else "0.45")
    ax.axvline(0, color="0.3", lw=1.0, zorder=2)
    ax.set_yticks(ys)
    # 臂组合不一定是论文那 6 条（`ablation_ladder.py --arms` 允许跑子集），
    # 相邻二级未必落在 STEP_COMPONENT 里 —— 直接下标会 KeyError 崩掉整张图。
    ax.set_yticklabels([STEP_COMPONENT.get((a, b), f"{a} → {b}")
                        for a, b in items], fontsize=7.6)
    ax.set_ylim(-0.62, len(items) - 0.38)
    ax.set_xlim(-0.06 * span, span * 1.30)
    ax.set_xlabel("ΔHV from adding this component (%)\n"
                  "(> 0 = the component helps)", fontsize=8.0)
    ax.set_title("(c) Component isolated at each step", fontsize=9.2,
                 pad=6, loc="left")
    clean(ax)
    ax.spines["left"].set_visible(True)
    # 给左侧标签留出空间
    fig.subplots_adjust(left=0.20)

    # ── (d) T 使用分布（仅 Q-PAS 臂）──
    #     固定 T 的臂恒为 100%，画进来会把 Q-PAS 的分布压扁，故只保留 Q-PAS 臂。
    ax = fig.add_subplot(gs[2, :])
    tdist = collections.defaultdict(collections.Counter)
    for r in rows:
        ht = r.get("hist_T")
        if not ht:
            continue
        if isinstance(ht, dict):
            for k, c in ht.items():
                tdist[r["label"]][int(k)] += int(c)
        else:
            for t in ht:
                if t is not None:
                    tdist[r["label"]][int(t)] += 1
    # 只保留「有 >1 种 T 被选到」的臂 = 真正在自适应选 T 的臂
    qpas_arms = [l for l in ladder
                 if tdist[l] and len(tdist[l]) > 1]
    if qpas_arms:
        ts = sorted({t for l in qpas_arms for t in tdist[l]})
        ws = 0.76 / len(qpas_arms)
        cmap = plt.get_cmap("Blues")
        for k, l in enumerate(qpas_arms):
            tot = sum(tdist[l].values())
            vals = [100.0 * tdist[l].get(t, 0) / tot for t in ts]
            ax.bar(np.arange(len(ts)) + (k - (len(qpas_arms) - 1) / 2) * ws,
                   vals, width=ws,
                   color=cmap(0.45 + 0.42 * k / max(1, len(qpas_arms) - 1)),
                   edgecolor="0.35", linewidth=0.6, zorder=3, label=l)
            for xi, v in enumerate(vals):
                ax.text(xi + (k - (len(qpas_arms) - 1) / 2) * ws, v + 0.9,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=6.6,
                        color="0.35")
        ax.legend(fontsize=7.4, ncol=len(qpas_arms), loc="upper center",
                  framealpha=0.92)
        ax.set_xticks(np.arange(len(ts)))
        ax.set_xticklabels([f"T={t}" for t in ts], fontsize=8)
        ax.set_ylim(0, max(38, max(
            100.0 * tdist[l].get(t, 0) / sum(tdist[l].values())
            for l in qpas_arms for t in ts) * 1.22))
        ax.yaxis.set_major_locator(MultipleLocator(10))
    else:
        ax.text(0.5, 0.5, "no adaptive-T arm in this run", ha="center",
                va="center", transform=ax.transAxes, fontsize=9, color="0.5")
    ax.set_ylabel("share of generations (%)", fontsize=8.4)
    ax.set_title("(d) Neighborhood size actually chosen by Q-PAS "
                 "(uniform would be 25% at each bar)", fontsize=9.2,
                 pad=6, loc="left")
    clean(ax)

    if len(instances) > 1:
        mat = np.array([M[l] for l in ladder])
        st, fp = stats.friedmanchisquare(*mat)
        foot = (f"instances: {', '.join(instances)}  |  n = 30 seeds per arm per instance  |  "
                f"$N_p$=100, $G$=200  |  Friedman over {len(ladder)} arms: "
                f"$\\chi^2$={st:.2f}, p={fp:.3g}\n"
                f"HV by reference-set normalization per instance, ref=(1.02, 1.02); "
                f"error bars: SEM of the paired difference; significance vs RMOEA/D1 "
                f"by Wilcoxon signed-rank")
    else:
        foot = (f"instance: {instances[0]}  |  n = 30 seeds per arm  |  "
                f"$N_p$=100, $G$=200  |  HV by reference-set normalization, ref=(1.02, 1.02)")
    fig.text(0.005, -0.012, foot, ha="left", va="top", fontsize=6.6, color="0.5")

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(args.out, dpi=300, facecolor="white")
    print("written:", args.out)

    # 顺带打印一份文本摘要
    print("\n-- ladder means --")
    for k, l in enumerate(ladder):
        print(f"  {l:<8} HV={M[l].mean():.5f}  Δvs{base}={rel[k]:+.2f}% "
              f"p={p_vs_base[k]:.4g} {stars(p_vs_base[k])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
