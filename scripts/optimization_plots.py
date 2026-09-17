#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本轮「算法优化 + 工程优化」的可视化。

产出（`charts/optimization/`）：

  01_speedup.png        热路径优化前后的墙钟对照 + 函数级耗时变化
  02_init_variants.png  初始化变体在 Mk10 上的 HV 分布与效应量
  03_holdout_forest.png 留出集森林图（Mk10 / Mk07 / Mk09 三方）
  04_wallclock_hv.png   墙钟–HV 律（等算力实验，回答"算力还是策略"）
  05_pareto_fronts.png  不同初始化的代表性 Pareto 前沿
  06_lever_summary.png  本轮所有杠杆的「效应量」排序

用法：
    python scripts/optimization_plots.py
"""
import collections
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import rcParams

from rmoea_d.utils import plot_helpers as ph
from rmoea_d.utils.metrics import compute_hv, estimate_hv_bounds

# 中文优先的字体链（本机 Microsoft YaHei / SimHei 均可用）
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun",
                               "DejaVu Sans", "Arial"]
rcParams["axes.unicode_minus"] = False

OUT = os.path.join(ROOT, "charts", "optimization")
REF = (1.02, 1.02)

C_BASE = "#9E9E9E"
C_NEW = "#2166AC"
C_POS = "#2166AC"
C_ZERO = "#BDBDBD"
C_NEG = "#E41A1C"
C_HL = "#D6604D"

VARIANT_LABEL = {
    "I_rand":   "纯随机 (论文 D1)",
    "I_no_r":   "1/2 LS + 1/2 GW",
    "I_half_r": "1/2 随机 + 1/4+1/4",
    "I_mix3":   "MIX3 论文口径",
    "I_gw_spt": "1/4×4 (含 OS-SPT)",
    "I_spt":    "1/3 R + 1/3 LS + 1/3 OS-SPT",
    "I_mwr":    "1/3 R + 1/3 LS + 1/3 OS-MWR",
}
VARIANT_ORDER = ["I_rand", "I_half_r", "I_no_r", "I_mix3", "I_gw_spt",
                 "I_spt", "I_mwr"]


def _ensure_out():
    os.makedirs(OUT, exist_ok=True)


def _save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("-> %s" % p, flush=True)


def _load_json(rel):
    p = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _hv_table(labs):
    """把多个 lab 合并，用共享盒算 HV。返回 {arm: {seed: hv}}。"""
    rows = []
    for p in labs:
        d = _load_json(p)
        if d:
            rows.extend(d)
    if not rows:
        return None
    fronts = [np.asarray(r["final_pf"], float) for r in rows if r.get("final_pf")]
    lo, hi = estimate_hv_bounds(fronts)
    hv = collections.defaultdict(dict)
    for r in rows:
        if r.get("final_pf"):
            hv[r["label"]][r["seed"]] = compute_hv(
                np.asarray(r["final_pf"], float), ref_point=REF,
                norm_bounds=(lo, hi))
    return hv


# ══════════════════════════════ 01 加速对照 ══════════════════════════════

def _parse_profile(rel):
    """解析 profile_wall.py 的输出表。

    注意：函数名里**可能含空格**（如 `moead._repair_ma_for_os(from ops)`）。
    早前的 `^(\\S+)\\s+...` 会把这些行整条漏掉 —— 于是图上恰好少了
    **收益最大的那一项**。这里改成非贪婪名字 + 多空格分隔。
    """
    p = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return {}
    out = {}
    pat = re.compile(r"^(.+?)\s+([\d.]+)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)\s*$")
    for line in open(p, encoding="utf-8"):
        if "#self" in line:
            continue
        m = pat.match(line.rstrip())
        if m:
            out[m.group(1).strip()] = float(m.group(2))
    return out


def fig1_speedup():
    base = _load_json("logs/_speedup_base.json")
    new = _load_json("logs/_speedup_new.json")
    if not base or not new:
        print("[skip] 01_speedup: 缺 _speedup_*.json")
        return
    gens = sorted(base["wall"], key=int)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # ── (a) 墙钟 ──
    ax = axes[0]
    x = np.arange(len(gens))
    w = 0.36
    tb = [min(base["wall"][g]) for g in gens]
    tn = [min(new["wall"][g]) for g in gens]
    ax.bar(x - w / 2, tb, w, label="优化前", color=C_BASE, edgecolor="white")
    ax.bar(x + w / 2, tn, w, label="优化后", color=C_NEW, edgecolor="white")
    top = max(max(tb), max(tn))
    for xi, (a, b) in enumerate(zip(tb, tn)):
        ax.text(xi - w / 2, a + top * 0.015, "%.2fs" % a, ha="center",
                va="bottom", fontsize=8.5)
        ax.text(xi + w / 2, b + top * 0.015, "%.2fs" % b, ha="center",
                va="bottom", fontsize=8.5, fontweight="bold")
        ax.annotate("%.2f×  (%.0f%%)" % (a / b, (b / a - 1) * 100),
                    xy=(xi, max(a, b) * 1.13), ha="center", fontsize=10,
                    color=ph.C_DECLINE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["%s 代" % g for g in gens])
    ax.set_ylabel("墙钟 (s)")
    ax.set_ylim(0, top * 1.28)
    ax.set_title("(a) 同配置单次运行墙钟  (Mk10, Np=100, 取 3 次最优)")
    ax.legend(framealpha=0.9, loc="center left")
    ph.style_ax(ax)

    # 同配置 HV 必须完全相同（逐位一致）
    # 注：Microsoft YaHei 无 U+2713/2717 字形，改用 ASCII 标记避免豆腐块
    hvs = [base["hv"][g] == new["hv"][g] for g in gens]
    ax.text(0.02, 0.97,
            "同代 HV 逐位相同:\n" + "\n".join(
                "G=%s  %.9f  %s" % (g, base["hv"][g], "OK" if ok else "NG")
                for g, ok in zip(gens, hvs)),
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
            color="#2E7D32" if all(hvs) else ph.C_DECLINE,
            bbox=dict(boxstyle="round,pad=0.4", fc="#F1F8E9", ec="#AED581",
                      lw=0.8))

    # ── (b) 函数级耗时 ──
    ax = axes[1]
    pb = _parse_profile("E:/!cwx/_rmoea_base/_pw_base.txt")
    pn = _parse_profile("logs/_pw_new.txt")
    if not pb or not pn:
        pb = pb or {}
        pn = pn or {}
    keys = sorted(set(pb) | set(pn), key=lambda k: -max(pb.get(k, 0), pn.get(k, 0)))
    keys = [k for k in keys if max(pb.get(k, 0), pn.get(k, 0)) > 0.02][:7][::-1]
    short = {k: k.replace("(from ops)", "").replace("(from enc)", "")
                .replace("moead.", "moead.").replace("rvns.", "rvns.") for k in keys}
    y = np.arange(len(keys))
    h = 0.36
    ax.barh(y + h / 2, [pb.get(k, 0) for k in keys], h, label="优化前",
            color=C_BASE, edgecolor="white")
    ax.barh(y - h / 2, [pn.get(k, 0) for k in keys], h, label="优化后",
            color=C_NEW, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([short[k] for k in keys], fontsize=8)
    ax.set_xlabel("自身+子调用累计耗时 (s, G=60)")
    ax.set_title("(b) 各函数耗时变化  (真实计时，非 cProfile)")
    ax.legend(framealpha=0.9, loc="lower right")
    ph.style_ax(ax)

    fig.suptitle("热路径优化：墙钟 -35%%（随机流逐位不变）".replace("%%", "%"),
                 fontsize=14, fontweight="bold", y=1.01)
    ph.source_footer(fig, "logs/_speedup_{base,new}.json; profile_wall.py")
    _save(fig, "01_speedup.png")


# ══════════════════════════ 02 初始化变体（Mk10） ══════════════════════════

def fig2_init_variants():
    hv = _hv_table(["logs/_mk10_init.json", "logs/_mk10_lab.json"])
    if not hv:
        print("[skip] 02_init_variants: 缺 Mk10 数据")
        return
    present = [v for v in VARIANT_ORDER if v in hv]
    # 子图间距放大 + 短标签：否则 (b) 的 y 标签会压到 (a) 的箱线图上
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8),
                             gridspec_kw=dict(wspace=0.42))
    SHORT = {"I_rand": "纯随机", "I_half_r": "1/2R+1/4L+1/4G",
             "I_no_r": "1/2L+1/2G", "I_mix3": "MIX3(论文)",
             "I_gw_spt": "1/4×4", "I_spt": "1/3R+1/3L+1/3SPT",
             "I_mwr": "1/3R+1/3L+1/3MWR"}

    # ── (a) 箱线图 ──
    ax = axes[0]
    data = [list(hv[v].values()) for v in present]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops=dict(color="#212121", lw=1.6),
                    flierprops=dict(marker="o", ms=3, mfc="#BDBDBD", mec="none"))
    for i, v in enumerate(present):
        bp["boxes"][i].set_facecolor(C_HL if v == "I_mwr" else
                                     (C_BASE if v in ("I_rand",) else "#90CAF9"))
        bp["boxes"][i].set_alpha(0.85)
    for i, v in enumerate(present):
        m = float(np.mean(data[i]))
        ax.text(i + 1, m, "%.3f" % m, ha="center", va="bottom", fontsize=8,
                fontweight="bold")
    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels([SHORT[v] for v in present], rotation=18,
                       ha="right", fontsize=8.5)
    ax.set_ylabel("最终 HV（共享归一化盒）")
    ax.set_title("(a) 各初始化变体的 HV 分布  (Mk10 开发集, 30 seeds)")
    ph.style_ax(ax)

    # ── (b) 效应量 ──
    ax = axes[1]
    paper = "I_mix3"
    from scipy import stats
    rows = []
    for v in present:
        if v == paper:
            continue
        ss = sorted(set(hv[v]) & set(hv[paper]))
        va = np.array([hv[v][s] for s in ss])
        vb = np.array([hv[paper][s] for s in ss])
        d = va - vb
        rows.append((v, d.mean() / vb.mean() * 100,
                     float(stats.wilcoxon(va, vb)[1]), int((d > 0).sum()), len(d)))
    rows.sort(key=lambda r: r[1])
    y = np.arange(len(rows))
    cols = [C_POS if r[1] > 0 and r[2] < 0.05 else (C_ZERO if r[2] >= 0.05 else C_NEG)
            for r in rows]
    ax.barh(y, [r[1] for r in rows], 0.62, color=cols, edgecolor="white")
    for i, r in enumerate(rows):
        star = "***" if r[2] < 0.001 else "**" if r[2] < 0.01 else "*" if r[2] < 0.05 else "n.s."
        ax.text(r[1] + (0.7 if r[1] >= 0 else -0.7), i, "%+.2f%%  %s (%d/%d)"
                % (r[1], star, r[3], r[4]),
                va="center", ha="left" if r[1] >= 0 else "right", fontsize=8.5)
    ax.axvline(0, color="#616161", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[r[0]] for r in rows], fontsize=9)
    ax.set_xlabel("相对论文口径 MIX3 的 HV 变化 (%)")
    ax.set_xlim(min(r[1] for r in rows) * 1.45 - 2,
                max(r[1] for r in rows) * 1.5 + 3)
    ax.set_title("(b) 相对 MIX3 的净增益  (Wilcoxon 配对, n=30)")
    ph.style_ax(ax)

    fig.suptitle("初始化变体扫描（Mk10 开发集）：MIX3 的三条分支从未动过 OS 维度；"
                 "OS-MWR 在此集上 +8.60%***，但留出集判负（见 03）",
                 fontsize=12.5, fontweight="bold", y=1.03)
    ph.source_footer(fig, "logs/_mk10_init.json + _mk10_lab.json")
    _save(fig, "02_init_variants.png")


# ══════════════════════════ 03 留出集森林图 ══════════════════════════

def fig3_holdout_forest():
    from scipy import stats
    sets = [("Mk10 (开发)", ["logs/_mk10_init.json", "logs/_mk10_lab.json"]),
            ("Mk07 (留出)", ["logs/_mk07_init.json"]),
            ("Mk09 (留出)", ["logs/_mk09_init.json"])]
    tables = {}
    for tag, labs in sets:
        t = _hv_table(labs)
        if t:
            tables[tag] = t
    if len(tables) < 2:
        print("[skip] 03_holdout_forest: 留出集数据未就绪")
        return

    arms = [v for v in VARIANT_ORDER if all(v in t for t in tables.values())]
    arms = [a for a in arms if a != "I_mix3"]
    if not arms:
        print("[skip] 03_holdout_forest: 无公共臂")
        return

    fig, ax = plt.subplots(figsize=(11.5, 0.62 * len(arms) + 2.4))
    tags = list(tables.keys())
    colors = ["#2166AC", "#D6604D", "#4DAF4A"]
    offs = np.linspace(-0.24, 0.24, len(tags))

    for k, tag in enumerate(tags):
        hv = tables[tag]
        for i, a in enumerate(arms):
            ss = sorted(set(hv[a]) & set(hv["I_mix3"]))
            va = np.array([hv[a][s] for s in ss])
            vb = np.array([hv["I_mix3"][s] for s in ss])
            d = (va - vb) / vb.mean() * 100
            m = d.mean()
            se = d.std(ddof=1) / np.sqrt(len(d))
            ci = 1.96 * se
            p = float(stats.wilcoxon(va, vb)[1])
            yy = i + offs[k]
            ax.errorbar(m, yy, xerr=ci, fmt="o", ms=6, capsize=3,
                        color=colors[k], ecolor=colors[k],
                        label=tag if i == 0 else None, alpha=0.9)
            # 每个实例都标出 rel% + 显著性：本轮的关键事实正是"符号一致但幅度塌陷"
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax.text(m + ci + 0.35, yy, "%+.2f%% %s" % (m, star), va="center",
                    fontsize=7.5, color=colors[k])
    ax.axvline(0, color="#616161", lw=1.2)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([VARIANT_LABEL[a] for a in arms], fontsize=9.5)
    ax.set_xlabel("相对 MIX3 的 HV 变化 (%)   [点=均值, 线=95%CI]")
    ax.set_title("留出集确认：符号跨实例一致，但幅度塌陷 → 不构成普适改进\n"
                 "OS-MWR：Mk10 +8.60%*** → Mk07 +0.34% n.s. → Mk09 +0.30% n.s."
                 "（方向 3/3 同号，幅度差 25×；增益是 Mk10 特有的）", fontsize=11.5)
    ax.legend(framealpha=0.9, loc="lower right")
    ax.invert_yaxis()
    ph.style_ax(ax)
    ph.source_footer(fig, "_mk{10,07,09}_init.json（相对差与盒无关）")
    _save(fig, "03_holdout_forest.png")


# ══════════════════════════ 04 墙钟–HV 律 ══════════════════════════

def fig4_wallclock_hv():
    """墙钟–HV 律：只画**等算力对照族**。

    这 6 个臂互相之间**只改预算的构成**（"多跑代数" vs "每代多试几次"），
    算法完全相同。它们的 HV 与墙钟严格同序（ρ=+1.000），
    就是"预算怎么构成无关、预算有多少才有关"这句话的原始证据。

    不要把 `_mk10_merged_eqc.json` 里的 Q-PAS 臂一起画进来：
    `QPAS2_dv_cvnorm` 的单次墙钟是 1120s（比其它点大两个数量级），
    会把横轴拉爆、41 个点全挤在左端 —— 图就没法看了。
    """
    rows = _load_json("logs/_mk10_merged_eqc.json")
    if not rows:
        print("[skip] 04_wallclock_hv: 缺 _mk10_merged_eqc.json")
        return

    FAMILY = ["RVNSonly", "RVNSonly_G290", "RVNSonly_t3",
              "RandVNS_G440", "RVNSonly_G440", "RVNSonly_G586"]
    hv = _hv_table(["logs/_mk10_merged_eqc.json"])
    agg = {}
    for r in rows:
        if r["label"] not in FAMILY:
            continue
        a = agg.setdefault(r["label"], {"t": [], "hv": []})
        a["t"].append(r["total_time"])
        a["hv"].append(hv[r["label"]][r["seed"]])
    missing = [k for k in FAMILY if k not in agg]
    if missing:
        print("[skip] 04_wallclock_hv: 缺臂 %s" % missing)
        return
    pts = sorted([(k, np.mean(v["t"]), np.mean(v["hv"]), np.std(v["hv"]))
                  for k, v in agg.items()], key=lambda p: p[1])

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    xs = np.array([p[1] for p in pts])
    ys = np.array([p[2] for p in pts])
    es = np.array([p[3] for p in pts])
    # 误差棒 = 30 个 seed 的 HV 标准差（很小，说明排序不是噪声）
    ax.errorbar(xs, ys, yerr=es, fmt="o", ms=8, capsize=4, color=C_NEW,
                ecolor="#90A4AE", elinewidth=1.2, zorder=3,
                markeredgecolor="white", markeredgewidth=0.9)
    for i, (k, t, v, _s) in enumerate(pts):
        ax.annotate(k, (t, v), textcoords="offset points",
                    xytext=(-6, 11) if i % 2 == 0 else (-6, -17),
                    ha="right", fontsize=8, color="#37474F")
    z = np.polyfit(xs, ys, 1)
    xr = np.linspace(xs.min() * 0.93, xs.max() * 1.04, 50)
    ax.plot(xr, np.polyval(z, xr), "--", color=C_HL, lw=1.6, label="线性拟合")
    from scipy import stats
    rho = stats.spearmanr(xs, ys)
    ax.text(0.03, 0.96,
            "Spearman ρ = %+.4f  (p=%.3g)\nn = %d 个臂，每臂 30 seeds"
            % (rho.statistic, rho.pvalue, len(pts)),
            transform=ax.transAxes, va="top", fontsize=10.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", fc="#FFF8E1", ec="#FFB74D"))
    ax.text(0.03, 0.13,
            "这 6 个臂**只改预算构成**（多跑代数 vs 每代多试几次），\n"
            "算法相同 → HV 与墙钟严格同序：\n"
            "预算怎么构成无关，预算有多少才有关。".replace("**", ""),
            transform=ax.transAxes, va="bottom", fontsize=8.5, color="#37474F",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5", ec="#BDBDBD"))
    ax.set_xlabel("平均单次运行墙钟 (s)")
    ax.set_ylabel("平均最终 HV（共享归一化盒）")
    ax.set_title("墙钟–HV 律：等算力对照族内，HV 是墙钟的单一函数")
    ax.legend(framealpha=0.9, loc="lower right")
    ph.style_ax(ax)
    ph.source_footer(fig, "logs/_mk10_merged_eqc.json（6 个等算力臂）")
    _save(fig, "04_wallclock_hv.png")


# ══════════════════════════ 05 前沿对比 ══════════════════════════

def fig5_pareto_fronts():
    labs = ["logs/_mk10_init.json"]
    rows = []
    for p in labs:
        d = _load_json(p)
        if d:
            rows.extend(d)
    if not rows:
        print("[skip] 05_pareto_fronts: 缺 Mk10 初始化数据")
        return
    # 选一个 seed 做代表：取 I_mwr 的 HV 中位数所在 seed（避免挑好看的单例）
    base_rows = [r for r in rows if r["label"] == "I_mwr"]
    if not base_rows:
        base_rows = rows
    hv_by_seed = collections.defaultdict(list)
    for p in ["logs/_mk10_init.json"]:
        d = _load_json(p) or []
        fts = [np.asarray(x["final_pf"], float) for x in d if x.get("final_pf")]
        lo, hi = estimate_hv_bounds(fts)
        for r in d:
            if r.get("final_pf") and r["label"] == "I_mwr":
                hv_by_seed[r["seed"]].append(compute_hv(
                    np.asarray(r["final_pf"], float), ref_point=REF,
                    norm_bounds=(lo, hi)))
    seeds = sorted(hv_by_seed)
    med = float(np.median([np.mean(hv_by_seed[s]) for s in seeds]))
    seed = min(seeds, key=lambda s: abs(np.mean(hv_by_seed[s]) - med))

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    style = {"I_rand": (C_BASE, "o", "纯随机 (论文 D1)"),
             "I_mix3": ("#2166AC", "s", "MIX3 论文口径"),
             "I_spt":  ("#4DAF4A", "^", "MIX3 + OS-SPT"),
             "I_mwr":  (C_HL, "D", "MIX3 + OS-MWR")}
    for label, (c, mk, name) in style.items():
        for r in rows:
            if r["label"] == label and r["seed"] == seed:
                pts = np.asarray(r["final_pf"], float)
                o = np.argsort(pts[:, 0])
                ax.plot(pts[o, 0], pts[o, 1], mk + "-", color=c, ms=5, lw=1.4,
                        alpha=0.9, label=name)
                break
    # 右下方补一个"前沿越靠左下越好"的提示，避免读者误读
    ax.text(0.98, 0.05, "越靠左下越好", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9, color="#616161",
            bbox=dict(boxstyle="round,pad=0.35", fc="#FAFAFA", ec="#BDBDBD"))
    ax.set_xlabel("Makespan（清晰值）")
    ax.set_ylabel("总机器工作负载（清晰值）")
    ax.set_title("同一 seed 下的 Pareto 前沿对比（Mk10, seed=%d，取 MWR 中位 HV 的 seed）" % seed)
    ax.legend(framealpha=0.9, loc="upper right")
    ph.style_ax(ax)
    ph.source_footer(fig, "logs/_mk10_init.json")
    _save(fig, "05_pareto_fronts.png")


# ══════════════════════════ 06 杠杆总览 ══════════════════════════

def fig6_lever_summary():
    """本轮（含之前几轮）所有杠杆的效应量排序。

    数据来源逐条列在下方注释里；这里只做展示，不做新的统计。
    """
    items = [
        ("初始化 MIX3\n(vs 纯随机)", 28.71, "算力 1.00×", "logs/_init_variant_report.txt"),
        ("初始化 OS-MWR\nMk10 (G200 / G440)", 8.60, "算力 1.00× / 2.2×", "logs/_init_variant_report.txt"),
        ("邻域尝试 1→3\n(算力不匹配)", 4.68, "算力 2.22×", "docs/qpas-optimization-plan.md"),
        ("初始化 OS-MWR\nMk07/Mk09 留出集", 0.32, "算力 1.00× · n.s.", "logs/_init_holdout_mk07mk09.txt"),
        ("邻域尝试 1→3\n(等算力，已归零)", 0.00, "算力 1.03×", "logs/_eqc_compare.txt"),
        ("RL 选 T (Q-PAS)\n(vs 最优固定 T)", 0.00, "算力 1.00×", "logs/_phase_oracle.txt"),
        ("预算分配 ABA\n(等算力)", -0.64, "算力 1.00×", "logs/_aba_holdout.txt"),
    ]
    items.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    y = np.arange(len(items))
    vals = [r[1] for r in items]
    cols = [C_POS if v > 0.5 else (C_NEG if v < -0.5 else C_ZERO) for v in vals]
    ax.barh(y, vals, 0.6, color=cols, edgecolor="white")
    for i, r in enumerate(items):
        ax.text(r[1] + (0.6 if r[1] >= 0 else -0.6), i, "%+.2f%%   %s" % (r[1], r[2]),
                va="center", ha="left" if r[1] >= 0 else "right", fontsize=8.5)
    ax.axvline(0, color="#616161", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in items], fontsize=9)
    ax.set_xlabel("HV 变化 (%)")
    ax.set_xlim(-6, 38)
    ax.set_title("本轮全部杠杆一览：初始化是唯一有空间的维度，但增益不可跨实例复现\n"
                 "（OS-MWR 在 Mk10 上 +8.60%***，到留出集只剩 +0.3% n.s.）")
    ph.style_ax(ax)
    ph.source_footer(fig, "各条来源见右侧标注文件")
    _save(fig, "06_lever_summary.png")


def main():
    _ensure_out()
    for fn in (fig1_speedup, fig2_init_variants, fig3_holdout_forest,
               fig4_wallclock_hv, fig5_pareto_fronts, fig6_lever_summary):
        try:
            fn()
        except Exception as e:      # 单张失败不影响其余
            print("[error] %s: %s" % (fn.__name__, e), flush=True)
    print("done -> %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
