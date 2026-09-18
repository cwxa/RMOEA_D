#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""新架构验证的可视化：口径纠正 / 饱和 / 等墙钟 / AIG 门控。

产出（`charts/new_arch/`）：

  01_caliber_amplification.png  两种 HV 口径的 rel% 与放大倍数（缺陷 22）
  02_saturation.png             饱和诊断：各臂每段（等代数）HV 增量
  03_wallclock_curves.png       等墙钟 anytime 曲线族
  04_aig_gating.png             AIG 门控：实例规模 vs MWR 收益
  05_ladder_decomposition.png   论文阶梯的真实分解（实例边界口径）
  06_equal_fe_vs_equal_gen.png  等代数 vs 等求值次数：算力口径造成的符号翻转
  07_mix3_decay_g4000.png       MIX3 残余优势 vs 算力：幂律衰减与零点穿越（§3.6 判定图）

用法：
    python scripts/new_arch_plots.py
    python scripts/new_arch_plots.py --anytime logs/anytime_g2000.json
    python scripts/new_arch_plots.py --g4000 logs/anytime_g4000.json
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib import rcParams  # noqa: E402

from response_surface import (fe_multiplier, per_instance_tmax,   # noqa: E402
                              index_curves, _pstat)              # 唯一真源

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun",
                               "DejaVu Sans", "Arial"]
rcParams["axes.unicode_minus"] = False

OUT = os.path.join(ROOT, "charts", "new_arch")
C_INST = "#2166AC"      # 实例边界口径
C_BOX = "#D6604D"       # 盒口径
C_POS = "#2166AC"
C_NEG = "#E41A1C"
C_ZERO = "#9E9E9E"

ARM_TITLE = {
    "D1": "D1 纯 MOEA/D（无 LS）",
    "D2": "D2 + MIX3 初始化",
    "D5": "D5 +LS+Q-PAS+Elite",
    "RMOEAD": "RMOEAD（RL 选算子）",
}


def _load(path):
    if not os.path.exists(path):
        raise SystemExit("缺少数据文件：%s（先跑对应实验台）" % path)
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return d.get("rows", []) if isinstance(d, dict) else d


def _need_font_ok(ax):
    ax.grid(alpha=0.25, linewidth=0.6)


def summarize_caliber(path):
    cal = json.load(open(path, encoding="utf-8"))
    per = cal["per_instance"]
    agg = collections.OrderedDict()
    for inst, d in per.items():
        for key, rec in d.items():
            a = agg.setdefault(key, {"inst": [], "box": []})
            a["inst"].append(rec["inst"]["rel_pct"])
            a["box"].append(rec["box"]["rel_pct"])
    return collections.OrderedDict(
        (k, (float(np.mean(v["inst"])), float(np.mean(v["box"])))) for k, v in agg.items())


def fig_caliber(path):
    agg = summarize_caliber(path)
    keys = list(agg)
    ri = np.array([agg[k][0] for k in keys])
    rb = np.array([agg[k][1] for k in keys])
    x = np.arange(len(keys))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.2))
    b1 = ax.bar(x - w / 2, ri, w, label="实例边界口径（真实）", color=C_INST)
    b2 = ax.bar(x + w / 2, rb, w, label="盒口径（放大后）", color=C_BOX, alpha=0.85)
    for xi, (a, b) in enumerate(zip(ri, rb)):
        amp = b / a if abs(a) > 1e-12 else float("nan")
        top = max(a, b)
        ax.text(xi, top + 1.05, "×%.1f" % amp if np.isfinite(amp) else "n/a",
                ha="center", fontsize=11, fontweight="bold", color="#333333")
    for rects in (b1, b2):
        for r in rects:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2,
                    0.25 if h >= 0 else -0.25, "%+.3f%%" % h,
                    ha="center", va="bottom" if h >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("|", " vs ") for k in keys], fontsize=10)
    ax.set_ylabel("相对增幅 rel%（10 实例平均）")
    ax.set_title("缺陷 22：同一批数据在两种 HV 口径下的效应量\n盒口径系统性放大 1.3–9 倍，且倍数不是常数", fontsize=12)
    ax.legend(fontsize=10)
    _need_font_ok(ax)
    fig.tight_layout()
    p = os.path.join(OUT, "01_caliber_amplification.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_lynch(caliber_path):
    agg = summarize_caliber(caliber_path)
    keys = list(agg)
    vals = np.array([agg[k][0] for k in keys])
    labels = [k.replace("|", " vs ") for k in keys]
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    colors = [C_NEG if vals[i] < 0 else (C_ZERO if abs(vals[i]) < 0.01 else C_POS)
              for i in order]
    ax.barh([labels[i] for i in order], [vals[i] for i in order], color=colors)
    for i, idx in enumerate(order):
        v = float(vals[idx])
        # 负值的标签放到 0 线右侧，否则会压到 y 轴刻度文字上
        ax.text(v + 0.06 if v >= 0 else 0.06, i, "%+.3f%%" % v,
                va="center", ha="left", fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(min(0.0, float(vals.min())) - 0.05, float(vals.max()) * 1.14)
    ax.set_xlabel("实例边界口径相对增幅 rel%（10 实例平均）")
    ax.set_title("论文阶梯的真实分解：两个 RL 组件都是零效应", fontsize=12)
    _need_font_ok(ax)
    fig.tight_layout()
    p = os.path.join(OUT, "05_ladder_decomposition.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_aig(path):
    d = json.load(open(path, encoding="utf-8"))
    recs = d["per_instance"]
    xs = np.array([r["total_ops"] for r in recs], dtype=float)
    ys = np.array([r["dhv_rel_pct"] for r in recs], dtype=float)
    names = [r["instance"] for r in recs]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    ax = axes[0]
    col = [C_POS if r["p"] < 0.05 else C_ZERO for r in recs]
    ax.scatter(xs, ys, c=col, s=70, zorder=3, edgecolor="white", linewidth=0.8)
    for xi, yi, nm in zip(xs, ys, names):
        ax.annotate(nm, (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=9)
    if xs.size > 2:
        z = np.polyfit(xs, ys, 1)
        xr = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(xr, np.polyval(z, xr), "--", color="#333333", linewidth=1.2)
        rho = d["correlations"].get("total_ops", {})
        ax.set_title("门控判据（事前可得）：总工序数\ntotal_ops vs MWR 收益  "
                     "ρ=%+.2f, p=%.4f" % (rho.get("spearman_rho", np.nan),
                                          rho.get("spearman_p", np.nan)), fontsize=11)
    ax.set_xlabel("实例总工序数 total_ops")
    ax.set_ylabel("ΔHV rel%（MWR 相对随机的提升）")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    _need_font_ok(ax)

    ax = axes[1]
    lam = np.array([r["ms_lever_pct"] for r in recs], dtype=float)
    wl = np.array([r["wl_lever_pct"] for r in recs], dtype=float)
    ax.scatter(lam, ys, c=C_INST, s=70, zorder=3, edgecolor="white", linewidth=0.8)
    for xi, yi, nm in zip(lam, ys, names):
        ax.annotate(nm, (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("makespan 杠杆 ms_lever%")
    ax.set_ylabel("ΔHV rel%")
    ax.set_title("机制量对照（事后可得）\nms 杠杆 ρ=%+.2f, p=%.4f"
                 % (d["correlations"].get("ms_lever_pct", {}).get("spearman_rho", np.nan),
                    d["correlations"].get("ms_lever_pct", {}).get("spearman_p", np.nan)),
                 fontsize=11)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    _need_font_ok(ax)

    fig.suptitle("AIG 初始化门控：收益能被「实例规模」事前预测", fontsize=12.5)
    fig.tight_layout()
    p = os.path.join(OUT, "04_aig_gating.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_saturation(path, segments=8):
    rows = _load(path)
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["instance"], r["label"])][r["seed"]] = np.asarray(r["hist_hv"], float)
    if not by:
        raise SystemExit("no rows in %s" % path)
    insts = sorted({k[0] for k in by})
    labels = sorted({k[1] for k in by})
    fig, axes = plt.subplots(1, len(insts), figsize=(5.2 * len(insts), 4.8), squeeze=False)

    curve = {}
    for j, inst in enumerate(insts):
        ax = axes[0][j]
        for lab in labels:
            d = by.get((inst, lab))
            if not d:
                continue
            gains = []
            for h in d.values():
                edges = np.linspace(0, h.size - 1, segments + 1).astype(int)
                gains.append(np.diff(h[edges]))
            g = np.mean(gains, axis=0)
            ax.plot(np.arange(1, segments + 1), g, marker="o",
                    label=ARM_TITLE.get(lab, lab), linewidth=1.5)
            curve.setdefault(inst, {})[lab] = g
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        # 对数轴：D1 的首段增益比其余臂大一个数量级，线性轴会把它们全压平
        ax.set_yscale("log")
        ax.set_xticks(np.arange(1, segments + 1))
        ax.set_xlabel("第 k 段（等代数，共 %d 段）" % segments)
        ax.set_ylabel("该段 HV 平均增量（对数轴）")
        ax.set_title(inst, fontsize=11)
        if j == 0:
            ax.legend(fontsize=8)
        _need_font_ok(ax)
    fig.suptitle("饱和诊断（对数轴）：每段 = 等代数 %.0f 代；末段/首段 越小越饱和"
                 % (max(h.size for d in by.values() for h in d.values()) / segments),
                 fontsize=12.5)
    fig.tight_layout()
    p = os.path.join(OUT, "02_saturation.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p, curve


def fig_wallclock(path):
    """等墙钟 anytime 曲线族：**逐实例**求公共时间上限。

    公共上限必须逐实例独立：Mk07 的 G2000 只要 ~60s，Mk10 要数百秒。
    取全局最小会让大实例只在自身预算的前十几个百分点处被比较，
    把「等墙钟」降级为「等一个很小的墙钟」（缺陷 22 同族）。
    """
    rows = _load(path)
    idx = collections.defaultdict(dict)
    for r in rows:
        t = np.asarray(r["hist_time"], float) + float(r.get("init_time", 0.0))
        idx[(r["instance"], r["label"])][r["seed"]] = (t, np.asarray(r["hist_hv"], float))
    if not idx:
        raise SystemExit("no rows")
    insts = sorted({k[0] for k in idx})
    labels = sorted({k[1] for k in idx})

    tmax_by = dict(per_instance_tmax(idx))

    fig, axes = plt.subplots(1, len(insts), figsize=(5.4 * len(insts), 5.0), squeeze=False)
    for j, inst in enumerate(insts):
        ax = axes[0][j]
        tmax = tmax_by.get(inst)
        if tmax is None:
            continue
        grid = np.linspace(0.02 * tmax, tmax, 120)
        series = {}
        for lab in labels:
            d = idx.get((inst, lab))
            if not d:
                continue
            curves = [np.interp(grid, t, h) for t, h in d.values() if t[-1] >= tmax * 0.999]
            if not curves:
                continue
            series[lab] = np.asarray(curves)
        for lab, curves in series.items():
            m = curves.mean(axis=0)
            ax.fill_between(grid, np.percentile(curves, 25, axis=0),
                            np.percentile(curves, 75, axis=0), alpha=0.13)
            ax.plot(grid, m, linewidth=1.7,
                    label="%s (%d seeds)" % (ARM_TITLE.get(lab, lab), len(curves)))
        # 末段放大：四条曲线在那里是否重合（= 算法只影响"多快到"，不影响"到哪"）
        if series:
            axin = ax.inset_axes([0.40, 0.10, 0.57, 0.40])
            lo, hi = 0.55 * tmax, tmax
            vmin, vmax = np.inf, -np.inf
            for lab, curves in series.items():
                m = curves.mean(axis=0)
                axin.plot(grid, m, linewidth=1.4)
                sel = grid >= lo
                vmin = min(vmin, float(m[sel].min()))
                vmax = max(vmax, float(m[sel].max()))
            pad = max(1e-4, 0.25 * (vmax - vmin))
            axin.set_xlim(lo, hi)
            axin.set_ylim(vmin - pad, vmax + pad)
            axin.tick_params(labelsize=7)
            axin.set_title("末段放大", fontsize=8)
            _need_font_ok(axin)
        ax.set_xlabel("累计墙钟 t（秒，含初始化）")
        if j == 0:
            ax.set_ylabel("HV（实例边界口径，同实例内可比）")
        ax.set_title("%s（公共上限 %.0fs）" % (inst, tmax), fontsize=11)
        ax.legend(fontsize=8)
        _need_font_ok(ax)
    fig.suptitle("等墙钟 anytime 曲线（阴影=IQR）：曲线重合 = 算法只是「更快到达」而非「到达得更好」",
                 fontsize=12.5)
    fig.tight_layout()
    p = os.path.join(OUT, "03_wallclock_curves.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p, tmax_by


def fig_equal_fe(path, pairs=(("D5", "D2"), ("D5", "D1"))):
    """等代数 vs 等求值次数（FE）：同一臂对在两种算力口径下的配对差。

    等代数对带 RVNS 的臂**不公平** —— 它每代拿 (1+ls_trials)×n_pop 次求值，
    非 RVNS 臂只有 1×n_pop。按 FE 对齐后符号会翻转，这是「算力杠杆是假的」的核心证据。
    """
    rows = _load(path)
    n_pop = int(rows[0].get("n_pop", 100)) if rows else 100
    idx = collections.defaultdict(dict)
    for r in rows:
        idx[(r["instance"], r["label"])][r["seed"]] = np.asarray(r["hist_hv"], float)
    if not idx:
        raise SystemExit("no rows in %s" % path)
    insts = sorted({k[0] for k in idx})
    labels = sorted({k[1] for k in idx})
    fe_pg = {l: n_pop * fe_multiplier(l) for l in labels}
    fracs = np.array([0.1, 0.25, 0.5, 0.75, 1.0])

    fig, axes = plt.subplots(len(pairs), len(insts),
                             figsize=(5.4 * len(insts), 4.4 * len(pairs)), squeeze=False)
    for i, (a, b) in enumerate(pairs):
        for j, inst in enumerate(insts):
            ax = axes[i][j]
            da, db = idx.get((inst, a)), idx.get((inst, b))
            if not da or not db:
                ax.axis("off")
                ax.text(0.5, 0.5, "%s：%s 或 %s 尚无数据" % (inst, a, b),
                        ha="center", va="center", fontsize=10, color="#888888",
                        transform=ax.transAxes)
                continue
            seeds = sorted(set(da) & set(db))
            gmax_a = max(h.size for h in da.values())
            gmax_b = max(h.size for h in db.values())
            fe_cap = min(gmax_a * fe_pg[a], gmax_b * fe_pg[b])
            eq_gen, eq_fe = [], []
            for f in fracs:
                vg, vf = [], []
                for s in seeds:
                    ha, hb = da[s], db[s]
                    xa = np.arange(1, ha.size + 1, dtype=float)
                    xb = np.arange(1, hb.size + 1, dtype=float)
                    vg.append(ha[int(round(f * (ha.size - 1)))]
                              - hb[int(round(f * (hb.size - 1)))])
                    vf.append(float(np.interp(fe_cap * f / fe_pg[a], xa, ha)
                                    - np.interp(fe_cap * f / fe_pg[b], xb, hb)))
                eq_gen.append(float(np.mean(vg)))
                eq_fe.append(float(np.mean(vf)))
            ax.plot(fracs, eq_gen, "o-", color=C_INST, linewidth=1.8,
                    label="等代数（同一代）")
            ax.plot(fracs, eq_fe, "s--", color=C_NEG, linewidth=1.8,
                    label="等求值次数 FE（公平）")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(fracs)
            ax.set_xlabel("预算比例")
            ax.set_title("%s：%s − %s" % (inst, a, b), fontsize=11)
            if j == 0:
                ax.set_ylabel("配对 HV 差\n（>0 = %s 更好）" % a)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
            _need_font_ok(ax)
    fig.suptitle("等代数对带 RVNS 的臂不公平：每代求值  D1/D2=%d、D5/RMOEAD=%d（n_pop=%d）；\n"
                 "按 FE 对齐后 %s−%s（等代数为正）翻转为负 → 局部搜索买的是早期速度，不是最终质量"
                 % (fe_pg.get("D2", n_pop), fe_pg.get("D5", 2 * n_pop), n_pop,
                    pairs[0][0], pairs[0][1]), fontsize=11.5)
    fig.tight_layout()
    p = os.path.join(OUT, "06_equal_fe_vs_equal_gen.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_decay(path, gens=(200, 400, 500, 1000, 2000, 3000, 4000)):
    """07：MIX3 的残余优势 vs 算力 —— 判定"起跑优势"还是"持久机制"（§3.6）。

    左：log-log 的 |Δrel%|（只看残差为正的点）→ 幂律指数一眼可见；
    右：带符号的 Δrel% 放大到零点附近 → 谁穿过了零点、谁没有。

    实心点 = 该点同 seed 配对 Wilcoxon p<0.05；空心 = 不显著。
    这是本轮唯一能区分 H1/H2 的图，故不接受手填数字：全部从 lab 现算。
    """
    rows = _load(path)
    idx = index_curves(rows)
    instances = [i for i in ("Mk07", "Mk09", "Mk10") if (i, "D1") in idx and (i, "D2") in idx]
    if not instances:
        print("[!] fig_decay: %s 里没有 D1/D2 配对，跳过" % path)
        return ""

    FIT_MIN = 500   # 与报告 §3.6 的拟合区间一致（G=200/400 不属于幂律段）
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(14.5, 4.6))
    cols = {"Mk07": "#2166AC", "Mk09": "#B2182B", "Mk10": "#1B7837"}

    for inst in instances:
        da, db = idx[(inst, "D2")], idx[(inst, "D1")]
        seeds = sorted(set(da) & set(db))
        Gs, rel, ps = [], [], []
        for G in gens:
            v = np.asarray([da[s][1][G - 1] - db[s][1][G - 1]
                            for s in seeds
                            if da[s][1].size >= G and db[s][1].size >= G])
            if v.size < 3:
                continue
            base = float(np.mean([db[s][1][G - 1] for s in seeds if db[s][1].size >= G]))
            Gs.append(G)
            rel.append(float(v.mean()) / base * 100.0)
            ps.append(_pstat(v)[2])
        Gs, rel, ps = np.asarray(Gs, float), np.asarray(rel), np.asarray(ps)
        if Gs.size == 0:
            continue
        col = cols.get(inst, "#444444")
        for ax, y in ((axa, np.abs(rel)), (axb, rel)):
            sig = ps < 0.05
            ax.plot(Gs, y, "-", color=col, lw=1.2, alpha=0.7)
            ax.plot(Gs[sig], y[sig], "o", color=col, ms=6, label=inst)
            ax.plot(Gs[~sig], y[~sig], "o", mfc="white", mec=col, ms=6)

        # 幂律拟合：只取 G>=FIT_MIN 的**正残差**点。
        # FIT_MIN 必须与报告 §3.6 的拟合区间一致，否则"图上一个指数、报告另一个指数"
        # 正是本工程最忌讳的形态（fig6 曾把 8.60/28.71 写死）。小预算段（G=200/400）
        # 不属于幂律段，含进去会把指数系统性地拉平。
        m = (Gs >= FIT_MIN) & (rel > 0)
        if m.sum() >= 3:
            Gp, yp = Gs[m], np.abs(rel[m])
            k = np.polyfit(np.log(Gp), np.log(yp), 1)[0]
            axc.plot(Gp, yp, "o-", color=col, ms=6,
                     label="%s  指数 %.2f" % (inst, k))
        else:
            axc.plot(Gs[rel > 0], np.abs(rel[rel > 0]), "o-", color=col, ms=6, label=inst)

    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_xlabel("预算 G（代）")
    axa.set_ylabel("|D2 − D1| 相对 %")
    axa.set_title("(a) log-log：幂律衰减", fontsize=11)
    axa.grid(alpha=0.25, which="both")

    axb.axhline(0.0, color=C_ZERO, lw=1.0, ls="--")
    axb.set_xscale("log")
    axb.set_xlabel("预算 G（代）")
    axb.set_ylabel("D2 − D1 相对 %")
    axb.set_title("(b) 带符号：谁穿过了零点（实心 = p<0.05）", fontsize=11)
    axb.grid(alpha=0.25)
    axb.legend(fontsize=9, loc="upper right")

    # (b) 的零点附近看不清 → 内嵌放大轴（只画 G>=1000，纵轴 ±1.5%）
    sub = axb.inset_axes([0.10, 0.08, 0.46, 0.34])
    sub.axhline(0.0, color=C_ZERO, lw=0.9, ls="--")
    for inst in instances:
        da, db = idx[(inst, "D2")], idx[(inst, "D1")]
        seeds = sorted(set(da) & set(db))
        Gs2, rs = [], []
        for G in gens:
            if G < 1000:
                continue
            v = np.asarray([da[s][1][G - 1] - db[s][1][G - 1] for s in seeds
                            if da[s][1].size >= G and db[s][1].size >= G])
            if v.size < 3:
                continue
            base = float(np.mean([db[s][1][G - 1] for s in seeds if db[s][1].size >= G]))
            Gs2.append(G)
            rs.append(float(v.mean()) / base * 100.0)
        sub.plot(Gs2, rs, "o-", color=cols.get(inst, "#444"), ms=4, lw=1.0)
    sub.set_xscale("log")
    sub.set_title("G≥1000 放大", fontsize=8)
    sub.tick_params(labelsize=7)
    sub.grid(alpha=0.25)
    sub.set_xticks([1000, 2000, 4000])
    sub.set_xticklabels(["1k", "2k", "4k"], fontsize=7)

    axc.set_xscale("log")
    axc.set_yscale("log")
    axc.set_xlabel("预算 G（代）")
    axc.set_ylabel("|Δrel%|（只取残差为正的点）")
    axc.set_title("(c) 同一条幂律，不同常数", fontsize=11)
    axc.grid(alpha=0.25, which="both")
    axc.legend(fontsize=9)

    for ax in (axa, axb, axc):
        ax.axvline(200, color="#888888", lw=0.9, ls=":")
        ax.set_xticks([200, 500, 1000, 2000, 4000])
        ax.set_xticklabels(["200", "500", "1000", "2000", "4000"])

    fig.suptitle("MIX3 残余优势的算力衰减（G=200 为论文预算，20× = G=4000）", fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "07_mix3_decay_g4000.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("  末段残差（G=4000）：" + "  ".join(
        "%s %+.3f%%" % (i, _resid(idx, i, 4000)) for i in instances))
    return p


def _resid(idx, inst, G):
    """G 代处 D2−D1 的相对残差（%）：**seed 交集**配对，分母 = D1 在交集中的均值。

    口径必须与 `surface_markdown.pair_table_at_gens` 一致，否则图与报告会分叉。
    轨迹短于 G 的 run 一律剔除（不得用末代顶替）；无可用配对时返回 nan。
    """
    da, db = idx[(inst, "D2")], idx[(inst, "D1")]
    seeds = sorted(set(da) & set(db))
    pairs = [(float(da[s][1][G - 1]), float(db[s][1][G - 1])) for s in seeds
             if da[s][1].size >= G and db[s][1].size >= G]
    if not pairs:
        return float("nan")
    base = float(np.mean([b for _a, b in pairs]))
    if base == 0.0:
        return float("nan")
    return float(np.mean([a - b for a, b in pairs])) / base * 100.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caliber", default=os.path.join(ROOT, "logs", "ablation_ladder.caliber.json"))
    ap.add_argument("--aig", default=os.path.join(ROOT, "logs", "aig_gating.json"))
    ap.add_argument("--anytime", default=os.path.join(ROOT, "logs", "anytime_g2000.json"))
    ap.add_argument("--g4000", default=os.path.join(ROOT, "logs", "anytime_g4000.json"),
                    help="G=4000 判定实验的 lab（出 07 图）")
    ap.add_argument("--no_anytime", action="store_true",
                    help="跳过依赖长程轨迹的 02/03/06/07 图（数据未就绪时用）")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    made = []
    made.append(fig_caliber(args.caliber))
    made.append(fig_lynch(args.caliber))
    made.append(fig_aig(args.aig))
    curve, tmax_by = {}, {}
    if args.no_anytime:
        print("[i] --no_anytime：跳过 02_saturation / 03_wallclock_curves / 06_equal_fe / 07")
    else:
        p2, curve = fig_saturation(args.anytime)
        made.append(p2)
        p3, tmax_by = fig_wallclock(args.anytime)
        made.append(p3)
        if os.path.exists(args.g4000):
            made.append(fig_decay(args.g4000))
        else:
            print("[!] 缺少 %s，跳过 07_mix3_decay_g4000" % args.g4000)
        made.append(fig_equal_fe(args.anytime))

    print("已生成：")
    for p in made:
        print("  ", os.path.relpath(p, ROOT))
    if curve:
        print("\n末段/首段增益比（= 饱和指标，越小越饱和）：")
        for inst, labs in curve.items():
            for lab, g in labs.items():
                r = g[-1] / g[0] if abs(g[0]) > 1e-12 else float("nan")
                print("   %-5s %-8s %.3f   (首 %.5f → 末 %.5f)" % (inst, lab, r, g[0], g[-1]))
    if tmax_by:
        print("\n等墙钟公共时间上限（逐实例）：")
        for inst, t in tmax_by.items():
            print("   %-5s %.1fs" % (inst, t))
    return 0


if __name__ == "__main__":
    sys.exit(main())
