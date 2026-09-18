# -*- coding: utf-8 -*-
"""anytime（逐代）分析：把「抬高渐近值」与「只加速早期收敛」分开。

动机
----
项目的头号未检验命题是「两个不同算法在同一预算下，HV 是否相同」。
终点指标 `final_hv` 无法区分两种机制：

    (a) 抬高渐近值 —— 同样预算下真的更好；
    (b) 只加速早期收敛 —— 曲线早期更高，后期被追平。

两者在 final_hv 上可能完全一样，逐代轨迹上却形状相反。
`logs/ablation_ladder.json` 里已经落盘了 `hist_hv`（每条 200 代 × 2400 runs），
本脚本直接用它，**零新增算力**。

口径（重要）
------------
`hist_hv` 由 `ablation_ladder._run_one` 用**实例确定性边界** `instance_hv_bounds`
计算：同一实例内所有臂共用同一边界 → **臂间可比**；
但**跨实例不可比**（Mk07 与 Mk10 量级差数倍），故一切统计一律**逐实例独立**，
绝不按 (label, seed) 摊平多实例。

为什么分段而不是逐代检验
------------------------
逐代做 200 次显著性检验会带来严重的多重比较问题。
故按 5 个世代区间各自「先在 run 内取均值、再跨 seed 配对检验」，
每对臂每实例只有 5 个检验，且给效应量 Cohen's dz。

用法：
    python scripts/anytime_analysis.py --labs logs/ablation_ladder.json
    python scripts/anytime_analysis.py --pairs RMOEAD=D5,RMOEAD=D5_fixedT
    python scripts/anytime_analysis.py --labs logs/_mk10_traj.json \\
        --pairs T15=T05,T50=T15
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 世代分段：边界固定，避免「挑一个好看的截点」这种事后选择
SEGMENTS = [(1, 25), (26, 50), (51, 100), (101, 150), (151, 200)]

# 论文阶梯里值得单独看的臂对（左 = 被测，右 = 对照）
DEFAULT_PAIRS = [
    ("RMOEAD", "D5"),          # RL 选算子 vs 随机选算子（同一阶梯位置）
    ("RMOEAD", "D5_fixedT"),   # RL 选算子 + Q-PAS vs 两者都关
    ("D5", "D4"),              # Elite archive
    ("D5_fixedT", "D5"),       # Q-PAS
    ("D5", "D3"),              # 随机选择 VNS
    ("D2", "D1"),              # MIX3 初始化
]

# label 之间的别名：不同实验文件对同一条臂的叫法可能不同
ALIAS = {
    "RVNSonly": "D5_fixedT",       # 无 Q-PAS、有 LS、随机选算子
    "RandVNS": "D5_fixedT",
}


def _canon(label):
    return ALIAS.get(label, label)


def load_rows(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("rows", [])
    return data


def index_hist(rows):
    """(instance, label_canonical) -> {seed: 1-D array of per-generation HV}。"""
    by = {}
    for r in rows:
        hv = r.get("hist_hv")
        if hv is None:
            hv = r.get("hv")          # `_mk10_traj.json` 用的是裸 `hv`
        if not hv:
            continue
        inst, lab, seed = r.get("instance"), _canon(str(r.get("label"))), r.get("seed")
        if inst is None or seed is None:
            continue
        by.setdefault((inst, lab), {})[seed] = np.asarray(hv, dtype=float)
    return by


def index_final(rows):
    """(instance, label) -> {seed: final_hv}（末代 archive 口径）。

    与 `index_hist` 的 hist_hv 是**两个不同口径**：
        hist_hv[g] 来自 `h["hv"]`      —— 每代 population 的 Pareto 前沿，描述搜索过程；
        final_hv   来自 `res["final_pf"]` —— 末代 archive 重新解码后的前沿，描述输出存档。
    `enable_elite` 只动 archive，所以它只可能影响 final_hv。
    任何只用一个口径的臂间比较都会漏掉或误判一类组件。
    """
    by = {}
    for r in rows:
        if r.get("final_hv") is None:
            continue
        inst, lab, seed = r.get("instance"), _canon(str(r.get("label"))), r.get("seed")
        if inst is None or seed is None:
            continue
        by.setdefault((inst, lab), {})[seed] = float(r["final_hv"])
    return by


def paired_stats(a, b):
    """两个等长一维数组的配对统计。返回 (diff, wins, p, dz)。"""
    d = a - b
    n = d.size
    if n == 0:
        return float("nan"), 0, float("nan"), float("nan")
    diff = float(np.mean(d))
    wins = int(np.sum(d > 0))
    if np.allclose(d, 0.0):
        p = 1.0
    elif n < 3:
        p = float("nan")
    else:
        try:
            p = float(stats.wilcoxon(a, b).pvalue)
        except ValueError:
            p = 1.0
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    dz = float(diff / sd) if sd > 0 else 0.0
    return diff, wins, p, dz


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labs", default=os.path.join(ROOT, "logs", "ablation_ladder.json"),
                    help="含 hist_hv 的实验结果文件")
    ap.add_argument("--pairs", default="",
                    help="逗号分隔的臂对，形如 A=B,C=D；默认取论文阶梯的几对")
    ap.add_argument("--out", default="",
                    help="结果 JSON 路径（默认为 <labs>.anytime.json）")
    args = ap.parse_args()

    pairs = DEFAULT_PAIRS
    if args.pairs.strip():
        pairs = [tuple(s.split("=")) for s in args.pairs.split(",") if "=" in s]
        pairs = [(_canon(a.strip()), _canon(b.strip())) for a, b in pairs]

    rows = load_rows(args.labs)
    by = index_hist(rows)
    if not by:
        print("没有可用的逐代轨迹（hist_hv / hv）。")
        return 1

    instances = sorted({k[0] for k in by})
    labels_all = sorted({k[1] for k in by})
    n_gen = max(arr.size for d in by.values() for arr in d.values())

    print("═" * 78)
    print(f"anytime 分析  labs={os.path.basename(args.labs)}")
    print(f"runs={len(rows)}  实例={len(instances)}  臂={len(labels_all)}  最长轨迹={n_gen} 代")
    print(f"臂: {', '.join(labels_all)}")
    print("═" * 78)

    out = {"labs": os.path.basename(args.labs), "n_gen": n_gen,
           "segments": SEGMENTS, "auc": {}, "segments_stats": {},
           "maturity": {}, "curves": {}}

    # ── 0) 臂重复自检。必须**分两个口径**判：搜索轨迹（hist_hv）与输出存档（final_hv）。
    #       只看 hist_hv 会把「Elite archive 型组件」误判成「重复臂」——
    #       它本来就不改搜索轨迹，只在终点从历史里挑更好的非支配集。──
    final_by = index_final(rows)
    print("\n【0】臂重复自检（区分：搜索轨迹相同 / 终点输出相同）")
    print("     hist_hv = population 逐代口径（搜索过程）   final_hv = 末代 archive 口径（输出）")
    labels = sorted({k[1] for k in by})
    dup_groups = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            n = same_tr = same_fin = 0
            for inst in instances:
                A, B = by.get((inst, a), {}), by.get((inst, b), {})
                FA, FB = final_by.get((inst, a), {}), final_by.get((inst, b), {})
                for s in set(A) & set(B):
                    n += 1
                    if A[s].size == B[s].size and np.array_equal(A[s], B[s]):
                        same_tr += 1
                    if s in FA and s in FB and FA[s] == FB[s]:
                        same_fin += 1
            if n and same_tr == n:
                if same_fin == n:
                    tag = "完全重复（搜索与输出都相同）"
                else:
                    tag = "搜索轨迹相同、仅终点不同（archive 型组件）"
                dup_groups.append([a, b, n, same_fin])
                print(f"  ⚠ {a} 与 {b}: hist_hv {n}/{n} 同, final_hv {same_fin}/{n} 同 → {tag}")
    if not dup_groups:
        print("  未发现搜索轨迹完全相同的臂。")
    out["duplicate_arms"] = dup_groups

    # ── 1) 归一化 AUC：用该实例「全部臂全部 seed」的 HV 极值归一化，
    #       使不同臂的 AUC 落在同一尺度上可比 ──
    print("\n【1】归一化 AUC（同一实例内可比；越大越好）")
    for inst in instances:
        lab_seeds = {lab: by[(inst, lab)] for (i, lab) in by if i == inst}
        allv = np.concatenate([arr for d in lab_seeds.values() for arr in d.values()])
        lo, hi = float(np.min(allv)), float(np.max(allv))
        span = hi - lo
        row = {}
        for lab, d in sorted(lab_seeds.items()):
            if span <= 0:
                continue
            curve = np.mean([(arr - lo) / span for arr in d.values()], axis=0)
            row[lab] = float(np.mean(curve))
            out["curves"].setdefault(inst, {})[lab] = [float(x) for x in curve]
        out["auc"][inst] = row
        order = sorted(row.items(), key=lambda kv: -kv[1])
        print(f"  {inst}:  " + "  ".join(f"{k}={v:.4f}" for k, v in order))

    # ── 2) 分段配对检验：每段先在 run 内取均值，再跨 seed 配对 ──
    print("\n【2】分段配对检验（diff>0 表示左侧臂更好）")
    for inst in instances:
        lab_seeds = {lab: by[(inst, lab)] for (i, lab) in by if i == inst}
        for a, b in pairs:
            if a not in lab_seeds or b not in lab_seeds:
                continue
            seeds = sorted(set(lab_seeds[a]) & set(lab_seeds[b]))
            if not seeds:
                print(f"  ⚠ {inst} {a} vs {b}: seed 交集为空，跳过")
                continue
            recs = []
            for lo_g, hi_g in SEGMENTS:
                va = np.array([np.mean(lab_seeds[a][s][lo_g - 1:hi_g])
                               for s in seeds if lab_seeds[a][s].size >= hi_g])
                vb = np.array([np.mean(lab_seeds[b][s][lo_g - 1:hi_g])
                               for s in seeds if lab_seeds[b][s].size >= hi_g])
                if va.size == 0 or va.size != vb.size:
                    continue
                diff, wins, p, dz = paired_stats(va, vb)
                recs.append({"seg": [lo_g, hi_g], "n": int(va.size),
                             "mean_a": float(np.mean(va)), "mean_b": float(np.mean(vb)),
                             "diff": diff, "wins": wins, "p": p, "dz": dz})
            out["segments_stats"].setdefault(inst, {})[f"{a}|{b}"] = recs
            if recs:
                head = f"  {inst:<5} {a:>11} vs {b:<11} "
                body = "  ".join(
                    f"[{r['seg'][0]:>3}-{r['seg'][1]:<3}] {r['diff']:+.4f}"
                    f"{'*' if r['p'] < 0.05 else ' '}({r['wins']}/{r['n']})"
                    for r in recs)
                print(head + body)

    # ── 3) 早熟度：前 50 代拿到了全程增益的百分之多少 ──
    print("\n【3】早熟度 = (HV@50 − HV@1) / (HV@200 − HV@1)   （>1 表示中途回撤）")
    for inst in instances:
        lab_seeds = {lab: by[(inst, lab)] for (i, lab) in by if i == inst}
        row = {}
        for lab, d in sorted(lab_seeds.items()):
            vals = []
            for arr in d.values():
                if arr.size < 200:
                    continue
                denom = arr[-1] - arr[0]
                if abs(denom) < 1e-12:
                    continue
                vals.append((arr[49] - arr[0]) / denom)
            if vals:
                row[lab] = float(np.mean(vals))
        out["maturity"][inst] = row
        if row:
            print(f"  {inst}:  " + "  ".join(f"{k}={v:+.3f}" for k, v in sorted(row.items())))

    # ── 4) 终点口径（final_hv = 末代 archive）：论文阶梯真正读的就是这个数 ──
    print("\n【4】终点口径（final_hv）配对比较  * = p<0.05")
    print("     量级(diff)不可跨实例平均（各实例 HV 尺度不同）；符号 / wins / p 可用。")
    for a, b in pairs:
        cells, agg, n_sig = [], [], 0
        for inst in instances:
            FA, FB = final_by.get((inst, a), {}), final_by.get((inst, b), {})
            seeds = sorted(set(FA) & set(FB))
            if not seeds:
                cells.append("  n/a    ")
                continue
            va = np.array([FA[s] for s in seeds])
            vb = np.array([FB[s] for s in seeds])
            diff, _w, p, _dz = paired_stats(va, vb)
            if p < 0.05:
                n_sig += 1
            agg.append(va - vb)
            mark = "*" if p < 0.05 else " "
            cells.append("%+.4f%s" % (diff, mark))
        if not agg:
            continue
        D = np.concatenate(agg)
        _d, wins, pw, dzw = paired_stats(D, np.zeros_like(D))
        out["final_stats"] = out.get("final_stats", {})
        out["final_stats"]["%s|%s" % (a, b)] = {
            "wins": wins, "n": int(D.size), "p": pw, "dz": dzw,
            "n_instance_sig": n_sig}
        print("  %11s vs %-11s  " % (a, b) + " ".join("%-9s" % c for c in cells)
              + "  显著实例 %d/%d  wins %d/%d  p=%.2e  dz=%+.2f"
              % (n_sig, len(instances), wins, D.size, pw, dzw))

    out_path = args.out or (args.labs + ".anytime.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    print(f"\n已写出 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
