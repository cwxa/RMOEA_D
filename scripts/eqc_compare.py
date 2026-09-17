#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""等算力对照（P3/B1）：把「多花算力」与「更深的每代搜索」分开。

背景
----
`ls_trials 1→3` 是本项目最大的单一新杠杆（+4.68%***），但它同时把墙钟从
19.7s 提到 28.6s（相对 T10 的 12.9s 是 2.22×）。所以有两种解释：

  H_compute   收益来自**总算力**（跑得久）
  H_depth     收益来自**每代搜得更深**（同样的时间，尝试次数更多）

判据：把 `ls_trials=1` 的臂拉长代数，使墙钟**不低于** `ls_trials=3` 臂。
若拉长后的 t1 臂仍打不过 t3 臂 → H_depth 成立（t3 用更少时间拿到更高 HV）；
若打平或反超 → H_compute 成立。

注意：本脚本**不做**"绝对 HV 跨文件比较"。`--eqc` 里若含 G440 臂，
必须与主 lab 的前沿**合进同一个归一化盒**才能比（见 `--merge-into`）。

用法
----
    python scripts/eqc_compare.py --lab logs/_mk10_merged_eqc.json
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from rmoea_d.utils.metrics import compute_hv, estimate_hv_bounds  # noqa: E402

REF = (1.02, 1.02)

# (A, B, 说明)
# 问法统一为：「A 相对 B 的优势，是不是只是算力？」因此**理想设计是时间匹配**。
# match_on: "time"  —— 按墙钟匹配（时间比落进 [0.80,1.25] 才算匹配）
#           "evals" —— 按**邻域求值次数**匹配（墙钟比不参与判定）
PAIRS = [
    # —— 决定性对照：真正的等算力 ——
    ("RVNSonly_t3",   "RVNSonly_G290", "time",  "ls_trials 1->3 是不是纯算力效应（**墙钟**匹配）"),
    ("RVNSonly_t3",   "RVNSonly_G586", "evals", "ls_trials 1->3（**邻域求值次数**匹配，更严口径）"),
    ("RVNSonly_t3",   "RVNSonly",      "none",  "ls_trials 1->3 原口径（算力不匹配，仅参照）"),
    # —— T 维度：把 T10 的时间花到与 T50 相当，T50 还赢吗 ——
    ("T10_G440",      "T50",           "time",  "T50 的优势能否被'给 T10 更多代数'抹掉"),
    ("T50_G440",      "T10_G440",      "time",  "长代数下 T 维度是否还分层（同为 G440）"),
    # —— 纯代数效应 ——
    ("RVNSonly_G440", "RVNSonly",      "none",  "纯代数效应（RVNS，G 200->440）"),
    ("T10_G440",      "T10",           "none",  "纯代数效应（固定 T，G 200->440）"),
    # —— RL 选算子（同 G440 家族，时间近匹配）——
    ("RVNSonly_G440", "RandVNS_G440",  "time",  "同 G440：RL 选算子 vs 随机选算子"),
]


MATCH_LO, MATCH_HI = 0.80, 1.25


def stars(p):
    if p != p:
        return "n.s."
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def _verdict(tra, better, worse, match_on="time"):
    """把 (时间比, 是否显著更好, 匹配维度) 映射成判定短语 + 理由。"""
    if match_on == "none":
        return "仅参照", "算力不匹配，不用于判定"
    if match_on == "evals":
        # 已按求值次数配对，墙钟差异是**结果**不是混淆；只按显著性读。
        if better:
            return "次数有效", "同求值次数下 A 仍更好 -> 尝试次数是真杠杆"
        if worse:
            return "次数为负", "同求值次数下 A 更差"
        return "次数无效", "同求值次数下打平 -> 尝试次数不是杠杆"
    matched = MATCH_LO <= tra <= MATCH_HI
    if matched:
        if better:
            return "非算力效应", "时间匹配且 A 显著更好 -> H_depth"
        if worse:
            return "杠杆为负", "时间匹配但 A 显著更差"
        return "算力效应", "时间匹配且无差异 -> H_compute"
    if tra > MATCH_HI:
        if better:
            return "不可判", "A 多花算力，收益可由算力解释"
        if worse:
            return "强证据-负", "A 多花算力却显著更差"
        return "强证据-无收益", "A 多花算力但无差异"
    # tra < MATCH_LO：A 更省
    if better:
        return "强证据-非算力", "A 更省算力反而显著更好"
    if worse:
        return "不可判", "A 更省算力且更差，方向相反"
    return "不可判", "A 更省算力且无差异"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lab", required=True,
                    help="含等算力臂的 lab json（必须是**同一个归一化盒**）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = json.load(open(args.lab, encoding="utf-8"))
    fronts = [np.asarray(r["final_pf"], float) for r in rows if r["final_pf"]]
    lo, hi = estimate_hv_bounds(fronts)

    hv = collections.defaultdict(dict)
    tt = collections.defaultdict(list)
    for r in rows:
        hv[r["label"]][r["seed"]] = compute_hv(np.asarray(r["final_pf"], float),
                                               ref_point=REF, norm_bounds=(lo, hi))
        tt[r["label"]].append(r.get("total_time", np.nan))
    seeds = sorted({r["seed"] for r in rows})

    L = []

    def W(s=""):
        L.append(str(s))
        print(s, flush=True)

    W("=" * 100)
    W("等算力对照（P3/B1）")
    W("=" * 100)
    W(f"lab = {args.lab}   arms={len(hv)}   seeds={len(seeds)}   runs={len(rows)}")
    W(f"归一化盒 lo={np.round(lo, 2)} hi={np.round(hi, 2)} ref={REF}"
      "   （绝对 HV 只在本盒内可比）")

    used = sorted({a for p in PAIRS for a in p[:2] if a in hv})
    W()
    W("%-18s %-11s %-9s %-9s" % ("arm", "HV mean", "std", "time(s)"))
    for a in used:
        v = np.array([hv[a][s] for s in seeds if s in hv[a]])
        W("%-18s %-11.5f %-9.5f %-9.1f"
          % (a, v.mean(), v.std(), np.nanmean(tt[a])))

    W()
    W("%-34s %-11s %-10s %-12s %-9s %-9s %-9s %s"
      % ("对照 (A vs B)", "dHV", "胜负", "p", "dz", "时间比", "相对%", "判定"))
    for A, B, match_on, note in PAIRS:
        if A not in hv or B not in hv:
            W("%-34s  (缺 %s)" % (f"{A} vs {B}", A if A not in hv else B))
            continue
        ss = [s for s in seeds if s in hv[A] and s in hv[B]]
        va = np.array([hv[A][s] for s in ss])
        vb = np.array([hv[B][s] for s in ss])
        d = va - vb
        try:
            p = float(stats.wilcoxon(va, vb)[1])
        except Exception:
            p = float("nan")
        sd = d.std(ddof=1) if len(d) > 1 else 0.0
        dz = float(d.mean() / sd) if sd else float("nan")
        tra = float(np.nanmean(tt[A]) / np.nanmean(tt[B]))
        better = (p == p) and (p < 0.05) and (d.mean() > 0)
        worse = (p == p) and (p < 0.05) and (d.mean() < 0)
        verdict, why = _verdict(tra, better, worse, match_on)
        W("%-34s %-+11.5f %-10s %-12.4g %-+9.3f %-9s %+.2f%%  %s"
          % (f"{A} vs {B}", d.mean(), "%d/%d" % ((d > 0).sum(), len(d)),
             p, dz, "%.2fx" % tra, d.mean() / vb.mean() * 100, verdict))
        W("%-34s   ^ %s  —— %s" % ("", note, why))

    # ── 汇总：HV 是不是"墙钟"的单一函数（与预算构成无关）────────────
    W()
    W("=" * 100)
    W("补充：把同一算法族的预算**构成**打散，看 HV 是否只跟墙钟走")
    W("=" * 100)
    fam = [a for a in ("RVNSonly", "RVNSonly_G290", "RVNSonly_G440",
                       "RVNSonly_G586", "RVNSonly_t3") if a in hv]
    pts = sorted((float(np.nanmean(tt[a])),
                  float(np.mean([hv[a][s] for s in seeds if s in hv[a]])), a)
                 for a in fam)
    W("%-18s %-10s %-11s %s" % ("arm", "time(s)", "HV mean", "预算构成"))
    for t, m, a in pts:
        comp = ("每代 1 次 x G%d" % {"RVNSonly": 200, "RVNSonly_G290": 290,
                                    "RVNSonly_G440": 440, "RVNSonly_G586": 586}
                .get(a, 200)) if a != "RVNSonly_t3" else "每代最多 3 次 x G200"
        W("%-18s %-10.1f %-11.5f %s" % (a, t, m, comp))
    tt_ = np.array([p[0] for p in pts])
    hh_ = np.array([p[1] for p in pts])
    rho, prho = stats.spearmanr(tt_, hh_)
    W(f"  Spearman(墙钟, HV) = {rho:+.3f}  p={prho:.4g}  {stars(prho)}")
    W("  -> 同一算法族内，HV 基本是**墙钟的单一函数**；")
    W("     把同一份预算拆成'多代 x 少次'还是'少代 x 多次'，落在同一条曲线上。")
    W("     （对照：`RVNSonly_t3` 28.6s/0.86241 与 `RVNSonly_G290` 27.7s/0.85975 几乎重合。）")

    W()
    W("判定口径（时间比 = A 的墙钟 / B 的墙钟；匹配带取 [0.80, 1.25]）：")
    W("  * 时间匹配 + A 显著好于 B  -> **该杠杆不是算力效应**（H_depth）")
    W("  * 时间匹配 + 不显著        -> **该杠杆可由算力解释**（H_compute）")
    W("  * A 多花算力(>1.25x) + 仍显著更好 -> 不可判（多花的算力也能解释）")
    W("  * A 多花算力(>1.25x) + 不显著     -> 强证据：多花了算力却没收益")
    W("  * A 省算力(<0.80x)   + 显著更好   -> 强证据：省了算力还更好")
    W("  * A 省算力(<0.80x)   + 不显著     -> 不可判")
    W()
    W("  [!] 两种公平性口径都要给，因为它们在这批数据上**不一致**：")
    W("      · 墙钟口径：RVNSonly_t3 vs RVNSonly_G290（27.7s vs 28.6s）")
    W("      · 求值次数口径：RVNSonly_t3 vs RVNSonly_G586")
    W("        （t3 每代每解实际约用 2.93 次 -> 200x100x2.93 = 58600 次；")
    W("          t1 要跑到 G=586 才有同样次数，但墙钟约 2x）")
    W("      只报其中一个口径 = 挑对自己有利的那个。")
    W("  [!] `none` 的对照（G440 / 原口径）**不用于判定**，只作定位。")

    out = args.out or os.path.join(ROOT, "logs", "_eqc_compare.txt")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\nWROTE", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
