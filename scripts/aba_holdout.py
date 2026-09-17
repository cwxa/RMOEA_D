#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ABA 留出集确认：在 Mk07 / Mk09 上复核 Mk10 开发集得到的结论。

为什么单独一个脚本
------------------
`t_leverage_analysis.py --lab_json` 能逐实例出 §[5]，但它：
  1. 每个实例单独看，**不给合并口径**；
  2. 只在**单个实例的盒**内归一化，跨实例的绝对 HV 不可比；
  3. 不做"实例间方向一致性"的检查——而留出集最重要的一条就是
     "效应在不同实例上是否同号"。

本脚本把 `docs/aba-budget-allocation.md` §3 预注册的 H1 检验
在两个留出实例上跑一遍，并给出：

  A. 逐实例：等算力核对 + H1（vs pool_random）+ H1'（vs 等算力基准） + 反极性臂
  B. 合并敏感性：把每实例的配对差**除以该实例基准臂均值**化成相对差
     （消掉实例尺度），60 个配对观测上做 Wilcoxon —— 注意这只是敏感性分析，
     **主判据仍是逐实例**（预注册 §3 写的是配对 Wilcoxon, n=30）。
  C. 方向一致性：效应符号在实例间是否一致（符号检验的直觉版）。

用法
----
    python scripts/aba_holdout.py --labs logs/_mk07_lab.json,logs/_mk09_lab.json

输出：stdout + `--out`（默认 logs/_aba_holdout.txt）
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from rmoea_d.utils.metrics import compute_hv, estimate_hv_bounds  # noqa: E402

REF = (1.02, 1.02)

# 预注册（docs/aba-budget-allocation.md §3）的对照关系
BASE_EQ = "RVNSonly_t2"      # 等算力基准（统一上限 2）
CTL_HET = "RVNSonly_Brand"   # 异质性对照（随机发放加码名额）
TREAT = ["RVNSonly_Bstate", "RVNSonly_Blearn", "RVNSonly_Bhot"]
REF_ARMS = ["RVNSonly", "RVNSonly_t3"]

L = []


def W(s=""):
    L.append(str(s))
    print(s, flush=True)


def stars(p):
    if p != p:
        return "n.s."
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def wlx(a, b):
    """配对 Wilcoxon 双侧 p；全零差时返回 nan（而不是崩）。"""
    try:
        if np.allclose(np.asarray(a) - np.asarray(b), 0.0):
            return float("nan")
        return float(stats.wilcoxon(a, b)[1])
    except Exception:
        return float("nan")


def paired(hv, a, b, seeds):
    """按 seed 交集取配对。返回 (dmean, wins, n, p, dz, rel%)。"""
    ss = [s for s in seeds if s in hv[a] and s in hv[b]]
    if not ss:
        return None
    va = np.array([hv[a][s] for s in ss])
    vb = np.array([hv[b][s] for s in ss])
    d = va - vb
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    dz = float(d.mean() / sd) if sd else float("nan")
    return dict(d=float(d.mean()), wins=int((d > 0).sum()), n=len(d),
                p=wlx(va, vb), dz=dz,
                rel=float(d.mean() / vb.mean() * 100.0) if vb.mean() else float("nan"),
                dvals=d, seeds=ss,
                base_mean=float(vb.mean()) if vb.mean() else float("nan"))


def load_instance(path):
    rows = json.load(open(path, encoding="utf-8"))
    fronts = [np.asarray(r["final_pf"], float) for r in rows if r["final_pf"]]
    lo, hi = estimate_hv_bounds(fronts)
    hv = collections.defaultdict(dict)
    tt = collections.defaultdict(list)
    bagg = collections.defaultdict(list)
    for r in rows:
        hv[r["label"]][r["seed"]] = compute_hv(np.asarray(r["final_pf"], float),
                                               ref_point=REF, norm_bounds=(lo, hi))
        tt[r["label"]].append(r.get("total_time", np.nan))
        if r.get("rvns_budget"):
            bagg[r["label"]].append(r["rvns_budget"])
    seeds = sorted({r["seed"] for r in rows})
    inst = rows[0].get("instance") or os.path.basename(path)
    return dict(path=path, inst=inst, hv=hv, tt=tt, bagg=bagg, seeds=seeds,
                lo=lo, hi=hi, n_rows=len(rows))


def budget_summary(bagg):
    out = {}
    for a, v in bagg.items():
        out[a] = dict(
            planned=float(np.mean([b["mean_planned_trials"] for b in v])),
            evals=float(np.mean([b["mean_actual_evals"] for b in v])),
            upgrades=float(np.mean([b["upgrades"] for b in v])),
            decisions=float(np.mean([b["decisions"] for b in v])),
            prop=np.mean([b["propensities"] for b in v], axis=0).tolist(),
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labs", required=True, help="逗号分隔的 lab json 路径")
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "_aba_holdout.txt"))
    ap.add_argument("--report_json", default=os.path.join(ROOT, "logs", "_aba_holdout.json"))
    args = ap.parse_args()

    paths = [p.strip() for p in args.labs.split(",") if p.strip()]
    insts = [load_instance(p) for p in paths]

    W("=" * 100)
    W("ABA 留出集确认  |  " + " + ".join(d["inst"] for d in insts))
    W("预注册：docs/aba-budget-allocation.md §3  "
      "H1 = {state,learn} 显著优于 pool_random（配对 Wilcoxon, n=30）")
    W("=" * 100)

    # 合并口径：按 **配对观测**（instance, seed）收集相对差，
    # 而不是先按实例平均再合并 —— 后者只剩 n=2，任何检验都无意义。
    pooled = collections.defaultdict(list)   # (arm, other) -> [ (inst, seed, rel) ]
    raw = collections.defaultdict(dict)      # (arm, other) -> {inst: d}

    for d in insts:
        hv, seeds, inst = d["hv"], d["seeds"], d["inst"]
        arms = sorted(hv)
        W()
        W("-" * 100)
        W(f"【{inst}】  rows={d['n_rows']}  seeds={len(seeds)}  arms={len(arms)}")
        W(f"  归一化盒 lo={np.round(d['lo'], 2)}  hi={np.round(d['hi'], 2)}  ref={REF}"
          f"   （盒只在实例内可比）")
        W("-" * 100)

        W(f"  {'arm':<18}{'HV mean':<11}{'std':<9}{'median':<11}{'time(s)'}")
        for a in arms:
            v = np.array([hv[a][s] for s in seeds if s in hv[a]])
            W(f"  {a:<18}{v.mean():<11.5f}{v.std():<9.5f}{np.median(v):<11.5f}"
              f"{np.nanmean(d['tt'][a]):<8.1f}")

        bagg = budget_summary(d["bagg"])
        W()
        W("  等算力核对（预注册必报项）")
        W(f"  {'arm':<18}{'计划均值':<12}{'实际求值均值':<14}{'升级率':<11}")
        for a in [x for x in (REF_ARMS + [BASE_EQ] + TREAT) if x in hv]:
            b = bagg.get(a)
            if b and b["decisions"]:
                W(f"  {a:<18}{b['planned']:<12.4f}{b['evals']:<14.4f}"
                  f"{100.0 * b['upgrades'] / b['decisions']:<11.1f}%")
            else:
                W(f"  {a:<18}{'-':<12}{'-':<14}{'-':<11}")
        pv = [bagg[a]["planned"] for a in [BASE_EQ] + TREAT
              if a in bagg and bagg[a]["decisions"]]
        if pv:
            W(f"  -> 计划均值极差 = {max(pv) - min(pv):.6f}（应为 0）")

        W()
        W("  H1 / H1' / 反极性（配对 Wilcoxon）")
        W(f"  {'对照':<38}{'dHV':<11}{'胜负':<10}{'p':<12}{'dz':<9}{'相对%':<9}{''}")
        base_mean = np.mean([hv[BASE_EQ][s] for s in seeds if s in hv[BASE_EQ]]) \
            if BASE_EQ in hv else np.nan
        for a in TREAT:
            if a not in hv:
                continue
            for other in (CTL_HET, BASE_EQ):
                if other not in hv:
                    continue
                r = paired(hv, a, other, seeds)
                if not r:
                    continue
                W(f"  {a + ' vs ' + other:<38}{r['d']:<+11.5f}"
                  f"{'%d/%d' % (r['wins'], r['n']):<10}{r['p']:<12.4g}"
                  f"{r['dz']:<+9.3f}{r['rel']:<+9.2f}{stars(r['p'])}")
                # 逐 seed 的相对差 -> 合并样本；见文末"合并敏感性分析"
                if r["base_mean"] == r["base_mean"] and r["base_mean"]:
                    for s, dv in zip(r["seeds"], r["dvals"]):
                        pooled[(a, other)].append(
                            (inst, s, float(dv) / r["base_mean"] * 100.0))
                raw[(a, other)][inst] = r["d"]
        if CTL_HET in hv:
            W(f"  [{CTL_HET} vs {BASE_EQ}] 异质性单独贡献：", )
            r = paired(hv, CTL_HET, BASE_EQ, seeds)
            if r:
                W(f"  {CTL_HET + ' vs ' + BASE_EQ:<38}{r['d']:<+11.5f}"
                  f"{'%d/%d' % (r['wins'], r['n']):<10}{r['p']:<12.4g}"
                  f"{r['dz']:<+9.3f}{r['rel']:<+9.2f}{stars(r['p'])}")
                if r["base_mean"] == r["base_mean"] and r["base_mean"]:
                    for s, dv in zip(r["seeds"], r["dvals"]):
                        pooled[(CTL_HET, BASE_EQ)].append(
                            (inst, s, float(dv) / r["base_mean"] * 100.0))

        W()
        W("  与固定预算参照组（算力不同，仅定位，不作等算力结论）")
        for a in [BASE_EQ] + TREAT:
            for b in REF_ARMS:
                if a in hv and b in hv:
                    r = paired(hv, a, b, seeds)
                    if r:
                        W(f"  {a + ' vs ' + b:<38}{r['d']:<+11.5f}"
                          f"{'%d/%d' % (r['wins'], r['n']):<10}{r['p']:<12.4g}"
                          f"{r['dz']:<+9.3f}{r['rel']:<+9.2f}{stars(r['p'])}")

        W()
        W("  学到的升级倾向 [stuck=0=刚改进过, stuck=1=卡住]")
        for a in [BASE_EQ] + TREAT:
            if a in bagg:
                W(f"    {a:<18}{[round(x, 4) for x in bagg[a]['prop']]}")

    # ── B. 合并敏感性分析（相对差，消实例尺度）────────────────────
    W()
    W("=" * 100)
    W("合并敏感性分析（相对差 ΔHV / 该实例等算力基准均值；主判据仍是逐实例）")
    W("=" * 100)
    W(f"  {'对照':<38}{'n_配对':<8}{'合并 p':<12}{'dz':<9}{'极差':<10}{'各实例均值相对%'}")
    conv = {}
    for (a, b), lst in sorted(pooled.items()):
        rel = np.array([x[2] for x in lst], dtype=float)
        insts_seen = sorted({x[0] for x in lst})
        # 双侧 Wilcoxon on 60 paired relative deltas
        if len(rel) >= 8:
            p = float(stats.wilcoxon(rel)[1])
        else:
            p = float(stats.binomtest(int((rel > 0).sum()), len(rel), 0.5).pvalue)
        sd = rel.std(ddof=1) if len(rel) > 1 else 0.0
        dz = float(rel.mean() / sd) if sd else float("nan")
        per_inst = {i: float(np.mean([x[2] for x in lst if x[0] == i]))
                    for i in insts_seen}
        same = "是" if (np.all(rel > 0) or np.all(rel < 0)) else "否"
        W(f"  {a + ' vs ' + b:<38}{len(rel):<8}{p:<12.4g}{dz:<+9.3f}"
          f"{rel.max() - rel.min():<10.2f}{ {k: round(v, 2) for k, v in per_inst.items()} }")
        conv[f"{a} vs {b}"] = dict(n=len(rel), p=p, dz=dz, same_sign=same,
                                   per_instance_rel=per_inst)
    W()
    W("  注：'同号?'看的是 60 个配对差是否全同号，几乎必然是'否'（尾部本来就双向）；")
    W("      真正要读的是 **各实例均值相对%** 是否同号、以及合并 p 是否显著。")

    W()
    W("判定口径（预注册 §3，不因结果改）：")
    W("  * H1 成立需 **逐实例** {state,learn} 显著优于 pool_random（配对 Wilcoxon, n=30）")
    W("    且两实例同号；合并 p 只作敏感性，不作为主判据。")
    W("  * H1 不成立 -> 结论只能是「加码发给谁不重要」，并另报「随机异质性」")
    W(f"    相对等算力基准 {BASE_EQ} 是否显著。")
    W("  * 任何一实例方向与开发集相反、却不显著，都**不得**宣称可迁移。")

    open(args.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({k: v for k, v in conv.items()},
              open(args.report_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\nWROTE", args.out, "|", args.report_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
