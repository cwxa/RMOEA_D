#!/usr/bin/env python3
"""邻域大小 T 杠杆 / Q-PAS 机理实验的分析（参考集归一化 HV + 配对显著性）。

输入：``scripts/t_leverage_sweep.py`` 产出的 JSON。

    python scripts/t_leverage_analysis.py --lab_json logs/_mk10_lab.json

输出：
  [1] 固定 T 的杠杆检验（Friedman + 逐 T 对照 + per-seed oracle）
  [2] Q-PAS 各设定 vs T=10 / vs 最优固定 T / 距离 oracle 的差距
  [3] Full（Q-PAS+RVNS）与各组件的配对比较
  [4] Q-PAS 实际学到的 T 使用分布

HV 一律用**参考集归一化**重算：所有臂、所有 run 的前沿并集作为共用归一化盒，
ref=(1.02,1.02) —— 与 ``scripts/ablation_analysis.py`` 口径一致。
（不要直接读 JSON 里的 final_hv，那是单 run 的实例边界口径。）
"""
import argparse
import collections
import json
import os
import re
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from rmoea_d.utils.metrics import compute_hv, estimate_hv_bounds  # noqa: E402

REF = (1.02, 1.02)


def stars(p):
    return ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s.")


def wlx(a, b=None):
    try:
        return float(stats.wilcoxon(a, b)[1])
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lab_json", required=True)
    args = ap.parse_args()

    rows = json.load(open(args.lab_json, encoding="utf-8"))
    if not rows:
        print("[错误] 结果为空:", args.lab_json)
        return 1

    fronts = [np.asarray(r["final_pf"], float) for r in rows if r["final_pf"]]
    lo, hi = estimate_hv_bounds(fronts)

    hv = collections.defaultdict(dict)
    tdist = collections.defaultdict(collections.Counter)
    qtabs = collections.defaultdict(list)
    tt = collections.defaultdict(list)
    for r in rows:
        hv[r["label"]][r["seed"]] = compute_hv(np.asarray(r["final_pf"], float),
                                               ref_point=REF, norm_bounds=(lo, hi))
        tt[r["label"]].append(r.get("total_time", np.nan))
        for k, c in (r.get("hist_T") or {}).items():
            tdist[r["label"]][int(k)] += c
        if r.get("q_table"):
            qtabs[r["label"]].append(r["q_table"])

    seeds = sorted({r["seed"] for r in rows})
    arms = sorted(hv, key=lambda a: (not re.fullmatch(r"T\d+", a), a))
    inst = rows[0].get("instance") or ""
    if not inst:
        # 兼容早期结果 JSON（无 instance 字段）——从文件名推断，如 _mk10_lab.json
        m = re.search(r"_([A-Za-z]+\d+)_lab", os.path.basename(args.lab_json))
        inst = m.group(1).upper() if m else "?"

    A = lambda a: np.array([hv[a][s] for s in seeds if s in hv[a]])
    T_arms = sorted([a for a in arms if re.fullmatch(r"T\d+", a)],
                    key=lambda x: int(x[1:]))
    Q = [a for a in arms if a.startswith("QPAS")]
    F = [a for a in arms if a.startswith("Full")]

    print("=" * 100)
    print(f"T-leverage / Q-PAS lab  |  {inst}  |  arms={len(arms)}  seeds={len(seeds)}  "
          f"runs={len(rows)}")
    print(f"reference-set normalization  lo={np.round(lo, 2)}  hi={np.round(hi, 2)}  ref={REF}")
    print("=" * 100)
    print(f"{'arm':<18}{'HV mean':<11}{'std':<9}{'median':<11}{'min':<10}{'max':<10}{'time(s)'}")
    print("-" * 100)
    for a in arms:
        v = A(a)
        print(f"{a:<18}{v.mean():<11.5f}{v.std():<9.5f}{np.median(v):<11.5f}"
              f"{v.min():<10.5f}{v.max():<10.5f}{np.nanmean(tt[a]):<8.1f}")

    oracle = None
    best_fixed = None
    if T_arms:
        mat = np.array([A(a) for a in T_arms])
        means = mat.mean(axis=1)
        best_fixed = T_arms[int(means.argmax())]
        oracle = mat.max(axis=0)
        st, p = stats.friedmanchisquare(*mat)
        base = A(T_arms[0]) if "T10" not in T_arms else A("T10")

        print("\n" + "=" * 100)
        print("[1] 固定 T 的杠杆检验（纯 T 效应）")
        print("=" * 100)
        print(f"Friedman over {T_arms}: chi2={st:.4f}  p={p:.6g} {stars(p)}")
        print(f"HV span (best-worst of means) = {means.max() - means.min():.5f}  "
              f"({best_fixed} {means.max():.5f} vs "
              f"{T_arms[int(means.argmin())]} {means.min():.5f})   "
              f"relative = {(means.max() - means.min()) / means.mean() * 100:.2f}%")
        ref_name = "T10" if "T10" in T_arms else T_arms[0]
        print(f"\n以 {ref_name} 为基准：")
        print(f"{'arm':<10}{'dHV':<13}{'wins':<10}{'p':<12}")
        for a in T_arms:
            v = A(a)
            d = v - A(ref_name)
            pv = wlx(v, A(ref_name))
            print(f"{a:<10}{d.mean():<+13.5f}{int((d > 0).sum()):>2}/{len(d):<7}"
                  f"{pv:<12.4g}{stars(pv)}")
        print(f"\nper-seed oracle (每个 seed 取最优 T) mean={oracle.mean():.5f}")
        for a in T_arms:
            print(f"   gain if fixed {a:<6} -> oracle: {oracle.mean() - A(a).mean():+.5f}")

    if Q:
        print("\n" + "=" * 100)
        print("[2] Q-PAS 各设定（关闭 RVNS）")
        print("=" * 100)
        ref_name = "T10" if "T10" in hv else None
        head = f"{'arm':<18}{'HV':<10}"
        if ref_name:
            head += f"{'vs ' + ref_name:<22}"
        if best_fixed:
            head += f"{'vs ' + best_fixed:<22}"
        if oracle is not None:
            head += "gap-oracle"
        print(head)
        for a in Q:
            v = A(a)
            line = f"{a:<18}{v.mean():<10.5f}"
            if ref_name:
                pv = wlx(v, A(ref_name))
                line += f"{v.mean() - A(ref_name).mean():+.5f} {stars(pv):<13}"
            if best_fixed:
                pbf = wlx(v, A(best_fixed))
                line += f"{v.mean() - A(best_fixed).mean():+.5f} {stars(pbf):<13}"
            if oracle is not None:
                line += f"{v.mean() - oracle.mean():+.5f}"
            print(line)
        pool = Q + ([ref_name] if ref_name else [])
        if len(pool) > 2:
            st2, p2 = stats.friedmanchisquare(*[A(a) for a in pool])
            print(f"\nFriedman over {pool}: chi2={st2:.4f} p={p2:.6g} {stars(p2)}")

        print("\n  Q-PAS 实际学到的 T 使用分布（占各 run 代数比例）:")
        for a in Q:
            tot = sum(tdist[a].values())
            if tot:
                dist = {k: f"{v / tot * 100:.1f}%" for k, v in sorted(tdist[a].items())}
                print(f"    {a:<18} {dist}")
        print("\n  Q-table（每个臂的末次 run，行 = state）:")
        for a in Q:
            if qtabs[a]:
                q = np.array(qtabs[a][-1])
                acts = sorted(tdist[a]) or list(range(q.shape[1]))
                print(f"    {a:<18} argmax/state={[int(x) for x in q.argmax(axis=1)]}  "
                      f"qmax={q.max():.3f}  T-grid={acts}")

    if F:
        print("\n" + "=" * 100)
        print("[3] Full（Q-PAS + RVNS）")
        print("=" * 100)
        cmp_arms = ([best_fixed] if best_fixed else []) + \
                   (["T10"] if "T10" in hv else []) + Q
        for fname in F:
            fh = A(fname)
            print(f"  {fname}: HV={fh.mean():.5f}")
            for a in cmp_arms:
                if a == fname:
                    continue
                d = fh - A(a)
                pv = wlx(fh, A(a))
                print(f"     vs {a:<18} dHV={d.mean():+.5f}  "
                      f"wins={int((d > 0).sum()):>2}/{len(d)}  p={pv:<9.4g} {stars(pv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
