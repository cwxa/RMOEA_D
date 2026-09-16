#!/usr/bin/env python3
"""消融实验结果分析：参考集归一化 HV + 配对显著性 + 2×2 因子分解。

用法:
    python scripts/ablation_analysis.py --results_dir results/_fix_mk01 --instance mk01
    python scripts/ablation_analysis.py --results_dir results/experiment --instance mk01

说明:
  * HV 一律用「参考集归一化」重算——以本目录所有变体、所有 run 的前沿并集作为
    共用归一化盒，参考点 (1.02, 1.02)。这样同一次分析内的 HV 严格可比。
    （不要直接读 JSON 里的 final_hv：那可能是旧口径或别的归一化盒。）
  * 2×2 因子分解把 rmoea_d / qpas_only / rvns_only / moea_d 视作
    Q-PAS ∈ {on,off} × RVNS ∈ {on,off} 的析因实验，分别给出两个主效应与交互效应。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from rmoea_d.utils.metrics import compute_hv, estimate_hv_bounds  # noqa: E402

VARIANTS = ["rmoea_d", "qpas_only", "rvns_only", "moea_d"]
LABEL = {"rmoea_d": "Full(Q-PAS+RVNS)", "qpas_only": "Q-PAS only",
         "rvns_only": "RVNS only", "moea_d": "baseline"}
REF = (1.02, 1.02)


def _stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def load_fronts(results_dir, instance):
    pattern = os.path.join(results_dir, "experiment", instance, "run_*.json")
    if not glob.glob(pattern):
        pattern = os.path.join(results_dir, instance, "run_*.json")
    runs = [json.load(open(fp, encoding="utf-8")) for fp in sorted(glob.glob(pattern))]
    fronts = {v: [] for v in VARIANTS}
    times = {v: [] for v in VARIANTS}
    for d in runs:
        for v in VARIANTS:
            r = d.get(v)
            if not r or not r.get("final_pf"):
                continue
            fronts[v].append(np.array([[p["Makespan"], p["Workload"]]
                                       for p in r["final_pf"]], dtype=float))
            times[v].append(r.get("total_time", np.nan))
    return runs, fronts, times, pattern


def main():
    ap = argparse.ArgumentParser(description="消融实验分析（参考集归一化 HV）")
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--instance", required=True, help="如 mk01 / Mk01")
    args = ap.parse_args()

    inst = args.instance
    runs, F, T, pattern = load_fronts(args.results_dir, inst)
    if not runs:
        print(f"[错误] 未找到结果: {pattern}")
        return 1

    lo, hi = estimate_hv_bounds([a for v in VARIANTS for a in F[v]])
    hv = {v: np.array([compute_hv(a, ref_point=REF, norm_bounds=(lo, hi)) for a in F[v]])
          for v in VARIANTS}

    print("=" * 96)
    print(f"消融分析 | {inst.upper()} | n={len(runs)} | {pattern}")
    print(f"参考集归一化 lo={lo.round(2)} hi={hi.round(2)} 参考点={REF}")
    print("=" * 96)
    print(f"{'variant':<20}{'HV mean±std':<26}{'median':<11}{'bestMS':<10}{'time(s)'}")
    print("-" * 96)
    for v in VARIANTS:
        if not F[v]:
            continue
        print(f"{LABEL[v]:<20}{hv[v].mean():.5f}±{hv[v].std():<17.5f}"
              f"{np.median(hv[v]):<11.5f}"
              f"{np.mean([a[:, 0].min() for a in F[v]]):<10.2f}"
              f"{np.nanmean(T[v]):<8.1f}")

    mat = np.array([hv[v] for v in VARIANTS if F[v]])
    st, p = stats.friedmanchisquare(*mat)
    ranks = np.mean([stats.rankdata(-mat[:, i]) for i in range(mat.shape[1])], axis=0)
    print("-" * 96)
    print(f"Friedman chi2={st:.4f} p={p:.6g}  " +
          "  ".join(f"{v}={r:.2f}" for v, r in zip(VARIANTS, ranks)) + "  (rank1=best)")

    print("-" * 96)
    for a, b in [("rmoea_d", "moea_d"), ("rvns_only", "moea_d"), ("qpas_only", "moea_d"),
                 ("rmoea_d", "rvns_only"), ("rmoea_d", "qpas_only"),
                 ("qpas_only", "rvns_only")]:
        d = hv[a] - hv[b]
        try:
            _, pv = stats.wilcoxon(hv[a], hv[b])
        except Exception:
            pv = np.nan
        print(f"  {a:<10} vs {b:<10} ΔHV={d.mean():+.5f} "
              f"wins={int((d > 0).sum()):>2}/{len(d)}  p={pv:<9.4g} {_stars(pv)}")

    print("-" * 96)
    print("2×2 因子分解（Q-PAS × RVNS）")
    A, B_, C, D = hv["rmoea_d"], hv["rvns_only"], hv["qpas_only"], hv["moea_d"]
    for nm, x in [("Q-PAS 主效应", ((A - B_) + (C - D)) / 2),
                  ("RVNS 主效应", ((A - C) + (B_ - D)) / 2),
                  ("交互 Q-PAS×RVNS", (A - B_) - (C - D))]:
        try:
            _, pv = stats.wilcoxon(x)
        except Exception:
            pv = np.nan
        print(f"  {nm:<20} 效应={x.mean():+.5f}  "
              f"胜={int((x > 0).sum()):>2}/{len(x)}  p={pv:<9.4g} {_stars(pv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
