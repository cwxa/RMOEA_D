# -*- coding: utf-8 -*-
"""自适应初始化门控（AIG）：初始化的收益能不能**事前**预测？

背景
----
派工式 OS 初始化 `I_mwr`（`init_variant="mix3_mwr"`）在 Mk10 开发集上
**+8.64%（30/30，p=1.9e-09，dz=+2.04）**，但在留出集 Mk07 / Mk09 上只剩
+0.34% / +0.30%（n.s.）。所以「要不要用 MWR」不能靠固定开关。

**门控判据必须事前可得** —— 只能用实例自身的结构特征，或用一次固定预算的廉价探针。
不能用「已经知道 MWR 效果好」这种事后信息，那是循环论证。

本脚本回答两问：
  Q1 机制量：初始化的 makespan 杠杆 λ 与 ΔHV 是否单调（Spearman）？
  Q2 可预测性：实例结构特征（规模 / 柔性度 / 加工时间离散度）能否预测 ΔHV？

口径
----
* **逐实例独立**，不跨实例合并（各实例 HV 归一化盒的尺度不同）；
* HV 重算时盒**只由参与比较的两个臂构造** —— 直接复用
  `init_variant_analysis.split_context / hv_table / paired`，保证与既有结论同源，
  避免又出现「同一份数据两套数字」；
* 结论只用同盒内的相对差与**秩相关**（10 个实例的样本量很小，ρ 与 p 都要报）。

用法：
    python scripts/aig_gating.py
    python scripts/aig_gating.py --labs logs/_mk01_init.json,logs/_mk10_init.json
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

import init_variant_analysis as iva  # noqa: E402

ARMS = ["I_rand", "I_mwr"]


def instance_features(name, data_dir):
    """与算法无关的实例结构特征（全部事前可得）。"""
    from rmoea_d.core.instance import load_instance
    inst = load_instance(name, data_dir)
    n_ma = inst["n_machines"]
    flex = np.mean([len(vm) for job in inst["valid_machines"] for vm in job])
    # crisp_times[job][op] 是"该工序在各可选机器上的加工时间"，可能含 None（该机器不可用）
    times = [t for job in inst["crisp_times"] for op in job for t in op if t is not None]
    tmean = float(np.mean(times)) if times else float("nan")
    tcv = float(np.std(times) / tmean) if times and tmean else float("nan")
    return {
        "n_jobs": int(inst["n_jobs"]),
        "n_machines": int(n_ma),
        "total_ops": int(inst["total_ops"]),
        "flex_ratio": float(flex / n_ma),                 # 平均可选机器数 / 机器总数
        "pt_cv": tcv,                                     # 加工时间离散度
        "load_ratio": float(inst["total_ops"] / n_ma),    # 每台机器平均工序数
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labs", default="", help="逗号分隔；默认 Mk01~Mk10 的 _mkNN_init.json")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "aig_gating.json"))
    args = ap.parse_args()

    labs = ([s.strip() for s in args.labs.split(",") if s.strip()] if args.labs
            else [os.path.join(ROOT, "logs", "_mk%02d_init.json" % i) for i in range(1, 11)])
    labs = [p for p in labs if os.path.exists(p)]
    if not labs:
        print("没有找到任何 _mkNN_init.json")
        return 1

    per, _rows = iva.load_by_instance(labs)
    print("=" * 96)
    print("AIG 门控可预测性检验   实例 %d 个：%s" % (len(per), ", ".join(per)))
    print("=" * 96)
    print("%-6s %-6s %-5s %-6s %-7s %-9s %-9s %-9s %-8s %-8s" %
          ("inst", "n_job", "n_ma", "flex", "pt_cv",
           "ms_lever%", "dHV_rel%", "放大×", "wins", "p"))
    print("-" * 96)

    recs = []
    for inst, rs in per.items():
        kept, dropped = iva.split_context(rs, keep=ARMS)
        if dropped:
            print("  [i] %s: 盒外臂已排除 %s" % (inst, dict(dropped)))
        # 主口径 = 实例边界（lab 里的 final_hv，边界由实例数据定，**与臂集无关**）
        hv_inst = collections.defaultdict(dict)
        for r in kept:
            if r.get("final_hv") is not None:
                hv_inst[r["label"]][r["seed"]] = float(r["final_hv"])
        # 对照口径 = 盒（边界取参与臂的极值，随臂集变化）—— 只用来量化放大倍数
        hv_box, box = iva.hv_table(kept)
        pr = iva.paired(hv_inst, "I_mwr", "I_rand")
        pr_box = iva.paired(hv_box, "I_mwr", "I_rand")
        if pr is None:
            print("  ⚠ %s seed 交集不足，跳过" % inst)
            continue

        # makespan 杠杆 λ：用原始目标值（与 HV 盒无关，可跨实例比对）
        seeds = sorted(set(hv_inst["I_rand"]) & set(hv_inst["I_mwr"]))
        ms_rand, ms_mwr, wl_rand, wl_mwr = [], [], [], []
        for s in seeds:
            rr = next(r for r in kept if r["label"] == "I_rand" and r["seed"] == s)
            rm = next(r for r in kept if r["label"] == "I_mwr" and r["seed"] == s)
            ms_rand.append(min(p[0] for p in rr["final_pf"]))
            ms_mwr.append(min(p[0] for p in rm["final_pf"]))
            wl_rand.append(min(p[1] for p in rr["final_pf"]))
            wl_mwr.append(min(p[1] for p in rm["final_pf"]))
        lam_ms = (np.mean(ms_rand) - np.mean(ms_mwr)) / np.mean(ms_rand) * 100.0
        lam_wl = (np.mean(wl_rand) - np.mean(wl_mwr)) / np.mean(wl_rand) * 100.0

        try:
            feat = instance_features(inst, args.data_dir)
        except Exception as e:                                    # noqa: BLE001
            print("  ⚠ %s 特征提取失败: %s" % (inst, e))
            feat = {}

        amp = ((pr_box["rel"] / pr["rel"])
               if pr_box and abs(pr["rel"]) > 1e-12 else float("nan"))
        rec = {"instance": inst, "n": pr["n"],
               "ms_lever_pct": float(lam_ms), "wl_lever_pct": float(lam_wl),
               "dhv_rel_pct": float(pr["rel"]), "dhv": float(pr["d"]),
               "wins": pr["wins"], "p": float(pr["p"]), "dz": float(pr["dz"]),
               "hv_rand": float(pr["mean_b"]), "hv_mwr": float(pr["mean_a"]),
               "dhv_rel_box_pct": (float(pr_box["rel"]) if pr_box else None),
               "amplification": (float(amp) if np.isfinite(amp) else None),
               "box": [list(map(float, box[0])), list(map(float, box[1]))]}
        rec.update(feat)
        recs.append(rec)
        print("%-6s %-6s %-5s %-6.3f %-7.3f %-9.2f %-9.2f %-9s %-8s %-8s" %
              (inst, feat.get("n_jobs", "-"), feat.get("n_machines", "-"),
               feat.get("flex_ratio", float("nan")), feat.get("pt_cv", float("nan")),
               lam_ms, pr["rel"], ("%.1f" % amp) if np.isfinite(amp) else "n/a",
               "%d/%d" % (pr["wins"], pr["n"]), iva.stars(pr["p"])))

    # ── 秩相关：哪些量能预测 ΔHV ──
    print("\n" + "=" * 96)
    print("秩相关（n=%d 个实例）：预测目标 = dHV_rel%%（MWR 相对随机的 HV 提升）" % len(recs))
    print("=" * 96)
    keys = ["ms_lever_pct", "wl_lever_pct", "n_jobs", "n_machines",
            "total_ops", "flex_ratio", "pt_cv", "load_ratio"]
    corr = {}
    y = np.array([r["dhv_rel_pct"] for r in recs])
    for k in keys:
        x = np.array([r.get(k, np.nan) for r in recs], dtype=float)
        ok = ~np.isnan(x)
        if ok.sum() < 4 or np.allclose(x[ok], x[ok][0]):
            continue
        rho, p = stats.spearmanr(x[ok], y[ok])
        pear, _ = stats.pearsonr(x[ok], y[ok])
        corr[k] = {"spearman_rho": float(rho), "spearman_p": float(p),
                   "pearson_r": float(pear), "n": int(ok.sum())}
        print("  %-14s rho=%+.3f  p=%.4f   (pearson r=%+.3f, n=%d)"
              % (k, rho, p, pear, ok.sum()))

    payload = {"labs": [os.path.basename(p) for p in labs], "arms": ARMS,
               "per_instance": recs, "correlations": corr}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print("\n已写出 %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
