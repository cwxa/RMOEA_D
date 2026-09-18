#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""廉价**事前**探针：初始种群的 makespan 杠杆（`I_mwr` vs `I_mix3`），零代搜索。

门控问题
--------
`aig_gating.py` 发现：OS-MWR 相对论文口径 MIX3 的**净增益**（目标 B，
`I_mwr vs I_mix3`，n=10）与"最终 makespan 杠杆"高度单调（ρ=+0.952）。
但那个杠杆是从 `final_pf`（**末代存档**）算的 —— 要算它必须先把两个臂都跑完，
**是事后量，不能当门控**。

本脚本问：把同一个量换成**只调一次 `_init_population()`**（`max_gen=0`，
一次搜索都不做）能不能保住这个判别力？如果能，门控就从"事后解释"升级为
**可执行的事前判据**（代价 ≈ 一次完整 run 的 1/G）。

口径
----
* 只读初始种群的 **crisp** 目标值 `(makespan, 总机器负载)`，不跑任何一代；
* 每个 (实例, 变体, seed) 取 `min makespan`（与 `aig_gating` 的杠杆定义同构：
  都是"最优个体"的比较，不是均值 —— 均值会被种群的多样性稀释）；
* 预测目标 `y` **直接读 `logs/aig_gating.json`**（目标 B 的 ΔHV rel%），
  不重算 —— 单一真源，避免"同一份数据两套数字"；
* 显著性一律给**置换家族错误率**（候选 = 本探针 + 6 个结构特征），
  以及留一区间。

用法：
    python scripts/init_probe.py
    python scripts/init_probe.py --instances Mk01,Mk02 --seeds 30
    python scripts/init_probe.py --aig logs/aig_gating.json --out logs/init_probe.json
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import aig_gating as aig  # noqa: E402  （复用置换/留一/星号，不另写一套）

DEFAULT_INSTANCES = ["Mk%02d" % i for i in range(1, 11)]
VARIANTS = {"mix3": "I_mix3", "mix3_mwr": "I_mwr"}


def probe_one(instance, seed, data_dir, n_pop):
    """返回 {label: (min_mc, mean_mc, min_wl, mean_wl)}，零代搜索。

    `solve()` 里实例是**懒加载**的（且 `load_instance` 吃 seed，模糊区间随 seed 变），
    所以这里必须照 `solve()` 的口径自己装一次，否则 `_init_population()` 里
    `instance` 还是 None。
    """
    from rmoea_d.algorithm import RMOEAD
    from rmoea_d.core.instance import load_instance
    from rmoea_d.utils.metrics import (compute_hv, instance_hv_bounds,
                                       non_dominated_sort)
    inst = load_instance(instance, data_dir, seed)
    bounds = instance_hv_bounds(inst)
    out = {}
    for variant, label in VARIANTS.items():
        solver = RMOEAD(instance_name=instance, n_pop=n_pop, max_gen=0,
                        seed=seed, data_dir=data_dir, init_variant=variant)
        solver.instance = inst
        _pop, objs = solver._init_population()
        mc = np.array([o[0] for o in objs], float)
        wl = np.array([o[1] for o in objs], float)
        # 初始前沿（第 0 代、无任何搜索）的 HV —— 与最终指标同一泛函、同一实例边界口径
        nd = non_dominated_sort([(float(a), float(b)) for a, b in zip(mc, wl)])
        hv0 = float(compute_hv(np.asarray(nd, float), ref_point=(1.02, 1.02),
                               norm_bounds=bounds))
        out[label] = (float(mc.min()), float(mc.mean()),
                      float(wl.min()), float(wl.mean()), hv0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", default=",".join(DEFAULT_INSTANCES))
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed_start", type=int, default=42)
    ap.add_argument("--n_pop", type=int, default=100)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--aig", default=os.path.join(ROOT, "logs", "aig_gating.json"),
                    help="预测目标来源（目标 B 的逐实例 ΔHV rel%）")
    ap.add_argument("--target", default="vs_I_mix3",
                    help="aig_gating.json 里 targets 的键（默认目标 B）")
    ap.add_argument("--perm", type=int, default=20000)
    ap.add_argument("--perm_seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "init_probe.json"))
    args = ap.parse_args()

    instances = [s.strip() for s in args.instances.split(",") if s.strip()]
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    print("=" * 96)
    print("廉价事前探针：初始种群 makespan 杠杆（%s vs %s），零代搜索"
          % (VARIANTS["mix3_mwr"], VARIANTS["mix3"]))
    print("=" * 96)
    print("实例 %d 个 × %d seeds × %d 变体，n_pop=%d，max_gen=0"
          % (len(instances), len(seeds), len(VARIANTS), args.n_pop))
    print("-" * 96)

    t0 = time.perf_counter()
    rows = []
    for inst in instances:
        acc = collections.defaultdict(lambda: collections.defaultdict(list))
        for s in seeds:
            got = probe_one(inst, s, args.data_dir, args.n_pop)
            for lab, (mn, mu, wmn, wmu, hv0) in got.items():
                acc[lab]["min_mc"].append(mn)
                acc[lab]["mean_mc"].append(mu)
                acc[lab]["min_wl"].append(wmn)
                acc[lab]["hv0"].append(hv0)
        a, b = acc["I_mix3"], acc["I_mwr"]
        base_mc = float(np.mean(a["min_mc"]))
        base_mu = float(np.mean(a["mean_mc"]))
        base_hv = float(np.mean(a["hv0"]))
        rows.append({
            "instance": inst, "n_seed": len(seeds),
            "init_min_mc_mix3": base_mc,
            "init_min_mc_mwr": float(np.mean(b["min_mc"])),
            # ★ 探针量：初始最优 makespan 杠杆（正 = MWR 的起点更好）
            "probe_min_mc_lever_pct": (base_mc - float(np.mean(b["min_mc"]))) / base_mc * 100.0,
            "probe_mean_mc_lever_pct": (base_mu - float(np.mean(b["mean_mc"]))) / base_mu * 100.0,
            "init_min_wl_mix3": float(np.mean(a["min_wl"])),
            "init_min_wl_mwr": float(np.mean(b["min_wl"])),
            "probe_min_wl_lever_pct": ((float(np.mean(a["min_wl"])) - float(np.mean(b["min_wl"])))
                                       / float(np.mean(a["min_wl"])) * 100.0),
            # ★ 最直接的探针量：初始前沿 HV 杠杆（与最终指标同泛函、同实例边界口径）
            "init_hv0_mix3": base_hv,
            "init_hv0_mwr": float(np.mean(b["hv0"])),
            "probe_init_hv_lever_pct": (float(np.mean(b["hv0"])) - base_hv) / base_hv * 100.0,
        })
        r = rows[-1]
        print("  %-6s 初始最优 makespan %8.1f → %8.1f  (%+7.2f%%)   初始前沿 HV %.5f → %.5f (%+6.2f%%)"
              % (inst, r["init_min_mc_mix3"], r["init_min_mc_mwr"],
                 r["probe_min_mc_lever_pct"],
                 r["init_hv0_mix3"], r["init_hv0_mwr"],
                 r["probe_init_hv_lever_pct"]), flush=True)
    print("  探测耗时 %.1fs（零代搜索）" % (time.perf_counter() - t0))

    # ── 对齐预测目标（单一真源：aig_gating.json 的目标 B）──
    payload = {"instances": instances, "seeds": len(seeds), "n_pop": args.n_pop,
               "probe": rows, "aig": os.path.basename(args.aig),
               "target": args.target}
    if not os.path.exists(args.aig):
        print("\n[!] 找不到 %s，只写出探针原始结果（不做相关分析）" % args.aig)
    else:
        with open(args.aig, encoding="utf-8") as fh:
            aigj = json.load(fh)
        tgt = aigj.get("targets", {}).get(args.target)
        if not tgt:
            print("\n[!] %s 里没有 targets['%s']，跳过相关分析" % (args.aig, args.target))
            tgt = None
        else:
            ymap = {r["instance"]: r["dhv_rel_pct"] for r in tgt["per_instance"]}
            feat = {r["instance"]: r for r in tgt["per_instance"]}
            use = [r for r in rows if r["instance"] in ymap]
            miss = [r["instance"] for r in rows if r["instance"] not in ymap]
            print("\n" + "=" * 96)
            print("探针 vs 目标 B（%s 的 %s）：净增量 ΔHV rel%%" % (args.target, os.path.basename(args.aig)))
            print("=" * 96)
            if miss:
                print("  [!] 缺预测目标的实例已跳过：%s" % ", ".join(miss))
            y = np.array([ymap[r["instance"]] for r in use], float)
            keys = ["probe_init_hv_lever_pct", "probe_min_mc_lever_pct",
                    "probe_mean_mc_lever_pct", "probe_min_wl_lever_pct"] + aig.EX_ANTE
            X = np.column_stack([[r.get(k, float("nan")) if k.startswith("probe")
                                  else feat[r["instance"]][k] for r in use]
                                 for k in keys])
            # 常量列必须从相关/置换家族里剔除：`probe_min_wl_lever_pct` 恒为 0
            # （两个变体的 MA 规则相同 → 该维度零信息），留着会把"候选数"报大、
            # 并给 Spearman 喂 nan。
            keep = [j for j in range(len(keys))
                    if not np.allclose(X[:, j], X[0, j])]
            dropped = [keys[j] for j in range(len(keys)) if j not in keep]
            keys = [keys[j] for j in keep]
            X = X[:, keep]
            if dropped:
                print("  [i] 常量列已剔除（零信息）：%s" % ", ".join(dropped))
            print("  %-12s %-8s %-12s %-9s %s"
                  % ("inst", "ops", "探针 HV 杠杆%", "ΔHV rel%", "p(配对)"))
            for r in use:
                f = feat[r["instance"]]
                print("  %-12s %-8d %+12.3f %+9.2f %s"
                      % (r["instance"], f["total_ops"],
                         r["probe_init_hv_lever_pct"],
                         ymap[r["instance"]], aig.iva.stars(f["p"])))

            print("\n  秩相关（n=%d）：" % len(use))
            corr = {}
            for j, k in enumerate(keys):
                ok = ~np.isnan(X[:, j])
                if ok.sum() < 4 or np.allclose(X[ok, j], X[ok, j][0]):
                    continue
                rho, p = stats.spearmanr(X[ok, j], y[ok])
                corr[k] = {"spearman_rho": float(rho), "spearman_p": float(p),
                           "n": int(ok.sum()),
                           "ex_ante": not k.startswith(("ms_lever", "wl_lever"))}
                tag = "  <- 本探针（事前）" if k.startswith("probe") else \
                      ("  <- 事前结构特征" if k in aig.EX_ANTE else "  (事后)")
                print("    %-26s rho=%+.3f  p=%.4f  (n=%d)%s"
                      % (k, rho, p, ok.sum(), tag))

            print("\n  ★ 置换家族错误率（候选 %d 个：探针量 + 结构特征）" % len(keys))
            res = aig.permutation_max_rho(X, y, args.perm, seed=args.perm_seed)
            res["keys"] = keys
            res["obs_argmax_key"] = keys[res["obs_argmax"]]
            print("    观测 max|rho| = %+.3f (%s)"
                  % (res["obs_max_abs_rho"], res["obs_argmax_key"]))
            print("    零假设: 均值 %.3f  95%% %.3f  99%% %.3f"
                  % (res["null_mean"], res["null_p95"], res["null_p99"]))
            print("    -> 校正 p = %.4f  %s"
                  % (res["mc_p_fwer"], aig.iva.stars(res["mc_p_fwer"])))

            loo = aig.leave_one_out(X, y, keys)
            print("\n  留一稳健性（按 |rho| 降序取前 4）")
            for k in sorted(keys, key=lambda k: -abs(loo[k]["rho_all"]))[:4]:
                d = loo[k]
                print("    %-26s rho(全)=%+.3f  留一区间 [%+.3f, %+.3f]  最不利: %s"
                      % (k, d["rho_all"], d["min_rho"], d["max_rho"],
                         [r["instance"] for r in use][d["worst_drop_idx"]]))
            payload.update({"correlations": corr, "permutation": res,
                            "leave_one_out": loo,
                            "y": {"key": args.target,
                                  "values": {r["instance"]: ymap[r["instance"]] for r in use}}})

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print("\n已写出 %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
