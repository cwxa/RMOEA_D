#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""热路径优化基准：同一算法配置下的真实墙钟（多次取最优）。

为什么要单独一个脚本
--------------------
优化前后的对照必须是**可复现的、落盘的**数字，而不是聊天记录里的一次性输出。
本脚本在两个代码树上各跑一次（旧版从 git worktree 检出），输出 JSON 供
`scripts/optimization_plots.py` 画图，也供文档引用。

配置固定为 Mk10 / n_pop=100 / fixed_T=10 / RVNS(ls_trials=1)，与
`logs/_mk10_lab.json` 里 `RVNSonly` 臂的配置一致 —— 即"论文主体配置"。

用法
----
    # 在优化后的树上
    python scripts/bench_speedup.py --label new --out logs/_speedup_new.json
    # 在旧代码树上（git worktree）
    python scripts/bench_speedup.py --label base --out logs/_speedup_base.json
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def run_once(instance, n_pop, max_gen, seed, ls_trials):
    from rmoea_d.algorithm import RMOEAD
    s = RMOEAD(instance_name=instance, n_pop=n_pop, max_gen=max_gen, seed=seed,
               enable_rvns=True, fixed_T=10, rvns_ls_trials=ls_trials)
    return s.solve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="Mk10")
    ap.add_argument("--n_pop", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ls_trials", type=int, default=1)
    ap.add_argument("--gens", default="60,200")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gens = [int(x) for x in args.gens.split(",")]
    out = {"label": args.label, "instance": args.instance, "n_pop": args.n_pop,
           "seed": args.seed, "ls_trials": args.ls_trials,
           "python": platform.python_version(), "numpy": np.__version__,
           "wall": {}, "hv": {}}

    for g in gens:
        res = run_once(args.instance, args.n_pop, g, args.seed, args.ls_trials)
        out["hv"][str(g)] = res["final_hv"]
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            run_once(args.instance, args.n_pop, g, args.seed, args.ls_trials)
            ts.append(time.perf_counter() - t0)
        out["wall"][str(g)] = ts
        print("[%s] G=%-4d best=%.3fs  all=%s  HV=%.10f"
              % (args.label, g, min(ts), ["%.2f" % x for x in ts],
                 out["hv"][str(g)]), flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("-> written %s" % args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
