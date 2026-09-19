#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""零代探针的 seed 方差：门控到底需要几个 seed？

问题
----
`init_probe.py` 用 **30 seeds** 的初始种群均值算 $\\lambda_0$，10 实例只要 14 s，
已经是"零代"级别的廉价。但 30 个 seed 仍是 30 次 `_init_population()`。
如果门控在 3 个 seed 上就能稳定判定，代价还能再降一个数量级。

更关键的是**这个量稳不稳**：开发集上两类实例的 $\\lambda_0$ 距离门限
（$-1.0\\%$）只有约 0.5 个百分点（$\\ge -0.53\\%$ vs $\\le -1.06\\%$），
所以"实例落在门限哪一侧"完全可能被 seed 噪声翻掉。

做法
----
对每个实例，先算出 30 个 seed 的**逐 seed** $\\lambda_0^{(s)}$，再对
$n \\in \\{1,2,3,4\\}$ 枚举所有 $\\binom{30}{n}$ 个子集、对 $n \\in \\{5,10,15,20,30\\}$
随机抽 `--n_sub` 个子集，用**与 `init_probe.py` 完全相同的估计量**
（$\\lambda_0(n) = (\\overline{\\mathrm{HV}}^{(\\mathrm{mwr})}_{0} -
\\overline{\\mathrm{HV}}^{(\\mathrm{mix3})}_{0}) / \\overline{\\mathrm{HV}}^{(\\mathrm{mix3})}_{0}$，
即"先对 seed 求均值再作比"，不是"先作比再求均值"）报告：

* $\\mathrm{SD}_n$：$\\lambda_0(n)$ 在子集之间的标准差；
* agree$_n$：与该实例 **30-seed 判定**一致（是否 $\\ge$ 门限）的子集比例。

口径
----
判定真值 = 该实例自己的 30-seed 判定，**不是**该实例的真实净增益
（后者在留出集上未必已知）。这里问的是"少采几个 seed 会不会改变门控输出"，
所以真值就该是 30-seed 输出本身。

用法：
    python scripts/probe_seed_var.py                                  # Mk01~Mk10
    python scripts/probe_seed_var.py --data_dir data/hurink \\
        --instances Hed01,Hed02,... --out logs/probe_seed_var_hurink.json
"""

import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import init_probe as ip  # noqa: E402  （复用 probe_one：同一估计量，不另写一套）

NS = [1, 2, 3, 4, 5, 10, 15, 20, 30]


def subsamples(n_seed, n, n_sub, rng):
    """返回 (子集列表, 是否枚举穷尽)。

    **必须用 math.comb 而不是自己乘除**：子集池的大小决定了能抽多少个**不重复**
    子集，目标数超过池子大小时 `while` 永远凑不齐 —— 会静默死循环
    （本脚本第一版就踩了这个：n_seed=8、n=5 的池子只有 56 个，却要 300 个）。
    """
    if n >= n_seed:
        return [tuple(range(n_seed))], True
    tot = math.comb(n_seed, n)
    if tot <= n_sub or tot <= 20000:
        return list(itertools.combinations(range(n_seed), n)), True
    seen, out = set(), []
    while len(out) < n_sub:
        s = tuple(sorted(rng.choice(n_seed, size=n, replace=False).tolist()))
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out, False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", default=",".join(ip.DEFAULT_INSTANCES))
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed_start", type=int, default=42)
    ap.add_argument("--n_pop", type=int, default=100)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--thr", type=float, default=-1.0,
                    help="门控门限（开发集标定，固定后不得再按留出集调整）")
    ap.add_argument("--n_sub", type=int, default=2000)
    ap.add_argument("--rng_seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "probe_seed_var.json"))
    args = ap.parse_args()

    instances = [s.strip() for s in args.instances.split(",") if s.strip()]
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    rng = np.random.default_rng(args.rng_seed)

    print("=" * 100)
    print("零代探针的 seed 方差：n = %s" % NS)
    print("=" * 100)
    print("实例 %d 个 × %d seeds；门限 %.2f%%（判定真值 = 该实例的 30-seed 判定）"
          % (len(instances), len(seeds), args.thr))
    print("-" * 100)

    t0 = time.perf_counter()
    rows, by_n = [], {str(n): {"sds": [], "agrees": []} for n in NS}
    for inst in instances:
        hv_mix3, hv_mwr = [], []
        for s in seeds:
            got = ip.probe_one(inst, s, args.data_dir, args.n_pop)
            hv_mix3.append(got["I_mix3"][4])
            hv_mwr.append(got["I_mwr"][4])
        hv_mix3 = np.asarray(hv_mix3, float)
        hv_mwr = np.asarray(hv_mwr, float)

        def lam(idx):
            b = hv_mix3[idx].mean()
            return (hv_mwr[idx].mean() - b) / b * 100.0

        lam_full = lam(np.arange(len(seeds)))
        dec_full = lam_full >= args.thr
        row = {"instance": inst, "lam30": float(lam_full),
               "dist_thr": float(lam_full - args.thr), "sd": {}, "agree": {},
               "n_sub": {}}
        for n in NS:
            subs, _ = subsamples(len(seeds), n, args.n_sub, rng)
            vals = np.array([lam(np.asarray(s)) for s in subs], float)
            agree = float(np.mean((vals >= args.thr) == dec_full))
            row["sd"][str(n)] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            row["agree"][str(n)] = agree
            row["n_sub"][str(n)] = len(subs)
            by_n[str(n)]["sds"].append(row["sd"][str(n)])
            by_n[str(n)]["agrees"].append(agree)
        rows.append(row)
        print("  %-7s lam(30)=%+7.3f%%  判定=%-5s  SD: n1=%.2f n3=%.2f n5=%.2f n10=%.2f  "
              "agree: n1=%.2f n3=%.2f n5=%.2f n10=%.2f"
              % (inst, row["lam30"], "gain" if dec_full else "none",
                 row["sd"]["1"], row["sd"]["3"], row["sd"]["5"], row["sd"]["10"],
                 row["agree"]["1"], row["agree"]["3"], row["agree"]["5"],
                 row["agree"]["10"]), flush=True)

    summary = {}
    print("\n" + "=" * 100)
    print("汇总（跨 %d 个实例）" % len(rows))
    print("=" * 100)
    print("%-6s %-12s %-12s %-12s" % ("n", "SD 均值", "SD 中位数", "判定一致率均值"))
    for n in NS:
        sds = np.asarray(by_n[str(n)]["sds"], float)
        ags = np.asarray(by_n[str(n)]["agrees"], float)
        summary[str(n)] = {"sd_mean": float(sds.mean()), "sd_median": float(np.median(sds)),
                           "agree_mean": float(ags.mean()),
                           "agree_min": float(ags.min())}
        print("%-6d %-12.3f %-12.3f %-12.3f"
              % (n, sds.mean(), np.median(sds), ags.mean()))

    payload = {"instances": instances, "n_seeds": len(seeds), "thr": args.thr,
               "n_sub": args.n_sub, "rng_seed": args.rng_seed, "ns": NS,
               "set_name": ("Mk01--Mk10（开发集）" if all(i.startswith("Mk") for i in instances)
                            else "Hurink $e$-data（留出集）" if all(i.startswith("Hed") for i in instances)
                            else "%s 等 %d 个实例" % (instances[0], len(instances))),
               "summary": summary, "rows": rows,
               "elapsed_s": time.perf_counter() - t0}
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print("\n耗时 %.1fs  已写出 %s" % (payload["elapsed_s"], args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
