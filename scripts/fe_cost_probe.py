# -*- coding: utf-8 -*-
"""单次求值成本探针：一次 RVNS 邻域尝试到底值多少"标准的"MOEA/D 子代评估？

动机（为什么这个数会决定一条结论）
----------------------------------
`response_surface.py` 的等 FE（等计数求值）口径隐含一条定价假设：

    一次 RVNS 邻域尝试  ==  一次 MOEA/D 子代评估  ==  1 FE

§3.3 因此得到「D5（含 RVNS）在等算力下落后 D2」。但若 RVNS 的单次尝试**更便宜**，
那么"等计数求值"就不是等算力，结论会**翻符号**。这不是可以靠推理绕过去的问题，
必须实测每次求值的墙钟成本。

为什么必须单进程 + 轮换顺序（缺陷 24）
--------------------------------------
12-worker 批量的墙钟会把臂间耗时排序弄反（同一臂对在批量 Mk07 上 D2 快 17%，
单进程 G=60 探针里 D2 慢 14%）。故本探针：

1. **单进程**（不并行，无争用）；
2. **轮换顺序**：第 r 轮把臂序循环左移 r 位，使每条臂在"第 k 个执行"的位置上
   出现次数相同 —— 取消预热/缓存/热漂移这类顺序效应（拉丁方设计）；
3. 报**中位数**而非均值，抗单次抖动。

口径
----
* `ms/gen` = 各代耗时之和 / 代数（不含初始化）；
* `ms/FE`  = `ms/gen` / (n_pop × fe_multiplier(臂))，即"每 100 次计数求值花多少毫秒"；
* 初始化开销单独报，因为它在等墙钟口径里也要算（`index_curves` 把 init 加进 t）。

用法
----
    python scripts/fe_cost_probe.py --instance Mk09 --arms D1,D2,D5,RMOEAD \\
        --max_gen 100 --reps 4 --out logs/fe_cost_probe.json
"""
import argparse
import json
import os
import statistics
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from ablation_ladder import LADDER_DICT                      # noqa: E402
from response_surface import fe_multiplier                   # noqa: E402

DEFAULT_ARMS = "D1,D2,D5,RMOEAD"
DEFAULT_SEEDS = [42, 43, 44, 45]


def run_once(instance, label, seed, n_pop, max_gen, data_dir):
    """跑一条 run，返回 (各代耗时之和, 初始化开销, 总墙钟)。"""
    import logging
    logging.disable(logging.WARNING)
    from rmoea_d.algorithm import RMOEAD

    t0 = time.perf_counter()
    solver = RMOEAD(instance_name=instance, n_pop=n_pop, max_gen=max_gen,
                    seed=seed, data_dir=data_dir, **dict(LADDER_DICT[label]))
    res = solver.solve()
    dt = time.perf_counter() - t0

    gen_sum = sum(float(h["time"]) for h in res["history"])
    return gen_sum, dt - gen_sum, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default="Mk09")
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--max_gen", type=int, default=100)
    ap.add_argument("--n_pop", type=int, default=100)
    ap.add_argument("--reps", type=int, default=4, help="轮换轮数（每轮每臂一条）")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    arms = [s.strip() for s in args.arms.split(",") if s.strip()]
    for a in arms:
        if a not in LADDER_DICT:
            print("[错误] 未知臂名: %s" % a)
            return 1

    print("=" * 78)
    print("单次求值成本探针  实例=%s  臂=%s  每轮 G=%d  n_pop=%d  轮数=%d"
          % (args.instance, arms, args.max_gen, args.n_pop, args.reps))
    print("单进程 + 轮换顺序（拉丁方）；报中位数。单位 ms。")
    print("=" * 78, flush=True)

    per_gen = {a: [] for a in arms}
    init = {a: [] for a in arms}
    for r in range(args.reps):
        order = arms[r % len(arms):] + arms[:r % len(arms)]
        seed = DEFAULT_SEEDS[r % len(DEFAULT_SEEDS)]
        for a in order:
            gsum, it, dt = run_once(args.instance, a, seed, args.n_pop,
                                    args.max_gen, args.data_dir)
            per_gen[a].append(gsum / args.max_gen * 1000.0)
            init[a].append(it * 1000.0)
            print("  轮%d/%d  臂=%-8s seed=%d  ms/gen=%7.3f  init=%6.1fms  total=%6.2fs"
                  % (r + 1, args.reps, a, seed, gsum / args.max_gen * 1000.0, it * 1000.0, dt),
                  flush=True)

    med = {a: statistics.median(per_gen[a]) for a in arms}
    fe_g = {a: args.n_pop * fe_multiplier(a) for a in arms}
    base = min(med.values())
    print("\n" + "=" * 78)
    print("%-9s %10s %12s %10s %14s %12s" %
          ("臂", "ms/gen", "ms/次求值", "相对D2", "每代求值", "init(ms)"))
    print("-" * 78)
    for a in arms:
        ms_fe = med[a] / fe_g[a]
        rel = ms_fe / (med["D2"] / fe_g["D2"]) if "D2" in med else float("nan")
        print("%-9s %10.3f %12.5f %10.2fx %14d %12.1f"
              % (a, med[a], ms_fe, rel, fe_g[a], statistics.median(init[a])))

    print("\n读法：'ms/次求值' 相等才意味着等计数求值 == 等算力。")
    print("      若 RVNS 臂 < 1.0x，则等 FE 口径**高估**了 RVNS 的算力消耗，")
    print("      §3.3 的 D5−D2 负号结论需要限定为「按计数求值计价」。")
    print("      两臂 ms/次求值之比 ≈ 1.0 时上述顾虑消失。")
    print("=" * 78)

    if args.out:
        rec = {"instance": args.instance, "max_gen": args.max_gen,
               "n_pop": args.n_pop, "reps": args.reps,
               "ms_per_gen": med, "ms_per_fe": {a: med[a] / fe_g[a] for a in arms},
               "fe_per_gen": fe_g,
               "init_ms": {a: statistics.median(init[a]) for a in arms},
               "raw_ms_per_gen": per_gen, "raw_init_ms": init}
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=1)
        print("已写出 %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
