"""单次运行指纹导出：用于「优化前 vs 优化后」的逐位一致比对。

输出内容刻意选得足够细 —— 只要随机流或数值路径有任何一位不同，
history 里的 hv/cv/dv/z 或 final_pf 就会对不上。

用法：
    python scripts/dump_run.py --instance Mk10 --seed 7 --max_gen 60 --out _a.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="Mk10")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_pop", type=int, default=100)
    ap.add_argument("--max_gen", type=int, default=60)
    ap.add_argument("--ls_trials", type=int, default=1)
    ap.add_argument("--fixed_T", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from rmoea_d.algorithm import RMOEAD

    s = RMOEAD(instance_name=args.instance, n_pop=args.n_pop,
               max_gen=args.max_gen, seed=args.seed,
               enable_rvns=True, fixed_T=args.fixed_T,
               rvns_ls_trials=args.ls_trials)
    res = s.solve()

    hist = res["history"]
    payload = {
        "instance": args.instance,
        "seed": args.seed,
        "n_pop": args.n_pop,
        "max_gen": args.max_gen,
        "final_hv": res.get("final_hv"),
        "n_gen": len(hist),
        # 逐代指纹：hv/cv/dv/z 与两个目标的最优值
        "traj": [[h["gen"], h["T"], h["pf_size"], h["archive_size"],
                  h["hv"], h["cv"], h["dv"],
                  h["z"][0], h["z"][1],
                  h["best_makespan"], h["best_workload"]]
                 for h in hist],
        # 末代邻域选择概率（RVNS 算子权重）——对随机流极其敏感
        "last_probs": [float(x) for x in s.rvns.get_probabilities()],
        "final_pf": [[float(p["Makespan"]), float(p["Workload"])]
                     for p in res["final_pf"]],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"[{args.instance} seed={args.seed} G={args.max_gen}] "
          f"HV={payload['final_hv']:.10f} pf={len(payload['final_pf'])} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
