#!/usr/bin/env python3
"""邻域大小 T 的杠杆扫参 + Q-PAS 设定实验台（分批 · 增量落盘 · 可断点续跑）。

用途
----
回答「自适应选 T（Q-PAS）在这个实例上到底有没有发挥空间」：
先扫固定 T，判断 T 是不是有效杠杆；再跑 Q-PAS 各设定，看它能否学到好 T。

用法
----
    # 固定 T 扫参（纯 T 效应，关闭 Q-PAS 与 RVNS）
    python scripts/t_leverage_sweep.py --instance Mk10 ^
        --arms T05,T10,T15,T20,T50,T100

    # Q-PAS 各设定（平局随机化后的 v2 才是当前默认口径）
    python scripts/t_leverage_sweep.py --instance Mk10 ^
        --arms QPAS2_dv,QPAS2_hv_wide,QPAS2_opt_hv,Full2

    # 重复运行会自动跳过已完成的 (arm, seed)，可分批续跑
    python scripts/t_leverage_analysis.py --lab_json logs/_mk10_lab.json

注意
----
* 所有 run 共用 ``instance_hv_bounds``（由实例数据确定性推出），但**跨臂比较请用
  ``t_leverage_analysis.py`` 统一重算**——它按参考集归一化（所有臂所有 run 的前沿并集，
  ref=(1.02,1.02)）重算 HV，与 ``scripts/ablation_analysis.py`` 口径一致。
* Q-PAS v1（``QPAS_*``）保留是为了复现「平局自锁」的历史结果；
  当前默认口径是 v2（``QPAS2_*``，``tie_break="random"``）。
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
logging.basicConfig(level=logging.WARNING)

# 实验臂定义：label -> (kind, kwargs)
#   kind: "fixed" 固定 T | "qpas" 仅 Q-PAS | "full" Q-PAS + RVNS
ARM_DEF = {
    # ── 纯 T 效应（关闭 Q-PAS 与 RVNS）──
    "T05":   ("fixed", dict(fixed_T=5)),
    "T10":   ("fixed", dict(fixed_T=10)),
    "T15":   ("fixed", dict(fixed_T=15)),
    "T20":   ("fixed", dict(fixed_T=20)),
    "T50":   ("fixed", dict(fixed_T=50)),
    "T100":  ("fixed", dict(fixed_T=100)),
    # ── Q-PAS v1：论文口径 + 仅修 ε 极性（平局仍取索引 0 → 自锁 T=5）──
    "QPAS_dv":      ("qpas", dict(ql_reward_mode="dv")),
    "QPAS_cv_dv":   ("qpas", dict(ql_reward_mode="cv_dv")),
    "QPAS_hv":      ("qpas", dict(ql_reward_mode="hv")),
    "QPAS_wide":    ("qpas", dict(ql_reward_mode="dv", ql_actions=[5, 10, 20, 50])),
    "QPAS_wide_hv": ("qpas", dict(ql_reward_mode="hv", ql_actions=[5, 10, 20, 50])),
    "Full":         ("full", dict(ql_reward_mode="dv")),
    # ── Q-PAS v2：平局随机化（打破自锁，当前默认口径）──
    "QPAS2_dv":       ("qpas", dict(ql_reward_mode="dv", ql_tie_break="random")),
    "QPAS2_wide":     ("qpas", dict(ql_reward_mode="dv", ql_tie_break="random",
                                    ql_actions=[5, 10, 20, 50])),
    "QPAS2_hv_wide":  ("qpas", dict(ql_reward_mode="hv", ql_tie_break="random",
                                    ql_actions=[5, 10, 20, 50])),
    "QPAS2_opt_wide": ("qpas", dict(ql_reward_mode="dv", ql_tie_break="random",
                                    ql_q_init="optimistic",
                                    ql_actions=[5, 10, 20, 50])),
    "QPAS2_wide_e5":  ("qpas", dict(ql_reward_mode="dv", ql_tie_break="random",
                                    ql_epsilon=0.5, ql_actions=[5, 10, 20, 50])),
    "Full2":          ("full", dict(ql_reward_mode="dv", ql_tie_break="random",
                                    ql_actions=[5, 10, 20, 50])),
    # ── Q-PAS v3：组合有效因子 / 更宽动作空间 / 连续 HV 奖励 ──
    "QPAS2_opt_hv":  ("qpas", dict(ql_reward_mode="hv", ql_tie_break="random",
                                   ql_q_init="optimistic",
                                   ql_actions=[5, 10, 20, 50])),
    "QPAS2_hv_fine": ("qpas", dict(ql_reward_mode="hv", ql_tie_break="random",
                                   ql_actions=[5, 10, 15, 20, 30, 50])),
    "QPAS3_hvc":     ("qpas", dict(ql_reward_mode="hv_cont", ql_tie_break="random",
                                   ql_actions=[5, 10, 20, 50])),
    "QPAS3_hvc_opt": ("qpas", dict(ql_reward_mode="hv_cont", ql_tie_break="random",
                                   ql_q_init="optimistic",
                                   ql_actions=[5, 10, 20, 50])),
    "Full3":         ("full", dict(ql_reward_mode="hv", ql_tie_break="random",
                                   ql_q_init="optimistic",
                                   ql_actions=[5, 10, 20, 50])),
}


def _run_one(job):
    instance, n_pop, max_gen, data_dir, label, kind, kwargs, seed = job
    from rmoea_d.algorithm import RMOEAD

    extra = dict(kwargs)
    if kind == "full":
        extra.update(enable_rvns=True, rvns_ls_trials=1)
    else:
        extra.update(enable_rvns=False)

    solver = RMOEAD(instance_name=instance, n_pop=n_pop, max_gen=max_gen,
                    seed=seed, data_dir=data_dir, **extra)
    t0 = time.perf_counter()
    res = solver.solve()
    dt = time.perf_counter() - t0

    hist_T = {}
    for h in solver.history:
        hist_T[int(h["T"])] = hist_T.get(int(h["T"]), 0) + 1

    return dict(
        label=label, seed=seed, instance=instance,
        final_hv=float(res["final_hv"]), total_time=float(dt), hist_T=hist_T,
        q_table=solver.ql.q_table.tolist() if solver.ql else None,
        actions=list(solver.ql.actions) if solver.ql else None,
        final_pf=[[float(p["Makespan"]), float(p["Workload"])] for p in res["final_pf"]],
    )


def _load(path):
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            return []
    return []


def _dump(rows, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default="Mk10")
    ap.add_argument("--arms", required=True, help="逗号分隔的臂名（见 ARM_DEF）")
    ap.add_argument("--n_pop", type=int, default=100)
    ap.add_argument("--max_gen", type=int, default=200)
    ap.add_argument("--seed_start", type=int, default=42)
    ap.add_argument("--n_runs", type=int, default=30)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None,
                    help="结果 JSON 路径（默认 logs/_<instance>_lab.json）")
    args = ap.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_runs))
    out = args.out or os.path.join(ROOT, "logs", f"_{args.instance.lower()}_lab.json")
    partial = out.replace(".json", ".partial.json")

    labels = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in labels:
        if a not in ARM_DEF:
            raise SystemExit(f"unknown arm: {a}\nknown: {sorted(ARM_DEF)}")

    rows = _load(out) + _load(partial)
    done = {(r["label"], r["seed"]) for r in rows}
    jobs = [(args.instance, args.n_pop, args.max_gen, args.data_dir,
             a, ARM_DEF[a][0], ARM_DEF[a][1], s)
            for a in labels for s in seeds if (a, s) not in done]

    print(f"instance={args.instance} arms={labels} todo={len(jobs)} "
          f"already_done={len(done)} workers={args.workers}", flush=True)
    if not jobs:
        print("nothing to do", flush=True)
        return

    t0 = time.perf_counter()
    with mp.Pool(args.workers, maxtasksperchild=2) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_one, jobs, chunksize=1), 1):
            rows.append(r)
            if i % 10 == 0 or i == len(jobs):
                _dump(rows, partial)
                el = time.perf_counter() - t0
                print(f"  [{i}/{len(jobs)}] elapsed={el:.0f}s "
                      f"eta={el / i * (len(jobs) - i):.0f}s  "
                      f"last={r['label']}/s{r['seed']} hv={r['final_hv']:.5f}", flush=True)

    all_done = {(r["label"], r["seed"]) for r in rows}
    target = {(a, s) for a in ARM_DEF for s in seeds}
    if all_done >= target:
        _dump(rows, out)
        if os.path.exists(partial):
            os.remove(partial)
        print("ALL ARMS COMPLETE ->", out, flush=True)
    else:
        _dump(rows, partial)
        missing = sorted({a for a, _ in (target - all_done)})
        print(f"partial saved -> {partial}  remaining arms={missing}", flush=True)
    print(f"batch done in {time.perf_counter() - t0:.0f}s  rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
