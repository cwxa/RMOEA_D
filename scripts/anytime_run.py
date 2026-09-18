# -*- coding: utf-8 -*-
"""长程 anytime 实验台：落盘逐代 HV 与**累计墙钟**，供等墙钟比较使用。

为什么单独一个脚本
------------------
`ablation_ladder.py` 落盘了 `hist_hv`（逐代 population HV），但**没有落盘每代耗时**——
`RMOEAD.history[i]["time"]` 每代都算了（`algorithm.py:355/379`），却从未导出。
于是「HV 是不是墙钟的单一函数」只能在**终点**口径上成立，任何「等墙钟」比较都做不了。
本脚本把累计墙钟一起落盘，补上这个缺口（缺陷 20）。

跑一次长 run 即可覆盖所有更小的预算
----------------------------------
已验证随机流按代连续：同一 (实例, 臂, seed) 在 G=60 与 G=200 下的前 60 代 `traj`
逐位相同（`logs/_verify_new_Mk10_s7_G60.json` vs `_G200.json`）。
因此 **G=2000 的一条 run 可以截断出 G=200/440/1000 的全部轨迹值**，
不必为每个预算档各跑一遍——这正是「算力-算法响应面」的算力来源。

臂定义
------
直接复用 `ablation_ladder.LADDER_DICT`，避免两处配置漂移。
注意 8 条阶梯臂里只有 5 条是不同配置（D3 == D4_fixedT 逐位相同；
D4/D5 只差 archive，见 logs/_anytime_ladder.txt 的【0】自检）。

用法：
    # 单 run 计时探针
    python scripts/anytime_run.py --instance Mk10 --arms D5 --seeds 1 --max_gen 2000

    # 完整批次
    python scripts/anytime_run.py --instances Mk07,Mk09,Mk10 --arms D1,D5,RMOEAD \\
        --seeds 30 --max_gen 2000 --workers 12 --out logs/anytime_g2000.json

支持断点续跑：每条 (instance, label, seed) 唯一，重跑跳过已完成的组合。
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from ablation_ladder import LADDER_DICT  # noqa: E402  复用同一份臂配置

DEFAULT_OUT = os.path.join(ROOT, "logs", "anytime_g2000.json")
DEFAULT_INSTANCES = "Mk07,Mk09,Mk10"
DEFAULT_ARMS = "D1,D5,RMOEAD"
SEED_START = 42


def _run_one(job):
    """跑一条 (实例, 臂, seed) 的长 run，落盘逐代 HV + 累计墙钟。"""
    instance, label, seed, n_pop, max_gen, data_dir = job
    import logging
    logging.disable(logging.WARNING)

    from rmoea_d.algorithm import RMOEAD
    from rmoea_d.utils.metrics import instance_hv_bounds, compute_hv

    kwargs = dict(LADDER_DICT[label])
    t0 = time.perf_counter()
    solver = RMOEAD(instance_name=instance, n_pop=n_pop, max_gen=max_gen,
                    seed=seed, data_dir=data_dir, **kwargs)
    res = solver.solve()
    dt = time.perf_counter() - t0

    # 终点用实例确定性边界（同实例内所有臂可比；跨实例不可比）
    bounds = instance_hv_bounds(solver.instance)
    pf = [(d["Makespan"], d["Workload"]) for d in res["final_pf"]]
    hv = compute_hv(pf, norm_bounds=bounds) if pf else 0.0

    hist = res["history"]
    gen_t = [float(h["time"]) for h in hist]
    cum, acc = [], 0.0
    for x in gen_t:
        acc += x
        cum.append(acc)

    return {
        "instance": instance,
        "label": label,
        "seed": seed,
        "n_pop": n_pop,
        "max_gen": max_gen,
        "final_hv": float(hv),
        "total_time": float(dt),
        # 初始化开销 = 总墙钟 − 各代耗时之和（首代之前的种群构造与评估）
        "init_time": float(dt - acc),
        "hist_hv": [float(h["hv"]) for h in hist],
        "hist_time": cum,                       # 累计墙钟（不含初始化）
        "hist_archive": [int(h["archive_size"]) for h in hist],
        "hist_T": [h["T"] for h in hist],
        "final_pf": [[float(a), float(b)] for a, b in pf],
    }


def _dump(rows, path, indent=None):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default="", help="只跑一个实例（覆盖 --instances）")
    ap.add_argument("--instances", default=DEFAULT_INSTANCES,
                    help="逗号分隔，必须逐个列出（不支持 ... 省略写法）")
    ap.add_argument("--arms", default=DEFAULT_ARMS, help="逗号分隔的阶梯臂名")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--n_pop", type=int, default=100)
    ap.add_argument("--max_gen", type=int, default=2000)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    instances = ([args.instance] if args.instance
                 else [s.strip() for s in args.instances.split(",") if s.strip()])
    arms = [s.strip() for s in args.arms.split(",") if s.strip()]
    for a in arms:
        if a not in LADDER_DICT:
            print("[错误] 未知臂名: %s（可选: %s）" % (a, ", ".join(sorted(LADDER_DICT))))
            return 1

    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(ROOT, args.data_dir)
    have = {os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith(".fjs")}
    missing = [i for i in instances if i not in have]
    if missing:
        print("[错误] data 目录没有这些实例: %s" % missing)
        return 1

    done = set()
    if os.path.exists(args.out):
        try:
            with open(args.out, encoding="utf-8") as fh:
                done = {(r["instance"], r["label"], r["seed"]) for r in json.load(fh)}
        except (json.JSONDecodeError, OSError):
            done = set()

    jobs = []
    for inst in instances:
        for lab in arms:
            for k in range(args.seeds):
                seed = SEED_START + k
                if (inst, lab, seed) in done:
                    continue
                jobs.append((inst, lab, seed, args.n_pop, args.max_gen, args.data_dir))

    print("=" * 74)
    print("anytime 长程实验台  G=%d  实例=%s  臂=%s" % (args.max_gen, instances, arms))
    print("已完成 %d 条，待跑 %d 条，workers=%d" % (len(done), len(jobs), args.workers))
    print("=" * 74, flush=True)
    if not jobs:
        print("无待跑条目。")
        return 0

    rows = []
    if os.path.exists(args.out):
        try:
            with open(args.out, encoding="utf-8") as fh:
                rows = json.load(fh)
        except (json.JSONDecodeError, OSError):
            rows = []

    t0 = time.perf_counter()
    n_new = 0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = {ex.submit(_run_one, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                r = fut.result()
            except Exception as e:                      # noqa: BLE001
                print("  ✗ %s/%s/%d 失败: %s: %s"
                      % (j[0], j[1], j[2], type(e).__name__, e), flush=True)
                continue
            rows.append(r)
            n_new += 1
            if n_new % 5 == 0 or n_new == len(jobs):
                el = time.perf_counter() - t0
                eta = el / n_new * (len(jobs) - n_new)
                print("  [%d/%d] %s/%s/%d  HV=%.6f  %.1fs  elapsed=%.0fs  ETA=%.0fs"
                      % (n_new, len(jobs), r["instance"], r["label"], r["seed"],
                         r["final_hv"], r["total_time"], el, eta), flush=True)
                _dump(rows, args.out)

    rows.sort(key=lambda r: (r["instance"], r["label"], r["seed"]))
    _dump(rows, args.out, indent=1)
    el = time.perf_counter() - t0
    print("\n完成 %d 条（用时 %.0fs = %.2fh），累计落盘 %d 条 → %s"
          % (n_new, el, el / 3600.0, len(rows), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
