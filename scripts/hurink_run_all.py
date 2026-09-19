#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 Hurink 留出集上批量驱动初始化变体实验台（`t_leverage_sweep.py` 的批处理外壳）。

为什么需要它
------------
`t_leverage_sweep.py` 一次只跑**一个实例**，而留出验证需要 20–30 个实例。
更要紧的是它的断点续跑判据是 `{(label, seed)}` —— **不含 instance**：
把多个实例写进同一个 `--out` 时，第二个实例起的所有 run 都会被判成"已完成"
而**静默跳过**（跑完 10 个实例却只有 1 个的数据，且不报错）。
本外壳因此**一实例一文件**（`logs/_hed01_init.json` …），与既有 `_mkNN_init.json`
同构；需要合并时再显式 `--merge`（只做拼接，不动单个文件）。

与既有脚本的关系
----------------
* 实验台本体：`scripts/t_leverage_sweep.py`（**未改动**，本外壳只做调度）
* 分析：`scripts/aig_gating.py --labs <合并文件> --data_dir data/hurink`
* 探针：`scripts/init_probe.py --instances Hed01,... --data_dir data/hurink`

用法
----
    python scripts/hurink_run_all.py --limit 30 --workers 8      # Hed01~Hed30
    python scripts/hurink_run_all.py --instances Hed01,Hed02 --n_runs 5
    python scripts/hurink_run_all.py --merge                      # 只合并已有结果
    python scripts/hurink_run_all.py --merge_only
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(ROOT, "scripts", "t_leverage_sweep.py")
DEFAULT_DATA = os.path.join(ROOT, "data", "hurink")


def per_instance_out(inst, logs_dir):
    return os.path.join(logs_dir, "_%s_init.json" % inst.lower())


def merge(logs_dir, merged_path, verbose=True):
    """把 logs/_hedNN_init.json 拼成一份合并 lab（供 --labs 一次传入）。

    按 (instance, label, seed) 去重；同键冲突时**保留后者并报数**，
    不静默覆盖（重复键意味着上游有 bug，必须看得见）。
    """
    files = sorted(glob.glob(os.path.join(logs_dir, "_hed*_init.json")))
    files = [f for f in files if ".partial." not in f and not f.endswith(".merged.json")]
    rows, seen, dup = [], {}, 0
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                sub = json.load(fh)
        except (json.JSONDecodeError, OSError):
            print("  !! 跳过读取失败的文件 %s" % os.path.basename(f))
            continue
        for r in sub:
            k = (r.get("instance"), r.get("label"), r.get("seed"))
            if k in seen:
                dup += 1
            seen[k] = True
            rows.append(r)
    rows.sort(key=lambda r: (str(r.get("instance")), str(r.get("label")), r.get("seed", 0)))
    with open(merged_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rows, fh, ensure_ascii=False)
    insts = sorted({r["instance"] for r in rows if r.get("instance")})
    if verbose:
        print("合并 %d 个文件 -> %s" % (len(files), os.path.relpath(merged_path, ROOT)))
        print("  行数 %d，实例 %d 个（%s），重复键 %d 个"
              % (len(rows), len(insts),
                 "%s..%s" % (insts[0], insts[-1]) if insts else "-", dup))
    return rows, files, dup


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", default=None,
                    help="逗号分隔的 Hurink 实例名；不给则由 --limit 生成 Hed01..HedNN")
    ap.add_argument("--limit", type=int, default=30,
                    help="生成 Hed01..HedNN（默认 30）")
    ap.add_argument("--arms", default="I_rand,I_mix3,I_mwr",
                    help="逗号分隔的臂名（见 t_leverage_sweep.ARM_DEF）")
    ap.add_argument("--n_runs", type=int, default=30)
    ap.add_argument("--seed_start", type=int, default=42)
    ap.add_argument("--max_gen", type=int, default=200)
    ap.add_argument("--n_pop", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--data_dir", default=DEFAULT_DATA)
    ap.add_argument("--logs_dir", default=os.path.join(ROOT, "logs"))
    ap.add_argument("--merged", default=os.path.join(ROOT, "logs", "hurink_init.json"))
    ap.add_argument("--per_batch_timeout", type=int, default=5400,
                    help="单实例超时秒数；超时视为「本批无进展」并进入下一轮重试")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--merge", action="store_true",
                    help="跑完后合并（默认也合并）")
    ap.add_argument("--merge_only", action="store_true",
                    help="只合并已有结果，不跑实验")
    args = ap.parse_args()

    os.makedirs(args.logs_dir, exist_ok=True)

    if args.merge_only:
        merge(args.logs_dir, args.merged)
        return 0

    if not os.path.exists(SWEEP):
        print("[错误] 找不到 %s" % SWEEP)
        return 1

    if args.instances:
        instances = [s.strip() for s in args.instances.split(",") if s.strip()]
    else:
        instances = ["Hed%02d" % i for i in range(1, args.limit + 1)]

    available = {os.path.splitext(f)[0] for f in os.listdir(args.data_dir)
                 if f.endswith(".fjs")}
    missing = [i for i in instances if i not in available]
    if missing:
        # 立刻报错，不要白跑一整轮（缺陷 19 同款：参数/数据装配错误要在开工前暴露）
        print("[错误] %s 里没有这些实例：%s" % (args.data_dir, missing))
        print("       可用 %d 个，例：%s" % (len(available), sorted(available)[:6]))
        return 2

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"

    print("实例 %d 个，臂 %s，seeds %d，workers %d"
          % (len(instances), args.arms, args.n_runs, args.workers))
    print("总任务 = %d" % (len(instances) * len(args.arms.split(",")) * args.n_runs))

    t_all = time.perf_counter()
    unfinished = []
    for inst in instances:
        out = per_instance_out(inst, args.logs_dir)
        for attempt in range(args.attempts):
            n_before = 0
            if os.path.exists(out):
                try:
                    n_before = len(json.load(open(out, encoding="utf-8")))
                except (json.JSONDecodeError, OSError):
                    n_before = 0
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    [sys.executable, SWEEP, "--instance", inst,
                     "--arms", args.arms, "--n_runs", str(args.n_runs),
                     "--seed_start", str(args.seed_start),
                     "--n_pop", str(args.n_pop), "--max_gen", str(args.max_gen),
                     "--workers", str(args.workers), "--data_dir", args.data_dir,
                     "--out", out],
                    cwd=ROOT, env=env, capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                    timeout=args.per_batch_timeout)
                rc = str(proc.returncode)
                note = ((proc.stdout or "").strip().splitlines()[-1:] or [""])[0]
            except subprocess.TimeoutExpired:
                rc = "TIMEOUT"
                note = ("超过 %ds 被杀；已完成的 run 落在 .partial.json，"
                        "下一轮续跑" % args.per_batch_timeout)
            n_after = 0
            if os.path.exists(out):
                try:
                    n_after = len(json.load(open(out, encoding="utf-8")))
                except (json.JSONDecodeError, OSError):
                    n_after = 0
            print("  %-6s rc=%-8s %6.0fs  rows %d -> %d  %s"
                  % (inst, rc, time.perf_counter() - t0, n_before, n_after,
                     note[:70]), flush=True)
            if n_after > n_before or rc == "0":
                break
        want = len(args.arms.split(",")) * args.n_runs
        if n_after < want:
            unfinished.append((inst, n_after, want))

    print("\n总耗时 %.0fs" % (time.perf_counter() - t_all))
    if unfinished:
        # 不静默：缺口的实例必须列出来，否则会拿着不完整的数据写论文
        print("!! 未跑满的实例（%d 个）：" % len(unfinished))
        for inst, got, want in unfinished:
            print("     %-6s %d/%d" % (inst, got, want))

    merge(args.logs_dir, args.merged)
    return 1 if unfinished else 0


if __name__ == "__main__":
    sys.exit(main())
