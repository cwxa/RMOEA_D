# -*- coding: utf-8 -*-
"""按实例分批驱动论文 6 级阶梯实验（`scripts/ablation_ladder.py` 的批处理外壳）。

为什么需要它：`spawn` 进程池下 Mk10（20×15）的单 run 明显慢于 Mk0x。
若一次性提交全部 10×8×30 = 2400 条，外层超时窗口很难覆盖，
中途被杀就会丢掉整批进度。**按实例切分**后每批只有 240 条，
远小于任何超时窗口，配合 `ablation_ladder.py` 的断点续跑即可稳定跑完。

用法：

    python scripts/ladder_run_all.py                       # Mk01~Mk10
    python scripts/ladder_run_all.py --seeds 30 --workers 6
    python scripts/ladder_run_all.py --instances Mk01,Mk02  # 只跑前两个实例

每一批结束后会把 `logs/ablation_ladder.json` 的累计进度打印出来；
全部完成时 `ablation_ladder.py` 会把 partial 提升为主结果文件并自动跑分析。

退出码：0 全部完成；1 有实例在重试次数内仍未跑完。
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "ablation_ladder.py")
DEFAULT_OUT = os.path.join(ROOT, "logs", "ablation_ladder.json")


def done_set(out_path):
    """读取主文件 + partial 的全部 (instance, label, seed)（去重）。"""
    total = set()
    for path in (out_path, out_path + ".partial.json"):
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                rows = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue        # 写入中的文件可能短暂不完整，跳过即可
        total |= {(r["instance"], r["label"], r["seed"]) for r in rows}
    return total


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", default=",".join(f"Mk{i:02d}" for i in range(1, 11)),
                    help="逗号分隔的实例名（默认 Mk01~Mk10）")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--per_batch_timeout", type=int, default=3000,
                    help="单批（一个实例）的秒级超时")
    ap.add_argument("--attempts", type=int, default=6,
                    help="每个实例最多重试几次")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    instances = [s.strip() for s in args.instances.split(",") if s.strip()]
    if not os.path.exists(SCRIPT):
        print("[错误] 找不到", SCRIPT)
        return 1

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"

    # 每实例的目标条数：臂数 x seeds。臂数由 ablation_ladder.py 的 LADDER 决定，
    # 这里用"完成一批后条数是否不再增长"来判断，不硬编码臂数。
    t_start = time.perf_counter()
    unfinished = []
    for inst in instances:
        for attempt in range(args.attempts):
            done = done_set(args.out)
            n_inst = sum(1 for i, _, _ in done if i == inst)
            print(f"{inst}: {n_inst} runs done, total {len(done)}  "
                  f"(attempt {attempt})", flush=True)
            t0 = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, SCRIPT, "--instance", inst,
                 "--seeds", str(args.seeds), "--workers", str(args.workers),
                 "--out", args.out],
                cwd=ROOT, env=env, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=args.per_batch_timeout)
            tail = (proc.stdout or "").strip().splitlines()[-1:] or [""]
            print(f"    rc={proc.returncode} {time.perf_counter() - t0:.0f}s  {tail[0]}",
                  flush=True)
            after = done_set(args.out)
            if sum(1 for i, _, _ in after if i == inst) > n_inst:
                continue            # 有进展，继续下一轮尝试
            break                   # 无进展，交给下一轮 attempt
        done = done_set(args.out)
        n_after = sum(1 for i, _, _ in done if i == inst)
        print(f"{inst}: {n_after} runs total  "
              f"(TOTAL {len(done)})\n", flush=True)

    done = done_set(args.out)
    by_inst = collections.Counter(i for i, _, _ in done)
    print(f"全部批次结束：累计 {len(done)} runs，用时 "
          f"{time.perf_counter() - t_start:.0f}s")
    print("  " + str(dict(sorted(by_inst.items()))))

    # 判据：某实例的条数明显少于其他实例的众数 => 还没跑完
    if by_inst:
        mode = collections.Counter(by_inst.values()).most_common(1)[0][0]
        unfinished = [i for i in instances if by_inst.get(i, 0) < mode]
    if unfinished:
        print(f"⚠ 以下实例未跑满（应为 {mode} 条）：{unfinished}")
        return 1
    print("✓ 全部实例已跑满。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
