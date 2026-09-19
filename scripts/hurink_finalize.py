#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Hurink 留出集收尾流水线：主实验跑完 -> 门控 -> 探针 -> 论文终版。

为什么需要它
------------
留出验证的收尾是**四步有严格先后**的链路，手工敲命令有两个已知的失败模式：

1. **在实验还在跑时就把上游文件当成"已完成"**：`logs/_hedNN_init.partial.json` 是
   **边跑边写**的，某一刻它可能只有 2 个臂的标签。手工合并会得到"看起来正常、
   实际上少一个臂"的 lab，而 `aig_gating` 只会安静地少算一个目标量。
   本脚本因此**检查每个 partial 的 mtime**：太新就判定"实验仍在跑"并拒绝开工。
2. **漏掉重跑探针**：`logs/hurink_probe.json` 里的 lambda_0 本身与搜索无关，
   但它的 `correlations/permutation` 是对着 `--aig` 指向的文件算的。
   拿着旧 aig 文件跑出来的探针，数字会**静默过期**。

所以这里把四步串成一条命令，并在每一步前做**可判定的前置检查**。

步骤
----
    [0] 合并 logs/_hed*_init.json -> logs/hurink_init.json（按 (instance,label,seed) 去重）
    [1] aig_gating.py  -> logs/hurink_aig.json   （目标 A/B 的逐实例净增量与置换校正）
    [2] init_probe.py  -> logs/hurink_probe.json （零代探针，带 --aig 重算，覆盖旧文件）
    [3] paper_export.py -> paper/tables/*.tex + paper/figures/* （**不带** --allow-missing）
    [4] paper/build.py  -> paper/main.pdf        （**不带** --allow-missing，终版）

用法
----
    python scripts/hurink_finalize.py --help
    python scripts/hurink_finalize.py                # 全链路
    python scripts/hurink_finalize.py --from 1       # 从第 1 步开始（合并已做过时）
    python scripts/hurink_finalize.py --dry-run      # 只做检查与打印，不动任何文件
"""

import argparse
import collections
import glob
import io
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
LOGS = os.path.join(ROOT, "logs")
DATA = os.path.join(ROOT, "data", "hurink")
EXPECT_ARMS = ("I_rand", "I_mix3", "I_mwr")
EXPECT_SEEDS = 30
# partial 文件在过去这么多秒内被写过 -> 判定"实验仍在跑"，拒绝开工
PARTIAL_QUIET_S = 240


def run(cmd, dry_run=False, tag=""):
    """跑一条子命令：**退出码原样返回**（不用管道，避免取到错误的退出码）。"""
    print("\n" + "=" * 92)
    print("[%s] %s" % (tag, " ".join(os.path.basename(c) if i == 1 else c
                                    for i, c in enumerate(cmd))))
    print("=" * 92)
    t0 = time.perf_counter()
    if dry_run:
        print("  （dry-run：未执行）")
        return 0
    r = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    print("[%s] rc=%d，耗时 %.1f s" % (tag, r.returncode, time.perf_counter() - t0))
    return r.returncode


def check_ready():
    """开工前检查：实验是否真的跑完、覆盖是否真的是预注册的全集。"""
    problems, warns = [], []

    parts = sorted(glob.glob(os.path.join(LOGS, "_hed*_init.partial.json")))
    now = time.time()
    for p in parts:
        age = now - os.path.getmtime(p)
        if age < PARTIAL_QUIET_S:
            problems.append("%s 在 %.0f s 前还被写过 -> 实验仍在跑"
                            % (os.path.basename(p), age))

    finals = sorted(glob.glob(os.path.join(LOGS, "_hed*_init.json")))
    finals = [f for f in finals if ".partial." not in f]
    done = set()
    for f in finals:
        try:
            with io.open(f, encoding="utf-8") as fh:
                rows = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            problems.append("%s 读不了（%s）" % (os.path.basename(f), e))
            continue
        c = collections.Counter(r.get("label") for r in rows)
        lack = [a for a in EXPECT_ARMS if c.get(a, 0) != EXPECT_SEEDS]
        if lack:
            problems.append("%s 臂不全：%s（期望每臂 %d 条）"
                            % (os.path.basename(f), {a: c.get(a, 0) for a in lack},
                               EXPECT_SEEDS))
        done.add(rows[0].get("instance") if rows else None)

    if not os.path.isdir(DATA):
        problems.append("找不到 %s" % DATA)
        prereg = []
    else:
        prereg = sorted(os.path.splitext(f)[0] for f in os.listdir(DATA)
                        if f.endswith(".fjs"))
    missing = sorted(set(prereg) - done)
    if missing:
        problems.append("还没跑完 %d 个预注册实例：%s%s"
                        % (len(missing), missing[:8], " …" if len(missing) > 8 else ""))

    print("预注册实例 %d 个；已完成 %d 个；partial 文件 %d 个"
          % (len(prereg), len(done & set(prereg)), len(parts)))
    for w in warns:
        print("  [!] " + w)
    for p in problems:
        print("  [X] " + p)
    return problems


def merge(merged_path, dry_run=False):
    """按 (instance,label,seed) 去重合并；重复键要**报数**而不是静默覆盖。"""
    files = [f for f in sorted(glob.glob(os.path.join(LOGS, "_hed*_init.json")))
             if ".partial." not in f and not f.endswith(".merged.json")]
    rows, seen, dup = [], set(), 0
    for f in files:
        try:
            with io.open(f, encoding="utf-8") as fh:
                sub = json.load(fh)
        except (json.JSONDecodeError, OSError):
            print("  !! 跳过读取失败的文件 %s" % os.path.basename(f))
            continue
        for r in sub:
            k = (r.get("instance"), r.get("label"), r.get("seed"))
            if k in seen:
                dup += 1
            seen.add(k)
            rows.append(r)
    rows.sort(key=lambda r: (str(r.get("instance")), str(r.get("label")),
                             r.get("seed", 0)))
    insts = sorted({r.get("instance") for r in rows if r.get("instance")})
    try:
        shown = os.path.relpath(merged_path, ROOT)
    except ValueError:
        # 跨盘符时 relpath 会抛（例如把中间产物指到 C: 而仓库在 E:）。
        # 这只影响一行显示，不该让整条流水线挂掉。
        shown = merged_path
    print("合并 %d 个文件 -> %s" % (len(files), shown))
    print("  行数 %d；实例 %d 个；重复键 %d 个" % (len(rows), len(insts), dup))
    if dup:
        print("  [!] 有重复键：上游可能有问题，别静默忽略")
    if not dry_run:
        with io.open(merged_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(rows, fh, ensure_ascii=False)
    return rows, dup


def merge_step(merged_path, dry_run=False):
    """第 0 步的 rc 适配器。

    **缺陷 35**：`merge()` 返回 `(rows, dup)` 二元组，而 `steps` 的约定是返回
    rc（int）。第一版直接写 `lambda: merge(...)`，于是 `rc` 恒为非空元组、
    真值恒为真 —— 每一次运行都会在第 0 步报"失败"退出，并且把整个 `rows`
    （实测 7.5 MB）当作错误信息打印出来。合并本身其实一直是好的。
    """
    _rows, dup = merge(merged_path, dry_run)
    return 1 if dup else 0


def report_gate():
    """把门控结果打印成人能读的形式（**不改任何判断，也不调门限**）。"""
    p = os.path.join(LOGS, "paper_export.json")
    if not os.path.exists(p):
        return
    with io.open(p, encoding="utf-8") as fh:
        hk = json.load(fh).get("hurink")
    if not hk:
        print("  paper_export.json 里没有 hurink 段 -> 门控未写出")
        return
    dev, hold = hk["dev"], hk["hold"]
    n = hk["n"]
    print("\n留出集门控（门限冻结在 lambda_0 >= %.1f）" % hk["thr"])
    print("  开发集 Mk01--Mk10：命中 %d 误放 %d 漏放 %d 正确拒绝 %d"
          % (dev.get("tp", 0), dev.get("fp", 0), dev.get("fn", 0), dev.get("tn", 0)))
    print("  留出集 %d 实例：命中 %d 误放 %d 漏放 %d 正确拒绝 %d"
          % (n, hold.get("tp", 0), hold.get("fp", 0), hold.get("fn", 0),
             hold.get("tn", 0)))
    print("  留出集显著为正 %d/%d -> 判对率 %.3f"
          % (hk["n_gain"], n, (hold.get("tp", 0) + hold.get("tn", 0)) / max(n, 1)))
    print("  ** 如实报告；本脚本不提供任何调门限的开关。**")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="from_step", type=int, default=0,
                    help="从第几步开始（0 合并 / 1 门控 / 2 探针 / 3 导出 / 4 编译）")
    ap.add_argument("--to", dest="to_step", type=int, default=4, help="做到第几步为止")
    ap.add_argument("--perm", type=int, default=20000, help="置换零假设次数")
    ap.add_argument("--dry-run", action="store_true", help="只检查并打印，不动文件")
    ap.add_argument("--skip-ready-check", action="store_true",
                    help="跳过实验完成度检查（仅在明知有临时缺失时用）")
    args = ap.parse_args()

    merged = os.path.join(LOGS, "hurink_init.json")
    aig = os.path.join(LOGS, "hurink_aig.json")
    probe = os.path.join(LOGS, "hurink_probe.json")

    if not args.skip_ready_check:
        print("=" * 92)
        print("开工前检查")
        print("=" * 92)
        problems = check_ready()
        if problems:
            print("\n[拒绝开工] 上面的检查没过。等实验跑完（或加 --skip-ready-check 强行继续）。")
            return 1

    steps = [
        (0, "合并", lambda: merge_step(merged, args.dry_run)),
        (1, "门控", lambda: run([PY, "scripts/aig_gating.py", "--labs", merged,
                                 "--data_dir", DATA,
                                 "--targets", "I_rand,I_mix3",
                                 "--perm", str(args.perm), "--out", aig],
                                args.dry_run, "1/4 门控")),
        (2, "探针", lambda: run([PY, "scripts/init_probe.py",
                                 "--instances", ",".join("Hed%02d" % i
                                                         for i in range(1, 67)),
                                 "--seeds", "30", "--data_dir", DATA,
                                 "--aig", aig, "--out", probe],
                                args.dry_run, "2/4 探针")),
        (3, "导出", lambda: run([PY, "scripts/paper_export.py"], args.dry_run, "3/4 导出")),
        (4, "编译", lambda: run([PY, "paper/build.py"], args.dry_run, "4/4 编译")),
    ]
    for i, name, fn in steps:
        if not (args.from_step <= i <= args.to_step):
            print("\n（跳过第 %d 步 %s）" % (i, name))
            continue
        rc = fn()
        if rc:
            print("\n[中断] 第 %d 步「%s」失败（rc=%s）。修好后用 --from %d 接着跑。"
                  % (i, name, rc, i))
            return rc

    if not args.dry_run:
        report_gate()
    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
