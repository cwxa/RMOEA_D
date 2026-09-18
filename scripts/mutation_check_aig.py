#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""变异测试：往 `scripts/aig_gating.py` 植入**真实缺陷**，确认被"预期的那一条"锁捕获。

为什么需要它
------------
"测试全绿"不等于"锁有区分度"。本项目的规矩是：新写的锁必须验**放开约束则必须失败**。
做法是把已知的真缺陷写进源码、跑指定测试、然后**按字节还原**。

⚠ 血泪教训（本脚本第一版踩的坑）
--------------------------------
用 `open(path, "w")` 写回会把 `\\n` 全部翻译成 `\\r\\n`（Windows 文本模式），
于是"还原"后的 md5 与备份不符 —— **还原步骤本身在改文件**。
本脚本一律用**二进制**读写（`"rb"` / `"wb"`），并在结束前断言 md5 与备份一致。

用法：
    python scripts/mutation_check_aig.py
    python scripts/mutation_check_aig.py --only M1
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "scripts", "aig_gating.py")
TESTS = os.path.join(ROOT, "tests", "test_refactor.py")

# (名字, 原文, 变异, 预期捕获它的测试类)
MUTATIONS = [
    ("M1 丢掉 max（多重比较校正失效）",
     b"max_null = np.nanmax(np.abs(rho_null), axis=0)",
     b"max_null = np.abs(rho_null[0])",
     "TestAIGPermutation"),
    ("M2 留一不真的留一",
     b"            m[i] = False",
     b"            m[i] = True",
     "TestAIGLeaveOneOut"),
    ("M3 用盒口径读数（退回缺陷 22）",
     b"        pr = iva.paired(hv_inst, TREAT, base)",
     b"        pr = iva.paired(hv_box, TREAT, base)",
     "TestAIGTargets"),
    ("M4 base 写死成 I_rand（缺陷 26 复发）",
     b"        pr = iva.paired(hv_inst, TREAT, base)",
     b'        pr = iva.paired(hv_inst, TREAT, "I_rand")',
     "TestAIGTargets"),
    ("M5 目标量读数改用盒 HV",
     b'        hv_inst[r["label"]][r["seed"]] = float(r["final_hv"])',
     b'        hv_inst[r["label"]][r["seed"]] = 0.0',
     "TestAIGTargets"),
]


def run(cls):
    r = subprocess.run([sys.executable, "-m", "pytest", TESTS, "-q", "-k", cls],
                       capture_output=True, text=True, cwd=ROOT)
    last = [ln for ln in r.stdout.strip().split("\n")
            if "passed" in ln or "failed" in ln]
    return r.returncode != 0, (last[-1] if last else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只跑名字含该子串的变异")
    args = ap.parse_args()

    with open(SRC, "rb") as fh:
        orig = fh.read()
    fd, bak = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    with open(bak, "wb") as fh:
        fh.write(orig)                       # ★ 二进制备份
    h0 = hashlib.md5(orig).hexdigest()

    rows, ok_all = [], True
    try:
        for name, old, new, cls in MUTATIONS:
            if args.only and args.only not in name:
                continue
            if orig.count(old) != 1:
                rows.append("%-40s [跳过] 锚点命中 %d 次（源码已变，锚点需更新）"
                            % (name, orig.count(old)))
                ok_all = False
                continue
            with open(SRC, "wb") as fh:      # ★ 二进制写入
                fh.write(orig.replace(old, new))
            caught, tail = run(cls)
            rows.append("%-40s 预期锁=%-20s 捕获=%-5s %s"
                        % (name, cls, caught, tail))
            if not caught:
                ok_all = False
            with open(SRC, "wb") as fh:
                fh.write(orig)
    finally:
        with open(SRC, "wb") as fh:
            fh.write(orig)

    h1 = hashlib.md5(open(SRC, "rb").read()).hexdigest()
    os.remove(bak)
    print("\n".join(rows))
    print()
    print("还原后 md5 与备份一致: %s" % (h0 == h1))
    print("全部变异都被捕获: %s" % ok_all)
    return 0 if (ok_all and h0 == h1) else 1


if __name__ == "__main__":
    sys.exit(main())
