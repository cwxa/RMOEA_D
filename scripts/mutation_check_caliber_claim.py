#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""变异测试：把**已被撤回的口径断言**写回文档，确认被 `TestDocsDoNotAssertRetractedCaliberClaims` 捕获。

为什么需要它
------------
缺陷 51（2026-09-21）正是"锁没有区分度"的产物：该锁原本只认一种**句式**
（`不受口径影响`），而同一论断的现代变体是**肯定式**
（"只有 ΔHV / p / wins **可跨批次安全比较**"），于是三条断言全部漏网。

本脚本把这三句写回文档，验证加固后的锁确实能抓住它们；这对任何"删掉一句限制语"
的回归都是必要的 —— 光看"测试全绿"不知道锁是不是空的。

做法与 `mutation_check_aig.py` 一致：**二进制**读写（Windows 文本模式会把 `\\n`
翻成 `\\r\\n`，"还原"本身就在改文件）、锚点必须唯一、结束前断言 md5 与备份一致。

用法：
    python scripts/mutation_check_caliber_claim.py
    python scripts/mutation_check_caliber_claim.py --only cross
"""
import argparse
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests", "test_refactor.py")
LOCK = "TestDocsDoNotAssertRetractedCaliberClaims"

# (名字, 相对路径, 正确的修订文本(锚点), 被撤回的旧断言, 预期捕获它的测试类)
MUTATIONS = [
    ("M1 paper-vs-reproduction 退回'只有 ΔHV/p/wins 可跨批次比'",
     "docs/paper-vs-reproduction.md",
     "故 **ΔHV / p / wins 也不得跨盒、跨口径引用**（缺陷 25/30/39）——只有**同一盒内**可比。",
     "只有 ΔHV / p / wins 可跨批次安全比对。",
     LOCK),
    ("M2 qpas-plan 口径提示退回'跨批只引用 ΔHV/p/wins'",
     "docs/qpas-optimization-plan.md",
     "**换盒不止改绝对值，也改 ΔHV / p / wins**（缺陷 25/39）：跨批只有**同一盒内**可比，",
     "跨批只引用 ΔHV / p / wins；",
     LOCK),
    ("M3 qpas-plan C1 行退回'只引用...可跨批次安全比较'",
     "docs/qpas-optimization-plan.md",
     "**同一盒内可比；跨盒、跨口径一律不可引用——包括 ΔHV / p / wins（缺陷 25）；绝对 HV 另须带盒指纹**",
     "**只引用 ΔHV/p/wins 可跨批次安全比较；绝对 HV 必须带盒**",
     LOCK),
    # 脚本侧：同一族断言若写进源码（尤其是**运行时会打印**的字符串，缺陷 41），
    # 由 TestNoStaleCaliberClaims 用 AST 扫字符串常量拦截。
    ("M4 hv_box.py 口径警告退化成'只有 ΔHV/p/wins 可跨批次比'",
     "scripts/hv_box.py",
     "**臂集**与**口径**，跨批次引用一律无效。",
     "**臂集**与**口径**；只有 ΔHV / p / wins 可跨批次安全比较。",
     "TestNoStaleCaliberClaims"),
]


def run(cls):
    r = subprocess.run([sys.executable, "-m", "pytest", TESTS, "-q", "-k", cls],
                       capture_output=True, text=True, cwd=ROOT,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    last = [ln for ln in out.strip().split("\n")
            if "passed" in ln or "failed" in ln]
    return r.returncode != 0, (last[-1] if last else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只跑名字含该子串的变异")
    args = ap.parse_args()

    # 备份所有涉及的文件（二进制），逐个还原
    targets = {}
    for _name, rel, _old, _new, _cls in MUTATIONS:
        p = os.path.join(ROOT, rel)
        if p not in targets:
            with open(p, "rb") as fh:
                targets[p] = fh.read()

    orig_md5 = {p: hashlib.md5(b).hexdigest() for p, b in targets.items()}

    rows, ok_all = [], True
    try:
        for name, rel, old, new, cls in MUTATIONS:
            if args.only and args.only not in name:
                continue
            p = os.path.join(ROOT, rel)
            base = targets[p]
            a = old.encode("utf-8")
            b = new.encode("utf-8")
            if base.count(a) != 1:
                rows.append("%-52s [跳过] 锚点命中 %d 次（文档已变，锚点需更新）"
                            % (name, base.count(a)))
                ok_all = False
                continue
            with open(p, "wb") as fh:
                fh.write(base.replace(a, b))
            caught, tail = run(cls)
            with open(p, "wb") as fh:          # 立即还原
                fh.write(base)
            rows.append("%-52s 捕获=%-5s %s" % (name, caught, tail))
            if not caught:
                ok_all = False
    finally:
        for p, b in targets.items():
            with open(p, "wb") as fh:
                fh.write(b)

    bad_md5 = [os.path.relpath(p, ROOT) for p, b in targets.items()
               if hashlib.md5(open(p, "rb").read()).hexdigest() != orig_md5[p]]
    print("\n".join(rows))
    print()
    print("还原后 md5 与备份一致: %s" % (not bad_md5))
    print("全部变异都被捕获: %s" % ok_all)
    return 0 if (ok_all and not bad_md5) else 1


if __name__ == "__main__":
    sys.exit(main())
