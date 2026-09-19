#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""反向扫描：列出 `paper/main.tex` 里**尚未宏化**的小数（手写数字）。

与 `check_paper_literals.py` 互补，方向相反
-------------------------------------------
* `check_paper_literals.py`：**宏 → 正文**。某个宏的值又原样出现在正文里 → 两处来源（缺陷 30/32）。
* 本脚本：**正文 → 宏**。正文里出现、却**没有任何宏**与之对应的小数 → 它是手写的，
  数据变了它不会跟着变。这一方向**没有别的办法发现**（缺陷 34 的 §4.1 阶梯数字、
  §4.2 响应面数字、§4.3 目标 A/B 区间就是这么被找出来的）。

预期内的输出（不算问题）
------------------------
版面/设置类常量天然不会有宏，看到它们属正常：
`0.52/0.86`（`\\includegraphics` 宽度）、`1.08/2.4/2.5/2.6`（`geometry` / `arraystretch`）、
`1.02`（`ref` 参考点，出现多次）。判据是：**这个数会不会随 `logs/` 里的数据变**。

用法
----
    python scripts/scan_paper_numbers.py            # 列出未宏化的小数（含上下文）
    python scripts/scan_paper_numbers.py -v         # 同时列出所有宏值
"""

import argparse
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BS = chr(92)


def load_macro_values(path):
    """返回 [(宏名, 值), ...]。"""
    out = []
    if not os.path.exists(path):
        return out
    for line in io.open(path, encoding="utf-8"):
        s = line.strip()
        if s.startswith(BS + "newcommand{"):
            name = s.split("{")[1].split("}")[0].lstrip(BS)
            out.append((name, s[s.index("}{") + 2:-1]))
    return out


def body_without_macros(src, names):
    """去掉注释行、`\\input` 行与所有宏名，剩下的才是"人写的正文"。"""
    keep = []
    for ln in src.splitlines():
        s = ln.strip()
        if s.startswith("%") or s.startswith(BS + "input{"):
            continue
        # 只认**未被反斜杠转义**的 % 为注释起点（`\%` 是正文里的百分号）
        keep.append(re.split(r"(?<!\\)%", ln, maxsplit=1)[0])
    body = "\n".join(keep)
    for n in sorted(names, key=len, reverse=True):
        body = body.replace(BS + n, " ")
    return body


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", default=os.path.join(ROOT, "paper", "main.tex"))
    ap.add_argument("--macros",
                    default=os.path.join(ROOT, "paper", "tables", "macros.tex"))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.main):
        print("[跳过] %s 不存在" % args.main)
        return 0
    vals = load_macro_values(args.macros)
    if not vals:
        print("[跳过] %s 不存在（先跑 scripts/paper_export.py）" % args.macros)
        return 0

    known = set()
    for _n, v in vals:
        known.update(re.findall(r"\d+\.\d+", v))
    body = body_without_macros(io.open(args.main, encoding="utf-8").read(),
                               [n for n, _ in vals])

    seen = {}
    for m in re.finditer(r"(?<![\w.])(\d+\.\d+)", body):
        seen.setdefault(m.group(1), []).append(
            body[max(0, m.start() - 45):m.end() + 25])

    if args.verbose:
        print("宏 %d 个" % len(vals))
    miss = {k: v for k, v in seen.items() if k not in known}
    print("正文（去宏后）不同小数值 %d 个；其中**无宏对应**的 %d 个："
          % (len(seen), len(miss)))
    for k in sorted(miss, key=float):
        print("\n  %-10s  x%d" % (k, len(miss[k])))
        for ctx in miss[k][:2]:
            print("      ...%s..." % ctx.replace("\n", " "))
    print("\n提示：版面/设置类常量（图片宽度、geometry、ref 参考点）属预期，"
          "判据是\"这个数会不会随 logs/ 里的数据变\"。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
