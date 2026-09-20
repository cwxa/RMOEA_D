#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查 `paper/main.tex` 里是否残留**已被宏化**的字面数字。

为什么需要它
------------
`paper/tables/macros.tex` 由 `scripts/paper_export.py` 从 `logs/` 现场生成，
正文只应写宏名（如 `\\ProbeRho`）。但"手写数字"的惯性很强：改了数据、
忘了改正文，或干脆没接线，就会出现\textbf{同一指标两套数字}。

这正是本项目的**缺陷 30**：论文 §5 把 `rho=+0.83` 归给式 (2) 定义的 probe，
而 0.83 实际属于另一个事前量（平均 makespan 杠杆），probe 自己是 0.77。
两个数都"真实存在"，所以肉眼看不出错——只有把正文与数据放在一起机械比对才暴露。

判据
----
对 `macros.tex` 里每个**纯数值**宏，若同一个数字串**原文**出现在 `main.tex`
（去掉 `\\input{...}` 行与注释行后），即判为"字面重复"。允许用 `--allow` 豁免
（某些量确实在正文里另有用途，但这种豁免必须显式写在命令行里，留下痕迹）。

用法
----
    python scripts/check_paper_literals.py            # 检查，有重复则 exit 1
    python scripts/check_paper_literals.py -v         # 列出全部宏的比对明细
    python scripts/check_paper_literals.py --allow HKthr
"""

import argparse
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MACROS = os.path.join(ROOT, "paper", "tables", "macros.tex")
MAIN = os.path.join(ROOT, "paper", "main.tex")

RE_MACRO = re.compile(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$")


def load_macros(path=MACROS):
    """返回 {宏名: 值}。文件不存在返回空 dict（分阶段出稿时允许）。"""
    if not os.path.exists(path):
        return {}
    out = {}
    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            m = RE_MACRO.match(line.strip())
            if m:
                out[m.group(1)] = m.group(2)
    return out


def strip_comments_and_inputs(src):
    """去掉注释行与 `\\input{...}` 行——宏名本来就写在正文里，不算重复。

    **注意**：LaTeX 里 `\\%` 是转义百分号（正文里的"百分之"），不是注释起点。
    按第一个 `%` 截断会把整行后半吃掉，于是漏掉真正的重复——本检查器最初就
    栽在这里（`0.81` 明明在正文里，却报"未出现"）。只认**未被反斜杠转义**的 `%`。
    """
    keep = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("%"):
            continue
        if s.startswith("\\input{"):
            continue
        line = re.split(r"(?<!\\)%", line, maxsplit=1)[0]
        keep.append(line)
    return "\n".join(keep)


def numeric_macros(macros):
    """只取纯数值宏，且跳过"过于通用"的小整数。

    `0` / `3` / `10` / `66` 这类值在任何正文里都会自然出现，拿它们做字面比对
    只会刷出一屏假阳性（实测：17 条命中里 14 条是 `0` 和 `10`）。
    真正会漂移、也真正值得比的是**浮点数**与大整数。
    """
    out = {}
    for k, v in macros.items():
        v = v.strip()
        if not re.fullmatch(r"[+-]?\d+(\.\d+)?", v):
            continue
        if re.fullmatch(r"[+-]?\d+", v) and abs(int(v)) < 100:
            continue
        out[k] = v
    return out


def find_literals(main_src, macros, allow=()):
    """返回 [(宏名, 值), ...]：值以字面形式出现、且未被豁免的宏。"""
    body = strip_comments_and_inputs(main_src)
    hits = []
    for k, v in sorted(numeric_macros(macros).items()):
        if k in allow:
            continue
        cand = {v}
        if v[0] in "+-":                      # ±号在正文里常省略
            cand.add(v[1:])
        # 用"两侧不是数字/小数点/符号"界定：避免 66 命中 660，
        # 也避免 `0.003` 命中 `-0.003`（符号方向不同，是两个不同的量）
        for c in sorted(cand):
            if not c or not c[0].isdigit() and c[0] not in "+-":
                continue
            pat = r"(?<![\d.+\-])" + re.escape(c) + r"(?![\d])"
            if re.search(pat, body):
                hits.append((k, v))
                break
    return hits


# ────────────────────── 反向扫描（正文 → 宏） ──────────────────────
# 正向只回答"宏值有没有被手写一遍"；反向回答"正文里的数字有没有宏"。
# 缺陷 39（2026-09-20）就是靠反向才发现的：§5 的 `+0.31%` / `p=0.78` 在正文里
# 存在、量级合理、也无从与任何宏比对——正向检查对它是**完全盲的**。

RE_NUM_ANY = re.compile(r"(?<![\w.\\])(\d+\.\d+|\d{3,})(?![\d])")

# 版式与排版参数：这些数字不是实验结果，不该进比对
_NON_DATA_CTX = ("linewidth", "textwidth", "geometry", "arraystretch",
                 "includegraphics", "hspace", "vspace", "documentclass",
                 "usepackage", "setlength", "tabcolsep", "paperwidth",
                 "paperheight", "graphicspath")

_RE_YEAR = re.compile(r"^(19|20)\d{2}$")


def macro_value_set(macros):
    """宏值的可比较集合（含省略 +/− 的版本）。只收数值型宏。"""
    out = set()
    for v in macros.values():
        v = v.strip().replace("\\%", "").replace("$", "").replace(" ", "")
        if not v or not re.match(r"^[+-]?\d", v):
            continue
        out.add(v)
        if v[0] in "+-":
            out.add(v[1:])
    return out


def find_unmacroed(main_src, macros):
    """反向扫描：正文里出现、却不对应任何宏的小数/大整数。

    返回 ``[(行号, 数字, 上下文), ...]``。跳过注释、``\\input``、版式参数
    与参考文献区——否则会刷出一屏假阳性（页码、版式尺寸、DOI 年份）。

    注意这是**提示性**检查：像 "SHA-256" 这类固定术语里的数字无法机械区分，
    所以默认只报告、不置错，由人复核。
    """
    vals = macro_value_set(macros)
    lines = main_src.splitlines()
    bib = next((i for i, l in enumerate(lines)
                if l.strip().startswith("\\begin{thebibliography}")), None)
    out = []
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if s.startswith("%") or s.startswith("\\input{"):
            continue
        if bib is not None and i - 1 >= bib:
            continue
        if any(k in line for k in _NON_DATA_CTX):
            continue
        line = re.split(r"(?<!\\)%", line, maxsplit=1)[0]
        for mm in RE_NUM_ANY.finditer(line):
            tok = mm.group(1)
            if tok in vals:
                continue
            if "." not in tok and (_RE_YEAR.match(tok) or len(tok) < 3):
                continue
            out.append((i, tok, s[:100]))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--macros", default=MACROS)
    ap.add_argument("--main", default=MAIN)
    ap.add_argument("--allow", default="",
                    help="逗号分隔的宏名白名单（这些宏的值允许在正文里字面出现）")
    ap.add_argument("--reverse", action="store_true",
                    help="同时反向扫描：列出正文里出现、却没有宏对应的数字（缺陷 39）")
    ap.add_argument("--reverse-strict", action="store_true",
                    help="反向扫描有命中即 exit 1（默认只报告，因为固定术语里的"
                         "数字无法机械区分，如 SHA-256）")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    macros = load_macros(args.macros)
    if not macros:
        print("[跳过] %s 不存在（先跑 scripts/paper_export.py）" % args.macros)
        return 0
    if not os.path.exists(args.main):
        print("[跳过] %s 不存在" % args.main)
        return 0
    allow = {s.strip() for s in args.allow.split(",") if s.strip()}
    with io.open(args.main, encoding="utf-8") as fh:
        src = fh.read()

    nums = numeric_macros(macros)
    hits = find_literals(src, macros, allow)
    if args.verbose:
        print("纯数值宏 %d 个；正文有效行 %d 行"
              % (len(nums), len(strip_comments_and_inputs(src).splitlines())))
        print("%-20s %-12s %s" % ("宏", "值", "正文里字面出现"))
        bad = dict(hits)
        for k, v in sorted(nums.items()):
            print("  %-18s %-12s %s" % (k, v, "YES" if k in bad else "no"))
    if hits:
        print("\n[失败] 下列数字在正文里**既有宏又在字面重复**——")
        print("       数字一旦有两处来源，改数据时必然漂移（缺陷 30）：")
        for k, v in hits:
            print("   \\%s = %s    <- 正文里请改用 \\%s" % (k, v, k))
        print("\n       若某个值在正文里确实另有用途，显式加 --allow %s 豁免。"
              % ",".join(k for k, _ in hits))
        return 1
    print("[通过] 正文未出现已宏化数字的字面副本（比对 %d 个纯数值宏）" % len(nums))

    if args.reverse or args.reverse_strict:
        unm = find_unmacroed(src, macros)
        if not unm:
            print("[通过] 反向扫描：正文里的数字都有宏对应")
        else:
            print("\n[提示] 反向扫描发现 %d 处**无宏对应**的数字——"
                  "逐一确认它是实验结果还是版式/术语：" % len(unm))
            for ln, tok, ctx in unm:
                print("   L%-5d %-12s %s" % (ln, tok, ctx))
            print("\n       若确认是实验结果，请让 paper_export.py 现场产出宏，"
                  "正文改用宏名（缺陷 34/39）。")
            if args.reverse_strict:
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
