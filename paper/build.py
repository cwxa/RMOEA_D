#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""论文构建：导出表格/图 -> 校验依赖 -> 用 Tectonic 编译 PDF。

为什么要有校验这一步
--------------------
`main.tex` 用 `\\input{tables/...}` 引用表格。如果某个表因为数据没跑完而没生成，
XeTeX 会直接报"文件不存在"；更糟的是如果上一轮的旧文件还在，**编译会静默通过，
论文里留着旧数字**——正是 fig6 硬编码事故的同款风险。

所以本脚本在编译前做两件事：
1. 跑 `paper_export.py`（不带 `--allow-missing`，数据缺失就当场失败）；
2. 比对 `main.tex` 里所有 `\\input`/`\\includegraphics` 引用的文件是否都存在且
   **修改时间不早于本轮导出**（防止旧产物混进 PDF）。

用法：
    python paper/build.py
    python paper/build.py --allow-missing      # 分阶段出稿：占位表会显式标注"未就绪"
    python paper/build.py --no-compile         # 只导出 + 校验
"""

import argparse
import io
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TECTONIC = os.environ.get("TECTONIC_EXE", r"E:\tools\tectonic\tectonic.exe")
CACHE = os.environ.get("TECTONIC_CACHE_DIR", r"E:\tools\tectonic-cache")


def run(cmd, cwd=HERE, env=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def scan_log(log):
    """从 LaTeX 日志里数出三类**交付级**告警。

    - `Missing character`：字体缺字，PDF 上直接少字。
    - `Overfull \\hbox`：内容超出版心宽度。
    - `Float too large for page`：浮体（表/图）比整页还高。**这一类不会产生
      Overfull 告警**，LaTeX 只是警告后把浮体强行排出纸张——即内容跑到页面外。
      2026-09-19 实测 `tab_hurink`（66 行单栏）超出 517 pt、`tab_instances`
      超出 97 pt，此前一直没被交付判据拦住（判据只认 "1000pt" 以上的超宽）。
      现在把三类一并计数，交付判据 = 三者全为 0。
    """
    lines = log.split("\n")
    miss = [l for l in lines if "Missing character" in l]
    over = [l for l in lines if "Overfull \\hbox" in l]
    flt = [l for l in lines if "Float too large" in l]
    return miss, over, flt


def referenced(tex):
    """收集 main.tex 里引用的相对路径。"""
    src = io.open(os.path.join(HERE, tex), encoding="utf-8").read()
    deps = re.findall(r"\\input\{([^}]+)\}", src)
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src)
    out = []
    for d in deps:
        out.append(("input", d if d.endswith(".tex") else d + ".tex"))
    for f in figs:
        out.append(("figure", f))
    return out


def resolve(rel):
    """按 main.tex 的 \\graphicspath 与常见子目录解析引用。"""
    cands = [os.path.join(HERE, rel),
             os.path.join(HERE, "figures", rel),
             os.path.join(HERE, "tables", rel)]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--tex", default="main.tex")
    args = ap.parse_args()

    print("=" * 92)
    print("[1/3] 导出表格与图")
    print("=" * 92)
    # 基准时刻必须取在**导出开始之前**：表是在导出过程中逐个写出的，
    # 若拿导出**结束**时刻当基准，导出耗时一旦超过容差（本机约 6–10 s），
    # 开头几张表就会被自己的校验判成"旧产物"——假阳，且会把终版编译挡死
    # （2026-09-19 实测：tab_instances / tab_ladder 被误报）。
    t_export = time.time()
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "paper_export.py")]
    if args.allow_missing:
        cmd.append("--allow-missing")
    rc, out = run(cmd, cwd=ROOT)
    print(out.strip()[-2500:])
    if rc != 0:
        print("\n[失败] 导出阶段返回 %d" % rc)
        return rc

    print("\n" + "=" * 92)
    print("[2/3] 校验 main.tex 的依赖")
    print("=" * 92)
    deps = referenced(args.tex)
    bad = []
    for kind, rel in deps:
        p = resolve(rel)
        if p is None:
            bad.append((kind, rel, "不存在"))
            continue
        if kind == "input" and os.path.getmtime(p) < t_export - 5:
            bad.append((kind, rel, "是旧产物（早于本轮导出）"))
            continue
        if kind == "input":
            # 端到端守卫（缺陷 31）：表里若带 TAB/换页符，说明 LaTeX 命令被写成了
            # 单反斜杠，PDF 上会排出 "extbf{...}" / "ootnotesize" 这类垃圾文本。
            # 编译不会报错，所以只能在这里拦。
            src = io.open(p, encoding="utf-8").read()
            ctl = sorted({c for c in src if c in "\t\f\r\v\a\b\0"})
            if ctl:
                bad.append((kind, rel, "含控制字符 %s（缺陷 31：LaTeX 命令单反斜杠）"
                            % ", ".join("U+%04X" % ord(c) for c in ctl)))
    for kind, rel, why in bad:
        print("  [!!] %-8s %s -> %s" % (kind, rel, why))
    if bad:
        print("\n[失败] 有 %d 个依赖不满足；**不要**用当前产物编译——"
              "会把旧数字带进 PDF。" % len(bad))
        return 2
    print("  全部依赖就绪（%d 个）" % len(deps))

    print("\n" + "=" * 92)
    print("[2b/3] 数字一致性（双向扫描）")
    print("=" * 92)
    # 缺陷 30/32/34 的正向守卫（宏值不许在正文里字面重现）+ 缺陷 39 的反向扫描
    # （正文里的数字有没有宏）。反向结果**只报告不阻塞**：`SHA-256`、计时这类
    # 数字无法机械区分，但每次构建都打印出来，就不会再烂在暗处——
    # 缺陷 39 正是靠它才被发现的（正向检查对"+0.31%"完全盲）。
    rc2, out2 = run([sys.executable, os.path.join(ROOT, "scripts",
                                                  "check_paper_literals.py"),
                     "--reverse"], cwd=ROOT)
    print(out2.strip()[-2000:])
    if rc2 != 0:
        print("\n[失败] 数字一致性守卫未通过：正文与宏有两处来源（缺陷 30/32/34/39）")
        return rc2

    if args.no_compile:
        return 0

    print("\n" + "=" * 92)
    print("[3/3] 编译 " + args.tex)
    print("=" * 92)
    if not os.path.exists(TECTONIC):
        print("[失败] 找不到 %s；把 TECTONIC_EXE 指到 tectonic 可执行文件。" % TECTONIC)
        return 3
    env = dict(os.environ)
    env["TECTONIC_CACHE_DIR"] = CACHE
    env["PYTHONIOENCODING"] = "utf-8"
    rc, out = run([TECTONIC, "-X", "compile", args.tex, "--keep-logs"], env=env)
    tail = [l for l in out.split("\n") if l.strip()][-12:]
    print("\n".join(tail))
    pdf = os.path.join(HERE, os.path.splitext(args.tex)[0] + ".pdf")
    if rc != 0 or not os.path.exists(pdf):
        print("\n[失败] 编译返回 %d" % rc)
        return rc
    log = io.open(os.path.join(HERE, os.path.splitext(args.tex)[0] + ".log"),
                  encoding="utf-8", errors="replace").read()
    miss, over, flt = scan_log(log)
    print("\n[OK] %s  %.1f KB  缺字 %d / 超宽 %d / 浮体过大 %d"
          % (os.path.basename(pdf), os.path.getsize(pdf) / 1024,
             len(miss), len(over), len(flt)))
    for l in (miss + over + flt)[:8]:
        print("   ", l[:150])
    return 0


if __name__ == "__main__":
    sys.exit(main())
