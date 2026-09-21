"""校验文档里的复现命令**真的能跑**（至少 argparse 愿意接受它）。

为什么需要它：缺陷 19（"文档复现命令未实测"）已经复发过一次。
本轮又发现 `docs/new-arch-report.md` 的 `caliber_audit.py` 调用漏了必填的
`--pairs`——命令一跑就报 `error: the following arguments are required: --pairs`，
而文档上写着"零算力，复用 2400 runs"，读者会以为是自己环境的问题。

**并且要防止判据自己空扫**（缺陷 43 的教训）：检查器必须能证明它"看得见东西"，
所以内置 `--self-test`：喂一个已知坏例子，若检查器不报，它自己失败。

用法：
    python scripts/check_doc_commands.py            # 检查 docs/*.md
    python scripts/check_doc_commands.py --self-test  # 自检（防止空扫）
"""

import io
import os
import re
import glob
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


_INTERP = re.compile(r"^(?:\S*python(?:\.exe)?|\$[A-Za-z_]\w*|\S*py\.exe)$", re.I)


def _is_command_position(prefix):
    """脚本之前的 token 只能是解释器（`python` / `py` / `C:/...python.exe` / `$PY`）。

    这条判断是为了**只认真正可执行的命令**：文档里大量出现
    "（`scripts/caliber_audit.py` 与 `scripts/paper_export.py` 两条独立路径）"
    这类散文提及，以及表格里的脚本清单——它们不是命令，不该被当成命令校验。
    """
    toks = prefix.strip().split()
    return all(_INTERP.match(t) for t in toks)


def script_invocations(text):
    """从一段 markdown 里抽出 `... python scripts/xxx.py ...` 调用（合并 `\\` 续行）。

    只收**命令位置**上的调用（见 `_is_command_position`）。
    返回 ``[(行号, 脚本相对路径, 参数串), ...]``。
    """
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines, 1):
        if "scripts/" not in line:
            continue
        # 合并续行（行尾反斜杠）
        cmd, j = line, i - 1
        while cmd.rstrip().endswith("\\") and j + 1 < len(lines):
            j += 1
            cmd = cmd.rstrip()[:-1] + " " + lines[j]
        for m in re.finditer(r"(scripts/[A-Za-z0-9_]+\.py)(.*)", cmd):
            if not _is_command_position(cmd[:m.start()]):
                continue
            out.append((i, m.group(1), m.group(2)))
    return out


def usage_of(pyexe, script):
    """取脚本的 argparse usage 一行（合并换行）。取不到返回 None。"""
    p = subprocess.run([pyexe, script, "--help"], capture_output=True,
                       text=True, encoding="utf-8", errors="replace",
                       cwd=ROOT)
    txt = (p.stdout or "") + (p.stderr or "")
    m = re.search(r"usage:\s*(.*?)(?:\n\s*\n|\Z)", txt, re.S)
    return " ".join(m.group(1).split()) if m else None


def required_flags(usage):
    """usage 里**不在** `[...]` 中的 `--flag` 就是必填。

    argparse 的 usage 用方括号表示可选，这是它稳定的对外契约。
    """
    if not usage:
        return None
    stripped = re.sub(r"\[[^\]]*\]", " ", usage)
    return set(re.findall(r"(--[a-z][a-z0-9-]*)", stripped))


def supported_flags(usage):
    if not usage:
        return None
    return set(re.findall(r"(--[a-z][a-z0-9-]*)", usage))


def check_text(pyexe, text, origin=""):
    """返回问题列表 ``[(origin, 行号, 脚本, 问题), ...]``。"""
    problems = []
    cache = {}
    for lineno, script, rest in script_invocations(text):
        path = os.path.join(ROOT, script)
        if not os.path.exists(path):
            continue
        if script not in cache:
            u = usage_of(pyexe, script)
            cache[script] = (required_flags(u), supported_flags(u), u)
        req, sup, usage = cache[script]
        if req is None:
            problems.append((origin, lineno, script, "取不到 --help 的 usage（无法校验）"))
            continue
        used = set(re.findall(r"(--[a-z][a-z0-9-]*)", rest))
        used -= {"--help"}          # argparse 恒支持 --help，只是 usage 里写作 [-h]
        missing = sorted(req - used)
        unknown = sorted(used - sup)
        if missing:
            problems.append((origin, lineno, script, "缺少必填旗标 %s" % missing))
        if unknown:
            problems.append((origin, lineno, script, "使用了不支持的旗标 %s" % unknown))
    return problems


SELF_TEST_MD = """
```bash
# 这个脚本需要 --pairs，这里故意不写 -> 检查器必须报出来
$PY scripts/caliber_audit.py --labs logs/ablation_ladder.json
```
"""


def self_test(pyexe):
    """防空扫：喂一个已知坏例子，检查器必须报。"""
    problems = check_text(pyexe, SELF_TEST_MD, origin="<self-test>")
    if not problems:
        print("[失败] 自检未通过：检查器对**已知错误命令**没有报出任何问题"
              "——说明它恒为空扫（缺陷 43 同类）。")
        return 1
    print("[OK] 自检通过：检查器对已知错误命令报了 %d 条 -> %s"
          % (len(problems), problems[0][3]))
    return 0


def main():
    ap = argparse.ArgumentParser(description="校验文档里的复现命令能被 argparse 接受")
    ap.add_argument("--glob", default="docs/*.md", help="待检查的 markdown（默认 docs/*.md）")
    ap.add_argument("--pyexe", default="C:/Python312/python.exe", help="用于取 --help 的解释器")
    ap.add_argument("--self-test", action="store_true", help="自检：确认检查器不是空扫")
    args = ap.parse_args()

    if args.self_test:
        return self_test(args.pyexe)

    files = sorted(glob.glob(os.path.join(ROOT, args.glob)))
    if not files:
        print("[失败] 没有匹配到任何文件：%s" % args.glob)
        return 2

    allbad = []
    n_calls = 0
    for f in files:
        with io.open(f, encoding="utf-8") as fh:
            text = fh.read()
        n_calls += len(script_invocations(text))
        allbad += check_text(args.pyexe, text, origin=os.path.relpath(f, ROOT))

    if allbad:
        print("[失败] 文档里有 %d 条命令跑不起来（共扫 %d 条调用）：" % (len(allbad), n_calls))
        for origin, lineno, script, why in allbad:
            print("   %s:%d  %s  %s" % (origin, lineno, script, why))
        return 1
    print("[OK] %d 个文件、%d 条脚本调用，旗标全部合法" % (len(files), n_calls))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
