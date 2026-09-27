#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文档引用与表述的机械复核（四条判据，带退出码）。

为什么单独有一个
----------------
既有的三个文档闸门各只管一面：
* `check_doc_commands.py` —— 文档里的**脚本命令**旗标是否合法；
* `TestDocsDoNotAssertRetractedCaliberClaims` —— "跨口径可引用"这类**断言**；
* `TestBoxCaliberNumbersAreAnnotated` —— `1.86%` / `0.01658` 两个**盒口径数字**的标注。

于是下面这些"看着在工作、其实没人看"的面**全部漏网**（2026-09-27 复核发现）：

1. **文档指向一个不存在的文件**。实证：`docs/qpas-optimization-plan.md` §8「交付物」
   把 `docs/prereg-qpas-v2.md` 与 `logs/_inst_T_scan.json` 当**已交付**列出，两者都不存在
   —— 读者按图索骥找不到，而且**任何闸门都不报错**。
   （与缺陷 53 同族：覆盖面由"判据恰好怎么写"决定，而不是由"哪些面需要检查"决定。）
2. **markdown 里写 LaTeX 交叉引用**。`§\\ref{sec:deviation}` 在 markdown 里**不渲染**，
   读者看到的是字面文本、**没有节号**。实证 3 处。
3. **把 ε 解释成"探索率"**。论文 Alg.3 第 5–8 行是 `if rand < ε then 取 max Q else 随机`，
   即 **ε 是"利用"的概率**；写成"探索率 / 随机动作概率"就把极性说反了。
   实证：`readme.md` 参数表与 `doc.md` §三.5/§四都写过反的，且与本仓库
   `paper-vs-reproduction.md`「ε 推向最'利用'的一端」**自相矛盾**——同一指标两处相反。

判据设计要点（都是本项目栽过的坑）
----------------------------------
* **覆盖面**：`docs/**/*.md`（递归）+ 仓库**根级** `*.md`。含门面 `readme.md`、`doc.md`
  与 `docs/superpowers/**`（它们直到缺陷 53 都不在任何守卫视野里）。
* **不许因"含空格/参数"而漏检**：`` `python scripts/qpas_audit.py --instance Mk10` ``
  这类 token 要**拆词**后逐词判定，否则最像"可执行命令"的那些反而没人看。
* **不许因"没写通配符"而漏检**：`logs/_mkN_init.json`、`scripts/{a,b}.py` 是**模式**，
  不是路径，跳过；`docs/prereg-qpas-v2.md` 是**具体路径**，必须查。
* **判据按语义族写、并给更正性表述留出白名单**（缺陷 51 的教训）：讲"曾经写反"
  的句子（含 `原实现` / `曾` / `不是` …）要放行，否则会把**正确的勘误**判成错误。

用法：
    python scripts/check_doc_refs.py             # 只报告，缺引用时退出码 1
    python scripts/check_doc_refs.py --self-test # 喂已知坏例，必须全部报出（防空扫）
    python scripts/check_doc_refs.py --no-path   # 跳过第 1 条（大改动时临时用）
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOP_DIRS = ("scripts", "src", "tests", "docs", "paper", "logs",
            "results", "charts", "data", "config", "tools")

# --- 判据 1：反引号 token 里的"具体路径" ---
TOKEN_RE = re.compile(r"`([^`\n]+)`")
# token 内的"词"：空白/逗号/顿号/分号切分
WORD_SPLIT = re.compile(r"[\s,;、]+")
# 表示"模式"而非具体路径的字符 —— 命中即跳过
GLOB_CHARS = ("*", "{", "}", "?", "<", ">", "…", "NN", "N_")
# markdown 链接
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")

# --- 判据 2：markdown 里当交叉引用用的 LaTeX 引用命令 ---
#  只匹配"引用标记"上下文（§ / 表~ / 图~ / 节~ / 式~），
#  以免误伤"在讲 LaTeX 源码"的 `\\label{...}` 与裸 `\\ref{...}`。
LATEX_REF_RE = re.compile(r"(?:§|表~|图~|节~|式~)\s*\\(?:ref|eqref|autoref|Cref|cref)\{")
LATEX_BARE_RE = re.compile(r"\\(?:eqref|autoref|Cref|cref)\{")

# --- 判据 3：ε 极性被说成"探索" ---
EPS_RE = re.compile(r"ε|epsilon|\\\\varepsilon")
EPS_WRONG_RE = re.compile(r"探索率|探索概率|随机动作概率")
# 更正性表述 / 否定 —— 放行（判据按语义族写，但要认出"在讲错的写法"）
NEG_RE = re.compile(r"不是|而非|非是|错|曾|原实现|原先|撤回|勘误|反过|相反|更正|改为")
# "引用"豁免：错误写法若被 **行内代码** 或 **引号** 包住，说明是在**引用**它
#   （勘误框 / 缺陷登记**必须**能写出错的样子，否则无从记录），不是自己主张。
#   这条是本判据上线后**立刻被自己触发**才补上的：
#   记录缺陷 56 的那张表逐行引用 `探索率` / "随机动作概率"，被判据当成新缺陷报出来。
QUOTE_SPANS = (
    re.compile(r"`([^`]*)`"),                  # 行内代码
    re.compile(r"\u201c([^\u201d]*)\u201d"),   # 中文弯引号 “...”
    re.compile(r"\u300c([^\u300d]*)\u300d"),   # 直角引号 「...」
    re.compile(r'"([^"]*)"'),                  # 直双引号 "..."
)

# --- 豁免：**已声明**的缺失不是缺陷，但必须能被审计（报告里打印豁免依据）---
# 文档级：某顶层目录"不入库"的声明 -> 该文档内该前缀的引用放行。
#   `logs/` / `charts/` / `results/` 在本仓库 `.gitignore` 里是**项目级事实**，
#   文档声明过一次即可；这类引用是**合规的溯源**，不是"指向不存在的文件"。
IGNORED_DECL_RE = re.compile(
    r"(logs|charts|results)\s*/?[^\n]{0,60}?(不入库|不进版本库|不纳入版本库|被忽略|gitignore|\.gitignore)")
# 行级：这是"计划 / 未产出"而非"已有"的引用。
PENDING_RE = re.compile(r"未产出|未执行|未生成|尚未|计划|建议|拟新增|将新增|待补|TODO")
# 只有"会入库"的顶层目录才需要逐个文件存在（其余由上面的目录级声明兜住）
GITIGNORED_TOP = ("logs", "charts", "results")
# 递归认定"这是一个文件名"的扩展名 —— 不能只靠"含点"，
# 否则 `scripts/paper_export.box_hv_map`（模块.函数）会被当成路径。
ALLOW_EXT = (".py", ".md", ".json", ".tex", ".txt", ".png", ".pdf", ".fjs",
             ".csv", ".tsv", ".yaml", ".yml", ".toml", ".sh", ".log", ".aux")


def doc_files(root=ROOT):
    """文档全集：`docs/**/*.md` + 仓库根级 `*.md`（与口径锁共用同一覆盖面定义）。"""
    out = []
    for dirpath, _dirs, names in os.walk(os.path.join(root, "docs")):
        for n in names:
            if n.endswith(".md"):
                out.append(os.path.join(dirpath, n))
    for n in os.listdir(root):
        p = os.path.join(root, n)
        if n.endswith(".md") and os.path.isfile(p):
            out.append(p)
    return sorted({os.path.normpath(p) for p in out})


def is_path_word(w):
    """这个词像不像"仓库内相对路径"。

    两个必须同时成立，否则会一屏假阳性：
    * 首段是仓库顶层目录；
    * 末段**以已知扩展名结尾**（`a.py`）或是**已存在的目录**（`charts/schedules/`）。
      只靠"含 `/`"或"含点"会把 `scripts/paper_export.box_hv_map` 这类"模块.函数"
      引用也当成路径。
    """
    if "/" not in w:
        return False
    if w.split("/", 1)[0] not in TOP_DIRS:
        return False
    if any(g in w for g in GLOB_CHARS):
        return False
    stripped = w.rstrip("/")
    tail = stripped.rsplit("/", 1)[-1]
    if not tail.lower().endswith(ALLOW_EXT) and not os.path.isdir(os.path.join(ROOT, w)):
        return False
    return True


def norm_path(w):
    """去掉调用点会带上的后缀：`::TestX`、`()`、结尾标点。"""
    w = w.split("::", 1)[0]
    w = w.split("(", 1)[0]
    w = w.split("#", 1)[0]
    w = re.sub(r"[.,;:：、）)]+$", "", w)
    return w


def _all_quoted(line, pattern):
    """pattern 在 line 里的**全部**匹配是否都落在引号/行内代码内。"""
    hits = list(pattern.finditer(line))
    if not hits:
        return False
    spans = []
    for q in QUOTE_SPANS:
        spans += [m.span(1) for m in q.finditer(line)]
    for m in hits:
        if not any(a <= m.start() and m.end() <= b for a, b in spans):
            return False
    return True


def scan(root=ROOT):
    """返回 (命中, 文档数, 检查路径数, 豁免记录)。"""
    bad = {"path": [], "latex": [], "eps": []}
    exempted = []
    n_checked_path = 0
    n_files = 0

    for path in doc_files(root):
        n_files += 1
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        lines = text.splitlines()

        # 文档级豁免：这份文档是否声明了某个 gitignore 目录"不入库"
        ignored = set()
        for m in IGNORED_DECL_RE.finditer(text):
            ignored.add(m.group(1))
        if ignored:
            exempted.append((rel, "目录级声明：%s 不入库"
                             % "/".join(sorted(ignored))))

        for i, line in enumerate(lines, 1):
            # --- 判据 1：反引号 token 拆词后逐词查 ---
            for m in TOKEN_RE.finditer(line):
                for w in WORD_SPLIT.split(m.group(1)):
                    if not is_path_word(w):
                        continue
                    t = norm_path(w)
                    if not is_path_word(t):
                        continue
                    n_checked_path += 1
                    top = t.split("/", 1)[0]
                    if top in GITIGNORED_TOP and top in ignored:
                        continue                      # 目录级豁免
                    if PENDING_RE.search(line):
                        continue                      # 行级："计划/未产出"
                    if not os.path.exists(os.path.join(root, t)):
                        bad["path"].append((rel, i, t))
            # markdown 链接目标
            for m in LINK_RE.finditer(line):
                tgt = m.group(1)
                if tgt.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                t = norm_path(tgt)
                if not t:
                    continue
                n_checked_path += 1
                if not os.path.exists(os.path.join(root, t)):
                    bad["path"].append((rel, i, t))

            # --- 判据 2：markdown 里的 LaTeX 交叉引用 ---
            #  错误写法若被引号/行内代码包住 -> 是在**引用**它（缺陷登记必须能写出错的样子）
            pat = None
            if LATEX_REF_RE.search(line):
                pat = LATEX_REF_RE
            elif LATEX_BARE_RE.search(line):
                pat = LATEX_BARE_RE
            if pat is not None and not _all_quoted(line, pat):
                bad["latex"].append((rel, i, line.strip()[:100]))

            # --- 判据 3：ε 极性 ---
            if EPS_RE.search(line) and EPS_WRONG_RE.search(line):
                # ① 整行的错误写法都是"引用"（引号/行内代码包住）-> 放行
                if _all_quoted(line, EPS_WRONG_RE):
                    continue
                # ② 附近（本行 + 前 2 行）有更正性表述 -> 放行
                ctx = "\n".join(lines[max(0, i - 2):i])
                if not NEG_RE.search(ctx):
                    bad["eps"].append((rel, i, line.strip()[:100]))
    return bad, n_files, n_checked_path, exempted


def report(bad, n_files, n_checked_path, skip_path=False, exempted=()):
    n = sum(len(v) for v in bad.values())
    print("扫描文档 %d 份；检查路径引用 %d 处；命中问题 %d 处"
          % (n_files, n_checked_path, n))
    if exempted:
        print("已声明豁免 %d 处（可审计，不是静默放行）：" % len(exempted))
        for rel, why in exempted:
            print("  - %s：%s" % (rel, why))
    print()

    def dump(title, rows, why, fix):
        if not rows:
            print("[OK] %s" % title)
            return
        print("[X] %s —— %d 处" % (title, len(rows)))
        print("    为什么算问题：%s" % why)
        print("    怎么修：%s" % fix)
        cur = None
        for rel, ln, what in rows:
            if rel != cur:
                print("    [%s]" % rel)
                cur = rel
            print("      L%-5d %s" % (ln, what))
        print()

    if not skip_path:
        dump("文档指向不存在的文件", bad["path"],
             "读者按图索骥找不到，而既有闸门不报错（缺陷 59）",
             "改成真实路径；确属未产出/临时产物则就地标注")
    else:
        print("[跳过] 路径引用检查（--no-path）\n")

    dump("markdown 里写了 LaTeX 交叉引用", bad["latex"],
         "`\\ref{}` 在 markdown 里**不渲染**，读者看到字面文本、**没有编号**（缺陷 58）",
         "写成真实编号（编号从 `paper/main.aux` 的 `\\newlabel` 读，不要靠数）")

    dump("ε 极性被说成「探索」", bad["eps"],
         "论文 Alg.3 第 5–8 行是 `if rand<ε then 取 max Q else 随机`，ε 是**利用**的概率（缺陷 56）",
         "改为「贪婪因子 / 走取 max Q 的概率」；若在讲错的写法，加「曾/原实现/不是」等更正标记")

    return 0 if n == 0 else 1


def self_test():
    """防空扫（缺陷 43 的教训）：判据必须真的能报出已知坏例。"""
    bad = {
        "path": [("docs/FAKE.md", 1, "docs/does-not-exist.md")],
        "latex": [("docs/FAKE.md", 2, "见 §\\ref{sec:nope}")],
        "eps": [("docs/FAKE.md", 3, "| ε | 0.8 | 探索率 |")],
    }
    print("自检：喂 3 条已知坏例，三条判据都必须报出\n")
    code = report(bad, 1, 3)
    if code != 1:
        print("[X] 自检失败：判据对已知坏例没报错——它恒为空扫")
        return 1

    # 语义族白名单必须生效：更正性表述 / 引号内引用 不得被误判
    ok_lines = [
        "以概率 ε=0.8 取 max Q（利用）——**不是**“探索率”",
        "原实现写成 `rand < ε → 随机`，于是 ε=0.8 变成 80% 随机探索",
        "`readme.md` 把 ε 标成“探索率”",          # 引号内引用错误写法
        "| ① | `readme.md` | `\\| Q-learning ε \\| 0.8 \\| 探索率 \\|` |",   # 行内代码引用
    ]
    print("自检：更正性表述 / 引号内引用必须放行（否则会把正确的勘误判成错误）")
    for ln in ok_lines:
        has_wrong = bool(EPS_RE.search(ln) and EPS_WRONG_RE.search(ln))
        if not has_wrong:
            print("    [--] 不含错误写法，跳过：%s" % ln[:40])
            continue
        neg = bool(NEG_RE.search(ln))
        quoted = _all_quoted(ln, EPS_WRONG_RE)
        if not (neg or quoted):
            print("[X] 自检失败：应放行却会被判错 -> %s" % ln)
            return 1
        print("    [OK] 放行（%s）：%s" % ("更正表述" if neg else "引号内引用", ln[:44]))

    # 反面：裸引用（无引号、无更正标记）必须仍被判错
    bare = "| Q-learning ε | 0.8 | 探索率 |"
    if _all_quoted(bare, EPS_WRONG_RE) or NEG_RE.search(bare):
        print("[X] 自检失败：裸引用被豁免放过了 -> %s" % bare)
        return 1
    print("    [OK] 裸引用仍会被判错：%s" % bare)

    # LaTeX 引用同样：引号内 = 引用（放行），裸写 = 缺陷（判错）
    print("自检：LaTeX 交叉引用也要区分'引用'与'裸写'")
    quoted_ref = "> 3 处：`docs/x.md` 的 `§\\ref{sec:deviation}`。"
    bare_ref = "见 §\\ref{sec:deviation} 一节。"
    if not _all_quoted(quoted_ref, LATEX_REF_RE):
        print("[X] 自检失败：引号内的 LaTeX 引用被误判 -> %s" % quoted_ref)
        return 1
    if _all_quoted(bare_ref, LATEX_REF_RE):
        print("[X] 自检失败：裸写的 LaTeX 引用被豁免放过 -> %s" % bare_ref)
        return 1
    print("    [OK] 引号内放行 / 裸写仍判错")

    print("\n[OK] 自检通过（判据有区分度 + 白名单生效）")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="文档引用与表述的机械复核（路径引用 / LaTeX 交叉引用 / ε 极性）")
    ap.add_argument("--self-test", action="store_true",
                    help="喂已知坏例验证判据有区分度（防空扫）")
    ap.add_argument("--no-path", action="store_true",
                    help="跳过路径引用检查")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    bad, n_files, n_checked, exempted = scan()
    if args.no_path:
        bad["path"] = []
    return report(bad, n_files, n_checked, skip_path=args.no_path,
                  exempted=exempted)


if __name__ == "__main__":
    sys.exit(main())
