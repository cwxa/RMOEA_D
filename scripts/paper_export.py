#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""论文导出器：把 `logs/` 下的结果现场转成 LaTeX 表格与 PDF 图。

为什么单独一个脚本
------------------
项目纪律（缺陷 19 / fig6 硬编码事故）：**论文里的每个数字都必须来自结果文件，
取不到就报错，绝不留旧值**。所以这里不做任何数值的字面量兜底：
所有表格单元、所有图上的点都是从 `logs/*.json` 现算的，缺一个数据源就 `raise`
（分阶段出稿时显式加 `--allow-missing`）。

口径纪律（缺陷 22 / 25 / 26）——本脚本的全部读数只有两个来源：
* **实例边界口径** = lab 文件里的 `final_hv` 字段（算法落盘时已用
  `instance_hv_bounds(instance)` 归一化，**与参与比较的臂集无关**）。
  报效应量与显著性一律用它。
* **盒口径** = 用 `estimate_hv_bounds(fronts)` 从 `final_pf` 重算。
  **只用于展示"口径会放大多少"这个对照本身**，正文任何结论都不引用它。
  而且**逐臂对**重算边界（臂集恰好等于参与比较的那两个臂）——
  否则就是缺陷 18 的重演：把无关臂放进盒里会撑大上界。

产出
----
    paper/tables/*.tex     完整 float（\\begin{table}…\\end{table}），主文件 \\input 即可
    paper/figures/*.pdf    矢量图（同时落一份 .png 供人眼抽查）
    logs/paper_export.json 本脚本读到的全部原始数字（审计用）

用法：
    python scripts/paper_export.py
    python scripts/paper_export.py --allow-missing   # Hurink 未跑完时先出 Mk 部分
"""

import argparse
import collections
import io
import json
import os
import re
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REF = (1.02, 1.02)
TAB = os.path.join(ROOT, "paper", "tables")
FIG = os.path.join(ROOT, "paper", "figures")

C_RED = "#c0392b"
C_GREEN = "#1e8449"
C_BLUE = "#2e5c8a"
C_GREY = "#7f8c8d"

LADDER_STEPS = [("D2|D1", "D2", "D1", "MIX3 初始化"),
                ("D3|D2", "D3", "D2", "随机 VNS"),
                ("D4|D3", "D4", "D3", "Q-PAS"),
                ("D5|D4", "D5", "D4", "精英档案"),
                ("RMOEAD|D5", "RMOEAD", "D5", "RL 选算子")]


# ────────────────────────────── 通用 ──────────────────────────────
def paired(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    d = a - b
    if n == 0:
        return None
    if np.allclose(d, 0.0) or n < 3:
        return {"d": float(d.mean()), "rel_pct": float("nan"), "wins": 0,
                "n": int(n), "p": 1.0, "dz": 0.0, "mean_b": float(b.mean()),
                "all_zero": True}
    sd = float(d.std(ddof=1))
    return {"d": float(d.mean()),
            "rel_pct": float(d.mean() / b.mean() * 100.0) if b.mean() else float("nan"),
            "wins": int((d > 0).sum()), "n": int(n),
            "p": float(stats.wilcoxon(d).pvalue),
            "dz": float(d.mean() / sd) if sd > 0 else 0.0,
            "mean_b": float(b.mean()), "all_zero": False}


def stars(p):
    if p is None or not np.isfinite(p):
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""


def ptex(p):
    """$p$ 值的 LaTeX（自带数学模式）；星号用上标附加。"""
    if p is None or not np.isfinite(p):
        return "--"
    s = stars(p)
    if p < 1e-4:
        body = "<10^{-4}"
    elif p < 1e-2:
        e = int(np.floor(np.log10(p)))
        m = p / 10.0 ** e
        body = "%.1f\\times 10^{%d}" % (m, e)
    else:
        body = "%.3f" % p
    return "$%s{}^{%s}$" % (body, s) if s else "$%s$" % body


def num(x, nd=3, signed=True):
    if x is None or not np.isfinite(x):
        return "--"
    return ("%+.*f" if signed else "%.*f") % (nd, x)


def need(path, allow_missing, what):
    if os.path.exists(path):
        return True
    msg = "缺少数据源 %s（%s）" % (os.path.relpath(path, ROOT), what)
    if allow_missing:
        print("  [!] " + msg + " -> 跳过")
        return False
    raise SystemExit("[错误] " + msg + "\n        先跑出该结果，或加 --allow-missing 只出已有部分。")


def _check_tex_clean(name, body):
    """驻留守卫（缺陷 31）：生成的 .tex 里**不许有控制字符**。

    成因：Python 普通字符串里 `\\t` 是 TAB、`\\f` 是换页符。把 `"\\footnotesize"` /
    `"\\textbf{...}"` 写成单反斜杠**不会报错**，只会静默产出 `<FF>ootnotesize`
    与 `<TAB>extbf{...}` —— 编译照过、缺字告警为 0，PDF 上却排出
    "ootnotesize"、"extbf{逐实例}" 这类垃圾文本（2026-09-19 实测 6 张表全中）。
    `ast.parse` 拦不住，只有查输出才拦得住，所以守卫放在写盘前。
    """
    bad = [c for c in ("\t", "\f", "\r", "\v", "\a", "\b", "\0") if c in body]
    if not bad:
        return
    names = ", ".join("U+%04X" % ord(c) for c in bad)
    ctx = []
    for i, ch in enumerate(body):
        if ch in ("\t", "\f", "\r", "\v", "\a", "\b", "\0"):
            ctx.append("      ...%s..." % body[max(0, i - 34):i + 22]
                       .replace(ch, "[%s]" % ("U+%04X" % ord(ch)))
                       .replace("\n", "|"))
            if len(ctx) >= 4:
                break
    raise SystemExit(
        "[错误] %s 里有控制字符 %s —— LaTeX 命令被写成了单反斜杠（缺陷 31）。\n"
        "%s\n"
        "        修法：把源码里对应的单反斜杠命令改成双反斜杠，"
        "可跑 scripts/_fix_tex_escapes.py --apply。" % (name, names, "\n".join(ctx)))


def write_tex(name, body):
    _check_tex_clean(name, body)
    os.makedirs(TAB, exist_ok=True)
    with io.open(os.path.join(TAB, name), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(body)
    print("  -> paper/tables/%s" % name)


def save_fig(fig, name):
    """落盘一张图（同时出一份 .png 供人眼抽查）。

    PDF 里**不写创建时间**：matplotlib 默认每次都会写入当前时间，于是重跑一次
    导出就把全部图标记成"已修改"。真实的图变化会淹没在这堆时间戳噪声里——
    二进制没法逐行 diff，没人会去逐张核对，改动就悄悄漏过去了。
    """
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", pad_inches=0.02,
                metadata={"CreationDate": None})
    fig.savefig(os.path.join(FIG, name.replace(".pdf", ".png")), dpi=170,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  -> paper/figures/%s (+ .png)" % name)


def table_wrap(caption, label, body, colspec, notes=None, pos="t",
               font="\\small", colsep=None):
    tab = ["  \\begin{tabular}{%s}" % colspec, "    \\toprule", body,
           "    \\bottomrule", "  \\end{tabular}"]
    if colsep is not None:
        # 列数多的表用默认 tabcolsep 会溢出页边（Overfull \\hbox）。用组把
        # \\setlength 限在表内，并且把整个 tabular 包在组里，避免泄漏到后文。
        tab = ["  {\\setlength{\\tabcolsep}{%spt}%%" % colsep] + tab + ["  }"]
    out = ["\\begin{table}[%s]" % pos, "  \\centering",
           "  \\caption{%s}" % caption, "  \\label{%s}" % label,
           "  %s" % font] + tab
    if notes:
        out.append("  \\par\\vspace{3pt}\\footnotesize\\begin{minipage}"
                   "{\\linewidth}%s\\end{minipage}" % notes)
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def load_json(path):
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def hv_map(rows, labels):
    out = collections.defaultdict(dict)
    for r in rows:
        if r.get("label") in labels and r.get("final_hv") is not None:
            out[(r["label"], r["instance"])][r["seed"]] = float(r["final_hv"])
    return out


def box_hv_map(rows, pair):
    """盒口径：**只用这一对臂**在每个实例内重算边界（臂集 == 参与比较的臂）。"""
    from rmoea_d.utils.metrics import compute_hv, estimate_hv_bounds
    per = collections.defaultdict(list)
    for r in rows:
        if r.get("label") in pair and r.get("final_pf"):
            per[r["instance"]].append(r)
    out = collections.defaultdict(dict)
    for inst, sub in per.items():
        fronts = [np.asarray(r["final_pf"], float) for r in sub]
        lo, hi = estimate_hv_bounds(fronts)
        for r in sub:
            out[(r["label"], inst)][r["seed"]] = compute_hv(
                np.asarray(r["final_pf"], float), ref_point=REF, norm_bounds=(lo, hi))
    return out


def pair_over(hv, a, b, instances):
    inst_rows, diffs = [], []
    for inst in instances:
        m1, m2 = hv.get((a, inst)), hv.get((b, inst))
        if not m1 or not m2:
            continue
        ss = sorted(set(m1) & set(m2))
        if not ss:
            continue
        st = paired([m1[s] for s in ss], [m2[s] for s in ss])
        st["instance"] = inst
        inst_rows.append(st)
        diffs.extend([m1[s] - m2[s] for s in ss])
    if not inst_rows:
        return None
    rel = np.array([r["rel_pct"] for r in inst_rows], float)
    rel = rel[np.isfinite(rel)]
    return {"per_instance": inst_rows, "n_inst": len(inst_rows),
            "rel_mean": float(rel.mean()) if rel.size else float("nan"),
            "wins_inst": int(sum(1 for r in inst_rows if r["d"] > 0)),
            "wins_run": int(sum(r["wins"] for r in inst_rows)),
            "n_run": int(sum(r["n"] for r in inst_rows)),
            "p_inst": float(stats.wilcoxon(rel).pvalue) if rel.size >= 3
            and not np.allclose(rel, 0) else 1.0,
            "dz_all": (paired(diffs, np.zeros(len(diffs)))["dz"] if diffs else float("nan"))}


# ────────────────────────────── 1. 实例集 ──────────────────────────────
def tab_instances(data, allow_missing):
    """实例集特征表。**必须排成两栏**：开发集 10 行 + Hurink 分组约 30 行，
    单栏时整表比 A4 文本区还高，LaTeX 会报 "Float too large for page"
    并把表强行排出页面（2026-09-19 实测超出 97 pt）。这里统一用 12 列
    （左右各 6 列）续排，(a) 段右侧留空。
    """
    from rmoea_d.core.instance import load_instance
    rows_mk = []
    for i in range(1, 11):
        n = "Mk%02d" % i
        inst = load_instance(n, os.path.join(ROOT, "data"), 42)
        rows_mk.append((n, inst["n_jobs"], inst["n_machines"], inst["total_ops"],
                        _flex(inst), _pt_cv(inst)))
    BLANK6 = " & ".join([""] * 6)
    # 表头必须**逐字段**包 \textit{}：把整串（含 &）包进一个 \textit{} 会让
    # & 出现在命令参数内部，LaTeX 直接报 "Misplaced alignment tab"。
    HEAD6 = " & ".join("\\textit{%s}" % x
                       for x in ("实例", "工件", "机器", "工序", "弹性比", "$c_v$"))
    HEAD6H = " & ".join("\\textit{%s}" % x
                        for x in ("实例数", "工件", "机器", "工序", "弹性比", ""))
    body = ["    \\multicolumn{12}{l}{\\textit{(a) Brandimarte Mk01--Mk10："
            "复现与消融的开发集}}\\\\", "    \\cmidrule(lr){1-12}",
            "    %s & %s \\\\" % (HEAD6, HEAD6),
            "    \\midrule"]
    for n, j, m, o, fx, cv in rows_mk:
        body.append("    %s & %d & %d & %d & %.2f & %.2f & %s \\\\"
                    % (n, j, m, o, fx, cv, BLANK6))

    meta = None
    if need(os.path.join(ROOT, "data", "hurink", "PROVENANCE.json"), allow_missing,
            "Hurink provenance"):
        meta = load_json(os.path.join(ROOT, "data", "hurink", "PROVENANCE.json"))
    n_hed = 0
    notes = ("弹性比 = 每道工序可选机器数的均值；$c_v$ = 模糊三角加工时间的变异系数。"
             "Mk01--Mk10 用于复现、消融与\\textbf{门限标定}。")
    if meta:
        grp = collections.defaultdict(int)
        for k, v in meta.items():
            if isinstance(v, dict) and "shape" in v:
                s = v["shape"]
                grp[(s["n_jobs"], s["n_machines"], s["total_ops"], s["flex_mean"])] += 1
        n_hed = sum(grp.values())
        cells = ["$\\times$%d & %d & %d & %d & %.2f &" % (c, j, m, o, fx)
                 for (j, m, o, fx), c in sorted(grp.items(), key=lambda kv: kv[0][2])]
        half = (len(cells) + 1) // 2
        left, right = cells[:half], cells[half:]
        body.append("    \\midrule")
        body.append("    \\multicolumn{12}{l}{\\textit{(b) Hurink $e$-data："
                    "%d 个实例，全部用作\\textbf{独立留出集}（不挑选）}}\\\\" % n_hed)
        body.append("    \\cmidrule(lr){1-12}")
        body.append("    %s & %s \\\\" % (HEAD6H, HEAD6H))
        for k in range(half):
            r = right[k] if k < len(right) else BLANK6
            body.append("    %s & %s \\\\" % (left[k], r))
        ops = sorted(v["shape"]["total_ops"] for v in meta.values()
                     if isinstance(v, dict) and "shape" in v)
        notes += ("Hurink $e$-data 共 %d 个实例，规模 %d--%d 道工序；"
                  "(b) 段左右两栏续排，按 (工件, 机器, 工序) 分组合并，$\\times c$ 为组内实例数。"
                  "全部实例都进留出集，\\textbf{没有任何按结果的事后挑选}。"
                  % (n_hed, ops[0], ops[-1]))
    else:
        body.append("    \\midrule\n    \\multicolumn{12}{l}{\\textit{"
                    "Hurink 留出集数据未就绪}}\\\\")
    write_tex("tab_instances.tex", table_wrap(
        "实例集特征：(a) 开发集 Brandimarte Mk01--Mk10（只占左栏）；"
        "(b) 留出集 Hurink $e$-data，按 (工件, 机器, 工序) 分组合并后"
        "续排为左右两栏",
        "tab:instances", "\n".join(body),
        "lrrrrr@{\\hspace{10pt}}lrrrrr", font="\\footnotesize", colsep=4,
        notes=notes))
    return {"brandimarte": rows_mk, "n_hurink": n_hed}


def _flex(inst):
    vs = inst["valid_machines"]
    tot = sum(len(v) for job in vs for v in job)
    cnt = sum(1 for job in vs for v in job)
    return tot / cnt if cnt else float("nan")


def _pt_cv(inst):
    vals = []
    for job in inst["jobs"]:
        for op in job:
            for (_m, t1, t2, t3) in op:
                vals.extend([t1, t2, t3])
    v = np.asarray(vals, float)
    return float(v.std() / v.mean()) if v.mean() else float("nan")


# ────────────────────────────── 2. 组件阶梯 ──────────────────────────────
def tab_ladder(data, allow_missing):
    p = os.path.join(ROOT, "logs", "ablation_ladder.json")
    if not need(p, allow_missing, "消融阶梯（8 臂 × 10 实例 × 30 seeds）"):
        return None
    rows = load_json(p)
    labels = sorted({r["label"] for r in rows})
    instances = sorted({r["instance"] for r in rows})
    hi = hv_map(rows, labels)
    res, body, dzs = {}, [], {}
    for key, a, b, comp in LADDER_STEPS:
        si = pair_over(hi, a, b, instances)
        if si is None:
            continue
        hbox = box_hv_map(rows, (a, b))
        sb = pair_over(hbox, a, b, instances)
        amp = (sb["rel_mean"] / si["rel_mean"] if abs(si["rel_mean"]) > 1e-12
               else float("nan"))
        res[key] = {"inst": si, "box": sb, "amp": amp, "component": comp}
        body.append("    %s & %s & %s & %d/%d & %s & %s & %s & %s \\\\"
                    % (key.replace("|", "$|$"), comp, num(si["rel_mean"]),
                       si["wins_inst"], si["n_inst"], num(si["dz_all"], 2),
                       ptex(si["p_inst"]), num(sb["rel_mean"]),
                       ("%.1f$\\times$" % amp) if np.isfinite(amp) else "--"))
        dzs[key] = si["dz_all"]
    write_tex("tab_ladder.tex", table_wrap(
        "组件阶梯的逐级增量：实例边界口径（正文引用）与盒口径（仅作对照）",
        "tab:ladder",
        "    \\multicolumn{6}{c}{\\textbf{实例边界口径}} & "
        "\\multicolumn{2}{c}{\\textbf{盒口径}} \\\\\n"
        "    \\cmidrule(lr){1-6}\\cmidrule(lr){7-8}\n"
        "    阶梯 & 组件 & $\\Delta$HV 相对增幅(\\%) & 实例胜出 & $d_z$ & $p$ "
        "& 相对增幅(\\%) & 放大 \\\\\n"
        "    \\midrule\n" + "\n".join(body),
        "llrrlrrr", font="\\footnotesize", colsep=4.5,
        notes=("Mk01--Mk10，每格 $n=30$ seeds 同 seed 配对。"
               "$\\Delta$HV 相对增幅为\\textbf{逐实例} $\\overline{\\Delta\\mathrm{HV}}/"
               "\\overline{\\mathrm{HV}}_{\\text{上一级}}$ 的跨实例均值"
               "（不用 $\\overline{\\Delta/\\mathrm{HV}}$：效应$\\approx 0$ 时会反号）。"
               "$d_z$ 为 run 级池化配对效应量（$n=300$），$p$ 为实例级 Wilcoxon（$n=10$）。"
               "盒口径一列是\\textbf{同一批数据}换口径重算的结果，用来量化口径的影响；"
               "正文的效应量一律取实例边界口径。")))
    data["ladder"] = {k: {"rel_inst": v["inst"]["rel_mean"], "rel_box": v["box"]["rel_mean"],
                          "amp": v["amp"], "wins": "%d/%d" % (v["inst"]["wins_inst"],
                                                              v["inst"]["n_inst"]),
                          "p": v["inst"]["p_inst"], "dz": v["inst"]["dz_all"]}
                      for k, v in res.items()}
    return res


# ────────────────────────────── 3. 算力响应 ──────────────────────────────
def fig_surface(data, allow_missing):
    p = os.path.join(ROOT, "logs", "anytime_g2000.surface.json")
    if not need(p, allow_missing, "算力–算法响应面（anytime，G 上限 2000）"):
        return None
    d = load_json(p)
    insts = d["instances"]
    pairs = ["D2|D1", "D5|D2", "RMOEAD|D5"]
    name = {"D2|D1": "MIX3 init ($D2|D1$)", "D5|D2": "Elite archive ($D5|D2$)",
            "RMOEAD|D5": "RL operator (RMOEAD$|$D5)"}
    col = {"D2|D1": C_RED, "D5|D2": C_BLUE, "RMOEAD|D5": C_GREEN}
    x = np.array(SURF_BUDGETS)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for key in pairs:
        if key not in d["wall"]:
            continue
        ys = np.asarray([[w["diff"] for w in d["wall"][key][i]] for i in insts], float)
        zs = np.asarray([[w["dz"] for w in d["wall"][key][i]] for i in insts], float)
        # 横轴长度必须与数据一致：`anytime_g2000.surface.json` 里换掉预算档
        # 而没同步 `SURF_BUDGETS`，`plot()` 会静默按较长者截断/补点，图仍能出。
        if ys.shape[1] != len(x):
            raise SystemExit("[错误] %s 的预算档有 %d 个，SURF_BUDGETS 有 %d 个——"
                             "两者必须同源" % (key, ys.shape[1], len(x)))
        axes[0].plot(x, ys.mean(axis=0), "o-", color=col[key], lw=1.4, ms=3.4, label=name[key])
        axes[1].plot(x, zs.mean(axis=0), "o-", color=col[key], lw=1.4, ms=3.4, label=name[key])
    axes[0].set_ylabel("$\\Delta$HV (abs.)", fontsize=7.4)
    axes[0].set_title("(a) raw effect", fontsize=7.8)
    axes[1].set_ylabel("paired $d_z$", fontsize=7.4)
    axes[1].set_title("(b) standardised effect", fontsize=7.8)
    for ax in axes:
        ax.set_xlabel("wall-clock budget fraction", fontsize=7.4)
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.tick_params(labelsize=6.6)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].legend(fontsize=6.2, frameon=False)
    save_fig(fig, "fig_surface.pdf")

    body = []
    for key, a, b, comp in LADDER_STEPS:
        if key not in d["wall"]:
            continue
        cells = []
        for inst in insts:
            w = d["wall"][key][inst][-1]
            cells.append("%s" % num(w["diff"], 3))
        body.append("    %s & %s & %s \\\\" % (key.replace("|", "$|$"), comp,
                                               " & ".join(cells)))
    # 表注里"最大者 = 1.1e-3"原先**写死**：表格数字是现场算的、表注却是手写的，
    # 数据一改就两处不一致（且在生成文件里，看起来比手写正文更权威）。改为现场算。
    _mx, _mxlab = 0.0, "--"
    for key, a, b, comp in LADDER_STEPS:
        if key not in d["wall"] or key == "D2|D1":
            continue
        for inst in insts:
            v = abs(d["wall"][key][inst][-1]["diff"])
            if v > _mx:
                _mx, _mxlab = v, "%s 的 %s" % (inst, comp)
    if _mx > 0:
        _e = int(np.floor(np.log10(_mx)))
        _mant = _mx / (10.0 ** _e)
        _mtxt = ("$%.1f\\times10^{-%d}$" % (_mant, abs(_e)) if _e < 0
                 else "$%.1f$" % _mx)
    else:
        _mtxt = "$0$"
    write_tex("tab_surface.tex", table_wrap(
        "算力全部花完时（100\\% 预算）各组件相对上一级的 $\\Delta$HV 绝对值",
        "tab:surface",
        "    臂对 & 组件 & " + " & ".join(insts) + " \\\\\n    \\midrule\n" + "\n".join(body),
        "ll" + "r" * len(insts),
        notes=("每列一个实例；预算按\\textbf{墙钟}（\\texttt{anytime} 轨迹）分五分位"
               # 预算档与 G 上限都引用正文宏，杜绝"表注写 2000、正文写 \SetGMax"。
               "（\\SurfBudgets，$G$ 上限 \\SetGMax）。把算力加到上限以后，"
               "除初始化以外的组件增量都不超过 " + _mtxt
               + "（最大者为 %s，绝对值）。" % _mxlab)))
    data["surface"] = {k: {i: [w["diff"] for w in d["wall"][k][i]] for i in insts}
                       for k in d["wall"]}
    # 供正文与表注共用同一份文本（表注与 §4.2 正文原先各写一遍 1.1e-3）
    data["surface_max"] = {"txt": _mtxt, "label": _mxlab}
    return d


# ────────────────────────────── 4. 目标 A / B ──────────────────────────────
def tab_targets(data, allow_missing):
    p = os.path.join(ROOT, "logs", "aig_gating.json")
    if not need(p, allow_missing, "AIG 门控（目标 A/B，Mk01--Mk10）"):
        return None
    d = load_json(p)
    tg = d["targets"]
    A = {r["instance"]: r for r in tg["vs_I_rand"]["per_instance"]}
    B = {r["instance"]: r for r in tg["vs_I_mix3"]["per_instance"]}
    insts = sorted(A)
    body = []
    for inst in insts:
        a, b = A[inst], B.get(inst)
        body.append("    %s & %s & %d/%d & %s & %s & %s & %d/%d & %s \\\\"
                    % (inst, num(a["dhv_rel_pct"]), a["wins"], a["n"], num(a["dz"], 2),
                       ptex(a["p"]), num(b["dhv_rel_pct"]), b["wins"], b["n"],
                       ptex(b["p"])))
    nz = [i for i in insts if abs(B[i]["dhv_rel_pct"]) > NZ_EPS]
    # 表注里"10/10 为正、3/10 显著、p>=0.17、最大者 +0.02%"原先全是**写死**的
    # （生成文件里的硬编码比手写正文更危险：它看起来最权威）。改为现场算。
    apos = [i for i in insts if A[i]["dhv_rel_pct"] > 0]
    asig = [i for i in insts if is_gain(A[i]["dhv_rel_pct"], A[i]["p"])]
    bsig = [i for i in insts if is_gain(B[i]["dhv_rel_pct"], B[i]["p"])]
    bns = [i for i in insts if i not in bsig]
    _bns_pmin = min([B[i]["p"] for i in bns], default=1.0)
    # 注意这里取的是"非显著实例中**为正**的最大 ΔHV"，**不是** |ΔHV| 的最大值：
    # 后者是 Mk04 的 **−0.224%**（负值，当"增益"的反例毫无意义）。
    # 这条论证要的是"确实有个小正数却不显著"，即 Mk05 的 +0.024%。
    _bns_posmax = max([B[i]["dhv_rel_pct"] for i in bns
                       if B[i]["dhv_rel_pct"] > 0], default=0.0)
    write_tex("tab_targets.tex", table_wrap(
        "同一批数据、两个目标量：目标 A（$I_{\\mathrm{mwr}}$ vs.\\ $I_{\\mathrm{rand}}$）与"
        "目标 B（$I_{\\mathrm{mwr}}$ vs.\\ $I_{\\mathrm{mix3}}$）",
        "tab:targets",
        "    \\multicolumn{5}{c}{\\textbf{目标 A：有好起点值多少}} & "
        "\\multicolumn{3}{c}{\\textbf{目标 B：MWR 相对论文口径的净增量}} \\\\\n"
        "    \\cmidrule(lr){1-5}\\cmidrule(lr){6-8}\n"
        "    实例 & $\\Delta$HV(\\%) & 胜出 & $d_z$ & $p$ "
        "& $\\Delta$HV(\\%) & 胜出 & $p$ \\\\\n"
        "    \\midrule\n" + "\n".join(body),
        "rrrrrrrr", font="\\footnotesize",
        notes=("$n=%d$ seeds，同 seed 配对；$\\Delta$HV 为实例边界口径的相对增幅。"
               "目标 A 上 %d/%d 为正（“好起点”这一级确有普遍收益），"
               "但\\textbf{目标 B 只有 %d/%d 显著、其余 %d 个\\textbf{与 0 不可区分}}"
               "（配对 Wilcoxon $p\\ge%s$；注意 $\\Delta$HV 并非精确的 0，"
               "非显著实例中仍为正的最大者是 $%s\\%%$，故“有增益”必须按显著性而非非零来判）："
               "MWR 的净增量是全有或全无，并非“普遍成立但幅度较小”。"
               "拿 A 的结论回答 B 的问题即为目标量错位（缺陷 26）。"
               % (SET_SEEDS, len(apos), len(insts), len(bsig), len(insts), len(bns),
                  "%.2f" % _bns_pmin,
                  num(_bns_posmax, 3)))))

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    yy = np.arange(len(insts))
    ax.barh(yy + 0.19, [A[i]["dhv_rel_pct"] for i in insts], height=0.36,
            color=C_BLUE, label="A: vs $I_{\\rm rand}$")
    ax.barh(yy - 0.19, [B[i]["dhv_rel_pct"] for i in insts], height=0.36,
            color=C_RED, label="B: vs $I_{\\rm mix3}$")
    ax.set_yticks(yy)
    ax.set_yticklabels(insts, fontsize=6.2)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("$\\Delta$HV relative gain (\\%)", fontsize=7)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax.tick_params(labelsize=6.4)
    ax.grid(alpha=0.25, lw=0.4, axis="x")
    save_fig(fig, "fig_targets.pdf")
    data["targets"] = {"n": len(insts), "nz_B": nz,
                       "A": {i: A[i]["dhv_rel_pct"] for i in insts},
                       "B": {i: B[i]["dhv_rel_pct"] for i in insts},
                       "Ap": {i: A[i]["p"] for i in insts},
                       "Bp": {i: B[i]["p"] for i in insts},
                       "a_pos": apos, "a_sig": asig, "b_sig": bsig, "b_nonsig": bns,
                       "b_nonsig_pmin": _bns_pmin, "b_nonsig_posmax": _bns_posmax,
                       "perm_B": tg["vs_I_mix3"].get("permutation", {})}
    return d


# ────────────────────────────── 5. Hurink 留出 + 门控 ──────────────────────────────
# 「有净增量」的判据 —— **配对检验显著为正**，不是"浮点非零"。
#
# 为什么必须把它写死成一个函数（缺陷 29）：Mk 与 Hurink 上都**没有任何一个实例的
# ΔHV 精确等于 0**（Mk 上最小的非零是 Mk07 的 −0.0090%；Hurink 已完成的 24 个实例
# 里 0/24 为零）。若沿用 `abs(ΔHV) > 1e-9` 当判据，开发集 10/10 个实例都会被判成
# "有增益"，混淆矩阵退化成 (tp=9, fp=0, fn=1, tn=0)、判对率 0.90 —— 与本文真正报告的
# 现象（3/10 显著、门控 10/10 判对）**直接矛盾**；留出集上更会 66/66 全是"有增益"，
# 门控表彻底失去意义。注意这不是笔误级别的风险：两套口径下都"非零"，所以不会报错，
# 只会静默给出错表。
#
# 判据显式定为：**ΔHV > 0 且配对 Wilcoxon p < 0.05**（与 docs 里"3/10 显著"同源）。
GAIN_P = 0.05

# "非零"的数值阈值。**只用于筛出可举例的最小非零值**（如表注要举一个
# "确实很小、却不显著"的反例），**绝不用作"有增益"的判据**——
# 判据只有一个地方，即下面的 `is_gain()`（缺陷 29：把"非零"当判据会让
# 判对率从 1.00 掉到 0.30，且不报任何错）。
# 具名而非内联字面量，是为了让"这里筛的是样例、不是判据"在阅读时一眼可辨；
# `TestGateGainCriterion` 另有锁确保它没被混进 `is_gain()`。
NZ_EPS = 1e-9

# ── 实验设置常量：正文里的**设置**数字与效应量守同一条规矩（单一来源）──
# 设置值（种群、代数、seed 区间、预算档）看似不会漂移，但正文里"30 个 seed"
# 与 `\SetSeeds` 混写的先例已经出现过：只要协议一改，手写的那几处就静默失配，
# 而它们分散在摘要、§3.2、§5.4、§6、§7 各处，肉眼很难查全。
DEV_N = 10                    # 开发集实例数（Brandimarte Mk01--Mk10）
SEED_LO, SEED_HI = 42, 71     # 每个配置的随机种子区间（含两端）
# 本轮实际执行的设置（`logs/` 里每个 run 的 components 可核）。
# 提到模块级：生成表的表注（如 `tab_targets`）也要用同一套值。
SET_NP, SET_G, SET_GMAX, SET_SEEDS = 100, 200, 2000, 30
# 响应面的五档墙钟预算分数。**这同一个列表必须同时驱动图与正文宏**：
# 图用 `np.array(SURF_BUDGETS)` 画横轴、宏从它生成"10%,25%,…"，
# 否则"图上有 5 个点、正文却写 10%"这类失配没有任何机制能发现。
SURF_BUDGETS = (0.10, 0.25, 0.50, 0.75, 1.00)


def is_gain(dhv_rel_pct, p):
    """净增量为正 = 配对检验显著（$p<0.05$）**且**方向为正。"""
    if dhv_rel_pct is None or p is None:
        return False
    return (dhv_rel_pct > 0.0) and (p < GAIN_P)


def hurink_gate(data, allow_missing):
    p = os.path.join(ROOT, "logs", "hurink_aig.json")
    pp = os.path.join(ROOT, "logs", "hurink_probe.json")
    if not (os.path.exists(p) and os.path.exists(pp)):
        msg = "Hurink 留出结果（%s；%s）" % (
            "有 hurink_aig.json" if os.path.exists(p) else "缺 hurink_aig.json",
            "有 hurink_probe.json" if os.path.exists(pp) else "缺 hurink_probe.json")
        if not allow_missing:
            raise SystemExit("[错误] 缺少 %s" % msg)
        print("  [!] 缺少 %s -> 写占位表（显式标注未就绪，绝不留旧数字）" % msg)
        # 占位表：让主文件始终可编译；但内容明确写"未就绪"，
        # 不可能被误当成结果（比"静默沿用上一版表格"安全）。
        for name, cap, lab in (
                ("tab_hurink.tex", "独立留出集（Hurink $e$-data）", "tab:hurink"),
                ("tab_gate.tex", "零代门控的混淆矩阵", "tab:gate")):
            write_tex(name, table_wrap(
                cap, lab,
                "    \\multicolumn{3}{l}{\\textbf{留出验证数据尚未生成}}\\\\",
                "lll",
                notes=r"本表为占位：\texttt{logs/hurink\_aig.json} 未就绪。"
                      "生成后由 \\texttt{scripts/paper\\_export.py} 覆盖。"))
        return None
    aig = load_json(p)
    prb = load_json(pp)
    B = {r["instance"]: r for r in aig["targets"]["vs_I_mix3"]["per_instance"]}
    A = {r["instance"]: r for r in aig["targets"]["vs_I_rand"]["per_instance"]}
    lam = {r["instance"]: r["probe_init_hv_lever_pct"] for r in prb["probe"]}

    mk = load_json(os.path.join(ROOT, "logs", "init_probe.json"))
    mk_aig = load_json(os.path.join(ROOT, "logs", "aig_gating.json"))
    mkT = {r["instance"]: r
           for r in mk_aig["targets"]["vs_I_mix3"]["per_instance"]}
    mkL = {r["instance"]: r["probe_init_hv_lever_pct"] for r in mk["probe"]}
    THR = -1.0

    def conf(lamd, tgt):
        c = collections.Counter()
        for i in lamd:
            gain = is_gain(tgt[i]["dhv_rel_pct"], tgt[i]["p"])
            pred = lamd[i] >= THR
            c["tp" if (pred and gain) else "fp" if pred else
              "fn" if gain else "tn"] += 1
        return c

    cdev = conf(mkL, mkT)
    hold = sorted(lam)
    chol = conf(lam, B)

    # 预注册守卫：留出集必须是 data/hurink 下的**全部**实例。
    # 否则（例如探针只跑了前 24 个）表格会安静地变成 n=24，
    # 而正文仍写着"全部 66 个"——两侧对不上却都不报错。
    dd = os.path.join(ROOT, "data", "hurink")
    if os.path.isdir(dd):
        prereg = sorted(os.path.splitext(f)[0] for f in os.listdir(dd)
                        if f.endswith(".fjs"))
        if set(prereg) != set(hold):
            msg = ("留出集与预注册不一致：data/hurink 有 %d 个实例，"
                   "而探针/aig 只覆盖 %d 个（缺 %s；多出 %s）"
                   % (len(prereg), len(hold),
                      sorted(set(prereg) - set(hold))[:5],
                      sorted(set(hold) - set(prereg))[:5]))
            if not allow_missing:
                raise SystemExit("[错误] " + msg)
            print("  [!] " + msg + " -> 表按实际覆盖写出，但正文\u201c全部实例\u201d的"
                  "说法会不成立")
    n_gain = sum(1 for i in hold if is_gain(B[i]["dhv_rel_pct"], B[i]["p"]))
    agree = sum(1 for i in hold
                if (lam[i] >= THR) == is_gain(B[i]["dhv_rel_pct"], B[i]["p"]))

    cells = []
    for i in hold:
        b = B[i]
        cells.append("%s & %s & %s & %d/%d & %s & %s & %s"
                     % (i, num(lam[i], 2), num(b["dhv_rel_pct"]), b["wins"], b["n"],
                        num(b["dz"], 2), ptex(b["p"]),
                        "$\\checkmark$" if is_gain(b["dhv_rel_pct"], b["p"])
                        else "--"))
    # **必须两栏续排**：66 行单栏时整表比 A4 文本区还高 517 pt，LaTeX 会报
    # "Float too large for page" 并把表强行排出纸张（2026-09-19 实测）。
    # 两栏后 33 行，余量充足。
    half = (len(cells) + 1) // 2
    left, right = cells[:half], cells[half:]
    HEAD = " & ".join(["实例", "$\\lambda_0$(\\%)", "$\\Delta$HV(\\%)", "胜出",
                       "$d_z$", "$p$", "显著"])
    rows = ["    " + HEAD + " & " + HEAD + " \\\\", "    \\midrule"]
    for k in range(half):
        r = right[k] if k < len(right) else " & ".join([""] * 7)
        rows.append("    %s & %s \\\\" % (left[k], r))
    # 表注断言"两个实例集上都没有 ΔHV 精确为 0 的实例"——这是**可检验**的，
    # 就不能只当修辞写：只要有一个实例真为 0，这句就是假陈述，而它恰恰是
    # "用非零当判据会把全部实例判成有增益"这条论证的前提。
    _zeros = [i for i, v in list(B.items()) + list(mkT.items())
              if abs(v["dhv_rel_pct"]) <= NZ_EPS]
    if _zeros:
        raise SystemExit("[错误] 表注断言「没有 ΔHV 精确为 0 的实例」不成立：%s" % _zeros)
    # "最小非零"只在**开发集**上取（表注要的正是开发集那一侧的反例）。
    _devnz = [v["dhv_rel_pct"] for v in mkT.values()
              if abs(v["dhv_rel_pct"]) > NZ_EPS]
    _min_nz = min(_devnz, key=abs) if _devnz else 0.0
    # 标题里的两栏区间**现场取**：原先写死"左栏 Hed01--"，"--"后面是空的
    # （一个悬空的区间），读者看不出左栏到哪儿为止，改了实例集更无从核对。
    cap = ("独立留出集（Hurink $e$-data，$n=%d$）：零代探针 $\\lambda_0$ 与 MWR 净增量"
           % len(hold))
    if half:
        cap += ("（左右两栏续排：左栏 %s--%s，右栏 %s--%s）"
                % (hold[0], hold[half - 1], hold[half], hold[-1]))
    write_tex("tab_hurink.tex", table_wrap(
        cap,
        "tab:hurink",
        "\n".join(rows),
        "rrrrrlr@{\\hspace{6pt}}rrrrrlr",
        font="\\scriptsize", colsep=3,
        notes=("$\\lambda_0$ 为\\textbf{零代探针}给出的初始前沿 HV 杠杆（MWR $-$ MIX3，"
               r"只调一次 \texttt{\_init\_population()}，不做任何搜索）。"
               # 门限与显著性门槛直接引用**正文用的那两个宏**（\HKthr / \SigLevel）：
               # 表注里手写的 "-1.0%" 与正文明的 `\HKthr` 是同一个量，
               # 两处来源必然漂移；写成宏则物理上不可能不一致。
               "门限 $\\lambda_0\\ge\\HKthr\\%$ 在开发集（Mk01--Mk10）上标定后"
               "\\textbf{冻结}。"
               "\\textbf{「显著为正」= $\\Delta$HV$>0$ 且配对 Wilcoxon $p<\\SigLevel$} —— "
               "不是「$\\Delta$HV$\\neq 0$」：两个实例集上都\\textbf{没有} $\\Delta$HV 精确为 0 的实例"
               + ("（开发集最小非零为 $%s\\%%$）" % num(_min_nz, 2))
               + "，用非零当判据会把全部实例判成有增益。"
               # 缺陷 36：这里**不能**对整段做 `%` 格式化 —— Python 的 `%` 不认
               # 反斜杠转义，LaTeX 的 `\%` 会被当成格式符（`\%` 后跟 `$` 直接
               # ValueError）。所以把两个计数单独格式化后再拼接。
               "留出集上显著为正的实例为 " + ("%d/%d" % (n_gain, len(hold)))
               + "，门控判对 " + ("%d/%d" % (agree, len(hold))) + "。")))

    # 门控混淆矩阵（开发 / 留出并排）
    write_tex("tab_gate.tex", table_wrap(
        # 门限用**正文那个宏**（\\HKthr 由 hk["thr"] 生成，与这里的 THR 同源）：
        # 标题若自己格式化成 "-1%"、正文却是 "-1.0%"，同一门限就有了两种写法。
        "零代门控的混淆矩阵（门限在开发集上标定后冻结：$\\lambda_0\\ge\\HKthr\\%$）",
        "tab:gate",
        "    集合 & $n$ & 命中 & 误放 & 漏放 & 正确拒绝 & 判对率 \\\\\n    \\midrule\n"
        "    开发集 Mk01--Mk10 & %d & %d & %d & %d & %d & %.2f \\\\\n"
        "    留出集 Hurink $e$-data & %d & %d & %d & %d & %d & %.2f \\\\"
        % (sum(cdev.values()), cdev["tp"], cdev["fp"], cdev["fn"], cdev["tn"],
           (cdev["tp"] + cdev["tn"]) / max(sum(cdev.values()), 1),
           len(hold), chol["tp"], chol["fp"], chol["fn"], chol["tn"],
           (chol["tp"] + chol["tn"]) / max(len(hold), 1)),
        "lrrrrrr",
        notes=("“命中”= 预测有增益且确实有；“误放”= 预测有而实际为 0；"
               "“漏放”= 预测没有而实际有。开发集是门限的标定集，"
               "只有留出集一列是无偏的。")))

    data["hurink"] = {"n": len(hold), "n_gain": n_gain, "thr": THR,
                      "dev": dict(cdev), "hold": dict(chol), "agree": agree,
                      "lambda": lam, "B": {i: B[i]["dhv_rel_pct"] for i in hold},
                      "gain": {i: is_gain(B[i]["dhv_rel_pct"], B[i]["p"])
                               for i in hold},
                      "A": {i: A[i]["dhv_rel_pct"] for i in hold},
                      "perm": aig["targets"]["vs_I_mix3"].get("permutation", {}),
                      "corr": aig["targets"]["vs_I_mix3"].get("correlations", {})}
    return data["hurink"]


def fig_gate(data):
    """门控分离图。**独立于 Hurink 数据是否就绪**：只画开发集也能出图，
    留出集就绪后自动加上三角点，避免"数据一变图就没了"。

    y 轴是**「净增量显著为正」**（配对 Wilcoxon $p<0.05$ 且 $\\Delta$HV$>0$），
    不是 $\\Delta$HV$\\neq0$ —— 见 `is_gain` 处的说明（缺陷 29）。
    """
    p = os.path.join(ROOT, "logs", "init_probe.json")
    pa = os.path.join(ROOT, "logs", "aig_gating.json")
    if not (os.path.exists(p) and os.path.exists(pa)):
        print("  [!] 缺 init_probe.json / aig_gating.json -> 无法出门控图")
        return None
    mkT = {r["instance"]: r
           for r in load_json(pa)["targets"]["vs_I_mix3"]["per_instance"]}
    mkL = {r["instance"]: r["probe_init_hv_lever_pct"] for r in load_json(p)["probe"]}
    THR = data.get("hurink", {}).get("thr", -1.0)

    hold, lam, B = None, None, None
    ph, pah = os.path.join(ROOT, "logs", "hurink_probe.json"), os.path.join(ROOT, "logs", "hurink_aig.json")
    if os.path.exists(ph) and os.path.exists(pah):
        lam = {r["instance"]: r["probe_init_hv_lever_pct"] for r in load_json(ph)["probe"]}
        B = {r["instance"]: r
             for r in load_json(pah)["targets"]["vs_I_mix3"]["per_instance"]}
        hold = sorted(lam)

    fig, ax = plt.subplots(figsize=(6.7, 2.8))
    sets = [(mkL, mkT, True)]
    if hold:
        sets.append((lam, B, False))
    for lamd, tgt, mkf in sets:
        ks = sorted(lamd)
        ax.scatter([lamd[k] for k in ks],
                   [1.0 if is_gain(tgt[k]["dhv_rel_pct"], tgt[k]["p"]) else 0.0
                    for k in ks],
                   s=26 if mkf else 20, marker="o" if mkf else "^", facecolor="none",
                   edgecolor=C_BLUE if mkf else C_RED, linewidth=1.0,
                   label=("development: Mk01--Mk10" if mkf
                          else "holdout: Hurink $e$-data ($n=%d$)" % len(hold)), zorder=3)
    ax.axvline(THR, color="k", lw=1.0, ls="--")
    ax.text(THR + 0.2, 1.16, "gate $\\lambda_0=%.1f\\%%$" % THR, fontsize=6.4)
    ax.set_xlabel("$\\lambda_0$ = initial-frontier HV leverage (\\%)", fontsize=7.4)
    ax.set_ylabel("net gain significant ($p<0.05$)", fontsize=7.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["no", "yes"], fontsize=6.8)
    ax.set_ylim(-0.35, 1.45)
    ax.legend(fontsize=6.3, frameon=False, loc="center left")
    ax.tick_params(labelsize=6.6)
    ax.grid(alpha=0.25, lw=0.4)
    save_fig(fig, "fig_gate.pdf")
    return {"n_dev": len(mkL), "n_hold": len(hold) if hold else 0}


# ────────────────────────────── 6. 探针 seed 方差 ──────────────────────────────
def tab_probe_seeds(data, allow_missing):
    p = os.path.join(ROOT, "logs", "probe_seed_var.json")
    if not os.path.exists(p):
        print("  [!] 缺 logs/probe_seed_var.json -> 跳过 seed 方差表")
        return None
    d = load_json(p)
    NS = [1, 3, 5, 10, 30]
    body = []
    for r in d["rows"]:
        cells = [num(r["lam30"], 2)]
        for n in NS[:-1]:
            cells.append(num(r["sd"][str(n)], 2, signed=False))
        for n in NS[:-1]:
            cells.append("%.2f" % r["agree"][str(n)])
        body.append("    %s & %s \\\\" % (r["instance"], " & ".join(cells)))
    summ = " & ".join("%.3f" % d["summary"][str(n)]["sd_mean"] for n in NS[:-1])
    agr = " & ".join("%.3f" % d["summary"][str(n)]["agree_mean"] for n in NS[:-1])
    write_tex("tab_probe_seeds.tex", table_wrap(
        "零代探针在不同 seed 预算下的稳定性（%s）" % d.get("set_name", "Mk01--Mk10"),
        "tab:probe_seeds",
        "    \\multicolumn{2}{c}{} & \\multicolumn{4}{c}{$\\lambda_0$ 的子集标准差} & "
        "\\multicolumn{4}{c}{与 30-seed 判定的一致率} \\\\\n"
        "    \\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\n"
        "    实例 & $\\lambda_0(30)$ & $n{=}1$ & $n{=}3$ & $n{=}5$ & $n{=}10$ "
        "& $n{=}1$ & $n{=}3$ & $n{=}5$ & $n{=}10$ \\\\\n"
        "    \\midrule\n" + "\n".join(body) +
        "\n    \\midrule\n    \\textit{跨实例均值} & -- & " + summ + " & " + agr + " \\\\",
        "rrrrrrrrrr", font="\\footnotesize",
        notes=("对每个实例枚举/抽样 $n$ 个 seed 的子集，用与正式探针完全相同的估计量"
               "（先对 seed 求均值再作比）重算 $\\lambda_0$，再看判定是否与 30-seed 一致。"
               r"门限 $-1.0\%$；子集数见 \texttt{logs/probe\_seed\_var.json}。")))
    data["probe_seeds"] = d
    return d


# ────────────────────────────── 7. RL 选算子 n=50 ──────────────────────────────
def tab_rl50(data, allow_missing):
    p = os.path.join(ROOT, "logs", "rl50.json")
    if not os.path.exists(p):
        print("  [!] 缺 logs/rl50.json -> 跳过 RL n=50 表")
        return None
    d = load_json(p)
    body = []
    for r in d["rows"]:
        body.append("    %s & %s & %s & %d/%d & %s & %s \\\\"
                    % (r["instance"], r["budget"], num(r["rel_pct"]), r["wins"], r["n"],
                       num(r["dz"], 2), ptex(r["p"])))
    write_tex("tab_rl50.tex", table_wrap(
        "RL 选算子效应在 $n=50$ seeds 下的复验（RMOEAD vs.\\ D5，等算力两档）",
        "tab:rl50",
        "    实例 & 算力档 & $\\Delta$HV(\\%) & 胜出 & $d_z$ & $p$ \\\\\n"
        "    \\midrule\n" + "\n".join(body), "llrrrr",
        notes=d.get("note", "")))
    data["rl50"] = d
    return d


# ────────────────────────────── 8. 阶梯图 ──────────────────────────────
def fig_ladder(data, allow_missing):
    if "ladder" not in data:
        return
    lad = data["ladder"]
    keys = [k for k, _, _, _ in LADDER_STEPS if k in lad]
    ri = [lad[k]["rel_inst"] for k in keys]
    rb = [lad[k]["rel_box"] for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.bar(x - 0.2, ri, width=0.4, color=C_BLUE, label="instance-boundary (reported)")
    ax.bar(x + 0.2, rb, width=0.4, color="none", edgecolor=C_RED, hatch="////",
           linewidth=0.8, label="box-normalised (contrast only)")
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("|", "$|$") for k in keys], fontsize=7)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel("$\\Delta$HV relative gain (\\%)", fontsize=7.4)
    for xi, v, b in zip(x, ri, rb):
        ax.text(xi - 0.2, v, "%+.2f" % v, ha="center", fontsize=5.8,
                va="bottom" if v >= 0 else "top")
        if abs(v) > 1e-12 and np.isfinite(b / v):
            ax.text(xi + 0.2, b, "%.1f$\\times$" % (b / v), ha="center", fontsize=5.8,
                    va="bottom" if b >= 0 else "top", color=C_RED)
    ax.legend(fontsize=6.4, frameon=False)
    ax.tick_params(labelsize=6.6)
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    save_fig(fig, "fig_ladder.pdf")


# ────────────────────────────── main ──────────────────────────────
# ─────────────────── 8. 正文数字宏：让正文数字也只有一个来源 ───────────────────
_TEX_NAME_OK = re.compile(r"[A-Za-z]+\Z")


def fig_concept(data, allow_missing):
    """概念示意图（**手工构造的示意点，不是实验结果**）。

    第 2 节面向不熟悉多目标优化的读者，用两栏把两件事讲清楚：
    (a) Pareto 支配与 Pareto 前沿；(b) 超体积 HV 是什么、为什么它必须先
    把两个目标归一化到同一量纲——而"归一化边界怎么取"正是评价口径一节的主题。

    与其它图不同，这张图**不读 logs/**：它的点是为讲清概念而构造的示意图，
    因此不参与任何结论、也不受"一个数字一个来源"的约束。代价是必须在
    caption 里显式声明"示意、非实验结果"，否则读者会把它当成实验读数。
    """
    # 前沿按 x 升序、y 递减（最小化问题，左下为优）
    pf = np.array([[0.05, 0.80], [0.18, 0.55], [0.34, 0.38],
                   [0.52, 0.24], [0.72, 0.14], [0.90, 0.07]])
    dom = np.array([[0.52, 0.62], [0.70, 0.44], [0.30, 0.72], [0.86, 0.30]])
    rx, ry = REF                      # 归一化后参考点 (1.02, 1.02)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    # ── (a) Pareto 支配：A 的右上矩形内，任何解都被 A 支配 ──
    ax = axes[0]
    a = pf[2]
    ax.add_patch(plt.Rectangle((a[0], a[1]), rx - a[0], ry - a[1],
                               facecolor=C_GREY, alpha=0.18,
                               edgecolor="none", zorder=1))
    ax.scatter(dom[:, 0], dom[:, 1], s=26, facecolor="none", edgecolor=C_GREY,
               linewidth=1.1, zorder=3, label="dominated solution")
    ax.plot(pf[:, 0], pf[:, 1], "-", color=C_RED, lw=1.2, zorder=2)
    ax.scatter(pf[:, 0], pf[:, 1], s=32, color=C_RED, zorder=4,
               label="non-dominated (Pareto front)")
    ax.plot([a[0]], [a[1]], marker="o", ms=7, mfc="none", mec="k", mew=1.0, zorder=5)
    ax.annotate("A", xy=(a[0], a[1]), xytext=(a[0] - 0.02, a[1] - 0.10),
                fontsize=7.5, fontweight="bold", zorder=6)
    ax.annotate("A dominates every solution here\n(both objectives no worse)",
                xy=(0.70, 0.62), xytext=(0.30, 0.95), fontsize=6.0,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C_GREY), zorder=6)
    ax.set_title("(a) No single best schedule:\nPareto domination", fontsize=8)
    ax.legend(fontsize=5.9, frameon=False, loc="lower left",
              bbox_to_anchor=(-0.015, -0.02))

    # ── (b) HV：把整条前沿压成一个数 ──
    ax = axes[1]
    xs = np.append(pf[:, 0], rx)
    ys = np.append(pf[:, 1], pf[-1, 1])
    ax.fill_between(xs, ys, ry, step="post", color=C_BLUE, alpha=0.22,
                    lw=0, zorder=1)
    ax.plot(pf[:, 0], pf[:, 1], "-", color=C_RED, lw=1.2, zorder=2)
    ax.scatter(pf[:, 0], pf[:, 1], s=32, color=C_RED, zorder=4,
               label="Pareto 前沿")
    ax.scatter([rx], [ry], s=46, marker="*", color="k", zorder=5,
               label="reference point ref $=(%.2f,%.2f)$" % REF)
    ax.text(0.06, 0.92, "HV = shaded area\n(one scalar per front)",
            fontsize=6.0, zorder=6)
    ax.set_title("(b) Hypervolume (HV):\nfirst normalize both objectives", fontsize=8)

    for ax in axes:
        ax.set_xlim(-0.03, 1.09)
        ax.set_ylim(-0.03, 1.09)
        ax.set_xlabel("$f_1$ = makespan (normalized, min)", fontsize=6.6)
        ax.set_ylabel("$f_2$ = machine load (normalized, min)", fontsize=6.6)
        ax.tick_params(labelsize=6.0)
        ax.grid(alpha=0.18, lw=0.35)
    save_fig(fig, "fig_concept.pdf")
    return {"n_pf": int(len(pf)), "n_dom": int(len(dom))}


def fig_caliber(data, allow_missing):
    """评价口径示意图（**手工构造的示意点，不是实验结果**）。

    §4 是全文的方法学核心，但此前只有文字：归一化的"盒"从哪来、为什么
    换一批比较对象读数就变。这张图用**同一批前沿点配两个不同的框**讲清它：

      (a) 实例边界口径——框由实例数据推出（临界路径下界 / 全部工序最长时间和），
          与参与比较的臂集**无关**；
      (b) 盒口径——框取参与比较的前沿极值，多放进一个更差的臂，框就被撑大，
          同一批点的归一化坐标随之改变。

    与 fig_concept 同：**不读 logs/**，点是为讲清概念手工构造的示意点，
    不参与任何结论；代价是必须在 caption 显式声明"示意、非实验结果"。
    """
    # 两批前沿点（最小化：x 增则 y 减，左下为优），两栏共用同一批点
    pf1 = np.array([[2.0, 9.0], [4.0, 5.0], [7.0, 2.0], [9.0, 1.0]])
    pf2 = np.array([[2.6, 10.5], [5.2, 6.2], [8.2, 3.0], [10.5, 1.6]])
    both = np.vstack([pf1, pf2])
    inst_lo, inst_hi = np.array([0.0, 0.0]), np.array([16.0, 16.0])
    box_lo, box_hi = both.min(axis=0), both.max(axis=0)
    extra = np.array([14.5, 14.5])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax in axes:
        ax.set_xlim(-0.7, 17.4)
        ax.set_ylim(-0.7, 17.4)
        ax.set_xlabel("$f_1$ = makespan (raw units)", fontsize=6.6)
        ax.set_ylabel("$f_2$ = machine load (raw units)", fontsize=6.6)
        ax.tick_params(labelsize=6.0)
        ax.grid(alpha=0.18, lw=0.35)

    def _arms(ax):
        ax.scatter(pf1[:, 0], pf1[:, 1], s=26, color=C_RED, zorder=4)
        ax.scatter(pf2[:, 0], pf2[:, 1], s=24, color=C_BLUE, marker="s", zorder=4)

    # ── (a) 实例边界口径：框由实例决定，与臂集无关 ──
    ax = axes[0]
    ax.add_patch(plt.Rectangle(inst_lo, *(inst_hi - inst_lo), fill=False,
                               edgecolor=C_GREEN, lw=1.2, zorder=2))
    _arms(ax)
    ax.annotate("box from the instance:\ncritical-path lower bound,\n"
                "sum of longest operation times",
                xy=(9.5, 16.0), xytext=(7.4, 11.4), fontsize=5.8,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C_GREEN), zorder=6)
    ax.annotate("fronts of two arms",
                xy=(10.5, 1.6), xytext=(10.3, 4.3), fontsize=5.8,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C_GREY), zorder=6)
    ax.set_title("(a) Instance-boundary box\n"
                 "box is independent of the arm set", fontsize=7.6)

    # ── (b) 盒口径：框随参与比较的臂集伸缩 ──
    ax = axes[1]
    ax.add_patch(plt.Rectangle(box_lo, *(box_hi - box_lo), fill=False,
                               edgecolor=C_RED, lw=1.3, zorder=3))
    ax.add_patch(plt.Rectangle(box_lo, *(extra - box_lo), fill=False,
                               edgecolor=C_BLUE, lw=1.0, ls=(0, (4, 3)), zorder=3))
    _arms(ax)
    ax.scatter([extra[0]], [extra[1]], s=30, facecolor="none", edgecolor=C_GREY,
               linewidth=1.1, zorder=4)
    ax.annotate("this box covers the two arms' fronts only;\n"
                "one extra, worse arm $\\rightarrow$ every reading moves",
                xy=(14.2, 14.2), xytext=(0.4, 17.1), fontsize=5.8, va="top",
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C_GREY), zorder=6)
    ax.set_title("(b) Box normalisation\n"
                 "box $=$ extremes of the arms compared", fontsize=7.6)

    save_fig(fig, "fig_caliber.pdf")
    return {"n_pf": int(len(both)), "n_box_arms": 1}


def tex_macro_line(name, val):
    """拼一行 `\\newcommand`；**宏名必须是纯字母**。

    TeX 控制序列名只能是字母：`\\NullP95` 会被解析成 `\\NullP` 紧跟 `95`，
    编译报的是 `Missing number, treated as zero.` —— **不报"名字非法"**，
    于是排查方向被完全带偏（本宏首版 6 个带数字的名字全中，且错误行号指向
    名字**前一行**）。这里在生成阶段直接拦下，数字改写成英文（Nine / Pct）。
    """
    if not _TEX_NAME_OK.match(name):
        raise SystemExit("[错误] 宏名 %r 含非字母字符——LaTeX 控制序列名只能是字母，"
                         "请改用英文数字（如 PSmkNineLam / NullPct）" % name)
    return "\\newcommand{\\" + name + "}{" + str(val) + "}"


def write_macros(data, allow_missing, write=True):
    """把**正文**（不只是表格）引用的每个数字导出成宏，正文只写宏名。

    动机与缺陷 30 同源：表格是现场生成的，正文数字却曾手写，于是出现
    "同一指标两套数字"——论文 §5 把 ``rho=+0.83`` 归给式 (2) 的 probe（实为
    ``probe_mean_mc_lever_pct``，而 probe 自己是 +0.77）。宏化之后正文与表格
    读的是同一份 ``logs/``，物理上无法再漂移。

    留出数据缺失时写 ``\\HKready=0``，正文据此**显式标注未就绪**，绝不沿用旧数字。
    """
    L = ["% 本文件由 scripts/paper_export.py 现场生成，请勿手改。",
         "% 正文引用宏名而非字面数字：数字只有一个来源（缺陷 30）。"]

    def m(name, val):
        L.append(tex_macro_line(name, val))

    def p(x, nd=2):
        return ("%+." + str(nd) + "f") % float(x)

    # —— 留出集（Hurink）门控 ——
    hk = data.get("hurink")
    if not hk:
        for n, v in (("HKready", 0), ("HKn", 0), ("HKthr", -1.0), ("HKgain", 0),
                     ("HKtp", 0), ("HKfp", 0), ("HKfn", 0), ("HKtn", 0),
                     ("HKaccDev", 0.0), ("HKaccHold", 0.0),
                     ("HKlamLo", 0), ("HKlamHi", 0), ("HKnearHalf", 0),
                     ("HKnearBand", 0.5),
                     ("HKholdPosN", 0), ("HKholdPosLamLo", 0), ("HKholdPosLamHi", 0),
                     ("HKaccRejectAll", 0), ("HKaccLiftAbs", 0), ("HKpassN", 0),
                     ("HKpassPrec", "--"), ("HKrecall", "--")):
            m(n, v)
        m("HKverdict", "\\textbf{（留出验证数据尚未生成。）}")
    else:
        hold, dev = hk["hold"], hk["dev"]
        n = hk["n"]
        m("HKready", 1)
        m("HKn", n)
        m("HKthr", hk["thr"])
        m("HKgain", hk["n_gain"])
        for k in ("tp", "fp", "fn", "tn"):
            m("HK" + k, hold.get(k, 0))
        m("HKaccHold", "%.2f" % ((hold.get("tp", 0) + hold.get("tn", 0)) / max(n, 1)))
        m("HKaccDev", "%.2f" % ((dev.get("tp", 0) + dev.get("tn", 0))
                                / max(sum(dev.values()), 1)))
        lam = [hk["lambda"][i] for i in sorted(hk["lambda"])]
        # 空留出集不该把导出器炸掉（`min([])` 抛 ValueError），而应写出 0 并让
        # 正文的 \HKready/\HKverdict 如实说明——否则"数据没跑出来"会伪装成"脚本坏了"。
        m("HKlamLo", p(min(lam), 1) if lam else 0)
        m("HKlamHi", p(max(lam), 1) if lam else 0)
        # 注意：`hk["lambda"]` 是**留出集**的 λ（来自 hurink_probe.json），
        # 名字里必须带 hold，否则会被误当成开发集数字（本宏最初的命名错误即此）。
        pl = [hk["lambda"][i] for i in sorted(hk["lambda"]) if hk["gain"][i]]
        m("HKholdPosN", len(pl))
        if pl:
            m("HKholdPosLamLo", p(min(pl), 2))
            m("HKholdPosLamHi", p(max(pl), 2))
        else:
            m("HKholdPosLamLo", 0)
            m("HKholdPosLamHi", 0)

        # 结论句按数据分支（措辞由人写定，数字由数据填）——不让"全有或全无"
        # 这类形态被模板句抹平。
        g, tp, fp, fn = hk["n_gain"], hold.get("tp", 0), hold.get("fp", 0), hold.get("fn", 0)
        # 贴着门限的实例数：§6 已证明这一带即使 30 seeds 也判不稳，
        # 所以留出集的判对率必须**连同"有多少实例落在决策边界附近"一起报**，
        # 否则"判对率低"会被误读成"探针无效"（实际是边界样本的固有不确定性）。
        # 决策边界带宽 ±0.5 个百分点。**不是** HKdevGap —— 后者是开发集两类实例的
        # λ 间距（数值凑巧同为 0.5，语义完全不同），所以给它自己的宏，
        # 免得正文里一个字面 0.5 同时被两边认领（守卫会报"两处来源"）。
        NEAR_BAND = 0.5
        m("HKnearBand", NEAR_BAND)
        near = sum(1 for i in hk["lambda"]
                   if abs(hk["lambda"][i] - hk["thr"]) <= NEAR_BAND)
        m("HKnearHalf", near)
        tail = ("其中 %d 个实例的 $\\lam$ 落在门限 $\\pm\\HKnearBand$ 个百分点内——"
                "这一带在开发集上已被证明即使 %d seeds 也判不稳"
                "（\\S\\ref{sec:seedvar}），故判对率要连同它一起读。"
                % (near, SET_SEEDS))
        share = 100.0 * g / max(n, 1)
        # 缺陷 37：判对率**必须与平凡基线并列报**。留出集 66 个实例里只有 g 个
        # 真赚，所以"一律拒绝"的判对率 = (n-g)/n —— 它完全可能**高于**门控本身。
        # 不报这一句，0.70 会被读成"尚可"；真相可能是"不如什么都不做"。
        acc = (tp + hold.get("tn", 0)) / max(n, 1)
        acc_rej = (n - g) / max(n, 1)
        pass_n = tp + fp
        m("HKaccRejectAll", "%.2f" % acc_rej)
        m("HKaccLiftAbs", "%.3f" % abs(acc - acc_rej))
        m("HKpassN", pass_n)
        m("HKpassPrec", ("%.2f" % (tp / pass_n)) if pass_n else "--")
        # 精确率之外还要给**召回**：漏放 (g-tp) 个真赚的实例在"判对率"里看不见，
        # 但"该用的时候没用上"和"不该用的时候用了"一样会让方案失效。
        m("HKrecall", ("%.2f" % (tp / g)) if g else "--")
        if g == 0:
            v = ("留出集上 MWR 的净增量\\textbf{一次也没有}达到显著（0/%d）："
                 "开发集上「少数实例显著为正」的形态在留出集上没有出现，"
                 "门控因此没有正例可判——它的适用边界比开发集暗示的要窄。" % n)
        else:
            v = ("留出集上净增量显著为正的实例为 %d/%d（%.1f\\%%），"
                 "门控命中 %d、误放 %d、漏放 %d，判对率 \\HKaccHold。"
                 % (g, n, share, tp, fp, fn))
            v += ("但留出集本身只有 %.1f\\%% 的实例真赚，\\textbf{一律拒绝}的判对率就有 "
                  "\\HKaccRejectAll —— 门控比它低 \\HKaccLiftAbs，"
                  "而门控放行的 \\HKpassN 个实例里只有 %d 个真赚（精确率 \\HKpassPrec）。"
                  "换言之，在这个分布上探针的判别力\\textbf{低于不做任何判别}。"
                  % (share, tp))
        m("HKverdict", v + " " + tail)

    # —— 目标 A / B 的区间与 Bonferroni 门槛（§4.3/§4.4）：原先手写 ——
    # 区间做成**整串宏**（值含中文"至"与 `\%`，非纯数值）：
    # 若拆成 "+0.05"/"+8.69" 两个纯数值宏，守卫会因正文里到处有 `p<0.05` 而误报。
    PC = chr(92) + "%"
    tg = data.get("targets") or {}
    if tg:
        A, B = tg["A"], tg["B"]
        apos, bsig, bns = tg["a_pos"], tg["b_sig"], tg["b_nonsig"]
        m("TgtAN", tg["n"])
        m("TgtAPosN", len(apos))
        m("TgtASigN", len(tg.get("a_sig", [])))
        m("TgtARange", ("+%.2f" % min(A[i] for i in apos)) + PC + " 至 +%.2f" % max(A[i] for i in apos) + PC
          if apos else "--")
        m("TgtBSigN", len(bsig))
        m("TgtBSigRange", ("+%.2f" % min(B[i] for i in bsig)) + PC + " 至 +%.2f" % max(B[i] for i in bsig) + PC
          if bsig else "--")
        m("TgtBNonSigN", len(bns))
        m("TgtBNonSigRange", ("%+.2f" % min(B[i] for i in bns)) + PC + " 至 %+.2f" % max(B[i] for i in bns) + PC
          if bns else "--")
        m("TgtBNonSigPMin", "%.2f" % tg.get("b_nonsig_pmin", 1.0))
        # Bonferroni 门槛 = 0.05 / 家族大小（家族大小取自置换零假设的 keys）
        _fa = (tg.get("perm_B") or {}).get("all", {})
        _nk = len(_fa.get("keys", [])) or 8
        m("BonfFamN", _nk)
        m("BonfThr", "%.5f" % (0.05 / _nk))
    else:
        for nm, v in (("TgtAN", 0), ("TgtAPosN", 0), ("TgtASigN", 0),
                      ("TgtARange", "--"),
                      ("TgtBSigN", 0), ("TgtBSigRange", "--"),
                      ("TgtBNonSigN", 0), ("TgtBNonSigRange", "--"),
                      ("TgtBNonSigPMin", 1), ("BonfFamN", 0), ("BonfThr", 0)):
            m(nm, v)

    # —— 算力–算法响应面（§4.2）：正文那两组数原先也是手写的 ——
    sf = (data.get("surface") or {}).get("D2|D1") or {}
    if sf:
        _ins = sorted(sf)                      # 与表/图的实例顺序一致（Mk07/Mk09/Mk10）
        if len(_ins) < 3:
            raise SystemExit("[错误] data['surface']['D2|D1'] 只有 %d 个实例，"
                             "正文 §4.2 需要 3 个" % len(_ins))
        for tag, i in (("A", 0), ("B", 1), ("C", 2)):
            g = sf[_ins[i]]
            m("SurfInitTen" + tag, "%.3f" % abs(g[0]))    # 10% 预算档
            m("SurfInitEnd" + tag, "%.4f" % abs(g[4]))    # 100% 预算档
        _ratio = [abs(sf[i][0]) / max(abs(sf[i][4]), 1e-12) for i in _ins]
        m("SurfDecayLo", "%.1f" % min(_ratio))
        m("SurfDecayHi", "%.1f" % max(_ratio))
    else:
        # 缺数据时写占位（与 HK / 其它宏块同一约定）。**不要在这里抛错**：
        # 严格性由上游 `fig_surface()` 里的 `need()` 保证（非 --allow-missing 时会硬失败），
        # 而 `write_macros` 会被测试以最小 data 调用，抛错会把"分阶段出稿"整条路堵死。
        for tag in ("A", "B", "C"):
            m("SurfInitTen" + tag, 0)
            m("SurfInitEnd" + tag, 0)
        m("SurfDecayLo", 0)
        m("SurfDecayHi", 0)
    # 表注与 §4.2 正文共用这同一份文本（原先两处各写一遍 1.1e-3）
    _sm = data.get("surface_max") or {}
    m("SurfNonInitMax", _sm.get("txt", "$0$"))
    m("SurfNonInitArg", _sm.get("label", "--"))

    # —— 组件阶梯（§4.1）：正文那几个数原先**手写**，与 tab_ladder.tex 构成
    # "同一指标两处来源"（缺陷 32 的同类形态）。今天恰好一致，改数据就会漂移。
    # 行键与 tab_ladder 的 `LADDER_STEPS` 同源（都来自 data["ladder"]）。
    lad = data.get("ladder") or {}
    _LAD = (("Mix", "D2|D1"), ("Vns", "D3|D2"), ("Qpas", "D4|D3"),
            ("Elite", "D5|D4"), ("Rl", "RMOEAD|D5"))
    if lad:
        for nm, key in _LAD:
            r = lad.get(key)
            if r is None:
                raise SystemExit(
                    "[错误] data['ladder'] 缺 %s —— 正文 §4.1 会引用 \\Ladder%sRel，"
                    "缺了就是静默出错" % (key, nm))
            # 3 位小数：与 tab_ladder.tex 的 `num()` 默认精度**逐位一致**，
            # 且避免 `-0.003` 被舍成 `-0.00`（丢掉真实的小负值）。
            m("Ladder" + nm + "Rel", p(r["rel_inst"], 3))
            m("Ladder" + nm + "Wins", r["wins"])
            m("Ladder" + nm + "P", "%.3f" % r["p"])
    else:
        # 同上：缺数据写占位，不抛错（严格性由 `tab_ladder()` 里的 `need()` 保证）。
        for nm, _ in _LAD:
            m("Ladder" + nm + "Rel", 0)
            m("Ladder" + nm + "Wins", "0/0")
            m("Ladder" + nm + "P", 1)

    # —— 开发集零代探针：probe 自身与家族最大必须分列（曾被混为同一个数）——
    ip = os.path.join(ROOT, "logs", "init_probe.json")
    if os.path.exists(ip):
        d = load_json(ip)
        c = d["correlations"]
        pv = c.get("probe_init_hv_lever_pct", {})
        m("ProbeRho", p(pv.get("spearman_rho", 0), 2))
        m("ProbeRhoP", "%.3f" % pv.get("spearman_p", 1.0))
        pm = c.get("probe_mean_mc_lever_pct", {})
        m("FamRho", p(pm.get("spearman_rho", 0), 2))
        m("FamP", "%.3f" % pm.get("spearman_p", 1.0))
        perm = d.get("permutation", {})
        m("PermP", "%.3f" % perm.get("mc_p_fwer", 1.0))
        m("NullPct", "%.2f" % perm.get("null_p95", 0.0))
        m("FamArgmax", perm.get("obs_argmax_key", "-"))
        # 开发集**正例**的 λ 区间与**其余实例**的 λ 上界：正文那句
        # "门限在开发集上完全分离" 靠这两个数，必须从开发集数据算，不能借留出集的。
        devpos, devoth = [], []
        ap2 = os.path.join(ROOT, "logs", "aig_gating.json")
        if os.path.exists(ap2):
            per = {r["instance"]: r for r in
                   load_json(ap2)["targets"]["vs_I_mix3"]["per_instance"]}
            for r in d["probe"]:
                t = per.get(r["instance"])
                if t is None:
                    continue
                (devpos if is_gain(t["dhv_rel_pct"], t["p"]) else devoth).append(
                    r["probe_init_hv_lever_pct"])
        m("HKdevPosN", len(devpos))
        m("HKdevPosLamLo", p(min(devpos), 2) if devpos else 0)
        m("HKdevPosLamHi", p(max(devpos), 2) if devpos else 0)
        m("HKdevOthLamMax", p(max(devoth), 2) if devoth else 0)
        # 两类实例的 λ 间距（§6 那句"只相隔约 0.5 个百分点"原先是手写的）
        m("HKdevGap", "%.1f" % (min(devpos) - max(devoth))
          if (devpos and devoth) else 0)
    else:
        for n, v in (("ProbeRho", 0), ("ProbeRhoP", 1), ("FamRho", 0), ("FamP", 1),
                     ("PermP", 1), ("NullPct", 0), ("FamArgmax", "-"),
                     ("HKdevPosN", 0), ("HKdevPosLamLo", 0), ("HKdevPosLamHi", 0),
                     ("HKdevOthLamMax", 0), ("HKdevGap", 0)):
            m(n, v)
    ps = data.get("probe_seeds")
    if ps:
        s = ps["summary"]
        m("PSsdOne", "%.2f" % s["1"]["sd_mean"])
        m("PSagreeOne", "%.2f" % s["1"]["agree_mean"])
        m("PSagreeMinOne", "%.2f" % s["1"]["agree_min"])
        rows = {r["instance"]: r for r in ps["rows"]}
        mk9 = rows.get("Mk09", {})
        m("PSmkNineLam", p(mk9.get("lam30", 0), 2))
        m("PSmkNineAgreeTen", "%.2f" % mk9.get("agree", {}).get("10", 0.0))
        # §6 那句"只比门限低 0.06 个百分点"原先也是手写：从门限与该实例的 λ 现算
        _thr9 = (data.get("hurink") or {}).get("thr", -1.0)
        m("PSmkNineMargin", "%.2f" % abs(float(_thr9) - mk9.get("lam30", 0)))
        m("PSn", len(ps["rows"]))
    else:
        for n, v in (("PSsdOne", 0), ("PSagreeOne", 0), ("PSagreeMinOne", 0),
                     ("PSmkNineLam", 0), ("PSmkNineAgreeTen", 0), ("PSn", 0),
                     ("PSmkNineMargin", 0)):
            m(n, v)

    # —— 开发集专家（事后量）：ρ=+0.95，用它做门控是循环论证 ——
    ap_ = os.path.join(ROOT, "logs", "aig_gating.json")
    if os.path.exists(ap_):
        ag = load_json(ap_)
        cc = ag["targets"]["vs_I_mix3"]["correlations"]
        m("MsRho", p(cc.get("ms_lever_pct", {}).get("spearman_rho", 0), 2))
        m("MsRhoP", "%.4f" % cc.get("ms_lever_pct", {}).get("spearman_p", 1.0))
        # 8 候选族（目标 B 的结构特征）的置换零假设分位——§4.3 用它说明
        # "Bonferroni 门槛过松"，与 init_probe 的 9 候选族是两个不同的族，
        # 数值上恰好都约 0.81，必须分开命名（否则就是"同名不同量"）
        fa = ag["targets"]["vs_I_mix3"].get("permutation", {}).get("all", {})
        m("CalibFamN", len(fa.get("keys", [])))
        m("CalibFamNullPct", "%.2f" % fa.get("null_p95", 0.0))
        # §3.2 那句"盒口径把双峰抹成处处微正"必须给出真实散点：
        # 非显著实例的盒口径读数并不都在 +0.3% 附近（实测 −0.58% ~ +0.95%），
        # 且其中有若干个**翻转符号**——这比"幅度塌了"更值得写出来。
        nosig = [r for r in ag["targets"]["vs_I_mix3"]["per_instance"]
                 if not is_gain(r["dhv_rel_pct"], r["p"])]
        m("BoxNoSigN", len(nosig))
        bx = [r["dhv_rel_box_pct"] for r in nosig]
        if bx:
            m("BoxNoSigLo", p(min(bx), 2))
            m("BoxNoSigHi", p(max(bx), 2))
            m("BoxNoSigMed", p(float(np.median(bx)), 2))
            m("BoxNoSigFlipN", sum(
                1 for r in nosig
                if abs(r["dhv_rel_pct"]) > 1e-12 and abs(r["dhv_rel_box_pct"]) > 1e-12
                and (r["dhv_rel_pct"] > 0) != (r["dhv_rel_box_pct"] > 0)))
        else:
            for nm in ("BoxNoSigLo", "BoxNoSigHi", "BoxNoSigMed", "BoxNoSigFlipN"):
                m(nm, 0)
    else:
        m("MsRho", 0)
        m("MsRhoP", 1)
        m("CalibFamN", 0)
        m("CalibFamNullPct", 0)
        for nm in ("BoxNoSigN", "BoxNoSigLo", "BoxNoSigHi", "BoxNoSigMed",
                   "BoxNoSigFlipN"):
            m(nm, 0)
    # —— 跨实例量级（§3.2 跨实例污染）：正文原先手写"Mk07 约 150/700、Mk10 约
    #    300/2000"并称"相差一个数量级"——但那两个实例只差 2.8 倍，例子与论断不符。
    #    实测开发集内**负载下界**最小/最大为 Mk02/Mk08，跨度 18×，"一个数量级"
    #    成立，只是例子选错了。改为从实例数据现场算（instance_hv_bounds，确定性）。
    try:
        from rmoea_d.core.instance import load_instance
        from rmoea_d.utils.metrics import instance_hv_bounds
        _b = []
        for _i in range(1, 11):
            _lo, _hi = instance_hv_bounds(
                load_instance("Mk%02d" % _i, os.path.join(ROOT, "data"), 42))
            _b.append(_lo)
        _wl = [x[1] for x in _b]
        _ms = [x[0] for x in _b]
        m("BndMinLoWl", "%.0f" % min(_wl))
        m("BndMaxLoWl", "%.0f" % max(_wl))
        m("BndWlSpan", "%.1f" % (max(_wl) / min(_wl)))
        m("BndMsSpan", "%.1f" % (max(_ms) / min(_ms)))
    except Exception:                       # 实例文件缺失时只写占位，绝不 raise
        for _n, _v in (("BndMinLoWl", 0), ("BndMaxLoWl", 0),
                       ("BndWlSpan", 0), ("BndMsSpan", 0)):
            m(_n, _v)

    # —— §3 盒口径放大：正文的核心方法学数字，全部来自 caliber_audit 的落盘 ——
    # 注意口径：正文引用的 Spearman 是对 **log10(放大倍数)** 取的秩——
    # 不写出来就是个"未声明口径"的数字（对原始放大取秩只有 -0.25, p=0.076）。
    # 两个都报，正文据此显式声明。
    cp = os.path.join(ROOT, "logs", "_caliber_pair.json")
    if os.path.exists(cp):
        cal = load_json(cp)
        PI = cal["per_instance"]
        prs = sorted({p for i in PI for p in PI[i]})
        ins = sorted(PI)
        cells = [(i, p, PI[i][p]["inst"]["rel_pct"], PI[i][p]["amplification"])
                 for i in ins for p in prs]
        raw = [abs(c[2]) for c in cells]
        amp = [c[3] for c in cells]
        # 放大倍数在个别格子上是负的（分子分母反号），取 log 前必须先取绝对值
        r_log, p_log = stats.spearmanr(raw, np.log10(np.abs(amp)))
        r_raw, p_raw = stats.spearmanr(raw, amp)
        m("CalibCells", len(cells))
        m("CalibRhoLog", p(r_log, 2))
        m("CalibRhoLogP", "%.3f" % p_log)
        m("CalibRhoRaw", p(r_raw, 2))
        m("CalibRhoRawP", "%.3f" % p_raw)
        m("CalibAmpAbsMin", "%.2f" % min(abs(a) for a in amp))
        m("CalibAmpMax", "%.1f" % max(amp))
        per = {}
        for pr in prs:
            ir = [PI[i][pr]["inst"]["rel_pct"] for i in ins]
            br = [PI[i][pr]["box"]["rel_pct"] for i in ins]
            per[pr] = (sum(br) / len(br)) / (sum(ir) / len(ir))
        m("CalibAmpLo", "%.1f" % min(per.values()))
        m("CalibAmpHi", "%.1f" % max(per.values()))
        m("CalibAmpMix", "%.1f" % per.get("D2|D1", 0))
        m("CalibAmpQpas", "%.1f" % per.get("D4|D3", 0))
        m("CalibWorstPair", max(per, key=lambda k: per[k]))
        # 阈值 0.02% 是"接近零"的显式定义（人为取的，必须写出来）。
        # 它同时也是下面分组统计的**分组依据**，所以必须与计数同源——
        # 正文里再手写一遍 0.02 就会在改阈值时漂移（缺陷 39 的同型问题）。
        CALIB_SMALL_THR = 0.02
        m("CalibSmallThr", "%.2f" % CALIB_SMALL_THR)
        lo_ = [c[3] for c in cells if abs(c[2]) <= CALIB_SMALL_THR]
        hi_ = [c[3] for c in cells if abs(c[2]) > CALIB_SMALL_THR]
        m("CalibSmallN", len(lo_))
        m("CalibSmallMed", "%.1f" % np.median(lo_) if lo_ else 0)
        m("CalibRestN", len(hi_))
        m("CalibRestMed", "%.1f" % np.median(hi_) if hi_ else 0)
        # §3.2 那句"实例边界 −0.017% → 盒口径 −0.829%"（Mk10 的 D4|D3）原先是手写的
        _m10 = (PI.get("Mk10") or {}).get("D4|D3")
        if _m10:
            m("CalMkTenQpasInst", p(_m10["inst"]["rel_pct"], 3))
            m("CalMkTenQpasBox", p(_m10["box"]["rel_pct"], 3))
    else:
        for n, v in (("CalibCells", 0), ("CalibRhoLog", 0), ("CalibRhoLogP", 1),
                     ("CalibRhoRaw", 0), ("CalibRhoRawP", 1), ("CalibAmpAbsMin", 0),
                     ("CalibAmpMax", 0), ("CalibAmpLo", 0), ("CalibAmpHi", 0),
                     ("CalibAmpMix", 0), ("CalibAmpQpas", 0), ("CalibWorstPair", "-"),
                     ("CalibSmallThr", 0), ("CalibSmallN", 0), ("CalibSmallMed", 0),
                     ("CalibRestN", 0),
                     ("CalibRestMed", 0), ("CalMkTenQpasInst", 0),
                     ("CalMkTenQpasBox", 0)):
            m(n, v)

    # —— 实验设置常量（正文里手写了 10 处以上）——
    # ref / N_p / G / n 这些值散落在 §1、§3.2、§3.6、§4.2、§5；改一次预算就要全改，
    # 且"改了数据忘了改正文"正是缺陷 30/32/34 的成因。宏化后只有一个来源。
    # 值取自本轮实际执行的设置（logs/ 里每个 run 的 components 可核）；
    # 这四个常量已提到模块级，好让生成表的表注（`tab_targets`）也能引用。
    m("SetNp", SET_NP)
    m("SetG", SET_G)
    m("SetGMax", SET_GMAX)
    m("SetSeeds", SET_SEEDS)
    m("SetRefLo", "1.02")
    # "有增益"的判据门槛：必须与代码里的 GAIN_P **同一来源**——
    # 正文写着 0.05、代码改成 0.01 而没人发现，就是最典型的静默漂移。
    m("SigLevel", "%.2f" % GAIN_P)
    # 与上述撞值的两个量必须**各自开宏**（同 HKnearBand vs HKdevGap 的先例）：
    # `100` 在正文里既是种群规模、又是百分号基数；`2000` 既是 G 上限、
    # 又是置换重抽次数。不加区分就会被守卫判成"同一指标两处来源"。
    m("PctBase", 100)
    m("PermSubsets", 2000)
    # 响应面的预算档是**列表**，做成整串宏（含 $ 与 \%，非纯数值）——
    # 若拆成单个数值会与正文里到处出现的百分数撞值。列表本身取自
    # `SURF_BUDGETS`，与 `fig_surface()` 画的横轴同源。
    m("SurfBudgets", "$" + ",".join("%g" % (100.0 * f) + r"\%"
                                    for f in SURF_BUDGETS) + "$")
    # 正文 §5.2"在 X% 预算处"那一档：从列表取，不手写（手写过一次 10）。
    m("SurfFirstBudget", "%g" % (100.0 * SURF_BUDGETS[0]))
    # 开发集实例数（§3.3/§5.4/§6/§7.4 的 $n=10$）：优先从实例表数据推导，
    # 取不到再退回协议常量——`write_macros()` 永远不许抛异常。
    _mk = (data.get("instances") or {}).get("brandimarte") or []
    m("DevN", len(_mk) if _mk else DEV_N)
    m("SeedRange", "%d--%d" % (SEED_LO, SEED_HI))
    # run 级样本量 = 开发集实例数 × seeds（正文 §3.4 那句 $n=300$）
    m("RunN", (len(_mk) if _mk else DEV_N) * SET_SEEDS)

    # —— 算力放大 10× 的回报（§5"多算一点也没有回报"那句的量化）——
    # 源 `logs/anytime_g2000.json` 的 hist_hv 轨迹（G=1..2000），取 G=200 与 G=2000 两点。
    # **用 hist_hv 而不是 final_hv**：后者是末代 archive（算法输出），前者是每代
    # population 前沿（搜索过程）。混用会把 Elite archive 的"存档增益"算进"算力回报"。
    _ag = os.path.join(ROOT, "logs", "anytime_g2000.json")
    _tenx = None
    if os.path.exists(_ag):
        try:
            _d5 = [r for r in load_json(_ag)
                   if r.get("label") == "D5" and isinstance(r.get("hist_hv"), list)
                   and len(r["hist_hv"]) >= SET_GMAX]
            if _d5:
                _tenx = float(np.mean([
                    100.0 * (r["hist_hv"][SET_GMAX - 1] - r["hist_hv"][SET_G - 1])
                    / r["hist_hv"][SET_G - 1] for r in _d5]))
        except Exception:
            _tenx = None
    m("AnytimeTenXPct", p(_tenx, 2) if _tenx is not None else 0)
    # 倍数本身（正文 §8.1"即 X× 算力"）：$G_\text{上限}/G$，不是一个独立观测，
    # 手写一次就会与上面两个设置常量脱钩。
    m("AnytimeTenX", SET_GMAX // SET_G)

    # —— T 动作空间穷举扫描（§5）：正文那句 "+0.31\%（p=0.78）" 原先是**盒口径** ——
    # 缺陷 39：该数字出自 `t_leverage_analysis.py`，它对**该文件里所有臂的前沿并集**
    # 取归一化边界（盒口径），在 Mk10 上把 T15 vs T50 放大成 +0.31\% 并给 p=0.777；
    # 同一批数据用**实例边界口径**（落盘 final_hv，与臂集无关）只有 +0.128\%
    # （p=0.073），且**方向相反**：T15 才是全部 6 档中的最优。
    # 论文其余数字全是实例边界口径，混用而不声明即违反本节的口气纪律。
    tl = os.path.join(ROOT, "logs", "_mk10_lab.json")
    _T_ALL = ("T05", "T10", "T15", "T20", "T50", "T100")
    _T_SPACE = ("T05", "T10", "T15", "T20")
    _tsc = None
    if os.path.exists(tl):
        try:
            _by = {}
            for _r in load_json(tl):
                if _r.get("label") in _T_ALL and _r.get("final_hv") is not None:
                    _by.setdefault(_r["label"], {})[_r["seed"]] = float(_r["final_hv"])
            _ss = sorted(set.intersection(*[set(_by[t]) for t in _T_ALL])) if _by else []
            if _ss:
                _mu = {t: float(np.mean([_by[t][s] for s in _ss])) for t in _T_ALL}
                _bi = max(_T_SPACE, key=lambda t: _mu[t])
                _wi = min(_T_SPACE, key=lambda t: _mu[t])
                _ba = max(_T_ALL, key=lambda t: _mu[t])
                _pr = paired([_by[_bi][s] for s in _ss], [_by[_wi][s] for s in _ss])
                _tsc = {"gap": 100.0 * (_mu[_bi] - _mu[_wi]) / _mu[_wi],
                        "p": _pr["p"], "wins": _pr["wins"], "n": _pr["n"],
                        "dz": _pr["dz"],
                        "space_is_global": _bi == _ba, "best": _bi,
                        "n_all": len(_T_ALL)}
        except Exception:
            _tsc = None
    if _tsc:
        m("TscanGapPct", p(_tsc["gap"], 3))
        m("TscanGapP", "%.3f" % _tsc["p"])
        m("TscanDz", p(_tsc["dz"], 2))
        m("TscanWins", "%d/%d" % (_tsc["wins"], _tsc["n"]))
        m("TscanSpaceGlobal", 1 if _tsc["space_is_global"] else 0)
        m("TscanBestTag", _tsc["best"])
        if _tsc["space_is_global"]:
            _note = ("空间内那个最优的 $T$ 同时就是全部 %d 档候选中的最优"
                     "——动作空间没有漏掉任何有意义的东西，" % _tsc["n_all"])
        else:
            _note = ("空间内最优并非全部 %d 档候选中的最优，"
                     "动作空间仍可能漏掉更好的 $T$，" % _tsc["n_all"])
        m("TscanNote", _note)
    else:
        for _n, _v in (("TscanGapPct", 0), ("TscanGapP", 1), ("TscanDz", 0),
                       ("TscanWins", "0/0"),
                       ("TscanSpaceGlobal", 0), ("TscanBestTag", "--")):
            m(_n, _v)
        m("TscanNote", "（$T$ 扫描数据缺失，本节结论待补。）")

    if write:
        write_tex("macros.tex", "\n".join(L) + "\n")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-missing", action="store_true",
                    help="数据源缺失时跳过而不是报错（分阶段出稿用）")
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "paper_export.json"))
    args = ap.parse_args()

    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    print("=" * 92)
    print("论文导出：logs/ -> paper/{tables,figures}")
    print("=" * 92)
    data = {}
    fig_concept(data, args.allow_missing)   # §2 概念示意图（示意点，不读 logs/）
    fig_caliber(data, args.allow_missing)   # §4 口径示意图（示意点，不读 logs/）
    data["instances"] = tab_instances(data, args.allow_missing)
    tab_ladder(data, args.allow_missing)
    fig_surface(data, args.allow_missing)
    fig_ladder(data, args.allow_missing)
    tab_targets(data, args.allow_missing)
    hurink_gate(data, args.allow_missing)
    fig_gate(data)
    tab_probe_seeds(data, args.allow_missing)
    tab_rl50(data, args.allow_missing)
    write_macros(data, args.allow_missing)
    with io.open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1, default=float)
    print("\n已写出 %s" % os.path.relpath(args.out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
