# -*- coding: utf-8 -*-
"""把「算力-算法响应面」的结果渲染成报告用的 markdown 表格。

为什么单独一个脚本
------------------
工程纪律里有一条「图上的数字不要写死」（fig6 曾硬编码 8.60/28.71，报告改了图没改）。
同一条推论适用于**报告正文里的表格**：手抄的表格会随数据更新而陈旧，且无法复核。
本脚本只做「结构化结果 → markdown」，**不重算任何统计量**：

* 等代数的绝对 HV / 配对差  —— 从原始 lab 的 `hist_hv` 逐 seed 取值（与绘图同源）
* 饱和诊断 / 等 FE 表        —— 直接读 `response_surface.py` 落盘的 surface json
  （避免两处各算一遍、口径悄悄分叉）

用法
----
    python scripts/surface_markdown.py --lab logs/anytime_g2000.json \\
        --surface logs/anytime_g2000.surface.json --gens 200,500,1000,2000 \\
        --out docs/_tables_anytime_g2000.md
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from response_surface import load_many, index_curves, _pstat  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_GENS = [200, 500, 1000, 2000]
DEFAULT_PAIRS = [("D2", "D1"), ("D5", "D2"), ("D5", "D1"), ("RMOEAD", "D5")]


def _fmt_p(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "**%.1e**" % p
    if p < 0.05:
        return "**%.3f**" % p
    return "%.3f n.s." % p


def abs_hv_table(idx, instances, labels, gens):
    """§3.1a：各臂在不同代数处的绝对 HV（实例边界口径，逐 seed 取该代值再平均）。"""
    out = ["| 实例 | 臂 | " + " | ".join("HV@G%d" % g for g in gens)
           + " | G%d/G%d |" % (gens[-1], gens[0])]
    out.append("|---|---|" + "---|" * (len(gens) + 1))
    for inst in instances:
        for lab in labels:
            d = idx.get((inst, lab))
            if not d:
                continue
            vals = []
            for g in gens:
                v = [h[g - 1] for _t, h in d.values() if h.size >= g]
                vals.append(float(np.mean(v)) if v else float("nan"))
            if not np.isfinite(vals[0]) or vals[0] == 0:
                continue
            rel = (vals[-1] / vals[0] - 1.0) * 100.0
            out.append("| %s | %s | " % (inst, lab)
                       + " | ".join("%.6f" % x for x in vals)
                       + " | **+%.3f%%** |" % rel)
    return out


def pair_table_at_gens(idx, instances, pairs, gens):
    """§3.1b：同一批代数处的配对差（同 seed，Wilcoxon）。"""
    out = ["| 实例 | 臂对 | " + " | ".join("G=%d" % g for g in gens) + " |"]
    out.append("|---|---|" + "---|" * len(gens))
    for inst in instances:
        for a, b in pairs:
            if (inst, a) not in idx or (inst, b) not in idx:
                continue
            da, db = idx[(inst, a)], idx[(inst, b)]
            seeds = sorted(set(da) & set(db))
            if not seeds:
                continue
            cells = []
            for g in gens:
                v = np.asarray([da[s][1][g - 1] - db[s][1][g - 1]
                                for s in seeds
                                if da[s][1].size >= g and db[s][1].size >= g])
                if v.size == 0:
                    cells.append("n/a")
                    continue
                diff, wins, p, _dz = _pstat(v)
                base = float(np.mean([db[s][1][g - 1] for s in seeds
                                      if db[s][1].size >= g]))
                rel = (diff / base * 100.0) if base else float("nan")
                cells.append("%+.5f, %+.3f%%, %d/%d, p=%s"
                             % (diff, rel, wins, v.size, _fmt_p(p)))
            out.append("| %s | **%s − %s** | " % (inst, a, b) + " | ".join(cells) + " |")
    return out


def saturation_table(surface, instances, labels):
    """§3.2：饱和诊断（每段 HV 平均增量、末/首比）。"""
    sat = surface.get("saturation", {})
    nseg = 0
    for inst in sat:
        for lab in sat[inst]:
            nseg = max(nseg, len(sat[inst][lab]["gains"]))
    out = ["| 实例 | 臂 | " + " | ".join("第%d段" % (i + 1) for i in range(nseg))
           + " | 末/首 |"]
    out.append("|---|---|" + "---|" * (nseg + 1))
    for inst in instances:
        for lab in labels:
            r = sat.get(inst, {}).get(lab)
            if not r:
                continue
            out.append("| %s | %s | " % (inst, lab)
                       + " | ".join("%+.5f" % x for x in r["gains"])
                       + " | **%.3f** |" % r["ratio_last_first"])
    return out


def fe_table(surface, instances, pairs):
    """§3.3：等求值次数（FE）配对差。"""
    fe = surface.get("fe", {})
    fracs, cap = [], {}
    for key in fe:
        for inst in fe[key]:
            for rec in fe[key][inst]:
                fracs.append(rec["fe_frac"])
                cap.setdefault(inst, rec["FE"] / rec["fe_frac"])
    fracs = sorted(set(fracs))
    # 各实例公共 FE 上限可不同 → 表头用「占上限比例」，并在脚注给出绝对值
    out = ["| 实例 | 臂对 | " + " | ".join("%.0f%% FE" % (f * 100.0) for f in fracs) + " |"]
    out.append("|---|---|" + "---|" * len(fracs))
    for inst in instances:
        for a, b in pairs:
            rec = fe.get("%s|%s" % (a, b), {}).get(inst)
            if not rec:
                continue
            by = {r["fe_frac"]: r for r in rec}
            cells = []
            for f in fracs:
                r = by.get(f)
                cells.append("%+.4f" % r["diff"] if r else "n/a")
            out.append("| %s | **%s − %s** | " % (inst, a, b) + " | ".join(cells) + " |")
    if cap:
        out.append("")
        out.append("各实例公共 FE 上限："
                   + "、".join("%s = %d" % (i, int(cap[i])) for i in instances if i in cap))
    return out


def final_table(idx, rows, instances, pairs):
    """§3.4：终点口径 `final_hv` 配对（实例边界，与臂集无关）。"""
    fhv = {}
    for r in rows:
        fhv.setdefault((r["instance"], r["label"]), {})[r["seed"]] = float(r["final_hv"])
    out = ["| 实例 | 臂对 | diff | rel% | wins | p | 结论 |", "|---|---|---|---|---|---|---|"]
    for inst in instances:
        for a, b in pairs:
            da, db = fhv.get((inst, a), {}), fhv.get((inst, b), {})
            seeds = sorted(set(da) & set(db))
            if not seeds:
                continue
            va = np.asarray([da[s] for s in seeds])
            vb = np.asarray([db[s] for s in seeds])
            v = va - vb
            diff, wins, p, _dz = _pstat(v)
            rel = diff / vb.mean() * 100.0 if vb.mean() else float("nan")
            concl = "显著" if p < 0.05 else "n.s."
            if p < 0.05 and diff < 0:
                concl = "**显著为负**"
            out.append("| %s | %s − %s | %+.6f | %+.4f%% | %d/%d | %s | %s |"
                       % (inst, a, b, diff, rel, wins, len(seeds), _fmt_p(p), concl))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lab", default=os.path.join(ROOT, "logs", "anytime_g2000.json"))
    ap.add_argument("--surface", default="")
    ap.add_argument("--gens", default=",".join(str(g) for g in DEFAULT_GENS))
    ap.add_argument("--pairs", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    gens = [int(x) for x in args.gens.split(",") if x.strip()]
    pairs = DEFAULT_PAIRS
    if args.pairs.strip():
        pairs = [tuple(s.split("=")) for s in args.pairs.split(",") if "=" in s]

    rows = load_many(args.lab)
    idx = index_curves(rows)
    instances = sorted({k[0] for k in idx})
    labels = [l for l in ("D1", "D2", "D3", "D4", "D5", "RMOEAD") if any(k[1] == l for k in idx)]

    surface = {}
    sp = args.surface or (args.lab.split(",")[0].strip() + ".surface.json")
    if os.path.exists(sp):
        with open(sp, "r", encoding="utf-8") as fh:
            surface = json.load(fh)

    L = []
    L.append("<!-- 由 scripts/surface_markdown.py 生成，请勿手改 -->")
    L.append("")
    L.append("### A. 绝对 HV（实例边界口径，30 seeds）")
    L.append("")
    L += abs_hv_table(idx, instances, labels, gens)
    L.append("")
    L.append("### B. 配对差（同 seed，Wilcoxon）")
    L.append("")
    L += pair_table_at_gens(idx, instances, pairs, gens)
    if surface.get("saturation"):
        L.append("")
        L.append("### C. 饱和诊断（每段 HV 平均增量）")
        L.append("")
        L += saturation_table(surface, instances, labels)
    if surface.get("fe"):
        L.append("")
        L.append("### D. 等求值次数（FE）配对差")
        L.append("")
        L += fe_table(surface, instances, pairs)
    L.append("")
    L.append("### E. 终点口径 `final_hv` 配对")
    L.append("")
    L += final_table(idx, rows, instances, pairs)
    L.append("")

    txt = "\n".join(L)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(txt)
        print("已写出 %s" % args.out)
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
