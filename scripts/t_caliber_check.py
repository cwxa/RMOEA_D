#!/usr/bin/env python3
"""Mk10 固定-T 扫参：**实例边界口径**统计（缺陷 40 的复核工具）。

背景
----
``scripts/t_leverage_analysis.py`` 的 HV 一律用**盒口径**（所有臂/所有 run 的
前沿并集取归一化边界）。按缺陷 30 / 39：盒边界随**臂集**漂移，同一份 lab
换一批臂就会得到另一套 rel% / p / 甚至另一个"最优档"。因此：

* 论文里的效应量一律读落盘的 ``final_hv``（**实例边界口径**，与臂集无关）；
* 盒口径**只允许出现在分析脚本**里，且必须同时声明臂集。

本脚本给出两种口径的并列结果，用来核对 ``docs/qpas-implementation-audit.md``
中"T 的收益曲线在 T=5 之外几乎是平的"这一论证是否仍然成立。

    python scripts/t_caliber_check.py --lab logs/_mk10_lab.json
    python scripts/t_caliber_check.py --md           # 输出 markdown 表格

输出：
  [1] 各档 mean / sd（实例边界口径）
  [2] 全档与"动作空间内"的极差 + 逐 seed 配对检验
  [3] 排除 T=5 后的极差
  [4] Friedman（两种口径并列）
  [5] 盒口径对照 —— **附臂集声明**，只作灵敏度参考
"""
import argparse
import collections
import io
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from rmoea_d.utils.metrics import compute_hv, estimate_hv_bounds  # noqa: E402

# 论文 §4.5.2 / Table 6 的动作空间 = {5,10,15,20}；T50/T100 是**空间外**的对照档
SPACE = ("T05", "T10", "T15", "T20")
OUTER = ("T50", "T100")
ALL_T = SPACE + OUTER


def load(path):
    return json.load(io.open(path, encoding="utf-8"))


def group(rows, labels):
    by = collections.defaultdict(dict)
    for r in rows:
        if r.get("label") in labels and r.get("final_pf") is not None:
            by[r["label"]][r["seed"]] = float(r["final_hv"])
    seeds = sorted(set.intersection(*[set(by[t]) for t in labels]))
    return by, seeds


def paired(by, seeds, a, b):
    """左 a vs 右 b；正号 = b 更高。"""
    x = np.array([by[a][s] for s in seeds])
    y = np.array([by[b][s] for s in seeds])
    d = y - x
    if np.allclose(d, 0):
        return 0.0, 1.0, 0.0, 0
    _, p = stats.wilcoxon(x, y)
    sd = d.std(ddof=1)
    return float(d.mean()), float(p), float(d.mean() / sd if sd > 0 else 0.0), int((d > 0).sum())


def box_hv(rows, labels, seeds):
    """盒口径对照：边界由**传入 labels 的全部 run** 前沿并集构造（臂集必须声明）。"""
    fronts = [np.asarray(r["final_pf"], float) for r in rows if r.get("final_pf")]
    lo, hi = estimate_hv_bounds(fronts)
    out = {}
    for t in labels:
        vals = []
        for s in seeds:
            r = next(r for r in rows if r.get("label") == t and r.get("seed") == s)
            vals.append(compute_hv(np.asarray(r["final_pf"], float), norm_bounds=(lo, hi)))
        out[t] = float(np.mean(vals))
    return out, lo, hi, len(fronts)


def main():
    ap = argparse.ArgumentParser(description="Mk10 固定-T 扫参的实例边界口径复核")
    ap.add_argument("--lab", default=os.path.join(ROOT, "logs", "_mk10_lab.json"),
                    help="t_leverage_sweep.py 产出的 JSON（默认 logs/_mk10_lab.json）")
    ap.add_argument("--md", action="store_true", help="输出 markdown 表格")
    ap.add_argument("--extra", default="",
                    help="额外对照臂（逗号分隔），逐个与最优固定 T 做配对检验；"
                         "例：--extra QPAS2_hv_wide,QPAS2_dv")
    args = ap.parse_args()

    rows = load(args.lab)
    labels = [t for t in ALL_T
              if any(r.get("label") == t for r in rows)]
    by, seeds = group(rows, labels)
    n = len(seeds)
    print("# 数据 %s ｜ 档 %s ｜ 公共 seed %d" % (os.path.basename(args.lab), ",".join(labels), n))

    sd = {t: float(np.std([by[t][s] for s in seeds], ddof=1)) for t in labels}
    mean = {t: float(np.mean([by[t][s] for s in seeds])) for t in labels}

    # [1] 逐档
    print("\n## [1] 各档（实例边界口径 = 落盘 final_hv）")
    if args.md:
        print("| 档 | mean HV | run 间 sd |")
        print("|---|---:|---:|")
        for t in labels:
            print("| `%s`%s | %.6f | %.6f |" % (t, " **← 最优**"
                  if t == max(labels, key=lambda x: mean[x]) else "", mean[t], sd[t]))
    else:
        for t in labels:
            print("  %-5s mean=%.6f  sd=%.6f" % (t, mean[t], sd[t]))

    # [2] 极差
    best = max(labels, key=lambda t: mean[t])
    worst = min(labels, key=lambda t: mean[t])
    gap_all = 100.0 * (mean[best] - mean[worst]) / mean[worst]
    sp = [t for t in labels if t in SPACE]
    b_sp = max(sp, key=lambda t: mean[t]); w_sp = min(sp, key=lambda t: mean[t])
    gap_sp = 100.0 * (mean[b_sp] - mean[w_sp]) / mean[w_sp]

    print("\n## [2] 极差")
    print("  全 %d 档: 最优 %s 最差 %s → %+.4f%%" % (len(labels), best, worst, gap_all))
    print("  空间内(%s): 最优 %s 最差 %s → %+.4f%%" % (",".join(SPACE), b_sp, w_sp, gap_sp))
    if b_sp == best:
        print("  ⇒ **空间内最优 == 全空间最优**（动作空间的天花板为 0）")
    else:
        print("  ⇒ 空间外还有更高档 %s（天花板 > 0）" % best)

    m, p, dz, w = paired(by, seeds, w_sp, b_sp)
    print("  配对 %s→%s: ΔHV=%+.6f (%+.4f%%) p=%.4f dz=%+.3f wins=%d/%d"
          % (w_sp, b_sp, m, 100.0 * m / mean[w_sp], p, dz, w, n))

    # [3] 排除 T=5
    no5 = [t for t in labels if t not in ("T05", "T5")]
    b2 = max(no5, key=lambda t: mean[t]); w2 = min(no5, key=lambda t: mean[t])
    print("\n## [3] 排除 T=5 后（audit 文档那句'极差 << run 间 sd'的复算）")
    print("  最优 %s=%.6f  最差 %s=%.6f  → 极差 %.4f%%（绝对 %.6f）"
          % (b2, mean[b2], w2, mean[w2], 100.0 * (mean[b2] - mean[w2]) / mean[w2],
             mean[b2] - mean[w2]))
    print("  run 间 sd = %.6f ~ %.6f  → 极差 / sd = %.2f"
          % (min(sd[t] for t in no5), max(sd[t] for t in no5),
             (mean[b2] - mean[w2]) / max(sd[t] for t in no5)))

    # [4] Friedman
    print("\n## [4] Friedman over %d 档" % len(labels))
    cols = [np.array([by[t][s] for s in seeds]) for t in labels]
    st, p_f = stats.friedmanchisquare(*cols)
    print("  实例边界口径: chi2=%.3f p=%.4g %s" % (st, p_f, "显著" if p_f < 0.05 else "不显著"))

    # [5] 盒口径对照
    bm, lo, hi, nf = box_hv(rows, labels, seeds)
    bb = max(labels, key=lambda t: bm[t]); ww = min(labels, key=lambda t: bm[t])
    print("\n## [5] 盒口径对照（⚠ 仅灵敏度参考，臂集绑定）")
    print("  臂集 = 本文件全部 %d 行（%d 档 × %d seeds）；盒边界 lo=%s hi=%s"
          % (nf, len(labels), n, lo, hi))
    print("  最优 %s 最差 %s → 极差 %+.4f%%（比实例边界口径放大 %.1f×）"
          % (bb, ww, 100.0 * (bm[bb] - bm[ww]) / bm[ww],
             (100.0 * (bm[bb] - bm[ww]) / bm[ww]) / gap_all))
    if bb != best:
        print("  ⚠ 两口径的**最优档不同**（盒 %s vs 实例边界 %s）—— 引用前必须声明口径"
              % (bb, best))

    # [6] 额外对照臂（如 Q-PAS 各变体）vs 最优固定 T
    ex = [a.strip() for a in args.extra.split(",") if a.strip()]
    ex = [a for a in ex if any(r.get("label") == a for r in rows)]
    if ex:
        byx, sx = group(rows, labels + ex)
        mx = {t: float(np.mean([byx[t][s] for s in sx])) for t in labels + ex}
        bt = max(labels, key=lambda t: mx[t])
        print("\n## [6] 额外臂 vs 最优固定 %s（实例边界口径，n=%d）" % (bt, len(sx)))
        for a in ex:
            m, p, dz, w = paired(byx, sx, bt, a)
            print("  %-24s HV=%.6f  vs %s: %+.6f (%+.3f%%) p=%.4f wins=%d/%d"
                  % (a, mx[a], bt, m, 100.0 * m / mx[bt], p, w, len(sx)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
