#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LS1 算子语义偏离的判定（论文 §4.6 原文 vs 本实现）。

问题
----
论文 §4.6 列了五个局部搜索算子，其 LS1 是

    LS1: Find the last finished operation O_{i,Θ_i}, move O_{i,Θ_i} to another
         machine M′ with minimum processing time.

即「**最后完工**的那道工序 → **加工时间最小**的另一台机器」——**完全确定性**，
不由随机数决定（"最后完工"与"最小加工时间"都由解本身算出）。

本实现的 `rvns.ls1_swap_machine` 是

    随机选一道工序 → 换成候选集里**随机**一台机器

两者都只改机器、不改工序序列（输出必然合法，可跳过 `_repair_ma_for_os`），
但**邻域完全不同**。这不是笔误级别的小事：LS1 是五个算子之一、由 SM/FM 轮盘赌
选出，它决定了一部分搜索步长，**因此会改变整条搜索轨迹**（而且论文 LS1 不消耗
随机数，换用它还会让后续随机流整体错位）。

> ⚠ 这条偏离此前从未被记录，原因之一是**审计基准本身错了**：
> `docs/rmoead-paper-cn.md` 把 LS1 误译成"随机选择两个工序并交换它们的位置"，
> 照那个描述去核对会得到"实现一致"的**假结论**（缺陷 55，已更正）。

做法
----
新增臂 `RVNSonly_LS1paper`（`rvns_ls_table="paper"`）与 `RVNSonly`
（`rvns_ls_table="impl"`，默认）**只差第 0 个算子**，其余（`fixed_T=10` +
RVNS + `ls_trials=1` + MIX3 + 精英档案）逐项相同。

口径
----
只读落盘 `final_hv`（**实例边界口径**），不重算；相对增幅用比值之比。

用法
----
    python scripts/ls1_variant_check.py
    python scripts/ls1_variant_check.py --labs logs/_mk07_lab.json,logs/_mk09_lab.json

落盘 `logs/ls1_variant.json`（供 `scripts/paper_export.py` 生成正文宏）。
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
# 载入与 mix3_order_check 共用的两个工具函数（避免两份重复的读盘逻辑）
from mix3_order_check import instance_of, load                      # noqa: E402

DEFAULT_LABS = ["logs/_mk07_lab.json", "logs/_mk09_lab.json", "logs/_mk10_lab.json"]
ARM_PAPER, ARM_IMPL = "RVNSonly_LS1paper", "RVNSonly"
GAIN_P = 0.05


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labs", default=",".join(DEFAULT_LABS))
    ap.add_argument("--out", default="logs/ls1_variant.json")
    args = ap.parse_args()
    labs = [x.strip() for x in args.labs.split(",") if x.strip()]

    print("=" * 104)
    print("LS1 语义：论文 [最后完工工序→最小加工时间机器]（%s） vs 本实现 [随机工序→随机候选机器]（%s）"
          % (ARM_PAPER, ARM_IMPL))
    print("=" * 104)

    rows, dw, dn = [], 0, 0
    for path in labs:
        inst = instance_of(path)
        if inst is None:
            continue
        by = load(path, inst)
        if ARM_PAPER not in by:
            print("  [!] %s 缺 %s（先跑：python scripts/t_leverage_sweep.py "
                  "--instance %s --arms %s --n_runs 30 --out %s）"
                  % (path, ARM_PAPER, inst, ARM_PAPER, path))
            continue
        ss = sorted(set(by[ARM_PAPER]) & set(by[ARM_IMPL]))
        if len(ss) < 2:
            print("  [!] %s 跨臂 seed 交集只有 %d 个" % (path, len(ss)))
            continue
        va = np.array([by[ARM_PAPER][s] for s in ss])
        vb = np.array([by[ARM_IMPL][s] for s in ss])
        d = va - vb
        p = float(stats.wilcoxon(va, vb)[1])
        rel = float(d.mean() / vb.mean() * 100.0)
        dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) else float("nan")
        wins = int((d > 0).sum())
        dw += wins
        dn += len(ss)
        rows.append({"instance": inst, "n": len(ss), "mean_paper": float(va.mean()),
                     "mean_impl": float(vb.mean()), "dhv": float(d.mean()),
                     "rel_pct": rel, "wins": wins, "dz": dz, "p": p})
        print("  %-5s n=%-3d  论文LS1 %.5f  本实现LS1 %.5f  ΔHV=%+.4e  rel=%+.4f%%  "
              "胜 %2d/%-3d  d_z=%+.3f  p=%.4g %s"
              % (inst, len(ss), va.mean(), vb.mean(), d.mean(), rel, wins, len(ss),
                 dz, p, "***" if p < 0.001 else "**" if p < 0.01 else
                 "*" if p < 0.05 else "n.s."))

    if not rows:
        return 1

    sign_p = float(stats.binomtest(dw, dn, 0.5).pvalue) if dn else float("nan")
    sig = [r for r in rows if r["p"] < GAIN_P]
    rels = [r["rel_pct"] for r in rows]
    same_sign = all(r < 0 for r in rels) or all(r > 0 for r in rels)
    print()
    print("  逐实例相对差：%s" % "  ".join("%s %+.4f%%" % (r["instance"], r["rel_pct"])
                                          for r in rows))
    print("  逐实例 p    ：%s" % "  ".join("%s %.3g" % (r["instance"], r["p"])
                                          for r in rows))
    print("  合并 run 级：论文 LS1 胜 %d/%d（<50%% 即论文 LS1 更差），双侧符号检验 p=%.4g"
          % (dw, dn, sign_p))
    print()
    if sig:
        print("  判定：**有 %d 个实例显著** —— 这不是可以默默带过的偏离，"
              "必须在复现说明里显式声明。" % len(sig))
    else:
        print("  判定：逐个实例都不显著；幅度最大 %.4f%%。" % max(abs(r) for r in rels))

    payload = {
        "rows": rows,
        "pooled": {"wins": dw, "n": dn, "p": sign_p},
        "summary": {
            "n_instances": len(rows),
            "n_seeds": min(r["n"] for r in rows),
            "rel_abs_max": max(abs(r) for r in rels),
            "n_sig": len(sig),
            "same_sign": same_sign,
            "sign": (-1 if all(r < 0 for r in rels)
                     else 1 if all(r > 0 for r in rels) else 0),
        },
        "meta": {"labs": labs, "arms": [ARM_PAPER, ARM_IMPL],
                 "caliber": "instance-boundary (final_hv, 落盘值直读)"},
    }
    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print("\n[OK] -> %s（%d 行）" % (args.out, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
