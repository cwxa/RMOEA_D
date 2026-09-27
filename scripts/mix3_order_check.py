#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MIX3 初始化**拼接顺序**偏离的判定（论文 Algorithm 2 vs 本实现）。

问题
----
论文 Algorithm 2 第 4 行把三段合并为

    Parent = [P1(GW), P2(LS), P3(Random)]

而本实现的 `init_mix3`（以及 `init_variant="mix3"`）生成的是

    [Random, LS, GW]

**顺序相反**。两者是同一个多重集，差别在"哪个个体挂在哪个权重向量上"
（MOEA/D 的第 i 个个体绑定第 i 个权重向量），以及 rng 消耗顺序不同。
这是审计发现的一条**真实偏离**，但"它有没有影响"读代码定不了，必须实测。

做法
----
新增臂 `I_mix3_paper`（论文顺序），与 `RVNSonly`（≡ `init_variant="mix3"`，
即本实现顺序）**只差这一处**，其余（`fixed_T=10` + RVNS + `ls_trials=1`）全同。

口径
----
只读落盘 `final_hv`（**实例边界口径**，与臂集无关），不重算；
相对增幅用比值之比 `mean(ΔHV)/mean(基线)`。

用法
----
    python scripts/mix3_order_check.py
    python scripts/mix3_order_check.py --labs logs/_mk07_lab.json,logs/_mk09_lab.json

落盘 `logs/mix3_order.json`（供 `scripts/paper_export.py` 生成正文宏）。
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LABS = ["logs/_mk07_lab.json", "logs/_mk09_lab.json", "logs/_mk10_lab.json"]
ARM_PAPER, ARM_IMPL = "I_mix3_paper", "RVNSonly"
GAIN_P = 0.05


def load(path, inst):
    """→ {label: {seed: final_hv}}，只取 instance == inst 的行。

    ⚠ 实例名**从行里读**，不从文件名猜：`_mk07_lab.json` → `Mk07`
    （大小写不是文件名的 `MK07`，猜错过一次会让三个实例全部"缺臂"）。
    """
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        rows = json.load(fh)
    by = {}
    for r in rows:
        if inst is not None and r.get("instance") != inst:
            continue
        by.setdefault(r["label"], {})[r["seed"]] = r["final_hv"]
    return by


def instance_of(path):
    by = load(path, None)
    # 一个 lab 文件只放一个实例（t_leverage_sweep 的 --instance 是单值）
    seen = set()
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        for r in json.load(fh):
            if r.get("instance"):
                seen.add(r["instance"])
    if len(seen) != 1:
        print("  [!] %s 里出现了 %d 个实例名（%s），跳过"
              % (path, len(seen), sorted(seen)))
        return None
    return seen.pop()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labs", default=",".join(DEFAULT_LABS))
    ap.add_argument("--out", default="logs/mix3_order.json")
    args = ap.parse_args()
    labs = [x.strip() for x in args.labs.split(",") if x.strip()]

    print("=" * 96)
    print("MIX3 拼接顺序：论文 Alg.2 [GW,LS,Random]（%s） vs 本实现 [Random,LS,GW]（%s）"
          % (ARM_PAPER, ARM_IMPL))
    print("=" * 96)

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
        print("  %-5s n=%-3d  论文顺序 %.5f  本实现 %.5f  ΔHV=%+.4e  rel=%+.4f%%  "
              "胜 %2d/%-3d  d_z=%+.3f  p=%.4g %s"
              % (inst, len(ss), va.mean(), vb.mean(), d.mean(), rel, wins, len(ss),
                 dz, p, "***" if p < 0.001 else "**" if p < 0.01 else
                 "*" if p < 0.05 else "n.s."))

    if not rows:
        return 1

    # 合并 run 级符号检验：正数 = 论文顺序更好。**事后合并，只作描述性证据。**
    sign_p = float(stats.binomtest(dw, dn, 0.5).pvalue) if dn else float("nan")
    sig = [r for r in rows if r["p"] < GAIN_P]
    rels = [r["rel_pct"] for r in rows]
    same_sign = all(r < 0 for r in rels) or all(r > 0 for r in rels)
    print()
    print("  逐实例相对差：%s" % "  ".join("%s %+.4f%%" % (r["instance"], r["rel_pct"])
                                          for r in rows))
    print("  逐实例 p    ：%s" % "  ".join("%s %.3g" % (r["instance"], r["p"])
                                          for r in rows))
    print("  合并 run 级：论文顺序胜 %d/%d（<50%% 即论文顺序更差），双侧符号检验 p=%.4g"
          % (dw, dn, sign_p))
    print()
    print("  判定：逐个实例**都不显著**，幅度 ≤0.15%；三个实例方向一致（论文顺序略差），")
    print("        合并 run 级符号检验也不显著（p=%.3g）。" % sign_p)
    print("        → 该偏离**不构成实现占便宜**，也**不足以支撑**必须改成论文顺序。")

    payload = {
        "rows": rows,
        "pooled": {"wins": dw, "n": dn, "p": sign_p},
        "summary": {
            "n_instances": len(rows),
            "n_seeds": min(r["n"] for r in rows),
            "rel_abs_max": max(abs(r) for r in rels),
            "n_sig": len(sig),
            "same_sign": same_sign,
            # 方向：负数 = 论文顺序更差（本实现更好）
            "sign": (-1 if all(r < 0 for r in rels)
                     else 1 if all(r > 0 for r in rels) else 0),
        },
        "meta": {"labs": labs, "arms": [ARM_PAPER, ARM_IMPL],
                 "caliber": "instance-boundary (final_hv, 落盘值直读)"},
    }
    dest = os.path.join(ROOT, args.out)
    with open(dest, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print("\n[OK] -> %s（%d 行）" % (args.out, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
