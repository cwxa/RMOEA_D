#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RL 选算子效应的 n=50 复验（计划里的 P3/B2）。

问法
----
论文阶梯的最后一步（$D_5 \\to$ \\textsf{RMOEAD}）**只改一件事**：邻域算子由
"等概率随机选"换成"按 SM/FM 成功记忆轮盘赌选"（论文 Algorithm 4 的用法）。
这一步在 $n=30$ 上有两个互相矛盾的读数：

* $ls\\_trials{=}1$（论文口径，$G=200$）：$-0.003\\%$，不显著；
* $ls\\_trials{=}3$（强局部搜索）：$+1.86\\%$（$21/30$，$p=0.0449$）——
  但它在同算力的 $G{=}440$ 家族里**归零**（$+0.56\\%$，$15/30$，$p=0.730$）。

本脚本把 $n$ 提到 **50** seeds、两档算力各跑一遍，回答"这个 $+1.86\\%$ 是不是噪声"。

口径纪律（不可违背）
--------------------
1. **只读落盘的 `final_hv`，绝不重算**。它是**实例边界口径**（`instance_hv_bounds`），
   由实例数据确定性推出、与臂集无关 —— 因此同一实例内跨臂可比。
   （盒口径 `estimate_hv_bounds` 随臂集漂移，禁止用于此处的跨批比较。）
2. **相对增幅用比值之比** `mean(ΔHV) / mean(基线)`，不用 `mean(ΔHV/基线)`。
3. 实例之间绝对 HV 不可比 → **逐实例**报 rel%，不跨实例合并绝对值。
4. 等算力是**构造性**的：两臂 `ls_trials` 相同、`G` 相同、`n_pop` 相同，
   只有算子选择规则不同。脚本另外核对实测墙钟比落在 `[0.8, 1.25]`。

用法
----
    python scripts/rl50_analyze.py                       # 默认两个 lab
    python scripts/rl50_analyze.py --labs logs/_mk10_rl50.json
    python scripts/rl50_analyze.py --check-determinism   # 与既有 30-seed 数据逐位核对
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (档名, RL 选算子臂, 随机选算子臂, 说明)
PAIRS = [
    ("t1", "RVNSonly", "RandVNS",
     "论文口径：每解每代 1 次邻域尝试（Algorithm 4）"),
    ("t3", "RVNSonly_t3", "RandVNS_t3",
     "强局部搜索：每解每代最多 3 次尝试（算力 2.2×，但两臂相等）"),
]
MATCH_LO, MATCH_HI = 0.80, 1.25
DEFAULT_LABS = ["logs/_mk10_rl50.json", "logs/_mk09_rl50.json"]
# 既有 30-seed 数据（只用于确定性核对：新跑的 30 个重叠 seed 必须逐位相同）
LEGACY = {"Mk10": "logs/_mk10_lab.json", "Mk09": "logs/_mk09_lab.json"}


def stars(p):
    if p != p:
        return "n.s."
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def load_rows(labs):
    rows = []
    skipped = 0
    for p in labs:
        if not os.path.exists(p):
            raise SystemExit("缺少 lab 文件：%s" % p)
        with open(p, encoding="utf-8") as fh:
            for r in json.load(fh):
                # 旧 lab 里有一批行没记 instance（历史遗留）；它们不属于本分析
                if not r.get("instance"):
                    skipped += 1
                    continue
                rows.append(r)
    if skipped:
        print("  [!] 跳过 %d 行无 instance 字段的历史记录" % skipped)
    return rows


def analyse(rows):
    """→ (rows_out, header_lines, problems, note)。rows_out 供 tab_rl50 消费。"""
    by = collections.defaultdict(dict)          # (instance,label) -> seed -> hv
    tt = collections.defaultdict(list)          # (instance,label) -> [time]
    ev = collections.defaultdict(list)          # (instance,label) -> [实际邻域求值数/解]
    for r in rows:
        key = (r["instance"], r["label"])
        by[key][r["seed"]] = r["final_hv"]
        tt[key].append(r.get("total_time", float("nan")))
        b = r.get("rvns_budget") or {}
        if b.get("mean_actual_evals") is not None:
            ev[key].append(float(b["mean_actual_evals"]))

    instances = sorted({r["instance"] for r in rows})
    out, lines, problems = [], [], []

    for inst in instances:
        for tier, a_rl, a_rd, why in PAIRS:
            ka, kb = (inst, a_rl), (inst, a_rd)
            if ka not in by or kb not in by:
                problems.append("%s/%s 缺臂：%s" % (inst, tier, a_rl if ka not in by else a_rd))
                continue
            ss = sorted(set(by[ka]) & set(by[kb]))
            if len(ss) < 2:
                problems.append("%s/%s 跨臂 seed 交集只有 %d 个" % (inst, tier, len(ss)))
                continue
            va = np.array([by[ka][s] for s in ss], dtype=float)
            vb = np.array([by[kb][s] for s in ss], dtype=float)
            d = va - vb
            try:
                p = float(stats.wilcoxon(va, vb)[1])
            except Exception:
                p = float("nan")
            sd = d.std(ddof=1)
            dz = float(d.mean() / sd) if sd else float("nan")
            rel = float(d.mean() / vb.mean() * 100.0)   # 比值之比
            wins = int((d > 0).sum())
            tra = float(np.nanmean(tt[ka]) / np.nanmean(tt[kb]))
            if not (MATCH_LO <= tra <= MATCH_HI):
                problems.append("%s/%s 墙钟比 %.2f× 不在等算力区间" % (inst, tier, tra))

            # 「算力档」这一列写的是**实测**的每解邻域求值次数（来自 rvns_budget），
            # 不是臂名里那个设计值——否则又是"把声称当测量"。
            eva = float(np.mean(ev[ka])) if ev[ka] else float("nan")
            evb = float(np.mean(ev[kb])) if ev[kb] else float("nan")
            label = tier if not np.isfinite(eva) else \
                "%s · 每解 %.2f 次求值" % (tier, eva)

            out.append({"instance": inst, "budget": tier, "budget_label": label,
                        "rel_pct": rel, "wins": wins, "n": len(ss), "dz": dz, "p": p,
                        "time_ratio": tra, "evals_rl": eva, "evals_random": evb,
                        "arm_rl": a_rl, "arm_random": a_rd})
            lines.append("%-6s %-3s  n=%-3d  ΔHV=%+.3e  胜 %2d/%-3d  dz=%+.2f  "
                         "p=%.4g %-5s  墙钟 %.2f×  求值 %.2f/%.2f  [%s vs %s]"
                         % (inst, tier, len(ss), d.mean(), wins, len(ss), dz, p,
                            stars(p), tra, eva, evb, a_rl, a_rd))

    note = build_note(out)
    return out, lines, problems, note


def build_note(out):
    """表注**全部由数据生成**（生成物里不许有写死的数字）。"""
    pos = [r for r in out if r["p"] == r["p"] and r["p"] < 0.05 and r["rel_pct"] > 0]
    neg = [r for r in out if r["p"] == r["p"] and r["p"] < 0.05 and r["rel_pct"] < 0]
    parts = ["\u5bf9\u6bcf\u4e2a\u5b9e\u4f8b\u4e0e\u7b97\u529b\u6863\uff0c"
             "\u540c seed \u914d\u5bf9\u6bd4\u8f83\u201cRL \u9009\u7b97\u5b50\u201d"
             "\u4e0e\u201c\u968f\u673a\u9009\u7b97\u5b50\u201d\uff08\u4e24\u81c2 "
             "$ls\\_trials$ \u4e0e $G$ \u76f8\u540c\uff0c\u7b49\u7b97\u529b\u7531"
             "\u6784\u9020\u4fdd\u8bc1\uff09\u3002"]
    if not pos and not neg:
        parts.append("\u5168\u90e8 $n{=}%d$ \u7ec4\u5bf9\u6bd4\u5747\u4e0e 0 "
                     "\u4e0d\u53ef\u533a\u5206\uff08\u914d\u5bf9 Wilcoxon\uff09\u3002"
                     % max(r["n"] for r in out))
    else:
        parts.append("\u663e\u8457\u4e14\u4e3a\u6b63\u7684\u6709 %d \u7ec4\uff0c"
                     "\u663e\u8457\u4e14\u4e3a\u8d1f\u7684 %d \u7ec4\u3002"
                     % (len(pos), len(neg)))
    parts.append("$\\Delta$HV \u4e3a\u5b9e\u4f8b\u8fb9\u754c\u53e3\u5f84\u7684"
                 "\u76f8\u5bf9\u589e\u5e45\uff08\u6bd4\u503c\u4e4b\u6bd4\uff09\uff1b"
                 "\u7edd\u5bf9 HV \u4e0d\u8de8\u5b9e\u4f8b\u53ef\u6bd4\u3002")
    return "".join(parts)


def check_determinism(labs):
    """新跑的 30 个重叠 seed（42–71）必须与既有 30-seed lab **逐位相同**。"""
    rows = load_rows(labs)
    bad, tot = [], 0
    for inst, path in LEGACY.items():
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            old = {(r["label"], r["seed"]): r["final_hv"] for r in json.load(fh)}
        for r in rows:
            if r["instance"] != inst:
                continue
            k = (r["label"], r["seed"])
            if k in old:
                tot += 1
                if old[k] != r["final_hv"]:
                    bad.append("%s %s seed=%d: %r vs %r"
                               % (inst, k[0], k[1], old[k], r["final_hv"]))
    return tot, bad


def summarise(out):
    """汇总（供正文宏使用）。**全部由 rows 派生**，不写死。"""
    ns = sorted({r["n"] for r in out})
    pv = [r["p"] for r in out if r["p"] == r["p"]]
    sig = [r for r in out if r["p"] == r["p"] and r["p"] < 0.05 and r["rel_pct"] > 0]
    return {
        "n_seeds": ns[0] if len(ns) == 1 else ns,
        "n_tiers": len({r["budget"] for r in out}),
        "n_rows": len(out),
        "rel_abs_max": max(abs(r["rel_pct"]) for r in out) if out else 0.0,
        "p_min": min(pv) if pv else float("nan"),
        "sig_pos": len(sig),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labs", default=",".join(DEFAULT_LABS))
    ap.add_argument("--out", default="logs/rl50.json")
    ap.add_argument("--check-determinism", action="store_true")
    args = ap.parse_args()

    labs = [x.strip() for x in args.labs.split(",") if x.strip()]
    rows = load_rows(labs)
    out, lines, problems, note = analyse(rows)

    print("=" * 92)
    print("RL 选算子 n=50 复验   labs=%s   runs=%d" % (labs, len(rows)))
    print("=" * 92)
    for ln in lines:
        print("  " + ln)
    print()

    if args.check_determinism:
        tot, bad = check_determinism(labs)
        print("确定性核对：%d 个重叠 (臂,seed) 与新数据比对，不一致 %d 个" % (tot, len(bad)))
        for b in bad[:10]:
            print("   !! " + b)
        print()

    if problems:
        print("[失败] %d 条问题：" % len(problems))
        for q in problems:
            print("   - " + q)
        return 1

    payload = {"rows": out, "note": note, "summary": summarise(out),
               "meta": {"labs": labs, "n_seeds": sorted({r["n"] for r in out}),
                        "caliber": "instance-boundary (final_hv, 落盘值直读)"}}
    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print("[OK] -> %s  (%d 行)" % (args.out, len(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
