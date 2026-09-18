#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""初始化变体扫描分析（Mk10 开发集 / Mk07·Mk09 留出集共用）。

背景
----
消融阶梯里 **MIX3 是最大的单一组件**（实例边界口径 **+3.406%**，10 实例平均、256/300、
p=2.7e-40；旧值 +14.46% 是**盒口径**，放大 4.9×，见 docs/new-arch-report.md §1），
却**从未做过变体扫描**。
而且它三条分支 random / LS / GW **全部把 OS 随机打乱**，只在机器选择（MA 维度）
做文章 —— 工序顺序这一维在初始化阶段完全没被利用。

本脚本回答三个问题：

  Q1 接线一致性：`I_mix3` 与既有 lab 里的 `RVNSonly`（两者配置完全相同）
                 是否逐位相同？不同就说明变体接线有 bug，后面全不用看。
  Q2 初始化总杠杆：各变体相对纯随机（`I_rand`，= 论文 RMOEA/D1）差多少？
  Q3 净增益：有没有变体显著优于论文口径的 MIX3？

口径（三条硬约束）
------------------
1. **归一化盒逐实例独立**。HV 的归一化盒是"参与比较的前沿并集"的包围盒，
   把不同实例（Mk07 量级 ~150/700，Mk10 量级 ~300/2000）的前沿并成**同一个盒**，
   会让 Mk07 的点被压进盒的左下角、HV 冲到 1.0，而 Mk10 只剩 ~0.1 ——
   这是纯粹的**量纲假象**，会读出"Mk07 上所有变体都近乎完美"这种胡说。
   因此本脚本按 `row["instance"]` 分组，**每组一个盒**。
2. **盒只由「参与比较的臂」构造**（`CONTEXT_ARMS`）。约定原文是
   "参与比较的所有臂、所有 run 的前沿并集"，所以盒**不能**把 lab 文件里
   顺带存在的无关臂也并进去。实测：`_mk10_lab.json` 里有 35 个 Q-PAS / T 臂，
   其中 `QPAS2_opt_hv` 的前沿把 Mk10 的 y 上界从 **2234.0 撑到 2297.25**，
   于是同一份变体数据被读出完全不同的 HV / rel% / dz
   （`I_rand` 杠杆 −25.69% → −22.31%）。凡"盒随臂集变化"，
   绝对 HV 与 rel% **都不能跨批次比**，所以宁可把臂集钉死。
3. **结论只用相对差**（dHV / 胜负 / p / dz / rel%），与盒无关，可跨实例比对。

修正的三个缺陷
--------------
* 缺陷 A（2026-09-17）：`load_hv` 把所有 lab 摊平成一个 `{(label, seed): hv}` 字典，
  跨实例时**同 (label, seed) 互相覆盖** → 留出集分析输出全是垃圾。
* 缺陷 B（2026-09-17）：实例标签用 `basename.replace("_init.json","")` 得到 `_mk07`，
  而查表键写的是 `mk07`（无下划线）→ 方向一致性表**永远为空**（静默）。
* 缺陷 C（2026-09-18）：盒按"该实例在 `--labs` 里的**全部行**"构造，
  于是被无关臂撑大（见口径 2）。修法：盒只用 `CONTEXT_ARMS` 的行；
  被排除的臂会在表头**显式列出**（不再静默）。

用法
----
    # 开发集（Mk10；两个 lab 同实例，共用一个盒是合法的）
    python scripts/init_variant_analysis.py --labs logs/_mk10_init.json,logs/_mk10_lab.json
    # 留出集
    python scripts/init_variant_analysis.py --holdout \
        --labs logs/_mk07_init.json,logs/_mk09_init.json
"""
import argparse
import collections
import json
import os
import re
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from rmoea_d.utils.metrics import compute_hv, estimate_hv_bounds  # noqa: E402

REF = (1.02, 1.02)

# 变体臂（与 scripts/t_leverage_sweep.py 的 ARM_DEF 保持一致）
VARIANTS = ["I_rand", "I_no_r", "I_half_r", "I_mix3", "I_gw_spt",
            "I_spt", "I_mwr"]
LABEL = {
    "I_rand":   "纯随机 (= 论文 D1)",
    "I_no_r":   "1/2 LS + 1/2 GW（无随机）",
    "I_half_r": "1/2 随机 + 1/4 LS + 1/4 GW",
    "I_mix3":   "MIX3 论文口径 (1/3,1/3,1/3)",
    "I_gw_spt": "1/4 随机 +1/4 LS +1/4 GW +1/4 OS-SPT",
    "I_spt":    "1/3 随机 + 1/3 LS + 1/3 OS-SPT",
    "I_mwr":    "1/3 随机 + 1/3 LS + 1/3 OS-MWR",
}
PAPER = "I_mix3"
RANDOM_BASE = "I_rand"

# 「参与比较的臂」——归一化盒**只**由这些臂的前沿并集决定（口径 2）。
# 加 RVNSonly 是零成本的：它与 I_mix3 逐位相同（Q1 就是验这个），
# 前沿集合不变 → 盒不变，但 Q1 的对照也就落在同一个盒里了。
CONTEXT_ARMS = VARIANTS + ["RVNSonly"]

L = []


def W(s=""):
    L.append(str(s))
    print(s, flush=True)


def stars(p):
    if p != p:
        return "n.s."
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def split_context(rows, keep=None):
    """把行分成「参与比较」与「无关臂」两部分（口径 2）。

    无关臂**不进归一化盒**，但也不丢——调用方负责把它们显式报出来，
    避免"少算了什么"这件事再次静默。
    """
    keep = set(CONTEXT_ARMS if keep is None else keep)
    kept, dropped = [], collections.Counter()
    for r in rows:
        if r.get("label") in keep:
            kept.append(r)
        else:
            dropped[r.get("label")] += 1
    return kept, dropped


def hv_table(rows):
    """单个实例内的行 -> ({label: {seed: hv}}, box)。

    盒由**传入行**的前沿并集决定 —— 所以调用方必须先 `split_context()`，
    否则无关臂会撑大盒（口径 2）。
    """
    fronts = [np.asarray(r["final_pf"], float) for r in rows if r.get("final_pf")]
    lo, hi = estimate_hv_bounds(fronts)
    hv = collections.defaultdict(dict)
    for r in rows:
        if not r.get("final_pf"):
            continue
        hv[r["label"]][r["seed"]] = compute_hv(
            np.asarray(r["final_pf"], float), ref_point=REF, norm_bounds=(lo, hi))
    return hv, (lo, hi)


def _infer_instance(path, rows):
    """从文件名兜底推断实例名（早期 lab 文件的行里没有 instance 字段）。

    `logs/_mk10_lab.json` -> `Mk10`。只要文件里成批的行缺字段就用它；
    推断不出又缺字段则报错退出（不静默塞一个假实例名）。
    """
    m = re.search(r"(mk\s*\d+)", os.path.basename(path), re.IGNORECASE)
    if not m:
        return None
    return "Mk" + re.sub(r"\D", "", m.group(1)).zfill(2)


def load_by_instance(labs):
    """按实例分组载入。**不跨实例合并**（见模块 docstring 口径 1）。"""
    rows = []
    for p in labs:
        with open(p, encoding="utf-8") as f:
            sub = json.load(f)
        if not sub:
            continue
        n_missing = sum(1 for r in sub if not r.get("instance"))
        if n_missing:
            guess = _infer_instance(p, sub)
            if guess is None:
                raise SystemExit(
                    "%s 有 %d/%d 行缺少 instance 字段，且文件名推断不出实例；"
                    "请用文件名含 MkNN 或补字段。" % (p, n_missing, len(sub)))
            print("[i] %s: %d/%d 行缺 instance，按文件名推断为 %s"
                  % (p, n_missing, len(sub), guess), flush=True)
            for r in sub:
                r.setdefault("instance", guess)
                if not r.get("instance"):
                    r["instance"] = guess
        for r in sub:
            if not r.get("label"):
                raise SystemExit("%s 的行缺少 label 字段" % p)
            rows.append(r)
    per = collections.OrderedDict()
    for r in rows:
        per.setdefault(r["instance"], []).append(r)
    return per, rows


def paired(hv, a, b, min_n=4):
    """按 seed 交集做配对 Wilcoxon，返回差值统计。"""
    ss = sorted(set(hv[a]) & set(hv[b]))
    if len(ss) < min_n:
        return None
    va = np.array([hv[a][s] for s in ss])
    vb = np.array([hv[b][s] for s in ss])
    d = va - vb
    if np.all(d == 0):
        # 逐位相同：Wilcoxon 无定义，显式标注而不是丢 nan 出去
        return dict(n=len(ss), d=0.0, mean_a=float(va.mean()),
                    mean_b=float(vb.mean()), wins=0, p=float("nan"),
                    dz=0.0, rel=0.0, identical=True)
    p = float(stats.wilcoxon(va, vb)[1])
    sd = d.std(ddof=1)
    return dict(n=len(ss), d=float(d.mean()), mean_a=float(va.mean()),
                mean_b=float(vb.mean()),
                wins=int((d > 0).sum()),
                p=p, dz=float(d.mean() / sd) if sd > 0 else float("nan"),
                rel=float(d.mean() / vb.mean() * 100.0), identical=False)


def report_instance(tag, rows, show_rel_to_random=True):
    hv, box = hv_table(rows)
    present = [v for v in VARIANTS if v in hv]
    missing = [v for v in VARIANTS if v not in hv]

    W("=" * 78)
    W("[实例 %s]  run 数 = %d   臂数 = %d" % (tag, len(rows), len(hv)))
    W("  归一化盒(本实例独立) lo=%s hi=%s"
      % (np.round(box[0], 4), np.round(box[1], 4)))
    if missing:
        W("  [!] 本实例缺少的变体臂: %s" % missing)
    W("=" * 78)

    # ── Q1 接线一致性 ──
    W("-" * 78)
    W("[Q1] 接线一致性：I_mix3 vs RVNSonly（两者配置完全相同）")
    W("-" * 78)
    if "RVNSonly" in hv and "I_mix3" in hv:
        r = paired(hv, "I_mix3", "RVNSonly")
        ss = sorted(set(hv["I_mix3"]) & set(hv["RVNSonly"]))
        ident = all(hv["I_mix3"][s] == hv["RVNSonly"][s] for s in ss)
        W("  共同 seed 数 = %d" % len(ss))
        W("  dHV = %+.10f   逐位相同: %s" % (r["d"], "是" if ident else "否"))
        W("  -> %s" % ("通过：init_variant 接线未改变论文口径行为"
                        if ident else "**失败**：变体重构改动了默认行为！"))
    else:
        W("  (本实例 lab 里没有 RVNSonly，跳过)")
    W()

    # ── Q2 初始化总杠杆（vs 纯随机）──
    if show_rel_to_random and RANDOM_BASE in hv:
        W("-" * 78)
        W("[Q2] 初始化总杠杆（配对基准 = 纯随机 %s）" % RANDOM_BASE)
        W("-" * 78)
        W("%-44s %9s %11s %10s %11s" % ("变体", "mean HV", "vs 随机 d",
                                        "vs 随机 %", "p"))
        W("-" * 78)
        for v in present:
            m = float(np.mean(list(hv[v].values())))
            if v == RANDOM_BASE:
                W("%-44s %9.5f %11s %10s %11s"
                  % (LABEL.get(v, v) + " [" + v + "]", m, "—", "—", "—"))
                continue
            r = paired(hv, v, RANDOM_BASE)
            if r is None:
                W("%-44s %9.5f %11s %10s %11s"
                  % (LABEL.get(v, v) + " [" + v + "]", m, "n<4", "—", "—"))
                continue
            W("%-44s %9.5f %+11.5f %+9.2f%% %8.4g %s"
              % (LABEL.get(v, v) + " [" + v + "]", m, r["d"], r["rel"],
                 r["p"], stars(r["p"])))
        W()

    # ── Q3 净增益（vs 论文口径 MIX3）──
    if PAPER in hv:
        W("-" * 78)
        W("[Q3] 净增益（配对基准 = 论文口径 %s）" % PAPER)
        W("-" * 78)
        W("%-44s %11s %9s %11s %9s %s"
          % ("变体", "dHV", "胜负", "rel%", "p", "dz"))
        W("-" * 78)
        for v in present:
            if v == PAPER:
                continue
            r = paired(hv, v, PAPER)
            if r is None:
                continue
            W("%-44s %+11.5f %9s %+10.2f%% %9.4g %+7.3f %s"
              % (v, r["d"], "%d/%d" % (r["wins"], r["n"]), r["rel"],
                 r["p"], r["dz"], stars(r["p"])))
        W()
    return hv, box, present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labs", required=True, help="逗号分隔的 lab JSON")
    ap.add_argument("--holdout", action="store_true",
                    help="留出集模式：额外给出实例间方向一致性")
    ap.add_argument("--arms", default=None,
                    help="参与比较的臂（逗号分隔）；归一化盒只由这些臂构造。"
                         "默认 = %s" % ",".join(CONTEXT_ARMS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    labs = [p.strip() for p in args.labs.split(",") if p.strip()]
    keep = [a.strip() for a in args.arms.split(",")] if args.arms else list(CONTEXT_ARMS)
    keep = [a for a in keep if a]
    per_all, rows = load_by_instance(labs)

    # 口径 2：盒只由「参与比较的臂」构造。无关臂显式列出，不静默丢弃。
    per = collections.OrderedDict()
    dropped = collections.Counter()
    n_kept = 0
    for inst, rr in per_all.items():
        k, d = split_context(rr, keep)
        per[inst] = k
        n_kept += len(k)
        dropped.update(d)
    tags = list(per.keys())

    W("=" * 78)
    W("初始化变体扫描分析%s" % ("（留出集）" if args.holdout else "（开发集）"))
    W("=" * 78)
    W("lab 文件        : %s" % ", ".join(labs))
    W("读入 run 数     : %d" % len(rows))
    W("参与比较的 run  : %d  （臂: %s）" % (n_kept, ", ".join(keep)))
    if dropped:
        W("已排除的无关臂  : %d 个 / %d runs —— **其前沿不进归一化盒**（口径 2）"
          % (len(dropped), sum(dropped.values())))
        W("                  %s" % ", ".join(
            "%s×%d" % (k, v) for k, v in sorted(dropped.items())[:12]))
        if len(dropped) > 12:
            W("                  ...（共 %d 个臂）" % len(dropped))
    W("实例            : %s" % ", ".join(tags))
    W("★ 归一化盒**逐实例独立**（跨实例合并会把 HV 变成量纲假象，见脚本 docstring）。")
    W("★ 归一化盒只由上述「参与比较的臂」构造 —— 无关臂会撑大盒并污染 rel% / dz。")
    W("★ 全部结论只用相对差（与盒无关）。")
    W()

    if len(tags) > 1 and not args.holdout:
        W("[!] 输入含 %d 个实例但未加 --holdout；各实例将分别独立报告，" % len(tags))
        W("    跨实例方向一致性表请加 --holdout 查看。")
        W()

    if len(tags) == 1:
        # 单实例（开发集）：保留 vs 纯随机 的总杠杆表
        report_instance(tags[0], per[tags[0]], show_rel_to_random=True)
    else:
        for t in tags:
            report_instance(t, per[t], show_rel_to_random=False)

    # ── 跨实例方向一致性（留出集的核心判据）──
    if args.holdout or len(tags) > 1:
        W("=" * 78)
        W("[留出集判据] 各变体 vs %s 的相对差**符号**必须跨实例一致" % PAPER)
        W("=" * 78)
        per_inst = collections.defaultdict(dict)
        for t in tags:
            hv, _ = hv_table(per[t])
            if PAPER not in hv:
                continue
            for v in VARIANTS:
                if v == PAPER or v not in hv:
                    continue
                r = paired(hv, v, PAPER)
                if r:
                    per_inst[v][t] = (r["rel"], r["p"], r["identical"])
        hdr = "%-10s" % "变体" + "".join("%-24s" % t for t in tags) + "  判定"
        W(hdr)
        W("-" * len(hdr))
        for v in VARIANTS:
            if v not in per_inst:
                continue
            cells = []
            signs = []
            for t in tags:
                if t in per_inst[v]:
                    rel, p, ident = per_inst[v][t]
                    cells.append("%+7.2f%% p=%-9.3g" % (rel, p)
                                 if not ident else "逐位相同        ")
                    signs.append(0 if ident else (1 if rel > 0 else -1))
                else:
                    cells.append("%-24s" % "—")
            same = len(set(signs)) <= 1 and len(signs) > 1
            if not signs:
                verdict = "无数据"
            elif all(s == 0 for s in signs):
                verdict = "无差异"
            elif same:
                verdict = "同号 ✓" + (" + 全部显著" if all(
                    per_inst[v][t][1] < 0.05 for t in tags if t in per_inst[v])
                    else "（未全显著）")
            else:
                verdict = "**符号不一致** → 不可复现"
            W("%-10s" % v + "".join("%-24s" % c for c in cells) + "  " + verdict)
        W()
        W("  判据：**所有实例同号**才算成立；同号但未全显著 -> 方向可复现但强度不足；")
        W("        符号不一致 -> 该变体没有可复现的净增益。")
    W()
    W("=" * 78)
    W("结论按预注册纪律：开发集只用于筛选，**留出集同号才算成立**。")
    W("=" * 78)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")
        print("-> written %s" % args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
