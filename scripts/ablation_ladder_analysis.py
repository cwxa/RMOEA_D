# -*- coding: utf-8 -*-
"""
论文 6 级消融阶梯的统计分析（Li et al., ESWA 203 (2022) 117380, §5.4）。

输入：``scripts/ablation_ladder.py`` 产出的 JSON（list，每条一个 run）。

    python scripts/ablation_ladder_analysis.py --lab_json logs/ablation_ladder.json

它回答四个问题，对应论文 §5.4 的四种证据强度：

  [1] 阶梯单调性 —— 逐级 HV 是否递增（论文声称 "each part improves the result
      against the last one"）。给出每级的 mean±std、Friedman 排名、与 D1 的配对差。
  [2] 相邻两级配对检验 —— **每一级只差一个组件**，因此这是组件效应的干净隔离。
      给出 ΔHV、胜出 run 数、Wilcoxon p、Cohen's d、相对增幅。
  [3] 整梯 Friedman —— 复现论文 Table 4 的排名（论文 p=0.00015 over 23 instances）。
  [4] 2×2 因子分解（论文之外的补充）—— 论文原始阶梯把「MIX3 与 Elite 的交互」
      和「Q-PAS 与 Elite 的交互」混在逐级差里；这里用 B/A 两条固定 T 对照臂把
      Q-PAS 的主效应从 Elite archive 存在与否中分离出来。

HV 一律**参考集归一化重算**：所有臂、所有 run 的前沿并集作为共用归一化盒，
ref=(1.02, 1.02) —— 与 ``scripts/t_leverage_analysis.py`` 口径一致
（不要直接读 JSON 里的 final_hv，那是单 run 的实例边界口径，跨臂差异被压扁）。
跨实例时先算**实例内相对增量**再平均，避免不同量级实例主导结论。

退出码：0 正常；1 数据为空。
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
from rmoea_d.utils.metrics import compute_hv, estimate_hv_bounds  # noqa: E402

REF = (1.02, 1.02)

# 论文 §5.4 的六级阶梯顺序（报告与图的横轴都按此排列）
PAPER_ORDER = ["D1", "D2", "D3", "D4", "D5", "RMOEAD"]
# 论文 Table 4 的 Friedman 平均排名（原文数值，用于并置比较）
PAPER_TABLE4 = {"D1": 4.6087, "D2": 4.3913, "D3": 3.6087,
                "D4": 3.0870, "D5": 2.9130, "RMOEAD": 2.3913}

# 论文 Table 5 的逐实例 HV（原文数值，23 个实例 x 6 个变体，列序同 PAPER_ORDER）。
# 用途：论文 §5.4 断言 "each part improves the result against the last one"，
# 这是**逐实例**的断言；把原文数据逐实例查一遍单调性，才知道这句断言是否真成立。
PAPER_TABLE5 = {
    "D1":    [0.096358, 0.097446, 0.099396, 0.098672, 0.099119, 0.099671],
    "D2":    [0.101298, 0.101899, 0.103487, 0.103056, 0.103296, 0.103148],
    "D3":    [0.064942, 0.067186, 0.066809, 0.067699, 0.067898, 0.067370],
    "D4":    [0.057389, 0.058733, 0.058901, 0.059791, 0.058731, 0.058470],
    "D5":    [0.048071, 0.050738, 0.052962, 0.053041, 0.052376, 0.052245],
    "R1":    [0.050917, 0.050711, 0.050713, 0.050458, 0.050879, 0.050919],
    "R2":    [0.031251, 0.032267, 0.033204, 0.031718, 0.031889, 0.031856],
    "R3":    [0.034702, 0.035149, 0.035013, 0.036010, 0.036115, 0.035549],
    "R4":    [0.039239, 0.039681, 0.041548, 0.041979, 0.041942, 0.042023],
    "R5":    [0.042622, 0.044959, 0.045299, 0.045912, 0.046284, 0.046105],
    "R6":    [0.045522, 0.050659, 0.051547, 0.052755, 0.051897, 0.051946],
    "R7":    [0.041646, 0.054828, 0.056424, 0.057074, 0.057371, 0.057465],
    "R8":    [0.043417, 0.069881, 0.073654, 0.073011, 0.073377, 0.074284],
    "FMk01": [0.058504, 0.055246, 0.056479, 0.057164, 0.056871, 0.057207],
    "FMk02": [0.042559, 0.040818, 0.042238, 0.043146, 0.042341, 0.042327],
    "FMk03": [0.063360, 0.063915, 0.064411, 0.063918, 0.064357, 0.064147],
    "FMk04": [0.094665, 0.095224, 0.094696, 0.095569, 0.095808, 0.096240],
    "FMk05": [0.048044, 0.048418, 0.048343, 0.048325, 0.048403, 0.048468],
    "FMk06": [0.045506, 0.043584, 0.044403, 0.045113, 0.044132, 0.044598],
    "FMk07": [0.071437, 0.070497, 0.070504, 0.070188, 0.070395, 0.070645],
    "FMk08": [0.022032, 0.021521, 0.021134, 0.021609, 0.021323, 0.021721],
    "FMk09": [0.042823, 0.043149, 0.042166, 0.042640, 0.042568, 0.042508],
    "FMk10": [0.060198, 0.062149, 0.062444, 0.062496, 0.062798, 0.063283],
}

# 论文之外的分离臂（见 ablation_ladder.py 的 LADDER）
EXTRA_LABEL = {
    "D5_fixedT": "D5  w/o Q-PAS  (Elite archive only)",
    "D4_fixedT": "D4  w/o Q-PAS",
}
# 阶梯相邻两级 -> 被隔离的组件名（与 ablation_ladder.STEPS 对齐）
STEP_NOTE = {
    ("D1", "D2"): "MIX3 initialization",
    ("D2", "D3"): "random-selection VNS",
    ("D3", "D4"): "Q-PAS",
    ("D4", "D5"): "Elite archive",
    ("D5", "RMOEAD"): "RVNS (RL operator selection)",
}


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def wilcoxon(a, b):
    """安全配对 Wilcoxon：全等或样本过少时返回 nan（而不是抛异常）。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    if n < 3 or np.allclose(a[:n], b[:n]):
        return float("nan")
    try:
        return float(stats.wilcoxon(a[:n], b[:n])[1])
    except Exception:                                          # noqa: BLE001
        return float("nan")


def cohens_d(a, b):
    """配对设计下的 Cohen's d（用差值的均值/标准差）。"""
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) < 2 or d.std(ddof=1) < 1e-12:
        return 0.0
    return float(d.mean() / d.std(ddof=1))


def sign_test_p(wins, n):
    """精确符号检验（双尾）。d==0 的样本按惯例剔除。"""
    if n == 0:
        return float("nan")
    k = min(wins, n - wins)
    # P(X <= k) * 2，X ~ Binomial(n, 0.5)
    p = 2.0 * stats.binom.cdf(k, n, 0.5)
    return float(min(1.0, p))


def _rank_mat(mat):
    """mat: (n_algos, n_obs)，返回每个算法的平均排名（1 = 最好）。"""
    return np.mean([stats.rankdata(-mat[:, i]) for i in range(mat.shape[1])], axis=0)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lab_json", required=True,
                    help="ablation_ladder.py 的输出，如 logs/ablation_ladder.json")
    ap.add_argument("--out_json", default=None,
                    help="把结构化结果另存一份 JSON（默认 <lab_json>.analysis.json）")
    ap.add_argument("--no_rel", action="store_true",
                    help="跨实例聚合时改用原始 ΔHV（默认用实例内相对增量）")
    args = ap.parse_args()

    if not os.path.exists(args.lab_json):
        print(f"[错误] 找不到 {args.lab_json}")
        return 1
    rows = json.load(open(args.lab_json, encoding="utf-8"))
    if not rows:
        print("[错误] 结果为空:", args.lab_json)
        return 1

    # ── 归一化：参考集口径，逐实例一套盒（跨实例不可共用盒） ──
    instances = sorted({r["instance"] for r in rows})
    hv = collections.defaultdict(dict)          # (inst, label) -> {seed: hv}
    fronts_of = collections.defaultdict(list)   # inst -> [front, ...]
    for r in rows:
        if r.get("final_pf"):
            fronts_of[r["instance"]].append(np.asarray(r["final_pf"], float))

    bounds = {}
    for inst in instances:
        fs = fronts_of[inst]
        bounds[inst] = estimate_hv_bounds(fs) if fs else (np.zeros(2), np.ones(2))

    for r in rows:
        lo, hi = bounds[r["instance"]]
        pf = np.asarray(r["final_pf"], float) if r.get("final_pf") else np.empty((0, 2))
        h = compute_hv(pf, ref_point=REF, norm_bounds=(lo, hi)) if len(pf) else 0.0
        hv[(r["instance"], r["label"])][r["seed"]] = float(h)

    labels = sorted({r["label"] for r in rows})
    ladder = [l for l in PAPER_ORDER if l in labels]
    extra = [l for l in labels if l not in ladder]

    # 逐实例、**跨臂取交集**的 seed 集：部分完成的数据集（某个实例还差若干 runs）
    # 里，各臂可用的 seed 不同；若直接用实例的 seed 并集取均值，缺 seed 的臂会 KeyError。
    # 配对检验本来就用交集口径，这里统一，保证 [1] 的均值与 [2] 的配对是同一批 run。
    seeds_by_inst = {inst: sorted(set.intersection(*[
        {r["seed"] for r in rows if r["instance"] == inst and r["label"] == lbl}
        for lbl in labels]) or set())
        for inst in instances}
    # 交集为空的实例无法做跨臂比较（某些臂在这个实例上一条都没有）——直接剔除，
    # 否则下游会算出 nan。这是"实例还没跑完"的正常情形，提示即可，不是错误。
    empty_inst = [i for i in instances if not seeds_by_inst[i]]
    if empty_inst:
        instances = [i for i in instances if i not in empty_inst]
        for i in empty_inst:
            seeds_by_inst.pop(i, None)
    # 两个不同的量，别混：
    #   n_total_runs = 结果文件里的 run 条数（臂 x 实例 x seed）
    #   n_obs        = 参与跨臂配对的观测单元数（实例 x 交集内的 seed）—— 配对检验的 n
    n_total_runs = len(rows)
    n_obs = sum(len(s) for s in seeds_by_inst.values())
    # 有臂在实例上缺 seed 时提示（不静默丢数据）
    dropped = {inst: sorted({r["seed"] for r in rows if r["instance"] == inst}
                            - set(seeds_by_inst[inst])) for inst in instances}
    dropped = {k: v for k, v in dropped.items() if v}

    print("=" * 100)
    print(f"论文 6 级消融阶梯分析  |  实例 {instances}  |  臂 {len(labels)}  |  "
          f"runs {n_total_runs}（配对观测 {n_obs}）")
    print(f"参考集归一化（逐实例，所有臂所有 run 的前沿并集），ref = {REF}")
    for inst in instances:
        lo, hi = bounds[inst]
        print(f"    {inst:<6} lo={np.round(lo, 1)}  hi={np.round(hi, 1)}")
    if empty_inst:
        print(f"⚠ 跳过尚未跑完的实例（无任何 seed 被全部臂覆盖）：{empty_inst}")
        if not instances:
            print("[错误] 没有任何实例可用于跨臂比较。")
            return 1
    if dropped:
        print("⚠ 以下实例存在『并非所有臂都跑满同一批 seed』的情况，"
              "已按跨臂交集的 seed 计算（被剔除的 seed 见括号）：")
        for inst, sd in sorted(dropped.items()):
            print(f"    {inst:<6} 剔除 {len(sd)} 个 seed: {sd[:8]}"
                  f"{' ...' if len(sd) > 8 else ''}")
    print("=" * 100)

    # 每臂每实例的 mean HV（矩阵形式，供 Friedman 用）
    M = {}
    for lbl in labels:
        M[lbl] = np.array([[np.mean([hv[(i, lbl)][s] for s in seeds_by_inst[i]])
                            for i in instances]]).ravel()

    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("[1] 阶梯逐级 HV（论文 §5.4 Table 4/5 的复现位）")
    print("=" * 100)
    print(f"{'arm':<12}{'variant':<36}{'HV mean':<12}{'std(inst)':<11}"
          f"{'Friedman rank':<15}{'paper rank':<12}")
    print("-" * 100)
    rank_all = _rank_mat(np.array([M[l] for l in labels])) if len(labels) > 1 else None
    for k, lbl in enumerate(labels):
        v = M[lbl]
        name = EXTRA_LABEL.get(lbl, "")
        if not name:
            # 论文变体名
            name = {"D1": "RMOEA/D1 (plain MOEA/D)",
                    "D2": "RMOEA/D2 (+MIX3)",
                    "D3": "RMOEA/D3 (+random VNS)",
                    "D4": "RMOEA/D4 (+Q-PAS)",
                    "D5": "RMOEA/D5 (+elite archive)",
                    "RMOEAD": "RMOEA/D (= D5 with RVNS)"}.get(lbl, lbl)
        rk = f"{rank_all[k]:.4f}" if rank_all is not None else "-"
        pr = f"{PAPER_TABLE4[lbl]:.4f}" if lbl in PAPER_TABLE4 else "-"
        # 单实例时 std(ddof=1) 未定义，显示 "-" 而不是 nan
        sd = f"{v.std(ddof=1):.5f}" if len(v) > 1 else "-"
        print(f"{lbl:<12}{name:<36}{v.mean():<12.5f}{sd:<11}"
              f"{rk:<15}{pr:<12}")

    if rank_all is not None and len(labels) > len(ladder):
        print("-" * 100)
        print(f"注：上面 'Friedman rank' 是**{len(labels)} 条臂**一起排的"
              f"（含 {len(labels) - len(ladder)} 条论文之外的分离臂）；"
              f"可与论文 Table 4 直接比的是 [3] 里**论文 {len(ladder)} 级**的排名。")

    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("[2] 相邻两级配对检验（每级只差一个组件 —— 组件效应的干净隔离）")
    print("=" * 100)
    if len(ladder) < 2:
        print("  只有一级臂，跳过相邻级检验。")
    print(f"{'step':<20}{'component':<28}{'dHV(add)':<14}{'rel %':<11}"
          f"{'better-pair':<12}{'p (inst)':<12}{'d':<8}{'sig'}")
    print("-" * 100)

    # 符号约定（全脚本统一，图上同）：
    #   dHV(add) = HV(加了这一级) − HV(上一级)  —— **>0 表示这个组件是正贡献**。
    # 早期版本用 HV(上一级) − HV(加了) 且表头也写 "dHV"，导致负数其实是"变好"，
    # 图表读起来正好相反，故统一翻转。
    steps_report = []
    for a_lbl, b_lbl in zip(ladder[:-1], ladder[1:]):
        # 臂组合可以是论文六级的子集（`--arms D1,D2,D3`），相邻二级未必在
        # STEP_NOTE 里 —— 退回用 "A->B" 命名，不要把组件名留空。
        comp = STEP_NOTE.get((a_lbl, b_lbl), f"{a_lbl}->{b_lbl}")
        va, vb = M[a_lbl], M[b_lbl]          # 每元素 = 一个实例的均值
        d = vb - va                          # >0 = 新加的这一级更好
        # 相对增幅用**比值之比**（mean(d) / mean(base)），保证与 dHV 永远同号；
        # 若用 mean(d/base)（各实例相对增幅的平均），效应接近 0 时可能与 dHV 反号。
        rel = float(d.mean() / va.mean() * 100.0) if np.all(va > 0) else float("nan")
        p = wilcoxon(vb, va)
        dd = cohens_d(vb, va)
        favor = b_lbl if d.mean() > 0 else a_lbl
        wins = int((d > 0).sum()) if d.mean() >= 0 else int((d < 0).sum())
        n_i = len(d)
        steps_report.append(dict(step=f"{a_lbl}->{b_lbl}", component=comp,
                                 dHV_add=float(d.mean()), rel_pct=rel,
                                 wins=wins, n_inst=n_i, p=p, cohens_d=dd,
                                 favors=favor))
        print(f"{a_lbl + ' -> ' + b_lbl:<20}{comp:<28}{d.mean():<+14.5f}"
              f"{rel:<+11.2f}{f'{favor} {wins}/{n_i}':<12}{p:<12.4g}"
              f"{dd:<+8.2f}{stars(p)}")

    print("-" * 100)
    print("注：dHV(add)>0 表示『加上这一级的组件』是正贡献（rel % 同号）。")
    print("    better-pair 列给出方向与赢家数。")
    print("    n_inst = 实例数；实例级配对检验（每个实例内先对 30 run 取均值再配对）")
    print("    比 run 级检验保守；下表给出 run 级口径作为对照。")

    # ── run 级（逐 seed 配对，跨实例合并）——同为 dHV(add) = HV(加) − HV(上一级) ──
    def _paired_runs(a_lbl, b_lbl):
        """返回 (差值数组, 基线 HV 数组)，差值 = HV(b) − HV(a)，>0 表示加组件更好。

        相对增幅由调用方用 `mean(diffs)/mean(base)` 计算，保证与 dHV 同号。
        """
        diffs, base = [], []
        for inst in instances:
            sa = hv[(inst, a_lbl)]
            sb = hv[(inst, b_lbl)]
            common = sorted(set(sa) & set(sb))
            if not common:
                continue
            xa = np.array([sa[s] for s in common])
            xb = np.array([sb[s] for s in common])
            diffs.extend(xb - xa)
            base.extend(xa)
        return np.array(diffs), np.array(base)

    print("-" * 100)
    print(f"{'step':<20}{'component':<28}{'dHV(add)':<14}{'rel %':<11}"
          f"{'wins(runs)':<12}{'p (run)':<12}{'d':<8}{'sig'}")
    print("-" * 100)
    run_report = []
    for a_lbl, b_lbl in zip(ladder[:-1], ladder[1:]):
        # 臂组合可以是论文六级的子集（`--arms D1,D2,D3`），相邻二级未必在
        # STEP_NOTE 里 —— 退回用 "A->B" 命名，不要把组件名留空。
        comp = STEP_NOTE.get((a_lbl, b_lbl), f"{a_lbl}->{b_lbl}")
        d, base = _paired_runs(a_lbl, b_lbl)
        if len(d) < 3:
            continue
        # d 是差值数组：Wilcoxon 单样本 vs 0（双侧）
        try:
            p = float(stats.wilcoxon(d)[1])
        except Exception:                                      # noqa: BLE001
            p = float("nan")
        dd = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 1e-12 else 0.0
        rel = float(d.mean() / base.mean() * 100.0) if base.mean() > 0 \
            else float("nan")
        favor = b_lbl if d.mean() > 0 else a_lbl
        wins = int((d > 0).sum()) if d.mean() >= 0 else int((d < 0).sum())
        run_report.append(dict(step=f"{a_lbl}->{b_lbl}", component=comp,
                               dHV_add=float(d.mean()), rel_mean=rel,
                               wins=wins, n=len(d), p=p, cohens_d=dd,
                               favors=favor))
        print(f"{a_lbl + ' -> ' + b_lbl:<20}{comp:<28}{d.mean():<+14.5f}"
              f"{rel:<+11.2f}{f'{favor} {wins}/{len(d)}':<12}"
              f"{p:<12.4g}{dd:<+8.2f}{stars(p)}")

    # ══════════════════════════════════════════════════════════════
    if len(ladder) > 2:
        print("\n" + "=" * 100)
        print("[3] 整梯 Friedman（论文 Table 4 的复现；论文报 p=0.00015 over 23 instances）")
        print("=" * 100)
        mat = np.array([M[l] for l in ladder])
        rk = _rank_mat(mat)
        if len(instances) > 1 and len(ladder) > 2:
            st, p = stats.friedmanchisquare(*mat)
            print(f"Friedman over {ladder}: chi2={st:.4f}  p={p:.6g} {stars(p)}")
        else:
            print(f"⚠ 只有 {len(instances)} 个实例 —— Friedman 需要 >=2 个实例（此处仅"
                  "报排名，不做检验）")
            p = float("nan")
        print(f"{'arm':<12}{'rank (here)':<15}{'rank (paper)':<15}{'Δ':<9}")
        for k, lbl in enumerate(ladder):
            pr = PAPER_TABLE4.get(lbl)
            d = (rk[k] - pr) if pr is not None else None
            print(f"{lbl:<12}{rk[k]:<15.4f}"
                  f"{(f'{pr:.4f}' if pr is not None else '-'):<15}"
                  f"{(f'{d:+.4f}' if d is not None else '-'):<9}")
        # 单调性检验：排名是否逐级下降
        mono = all(rk[i] >= rk[i + 1] for i in range(len(rk) - 1))
        print(f"\n排名单调不增（每级不劣于上一级）= {mono}")
        if "D1" in ladder and "RMOEAD" in ladder:
            print(f"D1 -> RMOEA/D 总排名增益 = {rk[0] - rk[-1]:+.4f}（论文 "
                  f"{PAPER_TABLE4['D1'] - PAPER_TABLE4['RMOEAD']:+.4f}）")
        neg = [f"{ladder[i]}->{ladder[i+1]}"
               for i in range(len(rk) - 1) if rk[i] < rk[i + 1]]
        if neg:
            print(f"⚠ 排名出现倒退的级: {neg}")
        else:
            print("✓ 无倒退级")

    # ══════════════════════════════════════════════════════════════
    # [4] 2×2 因子分解：Q-PAS ∈ {on,off} × Elite archive ∈ {on,off}
    #     论文的阶梯把 Q-PAS 与 Elite 的交错混在逐级差里，这里做干净分离。
    # ══════════════════════════════════════════════════════════════
    factor_report = {}
    if all(k in labels for k in ("D5", "D5_fixedT", "D4", "D4_fixedT")):
        print("\n" + "=" * 100)
        print("[4] 2×2 因子分解：Q-PAS × Elite archive（论文之外的补充隔离）")
        print("=" * 100)
        A = M["D5"]         # elite=1 qpas=1
        B = M["D5_fixedT"]  # elite=1 qpas=0
        C = M["D4"]         # elite=0 qpas=1
        D = M["D4_fixedT"]  # elite=0 qpas=0
        # 主效应/交互的**点估计**用实例级均值
        # 显著性用逐 (实例, seed) 配对：先按每个 seed 算出效应量，再 Wilcoxon vs 0。
        #    Q-PAS 主效应  = [(A-B) + (C-D)] / 2
        #    Elite 主效应  = [(A-C) + (B-D)] / 2
        #    交互          = (A-B) - (C-D)
        # 这三条都要用**同一 seed** 的四条臂 —— 它们本来就是同 seed 配对跑的。
        def _paired_effect(kind):
            out = []
            for inst in instances:
                d5 = hv[(inst, "D5")]
                d5f = hv[(inst, "D5_fixedT")]
                d4 = hv[(inst, "D4")]
                d4f = hv[(inst, "D4_fixedT")]
                for s in sorted(set(d5) & set(d5f) & set(d4) & set(d4f)):
                    ab = d5[s] - d5f[s]
                    cd = d4[s] - d4f[s]
                    ac = d5[s] - d4[s]
                    bd = d5f[s] - d4f[s]
                    if kind == "qpas":
                        out.append((ab + cd) / 2.0)
                    elif kind == "elite":
                        out.append((ac + bd) / 2.0)
                    else:
                        out.append(ab - cd)
            return np.array(out)

        for nm, x, kind in [("Q-PAS 主效应", ((A - B) + (C - D)) / 2, "qpas"),
                            ("Elite 主效应", ((A - C) + (B - D)) / 2, "elite"),
                            ("交互 Q-PAS×Elite", (A - B) - (C - D), "inter")]:
            pool = _paired_effect(kind)
            pv = float("nan")
            if len(pool) >= 3 and pool.std(ddof=1) > 1e-15:
                try:
                    pv = float(stats.wilcoxon(pool)[1])
                except Exception:                              # noqa: BLE001
                    pv = float("nan")
            factor_report[nm] = dict(effect=float(x.mean()), p=pv, n=len(pool),
                                     wins=int((pool > 0).sum()) if len(pool) else 0)
            print(f"  {nm:<20} 效应={x.mean():+.5f}  "
                  f"胜={int((pool > 0).sum()) if len(pool) else 0:>3}/{len(pool):<3}  "
                  f"p={pv:<9.4g} {stars(pv)}")
        print("\n  单元格均值：")
        for nm, v in [("elite=1, qpas=1 (D5)", A), ("elite=1, qpas=0 (D5_fixedT)", B),
                      ("elite=0, qpas=1 (D4)", C), ("elite=0, qpas=0 (D4_fixedT)", D)]:
            print(f"    {nm:<30} {v.mean():.5f}")

    # ══════════════════════════════════════════════════════════════
    # [5] Q-PAS 的 T 使用分布
    #     ablation_ladder.py 写出的 hist_T 是「每代的 T 列表」，这里聚合。
    # ══════════════════════════════════════════════════════════════
    tdist = collections.defaultdict(collections.Counter)
    for r in rows:
        ht = r.get("hist_T")
        if not ht:
            continue
        if isinstance(ht, dict):                 # 兼容 t_leverage_sweep 的 Counter 口径
            for k, c in ht.items():
                tdist[r["label"]][int(k)] += int(c)
        else:                                    # list：统计各 T 出现次数
            for t in ht:
                if t is not None:
                    tdist[r["label"]][int(t)] += 1
    if tdist:
        print("\n" + "=" * 100)
        print("[5] Q-PAS 选到的 T 分布（占该臂全部代数比例）")
        print("=" * 100)
        for lbl in [l for l in labels if tdist[l]]:
            tot = sum(tdist[lbl].values())
            dist = "  ".join(f"T{k}:{v / tot * 100:5.1f}%"
                             for k, v in sorted(tdist[lbl].items()))
            print(f"  {lbl:<12} {dist}")

    # ══════════════════════════════════════════════════════════════
    # [6] 论文 §5.4 那句断言的逐实例查证
    #     原文："each part improves the result against the last one."
    #     这是**逐实例**的断言。把论文 Table 5 的原文数值与本次复现都按
    #     实例查一遍"是否逐级严格递增"，就能看出这句话的适用范围。
    # HV 越大越好 -> 单调 = 相邻每级都严格更大。
    # ══════════════════════════════════════════════════════════════
    mono_report = {}
    if "D1" in PAPER_TABLE5:
        print("\n" + "=" * 100)
        print("[6] 逐实例单调性：论文 §5.4 断言 vs 论文自己的 Table 5")
        print("=" * 100)
        names = ["RMOEA/D1", "RMOEA/D2", "RMOEA/D3",
                 "RMOEA/D4", "RMOEA/D5", "RMOEA/D"]

        def breaks(v):
            return [i for i in range(len(v) - 1) if not (v[i + 1] > v[i])]

        p_mono = 0
        print(f"{'paper instance':<16}{'verdict':<12}{'broken step(s)'}")
        print("-" * 100)
        for inst, v in PAPER_TABLE5.items():
            bad = breaks(v)
            if not bad:
                p_mono += 1
                print(f"{inst:<16}{'MONOTONE':<12}-")
            else:
                print(f"{inst:<16}{'break':<12}"
                      + ", ".join(f"{names[i]}->{names[i+1]}" for i in bad))
        print("-" * 100)
        print(f"论文 Table 5：{p_mono}/{len(PAPER_TABLE5)} 个实例逐级单调递增")
        mono_report["paper_monotone"] = f"{p_mono}/{len(PAPER_TABLE5)}"

        # 本次复现（同一口径：逐实例、六级阶梯、HV 越大越好）
        ours_mono, ours_detail = 0, {}
        for inst in instances:
            v = [M[l][instances.index(inst)] for l in ladder]
            bad = breaks(v)
            ours_detail[inst] = [f"{ladder[i]}->{ladder[i+1]}" for i in bad]
            if not bad:
                ours_mono += 1
        print(f"本次复现（{len(instances)} 个实例）："
              f"{ours_mono}/{len(instances)} 个实例逐级单调递增")
        bad_steps = collections.Counter(
            s for v in ours_detail.values() for s in v)
        if bad_steps:
            print("  复现中最常被打断的级："
                  + "  ".join(f"{s}:{c} 次" for s, c in bad_steps.most_common()))
        print("\n  结论：论文那句断言在**逐实例**口径下几乎处处不成立（原文 23 个实例只有 "
              f"{p_mono} 个单调）；\n  它真正的含义是『Friedman 平均排名逐级改善』——"
              "平均排名确实是 D1 4.61 -> RMOEA/D 2.39。")
        mono_report["ours_monotone"] = f"{ours_mono}/{len(instances)}"
        mono_report["ours_broken_steps"] = dict(bad_steps)

    # ── 落盘 ──
    out_json = args.out_json or (args.lab_json + ".analysis.json")
    # 只需论文的六级排名（Friedman 需要 >=2 实例）；臂多于六级时单独再算一次
    rk_ladder = (_rank_mat(np.array([M[l] for l in ladder]))
                 if len(ladder) > 1 else None)
    payload = {
        "lab_json": args.lab_json,
        "instances": instances,
        "n_total_runs": n_total_runs,
        "n_paired_obs": n_obs,
        "hv_definition": "reference-set normalization per instance, ref=(1.02,1.02)",
        "mean_hv": {lbl: dict(zip(instances, M[lbl].round(8).tolist())) for lbl in labels},
        "friedman_rank_all_arms": (dict(zip(labels, rank_all.round(6).tolist()))
                                   if rank_all is not None else None),
        "friedman_rank_ladder": (dict(zip(ladder, rk_ladder.round(6).tolist()))
                                 if rk_ladder is not None else None),
        "paper_table4_rank": PAPER_TABLE4,
        "paper_table5_monotonicity": mono_report,
        "steps_instance_level": steps_report,
        "steps_run_level": run_report,
        "factor_2x2_qpas_elite": factor_report,
        "t_usage": {lbl: {str(k): int(v) for k, v in tdist[lbl].items()}
                    for lbl in labels if tdist[lbl]},
    }
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\n已写入 {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
