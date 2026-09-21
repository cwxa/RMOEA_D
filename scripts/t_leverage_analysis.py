#!/usr/bin/env python3
"""邻域大小 T 杠杆 / Q-PAS 机理实验的分析（参考集归一化 HV + 配对显著性）。

输入：``scripts/t_leverage_sweep.py`` 产出的 JSON。

    python scripts/t_leverage_analysis.py --lab_json logs/_mk10_lab.json

输出：
  [1] 固定 T 的杠杆检验（Friedman + 逐 T 对照 + per-seed oracle）
  [2] Q-PAS 各设定 vs T=10 / vs 最优固定 T / 距离 oracle 的差距
  [3] Full（Q-PAS+RVNS）与各组件的配对比较
  [4] Q-PAS 实际学到的 T 使用分布

HV 一律用**参考集归一化**重算（**盒口径**）：所有臂、所有 run 的前沿并集作为
共用归一化盒，ref=(1.02,1.02) —— 与 ``scripts/ablation_analysis.py`` 口径一致。

⚠ 本脚本输出的是**盒口径**，边界随臂集漂移（缺陷 30），而且它能让结论**反向**
（缺陷 39：Mk10 固定-T 扫参在盒口径下读作"最优 T=50、Friedman p=4.7e-05***"，
换成**实例边界口径**（落盘 ``final_hv``）则是"最优 T=15、p=0.54"）。
**论文与 docs 的效应量一律读 ``final_hv``**；本脚本只作**盒内灵敏度参考**，
引用任何数字都必须同时声明臂集。实例边界口径的对照见 ``scripts/t_caliber_check.py``。
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 便于 import hv_box
sys.path.insert(0, os.path.join(ROOT, "src"))
from rmoea_d.utils.metrics import compute_hv, estimate_hv_bounds  # noqa: E402
import hv_box  # noqa: E402

REF = (1.02, 1.02)


def stars(p):
    return ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s.")


def wlx(a, b=None):
    try:
        return float(stats.wilcoxon(a, b)[1])
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lab_json", required=True)
    args = ap.parse_args()

    rows = json.load(open(args.lab_json, encoding="utf-8"))
    if not rows:
        print("[错误] 结果为空:", args.lab_json)
        return 1

    fronts = [np.asarray(r["final_pf"], float) for r in rows if r["final_pf"]]
    lo, hi = estimate_hv_bounds(fronts)
    # ── 口径冻结：把盒的臂集与指纹落盘，使"这份绝对 HV 用的是哪个盒"可复核 ──
    box = hv_box.make_box(fronts, arms=sorted({r["label"] for r in rows}))

    hv = collections.defaultdict(dict)
    tdist = collections.defaultdict(collections.Counter)
    qtabs = collections.defaultdict(list)
    tt = collections.defaultdict(list)
    budget_raw = collections.defaultdict(list)
    for r in rows:
        hv[r["label"]][r["seed"]] = compute_hv(np.asarray(r["final_pf"], float),
                                               ref_point=REF, norm_bounds=(lo, hi))
        tt[r["label"]].append(r.get("total_time", np.nan))
        for k, c in (r.get("hist_T") or {}).items():
            tdist[r["label"]][int(k)] += c
        if r.get("q_table"):
            qtabs[r["label"]].append(r["q_table"])
        if r.get("rvns_budget"):
            budget_raw[r["label"]].append(r["rvns_budget"])

    seeds = sorted({r["seed"] for r in rows})
    arms = sorted(hv, key=lambda a: (not re.fullmatch(r"T\d+", a), a))
    inst = rows[0].get("instance") or ""
    if not inst:
        # 兼容早期结果 JSON（无 instance 字段）——从文件名推断，如 _mk10_lab.json
        m = re.search(r"_([A-Za-z]+\d+)_lab", os.path.basename(args.lab_json))
        inst = m.group(1).upper() if m else "?"

    A = lambda a: np.array([hv[a][s] for s in seeds if s in hv[a]])
    T_arms = sorted([a for a in arms if re.fullmatch(r"T\d+", a)],
                    key=lambda x: int(x[1:]))
    Q = [a for a in arms if a.startswith("QPAS")]
    F = [a for a in arms if a.startswith("Full")]

    print("=" * 100)
    print(f"T-leverage / Q-PAS lab  |  {inst}  |  arms={len(arms)}  seeds={len(seeds)}  "
          f"runs={len(rows)}")
    print(f"reference-set normalization  lo={np.round(lo, 2)}  hi={np.round(hi, 2)}  ref={REF}")
    print(hv_box.format_box(box))
    _sp = hv_box.write_box_sidecar(args.lab_json, box)
    print(f"norm-box sidecar -> {_sp}")
    print("  注：全部数字（含 ΔHV / p / wins）只在**同一盒**内可比（看 sha1）；"
          "换臂集会改变相对大小乃至最优臂归属（缺陷 25/30/39）。")
    print("=" * 100)
    print(f"{'arm':<18}{'HV mean':<11}{'std':<9}{'median':<11}{'min':<10}{'max':<10}{'time(s)'}")
    print("-" * 100)
    for a in arms:
        v = A(a)
        print(f"{a:<18}{v.mean():<11.5f}{v.std():<9.5f}{np.median(v):<11.5f}"
              f"{v.min():<10.5f}{v.max():<10.5f}{np.nanmean(tt[a]):<8.1f}")

    oracle = None
    best_fixed = None
    if T_arms:
        mat = np.array([A(a) for a in T_arms])
        means = mat.mean(axis=1)
        best_fixed = T_arms[int(means.argmax())]
        oracle = mat.max(axis=0)
        st, p = stats.friedmanchisquare(*mat)
        base = A(T_arms[0]) if "T10" not in T_arms else A("T10")

        print("\n" + "=" * 100)
        print("[1] 固定 T 的杠杆检验（纯 T 效应）")
        print("=" * 100)
        print(f"Friedman over {T_arms}: chi2={st:.4f}  p={p:.6g} {stars(p)}")
        print(f"HV span (best-worst of means) = {means.max() - means.min():.5f}  "
              f"({best_fixed} {means.max():.5f} vs "
              f"{T_arms[int(means.argmin())]} {means.min():.5f})   "
              f"relative = {(means.max() - means.min()) / means.mean() * 100:.2f}%")
        ref_name = "T10" if "T10" in T_arms else T_arms[0]
        print(f"\n以 {ref_name} 为基准：")
        print(f"{'arm':<10}{'dHV':<13}{'wins':<10}{'p':<12}")
        for a in T_arms:
            v = A(a)
            d = v - A(ref_name)
            pv = wlx(v, A(ref_name))
            print(f"{a:<10}{d.mean():<+13.5f}{int((d > 0).sum()):>2}/{len(d):<7}"
                  f"{pv:<12.4g}{stars(pv)}")
        print(f"\nper-seed oracle (每个 seed 取最优 T) mean={oracle.mean():.5f}")
        for a in T_arms:
            print(f"   gain if fixed {a:<6} -> oracle: {oracle.mean() - A(a).mean():+.5f}")

    if Q:
        print("\n" + "=" * 100)
        print("[2] Q-PAS 各设定（关闭 RVNS）")
        print("=" * 100)
        ref_name = "T10" if "T10" in hv else None
        head = f"{'arm':<18}{'HV':<10}"
        if ref_name:
            head += f"{'vs ' + ref_name:<22}"
        if best_fixed:
            head += f"{'vs ' + best_fixed:<22}"
        if oracle is not None:
            head += "gap-oracle"
        print(head)
        for a in Q:
            v = A(a)
            line = f"{a:<18}{v.mean():<10.5f}"
            if ref_name:
                pv = wlx(v, A(ref_name))
                line += f"{v.mean() - A(ref_name).mean():+.5f} {stars(pv):<13}"
            if best_fixed:
                pbf = wlx(v, A(best_fixed))
                line += f"{v.mean() - A(best_fixed).mean():+.5f} {stars(pbf):<13}"
            if oracle is not None:
                line += f"{v.mean() - oracle.mean():+.5f}"
            print(line)
        pool = Q + ([ref_name] if ref_name else [])
        if len(pool) > 2:
            st2, p2 = stats.friedmanchisquare(*[A(a) for a in pool])
            print(f"\nFriedman over {pool}: chi2={st2:.4f} p={p2:.6g} {stars(p2)}")

        print("\n  Q-PAS 实际学到的 T 使用分布（占各 run 代数比例）:")
        for a in Q:
            tot = sum(tdist[a].values())
            if tot:
                dist = {k: f"{v / tot * 100:.1f}%" for k, v in sorted(tdist[a].items())}
                print(f"    {a:<18} {dist}")
        print("\n  Q-table（每个臂的末次 run，行 = state）:")
        for a in Q:
            if qtabs[a]:
                q = np.array(qtabs[a][-1])
                acts = sorted(tdist[a]) or list(range(q.shape[1]))
                print(f"    {a:<18} argmax/state={[int(x) for x in q.argmax(axis=1)]}  "
                      f"qmax={q.max():.3f}  T-grid={acts}")

    if F:
        print("\n" + "=" * 100)
        print("[3] Full（Q-PAS + RVNS）")
        print("=" * 100)
        cmp_arms = ([best_fixed] if best_fixed else []) + \
                   (["T10"] if "T10" in hv else []) + Q
        for fname in F:
            fh = A(fname)
            print(f"  {fname}: HV={fh.mean():.5f}")
            for a in cmp_arms:
                if a == fname:
                    continue
                d = fh - A(a)
                pv = wlx(fh, A(a))
                print(f"     vs {a:<18} dHV={d.mean():+.5f}  "
                      f"wins={int((d > 0).sum()):>2}/{len(d)}  p={pv:<9.4g} {stars(pv)}")

    # ------------------------------------------------------------------
    # [4] 组件分解：把「加上局部搜索本身」与「RL 引导选择算子」拆开
    #     对应论文阶梯 D2 -> D3 -> ... -> RMOEA/D
    # ------------------------------------------------------------------
    base = "T10" if "T10" in hv else None
    if base and "RandVNS" in hv and "RVNSonly" in hv:
        print("\n" + "=" * 100)
        print("[4] 组件分解（对齐论文变体阶梯；局部搜索用固定 T=10）")
        print("=" * 100)
        pairs = [
            ("RandVNS", base,      "加上局部搜索本身（算子随机选）"),
            ("RVNSonly", "RandVNS", "RL 引导选算子 替代 随机选算子"),
            ("RVNSonly", base,      "加上 RL 局部搜索（合计）"),
        ]
        if F:
            pairs.append((F[0], "RVNSonly", "在已有局部搜索上再加 Q-PAS"))
        if Q:
            best_q = max(Q, key=lambda a: A(a).mean())
            pairs.append((best_q, base, f"Q-PAS 单独（选最优设定 {best_q}）"))
        # 局部搜索强度对照（若已有该批臂）
        if "RandVNS_t3" in hv and "RVNSonly_t3" in hv:
            pairs.append(("RandVNS_t3", base, "加上局部搜索本身（ls_trials=3）"))
            pairs.append(("RVNSonly_t3", "RandVNS_t3",
                          "RL 引导选算子 替代 随机选算子（ls_trials=3）"))
            pairs.append(("RVNSonly_t3", "RVNSonly",
                          "把每代邻域尝试 1 -> 3 次"))
            if "Full_t3" in hv:
                pairs.append(("Full_t3", "RVNSonly_t3",
                              "在 ls_trials=3 局部搜索上再加 Q-PAS（同强度隔离）"))
        print(f"{'A 相对 B':<34}{'dHV':<12}{'wins':<10}{'p':<12}{'相对 %':<9}")
        print("-" * 100)
        for a, b, note in pairs:
            if a not in hv or b not in hv:
                continue
            va, vb = A(a), A(b)
            d = va - vb
            pv = wlx(va, vb)
            print(f"{a + ' vs ' + b:<34}{d.mean():<+12.5f}"
                  f"{int((d > 0).sum()):>2}/{len(d):<7}{pv:<12.4g}{stars(pv):<9}"
                  f"{d.mean() / vb.mean() * 100:+.2f}%   {note}")
        print("-" * 100)
        print("注：'加上局部搜索本身' 与 'RL 引导选算子' 是两件事。论文阶梯里")
        print("    随机 VNS 已在 D3 就位，所以它测到的 RVNS 增益只是后者。")

    # ------------------------------------------------------------------
    # [5] ABA：等算力条件下的邻域搜索预算分配
    #     四臂的总预算逐代严格相等，"预算发给谁"是唯一的自变量
    # ------------------------------------------------------------------
    ABA = ["RVNSonly_t2", "RVNSonly_Brand", "RVNSonly_Bstate", "RVNSonly_Blearn",
           "RVNSonly_Bhot"]
    have = [a for a in ABA if a in hv]
    if have:
        bagg = {}
        for a, v in budget_raw.items():
            bagg[a] = {
                "planned": float(np.mean([b["mean_planned_trials"] for b in v])),
                "evals": float(np.mean([b["mean_actual_evals"] for b in v])),
                "upgrades": float(np.mean([b["upgrades"] for b in v])),
                "decisions": float(np.mean([b["decisions"] for b in v])),
                "prop": np.mean([b["propensities"] for b in v], axis=0).tolist(),
                # 分桶样本量 [n(stuck=0), n(stuck=1)] 的跨 run 均值 —— 用于证明
                # 不同策略下的"条件回报"是**在不同条件集上**算的，不可互推。
                "n0": float(np.mean([b["upgrade_records"][0][0] for b in v
                                     if b.get("upgrade_records")])),
                "n1": float(np.mean([b["upgrade_records"][0][1] for b in v
                                     if b.get("upgrade_records")])),
            }

        def _cmp(a, b):
            # 配对必须按 seed 交集，缺 seed 的臂不能被拉进来凑均值
            ss = [s for s in seeds if s in hv[a] and s in hv[b]]
            va = np.array([hv[a][s] for s in ss])
            vb = np.array([hv[b][s] for s in ss])
            d = va - vb
            p = wlx(va, vb)
            # 配对 Cohen's dz（预注册 §3「必报项」要求效应量，只报 p 不够）
            dz = float(d.mean() / d.std(ddof=1)) if len(d) > 1 and d.std(ddof=1) else float("nan")
            return d.mean(), int((d > 0).sum()), len(d), p, stars(p), dz

        print("\n" + "=" * 100)
        print("[5] ABA：等预算下的邻域搜索预算分配")
        print("=" * 100)
        print("  5.1 算力核对（先证明'等算力'不是声称）")
        print("  %-18s %-12s %-14s %-11s %-10s"
              % ("arm", "计划均值", "实际求值均值", "升级率", "单run(s)"))
        ref_arms = [x for x in ("RVNSonly", "RVNSonly_t2", "RVNSonly_t3",
                                "RandVNS", "RandVNS_t3") if x in hv]
        for a in [x for x in ref_arms if x not in have] + have:
            b = bagg.get(a)
            if b and b["decisions"]:
                print("  %-18s %-12.4f %-14.4f %-11s %-10.1f"
                      % (a, b["planned"], b["evals"],
                         "%.1f%%" % (100.0 * b["upgrades"] / b["decisions"]),
                         np.nanmean(tt[a])))
            else:
                print("  %-18s %-12s %-14s %-11s %-10.1f"
                      % (a, "(无统计)" if a not in bagg else "(无预算模式)",
                         "-", "-", np.nanmean(tt[a])))
        pv = [bagg[a]["planned"] for a in have if a in bagg]
        if pv:
            print("  -> pool_* 的计划均值极差 = %.6f（应为 0；非 0 即算力不匹配）"
                  % (max(pv) - min(pv)))

        print("\n  5.2 学习/状态驱动的分配 vs 等算力基准")
        base = "RVNSonly_t2"
        if base in hv:
            print("  %-36s %-11s %-10s %-12s %-9s %s"
                  % ("对照", "dHV", "胜负", "p", "相对%", "dz"))
            for a in have:
                if a == base:
                    continue
                d, wn, n, p, st, dz = _cmp(a, base)
                print("  %-36s %-+11.5f %-10s %-12s %+.2f%%  %-9s %s"
                      % (a + " vs " + base, d, "%d/%d" % (wn, n), "%.4g" % p,
                         d / A(base).mean() * 100, "%+.3f" % dz, st))
        else:
            print("  (缺等算力基准 %s)" % base)

        print("\n  5.3 隔离机制：与异质性对照比（同总预算、同升级名额数）")
        ctl = "RVNSonly_Brand"
        if ctl in hv:
            for a in ("RVNSonly_Bstate", "RVNSonly_Blearn", "RVNSonly_Bhot"):
                if a in hv:
                    d, wn, n, p, st, dz = _cmp(a, ctl)
                    print("  %-36s %-+11.5f %-10s %-12s %-9s %s"
                          % (a + " vs " + ctl, d, "%d/%d" % (wn, n), "%.4g" % p,
                             "%+.3f" % dz, st))
            print("  -> 所有行都不显著 = '发得准'相对'随机发'没有额外收益。")
        else:
            print("  (缺异质性对照 %s)" % ctl)

        print("\n  5.4 与固定预算参照组比较（算力不同，仅定位，不作等算力结论）")
        for a in have:
            for b in ("RVNSonly", "RVNSonly_t3"):
                if b in hv:
                    d, wn, n, p, st, dz = _cmp(a, b)
                    print("  %-36s %-+11.5f %-10s %-12s %-9s %s"
                          % (a + " vs " + b, d, "%d/%d" % (wn, n), "%.4g" % p,
                             "%+.3f" % dz, st))

        print("\n  5.5 学到的升级倾向（Laplace 估计 [stuck=0=刚改进过, stuck=1=卡住]）")
        print("    %-18s %-24s %s" % ("arm", "倾向 [stuck0, stuck1]", "分桶样本量 [n0, n1]"))
        for a in have:
            if a in bagg:
                print("    %-18s %-24s [%.1f, %.1f]"
                      % (a, [round(x, 4) for x in bagg[a]["prop"]],
                         bagg[a]["n0"], bagg[a]["n1"]))
        print("    0.5/0.5 = 没学到任何东西（先验）；显著偏离才说明升级有回报差异。")
        print("    [!] 该统计**带策略依赖**，两重原因，故不能反推方向：")
        print("        (a) 混杂性——**分桶样本量随策略剧变**。Bstate 的加码名额几乎")
        print("            全部落在 stuck=1 上，它的 [stuck=0] 桶样本量≈0.3/run")
        print("            （相对 10000 个升级名额），那个 0.486 基本就是 Laplace")
        print("            先验 (0+1)/(0+2)=0.5 本身，**不是测量值**。")
        print("            于是 Bstate 0.486 vs Brand 0.115 的'反转'根本不是同一条件集")
        print("            上的两个估计，而是'一个真实测量'对'一个空桶'。")
        print("        (b) 定义性——只有'命中发生在第2次及以后'才算回报，")
        print("            刚改进过的解更容易'首试失败、次试成功'，天然偏高。")
        print("        结论：倾向数字只能在**同一策略内**读；跨策略比较必须另设臂（见 5.6），")
        print("              且必须同时报分桶样本量——只报倾向会读出一个不存在的结果。")

        print("\n  5.6 反极性臂（探索性 / post-hoc）：直接检验方向")
        print("      5.5 里 Bstate 的 [stuck=0] 回报远高于 [stuck=1]，但那只说明")
        print("      '被升级的刚改进解回报率高'，不等于'该升级刚改进解'。Bhot 把")
        print("      同预算、同名额、同档位分布下的方向翻过来，这是唯一干净的判据。")
        for pair in (("RVNSonly_Bhot", "RVNSonly_Brand"),
                     ("RVNSonly_Bhot", "RVNSonly_Bstate"),
                     ("RVNSonly_Bhot", "RVNSonly_t2")):
            a, b = pair
            if a in hv and b in hv:
                d, wn, n, p, st, dz = _cmp(a, b)
                print("  %-36s %-+11.5f %-10s %-12s %-9s %s"
                      % (a + " vs " + b, d, "%d/%d" % (wn, n), "%.4g" % p,
                         "%+.3f" % dz, st))
        print("  -> 若 Bhot 相对 Brand / t2 都不显著，则 5.5 的方向反转是")
        print("     '回报定义'造成的伪影，**不得**写成方法学发现。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
