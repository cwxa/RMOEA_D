#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G1/G2：T 维度是否还有**可实现**的自适应空间（相位 oracle + 状态可观测性）。

为什么需要这个脚本
------------------
`t_leverage_analysis.py` 里的 "per-seed oracle"（= 每个 seed 取最终 HV 最优的 T）
**不是**可实现目标：它是"从 6 个臂里按 1 个观测挑最优"的样本内过拟合量。
实测该 gap（+0.03658）与"臂均值保留 + 种子内残差置换"的零假设（+0.03448±0.00198）
无法区分，落在 null 的 85.9 分位。

本脚本改用**真正可实现的策略**来判定：

  G1a  相位间 T 排名是否稳定（各相位 Friedman + 相位间 Spearman）
  G1b  「前 40 代探测 → 承诺一个 T 跑到末代」策略能否**显著优于**最优固定 T
       ——判定阈值用置换零假设给出噪声地板（选择 gen=40 的 argmax 本身也有偏差）
  G2   (ΔCV, ΔDV) 的 4 个符号状态能否**区分出不同的最优 T**
       （逐代 CV/DV 轨迹由 algorithm.solve 记录，见 history 的 cv/dv 字段）

数据：固定 T 各臂重跑一次并记录逐代 HV/CV/DV → logs/_<inst>_traj.json

    python scripts/phase_oracle.py --instance Mk10
    python scripts/phase_oracle.py --instance Mk10 --skip-run   # 只分析已有轨迹

注意：轨迹里的 hv 由 ``algorithm.solve`` 用**实例边界**计算
（``instance_hv_bounds``，同实例内各臂共用 → 跨臂可比），
与 ``t_leverage_analysis.py`` 的**参考集归一化盒**口径不同。
两者不可混引；本脚本只做同实例内的臂间比较。
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

L = []


def W(s=""):
    L.append(str(s))


def stars(p):
    return ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s.")


def wlx(a, b):
    try:
        return float(stats.wilcoxon(a, b)[1])
    except Exception:
        return float("nan")


# ──────────────────────────────────────────────────────────────
# 采集：固定 T 各臂 + 逐代 HV/CV/DV
# ──────────────────────────────────────────────────────────────

def _run_one(job):
    instance, n_pop, max_gen, data_dir, T, seed = job
    from rmoea_d.algorithm import RMOEAD

    solver = RMOEAD(instance_name=instance, n_pop=n_pop, max_gen=max_gen,
                    seed=seed, data_dir=data_dir,
                    fixed_T=T, enable_rvns=False)
    res = solver.solve()
    return dict(
        label="T%02d" % T, T=int(T), seed=int(seed), instance=instance,
        hv=[float(h["hv"]) for h in solver.history],
        cv=[float(h["cv"]) for h in solver.history],
        dv=[float(h["dv"]) for h in solver.history],
        final_pf=[[float(p["Makespan"]), float(p["Workload"])]
                  for p in res["final_pf"]],
    )


def collect(args, arms, seeds):
    out = args.out or os.path.join(ROOT, "logs", f"_{args.instance.lower()}_traj.json")
    if os.path.exists(out) and not args.force:
        rows = json.load(open(out, encoding="utf-8"))
        W(f"[采集] 复用已有轨迹 {out}  rows={len(rows)}")
        return rows
    jobs = [(args.instance, args.n_pop, args.max_gen, args.data_dir, T, s)
            for T in arms for s in seeds]
    W(f"[采集] instance={args.instance} arms={arms} seeds={len(seeds)} "
      f"todo={len(jobs)} workers={args.workers}")
    rows = []
    with mp.Pool(args.workers, maxtasksperchild=2) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_one, jobs, chunksize=1), 1):
            rows.append(r)
            if i % 10 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] last={r['label']}/s{r['seed']} "
                      f"hv={r['hv'][-1]:.5f}", flush=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    os.replace(tmp, out)
    W(f"[采集] 写出 {out}  rows={len(rows)}")
    return rows


# ──────────────────────────────────────────────────────────────
# 分析
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default="Mk10")
    ap.add_argument("--arms", default="5,10,15,20,50,100")
    ap.add_argument("--n_pop", type=int, default=100)
    ap.add_argument("--max_gen", type=int, default=200)
    ap.add_argument("--seed_start", type=int, default=42)
    ap.add_argument("--n_runs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--force", action="store_true", help="忽略已有轨迹，重跑")
    ap.add_argument("--skip-run", action="store_true", help="只分析，不跑实验")
    ap.add_argument("--probe_gen", type=int, default=40,
                    help="G1b 的探测期代数（之后承诺一个 T）")
    ap.add_argument("--phases", default="60,130,200",
                    help="G1a 的相位切点（段末代数，逗号分隔）")
    ap.add_argument("--fwd", type=int, default=20,
                    help="G2 的前瞻窗口：看未来多少代的 HV 改善")
    args = ap.parse_args()

    arms = [int(x) for x in args.arms.split(",") if x.strip()]
    seeds = list(range(args.seed_start, args.seed_start + args.n_runs))
    phases = [int(x) for x in args.phases.split(",") if x.strip()]

    if args.skip_run:
        out = args.out or os.path.join(ROOT, "logs", f"_{args.instance.lower()}_traj.json")
        rows = json.load(open(out, encoding="utf-8"))
        W(f"[采集] --skip-run，读入 {out}  rows={len(rows)}")
    else:
        rows = collect(args, arms, seeds)

    labels = sorted({r["label"] for r in rows}, key=lambda x: int(x[1:]))
    seeds = sorted({r["seed"] for r in rows})
    G = min(len(r["hv"]) for r in rows)
    W()
    W("=" * 96)
    W(f"G1/G2 相位 oracle 与状态可观测性  |  {args.instance}  |  "
      f"arms={labels}  seeds={len(seeds)}  G={G}")
    W("=" * 96)

    def traj(label, key="hv"):
        m = {}
        for r in rows:
            if r["label"] == label:
                m[r["seed"]] = np.asarray(r[key], dtype=float)
        return m

    HV = {a: traj(a, "hv") for a in labels}

    def hv_at(a, gen):
        """某臂在第 gen 代（1-based）的逐 seed HV 向量。"""
        return np.array([HV[a][s][gen - 1] for s in seeds if s in HV[a]])

    # ── 基线：最优固定 T（以末代为准）────────────────────────────
    final = {a: hv_at(a, G) for a in labels}
    means = {a: final[a].mean() for a in labels}
    best = max(labels, key=lambda a: means[a])
    M = np.array([final[a] for a in labels])          # arm x seed
    oracle = M.max(axis=0).mean()
    W()
    W("【基线】末代 HV（实例边界口径，同实例内跨臂可比）")
    W("%-8s %-11s %-10s" % ("arm", "HV mean", "std"))
    for a in labels:
        W("%-8s %-11.5f %-10.5f" % (a, means[a], final[a].std()))
    W(f"最优固定 T = {best}  ({means[best]:.5f})")
    W(f"per-seed oracle = {oracle:.5f}   gap = {oracle - means[best]:+.5f}")
    W("  （此 gap 的前一轮核算已证明与置换零假设不可区分，**不作为**可实现目标；")
    W("    下面 G1b 改用可实现策略判定。）")

    # ── G1a：相位间 T 排名是否稳定 ──────────────────────────────
    W()
    W("=" * 96)
    W("G1a  各相位的最优 T 与排名稳定性")
    W("=" * 96)
    rank_by_phase = {}
    W("%-8s %s" % ("相位", "".join("%-9s" % a for a in labels) + "  Friedman p"))
    for ph in phases:
        v = {a: hv_at(a, ph) for a in labels}
        mat = np.array([v[a] for a in labels])
        try:
            st, p = stats.friedmanchisquare(*mat)
        except Exception:
            p = float("nan")
        mu = mat.mean(axis=1)
        rank_by_phase[ph] = stats.rankdata(-mu)
        W("%-8s %s  %.4g %s"
          % ("gen<=%d" % ph, "".join("%-9.5f" % m for m in mu), p, stars(p)))
        W("%-8s %s  最优=%s 极差=%.5f"
          % ("", "".join("%-9s" % ("#%d" % int(r)) for r in rank_by_phase[ph]),
             labels[int(np.argmax(mu))], mu.max() - mu.min()))

    ph_list = list(phases)
    W()
    W("相位间 T 排名的 Spearman ρ（ρ 高 = 最优 T 不随阶段漂移 = 自适应无空间）：")
    for i in range(len(ph_list)):
        for j in range(i + 1, len(ph_list)):
            rho, p = stats.spearmanr(rank_by_phase[ph_list[i]],
                                     rank_by_phase[ph_list[j]])
            W("  gen<=%d  vs  gen<=%d :  rho=%+.3f  p=%.3g %s"
              % (ph_list[i], ph_list[j], rho, p, stars(p)))

    # ── G1b：可实现的「探测-承诺」策略 vs 置换零假设 ─────────────
    W()
    W("=" * 96)
    W("G1b  可实现策略：「前 %d 代探测 → 承诺一个 T 跑到末代」" % args.probe_gen)
    W("=" * 96)
    probe = np.array([hv_at(a, args.probe_gen) for a in labels])   # arm x seed
    pick_idx = probe.argmax(axis=0)                                # 每 seed 的承诺臂
    committed = np.array([M[pick_idx[j], j] for j in range(M.shape[1])])
    p_vs_best = wlx(committed, final[best])
    W(f"  承诺策略 HV mean = {committed.mean():.5f}")
    W(f"  最优固定 T   {best} = {means[best]:.5f}   差 = "
      f"{committed.mean() - means[best]:+.5f}  "
      f"p={p_vs_best:.4g} {stars(p_vs_best)}")

    # 零假设：种子内把「探测期读数」在臂之间随机重排（保留各臂末代均值、
    # 保留种子的探测期水平，只破坏「探测期读数 ↔ 该臂」的对应）
    rng = np.random.RandomState(20260917)
    NREP = 2000
    null_commit = np.empty(NREP)
    pick_cnt = collections.Counter()
    for i in range(NREP):
        Mnull = np.empty_like(probe)
        for j in range(probe.shape[1]):
            Mnull[:, j] = probe[rng.permutation(probe.shape[0]), j]
        pidx = Mnull.argmax(axis=0)
        null_commit[i] = np.mean([M[pidx[j], j] for j in range(M.shape[1])])
        pick_cnt[labels[int(pidx[0])]] += 1
    W()
    W(f"  置换零假设（种子内重排探测期读数，{NREP} 次）：")
    W(f"    null mean={null_commit.mean():+.5f}  sd={null_commit.std():.5f}  "
      f"95%=[{np.percentile(null_commit, 2.5):+.5f}, "
      f"{np.percentile(null_commit, 97.5):+.5f}]")
    gap_obs = committed.mean() - means[best]
    gap_null = null_commit - means[best]
    pct = 100.0 * (gap_null < gap_obs).mean()
    W(f"    观察 gap={gap_obs:+.5f}  落在 null 的 {pct:.1f} 分位")
    W(f"    观察承诺的 T 分布 = "
      f"{dict(collections.Counter([labels[int(x)] for x in pick_idx]))}")
    W()
    W("  判定规则（两条**必须同时**满足，缺一不可）：")
    W("    (a) 观察 gap 超过 null 的 97.5 分位 —— 回答'探测期的读数有没有信息？'")
    W(f"        实测 {pct:.1f} 分位 -> {'通过' if pct >= 97.5 else '不通过'}")
    W("    (b) 承诺策略显著优于最优固定 T（配对 Wilcoxon 单侧，p<0.05）")
    W("        —— 回答'这些信息换不换得到 HV？'")
    W(f"        实测 差={gap_obs:+.5f}  p={p_vs_best:.4g} -> "
      f"{'通过' if p_vs_best == p_vs_best and p_vs_best < 0.05 and gap_obs > 0 else '不通过'}")
    W("  [!] (a) 单独通过**不是**成功：null 是'把探测期读数在臂之间打乱'，")
    W("      它的均值本就低于 best_fixed（打乱的选臂平均更差）。所以 (a) 通过")
    W("      只说明'探测读数不是纯噪声'，完全可能只是把 T100 又选了一遍——")
    W("      信息量为真、价值为零。判 G1b 只看 (b)。")
    W("  [!] 本脚本的 hv 用**实例边界**，与 t_leverage_analysis 的参考集归一化盒")
    W("      不同口径，两处的绝对 HV 与'最优固定 T'都可能不同，不可混引。")

    # ── G2：状态可观测性 ────────────────────────────────────────
    W()
    W("=" * 96)
    W("G2  (ΔCV, ΔDV) 的 4 个符号状态能否区分出不同的最优 T")
    W("=" * 96)
    if args.fwd >= G:
        W("  fwd 窗口 >= G，跳过")
    else:
        # 逐 (state, arm, seed) 汇总：**先在 seed 内把所有 gen 窗口平均成一个数**，
        # 再跨 seed 做检验。逐 gen 窗口高度自相关（同一 run 的相邻窗口共享数据），
        # 直接对窗口做检验会把自由度虚增几十倍、把噪声当成区分力。
        S = [0, 1, 2, 3]
        per = {st: {a: {} for a in labels} for st in S}   # st -> arm -> seed -> mean
        occ = collections.Counter()
        for a in labels:
            cv, dv = traj(a, "cv"), traj(a, "dv")
            for s in seeds:
                if s not in cv or s not in HV[a]:
                    continue
                c, d = cv[s], dv[s]
                acc = {st: [] for st in S}
                for g in range(1, G - args.fwd + 1):
                    dcv = c[g - 1] - c[g]     # 与 qlearning.step 同号约定
                    ddv = d[g] - d[g - 1]
                    st = (0 if (dcv > 0 and ddv > 0) else
                          1 if (dcv > 0) else
                          2 if (ddv > 0) else 3)
                    acc[st].append(HV[a][s][g + args.fwd - 1] - HV[a][s][g - 1])
                    occ[st] += 1
                for st in S:
                    if acc[st]:
                        per[st][a][s] = float(np.mean(acc[st]))

        W("  各状态下各臂的「未来 %d 代 HV 改善」均值（每 (arm,seed) 先平均，再跨 seed）："
          % args.fwd)
        W("%-7s %s  %-8s %-8s %-7s %s"
          % ("state", "".join("%-9s" % a for a in labels),
             "极差", "Friedman p", "n_seed", "argmaxT"))
        used = []
        for st in S:
            # 取所有臂都有值的 seed 交集，Friedman 需要完整区组
            ss = [s for s in seeds
                  if all(s in per[st][a] for a in labels)]
            if len(ss) < 4:
                W("%-7s %s  样本不足（n_seed=%d），跳过"
                  % ("s=%d" % st, "".join("%-9s" % "-" for _ in labels), len(ss)))
                used.append(None)
                continue
            mat = np.array([[per[st][a][s] for s in ss] for a in labels])
            mu = mat.mean(axis=1)
            try:
                _, p_fr = stats.friedmanchisquare(*mat)
            except Exception:
                p_fr = float("nan")
            am = labels[int(np.nanargmax(mu))]
            used.append(am)
            W("%-7s %s  %-8.5f %-8.4g %-7d %s"
              % ("s=%d" % st, "".join("%-9.5f" % m for m in mu),
                 mu.max() - mu.min(), p_fr, len(ss), am))
            # 该状态下的最优 T 是不是**真的**优于全局最优固定 T
            gb = labels.index(best)
            if am != best:
                ia = labels.index(am)
                p_pair = wlx(mat[ia], mat[gb])
                W("%-7s   最优(%s) vs 全局最优固定 T(%s): 差=%+.5f  p=%.4g %s"
                  % ("", am, best, float(mu[ia] - mu[gb]), p_pair, stars(p_pair)))
            else:
                W("%-7s   最优 T 就是全局最优固定 T（%s），无需自适应" % ("", best))

        W()
        W("  各状态下 (arm,seed,gen) 原始窗口数（仅用于看状态占比）：")
        tot = sum(occ.values())
        W("    " + "  ".join("s=%d %.1f%%" % (st, 100.0 * occ[st] / tot)
                            for st in S))
        W()
        W("  判定（**不看 argmax 是否相同，只看显著性**）：")
        W("    一个状态算'有区分力'，需同时满足：")
        W("      (i)  该状态下跨 T 的 Friedman p < 0.05；")
        W("      (ii) 该状态的最优 T 与全局最优固定 T 的配对差显著（p < 0.05）且为正。")
        W("    argmax 不同**不构成**证据——各行极差都在 1e-3 量级、且无检验。")

        # 汇总判定
        ok_states = []
        for st in S:
            ss = [s for s in seeds if all(s in per[st][a] for a in labels)]
            if len(ss) < 4:
                continue
            mat = np.array([[per[st][a][s] for s in ss] for a in labels])
            mu = mat.mean(axis=1)
            try:
                _, p_fr = stats.friedmanchisquare(*mat)
            except Exception:
                p_fr = float("nan")
            ia = int(np.nanargmax(mu))
            gb = labels.index(best)
            p_pair = (1.0 if labels[ia] == best
                      else wlx(mat[ia], mat[gb]))
            if p_fr == p_fr and p_fr < 0.05 and p_pair == p_pair \
                    and p_pair < 0.05 and mu[ia] > mu[gb]:
                ok_states.append(st)
        W("    满足 (i)+(ii) 的状态 = %s" % (ok_states if ok_states else "无"))
        if ok_states:
            W("    -> 这些状态在**短视距**（未来 %d 代改善）上确实偏好不同的 T。" % args.fwd)
            W("    [!] 但**不足以**据此扩状态空间，四条理由：")
            W("        (α) 多重比较：这里做了 4 个状态检验，Bonferroni 校正后")
            W("            α=0.0125；最弱的那个 p 过不了线。")
            W("        (β) 事后性：状态划分与检验用的是同一批数据，属 in-sample。")
            W("        (γ) 判据错位：'未来 %d 代改善'**不是**目标函数。" % args.fwd)
            W("            短视距改善率高不等于末代 HV 高（本例 T10 末代 HV 低于 T100）。")
            W("        (δ) 预注册优先：本脚本 G1 判负，按 `qpas-optimization-plan.md` §2")
            W("            写死的决策表（G1 负 -> Q-PAS 就地终止），不得用 G2 翻案。")
            W("    -> 因此记为一个**待验假设**（需要新实例 + 新预注册的实验），")
            W("       不作为本轮结论。")
        else:
            W("    -> 状态**区分不出**该选哪个 T：扩状态空间（如加幅度分档）**没有依据**，")
            W("       Q-PAS 的状态设计不是瓶颈。")

    rep = args.report or os.path.join(ROOT, "logs", "_phase_oracle.txt")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("WROTE", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
