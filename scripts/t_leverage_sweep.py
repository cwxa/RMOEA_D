#!/usr/bin/env python3
"""邻域大小 T 的杠杆扫参 + Q-PAS 设定实验台（分批 · 增量落盘 · 可断点续跑）。

用途
----
回答「自适应选 T（Q-PAS）在这个实例上到底有没有发挥空间」：
先扫固定 T，判断 T 是不是有效杠杆；再跑 Q-PAS 各设定，看它能否学到好 T。

用法
----
    # 固定 T 扫参（纯 T 效应，关闭 Q-PAS 与 RVNS）
    python scripts/t_leverage_sweep.py --instance Mk10 ^
        --arms T05,T10,T15,T20,T50,T100

    # Q-PAS 各设定（平局随机化后的 v2 才是当前默认口径）
    python scripts/t_leverage_sweep.py --instance Mk10 ^
        --arms QPAS2_dv,QPAS2_hv_wide,QPAS2_opt_hv,Full2

    # 重复运行会自动跳过已完成的 (arm, seed)，可分批续跑
    python scripts/t_leverage_analysis.py --lab_json logs/_mk10_lab.json

注意
----
* 所有 run 共用 ``instance_hv_bounds``（由实例数据确定性推出），但**跨臂比较请用
  ``t_leverage_analysis.py`` 统一重算**——它按参考集归一化（所有臂所有 run 的前沿并集，
  ref=(1.02,1.02)）重算 HV，与 ``scripts/ablation_analysis.py`` 口径一致。
* Q-PAS v1（``QPAS_*``）保留是为了复现「平局自锁」的历史结果；
  当前默认口径是 v2（``QPAS2_*``，``tie_break="random"``）。
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
logging.basicConfig(level=logging.WARNING)

# 实验臂定义：label -> (kind, kwargs)
#   kind: "fixed"   固定 T（无 Q-PAS、无局部搜索）
#         "qpas"    仅 Q-PAS（无局部搜索）
#         "full"    Q-PAS + RVNS
#         "rvns"    仅局部搜索，算子按 SM/FM 轮盘赌选
#         "randvns" 仅局部搜索，算子等概率随机选（论文 RMOEA/D3）
ARM_DEF = {
    # ── 纯 T 效应（关闭 Q-PAS 与 RVNS）──
    "T05":   ("fixed", dict(fixed_T=5)),
    "T10":   ("fixed", dict(fixed_T=10)),
    "T15":   ("fixed", dict(fixed_T=15)),
    "T20":   ("fixed", dict(fixed_T=20)),
    "T50":   ("fixed", dict(fixed_T=50)),
    "T100":  ("fixed", dict(fixed_T=100)),
    # ── Q-PAS v1：论文口径 + 仅修 ε 极性（平局仍取索引 0 → 自锁 T=5）──
    "QPAS_dv":      ("qpas", dict(ql_reward_mode="dv")),
    "QPAS_cv_dv":   ("qpas", dict(ql_reward_mode="cv_dv")),
    "QPAS_hv":      ("qpas", dict(ql_reward_mode="hv")),
    "QPAS_wide":    ("qpas", dict(ql_reward_mode="dv", ql_actions=[5, 10, 20, 50])),
    "QPAS_wide_hv": ("qpas", dict(ql_reward_mode="hv", ql_actions=[5, 10, 20, 50])),
    "Full":         ("full", dict(ql_reward_mode="dv")),
    # ── Q-PAS v2：平局随机化（打破自锁，当前默认口径）──
    "QPAS2_dv":       ("qpas", dict(ql_reward_mode="dv", ql_tie_break="random")),
    "QPAS2_wide":     ("qpas", dict(ql_reward_mode="dv", ql_tie_break="random",
                                    ql_actions=[5, 10, 20, 50])),
    "QPAS2_hv_wide":  ("qpas", dict(ql_reward_mode="hv", ql_tie_break="random",
                                    ql_actions=[5, 10, 20, 50])),
    "QPAS2_opt_wide": ("qpas", dict(ql_reward_mode="dv", ql_tie_break="random",
                                    ql_q_init="optimistic",
                                    ql_actions=[5, 10, 20, 50])),
    "QPAS2_wide_e5":  ("qpas", dict(ql_reward_mode="dv", ql_tie_break="random",
                                    ql_epsilon=0.5, ql_actions=[5, 10, 20, 50])),
    "Full2":          ("full", dict(ql_reward_mode="dv", ql_tie_break="random",
                                    ql_actions=[5, 10, 20, 50])),
    # ── Q-PAS v3：组合有效因子 / 更宽动作空间 / 连续 HV 奖励 ──
    "QPAS2_opt_hv":  ("qpas", dict(ql_reward_mode="hv", ql_tie_break="random",
                                   ql_q_init="optimistic",
                                   ql_actions=[5, 10, 20, 50])),
    "QPAS2_hv_fine": ("qpas", dict(ql_reward_mode="hv", ql_tie_break="random",
                                   ql_actions=[5, 10, 15, 20, 30, 50])),
    "QPAS3_hvc":     ("qpas", dict(ql_reward_mode="hv_cont", ql_tie_break="random",
                                   ql_actions=[5, 10, 20, 50])),
    "QPAS3_hvc_opt": ("qpas", dict(ql_reward_mode="hv_cont", ql_tie_break="random",
                                   ql_q_init="optimistic",
                                   ql_actions=[5, 10, 20, 50])),
    "Full3":         ("full", dict(ql_reward_mode="hv", ql_tie_break="random",
                                   ql_q_init="optimistic",
                                   ql_actions=[5, 10, 20, 50])),
    # ── 论文变体阶梯对齐臂 ──
    # 论文 RMOEA/D3「randomly selection VNS」：局部搜索在但算子等概率随机选
    "RandVNS":  ("randvns", dict(fixed_T=10, rvns_mode="random")),
    # 同窗口对照：局部搜索在、算子由 SM/FM 轮盘赌选（= 论文的 RVNS 选择机制）
    # RandVNS -> RVNSonly 才是论文阶梯里 D5 -> RMOEA/D 那一步的净贡献
    "RVNSonly": ("rvns", dict(fixed_T=10)),
    # ── 局部搜索强度对照 ──
    # ls_trials=1（论文 Algorithm 4）时算子选择只有一次机会，RL 引导几乎无处发力。
    # 把每代的邻域尝试次数提到 3，检验「RL 选算子」相对「随机选算子」是否才开始有意义。
    "RandVNS_t3":   ("randvns", dict(fixed_T=10, rvns_mode="random",
                                     rvns_ls_trials=3)),
    "RVNSonly_t3":  ("rvns", dict(fixed_T=10, rvns_ls_trials=3)),
    # Q-PAS 在强局部搜索下的干净隔离：只有 ls_trials 与 Full 不同，
    # 否则「Full vs RVNSonly_t3」会把 Q-PAS 与邻域尝试次数的差异混在一起。
    "Full_t3":      ("full", dict(ql_reward_mode="dv", rvns_ls_trials=3)),
    # ── CV 归一化：论文式(14) 未规定归一化，量纲悬殊时状态会被大量纲目标独占 ──
    # 实测 Mk10 上 f2 占 CV^2 的 96.5%，归一化后 44.7% 的历史状态会翻转。
    # 这两臂用来回答「状态空间的信息量是否是 Q-PAS 失效的瓶颈」。
    "QPAS2_dv_cvnorm":      ("qpas", dict(ql_reward_mode="dv", ql_tie_break="random",
                                         ql_cv_normalize=True)),
    "QPAS2_hv_wide_cvnorm": ("qpas", dict(ql_reward_mode="hv", ql_tie_break="random",
                                         ql_actions=[5, 10, 20, 50],
                                         ql_cv_normalize=True)),
    # ── ABA：等预算下的邻域搜索预算分配 ──
    # 背景：把全局 ls_trials 由 1 提到 3 是全场最大的单一新杠杆（+4.68%），
    # 但算力同时涨到 2.22×——那是"更狠"，不是"更聪明"。
    # 这四臂把总算力构造性固定在「每代每解均值 = 2」，只改变"预算发给谁"：
    #   RVNSonly_t2     等算力基准：所有解统一上限 2
    #   RVNSonly_Brand  异质性对照：随机决定升级哪些解（必须能排除
    #                   "预算随机波动本身就有用"这一解释，否则归因不成立）
    #   RVNSonly_Bstate 状态驱动：优先升级上一代未被改进的解
    #   RVNSonly_Blearn 学习驱动：升级偏好由 SM/FM 式信用统计学到
    "RVNSonly_t2":     ("rvns", dict(fixed_T=10, rvns_ls_trials=2)),
    "RVNSonly_Brand":  ("rvns", dict(fixed_T=10, rvns_ls_trials=1,
                                     rvns_budget_mode="pool_random",
                                     rvns_budget_pool=[1, 3],
                                     rvns_budget_target_mean=2.0)),
    "RVNSonly_Bstate": ("rvns", dict(fixed_T=10, rvns_ls_trials=1,
                                     rvns_budget_mode="pool_state",
                                     rvns_budget_pool=[1, 3],
                                     rvns_budget_target_mean=2.0)),
    "RVNSonly_Blearn": ("rvns", dict(fixed_T=10, rvns_ls_trials=1,
                                     rvns_budget_mode="pool_learn",
                                     rvns_budget_pool=[1, 3],
                                     rvns_budget_target_mean=2.0)),
    # 反极性臂（**探索性 / post-hoc**）：Mk10 首轮实测显示"升级卡住的解"
    # 几乎买不到回报（回报率 0.069），而"升级刚被改进过的解"高得多（0.486）。
    # 注意该统计带策略依赖，不能直接反推方向，所以补一个臂直接检验。
    # 在留出集（Mk07/Mk09）确认之前，这一臂不得作为结论。
    "RVNSonly_Bhot":   ("rvns", dict(fixed_T=10, rvns_ls_trials=1,
                                     rvns_budget_mode="pool_improved",
                                     rvns_budget_pool=[1, 3],
                                     rvns_budget_target_mean=2.0)),
    # ── 等算力基线（P3/B1）：把"多花算力"与"更聪明"分开 ──
    # `RVNSonly_t3`（ls_trials=3）比 `RVNSonly`（ls_trials=1）快 2.22 倍算力，
    # 却拿到了全场最大的单一增益（+4.68%***）。审稿人第一刀就是：
    # "那只是多跑了算力"。等算力做法 = 把 t=1 的臂跑满 2.22 倍代数（G 200→440）。
    # 注意：**必须落在独立的 lab 文件里**（--out logs/_mk10_eqc.json），
    # 否则会把它的前沿并进 ABA 的归一化盒，破坏已冻结的盒指纹。
    "T10_G440":         ("fixed", dict(fixed_T=10, _max_gen=440)),
    "RVNSonly_G440":    ("rvns",  dict(fixed_T=10, rvns_ls_trials=1, _max_gen=440)),
    "RandVNS_G440":     ("randvns", dict(fixed_T=10, rvns_mode="random",
                                         rvns_ls_trials=1, _max_gen=440)),
    # G440 那一批是**过冲**：RVNSonly_G440 43.2s vs RVNSonly_t3 28.6s = 1.51×，
    # 不是时间匹配，测不出"ls_trials 是不是纯算力效应"。
    # 下面两个才是**真正时间匹配**的对照：
    #   RVNSonly_G290 ≈ 19.7s × 290/200 ≈ 28.6s  -> 对齐 RVNSonly_t3
    #   T50_G440                                       -> 看 T 维度在长代数下是否还分层
    "RVNSonly_G290":    ("rvns",  dict(fixed_T=10, rvns_ls_trials=1, _max_gen=290)),
    "T50_G440":         ("fixed", dict(fixed_T=50, _max_gen=440)),
    # 更严的公平性口径：按**邻域求值次数**而非墙钟对齐。
    # 实测每代每解上限 3 时实际约用 2.93 次，故 ls_trials=3 @ G=200 做了
    # ≈200×100×2.93 = 58600 次邻域求值；ls_trials=1 要跑到 G=586 才有同样次数。
    # 若 HV(t1@586) 也打平 t3@200，则"尝试次数"这个杠杆按两个口径都站不住。
    "RVNSonly_G586":    ("rvns",  dict(fixed_T=10, rvns_ls_trials=1, _max_gen=586)),

    # ── 初始化变体扫描（I_ 前缀）──
    # 动机：消融阶梯里 **MIX3 是最大的单一组件（+14.46%***）**，却**从未做过变体
    # 扫描**。而且它的三条分支 random / LS / GW **全部把 OS 随机打乱**，只在机器
    # 选择（MA 维度）做文章 —— 工序顺序这一维在初始化阶段完全没被利用。
    # 这里其余组件与 `RVNSonly` 完全一致（fixed_T=10 + RVNS + ls_trials=1），
    # 只换 init_variant，因此任何差异都只能归因于初始化。
    # `I_mix3` 与 `RVNSonly` 配置完全相同，是内建的一致性检查（须逐位相同）。
    "I_mix3":     ("rvns", dict(fixed_T=10, rvns_ls_trials=1,
                                init_variant="mix3")),
    "I_spt":      ("rvns", dict(fixed_T=10, rvns_ls_trials=1,
                                init_variant="mix3_spt")),
    "I_mwr":      ("rvns", dict(fixed_T=10, rvns_ls_trials=1,
                                init_variant="mix3_mwr")),
    "I_gw_spt":   ("rvns", dict(fixed_T=10, rvns_ls_trials=1,
                                init_variant="mix3_gw_spt")),
    "I_half_r":   ("rvns", dict(fixed_T=10, rvns_ls_trials=1,
                                init_variant="half_random")),
    "I_no_r":     ("rvns", dict(fixed_T=10, rvns_ls_trials=1,
                                init_variant="no_random")),
    "I_rand":     ("rvns", dict(fixed_T=10, rvns_ls_trials=1,
                                init_variant="random")),
}


def resolve_grid(kwargs, n_pop, max_gen):
    """抽出逐臂覆盖 (n_pop, max_gen) 的逻辑，返回 (extra, n_pop, max_gen)。

    等算力臂需要跑更多代（如 T10@G=440 对齐 t3 的墙钟）。用 `_max_gen` / `_n_pop`
    前缀是为了不与 ``RMOEAD`` 的显式位置参数撞名（撞名会 TypeError）。
    返回的 ``extra`` 里**必须**已经没有这两个键——否则会当成未知 kwargs 传下去。
    """
    extra = dict(kwargs)
    if "_max_gen" in extra:
        max_gen = int(extra.pop("_max_gen"))
    if "_n_pop" in extra:
        n_pop = int(extra.pop("_n_pop"))
    return extra, n_pop, max_gen


def _run_one(job):
    instance, n_pop, max_gen, data_dir, label, kind, kwargs, seed = job
    from rmoea_d.algorithm import RMOEAD

    extra, n_pop, max_gen = resolve_grid(kwargs, n_pop, max_gen)
    if kind in ("full", "rvns", "randvns"):
        # 含局部搜索的臂：Q-PAS 有无由 kwargs 决定
        extra["enable_rvns"] = True
        extra.setdefault("rvns_ls_trials", 1)
    else:
        extra["enable_rvns"] = False

    solver = RMOEAD(instance_name=instance, n_pop=n_pop, max_gen=max_gen,
                    seed=seed, data_dir=data_dir, **extra)
    t0 = time.perf_counter()
    res = solver.solve()
    dt = time.perf_counter() - t0

    hist_T = {}
    for h in solver.history:
        hist_T[int(h["T"])] = hist_T.get(int(h["T"]), 0) + 1

    return dict(
        label=label, seed=seed, instance=instance,
        final_hv=float(res["final_hv"]), total_time=float(dt), hist_T=hist_T,
        q_table=solver.ql.q_table.tolist() if solver.ql else None,
        actions=list(solver.ql.actions) if solver.ql else None,
        # ABA：预算分配统计（mean_planned_trials 用于核对"算力是否真相等"）
        rvns_budget=(solver.rvns.get_budget_stats() if solver.rvns else None),
        final_pf=[[float(p["Makespan"]), float(p["Workload"])] for p in res["final_pf"]],
    )


def _load(path):
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            return []
    return []


def _dump(rows, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default="Mk10")
    ap.add_argument("--arms", required=True, help="逗号分隔的臂名（见 ARM_DEF）")
    ap.add_argument("--n_pop", type=int, default=100)
    ap.add_argument("--max_gen", type=int, default=200)
    ap.add_argument("--seed_start", type=int, default=42)
    ap.add_argument("--n_runs", type=int, default=30)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None,
                    help="结果 JSON 路径（默认 logs/_<instance>_lab.json）")
    args = ap.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_runs))
    out = args.out or os.path.join(ROOT, "logs", f"_{args.instance.lower()}_lab.json")
    partial = out.replace(".json", ".partial.json")

    labels = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in labels:
        if a not in ARM_DEF:
            raise SystemExit(f"unknown arm: {a}\nknown: {sorted(ARM_DEF)}")

    rows = _load(out) + _load(partial)
    done = {(r["label"], r["seed"]) for r in rows}
    jobs = [(args.instance, args.n_pop, args.max_gen, args.data_dir,
             a, ARM_DEF[a][0], ARM_DEF[a][1], s)
            for a in labels for s in seeds if (a, s) not in done]

    print(f"instance={args.instance} arms={labels} todo={len(jobs)} "
          f"already_done={len(done)} workers={args.workers}", flush=True)
    if not jobs:
        print("nothing to do", flush=True)
        return

    t0 = time.perf_counter()
    with mp.Pool(args.workers, maxtasksperchild=2) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_one, jobs, chunksize=1), 1):
            rows.append(r)
            if i % 10 == 0 or i == len(jobs):
                _dump(rows, partial)
                el = time.perf_counter() - t0
                print(f"  [{i}/{len(jobs)}] elapsed={el:.0f}s "
                      f"eta={el / i * (len(jobs) - i):.0f}s  "
                      f"last={r['label']}/s{r['seed']} hv={r['final_hv']:.5f}", flush=True)

    all_done = {(r["label"], r["seed"]) for r in rows}
    # 目标集 = **本文件已有的臂 ∪ 本次请求的臂**，不是整个 ARM_DEF。
    # 否则用 --out 跑到独立 lab（如等算力臂）时永远判不出"完成"。
    want = {r["label"] for r in rows} | set(labels)
    target = {(a, s) for a in want for s in seeds}
    if all_done >= target:
        _dump(rows, out)
        if os.path.exists(partial):
            os.remove(partial)
        print("ALL ARMS COMPLETE ->", out, flush=True)
    else:
        _dump(rows, partial)
        missing = sorted({a for a, _ in (target - all_done)})
        print(f"partial saved -> {partial}  remaining arms={missing}", flush=True)
    print(f"batch done in {time.perf_counter() - t0:.0f}s  rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
