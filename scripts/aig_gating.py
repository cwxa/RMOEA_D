#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自适应初始化门控（AIG）：初始化的收益能不能**事前**预测？

背景
----
派工式 OS 初始化 `I_mwr`（`init_variant="mix3_mwr"`）在 Mk10 开发集上
**+8.64%（30/30，p=1.9e-09，dz=+2.04）**，但在留出集 Mk07 / Mk09 上只剩
+0.34% / +0.30%（n.s.）。所以「要不要用 MWR」不能靠固定开关。

**门控判据必须事前可得** —— 只能用实例自身的结构特征，或用一次固定预算的廉价探针。
不能用「已经知道 MWR 效果好」这种事后信息，那是循环论证。

两个目标量（**必须分清，混用是本脚本最大的坑**）
----------------------------------------------
同一批数据可以问两个不同的问题，它们的答案差一个数量级：

  A `I_mwr vs I_rand`   相对**纯随机起点**（= 论文 D1）。这测的是
                        「初始化这件事值多少」—— 任何好初始化都会赢，
                        **不能**用来决定"该不该用 MWR"。
  B `I_mwr vs I_mix3`   相对**论文口径 MIX3**。这才是「OS-MWR 相对论文的净增量」，
                        **决定该不该用**的就是它。

早期版本只算了 A，于是"门控"预测的其实是"这个实例有多吃初始化"，
而不是"MWR 在这个实例上有没有增量" —— 目标错位。现在两个都算，并显式区分。

口径
----
* **逐实例独立**，不跨实例合并（各实例 HV 归一化盒的尺度不同）；
* 主口径 = **实例边界口径**，直接读 lab 行里的 `final_hv`
  （边界由实例数据确定性推出，**与臂集无关**，可跨批次比较，见 `docs/new-arch-report.md` §1）；
  盒口径只用来量化「放大倍数」，**不作为结论口径**（缺陷 22）。
* 结论只用相对差与**秩相关**（10 个实例的样本量很小，ρ 与 p 都要报）。

Q1 的事后性（必须写进结论，别当门控）
------------------------------------
`ms_lever% / wl_lever%` 是从 `final_pf`（**末代存档前沿**）算出来的，不是初始前沿 ——
要得到它必须先跑完两个臂，也就是说**要预测的量已经在手里了**。
所以它是**机制/事后**量（回答"初始化靠什么传导到 HV"），**不能**当成实例级门控判据。
真正**事前可得**的只有结构特征（`total_ops / n_machines / n_jobs / flex_ratio / pt_cv /
load_ratio`）与一次固定预算的廉价探针。

多重比较（Q2 的真正风险）
------------------------
Q2 是「在 8 个候选预测变量里挑 ρ 最大的那个」。n=10、8 个变量，
单变量 p 值（0.0016 / 0.0034）**不能**直接当结论 —— 必须给
**max|ρ| 的置换零假设**（打乱实例↔目标量的对应，保留预测变量之间的相关结构），
用它当基准线。默认 20000 次置换，输出家族错误率校正后的 p。

留一稳健性
----------
n=10 的相关性最典型的失败模式是"全靠一个点撑着"（比如 Mk10 既最大、收益又最高），
所以每个目标量都要报 leave-one-out 的 ρ 区间。

用法：
    python scripts/aig_gating.py
    python scripts/aig_gating.py --labs logs/_mk01_init.json,logs/_mk10_init.json
    python scripts/aig_gating.py --targets I_rand,I_mix3 --perm 20000 --perm_seed 42
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import init_variant_analysis as iva  # noqa: E402

TREAT = "I_mwr"

# 两个目标量的配对基准（顺序即报告顺序）
BASELINES = {
    "I_rand": "A 相对纯随机起点（= 论文 D1）—— 「有好起点值多少」",
    "I_mix3": "B 相对论文口径 MIX3 —— 「OS-MWR 的净增量」，决定该不该用的就是它",
}

# **事前可得**的候选量：只用实例自身结构，不看任何 run 的结果。
# 其余候选（ms_lever_pct / wl_lever_pct）由 `final_pf` 算出 —— 事后量，只能当机制解释。
EX_ANTE = ["n_jobs", "n_machines", "total_ops", "flex_ratio", "pt_cv", "load_ratio"]

KEYS = ["ms_lever_pct", "wl_lever_pct", "n_jobs", "n_machines",
        "total_ops", "flex_ratio", "pt_cv", "load_ratio"]


def instance_features(name, data_dir):
    """与算法无关的实例结构特征（全部事前可得）。"""
    from rmoea_d.core.instance import load_instance
    inst = load_instance(name, data_dir)
    n_ma = inst["n_machines"]
    flex = np.mean([len(vm) for job in inst["valid_machines"] for vm in job])
    # crisp_times[job][op] 是"该工序在各可选机器上的加工时间"，可能含 None（该机器不可用）
    times = [t for job in inst["crisp_times"] for op in job for t in op if t is not None]
    tmean = float(np.mean(times)) if times else float("nan")
    tcv = float(np.std(times) / tmean) if times and tmean else float("nan")
    return {
        "n_jobs": int(inst["n_jobs"]),
        "n_machines": int(n_ma),
        "total_ops": int(inst["total_ops"]),
        "flex_ratio": float(flex / n_ma),                 # 平均可选机器数 / 机器总数
        "pt_cv": tcv,                                     # 加工时间离散度
        "load_ratio": float(inst["total_ops"] / n_ma),    # 每台机器平均工序数
    }


def _spearman_cols(X, y):
    """对 X 的每一列与 y 求 Spearman ρ（= 秩上的 Pearson），只做一次秩变换。"""
    xr = stats.rankdata(X, axis=0).astype(float)
    yr = stats.rankdata(y).astype(float)
    xc = xr - xr.mean(axis=0, keepdims=True)
    yc = yr - yr.mean()
    num = xc.T @ yc
    den = np.linalg.norm(xc, axis=0) * np.linalg.norm(yc)
    return num / np.where(den == 0, np.nan, den)


def permutation_max_rho(X, y, n_perm, seed=42):
    """max|ρ| 的置换零假设（家族错误率基准线）。

    零假设 = 「预测变量与 ΔHV 之间没有关联」。做法：固定预测变量的秩矩阵，
    **只打乱实例↔目标量的对应**（等价于置换 y 的秩），每次重算全部候选的 ρ 并取
    max|ρ|。因为预测变量之间的相关结构被完整保留，得到的 null 分布自动包含
    「候选彼此不独立」这件事 —— 这正是单变量 p 值漏掉的部分。

    返回 dict：观测 max|ρ| 及其变量名、null_p95 / null_p99 / mc_p（MC 校正 p）。
    退化输入（候选列数为 0、或 y 少于 3 个点、或全部 ρ 都是 nan）返回
    `skipped` 标记 + 全 None，**不抛异常** —— 缺陷 27：原先会对空数组
    `np.nanargmax` 直接崩，害得调用方（`init_probe.py` 小规模自测）无法运行。
    """
    xr = stats.rankdata(X, axis=0).astype(float)
    xc = xr - xr.mean(axis=0, keepdims=True)
    xn = np.linalg.norm(xc, axis=0)
    yr = stats.rankdata(y).astype(float)

    rho_obs = _spearman_cols(X, y)
    if rho_obs.size == 0 or y.size < 3 or np.all(np.isnan(rho_obs)):
        return {
            "n_perm": 0, "seed": int(seed), "skipped": "候选为空 / 样本数 < 3",
            "obs_max_abs_rho": None, "obs_argmax": None, "rho_obs": [],
            "null_mean": None, "null_p95": None, "null_p99": None,
            "mc_p_fwer": float("nan"),
        }
    j_obs = int(np.nanargmax(np.abs(rho_obs)))

    rng = np.random.default_rng(seed)
    B = max(0, int(n_perm))
    if B == 0:
        # 显式契约：不置换就把 null 统计标成 None（而不是在空数组上 percentile 崩掉）
        return {
            "n_perm": 0, "seed": int(seed),
            "obs_max_abs_rho": float(abs(rho_obs[j_obs])), "obs_argmax": j_obs,
            "rho_obs": [float(v) for v in rho_obs],
            "null_mean": None, "null_p95": None, "null_p99": None,
            "mc_p_fwer": float("nan"),
        }
    # 批量置换：B × n 的置换索引 -> B 组 y 秩
    perms = np.argsort(rng.random((B, y.size)), axis=1)
    ypb = yr[perms]                                  # B × n
    ypc = ypb - ypb.mean(axis=1, keepdims=True)
    yn = np.linalg.norm(ypc, axis=1)                 # B
    # (k × n) @ (n × B) -> k × B
    num = xc.T @ ypc.T
    den = xn[:, None] * yn[None, :]
    rho_null = num / np.where(den == 0, np.nan, den)
    max_null = np.nanmax(np.abs(rho_null), axis=0)   # B

    obs_abs = abs(rho_obs[j_obs])
    mc_p = float((1 + int(np.sum(max_null >= obs_abs))) / (1 + B))
    return {
        "n_perm": B, "seed": int(seed),
        "obs_max_abs_rho": float(obs_abs), "obs_argmax": j_obs,
        "rho_obs": [float(v) for v in rho_obs],
        "null_mean": float(np.mean(max_null)),
        "null_p95": float(np.percentile(max_null, 95)),
        "null_p99": float(np.percentile(max_null, 99)),
        "mc_p_fwer": mc_p,
    }


def leave_one_out(X, y, names):
    """留一稳健性：每次去掉一个实例，重算各候选的 ρ。

    n=10 的相关性最典型的失败模式是「全靠一个点撑着」——比如 Mk10 既最大
    （total_ops 最大）又是收益最高的那个，去掉它 ρ 就塌。
    返回 {名字: {...}}，含全量 ρ、留一区间、以及"最不利"的实例下标。
    """
    n = y.size
    res = {}
    for j, nm in enumerate(names):
        vals = []
        for i in range(n):
            m = np.ones(n, bool)
            m[i] = False
            vals.append(float(stats.spearmanr(X[m, j], y[m])[0]))
        vals = np.array(vals)
        k = int(np.argmin(np.abs(vals)))
        res[nm] = {"min_rho": float(vals.min()), "max_rho": float(vals.max()),
                   "min_abs_rho": float(np.abs(vals).min()),
                   "worst_drop_idx": k,
                   "rho_all": float(stats.spearmanr(X[:, j], y)[0])}
    return res


def build_recs(per, base, data_dir, verbose=True):
    """逐实例计算「`I_mwr` vs `base`」。

    只保留同时存在 `I_mwr` 与 `base` 的实例（缺则跳过并显式报出，不静默丢）。
    HV 一律取 lab 行里的 `final_hv`（实例边界口径）；盒口径只用来算放大倍数。
    """
    recs, skipped = [], []
    for inst, rs in per.items():
        kept, _dropped = iva.split_context(rs, keep=[TREAT, base])
        hv_inst = collections.defaultdict(dict)
        for r in kept:
            if r.get("final_hv") is not None:
                hv_inst[r["label"]][r["seed"]] = float(r["final_hv"])
        if TREAT not in hv_inst or base not in hv_inst:
            skipped.append(inst)
            continue
        hv_box, box = iva.hv_table(kept)
        pr = iva.paired(hv_inst, TREAT, base)
        pr_box = iva.paired(hv_box, TREAT, base)
        if pr is None:
            skipped.append(inst)
            continue

        # 原目标杠杆 λ：用原始目标值（与 HV 盒无关，可跨实例比对）
        seeds = sorted(set(hv_inst[base]) & set(hv_inst[TREAT]))
        ms_b, ms_t, wl_b, wl_t = [], [], [], []
        for s in seeds:
            rb = next(r for r in kept if r["label"] == base and r["seed"] == s)
            rt = next(r for r in kept if r["label"] == TREAT and r["seed"] == s)
            ms_b.append(min(p[0] for p in rb["final_pf"]))
            ms_t.append(min(p[0] for p in rt["final_pf"]))
            wl_b.append(min(p[1] for p in rb["final_pf"]))
            wl_t.append(min(p[1] for p in rt["final_pf"]))
        lam_ms = (np.mean(ms_b) - np.mean(ms_t)) / np.mean(ms_b) * 100.0
        lam_wl = (np.mean(wl_b) - np.mean(wl_t)) / np.mean(wl_b) * 100.0

        try:
            feat = instance_features(inst, data_dir)
        except Exception as e:                                    # noqa: BLE001
            if verbose:
                print("  ⚠ %s 特征提取失败: %s" % (inst, e))
            feat = {}

        amp = ((pr_box["rel"] / pr["rel"])
               if pr_box and abs(pr["rel"]) > 1e-12 else float("nan"))
        rec = {"instance": inst, "base": base, "n": pr["n"],
               "ms_lever_pct": float(lam_ms), "wl_lever_pct": float(lam_wl),
               "dhv_rel_pct": float(pr["rel"]), "dhv": float(pr["d"]),
               "wins": pr["wins"], "p": float(pr["p"]), "dz": float(pr["dz"]),
               "hv_base": float(pr["mean_b"]), "hv_mwr": float(pr["mean_a"]),
               "dhv_rel_box_pct": (float(pr_box["rel"]) if pr_box else None),
               "amplification": (float(amp) if np.isfinite(amp) else None),
               "box": [list(map(float, box[0])), list(map(float, box[1]))]}
        rec.update(feat)
        recs.append(rec)
        if verbose:
            print("%-6s %-6s %-5s %-6.3f %-7.3f %-9.2f %-9.2f %-9s %-8s %-8s" %
                  (inst, feat.get("n_jobs", "-"), feat.get("n_machines", "-"),
                   feat.get("flex_ratio", float("nan")),
                   feat.get("pt_cv", float("nan")),
                   lam_ms, pr["rel"],
                   ("%.1f" % amp) if np.isfinite(amp) else "n/a",
                   "%d/%d" % (pr["wins"], pr["n"]), iva.stars(pr["p"])))
    return recs, skipped


def analyze(recs, perm_n, perm_seed, keys=KEYS):
    """秩相关 + max|ρ| 置换零假设 + 留一。返回 (corr, perm_payload, loo_payload)。"""
    y = np.array([r["dhv_rel_pct"] for r in recs], float)
    corr = {}
    for k in keys:
        x = np.array([r.get(k, np.nan) for r in recs], dtype=float)
        ok = ~np.isnan(x)
        if ok.sum() < 4 or np.allclose(x[ok], x[ok][0]):
            continue
        rho, p = stats.spearmanr(x[ok], y[ok])
        pear, _ = stats.pearsonr(x[ok], y[ok])
        corr[k] = {"spearman_rho": float(rho), "spearman_p": float(p),
                   "pearson_r": float(pear), "n": int(ok.sum())}
        print("  %-14s rho=%+.3f  p=%.4f   (pearson r=%+.3f, n=%d)"
              % (k, rho, p, pear, ok.sum()))

    perm_payload = None
    if perm_n and perm_n > 0 and len(recs) >= 4:
        cand = [k for k in keys if all(r.get(k) is not None for r in recs)]
        cand = [k for k in cand
                if not np.allclose([r[k] for r in recs], recs[0][k])]
        print("\n  置换零假设 max|rho|（打乱实例↔目标量，保留候选间相关结构）")
        print("  候选 %d 个：%s" % (len(cand), ", ".join(cand)))
        print("  " + "-" * 84)
        perm_payload = {}
        for fam, sub in (("all", cand),
                         ("ex_ante", [k for k in cand if k in EX_ANTE])):
            if len(sub) < 2:
                continue
            X = np.column_stack([[r[k] for r in recs] for k in sub])
            res = permutation_max_rho(X, y, perm_n, seed=perm_seed)
            res["keys"] = sub
            res["obs_argmax_key"] = sub[res["obs_argmax"]]
            perm_payload[fam] = res
            print("  [%-7s] %d 个变量  max|rho|=%+.3f (%s)  null95=%.3f  null99=%.3f"
                  "  -> 校正 p = %.4f %s"
                  % (fam, len(sub), res["obs_max_abs_rho"], res["obs_argmax_key"],
                     res["null_p95"], res["null_p99"], res["mc_p_fwer"],
                     iva.stars(res["mc_p_fwer"])))

    loo_payload = None
    if len(recs) >= 5:
        names = [r["instance"] for r in recs]
        cand = [k for k in keys if all(r.get(k) is not None for r in recs)]
        cand = [k for k in cand
                if not np.allclose([r[k] for r in recs], recs[0][k])]
        X = np.column_stack([[r[k] for r in recs] for k in cand])
        loo = leave_one_out(X, y, cand)
        loo_payload = {"per_candidate": loo, "instances": names}
        print("\n  留一稳健性（每次去掉 1 个实例，重算 rho）—— 按 |rho| 降序取前 4")
        order = sorted(cand, key=lambda k: -abs(loo[k]["rho_all"]))
        for k in order[:4]:
            d = loo[k]
            print("    %-14s rho(全)=%+.3f  留一区间 [%+.3f, %+.3f]  最不利: %s%s"
                  % (k, d["rho_all"], d["min_rho"], d["max_rho"],
                     names[d["worst_drop_idx"]],
                     "  <- 事前可得" if k in EX_ANTE else "  (事后机制量)"))
    return corr, perm_payload, loo_payload


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labs", default="", help="逗号分隔；默认 Mk01~Mk10 的 _mkNN_init.json")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "aig_gating.json"))
    ap.add_argument("--targets", default="I_rand,I_mix3",
                    help="配对基准（逗号分隔，见 BASELINES）；默认两个都算")
    ap.add_argument("--perm", type=int, default=20000,
                    help="max|rho| 置换零假设的置换次数（0 = 跳过）")
    ap.add_argument("--perm_seed", type=int, default=42)
    args = ap.parse_args()

    labs = ([s.strip() for s in args.labs.split(",") if s.strip()] if args.labs
            else [os.path.join(ROOT, "logs", "_mk%02d_init.json" % i) for i in range(1, 11)])
    labs = [p for p in labs if os.path.exists(p)]
    if not labs:
        print("没有找到任何 _mkNN_init.json")
        return 1

    targets = [s.strip() for s in args.targets.split(",") if s.strip()]
    for t in targets:
        if t not in BASELINES:
            raise SystemExit("未知目标基准 %r，可选：%s" % (t, ", ".join(BASELINES)))

    per, _rows = iva.load_by_instance(labs)
    print("=" * 96)
    print("AIG 门控可预测性检验   实例 %d 个：%s" % (len(per), ", ".join(per)))
    print("=" * 96)

    out = {"labs": [os.path.basename(p) for p in labs],
           "treat": TREAT, "caliber": "instance-boundary (final_hv)",
           "targets": {}}

    for base in targets:
        print("\n" + "#" * 96)
        print("# 目标量 %s" % BASELINES[base])
        print("# 比较 = %s vs %s（实例边界口径 final_hv）" % (TREAT, base))
        print("#" * 96)
        print("%-6s %-6s %-5s %-6s %-7s %-9s %-9s %-9s %-8s %-8s" %
              ("inst", "n_job", "n_ma", "flex", "pt_cv",
               "ms_lever%", "dHV_rel%", "放大×", "wins", "p"))
        print("-" * 96)
        recs, skipped = build_recs(per, base, args.data_dir)
        if skipped:
            print("  [i] 缺 %s 或 %s 的实例已跳过：%s"
                  % (TREAT, base, ", ".join(skipped)))
        if len(recs) < 4:
            print("  [!] 只有 %d 个实例可用，不足以做秩相关/置换，仅记录逐实例表"
                  % len(recs))
        print("\n  秩相关（n=%d 个实例）：预测目标 = dHV_rel%%" % len(recs))
        print("  " + "-" * 84)
        corr, perm, loo = analyze(recs, args.perm, args.perm_seed)
        out["targets"]["vs_%s" % base] = {"base": base, "per_instance": recs,
                                          "skipped": skipped,
                                          "correlations": corr,
                                          "permutation": perm,
                                          "leave_one_out": loo}
        # 向后兼容：顶层 per_instance / correlations = 目标 A（I_rand）
        if base == "I_rand":
            out["per_instance"] = recs
            out["correlations"] = corr

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\n已写出 %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
