# -*- coding: utf-8 -*-
"""算力-算法响应面：在**等墙钟**下比较不同算法，而不是只在等代数下比。

动机
----
项目原结论「HV 是墙钟的单一函数」（Spearman=+1.000）只在 `RVNSonly` **一个家族**的
5 个点上验证过 —— 那是「同一算法不同预算」，**不是**「不同算法同一预算」。
后者从未测过，而它才是「算法有没有用」的判定口径。本脚本用 `anytime_run.py`
落盘的长 run（`hist_hv` + `hist_time`）补上这一格。

两个口径必须分清
----------------
* **等代数**：都截断到第 G 代 —— 对每代更快的算法（如无局部搜索的 D1）有利；
* **等墙钟**：都截断到同一累计秒数 —— 这才是算力公平的口径。
两者结论不一致时**以等墙钟为准**，并把差异显式报出来。

插值
----
hv(t) 由 `np.interp(t, init_time + cumsum(gen_time), hist_hv)` 线性插值。
累计时间**必须含初始化开销** `init_time`，否则首个网格点会落在「第 1 代之前」，
被静默插值成 hv[0]，把初始化阶段的差异抹掉。

用法：
    python scripts/response_surface.py --labs logs/anytime_g2000.json
    python scripts/response_surface.py --labs logs/anytime_g2000.json \\
        --pairs RMOEAD=D5,RMOEAD=D1,D5=D1 --wall_fracs 0.1,0.25,0.5,1.0

    # 多份 lab 合并（同一 (实例,臂,seed) 取轨迹更长的那条）：
    # 用于把 G=2000 与 G=4000 两次长跑拼起来，直接回答"再多给一倍算力会怎样"
    python scripts/response_surface.py \\
        --labs logs/anytime_g2000.json,logs/anytime_g4000.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_PAIRS = [("RMOEAD", "D5"), ("D5", "D1"), ("RMOEAD", "D1")]


def load(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("rows", []) if isinstance(data, dict) else data


def load_many(spec):
    """`--labs a.json,b.json`：多份 lab 合并，**同一 (实例,臂,seed) 取轨迹更长的那条**。

    用途：把 G=2000 与 G=4000 两次长跑拼起来（后者只跑了部分臂），
    这样"多给一倍算力会怎样"可以直接算。
    取"更长"而不是"后写覆盖"→ 结果与文件顺序无关，不会因命令行顺序不同而出两套数字。
    """
    paths = [s.strip() for s in spec.split(",") if s.strip()]
    best = {}
    for p in paths:
        for r in load(p):
            key = (r["instance"], r["label"], r["seed"])
            n = len(r.get("hist_hv") or [])
            if key not in best or n > best[key][0]:
                best[key] = (n, r)
    return [v[1] for v in best.values()]


def index_curves(rows):
    """(instance, label) -> {seed: (t_total, hv)}，t_total 含初始化开销。"""
    idx = {}
    for r in rows:
        if not r.get("hist_hv") or not r.get("hist_time"):
            continue
        t = np.asarray(r["hist_time"], dtype=float) + float(r.get("init_time", 0.0))
        h = np.asarray(r["hist_hv"], dtype=float)
        idx.setdefault((r["instance"], r["label"]), {})[r["seed"]] = (t, h)
    return idx


def fe_multiplier(label):
    """该臂**每代求值次数 / n_pop** —— 机器无关的算力单元。

    每代 `moead_generation` 评估 n_pop 个子代；`enable_rvns` 的臂额外为每个解做
    `rvns_ls_trials` 次邻域尝试（论文口径 =1，`budget_mode="fixed"` 时
    `plan_generation` 对每个解都发满档）。故：

        RVNS 臂     : 1 + ls_trials = 2 × n_pop / 代
        非 RVNS 臂  : 1             = 1 × n_pop / 代

    这条**只依赖臂定义、不依赖机器**，因此是「等算力」比较的正确口径。
    墙钟做不到这一点：同一台机器上不同臂的 Python 层常数因子不同，且并行实验台里
    各臂批次所处的系统负载不同（见 docs/new-arch-report.md §6）。
    """
    from ablation_ladder import LADDER_DICT
    cfg = LADDER_DICT.get(label, {})
    if cfg.get("enable_rvns"):
        return 1.0 + float(cfg.get("rvns_ls_trials", 1))
    return 1.0


def per_instance_tmax(idx):
    """每个实例的公共墙钟上限 = 该实例所有 (臂, seed) 中最小的终点时刻。

    **必须逐实例独立。** 取全局最小会让大实例（Mk10，数百秒）只在自身
    预算的前十几个百分点处被比较，把「等墙钟」降级成「等一个很小的墙钟」——
    与 §1.1「逐实例独立」原则冲突，属缺陷 22 同一家族（跨实例共用一个标量边界）。
    """
    out = {}
    for inst, lab in {k for k in idx}:
        tt = [t[-1] for t, _ in idx[(inst, lab)].values()]
        if tt:
            out[inst] = min(out.get(inst, float("inf")), float(min(tt)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labs", default=os.path.join(ROOT, "logs", "anytime_g2000.json"))
    ap.add_argument("--pairs", default="")
    ap.add_argument("--segments", type=int, default=8, help="饱和诊断的等代数段数")
    ap.add_argument("--wall_fracs", default="0.1,0.25,0.5,0.75,1.0",
                    help="等墙钟网格 = 公共时间上限 × 这些比例")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pairs = DEFAULT_PAIRS
    if args.pairs.strip():
        pairs = [tuple(s.split("=")) for s in args.pairs.split(",") if "=" in s]
    fracs = [float(x) for x in args.wall_fracs.split(",")]

    rows = load_many(args.labs)
    idx = index_curves(rows)
    if not idx:
        print("没有可用轨迹（需要 hist_hv + hist_time）。")
        return 1

    instances = sorted({k[0] for k in idx})
    labels = sorted({k[1] for k in idx})
    n_runs = sum(len(d) for d in idx.values())
    gmax = max(h.size for d in idx.values() for _, h in d.values())

    print("=" * 78)
    print("算力-算法响应面  labs=%s"
          % ", ".join(os.path.basename(s.strip())
                      for s in args.labs.split(",") if s.strip()))
    print("runs=%d  实例=%s  臂=%s  最长=%d 代" % (n_runs, instances, labels, gmax))
    print("=" * 78)

    out = {"labs": [os.path.basename(s.strip())
                    for s in args.labs.split(",") if s.strip()],
           "instances": instances,
           "labels": labels, "saturation": {}, "wall": {}, "alg": {}, "fe": {},
           "verdict": {}}

    # ── 1) 饱和诊断：把代数等分 K 段，看每段的 HV 增量是否趋近 0 ──
    print("\n【1】饱和诊断：每段（等代数）的 HV 平均增量。末段仍明显 >0 = 未饱和")
    for inst in instances:
        print("  -- %s" % inst)
        for lab in labels:
            d = idx.get((inst, lab))
            if not d:
                continue
            gains = []
            for _t, h in d.values():
                edges = np.linspace(0, h.size - 1, args.segments + 1).astype(int)
                gains.append(np.diff(h[edges]))
            Gm = np.mean(gains, axis=0)
            sfx = np.mean([np.std(g, ddof=1) for g in gains])
            first, last = Gm[0], Gm[-1]
            ratio = last / first if abs(first) > 1e-12 else float("nan")
            out["saturation"].setdefault(inst, {})[lab] = {
                "gains": [float(x) for x in Gm], "ratio_last_first": float(ratio)}
            print("     %-9s " % lab
                  + " ".join("%+.5f" % x for x in Gm)
                  + "   末/首=%.3f" % ratio)

    # ── 2) 等墙钟：**逐实例**公共时间上限内插值后配对 ──
    # 公共上限必须逐实例独立求。Mk07 的 G2000 只要 ~60s 而 Mk10 要数百秒，
    # 若取全局最小，Mk10 就只在自身预算的前十几个百分点处被比较 ——
    # 等于把「等墙钟」悄悄降级成「等一个很小的墙钟」，与 §1.1 的逐实例独立
    # 原则冲突（缺陷 22 同一家族：跨实例混用一个标量边界）。
    tmax_by = per_instance_tmax(idx)
    print("\n【2】等墙钟配对比较（**逐实例**公共上限，网格 = 各实例上限 × %s）" % fracs)
    print("     " + "   ".join("%s=%.1fs" % (i, tmax_by[i]) for i in instances))
    print("     diff>0 表示左侧臂在**同一时刻**的 HV 更高")
    for a, b in pairs:
        cells, allo = [], []
        for inst in instances:
            if (inst, a) not in idx or (inst, b) not in idx or inst not in tmax_by:
                continue
            da, db = idx[(inst, a)], idx[(inst, b)]
            seeds = sorted(set(da) & set(db))
            if not seeds:
                print("     ⚠ %s %s vs %s: seed 交集为空" % (inst, a, b))
                continue
            T_inst = tmax_by[inst]
            ds = []
            for f in fracs:
                T = T_inst * f
                vals = [np.interp(T, da[s][0], da[s][1]) - np.interp(T, db[s][0], db[s][1])
                        for s in seeds if T <= da[s][0][-1] and T <= db[s][0][-1]]
                ds.append(np.asarray(vals))
            if any(v.size == 0 for v in ds):
                continue
            allo.append((inst, ds))
            cells.append((inst, " ".join("%+.4f" % v.mean() for v in ds), T_inst,
                          min(v.size for v in ds)))
        for inst, ds in allo:
            T_inst = tmax_by[inst]
            rec = []
            for f, v in zip(fracs, ds):
                _, w, p, dz = _pstat(v)
                rec.append({"T": float(T_inst * f), "wall_frac": float(f),
                            "tmax": float(T_inst), "n": int(v.size),
                            "diff": float(v.mean()), "wins": w, "p": p, "dz": dz})
            out["wall"].setdefault("%s|%s" % (a, b), {})[inst] = rec
        print("  %11s vs %-11s" % (a, b))
        for inst, s, T_inst, nmin in cells:
            print("       %-5s %s   [上限 %.1fs, n>=%d]" % (inst, s, T_inst, nmin))

    # ── 3) 等代数对照（同一网格点数，按代数而非时间截断）──
    print("\n【3】等代数对照（同网格比例，但按代数截断）——与【2】对比即知口径影响")
    for a, b in pairs:
        for inst in instances:
            if (inst, a) not in idx or (inst, b) not in idx:
                continue
            da, db = idx[(inst, a)], idx[(inst, b)]
            seeds = sorted(set(da) & set(db))
            if not seeds:
                continue
            cells = []
            for f in fracs:
                vals = []
                for s in seeds:
                    ha, hb = da[s][1], db[s][1]
                    ia = max(0, min(ha.size - 1, int(round(f * (ha.size - 1)))))
                    ib = max(0, min(hb.size - 1, int(round(f * (hb.size - 1)))))
                    vals.append(ha[ia] - hb[ib])
                v = np.asarray(vals)
                cells.append("%+.4f" % v.mean())
            out["alg"].setdefault("%s|%s" % (a, b), {})[inst] = cells
            print("  %11s vs %-11s %-5s " % (a, b, inst) + " ".join("%-9s" % c for c in cells))

    # ── 4) 等求值次数（机器无关的算力口径）──
    # 墙钟受实现常数因子与并行负载影响（见 fe_multiplier 注释），求值次数不受。
    n_pop = int(rows[0].get("n_pop", 100)) if rows else 100
    fe_pg = {lab: n_pop * fe_multiplier(lab) for lab in labels}
    print("\n【4】等求值次数（FE）配对比较 —— 机器无关的算力口径")
    print("     每代求值： " + "  ".join("%s=%d" % (l, fe_pg[l]) for l in labels)
          + "   （n_pop=%d）" % n_pop)
    print("     ⚠ RVNS 臂每代拿 2× 算力 → 等代数比较对它们有利，等 FE 才公平")
    for inst in instances:
        labs_i = [l for l in labels if (inst, l) in idx]
        if not labs_i:
            continue
        gmax = {l: max(h.size for _, h in idx[(inst, l)].values()) for l in labs_i}
        fe_cap = min(gmax[l] * fe_pg[l] for l in labs_i)
        print("   -- %s  公共 FE 上限 = %d  （逐臂上限 %s）"
              % (inst, int(fe_cap), {l: int(gmax[l] * fe_pg[l]) for l in labs_i}))
        for a, b in pairs:
            if (inst, a) not in idx or (inst, b) not in idx:
                continue
            da, db = idx[(inst, a)], idx[(inst, b)]
            seeds = sorted(set(da) & set(db))
            if not seeds:
                continue
            cells = []
            rec = []
            for f in fracs:
                fe = fe_cap * f
                vals = []
                for s in seeds:
                    ha, hb = da[s][1], db[s][1]
                    ga = fe / fe_pg[a]
                    gb = fe / fe_pg[b]
                    if ga > ha.size or gb > hb.size:
                        continue
                    xa = np.arange(1, ha.size + 1, dtype=float)
                    xb = np.arange(1, hb.size + 1, dtype=float)
                    vals.append(float(np.interp(ga, xa, ha) - np.interp(gb, xb, hb)))
                v = np.asarray(vals)
                _, w, p, dz = _pstat(v)
                rec.append({"FE": float(fe), "fe_frac": float(f), "n": int(v.size),
                            "diff": float(v.mean()), "wins": w, "p": p, "dz": dz})
                cells.append("%+.4f" % v.mean())
            out["fe"].setdefault("%s|%s" % (a, b), {})[inst] = rec
            print("     %11s vs %-11s " % (a, b) + " ".join("%-9s" % c for c in cells))

    out_path = args.out or (args.labs.split(",")[0].strip() + ".surface.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    print("\n已写出 %s" % out_path)
    return 0


def _pstat(v):
    d = np.asarray(v, dtype=float)
    n = d.size
    if n == 0:
        return float("nan"), 0, float("nan"), float("nan")
    diff = float(d.mean())
    wins = int(np.sum(d > 0))
    if np.allclose(d, 0.0) or n < 3:
        return diff, wins, 1.0, 0.0
    try:
        p = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        p = 1.0
    sd = float(d.std(ddof=1))
    return diff, wins, p, (diff / sd if sd > 0 else 0.0)


if __name__ == "__main__":
    sys.exit(main())
