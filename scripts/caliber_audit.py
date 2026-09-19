# -*- coding: utf-8 -*-
"""口径审计：把「盒口径」与「实例边界口径」摆在一起，量化放大倍数。

背景
----
项目里同时存在两套 HV 归一化口径，混用是全部数字漂移的总根源：

* **实例边界口径** `instance_hv_bounds(instance)`（`src/rmoea_d/algorithm.py:257`）
  下界 / 上界由**实例数据**确定性推出（临界路径下界 … 全部工序最长时间之和），
  **与参与比较的臂集无关**。所有实验台落盘的 `final_hv` 与 `hist_hv` 用的都是它，
  因此它们**天然可跨批次、跨臂集直接比较**。

* **盒口径** `estimate_hv_bounds(fronts)`
  lo/hi **取参与比较的前沿的极值**，归一化后前沿贴满 [0,1]² 边界 → HV 整体偏大；
  又因为盒随"放进来的臂"变化，数字随臂集漂移（缺陷 14 / 18 的机制）。
  它同时**放大效应量**：同一批数据（Mk10，RVNSonly 家族，30 seeds）
  G200→G586 的 rel%，
      实例边界口径 **+0.77%**   vs   盒口径 **+11.95%**    —— 差 **15.5 倍**。

结论用法（**2026-09-19 修订：旧版这里说"两口径给同样的符号 / wins / p"，已被缺陷 25 证伪**）
------------------------------------------------------------------------------------------------
* 报「某组件有没有用」：**也必须只用实例边界口径**。本脚本的输出里，
  Mk10 的 `D4|D3` 实例边界 wins=17/30, p=0.919 → 盒口径 wins=13/30, p=0.490；
  `D3|D2` 实例边界 wins=27/30, p=1.64e-07 → 盒口径 wins=26/30, p=6.92e-06。
  **符号 / wins / p / dz 全都随口径变**，不是只有幅度变。
  更隐蔽的是它会**改变现象的形状**：盒口径能把"多数实例与 0 不可区分、少数显著"
  抹成"处处微小为正"（详见 `logs/ablation_ladder.caliber.json` 与本文 §4）。
* 报「效应有多大」（rel% / dz）：只用实例边界口径，否则会高估一个数量级；
* 引用历史盒口径数字时，必须同时给放大倍数。
* **任何数字（含 $p$、wins）离开口径都不可引用。**

用法：
    python scripts/caliber_audit.py --labs logs/ablation_ladder.json \\
        --pairs D2=D1,D3=D2,D4=D3,D5=D4,RMOEAD=D5
    python scripts/caliber_audit.py --labs logs/_mk10_init.json --pairs I_mwr=I_rand
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
from rmoea_d.utils.metrics import compute_hv, estimate_hv_bounds  # noqa: E402

REF = (1.02, 1.02)


def pair_stat(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    if d.size == 0:
        return None
    if np.allclose(d, 0.0) or d.size < 3:
        return {"d": float(d.mean()), "wins": int((d > 0).sum()), "n": d.size,
                "p": 1.0, "dz": 0.0}
    p = float(stats.wilcoxon(d).pvalue)
    sd = float(d.std(ddof=1))
    return {"d": float(d.mean()), "wins": int((d > 0).sum()), "n": d.size,
            "p": p, "dz": float(d.mean() / sd) if sd > 0 else 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labs", required=True, help="逗号分隔的 lab 文件")
    ap.add_argument("--pairs", required=True, help="形如 A=B,C=D（A 为被测，B 为对照）")
    ap.add_argument("--out", default="")
    ap.add_argument("--box-arms", choices=("pair", "all"), default="pair",
                    help="盒口径的臂集：pair=只用这一对臂（默认，与论文 §caliber(b) 的"
                         "「逐臂对重算边界」一致，也与 scripts/paper_export.py 的 "
                         "box_hv_map 一致）；all=用 --pairs 里出现的全部臂（缺陷 18 的旧行为，"
                         "保留以便量化「臂集污染」本身）")
    args = ap.parse_args()

    labs = [s.strip() for s in args.labs.split(",") if s.strip()]
    pairs = [tuple(s.split("=")) for s in args.pairs.split(",") if "=" in s]
    arms = sorted({x for p in pairs for x in p})

    per, _rows = iva.load_by_instance(labs)
    print("=" * 100)
    print("口径审计  labs=%s" % ",".join(os.path.basename(p) for p in labs))
    print("臂对（左=被测，右=对照）：%s" % ", ".join("%s vs %s" % p for p in pairs))
    print("盒口径臂集 = %s（%s）"
          % (args.box_arms, "逐臂对" if args.box_arms == "pair"
             else "全部 %d 个臂 %s" % (len(arms), arms)))
    print("=" * 100)

    out = {"labs": [os.path.basename(p) for p in labs], "per_instance": {},
           "box_arms": args.box_arms, "arms": arms}
    totals = collections.defaultdict(list)

    for inst in per:
        sub = [r for r in per[inst] if r.get("label") in arms and r.get("final_pf")]
        if not sub:
            continue
        hv_inst = collections.defaultdict(dict)
        for r in sub:
            hv_inst[r["label"]][r["seed"]] = float(r["final_hv"])
        fronts = [np.asarray(r["final_pf"], float) for r in sub]
        lo, hi = estimate_hv_bounds(fronts)
        hv_box = collections.defaultdict(dict)
        for r in sub:
            hv_box[r["label"]][r["seed"]] = compute_hv(
                np.asarray(r["final_pf"], float), ref_point=REF, norm_bounds=(lo, hi))

        print("\n-- %s   盒 lo=%s hi=%s"
              % (inst, np.round(lo, 2).tolist(), np.round(hi, 2).tolist()))
        for a, b in pairs:
            if a not in hv_inst or b not in hv_inst:
                continue
            # ── 逐臂对盒（默认，本文口径）：盒内恰好只有 (a, b) 两个臂 ──
            hv_box_pair = collections.defaultdict(dict)
            pair_rows = [r for r in sub if r["label"] in (a, b)]
            fpair = [np.asarray(r["final_pf"], float) for r in pair_rows]
            lo_p, hi_p = estimate_hv_bounds(fpair)
            for r in pair_rows:
                hv_box_pair[r["label"]][r["seed"]] = compute_hv(
                    np.asarray(r["final_pf"], float), ref_point=REF,
                    norm_bounds=(lo_p, hi_p))
            boxes = {"inst": hv_inst, "box": hv_box_pair, "box_all": hv_box}
            rec = {}
            for tag, hv in boxes.items():
                ss = sorted(set(hv[a]) & set(hv[b]))
                if not ss:
                    continue
                va = [hv[a][s] for s in ss]
                vb = [hv[b][s] for s in ss]
                st = pair_stat(va, vb)
                st["rel_pct"] = (st["d"] / float(np.mean(vb)) * 100.0) if np.mean(vb) else float("nan")
                st["mean_b"] = float(np.mean(vb))
                st["mean_a"] = float(np.mean(va))
                rec[tag] = st
            if "inst" not in rec or "box" not in rec:
                continue
            ratio = (rec["box"]["rel_pct"] / rec["inst"]["rel_pct"]
                     if abs(rec["inst"]["rel_pct"]) > 1e-12 else float("nan"))
            ratio_all = (rec["box_all"]["rel_pct"] / rec["inst"]["rel_pct"]
                         if abs(rec["inst"]["rel_pct"]) > 1e-12 else float("nan"))
            rec["amplification"] = float(ratio)
            rec["amplification_all_arms"] = float(ratio_all)
            rec["box_pair_bounds"] = [np.round(lo_p, 4).tolist(),
                                      np.round(hi_p, 4).tolist()]
            out["per_instance"].setdefault(inst, {})["%s|%s" % (a, b)] = rec
            totals["%s|%s" % (a, b)].append(rec)
            print("   %-16s 实例边界 rel=%+7.3f%%  d=%+.5f  wins=%d/%d p=%.2e dz=%+.2f"
                  % ("%s vs %s" % (a, b), rec["inst"]["rel_pct"], rec["inst"]["d"],
                     rec["inst"]["wins"], rec["inst"]["n"], rec["inst"]["p"],
                     rec["inst"]["dz"]))
            print("   %-16s 盒口径   rel=%+7.3f%%  d=%+.5f  wins=%d/%d p=%.2e dz=%+.2f"
                  % ("", rec["box"]["rel_pct"], rec["box"]["d"],
                     rec["box"]["wins"], rec["box"]["n"], rec["box"]["p"],
                     rec["box"]["dz"]))
            if np.isfinite(ratio):
                print("   %-16s → 盒口径(逐臂对) 把 rel%% 放大了 %.1f 倍" % ("", ratio))
            if np.isfinite(ratio_all):
                print("   %-16s   对照：盒口径(全部 %d 臂) 放大 %.1f 倍"
                      % ("", len(arms), ratio_all))

    print("\n" + "=" * 100)
    print("跨实例汇总（rel%% 直接平均；各实例效应同号时才有意义）")
    print("=" * 100)
    print("%-24s %-14s %-16s %-10s %-12s"
          % ("臂对", "实例边界 rel%", "盒口径(逐臂对) rel%", "放大", "放大(全部臂)"))
    for k, recs in totals.items():
        ri = np.array([r["inst"]["rel_pct"] for r in recs])
        rb = np.array([r["box"]["rel_pct"] for r in recs])
        w = sum(r["inst"]["wins"] for r in recs)
        n = sum(r["inst"]["n"] for r in recs)
        print("%-24s %-14s %-16s %-10s %-12s"
              % (k, "%+.3f%%" % ri.mean(), "%+.3f%%" % rb.mean(),
                 "%.1f×" % (rb.mean() / ri.mean()) if abs(ri.mean()) > 1e-12 else "n/a",
                 "%.1f×" % np.mean([r["amplification_all_arms"] for r in recs])
                 if abs(ri.mean()) > 1e-12 else "n/a"))
        print("%-24s wins(pooled) %d/%d" % ("", w, n))

    out_path = args.out or (os.path.splitext(args.labs.split(",")[0])[0] + ".caliber.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    print("\n已写出 %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
