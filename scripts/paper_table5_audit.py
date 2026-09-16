# -*- coding: utf-8 -*-
"""Audit the paper's own ablation ladder (Table 5) step by step.

The paper (Li et al., ESWA 203 (2022) 117380) reports a 6-rung variant ladder:

    RMOEA/D1  pure MOEA/D
    RMOEA/D2  + MIX3 initial strategy
    RMOEA/D3  + randomly-selected VNS
    RMOEA/D4  + Q-PAS                 <-- the component under audit
    RMOEA/D5  + elite archive
    RMOEA/D   with RVNS replacing the random VNS

Table 5 gives mean HV per instance (30 runs each) but NOT per-run values,
so only a sign test on instance-level means is available here. That test is
weaker than the paired Wilcoxon we run on our own reproduction, and the
script labels it accordingly.

Table 4 gives the Friedman average ranking of the same 6 variants over the
23 instances (p = 1.5e-4). The paper reads that ranking as evidence that
"each part improves the result against the last one" -- i.e. it attributes
a monotone rank ordering across the whole ladder to each component in turn.
This script separates those two pieces of evidence.
"""

import json
import os
import sys
from math import comb

# --------------------------------------------------------------------------
# Table 5, transcribed from the paper (page 9)
# --------------------------------------------------------------------------
INSTANCES = ["D1", "D2", "D3", "D4", "D5",
             "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8",
             "FMk01", "FMk02", "FMk03", "FMk04", "FMk05",
             "FMk06", "FMk07", "FMk08", "FMk09", "FMk10"]

VARIANTS = ["RMOEA/D1", "RMOEA/D2", "RMOEA/D3", "RMOEA/D4", "RMOEA/D5", "RMOEA/D"]

TABLE5 = {
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

# Table 4: Friedman average ranking, p = 1.5e-4 over the 23 instances
TABLE4_RANK = {
    "RMOEA/D1": 4.6087, "RMOEA/D2": 4.3913, "RMOEA/D3": 3.6087,
    "RMOEA/D4": 3.0870, "RMOEA/D5": 2.9130, "RMOEA/D": 2.3913,
}

# Table 3: Q-table after 200 iterations
TABLE3 = {
    "State1  (dCV>0, dDV>0)": [0.0, 1.798693, 0.0, 6.743954],
    "State2  (dCV>0, dDV<=0)": [0.0, 0.323268, 0.0, 5.271637],
    "State3  (dCV<=0, dDV>0)": [4.539988, 0.766909, 0.0, 0.0],
    "State4  (dCV<=0, dDV<=0)": [2.847096, 2.283770, 2.911486, 1.152641],
}
TABLE3_ACTIONS = ["T=5", "T=10", "T=15", "T=20"]

# The rungs, in the order the paper adds them
STEPS = [
    ("initial strategy (MIX3)", 0, 1),
    ("randomly-selected VNS",   1, 2),
    ("Q-PAS  <-- under audit",  2, 3),
    ("elite archive",           3, 4),
    ("RVNS (replaces rand. VNS)", 4, 5),
]

FMK = [i for i, n in enumerate(INSTANCES) if n.startswith("FMk")]


def sign_test_p(n_pos, n_tot):
    """Two-sided exact binomial test against p=0.5 (ties excluded)."""
    if n_tot == 0:
        return float("nan")
    k = max(n_pos, n_tot - n_pos)
    tail = sum(comb(n_tot, i) for i in range(k, n_tot + 1)) / 2.0 ** n_tot
    return min(1.0, 2.0 * tail)


def step_stats(i_from, i_to, idx=None):
    idx = list(range(len(INSTANCES))) if idx is None else idx
    d, rel = [], []
    for k in idx:
        name = INSTANCES[k]
        a = TABLE5[name][i_from]
        b = TABLE5[name][i_to]
        d.append(b - a)
        rel.append(100.0 * (b - a) / a)
    n_pos = sum(1 for x in d if x > 1e-12)
    n_neg = sum(1 for x in d if x < -1e-12)
    n_tie = len(d) - n_pos - n_neg
    return dict(n=len(d), n_pos=n_pos, n_neg=n_neg, n_tie=n_tie,
                mean=sum(d) / len(d), mean_rel=sum(rel) / len(rel),
                lo=min(d), hi=max(d),
                p_sign=sign_test_p(n_pos, n_pos + n_neg))


def main():
    out = []
    w = out.append

    w("=" * 92)
    w("AUDIT of Li et al. (2022) ESWA 203:117380 -- Table 5 ablation ladder")
    w("=" * 92)
    w("")
    w("HV is normalised per table, so only within-table deltas are comparable.")
    w("Table 5 gives only per-instance means over 30 runs -> no paired Wilcoxon")
    w("is possible from the published data; an exact sign test is used instead.")
    w("")

    w("-- 1. rung-by-rung attribution (all 23 instances) " + "-" * 44)
    w("")
    w(f"{'step':<28}{'rank gain':>10}{'mean dHV':>11}{'mean d%':>9}"
      f"{'improved':>10}{'sign p':>10}")
    w("-" * 92)
    for label, f, t in STEPS:
        s = step_stats(f, t)
        rk = TABLE4_RANK[VARIANTS[t]] - TABLE4_RANK[VARIANTS[f]]
        w(f"{label:<28}{rk:+10.4f}{s['mean']:+11.6f}{s['mean_rel']:+9.2f}%"
          f"{s['n_pos']:>6}/{s['n']:<3}{s['p_sign']:>10.4f}")
    w("-" * 92)
    w("rank gain = drop in Friedman average rank (Table 4). Larger = paper")
    w("credits this component more. mean d% is on the raw HV level.")
    w("")

    w("-- 2. the Q-PAS rung alone (RMOEA/D3 -> RMOEA/D4) " + "-" * 44)
    w("")
    for tag, idx in (("all 23 instances", None),
                     ("FMk01-FMk10 only (our benchmark)", FMK)):
        s = step_stats(2, 3, idx)
        w(f"{tag}")
        w(f"    mean dHV = {s['mean']:+.6f}   mean d% = {s['mean_rel']:+.2f}%   "
          f"range [{s['lo']:+.6f}, {s['hi']:+.6f}]")
        w(f"    improved {s['n_pos']}/{s['n']}  worsened {s['n_neg']}/{s['n']}  "
          f"tied {s['n_tie']}   exact sign-test p = {s['p_sign']:.4f}"
          + ("   ***" if s['p_sign'] < 0.001 else
             "  **" if s['p_sign'] < 0.01 else
             "  *" if s['p_sign'] < 0.05 else "   n.s."))
        w("")
    per_inst = [(INSTANCES[k], TABLE5[INSTANCES[k]][3] - TABLE5[INSTANCES[k]][2])
                for k in FMK]
    w("    per-instance FMk deltas:")
    w("      " + "  ".join(f"{n}:{d:+.5f}" for n, d in per_inst))
    w("")

    w("-- 3. same-window cross-check: every rung vs its predecessor, FMk only "
      + "-" * 20)
    w("")
    w(f"{'step':<28}{'mean d%':>10}{'improved':>10}{'sign p':>10}")
    w("-" * 92)
    for label, f, t in STEPS:
        s = step_stats(f, t, FMK)
        w(f"{label:<28}{s['mean_rel']:+10.2f}%"
          f"{s['n_pos']:>6}/{s['n']:<3}{s['p_sign']:>10.4f}")
    w("-" * 92)
    w("")

    w("-- 4. is the paper's HV normalisation held fixed across tables? " + "-" * 28)
    w("")
    w("RMOEA/D HV for the SAME algorithm on the SAME instance, reported in")
    w("Table 5 (variant comparison) vs Table 7 (algorithm comparison):")
    w("")
    TABLE7_RMOEAD = {"FMk01": 0.073803, "FMk02": 0.070362, "FMk03": 0.078470,
                     "FMk04": 0.098794, "FMk05": 0.062563, "FMk06": 0.132173,
                     "FMk07": 0.099004, "FMk08": 0.043060, "FMk09": 0.092580,
                     "FMk10": 0.172348}
    w(f"{'instance':<10}{'Table 5':>12}{'Table 7':>12}{'diff':>12}")
    w("-" * 92)
    for name, v7 in TABLE7_RMOEAD.items():
        v5 = TABLE5[name][-1]
        w(f"{name:<10}{v5:>12.6f}{v7:>12.6f}{v7 - v5:>+12.6f}")
    w("-" * 92)
    w("The same algorithm+instance gets two different HV values -> the")
    w("normalisation box is re-derived per experiment. Absolute HV cannot be")
    w("compared across tables (or against a reproduction); only the within-")
    w("table relative deltas carry information.")
    w("")

    w("-- 5. reading the paper's Q-table (Table 3) " + "-" * 46)
    w("")
    w(f"{'state':<26}" + "".join(f"{a:>12}" for a in TABLE3_ACTIONS)
      + f"{'min':>10}{'argmax':>9}")
    w("-" * 92)
    zeros = 0
    cells = 0
    for st, q in TABLE3.items():
        am = TABLE3_ACTIONS[max(range(4), key=lambda i: q[i])]
        nz = sum(1 for v in q if v == 0.0)
        zeros += nz
        cells += 4
        w(f"{st:<26}" + "".join(f"{v:>12.6f}" for v in q)
          + f"{min(q):>10.6f}{am:>9}")
    w("-" * 92)
    w(f"cells still exactly 0 after 200 iterations: {zeros}/{cells} "
      f"({100.0 * zeros / cells:.0f}%)")
    w("")
    w("A zero cell means that action never once produced dDV>0 in that state")
    w("in the entire run. With reward R=10 on success and alpha=0.4, gamma=0.6,")
    w("Q(s,a) ~ 10*P(success|s,a)/(1-gamma) = 25*P, so the announced ceiling is")
    w("25.0. The paper's largest entry is 6.743954 -> P ~ 0.27.")
    w("")
    w("argmax over such a table selects the single arm that happened to get")
    w("rewarded first; the other arms stay at 0 because eps=0.8 means 80% of")
    w("the steps re-pull the current argmax. The table therefore documents")
    w("early-luck lock-in, not a per-state optimum.")
    w("")

    txt = "\n".join(out)
    print(txt)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dst = os.path.join(root, "logs", "_paper_audit.txt")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\nwritten:", dst)

    jdst = os.path.join(root, "logs", "_paper_audit.json")
    payload = {
        "source": "Li et al., ESWA 203 (2022) 117380, Tables 3-5, 7",
        "steps": {label: step_stats(f, t) for label, f, t in STEPS},
        "qpas_fmk": step_stats(2, 3, FMK),
        "qpas_all": step_stats(2, 3),
        "fmk_steps": {label: step_stats(f, t, FMK) for label, f, t in STEPS},
        "table4_rank": TABLE4_RANK,
    }
    with open(jdst, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("written:", jdst)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
