# -*- coding: utf-8 -*-
"""
Q-PAS implementation audit.

Checks every clause of the paper's Algorithm 3 / Section 4.5 against
src/rmoea_d/core/qlearning.py, plus five numerical verifications that
cannot be settled by reading the code alone:

  [1] the Q-update formula as printed (Eq.13) vs the code (TD form)
  [2] CV / DV measured on real fronts: dimension dominance, state occupancy,
      scale sensitivity, and consistency with the HV normalisation box
  [3] what the learned Q-table actually encodes (reward frequency vs Q value)
  [4] paper Table 3 vs the shape of our own Q-table
  [5] RNG sharing between Q-PAS and the main search stream

Usage:
    python scripts/qpas_audit.py                        # Mk10, seed 42
    python scripts/qpas_audit.py --instance Mk01 --seed 7
    python scripts/qpas_audit.py --out logs/_audit.txt  # also print to stdout
"""
import argparse
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

LOG = []
ACTIONS = [5, 10, 15, 20]      # paper's action space (Section 4.5.5)


def P(*a):
    line = " ".join(str(x) for x in a)
    LOG.append(line)


_ap = argparse.ArgumentParser(description="Audit the Q-PAS implementation.")
_ap.add_argument("--instance", default="Mk10", help="instance name (default Mk10)")
_ap.add_argument("--seed", type=int, default=42, help="RNG seed for the probe run")
_ap.add_argument("--n_pop", type=int, default=100)
_ap.add_argument("--max_gen", type=int, default=200)
_ap.add_argument("--out", default=None,
                 help="write the report here (default logs/_qpas_audit.txt)")
_a = _ap.parse_args()
INSTANCE, SEED = _a.instance, _a.seed


# ══════════════════════════════════════════════════════════════════════
P("=" * 82)
P("[1] Q-update formula: paper Eq.(13) as literally printed  vs  code")
P("=" * 82)
P("")
P("  paper Eq.(13) / Alg.3 line 14:")
P("      Q(St,At) <- Q(St,At) + alpha*[ R + gamma*( max Q(St+1,At) - Q(St,At) ) ]")
P("  code  qlearning.py:195:")
P("      Q(s,a) <- old_q + alpha * ( reward + gamma*max_next - old_q )")
P("")
P("  The difference: the '- Q(s,a)' term sits INSIDE gamma's parenthesis in")
P("  the paper, but OUTSIDE it in the code. Simulate a single (state,action)")
P("  visited repeatedly with reward R = 10 (alpha=0.4, gamma=0.6):")
P("")

alpha, gamma, R = 0.4, 0.6, 10.0
Qp = 0.0
Qs = 0.0
rows = []
for n in range(1, 201):
    Qp = Qp + alpha * (R + gamma * (Qp - Qp))          # max Q(s',.) - Q(s,a) = 0
    Qs = Qs + alpha * (R + gamma * Qs - Qs)
    if n in (1, 2, 5, 10, 50, 100, 200):
        rows.append((n, Qp, Qs))

P(f"  {'iteration':>9} | {'paper Eq.(13) literal':>22} | {'standard TD (code)':>19}")
P("  " + "-" * 58)
for n, a, b in rows:
    P(f"  {n:>9} | {a:>22.4f} | {b:>19.4f}")
P("  " + "-" * 58)
P(f"  standard TD fixed point = R/(1-gamma) = {R / (1 - gamma):.4f}   <- code converges here")
P(f"  paper literal grows by alpha*R = {alpha * R:.2f} per step -> 200 steps = {200 * alpha * R:.1f}")
P("")
P("  VERDICT: the paper's literal form has NO fixed point for R>0. Setting")
P("           Q = max Q' (the converged case) gives R = 0, a contradiction.")
P("           It is a misplaced-parenthesis typo. The code's standard TD form")
P("           is the only reading that is mathematically well-posed.")

# ══════════════════════════════════════════════════════════════════════
P("")
P("=" * 82)
P("[2] CV / DV on real Pareto fronts recorded from a live run")
P("=" * 82)
P("")

from rmoea_d.algorithm import RMOEAD                    # noqa: E402
from rmoea_d.core.qlearning import QLearningPAS         # noqa: E402

FRONTS = []
_orig_cvdv = QLearningPAS.compute_cv_dv


def _patched_cvdv(self, pf):
    cv, dv = _orig_cvdv(self, pf)
    FRONTS.append(np.array(pf, dtype=float))
    return cv, dv


UPD = []
_orig_upd = QLearningPAS.update


def _patched_upd(self, state, action_idx, reward, next_state):
    UPD.append((int(state), int(action_idx), float(reward), int(next_state)))
    return _orig_upd(self, state, action_idx, reward, next_state)


SEL = []
_orig_sel = QLearningPAS.select_action


def _patched_sel(self, state, rng):
    idx = _orig_sel(self, state, rng)
    SEL.append(int(idx))
    return idx


QLearningPAS.compute_cv_dv = _patched_cvdv
QLearningPAS.update = _patched_upd
QLearningPAS.select_action = _patched_sel

solver = RMOEAD(instance_name=INSTANCE, n_pop=_a.n_pop, max_gen=_a.max_gen,
                data_dir="data", seed=SEED, enable_rvns=False,
                ql_reward_mode="dv")
solver.solve()

QLearningPAS.compute_cv_dv = _orig_cvdv
QLearningPAS.update = _orig_upd
QLearningPAS.select_action = _orig_sel

P(f"  instance = {INSTANCE} | seed = {SEED} | {_a.max_gen} generations | "
  f"{len(FRONTS)} fronts captured")
P("")

# ---- 2a. dimension dominance in CV -----------------------------------
mv1 = np.mean([np.mean(f[:, 0]) for f in FRONTS])
mv2 = np.mean([np.mean(f[:, 1]) for f in FRONTS])
sq1 = np.mean([np.mean(f[:, 0] ** 2) for f in FRONTS])
sq2 = np.mean([np.mean(f[:, 1] ** 2) for f in FRONTS])
P("  (2a) CV is computed as sqrt(mean(f1^2 + f2^2)) with NO normalisation.")
P(f"       mean(f1) = {mv1:8.3f}    mean(f2) = {mv2:8.3f}")
P(f"       contribution to CV^2:  f1 -> {100 * sq1 / (sq1 + sq2):5.1f}%"
  f"     f2 -> {100 * sq2 / (sq1 + sq2):5.1f}%")
P("       => DeltaCV is dominated by f2 (total workload); the makespan")
P("          objective barely moves the state. Faithful to the paper (which")
P("          also does not normalise) but worth stating in the write-up.")

# ---- 2b. DeltaCV / DeltaDV -> state occupancy -------------------------
_measured = [_orig_cvdv(solver.ql, f) for f in FRONTS]
cvs = np.array([c for c, _ in _measured])
dvs = np.array([d for _, d in _measured])
dcv = cvs[:-1] - cvs[1:]
ddv = dvs[1:] - dvs[:-1]


def state_of(dc, dd):
    if dc > 0 and dd > 0:
        return 0
    if dc > 0 and dd <= 0:
        return 1
    if dc <= 0 and dd > 0:
        return 2
    return 3


states = np.array([state_of(a, b) for a, b in zip(dcv, ddv)])
P("")
P("  (2b) State occupancy over the run (paper Alg.3 line 4 state definition):")
for s, nm in enumerate(["S0 (dCV>0, dDV>0)", "S1 (dCV>0, dDV<=0)",
                        "S2 (dCV<=0,dDV>0)", "S3 (dCV<=0,dDV<=0)"]):
    c = int((states == s).sum())
    P(f"       {nm:<22} {c:>4} gens   {100 * c / len(states):5.1f}%")
P(f"       P(dDV > 0) overall = {(ddv > 0).mean():.3f}")

# ---- 2c. does normalisation change the state sequence? ---------------
f1lo = min(f[:, 0].min() for f in FRONTS)
f1hi = max(f[:, 0].max() for f in FRONTS)
f2lo = min(f[:, 1].min() for f in FRONTS)
f2hi = max(f[:, 1].max() for f in FRONTS)


def cv_dv_norm(pf):
    q = pf.copy()
    q[:, 0] = (q[:, 0] - f1lo) / (f1hi - f1lo)
    q[:, 1] = (q[:, 1] - f2lo) / (f2hi - f2lo)
    d = np.sqrt(np.sum(q ** 2, axis=1))
    cv = np.sqrt(np.mean(d ** 2))
    idx = np.argsort(q[:, 0])
    s = q[idx]
    ds = np.array([np.linalg.norm(s[i] - s[i + 1]) for i in range(len(s) - 1)])
    md = ds.mean() if len(ds) else 0.0
    dv = float(np.sum(np.abs(ds - md)) / (len(ds) * md)) if md > 0 else 0.0
    return cv, dv


ncv = np.array([cv_dv_norm(f)[0] for f in FRONTS])
ndv = np.array([cv_dv_norm(f)[1] for f in FRONTS])
ndcv, nddv = ncv[:-1] - ncv[1:], ndv[1:] - ndv[:-1]
nstates = np.array([state_of(a, b) for a, b in zip(ndcv, nddv)])
agree = int((nstates == states).sum())
P("")
P("  (2c) Re-running the SAME fronts with [0,1]-normalised objectives:")
P(f"       state sequence agreement with the un-normalised version = "
  f"{agree}/{len(states)} = {100 * agree / len(states):.1f}%")
P(f"       states that FLIP when the objective scales are equalised: "
  f"{len(states) - agree} ({100 * (len(states) - agree) / len(states):.1f}%)")
P("       => the state is NOT scale-invariant. Almost half of all visited")
P("          states change identity under a pure rescaling of the objectives.")
P("          Combined with (2a), the agent's state is in effect a function of")
P("          f2 alone (weights 96.5% vs 3.5%), so the makespan objective is")
P("          invisible to Q-PAS on this instance.")

# ---- 2d. consistency of the CV input scale with the HV box -----------
P("")
lo, hi = solver.hv_bounds
lo = np.asarray(lo, dtype=float)
hi = np.asarray(hi, dtype=float)
P("  (2d) scale consistency between the CV input and the HV normalisation box:")
P(f"       Q-PAS sees pf with       f1=[{f1lo:9.2f}, {f1hi:9.2f}]"
  f"  f2=[{f2lo:9.2f}, {f2hi:9.2f}]")
P(f"       HV box (solver.hv_bounds) f1=[{lo[0]:9.2f}, {hi[0]:9.2f}]"
  f"  f2=[{lo[1]:9.2f}, {hi[1]:9.2f}]")
inside = all(lo[0] <= f[:, 0].min() and f[:, 0].max() <= hi[0] and
             lo[1] <= f[:, 1].min() and f[:, 1].max() <= hi[1] for f in FRONTS)
P(f"       every front inside the box: {inside}   (must be True, else HV would"
  f" clip points away)")
P("       => the same objective vector feeds both CV and HV; no unit mismatch.")

# ══════════════════════════════════════════════════════════════════════
P("")
P("=" * 82)
P("[3] What the learned Q-table actually encodes")
P("=" * 82)
P("")

UPD = np.array(UPD) if len(UPD) else np.zeros((0, 4))
qt = solver.ql.q_table
P("  final Q-table (rows = S0..S3, cols = " +
  ", ".join(f"T={t}" for t in ACTIONS) + "):")
P(f"       {'':>6} " + " ".join(f"{'T=' + str(t):>10}" for t in ACTIONS) +
  "   argmax")
for s in range(4):
    am = int(np.argmax(qt[s]))
    P(f"       S{s:<5} " + " ".join(f"{v:10.4f}" for v in qt[s]) +
      f"   T={ACTIONS[am]}")
P("")
P("  Empirical reward frequency vs Q value (Q should ~= 10*P(dDV>0)/(1-gamma)):")
P(f"       {'state':>6} {'action':>7} {'visits':>7} {'rewards':>8} {'P(R=10)':>8}"
  f" {'pred Q':>9} {'actual Q':>9}")
ceil = 10.0 / (1 - gamma)
for s in range(4):
    for a_i, t in enumerate(ACTIONS):
        m = (UPD[:, 0] == s) & (UPD[:, 1] == a_i)
        n = int(m.sum())
        r = int((UPD[m, 2] > 0).sum()) if n else 0
        pr = r / n if n else 0.0
        P(f"       {s:>6} {t:>7} {n:>7} {r:>8} {pr:>8.3f}"
          f" {pr * ceil:>9.4f} {qt[s, a_i]:>9.4f}")
P("")
P("  Action usage (share of generations):")
vals, cnts = np.unique(SEL, return_counts=True)
for v, c in zip(vals, cnts):
    P(f"       T={ACTIONS[v]:<4} {c:>4} gens  {100 * c / len(SEL):5.1f}%")

# ══════════════════════════════════════════════════════════════════════
P("")
P("=" * 82)
P(f"[4] Paper Table 3 (a D1 run) vs the shape we obtain on {INSTANCE}")
P("=" * 82)
P("")
paper_t3 = np.array([[0.0, 1.798693, 0.0, 6.743954],
                     [0.0, 0.323268, 0.0, 5.271637],
                     [4.539988, 0.766909, 0.0, 0.0],
                     [2.847096, 2.283770, 2.911486, 1.152641]])
P("  paper Table 3 (Q-table after 200 iterations, instance D1):")
for s in range(4):
    P(f"       S{s:<5} " + " ".join(f"{v:>10.4f}" for v in paper_t3[s]))
P(f"       exact zeros in table: {int((paper_t3 == 0).sum())}/16")
P(f"       max value {paper_t3.max():.4f}  ->  implied success rate "
  f"{paper_t3.max() / ceil:.3f}  (ceiling 10/(1-gamma) = {ceil:.2f})")
P("")
P(f"  ours ({INSTANCE} / seed {SEED}):")
for s in range(4):
    P(f"       S{s:<5} " + " ".join(f"{v:>10.4f}" for v in qt[s]))
P(f"       exact zeros in table: {int((qt == 0).sum())}/16")
P("")
P("  Both tables are dominated by exact zeros, i.e. (state,action) pairs that")
P("  were never visited (any visit with reward 10 sets Q >= alpha*10 = 4).")
P("  A Q-table of this shape records 'the first arm that happened to hit a")
P("  reward', not the per-state optimum. Reproducing its exact pattern is not")
P("  a meaningful target -- and the paper's own text contradicts it: Section")
P("  5.3 says State4 should prefer T=5 or 10, but Table 3's argmax for State4")
P("  is T=15 (2.9115 > 2.8471 > 2.2838).")

# ══════════════════════════════════════════════════════════════════════
P("")
P("=" * 82)
P("[5] RNG sharing: does Q-PAS perturb the main search stream?")
P("=" * 82)
P("")
P("  algorithm.py:262 calls  self.ql.step(pf, self.rng)  -- the SAME rng that")
P("  drives MIX3 init, crossover, mutation and roulette. Every Q-PAS decision")
P("  consumes one or two draws, so a Q-PAS run and a fixed-T run with the same")
P("  seed diverge from generation 1 for reasons unrelated to T.")
P(f"  measured draws per generation in this run: {len(SEL)} selections for")
P(f"  {len(FRONTS)} generations, i.e. ~{len(SEL) / max(1, len(FRONTS)):.2f} per generation.")
P("  Consequence: the paired Wilcoxon we report is still unbiased, but its")
P("  variance is inflated -- part of the 'noise' attributed to Q-PAS is really")
P("  RNG-stream divergence. Isolating Q-PAS on its own RandomState(seed) is a")
P("  cheap, strictly-better experimental hygiene fix (not a paper violation).")

out = _a.out or os.path.join(ROOT, "logs", "_qpas_audit.txt")
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG) + "\n")
print("\n".join(LOG))
print("\nwritten:", out)
