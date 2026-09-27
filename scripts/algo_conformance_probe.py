#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照论文 Algorithm 1–5 的**实测等价性探针**（读代码定不了的三条）。

三条命题靠"看代码"都无法定论，必须实测：

  A. **双解码路径一致**：`decode_crisp`（热路径，进化与 HV 用，零分配）与
     `decode`（模糊路径，输出与甘特图用，构造 `FuzzyNumber`）给出的 crisp 值
     是否**逐位相同**。
     理论上成立的前提是 `clear(max(·)) == max(clear(·))`——因为 fuzzy 最大值
     先比 `(t1+2t2+t3)/4`，而 crisp 值就是 `(t1+2t2+t3)/4`。浮点下需要实测。
     若不成立，则"算法内部优化的目标"与"对外报告的目标"是两回事。

  B. **`repair_os` 的非法分支不可达**：`mutate_os` 是交换、POX 保持各工件工序
     计数，故修复分支理论上永不触发。这一条非查不可的原因是：**该分支内部用的是
     未播种的 `np.random.RandomState()`**——一旦可达，结果就不可复现，
     而所有"逐位一致"的承诺会静默失效。

  C. **精英档案的两种写法等价**：代码是 `A ∪ population`，论文 Algorithm 5 第 1 行
     是 `A ← A ∪ PF`。两者的**非支配集**是否相同。
     （不同的话，"实现比论文更严"这句话就只能当断言，不能当事实。）

用法
----
    python scripts/algo_conformance_probe.py
    python scripts/algo_conformance_probe.py --instance Mk09 --n-decode 1000

退出码：任一条不成立 → 非零（可直接当判据用）。
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from rmoea_d.core.encoding import decode, decode_crisp            # noqa: E402
from rmoea_d.core.instance import load_instance                    # noqa: E402
from rmoea_d.core.operators import (init_by_variant, mutate_os,    # noqa: E402
                                    pox_crossover)
from rmoea_d.utils.metrics import non_dominated_sort               # noqa: E402


def probe_decode_parity(inst, n, seed=0):
    """A：两条解码路径的 crisp 值逐位一致。返回 (不一致数, 最大绝对差)。"""
    rng = np.random.RandomState(seed)
    bad = 0
    worst = 0.0
    for _ in range(n):
        os_vec, ma_vec = init_by_variant(inst, 1, rng, "mix3")[0]
        mc1, wc1 = decode_crisp(os_vec, ma_vec, inst)
        _, _, mc2, wc2 = decode(os_vec, ma_vec, inst)
        worst = max(worst, abs(mc1 - mc2), abs(wc1 - wc2))
        if not (mc1 == mc2 and wc1 == wc2):
            bad += 1
    return bad, worst


def probe_repair_unreachable(inst, n, seed=1):
    """B：`repair_os` 的非法分支可达次数。返回 (可达次数, 调用总次数)。"""
    rng = np.random.RandomState(seed)
    reached = 0
    calls = 0
    expect = [len(inst["jobs"][j]) for j in range(inst["n_jobs"])]
    for _ in range(n):
        os1, ma1 = init_by_variant(inst, 1, rng, "mix3")[0]
        os2, ma2 = init_by_variant(inst, 1, rng, "mix3")[0]
        for cand in (mutate_os(os1, rng), pox_crossover(os1, os2, rng)[0],
                     pox_crossover(os1, os2, rng)[1]):
            calls += 1
            act = [0] * inst["n_jobs"]
            for j in cand:
                act[j] += 1
            if act != expect:
                reached += 1
    return reached, calls


def probe_archive_equivalence(inst, n_pop, seed=2):
    """C：非支配集(A ∪ P) == 非支配集(A ∪ ND(P))？返回 (是否相同, |s1|, |s2|)。"""
    rng = np.random.RandomState(seed)
    pop = init_by_variant(inst, n_pop, rng, "mix3")
    objs = [decode_crisp(o, m, inst) for o, m in pop]
    # 造一个"上一代档案"：取前一半后加扰动，确保档案里有被支配/支配混合的点
    arch_objs = [(o[0] * 1.1, o[1] * 1.05) for o in objs[: n_pop // 2]]
    nd_pop = non_dominated_sort(objs)
    s1 = sorted({tuple(map(round, o)) for o in
                 non_dominated_sort(list(arch_objs) + list(objs))})
    s2 = sorted({tuple(map(round, o)) for o in
                 non_dominated_sort(list(arch_objs) + list(nd_pop))})
    return s1 == s2, len(s1), len(s2)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default="Mk10")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--n-decode", type=int, default=400)
    ap.add_argument("--n-repair", type=int, default=2000)
    ap.add_argument("--n-pop", type=int, default=60)
    args = ap.parse_args()

    inst = load_instance(args.instance, args.data_dir, 42)

    print("=" * 78)
    print("论文算法一致性探针   instance=%s  (%d jobs / %d machines)"
          % (args.instance, inst["n_jobs"], inst["n_machines"]))
    print("=" * 78)

    fails = []

    bad, worst = probe_decode_parity(inst, args.n_decode)
    ok = (bad == 0)
    fails += [] if ok else ["A"]
    print("A. decode_crisp vs decode(模糊路径)  [%s]  不一致 %d/%d  最大差 %.3e"
          % ("PASS" if ok else "FAIL", bad, args.n_decode, worst))

    reached, calls = probe_repair_unreachable(inst, args.n_repair)
    ok = (reached == 0)
    fails += [] if ok else ["B"]
    print("B. repair_os 非法分支（内含未播种 RNG）  [%s]  可达 %d/%d 次调用"
          % ("PASS" if ok else "FAIL", reached, calls))

    same, n1, n2 = probe_archive_equivalence(inst, args.n_pop)
    fails += [] if same else ["C"]
    print("C. ND(A∪P) == ND(A∪ND(P))  [%s]  |s1|=%d  |s2|=%d"
          % ("PASS" if same else "FAIL", n1, n2))

    if fails:
        print("\n[FAIL] 不成立的命题：%s" % ", ".join(fails))
        return 1
    print("\n[OK] 三条命题全部成立")
    return 0


if __name__ == "__main__":
    sys.exit(main())
