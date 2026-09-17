"""真实墙钟剖析（不用 cProfile）。

**为什么需要这个**

`scripts/profile_hotspots.py` 用 cProfile，它按行/按调用插桩，会把纯 Python
循环的相对成本放大数倍 —— 实测据此做的 numpy 批量化改造反而**变慢 8–11%**。
本脚本改用 `time.perf_counter()` 包裹目标函数累加真实耗时，只在函数边界插桩，
对被测代码的扰动小得多。

**怎么读**

`cum%` 是"含子调用的总占比"，会重叠（例如 decode_crisp 同时被 rvns/moead 调用）。
`self估计` = 总耗时 − 已单独计时的子函数耗时，用于近似自身成本。

用法：
    python scripts/profile_wall.py --instance Mk10 --max_gen 60
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


class Timer:
    def __init__(self):
        self.d = {}          # name -> [total, calls]
        self.stack = []      # 当前调用链（用于算自身耗时）

    def wrap_module(self, mod, name, label=None):
        orig = getattr(mod, name, None)
        if orig is None:
            return False
        label = label or ("%s.%s" % (mod.__name__.split(".")[-1], name))
        rec = self.d.setdefault(label, [0.0, 0])
        tim = self

        def timed(*a, **k):
            t0 = time.perf_counter()
            tim.stack.append(label)
            try:
                return orig(*a, **k)
            finally:
                tim.stack.pop()
                dt = time.perf_counter() - t0
                rec[0] += dt
                rec[1] += 1
                # 把这段耗时从所有祖先的"自身耗时"里扣掉
                sub = tim.d.setdefault(label + "#self", [0.0, 0])
                sub[0] += dt

        timed.__name__ = name
        setattr(mod, name, timed)
        return True

    def report(self, total_wall, top=20):
        print()
        print("%-38s %9s %8s %9s %10s" % ("函数", "总耗时s", "调用数", "占比%", "us/call"))
        print("-" * 80)
        rows = sorted(self.d.items(), key=lambda kv: -kv[1][0])
        for name, (t, c) in rows[:top]:
            print("%-38s %9.3f %8d %8.2f%% %10.2f"
                  % (name, t, c, 100.0 * t / total_wall, 1e6 * t / max(c, 1)))
        print("-" * 80)
        print("总墙钟 %.3fs" % total_wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="Mk10")
    ap.add_argument("--n_pop", type=int, default=100)
    ap.add_argument("--max_gen", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ls_trials", type=int, default=1)
    args = ap.parse_args()

    from rmoea_d.algorithm import RMOEAD
    from rmoea_d.core import encoding, moead, operators, rvns
    from rmoea_d.utils import metrics

    t = Timer()
    # 叶子函数
    t.wrap_module(encoding, "decode_crisp")
    t.wrap_module(operators, "_repair_ma_for_os")
    t.wrap_module(operators, "pox_crossover")
    t.wrap_module(operators, "ux_crossover")
    t.wrap_module(operators, "mutate_os")
    t.wrap_module(operators, "mutate_ma")
    t.wrap_module(operators, "repair_os")
    t.wrap_module(operators, "_get_op_index")
    t.wrap_module(rvns, "_build_schedule_info")
    t.wrap_module(moead, "tchebycheff")
    t.wrap_module(metrics, "compute_hv")
    t.wrap_module(metrics, "non_dominated_sort")
    # 上层
    t.wrap_module(rvns, "rvns_generation")
    t.wrap_module(moead, "moead_generation")
    t.wrap_module(moead, "compute_neighbors")
    # 被「from … import」绑定的引用也要各自 patch
    for mod, names in ((moead, ["decode_crisp", "_repair_ma_for_os"]),
                       (rvns, ["decode_crisp", "_repair_ma_for_os", "_get_op_index"])):
        for nm in names:
            t.wrap_module(mod, nm, "%s.%s(from %s)" % (mod.__name__.split(".")[-1], nm,
                                                       "enc" if nm == "decode_crisp" else "ops"))
    t.wrap_module(RVNS_mod := sys.modules["rmoea_d.core.rvns"].RVNS, "apply_local_search")
    t.wrap_module(sys.modules["rmoea_d.algorithm"].RMOEAD, "_compute_pf")

    def run():
        s = RMOEAD(instance_name=args.instance, n_pop=args.n_pop,
                   max_gen=args.max_gen, seed=args.seed,
                   enable_rvns=True, fixed_T=10, rvns_ls_trials=args.ls_trials)
        return s.solve()

    run()                       # warm
    for k in t.d.values():
        k[0] = 0.0
        k[1] = 0
    t0 = time.perf_counter()
    res = run()
    wall = time.perf_counter() - t0

    print("instance=%s n_pop=%d G=%d  HV=%.6f" % (args.instance, args.n_pop,
                                                  args.max_gen, res["final_hv"]))
    t.report(wall)
    return 0


if __name__ == "__main__":
    sys.exit(main())
