"""真实墙钟剖析（不用 cProfile）。

**为什么需要这个**

`scripts/profile_hotspots.py` 用 cProfile，它按行/按调用插桩，会把纯 Python
循环的相对成本放大数倍 —— 实测据此做的 numpy 批量化改造反而**变慢 8–11%**。
本脚本改用 `time.perf_counter()` 包裹目标函数累加真实耗时，只在函数边界插桩，
对被测代码的扰动小得多。

**怎么读**

* `总耗时s` = **含子调用**的真实累计耗时（会重叠：`decode_crisp` 同时被 rvns/moead 调用）。
* `自身s`   = 总耗时 **减去**已单独计时的子函数耗时，即函数体本身的开销。
* `调用数` 是最要紧的一列：**次数没变而耗时掉了** = 单次成本优化（等价改写成功）；
  **次数掉了** = 消掉了冗余计算（需要额外证明那部分确实冗余）。

用法：
    python scripts/profile_wall.py --instance Mk10 --max_gen 60
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


class Timer:
    """只在函数边界插桩的墙钟计时器。

    `d[label] = [总耗时, 调用数, 自身耗时]`。自身耗时靠"父扣子"算：
    每次进入函数压一个 0.0 到 stack，返回时把本次 dt 加到栈顶（父级），
    于是 `父的自身 = 父的 dt − 已计入的子调用 dt`。
    """

    def __init__(self):
        self.d = {}          # name -> [total, calls, self]
        self.stack = []      # 每层累积"子调用耗时"

    def wrap_module(self, mod, name, label=None):
        orig = getattr(mod, name, None)
        if orig is None:
            return False
        label = label or ("%s.%s" % (mod.__name__.split(".")[-1], name))
        tim = self

        def timed(*a, **k):
            t0 = time.perf_counter()
            tim.stack.append(0.0)
            try:
                return orig(*a, **k)
            finally:
                dt = time.perf_counter() - t0
                child = tim.stack.pop()
                if tim.stack:
                    tim.stack[-1] += dt      # 记到父级的"子调用"里
                tot, n, slf = tim.d.get(label, (0.0, 0, 0.0))
                tim.d[label] = (tot + dt, n + 1, slf + (dt - child))

        timed.__name__ = name
        setattr(mod, name, timed)
        return True

    def reset(self):
        for k in self.d:
            t, n, s = self.d[k]
            self.d[k] = (0.0, 0, 0.0)

    def report(self, total_wall, top=20):
        print()
        print("%-38s %9s %9s %8s %9s %10s" % ("函数", "总耗时s", "自身s",
                                              "调用数", "占比%", "us/call"))
        print("-" * 92)
        rows = sorted(self.d.items(), key=lambda kv: -kv[1][0])
        for name, (t, c, s) in rows[:top]:
            print("%-38s %9.3f %9.3f %8d %8.2f%% %10.2f"
                  % (name, t, s, c, 100.0 * t / total_wall, 1e6 * t / max(c, 1)))
        print("-" * 92)
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
    t.reset()
    t0 = time.perf_counter()
    res = run()
    wall = time.perf_counter() - t0

    print("instance=%s n_pop=%d G=%d  HV=%.6f" % (args.instance, args.n_pop,
                                                  args.max_gen, res["final_hv"]))
    t.report(wall)
    return 0


if __name__ == "__main__":
    sys.exit(main())
