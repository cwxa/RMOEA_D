"""诊断 numpy reduce/prod 热点的来源（哪个 rng 调用在烧 CPU）。"""
import cProfile
import io
import pstats
import sys

import numpy as np


def bench(name, fn, n=6000):
    rng0 = np.random.RandomState(7)
    fn(rng0)  # warm
    pr = cProfile.Profile()
    rng = np.random.RandomState(7)
    pr.enable()
    for _ in range(n):
        fn(rng)
    pr.disable()
    st = pstats.Stats(pr)
    tot = st.total_tt
    # 统计 np.prod / _wrapreduction 调用
    prod_calls = 0
    for (fname, lineno, func), (cc, nc, tt, ct, callers) in st.stats.items():
        if func in ("prod", "_wrapreduction", "_prod_dispatcher"):
            prod_calls += nc
    print("%-42s %8.4fs  (%6.1fus/call)  np_prod_calls=%d"
          % (name, tot, tot / n * 1e6, prod_calls))


probs = np.ones(5) / 5.0
list5 = [1, 2, 3, 4, 5]


def main():
    # 显式 CLI 契约：`--help` 必须可用、未知旗标必须报错（缺陷 27 的同族防护）。
    import argparse
    argparse.ArgumentParser(
        description="诊断 numpy reduce/prod 热点的来源（无参数，直接跑）").parse_args()
    print("instance: numpy", np.__version__, "python", sys.version.split()[0])
    print()
    bench("rng.choice(5, p=probs)          [select_operator]",
          lambda r: r.choice(5, p=probs))
    bench("rng.choice(150, 2, replace=False) [ls4/ls5/mutate]",
          lambda r: r.choice(150, 2, replace=False))
    bench("rng.choice(10, 2, replace=False)  [moead parents]",
          lambda r: r.choice(10, 2, replace=False))
    bench("rng.choice(list5)                 [repair fallback]",
          lambda r: r.choice(list5))
    bench("rng.randint(150)                  [ls1/ls2/ls3]",
          lambda r: r.randint(150))
    bench("rng.randint(0,2,size=150)         [ux_crossover]",
          lambda r: r.randint(0, 2, size=150))
    bench("rng.rand()                        [crossover_rate]",
          lambda r: r.rand())
    big = list(range(150))
    bench("rng.shuffle(list150)              [init_random]",
          lambda r: r.shuffle(big))

    print()
    print("--- 对照：同样次数的纯 Python 循环 ---")
    bench("python: sum of 150 ints",
          lambda r: sum(range(150)))


if __name__ == "__main__":
    main()
