"""墙钟热点剖析：RMOEA/D 单次运行的时间构成。

用途：回答「固定墙钟下，改进必须来自让每一秒更有价值」——先看清每一秒花在哪。

用法：
    python scripts/profile_hotspots.py --instance Mk10 --max_gen 60 --n_pop 100

输出：top-N 累积耗时（函数级）+ top-N 自身耗时（行级热点候选）。
"""
import argparse
import cProfile
import io
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="Mk10")
    ap.add_argument("--n_pop", type=int, default=100)
    ap.add_argument("--max_gen", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ls_trials", type=int, default=1)
    ap.add_argument("--top", type=int, default=28)
    args = ap.parse_args()

    from rmoea_d.algorithm import RMOEAD

    def build():
        return RMOEAD(instance_name=args.instance, n_pop=args.n_pop,
                      max_gen=args.max_gen, seed=args.seed,
                      enable_rvns=True, fixed_T=10, rvns_ls_trials=args.ls_trials)

    # 先热一次（导入/缓存），再计一次干净墙钟
    t0 = time.perf_counter()
    build().solve()
    wall = time.perf_counter() - t0
    print(f"[wall] 单次 {args.instance} n_pop={args.n_pop} G={args.max_gen} "
          f"ls_trials={args.ls_trials}: {wall:.2f}s")

    pr = cProfile.Profile()
    pr.enable()
    build().solve()
    pr.disable()

    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf)
    st.sort_stats("tottime")
    st.print_stats(args.top)
    txt = buf.getvalue()

    out = os.path.join("logs", "_profile_%s_G%d.txt" % (args.instance, args.max_gen))
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# wall={wall:.2f}s  instance={args.instance} n_pop={args.n_pop} "
                f"G={args.max_gen} ls_trials={args.ls_trials}\n\n")
        f.write(txt)
        buf2 = io.StringIO()
        st2 = pstats.Stats(pr, stream=buf2)
        st2.sort_stats("cumulative")
        st2.print_stats(args.top)
        f.write("\n\n=== sort by cumulative ===\n")
        f.write(buf2.getvalue())
    print("-> written", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
