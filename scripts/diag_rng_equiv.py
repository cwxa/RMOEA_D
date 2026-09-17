"""验证 `rng.choice(lst)` 与 `lst[rng.randint(len(lst))]` 是否消耗同一随机流。

背景：`RandomState.choice` 即使 `size=None` 也会走 `np.prod(size)` 通用路径，
实测 8.2us/次；而 `randint(k)` 只要 2.0us。若两者消耗的底层随机数序列相同，
就可以安全替换 —— 随机流逐位不变，已有实验数据无需重跑。

同时测 ux_crossover 的 `rng.randint(0,2,size=n)` 是否有等价快路径。
"""
import numpy as np

OK = True


def check(label, cond):
    global OK
    OK = OK and bool(cond)
    print("  %-58s %s" % (label, "OK" if cond else "*** FAIL ***"))


def main():
    print("numpy", np.__version__)
    print()
    print("[1] rng.choice(lst)  vs  lst[rng.randint(len(lst))]")

    for k in (1, 2, 3, 5, 8, 15):
        lst = [10 + 3 * i for i in range(k)]
        a = np.random.RandomState(12345)
        ra = [int(a.choice(lst)) for _ in range(64)]
        b = np.random.RandomState(12345)
        rb = [lst[int(b.randint(len(lst)))] for _ in range(64)]
        st_a, st_b = a.get_state(), b.get_state()
        same_pos = st_a[2] == st_b[2]
        same_buf = np.array_equal(st_a[1], st_b[1])
        check("k=%-3d values" % k, ra == rb)
        check("k=%-3d rng 内部位置 state[2]=%d vs %d" % (k, st_a[2], st_b[2]), same_pos)
        check("k=%-3d rng 状态缓冲逐位相同" % k, same_buf)

    print()
    print("[2] rng.randint(0,2,size=n) 是否有不触发 np.prod 的等价写法")

    n = 150
    a = np.random.RandomState(999)
    base = a.randint(0, 2, size=n)
    # 候选：逐标量（慢但可能等价）
    b = np.random.RandomState(999)
    scalar = np.array([b.randint(2) for _ in range(n)], dtype=base.dtype)
    check("标量循环逐位相同", np.array_equal(base, scalar))
    # 候选：randint(2**31, size=n) & 1 —— 大概率不等价，只作对照
    c = np.random.RandomState(999)
    alt = c.randint(0, 2 ** 31, size=n) & 1
    print("  %-58s %s" % ("(对照) randint(2**31)&1 相同？",
                          "同" if np.array_equal(base, alt) else "不同（不可替换）"))
    # 候选：rng.tomaxint / random_sample 切分
    d = np.random.RandomState(999)
    alt2 = (d.random_sample(n) < 0.5).astype(np.int64)
    print("  %-58s %s" % ("(对照) random_sample<0.5 相同？",
                          "同" if np.array_equal(base, alt2) else "不同（不可替换）"))

    print()
    print("结论:", "全部等价" if OK else "存在不等价项")
    print("注：只要 [1] 通过，`_repair_ma_for_os` 的回退分支就可以安全替换。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
