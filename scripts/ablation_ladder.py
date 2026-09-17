# -*- coding: utf-8 -*-
"""
论文 6 级消融阶梯（Li et al., ESWA 203 (2022) 117380, §5.4）。

论文的阶梯是**逐级累加**，相邻两级之差正好隔离一个组件：

    RMOEA/D1  纯 MOEA/D                     mix3=0 elite=0 rvns=0 qpas=0
    RMOEA/D2  + MIX3 初始化                  mix3=1 elite=0 rvns=0 qpas=0
    RMOEA/D3  + 随机选择 VNS                 mix3=1 elite=0 rvns=1(mode=random) qpas=0
    RMOEA/D4  + Q-PAS                        mix3=1 elite=0 rvns=1(mode=random) qpas=1
    RMOEA/D5  + Elite archive                mix3=1 elite=1 rvns=1(mode=random) qpas=1
    RMOEA/D   = D5 把 VNS 换成 RVNS(RL 选算子) mix3=1 elite=1 rvns=1(mode=rl)      qpas=1

另附两条论文之外的对照臂，用于把「加上局部搜索本身」与「RL 引导选算子」分开
（论文阶梯里这两者混在 D5→RMOEA/D 那一步里）：

    D5_fixedT = D5 关掉 Q-PAS（等价于 fixed_T=10 的 RMOEA/D5）
    D4_fixedT = D4 关掉 Q-PAS

用法（必须用有 numpy 的解释器，例如 C:\\Python312\\python.exe）：

    # 单实例
    python scripts/ablation_ladder.py --instance Mk01 --seeds 30 --workers 6

    # 多个实例：**必须逐个列出**，实例名要真实存在于 data_dir
    # （不支持 "Mk01,...,Mk10" 这类省略写法——"..." 会被当成一个实例名）
    python scripts/ablation_ladder.py \\
        --instances Mk01,Mk02,Mk03,Mk04,Mk05,Mk06,Mk07,Mk08,Mk09,Mk10 \\
        --seeds 30 --workers 6

    # 更省事：用分批驱动脚本，它默认就是 Mk01~Mk10，且能扛住单批超时
    python scripts/ladder_run_all.py --seeds 30 --workers 6

    # 只跑某几条臂
    python scripts/ablation_ladder.py --instance Mk10 --arms D1,D2,D3 --seeds 30

输出 logs/ablation_ladder.json（list，每条一个 run），支持断点续跑：
每条 (instance, label, seed) 唯一，重跑会自动跳过已完成的组合。
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── 禁止 BLAS/MKL 内部多线程，避免与进程池冲突 ──
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# ══════════════════════════════════════════════════════════════════════
# 阶梯定义：label -> (说明, RMOEAD 构造参数)
# ══════════════════════════════════════════════════════════════════════
LADDER = [
    ("D1", "MOEA/D (random init)",            dict(enable_mix3=False, enable_elite=False,
                                                    enable_rvns=False, fixed_T=10)),
    ("D2", "+ MIX3 init",                     dict(enable_mix3=True,  enable_elite=False,
                                                    enable_rvns=False, fixed_T=10)),
    ("D3", "+ random-selection VNS",          dict(enable_mix3=True,  enable_elite=False,
                                                    enable_rvns=True, rvns_mode="random",
                                                    fixed_T=10)),
    ("D4", "+ Q-PAS",                         dict(enable_mix3=True,  enable_elite=False,
                                                    enable_rvns=True, rvns_mode="random",
                                                    fixed_T=None)),
    ("D5", "+ Elite archive",                 dict(enable_mix3=True,  enable_elite=True,
                                                    enable_rvns=True, rvns_mode="random",
                                                    fixed_T=None)),
    ("RMOEAD", "RMOEA/D (= D5 with RVNS)",    dict(enable_mix3=True,  enable_elite=True,
                                                    enable_rvns=True, rvns_mode="rl",
                                                    fixed_T=None)),
    # ── 论文之外的分离臂（把「加 LS」与「RL 选算子」分开）──
    ("D5_fixedT", "D5 without Q-PAS",         dict(enable_mix3=True,  enable_elite=True,
                                                    enable_rvns=True, rvns_mode="random",
                                                    fixed_T=10)),
    ("D4_fixedT", "D4 without Q-PAS",         dict(enable_mix3=True,  enable_elite=False,
                                                    enable_rvns=True, rvns_mode="random",
                                                    fixed_T=10)),
]

# 论文阶梯相邻两级之差所隔离的组件（用来说明审计的切分点）
STEPS = [
    ("D1", "D2", "MIX3 initialization"),
    ("D2", "D3", "random-selection VNS"),
    ("D3", "D4", "Q-PAS"),
    ("D4", "D5", "Elite archive"),
    ("D5", "RMOEAD", "RVNS (RL operator selection)"),
]

LABELS = [x[0] for x in LADDER]
DEFAULTS = dict(n_pop=100, max_gen=200, data_dir="data",
                seed_start=42, n_runs=30, workers=6)


def _run_one(job):
    """跑一条臂的一个 seed。返回可序列化的 dict。"""
    (instance, label, seed, n_pop, max_gen, data_dir, ref_point) = job
    import logging
    logging.disable(logging.WARNING)

    from rmoea_d.algorithm import RMOEAD
    from rmoea_d.utils.metrics import instance_hv_bounds, compute_hv

    kwargs = dict(LADDER_DICT[label])
    t0 = time.perf_counter()
    solver = RMOEAD(instance_name=instance, n_pop=n_pop, max_gen=max_gen,
                    seed=seed, data_dir=data_dir, **kwargs)
    res = solver.solve()
    dt = time.perf_counter() - t0

    # 用实例确定性边界算 HV（同一实例内所有臂可比）；参考集重标定在分析阶段做
    bounds = instance_hv_bounds(solver.instance)
    pf = [(d["Makespan"], d["Workload"]) for d in res["final_pf"]]
    hv = compute_hv(pf, norm_bounds=bounds) if pf else 0.0

    hist = res["history"]
    return {
        "instance": instance,
        "label": label,
        "seed": seed,
        "final_hv": float(hv),
        "total_time": float(dt),
        "pf_size": len(pf),
        "final_pf": [[float(a), float(b)] for a, b in pf],
        "hist_hv": [float(h["hv"]) for h in hist],
        "hist_T": ([h["T"] for h in hist] if "T" in hist[0] else None),
        "q_table": res.get("q_table"),
        "components": res.get("components"),
    }


LADDER_DICT = {lbl: kw for lbl, _, kw in LADDER}


def _available_instances(data_dir):
    """列出 ``data_dir`` 下真实存在的实例名（``<name>.fjs``）。

    用于**早失败**：实例名写错（典型是 "Mk01,...,Mk10" 里的 "..."）时，
    原来要等到每个 job 在子进程里抛异常、跑满整个批次之后才发现，
    而且因为 ``requested`` 永远凑不齐，结果文件**永远不会提升为主文件** ——
    表现是"跑了一整轮但什么都没落盘"。这里提前拦掉。
    """
    d = data_dir if os.path.isabs(data_dir) else os.path.join(ROOT, data_dir)
    if not os.path.isdir(d):
        return None
    return {os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".fjs")}


def _dump(rows, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False)
    os.replace(tmp, path)


def _load(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []


def _load_merged(*paths):
    """按 (instance, label, seed) 去重地合并多个结果文件。

    为什么必须合并而不是只读主文件：``done``（跳过判断）读 main + partial，
    而 ``_dump`` 写的是 ``rows``。如果 ``rows`` 只从主文件读，重启后
    「进行中实例」已完成的 run 会被跳过、却不出现在新写出的 partial 里 ——
    进度被静默丢掉，下一批从零重算。见 ``tests/test_refactor.py``
    的 ``TestLadderResumeMerge``。
    """
    seen = set()
    out = []
    for p in paths:
        for r in _load(p):
            k = (r.get("instance"), r.get("label"), r.get("seed"))
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description="论文 6 级消融阶梯实验台")
    ap.add_argument("--instance", default=None, help="单个实例名，如 Mk01")
    ap.add_argument("--instances", default=None,
                    help="逗号分隔的实例名列表；与 --instance 二选一")
    ap.add_argument("--arms", default=None,
                    help=f"逗号分隔的臂标签，默认全部。可选：{','.join(LABELS)}")
    ap.add_argument("--seeds", type=int, default=DEFAULTS["n_runs"],
                    help="每臂的独立运行次数（论文为 30）")
    ap.add_argument("--seed_start", type=int, default=DEFAULTS["seed_start"])
    ap.add_argument("--n_pop", type=int, default=DEFAULTS["n_pop"])
    ap.add_argument("--max_gen", type=int, default=DEFAULTS["max_gen"])
    ap.add_argument("--data_dir", default=DEFAULTS["data_dir"])
    ap.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "ablation_ladder.json"))
    args = ap.parse_args()

    instances = ([args.instance] if args.instance
                 else [s.strip() for s in (args.instances or "").split(",") if s.strip()])
    if not instances:
        ap.error("必须给 --instance 或 --instances")

    # 实例名早校验：拼错的话下面每个 job 都会在子进程里失败，
    # 而 requested 永远凑不齐 -> 结果永远不落盘（最坏情况白跑一整轮）。
    known = _available_instances(args.data_dir)
    if known is not None:
        bad = [i for i in instances if i not in known]
        if bad:
            hint = ""
            if any(("..." in b) or ("," in b) for b in bad):
                hint = ("；实例名要逐个列出，不支持 'Mk01,...,Mk10' 这类省略写法")
            ap.error(f"data_dir 里找不到实例 {bad}；可用 {sorted(known)}{hint}")

    arms = ([s.strip() for s in args.arms.split(",") if s.strip()] if args.arms
            else LABELS)
    unknown = [a for a in arms if a not in LADDER_DICT]
    if unknown:
        ap.error(f"未知臂: {unknown}；可选 {LABELS}")

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    print("=" * 78)
    print("论文 6 级消融阶梯（Li et al. 2022, §5.4）")
    print("=" * 78)
    for lbl, desc, kw in LADDER:
        if lbl in arms:
            flags = " ".join(f"{k}={v}" for k, v in kw.items())
            print(f"  {lbl:<10} {desc:<34} {flags}")
    print(f"\n  实例 {instances}")
    print(f"  每臂 {len(seeds)} runs（seed {seeds[0]}~{seeds[-1]}）")
    print(f"  总任务 = {len(instances) * len(arms) * len(seeds)}")
    print(f"  输出 {args.out}\n")

    done = {(r["instance"], r["label"], r["seed"]) for r in _load(args.out)}
    partial = args.out + ".partial.json"
    done |= {(r["instance"], r["label"], r["seed"]) for r in _load(partial)}
    if done:
        print(f"  已完成 {len(done)} 条，将跳过（断点续跑）\n")

    jobs = [(inst, lbl, sd, args.n_pop, args.max_gen, args.data_dir, None)
            for inst in instances for lbl in arms for sd in seeds
            if (inst, lbl, sd) not in done]

    if not jobs:
        print("没有待跑任务，直接进入分析。")
        # partial 若还在，说明上一轮是在收尾前被中断的，它比主文件更新
        _report(partial if os.path.exists(partial) else args.out)
        return 0

    # rows 必须**同时**包含 partial 里已有的进度：
    # `done`（上面的跳过判断）读的是 main + partial，而 `_dump` 写的是 rows。
    # 若 rows 只从主文件读，那么重启后「进行中实例」已完成的 run 虽然会被跳过，
    # 却不会出现在写出的 partial 里 —— 进度被静默丢掉，下一批从零重算。
    rows = _load_merged(args.out, partial)
    ctx = mp.get_context("spawn")
    t_start = time.perf_counter()
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = {ex.submit(_run_one, j): j for j in jobs}
        try:
            for fut in as_completed(futs):
                inst, lbl, sd = futs[fut][:3]
                try:
                    rows.append(fut.result())
                except Exception as exc:                       # noqa: BLE001
                    print(f"  FAIL {inst}/{lbl}/seed{sd}: "
                          f"{type(exc).__name__}: {exc}")
                n_done += 1
                if n_done % 10 == 0:
                    _dump(rows, partial)
                    el = time.perf_counter() - t_start
                    rate = n_done / el if el > 0 else 0
                    eta = (len(jobs) - n_done) / rate if rate > 0 else 0
                    print(f"  [{n_done}/{len(jobs)}] {inst}/{lbl}/seed{sd}  "
                          f"{el:.0f}s elapsed, ETA {eta:.0f}s", flush=True)
        finally:
            # 无论正常结束还是被中断（KeyboardInterrupt / 超时 / 进程被杀），
            # 已完成的 run 都必须落盘 —— 这是断点续跑唯一的进度来源。
            _dump(rows, partial)

    # ── 只有「本次请求的全部 (inst, arm, seed) 都完成」才提升为主结果文件。
    # 分批驱动时每批只覆盖一部分组合，此时若把 rows 写进 args.out，会掩盖
    # 「尚未跑」的任务（因为主文件里缺少 (inst,label,seed) 组合，下次仍会跑，
    # 但主文件内容看起来已经是完整的了），并让 partial 被删除、进度丢失。
    requested = {(i, l, s) for i in instances for l in arms for s in seeds}
    have = {(r["instance"], r["label"], r["seed"]) for r in rows}
    missing = requested - have

    if not missing:
        _dump(rows, args.out)
        if os.path.exists(partial):
            try:
                os.remove(partial)
            except OSError:
                pass
        print(f"\n全部完成：{len(rows)} 条 run，用时 {time.perf_counter() - t_start:.0f}s")
        print(f"已写入 {args.out}\n")
        _report(args.out)
    else:
        # 保留 partial，主文件不动。注意此时**新完成的 run 只在 partial 里**，
        # 就地分析必须读 partial，否则会拿着上一批的旧数据出报告
        # （上一行刚说"累计 N 条"，报告却按旧条数算）。
        n_missing_inst = len({m[0] for m in missing})
        print(f"\n本批完成 {n_done} 条，累计 {len(rows)} 条；"
              f"仍有 {len(missing)} 条待跑（涉及 {n_missing_inst} 个实例）")
        print(f"进度已保存在 {partial}（断点续跑将自动跳过已完成组合）\n")
        _report(partial)

    return 0


def _report(path):
    """跑完就地打印一次阶梯分析（完整分析见 scripts/ablation_ladder_analysis.py）。

    ``path`` 必须是**含最新进度**的那个文件：收尾成功时是主文件，
    分批未跑满时是 ``*.partial.json``。
    """
    import subprocess
    script = os.path.join(ROOT, "scripts", "ablation_ladder_analysis.py")
    if not os.path.exists(script) or not os.path.exists(path):
        return
    print("=" * 78)
    print(f"就地阶梯分析（数据源 {path}）")
    print("=" * 78)
    r = subprocess.run([sys.executable, script, "--lab_json", path],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    print(r.stdout or r.stderr)


if __name__ == "__main__":
    sys.exit(main())
