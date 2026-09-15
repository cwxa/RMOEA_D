"""
一键并行运行所有MK实例的统一实验 (Benchmark + Ablation 一次产出) + 可视化 + 甘特图
用法:
    python src/rmoea_d/utils/run_all.py                          # 默认: 全10实例, 30次运行
    python src/rmoea_d/utils/run_all.py --instances mk01 mk02    # 仅MK01,MK02
    python src/rmoea_d/utils/run_all.py --skip_viz               # 只跑实验，不画图
    python src/rmoea_d/utils/run_all.py --max_workers 6          # 6个并行进程

架构: 两级并行 (v4.0 — unified)
  - 外层 ProcessPoolExecutor: 实例间并行 (max_workers 个子进程)
  - 内层 ProcessPoolExecutor: 实例内多轮运行并行 (experiment.py 内部)
  - 每个 seed 同时产出 4 种算法变体: rmoea_d, moea_d, qpas_only, rvns_only
  - 一次运行 → benchmark + ablation 两份数据
  - v4.0: 统一脚本，不再需要 benchmark.py + ablation.py 分步执行和注入
"""
import subprocess, sys, os, time, argparse, logging, glob, json, traceback, shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────────────
MK_INSTANCES = ["mk01", "mk02", "mk03", "mk04", "mk05",
                "mk06", "mk07", "mk08", "mk09", "mk10"]

UTILS_DIR = Path(__file__).resolve().parent  # src/rmoea_d/utils/
PROJECT_ROOT = UTILS_DIR.parent.parent.parent  # RMOEA_D/

# 算法变体定义
BENCHMARK_ALGOS = ["rmoea_d", "moea_d"]
ABLATION_ALGOS  = ["rmoea_d", "qpas_only", "rvns_only", "moea_d"]
ALGO_ALIAS = {"rmoea_d": "full"}  # rmoea_d 在消融中别名 "full"


# ═══════════════════════════════════════════════════════════
# 实例级任务函数：统一实验
# ═══════════════════════════════════════════════════════════

def run_unified_experiment_instance(instance, n_pop, max_gen, n_runs, seed,
                                    crossover_rate, fixed_T, data_dir, output_dir,
                                    n_workers_inner, exp_id, timeout_per_task=None):
    """直接调用 experiment.run_experiment，每个 seed 产出 4 种算法变体。
    
    一个函数替代原来的 benchmark + ablation 两个独立流程。
    完成后自动写入 legacy 格式的文件供可视化脚本使用。
    """
    from rmoea_d.utils.experiment import run_experiment, run_statistical_tests, print_report
    from rmoea_d.utils.logger_setup import setup_logging

    # ProcessPool worker 需独立初始化日志
    setup_logging(log_dir="logs")

    t0 = time.time()
    try:
        all_runs, agg = run_experiment(
            instances=[instance],
            n_pop=n_pop, max_gen=max_gen, n_runs=n_runs,
            base_seed=seed,
            crossover_rate=crossover_rate, fixed_T=fixed_T,
            data_dir=data_dir, output_dir=output_dir,
            n_workers=n_workers_inner,
            exp_id=exp_id,
            timeout_per_task=timeout_per_task,
        )

        # 统计检验 + 控制台报告
        if n_runs > 2:
            stats = run_statistical_tests(agg)
        else:
            stats = {"benchmark": {}, "ablation": {}}
        print_report(agg, stats)

        # ── 写入 legacy 格式文件（供可视化脚本兼容）──
        n_legacy = _write_legacy_files(
            all_runs, agg, instance, n_pop, max_gen, n_runs, seed,
            crossover_rate, fixed_T, output_dir, exp_id
        )

        elapsed = time.time() - t0
        return {
            "tag": f"unified/{instance}",
            "success": True,
            "elapsed": elapsed,
            "instance": instance,
            "type": "unified",
            "agg": agg,
            "n_legacy_files": n_legacy,
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "tag": f"unified/{instance}",
            "success": False,
            "elapsed": elapsed,
            "stderr_tail": traceback.format_exc(),
            "instance": instance,
            "type": "unified",
        }


# ═══════════════════════════════════════════════════════════
# Legacy 格式写入：兼容现有可视化脚本
# ═══════════════════════════════════════════════════════════

def _json_default_serialize(obj):
    """JSON 序列化：处理 NumPy 类型和其他特殊类型。"""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    elif isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, '__dict__'):
        try:
            return obj.__dict__
        except:
            return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _atomic_write_json(data, filepath):
    """原子写入 JSON：先写临时文件再 rename，避免写入中断导致文件损坏。"""
    import tempfile
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False,
                                     encoding='utf-8', dir=os.path.dirname(filepath)) as tmp_f:
        json.dump(data, tmp_f, indent=2, ensure_ascii=False, default=_json_default_serialize)
        tmp_path = tmp_f.name
    shutil.move(tmp_path, filepath)


def _write_legacy_files(all_runs, agg, instance, n_pop, max_gen, n_runs, seed,
                        crossover_rate, fixed_T, output_dir, exp_id):
    """将统一实验数据写入 legacy 格式文件，兼容现有可视化脚本。

    产出:
      results/benchmark/{inst}/RMOEA_D/*.json     # benchmark 单算法 per-run
      results/benchmark/{inst}/MOEA_D/*.json      # benchmark 单算法 per-run
      results/benchmark/benchmark_aggregate_{exp_id}.json  # benchmark 聚合
      results/ablation/{inst}/ablation_results_{exp_id}.json  # ablation 聚合
      results/schedules/{inst}/{inst}_full_schedule_run{idx}_{exp_id}.json  # 甘特图
    """
    import numpy as np
    n_written = 0

    runs = all_runs.get(instance, [])
    if not runs:
        return 0

    # ── 1. Benchmark: per-algorithm per-run JSON ──
    bench_dir = os.path.join(output_dir, "benchmark", instance)
    for algo in BENCHMARK_ALGOS:
        algo_dir = os.path.join(bench_dir, "RMOEA_D" if algo == "rmoea_d" else "MOEA_D")
        os.makedirs(algo_dir, exist_ok=True)
        for run_idx, run_data in enumerate(runs):
            if algo not in run_data:
                continue
            entry = run_data[algo]
            entry["instance"] = instance
            entry["seed"] = seed + run_idx
            entry["n_pop"] = n_pop
            entry["max_gen"] = max_gen
            path = os.path.join(algo_dir,
                f"{instance}_{'RMOEA_D' if algo == 'rmoea_d' else 'MOEA_D'}_Np{n_pop}_G{max_gen}_run{run_idx}_{exp_id}.json")
            _atomic_write_json(entry, path)
            n_written += 1

    # ── 2. Benchmark: 聚合 JSON (不再每实例写入——避免覆盖) ──
    # 聚合数据由 main() 在 Phase 1 完成后，从各实例的 agg 中合并写入
    # 见 _write_full_benchmark_aggregate()

    # ── 3. Ablation: 兼容 ablation_visualization.py 格式 ──
    # 格式: {inst: {algo: [run_dict, ...], ...}}
    # 每个 run_dict 含 final_hv, best_makespan, best_workload, total_time, pf_size, convergence
    abl_data = {}
    abl_data[instance] = {}
    for run_idx, run_data in enumerate(runs):
        for a in ABLATION_ALGOS:
            if a not in run_data:
                continue
            entry = run_data[a]
            alias = ALGO_ALIAS.get(a, a)  # rmoea_d → full
            if alias not in abl_data[instance]:
                abl_data[instance][alias] = []
            abl_data[instance][alias].append({
                "algorithm": alias,
                "instance": instance,
                "n_pop": n_pop, "max_gen": max_gen,
                "seed": seed + run_idx,
                "total_time": entry.get("total_time", 0),
                "final_hv": entry.get("final_hv", 0),
                "pf_size": entry.get("pf_size", 0),
                "best_makespan": entry.get("best_makespan", 0),
                "best_workload": entry.get("best_workload", 0),
                "avg_makespan": entry.get("avg_makespan", 0),
                "avg_workload": entry.get("avg_workload", 0),
                "convergence": entry.get("history", []),
                "parameters": entry.get("parameters", {}),
                "_source": "unified_experiment",
            })

    abl_dir = os.path.join(output_dir, "ablation", instance)
    os.makedirs(abl_dir, exist_ok=True)
    abl_path = os.path.join(abl_dir, f"ablation_results_{exp_id}.json")
    _atomic_write_json(abl_data, abl_path)
    n_written += 1

    # ── 4. 甘特图调度数据 ──
    sched_dir = os.path.join(output_dir, "schedules", instance)
    os.makedirs(sched_dir, exist_ok=True)
    for run_idx, run_data in enumerate(runs):
        rmoea = run_data.get("rmoea_d", {})
        schedules = rmoea.get("schedules", [])
        if schedules:
            sched_path = os.path.join(sched_dir,
                f"{instance}_full_schedule_run{run_idx}_{exp_id}.json")
            sched_data = {
                "instance": instance,
                "seed": seed + run_idx,
                "n_pop": n_pop, "max_gen": max_gen,
                "schedules": schedules,
                "final_pf": rmoea.get("final_pf", []),
                "final_hv": rmoea.get("final_hv", 0),
            }
            _atomic_write_json(sched_data, sched_path)
            n_written += 1

    return n_written


# ═══════════════════════════════════════════════════════════
# Viz 脚本启动器 (保留 subprocess，因为 viz 脚本是独立模块)
# ═══════════════════════════════════════════════════════════

def _run_subprocess(cmd_args, tag, cwd=None):
    """通用子进程启动器，用于 viz 等独立脚本"""
    if cwd is None:
        cwd = str(PROJECT_ROOT)
    t0 = time.time()
    logger.debug("[%s] Launching: %s", tag, ' '.join(cmd_args))
    proc = subprocess.Popen(
        cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding='utf-8', errors='replace', cwd=cwd
    )
    stdout, stderr = proc.communicate()
    elapsed = time.time() - t0
    if stdout:
        for line in stdout.strip().split('\n'):
            logger.debug("[%s] %s", tag, line)
    return {
        "tag": tag,
        "success": proc.returncode == 0,
        "elapsed": elapsed,
        "stderr_tail": (stderr or "")[-300:],
    }


def run_visualization_scripts():
    """运行可视化脚本（现在由 visualization.py 统一生成 benchmark + ablation 两张 TFN 表格图）。"""
    results = []
    for script in ["visualization.py"]:
        r = _run_subprocess(
            [sys.executable, str(UTILS_DIR / script)],
            f"viz/{script}"
        )
        r["script"] = script
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════
# 甘特图并行生成
# ═══════════════════════════════════════════════════════════

def _gantt_from_json(json_path, output_dir=None):
    """单个JSON文件的甘特图生成 (供 ThreadPoolExecutor 调度)

    Args:
        json_path: 调度 JSON 文件路径
        output_dir: 甘特图输出根目录 (None=使用默认 charts/schedules/)
    """
    import json as _json, re
    from rmoea_d.utils.visualization import generate_gantt_from_results

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = _json.load(f)
    except Exception as e:
        return json_path, False, str(e)

    if "schedules" not in data or not data["schedules"]:
        return json_path, False, "no schedule data"

    m = re.search(r'_(run\d+)', os.path.basename(json_path))
    run_tag = m.group(1) if m else ''

    try:
        # 传入 output_dir=None 让 generate_gantt_from_results 使用默认的 charts/schedules/{inst}/
        paths = generate_gantt_from_results(data, output_dir=output_dir, tag=run_tag)
        return json_path, True, len(paths)
    except Exception as e:
        return json_path, False, str(e)


def generate_gantt_for_benchmark(instances, sched_dir="results",
                                  max_workers=None):
    """为实验结果并行生成甘特图 (ThreadPoolExecutor)。

    甘特图统一输出到 charts/schedules/{inst}/ (由 visualize.py 的 GANNT_DIR 控制)。

    Args:
        instances: 实例名列表
        sched_dir: 调度 JSON 源目录 (默认 results/schedules/{inst}/)
        max_workers: 线程池大小
    """
    json_files = []
    sched_base = os.path.join(sched_dir, "schedules")
    for inst in instances:
        inst_sched_dir = os.path.join(sched_base, inst)
        if not os.path.exists(inst_sched_dir):
            continue
        for name_variant in (inst, inst.upper(), inst.lower(), inst.capitalize()):
            pattern = os.path.join(inst_sched_dir, f"{name_variant}_full_schedule_*.json")
            matches = glob.glob(pattern)
            if matches:
                json_files.extend(matches)
                break
        else:
            logger.warning("No schedule JSON found for %s in %s", inst.upper(), inst_sched_dir)

    if not json_files:
        logger.warning("No schedule JSON files found, skip Gantt.")
        return {}, 0

    logger.debug("Found %d schedule JSON files for Gantt generation", len(json_files))

    from concurrent.futures import ThreadPoolExecutor
    results = {}
    total = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_gantt_from_json, jf): jf for jf in json_files}
        for future in as_completed(futures):
            path, ok, info = future.result()
            results[path] = {"success": ok, "info": info}
            if ok:
                total += info
                logger.debug("Gantt OK: %s → %d charts", os.path.basename(path), info)
            else:
                logger.warning("Gantt FAIL: %s: %s", os.path.basename(path), info)
    return results, total


# ═══════════════════════════════════════════════════════════
# 全量 Benchmark 聚合写入 (所有实例完成后统一写入)
# ═══════════════════════════════════════════════════════════

def _write_full_benchmark_aggregate(all_results, output_dir, exp_id, n_pop, max_gen, n_runs,
                                    base_seed, crossover_rate, fixed_T):
    """从所有实例的运行结果中收集 benchmark 数据，写入单一聚合 JSON。

    解决之前 _write_legacy_files 每实例调用时覆盖写入同一文件的 bug。
    """
    benchmark_instances = {}
    for r in all_results:
        if not r.get("success") or "agg" not in r:
            continue
        bm = r["agg"].get("benchmark", {})
        # 去掉 _hv_all, _bm_all, _bw_all 等运行时数组，减小 JSON 体积
        for inst_data in bm.values():
            for algo_data in inst_data.values():
                if isinstance(algo_data, dict):
                    for raw_key in ["_hv_all", "_bm_all", "_bw_all"]:
                        algo_data.pop(raw_key, None)
        benchmark_instances.update(bm)

    if not benchmark_instances:
        logger.warning("No benchmark data to aggregate.")
        return 0

    bm_merged = {
        "exp_id": exp_id,
        "timestamp": exp_id,
        "config": {
            "n_pop": n_pop, "max_gen": max_gen, "n_runs": n_runs,
            "base_seed": base_seed, "crossover_rate": crossover_rate, "fixed_T": fixed_T,
        },
        "instances": benchmark_instances,
        "statistical_tests": {},
    }
    merged_dir = os.path.join(output_dir, "benchmark")
    os.makedirs(merged_dir, exist_ok=True)
    merged_path = os.path.join(merged_dir, f"benchmark_aggregate_{exp_id}.json")
    _atomic_write_json(bm_merged, merged_path)
    logger.info("Full benchmark aggregate written: %d instances → %s",
                len(benchmark_instances), merged_path)
    return len(benchmark_instances)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="一键并行运行所有MK实例统一实验 + 自动生成可视化图表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                                   # 全10实例, 30 runs
  %(prog)s --instances mk01 mk02 mk03                        # 仅MK01-03
  %(prog)s --skip_viz                                        # 跳过可视化, 仅实验
  %(prog)s --skip_gantt                                      # 跳过甘特图
  %(prog)s --max_workers 6                                   # 6个并行进程
        """,
    )
    parser.add_argument("--instances", type=str, nargs="+", default=MK_INSTANCES,
                        help="MK实例列表 (默认: 全部10个)")
    parser.add_argument("--n_pop", type=int, default=100, help="种群大小")
    parser.add_argument("--max_gen", type=int, default=200, help="最大代数")
    parser.add_argument("--n_runs", type=int, default=30, help="每实例独立运行次数")
    parser.add_argument("--seed", type=int, default=42, help="基础随机种子")
    parser.add_argument("--crossover_rate", type=float, default=0.8, help="交叉率")
    parser.add_argument("--fixed_T", type=int, default=10, help="MOEA/D 固定邻域大小")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--output_dir", type=str, default="results", help="结果输出目录")
    parser.add_argument("--skip_viz", action="store_true", help="跳过可视化")
    parser.add_argument("--skip_gantt", action="store_true", help="跳过甘特图生成")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="外层并行进程数 (实例间并发, 默认4)")
    parser.add_argument("--n_workers_inner", type=int, default=None,
                        help="内层并行进程数 (实例内多轮运行, 默认 auto)")
    parser.add_argument("--timeout_per_task", type=float, default=None,
                        help="Per-task timeout in seconds (None = no limit)")
    parser.add_argument("--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()
    instances = [i.lower() for i in args.instances]

    # ── 初始化日志 ──
    from rmoea_d.utils.logger_setup import setup_logging
    setup_logging(log_dir="logs")

    # ── 两级并行 worker 数计算 ──
    total_cpus = os.cpu_count() or 8
    outer_workers = min(args.max_workers, len(instances))
    os.environ["RMOEA_PARENT_WORKERS"] = str(outer_workers)

    if args.n_workers_inner is not None:
        nw_inner = args.n_workers_inner
    else:
        nw_inner = max(1, min(args.n_runs, total_cpus // outer_workers))

    # ── 生成统一实验ID ──
    exp_id = f"rmoea_{time.strftime('%Y%m%d_%H%M%S')}"
    os.environ["RMOEA_EXP_ID"] = exp_id

    # ── 打印配置 ──────────────────────────────────────────
    total_start = time.time()
    logger.info("=" * 70)
    logger.info("RMOEA/D Full Pipeline v4.0 (Unified) | ExpID=%s | CPU=%d cores | MK=%s",
                exp_id, total_cpus, ','.join(i.upper() for i in instances))
    logger.info("Parallel: outer=%d | inner=%d | n_runs=%d | 4 algos per seed",
                outer_workers, nw_inner, args.n_runs)
    phases = [f"Unified Experiment(n={args.n_runs},g={args.max_gen})"]
    if not args.skip_viz:
        phases.append("Viz")
    if not args.skip_gantt:
        phases.append("Gantt")
    logger.info("Phases: %s", ' -> '.join(phases))
    logger.info("=" * 70)

    all_results = []
    n_phases = sum(1 for p in [True, not args.skip_viz, not args.skip_gantt] if p)

    # ══════════════════════════════════════════════════════════
    # Phase 1: Unified Experiment — 实例级并行
    #   每个 seed 同时产出 4 种算法变体，一次实验 = 两份数据
    # ══════════════════════════════════════════════════════════
    logger.info("[Phase 1/%d] Unified Experiment: %d instances x %d runs x 4 algos, outer=%d inner=%d",
                n_phases, len(instances), args.n_runs, outer_workers, nw_inner)
    t0 = time.time()
    completed = 0

    merged_agg = None
    with ProcessPoolExecutor(max_workers=outer_workers) as executor:
        futures = {
            executor.submit(
                run_unified_experiment_instance, inst,
                args.n_pop, args.max_gen, args.n_runs, args.seed,
                args.crossover_rate, args.fixed_T, args.data_dir,
                args.output_dir, nw_inner, exp_id,
                args.timeout_per_task,
            ): inst for inst in instances
        }
        for future in as_completed(futures):
            r = future.result()
            all_results.append(r)
            completed += 1
            icon = "OK" if r["success"] else "FAIL"
            elapsed_pct = (time.time() - t0)
            logger.info("[%s] %2d/%2d | %-6s | unified  | %.0fs",
                        icon, completed, len(instances),
                        r['instance'].upper(), elapsed_pct)
            if not r["success"] and r["stderr_tail"]:
                logger.error("  └─ %s: %s", r['instance'],
                             r['stderr_tail'][:200].replace('\n', ' '))

    phase_time = time.time() - t0
    n_ok = sum(1 for r in all_results if r["success"])
    logger.info("[Phase 1 done] %d/%d succeeded in %.0fs (%.1f min)",
                n_ok, len(instances), phase_time, phase_time / 60)

    # ── Phase 1b: 写入全量 benchmark_aggregate JSON (所有实例合并，避免覆盖) ──
    _write_full_benchmark_aggregate(
        all_results, args.output_dir, exp_id,
        args.n_pop, args.max_gen, args.n_runs, args.seed,
        args.crossover_rate, args.fixed_T,
    )

    # ══════════════════════════════════════════════════════════
    # Phase 2: Visualization
    # ══════════════════════════════════════════════════════════
    if not args.skip_viz:
        logger.info("[Phase 2/%d] Visualization", n_phases)
        for vr in run_visualization_scripts():
            icon = "OK" if vr["success"] else "FAIL"
            logger.info("[%s] %-30s %.0fs", icon, vr['script'], vr['elapsed'])

    # ══════════════════════════════════════════════════════════
    # Phase 2b: 完整统计分析报告
    # ══════════════════════════════════════════════════════════
    if not args.skip_viz:
        logger.info("[Phase 2b/%d] Statistical Analysis Report", n_phases)
        vr = _run_subprocess(
            [sys.executable, str(UTILS_DIR / "analysis_report.py")],
            "viz/analysis_report"
        )
        icon = "OK" if vr["success"] else "FAIL"
        logger.info("[%s] %-30s %.0fs", icon, "analysis_report.py", vr['elapsed'])

    # ══════════════════════════════════════════════════════════
    # Phase 3: Gantt Charts (并行渲染)
    # ══════════════════════════════════════════════════════════
    if not args.skip_gantt:
        logger.info("[Phase 3/%d] Gantt Charts (%d instances, threads=%d)",
                    n_phases, len(instances), min(8, total_cpus))
        t0 = time.time()
        gantt_results, total_gantt = generate_gantt_for_benchmark(
            instances,
            sched_dir="results",   # 调度数据从 results/schedules/ 读取
            max_workers=min(8, total_cpus),
        )
        logger.info("[Phase 3 done] %d Gantt charts in %.0fs", total_gantt, time.time() - t0)

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    failures = [r for r in all_results if not r["success"]]
    total = time.time() - total_start
    ok_count = len(all_results) - len(failures)
    logger.info("=" * 70)
    if failures:
        logger.warning("Pipeline complete! %d/%d succeeded (%.0fs / %.1f min) | FAILED: %s",
                       ok_count, len(all_results), total, total / 60,
                       ', '.join(r['instance'] + '/' + r['type'] for r in failures))
    else:
        logger.info("Pipeline complete! All %d succeeded (%.0fs / %.1f min)",
                    ok_count, total, total / 60)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()