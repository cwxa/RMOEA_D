"""一键并行运行所有MK实例的benchmark + ablation + 可视化
用法:
    python src/rmoea_d/utils/run_all.py                          # 默认: 全10实例, 30次benchmark + 3次ablation
    python src/rmoea_d/utils/run_all.py --instances mk01 mk02    # 仅MK01,MK02
    python src/rmoea_d/utils/run_all.py --skip_ablation          # 只跑benchmark+可视化
    python src/rmoea_d/utils/run_all.py --max_workers 6          # 6个并行进程
"""
import subprocess, sys, os, time, argparse, logging
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


def run_benchmark_instance_all(instances, n_pop, max_gen, n_runs, seed,
                               crossover_rate, fixed_T, data_dir, output_dir,
                               n_workers_inner):
    """子进程: 一次运行所有MK实例的benchmark (内部ProcessPoolExecutor并行)"""
    cmd = [
        sys.executable,
        str(UTILS_DIR / "benchmark.py"),
        "--instances", *instances,
        "--n_pop", str(n_pop),
        "--max_gen", str(max_gen),
        "--n_runs", str(n_runs),
        "--seed", str(seed),
        "--crossover_rate", str(crossover_rate),
        "--fixed_T", str(fixed_T),
        "--data_dir", str(data_dir),
        "--output_dir", str(output_dir),
        "--n_workers", str(n_workers_inner),
    ]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=subprocess.PIPE,
                            text=True, encoding='utf-8', errors='replace',
                            cwd=str(PROJECT_ROOT))
    _, stderr = proc.communicate()
    elapsed = time.time() - t0
    return {
        "instance": "all", "type": "benchmark",
        "success": proc.returncode == 0, "elapsed": elapsed,
        "stderr_tail": (stderr or "")[-300:],
    }


def run_ablation_instance(instance, n_pop, max_gen, n_runs,
                          crossover_rate, data_dir, output_dir,
                          n_workers_inner):
    """子进程: 运行单个实例的ablation (内部并行 n_runs*4)"""
    cmd = [
        sys.executable,
        str(UTILS_DIR / "ablation.py"),
        "--instances", instance,
        "--n_pop", str(n_pop),
        "--max_gen", str(max_gen),
        "--n_runs", str(n_runs),
        "--crossover_rate", str(crossover_rate),
        "--data_dir", str(data_dir),
        "--output_dir", str(output_dir),
        "--n_workers", str(n_workers_inner),
    ]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=subprocess.PIPE,
                            text=True, encoding='utf-8', errors='replace',
                            cwd=str(PROJECT_ROOT))
    _, stderr = proc.communicate()
    elapsed = time.time() - t0
    return {
        "instance": instance, "type": "ablation",
        "success": proc.returncode == 0, "elapsed": elapsed,
        "stderr_tail": (stderr or "")[-300:],
    }


def run_visualization_scripts():
    """运行两个可视化脚本 + 自动生成甘特图"""
    results = []
    for script in ["visualization.py", "ablation_visualization.py"]:
        t0 = time.time()
        proc = subprocess.Popen(
            [sys.executable, str(UTILS_DIR / script)],
            stdout=sys.stdout, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace',
            cwd=str(PROJECT_ROOT)
        )
        _, _ = proc.communicate()
        results.append({
            "script": script,
            "success": proc.returncode == 0,
            "elapsed": time.time() - t0,
        })
    return results


def generate_gantt_for_benchmark(instances, output_dir="results"):
    """
    为benchmark结果自动生成甘特图。
    遍历每个实例的每个RMOEA/D运行结果JSON，生成甘特图到charts/schedules/<instance>/
    """
    from rmoea_d.utils.visualization import generate_gantt_from_results
    import json

    total_generated = 0
    total_failed = 0
    results = {}

    for inst in instances:
        logger.debug("Generating Gantt charts for %s...", inst.upper())
        rmoead_dir = os.path.join(output_dir, inst.lower(), "RMOEA_D")
        if not os.path.exists(rmoead_dir):
            logger.warning("RMOEA_D result directory not found: %s", rmoead_dir)
            continue

        # 查找所有RMOEA/D完整结果JSON文件（含调度数据）
        import glob
        pattern = os.path.join(rmoead_dir, f"{inst}_full_schedule_*.json")
        if not inst[0].islower():
            pattern_lower = os.path.join(rmoead_dir, f"{inst.lower()}_full_schedule_*.json")
            json_files = glob.glob(pattern_lower)
        else:
            pattern_upper = os.path.join(rmoead_dir, f"{inst.upper()}_full_schedule_*.json")
            json_files = glob.glob(pattern_upper)
        if not json_files:
            json_files = glob.glob(pattern)

        if not json_files:
            logger.warning("No result JSON found for %s in %s", inst.upper(), rmoead_dir)
            continue

        results[inst] = {"generated": 0, "failed": 0}
        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 检查是否有调度数据（我们新增的字段）
                if "schedules" in data and len(data["schedules"]) > 0:
                    gantt_paths = generate_gantt_from_results(data)
                    results[inst]["generated"] += len(gantt_paths)
                    total_generated += len(gantt_paths)
            except Exception as e:
                logger.warning("Failed to generate Gantt for %s: %s | file: %s",
                               inst.upper(), str(e), json_path)
                results[inst]["failed"] += 1
                total_failed += 1

    logger.info("Gantt chart generation complete: total %d generated, %d failed",
                total_generated, total_failed)
    return results, total_generated


def main():
    parser = argparse.ArgumentParser(
        description="一键并行运行所有MK实例实验 + 自动生成可视化图表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                                   # 全10实例, 30 runs benchmark + 3 runs ablation
  %(prog)s --instances mk01 mk02 mk03                        # 仅MK01-03
  %(prog)s --skip_ablation                                   # 跳过消融, 仅benchmark + viz
  %(prog)s --skip_benchmark --skip_viz                       # 仅消融实验
  %(prog)s --max_workers 8                                   # 8个并行进程
        """,
    )
    parser.add_argument("--instances", type=str, nargs="+", default=MK_INSTANCES,
                        help="MK实例列表 (默认: 全部10个)")
    parser.add_argument("--n_pop", type=int, default=100, help="种群大小")
    parser.add_argument("--max_gen", type=int, default=200, help="Benchmark最大代数")
    parser.add_argument("--max_gen_ablation", type=int, default=100, help="Ablation最大代数")
    parser.add_argument("--n_runs", type=int, default=30, help="Benchmark每实例独立运行次数")
    parser.add_argument("--n_runs_ablation", type=int, default=3, help="Ablation每实例重复次数")
    parser.add_argument("--seed", type=int, default=42, help="基础随机种子")
    parser.add_argument("--crossover_rate", type=float, default=0.8, help="交叉率")
    parser.add_argument("--fixed_T", type=int, default=10, help="MOEA/D固定邻域大小")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--output_dir", type=str, default="results", help="结果输出目录")
    parser.add_argument("--skip_benchmark", action="store_true", help="跳过对比实验")
    parser.add_argument("--skip_ablation", action="store_true", help="跳过消融实验")
    parser.add_argument("--skip_viz", action="store_true", help="跳过可视化")
    parser.add_argument("--skip_gantt", action="store_true", help="跳过甘特图生成")
    parser.add_argument("--max_workers", type=int, default=4, help="外层并行进程数 (MK实例间, 默认4)")
    parser.add_argument("--n_workers_inner", type=int, default=None,
                        help="内层并行进程数 (n_runs间, 默认auto)")
    parser.add_argument("--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()
    instances = [i.lower() for i in args.instances]

    # ── 初始化日志 ──
    from rmoea_d.utils.logger_setup import setup_logging
    setup_logging(log_dir="logs")

    # ── 自动计算内层并行数 ──
    total_cpus = os.cpu_count() or 8
    if args.n_workers_inner is None:
        # 内层 = total / 外层, 最少2, 最多n_runs
        args.n_workers_inner = max(2, min(args.n_runs, total_cpus // args.max_workers))

    # ── 打印配置(关键信息到控制台) ──────────────────────────
    total_start = time.time()
    logger.info("RMOEA/D Full Pipeline | CPU=%d | MK:%s | outer_w=%d inner_w=%d",
                total_cpus, ','.join(i.upper() for i in instances),
                args.max_workers, args.n_workers_inner)
    phases = []
    if not args.skip_benchmark:
        phases.append(f"Benchmark(n={args.n_runs},g={args.max_gen})")
    if not args.skip_ablation:
        phases.append(f"Ablation(n={args.n_runs_ablation},g={args.max_gen_ablation})")
    if not args.skip_viz:
        phases.append("Viz")
    if not args.skip_gantt:
        phases.append("Gantt")
    logger.info("Phases: %s", ' -> '.join(phases))

    all_results = []

    # ══════════════════════════════════════════════════════════
    # Phase 1: Benchmark (单次调用, 内部ProcessPoolExecutor并行所有实例)
    # ══════════════════════════════════════════════════════════
    if not args.skip_benchmark:
        logger.info("Phase 1/3: Benchmark (%d instances, %d runs each)", len(instances), args.n_runs)
        t0 = time.time()
        r = run_benchmark_instance_all(
            instances,
            n_pop=args.n_pop, max_gen=args.max_gen,
            n_runs=args.n_runs, seed=args.seed,
            crossover_rate=args.crossover_rate,
            fixed_T=args.fixed_T, data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_workers_inner=args.n_workers_inner,
        )
        all_results.append(r)
        icon = "OK" if r["success"] else "FAIL"
        logger.debug("[%s] All benchmark %.0fs", icon, r['elapsed'])
        if not r["success"] and r["stderr_tail"]:
            logger.error("Benchmark stderr: %s", r['stderr_tail'][:200])
        logger.info("Phase 1 done: %.0fs", time.time() - t0)

    # ══════════════════════════════════════════════════════════
    # Phase 2: Ablation (并行)
    # ══════════════════════════════════════════════════════════
    if not args.skip_ablation:
        logger.info("Phase 2/3: Ablation (%d instances)", len(instances))
        t0 = time.time()
        completed = 0
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    run_ablation_instance, inst,
                    n_pop=args.n_pop, max_gen=args.max_gen_ablation,
                    n_runs=args.n_runs_ablation,
                    crossover_rate=args.crossover_rate,
                    data_dir=args.data_dir, output_dir=os.path.join(args.output_dir, "ablation"),
                    n_workers_inner=args.n_workers_inner,
                ): inst for inst in instances
            }
            for future in as_completed(futures):
                r = future.result()
                all_results.append(r)
                completed += 1
                icon = "OK" if r["success"] else "FAIL"
                logger.debug("[%s] [%d/%d] %-5s ablation %.0fs",
                             icon, completed, len(instances),
                             r['instance'].upper(), r['elapsed'])
                if not r["success"] and r["stderr_tail"]:
                    logger.error("Ablation %s: %s", r['instance'], r['stderr_tail'][:200])
        logger.info("Phase 2 done: %.0fs", time.time() - t0)

    # ══════════════════════════════════════════════════════════
    # Phase 3: Visualization
    # ══════════════════════════════════════════════════════════
    if not args.skip_viz:
        logger.info("Phase 3/3: Visualization")
        for vr in run_visualization_scripts():
            icon = "OK" if vr["success"] else "FAIL"
            logger.info("[%s] %-30s %.0fs", icon, vr['script'], vr['elapsed'])

    # ══════════════════════════════════════════════════════════
    # Phase 4: Gantt Charts
    # ══════════════════════════════════════════════════════════
    if not args.skip_gantt:
        logger.info("Phase 4/4: Gantt Charts (%d instances)", len(instances))
        t0 = time.time()
        gantt_results, total_gantt = generate_gantt_for_benchmark(
            instances, output_dir=args.output_dir
        )
        logger.info("Phase 4 done: %d Gantt charts in %.0fs", total_gantt, time.time() - t0)

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    failures = [r for r in all_results if not r["success"]]
    total = time.time() - total_start
    logger.info("Pipeline complete! Total: %.0fs (%.1f min) | %d/%d succeeded",
                total, total / 60, len(all_results) - len(failures), len(all_results))
    if failures:
        logger.error("Failed: %s", ', '.join(r['instance'] + '/' + r['type'] for r in failures))


if __name__ == "__main__":
    main()