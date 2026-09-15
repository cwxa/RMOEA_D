#!/usr/bin/env python3
"""
RMOEA/D - Reinforcement Learning based MOEA/D for Bi-objective Fuzzy Flexible Job Shop Scheduling
基于强化学习的MOEA/D双目标模糊柔性作业车间调度求解器

Usage:
    python main.py optimize --instance Mk01 --n_pop 100 --max_gen 200
    python main.py analyze
    python main.py visualize
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RMOEA/D for Bi-objective Fuzzy Flexible Job Shop Scheduling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Optimize command
    opt_parser = subparsers.add_parser('optimize', help='Run optimization')
    opt_parser.add_argument(
        "--instance", type=str, default="Mk01",
        help="Brandimarte instance name (default: Mk01)"
    )
    opt_parser.add_argument(
        "--n_pop", type=int, default=100,
        help="Population size (default: 100)"
    )
    opt_parser.add_argument(
        "--max_gen", type=int, default=200,
        help="Maximum generations (default: 200)"
    )
    opt_parser.add_argument(
        "--crossover_rate", type=float, default=0.8,
        help="Crossover rate (default: 0.8)"
    )
    opt_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    opt_parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Data directory containing .fjs files (default: data)"
    )
    opt_parser.add_argument(
        "--ql_alpha", type=float, default=0.4,
        help="Q-learning learning rate (default: 0.4)"
    )
    opt_parser.add_argument(
        "--ql_gamma", type=float, default=0.6,
        help="Q-learning discount factor (default: 0.6)"
    )
    opt_parser.add_argument(
        "--ql_epsilon", type=float, default=0.8,
        help="Q-learning epsilon (default: 0.8)"
    )
    opt_parser.add_argument(
        "--ql_actions", type=int, nargs="+", default=[5, 10, 15, 20],
        help="Q-learning candidate T values (default: 5 10 15 20)"
    )
    opt_parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Output directory for results (default: results)"
    )
    opt_parser.add_argument(
        "--log_dir", type=str, default="logs",
        help="Log directory (default: logs)"
    )
    opt_parser.add_argument(
        "--timeout", type=float, default=None,
        help="Maximum wall-clock time in seconds (None = no limit)"
    )
    
    # Analyze command
    subparsers.add_parser('analyze', help='Analyze experiment results')
    
    # Visualize command
    subparsers.add_parser('visualize', help='Generate visualization charts')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run benchmark comparison (RMOEA/D vs MOEA/D)')
    bench_parser.add_argument(
        "--instances", type=str, nargs="+", default=["Mk01"],
        help="Instance names (default: Mk01)"
    )
    bench_parser.add_argument(
        "--n_pop", type=int, default=100,
        help="Population size (default: 100)"
    )
    bench_parser.add_argument(
        "--max_gen", type=int, default=200,
        help="Maximum generations (default: 200)"
    )
    bench_parser.add_argument(
        "--n_runs", type=int, default=30,
        help="Number of independent runs per instance (default: 30)"
    )
    bench_parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)"
    )
    bench_parser.add_argument(
        "--crossover_rate", type=float, default=0.8,
        help="Crossover rate (default: 0.8)"
    )
    bench_parser.add_argument(
        "--fixed_T", type=int, default=10,
        help="Fixed T value for MOEA/D (default: 10)"
    )
    bench_parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Data directory containing .fjs files (default: data)"
    )
    bench_parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Output directory for results (default: results)"
    )
    bench_parser.add_argument(
        "--timeout_per_task", type=float, default=None,
        help="Per-task timeout in seconds (None = no limit)"
    )
    
    # Ablation command
    ablation_parser = subparsers.add_parser('ablation', help='Run ablation study (via unified experiment)')
    ablation_parser.add_argument(
        "--instances", type=str, nargs="+", default=["Mk01", "Mk02"],
        help="Instance names (default: Mk01 Mk02)"
    )
    ablation_parser.add_argument(
        "--n_pop", type=int, default=100,
        help="Population size (default: 100)"
    )
    ablation_parser.add_argument(
        "--max_gen", type=int, default=200,
        help="Maximum generations (default: 200)"
    )
    ablation_parser.add_argument(
        "--n_runs", type=int, default=30,
        help="Number of repetitions per configuration (default: 30)"
    )
    ablation_parser.add_argument("--seed", type=int, default=42, help="随机种子")
    ablation_parser.add_argument("--crossover_rate", type=float, default=0.8)
    ablation_parser.add_argument("--fixed_T", type=int, default=10)
    ablation_parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Output directory for results (default: results)"
    )
    ablation_parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Data directory containing .fjs files (default: data)"
    )
    ablation_parser.add_argument(
        "--timeout_per_task", type=float, default=None,
        help="Per-task timeout in seconds (None = no limit)"
    )
    
    # Run-all command (一键并行全流程 — unified experiment)
    runall_parser = subparsers.add_parser(
        'run_all', help='一键并行: 统一实验 (benchmark+ablation 一次产出) + visualization'
    )
    runall_parser.add_argument(
        "--instances", type=str, nargs="+",
        default=["mk01", "mk02", "mk03", "mk04", "mk05",
                 "mk06", "mk07", "mk08", "mk09", "mk10"],
        help="MK实例列表 (默认: 全部10个)"
    )
    runall_parser.add_argument("--n_pop", type=int, default=100)
    runall_parser.add_argument("--max_gen", type=int, default=200)
    runall_parser.add_argument("--n_runs", type=int, default=30,
                               help="每实例独立运行次数 (每个seed同时产出4种算法变体)")
    runall_parser.add_argument("--seed", type=int, default=42)
    runall_parser.add_argument("--crossover_rate", type=float, default=0.8)
    runall_parser.add_argument("--fixed_T", type=int, default=10)
    runall_parser.add_argument("--data_dir", type=str, default="data")
    runall_parser.add_argument("--output_dir", type=str, default="results")
    runall_parser.add_argument("--skip_viz", action="store_true")
    runall_parser.add_argument("--skip_gantt", action="store_true")
    runall_parser.add_argument("--max_workers", type=int, default=4,
                               help="外层并行进程数 (MK实例间, 默认4)")
    runall_parser.add_argument("--n_workers_inner", type=int, default=None,
                               help="内层并行进程数 (n_runs间, 默认auto)")
    runall_parser.add_argument("--timeout_per_task", type=float, default=None,
                               help="Per-task timeout in seconds (None = no limit)")
    
    return parser.parse_args()


def run_optimize(args):
    """Run single instance optimization."""
    from rmoea_d.algorithm import RMOEAD
    from rmoea_d.utils.logger_setup import setup_logging
    
    setup_logging(log_dir=args.log_dir)
    
    solver = RMOEAD(
        instance_name=args.instance,
        n_pop=args.n_pop,
        max_gen=args.max_gen,
        crossover_rate=args.crossover_rate,
        seed=args.seed,
        data_dir=args.data_dir,
        ql_alpha=args.ql_alpha,
        ql_gamma=args.ql_gamma,
        ql_epsilon=args.ql_epsilon,
        ql_actions=args.ql_actions,
        timeout=args.timeout,
    )
    
    results = solver.solve()
    solver.save_results(results, output_dir=args.output_dir)
    
    print("\n" + "=" * 60)
    print("FINAL PARETO FRONT")
    print("=" * 60)
    print(f"{'Index':<8} {'Makespan':<15} {'Workload':<15}")
    print("-" * 60)
    for i, item in enumerate(results["final_pf"], 1):
        if isinstance(item, dict):
            f1, f2 = item["Makespan"], item["Workload"]
        else:
            f1, f2 = item[0], item[1]
        print(f"{i:<8} {f1:<15.4f} {f2:<15.4f}")
    print("=" * 60)
    print(f"Hypervolume (HV): {results['final_hv']:.6f}")
    print(f"Total runtime: {results['total_time']:.2f} s")
    print("=" * 60)


def run_analyze(args):
    """Run result analysis."""
    from rmoea_d.utils.analysis_report import main as analyze_main
    analyze_main()


def run_visualize(args):
    """Run visualization."""
    from rmoea_d.utils.logger_setup import setup_logging
    setup_logging(log_dir="logs")
    from rmoea_d.utils.visualization import main as visualize_main
    visualize_main()


def run_benchmark(args):
    """Run benchmark comparison (RMOEA/D vs MOEA/D) via unified experiment."""
    from rmoea_d.utils.experiment import main as experiment_main

    sys.argv = ['experiment.py',
                '--instances'] + args.instances + \
               ['--n_pop', str(args.n_pop),
                '--max_gen', str(args.max_gen),
                '--n_runs', str(args.n_runs),
                '--seed', str(args.seed),
                '--crossover_rate', str(args.crossover_rate),
                '--fixed_T', str(args.fixed_T),
                '--data_dir', args.data_dir,
                '--output_dir', args.output_dir,
                '--study', 'benchmark']
    if args.timeout_per_task is not None:
        sys.argv.extend(['--timeout_per_task', str(args.timeout_per_task)])
    experiment_main()


def run_ablation(args):
    """Run ablation study via unified experiment."""
    from rmoea_d.utils.experiment import main as experiment_main

    sys.argv = ['experiment.py',
                '--instances'] + args.instances + \
               ['--n_pop', str(args.n_pop),
                '--max_gen', str(args.max_gen),
                '--n_runs', str(args.n_runs),
                '--seed', str(args.seed),
                '--crossover_rate', str(args.crossover_rate),
                '--fixed_T', str(args.fixed_T),
                '--data_dir', args.data_dir,
                '--output_dir', args.output_dir,
                '--study', 'ablation']
    if args.timeout_per_task is not None:
        sys.argv.extend(['--timeout_per_task', str(args.timeout_per_task)])
    experiment_main()


def run_run_all(args):
    """一键并行全流程: 统一实验 (benchmark+ablation 一次产出) + visualization."""
    import sys
    argv = ['run_all.py', '--instances'] + args.instances + \
           ['--n_pop', str(args.n_pop), '--max_gen', str(args.max_gen),
            '--n_runs', str(args.n_runs),
            '--seed', str(args.seed), '--crossover_rate', str(args.crossover_rate),
            '--fixed_T', str(args.fixed_T), '--data_dir', args.data_dir,
            '--output_dir', args.output_dir, '--max_workers', str(args.max_workers)]
    if args.n_workers_inner is not None:
        argv.extend(['--n_workers_inner', str(args.n_workers_inner)])
    if args.timeout_per_task is not None:
        argv.extend(['--timeout_per_task', str(args.timeout_per_task)])
    if args.skip_viz:
        argv.append('--skip_viz')
    if getattr(args, 'skip_gantt', False):
        argv.append('--skip_gantt')
    sys.argv = argv
    from rmoea_d.utils.run_all import main as runall_main
    runall_main()


def main():
    args = parse_args()
    
    if args.command == 'optimize':
        run_optimize(args)
    elif args.command == 'analyze':
        run_analyze(args)
    elif args.command == 'visualize':
        run_visualize(args)
    elif args.command == 'benchmark':
        run_benchmark(args)
    elif args.command == 'ablation':
        run_ablation(args)
    elif args.command == 'run_all':
        run_run_all(args)
    else:
        print("Usage: python main.py [optimize|analyze|visualize|benchmark|ablation|run_all]")


if __name__ == "__main__":
    main()