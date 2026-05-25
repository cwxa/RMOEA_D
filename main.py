#!/usr/bin/env python3
"""
RMOEA/D - Reinforcement Learning based MOEA/D for Bi-objective Fuzzy Flexible Job Shop Scheduling
基于强化学习的MOEA/D双目标模糊柔性作业车间调度求解器

Usage:
    python main.py --instance Mk01 --n_pop 100 --max_gen 200 --seed 42
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rmoea_d.algorithm import RMOEAD
from rmoea_d.utils.logger_setup import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RMOEA/D for Bi-objective Fuzzy Flexible Job Shop Scheduling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--instance", type=str, default="Mk01",
        help="Brandimarte instance name (default: Mk01)"
    )
    parser.add_argument(
        "--n_pop", type=int, default=100,
        help="Population size (default: 100)"
    )
    parser.add_argument(
        "--max_gen", type=int, default=200,
        help="Maximum generations (default: 200)"
    )
    parser.add_argument(
        "--crossover_rate", type=float, default=0.8,
        help="Crossover rate (default: 0.8)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Data directory containing .fjs files (default: data)"
    )
    parser.add_argument(
        "--ql_alpha", type=float, default=0.4,
        help="Q-learning learning rate (default: 0.4)"
    )
    parser.add_argument(
        "--ql_gamma", type=float, default=0.6,
        help="Q-learning discount factor (default: 0.6)"
    )
    parser.add_argument(
        "--ql_epsilon", type=float, default=0.8,
        help="Q-learning epsilon (default: 0.8)"
    )
    parser.add_argument(
        "--ql_actions", type=int, nargs="+", default=[5, 10, 15, 20],
        help="Q-learning candidate T values (default: 5 10 15 20)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs",
        help="Log directory (default: logs)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    setup_logging(log_dir=args.log_dir)

    # Create solver
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
    )

    # Run optimization
    results = solver.solve()

    # Save results
    solver.save_results(results, output_dir=args.output_dir)

    # Print final Pareto front
    print("\n" + "=" * 60)
    print("FINAL PARETO FRONT")
    print("=" * 60)
    print(f"{'Index':<8} {'Makespan':<15} {'Workload':<15}")
    print("-" * 60)
    for i, (f1, f2) in enumerate(results["final_pf"], 1):
        print(f"{i:<8} {f1:<15.4f} {f2:<15.4f}")
    print("=" * 60)
    print(f"Hypervolume (HV): {results['final_hv']:.6f}")
    print(f"Total runtime: {results['total_time']:.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
