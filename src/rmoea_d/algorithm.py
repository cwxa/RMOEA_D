"""
RMOEA/D Algorithm: Main solver integrating Q-learning + MOEA/D.
RMOEA/D主算法：集成Q-learning参数自适应与MOEA/D演化框架。

核心流程：
1. 初始化种群（MIX3策略）
2. 初始化权重向量与Q-learning
3. 每代：Q-learning选择T → 更新邻居结构 → MOEA/D演化 → 更新Q-table
4. 维护精英存档，输出最终Pareto前沿
"""

import numpy as np
import time
import logging
import json
import os

from .core.instance import load_instance
from .core.operators import init_mix3
from .core.encoding import decode
from .core.moead import generate_weights, compute_neighbors, moead_generation
from .core.qlearning import QLearningPAS
from .utils.metrics import non_dominated_sort, compute_hv
from .core.fuzzy import fuzzy_dominates

logger = logging.getLogger(__name__)


class RMOEAD:
    """
    RMOEA/D solver for bi-objective fuzzy flexible job shop scheduling.
    面向双目标模糊柔性作业车间调度的RMOEA/D求解器。
    """

    def __init__(
        self,
        instance_name,
        n_pop=100,
        max_gen=200,
        crossover_rate=0.8,
        seed=42,
        data_dir="data",
        ql_alpha=0.4,
        ql_gamma=0.6,
        ql_epsilon=0.8,
        ql_actions=None,
    ):
        """
        Initialize RMOEA/D solver.
        初始化RMOEA/D求解器。

        Parameters:
            instance_name: Brandimarte instance name (e.g., "Mk01")
            n_pop: Population size (number of subproblems)
            max_gen: Maximum number of generations
            crossover_rate: Crossover probability
            seed: Random seed
            data_dir: Directory containing .fjs instance files
            ql_alpha: Q-learning learning rate
            ql_gamma: Q-learning discount factor
            ql_epsilon: Q-learning epsilon for exploration
            ql_actions: List of candidate T values
        """
        self.instance_name = instance_name
        self.n_pop = n_pop
        self.max_gen = max_gen
        self.crossover_rate = crossover_rate
        self.seed = seed
        self.data_dir = data_dir

        # Q-learning parameters
        self.ql_alpha = ql_alpha
        self.ql_gamma = ql_gamma
        self.ql_epsilon = ql_epsilon
        self.ql_actions = ql_actions if ql_actions is not None else [5, 10, 15, 20]

        # Internal state
        self.rng = np.random.RandomState(seed)
        self.instance = None
        self.weights = None
        self.ql = None
        self.archive = []  # Elite archive: list of (os, ma, obj)
        self.history = []  # Generation history for analysis

        logger.info("=" * 60)
        logger.info("RMOEA/D Solver Initialized")
        logger.info("Instance: %s, Np=%d, Gen=%d, CR=%.2f, Seed=%d",
                    instance_name, n_pop, max_gen, crossover_rate, seed)
        logger.info("Q-learning: alpha=%.2f, gamma=%.2f, epsilon=%.2f, actions=%s",
                    ql_alpha, ql_gamma, ql_epsilon, self.ql_actions)
        logger.info("=" * 60)

    def _init_population(self):
        """Initialize population using MIX3 strategy."""
        logger.info("Initializing population with MIX3 strategy...")
        start = time.perf_counter()
        pop = init_mix3(self.instance, self.n_pop, self.rng)
        objectives = []
        fuzzy_objs = []  # Store full fuzzy numbers for final output
        for os_vec, ma_vec in pop:
            m, w, mc, wc = decode(os_vec, ma_vec, self.instance)
            objectives.append((mc, wc))
            fuzzy_objs.append((m, w))
        elapsed = time.perf_counter() - start
        logger.info("Population initialized: %d individuals in %.4f s", len(pop), elapsed)
        return pop, objectives, fuzzy_objs

    def _compute_pf(self, objectives):
        """Extract non-dominated front from current objectives."""
        return non_dominated_sort(objectives)

    def _update_archive(self, population, objectives):
        """Update elite archive with non-dominated solutions."""
        # Combine archive and current population
        combined = self.archive + [
            (pop[0], pop[1], obj) for pop, obj in zip(population, objectives)
        ]
        # Extract objectives for sorting
        objs = [item[2] for item in combined]
        nd_objs = non_dominated_sort(objs)
        nd_set = set(nd_objs)
        # Filter archive to non-dominated solutions
        new_archive = []
        for item in combined:
            if item[2] in nd_set:
                new_archive.append(item)
                nd_set.remove(item[2])  # Avoid duplicates
        # Limit archive size
        if len(new_archive) > self.n_pop:
            # Keep first n_pop (could use crowding distance for better selection)
            new_archive = new_archive[:self.n_pop]
        self.archive = new_archive
        return len(new_archive)

    def solve(self):
        """
        Run the complete RMOEA/D algorithm.
        执行完整的RMOEA/D算法。

        Returns:
            dict: Results containing final PF, HV, and history
        """
        total_start = time.perf_counter()

        # Load instance
        self.instance = load_instance(self.instance_name, self.data_dir, self.seed)
        logger.info("Instance loaded: %d jobs, %d machines, %d operations",
                    self.instance["n_jobs"], self.instance["n_machines"],
                    self.instance["total_ops"])

        # Initialize weights
        self.weights = generate_weights(self.n_pop)
        logger.info("Weight vectors generated: %d vectors", len(self.weights))

        # Initialize Q-learning
        self.ql = QLearningPAS(
            alpha=self.ql_alpha,
            gamma=self.ql_gamma,
            epsilon=self.ql_epsilon,
            actions=self.ql_actions,
        )

        # Initialize population
        population, objectives, fuzzy_objectives = self._init_population()

        # Initialize reference point (crisp values for MOEA/D evolution)
        # 参考点使用清晰值，用于MOEA/D的标量化函数
        z = (
            min(o[0] for o in objectives),
            min(o[1] for o in objectives),
        )
        logger.info("Initial reference point: z=(%.4f, %.4f)", z[0], z[1])

        # Initial PF and archive (using crisp values for evolution)
        pf = self._compute_pf(objectives)
        self._update_archive(population, objectives)
        logger.info("Initial non-dominated front size: %d", len(pf))

        # Main loop
        for gen in range(1, self.max_gen + 1):
            gen_start = time.perf_counter()

            # Step 1: Q-learning selects T
            T, is_first = self.ql.step(pf, self.rng)
            logger.info("Generation %d/%d: Q-learning selected T=%d", gen, self.max_gen, T)

            # Step 2: Recompute neighbors with new T
            B = compute_neighbors(self.weights, T)

            # Step 3: MOEA/D generation
            population, objectives, z = moead_generation(
                population, objectives, self.weights, B,
                self.instance, z, self.crossover_rate, self.rng
            )

            # Step 4: Update PF and archive
            pf = self._compute_pf(objectives)
            archive_size = self._update_archive(population, objectives)

            # Step 5: Compute HV
            hv = compute_hv(pf)

            gen_time = time.perf_counter() - gen_start

            # Compute per-objective statistics from current PF
            # 从当前Pareto前沿计算各目标的统计值
            pf_array = np.array(pf)
            best_makespan = float(np.min(pf_array[:, 0]))
            best_workload = float(np.min(pf_array[:, 1]))
            avg_makespan = float(np.mean(pf_array[:, 0]))
            avg_workload = float(np.mean(pf_array[:, 1]))

            # Record history
            self.history.append({
                "gen": gen,
                "T": T,
                "pf_size": len(pf),
                "archive_size": archive_size,
                "hv": hv,
                "z": z,
                "best_makespan": best_makespan,
                "best_workload": best_workload,
                "avg_makespan": avg_makespan,
                "avg_workload": avg_workload,
                "time": gen_time,
            })

            # Periodic logging
            if gen % 20 == 0 or gen == 1:
                logger.info("Gen %d | T=%d | PF=%d | Archive=%d | HV=%.6f | z=(%.2f,%.2f) | Time=%.3fs",
                            gen, T, len(pf), archive_size, hv, z[0], z[1], gen_time)

        total_time = time.perf_counter() - total_start

        # Re-decode archive solutions to get fuzzy objectives for final output
        # 对存档中的解重新解码，获取完整的模糊目标值用于最终输出
        fuzzy_archive = []
        for os_vec, ma_vec, _ in self.archive:
            fm, fw, _, _ = decode(os_vec, ma_vec, self.instance)
            fuzzy_archive.append((fm.to_tuple(), fw.to_tuple()))

        # Final results (crisp PF for HV computation)
        final_pf = [(item[2][0], item[2][1]) for item in self.archive]
        final_pf = non_dominated_sort(final_pf)
        final_hv = compute_hv(final_pf)

        # Fuzzy PF for output using paper's ranking-based dominance
        # 使用论文排序算子进行模糊非支配排序
        fuzzy_pf = []
        for i, a in enumerate(fuzzy_archive):
            dominated = False
            for j, b in enumerate(fuzzy_archive):
                if i != j and fuzzy_dominates(b, a):
                    dominated = True
                    break
            if not dominated:
                fuzzy_pf.append(a)

        logger.info("=" * 60)
        logger.info("Optimization completed in %.4f s", total_time)
        logger.info("Final non-dominated front size: %d", len(final_pf))
        logger.info("Final HV: %.6f", final_hv)
        logger.info("=" * 60)

        # Format PF with objective names for readability
        # 为Pareto前沿添加目标名称，便于查看
        named_pf = [
            {"Makespan": m, "Workload": w}
            for m, w in final_pf
        ]
        named_fuzzy_pf = [
            {"Makespan": {"t1": fm[0], "t2": fm[1], "t3": fm[2]},
             "Workload": {"t1": fw[0], "t2": fw[1], "t3": fw[2]}}
            for fm, fw in fuzzy_pf
        ]

        results = {
            "algorithm": "RMOEA/D",
            "instance": self.instance_name,
            "n_pop": self.n_pop,
            "max_gen": self.max_gen,
            "crossover_rate": self.crossover_rate,
            "seed": self.seed,
            "final_pf": named_pf,
            "fuzzy_pf": named_fuzzy_pf,
            "final_hv": final_hv,
            "total_time": total_time,
            "history": self.history,
            "q_table": self.ql.q_table.tolist(),
        }
        return results

    def save_results(self, results, output_dir="results"):
        """Save results to JSON file with grouped directory structure.
        按实例和算法分组存储结果文件：results/Mk01/RMOEA_D/xxx.json
        """
        # Grouped directory: results/<instance>/<algorithm>/
        algo_dir = os.path.join(output_dir, self.instance_name, "RMOEA_D")
        os.makedirs(algo_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.instance_name}_RMOEA_D_Np{self.n_pop}_G{self.max_gen}_{timestamp}.json"
        filepath = os.path.join(algo_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info("Results saved to: %s", filepath)
        return filepath
