"""
Baseline MOEA/D solver without Q-learning (fixed T).
纯MOEA/D对比算法：固定邻域大小T，无Q-learning参数自适应。

用于与RMOEA/D进行对照实验，验证Q-learning自适应策略的有效性。
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
from .utils.metrics import non_dominated_sort, compute_hv
from .core.fuzzy import fuzzy_dominates

logger = logging.getLogger(__name__)


class MOEADBaseline:
    """
    Standard MOEA/D solver with fixed neighborhood size T.
    标准MOEA/D求解器，使用固定邻域大小T。
    """

    def __init__(
        self,
        instance_name,
        n_pop=100,
        max_gen=200,
        crossover_rate=0.8,
        fixed_T=10,
        seed=42,
        data_dir="data",
    ):
        """
        Initialize baseline MOEA/D solver.
        初始化标准MOEA/D求解器。

        Parameters:
            instance_name: Brandimarte instance name
            n_pop: Population size
            max_gen: Maximum generations
            crossover_rate: Crossover probability
            fixed_T: Fixed neighborhood size (no Q-learning adaptation)
            seed: Random seed
            data_dir: Directory containing .fjs files
        """
        self.instance_name = instance_name
        self.n_pop = n_pop
        self.max_gen = max_gen
        self.crossover_rate = crossover_rate
        self.fixed_T = fixed_T
        self.seed = seed
        self.data_dir = data_dir

        self.rng = np.random.RandomState(seed)
        self.instance = None
        self.weights = None
        self.B = None
        self.archive = []
        self.history = []

        logger.info("MOEA/D | %s | Np=%d G=%d T=%d", instance_name, n_pop, max_gen, fixed_T)
        logger.debug("Seed=%d CR=%.2f", seed, crossover_rate)

    def _init_population(self):
        """Initialize population using MIX3 strategy."""
        logger.debug("Initializing population with MIX3 strategy...")
        start = time.perf_counter()
        pop = init_mix3(self.instance, self.n_pop, self.rng)
        objectives = []
        for os_vec, ma_vec in pop:
            m, w, mc, wc = decode(os_vec, ma_vec, self.instance)
            objectives.append((mc, wc))
        elapsed = time.perf_counter() - start
        logger.debug("Population initialized: %d individuals in %.4f s", len(pop), elapsed)
        return pop, objectives

    def _compute_pf(self, objectives):
        """Extract non-dominated front from current objectives."""
        return non_dominated_sort(objectives)

    def _update_archive(self, population, objectives):
        """Update elite archive with non-dominated solutions."""
        combined = self.archive + [
            (pop[0], pop[1], obj) for pop, obj in zip(population, objectives)
        ]
        objs = [item[2] for item in combined]
        nd_objs = non_dominated_sort(objs)
        nd_set = set(nd_objs)
        new_archive = []
        for item in combined:
            if item[2] in nd_set:
                new_archive.append(item)
                nd_set.remove(item[2])
        if len(new_archive) > self.n_pop:
            new_archive = new_archive[:self.n_pop]
        self.archive = new_archive
        return len(new_archive)

    def solve(self):
        """
        Run the baseline MOEA/D algorithm.
        执行标准MOEA/D算法。

        Returns:
            dict: Results containing final PF, HV, and history
        """
        total_start = time.perf_counter()

        # Load instance
        self.instance = load_instance(self.instance_name, self.data_dir, self.seed)
        logger.debug("Instance loaded: %d jobs, %d machines, %d operations",
                    self.instance["n_jobs"], self.instance["n_machines"],
                    self.instance["total_ops"])

        # Initialize weights and fixed neighbors
        self.weights = generate_weights(self.n_pop)
        self.B = compute_neighbors(self.weights, self.fixed_T)
        logger.debug("Weight vectors generated: %d vectors, fixed T=%d",
                    len(self.weights), self.fixed_T)

        # Initialize population
        population, objectives = self._init_population()

        # Initialize reference point
        z = (
            min(o[0] for o in objectives),
            min(o[1] for o in objectives),
        )
        logger.debug("Initial reference point: z=(%.4f, %.4f)", z[0], z[1])

        # Initial PF and archive
        pf = self._compute_pf(objectives)
        self._update_archive(population, objectives)
        logger.debug("Initial non-dominated front size: %d", len(pf))

        # Main loop
        for gen in range(1, self.max_gen + 1):
            gen_start = time.perf_counter()

            # Fixed T, no Q-learning adaptation
            population, objectives, z = moead_generation(
                population, objectives, self.weights, self.B,
                self.instance, z, self.crossover_rate, self.rng
            )

            # Update PF and archive
            pf = self._compute_pf(objectives)
            archive_size = self._update_archive(population, objectives)

            # Compute HV
            hv = compute_hv(pf)

            gen_time = time.perf_counter() - gen_start

            # Compute per-objective statistics from current PF
            pf_array = np.array(pf)
            best_makespan = float(np.min(pf_array[:, 0]))
            best_workload = float(np.min(pf_array[:, 1]))
            avg_makespan = float(np.mean(pf_array[:, 0]))
            avg_workload = float(np.mean(pf_array[:, 1]))

            self.history.append({
                "gen": gen,
                "T": self.fixed_T,
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

            if gen % 20 == 0 or gen == 1:
                logger.debug("Gen %d | PF=%d | Archive=%d | HV=%.6f | z=(%.2f,%.2f) | Time=%.3fs",
                            gen, len(pf), archive_size, hv, z[0], z[1], gen_time)
            # 关键里程碑才打印控制台
            if gen % 100 == 0:
                logger.info("Gen %d/%d | HV=%.6f | %.1fs", gen, self.max_gen, hv, gen_time)

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

        logger.info("Done | PF=%d HV=%.6f | %.2fs", len(final_pf), final_hv, total_time)
        logger.debug("Final non-dominated front size: %d", len(final_pf))
        logger.debug("Final HV: %.6f", final_hv)

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
            "algorithm": "MOEA/D",
            "instance": self.instance_name,
            "n_pop": self.n_pop,
            "max_gen": self.max_gen,
            "crossover_rate": self.crossover_rate,
            "fixed_T": self.fixed_T,
            "seed": self.seed,
            "final_pf": named_pf,
            "fuzzy_pf": named_fuzzy_pf,
            "final_hv": final_hv,
            "total_time": total_time,
            "history": self.history,
        }
        return results

    def save_results(self, results, output_dir="results"):
        """Save results to JSON file with grouped directory structure.
        按实例和算法分组存储结果文件：results/Mk01/MOEA_D/xxx.json
        """
        # Grouped directory: results/<instance>/<algorithm>/
        algo_dir = os.path.join(output_dir, self.instance_name, "MOEA_D")
        os.makedirs(algo_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.instance_name}_MOEA_D_Np{self.n_pop}_G{self.max_gen}_T{self.fixed_T}_{timestamp}.json"
        filepath = os.path.join(algo_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.debug("Results saved to: %s", filepath)
        return filepath
