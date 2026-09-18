"""
RMOEA/D Algorithm: Main solver integrating Q-learning + MOEA/D.
RMOEA/D主算法：集成Q-learning参数自适应与MOEA/D演化框架。

核心流程：
1. 初始化种群（MIX3策略）
2. 初始化权重向量与Q-learning
3. 每代：Q-learning选择T → 更新邻居结构 → MOEA/D演化 → 更新Q-table
4. 维护精英存档，输出最终Pareto前沿
"""

# ── 禁止 BLAS/MKL 内部多线程，避免与 ProcessPoolExecutor 冲突 ──
import os as _os
_os.environ["OMP_NUM_THREADS"] = "1"
_os.environ["MKL_NUM_THREADS"] = "1"
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
_os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import time
import logging
import json
import os

from .core.instance import load_instance
from .core.operators import init_by_variant, init_mix3, init_random
from .core.encoding import decode, decode_with_schedule, decode_crisp
from .core.moead import generate_weights, compute_neighbors, moead_generation
from .core.qlearning import QLearningPAS, compute_cv_dv
from .core.rvns import RVNS, rvns_generation
from .utils.metrics import non_dominated_sort, compute_hv, instance_hv_bounds
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
        ql_reward_mode="dv",
        ql_cv_normalize=False,
        ql_tie_break="random",
        ql_q_init="zero",
        ql_q_init_scale=0.1,
        enable_rvns=True,
        rvns_lp=40,
        rvns_ls_trials=1,
        rvns_budget_mode="fixed",
        rvns_budget_pool=None,
        rvns_budget_target_mean=None,
        rvns_mode="rl",
        enable_mix3=True,
        init_variant="mix3",
        enable_elite=True,
        fixed_T=None,
        timeout=None,
        algorithm_name="RMOEA/D",
        algo_dir_name="RMOEA_D",
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
            ql_cv_normalize: 是否把 CV 的两个目标归一化到同一尺度
                             （默认 False，即严格照论文式(14) 用原始值）
            enable_rvns: Whether to enable RVNS local search
            rvns_lp: RVNS 成功/失败记忆窗口长度 LP
            rvns_ls_trials: RVNS 每个解每代最多尝试的邻域次数 (论文为 1)
            rvns_mode: "rl" 按 SM/FM 轮盘赌选算子；"random" 五算子等概率随机选
                       （论文 Section 4.6 用法 (1)，即 RMOEA/D3 的随机 VNS）
            rvns_budget_mode: 邻域尝试次数在种群内的**分配方式**（ABA）。
                       "fixed"（默认，论文口径）所有解统一用 rvns_ls_trials 次上限；
                       "pool_random" 每代总预算固定、随机决定升级哪些解（异质性对照）；
                       "pool_state" 优先升级上一代未被改进的解；
                       "pool_learn" 升级偏好由 SM/FM 式信用统计学习得到。
                       pool_* 三档的总预算逐代严格相等（见 core/rvns.py）。
            rvns_budget_pool: 允许的尝试次数档位，默认 [1, 3]
            rvns_budget_target_mean: 每代每解平均预算上限；给定时每代总预算
                       固定为 n_pop * rvns_budget_target_mean
            enable_mix3: True 用 MIX3 初始化（1/3 随机 + 1/3 最短时间 + 1/3 全局负载，
                        论文 RMOEA/D2 的贡献点）；False 退化为纯随机初始化 = RMOEA/D1
            enable_elite: True 启用精英档案（论文 RMOEA/D5 的贡献点）；
                         False 时最终前沿直接取末代种群的非支配集，档案不参与输出
            fixed_T: Fixed neighborhood size (if None, use Q-learning)
            timeout: Maximum wall-clock time in seconds (None = no limit)
            algorithm_name: Algorithm name for result metadata
            algo_dir_name: Directory name for saved results
        """
        self.instance_name = instance_name
        self.n_pop = n_pop
        self.max_gen = max_gen
        self.crossover_rate = crossover_rate
        self.seed = seed
        self.data_dir = data_dir
        self.enable_rvns = enable_rvns
        self.enable_mix3 = enable_mix3
        # 初始化变体（见 core.operators.INIT_VARIANTS）。
        # 仅在 enable_mix3=True 时生效；enable_mix3=False 一律走纯随机（论文 D1）。
        self.init_variant = init_variant
        self.enable_elite = enable_elite
        self.fixed_T = fixed_T
        self.timeout = timeout
        self.algorithm_name = algorithm_name
        self.algo_dir_name = algo_dir_name

        # Q-learning parameters
        self.ql_alpha = ql_alpha
        self.ql_gamma = ql_gamma
        self.ql_epsilon = ql_epsilon
        self.ql_actions = ql_actions if ql_actions is not None else [5, 10, 15, 20]
        self.ql_reward_mode = ql_reward_mode
        # CV 归一化：论文式(14) 用原始目标值，量纲悬殊时会让大量纲目标独占
        # CV、使状态「看不见」另一目标。默认 False = 严格照论文。
        self.ql_cv_normalize = ql_cv_normalize
        # 平局处理：Q 表初值并列时若固定取索引 0，会把策略锁死在 actions[0]
        self.ql_tie_break = ql_tie_break
        self.ql_q_init = ql_q_init
        self.ql_q_init_scale = ql_q_init_scale

        # RVNS parameters
        self.rvns_lp = rvns_lp
        self.rvns_ls_trials = rvns_ls_trials
        self.rvns_mode = rvns_mode
        # ABA：邻域搜索预算的分配方式（"fixed" 即论文口径，其余见 core/rvns.py）
        self.rvns_budget_mode = rvns_budget_mode
        self.rvns_budget_pool = rvns_budget_pool
        self.rvns_budget_target_mean = rvns_budget_target_mean
        self.rvns = (RVNS(n_operators=5, lp=rvns_lp, ls_trials=rvns_ls_trials,
                          mode=rvns_mode,
                          budget_mode=rvns_budget_mode,
                          budget_pool=rvns_budget_pool,
                          budget_target_mean=rvns_budget_target_mean)
                     if enable_rvns else None)

        # Internal state
        self.rng = np.random.RandomState(seed)
        self.instance = None
        self.weights = None
        self.ql = None
        self.archive = []  # Elite archive: list of (os, ma, obj)
        self.history = []  # Generation history for analysis

        logger.info("%s | %s | Np=%d G=%d | %s | %s",
                    algorithm_name, instance_name, n_pop, max_gen,
                    f"QL(T={self.ql_actions},R={self.ql_reward_mode})"
                    if fixed_T is None else f"FixedT={fixed_T}",
                    "RVNS" if enable_rvns else "noRVNS")
        logger.debug("Seed=%d CR=%.2f alpha=%.2f gamma=%.2f epsilon=%.2f",
                     seed, crossover_rate, ql_alpha, ql_gamma, ql_epsilon)

    def _init_population(self):
        """Initialize the population.

        默认用 MIX3 策略（论文 RMOEA/D2 的贡献点）；
        `enable_mix3=False` 时退化为纯随机初始化，即论文 RMOEA/D1。

        `init_variant` 进一步给出**变体扫描**入口（见 core.operators.INIT_VARIANTS）。
        动机：MIX3 是消融阶梯里最大的单一组件，却**从未做过变体扫描**；
        而且它的三条分支（random / LS / GW）全都把 OS 随机打乱，只在**机器选择**
        维度做文章 —— 工序顺序这一维在初始化阶段完全没被利用。
        `init_variant="mix3"` 与论文口径**逐位相同**（见 tests 的等价锁）。

        效应量按 **实例边界口径**（`instance_hv_bounds`，与臂集无关）为 **+3.406%**
        （10 实例平均，256/300，p=2.7e-40），是阶梯里最大的一级（次大者 +0.421%）；
        旧的 `+14.46%` 是**盒口径**产物（放大 4.9×，见 docs/new-arch-report.md §1）。
        另注：该 +3.406% 是**有限预算的起跑优势** —— G=200 处配对差 +1.937%，
        到 G=1000 只剩 +0.079%（n.s.）。

        初始化：按变体生成个体 → crisp decode 目标值（热路径零分配）。"""
        variant = self.init_variant if self.enable_mix3 else "random"
        logger.debug("Initializing population with %s strategy...", variant)
        start = time.perf_counter()
        pop = init_by_variant(self.instance, self.n_pop, self.rng, variant)
        objectives = []
        for os_vec, ma_vec in pop:
            mc, wc = decode_crisp(os_vec, ma_vec, self.instance)
            objectives.append((mc, wc))
        elapsed = time.perf_counter() - start
        logger.debug("Population initialized: %d individuals in %.4f s", len(pop), elapsed)
        return pop, objectives

    def _compute_pf(self, objectives):
        """Extract non-dominated front from current objectives."""
        return non_dominated_sort(objectives)

    def _update_archive(self, population, objectives):
        """Update the elite archive (论文 RMOEA/D5 的贡献点).

        `enable_elite=False` 时档案退化为「当前种群的非支配集」——
        即没有跨代积累能力，输出随之退化为末代种群的前沿。
        这与论文 RMOEA/D1–D4 的行为一致（它们尚无 Elite archive）。
        """
        if self.enable_elite:
            # Combine archive and current population
            combined = self.archive + [
                (pop[0], pop[1], obj) for pop, obj in zip(population, objectives)
            ]
        else:
            # 无档案积累：只保留当前代
            combined = [(pop[0], pop[1], obj)
                        for pop, obj in zip(population, objectives)]
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
        logger.debug("Instance loaded: %d jobs, %d machines, %d operations",
                    self.instance["n_jobs"], self.instance["n_machines"],
                    self.instance["total_ops"])

        # 固定的 HV 归一化边界（由实例数据确定性推出）
        # → 同一实例下所有 run / 所有算法变体共用，HV 才可直接比较
        self.hv_bounds = instance_hv_bounds(self.instance)
        logger.debug("HV normalization bounds: lo=%s hi=%s",
                     self.hv_bounds[0].tolist(), self.hv_bounds[1].tolist())

        # Initialize weights
        self.weights = generate_weights(self.n_pop)
        logger.debug("Weight vectors generated: %d vectors", len(self.weights))

        # 邻居结构只依赖 (weights, T)。固定 T 的臂每代 T 不变 → 缓存命中，
        # 省掉每代一次 O(Np²) 的距离矩阵 + argsort（Np=100、G=200 时约 0.4s）。
        self._neighbor_cache = {}

        # 档案开关变化时必须重置，避免复用上一次 solve() 的残留状态
        self.archive = []

        # Initialize Q-learning (only when adaptive T is needed)
        if self.fixed_T is None:
            self.ql = QLearningPAS(
                alpha=self.ql_alpha,
                gamma=self.ql_gamma,
                epsilon=self.ql_epsilon,
                actions=self.ql_actions,
                reward_mode=self.ql_reward_mode,
                hv_bounds=self.hv_bounds,
                cv_normalize=self.ql_cv_normalize,
                tie_break=self.ql_tie_break,
                q_init=self.ql_q_init,
                q_init_scale=self.ql_q_init_scale,
                q_init_seed=self.seed,
            )
        else:
            self.ql = None

        # Initialize population
        population, objectives = self._init_population()

        # Initialize reference point (crisp values for MOEA/D evolution)
        # 参考点使用清晰值，用于MOEA/D的标量化函数
        z = (
            min(o[0] for o in objectives),
            min(o[1] for o in objectives),
        )
        logger.debug("Initial reference point: z=(%.4f, %.4f)", z[0], z[1])

        # Initial PF and archive (using crisp values for evolution)
        pf = self._compute_pf(objectives)
        self._update_archive(population, objectives)
        logger.debug("Initial non-dominated front size: %d", len(pf))

        # Main loop
        for gen in range(1, self.max_gen + 1):
            gen_start = time.perf_counter()

            # ── 超时保护：每代开始前检查是否超过总时限 ──
            if self.timeout is not None and (gen_start - total_start) > self.timeout:
                logger.warning("Timeout reached at generation %d/%d (%.1fs > %.1fs), stopping early",
                               gen - 1, self.max_gen, gen_start - total_start, self.timeout)
                break

            # Step 1: Apply RVNS to each solution (论文 Algorithm 1 line 4)
            if self.enable_rvns:
                population, objectives, z = rvns_generation(
                    population, objectives, self.weights,
                    self.instance, z, self.rng, self.rvns
                )

            # Step 2: Q-learning selects T or use fixed T
            if self.fixed_T is not None:
                T = self.fixed_T
                logger.debug("Generation %d/%d: Using fixed T=%d", gen, self.max_gen, T)
            else:
                # 第二个返回值（是否是首次调用）对主循环无用，此处忽略即可
                T, _ = self.ql.step(pf, self.rng)
                logger.debug("Generation %d/%d: Q-learning selected T=%d", gen, self.max_gen, T)

            # Step 3: Recompute neighbors with new T
            B = self._neighbor_cache.get(T)
            if B is None:
                B = compute_neighbors(self.weights, T)
                self._neighbor_cache[T] = B

            # Step 4: MOEA/D generation
            population, objectives, z = moead_generation(
                population, objectives, self.weights, B,
                self.instance, z, self.crossover_rate, self.rng
            )

            # Step 5: Update PF and archive
            pf = self._compute_pf(objectives)
            archive_size = self._update_archive(population, objectives)

            # Step 6: Compute HV
            hv = compute_hv(pf, norm_bounds=self.hv_bounds)

            # 逐代 CV/DV（Q-PAS 的状态量）。固定 T 的臂没有 Q-learning 对象，
            # 但诊断「状态空间是否可观测到最优 T」需要这条轨迹，故统一记录。
            _cv, _dv = compute_cv_dv(pf, hv_bounds=self.hv_bounds)

            gen_time = time.perf_counter() - gen_start

            # Compute per-objective statistics from current PF
            # 从当前Pareto前沿计算各目标的统计值
            pf_array = np.array(pf)
            best_makespan = float(np.min(pf_array[:, 0]))
            best_workload = float(np.min(pf_array[:, 1]))
            avg_makespan = float(np.mean(pf_array[:, 0]))
            avg_workload = float(np.mean(pf_array[:, 1]))

            # Record history
            history_entry = {
                "gen": gen,
                "T": T,
                "pf_size": len(pf),
                "archive_size": archive_size,
                "hv": hv,
                "cv": _cv,
                "dv": _dv,
                "z": z,
                "best_makespan": best_makespan,
                "best_workload": best_workload,
                "avg_makespan": avg_makespan,
                "avg_workload": avg_workload,
                "time": gen_time,
            }
            if self.enable_rvns:
                history_entry["rvns_probs"] = self.rvns.get_probabilities()
            self.history.append(history_entry)

            # Periodic logging
            if gen % 20 == 0 or gen == 1:
                logger.debug("Gen %d | T=%d | PF=%d | Archive=%d | HV=%.6f | z=(%.2f,%.2f) | Time=%.3fs",
                            gen, T, len(pf), archive_size, hv, z[0], z[1], gen_time)
            # 关键里程碑才打印控制台
            if gen % 100 == 0:
                logger.info("Gen %d/%d | HV=%.6f | %.1fs", gen, self.max_gen, hv, gen_time)

        total_time = time.perf_counter() - total_start

        # Re-decode archive solutions to get fuzzy objectives for final output
        # 对存档中的解重新解码，获取完整的模糊目标值用于最终输出
        # enable_elite=False 时档案只是「末代种群的非支配集」，等价于纯 MOEA/D 输出
        fuzzy_archive = []
        for os_vec, ma_vec, _ in self.archive:
            fm, fw, _, _ = decode(os_vec, ma_vec, self.instance)
            fuzzy_archive.append((fm.to_tuple(), fw.to_tuple()))

        # Final results (crisp PF for HV computation)
        final_pf = [(item[2][0], item[2][1]) for item in self.archive]
        final_pf = non_dominated_sort(final_pf)
        final_hv = compute_hv(final_pf, norm_bounds=self.hv_bounds)

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

        # ── 对存档中的非支配解重新解码，获取详细调度过程数据 ──
        # 用于结果JSON和甘特图可视化
        best_schedules = []
        for rank, (os_vec, ma_vec, obj) in enumerate(self.archive):
            _, _, _, _, schedule = decode_with_schedule(os_vec, ma_vec, self.instance)
            entry = {
                "rank": rank + 1,
                "makespan_crisp": obj[0],
                "workload_crisp": obj[1],
                "num_operations": len(schedule),
                "schedule": schedule,  # 每道工序的详细调度信息
            }
            best_schedules.append(entry)

        # 找到最佳解（makespan最小的解，按三阶段排序算子比较）
        # 在模糊PF中对应选择rank最低的
        best_solution = best_schedules[0] if best_schedules else None

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
            "algorithm": self.algorithm_name,
            "instance": self.instance_name,
            "n_pop": self.n_pop,
            "max_gen": self.max_gen,
            "crossover_rate": self.crossover_rate,
            "seed": self.seed,
            "final_pf": named_pf,
            "fuzzy_pf": named_fuzzy_pf,
            "final_hv": final_hv,
            # HV 的固定归一化边界 (lo, hi)——同一实例内所有 run/算法共用，
            # 保证 HV 可跨算法比较
            "hv_bounds": [self.hv_bounds[0].tolist(), self.hv_bounds[1].tolist()],
            "total_time": total_time,
            "timed_out": self.timeout is not None and total_time > self.timeout,
            "timeout": self.timeout,
            "history": self.history,
            "q_table": self.ql.q_table.tolist() if self.ql else None,
            "actions": list(self.ql_actions) if self.ql else None,
            "ql_reward_mode": self.ql_reward_mode if self.ql else None,
            "ql_cv_normalize": self.ql_cv_normalize if self.ql else None,
            "rvns_final_probs": self.rvns.get_probabilities() if self.rvns else None,
            # ABA：预算分配统计。mean_planned_trials 是"算力是否真相等"的判据，
            # mean_actual_evals 是真实邻域求值次数——两个都要落盘，否则等算力只是声称。
            "rvns_budget": self.rvns.get_budget_stats() if self.rvns else None,
            # ── 组件开关（消融阶梯复现需要，便于从结果反查配置）──
            "components": {
                "mix3": self.enable_mix3,
                "qpas": self.fixed_T is None,
                "rvns": self.enable_rvns,
                "rvns_mode": self.rvns_mode if self.enable_rvns else None,
                "rvns_budget_mode": (self.rvns_budget_mode
                                     if self.enable_rvns else None),
                "elite": self.enable_elite,
                "fixed_T": self.fixed_T,
            },
            # ── 调度过程数据 (用于甘特图和结果分析) ──
            "schedules": best_schedules,
            "best_solution": best_solution,
            "num_machines": self.instance["n_machines"],
            "num_jobs": self.instance["n_jobs"],
        }
        return results

    def save_results(self, results, output_dir="results"):
        """Save results to JSON file with grouped directory structure.
        按实例和算法分组存储结果文件：results/benchmark/<instance>/RMOEA_D/xxx.json
        同时自动生成甘特图到 charts/schedules/<instance>/
        """
        # Grouped directory: results/benchmark/<instance>/<algorithm>/
        algo_dir = os.path.join(output_dir, "benchmark", self.instance_name, self.algo_dir_name)
        os.makedirs(algo_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.instance_name}_{self.algo_dir_name}_Np{self.n_pop}_G{self.max_gen}_{timestamp}.json"
        filepath = os.path.join(algo_dir, filename)

        def _json_default(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)

        logger.debug("Results saved to: %s", filepath)

        # ── 自动生成甘特图 ──
        try:
            from .utils.visualization import generate_gantt_from_results
            gantt_paths = generate_gantt_from_results(results)
            if gantt_paths:
                logger.debug("Gantt charts: %d generated for %s", len(gantt_paths), self.instance_name)
        except Exception as e:
            logger.warning("Gantt chart generation failed: %s", e)

        return filepath
