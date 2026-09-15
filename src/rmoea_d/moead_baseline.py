"""
Baseline MOEA/D solver without Q-learning (fixed T).
纯MOEA/D对比算法：固定邻域大小T，无Q-learning参数自适应。

用于与RMOEA/D进行对照实验，验证Q-learning自适应策略的有效性。

实现说明：
  MOEADBaseline 是 RMOEAD 在 fixed_T + disable RVNS 配置下的特化子类。
  通过继承复用 RMOEAD 的全部求解逻辑，仅覆盖元数据（算法名称、保存目录）。
"""

# ── 禁止 BLAS/MKL 内部多线程，避免与 ProcessPoolExecutor 冲突 ──
import os as _os
_os.environ["OMP_NUM_THREADS"] = "1"
_os.environ["MKL_NUM_THREADS"] = "1"
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
_os.environ["NUMEXPR_NUM_THREADS"] = "1"

from .algorithm import RMOEAD


class MOEADBaseline(RMOEAD):
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
        timeout=None,
    ):
        """
        Initialize baseline MOEA/D solver.
        初始化标准MOEA/D求解器。

        Parameters:
            instance_name: Brandimarte instance name
            n_pop: Population size
            max_gen: Maximum number of generations
            crossover_rate: Crossover probability
            fixed_T: Fixed neighborhood size (no Q-learning adaptation)
            seed: Random seed
            data_dir: Directory containing .fjs files
            timeout: Maximum wall-clock time in seconds (None = no limit)
        """
        super().__init__(
            instance_name=instance_name,
            n_pop=n_pop,
            max_gen=max_gen,
            crossover_rate=crossover_rate,
            seed=seed,
            data_dir=data_dir,
            fixed_T=fixed_T,
            enable_rvns=False,
            timeout=timeout,
            algorithm_name="MOEA/D",
            algo_dir_name="MOEA_D",
        )
