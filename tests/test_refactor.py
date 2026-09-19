"""
单元测试：验证重构后核心功能的正确性

覆盖范围：
  1. MOEADBaseline 继承 RMOEAD 后行为一致
  2. RMOEAD 在 fixed_T 模式下正确跳过 Q-learning 初始化
  3. ablation_visualization.py 导入与数据加载
  4. experiment._extract_run 字段提取完整性
  5. 消融诊断中修复的 4 个缺陷回归锁：
     HV 参考集归一化、RVNS Tchebycheff 接受准则、Q-PAS ε 极性、Q 表平局自锁
  6. Q-PAS 论文口径守卫：CV 默认不归一化、奖励默认按式(18)
  7. AIG 门控：置换零假设（家族错误率）、留一稳健性、目标量 A/B 分离
  8. 脚本 CLI：每个 scripts/*.py 的 --help 必须退出码 0（缺陷 19/27）
"""

import sys
import os
import inspect
import io
import subprocess
import tempfile
import unittest
import numpy as np

# 将 src 加入路径以支持模块导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rmoea_d.algorithm import RMOEAD
from rmoea_d.moead_baseline import MOEADBaseline
from rmoea_d.core.qlearning import QLearningPAS
from rmoea_d.core.rvns import RVNS
from rmoea_d.utils.metrics import (compute_hv, estimate_hv_bounds,
                                   instance_hv_bounds)
from rmoea_d.core.instance import load_instance
from rmoea_d.utils.ablation_visualization import load_ablation_results, _get_multi_run
from rmoea_d.utils.experiment import _extract_run


class TestMOEADBaselineRefactor(unittest.TestCase):
    """验证 MOEADBaseline 继承重构后的行为等价性"""

    def test_moead_baseline_is_rmoead_subclass(self):
        self.assertTrue(issubclass(MOEADBaseline, RMOEAD))

    def test_moead_baseline_algorithm_name(self):
        solver = MOEADBaseline("Mk01", n_pop=10, max_gen=5, fixed_T=10, seed=0)
        self.assertEqual(solver.algorithm_name, "MOEA/D")
        self.assertEqual(solver.algo_dir_name, "MOEA_D")
        self.assertFalse(solver.enable_rvns)
        self.assertEqual(solver.fixed_T, 10)

    def test_moead_baseline_no_ql(self):
        solver = MOEADBaseline("Mk01", n_pop=10, max_gen=5, fixed_T=10, seed=0)
        self.assertIsNone(solver.ql)

    def test_rmoead_with_fixed_t_no_ql(self):
        solver = RMOEAD("Mk01", n_pop=10, max_gen=5, fixed_T=10, seed=0)
        self.assertIsNone(solver.ql)
        self.assertTrue(solver.enable_rvns)  # 默认启用 RVNS

    def test_rmoead_adaptive_config(self):
        solver = RMOEAD("Mk01", n_pop=10, max_gen=5, seed=0)
        self.assertIsNone(solver.fixed_T)  # adaptive mode
        self.assertIsNotNone(solver.ql_actions)


class TestExtractRun(unittest.TestCase):
    """验证 _extract_run 字段提取完整性"""

    def _make_dummy_result(self):
        return {
            "final_pf": [{"Makespan": 100, "Workload": 50}],
            "fuzzy_pf": [
                {"Makespan": {"t1": 90, "t2": 100, "t3": 110}, "Workload": {"t1": 45, "t2": 50, "t3": 55}}
            ],
            "final_hv": 0.85,
            "schedules": [{"job": 0, "machine": 1, "start": 0, "end": 10}],
            "q_table": [[0.1, 0.2], [0.3, 0.4]],
            "history": [
                {"gen": 0, "hv": 0.5, "pf_size": 5, "best_makespan": 120, "best_workload": 60}
            ],
        }

    def test_extract_run_basic_fields(self):
        r = self._make_dummy_result()
        out = _extract_run(r, "rmoea_d", "Mk01", 42, 100, 200, 1.5, ql_params={"alpha": 0.4})
        self.assertEqual(out["algorithm"], "rmoea_d")
        self.assertEqual(out["instance"], "Mk01")
        self.assertEqual(out["seed"], 42)
        self.assertEqual(out["n_pop"], 100)
        self.assertEqual(out["max_gen"], 200)
        self.assertEqual(out["total_time"], 1.5)
        self.assertEqual(out["final_hv"], 0.85)
        self.assertEqual(out["pf_size"], 1)
        self.assertEqual(out["best_makespan"], 100.0)
        self.assertEqual(out["best_workload"], 50.0)

    def test_extract_run_history_conversion(self):
        r = self._make_dummy_result()
        out = _extract_run(r, "rmoea_d", "Mk01", 0, 10, 10, 0.0)
        self.assertEqual(len(out["history"]), 1)
        self.assertEqual(out["history"][0]["gen"], 0)
        self.assertEqual(out["history"][0]["hv"], 0.5)

    def test_extract_run_empty_pf(self):
        r = {"final_pf": [], "history": []}
        out = _extract_run(r, "moea_d", "Mk01", 0, 10, 10, 0.0)
        self.assertEqual(out["pf_size"], 0)
        self.assertEqual(out["best_makespan"], 0.0)
        self.assertEqual(out["best_workload"], 0.0)


class TestAblationVisualization(unittest.TestCase):
    """验证消融实验可视化辅助函数"""

    def test_get_multi_run_empty(self):
        mean, std, n = _get_multi_run([])
        self.assertEqual(mean, 0)
        self.assertEqual(std, 0)
        self.assertEqual(n, 0)

    def test_get_multi_run_single(self):
        mean, std, n = _get_multi_run([5.0])
        self.assertAlmostEqual(mean, 5.0)
        self.assertEqual(std, 0.0)
        self.assertEqual(n, 1)

    def test_get_multi_run_multi(self):
        mean, std, n = _get_multi_run([1.0, 2.0, 3.0])
        self.assertAlmostEqual(mean, 2.0)
        self.assertTrue(std > 0)
        self.assertEqual(n, 3)

    def test_load_ablation_results_no_data(self):
        # 使用不存在的目录应返回 None
        result = load_ablation_results("results/nonexistent_dir")
        self.assertIsNone(result)


class TestRMOEADResultConsistency(unittest.TestCase):
    """验证 RMOEAD 结果字典字段一致性"""

    def test_result_keys(self):
        # 构造一个足够小的实例进行快速 smoke test
        # 注意：不实际运行 solve()，仅验证结果字典结构
        solver = RMOEAD("Mk01", n_pop=4, max_gen=2, seed=0)
        # 模拟一个最小化的结果字典
        dummy = {
            "final_pf": [{"Makespan": 100, "Workload": 50}],
            "fuzzy_pf": [],
            "final_hv": 0.5,
            "schedules": [],
            "best_solution": {"os": [], "ma": []},
            "num_machines": 6,
            "num_jobs": 10,
        }
        out = _extract_run(dummy, "rmoea_d", "Mk01", 0, 4, 2, 0.1)
        required_keys = {
            "algorithm", "instance", "seed", "n_pop", "max_gen",
            "total_time", "final_hv", "pf_size", "best_makespan",
            "best_workload", "avg_makespan", "avg_workload",
            "final_pf", "fuzzy_pf", "schedules", "history",
        }
        self.assertTrue(required_keys.issubset(set(out.keys())))


class TestQlearningEpsilonPolarity(unittest.TestCase):
    """回归测试：Q-PAS 的 ε-greedy 必须与论文一致（rand<ε → 利用）。

    论文 Section 4.5.5 明确说明该写法与常规 ε-greedy 相反。若被写反
    （rand<ε → 随机），ε=0.8 会变成 80% 随机探索，Q-table 学不到东西。
    """

    def test_exploit_when_rand_below_epsilon(self):
        ql = QLearningPAS(epsilon=0.8, actions=[5, 10, 15, 20])
        ql.q_table[0] = [0.0, 0.0, 0.0, 9.9]          # 动作 3 明显最优
        rng = np.random.RandomState(0)
        picks = [ql.select_action(0, rng) for _ in range(600)]
        frac_best = picks.count(3) / len(picks)
        # 80% 利用 + 均摊 20% 探索 ⇒ 期望 ≈ 0.80 + 0.20/4 = 0.85
        self.assertGreater(frac_best, 0.70,
                           "ε=0.8 时应以利用为主（>70% 选 max-Q 动作）")
        self.assertLess(frac_best, 0.98)

    def test_explore_when_rand_above_epsilon(self):
        ql = QLearningPAS(epsilon=0.0, actions=[5, 10, 15, 20])   # 纯随机
        ql.q_table[0] = [0.0, 0.0, 0.0, 9.9]
        rng = np.random.RandomState(0)
        picks = [ql.select_action(0, rng) for _ in range(600)]
        self.assertGreater(len(set(picks)), 1, "ε=0 时应能探索到多个动作")


class TestQlearningRewardModes(unittest.TestCase):
    """回归测试：三种奖励模式的定义。默认 "dv" 必须严格照论文式 (20)。"""

    def test_default_mode_is_paper_dv(self):
        self.assertEqual(QLearningPAS().reward_mode, "dv")

    def test_reward_dv(self):
        ql = QLearningPAS(reward_mode="dv")
        self.assertEqual(ql._compute_reward(1.0, 0.5), 10.0)     # ΔDV > 0
        self.assertEqual(ql._compute_reward(1.0, 0.0), 0.0)      # ΔDV = 0
        self.assertEqual(ql._compute_reward(-1.0, -0.5), 0.0)    # ΔDV < 0
        # 论文口径下奖励只看 DV，CV 无关
        self.assertEqual(ql._compute_reward(-9.0, 0.1), 10.0)

    def test_reward_cv_dv(self):
        ql = QLearningPAS(reward_mode="cv_dv")
        self.assertEqual(ql._compute_reward(0.1, 0.1), 10.0)     # 两者都改善
        self.assertEqual(ql._compute_reward(0.1, -0.1), 5.0)     # 仅收敛改善
        self.assertEqual(ql._compute_reward(-0.1, 0.1), 5.0)     # 仅多样性改善
        self.assertEqual(ql._compute_reward(-0.1, -0.1), 0.0)

    def test_reward_hv(self):
        ql = QLearningPAS(reward_mode="hv")
        self.assertEqual(ql._compute_reward(0.0, 0.0), 0.0)      # 首次无 prev_hv
        ql.prev_hv = 0.5
        self.assertEqual(ql._compute_reward(0.0, 0.0, cur_hv=0.6), 10.0)
        self.assertEqual(ql._compute_reward(0.0, 0.0, cur_hv=0.4), 0.0)

    def test_reward_hv_cont(self):
        ql = QLearningPAS(reward_mode="hv_cont", w_hv_cont=100.0, w_hv_clip=10.0)
        self.assertEqual(ql._compute_reward(0.0, 0.0), 0.0)          # 首次无 prev_hv
        ql.prev_hv = 0.5
        self.assertAlmostEqual(ql._compute_reward(0.0, 0.0, cur_hv=0.51), 1.0)
        self.assertAlmostEqual(ql._compute_reward(0.0, 0.0, cur_hv=0.49), -1.0)
        self.assertEqual(ql._compute_reward(0.0, 0.0, cur_hv=0.90), 10.0)   # 上裁
        self.assertEqual(ql._compute_reward(0.0, 0.0, cur_hv=0.10), -10.0)  # 下裁

    def test_front_hv_uses_shared_bounds(self):
        ql = QLearningPAS(reward_mode="hv", hv_bounds=([0.0, 0.0], [10.0, 10.0]))
        front = np.array([[2.0, 2.0], [4.0, 1.0]])
        hv = ql._front_hv(front)
        self.assertGreater(hv, 0.0)
        self.assertLessEqual(hv, 1.0)          # 归一化后不超过单位盒


class TestQlearningTieBreak(unittest.TestCase):
    """回归测试：Q 表并列最大时必须能随机打破平局。

    Q 表零初始化时全表 Q=0，若利用分支固定取索引 0（np.argmax 原生行为），
    在论文的 ε=0.8（80% 利用）下策略会自锁到 actions[0]。实测 Mk10 上
    该缺陷使 47%~53% 的代数停在最差的 T=5，Q-PAS 因此全面劣于固定 T。
    """

    def test_zero_table_random_tie_break_covers_all_actions(self):
        ql = QLearningPAS(epsilon=1.0, actions=[5, 10, 15, 20],
                          tie_break="random")
        rng = np.random.RandomState(0)
        picks = [ql.select_action(0, rng) for _ in range(600)]
        # 全零表 + 纯利用 ⇒ 应在所有并列动作间均匀随机，而非恒取 0
        self.assertEqual(set(picks), {0, 1, 2, 3},
                         "并列时必须能随机覆盖所有动作，否则会自锁到 actions[0]")

    def test_zero_table_argmax_is_locked_to_index0(self):
        """记录旧口径的缺陷：全零表 + 纯利用 ⇒ 100% 停在索引 0。"""
        ql = QLearningPAS(epsilon=1.0, actions=[5, 10, 15, 20],
                          tie_break="argmax")
        rng = np.random.RandomState(0)
        picks = [ql.select_action(0, rng) for _ in range(200)]
        self.assertEqual(set(picks), {0})

    def test_unique_max_still_deterministic(self):
        ql = QLearningPAS(epsilon=1.0, actions=[5, 10, 15, 20])
        ql.q_table[0] = [0.0, 0.0, 7.7, 0.0]
        rng = np.random.RandomState(1)
        picks = [ql.select_action(0, rng) for _ in range(100)]
        self.assertEqual(set(picks), {2}, "唯一最大时不应受平局随机影响")

    def test_optimistic_init_breaks_symmetry(self):
        ql = QLearningPAS(actions=[5, 10, 15, 20], q_init="optimistic")
        self.assertFalse(np.allclose(ql.q_table, 0.0))
        self.assertEqual(ql.q_table.shape, (4, 4))
        # 固定种子 ⇒ 可复现
        ql2 = QLearningPAS(actions=[5, 10, 15, 20], q_init="optimistic")
        np.testing.assert_allclose(ql.q_table, ql2.q_table)

    def test_default_tie_break_is_random(self):
        self.assertEqual(QLearningPAS().tie_break, "random")

    def test_optimistic_init_seed_is_per_run(self):
        """乐观初始化必须随 run 的 seed 变化。

        若所有 run 共享同一张初始 Q 表，各 run 之间会引入人为相关性，
        压低方差并污染配对检验。algorithm.py 现在透传 self.seed。
        """
        a = QLearningPAS(actions=[5, 10, 15, 20], q_init="optimistic",
                         q_init_seed=42).q_table
        b = QLearningPAS(actions=[5, 10, 15, 20], q_init="optimistic",
                         q_init_seed=43).q_table
        c = QLearningPAS(actions=[5, 10, 15, 20], q_init="optimistic",
                         q_init_seed=42).q_table
        self.assertFalse(np.allclose(a, b), "不同 seed 应得到不同初始 Q 表")
        np.testing.assert_allclose(a, c, err_msg="同 seed 必须可复现")

    def test_optimistic_init_falls_back_to_constant_seed(self):
        """不传 q_init_seed 时行为不变（向后兼容）。"""
        a = QLearningPAS(actions=[5, 10, 15, 20], q_init="optimistic").q_table
        b = QLearningPAS(actions=[5, 10, 15, 20], q_init="optimistic",
                         q_init_seed=None).q_table
        np.testing.assert_allclose(a, b)

    def test_rmoead_passes_seed_to_optimistic_init(self):
        """RMOEAD 必须把自己的 seed 传给乐观初始化。"""
        sig = inspect.signature(QLearningPAS.__init__)
        self.assertIn("q_init_seed", sig.parameters)
        src = inspect.getsource(RMOEAD.solve)
        self.assertIn("q_init_seed=self.seed", src)


class TestHypervolumeNormalization(unittest.TestCase):
    """回归测试：HV 必须能区分算法优劣。

    旧口径（每条前沿自归一化）会把任意前沿拉伸到单位盒，
    使 HV 对整体优劣不敏感——这是"消融看不出差异"的元凶。
    """

    def test_shared_bounds_discriminates_quality(self):
        lo, hi = np.array([0.0, 0.0]), np.array([100.0, 100.0])
        good = np.array([[10.0, 60.0], [30.0, 30.0], [60.0, 10.0]])
        bad = np.array([[40.0, 90.0], [60.0, 60.0], [90.0, 40.0]])   # 整体更差
        hv_good = compute_hv(good, ref_point=(1.02, 1.02), norm_bounds=(lo, hi))
        hv_bad = compute_hv(bad, ref_point=(1.02, 1.02), norm_bounds=(lo, hi))
        self.assertGreater(hv_good, hv_bad)

    def test_self_normalization_hides_quality(self):
        """旧口径下好坏两前沿的 HV 几乎相同——记录该口径为何不可用。"""
        good = np.array([[10.0, 60.0], [30.0, 30.0], [60.0, 10.0]])
        bad = np.array([[40.0, 90.0], [60.0, 60.0], [90.0, 40.0]])
        hv_good = compute_hv(good, ref_point=(1.0, 1.0))
        hv_bad = compute_hv(bad, ref_point=(1.0, 1.0))
        self.assertLess(abs(hv_good - hv_bad), 0.05,
                        "自归一化口径对前沿整体优劣不敏感（故不用于跨算法比较）")

    def test_points_beyond_reference_excluded(self):
        lo, hi = np.array([0.0, 0.0]), np.array([10.0, 10.0])
        # 归一化后坐标 1.5 > ref 1.02 → 该点不应贡献体积
        front = np.array([[5.0, 5.0], [15.0, 1.0]])
        hv = compute_hv(front, ref_point=(1.02, 1.02), norm_bounds=(lo, hi))
        expected = (1.02 - 0.5) * (1.02 - 0.5)
        self.assertAlmostEqual(hv, expected, places=6)

    def test_empty_and_degenerate_front(self):
        self.assertEqual(compute_hv([]), 0.0)
        self.assertEqual(compute_hv(None), 0.0)
        self.assertEqual(compute_hv(np.empty((0, 2))), 0.0)

    def test_estimate_hv_bounds_covers_all_points(self):
        f1 = np.array([[1.0, 5.0], [3.0, 2.0]])
        f2 = np.array([[0.5, 9.0], [7.0, 1.0]])
        lo, hi = estimate_hv_bounds([f1, f2])
        allpts = np.vstack([f1, f2])
        self.assertTrue(np.all(lo <= allpts.min(axis=0) + 1e-12))
        self.assertTrue(np.all(hi >= allpts.max(axis=0) - 1e-12))

    def test_instance_hv_bounds_are_ordered(self):
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        if not os.path.exists(os.path.join(data_dir, 'Mk01.fjs')):
            self.skipTest("Mk01.fjs 不存在")
        inst = load_instance("Mk01", data_dir)
        lo, hi = instance_hv_bounds(inst)
        self.assertLess(lo[0], hi[0])
        self.assertLessEqual(lo[1], hi[1])


class TestRVNSTchebycheffAcceptance(unittest.TestCase):
    """回归测试：RVNS 必须用论文的接受准则（需要 weight/z 参与）。"""

    def test_ls_trials_stored(self):
        self.assertEqual(RVNS(n_operators=5, lp=40, ls_trials=3).ls_trials, 3)

    def test_apply_local_search_accepts_weight_and_z(self):
        import inspect
        sig = inspect.signature(RVNS.apply_local_search)
        params = set(sig.parameters)
        # 论文 Algorithm 4 的 g^te(P'|λ,Z) 需要 λ(weight) 与 Z(z)，
        # Pareto 支配判定不需要——这两个参数存在即为 Tchebycheff 口径
        self.assertIn("weight", params)
        self.assertIn("z", params)


class TestRVNSRandomMode(unittest.TestCase):
    """回归测试：rvns_mode="random" 必须等价于论文 RMOEA/D3 的随机选择 VNS。

    论文 Section 4.6 的用法 (1) 是「从五个算子里等概率随机选一个」。
    若 mode 失效（比如记忆仍然改变概率），这一臂就不再是论文的随机 VNS，
    我们与论文阶梯的对照随之失效——所以这个行为必须锁住。
    """

    def _fake_population(self, n=3):
        pop = [([0, 1, 0, 1], [0, 1, 0, 1]) for _ in range(n)]
        return pop

    def test_mode_stored(self):
        self.assertEqual(RVNS(mode="random").mode, "random")
        self.assertEqual(RVNS().mode, "rl")

    def test_random_mode_keeps_probabilities_uniform(self):
        rvns = RVNS(n_operators=5, mode="random")
        for op_idx, success in [(0, True), (0, True), (2, False), (3, True)]:
            rvns._update_memory(op_idx, success)
        rvns._update_probabilities()
        p = rvns.probabilities
        self.assertTrue(all(abs(v - 0.2) < 1e-12 for v in p), p)

    def test_rl_mode_does_learn_away_from_uniform(self):
        # 反向对照：同样的记忆在 rl 模式下必须偏离等概率，
        # 否则说明记忆机制根本没生效（测试会变成假阳性）
        rvns = RVNS(n_operators=5, mode="rl")
        for op_idx, success in [(0, True), (0, True), (2, False), (3, True)]:
            rvns._update_memory(op_idx, success)
        rvns._update_probabilities()
        p = rvns.probabilities
        self.assertGreater(abs(p[0] - 0.2), 1e-6, p)

    def test_random_mode_selects_all_operators(self):
        rvns = RVNS(n_operators=5, mode="random")
        seen = {int(rvns.select_operator(np.random.RandomState(s))) for s in range(50)}
        self.assertEqual(seen, {0, 1, 2, 3, 4})


class TestABABudgetAllocation(unittest.TestCase):
    """ABA 回归锁：等算力必须是**构造性**保证，不能被后续改动悄悄破坏。

    这组测试的核心是「四档策略的每代总预算严格相等」。这条一旦断了，
    `RVNSonly_Bstate vs RVNSonly_t2` 比出来的就不再是"预算发给谁"，
    而是"预算发多少"——整轮实验的归因随之失效。
    """

    N_POP = 10  # 偶数，使 n_pop*(target-base) 能被 (top-base) 整除

    def _rvns(self, mode, pool=(1, 3), target=2.0, ls_trials=1):
        return RVNS(n_operators=5, lp=40, ls_trials=ls_trials, mode="rl",
                    budget_mode=mode, budget_pool=list(pool),
                    budget_target_mean=target)

    def test_fixed_ignores_pool_and_uses_global_ls_trials(self):
        r = self._rvns("fixed", ls_trials=2)
        plan = r.plan_generation(self.N_POP, np.random.RandomState(0))
        self.assertTrue((plan == 2).all(), plan)

    def test_single_level_pool_degrades_to_fixed(self):
        """只有一个档位时不许假装自适应。"""
        r = self._rvns("pool_state", pool=(3,), target=3.0, ls_trials=3)
        self.assertEqual(r.budget_mode, "fixed")

    def test_equal_total_budget_across_modes(self):
        """四档策略的计划总预算必须逐代相等——本轮实验的地基。"""
        rng = np.random.RandomState(20260917)
        for mode in ("fixed", "pool_random", "pool_state", "pool_learn"):
            r = self._rvns(mode, ls_trials=2)
            total = 0
            for _ in range(5):
                plan = r.plan_generation(self.N_POP, rng)
                total += int(plan.sum())
                self.assertGreaterEqual(int(plan.min()), 1)
                self.assertLessEqual(int(plan.max()), 3)
            self.assertEqual(total, 5 * self.N_POP * 2, mode)

    def test_upgrade_quota_is_half(self):
        rng = np.random.RandomState(1)
        for mode in ("pool_random", "pool_state", "pool_learn"):
            plan = self._rvns(mode).plan_generation(self.N_POP, rng)
            self.assertEqual(int((plan == 3).sum()), self.N_POP // 2, mode)

    def test_state_upgrades_stuck_solutions(self):
        """上一代被改进过的解不应再抢升级名额。"""
        r = self._rvns("pool_state")
        n = self.N_POP
        r._last_success = [False] * (n // 2) + [True] * (n // 2)
        plan = r.plan_generation(n, np.random.RandomState(3))
        stuck = np.array([1] * (n // 2) + [0] * (n // 2))
        self.assertEqual(int((plan == 3).sum()), n // 2)
        self.assertEqual(int((plan[stuck == 1] == 3).sum()), n // 2,
                         "升级名额没有全部发给卡住的解")

    def test_learn_propensity_moves_to_rewarding_state(self):
        r = self._rvns("pool_learn")
        self.assertAlmostEqual(r.budget_propensities()[1], 0.5)  # Laplace 先验
        for _ in range(20):
            r._record_budget_outcome(True, True)     # stuck 时升级有回报
            r._record_budget_outcome(False, False)   # 不卡时升级没回报
        p = r.budget_propensities()
        self.assertGreater(p[1], p[0])

    def test_stats_expose_both_budget_and_real_evals(self):
        """两个口径都要在：一个是发放预算，一个是真实邻域求值次数。"""
        r = self._rvns("pool_state")
        r.plan_generation(self.N_POP, np.random.RandomState(0))
        st = r.get_budget_stats()
        self.assertEqual(st["planned_budget"], self.N_POP * 2)
        self.assertAlmostEqual(st["mean_planned_trials"], 2.0)
        self.assertIn("mean_actual_evals", st)
        self.assertIn("propensities", st)

    def test_switches_default_to_paper_behaviour(self):
        sig = inspect.signature(RMOEAD.__init__)
        self.assertEqual(sig.parameters["rvns_budget_mode"].default, "fixed")
        s = RMOEAD("Mk01", n_pop=8, max_gen=2, seed=0,
                   rvns_budget_mode="pool_state",
                   rvns_budget_pool=[1, 3], rvns_budget_target_mean=2.0)
        self.assertEqual(s.rvns.budget_mode, "pool_state")
        self.assertEqual(s.rvns.budget_target_mean, 2.0)

    def test_run_records_equal_compute_proof(self):
        """端到端：真实求解必须落盘预算统计，使"等算力"可被外部核对。"""
        s = RMOEAD("Mk01", n_pop=8, max_gen=3, seed=5,
                   enable_rvns=True, rvns_ls_trials=1,
                   rvns_budget_mode="pool_learn",
                   rvns_budget_pool=[1, 3], rvns_budget_target_mean=2.0)
        res = s.solve()
        b = res["rvns_budget"]
        self.assertIsNotNone(b)
        self.assertAlmostEqual(b["mean_planned_trials"], 2.0)
        self.assertEqual(b["planned_budget"], 8 * 2 * 3)
        self.assertGreater(b["actual_evals"], 0)
        self.assertEqual(res["components"]["rvns_budget_mode"], "pool_learn")


class TestCVDDModuleLevelRefactor(unittest.TestCase):
    """compute_cv_dv 提成模块级函数后，方法必须与之逐位等价。

    提出来是为了让**固定 T 的臂**也能逐代记录 CV/DV——G2「状态可观测性」
    完全依赖这条轨迹，而固定 T 的臂根本没有 Q-learning 对象。
    """

    PF = [(1.0, 2.0), (2.0, 1.0), (1.5, 1.5), (3.0, 0.5)]

    def test_method_delegates_to_module_function(self):
        from rmoea_d.core.qlearning import compute_cv_dv
        ql = QLearningPAS()
        self.assertEqual(ql.compute_cv_dv(self.PF), compute_cv_dv(self.PF))
        self.assertEqual(ql.compute_cv_dv(self.PF, normalize=True),
                         compute_cv_dv(self.PF, normalize=True))

    def test_algorithm_records_cv_dv_even_without_qpas(self):
        s = RMOEAD("Mk01", n_pop=8, max_gen=3, seed=3, fixed_T=10,
                   enable_rvns=False)
        res = s.solve()
        self.assertIsNone(res["q_table"], "固定 T 的臂不应有 Q-learning 对象")
        self.assertGreater(len(res["history"]), 0)
        self.assertTrue(all("cv" in h and "dv" in h for h in res["history"]))
        self.assertTrue(all(np.isfinite(h["cv"]) for h in res["history"]))


class TestHVBoxFingerprint(unittest.TestCase):
    """盒指纹：同一臂集同一批数据 → 同一指纹；数据变了 → 指纹必须变。

    绝对 HV 只在同一盒内可比，而盒随臂集漂移（同一个臂在项目里留下过三个
    不同的公开数值）。指纹就是"这两个绝对 HV 能不能比"的机器判据。
    """

    @staticmethod
    def _m():
        import importlib.util
        path = os.path.join(os.path.dirname(__file__), "..", "scripts", "hv_box.py")
        spec = importlib.util.spec_from_file_location("hv_box_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_fingerprint_is_deterministic(self):
        m = self._m()
        f = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[2.0, 3.0]])]
        self.assertEqual(m.fronts_fingerprint(f), m.fronts_fingerprint(f))

    def test_fingerprint_changes_with_data(self):
        m = self._m()
        self.assertNotEqual(m.fronts_fingerprint([np.array([[1.0, 2.0]])]),
                            m.fronts_fingerprint([np.array([[1.0, 2.5]])]))

    def test_assert_same_box_rejects_drift(self):
        m = self._m()
        b1 = m.make_box([np.array([[1.0, 2.0]])], arms=["A"])
        b2 = m.make_box([np.array([[1.0, 9.0]])], arms=["A"])
        self.assertNotEqual(b1["fronts_sha1"], b2["fronts_sha1"])
        with self.assertRaises(ValueError):
            m.assert_same_box(b1, b2)

    def test_box_records_arm_set(self):
        """盒必须带臂集——否则事后无法判断某绝对 HV 是哪个臂集算出来的。"""
        m = self._m()
        b = m.make_box([np.array([[1.0, 2.0]])], arms=["T50", "T10"])
        self.assertEqual(b["arms"], ["T10", "T50"])
        self.assertIn(b["fronts_sha1"], m.format_box(b))

    def test_sidecar_roundtrip(self):
        import tempfile
        m = self._m()
        b = m.make_box([np.array([[1.0, 2.0]])], arms=["A"])
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "lab.json")
            m.write_box_sidecar(src, b)
            self.assertEqual(m.read_box_sidecar(src)["fronts_sha1"],
                             b["fronts_sha1"])


class TestLadderSwitches(unittest.TestCase):
    """论文 6 级消融阶梯（Li et al. 2022, §5.4）所需的两个组件开关。

    论文的阶梯是**逐级累加**，相邻两级之差正好隔离一个组件：

        D1 纯 MOEA/D  →  D2 +MIX3  →  D3 +随机 VNS  →  D4 +Q-PAS
        →  D5 +Elite archive  →  RMOEA/D（把随机 VNS 换成 RVNS）

    在 `enable_mix3` / `enable_elite` 出现之前，代码只能表达 D3/D4/D5/RMOEA/D，
    D1 与 D2 无从构造（MIX3 与 Elite archive 是硬编码的）。这组测试锁住
    「开关确实改变行为」与「默认值不偏离论文」两件事。
    """

    def test_defaults_match_paper(self):
        sig = inspect.signature(RMOEAD.__init__)
        self.assertIs(sig.parameters["enable_mix3"].default, True)
        self.assertIs(sig.parameters["enable_elite"].default, True)

    def test_switches_stored(self):
        s = RMOEAD("Mk01", n_pop=8, max_gen=2, seed=0,
                   enable_mix3=False, enable_elite=False)
        self.assertFalse(s.enable_mix3)
        self.assertFalse(s.enable_elite)

    def test_baseline_passes_switches_through(self):
        sig = inspect.signature(MOEADBaseline.__init__)
        self.assertIn("enable_mix3", sig.parameters)
        self.assertIn("enable_elite", sig.parameters)

    def test_random_init_used_when_mix3_off(self):
        """D1 的初始化必须是纯随机，而不是 MIX3 的三个三分点。"""
        inst = load_instance("Mk01", os.path.join(os.path.dirname(__file__),
                                                  "..", "data"), seed=0)
        n_pop = 9
        # 同一 seed 下，D1（随机）与 D2（MIX3）的初始种群必须不同；
        # 且 D1 的每一个个体都应与 init_random 逐位相同（rng 流一致）。
        a = RMOEAD("Mk01", n_pop=n_pop, max_gen=1, seed=7, enable_mix3=False)
        a.instance = inst
        a.rng = np.random.RandomState(7)
        pop_off, _ = a._init_population()

        b = RMOEAD("Mk01", n_pop=n_pop, max_gen=1, seed=7, enable_mix3=True)
        b.instance = inst
        b.rng = np.random.RandomState(7)
        pop_on, _ = b._init_population()

        self.assertEqual(len(pop_off), n_pop)
        self.assertNotEqual([list(os_) for os_, _ in pop_off],
                            [list(os_) for os_, _ in pop_on])

        from rmoea_d.core.operators import init_random
        rng = np.random.RandomState(7)
        expect = [init_random(inst, rng) for _ in range(n_pop)]
        for got, exp in zip(pop_off, expect):
            self.assertEqual(list(got[0]), list(exp[0]))
            self.assertEqual(list(got[1]), list(exp[1]))

    def test_mix3_on_uses_three_strategies(self):
        """D2 的 MIX3 三个三分点必须都出现（否则退化成纯随机）。

        用 n_pop 能被 3 整除的规模，逐个比对 mix3 内的三条分支：
        前 1/3 = init_random、中 1/3 = init_ls、后 1/3 = init_gw。
        """
        inst = load_instance("Mk01", os.path.join(os.path.dirname(__file__),
                                                  "..", "data"), seed=0)
        n_pop = 9
        s = RMOEAD("Mk01", n_pop=n_pop, max_gen=1, seed=7, enable_mix3=True)
        s.instance = inst
        s.rng = np.random.RandomState(7)
        pop, _ = s._init_population()

        # GW 的机器选择是「当前负载增量最小」，随机初始化几乎不会复现；
        # 直接对照 GW 分支在无关 rng 状态下的机器分配特征（min t2 的 LS 分支同理）
        ma = [list(m) for _, m in pop]
        self.assertEqual(len(ma), n_pop)

        # 独立的 MIX3 复现（同一 rng 流）必须逐位相同 —— 证明三条分支都在被调用
        from rmoea_d.core.operators import init_mix3
        rng = np.random.RandomState(7)
        expect = init_mix3(inst, n_pop, rng)
        for got, exp in zip(pop, expect):
            self.assertEqual(list(got[0]), list(exp[0]))
            self.assertEqual(list(got[1]), list(exp[1]))

        # 反向对照：MIX3 的第二/第三段不是 init_random 生成的
        from rmoea_d.core.operators import init_random
        rng2 = np.random.RandomState(7)
        rnd = [init_random(inst, rng2) for _ in range(n_pop)]
        same = sum(1 for (go, _), (eo, _) in zip(pop, rnd) if list(go) == list(eo))
        self.assertLess(same, n_pop,
                        "MIX3 与纯随机初始化逐位相同 —— 三分点未生效")

    def test_archive_degrades_without_elite(self):
        """enable_elite=False 时档案不得跨代积累（= 末代种群的支配集）。"""
        inst = load_instance("Mk01", os.path.join(os.path.dirname(__file__),
                                                  "..", "data"), seed=0)
        s = RMOEAD("Mk01", n_pop=12, max_gen=1, seed=3, enable_elite=False)
        s.instance = inst
        pop, obj = s._init_population()
        s.archive = [(([0], [0]), ([0], [0]), (-999.0, -999.0))]   # 伪造的历史解
        s._update_archive(pop, obj)
        self.assertFalse(any(item[2] == (-999.0, -999.0) for item in s.archive),
                         "关闭精英档案后，历史解不得进入档案")

    def test_archive_accumulates_with_elite(self):
        """反向对照：enable_elite=True 时历史非支配解必须被保留。

        否则上面的测试会变成假阳性（档案本来就总是空）。
        """
        inst = load_instance("Mk01", os.path.join(os.path.dirname(__file__),
                                                  "..", "data"), seed=0)
        s = RMOEAD("Mk01", n_pop=12, max_gen=1, seed=3, enable_elite=True)
        s.instance = inst
        pop, obj = s._init_population()
        s.archive = [(([0], [0]), ([0], [0]), (-999.0, -999.0))]
        s._update_archive(pop, obj)
        self.assertTrue(any(item[2] == (-999.0, -999.0) for item in s.archive),
                        "启用精英档案时，历史非支配解必须被保留")

    def test_solve_resets_archive_between_calls(self):
        """solve() 必须清空上一次的档案，否则开关切换会带入残留状态。"""
        src = inspect.getsource(RMOEAD.solve)
        self.assertIn("self.archive = []", src)

    def test_result_reports_components(self):
        """结果 JSON 必须能反查该 run 的组件配置（阶梯审计需要）。"""
        src = inspect.getsource(RMOEAD.solve)
        for key in ('"mix3"', '"qpas"', '"rvns"', '"rvns_mode"', '"elite"',
                    '"fixed_T"'):
            self.assertIn(key, src, f"结果 JSON 缺少组件字段 {key}")


class TestLadderDefinition(unittest.TestCase):
    """`scripts/ablation_ladder.py` 的阶梯定义必须与论文 §5.4 逐级对应。"""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = os.path.join(os.path.dirname(__file__), "..", "scripts",
                            "ablation_ladder.py")
        if not os.path.exists(path):
            raise unittest.SkipTest("ablation_ladder.py 不存在")
        spec = importlib.util.spec_from_file_location("_abl_ladder", path)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_six_paper_arms_present(self):
        labels = [x[0] for x in self.mod.LADDER]
        for lbl in ("D1", "D2", "D3", "D4", "D5", "RMOEAD"):
            self.assertIn(lbl, labels)

    def test_adjacent_levels_isolate_one_component(self):
        """相邻两级必须只差一个组件——这是阶梯消融的全部意义。"""
        d = self.mod.LADDER_DICT
        self.assertEqual(d["D1"]["enable_mix3"], False)
        self.assertEqual(d["D2"]["enable_mix3"], True)
        self.assertEqual(d["D3"]["rvns_mode"], "random")
        self.assertEqual(d["D3"]["fixed_T"], 10)          # D4 才加 Q-PAS
        self.assertEqual(d["D4"]["fixed_T"], None)
        self.assertEqual(d["D4"]["enable_elite"], False)
        self.assertEqual(d["D5"]["enable_elite"], True)
        self.assertEqual(d["RMOEAD"]["rvns_mode"], "rl")

    def test_steps_cover_all_five_transitions(self):
        self.assertEqual(len(self.mod.STEPS), 5)
        self.assertEqual([s[2] for s in self.mod.STEPS],
                         ["MIX3 initialization", "random-selection VNS",
                          "Q-PAS", "Elite archive",
                          "RVNS (RL operator selection)"])

    def test_expected_reports_present(self):
        """每条臂都有对应论文变体名，报告要能直接对上 Table 4/5。"""
        src = open(os.path.join(os.path.dirname(__file__), "..", "scripts",
                                "ablation_ladder.py"), encoding="utf-8").read()
        self.assertIn("RMOEA/D1", src)
        self.assertIn("RMOEA/D5", src)


class TestQpasCVNormalization(unittest.TestCase):
    """Q-PAS 的 CV 归一化开关。

    论文式(14) 的 CV 用原始目标值，未规定归一化。当两个目标量纲悬殊时，
    CV 会被大量纲目标独占（实测 Mk10 上 f2 占 CV² 的 96.5%），ΔCV 的符号
    几乎只反映该目标的方向——状态对另一目标「隐形」。

    ``cv_normalize`` 是论文之外的可选项，默认必须为 False（复现优先）。
    这组测试同时锁住「默认不偏离论文」与「打开后确实抹平尺度」两件事。
    """

    PF = [(10.0, 100.0), (14.0, 160.0), (18.0, 240.0)]

    def test_default_keeps_paper_scale(self):
        self.assertFalse(QLearningPAS().cv_normalize)

    def test_rmoead_signature_defaults_to_paper_behavior(self):
        import inspect
        sig = inspect.signature(RMOEAD.__init__)
        self.assertIn("ql_cv_normalize", sig.parameters)
        self.assertIs(sig.parameters["ql_cv_normalize"].default, False)

    def test_unnormalized_cv_scales_linearly(self):
        # CV 是齐次一次的：目标整体放大 100 倍 -> CV 也放大 100 倍
        ql = QLearningPAS()
        cv_a, _ = ql.compute_cv_dv(self.PF)
        cv_b, _ = ql.compute_cv_dv([(a * 100.0, b * 100.0) for a, b in self.PF])
        self.assertAlmostEqual(cv_b / cv_a, 100.0, places=6)

    def test_unnormalized_cv_is_dominated_by_the_large_dimension(self):
        # 只放大第二个目标 -> CV 几乎完全跟着它走
        ql = QLearningPAS()
        cv_a, _ = ql.compute_cv_dv(self.PF)
        cv_b, _ = ql.compute_cv_dv([(a, b * 100.0) for a, b in self.PF])
        self.assertGreater(cv_b / cv_a, 50.0)

    def test_normalized_cv_is_scale_invariant(self):
        # 两个目标同比放大、归一化盒也同比放大 -> 点在盒中的相对位置不变
        # -> 归一化后的 CV 必须逐位相同
        cv_a, _ = QLearningPAS(
            cv_normalize=True, hv_bounds=([0.0, 0.0], [20.0, 300.0])
        ).compute_cv_dv(self.PF)
        cv_b, _ = QLearningPAS(
            cv_normalize=True, hv_bounds=([0.0, 0.0], [2000.0, 30000.0])
        ).compute_cv_dv([(a * 100.0, b * 100.0) for a, b in self.PF])
        self.assertAlmostEqual(cv_a, cv_b, places=12)

    def test_normalize_argument_overrides_the_instance_setting(self):
        ql = QLearningPAS(cv_normalize=False)
        cv_norm, _ = ql.compute_cv_dv(self.PF, normalize=True)
        cv_raw, _ = ql.compute_cv_dv(self.PF, normalize=False)
        self.assertNotAlmostEqual(cv_norm, cv_raw, places=6)


class TestLadderResumeMerge(unittest.TestCase):
    """断点续跑必须「写回 partial 里的既有进度」。

    曾经的缺陷：``done``（跳过判断）读 main + partial，但 ``rows``（写出的内容）
    只从 main 读。于是重启后「进行中实例」已完成的 run 会被跳过、却不出现在新
    写出的 partial 里 —— 进度静默丢失，下一批从零重算（实测 Mk08 的 220 条）。
    修法是把两处统一到 ``_load_merged``。
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = os.path.join(os.path.dirname(__file__), "..", "scripts",
                            "ablation_ladder.py")
        if not os.path.exists(path):
            raise unittest.SkipTest("ablation_ladder.py 不存在")
        spec = importlib.util.spec_from_file_location("_abl_ladder_r", path)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_merge_dedupes_and_keeps_both_sources(self):
        import json
        import tempfile
        d = tempfile.mkdtemp()

        def row(inst, lbl, seed):
            return dict(instance=inst, label=lbl, seed=seed)

        main = os.path.join(d, "m.json")
        part = os.path.join(d, "m.json.partial.json")
        # main 有 Mk01 的 2 条；partial 有 Mk01 的 1 条（重复）+ Mk02 的 2 条（新）
        json.dump([row("Mk01", "D1", 42), row("Mk01", "D1", 43)],
                  open(main, "w"))
        json.dump([row("Mk01", "D1", 43), row("Mk02", "D1", 42),
                   row("Mk02", "D1", 43)], open(part, "w"))

        merged = self.mod._load_merged(main, part)
        keys = {(r["instance"], r["label"], r["seed"]) for r in merged}
        self.assertEqual(len(merged), 4, "重复的 (inst,label,seed) 应被去重")
        self.assertIn(("Mk02", "D1", 42), keys)
        self.assertIn(("Mk02", "D1", 43), keys)
        # main 在前 -> 保留 main 的版本
        self.assertEqual(merged[0]["instance"], "Mk01")

    def test_merge_tolerates_missing_and_corrupt_files(self):
        import tempfile
        d = tempfile.mkdtemp()
        missing = os.path.join(d, "nope.json")
        corrupt = os.path.join(d, "bad.json")
        open(corrupt, "w").write("{not json")
        self.assertEqual(self.mod._load_merged(missing, corrupt), [])

    def test_rows_and_done_use_the_same_source(self):
        """源码层面锁住：跳过判断与写回内容必须用同一套合并口径。"""
        src = open(os.path.join(os.path.dirname(__file__), "..", "scripts",
                                "ablation_ladder.py"), encoding="utf-8").read()
        self.assertIn("_load_merged(args.out, partial)", src)


class TestLadderRunAll(unittest.TestCase):
    """分批驱动脚本 `scripts/ladder_run_all.py` 的进度统计。"""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = os.path.join(os.path.dirname(__file__), "..", "scripts",
                            "ladder_run_all.py")
        if not os.path.exists(path):
            raise unittest.SkipTest("ladder_run_all.py 不存在")
        spec = importlib.util.spec_from_file_location("_abl_runall", path)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_done_set_unions_main_and_partial(self):
        import json
        import tempfile
        d = tempfile.mkdtemp()
        out = os.path.join(d, "r.json")

        def row(inst, lbl, seed):
            return dict(instance=inst, label=lbl, seed=seed)

        json.dump([row("Mk01", "D1", 42), row("Mk01", "D1", 43)], open(out, "w"))
        json.dump([row("Mk01", "D1", 43), row("Mk02", "D1", 42)],
                  open(out + ".partial.json", "w"))
        got = self.mod.done_set(out)
        self.assertEqual(len(got), 3, "重复键应被去重，两处来源都要算")
        self.assertIn(("Mk02", "D1", 42), got)

    def test_done_set_tolerates_missing_files(self):
        import tempfile
        out = os.path.join(tempfile.mkdtemp(), "nope.json")
        self.assertEqual(self.mod.done_set(out), set())

    def test_smoke_promoted_driver_exists(self):
        """驱动脚本要能 import 到实验台本体，避免路径写错。"""
        self.assertTrue(os.path.exists(self.mod.SCRIPT),
                        f"SCRIPT 路径不存在: {self.mod.SCRIPT}")


class TestLadderAnalysisPartialData(unittest.TestCase):
    """阶梯分析脚本必须能吃「部分完成」的数据集。

    真实长跑里实例是**一个接一个**跑完的（见 ``_run_ladder_batches.py``），
    中途被杀会留下「某个实例只跑了部分 seed」的 JSON。早期版本的
    ``M[lbl]`` 直接用该实例的 seed 并集取均值，缺 seed 的臂会 ``KeyError``
    （实测 Mk08 只跑 220/240 时崩在 ``KeyError: 52``）。
    现在改为**跨臂取 seed 交集**，并显式提示被剔除的 seed。
    """

    def _run_analysis(self, rows):
        import json
        import subprocess
        import tempfile
        root = os.path.join(os.path.dirname(__file__), "..")
        script = os.path.join(root, "scripts", "ablation_ladder_analysis.py")
        if not os.path.exists(script):
            self.skipTest("ablation_ladder_analysis.py 不存在")
        d = tempfile.mkdtemp()
        lab = os.path.join(d, "lab.json")
        with open(lab, "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
        out = os.path.join(d, "lab.json.analysis.json")
        r = subprocess.run(
            [sys.executable, script, "--lab_json", lab, "--out_json", out],
            cwd=root, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=300)
        return r, out

    @staticmethod
    def _rows():
        """2 实例 x 4 臂 x 6 seed；故意让 Mk02 的 D5 臂少 2 个 seed。"""
        rows = []
        arms = ["D1", "D2", "D5", "D5_fixedT"]
        for inst_i, inst in enumerate(("Mk01", "Mk02")):
            for a_i, lbl in enumerate(arms):
                n = 6
                if inst == "Mk02" and lbl == "D5":
                    n = 4                       # 缺 seed 70/71
                for k in range(n):
                    seed = 42 + k
                    base = 10.0 + a_i * 2.0 + inst_i
                    # 小前沿：HV 只关心形状，不需要真实调度解
                    pf = [[float(base + j), float(20.0 - base - 2 * j)]
                          for j in range(4)]
                    rows.append(dict(instance=inst, label=lbl, seed=seed,
                                     final_pf=pf, hist_T=[5] * 5,
                                     final_hv=0.5, components={}))
        return rows

    def test_partial_instance_does_not_crash(self):
        r, out = self._run_analysis(self._rows())
        self.assertEqual(r.returncode, 0,
                         f"部分数据集不应崩溃\nstdout={r.stdout}\nstderr={r.stderr}")
        self.assertTrue(os.path.exists(out), "未写出分析 JSON")

    def test_warns_about_dropped_seeds(self):
        r, _ = self._run_analysis(self._rows())
        self.assertIn("seed", r.stdout)
        self.assertIn("Mk02", r.stdout)

    def test_uses_seed_intersection_not_union(self):
        """报告里的**配对观测数**必须按交集算：Mk01 交集 6 + Mk02 交集 4 = 10。
        若按并集（每实例 6）会得到 12，故 10 能区分两种口径。
        总 run 数另算，且不受交集影响：Mk01 4臂x6 + Mk02 (3臂x6 + 1臂x4) = 46。"""
        import json
        r, out = self._run_analysis(self._rows())
        payload = json.load(open(out, encoding="utf-8"))
        self.assertEqual(payload["n_paired_obs"], 6 + 4)
        self.assertEqual(payload["n_total_runs"], 24 + 22)


class TestRelativeGainConvention(unittest.TestCase):
    """图、表、审计脚本三处的「相对增幅」必须是同一个估计量。

    曾经的缺陷：``ablation_ladder_analysis.py`` 已统一为
    ``mean(ΔHV)/mean(基线)``（比值之比），但 ``ladder_plot.py`` 仍用
    ``mean(ΔHV/基线)``（各实例相对增幅的平均）。
    后果是**同一个量在图上是 +16.47%、在文档表格里是 +14.46%**（差 2 个百分点），
    因为比值之比与比值的均值在基线各实例不同的时候并不相等。
    更糟的是旧口径在效应≈0 时可能与 ΔHV 反号 —— 正是上一轮修掉的毛病。
    """

    ROOT = os.path.join(os.path.dirname(__file__), "..")

    def test_plot_uses_ratio_of_means(self):
        """源码层面锁住：不得再用 `np.mean(d / M[...])` 这种『比值的均值』。"""
        src = open(os.path.join(self.ROOT, "scripts", "ladder_plot.py"),
                   encoding="utf-8").read()
        self.assertNotIn("np.mean(d / M[", src,
                         "ladder_plot.py 又用回了『比值的均值』，会与表/文档不一致")
        self.assertIn("d.mean() / M[", src)

    def test_plot_summary_matches_analysis_for_the_same_step(self):
        """端到端：绘图脚本打印的 ΔvsD1 必须等于分析脚本给出的该级 rel_pct。"""
        import json
        import subprocess
        import tempfile
        script = os.path.join(self.ROOT, "scripts", "ladder_plot.py")
        if not os.path.exists(script):
            self.skipTest("ladder_plot.py 不存在")
        rows = TestLadderAnalysisPartialData._rows()
        d = tempfile.mkdtemp()
        lab = os.path.join(d, "lab.json")
        with open(lab, "w", encoding="utf-8") as fh:
            json.dump(rows, fh)

        # 分析侧：D1->D2 这一级的相对增幅
        an = TestLadderAnalysisPartialData()
        r, out = an._run_analysis(rows)
        self.assertEqual(r.returncode, 0, r.stderr)
        step = [s for s in json.load(open(out, encoding="utf-8"))["steps_instance_level"]
                if s["step"] == "D1->D2"]
        self.assertTrue(step, "分析结果里应含 D1->D2")
        want = step[0]["rel_pct"]

        # 绘图侧：摘要里 D2 那一行的 ΔvsD1
        rp = subprocess.run(
            [sys.executable, script, "--lab_json", lab,
             "--out", os.path.join(d, "fig.png")],
            cwd=self.ROOT, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=300)
        self.assertEqual(rp.returncode, 0, rp.stderr)
        got = None
        for line in rp.stdout.splitlines():
            if line.strip().startswith("D2 "):
                got = float(line.split("ΔvsD1=")[1].split("%")[0].replace("+", ""))
        self.assertIsNotNone(got, f"摘要里没找到 D2 行\n{rp.stdout}")
        self.assertAlmostEqual(got, want, places=2,
                               msg=f"图上 {got:+.2f}% vs 表里 {want:+.2f}%")


class TestLadderInstanceValidation(unittest.TestCase):
    """实例名必须早校验。

    曾经的缺陷：文档与 docstring 给的示例是 `--instances Mk01,...,Mk10`，
    而 `...` 会被当成一个实例名传给每个 job。由于失败只在子进程里发生，
    且 `requested` 永远凑不齐，结果文件**永远不会提升为主文件** ——
    表现是"跑了一整轮却什么都没落盘"。
    """

    ROOT = os.path.join(os.path.dirname(__file__), "..")

    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = os.path.join(os.path.dirname(__file__), "..", "scripts",
                            "ablation_ladder.py")
        if not os.path.exists(path):
            raise unittest.SkipTest("ablation_ladder.py 不存在")
        spec = importlib.util.spec_from_file_location("_abl_ladder_v", path)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_available_instances_lists_data_dir(self):
        got = self.mod._available_instances("data")
        if got is None:
            self.skipTest("data/ 目录不存在")
        self.assertIn("Mk01", got)
        self.assertIn("Mk10", got)
        self.assertNotIn("...", got)

    def test_missing_data_dir_returns_none(self):
        self.assertIsNone(self.mod._available_instances("no_such_dir_xyz"))

    def test_ellipsis_is_rejected_early(self):
        """`--instances Mk01,...,Mk10` 必须在跑任何 run 之前就失败。"""
        import subprocess
        script = os.path.join(self.ROOT, "scripts", "ablation_ladder.py")
        r = subprocess.run(
            [sys.executable, script, "--instances", "Mk01,...,Mk10",
             "--seeds", "1", "--workers", "1"],
            cwd=self.ROOT, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=300)
        self.assertNotEqual(r.returncode, 0, "带 '...' 的实例名应被拒绝")
        self.assertIn("...", r.stderr)

    def test_source_mentions_early_validation(self):
        src = open(os.path.join(self.ROOT, "scripts", "ablation_ladder.py"),
                   encoding="utf-8").read()
        self.assertIn("_available_instances(args.data_dir)", src)


class TestLadderRunAllTimeout(unittest.TestCase):
    """单批超时必须被当成『本批无进展』，而不是让驱动崩掉。

    曾经的缺陷：`subprocess.run(..., timeout=...)` 抛出的 `TimeoutExpired`
    没有捕获，一次慢批就会让整个驱动带着 traceback 退出，
    **剩余实例一个都不跑**，`--attempts` 重试机制形同虚设。
    """

    ROOT = os.path.join(os.path.dirname(__file__), "..")

    def test_driver_catches_timeout(self):
        src = open(os.path.join(self.ROOT, "scripts", "ladder_run_all.py"),
                   encoding="utf-8").read()
        self.assertIn("except subprocess.TimeoutExpired", src)
        self.assertIn("attempts", src)

    def test_driver_exposes_per_batch_timeout(self):
        src = open(os.path.join(self.ROOT, "scripts", "ladder_run_all.py"),
                   encoding="utf-8").read()
        self.assertIn("--per_batch_timeout", src)


class TestPaperCmpPlot(unittest.TestCase):
    """对照图脚本必须是**入库的**脚本，且配对要按 seed 取交集。"""

    ROOT = os.path.join(os.path.dirname(__file__), "..")

    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = os.path.join(os.path.dirname(__file__), "..", "scripts",
                            "paper_cmp_plot.py")
        if not os.path.exists(path):
            raise unittest.SkipTest("paper_cmp_plot.py 不存在")
        spec = importlib.util.spec_from_file_location("_paper_cmp", path)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_script_is_versioned_not_in_logs(self):
        """文档引用的复现脚本不能放在 `logs/`（那里被 gitignore）。"""
        self.assertTrue(os.path.exists(
            os.path.join(self.ROOT, "scripts", "paper_cmp_plot.py")))
        src = open(os.path.join(self.ROOT, "docs", "paper-vs-reproduction.md"),
                   encoding="utf-8").read()
        self.assertIn("scripts/paper_cmp_plot.py", src)
        self.assertNotIn("python logs/_plot_paper_cmp.py", src)

    def test_pairing_uses_seed_intersection(self):
        """两臂 seed 集不同时，必须按交集配对，不能按位置硬减。"""
        hv = {"a": {42: 0.10, 43: 0.12, 44: 0.14, 45: 0.16},
              "b": {43: 0.10, 44: 0.11, 45: 0.12, 46: 0.99}}
        v, e, p = self.mod.paired_stats(hv, "a", "b")
        # 交集 {43,44,45}：d = [.02,.03,.04] -> mean .03，基线 b 均值 .11
        self.assertAlmostEqual(v, 100.0 * 0.03 / 0.11, places=9)
        self.assertTrue(0.0 <= p <= 1.0)

    def test_too_few_pairs_yields_nan_not_a_crash(self):
        """交集不足时给 nan。旧写法会 numpy 广播报错（4 vs 1）。"""
        hv = {"a": {42: 0.1, 43: 0.2, 44: 0.3, 45: 0.4},
              "b": {42: 0.1}}
        v, e, p = self.mod.paired_stats(hv, "a", "b")
        self.assertTrue(v != v and e != e and p != p, "应为 nan")


class TestSweepPerArmGridOverride(unittest.TestCase):
    """逐臂覆盖 (n_pop, max_gen)：等算力臂靠它把 G 从 200 提到 440。

    这类"看起来能用但实际没生效"的开关最贵——`_max_gen` 若没被 pop 掉，
    RMOEAD 会因为重复关键字参数直接 TypeError（好一点）；但若被 pop 掉后
    **没被用上**，臂就会静默地跑成普通 G=200，"等算力"结论全废。
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = os.path.join(os.path.dirname(__file__), "..", "scripts",
                            "t_leverage_sweep.py")
        if not os.path.exists(path):
            raise unittest.SkipTest("t_leverage_sweep.py 不存在")
        spec = importlib.util.spec_from_file_location("_tl_sweep", path)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_default_passthrough_when_no_override(self):
        extra, n_pop, mg = self.mod.resolve_grid(dict(fixed_T=10), 100, 200)
        self.assertEqual((n_pop, mg), (100, 200))
        self.assertEqual(extra, dict(fixed_T=10))

    def test_override_is_consumed_and_applied(self):
        extra, n_pop, mg = self.mod.resolve_grid(
            dict(fixed_T=10, _max_gen=440, _n_pop=50), 100, 200)
        self.assertEqual((n_pop, mg), (50, 440))
        # 关键：两个私有键必须已被 pop，否则会当未知 kwargs 传给 RMOEAD
        self.assertNotIn("_max_gen", extra)
        self.assertNotIn("_n_pop", extra)
        self.assertEqual(extra, dict(fixed_T=10))

    def test_partial_override(self):
        _, n_pop, mg = self.mod.resolve_grid(dict(_max_gen=440), 100, 200)
        self.assertEqual((n_pop, mg), (100, 440))

    def test_equal_compute_arms_declare_override(self):
        """等算力臂必须在 ARM_DEF 里写死 _max_gen，否则它不是等算力的。"""
        for lbl in ("T10_G440", "RVNSonly_G440", "RandVNS_G440"):
            self.assertIn(lbl, self.mod.ARM_DEF)
            kwargs = self.mod.ARM_DEF[lbl][1]
            self.assertIn("_max_gen", kwargs, lbl)
            self.assertGreaterEqual(kwargs["_max_gen"], 400, lbl)


class TestABAHoldoutPaired(unittest.TestCase):
    """留出集确认脚本的配对统计：按 seed 交集、dz 符号、相对差口径。

    这里的口径直接决定 H1 的结论，写错了会读出不存在的效应。
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = os.path.join(os.path.dirname(__file__), "..", "scripts",
                            "aba_holdout.py")
        if not os.path.exists(path):
            raise unittest.SkipTest("aba_holdout.py 不存在")
        spec = importlib.util.spec_from_file_location("_aba_ho", path)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    HV = {"a": {42: 0.10, 43: 0.12, 44: 0.14, 45: 0.16},
          "b": {43: 0.10, 44: 0.11, 45: 0.12, 46: 0.99}}

    def test_uses_seed_intersection_only(self):
        r = self.mod.paired(self.HV, "a", "b", [42, 43, 44, 45])
        self.assertEqual(r["seeds"], [43, 44, 45])
        self.assertEqual(r["n"], 3)
        self.assertEqual(r["wins"], 3)
        # d = [.02,.03,.04] -> mean .03；相对差按 **baseline(b)** 均值 .11 缩放
        self.assertAlmostEqual(r["d"], 0.03, places=9)
        self.assertAlmostEqual(r["base_mean"], 0.11, places=9)
        self.assertAlmostEqual(r["rel"], 100.0 * 0.03 / 0.11, places=9)
        self.assertGreater(r["dz"], 0.0)

    def test_no_shared_seeds_returns_none(self):
        self.assertIsNone(self.mod.paired(self.HV, "a", "b", [42]))

    def test_identical_arms_give_nan_p_not_a_crash(self):
        """全零差时 Wilcoxon 会 RuntimeWarning 并返回 nan——不能崩。"""
        r = self.mod.paired(self.HV, "a", "a", [42, 43, 44, 45])
        self.assertTrue(r["p"] != r["p"], "应为 nan")
        self.assertEqual(r["d"], 0.0)

    def test_pooling_unit_is_paired_observation_not_instance_mean(self):
        """合并口径必须是 (instance, seed) 配对观测。若误压成实例均值，
        n 会退化成"实例数"（2），任何检验都无意义——这正是本轮修掉的缺陷。
        这里用源码断言把它锁住。"""
        src = open(os.path.join(os.path.dirname(__file__), "..", "scripts",
                                "aba_holdout.py"), encoding="utf-8").read()
        self.assertIn('pooled[(a, other)].append(', src)
        self.assertIn('(inst, s, float(dv) / r["base_mean"] * 100.0)', src)
        self.assertNotIn('pooled[(a, other)].append((inst, r["rel"]))', src)


class TestEqualComputeVerdict(unittest.TestCase):
    """等算力判定：时间比 + 显著性 -> 结论。

    这组判定是"最大杠杆是不是只是算力"的唯一判据，写错方向会得到相反结论。
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = os.path.join(os.path.dirname(__file__), "..", "scripts",
                            "eqc_compare.py")
        if not os.path.exists(path):
            raise unittest.SkipTest("eqc_compare.py 不存在")
        spec = importlib.util.spec_from_file_location("_eqc", path)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_time_matched_and_better_is_not_a_compute_effect(self):
        self.assertEqual(self.mod._verdict(1.0, True, False)[0], "非算力效应")

    def test_time_matched_and_tied_is_a_compute_effect(self):
        self.assertEqual(self.mod._verdict(1.0, False, False)[0], "算力效应")

    def test_more_compute_and_better_is_inconclusive_not_evidence(self):
        """A 多花 2x 算力还更好——不能据此宣称'不是算力效应'。"""
        self.assertEqual(self.mod._verdict(2.0, True, False)[0], "不可判")

    def test_less_compute_and_better_is_strong_evidence(self):
        self.assertEqual(self.mod._verdict(0.5, True, False)[0], "强证据-非算力")

    def test_match_band_endpoints_are_inclusive(self):
        lo, hi = self.mod.MATCH_LO, self.mod.MATCH_HI
        self.assertEqual(self.mod._verdict(lo, False, False)[0], "算力效应")
        self.assertEqual(self.mod._verdict(hi, False, False)[0], "算力效应")

    def test_rounding_does_not_flip_the_band(self):
        """1.51x / 2.19x 这类过冲必须落到'不可判/强证据'，不能落进匹配带。"""
        self.assertNotEqual(self.mod._verdict(1.51, True, False)[0], "非算力效应")
        self.assertNotEqual(self.mod._verdict(2.19, True, False)[0], "非算力效应")

    def test_evals_matched_ignores_wallclock_ratio(self):
        """求值次数口径下，墙钟比 2x 不是混淆而是结果，不能判成'不可判'。"""
        self.assertEqual(self.mod._verdict(2.0, False, False, "evals")[0], "次数无效")
        self.assertEqual(self.mod._verdict(2.0, True, False, "evals")[0], "次数有效")

    def test_none_match_is_reference_only(self):
        """算力不匹配的对照绝不能用来判定。"""
        self.assertEqual(self.mod._verdict(1.45, True, False, "none")[0], "仅参照")

    def test_every_pair_declares_a_match_dimension(self):
        """每个对照必须显式声明匹配维度，防止有人后来把不匹配的对照当结论。"""
        for p in self.mod.PAIRS:
            self.assertEqual(len(p), 4, p)
            self.assertIn(p[2], ("time", "evals", "none"), p)


class TestRepairFastPath(unittest.TestCase):
    """`_repair_ma_for_os` 热路径改写的等价性锁。

    改写把 `rng.choice(list(set))` 换成「预缓存候选列表 + `randint` 索引」
    （`RandomState.choice` 即便 `size=None` 也走 `np.prod(size)` 通用路径，
    实测 8.2us/次 vs `randint` 2.0us/次）。这里锁死两点：输出逐位相同、
    **rng 状态逐位相同**。后者是关键 —— 只要随机流变了，Mk10/Mk07/Mk09 的
    全部历史实验数据就作废，所有已发布结论都要重跑。
    """

    @staticmethod
    def _reference_repair(os_vec, ma_vec, instance, rng):
        """优化前的逐位实现（原样保留作为参考）。"""
        ma_vec = ma_vec.copy()
        op_counter = [0] * instance["n_jobs"]
        valid_machines = instance["valid_machines"]
        for idx, job_id in enumerate(os_vec):
            oi = op_counter[job_id]
            op_counter[job_id] += 1
            if ma_vec[idx] not in valid_machines[job_id][oi]:
                ma_vec[idx] = rng.choice(list(valid_machines[job_id][oi]))
        return ma_vec

    @staticmethod
    def _random_case(instance, seed):
        """构造「多数机器不合法」的 (os, ma)，逼出回退分支。"""
        rng = np.random.RandomState(seed)
        os_vec = []
        for j, ops in enumerate(instance["jobs"]):
            os_vec.extend([j] * len(ops))
        rng.shuffle(os_vec)
        ma_vec = [int(x) for x in rng.randint(0, instance["n_machines"],
                                              size=len(os_vec))]
        return os_vec, ma_vec

    def test_output_and_rng_stream_match_reference(self):
        from rmoea_d.core.operators import _repair_ma_for_os
        for inst_name in ("Mk07", "Mk10"):
            inst = load_instance(inst_name, "data", 42)
            for trial in range(15):
                os_vec, ma_vec = self._random_case(inst, 300 + trial)
                ra_rng = np.random.RandomState(7 + trial)
                rb_rng = np.random.RandomState(7 + trial)
                ra = self._reference_repair(os_vec, ma_vec, inst, ra_rng)
                rb = _repair_ma_for_os(os_vec, ma_vec, inst, rb_rng)
                self.assertEqual([int(x) for x in ra], [int(x) for x in rb],
                                 "repair 输出不一致 (%s trial=%d)"
                                 % (inst_name, trial))
                sa, sb = ra_rng.get_state(), rb_rng.get_state()
                self.assertEqual(sa[2], sb[2],
                                 "rng 内部位置不一致 (%s trial=%d)"
                                 % (inst_name, trial))
                self.assertTrue(np.array_equal(sa[1], sb[1]),
                                "rng 状态缓冲不一致 (%s trial=%d)"
                                % (inst_name, trial))

    def test_reference_really_consumes_rng(self):
        """反向对照：不合法输入必须真的让参考实现消耗 rng。

        若这一条不成立，说明测试用例根本没走到回退分支，
        上面「rng 流相同」就退化成永真的空断言。
        """
        inst = load_instance("Mk10", "data", 42)
        os_vec, ma_vec = self._random_case(inst, 999)
        rng = np.random.RandomState(3)
        before = rng.get_state()[2]
        for _ in range(30):
            self._reference_repair(os_vec, ma_vec, inst, rng)
        self.assertNotEqual(before, rng.get_state()[2])

    def test_fallback_cache_equals_freshly_built_list(self):
        """预缓存 `_fallback_candidates` 必须与现场 `list(set)` 逐位相同。"""
        from rmoea_d.core.operators import fallback_candidates
        inst = load_instance("Mk07", "data", 42)
        cached = fallback_candidates(inst)
        for j, job_valid in enumerate(inst["valid_machines"]):
            for oi, machines in enumerate(job_valid):
                self.assertEqual(cached[j][oi], list(machines))


class TestLSOutputsAreValid(unittest.TestCase):
    """LS1~LS3 的输出对「未变的 OS」必然合法 —— 据此才能跳过 repair。

    `rvns.apply_local_search` 现在只在 `LS_MAY_INVALIDATE_MA[op_idx]` 为真时
    才调用 `_repair_ma_for_os`。若 LS1~LS3 真能产出非法机器，跳过就会改变结果。
    """

    @staticmethod
    def _valid(os_vec, ma_vec, instance):
        op_counter = [0] * instance["n_jobs"]
        vm = instance["valid_machines"]
        for idx, job_id in enumerate(os_vec):
            oi = op_counter[job_id]
            op_counter[job_id] += 1
            if ma_vec[idx] not in vm[job_id][oi]:
                return False
        return True

    @staticmethod
    def _legal_case(instance, seed):
        from rmoea_d.core.operators import _repair_ma_for_os
        rng = np.random.RandomState(seed)
        os_vec = []
        for j, ops in enumerate(instance["jobs"]):
            os_vec.extend([j] * len(ops))
        rng.shuffle(os_vec)
        ma_vec = [int(x) for x in rng.randint(0, instance["n_machines"],
                                              size=len(os_vec))]
        return os_vec, _repair_ma_for_os(os_vec, ma_vec, instance,
                                         np.random.RandomState(seed + 1))

    def test_ls1_ls2_ls3_never_produce_illegal_machine(self):
        from rmoea_d.core.rvns import (ls1_swap_machine, ls2_min_time_machine,
                                       ls3_max_workload_machine)
        for inst_name in ("Mk01", "Mk07", "Mk10"):
            inst = load_instance(inst_name, "data", 42)
            for fn in (ls1_swap_machine, ls2_min_time_machine,
                       ls3_max_workload_machine):
                for trial in range(25):
                    os_vec, ma_vec = self._legal_case(inst, 500 + trial)
                    self.assertTrue(self._valid(os_vec, ma_vec, inst),
                                    "测试输入本身不合法")
                    new_os, new_ma = fn(os_vec, ma_vec, inst,
                                        np.random.RandomState(900 + trial))
                    self.assertTrue(
                        self._valid(new_os, new_ma, inst),
                        "%s 在 %s 上产出了非法机器 (trial=%d)"
                        % (fn.__name__, inst_name, trial))

    def test_flag_marks_exactly_the_order_breaking_operators(self):
        """LS4 交换 / LS5 插入会迁移 (job, oi) → 必须保留 repair。"""
        from rmoea_d.core.rvns import LS_MAY_INVALIDATE_MA
        self.assertEqual(LS_MAY_INVALIDATE_MA,
                         [False, False, False, True, True])


class TestHotPathEquivalence(unittest.TestCase):
    """其余三处热路径改写的等价性锁。"""

    def test_decode_crisp_matches_reference(self):
        """zip 版 decode_crisp 与原「下标计数器」版数值逐位相同。"""
        from rmoea_d.core.encoding import decode_crisp
        from rmoea_d.core.operators import _repair_ma_for_os

        def reference(os_vec, ma_vec, instance):
            n_jobs = instance["n_jobs"]
            n_machines = instance["n_machines"]
            crisp_times = instance["crisp_times"]
            op_counter = [0] * n_jobs
            job_ready = [0.0] * n_jobs
            machine_ready = [0.0] * n_machines
            total_workload = 0.0
            op_idx_global = 0
            for job_id in os_vec:
                oi = op_counter[job_id]
                op_counter[job_id] += 1
                chosen_m = ma_vec[op_idx_global]
                op_idx_global += 1
                ptime = crisp_times[job_id][oi][chosen_m]
                start = (job_ready[job_id] if job_ready[job_id] > machine_ready[chosen_m]
                         else machine_ready[chosen_m])
                finish = start + ptime
                job_ready[job_id] = finish
                machine_ready[chosen_m] = finish
                total_workload += ptime
            makespan = job_ready[0]
            for t in job_ready[1:]:
                if t > makespan:
                    makespan = t
            return makespan, total_workload

        for inst_name in ("Mk01", "Mk10"):
            inst = load_instance(inst_name, "data", 42)
            rng = np.random.RandomState(11)
            for _ in range(20):
                os_vec = []
                for j, ops in enumerate(inst["jobs"]):
                    os_vec.extend([j] * len(ops))
                rng.shuffle(os_vec)
                ma_vec = [int(x) for x in
                          rng.randint(0, inst["n_machines"], size=len(os_vec))]
                ma_vec = _repair_ma_for_os(os_vec, ma_vec, inst, rng)
                self.assertEqual(reference(os_vec, ma_vec, inst),
                                 decode_crisp(os_vec, ma_vec, inst))

    def test_moead_generation_matches_reference(self):
        """内联 Tchebycheff / 权重转 Python list 之后，整代 MOEA/D 更新逐位相同。"""
        import copy
        from rmoea_d.core.encoding import decode_crisp
        from rmoea_d.core.moead import (compute_neighbors, generate_weights,
                                        moead_generation, tchebycheff)
        from rmoea_d.core.operators import (init_mix3, mutate_ma, mutate_os,
                                            pox_crossover, repair_os,
                                            ux_crossover, _repair_ma_for_os)

        def reference(population, objectives, weights, B, instance, z,
                      crossover_rate, rng):
            n_pop = len(population)
            new_pop = [p for p in population]
            new_obj = [list(o) for o in objectives]
            z = list(z)
            for i in range(n_pop):
                neighbors = B[i]
                if len(neighbors) < 2:
                    continue
                p1_idx, p2_idx = rng.choice(neighbors, 2, replace=False)
                os1, ma1 = new_pop[p1_idx]
                os2, ma2 = new_pop[p2_idx]
                if rng.rand() < crossover_rate:
                    child_os, _ = pox_crossover(os1, os2, rng)
                    child_ma, _ = ux_crossover(ma1, ma2, rng)
                    child_ma = _repair_ma_for_os(child_os, child_ma, instance, rng)
                else:
                    child_os = os1.copy()
                    child_ma = ma1.copy()
                child_os = mutate_os(child_os, rng)
                child_os = repair_os(child_os, instance)
                child_ma = _repair_ma_for_os(child_os, child_ma, instance, rng)
                child_ma = mutate_ma(child_ma, child_os, instance, rng)
                mc, wc = decode_crisp(child_os, child_ma, instance)
                f = [mc, wc]
                z[0] = min(z[0], f[0])
                z[1] = min(z[1], f[1])
                for j in neighbors:
                    w = weights[j]
                    old_g = tchebycheff(new_obj[j], w, z)
                    new_g = tchebycheff(f, w, z)
                    if new_g < old_g:
                        new_pop[j] = (child_os, child_ma)
                        new_obj[j] = f
            return new_pop, [tuple(o) for o in new_obj], tuple(z)

        inst = load_instance("Mk10", "data", 42)
        weights = generate_weights(30)
        B = compute_neighbors(weights, 10)
        pop = init_mix3(inst, 30, np.random.RandomState(21))
        obj = [decode_crisp(os_v, ma_v, inst) for os_v, ma_v in pop]
        z0 = (min(o[0] for o in obj), min(o[1] for o in obj))

        ra_rng = np.random.RandomState(77)
        rb_rng = np.random.RandomState(77)
        p1, o1, zz1 = moead_generation(copy.deepcopy(pop), list(obj), weights, B,
                                       inst, z0, 0.9, ra_rng)
        p2, o2, zz2 = reference(copy.deepcopy(pop), list(obj), weights, B,
                                inst, z0, 0.9, rb_rng)

        self.assertEqual(o1, o2, "目标值不一致")
        self.assertEqual(zz1, zz2, "参考点不一致")
        self.assertEqual([(x[0], x[1]) for x in p1], [(x[0], x[1]) for x in p2],
                         "种群解不一致")
        self.assertEqual(ra_rng.get_state()[2], rb_rng.get_state()[2],
                         "rng 内部位置不一致")
        self.assertTrue(np.array_equal(ra_rng.get_state()[1], rb_rng.get_state()[1]),
                        "rng 状态缓冲不一致")

    def test_neighbor_cache_is_per_T_and_deterministic(self):
        """邻居只依赖 (weights, T)：同 T 必须复用同一份结构。"""
        from rmoea_d.core.moead import compute_neighbors, generate_weights
        weights = generate_weights(50)
        self.assertEqual(compute_neighbors(weights, 20),
                         compute_neighbors(weights, 20))
        solver = RMOEAD("Mk01", n_pop=20, max_gen=3, seed=1,
                        enable_rvns=False, fixed_T=10)
        solver.solve()
        self.assertEqual(list(solver._neighbor_cache.keys()), [10])

    def test_optimization_did_not_change_algorithm_semantics(self):
        """热路径优化不得顺手改掉算法默认口径。"""
        sig = inspect.signature(RMOEAD.__init__)
        self.assertIs(sig.parameters["enable_mix3"].default, True)
        self.assertIs(sig.parameters["enable_rvns"].default, True)
        self.assertIsNone(sig.parameters["fixed_T"].default)


class TestInitVariants(unittest.TestCase):
    """初始化变体族（`init_variant`）的等价性与合法性锁。

    动机：MIX3 是消融阶梯里最大的单一组件（+14.46%***），却**从未做过变体扫描**；
    而它三条分支（random / LS / GW）全部只优化 MA 维度，把 OS 随机打乱 ——
    工序顺序这一维在初始化阶段完全没被利用。
    """

    @staticmethod
    def _expect_os(instance):
        expect = []
        for j, ops in enumerate(instance["jobs"]):
            expect.extend([j] * len(ops))
        return sorted(expect)

    def test_mix3_variant_is_bit_identical_to_paper_init(self):
        """`init_variant="mix3"` 必须与论文口径 `init_mix3` 逐位相同。

        这是「变体扫描不污染基线」的前提：桶顺序、余数落点、rng 消耗顺序
        三者任何一处变了，n_pop 不整除时就会漂移。
        """
        from rmoea_d.core.operators import init_by_variant, init_mix3
        for inst_name in ("Mk01", "Mk07", "Mk10"):
            inst = load_instance(inst_name, "data", 42)
            for n in (10, 30, 50, 99, 100, 101, 200):
                self.assertEqual(
                    init_mix3(inst, n, np.random.RandomState(7)),
                    init_by_variant(inst, n, np.random.RandomState(7), "mix3"),
                    "init_variant='mix3' 与 init_mix3 不等价 (%s n_pop=%d)"
                    % (inst_name, n))

    def test_random_variant_matches_enable_mix3_false_path(self):
        """`"random"` 变体必须等于论文 D1 的纯随机初始化。"""
        from rmoea_d.core.operators import init_by_variant, init_random
        inst = load_instance("Mk07", "data", 42)
        rng_a = np.random.RandomState(13)
        rng_b = np.random.RandomState(13)
        self.assertEqual([init_random(inst, rng_a) for _ in range(12)],
                         init_by_variant(inst, 12, rng_b, "random"))

    def test_unknown_variant_raises(self):
        from rmoea_d.core.operators import init_by_variant
        inst = load_instance("Mk01", "data", 42)
        with self.assertRaises(ValueError):
            init_by_variant(inst, 10, np.random.RandomState(1), "nope")

    def test_every_variant_produces_legal_individuals(self):
        """所有变体产出的 (os, ma) 都必须合法。"""
        from rmoea_d.core.operators import INIT_VARIANTS, init_by_variant
        for inst_name in ("Mk01", "Mk10"):
            inst = load_instance(inst_name, "data", 42)
            expect = self._expect_os(inst)
            for variant in sorted(INIT_VARIANTS):
                pop = init_by_variant(inst, 20, np.random.RandomState(3), variant)
                self.assertEqual(len(pop), 20, variant)
                for os_vec, ma_vec in pop:
                    self.assertEqual(sorted(os_vec), expect,
                                     "%s: OS 不是合法工序排列" % variant)
                    op_counter = [0] * inst["n_jobs"]
                    for idx, job_id in enumerate(os_vec):
                        oi = op_counter[job_id]
                        op_counter[job_id] += 1
                        self.assertIn(ma_vec[idx],
                                      inst["valid_machines"][job_id][oi],
                                      "%s: 机器对工序非法" % variant)

    @staticmethod
    def _prefix_mean_t2(os_vec, instance, frac=0.2):
        """序列前 frac 段里各工序最小 t2 的均值。"""
        k = max(1, int(len(os_vec) * frac))
        op_counter = [0] * instance["n_jobs"]
        total = 0
        for job_id in os_vec[:k]:
            oi = op_counter[job_id]
            op_counter[job_id] += 1
            total += instance["min_t2"][job_id][oi]
        return total / k

    def test_spt_variant_really_puts_short_operations_first(self):
        """派工式变体必须真的改变了 OS —— 否则它只是 MIX3 的无用复制。"""
        from rmoea_d.core.operators import init_os_spt, init_random
        inst = load_instance("Mk10", "data", 42)
        rng = np.random.RandomState(5)
        spt = float(np.mean([self._prefix_mean_t2(init_os_spt(inst, rng)[0], inst)
                             for _ in range(30)]))
        rnd = float(np.mean([self._prefix_mean_t2(init_random(inst, rng)[0], inst)
                             for _ in range(30)]))
        self.assertLess(spt, rnd,
                        "SPT 变体没有把短工序排到序列前面 (%.3f vs %.3f)"
                        % (spt, rnd))

    def test_mwr_picks_largest_remaining_job_first(self):
        """MWR 的第一步必须选「剩余工作量最大」的工件（定义性检查）。

        原打算用「序列前半段已完成工件数」做统计对比，但 Mk10 上两种初始化
        都是 0（20 工件 × 12 工序，132 个位置里让任一工件完整出现 12 次太苛刻），
        该指标没有区分度。改用 explore=0 关掉随机探索后的**确定性规则检查**。
        """
        from rmoea_d.core.operators import init_os_mwr
        inst = load_instance("Mk10", "data", 42)
        expect = int(np.argmax([sum(row) for row in inst["min_t2"]]))
        for seed in range(20):
            os_vec, _ = init_os_mwr(inst, np.random.RandomState(seed),
                                    explore=0.0)
            self.assertEqual(os_vec[0], expect,
                             "MWR 首步没选剩余工作量最大的工件 (seed=%d)" % seed)

    def test_spt_picks_shortest_operation_first(self):
        """SPT 的第一步必须选 t2 最短的那道工序（定义性检查）。"""
        from rmoea_d.core.operators import init_os_spt
        inst = load_instance("Mk10", "data", 42)
        expect = int(np.argmin([row[0] for row in inst["min_t2"]]))
        for seed in range(20):
            os_vec, _ = init_os_spt(inst, np.random.RandomState(seed),
                                    explore=0.0)
            self.assertEqual(os_vec[0], expect,
                             "SPT 首步没选最短工序 (seed=%d)" % seed)

    def test_explore_switch_reintroduces_diversity(self):
        """explore=0 时同一个 run 内的派工个体完全相同 → 必须有 explore 兜底。

        这也是为什么两个派工式变体默认 explore=0.25：否则 1/3 的初始种群
        会被同一个确定性解占满，多样性直接崩掉。
        """
        from rmoea_d.core.operators import init_os_spt
        inst = load_instance("Mk10", "data", 42)
        rng = np.random.RandomState(2)
        det = {tuple(init_os_spt(inst, np.random.RandomState(2),
                                 explore=0.0)[0]) for _ in range(10)}
        self.assertEqual(len(det), 1, "explore=0 时本该完全确定")
        sto = {tuple(init_os_spt(inst, rng, explore=0.25)[0]) for _ in range(10)}
        self.assertEqual(len(sto), 10, "explore>0 时个体应互不相同")

    def test_algorithm_routes_variant_through_init_by_variant(self):
        """`init_variant` 参数必须真的被算法接上（不是只存了个字段）。"""
        inst = load_instance("Mk01", "data", 42)

        def build(**kw):
            s = RMOEAD("Mk01", n_pop=18, max_gen=1, seed=7, fixed_T=10,
                       enable_rvns=False, **kw)
            s.instance = inst          # `_init_population` 直接用 self.instance
            return s

        pa, _ = build(init_variant="mix3")._init_population()
        pb, _ = build(init_variant="mix3_spt")._init_population()
        self.assertNotEqual(pa, pb, "换 init_variant 后初始种群居然没变")

        # enable_mix3=False 时 init_variant 应被忽略，一律走纯随机（论文 D1）
        from rmoea_d.core.operators import init_random
        pc, _ = build(enable_mix3=False,
                      init_variant="mix3_spt")._init_population()
        rng = np.random.RandomState(7)
        self.assertEqual(pc, [init_random(inst, rng) for _ in range(18)])


class TestInitVariantAnalysisGrouping(unittest.TestCase):
    """分析台缺陷回归锁：HV 归一化盒必须**逐实例独立**。

    两条在 2026-09-17 留出集分析里实测踩到的缺陷：

    * **缺陷 A（会读出假结果）**：早前 `load_hv` 把所有 lab 摊平成一个
      `{(label, seed): hv}` 字典。Mk07 与 Mk10 的工序量级差 ~4 倍，
      合并后同一个盒把 Mk07 的前沿压进左下角 -> HV 冲到 **1.0**
      （盒体积上限 1.02²=1.0404），而只在 Mk10 出现的臂只剩 **0.16**。
      输出会显示"Mk07 上所有初始化变体都近乎完美"——纯粹的量纲假象。
    * **缺陷 B（静默失效）**：实例标签由文件名得到 `_mk07`，而一致性表的
      查表键写的是 `mk07`（无下划线）-> 方向一致性表**永远为空**，
      却打印"判定：两实例同号且都不显著"，等于把"没数据"报成"结论"。
    """

    @staticmethod
    def _mk_rows(instance, x0, y0, label):
        """造一组该实例量级的前沿：x 为 makespan，y 为总机器负载。"""
        return [dict(instance=instance, label=label, seed=s,
                     final_pf=[[float(x0 + 2 * s), float(y0 + 3 * s)],
                                [float(x0 + 8 + 2 * s), float(y0 - 6 + 3 * s)]])
                for s in range(5)]

    def _import(self):
        import importlib
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                        '..', 'scripts'))
        return importlib.import_module("init_variant_analysis")

    def test_instance_name_is_inferred_from_filename(self):
        """缺陷 B：`_mk07_init.json` 必须推出 `Mk07`，不是 `_mk07`/`mk07`。"""
        m = self._import()
        self.assertEqual(m._infer_instance("logs/_mk07_init.json", []), "Mk07")
        self.assertEqual(m._infer_instance("logs/_mk10_lab.json", []), "Mk10")
        self.assertEqual(m._infer_instance("logs/Mk3_x.json", []), "Mk03")
        self.assertIsNone(m._infer_instance("logs/_init.json", []))

    def test_rows_are_grouped_by_instance_not_collapsed(self):
        """缺陷 A：同 (label, seed) 跨实例**不得**互相覆盖。"""
        import json
        import tempfile
        m = self._import()
        rows = (self._mk_rows("Mk07", 150, 650, "I_mix3")
                + self._mk_rows("Mk10", 300, 1900, "I_mix3"))
        with tempfile.NamedTemporaryFile("w", suffix="_mk07.json", delete=False,
                                         encoding="utf-8") as fh:
            json.dump(rows, fh)
            path = fh.name
        try:
            per, flat = m.load_by_instance([path])
        finally:
            os.remove(path)
        self.assertEqual(list(per.keys()), ["Mk07", "Mk10"],
                         "两个实例必须被分开，不能摊平成一张表")
        self.assertEqual(len(flat), 10)
        # 旧实现会把 10 行压成 5 个 (label, seed) 键 —— 现在按实例各 5 条
        for k in per:
            self.assertEqual(len(per[k]), 5)

    def test_merged_box_produces_dimension_artifact(self):
        """合并盒会让**同一份数据**的 HV 从 0.0038 摆到 1.0402（270 倍）。

        实测（合成数据，Mk07 量级 150/650，Mk10 量级 300/1900，ref=(1.02,1.02)）：

            Mk07 前沿： 合并盒 1.0402   |  本实例盒 0.8737
            Mk10 前沿： 合并盒 0.0038   |  本实例盒 0.8737

        盒体积上限是 1.02²=1.0404，所以合并盒把 Mk07 抬到**顶格**、
        把 Mk10 踩到**近乎零**。这就是为什么 `load_by_instance` 必须分组。
        """
        m = self._import()
        from rmoea_d.utils.metrics import compute_hv, estimate_hv_bounds

        r7 = self._mk_rows("Mk07", 150, 650, "I_mix3")
        r10 = self._mk_rows("Mk10", 300, 1900, "I_mix3")

        def fronts(rows):
            return [np.asarray(r["final_pf"], float) for r in rows]

        ref = (1.02, 1.02)
        # 合并盒（错误做法）
        lo_m, hi_m = estimate_hv_bounds(fronts(r7) + fronts(r10))
        hv7_merged = compute_hv(fronts(r7)[0], ref_point=ref,
                                norm_bounds=(lo_m, hi_m))
        hv10_merged = compute_hv(fronts(r10)[0], ref_point=ref,
                                 norm_bounds=(lo_m, hi_m))
        # 逐实例独立盒（正确做法）
        lo7, hi7 = estimate_hv_bounds(fronts(r7))
        lo10, hi10 = estimate_hv_bounds(fronts(r10))
        hv7_own = compute_hv(fronts(r7)[0], ref_point=ref,
                             norm_bounds=(lo7, hi7))
        hv10_own = compute_hv(fronts(r10)[0], ref_point=ref,
                              norm_bounds=(lo10, hi10))

        # 同一构型的两条前沿，在各自盒里得分必须相同（盒是唯一变量）
        self.assertAlmostEqual(hv7_own, hv10_own, places=6)
        # 合并后方向相反且幅度离谱 —— 纯量纲假象
        self.assertGreater(hv7_merged, 1.03,
                           "合并盒本该把 Mk07 抬到接近盒体积上限 1.0404")
        self.assertLess(hv10_merged, 0.05,
                        "合并盒本该把 Mk10 踩到近乎零")
        self.assertGreater(hv7_merged / max(hv10_merged, 1e-12), 10.0,
                           "两种口径必须差出量级，才说明分组不是可选项")

        # 独立盒的下界必须落在本实例量级内（证据：盒没被别的实例污染）
        self.assertAlmostEqual(lo7[0], 150.0, delta=2.0)
        self.assertAlmostEqual(lo10[1], 1894.0, delta=2.0)

    def test_out_of_context_arms_do_not_touch_the_box(self):
        """缺陷 C：**不参与比较的臂不得进归一化盒**（2026-09-18 实测）。

        原型：`_mk10_lab.json` 里顺带存着 34 个 Q-PAS / T 臂，其中
        `QPAS2_opt_hv`(seed 43) 的前沿把 Mk10 的 y 上界从 2234.0 撑到 2297.25，
        于是同一份变体数据被读出**另一套** rel% / dz：
        `I_rand` 的初始化杠杆 −25.69% 被读成 −22.31%。

        这里用一个"极端无关臂"（前沿整体 +900）复现同一机制：它必须被排除，
        且盒与全部 HV 值**逐元素不变**。
        """
        m = self._import()
        ctx = (self._mk_rows("Mk10", 300, 1900, "I_mix3")
               + self._mk_rows("Mk10", 300, 1900, "I_mwr"))
        outsider = self._mk_rows("Mk10", 300, 1900, "QPAS2_opt_hv")
        for r in outsider:                       # 模拟离群臂把盒撑大
            r["final_pf"] = [[p[0], p[1] + 900.0] for p in r["final_pf"]]

        hv_ctx, box_ctx = m.hv_table(ctx)
        kept, dropped = m.split_context(ctx + outsider, m.CONTEXT_ARMS)
        hv_kept, box_kept = m.hv_table(kept)

        self.assertEqual(list(dropped), ["QPAS2_opt_hv"],
                         "无关臂必须被识别出来（而不是静默丢弃）")
        self.assertEqual(dropped["QPAS2_opt_hv"], 5)
        self.assertTrue(np.array_equal(box_ctx[0], box_kept[0]))
        self.assertTrue(np.array_equal(box_ctx[1], box_kept[1]),
                        "无关臂不得影响归一化盒")
        for lab in hv_ctx:
            for s in hv_ctx[lab]:
                self.assertEqual(hv_ctx[lab][s], hv_kept[lab][s],
                                 "同一份数据在两种臂集下必须给出同一个 HV")

        # 反证：真把无关臂并进盒，值就变了 —— 说明这个测试有区分度
        _, box_bad = m.hv_table(ctx + outsider)
        self.assertGreater(box_bad[1][1], box_ctx[1][1],
                           "并进盒本该把 y 上界抬高（这就是被污染的机制）")

    def test_context_arm_set_is_explicit(self):
        """参与比较的臂集必须是显式常量，且不含常见的高污染源。"""
        m = self._import()
        self.assertTrue(set(m.VARIANTS).issubset(set(m.CONTEXT_ARMS)),
                        "全部变体臂都必须在参与比较的臂集里")
        self.assertIn("RVNSonly", m.CONTEXT_ARMS,
                      "Q1 的对照臂 RVNSonly 也要在集内（它与 I_mix3 逐位相同，不改变盒）")
        for bad in ("QPAS2_opt_hv", "QPAS2_hv_wide", "RVNSonly_Brand", "T100", "Full"):
            self.assertNotIn(bad, m.CONTEXT_ARMS,
                             "%s 不参与变体比较，不得进盒" % bad)

        # split_context 默认口径 = CONTEXT_ARMS，无关臂要被报出来
        rows = (self._mk_rows("Mk10", 300, 1900, "I_mwr")
                + self._mk_rows("Mk10", 300, 1900, "T100"))
        kept, dropped = m.split_context(rows)
        self.assertEqual({r["label"] for r in kept}, {"I_mwr"})
        self.assertEqual(dict(dropped), {"T100": 5})

    @staticmethod
    def _run_main(m, labs):
        """按 CLI 口径跑一次 `main()`，返回它打印的全部文本。"""
        import contextlib
        import io as _io
        m.L.clear()
        buf = _io.StringIO()
        old = sys.argv
        sys.argv = ["init_variant_analysis.py", "--labs", labs]
        try:
            with contextlib.redirect_stdout(buf):
                m.main()
        finally:
            sys.argv = old
        return buf.getvalue()

    def test_cli_actually_excludes_out_of_context_arms(self):
        """端到端：**`main()` 必须真的用上排除逻辑**，而不只是 helper 正确。

        只验 helper 是不够的 —— 缺陷 18 的形态正是"helper 写了一版、
        调用方没接上"。所以这里直接跑 CLI，断言：
        (a) 混入离群无关臂后，归一化盒**一字不变**；
        (b) 该臂被显式报出来（"已排除的无关臂"），不是静默丢弃。
        """
        import json
        import re
        import tempfile
        m = self._import()
        rows = (self._mk_rows("Mk10", 300, 1900, "I_mix3")
                + self._mk_rows("Mk10", 300, 1900, "I_mwr"))
        # 让 I_mwr 与 I_mix3 有可测差异，否则 rel% 恒为 0（断言会变成平凡真）。
        # 实测：臂集放开时同一条 rel% 会从 +57.80% 被读成 +16.61% ——
        # 与真实案例（−25.69% 读成 −22.31%）是同一个机制。
        for r in rows:
            if r["label"] == "I_mwr":
                r["final_pf"] = [[p[0] - 2.0, p[1] - 3.0] for p in r["final_pf"]]
        bad = self._mk_rows("Mk10", 300, 1900, "QPAS2_opt_hv")
        for r in bad:
            r["final_pf"] = [[p[0], p[1] + 900.0] for p in r["final_pf"]]

        paths = []
        for payload in (rows, rows + bad):
            with tempfile.NamedTemporaryFile("w", suffix="_mk10_x.json",
                                             delete=False, encoding="utf-8") as fh:
                json.dump(payload, fh)
                paths.append(fh.name)
        try:
            out_clean = self._run_main(m, paths[0])
            out_dirty = self._run_main(m, paths[1])
        finally:
            for p in paths:
                os.remove(p)

        pat = re.compile(r"归一化盒\(本实例独立\) (.*)")
        box_clean = pat.search(out_clean)
        box_dirty = pat.search(out_dirty)
        self.assertIsNotNone(box_clean)
        self.assertIsNotNone(box_dirty)
        self.assertEqual(box_clean.group(1), box_dirty.group(1),
                         "无关臂不得改变报告里的归一化盒")
        self.assertIn("已排除的无关臂", out_dirty,
                      "被排除的臂必须显式报出来")
        self.assertIn("QPAS2_opt_hv", out_dirty)
        self.assertNotIn("已排除的无关臂", out_clean,
                         "没有无关臂时不该出现该行")
        # 报告的 rel% 必须也一致（盒一致 -> HV 一致 -> 效应量一致）
        rel_pat = re.compile(r"^\s*I_mwr\s+.*?([+-]\d+\.\d+)%", re.M)
        self.assertEqual(
            [g for g in rel_pat.findall(out_clean)],
            [g for g in rel_pat.findall(out_dirty)],
            "无关臂不得改变 I_mwr 的 rel%（缺陷 18 的原始症状）")


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
_DATA = os.path.join(_ROOT, "data")


class TestHVCaliberDivergence(unittest.TestCase):
    """缺陷 22 回归锁：两套 HV 口径的性质差异必须可复现。

    本轮最重要的口径纠正。若有人"顺手"把 `instance_hv_bounds` 换成
    `estimate_hv_bounds`，本研究所有效应量会突然放大约一个数量级，
    所以把两条性质钉死：
      (a) `estimate_hv_bounds` 的边界**严格等于输入前沿的极值** → 随臂集变化；
      (b) `instance_hv_bounds` 只依赖实例数据 → 与臂集无关。
    """

    def test_estimate_hv_bounds_equals_front_extremes(self):
        a = np.array([[10.0, 100.0], [20.0, 90.0]])
        b = np.array([[12.0, 95.0]])
        lo, hi = estimate_hv_bounds([a, b])
        self.assertTrue(np.allclose(lo, np.vstack([a, b]).min(axis=0)))
        self.assertTrue(np.allclose(hi, np.vstack([a, b]).max(axis=0)))

    def test_box_changes_when_arm_set_changes(self):
        """加入一个离群臂 → 盒必须变。这就是缺陷 14/18/22 的机制。"""
        a = np.array([[10.0, 100.0], [20.0, 90.0]])
        b = np.array([[12.0, 95.0]])
        lo1, hi1 = estimate_hv_bounds([a, b])
        lo2, hi2 = estimate_hv_bounds([a, b, np.array([[5.0, 200.0]])])
        self.assertFalse(np.allclose(lo1, lo2),
                         "盒没随臂集变化，说明这条锁失去区分度")
        self.assertFalse(np.allclose(hi1, hi2))

    def test_instance_hv_bounds_is_arm_set_invariant(self):
        """同一实例、两次独立加载 → 边界逐位相同（与任何臂集无关）。"""
        i1 = load_instance("Mk01", data_dir=_DATA)
        i2 = load_instance("Mk01", data_dir=_DATA)
        lo1, hi1 = instance_hv_bounds(i1)
        lo2, hi2 = instance_hv_bounds(i2)
        self.assertTrue(np.array_equal(lo1, lo2))
        self.assertTrue(np.array_equal(hi1, hi2))
        self.assertTrue(np.all(lo1 < hi1))

    def test_box_caliber_inflates_relative_gain(self):
        """同一份绝对改进：盒口径的 rel% 必须大于宽松固定边界下的 rel%。"""
        base = np.array([[300.0, 1900.0], [320.0, 1850.0], [340.0, 1810.0]])
        better = np.array([[295.0, 1900.0], [315.0, 1850.0], [335.0, 1810.0]])
        ref = (1.02, 1.02)
        lo_b, hi_b = estimate_hv_bounds([base, better])
        h1 = compute_hv(base, ref_point=ref, norm_bounds=(lo_b, hi_b))
        h2 = compute_hv(better, ref_point=ref, norm_bounds=(lo_b, hi_b))
        rel_box = (h2 - h1) / h1
        # 宽松固定边界 ≈ instance_hv_bounds 用的是"理论下界 … 理论上界"
        lo_w = np.array([0.0, 0.0])
        hi_w = np.array([1000.0, 4000.0])
        g1 = compute_hv(base, ref_point=ref, norm_bounds=(lo_w, hi_w))
        g2 = compute_hv(better, ref_point=ref, norm_bounds=(lo_w, hi_w))
        rel_wide = (g2 - g1) / g1
        self.assertGreater(rel_box, rel_wide,
                           "盒口径应放大相对增幅（缺陷 22 的核心症状）")


class TestAnytimeRunWallclock(unittest.TestCase):
    """缺陷 20 回归锁：长程实验台必须落盘**累计墙钟**。

    `RMOEAD.history[i]["time"]` 一直有，但从未被导出，导致「等墙钟」比较做不了。
    """

    def test_records_monotone_wallclock(self):
        import anytime_run
        job = ("Mk01", "D1", 42, 20, 6, _DATA)
        r = anytime_run._run_one(job)
        self.assertEqual(len(r["hist_time"]), r["max_gen"])
        self.assertEqual(len(r["hist_hv"]), r["max_gen"])
        t = np.asarray(r["hist_time"], dtype=float)
        self.assertTrue(np.all(np.diff(t) > 0),
                        "累计墙钟必须严格递增（否则插值无意义）")
        # 累计到末代的时间 + 初始化开销，不得超过总墙钟
        self.assertLessEqual(t[-1] + r["init_time"], r["total_time"] + 1e-6)

    def test_run_is_deterministic(self):
        import anytime_run
        job = ("Mk01", "D1", 42, 20, 6, _DATA)
        a = anytime_run._run_one(job)
        b = anytime_run._run_one(job)
        self.assertEqual(a["hist_hv"], b["hist_hv"])
        self.assertEqual(a["final_hv"], b["final_hv"])


class TestPerInstanceWallclockTmax(unittest.TestCase):
    """缺陷 23 回归锁：等墙钟公共上限必须**逐实例**求，不得取全局最小。

    取全局最小会让大实例（Mk10，数百秒）只在自身预算的前十几个百分点处
    被比较，把「等墙钟」降级成「等一个很小的墙钟」—— 与 §1.1「逐实例独立」
    原则冲突，是缺陷 22 的同一家族（跨实例共用一个标量边界）。
    """

    @staticmethod
    def _rows():
        h = [0.1, 0.2, 0.3]
        return [
            {"instance": "Mk07", "label": "D1", "seed": 42,
             "hist_hv": h, "hist_time": [5.0, 10.0, 15.0], "init_time": 1.0},
            {"instance": "Mk10", "label": "D1", "seed": 42,
             "hist_hv": h, "hist_time": [40.0, 90.0, 140.0], "init_time": 3.0},
            {"instance": "Mk10", "label": "D5", "seed": 42,
             "hist_hv": h, "hist_time": [70.0, 160.0, 250.0], "init_time": 3.0},
        ]

    def test_tmax_is_per_instance(self):
        import response_surface as rs
        tmax = rs.per_instance_tmax(rs.index_curves(self._rows()))
        self.assertAlmostEqual(tmax["Mk07"], 16.0)    # 15 + init 1
        self.assertAlmostEqual(tmax["Mk10"], 143.0)   # 取 D1（140+3），不是 D5（253）
        self.assertNotAlmostEqual(
            tmax["Mk07"], tmax["Mk10"],
            msg="退化成全局最小 → 两实例拿到同一个上限，这条锁失去区分度")

    def test_tmax_empty_index(self):
        import response_surface as rs
        self.assertEqual(rs.per_instance_tmax({}), {})

    def test_load_many_prefers_longer_trajectory(self):
        """多份 lab 合并时必须取**更长**的轨迹，且结果与文件顺序无关。

        用途：把 G=2000 与 G=4000 两次长跑拼起来算"多给一倍算力会怎样"。
        若实现成"后写覆盖"，换个命令行顺序就会出两套数字。
        """
        import json
        import tempfile
        import response_surface as rs
        short = [{"instance": "Mk07", "label": "D1", "seed": 42,
                  "hist_hv": [0.1], "hist_time": [1.0]}]
        longer = [{"instance": "Mk07", "label": "D1", "seed": 42,
                   "hist_hv": [0.1, 0.2, 0.3], "hist_time": [1.0, 2.0, 3.0]}]
        with tempfile.TemporaryDirectory() as td:
            a = os.path.join(td, "a.json")
            b = os.path.join(td, "b.json")
            with open(a, "w", encoding="utf-8") as fh:
                json.dump(short, fh)
            with open(b, "w", encoding="utf-8") as fh:
                json.dump(longer, fh)
            for spec in (a + "," + b, b + "," + a):
                got = rs.load_many(spec)
                self.assertEqual(len(got), 1)
                self.assertEqual(len(got[0]["hist_hv"]), 3,
                                 "必须取更长的那条，且与顺序无关")

    def test_main_runs_end_to_end(self):
        """端到端：确认 main() 真把 helper 接上了（缺陷 18 的教训——
        helper 写了但调用方没接，是历史上最容易漏的一类）。"""
        import json
        import tempfile
        import contextlib
        import io
        import unittest.mock
        import response_surface as rs
        with tempfile.TemporaryDirectory() as td:
            rows_path = os.path.join(td, "mini.json")
            out_path = os.path.join(td, "mini.surface.json")
            with open(rows_path, "w", encoding="utf-8") as fh:
                json.dump(self._rows(), fh)
            argv = ["response_surface.py", "--labs", rows_path,
                    "--pairs", "D5=D1", "--out", out_path]
            with unittest.mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(io.StringIO()) as buf:
                    rc = rs.main()
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, encoding="utf-8") as fh:
                got = json.load(fh)
            rec = got["wall"]["D5|D1"]["Mk10"]
            self.assertTrue(all(r["tmax"] == 143.0 for r in rec),
                            "写出的 tmax 必须是逐实例值，不是全局最小")
            self.assertIn("Mk07=16.0s", buf.getvalue())


class TestFEMultiplier(unittest.TestCase):
    """等求值次数（FE）口径的承重假设锁。

    「等算力」结论全部建立在"RVNS 臂每代求值 2×"这一条上。若有人改掉
    `rvns_ls_trials` 默认值、或让 `plan_generation` 在 fixed 模式下不再发满档，
    §3.3 的结论会**静默失效** —— 所以这条必须单独钉死。
    """

    def test_fe_multiplier_matches_arm_definition(self):
        import response_surface as rs
        self.assertEqual(rs.fe_multiplier("D1"), 1.0, "纯 MOEA/D 无局部搜索")
        self.assertEqual(rs.fe_multiplier("D2"), 1.0, "D2 只加初始化，不加求值")
        self.assertEqual(rs.fe_multiplier("D5"), 2.0)
        self.assertEqual(rs.fe_multiplier("RMOEAD"), 2.0)
        # 未登记的臂名保守按 1× 处理，不得抛异常
        self.assertEqual(rs.fe_multiplier("__no_such_arm__"), 1.0)

    def test_fixed_budget_gives_every_solution_full_trials(self):
        """端到端：`budget_mode="fixed"` 时每个解都拿到满档 ls_trials。"""
        from rmoea_d.core.rvns import RVNS
        r = RVNS(n_operators=5, ls_trials=1, budget_mode="fixed")
        plan = r.plan_generation(20, np.random.RandomState(0))
        self.assertTrue(np.array_equal(np.asarray(plan), np.ones(20, dtype=int)))

    def test_fe_section_reports_two_times_budget(self):
        """等 FE 一节必须把 RVNS 臂的公共上限算成 2× 非 RVNS 臂。"""
        import json
        import tempfile
        import contextlib
        import io
        import unittest.mock
        import response_surface as rs
        rows = [
            {"instance": "Mk07", "label": "D2", "seed": 42, "n_pop": 100,
             "hist_hv": [0.1, 0.2, 0.3, 0.4], "hist_time": [1.0, 2.0, 3.0, 4.0],
             "init_time": 0.0},
            {"instance": "Mk07", "label": "D5", "seed": 42, "n_pop": 100,
             "hist_hv": [0.1, 0.2, 0.3, 0.4], "hist_time": [2.0, 4.0, 6.0, 8.0],
             "init_time": 0.0},
        ]
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "m.json")
            o = os.path.join(td, "m.out.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(rows, fh)
            argv = ["response_surface.py", "--labs", p, "--pairs", "D5=D2",
                    "--segments", "2", "--out", o]
            with unittest.mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(io.StringIO()) as buf:
                    self.assertEqual(rs.main(), 0)
            txt = buf.getvalue()
            self.assertIn("D2=100", txt)
            self.assertIn("D5=200", txt)
            with open(o, encoding="utf-8") as fh:
                got = json.load(fh)
            self.assertIn("D5|D2", got["fe"])
            rec = got["fe"]["D5|D2"]["Mk07"]
            self.assertEqual(len(rec), 5, "网格点数应等于 --wall_fracs 的项数")
            self.assertAlmostEqual(rec[-1]["FE"], 400.0)   # min(4*100, 4*200) = 400


class TestSurfaceMarkdown(unittest.TestCase):
    """`scripts/surface_markdown.py` 的回归锁。

    这个脚本存在的理由是"报告里的数字不许手抄"（fig6 曾硬编码 8.60/28.71，
    报告改了图没改）。它一旦算错，错的会**直接被抄进报告**，所以要把三类
    最容易悄悄出错的映射钉死：

    1. 代数 → 数组下标是 `hist_hv[g-1]`（off-by-one，差一位 = 全表系统性偏移）；
    2. 轨迹比网格短的 run 必须**被排除**，不能拿末代冒充第 g 代；
    3. 表 E 必须用 `final_hv`，**不是** `hist_hv[-1]` —— 这两者在启用 elite
       archive 时本来就不同（末代 population 前沿 vs 末代 archive）。
    """

    @staticmethod
    def _rows():
        return [
            {"instance": "Mk07", "label": "D1", "seed": 42,
             "hist_hv": [0.1, 0.2, 0.3], "hist_time": [1.0, 2.0, 3.0],
             "final_hv": 0.99},
            {"instance": "Mk07", "label": "D1", "seed": 43,
             "hist_hv": [0.5, 0.6, 0.7], "hist_time": [1.0, 2.0, 3.0],
             "final_hv": 0.99},
            {"instance": "Mk07", "label": "D2", "seed": 42,
             "hist_hv": [0.2, 0.2, 0.2], "hist_time": [1.0, 2.0, 3.0],
             "final_hv": 0.90},
        ]

    def test_abs_hv_maps_generation_g_to_index_g_minus_1(self):
        import response_surface as rs
        import surface_markdown as sm
        idx = rs.index_curves(self._rows())
        lines = sm.abs_hv_table(idx, ["Mk07"], ["D1"], [1, 2, 3])
        row = [l for l in lines if l.startswith("| Mk07")][0]
        cells = [c.strip() for c in row.split("|")[3:6]]
        # 逐 seed 取该代值再平均：(0.1+0.5)/2, (0.2+0.6)/2, (0.3+0.7)/2
        self.assertEqual(cells, ["0.300000", "0.400000", "0.500000"],
                         "代数 g 必须读 hist_hv[g-1]；差一位会让整张表系统性偏移")

    def test_abs_hv_drops_runs_shorter_than_the_grid(self):
        """轨迹不足的 run 必须被剔除，不得用末代冒充第 g 代。"""
        import response_surface as rs
        import surface_markdown as sm
        rows = [
            {"instance": "Mk07", "label": "D1", "seed": 42,
             "hist_hv": [0.1, 0.2, 0.3], "hist_time": [1.0, 2.0, 3.0]},
            {"instance": "Mk07", "label": "D1", "seed": 43,
             "hist_hv": [0.9, 0.9], "hist_time": [1.0, 2.0]},
        ]
        idx = rs.index_curves(rows)
        lines = sm.abs_hv_table(idx, ["Mk07"], ["D1"], [3])
        row = [l for l in lines if l.startswith("| Mk07")][0]
        cell = row.split("|")[3].strip()
        self.assertEqual(cell, "0.300000",
                         "只有 seed42 有第 3 代；若把 seed43 的末代当第 3 代会得到 0.6")
        self.assertNotIn("0.600000", row)
        self.assertNotIn("0.550000", row)

    def test_pair_table_uses_seed_intersection(self):
        """配对只能在两臂都有的 seed 上做；交集大小决定 n。"""
        import response_surface as rs
        import surface_markdown as sm
        rows = self._rows() + [
            {"instance": "Mk07", "label": "D2", "seed": 43,
             "hist_hv": [0.8, 0.8, 0.8], "hist_time": [1.0, 2.0, 3.0],
             "final_hv": 0.90},
            {"instance": "Mk07", "label": "D2", "seed": 44,
             "hist_hv": [0.8, 0.8, 0.8], "hist_time": [1.0, 2.0, 3.0],
             "final_hv": 0.90},
        ]
        idx = rs.index_curves(rows)
        lines = sm.pair_table_at_gens(idx, ["Mk07"], [("D2", "D1")], [1])
        row = [l for l in lines if "D2 − D1" in l][0]
        # D1 有 {42,43}，D2 有 {42,43,44} → 交集 {42,43}。若错用并集会写 3/3。
        self.assertIn("2/2", row, "必须在 seed 交集上配对：D1∩D2 = {42,43}")
        self.assertNotIn("3/3", row, "并集(3) 会凭空多出 seed44 这个 D1 根本没有的配对")

    def test_final_table_uses_final_hv_not_hist_tail(self):
        """表 E 必须用 final_hv：本例两臂末代相同(0.3/0.2)，只有 final_hv 分得开。"""
        import response_surface as rs
        import surface_markdown as sm
        rows = self._rows()
        idx = rs.index_curves(rows)
        lines = sm.final_table(idx, rows, ["Mk07"], [("D1", "D2")])
        row = [l for l in lines if "D1 − D2" in l][0]
        # (0.99+0.99)/2 - 0.90 = 0.09；若误用 hist_hv[-1] 会得到 0.25-0.2 = 0.05
        self.assertIn("+0.090000", row,
                      "final_hv 与 hist_hv[-1] 在启用 elite 时本就不同，不得混用")
        self.assertNotIn("+0.050000", row)

    def test_fe_table_renders_from_surface_json(self):
        """表 D 只做渲染，不重算统计量（统计量唯一真源是 response_surface）。"""
        import surface_markdown as sm
        surface = {"fe": {"D5|D2": {"Mk07": [
            {"FE": 100000.0, "fe_frac": 0.5, "n": 30,
             "diff": 0.123456, "wins": 25, "p": 0.01, "dz": 1.0}]}}}
        lines = sm.fe_table(surface, ["Mk07"], [("D5", "D2")])
        row = [l for l in lines if "D5 − D2" in l][0]
        self.assertIn("+0.1235", row, "必须原样渲染 surface json 里的 diff")
        self.assertIn("Mk07 = 200000", "\n".join(lines), "脚注给出各实例公共 FE 上限")

    def test_main_runs_end_to_end_and_skips_sections_without_surface(self):
        import json
        import tempfile
        import contextlib
        import io
        import unittest.mock
        import surface_markdown as sm
        with tempfile.TemporaryDirectory() as td:
            lab = os.path.join(td, "m.json")
            out = os.path.join(td, "m.md")
            missing = os.path.join(td, "nope.surface.json")
            with open(lab, "w", encoding="utf-8") as fh:
                json.dump(self._rows(), fh)
            argv = ["surface_markdown.py", "--lab", lab, "--surface", missing,
                    "--gens", "2", "--pairs", "D2=D1", "--out", out]
            with unittest.mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(sm.main(), 0)
            with open(out, encoding="utf-8") as fh:
                txt = fh.read()
            for sec in ("### A.", "### B.", "### E."):
                self.assertIn(sec, txt)
            self.assertNotIn("### C.", txt, "无 surface json 时不应伪造饱和表")
            self.assertNotIn("### D.", txt)


class TestFigDecayHelper(unittest.TestCase):
    """07 图（MIX3 残差衰减）的取值 helper 锁。

    这张图是本轮唯一能把"起跑优势"与"持久机制"分开的证据，且它的指数会被
    直接抄进报告 §3.6 —— 所以它的取值口径必须和 `surface_markdown` 完全一致：
    代数 g 读 `hist_hv[g-1]`、配对用 seed 交集、分母是 D1 在该代的均值。
    """

    @staticmethod
    def _idx():
        import response_surface as rs
        rows = [
            {"instance": "Mk07", "label": "D1", "seed": 42,
             "hist_hv": [0.1, 0.2, 0.4], "hist_time": [1.0, 2.0, 3.0]},
            {"instance": "Mk07", "label": "D1", "seed": 43,
             "hist_hv": [0.2, 0.4, 0.6], "hist_time": [1.0, 2.0, 3.0]},
            {"instance": "Mk07", "label": "D2", "seed": 42,
             "hist_hv": [0.3, 0.5, 0.8], "hist_time": [1.0, 2.0, 3.0]},
        ]
        return rs.index_curves(rows)

    def test_resid_uses_g_minus_1_and_seed_intersection(self):
        """G=2：交集只有 seed42 → diff = 0.5−0.2 = 0.3，分母 = D1 在**交集内**的均值 0.2
        → rel = 0.3/0.2 = 150%。

        注意分母是"交集内"而不是"该实例全部 D1"：若用后者（(0.2+0.4)/2 = 0.3）
        会得到 100%，与 `surface_markdown.pair_table_at_gens` 的口径不一致 ——
        图与报告必须同源，这条锁就是钉这个。
        """
        import new_arch_plots as nap
        got = nap._resid(self._idx(), "Mk07", 2)
        self.assertAlmostEqual(got, 150.0, places=5)

    def test_resid_raises_no_crash_when_trajectory_too_short(self):
        """G 超过轨迹长度时不许崩、也不许拿末代顶替（返回 nan）。"""
        import new_arch_plots as nap
        got = nap._resid(self._idx(), "Mk07", 9)
        self.assertTrue(np.isnan(got) or got == 0.0,
                        "超出轨迹长度应给 nan/0，不得用末代冒充第 9 代")

    def test_fig_decay_skips_missing_pair_without_crashing(self):
        """缺 D1/D2 配对时要打印提示并返回空串，而不是抛异常。"""
        import json
        import tempfile
        import contextlib
        import io
        import new_arch_plots as nap
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "x.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump([{"instance": "Mk07", "label": "D1", "seed": 42,
                            "hist_hv": [0.1, 0.2], "hist_time": [1.0, 2.0]}], fh)
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                got = nap.fig_decay(p)
            self.assertEqual(got, "")
            self.assertIn("跳过", buf.getvalue())


class TestAIGPermutation(unittest.TestCase):
    """`aig_gating.permutation_max_rho`：max|rho| 的置换零假设。

    为什么必须有这一层：AIG 的结论形态是"在 8 个候选里挑 ρ 最大者"。
    n=10 时纯噪声下 max|ρ| 的 95% 分位就有 ~0.81，
    所以**单变量 p 值会把噪声报成发现**。这几条锁钉住：
    (a) 植入的单调关系必须被捞回来；
    (b) 纯噪声不能被判显著；
    (c) 置换必须保留候选之间的相关结构（重复列不改变 max|ρ|）；
    (d) 边界：`n_perm=0` 返回 None/nan 而不是崩。
    """

    @staticmethod
    def _m():
        import importlib
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                        '..', 'scripts'))
        return importlib.import_module("aig_gating")

    def test_planted_monotone_relation_is_detected(self):
        m = self._m()
        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 6))
        y = 3.0 * X[:, 3] + 0.01 * rng.normal(size=10)   # 第 3 列是真因
        res = m.permutation_max_rho(X, y, 2000, seed=1)
        self.assertGreater(res["obs_max_abs_rho"], 0.97)
        self.assertEqual(res["obs_argmax"], 3,
                         "必须指认出真正相关的那一列")
        self.assertLess(res["mc_p_fwer"], 0.01)

    def test_pure_noise_is_not_significant(self):
        m = self._m()
        rng = np.random.default_rng(12345)
        X = rng.normal(size=(10, 6))
        y = rng.normal(size=10)                          # 与 X 无关
        res = m.permutation_max_rho(X, y, 4000, seed=7)
        self.assertGreater(res["mc_p_fwer"], 0.05,
                           "纯噪声不得被判显著（否则这层校正就是摆设）")
        self.assertLess(res["null_p95"], 1.0)

    def test_duplicated_column_preserves_null_structure(self):
        """置换只打乱 y —— 所以完全相关的两列，其 max|ρ| 与单列**逐位相同**。

        这条钉住的是"保留候选间相关结构"这个设计点：
        若实现改成"独立重抽 X"，重复列会被当成两次独立检验，
        null 分布就会变窄（把多重比较的严苛度做没了）。
        """
        m = self._m()
        rng = np.random.default_rng(5)
        c0 = rng.normal(size=10)
        c1 = rng.normal(size=10)
        y = rng.normal(size=10)
        one = m.permutation_max_rho(c0[:, None], y, 500, seed=3)
        two = m.permutation_max_rho(np.column_stack([c0, c0]), y, 500, seed=3)
        self.assertAlmostEqual(one["obs_max_abs_rho"], two["obs_max_abs_rho"],
                               places=12)
        self.assertAlmostEqual(one["null_p95"], two["null_p95"], places=12)
        # 区分度：换成一列**无关**的新列，max|ρ| 的观测值必须变大或不变
        three = m.permutation_max_rho(np.column_stack([c0, c1]), y, 500, seed=3)
        self.assertGreaterEqual(three["obs_max_abs_rho"],
                                one["obs_max_abs_rho"] - 1e-12)

    def test_obs_matches_analytic_spearman(self):
        """k=1 时 max|ρ| 必须等于 `scipy.spearmanr` 的 |ρ|（不是 pearson）。"""
        from scipy import stats
        m = self._m()
        rng = np.random.default_rng(9)
        x = rng.normal(size=10)
        y = rng.normal(size=10)
        res = m.permutation_max_rho(x[:, None], y, 200, seed=2)
        self.assertAlmostEqual(res["obs_max_abs_rho"],
                               abs(stats.spearmanr(x, y).statistic), places=12)

    def test_zero_perm_returns_nan_without_crashing(self):
        m = self._m()
        X = np.arange(20, dtype=float).reshape(10, 2)
        y = np.arange(10, dtype=float)
        res = m.permutation_max_rho(X, y, 0)
        self.assertEqual(res["n_perm"], 0)
        self.assertIsNone(res["null_p95"])
        self.assertTrue(np.isnan(res["mc_p_fwer"]))
        self.assertGreater(res["obs_max_abs_rho"], 0.9)

    def test_degenerate_input_returns_skipped_instead_of_crashing(self):
        """缺陷 27 / 19：退化输入不得抛异常。

        `scripts/init_probe.py --instances Mk01 --seeds 1 --perm 0`（文档 §8 的
        最小自测形态）曾在这条路径上崩：
        `ValueError: attempt to get argmax of an empty sequence`
        —— n=1 时所有候选列恒定 → X 被剔成 0 列 → `np.nanargmax([])`。
        契约：返回 `skipped` 标记 + 全 None，而不是异常。
        """
        m = self._m()
        # (a) 0 列（全部候选被常量列过滤掉）
        empty = m.permutation_max_rho(np.zeros((3, 0)), np.array([1., 2., 3.]), 100)
        self.assertIn("skipped", empty)
        self.assertIsNone(empty["obs_argmax"])
        self.assertIsNone(empty["null_p95"])
        self.assertTrue(np.isnan(empty["mc_p_fwer"]))
        # (b) 样本数 < 3（相关系数无定义）
        few = m.permutation_max_rho(np.arange(4, dtype=float).reshape(2, 2),
                                    np.array([1., 2.]), 100)
        self.assertIn("skipped", few)
        self.assertIsNone(few["obs_argmax"])
        # (c) 正常输入必须**不受**这套退化分支影响
        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 3))
        y = 2.0 * X[:, 1] + 0.01 * rng.normal(size=10)
        ok = m.permutation_max_rho(X, y, 500, seed=11)
        self.assertNotIn("skipped", ok)
        self.assertEqual(ok["obs_argmax"], 1)

    def test_null_p95_grows_with_the_number_of_candidates(self):
        """零假设必须对**全部候选**取 max —— 这就是家族错误率校正的本体。

        这条是被一次真实变异测试**逼出来**的：把 `max_null` 写成 `|rho_null[0]|`
        （只看第一个候选）时，观测值、argmax、单点 p 全都不变，
        上面 5 条锁一条都抓不住 —— 但校正已经没了。
        判据：候选越多，`max|ρ|` 的零假设分布只能越宽。
        """
        m = self._m()
        rng = np.random.default_rng(31)
        X = rng.normal(size=(10, 6))
        y = rng.normal(size=10)
        one = m.permutation_max_rho(X[:, :1], y, 3000, seed=4)
        six = m.permutation_max_rho(X, y, 3000, seed=4)
        self.assertGreater(six["null_p95"] - one["null_p95"], 0.10,
                           "6 个候选的零假设 95% 分位必须明显高于 1 个候选；"
                           "若相等说明只对第一个候选取了 max，校正失效")
        self.assertGreaterEqual(six["null_mean"], one["null_mean"] - 1e-12)
        self.assertGreaterEqual(six["null_p99"], one["null_p99"] - 1e-12)


class TestAIGLeaveOneOut(unittest.TestCase):
    """`aig_gating.leave_one_out`：n=10 的相关性"是不是被一个点撑起来的"。"""

    @staticmethod
    def _m():
        import importlib
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                        '..', 'scripts'))
        return importlib.import_module("aig_gating")

    def test_perfect_relation_survives_every_fold(self):
        m = self._m()
        x = np.arange(10, dtype=float)
        y = 2.0 * x + 1.0
        res = m.leave_one_out(x[:, None], y, ["perfect"])
        d = res["perfect"]
        self.assertAlmostEqual(d["min_rho"], 1.0, places=12)
        self.assertAlmostEqual(d["max_rho"], 1.0, places=12)

    def test_single_outlier_can_destroy_the_relation(self):
        """一个离群点能让整体 ρ 掉到 0.5 以下；**去掉它 ρ 立刻回到 1.0**。

        这正是"该关联由单点决定"的判据 —— 报告里 `total_ops` 的留一区间
        [+0.756, +0.874] 就是靠这条排除"只有 Mk10 撑场"。
        注意：去掉离群点得到的是**最好**的那一折（max_rho），不是最差的那折。
        """
        m = self._m()
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)
        y[-1] = -100.0                                  # 单点反向离群
        res = m.leave_one_out(x[:, None], y, ["outlier"])
        d = res["outlier"]
        self.assertLess(d["rho_all"], 0.5)
        self.assertAlmostEqual(d["max_rho"], 1.0, places=12)
        self.assertEqual(d["worst_drop_idx"], 0,
                         "去掉首点后是一个错位排列，那才是 |ρ| 最小的那一折")

    def test_worst_index_is_consistent_with_min_abs_rho(self):
        from scipy import stats
        m = self._m()
        rng = np.random.default_rng(11)
        X = rng.normal(size=(10, 3))
        y = X[:, 1] + 0.3 * rng.normal(size=10)
        names = ["a", "b", "c"]
        res = m.leave_one_out(X, y, names)
        for j, nm in enumerate(names):
            d = res[nm]
            self.assertGreaterEqual(d["worst_drop_idx"], 0)
            self.assertLess(d["worst_drop_idx"], 10)
            msk = np.ones(10, bool)
            msk[d["worst_drop_idx"]] = False
            got = abs(stats.spearmanr(X[msk, j], y[msk]).statistic)
            self.assertAlmostEqual(d["min_abs_rho"], got, places=10)


class TestAIGTargets(unittest.TestCase):
    """目标量必须真的可切换，且读数只来自 `final_hv`（实例边界口径）。

    缺陷 26 的形态是"目标量错位"：拿 `vs I_rand` 去预测"该不该用 MWR"。
    这里钉住 (a) `base` 真的生效；(b) 缺 `base` 的实例被显式跳过；
    (c) 读数只认 `final_hv` —— 改 `final_pf` 不该动结果。
    """

    @staticmethod
    def _m():
        import importlib
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                        '..', 'scripts'))
        return importlib.import_module("aig_gating")

    @staticmethod
    def _rows(inst, base_hv, mwr_hv, with_mix3):
        rows = []
        for s in range(6):
            rows.append(dict(instance=inst, label="I_rand", seed=42 + s,
                             final_hv=0.80 + 0.001 * s,
                             final_pf=[[100.0 + s, 900.0 + s]]))
            rows.append(dict(instance=inst, label="I_mwr", seed=42 + s,
                             final_hv=mwr_hv + 0.001 * s,
                             final_pf=[[95.0 + s, 880.0 + s]]))
            if with_mix3:
                rows.append(dict(instance=inst, label="I_mix3", seed=42 + s,
                                 final_hv=base_hv + 0.001 * s,
                                 final_pf=[[97.0 + s, 890.0 + s]]))
        return rows

    def test_dhv_depends_on_the_chosen_base(self):
        m = self._m()
        per = {"Mk01": self._rows("Mk01", 0.81, 0.83, True),
               "Mk02": self._rows("Mk02", 0.90, 0.905, True)}
        a, _ = m.build_recs(per, "I_rand", "", verbose=False)
        b, _ = m.build_recs(per, "I_mix3", "", verbose=False)
        self.assertEqual([r["base"] for r in a], ["I_rand"] * 2)
        self.assertEqual([r["base"] for r in b], ["I_mix3"] * 2)
        for ra, rb in zip(a, b):
            self.assertNotAlmostEqual(ra["dhv_rel_pct"], rb["dhv_rel_pct"],
                                      places=6,
                                      msg="换 base 后 ΔHV 没变 -> base 根本没生效")
            self.assertGreater(ra["dhv_rel_pct"], rb["dhv_rel_pct"],
                               "I_mwr vs I_rand 必须大于 I_mwr vs I_mix3")

    def test_instances_lacking_the_base_are_skipped_explicitly(self):
        m = self._m()
        per = {"Mk07": self._rows("Mk07", 0.81, 0.83, True),
               "Mk08": self._rows("Mk08", 0.0, 0.83, False)}
        recs, skipped = m.build_recs(per, "I_mix3", "", verbose=False)
        self.assertEqual([r["instance"] for r in recs], ["Mk07"])
        self.assertEqual(skipped, ["Mk08"],
                         "缺 base 的实例必须被显式报出，不能静默出 nan")

    def test_reads_final_hv_not_the_box(self):
        """只改 `final_pf`（不动 `final_hv`）时，主口径读数必须一字不变。

        这条同时挡住"退回盒口径"的回归：盒由参与臂的前沿极值构造，
        所以**只放大其中一个臂**的前沿 → 盒会变。于是
        实例边界读数不动、盒口径读数必须动 —— 两边一起验才有区分度。
        （若把全部臂同比例放大则盒同步缩放，盒口径也不动 —— 那不是有效扰动。）
        """
        m = self._m()
        per_a = {"Mk10": self._rows("Mk10", 0.90, 0.93, True)}
        per_b = {"Mk10": self._rows("Mk10", 0.90, 0.93, True)}
        for r in per_b["Mk10"]:
            if r["label"] == "I_mwr":                   # 只动一个臂
                r["final_pf"] = [[p[0] * 1.5, p[1] * 1.5] for p in r["final_pf"]]
        ra, _ = m.build_recs(per_a, "I_mix3", "", verbose=False)
        rb, _ = m.build_recs(per_b, "I_mix3", "", verbose=False)
        self.assertAlmostEqual(ra[0]["dhv_rel_pct"], rb[0]["dhv_rel_pct"],
                               places=12)
        self.assertAlmostEqual(ra[0]["dhv"], rb[0]["dhv"], places=12)
        # 反证（区分度）：盒口径的量必须动，否则说明根本没算盒
        self.assertNotAlmostEqual(ra[0]["dhv_rel_box_pct"],
                                  rb[0]["dhv_rel_box_pct"], places=6)

    def test_analyze_skips_permutation_when_perm_zero(self):
        m = self._m()
        recs = [dict(instance="Mk%02d" % i, dhv_rel_pct=float(i),
                     total_ops=10 * i, n_jobs=i, n_machines=i,
                     flex_ratio=0.1 * i, pt_cv=0.01 * i, load_ratio=float(i),
                     ms_lever_pct=float(i), wl_lever_pct=float(i))
                for i in range(1, 7)]
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            corr, perm, loo = m.analyze(recs, 0, 42)
        self.assertIsNone(perm, "perm=0 时不应产出置换结果")
        self.assertIsNotNone(corr)
        self.assertIsNotNone(loo)


class TestGateGainCriterion(unittest.TestCase):
    """缺陷 29 回归锁：「有净增量」必须是**统计显著为正**，不是"浮点非零"。

    形态：`paper_export.hurink_gate` 原先用 `abs(ΔHV) > 1e-9` 判"有增益"。
    Mk 与 Hurink 上**都没有 ΔHV 精确为 0 的实例**（Mk 最小非零是 Mk07 −0.0090%），
    于是开发集 10/10 被判成"有增益"、门控判对率从 **1.00 掉到 0.30**；
    留出集上更会把全部实例判成有增益、门控表彻底失效。
    这个错误**不会报任何异常**，只会静默出一张错表 —— 所以必须钉住。
    """

    @staticmethod
    def _m():
        import importlib
        _d = os.path.join(os.path.dirname(__file__), '..', 'scripts')
        if _d not in sys.path:
            sys.path.insert(0, _d)
        return importlib.import_module("paper_export")

    def test_tiny_nonzero_effect_is_not_a_gain(self):
        """真实数值：Mk07 −0.0090%/p=1.000 与 Mk05 +0.0240%/p=1.000 都是非零，但都不算增益。"""
        m = self._m()
        self.assertFalse(m.is_gain(-0.0090, 1.000), "微小非零被误判成增益（旧 `>1e-9` 判据）")
        self.assertFalse(m.is_gain(0.0240, 1.000), "微小非零被误判成增益（旧 `>1e-9` 判据）")

    def test_real_gain_and_direction(self):
        """Mk06 +0.7303%/p=2e-11 是增益；显著为负不是增益；方向必须为正。"""
        m = self._m()
        self.assertTrue(m.is_gain(0.7303, 2e-11))
        self.assertFalse(m.is_gain(-0.7303, 2e-11), "显著为负不能算「有增益」")
        self.assertFalse(m.is_gain(0.5, m.GAIN_P), "p 必须严格小于阈值")
        self.assertFalse(m.is_gain(None, 1e-9))
        self.assertFalse(m.is_gain(0.5, None))

    def test_gate_actually_calls_is_gain(self):
        """端到端接线：hurink_gate / fig_gate 必须真的调用 is_gain。

        缺陷 18 的形态正是"helper 写了、调用方没接"——只测 helper 会漏掉这一类。
        """
        m = self._m()
        for fn in (m.hurink_gate, m.fig_gate):
            src = inspect.getsource(fn)
            self.assertIn("is_gain(", src, "%s 没有接上 is_gain" % fn.__name__)
            self.assertNotIn("> 1e-9", src,
                             "%s 里还留着「浮点非零」判据" % fn.__name__)

    def test_dev_confusion_matrix_is_perfect_under_the_fixed_criterion(self):
        """用仓库里现成的 logs 复核开发集：修正判据 -> tp=3, fp=0, fn=0, tn=7（判对率 1.00）。

        logs 缺失时跳过（数据不是代码的一部分，不能因为它不在就判失败）。
        """
        m = self._m()
        pa = os.path.join(os.path.dirname(__file__), '..', 'logs', 'aig_gating.json')
        pp = os.path.join(os.path.dirname(__file__), '..', 'logs', 'init_probe.json')
        if not (os.path.exists(pa) and os.path.exists(pp)):
            self.skipTest("缺 logs/aig_gating.json 或 logs/init_probe.json")
        import collections
        import json
        with open(pa, encoding="utf-8") as fh:
            aig = json.load(fh)
        with open(pp, encoding="utf-8") as fh:
            prb = json.load(fh)
        T = {r["instance"]: r for r in aig["targets"]["vs_I_mix3"]["per_instance"]}
        L = {r["instance"]: r["probe_init_hv_lever_pct"] for r in prb["probe"]}
        c = collections.Counter()
        for i in L:
            gain = m.is_gain(T[i]["dhv_rel_pct"], T[i]["p"])
            pred = L[i] >= -1.0
            c["tp" if (pred and gain) else "fp" if pred else
              "fn" if gain else "tn"] += 1
        self.assertEqual((c["tp"], c["fp"], c["fn"], c["tn"]), (3, 0, 0, 7),
                         "开发集门控矩阵与文档记载不一致：%s" % dict(c))
        self.assertEqual(sorted(i for i in L if L[i] >= -1.0),
                         ["Mk06", "Mk08", "Mk10"])


class TestTexOutputIsClean(unittest.TestCase):
    """缺陷 31 回归锁：生成的 .tex 里不许有控制字符。

    形态：Python 普通字符串里 `\\t` 是 TAB、`\\f` 是换页符。把 `"\\footnotesize"` /
    `"\\textbf{...}"` 写成单反斜杠**不报错**，静默产出 `<FF>ootnotesize` 与
    `<TAB>extbf{...}` —— `ast.parse` 过、TECTONIC 编译过、缺字告警为 0，
    但 PDF 上直接排出 "ootnotesize"、"extbf{逐实例}" 这类垃圾文本。
    2026-09-19 实测 **6 张表全中**（`tab_ladder` 的 `\\footnotesize` 失效还顺带
    让它超出页面 84.6pt），所以必须在写盘前与编译前各拦一次。
    """

    @staticmethod
    def _m():
        import importlib
        _d = os.path.join(os.path.dirname(__file__), '..', 'scripts')
        if _d not in sys.path:
            sys.path.insert(0, _d)
        return importlib.import_module("paper_export")

    CTRL = "\t\f\r\v\a\b\0"

    def test_write_tex_rejects_control_chars(self):
        """写盘守卫必须有区分度：带 TAB/FF/CR 的内容一律拒绝，且不留残文件。"""
        m = self._m()
        tabs = os.path.join(os.path.dirname(__file__), '..', 'paper', 'tables')
        probe = os.path.join(tabs, "_should_never_exist.tex")
        for ch in ("\t", "\f", "\r"):
            with self.assertRaises(SystemExit, msg="控制字符 %r 没被拦住" % ch):
                m.write_tex("_should_never_exist.tex", "x%sy" % ch)
        self.assertFalse(os.path.exists(probe),
                         "被拒绝的内容不该落盘（否则守卫只是报警，不是拦截）")

    def test_table_wrap_output_has_no_control_chars(self):
        m = self._m()
        t = m.table_wrap("标题", "tab:x", "    a & b \\\\", "ll",
                         notes="注：\\textbf{粗体} 与 \\texttt{code}。")
        bad = [c for c in t if c in self.CTRL]
        self.assertEqual(bad, [], "table_wrap 输出含控制字符 %r" % bad)
        # font 默认值必须真的是 \small 命令，而不是被吃成控制字符 + "mall"
        self.assertIn("\\small", t)

    def test_repo_tables_are_clean(self):
        """端到端：仓库里现成的 paper/tables/*.tex 必须干净（表不在就跳过）。"""
        import glob
        d = os.path.join(os.path.dirname(__file__), '..', 'paper', 'tables')
        files = sorted(glob.glob(os.path.join(d, "*.tex")))
        if not files:
            self.skipTest("paper/tables 下没有产物")
        for f in files:
            src = io.open(f, encoding="utf-8").read()
            bad = sorted({c for c in src if c in self.CTRL})
            self.assertEqual(bad, [], "%s 含控制字符 %r"
                             % (os.path.basename(f),
                                ["U+%04X" % ord(c) for c in bad]))


class TestScriptCLIHelp(unittest.TestCase):
    """缺陷 19（文档命令未实测）/ 27（argparse help 里的裸 `%` 直接崩）的机器化锁。

    2026-09-18 实测 `scripts/init_probe.py --help` 抛
    `ValueError: unsupported format character '?' (0x5f53)` —— argparse 会对 help
    字符串做一次 `%` 格式化，写 `（… rel%）` 这种中文括号紧跟百分号就崩。
    这类错误**只有真的跑一次 --help 才会发现**，所以把它钉成回归锁：
    每个含 argparse 的脚本 `--help` 必须退出码 0。
    """

    SCRIPTS = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")

    def test_scripts_dir_is_not_empty(self):
        names = [n for n in os.listdir(self.SCRIPTS)
                 if n.endswith(".py") and not n.startswith("_")]
        self.assertGreaterEqual(len(names), 20)

    def test_every_script_help_exits_zero(self):
        bad = []
        for n in sorted(os.listdir(self.SCRIPTS)):
            if not n.endswith(".py") or n.startswith("_"):
                continue
            path = os.path.join(self.SCRIPTS, n)
            with open(path, encoding="utf-8", errors="ignore") as fh:
                src = fh.read()
            if "argparse" not in src:
                continue
            r = subprocess.run([sys.executable, path, "--help"],
                               capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                tail = ((r.stderr or "").strip().splitlines() or [""])[-1]
                bad.append("%s -> %s" % (n, tail[:140]))
        self.assertEqual(
            bad, [],
            "这些脚本 --help 失败（多半是 help 里的裸 %%）：\n" + "\n".join(bad))

    def test_documented_init_probe_command_runs(self):
        """文档 §8 写死的 init_probe 调用必须真的能跑通（缺陷 19）。"""
        out = os.path.join(tempfile.gettempdir(), "_probe_argcheck.json")
        r = subprocess.run(
            [sys.executable, os.path.join(self.SCRIPTS, "init_probe.py"),
             "--instances", "Mk01", "--seeds", "1", "--perm", "0", "--out", out],
            capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.exists(out))


class TestPaperNumbersHaveOneSource(unittest.TestCase):
    """缺陷 30 系列回归锁：**正文数字必须来自数据**，且生成的 .tex 必须真能被 TeX 吃下。

    已发生过的三个形态（都不是笔误，而是"不做机械比对就发现不了"的坑）：

    (a) 正文手写的数字与表格漂移：论文 §5 把 ``rho=+0.83`` 归给式 (2) 定义的 probe，
        而 0.83 实际属于另一个事前量（平均 makespan 杠杆），probe 自己是 0.77 ——
        两个数**都真实存在**，肉眼怎么看都对。
    (b) 宏名带数字：``\\NullP95`` 被 TeX 读成 ``\\NullP`` 紧跟 ``95``，
        报 ``Missing number, treated as zero.`` —— **不报"名字非法"**，
        而且行号指向名字的**前一行**，排查方向被完全带偏。
    (c) 中文出现在 preamble 里的宏体中 → XeTeX 报 ``Missing \\begin{document}``。

    所以锁三件事：正文无字面副本、(b)/（c) 的机器守卫、以及 §hurink 必须有正文
    （曾经只剩 \\input 两张占位表，整节没有一句解释）。
    """

    BS = chr(92)

    @staticmethod
    def _m():
        import importlib
        _d = os.path.join(os.path.dirname(__file__), "..", "scripts")
        if _d not in sys.path:
            sys.path.insert(0, _d)
        return importlib.import_module("paper_export")

    @staticmethod
    def _paper():
        return os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "paper")

    # ---- (b) 宏名只能是字母：守卫必须有区分度 ----
    def test_macro_name_with_digit_is_rejected(self):
        m = self._m()
        for bad in ("NullP95", "PSmk9Lam", "CalibAmpD2D1", "A2B"):
            with self.assertRaises(SystemExit, msg="宏名 %r 竟被放行" % bad):
                m.tex_macro_line(bad, 1)
        good = m.tex_macro_line("NullPct", 1)
        self.assertEqual(good, self.BS + "newcommand{" + self.BS + "NullPct}{1}")

    def test_repo_macros_names_are_tex_legal(self):
        """端到端：仓库里现成的 macros.tex 名字必须全是字母（产物不在就跳过）。

        故意不用正则解析：本文件要写进 Python 源码的反斜杠会经过工具层，
        用字符串拼接（self.BS）比正则转义更不容易写错。
        """
        p = os.path.join(self._paper(), "tables", "macros.tex")
        if not os.path.exists(p):
            self.skipTest("paper/tables/macros.tex 不存在")
        head = self.BS + "newcommand" + "{"
        names = []
        for line in io.open(p, encoding="utf-8"):
            s = line.strip()
            if s.startswith(head):
                # 行形如 "\\newcommand{\\HKready}{0}"：split("{")[1] 是 "\\HKready}"
                names.append(s.split("{")[1].split("}")[0].lstrip(self.BS))
        self.assertGreater(len(names), 10, "没解析到宏名，产物结构变了")
        bad = [n for n in names if not n.isalpha()]
        self.assertEqual(bad, [], "这些宏名含非字母字符，TeX 会解析错：%r" % bad)

    # ---- (a) 正文里不许有已宏化数字的字面副本 ----
    def test_main_tex_has_no_duplicated_literals(self):
        sc = os.path.join(os.path.dirname(__file__), "..", "scripts",
                          "check_paper_literals.py")
        if not os.path.exists(sc):
            self.skipTest("check_paper_literals.py 不存在")
        r = subprocess.run([sys.executable, sc], capture_output=True, text=True,
                           timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    # ---- §hurink 必须有正文，不能只剩 \input ----
    def test_hurink_section_has_prose_body(self):
        p = os.path.join(self._paper(), "main.tex")
        if not os.path.exists(p):
            self.skipTest("paper/main.tex 不存在")
        src = io.open(p, encoding="utf-8").read()
        i = src.find(self.BS + "section{独立留出验证")
        self.assertGreater(i, 0, "找不到留出验证一节")
        j = src.find(self.BS + "input{", i)
        self.assertGreater(j, i, "该节里没有 input，结构变了")
        # 统计"解释性文字"：去掉 LaTeX 命令名（反斜杠+字母）、注释行与空白后
        # 仍应有足够字符——只剩两条 \input 占位表就是这里的失败形态。
        txt, k, body = [], 0, "\n".join(
            ln for ln in src[i:j].splitlines() if not ln.strip().startswith("%"))
        while k < len(body):
            if body[k] == self.BS:
                k += 1
                while k < len(body) and body[k].isalpha():
                    k += 1
                continue
            if not body[k].isspace() and body[k] not in "{}[]$":
                txt.append(body[k])
            k += 1
        self.assertGreater(len(txt), 150,
                           "留出验证一节没有正文——只剩占位表就无法解释结果")

    # ---- 结论句按数据分支：两种形态都要出得来且不崩 ----
    def _fake(self, n_gain, hold):
        lam = {"Hed01": -0.5, "Hed02": -3.0}
        gain = {"Hed01": bool(n_gain), "Hed02": False}
        return {"hurink": {"n": 2, "n_gain": n_gain, "thr": -1.0,
                           "dev": {"tp": 3, "fp": 0, "fn": 0, "tn": 7},
                           "hold": hold, "agree": hold.get("tp", 0) + hold.get("tn", 0),
                           "lambda": lam, "gain": gain}}

    def test_verdict_reports_zero_gain_branch(self):
        m = self._m()
        lines = m.write_macros(self._fake(0, {"tp": 0, "fp": 0, "fn": 0, "tn": 2}),
                               False, write=False)
        v = [l for l in lines if "HKverdict" in l][0]
        self.assertIn("一次也没有", v)
        self.assertIn("\\textbf", v)
        bad = [c for c in v if c in "\t\f\r\v"]
        self.assertEqual(bad, [])

    def test_verdict_reports_counts_when_gain_exists(self):
        m = self._m()
        lines = m.write_macros(self._fake(1, {"tp": 1, "fp": 0, "fn": 0, "tn": 1}),
                               False, write=False)
        v = [l for l in lines if "HKverdict" in l][0]
        self.assertIn("1/2", v, "有增益时必须报出计数而不是笼统措辞")
        self.assertNotIn("一次也没有", v)

    def test_macros_written_without_write_flag(self):
        """write=False 不许碰磁盘——测试用它，避免污染真产物。"""
        m = self._m()
        fp = os.path.join(self._paper(), "tables", "macros.tex")
        before = io.open(fp, encoding="utf-8").read() if os.path.exists(fp) else None
        m.write_macros(self._fake(0, {"tp": 0, "fp": 0, "fn": 0, "tn": 2}),
                       False, write=False)
        after = io.open(fp, encoding="utf-8").read() if os.path.exists(fp) else None
        self.assertEqual(before, after, "write=False 竟然改了产物")

    # ---- 阶梯 / 响应面：正文数字必须与生成的表格**同源** ----
    #
    # 这两个是 2026-09-19 第二次复核查出的"缺陷 32 同类"形态：正文 §4.1 的
    # `+3.41% / +0.42% / +0.08% / +0.03% / -0.003%` 与 §4.2 的
    # `0.024/0.064/0.091 → 0.0010/0.0028/0.0085` 全是**手写**，
    # 而 tab_ladder.tex / tab_surface.tex 是现场生成的 —— 两处来源，今天恰好一致。
    # 下面的锁把它们钉在同一份 `logs/` 上：改数据只改一处。
    @staticmethod
    def _macro_values(path):
        out = {}
        for line in io.open(path, encoding="utf-8"):
            s = line.strip()
            if s.startswith(chr(92) + "newcommand{"):
                name = s.split("{")[1].split("}")[0].lstrip(chr(92))
                out[name] = s[s.index("}{") + 2:-1]
        return out

    @staticmethod
    def _rows(path):
        rows = {}
        for line in io.open(path, encoding="utf-8"):
            if "&" not in line:
                continue
            cells = [c.strip() for c in line.split("&")]
            rows[cells[0].replace("$|$", "|")] = cells
        return rows

    def test_ladder_macros_agree_with_generated_table(self):
        d = self._paper()
        mp = os.path.join(d, "tables", "macros.tex")
        tp = os.path.join(d, "tables", "tab_ladder.tex")
        if not (os.path.exists(mp) and os.path.exists(tp)):
            self.skipTest("论文产物未生成（先跑 scripts/paper_export.py）")
        import re
        mac, rows = self._macro_values(mp), self._rows(tp)
        pairs = (("LadderMixRel", "D2|D1"), ("LadderVnsRel", "D3|D2"),
                 ("LadderQpasRel", "D4|D3"), ("LadderEliteRel", "D5|D4"),
                 ("LadderRlRel", "RMOEAD|D5"))
        for name, key in pairs:
            self.assertIn(name, mac, "宏 %s 缺失：正文 §4.1 会渲染成空" % name)
            self.assertIn(key, rows, "tab_ladder 里找不到 %s 行" % key)
            nums = [float(t) for t in re.findall(r"[+-]?\d+\.\d+", rows[key][2])]
            self.assertTrue(
                any(abs(float(mac[name]) - v) < 5e-4 for v in nums),
                "%s = %s 与表行 %s 第三列 %r 对不上（正文与表格不同源）"
                % (name, mac[name], key, rows[key][2]))

    def test_surface_macros_agree_with_generated_table(self):
        d = self._paper()
        mp = os.path.join(d, "tables", "macros.tex")
        tp = os.path.join(d, "tables", "tab_surface.tex")
        if not (os.path.exists(mp) and os.path.exists(tp)):
            self.skipTest("论文产物未生成（先跑 scripts/paper_export.py）")
        import re
        mac, rows = self._macro_values(mp), self._rows(tp)
        self.assertIn("D2|D1", rows)
        end = [float(t) for t in re.findall(r"[+-]?\d+\.\d+", " ".join(rows["D2|D1"][2:]))]
        # 表格是 3 位小数、宏是 4 位小数 → 容差取表末位的半格
        for name in ("SurfInitEndA", "SurfInitEndB", "SurfInitEndC"):
            self.assertIn(name, mac, "宏 %s 缺失：正文 §4.2 会渲染成空" % name)
            self.assertTrue(any(abs(float(mac[name]) - v) < 1e-3 for v in end),
                            "%s = %s 与表行对不上" % (name, mac[name]))
        # 衰减倍数必须 > 1（否则"显著衰减"这句话不成立）
        self.assertGreater(float(mac["SurfDecayHi"]), 1.0)
        self.assertGreaterEqual(float(mac["SurfDecayHi"]), float(mac["SurfDecayLo"]))
        # 表注里那个"最大者"文本必须含 ```\\times``` 或纯数字，且不能是空的
        self.assertTrue(mac["SurfNonInitMax"].strip("$").strip(),
                        "SurfNonInitMax 为空，表注与正文会渲染成空")


if __name__ == "__main__":
    unittest.main()