"""
单元测试：验证重构后核心功能的正确性

覆盖范围：
  1. MOEADBaseline 继承 RMOEAD 后行为一致
  2. RMOEAD 在 fixed_T 模式下正确跳过 Q-learning 初始化
  3. ablation_visualization.py 导入与数据加载
  4. experiment._extract_run 字段提取完整性
"""

import sys
import os
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

    def test_front_hv_uses_shared_bounds(self):
        ql = QLearningPAS(reward_mode="hv", hv_bounds=([0.0, 0.0], [10.0, 10.0]))
        front = np.array([[2.0, 2.0], [4.0, 1.0]])
        hv = ql._front_hv(front)
        self.assertGreater(hv, 0.0)
        self.assertLessEqual(hv, 1.0)          # 归一化后不超过单位盒


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


if __name__ == "__main__":
    unittest.main()
