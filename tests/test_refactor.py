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


if __name__ == "__main__":
    unittest.main()
