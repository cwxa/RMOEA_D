#!/usr/bin/env python3
"""
Generate fixed test cases for all Brandimarte instances.
为所有Brandimarte实例生成固定测试用例，确保实验可复现。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rmoea_d.core.instance import generate_test_cases


if __name__ == "__main__":
    generate_test_cases(data_dir="data", output_dir="test_cases", seed=42)
