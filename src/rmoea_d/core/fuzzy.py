"""
Triangular fuzzy number operations for MOFFJSP.
三角模糊数运算模块，用于模糊柔性作业车间调度问题。
"""


class FuzzyNumber:
    """Triangular fuzzy number (t1, t2, t3) where t1 <= t2 <= t3.
    三角模糊数 (t1, t2, t3)，t1为最早时间，t2为最可能时间，t3为最晚时间。"""
    __slots__ = ('t1', 't2', 't3')

    def __init__(self, t1, t2, t3):
        self.t1 = float(t1)
        self.t2 = float(t2)
        self.t3 = float(t3)

    def __add__(self, other):
        if not isinstance(other, FuzzyNumber):
            return NotImplemented
        return FuzzyNumber(self.t1 + other.t1, self.t2 + other.t2, self.t3 + other.t3)

    def __repr__(self):
        return f"FuzzyNumber({self.t1:.2f}, {self.t2:.2f}, {self.t3:.2f})"

    def clear_value(self):
        """Crisp value: (t1 + 2*t2 + t3) / 4
        清晰值计算，用于模糊数比较和标量化。"""
        return (self.t1 + 2.0 * self.t2 + self.t3) / 4.0

    def to_tuple(self):
        """Return (t1, t2, t3) tuple for serialization.
        返回三元组用于序列化。"""
        return (self.t1, self.t2, self.t3)

    @staticmethod
    def from_tuple(t):
        """Create FuzzyNumber from (t1, t2, t3) tuple.
        从三元组创建模糊数。"""
        return FuzzyNumber(t[0], t[1], t[2])


def fuzzy_max(a, b):
    """Return the larger fuzzy number using the three-stage ranking operator from the paper.
    论文中的三阶段排序算子 (Section 2.5):
    (a) f1(x) = (x1 + 2*x2 + x3) / 4,  if f1(s) > f1(t), then s > t;
    (b) f2(x) = x2,                    when f1(s) = f1(t), if f2(s) > f2(t), then s > t;
    (c) f3(x) = x3 - x1,               when f2(s) = f2(t), if f3(s) > f3(t), then s > t.
    """
    f1_a = (a.t1 + 2.0 * a.t2 + a.t3) / 4.0
    f1_b = (b.t1 + 2.0 * b.t2 + b.t3) / 4.0
    if f1_a > f1_b:
        return a
    if f1_a < f1_b:
        return b
    # f1 equal, compare f2 = t2
    if a.t2 > b.t2:
        return a
    if a.t2 < b.t2:
        return b
    # f2 equal, compare f3 = t3 - t1
    f3_a = a.t3 - a.t1
    f3_b = b.t3 - b.t1
    if f3_a > f3_b:
        return a
    if f3_a < f3_b:
        return b
    return a  # equal


def fuzzy_sort_key(fn):
    """Return a sortable tuple for fuzzy numbers.
    返回模糊数的可排序元组，用于排序操作。"""
    return (fn.clear_value(), fn.t2, fn.t3 - fn.t1)


def _fuzzy_lt(fa, fb):
    """fa < fb according to the paper's ranking operator (minimization).
    基于论文三阶段排序算子判断模糊数fa是否小于fb。
    """
    f1_a = (fa[0] + 2.0 * fa[1] + fa[2]) / 4.0
    f1_b = (fb[0] + 2.0 * fb[1] + fb[2]) / 4.0
    if f1_a != f1_b:
        return f1_a < f1_b
    if fa[1] != fb[1]:
        return fa[1] < fb[1]
    return (fa[2] - fa[0]) < (fb[2] - fb[0])


def _fuzzy_eq(fa, fb):
    return fa[0] == fb[0] and fa[1] == fb[1] and fa[2] == fb[2]


def fuzzy_tuple_clear_value(t):
    """Compute crisp value from a (t1, t2, t3) tuple.
    从三元组计算清晰值。
    """
    return (t[0] + 2.0 * t[1] + t[2]) / 4.0


def fuzzy_dominates(a, b):
    """True if fuzzy objective vector a dominates b (minimization).
    基于论文三阶段排序算子的模糊支配判断。
    a, b 均为 ((m1,m2,m3), (w1,w2,w3)) 格式的元组。
    """
    a_better = []
    for i in range(2):
        if _fuzzy_eq(a[i], b[i]):
            a_better.append(False)
        elif _fuzzy_lt(a[i], b[i]):
            a_better.append(True)
        else:
            return False  # a is worse on at least one objective
    return any(a_better)


# Constants
INF = FuzzyNumber(1e18, 1e18, 1e18)
ZERO = FuzzyNumber(0.0, 0.0, 0.0)
