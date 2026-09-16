"""
Q-learning Parameter Adaptation Strategy (Q-PAS) for neighborhood size T.
Q-learning参数自适应策略(Q-PAS)：自动选择邻域大小T。
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class QLearningPAS:
    """Q-learning for automatically selecting neighborhood size T.
    Q-learning自动选择邻域大小T，基于收敛性和多样性指标。

    ``reward_mode`` 控制奖励信号（默认 "dv" 严格照论文式(20)）：

    ==========  =================================================
    "dv"        论文原始口径：ΔDV > 0 → 10，否则 0
    "cv_dv"     ΔCV>0 与 ΔDV>0 各计 5 分（把收敛性也纳入奖励）
    "hv"        ΔHV > 0 → 10，否则 0（奖励直接对齐最终评价指标）
    ==========  =================================================

    背景：论文式(20) 只奖励「间距指标 DV 变大」。实测在 Mk01 上该信号会把
    策略推向恒选 T=5（论文 Table 3 的丰富策略 [5,15,20] 无法复现），而
    固定 T=5 的最终 HV 明显劣于 T=10。原因是 DV 是「间距均匀度」，
    与 HV 并不单调一致——奖励目标与优化目标错位。
    """

    def __init__(self, alpha=0.4, gamma=0.6, epsilon=0.8, actions=None,
                 reward_mode="dv", hv_bounds=None,
                 w_cv=5.0, w_dv=5.0, w_hv=10.0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions if actions is not None else [5, 10, 15, 20]
        self.n_actions = len(self.actions)
        self.n_states = 4
        self.q_table = np.zeros((self.n_states, self.n_actions))
        # ── 奖励设定 ──
        self.reward_mode = reward_mode
        self.hv_bounds = hv_bounds
        self.w_cv = w_cv
        self.w_dv = w_dv
        self.w_hv = w_hv
        # ── 状态缓存 ──
        self.prev_cv = None
        self.prev_dv = None
        self.prev_hv = None
        self.prev_state = None
        self.prev_action_idx = None

    def compute_cv_dv(self, pf):
        """
        Compute convergence (CV) and diversity (DV) from Pareto front.
        计算Pareto前沿的收敛性(CV)和多样性(DV)。
        pf: list of (f1, f2) crisp objective vectors.
        """
        if len(pf) == 0:
            return 0.0, 0.0
        pf = np.array(pf)

        # CV: sqrt of mean squared distance to ideal point (0,0)
        # CV: 到理想点(0,0)的均方距离平方根
        dists = np.sqrt(np.sum(pf ** 2, axis=1))
        cv = np.sqrt(np.mean(dists ** 2))

        # DV: spacing metric
        # DV: 间距指标，衡量前沿点分布均匀性
        if len(pf) < 2:
            dv = 0.0
        else:
            # Sort by first objective
            idx = np.argsort(pf[:, 0])
            sorted_pf = pf[idx]
            ds = []
            for i in range(len(sorted_pf) - 1):
                d = np.linalg.norm(sorted_pf[i] - sorted_pf[i + 1])
                ds.append(d)
            if len(ds) == 0:
                dv = 0.0
            else:
                mean_d = np.mean(ds)
                if mean_d == 0:
                    dv = 0.0
                else:
                    dv = sum(abs(d - mean_d) for d in ds) / ((len(ds)) * mean_d)
        return cv, dv

    def get_state(self, delta_cv, delta_dv):
        """Map (ΔCV, ΔDV) to state 0-3.
        将(ΔCV, ΔDV)映射到状态0-3。"""
        if delta_cv > 0 and delta_dv > 0:
            return 0
        elif delta_cv > 0 and delta_dv <= 0:
            return 1
        elif delta_cv <= 0 and delta_dv > 0:
            return 2
        else:
            return 3

    def _front_hv(self, pf):
        """用固定归一化边界计算 PF 的 HV（供 "hv" 奖励模式使用）。"""
        try:
            from ..utils.metrics import compute_hv
        except ImportError:  # pragma: no cover
            from rmoea_d.utils.metrics import compute_hv
        arr = np.asarray(pf, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return 0.0
        return float(compute_hv(arr, norm_bounds=self.hv_bounds))

    def _compute_reward(self, delta_cv, delta_dv, cur_hv=None):
        """按 reward_mode 计算奖励 R(S_t, A_t)。"""
        if self.reward_mode == "hv":
            if self.prev_hv is None or cur_hv is None:
                return 0.0
            return self.w_hv if (cur_hv - self.prev_hv) > 0 else 0.0
        if self.reward_mode == "cv_dv":
            return (self.w_cv if delta_cv > 0 else 0.0) + \
                   (self.w_dv if delta_dv > 0 else 0.0)
        # "dv"：论文式 (20) 原始口径
        return 10.0 if delta_dv > 0 else 0.0

    def select_action(self, state, rng):
        """
        ε-greedy action selection.

        按论文 Algorithm 3 第 5–8 行实现：

            if rand < ε:  选择当前 Q 值最大的动作 A_t   （利用）
            else:         随机选择一个动作               （探索）

        论文正文（Section 4.5.5）特别说明该写法「与传统的 ε-greedy 相反」，
        目的是让实验中的 ε 取值更直观。因此 ε = 0.8 的含义是
        **80% 利用、20% 探索**——若把分支写反（rand < ε 时随机），
        在 ε = 0.8 下会退化成 80% 随机，Q-table 基本学不到东西。
        """
        if rng.rand() < self.epsilon:
            return int(np.argmax(self.q_table[state]))
        return int(rng.randint(self.n_actions))

    def get_T(self, action_idx):
        """Get neighborhood size T from action index."""
        return self.actions[action_idx]

    def update(self, state, action_idx, reward, next_state):
        """Update Q-value using standard Q-learning formula.
        使用标准Q-learning公式更新Q值。"""
        old_q = self.q_table[state, action_idx]
        max_next = np.max(self.q_table[next_state])
        self.q_table[state, action_idx] = old_q + self.alpha * (reward + self.gamma * max_next - old_q)

    def step(self, pf, rng):
        """
        Execute one Q-learning step.
        执行一步Q-learning更新。
        Returns: selected T, is_first_step
        """
        cv, dv = self.compute_cv_dv(pf)
        cur_hv = self._front_hv(pf) if self.reward_mode == "hv" else None

        if self.prev_cv is None:
            # First generation
            self.prev_cv = cv
            self.prev_dv = dv
            self.prev_hv = cur_hv
            state = 0
            action_idx = self.select_action(state, rng)
            self.prev_state = state
            self.prev_action_idx = action_idx
            logger.debug("Q-learning init: CV=%.4f, DV=%.4f, state=%d, action=%d (T=%d)",
                         cv, dv, state, action_idx, self.get_T(action_idx))
            return self.get_T(action_idx), True

        delta_cv = self.prev_cv - cv
        delta_dv = dv - self.prev_dv
        next_state = self.get_state(delta_cv, delta_dv)
        reward = self._compute_reward(delta_cv, delta_dv, cur_hv)

        self.update(self.prev_state, self.prev_action_idx, reward, next_state)

        action_idx = self.select_action(next_state, rng)
        logger.debug("Q-learning step: delta_CV=%.4f, delta_DV=%.4f, state=%d->%d, "
                     "reward=%.1f, action=%d (T=%d)",
                     delta_cv, delta_dv, self.prev_state, next_state, reward,
                     action_idx, self.get_T(action_idx))

        self.prev_cv = cv
        self.prev_dv = dv
        self.prev_hv = cur_hv
        self.prev_state = next_state
        self.prev_action_idx = action_idx
        return self.get_T(action_idx), False
