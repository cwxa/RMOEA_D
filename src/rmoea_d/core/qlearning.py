"""
Q-learning Parameter Adaptation Strategy (Q-PAS) for neighborhood size T.
Q-learning参数自适应策略(Q-PAS)：自动选择邻域大小T。
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class QLearningPAS:
    """Q-learning for automatically selecting neighborhood size T.
    Q-learning自动选择邻域大小T，基于收敛性和多样性指标。"""

    def __init__(self, alpha=0.4, gamma=0.6, epsilon=0.8, actions=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions if actions is not None else [5, 10, 15, 20]
        self.n_actions = len(self.actions)
        self.n_states = 4
        self.q_table = np.zeros((self.n_states, self.n_actions))
        self.prev_cv = None
        self.prev_dv = None
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

    def select_action(self, state, rng):
        """Epsilon-greedy action selection (paper's reversed version).
        ε-贪心动作选择策略。"""
        if rng.rand() < self.epsilon:
            return rng.randint(self.n_actions)
        else:
            return int(np.argmax(self.q_table[state]))

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

        if self.prev_cv is None:
            # First generation
            self.prev_cv = cv
            self.prev_dv = dv
            state = 0
            action_idx = self.select_action(state, rng)
            self.prev_state = state
            self.prev_action_idx = action_idx
            logger.info("Q-learning init: CV=%.4f, DV=%.4f, state=%d, action=%d (T=%d)",
                        cv, dv, state, action_idx, self.get_T(action_idx))
            return self.get_T(action_idx), True

        delta_cv = self.prev_cv - cv
        delta_dv = dv - self.prev_dv
        next_state = self.get_state(delta_cv, delta_dv)
        reward = 10.0 if delta_dv > 0 else 0.0

        self.update(self.prev_state, self.prev_action_idx, reward, next_state)

        action_idx = self.select_action(next_state, rng)
        logger.info("Q-learning step: delta_CV=%.4f, delta_DV=%.4f, state=%d->%d, "
                    "reward=%.1f, action=%d (T=%d)",
                    delta_cv, delta_dv, self.prev_state, next_state, reward,
                    action_idx, self.get_T(action_idx))

        self.prev_cv = cv
        self.prev_dv = dv
        self.prev_state = next_state
        self.prev_action_idx = action_idx
        return self.get_T(action_idx), False
