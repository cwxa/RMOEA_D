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
    "hv_cont"   连续奖励 ΔHV×100（可负，裁到 ±10）——保留改善的**幅度**信息，
                而非只保留符号。实测 Mk10 上二值 "hv" 的 T 分布仍偏小 T。
    ==========  =================================================

    背景：论文式(20) 只奖励「间距指标 DV 变大」。实测在 Mk01 上该信号会把
    策略推向恒选 T=5（论文 Table 3 的丰富策略 [5,15,20] 无法复现），而
    固定 T=5 的最终 HV 明显劣于 T=10。原因是 DV 是「间距均匀度」，
    与 HV 并不单调一致——奖励目标与优化目标错位。

    ``tie_break`` 控制「多个动作 Q 值并列最大」时的选择方式：

    ==========  =====================================================
    "random"    在并列最优的动作中随机挑一个（默认，标准做法）
    "argmax"    固定取索引最小的（np.argmax 原生行为，保留以复现旧结果）
    ==========  =====================================================

    **为什么必须随机**：Q 表零初始化时全表 Q=0，``np.argmax`` 一律返回
    索引 0，即 ``actions[0]``。在论文的 ε=0.8（80% 利用）下，前期几乎所有
    决策都落到该动作上，而一旦它获得奖励就进一步被强化 —— 策略自锁。
    实测 Mk10（T 是强杠杆的实例）上，argmax 平局口径有 **47%~53% 的代数
    停在最差的 T=5**；即便把最优的 T=50 放进候选集，它也只被选中 16.8%。
    平局随机化打破该偏置后，Q-PAS 才可能真正学到「选大 T」。
    """

    def __init__(self, alpha=0.4, gamma=0.6, epsilon=0.8, actions=None,
                 reward_mode="dv", hv_bounds=None, cv_normalize=False,
                 w_cv=5.0, w_dv=5.0, w_hv=10.0,
                 w_hv_cont=100.0, w_hv_clip=10.0,
                 tie_break="random", q_init="zero", q_init_scale=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions if actions is not None else [5, 10, 15, 20]
        self.n_actions = len(self.actions)
        self.n_states = 4
        self.cv_normalize = cv_normalize
        self.tie_break = tie_break
        self.q_init = q_init
        if q_init == "optimistic":
            # 乐观初始化：同样用于打破「全零 → 恒取索引 0」的对称性。
            # 用固定种子保证可复现（后续 update 会按奖励把各动作分化开）。
            _r = np.random.RandomState(20240916)
            self.q_table = _r.uniform(0.0, q_init_scale,
                                      size=(self.n_states, self.n_actions))
        else:
            self.q_table = np.zeros((self.n_states, self.n_actions))
        # ── 奖励设定 ──
        self.reward_mode = reward_mode
        self.hv_bounds = hv_bounds
        self.w_cv = w_cv
        self.w_dv = w_dv
        self.w_hv = w_hv
        self.w_hv_cont = w_hv_cont
        self.w_hv_clip = w_hv_clip
        # ── 状态缓存 ──
        self.prev_cv = None
        self.prev_dv = None
        self.prev_hv = None
        self.prev_state = None
        self.prev_action_idx = None

    def compute_cv_dv(self, pf, normalize=None):
        """
        Compute convergence (CV) and diversity (DV) from Pareto front.
        计算Pareto前沿的收敛性(CV)和多样性(DV)。

        论文式 (14) 的 CV 是「到参考点 P* 的均方距离」，未规定归一化。
        当两个目标量纲悬殊时，CV 会被大量纲目标独占——实测 Mk10 上
        f2（总负载）占 CV² 的 96.5%，makespan 仅 3.5%，于是 ΔCV 的符号
        几乎只反映 f2 的方向；把目标拉回同一尺度后约 45% 的历史状态
        会翻转。因此提供 ``cv_normalize`` 开关：

        - ``False``（默认）：严格照论文，用原始目标值计算（复现性优先）
        - ``True``：先用 ``hv_bounds``（缺省则用本代前沿范围）归一到同一
          尺度，使 CV 同时反映两个目标

        Parameters
        ----------
        pf : list of (f1, f2) crisp objective vectors
        normalize : bool or None
            None 时沿用 ``self.cv_normalize``。
        """
        if normalize is None:
            normalize = self.cv_normalize
        if pf is None or len(pf) == 0:
            return 0.0, 0.0
        pf = np.asarray(pf, dtype=float)
        if pf.ndim != 2:
            return 0.0, 0.0

        if normalize:
            if self.hv_bounds is not None:
                lo = np.asarray(self.hv_bounds[0], dtype=float)
                hi = np.asarray(self.hv_bounds[1], dtype=float)
            else:
                lo, hi = pf.min(axis=0), pf.max(axis=0)
            span = np.where((hi - lo) > 1e-12, hi - lo, 1.0)
            pf = (pf - lo) / span

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
        if self.reward_mode == "hv_cont":
            if self.prev_hv is None or cur_hv is None:
                return 0.0
            return float(np.clip((cur_hv - self.prev_hv) * self.w_hv_cont,
                                 -self.w_hv_clip, self.w_hv_clip))
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

        利用分支里，论文未规定 Q 值并列时的取舍。这里默认 **并列随机**
        （``tie_break="random"``）——因为 Q 表初值为 0，若固定取索引 0，
        在 ε 很大时会把整个搜索锁死在 ``actions[0]`` 上。
        """
        if rng.rand() < self.epsilon:
            q = self.q_table[state]
            mx = q.max()
            best = np.flatnonzero(q >= mx - 1e-12)
            if len(best) == 1 or self.tie_break != "random":
                return int(best[0])
            return int(best[rng.randint(len(best))])
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
        cur_hv = (self._front_hv(pf)
                  if self.reward_mode in ("hv", "hv_cont") else None)

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
