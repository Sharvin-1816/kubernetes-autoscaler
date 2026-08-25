"""
KubernetesAutoscalingEnv
--------------------------
A Gymnasium-compatible environment for training RL agents to autoscale a
simulated Kubernetes service (horizontally and vertically) to minimize
SLO violations and cost.

State (observation), 9-dim continuous vector:
    [cpu_util, mem_util, latency_ms (normalized), active_pods (normalized),
     pending_pods (normalized), current_tier (normalized), traffic_trend,
     tier_cooldown_remaining (normalized), scaledown_cooldown_remaining (normalized)]

    traffic_trend = (request_rate_now - request_rate_prev) / base_rate
    -> gives the agent a signal about whether load is rising or falling,
       which matters a lot given the spin-up delay.

    tier_cooldown_remaining      = steps left before vertical (tier) change allowed again
    scaledown_cooldown_remaining = steps left before horizontal SCALE-DOWN allowed again
    (both normalized by their respective window length, 0.0 = free to act)
    See "Stabilization windows" below for why these are separate.

Action, MultiDiscrete([7, 3]):
    action[0] -> horizontal move: pod delta, mapped as (action[0] - 3):
                 0 -> -3 pods, 1 -> -2, 2 -> -1, 3 -> hold, 4 -> +1, 5 -> +2, 6 -> +3
                 (mirrors real Kubernetes HPA, which computes a target replica
                 count directly and can jump by more than one replica per
                 adjustment cycle - not just +/-1)
    action[1] -> vertical move:   0 = SMALL, 1 = MEDIUM, 2 = LARGE  (target tier)

Reward:
    reward = -(w_slo * slo_penalty) - (w_cost * cost_penalty) - (w_churn * churn_penalty)

    slo_penalty   : 1 if latency exceeds target this step, scaled by how far over
    cost_penalty  : normalized running cost this step
    churn_penalty : 1 if the agent changed horizontal or vertical action vs last step
                    (discourages flapping / thrashing)

    Weights are exposed as constructor args so a reward-ablation study is a
    one-line change (see agents/train.py).

Stabilization windows (root-cause fix, not a tuned constant) - TWO, asymmetric:
    Diagnostics showed the agent learned to rapidly toggle vertical tier
    (SMALL<->MEDIUM) near traffic peaks, exploiting the fact that tier
    changes instantly affect the WHOLE cluster's capacity. This produced
    severe latency oscillation (see MDP_SPECIFICATION.md). A first fix used
    ONE shared cooldown for both horizontal and vertical actions - this
    killed the oscillation but also throttled legitimate scale-UP
    responsiveness during ramp/spike, since scaling up got locked out for
    several steps right when more capacity was needed most (evidenced by
    higher p95 latency on ramp/spike/mixed after that fix, vs before it).

    Real Kubernetes HPA/VPA solve this with independent, ASYMMETRIC
    stabilization windows: scale-up reacts immediately by default (no
    window), while scale-down uses a cooldown (default 5 minutes) to avoid
    prematurely dropping capacity. We mirror that exactly:
        - Vertical (tier) changes: symmetric cooldown, both directions
          (VERTICAL_STABILIZATION_WINDOW) - this is what fixed oscillation,
          unchanged from the previous iteration.
        - Horizontal scale-UP: no cooldown, always immediate.
        - Horizontal scale-DOWN: cooldown-gated
          (HORIZONTAL_DOWN_STABILIZATION_WINDOW), preventing rapid
          down-up-down flapping without throttling upward responsiveness.
    Both remaining cooldowns are exposed in the observation so the
    environment stays Markovian - the agent can see it's constrained rather
    than experiencing invisible, state-dependent rule changes.

Episode:
    One episode = one traffic trace of `episode_length` steps, drawn from a
    workload pattern chosen at reset(). Training patterns and held-out
    evaluation patterns are kept separate (see `TRAIN_PATTERNS` /
    `EVAL_ONLY_PATTERNS` below) to support the generalization study.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.k8s_simulator import KubernetesSimulator, PodTier, LATENCY_TARGET_MS
from utils.workload_generator import WorkloadGenerator


# Patterns the agent is allowed to train on
TRAIN_PATTERNS = ["diurnal", "noisy_steady"]

# Patterns reserved ONLY for evaluation, never seen during training.
# This is what makes the generalization experiment valid.
EVAL_ONLY_PATTERNS = ["spike", "ramp", "mixed"]

MAX_LATENCY_FOR_NORM = 2000.0  # clip/normalize latency observation at this value

# MAX_PODS derived from actual peak-capacity analysis, not picked arbitrarily:
# worst-case traffic peak (spike pattern) ~= base_rate(50) + spike_height(150) = 200 req/s
# minimum pods needed at LARGE tier (55 req/s/pod) to cover that peak = ceil(200/55) = 4
# minimum pods needed at MEDIUM tier (30 req/s/pod) to cover that peak = ceil(200/30) = 7
# MAX_PODS set to ~2x the MEDIUM-tier minimum as operational headroom (imperfect
# scaling timing, spin-up delay, safety margin) -> NOT a scarcity constraint used
# to suppress a reward-hacking symptom.
MAX_PODS = 15

# Cost is normalized against a REALISTIC reference load (5 medium pods,
# INCLUDING the fixed per-pod overhead - see FIXED_POD_OVERHEAD in
# k8s_simulator.py), not the theoretical max. Keeps cost differences
# actually visible to the agent instead of vanishing to ~0 under normal load.
REFERENCE_COST = 5 * (2.2 + 0.3)  # 5 medium-tier pods' worth of cost per step (compute + overhead)

# Stabilization windows - mirror Kubernetes' real, ASYMMETRIC HPA/VPA
# behavior: scale-up reacts immediately, scale-down and vertical changes are
# cooldown-gated. See module docstring "Stabilization windows" for the full
# reasoning and the regression this fixes vs. the single shared-cooldown version.
VERTICAL_STABILIZATION_WINDOW = 3          # tier changes, both directions
HORIZONTAL_DOWN_STABILIZATION_WINDOW = 3   # horizontal scale-DOWN only; scale-up is never gated


class KubernetesAutoscalingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, episode_length: int = 288, pattern_pool=None,
                 w_slo: float = 1.0, w_cost: float = 0.6, w_churn: float = 0.1,
                 seed: int = None, render_mode: str = None):
        super().__init__()
        self.episode_length = episode_length
        # which patterns THIS env instance is allowed to sample from
        self.pattern_pool = pattern_pool if pattern_pool is not None else TRAIN_PATTERNS
        self.w_slo = w_slo
        self.w_cost = w_cost
        self.w_churn = w_churn
        self.render_mode = render_mode

        self.sim = KubernetesSimulator(min_pods=1, max_pods=MAX_PODS)
        self.workload_gen = WorkloadGenerator(base_rate=50.0, seed=seed)

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # [horizontal_move (7 options: -3..+3), vertical_tier (3 options)]
        self.action_space = spaces.MultiDiscrete([7, 3])

        self._step_idx = 0
        self._trace = None
        self._prev_rate = None
        self._prev_action = None
        self._tier_cooldown_remaining = 0
        self._down_cooldown_remaining = 0
        self._history = []  # for render()/logging

    def _sample_trace(self):
        pattern = np.random.choice(self.pattern_pool)
        trace = self.workload_gen.generate(pattern, self.episode_length)
        return pattern, trace

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self._step_idx = 0
        self._pattern_name, self._trace = self._sample_trace()
        self._prev_rate = self._trace[0]
        self._prev_action = (3, int(self.sim.state.tier))  # "hold, current tier" (3 = hold in new 7-option scheme)
        self._tier_cooldown_remaining = 0
        self._down_cooldown_remaining = 0
        self._history = []

        # take a zero-effect step just to populate an initial observation
        result = self.sim.step(request_rate=self._trace[0], horizontal_delta=0,
                                tier=self.sim.state.tier)
        obs = self._build_obs(result, trend=0.0)
        info = {"pattern": self._pattern_name}
        return obs, info

    def _build_obs(self, result, trend):
        latency_norm = min(result["latency_ms"], MAX_LATENCY_FOR_NORM) / MAX_LATENCY_FOR_NORM
        tier_cooldown_norm = self._tier_cooldown_remaining / VERTICAL_STABILIZATION_WINDOW
        down_cooldown_norm = self._down_cooldown_remaining / HORIZONTAL_DOWN_STABILIZATION_WINDOW
        obs = np.array([
            result["cpu_util"],
            result["mem_util"],
            latency_norm,
            result["active_pods"] / MAX_PODS,
            result["pending_pods"] / MAX_PODS,
            result["tier"] / 2.0,
            np.clip(trend, -5.0, 5.0),
            tier_cooldown_norm,
            down_cooldown_norm,
        ], dtype=np.float32)
        return obs

    def step(self, action):
        requested_horizontal, requested_tier = int(action[0]), int(action[1])
        requested_delta = requested_horizontal - 3
        prev_horizontal, prev_tier = self._prev_action

        # --- vertical (tier) stabilization window: SYMMETRIC, both directions ---
        # This is the fix that resolved the tier-oscillation root cause -
        # unchanged from the previous iteration.
        if self._tier_cooldown_remaining > 0:
            tier_choice = prev_tier
            self._tier_cooldown_remaining -= 1
        else:
            tier_choice = requested_tier
            if tier_choice != prev_tier:
                self._tier_cooldown_remaining = VERTICAL_STABILIZATION_WINDOW

        # --- horizontal stabilization window: ASYMMETRIC, mirrors real K8s HPA ---
        # scale-up is always immediate (no gating); scale-down is
        # cooldown-gated. This fixes the regression from the earlier shared
        # cooldown, which throttled legitimate scale-up during ramp/spike.
        if requested_delta < 0:
            if self._down_cooldown_remaining > 0:
                horizontal_choice = 3  # scale-down blocked - forced hold
                self._down_cooldown_remaining -= 1
            else:
                horizontal_choice = requested_horizontal
                self._down_cooldown_remaining = HORIZONTAL_DOWN_STABILIZATION_WINDOW
        else:
            horizontal_choice = requested_horizontal  # hold or scale-up: always allowed
            if self._down_cooldown_remaining > 0:
                self._down_cooldown_remaining -= 1  # still ticks down in the background

        horizontal_delta = horizontal_choice - 3  # 0->-3 ... 3->hold ... 6->+3
        tier = PodTier(tier_choice)

        request_rate = self._trace[self._step_idx]
        result = self.sim.step(request_rate=request_rate, horizontal_delta=horizontal_delta, tier=tier)

        # --- reward computation ---
        if result["slo_violation"]:
            over_ratio = (result["latency_ms"] - LATENCY_TARGET_MS) / LATENCY_TARGET_MS
            # capped at 3.0 (was 10.0) - a bad violation should hurt, but
            # shouldn't be so catastrophic that "never risk it" dominates
            # every other consideration regardless of cost
            slo_penalty = np.clip(over_ratio, 0.0, 3.0)
        else:
            slo_penalty = 0.0

        # normalized against a realistic reference load, not max theoretical
        # cost, so cost differences are actually visible in the reward signal
        cost_penalty = result["cost"] / REFERENCE_COST

        current_action = (horizontal_choice, tier_choice)
        churn_penalty = 1.0 if current_action != self._prev_action else 0.0
        self._prev_action = current_action

        reward = -(self.w_slo * slo_penalty) - (self.w_cost * cost_penalty) - (self.w_churn * churn_penalty)

        # --- bookkeeping ---
        trend = (request_rate - self._prev_rate) / max(self.workload_gen.base_rate, 1e-6)
        self._prev_rate = request_rate
        obs = self._build_obs(result, trend)

        self._step_idx += 1
        terminated = False
        truncated = self._step_idx >= self.episode_length

        info = dict(result)
        info["reward"] = reward
        info["pattern"] = self._pattern_name
        # diagnostic breakdown - lets us see WHICH reward term is driving
        # behavior, instead of only seeing the combined scalar reward.
        # This is what makes root-cause diagnosis possible instead of guessing.
        info["reward_slo_component"] = -(self.w_slo * slo_penalty)
        info["reward_cost_component"] = -(self.w_cost * cost_penalty)
        info["reward_churn_component"] = -(self.w_churn * churn_penalty)
        info["raw_slo_penalty"] = slo_penalty
        info["raw_cost_penalty"] = cost_penalty
        info["horizontal_action"] = horizontal_choice
        info["tier_action"] = tier_choice
        info["requested_horizontal"] = requested_horizontal
        info["requested_tier"] = requested_tier
        info["tier_cooldown_remaining"] = self._tier_cooldown_remaining
        info["down_cooldown_remaining"] = self._down_cooldown_remaining
        self._history.append(info)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if not self._history:
            return
        last = self._history[-1]
        print(f"[{self._pattern_name} | step {self._step_idx:>3}] "
              f"req={last['request_rate']:6.1f} pods={last['active_pods']:>2} "
              f"tier={last['tier']} cpu={last['cpu_util']*100:5.1f}% "
              f"lat={last['latency_ms']:7.1f}ms "
              f"{'SLO_VIOLATION' if last['slo_violation'] else '':>13} "
              f"cost={last['cost']:5.2f} reward={last['reward']:6.3f}")

    def get_episode_history(self):
        """Returns the full per-step log for the most recent episode (for plotting)."""
        return self._history


def make_train_env(**kwargs):
    kwargs.setdefault("pattern_pool", TRAIN_PATTERNS)
    return KubernetesAutoscalingEnv(**kwargs)


def make_eval_env(pattern: str, **kwargs):
    """Single-pattern eval env — pass any of TRAIN_PATTERNS + EVAL_ONLY_PATTERNS."""
    kwargs["pattern_pool"] = [pattern]
    return KubernetesAutoscalingEnv(**kwargs)


if __name__ == "__main__":
    # Basic sanity check with random actions - run this locally to confirm
    # the env doesn't crash and produces sane-looking numbers/shapes.
    env = make_train_env(episode_length=20, seed=0)
    obs, info = env.reset()
    print("Initial obs:", obs, "| pattern:", info["pattern"])
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    total_reward = 0.0
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"\nRan 20 random steps. Total reward: {total_reward:.3f}")
    print("Env sanity check passed if no errors were raised above.")