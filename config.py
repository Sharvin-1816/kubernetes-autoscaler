"""
Central configuration for the entire project.

Why this file exists: constants were previously scattered across
k8s_simulator.py, k8s_autoscale_env.py, and train.py. Centralizing them here:
    1. Makes every "why did we pick this number" traceable to one place
    2. Is a prerequisite for clean hyperparameter tuning (Optuna) later -
       a sweep needs a single source of truth to override, not five files
    3. Matches how real ML/RL projects are structured (config-driven, not
       magic numbers scattered through logic files)

Nothing in here changes behavior yet - this is a refactor, not a fix.
Once this is wired in (replacing the scattered constants), any future
change goes through this file only.
"""

from dataclasses import dataclass
from enum import IntEnum


class PodTier(IntEnum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2


@dataclass(frozen=True)
class TierSpec:
    capacity: float   # requests/sec a single pod of this tier can serve
    cost: float        # $ per timestep for one running pod of this tier


TIER_SPECS = {
    PodTier.SMALL:  TierSpec(capacity=15.0, cost=1.0),
    PodTier.MEDIUM: TierSpec(capacity=30.0, cost=2.2),
    PodTier.LARGE:  TierSpec(capacity=55.0, cost=4.5),
}


@dataclass(frozen=True)
class SimulatorConfig:
    spinup_delay_steps: int = 2
    latency_target_ms: float = 150.0
    base_service_time_ms: float = 40.0
    queuing_exponent_k: float = 2.0   # controls how sharply latency grows near saturation
    min_pods: int = 1
    # MAX_PODS derived from peak-capacity analysis (see MDP_SPECIFICATION.md
    # Section 7): worst-case traffic peak ~200 req/s, MEDIUM-tier minimum
    # pods to cover that = ceil(200/30) = 7, doubled for operational headroom.
    max_pods: int = 15


@dataclass(frozen=True)
class RewardConfig:
    w_slo: float = 1.0
    w_cost: float = 0.6
    w_churn: float = 0.1
    slo_penalty_cap: float = 3.0       # see MDP_SPECIFICATION.md for why 3.0, not 10.0
    reference_cost_pods: int = 5        # cost normalized against N medium pods' cost
    reference_tier: PodTier = PodTier.MEDIUM

    @property
    def reference_cost(self) -> float:
        return self.reference_tier_cost() * self.reference_cost_pods

    def reference_tier_cost(self) -> float:
        return TIER_SPECS[self.reference_tier].cost


@dataclass(frozen=True)
class EnvConfig:
    episode_length: int = 288          # ~1 simulated day at 5-min resolution
    base_request_rate: float = 50.0
    max_latency_for_norm: float = 2000.0
    horizontal_action_range: int = 3    # allows deltas of -3..+3 pods per step
    train_patterns: tuple = ("diurnal", "noisy_steady")
    eval_only_patterns: tuple = ("spike", "ramp", "mixed")


@dataclass(frozen=True)
class PPOConfig:
    total_timesteps: int = 200_000
    n_envs: int = 4
    learning_rate: float = 3e-4
    n_steps: int = 512          # 2048 // n_envs
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01


@dataclass(frozen=True)
class BaselineHPAConfig:
    scale_up_thresh: float = 0.75
    scale_down_thresh: float = 0.30
    window: int = 3


# Single importable instance of each - use these everywhere instead of
# re-declaring constants locally.
SIM_CONFIG = SimulatorConfig()
REWARD_CONFIG = RewardConfig()
ENV_CONFIG = EnvConfig()
PPO_CONFIG = PPOConfig()
BASELINE_CONFIG = BaselineHPAConfig()


if __name__ == "__main__":
    # Quick sanity print - run this locally to confirm values look right
    print("=== Tier economics (capacity per dollar) ===")
    for tier, spec in TIER_SPECS.items():
        print(f"  {tier.name:7s}: capacity={spec.capacity:5.1f}  cost={spec.cost:4.1f}  "
              f"capacity/$={spec.capacity/spec.cost:.2f}")

    print(f"\n=== Reward reference cost ===")
    print(f"  REFERENCE_COST = {REWARD_CONFIG.reference_cost:.2f}")

    print(f"\n=== Max pods (capacity check) ===")
    peak_traffic = 200.0
    for tier, spec in TIER_SPECS.items():
        needed = peak_traffic / spec.capacity
        print(f"  {tier.name:7s}: pods needed for peak={peak_traffic} -> {needed:.1f} "
              f"(MAX_PODS={SIM_CONFIG.max_pods} gives {SIM_CONFIG.max_pods/needed:.1f}x headroom)")
