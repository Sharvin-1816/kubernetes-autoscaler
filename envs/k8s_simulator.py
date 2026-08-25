"""
Kubernetes Cluster Simulator
-----------------------------
This is the "physics engine" of the project. Given:
    - current number of pods
    - current pod resource tier (vertical scaling choice)
    - incoming request rate

...it computes:
    - CPU utilization %
    - Memory utilization %
    - Request latency (via an M/M/c queuing approximation)
    - Running cost ($/timestep)

Design choices (documented for the report):
    - Latency modeled via M/M/c queuing theory: latency stays low until
      utilization approaches capacity, then rises steeply/non-linearly.
      This is far more realistic than a linear latency model and is what
      makes the RL problem actually interesting (an agent that waits too
      long to scale gets punished hard, non-linearly).
    - Pod spin-up delay: newly requested pods take `SPINUP_DELAY` steps
      before they start serving traffic (mirrors real K8s scheduling +
      container start time). This forces the agent to anticipate load
      rather than purely react to it.
    - Vertical scaling: 3 resource tiers per pod (small/medium/large),
      each with different capacity (requests/sec it can serve) and cost.
"""

from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np


class PodTier(IntEnum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2


# Capacity = max requests/sec a single pod of this tier can comfortably serve
# Cost = $ per timestep for one running pod of this tier (compute/resource cost only)
TIER_SPECS = {
    PodTier.SMALL:  {"capacity": 15.0, "cost": 1.0},
    PodTier.MEDIUM: {"capacity": 30.0, "cost": 2.2},
    PodTier.LARGE:  {"capacity": 55.0, "cost": 4.5},
}

# FIXED_POD_OVERHEAD: a per-pod cost that applies regardless of tier -
# models real Kubernetes overhead every pod carries independent of size
# (control-plane bookkeeping/scheduling, sidecar containers, base container
# runtime + OS footprint, network policy enforcement). This was MISSING
# before, which is why capacity-per-dollar strictly favored SMALL
# (15.0 > 13.6 > 12.2 for SMALL/MEDIUM/LARGE) with no fixed cost included.
#
# With FIXED_POD_OVERHEAD = 0.3, effective capacity-per-dollar becomes:
#   SMALL:  15.0 / (1.0+0.3) = 11.54
#   MEDIUM: 30.0 / (2.2+0.3) = 12.00   <- now the most cost-efficient tier
#   LARGE:  55.0 / (4.5+0.3) = 11.46
# This matches real cloud economics, where fixed per-instance overhead
# means mid-size instances are often the cost-efficiency "sweet spot" -
# neither smallest nor largest wins outright. No tier is now trivially
# dominant, so the agent has a genuine incentive to use MEDIUM/LARGE when
# load justifies it, rather than a rational reason to always pick SMALL.
FIXED_POD_OVERHEAD = 0.3

SPINUP_DELAY = 2          # timesteps before a newly added pod starts serving
LATENCY_TARGET_MS = 150.0  # SLO target: requests should complete under this
BASE_SERVICE_TIME_MS = 40.0  # baseline service time per request at low load


@dataclass
class ClusterState:
    """Tracks everything about the current simulated cluster."""
    active_pods: int = 3
    tier: PodTier = PodTier.MEDIUM
    # queue of (pods_pending, steps_remaining) for pods still spinning up
    pending_pods: list = field(default_factory=list)

    def total_capacity(self) -> float:
        """Total requests/sec the cluster can currently serve (active pods only)."""
        return self.active_pods * TIER_SPECS[self.tier]["capacity"]

    def total_cost(self) -> float:
        """Cost per timestep for active + pending (already provisioned) pods.
        Includes FIXED_POD_OVERHEAD per pod on top of tier compute cost -
        see FIXED_POD_OVERHEAD comment above for why this exists."""
        pending_count = sum(p[0] for p in self.pending_pods)
        total_pods = self.active_pods + pending_count
        return total_pods * (TIER_SPECS[self.tier]["cost"] + FIXED_POD_OVERHEAD)


class KubernetesSimulator:
    """
    Simulates one Kubernetes-like service cluster over discrete timesteps.
    This class is intentionally independent of Gym/RL so it can be tested
    and reused on its own.
    """

    def __init__(self, min_pods: int = 1, max_pods: int = 20):
        self.min_pods = min_pods
        self.max_pods = max_pods
        self.state = ClusterState(active_pods=3, tier=PodTier.MEDIUM)

    def reset(self):
        self.state = ClusterState(active_pods=3, tier=PodTier.MEDIUM)
        return self.state

    def _advance_pending_pods(self):
        """Move pending pods closer to activation; activate any that are ready."""
        still_pending = []
        for count, steps_left in self.state.pending_pods:
            steps_left -= 1
            if steps_left <= 0:
                self.state.active_pods += count
            else:
                still_pending.append((count, steps_left))
        self.state.pending_pods = still_pending

    def apply_action(self, horizontal_delta: int, tier: PodTier):
        """
        horizontal_delta : pods to add (+) or remove (-) this step
        tier              : desired resource tier (vertical scaling choice)

        Removing pods is instant (draining is out of scope for this sim).
        Adding pods incurs SPINUP_DELAY before they start serving.
        """
        self.state.tier = tier

        if horizontal_delta > 0:
            new_total = self.state.active_pods + sum(p[0] for p in self.state.pending_pods) + horizontal_delta
            new_total = min(new_total, self.max_pods)
            actual_add = new_total - (self.state.active_pods + sum(p[0] for p in self.state.pending_pods))
            if actual_add > 0:
                self.state.pending_pods.append((actual_add, SPINUP_DELAY))

        elif horizontal_delta < 0:
            remove = min(-horizontal_delta, self.state.active_pods - self.min_pods)
            self.state.active_pods -= max(remove, 0)

    def _compute_latency_ms(self, request_rate: float) -> float:
        """
        M/M/c-inspired latency approximation.
        rho = utilization (arrival rate / total capacity)
        As rho -> 1, latency blows up (queuing effect). We use a standard
        Erlang-C style approximation, simplified for tractability:

            latency = base_service_time / (1 - rho)^k   for rho < 1
            latency = very large (saturation)            for rho >= 1
        """
        capacity = self.state.total_capacity()
        if capacity <= 0:
            return 5000.0  # no capacity at all -> effectively down

        rho = request_rate / capacity
        rho = min(rho, 0.999)  # avoid divide-by-zero

        if rho >= 0.999:
            return 5000.0

        # k controls how sharply latency grows near saturation
        k = 2.0
        latency = BASE_SERVICE_TIME_MS / ((1 - rho) ** k)
        return float(np.clip(latency, BASE_SERVICE_TIME_MS, 5000.0))

    def step(self, request_rate: float, horizontal_delta: int, tier: PodTier):
        """
        Advances the simulation by one timestep.

        Returns a dict of observable/measurable quantities:
            cpu_util, mem_util, latency_ms, active_pods, cost, slo_violation
        """
        self.apply_action(horizontal_delta, tier)
        self._advance_pending_pods()

        capacity = self.state.total_capacity()
        cpu_util = float(np.clip(request_rate / capacity, 0.0, 1.5)) if capacity > 0 else 1.5
        # memory util correlates with cpu but with its own noise/offset
        mem_util = float(np.clip(cpu_util * 0.8 + 0.1 + np.random.normal(0, 0.03), 0.0, 1.5))

        latency_ms = self._compute_latency_ms(request_rate)
        slo_violation = latency_ms > LATENCY_TARGET_MS
        cost = self.state.total_cost()

        return {
            "cpu_util": cpu_util,
            "mem_util": mem_util,
            "latency_ms": latency_ms,
            "active_pods": self.state.active_pods,
            "pending_pods": sum(p[0] for p in self.state.pending_pods),
            "tier": int(self.state.tier),
            "cost": cost,
            "slo_violation": slo_violation,
            "request_rate": request_rate,
        }


if __name__ == "__main__":
    # Quick sanity check: fixed pod count, ramp up traffic, watch latency curve
    import matplotlib.pyplot as plt
    from workload_generator import WorkloadGenerator  # noqa

    sim = KubernetesSimulator()
    sim.reset()
    sim.state.active_pods = 5
    sim.state.tier = PodTier.MEDIUM

    rates = np.linspace(5, 200, 100)
    latencies = []
    for r in rates:
        result = sim.step(request_rate=r, horizontal_delta=0, tier=PodTier.MEDIUM)
        latencies.append(result["latency_ms"])

    plt.figure(figsize=(8, 4))
    plt.plot(rates, latencies)
    plt.axhline(LATENCY_TARGET_MS, color="red", linestyle="--", label="SLO target")
    plt.xlabel("request rate (req/sec)")
    plt.ylabel("latency (ms)")
    plt.title("Latency vs load, fixed at 5 MEDIUM pods (capacity=150 req/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/claude/k8s-rl-autoscaler/results/latency_curve_sanity_check.png", dpi=120)
    print("Saved sanity-check plot to results/latency_curve_sanity_check.png")