# Formal MDP Specification

## Kubernetes Autoscaling as a Markov Decision Process

This document formally defines the environment as an MDP: (S, A, P, R, γ).
Use this directly in your report's "Methodology" section.

---

## 1. State Space (S)

A 7-dimensional continuous observation vector:

| Index | Variable          | Range        | Description                                                              |
|-------|--------------------|--------------|----------------------------------------------------------------------------|
| 0     | cpu_util           | [0.0, 1.5]   | CPU utilization = request_rate / total_cluster_capacity                    |
| 1     | mem_util           | [0.0, 1.5]   | Memory utilization (correlated with CPU + noise)                           |
| 2     | latency_norm       | [0.0, 1.0]   | min(latency_ms, 2000) / 2000 — normalized request latency                  |
| 3     | active_pods_norm   | [0.0, 1.0]   | active_pods / MAX_PODS                                                     |
| 4     | pending_pods_norm  | [0.0, 1.0]   | pods currently spinning up (not yet serving) / MAX_PODS                    |
| 5     | tier_norm          | [0.0, 1.0]   | current resource tier / 2 (0=SMALL, 0.5=MEDIUM, 1.0=LARGE)                 |
| 6     | traffic_trend      | [-5.0, 5.0]  | (request_rate_t − request_rate_{t-1}) / base_rate — rising/falling signal  |

**Design rationale:** Dimensions 0-2 mirror the Prometheus-style metrics used
in the reference paper (Pandey, 2026). Dimensions 3-5 give the agent
awareness of its own current cluster configuration (necessary since actions
are *deltas*, not absolute targets). Dimension 6 (traffic trend) is included
specifically because of the pod spin-up delay — without a trend signal, a
purely reactive policy cannot anticipate load and will systematically lag
during fast-rising traffic (ramp/spike patterns).

## 2. Action Space (A)

MultiDiscrete([7, 3]) — two independent decisions per timestep:

**Horizontal (pod count) decision** — 7 discrete choices, mapped to pod deltas:
`{-3, -2, -1, 0 (hold), +1, +2, +3}`

**Vertical (resource tier) decision** — 3 discrete choices:
`{SMALL, MEDIUM, LARGE}`

**Design rationale:** Multi-pod deltas (not just ±1) mirror real Kubernetes
HPA, which computes a target replica count directly rather than incrementing
by exactly one pod per adjustment cycle. This matters because with a
spin-up delay, the *rate* at which capacity can grow — not just the eventual
ceiling — determines whether the cluster can keep pace with fast-rising
traffic (see Section 5, "Known environment dynamics").

## 3. Transition Dynamics (P)

Deterministic simulation with one stochastic element (memory utilization
noise). Given state s_t and action a_t:

1. Apply horizontal delta (bounded by [1, MAX_PODS]); newly added pods enter
   a `SPINUP_DELAY`-step pending queue before becoming active.
2. Apply vertical tier change (instant — no delay modeled for tier changes,
   only for horizontal pod count, matching the assumption that resizing
   existing pod resource requests is faster than provisioning new ones).
3. Advance any already-pending pods by one step; activate any that reach 0
   steps remaining.
4. Compute cpu_util = request_rate_t / total_capacity(active_pods, tier).
5. Compute latency via M/M/c-inspired queuing approximation:
   `latency = BASE_SERVICE_TIME / (1 - ρ)^2` where ρ = min(cpu_util, 0.999).
6. Compute cost = (active_pods + pending_pods) × tier_cost_per_pod.

## 4. Reward Function (R)

```
R(s_t, a_t) = -(w_slo × slo_penalty) - (w_cost × cost_penalty) - (w_churn × churn_penalty)
```

Where:
- `slo_penalty = clip((latency_ms - 150) / 150, 0, 3.0)` if latency exceeds
  the 150ms SLO target, else 0. Capped at 3.0 so a violation is costly but
  not so catastrophic that it dominates all other considerations regardless
  of cost (see reward-hacking diagnosis below).
- `cost_penalty = cost_this_step / REFERENCE_COST`, where REFERENCE_COST =
  5 × MEDIUM_tier_cost, i.e. normalized against a realistic operating cost,
  not the theoretical maximum, so cost differences remain visible to the
  agent under normal load.
- `churn_penalty = 1.0` if the action differs from the previous timestep's
  action (either dimension), else 0. Discourages scaling flap/thrash.

**Default weights:** w_slo = 1.0, w_cost = 0.6, w_churn = 0.1

**Note on iteration history (include this in your report — it's a
legitimate part of the methodology, not a footnote):** An earlier reward
formulation (w_cost = 0.3, slo_penalty uncapped to 10.0, cost normalized
against theoretical max capacity) produced a degenerate policy that always
scaled to maximum pod count regardless of traffic, because avoiding any SLO
violation was worth ~100x more than any achievable cost saving under that
formulation. This was diagnosed via: (a) near-identical cost across all
traffic patterns despite very different loads, (b) scale-event counts
exactly matching the one-time ramp from initial to max pod count, (c)
near-zero pod-count/traffic correlation. The reward function above was
redesigned specifically to fix this mechanism — not by artificially
shrinking MAX_PODS, but by making cost and latency penalties comparable in
magnitude so the agent must make a genuine trade-off.

## 5. Episode Structure

- One episode = one full traffic trace of `episode_length` = 288 steps
  (≈ 1 simulated day at 5-minute resolution).
- Initial state: 3 active pods, MEDIUM tier, no pending pods.
- Episode terminates only by truncation at step 288 (no early-termination
  condition — an outage/SLO violation does not end the episode, since
  recovering from a bad state is itself part of what the agent should learn).

## 6. Train / Evaluation Pattern Split

To support a generalization study (does the learned policy transfer to
unseen traffic shapes, or does it overfit to training patterns?):

- **Training patterns:** `diurnal`, `noisy_steady`
- **Held-out evaluation-only patterns:** `spike`, `ramp`, `mixed`

The agent never observes spike/ramp/mixed traffic shapes during training.

## 7. Known Environment Dynamics Worth Discussing in Report

- **Spin-up delay (2 steps)** creates an inherent anticipation requirement —
  a purely reactive agent will systematically lag during fast-rising load.
- **Tier economics** (capacity-per-dollar: SMALL=15.0, MEDIUM=13.6,
  LARGE=12.2) make SMALL tier strictly cost-dominant under the current cost
  model, absent any fixed per-pod overhead. This is a second identified
  mechanism (separate from the reward-weight issue above) under active
  investigation as of the current training iteration — see diagnostics log.
