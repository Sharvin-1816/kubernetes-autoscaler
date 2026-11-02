"""
Diagnostics: root-cause analysis tool for agent behavior.

Instead of only looking at headline metrics (cost, SLO%), this breaks down
WHY the agent is behaving the way it is:
    - which reward component (SLO / cost / churn) is dominating
    - the action distribution (is it actually varying actions, or stuck?)
    - whether pod count tracks traffic (correlation check)

Usage (from project root, after training):
    python -m agents.diagnose
"""

import numpy as np
from stable_baselines3 import PPO

import os
import sys

# Ensure project root is in sys.path so envs can be imported regardless of execution method
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.k8s_autoscale_env import make_eval_env

MODEL_PATH = "models/best_model/best_model.zip"


def run_and_diagnose(pattern: str, episode_length: int = 288, seed: int = 0):
    model = PPO.load(MODEL_PATH)
    env = make_eval_env(pattern=pattern, episode_length=episode_length, seed=seed)
    obs, info = env.reset()
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

    history = env.get_episode_history()

    print(f"\n{'=' * 70}\nDIAGNOSTICS: pattern='{pattern}'\n{'=' * 70}")

    # 1. Reward component breakdown - which term actually drives total reward?
    slo_total = sum(h["reward_slo_component"] for h in history)
    cost_total = sum(h["reward_cost_component"] for h in history)
    churn_total = sum(h["reward_churn_component"] for h in history)
    grand_total = slo_total + cost_total + churn_total
    print(f"\n[1] Reward component breakdown (sum over episode):")
    print(f"    SLO penalty total   : {slo_total:8.2f}  ({100*abs(slo_total)/max(abs(grand_total),1e-6):5.1f}% of total penalty)")
    print(f"    Cost penalty total  : {cost_total:8.2f}  ({100*abs(cost_total)/max(abs(grand_total),1e-6):5.1f}% of total penalty)")
    print(f"    Churn penalty total : {churn_total:8.2f}  ({100*abs(churn_total)/max(abs(grand_total),1e-6):5.1f}% of total penalty)")
    print(f"    -> If one component is >80-90% of the total, that's what's driving")
    print(f"       the policy's behavior almost exclusively.")

    # 2. Action distribution - is the agent actually varying its decisions?
    horiz_actions = [h["horizontal_action"] for h in history]
    tier_actions = [h["tier_action"] for h in history]
    # 7-option horizontal scheme: 0->-3, 1->-2, 2->-1, 3->hold, 4->+1, 5->+2, 6->+3
    horiz_labels = {0: "-3", 1: "-2", 2: "-1", 3: "hold", 4: "+1", 5: "+2", 6: "+3"}
    horiz_counts = {a: horiz_actions.count(a) for a in range(7)}
    tier_counts = {t: tier_actions.count(t) for t in [0, 1, 2]}
    print(f"\n[2] Action distribution over {len(history)} steps:")
    horiz_str = "  ".join(f"{horiz_labels[a]}:{horiz_counts[a]:3d}" for a in range(7))
    print(f"    Horizontal -> {horiz_str}   (sum={sum(horiz_counts.values())}, should equal {len(history)})")
    print(f"    Tier       -> small: {tier_counts[0]:3d}  medium: {tier_counts[1]:3d}  large: {tier_counts[2]:3d}")
    print(f"    -> If one action dominates >90% of steps, the agent has collapsed")
    print(f"       to a near-constant policy rather than a traffic-responsive one.")

    # 2c. Requested vs actual actions.
    # requested_* comes directly from the PPO output; *_action is what the
    # environment actually applied after cooldown/stabilization constraints.
    requested_h = [h.get("requested_horizontal", 3) for h in history]
    actual_h = [h["horizontal_action"] for h in history]
    requested_t = [h.get("requested_tier", h["tier_action"]) for h in history]
    actual_t = [h["tier_action"] for h in history]

    h_blocked = sum(r != a for r, a in zip(requested_h, actual_h))
    t_blocked = sum(r != a for r, a in zip(requested_t, actual_t))

    print("\n[2c] Requested vs actual actions:")
    print(f"    Horizontal overrides : {h_blocked:3d} / {len(history)} "
          f"({100*h_blocked/max(len(history),1):5.1f}%)")
    print(f"    Tier overrides       : {t_blocked:3d} / {len(history)} "
          f"({100*t_blocked/max(len(history),1):5.1f}%)")

    # Requested/actual horizontal distribution.
    req_h_counts = {a: requested_h.count(a) for a in range(7)}
    act_h_counts = {a: actual_h.count(a) for a in range(7)}
    print("    Horizontal requested -> " +
          "  ".join(f"{horiz_labels[a]}:{req_h_counts[a]:3d}" for a in range(7)))
    print("    Horizontal actual    -> " +
          "  ".join(f"{horiz_labels[a]}:{act_h_counts[a]:3d}" for a in range(7)))

    # Requested/actual tier distribution.
    req_t_counts = {t: requested_t.count(t) for t in [0, 1, 2]}
    act_t_counts = {t: actual_t.count(t) for t in [0, 1, 2]}
    print(f"    Tier requested -> small: {req_t_counts[0]:3d}  medium: {req_t_counts[1]:3d}  large: {req_t_counts[2]:3d}")
    print(f"    Tier actual    -> small: {act_t_counts[0]:3d}  medium: {act_t_counts[1]:3d}  large: {act_t_counts[2]:3d}")

    # Scale-down cooldown specifically.
    blocked_downs = sum(
        1 for h in history
        if h.get("requested_horizontal", 3) < 3 and h["horizontal_action"] == 3
    )
    requested_downs = sum(
        1 for h in history if h.get("requested_horizontal", 3) < 3
    )
    print(f"    Scale-down cooldown blocks: {blocked_downs} / {requested_downs} requested")

    # 2d. What happens to SLO immediately after an applied scaling action?
    scaling_events = []
    for i, h in enumerate(history):
        if h["horizontal_action"] != 3 or (
            i > 0 and h["tier_action"] != history[i - 1]["tier_action"]
        ):
            before_slo = history[i - 1]["latency_ms"] if i > 0 else h["latency_ms"]
            after_slo = h["latency_ms"]
            scaling_events.append((before_slo, after_slo, h.get("reward", 0.0)))

    print("\n[2d] Scaling-event response:")
    if scaling_events:
        improved = sum(after < before for before, after, _ in scaling_events)
        avg_delta = np.mean([after - before for before, after, _ in scaling_events])
        print(f"    Scaling events         : {len(scaling_events)}")
        print(f"    Improved latency       : {improved}/{len(scaling_events)}")
        print(f"    Mean immediate latency change: {avg_delta:+.2f} ms")
    else:
        print("    No scaling events detected.")

    # 2b. Timing of tier switches relative to peak traffic - are switches
    # clustered right at the peak (potential cause of latency oscillation)?
    pods = np.array([h["active_pods"] for h in history])
    traffic = np.array([h["request_rate"] for h in history])
    tier_switch_steps = [i for i in range(1, len(tier_actions)) if tier_actions[i] != tier_actions[i - 1]]
    peak_step = int(np.argmax(traffic))
    print(f"\n[2b] Tier switch timing: {len(tier_switch_steps)} switches at steps {tier_switch_steps}")
    print(f"     Traffic peak occurs at step {peak_step}")
    if tier_switch_steps:
        near_peak = [s for s in tier_switch_steps if abs(s - peak_step) <= 15]
        print(f"     Switches within 15 steps of peak: {len(near_peak)} / {len(tier_switch_steps)}")
        print(f"     -> If most switches cluster near the peak, tier-flapping under high")
        print(f"        load is a likely contributor to latency oscillation/spikes there.")

    # 3. Does pod count actually correlate with traffic?
    if pods.std() > 0 and traffic.std() > 0:
        correlation = np.corrcoef(pods, traffic)[0, 1]
    else:
        correlation = float("nan")
    print(f"\n[3] Correlation between pod count and traffic: {correlation:.3f}")
    print(f"    -> Close to 0 or negative means pods are NOT tracking load")
    print(f"       (e.g. agent maxed out immediately and ignored traffic changes).")
    print(f"    -> Meaningfully positive (>0.4-0.5) suggests genuine load-responsive scaling.")

    # 4. Cost variability check (was the earlier symptom: near-identical cost everywhere)
    costs = np.array([h["cost"] for h in history])
    print(f"\n[4] Cost stats this episode: mean={costs.mean():.2f}, std={costs.std():.2f}, "
          f"min={costs.min():.2f}, max={costs.max():.2f}")
    print(f"    -> std near 0 means cost never changes -> pod count is static -> not responsive.")

    return history


if __name__ == "__main__":
    for pattern in ["diurnal", "noisy_steady", "spike", "ramp", "mixed"]:
        run_and_diagnose(pattern)