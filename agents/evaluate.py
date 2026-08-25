"""
Evaluate the trained PPO agent against the ThresholdHPAAgent baseline,
across ALL patterns - both the ones PPO trained on (diurnal, noisy_steady)
and the ones it never saw during training (spike, ramp, mixed).

This is the core experiment of the project: does the learned policy
generalize to unseen traffic shapes, or does it overfit to what it trained on?

Usage (from project root):
    python -m agents.evaluate

Outputs:
    - results/comparison_table.csv       (full numeric results)
    - results/comparison_barplot.png     (SLO violations + cost, per pattern/agent)
    - results/episode_trace_<pattern>.png  (pod count / latency over time, for one
                                            spike-pattern episode, PPO vs baseline)
    - Printed summary table in the console
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from envs.k8s_autoscale_env import make_eval_env, TRAIN_PATTERNS, EVAL_ONLY_PATTERNS
from agents.baseline_hpa import ThresholdHPAAgent, run_episode

MODEL_PATH = "models/best_model/best_model.zip"
ALL_PATTERNS = TRAIN_PATTERNS + EVAL_ONLY_PATTERNS
N_EPISODES_PER_PATTERN = 10   # average over multiple seeds for stable numbers
EPISODE_LENGTH = 288
RESULTS_DIR = "results"


def run_ppo_episode(env, model):
    obs, info = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
    return env.get_episode_history()


def summarize_history(history, pattern, agent_name, seed):
    total_cost = sum(h["cost"] for h in history)
    slo_violations = sum(1 for h in history if h["slo_violation"])
    avg_latency = np.mean([h["latency_ms"] for h in history])
    p95_latency = np.percentile([h["latency_ms"] for h in history], 95)
    avg_pods = np.mean([h["active_pods"] for h in history])

    # stability: how many times the horizontal action actually changed pod count
    pod_counts = [h["active_pods"] for h in history]
    scale_events = sum(1 for i in range(1, len(pod_counts)) if pod_counts[i] != pod_counts[i - 1])

    return {
        "pattern": pattern,
        "agent": agent_name,
        "seed": seed,
        "total_cost": total_cost,
        "slo_violations": slo_violations,
        "slo_violation_pct": 100 * slo_violations / len(history),
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "avg_pods": avg_pods,
        "scale_events": scale_events,
        "seen_during_training": pattern in TRAIN_PATTERNS,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading PPO model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)

    rows = []
    sample_traces = {}  # store one full trace per (pattern, agent) for plotting

    for pattern in ALL_PATTERNS:
        tag = "TRAIN" if pattern in TRAIN_PATTERNS else "HELD-OUT"
        print(f"\nEvaluating pattern: {pattern} [{tag}]")

        for seed in range(N_EPISODES_PER_PATTERN):
            # --- PPO ---
            env = make_eval_env(pattern=pattern, episode_length=EPISODE_LENGTH, seed=seed)
            history = run_ppo_episode(env, model)
            rows.append(summarize_history(history, pattern, "PPO", seed))
            if seed == 0:
                sample_traces[(pattern, "PPO")] = history

            # --- Baseline HPA ---
            env = make_eval_env(pattern=pattern, episode_length=EPISODE_LENGTH, seed=seed)
            agent = ThresholdHPAAgent()
            history = run_episode(env, agent)
            rows.append(summarize_history(history, pattern, "HPA_Baseline", seed))
            if seed == 0:
                sample_traces[(pattern, "HPA_Baseline")] = history

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "raw_results.csv"), index=False)

    # aggregate across seeds -> mean +/- std per pattern/agent
    summary = df.groupby(["pattern", "agent", "seen_during_training"]).agg(
        avg_cost=("total_cost", "mean"),
        std_cost=("total_cost", "std"),
        avg_slo_violation_pct=("slo_violation_pct", "mean"),
        std_slo_violation_pct=("slo_violation_pct", "std"),
        avg_latency_ms=("avg_latency_ms", "mean"),
        avg_p95_latency_ms=("p95_latency_ms", "mean"),
        avg_scale_events=("scale_events", "mean"),
    ).reset_index()

    summary.to_csv(os.path.join(RESULTS_DIR, "comparison_table.csv"), index=False)

    print("\n" + "=" * 100)
    print("SUMMARY (averaged over {} episodes per pattern/agent)".format(N_EPISODES_PER_PATTERN))
    print("=" * 100)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", None)
    print(summary.to_string(index=False))

    _plot_comparison(summary)
    _plot_sample_traces(sample_traces)

    print(f"\nSaved: {RESULTS_DIR}/comparison_table.csv, raw_results.csv, "
          f"comparison_barplot.png, episode_trace_*.png")


def _plot_comparison(summary):
    patterns = ALL_PATTERNS
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(patterns))
    width = 0.35

    for ax, metric, title in [
        (axes[0], "avg_slo_violation_pct", "SLO Violation % (lower is better)"),
        (axes[1], "avg_cost", "Total Cost per Episode (lower is better)"),
    ]:
        ppo_vals = [summary[(summary.pattern == p) & (summary.agent == "PPO")][metric].values[0]
                    for p in patterns]
        hpa_vals = [summary[(summary.pattern == p) & (summary.agent == "HPA_Baseline")][metric].values[0]
                    for p in patterns]

        ax.bar(x - width / 2, ppo_vals, width, label="PPO (RL)", color="#2E86AB")
        ax.bar(x + width / 2, hpa_vals, width, label="HPA Baseline", color="#A23B72")
        ax.set_xticks(x)
        labels = [f"{p}\n(seen)" if p in TRAIN_PATTERNS else f"{p}\n(unseen)" for p in patterns]
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.legend()
        ax.axvline(len(TRAIN_PATTERNS) - 0.5, color="gray", linestyle="--", alpha=0.5)

    plt.suptitle("PPO vs HPA Baseline — Trained Patterns vs Held-Out (Generalization Test)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison_barplot.png"), dpi=130)
    plt.close()


def _plot_sample_traces(sample_traces):
    # Focus on 'spike' since it's the clearest visual story (held-out pattern)
    pattern = "spike"
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for agent_name, color in [("PPO", "#2E86AB"), ("HPA_Baseline", "#A23B72")]:
        history = sample_traces.get((pattern, agent_name))
        if not history:
            continue
        steps = range(len(history))
        req_rates = [h["request_rate"] for h in history]
        pods = [h["active_pods"] for h in history]
        latency = [h["latency_ms"] for h in history]

        axes[0].plot(steps, pods, label=f"{agent_name} pods", color=color)
        axes[1].plot(steps, latency, label=f"{agent_name} latency", color=color)

    ax2 = axes[0].twinx()
    ax2.plot(range(len(req_rates)), req_rates, color="gray", alpha=0.4, linestyle=":", label="traffic (req/s)")
    ax2.set_ylabel("request rate")

    axes[0].set_ylabel("active pods")
    axes[0].set_title(f"Pod count over time — {pattern} pattern (held-out, unseen during training)")
    axes[0].legend(loc="upper left")

    axes[1].axhline(150, color="red", linestyle="--", alpha=0.6, label="SLO target")
    axes[1].set_ylabel("latency (ms)")
    axes[1].set_xlabel("timestep")
    axes[1].set_title("Latency over time")
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"episode_trace_{pattern}.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
