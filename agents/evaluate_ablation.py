"""
Evaluate all three reward ablation variants (slo_only, balanced, cost_priority)
across all traffic patterns, and produce the trade-off comparison that is
the actual point of this study: does reward weighting causally control the
cost-vs-reliability trade-off?

Usage (from project root, AFTER agents.train_ablation has completed):
    python -m agents.evaluate_ablation

Outputs:
    - results/ablation_comparison.csv
    - results/ablation_tradeoff.png   (the headline scatter plot: cost vs
      SLO violation %, one point per variant, per pattern)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from envs.k8s_autoscale_env import KubernetesAutoscalingEnv, TRAIN_PATTERNS, EVAL_ONLY_PATTERNS

ALL_PATTERNS = TRAIN_PATTERNS + EVAL_ONLY_PATTERNS
N_EPISODES = 8
EPISODE_LENGTH = 288
RESULTS_DIR = "results"

VARIANTS = {
    "slo_only":      {"path": "models/ablation_slo_only/best_model/best_model.zip",
                       "weights": {"w_slo": 1.0, "w_cost": 0.0, "w_churn": 0.0}, "color": "#F5A623"},
    "balanced":       {"path": "models/best_model/best_model.zip",
                       "weights": {"w_slo": 1.0, "w_cost": 0.6, "w_churn": 0.1}, "color": "#35D0BA"},
    "cost_priority":  {"path": "models/ablation_cost_priority/best_model/best_model.zip",
                       "weights": {"w_slo": 0.5, "w_cost": 1.2, "w_churn": 0.1}, "color": "#FF5A5F"},
}


def run_episode(env, model):
    obs, info = env.reset()
    terminated = truncated = False
    total_cost, violations, steps = 0.0, 0, 0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_cost += info["cost"]
        violations += int(info["slo_violation"])
        steps += 1
    return total_cost, 100 * violations / steps


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    for variant_name, cfg in VARIANTS.items():
        print(f"\nEvaluating variant: {variant_name}")
        if not os.path.exists(cfg["path"]):
            print(f"  SKIPPING - model not found at {cfg['path']}")
            continue
        model = PPO.load(cfg["path"])

        for pattern in ALL_PATTERNS:
            costs, violation_pcts = [], []
            for seed in range(N_EPISODES):
                env = KubernetesAutoscalingEnv(
                    episode_length=EPISODE_LENGTH, pattern_pool=[pattern], seed=seed, **cfg["weights"])
                cost, violation_pct = run_episode(env, model)
                costs.append(cost)
                violation_pcts.append(violation_pct)
            rows.append({
                "variant": variant_name, "pattern": pattern,
                "seen_during_training": pattern in TRAIN_PATTERNS,
                "avg_cost": np.mean(costs), "avg_slo_violation_pct": np.mean(violation_pcts),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "ablation_comparison.csv"), index=False)

    print("\n" + "=" * 90)
    print("REWARD ABLATION RESULTS (averaged across all 5 patterns per variant)")
    print("=" * 90)
    overall = df.groupby("variant").agg(avg_cost=("avg_cost", "mean"), avg_slo_violation_pct=("avg_slo_violation_pct", "mean"))
    print(overall.to_string())

    _plot_tradeoff(df)
    print(f"\nSaved: {RESULTS_DIR}/ablation_comparison.csv, ablation_tradeoff.png")


def _plot_tradeoff(df):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for variant_name, cfg in VARIANTS.items():
        sub = df[df.variant == variant_name]
        if sub.empty:
            continue
        ax.scatter(sub.avg_slo_violation_pct, sub.avg_cost, color=cfg["color"], s=70, alpha=0.75,
                   label=f"{variant_name}  (w_slo={cfg['weights']['w_slo']}, w_cost={cfg['weights']['w_cost']})")
        mean_x, mean_y = sub.avg_slo_violation_pct.mean(), sub.avg_cost.mean()
        ax.scatter([mean_x], [mean_y], color=cfg["color"], s=260, marker="*", edgecolor="black", linewidth=0.8, zorder=5)

    ax.set_xlabel("Avg SLO violation % (lower is better)")
    ax.set_ylabel("Avg episode cost (lower is better)")
    ax.set_title("Reward Weighting Controls the Cost-vs-Reliability Trade-off\n(small dots = per-pattern, star = variant average)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ablation_tradeoff.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
