"""
Final analysis: polished summary visuals from the results you already have.
No retraining, no new environment runs - reads results/raw_results.csv
(already produced by agents/evaluate.py) and produces:

    1. results/final_improvement_summary.png
       - % improvement of PPO over HPA baseline, cost and SLO violation %,
         per pattern, with seen/unseen patterns visually separated.
    2. results/final_summary_table.csv
       - clean, presentation-ready numbers (rounded, renamed columns).

Usage (from project root, AFTER you've already run agents.evaluate at least
once so results/raw_results.csv exists):
    python -m agents.final_analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from envs.k8s_autoscale_env import TRAIN_PATTERNS, EVAL_ONLY_PATTERNS

RESULTS_DIR = "results"
RAW_CSV = os.path.join(RESULTS_DIR, "raw_results.csv")
ALL_PATTERNS = TRAIN_PATTERNS + EVAL_ONLY_PATTERNS


def main():
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"{RAW_CSV} not found. Run `python -m agents.evaluate` first to "
            f"generate it, then run this script."
        )

    df = pd.read_csv(RAW_CSV)

    summary = df.groupby(["pattern", "agent"]).agg(
        avg_cost=("total_cost", "mean"),
        avg_slo_violation_pct=("slo_violation_pct", "mean"),
    ).reset_index()

    rows = []
    for pattern in ALL_PATTERNS:
        ppo_row = summary[(summary.pattern == pattern) & (summary.agent == "PPO")].iloc[0]
        hpa_row = summary[(summary.pattern == pattern) & (summary.agent == "HPA_Baseline")].iloc[0]

        cost_improvement_pct = 100 * (hpa_row.avg_cost - ppo_row.avg_cost) / hpa_row.avg_cost
        slo_improvement_pct = 100 * (hpa_row.avg_slo_violation_pct - ppo_row.avg_slo_violation_pct) / max(hpa_row.avg_slo_violation_pct, 1e-6)

        rows.append({
            "pattern": pattern,
            "seen_during_training": pattern in TRAIN_PATTERNS,
            "ppo_cost": round(ppo_row.avg_cost, 1),
            "hpa_cost": round(hpa_row.avg_cost, 1),
            "cost_improvement_pct": round(cost_improvement_pct, 1),
            "ppo_slo_violation_pct": round(ppo_row.avg_slo_violation_pct, 1),
            "hpa_slo_violation_pct": round(hpa_row.avg_slo_violation_pct, 1),
            "slo_improvement_pct": round(slo_improvement_pct, 1),
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(RESULTS_DIR, "final_summary_table.csv"), index=False)

    print("=" * 90)
    print("FINAL SUMMARY: PPO improvement over HPA baseline (positive = PPO better)")
    print("=" * 90)
    pd.set_option("display.width", 140)
    print(result_df.to_string(index=False))

    # Overall averages, split seen vs unseen - the generalization headline numbers
    seen_avg_cost = result_df[result_df.seen_during_training]["cost_improvement_pct"].mean()
    unseen_avg_cost = result_df[~result_df.seen_during_training]["cost_improvement_pct"].mean()
    seen_avg_slo = result_df[result_df.seen_during_training]["slo_improvement_pct"].mean()
    unseen_avg_slo = result_df[~result_df.seen_during_training]["slo_improvement_pct"].mean()

    print(f"\nAverage cost improvement  - seen patterns: {seen_avg_cost:+.1f}%  |  unseen patterns: {unseen_avg_cost:+.1f}%")
    print(f"Average SLO improvement   - seen patterns: {seen_avg_slo:+.1f}%  |  unseen patterns: {unseen_avg_slo:+.1f}%")
    print("(Small gap between seen/unseen = good generalization; large gap = overfitting to trained patterns.)")

    _plot_improvement_summary(result_df)
    print(f"\nSaved: {RESULTS_DIR}/final_summary_table.csv, final_improvement_summary.png")


def _plot_improvement_summary(result_df):
    patterns = ALL_PATTERNS
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(patterns))
    colors = ["#2E86AB" if p in TRAIN_PATTERNS else "#F18F01" for p in patterns]

    for ax, col, title in [
        (axes[0], "cost_improvement_pct", "Cost Improvement vs Baseline (%)"),
        (axes[1], "slo_improvement_pct", "SLO Violation Reduction vs Baseline (%)"),
    ]:
        vals = [result_df[result_df.pattern == p][col].values[0] for p in patterns]
        bars = ax.bar(x, vals, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        labels = [f"{p}\n(seen)" if p in TRAIN_PATTERNS else f"{p}\n(unseen)" for p in patterns]
        ax.set_xticklabels(labels)
        ax.set_ylabel("% improvement (higher = PPO better)")
        ax.set_title(title)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + (2 if val >= 0 else -6),
                     f"{val:+.0f}%", ha="center", fontsize=9)
        ax.axvline(len(TRAIN_PATTERNS) - 0.5, color="gray", linestyle="--", alpha=0.5)

    plt.suptitle("PPO Agent vs Threshold-Based HPA Baseline — Overall Improvement Summary", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "final_improvement_summary.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
