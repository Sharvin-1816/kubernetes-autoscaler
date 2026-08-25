"""
Reward Ablation Study
----------------------
Trains PPO under different reward weightings to empirically demonstrate how
reward shaping causally determines the learned policy's behavior - directly
building on the reward-hacking investigation earlier in this project (see
MDP_SPECIFICATION.md for that history).

Three variants:
    A. slo_only      : w_slo=1.0, w_cost=0.0, w_churn=0.0
                        Agent only cares about avoiding violations, no cost
                        penalty at all. Expect: heavy overprovisioning,
                        near-zero violations, high cost.
    B. balanced       : w_slo=1.0, w_cost=0.6, w_churn=0.1
                        THIS IS YOUR ALREADY-TRAINED, LOCKED FINAL MODEL
                        (models/best_model/best_model.zip). NOT retrained
                        here - reused as-is.
    C. cost_priority  : w_slo=0.5, w_cost=1.2, w_churn=0.1
                        Cost weighted above SLO. Expect: leaner pod counts,
                        more violations, lower cost.

This script trains ONLY variants A and C. Variant B already exists and is
never overwritten by this script.

Usage (from project root):
    python -m agents.train_ablation
Takes ~15-18 minutes total (two ~8 minute training runs, back to back,
unattended - no need to babysit each one separately).
"""
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from envs.k8s_autoscale_env import KubernetesAutoscalingEnv, TRAIN_PATTERNS

EPISODE_LENGTH = 288
N_ENVS = 4
TOTAL_TIMESTEPS = 200_000

VARIANTS = {
    "slo_only":      {"w_slo": 1.0, "w_cost": 0.0, "w_churn": 0.0},
    "cost_priority":  {"w_slo": 0.5, "w_cost": 1.2, "w_churn": 0.1},
}


def train_variant(name, weights):
    print(f"\n{'='*70}\nTraining variant '{name}': {weights}\n{'='*70}")
    save_dir = os.path.join("models", f"ablation_{name}")
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, "tensorboard_logs")

    def build_train_env():
        return Monitor(KubernetesAutoscalingEnv(
            episode_length=EPISODE_LENGTH, pattern_pool=TRAIN_PATTERNS, **weights))

    def build_eval_env():
        return Monitor(KubernetesAutoscalingEnv(
            episode_length=EPISODE_LENGTH, pattern_pool=["noisy_steady"], seed=999, **weights))

    train_env = make_vec_env(build_train_env, n_envs=N_ENVS)
    eval_env = build_eval_env()

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=os.path.join(save_dir, "best_model"),
        log_path=os.path.join(save_dir, "eval_logs"),
        eval_freq=max(5000 // N_ENVS, 1), n_eval_episodes=5, deterministic=True, render=False,
    )

    model = PPO(
        "MlpPolicy", train_env, verbose=0, tensorboard_log=log_dir,
        learning_rate=3e-4, n_steps=2048 // N_ENVS, batch_size=64,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)
    model.save(os.path.join(save_dir, f"ppo_{name}"))
    print(f"Saved variant '{name}' to {save_dir}")


if __name__ == "__main__":
    print("Reward ablation training: 2 variants, ~15-18 min total, unattended.")
    print("(Variant B 'balanced' is your existing locked model - not retrained.)")
    for name, weights in VARIANTS.items():
        train_variant(name, weights)
    print("\nAll ablation variants trained.")
    print("Next: python -m agents.evaluate_ablation")
