"""
Train a PPO agent on the KubernetesAutoscalingEnv.

Usage (from project root):
    python -m agents.train

This will:
    1. Build a vectorized training env (sampling from TRAIN_PATTERNS only)
    2. Train PPO for TOTAL_TIMESTEPS steps
    3. Log to TensorBoard (models/tensorboard_logs/)
    4. Periodically evaluate on a held-out pattern and save the best model
    5. Save the final model to models/ppo_k8s_autoscaler.zip

To watch training progress live:
    tensorboard --logdir models/tensorboard_logs
    (then open the printed localhost URL in your browser)
"""

import os
import sys

# Ensure project root is in sys.path so envs can be imported regardless of execution method
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from envs.k8s_autoscale_env import make_train_env, make_eval_env, TRAIN_PATTERNS

TOTAL_TIMESTEPS = 200_000      # adjust down (e.g. 50_000) for a quick first test run
EPISODE_LENGTH = 288            # ~1 simulated day per episode
N_ENVS = 4                      # parallel envs for faster data collection
MODELS_DIR = "models"
LOGS_DIR = os.path.join(MODELS_DIR, "tensorboard_logs")


def build_train_env():
    return Monitor(make_train_env(episode_length=EPISODE_LENGTH))


def build_eval_env():
    # Evaluate periodically on noisy_steady (a TRAIN pattern) just to track
    # "is it learning at all" during training. True generalization testing
    # on EVAL_ONLY_PATTERNS happens later in evaluate.py, after training.
    return Monitor(make_eval_env(pattern="noisy_steady", episode_length=EPISODE_LENGTH, seed=999))


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print(f"Training PPO on patterns: {TRAIN_PATTERNS}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS}")

    train_env = make_vec_env(build_train_env, n_envs=N_ENVS)
    eval_env = build_eval_env()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODELS_DIR, "best_model"),
        log_path=os.path.join(MODELS_DIR, "eval_logs"),
        eval_freq=max(5000 // N_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=LOGS_DIR,
        learning_rate=3e-4,
        n_steps=2048 // N_ENVS,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,   # slightly encourages exploration - useful given
                          # the churn penalty could otherwise push the agent
                          # to a lazy "never change anything" policy early on
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)

    final_path = os.path.join(MODELS_DIR, "ppo_k8s_autoscaler")
    model.save(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}.zip")
    print(f"Best model (by eval reward) saved to {MODELS_DIR}/best_model/best_model.zip")
    print(f"\nTo view training curves, run:\n  tensorboard --logdir {LOGS_DIR}")


if __name__ == "__main__":
    main()
