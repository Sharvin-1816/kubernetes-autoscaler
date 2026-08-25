"""
Baseline Agent: Threshold-based HPA mimic
-------------------------------------------
Mimics real Kubernetes Horizontal Pod Autoscaler (HPA) behavior:
    - If average CPU utilization over the last `window` steps > `scale_up_thresh`,
      scale up by 1 pod.
    - If average CPU utilization < `scale_down_thresh`, scale down by 1 pod.
    - Otherwise hold.

This agent NEVER changes the pod tier (vertical scaling) - real HPA doesn't
do vertical scaling by default either (that's VPA's job, and the two aren't
usually run together on the same metric). So this baseline always stays on
MEDIUM tier, which is a fair, realistic comparison point.

No learning involved - this is the "dumb but standard" control group your
RL agent needs to beat.
"""

from collections import deque
from envs.k8s_simulator import PodTier


class ThresholdHPAAgent:
    def __init__(self, scale_up_thresh: float = 0.75, scale_down_thresh: float = 0.30,
                 window: int = 3):
        self.scale_up_thresh = scale_up_thresh
        self.scale_down_thresh = scale_down_thresh
        self.window = window
        self.cpu_history = deque(maxlen=window)

    def reset(self):
        self.cpu_history.clear()

    def act(self, cpu_util: float):
        """
        cpu_util : current CPU utilization (0.0 - 1.5 range, matches obs space)

        Returns an action in the SAME format the Gym env expects:
        [horizontal_choice, tier_choice] where
            horizontal_choice: 0..6 mapping to pod delta (-3..+3), 3 = hold
                                (HPA baseline only ever uses -1/hold/+1, i.e. 2/3/4,
                                to stay a fair single-pod-at-a-time comparison point)
            tier_choice: always 1 (MEDIUM) - HPA doesn't do vertical scaling
        """
        self.cpu_history.append(cpu_util)
        avg_cpu = sum(self.cpu_history) / len(self.cpu_history)

        if avg_cpu > self.scale_up_thresh:
            horizontal_choice = 4  # scale up by 1 (index 4 = +1 in the 7-option scheme)
        elif avg_cpu < self.scale_down_thresh:
            horizontal_choice = 2  # scale down by 1 (index 2 = -1)
        else:
            horizontal_choice = 3  # hold

        tier_choice = int(PodTier.MEDIUM)
        return [horizontal_choice, tier_choice]


def run_episode(env, agent, deterministic_reset_pattern: str = None):
    """
    Runs one full episode of `agent` on `env`, returns the episode history
    (list of per-step info dicts, same format as env.get_episode_history()).

    Works for ThresholdHPAAgent (has .act(cpu_util)) - for RL agents (SB3
    models), use run_episode_sb3 in evaluate.py instead, since SB3 models
    take the full observation, not just cpu_util.
    """
    agent.reset()
    obs, info = env.reset()
    terminated = truncated = False

    while not (terminated or truncated):
        cpu_util = obs[0]  # first element of observation is cpu_util
        action = agent.act(cpu_util)
        obs, reward, terminated, truncated, info = env.step(action)

    return env.get_episode_history()


if __name__ == "__main__":
    # Quick standalone test: run the HPA baseline on a spike pattern and
    # print summary stats - run this locally to sanity-check the baseline.
    from envs.k8s_autoscale_env import make_eval_env

    env = make_eval_env(pattern="spike", episode_length=100, seed=1)
    agent = ThresholdHPAAgent()
    history = run_episode(env, agent)

    total_cost = sum(h["cost"] for h in history)
    slo_violations = sum(1 for h in history if h["slo_violation"])
    avg_latency = sum(h["latency_ms"] for h in history) / len(history)

    print(f"Pattern: spike | Steps: {len(history)}")
    print(f"Total cost: {total_cost:.2f}")
    print(f"SLO violations: {slo_violations} / {len(history)} steps ({100*slo_violations/len(history):.1f}%)")
    print(f"Average latency: {avg_latency:.1f} ms")
    print("\nBaseline sanity check passed if no errors were raised above.")