"""
Workload Generator
-------------------
Produces synthetic but realistic request-rate traces used to drive the
Kubernetes simulation. Each function returns a numpy array of length
`num_steps`, where each value is the incoming requests/sec at that timestep.

Patterns implemented:
    - diurnal        : smooth day/night cycle (sine wave) + noise
    - spike          : baseline traffic with a sudden burst and decay
    - ramp           : gradual, sustained increase in load
    - noisy_steady   : flat baseline with random jitter
    - mixed          : concatenation of several patterns back-to-back
                        (used for realistic evaluation traces)
"""

import numpy as np


class WorkloadGenerator:
    def __init__(self, base_rate: float = 50.0, seed: int = None):
        """
        base_rate : baseline requests/sec around which patterns are built
        seed      : RNG seed for reproducibility
        """
        self.base_rate = base_rate
        self.rng = np.random.default_rng(seed)

    def diurnal(self, num_steps: int, amplitude: float = 40.0, period: int = 288, noise_std: float = 3.0):
        """
        Smooth day/night cycle. `period` = number of steps per full cycle
        (e.g. 288 steps if 1 step = 5 min -> 1 day).
        """
        t = np.arange(num_steps)
        signal = self.base_rate + amplitude * np.sin(2 * np.pi * t / period - np.pi / 2)
        noise = self.rng.normal(0, noise_std, num_steps)
        return np.clip(signal + noise, a_min=1.0, a_max=None)

    def spike(self, num_steps: int, spike_start: int = None, spike_height: float = 150.0,
              spike_width: int = 20, noise_std: float = 3.0):
        """
        Flat baseline traffic with one sudden burst (Gaussian-shaped) that
        decays back to baseline. Simulates an incident / viral traffic event.
        """
        if spike_start is None:
            spike_start = num_steps // 2
        t = np.arange(num_steps)
        spike_curve = spike_height * np.exp(-((t - spike_start) ** 2) / (2 * spike_width ** 2))
        signal = self.base_rate + spike_curve
        noise = self.rng.normal(0, noise_std, num_steps)
        return np.clip(signal + noise, a_min=1.0, a_max=None)

    def ramp(self, num_steps: int, final_rate: float = 200.0, noise_std: float = 3.0):
        """
        Gradual, sustained increase from base_rate to final_rate over the
        whole episode. Simulates organic growth or a slow-building event.
        """
        signal = np.linspace(self.base_rate, final_rate, num_steps)
        noise = self.rng.normal(0, noise_std, num_steps)
        return np.clip(signal + noise, a_min=1.0, a_max=None)

    def noisy_steady(self, num_steps: int, noise_std: float = 8.0):
        """
        Flat baseline with random jitter. The 'boring but realistic' case.
        """
        signal = np.full(num_steps, self.base_rate)
        noise = self.rng.normal(0, noise_std, num_steps)
        return np.clip(signal + noise, a_min=1.0, a_max=None)

    def mixed(self, num_steps: int):
        """
        Stitches together diurnal + spike + ramp + noisy_steady segments to
        form one long, realistic evaluation trace ("a production week").
        """
        segment_len = num_steps // 4
        remainder = num_steps - segment_len * 4

        parts = [
            self.diurnal(segment_len),
            self.spike(segment_len),
            self.ramp(segment_len),
            self.noisy_steady(segment_len + remainder),
        ]
        return np.concatenate(parts)

    def generate(self, pattern: str, num_steps: int, **kwargs):
        """Convenience dispatcher: generate(pattern_name, num_steps)."""
        patterns = {
            "diurnal": self.diurnal,
            "spike": self.spike,
            "ramp": self.ramp,
            "noisy_steady": self.noisy_steady,
            "mixed": self.mixed,
        }
        if pattern not in patterns:
            raise ValueError(f"Unknown pattern '{pattern}'. Choose from {list(patterns.keys())}")
        return patterns[pattern](num_steps, **kwargs)


if __name__ == "__main__":
    # Quick sanity check / visualization when run directly
    import matplotlib.pyplot as plt

    gen = WorkloadGenerator(base_rate=50.0, seed=42)
    steps = 288  # e.g. 1 day at 5-min resolution

    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    patterns = ["diurnal", "spike", "ramp", "noisy_steady", "mixed"]
    for ax, p in zip(axes, patterns):
        trace = gen.generate(p, steps if p != "mixed" else steps * 2)
        ax.plot(trace)
        ax.set_title(p)
        ax.set_ylabel("req/sec")
    axes[-1].set_xlabel("timestep")
    plt.tight_layout()
    plt.savefig("/home/claude/k8s-rl-autoscaler/results/workload_patterns.png", dpi=120)
    print("Saved sanity-check plot to results/workload_patterns.png")