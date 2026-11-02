# Kubernetes RL Autoscaler

Reinforcement-learning-based pod autoscaling for a simulated Kubernetes cluster, benchmarked against a threshold-based (HPA-style) baseline. Built for an RL course project, based on:

> Pandey, V. (2026). *Reinforcement Learning-Based Autoscaling for Cost and Performance Optimization in Kubernetes Clusters.* Springer LNDECT vol. 277.

This implements the paper's core idea — an RL agent jointly performing horizontal and vertical pod scaling to minimize SLO violations and cost — in a custom simulated environment, with a queuing-theoretic latency model rather than a real EKS cluster. See [`MDP_SPECIFICATION.md`](MDP_SPECIFICATION.md) for the full formal writeup, including a documented debugging history (reward hacking, tier-oscillation, and a known cluster-wide-tier limitation) worth reading if you want to understand *why* things are built the way they are, not just *what* they are.

**Repository:** https://github.com/Sharvin-1816/kubernetes-autoscaler

## My contribution

The paper's idea is joint horizontal + vertical scaling by an RL agent. This project adds:

1. **Queuing-based cluster simulator** — M/M/c-style latency `base / (1 - rho)^2`, 2-step pod spin-up delay, and a fixed per-pod overhead that makes the medium tier the cheapest per unit of capacity. Fully reproducible, no real cluster needed.
2. **Asymmetric stabilization windows inside the MDP** — tier changes gated both ways, scale-down gated, scale-up never gated (like Kubernetes HPA/VPA), with both cooldown timers in the state so it stays Markov. This removed tier flapping near traffic peaks; a first single shared cooldown made ramp/spike worse, hence the asymmetric design.
3. **Evidence-based reward-hacking fix** — `agents/diagnose.py` splits the reward into SLO/cost/churn parts; it showed the agent had learned to max out pods. Capping the SLO penalty at 3.0 and weighting cost at 0.6 fixed it.
4. **Generalization protocol** — train on `diurnal` + `noisy_steady` only; evaluate on held-out `spike`, `ramp`, `mixed`.
5. **Live dashboard on real data** — streams simulator state, logs every decision, controls real PPO training, and shows the benchmark results and generated figures.

## Architecture

![Architecture diagram](architecture-diagram.svg)

Three layers, all built on the same underlying simulation core:

1. **Simulation core** — a Gymnasium-compatible environment wrapping a custom Kubernetes cluster simulator
2. **RL training & evaluation** — trains and analyzes agents against the simulation core
3. **Live dashboard** — a local web app that visualizes the simulation core in real time and can trigger real training runs

## What's actually used here (all real, no mocked tooling)

| Purpose | Tool |
|---|---|
| RL environment interface | [Gymnasium](https://gymnasium.farama.org/) (industry-standard Gym API) |
| RL algorithm | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO) |
| Neural network backend | [PyTorch](https://pytorch.org/) |
| Training visualization | [TensorBoard](https://www.tensorflow.org/tensorboard) |
| Data analysis | [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| Plotting | [Matplotlib](https://matplotlib.org/) |
| Dashboard backend | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) + WebSockets |
| Dashboard frontend | [React](https://react.dev/) (via CDN + Babel standalone) + [Tailwind CSS](https://tailwindcss.com/) (via CDN) + [Chart.js](https://www.chartjs.org/) |
| Decision logging | Structured JSONL files (`dashboard/backend/logs/`) |

**Not used, deliberately:** no cloud observability stack (Prometheus/Grafana would be the natural next step, matching the reference paper's real deployment, but is out of scope for a local single-user course project). No hyperparameter search tool yet (Optuna planned next, see Roadmap below).

## Project structure

```
k8s-rl-autoscaler/
├── README.md                       <- you are here
├── MDP_SPECIFICATION.md            <- formal MDP writeup + debugging history
├── architecture-diagram.svg
├── config.py                       <- centralized constants (not yet wired into all files)
├── requirements.txt
│
├── envs/
│   ├── k8s_simulator.py            <- cluster "physics": queuing latency, cost, spin-up delay
│   └── k8s_autoscale_env.py        <- Gymnasium env: state/action/reward, stabilization windows
│
├── utils/
│   └── workload_generator.py       <- synthetic traffic patterns (diurnal, spike, ramp, mixed, ...)
│
├── agents/
│   ├── baseline_hpa.py             <- threshold-based HPA-mimic baseline agent
│   ├── train.py                    <- PPO training
│   ├── evaluate.py                 <- PPO vs baseline comparison across patterns
│   ├── diagnose.py                 <- root-cause diagnostics (reward breakdown, action distribution)
│   ├── final_analysis.py           <- polished improvement-summary plots
│   ├── train_ablation.py           <- reward ablation study: trains alternate reward weightings
│   └── evaluate_ablation.py        <- compares reward variants (cost-vs-reliability trade-off)
│
├── dashboard/
│   ├── backend/server.py           <- FastAPI app: simulation control, training control, logging
│   └── frontend/index.html         <- single-file React dashboard
│
├── models/                         <- trained model checkpoints (generated)
└── results/                        <- CSVs and plots (generated)
```

## Quickstart

```bash
pip install -r requirements.txt

# Train the main PPO agent (~8 min for 200k timesteps)
python -m agents.train

# Compare PPO vs the HPA baseline across seen and unseen traffic patterns
python -m agents.evaluate

# Root-cause diagnostics: reward breakdown, action distribution, cooldown behavior
python -m agents.diagnose

# Polished improvement-summary plot
python -m agents.final_analysis

# Run the live dashboard (train, simulate, and inspect the agent from a browser)
python -m dashboard.backend.server
# then open http://localhost:8000
```

## Method summary

- **State:** 9-dim vector — CPU/memory utilization, latency, active/pending pod counts, current tier, traffic trend, and two stabilization-window timers.
- **Action:** `MultiDiscrete([7, 3])` — horizontal pod delta (-3..+3) and vertical tier (small/medium/large).
- **Reward:** weighted combination of SLO-violation penalty, cost penalty, and a churn penalty, with weights tuned specifically to avoid a reward-hacking failure mode found during development (see MDP_SPECIFICATION.md).
- **Baseline:** a threshold-based scaler mimicking real Kubernetes HPA (reacts to average CPU over a rolling window).
- **Generalization test:** the agent trains only on `diurnal` and `noisy_steady` traffic patterns; `spike`, `ramp`, and `mixed` are held out entirely for evaluation.

## Dashboard

Three tabs at http://localhost:8000: **Live Simulation** (hex-pod cluster view, traffic/latency chart, decision log, SLO-compliance gauge, tier-time donut, reward breakdown, cumulative reward), **Benchmark Results** (PPO vs HPA charts, full table and the generated PNGs from `results/`), and **Train** (real PPO training with a live reward curve).

## Results (locked final model)

PPO has fewer SLO violations than the HPA baseline on every pattern, seen and unseen (10 episodes per pattern, 150 ms latency target):

| Pattern | Seen in training | HPA SLO viol. | PPO SLO viol. | PPO cost vs HPA |
|---|---|---|---|---|
| diurnal | yes | 47.2% | 16.7% | +24.4% |
| noisy_steady | yes | 78.5% | 9.0% | +50.4% |
| spike | no | 55.8% | 24.4% | +2.2% |
| ramp | no | 97.9% | 80.5% | +7.0% |
| mixed | no | 53.1% | 44.2% | -3.5% |

The cost premium follows from the reward weighting (reliability prioritized over cost). Raw numbers: `results/comparison_table.csv`, `results/final_summary_table.csv`, `results/final_improvement_summary.png`.

## Known limitations (documented, not hidden)

- **Cluster-wide tier, not per-pod:** all pods share one tier at a time. This makes the trained policy strongly favor horizontal-only scaling, since committing the whole cluster to a higher tier for a temporary peak is rarely worth it. A genuinely per-pod tier model would be a real architecture change, not a quick fix.
- **Held-out weak spots:** on `ramp` and `mixed`, traffic approaches the 15-pod ceiling, queuing latency blows up near saturation, and the agent stays on the small tier. PPO's p95 latency on held-out patterns is also higher than HPA's.
- **Cumulative cost metric:** the dashboard's "Episode cost" stat is a running total (always increasing by design), not a live rate — see the per-step `cost` field in the decision log for the fluctuating value.
- **Synthetic traffic, not live data:** traffic is generated from parameterized formulas (sine waves, Gaussian bumps, linear ramps), not pulled from a real service. This is a standard, disclosed simplification given no real cluster is available — see MDP_SPECIFICATION.md for the full reasoning.

## Roadmap

- [ ] Reward ablation study (in progress — `train_ablation.py` / `evaluate_ablation.py`)
- [ ] Algorithm comparison (DQN or A2C alongside PPO)
- [ ] Optuna hyperparameter sweep
- [ ] `config.py` fully wired into all modules (currently exists but unused by most files)
