"""
FastAPI backend for the Kubernetes RL Autoscaler Dashboard.

Serves the frontend and exposes:
    - REST endpoints to configure/control the live simulation and training
    - A WebSocket that streams simulation state every tick (for live visuals)

Run from project root:
    pip install fastapi "uvicorn[standard]" websockets
    python -m dashboard.backend.server

Then open http://localhost:8000 in a browser.
"""
import os
import sys
import csv
import json
import time
import threading
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from envs.k8s_autoscale_env import make_eval_env, TRAIN_PATTERNS, EVAL_ONLY_PATTERNS, MAX_PODS
from envs.k8s_simulator import TIER_SPECS, FIXED_POD_OVERHEAD, SPINUP_DELAY, PodTier
from agents.baseline_hpa import ThresholdHPAAgent

app = FastAPI(title="K8s RL Autoscaler Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"decisions_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

TIER_NAMES = ["small", "medium", "large"]


# ---------------------------------------------------------------------------
# Global simulation + training state.
# This is a single-user LOCAL dashboard (you run it on your own machine), so
# simple global state + a lock is intentional - no multi-client concerns.
# ---------------------------------------------------------------------------
class SimState:
    def __init__(self):
        self.env = None
        self.agent_type = "hpa"   # "hpa" or "ppo"
        self.ppo_model = None
        self.hpa_agent = None
        self.pattern = "spike"
        self.speed = 2.0          # steps per second
        self.playing = False
        self.obs = None
        self.info = None
        self.episode_cost_total = 0.0
        self.lock = threading.Lock()

        # --- Real per-pod bookkeeping ---
        # The underlying KubernetesSimulator only tracks AGGREGATE counts
        # (active_pods, pending_pods) - it has no concept of individual pod
        # identity. This list is a genuine, real display layer built by
        # tracking actual count deltas each step (see _advance_one_step):
        # every pod here corresponds to a real increment/decrement the
        # simulator made, with a REAL creation step (so age is accurate).
        # What's NOT independently real: CPU/memory per pod, since the
        # simulator models cluster-wide utilization uniformly - every pod
        # is shown with the same true cluster utilization value, not
        # fabricated per-pod variance.
        self.pod_objects = []
        self.next_pod_id = 1
        self.decision_log = []  # capped list of real per-step decision entries

        # --- Live analytics accumulators (real, reset every episode) ---
        self.tier_steps = {"small": 0, "medium": 0, "large": 0}
        self.slo_violations = 0
        self.steps_taken = 0
        self.reward_total = 0.0
        self.reward_components_total = {"slo": 0.0, "cost": 0.0, "churn": 0.0}
        self.cumulative_reward_history = []  # [{step, reward}] this episode

    def reset_env(self):
        self.env = make_eval_env(pattern=self.pattern, episode_length=288)
        self.obs, _ = self.env.reset()
        self.hpa_agent = ThresholdHPAAgent()
        self.hpa_agent.reset()
        self.info = None
        self.episode_cost_total = 0.0

        self.pod_objects = []
        self.next_pod_id = 1
        for _ in range(self.env.sim.state.active_pods):
            self.pod_objects.append({
                "id": f"pod-{self.next_pod_id}", "status": "active",
                "created_step": 0, "spinup_remaining": 0,
            })
            self.next_pod_id += 1
        self.decision_log = []

        self.tier_steps = {"small": 0, "medium": 0, "large": 0}
        self.slo_violations = 0
        self.steps_taken = 0
        self.reward_total = 0.0
        self.reward_components_total = {"slo": 0.0, "cost": 0.0, "churn": 0.0}
        self.cumulative_reward_history = []


sim = SimState()
sim.reset_env()


class TrainingState:
    def __init__(self):
        self.running = False
        self.stop_requested = False
        self.total_timesteps = 200_000
        self.timesteps_done = 0
        self.latest_reward = None
        self.reward_history = []
        self.thread = None
        self.error = None


training = TrainingState()


# ---------------------------------------------------------------------------
# REST: simulation configuration + control
# ---------------------------------------------------------------------------
class SimConfig(BaseModel):
    pattern: str
    agent: str


@app.post("/api/simulation/config")
def configure_simulation(cfg: SimConfig):
    with sim.lock:
        sim.pattern = cfg.pattern
        sim.agent_type = cfg.agent
        if cfg.agent == "ppo":
            try:
                _load_ppo_model()
            except FileNotFoundError as e:
                return {"status": "error", "message": str(e)}
        sim.reset_env()
        sim.playing = False
    return {"status": "ok", "pattern": sim.pattern, "agent": sim.agent_type}


def _load_ppo_model():
    if sim.ppo_model is not None:
        return
    from stable_baselines3 import PPO
    model_path = MODELS_DIR / "best_model" / "best_model.zip"
    if not model_path.exists():
        raise FileNotFoundError("No trained PPO model found yet. Train one first.")
    sim.ppo_model = PPO.load(str(model_path))


class ControlAction(BaseModel):
    action: str  # "play" | "pause" | "reset" | "step"


@app.post("/api/simulation/control")
def control_simulation(ctrl: ControlAction):
    with sim.lock:
        if ctrl.action == "play":
            sim.playing = True
        elif ctrl.action == "pause":
            sim.playing = False
        elif ctrl.action == "reset":
            sim.reset_env()
            sim.playing = False
        elif ctrl.action == "step":
            _advance_one_step()
    return {"status": "ok", "playing": sim.playing}


class SpeedConfig(BaseModel):
    speed: float


@app.post("/api/simulation/speed")
def set_speed(cfg: SpeedConfig):
    sim.speed = max(0.5, min(cfg.speed, 20.0))
    return {"status": "ok", "speed": sim.speed}


@app.get("/api/patterns")
def get_patterns():
    return {"train_patterns": TRAIN_PATTERNS, "eval_only_patterns": EVAL_ONLY_PATTERNS}


@app.get("/api/models")
def list_models():
    best = MODELS_DIR / "best_model" / "best_model.zip"
    return {"ppo_available": best.exists()}


@app.get("/api/tier_info")
def tier_info():
    """Real tier economics from the simulator - single source of truth so
    the frontend never hardcodes duplicate numbers that could drift out of
    sync with the actual model."""
    out = {}
    for name, tier_enum in zip(TIER_NAMES, [PodTier.SMALL, PodTier.MEDIUM, PodTier.LARGE]):
        spec = TIER_SPECS[tier_enum]
        out[name] = {"capacity": spec["capacity"], "cost_per_step": round(spec["cost"] + FIXED_POD_OVERHEAD, 2)}
    return out


@app.get("/api/logs/recent")
def recent_logs():
    return {"log_file": str(LOG_FILE), "entries": sim.decision_log[-100:]}


RESULTS_DIR = PROJECT_ROOT / "results"


def _read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@app.get("/api/results/summary")
def results_summary():
    """Real, pre-computed PPO-vs-baseline benchmark numbers from the locked
    model's evaluation run (results/*.csv, committed to the repo - not
    regenerated on the fly). This is what agents/evaluate.py and
    agents/final_analysis.py produced; the dashboard just reads and charts it."""
    comparison = _read_csv(RESULTS_DIR / "comparison_table.csv")
    improvement = _read_csv(RESULTS_DIR / "final_summary_table.csv")
    ablation = _read_csv(RESULTS_DIR / "ablation_comparison.csv")  # [] until ablation is run

    def num(row, key):
        try:
            return round(float(row[key]), 2)
        except (KeyError, ValueError, TypeError):
            return None

    patterns = {}
    for row in comparison:
        p = row["pattern"]
        patterns.setdefault(p, {"pattern": p, "seen_during_training": row["seen_during_training"] == "True"})
        agent_key = "ppo" if row["agent"] == "PPO" else "hpa"
        patterns[p][agent_key] = {
            "avg_cost": num(row, "avg_cost"),
            "avg_slo_violation_pct": num(row, "avg_slo_violation_pct"),
            "avg_latency_ms": num(row, "avg_latency_ms"),
            "avg_p95_latency_ms": num(row, "avg_p95_latency_ms"),
            "avg_scale_events": num(row, "avg_scale_events"),
        }

    improvements = [
        {
            "pattern": row["pattern"],
            "seen_during_training": row["seen_during_training"] == "True",
            "cost_improvement_pct": num(row, "cost_improvement_pct"),
            "slo_improvement_pct": num(row, "slo_improvement_pct"),
        }
        for row in improvement
    ]

    return {
        "available": len(comparison) > 0,
        "patterns": list(patterns.values()),
        "improvements": improvements,
        "ablation_available": len(ablation) > 0,
        "ablation": ablation,
    }


# Known result figures, in presentation order, with a caption each.
# Only the ones that actually exist on disk are ever returned -
# the frontend never has to guess a filename that isn't there.
RESULT_FIGURES = [
    {"file": "comparison_barplot.png", "title": "PPO vs. HPA — full comparison",
     "caption": "avg_results.csv plotted directly by agents/evaluate.py: cost, SLO violation %, latency across all 5 patterns."},
    {"file": "final_improvement_summary.png", "title": "Improvement summary",
     "caption": "agents/final_analysis.py — % improvement of PPO over HPA baseline, per pattern."},
    {"file": "episode_trace_spike.png", "title": "Episode trace — spike pattern",
     "caption": "Step-by-step trace of one held-out spike episode: traffic, pod count, latency, tier."},
    {"file": "ablation_tradeoff.png", "title": "Reward ablation — SLO vs. cost trade-off",
     "caption": "slo_only / balanced / cost_priority reward variants, from agents/evaluate_ablation.py."},
]


@app.get("/api/results/images")
def results_images():
    out = []
    for fig in RESULT_FIGURES:
        if (RESULTS_DIR / fig["file"]).exists():
            out.append({**fig, "url": f"/results-assets/{fig['file']}"})
    return {"images": out}


# ---------------------------------------------------------------------------
# Stepping logic
# ---------------------------------------------------------------------------
def _persist_log_entry(entry):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # a logging failure should never crash the simulation


def _reconcile_pods(prev_total, new_active, new_pending, current_step):
    """Updates sim.pod_objects to match the simulator's real aggregate
    counts, giving each pod a genuine id and creation step. The underlying
    simulator only tracks counts, not which specific pod was added/removed,
    so the removal order below is a disclosed DISPLAY-LAYER convention
    (cancel pending pods first, then remove the most-recently-activated
    active pods) - not something the real simulator itself decides."""
    new_total = new_active + new_pending
    delta = new_total - prev_total

    if delta > 0:
        for _ in range(delta):
            sim.pod_objects.append({
                "id": f"pod-{sim.next_pod_id}", "status": "pending",
                "created_step": current_step, "spinup_remaining": SPINUP_DELAY,
            })
            sim.next_pod_id += 1
    elif delta < 0:
        to_remove = -delta
        pending_idx = [i for i, p in enumerate(sim.pod_objects) if p["status"] == "pending"]
        for i in reversed(pending_idx[:to_remove]):
            sim.pod_objects.pop(i)
            to_remove -= 1
        if to_remove > 0:
            active_idx = [i for i, p in enumerate(sim.pod_objects) if p["status"] == "active"]
            active_idx.sort(key=lambda i: sim.pod_objects[i]["created_step"], reverse=True)
            for i in sorted(active_idx[:to_remove], reverse=True):
                sim.pod_objects.pop(i)

    for p in sim.pod_objects:
        if p["status"] == "pending":
            p["spinup_remaining"] -= 1
            if p["spinup_remaining"] <= 0:
                p["status"] = "active"


def _advance_one_step():
    prev_active = sim.env.sim.state.active_pods
    prev_pending = sum(c for c, _ in sim.env.sim.state.pending_pods)
    prev_total = prev_active + prev_pending

    if sim.agent_type == "ppo" and sim.ppo_model is not None:
        action, _ = sim.ppo_model.predict(sim.obs, deterministic=True)
    else:
        cpu_util = sim.obs[0]
        action = sim.hpa_agent.act(cpu_util)
    requested_horizontal, requested_tier = int(action[0]), int(action[1])

    obs, reward, terminated, truncated, info = sim.env.step(action)
    sim.obs = obs
    sim.info = info
    sim.episode_cost_total += info.get("cost", 0.0)

    current_step = sim.env._step_idx

    # --- live analytics bookkeeping (all real, derived from this real step) ---
    sim.steps_taken += 1
    if info.get("slo_violation"):
        sim.slo_violations += 1
    tier_name = TIER_NAMES[info.get("tier", 1)]
    sim.tier_steps[tier_name] = sim.tier_steps.get(tier_name, 0) + 1
    sim.reward_total += reward
    sim.reward_components_total["slo"] += info.get("reward_slo_component", 0.0)
    sim.reward_components_total["cost"] += info.get("reward_cost_component", 0.0)
    sim.reward_components_total["churn"] += info.get("reward_churn_component", 0.0)
    sim.cumulative_reward_history.append({"step": current_step, "reward": round(sim.reward_total, 2)})
    sim.cumulative_reward_history = sim.cumulative_reward_history[-300:]
    new_total = info["active_pods"] + info["pending_pods"]
    _reconcile_pods(prev_total, info["active_pods"], info["pending_pods"], current_step)

    # --- real decision log entry, built entirely from actual step results ---
    applied_h_delta = info["horizontal_action"] - 3
    requested_h_delta = requested_horizontal - 3
    h_blocked = requested_h_delta != applied_h_delta
    tier_blocked = requested_tier != info["tier_action"]

    parts = []
    if applied_h_delta > 0:
        parts.append(f"scaled up {applied_h_delta:+d} pod(s) ({prev_total}\u2192{new_total})")
    elif applied_h_delta < 0:
        parts.append(f"scaled down {applied_h_delta} pod(s) ({prev_total}\u2192{new_total})")
    else:
        parts.append("held pod count")
    if h_blocked:
        parts.append(f"requested {requested_h_delta:+d} BLOCKED by scale-down cooldown")
    if tier_blocked:
        parts.append(f"tier change to {TIER_NAMES[requested_tier]} blocked by tier cooldown")
    if info["slo_violation"]:
        parts.append(f"SLO VIOLATION ({info['latency_ms']:.0f}ms)")

    log_entry = {
        "step": current_step,
        "agent": sim.agent_type,
        "message": f"[{sim.agent_type.upper()}] " + "; ".join(parts),
        "cpu_util": round(info["cpu_util"], 3),
        "latency_ms": round(info["latency_ms"], 1),
        "slo_violation": info["slo_violation"],
        "blocked": h_blocked or tier_blocked,
        "cost": round(info["cost"], 2),
        "tier": info["tier"],
        "active_pods": info["active_pods"],
    }
    sim.decision_log.append(log_entry)
    sim.decision_log = sim.decision_log[-500:]
    _persist_log_entry(log_entry)

    if terminated or truncated:
        sim.reset_env()
    return info


def _current_state_payload():
    info = sim.info or {}
    return {
        "type": "simulation_state",
        "pattern": sim.pattern,
        "agent": sim.agent_type,
        "playing": sim.playing,
        "step": sim.env._step_idx if sim.env else 0,
        "episode_length": sim.env.episode_length if sim.env else 288,
        "request_rate": info.get("request_rate", 0),
        "active_pods": info.get("active_pods", 0),
        "pending_pods": info.get("pending_pods", 0),
        "tier": info.get("tier", 1),
        "cpu_util": info.get("cpu_util", 0),
        "mem_util": info.get("mem_util", 0),
        "latency_ms": info.get("latency_ms", 0),
        "slo_violation": info.get("slo_violation", False),
        "cost": info.get("cost", 0),
        "episode_cost_total": sim.episode_cost_total,
        "max_pods": MAX_PODS,
        "reward_step": info.get("reward", 0),
        "reward_components": {
            "slo": info.get("reward_slo_component", 0.0),
            "cost": info.get("reward_cost_component", 0.0),
            "churn": info.get("reward_churn_component", 0.0),
        },
        "reward_components_total": sim.reward_components_total,
        "reward_total": round(sim.reward_total, 2),
        "cumulative_reward_history": sim.cumulative_reward_history,
        "tier_steps": sim.tier_steps,
        "slo_compliance_pct": round(100.0 * (1 - sim.slo_violations / sim.steps_taken), 1) if sim.steps_taken else 100.0,
        "steps_taken": sim.steps_taken,
        "pods": [
            {
                "id": p["id"],
                "status": p["status"],
                "tier": TIER_NAMES[info.get("tier", 1)],
                "age_steps": max(0, (sim.env._step_idx if sim.env else 0) - p["created_step"]),
                "cpu_util": info.get("cpu_util", 0),  # real, but cluster-wide - see SimState docstring
            }
            for p in sim.pod_objects
        ],
        "recent_logs": sim.decision_log[-15:],
    }


@app.websocket("/ws/simulation")
async def ws_simulation(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if sim.playing:
                with sim.lock:
                    _advance_one_step()
                await websocket.send_json(_current_state_payload())
                await asyncio.sleep(1.0 / sim.speed)
            else:
                await websocket.send_json(_current_state_payload())
                await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Training control (runs in a background thread; dashboard polls status)
# ---------------------------------------------------------------------------
class TrainConfig(BaseModel):
    total_timesteps: int = 200_000


@app.post("/api/training/start")
def start_training(cfg: TrainConfig):
    if training.running:
        return {"status": "already_running"}
    training.running = True
    training.stop_requested = False
    training.timesteps_done = 0
    training.reward_history = []
    training.error = None
    training.total_timesteps = cfg.total_timesteps
    training.thread = threading.Thread(target=_run_training, daemon=True)
    training.thread.start()
    return {"status": "started"}


@app.post("/api/training/stop")
def stop_training():
    training.stop_requested = True
    return {"status": "stop_requested"}


@app.get("/api/training/status")
def training_status():
    return {
        "running": training.running,
        "timesteps_done": training.timesteps_done,
        "total_timesteps": training.total_timesteps,
        "latest_reward": training.latest_reward,
        "reward_history": training.reward_history[-300:],
        "error": training.error,
    }


def _run_training():
    """Runs PPO training in a background thread. A custom callback reports
    progress to `training` (polled by the frontend) and checks
    `training.stop_requested` each step so the dashboard's Stop button can
    interrupt training early (SB3 stops when the callback returns False)."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        from envs.k8s_autoscale_env import make_train_env

        class DashboardCallback(BaseCallback):
            def _on_step(self) -> bool:
                training.timesteps_done = self.num_timesteps
                if len(self.model.ep_info_buffer) > 0:
                    recent = [ep["r"] for ep in self.model.ep_info_buffer]
                    mean_r = sum(recent) / len(recent)
                    training.latest_reward = mean_r
                    training.reward_history.append(
                        {"timestep": self.num_timesteps, "reward": mean_r}
                    )
                return not training.stop_requested

        def build_env():
            return Monitor(make_train_env(episode_length=288))

        train_env = make_vec_env(build_env, n_envs=4)
        model = PPO(
            "MlpPolicy", train_env, verbose=0,
            learning_rate=3e-4, n_steps=512, batch_size=64,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        )
        model.learn(total_timesteps=training.total_timesteps, callback=DashboardCallback())

        save_dir = MODELS_DIR / "best_model"
        os.makedirs(save_dir, exist_ok=True)
        model.save(str(save_dir / "best_model"))

        # invalidate any cached PPO model in the simulator so it reloads fresh
        sim.ppo_model = None
    except Exception as e:
        training.error = str(e)
    finally:
        training.running = False


# ---------------------------------------------------------------------------
# Serve frontend (must be mounted LAST - it's a catch-all)
# ---------------------------------------------------------------------------
if RESULTS_DIR.exists():
    app.mount("/results-assets", StaticFiles(directory=str(RESULTS_DIR)), name="results-assets")

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)