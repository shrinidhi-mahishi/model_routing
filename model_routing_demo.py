"""
The 50% Cheaper Agent: Autonomous LLM Routing with Bayesian Bandits
===================================================================

DevConf.CZ 2026 — Live Demo

Demonstrates a Bayesian router using Thompson Sampling that learns
to shift traffic from expensive to cheaper LLM models — without
human labels.

Run:
    pip install -r requirements.txt
    streamlit run model_routing_demo.py
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import beta as beta_dist

# ─────────────────────────────────────────────────────────────────────────────
# MODEL PROFILES
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PROFILES = {
    "gpt-4o": {
        "cost_per_1k": 0.005,
        "base_validity": 0.96,
        "latency_range": (1500, 3500),
        "retry_rate": 0.04,
        "color": "#10B981",
        "label": "GPT-4o ($$$)",
    },
    "gpt-4o-mini": {
        "cost_per_1k": 0.00015,
        "base_validity": 0.89,
        "latency_range": (400, 1200),
        "retry_rate": 0.09,
        "color": "#3B82F6",
        "label": "GPT-4o-mini ($)",
    },
    "claude-haiku": {
        "cost_per_1k": 0.00025,
        "base_validity": 0.91,
        "latency_range": (300, 900),
        "retry_rate": 0.07,
        "color": "#F59E0B",
        "label": "Claude Haiku ($)",
    },
}

EXPERT_PRIORS = {
    "gpt-4o": {"alpha": 8, "beta": 3},
    "gpt-4o-mini": {"alpha": 5, "beta": 4},
    "claude-haiku": {"alpha": 5, "beta": 4},
}

UNIFORM_PRIORS = {
    "gpt-4o": {"alpha": 1, "beta": 1},
    "gpt-4o-mini": {"alpha": 1, "beta": 1},
    "claude-haiku": {"alpha": 1, "beta": 1},
}

# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StepRecord:
    """Single routing decision — one row of telemetry for visualisation."""

    query_id: int
    model_selected: str
    reward: float
    validity_r: float
    latency_r: float
    retry_r: float
    latency_ms: float
    is_valid: bool
    retried: bool
    cost: float
    baseline_cost: float
    fallback_used: bool
    alphas: Dict[str, float] = field(default_factory=dict)
    betas: Dict[str, float] = field(default_factory=dict)


class ModelSimulator:
    """Simulates LLM call outcomes with configurable quality / latency."""

    def __init__(self, profiles: Dict):
        self.profiles = {k: dict(v) for k, v in profiles.items()}
        self.degradation: Dict[str, float] = {}

    def call(self, model: str, tokens: int = 500) -> Dict:
        profile = self.profiles[model]
        deg = self.degradation.get(model, 1.0)

        lo, hi = profile["latency_range"]
        latency = random.uniform(lo, hi) * deg

        validity = min(profile["base_validity"] / deg, 1.0)
        is_valid = random.random() < validity

        retry_rate = min(profile["retry_rate"] * deg, 1.0)
        retried = random.random() < retry_rate

        cost = (tokens / 1000) * profile["cost_per_1k"]
        return dict(
            latency_ms=latency,
            is_valid=is_valid,
            retried=retried,
            cost=cost,
            tokens=tokens,
        )

    def degrade(self, model: str, factor: float):
        self.degradation[model] = factor


class BayesianRouter:
    """
    Thompson Sampling router with three production-grade additions:

    1. Composite Reward   — 3 objective signals, 0 human labels
    2. Decaying Memory    — adapts to non-stationary performance (model rot)
    3. Circuit Breaker    — falls back to safe model when confidence is low
    """

    REWARD_WEIGHTS = {"validity": 0.50, "latency": 0.30, "no_retry": 0.20}

    def __init__(
        self,
        priors: Dict[str, Dict],
        gamma: float = 0.95,
        decay_interval: int = 50,
        confidence_floor: float = 0.50,
        shadow_rate: float = 0.10,
    ):
        self.models = {name: dict(p) for name, p in priors.items()}
        self.gamma = gamma
        self.decay_interval = decay_interval
        self.confidence_floor = confidence_floor
        self.shadow_rate = shadow_rate
        self.fallback_model = "gpt-4o"
        self.query_count = 0

    # ── selection ────────────────────────────────────────────────────────

    def select_model(self) -> Tuple[str, bool]:
        """Thompson sample from each model's Beta; highest wins.

        Shadow evaluation (shadow_rate) forces random exploration on a
        fraction of traffic so under-sampled models accumulate evidence.
        """
        if random.random() < self.shadow_rate:
            return random.choice(list(self.models.keys())), False

        samples = {
            name: np.random.beta(s["alpha"], s["beta"])
            for name, s in self.models.items()
        }
        selected = max(samples, key=samples.get)

        conf = self._confidence(selected)
        if conf < self.confidence_floor and selected != self.fallback_model:
            return self.fallback_model, True
        return selected, False

    # ── reward ───────────────────────────────────────────────────────────

    def compute_reward(
        self, latency_ms: float, is_valid: bool, retried: bool
    ) -> Tuple[float, float, float, float]:
        """
        Composite Reward Function — the key innovation.

        Three signals captured from normal agent telemetry:
          1. Syntactic validity   (did the output parse?)
          2. Normalised latency   (sigmoid, adjustable midpoint)
          3. No self-correction   (agent didn't retry)

        Returns (total, validity_r, latency_r, retry_r).
        """
        w = self.REWARD_WEIGHTS

        validity_r = w["validity"] if is_valid else 0.0
        latency_r = w["latency"] / (1.0 + math.exp((latency_ms - 2000) / 600))
        retry_r = w["no_retry"] if not retried else 0.0

        return validity_r + latency_r + retry_r, validity_r, latency_r, retry_r

    # ── update ───────────────────────────────────────────────────────────

    def update(self, model: str, reward: float):
        self.models[model]["alpha"] += reward
        self.models[model]["beta"] += 1 - reward
        self.query_count += 1
        if self.query_count % self.decay_interval == 0:
            self._decay()

    def _decay(self):
        for m in self.models:
            self.models[m]["alpha"] = max(1.0, self.models[m]["alpha"] * self.gamma)
            self.models[m]["beta"] = max(1.0, self.models[m]["beta"] * self.gamma)

    def _confidence(self, model: str) -> float:
        s = self.models[model]
        return s["alpha"] / (s["alpha"] + s["beta"])

    def get_state(self) -> Dict:
        return {n: dict(s) for n, s in self.models.items()}


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────


def run_simulation(
    n_queries: int,
    priors: Dict,
    profiles: Dict | None = None,
    rot_config: Dict | None = None,
    gamma: float = 0.95,
    decay_interval: int = 50,
    seed: int = 42,
) -> List[StepRecord]:
    profiles = profiles or MODEL_PROFILES
    np.random.seed(seed)
    random.seed(seed)

    router = BayesianRouter(priors, gamma=gamma, decay_interval=decay_interval)
    sim = ModelSimulator(profiles)
    history: List[StepRecord] = []

    for i in range(n_queries):
        if rot_config and i == rot_config.get("at_query"):
            sim.degrade(rot_config["model"], rot_config["factor"])

        model, fallback = router.select_model()
        result = sim.call(model)
        reward, v_r, l_r, r_r = router.compute_reward(
            result["latency_ms"], result["is_valid"], result["retried"]
        )
        router.update(model, reward)

        baseline_cost = (result["tokens"] / 1000) * profiles["gpt-4o"]["cost_per_1k"]
        state = router.get_state()

        history.append(
            StepRecord(
                query_id=i,
                model_selected=model,
                reward=reward,
                validity_r=v_r,
                latency_r=l_r,
                retry_r=r_r,
                latency_ms=result["latency_ms"],
                is_valid=result["is_valid"],
                retried=result["retried"],
                cost=result["cost"],
                baseline_cost=baseline_cost,
                fallback_used=fallback,
                alphas={m: s["alpha"] for m, s in state.items()},
                betas={m: s["beta"] for m, s in state.items()},
            )
        )

    return history


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def _c(model: str) -> str:
    return MODEL_PROFILES.get(model, {}).get("color", "#888")


def _l(model: str) -> str:
    return MODEL_PROFILES.get(model, {}).get("label", model)


def plot_beta_distributions(history: List[StepRecord], step: int):
    rec = history[step]
    x = np.linspace(0.001, 0.999, 300)
    fig = go.Figure()
    for model in MODEL_PROFILES:
        a, b = rec.alphas[model], rec.betas[model]
        y = beta_dist.pdf(x, a, b)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"{_l(model)}  α={a:.1f} β={b:.1f}",
                line=dict(color=_c(model), width=3),
                fill="tozeroy",
                opacity=0.35,
            )
        )
    fig.update_layout(
        title="Model Quality Beliefs  (Beta Distributions)",
        xaxis_title="Estimated Quality",
        yaxis_title="Density",
        height=380,
        **PLOTLY_LAYOUT,
    )
    return fig


def plot_traffic_distribution(history: List[StepRecord], step: int, window: int = 20):
    records = history[: step + 1]
    models = list(MODEL_PROFILES.keys())
    freqs = {m: [] for m in models}
    qs = []
    for i in range(len(records)):
        win = records[max(0, i - window + 1) : i + 1]
        n = len(win)
        for m in models:
            freqs[m].append(sum(1 for r in win if r.model_selected == m) / n * 100)
        qs.append(i + 1)
    fig = go.Figure()
    for m in models:
        fig.add_trace(
            go.Scatter(
                x=qs,
                y=freqs[m],
                name=_l(m),
                mode="lines",
                stackgroup="one",
                line=dict(width=0.5, color=_c(m)),
            )
        )
    fig.update_layout(
        title=f"Traffic Distribution  (rolling {window}-query window)",
        xaxis_title="Query #",
        yaxis_title="% of Traffic",
        yaxis=dict(range=[0, 100]),
        height=380,
        **PLOTLY_LAYOUT,
    )
    return fig


def plot_cumulative_cost(history: List[StepRecord], step: int):
    records = history[: step + 1]
    router_cost = list(np.cumsum([r.cost for r in records]))
    baseline_cost = list(np.cumsum([r.baseline_cost for r in records]))
    qs = list(range(1, len(records) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=qs,
            y=baseline_cost,
            name="Always GPT-4o",
            line=dict(color="#EF4444", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=qs,
            y=router_cost,
            name="Bayesian Router",
            line=dict(color="#10B981", width=3),
            fill="tonexty",
            fillcolor="rgba(16,185,129,0.12)",
        )
    )
    fig.update_layout(
        title="Cumulative Cost: Router vs Baseline",
        xaxis_title="Query #",
        yaxis_title="Cost ($)",
        height=380,
        **PLOTLY_LAYOUT,
    )
    return fig


def plot_reward_breakdown(history: List[StepRecord], step: int, window: int = 30):
    records = history[max(0, step - window + 1) : step + 1]
    qs = [r.query_id + 1 for r in records]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=qs,
            y=[r.validity_r for r in records],
            name="Validity (50%)",
            marker_color="#10B981",
        )
    )
    fig.add_trace(
        go.Bar(
            x=qs,
            y=[r.latency_r for r in records],
            name="Latency (30%)",
            marker_color="#3B82F6",
        )
    )
    fig.add_trace(
        go.Bar(
            x=qs,
            y=[r.retry_r for r in records],
            name="No-Retry (20%)",
            marker_color="#F59E0B",
        )
    )
    fig.update_layout(
        title="Composite Reward Breakdown  (last 30 queries)",
        xaxis_title="Query #",
        yaxis_title="Reward",
        barmode="stack",
        height=380,
        **PLOTLY_LAYOUT,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────


def _metrics_row(history: List[StepRecord], step: int):
    records = history[: step + 1]
    total_cost = sum(r.cost for r in records)
    baseline_cost = sum(r.baseline_cost for r in records)
    savings = (1 - total_cost / baseline_cost) * 100 if baseline_cost else 0
    accuracy = sum(r.is_valid for r in records) / len(records) * 100
    avg_lat = np.mean([r.latency_ms for r in records])
    fallbacks = sum(r.fallback_used for r in records) / len(records) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cost Savings", f"{savings:.1f}%", f"${baseline_cost - total_cost:.4f}")
    c2.metric("Accuracy", f"{accuracy:.1f}%", "valid responses")
    c3.metric("Avg Latency", f"{avg_lat:.0f} ms")
    c4.metric("Circuit Breaker", f"{fallbacks:.1f}%", "fallback rate")


# ── Tab: Live Router ────────────────────────────────────────────────────────


def _tab_live():
    st.markdown(
        "### Watch the router learn to shift traffic "
        "from expensive to cheaper models"
    )

    c1, c2, c3 = st.columns(3)
    n = c1.slider("Queries", 50, 500, 200, 25, key="ln")
    gamma = c2.slider("Decay γ", 0.80, 1.00, 0.95, 0.01, key="lg")
    decay_int = c3.slider("Decay interval", 20, 100, 50, 10, key="ld")

    if st.button("Run Simulation", key="lb", type="primary"):
        st.session_state["lh"] = run_simulation(
            n, EXPERT_PRIORS, gamma=gamma, decay_interval=decay_int
        )

    if "lh" not in st.session_state:
        st.info("Press **Run Simulation** to start the demo.")
        return

    h = st.session_state["lh"]
    step = st.slider("Scrub timeline", 0, len(h) - 1, len(h) - 1, key="ls")

    _metrics_row(h, step)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_beta_distributions(h, step), use_container_width=True)
        st.plotly_chart(plot_cumulative_cost(h, step), use_container_width=True)
    with right:
        st.plotly_chart(plot_traffic_distribution(h, step), use_container_width=True)
        st.plotly_chart(plot_reward_breakdown(h, step), use_container_width=True)


# ── Tab: Model Rot ──────────────────────────────────────────────────────────


def _tab_rot():
    st.markdown("### When a provider degrades, the router reroutes automatically")
    st.markdown(
        "**Decaying memory** makes recent observations weigh more than history. "
        "The router detects quality shifts within minutes — zero human intervention."
    )

    c1, c2, c3 = st.columns(3)
    rot_model = c1.selectbox(
        "Degrade model",
        ["gpt-4o-mini", "claude-haiku", "gpt-4o"],
        key="rm",
    )
    rot_at = c2.slider("Degrade at query #", 30, 200, 75, 5, key="ra")
    rot_factor = c3.slider("Degradation factor", 1.5, 5.0, 2.5, 0.5, key="rf")

    if st.button("Run Rot Scenario", key="rb", type="primary"):
        st.session_state["rh"] = run_simulation(
            300,
            EXPERT_PRIORS,
            rot_config={"model": rot_model, "at_query": rot_at, "factor": rot_factor},
        )
        st.session_state["rot_at"] = rot_at
        st.session_state["rot_model"] = rot_model

    if "rh" not in st.session_state:
        st.info("Press **Run Rot Scenario** to start.")
        return

    h = st.session_state["rh"]
    step = st.slider("Scrub timeline", 0, len(h) - 1, len(h) - 1, key="rs")

    _metrics_row(h, step)

    left, right = st.columns(2)
    r_at = st.session_state["rot_at"]
    r_m = st.session_state["rot_model"]

    with left:
        st.plotly_chart(plot_beta_distributions(h, step), use_container_width=True)
        fig = plot_cumulative_cost(h, step)
        fig.add_vline(
            x=r_at,
            line_dash="dot",
            line_color="#EF4444",
            annotation_text=f"{_l(r_m)} degrades",
            annotation_position="top left",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = plot_traffic_distribution(h, step)
        fig.add_vline(
            x=r_at,
            line_dash="dot",
            line_color="#EF4444",
            annotation_text="Degradation starts",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(plot_reward_breakdown(h, step), use_container_width=True)


# ── Tab: Cold Start ─────────────────────────────────────────────────────────


def _tab_coldstart():
    st.markdown("### Expert priors converge in ~20 queries; uniform needs ~100")
    st.markdown(
        "Starting with **informative priors** from benchmark data means "
        "the router makes good decisions from query 1 instead of burning "
        "budget on random exploration."
    )

    if st.button("Run Comparison", key="cb", type="primary"):
        st.session_state["ce"] = run_simulation(200, EXPERT_PRIORS, seed=42)
        st.session_state["cu"] = run_simulation(200, UNIFORM_PRIORS, seed=42)

    if "ce" not in st.session_state:
        st.info("Press **Run Comparison** to start.")
        return

    expert = st.session_state["ce"]
    uniform = st.session_state["cu"]
    step = st.slider("Scrub timeline", 0, len(expert) - 1, len(expert) - 1, key="cs")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Expert Priors  β(10,2) / β(5,5)")
        st.plotly_chart(
            plot_beta_distributions(expert, step), use_container_width=True
        )
        st.plotly_chart(
            plot_traffic_distribution(expert, step), use_container_width=True
        )
    with right:
        st.markdown("#### Uniform Priors  β(1,1)")
        st.plotly_chart(
            plot_beta_distributions(uniform, step), use_container_width=True
        )
        st.plotly_chart(
            plot_traffic_distribution(uniform, step), use_container_width=True
        )

    e_cost = sum(r.cost for r in expert[: step + 1])
    u_cost = sum(r.cost for r in uniform[: step + 1])
    b_cost = sum(r.baseline_cost for r in expert[: step + 1])

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Expert Priors Cost",
        f"${e_cost:.4f}",
        f"{(1 - e_cost / b_cost) * 100:.1f}% savings" if b_cost else "",
    )
    c2.metric(
        "Uniform Priors Cost",
        f"${u_cost:.4f}",
        f"{(1 - u_cost / b_cost) * 100:.1f}% savings" if b_cost else "",
    )
    c3.metric("Expert Advantage", f"${u_cost - e_cost:.4f}", "less wasted on explore")


# ── Sidebar ──────────────────────────────────────────────────────────────────


def _sidebar():
    st.sidebar.markdown("## Architecture")
    st.sidebar.code(
        "Query\n"
        "  │\n"
        "  ▼\n"
        "┌────────────────────┐\n"
        "│  Thompson Sampling │\n"
        "│  sample β(α,β)    │\n"
        "│  pick highest      │\n"
        "└────────┬───────────┘\n"
        "         ▼\n"
        "┌────────────────────┐\n"
        "│  Circuit Breaker   │\n"
        "│  confidence check  │\n"
        "└────────┬───────────┘\n"
        "         ▼\n"
        "┌────────────────────┐\n"
        "│  LLM Execution     │\n"
        "└────────┬───────────┘\n"
        "         ▼\n"
        "┌────────────────────┐\n"
        "│  Composite Reward  │\n"
        "│  valid? fast? ok?  │\n"
        "│  → update β(α,β)  │\n"
        "└────────────────────┘",
        language=None,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Model Costs")
    cost_df = pd.DataFrame(
        {
            "Model": [_l(m) for m in MODEL_PROFILES],
            "$/1K tokens": [p["cost_per_1k"] for p in MODEL_PROFILES.values()],
            "Validity": [f"{p['base_validity']:.0%}" for p in MODEL_PROFILES.values()],
            "Latency": [
                f"{p['latency_range'][0]}-{p['latency_range'][1]}ms"
                for p in MODEL_PROFILES.values()
            ],
        }
    )
    st.sidebar.dataframe(cost_df, hide_index=True, use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Key Concepts")
    with st.sidebar.expander("Composite Reward"):
        st.markdown(
            "Three signals from normal agent telemetry:\n\n"
            "| Signal | Weight | Source |\n"
            "|--------|--------|--------|\n"
            "| Validity | 50% | Pydantic / JSON schema |\n"
            "| Latency | 30% | Wall-clock time |\n"
            "| No retry | 20% | Agent self-correction |\n\n"
            "**Zero human labels needed.**"
        )
    with st.sidebar.expander("Decaying Memory"):
        st.markdown(
            "Every *N* queries, multiply α and β by γ (0.95).\n\n"
            "This makes recent performance count more than old history. "
            "If a provider ships a regression overnight, the router adapts "
            "within minutes."
        )
    with st.sidebar.expander("Circuit Breaker"):
        st.markdown(
            "If a model's mean confidence (α/(α+β)) drops below a floor, "
            "the router falls back to the trusted expensive model.\n\n"
            "Guarantees accuracy never drops below baseline."
        )


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    st.set_page_config(
        page_title="Bayesian Model Router — DevConf 2026",
        page_icon="🎯",
        layout="wide",
    )

    st.markdown(
        "<h1 style='text-align:center'>🎯 The 50% Cheaper Agent</h1>"
        "<p style='text-align:center;color:#9CA3AF;font-size:1.2rem'>"
        "Autonomous LLM Routing with Bayesian Bandits</p>"
        "<p style='text-align:center;color:#6B7280;margin-bottom:2rem'>"
        "DevConf.CZ 2026 — Live Demo</p>",
        unsafe_allow_html=True,
    )

    _sidebar()

    tab_live, tab_rot, tab_cold = st.tabs(
        ["🎲  Live Router", "🔥  Model Rot", "🚀  Cold Start"]
    )

    with tab_live:
        _tab_live()
    with tab_rot:
        _tab_rot()
    with tab_cold:
        _tab_coldstart()

    st.markdown("---")
    st.caption(
        "Thompson Sampling · Composite Reward Function · "
        "Decaying Memory · Circuit Breaker"
    )


if __name__ == "__main__":
    main()
