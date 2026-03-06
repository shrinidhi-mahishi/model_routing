"""Interactive Streamlit demo for DevConf.CZ 2026 talk.

Run:
    pip install -e ".[demo]"
    streamlit run examples/04_streamlit_demo.py
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import beta as beta_dist

from bayesian_router import (
    DEFAULT_PROFILES,
    EXPERT_PRIORS,
    UNIFORM_PRIORS,
    ModelSimulator,
    Router,
)

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY CONSTANTS (colours / labels for charts — not part of the package)
# ─────────────────────────────────────────────────────────────────────────────

DISPLAY = {
    "gpt-4o": {"color": "#10B981", "label": "GPT-4o ($$$)"},
    "gpt-4o-mini": {"color": "#3B82F6", "label": "GPT-4o-mini ($)"},
    "claude-haiku": {"color": "#F59E0B", "label": "Claude Haiku ($)"},
}

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def _c(m: str) -> str:
    return DISPLAY.get(m, {}).get("color", "#888")


def _l(m: str) -> str:
    return DISPLAY.get(m, {}).get("label", m)


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION RECORD
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StepRecord:
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


def run_simulation(
    n_queries: int,
    priors,
    rot_config=None,
    gamma: float = 0.95,
    decay_interval: int = 50,
    seed: int = 42,
) -> List[StepRecord]:
    np.random.seed(seed)
    random.seed(seed)

    router = Router(models=priors, gamma=gamma, decay_interval=decay_interval)
    sim = ModelSimulator()
    baseline_rate = DEFAULT_PROFILES["gpt-4o"].cost_per_1k
    history: List[StepRecord] = []

    for i in range(n_queries):
        if rot_config and i == rot_config.get("at_query"):
            sim.degrade(rot_config["model"], rot_config["factor"])

        result = router.select()
        t = sim.call(result.model)
        reward = router.update(
            result.model,
            latency_ms=t["latency_ms"],
            is_valid=t["is_valid"],
            retried=t["retried"],
        )
        state = router.get_distributions()

        history.append(
            StepRecord(
                query_id=i,
                model_selected=result.model,
                reward=reward.total,
                validity_r=reward.validity,
                latency_r=reward.latency,
                retry_r=reward.retry,
                latency_ms=t["latency_ms"],
                is_valid=t["is_valid"],
                retried=t["retried"],
                cost=t["cost"],
                baseline_cost=(t["tokens"] / 1000) * baseline_rate,
                fallback_used=result.fallback_used,
                alphas={m: s.alpha for m, s in state.items()},
                betas={m: s.beta for m, s in state.items()},
            )
        )
    return history


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def plot_beta_distributions(history, step):
    rec = history[step]
    x = np.linspace(0.001, 0.999, 300)
    fig = go.Figure()
    for m in DISPLAY:
        a, b = rec.alphas[m], rec.betas[m]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=beta_dist.pdf(x, a, b),
                mode="lines",
                name=f"{_l(m)}  α={a:.1f} β={b:.1f}",
                line=dict(color=_c(m), width=3),
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


def plot_traffic(history, step, window=20):
    records = history[: step + 1]
    models = list(DISPLAY.keys())
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


def plot_cost(history, step):
    records = history[: step + 1]
    qs = list(range(1, len(records) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=qs,
            y=list(np.cumsum([r.baseline_cost for r in records])),
            name="Always GPT-4o",
            line=dict(color="#EF4444", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=qs,
            y=list(np.cumsum([r.cost for r in records])),
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


def plot_rewards(history, step, window=30):
    records = history[max(0, step - window + 1) : step + 1]
    qs = [r.query_id + 1 for r in records]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=qs, y=[r.validity_r for r in records], name="Validity (50%)", marker_color="#10B981")
    )
    fig.add_trace(
        go.Bar(x=qs, y=[r.latency_r for r in records], name="Latency (30%)", marker_color="#3B82F6")
    )
    fig.add_trace(
        go.Bar(x=qs, y=[r.retry_r for r in records], name="No-Retry (20%)", marker_color="#F59E0B")
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


def _metrics(history, step):
    recs = history[: step + 1]
    tc = sum(r.cost for r in recs)
    bc = sum(r.baseline_cost for r in recs)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cost Savings", f"{(1 - tc / bc) * 100:.1f}%", f"${bc - tc:.4f}")
    c2.metric("Accuracy", f"{sum(r.is_valid for r in recs) / len(recs) * 100:.1f}%", "valid responses")
    c3.metric("Avg Latency", f"{np.mean([r.latency_ms for r in recs]):.0f} ms")
    c4.metric("Circuit Breaker", f"{sum(r.fallback_used for r in recs) / len(recs) * 100:.1f}%", "fallback rate")


def _tab_live():
    st.markdown("### Watch the router learn to shift traffic from expensive to cheaper models")
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Queries", 50, 500, 200, 25, key="ln")
    gamma = c2.slider("Decay γ", 0.80, 1.00, 0.95, 0.01, key="lg")
    d_int = c3.slider("Decay interval", 20, 100, 50, 10, key="ld")
    if st.button("Run Simulation", key="lb", type="primary"):
        st.session_state["lh"] = run_simulation(n, EXPERT_PRIORS, gamma=gamma, decay_interval=d_int)
    if "lh" not in st.session_state:
        st.info("Press **Run Simulation** to start the demo.")
        return
    h = st.session_state["lh"]
    step = st.slider("Scrub timeline", 0, len(h) - 1, len(h) - 1, key="ls")
    _metrics(h, step)
    l, r = st.columns(2)
    with l:
        st.plotly_chart(plot_beta_distributions(h, step), use_container_width=True)
        st.plotly_chart(plot_cost(h, step), use_container_width=True)
    with r:
        st.plotly_chart(plot_traffic(h, step), use_container_width=True)
        st.plotly_chart(plot_rewards(h, step), use_container_width=True)


def _tab_rot():
    st.markdown("### When a provider degrades, the router reroutes automatically")
    st.markdown(
        "**Decaying memory** makes recent observations weigh more than history. "
        "The router detects quality shifts within minutes — zero human intervention."
    )
    c1, c2, c3 = st.columns(3)
    rot_m = c1.selectbox("Degrade model", ["gpt-4o-mini", "claude-haiku", "gpt-4o"], key="rm")
    rot_at = c2.slider("Degrade at query #", 30, 200, 75, 5, key="ra")
    rot_f = c3.slider("Degradation factor", 1.5, 5.0, 2.5, 0.5, key="rf")
    if st.button("Run Rot Scenario", key="rb", type="primary"):
        st.session_state["rh"] = run_simulation(
            300, EXPERT_PRIORS, rot_config={"model": rot_m, "at_query": rot_at, "factor": rot_f}
        )
        st.session_state["rot_at"] = rot_at
        st.session_state["rot_m"] = rot_m
    if "rh" not in st.session_state:
        st.info("Press **Run Rot Scenario** to start.")
        return
    h = st.session_state["rh"]
    step = st.slider("Scrub timeline", 0, len(h) - 1, len(h) - 1, key="rs")
    _metrics(h, step)
    r_at = st.session_state["rot_at"]
    l, r = st.columns(2)
    with l:
        st.plotly_chart(plot_beta_distributions(h, step), use_container_width=True)
        fig = plot_cost(h, step)
        fig.add_vline(x=r_at, line_dash="dot", line_color="#EF4444",
                      annotation_text=f"{_l(st.session_state['rot_m'])} degrades",
                      annotation_position="top left")
        st.plotly_chart(fig, use_container_width=True)
    with r:
        fig = plot_traffic(h, step)
        fig.add_vline(x=r_at, line_dash="dot", line_color="#EF4444",
                      annotation_text="Degradation starts")
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(plot_rewards(h, step), use_container_width=True)


def _tab_cold():
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
    expert, uniform = st.session_state["ce"], st.session_state["cu"]
    step = st.slider("Scrub timeline", 0, len(expert) - 1, len(expert) - 1, key="cs")
    l, r = st.columns(2)
    with l:
        st.markdown("#### Expert Priors  β(8,3) / β(5,4)")
        st.plotly_chart(plot_beta_distributions(expert, step), use_container_width=True)
        st.plotly_chart(plot_traffic(expert, step), use_container_width=True)
    with r:
        st.markdown("#### Uniform Priors  β(1,1)")
        st.plotly_chart(plot_beta_distributions(uniform, step), use_container_width=True)
        st.plotly_chart(plot_traffic(uniform, step), use_container_width=True)
    ec = sum(rr.cost for rr in expert[: step + 1])
    uc = sum(rr.cost for rr in uniform[: step + 1])
    bc = sum(rr.baseline_cost for rr in expert[: step + 1])
    c1, c2, c3 = st.columns(3)
    c1.metric("Expert Cost", f"${ec:.4f}", f"{(1 - ec / bc) * 100:.1f}% savings" if bc else "")
    c2.metric("Uniform Cost", f"${uc:.4f}", f"{(1 - uc / bc) * 100:.1f}% savings" if bc else "")
    c3.metric("Expert Advantage", f"${uc - ec:.4f}", "less wasted on explore")


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
    rows = []
    for m, p in DEFAULT_PROFILES.items():
        rows.append({
            "Model": _l(m),
            "$/1K tok": p.cost_per_1k,
            "Validity": f"{p.base_validity:.0%}",
            "Latency": f"{p.latency_range[0]}-{p.latency_range[1]}ms",
        })
    st.sidebar.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
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
            "Recent performance counts more than old history. "
            "If a provider ships a regression overnight, the router adapts within minutes."
        )
    with st.sidebar.expander("Circuit Breaker"):
        st.markdown(
            "If a model's mean confidence (α/(α+β)) drops below a floor, "
            "the router falls back to the trusted expensive model.\n\n"
            "Guarantees accuracy never drops below baseline."
        )


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
    t1, t2, t3 = st.tabs(["🎲  Live Router", "🔥  Model Rot", "🚀  Cold Start"])
    with t1:
        _tab_live()
    with t2:
        _tab_rot()
    with t3:
        _tab_cold()
    st.markdown("---")
    st.caption("Thompson Sampling · Composite Reward Function · Decaying Memory · Circuit Breaker")


if __name__ == "__main__":
    main()
