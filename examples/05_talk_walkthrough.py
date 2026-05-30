"""Notebook walkthrough as a Streamlit app — mirrors bayesian_router_talk_end_to_end.ipynb.

Run:
    pip install -e ".[demo]"
    streamlit run examples/05_talk_walkthrough.py
"""

import copy
import random
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import beta as beta_dist

from bayesian_router import (
    DEFAULT_PROFILES,
    EXPERT_PRIORS,
    UNIFORM_PRIORS,
    CompositeReward,
    ModelConfig,
    ModelSimulator,
    Router,
)

# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "gpt-4o": "#10B981",
    "gpt-4o-mini": "#3B82F6",
    "claude-haiku": "#F59E0B",
}

BASE_FONT_FAMILY = "Inter, Segoe UI, Arial, sans-serif"
BOLD_FONT_FAMILY = "Arial Black, Inter, Segoe UI, Arial, sans-serif"
DARK_LABEL_COLOR = "#111827"

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=21, family=BASE_FONT_FAMILY, color=DARK_LABEL_COLOR),
    margin=dict(l=95, r=40, t=110, b=95),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=18, family=BOLD_FONT_FAMILY, color=DARK_LABEL_COLOR),
    ),
)

PLOTLY_EXPORT_CONFIG = dict(
    displaylogo=False,
    toImageButtonOptions=dict(
        format="png",
        filename="bayesian_router_chart_hd",
        height=1400,
        width=2400,
        scale=3,
    ),
)


def _c(m: str) -> str:
    return MODEL_COLORS.get(m, "#888")


def _as_bold_html(text: Optional[str]) -> Optional[str]:
    """Ensure labels render in bold without double-wrapping."""
    if not text:
        return text
    if "<b>" in text and "</b>" in text:
        return text
    return f"<b>{text}</b>"


def apply_presentation_axes(fig: go.Figure) -> go.Figure:
    """Apply presentation typography and layout to every chart."""
    current_height = fig.layout.height if fig.layout.height is not None else 0

    if fig.layout.title.text:
        fig.update_layout(title=dict(text=_as_bold_html(fig.layout.title.text)))

    fig.update_layout(
        title_font=dict(size=34, family=BOLD_FONT_FAMILY, color=DARK_LABEL_COLOR),
        legend=dict(font=dict(size=18, family=BOLD_FONT_FAMILY, color=DARK_LABEL_COLOR)),
        hoverlabel=dict(font=dict(size=18, family=BASE_FONT_FAMILY)),
        height=max(current_height, 620),
    )

    fig.for_each_xaxis(
        lambda axis: axis.update(
            title=dict(
                text=_as_bold_html(axis.title.text),
                font=dict(size=24, family=BOLD_FONT_FAMILY, color=DARK_LABEL_COLOR),
                standoff=20,
            ),
            tickfont=dict(size=19, family=BASE_FONT_FAMILY, color=DARK_LABEL_COLOR),
            automargin=True,
        )
    )
    fig.for_each_yaxis(
        lambda axis: axis.update(
            title=dict(
                text=_as_bold_html(axis.title.text),
                font=dict(size=24, family=BOLD_FONT_FAMILY, color=DARK_LABEL_COLOR),
                standoff=20,
            ),
            tickfont=dict(size=19, family=BASE_FONT_FAMILY, color=DARK_LABEL_COLOR),
            automargin=True,
        )
    )

    if fig.layout.annotations:
        fig.update_annotations(
            font=dict(size=18, family=BOLD_FONT_FAMILY, color=DARK_LABEL_COLOR)
        )

    return fig


def show_fig(fig: go.Figure):
    """Render charts with presentation-friendly export settings."""
    st.plotly_chart(fig, width="stretch", config=PLOTLY_EXPORT_CONFIG)


# ---------------------------------------------------------------------------
# Helpers (mirror notebook setup cell)
# ---------------------------------------------------------------------------

def clone_priors(p):
    return {k: ModelConfig(alpha=v.alpha, beta=v.beta, cost_per_1k=v.cost_per_1k)
            for k, v in p.items()}


def prior_mean(cfg):
    return cfg.alpha / (cfg.alpha + cfg.beta)


def run_router_simulation(
    *,
    priors,
    num_queries=100,
    seed=42,
    gamma=0.95,
    decay_interval=50,
    confidence_floor=0.50,
    shadow_rate=0.05,
    degradation=None,
    fallback_model=None,
    reward_fn=None,
    tokens=500,
):
    np.random.seed(seed)
    random.seed(seed)

    router = Router(
        models=clone_priors(priors),
        gamma=gamma,
        decay_interval=decay_interval,
        confidence_floor=confidence_floor,
        shadow_rate=shadow_rate,
        fallback_model=fallback_model or "gpt-4o",
        reward_fn=reward_fn,
    )
    sim = ModelSimulator(profiles=copy.deepcopy(DEFAULT_PROFILES))
    records = []

    for q in range(num_queries):
        if degradation and q == degradation["at_query"]:
            sim.degrade(degradation["model"], degradation["factor"])

        result = router.select()
        t = sim.call(result.model, tokens=tokens)
        vs = t.get("validity_score", float(t.get("is_valid", 0)))
        rc = t.get("retry_count", int(t.get("retried", 0)))
        rr = router.update(result.model, latency_ms=t["latency_ms"],
                           validity_score=vs, retry_count=rc)

        shadow_cost = 0.0
        if result.shadow_model:
            st2 = sim.call(result.shadow_model, tokens=tokens)
            svs = st2.get("validity_score", float(st2.get("is_valid", 0)))
            src = st2.get("retry_count", int(st2.get("retried", 0)))
            router.update_shadow(result.shadow_model, latency_ms=st2["latency_ms"],
                                 validity_score=svs, retry_count=src)
            shadow_cost = st2["cost"]

        state = router.get_distributions()
        records.append({
            "query_id": q + 1,
            "served_model": result.model,
            "shadow_model": result.shadow_model,
            "fallback_used": result.fallback_used,
            "selection_reason": result.selection_reason,
            "latency_ms": t["latency_ms"],
            "validity_score": vs,
            "retry_count": rc,
            "reward_total": rr.total,
            "reward_validity": rr.validity,
            "reward_latency": rr.latency,
            "reward_retry": rr.retry,
            "cost": t["cost"],
            "shadow_cost": shadow_cost,
            "total_cost": t["cost"] + shadow_cost,
            "alphas": {m: s.alpha for m, s in state.items()},
            "betas": {m: s.beta for m, s in state.items()},
        })

    return router, records


def run_fixed_policy(model_name, *, num_queries=100, seed=42, reward_fn=None, tokens=500):
    np.random.seed(seed)
    random.seed(seed)
    sim = ModelSimulator(profiles=copy.deepcopy(DEFAULT_PROFILES))
    reward_fn = reward_fn or CompositeReward()
    records = []
    for q in range(num_queries):
        t = sim.call(model_name, tokens=tokens)
        vs = t.get("validity_score", float(t.get("is_valid", 0)))
        rc = t.get("retry_count", int(t.get("retried", 0)))
        rr = reward_fn.compute(latency_ms=t["latency_ms"], validity_score=vs, retry_count=rc)
        records.append({
            "query_id": q + 1,
            "served_model": model_name,
            "latency_ms": t["latency_ms"],
            "validity_score": vs,
            "retry_count": rc,
            "reward_total": rr.total,
            "cost": t["cost"],
        })
    return records


def rolling_share(records, model, window=30):
    shares = []
    for i in range(len(records)):
        win = records[max(0, i - window + 1): i + 1]
        shares.append(sum(1 for r in win if r["served_model"] == model) / len(win))
    return shares


# ---------------------------------------------------------------------------
# Part renderers
# ---------------------------------------------------------------------------

def render_part1():
    st.header("Part 1: Why Model Routing Exists")
    st.markdown("The cost/quality tradeoff across models.")

    profiles = DEFAULT_PROFILES
    models = list(profiles.keys())
    costs = [(profiles[m].cost_per_1k / 1000) * 500 for m in models]
    validities = [profiles[m].base_validity for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=models, y=costs, name="Cost per 500-token request ($)",
                         marker_color=[_c(m) for m in models]))
    fig.update_layout(title="Cost per Request", yaxis_title="Cost ($)", height=350, **PLOTLY_LAYOUT)
    show_fig(apply_presentation_axes(fig))

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=models, y=validities, name="Base Validity",
                          marker_color=[_c(m) for m in models]))
    fig2.update_layout(title="Base Validity", yaxis_title="Validity", yaxis=dict(range=[0, 1.05]),
                       height=350, **PLOTLY_LAYOUT)
    show_fig(apply_presentation_axes(fig2))

    st.info("GPT-4o is ~33x more expensive but only ~16 percentage points more valid. "
            "Routing can exploit this gap.")


def render_part2():
    st.header("Part 2: Composite Reward Without Human Labels")
    st.markdown("Three signals, zero human effort, one score.")

    reward_fn = CompositeReward()

    col1, col2, col3 = st.columns(3)

    with col1:
        latencies = np.linspace(0, 5000, 300)
        lat_scores = [reward_fn.compute(latency_ms=l, validity_score=0.0, retry_count=3).latency
                      for l in latencies]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=latencies, y=lat_scores, mode="lines",
                                 line=dict(color="#3B82F6", width=3)))
        fig.add_vline(x=2000, line_dash="dot", line_color="white", opacity=0.5)
        fig.update_layout(title="Latency Component", xaxis_title="Latency (ms)",
                          yaxis_title="Score", height=300, **PLOTLY_LAYOUT)
        show_fig(apply_presentation_axes(fig))

    with col2:
        vals = np.linspace(0, 1, 200)
        val_scores = [reward_fn.compute(latency_ms=0.0, validity_score=v, retry_count=3).validity
                      for v in vals]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vals, y=val_scores, mode="lines",
                                 line=dict(color="#10B981", width=3)))
        fig.update_layout(title="Validity Component", xaxis_title="Validity Score",
                          yaxis_title="Score", height=300, **PLOTLY_LAYOUT)
        show_fig(apply_presentation_axes(fig))

    with col3:
        retries = list(range(6))
        retry_scores = [reward_fn.compute(latency_ms=0.0, validity_score=0.0, retry_count=r).retry
                        for r in retries]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=retries, y=retry_scores, marker_color="#F59E0B"))
        fig.update_layout(title="Retry Penalty", xaxis_title="Retry Count",
                          yaxis_title="Score", height=300, **PLOTLY_LAYOUT)
        show_fig(apply_presentation_axes(fig))


def render_part3():
    st.header("Part 3: Beta Intuition and Thompson Sampling")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Beta Belief Distributions")
        x = np.linspace(0.001, 0.999, 300)
        fig = go.Figure()
        for m, cfg in EXPERT_PRIORS.items():
            fig.add_trace(go.Scatter(
                x=x, y=beta_dist.pdf(x, cfg.alpha, cfg.beta), mode="lines",
                name=f"{m}  β({cfg.alpha},{cfg.beta})",
                line=dict(color=_c(m), width=3), fill="tozeroy", opacity=0.35,
            ))
        fig.update_layout(title="Expert Prior Beliefs", xaxis_title="Estimated Quality",
                          yaxis_title="Density", height=400, **PLOTLY_LAYOUT)
        show_fig(apply_presentation_axes(fig))

    with col2:
        st.subheader("Thompson Sampling Draws")
        np.random.seed(42)
        rounds = []
        for r in range(8):
            draws = {m: np.random.beta(cfg.alpha, cfg.beta) for m, cfg in EXPERT_PRIORS.items()}
            winner = max(draws, key=draws.get)
            rounds.append({**{f"{m}_draw": f"{v:.3f}" for m, v in draws.items()}, "winner": winner})
        st.dataframe(rounds, use_container_width=True)
        st.caption("Each round: draw one sample per model, pick the highest. "
                   "Strong models usually win, but uncertain ones get a shot.")


def render_part4():
    st.header("Part 4: One Request End to End")
    st.markdown("Watch alpha/beta shift after a single request — including shadow evaluation.")

    reward_fn = CompositeReward(validity_weight=0.70, latency_weight=0.15, retry_weight=0.15)

    np.random.seed(7)
    random.seed(7)
    router = Router(models=clone_priors(EXPERT_PRIORS), gamma=0.95, decay_interval=50,
                    confidence_floor=0.0, shadow_rate=1.0, fallback_model="gpt-4o",
                    reward_fn=reward_fn)
    sim = ModelSimulator()

    before = {m: {"alpha": round(s.alpha, 3), "beta": round(s.beta, 3),
                   "confidence": round(s.alpha / (s.alpha + s.beta), 4)}
              for m, s in router.get_distributions().items()}

    result = router.select()

    # Primary: crafted telemetry for a clean demo → reward ~0.86
    primary_latency = 3400.0
    primary_validity = 1.0
    primary_retries = 0
    rr = router.update(result.model, latency_ms=primary_latency,
                       validity_score=primary_validity, retry_count=primary_retries)

    # Shadow: slightly lower reward → ~0.80 (partial validity + one retry)
    shadow_model = result.shadow_model or "gpt-4o-mini"
    shadow_latency = 650.0
    shadow_validity = 0.8
    shadow_retries = 1
    shadow_rr = router.update_shadow(shadow_model, latency_ms=shadow_latency,
                                     validity_score=shadow_validity, retry_count=shadow_retries)

    after = {m: {"alpha": round(s.alpha, 3), "beta": round(s.beta, 3),
                  "confidence": round(s.alpha / (s.alpha + s.beta), 4)}
             for m, s in router.get_distributions().items()}

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("State Before")
        st.dataframe([{"model": m, **v} for m, v in before.items()], use_container_width=True)
    with col2:
        st.subheader("State After")
        st.dataframe([{"model": m, **v} for m, v in after.items()], use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.subheader(f"Primary: {result.model}")
        st.markdown(f"**Reward:** {rr.total:.4f}  "
                    f"(validity={rr.validity:.3f}, latency={rr.latency:.3f}, retry={rr.retry:.3f})")
        st.markdown(f"Latency: {primary_latency:.0f} ms  |  "
                    f"Validity: {primary_validity}  |  Retries: {primary_retries}")
    with col4:
        st.subheader(f"Shadow: {shadow_model}")
        st.markdown(f"**Reward:** {shadow_rr.total:.4f}  "
                    f"(validity={shadow_rr.validity:.3f}, latency={shadow_rr.latency:.3f}, "
                    f"retry={shadow_rr.retry:.3f})")
        st.markdown(f"Latency: {shadow_latency:.0f} ms  |  "
                    f"Validity: {shadow_validity}  |  Retries: {shadow_retries}")
        st.caption("Shadow model's belief was updated, but the user never sees its response.")


def render_part5():
    st.header("Part 5: Cold Start — Expert vs Uniform Priors")

    PART5_EXPERT = {
        "gpt-4o":       ModelConfig(alpha=25, beta=2, cost_per_1k=0.005),
        "gpt-4o-mini":  ModelConfig(alpha=4,  beta=2, cost_per_1k=0.00015),
        "claude-haiku": ModelConfig(alpha=4,  beta=2, cost_per_1k=0.00025),
    }
    part5_reward = CompositeReward(validity_weight=1.0, latency_weight=0.0, retry_weight=0.0)

    _, expert_recs = run_router_simulation(
        priors=PART5_EXPERT, num_queries=50, seed=17, gamma=1.0,
        confidence_floor=0.0, shadow_rate=0.0, reward_fn=part5_reward,
    )
    _, uniform_recs = run_router_simulation(
        priors=UNIFORM_PRIORS, num_queries=50, seed=17, gamma=1.0,
        confidence_floor=0.0, shadow_rate=0.0, reward_fn=part5_reward,
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for label, recs in [("Expert priors", expert_recs), ("Uniform priors", uniform_recs)]:
            cum = np.cumsum([r["reward_total"] for r in recs])
            fig.add_trace(go.Scatter(x=[r["query_id"] for r in recs], y=cum,
                                     mode="lines", name=label, line=dict(width=3)))
        fig.update_layout(title="Cumulative Reward During Cold Start",
                          xaxis_title="Query #", yaxis_title="Cumulative Reward",
                          height=400, **PLOTLY_LAYOUT)
        show_fig(apply_presentation_axes(fig))

    with col2:
        models = list(PART5_EXPERT.keys())
        exp_counts = Counter(r["served_model"] for r in expert_recs[:20])
        uni_counts = Counter(r["served_model"] for r in uniform_recs[:20])
        x = np.arange(len(models))
        fig = go.Figure()
        fig.add_trace(go.Bar(x=models, y=[exp_counts.get(m, 0) for m in models],
                             name="Expert priors", marker_color="#10B981"))
        fig.add_trace(go.Bar(x=models, y=[uni_counts.get(m, 0) for m in models],
                             name="Uniform priors", marker_color="#F59E0B"))
        fig.update_layout(title="Early Traffic Allocation (first 20 queries)", barmode="group",
                          yaxis_title="Selections", height=400, **PLOTLY_LAYOUT)
        show_fig(apply_presentation_axes(fig))

    st.info("Expert priors send traffic to the stronger model immediately. "
            "Uniform priors waste early queries rediscovering the obvious.")


def render_part6():
    st.header("Part 6: Decay and Model Rot")

    part6_priors = {
        "gpt-4o":       ModelConfig(alpha=20, beta=5, cost_per_1k=0.005),
        "gpt-4o-mini":  ModelConfig(alpha=5,  beta=5, cost_per_1k=0.00015),
        "claude-haiku": ModelConfig(alpha=5,  beta=5, cost_per_1k=0.00025),
    }
    part6_reward = CompositeReward(validity_weight=0.90, latency_weight=0.05, retry_weight=0.05)
    degradation = {"model": "gpt-4o", "factor": 3.0, "at_query": 150}

    _, no_decay_recs = run_router_simulation(
        priors=part6_priors, num_queries=400, seed=287, gamma=1.0,
        confidence_floor=0.0, shadow_rate=0.10, reward_fn=part6_reward,
        degradation=degradation,
    )
    _, decay_recs = run_router_simulation(
        priors=part6_priors, num_queries=400, seed=287, gamma=0.90,
        confidence_floor=0.0, shadow_rate=0.10, reward_fn=part6_reward,
        degradation=degradation,
    )

    # Row 1: Full-timeline cumulative reward + traffic share
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        for label, recs, color, dash in [
            ("No decay (γ=1.0)", no_decay_recs, "#EF4444", "dash"),
            ("With decay (γ=0.9)", decay_recs, "#10B981", "solid"),
        ]:
            cum = np.cumsum([r["reward_total"] for r in recs])
            fig.add_trace(go.Scatter(x=[r["query_id"] for r in recs], y=cum,
                                     mode="lines", name=label,
                                     line=dict(width=3, dash=dash, color=color)))
        fig.add_vline(x=150, line_dash="dot", line_color="white", opacity=0.5,
                      annotation_text="gpt-4o degrades", annotation_font_color="white")
        fig.update_layout(title="Cumulative Reward (full timeline)",
                          xaxis_title="Query #", yaxis_title="Cumulative Reward",
                          height=400, **PLOTLY_LAYOUT)
        show_fig(apply_presentation_axes(fig))

    with col2:
        fig = go.Figure()
        for label, recs, color, dash in [
            ("No decay (γ=1.0)", no_decay_recs, "#EF4444", "dash"),
            ("With decay (γ=0.9)", decay_recs, "#10B981", "solid"),
        ]:
            shares = rolling_share(recs, "gpt-4o", window=30)
            fig.add_trace(go.Scatter(
                x=[r["query_id"] for r in recs], y=[s * 100 for s in shares],
                mode="lines", name=label, line=dict(width=3, dash=dash, color=color),
            ))
        fig.add_vline(x=150, line_dash="dot", line_color="white", opacity=0.5,
                      annotation_text="gpt-4o degrades", annotation_font_color="white")
        fig.update_layout(title="GPT-4o Traffic Share (rolling 30-query window)",
                          xaxis_title="Query #", yaxis_title="% Traffic to gpt-4o",
                          yaxis=dict(range=[0, 105]), height=400, **PLOTLY_LAYOUT)
        show_fig(apply_presentation_axes(fig))

    # Row 2: α/β trajectories + final beliefs
    col3, col4 = st.columns(2)
    with col3:
        fig = go.Figure()
        for label, recs, dash in [("No decay", no_decay_recs, "dash"), ("With decay", decay_recs, "solid")]:
            fig.add_trace(go.Scatter(x=[r["query_id"] for r in recs],
                                     y=[r["alphas"]["gpt-4o"] for r in recs],
                                     mode="lines", name=f"{label} α",
                                     line=dict(width=2, dash=dash, color="#10B981")))
            fig.add_trace(go.Scatter(x=[r["query_id"] for r in recs],
                                     y=[r["betas"]["gpt-4o"] for r in recs],
                                     mode="lines", name=f"{label} β",
                                     line=dict(width=2, dash=dash, color="#EF4444")))
        fig.add_vline(x=150, line_dash="dot", line_color="white", opacity=0.3)
        fig.update_layout(title="GPT-4o α/β Trajectories",
                          xaxis_title="Query #", yaxis_title="Value", height=400, **PLOTLY_LAYOUT)
        show_fig(apply_presentation_axes(fig))

    with col4:
        rng = np.random.default_rng(12345)
        x = np.linspace(0.001, 0.999, 300)
        fig = go.Figure()
        for label, recs, dash in [("No decay", no_decay_recs, "dash"), ("With decay", decay_recs, "solid")]:
            final = recs[-1]
            for m in ["gpt-4o", "gpt-4o-mini", "claude-haiku"]:
                a, b = final["alphas"][m], final["betas"][m]
                fig.add_trace(go.Scatter(
                    x=x, y=beta_dist.pdf(x, a, b), mode="lines",
                    name=f"{label} {m}", line=dict(color=_c(m), width=2, dash=dash),
                ))
        fig.update_layout(title="Final Belief Distributions", xaxis_title="Quality",
                          yaxis_title="Density", height=400, **PLOTLY_LAYOUT)
        show_fig(apply_presentation_axes(fig))

    st.info("Without decay, the router clings to stale history. "
            "With decay (γ=0.90 every 50 queries), traffic moves away from the degraded model.")


def render_part7():
    st.header("Part 7: Safety — Fallback and Shadow Evaluation")

    _, recs = run_router_simulation(
        priors=UNIFORM_PRIORS, num_queries=60, seed=21, gamma=0.95,
        confidence_floor=0.65, shadow_rate=0.25, fallback_model="gpt-4o",
    )

    col1, col2 = st.columns(2)
    with col1:
        fb_rate = []
        for i in range(len(recs)):
            window = recs[:i+1]
            fb_rate.append(sum(1 for r in window if r["fallback_used"]) / len(window) * 100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[r["query_id"] for r in recs], y=fb_rate,
                                 mode="lines", line=dict(color="#EF4444", width=3)))
        fig.update_layout(title="Cumulative Fallback Rate", xaxis_title="Query #",
                          yaxis_title="% Fallback", height=350, **PLOTLY_LAYOUT)
        show_fig(apply_presentation_axes(fig))

    with col2:
        sh_rate = []
        for i in range(len(recs)):
            window = recs[:i+1]
            sh_rate.append(sum(1 for r in window if r["shadow_model"]) / len(window) * 100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[r["query_id"] for r in recs], y=sh_rate,
                                 mode="lines", line=dict(color="#3B82F6", width=3)))
        fig.update_layout(title="Cumulative Shadow Evaluation Rate", xaxis_title="Query #",
                          yaxis_title="% Shadow", height=350, **PLOTLY_LAYOUT)
        show_fig(apply_presentation_axes(fig))

    reasons = Counter(r["selection_reason"] for r in recs)
    st.markdown(f"**Selection reasons:** {dict(reasons)}")

    x = np.linspace(0.001, 0.999, 300)
    final = recs[-1]
    fig = go.Figure()
    for m in MODEL_COLORS:
        a, b = final["alphas"][m], final["betas"][m]
        fig.add_trace(go.Scatter(x=x, y=beta_dist.pdf(x, a, b), mode="lines",
                                 name=f"{m}  α={a:.1f} β={b:.1f}",
                                 line=dict(color=_c(m), width=3), fill="tozeroy", opacity=0.35))
    fig.update_layout(title="Final Beliefs After Fallback + Shadow Learning",
                      xaxis_title="Quality", yaxis_title="Density", height=350, **PLOTLY_LAYOUT)
    show_fig(apply_presentation_axes(fig))


def render_part9():
    st.header("Part 9: Long-Run End-to-End Outcome")

    part9_reward = CompositeReward(validity_weight=0.80, latency_weight=0.1, retry_weight=0.1)
    num_queries = 50

    _, router_recs = run_router_simulation(
        priors=EXPERT_PRIORS, num_queries=num_queries, seed=31, gamma=0.95,
        confidence_floor=0.50, shadow_rate=0.05, fallback_model="gpt-4o",
        reward_fn=part9_reward,
    )
    gpt4o_recs = run_fixed_policy("gpt-4o", num_queries=num_queries, seed=31, reward_fn=part9_reward)
    mini_recs = run_fixed_policy("gpt-4o-mini", num_queries=num_queries, seed=31, reward_fn=part9_reward)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[r["query_id"] for r in gpt4o_recs],
            y=list(np.cumsum([r["cost"] for r in gpt4o_recs])),
            name="Always gpt-4o", line=dict(color="#EF4444", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=[r["query_id"] for r in mini_recs],
            y=list(np.cumsum([r["cost"] for r in mini_recs])),
            name="Always gpt-4o-mini", line=dict(color="#3B82F6", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=[r["query_id"] for r in router_recs],
            y=list(np.cumsum([r["total_cost"] for r in router_recs])),
            name="Bayesian router", line=dict(color="#10B981", width=3),
        ))
        fig.update_layout(title="Cumulative Cost", xaxis_title="Query #",
                          yaxis_title="Cost ($)", height=400, **PLOTLY_LAYOUT)
        show_fig(apply_presentation_axes(fig))

    with col2:
        policies = ["Always gpt-4o", "Always gpt-4o-mini", "Bayesian router"]
        avg_val = [
            np.mean([r["validity_score"] for r in gpt4o_recs]),
            np.mean([r["validity_score"] for r in mini_recs]),
            np.mean([r["validity_score"] for r in router_recs]),
        ]
        avg_rwd = [
            np.mean([r["reward_total"] for r in gpt4o_recs]),
            np.mean([r["reward_total"] for r in mini_recs]),
            np.mean([r["reward_total"] for r in router_recs]),
        ]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=policies, y=avg_val, name="Avg Validity", marker_color="#3B82F6"))
        fig.add_trace(go.Bar(x=policies, y=avg_rwd, name="Avg Composite Reward", marker_color="#F59E0B"))
        fig.update_layout(title="Validity vs Composite Reward", barmode="group",
                          yaxis_title="Score", height=400, **PLOTLY_LAYOUT)
        show_fig(apply_presentation_axes(fig))

    gpt4o_cost = sum(r["cost"] for r in gpt4o_recs)
    router_cost = sum(r["total_cost"] for r in router_recs)
    savings = (1 - router_cost / gpt4o_cost) * 100

    traffic = Counter(r["served_model"] for r in router_recs)
    shares = {m: f"{c / len(router_recs):.0%}" for m, c in traffic.items()}

    c1, c2, c3 = st.columns(3)
    c1.metric("Cost Savings vs Always gpt-4o", f"{savings:.1f}%")
    c2.metric("Router Avg Validity", f"{avg_val[2]:.3f}")
    c3.metric("Traffic Split", " / ".join(f"{m}: {s}" for m, s in shares.items()))


def render_part10():
    st.header("Part 10: Contextual Routing (Future Work)")

    context_profiles = {
        "short": {"gpt-4o": 0.55, "gpt-4o-mini": 0.95, "claude-haiku": 0.78},
        "long":  {"gpt-4o": 0.92, "gpt-4o-mini": 0.72, "claude-haiku": 0.70},
    }
    context_mix = {"short": 0.55, "long": 0.45}
    models = ["gpt-4o", "gpt-4o-mini", "claude-haiku"]

    random.seed(445)
    np.random.seed(445)

    global_beliefs = {m: {"alpha": 1.0, "beta": 1.0} for m in models}
    ctx_beliefs = {ctx: {m: {"alpha": 1.0, "beta": 1.0} for m in models}
                   for ctx in context_profiles}
    g_records, c_records = [], []

    for q in range(500):
        ctx = random.choices(list(context_mix.keys()), weights=list(context_mix.values()))[0]

        g_draws = {m: np.random.beta(global_beliefs[m]["alpha"], global_beliefs[m]["beta"])
                   for m in models}
        g_choice = max(g_draws, key=g_draws.get)
        g_reward = 1 if random.random() < context_profiles[ctx][g_choice] else 0
        global_beliefs[g_choice]["alpha"] += g_reward
        global_beliefs[g_choice]["beta"] += 1 - g_reward
        g_records.append({"query": q + 1, "context": ctx, "model": g_choice, "reward": g_reward})

        c_draws = {m: np.random.beta(ctx_beliefs[ctx][m]["alpha"], ctx_beliefs[ctx][m]["beta"])
                   for m in models}
        c_choice = max(c_draws, key=c_draws.get)
        c_reward = 1 if random.random() < context_profiles[ctx][c_choice] else 0
        ctx_beliefs[ctx][c_choice]["alpha"] += c_reward
        ctx_beliefs[ctx][c_choice]["beta"] += 1 - c_reward
        c_records.append({"query": q + 1, "context": ctx, "model": c_choice, "reward": c_reward})

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, 501)),
                                 y=list(np.cumsum([r["reward"] for r in g_records])),
                                 mode="lines", name="Non-contextual router",
                                 line=dict(width=3, color="#EF4444")))
        fig.add_trace(go.Scatter(x=list(range(1, 501)),
                                 y=list(np.cumsum([r["reward"] for r in c_records])),
                                 mode="lines", name="Contextual router",
                                 line=dict(width=3, color="#10B981")))
        fig.update_layout(title="Cumulative Reward: Non-contextual vs Contextual",
                          xaxis_title="Query #", yaxis_title="Cumulative Reward",
                          height=400, **PLOTLY_LAYOUT)
        show_fig(apply_presentation_axes(fig))

    with col2:
        last200_g = [r for r in g_records[-200:]]
        last200_c = [r for r in c_records[-200:]]
        ctx_names = ["short", "long"]
        g_avgs, c_avgs = [], []
        for ctx_name in ctx_names:
            g_ctx = [r for r in last200_g if r["context"] == ctx_name]
            c_ctx = [r for r in last200_c if r["context"] == ctx_name]
            g_avgs.append(np.mean([r["reward"] for r in g_ctx]) if g_ctx else 0)
            c_avgs.append(np.mean([r["reward"] for r in c_ctx]) if c_ctx else 0)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ctx_names, y=g_avgs, name="Non-contextual router",
                             marker_color="#EF4444"))
        fig.add_trace(go.Bar(x=ctx_names, y=c_avgs, name="Contextual router",
                             marker_color="#10B981"))
        fig.update_layout(title="Avg Reward by Context (last 200 queries)", barmode="group",
                          xaxis_title="Context Type", yaxis_title="Avg Reward",
                          height=400, **PLOTLY_LAYOUT)
        show_fig(apply_presentation_axes(fig))

    last200_g_short = Counter(r["model"] for r in last200_g if r["context"] == "short")
    last200_g_long = Counter(r["model"] for r in last200_g if r["context"] == "long")
    last200_c_short = Counter(r["model"] for r in last200_c if r["context"] == "short")
    last200_c_long = Counter(r["model"] for r in last200_c if r["context"] == "long")

    col3, col4 = st.columns(2)
    for col, ctx_name, g_cnt, c_cnt in [(col3, "short", last200_g_short, last200_c_short),
                                         (col4, "long", last200_g_long, last200_c_long)]:
        with col:
            total_g = sum(g_cnt.values()) or 1
            total_c = sum(c_cnt.values()) or 1
            fig = go.Figure()
            fig.add_trace(go.Bar(x=models, y=[g_cnt.get(m, 0) / total_g * 100 for m in models],
                                 name="Non-contextual", marker_color="#EF4444"))
            fig.add_trace(go.Bar(x=models, y=[c_cnt.get(m, 0) / total_c * 100 for m in models],
                                 name="Contextual", marker_color="#10B981"))
            fig.update_layout(title=f"Model Selection — {ctx_name} queries (last 200)",
                              barmode="group", yaxis_title="% Share", height=350, **PLOTLY_LAYOUT)
            show_fig(apply_presentation_axes(fig))

    st.info("Contextual routing learns different preferences per query type. "
            "Short queries go to gpt-4o-mini (fast, cheap, accurate enough). "
            "Long queries go to gpt-4o (higher quality needed). "
            "A non-contextual router can't make this distinction.")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Bayesian Router — Talk Walkthrough",
        page_icon="📊",
        layout="wide",
    )
    st.markdown(
        "<h1 style='text-align:center'>📊 Bayesian Router — Talk Walkthrough</h1>"
        "<p style='text-align:center;color:#9CA3AF;font-size:1.1rem'>"
        "All notebook outputs rendered as interactive charts</p>",
        unsafe_allow_html=True,
    )

    parts = {
        "Part 1: Why Model Routing Exists": render_part1,
        "Part 2: Composite Reward": render_part2,
        "Part 3: Beta Intuition & Thompson Sampling": render_part3,
        "Part 4: One Request End to End": render_part4,
        "Part 5: Cold Start — Expert vs Uniform": render_part5,
        "Part 6: Decay and Model Rot": render_part6,
        "Part 7: Fallback & Shadow Evaluation": render_part7,
        "Part 9: Long-Run Results": render_part9,
        "Part 10: Contextual Routing (Future)": render_part10,
    }

    tabs = st.tabs(list(parts.keys()))
    for tab, (name, renderer) in zip(tabs, parts.items()):
        with tab:
            renderer()

    st.markdown("---")
    st.caption("Source: bayesian_router_talk_end_to_end.ipynb · DevConf.CZ 2026")


if __name__ == "__main__":
    main()
