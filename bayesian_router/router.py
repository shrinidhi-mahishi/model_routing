"""Bayesian model router using Thompson Sampling."""

import random
from typing import Dict, Optional

import numpy as np

from .rewards import CompositeReward
from .types import ModelConfig, ModelState, RewardResult, RoutingResult


class Router:
    """Bayesian Multi-Armed Bandit for LLM model routing.

    Uses Thompson Sampling with three production-grade additions:

    1. **Composite Reward** — 3 objective signals, 0 human labels.
    2. **Decaying Memory** — adapts to non-stationary model performance
       (*model rot*) by exponentially discounting old observations.
    3. **Circuit Breaker** — falls back to the trusted (expensive) model
       when a cheaper model's confidence drops below a floor.
    4. **Shadow Evaluation** — forces a fraction of traffic to random
       models so under-sampled alternatives accumulate evidence.

    Args:
        models:           ``{name: ModelConfig}`` dict.  ``None`` → built-in
                          defaults for GPT-4o / GPT-4o-mini / Claude Haiku.
        reward_fn:        Custom :class:`CompositeReward`, or ``None`` for
                          default weights.
        gamma:            Decay factor applied every *decay_interval* queries.
        decay_interval:   How often (in queries) to apply memory decay.
        confidence_floor: Minimum mean confidence for a model to be used.
                          Below this the circuit breaker redirects to
                          *fallback_model*.
        shadow_rate:      Fraction of traffic routed randomly for exploration.
        fallback_model:   Name of the trusted model for circuit-breaker
                          fallback.  Defaults to the first model in *models*.
    """

    DEFAULT_MODELS = {
        "gpt-4o": ModelConfig(alpha=8, beta=3, cost_per_1k=0.005),
        "gpt-4o-mini": ModelConfig(alpha=5, beta=4, cost_per_1k=0.00015),
        "claude-haiku": ModelConfig(alpha=5, beta=4, cost_per_1k=0.00025),
    }

    def __init__(
        self,
        models: Optional[Dict[str, ModelConfig]] = None,
        reward_fn: Optional[CompositeReward] = None,
        gamma: float = 0.95,
        decay_interval: int = 50,
        confidence_floor: float = 0.50,
        shadow_rate: float = 0.10,
        fallback_model: Optional[str] = None,
    ):
        configs = models or self.DEFAULT_MODELS
        self._state: Dict[str, dict] = {
            name: {
                "alpha": c.alpha,
                "beta": c.beta,
                "cost": c.cost_per_1k,
                "selections": 0,
            }
            for name, c in configs.items()
        }
        self.reward_fn = reward_fn or CompositeReward()
        self.gamma = gamma
        self.decay_interval = decay_interval
        self.confidence_floor = confidence_floor
        self.shadow_rate = shadow_rate
        self.fallback_model = fallback_model or next(iter(self._state))
        self._query_count = 0

    # ── selection ────────────────────────────────────────────────────────

    def select(self) -> RoutingResult:
        """Pick a model via Thompson Sampling.

        Returns a :class:`RoutingResult` with the chosen model name and
        whether the circuit breaker redirected to the fallback.
        """
        if random.random() < self.shadow_rate:
            model = random.choice(list(self._state.keys()))
            self._state[model]["selections"] += 1
            return RoutingResult(model=model, fallback_used=False)

        samples = {
            name: np.random.beta(s["alpha"], s["beta"])
            for name, s in self._state.items()
        }
        selected = max(samples, key=samples.get)

        conf = self._confidence(selected)
        if conf < self.confidence_floor and selected != self.fallback_model:
            self._state[self.fallback_model]["selections"] += 1
            return RoutingResult(model=self.fallback_model, fallback_used=True)

        self._state[selected]["selections"] += 1
        return RoutingResult(model=selected, fallback_used=False)

    # ── update ───────────────────────────────────────────────────────────

    def update(
        self,
        model: str,
        *,
        latency_ms: float,
        is_valid: bool,
        retried: bool,
    ) -> RewardResult:
        """Update the model's Beta distribution with observed telemetry.

        Returns the :class:`RewardResult` breakdown so callers can log it.
        """
        result = self.reward_fn.compute(latency_ms, is_valid, retried)
        self._state[model]["alpha"] += result.total
        self._state[model]["beta"] += 1 - result.total

        self._query_count += 1
        if self._query_count % self.decay_interval == 0:
            self._decay()

        return result

    # ── introspection ────────────────────────────────────────────────────

    def get_distributions(self) -> Dict[str, ModelState]:
        """Current Beta distribution state for every model."""
        return {
            name: ModelState(
                alpha=s["alpha"],
                beta=s["beta"],
                confidence=self._confidence(name),
                selections=s["selections"],
            )
            for name, s in self._state.items()
        }

    def get_stats(self) -> dict:
        """Summary statistics dict (JSON-serialisable)."""
        total = sum(s["selections"] for s in self._state.values())
        return {
            "total_queries": self._query_count,
            "total_selections": total,
            "model_share": {
                name: s["selections"] / total if total else 0
                for name, s in self._state.items()
            },
            "distributions": {
                name: f"α={s['alpha']:.1f} β={s['beta']:.1f}"
                for name, s in self._state.items()
            },
        }

    # ── internals ────────────────────────────────────────────────────────

    def _decay(self):
        for s in self._state.values():
            s["alpha"] = max(1.0, s["alpha"] * self.gamma)
            s["beta"] = max(1.0, s["beta"] * self.gamma)

    def _confidence(self, model: str) -> float:
        s = self._state[model]
        return s["alpha"] / (s["alpha"] + s["beta"])
