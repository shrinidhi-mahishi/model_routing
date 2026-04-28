"""Bayesian model router using Thompson Sampling."""

import random
from collections import deque
from typing import Dict, Optional

import numpy as np

from .rewards import CompositeReward
from .types import ModelConfig, ModelState, RewardResult, RoutingResult


class Router:
    """Bayesian Multi-Armed Bandit for LLM model routing.

    Uses Thompson Sampling with four production-grade additions:

    1. **Composite Reward** — 3 objective signals, 0 human labels.
    2. **Decaying Memory** — adapts to non-stationary model performance
       (*model rot*) by exponentially discounting old observations.
    3. **Automated Circuit Breakers** — track recent failures per model and
       move them through closed/open/half-open states.
    4. **Shadow Evaluation** — mirrors a fraction of requests to a shadow
       model so alternatives can be evaluated without serving user traffic.

    Args:
        models:           ``{name: ModelConfig}`` dict.  ``None`` → built-in
                          defaults for GPT-4o / GPT-4o-mini / Claude Haiku.
        reward_fn:        Custom :class:`CompositeReward`, or ``None`` for
                          default weights.
        gamma:            Decay factor applied every *decay_interval* queries.
        decay_interval:   How often (in queries) to apply memory decay.
        confidence_floor: Minimum mean confidence for a model to be served.
                          Below this the router falls back to
                          *fallback_model*.
        shadow_rate:      Fraction of primary requests mirrored to a shadow
                          model for hidden evaluation.
        fallback_model:   Name of the trusted model for circuit-breaker
                          fallback.  Defaults to the first model in *models*.
        circuit_window_size: Number of recent outcomes tracked per model.
        circuit_failure_threshold: Failures inside the recent window needed
                          to open the breaker.
        circuit_reset_queries: Number of completed primary queries to wait
                          before moving an open breaker to half-open.
        half_open_max_requests: Successful half-open probes required to
                          close the breaker again.
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
        shadow_rate: float = 0.05,
        fallback_model: Optional[str] = None,
        circuit_window_size: int = 5,
        circuit_failure_threshold: int = 3,
        circuit_reset_queries: int = 20,
        half_open_max_requests: int = 2,
    ):
        configs = models or self.DEFAULT_MODELS
        if not configs:
            raise ValueError("Router requires at least one model")
        if not 0.0 <= shadow_rate <= 1.0:
            raise ValueError("shadow_rate must be between 0 and 1")
        if decay_interval <= 0:
            raise ValueError("decay_interval must be positive")
        if circuit_window_size <= 0:
            raise ValueError("circuit_window_size must be positive")
        if circuit_failure_threshold <= 0:
            raise ValueError("circuit_failure_threshold must be positive")
        if circuit_failure_threshold > circuit_window_size:
            raise ValueError(
                "circuit_failure_threshold cannot exceed circuit_window_size"
            )
        if circuit_reset_queries <= 0:
            raise ValueError("circuit_reset_queries must be positive")
        if half_open_max_requests <= 0:
            raise ValueError("half_open_max_requests must be positive")

        self._state: Dict[str, dict] = {
            name: {
                "alpha": c.alpha,
                "beta": c.beta,
                "cost": c.cost_per_1k,
                "selections": 0,
                "shadow_selections": 0,
                "recent_failures": deque(maxlen=circuit_window_size),
                "circuit_state": "closed",
                "circuit_reopen_query": None,
                "half_open_successes": 0,
                "half_open_attempts": 0,
            }
            for name, c in configs.items()
        }
        self.reward_fn = reward_fn or CompositeReward()
        self.gamma = gamma
        self.decay_interval = decay_interval
        self.confidence_floor = confidence_floor
        self.shadow_rate = shadow_rate
        self.fallback_model = fallback_model or next(iter(self._state))
        if self.fallback_model not in self._state:
            raise ValueError(f"Unknown fallback_model: {self.fallback_model}")
        self.circuit_window_size = circuit_window_size
        self.circuit_failure_threshold = circuit_failure_threshold
        self.circuit_reset_queries = circuit_reset_queries
        self.half_open_max_requests = half_open_max_requests
        self._query_count = 0

    # ── selection ────────────────────────────────────────────────────────

    def select(self) -> RoutingResult:
        """Pick a model via Thompson Sampling.

        Returns a :class:`RoutingResult` with the chosen model name and
        whether the router fell back to the trusted model.  A shadow model may
        also be returned for background evaluation on the same request.
        """
        self._advance_circuit_breakers()

        candidates = self._primary_candidates()
        samples = {
            name: np.random.beta(self._state[name]["alpha"], self._state[name]["beta"])
            for name in candidates
        }
        selected = max(samples, key=samples.get)
        served_model = selected
        fallback_used = False
        selection_reason = "thompson"

        if (
            len(candidates) == 1
            and candidates[0] == self.fallback_model
            and any(
                state["circuit_state"] != "closed"
                for name, state in self._state.items()
                if name != self.fallback_model
            )
        ):
            fallback_used = True
            selection_reason = "circuit_open"

        conf = self._confidence(selected)
        if conf < self.confidence_floor and selected != self.fallback_model:
            served_model = self.fallback_model
            fallback_used = True
            selection_reason = "confidence_floor"

        self._state[served_model]["selections"] += 1

        preferred_shadow = (
            selected if fallback_used and selected != served_model else None
        )
        shadow_model = self._choose_shadow_model(
            primary_model=served_model,
            preferred_model=preferred_shadow,
        )
        if shadow_model is not None:
            self._state[shadow_model]["shadow_selections"] += 1

        return RoutingResult(
            model=served_model,
            fallback_used=fallback_used,
            shadow_model=shadow_model,
            selection_reason=selection_reason,
        )

    # ── update ───────────────────────────────────────────────────────────

    def update(
        self,
        model: str,
        *,
        latency_ms: float,
        validity_score: float,
        retry_count: int,
    ) -> RewardResult:
        """Update the model's Beta distribution with observed telemetry.

        Args:
            model:          Name of the model that served the request.
            latency_ms:     Wall-clock response latency in milliseconds.
            validity_score: Continuous quality score in [0.0, 1.0].
                            1.0 = fully valid; 0.0 = complete failure;
                            intermediate values give partial credit.
            retry_count:    Number of retries performed (non-negative integer).
                            0 = no retry (full credit); 3+ = no retry credit.

        Returns the :class:`RewardResult` breakdown so callers can log it.
        """
        return self._record_observation(
            model,
            latency_ms=latency_ms,
            validity_score=validity_score,
            retry_count=retry_count,
            count_as_query=True,
        )

    def update_shadow(
        self,
        model: str,
        *,
        latency_ms: float,
        validity_score: float,
        retry_count: int,
    ) -> RewardResult:
        """Update the mirrored shadow model without incrementing query count.

        Args:
            model:          Name of the shadow model.
            latency_ms:     Wall-clock response latency in milliseconds.
            validity_score: Continuous quality score in [0.0, 1.0].
            retry_count:    Number of retries performed (non-negative integer).
        """
        return self._record_observation(
            model,
            latency_ms=latency_ms,
            validity_score=validity_score,
            retry_count=retry_count,
            count_as_query=False,
        )

    def _record_observation(
        self,
        model: str,
        *,
        latency_ms: float,
        validity_score: float,
        retry_count: int,
        count_as_query: bool,
    ) -> RewardResult:
        self._ensure_model(model)
        result = self.reward_fn.compute(latency_ms, validity_score, retry_count)
        self._state[model]["alpha"] += result.total
        self._state[model]["beta"] += 1 - result.total

        if count_as_query:
            self._query_count += 1
            self._update_circuit_breaker(
                model, validity_score=validity_score, retry_count=retry_count
            )
            if self._query_count % self.decay_interval == 0:
                self._decay()
        else:
            self._update_circuit_breaker(
                model, validity_score=validity_score, retry_count=retry_count
            )

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
                shadow_selections=s["shadow_selections"],
                circuit_state=s["circuit_state"],
                recent_failures=sum(1 for failed in s["recent_failures"] if failed),
            )
            for name, s in self._state.items()
        }

    def get_stats(self) -> dict:
        """Summary statistics dict (JSON-serialisable)."""
        total = sum(s["selections"] for s in self._state.values())
        total_shadow = sum(s["shadow_selections"] for s in self._state.values())
        return {
            "total_queries": self._query_count,
            "total_selections": total,
            "total_shadow_selections": total_shadow,
            "model_share": {
                name: s["selections"] / total if total else 0
                for name, s in self._state.items()
            },
            "shadow_share": {
                name: s["shadow_selections"] / total_shadow if total_shadow else 0
                for name, s in self._state.items()
            },
            "distributions": {
                name: f"α={s['alpha']:.1f} β={s['beta']:.1f}"
                for name, s in self._state.items()
            },
            "circuit_state": {
                name: s["circuit_state"] for name, s in self._state.items()
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

    def _primary_candidates(self) -> list[str]:
        candidates = [
            name
            for name, state in self._state.items()
            if state["circuit_state"] == "closed"
        ]
        return candidates or [self.fallback_model]

    def _choose_shadow_model(
        self, primary_model: str, preferred_model: Optional[str] = None
    ) -> Optional[str]:
        half_open_candidates = [
            name
            for name, state in self._state.items()
            if state["circuit_state"] == "half_open" and name != primary_model
        ]
        if half_open_candidates:
            return min(half_open_candidates, key=self._shadow_priority)

        if (
            preferred_model
            and preferred_model != primary_model
            and self._state[preferred_model]["circuit_state"] != "open"
        ):
            return preferred_model

        if self.shadow_rate <= 0 or random.random() >= self.shadow_rate:
            return None

        candidates = [
            name
            for name, state in self._state.items()
            if name != primary_model and state["circuit_state"] == "closed"
        ]
        if not candidates:
            return None

        return min(candidates, key=self._shadow_priority)

    def _shadow_priority(self, model: str) -> tuple[int, int, float]:
        state = self._state[model]
        return (
            state["shadow_selections"],
            state["selections"],
            state["alpha"] + state["beta"],
        )

    def _update_circuit_breaker(
        self, model: str, *, validity_score: float, retry_count: int
    ) -> None:
        state = self._state[model]
        # A response is treated as a circuit-breaker failure when either:
        # - validity_score < 0.5 (output was not good enough), or
        # - retry_count > 0    (agent had to retry at all)
        failure = (validity_score < 0.5) or (retry_count > 0)

        if state["circuit_state"] == "open":
            return

        if state["circuit_state"] == "half_open":
            state["half_open_attempts"] += 1
            if failure:
                self._open_circuit(model)
                return

            state["half_open_successes"] += 1
            if state["half_open_successes"] >= self.half_open_max_requests:
                self._close_circuit(model)
            return

        state["recent_failures"].append(failure)
        if sum(1 for failed in state["recent_failures"] if failed) >= (
            self.circuit_failure_threshold
        ):
            self._open_circuit(model)

    def _advance_circuit_breakers(self) -> None:
        for state in self._state.values():
            if (
                state["circuit_state"] == "open"
                and state["circuit_reopen_query"] is not None
                and self._query_count >= state["circuit_reopen_query"]
            ):
                state["circuit_state"] = "half_open"
                state["half_open_successes"] = 0
                state["half_open_attempts"] = 0

    def _open_circuit(self, model: str) -> None:
        state = self._state[model]
        state["circuit_state"] = "open"
        state["circuit_reopen_query"] = self._query_count + self.circuit_reset_queries
        state["half_open_successes"] = 0
        state["half_open_attempts"] = 0
        state["recent_failures"].clear()

    def _close_circuit(self, model: str) -> None:
        state = self._state[model]
        state["circuit_state"] = "closed"
        state["circuit_reopen_query"] = None
        state["half_open_successes"] = 0
        state["half_open_attempts"] = 0
        state["recent_failures"].clear()

    def _ensure_model(self, model: str) -> None:
        if model not in self._state:
            raise KeyError(f"Unknown model: {model}")
