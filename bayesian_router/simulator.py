"""Model simulator for testing, benchmarking, and live demos."""

import random
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SimulationProfile:
    """Performance profile for simulating a single model."""

    cost_per_1k: float = 0.001
    base_validity: float = 0.90
    latency_range: tuple = (300, 1000)
    retry_rate: float = 0.08


DEFAULT_PROFILES: Dict[str, SimulationProfile] = {
    "gpt-4o": SimulationProfile(
        cost_per_1k=0.005,
        base_validity=0.96,
        latency_range=(1500, 3500),
        retry_rate=0.04,
    ),
    "gpt-4o-mini": SimulationProfile(
        cost_per_1k=0.00015,
        base_validity=0.89,
        latency_range=(400, 1200),
        retry_rate=0.09,
    ),
    "claude-haiku": SimulationProfile(
        cost_per_1k=0.00025,
        base_validity=0.91,
        latency_range=(300, 900),
        retry_rate=0.07,
    ),
}


class ModelSimulator:
    """Simulates LLM call outcomes with configurable quality / latency.

    Supports dynamic degradation to demonstrate model-rot adaptation.

    Args:
        profiles: ``{model_name: SimulationProfile}``.  ``None`` → built-in
                  defaults for GPT-4o / GPT-4o-mini / Claude Haiku.
    """

    def __init__(self, profiles: Optional[Dict[str, SimulationProfile]] = None):
        self.profiles = profiles or dict(DEFAULT_PROFILES)
        self._degradation: Dict[str, float] = {}

    def call(self, model: str, tokens: int = 500) -> dict:
        """Simulate an LLM call.

        Returns a dict with keys:

        - ``latency_ms``
        - ``validity_score``   (float in [0.0, 1.0])
        - ``retry_count``      (integer)
        - ``cost``
        - ``tokens``

        For backward compatibility with older examples, it also returns:

        - ``is_valid``   (bool alias for ``validity_score == 1.0``)
        - ``retried``    (bool alias for ``retry_count > 0``)
        """
        p = self.profiles[model]
        deg = self._degradation.get(model, 1.0)

        lo, hi = p.latency_range
        latency = random.uniform(lo, hi) * deg

        validity = min(p.base_validity / deg, 1.0)
        is_valid = random.random() < validity
        validity_score = 1.0 if is_valid else 0.0

        retry_rate = min(p.retry_rate * deg, 1.0)
        retried = random.random() < retry_rate
        retry_count = 1 if retried else 0

        cost = (tokens / 1000) * p.cost_per_1k
        return dict(
            latency_ms=latency,
            validity_score=validity_score,
            retry_count=retry_count,
            is_valid=is_valid,
            retried=retried,
            cost=cost,
            tokens=tokens,
        )

    def degrade(self, model: str, factor: float):
        """Simulate model quality degradation (model rot).

        ``factor > 1`` multiplies latency and divides validity.
        """
        self._degradation[model] = factor

    def reset(self, model: str):
        """Remove degradation from a model."""
        self._degradation.pop(model, None)
