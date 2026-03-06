"""Composite reward function — label-free quality signal from agent telemetry."""

import math

from .types import RewardResult


class CompositeReward:
    """Three objective signals, zero human labels.

    +-----------+--------+-------------------------------+
    | Signal    | Weight | Source                        |
    +-----------+--------+-------------------------------+
    | Validity  |  50 %  | Pydantic / JSON schema check  |
    | Latency   |  30 %  | Wall-clock response time      |
    | No-retry  |  20 %  | Agent self-correction loop    |
    +-----------+--------+-------------------------------+

    Args:
        validity_weight:    Weight for the schema-pass signal.
        latency_weight:     Weight for the normalised latency signal.
        retry_weight:       Weight for the no-retry signal.
        latency_midpoint_ms: Sigmoid midpoint (responses faster than this
                             score > 0.5 on the latency dimension).
        latency_steepness:  Sigmoid steepness — smaller = sharper curve.
    """

    def __init__(
        self,
        validity_weight: float = 0.50,
        latency_weight: float = 0.30,
        retry_weight: float = 0.20,
        latency_midpoint_ms: float = 2000.0,
        latency_steepness: float = 600.0,
    ):
        total = validity_weight + latency_weight + retry_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total:.2f}")

        self.validity_weight = validity_weight
        self.latency_weight = latency_weight
        self.retry_weight = retry_weight
        self.latency_midpoint_ms = latency_midpoint_ms
        self.latency_steepness = latency_steepness

    def compute(
        self, latency_ms: float, is_valid: bool, retried: bool
    ) -> RewardResult:
        """Score a single LLM response. Returns a value in [0, 1]."""
        v = self.validity_weight if is_valid else 0.0
        l = self.latency_weight / (
            1.0
            + math.exp(
                (latency_ms - self.latency_midpoint_ms) / self.latency_steepness
            )
        )
        r = self.retry_weight if not retried else 0.0
        return RewardResult(total=v + l + r, validity=v, latency=l, retry=r)
