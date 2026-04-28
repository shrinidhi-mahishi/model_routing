"""Composite reward function — label-free quality signal from agent telemetry."""

import math

from .types import RewardResult


class CompositeReward:
    """Three objective signals, zero human labels.

    +-----------+--------+-----------------------------------------------+
    | Signal    | Weight | Source                                        |
    +-----------+--------+-----------------------------------------------+
    | Validity  |  50 %  | Continuous quality score in [0.0, 1.0]        |
    | Latency   |  30 %  | Wall-clock response time (sigmoid-normalised) |
    | No-retry  |  20 %  | Agent self-correction count (graduated)       |
    +-----------+--------+-----------------------------------------------+

    ``validity_score`` accepts a continuous float in ``[0.0, 1.0]``:

    - ``validity_score=1.0`` → full validity credit (e.g. schema passed cleanly)
    - ``validity_score=0.6`` → partial credit (e.g. 6 of 10 required fields found)
    - ``validity_score=0.0`` → no validity credit (hard failure)

    ``retry_count`` is a non-negative integer (the raw telemetry value from your
    agent loop or orchestration layer). It is converted internally to a graduated
    credit using ``max(0, 1 - retry_count / 3)``:

    - retry_count=0 → 1.00 × retry_weight  (full credit, no retry)
    - retry_count=1 → 0.67 × retry_weight
    - retry_count=2 → 0.33 × retry_weight
    - retry_count=3+ → 0.00 × retry_weight  (no credit)

    Args:
        validity_weight:    Weight for the validity signal.
        latency_weight:     Weight for the normalised latency signal.
        retry_weight:       Weight for the retry signal.
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
        self,
        latency_ms: float,
        validity_score: float,
        retry_count: int,
    ) -> RewardResult:
        """Score a single LLM response.  Returns a value in [0, 1].

        Args:
            latency_ms:     Wall-clock response latency in milliseconds.
            validity_score: Continuous quality score in **[0.0, 1.0]**.
                            1.0 = fully valid output (e.g. schema passed).
                            0.0 = complete failure.
                            Values in between give partial credit
                            (e.g. ``fields_found / fields_expected``).
            retry_count:    Number of retries the agent performed (non-negative
                            integer from your orchestration layer or validator).
                            0 = no retry (full credit).
                            1 = one retry (~2/3 credit).
                            2 = two retries (~1/3 credit).
                            3+ = no retry credit.
        """
        if not 0.0 <= validity_score <= 1.0:
            raise ValueError(
                f"validity_score must be in [0.0, 1.0], got {validity_score}"
            )
        if retry_count < 0:
            raise ValueError(
                f"retry_count must be >= 0, got {retry_count}"
            )

        # Validity: linear partial credit — supports richer signals than binary pass/fail.
        v = self.validity_weight * validity_score

        # Latency: smooth sigmoid — penalises slow responses without a hard cutoff.
        l = self.latency_weight / (
            1.0
            + math.exp(
                (latency_ms - self.latency_midpoint_ms) / self.latency_steepness
            )
        )

        # Retry: graduated penalty. Each extra retry costs ~1/3 of retry credit.
        # Formula: max(0, 1 - retry_count / 3) * retry_weight
        retry_fraction = max(0.0, 1.0 - retry_count / 3.0)
        r = self.retry_weight * retry_fraction

        return RewardResult(total=v + l + r, validity=v, latency=l, retry=r)
