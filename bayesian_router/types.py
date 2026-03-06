"""Data models for the bayesian_router package."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Prior and cost configuration for a single model.

    Attributes:
        alpha:  Initial α for Beta(α, β). Higher = stronger quality belief.
        beta:   Initial β for Beta(α, β). Higher = stronger uncertainty.
        cost_per_1k: Cost per 1 000 tokens (used for reporting, not routing).
    """

    alpha: float = 1.0
    beta: float = 1.0
    cost_per_1k: float = 0.001


@dataclass
class RoutingResult:
    """Returned by ``Router.select()``."""

    model: str
    fallback_used: bool = False


@dataclass
class RewardResult:
    """Composite reward breakdown returned by ``Router.update()``."""

    total: float
    validity: float
    latency: float
    retry: float


@dataclass
class ModelState:
    """Snapshot of a model's Beta distribution at query time."""

    alpha: float
    beta: float
    confidence: float
    selections: int
