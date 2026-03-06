"""Ready-made prior configurations for common scenarios."""

from .types import ModelConfig

EXPERT_PRIORS: dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(alpha=8, beta=3, cost_per_1k=0.005),
    "gpt-4o-mini": ModelConfig(alpha=5, beta=4, cost_per_1k=0.00015),
    "claude-haiku": ModelConfig(alpha=5, beta=4, cost_per_1k=0.00025),
}
"""Priors informed by public benchmarks — converge in ~20 queries."""

UNIFORM_PRIORS: dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(alpha=1, beta=1, cost_per_1k=0.005),
    "gpt-4o-mini": ModelConfig(alpha=1, beta=1, cost_per_1k=0.00015),
    "claude-haiku": ModelConfig(alpha=1, beta=1, cost_per_1k=0.00025),
}
"""Flat Beta(1,1) — maximum initial uncertainty, needs ~100 queries."""
