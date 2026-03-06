"""Bayesian Router — autonomous LLM model routing with Thompson Sampling.

Cut LLM API costs by 40-70 % with < 1 % accuracy drop and zero human labels.
"""

from .presets import EXPERT_PRIORS, UNIFORM_PRIORS
from .rewards import CompositeReward
from .router import Router
from .simulator import DEFAULT_PROFILES, ModelSimulator, SimulationProfile
from .types import ModelConfig, ModelState, RewardResult, RoutingResult

__version__ = "0.1.0"
__all__ = [
    "Router",
    "CompositeReward",
    "ModelConfig",
    "RoutingResult",
    "RewardResult",
    "ModelState",
    "ModelSimulator",
    "SimulationProfile",
    "DEFAULT_PROFILES",
    "EXPERT_PRIORS",
    "UNIFORM_PRIORS",
]
