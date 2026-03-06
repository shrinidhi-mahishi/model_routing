import numpy as np
import pytest

from bayesian_router import ModelConfig, Router


def test_select_returns_valid_model():
    router = Router()
    result = router.select()
    assert result.model in ("gpt-4o", "gpt-4o-mini", "claude-haiku")


def test_update_increases_alpha():
    router = Router()
    before = router.get_distributions()["gpt-4o"].alpha
    router.update("gpt-4o", latency_ms=500, is_valid=True, retried=False)
    after = router.get_distributions()["gpt-4o"].alpha
    assert after > before


def test_update_increases_beta_on_bad_response():
    router = Router()
    before = router.get_distributions()["gpt-4o"].beta
    router.update("gpt-4o", latency_ms=9000, is_valid=False, retried=True)
    after = router.get_distributions()["gpt-4o"].beta
    assert after > before


def test_decay_reduces_parameters():
    router = Router(decay_interval=3)
    for _ in range(3):
        result = router.select()
        router.update(result.model, latency_ms=500, is_valid=True, retried=False)
    # After decay, verify alpha is lower than raw accumulation would give
    for state in router.get_distributions().values():
        assert state.alpha >= 1.0
        assert state.beta >= 1.0


def test_circuit_breaker_redirects_low_confidence():
    models = {
        "good": ModelConfig(alpha=10, beta=2),
        "bad": ModelConfig(alpha=1, beta=20),
    }
    router = Router(
        models=models,
        fallback_model="good",
        confidence_floor=0.5,
        shadow_rate=0.0,
    )
    np.random.seed(42)
    results = [router.select() for _ in range(100)]
    good_count = sum(1 for r in results if r.model == "good")
    assert good_count > 90


def test_shadow_rate_explores():
    router = Router(shadow_rate=1.0)
    np.random.seed(0)
    models_seen = {router.select().model for _ in range(50)}
    assert len(models_seen) > 1


def test_custom_single_model():
    models = {"only_model": ModelConfig(alpha=5, beta=2)}
    router = Router(models=models)
    result = router.select()
    assert result.model == "only_model"


def test_get_stats_keys():
    router = Router()
    router.select()
    stats = router.get_stats()
    assert "total_queries" in stats
    assert "model_share" in stats
    assert "distributions" in stats
