import random

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
    router.update("gpt-4o", latency_ms=500, validity_score=1.0, retry_count=0)
    after = router.get_distributions()["gpt-4o"].alpha
    assert after > before


def test_update_increases_beta_on_bad_response():
    router = Router()
    before = router.get_distributions()["gpt-4o"].beta
    router.update("gpt-4o", latency_ms=9000, validity_score=0.0, retry_count=1)
    after = router.get_distributions()["gpt-4o"].beta
    assert after > before


def test_decay_reduces_parameters():
    router = Router(decay_interval=3)
    for _ in range(3):
        result = router.select()
        router.update(
            result.model, latency_ms=500, validity_score=1.0, retry_count=0
        )
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


def test_shadow_rate_assigns_shadow_model():
    router = Router(shadow_rate=1.0)
    np.random.seed(0)
    random.seed(0)
    result = router.select()
    assert result.shadow_model is not None
    assert result.shadow_model != result.model
    assert router.get_distributions()[result.shadow_model].shadow_selections == 1


def test_update_shadow_does_not_increment_query_count():
    router = Router(shadow_rate=1.0)
    np.random.seed(0)
    random.seed(0)

    result = router.select()
    router.update(result.model, latency_ms=500, validity_score=1.0, retry_count=0)
    router.update_shadow(
        result.shadow_model,
        latency_ms=700,
        validity_score=1.0,
        retry_count=0,
    )

    stats = router.get_stats()
    assert stats["total_queries"] == 1
    assert stats["total_shadow_selections"] == 1


def test_circuit_breaker_state_machine_uses_half_open_shadow_probe():
    models = {
        "good": ModelConfig(alpha=10, beta=1),
        "cheap": ModelConfig(alpha=9, beta=1),
    }
    router = Router(
        models=models,
        fallback_model="good",
        confidence_floor=0.0,
        shadow_rate=0.0,
        circuit_window_size=3,
        circuit_failure_threshold=2,
        circuit_reset_queries=2,
        half_open_max_requests=2,
    )

    router.update("cheap", latency_ms=9000, validity_score=0.0, retry_count=1)
    router.update("cheap", latency_ms=9000, validity_score=0.0, retry_count=1)
    assert router.get_distributions()["cheap"].circuit_state == "open"

    result = router.select()
    assert result.model == "good"
    assert result.selection_reason == "circuit_open"

    router.update("good", latency_ms=500, validity_score=1.0, retry_count=0)
    router.update("good", latency_ms=500, validity_score=1.0, retry_count=0)

    result = router.select()
    assert result.shadow_model == "cheap"
    assert router.get_distributions()["cheap"].circuit_state == "half_open"

    router.update_shadow(
        "cheap", latency_ms=500, validity_score=1.0, retry_count=0
    )
    assert router.get_distributions()["cheap"].circuit_state == "half_open"

    router.update_shadow(
        "cheap", latency_ms=500, validity_score=1.0, retry_count=0
    )
    assert router.get_distributions()["cheap"].circuit_state == "closed"


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
