import pytest

from bayesian_router import CompositeReward


def test_perfect_response_scores_high():
    reward = CompositeReward()
    result = reward.compute(latency_ms=100, is_valid=True, retried=False)
    assert result.total > 0.90


def test_invalid_response_loses_validity():
    reward = CompositeReward()
    result = reward.compute(latency_ms=100, is_valid=False, retried=False)
    assert result.validity == 0.0
    assert result.total < 0.55


def test_high_latency_penalised():
    reward = CompositeReward()
    fast = reward.compute(latency_ms=100, is_valid=True, retried=False)
    slow = reward.compute(latency_ms=10_000, is_valid=True, retried=False)
    assert fast.latency > slow.latency


def test_retry_penalised():
    reward = CompositeReward()
    ok = reward.compute(latency_ms=500, is_valid=True, retried=False)
    bad = reward.compute(latency_ms=500, is_valid=True, retried=True)
    assert ok.retry > bad.retry
    assert bad.retry == 0.0


def test_custom_weights():
    reward = CompositeReward(
        validity_weight=0.80, latency_weight=0.10, retry_weight=0.10
    )
    result = reward.compute(latency_ms=500, is_valid=True, retried=False)
    assert result.validity == 0.80


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        CompositeReward(validity_weight=0.5, latency_weight=0.5, retry_weight=0.5)


def test_reward_bounded_zero_one():
    reward = CompositeReward()
    for latency in [0, 500, 2000, 5000, 20_000]:
        for valid in [True, False]:
            for retried in [True, False]:
                r = reward.compute(latency_ms=latency, is_valid=valid, retried=retried)
                assert 0.0 <= r.total <= 1.0, f"Out of range: {r}"
