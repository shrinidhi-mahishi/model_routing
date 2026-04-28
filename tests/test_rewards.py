import pytest

from bayesian_router import CompositeReward


def test_perfect_response_scores_high():
    reward = CompositeReward()
    result = reward.compute(latency_ms=100, validity_score=1.0, retry_count=0)
    assert result.total > 0.90


def test_invalid_response_loses_validity():
    reward = CompositeReward()
    result = reward.compute(latency_ms=100, validity_score=0.0, retry_count=0)
    assert result.validity == 0.0
    assert result.total < 0.55


def test_high_latency_penalised():
    reward = CompositeReward()
    fast = reward.compute(latency_ms=100, validity_score=1.0, retry_count=0)
    slow = reward.compute(latency_ms=10_000, validity_score=1.0, retry_count=0)
    assert fast.latency > slow.latency


def test_retry_penalised():
    reward = CompositeReward()
    ok  = reward.compute(latency_ms=500, validity_score=1.0, retry_count=0)
    bad = reward.compute(latency_ms=500, validity_score=1.0, retry_count=3)
    assert ok.retry > bad.retry
    assert bad.retry == 0.0


def test_graduated_retry_penalty():
    reward = CompositeReward()
    r0 = reward.compute(latency_ms=500, validity_score=1.0, retry_count=0)
    r1 = reward.compute(latency_ms=500, validity_score=1.0, retry_count=1)
    r2 = reward.compute(latency_ms=500, validity_score=1.0, retry_count=2)
    r3 = reward.compute(latency_ms=500, validity_score=1.0, retry_count=3)
    r9 = reward.compute(latency_ms=500, validity_score=1.0, retry_count=9)

    # each retry reduces credit; 3+ gives zero
    assert r0.retry == pytest.approx(0.20)
    assert r1.retry == pytest.approx(0.20 * 2 / 3)
    assert r2.retry == pytest.approx(0.20 * 1 / 3)
    assert r3.retry == 0.0
    assert r9.retry == 0.0
    assert r0.total > r1.total > r2.total > r3.total


def test_partial_validity_gives_partial_credit():
    reward = CompositeReward()
    full = reward.compute(latency_ms=500, validity_score=1.0, retry_count=0)
    half = reward.compute(latency_ms=500, validity_score=0.5, retry_count=0)
    zero = reward.compute(latency_ms=500, validity_score=0.0, retry_count=0)

    # validity component scales linearly with validity_score
    assert half.validity == pytest.approx(full.validity * 0.5)
    assert zero.validity == 0.0
    assert zero.total < half.total < full.total


def test_custom_weights():
    reward = CompositeReward(
        validity_weight=0.80, latency_weight=0.10, retry_weight=0.10
    )
    result = reward.compute(latency_ms=500, validity_score=1.0, retry_count=0)
    assert result.validity == pytest.approx(0.80)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        CompositeReward(validity_weight=0.5, latency_weight=0.5, retry_weight=0.5)


def test_validity_score_out_of_range():
    reward = CompositeReward()
    with pytest.raises(ValueError, match="validity_score"):
        reward.compute(latency_ms=500, validity_score=1.5, retry_count=0)
    with pytest.raises(ValueError, match="validity_score"):
        reward.compute(latency_ms=500, validity_score=-0.1, retry_count=0)


def test_negative_retry_count_rejected():
    reward = CompositeReward()
    with pytest.raises(ValueError, match="retry_count"):
        reward.compute(latency_ms=500, validity_score=1.0, retry_count=-1)


def test_reward_bounded_zero_one():
    reward = CompositeReward()
    for latency in [0, 500, 2000, 5000, 20_000]:
        for validity_score in [0.0, 0.5, 1.0]:
            for retry_count in [0, 1, 2, 3, 10]:
                r = reward.compute(
                    latency_ms=latency,
                    validity_score=validity_score,
                    retry_count=retry_count,
                )
                assert 0.0 <= r.total <= 1.0, f"Out of range: {r}"
