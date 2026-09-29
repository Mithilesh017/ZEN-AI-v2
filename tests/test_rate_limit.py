"""Tests for the sliding-window rate limiter."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rate_limit import RateLimiter


def test_allows_calls_up_to_the_limit():
    limiter = RateLimiter([(3, 60)])
    assert [limiter.check("a", now=1.0) for _ in range(3)] == [None, None, None]


def test_blocks_once_the_limit_is_reached():
    limiter = RateLimiter([(3, 60)])
    for _ in range(3):
        limiter.check("a", now=1.0)
    assert limiter.check("a", now=1.0) == 61


def test_limits_are_per_key():
    limiter = RateLimiter([(1, 60)])
    assert limiter.check("a", now=1.0) is None
    assert limiter.check("b", now=1.0) is None
    assert limiter.check("a", now=1.0) is not None


def test_window_slides_so_calls_are_allowed_again():
    limiter = RateLimiter([(2, 60)])
    limiter.check("a", now=1.0)
    limiter.check("a", now=2.0)
    assert limiter.check("a", now=30.0) is not None
    # The first call ages out at t=61, the second at t=62.
    assert limiter.check("a", now=61.5) is None
    assert limiter.check("a", now=61.5) is not None


def test_rejected_calls_do_not_extend_the_penalty():
    limiter = RateLimiter([(1, 60)])
    limiter.check("a", now=1.0)
    for moment in (10.0, 20.0, 30.0):
        assert limiter.check("a", now=moment) is not None
    assert limiter.check("a", now=61.5) is None


def test_every_rule_applies():
    limiter = RateLimiter([(5, 60), (6, 3600)])
    for minute in range(6):
        assert limiter.check("a", now=minute * 61.0) is None
    # Under the per-minute rule, but the hourly one is now exhausted.
    retry_after = limiter.check("a", now=6 * 61.0)
    assert retry_after is not None
    assert retry_after > 3000


def test_retry_after_counts_down_as_the_window_passes():
    limiter = RateLimiter([(1, 60)])
    limiter.check("a", now=0.0)
    assert limiter.check("a", now=10.0) > limiter.check("a", now=50.0)


def test_idle_keys_are_swept():
    limiter = RateLimiter([(1, 60)])
    limiter.check("gone", now=1.0)
    limiter.check("here", now=1000.0)      # triggers a sweep
    assert "gone" not in limiter._hits


def test_rejects_invalid_rules():
    for rules in ([], [(0, 60)], [(1, 0)]):
        with pytest.raises(ValueError):
            RateLimiter(rules)
