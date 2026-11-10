"""Tests for the router's failure classification (pure parts of provider_router.py)."""
from provider_router import (BilledFailure, NoResults, OutputRejected, ProviderError, counts_against_breaker,
                             is_out_of_credit, route_failure)


def test_own_402_is_out_of_credit():
    # Arrange
    exc = ProviderError("llm_gateway 402: payment required")
    # Act / Assert
    assert is_out_of_credit(exc) is True


def test_402_inside_other_text_is_not_out_of_credit():
    # Arrange: a 402 that is a scene number, not an HTTP status at the start
    exc = ProviderError("render failed at scene 402: ffmpeg exited 1")
    # Act / Assert
    assert is_out_of_credit(exc) is False


def test_escaped_json_code_402_counts():
    # Arrange
    exc = ProviderError('upstream said {\\"error\\": {\\"code\\": 402}}')
    # Act / Assert
    assert is_out_of_credit(exc) is True


def test_billing_on_429_quota_error_is_not_out_of_credit():
    # Arrange: rate limits reset by themselves; they must not park the provider
    exc = ProviderError("image_api 429: quota exceeded, check your plan and billing details")
    # Act / Assert
    assert is_out_of_credit(exc) is False


def test_billing_on_own_403_is_out_of_credit():
    # Arrange
    exc = ProviderError("video_api 403: billing account disabled")
    # Act / Assert
    assert is_out_of_credit(exc) is True


def test_rejected_output_wording_never_counts():
    # Arrange: model text that happens to mention credits
    exc = OutputRejected("model said: insufficient credits to answer")
    # Act / Assert
    assert is_out_of_credit(exc) is False


def test_explicit_flag_wins():
    # Arrange
    exc = ProviderError("opaque failure", out_of_credit=True)
    # Act / Assert
    assert is_out_of_credit(exc) is True


def test_breaker_counting_rules():
    # Arrange
    cases = {
        "transport bug": (RuntimeError("connection reset"), True),
        "retryable provider error": (ProviderError("http 503"), True),
        "non-retryable provider error": (ProviderError("http 400", retryable=False), False),
        "rejected output": (OutputRejected("blocked"), False),
        "retryable billed failure": (BilledFailure("job failed after acceptance"), True),
        "final billed failure": (BilledFailure("job failed", retryable=False), False),
    }
    # Act
    results = {name: counts_against_breaker(exc) for name, (exc, _) in cases.items()}
    # Assert
    assert results == {name: expected for name, (_, expected) in cases.items()}


def test_route_failure_dispatch():
    # Arrange
    no_results = NoResults("nothing found")
    no_credit = ProviderError("llm_gateway 402")
    outage = ProviderError("http 503")
    rejected = OutputRejected("empty reply")
    # Act
    routes = [route_failure(e) for e in (no_results, no_credit, outage, rejected)]
    # Assert
    assert routes == ["no_results", "out_of_credit", "breaker_failure", "error_only"]
