"""Capability router guards: failure classification, a cross-process circuit breaker, out-of-credit parking.

Excerpt from three Video Studio modules:
- `studio.providers.base`    -> the error types and `is_out_of_credit` (pure, unit-tested);
- `studio.providers.router`  -> `counts_against_breaker` and the failure dispatch of `Router._call_in`;
- `studio.providers.guards`  -> `breaker_allows`, `_record_failure`, `mark_out_of_credit` (SQL).

Stages never call a vendor. They ask for a capability (`text.script`, `voice.narrate`, `footage.search`, ...)
and the router walks that capability's connector chain from a YAML routing file: skip disabled / parked /
over-cap / over-budget connectors, take a rate-limit token, ask the breaker, call, validate, cache, log cost.
Free or local connectors sit in each chain, so when paid ones are parked the chain falls through to them.

Removed for the excerpt: the full `Router` class (budget reservation, rate limiter, cache, cost logging,
work-dir handling), event emission and secret redaction, the once-a-day operator alert on out-of-credit,
and the vendor names in `_STATUS_SOURCES` (replaced with generic ones). `route_failure` is the body of the
`except` branch in `Router._call_in`, lifted into a function so it can be tested; its branches are unchanged.
"""

from __future__ import annotations

import re

# --- error types (studio.providers.base) ----------------------------------------------------------


class ProviderError(Exception):
    """A connector call failed. `retryable=False` means trying again with the same input is pointless."""

    def __init__(self, msg: str, *, retryable: bool = True, out_of_credit: bool = False):
        super().__init__(msg)
        self.retryable, self.out_of_credit = retryable, out_of_credit


class OutputRejected(ProviderError):
    """The call succeeded but the output is unusable (blocked, empty, wrong shape). Never trips the breaker."""

    def __init__(self, msg: str, *, retryable: bool = True, cost_usd: float = 0.0, model: str = ""):
        super().__init__(msg, retryable=retryable)
        self.cost_usd, self.model = cost_usd, model


class BilledFailure(OutputRejected):
    """A paid job the provider accepted (and bills) failed afterwards: poll, result or download.

    Unlike other rejections a retryable BilledFailure counts against the circuit breaker, so a provider
    outage that keeps failing after acceptance stops costing a new job on every call.
    """


class NoResults(ProviderError):
    """A search connector found nothing: a normal answer, not a provider fault."""

    def __init__(self, msg: str):
        super().__init__(msg, retryable=False)


# Connectors report an HTTP failure as "<source> <status>[: <body>]" at the start of the message. Only
# these sources count, so a 402 inside other text ("scene 402: render", "line 402") never does.
_STATUS_SOURCES = r"(?:llm_gateway|http|image_api|video_api|tts_api)"
_OWN_402 = re.compile(rf"^{_STATUS_SOURCES} 402\b", re.IGNORECASE)
# A provider's own JSON error code, also when the JSON is embedded as an escaped string (\"code\": 402).
_JSON_402 = re.compile(r'\\*"code\\*"\s*:\s*402\b')
_CREDIT_WORDS = re.compile(
    r"credits? (?:are |is |have been )?(?:depleted|exhausted)|insufficient (?:credits?|balance)"
    r"|payment required|credit balance (?:is )?too low",
    re.IGNORECASE,
)
# "billing" alone is weak: it counts only on a connector's own non-429 4xx (some 429 quota errors say
# "check your plan and billing details"; a 5xx "billing service unavailable" is an outage).
_OWN_4XX = re.compile(r"^[\w.-]+(?: [\w.-]+)? 4(?!29)\d\d\b")
_BILLING = re.compile(r"\bbilling\b", re.IGNORECASE)
_RATE_LIMITED = re.compile(r"\b429\b|rate.?limit|quota", re.IGNORECASE)


def is_out_of_credit(exc: BaseException) -> bool:
    """Return True when `exc` means the provider account has run out of paid credit.

    Such a failure is not the provider being flaky: retrying is pointless until someone tops up, so the
    router parks the provider instead of counting it against the circuit breaker. Subscription "usage
    limit" messages are deliberately NOT matched (they reset on their own within hours). An
    OutputRejected carries model text, so only its explicit flag counts, never its wording.
    """
    if getattr(exc, "out_of_credit", False) is True:
        return True
    if isinstance(exc, OutputRejected):
        return False
    text = str(exc)
    if _OWN_402.search(text) or _JSON_402.search(text) or _CREDIT_WORDS.search(text):
        return True
    return bool(_OWN_4XX.search(text) and _BILLING.search(text)) and not _RATE_LIMITED.search(text)


# --- failure dispatch (studio.providers.router) ---------------------------------------------------


def counts_against_breaker(exc: Exception) -> bool:
    """Whether a failed call counts as a breaker failure.

    True for a non-ProviderError (a bug or transport fault), a retryable BilledFailure, or a retryable
    ProviderError that is not an OutputRejected.
    """
    if not isinstance(exc, ProviderError):
        return True
    if isinstance(exc, BilledFailure):
        return exc.retryable
    return exc.retryable and not isinstance(exc, OutputRejected)


def route_failure(exc: Exception) -> str:
    """What the router does with a failed connector call before moving to the next connector.

    Returns:
        "no_results"      -> info event only; the breaker is not touched;
        "out_of_credit"   -> park the provider for hours (`mark_out_of_credit`), alert once a day;
        "breaker_failure" -> record an error and count one breaker failure (`breaker_record(ok=False)`);
        "error_only"      -> record an error, breaker untouched (e.g. a rejected, unusable reply).
    """
    if isinstance(exc, NoResults):
        return "no_results"
    if is_out_of_credit(exc):
        return "out_of_credit"
    return "breaker_failure" if counts_against_breaker(exc) else "error_only"


# --- DB-backed breaker shared by all worker processes (studio.providers.guards) -------------------

_PROBE_WINDOW_S = 60  # how long a claimed half-open probe blocks other callers
OUT_OF_CREDIT_PREFIX = "out_of_credit:"
OUT_OF_CREDIT_HOURS = 6


def breaker_allows(conn, provider: str) -> bool:
    """Decide whether a call to `provider` may go ahead.

    While the circuit is open and its cooldown runs, every caller is refused. Once the cooldown has passed
    the circuit half-opens: exactly one caller (across all processes) atomically claims the probe by pushing
    `opened_until` forward; the state stays 'open' until the probe's outcome is recorded.
    """
    row = conn.execute("SELECT state FROM provider_health WHERE provider=%s", (provider,)).fetchone()
    if row is None or row["state"] != "open":
        return True
    claimed = conn.execute(
        "UPDATE provider_health SET opened_until = now() + make_interval(secs => %s), updated_at=now()"
        " WHERE provider=%s AND state='open' AND (opened_until IS NULL OR opened_until <= now())"
        " RETURNING 1",
        (_PROBE_WINDOW_S, provider),
    ).fetchone()
    return claimed is not None


def record_failure(conn, provider: str, error: str | None, threshold: int = 5, cooldown_s: int = 600) -> str:
    """Atomically count one failure; open the circuit at `threshold`. An open circuit is never downgraded,
    and a failure on it (e.g. a failed half-open probe) re-arms the full cooldown."""
    params = {"p": provider, "e": error, "t": threshold, "c": cooldown_s}
    with conn.transaction():
        row = conn.execute(
            "INSERT INTO provider_health AS h (provider, state, failures, last_error, opened_until)"
            " VALUES (%(p)s, CASE WHEN 1 >= %(t)s THEN 'open' ELSE 'degraded' END, 1, %(e)s,"
            "  CASE WHEN 1 >= %(t)s THEN now() + make_interval(secs => %(c)s) END)"
            " ON CONFLICT (provider) DO UPDATE SET failures = h.failures + 1,"
            "  last_error = COALESCE(EXCLUDED.last_error, h.last_error),"
            "  state = CASE WHEN h.state = 'open' OR h.failures + 1 >= %(t)s THEN 'open' ELSE 'degraded' END,"
            "  opened_until = CASE WHEN h.state = 'open' OR h.failures + 1 >= %(t)s"
            "   THEN GREATEST(h.opened_until, now() + make_interval(secs => %(c)s)) END,"
            "  updated_at = now()"
            " RETURNING state, failures",
            params,
        ).fetchone()
    return row["state"]


def mark_out_of_credit(conn, provider: str, error: str, hours: int = OUT_OF_CREDIT_HOURS) -> None:
    """Park `provider` for `hours`: its circuit opens without counting a breaker failure.

    `last_error` is prefixed `out_of_credit:` so readers (the router's pre-check, the doctor command,
    the UI) can tell a parked provider from a flaky one. When the park ends, one half-open probe is let
    through; success resets it, another out-of-credit failure re-arms the full park.
    """
    stored = (OUT_OF_CREDIT_PREFIX + " " + error)[:500]
    conn.execute(
        "INSERT INTO provider_health (provider, state, failures, opened_until, last_error)"
        " VALUES (%(p)s, 'open', 0, now() + make_interval(hours => %(h)s), %(e)s)"
        " ON CONFLICT (provider) DO UPDATE SET state='open', last_error=EXCLUDED.last_error,"
        "  opened_until=EXCLUDED.opened_until, updated_at=now()",
        {"p": provider, "h": hours, "e": stored},
    )
