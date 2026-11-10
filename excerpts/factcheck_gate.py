"""Fact-check gate between script and voice: 2-of-3 judge votes per claim, then REWRITE or CUT.

Excerpt from two Video Studio modules:
- `studio.stages.claims`     -> vote bookkeeping (`_fails`, `_undecided`, `_decided`, `flipped`) and the
                                voting loop of `confirm`;
- `studio.stages.factcheck`  -> deciding, per confirmed failure, whether the next repair pass should
                                REWRITE the sentence from its cited passage or CUT it (`actions`).

How the gate runs: research stores Wikipedia / Wikidata text as numbered passages. An LLM extracts every
checkable claim from the script and a judge marks each one supported / unsupported / contradicted against
those passages only. A claim that fails vote 1 gets vote 2 over a fresh source pull; if votes 1 and 2
disagree, vote 3 decides. Confirmed failures go back to the script model once per pass, constrained to the
sources; a bounded number of passes later the video either passes or is held for a human. Nothing is
voiced until the gate passes.

Removed for the excerpt: the LLM calls (`extract`, `verify`, the fresh Wikipedia pull) -- `confirm` takes
a `revote` callable instead of building those requests, and vote-1 records here omit the `cited` passages
and `pull` tag the original attaches; the exact-number guard; storage, events and the
operator alert. The decision logic is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

VERDICTS = ("supported", "unsupported", "contradicted")
VOTES = 3        # at most three independent judge votes per claim
FAIL_VOTES = 2   # a claim fails only when this many of its votes fail
_VOTE_KEYS = ("verdict", "note", "passage_ids", "source_urls")


# --- vote bookkeeping (studio.stages.claims) ------------------------------------------------------


def _fails(claim: dict) -> int:
    """Number of the claim's votes that are not `supported`."""
    return sum(v["verdict"] != "supported" for v in claim["votes"])


def _undecided(claim: dict) -> bool:
    """True when a failed vote exists but neither FAIL_VOTES failures nor VOTES votes are reached yet."""
    return 0 < _fails(claim) < FAIL_VOTES and len(claim["votes"]) < VOTES


def _decided(claim: dict) -> dict:
    """The claim with the verdict, note, passages and URLs of its last vote that has the majority outcome."""
    supported = _fails(claim) < FAIL_VOTES
    last = next(v for v in reversed(claim["votes"]) if (v["verdict"] == "supported") == supported)
    return claim | {k: last[k] for k in _VOTE_KEYS}


def flipped(claim: dict) -> bool:
    """True when the claim's votes hold both a supported and a not-supported verdict."""
    verdicts = {v["verdict"] == "supported" for v in claim.get("votes", [])}
    return verdicts == {True, False}


def confirm(checked: list[dict], revote: Callable[[list[dict], int], list[dict]]) -> list[dict]:
    """Settle every failed claim by independent votes: it fails only when FAIL_VOTES of VOTES votes fail.

    Args:
        checked: Vote-1 claims (each with `verdict`, `note`, `passage_ids`, `source_urls`).
        revote: `revote(todo, vote_no)` returns one new vote dict per claim in `todo`. In the original this
            re-judges over a fresh source pull with a different seed and a higher temperature, so no vote
            replays another from the provider cache.

    Returns:
        The claims in order, each with `votes` and the fields of its last majority vote.
    """
    voted = [c | {"votes": [{k: c[k] for k in _VOTE_KEYS}]} for c in checked]
    for vote_no in range(2, VOTES + 1):
        todo = [c for c in voted if _undecided(c)]
        if not todo:
            break
        for claim, vote in zip(todo, revote(todo, vote_no), strict=True):
            claim["votes"].append(vote)
    return [_decided(c) for c in voted]


# --- repair actions (studio.stages.factcheck) -----------------------------------------------------


def _failing_cited(claim: dict) -> list[dict] | None:
    """The passages the claim's failing votes cited, de-duplicated by text in vote order; None when the
    claim has no vote records."""
    votes = claim.get("votes")
    if not votes:
        return None
    unique: dict[str, dict] = {}
    for vote in (v for v in votes if v["verdict"] != "supported"):
        for passage in vote.get("cited", []):
            unique.setdefault(passage["text"], passage)
    return list(unique.values())


def _cut_reason(claim: dict, *, last: bool = False,
                rewritten_scenes: frozenset[int] | set[int] = frozenset()) -> str | None:
    """Why a confirmed failure is CUT rather than rewritten, or None when it is rewritten.

    Returns:
        "no source" (no failing vote cited a passage), "already rewritten once" (its scene was rewritten or
        sent for repair and fails again), "last rewrite" (no further pass will check a rewrite), or None.
    """
    cited = _failing_cited(claim)
    if not (claim.get("passage_ids") if cited is None else cited):
        return "no source"
    if claim.get("scene_idx") in rewritten_scenes:
        return "already rewritten once"
    return "last rewrite" if last else None


def actions(failing: list[dict], *, last: bool = False,
            rewritten_scenes: frozenset[int] | set[int] = frozenset()) -> list[dict]:
    """The claims with the repair action each not-supported claim needs; supported claims pass unchanged.

    The rule this encodes: a rewrite is only allowed when a cited passage exists to restate and a later
    pass will check it. Otherwise the sentence is cut, so the gate can't loop on a claim the sources don't
    support.
    """
    out = []
    for c in failing:
        if c["verdict"] == "supported":
            out.append(c)
            continue
        reason = _cut_reason(c, last=last, rewritten_scenes=rewritten_scenes)
        out.append(c | {"action": "cut" if reason else "rewrite", "cut_reason": reason})
    return out
