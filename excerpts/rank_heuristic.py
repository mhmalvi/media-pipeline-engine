"""Excerpt from the clip-engine module `rank_heuristic` (copied whole; only the docstring paragraph
that quoted measurements on a private test fixture was shortened, and the 7-of-10 figure in the
weights comment was removed). Pure stdlib; unit-tested in tests/test_rank_heuristic.py.

Deterministic segment scoring — the floor beneath LLM ranking.

This exists so the pipeline never hard-stops when the LLM is unavailable. It is
intentionally simple and fully deterministic: identical input always produces
identical order, which is what makes it testable.

Five cheap, continuous signals feed the score so that real transcripts spread
out instead of clumping onto a handful of discrete values:

- hook words (count + how early the first one lands)
- speaking pace, scored against a natural-rate peak rather than a saturating
  threshold
- question / exclamation presence, weighted by how early it lands
- numeric specificity (concrete numbers read as more watchable claims)
- the strength of the first sentence alone, since that sentence *is* the
  ``hook`` field shown to a human later — a two-word fragment there should
  drag the score down even if the rest of the block is fine

In practice hook words and numbers do most of the separating on real transcripts; the other
three signals mostly break ties. (Whisper rarely emits "?" or "!" in conversational speech, so
the punctuation channel only earns its keep on scripted or captioned input.)
"""
from __future__ import annotations

import math
import re

HOOK_WORDS = (
    "secret", "mistake", "never", "always", "biggest", "worst", "best",
    "why", "how", "stop", "nobody", "everyone", "truth", "actually",
    "wrong", "instead", "here's", "this is",
)

# Concrete numbers ("three things", "40%") read as specific, watchable claims
# rather than vague filler — spelled out or digits, either counts.
NUMBER_WORDS = frozenset({
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "dozen", "hundred", "thousand", "million", "billion", "percent",
})
_NUMERIC_PATTERN = re.compile(r"\d+(?:\.\d+)?%?")

# Typical conversational English runs ~2.5-3 words/sec; 2.75 is the midpoint
# used as the peak of the density curve below.
_DENSITY_PEAK = 2.75
_DENSITY_SIGMA = 1.3

# Component weight caps — must sum to 100.
#
# The three first-sentence terms sum to 30, the largest bucket, because the
# `hook` field IS the first sentence and a clip that opens on a fragment
# loses the viewer in the first two seconds. Note what that does and does not
# buy: 22 of those 30 points need a hook word or a question mark IN the
# opening sentence, so an ordinary complete opening collects only the
# 8-point completeness term. The bucket is the
# largest by cap, not by typical contribution; on real speech `density` (20)
# and hook words (25) contribute more.
_HOOK_COUNT_CAP = 15.0
_HOOK_POSITION_CAP = 10.0
_DENSITY_CAP = 20.0
_QUESTION_CAP = 10.0
_EXCLAIM_CAP = 5.0
_NUMERIC_CAP = 10.0
_FIRST_SENTENCE_HOOK_CAP = 14.0
_FIRST_SENTENCE_QUESTION_CAP = 8.0
_FIRST_SENTENCE_COMPLETE_CAP = 8.0
_FIRST_SENTENCE_COMPLETE_WORDS = 6.0


def _first_sentence(text: str) -> str:
    """Return the first sentence of ``text`` (empty string if ``text`` is empty).

    Args:
        text: Full segment text.

    Returns:
        The text up to and including the first sentence-ending punctuation,
        or the whole string if none is found.
    """
    if not text:
        return ""
    return re.split(r"(?<=[.!?])\s+", text)[0]


def _breakdown(text: str, start: float, end: float) -> dict:
    """Compute every scoring component for one segment.

    Args:
        text: Non-empty, stripped segment text.
        start: Segment start time in seconds.
        end: Segment end time in seconds.

    Returns:
        A dict with each named component's contribution plus ``"total"``.
        Every component is independently bounded and non-negative, so the
        total cannot exceed the sum of the caps (100) or go below 0.
    """
    lowered = text.lower()
    words = re.findall(r"[a-z']+", lowered)

    # -- hook words: count (which word) + position (how early it lands) ----
    matched = [w for w in HOOK_WORDS if w in lowered]
    hook_count_component = min(len(matched) / 4.0, 1.0) * _HOOK_COUNT_CAP
    if matched:
        earliest_idx = min(lowered.find(w) for w in matched)
        position_fraction = earliest_idx / max(len(lowered), 1)
        hook_position_component = max(0.0, 1.0 - position_fraction) * _HOOK_POSITION_CAP
    else:
        hook_position_component = 0.0
    hook_component = hook_count_component + hook_position_component

    # -- speech density: bell curve peaking at a natural speaking rate -----
    # Below the old cap-at-2.5 scheme, everything from 2.5 w/s to a
    # machine-gun 6 w/s scored identically. A Gaussian peaking at 2.75 w/s
    # rewards a natural pace and marks down both hesitant and rushed
    # delivery, which is what actually makes a clip watchable.
    duration = max(float(end) - float(start), 1e-6)
    density = len(words) / duration
    density_component = _DENSITY_CAP * math.exp(-((density - _DENSITY_PEAK) / _DENSITY_SIGMA) ** 2)

    # -- question / exclamation, position-aware for the question ----------
    punct_component = 0.0
    if "?" in text:
        q_fraction = text.find("?") / max(len(text), 1)
        punct_component += _QUESTION_CAP * (1.0 - 0.6 * q_fraction)
    if "!" in text:
        punct_component += _EXCLAIM_CAP

    # -- numeric specificity -------------------------------------------
    digit_hits = len(_NUMERIC_PATTERN.findall(lowered))
    word_hits = len(set(words) & NUMBER_WORDS)
    numeric_component = min(digit_hits + word_hits, 2) * (_NUMERIC_CAP / 2.0)

    # -- first-sentence strength: score the literal `hook` field --------
    # The hook shown downstream (in .meta.json and the renderer) is just the
    # first sentence, so it needs its own signal rather than riding on the
    # whole block's density/hook-word scores. A two-word fragment like
    # "till now." should score near zero here even if the rest of the
    # segment is dense with hook words.
    first_sentence = _first_sentence(text)
    fs_lower = first_sentence.lower()
    fs_words = re.findall(r"[a-z']+", fs_lower)
    fs_hook = _FIRST_SENTENCE_HOOK_CAP if any(w in fs_lower for w in HOOK_WORDS) else 0.0
    fs_question = _FIRST_SENTENCE_QUESTION_CAP if "?" in first_sentence else 0.0
    fs_complete = _FIRST_SENTENCE_COMPLETE_CAP * min(
        len(fs_words) / _FIRST_SENTENCE_COMPLETE_WORDS, 1.0
    )
    first_sentence_component = fs_hook + fs_question + fs_complete

    total = (
        hook_component + density_component + punct_component
        + numeric_component + first_sentence_component
    )
    return {
        "hook": hook_component,
        "density": density_component,
        "punctuation": punct_component,
        "numeric": numeric_component,
        "first_sentence": first_sentence_component,
        "total": min(max(total, 0.0), 100.0),
    }


def _describe_reason(breakdown: dict) -> str:
    """Name the components that actually drove a segment's score.

    Args:
        breakdown: The dict returned by :func:`_breakdown`.

    Returns:
        A short string naming up to three signals whose contribution exceeded
        30% of their own cap, ranked strongest first; a flat fallback string
        if nothing cleared that bar.
    """
    signals = (
        ("hook words", breakdown["hook"], _HOOK_COUNT_CAP + _HOOK_POSITION_CAP),
        ("strong opening", breakdown["first_sentence"],
         _FIRST_SENTENCE_HOOK_CAP + _FIRST_SENTENCE_QUESTION_CAP + _FIRST_SENTENCE_COMPLETE_CAP),
        ("natural speaking pace", breakdown["density"], _DENSITY_CAP),
        ("question/exclamation", breakdown["punctuation"], _QUESTION_CAP + _EXCLAIM_CAP),
        ("concrete numbers", breakdown["numeric"], _NUMERIC_CAP),
    )
    ranked = sorted(signals, key=lambda s: -(s[1] / s[2]))
    top = [name for name, value, cap in ranked if value / cap > 0.3][:3]
    if not top:
        return "deterministic heuristic: no strong signal (flat across hook words, opening, pace, punctuation, numbers)"
    return "deterministic heuristic: driven by " + ", ".join(top)


def score_segment(seg: dict) -> float:
    """Score a segment's short-form potential from 0 to 100.

    Combines five signals: position-weighted hook-word density, a
    natural-pace speech-density curve, position-aware question/exclamation
    presence, numeric specificity, and the strength of the first sentence
    alone (see module docstring for detail on each).

    Args:
        seg: Segment dict with "text", "start" and "end".

    Returns:
        A score in [0, 100].
    """
    text = (seg.get("text") or "").strip()
    if not text:
        return 0.0
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0

    breakdown = _breakdown(text, seg["start"], seg["end"])
    return round(breakdown["total"], 2)


def rank(segments: list[dict], target_count: int) -> list[dict]:
    """Score and select the strongest segments.

    Args:
        segments: Candidate segment dicts.
        target_count: How many to return.

    Returns:
        Up to ``target_count`` ranked dicts, highest score first. Ties break
        on segment index so the order is stable.
    """
    scored = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        words = re.findall(r"[a-z']+", text.lower()) if text else []
        if not text or not words:
            breakdown = {"hook": 0.0, "density": 0.0, "punctuation": 0.0,
                         "numeric": 0.0, "first_sentence": 0.0, "total": 0.0}
        else:
            breakdown = _breakdown(text, seg["start"], seg["end"])
        first_sentence = _first_sentence(text)
        scored.append({
            **seg,
            "score": round(breakdown["total"], 2),
            "hook": first_sentence[:90],
            "reason": _describe_reason(breakdown),
            "ranker": "heuristic",
        })
    scored.sort(key=lambda s: (-s["score"], s["index"]))
    return scored[:target_count]
