"""LLM clip ranking in small batches, with a deterministic heuristic underneath and a scale-safe merge.

Excerpt from the clip-engine module `rank`.

Candidate windows (transcript spans that start on a sentence boundary and never straddle a hard scene cut) are scored for short-form
potential. The LLM call goes through a shared, local-first routing layer (a task class that tries a local
Ollama model first), so the stage works with zero cloud budget. If a batch times out it is split in half
and each half is tried once; anything the LLM still doesn't score goes to the heuristic ranker
(`rank_heuristic.py`). LLM unavailability degrades quality; it never stops the pipeline.

Removed for the excerpt: the import of the private routing layer (the `extract` callable is injected
instead), progress printing, and comments quoting measured timings. `rank_segments` takes `target`
directly rather than reading it from the preset. Logic is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

import rank_heuristic

# The routing layer's extract task class has a fixed timeout that is not this stage's to change; one call
# per ~10 candidates sat right at it. Fewer candidates per call clears it comfortably even under load.
BATCH_SIZE = 5
# The model judges hook strength, not full transcripts: only the prompt copy of each text is capped.
PROMPT_TEXT_LIMIT = 200
REQUIRED_KEYS = ("clips",)

Extract = Callable[[str, tuple], dict]


def _truncate_for_prompt(text: str, limit: int = PROMPT_TEXT_LIMIT) -> str:
    """Shorten a window's text for the prompt only, on a word boundary, marked with an ellipsis."""
    text = text or ""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut.rstrip() + "…"


def build_prompt(segments: list[dict]) -> str:
    """Build the ranking prompt (JSON-only answer keyed by candidate index)."""
    lines = [
        "You are selecting moments from a long video to cut into short vertical clips.",
        "Score each candidate 0-100 for standalone short-form potential: does it hook fast,",
        "make sense without surrounding context, and resolve within its own duration?",
        "Some candidate text below is truncated (marked with …); judge the hook from",
        "what is shown.",
        "",
        "Return JSON only, shaped exactly:",
        '{"clips": [{"index": <int>, "score": <0-100>, "hook": "<max 12 words>",',
        ' "reason": "<one short sentence>"}]}',
        "",
        "Candidates:",
    ]
    for seg in segments:
        duration = float(seg["end"]) - float(seg["start"])
        lines.append(f"[{seg['index']}] ({duration:.0f}s) {_truncate_for_prompt(seg['text'])}")
    return "\n".join(lines)


def parse_scores(payload: dict, segments: list[dict]) -> list[dict]:
    """Map an LLM payload back onto the segments it scored; ValueError if nothing usable maps.

    Keyed by index rather than appended: the model can emit the same index twice, and only the
    highest-scoring occurrence survives. Unknown indexes and non-numeric scores are dropped; scores are
    clamped to 0-100.
    """
    by_index = {seg["index"]: seg for seg in segments}
    clips = payload.get("clips")
    rows = clips if isinstance(clips, list) else []
    best_by_index: dict[int, dict] = {}
    for row in rows:
        try:
            index = int(row["index"])
        except (KeyError, TypeError, ValueError):
            continue
        seg = by_index.get(index)
        if seg is None:
            continue
        try:
            score = float(row.get("score", 0))
        except (TypeError, ValueError):
            continue
        score = max(0.0, min(100.0, score))
        existing = best_by_index.get(index)
        if existing is not None and existing["score"] >= score:
            continue
        best_by_index[index] = {**seg, "score": score, "hook": str(row.get("hook", ""))[:90],
                                "reason": str(row.get("reason", "")), "ranker": "llm"}
    if not best_by_index:
        raise ValueError("LLM response contained no usable clip scores")
    ranked = list(best_by_index.values())
    ranked.sort(key=lambda c: (-c["score"], c["index"]))
    return ranked


def _chunk(items: list, size: int) -> list[list]:
    """Split a list into consecutive chunks of at most `size` items."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def _rank_chunk(chunk: list[dict], extract: Extract) -> tuple[list[dict], list[dict]]:
    """Score one chunk in a single call and report anything it left unscored.

    A model asked about five candidates can answer about three. A partial answer used to look like a
    success and the unanswered segments silently vanished; they are returned as unscored instead, which
    puts them on the same path as a batch that failed outright.
    """
    ranked = parse_scores(extract(build_prompt(chunk), REQUIRED_KEYS), chunk)
    scored = {clip["index"] for clip in ranked}
    return ranked, [seg for seg in chunk if seg["index"] not in scored]


def _rank_batch(batch: list[dict], extract: Extract) -> tuple[list[dict], list[dict]]:
    """Rank one batch through the LLM, retrying once on timeout with it halved (no further splitting).

    Returns (ranked, failed): LLM-scored clip dicts, and raw segments that still need a heuristic score.
    """
    try:
        return _rank_chunk(batch, extract)
    except TimeoutError:
        if len(batch) <= 1:
            return [], list(batch)
        mid = len(batch) // 2
        ranked: list[dict] = []
        failed: list[dict] = []
        for half in (batch[:mid], batch[mid:]):
            if not half:
                continue
            try:
                half_ranked, half_unscored = _rank_chunk(half, extract)
            except Exception:
                failed.extend(half)
            else:
                ranked.extend(half_ranked)
                failed.extend(half_unscored)
        return ranked, failed
    except Exception:
        return [], list(batch)


def merge_ranked(llm_ranked: list[dict], heuristic_ranked: list[dict], target: int) -> list[dict]:
    """Combine LLM-scored and heuristic-scored clips without comparing their scales.

    The two rankers do not share a scale: LLM scores run well above heuristic ones for the same
    candidates, so sorting both on raw `score` would not "degrade" a failed batch, it would ELIMINATE it.
    Slots are allocated proportionally instead: a ranker that scored half the candidates fills half the
    output, each with its own top N. Rounding favours the LLM (it keeps at least one slot whenever it
    scored anything), and any slot a ranker cannot fill goes to the other. Reported scores are left as
    each ranker produced them; the `ranker` field tells a reader which scale a score is on.
    """
    llm = sorted(llm_ranked, key=lambda c: (-c["score"], c["index"]))
    fallback = sorted(heuristic_ranked, key=lambda c: (-c["score"], c["index"]))
    total = len(llm) + len(fallback)
    if total == 0 or target <= 0:
        return []

    target = min(target, total)
    fallback_slots = int(round(target * len(fallback) / total))
    if llm:
        fallback_slots = min(fallback_slots, target - 1)  # the better ranker never loses its last slot
    llm_slots = min(target - fallback_slots, len(llm))
    fallback_slots = min(len(fallback), target - llm_slots)
    llm_slots = min(target - fallback_slots, len(llm))

    chosen_llm, chosen_fallback = llm[:llm_slots], fallback[:fallback_slots]
    merged: list[dict] = []
    for position in range(max(len(chosen_llm), len(chosen_fallback))):
        if position < len(chosen_llm):
            merged.append(chosen_llm[position])
        if position < len(chosen_fallback):
            merged.append(chosen_fallback[position])
    return merged


def rank_segments(segments: list[dict], target: int, extract: Extract) -> list[dict]:
    """Rank candidate segments, preferring the LLM and falling back per batch to the heuristic."""
    if not segments:
        return []
    llm_ranked: list[dict] = []
    fallback_segments: list[dict] = []
    for batch in _chunk(segments, BATCH_SIZE):
        try:
            ranked, failed = _rank_batch(batch, extract)
        except Exception:  # belt-and-braces: _rank_batch should not raise
            ranked, failed = [], list(batch)
        llm_ranked.extend(ranked)
        fallback_segments.extend(failed)
    heuristic_ranked = (rank_heuristic.rank(fallback_segments, len(fallback_segments))
                        if fallback_segments else [])
    return merge_ranked(llm_ranked, heuristic_ranked, target)
