"""Tests for batched LLM ranking, timeout halving and the scale-safe merge."""
from clip_ranking import BATCH_SIZE, merge_ranked, parse_scores, rank_segments


def _segments(n):
    return [{"index": i, "start": 0.0, "end": 30.0,
             "text": "Here's the biggest mistake nobody talks about. It costs three hours a week."}
            for i in range(n)]


def test_duplicate_index_keeps_highest_score_and_clamps():
    # Arrange
    payload = {"clips": [{"index": 0, "score": 40}, {"index": 0, "score": 140}, {"index": 9, "score": 99}]}
    # Act
    ranked = parse_scores(payload, _segments(1))
    # Assert
    assert [(c["index"], c["score"]) for c in ranked] == [(0, 100.0)]


def test_llm_down_falls_back_to_heuristic_for_everything():
    # Arrange
    def extract(prompt, keys):
        raise ConnectionError("gateway down")
    # Act
    out = rank_segments(_segments(7), 3, extract)
    # Assert
    assert len(out) == 3 and {c["ranker"] for c in out} == {"heuristic"}


def test_timeout_splits_batch_in_half_once():
    # Arrange
    sizes = []
    def extract(prompt, keys):
        n = prompt.count("\n[")
        sizes.append(n)
        if n == BATCH_SIZE:
            raise TimeoutError
        idx = [int(line[1:line.index("]")]) for line in prompt.splitlines() if line.startswith("[")]
        return {"clips": [{"index": i, "score": 50 + i} for i in idx]}
    # Act
    out = rank_segments(_segments(BATCH_SIZE), BATCH_SIZE, extract)
    # Assert
    assert sizes == [BATCH_SIZE, BATCH_SIZE // 2, BATCH_SIZE - BATCH_SIZE // 2]
    assert {c["ranker"] for c in out} == {"llm"}


def test_partial_answer_sends_the_rest_to_the_heuristic():
    # Arrange: the model scores only index 0
    def extract(prompt, keys):
        return {"clips": [{"index": 0, "score": 90}]}
    # Act
    out = rank_segments(_segments(3), 3, extract)
    # Assert
    assert sorted(c["ranker"] for c in out) == ["heuristic", "heuristic", "llm"]


def test_merge_is_proportional_not_by_raw_score():
    # Arrange: LLM scores are all higher than any heuristic score
    llm = [{"index": i, "score": 90.0 - i} for i in range(2)]
    heur = [{"index": 10 + i, "score": 30.0 - i} for i in range(2)]
    # Act
    merged = merge_ranked(llm, heur, 2)
    # Assert: one slot each, LLM first, despite the scale gap
    assert [c["index"] for c in merged] == [0, 10]


def test_merge_keeps_one_llm_slot_against_rounding():
    # Arrange
    llm = [{"index": 0, "score": 80.0}]
    heur = [{"index": 10 + i, "score": 20.0} for i in range(9)]
    # Act
    merged = merge_ranked(llm, heur, 2)
    # Assert
    assert merged[0]["index"] == 0 and len(merged) == 2
