"""Tests for the deterministic heuristic ranker."""
from rank_heuristic import rank, score_segment


def test_empty_text_scores_zero():
    # Arrange / Act / Assert
    assert score_segment({"text": "  ", "start": 0, "end": 10}) == 0.0


def test_hook_and_number_beat_flat_filler():
    # Arrange
    strong = {"text": "Here's the biggest mistake: 3 teams lose 40% of their week.", "start": 0, "end": 5}
    flat = {"text": "and so then we went over there and it was fine.", "start": 0, "end": 5}
    # Act / Assert
    assert score_segment(strong) > score_segment(flat)


def test_score_is_bounded():
    # Arrange
    seg = {"text": "Why? Why? Never! " * 20 + " 1 2 3 4", "start": 0, "end": 30}
    # Act
    score = score_segment(seg)
    # Assert
    assert 0.0 <= score <= 100.0


def test_rank_is_deterministic_and_tagged():
    # Arrange
    segs = [{"index": i, "text": t, "start": 0, "end": 5}
            for i, t in enumerate(["ok then.", "The truth is nobody tells you this.", "ok then."])]
    # Act
    first, second = rank(segs, 3), rank(segs, 3)
    # Assert
    assert first == second
    assert first[0]["index"] == 1
    assert [c["index"] for c in first[1:]] == [0, 2]  # ties break on index
    assert {c["ranker"] for c in first} == {"heuristic"}
