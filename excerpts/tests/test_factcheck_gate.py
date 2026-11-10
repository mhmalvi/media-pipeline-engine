"""Tests for the fact-check gate's voting and repair-action rules."""
from factcheck_gate import actions, confirm, flipped


def _claim(verdict, scene_idx=0, passage_ids=("P1",)):
    return {"id": "C1", "scene_idx": scene_idx, "text": "claim", "verdict": verdict, "note": "",
            "passage_ids": list(passage_ids), "source_urls": []}


def _vote(verdict, cited=()):
    return {"verdict": verdict, "note": "", "passage_ids": [], "source_urls": [], "cited": list(cited)}


def test_supported_first_vote_is_never_rejudged():
    # Arrange
    calls = []
    # Act
    result = confirm([_claim("supported")], lambda todo, n: calls.append(n) or [])
    # Assert
    assert calls == [] and result[0]["verdict"] == "supported"


def test_one_failed_vote_then_two_supported_passes_and_counts_as_flip():
    # Arrange
    def revote(todo, vote_no):
        return [_vote("supported") for _ in todo]
    # Act
    [claim] = confirm([_claim("unsupported")], revote)
    # Assert
    assert claim["verdict"] == "supported"
    assert len(claim["votes"]) == 3
    assert flipped(claim)


def test_two_failed_votes_fail_without_a_third_vote():
    # Arrange
    seen = []
    def revote(todo, vote_no):
        seen.append(vote_no)
        return [_vote("contradicted") for _ in todo]
    # Act
    [claim] = confirm([_claim("unsupported")], revote)
    # Assert
    assert seen == [2]
    assert claim["verdict"] == "contradicted"


def test_failure_with_cited_passage_is_rewritten():
    # Arrange
    claim = _claim("unsupported") | {"votes": [_vote("unsupported", cited=[{"text": "passage"}])]}
    # Act
    [out] = actions([claim])
    # Assert
    assert (out["action"], out["cut_reason"]) == ("rewrite", None)


def test_failure_without_any_source_is_cut():
    # Arrange
    claim = _claim("unsupported", passage_ids=()) | {"votes": [_vote("unsupported")]}
    # Act
    [out] = actions([claim])
    # Assert
    assert (out["action"], out["cut_reason"]) == ("cut", "no source")


def test_failure_in_already_rewritten_scene_is_cut():
    # Arrange
    claim = _claim("contradicted", scene_idx=4) | {"votes": [_vote("contradicted", cited=[{"text": "p"}])]}
    # Act
    [out] = actions([claim], rewritten_scenes={4})
    # Assert
    assert out["cut_reason"] == "already rewritten once"


def test_last_allowed_repair_cuts_everything_failing():
    # Arrange
    claim = _claim("unsupported") | {"votes": [_vote("unsupported", cited=[{"text": "p"}])]}
    # Act
    [out] = actions([claim], last=True)
    # Assert
    assert out["cut_reason"] == "last rewrite"


def test_supported_claims_pass_through_unchanged():
    # Arrange
    claim = _claim("supported")
    # Act
    [out] = actions([claim], last=True)
    # Assert
    assert out is claim and "action" not in out
