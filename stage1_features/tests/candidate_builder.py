from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.preprocessing.loader import load_candidates
from backend.schema.candidate_profile import CandidateProfile
from stage1_features.candidate_builder import build_candidate_document


def test_build_candidate_document_is_deterministic() -> None:
    candidate = CandidateProfile.from_dict(next(load_candidates("data/candidates.jsonl")).raw)
    first = build_candidate_document(candidate)
    second = build_candidate_document(candidate)

    assert first.to_dict() == second.to_dict()
    assert first.candidate_id == candidate.candidate_id
    assert "skills:" in first.embedding_text.lower()
    assert "profile_views_received_30d" not in first.embedding_text
    assert first.metadata.skills == sorted(set(first.metadata.skills), key=first.metadata.skills.index)
    assert all(skill == skill.strip().lower() for skill in first.metadata.skills)


def test_build_candidate_document_has_structured_outputs() -> None:
    candidate = CandidateProfile.from_dict(next(load_candidates("data/candidates.jsonl")).raw)
    document = build_candidate_document(candidate)

    assert document.metadata.candidate_id == candidate.candidate_id
    assert document.metadata.location == candidate.profile.location
    assert document.ranking_features.profile_completeness_score == candidate.redrob_signals.profile_completeness_score
    assert document.ranking_features.expected_salary_range_inr_lpa == candidate.redrob_signals.expected_salary_range_inr_lpa.to_dict()


if __name__ == "__main__":
    test_build_candidate_document_is_deterministic()
    test_build_candidate_document_has_structured_outputs()
    print("Stage 1 smoke tests passed")
