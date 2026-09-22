from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.preprocessing.sample import parse_job_profile_from_text
from backend.preprocessing.loader import load_candidates
from backend.schema.candidate_profile import CandidateProfile
from backend.stage3_retrieval.models import RetrievedCandidate
from backend.stage4_scoring.scorer import score_candidates
from stage1_features.candidate_builder import build_candidate_document


SAMPLE_JD = """
Job Title: Backend Engineer
Industry: IT Services
Location: Toronto, Canada
Work Mode: Onsite
Minimum Experience: 5+ years of experience

Required Skills
- Python
- SQL
- Spark
- Kafka

Preferred Skills
- Airflow
- AWS
""".strip()


def _sample_candidates(count: int = 3):
    candidates = []
    for loaded in load_candidates("data/candidates.jsonl"):
        profile = CandidateProfile.from_dict(loaded.raw)
        document = build_candidate_document(profile)
        candidates.append((profile, document))
        if len(candidates) >= count:
            break
    return candidates


def test_score_candidates_end_to_end() -> None:
    sample_candidates = _sample_candidates()
    candidate_profiles = {profile.candidate_id: profile for profile, _ in sample_candidates}
    candidate_documents = {document.candidate_id: document for _, document in sample_candidates}
    retrieved_candidates = [
        RetrievedCandidate(candidate_id=document.candidate_id, semantic_score=1.0 - index * 0.1, bm25_score=10.0 - index, rrf_score=1.0 / (index + 1))
        for index, (_, document) in enumerate(sample_candidates)
    ]
    job_profile = parse_job_profile_from_text(SAMPLE_JD, source_name="sample-jd.txt")

    scored = score_candidates(
        retrieved_candidates=retrieved_candidates,
        candidate_profiles=candidate_profiles,
        candidate_documents=candidate_documents,
        job_profile=job_profile,
    )

    assert len(scored) == len(sample_candidates)
    assert scored[0].final_score >= scored[-1].final_score
    assert scored[0].candidate_id == sample_candidates[0][0].candidate_id
    for item in scored:
        assert 0.0 <= item.final_score <= 100.0
        assert set(item.individual_scores.to_dict().keys()) == {
            "retrieval",
            "technical",
            "experience",
            "education",
            "company",
            "behavioral",
            "consistency",
        }
        assert "technical_details" in "\n".join(item.score_breakdown.notes)


if __name__ == "__main__":
    test_score_candidates_end_to_end()
    print("Stage 4 smoke test passed")
