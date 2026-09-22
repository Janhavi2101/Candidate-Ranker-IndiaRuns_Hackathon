from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from backend.preprocessing.sample import parse_job_profile
from backend.schema.candidate_profile import CandidateProfile
from backend.schema.job_profile import JobProfile
from backend.stage3_retrieval.models import RetrievedCandidate
from stage1_features.models import CandidateMetadata, CandidateDocument, RankingFeatures

from .config import Stage4Config, load_stage4_config
from .matchers.behavioral import score_behavioral_match
from .matchers.company import score_company_match
from .matchers.consistency import score_consistency_match
from .matchers.education import score_education_match
from .matchers.experience import score_experience_match
from .matchers.retrieval import score_retrieval_match
from .matchers.technical import score_technical_match
from .models import IndividualScores, ScoreBreakdown, ScoredCandidate
from .rules import apply_bonus_rules, apply_penalty_rules


@dataclass(slots=True)
class CandidateContext:
    candidate_profile: CandidateProfile
    candidate_document: CandidateDocument


def _weighted_sum(component_scores: IndividualScores, weights: dict[str, float]) -> tuple[float, dict[str, float]]:
    raw = component_scores.to_dict()
    total_weight = sum(max(weights.get(name, 0.0), 0.0) for name in raw)
    if total_weight <= 0:
        weighted = {name: raw[name] for name in raw}
        return sum(weighted.values()) / max(len(weighted), 1), weighted
    weighted = {name: raw[name] * max(weights.get(name, 0.0), 0.0) / total_weight for name in raw}
    return sum(weighted.values()), weighted


def _clamp_score(value: float, max_score: float) -> float:
    return max(0.0, min(max_score, float(value)))


def _build_context(
    candidate_id: str,
    candidate_profiles: Mapping[str, CandidateProfile],
    candidate_documents: Mapping[str, CandidateDocument],
) -> CandidateContext:
    try:
        candidate_profile = candidate_profiles[candidate_id]
        candidate_document = candidate_documents[candidate_id]
    except KeyError as exc:
        raise KeyError(f"Missing candidate lookup for {candidate_id}") from exc
    return CandidateContext(candidate_profile=candidate_profile, candidate_document=candidate_document)


def score_candidates(
    retrieved_candidates: Sequence[RetrievedCandidate],
    candidate_profiles: Mapping[str, CandidateProfile],
    candidate_documents: Mapping[str, CandidateDocument],
    job_profile: JobProfile,
    config: Stage4Config | None = None,
) -> list[ScoredCandidate]:
    scoring_config = config or load_stage4_config()
    pool = list(retrieved_candidates)
    scored: list[ScoredCandidate] = []

    for retrieved_candidate in pool:
        context = _build_context(
            candidate_id=retrieved_candidate.candidate_id,
            candidate_profiles=candidate_profiles,
            candidate_documents=candidate_documents,
        )

        candidate_profile = context.candidate_profile
        candidate_document = context.candidate_document
        metadata = candidate_document.metadata
        ranking_features = candidate_document.ranking_features

        retrieval_result = score_retrieval_match(retrieved_candidate, pool, scoring_config.retrieval_subweights)
        technical_result = score_technical_match(job_profile, candidate_profile, scoring_config.technical_weights)
        experience_result = score_experience_match(job_profile, candidate_profile, scoring_config.experience_weights)
        education_result = score_education_match(job_profile, candidate_profile, scoring_config.education_weights)
        company_result = score_company_match(job_profile, candidate_profile, scoring_config.company_weights)
        behavioral_result = score_behavioral_match(candidate_profile, ranking_features, scoring_config.behavioral_weights)
        consistency_result = score_consistency_match(ranking_features)

        individual_scores = IndividualScores(
            retrieval=retrieval_result.score,
            technical=technical_result.score,
            experience=experience_result.score,
            education=education_result.score,
            company=company_result.score,
            behavioral=behavioral_result.score,
            consistency=consistency_result.score,
        )

        weighted_score, weighted_components = _weighted_sum(individual_scores, scoring_config.component_weights)
        bonus_result = apply_bonus_rules(job_profile, candidate_profile, metadata, ranking_features, scoring_config)
        penalty_result = apply_penalty_rules(job_profile, candidate_profile, metadata, ranking_features, scoring_config)

        final_score = _clamp_score(
            scoring_config.base_score + weighted_score + bonus_result.score - penalty_result.score,
            scoring_config.final_score_scale,
        )
        breakdown = ScoreBreakdown(
            raw_component_scores={
                "retrieval": retrieval_result.score,
                "technical": technical_result.score,
                "experience": experience_result.score,
                "education": education_result.score,
                "company": company_result.score,
                "behavioral": behavioral_result.score,
                "consistency": consistency_result.score,
            },
            weighted_component_scores=weighted_components,
            bonus_details=bonus_result.details,
            penalty_details=penalty_result.details,
            notes=[
                f"retrieval_details={retrieval_result.details}",
                f"technical_details={technical_result.details}",
                f"experience_details={experience_result.details}",
                f"education_details={education_result.details}",
                f"company_details={company_result.details}",
                f"behavioral_details={behavioral_result.details}",
                f"consistency_details={consistency_result.details}",
            ],
        )

        scored.append(
            ScoredCandidate(
                candidate_id=retrieved_candidate.candidate_id,
                individual_scores=individual_scores,
                bonuses=bonus_result.score,
                penalties=penalty_result.score,
                final_score=final_score,
                score_breakdown=breakdown,
            )
        )

    return sorted(scored, key=lambda item: (-item.final_score, item.candidate_id))


def score_candidates_from_job_source(
    retrieved_candidates: Sequence[RetrievedCandidate],
    candidate_profiles: Mapping[str, CandidateProfile],
    candidate_documents: Mapping[str, CandidateDocument],
    job_source: str | Path | JobProfile,
    config: Stage4Config | None = None,
) -> list[ScoredCandidate]:
    if isinstance(job_source, JobProfile):
        job_profile = job_source
    else:
        job_profile = parse_job_profile(job_source)
    return score_candidates(
        retrieved_candidates=retrieved_candidates,
        candidate_profiles=candidate_profiles,
        candidate_documents=candidate_documents,
        job_profile=job_profile,
        config=config,
    )
