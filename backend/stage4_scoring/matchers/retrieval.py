from __future__ import annotations

from backend.stage4_scoring.config import DEFAULT_STAGE4_CONFIG
from backend.stage3_retrieval.models import RetrievedCandidate

from .common import MatcherResult, clamp01, to_percent, weighted_average


def _normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return [0.5 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def score_retrieval_match(
    retrieved_candidate: RetrievedCandidate,
    pool: list[RetrievedCandidate],
    weights: dict[str, float] | None = None,
) -> MatcherResult:
    retrieval_weights = weights or DEFAULT_STAGE4_CONFIG.retrieval_subweights
    semantic_scores = [candidate.semantic_score or 0.0 for candidate in pool]
    bm25_scores = [candidate.bm25_score or 0.0 for candidate in pool]
    rrf_scores = [candidate.rrf_score for candidate in pool]

    normalized_semantic = _normalize_scores(semantic_scores)
    normalized_bm25 = _normalize_scores(bm25_scores)
    normalized_rrf = _normalize_scores(rrf_scores)

    index = pool.index(retrieved_candidate)
    semantic_component = normalized_semantic[index] if normalized_semantic else 0.0
    bm25_component = normalized_bm25[index] if normalized_bm25 else 0.0
    rrf_component = normalized_rrf[index] if normalized_rrf else 0.0

    score = weighted_average(
        {
            "semantic": semantic_component,
            "bm25": bm25_component,
            "rrf": rrf_component,
        },
        retrieval_weights,
    )
    return MatcherResult(
        score=to_percent(score),
        details={
            "semantic_component": semantic_component,
            "bm25_component": bm25_component,
            "rrf_component": rrf_component,
            "weights": dict(retrieval_weights),
        },
    )
