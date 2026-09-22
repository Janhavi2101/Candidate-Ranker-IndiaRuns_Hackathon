from __future__ import annotations

from typing import Optional, Union

from backend.schema.candidate_profile import CandidateProfile
from stage1_features.models import CandidateDocument

from .embedding_builder import EmbeddingBuilder, build_embedded_candidate as _build_embedded_candidate
from .models import EmbeddedCandidate


def build_embedded_candidate(
    candidate: Union[CandidateProfile, CandidateDocument],
    builder: Optional[EmbeddingBuilder] = None,
) -> EmbeddedCandidate:
    return _build_embedded_candidate(candidate, builder=builder)

