from __future__ import annotations

from importlib import import_module

__all__ = [
    "build_embedded_candidate",
    "EmbeddingBuilder",
    "SentenceTransformerEmbeddingBuilder",
    "CandidateSearchResult",
    "EmbeddedCandidate",
    "CandidateVectorStore",
]


_LAZY_IMPORTS = {
    "build_embedded_candidate": ("backend.stage2_embeddings.candidate_builder", "build_embedded_candidate"),
    "EmbeddingBuilder": ("backend.stage2_embeddings.embedding_builder", "EmbeddingBuilder"),
    "SentenceTransformerEmbeddingBuilder": ("backend.stage2_embeddings.embedding_builder", "SentenceTransformerEmbeddingBuilder"),
    "CandidateSearchResult": ("backend.stage2_embeddings.models", "CandidateSearchResult"),
    "EmbeddedCandidate": ("backend.stage2_embeddings.models", "EmbeddedCandidate"),
    "CandidateVectorStore": ("backend.stage2_embeddings.vector_store", "CandidateVectorStore"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    return getattr(module, attribute_name)
