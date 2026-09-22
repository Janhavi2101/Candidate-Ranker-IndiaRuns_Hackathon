from __future__ import annotations

from importlib import import_module

__all__ = [
    "BM25Index",
    "build_bm25_index_from_documents",
    "HybridCandidateRetriever",
    "RetrievedCandidate",
    "QueryEncoder",
    "SentenceTransformerQueryEncoder",
]


_LAZY_IMPORTS = {
    "BM25Index": ("backend.stage2_embeddings.bm25_builder", "BM25Index"),
    "build_bm25_index_from_documents": ("backend.stage2_embeddings.bm25_builder", "build_bm25_index_from_documents"),
    "HybridCandidateRetriever": ("backend.stage3_retrieval.hybrid_retriever", "HybridCandidateRetriever"),
    "RetrievedCandidate": ("backend.stage3_retrieval.models", "RetrievedCandidate"),
    "QueryEncoder": ("backend.stage3_retrieval.query_encoder", "QueryEncoder"),
    "SentenceTransformerQueryEncoder": ("backend.stage3_retrieval.query_encoder", "SentenceTransformerQueryEncoder"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    return getattr(module, attribute_name)
