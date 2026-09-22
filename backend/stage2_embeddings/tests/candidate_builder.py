from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.preprocessing.loader import load_candidates
from backend.schema.candidate_profile import CandidateProfile
from backend.stage2_embeddings.candidate_builder import build_embedded_candidate
from backend.stage2_embeddings.embedding_builder import EmbeddingBuilder
from backend.stage2_embeddings.vector_store import CandidateVectorStore
from stage1_features.candidate_builder import build_candidate_document


class FakeEmbeddingModel:
    model_name = "fake-model"

    def encode(self, texts):
        vectors = []
        for text in texts:
            seed = sum(ord(char) for char in text)
            vector = np.array(
                [((seed + index * 17) % 101) / 100.0 for index in range(8)],
                dtype=np.float32,
            )
            vectors.append(vector)
        return np.vstack(vectors)


def _first_two_documents():
    candidates = []
    for loaded in load_candidates("data/candidates.jsonl"):
        candidates.append(CandidateProfile.from_dict(loaded.raw))
        if len(candidates) == 2:
            break
    return [build_candidate_document(candidate) for candidate in candidates]


def test_build_embedded_candidate_is_deterministic() -> None:
    document = _first_two_documents()[0]
    builder = EmbeddingBuilder(model=FakeEmbeddingModel())

    first = build_embedded_candidate(document, builder=builder)
    second = build_embedded_candidate(document, builder=builder)

    assert first.to_dict() == second.to_dict()
    assert first.candidate_id == document.candidate_id
    assert first.embedding_model_name == "fake-model"
    assert len(first.embedding) == 8
    assert first.metadata.to_dict() == document.metadata.to_dict()
    assert first.ranking_features.to_dict() == document.ranking_features.to_dict()


def test_vector_store_round_trip() -> None:
    document_one, document_two = _first_two_documents()
    builder = EmbeddingBuilder(model=FakeEmbeddingModel())
    embedded_one = build_embedded_candidate(document_one, builder=builder)
    embedded_two = build_embedded_candidate(document_two, builder=builder)

    store = CandidateVectorStore(dimension=len(embedded_one.embedding))
    store.add([embedded_one, embedded_two])

    results = store.search(embedded_one.embedding, top_k=2)
    assert results[0].candidate_id == embedded_one.candidate_id

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        store.save(temp_dir)
        loaded = CandidateVectorStore.load(temp_dir)
        loaded_results = loaded.search(embedded_one.embedding, top_k=2)
        assert loaded_results[0].candidate_id == embedded_one.candidate_id
        assert loaded_results[0].embedded_candidate.to_dict() == embedded_one.to_dict()


if __name__ == "__main__":
    test_build_embedded_candidate_is_deterministic()
    test_vector_store_round_trip()
    print("Stage 2 smoke tests passed")

