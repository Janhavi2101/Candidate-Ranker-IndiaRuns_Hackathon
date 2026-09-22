from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.preprocessing.loader import load_candidates
from backend.schema.candidate_profile import CandidateProfile
from backend.stage2_embeddings.build_index import build_stage2_index
from backend.stage2_embeddings.embedding_builder import EmbeddingBuilder
from backend.stage2_embeddings.tests.candidate_builder import FakeEmbeddingModel
from backend.stage2_embeddings.bm25_builder import build_bm25_index_from_documents, BM25Index
from backend.stage3_retrieval.hybrid_retriever import HybridCandidateRetriever
from backend.stage3_retrieval.query_encoder import QueryEncoder
from stage1_features.candidate_builder import build_candidate_document


class FakeQueryModel:
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


def _build_documents(count: int = 2):
    documents = []
    for loaded in load_candidates("data/candidates.jsonl"):
        profile = CandidateProfile.from_dict(loaded.raw)
        documents.append(build_candidate_document(profile))
        if len(documents) >= count:
            break
    return documents


def test_hybrid_retriever_ranks_and_deduplicates() -> None:
    documents = _build_documents()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        stage0_path = temp_path / "candidate_profiles.jsonl"
        stage2_dir = temp_path / "stage2"
        bm25_path = temp_path / "bm25.json"

        with stage0_path.open("w", encoding="utf-8") as handle:
            written = 0
            for loaded in load_candidates("data/candidates.jsonl"):
                profile = CandidateProfile.from_dict(loaded.raw)
                handle.write(json.dumps(profile.to_dict(), ensure_ascii=False) + "\n")
                written += 1
                if written >= 2:
                    break

        build_stage2_index(
            input_path=stage0_path,
            output_dir=stage2_dir,
            builder=EmbeddingBuilder(model=FakeEmbeddingModel()),
        )

        bm25_index = build_bm25_index_from_documents(documents)
        bm25_index.save(bm25_path)

        retriever = HybridCandidateRetriever.load(
            vector_store_dir=stage2_dir,
            bm25_index_path=bm25_path,
            query_encoder=QueryEncoder(model=FakeQueryModel()),
        )

        results = retriever.retrieve("backend engineer spark cloud", top_k=2)

        assert len(results) == 2
        assert len({result.candidate_id for result in results}) == 2
        assert any(result.semantic_score is not None for result in results)
        assert any(result.bm25_score is not None for result in results)
        assert results[0].rrf_score >= results[1].rrf_score


if __name__ == "__main__":
    test_hybrid_retriever_ranks_and_deduplicates()
    print("Stage 3 smoke test passed")
