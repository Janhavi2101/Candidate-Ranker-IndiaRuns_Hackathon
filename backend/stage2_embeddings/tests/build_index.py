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
from backend.stage2_embeddings.vector_store import CandidateVectorStore


def _write_stage0_fixture(target: Path, count: int = 2) -> None:
    with target.open("w", encoding="utf-8") as handle:
        written = 0
        for loaded in load_candidates("data/candidates.jsonl"):
            profile = CandidateProfile.from_dict(loaded.raw)
            handle.write(json.dumps(profile.to_dict(), ensure_ascii=False) + "\n")
            written += 1
            if written >= count:
                break


def test_build_stage2_index_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        stage0_path = temp_path / "candidate_profiles.jsonl"
        output_dir = temp_path / "stage2"
        _write_stage0_fixture(stage0_path)

        report = build_stage2_index(
            input_path=stage0_path,
            output_dir=output_dir,
            builder=EmbeddingBuilder(model=FakeEmbeddingModel()),
        )

        assert report.indexed_candidates == 2
        assert (output_dir / "faiss.index").exists()
        assert (output_dir / "lookup.json").exists()

        store = CandidateVectorStore.load(output_dir)
        assert store.index.ntotal == 2
        assert store.dimension == 8
        assert len(store.candidate_ids) == 2
        assert len(store.search(np.array(store.candidates[store.candidate_ids[0]].embedding, dtype=np.float32), top_k=1)) == 1


if __name__ == "__main__":
    test_build_stage2_index_end_to_end()
    print("Stage 2 build-index smoke test passed")

