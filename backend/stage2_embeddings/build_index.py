from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.preprocessing.loader import load_candidates
from backend.schema.candidate_profile import CandidateProfile
from stage1_features.candidate_builder import build_candidate_document

from backend.stage2_embeddings.candidate_builder import build_embedded_candidate
from backend.stage2_embeddings.embedding_builder import EmbeddingBuilder, SentenceTransformerEmbeddingBuilder
from backend.stage2_embeddings.vector_store import CandidateVectorStore
from backend.stage2_embeddings.bm25_builder import (
    build_bm25_index_from_documents,
)


@dataclass(slots=True)
class Stage2IndexReport:
    input_path: str
    output_dir: str
    total_candidates: int
    indexed_candidates: int
    embedding_model_name: str
    embedding_dimension: int
    use_cosine_similarity: bool
    bm25_index_created: bool = True

    def to_dict(self) -> dict:
        return {
            "input_path": self.input_path,
            "output_dir": self.output_dir,
            "total_candidates": self.total_candidates,
            "indexed_candidates": self.indexed_candidates,
            "embedding_model_name": self.embedding_model_name,
            "embedding_dimension": self.embedding_dimension,
            "use_cosine_similarity": self.use_cosine_similarity,
            "bm25_index_created": self.bm25_index_created,
        }


def _candidate_profiles(input_path: str | Path) -> Iterable[CandidateProfile]:
    for loaded in load_candidates(input_path):
        yield CandidateProfile.from_dict(loaded.raw)


def build_stage2_index(
    input_path: str | Path,
    output_dir: str | Path,
    builder: Optional[EmbeddingBuilder] = None,
    use_cosine_similarity: bool = True,
) -> Stage2IndexReport:

    source_path = Path(input_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if builder is None:
        builder = EmbeddingBuilder(model=SentenceTransformerEmbeddingBuilder())

    total_candidates = 0
    indexed_candidates = 0

    vector_store: CandidateVectorStore | None = None
    embedding_dimension = 0
    embedding_model_name = getattr(builder.model, "model_name", "")

    # Store CandidateDocuments for BM25
    candidate_documents = []

    # --------------------------------------------------------
    # Build FAISS embeddings
    # --------------------------------------------------------

    for candidate in _candidate_profiles(source_path):

        total_candidates += 1

        candidate_document = build_candidate_document(candidate)
        candidate_documents.append(candidate_document)

        embedded_candidate = build_embedded_candidate(
            candidate_document,
            builder=builder,
        )

        if vector_store is None:

            embedding_dimension = len(embedded_candidate.embedding)

            vector_store = CandidateVectorStore(
                dimension=embedding_dimension,
                use_cosine_similarity=use_cosine_similarity,
            )

        vector_store.add([embedded_candidate])

        indexed_candidates += 1

    if vector_store is None:
        raise ValueError(f"No candidates found in {source_path}")

    # --------------------------------------------------------
    # Save FAISS Vector Store
    # --------------------------------------------------------

    vector_store.save(target_dir)

    # --------------------------------------------------------
    # Build & Save BM25 Index
    # --------------------------------------------------------

    bm25_index = build_bm25_index_from_documents(candidate_documents)

    bm25_index.save(target_dir / "bm25_index.json")

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = Stage2IndexReport(
        input_path=str(source_path),
        output_dir=str(target_dir),
        total_candidates=total_candidates,
        indexed_candidates=indexed_candidates,
        embedding_model_name=embedding_model_name,
        embedding_dimension=embedding_dimension,
        use_cosine_similarity=use_cosine_similarity,
        bm25_index_created=True,
    )

    (target_dir / "report.json").write_text(
        json.dumps(
            report.to_dict(),
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Stage 2 FAISS index from Stage 0 candidate profiles")
    parser.add_argument("--input", default="data/stage0/candidate_profiles.jsonl", help="Stage 0 candidate profile JSONL")
    parser.add_argument("--output-dir", default="data/stage2", help="Directory for the FAISS index and lookup data")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--device", default=None, help="SentenceTransformer device")
    parser.add_argument("--no-normalize", action="store_true", help="Disable cosine normalization and use L2 distance")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    builder = EmbeddingBuilder(
        model=SentenceTransformerEmbeddingBuilder(
            model_name=args.model_name,
            device=args.device,
            normalize_embeddings=not args.no_normalize,
        )
    )
    report = build_stage2_index(
        input_path=args.input,
        output_dir=args.output_dir,
        builder=builder,
        use_cosine_similarity=not args.no_normalize,
    )
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
