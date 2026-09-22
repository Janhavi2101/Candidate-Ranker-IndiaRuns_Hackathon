from pathlib import Path

from backend.stage2_embeddings.build_index import build_stage2_index
from backend.stage2_embeddings.vector_store import CandidateVectorStore
from backend.stage2_embeddings.bm25_builder import BM25Index


INPUT_PATH = "data/stage0/candidate_profiles.jsonl"
OUTPUT_DIR = "data/stage2"


def main():

    print("=" * 60)
    print("Building Stage 2 Index...")
    print("=" * 60)

    report = build_stage2_index(
        input_path=INPUT_PATH,
        output_dir=OUTPUT_DIR,
    )

    print("\nStage 2 Report")
    print("-" * 60)
    print(f"Candidates Loaded      : {report.total_candidates}")
    print(f"Candidates Indexed     : {report.indexed_candidates}")
    print(f"Embedding Model        : {report.embedding_model_name}")
    print(f"Embedding Dimension    : {report.embedding_dimension}")
    print(f"Cosine Similarity      : {report.use_cosine_similarity}")

    print("\nLoading Vector Store...")
    vector_store = CandidateVectorStore.load(OUTPUT_DIR)

    print(f"✓ Vector Store Loaded")
    print(f"Dimension              : {vector_store.dimension}")
    print(f"Candidates Indexed     : {len(vector_store.candidate_ids)}")

    print("\nLoading BM25 Index...")
    bm25 = BM25Index.load(Path(OUTPUT_DIR) / "bm25_index.json")

    print("✓ BM25 Index Loaded")
    print(f"Candidates Indexed     : {len(bm25.candidate_ids)}")

    print("\nArtifacts Created")
    print("-" * 60)

    files = [
        "faiss.index",
        "lookup.json",
        "bm25_index.json",
        "report.json",
    ]

    for file in files:
        path = Path(OUTPUT_DIR) / file
        if path.exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file}")

    print("\nStage 2 PASSED")


if __name__ == "__main__":
    main()