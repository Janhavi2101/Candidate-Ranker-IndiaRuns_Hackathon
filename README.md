# Candidate Ranker

## Stage 0

Stage 0 loads `data/candidates.jsonl`, validates each record, and writes normalized `CandidateProfile` rows.

Run it with:

```bash
python offline/stage0_pipeline.py \
  --input data/candidates.jsonl \
  --output data/stage0/candidate_profiles.jsonl \
  --report data/stage0/report.json
```

The stage writes a JSONL of validated profiles and a JSON validation report with any rejected rows.

## Stage 1

Stage 1 turns each validated `CandidateProfile` into a deterministic `CandidateDocument` with:

- `embedding_text` for later vectorization
- `metadata` for pre-retrieval filtering
- `ranking_features` for downstream scoring

Run the smoke test with:

```bash
python stage1_features/tests/candidate_builder.py
```

## Stage 2

Stage 2 converts a `CandidateDocument` into an embedding and stores it in a FAISS-ready structure.

Run the smoke test with:

```bash
python backend/stage2_embeddings/tests/candidate_builder.py
```

Build the full Stage 2 index from Stage 0 output with:

```bash
python backend/stage2_embeddings/build_index.py \
  --input data/stage0/candidate_profiles.jsonl \
  --output-dir data/stage2
```

## Stage 3

Stage 3 retrieves candidates with semantic FAISS search and lexical BM25 search, then merges both using RRF.

Run the smoke test with:

```bash
python backend/stage3_retrieval/tests/hybrid_retriever.py
```

## Stage 4

Stage 4 scores retrieved candidates using JobProfile matchers, bonuses, penalties, and configurable weights on a 0-100 scale.

Run the smoke test with:

```bash
python backend/stage4_scoring/tests/scorer.py
```
