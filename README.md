# Candidate Ranker — Intelligent Candidate Discovery & Ranking

An intelligent **candidate discovery and ranking system** designed to identify and rank the most relevant candidates for a given job description.

The system combines **hybrid retrieval, feature engineering, rule-based scoring, and LLM-assisted reranking** to move beyond simple keyword matching and produce a more relevant candidate shortlist.

> Built for the **IndiaRuns / Redrob Intelligent Candidate Discovery & Ranking Challenge**.

---

## 🚀 Overview

Traditional resume screening systems often rely heavily on keyword matching. This can result in:

* Candidates being ranked highly because of superficial keyword overlap
* Strong candidates being missed because they use different terminology
* Difficulty distinguishing between similar but contextually different roles
* Poor handling of experience, skills, seniority, and role relevance

**Candidate Ranker** addresses these problems through a multi-stage retrieval and ranking pipeline.

```

---

## ✨ Key Features

### 🔎 Hybrid Candidate Retrieval

Combines two complementary retrieval approaches:

* **Dense semantic retrieval** — captures contextual and semantic similarity
* **BM25 retrieval** — captures important exact keyword and terminology matches

The hybrid retriever combines both signals using:

```text
Hybrid Score = 0.65 × Dense Score + 0.35 × BM25 Score
```

This helps balance semantic understanding with exact skill and terminology matching.

---

### 🧠 Multi-Stage Ranking

Rather than comparing every candidate in a lakhs of others using an expensive ranking model, the system progressively narrows the candidate pool.

```text
Large Candidate Pool
        ↓
Initial Retrieval
        ↓
Hybrid Ranking
        ↓
Top Candidates
        ↓
Feature-Based Scoring
        ↓
LLM Reranking
        ↓
Final Top Candidates
```

This provides a practical balance between **ranking quality and computational efficiency**.

---

### 📊 Feature-Based Candidate Scoring

Candidates are evaluated using multiple signals extracted from their profiles.

Examples include:

* Skills
* Years of experience
* Job titles
* Education
* Relevant domains
* Role similarity
* Seniority
* Required vs. optional skills
* Experience alignment
* Semantic relevance

The scoring stage combines these signals into a final candidate relevance score.

---

### 🤖 LLM-Assisted Reranking

The final shortlist can be further evaluated using an LLM to capture contextual relationships that simple similarity metrics may miss.

For example, the system can distinguish between:

> A candidate with the right keywords but experience in an unrelated role

and

> A candidate whose experience is highly relevant even though their resume uses different terminology.


## 🖥️ Interface

The project includes a **Streamlit interface** for interacting with the candidate ranking pipeline.

Typical workflow:

```text
1. Provide Job Description
          ↓
2. Parse JD
          ↓
3. Retrieve Candidates
          ↓
4. Score Candidates
          ↓
5. Rerank
          ↓
6. Display Ranked Candidates
```

---

## 📁 Data

The project works with candidate profile data and job descriptions.

Large datasets and generated retrieval artifacts are **not stored in the Git repository** to keep the repository lightweight.

Examples of excluded artifacts include:

```text
data/stage0/
data/stage2/
data/candidates.jsonl
```

These files can be generated locally as part of the preprocessing and indexing pipeline.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Janhavi2101/Candidate-Ranker-IndiaRuns_Hackathon.git

cd Candidate-Ranker-IndiaRuns_Hackathon
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app/streamlit.py
```

The application will open in your browser.

---

## 📌 Example

Given a job description such as:

```text
Senior NLP Engineer

Requirements:
- Python
- NLP
- Transformers
- PyTorch
- Machine Learning
- Experience building NLP systems
```

the system:

```text
Job Description
       ↓
Extract requirements
       ↓
Generate semantic representation
       ↓
Dense retrieval
       +
BM25 retrieval
       ↓
Hybrid candidate pool
       ↓
Feature-based scoring
       ↓
LLM reranking
       ↓
Ranked candidates
```

This allows the system to identify candidates based on **overall relevance**, rather than simply counting matching keywords.

---

## 🧪 Diagnostics & Testing

The repository includes diagnostic and testing utilities for evaluating retrieval and scoring behavior.

Examples:

```text
backend/stage3_retrieval/tests/
backend/test/
```

These can be used to investigate:

* Retrieval quality
* Candidate demotions
* Keyword-vs-semantic conflicts
* Scoring behavior
* Ranking consistency

---

## 🎯 Design Goals

The project was designed around four main principles:

### 1. Relevance over keyword matching

A candidate should not rank highly solely because their profile contains many matching keywords.

### 2. Multi-signal ranking

Candidate relevance should be determined using multiple independent signals.

### 3. Efficient retrieval

Expensive ranking operations should be applied only to a smaller candidate pool.

### 4. Explainable ranking

The ranking pipeline should provide interpretable signals that help understand why candidates are ranked differently.

---

## 🚧 Future Improvements

Potential improvements include:

* Fine-tuned domain-specific embedding models
* Learning-to-rank models
* Improved skill taxonomy and normalization
* Better handling of synonymous skills
* Candidate-to-JD skill gap analysis
* More robust seniority detection
* Explainable candidate ranking
* Evaluation using Precision@K, Recall@K, NDCG@K and MRR
* Automated hyperparameter optimization
* Fairness and bias evaluation
* Scalable vector database integration
* Improved LLM reranking efficiency

---

## 📈 Evaluation

The ranking pipeline can be evaluated using information-retrieval metrics such as:

```text
Precision@K
Recall@K
MRR
NDCG@K
```

These metrics can help measure whether relevant candidates consistently appear near the top of the ranking.

---

## 👩‍💻 Author

**Janhavi Gangawane**

B.E. Artificial Intelligence & Data Science

Interested in **Machine Learning, Generative AI, Data Science, and AI Engineering**.

---

## ⭐ Acknowledgements

Developed as part of the **IndiaRuns / Redrob Intelligent Candidate Discovery & Ranking Challenge**.

---

## 📜 License

This project is intended for educational, research, and hackathon purposes.
