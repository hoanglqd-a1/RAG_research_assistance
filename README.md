# RAG Research Assistance

A research prototype for document-grounded question answering and tool-using
agents, using local Ollama inference and reproducible benchmarks.

## Capabilities

- PDF and text ingestion with filename/page citations.
- Three retrieval modes: **dense** (embeddings + FAISS), **BM25** (keyword search),
  and **hybrid** (dense + BM25 with reciprocal rank fusion).
- Agent tools: document search, metadata lookup, calculator, and summarization.
- Retrieval and agent evaluation with saved answers, chunks, tool traces,
  configuration, corpus hashes, latency, and VRAM snapshots.

Select retrieval with `RETRIEVAL_MODE=dense|bm25|hybrid`; the default is `dense`.
The same mode is used by pipeline queries, agent search, and benchmark runners.

## Setup and configuration

Requires Python 3.10+ and Ollama for generation. Commands below assume Linux
and execution from the project root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
ollama pull qwen3:14b-q4_K_M
ollama serve
```

Run project commands in another terminal while Ollama is serving. Main settings
in `.env`:

```dotenv
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=qwen3:14b-q4_K_M
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
RETRIEVAL_MODE=hybrid
CHUNK_SIZE=500
CHUNK_OVERLAP=75
DEFAULT_TOP_K=5
BM25_K1=1.5
BM25_B=0.75
HYBRID_CANDIDATE_K=20
HYBRID_RRF_K=60
MAX_AGENT_STEPS=6
```

See [.env.example](.env.example) for all settings. Process environment variables
override `.env`. Embedding models download on first use; indexes are rebuilt
for each invocation.

For the existing Docker environment:

```bash
docker start RAGagent
docker exec -it RAGagent bash
cd /source/RAG_research_assistance
source .venv/bin/activate
```

Serve Ollama inside the container when using `localhost:11434`.

## Usage

```bash
python -m scripts.ask_agent data/paper.pdf "What evidence supports the main result?"
python -m scripts.ask_agent data/paper_a.pdf data/paper_b.pdf "Compare the methods."
python -m scripts.demo_agent data/paper.pdf
python -m scripts.demo_retrieval data/paper.pdf "What is the method?" --top-k 5
```

## Evaluation

Both runners index only the three SHA-256-verified PDFs in
[the benchmark manifest](benchmarks/rag_agent_benchmark.json): Shape of Motion
(`q001.pdf`), MoSca (`q002.pdf`), and SplineGS (`q003.pdf`).

- **Retrieval:** 23 questions at K=1,3,5; page-level Hit, Recall, MRR, nDCG, and
  complete-evidence coverage. No generation model is required.
- **Agent:** 31 summary, narrative, quantitative, abstention, false-premise, and
  behavior cases. Measures source/citation coverage, tool selection, behavior,
  grounding proxies, and runtime. Ollama is required; web cases are excluded.

```bash
RETRIEVAL_MODE=dense python -m scripts.evaluate_retrieval --output-dir benchmark_results/retrieval_dense_new
RETRIEVAL_MODE=bm25 python -m scripts.evaluate_retrieval --output-dir benchmark_results/retrieval_bm25_new
RETRIEVAL_MODE=hybrid python -m scripts.evaluate_retrieval --output-dir benchmark_results/retrieval_hybrid_new

RETRIEVAL_MODE=dense python -m scripts.evaluate_agent --output-dir benchmark_results/agent_dense_new
RETRIEVAL_MODE=hybrid python -m scripts.evaluate_agent --output-dir benchmark_results/agent_hybrid_new
```

Run agent comparisons sequentially with identical model, prompt, corpus, chunking,
and scoring settings. Use new output directories to preserve results.
`--case-id Q12 Q18 Q27 B01` selects agent cases; `--judge-faithfulness` adds an
optional local model judgment. Both runners expose further options through `--help`.

Results default to timestamped folders under `benchmark_results/`:

| Shared artifacts | Retrieval artifacts | Agent artifacts |
| --- | --- | --- |
| `aggregate_scores.json` | `per_question_results.json` | `per_case_results.json` |
| `run_metadata.json` | `retrieved_chunks.jsonl` | `answers.jsonl`, `tool_traces.jsonl` |

## Benchmark results

The comparisons use `all-MiniLM-L6-v2`, 500-character chunks with 75-character
overlap, BM25 `k1=1.5/b=0.75`, and hybrid RRF with 20 candidates per backend
and constant 60. Corpus and benchmark hashes match within each comparison;
retrieval and agent comparisons record different benchmark hashes.

### Retrieval

Key results at K=5; saved artifacts include K=1 and K=3.

| Mode | Hit | Recall | MRR | nDCG | Complete evidence | Mean query latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 69.6% | 64.1% | 0.501 | 0.516 | 60.9% | 14.0 ms |
| BM25 | 78.3% | 70.7% | 0.614 | 0.595 | 65.2% | 3.2 ms |
| Hybrid | 82.6% | 75.7% | 0.688 | 0.683 | 69.6% | 16.3 ms |

Saved scores: [dense](benchmark_results/comparison_dense_20261008T152846Z/aggregate_scores.json),
[BM25](benchmark_results/comparison_bm25_20261008T152846Z/aggregate_scores.json),
[hybrid](benchmark_results/comparison_hybrid_20261008T152846Z/aggregate_scores.json).
Query latency excludes indexing and generation.

### Agent

Both runs use `qwen3:14b-q4_K_M`, the same prompt, a six-step limit, and no
faithfulness judge. Evidence/citation metrics cover 27 cases; behavior checks
cover six.

| Measure | Dense | Hybrid |
| --- | ---: | ---: |
| Completed cases | 31/31 | 31/31 |
| Source-page hit | 59.3% | 70.4% |
| Source-page recall | 54.6% | 63.9% |
| Source-page MRR | 0.463 | 0.543 |
| Complete source evidence | 51.9% | 59.3% |
| Complete citation evidence | 44.4% | 51.9% |
| Expected tool plan satisfied | 45.2% | 45.2% |
| Behavior checks passed (heuristic) | 5/6 | 3/6 |
| Grounding proxy score | 0.596 | 0.676 |
| Mean case latency | 36.69 s | 36.67 s |

Saved scores: [dense agent](benchmark_results/agent_dense_comparison/aggregate_scores.json),
[hybrid agent](benchmark_results/agent_hybrid_comparison/aggregate_scores.json).
Each run took approximately 19 minutes. Post-run GPU snapshots recorded 9,977 MB
used on a 16,380 MB RTX 4060 Ti; these are not peak-VRAM measurements.

Hybrid improves evidence coverage, but responses still contain errors: Q12 finds
the labeled page yet reports 5.3 instead of the reference 9.5 percentage points;
neither mode uses the calculator in the 15 quantitative cases. Both accept the
Q27 false premise. Hybrid's Q26 abstention is correct but fails the keyword scorer.

These are single runs. Page hits and grounding proxies do not establish answer
correctness; citation checks do not verify support for each claim. There is no
separate scored evaluation of fixed-path RAG generation yet. Earlier runs remain
under `benchmark_results/`.

## Development and limitations

Code is organized under `app/rag/`, `app/agent/`, `app/llm/`, and
`app/evaluation/`; `app/services.py` constructs shared services.

```bash
python -m pytest -q
```

Tests use deterministic substitutes without Ollama calls or model downloads.
Current limitations include in-memory storage, incomplete PDF table extraction,
document-title resolution, unreliable tool selection, and heuristic answer scoring.
Web search, reranking, and query rewriting are not implemented. The small
three-paper benchmark does not establish generalization to other corpora.
