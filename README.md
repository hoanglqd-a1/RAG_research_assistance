# Beginner-friendly Agentic RAG

This project implements both a normal Retrieval-Augmented Generation (RAG)
pipeline and a small, explicit tool-using agent. It avoids LangChain, LangGraph,
CrewAI, and other agent frameworks so the control flow remains visible.

## Normal RAG versus Agentic RAG

Normal RAG always follows one fixed path:

```text
question -> retrieve top-k chunks -> construct grounded prompt -> LLM answer
```

Agentic RAG lets the model choose the next operation. It can answer directly,
search documents, inspect the collection, calculate a value, summarize a whole
document, or make several sequential tool calls before answering.

```mermaid
flowchart TD
    U[User request] --> A[Agent LLM]
    A --> D{Tool needed?}
    D -- No --> F[Final answer]
    D -- Yes --> T{Selected tool}
    T --> S[search_documents]
    T --> M[get_document_metadata]
    T --> C[calculator]
    T --> Z[summarize_document]
    S --> O[Validated tool result]
    M --> O
    C --> O
    Z --> O
    O --> A
```

The implementation uses local Ollama chat/tool calling. The provider-specific
conversion is isolated in `app/llm/ollama_client.py`; the loop itself remains
ordinary Python.

<!-- ## Repository architecture

```text
app/
  services.py                constructs shared RAG and agent services
  core/config.py             .env configuration
  rag/
    document_loader.py       PDF/text extraction
    chunker.py               overlapping character chunks
    embeddings.py            sentence-transformer vectors
    vector_store.py          FAISS and document catalog access
    retriever.py             query embedding plus top-k search
    generator.py             direct RAG grounded prompt
    pipeline.py              ingestion orchestration
  agent/
    agent.py                 explicit tool loop
    state.py                 ordered conversation events
    registry.py              definitions, validation, and dispatch
    prompts.py               editable agent instructions
    schemas.py               agent events, traces, and results
    tools/
      document_search.py
      document_metadata.py
      calculator.py
      summarize_document.py
scripts/
  demo_retrieval.py
  demo_agent.py
tests/
``` -->

## Retrieval pipeline

`DocumentLoader` receives a path and emits clean `Document` objects. PDFs remain
separated by page so page citations survive. `TextChunker` emits overlapping
`TextChunk` objects containing text, filename, page, chunk ID, and extra metadata.
Overlap protects context near a chunk boundary.

`EmbeddingService` maps N strings to an array of shape `(N, D)`. A query is
mapped by the same model to shape `(D,)`, placing documents and questions in the
same semantic vector space. FAISS L2-normalizes these vectors and performs inner
product search; for normalized vectors, that score is cosine similarity.

The FAISS adapter also exposes read-only `list_chunks()` catalog access. This is
needed because metadata lookup and whole-document summarization require complete
coverage, not similarity search. A future Qdrant or pgvector adapter can implement
the same `VectorStore` protocol.

## Agent loop

`Agent.run()` performs these operations explicitly:

1. Add the user message to `AgentState`.
2. Send the ordered state and JSON tool definitions to the LLM.
3. If the model returns function calls, append each call to state.
4. Look up the tool in `ToolRegistry`.
5. Validate its JSON arguments with its Pydantic input model.
6. Execute the handler and wrap success or failure in a structured result.
7. Append the tool result using the matching call ID.
8. Send the expanded state to the LLM again.
9. Stop when the model returns text without tool calls, or when the configured
   maximum step count is reached.

The state contains user messages, assistant messages, tool requests, and tool
results. Optional conversation IDs retain that state in memory for subsequent
requests. This is request/session history, not long-term memory, and disappears
when the process restarts.

### Available tools

- `search_documents(query, top_k=5)` calls the existing `Retriever` and returns
  text, filename, page, chunk ID, metadata, and cosine score.
- `get_document_metadata(filename=None)` lists indexed filenames, pages, chunk
  counts, and stored metadata.
- `calculator(expression)` parses arithmetic with an AST whitelist. It permits
  numbers, parentheses, `+`, `-`, `*`, `/`, and bounded `**`; it never uses
  `eval` and rejects names or function calls.
- `summarize_document(filename)` enumerates all chunks for an exact filename.
  It summarizes bounded batches and then reduces the partial summaries. Both
  batch size and maximum batch count are configurable.

Tool failures and unknown tools become structured observations for the LLM, so it
can correct arguments or answer differently. Logs include steps, selected tools,
arguments, durations, success/failure, and final LLM/tool counts. They never log
secrets or private model reasoning.

Structured result sources are collected from actual tool outputs rather than being
trusted from generated text. The model is separately instructed to cite sources
as `[paper.pdf, page 5]`.

## Installation and configuration

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Make sure Ollama is running and the local model is available:

```powershell
ollama pull qwen3:14b-q4_K_M
ollama serve
```

Configure `.env` without committing it:

```env
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=qwen3:14b-q4_K_M
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=75
DEFAULT_TOP_K=5
MAX_AGENT_STEPS=6
SUMMARY_BATCH_CHARS=8000
MAX_SUMMARY_BATCHES=12
```

For Vietnamese or mixed-language documents, use a multilingual embedding model:

```env
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Changing the embedding model requires rebuilding the in-memory index.

## Run the project

Index one or more files and run six examples: direct response, retrieval,
calculation, comparison, summary, and missing information.

```powershell
python -m scripts.demo_agent sample.txt
python -m scripts.demo_agent data\paper_a.pdf data\paper_b.pdf
```

Ask one custom question after indexing local files:

```powershell
python -m scripts.ask_agent sample.txt "What is this document about?"
python -m scripts.ask_agent data\paper_a.pdf data\paper_b.pdf "Compare the methods."
```

The script prints every selected tool and its validated arguments. Model tool
selection is probabilistic, so the exact calls can vary; automated tests use a
scripted fake LLM for deterministic loop verification. The embedding model is
downloaded on first use, and the FAISS index exists only for the lifetime of the
script.

The retrieval-only demo remains available:

```powershell
python -m scripts.demo_retrieval sample.txt "What is this document about?" --top-k 3
```

## Tests

```powershell
pytest -q
```

Tests never call Ollama. They use deterministic embeddings, a recording text
generator, and scripted agent responses. Coverage includes chunking, metadata,
retrieval, safe arithmetic, invalid arguments, unknown tools, map-reduce
summarization, source collection, the two-turn tool loop, and maximum steps.

## Example control flow: compare Method A and Method B

1. The user message, existing conversation events, system instruction, and tool
   definitions are sent to the agent LLM.
2. The model requests `search_documents` with a query focused on Method A.
3. The registry validates the arguments. The existing retriever embeds the query,
   searches FAISS, and returns relevant chunks with source metadata.
4. The request and result are appended to `AgentState`, then sent to the LLM.
5. The model can request `search_documents` again with a Method B query.
6. That second call and result are also appended. Both evidence sets are now
   visible to the next model call.
7. The model returns final comparison text rather than another tool call.
8. The agent result contains that text, deduplicated sources collected from both
   searches, and the tool-call trace. If the model keeps calling tools, the loop stops at
   `MAX_AGENT_STEPS` and returns a controlled limit response.

This is what makes the system agentic: the application supplies capabilities and
safety boundaries, while the model selects the sequence dynamically.

<!-- ## Current limitations

- FAISS data and conversations are process-local and not persisted.
- Scanned PDFs, complex tables, and layout reconstruction require OCR/parsing work.
- Chunk size and summary limits use characters rather than tokenizer counts.
- Summary input is capped by `MAX_SUMMARY_BATCHES`; the result reports truncation.
- There is no authentication, web search, reranking, hybrid search, long-term
  memory, background autonomy, or multi-agent orchestration.
- Tool traces show observable actions, not private chain-of-thought. -->
